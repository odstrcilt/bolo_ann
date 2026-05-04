import numpy as np
import jax
import jax.numpy as jnp
import optax
import flax.linen as nn
from sklearn.decomposition import PCA
from functools import partial
from IPython import embed
print(f'JAX backend: {jax.default_backend()}')     
#jax.config.update("jax_disable_jit", True)
import matplotlib.pylab as plt

# ------------------------------------------------------------
# Load data
# ------------------------------------------------------------
data = np.load("tomo_training_data.npz")

tomo = data["tomo"]
tomo_norm = np.single(data["tomo_norm"])
#EFIT_data = data["EFIT_data"]
raw_data = data["raw_data"].T
EFIT_data = data["EFIT_data"][:,1:]


efit_name = ['RXPT1', 'RXPT2', 'ZXPT1', 'ZXPT2', 'Z0', 'R0',
             'TRIBOT', 'TRITOP', 'KAPPA', 'AMINOR', 'DRSEP']
 
# ------------------------
# Data loading
# ------------------------
def clip_EFIT_inputs(EFIT_values):
    
    missing_lower_Xpoint = EFIT_values[0] < 0
    missing_upper_Xpoint = EFIT_values[1] < 0
 
    
    valid = EFIT_values[0] != 0
 
    EFIT_values[-1, EFIT_values[-1] > 0.3] = 0.06 #usualy missing one X-point
    EFIT_values[-1, EFIT_values[-1] < -0.3] = -0.07#usualy missing one X-point
    EFIT_values[2, missing_lower_Xpoint] = -1.33
    EFIT_values[0, missing_lower_Xpoint] = 1.28
    EFIT_values[3, missing_upper_Xpoint] = 1.4
    EFIT_values[1, missing_upper_Xpoint] = 1.23
    
    
    return valid, EFIT_values

valid, EFIT_data = clip_EFIT_inputs(EFIT_data.T)
EFIT_data = EFIT_data.T

embed()
norm_power = np.abs(raw_data).mean(1)

tomo = tomo.reshape(len(tomo), -1)
nnz = ~np.all((tomo==0), 0)
tomo = tomo[:,nnz]

tomo = np.single(tomo) * tomo_norm[:, None]
tomo /= norm_power[:, None]
tomo =  np.maximum(tomo,0 )

 
raw_data /= norm_power[:, None]

# ------------------------------------------------------------
# dimensions
# ------------------------------------------------------------
N_INPUT_BASIS = 16
N_OUTPUT_BASIS = 32
HIDDEN = 128
IMG_DIM = tomo.shape[1]

# ------------------------------------------------------------
# PCA init for output basis
# ------------------------------------------------------------
pca = PCA(n_components=N_OUTPUT_BASIS)
pca.fit(np.maximum(tomo,0 ))

B_init = pca.components_.T.astype(np.float32)

 
#from sklearn.decomposition import NMF
 
#model = NMF(n_components= N_OUTPUT_BASIS, init="nndsvda", max_iter=500)
#W = model.fit_transform(tomo)   # (n, k)
#H = model.components_        # (k, m)

 
#B_full = np.zeros((len(nnz), 16))
#B_full[nnz] = W
#B_full = B_full.T.reshape(16, 120, 80)

# data shape: (16, 120, 80)
#fig, axes = plt.subplots(4, 4, figsize=(8, 8))

#for i, ax in enumerate(axes.flat):
    #ax.imshow(B_full[i], aspect="auto")
    #ax.axis("off")

#plt.tight_layout()
#plt.show()


# ------------------------------------------------------------
# train / test split
# ------------------------------------------------------------
N = tomo.shape[0]
split = int(0.8 * N)

raw_tr, raw_te = raw_data[:split], raw_data[split:]
efit_tr, efit_te = EFIT_data[:split], EFIT_data[split:]
tomo_tr, tomo_te = tomo[:split], tomo[split:]

# ------------------------------------------------------------
# normalization
# ------------------------------------------------------------
def norm_stats(x):
    mu = jnp.mean(x, axis=0)
    std = jnp.std(x, axis=0) + 1e-6
    return mu, std

def normalize(x, mu, std):
    return (x - mu) / std

raw_mu, raw_std = norm_stats(raw_tr)
efit_mu, efit_std = norm_stats(efit_tr)

raw_tr = normalize(raw_tr, raw_mu, raw_std)
raw_te = normalize(raw_te, raw_mu, raw_std)

efit_tr = normalize(efit_tr, efit_mu, efit_std)
efit_te = normalize(efit_te, efit_mu, efit_std)

# ------------------------------------------------------------
# Model
# ------------------------------------------------------------
class TomoNet(nn.Module):
    n_in_basis: int
    n_out_basis: int
    hidden: int
    img_dim: int

    def setup(self):

        self.A = nn.Dense(self.n_in_basis, use_bias=False)

        self.fc1 = nn.Dense(self.hidden)
        self.fc2 = nn.Dense(self.hidden)
        self.fc3 = nn.Dense(self.n_out_basis)

        self.B = self.param(
            "B",
            lambda k, s: B_init,
            (self.img_dim, self.n_out_basis),
        )

    def __call__(self, raw, efit):

        z = self.A(raw)

        x = jnp.concatenate([z, efit], axis=-1)

        x = nn.silu(self.fc1(x))
        x = nn.silu(self.fc2(x))

        y = self.fc3(x)*50 #get the right magnitude of teh output
 
        img = jax.nn.softplus(jnp.dot(y, self.B.T))

        return img 

# ------------------------------------------------------------
# loss
# ------------------------------------------------------------
alpha = 1e-4
beta = 1e-4

 
def loss_fn(params, raw, efit, target):
   
    pred = model.apply(params, raw, efit, mutable=False)

    mse = jnp.mean((pred - target) ** 2)

    A = params["params"]["A"]["kernel"]
    B = params["params"]["B"] 
    
 
    reg = alpha * jnp.sum(B**2) + beta * jnp.sum(A**2)

    return mse + reg, (mse, reg)

# ------------------------------------------------------------
# training step
# ------------------------------------------------------------
@jax.jit
def train_step(params, opt_state, raw, efit, target):

    (loss, (mse, reg)), grads = jax.value_and_grad(loss_fn, has_aux=True)(
        params, raw, efit, target
    )

    updates, opt_state = optimizer.update(grads, opt_state)

    params = optax.apply_updates(params, updates)

    return params, opt_state, loss, mse

@jax.jit
def eval_step(params, raw, efit, target):

    _, (mse, reg) = loss_fn(params, raw, efit, target)

    return mse

# ------------------------------------------------------------
# init
# ------------------------------------------------------------
key = jax.random.PRNGKey(0)

model = TomoNet(
    n_in_basis=N_INPUT_BASIS,
    n_out_basis=N_OUTPUT_BASIS,
    hidden=HIDDEN,
    img_dim=IMG_DIM,
)

params = model.init(key, raw_tr[:1], efit_tr[:1])

optimizer = optax.adam(1e-3)
opt_state = optimizer.init(params)

# ------------------------------------------------------------
# training loop
# ------------------------------------------------------------
EPOCHS = 200
BATCH = 2**10
import time as time
for epoch in range(EPOCHS):

    perm = np.random.permutation(split)

    train_mse = 0.0
    n_batches = 0
    t = time.time()
    for i in range(0, split, BATCH):

        #idx = slice(i,i+BATCH)
        idx = perm[i:i+BATCH]

        raw_b =  raw_tr[idx]
        efit_b =  efit_tr[idx]
        tomo_b =  tomo_tr[idx]

        params, opt_state, loss, mse = train_step(
            params, opt_state, raw_b, efit_b, tomo_b
        )

        train_mse += mse
        n_batches += 1

    train_mse /= n_batches

    # --- test ---
    test_mse = 0.0
    n_batches = 0

    for i in range(0, len(tomo_te), BATCH):

        raw_b = raw_te[i:i+BATCH]
        efit_b = efit_te[i:i+BATCH]
        tomo_b = tomo_te[i:i+BATCH]

        mse = eval_step(params, raw_b, efit_b, tomo_b)

        test_mse += mse
        n_batches += 1

    test_mse /= n_batches

    if epoch % 10 == 0:
        print(
            f"{epoch:04d}  "
            f"train MSE {train_mse:.4e}  "
            f"test MSE {test_mse:.4e}  "
            f"time {time.time()-t: .1f}s  "
            #f"test reg {test_reg:.4e}"

        )
         


import h5py
from flax.serialization import to_state_dict

state = to_state_dict(params)

def save_dict(h5group, d):
    for k, v in d.items():
        if isinstance(v, dict):
            grp = h5group.create_group(k)
            save_dict(grp, v)
        else:
            h5group.create_dataset(k, data=np.asarray(v))

with h5py.File("tomo_nn_model.h5", "w") as f:

    save_dict(f.create_group("params"), state)

    f.create_dataset("raw_mu", data=np.asarray(raw_mu))
    f.create_dataset("raw_std", data=np.asarray(raw_std))
    f.create_dataset("efit_mu", data=np.asarray(efit_mu))
    f.create_dataset("efit_std", data=np.asarray(efit_std))
    f.create_dataset("nnz", data=np.asarray(nnz))

    f.attrs["N_INPUT_BASIS"] = N_INPUT_BASIS
    f.attrs["N_OUTPUT_BASIS"] = N_OUTPUT_BASIS
    f.attrs["HIDDEN"] = HIDDEN
    f.attrs["IMG_DIM"] = IMG_DIM
    
embed()
