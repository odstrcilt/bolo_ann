#from flax import linen as nn
#from flax.core import freeze, unfreeze
#import os
#import time
#import h5py
#import jax
#import jax.numpy as jnp
#import numpy as np
#import optax
#from functools import partial
#from tqdm import tqdm
#from matplotlib import pylab as plt
#from IPython import embed
#from sklearn.decomposition import PCA

#shot = 202112

#data = np.load('tomo_training_data.npz')

#tomo = data['tomo']
#tomo_norm = data['tomo_norm']
#EFIT_data = data['EFIT_data']
#raw_data= data['raw_data'].T
##shots=shots, time=time, EFIT_data=EFIT_data,efit_name=efit_name)

#norm_power = raw_data.mean(1)
#tomo = np.single(tomo) * tomo_norm[:,None,None]
##tomo_mean = np.mean(tomo,axis=(1,2))
##tomo /= norm_power[:,None,None]
##tomo = tomo.reshape(-1, 120*80)
#raw_data /= norm_power[:,None]

#N_INPUT_BASIS = 16
#N_OUTPUT_BASIS = 64

##from sklearn.decomposition import PCA
##pca = PCA(n_components=N_INPUT_BASIS)
##pca.fit_transform(raw_data)
##pseudoinversion is done trivially by transposition
###data_pca = pca.components_.T
##this should be equivalent to pca.transform(profiles_low)



##pca = PCA(n_components=N_OUTPUT_BASIS)
##pca.fit_transform(tomo )
###pseudoinversion is done trivially by transposition
##emiss_pca = pca.components_.T


##raw_data_mean = raw_data.mean(0)
##in_coeff = np.dot(raw_data-raw_data_mean, data_pca)


#raw_data

##inp

##emiss_mean = tomo.mean(0)
##Y = np.dot(tomo-emiss_mean, emiss_pca)


#X = np.hstack((EFIT_data, raw_data))



import h5py
import jax
import jax.numpy as jnp
import flax.linen as nn
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from flax.serialization import from_state_dict
from IPython import embed
import numpy as np

# -----------------------
# Load model and stats
# -----------------------
#N_INPUT_BASIS = 16
#N_OUTPUT_BASIS = 32
#HIDDEN = 128

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



with h5py.File("tomo_nn_model.h5", "r") as f:
    # Load params
    def h5_to_dict(group):
        d = {}
        for k, v in group.items():
            if isinstance(v, h5py.Group):
                d[k] = h5_to_dict(v)
            else:
                d[k] = jnp.array(v)
        return d
    params_dict = h5_to_dict(f["params"])
    # Load normalization
    raw_mu = np.array(f["raw_mu"])
    raw_std = np.array(f["raw_std"])
    efit_mu = np.array(f["efit_mu"])
    efit_std = np.array(f["efit_std"])
    
    N_INPUT_BASIS = f.attrs["N_INPUT_BASIS"]
    N_OUTPUT_BASIS = f.attrs["N_OUTPUT_BASIS"] 
    HIDDEN = f.attrs["HIDDEN"]
    IMG_DIM = f.attrs["IMG_DIM"] 
 

 # ------------------------------------------------------------
data = np.load("tomo_training_data.npz")

tomo = data["tomo"]
tomo_norm = np.single(data["tomo_norm"])
EFIT_data = data["EFIT_data"]
raw_data = data["raw_data"].T

tomo = np.single(tomo) * tomo_norm[:, None,None]
#tomo /= norm_power[:, None]
 
 
 
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


norm_power = np.abs(raw_data).mean(1)
raw_data /= norm_power[:, None]
 
nnz = ~np.all((tomo==0), 0)
 
def normalize(x, mu, std):
    return (x - mu) / std

# -----------------------
# Normalize X
# -----------------------


raw = normalize(raw_data, raw_mu, raw_std)
efit = normalize(EFIT_data, efit_mu, efit_std)

B_init = np.zeros((nnz.sum(), N_OUTPUT_BASIS), dtype='single')

# -----------------------
# Rebuild model and load params
# -----------------------

key = jax.random.PRNGKey(0)

model = TomoNet(
    n_in_basis=N_INPUT_BASIS,
    n_out_basis=N_OUTPUT_BASIS,
    hidden=HIDDEN,
    img_dim=nnz.sum(),
)



key = jax.random.PRNGKey(1)

params = model.init(key, raw[:1], efit[:1])
 
params = from_state_dict(model.init(key, raw[:1], efit[:1]), params_dict)
# -----------------------
# Predict
# -----------------------
y_pred = model.apply(params, raw, efit)
y_pred *= norm_power[:,None]  # denormalize
 

reconstructed = np.zeros_like(tomo)
reconstructed[:,nnz] = y_pred

# -----------------------
# Multiply with PCA basis
# -----------------------
# Assume pca_basis is (l, 120*80)
# Replace this with your actual PCA basis
#pca_basis = jax.random.normal(key, (y_pred.shape[1], 120*80))
#embed()
#reconstructed = np.array(y_pred @ pca_basis.T)

# -----------------------
# Reshape to (N, 120, 80)
# -----------------------
#reconstructed = reconstructed.reshape(X.shape[0], 120, 80)
#reconstructed *= norm_power[:,None,None]
# For demo, let's make "original" images same as reconstructed
original = tomo
 

original[:,~nnz] = np.nan 
reconstructed[:,~nnz] = np.nan 

# -----------------------
# Matplotlib animation with slider
# -----------------------
fig, axes = plt.subplots(1,2, figsize=(10,5))
plt.subplots_adjust(bottom=0.2)
time_idx = 0
im_orig = axes[0].imshow(original[time_idx], cmap="viridis", vmin=0, vmax=1, origin='lower')
axes[0].set_title("Original")
im_recon = axes[1].imshow(reconstructed[time_idx], cmap="viridis", vmin=0, vmax=1, origin='lower')
axes[1].set_title("Reconstructed")

# Slider
ax_slider = plt.axes([0.2, 0.05, 0.6, 0.03])
slider = Slider(ax_slider, 'Time', 0, len(tomo)-1, valinit=0, valstep=1)

 
def update(val):
    t = int(slider.val)
    im_orig.set_data(original[t])
    im_recon.set_data(reconstructed[t])
    
    # dynamically set vmin and vmax based on current original frame
    vmax = max(np.nanmean(original[t]),1) * 3
    im_orig.set_clim(0, vmax)
    im_recon.set_clim(0, vmax)
    
    fig.canvas.draw_idle()
    
    
slider.on_changed(update)

# Keyboard controls
def on_key(event):
    """Handle left/right arrow keys for time navigation."""
    current = int(slider.val)
    if event.key == "right":
        new_val = min(current + 1, len(original) - 1)
        slider.set_val(new_val)
    elif event.key == "left":
        new_val = max(current - 1, 0)
        slider.set_val(new_val)
       
update(0)
fig.canvas.mpl_connect("key_press_event", on_key)
plt.show()
 
