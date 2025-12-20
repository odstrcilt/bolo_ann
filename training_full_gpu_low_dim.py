import os
import time
import h5py
import jax
import jax.numpy as jnp
import numpy as np
import optax
from functools import partial
from tqdm import tqdm
from matplotlib import pylab as plt
from IPython import embed
from sklearn.decomposition import PCA

# --- JAX Configuration ---
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.9"
jax.config.update('jax_platform_name', 'gpu')

efit_name = ['RXPT1', 'RXPT2', 'ZXPT1', 'ZXPT2', 'Z0', 'R0',
             'TRIBOT', 'TRITOP', 'KAPPA', 'AMINOR', 'DRSEP']

#power_params = ['P_SOL','P_ldivL','P_ldivR','P_udivL','P_udivR',
                #'P_ldiv','P_udiv', 'P_core','P_axis','P_tot']


power_params = ['P_SOL','P_ldivi','P_ldivo','P_udivi','P_udivo',
                'P_ldiv','P_udiv', 'P_core','P_axis','P_tot' ]

# ------------------------
# Model Definition
# ------------------------
def init_mlp(sizes, key):
    params = []
    keys = jax.random.split(key, len(sizes) - 1)
    for k, (m, n) in zip(keys, zip(sizes[:-1], sizes[1:])):
        W = jax.random.normal(k, (m, n)) * (1.0 / jnp.sqrt(m))
        b = jnp.zeros((n,))
        params.append([W, b])
    return params

def mlp(params, x):
    for W, b in params[:-1]:
        x = jnp.dot(x, W) + b
        x = jax.nn.relu(x) #Rectified linear unit activation function.
    W, b = params[-1]
    return jnp.dot(x, W) + b

@jax.jit
def predict(input_parameters,  x):
    params, Wlin, W0, invInBasis = input_parameters
    
    return W0 + jnp.dot(x, Wlin) + mlp(params, x)

# ------------------------
# Losses
# ------------------------

def loss_consistency(P):
    """consistency check
    P_ldivL+P_ldivR = P_ldiv
    P_udivL+P_udivR = P_udiv
    P_SOL+P_core+P_ldiv+P_udiv = P_tot
    from definition. 
    It does not have to be valid exactly, it is enought if approximatelly. 
    """


    err = 0
    err += jnp.mean((P[:,power_params.index('P_ldiv')] - P[:,power_params.index('P_ldivi')] - P[:,power_params.index('P_ldivo')]) ** 2)
    err += jnp.mean((P[:,power_params.index('P_udiv')] - P[:,power_params.index('P_udivi')] - P[:,power_params.index('P_udivo')]) ** 2)
    err += jnp.mean((P[:,power_params.index('P_tot')]  - P[:,power_params.index('P_SOL')]   - P[:,power_params.index('P_core')] 
                          - P[:,power_params.index('P_ldiv')]- P[:,power_params.index('P_udiv')]) ** 2)

    return err
  
def loss_syn(input_parameters, X, Y, P, lam=1e-3, wd = 1e-4, wc = 1e-4):
 
    N = X.shape[0]
    Wpred = predict(input_parameters, X)
    
    _, _, _, invInBasis = input_parameters
 
    Wpred = Wpred.reshape(N, P.shape[1], invInBasis.shape[0])  # (N,n_p,n_pca)
    
    #multiply data with inv of basis and the weight predicted by NN
    Phat = jnp.einsum('aw,bwk,bta->btk', invInBasis, Y, Wpred)  # (N,n_p,n_s)
 
    err = jnp.mean((Phat - P) ** 2)
    pos_reg = jnp.mean(jnp.maximum(-Phat, 0) ** 2)
    reg = jnp.mean(jnp.sum(Wpred**2, axis=(1,2)))
    reg2 = loss_consistency(Phat)
    
    return err + lam * pos_reg  + wd * reg + wc * reg2

def loss_real(input_parameters, rX, rY, lam=1e-3, wd=1e-4, wc = 1e-4):
    #there is no radiated power co compared with for real data, use only consistency and positivity 
   
    N = rX.shape[0]

    Wpred = predict(input_parameters, rX)
    _, _, _, invInBasis = input_parameters

    
    Wpred = Wpred.reshape(N, -1, invInBasis.shape[0])  # (N,n_p,n_pca)
    
    #multiply data with inv of basis and the weight predicted by NN
    rPhat = jnp.einsum('aw,bw,bta->bt', invInBasis, rY, Wpred)  # (N,n_p)
    
    pos_reg = jnp.mean(jnp.maximum(-rPhat, 0) ** 2)
    reg = jnp.mean(jnp.sum(Wpred**2, axis=(1,2)))
    reg2 = loss_consistency(rPhat)

    return  lam * pos_reg +  wd * reg + reg2 * wc 


def total_loss(input_parameters, X, Y, P, rX, rY, wd=1e-4, lam=1.0):
    Lsyn  = loss_syn(input_parameters, X, Y, P, lam, wd)
    Lreal = loss_real(input_parameters, rX, rY, lam, wd)
    
    return Lsyn + Lreal

# ------------------------
# Training step
# ------------------------
@jax.jit
def train_step_both(input_parameters,  opt_state,
                    Xb, Yb, Pb, rXb, rYb, wd, lam):
    
    (L, grads) = jax.value_and_grad(total_loss, argnums=(0))(
        input_parameters, Xb, Yb, Pb, rXb, rYb, wd, lam
    )
    updates, opt_state = opt.update(grads, opt_state, input_parameters)
    input_parameters = optax.apply_updates(input_parameters, updates)
    return input_parameters, opt_state, L

# ------------------------
# Training loop
# ------------------------



def train_model(X, Y, P, rX, rY,
                hidden_sizes=(128,128),
                lr=1e-3, wd=1e-4, lam=1.0, N_PCA=32,
                batch_size=1024, epochs=20,
                key=jax.random.PRNGKey(0)):

    N_syn, Din = X.shape
    Dout = P.shape[1] * N_PCA
    N_real = rX.shape[0]

    # init model
    params = init_mlp((Din,) + hidden_sizes + (Dout,), key)
    Wlin = jnp.zeros((Din, Dout))
    W0   = jnp.zeros((Dout))

    V, W = np.linalg.eigh(rY.T @ rY)
    inv_InputBasis = W[:, -N_PCA:].T

    input_parameters = (params, Wlin, W0, inv_InputBasis)

    schedule = optax.piecewise_constant_schedule(
        1e-3, {1000: 0.1, 2000: 0.5}
    )
    global opt

    opt = optax.adamw(schedule, weight_decay=0.0)
    opt_state = opt.init(input_parameters)

    rng = np.random.default_rng(0)

    # -------- fixed split --------
    frac = 7 / 8
    syn_idx = rng.permutation(N_syn)
    real_idx = rng.permutation(N_real)

    ns_tr = int(frac * N_syn)
    nr_tr = int(frac * N_real)

    syn_tr, syn_te = syn_idx[:ns_tr], syn_idx[ns_tr:]
    real_tr, real_te = real_idx[:nr_tr], real_idx[nr_tr:]
    # -----------------------------

    def run_epoch(syn_idx, real_idx, train=True):
        losses = []
        steps = int(np.ceil(len(syn_idx) / batch_size))

        syn_idx = rng.permutation(syn_idx)
        real_idx = rng.permutation(real_idx)

        for i in range(steps):
            js = syn_idx[i*batch_size:(i+1)*batch_size]
            jr = real_idx[i*batch_size*nr_tr//ns_tr:
                          (i+1)*batch_size*nr_tr//ns_tr]

            Xb, Yb, Pb = map(lambda a: jnp.array(a[js]), (X, Y, P))
            rXb, rYb   = map(lambda a: jnp.array(a[jr]), (rX, rY))

            if train:
                nonlocal input_parameters, opt_state
                input_parameters, opt_state, L = train_step_both(
                    input_parameters, opt_state,
                    Xb, Yb, Pb, rXb, rYb, wd, lam
                )
            else:
                L = total_loss(
                    input_parameters,
                    Xb, Yb, Pb, rXb, rYb, wd, lam
                )

            losses.append(float(L))

        return np.mean(losses)

    losses, losses_test = [], []

    for ep in tqdm(range(epochs), leave=False):
        t0 = time.time()

        losses.append(run_epoch(syn_tr, real_tr, train=True))
        losses_test.append(run_epoch(syn_te, real_te, train=False))

        print(f"avg loss {losses[-1]:.4e} "
              f"test {losses_test[-1]:.4e} "
              f"time {time.time()-t0:.2f}s")

    return input_parameters, losses, losses_test



# ------------------------
# Predict function
# ------------------------
def predict_model(input_parameters, X, batch_size=8192):
    """
    Compute predictions W for X in batches, with preallocated output.
    """
    N, Din = X.shape
    Dout = sum(Wlin.shape) - Din  # safer than hardcoding
    n_p = Dout // 48   
    n_ch = Dout // n_p

    out = np.empty((N, n_p, n_ch), dtype=np.float32)  # preallocate host array

    steps = int(np.ceil(N / batch_size))
    for i in range(steps):
        start, end = i * batch_size, min((i + 1) * batch_size, N)
        Xb = jnp.array(X[start:end])
        Wb = predict(input_parameters, Xb).reshape(-1, n_p, n_ch)
        out[start:end] = np.array(Wb)  # copy from device to host

    return out

 

def estimate_weights_ls(Y, P, alpha=1):
    #least squares estimate of the weight factors 
    
    YTY = np.einsum("nji,nki->njk", Y, Y)

    from scipy.linalg import cho_factor, cho_solve
    import time
   
    YTP = np.einsum("nji,nmi->mnj", Y, P)

    W = np.empty_like(YTP)
 

    # Add regularization
    reg = YTY + alpha * np.eye(YTY.shape[-1], dtype=Y.dtype)  # (33829, 48, 48)


    for i in range(reg.shape[0]):  # loop over 33829
        try:
            c, lower = cho_factor(reg[i], overwrite_a=False, check_finite=False)
            W[:, i, :] = cho_solve((c, lower), YTP[:, i, :].T, check_finite=False).T
        except:
            W[:, i, :] = np.linalg.solve(reg[i],YTP[:, i, :].T).T

    return W.swapaxes(0,1)




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

def load_synth_data(file_path):
    
    power_params = ['P_SOL','P_ldivL','P_ldivR','P_udivL','P_udivR',
                'P_ldiv','P_udiv', 'P_core','P_axis','P_tot']

    with h5py.File(file_path, "r") as f:
        EFIT_values = np.array([f[k][:] for k in efit_name])
        power_values = np.array([f[k][:] / 1e6 for k in power_params])
        synthetics_brightness = f["synthetics_brightness+noise"][:] / 1e6
        channels = f["channels"][:]
        missing_channels = [ch.strip() for ch in f["missing_channels"][:]]
    
 
    valid, EFIT_values = clip_EFIT_inputs(EFIT_values)
    valid_ch = ~np.in1d(channels, missing_channels)
    #trainign data are availible even for the broken channels, but it wil mess up with real data
    
    synthetics_brightness[:, ~valid_ch] = 0
    nch = synthetics_brightness.shape[1] 
    
    
    err = loss_consistency(power_values.T)
    print('Check data consistency: ',  err)
   
    synthetics_brightness = synthetics_brightness.T.reshape(nch, -1, 1000).swapaxes(0, 1)[valid]
    power_values = power_values.reshape(len(power_params), -1, 1000).swapaxes(0, 1)[valid]
    EFIT_values = EFIT_values.T[valid]

    return (np.ascontiguousarray(EFIT_values),
            np.ascontiguousarray(synthetics_brightness),
            np.ascontiguousarray(power_values))

def load_real_data(file_path):
    with h5py.File(file_path, "r") as f:
        EFIT_values = np.array([f[k][:] for k in efit_name])
        real_brightness = f["real_brightness"][:] / 1e6
        channels = f["channels"][:]
        missing_channels = [ch.strip() for ch in f["missing_channels"][:]]
        shots = f["shots"][:]
        times = f["times"][:]
        
        

    shot_time = shots + times / 10
    
 
    valid, EFIT_values = clip_EFIT_inputs(EFIT_values)
    valid_ch = ~np.in1d(channels, missing_channels)
 
    real_brightness[:, ~valid_ch] = 0
    
   
    #NOTE!!  correct one broken channel
    real_brightness[shot_time > 196700,45] = (real_brightness[shot_time > 196700,44]+real_brightness[shot_time > 196700,46]) / 2
  
    return shot_time[valid],  EFIT_values.T[valid], real_brightness[valid]


# ------------------------
# Saving Network
# ------------------------
def save_network(filepath, input_parameters):
    params, Wlin, W0, inv_basis = input_parameters
    """Saves the network parameters to an HDF5 file."""
    print(f"Saving network to {filepath}...")
    with h5py.File(filepath, 'w') as f:
        # Save MLP parameters (weights and biases)
        for i, (W, b) in enumerate(params):
            f.create_dataset(f'mlp/layer_{i}/W', data=np.array(W))
            f.create_dataset(f'mlp/layer_{i}/b', data=np.array(b))
        # Save linear part parameters
        f.create_dataset('linear/Wlin', data=np.array(Wlin))
        f.create_dataset('linear/W0', data=np.array(W0))
        f.create_dataset('inv_basis', data=np.array(inv_basis))
        f.create_dataset('basis', data=np.linalg.pinv(inv_basis))

    print("Save complete.")

@jax.jit
def r2_score(y_true, y_pred):        
    ss_res = jnp.sum((y_true - y_pred) ** 2)
    ss_tot = jnp.sum((y_true - jnp.mean(y_true)) ** 2)
    return 1 - ss_res / ss_tot


# ------------------------
# Main
# ------------------------
if __name__ == "__main__":
    #synthetics data datset with radiated powers
    X, Y, P = load_synth_data("/mnt/c/synthetics_bolom_data_5.h5")
    
    #real data datset, without powers
    shot_time, rX, rY = load_real_data("/mnt/c/real_bolom_data_2.h5")
    
    
 

    key = jax.random.PRNGKey(0)
    input_parameters, losses, looses_test = train_model(
        X, Y, P, rX, rY,
        hidden_sizes=(  96, 128, 64, ),
        lr=1e-3,
        wd=1e-5,
        lam=0.1,              # weight of positivity constrain 
        batch_size=2**10,
        epochs=3000,
        N_PCA = 16, 
        key=key
    )
    
    
    
    # --- Save the trained network ---
    save_network('trained_network.h5', input_parameters)

    # --- Plotting the loss ---
    plt.figure(figsize=(6, 6))
    plt.plot(losses)
    plt.plot(looses_test)

    plt.yscale('log')
    plt.title('Training Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Log Loss')
    plt.grid(True, which="both", ls="--")
    plt.show()

    print("Script finished.")
    embed()

    
    
    W  = predict_model(input_parameters,  X, batch_size=8192)
    rW = predict_model(input_parameters, rX, batch_size=8192)

    Phat = np.einsum('ijk,ikl->ijl',W,Y)
   
    What = estimate_weights_ls(Y, P, alpha=0.01)
    Phat_min = np.einsum('ijk,ikl->ijl',What,Y)
    Phat_real = np.einsum('ijk,ik->ij',rW,rY)

    
    r2_score(Phat, P)
    r2_score(Phat_min, P)
    
    #weight vectors
    f,ax = plt.subplots(2, len(power_params), sharex=True, sharey=True, figsize=(15,10))
    for i, p in enumerate(power_params):
        vmax = np.percentile(np.abs(What[:,i]),99)
        ax[0, i].set_title(p)
        ax[0, i].imshow(W[:,i], vmax=vmax,vmin=-vmax, aspect='auto', interpolation='nearest',cmap='seismic')
        ax[1, i].imshow(What[:,i], vmax=vmax,vmin=-vmax, aspect='auto', interpolation='nearest',cmap='seismic')
    plt.tight_layout()

    
    #synthetic data
    f,ax = plt.subplots(2,5, sharex=True, sharey=True)
    ax = np.ravel(ax)
    f.suptitle('Prediction of sythetics radiated power')
    j = 0
    from matplotlib.colors import LogNorm
    for i, p in enumerate(power_params):
        ax[i].set_title(p)
        #ax[i].hist2d(P[:,i].ravel(), Phat[:,i].ravel(), bins=200, norm =  LogNorm(), cmap= 'Reds')    
        ax[i].plot( P[:,i,j])
        ax[i].plot( Phat[:,i,j],'--')
        ax[i].plot( Phat_min[:,i,j],':')
    
    
    
    #positivity
    f,ax = plt.subplots(2,5, sharex=True, sharey=True)
    f.suptitle('Prediction of real radiated power')
    ax = np.ravel(ax)
    for i, p in enumerate(power_params):
        ax[i].set_title(p)
        ax[i].plot(shot_time,  Phat_real[:,i])
        ax[i].axhline(0)
        
    
    #consistency
    f,ax = plt.subplots(1,3, sharex=True, sharey=True)
    ax[0].plot(Phat_real[:,power_params.index('P_ldiv')])
    ax[0].plot(Phat_real[:,power_params.index('P_ldivi')] +  Phat_real[:,power_params.index('P_ldivo')])

    ax[1].plot(Phat_real[:,power_params.index('P_udiv')])
    ax[1].plot(Phat_real[:,power_params.index('P_udivi')] +  Phat_real[:,power_params.index('P_udivo')])

    ax[2].plot(Phat_real[:,power_params.index('P_tot')])
    ax[2].plot(Phat_real[:,power_params.index('P_SOL')] + Phat_real[:,power_params.index('P_core')] 
             + Phat_real[:,power_params.index('P_ldiv')]+ Phat_real[:,power_params.index('P_udiv')])

    
    plt.figure()
    P_ratio = Phat_real[:,power_params.index('P_axis')]  / Phat_real[:,power_params.index('P_core')]
    P_ratio[(P_ratio > 2)|(P_ratio < -0.1)] = np.nan
 
    plt.plot(shot_time,  P_ratio,'--' )
    plt.plot(shot_time,  Phat_real[:,power_params.index('P_core')],'-' )
    plt.plot(shot_time, Phat_real[:,power_params.index('P_axis')],'-' )
    plt.show()

    #lr1e-3, 2000 epoch - 0.008 was minimum 
    #Epoch 10000/10000 - avg loss 6.1415e-03 
    #epoch 10000 avg loss 9.6602e-03 test: 9.4833e-03  (64, 128, 64) NPCA=32
    #epoch 3000  loss 1.5709e-02 test: 1.6244e-02 -  (64, 64, 128) NPCA=32
    #epoch 3000vg loss 1.2395e-02 test: 1.3430e-02 - tim -  (128, 64, 64, ) NPCA=32
    #avg loss 1.2440e-02 test: 1.3041e-02 - time 3.01s  (  64, 128, 64, ) NPCA=32
    #avg loss 1.2655e-02 test: 1.1822e-02  full real dataset, (  64, 128, 64, )  NPCA=24
    #avg loss 1.2368e-02 test: 1.2436e-02 - full real dataset,(  96, 96, 64, )  NPCA=24
    #(  96, 128, 64, ) N_PCA = 16
    
    #f,ax = plt.subplots(2,5, sharex=True, sharey=True)
    #ax = np.ravel(ax)
    #j = 0
    #for i, p in enumerate(power_params):
        #r = r2_score(P[:,i,:].flatten(), Phat[:,i,:].flatten())
        #ax[i].set_title(p + ' %.3f'%r  )
        #ax[i].plot( P[:,i,:].flatten(), Phat[:,i,:].flatten(),',')
        #ax[i].plot([0, 50], [0, 50], c='k')
        ##ax[i].plot( P[:,i,j], Phat_min[:,i,j],',')
    #plt.show()
        
     
 
    embed()
    


#TODO  do dimensionality reduction, Calculate basis, then multiple data with pinv of basis, use outputs to predict rediated power 
#TODO how is defined PSOL? Does it match the definition in load_ANN??
#reduce NPCA
