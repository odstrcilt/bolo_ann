import os
import time
import h5py
import jax
import jax.numpy as jnp
import numpy as np
import optax
from functools import partial
from tqdm import trange, tqdm
from matplotlib.pylab import plt
from IPython import embed
from sklearn.model_selection import train_test_split
# --- JAX Configuration ---
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.9"
jax.config.update('jax_platform_name', 'gpu')
efit_name = ['RXPT1', 'RXPT2', 'ZXPT1' ,'ZXPT2' , 'Z0','R0' ,'TRIBOT', 'TRITOP', 'KAPPA' ,'AMINOR', 'DRSEP']
power_params = ['P_FarSOL','P_ldivL','P_ldivR','P_udivL','P_udivR','P_edge','P_core','P_axis','P_core_all','P_tot']
power_params = ['P_SOL','P_ldivi','P_ldivo','P_udivi','P_udivo','P_ldiv','P_udiv', 'P_core','P_axis','P_tot']


# ------------------------
# Exp Data loading
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

    
def load_real_data(file_path):
    with h5py.File(file_path, "r") as f:
        EFIT_values = np.array([f[k][:] for k in efit_name])
        real_brightness = f["real_brightness"][:] / 1e6
        channels = f["channels"][:]
        missing_channels = [ch.strip() for ch in f["missing_channels"][:]]
        shots = f["shots"][:]
        times = f["times"][:]
    shot_time = shots + times / 10
    EFIT_values = clip_EFIT_inputs(EFIT_values)
    valid_ch = ~np.in1d(channels, missing_channels)
    brightness_valid = real_brightness[:, valid_ch]
    X = EFIT_values.T
    print("Real Data loading complete.")
    return shot_time,  EFIT_values.T, brightness_valid

def load_data(file_path):
    """Loads and preprocesses data from the HDF5 file."""
    print(f"Loading data from {file_path}...")

    with h5py.File(file_path, "r") as f:
        EFIT_values = np.array([f[k][:] for k in efit_name])
        power_values = np.array([f[k][:] / 1e6 for k in power_params]) # MW
        ##synthetics_brightness = f["synthetics_brightness"][:] / 1e6 #MW/m^2
        synthetics_brightness = f["synthetics_brightness+noise"][:] / 1e6 # MW/m^2
        channels = f["channels"][:]
        missing_channels = [ch.strip() for ch in f["missing_channels"][:]]

    # --- Data Cleaning & Reshaping (as in original script) ---
     
    #when it does not work, assume 0
    EFIT_values[-1, EFIT_values[-1] > 0.3] = 0 #DRSEP
    EFIT_values[-1, EFIT_values[-1] < -0.3] = 0 #DRSEP
    
    #Wierd X-point locations
    EFIT_values[2, EFIT_values[0] > 1.8] = -1.15 #ZXPT1
    EFIT_values[0, EFIT_values[0] > 1.8] = 1.25 #RXPT1
    EFIT_values[2, EFIT_values[0] < 0] = -1.15 #ZXPT1
    EFIT_values[0, EFIT_values[0] < 0] = 1.25 #RXPT1
 
    #Wierd X-point locations
    EFIT_values[3, EFIT_values[1] > 1.8] = 1.2 #ZXPT2
    EFIT_values[1, EFIT_values[1] > 1.8] = 1.2 #RXPT2
    EFIT_values[3, EFIT_values[1] < 0] = 1.2 #ZXPT2
    EFIT_values[1, EFIT_values[1] < 0] = 1.2 #RXPT2
    

    valid_ch = ~np.in1d(channels, missing_channels)
    synthetics_brightness_valid = synthetics_brightness[:, valid_ch]
    nch = valid_ch.sum()

    Y = synthetics_brightness_valid.T.reshape(nch, -1, 1000).swapaxes(0, 1)
    P = power_values.reshape(len(power_params), -1, 1000).swapaxes(0, 1)
    X = EFIT_values.T

    # Ensure contiguous arrays for better performance
    X = np.ascontiguousarray(X, dtype=np.float32)
    Y = np.ascontiguousarray(Y, dtype=np.float32)
    P = np.ascontiguousarray(P, dtype=np.float32)
    print("Synthetic data loading complete.")
    return X, Y, P

# ------------------------
# Saving Network
# ------------------------
def save_network(filepath, params, Wlin, W0):
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
    print("Save complete.")
    
# ------------------------
# Model Definition
# ------------------------
def normalize_data(x, mean=None, std=None, eps=1e-8):
    """
    If mean/std are provided, use them (e.g., for test set).
    """
    if mean is None:
        mean = jnp.mean(x, axis=0, keepdims=True)
    if std is None:
        std = jnp.std(x, axis=0, keepdims=True)
    
    x_norm = (x - mean) / (std + eps)
    return x_norm, mean, std
    
def init_mlp(sizes, key):
    params = []
    keys = jax.random.split(key, len(sizes)-1)
    for k,(m,n) in zip(keys, zip(sizes[:-1], sizes[1:])):
        W = jax.random.normal(k, (m,n)) * (1.0/jnp.sqrt(m))
        b = jnp.zeros((n,))
        params.append([W,b])
    return params

def mlp(params, x):
    for W,b in params[:-1]:
        x = jnp.dot(x, W) + b
        # add batch norm (per feature dim)
        # mean = jnp.mean(x, axis=0, keepdims=True)
        # var = jnp.var(x, axis=0, keepdims=True)
        # x = (x - mean) / jnp.sqrt(var + 1e-8)
        ## 
        x = jax.nn.silu(x)
    W,b = params[-1]
    return jnp.dot(x, W) + b

@jax.jit
def predict(params, Wlin, W0, x, Dout):
    return W0 + jnp.dot(x, Wlin) + mlp(params, x)

# ------------------------
# Loss: ||W Y - P||^2 + wd*||W||^2
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
    err = err + jnp.mean((P[:,power_params.index('P_ldiv')] - P[:,power_params.index('P_ldivL')] - P[:,power_params.index('P_ldivR')]) ** 2)
    err = err + jnp.mean((P[:,power_params.index('P_udiv')] - P[:,power_params.index('P_udivL')] - P[:,power_params.index('P_udivR')]) ** 2)
    err = err + jnp.mean((P[:,power_params.index('P_tot')] - P[:,power_params.index('P_SOL')] - P[:,power_params.index('P_core')] 
                          - P[:,power_params.index('P_ldiv')]- P[:,power_params.index('P_udiv')]) ** 2)
    return err

def loss_syn(params, Wlin, W0, X, Y, P, Dout, wd, wc=1e-4):
    N = X.shape[0]

    # predict W for each X
    Wpred = predict(params, Wlin, W0, X, Dout).reshape(N, P.shape[1], Y.shape[1])  # (N,9,44)

    # Batched matmul: (N,9,44) @ (N,44,1000) -> (N,9,1000)
    Phat = jnp.matmul(Wpred, Y)   # jax handles leading batch dim

    ## Mean squared error
    err = jnp.mean((Phat - P) ** 2)

    # L2 reg on W
    reg = jnp.mean(jnp.sum(Wpred**2, axis=(1,2)))
    cst = loss_consistency(Phat)
    return err + wd * reg + wc * cst

def loss_real(params, Wlin, W0, rX, rY, Dout, lam, wc=1e-4):
    N = rX.shape[0]
    Wpred = predict(params, Wlin, W0, rX, Dout).reshape(N, -1, rY.shape[1])
    rPhat = jnp.matmul(Wpred, rY[...,None]).squeeze(-1)
    err = jnp.mean(jnp.maximum(-rPhat, 0) **2 )
    cst = loss_consistency(rPhat)
    return lam * err + wc * cst 

def total_loss(params, Wlin, W0, X, Y, P, rX, rY, Dout, wd, lam, wc=1e-4):
    return loss_syn(params, Wlin, W0, X, Y, P, Dout, wd, wc) + loss_real(params, Wlin, W0, rX, rY, Dout, lam, wc)
    
# ------------------------
# Training step (jit-compiled)
# ------------------------
@jax.jit
def train_step2(params, Wlin, W0, opt_state, 
                Xb, Yb, Pb, rXb, rYb, 
                Dout, wd, lam):
    (L, grads) = jax.value_and_grad(total_loss, argnums=(0,1,2))(
        params, Wlin, W0, Xb, Yb, Pb, rXb, rYb, Dout, wd, lam
    )
    updates, opt_state = opt.update(grads, opt_state, (params,Wlin, W0))
    params, Wlin, W0 = optax.apply_updates((params,Wlin, W0), updates)
    return params, Wlin, W0, opt_state, L
    
# Test step    
# ------------------------
def test_step2(params, Wlin, W0, Xtest, Ytest, Ptest, rXtest, rYtest, lam, wc=1e-4):
    # Wpred = predict(params, Wlin, W0, Xtest, Dout).reshape(Xtest.shape[0], Ptest.shape[1], Ytest.shape[1]) 
    Wpred = predict_model(params, Wlin, W0, Xtest, batch_size=8192) # <-
    Phat = jnp.matmul(Wpred, Ytest)

    Wpred_real = predict_model(params, Wlin, W0, rXtest, batch_size=8192)
    rPhat = jnp.matmul(Wpred_real, rYtest[...,None]).squeeze(-1)
    err = jnp.mean((Phat - Ptest) ** 2) #+ lam*jnp.mean(jnp.maximum(-rPhat, 0) **2) + wc*(loss_consistency(rPhat)+loss_consistency(Phat))
    return err
# ------------------------
# Training loop
# ------------------------
def train_model(X, Y, P, rX, rY,
                hidden_sizes=(128,128),
                lr=1e-3, wd=1e-4, lam=0.1,
                batch_size=1024, epochs=20,
                key=jax.random.PRNGKey(0)):
    
    # ---split training and test data
    Xtrain, Xtest, Ytrain, Ytest, Ptrain, Ptest, rXtrain, rXtest, rYtrain, rYtest = train_test_split(X, Y, P, rX, rY, test_size=0.2, random_state=42)
    print(Xtrain.shape, Xtest.shape, Ytrain.shape, Ytest.shape, Ptrain.shape, Ptest.shape, rXtrain.shape, rXtest.shape, rYtrain.shape, rYtest.shape)
    # normalize data
    Xtrain, Xmean, Xstd = normalize_data(Xtrain)
    Xtest, _, _ = normalize_data(Xtest, mean=Xmean, std=Xstd)
    rXtrain, rXmean, rXstd = normalize_data(rXtrain)
    rXtest, _, _ = normalize_data(rXtest, mean=rXmean, std=rXstd)
    
    N_syn, Din = Xtrain.shape
    Dout = Ptrain.shape[1] * Ytrain.shape[1]   # general output dimension
    N_real = rXtrain.shape[0]
    print(N_syn, N_real)
    key_mlp, key_lin = jax.random.split(key)
    params = init_mlp((Din,) + hidden_sizes + (Dout,), key_mlp)  # correct
    Wlin   = jnp.zeros((Din,Dout))
    W0   = jnp.zeros(( Dout))

    global opt
    opt = optax.adamw(lr, weight_decay=0.0)
    opt_state = opt.init((params, Wlin, W0))

    steps_per_epoch = int(np.ceil(N_syn / batch_size))
    rng = np.random.default_rng(0)
    losses = []
    test_losses = []
    
    for ep in range(epochs):
        idx_syn = rng.permutation(N_syn)
        idx_real = rng.permutation(N_real)
        t0 = time.time()
        epoch_losses = []
        

        for i in tqdm(range(steps_per_epoch), desc=f"Epoch {ep+1}/{epochs}", leave=False):
            j_syn = idx_syn[i*batch_size:(i+1)*batch_size]
            # j_real = idx_real[i*batch_size:(i+1)*batch_size % N_real]
            j_real = idx_real[i*batch_size:(i+1)*batch_size]

            Xb, Yb, Pb = jnp.array(Xtrain[j_syn]), jnp.array(Ytrain[j_syn]), jnp.array(Ptrain[j_syn])
            rXb, rYb = jnp.array(rXtrain[j_real]), jnp.array(rYtrain[j_real])

            params, Wlin, W0, opt_state, L = train_step2(params, Wlin, W0, opt_state, Xb, Yb, Pb, rXb, rYb, Dout, wd, lam)
            epoch_losses.append(float(L))

        avg_loss = np.mean(epoch_losses)
        losses.append(avg_loss)
        
        test_err = test_step2(params, Wlin, W0, Xtest, Ytest, Ptest, rXtest, rYtest, lam)
        test_losses.append(float(test_err))
        print(f"Epoch {ep+1}/{epochs} - avg train loss {avg_loss:.4e} - test loss {test_err:.4e} - time {time.time()-t0:.2f}s")
        
        # t1 = time.time()
        # print(len(params), Wlin.shape, W0.shape, len(opt_state), Dout) # 4 (11, 440) (440,) 3 440
        # print(f"Epoch {ep+1}/{epochs} - avg test loss {test_err:.4e} - time {time.time()-t1:.2f}s")

    return params, Wlin, W0, losses, test_losses

# ------------------------
# Predict function
# ------------------------
def predict_model(params, Wlin, W0, X, batch_size=8192):
    """
    Compute predictions W for X in batches, with preallocated output.
    """
    N, Din = X.shape
    Dout = sum(Wlin.shape) - Din  # safer than hardcoding
    n_p = Dout // 44   
    n_ch = Dout // n_p

    out = np.empty((N, n_p, n_ch), dtype=np.float32)  # preallocate host array
    steps = int(np.ceil(N / batch_size))

    for i in range(steps):
        start, end = i * batch_size, min((i + 1) * batch_size, N)
        Xb = jnp.array(X[start:end])
        Wb = predict(params, Wlin, W0, Xb, Dout).reshape(-1, n_p, n_ch)
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
    reg = YTY + alpha * np.eye(YTY.shape[-1], dtype=Y.dtype)  # (33829, 44, 44)


    for i in range(reg.shape[0]):  # loop over 33829
        try:
            c, lower = cho_factor(reg[i], overwrite_a=False, check_finite=False)
            W[:, i, :] = cho_solve((c, lower), YTP[:, i, :].T, check_finite=False).T
        except:
            W[:, i, :] = np.linalg.solve(reg[i],YTP[:, i, :].T).T

    return W.swapaxes(0,1)

    
    
@jax.jit
def r2_score(y_true, y_pred):        
    ss_res = jnp.sum((y_true - y_pred) ** 2)
    ss_tot = jnp.sum((y_true - jnp.mean(y_true)) ** 2)
    return 1 - ss_res / ss_tot


# ------------------------
# Main Execution
# ------------------------
if __name__ == "__main__":
    
    f_path = './' # import file path
    X, Y, P = load_data(f_path + 'synthetics_bolom_data_5.h5')

    #real data datset, without powers
    shot_time, rX, rY = load_real_data(f_path + 'real_bolom_data_2.h5')
    # --- Train the final model with optimized hyperparameters ---
    key = jax.random.PRNGKey(0)
    k = 100
    params, Wlin, W0, losses, test_losses = train_model(X, Y, P, rX[::k][:X.shape[0]], rY[::k][:X.shape[0]],
                                           hidden_sizes=(512, 512, 128), # Deeper network
                                           lr=1e-4,
                                           wd=1e-4,
                                           lam=0.1,  # weight of real loss positivity
                                           batch_size=2**8, # Larger batch for GPU efficiency
                                           epochs=500,
                                           key = key) #1000

    # --- Save the trained network ---
    save_network('trained_network.h5', params, Wlin, W0)

    # --- Plotting the loss ---
    plt.figure(figsize=(10, 6))
    plt.plot(losses, label = 'train')
    plt.plot(test_losses, label = 'test')
    plt.legend()
    plt.yscale('log')
    plt.title('Training/test Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Log Loss')
    plt.grid(True, which="both", ls="--")
    # plt.show()
    # embed()
    
    
    W = predict_model(params, Wlin, W0, X, batch_size=8192)
    rW = predict_model(params, Wlin, W0, rX, batch_size=8192)
    
    Phat = np.einsum('ijk,ikl->ijl',W,Y)
   
    What = estimate_weights_ls(Y, P, alpha=.01)

    Phat_min = np.einsum('ijk,ikl->ijl',What,Y)
    Phat_real = np.einsum('ijk,ik->ij', rW, rY)

    # print(r2_score(Phat_real, P))
    
    f,ax = plt.subplots(2, len(power_params), sharex=True, sharey=True, figsize=(15,10))
    for i, p in enumerate(power_params):
        vmax = np.percentile(np.abs(What[:,i]),99)
        ax[0, i].set_title(p)
        ax[0, i].imshow(W[:,i], vmax=vmax,vmin=-vmax, aspect='auto', interpolation='nearest',cmap='seismic')
        ax[1, i].imshow(What[:,i], vmax=vmax,vmin=-vmax, aspect='auto', interpolation='nearest',cmap='seismic')
    plt.tight_layout()
    plt.show()
    
    f,ax = plt.subplots(2,5, sharex=True, sharey=True)
    ax = np.ravel(ax)
    f.suptitle('Prediction of synthetic radiated power')
    j = 0
    for i, p in enumerate(power_params):
        ax[i].set_title(p)
        ax[i].plot( P[:,i,j])
        ax[i].plot( Phat[:,i,j],'--')
        ax[i].plot( Phat_min[:,i,j],':')
    plt.tight_layout()
    plt.show()

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
    ax[0].plot(Phat_real[:,power_params.index('P_ldivL')] +  Phat_real[:,power_params.index('P_ldivR')])

    ax[1].plot(Phat_real[:,power_params.index('P_udiv')])
    ax[1].plot(Phat_real[:,power_params.index('P_udivL')] +  Phat_real[:,power_params.index('P_udivR')])

    ax[2].plot(Phat_real[:,power_params.index('P_tot')])
    ax[2].plot(Phat_real[:,power_params.index('P_SOL')] + Phat_real[:,power_params.index('P_core')] 
             + Phat_real[:,power_params.index('P_ldiv')]+ Phat_real[:,power_params.index('P_udiv')])

