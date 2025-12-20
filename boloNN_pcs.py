import numpy as np
import h5py
from matplotlib.pylab import plt
from scipy.signal import lfilter
import MDSplus
# ------------------------
# Model Definition
# ------------------------
def silu(x):
    """NumPy implementation of the SiLU (Swish) activation function."""
    return x / (1 + np.exp(-x))

def mlp(params, x):
    """implementation of the MLP forward pass."""
    for W, b in params[:-1]:
        x = np.dot(x, W) + b
        x = silu(x)
    W, b = params[-1]
    return np.dot(x, W) + b

def predict(params, Wlin, W0, x):
    """Full prediction function."""
    return W0 + np.dot(x, Wlin) + mlp(params, x)

# ------------------------
# Network Loading
# ------------------------
def load_network(filepath):
    """Loads the network parameters from an HDF5 file into NumPy arrays."""
    params = []
    with h5py.File(filepath, 'r') as f:
        # Load MLP parameters
        i = 0
        while f'mlp/layer_{i}/W' in f:
            W = f[f'mlp/layer_{i}/W'][:]
            b = f[f'mlp/layer_{i}/b'][:]
            params.append([W, b])
            i += 1

        # Load linear part parameters
        Wlin = f['linear/Wlin'][:]
        W0 = f['linear/W0'][:]

    print(f"Network loaded successfully from {filepath}")
    return params, Wlin, W0

# ------------------------
# Main Application Logic
# ------------------------
def apply_model(params, Wlin, W0, X, Y, batch_size=4096):
    """
    Applies the loaded network to input data X and Y to get predictions P_hat.
    Computes W_hat = network(X) and then P_hat = W_hat @ Y.
    """

    num_ch = Y.shape[1] # Inferred from problem, should be passed or stored if variable
    Dout = W0.size
    num_p = Dout // num_ch

   

    # Predict the weights W_hat for the batch
    W_hat_flat = predict(params, Wlin, W0, X)
    W_hat = W_hat_flat.reshape(-1, num_p, num_ch)

    # Apply the weights to Y to get the final prediction P_hat
    P_hat = np.einsum("nd,nkd->nk", Y, W_hat)


    return P_hat

shot = 201733
MDSconn = MDSplus.Connection('atlas.gat.com')

# ------------------------
# Grab and process RTBolometry data
# ------------------------
etendue = { 'U':  [3.0206e4,2.9034e4,2.8066e4,2.7273e4,2.6635e4,4.0340e4,\
        3.9855e4,3.9488e4,3.9235e4,3.9091e4,3.9055e4,3.9126e4,\
        0.7972e4,0.8170e4,0.8498e4,0.7549e4,0.7129e4,0.6854e4,\
        1.1162e4,1.1070e4,1.1081e4,1.1196e4,1.1419e4,1.1761e4],

            'L': [2.9321e4,2.8825e4,2.8449e4,2.8187e4,2.8033e4,0.7058e4,\
        0.7140e4,0.7334e4,0.7657e4,0.8136e4,
        
        
        0.8819e4,
        
        0.7112e4,\
        0.6654e4,0.6330e4,0.6123e4,2.9621e4,2.9485e4,2.9431e4,\
        
        
        2.9458e4,2.9565e4,
        
        
        2.9756e4,3.0032e4,3.0397e4,0.6406e4]}
#channels not availible in realtme 
missing_channels =  ['U01', 'L11', 'L19', 'L20']
def lowpass_filter(x, alpha):
    b = [alpha]
    a = [1, -(1 - alpha)]
    return lfilter(b, a, x)

bolo_brightness = []
bolo_signals_list = []
#load realtime bolometer power 
for fan in ['U', 'L']:
    for ich in range(24):
        ch = f'{fan}{ich+1:02}'
   
        if ch in missing_channels:
            continue
        
        #reatime hadata by itself has ~50ms delay

        data = MDSconn.get(f'_x=PTDATA2("DGSDPWR{ch}", {shot})').data()
        bolo_signals_list.append(f'DGSDPWR{ch}')
        #this adds a small delay, but negligible compared to the existing delay in the data
        data = lowpass_filter(data, alpha=0.01)

        if len(data) <= 1:
            raise Exception(f'No data for channel {ch}')
     
        data *= etendue[fan][ich] * 1e4 #W/m^2
        bolo_brightness.append(data)
bolo_brightness = np.array(bolo_brightness)
Y = bolo_brightness.T     


# ------------------------
# Grab and process RTEFIT data
# ------------------------
efit_type = 'EFITRT1'
efit_signals = ['RXPT1', 'RXPT2', 'ZXPT1', 'ZXPT2', 'Z0', 'R0', 'TRIBOT', 'TRITOP', 'KAPPA', 'AMINOR', 'DRSEP']
MDSconn.openTree(efit_type, shot)
AEQDSK_data = {}
for ename in efit_signals + ['ATIME']:
    AEQDSK_data[ename] = MDSconn.get(f'\\{efit_type}::TOP.RESULTS.AEQDSK:{ename}').data()

tvec = MDSconn.get('dim_of(_x)').data()  #get bolometry times in ms
nearest = AEQDSK_data['ATIME'][:-1].searchsorted(tvec)
inputs = np.array([AEQDSK_data[ename][nearest] for ename in efit_signals]) # puts EFITS into bolometry timebase

# --- Data Cleaning & Reshaping (as in original script) ---
inputs[-1, (inputs[-1] > 0.3) | (inputs[-1] < -0.3)] = 0
inputs[2, inputs[0] > 1.8] = -1.15
inputs[0, inputs[0] > 1.8] = 1.25
inputs[2, inputs[0] < 0] = -1.15
inputs[0, inputs[0] < 0] = 1.25
inputs[3, inputs[1] > 1.8] = 1.2
inputs[1, inputs[1] > 1.8] = 1.2
inputs[3, inputs[1] < 0] = 1.2
inputs[1, inputs[1] < 0] = 1.2

X = inputs.T

mlp_params, Wlin, W0 = load_network('trained_network.h5')
P_predicted = apply_model(mlp_params, Wlin, W0, X, Y) # shape (t_steps, len(power_signals))
print(P_predicted.shape)
print(P_predicted[100000,:])
