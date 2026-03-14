import h5py
import jax
import jax.numpy as jnp
import flax.linen as nn
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from flax.serialization import from_state_dict
from IPython import embed
import numpy as np


shot = 175860
print(shot)
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


#print('NBUG!!!!!!!!!!!!!!')
efit_name = [ 'RXPT1', 'RXPT2', 'ZXPT1' ,'ZXPT2' , 'Z0','R0' ,'TRIBOT', 'TRITOP', 'KAPPA' ,'AMINOR', 'DRSEP']

def load_mds_data(shot, EFIT = 'EFIT01', realtime_bolo=False):

    if realtime_bolo:
        print('Load realtime BOLO data')
        EFIT = 'EFITRT1'
    else:
        print('Load standart BOLO data')
    
    
    import MDSplus
    mdsserver = 'localhost'
    MDSconn = MDSplus.Connection(mdsserver)

   
    MDSconn.openTree(EFIT, shot)
    AEQDSK_data = {}
    for ename in efit_name + ['ATIME']:
        AEQDSK_data[ename] = MDSconn.get(f'\\{EFIT}::TOP.RESULTS.AEQDSK:{ename}').data()
    
        
    etendue = { 'U':  [3.0206e4,2.9034e4,2.8066e4,2.7273e4,2.6635e4,4.0340e4,\
            3.9855e4,3.9488e4,3.9235e4,3.9091e4,3.9055e4,3.9126e4,\
            0.7972e4,0.8170e4,0.8498e4,0.7549e4,0.7129e4,0.6854e4,\
            1.1162e4,1.1070e4,1.1081e4,1.1196e4,1.1419e4,1.1761e4],
                'L': [2.9321e4,2.8825e4,2.8449e4,2.8187e4,2.8033e4,0.7058e4,\
            0.7140e4,0.7334e4,0.7657e4,0.8136e4,0.8819e4,0.7112e4,\
            0.6654e4,0.6330e4,0.6123e4,2.9621e4,2.9485e4,2.9431e4,\
            2.9458e4,2.9565e4,2.9756e4,3.0032e4,3.0397e4,0.6406e4]}
    
    #channels not availible in realtme 
    missing_channels =  ['U01', 'L11', 'L19', 'L20']
    if not realtime_bolo:
        TDIcall = "_x=\\BOLOM::TOP.PRAD_01.POWER:"
        MDSconn.openTree('BOLOM', shot)

    from scipy.signal import lfilter
    def lowpass_filter(x, tau, dt):
        alpha = dt / (dt + tau)
        b = [alpha]
        a = [1, -(1 - alpha)]
        return lfilter(b, a, x, axis=-1)
        
    tau_smooth = 0.04
    bolo_brightness = []
    tvec = None
    #load realtime bolometer power 
    for fan in ['U', 'L']:
        for ich in range(24):
            ch = f'{fan}{ich+1:02}'
            
            #reatime hadata by itself has ~50ms delay
            if realtime_bolo:
                if ch in missing_channels:
                    data = MDSconn.get(f'_x=PTDATA2("DGSDPWRL01", {shot})').data()*0 
                else:
                    data = MDSconn.get(f'_x=PTDATA2("DGSDPWR{ch}", {shot})').data()
                #this adds a small delay, but negligible compared to the existing delay in the data
                #if tvec is None:
                    #tvec = MDSconn.get('dim_of(_x)').data()  #ms
      
                
            else:
                data = MDSconn.get(TDIcall+f'BOL_{ch}_P').data() #W
                
                
            if len(data) <= 1:
                raise Exception(f'No data for channel {ch}')
         
            data *= etendue[fan][ich] * 1e4 #W/m^2
            bolo_brightness.append(data) 
            
             
    tvec = MDSconn.get('dim_of(_x)').data()  #ms
    bolo_brightness = np.array(bolo_brightness)
    dt = np.diff(tvec[tvec > 0]).mean() / 1e3
 
    bolo_brightness = lowpass_filter(bolo_brightness, tau_smooth, dt)
                
                
    
    #NOTE Important correct one broken channel
    if shot > 196700:
        bolo_brightness[45] = (bolo_brightness[44]+bolo_brightness[46]) / 2
 
    #reinterpolate on realtime bolo timegrid 
    nearest = AEQDSK_data['ATIME'][:-1].searchsorted(tvec)
    inputs = np.array([AEQDSK_data[ename][nearest] for ename in efit_name])
    
    tmin, tmax = AEQDSK_data['ATIME'][[0,-1]]
    tind = slice(*tvec.searchsorted([tmin,tmax]))
    
    valid, inputs = clip_EFIT_inputs(inputs)
 
    
    return tvec[tind], inputs.T[tind], bolo_brightness.T[tind]
        
 
 
 

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
    nnz = np.array(f["nnz"])

    N_INPUT_BASIS = f.attrs["N_INPUT_BASIS"]
    N_OUTPUT_BASIS = f.attrs["N_OUTPUT_BASIS"] 
    HIDDEN = f.attrs["HIDDEN"]
    IMG_DIM = f.attrs["IMG_DIM"] 
 

 # ------------------------------------------------------------

tvec, EFIT_data, BOLO_data = load_mds_data(shot, EFIT = 'EFIT01')
 
  
EFIT_data = clip_EFIT_inputs(EFIT_data.T)[1].T

#embed()
norm_power = np.abs(BOLO_data).mean(1)
BOLO_data_norm = BOLO_data / norm_power[:, None]


# -----------------------
# Normalize X
# -----------------------
def normalize(x, mu, std):
    return (x - mu) / std


BOLO_data_norm = normalize(BOLO_data_norm, raw_mu, raw_std)
EFIT_data = normalize(EFIT_data, efit_mu, efit_std)

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

params = model.init(key, BOLO_data_norm[:1], EFIT_data[:1])
 
params = from_state_dict(model.init(key, BOLO_data_norm[:1], EFIT_data[:1]), params_dict)
# -----------------------
# Predict
# -----------------------
y_pred = model.apply(params, BOLO_data_norm, EFIT_data)
y_pred *= norm_power[:,None]  # denormalize
 

reconstructed = np.zeros((len(y_pred), len(nnz)))
reconstructed[:,nnz] = y_pred
reconstructed[:,~nnz] = np.nan 
reconstructed = reconstructed.reshape(-1, 120, 80)

# -----------------------
# Matplotlib animation with slider
# -----------------------
fig, axes = plt.subplots(1,1, figsize=(10,5))
plt.subplots_adjust(bottom=0.2)
time_idx = 0
axes = np.atleast_1d(axes)
#im_orig = axes[0].imshow(original[time_idx], cmap="viridis", vmin=0, vmax=1)
#axes[0].set_title("Original")
im_recon = axes[0].imshow(reconstructed[time_idx], cmap="viridis",
                          vmin=0, vmax=1, origin='lower')
axes[0].set_title("Reconstructed")

# Slider
ax_slider = plt.axes([0.2, 0.05, 0.6, 0.03])
slider = Slider(ax_slider, 'Time', 0, len(reconstructed)-1, valinit=0, valstep=1)

 
def update(val):
    t = int(slider.val)
    #im_orig.set_data(original[t])
    im_recon.set_data(reconstructed[t])
    
    # dynamically set vmin and vmax based on current original frame
    vmax = max(np.nanmean(reconstructed[t]),1) * 3
    #im_orig.set_clim(0, vmax)
    im_recon.set_clim(0, vmax)
    
    fig.canvas.draw_idle()
    
    
slider.on_changed(update)

# Keyboard controls
def on_key(event):
    """Handle left/right arrow keys for time navigation."""
    current = int(slider.val)
    if event.key == "right":
        new_val = min(current + 1, len(reconstructed) - 1)
        slider.set_val(new_val)
    elif event.key == "left":
        new_val = max(current - 1, 0)
        slider.set_val(new_val)
       
update(0)
fig.canvas.mpl_connect("key_press_event", on_key)
plt.show()
 
