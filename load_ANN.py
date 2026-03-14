import h5py
import numpy as np
from IPython import embed
from matplotlib.pylab import plt


efit_name = ['RXPT1', 'RXPT2', 'ZXPT1' ,'ZXPT2' , 'Z0','R0' ,'TRIBOT', 'TRITOP', 'KAPPA' ,'AMINOR', 'DRSEP']
#power_params = ['P_SOL','P_ldivL','P_ldivR','P_udivL','P_udivR','P_ldiv','P_udiv', 'P_core','P_axis','P_tot']
power_params = ['P_SOL','P_ldivi','P_ldivo','P_udivi','P_udivo','P_ldiv','P_udiv', 'P_core','P_axis','P_tot']



# ------------------------
# Model Definition
# ------------------------
def silu(x):
    """NumPy implementation of the SiLU (Swish) activation function."""
    return x / (1 + np.exp(-x))

def relu(x):
    return np.maximum(0, x)

def mlp(params, x):
    """implementation of the MLP forward pass."""
    for W, b in params[:-1]:
        x = np.dot(x, W) + b
        #x = silu(x)
        x =  relu(x) #Rectified linear unit activation function.

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
        low_rank_basis = f['basis'][:]
      
    print(f"Network loaded successfully from {filepath}")
    
    
    missing_channels =  ['U01', 'L11', 'L19', 'L20']
    #missing_channels +=  ['U10', 'L13', 'L16', 'L12']            
    bolo_channels = [fan+'%.2d'%ich for fan in 'UL' for ich in range(1,25)]
                    
    
    invalid = np.array([ch in missing_channels for ch in bolo_channels])
    
    
    pinv_low_rank_basis = np.linalg.pinv(low_rank_basis[~invalid])
    pinv_low_rank_basis_full = np.zeros_like(low_rank_basis).T
    pinv_low_rank_basis_full[:,~invalid] = pinv_low_rank_basis
    
    
    return params, Wlin, W0, pinv_low_rank_basis_full

# ------------------------
# Main Application Logic
# ------------------------
def apply_model(nn_params, X, Y ):
    """
    Applies the loaded network to input data X and Y to get predictions P_hat.
    Computes W_hat = network(X) and then P_hat = W_hat @ Y.
    """
    
    params, Wlin, W0, pinv_low_rank_basis = nn_params

    Dout = W0.size

   

    # Predict the weights W_hat for the batch
    W_hat_flat = predict(params, Wlin, W0, X)

    # Apply the weights to Y to get the final prediction P_hat
  
    num_b = pinv_low_rank_basis.shape[0]
    num_p = Dout // num_b

    W_hat = W_hat_flat.reshape(-1, num_p, num_b)
 
    P_hat = np.einsum('aw,bw,bta->bt', pinv_low_rank_basis, Y, W_hat)  # (Nt,n_p)


    return P_hat

 

def get_region_mask(t, rvec, zvec, ATIME, PsinEmiss,BdMat,ZXPT1,ZXPT2,RXPT1,RXPT2,R0,Z0):
    
    R,Z = np.meshgrid(rvec, zvec)

    a_nearest = np.argmin(np.abs(ATIME[:-1]-t))

    
    div_low_side = (R - RXPT1[a_nearest]) * (Z0[a_nearest] - ZXPT1[a_nearest]) - (Z - ZXPT1[a_nearest]) * (R0[a_nearest] - RXPT1[a_nearest])
    div_up_side = (R - RXPT2[a_nearest]) * (Z0[a_nearest] - ZXPT2[a_nearest]) - (Z - ZXPT2[a_nearest]) * (R0[a_nearest] - RXPT2[a_nearest])
   
    mask = PsinEmiss  * 0 # FarSOL
    mask[PsinEmiss<1.2] = 5 # sol
    #mask[PsinEmiss<1.0] = 8 # ped #BUG    
    mask[PsinEmiss<0.9] = 6 # core #BUG
    mask[PsinEmiss<0.2] = 7 # axis
    mask[(Z < ZXPT1[a_nearest] + 0.2) & (div_low_side > 0) & (ZXPT1[a_nearest] > -2)&(PsinEmiss < 1.2) ] = 1 # ldivo
    mask[(Z < ZXPT1[a_nearest] + 0.2) & (div_low_side < 0) & (ZXPT1[a_nearest] > -2)&(PsinEmiss < 1.2) ] = 2 # ldivi
    mask[(Z > ZXPT2[a_nearest] - 0.2) & (div_up_side < 0) & (ZXPT2[a_nearest] > -2)&(PsinEmiss < 1.2) ] = 3 # udivo
    mask[(Z > ZXPT2[a_nearest] - 0.2) & (div_up_side > 0) & (ZXPT2[a_nearest] > -2)&(PsinEmiss < 1.2)] = 4 # udivi
    mask[ BdMat ] = -1
    
 
        
            
    return mask


def get_powers(power, mask):
     
    emiss_regions = {}

    emiss_regions['P_ldivo'] = np.sum(power[mask == 1],0)
    emiss_regions['P_ldivi'] = np.sum(power[mask == 2],0)
    emiss_regions['P_ldiv'] = np.sum(power[(mask == 1)|(mask==2) ],0)
    
    emiss_regions['P_udivo'] = np.sum(power[mask == 3],0)
    emiss_regions['P_udivi'] =  np.sum(power[mask == 4],0)
    emiss_regions['P_udiv'] = np.sum(power[(mask == 3)|(mask==4)],0)

    #emiss_regions['P_SOL'] = np.sum(power[(mask == 0)| (mask==5)],0)
    emiss_regions['P_SOL_far'] = np.sum(power[(mask == 0)],0)
    emiss_regions['P_SOL'] = np.sum(power[(mask==5)],0)
    #emiss_regions['P_ped'] = np.sum(power[(mask==8)],0)

    emiss_regions['P_axis'] = np.sum(power[mask == 7],0)
    emiss_regions['P_core'] = np.sum(power[(mask == 7)|(mask==6)],0)
    emiss_regions['P_tot'] = np.sum(power[mask>=0],0)
    
    return emiss_regions






def load_efit(shot, efit='EFIT01', load_psi=True):
    
    
    import MDSplus
    mdsserver = 'localhost'
    MDSconn = MDSplus.Connection(mdsserver)

    tree = efit
    MDSconn.openTree(tree, shot)

    
    
    AEQDSK = {}
    for ename in efit_name:
        AEQDSK[ename] = MDSconn.get(f'\\{tree}::TOP.RESULTS.AEQDSK:{ename}').data() 
    
    ATIME = MDSconn.get(f'\\{tree}::TOP.RESULTS.AEQDSK:ATIME').data() /1e3


    # --- Data Cleaning & Reshaping (as in original script) ---
    AEQDSK['DRSEP'][AEQDSK['DRSEP'] > 0.3] = 0.06
    AEQDSK['DRSEP'][AEQDSK['DRSEP'] < -0.3] = -0.07


    missing_lower_Xpoint = AEQDSK['RXPT1'] < 0
    missing_upper_Xpoint = AEQDSK['RXPT2'] < 0
    AEQDSK['ZXPT1'][missing_lower_Xpoint] = -1.33
    AEQDSK['RXPT1'][missing_lower_Xpoint] = 1.28
    AEQDSK['ZXPT2'][missing_upper_Xpoint] = 1.4
    AEQDSK['RXPT2'][missing_upper_Xpoint] = 1.23
    
    #TODO invalid EFIT?? 
    valid = AEQDSK['RXPT1'] != 0
    ATIME = ATIME[valid]
    
    for k in AEQDSK.keys():
        AEQDSK[k] = AEQDSK[k][valid]
 

    if load_psi:
        
        SSIMAG = MDSconn.get(f'\\{tree}::TOP.RESULTS.GEQDSK:SSIMAG').data()
        valid = SSIMAG != 0
        SSIMAG = SSIMAG[valid]
        

        PSIRZ = MDSconn.get(f'\\{tree}::TOP.RESULTS.GEQDSK:PSIRZ').data()[valid]
        GTIME = MDSconn.get(f'\\{tree}::TOP.RESULTS.GEQDSK:GTIME').data()[valid]/1e3
        SSIBRY = MDSconn.get(f'\\{tree}::TOP.RESULTS.GEQDSK:SSIBRY').data()[valid]

        Rgrid = MDSconn.get(f'\\{tree}::TOP.RESULTS.GEQDSK:R').data()
        Zgrid = MDSconn.get(f'\\{tree}::TOP.RESULTS.GEQDSK:Z').data()
        
        
        PSIN = (PSIRZ - SSIMAG[:,None,None])/(SSIBRY-SSIMAG)[:,None,None]
 
        return ATIME, AEQDSK, Rgrid, Zgrid, PSIN, GTIME
    
 
    
    return ATIME, AEQDSK 

    

def load_GAPROFILES(shot):

    import os, re
    folder = './GAPROFILES/'
    files = sorted(f for f in os.listdir(folder) if re.match(r'[ie]\d+\.\d+', f))

    ie_labels = ['i', 'e']
    times_per_shot =  {"i": [], "e": []}
    raw_data = {}
    R_vals = None
    Z_vals = None

    for fname in files:
        kind = fname[0]                # i or e
        shot_file = int(fname[1:7])         # e160528.03000 -> 160528
        t_ms = int(fname[8:])          # 03000
        if shot != shot_file:
            continue

        path = os.path.join(folder, fname)
        arr = np.loadtxt(path)

        R = arr[:, 0]
        Z = arr[:, 1]
        P = arr[:, 2]

        if R_vals is None:
            R_vals = np.unique(R)
        if Z_vals is None:
            Z_vals = np.unique(Z)
 
        times_per_shot[kind].append(t_ms)
        raw_data[(shot, kind, t_ms)] = P.reshape(len(Z_vals), len(R_vals))
        
        
    if len(raw_data) == 0:
        return {} 
 
    nz, nr = len(Z_vals), len(R_vals)
    
    ATIME, AEQDSK_data = load_efit(shot, efit='EFIT01', load_psi=False)
 

 
    times = sorted(times_per_shot['i'])
    data_shot = np.zeros(( 2,  len(times), nz, nr), dtype='single')
    for ie_i, ie in enumerate(ie_labels):
        for t_i, t_ms in enumerate(times):
            data_shot[ie_i, t_i] = raw_data[(shot, ie, t_ms)]
 
    dr = R_vals[1]-R_vals[0]
    dz = Z_vals[1]-Z_vals[0]
    R,Z = np.meshgrid(R_vals, Z_vals)

    #PsinEmiss = np.zeros_like(gres)
     
    emiss_regions = {}
    from scipy.interpolate import RectBivariateSpline
    for it, t in enumerate(np.array(times) / 1000):

        PsinEmiss =  data_shot[1, it]
        emiss =  data_shot[0, it]


        mask = get_region_mask(t, R_vals, Z_vals, ATIME, PsinEmiss, np.bool_(PsinEmiss * 0), 
                                AEQDSK_data['ZXPT1'],AEQDSK_data['ZXPT2'],
                                AEQDSK_data['RXPT1'],AEQDSK_data['RXPT2'],
                                AEQDSK_data['R0'], AEQDSK_data['Z0'])
        

        
        power = np.single(emiss  * R  * 2 * np.pi * dr * dz)  * 1e6 #W per cell
        
        for k, v in get_powers(power, mask).items():
            emiss_regions.setdefault(k,[])
            emiss_regions[k].append(v)
         
 
    
    emiss_regions = {p: np.array(v) for p,v in emiss_regions.items()}
    emiss_regions['tvec'] =  np.array(times) / 1000
    
 
    return emiss_regions


def load_tomography(shot):

 
        
    import glob

    try:
        file = glob.glob(f"./Database/Emissivity_*_{shot}.npz")[0]
    except IndexError:
        return {}

        
    emiss = dict(np.load( file, allow_pickle=True))

    import MDSplus
    mdsserver = 'localhost'
    MDSconn = MDSplus.Connection(mdsserver)

    tree = 'EFITRT1'
    MDSconn.openTree(tree, shot)
    SSIMAG = MDSconn.get(f'\\{tree}::TOP.RESULTS.GEQDSK:SSIMAG').data()
    valid = SSIMAG != 0
    SSIMAG = SSIMAG[valid]

    #PSIN = MDSconn.get(f'\\{tree}::TOP.RESULTS.GEQDSK:PSIN').data()
    PSIRZ = MDSconn.get(f'\\{tree}::TOP.RESULTS.GEQDSK:PSIRZ').data()[valid]
    GTIME = MDSconn.get(f'\\{tree}::TOP.RESULTS.GEQDSK:GTIME').data()[valid]/1e3
    SSIBRY = MDSconn.get(f'\\{tree}::TOP.RESULTS.GEQDSK:SSIBRY').data()[valid]

    Rgrid = MDSconn.get(f'\\{tree}::TOP.RESULTS.GEQDSK:R').data()
    Zgrid = MDSconn.get(f'\\{tree}::TOP.RESULTS.GEQDSK:Z').data()
    ATIME = MDSconn.get(f'\\{tree}::TOP.RESULTS.AEQDSK:ATIME').data()[valid]/1e3


    AEQDSK_data = {}
    for ename in efit_name:
        AEQDSK_data[ename] = MDSconn.get(f'\\{tree}::TOP.RESULTS.AEQDSK:{ename}').data()[valid]
    


    PSIN = (PSIRZ - SSIMAG[:,None,None])/(SSIBRY-SSIMAG)[:,None,None]
    
        
    #convert form float16 and renormalise
    gres = np.single(emiss['gres']) * emiss['gres_norm']


    rvec = emiss['rvec']
    zvec = emiss['zvec']
    tvec = emiss['tvec']
    dr = rvec[1]-rvec[0]
    dz = zvec[1]-zvec[0]
    R,Z = np.meshgrid(rvec, zvec)

    PsinEmiss = np.zeros_like(gres)
     
    emiss_regions = {}

    from scipy.interpolate import RectBivariateSpline
    for it, t in enumerate(emiss['tvec']):
        g_nearest = np.argmin(np.abs(GTIME-t))
        PsinEmiss = RectBivariateSpline(Zgrid,Rgrid,PSIN[g_nearest])(zvec, rvec)


        mask = get_region_mask(t, rvec, zvec, ATIME, PsinEmiss , emiss['BdMat'], 
                                AEQDSK_data['ZXPT1'],AEQDSK_data['ZXPT2'],
                                AEQDSK_data['RXPT1'],AEQDSK_data['RXPT2'],
                                AEQDSK_data['R0'],   AEQDSK_data['Z0'])
        
        power = np.single(gres[:,:,it] * R  * 2 * np.pi * dr * dz) #W per cell
 
        for k, v in get_powers(power, mask).items():
            emiss_regions.setdefault(k,[])
            emiss_regions[k].append(v)
          
    emiss_regions = {p: np.array(v) for p,v in emiss_regions.items()}
    emiss_regions['tvec'] = emiss['tvec']
    
    
    return emiss_regions



def generate_tomography(shot, emiss_regions):

 
   
    import MDSplus
    mdsserver = 'localhost'
    MDSconn = MDSplus.Connection(mdsserver)

    tree = 'EFIT01'
    MDSconn.openTree(tree, shot)
    SSIMAG = MDSconn.get(f'\\{tree}::TOP.RESULTS.GEQDSK:SSIMAG').data()
    valid = SSIMAG != 0
    SSIMAG = SSIMAG[valid]

    #PSIN = MDSconn.get(f'\\{tree}::TOP.RESULTS.GEQDSK:PSIN').data()
    PSIRZ = MDSconn.get(f'\\{tree}::TOP.RESULTS.GEQDSK:PSIRZ').data()[valid]
    GTIME = MDSconn.get(f'\\{tree}::TOP.RESULTS.GEQDSK:GTIME').data()[valid]/1e3
    SSIBRY = MDSconn.get(f'\\{tree}::TOP.RESULTS.GEQDSK:SSIBRY').data()[valid]

    Rgrid = MDSconn.get(f'\\{tree}::TOP.RESULTS.GEQDSK:R').data()
    Zgrid = MDSconn.get(f'\\{tree}::TOP.RESULTS.GEQDSK:Z').data()
    ATIME = MDSconn.get(f'\\{tree}::TOP.RESULTS.AEQDSK:ATIME').data()[valid]/1e3

    limiter = MDSconn.get(f'\\{tree}::TOP.RESULTS.GEQDSK.LIM').data() 

    AEQDSK_data = {}
    for ename in efit_name:
        AEQDSK_data[ename] = MDSconn.get(f'\\{tree}::TOP.RESULTS.AEQDSK:{ename}').data()[valid]
    


    PSIN = (PSIRZ - SSIMAG[:,None,None])/(SSIBRY-SSIMAG)[:,None,None]
 
    from skimage.measure import points_in_poly
    
    R,Z = np.meshgrid(Rgrid, Zgrid)

    grid = np.array([R.flatten(),Z.flatten()]).T
     
    pip = points_in_poly(grid, limiter)

    boundary = ~pip.reshape(R.shape)
 
    dS = np.diff(Rgrid)[0] * np.diff(Zgrid)[0] 
  
    emiss_2d = np.zeros((len(emiss_regions['tvec']), len(Rgrid), len(Zgrid)), dtype='single')

    #from scipy.interpolate import RectBivariateSpline
    for it, t in enumerate(emiss_regions['tvec']):
        g_nearest = np.argmin(np.abs(GTIME-t))
      

        mask = get_region_mask(t, Rgrid, Zgrid, ATIME,PSIN[g_nearest] , boundary, 
                                AEQDSK_data['ZXPT1'],AEQDSK_data['ZXPT2'],
                                AEQDSK_data['RXPT1'],AEQDSK_data['RXPT2'],
                                AEQDSK_data['R0'],   AEQDSK_data['Z0'])
         
        emiss_2d[it][mask == 1] = emiss_regions['P_ldivo'][it] / np.sum(R[mask == 1] * dS * 2 * np.pi)
        emiss_2d[it][mask == 2] = emiss_regions['P_ldivi'][it] / np.sum(R[mask == 2] * dS * 2 * np.pi)
        emiss_2d[it][mask == 3] = emiss_regions['P_udivo'][it] / np.sum(R[mask == 3] * dS * 2 * np.pi)
        emiss_2d[it][mask == 4] = emiss_regions['P_udivi'][it] / np.sum(R[mask == 4] * dS * 2 * np.pi)
        emiss_2d[it][mask == 7] = emiss_regions['P_axis'][it] / np.sum(R[mask == 7] * dS * 2 * np.pi)
        emiss_2d[it][mask == 6] = (emiss_regions['P_core'][it] - emiss_regions['P_axis'][it])/np.sum(R[mask == 6] * dS * 2 * np.pi)
        emiss_2d[it][mask == 5] = emiss_regions['P_SOL'][it] / np.sum(R[mask == 5] * dS * 2 * np.pi)
 
    
    return Rgrid, Zgrid, emiss_regions['tvec'], emiss_2d, limiter, PSIN, GTIME



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

def load_pcs_data(shot ):

    try:
        import MDSplus
        mdsserver = 'localhost'
        MDSconn = MDSplus.Connection(mdsserver)
   
        legacy_power = {}
        legacy_power['P_tot'] =  MDSconn.get(f'PTDATA("dgsnptot", {shot})').data()
        legacy_power['P_ldiv'] = MDSconn.get(f'PTDATA("dgsnpdivl", {shot})').data()
        legacy_power['P_udiv'] = MDSconn.get(f'PTDATA("dgsnpdivu", {shot})').data()
        legacy_power['P_udivo'] = MDSconn.get(f'PTDATA("dgsnpdivuo", {shot})').data()
        legacy_power['P_udivi'] = MDSconn.get(f'PTDATA("dgsnpdivui", {shot})').data()
        legacy_power['P_ldivo'] = MDSconn.get(f'PTDATA("dgsnpdivlo", {shot})').data()
        legacy_power['P_ldivi'] = MDSconn.get(f'PTDATA("dgsnpdivli", {shot})').data()
        legacy_power['P_core'] =  MDSconn.get(f'PTDATA("dgsnpcore", {shot})').data()
        legacy_power['P_SOL'] =   MDSconn.get(f'PTDATA("dgsnpsol", {shot})').data()
        legacy_power['P_axis'] =  MDSconn.get(f'_x = PTDATA("dgsnpaxis", {shot})').data()
        legacy_power['time'] = MDSconn.get('dim_of(_x)').data() #ms
    except:
        return None
    return legacy_power
 
def load_mds_data(shot, EFIT = 'EFIT01', realtime_bolo=True):

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
        print(alpha)
        b = [alpha]
        a = [1, -(1 - alpha)]
        return lfilter(b, a, x)
        
    tau_smooth = 0.04
    bolo_brightness = []
    #load realtime bolometer power 
    for fan in ['U', 'L']:
        for ich in range(24):
            ch = f'{fan}{ich+1:02}'
       
            #if ch in missing_channels:
                #continue
            
            #reatime hadata by itself has ~50ms delay
            if realtime_bolo:
                if ch in missing_channels:
                    data = MDSconn.get(f'_x=PTDATA2("DGSDPWRL01", {shot})').data()*0 
                else:
                    data = MDSconn.get(f'_x=PTDATA2("DGSDPWR{ch}", {shot})').data()
                #this adds a small delay, but negligible compared to the existing delay in the data
                #if tvec is None:
                    #tvec = MDSconn.get('dim_of(_x)').data()  #ms
                #dt = np.diff(tvec[tvec > 0]).mean() / 1e3
                data = lowpass_filter(data, alpha=0.01)
                
            else:
                data = MDSconn.get(TDIcall+f'BOL_{ch}_P').data() #W
                
                
            if len(data) <= 1:
                raise Exception(f'No data for channel {ch}')
         
            data *= etendue[fan][ich] * 1e4 #W/m^2
            bolo_brightness.append(data) 
            
             
    tvec = MDSconn.get('dim_of(_x)').data()  #ms
    bolo_brightness = np.array(bolo_brightness)
    
    
        
    #NOTE Important correct one broken channel
    if shot > 196700:
        bolo_brightness[45] = (bolo_brightness[44]+bolo_brightness[46]) / 2
 
    
    MDSconn.openTree('BOLOM', shot)


    legacy_power = {}
    legacy_power['time'] = MDSconn.get('\\BOLOM::TOP.PRAD_01.TIME').data() #ms
    legacy_power['P_tot'] = MDSconn.get('\\BOLOM::TOP.PRAD_01.PRAD.PRAD_TOT').data() #W
    legacy_power['P_ldiv'] = MDSconn.get('\\BOLOM::TOP.PRAD_01.PRAD.PRAD_DIVL').data() #W
    legacy_power['P_udiv'] = MDSconn.get('\\BOLOM::TOP.PRAD_01.PRAD.PRAD_DIVU').data() #W  
    legacy_power['P_core'] = MDSconn.get('\\BOLOM::TOP.BOLFIT01.ONED.POWER_CORE').data() #W
    legacy_power['P_SOL'] = MDSconn.get('_x=\\BOLOM::TOP.BOLFIT01.ONED.POWER_SOL').data() #W
    
    tvec_bolo = MDSconn.get('dim_of(_x)').data()  #ms
    legacy_power['P_core'] = np.interp(legacy_power['time'], tvec_bolo, legacy_power['P_core'])
    legacy_power['P_SOL'] = np.interp(legacy_power['time'], tvec_bolo, legacy_power['P_SOL'])
    
  

    #reinterpolate on realtime bolo timegrid 
    nearest = AEQDSK_data['ATIME'][:-1].searchsorted(tvec)
    inputs = np.array([AEQDSK_data[ename][nearest] for ename in efit_name])
    
    tmin, tmax = AEQDSK_data['ATIME'][[0,-1]]
    tind = slice(*tvec.searchsorted([tmin,tmax]))
    
    valid, inputs = clip_EFIT_inputs(inputs)

    print('Invalid points:', np.sum(~valid) )

     
    return tvec[tind], inputs.T[tind], bolo_brightness.T[tind], legacy_power
        
 
 
def W_ring_campaign():
    
    #167501 - 167873
    
    # --- Load the saved network ---
    network_file = 'trained_network_weighted.h5'
    nn_params = load_network(network_file)
    
    P_predicted = {}
    #for shot in range(167501, 167873):
    #170115 - 173902
    for shot in range(190000, 200000):
    #for shot in range(167252, 168500):
        #shot = 167873
        try:
            tvec, X, Y, legacy_power = load_mds_data(shot, realtime_bolo=False)
            if tvec[-1] < 3000:
                continue

            # --- Apply the network to the new data ---
            P_predicted[shot] = tvec, apply_model(nn_params,  X, Y)
            print(shot)
        except:
            pass
        
    #TODO select the range of W ring campaign 
    embed()
    
    P_axis = []
    P_core = []
    shot_time = []
    for shot, (tvec, data) in P_predicted.items():
        if shot > 167252:
            P_axis.append(data[:,power_params.index('P_axis')])
            P_core.append(data[:,power_params.index('P_core')])
            shot_time.append(shot + tvec/1e4)
    P_axis = np.hstack(P_axis)
    P_core = np.hstack(P_core)
    shot_time = np.hstack(shot_time)

    
    plt.plot(shot_time, P_axis)
    plt.show()
        

    P_axis = Phat_real[:,power_params.index('P_axis')]*1e6
    P_core = Phat_real[:,power_params.index('P_core')]*1e6
            
    import matplotlib.colors as colors
    plt.hist2d(P_axis/1e6,P_axis / (P_core-P_axis),cmap='Grays',  density=True,
               bins=100,range=[(0,1),(0,1.5)],norm=colors.LogNorm(),cmin=0.09,cmax=110)
    plt.xlabel('$P_{axis}$ [MW]')
    plt.ylabel('$P_{axis} / (P_{core}-P_{axis})$')
    plt.colorbar(label="Density")
    plt.show()
    

    import matplotlib.colors as colors
    plt.hist2d(P_core/1e6,P_axis/1e6,cmap='Grays',  density=True,
               bins=100,range=[(0,2),(0,1)],norm=colors.LogNorm(),
               cmin=0.09, cmax=110)
               #cmin=0.09,cmax=110)
    plt.xlabel('$P_{core}$ [MW]')
    plt.ylabel('$P_{axis}$ [MW]')
    plt.colorbar(label="Density")
    plt.show()    
        
    embed()
      
      
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

def plot_time_imshow(R, Z, t, data, limiter, psin, tvec_psin):
    """
    t     : (nt,)
    R, Z  : (nR,), (nZ,)
    data  : (nt, nR, nZ)
    """
    nt = data.shape[0]

    fig, ax = plt.subplots(figsize=(6,10))
    plt.subplots_adjust(bottom=0.2)
    extend = [R.min(), R.max(), Z.min(), Z.max()]
    im = ax.imshow(
        data[0]/1e6,
        extent=extend,
        origin="lower",
        aspect="equal",
        cmap='hot_r',
        interpolation='nearest'
    )
    ax.axis(extend)
    cbar = plt.colorbar(im, ax=ax, label='Radiated power [MW/m$^3$]')
    ax.set_xlabel("R [m]")
    ax.set_ylabel("Z [m]")
    ax.set_title(f"t = {t[0]:.3f}")
 
    pos_levels = np.linspace(0,1,11)
    cs = {}
    cs['pos'] = ax.contour(
        R, Z, psin[0],
        levels=pos_levels,
        linewidths=0.5,
        colors="k",
    )
    neg_levels = np.linspace(1,2,11)[1:]
    cs['neg'] = ax.contour(
        R, Z, psin[0],
        levels=neg_levels,
        colors="k",
        linewidths=0.5,
        linestyles='--',
    )   
    cs['zero'] = ax.contour(
        R, Z, psin[0],
        levels=[0],
        colors="k",
        linewidths=2,
        linestyles='-',
    )    
 
    ax.plot(limiter[:,0], limiter[:,1], c='k')

    ax_slider = plt.axes([0.2, 0.08, 0.6, 0.03])
    slider = Slider(ax_slider, "time", 0, nt - 1, valinit=0, valstep=1)

    def update(i):
        i = int(i)
        frame = data[i]/1e6
        im.set_data(frame)
        im.set_clim(0, frame.max())
        cbar.update_normal(im)
        ax.set_title(f"t = {t[i]:.3f}")
        
        # Remove previous contours
        for key in ["pos", "neg", 'zero']:
            cs[key].remove()
            
        # Draw new contours
        it = np.argmin(np.abs(tvec_psin - t[i]))
        cs["pos"] = ax.contour(R, Z, psin[it], levels=neg_levels, colors="k",linestyles='--',linewidths=0.5,  )
        cs["neg"] = ax.contour(R, Z, psin[it], levels=pos_levels, colors="k",linewidths=0.5, )
        cs["zero"] = ax.contour(R, Z, psin[it], levels=[0], colors="k",linewidths=2, )

        
        fig.canvas.draw_idle()

    slider.on_changed(update)

    def on_key(event):
        i = int(slider.val)
        if event.key == "right":
            i = min(i + 1, nt - 1)
        elif event.key == "left":
            i = max(i - 1, 0)
        else:
            return
        slider.set_val(i)

    fig.canvas.mpl_connect("key_press_event", on_key)

 
 
# ------------------------
# Example Usage
# ------------------------
if __name__ == "__main__":
    
 
    
    # --- Load the saved network ---
    network_file = 'trained_network_weighted.h5'
 
    nn_params = load_network(network_file)
    
    import sys
    real_time = False
    if len(sys.argv) > 2:
        shot = int(sys.argv[1])
        real_time = True
    elif len(sys.argv) > 1:
        shot = int(sys.argv[1])
    else:
        shot = 203401
    
        
    power_tomo = load_tomography(shot)
  
    
    power_tomo_old = load_GAPROFILES(shot)

    pcs_power  = load_pcs_data(shot )
    tvec, X, Y, legacy_power = load_mds_data(shot, realtime_bolo=real_time)
 

    # --- Apply the network to the new data ---
    P_predicted = apply_model(nn_params,  X, Y)
      
    powers = {p:P_predicted[:,i] for i,p in enumerate(power_params)}
    powers['tvec'] = tvec / 1e3
     
    
    plot_time_imshow(*generate_tomography(shot, powers))
 

    f,ax = plt.subplots(2,5, sharex=True, sharey=True, figsize=(10,8))
    ax = np.ravel(ax)
    for i, p in enumerate(power_params):
        ax[i].set_title(p)
        ax[i].plot(tvec/1e3,  P_predicted[:,i],'b-', label='Prediction')
        if p in power_tomo:
            ax[i].plot(power_tomo['tvec'],  power_tomo[p],'r--',label='PyTomo')
        if p in power_tomo_old:
            ax[i].plot(power_tomo_old['tvec'],  power_tomo_old[p],'g--o',label='GAPROFILES')
      
        if pcs_power is not None:
            if p in pcs_power:
                ax[i].plot(pcs_power['time']/1e3, pcs_power[p],':', label='PCS')
        
        elif p in legacy_power:
            ax[i].plot(legacy_power['time']/1e3, legacy_power[p],':', label='legacy')
                
                        
                        
        ax[i].axhline(0)
    ax[0].set_xlim(0, 7)
    ax[0].set_ylim(0, np.median(P_predicted[P_predicted[:,-1] > np.median(P_predicted[:,-1]),-1]) * 2)
    ax[-1].legend(loc='best')
    plt.tight_layout()
    f.savefig(f'bolo_{shot}')

 
    
    plt.show()
    
    
