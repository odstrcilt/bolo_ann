"""
2D Bolometric Tomography Reconstruction - Neural Network Inference
===================================================================
Reconstructs 2D plasma emissivity profiles from DIII-D bolometer
measurements using a trained JAX/Flax neural network (TomoNet).

Data is loaded from MDSplus, EFIT equilibrium parameters are used as
auxiliary inputs, and the result is displayed as an interactive
time-slider plot.

 
 
"""

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------
import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from scipy.signal import lfilter
import sys

import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.serialization import from_state_dict


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
real_time = False
if len(sys.argv) > 2:
    shot = int(sys.argv[1])
    real_time = True
elif len(sys.argv) > 1:
    shot = int(sys.argv[1])
else:
    shot = 199099
    
        
SHOT         = shot      # DIII-D shot number
EFIT_TREE    = "EFIT01"    # EFIT tree to load equilibrium data from
MODEL_FILE   = "tomo_nn_model.h5"  # Path to the trained model weights (HDF5)
TAU_SMOOTH   = 0.04        # Low-pass filter time constant for bolometer [s]
MDS_SERVER   = "localhost" # MDSplus server address

# EFIT scalar quantities used as auxiliary network inputs
EFIT_SCALARS = [
    "RXPT1",   # R of lower X-point [m]
    "RXPT2",   # R of upper X-point [m]
    "ZXPT1",   # Z of lower X-point [m]
    "ZXPT2",   # Z of upper X-point [m]
    "Z0",      # Z of magnetic axis [m]
    "R0",      # R of magnetic axis [m]
    "TRIBOT",  # Lower triangularity
    "TRITOP",  # Upper triangularity
    "KAPPA",   # Elongation
    "AMINOR",  # Minor radius [m]
    "DRSEP",   # Separation between primary and secondary separatrix [m]
]

# Bolometer etendue calibration factors [m^2*sr], indexed by fan and channel
ETENDUE = {
    "U": [
        3.0206e4, 2.9034e4, 2.8066e4, 2.7273e4, 2.6635e4,
        4.0340e4, 3.9855e4, 3.9488e4, 3.9235e4, 3.9091e4,
        3.9055e4, 3.9126e4, 0.7972e4, 0.8170e4, 0.8498e4,
        0.7549e4, 0.7129e4, 0.6854e4, 1.1162e4, 1.1070e4,
        1.1081e4, 1.1196e4, 1.1419e4, 1.1761e4,
    ],
    "L": [
        2.9321e4, 2.8825e4, 2.8449e4, 2.8187e4, 2.8033e4,
        0.7058e4, 0.7140e4, 0.7334e4, 0.7657e4, 0.8136e4,
        0.8819e4, 0.7112e4, 0.6654e4, 0.6330e4, 0.6123e4,
        2.9621e4, 2.9485e4, 2.9431e4, 2.9458e4, 2.9565e4,
        2.9756e4, 3.0032e4, 3.0397e4, 0.6406e4,
    ],
}

# Channels that are unavailable in real-time bolometer data
MISSING_RT_CHANNELS = ["U01", "L11", "L19", "L20"]


# ---------------------------------------------------------------------------
# Neural network definition
# ---------------------------------------------------------------------------
class TomoNet(nn.Module):
    """
    Fully-connected neural network for 2D bolometric tomography.

    Architecture
    ------------
    1. A linear projection (A) reduces the 48-channel bolometer
       brightness vector to a compact latent representation.
    2. The latent vector is concatenated with normalised EFIT scalars.
    3. Three fully-connected layers with SiLU activations map the
       combined input to output basis coefficients.
    4. A learnable spatial basis matrix (B) expands the coefficients
       into the reconstructed emissivity on the 2D pixel grid.
       Softplus is applied to enforce non-negativity.

    Parameters
    ----------
    n_in_basis  : number of latent bolometer dimensions
    n_out_basis : number of spatial basis functions
    hidden      : width of the hidden layers
    img_dim     : number of active (non-zero) pixels in the image mask
    """

    n_in_basis: int
    n_out_basis: int
    hidden: int
    img_dim: int

    def setup(self):
        # Bolometer projection (no bias - acts as a PCA-like compression)
        self.A = nn.Dense(self.n_in_basis, use_bias=False)

        # MLP trunk
        self.fc1 = nn.Dense(self.hidden)
        self.fc2 = nn.Dense(self.hidden)
        self.fc3 = nn.Dense(self.n_out_basis)

        # Learnable spatial basis: shape (img_dim, n_out_basis)
        # Initialised from the pre-computed B_init array (set before model.init)
        self.B = self.param(
            "B",
            lambda rng, shape: B_init,
            (self.img_dim, self.n_out_basis),
        )

    def __call__(self, raw, efit):
        """
        Forward pass.

        Parameters
        ----------
        raw  : (batch, 48)  - normalised bolometer brightness
        efit : (batch, 11)  - normalised EFIT scalar inputs

        Returns
        -------
        img  : (batch, img_dim) - predicted emissivity [W m^-3 sr^-1]
        """
        # Compress bolometer channels to latent space
        z = self.A(raw)

        # Concatenate with EFIT scalars
        x = jnp.concatenate([z, efit], axis=-1)

        # MLP with SiLU activations
        x = nn.silu(self.fc1(x))
        x = nn.silu(self.fc2(x))

        # Output coefficients - scaled to achieve the correct magnitude
        y = self.fc3(x) * 50

        # Linear combination of spatial basis functions; softplus ? non-negative
        img = jax.nn.softplus(jnp.dot(y, self.B.T))

        return img


# ---------------------------------------------------------------------------
# EFIT input preprocessing
# ---------------------------------------------------------------------------
def clip_efit_inputs(efit_values):
    """
    Replace physically implausible or missing EFIT values with sensible defaults.

    EFIT reconstructions occasionally fail to locate X-points, returning
    zero or out-of-range values.  This function detects those cases and
    substitutes typical DIII-D geometry values so the network receives
    a meaningful input rather than a degenerate one.

    Parameters
    ----------
    efit_values : ndarray, shape (n_scalars, n_times)
        Raw EFIT scalar time series in the order defined by EFIT_SCALARS.

    Returns
    -------
    valid       : boolean array of length n_times - True where RXPT1 > 0
    efit_values : ndarray - clipped in-place and returned
    """
    missing_lower_xpoint = efit_values[0] < 0   # RXPT1 < 0 ? no lower X-point
    missing_upper_xpoint = efit_values[1] < 0   # RXPT2 < 0 ? no upper X-point
    valid = efit_values[0] != 0

    # DRSEP: clamp occasional spikes to the typical double-null range
    efit_values[-1, efit_values[-1] >  0.3] =  0.06
    efit_values[-1, efit_values[-1] < -0.3] = -0.07

    # Substitute typical X-point coordinates when they are missing
    efit_values[2, missing_lower_xpoint] = -1.33   # ZXPT1
    efit_values[0, missing_lower_xpoint] =  1.28   # RXPT1
    efit_values[3, missing_upper_xpoint] =  1.40   # ZXPT2
    efit_values[1, missing_upper_xpoint] =  1.23   # RXPT2

    return valid, efit_values


# ---------------------------------------------------------------------------
# MDSplus data loader
# ---------------------------------------------------------------------------
def load_mds_data(shot, efit_tree=EFIT_TREE, realtime_bolo=False):
    """
    Load bolometer brightness and EFIT equilibrium data from MDSplus.

    Steps
    -----
    1. Connect to the local MDSplus server.
    2. Retrieve EFIT scalar time series from the AEQDSK node.
    3. Load per-channel bolometer power (W) from BOLOM tree.
    4. Apply etendue calibration to convert to brightness (W m^-2).
    5. Low-pass filter the brightness with time constant TAU_SMOOTH.
    6. Correct a known broken channel for shots after 196700.
    7. Interpolate EFIT scalars onto the bolometer time grid.
    8. Clip to the time window covered by both diagnostics.

    Parameters
    ----------
    shot         : int  - DIII-D discharge number
    efit_tree    : str  - MDSplus EFIT tree name (e.g. 'EFIT01', 'EFITRT1')
    realtime_bolo: bool - use real-time bolometer signals if True

    Returns
    -------
    tvec         : (n_times,)       time vector [ms]
    efit_data    : (n_times, 11)    EFIT scalar inputs
    bolo_data    : (n_times, 48)    calibrated bolometer brightness [W m^-2]
    """
    import MDSplus

    if realtime_bolo:
        print("Loading real-time bolometer data + EFITRT1.")
        efit_tree = "EFITRT1"
    else:
        print(f"Loading standard bolometer data + {efit_tree}.")

    conn = MDSplus.Connection(MDS_SERVER)

    # ------------------------------------------------------------------
    # 1.  EFIT equilibrium scalars
    # ------------------------------------------------------------------
    conn.openTree(efit_tree, shot)
    aeqdsk = {}
    for name in EFIT_SCALARS + ["ATIME"]:
        aeqdsk[name] = conn.get(f"\\{efit_tree}::TOP.RESULTS.AEQDSK:{name}").data()

    # ------------------------------------------------------------------
    # 2.  Bolometer brightness
    # ------------------------------------------------------------------
    if not realtime_bolo:
        tdi_prefix = "_x=\\BOLOM::TOP.PRAD_01.POWER:"
        conn.openTree("BOLOM", shot)

    def lowpass_filter(signal, tau, dt):
        """First-order IIR low-pass filter along the last axis."""
        alpha = dt / (dt + tau)
        return lfilter([alpha], [1, -(1 - alpha)], signal, axis=-1)

    bolo_brightness = []
    for fan in ["U", "L"]:
        for ich in range(24):
            ch = f"{fan}{ich + 1:02d}"

            if realtime_bolo:
                # Missing channels are zero-filled (hardware not available in RT)
                if ch in MISSING_RT_CHANNELS:
                    data = conn.get(f'_x=PTDATA2("DGSDPWRL01", {shot})').data() * 0
                else:
                    data = conn.get(f'_x=PTDATA2("DGSDPWR{ch}", {shot})').data()
            else:
                data = conn.get(tdi_prefix + f"BOL_{ch}_P").data()  # [W]

            if len(data) <= 1:
                raise RuntimeError(f"No bolometer data found for channel {ch}.")

            # Apply etendue to convert power ? brightness [W m^-2]
            data = data * ETENDUE[fan][ich] * 1e4
            bolo_brightness.append(data)

    # Time vector [ms] taken from the last loaded channel
    tvec = conn.get("dim_of(_x)").data()
    bolo_brightness = np.array(bolo_brightness)  # shape: (48, n_times)

    # Smooth with a low-pass filter to suppress high-frequency noise
    dt = np.diff(tvec[tvec > 0]).mean() / 1e3   # convert ms ? s
    bolo_brightness = lowpass_filter(bolo_brightness, TAU_SMOOTH, dt)

    # ------------------------------------------------------------------
    # 3.  Broken-channel correction (hardware fault after shot 196700)
    # ------------------------------------------------------------------
    if shot > 196700:
        # Channel L21 (index 45) is interpolated from its neighbours
        bolo_brightness[45] = (bolo_brightness[44] + bolo_brightness[46]) / 2

    # ------------------------------------------------------------------
    # 4.  Interpolate EFIT scalars onto the bolometer time grid
    # ------------------------------------------------------------------
    nearest = aeqdsk["ATIME"][:-1].searchsorted(tvec)
    efit_interp = np.array([aeqdsk[name][nearest] for name in EFIT_SCALARS])

    # Restrict to the time window covered by EFIT
    tmin, tmax = aeqdsk["ATIME"][[0, -1]]
    tind = slice(*tvec.searchsorted([tmin, tmax]))

    _, efit_interp = clip_efit_inputs(efit_interp)

    return tvec[tind], efit_interp.T[tind], bolo_brightness.T[tind]

 
# ---------------------------------------------------------------------------
# Main script
# ---------------------------------------------------------------------------
if __name__ == "__main__":

    print(f"Shot: {SHOT}")

    # -----------------------------------------------------------------------
    # 1.  Load trained model weights and normalisation statistics from HDF5
    # -----------------------------------------------------------------------
    def h5_to_dict(group):
        """Recursively convert an HDF5 group into a nested dict of JAX arrays."""
        d = {}
        for key, value in group.items():
            if isinstance(value, h5py.Group):
                d[key] = h5_to_dict(value)
            else:
                d[key] = jnp.array(value)
        return d

    print(f"Loading model from '{MODEL_FILE}' ...")
    with h5py.File(MODEL_FILE, "r") as f:
        params_dict    = h5_to_dict(f["params"])
        raw_mu         = np.array(f["raw_mu"])
        raw_std        = np.array(f["raw_std"])
        efit_mu        = np.array(f["efit_mu"])
        efit_std       = np.array(f["efit_std"])
        nnz            = np.array(f["nnz"]).astype(bool)   # pixel mask
        N_INPUT_BASIS  = int(f.attrs["N_INPUT_BASIS"])
        N_OUTPUT_BASIS = int(f.attrs["N_OUTPUT_BASIS"])
        HIDDEN         = int(f.attrs["HIDDEN"])
        IMG_DIM        = int(f.attrs["IMG_DIM"])

    # -----------------------------------------------------------------------
    # 2.  Load experimental data from MDSplus
    # -----------------------------------------------------------------------
    tvec, efit_data, bolo_data = load_mds_data(SHOT, efit_tree=EFIT_TREE, realtime_bolo=real_time)

    # Re-clip EFIT after interpolation (safety pass)
    efit_data = clip_efit_inputs(efit_data.T)[1].T

    # -----------------------------------------------------------------------
    # 3.  Preprocess inputs
    # -----------------------------------------------------------------------
    # Normalise bolometer data by its mean absolute value so the network
    # receives scale-invariant inputs; the scale is restored after inference.
    norm_power = np.abs(bolo_data).mean(axis=1)          # (n_times,)
    bolo_norm  = bolo_data / norm_power[:, None]
    
    def normalize(x, mu, std):
        """Z-score normalise x using pre-computed mean and standard deviation."""
        return (x - mu) / std


    bolo_norm  = normalize(bolo_norm,  raw_mu,  raw_std)
    efit_norm  = normalize(efit_data,  efit_mu, efit_std)

    # -----------------------------------------------------------------------
    # 4.  Build network and restore weights
    # -----------------------------------------------------------------------
    # B_init is a module-level variable used inside TomoNet.setup()
    B_init = np.zeros((nnz.sum(), N_OUTPUT_BASIS), dtype="float32")

    model = TomoNet(
        n_in_basis  = N_INPUT_BASIS,
        n_out_basis = N_OUTPUT_BASIS,
        hidden      = HIDDEN,
        img_dim     = int(nnz.sum()),
    )

    key    = jax.random.PRNGKey(0)
    params = from_state_dict(
        model.init(key, bolo_norm[:1], efit_norm[:1]),
        params_dict,
    )
    print("Model loaded and weights restored.")

    # -----------------------------------------------------------------------
    # 5.  Run inference
    # -----------------------------------------------------------------------
    print("Running tomographic reconstruction ...")
    y_pred  = model.apply(params, bolo_norm, efit_norm)
    y_pred  = np.array(y_pred) * norm_power[:, None]   # restore physical units

    # Map predictions onto the full pixel grid (NaN outside the plasma mask)
    n_pixels  = len(nnz)
    recon_flat = np.full((len(y_pred), n_pixels), np.nan, dtype="float32")
    recon_flat[:, nnz]  = y_pred
    recon_flat[:, ~nnz] = np.nan

    # Reshape to (n_times, n_rows=120, n_cols=80) for 2D display
    reconstructed = recon_flat.reshape(-1, 120, 80)/1e6
    print(f"Reconstruction complete: {len(reconstructed)} time frames.")

    # -----------------------------------------------------------------------
    # 6.  Interactive visualisation with time slider
    # -----------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(6, 8))
    plt.subplots_adjust(bottom=0.15)

    t0   = 0
    vmax = max(np.nanmean(reconstructed[t0]), 1) * 3

    im = ax.imshow(
        reconstructed[t0],
        cmap   = "inferno",
        vmin   = 0,
        vmax   = vmax,
        origin = "lower",
        aspect = "auto",
    )
    cbar = fig.colorbar(im, ax=ax, label="Emissivity [MW m^-2 sr^-1]")
    ax.set_title(f"DIII-D #{SHOT} - 2D Bolometric Tomography", fontsize=13)
    ax.set_xlabel("R index (pixel)")
    ax.set_ylabel("Z index (pixel)")

    # Time annotation in the corner
    time_text = ax.text(
        0.7, 0.97, f"t = {tvec[t0]:.0f} ms",
        transform=ax.transAxes,
        color="black", fontsize=10,
        verticalalignment="top",
    )

    # Slider widget
    ax_slider = plt.axes([0.15, 0.05, 0.70, 0.03])
    slider    = Slider(
        ax_slider, "Time index",
        valmin  = 0,
        valmax  = len(reconstructed) - 1,
        valinit = 0,
        valstep = 1,
    )

    def update(val):
        """Callback: redraw image and adjust colour scale for the selected frame."""
        t    = int(slider.val)
        vmax = max(np.nanmean(reconstructed[t]), 0.01) * 5
        im.set_data(reconstructed[t])
        im.set_clim(0, vmax)
        time_text.set_text(f"t = {tvec[t]:.0f} ms")
        fig.canvas.draw_idle()

    slider.on_changed(update)

    def on_key(event):
        """Keyboard navigation: ? / ? arrows step through time frames."""
        current = int(slider.val)
        if event.key == "right":
            slider.set_val(min(current + 1, len(reconstructed) - 1))
        elif event.key == "left":
            slider.set_val(max(current - 1, 0))

    fig.canvas.mpl_connect("key_press_event", on_key)

    update(0)
    plt.show()
