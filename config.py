"""
Physical constants, noise levels and MBHB priors shared by stages 2 and 3
(pe_smbhb/, pe_ucb/, heron/stages.py), and the confusion PSD.

The detector (stage 1) has its own configuration, yolo/yolo_config.py: band,
spectrogram, dataset sizes and training settings. Where a name appears in both
files, the detector uses the value in yolo/yolo_config.py.
"""
import numpy as np


# LISA / OBSERVATION

YRSID_SI = 3.15576e7
T_OBS_DAYS = 30.0
T_OBS = T_OBS_DAYS * 86400       # seconds
DT = 15.0                        # sampling interval (seconds)


# NARROW BAND (for YOLO detection spectrograms)

F_MIN = 2.53e-3                  # Hz  (2.53 mHz)
F_MAX = 2.64e-3                  # Hz  (2.64 mHz, 110 uHz bandwidth)


# WIDEBAND (for SBI sky localization)
# The SMBHB chirp spans ~0.1-10 mHz over 30 days.
# Log-spaced sub-bands sample different orbital phases / antenna patterns.
# More bands = more measurements of the time-varying A/E ratio.

WB_F_MIN = 1e-4                  # Hz  (0.1 mHz — matches LISA sensitivity)
WB_F_MAX = 1e-2                  # Hz  (10 mHz — covers SMBHB inspiral)
WB_N_BANDS = 64
WB_FEATURES_PER_BAND = 11           # 7 power/cross + 2 phase gradients + 2 SNR proxies


# UCB PARAMETERS

UCB_LOG10A_MIN = -22.5
UCB_LOG10A_MAX = -20.5
UCB_F_MIN = 2.54e-3
UCB_F_MAX = 2.63e-3
MIN_SNR = 10.0


# SMBHB PARAMETERS

SMBHB_M1_MIN = 7e5              # m1,m2 ~ U[7e5, 3e7] independent (m2<=m1)
SMBHB_M1_MAX = 3e7              # => Mc in ~[6e5, 2.6e7], q = m1/m2 in [1, ~43]
SMBHB_M2_MIN = 7e5
SMBHB_M2_MAX = 3e7
SMBHB_DIST_MIN = 5000             # Mpc  (5 Gpc)
SMBHB_DIST_MAX = 200000           # Mpc  (200 Gpc) — wider to cover logD up to 5.3
SMBHB_T_MERGE_MIN = 2           # hours after end of observation
SMBHB_T_MERGE_MAX = 600


# NOISE (LISA SciRD)

SA = np.log10((3.0e-15)**2)
SI = np.log10((15.0e-12)**2)


# DATASET SPLITS

TRAIN_UCB_MIN = 3
TRAIN_UCB_MAX = 8
TRAIN_SMBHB_PROB = 1.0
VAL_UCB_MIN = 3
VAL_UCB_MAX = 8
VAL_SMBHB_PROB = 1.0
N_TRAIN = 30000
N_VAL = 2000
N_TEST = 1000
SEED = 42


# SPECTROGRAM / IMAGE (for YOLO)

NPERSEG = 16384
IMG_WIDTH = 640
IMG_HEIGHT = 640


# YOLO

CLASS_NAMES = ['ucb', 'smbhb']
NUM_CLASSES = 2
CLS_UCB = 0
CLS_SMBHB = 1
CONF_THRESHOLD = 0.25
IOU_THRESHOLD = 0.45
IMG_SIZE = 640
BATCH_SIZE = 64
EPOCHS = 50
PATIENCE = 15


# GALACTIC CONFUSION NOISE (Cornish & Robson 2017)
# Unresolved galactic binary foreground for T_obs = 0.5-4 yr
# Parameterized as S_c(f) = A * f^(-7/3) * exp(-f^a + b*f*sin(k*f))
# with coefficients for T_obs ~ 1 month (conservative, use 0.5yr fit)

CONF_A = 9e-45          # amplitude
CONF_ALPHA = 0.138      # exponent a
CONF_BETA = -221.0      # coefficient b
CONF_KAPPA = 521.0      # coefficient k
CONF_GAMMA = 1680.0     # knee frequency factor
CONF_FK = 0.00113       # knee frequency (Hz)

def galactic_confusion_psd(f):
    """
    Galactic confusion noise PSD from Cornish & Robson (2017).
    Returns S_c(f) in strain^2/Hz for the A/E TDI channels.

    Args:
        f: frequency array in Hz
    Returns:
        S_c: confusion noise PSD array
    """
    f = np.atleast_1d(f)
    S_c = np.zeros_like(f)
    mask = f > 0
    fm = f[mask]
    S_c[mask] = (CONF_A * fm**(-7.0/3.0)
                 * np.exp(-fm**CONF_ALPHA
                          + CONF_BETA * fm * np.sin(CONF_KAPPA * fm))
                 * (1.0 + np.tanh(CONF_GAMMA * (CONF_FK - fm))))
    return S_c


# ROLLING WINDOW MODE (5-day window, updated daily)

T_WINDOW_DAYS = 5.0              # rolling window duration
T_WINDOW = T_WINDOW_DAYS * 86400  # 432000 seconds
DT_WINDOW = 15.0                 # sampling interval
DF_WINDOW = 1.0 / T_WINDOW      # ~2.31e-6 Hz frequency resolution
# Training band (narrow band, ~48 bins at 5-day resolution)
WINDOW_F_MIN = F_MIN             # 2.53 mHz
WINDOW_F_MAX = F_MAX             # 2.64 mHz
WINDOW_N_BINS = int(round((WINDOW_F_MAX - WINDOW_F_MIN) / DF_WINDOW))  # ~48
# Full-band inference range
WINDOW_FULL_F_MIN = 1e-4         # 0.1 mHz
WINDOW_FULL_F_MAX = 1e-2         # 10 mHz

# Wideband PE mode (0.5-5 mHz, ~1950 bins at 5-day resolution)
# The chirp sweeps across many freq bins → each bin = different time →
# different antenna orientation → sky information
WB_WINDOW_F_MIN = 5e-4           # 0.5 mHz (above LISA confusion noise knee)
WB_WINDOW_F_MAX = 5e-3           # 5 mHz (covers SMBHB inspiral)
WB_WINDOW_N_BINS = int(round((WB_WINDOW_F_MAX - WB_WINDOW_F_MIN) / DF_WINDOW))  # ~1950


# YEARLY MODE (1 year observation, daily updates)
# Full orbital modulation → proper sky localization

T_YEAR_DAYS = 365.25
T_YEAR = T_YEAR_DAYS * 86400         # 1 year in seconds
DT_YEAR = 15.0                       # same sampling interval
DF_YEAR = 1.0 / T_YEAR              # ~3.17e-8 Hz (very fine)
# Analysis: extract at coarser resolution (Δf = 1e-6 Hz → ~4500 bins in 0.5-5 mHz)
DF_YEAR_ANALYSIS = 1e-6              # Hz — practical analysis resolution
YEAR_F_MIN = 5e-4                    # 0.5 mHz
YEAR_F_MAX = 5e-3                    # 5 mHz
YEAR_N_BINS = int(round((YEAR_F_MAX - YEAR_F_MIN) / DF_YEAR_ANALYSIS))  # ~4500


# PATHS  (adjust these for your system)

import os as _os
NGB_PATH = _os.path.dirname(_os.path.abspath(__file__))  # vendored data_utils.py lives in-repo
