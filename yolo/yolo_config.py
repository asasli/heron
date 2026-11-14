"""
Configuration for LISA-YOLO Pipeline (UCB + SMBHB Detection)

STRATEGY:
- Train: 1-5 UCBs + 0-1 SMBHB per image
- Val/Test: 10-20 UCBs + 0-1 SMBHB per image
- Observation time: 1 month
"""

import numpy as np


# LISA / OBSERVATION PARAMETERS


YRSID_SI = 3.15576e7  # seconds per year
T_OBS_DAYS = 30.0  # 1 month
T_OBS = T_OBS_DAYS * 86400  # seconds
DT = 15.0  # sampling interval (seconds)


# FREQUENCY BAND (narrow band for UCB detection)


F_MIN = 2.53e-3   # 2.53 mHz
F_MAX = 2.64e-3   # 2.64 mHz (110 μHz bandwidth)


# UCB PARAMETERS


# Amplitude range (log10)
UCB_LOG10A_MIN = -22.5
UCB_LOG10A_MAX = -20.5

# Frequency range for UCBs (within F_MIN-F_MAX, with margin)
UCB_F_MIN = 2.54e-3  # 2.54 mHz (with margin)
UCB_F_MAX = 2.63e-3  # 2.63 mHz (with margin)

# Minimum SNR for detectable sources
MIN_SNR = 10.0


# SMBHB PARAMETERS (for narrow band ~2.5-2.6 mHz)


# Mass ranges - smaller masses for signals in mHz band
SMBHB_M1_MIN = 1e6
SMBHB_M1_MAX = 5e6
SMBHB_M2_MIN = 5e5
SMBHB_M2_MAX = 3e6

# Distance range (Mpc) - closer for stronger signals
SMBHB_DIST_MIN = 200
SMBHB_DIST_MAX = 2000

# Time to merger (hours AFTER end of observation)
# Small values = chirp near end of image (right edge)
# Large values = chirp earlier in image (left side)
# For 30-day observation: 720 hours = 30 days, so 360h = middle
SMBHB_T_MERGE_MIN = 2      # hours (chirp at ~day 30, right edge)
SMBHB_T_MERGE_MAX = 600    # hours (chirp at ~day 5, left side)


# NOISE PARAMETERS (LISA SciRD)


SA = np.log10((3.0e-15)**2)  # acceleration noise
SI = np.log10((15.0e-12)**2)  # position noise


# DATASET CONFIGURATION


# Training: fewer sources (learn detection)
TRAIN_UCB_MIN = 1
TRAIN_UCB_MAX = 5
TRAIN_SMBHB_PROB = 0.5  # 50% chance of having SMBHB

# Validation/Test: more sources (test generalization)
VAL_UCB_MIN = 10
VAL_UCB_MAX = 20
VAL_SMBHB_PROB = 0.5

N_TRAIN = 5000
N_VAL = 1000
N_TEST = 500
SEED = 42            # dataset generation seed
TRAIN_SEED = 0       # detector training seed


# SPECTROGRAM / IMAGE PARAMETERS


NPERSEG = 16384  # FFT segment size for STFT (gives ~4 μHz resolution)
NOVERLAP_RATIO = 0.9  # overlap ratio

IMG_WIDTH = 640
IMG_HEIGHT = 640


# YOLO CONFIGURATION


YOLO_MODEL = 'yolo26n.pt'
EPOCHS = 80          # the published detector: 80 epochs (docs/provenance/yolo/args.yaml)
BATCH_SIZE = 16
IMG_SIZE = 640
PATIENCE = 20

# Inference thresholds
CONF_THRESHOLD = 0.25
IOU_THRESHOLD = 0.45


# CLASS DEFINITIONS


CLASS_NAMES = ['ucb', 'smbhb']
NUM_CLASSES = 2

# Class indices
CLS_UCB = 0
CLS_SMBHB = 1


# PATHS


DATA_DIR = 'data/yolo'

# Path to NGB library
import os as _os
NGB_PATH = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))  # repo root (vendored data_utils.py)
