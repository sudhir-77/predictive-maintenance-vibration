"""Fixed vibration settings. Thresholds are the project contract, not tuned on the test load."""

from __future__ import annotations

SEED = 0
SAMPLE_RATE_HZ = 12_000
WINDOW_LENGTH = 2048
N_WINDOWS = 36
# Load 3 hp is the test set. Loads 0-2 hp are development data, and each file
# is placed in train or validation, never both. The 2 hp recordings stay in
# training: 1750 rpm is the development speed closest to the 1730 rpm test load.
# Validation alternates between the 0 hp and 1 hp recording of each fault class.
N_FFT_BANDS = 8
N_ENVELOPE_BANDS = 8
ENVELOPE_BAND_HZ = 500.0
ENVELOPE_LOW_HZ = 2_000.0
ENVELOPE_HIGH_HZ = 5_000.0

# SKF 6205-2RS JEM, the drive-end bearing on the CWRU rig.
N_BALLS = 9
BALL_DIAMETER_IN = 0.3126
PITCH_DIAMETER_IN = 1.537
CONTACT_ANGLE_RAD = 0.0

RPM_BY_LOAD = {0: 1797, 1: 1772, 2: 1750, 3: 1730}

CLASS_ORDER = (
    "normal",
    "ir_007",
    "ir_014",
    "ir_021",
    "or_007",
    "or_014",
    "or_021",
    "ball_007",
    "ball_014",
    "ball_021",
)

DEV_LOADS = (0, 1, 2)
VAL_CANDIDATE_LOADS = (0, 1)
TEST_LOADS = (3,)
SPLIT_VERSION = 3

RF_TREES = 200
RF_MAX_DEPTH = 24
CNN_EPOCHS = 20
CNN_PATIENCE = 5
CNN_BATCH = 32
CNN_LR = 1e-3
CNN_WEIGHT_DECAY = 1e-4

QUICK_LABELS = ("ir_007", "or_007", "ball_007")
QUICK_LOADS = (0, 2, 3)
QUICK_WINDOWS = 6
QUICK_EPOCHS = 2
QUICK_TREES = 30

MACRO_F1_MIN = 0.90
BASELINE_MARGIN = 0.03
BOTH_HIGH_F1 = 0.95
BPFO_REL_ERROR_MAX = 0.02
COVERAGE_MIN = 75.0
PARENT_REL_TOL = 0.02

# Headline numbers from the verified parent run this extension must reproduce.
PARENT_FD001_RMSE = 15.110356
PARENT_FD003_RMSE = 15.980509
PARENT_LEAD_FRACTION = 1.0
PARENT_FALSE_ALERTS = 0.0

CWRU_URL = "https://engineering.case.edu/sites/default/files/{file_id}.mat"
PROJECT_SLUG = "predictive-maintenance-vibration"
