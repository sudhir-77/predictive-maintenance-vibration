"""Fixed settings. Thresholds below are the project contract, not tuned on the test set."""

from __future__ import annotations

SEED = 0
N_THREADS = 4
PROJECT_SLUG = "predictive-maintenance-ops"
SPLIT_VERSION = 1

RUL_CLIP = 125
VAL_FRACTION = 0.20
SENSOR_STD_MIN = 1e-2

ALERT_LEAD_CYCLES = 15
ALERT_LEAD_FRACTION = 0.80
FALSE_ALERT_RUL = 50.0
FALSE_ALERTS_MAX = 0.10
FD001_RMSE_MAX = 18.0
FD003_RMSE_MAX = 20.0
COVERAGE_MIN = 70.0

ROLL_MEAN_WINDOWS = (5, 10, 20)
ROLL_STD_WINDOWS = (10,)
SLOPE_WINDOWS = (10, 20)

CNN_WINDOW = 20
CNN_EPOCHS = 20
CNN_PATIENCE = 4
CNN_BATCH = 128
CNN_LR = 1e-3
LGBM_MAX_ROUNDS = 400
LGBM_EARLY_STOPPING = 40

QUICK_DEV_UNITS = 12
QUICK_TEST_UNITS = 4
QUICK_CNN_EPOCHS = 2
QUICK_LGBM_ROUNDS = 30
QUICK_LGBM_EARLY_STOPPING = 10
QUICK_IFOREST_TREES = 40
IFOREST_TREES = 200

DATASETS = ("FD001", "FD003")

CMAPSS_URLS = (
    "https://phm-datasets.s3.amazonaws.com/NASA/6.+Turbofan+Engine+Degradation+Simulation+Data+Set.zip",
)

FILES = {
    "FD001": {
        "train": "train_FD001.txt",
        "test": "test_FD001.txt",
        "rul": "RUL_FD001.txt",
    },
    "FD003": {
        "train": "train_FD003.txt",
        "test": "test_FD003.txt",
        "rul": "RUL_FD003.txt",
    },
}

EXPECTED_ROWS = {
    "train_FD001.txt": 20631,
    "test_FD001.txt": 13096,
    "RUL_FD001.txt": 100,
    "train_FD003.txt": 24720,
    "test_FD003.txt": 16596,
    "RUL_FD003.txt": 100,
}

# SHA-256 of the extracted NASA text files (public-domain C-MAPSS release).
EXPECTED_SHA256 = {
    "train_FD001.txt": "963b5e22825b34d8b21c69e1aeb4af3e647050eb672ee8834ba4b5d91d2de0f8",
    "test_FD001.txt": "3cda7109ce17bafb5443f2ac926cfcf88154b941b8c4cf95eb55d1ddd6f52851",
    "RUL_FD001.txt": "a19c8ec94931949d0485bdc35118206e9c81c4547b422efb9cf86f4ceddbceca",
    "train_FD003.txt": "2abbe9968cc5e8eb091980f51b20f62bb4127336d3482cb52071d53bf23329e2",
    "test_FD003.txt": "299babd63c8d987cef079c4a425429f33b3a34797d803bbe2ad48c29dbd0d790",
    "RUL_FD003.txt": "df1e0566306b174a2de41c67a3e7a51877889598b78643fc3e5685259091b7cb",
}

OP_COLS = ["op1", "op2", "op3"]
SENSOR_COLS = [f"s{i}" for i in range(1, 22)]
BASE_COLS = ["unit_id", "cycle", *OP_COLS, *SENSOR_COLS]

ID_COLS = [
    "dataset",
    "origin",
    "split",
    "engine_key",
    "unit_id",
    "cycle",
    "source_row",
    "regime",
]
