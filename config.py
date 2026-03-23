"""
Configuration for RUL prediction pipeline.
"""
from pathlib import Path
import os

# Data paths
DATA_DIR = Path(
    os.environ.get(
        "RUL_DATA_DIR",
        r"C:\Users\KNH4ABT\OneDrive - Bosch Group\Data Science 25\Project-Jet Enginer Predictive Maintenance\CMAPSSData"
    )
)

# Column definitions
COLUMNS = (
    ['unit_number', 'time_in_cycles',
     'op_setting_1', 'op_setting_2', 'op_setting_3'] +
    [f'sensor_{i}' for i in range(1, 22)]
)

# Training
RANDOM_STATE = 42
VAL_SIZE = 0.2

# Model hyperparameters
RF_PARAMS = {
    "n_estimators": 800,
    "max_depth": None,
    "min_samples_leaf": 2,
    "max_features": "sqrt",
    "bootstrap": True,
    "n_jobs": -1,
}

GB_PARAMS = {
    "n_estimators": 1000,
    "learning_rate": 0.05,
    "max_depth": 4,
    "subsample": 0.8,
}

LASSO_PARAMS = {
    "alpha": 0.001,
    "max_iter": 20000,
}
