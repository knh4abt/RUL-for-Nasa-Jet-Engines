"""
Model definitions for RUL prediction.
"""
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.pipeline import Pipeline

import config


def get_models():
    """Returns dict of model name -> sklearn estimator."""
    return {
        "LinearRegression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LinearRegression())
        ]),
        
        "Lasso": Pipeline([
            ("scaler", StandardScaler()),
            ("model", Lasso(
                random_state=config.RANDOM_STATE,
                **config.LASSO_PARAMS
            ))
        ]),
        
        "RandomForest": RandomForestRegressor(
            random_state=config.RANDOM_STATE,
            **config.RF_PARAMS
        ),
        
        "GradientBoosting": GradientBoostingRegressor(
            random_state=config.RANDOM_STATE,
            **config.GB_PARAMS
        ),
    }
