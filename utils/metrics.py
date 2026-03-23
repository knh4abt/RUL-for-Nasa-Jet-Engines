"""
Evaluation metrics for RUL prediction.
"""
import math
from sklearn.metrics import mean_squared_error, mean_absolute_error


def rmse(y_true, y_pred):
    """Root mean squared error."""
    return math.sqrt(mean_squared_error(y_true, y_pred))


def mae(y_true, y_pred):
    """Mean absolute error."""
    return mean_absolute_error(y_true, y_pred)


def evaluate_model(name, model, X_train, y_train, X_val, y_val, 
                   X_test, last_cycles, y_rul_truth):
    """
    Train model on lifespan, evaluate on validation and test.
    Returns dict with all metrics.
    """
    model.fit(X_train, y_train)
    
    # Validation (lifespan)
    y_val_pred = model.predict(X_val)
    life_rmse_val = rmse(y_val, y_val_pred)
    life_mae_val = mae(y_val, y_val_pred)
    
    # Test (convert lifespan -> RUL)
    y_test_pred_life = model.predict(X_test)
    y_test_pred_rul = y_test_pred_life - last_cycles
    rul_rmse_test = rmse(y_rul_truth, y_test_pred_rul)
    rul_mae_test = mae(y_rul_truth, y_test_pred_rul)
    
    return {
        "model": name,
        "lifespan_rmse_val": life_rmse_val,
        "lifespan_mae_val": life_mae_val,
        "rul_rmse_test": rul_rmse_test,
        "rul_mae_test": rul_mae_test,
    }
