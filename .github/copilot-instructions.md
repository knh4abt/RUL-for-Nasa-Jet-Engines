# Copilot Instructions — Jet Engine Predictive Maintenance

## Project Overview
End-to-end RUL (Remaining Useful Life) prediction for jet engines using the NASA C-MAPSS FD001 dataset. The core strategy is to **predict total lifespan** (max cycles until failure) rather than RUL directly, then derive RUL as `lifespan_pred - current_cycle`.

## Architecture & Data Flow
```
Raw Data (train/test/RUL .txt files)
    ↓
Per-Engine Feature Engineering (aggregations + trends)
    ↓
Lifespan Prediction Models (scikit-learn pipelines)
    ↓
RUL Conversion (lifespan - current_cycle)
    ↓
Online RUL Curves (predictions at each cycle)
```

## Key Patterns

### Data Schema (26 columns)
Columns: `unit_number`, `time_in_cycles`, `op_setting_1-3`, `sensor_1-21`  
- Zero-variance columns are automatically dropped during preprocessing
- Settings with no cross-engine variance are excluded from features

### Feature Engineering (Critical)
Features are **per-engine aggregations**, not per-timestep. See `build_agg_features()` and `add_trend_features()` in [main.py](../main.py#L114-L145):
- **Aggregated**: mean, std, min, max of sensors/settings
- **Trend**: slope `(last-first)/cycles`, range, delta from mean

### Model Pipeline
Models use `sklearn.pipeline.Pipeline` with `StandardScaler` for linear models. Tree-based models (RandomForest, GradientBoosting) don't require scaling.

### Evaluation Strategy
- **Validation**: 80/20 split by engine (no data leakage between engines)
- **Metrics**: RMSE and MAE on both lifespan (validation) and RUL (test)
- **Best model** is selected on `lifespan_rmse_val`, then refitted on all training data

## Developer Workflow

### Environment Setup
```powershell
conda activate C:\Users\KNH4ABT\.conda\envs\ml-env
pip install -r requirements.txt
```

### Running the Pipeline
```powershell
python main.py
```
Outputs: model comparison table, online RUL curves for 100 test engines, visualization plots.

### Data Location
- **Production data**: `CMAPSSData/` directory (train_FD001.txt, test_FD001.txt, RUL_FD001.txt)
- **Local backup**: `local_assets/` contains same files for offline work

## Conventions

### Adding New Models
Add to the `models` dict in Step 7 (~line 252). Follow existing pattern:
```python
"ModelName": Pipeline([("scaler", StandardScaler()), ("model", YourModel())])
```

### Adding New Features
1. Compute in `build_agg_features()` or `add_trend_features()`
2. Features auto-propagate to train/test via `feature_cols` exclusion list
3. Ensure feature works with partial data for online predictions

### Online Predictions
The `build_features_partial()` function must work with incomplete engine histories (cycle 1 to current) for real-time RUL estimation.

## Dataset Notes
- **FD001**: Single operating condition (sea level), single fault mode (HPC degradation)
- 100 train engines (run to failure), 100 test engines (truncated before failure)
- Ground truth RUL provided only for test set's last recorded cycle
