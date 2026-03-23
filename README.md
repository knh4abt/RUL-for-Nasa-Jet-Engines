# Jet Engine RUL Prediction

![Python](https://img.shields.io/badge/python-3.8+-blue) ![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-orange) ![License](https://img.shields.io/badge/license-MIT-green)

Predicting Remaining Useful Life (RUL) for turbofan jet engines using the NASA C-MAPSS dataset. This project demonstrates an end-to-end ML pipeline with feature engineering, model comparison, and online prediction capabilities.

RandomForest with engineered features achieves **RMSE ~46 cycles** on the test set, outperforming linear baselines by 4x.

Built with scikit-learn and pandas.

---

## Results

Trained on 80/20 engine split. Best model selected by validation RMSE.

**Test RUL RMSE: 45.97 cycles** (RandomForest)

| Model | RUL RMSE | Notes |
|-------|----------|-------|
| RandomForest | 45.97 | Best performer |
| GradientBoosting | 49.05 | Strong alternative |
| Lasso | 185.23 | Linear baseline |
| LinearRegression | 185.88 | Simple baseline |

---

## Dataset

[NASA C-MAPSS FD001](https://data.nasa.gov/Aerospace/CMAPSS-Jet-Engine-Simulated-Data/ff5v-kuh6) — 100 turbofan engines run to failure with multivariate sensor time-series:

- **100 training engines** (complete run-to-failure)
- **100 test engines** (truncated before failure)
- **26 columns:** unit_number, time_in_cycles, 3 operational settings, 21 sensors
- **Single fault mode:** HPC degradation

---

## Approach

**Strategy:** Predict total lifespan (cycles until failure), then derive RUL:

```
RUL = predicted_lifespan - current_cycle
```

This avoids data leakage from direct RUL regression on sequential data.

**Feature Engineering:**
- Aggregated features: mean, std, min, max per sensor
- Trend features: slope, range, delta from mean
- 137 total features per engine

**Models Compared:**
- Linear: LinearRegression, Lasso (with StandardScaler)
- Ensemble: RandomForest, GradientBoosting

---

## Setup

```bash
pip install -r requirements.txt
```

Set dataset path in `main.py`:

```python
DATA_DIR = Path("/path/to/CMAPSSData")
```

The directory should contain `train_FD001.txt`, `test_FD001.txt`, and `RUL_FD001.txt`.

---

## Usage

Run the full pipeline:

```bash
python main.py
```

This will:
1. Load and preprocess data
2. Engineer features
3. Train and compare models
4. Generate online RUL predictions
5. Display comparison plots

---

## Project Structure

```
RUL-for-Nasa-Jet-Engines/
├── main.py              # Full pipeline: data → features → train → evaluate
├── requirements.txt     # Python dependencies
├── README.md            # This file
└── local_assets/        # Data files (not tracked)
    ├── train_FD001.txt
    ├── test_FD001.txt
    └── RUL_FD001.txt
```

---

## Key Implementation Details

- **No data leakage:** Train/val split is per-engine, not per-row
- **Zero-variance removal:** Constant columns dropped before training
- **Online prediction:** RUL estimated at each cycle using partial history
- **Reproducibility:** Fixed random seed (42) for all splits and models

---

## License

MIT
