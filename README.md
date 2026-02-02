# RUL Prediction for NASA Turbofan Jet Engines (C-MAPSS FD001)

Predict **Remaining Useful Life (RUL)** for NASA’s C-MAPSS turbofan engines using a simple, interpretable baseline:

1. **Collapse each engine time series into per-engine features** (aggregations + trends)
2. **Predict total lifespan** (max cycle) per engine
3. Convert to RUL on test: `RUL_pred = lifespan_pred - current_cycle`
4. Also generate **online RUL curves** (a prediction at each cycle using data up to that cycle)

This repo is intentionally focused on clarity + a complete pipeline: data loading → feature engineering → modeling → evaluation → plotting.

---

## Dataset

This project uses the **NASA C-MAPSS** turbofan engine degradation dataset.

- Subset used: **FD001**
- Files expected:
  - `train_FD001.txt`
  - `test_FD001.txt`
  - `RUL_FD001.txt`

You can obtain the dataset by searching for: *“NASA C-MAPSS FD001 dataset download”* (it is commonly hosted on NASA / university mirrors / Kaggle).

---

## Project Approach (What the code is doing)

### 1) Feature Engineering (Per Engine)
The raw data is a multivariate time series per engine. Instead of sequence models, we build a **single feature vector per engine**:

- **Aggregations** over the engine history:
  - mean, std, min, max
- **Trend features** (per sensor):
  - slope: `(last - first) / total_cycles`
  - range: `max - min`
  - delta: `last - mean`

Operational settings are included only if they vary meaningfully across engines.

### 2) Modeling Strategy
We train regressors to predict **engine lifespan** (the maximum cycle observed in the run-to-failure training set).

Models included:
- Linear Regression (with scaling)
- Lasso (with scaling)
- Random Forest Regressor
- Gradient Boosting Regressor

We select the “best” model by **validation RMSE on lifespan**, then:
- Refit it on all training engines
- Predict test engine lifespans
- Convert to RUL at the last recorded cycle and evaluate against NASA-provided RUL

### 3) Online RUL Curves
For each test engine and each time step, we recompute features from the partial history and predict an “online” RUL curve.

---

## Repository Structure

- `main.py` — end-to-end pipeline:
  - load data
  - engineer features
  - train/validate models
  - pick best model
  - evaluate on test
  - generate online RUL curves + plots

---

## How to Run

### 1) Install dependencies
Create and activate a virtual environment (recommended), then install:

```bash
pip install -r requirements.txt
```

If you don’t have a `requirements.txt` yet, a minimal set is:

```bash
pip install numpy pandas matplotlib scikit-learn
```

### 2) Set dataset path
The script currently expects the dataset directory to be configured in `main.py` via `DATA_DIR`.

Update this line in `main.py` to point to your local dataset folder:

```python
DATA_DIR = Path("path/to/CMAPSSData")
```

### 3) Run
```bash
python main.py
```

---

## Output / What to Expect

When you run the script, it prints:

- Train/test shapes
- Which columns were dropped due to zero variance
- Feature matrix sizes
- A model comparison table including:
  - lifespan RMSE/MAE (validation)
  - RUL RMSE/MAE (test)

It also generates plots:
- Online RUL curve for an example test engine
- Model comparison bar chart (RUL RMSE)

---

## Notes / Limitations

- This is a **baseline feature-based** approach (no LSTM/Transformer).
- “Online” predictions currently recompute features repeatedly from scratch; this is easy to understand but not optimized for speed.
- The model selection is based on validation RMSE of lifespan; alternative selection criteria may be more appropriate depending on your goals.
- FD001 is the simplest subset (single operating condition); generalizing to FD002–FD004 requires more care.

---

## Next Improvements (Planned)

Ideas if you want to take this further:
- Add a CLI (`argparse`) so you can run:
  - `python main.py --data-dir ./CMAPSSData`
- Refactor into modules (`data.py`, `features.py`, `train.py`, `plots.py`)
- Add unit tests for feature computation (`pytest`)
- Add cross-validation at the engine level
- Add more domain-specific RUL scoring metrics used in C-MAPSS literature

---

## License

Add a license if you plan to share this publicly (MIT is common for personal projects).
