# =========================================================
# Predict Engine Lifespan (then RUL) on NASA C-MAPSS FD001
# Strategy:
#   - Collapse each engine's time series into per-engine
#     aggregated + trend-based features.
#   - Predict total lifespan (max cycle).
#   - Convert to RUL on test:
#         RUL_pred = lifespan_pred - current_cycle
#   - Also produce "online" RUL curves per test engine
#     (prediction at each cycle, given data up to that cycle).
# =========================================================

from pathlib import Path
import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error

# -----------------------
# STEP 0 — CONFIG
# -----------------------
DATA_DIR = Path(
    r"C:\Users\KNH4ABT\OneDrive - Bosch Group\Data Science 25\Project-Jet Enginer Predictive Maintenance\CMAPSSData"
)
RANDOM_STATE = 42

# -----------------------
# STEP 1 — LOAD RAW DATA
# -----------------------
cols = (
    ['unit_number', 'time_in_cycles',
     'op_setting_1', 'op_setting_2', 'op_setting_3'] +
    [f'sensor_{i}' for i in range(1, 22)]
)

read_opts = dict(sep=r"\s+", header=None, engine="python")

train_df = pd.read_csv(DATA_DIR / "train_FD001.txt", **read_opts).dropna(axis=1, how="all")
test_df  = pd.read_csv(DATA_DIR / "test_FD001.txt",  **read_opts).dropna(axis=1,  how="all")
rul_df   = pd.read_csv(DATA_DIR / "RUL_FD001.txt",   header=None, names=["RUL"])

train_df.columns = cols
test_df.columns  = cols

print("Shapes (train / test / RUL):", train_df.shape, "/", test_df.shape, "/", rul_df.shape)
print("Engines (train / test):", train_df['unit_number'].nunique(), "/", test_df['unit_number'].nunique())

# -------------------------------------------------
# STEP 1.1 — DROP ZERO-VARIANCE COLUMNS GLOBALLY
# -------------------------------------------------
# Columns that never change in the whole training set
# carry no information and hurt linear models.
zero_var_cols = []
for c in train_df.columns:
    if c not in ["unit_number", "time_in_cycles"]:
        if train_df[c].std() == 0:
            zero_var_cols.append(c)

print("Dropped zero-variance columns:", zero_var_cols)

train_df = train_df.drop(columns=zero_var_cols)
test_df  = test_df.drop(columns=zero_var_cols)

# -------------------------------
# STEP 2 — LIFESPAN LABEL (TRAIN)
# -------------------------------
# Lifespan per engine = max time_in_cycles in train (run-to-failure).
lifespan_train = (
    train_df.groupby('unit_number')['time_in_cycles']
    .max()
    .rename('lifespan')
)

print("\nTrain lifespan (cycles) — summary:\n", lifespan_train.describe())

# ----------------------------------------------------
# STEP 3 — TEST LAST-CYCLE TABLE (UNIT, LAST_CYCLE, RUL_truth)
# ----------------------------------------------------
test_last = (
    test_df
    .sort_values(['unit_number', 'time_in_cycles'])
    .groupby('unit_number')
    .tail(1)
    .reset_index(drop=True)
)

test_last = test_last[['unit_number', 'time_in_cycles']].rename(columns={'time_in_cycles': 'last_cycle'})
print("\nTest last-cycle frame shape (expect 100 rows):", test_last.shape)

# NASA gives true RUL at the last recorded cycle for each test engine
rul_map = pd.Series(rul_df['RUL'].values, index=np.arange(1, len(rul_df) + 1))
test_last['RUL_truth'] = test_last['unit_number'].map(rul_map)

# ---------------------------------------------------------
# STEP 4 — AGGREGATED + TREND FEATURES PER ENGINE
# ---------------------------------------------------------

# Sensor columns AFTER dropping zero-variance
sensor_cols = [c for c in train_df.columns if c.startswith('sensor_')]

# Settings present AFTER dropping zero-variance
raw_setting_cols = ['op_setting_1', 'op_setting_2', 'op_setting_3']
setting_cols = [c for c in raw_setting_cols if c in train_df.columns]


def is_informative_series(df: pd.DataFrame, col: str, tol: float = 1e-8) -> bool:
    """Check if a setting varies across engines (keep only if variance > tol)."""
    eng_means = df.groupby('unit_number')[col].mean()
    return eng_means.var() > tol


informative_settings = [c for c in setting_cols if is_informative_series(train_df, c)]
print("\nIncluded operational settings:", informative_settings if informative_settings else "None")

agg_sources = sensor_cols + informative_settings

# Basic aggregations over the full engine lifespan
aggs = ['mean', 'std', 'min', 'max']


def build_agg_features(df: pd.DataFrame, cols_to_agg, aggs):
    """
    Per-engine aggregated features:
      engine -> [sensor/setting]_mean, _std, _min, _max
    """
    grouped = df.groupby('unit_number')[cols_to_agg].agg(aggs)
    grouped.columns = [f"{c}_{a}" for c, a in grouped.columns]
    grouped = grouped.reset_index()
    return grouped


agg_train = build_agg_features(train_df, agg_sources, aggs)
agg_test  = build_agg_features(test_df,  agg_sources, aggs)


def add_trend_features(full_df: pd.DataFrame, cols):
    """
    Trend features per engine and column:
      slope = (last - first) / total_cycles
      range = max - min
      delta = last - mean
    """
    rows = []
    for unit, grp in full_df.groupby('unit_number'):
        row = {'unit_number': unit}
        total_cycles = grp['time_in_cycles'].max()
        for c in cols:
            first = grp[c].iloc[0]
            last  = grp[c].iloc[-1]
            mean_c = grp[c].mean()
            row[f"{c}_slope"] = (last - first) / total_cycles
            row[f"{c}_range"] = grp[c].max() - grp[c].min()
            row[f"{c}_delta"] = last - mean_c
        rows.append(row)
    return pd.DataFrame(rows)


trend_train = add_trend_features(train_df, sensor_cols)
trend_test  = add_trend_features(test_df,  sensor_cols)

# Merge aggregated + trend features
train_agg = pd.merge(agg_train, trend_train, on='unit_number', how='left')
test_agg  = pd.merge(agg_test,  trend_test,  on='unit_number', how='left')

print("Feature matrix shapes (train/test):", train_agg.shape, "/", test_agg.shape)

# ---------------------------------------------------------
# STEP 5 — ASSEMBLE PER-ENGINE MATRICES (TRAIN / VAL / TEST)
# ---------------------------------------------------------
# Add labels for training (lifespan) and last-cycle/RUL for test
train_agg = train_agg.merge(lifespan_train.reset_index(), on='unit_number', how='left')
test_agg  = test_agg.merge(test_last, on='unit_number', how='left')

# Features = everything except id and labels
exclude_cols = {'unit_number', 'lifespan', 'last_cycle', 'RUL_truth'}
feature_cols = [c for c in train_agg.columns if c not in exclude_cols]
print("Final feature count:", len(feature_cols))

# Split engines into train/validation (per-engine, no leakage)
all_units = train_agg['unit_number'].values
tr_units, val_units = train_test_split(all_units, test_size=0.2, random_state=RANDOM_STATE)


def xy_from_units(df: pd.DataFrame, units, feats, target: str):
    sub = df[df['unit_number'].isin(units)].copy()
    X = sub[feats].values
    y = sub[target].values
    return X, y


X_tr, y_tr = xy_from_units(train_agg, tr_units, feature_cols, target='lifespan')
X_val, y_val = xy_from_units(train_agg, val_units, feature_cols, target='lifespan')

# Test (all engines)
X_test = test_agg[feature_cols].values
y_RUL_truth = test_agg['RUL_truth'].values
last_cycle_test = test_agg['last_cycle'].values

print("Rows — train / val / test:", X_tr.shape, "/", X_val.shape, "/", X_test.shape)

# ---------------------------------------------------------
# STEP 6 — METRICS + EVALUATION (LIFESPAN + RUL)
# ---------------------------------------------------------
def rmse(y_true, y_pred):
    return math.sqrt(mean_squared_error(y_true, y_pred))


def mae(y_true, y_pred):
    return mean_absolute_error(y_true, y_pred)


def fit_eval_lifespan(name, model, Xtr, ytr, Xval, yval, Xte, last_cycles, y_rul_truth):
    """
    1) Fit model to predict lifespan.
    2) Evaluate lifespan RMSE/MAE on validation.
    3) On test: predict lifespan, convert to RUL, evaluate RUL RMSE/MAE.
    """
    model.fit(Xtr, ytr)

    # Lifespan metrics on validation
    yval_pred_life = model.predict(Xval)
    life_rmse_val = rmse(y_val, yval_pred_life)
    life_mae_val  = mae(y_val, yval_pred_life)

    # Test: predict lifespan, convert to RUL at last recorded cycle
    ytest_pred_life = model.predict(Xte)
    ytest_pred_rul  = ytest_pred_life - last_cycles
    rul_rmse_test   = rmse(y_rul_truth, ytest_pred_rul)
    rul_mae_test    = mae(y_rul_truth, ytest_pred_rul)

    return {
        "model": name,
        "lifespan_rmse_val": life_rmse_val,
        "lifespan_mae_val":  life_mae_val,
        "rul_rmse_test":     rul_rmse_test,
        "rul_mae_test":      rul_mae_test
    }

# ---------------------------------------------------------
# STEP 7 — MODELS (PREDICT LIFESPAN PER ENGINE)
# ---------------------------------------------------------
models = {
    "LinearRegression": Pipeline([
        ("scaler", StandardScaler()),
        ("model", LinearRegression())
    ]),
    "Lasso": Pipeline([
        ("scaler", StandardScaler()),
        ("model", Lasso(alpha=0.001, max_iter=20000, random_state=RANDOM_STATE))
    ]),
    "RandomForest": RandomForestRegressor(
        n_estimators=800,
        max_depth=None,
        min_samples_leaf=2,
        max_features="sqrt",
        bootstrap=True,
        random_state=RANDOM_STATE,
        n_jobs=-1
    ),
    "GradientBoosting": GradientBoostingRegressor(
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        random_state=RANDOM_STATE
    ),
}

results = []
for name, model in models.items():
    m = fit_eval_lifespan(
        name, model,
        X_tr, y_tr,
        X_val, y_val,
        X_test, last_cycle_test, y_RUL_truth
    )
    results.append(m)
    print(f"{name}: lifespan_RMSE_val={m['lifespan_rmse_val']:.2f} | "
          f"RUL_RMSE_test={m['rul_rmse_test']:.2f}")

# ---------------------------------------------------------
# STEP 8 — COMPARISON TABLE & BEST MODEL
# ---------------------------------------------------------
cmp = (
    pd.DataFrame(results)
    .sort_values("lifespan_rmse_val")
    .reset_index(drop=True)
)

print("\n=== Lifespan + RUL Model Comparison (lower = better) ===")
print(cmp.round(3))

best = cmp.iloc[0]
print(f"\nSelected model: {best['model']} | "
      f"Lifespan RMSE_val={best['lifespan_rmse_val']:.2f} | "
      f"Lifespan MAE_val={best['lifespan_mae_val']:.2f} | "
      f"RUL RMSE_test={best['rul_rmse_test']:.2f} | "
      f"RUL MAE_test={best['rul_mae_test']:.2f}")

best_name = best['model']

# ---------------------------------------------------------
# STEP 9 — REFIT BEST MODEL ON ALL TRAIN ENGINES
# ---------------------------------------------------------
# Use all 100 train engines (train + val) for the final model.
X_full, y_full = xy_from_units(train_agg, all_units, feature_cols, target='lifespan')
best_model = models[best_name]
best_model.fit(X_full, y_full)

# ---------------------------------------------------------
# STEP 10 — ONLINE RUL PREDICTION PER TEST ENGINE
# ---------------------------------------------------------
def build_features_partial(df_partial: pd.DataFrame) -> pd.DataFrame:
    """
    Build ONE per-engine feature row from partial engine data:
      - aggregated features (mean/std/min/max)
      - trend features (slope/range/delta)
    Assumes df_partial contains exactly ONE engine.
    """
    agg_part = build_agg_features(df_partial, agg_sources, aggs)
    trend_part = add_trend_features(df_partial, sensor_cols)
    merged = pd.merge(agg_part, trend_part, on="unit_number", how="left")
    return merged


def online_rul_predictions(full_test_df: pd.DataFrame,
                           model,
                           feature_cols):
    """
    For each test engine and for each time step i:
      - use data from cycle 0..i
      - compute features
      - predict lifespan_i
      - RUL_i = lifespan_i - current_cycle
    Returns dict: unit_number -> DataFrame(cycle, RUL_pred)
    """
    results = {}

    for unit, df_eng in full_test_df.groupby("unit_number"):
        df_eng = df_eng.sort_values("time_in_cycles")
        preds = []

        for _, row in df_eng.iterrows():
            current_cycle = row["time_in_cycles"]

            # partial history up to current cycle
            partial = df_eng[df_eng["time_in_cycles"] <= current_cycle]

            # build features on partial data
            feat_df = build_features_partial(partial)
            X_part = feat_df[feature_cols].values.reshape(1, -1)

            # lifespan prediction at this moment
            life_pred = model.predict(X_part)[0]

            # online RUL prediction
            rul_pred = life_pred - current_cycle

            preds.append({
                "cycle": current_cycle,
                "rul_pred": rul_pred,
            })

        results[unit] = pd.DataFrame(preds)

    return results


online_preds = online_rul_predictions(test_df, best_model, feature_cols)
print("\n[OK] Generated online RUL prediction curves for all 100 test engines.")

# ---------------------------------------------------------
# STEP 11 — SIMPLE RUL PLOT FOR ONE TEST ENGINE
# ---------------------------------------------------------
# Example: pick engine 1 in the TEST set
example_unit = 1

# Online predictions (model-based)
df_online = online_preds[example_unit]

# Approximate "true" RUL curve for that engine:
# failure_cycle = last_cycle + RUL_truth  (from NASA file)
info_row = test_last[test_last["unit_number"] == example_unit].iloc[0]
failure_cycle = info_row["last_cycle"] + info_row["RUL_truth"]

df_engine = (
    test_df[test_df["unit_number"] == example_unit]
    .sort_values("time_in_cycles")
    .copy()
)
df_engine["RUL_true_approx"] = failure_cycle - df_engine["time_in_cycles"]

plt.figure(figsize=(8, 5))
plt.plot(df_engine["time_in_cycles"], df_engine["RUL_true_approx"],
         label="Approx. true RUL", linewidth=2)
plt.plot(df_online["cycle"], df_online["rul_pred"],
         label="Model online RUL", linestyle="--")

plt.xlabel("Time in cycles")
plt.ylabel("RUL (cycles)")
plt.title(f"Online RUL prediction — Test engine {example_unit}")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Extract model names + RUL RMSE
models = cmp["model"].values
rul_rmse = cmp["rul_rmse_test"].values

# Plot
plt.figure(figsize=(8,5))
bars = plt.bar(models, rul_rmse)

plt.title("Model Comparison – RUL RMSE")
plt.ylabel("RUL RMSE (lower = better)")
plt.xlabel("Models")

# Add RMSE values on top of bars
for bar, value in zip(bars, rul_rmse):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
             f"{value:.1f}", ha='center', va='bottom')

plt.tight_layout()
plt.show()

