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
#   - Includes LSTM neural network for sequence-based prediction.
# =========================================================

from pathlib import Path
import math
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Deep Learning imports
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
tf.random.set_seed(42)

# -----------------------
# STEP 0 — CONFIG
# -----------------------
DATA_DIR = Path(
    r"C:\Users\KNH4ABT\OneDrive - Bosch Group\Data Science 25\Project-Jet Enginer Predictive Maintenance\local_assets"
)
OUTPUT_DIR = Path(
    r"C:\Users\KNH4ABT\OneDrive - Bosch Group\Data Science 25\Project-Jet Enginer Predictive Maintenance\outputs"
)
OUTPUT_DIR.mkdir(exist_ok=True)  # Create folder if it doesn't exist

RANDOM_STATE = 42
MAX_SEQUENCE_LENGTH = 50  # Max cycles for LSTM input

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
    "Ridge": Pipeline([
        ("scaler", StandardScaler()),
        ("model", Ridge(alpha=1.0, random_state=RANDOM_STATE))
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

# ---------------------------------------------------------
# STEP 7.1 — LSTM MODEL (SEQUENCE-BASED PREDICTION)
# ---------------------------------------------------------
# LSTM needs sequence data, not aggregated features.
# We'll prepare sequences from raw sensor data.

def prepare_sequences_for_lstm(df, sensor_cols, max_len, lifespan_dict=None):
    """
    Convert raw time-series data into LSTM-ready sequences.
    
    Each engine becomes ONE sequence:
      - Input: padded sensor readings (max_len x num_sensors)
      - Output: lifespan (total cycles until failure)
    
    Args:
        df: DataFrame with unit_number, time_in_cycles, sensors
        sensor_cols: List of sensor column names
        max_len: Maximum sequence length (pad/truncate to this)
        lifespan_dict: Dict {unit_number: lifespan} for training labels
    
    Returns:
        X: array of shape (n_engines, max_len, n_sensors)
        y: array of shape (n_engines,) - lifespan labels (if provided)
        units: list of unit numbers in order
    """
    X_list = []
    y_list = []
    units = []
    
    for unit, grp in df.groupby('unit_number'):
        grp = grp.sort_values('time_in_cycles')
        seq = grp[sensor_cols].values  # shape: (cycles, n_sensors)
        
        # Pad or truncate to max_len
        if len(seq) > max_len:
            seq = seq[-max_len:]  # Take last max_len cycles
        elif len(seq) < max_len:
            # Pad with zeros at the beginning
            pad = np.zeros((max_len - len(seq), len(sensor_cols)))
            seq = np.vstack([pad, seq])
        
        X_list.append(seq)
        units.append(unit)
        
        if lifespan_dict is not None:
            y_list.append(lifespan_dict.get(unit, 0))
    
    X = np.array(X_list)
    y = np.array(y_list) if lifespan_dict else None
    
    return X, y, units


def build_lstm_model(input_shape):
    """
    Build a simple LSTM model for lifespan prediction.
    
    Architecture:
      - LSTM layer (64 units) - learns patterns in sequences
      - Dropout (20%) - prevents overfitting
      - LSTM layer (32 units) - deeper pattern recognition
      - Dense output (1 unit) - predicts lifespan
    """
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(32, return_sequences=False),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(1)  # Linear output for regression
    ])
    
    model.compile(
        optimizer='adam',
        loss='mse',  # Mean Squared Error for regression
        metrics=['mae']
    )
    
    return model


def train_and_eval_lstm(train_df, val_units, test_df, sensor_cols, 
                        lifespan_dict, test_last_df, max_len=50):
    """
    Train LSTM and evaluate on validation + test sets.
    
    Returns dict with same structure as fit_eval_lifespan().
    """
    # Split train data by units
    train_units = [u for u in lifespan_dict.keys() if u not in val_units]
    
    train_subset = train_df[train_df['unit_number'].isin(train_units)]
    val_subset = train_df[train_df['unit_number'].isin(val_units)]
    
    # Prepare sequences
    X_tr_lstm, y_tr_lstm, _ = prepare_sequences_for_lstm(
        train_subset, sensor_cols, max_len, lifespan_dict
    )
    X_val_lstm, y_val_lstm, _ = prepare_sequences_for_lstm(
        val_subset, sensor_cols, max_len, lifespan_dict
    )
    
    # Normalize sensor values
    scaler = MinMaxScaler()
    n_samples, n_steps, n_features = X_tr_lstm.shape
    
    # Reshape for scaling, then reshape back
    X_tr_flat = X_tr_lstm.reshape(-1, n_features)
    X_val_flat = X_val_lstm.reshape(-1, n_features)
    
    X_tr_flat = scaler.fit_transform(X_tr_flat)
    X_val_flat = scaler.transform(X_val_flat)
    
    X_tr_lstm = X_tr_flat.reshape(n_samples, n_steps, n_features)
    X_val_lstm = X_val_flat.reshape(len(val_units), n_steps, n_features)
    
    # Build and train model
    model = build_lstm_model((max_len, len(sensor_cols)))
    
    early_stop = EarlyStopping(
        monitor='val_loss', 
        patience=10, 
        restore_best_weights=True,
        verbose=0
    )
    
    print("\nTraining LSTM...")
    model.fit(
        X_tr_lstm, y_tr_lstm,
        validation_data=(X_val_lstm, y_val_lstm),
        epochs=100,
        batch_size=16,
        callbacks=[early_stop],
        verbose=0
    )
    
    # Evaluate on validation (lifespan)
    y_val_pred = model.predict(X_val_lstm, verbose=0).flatten()
    life_rmse_val = math.sqrt(mean_squared_error(y_val_lstm, y_val_pred))
    life_mae_val = mean_absolute_error(y_val_lstm, y_val_pred)
    
    # Prepare test sequences
    X_test_lstm, _, test_units = prepare_sequences_for_lstm(
        test_df, sensor_cols, max_len
    )
    
    X_test_flat = X_test_lstm.reshape(-1, n_features)
    X_test_flat = scaler.transform(X_test_flat)
    X_test_lstm = X_test_flat.reshape(len(test_units), n_steps, n_features)
    
    # Predict lifespan on test
    y_test_pred_life = model.predict(X_test_lstm, verbose=0).flatten()
    
    # Convert to RUL: RUL = lifespan_pred - last_cycle
    test_last_sorted = test_last_df.sort_values('unit_number')
    last_cycles = test_last_sorted['last_cycle'].values
    rul_truth = test_last_sorted['RUL_truth'].values
    
    y_test_pred_rul = y_test_pred_life - last_cycles
    rul_rmse_test = math.sqrt(mean_squared_error(rul_truth, y_test_pred_rul))
    rul_mae_test = mean_absolute_error(rul_truth, y_test_pred_rul)
    
    return {
        "model": "LSTM",
        "lifespan_rmse_val": life_rmse_val,
        "lifespan_mae_val": life_mae_val,
        "rul_rmse_test": rul_rmse_test,
        "rul_mae_test": rul_mae_test
    }

results = []

# Train traditional ML models (use aggregated features)
print("\n--- Training Traditional ML Models ---")
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

# Train LSTM model (uses raw sequence data)
print("\n--- Training LSTM Neural Network ---")
lifespan_dict = lifespan_train.to_dict()
lstm_result = train_and_eval_lstm(
    train_df=train_df,
    val_units=val_units,
    test_df=test_df,
    sensor_cols=sensor_cols,
    lifespan_dict=lifespan_dict,
    test_last_df=test_last,
    max_len=MAX_SEQUENCE_LENGTH
)
results.append(lstm_result)
print(f"LSTM: lifespan_RMSE_val={lstm_result['lifespan_rmse_val']:.2f} | "
      f"RUL_RMSE_test={lstm_result['rul_rmse_test']:.2f}")

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
# Note: LSTM is handled separately above; for online predictions we use ML models.
X_full, y_full = xy_from_units(train_agg, all_units, feature_cols, target='lifespan')

# If best model is LSTM, use GradientBoosting for online predictions
# (LSTM requires different data format for online use)
if best_name == "LSTM":
    print("\nNote: LSTM selected, but using GradientBoosting for online RUL curves.")
    best_model_for_online = models["GradientBoosting"]
else:
    best_model_for_online = models[best_name]

best_model_for_online.fit(X_full, y_full)

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


online_preds = online_rul_predictions(test_df, best_model_for_online, feature_cols)
print("\n[OK] Generated online RUL prediction curves for all 100 test engines.")

# ---------------------------------------------------------
# STEP 11 — ENHANCED VISUALIZATION DASHBOARD
# ---------------------------------------------------------
# Set global plot style for professional look
plt.style.use('seaborn-v0_8-whitegrid')
COLORS = {
    'primary': '#2E86AB',
    'secondary': '#A23B72', 
    'success': '#28A745',
    'warning': '#F18F01',
    'danger': '#C73E1D',
    'info': '#17A2B8'
}

# --- FIGURE 1: Multi-panel Dashboard ---
fig = plt.figure(figsize=(16, 10))
fig.suptitle('Jet Engine RUL Prediction Dashboard', fontsize=16, fontweight='bold', y=1.02)

# Panel 1: Model Comparison Bar Chart (top-left)
ax1 = fig.add_subplot(2, 2, 1)
model_names = cmp["model"].values
rul_rmse_values = cmp["rul_rmse_test"].values
colors = [COLORS['success'] if v == min(rul_rmse_values) else COLORS['primary'] for v in rul_rmse_values]
bars = ax1.bar(model_names, rul_rmse_values, color=colors, edgecolor='black', linewidth=0.5)
ax1.set_title('Model Comparison – RUL RMSE', fontsize=12, fontweight='bold')
ax1.set_ylabel('RUL RMSE (lower = better)')
ax1.set_xlabel('Models')
ax1.tick_params(axis='x', rotation=20)
for bar, value in zip(bars, rul_rmse_values):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
             f"{value:.1f}", ha='center', va='bottom', fontweight='bold', fontsize=9)
ax1.set_ylim(0, max(rul_rmse_values) * 1.15)

# Panel 2: Online RUL Prediction for Example Engine (top-right)
ax2 = fig.add_subplot(2, 2, 2)
example_unit = 1
df_online = online_preds[example_unit]
info_row = test_last[test_last["unit_number"] == example_unit].iloc[0]
failure_cycle = info_row["last_cycle"] + info_row["RUL_truth"]
df_engine = (
    test_df[test_df["unit_number"] == example_unit]
    .sort_values("time_in_cycles")
    .copy()
)
df_engine["RUL_true_approx"] = failure_cycle - df_engine["time_in_cycles"]

ax2.plot(df_engine["time_in_cycles"], df_engine["RUL_true_approx"],
         label="True RUL", linewidth=2.5, color=COLORS['success'])
ax2.plot(df_online["cycle"], df_online["rul_pred"],
         label="Predicted RUL", linewidth=2, linestyle="--", color=COLORS['secondary'])
ax2.fill_between(df_engine["time_in_cycles"], df_engine["RUL_true_approx"], 
                  alpha=0.2, color=COLORS['success'])
ax2.set_xlabel("Time in Cycles")
ax2.set_ylabel("RUL (cycles)")
ax2.set_title(f"Online RUL Prediction — Engine {example_unit}", fontsize=12, fontweight='bold')
ax2.legend(loc='upper right')

# Panel 3: Lifespan Distribution (bottom-left)
ax3 = fig.add_subplot(2, 2, 3)
ax3.hist(lifespan_train.values, bins=20, color=COLORS['info'], edgecolor='black', alpha=0.7)
ax3.axvline(lifespan_train.mean(), color=COLORS['danger'], linestyle='--', linewidth=2, label=f'Mean: {lifespan_train.mean():.0f}')
ax3.axvline(lifespan_train.median(), color=COLORS['warning'], linestyle='-.', linewidth=2, label=f'Median: {lifespan_train.median():.0f}')
ax3.set_xlabel("Lifespan (cycles)")
ax3.set_ylabel("Frequency")
ax3.set_title("Engine Lifespan Distribution (Training Set)", fontsize=12, fontweight='bold')
ax3.legend()

# Panel 4: Predicted vs Actual RUL Scatter (bottom-right)
ax4 = fig.add_subplot(2, 2, 4)
# Get predictions from best model for scatter
y_test_pred_life = best_model_for_online.predict(X_test)
y_test_pred_rul = y_test_pred_life - last_cycle_test
ax4.scatter(y_RUL_truth, y_test_pred_rul, alpha=0.6, c=COLORS['primary'], edgecolors='black', s=60)
# Perfect prediction line
max_val = max(max(y_RUL_truth), max(y_test_pred_rul))
ax4.plot([0, max_val], [0, max_val], 'r--', linewidth=2, label='Perfect Prediction')
ax4.set_xlabel("Actual RUL")
ax4.set_ylabel("Predicted RUL")
ax4.set_title("Predicted vs Actual RUL (Test Set)", fontsize=12, fontweight='bold')
ax4.legend()

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'rul_dashboard.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.show()

# --- FIGURE 2: Multiple Engine RUL Curves ---
fig2, axes = plt.subplots(2, 3, figsize=(15, 8))
fig2.suptitle('Online RUL Predictions — Sample Engines', fontsize=14, fontweight='bold')
sample_engines = [1, 10, 25, 50, 75, 100]

for ax, unit in zip(axes.flatten(), sample_engines):
    df_online_eng = online_preds[unit]
    info = test_last[test_last["unit_number"] == unit].iloc[0]
    fail_cycle = info["last_cycle"] + info["RUL_truth"]
    
    df_eng = test_df[test_df["unit_number"] == unit].sort_values("time_in_cycles").copy()
    df_eng["RUL_true"] = fail_cycle - df_eng["time_in_cycles"]
    
    ax.plot(df_eng["time_in_cycles"], df_eng["RUL_true"], 
            label="True", linewidth=2, color=COLORS['success'])
    ax.plot(df_online_eng["cycle"], df_online_eng["rul_pred"],
            label="Predicted", linestyle="--", linewidth=1.5, color=COLORS['secondary'])
    ax.set_title(f"Engine {unit}", fontsize=10, fontweight='bold')
    ax.set_xlabel("Cycle", fontsize=8)
    ax.set_ylabel("RUL", fontsize=8)
    ax.legend(fontsize=7, loc='upper right')
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'rul_multiple_engines.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.show()

# ---------------------------------------------------------
# STEP 12 — PRETTY CONSOLE OUTPUT SUMMARY
# ---------------------------------------------------------
def print_box(title, content_lines, width=60):
    """Print a nice boxed section in the terminal."""
    print("\n" + "╔" + "═" * width + "╗")
    print("║" + title.center(width) + "║")
    print("╠" + "═" * width + "╣")
    for line in content_lines:
        print("║ " + line.ljust(width - 1) + "║")
    print("╚" + "═" * width + "╝")

# Summary box
summary_lines = [
    f"Best Model: {best_name}",
    f"",
    f"Lifespan RMSE (validation): {best['lifespan_rmse_val']:.2f} cycles",
    f"Lifespan MAE  (validation): {best['lifespan_mae_val']:.2f} cycles",
    f"",
    f"RUL RMSE (test): {best['rul_rmse_test']:.2f} cycles",
    f"RUL MAE  (test): {best['rul_mae_test']:.2f} cycles",
    f"",
    f"Models compared: {len(cmp)}",
    f"Features used: {len(feature_cols)}",
    f"Training engines: {len(all_units)}",
]
print_box("🏆 FINAL RESULTS", summary_lines)

# Rankings box
ranking_lines = []
medals = ["🥇", "🥈", "🥉", "4.", "5.", "6."]
for i, row in cmp.iterrows():
    medal = medals[i] if i < len(medals) else f"{i+1}."
    ranking_lines.append(f"{medal} {row['model']:18} RUL RMSE: {row['rul_rmse_test']:>7.2f}")
print_box("📊 MODEL RANKINGS (by RUL RMSE)", ranking_lines)

# Saved files
files_lines = [
    f"Location: {OUTPUT_DIR}",
    "",
    "📁 rul_dashboard.png        - Main dashboard with 4 panels",
    "📁 rul_multiple_engines.png - RUL curves for 6 sample engines",
]
print_box("💾 SAVED FILES", files_lines)

print("\n✅ PIPELINE COMPLETE!\n")

