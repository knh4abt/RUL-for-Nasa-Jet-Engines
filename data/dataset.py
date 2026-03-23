"""
Data loading and feature engineering for NASA C-MAPSS dataset.
"""
import numpy as np
import pandas as pd

import config


def load_cmapss(data_dir=None):
    """Load train, test, and RUL files from C-MAPSS FD001."""
    data_dir = data_dir or config.DATA_DIR
    
    read_opts = dict(sep=r"\s+", header=None, engine="python")
    
    train_df = pd.read_csv(data_dir / "train_FD001.txt", **read_opts)
    train_df = train_df.dropna(axis=1, how="all")
    
    test_df = pd.read_csv(data_dir / "test_FD001.txt", **read_opts)
    test_df = test_df.dropna(axis=1, how="all")
    
    rul_df = pd.read_csv(data_dir / "RUL_FD001.txt", header=None, names=["RUL"])
    
    train_df.columns = config.COLUMNS
    test_df.columns = config.COLUMNS
    
    return train_df, test_df, rul_df


def drop_zero_variance(train_df, test_df):
    """Remove columns with zero variance in training set."""
    zero_var_cols = []
    for col in train_df.columns:
        if col not in ["unit_number", "time_in_cycles"]:
            if train_df[col].std() == 0:
                zero_var_cols.append(col)
    
    train_df = train_df.drop(columns=zero_var_cols)
    test_df = test_df.drop(columns=zero_var_cols)
    
    return train_df, test_df, zero_var_cols


def _is_informative(df, col, tol=1e-8):
    """Check if column varies across engines."""
    eng_means = df.groupby('unit_number')[col].mean()
    return eng_means.var() > tol


def build_agg_features(df, cols_to_agg, aggs=None):
    """
    Compute per-engine aggregated features.
    Returns DataFrame with unit_number and aggregated columns.
    """
    aggs = aggs or ['mean', 'std', 'min', 'max']
    grouped = df.groupby('unit_number')[cols_to_agg].agg(aggs)
    grouped.columns = [f"{c}_{a}" for c, a in grouped.columns]
    return grouped.reset_index()


def add_trend_features(df, sensor_cols):
    """
    Compute trend features per engine: slope, range, delta.
    """
    rows = []
    for unit, grp in df.groupby('unit_number'):
        row = {'unit_number': unit}
        total_cycles = grp['time_in_cycles'].max()
        
        for col in sensor_cols:
            first_val = grp[col].iloc[0]
            last_val = grp[col].iloc[-1]
            mean_val = grp[col].mean()
            
            row[f"{col}_slope"] = (last_val - first_val) / total_cycles
            row[f"{col}_range"] = grp[col].max() - grp[col].min()
            row[f"{col}_delta"] = last_val - mean_val
        
        rows.append(row)
    
    return pd.DataFrame(rows)


def prepare_features(train_df, test_df, rul_df):
    """
    Full feature engineering pipeline.
    Returns feature matrices and metadata needed for training.
    """
    # Get sensor columns
    sensor_cols = [c for c in train_df.columns if c.startswith('sensor_')]
    
    # Check which settings are informative
    raw_settings = ['op_setting_1', 'op_setting_2', 'op_setting_3']
    setting_cols = [c for c in raw_settings if c in train_df.columns]
    informative_settings = [c for c in setting_cols if _is_informative(train_df, c)]
    
    agg_sources = sensor_cols + informative_settings
    aggs = ['mean', 'std', 'min', 'max']
    
    # Build features
    agg_train = build_agg_features(train_df, agg_sources, aggs)
    agg_test = build_agg_features(test_df, agg_sources, aggs)
    
    trend_train = add_trend_features(train_df, sensor_cols)
    trend_test = add_trend_features(test_df, sensor_cols)
    
    # Merge
    train_feat = pd.merge(agg_train, trend_train, on='unit_number', how='left')
    test_feat = pd.merge(agg_test, trend_test, on='unit_number', how='left')
    
    # Lifespan labels for training
    lifespan = train_df.groupby('unit_number')['time_in_cycles'].max().rename('lifespan')
    train_feat = train_feat.merge(lifespan.reset_index(), on='unit_number', how='left')
    
    # Test labels
    test_last = (
        test_df.sort_values(['unit_number', 'time_in_cycles'])
        .groupby('unit_number').tail(1)
        .reset_index(drop=True)[['unit_number', 'time_in_cycles']]
        .rename(columns={'time_in_cycles': 'last_cycle'})
    )
    rul_map = pd.Series(rul_df['RUL'].values, index=np.arange(1, len(rul_df) + 1))
    test_last['RUL_truth'] = test_last['unit_number'].map(rul_map)
    test_feat = test_feat.merge(test_last, on='unit_number', how='left')
    
    # Feature column names
    exclude = {'unit_number', 'lifespan', 'last_cycle', 'RUL_truth'}
    feature_cols = [c for c in train_feat.columns if c not in exclude]
    
    return {
        'train_feat': train_feat,
        'test_feat': test_feat,
        'feature_cols': feature_cols,
        'sensor_cols': sensor_cols,
        'agg_sources': agg_sources,
    }
