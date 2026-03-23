"""
RUL Prediction Pipeline for NASA C-MAPSS Jet Engines
"""
import numpy as np
import pandas as pd

from data import load_cmapss, drop_zero_variance, prepare_features
from training import Trainer
from utils import plot_rul_curve, plot_model_comparison


def main():
    print("Loading C-MAPSS FD001 data...")
    train_df, test_df, rul_df = load_cmapss()
    print(f"Loaded {train_df['unit_number'].nunique()} train / "
          f"{test_df['unit_number'].nunique()} test engines")
    
    # Preprocessing
    train_df, test_df, dropped = drop_zero_variance(train_df, test_df)
    if dropped:
        print(f"Dropped {len(dropped)} zero-variance columns")
    
    # Feature engineering
    print("\nEngineering features...")
    feat_data = prepare_features(train_df, test_df, rul_df)
    print(f"Created {len(feat_data['feature_cols'])} features per engine")
    
    # Training
    print("\n" + "="*50)
    print("Training models...")
    print("="*50)
    
    trainer = Trainer(
        train_feat=feat_data['train_feat'],
        test_feat=feat_data['test_feat'],
        feature_cols=feat_data['feature_cols'],
        sensor_cols=feat_data['sensor_cols'],
        agg_sources=feat_data['agg_sources'],
    )
    
    results_df = trainer.train_and_compare()
    
    print("\n" + "="*50)
    print("Model Comparison (sorted by RUL RMSE)")
    print("="*50)
    print(results_df.round(2).to_string(index=False))
    
    # Refit best model
    best_model = trainer.fit_best()
    
    # Online predictions
    print("\nGenerating online RUL predictions...")
    online_preds = trainer.online_predictions(test_df)
    print(f"Generated predictions for {len(online_preds)} test engines")
    
    # Visualization
    example_unit = 1
    df_online = online_preds[example_unit]
    
    test_last = feat_data['test_feat'][['unit_number', 'last_cycle', 'RUL_truth']]
    info = test_last[test_last['unit_number'] == example_unit].iloc[0]
    failure_cycle = info['last_cycle'] + info['RUL_truth']
    
    df_eng = test_df[test_df['unit_number'] == example_unit].sort_values('time_in_cycles')
    true_rul = failure_cycle - df_eng['time_in_cycles'].values
    
    plot_rul_curve(
        df_eng['time_in_cycles'].values, true_rul,
        df_online['cycle'].values, df_online['rul_pred'].values,
        engine_id=example_unit
    )
    
    plot_model_comparison(
        results_df['model'].values,
        results_df['rul_rmse_test'].values
    )


if __name__ == "__main__":
    main()

