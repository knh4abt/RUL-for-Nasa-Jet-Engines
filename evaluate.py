"""
Standalone evaluation script for RUL models.
Loads data, trains best model, and prints detailed metrics.
"""
import pandas as pd
import numpy as np

from data import load_cmapss, drop_zero_variance, prepare_features
from training import Trainer
from utils import rmse, mae


def main():
    # Load and prepare
    train_df, test_df, rul_df = load_cmapss()
    train_df, test_df, _ = drop_zero_variance(train_df, test_df)
    feat_data = prepare_features(train_df, test_df, rul_df)
    
    # Train
    trainer = Trainer(
        train_feat=feat_data['train_feat'],
        test_feat=feat_data['test_feat'],
        feature_cols=feat_data['feature_cols'],
        sensor_cols=feat_data['sensor_cols'],
        agg_sources=feat_data['agg_sources'],
    )
    
    results_df = trainer.train_and_compare()
    best_model = trainer.fit_best()
    
    # Final evaluation on test set
    X_test = feat_data['test_feat'][feat_data['feature_cols']].values
    last_cycles = feat_data['test_feat']['last_cycle'].values
    y_rul_truth = feat_data['test_feat']['RUL_truth'].values
    
    y_pred_life = best_model.predict(X_test)
    y_pred_rul = y_pred_life - last_cycles
    
    print("\n" + "="*50)
    print(f"Final Evaluation: {trainer.best_name}")
    print("="*50)
    print(f"RUL RMSE: {rmse(y_rul_truth, y_pred_rul):.2f} cycles")
    print(f"RUL MAE:  {mae(y_rul_truth, y_pred_rul):.2f} cycles")
    
    # Per-engine breakdown
    errors = y_pred_rul - y_rul_truth
    print(f"\nPrediction error stats:")
    print(f"  Mean:   {errors.mean():+.1f} cycles")
    print(f"  Std:    {errors.std():.1f} cycles")
    print(f"  Min:    {errors.min():+.1f} cycles")
    print(f"  Max:    {errors.max():+.1f} cycles")
    
    # Save results
    results_df.to_csv("results/model_comparison.csv", index=False)
    print("\nSaved results to results/model_comparison.csv")


if __name__ == "__main__":
    main()
