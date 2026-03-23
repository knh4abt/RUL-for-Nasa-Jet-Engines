"""
Training orchestration for RUL models.
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

import config
from models import get_models
from utils import evaluate_model
from data import build_agg_features, add_trend_features


class Trainer:
    def __init__(self, train_feat, test_feat, feature_cols, 
                 sensor_cols, agg_sources):
        self.train_feat = train_feat
        self.test_feat = test_feat
        self.feature_cols = feature_cols
        self.sensor_cols = sensor_cols
        self.agg_sources = agg_sources
        
        self.models = get_models()
        self.results = []
        self.best_model = None
        self.best_name = None
    
    def _get_xy(self, df, units, target):
        """Extract X, y arrays for given engine units."""
        sub = df[df['unit_number'].isin(units)]
        X = sub[self.feature_cols].values
        y = sub[target].values
        return X, y
    
    def train_and_compare(self):
        """Train all models, compare performance, return results DataFrame."""
        all_units = self.train_feat['unit_number'].values
        train_units, val_units = train_test_split(
            all_units, 
            test_size=config.VAL_SIZE, 
            random_state=config.RANDOM_STATE
        )
        
        X_train, y_train = self._get_xy(self.train_feat, train_units, 'lifespan')
        X_val, y_val = self._get_xy(self.train_feat, val_units, 'lifespan')
        
        X_test = self.test_feat[self.feature_cols].values
        y_rul_truth = self.test_feat['RUL_truth'].values
        last_cycles = self.test_feat['last_cycle'].values
        
        self.results = []
        for name, model in self.models.items():
            metrics = evaluate_model(
                name, model,
                X_train, y_train,
                X_val, y_val,
                X_test, last_cycles, y_rul_truth
            )
            self.results.append(metrics)
            print(f"{name}: lifespan_RMSE={metrics['lifespan_rmse_val']:.2f} | "
                  f"RUL_RMSE={metrics['rul_rmse_test']:.2f}")
        
        results_df = pd.DataFrame(self.results).sort_values('rul_rmse_test')
        self.best_name = results_df.iloc[0]['model']
        
        return results_df
    
    def fit_best(self):
        """Refit best model on all training data."""
        if self.best_name is None:
            raise ValueError("Run train_and_compare() first")
        
        all_units = self.train_feat['unit_number'].values
        X_full, y_full = self._get_xy(self.train_feat, all_units, 'lifespan')
        
        self.best_model = get_models()[self.best_name]
        self.best_model.fit(X_full, y_full)
        
        print(f"\nFitted {self.best_name} on all {len(all_units)} training engines.")
        return self.best_model
    
    def online_predictions(self, test_df):
        """
        Generate RUL predictions at each time step for all test engines.
        Returns dict: unit_number -> DataFrame(cycle, rul_pred)
        """
        if self.best_model is None:
            raise ValueError("Run fit_best() first")
        
        aggs = ['mean', 'std', 'min', 'max']
        results = {}
        
        for unit, df_eng in test_df.groupby("unit_number"):
            df_eng = df_eng.sort_values("time_in_cycles")
            preds = []
            
            for _, row in df_eng.iterrows():
                current_cycle = row["time_in_cycles"]
                partial = df_eng[df_eng["time_in_cycles"] <= current_cycle]
                
                # Build features from partial data
                agg_part = build_agg_features(partial, self.agg_sources, aggs)
                trend_part = add_trend_features(partial, self.sensor_cols)
                feat_df = pd.merge(agg_part, trend_part, on="unit_number", how="left")
                
                X = feat_df[self.feature_cols].values.reshape(1, -1)
                life_pred = self.best_model.predict(X)[0]
                rul_pred = life_pred - current_cycle
                
                preds.append({"cycle": current_cycle, "rul_pred": rul_pred})
            
            results[unit] = pd.DataFrame(preds)
        
        return results
