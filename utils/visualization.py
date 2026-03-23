"""
Plotting utilities for RUL analysis.
"""
import matplotlib.pyplot as plt
from pathlib import Path


def plot_rul_curve(true_cycles, true_rul, pred_cycles, pred_rul, 
                   engine_id, save_path=None):
    """Plot predicted vs true RUL for a single engine."""
    plt.figure(figsize=(8, 5))
    plt.plot(true_cycles, true_rul, label="True RUL", linewidth=2)
    plt.plot(pred_cycles, pred_rul, label="Predicted RUL", linestyle="--")
    
    plt.xlabel("Time in cycles")
    plt.ylabel("RUL (cycles)")
    plt.title(f"RUL Prediction — Engine {engine_id}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()


def plot_model_comparison(model_names, rul_rmse_values, save_path=None):
    """Bar chart comparing model RUL RMSE."""
    plt.figure(figsize=(8, 5))
    bars = plt.bar(model_names, rul_rmse_values)
    
    plt.title("Model Comparison — RUL RMSE")
    plt.ylabel("RUL RMSE (lower is better)")
    plt.xlabel("Model")
    
    for bar, val in zip(bars, rul_rmse_values):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                 f"{val:.1f}", ha='center', va='bottom')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()
