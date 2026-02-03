# 🛩️ Jet Engine Predictive Maintenance — RUL Prediction

End-to-end **Remaining Useful Life (RUL)** prediction for jet engines using the NASA C-MAPSS FD001 dataset. This project demonstrates a complete machine learning pipeline with both traditional ML models and deep learning (LSTM).

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-green.svg)

---

## 📋 Project Overview

**Goal:** Predict how many operational cycles a jet engine has left before failure.

**Strategy:** Instead of predicting RUL directly, we predict **total lifespan** (max cycles until failure), then derive RUL as:
```
RUL = predicted_lifespan - current_cycle
```

This approach avoids data leakage and provides more stable predictions.

---

## 🏗️ Architecture

```
Raw Data (train/test/RUL .txt files)
    ↓
Per-Engine Feature Engineering (aggregations + trends)
    ↓
Model Training (ML + LSTM)
    ↓
Lifespan Prediction → RUL Conversion
    ↓
Online RUL Curves + Dashboard Visualization
```

---

## 📊 Models Compared

| Model | RUL RMSE (Test) | Notes |
|-------|-----------------|-------|
| 🥇 **RandomForest** | ~46 cycles | Best performer |
| 🥈 GradientBoosting | ~49 cycles | Strong alternative |
| 🥉 LSTM | ~73 cycles | Deep learning baseline |
| Ridge | ~82 cycles | Regularized linear |
| LinearRegression | ~186 cycles | Simple baseline |
| Lasso | ~185 cycles | Sparse features |

---

## 🔧 Features

### Feature Engineering
- **Aggregated features:** mean, std, min, max of all sensors
- **Trend features:** slope, range, delta from mean
- **137 total features** per engine

### Models
- **Traditional ML:** LinearRegression, Lasso, Ridge, RandomForest, GradientBoosting
- **Deep Learning:** LSTM neural network (sequence-based)

### Visualizations
- 4-panel dashboard (model comparison, RUL curves, distributions, scatter)
- Multi-engine RUL prediction curves
- Pretty console output with rankings

---

## 🚀 Quick Start

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/RUL-for-Nasa-Jet-Engines.git
cd RUL-for-Nasa-Jet-Engines
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the pipeline
```bash
python main.py
```

### 4. Check outputs
- Console: Model rankings and metrics
- `outputs/`: Saved visualization plots

---

## 📁 Project Structure

```
├── main.py                 # Main pipeline script
├── requirements.txt        # Python dependencies
├── README.md              # This file
├── outputs/               # Generated plots (auto-created)
│   ├── rul_dashboard.png
│   └── rul_multiple_engines.png
└── local_assets/          # Data files
    ├── train_FD001.txt
    ├── test_FD001.txt
    └── RUL_FD001.txt
```

---

## 📈 Sample Output

### Model Comparison
```
╔════════════════════════════════════════════════════════════╗
║              📊 MODEL RANKINGS (by RUL RMSE)               ║
╠════════════════════════════════════════════════════════════╣
║ 🥇 RandomForest       RUL RMSE:   45.97                   ║
║ 🥈 GradientBoosting   RUL RMSE:   49.05                   ║
║ 🥉 LSTM               RUL RMSE:   72.69                   ║
║ 4. Ridge              RUL RMSE:   81.56                   ║
║ 5. Lasso              RUL RMSE:  185.23                   ║
║ 6. LinearRegression   RUL RMSE:  185.88                   ║
╚════════════════════════════════════════════════════════════╝
```

---

## 📚 Dataset

**NASA C-MAPSS FD001:**
- 100 training engines (run to failure)
- 100 test engines (truncated before failure)
- Single operating condition (sea level)
- Single fault mode (HPC degradation)
- 26 columns: unit_number, time_in_cycles, 3 operational settings, 21 sensors

---

## 🧠 Key Learnings

1. **Traditional ML can outperform Deep Learning** on small datasets (100 engines)
2. **Feature engineering matters** — aggregated + trend features capture degradation patterns
3. **Lifespan prediction** is more stable than direct RUL prediction
4. **LSTM requires more data** to show its full potential

---

## 🛠️ Technologies

- **Python 3.8+**
- **pandas, numpy** — Data manipulation
- **scikit-learn** — ML models and preprocessing
- **TensorFlow/Keras** — LSTM neural network
- **matplotlib** — Visualization

---

## 📄 License

MIT License — Feel free to use and modify.

---

## 👤 Author

Built as a Data Science portfolio project demonstrating:
- End-to-end ML pipeline development
- Feature engineering for time-series data
- Model comparison and evaluation
- Deep learning integration
- Professional code structure and visualization
