#!/usr/bin/env python3
"""
Random Forest Regressor Training + Evaluation Script
"""

# require dependencies
import os
import json
import time
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    root_mean_squared_error,
    median_absolute_error,
    explained_variance_score
)


# data directory paths - root path (project directory) + train and test CSVs paths
root_path = Path('Project Directory Path')
train_path = root_path / "comboCSV_train_regression.csv"
test_path  = root_path / "comboCSV_test_regression.csv"

# a few helper functions 
def safe_mape(y_true, y_pred, eps=1e-8):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    mask = np.abs(y_true) > eps
    if mask.sum() == 0:
        return np.nan
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100.0

def nse(y_true, y_pred):
    y_true = np.asarray(y_true)
    return 1.0 - np.sum((y_true - y_pred)**2) / np.sum((y_true - np.mean(y_true))**2)


# load the dataset
print("Loading training and testing data...")
df_train = pd.read_csv(train_path, low_memory=False)
df_test  = pd.read_csv(test_path, low_memory=False)

target_variable = "CHLAVESurfaceMean"
x_train, y_train = df_train.drop(columns=[target_variable]), df_train[target_variable]
x_test,  y_test  = df_test.drop(columns=[target_variable]),  df_test[target_variable]

print(f"Train size: {x_train.shape}, Test size: {x_test.shape}")


# fit the model
print("Fitting RandomForestRegressor...")
rf = RandomForestRegressor(
    n_estimators=100,
    n_jobs=-1,
    verbose=1,
    random_state=14912
)

start_time = time.time()
rf.fit(x_train, y_train)
train_time = time.time() - start_time
print(f"Training finished in {train_time:.2f} seconds.")

# performance evaluations
print("Generating predictions on test set...")
y_pred = rf.predict(x_test)


mae   = mean_absolute_error(y_test, y_pred)
rmse = root_mean_squared_error(y_test, y_pred)
medae = median_absolute_error(y_test, y_pred)
r2    = r2_score(y_test, y_pred)
evs   = explained_variance_score(y_test, y_pred)
bias  = float(np.mean(y_pred - y_test))   
mape  = safe_mape(y_test, y_pred)
rho   = float(np.corrcoef(y_test, y_pred)[0, 1])
nse_  = nse(y_test, y_pred)

results = {
    "MAE": round(mae, 6),
    "RMSE": round(rmse, 6),
    "Median_AE": round(medae, 6),
    "Bias": round(bias, 6),
    "MAPE": round(mape, 6),
    "R2": round(r2, 6),
    "ExplainedVariance": round(evs, 6),
    "Pearson_r": round(rho, 6),
    "NSE": round(nse_, 6),
    "TrainingTime_sec": round(train_time, 2)
}


print("\nModel Evaluation Metrics:")
for k, v in results.items():
    print(f"{k:20s}: {v}")

# save the model and results in json file
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
out_dir = root_path / "Random_Forest/Regression/Weights"
out_dir.mkdir(parents=True, exist_ok=True)

json_path = out_dir / f"rf_metrics_{timestamp}.json"
with open(json_path, "w") as f:
    json.dump(results, f, indent=4)
print(f"\nMetrics saved to {json_path}")

model_path = out_dir / f"rf_regression.pkl"
joblib.dump(rf, model_path)
print(f"Model saved to {model_path}")
