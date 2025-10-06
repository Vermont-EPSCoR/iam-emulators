#!/usr/bin/env python3
"""
Train a LightGBM Regressor (GPU if available), evaluate on test set,
and save the trained model + metrics JSON.

"""

import os
import json
import pickle
import time
import numpy as np
import pandas as pd
from pathlib import Path

from lightgbm import LGBMRegressor, early_stopping, log_evaluation
from lightgbm.basic import LightGBMError

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    median_absolute_error,
    explained_variance_score,
    root_mean_squared_error
)


# set the paths and load the data
root_path = Path('Project Directory Path')
train_path = root_path / "comboCSV_train_regression.csv"
test_path  = root_path / "comboCSV_test_regression.csv"
outdir = root_path / "LightGBM"/ "Regression" / "Weights"
outdir.mkdir(parents=True, exist_ok=True)


#  few helper functions
try:
    from sklearn.metrics import root_mean_squared_error
    def RMSE(y_true, y_pred): return root_mean_squared_error(y_true, y_pred)
except Exception:
    from sklearn.metrics import mean_squared_error
    def RMSE(y_true, y_pred): return mean_squared_error(y_true, y_pred, squared=False)
def safe_mape(y_true, y_pred, eps=1e-8):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    mask = np.abs(y_true) > eps
    if mask.sum() == 0:
        return float("nan")
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100.0)

def nse(y_true, y_pred):
    y_true = np.asarray(y_true)
    return float(1.0 - np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2))



df_train = pd.read_csv(train_path, low_memory=False)
df_test  = pd.read_csv(test_path, low_memory=False)

target_variable = "CHLAVESurfaceMean"
X_train, Y_train = df_train.drop(columns=[target_variable]), df_train[target_variable]
X_test,  Y_test  = df_test.drop(columns=[target_variable]),  df_test[target_variable]

# getting the validation split
x_train, x_val, y_train, y_val = train_test_split(
    X_train, Y_train, test_size=0.10, random_state=14912
)


# build the regressor
common_params = dict(
    objective="regression",   # L2 by default
    learning_rate=0.05,
    n_estimators=50000,       # upper bound - early stopping will trim
    max_depth=-1,
    subsample=0.8,
    colsample_bytree=0.8,
    n_jobs=-1,
    random_state=14912,
)

print("Initializing LightGBM Regressor  - We will try to use GPU")
lgbm_reg = LGBMRegressor(device_type="gpu", **common_params)

# lets fit the model
def fit_model(estimator):
    start = time.time()
    estimator.fit(
        x_train, y_train,
        eval_set=[(x_val, y_val)],
        eval_metric="rmse",
        callbacks=[early_stopping(stopping_rounds=200, verbose=True),
                   log_evaluation(period=500)]
    )
    train_time = time.time() - start
    return train_time

try:
    train_time = fit_model(lgbm_reg)
except LightGBMError as e:
    print(f"GPU training failed ({e}). Falling back to CPU...")
    lgbm_reg = LGBMRegressor(device_type="cpu", **common_params)
    train_time = fit_model(lgbm_reg)

print(f"Training finished in {train_time:.2f} seconds.")
best_iter = getattr(lgbm_reg, "best_iteration_", None)

# now lets evaluate the model and save weights and metrices
print("\nEvaluating on test set")
y_pred = lgbm_reg.predict(X_test, num_iteration=best_iter)

mae   = mean_absolute_error(Y_test, y_pred)
rmse  = RMSE(Y_test, y_pred)
medae = median_absolute_error(Y_test, y_pred)
r2    = r2_score(Y_test, y_pred)
evs   = explained_variance_score(Y_test, y_pred)
bias  = float(np.mean(y_pred - Y_test))   # Mean Error
mape  = safe_mape(Y_test, y_pred)
rho   = float(np.corrcoef(Y_test, y_pred)[0, 1])
nse_  = nse(Y_test, y_pred)

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
    "TrainingTime_sec": round(train_time, 2),
    "BestIteration": int(best_iter) if best_iter is not None else None,
    "Device": lgbm_reg.get_params().get("device_type", "unknown"),
}

print("\nModel Evaluation Metrics:")
for k, v in results.items():
    print(f"{k:20s}: {v}")


metrics_path = outdir / "metrics.json"
with open(metrics_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nMetrics saved to: {metrics_path}")

model_path = outdir / "lgbm_model_regression.pkl"
with open(model_path, "wb") as f:
    pickle.dump(lgbm_reg, f)
print(f"Model saved to: {model_path}")

print(f"\nSaved model and metrics to: {outdir}")
