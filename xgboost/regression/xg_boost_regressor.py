#!/usr/bin/env python3
"""
Train an XGBoost Regressor with validation, evaluate it on the test set,
save the trained model and metrics (JSON). Uses GPU if available.

"""

import os
import json
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    median_absolute_error,
    explained_variance_score,
    root_mean_squared_error
)

# a few helper functions
def safe_mape(y_true, y_pred, eps=1e-8):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    mask = np.abs(y_true) > eps
   
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100.0

def nse(y_true, y_pred):
    y_true = np.asarray(y_true)
    return 1.0 - np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2)


# lets set the paths and load the data
root_path = Path('Project Directory Path.')
train_path = root_path / "comboCSV_train_regression.csv"
test_path = root_path / "comboCSV_test_regression.csv"

df_train = pd.read_csv(train_path, low_memory=False)
df_test = pd.read_csv(test_path, low_memory=False)

target_variable = "CHLAVESurfaceMean"
X_train, Y_train = df_train.drop(target_variable, axis=1), df_train[target_variable]
X_test, Y_test = df_test.drop(target_variable, axis=1), df_test[target_variable]

# lets also get the validation set for this one
x_train, x_val, y_train, y_val = train_test_split(
    X_train, Y_train, test_size=0.10, random_state=14912
)

# n_jobs = int(os.environ.get("SLURM_CPUS_PER_TASK", 1)) # in case if the script is running on VACC - However, no need to use it, set n_jobs=-1 and it will use all the cpus


# building the regressor
xgb_reg = xgb.XGBRegressor(
    objective="reg:squarederror",
    learning_rate=0.05,
    n_estimators=50000,        # high upper bound, early stopping will cut - This is important, noticed that loss continuously decreasing
    max_depth=8,
    subsample=0.8,
    colsample_bytree=0.8,
    tree_method="hist",    # use GPU if available
    device="cuda",
    n_jobs=-1,
    eval_metric="rmse",
    early_stopping_rounds=200,
    random_state=14912,
)

# now lets fit the model
print("Training XGBoost Regressor - This may take a while...")
xgb_reg.fit(
    x_train, y_train,
    eval_set=[(x_val, y_val)],
    verbose=500
)

# evaluating the model and save the weights and results
print("\nEvaluating best iteration on test set...")
y_pred = xgb_reg.predict(X_test)

mae   = mean_absolute_error(Y_test, y_pred)
rmse  = root_mean_squared_error(Y_test, y_pred)
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
    "BestIteration": int(xgb_reg.best_iteration)
}


print("\nModel Evaluation Metrics:")
for k, v in results.items():
    print(f"{k:20s}: {v}")


timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
out_dir = root_path / "XGboost" / "Regression" / "Weights"
out_dir.mkdir(parents=True, exist_ok=True)


json_path = out_dir / f"xgb_metrics_{timestamp}.json"
with open(json_path, "w") as f:
    json.dump(results, f, indent=4)
print(f"\nMetrics saved to {json_path}")


model_path = out_dir / f"xgb_model_regression.pkl"
with open(model_path, "wb") as f:
    pickle.dump(xgb_reg, f)
print(f"Model saved to {model_path}")
