#!/usr/bin/env python3
"""
Script to Generate XGBoost Feature Importances

"""

import os
import pickle
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
)


# setting paths
root_path = Path('Project Directory Path')
xgboost_weights = root_path / "XGBoost" / "Classification" / "xgb_model_classification.pkl"
plots_dir = root_path / "XGBoost" / "Classification" / "Plots"
plots_dir.mkdir(parents=True, exist_ok=True)

train_path = root_path / "comboCSV_train.csv"
test_path = root_path / "comboCSV_test.csv"

# loading data and setting it for training and evaluation
train_data = pd.read_csv(train_path, low_memory=False)
test_data = pd.read_csv(test_path, low_memory=False)

target_variable = "CHLAClass"
X_train = train_data.drop(columns=[target_variable])
y_train = train_data[target_variable]
X_test  = test_data.drop(columns=[target_variable])
y_test  = test_data[target_variable]

# loading the weights
with open(xgboost_weights, "rb") as f:
    xgb_clf = pickle.load(f)

# lets test the model
y_pred = xgb_clf.predict(X_test)
print("Test accuracy:", (y_pred == y_test).mean())



imp_mean = xgb_clf.feature_importances_

imp_df = (pd.DataFrame({"mean": imp_mean}, index=X_test.columns)
            .sort_values("mean", ascending=True))

fig, ax = plt.subplots(figsize=(12, max(6, len(imp_df) * 0.25)), dpi=300) 
ax.barh(
    imp_df.index,
    imp_df["mean"],
    align="center",
)

ax.set_title("XGBoost Feature Importances", fontsize=18, fontweight="bold", pad=12)
ax.set_xlabel("Feature Importance", fontsize=14, fontweight="bold", labelpad=12)
ax.set_ylabel("Features", fontsize=16, fontweight="bold", labelpad=10)

tick_size = 12
for t in ax.get_xticklabels():
    t.set_fontsize(tick_size)
    t.set_fontweight("bold")
for t in ax.get_yticklabels():
    t.set_fontsize(tick_size)
    t.set_fontweight("bold")

fig.tight_layout()

out_path = plots_dir / "XGBoost Feature Importances Classification.png"
fig.savefig(out_path, dpi=600)

plt.close()
print(f"XGBoost Classification Feature Importance is Created and Saved at {out_path}")

