#!/usr/bin/env python3
"""
Script to Generate LightGBM Feature Importances

"""

import os
import pickle
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd

# setting paths
root_path = Path('Project Directory Path')
lgbm_weights = root_path / 'LightGBM' / 'Classification' / f'lgbm_model.pkl'
plots_dir = root_path / "LightGBM" / "Classification" / "Plots"
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

with open(lgbm_weights, "rb") as f:
    lgbm_clf = pickle.load(f)

# lets test the model
y_pred = lgbm_clf.predict(X_test)
print("Test accuracy:", (y_pred == y_test).mean())

booster = getattr(lgbm_clf, "booster_", lgbm_clf)
feat_names = booster.feature_name()

imp_gain = booster.feature_importance(importance_type='gain').astype(float)
imp_gain = imp_gain / (imp_gain.sum() + 1e-12)   # normalize like XGB

imp_df = (pd.DataFrame({'feature': feat_names, 'gain': imp_gain})
            .set_index('feature')
            .reindex(X_test.columns)     
            .fillna(0.0).sort_values("gain", ascending=True)
            )


fig, ax = plt.subplots(figsize=(12, max(6, len(imp_df) * 0.25)), dpi=600) 
ax.barh(
    imp_df.index, 
    imp_df['gain'], 
    align='center'
    )


# ax.invert_yaxis()

ax.set_title("LightGBM Feature Importances", fontsize=18, fontweight="bold", pad=12)
ax.set_xlabel("Feature Importance", fontsize=14, fontweight="bold", labelpad=12)
ax.set_ylabel("Features", fontsize=16, fontweight="bold", labelpad=10)

tick_size = 12
for t in ax.get_xticklabels() + ax.get_yticklabels():
    t.set_fontsize(tick_size)
    t.set_fontweight("bold")

fig.tight_layout()
out_path = plots_dir / "LightGBM Feature Importances Classification.png"
fig.savefig(out_path, dpi=600)
plt.close()

print(f"LightBGM Classification Feature Importance is Created and Saved at {out_path}")

