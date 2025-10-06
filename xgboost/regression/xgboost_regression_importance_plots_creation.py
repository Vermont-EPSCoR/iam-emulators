
# loading libararies
import os, joblib, json, pickle
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math


# setting paths
root_path = Path('Project Directory Path')
xgboost_weights = root_path / "XGboost" / "Regression" / "Weights" / "xgb_model_regression.pkl"
plots_dir    = root_path / "XGBoost"  / "Regression" / "Plots"
test_csv_path = root_path / 'comboCSV_test_regression.csv'
train_csv_path = root_path / "comboCSV_train_regression.csv"
plots_dir.mkdir(parents=True, exist_ok=True)


# loading data
train_data = pd.read_csv(train_csv_path, low_memory=False)
test_data = pd.read_csv(test_csv_path, low_memory=False)

target_variable = "CHLAVESurfaceMean"
X_train = train_data.drop(columns=[target_variable])
y_train = train_data[target_variable]
X_test  = test_data.drop(columns=[target_variable])
y_test  = test_data[target_variable]

# laoding weights
print('Data has been loaded - Loading Weights')
try:
    # this works if model was saved with pickle or joblib
    xgb_clf = joblib.load(xgboost_weights)
except Exception:
    with open(xgboost_weights, "rb") as f:
        xgb_clf = pickle.load(f)
    
y_pred = xgb_clf.predict(X_test)

# visualization
print("Model is loaded and tested - Creating Visualization")

# lets first get the feature importances
imp_mean = xgb_clf.feature_importances_

imp_df = (pd.DataFrame({"mean": imp_mean}, index=X_test.columns)
            .sort_values("mean", ascending=True))

print(imp_df)

# lets initialize the plots - also initialize the outpath
fig, ax = plt.subplots(figsize=(12, max(6, len(imp_df) * 0.25)), dpi=500) 
ax.barh(
    imp_df.index,
    imp_df["mean"],
    align="center",
)

# titles and labels
ax.set_title("XGBoost Feature Importances", fontsize=18, fontweight="bold", pad=12)
ax.set_xlabel("Feature Importance", fontsize=14, fontweight="bold", labelpad=12)
ax.set_ylabel("Features", fontsize=16, fontweight="bold", labelpad=10)

# bold tick labels + sizes
tick_size = 12
for t in ax.get_xticklabels():
    t.set_fontsize(tick_size)
    t.set_fontweight("bold")
for t in ax.get_yticklabels():
    t.set_fontsize(tick_size)
    t.set_fontweight("bold")

fig.tight_layout()

out_path = plots_dir / "XGBoost Feature Importances Regression.png"
fig.savefig(out_path, dpi=600)
plt.close()
print(f"Visualization is created and saved at {out_path}")


