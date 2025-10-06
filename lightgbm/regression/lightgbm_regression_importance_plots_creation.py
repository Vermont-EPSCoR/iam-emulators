
# loading libararies
import os, joblib, json, pickle
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# setting paths
root_path = Path('Project Directory Path')
lgbm_weights = root_path / 'LightGBM' / 'Regression'/ 'Weights' / 'lightgbm_regression.pkl'
plots_dir    = root_path / "LightGBM" / "Regression" / "Plots"
plots_dir.mkdir(parents=True, exist_ok=True)

test_csv_path = root_path / 'comboCSV_test_regression.csv'
train_csv_path = root_path / "comboCSV_train_regression.csv"

# loading data
train_data = pd.read_csv(train_csv_path, low_memory=False)
test_data = pd.read_csv(test_csv_path, low_memory=False)

target_variable = "CHLAVESurfaceMean"
X_train = train_data.drop(columns=[target_variable])
y_train = train_data[target_variable]
X_test  = test_data.drop(columns=[target_variable])
y_test  = test_data[target_variable]

with open(lgbm_weights, "rb") as f:
    lgbm_clf = pickle.load(f)

print("Loaded LightGBM model:", type(lgbm_clf))

y_pred = lgbm_clf.predict(X_test)

# lets first get the feature importances
booster = getattr(lgbm_clf, "booster_", lgbm_clf)
feat_names = booster.feature_name()

imp_gain = booster.feature_importance(importance_type='gain').astype(float)
imp_gain = imp_gain / (imp_gain.sum() + 1e-12)   # normalize like XGB

imp_df = (pd.DataFrame({'feature': feat_names, 'gain': imp_gain})
            .set_index('feature')
            .reindex(X_test.columns)    
            .fillna(0.0).sort_values("gain", ascending=True))


print(imp_df)

# lets initialize the plots - also initialize the outpath
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
out_path = plots_dir / "LightGBM Feature Importances Regression.png"
fig.savefig(out_path, dpi=300)
plt.close()

print(f"Visualization is created and saved at {out_path}")


