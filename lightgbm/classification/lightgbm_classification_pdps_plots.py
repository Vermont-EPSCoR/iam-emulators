#!/usr/bin/env python3
"""
Script to Generate LightGBM PDPs

"""

import os
import pickle
from pathlib import Path
import matplotlib.pyplot as plt
import math
import numpy as np
import pandas as pd
from sklearn.inspection import PartialDependenceDisplay
from joblib import parallel_backend


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


subsample_n = 500
rng = np.random.default_rng(14912)
idx = rng.choice(X_train.index, size=min(subsample_n, len(X_train)), replace=False)
X_small = X_train.loc[idx].copy()

variables = ['JPO4', 'POPR', 'JPON', 'sat_def', 'snowpack', 'PO4T2R',
             'JPOP', 'baseflow', r'%sat_area', '0_redux', 'evap', 'trans']

variables_avail = [v for v in variables if v in X_small.columns]
int_cols = X_small.select_dtypes(include=['int']).columns

X_small[int_cols] = X_small[int_cols].astype(float)

missing = sorted(set(variables) - set(variables_avail))
if missing:
    print("Skipping missing variables:", missing)

title_size = 18
panel_title = 14
label_size  = 14
tick_size   = 11
line_width  = 3.0


try:
    lgbm_clf.set_params(n_jobs=1)    # alias of num_threads for sklearn wrapper
except Exception:
    if hasattr(lgbm_clf, "n_jobs"):
        lgbm_clf.n_jobs = 1

# grid layout
n_cols = 3
n_rows = math.ceil(len(variables_avail) / n_cols)


for CHLAclass in range(0, 4):
    
    print(f"[PDP] Class {CHLAclass} — grid (readable layout)")

    fig, axs = plt.subplots(n_rows, n_cols, figsize=(n_cols*6.5, n_rows*3.6), dpi=300)
    axs = np.atleast_2d(axs)

    # Compute PDPs - parallelized across features
    with parallel_backend("threading", n_jobs=4):
        disp = PartialDependenceDisplay.from_estimator(
            lgbm_clf,
            X_small,
            features=variables_avail,
            target=CHLAclass,              # index into predict_proba columns
            n_jobs=4,
            grid_resolution=30,
            verbose=0,
            ax=axs,
            # response_method='auto'       # default uses predict_proba for classifiers
        )

    fig.suptitle(f"Partial Dependence - LightGBM - Classification - Class:{CHLAclass}",
                 fontsize=title_size, fontweight="bold")

    ymins, ymaxs = [], []
    for ax in axs.flat:
        for ln in ax.lines:
            y = ln.get_ydata()
            if y.size:
                ymins.append(np.nanmin(y))
                ymaxs.append(np.nanmax(y))
    if ymins and ymaxs:
        ymin, ymax = float(np.min(ymins)), float(np.max(ymaxs))
        pad = 0.05*(ymax - ymin) if ymax > ymin else 0.01

    for r in range(n_rows):
        for c in range(n_cols):
            i = r*n_cols + c
            ax = axs[r, c]
            if i >= len(variables_avail):
                ax.axis("off")
                continue

            for ln in ax.lines:
                ln.set_linewidth(line_width)

            ax.set_title(ax.get_title(), fontsize=panel_title, fontweight="bold")

            for t in ax.get_xticklabels() + ax.get_yticklabels():
                t.set_fontsize(tick_size)
                t.set_fontweight("bold")

            # y-label only on left column
            if c == 0:
                ax.set_ylabel(f"Partial Dependence",
                              fontsize=label_size, fontweight="bold")
            else:
                ax.set_ylabel("")

            # lets also add the variable name as x-label
            varname = variables_avail[i]
            ax.set_xlabel(varname, fontsize=label_size, fontweight="bold", labelpad=8)

            if ymins and ymaxs:
                ax.set_ylim(ymin - pad, ymax + pad)

            xmin, xmax = ax.get_xlim()
            ymin, ymax = ax.get_ylim()
            xpad = 0.05 * (xmax - xmin)
            ypad = 0.05 * (ymax - ymin)
            ax.set_xlim(xmin - xpad, xmax + xpad)
            ax.set_ylim(ymin - ypad, ymax + ypad)        
            # ax.grid(alpha=0.25)

    # fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.subplots_adjust(
    left=0.07,    # space from left edge
    right=0.98,   # space from right edge
    top=0.90,     # space below suptitle
    bottom=0.08,  # space above x-axis labels
    wspace=0.05,  # width space between subplots
    hspace=0.45   # height space between subplots
            )

    out_grid = plots_dir / f"LightGBM Partial Dependence Plot - Class{CHLAclass}.png"
    fig.savefig(out_grid, dpi=500)
    print("Saved:", out_grid)
    plt.close(fig)
    
