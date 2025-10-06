
# loading libararies
import os, joblib, json, pickle
from joblib import parallel_backend
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn.metrics import ConfusionMatrixDisplay, classification_report
from sklearn.inspection import PartialDependenceDisplay
import math


# setting paths
root_path = Path('Project Directory Path')
random_weights = root_path / 'Random_Forest' / 'Classification' / 'Weights' / 'rf.pkl'
plots_dir    = root_path / "Random_Forest" / "Classification" / "Plots"
test_csv_path = root_path / 'comboCSV_test.csv'
train_csv_path = root_path / "comboCSV_train.csv"
plots_dir.mkdir(parents=True, exist_ok=True)


# loading data
train_data = pd.read_csv(train_csv_path, low_memory=False)
test_data = pd.read_csv(test_csv_path, low_memory=False)

target_variable = "CHLAClass"
X_train = train_data.drop(columns=[target_variable])
y_train = train_data[target_variable]
X_test  = test_data.drop(columns=[target_variable])
y_test  = test_data[target_variable]

# laoding weights
print('Data has been loaded - Loading Weights')
rf = joblib.load(random_weights)

# visualization
print("Model is loaded and tested - Creating Visualization")


subsample_n = 500
rng = np.random.default_rng(14912)
idx = rng.choice(X_train.index, size=min(subsample_n, len(X_train)), replace=False)
X_small = X_train.loc[idx].copy()
int_cols = X_small.select_dtypes(include=['int']).columns
X_small[int_cols] = X_small[int_cols].astype(float)

variables = ['JPO4', 'POPR', 'JPON', 'sat_def', 'snowpack', 'PO4T2R',
             'JPOP', 'baseflow', r'%sat_area', '0_redux', 'evap', 'trans']
variables_avail = [v for v in variables if v in X_small.columns]
missing = sorted(set(variables) - set(variables_avail))
if missing:
    print("Skipping missing variables:", missing)

# lets set the fonts
title_size = 18
panel_title = 14
label_size = 14
tick_size  = 11
line_width = 3.0


try:
    rf.set_params(n_jobs=1)
except Exception:
    rf.n_jobs = 1

# grid layout
n_cols = 3
n_rows = math.ceil(len(variables_avail) / n_cols)

for CHLAclass in range(0, 4):
    print(f"[PDP] Class {CHLAclass} — grid (readable layout)")

    fig, axs = plt.subplots(n_rows, n_cols, figsize=(n_cols*9, n_rows*3.6), dpi=300)
    axs = np.atleast_2d(axs)

    with parallel_backend("threading", n_jobs=4):
        disp = PartialDependenceDisplay.from_estimator(
            rf, X_small, variables_avail,
            target=CHLAclass,
            n_jobs=4,            # threads
            grid_resolution=30,
            verbose=0,
            ax=axs               
        )

    # overall title
    fig.suptitle(f"Partial Dependence - RF - Classification - Class:{CHLAclass}",
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

    # working to style each panel
    for r in range(n_rows):
        for c in range(n_cols):
            i = r*n_cols + c
            ax = axs[r, c]
            if i >= len(variables_avail):
                ax.axis("off")
                continue

            for ln in ax.lines:
                ln.set_linewidth(line_width)

            # panel title bold
            ax.set_title(ax.get_title(), fontsize=panel_title, fontweight="bold")

            # ticks bold
            for t in ax.get_xticklabels() + ax.get_yticklabels():
                t.set_fontsize(tick_size)
                t.set_fontweight("bold")

            # only left column gets y-label
            if c == 0:
                ax.set_ylabel("Partial Dependence", fontsize=label_size, fontweight="bold", labelpad=8)
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

    # fig.tight_layout(rect=[0, 0, 1, 0.93])  #leave room for suptitle
    fig.subplots_adjust(
    left=0.07,    # space from left edge
    right=0.98,   # space from right edge
    top=0.90,     # space below suptitle
    bottom=0.08,  # space above x-axis labels
    wspace=0.05,  # width space between subplots
    hspace=0.45   # height space between subplots
            )
    plt.show()

    out_grid = plots_dir / f"RF Partial Dependence Plot - Class{CHLAclass}.png"
    fig.savefig(out_grid, dpi=500)
    print("Saved:", out_grid)
    plt.close(fig)
    

print(f"Visualization is created and saved successfully.")


