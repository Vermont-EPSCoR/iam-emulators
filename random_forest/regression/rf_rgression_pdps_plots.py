
import pandas as pd, matplotlib.pyplot as plt, numpy as np, joblib
from pathlib import Path
import math
from sklearn.inspection import PartialDependenceDisplay
from joblib import parallel_backend


# setting paths
root_path = Path('Project Directory Path')
random_weights = root_path / 'Random_Forest' / 'Regression' / 'Weights' / 'rf_regression.pkl'
plots_dir    = root_path / "Random_Forest" / "Regression" / "Plots"
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
rf = joblib.load(random_weights)
print('Weights have been loaded - Generating PDPs Now')


subsample_n = 500 # using 500 samples for PDPs
rng = np.random.default_rng(14912)
idx = rng.choice(X_train.index, size=min(subsample_n, len(X_train)), replace=False)
X_small = X_train.loc[idx].copy()

# convert integer columns to float as PDPs do not support integer dtypes
int_cols = X_small.select_dtypes(include=['int']).columns
X_small[int_cols] = X_small[int_cols].astype(float)

# variables of interest
variables = ['JPO4', 'POPR', 'JPON', 'sat_def', 'snowpack', 'PO4T2R',
             'JPOP', 'baseflow', r'%sat_area', '0_redux', 'evap', 'trans']

variables_avail = [v for v in variables if v in X_small.columns] # just precautionary step - we wont need it though
missing = sorted(set(variables) - set(variables_avail))
if missing:
    print("Skipping missing variables:", missing)

# font settings
title_size = 18
panel_title = 14
label_size = 14
tick_size  = 11
line_width = 3.0

# ensure RF runs single-threaded inside PDPs
try:
    rf.set_params(n_jobs=1)
except Exception:
    rf.n_jobs = 1

# grid layout - total of three columns
n_cols = 3
n_rows = math.ceil(len(variables_avail) / n_cols)

print("Regression Task — grid (readable layout)")

fig, axs = plt.subplots(n_rows, n_cols, figsize=(n_cols*9, n_rows*3.6), dpi=300)
axs = np.atleast_2d(axs)

# compute PDPs with threading
with parallel_backend("threading", n_jobs=4):
    disp = PartialDependenceDisplay.from_estimator(
        rf, X_small, variables_avail,
        n_jobs=4,
        grid_resolution=30,
        verbose=0,
        ax=axs
    )

# overall title
fig.suptitle("Partial Dependence - RF - Regression: CHLAVESurfaceMean",
             fontsize=title_size, fontweight="bold")

# collect y-lims across panels dynamically
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

# style each panel
for r in range(n_rows):
    for c in range(n_cols):
        i = r*n_cols + c
        ax = axs[r, c]
        if i >= len(variables_avail):
            ax.axis("off")
            continue

        for ln in ax.lines:
            ln.set_linewidth(line_width)

        # panel title
        ax.set_title(ax.get_title(), fontsize=panel_title, fontweight="bold")

        # ticks bold
        for t in ax.get_xticklabels() + ax.get_yticklabels():
            t.set_fontsize(tick_size)
            t.set_fontweight("bold")

        if c == 0:
            ax.set_ylabel("Partial dependence", fontsize=label_size, fontweight="bold", labelpad=8)
        else:
            ax.set_ylabel("")

        # always put variable name under x-axis
        varname = variables_avail[i]
        ax.set_xlabel(varname, fontsize=label_size, fontweight="bold", labelpad=8)

        # consistent y-lims and adding padding around data
        if ymins and ymaxs:
            ax.set_ylim(ymin - pad, ymax + pad)
        xmin, xmax = ax.get_xlim()
        ymin, ymax = ax.get_ylim()
        xpad = 0.05 * (xmax - xmin)
        ypad = 0.05 * (ymax - ymin)
        ax.set_xlim(xmin - xpad, xmax + xpad)
        ax.set_ylim(ymin - ypad, ymax + ypad)

# adjust subplot spacing
fig.subplots_adjust(
    left=0.07, right=0.98, top=0.90, bottom=0.08,
    wspace=0.05, hspace=0.45
)

# save figure
out_grid = plots_dir / "RF PDPs Regression.png"
fig.savefig(out_grid, dpi=500)
print("Saved:", out_grid)
plt.close(fig)