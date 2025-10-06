#!/usr/bin/env python3
"""
Train a LightGBM model on GPU with validation, evaluate it on the test set,
and save the trained model.
"""

import os
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, early_stopping, log_evaluation

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
)

# setting the paths 
root_path = Path('Project Directory Path')
train_path = root_path / "comboCSV_train.csv"
test_path = root_path / "comboCSV_test.csv"

outdir = root_path / "LightGBM" / 'Classification'
outdir.mkdir(parents=True, exist_ok=True)

df_train = pd.read_csv(train_path, low_memory=False)
df_test = pd.read_csv(test_path, low_memory=False)

target_variable = "CHLAClass"
X_train, Y_train = df_train.drop(target_variable, axis=1), df_train[target_variable]
x_test, y_test = df_test.drop(target_variable, axis=1), df_test[target_variable]

x_train, x_val, y_train, y_val = train_test_split(
    X_train, Y_train, test_size=0.10, stratify=Y_train, random_state=14912
)


# Build classifier - uses GPU
lgbm_clf = LGBMClassifier(
    objective="multiclass",
    num_class=4,
    learning_rate=0.05,
    n_estimators=50000,      # upper bound, use early stopping
    max_depth=-1,
    subsample=0.8,
    colsample_bytree=0.8,
    device_type="gpu",      # GPU inference/training
    n_jobs=-1,
    random_state=14912,
)

# we will train with early stopping
print("Training LightGBM model on GPU ")
lgbm_clf.fit(
    x_train,
    y_train,
    eval_set=[(x_val, y_val)],
    eval_metric="multi_logloss",
    callbacks=[],
)

# evaluating on test set
print("\nEvaluating best iteration on test set...")
y_pred = lgbm_clf.predict(x_test)

precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)
f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

print(f"\nWeighted Precision: {precision:.4f}")
print(f"Weighted Recall:    {recall:.4f}")
print(f"Weighted F1 Score:  {f1:.4f}")

print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred, digits=4))

with open(outdir / f"lgbm_model.pkl", "wb") as f:
    pickle.dump(lgbm_clf, f)

print(f"\nSaved model to: {outdir}")
