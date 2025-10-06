#!/usr/bin/env python3
"""
Train an XGBoost model with validation, evaluate it on the test set,
and save the trained model for reuse.

"""

import os
import pickle
from pathlib import Path
from datetime import datetime

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
train_path = root_path / "comboCSV_train.csv"
test_path = root_path / "comboCSV_test.csv"

# loading data and setting it for training and evaluation
df_train = pd.read_csv(train_path, low_memory=False)
df_test = pd.read_csv(test_path, low_memory=False)

target_variable = "CHLAClass"
X_train, Y_train = df_train.drop(target_variable, axis=1), df_train[target_variable]
x_test, y_test = df_test.drop(target_variable, axis=1), df_test[target_variable]

# adding validation data too
x_train, x_val, y_train, y_val = train_test_split(
    X_train, Y_train, test_size=0.10, stratify=Y_train, random_state=14912
)


# building classifier
xgb_clf = xgb.XGBClassifier(
    objective="multi:softprob",
    num_class=4,
    learning_rate=0.05,       # we will use small learning rate since n_estimators are high
    n_estimators=50000,       # high upper bound, will stop early due to the early stopping criteria set to 200
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    tree_method="hist",       # fast histogram algorithm
    n_jobs=-1,
    eval_metric="mlogloss",
    early_stopping_rounds=200,
    random_state=14912,
)

# training and evaluation and testing
print("Training XGBoost model - This may take a while since we have large number of estimators.")
xgb_clf.fit(
    x_train, y_train,
    eval_set=[(x_val, y_val)],
    verbose=500
)

print("\nEvaluating best iteration on test set...")
y_pred = xgb_clf.predict(x_test)

precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)
f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

print(f"\nWeighted Precision: {precision:.4f}")
print(f"Weighted Recall:    {recall:.4f}")
print(f"Weighted F1 Score:  {f1:.4f}")

print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred, digits=4))


# saving the model
model_dir = root_path / "XGBoost" / "Classification" / "Weights"
model_dir.mkdir(parents=True, exist_ok=True)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
model_path = model_dir / "xgb_model_classification.pkl"

with open(model_path, "wb") as f:
    pickle.dump(xgb_clf, f)

print(f"\nTrained XGBoost model saved to: {model_path}")
