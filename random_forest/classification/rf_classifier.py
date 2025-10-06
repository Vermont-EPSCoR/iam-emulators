#!/usr/bin/env python3
"""
Random Forest Classification Training + Evaluation Script

"""

# require dependencies
import os, numpy as np, pandas as pd, matplotlib.pyplot as plt
from pathlib import Path
import pickle
import joblib, json, time, sklearn

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (

            classification_report, confusion_matrix,
            precision_score, recall_score, f1_score

                        )

# data directory paths - root path (project directory) + train and test CSVs paths
root_path = Path('Project Directory Path')
train_path = root_path / 'comboCSV_train.csv'
test_path  = root_path / 'comboCSV_test.csv'

# load the dataset
print("Loading training and testing data...")
df_train = pd.read_csv(train_path, low_memory=False)
df_test  = pd.read_csv(test_path, low_memory=False)

target_variable = "CHLAClass"
x_train, y_train = df_train.drop(target_variable, axis=1), df_train[target_variable]
x_test, y_test = df_test.drop(target_variable, axis=1), df_test[target_variable]

print(f"Train size: {x_train.shape}, Test size: {x_test.shape}")


# fit the model
print("Fitting RandomForestRegressor...")
start_time = time.time()
rf = RandomForestClassifier(n_estimators=100, 
                            n_jobs=-1, 
                            verbose=1, 
                            random_state=14912)

rf.fit(x_train, y_train)
train_time = time.time() - start_time
print(f"Training finished in {train_time:.2f} seconds.")

# performance evaluations
print("Generating predictions on test set...")
y_pred = rf.predict(x_test)

# lets print the metrices
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

print(f"\nWeighted Precision: {precision:.4f}")
print(f"Weighted Recall:    {recall:.4f}")
print(f"Weighted F1 Score:  {f1:.4f}")

print("\nConfusion Matrix:\n", confusion_matrix(y_test, eval))
print("\nClassification Report:\n", classification_report(y_test, eval, digits=4))

# save the model and results in json file
out_dir = root_path / "Random_Forest" / "Classification" / "Weights"  / 'rf_classification.pkl'
out_dir.mkdir(parents=True, exist_ok=True)

joblib.dump(rf, out_dir)
print(f"Model saved to {out_dir}")
