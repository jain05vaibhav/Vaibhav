#!/usr/bin/env python3
"""
Script to plot confusion matrix for the failure prediction model.
This generates test data, loads the trained failure model, makes predictions,
and plots the confusion matrix.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from cloud_ai.data_generation import generate_synthetic_cloud_history

def plot_failure_confusion_matrix():
    """
    Load the failure model, generate test data, predict, and plot confusion matrix.
    """
    print("Loading failure model...")
    try:
        model_failure = joblib.load("failure_model.pkl")
    except FileNotFoundError:
        print("Error: failure_model.pkl not found. Please train the models first.")
        return

    print("Generating test data...")
    # Generate a separate test dataset
    test_df = generate_synthetic_cloud_history(
        output_path="test_data.csv",
        rows=1000,
        seed=123  # Different seed for test data
    )

    # Prepare features for failure model
    failure_features = [
        "engine_rul_pct",
        "brake_rul_pct",
        "battery_rul_pct",
        "thermal_stress_index",
        "mechanical_vibration_anomaly_score"
    ]

    X_test = test_df[failure_features]
    y_true = test_df["failure_next_7_days"]

    print("Making predictions...")
    y_pred = model_failure.predict(X_test)

    print("Computing confusion matrix...")
    cm = confusion_matrix(y_true, y_pred)

    print("Confusion Matrix:")
    print(cm)

    # Plot confusion matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No Failure", "Failure"])
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix for Failure Prediction Model")
    plt.savefig("confusion_matrix.png")
    print("Confusion matrix plot saved to confusion_matrix.png")
    plt.show()  # Optional, can be commented out for headless

if __name__ == "__main__":
    plot_failure_confusion_matrix()