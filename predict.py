"""
This script takes the volumes of 14 subcortical brain regions,
runs them through the trained Random Forest model,
and tells you whether the subject is a control or has Alzheimer's,
plus the probability of Alzheimer's.

How to use:
    python predict.py                        # self-test with example data
    python predict.py --csv data/one.csv     # predict from a CSV file
"""

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


# --------------------------------------------------------------------------- #
# Load the files saved during training
# --------------------------------------------------------------------------- #

# These three files were created during training and must be reused here.
model = joblib.load("models/alzheimer_model.pkl")
scaler = joblib.load("models/scaler.pkl")
structure_names = joblib.load("models/structure_names.pkl")

# Important: we do NOT refit the scaler!
# The input values must be scaled the same way the model learned during training.

N_REGIONS = len(structure_names)           # 14 regions
LABELS = {0: "Control", 1: "Alzheimer's"}  # 0 = healthy, 1 = Alzheimer's


# --------------------------------------------------------------------------- #
# Predict from 14 numbers
# --------------------------------------------------------------------------- #

def predict_from_values(volumes):
    """
    Takes the 14 brain region volumes directly and makes a prediction.

    Input: a list of 14 numbers (in the exact order the model was trained on)
    Output: the probability that the subject has Alzheimer's (between 0 and 1)
    """

    # Make sure exactly 14 values came in
    volumes = list(volumes)
    if len(volumes) != N_REGIONS:
        raise ValueError(
            f"Expected {N_REGIONS} values, got {len(volumes)}.\n"
            f"Correct order: {structure_names}"
        )

    # Convert to a 2D array (that's what the model expects)
    X = np.asarray(volumes, dtype=float).reshape(1, -1)

    # Scale using the same scaler from training
    X_scaled = scaler.transform(X)

    # Predict
    proba = float(model.predict_proba(X_scaled)[0, 1])  # Alzheimer's probability
    label = int(model.predict(X_scaled)[0])             # 0 or 1

    print(f"Result: {LABELS[label]}  |  Alzheimer's probability: {proba:.1%}")
    return proba


# --------------------------------------------------------------------------- #
# Predict from a CSV file
# --------------------------------------------------------------------------- #

def predict_from_csv(file_path):
    """
    Takes a one-row CSV file (a single subject) and makes a prediction.

    If it has an ID column, that's ignored.
    It must contain all 14 region columns.
    """

    file_path = Path(file_path)
    if not file_path.is_file():
        raise FileNotFoundError(f"File not found: {file_path}")

    df = pd.read_csv(file_path)

    # Check that all required columns are present
    missing = [c for c in structure_names if c not in df.columns]
    if missing:
        raise ValueError(f"These columns are missing from the CSV: {missing}")

    # Must contain exactly one subject
    if len(df) != 1:
        raise ValueError(
            f"File must contain exactly one row, but has {len(df)} rows."
        )

    # Pull the values in the same order used during training
    volumes = df.loc[0, structure_names].to_numpy(dtype=float)
    return predict_from_values(volumes)


# --------------------------------------------------------------------------- #
# Run directly from the command line
# --------------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser(
        description="Detect Alzheimer's from brain region volumes"
    )
    parser.add_argument(
        "--csv", type=Path, default=None,
        help="Path to a one-row CSV file",
    )
    args = parser.parse_args()

    if args.csv:
        # Predict from the CSV file
        predict_from_csv(args.csv)
    else:
        # Self-test with a manual example
        print("No CSV given; running self-test with example data:\n")
        example = [
            2655, 2948, 5075, 4150, 9658, 9903, 5973, 5651,
            433, 630, 3457, 3301, 1342, 1238,
        ]
        predict_from_values(example)


if __name__ == "__main__":
    main()