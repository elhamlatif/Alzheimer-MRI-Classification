"""
Train a Random Forest to separate Alzheimer's patients from healthy controls
based on subcortical brain volumes.

The input CSV must contain an "ID" column and one column per region
(see REGIONS). Subject IDs containing the control marker are labeled as
healthy controls; all others are treated as patients.
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

log = logging.getLogger("alzheimer")


# Subcortical regions used as features: pallidum, caudate, thalamus,
# putamen, accumbens, hippocampus, amygdala -- left and right.
REGIONS: tuple[str, ...] = (
    "R_pall", "L_pall",
    "R_Caud", "L_Caud",
    "R_Thal", "L_Thal",
    "R_Puta", "L_Puta",
    "R_Accu", "L_Accu",
    "R_Hipp", "L_Hipp",
    "R_Amyg", "L_Amyg",
)

# Subject IDs containing this marker belong to healthy controls.
CONTROL_MARKER = "_S_"

RANDOM_STATE = 42
TEST_SIZE = 0.2


# --------------------------------------------------------------------------- #
# Data structures
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class Dataset:
    X: np.ndarray
    y: np.ndarray

    @property
    def n_total(self) -> int:
        return len(self.y)

    @property
    def n_patients(self) -> int:
        return int(self.y.sum())

    @property
    def n_controls(self) -> int:
        return self.n_total - self.n_patients


@dataclass
class TrainedModel:
    clf: RandomForestClassifier
    scaler: StandardScaler
    metrics: dict[str, float]
    importances: pd.Series


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #

def load_data(csv_path: Path) -> Dataset:
    """Read the CSV, validate its shape, and derive binary labels from IDs."""
    if not csv_path.is_file():
        raise FileNotFoundError(f"Can't find {csv_path}")

    df = pd.read_csv(csv_path)

    missing = [c for c in REGIONS if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing columns: {missing}")
    if "ID" not in df.columns:
        raise ValueError("Dataset needs an 'ID' column")

    X = df.loc[:, REGIONS].to_numpy(dtype=float)

    # ID contains the control marker -> healthy control (0), otherwise patient (1).
    is_control = df["ID"].astype(str).str.contains(CONTROL_MARKER, regex=False)
    y = (~is_control).to_numpy(dtype=int)

    ds = Dataset(X=X, y=y)
    log.info(
        "Loaded %d subjects: %d patients, %d controls",
        ds.n_total, ds.n_patients, ds.n_controls,
    )
    return ds


# --------------------------------------------------------------------------- #
# Training & evaluation
# --------------------------------------------------------------------------- #

def _build_estimator() -> RandomForestClassifier:
    """Random Forest tuned for a small, mildly imbalanced dataset."""
    return RandomForestClassifier(
        n_estimators=150,
        max_depth=10,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )


def train(ds: Dataset) -> TrainedModel:
    """Split the data, scale the features, fit the classifier, and evaluate."""
    X_train, X_test, y_train, y_test = train_test_split(
        ds.X, ds.y,
        test_size=TEST_SIZE,
        stratify=ds.y,
        random_state=RANDOM_STATE,
    )

    scaler = StandardScaler().fit(X_train)
    clf = _build_estimator().fit(scaler.transform(X_train), y_train)

    X_test_scaled = scaler.transform(X_test)
    y_pred = clf.predict(X_test_scaled)
    y_proba = clf.predict_proba(X_test_scaled)[:, 1]

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "auc": float(roc_auc_score(y_test, y_proba)),
    }

    importances = (
        pd.Series(clf.feature_importances_, index=REGIONS)
        .sort_values(ascending=False)
    )

    return TrainedModel(
        clf=clf,
        scaler=scaler,
        metrics=metrics,
        importances=importances,
    )


def report(model: TrainedModel) -> None:
    """Log the headline metrics and the top regions by importance."""
    log.info(
        "Test accuracy: %.1f%%   AUC: %.3f",
        model.metrics["accuracy"] * 100,
        model.metrics["auc"],
    )
    log.info("Top 5 regions by importance:")
    for region, score in model.importances.head(5).items():
        log.info("  %-10s %5.1f%%", region, score * 100)


# --------------------------------------------------------------------------- #
# Persistence
# --------------------------------------------------------------------------- #

def save_model(model: TrainedModel, outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model.clf, outdir / "alzheimer_model.pkl")
    joblib.dump(model.scaler, outdir / "scaler.pkl")
    joblib.dump(list(REGIONS), outdir / "structure_names.pkl")
    log.info("Saved model artifacts to %s", outdir.resolve())


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train an Alzheimer's classifier from subcortical brain volumes."
    )
    parser.add_argument(
        "--data", type=Path, default=Path("data/brain_regions_data.csv"),
        help="Path to the volumetric CSV dataset",
    )
    parser.add_argument(
        "--outdir", type=Path, default=Path("models"),
        help="Where to save the trained model artifacts",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Enable debug-level logging",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
    )

    dataset = load_data(args.data)
    model = train(dataset)
    report(model)
    save_model(model, args.outdir)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())