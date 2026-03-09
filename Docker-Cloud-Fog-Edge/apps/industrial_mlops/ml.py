from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import tempfile
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split

from .cnc_data import FEATURE_COLUMNS, TARGET_COLUMN
from .security import compute_directory_digest


@dataclass
class TrainingResult:
    model: HistGradientBoostingClassifier
    metrics: dict[str, float]
    reference_profile: dict[str, Any]
    checksum: str
    model_dir: Path
    train_size: int
    test_size: int
    feature_columns: list[str]
    target_column: str


def _metric_bundle(y_true: pd.Series, y_pred: np.ndarray, y_score: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 6),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 6),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 6),
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 6),
        "roc_auc": round(float(roc_auc_score(y_true, y_score)), 6),
    }


def build_reference_profile(frame: pd.DataFrame, feature_columns: list[str]) -> dict[str, dict[str, float | list[float]]]:
    profile: dict[str, dict[str, float | list[float]]] = {}
    for feature in feature_columns:
        values = frame[feature].astype(float).to_numpy()
        quantiles = np.quantile(values, np.linspace(0.0, 1.0, 11)).tolist()
        profile[feature] = {
            "mean": round(float(np.mean(values)), 6),
            "std": round(float(np.std(values) + 1e-9), 6),
            "quantiles": [round(float(item), 6) for item in quantiles],
            "sample": [round(float(item), 6) for item in values[: min(len(values), 400)]],
        }
    return profile


def train_model(
    frame: pd.DataFrame,
    random_state: int = 42,
    feature_columns: list[str] | None = None,
    target_column: str = TARGET_COLUMN,
) -> TrainingResult:
    feature_columns = list(feature_columns or FEATURE_COLUMNS)
    if len(frame) < 200:
        raise ValueError("At least 200 events are required to train the CNC model.")
    x = frame[feature_columns].astype(float)
    y = frame[target_column].astype(int)
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.25,
        random_state=random_state,
        stratify=y,
    )
    model = HistGradientBoostingClassifier(
        max_depth=6,
        learning_rate=0.08,
        max_iter=220,
        random_state=random_state,
    )
    model.fit(x_train, y_train)
    y_score = model.predict_proba(x_test)[:, 1]
    y_pred = (y_score >= 0.5).astype(int)
    metrics = _metric_bundle(y_test, y_pred, y_score)

    temp_dir = Path(tempfile.mkdtemp(prefix="cnc-model-"))
    import mlflow.sklearn  # Imported lazily to keep package import side effects low.

    mlflow.sklearn.save_model(model, str(temp_dir))
    checksum = compute_directory_digest(temp_dir)
    return TrainingResult(
        model=model,
        metrics=metrics,
        reference_profile=build_reference_profile(x_train.reset_index(drop=True), feature_columns),
        checksum=checksum,
        model_dir=temp_dir,
        train_size=len(x_train),
        test_size=len(x_test),
        feature_columns=feature_columns,
        target_column=target_column,
    )


def evaluate_model(
    model: Any,
    frame: pd.DataFrame,
    feature_columns: list[str] | None = None,
    target_column: str = TARGET_COLUMN,
) -> dict[str, float]:
    feature_columns = list(feature_columns or FEATURE_COLUMNS)
    if frame.empty:
        return {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0, "roc_auc": 0.0}
    x = frame[feature_columns].astype(float)
    y_true = frame[target_column].astype(int)
    y_score = model.predict_proba(x)[:, 1]
    y_pred = (y_score >= 0.5).astype(int)
    return _metric_bundle(y_true, y_pred, y_score)


def build_training_summary(result: TrainingResult, frame: pd.DataFrame, reason: str, dataset_name: str = "synthetic-cnc") -> dict[str, Any]:
    positive_rate = float(frame[result.target_column].mean()) if not frame.empty else 0.0
    return {
        "dataset_name": dataset_name,
        "reason": reason,
        "rows": int(len(frame)),
        "train_size": result.train_size,
        "test_size": result.test_size,
        "positive_rate": round(positive_rate, 6),
        "metrics": result.metrics,
        "checksum": result.checksum,
        "features": result.feature_columns,
        "target": result.target_column,
    }


def write_json_artifact(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
