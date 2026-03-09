from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, wasserstein_distance

from .cnc_data import FEATURE_COLUMNS


def _psi(reference: np.ndarray, current: np.ndarray, bins: np.ndarray) -> float:
    if len(bins) < 3:
        bins = np.linspace(min(reference.min(), current.min()), max(reference.max(), current.max()), 5)
    ref_hist, _ = np.histogram(reference, bins=bins)
    cur_hist, _ = np.histogram(current, bins=bins)
    ref_dist = np.where(ref_hist == 0, 1e-6, ref_hist / max(ref_hist.sum(), 1))
    cur_dist = np.where(cur_hist == 0, 1e-6, cur_hist / max(cur_hist.sum(), 1))
    return float(np.sum((cur_dist - ref_dist) * np.log(cur_dist / ref_dist)))


def compute_drift_report(reference_profile: dict[str, Any], current_frame: pd.DataFrame) -> dict[str, Any]:
    if current_frame.empty:
        raise ValueError("Cannot compute drift on an empty current frame.")
    feature_reports: list[dict[str, Any]] = []
    drift_votes = 0
    drift_score = 0.0
    for feature in FEATURE_COLUMNS:
        current_values = current_frame[feature].astype(float).to_numpy()
        reference_values = np.array(reference_profile[feature]["sample"], dtype=float)
        ks_result = ks_2samp(reference_values, current_values)
        bins = np.unique(np.array(reference_profile[feature]["quantiles"], dtype=float))
        psi = _psi(reference_values, current_values, bins)
        wasserstein = float(wasserstein_distance(reference_values, current_values))
        mean_shift = abs(float(np.mean(current_values)) - float(reference_profile[feature]["mean"])) / max(
            float(reference_profile[feature]["std"]), 1e-6
        )
        detected = bool(ks_result.pvalue < 0.05 or psi >= 0.2 or mean_shift >= 1.0)
        drift_votes += int(detected)
        weighted_score = min(1.0, (1.0 - min(float(ks_result.pvalue), 1.0)) * 0.35 + min(psi, 1.0) * 0.4 + min(mean_shift / 3.0, 1.0) * 0.25)
        drift_score += weighted_score
        feature_reports.append(
            {
                "feature": feature,
                "ks_statistic": round(float(ks_result.statistic), 6),
                "ks_pvalue": round(float(ks_result.pvalue), 6),
                "psi": round(float(psi), 6),
                "wasserstein": round(wasserstein, 6),
                "mean_shift": round(mean_shift, 6),
                "drift_detected": detected,
            }
        )
    average_score = drift_score / max(len(FEATURE_COLUMNS), 1)
    severity = "low"
    if average_score >= 0.65 or drift_votes >= 4:
        severity = "high"
    elif average_score >= 0.35 or drift_votes >= 2:
        severity = "medium"
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "overall": {
            "drift_score": round(float(average_score), 6),
            "drifted_features": drift_votes,
            "severity": severity,
            "requires_retraining": severity in {"medium", "high"},
        },
        "feature_reports": feature_reports,
    }
