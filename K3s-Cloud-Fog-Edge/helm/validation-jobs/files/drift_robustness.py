from __future__ import annotations

import csv
from dataclasses import dataclass
import json
import os
from pathlib import Path
import statistics
from typing import Any

import numpy as np
import pandas as pd

from industrial_mlops.cnc_data import FEATURE_COLUMNS, build_reference_profile, generate_historical_dataset
from industrial_mlops.drift import compute_drift_report


RESULT_DIR = Path(os.environ.get("DRIFT_RESULTS_DIR", "/results/drift_robustness"))
WINDOW_SIZE = 300
REFERENCE_SAMPLES = 2500
NOMINAL_REPEATS = int(os.environ.get("DRIFT_NOMINAL_REPEATS", "25"))

KS_THRESHOLD = 0.05
PSI_THRESHOLD = 0.2
MEAN_SHIFT_THRESHOLD = 1.0
MEDIUM_SCORE_THRESHOLD = 0.35
HIGH_SCORE_THRESHOLD = 0.65
MEDIUM_VOTE_THRESHOLD = 2
HIGH_VOTE_THRESHOLD = 4


@dataclass(frozen=True)
class FeatureShift:
    shift_std: float = 0.0
    noise_std: float = 0.0
    scale: float = 1.0


def nominal_window(seed: int, window_size: int = WINDOW_SIZE) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    frame = generate_historical_dataset(n_samples=5000, seed=seed)
    indices = np.sort(rng.choice(len(frame), size=window_size, replace=False))
    return frame.iloc[indices].reset_index(drop=True)


def clip_feature(feature: str, values: np.ndarray) -> np.ndarray:
    if feature == "feed_rate":
        return np.clip(values, 0.01, None)
    if feature in {"vibration_x", "vibration_y"}:
        return np.clip(values, 0.05, None)
    if feature == "acoustic_emission":
        return np.clip(values, 10.0, None)
    if feature == "motor_current":
        return np.clip(values, 1.0, None)
    if feature == "tool_wear":
        return np.clip(values, 0.0, 1.4)
    return values


def apply_feature_shifts(
    frame: pd.DataFrame,
    reference_profile: dict[str, Any],
    shifts: dict[str, FeatureShift],
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    perturbed = frame.copy()
    for feature, shift in shifts.items():
        std = float(reference_profile[feature]["std"])
        values = perturbed[feature].astype(float).to_numpy(copy=True)
        values = values * shift.scale
        if shift.shift_std:
            values = values + shift.shift_std * std
        if shift.noise_std:
            values = values + rng.normal(0.0, shift.noise_std * std, size=len(values))
        perturbed[feature] = clip_feature(feature, values)
    return perturbed


def overall_row(condition: str, repeat: int, report: dict[str, Any]) -> dict[str, Any]:
    overall = report["overall"]
    return {
        "condition": condition,
        "repeat": repeat,
        "row_type": "overall",
        "feature": "__overall__",
        "ks_statistic": "",
        "ks_pvalue": "",
        "psi": "",
        "wasserstein": "",
        "mean_shift": "",
        "drift_detected": "",
        "trigger_by_ks": "",
        "trigger_by_psi": "",
        "trigger_by_mean_shift": "",
        "drift_score": overall["drift_score"],
        "drifted_features": overall["drifted_features"],
        "severity": overall["severity"],
        "requires_retraining": overall["requires_retraining"],
    }


def feature_rows(condition: str, repeat: int, report: dict[str, Any]) -> list[dict[str, Any]]:
    overall = report["overall"]
    rows: list[dict[str, Any]] = []
    for feature_report in report["feature_reports"]:
        rows.append(
            {
                "condition": condition,
                "repeat": repeat,
                "row_type": "feature",
                "feature": feature_report["feature"],
                "ks_statistic": feature_report["ks_statistic"],
                "ks_pvalue": feature_report["ks_pvalue"],
                "psi": feature_report["psi"],
                "wasserstein": feature_report["wasserstein"],
                "mean_shift": feature_report["mean_shift"],
                "drift_detected": feature_report["drift_detected"],
                "trigger_by_ks": float(feature_report["ks_pvalue"]) < KS_THRESHOLD,
                "trigger_by_psi": float(feature_report["psi"]) >= PSI_THRESHOLD,
                "trigger_by_mean_shift": float(feature_report["mean_shift"]) >= MEAN_SHIFT_THRESHOLD,
                "drift_score": overall["drift_score"],
                "drifted_features": overall["drifted_features"],
                "severity": overall["severity"],
                "requires_retraining": overall["requires_retraining"],
            }
        )
    return rows


def summarize_condition(overall_rows: list[dict[str, Any]]) -> dict[str, Any]:
    scores = [float(row["drift_score"]) for row in overall_rows]
    drifted_features = [int(row["drifted_features"]) for row in overall_rows]
    severities = [str(row["severity"]) for row in overall_rows]
    retraining = [bool(row["requires_retraining"]) for row in overall_rows]
    return {
        "repeats": len(overall_rows),
        "mean_drift_score": round(statistics.mean(scores), 6),
        "min_drift_score": round(min(scores), 6),
        "max_drift_score": round(max(scores), 6),
        "mean_drifted_features": round(statistics.mean(drifted_features), 6),
        "severity_mode": statistics.multimode(severities)[0],
        "retraining_count": int(sum(retraining)),
        "retraining_rate": round(sum(retraining) / len(retraining), 6),
    }


def main() -> int:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    reference_frame = generate_historical_dataset(n_samples=REFERENCE_SAMPLES, seed=42)
    reference_profile = build_reference_profile(reference_frame)

    perturbations: dict[str, dict[str, FeatureShift]] = {
        "nominal": {},
        "mild": {
            "spindle_temp": FeatureShift(shift_std=1.1, noise_std=0.05),
        },
        "medium": {
            "spindle_temp": FeatureShift(shift_std=1.1, noise_std=0.05),
            "motor_current": FeatureShift(shift_std=1.1, noise_std=0.05),
            "acoustic_emission": FeatureShift(shift_std=1.1, noise_std=0.05),
        },
        "strong": {
            "spindle_temp": FeatureShift(shift_std=1.6, noise_std=0.08),
            "motor_current": FeatureShift(shift_std=1.5, noise_std=0.08),
            "acoustic_emission": FeatureShift(shift_std=1.4, noise_std=0.08),
            "vibration_x": FeatureShift(shift_std=1.3, noise_std=0.08, scale=1.05),
            "vibration_y": FeatureShift(shift_std=1.3, noise_std=0.08, scale=1.05),
            "feed_rate": FeatureShift(shift_std=-1.2, noise_std=0.05),
        },
    }

    experiment_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    nominal_reports: list[dict[str, Any]] = []

    for repeat in range(1, NOMINAL_REPEATS + 1):
        current = nominal_window(1000 + repeat)
        report = compute_drift_report(reference_profile, current[FEATURE_COLUMNS])
        nominal_reports.append(report)
        experiment_rows.extend(feature_rows("nominal", repeat, report))
        experiment_rows.append(overall_row("nominal", repeat, report))

    for idx, condition in enumerate(["mild", "medium", "strong"], start=1):
        current = nominal_window(2000 + idx)
        perturbed = apply_feature_shifts(current[FEATURE_COLUMNS], reference_profile, perturbations[condition], seed=5000 + idx)
        report = compute_drift_report(reference_profile, perturbed)
        experiment_rows.extend(feature_rows(condition, 1, report))
        experiment_rows.append(overall_row(condition, 1, report))

    overall_by_condition: dict[str, list[dict[str, Any]]] = {}
    for row in experiment_rows:
        if row["row_type"] == "overall":
            overall_by_condition.setdefault(str(row["condition"]), []).append(row)

    for condition, rows in overall_by_condition.items():
        summary_rows.append({"condition": condition, **summarize_condition(rows)})

    results_path = RESULT_DIR / "results.csv"
    fieldnames = [
        "condition",
        "repeat",
        "row_type",
        "feature",
        "ks_statistic",
        "ks_pvalue",
        "psi",
        "wasserstein",
        "mean_shift",
        "drift_detected",
        "trigger_by_ks",
        "trigger_by_psi",
        "trigger_by_mean_shift",
        "drift_score",
        "drifted_features",
        "severity",
        "requires_retraining",
    ]
    with results_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(experiment_rows)

    nominal_feature_rows = [row for row in experiment_rows if row["condition"] == "nominal" and row["row_type"] == "feature"]
    nominal_false_feature_flags = sum(1 for row in nominal_feature_rows if str(row["drift_detected"]).lower() == "true")

    summary = {
        "conditions": summary_rows,
        "thresholds": {
            "ks_pvalue": KS_THRESHOLD,
            "psi": PSI_THRESHOLD,
            "mean_shift": MEAN_SHIFT_THRESHOLD,
            "medium_score": MEDIUM_SCORE_THRESHOLD,
            "high_score": HIGH_SCORE_THRESHOLD,
            "medium_votes": MEDIUM_VOTE_THRESHOLD,
            "high_votes": HIGH_VOTE_THRESHOLD,
        },
        "nominal_feature_level_false_flags": nominal_false_feature_flags,
        "nominal_feature_level_checks": len(nominal_feature_rows),
    }
    (RESULT_DIR / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    lines = [
        "# Drift Robustness Report",
        "",
        "## Drift logic exercised",
        "",
        f"- Feature-level drift is flagged when `KS p-value < {KS_THRESHOLD}`, `PSI >= {PSI_THRESHOLD}`, or `normalized mean shift >= {MEAN_SHIFT_THRESHOLD}`.",
        f"- Aggregated severity is `high` when `average drift score >= {HIGH_SCORE_THRESHOLD}` or `drifted features >= {HIGH_VOTE_THRESHOLD}`.",
        f"- Aggregated severity is `medium` when `average drift score >= {MEDIUM_SCORE_THRESHOLD}` or `drifted features >= {MEDIUM_VOTE_THRESHOLD}`.",
        "- Retraining is required when aggregated severity is `medium` or `high`.",
        "",
        "## Summary table",
        "",
        "| Condition | Repeats | Mean drift score | Drift score range | Mean drifted features | Severity mode | Retraining count | Retraining rate |",
        "| --- | ---: | ---: | --- | ---: | --- | ---: | ---: |",
    ]
    ordered = [
        next(row for row in summary_rows if row["condition"] == "nominal"),
        next(row for row in summary_rows if row["condition"] == "mild"),
        next(row for row in summary_rows if row["condition"] == "medium"),
        next(row for row in summary_rows if row["condition"] == "strong"),
    ]
    for row in ordered:
        lines.append(
            f"| {row['condition']} | {row['repeats']} | {row['mean_drift_score']} | {row['min_drift_score']} to {row['max_drift_score']} | {row['mean_drifted_features']} | {row['severity_mode']} | {row['retraining_count']} | {row['retraining_rate']} |"
        )
    lines.extend(
        [
            "",
            "## False-positive control observed",
            "",
            f"- Nominal feature-level flags: `{nominal_false_feature_flags}` / `{len(nominal_feature_rows)}` checks.",
            f"- Nominal retraining triggers: `{next(row for row in summary_rows if row['condition'] == 'nominal')['retraining_count']}` / `{NOMINAL_REPEATS}` windows.",
            "",
            "## Caveats",
            "",
            "- This Job exercises the current statistical drift logic without modifying production thresholds.",
            "- The perturbations are synthetic and deterministic. They support threshold interpretation, not transferability claims across machines.",
        ]
    )
    report = "\n".join(lines) + "\n"
    (RESULT_DIR / "report.md").write_text(report, encoding="utf-8")

    print(report)
    print("DRIFT_ROBUSTNESS_SUMMARY_JSON=" + json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
