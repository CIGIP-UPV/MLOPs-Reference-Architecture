# Drift Robustness Report

## Drift logic exercised

- Feature-level drift is flagged when any of the following is true: `KS p-value < 0.05`, `PSI >= 0.2`, or `normalized mean shift >= 1.0`.
- Aggregated severity is `high` when `average drift score >= 0.65` or `drifted features >= 4`.
- Aggregated severity is `medium` when `average drift score >= 0.35` or `drifted features >= 2`.
- Retraining is required when aggregated severity is `medium` or `high`.

## Experimental setup

- Reference profile built from `generate_historical_dataset(n_samples=2500, seed=42)`.
- Evaluation window size: `300` events.
- Nominal false-positive estimation repeats: `25`.
- Perturbations were synthetic and deterministic, applied as feature shifts measured in reference standard deviations.

## Summary table

| Condition | Repeats | Mean drift score | Drift score range | Mean drifted features | Severity mode | Retraining count | Retraining rate |
| --- | ---: | ---: | --- | ---: | --- | ---: | ---: |
| nominal | 25 | 0.173927 | 0.118649 - 0.224253 | 0.12 | low | 0 | 0.0 |
| mild | 1 | 0.244176 | 0.244176 - 0.244176 | 1 | low | 0 | 0.0 |
| medium | 1 | 0.403308 | 0.403308 - 0.403308 | 3 | medium | 1 | 1.0 |
| strong | 1 | 0.62838 | 0.62838 - 0.62838 | 6 | high | 1 | 1.0 |

## Thresholds exercised by the observed conditions

- `nominal`: no repeat reached aggregated `medium` severity or retraining; isolated feature-level flags occurred without escalating the overall decision.
- `mild`: one feature was flagged through `KS` and `PSI`, but the aggregated result remained `low` and `requires_retraining = False`.
- `medium`: three features exceeded the feature-level trigger logic and the aggregated result reached `medium`, so `requires_retraining = True`.
- `strong`: six features exceeded the feature-level trigger logic and the aggregated result reached `high`, so `requires_retraining = True`.

## False-positive control evidence

- Nominal repeats that triggered retraining: `0` out of `25`.
- Nominal retraining rate: `0.0`.
- Nominal per-feature drift flags: `3` out of `225` feature evaluations.
- This provides empirical evidence for false-positive control under the current synthetic nominal generator, not under all industrial operating regimes.

## Interpretation

- The nominal condition produced `low` as the modal severity with mean drift score `0.173927`.
- The mild condition produced severity `low` with drift score `0.244176` and retraining count `0`.
- The medium condition produced severity `medium` with drift score `0.403308` and retraining count `1`.
- The strong condition produced severity `high` with drift score `0.62838` and retraining count `1`.

## What remains unvalidated

- The experiment does not establish threshold optimality for real CNC data or multi-machine deployments.
- It does not quantify false negatives against a labeled real drift corpus.
- It does not validate online Airflow scheduling behavior under concurrent production load.
- It does not validate cross-dataset transferability of the current threshold set.

## Reproducibility

- The experiment is deterministic given the fixed seeds embedded in the script.
- All per-feature and aggregated outputs are stored in `results.csv`.
