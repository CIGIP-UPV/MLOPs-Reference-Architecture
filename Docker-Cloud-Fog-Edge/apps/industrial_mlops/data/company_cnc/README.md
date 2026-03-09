# Company CNC Reference Dataset

This folder intentionally contains only the minimum company data needed by the repository.

## What Was Reviewed

The company archive provided multiple layers of information:

- `TRAINING DATA/data.csv`: labelled time windows with the 12 Nakamura 2 signals and an `Alarm` column.
- `DATA_PREPARATION/`: intermediate exports, filled datasets and alternate aggregations.
- `DATA_LRT/ALARMS/`: raw alarm logs exported from MTLinki.
- `DATA_LRT/MANUFACTURING DATA*`: raw signal dumps by period and channel.
- `QUALITY DATA/` and `Data External Sensors/`: extra sources not required by the current project.

## What Was Kept

Only a compact curated dataset derived from `TRAINING DATA/data.csv`:

- `nakamura_reference_windows.csv`
- `nakamura_reference_windows.metadata.json`

The repository keeps the 12 average signals described by the company for the Nakamura 2:

- spindle speeds P1/P2
- feed rates P1/P2
- spindle motor loads P1/P2
- servo load currents P1/P2
- cutting times P1/P2
- APC temperatures P1/P2

## What Was Removed

The following elements were intentionally excluded from the repository:

- the original ZIP archive, because it is too large for version control
- raw MTLinki per-signal dumps, because they are not needed for the reference implementation
- quality spreadsheets and external-sensor files, because the current project does not consume them
- max/min feature variants, because the average-signal version is enough for the compact reference dataset

## Label Definition

`actual_breakage = 1` means that a target CNC alarm (`409`, `410` or `411`) occurs within the next 120 seconds.

This turns the company data into a predictive-maintenance reference set instead of a pure alarm replay.

## Review Findings

- The provided ZIP needed recovery handling because it contained extra bytes before the central directory.
- The labelled training export covers the period from `2023-03-28` to `2023-05-05`.
- In the supplied slice, only alarm `409` appears among the target alarms.
- The labelled export contains strong class imbalance, so the curated dataset keeps all positive windows and deterministically samples negatives.
