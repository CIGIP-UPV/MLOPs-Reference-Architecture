from __future__ import annotations

import csv
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
COMPOSE_FILE = ROOT / "Docker-Cloud-Fog-Edge" / "docker-compose.yml"
RESULT_DIR = ROOT / "experiments" / "results" / "ota_continuity"
PUBLISHER_SCRIPT = ROOT / "experiments" / "publish_synthetic_edge_load.py"
PROMOTION_SCRIPT = ROOT / "experiments" / "promote_synthetic_generation.py"

PROJECT_ENV = {
    "SIMULATOR_PROFILE": "synthetic",
    "EDGE_BOOTSTRAP_MODEL_NAME": "cnc_tool_breakage_classifier",
    "MLFLOW_HOST_PORT": "5500",
}

EVENT_COUNT = int(os.environ.get("OTA_PROFILE_EVENT_COUNT", "1200"))
EVENT_INTERVAL = float(os.environ.get("OTA_PROFILE_EVENT_INTERVAL", "0.05"))
UPDATE_TRIGGER_SECONDS = float(os.environ.get("OTA_UPDATE_TRIGGER_SECONDS", "8.0"))
EDGE_CONTAINER = "ind-edge-inference"
SYNC_CONTAINER = "ind-edge-sync"
DB_CONTAINER = "ind-timescale"
DEPLOYMENT_STATE_PATH = "/var/lib/industrial-mlops/edge/deployment_state.json"


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


def iso_timestamp(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat()


def parse_ts(raw: str | None) -> datetime | None:
    if not raw:
        return None
    return datetime.fromisoformat(str(raw).replace("Z", "+00:00")).astimezone(timezone.utc)


def run(
    cmd: list[str],
    *,
    check: bool = True,
    env: dict[str, str] | None = None,
    input_text: str | None = None,
) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(
        cmd,
        input=input_text,
        text=True,
        capture_output=True,
        env=env,
        cwd=str(ROOT),
    )
    if check and completed.returncode != 0:
        raise RuntimeError(
            f"Command failed ({completed.returncode}): {' '.join(cmd)}\n"
            f"STDOUT:\n{completed.stdout}\nSTDERR:\n{completed.stderr}"
        )
    return completed


def docker_compose_env() -> dict[str, str]:
    env = os.environ.copy()
    env.update(PROJECT_ENV)
    return env


def run_compose(args: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    return run(["docker", "compose", "-f", str(COMPOSE_FILE), *args], check=check, env=docker_compose_env())


def wait_for(condition, timeout_seconds: float, label: str) -> None:
    deadline = time.time() + timeout_seconds
    last_error: str | None = None
    while time.time() < deadline:
        try:
            if condition():
                return
        except Exception as exc:  # pragma: no cover - defensive
            last_error = str(exc)
        time.sleep(1)
    detail = f" Last error: {last_error}" if last_error else ""
    raise TimeoutError(f"Timed out waiting for {label}.{detail}")


def deployment_state_ready() -> bool:
    result = run(
        ["docker", "exec", SYNC_CONTAINER, "sh", "-lc", f"test -f {DEPLOYMENT_STATE_PATH} && echo ready"],
        check=False,
    )
    return result.returncode == 0 and "ready" in result.stdout


def edge_metrics_ready() -> bool:
    probe = (
        "import urllib.request; "
        "print(urllib.request.urlopen('http://localhost:8010/metrics', timeout=2).status)"
    )
    result = run(["docker", "exec", EDGE_CONTAINER, "python", "-c", probe], check=False)
    return result.returncode == 0 and "200" in result.stdout


def edge_sync_metrics_ready() -> bool:
    probe = (
        "import urllib.request; "
        "print(urllib.request.urlopen('http://localhost:8012/metrics', timeout=2).status)"
    )
    result = run(["docker", "exec", SYNC_CONTAINER, "python", "-c", probe], check=False)
    return result.returncode == 0 and "200" in result.stdout


def load_deployment_state() -> dict[str, Any]:
    return json.loads(run(["docker", "exec", SYNC_CONTAINER, "cat", DEPLOYMENT_STATE_PATH]).stdout)


def fetch_metrics_text(container: str, port: int) -> str:
    probe = (
        "import urllib.request; "
        f"print(urllib.request.urlopen('http://localhost:{port}/metrics', timeout=5).read().decode('utf-8'))"
    )
    return run(["docker", "exec", container, "python", "-c", probe]).stdout


def extract_metric(metrics_text: str, name: str) -> float | None:
    pattern = re.compile(rf"^{re.escape(name)}(?:\{{.*\}})?\s+([0-9.eE+-]+)$", re.MULTILINE)
    matches = pattern.findall(metrics_text)
    if not matches:
        return None
    return float(matches[-1])


def metrics_snapshot(label: str) -> dict[str, Any]:
    edge_metrics = fetch_metrics_text(EDGE_CONTAINER, 8010)
    sync_metrics = fetch_metrics_text(SYNC_CONTAINER, 8012)
    return {
        "label": label,
        "captured_at": iso_timestamp(now_utc()),
        "edge_inference_predictions_total": extract_metric(edge_metrics, "edge_inference_predictions_total"),
        "edge_inference_schema_mismatch_total": extract_metric(edge_metrics, "edge_inference_schema_mismatch_total"),
        "edge_sync_generation": extract_metric(sync_metrics, "edge_sync_generation"),
        "edge_sync_commands_total": extract_metric(sync_metrics, "edge_sync_commands_total"),
        "edge_sync_failures_total": extract_metric(sync_metrics, "edge_sync_failures_total"),
    }


def run_psql_copy(sql: str) -> str:
    return run(
        ["docker", "exec", DB_CONTAINER, "psql", "-U", "admin", "-d", "factory_db", "-c", sql]
    ).stdout


def query_inference_rows(started_at: datetime, ended_at: datetime) -> list[dict[str, Any]]:
    start = iso_timestamp(started_at - timedelta(seconds=2))
    end = iso_timestamp(ended_at + timedelta(seconds=2))
    sql = (
        "COPY ("
        "SELECT event_time, cycle_id, deployment_generation, model_name, model_version, prediction, actual_breakage, drift_score, latency_ms "
        "FROM inference_events "
        f"WHERE event_time >= TIMESTAMPTZ '{start}' AND event_time <= TIMESTAMPTZ '{end}' "
        "ORDER BY event_time"
        ") TO STDOUT WITH CSV HEADER"
    )
    text = run_psql_copy(sql)
    reader = csv.DictReader(text.splitlines())
    rows: list[dict[str, Any]] = []
    for row in reader:
        rows.append(
            {
                "event_time": row["event_time"],
                "cycle_id": int(row["cycle_id"]),
                "deployment_generation": int(row["deployment_generation"]),
                "model_name": row["model_name"],
                "model_version": row["model_version"],
                "prediction": int(row["prediction"]),
                "actual_breakage": int(row["actual_breakage"]),
                "drift_score": float(row["drift_score"]),
                "latency_ms": float(row["latency_ms"]),
            }
        )
    return rows


def query_deployment_events(started_at: datetime, ended_at: datetime) -> list[dict[str, Any]]:
    start = iso_timestamp(started_at - timedelta(seconds=5))
    end = iso_timestamp(ended_at + timedelta(seconds=5))
    sql = (
        "COPY ("
        "SELECT event_time, action, reason, model_name, model_version, deployment_generation, checksum, status "
        "FROM deployment_events "
        f"WHERE event_time >= TIMESTAMPTZ '{start}' AND event_time <= TIMESTAMPTZ '{end}' "
        "ORDER BY event_time"
        ") TO STDOUT WITH CSV HEADER"
    )
    text = run_psql_copy(sql)
    return list(csv.DictReader(text.splitlines()))


def query_edge_sync_rows(started_at: datetime, ended_at: datetime) -> list[dict[str, Any]]:
    start = iso_timestamp(started_at - timedelta(seconds=5))
    end = iso_timestamp(ended_at + timedelta(seconds=5))
    sql = (
        "COPY ("
        "SELECT observed_at, machine_id, deployment_generation, model_version, sync_state, ota_latency_ms, notes "
        "FROM edge_sync_status "
        f"WHERE observed_at >= TIMESTAMPTZ '{start}' AND observed_at <= TIMESTAMPTZ '{end}' "
        "ORDER BY observed_at"
        ") TO STDOUT WITH CSV HEADER"
    )
    text = run_psql_copy(sql)
    return list(csv.DictReader(text.splitlines()))


def start_publisher() -> subprocess.Popen[str]:
    env_flags = [
        "-e",
        f"EDGE_PROFILE_EVENT_COUNT={EVENT_COUNT}",
        "-e",
        f"EDGE_PROFILE_EVENT_INTERVAL={EVENT_INTERVAL}",
    ]
    command = ["docker", "exec", *env_flags, "-i", SYNC_CONTAINER, "python", "-"]
    return subprocess.Popen(
        command,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd=str(ROOT),
    )


def run_promotion() -> dict[str, Any]:
    command = ["docker", "exec", "-i", SYNC_CONTAINER, "python", "-"]
    proc = subprocess.Popen(
        command,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd=str(ROOT),
    )
    script_text = PROMOTION_SCRIPT.read_text(encoding="utf-8")
    assert proc.stdin is not None
    proc.stdin.write(script_text)
    proc.stdin.close()
    proc.wait(timeout=180)
    stdout = proc.stdout.read() if proc.stdout is not None else ""
    stderr = proc.stderr.read() if proc.stderr is not None else ""
    if proc.returncode != 0:
        raise RuntimeError(f"Promotion failed:\nSTDOUT:\n{stdout}\nSTDERR:\n{stderr}")
    return json.loads(stdout.strip().splitlines()[-1])


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def event_time_deltas(rows: list[dict[str, Any]]) -> list[float]:
    parsed = [parse_ts(row["event_time"]) for row in rows]
    values: list[float] = []
    for prev, curr in zip(parsed, parsed[1:]):
        if prev is None or curr is None:
            continue
        values.append((curr - prev).total_seconds())
    return values


def max_or_none(values: list[float]) -> float | None:
    return round(max(values), 6) if values else None


def main() -> int:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    commands: list[str] = []

    def log_command(cmd: list[str]) -> None:
        commands.append(" ".join(cmd))

    try:
        up_cmd = ["docker", "compose", "-f", str(COMPOSE_FILE), "up", "-d", "edge-inference"]
        log_command(up_cmd)
        run_compose(["up", "-d", "edge-inference"])

        wait_for(deployment_state_ready, timeout_seconds=180, label="edge deployment state")
        wait_for(edge_metrics_ready, timeout_seconds=120, label="edge metrics endpoint")
        wait_for(edge_sync_metrics_ready, timeout_seconds=120, label="edge sync metrics endpoint")

        started_at = now_utc()
        state_before = load_deployment_state()
        metrics_before = metrics_snapshot("before_update")

        log_command(["docker", "exec", "-i", SYNC_CONTAINER, "python", "-", "<", str(PUBLISHER_SCRIPT)])
        publisher = start_publisher()
        publisher_script = PUBLISHER_SCRIPT.read_text(encoding="utf-8")
        assert publisher.stdin is not None
        publisher.stdin.write(publisher_script)
        publisher.stdin.close()

        time.sleep(UPDATE_TRIGGER_SECONDS)

        promotion_started_at = now_utc()
        log_command(["docker", "exec", "-i", SYNC_CONTAINER, "python", "-", "<", str(PROMOTION_SCRIPT)])
        promotion_result = run_promotion()
        promotion_finished_at = now_utc()
        metrics_after_promotion = metrics_snapshot("after_promotion")

        publisher.wait(timeout=max(180, int(EVENT_COUNT * EVENT_INTERVAL) + 60))
        publisher_stdout = publisher.stdout.read() if publisher.stdout is not None else ""
        publisher_stderr = publisher.stderr.read() if publisher.stderr is not None else ""
        if publisher.returncode != 0:
            raise RuntimeError(f"Publisher failed:\nSTDOUT:\n{publisher_stdout}\nSTDERR:\n{publisher_stderr}")
        publisher_summary = json.loads(publisher_stdout.strip().splitlines()[-1])

        time.sleep(5)
        ended_at = now_utc()
        state_after = load_deployment_state()
        metrics_after_stream = metrics_snapshot("after_stream")

        inference_rows = query_inference_rows(started_at, ended_at)
        if not inference_rows:
            raise RuntimeError("No inference events were persisted during the OTA continuity experiment.")

        deployment_events = query_deployment_events(started_at, ended_at)
        edge_sync_rows = query_edge_sync_rows(started_at, ended_at)

        if not deployment_events:
            raise RuntimeError("No deployment events were recorded during the OTA continuity experiment.")

        previous_generation = int(state_before["deployment_generation"])
        current_generation = int(state_after["deployment_generation"])
        update_event = deployment_events[-1]
        apply_row = next(
            (
                row
                for row in edge_sync_rows
                if row["sync_state"] == "applied" and int(row["deployment_generation"]) == current_generation
            ),
            None,
        )
        if apply_row is None:
            raise RuntimeError("No applied edge sync status was recorded during the OTA continuity experiment.")

        issue_time = parse_ts(update_event["event_time"])
        apply_time = parse_ts(apply_row["observed_at"])
        if issue_time is None or apply_time is None:
            raise RuntimeError("Could not parse issue/apply timestamps from update evidence.")

        timeline_rows: list[dict[str, Any]] = []
        for row in inference_rows:
            event_ts = parse_ts(row["event_time"])
            assert event_ts is not None
            if event_ts < issue_time:
                segment = "before_update"
            elif event_ts < apply_time:
                segment = "during_update"
            else:
                segment = "after_apply"
            timeline_rows.append({**row, "segment": segment})

        before_rows = [row for row in timeline_rows if row["segment"] == "before_update"]
        during_rows = [row for row in timeline_rows if row["segment"] == "during_update"]
        after_rows = [row for row in timeline_rows if row["segment"] == "after_apply"]

        rows_by_cycle = sorted(timeline_rows, key=lambda row: int(row["cycle_id"]))
        generations_by_cycle = [int(row["deployment_generation"]) for row in rows_by_cycle]
        switch_count = sum(1 for prev, curr in zip(generations_by_cycle, generations_by_cycle[1:]) if prev != curr)
        first_new_generation_row = next(
            (row for row in rows_by_cycle if int(row["deployment_generation"]) == current_generation),
            None,
        )
        last_previous_generation_row = next(
            (row for row in reversed(rows_by_cycle) if int(row["deployment_generation"]) == previous_generation),
            None,
        )

        before_deltas = event_time_deltas(before_rows)
        overall_deltas = event_time_deltas(timeline_rows)

        last_before = parse_ts(before_rows[-1]["event_time"]) if before_rows else None
        first_after = parse_ts(after_rows[0]["event_time"]) if after_rows else None
        boundary_gap_seconds = None
        if last_before and first_after:
            boundary_gap_seconds = round((first_after - last_before).total_seconds(), 6)

        clean_single_switch = bool(
            switch_count == 1
            and generations_by_cycle
            and generations_by_cycle[0] == previous_generation
            and generations_by_cycle[-1] == current_generation
        )
        switched_after_apply = any(int(row["deployment_generation"]) == current_generation for row in after_rows)

        inference_fields = [
            "event_time",
            "cycle_id",
            "deployment_generation",
            "model_name",
            "model_version",
            "prediction",
            "actual_breakage",
            "drift_score",
            "latency_ms",
            "segment",
        ]
        write_csv(RESULT_DIR / "inference_timeline.csv", inference_fields, timeline_rows)

        update_rows: list[dict[str, Any]] = [
            {
                "record_type": "deployment_state_before",
                "timestamp": iso_timestamp(started_at),
                "deployment_generation": state_before.get("deployment_generation"),
                "model_name": state_before.get("model_name"),
                "model_version": state_before.get("model_version"),
                "sync_state": "",
                "ota_latency_ms": "",
                "status": "",
                "action": "",
                "reason": "",
                "metric_predictions_total": metrics_before.get("edge_inference_predictions_total"),
                "metric_schema_mismatch_total": metrics_before.get("edge_inference_schema_mismatch_total"),
                "metric_sync_generation": metrics_before.get("edge_sync_generation"),
                "metric_sync_commands_total": metrics_before.get("edge_sync_commands_total"),
                "metric_sync_failures_total": metrics_before.get("edge_sync_failures_total"),
                "notes": "state before update",
            }
        ]
        update_rows.extend(
            {
                "record_type": "deployment_event",
                "timestamp": row["event_time"],
                "deployment_generation": row["deployment_generation"],
                "model_name": row["model_name"],
                "model_version": row["model_version"],
                "sync_state": "",
                "ota_latency_ms": "",
                "status": row["status"],
                "action": row["action"],
                "reason": row["reason"],
                "metric_predictions_total": "",
                "metric_schema_mismatch_total": "",
                "metric_sync_generation": "",
                "metric_sync_commands_total": "",
                "metric_sync_failures_total": "",
                "notes": row.get("checksum", ""),
            }
            for row in deployment_events
        )
        update_rows.extend(
            {
                "record_type": "edge_sync_status",
                "timestamp": row["observed_at"],
                "deployment_generation": row["deployment_generation"],
                "model_name": state_after.get("model_name"),
                "model_version": row["model_version"],
                "sync_state": row["sync_state"],
                "ota_latency_ms": row["ota_latency_ms"],
                "status": "",
                "action": "",
                "reason": "",
                "metric_predictions_total": "",
                "metric_schema_mismatch_total": "",
                "metric_sync_generation": "",
                "metric_sync_commands_total": "",
                "metric_sync_failures_total": "",
                "notes": row["notes"],
            }
            for row in edge_sync_rows
        )
        for snapshot in (metrics_after_promotion, metrics_after_stream):
            update_rows.append(
                {
                    "record_type": f"metrics_snapshot_{snapshot['label']}",
                    "timestamp": snapshot["captured_at"],
                    "deployment_generation": snapshot.get("edge_sync_generation"),
                    "model_name": "",
                    "model_version": "",
                    "sync_state": "",
                    "ota_latency_ms": "",
                    "status": "",
                    "action": "",
                    "reason": "",
                    "metric_predictions_total": snapshot.get("edge_inference_predictions_total"),
                    "metric_schema_mismatch_total": snapshot.get("edge_inference_schema_mismatch_total"),
                    "metric_sync_generation": snapshot.get("edge_sync_generation"),
                    "metric_sync_commands_total": snapshot.get("edge_sync_commands_total"),
                    "metric_sync_failures_total": snapshot.get("edge_sync_failures_total"),
                    "notes": snapshot["label"],
                }
            )
        update_rows.append(
            {
                "record_type": "deployment_state_after",
                "timestamp": iso_timestamp(ended_at),
                "deployment_generation": state_after.get("deployment_generation"),
                "model_name": state_after.get("model_name"),
                "model_version": state_after.get("model_version"),
                "sync_state": "",
                "ota_latency_ms": "",
                "status": "",
                "action": "",
                "reason": "",
                "metric_predictions_total": metrics_after_stream.get("edge_inference_predictions_total"),
                "metric_schema_mismatch_total": metrics_after_stream.get("edge_inference_schema_mismatch_total"),
                "metric_sync_generation": metrics_after_stream.get("edge_sync_generation"),
                "metric_sync_commands_total": metrics_after_stream.get("edge_sync_commands_total"),
                "metric_sync_failures_total": metrics_after_stream.get("edge_sync_failures_total"),
                "notes": "state after update",
            }
        )

        update_fields = [
            "record_type",
            "timestamp",
            "deployment_generation",
            "model_name",
            "model_version",
            "sync_state",
            "ota_latency_ms",
            "status",
            "action",
            "reason",
            "metric_predictions_total",
            "metric_schema_mismatch_total",
            "metric_sync_generation",
            "metric_sync_commands_total",
            "metric_sync_failures_total",
            "notes",
        ]
        write_csv(RESULT_DIR / "update_timeline.csv", update_fields, update_rows)

        (RESULT_DIR / "deployment_state_before.json").write_text(
            json.dumps(state_before, indent=2, sort_keys=True), encoding="utf-8"
        )
        (RESULT_DIR / "deployment_state_after.json").write_text(
            json.dumps(state_after, indent=2, sort_keys=True), encoding="utf-8"
        )
        (RESULT_DIR / "promotion_result.json").write_text(
            json.dumps(promotion_result, indent=2, sort_keys=True, default=str), encoding="utf-8"
        )
        (RESULT_DIR / "metrics_snapshots.json").write_text(
            json.dumps(
                {
                    "before_update": metrics_before,
                    "after_promotion": metrics_after_promotion,
                    "after_stream": metrics_after_stream,
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        sync_logs = run(["docker", "logs", SYNC_CONTAINER], check=False)
        inference_logs = run(["docker", "logs", EDGE_CONTAINER], check=False)
        (RESULT_DIR / "edge_sync.log").write_text(sync_logs.stdout + sync_logs.stderr, encoding="utf-8")
        (RESULT_DIR / "edge_inference.log").write_text(
            inference_logs.stdout + inference_logs.stderr,
            encoding="utf-8",
        )

        counts = {
            "before_update": len(before_rows),
            "during_update": len(during_rows),
            "after_apply": len(after_rows),
        }
        failed_syncs = [
            row for row in edge_sync_rows if row["sync_state"] not in {"applied", "accepted", "bootstrap"}
        ]

        report_lines = [
            "# OTA Continuity Report",
            "",
            "## Scope",
            "",
            "This experiment assessed whether the edge inference path remained operational while a newly registered synthetic model generation was promoted and applied through the existing OTA path.",
            "",
            "## Minimal execution path",
            "",
            "1. Start the minimal edge stack with the synthetic production model.",
            "2. Start a continuous stream of valid synthetic telemetry events over MQTT.",
            "3. Trigger registration and manual promotion of a new model version while the stream is active.",
            "4. Extract deployment state, edge sync status, Prometheus-format metric snapshots, logs, and persisted inference events.",
            "",
            "## Observed continuity evidence",
            "",
            f"- Previous accepted generation: `{previous_generation}` / version `{state_before.get('model_version')}`",
            f"- New accepted generation: `{current_generation}` / version `{state_after.get('model_version')}`",
            f"- Predictions associated with the previous generation in observed input order: `{sum(1 for row in rows_by_cycle if int(row['deployment_generation']) == previous_generation)}`",
            f"- Predictions associated with the new generation in observed input order: `{sum(1 for row in rows_by_cycle if int(row['deployment_generation']) == current_generation)}`",
            f"- Generation switches observed in input order: `{switch_count}`",
            f"- First cycle served by the new generation: `{first_new_generation_row['cycle_id'] if first_new_generation_row else 'n/a'}`",
            f"- Last cycle served by the previous generation: `{last_previous_generation_row['cycle_id'] if last_previous_generation_row else 'n/a'}`",
            f"- OTA apply latency reported by `edge_sync_status`: `{apply_row['ota_latency_ms']}` ms",
            f"- Sync failures observed in metrics: `{int(metrics_after_stream.get('edge_sync_failures_total') or 0)}`",
            f"- Schema mismatches observed in metrics: `{int(metrics_after_stream.get('edge_inference_schema_mismatch_total') or 0)}`",
            f"- Single clean generation switch observed in the persisted inference sequence: `{clean_single_switch}`",
            f"- New generation observed on persisted inference events after apply: `{switched_after_apply}`",
            "",
            "## Observed downtime or gaps",
            "",
            f"- Update issue time: `{iso_timestamp(issue_time)}`",
            f"- Update apply time: `{iso_timestamp(apply_time)}`",
            f"- Predictions with payload timestamps before issue time: `{counts['before_update']}`",
            f"- Predictions with payload timestamps in the `issue -> apply` window: `{counts['during_update']}`",
            f"- Predictions with payload timestamps after apply time: `{counts['after_apply']}`",
            f"- Boundary gap between the last pre-apply prediction and the first post-apply prediction: `{boundary_gap_seconds}` seconds",
            f"- Maximum consecutive prediction gap before update: `{max_or_none(before_deltas)}` seconds",
            f"- Maximum consecutive prediction gap over the full experiment: `{max_or_none(overall_deltas)}` seconds",
            "",
            "## Unsupported claims",
            "",
            "- No hard real-time guarantee is established by this experiment.",
            "- This does not prove microsecond-level continuity or deterministic scheduling.",
            "- The experiment uses the local Docker deployment on the current host, not a cross-device benchmark.",
            "- The promoted version was created and manually promoted for a controlled update event; this is continuity evidence, not a claim of universal promotion policy adequacy.",
            "",
            "## Caveats",
            "",
            "- Continuity is assessed from persisted inference timestamps, edge sync status, deployment state, and Prometheus-format counters.",
            "- `inference_events.event_time` comes from the sensor payload, not from the database commit time of the inference result.",
            "- For that reason, the payload-timestamp counts around the `issue -> apply` window are supportive context, not a direct proof of zero processing interruption.",
            "- If no predictions occur in the `issue -> apply` window, the result should be interpreted as an observed short update window, not as proof of zero interruption.",
            f"- Failed/rejected sync rows observed in the experiment window: `{len(failed_syncs)}`",
            "",
            "## Commands executed",
            "",
        ]
        report_lines.extend(f"- `{command}`" for command in commands)
        (RESULT_DIR / "report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
        return 0
    finally:
        run_compose(
            ["stop", "edge-inference", "edge-sync", "mosquitto", "db-bootstrap", "mlflow", "create_buckets", "minio", "timescale"],
            check=False,
        )


if __name__ == "__main__":
    sys.exit(main())
