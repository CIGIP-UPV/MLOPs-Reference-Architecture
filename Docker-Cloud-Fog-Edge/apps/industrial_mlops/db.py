from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import json
from typing import Any, Iterator

import pandas as pd
import psycopg2
from psycopg2 import extras

from .cnc_data import FEATURE_COLUMNS, generate_historical_dataset
from .company_dataset import COMPANY_FEATURE_COLUMNS, COMPANY_SCHEMA_NAME, company_dataset_exists, load_company_reference_dataset
from .config import CONFIG


@contextmanager
def connection(database: str) -> Iterator[psycopg2.extensions.connection]:
    conn = psycopg2.connect(CONFIG.admin_dsn(database))
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def ensure_platform_databases() -> None:
    with connection("postgres") as conn:
        conn.autocommit = True
        with conn.cursor() as cursor:
            for database in (CONFIG.airflow_db_name, CONFIG.mlflow_db_name, CONFIG.factory_db_name):
                cursor.execute("SELECT 1 FROM pg_database WHERE datname = %s", (database,))
                if cursor.fetchone() is None:
                    cursor.execute(f'CREATE DATABASE "{database}"')
    with connection(CONFIG.factory_db_name) as conn:
        with conn.cursor() as cursor:
            cursor.execute("CREATE EXTENSION IF NOT EXISTS timescaledb")
            cursor.execute("CREATE EXTENSION IF NOT EXISTS pgcrypto")
    ensure_factory_schema()


def ensure_factory_schema() -> None:
    with connection(CONFIG.factory_db_name) as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS cnc_sensor_events (
                    id BIGSERIAL,
                    event_time TIMESTAMPTZ NOT NULL,
                    machine_id TEXT NOT NULL,
                    cell_id TEXT NOT NULL,
                    cycle_id BIGINT NOT NULL,
                    tool_id BIGINT NOT NULL,
                    spindle_speed DOUBLE PRECISION NOT NULL,
                    feed_rate DOUBLE PRECISION NOT NULL,
                    vibration_x DOUBLE PRECISION NOT NULL,
                    vibration_y DOUBLE PRECISION NOT NULL,
                    acoustic_emission DOUBLE PRECISION NOT NULL,
                    spindle_temp DOUBLE PRECISION NOT NULL,
                    motor_current DOUBLE PRECISION NOT NULL,
                    tool_wear DOUBLE PRECISION NOT NULL,
                    material_hardness DOUBLE PRECISION NOT NULL,
                    actual_breakage SMALLINT NOT NULL,
                    breakage_probability DOUBLE PRECISION,
                    payload_signature TEXT,
                    source_protocol TEXT DEFAULT 'mqtt'
                )
                """
            )
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS inference_events (
                    id BIGSERIAL,
                    event_time TIMESTAMPTZ NOT NULL,
                    machine_id TEXT NOT NULL,
                    cycle_id BIGINT NOT NULL,
                    deployment_generation BIGINT NOT NULL,
                    model_name TEXT NOT NULL,
                    model_version TEXT NOT NULL,
                    prediction SMALLINT NOT NULL,
                    risk_score DOUBLE PRECISION NOT NULL,
                    actual_breakage SMALLINT NOT NULL,
                    drift_score DOUBLE PRECISION,
                    latency_ms DOUBLE PRECISION,
                    outcome TEXT,
                    command_generation BIGINT,
                    sync_state TEXT DEFAULT 'accepted'
                )
                """
            )
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS drift_reports (
                    id BIGSERIAL,
                    report_time TIMESTAMPTZ NOT NULL,
                    window_start TIMESTAMPTZ,
                    window_end TIMESTAMPTZ,
                    model_version TEXT,
                    overall_drift_score DOUBLE PRECISION NOT NULL,
                    drift_severity TEXT NOT NULL,
                    requires_retraining BOOLEAN NOT NULL,
                    drifted_features INTEGER NOT NULL,
                    feature_reports JSONB NOT NULL
                )
                """
            )
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS deployment_events (
                    id BIGSERIAL,
                    event_time TIMESTAMPTZ NOT NULL,
                    action TEXT NOT NULL,
                    reason TEXT,
                    model_name TEXT NOT NULL,
                    model_version TEXT NOT NULL,
                    source_run_id TEXT,
                    deployment_generation BIGINT NOT NULL,
                    checksum TEXT,
                    signature TEXT,
                    status TEXT NOT NULL,
                    target_stage TEXT,
                    manifest JSONB NOT NULL
                )
                """
            )
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS edge_sync_status (
                    id BIGSERIAL,
                    observed_at TIMESTAMPTZ NOT NULL,
                    machine_id TEXT NOT NULL,
                    deployment_generation BIGINT NOT NULL,
                    model_version TEXT NOT NULL,
                    sync_state TEXT NOT NULL,
                    ota_latency_ms DOUBLE PRECISION,
                    notes TEXT
                )
                """
            )
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS company_reference_events (
                    id BIGSERIAL,
                    event_time TIMESTAMPTZ NOT NULL,
                    selection_reason TEXT NOT NULL,
                    current_alarm_code TEXT NOT NULL,
                    seconds_to_target_alarm DOUBLE PRECISION,
                    actual_breakage SMALLINT NOT NULL,
                    spindle_speed_p1_rpm DOUBLE PRECISION NOT NULL,
                    spindle_speed_p2_rpm DOUBLE PRECISION NOT NULL,
                    feed_rate_p1_mm_min DOUBLE PRECISION NOT NULL,
                    feed_rate_p2_mm_min DOUBLE PRECISION NOT NULL,
                    spindle_motor_load_p1_pct DOUBLE PRECISION NOT NULL,
                    spindle_motor_load_p2_pct DOUBLE PRECISION NOT NULL,
                    servo_load_current_p1_pct DOUBLE PRECISION NOT NULL,
                    servo_load_current_p2_pct DOUBLE PRECISION NOT NULL,
                    cutting_time_p1_s DOUBLE PRECISION NOT NULL,
                    cutting_time_p2_s DOUBLE PRECISION NOT NULL,
                    temp_apc_p1_c DOUBLE PRECISION NOT NULL,
                    temp_apc_p2_c DOUBLE PRECISION NOT NULL,
                    split TEXT NOT NULL,
                    dataset_name TEXT DEFAULT 'nakamura_reference_windows'
                )
                """
            )
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS company_sensor_events (
                    id BIGSERIAL,
                    event_time TIMESTAMPTZ NOT NULL,
                    machine_id TEXT NOT NULL,
                    cell_id TEXT NOT NULL,
                    cycle_id BIGINT NOT NULL,
                    tool_id BIGINT NOT NULL,
                    spindle_speed_p1_rpm DOUBLE PRECISION NOT NULL,
                    spindle_speed_p2_rpm DOUBLE PRECISION NOT NULL,
                    feed_rate_p1_mm_min DOUBLE PRECISION NOT NULL,
                    feed_rate_p2_mm_min DOUBLE PRECISION NOT NULL,
                    spindle_motor_load_p1_pct DOUBLE PRECISION NOT NULL,
                    spindle_motor_load_p2_pct DOUBLE PRECISION NOT NULL,
                    servo_load_current_p1_pct DOUBLE PRECISION NOT NULL,
                    servo_load_current_p2_pct DOUBLE PRECISION NOT NULL,
                    cutting_time_p1_s DOUBLE PRECISION NOT NULL,
                    cutting_time_p2_s DOUBLE PRECISION NOT NULL,
                    temp_apc_p1_c DOUBLE PRECISION NOT NULL,
                    temp_apc_p2_c DOUBLE PRECISION NOT NULL,
                    actual_breakage SMALLINT NOT NULL,
                    current_alarm_code TEXT,
                    seconds_to_target_alarm DOUBLE PRECISION,
                    selection_reason TEXT,
                    payload_signature TEXT,
                    source_protocol TEXT DEFAULT 'mqtt-company',
                    schema_name TEXT DEFAULT 'company-nakamura-reference'
                )
                """
            )
            # Older persisted volumes may still contain non-Timescale-compatible
            # primary keys and indexes. Remove them before hypertable creation.
            for table_name in (
                "cnc_sensor_events",
                "inference_events",
                "drift_reports",
                "deployment_events",
                "edge_sync_status",
                "company_reference_events",
                "company_sensor_events",
            ):
                cursor.execute(f"ALTER TABLE {table_name} DROP CONSTRAINT IF EXISTS {table_name}_pkey")
            cursor.execute("DROP INDEX IF EXISTS ux_cnc_sensor_cycle")
            cursor.execute("DROP INDEX IF EXISTS ux_company_sensor_cycle")
            hypertable_time_columns = {
                "cnc_sensor_events": "event_time",
                "inference_events": "event_time",
                "drift_reports": "report_time",
                "deployment_events": "event_time",
                "edge_sync_status": "observed_at",
                "company_reference_events": "event_time",
                "company_sensor_events": "event_time",
            }
            for hypertable, time_column in hypertable_time_columns.items():
                cursor.execute(
                    "SELECT create_hypertable(%s, %s, if_not_exists => TRUE)",
                    (hypertable, time_column),
                )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS ix_cnc_sensor_cycle ON cnc_sensor_events(machine_id, cycle_id, tool_id, event_time)"
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS ix_company_sensor_cycle ON company_sensor_events(machine_id, cycle_id, event_time)"
            )


def seed_historical_data_if_empty(n_samples: int = 2500) -> int:
    with connection(CONFIG.factory_db_name) as conn:
        with conn.cursor() as cursor:
            cursor.execute("SELECT COUNT(*) FROM cnc_sensor_events")
            existing = cursor.fetchone()[0]
    if existing:
        return int(existing)
    dataset = generate_historical_dataset(n_samples=n_samples, machine_id=CONFIG.edge_machine_id, cell_id=CONFIG.edge_cell_id)
    rows = [
        (
            row.event_time,
            row.machine_id,
            row.cell_id,
            int(row.cycle_id),
            int(row.tool_id),
            float(row.spindle_speed),
            float(row.feed_rate),
            float(row.vibration_x),
            float(row.vibration_y),
            float(row.acoustic_emission),
            float(row.spindle_temp),
            float(row.motor_current),
            float(row.tool_wear),
            float(row.material_hardness),
            int(row.actual_breakage),
            float(row.breakage_probability),
            None,
            str(row.source_protocol),
        )
        for row in dataset.itertuples(index=False)
    ]
    with connection(CONFIG.factory_db_name) as conn:
        with conn.cursor() as cursor:
            extras.execute_batch(
                cursor,
                """
                INSERT INTO cnc_sensor_events (
                    event_time, machine_id, cell_id, cycle_id, tool_id,
                    spindle_speed, feed_rate, vibration_x, vibration_y, acoustic_emission,
                    spindle_temp, motor_current, tool_wear, material_hardness,
                    actual_breakage, breakage_probability, payload_signature, source_protocol
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT DO NOTHING
                """,
                rows,
                page_size=250,
            )
    return len(rows)


def seed_company_reference_data_if_empty() -> int:
    if not company_dataset_exists():
        return 0
    with connection(CONFIG.factory_db_name) as conn:
        with conn.cursor() as cursor:
            cursor.execute("SELECT COUNT(*) FROM company_reference_events")
            existing = cursor.fetchone()[0]
    if existing:
        return int(existing)
    dataset = load_company_reference_dataset()
    rows = [
        (
            row.event_time,
            str(row.selection_reason),
            str(row.current_alarm_code),
            float(row.seconds_to_target_alarm) if pd.notna(row.seconds_to_target_alarm) and row.seconds_to_target_alarm != "" else None,
            int(row.actual_breakage),
            float(row.spindle_speed_p1_rpm),
            float(row.spindle_speed_p2_rpm),
            float(row.feed_rate_p1_mm_min),
            float(row.feed_rate_p2_mm_min),
            float(row.spindle_motor_load_p1_pct),
            float(row.spindle_motor_load_p2_pct),
            float(row.servo_load_current_p1_pct),
            float(row.servo_load_current_p2_pct),
            float(row.cutting_time_p1_s),
            float(row.cutting_time_p2_s),
            float(row.temp_apc_p1_c),
            float(row.temp_apc_p2_c),
            str(row.split),
        )
        for row in dataset.itertuples(index=False)
    ]
    with connection(CONFIG.factory_db_name) as conn:
        with conn.cursor() as cursor:
            extras.execute_batch(
                cursor,
                """
                INSERT INTO company_reference_events (
                    event_time, selection_reason, current_alarm_code, seconds_to_target_alarm,
                    actual_breakage, spindle_speed_p1_rpm, spindle_speed_p2_rpm,
                    feed_rate_p1_mm_min, feed_rate_p2_mm_min,
                    spindle_motor_load_p1_pct, spindle_motor_load_p2_pct,
                    servo_load_current_p1_pct, servo_load_current_p2_pct,
                    cutting_time_p1_s, cutting_time_p2_s,
                    temp_apc_p1_c, temp_apc_p2_c,
                    split
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                rows,
                page_size=250,
            )
    return len(rows)


def insert_sensor_event(payload: dict[str, Any]) -> None:
    event_time = pd.to_datetime(payload["event_time"], utc=True).to_pydatetime()
    schema_name = str(payload.get("schema_name", "synthetic-cnc"))
    if schema_name == COMPANY_SCHEMA_NAME or all(feature in payload for feature in COMPANY_FEATURE_COLUMNS):
        with connection(CONFIG.factory_db_name) as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    """
                    INSERT INTO company_sensor_events (
                        event_time, machine_id, cell_id, cycle_id, tool_id,
                        spindle_speed_p1_rpm, spindle_speed_p2_rpm,
                        feed_rate_p1_mm_min, feed_rate_p2_mm_min,
                        spindle_motor_load_p1_pct, spindle_motor_load_p2_pct,
                        servo_load_current_p1_pct, servo_load_current_p2_pct,
                        cutting_time_p1_s, cutting_time_p2_s,
                        temp_apc_p1_c, temp_apc_p2_c,
                        actual_breakage, current_alarm_code, seconds_to_target_alarm,
                        selection_reason, payload_signature, source_protocol, schema_name
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT DO NOTHING
                    """,
                    (
                        event_time,
                        payload["machine_id"],
                        payload["cell_id"],
                        int(payload["cycle_id"]),
                        int(payload.get("tool_id", 1)),
                        float(payload["spindle_speed_p1_rpm"]),
                        float(payload["spindle_speed_p2_rpm"]),
                        float(payload["feed_rate_p1_mm_min"]),
                        float(payload["feed_rate_p2_mm_min"]),
                        float(payload["spindle_motor_load_p1_pct"]),
                        float(payload["spindle_motor_load_p2_pct"]),
                        float(payload["servo_load_current_p1_pct"]),
                        float(payload["servo_load_current_p2_pct"]),
                        float(payload["cutting_time_p1_s"]),
                        float(payload["cutting_time_p2_s"]),
                        float(payload["temp_apc_p1_c"]),
                        float(payload["temp_apc_p2_c"]),
                        int(payload["actual_breakage"]),
                        str(payload.get("current_alarm_code", "0.0")),
                        float(payload["seconds_to_target_alarm"]) if payload.get("seconds_to_target_alarm") not in (None, "") else None,
                        payload.get("selection_reason"),
                        payload.get("signature"),
                        payload.get("source_protocol", "mqtt-company"),
                        schema_name,
                    ),
                )
        return
    with connection(CONFIG.factory_db_name) as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO cnc_sensor_events (
                    event_time, machine_id, cell_id, cycle_id, tool_id,
                    spindle_speed, feed_rate, vibration_x, vibration_y, acoustic_emission,
                    spindle_temp, motor_current, tool_wear, material_hardness,
                    actual_breakage, breakage_probability, payload_signature, source_protocol
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT DO NOTHING
                """,
                (
                    event_time,
                    payload["machine_id"],
                    payload["cell_id"],
                    int(payload["cycle_id"]),
                    int(payload["tool_id"]),
                    float(payload["spindle_speed"]),
                    float(payload["feed_rate"]),
                    float(payload["vibration_x"]),
                    float(payload["vibration_y"]),
                    float(payload["acoustic_emission"]),
                    float(payload["spindle_temp"]),
                    float(payload["motor_current"]),
                    float(payload["tool_wear"]),
                    float(payload["material_hardness"]),
                    int(payload["actual_breakage"]),
                    float(payload.get("breakage_probability", 0.0)),
                    payload.get("signature"),
                    payload.get("source_protocol", "mqtt"),
                ),
            )


def load_recent_company_sensor_window(limit: int = 250) -> pd.DataFrame:
    frame = query_frame(
        "SELECT * FROM company_sensor_events ORDER BY event_time DESC LIMIT %s",
        (limit,),
    )
    return frame.sort_values("event_time").reset_index(drop=True)


def insert_inference_event(payload: dict[str, Any]) -> None:
    with connection(CONFIG.factory_db_name) as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO inference_events (
                    event_time, machine_id, cycle_id, deployment_generation, model_name,
                    model_version, prediction, risk_score, actual_breakage, drift_score,
                    latency_ms, outcome, command_generation, sync_state
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    pd.to_datetime(payload["event_time"], utc=True).to_pydatetime(),
                    payload["machine_id"],
                    int(payload["cycle_id"]),
                    int(payload["deployment_generation"]),
                    payload["model_name"],
                    payload["model_version"],
                    int(payload["prediction"]),
                    float(payload["risk_score"]),
                    int(payload["actual_breakage"]),
                    float(payload.get("drift_score", 0.0)),
                    float(payload.get("latency_ms", 0.0)),
                    payload.get("outcome", "observed"),
                    int(payload.get("command_generation", payload["deployment_generation"])),
                    payload.get("sync_state", "accepted"),
                ),
            )


def insert_drift_report(report: dict[str, Any], model_version: str | None, window_start: datetime | None, window_end: datetime | None) -> None:
    with connection(CONFIG.factory_db_name) as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO drift_reports (
                    report_time, window_start, window_end, model_version,
                    overall_drift_score, drift_severity, requires_retraining,
                    drifted_features, feature_reports
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    pd.to_datetime(report["generated_at"], utc=True).to_pydatetime(),
                    window_start,
                    window_end,
                    model_version,
                    float(report["overall"]["drift_score"]),
                    report["overall"]["severity"],
                    bool(report["overall"]["requires_retraining"]),
                    int(report["overall"]["drifted_features"]),
                    json.dumps(report["feature_reports"]),
                ),
            )


def record_deployment_event(payload: dict[str, Any]) -> None:
    manifest = payload.get("manifest", payload)
    with connection(CONFIG.factory_db_name) as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO deployment_events (
                    event_time, action, reason, model_name, model_version,
                    source_run_id, deployment_generation, checksum, signature,
                    status, target_stage, manifest
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    pd.to_datetime(payload.get("event_time") or manifest["issued_at"], utc=True).to_pydatetime(),
                    payload.get("action", manifest.get("action", "promote")),
                    payload.get("reason"),
                    payload["model_name"],
                    str(payload["model_version"]),
                    payload.get("source_run_id"),
                    int(payload["deployment_generation"]),
                    payload.get("checksum"),
                    payload.get("signature"),
                    payload.get("status", "requested"),
                    payload.get("target_stage", manifest.get("target_stage", "Production")),
                    json.dumps(manifest),
                ),
            )


def record_edge_sync_status(payload: dict[str, Any]) -> None:
    with connection(CONFIG.factory_db_name) as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO edge_sync_status (
                    observed_at, machine_id, deployment_generation, model_version,
                    sync_state, ota_latency_ms, notes
                ) VALUES (%s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    pd.to_datetime(payload.get("observed_at") or datetime.now(timezone.utc), utc=True).to_pydatetime(),
                    payload.get("machine_id", CONFIG.edge_machine_id),
                    int(payload["deployment_generation"]),
                    str(payload["model_version"]),
                    payload["sync_state"],
                    float(payload.get("ota_latency_ms", 0.0)),
                    payload.get("notes"),
                ),
            )


def query_frame(sql: str, params: tuple[Any, ...] | None = None) -> pd.DataFrame:
    with connection(CONFIG.factory_db_name) as conn:
        return pd.read_sql_query(sql, conn, params=params)


def load_training_dataset(limit: int | None = None) -> pd.DataFrame:
    sql = "SELECT * FROM cnc_sensor_events ORDER BY event_time DESC"
    if limit:
        sql += f" LIMIT {int(limit)}"
    frame = query_frame(sql)
    return frame.sort_values("event_time").reset_index(drop=True)


def load_recent_sensor_window(event_count: int | None = None) -> pd.DataFrame:
    limit = event_count or CONFIG.drift_window_events
    frame = query_frame(
        "SELECT * FROM cnc_sensor_events ORDER BY event_time DESC LIMIT %s",
        (limit,),
    )
    return frame.sort_values("event_time").reset_index(drop=True)


def load_recent_inference_window(limit: int = 300) -> pd.DataFrame:
    frame = query_frame(
        "SELECT * FROM inference_events ORDER BY event_time DESC LIMIT %s",
        (limit,),
    )
    return frame.sort_values("event_time").reset_index(drop=True)


def load_company_reference_training_dataset(limit: int | None = None, split: str | None = None) -> pd.DataFrame:
    sql = "SELECT * FROM company_reference_events"
    params: list[Any] = []
    if split:
        sql += " WHERE split = %s"
        params.append(split)
    sql += " ORDER BY event_time ASC"
    if limit:
        sql += " LIMIT %s"
        params.append(limit)
    frame = query_frame(sql, tuple(params) if params else None)
    return frame.sort_values("event_time").reset_index(drop=True)


def latest_drift_report() -> dict[str, Any] | None:
    frame = query_frame("SELECT * FROM drift_reports ORDER BY report_time DESC LIMIT 1")
    if frame.empty:
        return None
    record = frame.iloc[0].to_dict()
    record["feature_reports"] = json.loads(record["feature_reports"]) if isinstance(record["feature_reports"], str) else record["feature_reports"]
    return record


def latest_deployment_event() -> dict[str, Any] | None:
    frame = query_frame("SELECT * FROM deployment_events ORDER BY event_time DESC LIMIT 1")
    if frame.empty:
        return None
    record = frame.iloc[0].to_dict()
    record["manifest"] = json.loads(record["manifest"]) if isinstance(record["manifest"], str) else record["manifest"]
    return record


def recent_deployment_events(limit: int = 20) -> pd.DataFrame:
    return query_frame("SELECT * FROM deployment_events ORDER BY event_time DESC LIMIT %s", (limit,))


def recent_edge_sync_status(limit: int = 20) -> pd.DataFrame:
    return query_frame("SELECT * FROM edge_sync_status ORDER BY observed_at DESC LIMIT %s", (limit,))


def latest_event_window_bounds(limit: int = 250) -> tuple[datetime | None, datetime | None]:
    frame = load_recent_sensor_window(limit)
    if frame.empty:
        return None, None
    return pd.to_datetime(frame["event_time"], utc=True).min().to_pydatetime(), pd.to_datetime(frame["event_time"], utc=True).max().to_pydatetime()
