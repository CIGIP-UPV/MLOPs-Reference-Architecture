from __future__ import annotations

from typing import Any

import boto3
import time

from .config import CONFIG
from .db import (
    ensure_platform_databases,
    insert_drift_report,
    latest_deployment_event,
    latest_event_window_bounds,
    load_company_reference_training_dataset,
    load_recent_inference_window,
    load_recent_sensor_window,
    load_training_dataset,
    recent_deployment_events,
    record_deployment_event,
    seed_company_reference_data_if_empty,
    seed_historical_data_if_empty,
)
from .deployment import build_manifest, publish_control_command
from .drift import compute_drift_report
from .monitoring import summarize_recent_health
from .registry import (
    get_checksum_for_version,
    get_previous_non_current_production,
    list_model_versions,
    get_production_version,
    get_reference_profile_for_version,
    get_training_summary_for_version,
    mark_version_deployed,
    synchronize_checksum_for_version,
    train_and_register,
    train_and_register_company_reference,
    transition_model_version,
)


def next_generation() -> int:
    latest = latest_deployment_event()
    if not latest:
        return 1
    return int(latest["deployment_generation"]) + 1


def ensure_object_buckets() -> None:
    deadline = time.time() + 60
    while True:
        try:
            client = boto3.client(
                "s3",
                endpoint_url=CONFIG.mlflow_s3_endpoint_url,
                aws_access_key_id=CONFIG.aws_access_key_id,
                aws_secret_access_key=CONFIG.aws_secret_access_key,
            )
            existing = {bucket["Name"] for bucket in client.list_buckets().get("Buckets", [])}
            for bucket_name in (
                CONFIG.minio_bucket_artifacts,
                CONFIG.minio_bucket_enterprise,
                CONFIG.minio_bucket_ota,
            ):
                if bucket_name not in existing:
                    client.create_bucket(Bucket=bucket_name)
            return
        except Exception:
            if time.time() >= deadline:
                raise
            time.sleep(2)


def _publish_deployment(
    *,
    action: str,
    model_name: str,
    model_version: str,
    reason: str,
    issued_by: str,
    source_run_id: str | None,
    previous_version: str | None,
) -> dict[str, Any]:
    generation = next_generation()
    checksum = get_checksum_for_version(model_version, model_name=model_name)
    manifest = build_manifest(
        action=action,
        model_name=model_name,
        model_version=model_version,
        checksum=checksum,
        generation=generation,
        source_run_id=source_run_id,
        reason=reason,
        issued_by=issued_by,
        previous_version=previous_version,
    )
    publish_control_command(manifest)
    record_deployment_event(
        {
            **manifest,
            "event_time": manifest["issued_at"],
            "status": "command-issued",
            "reason": reason,
            "manifest": manifest,
        }
    )
    mark_version_deployed(model_version, generation, model_name=model_name)
    return manifest


def bootstrap_platform(seed_samples: int = 2500) -> dict[str, Any]:
    ensure_platform_databases()
    ensure_object_buckets()
    seeded = seed_historical_data_if_empty(seed_samples)
    company_seeded = seed_company_reference_data_if_empty()
    production = get_production_version()
    manifest = None
    if production is None:
        frame = load_training_dataset(limit=max(CONFIG.training_min_events, 1800))
        candidate = train_and_register(frame, reason="bootstrap-baseline", run_name="bootstrap-baseline")
        transition_model_version(candidate["model_version"], "Production", archive_existing_versions=True)
        manifest = _publish_deployment(
        action="promote",
        model_name=CONFIG.model_name,
        model_version=candidate["model_version"],
        reason="Initial baseline deployment for CNC tool breakage.",
        issued_by="bootstrap-service",
            source_run_id=candidate["run_id"],
            previous_version=None,
        )
        production = {"version": candidate["model_version"], "run_id": candidate["run_id"], "tags": {}}
    synchronize_checksum_for_version(production["version"], model_name=CONFIG.model_name)
    company_reference = None
    if CONFIG.bootstrap_company_reference and company_seeded:
        company_reference = get_production_version(CONFIG.company_model_name)
        if company_reference is None and not list_model_versions(CONFIG.company_model_name):
            company_frame = load_company_reference_training_dataset()
            if not company_frame.empty:
                company_candidate = train_and_register_company_reference(
                    company_frame,
                    reason="company-reference-bootstrap",
                    run_name="company-reference-bootstrap",
                )
                transition_model_version(
                    company_candidate["model_version"],
                    "Production",
                    archive_existing_versions=True,
                    model_name=CONFIG.company_model_name,
                )
                company_reference = {
                    "version": company_candidate["model_version"],
                    "run_id": company_candidate["run_id"],
                    "tags": {"dataset_name": "company-nakamura-reference"},
                }
        if company_reference is not None:
            synchronize_checksum_for_version(company_reference["version"], model_name=CONFIG.company_model_name)
    return {
        "seeded_events": seeded,
        "seeded_company_reference_events": company_seeded,
        "production_version": production["version"],
        "company_reference_version": company_reference["version"] if company_reference else None,
        "initial_manifest": manifest,
    }


def evaluate_recent_drift() -> dict[str, Any]:
    production = get_production_version()
    if production is None:
        raise RuntimeError("No production model is available for drift evaluation.")
    current_window = load_recent_sensor_window(CONFIG.drift_window_events)
    reference_profile = get_reference_profile_for_version(production["version"])
    report = compute_drift_report(reference_profile, current_window)
    window_start, window_end = latest_event_window_bounds(CONFIG.drift_window_events)
    insert_drift_report(report, production["version"], window_start, window_end)
    return report


def _candidate_beats_champion(candidate_metrics: dict[str, float], champion_metrics: dict[str, float] | None) -> bool:
    if not champion_metrics:
        return True
    weighted_candidate = candidate_metrics["f1"] * 0.45 + candidate_metrics["recall"] * 0.35 + candidate_metrics["roc_auc"] * 0.20
    weighted_champion = champion_metrics["f1"] * 0.45 + champion_metrics["recall"] * 0.35 + champion_metrics["roc_auc"] * 0.20
    return weighted_candidate >= weighted_champion - 0.01 and candidate_metrics["recall"] >= champion_metrics["recall"] - 0.02


def train_and_promote_candidate(reason: str, issued_by: str = "airflow-retraining") -> dict[str, Any]:
    frame = load_training_dataset(limit=5000)
    if len(frame) < CONFIG.training_min_events:
        return {"status": "skipped", "reason": f"Only {len(frame)} training events available."}
    production = get_production_version()
    champion_metrics = None
    previous_version = None
    if production:
        champion_metrics = get_training_summary_for_version(production["version"])["metrics"]
        previous_version = production["version"]
    candidate = train_and_register(frame, reason=reason, run_name=reason)
    if not _candidate_beats_champion(candidate["metrics"], champion_metrics):
        transition_model_version(candidate["model_version"], "Archived")
        return {
            "status": "rejected",
            "candidate_version": candidate["model_version"],
            "candidate_metrics": candidate["metrics"],
            "champion_metrics": champion_metrics,
        }
    transition_model_version(candidate["model_version"], "Production", archive_existing_versions=True)
    manifest = _publish_deployment(
        action="promote",
        model_name=CONFIG.model_name,
        model_version=candidate["model_version"],
        reason=reason,
        issued_by=issued_by,
        source_run_id=candidate["run_id"],
        previous_version=previous_version,
    )
    return {
        "status": "promoted",
        "candidate_version": candidate["model_version"],
        "candidate_metrics": candidate["metrics"],
        "manifest": manifest,
    }


def find_last_stable_version(current_version: str | None) -> dict[str, Any] | None:
    events = recent_deployment_events(limit=40)
    if events.empty:
        return get_previous_non_current_production(current_version)
    for _, row in events.iterrows():
        if str(row["model_version"]) != str(current_version) and row["action"] in {"promote", "rollback"}:
            return {"version": str(row["model_version"]), "run_id": row.get("source_run_id")}
    return get_previous_non_current_production(current_version)


def rollback_latest(reason: str, issued_by: str = "airflow-governance") -> dict[str, Any]:
    production = get_production_version()
    current_version = production["version"] if production else None
    previous = find_last_stable_version(current_version)
    if previous is None:
        return {"status": "skipped", "reason": "No previous stable version available."}
    transition_model_version(previous["version"], "Production", archive_existing_versions=True)
    manifest = _publish_deployment(
        action="rollback",
        model_name=CONFIG.model_name,
        model_version=previous["version"],
        reason=reason,
        issued_by=issued_by,
        source_run_id=previous.get("run_id"),
        previous_version=current_version,
    )
    return {"status": "rolled-back", "target_version": previous["version"], "manifest": manifest}


def governance_snapshot() -> dict[str, Any]:
    drift_report = evaluate_recent_drift()
    health = summarize_recent_health(load_recent_inference_window())
    return {
        "drift": drift_report,
        "health": health,
        "production": get_production_version(),
    }


def run_closed_loop_cycle(trigger: str = "scheduled") -> dict[str, Any]:
    bootstrap_summary = bootstrap_platform()
    drift_report = evaluate_recent_drift()
    health = summarize_recent_health(load_recent_inference_window())
    actions: list[dict[str, Any]] = []
    if drift_report["overall"]["requires_retraining"]:
        actions.append(
            train_and_promote_candidate(
                reason=f"drift-triggered-retraining:{trigger}:{drift_report['overall']['severity']}",
                issued_by="airflow-closed-loop",
            )
        )
    if health["rollback_recommended"]:
        actions.append(
            rollback_latest(
                reason=f"automatic-rollback:{trigger}:accuracy={health['accuracy']}:drift={health['average_drift']}",
                issued_by="airflow-governance",
            )
        )
    return {
        "bootstrap": bootstrap_summary,
        "drift": drift_report,
        "health": health,
        "actions": actions,
    }


def manual_promote(version: str, reason: str, issued_by: str = "enterprise-api") -> dict[str, Any]:
    production = get_production_version()
    previous_version = production["version"] if production else None
    transition_model_version(version, "Production", archive_existing_versions=True)
    details = get_training_summary_for_version(version)
    manifest = _publish_deployment(
        action="promote",
        model_name=CONFIG.model_name,
        model_version=version,
        reason=reason,
        issued_by=issued_by,
        source_run_id=None,
        previous_version=previous_version,
    )
    return {"status": "promoted", "version": version, "metrics": details["metrics"], "manifest": manifest}
