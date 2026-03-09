from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import tempfile
import time
from typing import Any

import mlflow
import mlflow.sklearn
from mlflow import MlflowClient
from mlflow.entities.model_registry.model_version_status import ModelVersionStatus
from mlflow.exceptions import MlflowException

from .cnc_data import FEATURE_COLUMNS, TARGET_COLUMN
from .company_dataset import COMPANY_FEATURE_COLUMNS, COMPANY_TARGET_COLUMN
from .config import CONFIG
from .ml import build_training_summary, train_model, write_json_artifact
from .security import compute_directory_digest


def _client() -> MlflowClient:
    mlflow.set_tracking_uri(CONFIG.mlflow_tracking_uri)
    return MlflowClient(tracking_uri=CONFIG.mlflow_tracking_uri)


def ensure_experiment(experiment_name: str | None = None) -> str:
    experiment_name = experiment_name or CONFIG.mlflow_experiment_name
    client = _client()
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment:
        return experiment.experiment_id
    return client.create_experiment(experiment_name)


def wait_until_version_ready(client: MlflowClient, version: str, timeout_seconds: int = 60, model_name: str | None = None) -> None:
    model_name = model_name or CONFIG.model_name
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        details = client.get_model_version(model_name, str(version))
        if details.status == ModelVersionStatus.to_string(ModelVersionStatus.READY):
            return
        time.sleep(1)
    raise TimeoutError(f"Model version {version} did not reach READY state in time.")


def train_and_register(
    frame,
    reason: str,
    run_name: str | None = None,
    *,
    feature_columns: list[str] | None = None,
    target_column: str = TARGET_COLUMN,
    experiment_name: str | None = None,
    model_name: str | None = None,
    dataset_name: str = "synthetic-cnc",
) -> dict[str, Any]:
    experiment_name = experiment_name or CONFIG.mlflow_experiment_name
    model_name = model_name or CONFIG.model_name
    feature_columns = list(feature_columns or FEATURE_COLUMNS)
    ensure_experiment(experiment_name)
    mlflow.set_tracking_uri(CONFIG.mlflow_tracking_uri)
    mlflow.set_experiment(experiment_name)
    result = train_model(frame, feature_columns=feature_columns, target_column=target_column)
    temp_dir = Path(tempfile.mkdtemp(prefix="cnc-artifacts-"))
    reference_profile_path = temp_dir / "reference_profile.json"
    training_summary_path = temp_dir / "training_summary.json"
    schema_path = temp_dir / "feature_schema.json"
    write_json_artifact(reference_profile_path, result.reference_profile)
    write_json_artifact(training_summary_path, build_training_summary(result, frame, reason, dataset_name=dataset_name))
    write_json_artifact(schema_path, {"features": feature_columns, "target": target_column, "dataset_name": dataset_name})

    with mlflow.start_run(run_name=run_name or f"cnc-{reason.replace(' ', '-').lower()}") as run:
        mlflow.log_params(
            {
                "model_family": "HistGradientBoostingClassifier",
                "reason": reason,
                "dataset_name": dataset_name,
                "rows": len(frame),
                "feature_count": len(feature_columns),
                "target_column": target_column,
            }
        )
        mlflow.log_metrics(result.metrics)
        mlflow.log_text(result.checksum, "model_checksum.txt")
        mlflow.log_artifact(str(reference_profile_path), artifact_path="governance")
        mlflow.log_artifact(str(training_summary_path), artifact_path="governance")
        mlflow.log_artifact(str(schema_path), artifact_path="governance")
        mlflow.sklearn.log_model(result.model, artifact_path="model")
        model_version = mlflow.register_model(model_uri=f"runs:/{run.info.run_id}/model", name=model_name)
    client = _client()
    wait_until_version_ready(client, model_version.version, model_name=model_name)
    stored_model_path = download_registered_model_artifact(model_version.version, model_name=model_name)
    stored_checksum = compute_directory_digest(stored_model_path)
    client.set_model_version_tag(model_name, model_version.version, "sha256", stored_checksum)
    client.set_model_version_tag(model_name, model_version.version, "reason", reason)
    client.set_model_version_tag(model_name, model_version.version, "dataset_name", dataset_name)
    for metric_name, metric_value in result.metrics.items():
        client.set_model_version_tag(model_name, model_version.version, f"metric_{metric_name}", str(metric_value))
    return {
        "run_id": run.info.run_id,
        "model_version": str(model_version.version),
        "model_name": model_name,
        "metrics": result.metrics,
        "checksum": stored_checksum,
        "reference_profile": result.reference_profile,
    }


def train_and_register_company_reference(frame, reason: str, run_name: str | None = None) -> dict[str, Any]:
    return train_and_register(
        frame,
        reason=reason,
        run_name=run_name,
        feature_columns=COMPANY_FEATURE_COLUMNS,
        target_column=COMPANY_TARGET_COLUMN,
        experiment_name=CONFIG.company_mlflow_experiment_name,
        model_name=CONFIG.company_model_name,
        dataset_name="company-nakamura-reference",
    )


def list_model_versions(model_name: str | None = None) -> list[dict[str, Any]]:
    model_name = model_name or CONFIG.model_name
    client = _client()
    try:
        versions = client.search_model_versions(f"name='{model_name}'")
    except MlflowException:
        return []
    payload: list[dict[str, Any]] = []
    for version in versions:
        payload.append(
            {
                "name": version.name,
                "version": version.version,
                "current_stage": version.current_stage,
                "run_id": version.run_id,
                "creation_timestamp": version.creation_timestamp,
                "last_updated_timestamp": version.last_updated_timestamp,
                "description": version.description,
                "source": version.source,
                "tags": dict(version.tags or {}),
            }
        )
    return sorted(payload, key=lambda item: int(item["version"]), reverse=True)


def get_version(version: str, model_name: str | None = None) -> dict[str, Any]:
    model_name = model_name or CONFIG.model_name
    client = _client()
    details = client.get_model_version(model_name, str(version))
    return {
        "name": details.name,
        "version": details.version,
        "current_stage": details.current_stage,
        "run_id": details.run_id,
        "source": details.source,
        "tags": dict(details.tags or {}),
    }


def get_production_version(model_name: str | None = None) -> dict[str, Any] | None:
    model_name = model_name or CONFIG.model_name
    client = _client()
    try:
        versions = client.get_latest_versions(model_name, stages=["Production"])
    except MlflowException:
        return None
    if not versions:
        return None
    chosen = versions[0]
    return get_version(chosen.version, model_name=model_name)


def get_previous_non_current_production(current_version: str | None, model_name: str | None = None) -> dict[str, Any] | None:
    versions = list_model_versions(model_name=model_name)
    for version in versions:
        if version["version"] != str(current_version) and version["current_stage"] in {"Production", "Archived", "Staging"}:
            return version
    return None


def transition_model_version(
    version: str,
    stage: str,
    archive_existing_versions: bool = False,
    model_name: str | None = None,
) -> dict[str, Any]:
    model_name = model_name or CONFIG.model_name
    client = _client()
    client.transition_model_version_stage(
        name=model_name,
        version=str(version),
        stage=stage,
        archive_existing_versions=archive_existing_versions,
    )
    return get_version(version, model_name=model_name)


def download_run_artifact(run_id: str, artifact_path: str) -> Path:
    client = _client()
    downloaded = client.download_artifacts(run_id, artifact_path)
    return Path(downloaded)


def download_registered_model_artifact(version: str, model_name: str | None = None) -> Path:
    model_name = model_name or CONFIG.model_name
    mlflow.set_tracking_uri(CONFIG.mlflow_tracking_uri)
    destination = Path(tempfile.mkdtemp(prefix="mlflow-registered-model-"))
    downloaded = mlflow.artifacts.download_artifacts(
        artifact_uri=f"models:/{model_name}/{version}",
        dst_path=str(destination),
    )
    return Path(downloaded)


def get_reference_profile_for_version(version: str, model_name: str | None = None) -> dict[str, Any]:
    details = get_version(version, model_name=model_name)
    artifact_path = download_run_artifact(details["run_id"], "governance/reference_profile.json")
    return json.loads(artifact_path.read_text(encoding="utf-8"))


def get_training_summary_for_version(version: str, model_name: str | None = None) -> dict[str, Any]:
    details = get_version(version, model_name=model_name)
    artifact_path = download_run_artifact(details["run_id"], "governance/training_summary.json")
    return json.loads(artifact_path.read_text(encoding="utf-8"))


def get_feature_schema_for_version(version: str, model_name: str | None = None) -> dict[str, Any]:
    details = get_version(version, model_name=model_name)
    artifact_path = download_run_artifact(details["run_id"], "governance/feature_schema.json")
    return json.loads(artifact_path.read_text(encoding="utf-8"))


def get_checksum_for_version(version: str, model_name: str | None = None) -> str | None:
    details = get_version(version, model_name=model_name)
    tags = details.get("tags", {})
    return tags.get("sha256")


def synchronize_checksum_for_version(version: str, model_name: str | None = None) -> str:
    artifact_path = download_registered_model_artifact(version, model_name=model_name)
    checksum = compute_directory_digest(artifact_path)
    set_version_tag(version, "sha256", checksum, model_name=model_name)
    return checksum


def set_version_tag(version: str, key: str, value: str, model_name: str | None = None) -> None:
    model_name = model_name or CONFIG.model_name
    client = _client()
    client.set_model_version_tag(model_name, str(version), key, value)


def mark_version_deployed(version: str, generation: int, model_name: str | None = None) -> None:
    timestamp = datetime.now(timezone.utc).isoformat()
    set_version_tag(version, "last_deployed_at", timestamp, model_name=model_name)
    set_version_tag(version, "deployment_generation", str(generation), model_name=model_name)
