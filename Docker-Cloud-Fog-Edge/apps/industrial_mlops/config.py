from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


def _env_bool(name: str, default: str) -> bool:
    return _env(name, default).strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class PlatformConfig:
    timescale_host: str = _env("TIMESCALE_HOST", "timescale")
    timescale_port: int = int(_env("TIMESCALE_PORT", "5432"))
    timescale_admin_user: str = _env("TIMESCALE_ADMIN_USER", "admin")
    timescale_admin_password: str = _env("TIMESCALE_ADMIN_PASSWORD", "password123")
    airflow_db_name: str = _env("AIRFLOW_DB_NAME", "airflow_db")
    mlflow_db_name: str = _env("MLFLOW_DB_NAME", "mlflow_db")
    factory_db_name: str = _env("FACTORY_DB_NAME", "factory_db")
    mlflow_tracking_uri: str = _env("MLFLOW_TRACKING_URI", "http://mlflow:5000")
    mlflow_experiment_name: str = _env("MLFLOW_EXPERIMENT_NAME", "CNC_Tool_Breakage_Closed_Loop")
    model_name: str = _env("MLFLOW_MODEL_NAME", "cnc_tool_breakage_classifier")
    company_mlflow_experiment_name: str = _env(
        "COMPANY_MLFLOW_EXPERIMENT_NAME",
        "CNC_Tool_Breakage_Company_Reference",
    )
    company_model_name: str = _env("COMPANY_MLFLOW_MODEL_NAME", "cnc_company_reference_classifier")
    mlflow_s3_endpoint_url: str = _env("MLFLOW_S3_ENDPOINT_URL", "http://minio:9000")
    aws_access_key_id: str = _env("AWS_ACCESS_KEY_ID", "admin")
    aws_secret_access_key: str = _env("AWS_SECRET_ACCESS_KEY", "password123")
    minio_bucket_artifacts: str = _env("MINIO_ARTIFACT_BUCKET", "mlflow-bucket")
    minio_bucket_enterprise: str = _env("MINIO_ENTERPRISE_BUCKET", "enterprise-artifacts")
    minio_bucket_ota: str = _env("MINIO_OTA_BUCKET", "ota-bundles")
    mqtt_host: str = _env("MQTT_HOST", "mosquitto")
    mqtt_port: int = int(_env("MQTT_PORT", "1883"))
    mqtt_username: str = _env("MQTT_USERNAME", "")
    mqtt_password: str = _env("MQTT_PASSWORD", "")
    mqtt_sensor_topic: str = _env("MQTT_SENSOR_TOPIC", "factory/cnc/cell-01/sensors/raw")
    mqtt_company_sensor_topic: str = _env("MQTT_COMPANY_SENSOR_TOPIC", "factory/cnc/cell-01/sensors/company")
    mqtt_prediction_topic: str = _env("MQTT_PREDICTION_TOPIC", "factory/cnc/cell-01/predictions")
    mqtt_company_prediction_topic: str = _env("MQTT_COMPANY_PREDICTION_TOPIC", "factory/cnc/cell-01/predictions/company")
    mqtt_control_topic: str = _env("MQTT_CONTROL_TOPIC", "factory/cnc/cell-01/control/model")
    mqtt_heartbeat_topic: str = _env("MQTT_HEARTBEAT_TOPIC", "factory/cnc/cell-01/control/heartbeat")
    kafka_bootstrap_servers: str = _env("KAFKA_BOOTSTRAP_SERVERS", "kafka:9092")
    kafka_sensor_topic: str = _env("KAFKA_SENSOR_TOPIC", "cnc.sensor.telemetry")
    edge_machine_id: str = _env("EDGE_MACHINE_ID", "cnc-cell-01")
    edge_cell_id: str = _env("EDGE_CELL_ID", "cell-01")
    enterprise_api_key: str = _env("ENTERPRISE_API_KEY", "enterprise-demo-key")
    shared_secret: str = _env("INDUSTRIAL_SHARED_SECRET", "industrial-mlops-shared-secret")
    deployment_root: Path = Path(_env("DEPLOYMENT_ROOT", "/var/lib/industrial-mlops/edge"))
    spool_root: Path = Path(_env("SPOOL_ROOT", "/var/lib/industrial-mlops/spool"))
    edge_state_file: str = _env("EDGE_STATE_FILE", "deployment_state.json")
    edge_model_dir: str = _env("EDGE_MODEL_DIR", "current_model")
    edge_bootstrap_model_name: str = _env("EDGE_BOOTSTRAP_MODEL_NAME", "cnc_tool_breakage_classifier")
    monitoring_port_edge_inference: int = int(_env("EDGE_INFERENCE_METRICS_PORT", "8010"))
    monitoring_port_fog_bridge: int = int(_env("FOG_BRIDGE_METRICS_PORT", "8011"))
    monitoring_port_edge_sync: int = int(_env("EDGE_SYNC_METRICS_PORT", "8012"))
    monitoring_port_cnc_simulator: int = int(_env("CNC_SIMULATOR_METRICS_PORT", "8013"))
    drift_window_events: int = int(_env("DRIFT_WINDOW_EVENTS", "250"))
    training_min_events: int = int(_env("TRAINING_MIN_EVENTS", "1200"))
    bootstrap_company_reference: bool = _env_bool("BOOTSTRAP_COMPANY_REFERENCE", "true")
    simulator_profile: str = _env("SIMULATOR_PROFILE", "synthetic")
    tz_name: str = _env("TZ", "Europe/Madrid")

    @property
    def edge_state_path(self) -> Path:
        return self.deployment_root / self.edge_state_file

    @property
    def edge_model_path(self) -> Path:
        return self.deployment_root / self.edge_model_dir

    def admin_dsn(self, database: str = "postgres") -> str:
        return (
            f"host={self.timescale_host} port={self.timescale_port} "
            f"dbname={database} user={self.timescale_admin_user} "
            f"password={self.timescale_admin_password}"
        )


CONFIG = PlatformConfig()
