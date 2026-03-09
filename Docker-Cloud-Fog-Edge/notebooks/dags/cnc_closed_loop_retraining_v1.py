from __future__ import annotations

from datetime import datetime, timedelta

from airflow.decorators import dag, task
from airflow.operators.empty import EmptyOperator

from industrial_mlops.db import load_recent_inference_window
from industrial_mlops.monitoring import summarize_recent_health
from industrial_mlops.orchestration import bootstrap_platform, evaluate_recent_drift, train_and_promote_candidate

DAG_VERSION = "1.0.0"


@dag(
    dag_id="cnc_closed_loop_retraining_v1",
    description="Versioned DAG for drift evaluation and automatic retraining/promotions.",
    start_date=datetime(2026, 1, 1),
    schedule=timedelta(minutes=30),
    catchup=False,
    tags=["cnc", "closed-loop", "drift", DAG_VERSION],
)
def cnc_closed_loop_retraining_v1():
    @task
    def bootstrap() -> dict:
        return bootstrap_platform()

    @task
    def drift() -> dict:
        return evaluate_recent_drift()

    @task.branch
    def route_retraining(report: dict) -> str:
        if report["overall"]["requires_retraining"]:
            return "retrain_and_promote"
        return "skip_retraining"

    @task(task_id="retrain_and_promote")
    def retrain_and_promote() -> dict:
        return train_and_promote_candidate("scheduled-closed-loop-retraining", issued_by="airflow-closed-loop")

    @task
    def recent_health() -> dict:
        frame = load_recent_inference_window(limit=300)
        return summarize_recent_health(frame)

    skip_retraining = EmptyOperator(task_id="skip_retraining")

    bootstrap_result = bootstrap()
    drift_result = drift()
    route = route_retraining(drift_result)
    retrain = retrain_and_promote()
    health_result = recent_health()
    bootstrap_result >> drift_result
    drift_result >> route
    route >> [retrain, skip_retraining]
    [retrain, skip_retraining] >> health_result


cnc_closed_loop_retraining_v1()
