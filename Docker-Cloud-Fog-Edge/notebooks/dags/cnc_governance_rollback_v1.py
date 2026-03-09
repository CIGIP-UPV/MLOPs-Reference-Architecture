from __future__ import annotations

from datetime import datetime, timedelta

from airflow.decorators import dag, task
from airflow.operators.empty import EmptyOperator

from industrial_mlops.db import load_recent_inference_window
from industrial_mlops.monitoring import summarize_recent_health
from industrial_mlops.orchestration import bootstrap_platform, rollback_latest

DAG_VERSION = "1.0.0"


@dag(
    dag_id="cnc_governance_rollback_v1",
    description="Continuous governance DAG for post-deployment health checks and automatic rollback.",
    start_date=datetime(2026, 1, 1),
    schedule=timedelta(minutes=15),
    catchup=False,
    tags=["cnc", "governance", "rollback", DAG_VERSION],
)
def cnc_governance_rollback_v1():
    @task
    def bootstrap() -> dict:
        return bootstrap_platform()

    @task
    def health_snapshot() -> dict:
        return summarize_recent_health(load_recent_inference_window(limit=300))

    @task.branch
    def rollback_route(snapshot: dict) -> str:
        if snapshot["rollback_recommended"]:
            return "rollback_task"
        return "skip_rollback"

    @task(task_id="rollback_task")
    def rollback_task() -> dict:
        return rollback_latest("automatic-governance-rollback", issued_by="airflow-governance")

    skip_rollback = EmptyOperator(task_id="skip_rollback")

    boot = bootstrap()
    snapshot = health_snapshot()
    route = rollback_route(snapshot)
    rollback = rollback_task()
    boot >> snapshot
    snapshot >> route
    route >> [rollback, skip_rollback]


cnc_governance_rollback_v1()
