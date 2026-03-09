from __future__ import annotations

from datetime import datetime

from airflow.decorators import dag, task

from industrial_mlops.orchestration import bootstrap_platform

DAG_VERSION = "1.0.0"


@dag(
    dag_id="cnc_bootstrap_reference_architecture_v1",
    description="Bootstrap databases, seed CNC history and issue the first OTA deployment.",
    start_date=datetime(2026, 1, 1),
    schedule=None,
    catchup=False,
    tags=["cnc", "bootstrap", DAG_VERSION],
)
def cnc_bootstrap_reference_architecture_v1():
    @task
    def bootstrap() -> dict:
        return bootstrap_platform()

    bootstrap()


cnc_bootstrap_reference_architecture_v1()
