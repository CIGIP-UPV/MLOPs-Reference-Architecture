from __future__ import annotations

from datetime import datetime
import json
from typing import Any

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

from industrial_mlops.db import (
    latest_deployment_event,
    latest_drift_report,
    load_recent_inference_window,
    query_frame,
    recent_deployment_events,
    recent_edge_sync_status,
)
from industrial_mlops.monitoring import summarize_recent_health
from industrial_mlops.orchestration import bootstrap_platform, manual_promote, rollback_latest, run_closed_loop_cycle
from industrial_mlops.registry import get_production_version, list_model_versions
from industrial_mlops.config import CONFIG

app = FastAPI(title="Industrial MLOps Enterprise API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def frame_to_records(frame):
    return json.loads(frame.to_json(orient="records", date_format="iso"))


def dict_to_jsonable(payload: dict[str, Any] | None) -> dict[str, Any] | None:
    if payload is None:
        return None
    import pandas as pd

    return json.loads(pd.DataFrame([payload]).to_json(orient="records", date_format="iso"))[0]


class PromotionRequest(BaseModel):
    version: str
    reason: str = "Manual promotion from enterprise dashboard"


class RollbackRequest(BaseModel):
    reason: str = "Manual rollback from enterprise dashboard"


class ClosedLoopRequest(BaseModel):
    trigger: str = "manual"


@app.on_event("startup")
def startup() -> None:
    bootstrap_platform()


async def api_key_guard(x_api_key: str = Header(default="")) -> None:
    if x_api_key != CONFIG.enterprise_api_key:
        raise HTTPException(status_code=401, detail="Missing or invalid X-API-Key header.")


@app.get("/health")
def health() -> dict[str, Any]:
    inference = load_recent_inference_window(limit=250)
    return {
        "status": "ok",
        "production": dict_to_jsonable(get_production_version()),
        "recent_health": summarize_recent_health(inference),
    }


@app.get("/api/overview")
def overview() -> dict[str, Any]:
    inference = load_recent_inference_window(limit=250)
    telemetry = query_frame(
        "SELECT * FROM cnc_sensor_events ORDER BY event_time DESC LIMIT %s",
        (25,),
    )
    return {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "production": dict_to_jsonable(get_production_version()),
        "latest_deployment": dict_to_jsonable(latest_deployment_event()),
        "latest_drift": dict_to_jsonable(latest_drift_report()),
        "recent_health": summarize_recent_health(inference),
        "recent_predictions": frame_to_records(inference.tail(25)),
        "recent_telemetry": frame_to_records(telemetry.tail(25)),
        "edge_sync": frame_to_records(recent_edge_sync_status(limit=10)),
    }


@app.get("/api/models")
def models() -> dict[str, Any]:
    return {"items": list_model_versions()}


@app.get("/api/deployments")
def deployments() -> dict[str, Any]:
    frame = recent_deployment_events(limit=30)
    return {"items": frame_to_records(frame)}


@app.get("/api/drift")
def drift() -> dict[str, Any]:
    frame = query_frame("SELECT * FROM drift_reports ORDER BY report_time DESC LIMIT %s", (20,))
    return {"items": frame_to_records(frame)}


@app.get("/api/predictions/recent")
def predictions() -> dict[str, Any]:
    frame = load_recent_inference_window(limit=250)
    return {"items": frame_to_records(frame)}


@app.post("/api/deployments/promote")
def promote(request: PromotionRequest, _auth: None = Depends(api_key_guard)) -> dict[str, Any]:
    return manual_promote(request.version, request.reason)


@app.post("/api/deployments/rollback")
def rollback(request: RollbackRequest, _auth: None = Depends(api_key_guard)) -> dict[str, Any]:
    return rollback_latest(request.reason, issued_by="enterprise-api")


@app.post("/api/closed-loop/run")
def closed_loop(request: ClosedLoopRequest, _auth: None = Depends(api_key_guard)) -> dict[str, Any]:
    return run_closed_loop_cycle(trigger=request.trigger)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8085)
