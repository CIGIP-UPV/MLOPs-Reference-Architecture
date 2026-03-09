# Industrial MLOps Laboratory 🏭🤖 (Cloud-Fog-Edge Edition)

**IndMLOps-Lab** is a containerized reference implementation of a modern **Distributed Industrial AI Architecture**.

Unlike traditional monolithic setups, this project implements a hierarchical **Cloud-Fog-Edge** topology, bridging the gap between Operational Technology (OT) and IT. It creates a complete environment to simulate, ingest, train, deploy, monitor and govern Machine Learning lifecycles in manufacturing settings using industry standards like **Kafka, Airflow, MLflow and TimescaleDB**.

## What Is New in This Version

This Docker flavor now includes the implementation gaps that were previously only described in the article:

- Versioned **Airflow DAGs** for bootstrap, retraining and rollback.
- A real **closed-loop CNC tool-breakage** lifecycle with historical seeding and live telemetry.
- A compact **company Nakamura 2 reference dataset** curated from the industrial archive.
- A parallel **company reference model bootstrap** in MLflow and Timescale, isolated from the edge demo path.
- A **Fog Bridge** for MQTT -> Kafka -> Timescale persistence with spool-based buffering.
- **Edge Sync** for signed OTA deployment manifests and checksum verification.
- **Edge Inference** with local low-latency scoring and Prometheus metrics.
- An **Enterprise API** plus **React dashboard** for supervisory control.
- Explicit **promotion / rollback** mechanics recorded in Timescale and MLflow.

---

## 🏗 Architecture Overview

The stack is organized into five logical tiers, following ISA-95 ideas and edge-computing principles:

### 1. ☁️ Cloud Tier (Orchestration & Training)
*The brain of the operation. Handles heavy workloads and lifecycle automation.*

- **Apache Airflow:** orchestrates bootstrap, drift-triggered retraining and rollback governance.
- **MLflow:** centralized model registry and experiment tracking.
- **Jupyter Lab:** Data Science IDE for exploratory analysis and algorithm development.

### 2. 🌫️ Fog Tier (Data Processing & Storage)
*The bridge between Edge and Cloud. Provides low-latency persistence and aggregation.*

- **TimescaleDB:** unified storage for MLflow/Airflow metadata, CNC telemetry, inference logs and deployment events.
- **MinIO:** S3-compatible object storage for artifacts, governance reports and OTA bundles.
- **Fog Bridge:** subscribes to MQTT, persists to Timescale, forwards to Kafka and buffers locally if downstream services fail.

### 3. 🏭 Edge Tier (Ingestion & Plant)
*The factory floor. Handles real-time telemetry and low-latency inference.*

- **Mosquitto:** MQTT broker for industrial telemetry.
- **Node-RED:** local HMI and OT-side integration point.
- **Kafka:** event streaming backbone.
- **CNC Simulator:** generates single-cell machining telemetry with tool wear and breakage labels.
- **Edge Sync:** applies signed OTA deployment manifests and verifies model checksums.
- **Edge Inference:** executes the production model close to the machine and publishes predictions.

### 4. 🛡️ Governance Tier (Observability)

- **Prometheus:** scrapes metrics from simulator, bridge, edge services and platform components.
- **Grafana:** visualizes system health and operational telemetry.

### 5. 🏢 Enterprise Tier (Supervision)

- **Enterprise API:** lifecycle control endpoints for promotion, rollback and manual closed-loop execution.
- **Enterprise UI:** reviewer-facing dashboard for versions, drift, deployments and prediction stream.
- **Optional Rancher profile:** infrastructure governance hook for enterprise-like demos.

---

## 🚀 Deployment Guide

### Prerequisites

- **Docker Desktop** installed.
- Recommended allocation: **8 GB RAM** and **4 CPUs** minimum.
- Available ports: `8080`, `5000`, `8888`, `1880`, `1883`, `3000`, `8085`, `8088`, `9001`, `9090`, `9092`.

### Initial Configuration

Before running, ensure you have the `.env` file in this folder to set versions and user IDs.

```bash
# Check your User ID (typically 501 on macOS, 1000 on Linux)
id -u

# If needed, edit the .env file
# AIRFLOW_UID=501
```

### Build and Launch

Navigate to `Docker-Cloud-Fog-Edge` and start the stack:

```bash
docker compose up --build -d
```

The first bootstrap can take a few minutes because it builds the custom images, initializes TimescaleDB/MinIO, seeds historical CNC data and registers the baseline model.

The repository also includes a compact company dataset reference at `apps/industrial_mlops/data/company_cnc/`. The raw corporate archive is intentionally not committed.

### First Validation Commands

```bash
docker compose ps
docker compose logs db-bootstrap
docker compose logs edge-sync
docker compose logs cnc-simulator
```

---

## 🔌 Services & Access

| Tier | Service | URL / Port | Credentials | Description |
| :--- | :--- | :--- | :--- | :--- |
| **Cloud** | Airflow UI | http://localhost:8080 | `admin` / `admin` | Pipeline Orchestrator |
| **Cloud** | MLflow UI | http://localhost:5000 | `N/A` | Model Registry |
| **Cloud** | Jupyter Lab | http://localhost:8888 | check container logs | Data Science IDE |
| **Fog** | MinIO Console | http://localhost:9001 | `admin` / `password123` | Object Storage Browser |
| **Fog** | TimescaleDB | `localhost:5432` | `admin` / `password123` | Unified Database |
| **Edge** | Node-RED | http://localhost:1880 | `N/A` | OT/HMI Sandbox |
| **Gov** | Grafana | http://localhost:3000 | `admin` / `admin` | Dashboards |
| **Enterprise** | Enterprise API | http://localhost:8085 | `X-API-Key: enterprise-demo-key` | Governance API |
| **Enterprise** | Enterprise UI | http://localhost:8088 | `N/A` | Supervisory Dashboard |
| **Enterprise (optional)** | Rancher | https://localhost:8443 | bootstrap on first run | Optional infra governance profile |

To get the Jupyter token:

```bash
docker logs ind-jupyter-lab 2>&1 | grep "token="
```

To run the optional Rancher profile:

```bash
docker compose --profile enterprise-ops up -d rancher
```

---

## 🧪 Main Scenarios

### Scenario A: Bootstrap and Initial Deployment

1. `db-bootstrap` creates the required databases and buckets.
2. Historical CNC telemetry is seeded into TimescaleDB.
3. The curated company reference dataset is seeded into `company_reference_events`.
4. A deployable synthetic baseline model is trained and registered in MLflow.
5. A company-reference baseline model is trained in parallel under a separate MLflow model name.
6. A signed OTA manifest is published only for the deployable synthetic model.
7. `edge-sync` downloads and verifies the production edge model.

### Scenario B: Hot Path Inference

1. `cnc-simulator` publishes CNC telemetry to MQTT.
2. `fog-bridge` persists telemetry in TimescaleDB and forwards it to Kafka.
3. `edge-inference` consumes the telemetry, scores locally and publishes predictions.
4. Metrics become visible in Prometheus/Grafana and the Enterprise UI.

### Scenario C: Closed-Loop Retraining

1. Airflow DAG `cnc_closed_loop_retraining_v1` computes drift using recent telemetry windows.
2. If drift exceeds thresholds, a candidate model is trained and registered.
3. Governance criteria compare the candidate with the current champion.
4. If accepted, the new version is promoted and a signed deployment command is issued.

### Scenario D: Governance and Rollback

1. Airflow DAG `cnc_governance_rollback_v1` evaluates recent post-deployment health.
2. If the deployed model degrades, a rollback command is generated.
3. `edge-sync` applies the previous stable version and records the action.

---

## 📂 Project Structure

```text
Docker-Cloud-Fog-Edge/
├── apps/
│   ├── industrial_mlops/      # Shared lifecycle, drift, security and registry logic
│   │   └── data/company_cnc/  # Compact curated Nakamura 2 reference dataset
│   └── services/              # Runtime services: simulator, bridge, edge, enterprise API
├── build/
│   ├── airflow/               # Airflow image with DAGs and industrial dependencies
│   ├── enterprise-ui/         # React UI build stage
│   ├── jupyter/               # Jupyter image
│   ├── mlflow/                # MLflow server image
│   └── python-service/        # Generic runtime image for services
├── config/
│   ├── initdb.sql             # Initial databases for TimescaleDB
│   ├── mosquitto.conf         # MQTT broker config
│   └── prometheus.yml         # Metrics scrape targets
├── enterprise-ui/             # React source code for the Enterprise Tier
├── notebooks/
│   ├── dags/                  # Versioned Airflow DAGs
│   ├── CompanyDatasetTraining.ipynb # Reproducible company-dataset training notebook
│   ├── logs/                  # Airflow logs
│   ├── plugins/               # Airflow plugins placeholder
│   ├── Prediction.ipynb       # Legacy notebook kept for continuity
│   ├── TestConection.ipynb    # Legacy connectivity notebook
│   └── Training.ipynb         # Legacy training notebook
├── docker-compose.yml         # Complete distributed stack definition
├── scripts/                   # Utility scripts, including company-dataset curation
└── .env                       # Environment variables
```

---

## Legacy Notes Kept for Continuity

The original intent of this folder remains valid: a fast, local, containerized environment for Cloud-Fog-Edge industrial AI experimentation. The earlier notebooks are intentionally preserved and still coexist with the new code-based services.
