# Industrial MLOps Laboratory 🏭🤖 (Cloud-Fog-Edge Edition)

**IndMLOps-Lab** is a containerized Reference Implementation of a modern **Distributed Industrial AI Architecture**.

Unlike traditional monolithic setups, this project implements a hierarchical **Cloud-Fog-Edge** topology, bridging the gap between Operational Technology (OT) and IT. It creates a complete environment to simulate, ingest, train, and govern Machine Learning lifecycles in manufacturing settings using industry standards like **Kafka, Airflow, and TimescaleDB**.

---

## 🏗 Architecture Overview

The stack is organized into three logical tiers, following the ISA-95 model and Edge Computing principles:

### 1. ☁️ Cloud Tier (Orchestration & Training)
*The "Brain" of the operation. Handles heavy workloads and automation.*
* **Apache Airflow:** Orchestrates complex data pipelines (ETL) and automates model retraining based on triggers (e.g., data drift).
* **MLflow:** Centralized Model Registry and Experiment Tracking.
* **Jupyter Lab:** Data Science IDE for exploratory analysis and algorithm development.

### 2. 🌫️ Fog Tier (Data Processing & Storage)
*The "Bridge" between Edge and Cloud. Provides low-latency persistence and aggregation.*
* **TimescaleDB:** Unified database engine. Stores **MLflow/Airflow metadata** AND high-frequency **IoT Sensor Time-Series** data.
* **MinIO:** S3-compatible Object Storage for model artifacts and datasets.

### 3. 🏭 Edge Tier (Ingestion & Plant)
*The "Factory Floor". Handles real-time streams and device connectivity.*
* **Apache Kafka:** High-throughput Event Streaming backbone. Buffers massive amounts of sensor data.
* **Mosquitto:** MQTT Broker for lightweight, standard industrial telemetry.
* **Node-RED:** Simulates industrial PLCs, acts as the IoT Gateway, and visualizes the local HMI.

### 4. 🛡️ Governance (Observability)
* **Prometheus:** Scrapes metrics from the entire stack (Airflow, MLflow, System).
* **Grafana:** Visualizes business KPIs and system health.

---

## 🚀 Deployment Guide

### Prerequisites
* **Docker Desktop** installed (Allocated: **8GB+ RAM** and **4 CPUs** recommended due to Kafka/Airflow).
* **Ports Available:** 8080 (Airflow), 5000 (MLflow), 8888 (Jupyter), 1880 (Node-RED), 9092 (Kafka).

### Initial Configuration
Before running, ensure you have the `.env` file in the root of `Docker-Lab` to set versions and user IDs.

```bash
# Check your User ID (typically 501 on Mac, 1000 on Linux)
id -u 

# If needed, edit the .env file
# AIRFLOW_UID=501
```

### Build and Launch

Navigate to the Docker-Lab folder and fire up the stack:
```bash
# Build custom images (Airflow, Jupyter) and start services
docker-compose up --build -d
```

Wait 2-3 minutes for Airflow and Kafka to fully initialize.

---

## 🔌 Services & Access 

| Tier | Service | URL / Port | Credentials | Description |
| :--- | :--- | :--- | :--- | :--- |
| **Cloud** | Airflow UI | http://localhost:8080 | `admin` / `admin` | Pipeline Orchestrator |
| **Cloud** | MLflow UI | http://localhost:5000 | `N/A` | Model Registry |
| **Cloud** | Jupyter Lab | http://localhost:8888 | *(Check logs\*)* | Data Science IDE |
| **Fog** | MinIO Console | http://localhost:9001 | `admin` / `password123` | S3 Storage Browser |
| **Fog** | TimescaleDB | localhost:5432 | `admin` / `password123` | Unified Database |
| **Edge** | Node-RED | http://localhost:1880 | `N/A` | Factory Simulator & HMI |
| **Gov** | Grafana | http://localhost:3000 | `admin` / `admin` | Dashboards |

To get the Jupyter Token: docker logs ind-jupyter-lab 2>&1 | grep "token="

---
## 🧪 Usage Scenarios

### Scenario A: The "Cold" Path (Automated Training)

Instead of running notebooks manually, we use Airflow to automate the lifecycle.
1. Access Airflow (http://localhost:8080).
2. Trigger the training_pipeline DAG.
3. What happens? 
- Airflow fetches historical data from TimescaleDB.
- Trains a model using Scikit-Learn.
- Registers the new version in MLflow.
- Saves artifacts to MinIO.

### Scenario B: The "Hot" Path (Real-Time Ingestion)
Simulate a high-speed production line.
1. Open Node-RED (http://localhost:1880).
2. Sensors generate data -> Publish to MQTT.
3. Kafka Connect (or script) ingests MQTT -> Kafka Topic.
4. Node-RED or a Consumer Service reads from Kafka for real-time inference using the model loaded from MLflow.

---

### Project StructurePlaintextDocker-Lab/
```text
   ├── build/                  # Custom Dockerfiles
   │   ├── airflow/            # Airflow with ML Providers
   │   ├── jupyter/            # Notebooks with Kafka/ML libs
   │   └── mlflow/             # MLflow server
   ├── config/                 # Service Configurations
   │   ├── mosquitto.conf
   │   └── prometheus.yml
   ├── notebooks/              # Persistent storage for Jupyter & Airflow DAGs
   ├── docker-compose.yml      # The Stack Definition
   └── .env                    # Environment Variables
  ```