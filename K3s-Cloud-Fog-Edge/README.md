# Kubernetes Industrial MLOps Lab ☸️ (Cloud-Fog-Edge)

This folder contains the **Kubernetes Manifests** to deploy a distributed **Cloud-Fog-Edge** architecture for Industrial AI.

Unlike the flat Docker Compose setup, this deployment models a real-world production environment using **Namespaces** to isolate the Cloud (Datacenter), Fog (Regional/On-Prem), and Edge (Factory Floor) layers.

---

## 🏗 Architecture Layers

The cluster is divided into 4 logical tiers:

| Tier | Namespace | Key Components                 | Role |
| :--- | :--- |:-------------------------------| :--- |
| **1. Cloud** | `cloud-tier` | **Airflow, MLflow**            | Orchestration, Heavy Training, Model Registry. |
| **2. Fog** | `fog-tier` | **TimescaleDB, MinIO**         | Persistence Layer. Unifies Metadata & Time-Series storage. |
| **3. Edge** | `edge-tier` | **Kafka, Node-RED, Mosquitto** | High-throughput ingestion & Plant Simulation. |
| **4. Gov** | `governance` | **Prometheus, Grafana**        | Cross-tier Observability & Dashboards. |

---

## 📂 Directory Structure

The manifests are organized by execution order and tier:

```text
k8-manifests/
├── 00-base/              # 1. Namespaces & Persistent Volume Claims (PVCs)
├── 01-fog-tier/          # 2. Data Layer (Must start first for DB)
│   ├── timescale.yaml    # Unified Database
│   └── minio.yaml        # Artifact Store
├── 02-cloud-tier/        # 3. AI Core
│   ├── airflow.yaml      # Pipeline Orchestrator
│   ├── mlflow.yaml       # Registry
├── 03-edge-tier/         # 4. Factory Floor
│   ├── kafka.yaml        # Streaming Backbone
│   ├── nodered.yaml      # Simulator
│   └── mosquitto.yaml    # MQTT Broker
└── 04-governance/        # 5. Monitoring tools
```

# 🚀 Deployment Guide

### Prerequisites
* **Kubernetes Cluster** (Docker Desktop, Minikube, K3s, or Cloud).
* `kubectl` CLI configured.
* **Resources:** Ensure your cluster has at least **4 CPUs** and **8GB RAM** allocated.

---

## Step-by-Step Installation

Run the commands in this specific order to ensure dependencies (like the Database) are ready.

**1. Base Infrastructure**
Create the namespaces and storage claims.
```bash
kubectl apply -f k8-manifests/00-base/
```

**2. Fog Tier (Data Layer)**
Deploy the storage engines.
```bash
kubectl apply -f k8-manifests/01-fog-tier/
```
> **⏳ WAIT:** Wait 1 minute here to ensure TimescaleDB is fully running before starting Airflow.

**3. Cloud Tier (The Brain)**
Deploy the AI services.
```bash
kubectl apply -f k8-manifests/02-cloud-tier/
```

**4. Edge Tier (The Plant)**
Deploy the ingestion services.
```bash
kubectl apply -f k8-manifests/03-edge-tier/
```

**5. Governance**
Deploy monitoring.
```bash
kubectl apply -f k8-manifests/04-governance/
```

---

## 🔌 Accessing Services (Port Forwarding)

Since services are isolated in namespaces, use `kubectl port-forward` to access them from localhost. Run each command in a separate terminal tab.

### ☁️ Cloud Tier Services

**Airflow UI (Orchestrator)**
```bash
kubectl port-forward svc/airflow-svc -n cloud-tier 8080:8080
```
* **URL:** [http://localhost:8080](http://localhost:8080)
* **Creds:** `admin` / `admin`

**MLflow UI (Registry)**
```bash
kubectl port-forward svc/mlflow-svc -n cloud-tier 5000:5000
```
* **URL:** [http://localhost:5000](http://localhost:5000)

### 🏭 Edge Tier Services

**Node-RED (Factory Simulator)**
```bash
kubectl port-forward svc/nodered-svc -n edge-tier 1880:1880
```
* **URL:** [http://localhost:1880](http://localhost:1880)

### 🌫️ Fog Tier Services

**MinIO Console (Storage)**
```bash
kubectl port-forward svc/minio-svc -n fog-tier 9001:9001
```
* **URL:** [http://localhost:9001](http://localhost:9001)
* **Creds:** `admin` / `password123`

### 🛡️ Governance

**Grafana (Dashboards)**
```bash
kubectl port-forward svc/grafana-svc -n governance 3000:3000
```
* **URL:** [http://localhost:3000](http://localhost:3000)
* **Creds:** `admin` / `admin`

---

## ⚙️ Essential Configuration (First Run)

**1. Create the S3 Bucket**
The storage is fresh. You must manually create the bucket for MLflow to work.
1. Go to **MinIO** ([http://localhost:9001](http://localhost:9001)).
2. Login with `admin` / `password123`.
3. Click **Create Bucket** -> Name it `mlflow-bucket`.

**2. Kafka Connection**
Inside the cluster, services talk via DNS.
* **Broker Internal URL:** `kafka-svc.edge-tier.svc.cluster.local:9092`
* Use this URL inside Airflow or your local scripts when configuring the Kafka Producer/Consumer.

---

## 🗑️ Cleanup

To remove the entire stack and all namespaces:

```bash
kubectl delete namespace cloud-tier fog-tier edge-tier governance
```

> **Note:** This will delete your Persistent Volume Claims (PVCs) and data depending on your cluster's storage policy.