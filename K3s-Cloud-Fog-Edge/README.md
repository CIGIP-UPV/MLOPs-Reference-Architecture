# Kubernetes Industrial MLOps Lab ☸️ (Cloud-Fog-Edge)

This folder contains the **Kubernetes manifests** to deploy the distributed **Cloud-Fog-Edge** architecture for Industrial AI described in the article.

Unlike the flat Docker Compose setup, this deployment models a more production-oriented environment using **Namespaces** to isolate Cloud, Fog, Edge, Governance and Enterprise responsibilities.

## What This Kubernetes Flavor Adds

This K3s flavor now mirrors the expanded reference implementation:

- Cloud bootstrap job for databases, buckets, seeded CNC data and baseline model registration.
- Versioned Airflow image aligned with the Docker flavor.
- Fog bridge deployment with spool storage.
- Edge sync and edge inference deployments with persistent deployment state.
- Enterprise API and Enterprise UI namespace.
- Extended Prometheus scraping for the closed-loop services.

---

## 🏗 Architecture Layers

The cluster is divided into 5 logical tiers:

| Tier | Namespace | Key Components | Role |
| :--- | :--- | :--- | :--- |
| **1. Cloud** | `cloud-tier` | **Airflow, MLflow, bootstrap job** | Orchestration, retraining, registry, lifecycle bootstrap |
| **2. Fog** | `fog-tier` | **TimescaleDB, MinIO, Fog Bridge** | Persistence, buffering, storage and data plane |
| **3. Edge** | `edge-tier` | **Kafka, Mosquitto, Node-RED, CNC Simulator, Edge Sync, Edge Inference** | Real-time plant ingestion and low-latency scoring |
| **4. Governance** | `governance` | **Prometheus, Grafana** | Cross-tier observability |
| **5. Enterprise** | `enterprise-tier` | **Enterprise API, Enterprise UI** | Supervisory control and reviewer-facing dashboard |

---

## 📂 Directory Structure

```text
k8-manifests/
├── 00-base/                  # Namespaces and PVCs
├── 01-fog-tier/              # TimescaleDB, MinIO, Fog Bridge
├── 02-cloud-tier/            # Airflow, MLflow, bootstrap job
├── 03-edge-tier/             # Kafka, Mosquitto, Node-RED, edge services, CNC simulator
├── 04-governance/            # Prometheus and Grafana
└── 05-enterprise-tier/       # Enterprise API and Enterprise UI
```

---

# 🚀 Deployment Guide

### Prerequisites

- **Kubernetes Cluster** (Docker Desktop K8s, Minikube, K3s or equivalent).
- `kubectl` configured.
- Minimum recommended resources: **4 CPUs** and **8 GB RAM**.
- Local images built from this repository or pushed to a registry accessible by the cluster.

## Build the Custom Images First

The Kubernetes manifests expect images built from the repository:

```bash
docker build -t ind-mlops-airflow:latest -f Docker-Cloud-Fog-Edge/build/airflow/Dockerfile Docker-Cloud-Fog-Edge
docker build -t ind-mlops-mlflow:latest -f Docker-Cloud-Fog-Edge/build/mlflow/Dockerfile Docker-Cloud-Fog-Edge
docker build -t ind-mlops-python-service:latest -f Docker-Cloud-Fog-Edge/build/python-service/Dockerfile Docker-Cloud-Fog-Edge
docker build -t ind-mlops-enterprise-ui:latest -f Docker-Cloud-Fog-Edge/build/enterprise-ui/Dockerfile Docker-Cloud-Fog-Edge
```

If you are using K3s or a remote cluster, load or push these images to the runtime registry used by the cluster.

---

## Step-by-Step Installation

Run the commands in this order so the persistence layer is available before the bootstrap and control plane.

**1. Base Infrastructure**

```bash
kubectl apply -f k8-manifests/00-base/
```

**2. Fog Tier**

```bash
kubectl apply -f k8-manifests/01-fog-tier/
```

Wait until `timescale` and `minio` are running.

**3. Cloud Tier**

```bash
kubectl apply -f k8-manifests/02-cloud-tier/
```

This includes the `platform-bootstrap` Job that creates databases/buckets, seeds CNC telemetry and registers the initial production model.

**4. Edge Tier**

```bash
kubectl apply -f k8-manifests/03-edge-tier/
```

**5. Governance Tier**

```bash
kubectl apply -f k8-manifests/04-governance/
```

**6. Enterprise Tier**

```bash
kubectl apply -f k8-manifests/05-enterprise-tier/
```

---

## 🔌 Accessing Services (Port Forwarding)

### ☁️ Cloud Tier Services

**Airflow UI**

```bash
kubectl port-forward svc/airflow-svc -n cloud-tier 8080:8080
```

**MLflow UI**

```bash
kubectl port-forward svc/mlflow-svc -n cloud-tier 5000:5000
```

### 🌫️ Fog Tier Services

**MinIO Console**

```bash
kubectl port-forward svc/minio-svc -n fog-tier 9001:9001
```

### 🏭 Edge Tier Services

**Node-RED**

```bash
kubectl port-forward svc/nodered-svc -n edge-tier 1880:1880
```

### 🛡️ Governance

**Grafana**

```bash
kubectl port-forward svc/grafana-svc -n governance 3000:3000
```

### 🏢 Enterprise Tier

**Enterprise API**

```bash
kubectl port-forward svc/enterprise-api-svc -n enterprise-tier 8085:8085
```

**Enterprise UI**

```bash
kubectl port-forward svc/enterprise-ui-svc -n enterprise-tier 8088:80
```

---

## ⚙️ Operational Notes

- The MinIO buckets are created by the bootstrap job through the shared Python lifecycle logic.
- The edge sync agent stores the applied deployment state in `edge-state-pvc`.
- The fog bridge stores buffered events in `fog-spool-pvc` if downstream delivery fails.
- Airflow DAGs are embedded into the custom Airflow image and aligned with the Docker implementation.

---

## 🗑️ Cleanup

```bash
kubectl delete namespace cloud-tier fog-tier edge-tier governance enterprise-tier
```

> Note: PVC cleanup behavior depends on your storage class reclaim policy.
