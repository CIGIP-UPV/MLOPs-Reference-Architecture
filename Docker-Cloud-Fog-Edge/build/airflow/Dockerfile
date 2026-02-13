FROM apache/airflow:2.7.1
LABEL authors="mmateo@cigip.upv.es"

USER root
# Install system dependencies if needed (e.g. for psycopg2)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

USER airflow

# Install Python libraries for our pipeline
# - mlflow: to register models
# - kafka-python: to read/write to Kafka
# - scikit-learn & pandas: for training tasks inside Airflow
# - apache-airflow-providers-postgres: to connect to TimescaleDB
RUN pip install --no-cache-dir \
    mlflow==2.7.1 \
    kafka-python \
    pandas \
    scikit-learn \
    apache-airflow-providers-postgres