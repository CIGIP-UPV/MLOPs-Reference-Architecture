FROM apache/airflow:2.7.1
LABEL authors="mmateo@cigip.upv.es"

USER root
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    libpq-dev && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

USER airflow
RUN pip install --no-cache-dir \
    mlflow==2.7.1 \
    boto3 \
    kafka-python \
    pandas \
    paho-mqtt \
    psycopg2-binary \
    scikit-learn \
    scipy \
    apache-airflow-providers-postgres

COPY apps /opt/industrial_mlops/apps
COPY notebooks/dags /opt/airflow/dags
ENV PYTHONPATH=/opt/industrial_mlops/apps
