FROM python:3.9-slim
LABEL authors="mmateo@cigip.upv.es"

# Install MLflow and connectors for MinIO (boto3) and Postgres/Timescale (psycopg2)
RUN pip install mlflow==2.7.1 boto3 psycopg2-binary

# Expose the default port
EXPOSE 5000

# The command uses environment variables defined in docker-compose.yml
# This allows dynamic configuration without rebuilding the image
CMD mlflow server \
    --host 0.0.0.0 \
    --port 5000 \
    --backend-store-uri ${BACKEND_STORE_URI} \
    --default-artifact-root s3://mlflow-bucket/