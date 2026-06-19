# Tradite autonomous paper-trading autopilot — DigitalOcean container.
# Build from the ml_lab/ directory:
#   docker build -f deploy/Dockerfile -t tradite-autopilot .
FROM python:3.12-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    TRADITE_DATA_SOURCE=yfinance

# deps first (better layer caching)
COPY deploy/requirements-deploy.txt /app/deploy/requirements-deploy.txt
RUN pip install -r /app/deploy/requirements-deploy.txt

# app code (the whole ml_lab package)
COPY . /app/

# data_cache holds the persistent paper book + reports — mount a volume here.
VOLUME ["/app/data_cache", "/app/reports"]

# Live dashboard + autopilot in one process: uvicorn serves the status page and
# starts the scheduler loop (monitor 30m / decide daily) in a background thread.
EXPOSE 8080
CMD ["uvicorn", "sentiment.status_server:app", "--host", "0.0.0.0", "--port", "8080"]
