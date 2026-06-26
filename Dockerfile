FROM python:3.11-slim

# System deps (xgboost needs libgomp)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install python deps first for layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# App code
COPY . .

RUN chmod +x docker/entrypoint.sh

ENV PYTHONUNBUFFERED=1 \
    MODEL_DIR=/app/artifacts \
    DATA_DIR=/app/data/generated

ENTRYPOINT ["docker/entrypoint.sh"]
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
