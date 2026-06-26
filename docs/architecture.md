# TelecomIQ — Architecture

## Data flow

1. **Generation / ingestion** — `data/generate_data.py` produces three labeled
   datasets (customers, CDRs, network telemetry). In production these come from
   the billing system, the call-mediation platform, and the OSS/EMS respectively.

2. **Streaming** — `pipeline/producer.py` replays CDRs onto the Kafka
   `cdr-stream` topic to simulate a live feed. `pipeline/spark_consumer.py`
   (Spark Structured Streaming) or `pipeline/consumer.py` (lightweight) consume
   the topic and score each record with the fraud model in real time.

3. **Modeling** — `models/` trains and serves three models:
   - **Churn** (`churn.py`) — XGBoost over subscriber features, with a
     scikit-learn `ColumnTransformer` for preprocessing.
   - **Fraud** (`fraud.py`) — IsolationForest over CDR features; unsupervised so
     it adapts to novel fraud without labels.
   - **Network anomaly** (`anomaly.py`) — a PyTorch autoencoder learns the normal
     telemetry envelope; reconstruction error is the anomaly signal. Falls back
     to IsolationForest where torch isn't installed.

4. **Serving** — `api/main.py` (FastAPI) loads the artifacts lazily and exposes
   per-record scoring plus the aggregate revenue endpoint.

5. **AI insights** — `insights/claude_insights.py` aggregates all model outputs
   into a `RevenueSnapshot` and asks Claude (`claude-opus-4-8`, adaptive
   thinking) to write a CEO-level briefing.

6. **Presentation** — `dashboard/app.py` (Streamlit) calls the API for live
   scoring and reads the datasets directly for fleet-wide charts.

7. **Storage** — PostgreSQL is provisioned for persisting scored insights /
   alerts (DSN exposed via `config.settings`).

## Why these model choices

| Problem | Model | Rationale |
|---------|-------|-----------|
| Churn (labeled, tabular) | XGBoost | Best-in-class on tabular data; calibrated probabilities; fast inference. |
| Fraud (rare, evolving, mostly unlabeled) | IsolationForest | Unsupervised — detects novel patterns without waiting for labels; O(ms) scoring for the <2s SLA. |
| Network anomaly (multivariate, normal-dominated) | Autoencoder | Learns the joint normal envelope; reconstruction error catches multi-KPI degradations a per-metric threshold would miss. |
| Executive summary | Claude | Turns numbers into prioritized, decision-ready narrative. |

## Scaling to production

- Replace synthetic generators with Kafka Connect sources from billing / OSS.
- Land the raw + scored streams in **Delta Lake** (sink stub in
  `spark_consumer.py`) on object storage for a lakehouse.
- Add **Grafana** dashboards over the PostgreSQL/Delta metrics for ops.
- Schedule model retraining (Airflow / cron) and add a model registry (MLflow).
- Add auth (OAuth2/JWT) to the FastAPI layer and rate-limit the public endpoints.
