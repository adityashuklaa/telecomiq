# 📡 TelecomIQ — AI-Powered Telecom Analytics Platform

End-to-end telecom analytics: real-time data pipelines, ML/DL models, an AI
insights layer, and a live dashboard — all runnable with **one command**.

[![Python](https://img.shields.io/badge/python-3.11-blue)]()
[![FastAPI](https://img.shields.io/badge/API-FastAPI-009688)]()
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B)]()
[![Claude](https://img.shields.io/badge/AI-Claude-orange)]()

---

## What it does

| Module | Name | Tech | Output |
|--------|------|------|--------|
| 1 | **Churn Predictor** | XGBoost + scikit-learn | Per-customer risk score, reason, recommended action |
| 2 | **Fraud Detector** | IsolationForest (real-time, <2s) | Anomaly score + alert + reasons on each call record |
| 3 | **Network Anomaly Detection** | PyTorch autoencoder (sklearn fallback) | Severity + predicted root cause per tower |
| 4 | **Revenue Intelligence** | Claude API (`anthropic`) | Plain-English CEO briefing over all signals |

## Architecture

```
                          ┌──────────────────────────────────────────────┐
                          │                  TelecomIQ                     │
                          └──────────────────────────────────────────────┘

  data/generate_data.py            pipeline/                     models/
  ┌───────────────────┐    CDRs    ┌──────────────┐   score    ┌───────────────┐
  │ synthetic telecom │ ─────────► │ Kafka topic  │ ─────────► │ Fraud model   │
  │  CDR / churn /    │            │ cdr-stream   │            │ (IsolationF.) │
  │  network datasets │            └──────┬───────┘            └───────────────┘
  └─────────┬─────────┘                   │ Spark Structured Streaming
            │                             │ (pipeline/spark_consumer.py)
            │ train                       ▼
            ▼                      real-time fraud alerts
  ┌───────────────────────────────────────────────┐
  │ models/  churn (XGBoost) · fraud (IForest) ·   │
  │          network anomaly (PyTorch autoencoder) │
  └───────────────┬───────────────────────────────┘
                  │ joblib / .pt artifacts
                  ▼
  ┌───────────────────────────┐      ┌──────────────────────────────┐
  │  api/  FastAPI             │ ───► │ insights/ Claude API          │
  │  /churn /fraud /anomaly    │      │ → CEO-level revenue briefing  │
  │  /revenue/summary          │      └──────────────────────────────┘
  └───────────────┬───────────┘
                  │ REST
                  ▼
  ┌───────────────────────────┐      ┌──────────────────────────────┐
  │ dashboard/ Streamlit UI    │      │ PostgreSQL (insights store)   │
  └───────────────────────────┘      └──────────────────────────────┘
```

See [`docs/architecture.md`](docs/architecture.md) for the detailed data flow.

---

## 🌐 Live demo (Streamlit Cloud — free)

A self-contained version (`streamlit_app.py`) deploys to **Streamlit Community
Cloud** for free — it runs the ML models in-process, so no server, Kafka, or
Postgres is needed.

**Deploy in 4 clicks:**
1. Push this repo to GitHub.
2. Go to **[share.streamlit.io](https://share.streamlit.io)** → sign in with GitHub.
3. **New app** → pick this repo → main file = `streamlit_app.py` → **Deploy**.
4. (Optional) In the app's **Settings → Secrets**, add:
   ```toml
   ANTHROPIC_API_KEY = "sk-ant-..."
   CLAUDE_MODEL = "claude-opus-4-8"
   ```
   to enable the Claude-written CEO briefing (works without it via a template).

First load generates data + trains the models (~30s), then it's instant. The
full distributed stack (below) lives in the same repo for the architecture story.

---

## Quick start (Docker — one command, full stack)

```bash
cp .env.example .env          # then add your ANTHROPIC_API_KEY (optional)
docker compose up --build
```

This brings up **Kafka, PostgreSQL, the API, the dashboard, a CDR producer, and
the real-time fraud consumer**. On first boot the API container generates the
synthetic datasets and trains all three models automatically.

| Service | URL |
|---------|-----|
| 📊 Dashboard | http://localhost:8501 |
| ⚙️ API docs (Swagger) | http://localhost:8000/docs |
| ❤️ Health | http://localhost:8000/health |

> The Claude-powered CEO briefing works without a key (returns a template
> summary). Add `ANTHROPIC_API_KEY` to `.env` for the AI-written narrative.

---

## Local development (full stack: PyTorch autoencoder + Spark)

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 1. Generate data + train all models
python -m models.train_all

# 2. Run the API
uvicorn api.main:app --reload --port 8000

# 3. Run the dashboard (new terminal)
streamlit run dashboard/app.py
```

### Real-time fraud streaming (needs a running Kafka)

```bash
python -m pipeline.producer --rate 20          # stream CDRs onto Kafka
python -m pipeline.consumer                     # lightweight real-time scorer
# or, with Spark installed:
spark-submit --packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.2 \
    pipeline/spark_consumer.py
```

---

## Using a real dataset

The synthetic customer schema mirrors the **IBM Telco Customer Churn** dataset
([Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)).
To use the real data, drop `WA_Fn-UseC_-Telco-Customer-Churn.csv` into
`data/generated/customers.csv` (matching the column names) and run
`python -m models.churn`. The rest of the pipeline is unchanged.

---

## API reference

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Liveness + which models are loaded |
| POST | `/churn/predict` | Churn risk for one customer |
| POST | `/fraud/score` | Fraud score for one CDR (real-time) |
| POST | `/anomaly/score` | Network anomaly for one telemetry row |
| GET | `/revenue/summary` | Claude CEO briefing over the full dataset |

Example:

```bash
curl -X POST http://localhost:8000/churn/predict \
  -H "Content-Type: application/json" \
  -d '{"tenure_months":3,"contract":"Month-to-month","internet_service":"Fiber optic",
       "payment_method":"Electronic check","support_calls":5,"monthly_charges":95,"total_charges":285}'
```

---

## Project structure

```
telecomiq/
├── docker-compose.yml        # Kafka + Postgres + API + dashboard + stream
├── Dockerfile
├── requirements.txt          # full deps (torch + pyspark)
├── requirements-app.txt      # slim deps for the Docker image
├── config/settings.py        # env-driven config + logging (no hardcoded secrets)
├── data/generate_data.py     # synthetic CDR / churn / network generator
├── models/                   # churn · fraud · anomaly + train_all
├── pipeline/                 # Kafka producer + Spark/light consumers
├── api/main.py               # FastAPI service
├── insights/claude_insights.py  # Claude CEO-summary (Module 4)
├── dashboard/app.py          # Streamlit UI
├── docker/entrypoint.sh
└── docs/architecture.md
```

## Production notes

- **Config & secrets** — everything is env-driven (`config/settings.py`); secrets
  live in `.env` (git-ignored). Never commit real keys.
- **Logging** — structured stdout logging across every module.
- **Error handling** — API returns `503` with remediation when a model isn't
  trained; the Claude layer falls back to a template if the API is unreachable.
- **Swap-in production sinks** — `pipeline/spark_consumer.py` includes a commented
  Delta Lake sink; point it at S3/ADLS for a real lakehouse.

## Demo GIF

To record a demo for the README/portfolio: start the stack, open the dashboard,
walk through each tab, and capture with [ScreenToGif](https://www.screentogif.com/)
(Windows) or [Peek](https://github.com/phw/peek) (Linux). Save to
`docs/demo.gif` and embed with `![demo](docs/demo.gif)`.

---

Built by **Aditya Shukla** — Data Analyst.
