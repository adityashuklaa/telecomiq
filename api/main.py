"""TelecomIQ FastAPI service — serves all four analytics modules.

Endpoints
  GET  /                  service banner
  GET  /health           liveness + which models are loaded
  POST /churn/predict    Module 1 — churn risk for one customer
  POST /fraud/score      Module 2 — fraud score for one CDR (real-time)
  POST /anomaly/score    Module 3 — network anomaly for one telemetry row
  GET  /revenue/summary  Module 4 — Claude CEO briefing over the full dataset

Models are loaded lazily and cached. If a model artifact is missing the relevant
endpoint returns 503 with instructions to train it.
"""
from __future__ import annotations

from functools import lru_cache

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from config.settings import configure_logging, get_settings

log = configure_logging("api")
settings = get_settings()

app = FastAPI(
    title="TelecomIQ API",
    description="AI-powered telecom analytics: churn, fraud, network anomaly, revenue intelligence.",
    version="1.0.0",
)


# ─────────────────────────── lazy model loaders ───────────────────────────
@lru_cache
def _churn():
    from models.churn import ChurnPredictor

    return ChurnPredictor()


@lru_cache
def _fraud():
    from models.fraud import FraudDetector

    return FraudDetector()


@lru_cache
def _anomaly():
    from models.anomaly import NetworkAnomalyDetector

    return NetworkAnomalyDetector()


def _require(loader, name: str):
    try:
        return loader()
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=503,
            detail=f"{name} model not trained. Run `python -m models.train_all`. ({exc})",
        )


# ─────────────────────────── request schemas ───────────────────────────
class CustomerIn(BaseModel):
    customer_id: str = "CUST-000000"
    tenure_months: int = Field(12, ge=0)
    contract: str = "Month-to-month"
    internet_service: str = "Fiber optic"
    payment_method: str = "Electronic check"
    paperless_billing: int = Field(1, ge=0, le=1)
    senior_citizen: int = Field(0, ge=0, le=1)
    support_calls: int = Field(0, ge=0)
    monthly_charges: float = Field(70.0, ge=0)
    total_charges: float = Field(840.0, ge=0)


class CdrIn(BaseModel):
    cdr_id: str = "CDR-00000000"
    hour_of_day: int = Field(14, ge=0, le=23)
    duration_sec: float = Field(120.0, ge=0)
    cost: float = Field(0.25, ge=0)
    is_international: int = Field(0, ge=0, le=1)
    is_premium_rate: int = Field(0, ge=0, le=1)
    calls_last_hour: int = Field(2, ge=0)
    unique_numbers_called: int = Field(3, ge=0)


class TelemetryIn(BaseModel):
    tower_id: str = "TOWER-000"
    cpu_utilization: float = 45.0
    latency_ms: float = 35.0
    packet_loss_pct: float = 0.5
    throughput_mbps: float = 820.0
    dropped_calls: int = 3
    signal_strength_dbm: float = -75.0
    temperature_c: float = 38.0


# ─────────────────────────── endpoints ───────────────────────────
@app.get("/")
def root() -> dict:
    return {"service": "TelecomIQ API", "version": app.version, "docs": "/docs"}


@app.get("/health")
def health() -> dict:
    status = {}
    for name, loader in (("churn", _churn), ("fraud", _fraud), ("anomaly", _anomaly)):
        try:
            loader()
            status[name] = "loaded"
        except FileNotFoundError:
            status[name] = "not_trained"
    return {"status": "ok", "models": status}


@app.post("/churn/predict")
def churn_predict(customer: CustomerIn) -> dict:
    predictor = _require(_churn, "Churn")
    from dataclasses import asdict

    return asdict(predictor.predict_one(customer.model_dump()))


@app.post("/fraud/score")
def fraud_score(cdr: CdrIn) -> dict:
    detector = _require(_fraud, "Fraud")
    from dataclasses import asdict

    return asdict(detector.score_one(cdr.model_dump()))


@app.post("/anomaly/score")
def anomaly_score(row: TelemetryIn) -> dict:
    detector = _require(_anomaly, "Network anomaly")
    from dataclasses import asdict

    return asdict(detector.score_one(row.model_dump()))


@app.get("/revenue/summary")
def revenue_summary() -> dict:
    """Score the full generated dataset and return a Claude CEO briefing."""
    churn = _require(_churn, "Churn")
    fraud = _require(_fraud, "Fraud")
    anomaly = _require(_anomaly, "Network anomaly")

    data = settings.data_dir
    try:
        customers = pd.read_csv(data / "customers.csv")
        cdrs = pd.read_csv(data / "cdrs.csv")
        if len(cdrs) > 20000:  # cap for snapshot speed
            cdrs = cdrs.sample(20000, random_state=1)
        network = pd.read_csv(data / "network.csv")
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=f"Data not generated: {exc}")

    churn_scored = churn.predict_frame(customers)
    fraud_scored = fraud.score_frame(cdrs)
    net_scored = anomaly.score_frame(network)

    from insights.claude_insights import build_snapshot, generate_ceo_summary

    snap = build_snapshot(churn_scored, fraud_scored, net_scored)
    return generate_ceo_summary(snap)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api.main:app", host=settings.api_host, port=settings.api_port, reload=False)
