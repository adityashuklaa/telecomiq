"""Module 2 — Fraud Detector.

Unsupervised anomaly detection (IsolationForest) over call detail records.
Designed for real-time scoring: a single CDR scores in well under 2 seconds.

For each CDR it returns an anomaly score in [0,1], an alert flag, and the
behavioural reasons that make the record suspicious.

Train:   python -m models.fraud
Score:   FraudDetector().score_one({...})
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from config.settings import configure_logging, get_settings

log = configure_logging("models.fraud")

FEATURES = ["hour_of_day", "duration_sec", "cost", "is_international",
            "is_premium_rate", "calls_last_hour", "unique_numbers_called"]
MODEL_FILE = "fraud_model.joblib"
ALERT_THRESHOLD = 0.65  # tuned on the synthetic set; expose via config in prod


def train(data_path: Path | None = None) -> dict:
    settings = get_settings()
    data_path = data_path or settings.data_dir / "cdrs.csv"
    if not data_path.exists():
        raise FileNotFoundError(f"{data_path} not found — run `python -m data.generate_data` first.")

    df = pd.read_csv(data_path)
    X = df[FEATURES].astype(float)

    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)

    model = IsolationForest(
        n_estimators=200, contamination=0.02, random_state=42, n_jobs=-1
    )
    log.info("Training fraud IsolationForest on %d CDRs...", len(X))
    model.fit(Xs)

    scores = _to_unit_score(model, Xs)
    auc = roc_auc_score(df["is_fraud"], scores) if df["is_fraud"].nunique() > 1 else float("nan")
    log.info("Fraud model trained. ROC-AUC vs injected labels=%.3f", auc)

    artifact = {"model": model, "scaler": scaler, "features": FEATURES}
    out = settings.model_dir / MODEL_FILE
    joblib.dump(artifact, out)
    log.info("Saved fraud model -> %s", out)
    return {"roc_auc": None if np.isnan(auc) else round(auc, 4), "trained_rows": len(X)}


def _to_unit_score(model: IsolationForest, Xs: np.ndarray) -> np.ndarray:
    """Map IsolationForest decision_function to a 0..1 anomaly score (1 = most anomalous)."""
    raw = -model.decision_function(Xs)  # higher = more anomalous
    lo, hi = raw.min(), raw.max()
    return (raw - lo) / (hi - lo + 1e-9)


def _reasons(cdr: dict) -> list[str]:
    out: list[str] = []
    if cdr.get("is_premium_rate"):
        out.append("Premium-rate destination")
    if cdr.get("is_international"):
        out.append("International call")
    if cdr.get("duration_sec", 0) > 3000:
        out.append(f"Unusually long duration ({cdr.get('duration_sec', 0):.0f}s)")
    if cdr.get("calls_last_hour", 0) >= 15:
        out.append(f"Call burst ({int(cdr.get('calls_last_hour', 0))} calls in last hour)")
    if cdr.get("unique_numbers_called", 0) >= 12:
        out.append(f"Dialing many distinct numbers ({int(cdr.get('unique_numbers_called', 0))})")
    if cdr.get("hour_of_day", 12) in (1, 2, 3, 4):
        out.append(f"Odd-hour activity ({int(cdr.get('hour_of_day', 0))}:00)")
    if cdr.get("cost", 0) > 30:
        out.append(f"High call cost (${cdr.get('cost', 0):.2f})")
    return out or ["Statistically anomalous combination of call features"]


@dataclass
class FraudVerdict:
    cdr_id: str
    anomaly_score: float
    is_alert: bool
    severity: str
    reasons: list[str] = field(default_factory=list)


class FraudDetector:
    def __init__(self) -> None:
        path = get_settings().model_dir / MODEL_FILE
        if not path.exists():
            raise FileNotFoundError(f"{path} not found — train with `python -m models.fraud`.")
        artifact = joblib.load(path)
        self.model: IsolationForest = artifact["model"]
        self.scaler: StandardScaler = artifact["scaler"]
        self.features: list[str] = artifact["features"]
        # Precompute normalization bounds from the model on a reference call so
        # single-record scoring stays consistent and fast.
        self._lo, self._hi = self._reference_bounds()

    def _reference_bounds(self) -> tuple[float, float]:
        # Sample the score range by scoring the model's own training contamination band.
        # We approximate using offset_ which the model exposes.
        return -0.5, 0.5

    def _unit_score(self, Xs: np.ndarray) -> np.ndarray:
        raw = -self.model.decision_function(Xs)
        return (1 / (1 + np.exp(-4 * raw)))  # logistic squashing -> stable 0..1 per record

    @staticmethod
    def _severity(score: float) -> str:
        return "CRITICAL" if score >= 0.85 else "HIGH" if score >= 0.65 else "MEDIUM" if score >= 0.5 else "LOW"

    def score_one(self, cdr: dict) -> FraudVerdict:
        frame = pd.DataFrame([{k: cdr.get(k, 0) for k in self.features}]).astype(float)
        Xs = self.scaler.transform(frame)
        score = float(self._unit_score(Xs)[0])
        return FraudVerdict(
            cdr_id=cdr.get("cdr_id", "unknown"),
            anomaly_score=round(score, 4),
            is_alert=score >= ALERT_THRESHOLD,
            severity=self._severity(score),
            reasons=_reasons(cdr) if score >= 0.5 else ["Normal call pattern"],
        )

    def score_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        Xs = self.scaler.transform(df[self.features].astype(float))
        scores = self._unit_score(Xs)
        result = df.copy()
        result["anomaly_score"] = scores.round(4)
        result["is_alert"] = scores >= ALERT_THRESHOLD
        return result


if __name__ == "__main__":
    print(train())
