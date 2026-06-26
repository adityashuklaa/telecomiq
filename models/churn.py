"""Module 1 — Churn Predictor.

XGBoost classifier over subscriber features. For each customer it returns:
  - churn risk score (probability)
  - risk band (LOW / MEDIUM / HIGH)
  - human-readable reasons (the risk drivers present for THIS customer)
  - a recommended retention action

Train:   python -m models.churn
Predict: ChurnPredictor().predict_one({...})
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from config.settings import configure_logging, get_settings

log = configure_logging("models.churn")

NUMERIC = ["tenure_months", "support_calls", "monthly_charges", "total_charges",
           "senior_citizen", "paperless_billing"]
CATEGORICAL = ["contract", "internet_service", "payment_method"]
FEATURES = NUMERIC + CATEGORICAL
TARGET = "churn"
MODEL_FILE = "churn_model.joblib"


def _build_pipeline() -> Pipeline:
    pre = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
        ]
    )
    clf = XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.08,
        subsample=0.9,
        colsample_bytree=0.9,
        eval_metric="logloss",
        n_jobs=-1,
        random_state=42,
    )
    return Pipeline([("pre", pre), ("clf", clf)])


def train(data_path: Path | None = None) -> dict:
    """Train and persist the churn model. Returns evaluation metrics."""
    settings = get_settings()
    data_path = data_path or settings.data_dir / "customers.csv"
    if not data_path.exists():
        raise FileNotFoundError(f"{data_path} not found — run `python -m data.generate_data` first.")

    df = pd.read_csv(data_path)
    X, y = df[FEATURES], df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    pipe = _build_pipeline()
    log.info("Training churn model on %d rows...", len(X_train))
    pipe.fit(X_train, y_train)

    proba = pipe.predict_proba(X_test)[:, 1]
    preds = (proba >= 0.5).astype(int)
    auc = roc_auc_score(y_test, proba)
    report = classification_report(y_test, preds, output_dict=True, zero_division=0)
    log.info("Churn model trained. ROC-AUC=%.3f | recall(churn)=%.3f", auc, report["1"]["recall"])

    out = settings.model_dir / MODEL_FILE
    joblib.dump(pipe, out)
    log.info("Saved churn model -> %s", out)
    return {"roc_auc": round(auc, 4), "accuracy": round(report["accuracy"], 4),
            "churn_recall": round(report["1"]["recall"], 4)}


def _reasons(row: dict, risk: float) -> list[str]:
    """Explain the prediction using the risk drivers present for this customer."""
    out: list[str] = []
    if row.get("contract") == "Month-to-month":
        out.append("On a month-to-month contract (no lock-in)")
    if row.get("tenure_months", 99) <= 6:
        out.append(f"New customer ({int(row.get('tenure_months', 0))} months tenure)")
    if row.get("support_calls", 0) >= 3:
        out.append(f"High support contact volume ({int(row.get('support_calls', 0))} calls)")
    if row.get("internet_service") == "Fiber optic" and row.get("monthly_charges", 0) > 80:
        out.append("Premium fiber plan with high monthly bill (price sensitivity)")
    if row.get("payment_method") == "Electronic check":
        out.append("Pays by electronic check (correlates with higher churn)")
    if not out:
        out.append("No single dominant driver — broad mild risk factors" if risk >= 0.3
                   else "Stable profile — low risk")
    return out


def _action(risk: float, row: dict) -> str:
    if risk >= 0.7:
        if row.get("contract") == "Month-to-month":
            return "Offer a discounted 1-year contract + proactive retention call within 48h"
        return "Escalate to retention team; offer loyalty credit + service health check"
    if risk >= 0.4:
        return "Send targeted loyalty offer; monitor support tickets closely"
    return "No action needed — include in standard nurture campaign"


@dataclass
class ChurnPrediction:
    customer_id: str
    risk_score: float
    risk_band: str
    reasons: list[str]
    recommended_action: str


class ChurnPredictor:
    """Loads the trained pipeline and scores customers."""

    def __init__(self) -> None:
        path = get_settings().model_dir / MODEL_FILE
        if not path.exists():
            raise FileNotFoundError(f"{path} not found — train with `python -m models.churn`.")
        self.pipe: Pipeline = joblib.load(path)

    @staticmethod
    def _band(risk: float) -> str:
        return "HIGH" if risk >= 0.7 else "MEDIUM" if risk >= 0.4 else "LOW"

    def predict_one(self, customer: dict) -> ChurnPrediction:
        frame = pd.DataFrame([{k: customer.get(k) for k in FEATURES}])
        risk = float(self.pipe.predict_proba(frame)[:, 1][0])
        return ChurnPrediction(
            customer_id=customer.get("customer_id", "unknown"),
            risk_score=round(risk, 4),
            risk_band=self._band(risk),
            reasons=_reasons(customer, risk),
            recommended_action=_action(risk, customer),
        )

    def predict_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        risks = self.pipe.predict_proba(df[FEATURES])[:, 1]
        result = df.copy()
        result["risk_score"] = risks.round(4)
        result["risk_band"] = [self._band(r) for r in risks]
        return result


if __name__ == "__main__":
    print(train())
