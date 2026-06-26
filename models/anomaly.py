"""Module 3 — Network Anomaly Detection.

Learns the normal operating envelope of tower telemetry and flags deviations
before they become outages. Primary backend is a PyTorch autoencoder
(reconstruction error = anomaly signal); if torch is unavailable it transparently
falls back to an IsolationForest so the platform still runs.

For each telemetry row it returns:
  - anomaly score (0..1)
  - severity (LOW / MEDIUM / HIGH / CRITICAL)
  - predicted root cause (the KPI deviating most from the learned normal)

Train:  python -m models.anomaly
Score:  NetworkAnomalyDetector().score_one({...})
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from config.settings import configure_logging, get_settings

log = configure_logging("models.anomaly")

FEATURES = ["cpu_utilization", "latency_ms", "packet_loss_pct", "throughput_mbps",
            "dropped_calls", "signal_strength_dbm", "temperature_c"]
MODEL_FILE = "anomaly_model.joblib"
TORCH_FILE = "anomaly_autoencoder.pt"

# Human-friendly root-cause messages keyed by the deviating KPI.
ROOT_CAUSE = {
    "cpu_utilization": "Tower CPU saturation — compute overload",
    "latency_ms": "Elevated latency — backhaul congestion or routing fault",
    "packet_loss_pct": "High packet loss — RF interference or link degradation",
    "throughput_mbps": "Throughput collapse — capacity or hardware fault",
    "dropped_calls": "Call drop spike — radio/handover failure",
    "signal_strength_dbm": "Weak signal — antenna or power amplifier fault",
    "temperature_c": "Overheating — cooling/HVAC failure risk",
}

try:  # optional deep-learning backend
    import torch
    import torch.nn as nn

    _TORCH = True
except Exception:  # pragma: no cover - torch optional
    _TORCH = False


if _TORCH:

    class _AutoEncoder(nn.Module):
        def __init__(self, n_features: int) -> None:
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Linear(n_features, 16), nn.ReLU(),
                nn.Linear(16, 8), nn.ReLU(),
                nn.Linear(8, 3), nn.ReLU(),
            )
            self.decoder = nn.Sequential(
                nn.Linear(3, 8), nn.ReLU(),
                nn.Linear(8, 16), nn.ReLU(),
                nn.Linear(16, n_features),
            )

        def forward(self, x):  # noqa: D401
            return self.decoder(self.encoder(x))


def _train_autoencoder(Xs: np.ndarray, epochs: int = 40) -> tuple["torch.nn.Module", np.ndarray]:
    torch.manual_seed(42)
    model = _AutoEncoder(Xs.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()
    X = torch.tensor(Xs, dtype=torch.float32)
    model.train()
    for epoch in range(epochs):
        opt.zero_grad()
        recon = model(X)
        loss = loss_fn(recon, X)
        loss.backward()
        opt.step()
        if (epoch + 1) % 10 == 0:
            log.info("  autoencoder epoch %d/%d loss=%.4f", epoch + 1, epochs, loss.item())
    model.eval()
    with torch.no_grad():
        errors = ((model(X) - X) ** 2).mean(dim=1).numpy()
    return model, errors


def train(data_path: Path | None = None) -> dict:
    settings = get_settings()
    data_path = data_path or settings.data_dir / "network.csv"
    if not data_path.exists():
        raise FileNotFoundError(f"{data_path} not found — run `python -m data.generate_data` first.")

    df = pd.read_csv(data_path)
    X = df[FEATURES].astype(float)
    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)

    backend = "autoencoder" if _TORCH else "isolation_forest"
    artifact = {"scaler": scaler, "features": FEATURES, "backend": backend,
                "feat_mean": X.mean().to_dict(), "feat_std": X.std().replace(0, 1).to_dict()}

    if _TORCH:
        log.info("Training PyTorch autoencoder on %d telemetry rows...", len(X))
        model, errors = _train_autoencoder(Xs)
        threshold = float(np.quantile(errors, 0.97))
        artifact["threshold"] = threshold
        artifact["err_scale"] = float(errors.max() + 1e-9)
        torch.save(model.state_dict(), settings.model_dir / TORCH_FILE)
        scores = np.clip(errors / artifact["err_scale"], 0, 1)
    else:
        from sklearn.ensemble import IsolationForest

        log.info("torch unavailable — training IsolationForest fallback on %d rows...", len(X))
        model = IsolationForest(n_estimators=200, contamination=0.03, random_state=42, n_jobs=-1)
        model.fit(Xs)
        artifact["sk_model"] = model
        raw = -model.decision_function(Xs)
        scores = (raw - raw.min()) / (raw.max() - raw.min() + 1e-9)
        artifact["threshold"] = float(np.quantile(scores, 0.97))
        artifact["err_scale"] = 1.0

    auc = roc_auc_score(df["is_failure"], scores) if df["is_failure"].nunique() > 1 else float("nan")
    log.info("Network anomaly model trained (%s). ROC-AUC=%.3f", backend, auc)

    joblib.dump(artifact, settings.model_dir / MODEL_FILE)
    log.info("Saved network anomaly model -> %s", settings.model_dir / MODEL_FILE)
    return {"backend": backend, "roc_auc": None if np.isnan(auc) else round(auc, 4)}


@dataclass
class AnomalyVerdict:
    tower_id: str
    anomaly_score: float
    is_anomaly: bool
    severity: str
    root_cause: str


class NetworkAnomalyDetector:
    def __init__(self) -> None:
        settings = get_settings()
        path = settings.model_dir / MODEL_FILE
        if not path.exists():
            raise FileNotFoundError(f"{path} not found — train with `python -m models.anomaly`.")
        self.a = joblib.load(path)
        self.features: list[str] = self.a["features"]
        self.scaler: StandardScaler = self.a["scaler"]
        self.backend: str = self.a["backend"]
        self.threshold: float = self.a["threshold"]
        self._torch_model = None
        if self.backend == "autoencoder" and _TORCH:
            self._torch_model = _AutoEncoder(len(self.features))
            self._torch_model.load_state_dict(torch.load(settings.model_dir / TORCH_FILE))
            self._torch_model.eval()

    @staticmethod
    def _severity(score: float) -> str:
        return ("CRITICAL" if score >= 0.85 else "HIGH" if score >= 0.6
                else "MEDIUM" if score >= 0.4 else "LOW")

    def _root_cause(self, row: dict) -> str:
        """KPI whose standardized deviation from the learned normal is largest."""
        devs = {}
        for f in self.features:
            mean = self.a["feat_mean"][f]
            std = self.a["feat_std"][f]
            devs[f] = abs((float(row.get(f, mean)) - mean) / (std or 1))
        worst = max(devs, key=devs.get)
        return ROOT_CAUSE.get(worst, f"Deviation in {worst}")

    def _score(self, Xs: np.ndarray) -> np.ndarray:
        if self.backend == "autoencoder" and self._torch_model is not None:
            with torch.no_grad():
                X = torch.tensor(Xs, dtype=torch.float32)
                err = ((self._torch_model(X) - X) ** 2).mean(dim=1).numpy()
            return np.clip(err / self.a["err_scale"], 0, 1)
        raw = -self.a["sk_model"].decision_function(Xs)
        # reuse training-time min/max via threshold context; logistic squash for stability
        return 1 / (1 + np.exp(-4 * raw))

    def score_one(self, row: dict) -> AnomalyVerdict:
        frame = pd.DataFrame([{k: row.get(k, self.a["feat_mean"][k]) for k in self.features}]).astype(float)
        Xs = self.scaler.transform(frame)
        score = float(self._score(Xs)[0])
        is_anom = score >= self.threshold
        return AnomalyVerdict(
            tower_id=row.get("tower_id", "unknown"),
            anomaly_score=round(score, 4),
            is_anomaly=is_anom,
            severity=self._severity(score),
            root_cause=self._root_cause(row) if is_anom else "Healthy — within normal envelope",
        )

    def score_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        Xs = self.scaler.transform(df[self.features].astype(float))
        scores = self._score(Xs)
        result = df.copy()
        result["anomaly_score"] = scores.round(4)
        result["is_anomaly"] = scores >= self.threshold
        return result


if __name__ == "__main__":
    print(train())
