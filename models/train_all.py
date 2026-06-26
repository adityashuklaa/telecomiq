"""Train all three ML models in one shot. Generates data first if missing.

Run: python -m models.train_all
"""
from __future__ import annotations

from config.settings import configure_logging, get_settings
from data import generate_data
from models import anomaly, churn, fraud

log = configure_logging("models.train_all")


def main() -> None:
    settings = get_settings()
    if not (settings.data_dir / "customers.csv").exists():
        log.info("No data found — generating synthetic datasets...")
        # default sizes
        import sys

        sys.argv = ["generate_data"]
        generate_data.main()

    log.info("=== Module 1: Churn ===")
    log.info("metrics: %s", churn.train())
    log.info("=== Module 2: Fraud ===")
    log.info("metrics: %s", fraud.train())
    log.info("=== Module 3: Network Anomaly ===")
    log.info("metrics: %s", anomaly.train())
    log.info("All models trained -> %s", settings.model_dir.resolve())


if __name__ == "__main__":
    main()
