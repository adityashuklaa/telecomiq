"""Synthetic telecom data generator for TelecomIQ.

Produces three realistic, labeled datasets so the whole platform runs offline:

  1. customers.csv  — subscriber profiles + churn label   (Module 1)
  2. cdrs.csv       — call detail records w/ fraud labels  (Module 2)
  3. network.csv    — tower telemetry w/ failure labels    (Module 3)

The customer schema mirrors the well-known IBM "Telco Customer Churn" dataset
(Kaggle: blastchar/telco-customer-churn) so the model transfers to real data —
swap `generate_customers()` for a `pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")`
and the rest of the pipeline is unchanged.

Run:  python -m data.generate_data --customers 8000 --cdrs 60000 --towers 200
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from config.settings import configure_logging, get_settings

log = configure_logging("data.generator")
RNG = np.random.default_rng(42)

CONTRACTS = ["Month-to-month", "One year", "Two year"]
INTERNET = ["DSL", "Fiber optic", "No"]
PAYMENT = ["Electronic check", "Mailed check", "Bank transfer", "Credit card"]


def generate_customers(n: int) -> pd.DataFrame:
    """Subscriber profiles with a churn label driven by realistic risk factors."""
    tenure = RNG.integers(0, 73, n)
    contract = RNG.choice(CONTRACTS, n, p=[0.55, 0.21, 0.24])
    internet = RNG.choice(INTERNET, n, p=[0.34, 0.44, 0.22])
    monthly = np.where(
        internet == "Fiber optic",
        RNG.normal(85, 18, n),
        np.where(internet == "DSL", RNG.normal(58, 15, n), RNG.normal(22, 6, n)),
    ).clip(18, 130)
    support_calls = RNG.poisson(1.2, n)
    senior = RNG.choice([0, 1], n, p=[0.84, 0.16])
    payment = RNG.choice(PAYMENT, n, p=[0.34, 0.23, 0.22, 0.21])
    paperless = RNG.choice([0, 1], n, p=[0.41, 0.59])
    total = (monthly * np.maximum(tenure, 1) * RNG.normal(1.0, 0.05, n)).clip(20, None)

    # Churn probability rises with: short tenure, month-to-month, fiber bill shock,
    # many support calls, electronic-check payment, senior citizen.
    logit = (
        -1.2
        - 0.045 * tenure
        + 1.1 * (contract == "Month-to-month")
        + 0.6 * (internet == "Fiber optic")
        + 0.28 * support_calls
        + 0.5 * (payment == "Electronic check")
        + 0.35 * senior
        + 0.010 * (monthly - 60)
    )
    churn_prob = 1 / (1 + np.exp(-logit))
    churn = (RNG.random(n) < churn_prob).astype(int)

    return pd.DataFrame(
        {
            "customer_id": [f"CUST-{i:06d}" for i in range(n)],
            "tenure_months": tenure,
            "contract": contract,
            "internet_service": internet,
            "payment_method": payment,
            "paperless_billing": paperless,
            "senior_citizen": senior,
            "support_calls": support_calls,
            "monthly_charges": monthly.round(2),
            "total_charges": total.round(2),
            "churn": churn,
        }
    )


def generate_cdrs(n: int, customer_ids: list[str]) -> pd.DataFrame:
    """Call detail records. ~1.5% are fraud with anomalous patterns
    (premium-rate / international bursts, abnormal duration, odd hours)."""
    caller = RNG.choice(customer_ids, n)
    hour = RNG.integers(0, 24, n)
    duration = RNG.gamma(2.0, 90, n).clip(1, 7200)  # seconds
    cost = (duration / 60 * RNG.normal(0.12, 0.03, n)).clip(0, None)
    intl = RNG.choice([0, 1], n, p=[0.92, 0.08])
    premium = RNG.choice([0, 1], n, p=[0.97, 0.03])
    calls_last_hour = RNG.poisson(2, n)
    unique_numbers = RNG.poisson(3, n) + 1

    fraud = np.zeros(n, dtype=int)
    fraud_idx = RNG.choice(n, size=int(n * 0.015), replace=False)
    fraud[fraud_idx] = 1
    # Make fraud records anomalous
    duration[fraud_idx] = RNG.gamma(2.0, 900, len(fraud_idx)).clip(60, 14400)
    cost[fraud_idx] = (duration[fraud_idx] / 60 * RNG.normal(0.9, 0.2, len(fraud_idx))).clip(0, None)
    intl[fraud_idx] = 1
    premium[fraud_idx] = RNG.choice([0, 1], len(fraud_idx), p=[0.4, 0.6])
    hour[fraud_idx] = RNG.choice([1, 2, 3, 4], len(fraud_idx))  # odd hours
    calls_last_hour[fraud_idx] = RNG.poisson(25, len(fraud_idx)) + 8
    unique_numbers[fraud_idx] = RNG.poisson(20, len(fraud_idx)) + 5

    return pd.DataFrame(
        {
            "cdr_id": [f"CDR-{i:08d}" for i in range(n)],
            "caller_id": caller,
            "hour_of_day": hour,
            "duration_sec": duration.round(1),
            "cost": cost.round(4),
            "is_international": intl,
            "is_premium_rate": premium,
            "calls_last_hour": calls_last_hour,
            "unique_numbers_called": unique_numbers,
            "is_fraud": fraud,
        }
    )


def generate_network(n: int) -> pd.DataFrame:
    """Per-tower telemetry snapshots. ~3% are failures (degraded KPIs)."""
    tower = [f"TOWER-{i % 200:03d}" for i in range(n)]
    region = RNG.choice(["North", "South", "East", "West", "Central"], n)
    cpu = RNG.normal(45, 12, n).clip(2, 100)
    latency = RNG.normal(35, 8, n).clip(2, 400)       # ms
    packet_loss = RNG.gamma(1.2, 0.4, n).clip(0, 100)  # %
    throughput = RNG.normal(820, 110, n).clip(5, 1500)  # Mbps
    dropped_calls = RNG.poisson(3, n)
    signal = RNG.normal(-75, 8, n).clip(-120, -40)     # dBm
    temperature = RNG.normal(38, 6, n).clip(10, 95)    # °C

    failure = np.zeros(n, dtype=int)
    fail_idx = RNG.choice(n, size=int(n * 0.03), replace=False)
    failure[fail_idx] = 1
    cpu[fail_idx] = RNG.normal(94, 4, len(fail_idx)).clip(60, 100)
    latency[fail_idx] = RNG.normal(220, 60, len(fail_idx)).clip(120, 400)
    packet_loss[fail_idx] = RNG.normal(28, 10, len(fail_idx)).clip(8, 100)
    throughput[fail_idx] = RNG.normal(120, 60, len(fail_idx)).clip(5, 350)
    dropped_calls[fail_idx] = RNG.poisson(40, len(fail_idx)) + 10
    signal[fail_idx] = RNG.normal(-105, 6, len(fail_idx)).clip(-120, -90)
    temperature[fail_idx] = RNG.normal(78, 8, len(fail_idx)).clip(60, 95)

    return pd.DataFrame(
        {
            "tower_id": tower,
            "region": region,
            "cpu_utilization": cpu.round(2),
            "latency_ms": latency.round(2),
            "packet_loss_pct": packet_loss.round(3),
            "throughput_mbps": throughput.round(1),
            "dropped_calls": dropped_calls,
            "signal_strength_dbm": signal.round(1),
            "temperature_c": temperature.round(1),
            "is_failure": failure,
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic telecom datasets")
    parser.add_argument("--customers", type=int, default=8000)
    parser.add_argument("--cdrs", type=int, default=60000)
    parser.add_argument("--towers", type=int, default=12000, help="network telemetry rows")
    args = parser.parse_args()

    settings = get_settings()
    out = settings.data_dir
    out.mkdir(parents=True, exist_ok=True)

    log.info("Generating %d customers...", args.customers)
    customers = generate_customers(args.customers)
    customers.to_csv(out / "customers.csv", index=False)

    log.info("Generating %d CDRs...", args.cdrs)
    cdrs = generate_cdrs(args.cdrs, customers["customer_id"].tolist())
    cdrs.to_csv(out / "cdrs.csv", index=False)

    log.info("Generating %d network telemetry rows...", args.towers)
    network = generate_network(args.towers)
    network.to_csv(out / "network.csv", index=False)

    log.info(
        "Done. churn=%.1f%% | fraud=%.2f%% | net_failures=%.2f%% -> %s",
        100 * customers["churn"].mean(),
        100 * cdrs["is_fraud"].mean(),
        100 * network["is_failure"].mean(),
        out.resolve(),
    )


if __name__ == "__main__":
    main()
