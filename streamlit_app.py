"""TelecomIQ — Streamlit Cloud entry point (self-contained live demo).

Unlike `dashboard/app.py` (which talks to the FastAPI service), this version
loads the ML models *in-process* so it deploys to Streamlit Community Cloud for
free with no separate API, Kafka, or Postgres. On first load it generates a
small synthetic dataset and trains all three models, then caches them.

Deploy: push to GitHub → share.streamlit.io → pick this file as the entry point.
Set ANTHROPIC_API_KEY in the app's Secrets for the Claude-written CEO briefing.
"""
from __future__ import annotations

import os

import streamlit as st

# ── Bridge Streamlit secrets → env vars BEFORE importing settings ──
try:
    if "ANTHROPIC_API_KEY" in st.secrets:
        os.environ["ANTHROPIC_API_KEY"] = st.secrets["ANTHROPIC_API_KEY"]
    if "CLAUDE_MODEL" in st.secrets:
        os.environ["CLAUDE_MODEL"] = st.secrets["CLAUDE_MODEL"]
except Exception:
    pass  # no secrets configured — Claude layer falls back to a template

import pandas as pd
import plotly.express as px

from config.settings import get_settings

st.set_page_config(page_title="TelecomIQ", page_icon="📡", layout="wide")


@st.cache_resource(show_spinner="🚀 First load: generating data & training models (~30s)...")
def load_platform():
    """Generate data (if missing) + train all three models, return predictors."""
    from data.generate_data import generate_cdrs, generate_customers, generate_network
    from models import anomaly, churn, fraud
    from models.anomaly import NetworkAnomalyDetector
    from models.churn import ChurnPredictor
    from models.fraud import FraudDetector

    settings = get_settings()
    data = settings.data_dir
    data.mkdir(parents=True, exist_ok=True)

    if not (data / "customers.csv").exists():
        customers = generate_customers(3000)
        customers.to_csv(data / "customers.csv", index=False)
        generate_cdrs(12000, customers["customer_id"].tolist()).to_csv(data / "cdrs.csv", index=False)
        generate_network(4000).to_csv(data / "network.csv", index=False)

    if not (settings.model_dir / "churn_model.joblib").exists():
        churn.train()
        fraud.train()
        anomaly.train()

    return ChurnPredictor(), FraudDetector(), NetworkAnomalyDetector()


churn_model, fraud_model, anomaly_model = load_platform()
settings = get_settings()


@st.cache_data
def load_csv(name: str) -> pd.DataFrame:
    path = settings.data_dir / name
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


st.title("📡 TelecomIQ — AI Telecom Analytics")
st.caption("Live demo · churn · fraud · network anomaly · Claude revenue intelligence")

tab_rev, tab_churn, tab_fraud, tab_net = st.tabs(
    ["💰 Revenue Intelligence", "🔮 Churn", "🚨 Fraud", "🛰️ Network"]
)

# ─────────────────────────── Revenue Intelligence ───────────────────────────
with tab_rev:
    st.subheader("Revenue Intelligence — CEO Briefing")
    st.write("Aggregates churn, fraud & network signals into a Claude-written summary.")
    if st.button("Generate executive briefing", type="primary"):
        with st.spinner("Scoring fleet & asking Claude..."):
            from insights.claude_insights import build_snapshot, generate_ceo_summary

            cust = churn_model.predict_frame(load_csv("customers.csv"))
            cdr = fraud_model.score_frame(load_csv("cdrs.csv"))
            net = anomaly_model.score_frame(load_csv("network.csv"))
            snap = build_snapshot(cust, cdr, net)
            result = generate_ceo_summary(snap)
        s = result["snapshot"]
        c = st.columns(4)
        c[0].metric("Customers", f"{s['total_customers']:,}")
        c[1].metric("ARPU", f"${s['arpu']:,.2f}")
        c[2].metric("Revenue at risk", f"${s['revenue_at_risk']:,.0f}/mo")
        c[3].metric("Network health", f"{s['network_health_pct']}%")
        c2 = st.columns(3)
        c2[0].metric("High-risk customers", f"{s['high_risk_customers']:,}")
        c2[1].metric("Fraud alerts", f"{s['fraud_alerts']:,}")
        c2[2].metric("Towers at risk", f"{s['towers_at_risk']}")
        st.markdown("---")
        st.markdown(result["summary"])
        st.caption(f"Source: {result['source']}")

# ─────────────────────────── Churn ───────────────────────────
with tab_churn:
    st.subheader("Churn Predictor")
    df = load_csv("customers.csv")
    if not df.empty:
        st.plotly_chart(
            px.histogram(df, x="contract", color="churn", barmode="group",
                         title="Historical churn by contract type"),
            use_container_width=True,
        )
    with st.form("churn_form"):
        c = st.columns(3)
        tenure = c[0].number_input("Tenure (months)", 0, 120, 4)
        monthly = c[1].number_input("Monthly charges", 0.0, 200.0, 95.0)
        support = c[2].number_input("Support calls", 0, 30, 5)
        c2 = st.columns(3)
        contract = c2[0].selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        internet = c2[1].selectbox("Internet", ["Fiber optic", "DSL", "No"])
        payment = c2[2].selectbox("Payment", ["Electronic check", "Mailed check", "Bank transfer", "Credit card"])
        submitted = st.form_submit_button("Predict churn risk")
    if submitted:
        from dataclasses import asdict

        res = asdict(churn_model.predict_one({
            "tenure_months": tenure, "monthly_charges": monthly, "support_calls": support,
            "total_charges": monthly * max(tenure, 1), "contract": contract,
            "internet_service": internet, "payment_method": payment,
            "paperless_billing": 1, "senior_citizen": 0,
        }))
        band = res["risk_band"]
        st.metric("Churn risk", f"{res['risk_score']*100:.1f}%", band)
        st.write({"HIGH": "🔴", "MEDIUM": "🟠", "LOW": "🟢"}[band] + f" **{band} risk**")
        st.write("**Why:**")
        for r in res["reasons"]:
            st.write(f"- {r}")
        st.success(f"**Recommended action:** {res['recommended_action']}")

# ─────────────────────────── Fraud ───────────────────────────
with tab_fraud:
    st.subheader("Fraud Detector — real-time CDR scoring")
    with st.form("fraud_form"):
        c = st.columns(3)
        duration = c[0].number_input("Duration (sec)", 0, 14400, 3600)
        cost = c[1].number_input("Cost ($)", 0.0, 500.0, 48.0)
        hour = c[2].number_input("Hour of day", 0, 23, 3)
        c2 = st.columns(3)
        intl = c2[0].checkbox("International", True)
        premium = c2[1].checkbox("Premium-rate", True)
        calls = c2[2].number_input("Calls last hour", 0, 100, 30)
        unique = st.number_input("Unique numbers called", 0, 100, 18)
        submitted = st.form_submit_button("Score for fraud")
    if submitted:
        from dataclasses import asdict

        res = asdict(fraud_model.score_one({
            "duration_sec": duration, "cost": cost, "hour_of_day": hour,
            "is_international": int(intl), "is_premium_rate": int(premium),
            "calls_last_hour": calls, "unique_numbers_called": unique,
        }))
        st.metric("Anomaly score", f"{res['anomaly_score']*100:.1f}%", res["severity"])
        st.error("🚨 FRAUD ALERT") if res["is_alert"] else st.success("✅ No alert")
        for r in res["reasons"]:
            st.write(f"- {r}")

# ─────────────────────────── Network ───────────────────────────
with tab_net:
    st.subheader("Network Anomaly Detection")
    df = load_csv("network.csv")
    if not df.empty:
        st.plotly_chart(
            px.scatter(df.sample(min(3000, len(df))), x="latency_ms", y="packet_loss_pct",
                       color="is_failure", title="Latency vs packet loss (failures highlighted)"),
            use_container_width=True,
        )
    with st.form("net_form"):
        c = st.columns(3)
        cpu = c[0].number_input("CPU %", 0.0, 100.0, 95.0)
        latency = c[1].number_input("Latency (ms)", 0.0, 400.0, 220.0)
        loss = c[2].number_input("Packet loss %", 0.0, 100.0, 28.0)
        c2 = st.columns(3)
        throughput = c2[0].number_input("Throughput (Mbps)", 0.0, 1500.0, 120.0)
        dropped = c2[1].number_input("Dropped calls", 0, 200, 40)
        temp = c2[2].number_input("Temperature °C", 0.0, 100.0, 78.0)
        signal = st.number_input("Signal (dBm)", -120.0, -40.0, -105.0)
        submitted = st.form_submit_button("Score telemetry")
    if submitted:
        from dataclasses import asdict

        res = asdict(anomaly_model.score_one({
            "cpu_utilization": cpu, "latency_ms": latency, "packet_loss_pct": loss,
            "throughput_mbps": throughput, "dropped_calls": dropped,
            "temperature_c": temp, "signal_strength_dbm": signal,
        }))
        st.metric("Anomaly score", f"{res['anomaly_score']*100:.1f}%", res["severity"])
        if res["is_anomaly"]:
            st.error(f"⚠️ Anomaly — likely cause: {res['root_cause']}")
        else:
            st.success("✅ Healthy")

st.markdown("---")
st.caption("⚙️ Full distributed architecture (Kafka · Spark · FastAPI · Postgres · PyTorch) "
           "is in the GitHub repo. This hosted demo runs the models in-process for a free live deploy.")
