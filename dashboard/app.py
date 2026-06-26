"""TelecomIQ — Streamlit dashboard.

Live, interactive front-end for all four modules. Talks to the FastAPI service
for single-record scoring and reads the generated datasets directly for the
fleet-wide charts.

Run:  streamlit run dashboard/app.py
"""
from __future__ import annotations

import os

import pandas as pd
import plotly.express as px
import requests
import streamlit as st

API_BASE = os.getenv("API_BASE_URL", "http://localhost:8000")
DATA_DIR = os.getenv("DATA_DIR", "data/generated")

st.set_page_config(page_title="TelecomIQ", page_icon="📡", layout="wide")


@st.cache_data(ttl=60)
def load_csv(name: str) -> pd.DataFrame:
    path = os.path.join(DATA_DIR, name)
    return pd.read_csv(path) if os.path.exists(path) else pd.DataFrame()


def api_post(path: str, payload: dict) -> dict | None:
    try:
        r = requests.post(f"{API_BASE}{path}", json=payload, timeout=30)
        r.raise_for_status()
        return r.json()
    except Exception as exc:
        st.error(f"API error calling {path}: {exc}")
        return None


def api_get(path: str) -> dict | None:
    try:
        r = requests.get(f"{API_BASE}{path}", timeout=120)
        r.raise_for_status()
        return r.json()
    except Exception as exc:
        st.error(f"API error calling {path}: {exc}")
        return None


st.title("📡 TelecomIQ — AI Telecom Analytics")
st.caption(f"API: {API_BASE}")

health = api_get("/health")
if health:
    cols = st.columns(3)
    for col, (name, state) in zip(cols, health.get("models", {}).items()):
        col.metric(name.title() + " model", state)

tab_rev, tab_churn, tab_fraud, tab_net = st.tabs(
    ["💰 Revenue Intelligence", "🔮 Churn", "🚨 Fraud", "🛰️ Network"]
)

# ─────────────────────────── Revenue Intelligence ───────────────────────────
with tab_rev:
    st.subheader("Revenue Intelligence — CEO Briefing")
    st.write("Aggregates churn, fraud and network signals into a Claude-written summary.")
    if st.button("Generate executive briefing", type="primary"):
        with st.spinner("Scoring fleet & asking Claude..."):
            result = api_get("/revenue/summary")
        if result:
            snap = result.get("snapshot", {})
            c = st.columns(4)
            c[0].metric("Customers", f"{snap.get('total_customers', 0):,}")
            c[1].metric("ARPU", f"${snap.get('arpu', 0):,.2f}")
            c[2].metric("Revenue at risk", f"${snap.get('revenue_at_risk', 0):,.0f}/mo")
            c[3].metric("Network health", f"{snap.get('network_health_pct', 0)}%")
            c2 = st.columns(3)
            c2[0].metric("High-risk customers", f"{snap.get('high_risk_customers', 0):,}")
            c2[1].metric("Fraud alerts", f"{snap.get('fraud_alerts', 0):,}")
            c2[2].metric("Towers at risk", f"{snap.get('towers_at_risk', 0)}")
            st.markdown("---")
            st.markdown(result.get("summary", "_no summary_"))
            st.caption(f"Source: {result.get('source')}")

# ─────────────────────────── Churn ───────────────────────────
with tab_churn:
    st.subheader("Churn Predictor")
    df = load_csv("customers.csv")
    if not df.empty:
        fig = px.histogram(df, x="contract", color="churn", barmode="group",
                           title="Historical churn by contract type")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("**Score a customer**")
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
        res = api_post("/churn/predict", {
            "tenure_months": tenure, "monthly_charges": monthly, "support_calls": support,
            "total_charges": monthly * max(tenure, 1), "contract": contract,
            "internet_service": internet, "payment_method": payment,
            "paperless_billing": 1, "senior_citizen": 0,
        })
        if res:
            band = res["risk_band"]
            color = {"HIGH": "🔴", "MEDIUM": "🟠", "LOW": "🟢"}[band]
            st.metric("Churn risk", f"{res['risk_score']*100:.1f}%", band)
            st.write(f"{color} **{band} risk**")
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
        res = api_post("/fraud/score", {
            "duration_sec": duration, "cost": cost, "hour_of_day": hour,
            "is_international": int(intl), "is_premium_rate": int(premium),
            "calls_last_hour": calls, "unique_numbers_called": unique,
        })
        if res:
            st.metric("Anomaly score", f"{res['anomaly_score']*100:.1f}%", res["severity"])
            if res["is_alert"]:
                st.error("🚨 FRAUD ALERT")
            else:
                st.success("✅ No alert")
            for r in res["reasons"]:
                st.write(f"- {r}")

# ─────────────────────────── Network ───────────────────────────
with tab_net:
    st.subheader("Network Anomaly Detection")
    df = load_csv("network.csv")
    if not df.empty:
        fig = px.scatter(df.sample(min(3000, len(df))), x="latency_ms", y="packet_loss_pct",
                         color="is_failure", title="Latency vs packet loss (failures highlighted)")
        st.plotly_chart(fig, use_container_width=True)

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
        res = api_post("/anomaly/score", {
            "cpu_utilization": cpu, "latency_ms": latency, "packet_loss_pct": loss,
            "throughput_mbps": throughput, "dropped_calls": dropped,
            "temperature_c": temp, "signal_strength_dbm": signal,
        })
        if res:
            st.metric("Anomaly score", f"{res['anomaly_score']*100:.1f}%", res["severity"])
            if res["is_anomaly"]:
                st.error(f"⚠️ Anomaly — likely cause: {res['root_cause']}")
            else:
                st.success("✅ Healthy")
