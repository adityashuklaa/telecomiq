"""Module 4 — Revenue Intelligence.

Aggregates the outputs of the churn, fraud, and network models into a single
business snapshot, then asks Claude to turn the numbers into a plain-English,
CEO-level briefing (ARPU trends, churn risk, fraud losses, network health).

If no ANTHROPIC_API_KEY is configured, a deterministic template summary is
returned so the dashboard still works offline.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

from config.settings import configure_logging, get_settings

log = configure_logging("insights.claude")

SYSTEM_PROMPT = (
    "You are the Chief Data Officer of a telecom operator briefing the CEO. "
    "You receive a JSON snapshot of analytics metrics. Write a concise, decision-ready "
    "executive summary in plain English. Be specific with numbers, lead with the most "
    "material business risk, quantify revenue impact where possible, and end with 3 "
    "prioritized recommended actions. Do not invent data not present in the snapshot. "
    "Use short paragraphs and a final bulleted action list."
)


@dataclass
class RevenueSnapshot:
    total_customers: int
    high_risk_customers: int
    high_risk_pct: float
    monthly_revenue: float
    arpu: float
    revenue_at_risk: float
    fraud_alerts: int
    estimated_fraud_loss: float
    towers_monitored: int
    towers_at_risk: int
    network_health_pct: float


def build_snapshot(
    churn_df,
    fraud_df,
    network_df,
) -> RevenueSnapshot:
    """Compute the aggregate business snapshot from scored dataframes."""
    total = len(churn_df)
    high = int((churn_df["risk_band"] == "HIGH").sum()) if "risk_band" in churn_df else 0
    monthly_rev = float(churn_df["monthly_charges"].sum()) if "monthly_charges" in churn_df else 0.0
    arpu = monthly_rev / total if total else 0.0
    # Revenue at risk = monthly charges of HIGH risk customers
    rev_at_risk = (
        float(churn_df.loc[churn_df["risk_band"] == "HIGH", "monthly_charges"].sum())
        if "risk_band" in churn_df and "monthly_charges" in churn_df
        else 0.0
    )

    alerts = int(fraud_df["is_alert"].sum()) if "is_alert" in fraud_df else 0
    fraud_loss = float(fraud_df.loc[fraud_df["is_alert"], "cost"].sum()) if "is_alert" in fraud_df else 0.0

    towers = int(network_df["tower_id"].nunique()) if "tower_id" in network_df else 0
    at_risk_rows = network_df[network_df["is_anomaly"]] if "is_anomaly" in network_df else network_df.iloc[0:0]
    towers_at_risk = int(at_risk_rows["tower_id"].nunique()) if "tower_id" in at_risk_rows else 0
    health = 100.0 * (1 - towers_at_risk / towers) if towers else 100.0

    return RevenueSnapshot(
        total_customers=total,
        high_risk_customers=high,
        high_risk_pct=round(100 * high / total, 2) if total else 0.0,
        monthly_revenue=round(monthly_rev, 2),
        arpu=round(arpu, 2),
        revenue_at_risk=round(rev_at_risk, 2),
        fraud_alerts=alerts,
        estimated_fraud_loss=round(fraud_loss, 2),
        towers_monitored=towers,
        towers_at_risk=towers_at_risk,
        network_health_pct=round(health, 2),
    )


def _template_summary(snap: RevenueSnapshot) -> str:
    return (
        f"**Executive Summary (offline template — set ANTHROPIC_API_KEY for AI narrative)**\n\n"
        f"We serve {snap.total_customers:,} subscribers generating "
        f"${snap.monthly_revenue:,.0f}/month (ARPU ${snap.arpu:,.2f}). "
        f"{snap.high_risk_customers:,} customers ({snap.high_risk_pct}%) are at HIGH churn risk, "
        f"putting ${snap.revenue_at_risk:,.0f}/month of recurring revenue in jeopardy. "
        f"Fraud monitoring raised {snap.fraud_alerts:,} alerts with an estimated "
        f"${snap.estimated_fraud_loss:,.0f} in exposed call cost. "
        f"Network health is {snap.network_health_pct}% with {snap.towers_at_risk} of "
        f"{snap.towers_monitored} towers flagged for intervention.\n\n"
        f"**Recommended actions:**\n"
        f"- Launch a targeted retention campaign for HIGH-risk subscribers.\n"
        f"- Investigate and block the flagged fraudulent call patterns.\n"
        f"- Dispatch maintenance to at-risk towers before failures occur."
    )


def generate_ceo_summary(snap: RevenueSnapshot) -> dict:
    """Return {summary, snapshot, source}. Uses Claude when a key is present."""
    settings = get_settings()
    snapshot_json = asdict(snap)

    if not settings.anthropic_api_key:
        log.warning("ANTHROPIC_API_KEY not set — returning template summary.")
        return {"summary": _template_summary(snap), "snapshot": snapshot_json, "source": "template"}

    try:
        import anthropic

        client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
        import json

        message = client.messages.create(
            model=settings.claude_model,
            max_tokens=1200,
            system=SYSTEM_PROMPT,
            thinking={"type": "adaptive"},
            messages=[
                {
                    "role": "user",
                    "content": (
                        "Here is today's TelecomIQ analytics snapshot (JSON). "
                        "Write the CEO briefing.\n\n" + json.dumps(snapshot_json, indent=2)
                    ),
                }
            ],
        )
        text = next((b.text for b in message.content if b.type == "text"), "")
        log.info("Generated Claude CEO summary (%d chars).", len(text))
        return {"summary": text, "snapshot": snapshot_json, "source": settings.claude_model}
    except Exception as exc:  # pragma: no cover - network/auth failures
        log.error("Claude call failed (%s) — falling back to template.", exc)
        return {"summary": _template_summary(snap), "snapshot": snapshot_json, "source": "template-fallback"}
