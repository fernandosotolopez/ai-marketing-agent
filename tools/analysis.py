from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from registry.tool_registry import register_tool


@dataclass
class Insight:
    """
    A single interpretable insight derived from campaign metrics.
    """
    category: str           # e.g. "efficiency", "trend", "risk", "maturity"
    message: str            # human-readable
    importance: str         # "low" | "medium" | "high"


@dataclass
class AnalysisResult:
    """
    Structured output from analyzing one campaign row.
    """
    campaign_id: str
    overall_risk: str                 # "low" | "medium" | "high"
    summary: str
    insights: List[Insight]
    suggested_actions: List[str]
    debug: Dict[str, Any]

@register_tool(tags=["analysis", "insights"])
def analyze_campaign_row(
    row: Dict[str, Any],
    decision: Optional[Dict[str, Any]] = None,
) -> AnalysisResult:
    """
    Analyze one enriched campaign row and produce insights + suggested actions.

    Input expectation:
      row has keys like CPA, ROAS, target_CPA, CPA_trend_7d, ROAS_trend_7d, days_active
      (ideally after validate_and_enrich_row from tools/metrics.py)

    decision (optional):
      agent decision dict like {"stance": "...", "severity": "...", "reasons": [...]}
      If provided, analysis will align its suggested actions to the stance.
    """
    campaign_id = str(row.get("campaign_id") or "UNKNOWN")

    cpa = row.get("CPA")
    roas = row.get("ROAS")
    target_cpa = row.get("target_CPA")
    cpa_trend = row.get("CPA_trend_7d")
    roas_trend = row.get("ROAS_trend_7d")
    days_active = row.get("days_active")

    insights: List[Insight] = []
    suggested_actions: List[str] = []
    debug: Dict[str, Any] = {}

    # -------------------------
    # Maturity context
    # -------------------------
    if isinstance(days_active, int) and days_active < 14:
        insights.append(Insight(
            category="maturity",
            message=f"Campaign is still early-stage ({days_active} days active). Early volatility is normal.",
            importance="medium",
        ))
        suggested_actions.append("Collect more data before making aggressive changes (avoid overreacting early).")

    # -------------------------
    # Efficiency: CPA vs target
    # -------------------------
    overage_pct: Optional[float] = None
    if isinstance(cpa, (int, float)) and isinstance(target_cpa, (int, float)) and target_cpa > 0:
        overage_pct = (cpa - target_cpa) / target_cpa
        debug["cpa_overage_pct"] = overage_pct

        if overage_pct <= 0:
            insights.append(Insight(
                category="efficiency",
                message=f"CPA is within target (CPA={cpa:.2f} <= target={target_cpa:.2f}).",
                importance="low",
            ))
        elif overage_pct < 0.20:
            insights.append(Insight(
                category="efficiency",
                message=f"CPA is above target by {overage_pct:.0%} (CPA={cpa:.2f}, target={target_cpa:.2f}).",
                importance="medium",
            ))
            suggested_actions.append("Inspect audience/creative/landing page friction; test small optimizations before pausing.")
        else:
            insights.append(Insight(
                category="efficiency",
                message=f"CPA is significantly above target by {overage_pct:.0%} (CPA={cpa:.2f}, target={target_cpa:.2f}).",
                importance="high",
            ))
            suggested_actions.append("Consider reducing budget or pausing if performance does not improve after targeted tests.")

    else:
        insights.append(Insight(
            category="efficiency",
            message="Could not evaluate CPA vs target (missing CPA or valid target_CPA).",
            importance="medium",
        ))
        suggested_actions.append("Ensure CPA and target_CPA are available (or set a default target_CPA).")

    # -------------------------
    # Trends: CPA & ROAS movement
    # -------------------------
    if isinstance(cpa_trend, (int, float)):
        if cpa_trend > 0.25:
            insights.append(Insight(
                category="trend",
                message=f"CPA is worsening fast (+{cpa_trend:.0%} over last 7d).",
                importance="high",
            ))
            suggested_actions.append("Investigate what changed recently (targeting, creative fatigue, landing page, tracking).")
        elif cpa_trend > 0.10:
            insights.append(Insight(
                category="trend",
                message=f"CPA is trending worse (+{cpa_trend:.0%} over last 7d).",
                importance="medium",
            ))
        elif cpa_trend < -0.10:
            insights.append(Insight(
                category="trend",
                message=f"CPA is improving ({cpa_trend:.0%} over last 7d).",
                importance="low",
            ))
    else:
        insights.append(Insight(
            category="trend",
            message="CPA trend unavailable (CPA_trend_7d missing).",
            importance="low",
        ))

    if isinstance(roas_trend, (int, float)):
        if roas_trend < -0.25:
            insights.append(Insight(
                category="trend",
                message=f"ROAS is dropping fast ({roas_trend:.0%} over last 7d).",
                importance="high",
            ))
            suggested_actions.append("Check funnel changes: conversion rate, AOV, attribution windows, offer competitiveness.")
        elif roas_trend < -0.10:
            insights.append(Insight(
                category="trend",
                message=f"ROAS is trending worse ({roas_trend:.0%} over last 7d).",
                importance="medium",
            ))
        elif roas_trend > 0.10:
            insights.append(Insight(
                category="trend",
                message=f"ROAS is improving (+{roas_trend:.0%} over last 7d).",
                importance="low",
            ))
    else:
        insights.append(Insight(
            category="trend",
            message="ROAS trend unavailable (ROAS_trend_7d missing).",
            importance="low",
        ))

    # -------------------------
    # Absolute ROAS sanity (helpful context)
    # -------------------------
    if isinstance(roas, (int, float)):
        if roas < 1.0:
            insights.append(Insight(
                category="risk",
                message=f"ROAS < 1.0 (ROAS={roas:.2f}) suggests you may be losing money on ad spend (before LTV considerations).",
                importance="high",
            ))
            suggested_actions.append("If ROAS stays < 1.0, reduce spend and isolate profitable segments/creatives.")
        elif roas < 2.0:
            insights.append(Insight(
                category="risk",
                message=f"ROAS is modest (ROAS={roas:.2f}). Profitability depends on margins/LTV.",
                importance="medium",
            ))
        else:
            insights.append(Insight(
                category="risk",
                message=f"ROAS looks strong (ROAS={roas:.2f}).",
                importance="low",
            ))

    # -------------------------
    # Align suggestions with agent decision (if provided)
    # -------------------------
    if decision is not None:
        stance = decision.get("stance")
        severity = decision.get("severity")
        debug["agent_stance"] = stance
        debug["agent_severity"] = severity

        # If agent says observe, soften actions to monitoring
        if stance == "observe":
            suggested_actions = [
                "Monitor performance; avoid large changes until more evidence accumulates."
            ] + [a for a in suggested_actions if "Collect more data" in a]

        # If agent escalates, add stakeholder-facing action
        if stance == "escalate":
            suggested_actions.insert(0, "Escalate to a marketing owner with context and supporting metrics.")

    # -------------------------
    # Overall risk score (simple heuristic)
    # -------------------------
    risk = "low"
    if any(i.importance == "high" for i in insights):
        risk = "high"
    elif any(i.importance == "medium" for i in insights):
        risk = "medium"

    # Summary line
    summary = _build_summary(campaign_id, cpa, target_cpa, roas, cpa_trend, roas_trend, risk)

    return AnalysisResult(
        campaign_id=campaign_id,
        overall_risk=risk,
        summary=summary,
        insights=insights,
        suggested_actions=_dedupe_keep_order(suggested_actions),
        debug=debug,
    )


def _build_summary(
    campaign_id: str,
    cpa: Any,
    target_cpa: Any,
    roas: Any,
    cpa_trend: Any,
    roas_trend: Any,
    risk: str,
) -> str:
    def fmt(x: Any, kind: str) -> str:
        if isinstance(x, (int, float)):
            if kind == "pct":
                return f"{x:+.0%}"
            return f"{x:.2f}"
        return "N/A"

    return (
        f"[{campaign_id}] risk={risk} | "
        f"CPA={fmt(cpa,'num')} (target={fmt(target_cpa,'num')}) | "
        f"ROAS={fmt(roas,'num')} | "
        f"CPA_trend_7d={fmt(cpa_trend,'pct')} | "
        f"ROAS_trend_7d={fmt(roas_trend,'pct')}"
    )


def _dedupe_keep_order(items: List[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for x in items:
        if x not in seen:
            out.append(x)
            seen.add(x)
    return out
