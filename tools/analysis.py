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
        goal_evaluations = decision.get("goal_evaluations") or []
        recommendation_hierarchy = decision.get("recommendation_hierarchy") or []
        debug["goal_evaluations"] = goal_evaluations
        debug["recommendation_hierarchy"] = recommendation_hierarchy

        if goal_evaluations:
            insights = _merge_goal_evaluation_insights(insights, goal_evaluations)

        deterministic_actions = _build_actions_from_decision(
            decision=decision,
            fallback_actions=suggested_actions,
        )
        if deterministic_actions:
            suggested_actions = deterministic_actions

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


def _build_actions_from_decision(
    *,
    decision: Dict[str, Any],
    fallback_actions: List[str],
) -> List[str]:
    stance = str(decision.get("stance") or "observe")
    hierarchy = decision.get("recommendation_hierarchy") or []
    goal_evaluations = decision.get("goal_evaluations") or []

    if hierarchy:
        return _hierarchy_actions(stance=stance, hierarchy=hierarchy)

    if goal_evaluations:
        return _goal_evaluation_actions(stance=stance, goal_evaluations=goal_evaluations)

    if stance == "observe":
        return ["Monitor performance; avoid large changes until more evidence accumulates."]
    if stance == "escalate":
        return ["Escalate to a marketing owner with context and supporting metrics."] + fallback_actions
    return fallback_actions


def _hierarchy_actions(*, stance: str, hierarchy: List[Dict[str, Any]]) -> List[str]:
    actions: List[str] = []
    for item in hierarchy:
        label = "Primary" if int(item.get("priority") or 0) == 1 else "Secondary"
        goal_name = str(item.get("goal_name") or "goal")
        role = str(item.get("role") or "")
        message = str(item.get("message") or item.get("reason") or "").strip()
        if not message:
            continue
        actions.append(f"{label}: {_action_prefix(stance, goal_name, role, message)}")
    return _dedupe_keep_order(actions)


def _goal_evaluation_actions(*, stance: str, goal_evaluations: List[Dict[str, Any]]) -> List[str]:
    unmet = [g for g in goal_evaluations if not g.get("met")]
    if unmet:
        synthetic_hierarchy = [
            {
                "priority": idx,
                "goal_name": g.get("goal_name"),
                "role": "primary_driver" if idx == 1 else "supporting_driver",
                "message": g.get("message"),
            }
            for idx, g in enumerate(unmet, start=1)
        ]
        return _hierarchy_actions(stance=stance, hierarchy=synthetic_hierarchy)

    evidence_notes = [
        str(g.get("message") or "").strip()
        for g in goal_evaluations
        if _message_is_uncertain(str(g.get("message") or ""))
    ]
    if evidence_notes:
        return [
            "Primary: continue observing; no evaluated goal currently supports intervention.",
            f"Secondary: keep confidence modest because {evidence_notes[0]}.",
        ]

    return ["Primary: continue observing; available goal checks do not support a change."]


def _action_prefix(stance: str, goal_name: str, role: str, message: str) -> str:
    evidence_qualified = _message_is_uncertain(message)

    if role == "no_action" or goal_name == "all_goals":
        return "continue observing; available goal checks do not support a change."

    if role == "blocking_constraint":
        return f"stay in observe mode because {message}."

    if role == "deferred_context":
        return f"keep this as supporting context while the primary gate is unresolved: {message}."

    if goal_name == "cost_efficiency" and _message_mentions_configuration_issue(message):
        return f"verify CPA/target configuration before changing spend because {message}."

    if evidence_qualified:
        if stance == "escalate":
            return f"escalate for review, but keep the evidence caveat explicit because {message}."
        if stance == "recommend":
            return f"run a targeted review, but treat the evidence as provisional because {message}."
        return f"continue monitoring because {message}."

    if stance == "escalate":
        return f"escalate to a marketing owner for review because {message}."
    if stance == "recommend":
        return f"run a targeted optimization review because {message}."
    return f"continue monitoring because {message}."


def _message_mentions_configuration_issue(message: str) -> bool:
    lowered = message.lower()
    return "missing cpa" in lowered or "target_cpa" in lowered or "invalid target_cpa" in lowered


def _message_is_uncertain(message: str) -> bool:
    lowered = message.lower()
    markers = (
        "weak",
        "partial",
        "unavailable",
        "unverified",
        "not verified",
        "snapshot-derived",
        "collect more data",
    )
    return any(marker in lowered for marker in markers)


def _merge_goal_evaluation_insights(
    insights: List[Insight],
    goal_evaluations: List[Dict[str, Any]],
) -> List[Insight]:
    existing = {(ins.category, ins.message) for ins in insights}
    merged = list(insights)

    for evaluation in goal_evaluations:
        message = str(evaluation.get("message") or "").strip()
        if not message:
            continue
        if not _should_surface_goal_evaluation(evaluation):
            continue

        category = _goal_category(str(evaluation.get("goal_name") or ""))
        insight = Insight(
            category=category,
            message=message,
            importance=_goal_importance(evaluation),
        )
        key = (insight.category, insight.message)
        if key not in existing:
            merged.append(insight)
            existing.add(key)

    return merged


def _should_surface_goal_evaluation(evaluation: Dict[str, Any]) -> bool:
    if not evaluation.get("met"):
        return False
    return _message_is_uncertain(str(evaluation.get("message") or ""))


def _goal_category(goal_name: str) -> str:
    return {
        "campaign_maturity": "maturity",
        "cost_efficiency": "efficiency",
        "performance_trend": "trend",
    }.get(goal_name, "risk")


def _goal_importance(evaluation: Dict[str, Any]) -> str:
    severity = str(evaluation.get("severity") or "low").lower()
    if severity in {"low", "medium", "high"}:
        return severity
    return "low"
