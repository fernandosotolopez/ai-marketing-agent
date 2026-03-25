from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

from tools.metric_definitions import (
    SOURCE_CSV_SUPPLIED,
    SOURCE_LEGACY_OR_UNKNOWN,
    SOURCE_MEMORY_SNAPSHOT_DERIVED,
    SOURCE_METRICS_PERCENT_RESCALED,
    SOURCE_MISSING,
    get_field_provenance,
)


@dataclass
class GoalResult:
    """
    Result of evaluating a goal against the current agent state.
    """
    goal_name: str
    met: bool
    message: str
    severity: str  # "low", "medium", "high"


class CostEfficiencyGoal:
    """
    Evaluates whether a campaign's CPA is within the acceptable target.
    """
    name = "cost_efficiency"

    def evaluate(self, state: dict) -> GoalResult:
        cpa = state.get("CPA")
        target_cpa = state.get("target_CPA")

        if cpa is None or target_cpa is None:
            return GoalResult(
                goal_name=self.name,
                met=False,
                message="Missing CPA or target_CPA in state",
                severity="high"
            )

        # Safety: avoid division by zero / bad config
        if target_cpa <= 0:
            return GoalResult(
                goal_name=self.name,
                met=False,
                message="Invalid target_CPA (must be > 0)",
                severity="high"
            )

        if cpa <= target_cpa:
            return GoalResult(
                goal_name=self.name,
                met=True,
                message=f"CPA ({cpa:.2f}) is within target ({target_cpa:.2f})",
                severity="low"
            )

        overage_pct = (cpa - target_cpa) / target_cpa
        severity = "medium" if overage_pct < 0.2 else "high"

        return GoalResult(
            goal_name=self.name,
            met=False,
            message=(
                f"CPA ({cpa:.2f}) exceeds target ({target_cpa:.2f}) "
                f"by {overage_pct:.0%}"
            ),
            severity=severity
        )


class PerformanceTrendGoal:
    """
    Evaluates whether campaign performance is degrading over time.
    """
    name = "performance_trend"

    def evaluate(self, state: dict) -> GoalResult:
        cpa_result = self._evaluate_trend_signal(
            state=state,
            field_name="CPA_trend_7d",
            metric_label="CPA",
            issue_when=lambda value: value > 0.10,
            severe_when=lambda value: value > 0.25,
            format_issue=lambda value: f"CPA increased by {value:.0%}",
        )
        roas_result = self._evaluate_trend_signal(
            state=state,
            field_name="ROAS_trend_7d",
            metric_label="ROAS",
            issue_when=lambda value: value < -0.10,
            severe_when=lambda value: value < -0.25,
            format_issue=lambda value: f"ROAS decreased by {abs(value):.0%}",
        )

        results = [cpa_result, roas_result]
        issues = [r for r in results if r["status"] == "issue"]
        weak_notes = [r["message"] for r in results if r["status"] == "weak"]
        missing_notes = [r["message"] for r in results if r["status"] == "missing"]

        if issues:
            severity = "high" if any(r["severity"] == "high" for r in issues) else "medium"
            messages = [r["message"] for r in issues]
            messages.extend(missing_notes)
            messages.extend(weak_notes)
            return GoalResult(
                goal_name=self.name,
                met=False,
                message="; ".join(messages),
                severity=severity,
            )

        if len(missing_notes) == len(results):
            return GoalResult(
                goal_name=self.name,
                met=True,
                message="Trend evidence unavailable; no negative trend action taken",
                severity="low",
            )

        if len(missing_notes) + len(weak_notes) == len(results):
            return GoalResult(
                goal_name=self.name,
                met=True,
                message="Trend evidence is weak or partial; no negative trend action taken",
                severity="low",
            )

        message = "No significant negative performance trends detected from available evidence"
        notes = missing_notes + weak_notes
        if notes:
            message = f"{message}; " + "; ".join(notes)

        return GoalResult(
            goal_name=self.name,
            met=True,
            message=message,
            severity="low",
        )

    def _evaluate_trend_signal(
        self,
        *,
        state: Dict[str, Any],
        field_name: str,
        metric_label: str,
        issue_when: Callable[[float], bool],
        severe_when: Callable[[float], bool],
        format_issue: Callable[[float], str],
    ) -> Dict[str, str]:
        raw_value = state.get(field_name)
        provenance = get_field_provenance(state, field_name)
        source = provenance.get("source") if provenance else None

        if raw_value is None or source == SOURCE_MISSING:
            return {
                "status": "missing",
                "message": f"{metric_label} trend unavailable",
            }

        try:
            value = float(raw_value)
        except Exception:
            return {
                "status": "missing",
                "message": f"{metric_label} trend unavailable",
            }

        evidence_note = self._trend_evidence_note(source, provenance)
        weak_evidence = self._is_weak_trend_source(source)

        if not issue_when(value):
            if weak_evidence:
                return {
                    "status": "weak",
                    "message": f"{metric_label} trend is non-negative but only weakly verified ({evidence_note})",
                }
            return {"status": "ok", "message": ""}

        severity = "high" if severe_when(value) else "medium"

        if weak_evidence:
            severity = self._downgrade_severity(severity)

        issue_message = format_issue(value)
        if evidence_note:
            issue_message = f"{issue_message} ({evidence_note})"

        return {
            "status": "issue",
            "severity": severity,
            "message": issue_message,
        }

    def _is_weak_trend_source(self, source: Optional[str]) -> bool:
        return source in {
            None,
            SOURCE_MEMORY_SNAPSHOT_DERIVED,
            SOURCE_LEGACY_OR_UNKNOWN,
            SOURCE_METRICS_PERCENT_RESCALED,
        }

    def _trend_evidence_note(
        self,
        source: Optional[str],
        provenance: Optional[Dict[str, Any]],
    ) -> str:
        semantics = ""
        if provenance:
            semantics = str(provenance.get("semantics") or provenance.get("detail") or "")

        if source == SOURCE_CSV_SUPPLIED:
            return semantics or "CSV-supplied trend semantics from provider"
        if source == SOURCE_MEMORY_SNAPSHOT_DERIVED:
            return semantics or "snapshot-derived trend, not verified calendar 7-day"
        if source == SOURCE_METRICS_PERCENT_RESCALED:
            return semantics or "trend value was rescaled from percent-like input"
        if source == SOURCE_LEGACY_OR_UNKNOWN or source is None:
            return semantics or "trend provenance is unverified"
        return semantics

    def _downgrade_severity(self, severity: str) -> str:
        if severity == "high":
            return "medium"
        if severity == "medium":
            return "low"
        return severity


class MaturityGoal:
    """
    Ensures the campaign has enough historical data
    before allowing aggressive optimization decisions.
    """
    name = "campaign_maturity"

    def __init__(self, min_days_active: int = 14):
        self.min_days_active = min_days_active

    def evaluate(self, state: dict) -> GoalResult:
        days_active = state.get("days_active")

        if days_active is None:
            return GoalResult(
                goal_name=self.name,
                met=False,
                message="Missing days_active in state",
                severity="high"
            )

        if days_active < self.min_days_active:
            return GoalResult(
                goal_name=self.name,
                met=False,
                message=(
                    f"Campaign is immature "
                    f"({days_active} days active; "
                    f"minimum {self.min_days_active} required)"
                ),
                severity="medium"
            )

        return GoalResult(
            goal_name=self.name,
            met=True,
            message="Campaign has sufficient history for evaluation",
            severity="low"
        )
