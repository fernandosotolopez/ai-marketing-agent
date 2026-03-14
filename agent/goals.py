from dataclasses import dataclass


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
        cpa_trend = state.get("CPA_trend_7d")
        roas_trend = state.get("ROAS_trend_7d")

        if cpa_trend is None or roas_trend is None:
            return GoalResult(
                goal_name=self.name,
                met=False,
                message="Missing CPA_trend_7d or ROAS_trend_7d in state",
                severity="high"
            )

        issues = []

        if cpa_trend > 0.10:
            issues.append(f"CPA increased by {cpa_trend:.0%}")

        if roas_trend < -0.10:
            issues.append(f"ROAS decreased by {abs(roas_trend):.0%}")

        if not issues:
            return GoalResult(
                goal_name=self.name,
                met=True,
                message="No significant negative performance trends detected",
                severity="low"
            )

        severe = (cpa_trend > 0.25) or (roas_trend < -0.25)
        severity = "high" if severe else "medium"

        return GoalResult(
            goal_name=self.name,
            met=False,
            message="; ".join(issues),
            severity=severity
        )


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
