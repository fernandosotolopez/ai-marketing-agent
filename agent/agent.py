from typing import Any, Dict, List
from agent.goals import GoalResult


class AgentDecision:
    """
    Final decision produced by the agent after evaluating all goals.
    """

    def __init__(
        self,
        stance: str,
        severity: str,
        reasons: List[str],
        goal_evaluations: List[Dict[str, Any]],
        recommendation_hierarchy: List[Dict[str, Any]],
    ):
        # stance: "observe" | "recommend" | "escalate"
        self.stance = stance
        self.severity = severity  # "low" | "medium" | "high"
        self.reasons = reasons
        self.goal_evaluations = goal_evaluations
        self.recommendation_hierarchy = recommendation_hierarchy

    def to_dict(self) -> Dict:
        return {
            "stance": self.stance,
            "severity": self.severity,
            "reasons": self.reasons,
            "goal_evaluations": self.goal_evaluations,
            "recommendation_hierarchy": self.recommendation_hierarchy,
        }


class MarketingAgent:
    """
    Evaluates campaign state against business goals and produces a single decision.
    The agent is the ONLY place where decision logic lives.
    """

    def __init__(self, goals: List):
        self.goals = goals

    def evaluate(self, state: dict) -> AgentDecision:
        results: List[GoalResult] = []
        for goal in self.goals:
            results.append(goal.evaluate(state))

        return self._synthesize_decision(results)

    def _synthesize_decision(self, results: List[GoalResult]) -> AgentDecision:
        goal_evaluations = self._serialize_goal_evaluations(results)
        unmet = [r for r in results if not r.met]

        # If nothing failed, we simply observe (no action needed)
        if not unmet:
            return AgentDecision(
                stance="observe",
                severity="low",
                reasons=["All goals satisfied"],
                goal_evaluations=goal_evaluations,
                recommendation_hierarchy=self._build_recommendation_hierarchy(
                    results=results,
                    stance="observe",
                    severity="low",
                    maturity_failed=False,
                ),
            )

        # Governance constraint: if maturity failed, we do NOT escalate or recommend aggressive action
        maturity_failed = any(
            r.goal_name == "campaign_maturity" and not r.met
            for r in results
        )
        if maturity_failed:
            return AgentDecision(
                stance="observe",
                severity="medium",
                reasons=self._build_reasons(results, maturity_failed=True),
                goal_evaluations=goal_evaluations,
                recommendation_hierarchy=self._build_recommendation_hierarchy(
                    results=results,
                    stance="observe",
                    severity="medium",
                    maturity_failed=True,
                ),
            )

        severities = [r.severity for r in unmet]
        high = severities.count("high")
        medium = severities.count("medium")

        if high >= 1:
            return AgentDecision(
                stance="escalate",
                severity="high",
                reasons=self._build_reasons(results, maturity_failed=False),
                goal_evaluations=goal_evaluations,
                recommendation_hierarchy=self._build_recommendation_hierarchy(
                    results=results,
                    stance="escalate",
                    severity="high",
                    maturity_failed=False,
                ),
            )

        # If multiple medium issues, escalate; otherwise recommend
        if medium >= 2:
            return AgentDecision(
                stance="escalate",
                severity="medium",
                reasons=self._build_reasons(results, maturity_failed=False),
                goal_evaluations=goal_evaluations,
                recommendation_hierarchy=self._build_recommendation_hierarchy(
                    results=results,
                    stance="escalate",
                    severity="medium",
                    maturity_failed=False,
                ),
            )

        return AgentDecision(
            stance="recommend",
            severity="medium",
            reasons=self._build_reasons(results, maturity_failed=False),
            goal_evaluations=goal_evaluations,
            recommendation_hierarchy=self._build_recommendation_hierarchy(
                results=results,
                stance="recommend",
                severity="medium",
                maturity_failed=False,
            ),
        )

    def _build_reasons(self, results: List[GoalResult], maturity_failed: bool) -> List[str]:
        reasons: List[str] = []
        for result in results:
            if result.met:
                continue
            label = f"[{result.goal_name}] {result.message}"
            if maturity_failed and result.goal_name != "campaign_maturity":
                label = f"{label} (supporting context; maturity gate keeps stance at observe)"
            reasons.append(label)
        return reasons

    def _serialize_goal_evaluations(self, results: List[GoalResult]) -> List[Dict[str, Any]]:
        return [
            {
                "priority": idx,
                "goal_name": result.goal_name,
                "met": result.met,
                "severity": result.severity,
                "message": result.message,
            }
            for idx, result in enumerate(results, start=1)
        ]

    def _build_recommendation_hierarchy(
        self,
        *,
        results: List[GoalResult],
        stance: str,
        severity: str,
        maturity_failed: bool,
    ) -> List[Dict[str, Any]]:
        hierarchy: List[Dict[str, Any]] = []

        for result in results:
            if result.met:
                continue

            role = "primary_driver" if not hierarchy else "supporting_driver"
            if maturity_failed and result.goal_name == "campaign_maturity":
                role = "blocking_constraint"
            elif maturity_failed:
                role = "deferred_context"

            hierarchy.append(
                {
                    "priority": len(hierarchy) + 1,
                    "goal_name": result.goal_name,
                    "severity": result.severity,
                    "message": result.message,
                    "reason": f"[{result.goal_name}] {result.message}",
                    "role": role,
                    "stance": stance,
                    "decision_severity": severity,
                }
            )

        if hierarchy:
            return hierarchy

        return [
            {
                "priority": 1,
                "goal_name": "all_goals",
                "severity": "low",
                "message": "All goals satisfied",
                "reason": "All goals satisfied",
                "role": "no_action",
                "stance": stance,
                "decision_severity": severity,
            }
        ]
