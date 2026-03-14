from typing import List, Dict
from agent.goals import GoalResult


class AgentDecision:
    """
    Final decision produced by the agent after evaluating all goals.
    """

    def __init__(self, stance: str, severity: str, reasons: List[str]):
        # stance: "observe" | "recommend" | "escalate"
        self.stance = stance
        self.severity = severity  # "low" | "medium" | "high"
        self.reasons = reasons

    def to_dict(self) -> Dict:
        return {
            "stance": self.stance,
            "severity": self.severity,
            "reasons": self.reasons
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
        # Collect unmet goal messages
        unmet = [r for r in results if not r.met]
        reasons = [f"[{r.goal_name}] {r.message}" for r in unmet]

        # If nothing failed, we simply observe (no action needed)
        if not unmet:
            return AgentDecision(
                stance="observe",
                severity="low",
                reasons=["All goals satisfied"]
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
                reasons=reasons
            )

        severities = [r.severity for r in unmet]
        high = severities.count("high")
        medium = severities.count("medium")

        if high >= 1:
            return AgentDecision(
                stance="escalate",
                severity="high",
                reasons=reasons
            )

        # If multiple medium issues, escalate; otherwise recommend
        if medium >= 2:
            return AgentDecision(
                stance="escalate",
                severity="medium",
                reasons=reasons
            )

        return AgentDecision(
            stance="recommend",
            severity="medium",
            reasons=reasons
        )
