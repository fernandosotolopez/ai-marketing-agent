from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from registry.tool_registry import register_tool


@dataclass
class SimulationConfig:
    """
    Controls the 'shape' of the what-if model.

    Elasticity interpretation (simple power model):
      projected_metric = baseline_metric * (budget_multiplier ** elasticity)

    Typical assumptions:
      - CPA elasticity > 0  (more budget tends to increase CPA)
      - ROAS elasticity < 0 (more budget tends to decrease ROAS)

    We'll implement ROAS with a negative elasticity default.
    """
    cpa_elasticity: float = 0.15
    roas_elasticity: float = -0.10

    # Safety: how far we allow simulated changes to go
    max_budget_multiplier: float = 1.50   # +50%
    min_budget_multiplier: float = 0.50   # -50%


@dataclass
class ScenarioResult:
    scenario_name: str
    budget_multiplier: float
    projected_CPA: Optional[float]
    projected_ROAS: Optional[float]
    notes: List[str]

@register_tool(tags=["simulation", "what_if"])
def run_budget_scenarios(
    row: Dict[str, Any],
    multipliers: Optional[List[float]] = None,
    config: Optional[SimulationConfig] = None,
) -> List[ScenarioResult]:
    """
    Run a set of budget what-if scenarios for a single campaign.

    row should be enriched (ideally from tools/metrics.py),
    containing CPA and ROAS as floats.

    multipliers: budget multipliers to simulate (e.g. [0.8, 1.0, 1.2])
    """
    cfg = config or SimulationConfig()
    if multipliers is None:
        multipliers = [0.80, 0.90, 1.00, 1.10, 1.20]

    baseline_cpa = row.get("CPA")
    baseline_roas = row.get("ROAS")

    results: List[ScenarioResult] = []

    for m in multipliers:
        notes: List[str] = []

        if m < cfg.min_budget_multiplier or m > cfg.max_budget_multiplier:
            notes.append(
                f"Multiplier {m:.2f} outside safe range "
                f"[{cfg.min_budget_multiplier:.2f}, {cfg.max_budget_multiplier:.2f}]"
            )

        projected_cpa = _project_metric(
            baseline=baseline_cpa,
            multiplier=m,
            elasticity=cfg.cpa_elasticity,
            metric_name="CPA",
            notes=notes,
        )

        projected_roas = _project_metric(
            baseline=baseline_roas,
            multiplier=m,
            elasticity=cfg.roas_elasticity,
            metric_name="ROAS",
            notes=notes,
        )

        name = _scenario_name(m)

        results.append(
            ScenarioResult(
                scenario_name=name,
                budget_multiplier=m,
                projected_CPA=projected_cpa,
                projected_ROAS=projected_roas,
                notes=notes,
            )
        )

    return results

@register_tool(tags=["simulation"])
def choose_default_multipliers(decision: Dict[str, Any]) -> List[float]:
    """
    Simple policy: if escalating, include more aggressive downside scenarios.
    If observing, keep it conservative.

    This isn't "the agent" deciding — it's just a helper for demo/UI.
    """
    stance = decision.get("stance")
    if stance == "escalate":
        return [0.60, 0.75, 0.90, 1.00, 1.10]
    if stance == "recommend":
        return [0.80, 0.90, 1.00, 1.10, 1.20]
    return [0.90, 1.00, 1.10]


# -----------------------
# Internals
# -----------------------

def _scenario_name(multiplier: float) -> str:
    pct = (multiplier - 1.0) * 100.0
    if abs(pct) < 0.001:
        return "baseline"
    sign = "+" if pct > 0 else ""
    return f"budget_{sign}{pct:.0f}%"


def _project_metric(
    baseline: Any,
    multiplier: float,
    elasticity: float,
    metric_name: str,
    notes: List[str],
) -> Optional[float]:
    """
    projected = baseline * (multiplier ** elasticity)
    """
    if baseline is None:
        notes.append(f"Missing baseline {metric_name}; cannot project.")
        return None

    try:
        b = float(baseline)
    except Exception:
        notes.append(f"Baseline {metric_name} not numeric: {baseline!r}")
        return None

    if b < 0:
        notes.append(f"Baseline {metric_name} is negative ({b}); projection may be meaningless.")

    try:
        return b * (multiplier ** elasticity)
    except Exception:
        notes.append(f"Failed projection for {metric_name} with multiplier={multiplier}, elasticity={elasticity}")
        return None
