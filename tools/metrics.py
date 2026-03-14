from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, TYPE_CHECKING
from registry.tool_registry import register_tool

if TYPE_CHECKING:
    from agent.memory import MemoryStore


CANONICAL_FIELDS = [
    "campaign_id",
    "CPA",
    "ROAS",
    "target_CPA",
    "CPA_trend_7d",
    "ROAS_trend_7d",
    "days_active",
]


@dataclass
class MetricsConfig:
    """
    Configuration for validation and optional metric/trend computation.
    """
    default_target_cpa: Optional[float] = None
    trend_window_days: int = 7

    # If CPA or ROAS missing, try to compute from raw columns (if available)
    allow_compute_from_raw: bool = True

    # If trend missing, try to compute from memory (if provided)
    allow_compute_trends_from_memory: bool = True


@dataclass
class MetricsResult:
    row: Dict[str, Any]
    warnings: List[str]
    errors: List[str]


# ---- Raw-column aliases (only used if allow_compute_from_raw=True) ----
RAW_ALIASES = {
    "spend": ["spend", "cost", "amount_spent", "ad_spend", "total_spend"],
    "conversions": ["conversions", "purchases", "orders", "leads", "results"],
    "revenue": ["revenue", "sales", "value", "conversion_value", "total_revenue"],
}

@register_tool(tags=["data", "metrics", "validation"])
def validate_and_enrich_row(
    row: Dict[str, Any],
    config: Optional[MetricsConfig] = None,
    memory: Optional["MemoryStore"] = None,
) -> MetricsResult:
    """
    Validates and enriches a normalized row.

    Input expectation:
      row contains canonical keys (from tools/data_loader.py),
      and may include row["_raw"] with original CSV columns.

    Output:
      A canonical-ready row (same shape), plus warnings/errors.
    """
    cfg = config or MetricsConfig()
    warnings: List[str] = []
    errors: List[str] = []

    # Ensure canonical keys exist
    for k in CANONICAL_FIELDS:
        if k not in row:
            errors.append(f"Missing canonical key '{k}' in row (data_loader output mismatch).")

    if errors:
        return MetricsResult(row=row, warnings=warnings, errors=errors)

    # campaign_id
    if not row.get("campaign_id"):
        errors.append("campaign_id is missing or empty.")

    # days_active
    days_active = row.get("days_active")
    if days_active is None:
        warnings.append("days_active missing; MaturityGoal will fail.")
    else:
        try:
            row["days_active"] = int(days_active)
        except Exception:
            warnings.append(f"days_active not an int: {days_active!r}")
            row["days_active"] = None

    # target_CPA
    if row.get("target_CPA") is None and cfg.default_target_cpa is not None:
        row["target_CPA"] = cfg.default_target_cpa
        warnings.append(f"target_CPA missing; using default_target_cpa={cfg.default_target_cpa}")

    # CPA / ROAS may be missing — compute if possible
    if cfg.allow_compute_from_raw:
        _maybe_compute_cpa_roas_from_raw(row, warnings)

    # Validate CPA/ROAS presence
    if row.get("CPA") is None:
        warnings.append("CPA missing; CostEfficiencyGoal will fail.")
    else:
        row["CPA"] = _to_float(row["CPA"], "CPA", warnings)

    if row.get("ROAS") is None:
        warnings.append("ROAS missing (not required by goals yet, but useful context).")
    else:
        row["ROAS"] = _to_float(row["ROAS"], "ROAS", warnings)

    # Validate target_CPA for later division in goals
    if row.get("target_CPA") is None:
        warnings.append("target_CPA missing; CostEfficiencyGoal will fail.")
    else:
        row["target_CPA"] = _to_float(row["target_CPA"], "target_CPA", warnings)
        if row["target_CPA"] is not None and row["target_CPA"] <= 0:
            warnings.append("target_CPA <= 0; would cause invalid CPA comparison/division.")

    # Trends: ensure they are FRACTIONS (0.12 means 12%)
    row["CPA_trend_7d"] = _normalize_trend_fraction(row.get("CPA_trend_7d"), "CPA_trend_7d", warnings)
    row["ROAS_trend_7d"] = _normalize_trend_fraction(row.get("ROAS_trend_7d"), "ROAS_trend_7d", warnings)

    # If trends are missing, compute from memory if allowed and available
    if cfg.allow_compute_trends_from_memory and memory is not None:
        _maybe_fill_trends_from_memory(row, memory, cfg.trend_window_days, warnings)

    # After memory attempt, warn if still missing (because PerformanceTrendGoal needs them)
    if row.get("CPA_trend_7d") is None or row.get("ROAS_trend_7d") is None:
        warnings.append("One or both trends missing; PerformanceTrendGoal may fail.")

    return MetricsResult(row=row, warnings=warnings, errors=errors)


# -------------------------
# Internals
# -------------------------

def _to_float(value: Any, field: str, warnings: List[str]) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        warnings.append(f"Could not coerce {field} to float: {value!r}")
        return None


def _normalize_trend_fraction(value: Any, field: str, warnings: List[str]) -> Optional[float]:
    """
    Ensures trend is a fraction:
      0.12 = 12%
     -0.05 = -5%

    If we detect a likely percent value like 12 or -5, we convert to 0.12 / -0.05.
    """
    if value is None:
        return None

    try:
        v = float(value)
    except Exception:
        warnings.append(f"Could not parse {field} as float: {value!r}")
        return None

    # Heuristic: if abs(v) > 1.5, it's likely a percent like 12 (meaning 12%)
    if abs(v) > 1.5 and abs(v) <= 100:
        warnings.append(f"{field} looked like percent ({v}); converting to fraction ({v/100.0}).")
        v = v / 100.0

    return v


def _get_raw_number(raw: Dict[str, Any], aliases: List[str]) -> Optional[float]:
    if not raw:
        return None

    for key in raw.keys():
        k_norm = str(key).strip().lower().replace(" ", "").replace("-", "_")
        for alias in aliases:
            a_norm = str(alias).strip().lower().replace(" ", "").replace("-", "_")
            if k_norm == a_norm:
                try:
                    return float(str(raw[key]).strip().replace(",", ""))
                except Exception:
                    return None
    return None


def _maybe_compute_cpa_roas_from_raw(row: Dict[str, Any], warnings: List[str]) -> None:
    """
    If CPA or ROAS missing, try to compute from raw spend/conversions/revenue.
    CPA = spend / conversions
    ROAS = revenue / spend
    """
    raw = row.get("_raw") or {}

    spend = _get_raw_number(raw, RAW_ALIASES["spend"])
    conv = _get_raw_number(raw, RAW_ALIASES["conversions"])
    rev = _get_raw_number(raw, RAW_ALIASES["revenue"])

    if row.get("CPA") is None and spend is not None and conv is not None:
        if conv > 0:
            row["CPA"] = spend / conv
            warnings.append("Computed CPA from raw spend/conversions.")
        else:
            warnings.append("Could not compute CPA: conversions <= 0.")

    if row.get("ROAS") is None and spend is not None and rev is not None:
        if spend > 0:
            row["ROAS"] = rev / spend
            warnings.append("Computed ROAS from raw revenue/spend.")
        else:
            warnings.append("Could not compute ROAS: spend <= 0.")


def _mean(values: List[float]) -> Optional[float]:
    if not values:
        return None
    return sum(values) / len(values)


def _maybe_fill_trends_from_memory(
    row: Dict[str, Any],
    memory: "MemoryStore",
    window: int,
    warnings: List[str],
) -> None:
    """
    Fill missing trends using memory snapshots.

    Trend = (mean(last window) - mean(previous window)) / mean(previous window)

    Requires at least 2*window historical states in memory for that campaign.
    """
    cid = row.get("campaign_id")
    if not cid:
        return

    need_cpa_trend = row.get("CPA_trend_7d") is None
    need_roas_trend = row.get("ROAS_trend_7d") is None
    if not (need_cpa_trend or need_roas_trend):
        return

    states = memory.get_last_n_states(cid, n=2 * window)
    if len(states) < 2 * window:
        warnings.append(f"Not enough memory to compute trends (need {2*window} states, have {len(states)}).")
        return

    prev = states[:window]
    last = states[window:]

    def extract(metric: str, bucket: List[Dict[str, Any]]) -> List[float]:
        out: List[float] = []
        for s in bucket:
            v = s.get(metric)
            if v is None:
                continue
            try:
                out.append(float(v))
            except Exception:
                continue
        return out

    if need_cpa_trend:
        prev_mean = _mean(extract("CPA", prev))
        last_mean = _mean(extract("CPA", last))
        if prev_mean and prev_mean != 0 and last_mean is not None:
            row["CPA_trend_7d"] = (last_mean - prev_mean) / prev_mean
            warnings.append("Computed CPA_trend_7d from memory.")
        else:
            warnings.append("Could not compute CPA_trend_7d from memory (missing/zero baselines).")

    if need_roas_trend:
        prev_mean = _mean(extract("ROAS", prev))
        last_mean = _mean(extract("ROAS", last))
        if prev_mean and prev_mean != 0 and last_mean is not None:
            row["ROAS_trend_7d"] = (last_mean - prev_mean) / prev_mean
            warnings.append("Computed ROAS_trend_7d from memory.")
        else:
            warnings.append("Could not compute ROAS_trend_7d from memory (missing/zero baselines).")
