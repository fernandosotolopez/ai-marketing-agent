from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from registry.tool_registry import register_tool

from tools.metric_definitions import (
    CANONICAL_CAMPAIGN_FIELDS,
    SEMANTICS_CSV_TREND_UNKNOWN_WINDOW,
    SOURCE_CSV_SUPPLIED,
    SOURCE_MISSING,
    set_field_provenance,
)


# Canonical fields your agent/goals expect (single source: metric_definitions)
CANONICAL_FIELDS = CANONICAL_CAMPAIGN_FIELDS


# Aliases to handle messy business CSV column names
DEFAULT_COLUMN_ALIASES: Dict[str, List[str]] = {
    "campaign_id": ["campaign_id", "campaign", "campaignid", "id", "camp_id", "name"],
    "CPA": ["cpa", "cost_per_acquisition", "cost_per_conversion", "cac"],
    "ROAS": ["roas", "return_on_ad_spend"],
    "target_CPA": ["target_cpa", "cpa_target", "target_cac", "target_cost_per_acquisition"],
    "CPA_trend_7d": ["cpa_trend_7d", "cpa_7d_trend", "cpa_trend", "trend_cpa_7d"],
    "ROAS_trend_7d": ["roas_trend_7d", "roas_7d_trend", "roas_trend", "trend_roas_7d"],
    "days_active": ["days_active", "days_live", "age_days", "days_running", "days", "duration_days"],
}


@dataclass
class LoadResult:
    rows: List[Dict[str, Any]]
    warnings: List[str]

@register_tool(tags=["data", "io", "csv"])
def load_campaign_csv(
    csv_path: str,
    column_aliases: Optional[Dict[str, List[str]]] = None,
) -> LoadResult:
    """
    Load a campaign CSV and normalize rows into the canonical fields.

    Returns:
      LoadResult(rows=[...], warnings=[...])
    """
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    aliases = column_aliases or DEFAULT_COLUMN_ALIASES

    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError("CSV has no header row (fieldnames missing).")

        header_map = _build_header_map(reader.fieldnames)
        col_lookup = _resolve_columns(header_map, aliases)

        rows: List[Dict[str, Any]] = []
        warnings: List[str] = []

        for i, raw_row in enumerate(reader, start=2):  # start=2 because header is row 1
            norm_row, row_warnings = _normalize_row(raw_row, col_lookup, header_map, row_number=i)
            rows.append(norm_row)
            warnings.extend(row_warnings)

    return LoadResult(rows=rows, warnings=warnings)


# -----------------------
# Internals
# -----------------------

def _norm(s: str) -> str:
    return (s or "").strip().lower().replace(" ", "").replace("-", "_")


def _build_header_map(fieldnames: List[str]) -> Dict[str, str]:
    """
    Maps normalized header -> original header.
    """
    out: Dict[str, str] = {}
    for h in fieldnames:
        out[_norm(h)] = h
    return out


def _resolve_columns(
    header_map: Dict[str, str],
    aliases: Dict[str, List[str]],
) -> Dict[str, Optional[str]]:
    """
    For each canonical field, finds the actual CSV column name (original header)
    that matches one of the aliases. If not found, returns None.
    """
    resolved: Dict[str, Optional[str]] = {}
    for canonical, options in aliases.items():
        found: Optional[str] = None
        for opt in options:
            key = _norm(opt)
            if key in header_map:
                found = header_map[key]
                break
        resolved[canonical] = found
    return resolved


def _normalize_row(
    raw_row: Dict[str, Any],
    col_lookup: Dict[str, Optional[str]],
    header_map: Dict[str, str],
    row_number: int,
) -> Tuple[Dict[str, Any], List[str]]:
    """
    Convert a raw CSV row into the canonical shape expected by the agent.
    """
    warnings: List[str] = []

    def get(canonical: str) -> Any:
        col = col_lookup.get(canonical)
        if not col:
            return None
        return raw_row.get(col)

    # Build canonical row
    out: Dict[str, Any] = {k: None for k in CANONICAL_FIELDS}

    # campaign_id (string)
    campaign_id = get("campaign_id")
    if campaign_id is None or str(campaign_id).strip() == "":
        warnings.append(f"[row {row_number}] Missing campaign_id (will be None)")
        out["campaign_id"] = None
    else:
        out["campaign_id"] = str(campaign_id).strip()

    # CPA, ROAS, target_CPA (floats)
    out["CPA"] = _coerce_float(get("CPA"), warn=warnings, row=row_number, field="CPA")
    out["ROAS"] = _coerce_float(get("ROAS"), warn=warnings, row=row_number, field="ROAS")
    out["target_CPA"] = _coerce_float(get("target_CPA"), warn=warnings, row=row_number, field="target_CPA")

    # Trends: accept "12%", "+12%", "-0.12", "0.12" -> store as fraction (0.12)
    cpa_trend_raw = get("CPA_trend_7d")
    roas_trend_raw = get("ROAS_trend_7d")
    out["CPA_trend_7d"], cpa_trend_kind = _coerce_fraction(
        cpa_trend_raw, warn=warnings, row=row_number, field="CPA_trend_7d", col_resolved=col_lookup.get("CPA_trend_7d")
    )
    out["ROAS_trend_7d"], roas_trend_kind = _coerce_fraction(
        roas_trend_raw, warn=warnings, row=row_number, field="ROAS_trend_7d", col_resolved=col_lookup.get("ROAS_trend_7d")
    )
    _attach_trend_provenance_loader(
        out, "CPA_trend_7d", out["CPA_trend_7d"], cpa_trend_kind, col_lookup.get("CPA_trend_7d")
    )
    _attach_trend_provenance_loader(
        out, "ROAS_trend_7d", out["ROAS_trend_7d"], roas_trend_kind, col_lookup.get("ROAS_trend_7d")
    )

    # days_active (int)
    out["days_active"] = _coerce_int(get("days_active"), warn=warnings, row=row_number, field="days_active")

    # Scalar metrics: provenance for trust layer (CSV vs missing)
    _attach_scalar_provenance(
        out,
        "CPA",
        out["CPA"],
        col_lookup.get("CPA"),
    )
    _attach_scalar_provenance(
        out,
        "ROAS",
        out["ROAS"],
        col_lookup.get("ROAS"),
    )
    _attach_scalar_provenance(
        out,
        "target_CPA",
        out["target_CPA"],
        col_lookup.get("target_CPA"),
    )
    _attach_scalar_provenance(
        out,
        "days_active",
        out["days_active"],
        col_lookup.get("days_active"),
    )
    if out["campaign_id"]:
        set_field_provenance(
            out,
            "campaign_id",
            SOURCE_CSV_SUPPLIED,
            detail="Present in CSV",
            semantics="",
        )
    else:
        set_field_provenance(out, "campaign_id", SOURCE_MISSING, detail="Missing or empty in CSV", semantics="")

    # Optional: keep extra raw columns for debugging
    # (won't be used by goals, but helps when CSV is messy)
    out["_raw"] = raw_row

    return out, warnings


def _attach_scalar_provenance(
    row: Dict[str, Any],
    field_name: str,
    value: Any,
    csv_column: Optional[str],
) -> None:
    if csv_column is None:
        set_field_provenance(
            row,
            field_name,
            SOURCE_MISSING,
            detail="No CSV column mapped for this field",
            semantics="",
        )
        return
    if value is None:
        set_field_provenance(
            row,
            field_name,
            SOURCE_MISSING,
            detail="Empty or unparseable cell",
            semantics="",
        )
        return
    set_field_provenance(
        row,
        field_name,
        SOURCE_CSV_SUPPLIED,
        detail=f"From column {csv_column!r}",
        semantics="",
    )


def _attach_trend_provenance_loader(
    row: Dict[str, Any],
    field_name: str,
    value: Optional[float],
    kind: Optional[str],
    csv_column: Optional[str],
) -> None:
    """kind: 'percent_suffix' | 'bare_numeric' | None if missing/unparsed."""
    if csv_column is None:
        set_field_provenance(
            row,
            field_name,
            SOURCE_MISSING,
            detail="No CSV column mapped for this trend field",
            semantics=SEMANTICS_CSV_TREND_UNKNOWN_WINDOW,
        )
        return
    if value is None:
        set_field_provenance(
            row,
            field_name,
            SOURCE_MISSING,
            detail="Empty, missing, or unparseable cell",
            semantics=SEMANTICS_CSV_TREND_UNKNOWN_WINDOW,
        )
        return
    detail = (
        "Parsed with % suffix (interpreted as fraction)"
        if kind == "percent_suffix"
        else "Parsed as bare number (fraction; large values may be rescaled in metrics step)"
    )
    set_field_provenance(
        row,
        field_name,
        SOURCE_CSV_SUPPLIED,
        detail=detail,
        semantics=SEMANTICS_CSV_TREND_UNKNOWN_WINDOW,
    )


def _coerce_float(value: Any, warn: List[str], row: int, field: str) -> Optional[float]:
    if value is None:
        warn.append(f"[row {row}] Missing {field}")
        return None

    s = str(value).strip()
    if s == "":
        warn.append(f"[row {row}] Empty {field}")
        return None

    # handle commas like "1,234.56" or "1234,56" (best-effort)
    s = s.replace(",", "")
    try:
        return float(s)
    except ValueError:
        warn.append(f"[row {row}] Could not parse {field}='{value}' as float")
        return None


def _coerce_int(value: Any, warn: List[str], row: int, field: str) -> Optional[int]:
    if value is None:
        warn.append(f"[row {row}] Missing {field}")
        return None

    s = str(value).strip()
    if s == "":
        warn.append(f"[row {row}] Empty {field}")
        return None

    try:
        return int(float(s))  # allows "42.0"
    except ValueError:
        warn.append(f"[row {row}] Could not parse {field}='{value}' as int")
        return None


def _coerce_fraction(
    value: Any,
    warn: List[str],
    row: int,
    field: str,
    col_resolved: Optional[str] = None,
) -> Tuple[Optional[float], Optional[str]]:
    """
    Accepts:
      "12%"  -> 0.12
      "+12%" -> 0.12
      "-5%"  -> -0.05
      "0.12" -> 0.12
      "-0.05"-> -0.05
      "12"   -> 12.0  (ambiguous vs percent; metrics may rescale)

    Returns (value, kind) where kind is 'percent_suffix', 'bare_numeric', or None if missing.
    If col_resolved is None, the column was not mapped — caller still gets (None, None).
    """
    if col_resolved is None:
        # Column not present in CSV header mapping — do not treat as "missing cell".
        return None, None

    if value is None:
        warn.append(f"[row {row}] Missing {field}")
        return None, None

    s = str(value).strip()
    if s == "":
        warn.append(f"[row {row}] Empty {field}")
        return None, None

    try:
        if s.endswith("%"):
            s2 = s[:-1].strip()
            return float(s2) / 100.0, "percent_suffix"
        return float(s), "bare_numeric"
    except ValueError:
        warn.append(f"[row {row}] Could not parse {field}='{value}' as fraction")
        return None, None
