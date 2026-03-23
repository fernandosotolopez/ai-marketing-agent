"""
Canonical metric semantics and provenance labels for the marketing agent pipeline.

This module is the single source of truth for how we describe where a value came from
and what "7d trend" means in different contexts. Downstream code (goals, DB, dashboard)
can read row["_metric_provenance"] without re-deriving trust from field names alone.

Note: Field names CPA_trend_7d / ROAS_trend_7d are historical; the numeric meaning is
relative change as a fraction (0.12 = +12%), but the *time basis* depends on provenance.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Canonical row keys (same as historical CANONICAL_FIELDS)
# ---------------------------------------------------------------------------

CANONICAL_CAMPAIGN_FIELDS: List[str] = [
    "campaign_id",
    "CPA",
    "ROAS",
    "target_CPA",
    "CPA_trend_7d",
    "ROAS_trend_7d",
    "days_active",
]

TREND_FIELDS: tuple[str, ...] = ("CPA_trend_7d", "ROAS_trend_7d")

# Where the row stores structured provenance (opaque to goals until they opt in).
ROW_PROVENANCE_KEY = "_metric_provenance"

# ---------------------------------------------------------------------------
# Provenance sources (finite vocabulary)
# ---------------------------------------------------------------------------

# Value came from the normalized CSV column for this field.
SOURCE_CSV_SUPPLIED = "csv_supplied"

# No value after load (missing column or empty cell).
SOURCE_MISSING = "missing"

# metrics.py rescale heuristic: bare number looked like percent (e.g. 12 -> 0.12).
SOURCE_METRICS_PERCENT_RESCALED = "metrics_percent_rescaled"

# Filled by comparing two windows of in-memory snapshots (not calendar days).
SOURCE_MEMORY_SNAPSHOT_DERIVED = "memory_snapshot_derived"

# CPA/ROAS computed from raw spend/conversions/revenue in metrics.py.
SOURCE_COMPUTED_FROM_RAW = "computed_from_raw"

# Row was not produced by data_loader (tests/manual dict); provenance unknown.
SOURCE_LEGACY_OR_UNKNOWN = "legacy_or_unknown"

# target_CPA filled from MetricsConfig.default_target_cpa in metrics.py.
SOURCE_METRICS_DEFAULT_TARGET_CPA = "metrics_default_target_cpa"

# ---------------------------------------------------------------------------
# Window / semantics hints (human-facing and for future UI)
# ---------------------------------------------------------------------------

SEMANTICS_CSV_TREND_UNKNOWN_WINDOW = (
    "CSV supplied trend; window is defined by the data provider export, not enforced here."
)

SEMANTICS_MEMORY_TWO_WINDOWS_OF_SNAPSHOTS = (
    "Derived from the last 2×N stored snapshots per campaign (N=trend_window_days); "
    "not necessarily seven calendar days."
)


@dataclass
class FieldProvenance:
    """Provenance for one canonical field."""

    source: str
    detail: str = ""
    semantics: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source,
            "detail": self.detail,
            "semantics": self.semantics,
        }


def _new_provenance_container() -> Dict[str, Any]:
    return {"version": 1, "fields": {}, "ingestion_notes": []}


def get_or_create_provenance(row: Dict[str, Any]) -> Dict[str, Any]:
    """Return row's provenance dict, creating a minimal one if absent."""
    existing = row.get(ROW_PROVENANCE_KEY)
    if isinstance(existing, dict) and "fields" in existing:
        if "ingestion_notes" not in existing:
            existing["ingestion_notes"] = []
        if "version" not in existing:
            existing["version"] = 1
        return existing
    row[ROW_PROVENANCE_KEY] = _new_provenance_container()
    return row[ROW_PROVENANCE_KEY]


def set_field_provenance(
    row: Dict[str, Any],
    field_name: str,
    source: str,
    *,
    detail: str = "",
    semantics: str = "",
) -> None:
    prov = get_or_create_provenance(row)
    fields = prov.setdefault("fields", {})
    fields[field_name] = FieldProvenance(
        source=source, detail=detail, semantics=semantics
    ).to_dict()


def get_field_provenance(row: Dict[str, Any], field_name: str) -> Optional[Dict[str, Any]]:
    prov = row.get(ROW_PROVENANCE_KEY)
    if not isinstance(prov, dict):
        return None
    fields = prov.get("fields")
    if not isinstance(fields, dict):
        return None
    out = fields.get(field_name)
    return out if isinstance(out, dict) else None


def add_ingestion_note(row: Dict[str, Any], note: str) -> None:
    prov = get_or_create_provenance(row)
    notes = prov.setdefault("ingestion_notes", [])
    if isinstance(notes, list) and note and note not in notes:
        notes.append(note)
