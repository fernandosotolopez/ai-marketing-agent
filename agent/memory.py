from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
import json
import os


@dataclass
class MemoryRecord:
    """
    One historical record: state snapshot + decision taken at that time.
    Stored in a serializable shape (dicts), so it can be saved to disk.
    """
    timestamp_utc: str
    campaign_id: str
    state: Dict[str, Any]
    decision: Dict[str, Any]


class MemoryStore:
    """
    Simple in-memory storage for past campaign states and decisions.

    Design:
    - Memory is a log of immutable records.
    - The agent/loop can write to memory each cycle.
    - Tools/state-builders can read memory to compute trends.
    """

    def __init__(self) -> None:
        self._records: List[MemoryRecord] = []

    # ---------- Write ----------

    def add(self, campaign_id: str, state: Any, decision: Any) -> None:
        """
        Add a new record.
        `state` can be a dict or an object with `to_dict()`.
        `decision` can be a dict or an object with `to_dict()`.
        """
        state_dict = self._to_dict(state)
        decision_dict = self._to_dict(decision)

        # If campaign_id wasn't provided correctly, try to infer it from state
        inferred_campaign_id = state_dict.get("campaign_id")
        cid = campaign_id or inferred_campaign_id
        if not cid:
            raise ValueError("campaign_id missing: provide campaign_id or include campaign_id in state")

        ts = datetime.now(timezone.utc).isoformat()

        self._records.append(
            MemoryRecord(
                timestamp_utc=ts,
                campaign_id=cid,
                state=state_dict,
                decision=decision_dict,
            )
        )

    # ---------- Read ----------

    def all_records(self) -> List[MemoryRecord]:
        return list(self._records)

    def get_last_n_states(self, campaign_id: str, n: int) -> List[Dict[str, Any]]:
        """
        Returns the last N state snapshots for a campaign (most recent last).
        """
        filtered = [r for r in self._records if r.campaign_id == campaign_id]
        return [r.state for r in filtered[-n:]]

    def get_last_decision(self, campaign_id: str) -> Optional[Dict[str, Any]]:
        """
        Returns the most recent decision dict for a campaign, or None if not found.
        """
        for r in reversed(self._records):
            if r.campaign_id == campaign_id:
                return r.decision
        return None

    def get_last_record(self, campaign_id: str) -> Optional[MemoryRecord]:
        """
        Returns the most recent full record for a campaign, or None if not found.
        """
        for r in reversed(self._records):
            if r.campaign_id == campaign_id:
                return r
        return None

    # ---------- Persistence (optional but very useful) ----------

    def save_json(self, filepath: str) -> None:
        """
        Save all memory records to a JSON file.
        """
        os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
        payload = [asdict(r) for r in self._records]
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    def load_json(self, filepath: str) -> None:
        """
        Load memory records from a JSON file (replaces current memory).
        """
        with open(filepath, "r", encoding="utf-8") as f:
            payload = json.load(f)

        records: List[MemoryRecord] = []
        for item in payload:
            records.append(
                MemoryRecord(
                    timestamp_utc=item["timestamp_utc"],
                    campaign_id=item["campaign_id"],
                    state=item["state"],
                    decision=item["decision"],
                )
            )
        self._records = records

    # ---------- Helpers ----------

    def _to_dict(self, obj: Any) -> Dict[str, Any]:
        if obj is None:
            return {}
        if isinstance(obj, dict):
            return obj
        to_dict = getattr(obj, "to_dict", None)
        if callable(to_dict):
            return to_dict()
        raise TypeError(f"Object of type {type(obj).__name__} is not dict and has no to_dict()")
