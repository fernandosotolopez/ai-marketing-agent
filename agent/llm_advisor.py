from __future__ import annotations

import os
import time
from typing import Any, Dict, List, Optional, Literal

from pydantic import BaseModel, Field


class AdvisorOutput(BaseModel):
    advisor_summary: str = Field(..., description="Short summary of what to do.")
    advisor_actions: List[str] = Field(..., description="3-6 concrete next steps.")
    advisor_confidence: Literal["low", "medium", "high"]


class LLMAdvisor:
    """
    LLM-backed advisor layer (presentation / narrative), not decision authority.

    Uses OpenAI Structured Outputs via client.responses.parse(...) with a Pydantic schema.
    On any failure, invalid parse, empty content, or stance guard mismatch → deterministic fallback.

    Env vars:
      - OPENAI_API_KEY
      - OPENAI_MODEL (optional) default: gpt-4.1-mini
      - OPENAI_ADVISOR_TIMEOUT (optional) seconds, default 90
    """

    def __init__(self, model: Optional[str] = None, use_llm: bool = True) -> None:
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
        self.use_llm = use_llm
        self._timeout = float(os.getenv("OPENAI_ADVISOR_TIMEOUT", "90"))

    def advise(self, state: Dict[str, Any], decision: Dict[str, Any]) -> Dict[str, Any]:
        if not self.use_llm:
            return self._fallback(
                state=state,
                decision=decision,
                reason="llm_disabled",
            )

        if not os.getenv("OPENAI_API_KEY"):
            return self._fallback(
                state=state,
                decision=decision,
                reason="no_api_key",
            )

        slim_state = _slim_state_for_llm(state)

        for attempt in range(2):
            try:
                from openai import OpenAI

                client = OpenAI(timeout=self._timeout, max_retries=1)

                system_msg, user_msg = self._build_messages(
                    state=slim_state, decision=decision
                )

                resp = client.responses.parse(
                    model=self.model,
                    input=[
                        {"role": "system", "content": system_msg},
                        {"role": "user", "content": user_msg},
                    ],
                    text_format=AdvisorOutput,
                )

                parsed: Optional[AdvisorOutput] = getattr(resp, "output_parsed", None)
                if parsed is None:
                    if attempt == 0:
                        time.sleep(0.5)
                        continue
                    return self._fallback(
                        state=state,
                        decision=decision,
                        reason="empty_parse",
                    )

                out = {
                    "advisor_summary": (parsed.advisor_summary or "").strip(),
                    "advisor_actions": [
                        a.strip()
                        for a in (parsed.advisor_actions or [])
                        if a and str(a).strip()
                    ],
                    "advisor_confidence": _normalize_confidence(
                        parsed.advisor_confidence
                    ),
                    "advisor_reasons_seen": decision.get("reasons", []),
                    "advisor_used_llm": True,
                    "advisor_model": self.model,
                    "advisor_raw_text": None,
                    "advisor_source": "llm",
                    "advisor_fallback_reason": None,
                }
                return self._finalize_output(
                    state=state, decision=decision, out=out
                )

            except Exception:
                if attempt == 0:
                    time.sleep(0.8)
                    continue
                return self._fallback(
                    state=state,
                    decision=decision,
                    reason="api_failure",
                )

    def _build_messages(
        self, state: Dict[str, Any], decision: Dict[str, Any]
    ) -> tuple[str, str]:
        system_msg = (
            "You are a senior performance marketing expert. "
            "You are advising a human reviewer. The RULE-BASED DECISION (stance, severity, reasons) "
            "is already final — do not contradict it. "
            "Write practical next steps that align with that stance. "
            "Use plain business language only. "
            "Never mention APIs, models, errors, stack traces, or system internals."
        )

        user_msg = (
            "Given the campaign STATE (metrics only) and the rule-based DECISION, output:\n"
            "- advisor_summary: 1-2 sentences\n"
            "- advisor_actions: 3-6 short imperative steps (each under 200 characters)\n"
            "- advisor_confidence: low|medium|high\n\n"
            "Constraints:\n"
            "- If stance is 'observe', avoid aggressive actions (no full pause unless ROAS is far below 1.0 and the decision already implies risk).\n"
            "- If stance is 'recommend', prefer measured optimizations, not escalation language.\n"
            "- If stance is 'escalate', be direct about owner review and risk control.\n"
            "- If ROAS < 1.0, include at least one risk-control or spend-discipline action.\n\n"
            f"STATE: {state}\n"
            f"DECISION: {decision}\n"
        )
        return system_msg, user_msg

    def _fallback(
        self,
        state: Dict[str, Any],
        decision: Dict[str, Any],
        reason: str = "deterministic",
    ) -> Dict[str, Any]:
        stance = decision.get("stance", "observe")
        severity = decision.get("severity", "low")
        reasons: List[str] = decision.get("reasons", [])
        roas = _safe_float(state.get("ROAS"))
        cpa = _safe_float(state.get("CPA"))
        target_cpa = _safe_float(state.get("target_CPA"))
        days_active = _safe_int(state.get("days_active"))

        if stance == "observe":
            if days_active is not None and days_active < 14:
                summary = "The campaign is still in its learning window. Keep the setup stable while monitoring efficiency closely."
                actions = [
                    "Keep major changes minimal until the campaign has at least 14 days of delivery.",
                    "Monitor CPA and ROAS daily for signal stability.",
                    "Reassess the campaign once it exits the learning period.",
                ]
                if roas is not None and roas < 1.0:
                    actions.insert(
                        1,
                        "Keep spend disciplined while ROAS remains below 1.0 during the learning period.",
                    )
            else:
                summary = "Monitor performance for now and avoid aggressive edits until the signal becomes clearer."
                actions = [
                    "Track CPA and ROAS closely over the next reporting window.",
                    "Set an alert for sustained movement away from target efficiency.",
                    "Re-review once there is enough new data to support a stronger action.",
                ]
                if roas is not None and roas < 1.0:
                    actions.insert(0, "Keep spend disciplined while ROAS remains below 1.0.")
        elif stance == "recommend":
            if cpa is not None and target_cpa is not None and cpa > target_cpa:
                summary = "Performance is close enough to target for measured optimization rather than escalation."
            else:
                summary = "Measured optimizations are recommended to improve efficiency without disrupting volume."
            actions = [
                "Tighten bids or budget allocation on the weakest segments first.",
                "Refresh creative, audience, or offer targeting where efficiency has softened.",
                "Check landing-page conversion friction before making broader scale changes.",
            ]
        else:
            if roas is not None and roas < 1.0:
                summary = "Efficiency is materially off target and returns are below break-even. Escalate for immediate owner review."
            else:
                summary = "Efficiency is materially off target and the campaign needs prompt owner review."
            actions = [
                "Reduce spend on the weakest segments while the issue is investigated.",
                "Audit tracking and recent change history to confirm the signal.",
                "Review targeting, placements, search terms, and creative fatigue for rapid corrections.",
            ]

        if stance == "escalate" and roas is not None and roas < 1.0:
            actions.insert(0, "Set a temporary spend cap until ROAS recovers above 1.0.")

        actions = _cap_action_list(actions, max_items=6, max_chars=220)

        return {
            "advisor_summary": summary,
            "advisor_actions": actions,
            "advisor_reasons_seen": reasons,
            "advisor_confidence": "medium" if severity in ("medium", "high") else "high",
            "advisor_used_llm": False,
            "advisor_model": None,
            "advisor_raw_text": None,
            "advisor_source": "deterministic_fallback",
            "advisor_fallback_reason": reason,
        }

    def _finalize_output(
        self,
        state: Dict[str, Any],
        decision: Dict[str, Any],
        out: Dict[str, Any],
    ) -> Dict[str, Any]:
        summary = _clean_business_text(out.get("advisor_summary"))
        actions = _clean_action_list(out.get("advisor_actions", []))
        actions = _cap_action_list(actions, max_items=6, max_chars=220)

        if not summary or not actions:
            return self._fallback(
                state=state,
                decision=decision,
                reason="empty_llm_output",
            )

        if _output_conflicts_with_diagnosis(
            state=state, decision=decision, summary=summary, actions=actions
        ):
            return self._fallback(
                state=state,
                decision=decision,
                reason="stance_guard",
            )

        out["advisor_summary"] = summary
        out["advisor_actions"] = actions
        out["advisor_confidence"] = _normalize_confidence(out.get("advisor_confidence"))
        out["advisor_source"] = "llm"
        out["advisor_fallback_reason"] = None
        return out


def _slim_state_for_llm(state: Dict[str, Any]) -> Dict[str, Any]:
    """Avoid huge payloads (provenance, raw rows) that hurt reliability and latency."""
    keys = (
        "campaign_id",
        "CPA",
        "ROAS",
        "target_CPA",
        "CPA_trend_7d",
        "ROAS_trend_7d",
        "days_active",
    )
    return {k: state.get(k) for k in keys}


def _normalize_confidence(value: Any) -> str:
    v = _clean_text(value).lower()
    if v in {"low", "medium", "high"}:
        return v
    return "medium"


def _cap_action_list(actions: List[str], *, max_items: int, max_chars: int) -> List[str]:
    out: List[str] = []
    for a in actions:
        t = a.strip()
        if len(t) > max_chars:
            t = t[: max_chars - 1].rstrip() + "…"
        if t:
            out.append(t)
        if len(out) >= max_items:
            break
    return out


def _safe_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        return float(value)
    except Exception:
        return None


def _safe_int(value: Any) -> Optional[int]:
    try:
        if value is None:
            return None
        return int(float(value))
    except Exception:
        return None


def _clean_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    return "" if not text else text


def _contains_technical_error_text(value: Any) -> bool:
    text = _clean_text(value).lower()
    if not text:
        return False
    markers = [
        "apiconnectionerror",
        "openaierror",
        "openai api",
        "apierror",
        "llm error",
        "connection error",
        "connection failure",
        "connection reset",
        "openai_api_key",
        "traceback",
        "timeout",
        "timed out",
        "rate limit",
        "authentication error",
        "401",
        "403",
        "429",
        "500",
        "503",
        "error code",
        "request id",
        "internal server error",
        "bad gateway",
    ]
    return any(marker in text for marker in markers)


def _clean_business_text(value: Any) -> str:
    text = _clean_text(value)
    if not text or _contains_technical_error_text(text):
        return ""
    return text


def _clean_action_list(actions: Any) -> List[str]:
    if not isinstance(actions, list):
        return []

    cleaned: List[str] = []
    seen: set[str] = set()
    for action in actions:
        text = _clean_business_text(action)
        key = text.lower()
        if text and key not in seen:
            cleaned.append(text)
            seen.add(key)
    return cleaned


def _output_conflicts_with_diagnosis(
    state: Dict[str, Any],
    decision: Dict[str, Any],
    summary: str,
    actions: List[str],
) -> bool:
    stance = _clean_text(decision.get("stance")).lower()
    days_active = _safe_int(state.get("days_active"))
    roas = _safe_float(state.get("ROAS"))
    combined = " ".join([summary, *actions]).lower()

    if stance == "observe":
        aggressive_markers = [
            "escalate",
            "owner review",
            "pause campaign",
            "shut off",
            "shut down",
        ]
        if any(marker in combined for marker in aggressive_markers):
            return True
        if days_active is not None and days_active < 14 and "aggressive" in combined:
            return True

    if stance == "recommend":
        contradictory_markers = [
            "escalate",
            "immediate owner review",
            "pause campaign",
            "shut off",
            "shut down",
        ]
        if any(marker in combined for marker in contradictory_markers):
            return True
        if roas is not None and roas >= 1.0 and "below break-even" in combined:
            return True

    if stance == "escalate" and roas is not None and roas < 1.0:
        profit_markers = (
            "roas",
            "return on ad spend",
            "break-even",
            "breakeven",
            "profit",
            "loss",
            "margin",
            "below 1",
        )
        if not any(m in combined for m in profit_markers):
            return True

    return False
