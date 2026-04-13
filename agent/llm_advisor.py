from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Literal

from pydantic import BaseModel, Field


class AdvisorOutput(BaseModel):
    advisor_summary: str = Field(..., description="Short summary of what to do.")
    advisor_actions: List[str] = Field(..., description="3-6 concrete next steps.")
    advisor_confidence: Literal["low", "medium", "high"]


class LLMAdvisor:
    """
    LLM-backed advisor.

    Uses OpenAI Structured Outputs via client.responses.parse(...) with a Pydantic schema.
    If the call fails, falls back to deterministic advice.

    Env vars:
      - OPENAI_API_KEY
      - OPENAI_MODEL (optional) default: gpt-4.1-mini
    """

    def __init__(self, model: Optional[str] = None, use_llm: bool = True) -> None:
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
        self.use_llm = use_llm

    def advise(self, state: Dict[str, Any], decision: Dict[str, Any]) -> Dict[str, Any]:
        if not self.use_llm or not os.getenv("OPENAI_API_KEY"):
            return self._fallback(state=state, decision=decision)

        try:
            from openai import OpenAI  # openai>=1.x / 2.x
            client = OpenAI()

            system_msg, user_msg = self._build_messages(state=state, decision=decision)

            # Structured Outputs: parse directly into AdvisorOutput
            resp = client.responses.parse(
                model=self.model,
                input=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_msg},
                ],
                text_format=AdvisorOutput,
            )

            parsed: AdvisorOutput = resp.output_parsed  # type: ignore

            out = {
                "advisor_summary": parsed.advisor_summary.strip(),
                "advisor_actions": [a.strip() for a in parsed.advisor_actions if a.strip()],
                "advisor_confidence": parsed.advisor_confidence,
                "advisor_reasons_seen": decision.get("reasons", []),
                "advisor_used_llm": True,
                "advisor_model": self.model,
                "advisor_raw_text": None,
            }
            return self._finalize_output(state=state, decision=decision, out=out)

        except Exception:
            return self._fallback(state=state, decision=decision)

    def _build_messages(self, state: Dict[str, Any], decision: Dict[str, Any]) -> tuple[str, str]:
        system_msg = (
            "You are a senior performance marketing expert. "
            "Give concise, practical advice that matches the agent stance."
        )

        # Keep user msg very explicit; schema enforcement is handled by parse()
        user_msg = (
            "Given the campaign STATE and the rule-based DECISION, output:\n"
            "- advisor_summary: 1-2 sentences\n"
            "- advisor_actions: 3-6 bullet steps (specific)\n"
            "- advisor_confidence: low|medium|high\n\n"
            "Constraints:\n"
            "- If stance is 'observe', avoid aggressive actions.\n"
            "- If ROAS < 1.0, include a risk-control action.\n\n"
            f"STATE: {state}\n"
            f"DECISION: {decision}\n"
        )
        return system_msg, user_msg

    def _fallback(self, state: Dict[str, Any], decision: Dict[str, Any], note: str = "") -> Dict[str, Any]:
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
                    actions.insert(1, "Keep spend disciplined while ROAS remains below 1.0 during the learning period.")
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

        return {
            "advisor_summary": summary,
            "advisor_actions": actions,
            "advisor_reasons_seen": reasons,
            "advisor_confidence": "medium" if severity in ("medium", "high") else "high",
            "advisor_used_llm": False,
            "advisor_model": None,
            "advisor_raw_text": None,
            "advisor_fallback_note": note or "Deterministic advisor fallback used.",
        }

    def _finalize_output(self, state: Dict[str, Any], decision: Dict[str, Any], out: Dict[str, Any]) -> Dict[str, Any]:
        summary = _clean_business_text(out.get("advisor_summary"))
        actions = _clean_action_list(out.get("advisor_actions", []))

        if not summary or not actions:
            return self._fallback(state=state, decision=decision)

        if _output_conflicts_with_diagnosis(state=state, decision=decision, summary=summary, actions=actions):
            return self._fallback(state=state, decision=decision)

        out["advisor_summary"] = summary
        out["advisor_actions"] = actions
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
        "llm error",
        "connection error",
        "connection failure",
        "openai_api_key",
        "traceback",
        "timeout",
        "rate limit",
        "authentication error",
        "error code",
        "request id",
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
        aggressive_markers = ["escalate", "owner review", "pause campaign", "shut off"]
        if any(marker in combined for marker in aggressive_markers):
            return True
        if days_active is not None and days_active < 14 and "aggressive" in combined:
            return True

    if stance == "recommend":
        contradictory_markers = ["escalate", "immediate owner review", "pause campaign", "shut off"]
        if any(marker in combined for marker in contradictory_markers):
            return True
        if roas is not None and roas >= 1.0 and "below break-even" in combined:
            return True

    if stance == "escalate" and roas is not None and roas < 1.0:
        if "below break-even" not in combined and "roas" not in combined:
            return True

    return False
