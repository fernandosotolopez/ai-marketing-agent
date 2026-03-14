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

    def __init__(self, model: Optional[str] = None) -> None:
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-4.1-mini")

    def advise(self, state: Dict[str, Any], decision: Dict[str, Any]) -> Dict[str, Any]:
        if not os.getenv("OPENAI_API_KEY"):
            return self._fallback(state=state, decision=decision, note="OPENAI_API_KEY not set")

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

            if not out["advisor_summary"]:
                out["advisor_summary"] = "LLM returned empty summary (unexpected)."
            if not out["advisor_actions"]:
                out["advisor_actions"] = ["Review metrics and investigate root cause."]

            return out

        except Exception as e:
            return self._fallback(
                state=state,
                decision=decision,
                note=f"LLM error ({type(e).__name__}): {repr(e)}",
            )

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
        roas = state.get("ROAS")

        if stance == "observe":
            summary = "Monitor the campaign and avoid aggressive changes for now."
            actions = [
                "Keep tracking CPA and ROAS daily",
                "Wait for more stable data before making big edits",
            ]
        elif stance == "recommend":
            summary = "Small optimizations recommended to improve efficiency."
            actions = [
                "Test 1 new creative variation",
                "Try a small audience refinement",
                "Review landing page conversion friction",
            ]
        else:
            summary = "Escalate: risk is significant. Needs human review with context."
            actions = [
                "Reduce budget slightly while investigating",
                "Check if tracking / attribution changed recently",
                "Inspect creatives for fatigue and audience overlap",
            ]

        try:
            if roas is not None and float(roas) < 1.0:
                actions.insert(0, "Risk control: cap or reduce spend until ROAS recovers above 1.0")
        except Exception:
            pass

        return {
            "advisor_summary": summary + (f" ({note})" if note else ""),
            "advisor_actions": actions,
            "advisor_reasons_seen": reasons,
            "advisor_confidence": "medium" if severity in ("medium", "high") else "high",
            "advisor_used_llm": False,
            "advisor_model": None,
            "advisor_raw_text": None,
        }
