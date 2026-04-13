from __future__ import annotations

import json
import os
import random
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
      - OPENAI_ADVISOR_MAX_ATTEMPTS (optional) default 6
      - OPENAI_ADVISOR_MIN_INTERVAL (optional) min seconds between API calls, default 0.25
    """

    def __init__(self, model: Optional[str] = None, use_llm: bool = True) -> None:
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
        self.use_llm = use_llm
        self._timeout = float(os.getenv("OPENAI_ADVISOR_TIMEOUT", "90"))
        self._max_attempts = max(1, int(os.getenv("OPENAI_ADVISOR_MAX_ATTEMPTS", "6")))

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
        slim_decision = _slim_decision_for_llm(decision)

        from openai import (
            APIConnectionError,
            APIStatusError,
            APITimeoutError,
            AuthenticationError,
            BadRequestError,
            OpenAI,
            RateLimitError,
        )

        # One client per advise(); httpx max_retries handles low-level connection blips.
        client = OpenAI(timeout=self._timeout, max_retries=2)

        last_failure_reason = "api_failure"

        for attempt in range(self._max_attempts):
            try:
                _advisor_request_throttle()
                system_msg, user_msg = self._build_messages(
                    state=slim_state, decision=slim_decision
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
                    last_failure_reason = "empty_parse"
                    _sleep_transient_backoff(attempt, base=0.75, cap=12.0)
                    continue

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

            except RateLimitError as e:
                last_failure_reason = "rate_limited"
                _sleep_for_rate_limit(e, attempt)
                continue

            except APITimeoutError:
                last_failure_reason = "timeout"
                _sleep_transient_backoff(attempt, base=1.0, cap=20.0)
                continue

            except APIConnectionError:
                last_failure_reason = "connection_error"
                _sleep_transient_backoff(attempt, base=0.6, cap=15.0)
                continue

            except APIStatusError as e:
                code = getattr(e, "status_code", None)
                if code in (408, 429, 500, 502, 503, 504):
                    last_failure_reason = (
                        "rate_limited" if code == 429 else "server_error"
                    )
                    _sleep_for_status_code(code, attempt, e)
                    continue
                if code in (401, 403):
                    return self._fallback(
                        state=state,
                        decision=decision,
                        reason="auth_error",
                    )
                if code == 400:
                    return self._fallback(
                        state=state,
                        decision=decision,
                        reason="bad_request",
                    )
                return self._fallback(
                    state=state,
                    decision=decision,
                    reason="api_failure",
                )

            except AuthenticationError:
                return self._fallback(
                    state=state,
                    decision=decision,
                    reason="auth_error",
                )

            except BadRequestError:
                return self._fallback(
                    state=state,
                    decision=decision,
                    reason="bad_request",
                )

            except KeyboardInterrupt:
                raise
            except Exception:
                last_failure_reason = "api_failure"
                _sleep_transient_backoff(attempt, base=0.5, cap=8.0)
                continue

        return self._fallback(
            state=state,
            decision=decision,
            reason=last_failure_reason,
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

        state_json = json.dumps(state, ensure_ascii=False, default=str, separators=(",", ":"))
        decision_json = json.dumps(
            decision, ensure_ascii=False, default=str, separators=(",", ":")
        )
        user_msg = (
            "Given STATE_JSON (metrics) and DECISION_JSON (stance, severity, reasons), output:\n"
            "- advisor_summary: 1-2 sentences\n"
            "- advisor_actions: 3-6 short imperative steps (each under 200 characters)\n"
            "- advisor_confidence: low|medium|high\n\n"
            "Constraints:\n"
            "- If stance is 'observe', avoid aggressive actions.\n"
            "- If stance is 'recommend', prefer measured optimizations, not escalation language.\n"
            "- If stance is 'escalate', be direct about owner review and risk control.\n"
            "- If ROAS < 1.0 in STATE_JSON, include at least one risk-control or spend-discipline action.\n\n"
            f"STATE_JSON:\n{state_json}\n\n"
            f"DECISION_JSON:\n{decision_json}\n"
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


# Serialize OpenAI calls slightly across a batch run (reduces 429 bursts on low-RPM keys).
_LAST_ADVISOR_CALL_MONO: float = 0.0
_DECISION_REASON_MAX_LEN = 320


def _advisor_request_throttle() -> None:
    global _LAST_ADVISOR_CALL_MONO
    min_interval = float(os.getenv("OPENAI_ADVISOR_MIN_INTERVAL", "0.25"))
    if min_interval <= 0:
        return
    now = time.monotonic()
    gap = min_interval - (now - _LAST_ADVISOR_CALL_MONO)
    if gap > 0:
        time.sleep(gap)
    _LAST_ADVISOR_CALL_MONO = time.monotonic()


def _slim_decision_for_llm(decision: Dict[str, Any]) -> Dict[str, Any]:
    """
    Decision dicts can include very long reason strings (e.g. trend boilerplate).
    That inflates tokens, latency, and timeout risk versus slimmer rows — same run, uneven failures.
    """
    reasons = decision.get("reasons", [])
    slim_reasons: List[str] = []
    if isinstance(reasons, list):
        for item in reasons[:8]:
            text = _clean_text(item)
            if not text:
                continue
            if len(text) > _DECISION_REASON_MAX_LEN:
                text = text[: _DECISION_REASON_MAX_LEN - 1].rstrip() + "…"
            slim_reasons.append(text)
    return {
        "stance": decision.get("stance"),
        "severity": decision.get("severity"),
        "reasons": slim_reasons,
    }


def _sleep_transient_backoff(attempt: int, *, base: float, cap: float) -> None:
    exp = min(cap, base * (2**attempt))
    jitter = random.uniform(0, min(1.0, max(0.05, exp * 0.12)))
    time.sleep(min(cap, exp + jitter))


def _retry_after_seconds(exc: BaseException) -> Optional[float]:
    resp = getattr(exc, "response", None)
    if resp is None:
        return None
    headers = getattr(resp, "headers", None)
    if not headers:
        return None
    raw = headers.get("retry-after") or headers.get("Retry-After")
    if raw is None:
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def _sleep_for_rate_limit(exc: BaseException, attempt: int) -> None:
    ra = _retry_after_seconds(exc)
    if ra is not None and ra > 0:
        time.sleep(min(90.0, ra + random.uniform(0, 0.35)))
        return
    _sleep_transient_backoff(attempt, base=2.0, cap=45.0)


def _sleep_for_status_code(code: Optional[int], attempt: int, exc: BaseException) -> None:
    if code == 429:
        _sleep_for_rate_limit(exc, attempt)
    else:
        _sleep_transient_backoff(attempt, base=1.0, cap=20.0)


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
