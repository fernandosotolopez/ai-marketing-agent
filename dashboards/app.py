# dashboards/app.py
import json
import re
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


# -----------------------------
# Page config
# -----------------------------
st.set_page_config(
    page_title="Marketing Decision Review",
    page_icon="📊",
    layout="wide",
)


# -----------------------------
# Constants / schema helpers
# -----------------------------
REQUIRED_COLS = [
    "campaign_id",
    "stance",
    "severity",
    "cpa",
    "target_cpa",
    "roas",
    "cpa_trend_7d",
    "roas_trend_7d",
    "days_active",
]

OPTIONAL_COLS = [
    "risk_score",
    "why_flagged",
    "run_id",
    "reasons",
    "reasons_text",
    "decision_explanation",
    "warnings",
    "warning_count",
    "provenance",
    "execution_metadata",
    "advisor_summary",
    "advisor_actions",
    "advisor_confidence",
    "advisor_used_llm",
    "advisor_model",
    "analysis_insights",
    "analysis_suggested_actions",
    "analysis_summary",
    "scenarios",
    # optional business impact fields (if you later add them)
    "spend_7d",
    "revenue_7d",
    "conversions_7d",
]

SEVERITY_ORDER = {"high": 3, "medium": 2, "low": 1, "unknown": 0}
STANCE_ORDER = {"escalate": 3, "adjust": 2, "observe": 1, "unknown": 0}


def _safe_float(x, default=np.nan) -> float:
    try:
        if pd.isna(x):
            return default
        return float(x)
    except Exception:
        return default


def _safe_int(x, default=np.nan) -> float:
    try:
        if pd.isna(x):
            return default
        return int(float(x))
    except Exception:
        return default


def _safe_json_loads(s: Any, default: Any) -> Any:
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return default
    if isinstance(s, (dict, list)):
        return s
    try:
        return json.loads(str(s))
    except Exception:
        return default


def _first_nonempty_line(text: Any) -> str:
    if text is None or (isinstance(text, float) and np.isnan(text)):
        return ""
    s = str(text).strip()
    if not s or s.lower() == "nan":
        return ""
    # take first sentence-ish chunk
    for sep in ["\n", ". "]:
        if sep in s:
            s = s.split(sep)[0].strip()
            break
    return s


def _short_reason(why_flagged: Any, max_len: int = 120) -> str:
    if why_flagged is None or (isinstance(why_flagged, float) and np.isnan(why_flagged)):
        return ""
    s = str(why_flagged).strip()
    if not s or s.lower() == "nan":
        return ""
    # take first reason segment
    first = s.split("|")[0].strip()
    if len(first) > max_len:
        first = first[: max_len - 1].rstrip() + "…"
    return first


def _clean_text(value: Any) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return ""
    return text


def _string_list(value: Any) -> List[str]:
    if isinstance(value, list):
        return [_clean_text(item) for item in value if _clean_text(item)]
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    text = _clean_text(value)
    return [text] if text else []


def _decision_explanation_from_row(row: pd.Series) -> str:
    reasons = row.get("reasons", [])
    if isinstance(reasons, list):
        for reason in reasons:
            clean = _clean_text(reason)
            if clean:
                return clean

    for key in ["analysis_summary", "advisor_summary", "why_flagged"]:
        clean = _clean_text(row.get(key))
        if clean:
            return clean
    return ""


def _format_pct(value: Any) -> str:
    if pd.isna(value):
        return "—"
    return f"{float(value) * 100:+.0f}%"


def _format_risk_label(value: Any) -> str:
    sev = _clean_text(value).lower()
    return sev.upper() if sev in {"high", "medium", "low"} else "UNKNOWN"


def _format_decision_label(value: Any) -> str:
    decision = _clean_text(value).lower()
    mapping = {
        "escalate": "Escalate",
        "recommend": "Review",
        "adjust": "Review",
        "observe": "Monitor",
    }
    return mapping.get(decision, "Review")


def _decision_css_class(value: Any) -> str:
    decision = _clean_text(value).lower()
    mapping = {
        "escalate": "decision-escalate",
        "recommend": "decision-review",
        "adjust": "decision-review",
        "observe": "decision-monitor",
    }
    return mapping.get(decision, "decision-review")


def _priority_css_class(value: Any) -> str:
    priority = _clean_text(value).lower()
    mapping = {
        "high": "priority-high",
        "medium": "priority-medium",
        "low": "priority-low",
    }
    return mapping.get(priority, "priority-neutral")


def _badge_html(label: str, css_class: str) -> str:
    return f'<span class="badge {css_class}">{label}</span>'


def _humanize_reason(value: Any) -> str:
    text = _clean_text(value)
    if not text:
        return ""

    text = re.sub(r"^\[[^\]]+\]\s*", "", text)
    text = text.replace(
        "(supporting context; maturity gate keeps stance at observe)",
        "(supporting context; early-campaign guardrail keeps this in monitor mode)",
    )

    replacements = {
        "ROAS < 1.0": "ROAS below 1.0",
        "pre-LTV": "before lifetime value assumptions",
        "Campaign is immature": "Campaign is still early",
        "maturity gate keeps stance at observe": "early-campaign guardrail keeps this in monitor mode",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)

    return " ".join(text.split())


def _key_evidence_line(row: pd.Series, max_items: int = 2) -> str:
    bits = _build_evidence_bits(row)
    if not bits:
        return ""
    return " | ".join(bits[:max_items])


def _top_portfolio_issue(df: pd.DataFrame) -> str:
    if df.empty:
        return ""

    reasons = (
        df["decision_explanation"]
        .dropna()
        .astype("string")
        .map(_humanize_reason)
        .tolist()
    )
    for reason in reasons:
        if reason:
            return reason
    return ""


def _portfolio_signal_text(df: pd.DataFrame) -> str:
    if df.empty:
        return "No campaigns are currently visible in the review set."

    campaigns_count = len(df)
    action_count = int(df["stance"].isin(["escalate", "recommend", "adjust"]).sum())
    high_count = int((df["severity"] == "high").sum())

    if action_count == 0:
        return f"All {campaigns_count} visible campaigns are in monitor mode."
    if high_count > 0:
        return f"{action_count} of {campaigns_count} campaigns need attention, including {high_count} high-priority item(s)."
    return f"{action_count} of {campaigns_count} campaigns need review, with no high-priority escalations in the current filter set."


def _build_evidence_bits(row: pd.Series) -> List[str]:
    bits: List[str] = []

    cpa = row.get("cpa")
    target_cpa = row.get("target_cpa")
    roas = row.get("roas")
    cpa_trend = row.get("cpa_trend_7d")
    roas_trend = row.get("roas_trend_7d")
    days_active = row.get("days_active")

    if pd.notna(cpa) and pd.notna(target_cpa):
        bits.append(f"CPA {float(cpa):.2f} vs target {float(target_cpa):.2f}")
    elif pd.notna(cpa):
        bits.append(f"CPA {float(cpa):.2f}")

    if pd.notna(roas):
        bits.append(f"ROAS {float(roas):.2f}")

    if pd.notna(cpa_trend):
        bits.append(f"CPA 7d {_format_pct(cpa_trend)}")

    if pd.notna(roas_trend):
        bits.append(f"ROAS 7d {_format_pct(roas_trend)}")

    if pd.notna(days_active):
        bits.append(f"{int(days_active)} days active")

    return bits


def _build_decision_signal_bits(row: pd.Series) -> List[str]:
    bits: List[str] = []

    cpa = row.get("cpa")
    target_cpa = row.get("target_cpa")
    roas = row.get("roas")
    cpa_trend = row.get("cpa_trend_7d")
    roas_trend = row.get("roas_trend_7d")

    if pd.notna(cpa) and pd.notna(target_cpa) and float(target_cpa) > 0:
        over_pct = (float(cpa) / float(target_cpa) - 1) * 100
        if over_pct > 0:
            bits.append(f"CPA is {over_pct:.0f}% above target ({float(cpa):.2f} vs {float(target_cpa):.2f}).")

    if pd.notna(cpa_trend) and float(cpa_trend) > 0:
        bits.append(f"CPA has worsened {_format_pct(cpa_trend)} over the last 7 days.")

    if pd.notna(roas_trend) and float(roas_trend) < 0:
        bits.append(f"ROAS has weakened {_format_pct(roas_trend)} over the last 7 days.")

    if pd.notna(roas) and float(roas) < 1.0:
        bits.append(f"ROAS is below 1.0 at {float(roas):.2f}, indicating potential loss on ad spend before lifetime value.")

    return bits


def _build_action_options(row: pd.Series, primary_action: str) -> List[str]:
    options: List[str] = []
    seen: set[str] = set()

    primary_key = _clean_text(primary_action).strip().lower()
    if primary_key:
        seen.add(primary_key)

    advisor_actions = row.get("advisor_actions", [])
    if isinstance(advisor_actions, list):
        for action in advisor_actions:
            clean_action = _clean_text(action)
            key = clean_action.strip().lower()
            if clean_action and key not in seen:
                options.append(clean_action)
                seen.add(key)

    suggested_actions = row.get("analysis_suggested_actions", [])
    if isinstance(suggested_actions, list):
        for action in suggested_actions:
            clean_action = _clean_text(action)
            key = clean_action.strip().lower()
            if clean_action and key not in seen:
                options.append(clean_action)
                seen.add(key)

    return options


def build_hero_impact_data(row: pd.Series) -> Dict[str, str]:
    cpa = _safe_float(row.get("cpa"))
    target_cpa = _safe_float(row.get("target_cpa"))
    cpa_trend = _safe_float(row.get("cpa_trend_7d"))

    if np.isnan(cpa) and np.isnan(target_cpa):
        return {
            "delta_value": "--",
            "delta_label": "target unavailable",
            "current_cpa": "—",
            "target_cpa": "—",
            "trend_value": "—",
        }

    if pd.notna(target_cpa) and float(target_cpa) > 0 and pd.notna(cpa):
        delta_pct = (float(cpa) / float(target_cpa) - 1) * 100
        delta_value = f"{delta_pct:+.0f}%"
        delta_label = "above target" if delta_pct >= 0 else "below target"
    else:
        delta_value = "--"
        delta_label = "target unavailable"

    trend_value = "—"
    if pd.notna(cpa_trend):
        trend_value = _format_pct(cpa_trend)

    return {
        "delta_value": delta_value,
        "delta_label": delta_label,
        "current_cpa": "—" if np.isnan(cpa) else f"{float(cpa):.2f}",
        "target_cpa": "—" if np.isnan(target_cpa) else f"{float(target_cpa):.2f}",
        "trend_value": trend_value,
    }


def render_hero_impact_card(data: Dict[str, str]) -> None:
    with st.container(border=True):
        st.markdown(
            f"""
            <div class="hero-impact-shell">
                <div class="hero-impact-title">CPA gap vs target</div>
                <div class="hero-impact-hero">
                    <div class="hero-impact-value alert">{data['delta_value']}</div>
                    <div class="hero-impact-label">{data['delta_label']}</div>
                </div>
                <div class="hero-impact-metrics">
                    <div class="hero-impact-metric">
                        <div class="hero-impact-metric-label">Current CPA</div>
                        <div class="hero-impact-metric-value">{data['current_cpa']}</div>
                    </div>
                    <div class="hero-impact-metric">
                        <div class="hero-impact-metric-label">Target CPA</div>
                        <div class="hero-impact-metric-value">{data['target_cpa']}</div>
                    </div>
                    <div class="hero-impact-metric">
                        <div class="hero-impact-metric-label">7d CPA trend</div>
                        <div class="hero-impact-metric-value tertiary">{data['trend_value']}</div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def sort_campaigns_for_review(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()

    out = df.copy()
    out["_severity_rank"] = out["severity"].map(SEVERITY_ORDER).fillna(0)
    out["_stance_rank"] = out["stance"].map(STANCE_ORDER).fillna(0)
    if "warning_count" not in out.columns:
        out["warning_count"] = 0
    out["_warning_rank"] = out["warning_count"].fillna(0)
    out = out.sort_values(
        ["_severity_rank", "_stance_rank", "_warning_rank", "campaign_id"],
        ascending=[False, False, False, True],
    )
    return out.drop(columns=["_severity_rank", "_stance_rank", "_warning_rank"], errors="ignore")


def build_campaign_display_df(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(
            columns=["Campaign", "Recommended review", "Priority", "Why it matters", "Key evidence", "Data notes"]
        )

    rows: List[Dict[str, Any]] = []
    for _, row in sort_campaigns_for_review(df).iterrows():
        warnings = row.get("warnings", [])
        warning_text = " | ".join(warnings[:2]) if isinstance(warnings, list) else ""
        rows.append(
            {
                "Campaign": row.get("campaign_id"),
                "Recommended review": _format_decision_label(row.get("stance")),
                "Priority": _format_risk_label(row.get("severity")).title(),
                "Why it matters": _humanize_reason(_decision_explanation_from_row(row)),
                "Key evidence": _key_evidence_line(row),
                "Data notes": _humanize_reason(warning_text),
            }
        )
    return pd.DataFrame(rows)


def compute_risk_score(df: pd.DataFrame) -> pd.Series:
    """
    Heuristic risk score (dashboard-side).
    Not a "ground truth" metric; it's for prioritization / triage.
    """
    cpa_over = (df["cpa"] / df["target_cpa"]).replace([np.inf, -np.inf], np.nan)
    cpa_over = cpa_over.fillna(1.0)

    roas_bad = (1.0 - df["roas"]).clip(lower=0)
    trend_penalty = (
        df["cpa_trend_7d"].fillna(0).clip(lower=0)
        + (-df["roas_trend_7d"].fillna(0)).clip(lower=0)
    )

    sev_w = df["severity"].map(SEVERITY_ORDER).fillna(1)

    score = (cpa_over - 1.0).clip(lower=0) * 8 + roas_bad * 10 + trend_penalty * 6
    score = score * (1 + 0.15 * (sev_w - 1))
    return score.round(4)


def default_why_flagged(row: pd.Series) -> str:
    msgs = []

    if pd.notna(row.get("cpa")) and pd.notna(row.get("target_cpa")) and row["target_cpa"] > 0:
        over_pct = (row["cpa"] / row["target_cpa"] - 1) * 100
        if over_pct > 0:
            msgs.append(
                f"[cost_efficiency] CPA ({row['cpa']:.2f}) exceeds target ({row['target_cpa']:.2f}) by {over_pct:.0f}%"
            )

    if pd.notna(row.get("roas")) and row["roas"] < 1.0:
        msgs.append(
            f"[profitability] ROAS < 1.0 (ROAS={row['roas']:.2f}) suggests potential loss on ad spend (pre-LTV)."
        )

    if pd.notna(row.get("cpa_trend_7d")) and row["cpa_trend_7d"] > 0.15:
        msgs.append(f"[performance_trend] CPA rising fast (+{row['cpa_trend_7d']*100:.0f}% over 7d).")

    if pd.notna(row.get("roas_trend_7d")) and row["roas_trend_7d"] < -0.10:
        msgs.append(f"[performance_trend] ROAS dropping fast ({row['roas_trend_7d']*100:.0f}% over 7d).")

    if pd.notna(row.get("days_active")) and row["days_active"] < 14:
        msgs.append(
            f"[campaign_maturity] Campaign is immature ({int(row['days_active'])} days active; minimum 14 suggested)."
        )

    if not msgs:
        return "All goals satisfied"

    return " | ".join(msgs)


def normalize_df(df: pd.DataFrame, *, fill_dashboard_fallbacks: bool = True) -> pd.DataFrame:
    """Normalize column names & dtypes, add missing columns if needed."""
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]

    rename_map = {
        "CPA": "cpa",
        "ROAS": "roas",
        "target_CPA": "target_cpa",
        "targetCPA": "target_cpa",
        "CPA_trend_7d": "cpa_trend_7d",
        "ROAS_trend_7d": "roas_trend_7d",
        "campaignId": "campaign_id",
    }
    for k, v in rename_map.items():
        if k in df.columns and v not in df.columns:
            df.rename(columns={k: v}, inplace=True)

    for col in REQUIRED_COLS:
        if col not in df.columns:
            df[col] = pd.NA
    for col in OPTIONAL_COLS:
        if col not in df.columns:
            df[col] = pd.NA

    df["campaign_id"] = df["campaign_id"].astype("string").str.strip()
    df["stance"] = df["stance"].astype("string").str.lower().str.strip()
    df["severity"] = df["severity"].astype("string").str.lower().str.strip()

    df["cpa"] = df["cpa"].apply(_safe_float)
    df["target_cpa"] = df["target_cpa"].apply(_safe_float)
    df["roas"] = df["roas"].apply(_safe_float)
    df["cpa_trend_7d"] = df["cpa_trend_7d"].apply(_safe_float)
    df["roas_trend_7d"] = df["roas_trend_7d"].apply(_safe_float)
    df["days_active"] = df["days_active"].apply(_safe_int)

    # ratios for visual clarity
    df["cpa_ratio"] = (df["cpa"] / df["target_cpa"]).replace([np.inf, -np.inf], np.nan)

    if "reasons" not in df.columns:
        df["reasons"] = [[] for _ in range(len(df))]
    else:
        df["reasons"] = df["reasons"].apply(_string_list)

    if "warnings" not in df.columns:
        df["warnings"] = [[] for _ in range(len(df))]
    else:
        df["warnings"] = df["warnings"].apply(_string_list)

    if "provenance" not in df.columns:
        df["provenance"] = [{} for _ in range(len(df))]

    if "execution_metadata" not in df.columns:
        df["execution_metadata"] = [{} for _ in range(len(df))]

    df["warning_count"] = df["warnings"].apply(lambda x: len(x) if isinstance(x, list) else 0)
    df["reasons_text"] = df["reasons"].apply(lambda x: " | ".join(x) if isinstance(x, list) and x else pd.NA)

    if fill_dashboard_fallbacks and df["risk_score"].isna().all():
        df["risk_score"] = compute_risk_score(df)

    missing_why = df["why_flagged"].isna() | (df["why_flagged"].astype("string").str.strip() == "")
    if fill_dashboard_fallbacks and missing_why.any():
        df.loc[missing_why, "why_flagged"] = df[missing_why].apply(default_why_flagged, axis=1)

    if "decision_explanation" not in df.columns:
        df["decision_explanation"] = pd.NA
    missing_explanation = df["decision_explanation"].isna() | (
        df["decision_explanation"].astype("string").str.strip() == ""
    )
    if missing_explanation.any():
        df.loc[missing_explanation, "decision_explanation"] = df[missing_explanation].apply(
            _decision_explanation_from_row, axis=1
        )

    return df


# -----------------------------
# Path resolution
# -----------------------------
def resolve_paths() -> Dict[str, Path]:
    app_dir = Path(__file__).resolve().parent
    project_dir = app_dir.parent
    cwd = Path.cwd()

    data_candidates = [
        project_dir / "data",
        app_dir / "data",
        cwd / "data",
    ]
    chosen_data = next((p for p in data_candidates if p.exists()), data_candidates[0])

    db_candidates = [
        chosen_data / "agent_runs.db",
        project_dir / "data" / "agent_runs.db",
        cwd / "data" / "agent_runs.db",
    ]
    chosen_db = next((p for p in db_candidates if p.exists()), db_candidates[0])

    return {
        "APP_DIR": app_dir,
        "PROJECT_DIR": project_dir,
        "CWD": cwd,
        "DATA_DIR": chosen_data,
        "RUNS_DIR": chosen_data / "runs",
        "DB_PATH": chosen_db,
    }


PATHS = resolve_paths()


# -----------------------------
# Data loaders
# -----------------------------
def list_runs_folder(runs_dir: Path) -> List[str]:
    if not runs_dir.exists():
        return []
    runs = [p.name for p in runs_dir.iterdir() if p.is_dir()]
    runs.sort(reverse=True)
    return runs


def list_local_csvs(data_dir: Path) -> List[Path]:
    if not data_dir.exists():
        return []
    csvs = sorted(list(data_dir.glob("*.csv")), key=lambda p: p.stat().st_mtime, reverse=True)
    return csvs


@st.cache_data(show_spinner=False)
def load_csv_path(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    return normalize_df(df)


@st.cache_data(show_spinner=False)
def load_run_csv(run_dir: Path) -> Optional[pd.DataFrame]:
    csv_path = run_dir / "campaigns.csv"
    if not csv_path.exists():
        csvs = list(run_dir.glob("*.csv"))
        if not csvs:
            return None
        csv_path = csvs[0]
    return load_csv_path(csv_path)


@st.cache_data(show_spinner=False)
def list_db_runs(db_path: Path) -> pd.DataFrame:
    if not db_path.exists():
        return pd.DataFrame()

    con = sqlite3.connect(str(db_path))
    try:
        run_cols = {row[1] for row in con.execute("PRAGMA table_info(runs)").fetchall()}
        select_cols = [
            "run_id",
            "started_at_utc",
            "input_csv",
            "max_rows",
            "save_memory",
            "model",
            "used_llm",
            "notes",
        ]
        if "run_metadata_json" in run_cols:
            select_cols.append("run_metadata_json")
        else:
            select_cols.append("'{}' AS run_metadata_json")
        df = pd.read_sql_query(
            f"SELECT {', '.join(select_cols)} FROM runs ORDER BY started_at_utc DESC",
            con,
        )
    finally:
        con.close()

    if df.empty:
        return df

    df["label"] = (
        df["run_id"].astype("string")
        + "  |  "
        + df["started_at_utc"].astype("string")
        + "  |  "
        + df["input_csv"].astype("string")
    )
    return df


@st.cache_data(show_spinner=False)
def load_db_run_outputs(db_path: Path, run_id: str) -> pd.DataFrame:
    if not db_path.exists():
        return pd.DataFrame()

    con = sqlite3.connect(str(db_path))
    try:
        campaign_cols = {row[1] for row in con.execute("PRAGMA table_info(campaign_outputs)").fetchall()}
        run_cols = {row[1] for row in con.execute("PRAGMA table_info(runs)").fetchall()}

        select_cols = [
            "co.run_id",
            "co.campaign_id",
            "co.stance",
            "co.severity",
            "co.cpa",
            "co.roas",
            "co.target_cpa",
            "co.cpa_trend_7d",
            "co.roas_trend_7d",
            "co.days_active",
            "co.state_json",
            "co.decision_json",
            "co.analysis_json",
            "co.advisor_json",
            "co.scenarios_json",
            "r.started_at_utc AS run_started_at_utc",
            "r.input_csv AS run_input_csv",
            "r.max_rows AS run_max_rows",
            "r.save_memory AS run_save_memory",
            "r.model AS run_model",
            "r.used_llm AS run_used_llm",
            "r.notes AS run_notes",
        ]
        if "run_metadata_json" in run_cols:
            select_cols.append("r.run_metadata_json")
        else:
            select_cols.append("'{}' AS run_metadata_json")

        optional_json_defaults = {
            "warnings_json": "'[]' AS warnings_json",
            "provenance_json": "'{}' AS provenance_json",
            "execution_metadata_json": "'{}' AS execution_metadata_json",
        }
        for col, default_sql in optional_json_defaults.items():
            if col in campaign_cols:
                select_cols.append(f"co.{col}")
            else:
                select_cols.append(default_sql)

        df = pd.read_sql_query(
            f"SELECT {', '.join(select_cols)} "
            "FROM campaign_outputs co "
            "LEFT JOIN runs r ON r.run_id = co.run_id "
            "WHERE co.run_id = ?",
            con,
            params=(run_id,),
        )
    finally:
        con.close()

    if df.empty:
        return df

    def _decision_fields(dec_json: Any) -> Dict[str, Any]:
        d = _safe_json_loads(dec_json, {})
        if not isinstance(d, dict):
            return {
                "reasons": [],
                "reasons_text": pd.NA,
                "decision_explanation": pd.NA,
                "decision_stance": pd.NA,
                "decision_severity": pd.NA,
            }
        reasons = _string_list(d.get("reasons"))
        joined = " | ".join(reasons) if reasons else pd.NA
        return {
            "reasons": reasons,
            "reasons_text": joined,
            "decision_explanation": reasons[0] if reasons else pd.NA,
            "decision_stance": d.get("stance", pd.NA),
            "decision_severity": d.get("severity", pd.NA),
        }

    decision_expanded = df["decision_json"].apply(_decision_fields).apply(pd.Series)
    for c in decision_expanded.columns:
        df[c] = decision_expanded[c]

    df["why_flagged"] = df["reasons_text"]
    missing_stance = df["stance"].isna() | (df["stance"].astype("string").str.strip() == "")
    if missing_stance.any():
        df.loc[missing_stance, "stance"] = df.loc[missing_stance, "decision_stance"]
    missing_severity = df["severity"].isna() | (df["severity"].astype("string").str.strip() == "")
    if missing_severity.any():
        df.loc[missing_severity, "severity"] = df.loc[missing_severity, "decision_severity"]

    def _advisor_fields(ad_json: Any) -> Dict[str, Any]:
        a = _safe_json_loads(ad_json, {})
        if not isinstance(a, dict):
            return {
                "advisor_summary": pd.NA,
                "advisor_actions": [],
                "advisor_confidence": pd.NA,
                "advisor_used_llm": pd.NA,
                "advisor_model": pd.NA,
            }
        actions = a.get("advisor_actions", [])
        if isinstance(actions, list):
            actions = [str(x).strip() for x in actions if str(x).strip()]
        else:
            actions = []
        return {
            "advisor_summary": a.get("advisor_summary", pd.NA),
            "advisor_actions": actions,
            "advisor_confidence": a.get("advisor_confidence", pd.NA),
            "advisor_used_llm": a.get("advisor_used_llm", pd.NA),
            "advisor_model": a.get("advisor_model", pd.NA),
        }

    advisor_expanded = df["advisor_json"].apply(_advisor_fields).apply(pd.Series)
    for c in advisor_expanded.columns:
        df[c] = advisor_expanded[c]

    def _analysis_fields(an_json: Any) -> Dict[str, Any]:
        a = _safe_json_loads(an_json, {})
        if not isinstance(a, dict):
            return {"analysis_insights": [], "analysis_suggested_actions": [], "analysis_summary": pd.NA}

        insights = a.get("insights", [])
        out_ins = []
        if isinstance(insights, list):
            for it in insights:
                if isinstance(it, dict):
                    imp = str(it.get("importance", "")).upper()
                    cat = str(it.get("category", "")).strip()
                    msg = str(it.get("message", "")).strip()
                    if msg:
                        out_ins.append(f"[{imp}] ({cat}) {msg}".strip())
                else:
                    s = str(it).strip()
                    if s:
                        out_ins.append(s)

        acts = a.get("suggested_actions", [])
        out_acts = []
        if isinstance(acts, list):
            out_acts = [str(x).strip() for x in acts if str(x).strip()]

        return {
            "analysis_insights": out_ins,
            "analysis_suggested_actions": out_acts,
            "analysis_summary": a.get("summary", pd.NA),
        }

    analysis_expanded = df["analysis_json"].apply(_analysis_fields).apply(pd.Series)
    for c in analysis_expanded.columns:
        df[c] = analysis_expanded[c]

    def _scenarios_list(sc_json: Any) -> List[Dict[str, Any]]:
        s = _safe_json_loads(sc_json, [])
        if isinstance(s, list):
            out = []
            for it in s:
                if isinstance(it, dict):
                    out.append(it)
            return out
        return []

    df["scenarios"] = df["scenarios_json"].apply(_scenarios_list)

    def _warning_list(w_json: Any) -> List[str]:
        return _string_list(_safe_json_loads(w_json, []))

    def _dict_json(v: Any) -> Dict[str, Any]:
        parsed = _safe_json_loads(v, {})
        return parsed if isinstance(parsed, dict) else {}

    df["warnings"] = df["warnings_json"].apply(_warning_list)
    df["provenance"] = df["provenance_json"].apply(_dict_json)
    df["execution_metadata"] = df["execution_metadata_json"].apply(_dict_json)
    df["run_metadata"] = df["run_metadata_json"].apply(_dict_json)

    df = normalize_df(df, fill_dashboard_fallbacks=False)
    return df


def apply_filters(df: pd.DataFrame, stances: List[str], severities: List[str], query: str) -> pd.DataFrame:
    out = df.copy()
    if stances:
        out = out[out["stance"].isin(stances)]
    if severities:
        out = out[out["severity"].isin(severities)]
    if query.strip():
        q = query.strip().lower()
        out = out[out["campaign_id"].astype("string").str.lower().str.contains(q)]
    return out


# -----------------------------
# Visuals (more obvious)
# -----------------------------
def make_scatter_ratio(df: pd.DataFrame) -> go.Figure:
    """
    Easy-to-read plot:
      X = CPA / target_CPA (1.0 is on target)
      Y = ROAS (1.0 is break-even pre-LTV)
    """
    fig = px.scatter(
        df,
        x="cpa_ratio",
        y="roas",
        color="severity",
        hover_data={
            "campaign_id": True,
            "run_id": True,
            "stance": True,
            "severity": True,
            "cpa": ":.2f",
            "target_cpa": ":.2f",
            "cpa_ratio": ":.2f",
            "roas": ":.2f",
            "cpa_trend_7d": ":.2f",
            "roas_trend_7d": ":.2f",
            "days_active": True,
            "decision_explanation": True,
            "warning_count": True,
        },
    )

    fig.add_vline(x=1.0, line_width=1, opacity=0.6)
    fig.add_hline(y=1.0, line_width=1, opacity=0.6)

    fig.update_layout(
        height=420,
        margin=dict(l=10, r=10, t=60, b=10),
        legend_title_text="Priority",
        xaxis_title="Efficiency vs target CPA (1.0 = on target)",
        yaxis_title="ROAS (1.0 = break-even before lifetime value assumptions)",
    )
    return fig


def scenarios_to_df(scenarios: List[Dict[str, Any]]) -> pd.DataFrame:
    if not scenarios:
        return pd.DataFrame(columns=["scenario_name", "budget_multiplier", "projected_CPA", "projected_ROAS", "notes"])
    df = pd.DataFrame(scenarios).copy()
    if "projected_CPA" not in df.columns and "projected_cpa" in df.columns:
        df["projected_CPA"] = df["projected_cpa"]
    if "projected_ROAS" not in df.columns and "projected_roas" in df.columns:
        df["projected_ROAS"] = df["projected_roas"]

    for c in ["budget_multiplier", "projected_CPA", "projected_ROAS"]:
        if c in df.columns:
            df[c] = df[c].apply(_safe_float)

    if "scenario_name" not in df.columns:
        df["scenario_name"] = pd.NA
    if "notes" not in df.columns:
        df["notes"] = ""

    df = df[["scenario_name", "budget_multiplier", "projected_CPA", "projected_ROAS", "notes"]]
    df = df.sort_values("budget_multiplier", ascending=True)
    return df


def make_what_if_charts(df_whatif: pd.DataFrame) -> Tuple[go.Figure, go.Figure]:
    fig_roas = px.line(df_whatif, x="budget_multiplier", y="projected_ROAS", markers=True)
    fig_roas.update_layout(
        height=260,
        margin=dict(l=10, r=10, t=30, b=10),
        xaxis_title="Budget change multiplier",
        yaxis_title="Illustrative ROAS",
    )
    fig_roas.add_hline(y=1.0, line_width=1, opacity=0.6)

    fig_cpa = px.line(df_whatif, x="budget_multiplier", y="projected_CPA", markers=True)
    fig_cpa.update_layout(
        height=260,
        margin=dict(l=10, r=10, t=30, b=10),
        xaxis_title="Budget change multiplier",
        yaxis_title="Illustrative CPA",
    )
    return fig_roas, fig_cpa


def diff_vs_previous(current: pd.Series, prev: pd.Series) -> Dict[str, float]:
    keys = ["cpa", "roas", "cpa_trend_7d", "roas_trend_7d", "risk_score"]
    out = {}
    for k in keys:
        out[f"delta_{k}"] = _safe_float(current.get(k)) - _safe_float(prev.get(k))
    return out


# -----------------------------
# UI
# -----------------------------
st.markdown(
    """
    <style>
    .hero-card, .summary-card, .queue-card {
        border: 1px solid rgba(128, 128, 128, 0.25);
        border-radius: 12px;
        padding: 0.9rem 1rem;
        background: rgba(250, 250, 250, 0.02);
        margin-bottom: 0.75rem;
    }
    .hero-eyebrow {
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        opacity: 0.75;
        margin-bottom: 0.4rem;
    }
    .hero-title {
        font-size: 1.1rem;
        font-weight: 600;
        margin-bottom: 0.35rem;
    }
    .hero-text, .summary-text, .queue-text {
        font-size: 0.95rem;
        line-height: 1.45;
    }
    .summary-label {
        font-size: 0.76rem;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        opacity: 0.75;
        margin-bottom: 0.35rem;
    }
    .summary-value {
        font-size: 1rem;
        font-weight: 600;
        line-height: 1.35;
    }
    .badge {
        display: inline-block;
        padding: 0.18rem 0.55rem;
        border-radius: 999px;
        font-size: 0.76rem;
        font-weight: 600;
        margin-right: 0.4rem;
        border: 1px solid transparent;
    }
    .decision-escalate, .priority-high {
        background: rgba(220, 38, 38, 0.12);
        color: #ffb4b4;
        border-color: rgba(220, 38, 38, 0.35);
    }
    .decision-review, .priority-medium {
        background: rgba(245, 158, 11, 0.12);
        color: #ffd58a;
        border-color: rgba(245, 158, 11, 0.35);
    }
    .decision-monitor, .priority-low {
        background: rgba(34, 197, 94, 0.12);
        color: #a7f3c1;
        border-color: rgba(34, 197, 94, 0.35);
    }
    .priority-neutral {
        background: rgba(148, 163, 184, 0.12);
        color: #cbd5e1;
        border-color: rgba(148, 163, 184, 0.35);
    }
    .compact-list {
        margin: 0.25rem 0 0 0;
        padding-left: 1.1rem;
    }
    .compact-list li {
        margin-bottom: 0.25rem;
    }
    .queue-header {
        font-size: 0.78rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        opacity: 0.7;
        margin-bottom: 0.15rem;
    }
    .queue-row {
        padding: 0.45rem 0;
        border-bottom: 1px solid rgba(128, 128, 128, 0.18);
    }
    .mini-stat-card {
        border: 1px solid rgba(128, 128, 128, 0.22);
        border-radius: 10px;
        padding: 0.7rem 0.85rem;
        background: rgba(250, 250, 250, 0.02);
        margin-bottom: 0.6rem;
    }
    .mini-stat-value {
        font-size: 1.05rem;
        font-weight: 700;
        line-height: 1.2;
    }
    .decision-panel {
        border: 1px solid rgba(128, 128, 128, 0.22);
        border-radius: 14px;
        padding: 0.85rem 1rem;
        background: rgba(250, 250, 250, 0.02);
        margin-bottom: 0.75rem;
    }
    .queue-item {
        border: 1px solid rgba(128, 128, 128, 0.16);
        border-radius: 12px;
        padding: 0.65rem 0.75rem;
        background: rgba(250, 250, 250, 0.015);
        margin-bottom: 0.55rem;
    }
    .queue-item-active {
        border-color: rgba(245, 158, 11, 0.4);
        background: rgba(245, 158, 11, 0.06);
    }
    .queue-campaign {
        font-size: 0.95rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }
    .queue-reason {
        font-size: 0.9rem;
        line-height: 1.35;
        margin-top: 0.35rem;
        margin-bottom: 0.25rem;
    }
    .queue-evidence {
        font-size: 0.82rem;
        opacity: 0.82;
        line-height: 1.3;
    }
    .brief-highlight {
        border-left: 3px solid rgba(245, 158, 11, 0.65);
        padding-left: 0.8rem;
        margin-bottom: 0.75rem;
    }
    .metric-row-label {
        font-size: 0.74rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        opacity: 0.72;
        margin-bottom: 0.15rem;
    }
    .metric-row-value {
        font-size: 1rem;
        font-weight: 600;
        line-height: 1.2;
    }
    .top-strip {
        border: 1px solid rgba(128, 128, 128, 0.18);
        border-radius: 14px;
        padding: 0.8rem 0.95rem;
        background: linear-gradient(180deg, rgba(250, 250, 250, 0.03), rgba(250, 250, 250, 0.015));
        margin-bottom: 0.85rem;
    }
    .top-strip-title {
        font-size: 1.08rem;
        font-weight: 700;
        line-height: 1.25;
        margin-bottom: 0.2rem;
    }
    .top-strip-text {
        font-size: 0.92rem;
        line-height: 1.35;
        opacity: 0.9;
    }
    .hero-stat {
        border-left: 1px solid rgba(128, 128, 128, 0.18);
        padding-left: 0.9rem;
        height: 100%;
    }
    .hero-stat-value {
        font-size: 1rem;
        font-weight: 700;
        line-height: 1.25;
    }
    .queue-shell {
        border: 1px solid rgba(128, 128, 128, 0.18);
        border-radius: 14px;
        padding: 0.75rem 0.8rem;
        background: rgba(250, 250, 250, 0.015);
    }
    .hero-kicker {
        font-size: 0.76rem;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        opacity: 0.72;
        margin-bottom: 0.35rem;
    }
    .hero-main {
        font-size: 1.08rem;
        font-weight: 700;
        line-height: 1.28;
        margin-bottom: 0.28rem;
    }
    .hero-sub {
        font-size: 0.92rem;
        line-height: 1.35;
        opacity: 0.9;
    }
    .hero-inline-metric {
        font-size: 0.9rem;
        line-height: 1.35;
        margin-top: 0.4rem;
    }
    .hero-band {
        border: 1px solid rgba(128, 128, 128, 0.18);
        border-radius: 16px;
        padding: 0.95rem 1.05rem;
        background: linear-gradient(180deg, rgba(250, 250, 250, 0.03), rgba(250, 250, 250, 0.015));
        margin-bottom: 1rem;
    }
    .hero-band-main {
        font-size: 1.08rem;
        font-weight: 700;
        line-height: 1.28;
        margin-bottom: 0.28rem;
    }
    .hero-band-sub {
        font-size: 0.92rem;
        line-height: 1.38;
        opacity: 0.9;
    }
    .hero-band-focus {
        margin-top: 0.85rem;
        padding-top: 0.85rem;
        border-top: 1px solid rgba(128, 128, 128, 0.14);
        display: grid;
        grid-template-columns: 1.2fr 1fr;
        gap: 1rem;
    }
    .hero-focus-title {
        font-size: 0.76rem;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        opacity: 0.68;
        margin-bottom: 0.2rem;
    }
    .hero-focus-value {
        font-size: 1rem;
        font-weight: 700;
        line-height: 1.3;
    }
    .hero-focus-support {
        font-size: 0.88rem;
        line-height: 1.36;
        opacity: 0.84;
        margin-top: 0.2rem;
    }
    .decision-console {
        border: 1px solid rgba(128, 128, 128, 0.18);
        border-radius: 18px;
        padding: 1.15rem 1.2rem;
        background: linear-gradient(180deg, rgba(250, 250, 250, 0.03), rgba(250, 250, 250, 0.012));
        margin-bottom: 0.95rem;
    }
    .decision-console-kicker {
        font-size: 0.76rem;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        opacity: 0.68;
        margin-bottom: 0.35rem;
    }
    .decision-console-campaign {
        font-size: 1.45rem;
        font-weight: 700;
        line-height: 1.2;
        margin-bottom: 0.35rem;
    }
    .decision-console-why {
        font-size: 1.02rem;
        line-height: 1.45;
        font-weight: 600;
        margin-top: 0.7rem;
        margin-bottom: 0.9rem;
    }
    .decision-console-next {
        font-size: 0.96rem;
        line-height: 1.4;
    }
    .decision-console-side {
        padding-left: 1rem;
        border-left: 1px solid rgba(128, 128, 128, 0.14);
        height: 100%;
    }
    .decision-console-metric {
        margin-bottom: 0.85rem;
    }
    .selector-strip {
        margin-top: 0.25rem;
        margin-bottom: 1rem;
    }
    .selector-strip-title {
        font-size: 0.78rem;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        opacity: 0.68;
        margin-bottom: 0.45rem;
    }
    .selector-caption {
        font-size: 0.85rem;
        line-height: 1.3;
        opacity: 0.8;
        margin-top: 0.25rem;
    }
    .brief-card {
        border: 1px solid rgba(128, 128, 128, 0.18);
        border-radius: 14px;
        padding: 1.05rem 1.1rem;
        background: rgba(250, 250, 250, 0.02);
        margin-bottom: 0.95rem;
    }
    .brief-section-title {
        font-size: 0.78rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        opacity: 0.72;
        margin-bottom: 0.35rem;
    }
    .brief-action {
        font-size: 1.02rem;
        font-weight: 600;
        line-height: 1.35;
    }
    .quiet-label {
        font-size: 0.72rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        opacity: 0.65;
        margin-bottom: 0.2rem;
    }
    .brief-metric-strip {
        padding: 0.2rem 0 0.8rem 0;
        border-bottom: 1px solid rgba(128, 128, 128, 0.14);
        margin-bottom: 1rem;
    }
    .brief-secondary-card {
        border: 1px solid rgba(128, 128, 128, 0.14);
        border-radius: 12px;
        padding: 0.9rem 0.95rem;
        background: rgba(250, 250, 250, 0.015);
        margin-bottom: 0.8rem;
    }
    .brief-evidence-list {
        margin: 0.2rem 0 0 0;
        padding-left: 1rem;
    }
    .brief-evidence-list li {
        margin-bottom: 0.35rem;
        line-height: 1.35;
    }
    .brief-primary-text {
        font-size: 1.02rem;
        line-height: 1.45;
        font-weight: 600;
        margin-bottom: 0.8rem;
    }
    .brief-section-block {
        margin-top: 0.95rem;
        padding-top: 0.95rem;
        border-top: 1px solid rgba(128, 128, 128, 0.12);
    }
    .brief-muted {
        font-size: 0.88rem;
        line-height: 1.4;
        opacity: 0.82;
    }
    .secondary-heading {
        font-size: 0.95rem;
        font-weight: 600;
        margin-bottom: 0.35rem;
    }
    .executive-hero {
        border: 1px solid rgba(128, 128, 128, 0.18);
        border-radius: 18px;
        padding: 1.05rem 1.15rem;
        background: linear-gradient(180deg, rgba(250, 250, 250, 0.03), rgba(250, 250, 250, 0.012));
        margin-bottom: 0.95rem;
    }
    .executive-hero-top {
        display: flex;
        justify-content: space-between;
        align-items: flex-start;
        gap: 1rem;
        margin-bottom: 0.55rem;
    }
    .executive-hero-label {
        font-size: 0.76rem;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        opacity: 0.68;
        margin-bottom: 0.25rem;
    }
    .executive-hero-campaign {
        font-size: 1.14rem;
        font-weight: 700;
        line-height: 1.24;
    }
    .executive-hero-problem {
        font-size: 1.08rem;
        line-height: 1.44;
        font-weight: 600;
        max-width: 54rem;
        margin-bottom: 0.9rem;
    }
    .executive-hero-grid {
        display: grid;
        grid-template-columns: 1.25fr 0.95fr;
        gap: 1rem;
        align-items: start;
    }
    .executive-hero-panel {
        border-top: 1px solid rgba(128, 128, 128, 0.12);
        padding-top: 0.8rem;
    }
    .executive-hero-action {
        font-size: 0.98rem;
        line-height: 1.42;
        font-weight: 600;
    }
    .executive-hero-evidence {
        margin: 0.2rem 0 0 0;
        padding-left: 1rem;
    }
    .executive-hero-evidence li {
        margin-bottom: 0.32rem;
        line-height: 1.34;
    }
    .executive-hero-metrics {
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 0.85rem;
        margin-top: 0.75rem;
    }
    .campaign-switcher {
        margin-bottom: 1rem;
    }
    .campaign-switcher-title {
        font-size: 0.78rem;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        opacity: 0.68;
        margin-bottom: 0.35rem;
    }
    .campaign-switcher-caption {
        font-size: 0.85rem;
        line-height: 1.3;
        opacity: 0.78;
        margin-bottom: 0.65rem;
    }
    .hero-console-panel {
        border: 1px solid rgba(128, 128, 128, 0.18);
        border-radius: 18px;
        padding: 0.9rem 1rem;
        background: linear-gradient(180deg, rgba(250, 250, 250, 0.03), rgba(250, 250, 250, 0.012));
        margin-bottom: 0.75rem;
        height: 100%;
    }
    .hero-console-primary {
        padding-bottom: 0.82rem;
    }
    .hero-console-kicker {
        font-size: 0.76rem;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        opacity: 0.68;
        margin-bottom: 0.28rem;
    }
    .hero-console-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 0.9rem;
        margin-bottom: 0.5rem;
    }
    .hero-console-campaign {
        font-size: 1.14rem;
        font-weight: 700;
        line-height: 1.2;
    }
    .hero-console-badges {
        display: flex;
        flex-wrap: wrap;
        justify-content: flex-end;
        gap: 0.35rem;
    }
    .hero-console-problem {
        font-size: 1.16rem;
        line-height: 1.38;
        font-weight: 600;
        margin: 0;
    }
    .hero-console-section-title {
        font-size: 0.76rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        opacity: 0.68;
        margin-bottom: 0.4rem;
    }
    .hero-console-action {
        font-size: 1.02rem;
        line-height: 1.4;
        font-weight: 600;
        margin-bottom: 0.65rem;
    }
    .hero-console-evidence {
        margin: 0.1rem 0 0 0;
        padding-left: 1rem;
    }
    .hero-console-evidence li {
        margin-bottom: 0.3rem;
        line-height: 1.33;
    }
    .hero-console-switcher-copy {
        font-size: 0.85rem;
        line-height: 1.32;
        opacity: 0.8;
        margin-bottom: 0.55rem;
    }
    .hero-decision-panel {
        border: 1px solid rgba(128, 128, 128, 0.18);
        border-radius: 18px;
        padding: 0.95rem 1.05rem;
        background: linear-gradient(180deg, rgba(250, 250, 250, 0.03), rgba(250, 250, 250, 0.012));
        min-height: 100%;
    }
    .hero-decision-top {
        display: flex;
        justify-content: space-between;
        align-items: flex-start;
        gap: 0.9rem;
        margin-bottom: 0.55rem;
    }
    .hero-decision-campaign {
        font-size: 1.2rem;
        font-weight: 700;
        line-height: 1.2;
    }
    .hero-diagnosis {
        font-size: 1.14rem;
        line-height: 1.4;
        font-weight: 600;
        margin-bottom: 0.85rem;
    }
    .hero-next-step {
        padding-top: 0.72rem;
        margin-top: 0.72rem;
        border-top: 1px solid rgba(128, 128, 128, 0.12);
    }
    .hero-next-step-label {
        font-size: 0.76rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        opacity: 0.68;
        margin-bottom: 0.3rem;
    }
    .hero-next-step-text {
        font-size: 1.02rem;
        line-height: 1.4;
        font-weight: 600;
    }
    .hero-why-list {
        margin: 0.55rem 0 0 0;
        padding-left: 1rem;
    }
    .hero-why-list li {
        margin-bottom: 0.28rem;
        line-height: 1.34;
    }
    .hero-visual-title {
        font-size: 0.76rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        opacity: 0.68;
        margin-bottom: 0.3rem;
    }
    .hero-visual-caption {
        font-size: 0.84rem;
        line-height: 1.3;
        opacity: 0.78;
        margin-bottom: 0.3rem;
    }
    .hero-impact-shell {
        min-height: 214px;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        padding: 0.55rem 1.8rem 0.85rem 1.8rem;
    }
    .hero-impact-title {
        font-size: 0.76rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        opacity: 0.68;
        text-align: left;
        margin-bottom: 0.35rem;
    }
    .hero-impact-hero {
        flex: 1.12;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        text-align: center;
        min-height: 126px;
        padding: 0.6rem 0 0.85rem 0;
    }
    .hero-impact-value {
        font-size: 3.38rem;
        font-weight: 700;
        line-height: 0.9;
        margin-bottom: 0.08rem;
        letter-spacing: -0.03em;
    }
    .hero-impact-value.alert {
        color: #fb7185;
    }
    .hero-impact-value.good {
        color: #34d399;
    }
    .hero-impact-value.neutral {
        color: rgba(248, 250, 252, 0.78);
    }
    .hero-impact-label {
        font-size: 0.88rem;
        line-height: 1.08;
        opacity: 0.78;
    }
    .hero-impact-metrics {
        display: grid;
        grid-template-columns: 1.05fr 1.05fr 0.78fr;
        gap: 0.62rem;
        align-items: end;
        width: 72%;
        margin: 0 auto;
        padding-top: 0.82rem;
        padding-left: 0.32rem;
        padding-right: 0.32rem;
        border-top: 1px solid rgba(128, 128, 128, 0.12);
    }
    .hero-impact-metric {
        min-width: 0;
        display: flex;
        flex-direction: column;
        justify-content: flex-end;
    }
    .hero-impact-metric-label {
        font-size: 0.69rem;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        opacity: 0.56;
        margin-bottom: 0.12rem;
        line-height: 1.15;
    }
    .hero-impact-metric-value {
        font-size: 0.96rem;
        font-weight: 550;
        line-height: 1.08;
    }
    .hero-impact-metric-value.tertiary {
        font-size: 0.78rem;
        font-weight: 500;
        opacity: 0.6;
        padding-bottom: 0.02rem;
    }
    .hero-selector-shell {
        padding: 0.12rem 0 0 0;
        background: transparent;
        border: 0;
        border-radius: 0;
        margin: 0.56rem 0 0.18rem 0;
    }
    .hero-selector-shell .hero-console-section-title {
        margin-bottom: 0.14rem;
        opacity: 0.66;
        font-weight: 600;
        letter-spacing: 0.06em;
    }
    .hero-selector-shell .hero-console-switcher-copy {
        font-size: 0.82rem;
        line-height: 1.28;
        opacity: 0.64;
        margin-bottom: 0.12rem;
        max-width: 36rem;
    }
    .main .block-container [data-testid="stRadio"] {
        margin-top: 0.08rem;
        margin-bottom: 0.72rem;
    }
    .main .block-container [data-testid="stRadio"] > div {
        gap: 0.08rem;
    }
    .main .block-container [data-testid="stRadio"] div[role="radiogroup"] {
        display: flex;
        flex-wrap: nowrap;
        gap: 1.1rem;
        overflow-x: auto;
        align-items: flex-end;
        padding: 0.02rem 0 0.18rem 0;
        border-bottom: 1px solid rgba(128, 128, 128, 0.12);
        scrollbar-width: none;
    }
    .main .block-container [data-testid="stRadio"] div[role="radiogroup"]::-webkit-scrollbar {
        display: none;
    }
    .main .block-container [data-testid="stRadio"] label[data-baseweb="radio"] {
        margin: 0;
        padding: 0.02rem 0 0.46rem 0;
        min-height: auto !important;
        background: transparent !important;
        border: 0 !important;
        border-bottom: 2px solid transparent !important;
        border-radius: 0 !important;
        box-shadow: none !important;
        flex: 0 0 auto;
    }
    .main .block-container [data-testid="stRadio"] label[data-baseweb="radio"] > div:first-child {
        display: none;
    }
    .main .block-container [data-testid="stRadio"] label[data-baseweb="radio"] p {
        margin: 0;
        font-size: 0.86rem;
        line-height: 1.18;
        color: rgba(248, 250, 252, 0.6);
        white-space: nowrap;
        transition: color 120ms ease, opacity 120ms ease, transform 120ms ease;
    }
    .main .block-container [data-testid="stRadio"] label[data-baseweb="radio"]:hover p {
        color: rgba(248, 250, 252, 0.84);
    }
    .main .block-container [data-testid="stRadio"] label[data-baseweb="radio"]:has(input:checked) {
        border-bottom-color: rgba(248, 250, 252, 0.9) !important;
    }
    .main .block-container [data-testid="stRadio"] label[data-baseweb="radio"]:has(input:checked) p {
        color: rgba(248, 250, 252, 0.96);
        font-weight: 600;
        transform: translateY(-1px);
    }
    .snapshot-grid {
        display: grid;
        grid-template-columns: repeat(2, minmax(0, 1fr));
        gap: 0.75rem 1rem;
        margin-top: 0.15rem;
    }
    .snapshot-item {
        padding-bottom: 0.08rem;
    }
    .snapshot-label {
        font-size: 0.72rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        opacity: 0.66;
        margin-bottom: 0.12rem;
    }
    .snapshot-value {
        font-size: 0.96rem;
        font-weight: 600;
        line-height: 1.24;
    }
    .section-intro {
        font-size: 0.9rem;
        line-height: 1.4;
        opacity: 0.82;
        margin-bottom: 0.65rem;
    }
    .context-grid {
        display: grid;
        grid-template-columns: repeat(2, minmax(0, 1fr));
        gap: 0.75rem 1rem;
        margin-top: 0.35rem;
    }
    .appendix-card {
        border: 1px solid rgba(128, 128, 128, 0.12);
        border-radius: 12px;
        padding: 0.8rem 0.9rem;
        background: rgba(250, 250, 250, 0.012);
        margin-bottom: 0.8rem;
    }
    .stButton > button {
        border-radius: 12px;
        padding: 0.72rem 0.82rem;
        white-space: pre-wrap;
        text-align: left;
        line-height: 1.28;
        min-height: 84px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("📊 Marketing Decision Review")
st.caption(
    "Portfolio demo: a decision-support review surface that turns campaign diagnostics into prioritized business actions."
)

# Sidebar
st.sidebar.header("Review controls")
developer_mode = st.sidebar.toggle("Show technical details", value=False)

if developer_mode:
    with st.expander("Technical paths", expanded=False):
        st.write(f"APP_DIR: {PATHS['APP_DIR']}")
        st.write(f"PROJECT_DIR: {PATHS['PROJECT_DIR']}")
        st.write(f"CWD: {PATHS['CWD']}")
        st.write(f"DATA_DIR: {PATHS['DATA_DIR']}")
        st.write(f"RUNS_DIR: {PATHS['RUNS_DIR']} (exists={PATHS['RUNS_DIR'].exists()})")
        st.write(f"DB_PATH: {PATHS['DB_PATH']} (exists={PATHS['DB_PATH'].exists()})")

# Data source
runs_dir = PATHS["RUNS_DIR"]
data_dir = PATHS["DATA_DIR"]
db_path = PATHS["DB_PATH"]

db_runs_df = list_db_runs(db_path)
runs_folder = list_runs_folder(runs_dir)
local_csvs = list_local_csvs(data_dir)

source_options = ["Saved run", "Run folder", "Local CSV", "Upload CSV"]
default_index = 0 if not db_runs_df.empty else (1 if runs_folder else (2 if local_csvs else 3))
source_mode = st.sidebar.radio("Source data", options=source_options, index=default_index)

uploaded = None
if source_mode == "Upload CSV":
    uploaded = st.sidebar.file_uploader("Upload a campaigns CSV", type=["csv"])

run_meta = ""
df_run = None
selected_run_id = None
run_details_text = ""

if source_mode == "Saved run":
    if db_runs_df.empty:
        st.warning(f"No saved runs found in: {db_path}")
        st.stop()

    chosen_label = st.sidebar.selectbox("Select a saved run", options=db_runs_df["label"].tolist(), index=0)
    selected_run_id = str(db_runs_df.loc[db_runs_df["label"] == chosen_label, "run_id"].iloc[0])
    df_run = load_db_run_outputs(db_path, selected_run_id)
    if df_run.empty:
        st.warning(f"The saved run exists but has no campaign outputs for run_id={selected_run_id}")
        st.stop()
    run_meta = f"Saved run: {chosen_label}"

elif source_mode == "Upload CSV":
    if uploaded is None:
        st.info("Upload a CSV to continue.")
        st.stop()
    df_run = normalize_df(pd.read_csv(uploaded))
    run_meta = f"Uploaded file: {uploaded.name}"

elif source_mode == "Local CSV":
    if not local_csvs:
        st.warning(f"No CSV files found in: {data_dir}")
        st.stop()
    csv_labels = [p.name for p in local_csvs]
    chosen_label = st.sidebar.selectbox("Select a local CSV", options=csv_labels, index=0)
    chosen_path = next(p for p in local_csvs if p.name == chosen_label)
    df_run = load_csv_path(chosen_path)
    run_meta = f"Local file: {chosen_path.name}"

else:  # Runs folder
    if not runs_folder:
        st.warning(
            f"No runs found in: {runs_dir}\n\n"
            "Fix options:\n"
            "1) Put runs under that folder (e.g. data/runs/<run_id>/campaigns.csv)\n"
            "2) Use 'Local CSV'\n"
            "3) Upload a CSV\n"
            "4) Use 'Saved run' if available"
        )
        st.stop()

    chosen_run = st.sidebar.selectbox("Select a run", options=runs_folder, index=0)
    run_dir = runs_dir / chosen_run
    df_run = load_run_csv(run_dir)
    if df_run is None:
        st.error(f"Run found but no CSV inside: {run_dir}")
        st.stop()
    run_meta = f"Run folder: {chosen_run}"

if source_mode == "Saved run" and not df_run.empty:
    run_row = df_run.iloc[0]
    run_bits = []
    for label, key in [
        ("started", "run_started_at_utc"),
        ("input file", "run_input_csv"),
        ("model", "run_model"),
        ("ai assist", "run_used_llm"),
    ]:
        value = _clean_text(run_row.get(key))
        if value:
            run_bits.append(f"{label}={value}")
    notes_text = _clean_text(run_row.get("run_notes"))
    if notes_text:
        run_bits.append(f"notes={notes_text}")
    run_details_text = " | ".join(run_bits)

if not run_details_text:
    run_details_text = run_meta

# Filters
all_stances = sorted([s for s in df_run["stance"].dropna().unique() if str(s).strip()])
all_severities = sorted(
    [s for s in df_run["severity"].dropna().unique() if str(s).strip()],
    key=lambda x: -SEVERITY_ORDER.get(str(x), 0),
)

stance_filter = st.sidebar.multiselect("Filter recommendation", options=all_stances, default=[])
severity_filter = st.sidebar.multiselect("Filter priority", options=all_severities, default=[])
search_id = st.sidebar.text_input("Search campaign", value="")

df_f = apply_filters(df_run, stance_filter, severity_filter, search_id)

portfolio_signal = _portfolio_signal_text(df_f)
top_issue = _top_portfolio_issue(sort_campaigns_for_review(df_f).head(5)) if not df_f.empty else ""
action_count = int(df_f["stance"].isin(["escalate", "recommend", "adjust"]).sum()) if not df_f.empty else 0
campaigns_count = len(df_f)
high_count = int((df_f["severity"] == "high").sum()) if campaigns_count else 0
escalate_count = int((df_f["stance"] == "escalate").sum()) if campaigns_count else 0
roas_lt_1 = int((df_f["roas"] < 1.0).sum()) if campaigns_count else 0
avg_cpa_ratio = float(np.nanmean(df_f["cpa_ratio"])) if campaigns_count else np.nan
sorted_df = sort_campaigns_for_review(df_f) if not df_f.empty else df_f.copy()
campaign_options = sorted_df["campaign_id"].tolist() if not sorted_df.empty else []
if campaign_options:
    default_campaign = st.session_state.get("selected_campaign", campaign_options[0])
    if default_campaign not in campaign_options:
        default_campaign = campaign_options[0]
    st.session_state["selected_campaign"] = default_campaign
    top_row = sorted_df.iloc[0]
    top_campaign = str(top_row.get("campaign_id"))
    top_action = _format_decision_label(top_row.get("stance"))
    top_priority = _format_risk_label(top_row.get("severity")).title()
    top_reason = _humanize_reason(_short_reason(top_row.get("decision_explanation"), max_len=110))
else:
    top_campaign = "None selected"
    top_action = "Monitor"
    top_priority = "Low"
    top_reason = "No campaigns match the current filters."

# -----------------------------
# Priority review queue
# -----------------------------
if df_f.empty:
    st.info("No campaigns match the current filters.")
    st.stop()

if "campaign_switcher" in st.session_state and st.session_state["campaign_switcher"] in campaign_options:
    st.session_state["selected_campaign"] = st.session_state["campaign_switcher"]

selected_campaign = st.session_state["selected_campaign"]
if selected_campaign not in campaign_options:
    selected_campaign = campaign_options[0]
    st.session_state["selected_campaign"] = selected_campaign

selected_campaign = st.session_state["selected_campaign"]
row = df_run[df_run["campaign_id"] == selected_campaign].iloc[0]

stance_val = row.get("stance")
severity_val = row.get("severity")
explanation = _clean_text(row.get("decision_explanation"))
explanation_human = _humanize_reason(explanation)
primary_action = row.get("advisor_actions", [])
action_text = "No recommended next steps are available for this campaign."
if isinstance(primary_action, list) and primary_action:
    action_text = primary_action[0]
reasons = row.get("reasons", [])
extra_reasons = []
if isinstance(reasons, list):
    seen_reason = explanation_human.strip().lower() if explanation_human else ""
    for reason in reasons:
        clean_reason = _humanize_reason(reason)
        if not clean_reason:
            continue
        if seen_reason and clean_reason.strip().lower() == seen_reason:
            continue
        extra_reasons.append(clean_reason)

warnings = row.get("warnings", [])
advisor_summary = row.get("advisor_summary", pd.NA)

context_items = []
advisor_confidence = row.get("advisor_confidence", pd.NA)
if pd.notna(advisor_confidence) and str(advisor_confidence).strip() and str(advisor_confidence).lower() != "nan":
    context_items.append(("Confidence", str(advisor_confidence)))
advisor_used_llm = row.get("advisor_used_llm", pd.NA)
if pd.notna(advisor_used_llm) and str(advisor_used_llm).strip() and str(advisor_used_llm).lower() != "nan":
    review_method = "AI-assisted review" if str(advisor_used_llm).strip().lower() in {"1", "true", "yes"} else "Rule-based review"
    context_items.append(("Review method", review_method))
advisor_model = row.get("advisor_model", pd.NA)
if pd.notna(advisor_model) and str(advisor_model).strip() and str(advisor_model).lower() != "nan":
    context_items.append(("Model", str(advisor_model)))

rationale_summary = ""
if pd.notna(advisor_summary) and str(advisor_summary).strip():
    rationale_summary = str(advisor_summary).strip()
elif extra_reasons:
    rationale_summary = extra_reasons[0]
elif warnings:
    rationale_summary = _humanize_reason(warnings[0])

hero_story_keys = set()
for value in [explanation_human, action_text]:
    key = _clean_text(value).strip().lower() if value else ""
    if key:
        hero_story_keys.add(key)

rationale_points = []
rationale_seen = set(hero_story_keys)
for candidate in extra_reasons[:4]:
    key = _clean_text(candidate).strip().lower() if candidate else ""
    if key and key not in rationale_seen:
        rationale_points.append(candidate)
        rationale_seen.add(key)

if not rationale_points and rationale_summary:
    summary_key = _clean_text(rationale_summary).strip().lower()
    if summary_key and summary_key not in rationale_seen:
        rationale_points.append(rationale_summary)
        rationale_seen.add(summary_key)

note_points = []
for warning in warnings[:3]:
    clean_warning = _humanize_reason(warning)
    key = _clean_text(clean_warning).strip().lower() if clean_warning else ""
    if key and key not in rationale_seen:
        note_points.append(clean_warning)
        rationale_seen.add(key)
diagnostic_points = []
diagnostic_seen = set()
for signal in _build_decision_signal_bits(row):
    key = _clean_text(signal).strip().lower()
    if key and key not in diagnostic_seen:
        diagnostic_points.append(signal)
        diagnostic_seen.add(key)

hero_why_points = diagnostic_points[:3]
support_points = []
support_seen = set(hero_story_keys)
for item in hero_why_points:
    key = _clean_text(item).strip().lower()
    if key:
        support_seen.add(key)

for candidate in diagnostic_points[3:] + rationale_points + note_points:
    key = _clean_text(candidate).strip().lower() if candidate else ""
    if key and key not in support_seen:
        support_points.append(candidate)
        support_seen.add(key)

action_options = _build_action_options(row, action_text)
hero_why_html = "".join(f"<li>{point}</li>" for point in hero_why_points)
hero_left, hero_right = st.columns([1.25, 0.95], gap="large")
with hero_left:
    st.markdown(
        f"""
        <div class="hero-decision-panel">
            <div class="hero-console-kicker">Executive decision surface</div>
            <div class="hero-decision-top">
                <div class="hero-decision-campaign">{selected_campaign}</div>
                <div class="hero-console-badges">
                    {_badge_html(_format_decision_label(stance_val), _decision_css_class(stance_val))}
                    {_badge_html(f"{_format_risk_label(severity_val).title()} priority", _priority_css_class(severity_val))}
                </div>
            </div>
            <div class="hero-diagnosis">{explanation_human or "No summary available."}</div>
            <div class="hero-next-step">
                <div class="hero-next-step-label">Recommended next move</div>
                <div class="hero-next-step-text">{action_text}</div>
            </div>
            <div class="hero-next-step">
                <div class="hero-next-step-label">Why this matters</div>
                <ul class="hero-why-list">{hero_why_html or "<li>No diagnostic detail available.</li>"}</ul>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with hero_right:
    impact_data = build_hero_impact_data(row)
    render_hero_impact_card(impact_data)
df_queue = sorted_df.head(4).copy()
switcher_options = []
switcher_labels = {}
for _, r in df_queue.iterrows():
    cid = str(r.get("campaign_id"))
    sev = str(r.get("severity", "—"))
    stance = str(r.get("stance", "—"))
    label = f"{cid} | {_format_decision_label(stance)} | {_format_risk_label(sev).title()}"
    switcher_options.append(cid)
    switcher_labels[cid] = label
if selected_campaign not in switcher_options:
    switcher_options.insert(0, selected_campaign)
    switcher_labels[selected_campaign] = (
        f"{selected_campaign} | {_format_decision_label(stance_val)} | {_format_risk_label(severity_val).title()}"
    )
st.markdown(
    """
    <div class="hero-selector-shell">
        <div class="hero-console-section-title">Review other campaigns</div>
        <div class="hero-console-switcher-copy">Shift focus without leaving the decision flow.</div>
    </div>
    """,
    unsafe_allow_html=True,
)
selected_campaign = st.radio(
    "Review other campaigns",
    options=switcher_options,
    index=switcher_options.index(selected_campaign),
    horizontal=True,
    format_func=lambda cid: switcher_labels[cid],
    label_visibility="collapsed",
    key="campaign_switcher",
)
st.session_state["selected_campaign"] = selected_campaign
row = df_run[df_run["campaign_id"] == selected_campaign].iloc[0]
stance_val = row.get("stance")
severity_val = row.get("severity")
explanation = _clean_text(row.get("decision_explanation"))
explanation_human = _humanize_reason(explanation)
primary_action = row.get("advisor_actions", [])
action_text = "No recommended next steps are available for this campaign."
if isinstance(primary_action, list) and primary_action:
    action_text = primary_action[0]
reasons = row.get("reasons", [])
extra_reasons = []
if isinstance(reasons, list):
    seen_reason = explanation_human.strip().lower() if explanation_human else ""
    for reason in reasons:
        clean_reason = _humanize_reason(reason)
        if not clean_reason:
            continue
        if seen_reason and clean_reason.strip().lower() == seen_reason:
            continue
        extra_reasons.append(clean_reason)
warnings = row.get("warnings", [])
advisor_summary = row.get("advisor_summary", pd.NA)
context_items = []
advisor_confidence = row.get("advisor_confidence", pd.NA)
if pd.notna(advisor_confidence) and str(advisor_confidence).strip() and str(advisor_confidence).lower() != "nan":
    context_items.append(("Confidence", str(advisor_confidence)))
advisor_used_llm = row.get("advisor_used_llm", pd.NA)
if pd.notna(advisor_used_llm) and str(advisor_used_llm).strip() and str(advisor_used_llm).lower() != "nan":
    review_method = "AI-assisted review" if str(advisor_used_llm).strip().lower() in {"1", "true", "yes"} else "Rule-based review"
    context_items.append(("Review method", review_method))
advisor_model = row.get("advisor_model", pd.NA)
if pd.notna(advisor_model) and str(advisor_model).strip() and str(advisor_model).lower() != "nan":
    context_items.append(("Model", str(advisor_model)))
rationale_summary = ""
if pd.notna(advisor_summary) and str(advisor_summary).strip():
    rationale_summary = str(advisor_summary).strip()
elif extra_reasons:
    rationale_summary = extra_reasons[0]
elif warnings:
    rationale_summary = _humanize_reason(warnings[0])

hero_story_keys = set()
for value in [explanation_human, action_text]:
    key = _clean_text(value).strip().lower() if value else ""
    if key:
        hero_story_keys.add(key)

rationale_points = []
rationale_seen = set(hero_story_keys)
for candidate in extra_reasons[:4]:
    key = _clean_text(candidate).strip().lower() if candidate else ""
    if key and key not in rationale_seen:
        rationale_points.append(candidate)
        rationale_seen.add(key)

if not rationale_points and rationale_summary:
    summary_key = _clean_text(rationale_summary).strip().lower()
    if summary_key and summary_key not in rationale_seen:
        rationale_points.append(rationale_summary)
        rationale_seen.add(summary_key)

note_points = []
for warning in warnings[:3]:
    clean_warning = _humanize_reason(warning)
    key = _clean_text(clean_warning).strip().lower() if clean_warning else ""
    if key and key not in rationale_seen:
        note_points.append(clean_warning)
        rationale_seen.add(key)

diagnostic_points = []
diagnostic_seen = set()
for signal in _build_decision_signal_bits(row):
    key = _clean_text(signal).strip().lower()
    if key and key not in diagnostic_seen:
        diagnostic_points.append(signal)
        diagnostic_seen.add(key)

hero_why_points = diagnostic_points[:3]
support_points = []
support_seen = set(hero_story_keys)
for item in hero_why_points:
    key = _clean_text(item).strip().lower()
    if key:
        support_seen.add(key)

for candidate in diagnostic_points[3:] + rationale_points + note_points:
    key = _clean_text(candidate).strip().lower() if candidate else ""
    if key and key not in support_seen:
        support_points.append(candidate)
        support_seen.add(key)

action_options = _build_action_options(row, action_text)

st.divider()
has_support = bool(support_points or context_items)
if has_support:
    st.markdown("### Why this needs action")
    support_cols = st.columns([1.2, 1.0], gap="large")
    with support_cols[0]:
        if support_points:
            st.markdown("**Key signals**")
            for reason in support_points[:4]:
                st.write(f"- {reason}")
    with support_cols[1]:
        if context_items:
            st.markdown("**Review metadata**")
            context_html = "".join(
                f"<div class='snapshot-item'><div class='snapshot-label'>{label}</div><div class='snapshot-value'>{value}</div></div>"
                for label, value in context_items
            )
            st.markdown(f"<div class='context-grid'>{context_html}</div>", unsafe_allow_html=True)

if action_options:
    st.divider()
    st.markdown("### Actions to consider")
    for option in action_options[:4]:
        st.write(f"- {option}")

st.divider()
st.markdown("### Scenario explorer")
st.caption(
    "These scenarios are illustrative decision-support views based on simplified assumptions. They are directional, not predictive forecasts."
)
scenarios = row.get("scenarios", [])
df_whatif = scenarios_to_df(scenarios if isinstance(scenarios, list) else [])
if df_whatif.empty:
    st.caption("No illustrative scenarios were stored for this campaign.")
else:
    df_whatif_display = df_whatif.rename(
        columns={
            "scenario_name": "Scenario",
            "budget_multiplier": "Budget change",
            "projected_CPA": "Illustrative CPA",
            "projected_ROAS": "Illustrative ROAS",
            "notes": "Notes",
        }
    )
    whatif_chart_col, whatif_table_col = st.columns([1.35, 1.0], gap="large")
    with whatif_chart_col:
        chart_left, chart_right = st.columns(2)
        fig_roas, fig_cpa = make_what_if_charts(df_whatif)
        with chart_left:
            st.plotly_chart(fig_roas, use_container_width=True)
        with chart_right:
            st.plotly_chart(fig_cpa, use_container_width=True)
    with whatif_table_col:
        st.dataframe(df_whatif_display, use_container_width=True, hide_index=True)

analysis_sum = row.get("analysis_summary", pd.NA)
ins = row.get("analysis_insights", [])
acts = row.get("analysis_suggested_actions", [])
appendix_seen = set(support_seen)
for item in hero_why_points + support_points + action_options:
    key = _clean_text(item).strip().lower() if item else ""
    if key:
        appendix_seen.add(key)
appendix_summary = ""
if pd.notna(analysis_sum) and str(analysis_sum).strip():
    summary_text = str(analysis_sum).strip()
    summary_key = _clean_text(summary_text).strip().lower()
    if summary_key and summary_key not in appendix_seen:
        appendix_summary = summary_text
        appendix_seen.add(summary_key)

appendix_insights = []
if isinstance(ins, list):
    for item in ins:
        clean_item = _humanize_reason(item)
        key = _clean_text(clean_item).strip().lower() if clean_item else ""
        if key and key not in appendix_seen:
            appendix_insights.append(clean_item)
            appendix_seen.add(key)

appendix_actions = []
if isinstance(acts, list):
    for item in acts:
        clean_item = str(item).strip()
        key = _clean_text(clean_item).strip().lower() if clean_item else ""
        if key and key not in appendix_seen:
            appendix_actions.append(clean_item)
            appendix_seen.add(key)

has_context = bool(
    appendix_summary
    or appendix_insights
    or appendix_actions
    or (isinstance(row.get("provenance", {}), dict) and row.get("provenance", {}))
    or (isinstance(row.get("execution_metadata", {}), dict) and row.get("execution_metadata", {}))
)
if has_context or developer_mode:
    with st.expander("Reference details", expanded=False):
        st.caption("Technical appendix for deeper analytical backup and execution context.")
        if appendix_summary:
            st.markdown("**Stored analysis note**")
            st.markdown(f"<div class='appendix-card'>{appendix_summary}</div>", unsafe_allow_html=True)

        if appendix_insights:
            st.markdown("**Analyst notes**")
            for it in appendix_insights[:5]:
                st.write(f"- {it}")

        if appendix_actions:
            st.markdown("**Additional actions considered**")
            for it in appendix_actions[:5]:
                st.write(f"- {it}")

        provenance = row.get("provenance", {})
        execution_metadata = row.get("execution_metadata", {})
        if isinstance(provenance, dict) and provenance:
            st.markdown("**Data provenance**")
            st.json(provenance)
        if isinstance(execution_metadata, dict) and execution_metadata:
            st.markdown("**Execution details**")
            st.json(execution_metadata)
        if developer_mode:
            st.markdown("**Raw record**")
            st.code(json.dumps(row.to_dict(), indent=2, default=str), language="json")

with st.expander("Portfolio overview", expanded=False):
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Campaigns in review", f"{campaigns_count}")
    k2.metric("Escalate now", f"{escalate_count}")
    k3.metric("High priority", f"{high_count}")
    k4.metric("Below break-even", f"{roas_lt_1}")
    k5.metric("Avg efficiency vs target", f"{avg_cpa_ratio:.2f}" if campaigns_count else "—")
    st.caption("Portfolio scan for broader patterns, not campaign-level decision making.")
    with st.expander("Performance map", expanded=False):
        st.caption("Compare efficiency versus profitability across the filtered portfolio.")
        fig = make_scatter_ratio(df_f)
        st.plotly_chart(fig, use_container_width=True)

with st.expander("Campaign table", expanded=False):
    df_all = build_campaign_display_df(df_f)
    st.dataframe(df_all, use_container_width=True, hide_index=True)
    st.download_button(
        "⬇️ Download CSV (filtered)",
        data=df_all.to_csv(index=False).encode("utf-8"),
        file_name="campaigns_filtered.csv",
        mime="text/csv",
    )

with st.expander("Run details", expanded=False):
    st.write(run_details_text or "No run details available.")