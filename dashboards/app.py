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
presentation_mode = st.sidebar.toggle("Presentation mode", value=True)

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

st.caption("Reviewing: " + run_meta)

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
    if run_bits:
        st.caption("Review context: " + " | ".join(run_bits))

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

hero_left, hero_right = st.columns([2.5, 1.5])
with hero_left:
    st.markdown(
        f"""
        <div class="hero-card">
            <div class="hero-eyebrow">Phase 1 portfolio artifact</div>
            <div class="hero-title">Executive review surface for AI-assisted marketing decisions</div>
            <div class="hero-text">{portfolio_signal}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with hero_right:
    walkthrough_lines = [
        "Start with the priority queue to see what needs action.",
        "Open an executive brief to review rationale and next steps.",
        "Use illustrative scenarios to support discussion, not prediction.",
    ]
    walkthrough_html = "".join(f"<li>{line}</li>" for line in walkthrough_lines)
    st.markdown(
        f"""
        <div class="summary-card">
            <div class="summary-label">Suggested demo walkthrough</div>
            <div class="summary-text">
                <ol class="compact-list">{walkthrough_html}</ol>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# -----------------------------
# Priority review queue
# -----------------------------
st.subheader("Priority review queue")
st.caption("Start here: the queue below surfaces the campaigns most likely to need a decision or follow-up.")

if df_f.empty:
    st.info("No campaigns match the current filters.")
else:
    df_queue = sort_campaigns_for_review(df_f).head(5).copy()
    summary_bits = [f"{action_count} campaign(s) currently need attention"]
    if top_issue:
        summary_bits.append(f"Top issue: {top_issue}")
    st.write(" | ".join(summary_bits))

    # allow click-to-open via session_state
    if "selected_campaign" not in st.session_state:
        st.session_state["selected_campaign"] = df_queue["campaign_id"].iloc[0]

    for _, r in df_queue.iterrows():
        cid = str(r["campaign_id"])
        sev = str(r.get("severity", "—"))
        stance = str(r.get("stance", "—"))
        reason = _short_reason(r.get("decision_explanation"), max_len=150)
        advisor_actions = r.get("advisor_actions", [])
        decision_label = _format_decision_label(stance)
        priority_label = _format_risk_label(sev).title()
        reason = _humanize_reason(reason)
        evidence = _key_evidence_line(r)
        warnings = r.get("warnings", [])

        with st.container(border=True):
            top = st.columns([4, 1])
            with top[0]:
                st.markdown(f"**Campaign {cid}**")
                st.markdown(
                    _badge_html(decision_label, _decision_css_class(stance))
                    + _badge_html(f"{priority_label} priority", _priority_css_class(sev)),
                    unsafe_allow_html=True,
                )
            with top[1]:
                if st.button("Open brief", key=f"open_{cid}"):
                    st.session_state["selected_campaign"] = cid

            if reason:
                st.write(reason)
            if evidence:
                st.caption("Key evidence: " + evidence)
            if isinstance(advisor_actions, list) and advisor_actions:
                st.write(f"**Recommended next step:** {advisor_actions[0]}")
            if isinstance(warnings, list) and warnings:
                st.caption("Data note: " + _humanize_reason(warnings[0]))

# Tabs
tab_overview, tab_details, tab_all = st.tabs(["📌 Portfolio overview", "🧠 Executive brief", "📋 Campaign table"])

# -----------------------------
# Overview (simplified & clearer)
# -----------------------------
with tab_overview:
    st.subheader("At a glance")
    k1, k2, k3, k4, k5 = st.columns(5)

    campaigns_count = len(df_f)
    high_count = int((df_f["severity"] == "high").sum())
    escalate_count = int((df_f["stance"] == "escalate").sum())
    roas_lt_1 = int((df_f["roas"] < 1.0).sum())
    avg_cpa_ratio = float(np.nanmean(df_f["cpa_ratio"])) if campaigns_count else np.nan

    k1.metric("Campaigns in review", f"{campaigns_count}")
    k2.metric("Escalate now", f"{escalate_count}")
    k3.metric("High priority", f"{high_count}")
    k4.metric("Below break-even", f"{roas_lt_1}")
    k5.metric("Avg efficiency vs target", f"{avg_cpa_ratio:.2f}" if campaigns_count else "—")

    if campaigns_count:
        st.caption(
            f"{action_count} of {campaigns_count} campaigns need attention across the current filter set."
        )

    st.divider()

    s1, s2, s3 = st.columns(3)
    with s1:
        st.markdown(
            f"""
            <div class="summary-card">
                <div class="summary-label">Portfolio signal</div>
                <div class="summary-value">{portfolio_signal}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with s2:
        primary_focus = top_issue or "No major issue is currently surfaced."
        st.markdown(
            f"""
            <div class="summary-card">
                <div class="summary-label">Primary focus</div>
                <div class="summary-value">{primary_focus}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with s3:
        focus_area = "Move from queue to executive brief for the highest-priority campaign."
        if action_count == 0:
            focus_area = "Use the portfolio view to confirm there are no immediate intervention needs."
        st.markdown(
            f"""
            <div class="summary-card">
                <div class="summary-label">Recommended review path</div>
                <div class="summary-value">{focus_area}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.divider()

    st.subheader("Decision queue")
    st.caption("A compact view of the highest-priority items, translated into business-facing review language.")
    df_top = build_campaign_display_df(sort_campaigns_for_review(df_f).head(10))
    st.dataframe(df_top, use_container_width=True, hide_index=True)

    st.download_button(
        "⬇️ Download CSV (Decision queue)",
        data=df_top.to_csv(index=False).encode("utf-8"),
        file_name="decision_queue.csv",
        mime="text/csv",
    )

    st.divider()

    st.subheader("📈 Performance map")
    st.caption(
        "Right of 1.0 signals cost above target. Below 1.0 on the y-axis signals sub-break-even ROAS. Color shows priority."
    )

    fig = make_scatter_ratio(df_f)
    st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# Campaign details
# -----------------------------
with tab_details:
    if df_f.empty:
        st.warning("No campaigns match your filters.")
        st.stop()

    # Use the queue-selected campaign by default
    default_campaign = st.session_state.get("selected_campaign", df_f["campaign_id"].iloc[0])
    campaign_options = sort_campaigns_for_review(df_f)["campaign_id"].tolist()
    if default_campaign not in campaign_options:
        default_campaign = campaign_options[0]
    idx = campaign_options.index(default_campaign)

    selected_campaign = st.selectbox("Choose a campaign", options=campaign_options, index=idx)
    st.session_state["selected_campaign"] = selected_campaign

    row = df_run[df_run["campaign_id"] == selected_campaign].iloc[0]

    st.subheader(f"Executive brief: {selected_campaign}")
    d1, d2, d3, d4, d5 = st.columns(5)

    stance_val = row.get("stance")
    severity_val = row.get("severity")

    d1.metric("Recommended review", _format_decision_label(stance_val))
    d2.metric("Priority", _format_risk_label(severity_val).title())
    d3.metric("CPA", f"{row['cpa']:.2f}" if pd.notna(row.get("cpa")) else "—")
    d4.metric("Target CPA", f"{row['target_cpa']:.2f}" if pd.notna(row.get("target_cpa")) else "—")
    d5.metric("Efficiency vs target", f"{row['cpa_ratio']:.2f}" if pd.notna(row.get("cpa_ratio")) else "—")

    st.divider()

    st.markdown("### Recommendation summary")
    st.markdown(
        _badge_html(_format_decision_label(row.get("stance")), _decision_css_class(row.get("stance")))
        + _badge_html(
            f"{_format_risk_label(row.get('severity')).title()} priority",
            _priority_css_class(row.get("severity")),
        ),
        unsafe_allow_html=True,
    )
    st.write(
        f"**Decision posture:** {_format_decision_label(row.get('stance'))} | **Priority:** {_format_risk_label(row.get('severity')).title()}"
    )

    explanation = _clean_text(row.get("decision_explanation"))
    if explanation:
        st.write(f"**Why it matters:** {_humanize_reason(explanation)}")

    primary_action = row.get("advisor_actions", [])
    if isinstance(primary_action, list) and primary_action:
        st.write(f"**Recommended next step:** {primary_action[0]}")

    evidence_bits = _build_evidence_bits(row)
    if evidence_bits:
        st.write("**Supporting evidence:**")
        for bit in evidence_bits:
            st.write(f"- {bit}")

    reasons = row.get("reasons", [])
    st.write("**Why this was flagged:**")
    if isinstance(reasons, list) and reasons:
        for reason in reasons:
            st.write(f"- {_humanize_reason(reason)}")
    else:
        st.write("- —")

    warnings = row.get("warnings", [])
    if isinstance(warnings, list) and warnings:
        with st.expander("Data notes", expanded=False):
            for warning in warnings:
                st.write(f"- {_humanize_reason(warning)}")

    st.divider()

    st.markdown("### Recommended next steps")
    advisor_summary = row.get("advisor_summary", pd.NA)
    advisor_actions = row.get("advisor_actions", [])

    meta_bits = []
    meta_label_map = {
        "advisor_confidence": "confidence",
        "advisor_used_llm": "uses llm",
        "advisor_model": "model",
    }
    for k in ["advisor_confidence", "advisor_used_llm", "advisor_model"]:
        v = row.get(k, pd.NA)
        if pd.notna(v) and str(v).strip() and str(v).lower() != "nan":
            meta_bits.append(f"{meta_label_map[k]}={v}")
    if meta_bits:
        st.caption(" | ".join(meta_bits))

    if pd.notna(advisor_summary) and str(advisor_summary).strip():
        st.write(str(advisor_summary))
    else:
        st.caption("No additional advisory summary is available for this campaign.")

    if isinstance(advisor_actions, list) and advisor_actions:
        for i, a in enumerate(advisor_actions, 1):
            st.write(f"{i}. {a}")
    else:
        st.caption("No recommended next steps are available for this campaign.")

    if presentation_mode:
        st.info(
            "Demo cue: narrate the recommendation first, then the supporting evidence, then the illustrative scenarios."
        )

    st.divider()

    st.markdown("### Illustrative what-if scenarios")
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
        w1, w2 = st.columns(2)
        fig_roas, fig_cpa = make_what_if_charts(df_whatif)
        with w1:
            st.plotly_chart(fig_roas, use_container_width=True)
        with w2:
            st.plotly_chart(fig_cpa, use_container_width=True)
        st.dataframe(df_whatif_display, use_container_width=True, hide_index=True)

    st.divider()

    st.markdown("### Additional context")
    analysis_sum = row.get("analysis_summary", pd.NA)
    if pd.notna(analysis_sum) and str(analysis_sum).strip():
        st.write(str(analysis_sum))

    st.markdown("**Supporting insights**")
    ins = row.get("analysis_insights", [])
    if isinstance(ins, list) and ins:
        for it in ins:
            st.write(f"- {_humanize_reason(it)}")
    else:
        st.write("- —")

    st.markdown("**Additional suggested actions**")
    acts = row.get("analysis_suggested_actions", [])
    if isinstance(acts, list) and acts:
        for it in acts:
            st.write(f"- {it}")
    else:
        st.write("- —")

    provenance = row.get("provenance", {})
    execution_metadata = row.get("execution_metadata", {})
    if isinstance(provenance, dict) and provenance:
        with st.expander("Metric provenance", expanded=False):
            st.json(provenance)
    if isinstance(execution_metadata, dict) and execution_metadata:
        with st.expander("Technical metadata", expanded=False):
            st.json(execution_metadata)

    if developer_mode:
        with st.expander("Raw row JSON", expanded=False):
            st.code(json.dumps(row.to_dict(), indent=2, default=str), language="json")

# -----------------------------
# All campaigns
# -----------------------------
with tab_all:
    st.subheader("Campaign table")
    df_all = build_campaign_display_df(df_f)
    st.dataframe(df_all, use_container_width=True, hide_index=True)
    st.download_button(
        "⬇️ Download CSV (filtered)",
        data=df_all.to_csv(index=False).encode("utf-8"),
        file_name="campaigns_filtered.csv",
        mime="text/csv",
    )