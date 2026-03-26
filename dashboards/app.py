# dashboards/app.py
import json
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
    page_title="Marketing Agent — Run Dashboard",
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
            columns=["campaign_id", "stance", "risk", "explanation", "evidence", "warnings"]
        )

    rows: List[Dict[str, Any]] = []
    for _, row in sort_campaigns_for_review(df).iterrows():
        warnings = row.get("warnings", [])
        warning_text = " | ".join(warnings[:2]) if isinstance(warnings, list) else ""
        rows.append(
            {
                "campaign_id": row.get("campaign_id"),
                "stance": _clean_text(row.get("stance")).upper() or "UNKNOWN",
                "risk": _format_risk_label(row.get("severity")),
                "explanation": _decision_explanation_from_row(row),
                "evidence": " | ".join(_build_evidence_bits(row)),
                "warnings": warning_text,
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
        legend_title_text="Severity",
        xaxis_title="CPA / Target CPA (1.0 = on target)",
        yaxis_title="ROAS (1.0 = break-even pre-LTV)",
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
        xaxis_title="Budget multiplier",
        yaxis_title="Projected ROAS",
    )
    fig_roas.add_hline(y=1.0, line_width=1, opacity=0.6)

    fig_cpa = px.line(df_whatif, x="budget_multiplier", y="projected_CPA", markers=True)
    fig_cpa.update_layout(
        height=260,
        margin=dict(l=10, r=10, t=30, b=10),
        xaxis_title="Budget multiplier",
        yaxis_title="Projected CPA",
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
st.title("📊 Marketing Agent — Run Dashboard")

# Sidebar
st.sidebar.header("⚙️ Controls")
developer_mode = st.sidebar.toggle("Developer mode", value=False)

if developer_mode:
    with st.expander("Debug paths", expanded=False):
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

source_options = ["SQLite (agent_runs.db)", "Runs folder", "Local CSV in data/", "Upload CSV"]
default_index = 0 if not db_runs_df.empty else (1 if runs_folder else (2 if local_csvs else 3))
source_mode = st.sidebar.radio("Data source", options=source_options, index=default_index)

uploaded = None
if source_mode == "Upload CSV":
    uploaded = st.sidebar.file_uploader("Upload a campaigns CSV", type=["csv"])

run_meta = ""
df_run = None
selected_run_id = None

if source_mode == "SQLite (agent_runs.db)":
    if db_runs_df.empty:
        st.warning(f"No runs found in DB: {db_path}")
        st.stop()

    chosen_label = st.sidebar.selectbox("Select a DB run", options=db_runs_df["label"].tolist(), index=0)
    selected_run_id = str(db_runs_df.loc[db_runs_df["label"] == chosen_label, "run_id"].iloc[0])
    df_run = load_db_run_outputs(db_path, selected_run_id)
    if df_run.empty:
        st.warning(f"Run exists but no campaign outputs found for run_id={selected_run_id}")
        st.stop()
    run_meta = f"DB run: {selected_run_id}  |  {chosen_label}"

elif source_mode == "Upload CSV":
    if uploaded is None:
        st.info("Upload a CSV to continue.")
        st.stop()
    df_run = normalize_df(pd.read_csv(uploaded))
    run_meta = f"Uploaded file: {uploaded.name}"

elif source_mode == "Local CSV in data/":
    if not local_csvs:
        st.warning(f"No CSV files found in: {data_dir}")
        st.stop()
    csv_labels = [p.name for p in local_csvs]
    chosen_label = st.sidebar.selectbox("Select a local CSV", options=csv_labels, index=0)
    chosen_path = next(p for p in local_csvs if p.name == chosen_label)
    df_run = load_csv_path(chosen_path)
    run_meta = f"Local CSV: {chosen_path}"

else:  # Runs folder
    if not runs_folder:
        st.warning(
            f"No runs found in: {runs_dir}\n\n"
            "Fix options:\n"
            "1) Put runs under that folder (e.g. data/runs/<run_id>/campaigns.csv)\n"
            "2) Use 'Local CSV in data/'\n"
            "3) Upload a CSV\n"
            "4) Prefer SQLite if available"
        )
        st.stop()

    chosen_run = st.sidebar.selectbox("Select a run", options=runs_folder, index=0)
    run_dir = runs_dir / chosen_run
    df_run = load_run_csv(run_dir)
    if df_run is None:
        st.error(f"Run found but no CSV inside: {run_dir}")
        st.stop()
    run_meta = f"Run folder: {chosen_run} | dir={run_dir}"

st.caption(run_meta)

if source_mode == "SQLite (agent_runs.db)" and not df_run.empty:
    run_row = df_run.iloc[0]
    run_bits = []
    for label, key in [
        ("started", "run_started_at_utc"),
        ("input", "run_input_csv"),
        ("model", "run_model"),
        ("used_llm", "run_used_llm"),
    ]:
        value = _clean_text(run_row.get(key))
        if value:
            run_bits.append(f"{label}={value}")
    notes_text = _clean_text(run_row.get("run_notes"))
    if notes_text:
        run_bits.append(f"notes={notes_text}")
    if run_bits:
        st.caption("Run context: " + " | ".join(run_bits))

# Filters
all_stances = sorted([s for s in df_run["stance"].dropna().unique() if str(s).strip()])
all_severities = sorted(
    [s for s in df_run["severity"].dropna().unique() if str(s).strip()],
    key=lambda x: -SEVERITY_ORDER.get(str(x), 0),
)

stance_filter = st.sidebar.multiselect("Filter stance", options=all_stances, default=[])
severity_filter = st.sidebar.multiselect("Filter severity", options=all_severities, default=[])
search_id = st.sidebar.text_input("Search campaign_id", value="")

df_f = apply_filters(df_run, stance_filter, severity_filter, search_id)

# -----------------------------
# NEW: Today’s Queue (top action list)
# -----------------------------
st.subheader("✅ Today’s Queue (what to review first)")
st.caption("This review list is ordered by persisted severity and stance, with explanations pulled from the stored agent output.")

if df_f.empty:
    st.info("No campaigns match the current filters.")
else:
    df_queue = sort_campaigns_for_review(df_f).head(5).copy()

    # allow click-to-open via session_state
    if "selected_campaign" not in st.session_state:
        st.session_state["selected_campaign"] = df_queue["campaign_id"].iloc[0]

    for _, r in df_queue.iterrows():
        cid = str(r["campaign_id"])
        sev = str(r.get("severity", "—"))
        stance = str(r.get("stance", "—"))
        reason = _short_reason(r.get("decision_explanation"), max_len=150)
        advisor_actions = r.get("advisor_actions", [])
        evidence = " | ".join(_build_evidence_bits(r))
        warnings = r.get("warnings", [])

        with st.container(border=True):
            top = st.columns([3, 1])
            with top[0]:
                st.markdown(
                    f"**{cid}**  •  stance=`{stance}`  •  risk=`{_format_risk_label(sev)}`"
                )
            with top[1]:
                if st.button("Open", key=f"open_{cid}"):
                    st.session_state["selected_campaign"] = cid

            if reason:
                st.write(f"**Explanation:** {reason}")
            if evidence:
                st.caption(evidence)
            if isinstance(advisor_actions, list) and advisor_actions:
                st.write(f"**Recommended action:** {advisor_actions[0]}")
            if isinstance(warnings, list) and warnings:
                st.warning("Warning: " + warnings[0])

# Tabs
tab_overview, tab_details, tab_all = st.tabs(["📌 Overview", "🧠 Campaign details", "📋 All campaigns"])

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

    k1.metric("Campaigns", f"{campaigns_count}")
    k2.metric("Escalate", f"{escalate_count}")
    k3.metric("High severity", f"{high_count}")
    k4.metric("ROAS < 1.0", f"{roas_lt_1}")
    k5.metric("Avg CPA/Target", f"{avg_cpa_ratio:.2f}" if campaigns_count else "—")

    st.divider()

    st.subheader("Decision queue")
    st.caption("Ordered by persisted severity and stance, with agent explanations and supporting evidence.")
    df_top = build_campaign_display_df(sort_campaigns_for_review(df_f).head(10))
    st.dataframe(df_top, use_container_width=True, hide_index=True)

    st.download_button(
        "⬇️ Download CSV (Decision queue)",
        data=df_top.to_csv(index=False).encode("utf-8"),
        file_name="decision_queue.csv",
        mime="text/csv",
    )

    st.divider()

    st.subheader("📈 Efficiency vs Profitability")
    st.caption("Right of 1.0 = CPA above target. Below 1.0 = ROAS under 1.0. Color reflects persisted risk severity.")

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

    selected_campaign = st.selectbox("Select a campaign", options=campaign_options, index=idx)
    st.session_state["selected_campaign"] = selected_campaign

    row = df_run[df_run["campaign_id"] == selected_campaign].iloc[0]

    st.subheader(f"Details: {selected_campaign}")
    d1, d2, d3, d4, d5 = st.columns(5)

    stance_val = row.get("stance")
    severity_val = row.get("severity")

    d1.metric("Stance", "—" if pd.isna(stance_val) else str(stance_val))
    d2.metric("Severity", "—" if pd.isna(severity_val) else str(severity_val))
    d3.metric("CPA", f"{row['cpa']:.2f}" if pd.notna(row.get("cpa")) else "—")
    d4.metric("Target CPA", f"{row['target_cpa']:.2f}" if pd.notna(row.get("target_cpa")) else "—")
    d5.metric("CPA/Target", f"{row['cpa_ratio']:.2f}" if pd.notna(row.get("cpa_ratio")) else "—")

    st.divider()

    st.markdown("### Decision")
    st.write(f"**Risk:** {_format_risk_label(row.get('severity'))}")

    explanation = _clean_text(row.get("decision_explanation"))
    if explanation:
        st.write(f"**Short explanation:** {explanation}")

    evidence_bits = _build_evidence_bits(row)
    if evidence_bits:
        st.write("**Supporting evidence:**")
        for bit in evidence_bits:
            st.write(f"- {bit}")

    reasons = row.get("reasons", [])
    st.write("**Persisted reasons:**")
    if isinstance(reasons, list) and reasons:
        for reason in reasons:
            st.write(f"- {reason}")
    else:
        st.write("- —")

    warnings = row.get("warnings", [])
    if isinstance(warnings, list) and warnings:
        st.write("**Warnings / data quality notes:**")
        for warning in warnings:
            st.write(f"- {warning}")

    # Advisor
    st.markdown("### Advisor")
    advisor_summary = row.get("advisor_summary", pd.NA)
    advisor_actions = row.get("advisor_actions", [])

    meta_bits = []
    for k in ["advisor_confidence", "advisor_used_llm", "advisor_model"]:
        v = row.get(k, pd.NA)
        if pd.notna(v) and str(v).strip() and str(v).lower() != "nan":
            meta_bits.append(f"{k.replace('advisor_','')}={v}")
    if meta_bits:
        st.caption(" | ".join(meta_bits))

    if pd.notna(advisor_summary) and str(advisor_summary).strip():
        st.write(str(advisor_summary))
    else:
        st.caption("No advisor summary was stored for this campaign.")

    if isinstance(advisor_actions, list) and advisor_actions:
        for i, a in enumerate(advisor_actions, 1):
            st.write(f"{i}. {a}")
    else:
        st.caption("No advisor actions were stored for this campaign.")

    st.divider()

    # What-if (simulation)
    st.markdown("### 🧪 What-if scenarios (simulation)")
    scenarios = row.get("scenarios", [])
    df_whatif = scenarios_to_df(scenarios if isinstance(scenarios, list) else [])
    if df_whatif.empty:
        st.caption("No scenarios found in this run.")
    else:
        w1, w2 = st.columns(2)
        fig_roas, fig_cpa = make_what_if_charts(df_whatif)
        with w1:
            st.plotly_chart(fig_roas, use_container_width=True)
        with w2:
            st.plotly_chart(fig_cpa, use_container_width=True)
        st.dataframe(df_whatif, use_container_width=True, hide_index=True)

    st.divider()

    # Analysis
    st.markdown("### Analysis")
    analysis_sum = row.get("analysis_summary", pd.NA)
    if pd.notna(analysis_sum) and str(analysis_sum).strip():
        st.write(str(analysis_sum))

    st.markdown("**Insights**")
    ins = row.get("analysis_insights", [])
    if isinstance(ins, list) and ins:
        for it in ins:
            st.write(f"- {it}")
    else:
        st.write("- —")

    st.markdown("**Suggested actions (analysis)**")
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
        with st.expander("Execution metadata", expanded=False):
            st.json(execution_metadata)

    if developer_mode:
        with st.expander("Raw row JSON", expanded=False):
            st.code(json.dumps(row.to_dict(), indent=2, default=str), language="json")

# -----------------------------
# All campaigns
# -----------------------------
with tab_all:
    st.subheader("All campaigns (filtered)")
    df_all = build_campaign_display_df(df_f)
    st.dataframe(df_all, use_container_width=True, hide_index=True)
    st.download_button(
        "⬇️ Download CSV (filtered)",
        data=df_all.to_csv(index=False).encode("utf-8"),
        file_name="campaigns_filtered.csv",
        mime="text/csv",
    )