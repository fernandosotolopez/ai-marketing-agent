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


def normalize_df(df: pd.DataFrame) -> pd.DataFrame:
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

    if df["risk_score"].isna().all():
        df["risk_score"] = compute_risk_score(df)

    missing_why = df["why_flagged"].isna() | (df["why_flagged"].astype("string").str.strip() == "")
    if missing_why.any():
        df.loc[missing_why, "why_flagged"] = df[missing_why].apply(default_why_flagged, axis=1)

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
        df = pd.read_sql_query(
            "SELECT run_id, started_at_utc, input_csv, max_rows, save_memory, model, used_llm, notes "
            "FROM runs ORDER BY started_at_utc DESC",
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
        df = pd.read_sql_query(
            "SELECT run_id, campaign_id, stance, severity, "
            "cpa, roas, target_cpa, cpa_trend_7d, roas_trend_7d, days_active, "
            "state_json, decision_json, analysis_json, advisor_json, scenarios_json "
            "FROM campaign_outputs WHERE run_id = ?",
            con,
            params=(run_id,),
        )
    finally:
        con.close()

    if df.empty:
        return df

    def _reasons_to_why(dec_json: Any) -> Any:
        d = _safe_json_loads(dec_json, {})
        reasons = d.get("reasons") if isinstance(d, dict) else None
        if not reasons:
            return pd.NA
        parts = [str(r).strip() for r in reasons if str(r).strip()]
        return " | ".join(parts) if parts else pd.NA

    df["why_flagged"] = df["decision_json"].apply(_reasons_to_why)

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

    df = normalize_df(df)
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
    size = df["risk_score"].fillna(0).clip(lower=0)
    size = (np.sqrt(size + 1) * 8).clip(8, 34)

    fig = px.scatter(
        df,
        x="cpa_ratio",
        y="roas",
        color="severity",
        size=size,
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
            "risk_score": ":.2f",
            "why_flagged": True,
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
st.caption("This is your prioritized review list. Each card shows the top reason + the next action.")

if df_f.empty:
    st.info("No campaigns match the current filters.")
else:
    df_queue = df_f.sort_values("risk_score", ascending=False).head(5).copy()

    # choose a 1-line next action
    def _next_action(row: pd.Series) -> str:
        actions = row.get("advisor_actions", [])
        if isinstance(actions, list) and actions:
            return str(actions[0]).strip()
        # fallback to deterministic analysis suggested actions
        acts = row.get("analysis_suggested_actions", [])
        if isinstance(acts, list) and acts:
            return str(acts[0]).strip()
        return "Review metrics and investigate root cause."

    df_queue["queue_reason"] = df_queue["why_flagged"].apply(_short_reason)
    df_queue["queue_next_action"] = df_queue.apply(_next_action, axis=1)

    # allow click-to-open via session_state
    if "selected_campaign" not in st.session_state:
        st.session_state["selected_campaign"] = df_queue["campaign_id"].iloc[0]

    for _, r in df_queue.iterrows():
        cid = str(r["campaign_id"])
        sev = str(r.get("severity", "—"))
        stance = str(r.get("stance", "—"))
        reason = r.get("queue_reason", "")
        next_action = r.get("queue_next_action", "")

        with st.container(border=True):
            top = st.columns([3, 1])
            with top[0]:
                st.markdown(f"**{cid}**  •  stance=`{stance}`  •  severity=`{sev}`")
            with top[1]:
                if st.button("Open", key=f"open_{cid}"):
                    st.session_state["selected_campaign"] = cid

            if reason:
                st.write(f"**Why:** {reason}")
            if next_action:
                st.write(f"**Next:** {next_action}")

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

    st.subheader("🔥 Top risks table")
    st.caption("Ranked by heuristic risk_score (triage).")
    df_top = df_f.sort_values("risk_score", ascending=False).head(10).copy()

    show_cols = [
        "campaign_id", "stance", "severity",
        "cpa", "target_cpa", "cpa_ratio",
        "roas", "cpa_trend_7d", "roas_trend_7d",
        "days_active", "risk_score", "why_flagged",
    ]
    if "run_id" in df_top.columns and df_top["run_id"].notna().any():
        show_cols = ["run_id"] + show_cols
    show_cols = [c for c in show_cols if c in df_top.columns]

    st.dataframe(df_top[show_cols], use_container_width=True, hide_index=True)

    st.download_button(
        "⬇️ Download CSV (Top risks)",
        data=df_top[show_cols].to_csv(index=False).encode("utf-8"),
        file_name="top_risks.csv",
        mime="text/csv",
    )

    st.divider()

    st.subheader("📈 Efficiency vs Profitability (easy view)")
    st.caption("How to read: Right of 1.0 = CPA above target. Below 1.0 = losing money (pre-LTV). Bigger bubble = higher triage risk.")

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
    campaign_options = (
        df_f.sort_values(["risk_score", "campaign_id"], ascending=[False, True])["campaign_id"].tolist()
    )
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

    # Why flagged
    why_val = row.get("why_flagged")
    st.markdown("### 🧾 Why flagged (deterministic agent reasons)")
    if pd.isna(why_val):
        st.write("- —")
    else:
        parts = [r.strip() for r in str(why_val).split("|") if r.strip()]
        for r in parts:
            st.write(f"- {r}")

    # Advisor
    st.markdown("### ✅ Action plan (Advisor)")
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
        st.caption("No advisor summary available.")

    if isinstance(advisor_actions, list) and advisor_actions:
        for i, a in enumerate(advisor_actions, 1):
            st.write(f"{i}. {a}")
    else:
        st.caption("No advisor actions available.")

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
    st.markdown("### 🔎 Analysis (deterministic)")
    analysis_sum = row.get("analysis_summary", pd.NA)
    if pd.notna(analysis_sum) and str(analysis_sum).strip():
        st.caption(str(analysis_sum))

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

    if developer_mode:
        with st.expander("Raw row JSON", expanded=False):
            st.code(json.dumps(row.to_dict(), indent=2, default=str), language="json")

# -----------------------------
# All campaigns
# -----------------------------
with tab_all:
    st.subheader("All campaigns (filtered)")
    st.dataframe(df_f.sort_values("risk_score", ascending=False), use_container_width=True, hide_index=True)
    st.download_button(
        "⬇️ Download CSV (filtered)",
        data=df_f.to_csv(index=False).encode("utf-8"),
        file_name="campaigns_filtered.csv",
        mime="text/csv",
    )

st.caption("Tip: If you add spend/revenue/conversions, you can prioritize by $ impact, not just efficiency ratios.")