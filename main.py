from __future__ import annotations

from pathlib import Path
from typing import Any, Dict
import argparse

from dotenv import load_dotenv

load_dotenv()

# Importing tools package triggers decorators in tools/__init__.py
# which registers all @register_tool functions into GLOBAL_TOOL_REGISTRY.
import tools  # noqa: F401

from agent.agent import MarketingAgent
from agent.goals import CostEfficiencyGoal, PerformanceTrendGoal, MaturityGoal
from agent.loop import AgentLoop
from agent.memory import MemoryStore
from agent.llm_advisor import LLMAdvisor

from storage import db as run_db

from tools.data_loader import load_campaign_csv
from tools.metrics import validate_and_enrich_row
from tools.analysis import analyze_campaign_row
from tools.reporting import format_console_report
from tools.simulation import run_budget_scenarios, choose_default_multipliers


def to_jsonable(obj: Any) -> Any:
    """Convierte objetos (pydantic/dataclasses/etc.) a algo serializable a JSON (recursivo)."""
    if obj is None:
        return None

    # Tipos primitivos
    if isinstance(obj, (str, int, float, bool)):
        return obj

    # Listas/tuplas: convertir cada elemento
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(x) for x in obj]

    # Diccionarios: convertir valores
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}

    # Pydantic v2
    if hasattr(obj, "model_dump"):
        try:
            return to_jsonable(obj.model_dump())
        except Exception:
            pass

    # Pydantic v1
    if hasattr(obj, "dict"):
        try:
            return to_jsonable(obj.dict())
        except Exception:
            pass

    # Dataclasses
    try:
        import dataclasses
        if dataclasses.is_dataclass(obj):
            return to_jsonable(dataclasses.asdict(obj))
    except Exception:
        pass

    # Objetos con __dict__
    if hasattr(obj, "__dict__"):
        try:
            return {k: to_jsonable(v) for k, v in obj.__dict__.items() if not k.startswith("_")}
        except Exception:
            pass

    # Fallback final
    return str(obj)



def scenarios_to_jsonable(scenarios: Any) -> Any:
    out = []
    for s in scenarios or []:
        normalized_notes = _normalize_scenario_notes(getattr(s, "notes", None))
        out.append(
            {
                "scenario_name": getattr(s, "scenario_name", None),
                "budget_multiplier": getattr(s, "budget_multiplier", None),
                "projected_CPA": getattr(s, "projected_CPA", None),
                "projected_ROAS": getattr(s, "projected_ROAS", None),
                "notes": normalized_notes,
            }
        )
    return out


def _normalize_scenario_notes(value: Any) -> Any:
    if isinstance(value, list):
        clean_notes = [str(item).strip() for item in value if str(item).strip()]
        return " | ".join(clean_notes) if clean_notes else None

    text = str(value).strip() if value is not None else ""
    if not text or text.lower() in {"none", "null", "nan", "[]"}:
        return None
    return text


def _append_unique_text(items: list[str], value: Any) -> None:
    text = str(value or "").strip()
    if text and text not in items:
        items.append(text)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run marketing agent on a campaign CSV.")

    p.add_argument(
        "--csv",
        type=str,
        default=str(Path("data") / "campaign_data.csv"),
        help="Path to CSV file to analyze (default: data/campaign_data.csv)",
    )

    p.add_argument(
        "--memory",
        type=str,
        default=str(Path("data") / "memory.json"),
        help="Path to memory JSON file (default: data/memory.json)",
    )

    p.add_argument(
        "--max-rows",
        type=int,
        default=5,
        help="How many rows to process (default: 5). Use 0 to process all rows.",
    )

    p.add_argument(
        "--save-memory",
        type=int,
        default=1,
        help="1 = save memory.json, 0 = do not save (safe testing mode)",
    )

    p.add_argument(
        "--advisor-mode",
        choices=["auto", "llm", "deterministic"],
        default="auto",
        help=(
            "LLM advisor: auto = use LLM when API key is set, except for data/demo_campaigns.csv "
            "(deterministic freeze file); llm = always attempt LLM; deterministic = never call LLM."
        ),
    )

    return p.parse_args()


def main() -> None:
    args = parse_args()

    # DB for run history (SQLite file: data/agent_runs.db)
    con = run_db.connect()
    run_db.init_db(con)

    try:
        # 0) Load persistent memory (so trends can use history across runs)
        memory = MemoryStore()
        memory_path = Path(args.memory)
        if memory_path.exists():
            try:
                memory.load_json(str(memory_path))
            except Exception as e:
                print(f"WARNING: Could not load memory file '{memory_path}': {e}")
                print("Continuing with empty memory.\n")

        # 1) Load CSV
        csv_path = Path(args.csv)
        load_result = load_campaign_csv(str(csv_path))

        if load_result.warnings:
            print("CSV LOAD WARNINGS:")
            for w in load_result.warnings:
                print(f"  - {w}")
            print()

        rows = load_result.rows
        if not rows:
            print("No rows found in CSV.")
            return

        # 2) Build agent + loop
        goals = [
            MaturityGoal(min_days_active=14),
            CostEfficiencyGoal(),
            PerformanceTrendGoal(),
        ]
        agent = MarketingAgent(goals=goals)
        loop = AgentLoop(agent=agent)
        is_demo_freeze_file = csv_path.name == "demo_campaigns.csv"
        if args.advisor_mode == "deterministic":
            advisor_use_llm = False
        elif args.advisor_mode == "llm":
            advisor_use_llm = True
        else:
            # auto: deterministic advisor for the curated demo CSV; LLM elsewhere when key exists
            advisor_use_llm = not is_demo_freeze_file

        advisor = LLMAdvisor(use_llm=advisor_use_llm)

        # 3) Run through campaigns
        if args.max_rows and args.max_rows > 0:
            max_rows = min(args.max_rows, len(rows))
        else:
            max_rows = len(rows)

        initial_run_metadata: Dict[str, Any] = {
            "phase": "phase_1_block_4",
            "pipeline_order": [
                "load_campaign_csv",
                "validate_and_enrich_row",
                "AgentLoop.run",
                "analyze_campaign_row",
                "run_budget_scenarios",
                "format_console_report",
                "LLMAdvisor.advise",
                "save_campaign_output",
                "MemoryStore.add",
                "MemoryStore.save_json",
            ],
            "csv_load_warnings": list(load_result.warnings),
            "csv_load_warning_count": len(load_result.warnings),
            "requested_max_rows": int(args.max_rows),
            "effective_max_rows": int(max_rows),
            "input_memory_path": str(memory_path),
            "save_memory_requested": int(args.save_memory),
            "advisor_configured_model": getattr(advisor, "model", None),
            "advisor_mode_cli": args.advisor_mode,
            "demo_freeze_csv": is_demo_freeze_file,
        }

        # Create a run record once per execution
        run_id = run_db.start_run(
            con,
            input_csv=str(csv_path),
            max_rows=int(max_rows),
            save_memory=int(args.save_memory),
            model=None,
            used_llm=None,
            notes="",
            run_metadata=initial_run_metadata,
        )
        print(f"[DB] run_id={run_id} (saving outputs to data/agent_runs.db)\n")
        print(
            f"[Advisor] mode={args.advisor_mode} "
            f"use_llm={advisor_use_llm} "
            f"model={getattr(advisor, 'model', None)}\n"
        )

        persisted_rows = 0
        skipped_validation_rows = []
        advisor_used_llm = False
        advisor_models_seen: list[str] = []

        for i in range(max_rows):
            row = rows[i]

            # 3a) Validate/enrich metrics
            metrics_result = validate_and_enrich_row(row=row, memory=memory)

            if metrics_result.errors:
                print("=" * 72)
                print(f"ROW {i+1} FAILED METRICS VALIDATION")
                print("=" * 72)
                for e in metrics_result.errors:
                    print(f"  - {e}")
                print()
                skipped_validation_rows.append(
                    {
                        "row_index": i + 1,
                        "campaign_id": row.get("campaign_id"),
                        "errors": list(metrics_result.errors),
                        "warnings": list(metrics_result.warnings),
                    }
                )
                continue

            enriched: Dict[str, Any] = metrics_result.row
            campaign_id = str(enriched.get("campaign_id") or f"ROW_{i+1}")

            # 3b) Agent decision
            decision: Dict[str, Any] = loop.run(enriched)

            # 3c) Analysis
            analysis = analyze_campaign_row(enriched, decision=decision)

            # 3d) Simulation
            multipliers = choose_default_multipliers(decision)
            scenarios = run_budget_scenarios(enriched, multipliers=multipliers)

            # 3e) Report (deterministic report)
            report = format_console_report(
                decision=decision,
                analysis=analysis,
                warnings=metrics_result.warnings,
                errors=metrics_result.errors,
            )
            print(report)

            # 3e.1) Advisor block (future LLM)
            advice = advisor.advise(state=enriched, decision=decision)
            print(
                f"(advisor_source={advice.get('advisor_source')}, "
                f"advisor_used_llm={advice.get('advisor_used_llm')}, "
                f"model={advice.get('advisor_model')}, "
                f"fallback_reason={advice.get('advisor_fallback_reason')})"
            )
            print("ADVISOR (future LLM)")
            print("-" * 72)
            print(advice["advisor_summary"])
            print("\nRecommended actions:")
            for a in advice["advisor_actions"]:
                print(f"  - {a}")
            print()

            # 3f) Print simulation summary
            print("\nWHAT-IF SCENARIOS (budget multipliers)")
            for s in scenarios:
                cpa = "N/A" if s.projected_CPA is None else f"{s.projected_CPA:.2f}"
                roas = "N/A" if s.projected_ROAS is None else f"{s.projected_ROAS:.2f}"
                print(f"- {s.scenario_name:12} | m={s.budget_multiplier:.2f} | CPA={cpa:>8} | ROAS={roas:>8}")
                if s.notes:
                    for n in s.notes:
                        print(f"    note: {n}")

            print("\n" + ("#" * 72) + "\n")

            enriched_json = to_jsonable(enriched)
            decision_json = to_jsonable(decision)
            analysis_json = to_jsonable(analysis)
            advisor_json = to_jsonable(advice)
            scenarios_json = scenarios_to_jsonable(scenarios)
            provenance_json = {}
            if isinstance(enriched_json, dict):
                provenance_json = to_jsonable(enriched_json.get("_metric_provenance") or {})
            execution_metadata = {
                "row_index": i + 1,
                "metrics_warning_count": len(metrics_result.warnings),
                "advisor_used_llm": advisor_json.get("advisor_used_llm")
                if isinstance(advisor_json, dict)
                else None,
                "advisor_model": advisor_json.get("advisor_model")
                if isinstance(advisor_json, dict)
                else None,
                "advisor_source": advisor_json.get("advisor_source")
                if isinstance(advisor_json, dict)
                else None,
                "advisor_fallback_reason": advisor_json.get("advisor_fallback_reason")
                if isinstance(advisor_json, dict)
                else None,
            }

            # Save to SQLite (run history)
            run_db.save_campaign_output(
                con,
                run_id=run_id,
                campaign_id=campaign_id,
                state=enriched_json,
                decision=decision_json,
                analysis=analysis_json,
                advisor=advisor_json,
                scenarios=scenarios_json,
                warnings=list(metrics_result.warnings),
                provenance=provenance_json,
                execution_metadata=execution_metadata,
            )
            persisted_rows += 1
            if isinstance(advisor_json, dict):
                advisor_used_llm = advisor_used_llm or bool(advisor_json.get("advisor_used_llm"))
                _append_unique_text(advisor_models_seen, advisor_json.get("advisor_model"))

            # 3g) Store memory (in RAM)
            memory.add(campaign_id=campaign_id, state=enriched, decision=decision)

        # 4) Persist memory to disk so next run can use it (ONLY if save-memory=1)
        if args.save_memory == 1:
            memory_path.parent.mkdir(parents=True, exist_ok=True)
            memory.save_json(str(memory_path))
            print(f"Saved memory to: {memory_path}")
        else:
            print("Memory NOT saved (save-memory=0).")

        run_notes = (
            f"csv_load_warnings={len(load_result.warnings)}; "
            f"rows_persisted={persisted_rows}; "
            f"rows_skipped_validation={len(skipped_validation_rows)}"
        )
        final_run_metadata: Dict[str, Any] = {
            **initial_run_metadata,
            "rows_in_csv": len(rows),
            "rows_persisted": persisted_rows,
            "rows_skipped_validation_count": len(skipped_validation_rows),
            "skipped_validation_rows": to_jsonable(skipped_validation_rows),
            "advisor_used_llm": advisor_used_llm,
            "advisor_models_seen": advisor_models_seen,
            "memory_saved": args.save_memory == 1,
        }
        run_db.finalize_run(
            con,
            run_id=run_id,
            model=advisor_models_seen[0] if advisor_models_seen else None,
            used_llm=advisor_used_llm,
            notes=run_notes,
            run_metadata=final_run_metadata,
        )

    finally:
        con.close()


if __name__ == "__main__":
    main()
