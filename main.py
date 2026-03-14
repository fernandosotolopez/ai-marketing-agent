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
        out.append(
            {
                "scenario_name": getattr(s, "scenario_name", None),
                "budget_multiplier": getattr(s, "budget_multiplier", None),
                "projected_CPA": getattr(s, "projected_CPA", None),
                "projected_ROAS": getattr(s, "projected_ROAS", None),
                "notes": getattr(s, "notes", None),
            }
        )
    return out


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
        advisor = LLMAdvisor()

        # 3) Run through campaigns
        if args.max_rows and args.max_rows > 0:
            max_rows = min(args.max_rows, len(rows))
        else:
            max_rows = len(rows)

        # Create a run record once per execution
        run_id = run_db.start_run(
            con,
            input_csv=str(csv_path),
            max_rows=int(max_rows),
            save_memory=int(args.save_memory),
            model=None,
            used_llm=None,
            notes="",
        )
        print(f"[DB] run_id={run_id} (saving outputs to data/agent_runs.db)\n")

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
            print(f"(advisor_used_llm={advice.get('advisor_used_llm')}, model={advice.get('advisor_model')})")
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

            # Save to SQLite (run history)
            run_db.save_campaign_output(
                con,
                run_id=run_id,
                campaign_id=campaign_id,
                state=enriched,
                decision=decision,
                analysis=to_jsonable(analysis),
                advisor=to_jsonable(advice),
                scenarios=scenarios_to_jsonable(scenarios),
            )

            # 3g) Store memory (in RAM)
            memory.add(campaign_id=campaign_id, state=enriched, decision=decision)

        # 4) Persist memory to disk so next run can use it (ONLY if save-memory=1)
        if args.save_memory == 1:
            memory_path.parent.mkdir(parents=True, exist_ok=True)
            memory.save_json(str(memory_path))
            print(f"Saved memory to: {memory_path}")
        else:
            print("Memory NOT saved (save-memory=0).")

    finally:
        con.close()


if __name__ == "__main__":
    main()
