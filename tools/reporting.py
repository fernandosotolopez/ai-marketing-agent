from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict, List, Optional

from tools.analysis import AnalysisResult, Insight
from registry.tool_registry import register_tool

@register_tool(tags=["reporting", "console"])
def format_console_report(
    decision: Dict[str, Any],
    analysis: AnalysisResult,
    warnings: Optional[List[str]] = None,
    errors: Optional[List[str]] = None,
) -> str:
    """
    Returns a human-readable console report string.

    decision: {"stance": "...", "severity": "...", "reasons": [...]}
    analysis: AnalysisResult from tools/analysis.py
    warnings/errors: from loader/metrics stages (optional)
    """
    lines: List[str] = []

    lines.append("=" * 72)
    lines.append("MARKETING AGENT REPORT")
    lines.append("=" * 72)

    lines.append(f"Campaign: {analysis.campaign_id}")
    lines.append(f"Summary:  {analysis.summary}")
    lines.append("")

    # Decision block
    stance = decision.get("stance", "unknown")
    severity = decision.get("severity", "unknown")
    lines.append("DECISION")
    lines.append("-" * 72)
    lines.append(f"Stance:   {stance}")
    lines.append(f"Severity: {severity}")

    reasons = decision.get("reasons") or []
    if reasons:
        lines.append("Reasons:")
        for r in reasons:
            lines.append(f"  - {r}")
    else:
        lines.append("Reasons:  (none)")
    lines.append("")

    # Insights
    lines.append("INSIGHTS")
    lines.append("-" * 72)
    if analysis.insights:
        for ins in analysis.insights:
            lines.append(f"  [{ins.importance.upper():6}] ({ins.category}) {ins.message}")
    else:
        lines.append("  (none)")
    lines.append("")

    # Actions
    lines.append("SUGGESTED ACTIONS")
    lines.append("-" * 72)
    if analysis.suggested_actions:
        for a in analysis.suggested_actions:
            lines.append(f"  - {a}")
    else:
        lines.append("  (none)")
    lines.append("")

    # Pipeline warnings/errors
    if errors:
        lines.append("ERRORS")
        lines.append("-" * 72)
        for e in errors:
            lines.append(f"  - {e}")
        lines.append("")

    if warnings:
        lines.append("WARNINGS")
        lines.append("-" * 72)
        for w in warnings:
            lines.append(f"  - {w}")
        lines.append("")

    lines.append("=" * 72)
    return "\n".join(lines)

@register_tool(tags=["reporting", "markdown"])
def format_markdown_report(
    decision: Dict[str, Any],
    analysis: AnalysisResult,
    warnings: Optional[List[str]] = None,
    errors: Optional[List[str]] = None,
) -> str:
    """
    Returns a Markdown version (great for README / Notion / GitHub).
    """
    stance = decision.get("stance", "unknown")
    severity = decision.get("severity", "unknown")
    reasons = decision.get("reasons") or []

    md: List[str] = []
    md.append(f"# Marketing Agent Report — `{analysis.campaign_id}`")
    md.append("")
    md.append(f"**Summary:** {analysis.summary}")
    md.append("")
    md.append("## Decision")
    md.append(f"- **Stance:** `{stance}`")
    md.append(f"- **Severity:** `{severity}`")
    if reasons:
        md.append("- **Reasons:**")
        for r in reasons:
            md.append(f"  - {r}")
    else:
        md.append("- **Reasons:** (none)")
    md.append("")

    md.append("## Insights")
    if analysis.insights:
        for ins in analysis.insights:
            md.append(f"- **{ins.importance.upper()}** ({ins.category}): {ins.message}")
    else:
        md.append("- (none)")
    md.append("")

    md.append("## Suggested Actions")
    if analysis.suggested_actions:
        for a in analysis.suggested_actions:
            md.append(f"- {a}")
    else:
        md.append("- (none)")
    md.append("")

    if errors:
        md.append("## Errors")
        for e in errors:
            md.append(f"- {e}")
        md.append("")

    if warnings:
        md.append("## Warnings")
        for w in warnings:
            md.append(f"- {w}")
        md.append("")

    return "\n".join(md)


def analysis_to_dict(analysis: AnalysisResult) -> Dict[str, Any]:
    """
    Helpful if you want to output JSON for integration later.
    """
    # AnalysisResult contains dataclasses; asdict handles nested dataclasses.
    return asdict(analysis)
