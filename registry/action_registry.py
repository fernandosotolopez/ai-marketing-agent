from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from registry.tool_registry import GLOBAL_TOOL_REGISTRY, ToolSpec


@dataclass
class ActionRegistry:
    """
    Curated subset of tools for an agent.

    You can build it by:
      - tags=["file_operations"]
      - names=["load_campaign_csv", "validate_and_enrich_row"]
      - or both
    """
    tags: Optional[List[str]] = None
    names: Optional[List[str]] = None

    def __post_init__(self) -> None:
        self._tools: Dict[str, ToolSpec] = {}
        self._build()

    def _build(self) -> None:
        selected: Dict[str, ToolSpec] = {}

        if self.tags:
            for t in self.tags:
                for spec in GLOBAL_TOOL_REGISTRY.list_by_tag(t):
                    selected[spec.name] = spec

        if self.names:
            for n in self.names:
                spec = GLOBAL_TOOL_REGISTRY.get(n)
                selected[spec.name] = spec

        self._tools = selected

    def list_names(self) -> List[str]:
        return list(self._tools.keys())

    def list_specs(self) -> List[ToolSpec]:
        return list(self._tools.values())

    def get(self, name: str) -> ToolSpec:
        if name not in self._tools:
            raise KeyError(f"Tool not available in this ActionRegistry: {name}")
        return self._tools[name]

    def call(self, name: str, **kwargs: Any) -> Any:
        """
        Execute a tool by name with kwargs.

        Later: this is what an LLM “function call” would trigger.
        """
        spec = self.get(name)
        return spec.fn(**kwargs)

    def describe(self) -> Dict[str, Any]:
        """
        Minimal description of the curated tool set.
        """
        return {
            "available_tools": {
                name: {
                    "tags": spec.tags,
                    "description": spec.description,
                    "parameters": spec.parameters,
                    "returns": spec.returns,
                }
                for name, spec in self._tools.items()
            }
        }
