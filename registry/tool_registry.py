from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, get_args, get_origin
import inspect


@dataclass
class ToolSpec:
    """
    Metadata about a tool function so we can organize, inspect, and call it.
    """
    name: str
    fn: Callable[..., Any]
    tags: List[str] = field(default_factory=list)
    description: str = ""
    parameters: Dict[str, Any] = field(default_factory=dict)
    returns: Optional[str] = None


class ToolRegistry:
    """
    Central storage of tools + tag index.
    """

    def __init__(self) -> None:
        self.tools: Dict[str, ToolSpec] = {}
        self.tools_by_tag: Dict[str, List[str]] = {}

    def register(self, fn: Callable[..., Any], tags: Optional[List[str]] = None) -> Callable[..., Any]:
        tags = tags or []
        name = fn.__name__
        description = (inspect.getdoc(fn) or "").strip()

        spec = ToolSpec(
            name=name,
            fn=fn,
            tags=tags,
            description=description,
            parameters=_infer_parameters_schema(fn),
            returns=_infer_return_type(fn),
        )

        self.tools[name] = spec

        for t in tags:
            self.tools_by_tag.setdefault(t, [])
            if name not in self.tools_by_tag[t]:
                self.tools_by_tag[t].append(name)

        return fn

    def get(self, name: str) -> ToolSpec:
        if name not in self.tools:
            raise KeyError(f"Tool not found: {name}")
        return self.tools[name]

    def list_tools(self) -> List[ToolSpec]:
        return list(self.tools.values())

    def list_names(self) -> List[str]:
        return list(self.tools.keys())

    def list_by_tag(self, tag: str) -> List[ToolSpec]:
        names = self.tools_by_tag.get(tag, [])
        return [self.tools[n] for n in names]

    def describe(self) -> Dict[str, Any]:
        return {
            "tools": {
                name: {
                    "tags": spec.tags,
                    "description": spec.description,
                    "parameters": spec.parameters,
                    "returns": spec.returns,
                }
                for name, spec in self.tools.items()
            },
            "tools_by_tag": dict(self.tools_by_tag),
        }


GLOBAL_TOOL_REGISTRY = ToolRegistry()


def register_tool(tags: Optional[List[str]] = None) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        return GLOBAL_TOOL_REGISTRY.register(fn, tags=tags)

    return decorator


# -----------------------
# Schema inference (safe on forward refs like "MemoryStore")
# -----------------------

def _safe_get_type_hints(fn: Callable[..., Any]) -> Dict[str, Any]:
    """
    get_type_hints() can crash if it tries to resolve forward references
    that aren't imported at runtime (e.g., "MemoryStore").

    We'll try it and fall back to raw __annotations__ if it fails.
    """
    try:
        from typing import get_type_hints as _gth
        return _gth(fn)
    except Exception:
        return dict(getattr(fn, "__annotations__", {}) or {})


def _pytype_to_json_type(py_type: Any) -> str:
    if isinstance(py_type, str):
        return "string"

    origin = get_origin(py_type)
    args = get_args(py_type)

    if origin in (list, List):
        return "array"
    if origin in (dict, Dict):
        return "object"

    # Union / Optional handling
    if origin is getattr(__import__("typing"), "Union"):
        non_none = [a for a in args if a is not type(None)]
        if len(non_none) == 1:
            return _pytype_to_json_type(non_none[0])
        return "string"

    if py_type is int:
        return "integer"
    if py_type is float:
        return "number"
    if py_type is bool:
        return "boolean"
    if py_type is str:
        return "string"

    return "string"


def _infer_parameters_schema(fn: Callable[..., Any]) -> Dict[str, Any]:
    sig = inspect.signature(fn)
    hints = _safe_get_type_hints(fn)

    properties: Dict[str, Any] = {}
    required: List[str] = []

    for name, param in sig.parameters.items():
        if name in ("self", "cls"):
            continue

        hint = hints.get(name, str)
        jtype = _pytype_to_json_type(hint)

        prop: Dict[str, Any] = {"type": jtype}

        if param.default is not inspect._empty:
            prop["default"] = param.default
        else:
            required.append(name)

        properties[name] = prop

    return {"type": "object", "properties": properties, "required": required}


def _infer_return_type(fn: Callable[..., Any]) -> Optional[str]:
    hints = _safe_get_type_hints(fn)
    ret = hints.get("return")
    return None if ret is None else str(ret)
