"""Gate a LangChain-style tool without importing LangChain.

A LangChain tool is an object with a `name`, a `description`, and something callable —
`_run`, `func`, or `invoke` depending on the version and whether it was built with the
decorator or the class. This adapter asks for that shape and nothing else, so it works
with `@tool`-decorated functions, `StructuredTool`, and anything else that looks the
same, and it never becomes a dependency of your agent.

The returned object is a *new* wrapper: the original tool is untouched, so a caller can
keep an ungated handle for trusted internal use if it wants one — deliberately, and in
one place, rather than by accident.
"""

from __future__ import annotations

from typing import Any, Callable

from ..decision import Effect, ToolCall
from ..errors import NeedsHumanApproval, PolicyBlocked
from ..gate import Gate

_CALLABLE_ATTRIBUTES = ("_run", "run", "func", "invoke", "__call__")


def _find_callable(tool: Any) -> Callable[..., Any]:
    for attribute in _CALLABLE_ATTRIBUTES:
        candidate = getattr(tool, attribute, None)
        if callable(candidate):
            return candidate
    if callable(tool):
        return tool
    raise TypeError(
        "cannot find a callable on this tool; expected one of "
        f"{', '.join(_CALLABLE_ATTRIBUTES)} or the tool itself to be callable"
    )


class GatedTool:
    """A drop-in replacement for a tool that consults the gate before running."""

    def __init__(self, tool: Any, gate: Gate, *, name: str | None = None) -> None:
        self._tool = tool
        self._gate = gate
        self._call = _find_callable(tool)
        self.name = name or getattr(tool, "name", type(tool).__name__)
        self.description = getattr(tool, "description", "")
        self.args_schema = getattr(tool, "args_schema", None)

    # -- the gated path ----------------------------------------------------

    def _gate_or_raise(self, arguments: dict[str, Any]) -> None:
        decision = self._gate.evaluate(ToolCall(tool=self.name, args=arguments))
        if decision.effect is Effect.DENY:
            raise PolicyBlocked(decision)
        if decision.effect is Effect.ESCALATE:
            raise NeedsHumanApproval(decision)

    def _run(self, *args: Any, **kwargs: Any) -> Any:
        # LangChain calls tools positionally with a single tool-input argument; the gate
        # needs a mapping. Both shapes are folded into one for evaluation, with positional
        # arguments recorded under their index so a rule can still match on them.
        arguments = dict(kwargs)
        if args:
            if len(args) == 1 and isinstance(args[0], dict):
                arguments = {**args[0], **kwargs}
            else:
                arguments.update({f"arg{i}": a for i, a in enumerate(args)})
        self._gate_or_raise(arguments)
        return self._call(*args, **kwargs)

    def run(self, *args: Any, **kwargs: Any) -> Any:
        return self._run(*args, **kwargs)

    def invoke(self, input: Any, **kwargs: Any) -> Any:
        if isinstance(input, dict):
            return self._run(**input)
        return self._run(input)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self._run(*args, **kwargs)

    def __repr__(self) -> str:
        return f"GatedTool(name={self.name!r}, gate={self._gate.policy.default.value} default)"


def gated_tool(tool: Any, gate: Gate, *, name: str | None = None) -> GatedTool:
    """Wrap one tool. `gated_tools([...], gate)` wraps a list."""
    return GatedTool(tool, gate, name=name)


def gated_tools(tools: list[Any], gate: Gate) -> list[GatedTool]:
    """Wrap every tool in a list, preserving order."""
    return [GatedTool(tool, gate) for tool in tools]
