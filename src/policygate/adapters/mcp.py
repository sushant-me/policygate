"""Gate an MCP-style server: check the tool list, then gate every call.

Two different jobs, both of which matter before an agent talks to an MCP server:

1. **At construction**, refuse a server whose tool names collide with names the agent
   framework reserves for itself. This is the bug class I filed against three of Google's
   agent frameworks: in-model built-ins (`google_search`, `set_model_response`, `finish`)
   live outside the tool table, so a server-supplied tool with that name is never seen by
   the duplicate-name guard and wins dispatch. Nothing downstream can detect it — by the
   time a call happens, the name is simply taken.
2. **At call time**, evaluate the call against the policy before forwarding it.

The adapter is duck-typed: the server needs `list_tools()` (or a `tools` attribute) and
`call_tool(name, arguments)`. No MCP SDK import, so this works with any client.
"""

from __future__ import annotations

from typing import Any, Callable, Iterable

from ..decision import Effect, ToolCall
from ..errors import NeedsHumanApproval, PolicyBlocked
from ..gate import Gate

#: Names that agent frameworks install outside the normal tool table. A server tool that
#: takes one of these is not "a tool with a popular name" — it is a collision with a
#: framework primitive, and it is refused rather than renamed.
RESERVED_TOOL_NAMES = frozenset({
    "finish",
    "set_model_response",
    "transfer_to_agent",
    "task_completed",
    "google_search",
    "google_maps",
    "google_maps_grounding",
    "url_context",
    "code_execution",
    "vertex_ai_search",
    "load_artifacts",
    "load_memory",
    "list_skills",
    "load_skill",
    "load_skill_resource",
    "exit_loop",
})


class ToolNameCollision(Exception):
    """A server tool name collides with a framework-reserved name."""


class GatedMCPServer:
    """An MCP-style server with a policy gate in front of every call."""

    def __init__(
        self,
        server: Any,
        gate: Gate,
        *,
        reserved: Iterable[str] = RESERVED_TOOL_NAMES,
        allow_reserved: bool = False,
    ) -> None:
        self.server = server
        self.gate = gate
        self.reserved = frozenset(reserved)
        self.allow_reserved = allow_reserved
        self._check_tool_names()

    # -- tool inventory ----------------------------------------------------

    def _tool_names(self) -> list[str]:
        if hasattr(self.server, "list_tools"):
            tools = self.server.list_tools()
            return [
                (t.get("name") if isinstance(t, dict) else getattr(t, "name", None))
                for t in tools
            ]
        tools = getattr(self.server, "tools", [])
        return [
            (t.get("name") if isinstance(t, dict) else getattr(t, "name", None))
            for t in tools
        ]

    def _check_tool_names(self) -> None:
        if self.allow_reserved:
            return
        collisions = sorted({n for n in self._tool_names() if n in self.reserved})
        if collisions:
            raise ToolNameCollision(
                "server advertises tool name(s) reserved by the agent framework: "
                + ", ".join(collisions)
                + ". These live outside the normal tool table, so a collision silently "
                "displaces a framework primitive rather than raising a duplicate-name "
                "error. Rename them on the server, or pass allow_reserved=True to "
                "accept the risk explicitly."
            )

    def list_tools(self) -> Any:
        """The server's tools, once the names have been checked."""
        if hasattr(self.server, "list_tools"):
            return self.server.list_tools()
        return getattr(self.server, "tools", [])

    # -- the gated call path ----------------------------------------------

    def call_tool(self, name: str, arguments: dict[str, Any] | None = None) -> Any:
        arguments = dict(arguments or {})
        decision = self.gate.evaluate(ToolCall(tool=name, args=arguments))

        if decision.effect is Effect.DENY:
            raise PolicyBlocked(decision)
        if decision.effect is Effect.ESCALATE:
            # Nothing is forwarded: an escalation is a pause, and a paused action that
            # still happened is not paused.
            raise NeedsHumanApproval(decision)

        assert decision.effect is Effect.ALLOW
        return self.server.call_tool(name, arguments)

    # -- convenience -------------------------------------------------------

    def as_dispatch(self) -> Callable[[str, dict[str, Any]], Any]:
        """A `(name, arguments) -> result` callable, for wiring into an agent loop."""
        return self.call_tool

    def __repr__(self) -> str:
        return f"GatedMCPServer({len(self.list_tools())} tools, gate={self.gate.policy.default.value} default)"
