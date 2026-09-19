"""Adapter tests: the gate must stop the side effect, not just report on it.

The property under test in every adapter is the same, and it is the only one that
matters: **a call the gate does not allow must not reach the underlying tool.** A counter
is incremented by the tool itself, so a passing test means the function was never
entered — not that its return value was ignored.

A gate that returns a decision nobody acts on is worse than no gate, because it produces
an audit log that says the system is protected.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from policygate import (  # noqa: E402
    AuditLog, Effect, Gate, NeedsHumanApproval, PolicyBlocked, StaticModel, ToolCall, load_policy,
)
from policygate.adapters import (  # noqa: E402
    GatedMCPServer, execute_function_call, gate_function_call, gated_tool, gated_tools,
    parse_function_call,
)
from policygate.adapters.langchain import GatedTool  # noqa: E402
from policygate.adapters.mcp import ToolNameCollision  # noqa: E402

POLICY = ROOT / "examples" / "policy.toml"


@pytest.fixture()
def policy():
    return load_policy(POLICY)


@pytest.fixture()
def gate(policy):
    return Gate(policy, audit=AuditLog())


class Counter:
    """A tool that records that it was entered."""

    def __init__(self, result="ok"):
        self.calls = 0
        self.result = result

    def __call__(self, *args, **kwargs):
        self.calls += 1
        return self.result


# -- MCP ---------------------------------------------------------------------


class FakeMCPServer:
    def __init__(self, tools, impl=None):
        self._tools = [{"name": n} for n in tools]
        self.impl = impl or {}
        self.calls = []

    def list_tools(self):
        return self._tools

    def call_tool(self, name, arguments):
        self.calls.append((name, arguments))
        return self.impl.get(name, lambda **_: "done")(**arguments)


def test_mcp_allowed_call_is_forwarded(gate) -> None:
    server = FakeMCPServer(["lookup_order"])
    gated = GatedMCPServer(server, gate)
    assert gated.call_tool("lookup_order", {"order_id": "A-1"}) == "done"
    assert server.calls == [("lookup_order", {"order_id": "A-1"})]


def test_mcp_denied_call_never_reaches_the_server(gate) -> None:
    counter = Counter()
    server = FakeMCPServer(["run_shell"], impl={"run_shell": counter})
    gated = GatedMCPServer(server, gate)

    with pytest.raises(PolicyBlocked):
        gated.call_tool("run_shell", {"command": "echo x >> /etc/sudoers"})

    # The decisive assertion: the server was not called at all.
    assert server.calls == [], "a denied call must not be forwarded"
    assert counter.calls == 0


def test_mcp_escalation_never_reaches_the_server(gate) -> None:
    counter = Counter()
    server = FakeMCPServer(["send_email"], impl={"send_email": counter})
    gated = GatedMCPServer(server, gate)

    with pytest.raises(NeedsHumanApproval) as caught:
        gated.call_tool("send_email", {"to": "a@b.c", "body": "refund"})

    assert server.calls == []
    assert counter.calls == 0
    assert caught.value.decision.effect is Effect.ESCALATE
    assert "human" in caught.value.decision.reason


def test_mcp_refuses_reserved_tool_names_at_construction(gate) -> None:
    server = FakeMCPServer(["set_model_response", "lookup_order"])
    with pytest.raises(ToolNameCollision) as caught:
        GatedMCPServer(server, gate)
    assert "set_model_response" in str(caught.value)


def test_mcp_reserved_names_can_be_accepted_explicitly(gate) -> None:
    server = FakeMCPServer(["google_search"])
    gated = GatedMCPServer(server, gate, allow_reserved=True, reserved=frozenset({"google_search"}))
    assert gated.list_tools() == [{"name": "google_search"}]


def test_mcp_adapter_records_every_decision(gate) -> None:
    server = FakeMCPServer(["lookup_order", "send_email"])
    gated = GatedMCPServer(server, gate)
    gated.call_tool("lookup_order", {"order_id": "A-1"})
    with pytest.raises(NeedsHumanApproval):
        gated.call_tool("send_email", {"to": "a@b.c"})
    assert len(gate.audit.decisions()) == 2


def test_mcp_dispatch_callable_works(gate) -> None:
    gated = GatedMCPServer(FakeMCPServer(["lookup_order"]), gate)
    dispatch = gated.as_dispatch()
    assert dispatch("lookup_order", {"order_id": "A-1"}) == "done"


# -- LangChain-style tools ---------------------------------------------------


class FakeLangChainTool:
    """The shape a LangChain tool presents: name, description, a callable."""

    def __init__(self, name, fn, description="a tool"):
        self.name = name
        self.description = description
        self._run_fn = fn

    def _run(self, *args, **kwargs):
        return self._run_fn(*args, **kwargs)


def test_langchain_allowed_tool_runs(gate) -> None:
    counter = Counter("order A-1")
    tool = gated_tool(FakeLangChainTool("lookup_order", counter), gate)
    assert tool.run(order_id="A-1") == "order A-1"
    assert counter.calls == 1


def test_langchain_denied_tool_does_not_run(gate) -> None:
    counter = Counter()
    tool = gated_tool(FakeLangChainTool("run_shell", counter), gate)
    with pytest.raises(PolicyBlocked):
        tool.run(command="chmod 777 /")
    assert counter.calls == 0, "the tool body must not execute"


def test_langchain_invoke_with_dict_input_is_gated(gate) -> None:
    counter = Counter("ok")
    tool = gated_tool(FakeLangChainTool("lookup_order", counter), gate)
    assert tool.invoke({"order_id": "A-2"}) == "ok"
    assert counter.calls == 1


def test_langchain_wrapper_preserves_identity(gate) -> None:
    tool = gated_tool(FakeLangChainTool("summarise", Counter(), "summarise text"), gate)
    assert tool.name == "summarise"
    assert tool.description == "summarise text"
    assert isinstance(tool, GatedTool)


def test_langchain_gated_tools_wraps_a_list(gate) -> None:
    tools = gated_tools([FakeLangChainTool("lookup_order", Counter()), FakeLangChainTool("summarise", Counter())], gate)
    assert [t.name for t in tools] == ["lookup_order", "summarise"]


def test_langchain_escalation_raises_pending_not_denied(gate) -> None:
    counter = Counter()
    tool = gated_tool(FakeLangChainTool("send_email", counter), gate)
    with pytest.raises(NeedsHumanApproval):
        tool.run(to="a@b.c", body="hi")
    assert counter.calls == 0


def test_langchain_positional_arguments_are_still_evaluated(gate) -> None:
    """A tool called positionally must not bypass the gate by hiding its arguments."""
    counter = Counter()
    policy = load_policy({
        "rules": [
            {"id": "deny-arg0", "effect": "deny", "tool": "run_shell",
             "arg_matches": {"arg0": "sudoers"},
             "rationale": "positional arguments must be visible to rules"},
            {"id": "allow-shell", "effect": "allow", "tool": "run_shell",
             "rationale": "otherwise allow shell so the deny is what stops it"},
        ],
    })
    tool = gated_tool(FakeLangChainTool("run_shell", counter), Gate(policy))
    with pytest.raises(PolicyBlocked):
        tool.run("echo x >> /etc/sudoers")
    assert counter.calls == 0


# -- provider-shaped function calls -----------------------------------------


def test_function_call_parsing_handles_the_openai_shape() -> None:
    call = parse_function_call({
        "function": {"name": "lookup_order", "arguments": json.dumps({"order_id": "A-1"})}
    })
    assert call.tool == "lookup_order"
    assert call.args == {"order_id": "A-1"}


def test_function_call_parsing_handles_a_dict_arguments_field() -> None:
    call = parse_function_call({"name": "lookup_order", "arguments": {"order_id": "A-9"}})
    assert call.args == {"order_id": "A-9"}


def test_function_call_parsing_handles_an_empty_argument_string() -> None:
    assert parse_function_call({"name": "ping", "arguments": ""}).args == {}


def test_malformed_arguments_escalate_instead_of_crashing(gate) -> None:
    decision = gate_function_call(gate, {"name": "lookup_order", "arguments": "{not json"})
    assert decision.effect is Effect.ESCALATE
    assert decision.decided_by == "unparseable"
    assert "could not be parsed" in decision.reason


def test_malformed_arguments_are_not_allowable_by_a_name_only_rule(gate) -> None:
    """The hole a test found in my first version of this adapter.

    `allow-read-only` matches `lookup_order` on the tool name alone. If a parse failure
    is passed through as a normal argument, that rule approves a call whose arguments
    were never read — so the gate must escalate a call it cannot parse *without*
    consulting the policy at all.
    """
    decision = gate_function_call(gate, {"name": "lookup_order", "arguments": "{"})
    assert decision.effect is not Effect.ALLOW
    assert decision.rule is None, "an unparseable call must not be attributed to a rule"


def test_a_call_with_no_name_escalates(gate) -> None:
    decision = gate_function_call(gate, {"arguments": "{}"})
    assert decision.effect is Effect.ESCALATE


def test_execute_function_call_runs_an_allowed_tool(gate) -> None:
    counter = Counter("A-1 details")
    result = execute_function_call(
        gate,
        {"name": "lookup_order", "arguments": json.dumps({"order_id": "A-1"})},
        {"lookup_order": counter},
    )
    assert result == "A-1 details"
    assert counter.calls == 1


def test_execute_function_call_does_not_run_a_denied_tool(gate) -> None:
    counter = Counter()
    with pytest.raises(PolicyBlocked):
        execute_function_call(
            gate,
            {"name": "run_shell", "arguments": json.dumps({"command": "echo >> /etc/sudoers"})},
            {"run_shell": counter},
        )
    assert counter.calls == 0


def test_execute_function_call_does_not_run_an_escalated_tool(gate) -> None:
    counter = Counter()
    with pytest.raises(NeedsHumanApproval):
        execute_function_call(
            gate,
            {"name": "send_email", "arguments": json.dumps({"to": "a@b.c", "body": "hi"})},
            {"send_email": counter},
        )
    assert counter.calls == 0


def test_a_dispatch_bug_is_not_disguised_as_a_policy_decision(gate) -> None:
    """An allowed call with no implementation is a wiring bug and must say so."""
    with pytest.raises(KeyError):
        execute_function_call(
            gate,
            {"name": "lookup_order", "arguments": json.dumps({"order_id": "A-1"})},
            {},  # nothing registered
        )


def test_model_unavailable_still_stops_the_call(policy) -> None:
    """The end-to-end fail-closed path, through an adapter."""
    counter = Counter()
    gate = Gate(policy, model=StaticModel(raises=ConnectionError("no backend")))
    gated = gated_tool(FakeLangChainTool("rotate_api_key", counter), gate)
    with pytest.raises(NeedsHumanApproval):
        gated.run(service="billing")
    assert counter.calls == 0


def test_tool_call_digest_is_stable_across_dict_order() -> None:
    a = ToolCall(tool="t", args={"x": 1, "y": 2})
    b = ToolCall(tool="t", args={"y": 2, "x": 1})
    assert a.digest() == b.digest()
