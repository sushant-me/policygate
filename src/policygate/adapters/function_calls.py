"""Gate OpenAI-style function calls: the `{"name": ..., "arguments": "{...}"}` shape.

Every provider emits this shape slightly differently (`arguments` is a JSON *string* in
OpenAI's API, an object in some others; Anthropic uses `input`). `parse_function_call`
normalises them, and — this is the part that matters — **a malformed argument string
escalates rather than raising or being coerced**.

A JSON parse error is exactly where an attacker would like you to be lenient: if a gate
cannot read the arguments, it cannot evaluate the call, and "cannot evaluate" must mean
"do not run", not "run with whatever we parsed so far".
"""

from __future__ import annotations

import json
from typing import Any, Callable

from ..decision import Effect, ToolCall
from ..errors import NeedsHumanApproval, PolicyBlocked
from ..gate import Gate

__all__ = ["gate_function_call", "parse_function_call", "execute_function_call"]


class MalformedFunctionCall(Exception):
    """The call could not be read. The gate escalates instead of guessing."""


def parse_function_call(raw: dict[str, Any]) -> ToolCall:
    """Normalise an OpenAI/Anthropic-shaped tool call into a `ToolCall`.

    Raises :class:`MalformedFunctionCall` when the arguments cannot be read; callers
    turn that into an escalation rather than a crash or a best-effort guess.
    """
    # OpenAI: {"function": {"name": ..., "arguments": "{\"a\": 1}"}}
    if isinstance(raw.get("function"), dict):
        raw = {**raw["function"], **{k: v for k, v in raw.items() if k != "function"}}

    name = raw.get("name") or raw.get("tool") or raw.get("tool_name")
    if not isinstance(name, str) or not name:
        raise MalformedFunctionCall(f"no tool name in {raw!r}")

    arguments = raw.get("arguments", raw.get("input", raw.get("args", {})))
    if isinstance(arguments, str):
        if not arguments.strip():
            arguments = {}
        else:
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError as exc:
                raise MalformedFunctionCall(
                    f"arguments for {name!r} are not valid JSON: {exc}"
                ) from exc
    if not isinstance(arguments, dict):
        raise MalformedFunctionCall(
            f"arguments for {name!r} must be an object, got {type(arguments).__name__}"
        )

    principal = raw.get("principal", "agent")
    return ToolCall(tool=name, args=arguments, principal=str(principal))


def gate_function_call(gate: Gate, raw: dict[str, Any]):
    """Evaluate one provider-shaped call and return the decision.

    Use when the caller wants the verdict (for logging, or to route it itself). Use
    :func:`execute_function_call` when the caller wants the tool actually invoked under
    the gate.
    """
    try:
        call = parse_function_call(raw)
    except MalformedFunctionCall as exc:
        # A call the gate cannot read is escalated, not executed and not crashed on.
        #
        # It is escalated *without* consulting the policy, and that is deliberate: a
        # rule that matches on tool name alone (the common shape for an allow rule) would
        # otherwise approve a call whose arguments were never parsed. My first version of
        # this passed the parse failure through as a normal argument and the
        # `allow-read-only` rule in the sample policy allowed it — a test caught it.
        return gate.escalate(
            ToolCall(tool=str(raw.get("name", "<unparseable>")), args={}),
            reason=f"function call could not be parsed, so it cannot be evaluated: {exc}",
        )
    return gate.evaluate(call)


def execute_function_call(
    gate: Gate, raw: dict[str, Any], dispatch: dict[str, Callable[..., Any]]
) -> Any:
    """Run one provider-shaped call if the gate allows it, else raise.

    `dispatch` maps tool name to the callable that implements it. A name the gate allows
    but `dispatch` does not have raises `KeyError` — that is a wiring bug in the caller,
    not a policy decision, and conflating the two hides real failures.
    """
    decision = gate_function_call(gate, raw)
    if decision.effect is Effect.DENY:
        raise PolicyBlocked(decision)
    if decision.effect is Effect.ESCALATE:
        raise NeedsHumanApproval(decision)

    call = parse_function_call(raw)
    return dispatch[call.tool](**call.args)
