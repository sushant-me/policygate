"""The vocabulary: what a tool call is, and what a decision can be."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class Effect(str, Enum):
    """What the gate decided.

    ``ESCALATE`` is deliberately the default answer for anything the policy does not
    cover. The alternative — allowing what nobody wrote a rule about — turns every gap
    in the policy into a silent hole, and gaps are the normal state of a policy that is
    written by hand while a product changes.
    """

    ALLOW = "allow"
    DENY = "deny"
    ESCALATE = "escalate"


@dataclass(frozen=True)
class ToolCall:
    """One proposed action, as the gate sees it."""

    tool: str
    args: dict[str, Any] = field(default_factory=dict)
    #: Who or what proposed it — an agent id, a session id, a username.
    principal: str = "agent"
    #: Free-text intent, if the agent supplies one. Never trusted, only logged.
    intent: str = ""

    def digest(self) -> str:
        """A stable hash of the call, for the audit log and for dedup.

        Arguments are serialised with sorted keys so that two identical calls hash the
        same regardless of dict ordering.
        """
        payload = json.dumps(
            {"tool": self.tool, "args": self.args, "principal": self.principal},
            sort_keys=True, default=str,
        )
        return hashlib.sha256(payload.encode()).hexdigest()[:16]

    def render(self) -> str:
        """A one-line human-readable form, used in escalation messages."""
        args = ", ".join(f"{k}={v!r}" for k, v in sorted(self.args.items()))
        return f"{self.tool}({args})"


@dataclass(frozen=True)
class Decision:
    """The verdict, with enough context to be argued with.

    ``rule`` is the part that matters. A decision that cannot name the rule it came
    from is an opinion, and this package refuses to produce one: an evaluator that
    returns a verdict without a rule id is turned into an escalation.
    """

    effect: Effect
    reason: str
    rule: str | None = None
    decided_by: str = "rule"
    call: ToolCall | None = None
    #: Model-reported confidence, when a model was involved. Reported, never trusted.
    confidence: float | None = None

    def allowed(self) -> bool:
        return self.effect is Effect.ALLOW

    def blocked(self) -> bool:
        """True when the action must not proceed (deny or escalate)."""
        return self.effect in (Effect.DENY, Effect.ESCALATE)

    def __str__(self) -> str:
        call = self.call.render() if self.call else "?"
        rule = f" [{self.rule}]" if self.rule else ""
        return f"{self.effect.value.upper()}{rule} {call} — {self.reason}"
