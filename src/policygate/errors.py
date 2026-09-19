"""Exceptions a caller routes on.

Two different failures deserve two different reactions, and collapsing them into one
`PermissionError` is how an escalation ends up being logged as a bug and retried:

* :class:`PolicyBlocked` — a rule refused this. Do not retry, do not rephrase, do not
  ask a model to try again. Report it.
* :class:`NeedsHumanApproval` — the policy does not cover this call, or a human is
  required by a rule. The action is *pending*, not refused, and a person can approve it.

Both carry the decision, so a caller never has to re-evaluate to find out why.
"""

from __future__ import annotations

from .decision import Decision


class PolicyGateError(Exception):
    """Base class, so a caller can catch everything this package raises."""


class PolicyBlocked(PolicyGateError):
    """A rule denied the call. Irreversible as far as the agent is concerned."""

    def __init__(self, decision: Decision) -> None:
        self.decision = decision
        super().__init__(str(decision))


class NeedsHumanApproval(PolicyGateError):
    """The call is uncovered or explicitly human-gated. It is pending, not refused."""

    def __init__(self, decision: Decision) -> None:
        self.decision = decision
        super().__init__(str(decision))


class EvaluatorUnavailable(PolicyGateError):
    """The model evaluator could not answer, so the gate used its refusing default."""

    def __init__(self, decision: Decision) -> None:
        self.decision = decision
        super().__init__(str(decision))
