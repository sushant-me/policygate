"""Evaluators: the deterministic one, the model one, and the contract between them.

The security-relevant decision in this file is that the two are not interchangeable.
A model evaluator's verdict is **only ever allowed to tighten** a decision:

* ``DENY`` from a model is honoured — refusing is always safe.
* ``ESCALATE`` from a model is honoured.
* ``ALLOW`` from a model is **not** honoured for a call no rule covers. It is recorded
  as evidence that the model wanted to allow it, and the outcome is an escalation.

That is the whole point of the package. The evaluation behind
``Edge-Native Semantic Firewall`` measured a locally served 3.8B model approving 23.5%
of the proposals the policy would have blocked — 46.2% in its JSON-only configuration.
A component with that error rate cannot be the authority for an irreversible action, so
here it is an advisor that can say no and can say "not sure", and cannot say yes. If you
want a class of action allowed, you write an allow rule, and that rule is reviewable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from .decision import Effect, ToolCall
from .policy import Policy, Rule


@dataclass(frozen=True)
class ModelVerdict:
    """What a model evaluator reports. `rule_id` is what makes it usable."""

    effect: Effect
    rule_id: str | None
    reason: str
    confidence: float | None = None


@runtime_checkable
class ModelEvaluator(Protocol):
    """The contract a model backend must satisfy."""

    def evaluate(self, call: ToolCall, policy: Policy) -> ModelVerdict:
        """Score one tool call against the policy.

        Raising is allowed and is treated as "the evaluator is unavailable", which the
        gate turns into its default (fail-closed) outcome. A backend that cannot answer
        must not answer.
        """
        ...


class RuleEvaluator:
    """The deterministic evaluator: no model, no ambiguity, no exceptions.

    This is the only evaluator that can produce ``ALLOW``, and only from a rule whose
    effect is ``ALLOW``.
    """

    name = "rule"

    def __init__(self, policy: Policy) -> None:
        self.policy = policy

    def match(self, call: ToolCall) -> Rule | None:
        return self.policy.first_match(call)


class StaticModel:
    """A model double for tests and for dry runs.

    Returns a fixed verdict, or raises if ``raises`` is set — which is how the
    fail-closed behaviour is tested: an unavailable evaluator must not become an
    approval.
    """

    name = "static-model"

    def __init__(
        self,
        effect: Effect = Effect.ESCALATE,
        rule_id: str | None = None,
        reason: str = "static test verdict",
        confidence: float | None = 0.5,
        raises: Exception | None = None,
    ) -> None:
        self.effect = effect
        self.rule_id = rule_id
        self.reason = reason
        self.confidence = confidence
        self.raises = raises
        self.calls: list[ToolCall] = []

    def evaluate(self, call: ToolCall, policy: Policy) -> ModelVerdict:
        self.calls.append(call)
        if self.raises is not None:
            raise self.raises
        return ModelVerdict(
            effect=self.effect, rule_id=self.rule_id,
            reason=self.reason, confidence=self.confidence,
        )


class CommandModel:
    """A model backend that shells out to a command and parses a strict reply.

    The reply format is three lines, because a parser that accepts loose output is a
    parser that accepts an injected verdict::

        decision: allow|deny|escalate
        rule: <rule id from the policy, or none>
        reason: <one line>

    Anything unparseable, any unknown decision word, or an absent rule id produces a
    verdict that the gate turns into an escalation — never an allow. The command is
    given the call as JSON on stdin.
    """

    name = "command-model"

    def __init__(self, argv: list[str], timeout: float = 30.0) -> None:
        self.argv = argv
        self.timeout = timeout

    def evaluate(self, call: ToolCall, policy: Policy) -> ModelVerdict:
        import json
        import subprocess

        payload = json.dumps(
            {
                "tool": call.tool,
                "args": call.args,
                "principal": call.principal,
                "rules": [
                    {"id": r.id, "effect": r.effect.value, "rationale": r.rationale}
                    for r in policy.rules
                ],
            },
            default=str,
        )
        try:
            completed = subprocess.run(
                self.argv, input=payload, capture_output=True, text=True,
                timeout=self.timeout, check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise RuntimeError(f"model backend unavailable: {exc}") from exc

        if completed.returncode != 0:
            raise RuntimeError(f"model backend exited {completed.returncode}")

        return self._parse(completed.stdout)

    @staticmethod
    def _parse(output: str) -> ModelVerdict:
        fields: dict[str, str] = {}
        for line in output.splitlines():
            key, sep, value = line.partition(":")
            if sep and key.strip().lower() in {"decision", "rule", "reason", "confidence"}:
                fields[key.strip().lower()] = value.strip()

        raw_decision = fields.get("decision", "").lower()
        try:
            effect = Effect(raw_decision)
        except ValueError:
            # An unparseable or unknown decision must not be guessed at.
            return ModelVerdict(
                effect=Effect.ESCALATE, rule_id=None,
                reason=f"unparseable model decision {raw_decision!r}; escalating",
            )

        rule_id = fields.get("rule") or None
        if rule_id and rule_id.lower() in {"none", "null", "-"}:
            rule_id = None

        confidence: float | None = None
        if "confidence" in fields:
            try:
                confidence = float(fields["confidence"])
            except ValueError:
                confidence = None

        return ModelVerdict(
            effect=effect,
            rule_id=rule_id,
            reason=fields.get("reason", "no reason given"),
            confidence=confidence,
        )
