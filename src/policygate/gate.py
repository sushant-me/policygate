"""The gate: the one function everything else exists to support.

``Gate.evaluate(call)`` is the whole interface. The order of operations is the security
property, so it is spelled out here rather than left to the reader:

1. **Rules first, by severity.** A matching ``deny`` rule refuses immediately. A matching
   ``escalate`` rule escalates. A matching ``allow`` rule allows — this is the only path
   in the package that produces an ``ALLOW``.
2. **Nothing matched → the model may be consulted, but only to tighten.** A model
   ``DENY`` is honoured; a model ``ESCALATE`` is honoured; a model ``ALLOW`` is recorded
   and then escalated anyway (see `evaluators` for why).
3. **A model that fails, times out, or answers unparseably is not an approval.** The
   call falls through to the policy default, which is ``escalate`` unless the author
   explicitly and visibly chose otherwise.

Every step writes to the audit log, including the decisions that did not come from a
rule, because the interesting entries in an incident review are the ones where the
policy had nothing to say.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from .audit import AuditLog
from .decision import Decision, Effect, ToolCall
from .evaluators import ModelEvaluator
from .policy import PolicyError, load_policy  # re-exported for convenience


class Gate:
    """Evaluates tool calls against a policy, fail-closed by construction."""

    def __init__(
        self,
        policy,
        model: ModelEvaluator | None = None,
        audit: AuditLog | None = None,
    ) -> None:
        self.policy = policy
        self.model = model
        self.audit = audit if audit is not None else AuditLog.null()

    # -- construction helpers ---------------------------------------------

    @classmethod
    def from_file(cls, path: str | Path, **kwargs: Any) -> "Gate":
        return cls(load_policy(path), **kwargs)

    # -- the interface -----------------------------------------------------

    def evaluate(self, call: ToolCall) -> Decision:
        started = time.perf_counter()

        rule = self.policy.first_match(call)
        if rule is not None:
            decision = Decision(
                effect=rule.effect,
                reason=rule.rationale,
                rule=rule.id,
                decided_by="rule",
                call=call,
            )
            return self._record(decision, started)

        # Nothing matched: either consult the model, or fall through to the default.
        if self.policy.consult_model and self.model is not None:
            decision = self._consult_model(call)
            if decision is not None:
                return self._record(decision, started)

        decision = Decision(
            effect=self.policy.default,
            reason=(
                "no rule matches this call, and the model was not consulted or "
                "returned nothing usable"
            ),
            rule=None,
            decided_by="default",
            call=call,
        )
        return self._record(decision, started)

    # -- internals ---------------------------------------------------------

    def _consult_model(self, call: ToolCall) -> Decision | None:
        assert self.model is not None
        try:
            verdict = self.model.evaluate(call, self.policy)
        except Exception as exc:  # noqa: BLE001 - any failure means "unavailable"
            # Deliberately broad: a backend that raises, times out, or is misconfigured
            # must not become an approval. It becomes the policy default.
            self.audit.note(
                "model_unavailable", call=call, detail=f"{type(exc).__name__}: {exc}"
            )
            return Decision(
                effect=self.policy.default,
                reason=f"model evaluator unavailable ({type(exc).__name__}); defaulting",
                rule=None,
                decided_by="default",
                call=call,
            )

        known_ids = {r.id for r in self.policy.rules}
        cited = verdict.rule_id if verdict.rule_id in known_ids else None

        if verdict.effect is Effect.DENY:
            return Decision(
                effect=Effect.DENY,
                reason=f"model refused: {verdict.reason}",
                rule=cited,
                decided_by="model",
                call=call,
                confidence=verdict.confidence,
            )

        if verdict.effect is Effect.ALLOW:
            # The rule this package is built on: a model cannot allow what the policy
            # does not cover. Record the attempt, then escalate.
            self.audit.note(
                "model_allow_rejected",
                call=call,
                detail="model returned allow for a call no rule covers",
            )
            cited_note = f" (it cited {verdict.rule_id!r})" if verdict.rule_id else ""
            return Decision(
                effect=Effect.ESCALATE,
                reason=(
                    "model wanted to allow this call, but no rule covers it; the model "
                    f"is an advisor and cannot authorise an uncovered action{cited_note}"
                ),
                # Deliberately no rule: attributing this escalation to the rule the
                # model named would read as if that rule had produced it.
                rule=None,
                decided_by="model-allow-rejected",
                call=call,
                confidence=verdict.confidence,
            )

        return Decision(
            effect=Effect.ESCALATE,
            reason=f"model escalated: {verdict.reason}",
            rule=cited,
            decided_by="model",
            call=call,
            confidence=verdict.confidence,
        )

    def _record(self, decision: Decision, started: float) -> Decision:
        self.audit.record(decision, latency_ms=(time.perf_counter() - started) * 1000)
        return decision

    # -- convenience -------------------------------------------------------

    def escalate(self, call: ToolCall, reason: str, decided_by: str = "unparseable") -> Decision:
        """Escalate without consulting the policy, and record it.

        For the case where the gate cannot evaluate a call at all — arguments that will
        not parse, a shape it does not recognise. **This must not fall through to rule
        matching**: a rule that matches on tool name alone would happily allow a call
        whose arguments the gate never managed to read, which is exactly the input an
        attacker controls. A call that cannot be read is not a call that can be allowed.
        """
        return self._record(
            Decision(
                effect=Effect.ESCALATE,
                reason=reason,
                rule=None,
                decided_by=decided_by,
                call=call,
            ),
            started=time.perf_counter(),
        )

    def check(self, tool: str, **args: Any) -> Decision:
        """Evaluate a call written as keyword arguments, for tests and scripts."""
        return self.evaluate(ToolCall(tool=tool, args=args))

    def explain(self, call: ToolCall) -> str:
        """A human-readable account of how this call would be decided."""
        decision = self.evaluate(call)
        lines = [f"call:     {call.render()}"]
        for rule in self.policy.ordered_rules():
            mark = "MATCH" if rule.matches(call) else "     "
            lines.append(f"  [{mark}] {rule.effect.value:<8} {rule.id} — {rule.rationale}")
        lines.append(f"decision: {decision}")
        return "\n".join(lines)
