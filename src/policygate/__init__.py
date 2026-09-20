"""Fail-closed policy gate for autonomous agent tool calls.

The design rule of this package, in one sentence: **a language model is never the last
word.** Hard rules are deterministic and are evaluated first; the model is consulted
only where the policy is genuinely ambiguous; and any model answer that cannot be tied
to a rule escalates to a human instead of being trusted.

That ordering exists because of a measurement, not a preference. In the evaluation
behind ``sushant-me/Edge-Native_Semantic_Firewall_``, a locally served 3.8B model acting
as the policy evaluator approved **23.5%** of proposals the policy would have blocked —
46.2% in its JSON-only configuration — and roughly one verdict in ten changed between
identical runs. A component with those numbers is useful for ranking ambiguous cases and
unusable as an authority.
"""

from .audit import AuditLog
from .decision import Decision, Effect, ToolCall
from .errors import EvaluatorUnavailable, NeedsHumanApproval, PolicyBlocked, PolicyGateError
from .evaluators import ModelEvaluator, RuleEvaluator, StaticModel
from .gate import Gate, PolicyError, load_policy

__all__ = [
    "AuditLog",
    "Decision",
    "Effect",
    "EvaluatorUnavailable",
    "Gate",
    "ModelEvaluator",
    "NeedsHumanApproval",
    "PolicyBlocked",
    "PolicyError",
    "PolicyGateError",
    "RuleEvaluator",
    "StaticModel",
    "ToolCall",
    "load_policy",
]

__version__ = "0.1.1"
