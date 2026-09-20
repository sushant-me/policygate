"""Parsing and matching for the policy file.

A policy is TOML: rules, each with an id, an effect, a matcher, and a rationale. Three
decisions here are load-bearing and each is tested:

1. **Rules are evaluated by severity, not by file order.** All ``deny`` rules are
   considered before any ``escalate``, and those before any ``allow``. A broad allow
   written at the top of the file therefore cannot shadow a narrow deny written below
   it. Ordering by position is the obvious implementation and it is the wrong one: it
   makes the policy's meaning depend on line order, which review cannot see.
2. **Every rule needs a rationale.** A rule nobody can explain is a rule nobody can
   safely change, and the rationale is what a human reads when the gate escalates.
3. **Loading is strict.** An unparseable regex, a duplicate id, an unknown effect or a
   policy with no rules raises rather than loading a partial policy. A gate that starts
   with half a policy is worse than one that refuses to start.
"""

from __future__ import annotations

import re
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .decision import Effect, ToolCall

#: Severity order used to sort rules before matching. Deny first, always.
_EFFECT_ORDER = {Effect.DENY: 0, Effect.ESCALATE: 1, Effect.ALLOW: 2}


class PolicyError(Exception):
    """Raised when a policy cannot be loaded. The gate refuses to start."""


@dataclass(frozen=True)
class Rule:
    id: str
    effect: Effect
    rationale: str
    tool_patterns: tuple[str, ...] = ("*",)
    arg_matches: dict[str, re.Pattern[str]] = None  # type: ignore[assignment]
    principal_patterns: tuple[str, ...] = ("*",)

    def __post_init__(self) -> None:
        if self.arg_matches is None:
            object.__setattr__(self, "arg_matches", {})

    # -- matching ----------------------------------------------------------

    @staticmethod
    def _glob_match(pattern: str, value: str) -> bool:
        if pattern == "*":
            return True
        if "*" not in pattern:
            return pattern == value
        # re.DOTALL, and fullmatch rather than an anchored match.
        #
        # Without DOTALL a `.` will not match a newline, so `send_*` failed to match the value
        # "send_\nemail" and the deny rule was silently skipped - the call fell through to a
        # broader allow and proceeded with no human in the loop. A deny rule that anything can
        # step around by embedding a newline is not a deny rule. The value is attacker-influenced:
        # tool names come from the MCP server being gated, and principals can be caller-supplied.
        #
        # fullmatch also removes the `$`-before-a-trailing-newline quirk, which made `delete_*`
        # match "delete_file\n" - a pattern should match the string it was written for and
        # nothing that merely resembles it.
        regex = re.escape(pattern).replace(r"\*", ".*")
        return re.fullmatch(regex, value, re.DOTALL) is not None

    def matches(self, call: ToolCall) -> bool:
        if not any(self._glob_match(p, call.tool) for p in self.tool_patterns):
            return False
        if not any(self._glob_match(p, call.principal) for p in self.principal_patterns):
            return False
        for name, pattern in self.arg_matches.items():
            if name not in call.args:
                # A rule that names an argument only applies when that argument is
                # present. Absence is not a match: otherwise every "deny if command
                # contains X" rule would fire on calls that have no command at all.
                return False
            if pattern.search(str(call.args[name])) is None:
                return False
        return True


@dataclass(frozen=True)
class Policy:
    rules: tuple[Rule, ...]
    #: Effect for a call no rule matches. Defaults to escalate; see `load_policy`.
    default: Effect = Effect.ESCALATE
    #: Whether a model evaluator may be consulted for unmatched/ambiguous calls.
    consult_model: bool = True

    def ordered_rules(self) -> tuple[Rule, ...]:
        """Rules sorted by severity, preserving order within an effect."""
        return tuple(sorted(self.rules, key=lambda r: _EFFECT_ORDER[r.effect]))

    def first_match(self, call: ToolCall) -> Rule | None:
        for rule in self.ordered_rules():
            if rule.matches(call):
                return rule
        return None


def _as_patterns(value: Any, field: str, rule_id: str) -> tuple[str, ...]:
    if isinstance(value, str):
        return (value,)
    if isinstance(value, list) and all(isinstance(v, str) for v in value):
        return tuple(value)
    raise PolicyError(f"rule {rule_id!r}: {field} must be a string or a list of strings")


def _parse_rule(raw: dict[str, Any]) -> Rule:
    rule_id = raw.get("id")
    if not isinstance(rule_id, str) or not rule_id.strip():
        raise PolicyError("every rule needs a non-empty string 'id'")

    try:
        effect = Effect(str(raw.get("effect", "")).lower())
    except ValueError as exc:
        valid = ", ".join(e.value for e in Effect)
        raise PolicyError(
            f"rule {rule_id!r}: 'effect' must be one of {valid}, got {raw.get('effect')!r}"
        ) from exc

    rationale = raw.get("rationale")
    if not isinstance(rationale, str) or len(rationale.strip()) < 10:
        raise PolicyError(
            f"rule {rule_id!r}: needs a 'rationale' of at least 10 characters. A rule "
            "nobody can explain is a rule nobody can safely change."
        )

    arg_matches: dict[str, re.Pattern[str]] = {}
    raw_args = raw.get("arg_matches", {})
    if not isinstance(raw_args, dict):
        raise PolicyError(f"rule {rule_id!r}: 'arg_matches' must be a table")
    for name, pattern in raw_args.items():
        if not isinstance(pattern, str):
            raise PolicyError(f"rule {rule_id!r}: arg_matches.{name} must be a regex string")
        try:
            arg_matches[name] = re.compile(pattern)
        except re.error as exc:
            raise PolicyError(
                f"rule {rule_id!r}: arg_matches.{name} is not a valid regex: {exc}"
            ) from exc

    rule = Rule(
        id=rule_id.strip(),
        effect=effect,
        rationale=rationale.strip(),
        tool_patterns=_as_patterns(raw.get("tool", "*"), "tool", rule_id),
        arg_matches=arg_matches,
        principal_patterns=_as_patterns(raw.get("principal", "*"), "principal", rule_id),
    )
    # Force a match evaluation against an empty call so a malformed pattern surfaces at
    # load time rather than at the first tool call in production.
    rule.matches(ToolCall(tool="__load_check__"))
    return rule


def load_policy(source: str | Path | dict[str, Any]) -> Policy:
    """Load and validate a policy from a path or an already-parsed mapping."""
    if isinstance(source, dict):
        raw = source
    else:
        path = Path(source)
        try:
            raw = tomllib.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise PolicyError(f"policy file not found: {path}") from exc
        except tomllib.TOMLDecodeError as exc:
            raise PolicyError(f"{path} is not valid TOML: {exc}") from exc

    raw_rules = raw.get("rules")
    if not isinstance(raw_rules, list) or not raw_rules:
        raise PolicyError("a policy must define at least one [[rules]] entry")
    rules = tuple(_parse_rule(r) for r in raw_rules)

    ids = [r.id for r in rules]
    duplicates = {i for i in ids if ids.count(i) > 1}
    if duplicates:
        raise PolicyError(f"duplicate rule ids: {', '.join(sorted(duplicates))}")

    gate = raw.get("gate", {})
    if not isinstance(gate, dict):
        raise PolicyError("'gate' must be a table")

    default = Effect(str(gate.get("default", "escalate")).lower())
    if default is Effect.ALLOW and not gate.get("allow_unmatched_acknowledged_unsafe"):
        raise PolicyError(
            "gate.default = 'allow' lets through anything no rule covers, which is the "
            "failure this gate exists to prevent. If you really want it, also set "
            "allow_unmatched_acknowledged_unsafe = true."
        )

    return Policy(
        rules=rules,
        default=default,
        consult_model=bool(gate.get("consult_model", True)),
    )
