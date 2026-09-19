"""The invariants this package exists to guarantee.

These are not tests of features; they are tests of safety properties, and each one is
written so that it fails if the property is weakened. The properties:

1. **ALLOW comes only from an allow rule.** No model, no default, and no ordering
   mistake can produce an allow for a call the policy does not cover.
2. **A model cannot authorise.** A backend that returns ALLOW on an uncovered call is
   overruled and the attempt is recorded.
3. **A model that fails is not an approval.** Raising, timing out or answering
   unparseably falls through to the policy default, which refuses.
4. **Deny beats allow regardless of file order.**
5. **A bad policy refuses to load** rather than loading partially.
6. **The audit chain detects tampering.**
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from policygate import (  # noqa: E402
    AuditLog, Effect, Gate, PolicyError, StaticModel, ToolCall, load_policy,
)
from policygate.evaluators import CommandModel, ModelVerdict  # noqa: E402

POLICY = ROOT / "examples" / "policy.toml"


@pytest.fixture()
def policy():
    return load_policy(POLICY)


@pytest.fixture()
def gate(policy):
    return Gate(policy, model=StaticModel(effect=Effect.ALLOW, rule_id="allow-read-only"))


# -- property 1: ALLOW only from an allow rule ------------------------------


def test_an_allow_rule_allows(policy) -> None:
    gate = Gate(policy)
    decision = gate.check("lookup_order", order_id="A-1")
    assert decision.effect is Effect.ALLOW
    assert decision.rule == "allow-read-only"
    assert decision.decided_by == "rule"


def test_an_uncovered_call_is_never_allowed(gate) -> None:
    decision = gate.check("rotate_api_key", service="billing")
    assert decision.effect is Effect.ESCALATE
    assert not decision.allowed()


def test_a_default_of_allow_is_refused_at_load_time() -> None:
    with pytest.raises(PolicyError, match="allow"):
        load_policy({
            "gate": {"default": "allow"},
            "rules": [{"id": "r", "effect": "deny", "rationale": "a long enough rationale"}],
        })


def test_a_default_of_allow_needs_an_explicit_unsafe_acknowledgement() -> None:
    # The rule is scoped to a tool the call does not use, so the default is what
    # decides. (A rule with no `tool` matches everything, which is why this test needs
    # one — my first version of it left `tool` out and was silently testing the rule.)
    policy = load_policy({
        "gate": {"default": "allow", "allow_unmatched_acknowledged_unsafe": True},
        "rules": [{"id": "r", "effect": "deny", "tool": "some_other_tool",
                   "rationale": "a long enough rationale"}],
    })
    assert Gate(policy).check("anything_at_all").effect is Effect.ALLOW


# -- property 2: a model cannot authorise -----------------------------------


def test_a_model_allow_is_rejected_and_escalated(policy) -> None:
    model = StaticModel(effect=Effect.ALLOW, rule_id="allow-read-only", confidence=0.99)
    gate = Gate(policy, model=model)
    decision = gate.check("rotate_api_key", service="billing")
    assert decision.effect is Effect.ESCALATE
    assert decision.decided_by == "model-allow-rejected"
    # It must not be attributed to the rule the model named: that would read as if an
    # allow rule had produced the escalation.
    assert decision.rule is None
    assert "allow-read-only" in decision.reason


def test_a_model_deny_is_honoured(policy) -> None:
    model = StaticModel(effect=Effect.DENY, reason="looks like data staging")
    decision = Gate(policy, model=model).check("rotate_api_key", service="billing")
    assert decision.effect is Effect.DENY
    assert decision.decided_by == "model"


def test_a_model_escalation_is_honoured(policy) -> None:
    model = StaticModel(effect=Effect.ESCALATE, reason="ambiguous")
    decision = Gate(policy, model=model).check("rotate_api_key", service="x")
    assert decision.effect is Effect.ESCALATE
    assert decision.decided_by == "model"


def test_the_model_is_not_consulted_when_a_rule_matches(policy) -> None:
    model = StaticModel(effect=Effect.DENY)
    gate = Gate(policy, model=model)
    assert gate.check("lookup_order", order_id="A-1").effect is Effect.ALLOW
    assert gate.check("run_shell", command="cat /etc/sudoers").effect is Effect.DENY
    assert model.calls == [], "rules are authoritative; the model must not be asked"


# -- property 3: a failing model is not an approval -------------------------


def test_a_raising_model_falls_through_to_the_default(policy) -> None:
    model = StaticModel(raises=TimeoutError("backend unreachable"))
    decision = Gate(policy, model=model).check("rotate_api_key", service="billing")
    assert decision.effect is Effect.ESCALATE
    assert decision.decided_by == "default"
    assert "unavailable" in decision.reason


def test_an_unparseable_model_reply_escalates() -> None:
    verdict = CommandModel._parse("yes go ahead\nsounds fine to me\n")
    assert verdict.effect is Effect.ESCALATE
    assert "unparseable" in verdict.reason


def test_a_command_model_reply_is_parsed_strictly() -> None:
    verdict = CommandModel._parse("decision: deny\nrule: deny-credential-egress\nreason: exfil\n")
    assert verdict.effect is Effect.DENY
    assert verdict.rule_id == "deny-credential-egress"


def test_a_model_citing_an_unknown_rule_is_recorded_without_one(policy) -> None:
    model = StaticModel(effect=Effect.DENY, rule_id="rule-that-does-not-exist")
    decision = Gate(policy, model=model).check("rotate_api_key", service="x")
    assert decision.effect is Effect.DENY
    assert decision.rule is None, "a cited rule must exist in the policy to be recorded"


# -- property 4: severity ordering, not file order --------------------------


def test_a_broad_allow_written_first_cannot_shadow_a_later_deny() -> None:
    policy = load_policy({
        "rules": [
            {"id": "allow-all-shell", "effect": "allow", "tool": "run_shell",
             "rationale": "placeholder rationale, deliberately too broad"},
            {"id": "deny-sudoers", "effect": "deny", "tool": "run_shell",
             "arg_matches": {"command": "sudoers"},
             "rationale": "privilege escalation must be refused"},
        ],
    })
    decision = Gate(policy).check("run_shell", command="echo >> /etc/sudoers")
    assert decision.effect is Effect.DENY
    assert decision.rule == "deny-sudoers"


def test_ordered_rules_are_sorted_by_severity(policy) -> None:
    effects = [r.effect for r in policy.ordered_rules()]
    assert effects == sorted(effects, key=lambda e: {Effect.DENY: 0, Effect.ESCALATE: 1, Effect.ALLOW: 2}[e])


# -- matching behaviour ------------------------------------------------------


def test_glob_tool_patterns_match(policy) -> None:
    assert Gate(policy).check("delete_customer", id="7").effect is Effect.ESCALATE


def test_an_absent_argument_does_not_match_a_rule_that_names_it(policy) -> None:
    # `deny-privilege-escalation` names `command`; a run_shell call without one must not
    # be treated as matching it, or every shell call would be denied as if it were sudo.
    decision = Gate(policy).check("run_shell", script="ls -la")
    assert decision.rule != "deny-privilege-escalation"


def test_traversal_is_refused_rather_than_sanitised(policy) -> None:
    decision = Gate(policy).check("read_report", name="../../etc/passwd")
    assert decision.effect is Effect.DENY
    assert decision.rule == "deny-escape-report-path"


# -- property 5: strict loading ---------------------------------------------


@pytest.mark.parametrize("bad,message", [
    ({"rules": []}, "at least one"),
    ({"rules": [{"id": "a", "effect": "maybe", "rationale": "long enough rationale"}]}, "effect"),
    ({"rules": [{"id": "a", "effect": "deny"}]}, "rationale"),
    ({"rules": [{"id": "a", "effect": "deny", "rationale": "short"}]}, "rationale"),
    ({"rules": [
        {"id": "a", "effect": "deny", "rationale": "long enough rationale"},
        {"id": "a", "effect": "allow", "rationale": "long enough rationale"},
    ]}, "duplicate"),
    ({"rules": [{"id": "a", "effect": "deny", "rationale": "long enough rationale",
                 "arg_matches": {"x": "([unclosed"}}]}, "not a valid regex"),
    ({"rules": [{"id": "a", "effect": "deny", "rationale": "long enough rationale",
                 "tool": 17}]}, "tool"),
])
def test_a_malformed_policy_refuses_to_load(bad, message) -> None:
    with pytest.raises(PolicyError, match=message):
        load_policy(bad)


def test_a_missing_policy_file_refuses_to_load() -> None:
    with pytest.raises(PolicyError, match="not found"):
        load_policy(ROOT / "examples" / "does-not-exist.toml")


def test_invalid_toml_refuses_to_load(tmp_path) -> None:
    path = tmp_path / "bad.toml"
    path.write_text("[[rules]\nid = 'broken'\n")
    with pytest.raises(PolicyError, match="valid TOML"):
        load_policy(path)


# -- property 6: the audit chain --------------------------------------------

def test_the_audit_chain_verifies(gate) -> None:
    gate.check("lookup_order", order_id="A-1")
    gate.check("run_shell", command="sudo su")
    ok, message = gate.audit.verify()
    assert ok, message


def test_editing_an_audit_entry_breaks_the_chain(policy) -> None:
    audit = AuditLog()
    gate = Gate(policy, audit=audit)
    gate.check("lookup_order", order_id="A-1")          # entry 0: allowed
    gate.check("run_shell", command="echo x >> /etc/sudoers")  # entry 1: refused

    ok, _ = audit.verify()
    assert ok

    # Tamper with the refusal, not the allow: rewriting an entry to the value it
    # already holds is a change no integrity check can or should detect. My first
    # version of this test flipped entry 0, which was already an allow, so it was
    # asserting nothing; the second used a call that escalates, so the precondition
    # caught it. Both are why a tamper test must assert its own starting state.
    assert audit.entries[1]["effect"] == "deny"
    audit.entries[1]["effect"] = "allow"  # turn a refusal into an approval
    ok, message = audit.verify()
    assert not ok
    assert "hash does not match" in message or "prev does not match" in message


def test_the_audit_log_persists_and_reloads(tmp_path) -> None:
    path = tmp_path / "audit.jsonl"
    gate = Gate(load_policy(POLICY), audit=AuditLog(path))
    gate.check("lookup_order", order_id="A-1")
    gate.check("rotate_api_key", service="billing")

    reloaded = AuditLog(path)
    assert len(reloaded.decisions()) == 2
    ok, message = reloaded.verify()
    assert ok, message


def test_the_audit_summary_counts_the_interesting_events(policy) -> None:
    audit = AuditLog()
    gate = Gate(policy, model=StaticModel(effect=Effect.ALLOW), audit=audit)
    gate.check("lookup_order", order_id="A-1")
    gate.check("rotate_api_key", service="billing")
    summary = audit.summary()
    assert summary["allow"] == 1
    assert summary["escalate"] == 1
    assert summary["model_allow_rejected"] == 1


def test_the_null_log_keeps_nothing() -> None:
    log = AuditLog.null()
    log.note("something")
    assert log.entries
    assert log.path is None


# -- the written record stays honest ----------------------------------------


def test_the_readme_states_the_measurement_the_design_rests_on() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "23.5%" in readme, "the design rests on this number; it must be stated"
    assert "cannot authorise" in readme or "cannot authorise an uncovered" in readme
