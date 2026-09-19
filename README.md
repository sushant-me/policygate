# policygate

**A fail-closed policy gate for autonomous agent tool calls, built so that a language
model can never authorise an action.**

```bash
pip install -e .            # stdlib only, no runtime dependencies
python3 examples/demo.py    # six tool calls, decided and audited
pytest tests/ -q            # the safety properties, as tests
```

```python
from policygate import Gate, StaticModel, ToolCall

gate = Gate.from_file("policy.toml", model=my_model)
decision = gate.evaluate(ToolCall(tool="run_shell", args={"command": "…"}))
if decision.blocked():
    raise PermissionError(str(decision))
```

---

## Why it is built this way

The design rests on a measurement, not a preference. In the evaluation behind
[*Edge-Native Semantic Firewall for Autonomous LLM Agents*](https://github.com/sushant-me/Edge-Native_Semantic_Firewall_),
a locally served 3.8B model acting as the policy evaluator approved **23.5%** of
proposals the policy would have blocked — **46.2%** in its JSON-only configuration, and
roughly one verdict in ten changed between identical runs at temperature 0.

A component with that error rate is useful for ranking the genuinely ambiguous cases and
unusable as an authority. So this package draws the line explicitly:

| who may decide | what they may do |
|---|---|
| **a rule in the policy** | allow, deny, or escalate — the only source of `ALLOW` |
| **a model evaluator** | deny, or escalate. **Never allow.** |
| **the default** | escalate (deny is also available; allow requires an explicit unsafe opt-in) |

A model that returns `ALLOW` on a call no rule covers is **overruled**, and the attempt
is written to the audit log — that is the interesting signal, because it says the policy
has a gap where the model thinks it sees a routine action.

## Rules are matched by severity, not by position

All `deny` rules are considered before any `escalate`, and those before any `allow`. A
broad allow written at the top of the file cannot shadow a narrow deny written below it.
Ordering by line position is the obvious implementation and it is the wrong one: it makes
the policy's meaning depend on file order, which review cannot see.

```toml
[gate]
default = "escalate"     # what happens to a call no rule covers
consult_model = true

[[rules]]
id = "deny-privilege-escalation"
effect = "deny"
tool = "run_shell"
arg_matches = { command = "(?i)(sudoers|/etc/shadow|NOPASSWD)" }
rationale = "Writing to sudoers or shadow grants persistent privilege."
```

Every rule needs a **rationale**, and the gate refuses to load without one: that string
is what a human reads when the gate escalates. Loading is strict in general — an
unparseable regex, a duplicate id, an unknown effect, or a policy with no rules raises,
because a gate that starts with half a policy is worse than one that refuses to start.

## Adapters: putting it in front of real agent machinery

```python
from policygate import Gate, load_policy
from policygate.adapters import GatedMCPServer, gated_tools, execute_function_call
```

| adapter | for | the shape it asks for |
|---|---|---|
| `GatedMCPServer` | any MCP-style server | `list_tools()` (or `.tools`) and `call_tool(name, arguments)` |
| `gated_tool` / `gated_tools` | LangChain-style tools | an object with `name` and a callable (`_run`, `run`, `func`, `invoke`, or itself) |
| `execute_function_call` | provider-shaped tool calls | `{"name": …, "arguments": "…"}` or `{"function": {…}}` or an `input` field |

Each is duck-typed, so gating a call never makes LangChain or an MCP SDK a dependency of
your agent.

`GatedMCPServer` does two separate jobs, and the first one happens before any call:
it **refuses a server whose tool names collide with names the agent framework reserves
for itself** (`set_model_response`, `google_search`, `finish`, the skill and loop tools…).
That collision is invisible downstream — those primitives live outside the normal tool
table, so a server tool taking one of their names is never seen by the duplicate-name
guard and simply wins dispatch. It is the bug class I filed against three of Google's
agent frameworks, and here it is refused rather than renamed.

```python
server = GatedMCPServer(mcp_server, gate)           # raises ToolNameCollision if it collides
try:
    result = server.call_tool("lookup_order", {"order_id": "A-1"})
except PolicyBlocked as exc:                        # a rule refused it
    ...
except NeedsHumanApproval as exc:                   # uncovered or human-gated: pending, not refused
    ...
```

**Every adapter enforces the same thing, and the tests assert it with a call counter
rather than by checking a return value: a call the gate does not allow never reaches the
underlying tool.** `examples/agent_integration.py` prints it as evidence — four attempts,
two reach the server, and an evaluator that is offline executes nothing.

## A hole a test found in the first version of the function-call adapter

When the arguments would not parse, my first implementation passed the parse failure
through as an ordinary argument and let the policy decide. The sample policy's
`allow-read-only` rule matches `lookup_order` **on the tool name alone**, so a call whose
arguments the gate never managed to read came back **ALLOW**.

That is the exact input an attacker controls. The fix is that an unreadable call is
escalated **without consulting the policy at all** (`Gate.escalate`), so no name-only
rule can approve something the gate did not read. The regression test asserts
`decision.rule is None` for that path, and removing the fix fails two tests.

## Everything is audited, and the audit is tamper-evident

Every decision — and every non-decision event, like a rejected allow or an unreachable
evaluator — is appended to a JSON Lines log with a **hash chain**: each entry carries
the hash of the one before it. `AuditLog.verify()` walks the chain and names the first
broken entry, so editing a refusal into an approval is detectable.

In an incident review, the entries that matter most are the escalations and the moments
the evaluator was unavailable. Both are first-class here.

## The properties, as tests

`tests/test_invariants.py` is organised around the safety properties rather than the
functions, and each test fails if the property is weakened:

1. `ALLOW` comes only from an allow rule — a model, a default, or a file-ordering
   mistake cannot produce one for an uncovered call.
2. A model cannot authorise: an `ALLOW` verdict on an uncovered call is escalated and
   recorded, and is *not* attributed to the rule the model named.
3. A model that fails is not an approval: raising, timing out, or answering unparseably
   falls through to the policy default.
4. Deny beats allow regardless of where either appears in the file.
5. A malformed policy refuses to load rather than loading partially.
6. A call the gate cannot read is escalated without consulting the policy, so no
   name-only allow rule can approve arguments that were never parsed.
7. The audit chain detects tampering, and survives a reload from disk.

**Checked by mutation, not by assertion.** Weakening two things — letting a model's
`ALLOW` through on an uncovered call, and letting an unparseable call fall through to
rule matching — makes **six tests fail**, and each failure names the path it protects.
Restored, all 58 pass. A test suite for a security control that cannot fail is
documentation, not a control.

Two of those tests were wrong when I first wrote them, and running them is what showed
it: one left `tool` off a rule so the rule matched everything and the default was never
exercised, and the tamper test flipped an entry that already held the value it was
"changing". Both now assert their own starting state.

## Hooking up a real model

`CommandModel` shells out to any command, passes the call and the rules as JSON on
stdin, and parses a strict three-line reply:

```
decision: allow|deny|escalate
rule: <a rule id from the policy, or none>
reason: <one line>
```

Anything unparseable, any unknown decision word, and any rule id that is not in the
policy produce an escalation. A parser that accepts loose output is a parser that
accepts an injected verdict.

```python
from policygate import CommandModel, Gate
model = CommandModel(["ollama", "run", "phi3:mini", "--format", "json"])
gate = Gate.from_file("policy.toml", model=model)
```

## What this does not do

Stated here rather than left for you to discover:

- **It does not stop a prompt injection.** It bounds the *effect* of one: the actions a
  confused agent can take. The instruction layer cannot be kept clean by any filter this
  package could ship, and that is the finding of the paper above, not a gap in the code.
- **It does not decide whether a policy is good.** A rule with a wrong regex allows what
  it should refuse. The gate makes the policy's meaning explicit, reviewable and
  auditable; it cannot make it correct.
- **Model quality is your problem, and it is bounded.** Measured at 66.3% decision
  accuracy and 90.8% on decisive rules in the configuration above — which is why the
  model cannot allow.
- **No de-escalation, no learning.** The policy is static and hand-written. Nothing here
  observes outcomes or adjusts rules.
- **The adapters are tested against stand-ins, not against a live SDK.** `GatedMCPServer`
  and `gated_tool` are exercised against duck-typed fakes that match the shapes I
  documented, not against a pinned MCP SDK release or a specific LangChain version. If
  you wire this into a real framework and it breaks, that is a bug I want: the shape
  assumption is written down in each adapter's docstring precisely so it can be
  contradicted.

## Status

`v0.1.0`, stdlib only, Python 3.11+, CI on 3.11/3.12/3.13. The design and the numbers it
rests on are public and re-checked weekly by
[sushant-me/reputation](https://github.com/sushant-me/reputation).
