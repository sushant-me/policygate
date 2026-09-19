"""Put the gate in front of a server and an agent loop, end to end.

    python3 examples/agent_integration.py

Three things this shows that the unit tests cannot:

1. A server whose tool names collide with framework-reserved names is refused **before**
   any call happens — the collision is a property of the tool list, not of a call.
2. Every call in the loop goes through the gate, and a call the gate does not allow
   never reaches the server. The server counts its own invocations, so the output is
   evidence rather than assertion.
3. A model backend that is unavailable does not become an approval.

No model, no network, no dependencies: the "agent" is a fixed list of attempted calls,
which is enough to exercise the paths that matter.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from policygate import (  # noqa: E402
    AuditLog, Effect, Gate, NeedsHumanApproval, PolicyBlocked, StaticModel, load_policy,
)
from policygate.adapters import GatedMCPServer  # noqa: E402
from policygate.adapters.mcp import ToolNameCollision  # noqa: E402


class ShopServer:
    """A stand-in for an MCP server. Counts its own invocations."""

    def __init__(self, tools: list[str]) -> None:
        self._tools = [{"name": n} for n in tools]
        self.executed: list[str] = []

    def list_tools(self):
        return self._tools

    def call_tool(self, name, arguments):
        self.executed.append(name)
        return f"{name} -> ok"


def main() -> int:
    policy = load_policy(ROOT / "examples" / "policy.toml")

    print("1. A server advertising a framework-reserved tool name is refused up front\n")
    bad = ShopServer(["lookup_order", "set_model_response"])
    try:
        GatedMCPServer(bad, Gate(policy))
    except ToolNameCollision as exc:
        print(f"   refused: {exc}\n")

    print("2. A well-named server, gated — the server counts what it actually ran\n")
    server = ShopServer(["lookup_order", "summarise", "send_email", "run_shell"])
    # The model is configured to approve everything, which is the most permissive a
    # backend can be. It is overruled on uncovered calls.
    gate = Gate(
        policy,
        model=StaticModel(effect=Effect.ALLOW, rule_id="allow-read-only",
                          reason="looks routine", confidence=0.97),
        audit=AuditLog(),
    )
    gated = GatedMCPServer(server, gate)

    attempts = [
        ("lookup_order", {"order_id": "A-1042"}),
        ("summarise", {"text": "quarterly figures"}),
        ("run_shell", {"command": "echo 'app ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers"}),
        ("send_email", {"to": "customer@example.com", "body": "refund issued"}),
    ]
    for name, arguments in attempts:
        try:
            result = gated.call_tool(name, arguments)
            print(f"   ran      {name}({arguments})  -> {result}")
        except PolicyBlocked as exc:
            print(f"   REFUSED  {name}({arguments})\n              rule: {exc.decision.rule}")
        except NeedsHumanApproval as exc:
            print(f"   PENDING  {name}({arguments})\n              {exc.decision.reason[:88]}")

    print(f"\n   server executed: {server.executed}")
    print(f"   (4 attempts, {len(server.executed)} reached the server)\n")

    print("3. An unavailable evaluator is not an approval\n")
    offline = Gate(policy, model=StaticModel(raises=ConnectionError("no backend")))
    offline_server = GatedMCPServer(ShopServer(["rotate_api_key"]), offline, allow_reserved=True)
    try:
        offline_server.call_tool("rotate_api_key", {"service": "billing"})
    except NeedsHumanApproval as exc:
        print(f"   PENDING  rotate_api_key\n              {exc.decision.reason[:88]}\n")
    print(f"   offline server executed: {offline_server.server.executed}")

    ok, message = gate.audit.verify()
    print(f"\naudit chain: {'OK' if ok else 'BROKEN'} — {message}")
    print("summary:", gate.audit.summary())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
