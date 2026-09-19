"""The audit log: a hash-chained record of every decision.

An agent security control that cannot reconstruct what it allowed is not much of a
control, and the entries that matter most in a review are the ones where the policy had
nothing to say: the escalations, the model's rejected allow attempts, and the moments
the evaluator was unavailable.

Each entry carries the hash of the previous one, so removing or editing a line is
detectable. `verify()` walks the chain and reports the first break — it is cheap enough
to run on every startup.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .decision import Decision, Effect, ToolCall

GENESIS = "0" * 64


def _entry_hash(previous: str, payload: dict[str, Any]) -> str:
    blob = previous + json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()


class AuditLog:
    """Append-only JSONL with a hash chain. `null()` discards everything."""

    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path) if path is not None else None
        self._entries: list[dict[str, Any]] = []
        self._previous = GENESIS
        if self.path is not None and self.path.exists():
            self._load()

    # -- constructors ------------------------------------------------------

    @classmethod
    def null(cls) -> "AuditLog":
        """A log that keeps nothing. Used when a caller does not want persistence."""
        return cls(path=None)

    # -- writing -----------------------------------------------------------

    def record(self, decision: Decision, latency_ms: float = 0.0) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "ts": round(time.time(), 3),
            "kind": "decision",
            "effect": decision.effect.value,
            "rule": decision.rule,
            "decided_by": decision.decided_by,
            "reason": decision.reason,
            "confidence": decision.confidence,
            "latency_ms": round(latency_ms, 3),
            "tool": decision.call.tool if decision.call else None,
            "principal": decision.call.principal if decision.call else None,
            "digest": decision.call.digest() if decision.call else None,
            "args": decision.call.args if decision.call else None,
        }
        return self._append(payload)

    def note(self, kind: str, call: ToolCall | None = None, detail: str = "") -> dict[str, Any]:
        """Record something that is not a decision: a rejection, an outage, a warning."""
        payload: dict[str, Any] = {
            "ts": round(time.time(), 3),
            "kind": kind,
            "detail": detail,
            "tool": call.tool if call else None,
            "principal": call.principal if call else None,
            "digest": call.digest() if call else None,
        }
        return self._append(payload)

    def _append(self, payload: dict[str, Any]) -> dict[str, Any]:
        entry = {**payload, "prev": self._previous}
        entry["hash"] = _entry_hash(self._previous, {k: v for k, v in entry.items() if k != "hash"})
        self._previous = entry["hash"]
        self._entries.append(entry)
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(entry, sort_keys=True, default=str) + "\n")
        return entry

    # -- reading -----------------------------------------------------------

    def _load(self) -> None:
        assert self.path is not None
        for line in self.path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            self._entries.append(entry)
            self._previous = entry.get("hash", self._previous)

    @property
    def entries(self) -> list[dict[str, Any]]:
        return list(self._entries)

    def decisions(self) -> list[dict[str, Any]]:
        return [e for e in self._entries if e.get("kind") == "decision"]

    def summary(self) -> dict[str, int]:
        """Counts by effect, plus the non-decision events worth watching."""
        counts: dict[str, int] = {effect.value: 0 for effect in Effect}
        for entry in self._entries:
            if entry.get("kind") == "decision":
                counts[entry["effect"]] = counts.get(entry["effect"], 0) + 1
            else:
                counts[entry["kind"]] = counts.get(entry["kind"], 0) + 1
        counts["total"] = len(self._entries)
        return counts

    def verify(self) -> tuple[bool, str]:
        """Walk the chain. Returns (ok, message) and names the first broken entry."""
        previous = GENESIS
        for index, entry in enumerate(self._entries):
            stored = entry.get("hash")
            recomputed = _entry_hash(
                previous, {k: v for k, v in entry.items() if k != "hash"}
            )
            if entry.get("prev") != previous:
                return False, f"entry {index}: prev does not match the previous hash"
            if stored != recomputed:
                return False, f"entry {index}: hash does not match its contents"
            previous = stored
        return True, f"{len(self._entries)} entries verified"

    # -- convenience -------------------------------------------------------

    def last(self, n: int = 5) -> list[dict[str, Any]]:
        return self._entries[-n:]

    def as_dicts(self) -> list[dict[str, Any]]:
        return [asdict(d) if not isinstance(d, dict) else d for d in self._entries]
