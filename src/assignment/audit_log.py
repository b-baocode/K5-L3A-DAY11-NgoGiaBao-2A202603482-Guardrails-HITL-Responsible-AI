"""
Assignment 11 — Audit Log starter (TODO).

Records every interaction for forensics. Never blocks by itself —
other layers catch attacks; this layer makes them reviewable.
"""
from __future__ import annotations

import json
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path


def default_audit_log_path() -> str:
    """Always resolve to <repo>/outputs/… (safe when cwd is src/)."""
    repo_root = Path(__file__).resolve().parents[2]
    return str(repo_root / "outputs" / "audit_log.json")


class AuditLogPlugin:
    """Framework-agnostic audit logger (wire into ADK callbacks or your pipeline)."""

    def __init__(self):
        self.name = "audit_log"
        self.logs: list[dict] = []
        self._open: dict[str, float] = {}
        self._pending: dict[str, dict] = {}

    def record_input(self, *, user_id: str, text: str, request_id: str | None = None) -> str:
        """Store input + start timestamp; returns the request_id to pass to record_output."""
        key = request_id or f"{user_id}:{uuid.uuid4().hex[:8]}"
        self._open[key] = time.perf_counter()
        self._pending[key] = {
            "request_id": key,
            "user_id": user_id,
            "timestamp": utc_now_iso(),
            "input": text,
        }
        return key

    def record_output(
        self,
        *,
        user_id: str,
        text: str,
        blocked: bool = False,
        layer: str | None = None,
        request_id: str | None = None,
    ):
        """Store output, layer decision, latency; append to self.logs."""
        key = request_id
        if key is None:
            # Fall back to the most recent open request of this user.
            key = next(
                (k for k in reversed(self._pending) if self._pending[k]["user_id"] == user_id),
                None,
            )
        entry = self._pending.pop(key, None) or {
            "request_id": key,
            "user_id": user_id,
            "timestamp": utc_now_iso(),
            "input": None,
        }
        start = self._open.pop(key, None)
        entry.update({
            "output": text,
            "blocked": blocked,
            "layer": layer,
            "latency_ms": round((time.perf_counter() - start) * 1000, 2) if start else None,
        })
        self.logs.append(entry)
        return entry

    def export_json(self, filepath: str | None = None):
        """Write logs to disk (JSON array) under repo-root ``outputs/`` by default."""
        path = Path(filepath or default_audit_log_path())
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.logs, indent=2, ensure_ascii=False), encoding="utf-8")
        return str(path)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
