#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
In-Memory Pipeline Session & State Management.
Prevents stale disk IPC leakage and provides an atomic execution context across pipeline steps.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class StepRecord:
    """Record of execution for a single pipeline step."""
    step_name: str
    status: str = "pending"  # pending, running, success, failed
    content: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    tokens_used: int = 0
    latency_ms: float = 0.0
    started_at: float = 0.0
    completed_at: float = 0.0
    errors: List[str] = field(default_factory=list)


class PipelineSession:
    """
    Session context holding state in-memory during pipeline execution.
    Eliminates reliance on reading stale markdown outputs from prior runs.
    """

    def __init__(self, task_text: str = "", session_id: Optional[str] = None):
        self.task_text: str = task_text.strip()
        self.session_id: str = session_id or f"sess_{int(time.time())}_{hashlib.md5(task_text.encode('utf-8')).hexdigest()[:8]}"
        self.steps: Dict[str, StepRecord] = {}
        self.artifacts: Dict[str, str] = {}  # filename -> content
        self.metadata: Dict[str, Any] = {}
        self.created_at: float = time.time()
        self.updated_at: float = time.time()

    def set_task(self, task_text: str) -> None:
        """Update task and reset session state for a fresh run."""
        self.task_text = task_text.strip()
        self.session_id = f"sess_{int(time.time())}_{hashlib.md5(self.task_text.encode('utf-8')).hexdigest()[:8]}"
        self.steps.clear()
        self.artifacts.clear()
        self.metadata.clear()
        self.updated_at = time.time()

    def record_step(
        self,
        step_name: str,
        content: str,
        status: str = "success",
        metadata: Optional[Dict[str, Any]] = None,
        latency_ms: float = 0.0,
        errors: Optional[List[str]] = None,
    ) -> StepRecord:
        """Record the output of a completed step."""
        rec = StepRecord(
            step_name=step_name,
            status=status,
            content=content,
            metadata=metadata or {},
            latency_ms=latency_ms,
            completed_at=time.time(),
            errors=errors or [],
        )
        self.steps[step_name] = rec
        self.updated_at = time.time()
        return rec

    def get_step_content(self, step_name: str) -> Optional[str]:
        """Get the content string of a specific step."""
        rec = self.steps.get(step_name)
        return rec.content if rec and rec.status == "success" else None

    def get_accumulated_context(self, exclude_steps: Optional[List[str]] = None) -> str:
        """
        Gather verified outputs from previous steps in execution order.
        Used to feed context into downstream steps without disk file I/O.
        """
        excluded = set(exclude_steps or [])
        blocks = []
        for name, rec in self.steps.items():
            if name in excluded or rec.status != "success" or not rec.content.strip():
                continue
            blocks.append(f"### [OUTPUT FROM {name.upper()}]\n{rec.content.strip()}")
        return "\n\n".join(blocks)

    def set_artifact(self, filename: str, content: str) -> None:
        """Store an assembled file or artifact in memory."""
        self.artifacts[filename] = content
        self.updated_at = time.time()

    def get_artifacts(self) -> Dict[str, str]:
        """Return all generated artifacts."""
        return dict(self.artifacts)

    def atomic_reset(self, output_dirs: Optional[List[Path]] = None) -> None:
        """
        Clear all in-memory step data and artifacts.
        Optionally wipe specified disk directories to guarantee no stale files persist.
        """
        self.steps.clear()
        self.artifacts.clear()
        self.metadata.clear()
        self.updated_at = time.time()

        if output_dirs:
            for out_dir in output_dirs:
                if out_dir.exists() and out_dir.is_dir():
                    for item in out_dir.iterdir():
                        if item.name == ".gitkeep":
                            continue
                        try:
                            if item.is_file():
                                item.unlink()
                            elif item.is_dir():
                                shutil.rmtree(item)
                        except Exception:
                            pass

    def save_checkpoint(self, filepath: Path) -> None:
        """Save complete session snapshot to JSON for debugging and auditing."""
        filepath.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "session_id": self.session_id,
            "task_text": self.task_text,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "steps": {k: asdict(v) for k, v in self.steps.items()},
            "artifacts": self.artifacts,
            "metadata": self.metadata,
        }
        filepath.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load_checkpoint(cls, filepath: Path) -> PipelineSession:
        """Restore session from JSON snapshot."""
        data = json.loads(filepath.read_text(encoding="utf-8"))
        sess = cls(task_text=data.get("task_text", ""), session_id=data.get("session_id"))
        sess.created_at = data.get("created_at", time.time())
        sess.updated_at = data.get("updated_at", time.time())
        sess.artifacts = data.get("artifacts", {})
        sess.metadata = data.get("metadata", {})

        for name, sdata in data.get("steps", {}).items():
            sess.steps[name] = StepRecord(**sdata)
        return sess
