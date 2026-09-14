"""Run manifests — the reproducibility record every stage writes.

A manifest answers, for one stage of one run: what went in (path, size, checksum),
which tool versions touched it, which parameters applied, what came out (counts), and
when. A rerun by someone else should reproduce the manifest, not merely the answer.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from engine import __version__


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def tool_version(cmd: str) -> str:
    """First line of ``<cmd> --version``, or ``"missing"`` if not on PATH."""
    try:
        out = subprocess.run([cmd, "--version"], capture_output=True, text=True, check=False)
    except FileNotFoundError:
        return "missing"
    line = (out.stdout or out.stderr).strip().splitlines()
    return line[0] if line else "unknown"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class Manifest:
    stage: str
    engine_version: str = __version__
    started_at: str = field(default_factory=_now)
    finished_at: str | None = None
    platform: str = field(default_factory=lambda: f"{platform.system()} {platform.machine()}")
    inputs: dict[str, dict[str, Any]] = field(default_factory=dict)
    tools: dict[str, str] = field(default_factory=dict)
    params: dict[str, Any] = field(default_factory=dict)
    counts: dict[str, Any] = field(default_factory=dict)
    outputs: dict[str, dict[str, Any]] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def add_input(self, name: str, path: Path, checksum: bool = True) -> None:
        rec: dict[str, Any] = {"path": str(path), "bytes": path.stat().st_size}
        if checksum:
            rec["sha256"] = sha256_file(path)
        self.inputs[name] = rec

    def add_output(self, name: str, path: Path) -> None:
        self.outputs[name] = {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}

    def add_tool(self, cmd: str) -> None:
        self.tools[cmd] = tool_version(cmd)

    def note(self, text: str) -> None:
        self.notes.append(text)

    def finish(self) -> None:
        self.finished_at = _now()

    def write(self, path: Path) -> None:
        if self.finished_at is None:
            self.finish()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.__dict__, indent=2, sort_keys=False, default=str) + "\n")

    @staticmethod
    def read(path: Path) -> dict[str, Any]:
        return json.loads(Path(path).read_text())
