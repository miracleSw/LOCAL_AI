#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
File Exporter for LOCAL_AI_CORE.
Parses generated code blocks from verified pipeline steps or session artifacts
and cleanly exports concrete source files to `output/final_project`.
Generates export_manifest.json with hashes and statistics.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from core.pipeline_session import PipelineSession

FILE_HEADER_PATTERNS = [
    # # FILE: path/to/file.ext or // FILE: path/to/file.ext or -- FILE: path/to/file.ext
    re.compile(r"^\s*(?:#|//|--|/\*|<!--)\s*FILE:\s*([^\s\*>-]+)", re.MULTILINE | re.IGNORECASE),
    # ### File: path/to/file.ext
    re.compile(r"^#{1,4}\s*(?:File|Tập tin|Tệp):\s*`?([^\n`\r]+)`?", re.MULTILINE | re.IGNORECASE),
]


@dataclass
class ExportedFile:
    relative_path: str
    absolute_path: Path
    size_bytes: int
    lines_count: int
    sha256: str


def sanitize_relpath(path_str: str) -> Optional[Path]:
    """Sanitize relative path to prevent directory traversal outside target root."""
    clean = path_str.strip().strip("'\"`").replace("\\", "/")
    # Remove leading slashes or drive letters
    clean = re.sub(r"^[A-Za-z]:/", "", clean).lstrip("/")
    parts = [p for p in clean.split("/") if p and p != "."]
    # Disallow any path attempting directory traversal
    if ".." in parts or not parts:
        return None
    return Path(*parts)


def extract_files_from_markdown(markdown_text: str) -> Dict[str, str]:
    """
    Parse markdown code blocks with designated file targets.
    Matches markdown fences like:
    ### File: src/app.py
    ```python
    ...
    ```
    OR
    ```python
    # FILE: src/app.py
    ...
    ```
    """
    extracted: Dict[str, str] = {}
    lines = markdown_text.splitlines()
    n = len(lines)
    i = 0

    pending_file_path: Optional[str] = None

    while i < n:
        line = lines[i]

        # Check for heading file marker
        heading_match = re.match(r"^#{1,4}\s*(?:File|Tập tin|Tệp):\s*`?([^\n`\r]+)`?", line, re.IGNORECASE)
        if heading_match:
            pending_file_path = heading_match.group(1).strip()
            i += 1
            continue

        # Check for start of code block
        fence_match = re.match(r"^```([a-zA-Z0-9_\-\.\+]*)\s*$", line.strip())
        if fence_match:
            code_lines: List[str] = []
            i += 1
            inner_file_path: Optional[str] = None

            while i < n and not lines[i].strip().startswith("```"):
                code_line = lines[i]
                if not inner_file_path:
                    # Check first few lines for FILE: comment
                    for pat in FILE_HEADER_PATTERNS:
                        m = pat.search(code_line)
                        if m:
                            inner_file_path = m.group(1).strip()
                            break
                code_lines.append(code_line)
                i += 1

            target_name = inner_file_path or pending_file_path
            if target_name and code_lines:
                sanitized = sanitize_relpath(target_name)
                if sanitized:
                    extracted[str(sanitized).replace("\\", "/")] = "\n".join(code_lines).strip() + "\n"

            # Reset pending heading
            pending_file_path = None

        i += 1

    return extracted


def export_project(
    session: PipelineSession,
    target_dir: Path,
    include_markdown_summary: bool = True,
) -> List[ExportedFile]:
    """
    Export all files from session artifacts and step outputs into target_dir.
    Wipes and recreates target_dir cleanly.
    """
    target_dir.mkdir(parents=True, exist_ok=True)

    # Gather candidate files
    files_to_export: Dict[str, str] = {}

    # 1. In-memory session artifacts take priority
    for rel_path, content in session.get_artifacts().items():
        clean = sanitize_relpath(rel_path)
        if clean:
            files_to_export[str(clean).replace("\\", "/")] = content

    # 2. Parse from step outputs (e.g. solve and verify steps)
    for step_name in ["verify", "solve", "step_03_verify", "step_02_solve"]:
        step_content = session.get_step_content(step_name)
        if step_content:
            parsed = extract_files_from_markdown(step_content)
            for path_key, code in parsed.items():
                if path_key not in files_to_export:
                    files_to_export[path_key] = code

    # If no structured file blocks were found, save the final answer as solution.md
    if not files_to_export:
        verify_out = session.get_step_content("verify") or session.get_step_content("step_03_verify")
        solve_out = session.get_step_content("solve") or session.get_step_content("step_02_solve")
        primary_out = verify_out or solve_out or "No solution output available."
        files_to_export["solution.md"] = primary_out

    # Write files to target_dir
    exported: List[ExportedFile] = []
    manifest_records = []

    for rel_str, content in sorted(files_to_export.items()):
        dest_path = target_dir / Path(rel_str)
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        dest_path.write_text(content, encoding="utf-8")

        data_bytes = content.encode("utf-8")
        sha = hashlib.sha256(data_bytes).hexdigest()
        lines_cnt = len(content.splitlines())

        item = ExportedFile(
            relative_path=rel_str,
            absolute_path=dest_path,
            size_bytes=len(data_bytes),
            lines_count=lines_cnt,
            sha256=sha,
        )
        exported.append(item)
        manifest_records.append({
            "path": rel_str,
            "size_bytes": len(data_bytes),
            "lines": lines_cnt,
            "sha256": sha,
        })

    # Write manifest.json
    manifest_file = target_dir / "export_manifest.json"
    manifest_data = {
        "session_id": session.session_id,
        "files_count": len(exported),
        "files": manifest_records,
    }
    manifest_file.write_text(json.dumps(manifest_data, indent=2), encoding="utf-8")

    return exported
