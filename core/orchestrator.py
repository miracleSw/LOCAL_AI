#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LOCAL_AI_CORE Orchestrator CLI.
Coordinates end-to-end execution, step-by-step interactive runs, health checks, and state resets.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Optional

# Defensive console UTF-8 handling on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Ensure base and core directories are in python path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from core.config import AppConfig, load_config
from core.bm25_rag import BM25Engine, format_rag_context
from core.llm_client import LLMClient
from core.pipeline_session import PipelineSession
from core.exporter import export_project
from steps import STEP_ORDER, STEP_RUNNERS


def check_system(config: AppConfig, client: LLMClient) -> int:
    """Run comprehensive system diagnostic checks."""
    print("=" * 65)
    print("LOCAL_AI_CORE — SYSTEM INTEGRITY & DIAGNOSTIC CHECK")
    print("=" * 65)

    all_passed = True

    # 1. Check Python version
    py_ver = sys.version.split()[0]
    print(f"[+] Python Version: {py_ver} (OK)")

    # 2. Check Model existence & size
    if config.model_path.exists():
        size_gb = config.model_path.stat().st_size / (1024 ** 3)
        print(f"[+] Model GGUF:     {config.model_path.name} ({size_gb:.2f} GB) (OK)")
    else:
        print(f"[!] Model GGUF:     MISSING at {config.model_path} (FAILED)")
        all_passed = False

    # 3. Check llama binaries
    if config.llama_server_path.exists():
        print(f"[+] llama-server:   Found ({config.llama_server_path.name}) (OK)")
    else:
        print(f"[!] llama-server:   MISSING at {config.llama_server_path} (FAILED)")
        all_passed = False

    if config.llama_cli_path.exists():
        print(f"[+] llama-cli:      Found ({config.llama_cli_path.name}) (OK)")
    else:
        print(f"[!] llama-cli:      MISSING at {config.llama_cli_path} (FAILED)")
        all_passed = False

    # 4. Check RAG indexing
    rag_engine = BM25Engine()
    chunk_count = rag_engine.index_documents_from_dir(config.rag_dir)
    if chunk_count > 0:
        print(f"[+] BM25 RAG Engine: Indexed {chunk_count} chunks from {config.rag_dir} (OK)")
    else:
        print(f"[*] BM25 RAG Engine: No documents in {config.rag_dir} (Directory ready for docs)")

    # 5. Check llama-server connectivity
    server_online = client.is_server_available(timeout=1.0)
    if server_online:
        print(f"[+] llama-server:   ONLINE at {config.server_endpoint} (Ultra-fast HTTP mode)")
    else:
        print(f"[*] llama-server:   OFFLINE (Will smoothly use fallback llama-cli.exe)")

    print("-" * 65)
    if all_passed:
        print("[SUCCESS] All core components are healthy and ready to run.")
        return 0
    else:
        print("[ERROR] Critical components are missing. Please review failures above.")
        return 1


def show_status(config: AppConfig, client: LLMClient) -> None:
    """Display current pipeline execution status and outputs."""
    print("=" * 65)
    print("LOCAL_AI_CORE — PIPELINE STATUS")
    print("=" * 65)

    # Server status
    server_online = client.is_server_available(timeout=1.0)
    server_msg = "ONLINE (Persistent HTTP)" if server_online else "OFFLINE (Standalone CLI Fallback)"
    print(f"Inference Mode:  {server_msg}")

    # Task input status
    if config.input_task_path.exists():
        task_text = config.input_task_path.read_text(encoding="utf-8", errors="replace").strip()
        lines = len(task_text.splitlines())
        chars = len(task_text)
        preview = task_text[:70].replace("\n", " ") + ("..." if chars > 70 else "")
        print(f"Current Task:    {config.input_task_path.name} ({lines} lines, {chars} chars)")
        print(f"Task Preview:    \"{preview}\"")
    else:
        print("Current Task:    No input task file found.")

    print("\nStep Deliverables:")
    for step in STEP_ORDER:
        step_file = config.steps_dir / f"step_{step}.md"
        # Also check numbered filenames like step_01_analyze.md
        numbered_matches = list(config.steps_dir.glob(f"step_*_{step}.md"))
        target = numbered_matches[0] if numbered_matches else step_file

        if target.exists():
            size = target.stat().st_size
            mtime = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(target.stat().st_mtime))
            print(f"  [x] {step:10s} -> {target.name} ({size} bytes, modified {mtime})")
        else:
            print(f"  [ ] {step:10s} -> Not generated")

    # Exported files
    manifest_file = config.final_project_dir / "export_manifest.json"
    if manifest_file.exists():
        try:
            mdata = json.loads(manifest_file.read_text(encoding="utf-8"))
            print(f"\nFinal Project:   {mdata.get('files_count', 0)} files exported in {config.final_project_dir}")
        except Exception:
            print(f"\nFinal Project:   Folder exists at {config.final_project_dir}")
    else:
        print(f"\nFinal Project:   Not yet exported.")
    print("=" * 65)


def reset_pipeline(config: AppConfig) -> None:
    """Atomically wipe all previous outputs and temporary state."""
    session = PipelineSession()
    session.atomic_reset(output_dirs=[config.steps_dir, config.final_project_dir, config.logs_dir])
    print("[RESET] Cleared step outputs, exported project files, and logs.")


def load_task_text(config: AppConfig, override_text: Optional[str] = None, override_file: Optional[Path] = None) -> str:
    """Retrieve task text from CLI argument, file, or default input file."""
    if override_text:
        return override_text.strip()
    if override_file and override_file.exists():
        return override_file.read_text(encoding="utf-8", errors="replace").strip()
    if config.input_task_path.exists():
        text = config.input_task_path.read_text(encoding="utf-8", errors="replace").strip()
        # Strip template instructions if needed
        return text
    raise ValueError(f"Task file not found: {config.input_task_path}")


def populate_session_from_disk(session: PipelineSession, config: AppConfig) -> None:
    """
    Restore completed steps from disk into session when executing in step-by-step interactive mode.
    Guarantees subsequent steps have required context.
    """
    mapping = {
        "analyze": ["step_01_analyze.md", "step_analyze.md"],
        "solve": ["step_02_solve.md", "step_solve.md"],
        "verify": ["step_03_verify.md", "step_verify.md"],
    }
    for step_name, candidate_files in mapping.items():
        if session.get_step_content(step_name):
            continue
        for cname in candidate_files:
            cpath = config.steps_dir / cname
            if cpath.exists():
                text = cpath.read_text(encoding="utf-8", errors="replace").strip()
                if text:
                    session.record_step(step_name=step_name, content=text, status="success")
                break


def run_pipeline(
    config: AppConfig,
    client: LLMClient,
    task_text: str,
    target_step: Optional[str] = None,
) -> int:
    """Run full pipeline or targeted step."""
    # Initialize RAG engine
    rag_engine = BM25Engine()
    rag_engine.index_documents_from_dir(config.rag_dir)
    rag_chunks = rag_engine.retrieve(task_text, top_k=config.rag_top_k)
    rag_context = format_rag_context(rag_chunks)

    session = PipelineSession(task_text=task_text)

    # If running a single downstream step, populate prior state from disk
    if target_step and target_step not in ("analyze", "1", "step_01_analyze"):
        populate_session_from_disk(session, config)

    steps_to_run = [target_step] if target_step else STEP_ORDER

    print(f"\n[ORCHESTRATOR] Starting pipeline execution (Session: {session.session_id})")
    print(f"[ORCHESTRATOR] Mode: {'HTTP Server' if client.is_server_available() else 'CLI Fallback'}")
    t_start = time.perf_counter()

    for step_key in steps_to_run:
        runner = STEP_RUNNERS.get(step_key.lower())
        if not runner:
            print(f"[ERROR] Unknown step: '{step_key}'. Supported: {STEP_ORDER}", file=sys.stderr)
            return 1

        canon_name = "analyze" if step_key in ("1", "analyze", "step_01_analyze") else (
            "solve" if step_key in ("2", "solve", "step_02_solve") else "verify"
        )

        print("\n" + "=" * 65)
        print(f" RUNNING STEP: {canon_name.upper()}")
        print("=" * 65)

        try:
            if canon_name == "analyze":
                runner(session, client, config, rag_context=rag_context)
            elif canon_name == "solve":
                runner(session, client, config, rag_context=rag_context)
            elif canon_name == "verify":
                runner(session, client, config, auto_export=True)
        except Exception as exc:
            print(f"[FATAL] Step '{canon_name}' failed with exception: {exc}", file=sys.stderr)
            return 1

        rec = session.steps.get(canon_name)
        if rec and rec.status != "success":
            print(f"[WARNING] Step '{canon_name}' completed with validation warnings: {rec.errors}")

    elapsed = time.perf_counter() - t_start
    print("\n" + "=" * 65)
    print(f"[ORCHESTRATOR] Pipeline finished in {elapsed:.2f} seconds.")
    print("=" * 65)

    # Save session summary checkpoint
    config.logs_dir.mkdir(parents=True, exist_ok=True)
    summary_path = config.logs_dir / f"session_{session.session_id}.json"
    session.save_checkpoint(summary_path)
    print(f"[CHECKPOINT] Session state recorded at: {summary_path}")

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="LOCAL_AI_CORE Modular Pipeline Orchestrator")
    parser.add_argument("--run-all", action="store_true", help="Run the complete end-to-end pipeline")
    parser.add_argument("--step", type=str, help="Run a specific step (analyze, solve, verify)")
    parser.add_argument("--check", action="store_true", help="Run system diagnostics and verify dependencies")
    parser.add_argument("--status", action="store_true", help="Show current status of task and deliverables")
    parser.add_argument("--reset", action="store_true", help="Reset all outputs and logs atomically")
    parser.add_argument("--export", action="store_true", help="Export final files to output/final_project")
    parser.add_argument("--task", type=str, help="Direct task string override")
    parser.add_argument("--task-file", type=Path, help="Task file path override")
    parser.add_argument("--config", type=Path, help="Custom config.ini path")

    args = parser.parse_args()

    config = load_config(args.config)
    client = LLMClient(config)

    if args.check:
        return check_system(config, client)

    if args.status:
        show_status(config, client)
        return 0

    if args.reset:
        reset_pipeline(config)
        return 0

    if args.export:
        session = PipelineSession()
        populate_session_from_disk(session, config)
        exported = export_project(session, config.final_project_dir)
        print(f"[EXPORT] Successfully exported {len(exported)} files to {config.final_project_dir}")
        return 0

    if args.run_all or args.step:
        try:
            task_text = load_task_text(config, override_text=args.task, override_file=args.task_file)
        except Exception as exc:
            print(f"[ERROR] Failed to load task: {exc}", file=sys.stderr)
            return 1

        if not task_text.strip():
            print(f"[ERROR] Task text in '{config.input_task_path}' is empty. Please enter your problem description first.", file=sys.stderr)
            return 1

        if args.run_all:
            # Perform atomic reset before complete run to eliminate any stale outputs
            reset_pipeline(config)
            return run_pipeline(config, client, task_text, target_step=None)
        else:
            return run_pipeline(config, client, task_text, target_step=args.step)

    # If no flags passed, display status by default
    show_status(config, client)
    return 0


if __name__ == "__main__":
    sys.exit(main())
