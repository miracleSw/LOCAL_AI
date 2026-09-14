#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Safe typed configuration reader for LOCAL_AI_CORE.
Resolves paths relative to the project root and provides type-checked settings.
"""
from __future__ import annotations

import configparser
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = BASE_DIR / "config.ini"


def resolve_project_path(raw_path: str | Path, base: Path = BASE_DIR) -> Path:
    """Resolve a relative path against the project root, or keep absolute."""
    p = Path(raw_path)
    if not p.is_absolute():
        p = base / p
    return p.resolve()


@dataclass
class AppConfig:
    """Strongly typed application configuration."""
    # Paths
    llama_server_path: Path
    llama_cli_path: Path
    model_path: Path
    rag_dir: Path
    input_task_path: Path
    output_dir: Path
    steps_dir: Path
    final_project_dir: Path
    logs_dir: Path

    # Server settings
    server_host: str
    server_port: int
    server_endpoint: str
    server_health_endpoint: str
    request_timeout_seconds: int

    # Llama parameters
    ctx_size: int
    threads: int
    gpu_layers: int
    no_mmap: bool
    no_kv_offload: bool
    temperature: float
    top_p: float
    repeat_penalty: float
    max_tokens: int
    max_runtime_seconds: int
    no_warmup: bool

    # RAG settings
    rag_top_k: int
    rag_chunk_chars: int
    rag_overlap_chars: int
    always_include_rules: bool

    # Step execution settings
    step_01_max_tokens: int
    step_02_max_tokens: int
    step_03_max_tokens: int
    max_repair_attempts: int

    # Raw parser for fallback/custom extensions
    raw_parser: configparser.ConfigParser

    def get_step_tokens(self, step_name: str) -> int:
        """Return token limit for specific step."""
        key = f"step_{step_name}_max_tokens"
        if hasattr(self, key):
            return getattr(self, key)
        # Match by name e.g. "analyze" -> step_01
        mapping = {
            "analyze": self.step_01_max_tokens,
            "solve": self.step_02_max_tokens,
            "verify": self.step_03_max_tokens,
            "step_01_analyze": self.step_01_max_tokens,
            "step_02_solve": self.step_02_max_tokens,
            "step_03_verify": self.step_03_max_tokens,
        }
        return mapping.get(step_name, self.max_tokens)

    def check_integrity(self) -> List[str]:
        """Check if critical files and directories exist; return list of warnings/errors."""
        issues: List[str] = []
        if not self.model_path.exists():
            issues.append(f"Model file missing: {self.model_path}")
        if not self.llama_server_path.exists():
            issues.append(f"llama-server executable missing: {self.llama_server_path}")
        if not self.llama_cli_path.exists():
            issues.append(f"llama-cli executable missing: {self.llama_cli_path}")
        if not self.rag_dir.exists():
            issues.append(f"RAG directory missing: {self.rag_dir}")
        return issues


def load_config(config_file: Optional[Path] = None) -> AppConfig:
    """Load and parse config.ini with safe defaults."""
    cfg_path = config_file or DEFAULT_CONFIG_PATH
    parser = configparser.ConfigParser()

    if cfg_path.exists():
        parser.read(cfg_path, encoding="utf-8")

    # Safe accessors
    def _p(section: str, key: str, default: str) -> Path:
        val = parser.get(section, key, fallback=default).strip()
        return resolve_project_path(val)

    def _s(section: str, key: str, default: str) -> str:
        return parser.get(section, key, fallback=default).strip()

    def _i(section: str, key: str, default: int) -> int:
        try:
            return parser.getint(section, key, fallback=default)
        except Exception:
            return default

    def _f(section: str, key: str, default: float) -> float:
        try:
            return parser.getfloat(section, key, fallback=default)
        except Exception:
            return default

    def _b(section: str, key: str, default: bool) -> bool:
        try:
            return parser.getboolean(section, key, fallback=default)
        except Exception:
            return default

    server_host = _s("server", "host", "127.0.0.1")
    server_port = _i("server", "port", 8080)
    server_endpoint = _s("server", "endpoint", f"http://{server_host}:{server_port}/completion")
    server_health_endpoint = _s("server", "health_endpoint", f"http://{server_host}:{server_port}/health")

    return AppConfig(
        llama_server_path=_p("paths", "llama_server", "llama-vulkan/llama-server.exe"),
        llama_cli_path=_p("paths", "llama_cli", "llama-vulkan/llama-cli.exe"),
        model_path=_p("paths", "model", "models/qwen25-coder-3b-q4km.gguf"),
        rag_dir=_p("paths", "rag_dir", "rag/active"),
        input_task_path=_p("paths", "input_task", "input/current_task.txt"),
        output_dir=_p("paths", "output_dir", "output"),
        steps_dir=_p("paths", "steps_dir", "output/steps"),
        final_project_dir=_p("paths", "final_project_dir", "output/final_project"),
        logs_dir=_p("paths", "logs_dir", "output/logs"),

        server_host=server_host,
        server_port=server_port,
        server_endpoint=server_endpoint,
        server_health_endpoint=server_health_endpoint,
        request_timeout_seconds=_i("server", "request_timeout_seconds", 300),

        ctx_size=_i("llama", "ctx_size", 6192),
        threads=_i("llama", "threads", 6),
        gpu_layers=_i("llama", "gpu_layers", 12),
        no_mmap=_b("llama", "no_mmap", True),
        no_kv_offload=_b("llama", "no_kv_offload", True),
        temperature=_f("llama", "temperature", 0.05),
        top_p=_f("llama", "top_p", 0.85),
        repeat_penalty=_f("llama", "repeat_penalty", 1.05),
        max_tokens=_i("llama", "max_tokens", 2048),
        max_runtime_seconds=_i("llama", "max_runtime_seconds", 300),
        no_warmup=_b("llama", "no_warmup", False),

        rag_top_k=_i("rag", "top_k", 5),
        rag_chunk_chars=_i("rag", "chunk_chars", 1200),
        rag_overlap_chars=_i("rag", "overlap_chars", 150),
        always_include_rules=_b("rag", "always_include_rules", True),

        step_01_max_tokens=_i("steps", "step_01_max_tokens", 1200),
        step_02_max_tokens=_i("steps", "step_02_max_tokens", 2048),
        step_03_max_tokens=_i("steps", "step_03_max_tokens", 2048),
        max_repair_attempts=_i("steps", "max_repair_attempts", 2),

        raw_parser=parser,
    )
