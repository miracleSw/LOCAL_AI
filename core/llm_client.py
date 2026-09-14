#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dual-mode LLM Client for LOCAL_AI_CORE.
Supports:
1. High-speed HTTP completion via persistent llama-server.exe (zero model reload latency, KV-cache reuse).
2. Graceful fallback to standalone llama-cli.exe with anti-deadlock flags and non-blocking I/O.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from core.config import AppConfig, load_config

NOISE_TAIL_PATTERNS = [
    r"\nllama_perf_[\s\S]*$",
    r"\n\[end of text\][\s\S]*$",
    r"\nExiting\.\.\.[\s\S]*$",
    r"\n\s*>\s*$",  # trailing interactive prompt symbol
]


@dataclass
class LLMResponse:
    """Standardized response from LLM inference."""
    content: str
    mode: str  # 'server' or 'cli'
    latency_ms: float
    prompt_tokens: int = 0
    completion_tokens: int = 0
    finish_reason: str = "stop"
    raw_response: Dict[str, Any] = field(default_factory=dict)


def clean_llm_output(raw: str, prompt: Optional[str] = None) -> str:
    """Strip banner, timings, echoed prompt, and trailing noise from raw CLI/server output."""
    text = raw.replace("\r\n", "\n").replace("\r", "\n")

    # If CLI banner is present, strip up to the end of available commands or help text
    for cmd_marker in ["available commands:", "/exit or Ctrl+C", "stop or exit"]:
        if cmd_marker in text:
            idx = text.find(cmd_marker)
            text = text[idx + len(cmd_marker):]

    # Strip any echoed ChatML tags from prompt
    for tag in ["<|im_start|>assistant", "<|im_start|>user", "<|im_start|>system", "... (truncated)"]:
        if tag in text:
            last_idx = text.rfind(tag)
            text = text[last_idx + len(tag):]

    # Strip prompt echo if present
    if prompt:
        p_clean = prompt.strip().replace("\r\n", "\n").replace("\r", "\n")
        if text.strip().startswith(p_clean):
            text = text.strip()[len(p_clean):]
        elif ("> " + p_clean) in text:
            idx = text.find("> " + p_clean)
            text = text[idx + len("> " + p_clean):]
        else:
            p_lines = [line.strip() for line in p_clean.splitlines() if line.strip()]
            if p_lines:
                last_line = p_lines[-1]
                if last_line in text:
                    idx = text.find(last_line)
                    text = text[idx + len(last_line):]

    # Strip any leading interactive prompt marker
    text = re.sub(r"^\s*>\s*", "", text)

    for pat in NOISE_TAIL_PATTERNS:
        text = re.sub(pat, "", text, flags=re.IGNORECASE)

    # Strip prompt/generation speed footer e.g. [ Prompt: 149.3 t/s | Generation: 23.3 t/s ]
    text = re.sub(r"\n\[\s*Prompt:[\s\S]*$", "", text, flags=re.IGNORECASE)

    return text.strip() + ("\n" if text.strip() else "")


class LLMClient:
    """Client that coordinates fast server requests with reliable CLI fallback."""

    def __init__(self, config: Optional[AppConfig] = None):
        self.config = config or load_config()

    def is_server_available(self, timeout: float = 1.5) -> bool:
        """Check if llama-server is running on configured host and port."""
        req = urllib.request.Request(
            self.config.server_health_endpoint,
            headers={"User-Agent": "LOCAL_AI_CORE/1.0"},
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.status in (200, 204)
        except Exception:
            return False

    def generate_via_server(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> LLMResponse:
        """Query persistent llama-server HTTP /completion endpoint."""
        t_start = time.perf_counter()
        n_predict = max_tokens if max_tokens is not None else self.config.max_tokens
        temp = temperature if temperature is not None else self.config.temperature

        payload = {
            "prompt": prompt,
            "n_predict": n_predict,
            "temperature": temp,
            "top_p": self.config.top_p,
            "repeat_penalty": self.config.repeat_penalty,
            "stop": ["<|im_end|>", "<|endoftext|>", "### COMPLETE_ANSWER"],
            "stream": False,
        }

        data_bytes = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            self.config.server_endpoint,
            data=data_bytes,
            headers={"Content-Type": "application/json; charset=utf-8"},
            method="POST",
        )

        with urllib.request.urlopen(req, timeout=self.config.request_timeout_seconds) as resp:
            body = resp.read().decode("utf-8", errors="replace")
            res_json = json.loads(body)

        latency = (time.perf_counter() - t_start) * 1000.0
        content = clean_llm_output(res_json.get("content", ""), prompt=prompt)

        return LLMResponse(
            content=content,
            mode="server",
            latency_ms=latency,
            prompt_tokens=res_json.get("tokens_evaluated", 0),
            completion_tokens=res_json.get("tokens_predicted", 0),
            finish_reason=res_json.get("stopped_limit", False) and "length" or "stop",
            raw_response=res_json,
        )

    def stream_generate_via_server(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ):
        """Yield tokens from persistent llama-server as they arrive via SSE stream."""
        n_predict = max_tokens if max_tokens is not None else self.config.max_tokens
        temp = temperature if temperature is not None else self.config.temperature

        payload = {
            "prompt": prompt,
            "n_predict": n_predict,
            "temperature": temp,
            "top_p": self.config.top_p,
            "repeat_penalty": self.config.repeat_penalty,
            "stop": ["<|im_end|>", "<|endoftext|>", "### COMPLETE_ANSWER"],
            "stream": True,
        }

        data_bytes = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            self.config.server_endpoint,
            data=data_bytes,
            headers={"Content-Type": "application/json; charset=utf-8"},
            method="POST",
        )

        with urllib.request.urlopen(req, timeout=self.config.request_timeout_seconds) as resp:
            for raw_line in resp:
                line_str = raw_line.decode("utf-8", errors="replace").strip()
                if not line_str.startswith("data:"):
                    continue
                data_str = line_str[5:].strip()
                if not data_str or data_str == "[DONE]":
                    break
                try:
                    data = json.loads(data_str)
                    content = data.get("content", "")
                    if content:
                        yield content
                    if data.get("stop", False):
                        break
                except Exception:
                    continue

    def generate_via_cli(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        prompt_save_path: Optional[Path] = None,
    ) -> LLMResponse:
        """Fallback execution using llama-cli.exe with anti-deadlock guards."""
        t_start = time.perf_counter()
        n_predict = max_tokens if max_tokens is not None else self.config.max_tokens
        temp = temperature if temperature is not None else self.config.temperature

        # Ensure steps directory exists for temporary prompt file
        self.config.steps_dir.mkdir(parents=True, exist_ok=True)
        if prompt_save_path is None:
            prompt_file = self.config.steps_dir / "last_cli_prompt.txt"
        else:
            prompt_file = prompt_save_path

        prompt_file.write_text(prompt, encoding="utf-8")

        # Build CLI command with anti-deadlock flags
        cmd = [
            str(self.config.llama_cli_path),
            "-m", str(self.config.model_path),
            "-c", str(self.config.ctx_size),
            "-t", str(self.config.threads),
            "-ngl", str(self.config.gpu_layers),
            "--temp", str(temp),
            "--top-p", str(self.config.top_p),
            "--repeat-penalty", str(self.config.repeat_penalty),
            "-n", str(n_predict),
            "--single-turn",        # Prevent entering REPL prompt
            "--no-conversation",    # Disable interactive session completely
            "--no-display-prompt",  # Prevent echoing prompt back
            "--log-disable",        # Cleaner output stream
            "--simple-io",          # Line-buffered I/O
            "--color", "off",       # No ANSI escape sequences
        ]

        if self.config.no_mmap:
            cmd.append("--no-mmap")
        if self.config.no_kv_offload:
            cmd.append("--no-kv-offload")
        if self.config.no_warmup:
            cmd.append("--no-warmup")

        cmd.extend(["-f", str(prompt_file)])

        # Run process with stdin=DEVNULL to prevent any interactive read block
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
        )

        try:
            stdout_text, stderr_text = proc.communicate(timeout=self.config.max_runtime_seconds)
        except subprocess.TimeoutExpired:
            # Kill process tree on Windows safely
            if os.name == "nt":
                subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            else:
                proc.kill()
            raise TimeoutError(f"llama-cli timed out after {self.config.max_runtime_seconds} seconds")

        if proc.returncode != 0 and not stdout_text:
            raise RuntimeError(f"llama-cli failed with exit code {proc.returncode}:\n{stderr_text}")

        latency = (time.perf_counter() - t_start) * 1000.0
        content = clean_llm_output(stdout_text, prompt=prompt)

        return LLMResponse(
            content=content,
            mode="cli",
            latency_ms=latency,
            raw_response={"stderr": stderr_text, "returncode": proc.returncode},
        )

    def generate(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        prompt_save_path: Optional[Path] = None,
    ) -> LLMResponse:
        """
        Execute completion with automatic server detection:
        - If llama-server is online: use fast HTTP completion.
        - If llama-server is offline: fallback smoothly to llama-cli.exe.
        """
        if self.is_server_available():
            try:
                return self.generate_via_server(prompt, max_tokens=max_tokens, temperature=temperature)
            except Exception as exc:
                print(f"[LLMClient] Warning: llama-server request failed ({exc}). Falling back to llama-cli.exe...", file=sys.stderr)

        return self.generate_via_cli(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            prompt_save_path=prompt_save_path,
        )

    def stream_generate(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ):
        """
        Stream tokens if persistent llama-server is online.
        Gracefully fall back to CLI if server is offline.
        """
        if self.is_server_available():
            try:
                for token in self.stream_generate_via_server(prompt, max_tokens=max_tokens, temperature=temperature):
                    yield token
                return
            except Exception as exc:
                print(f"[LLMClient] Server stream interrupted ({exc}). Falling back to CLI...", file=sys.stderr)

        resp = self.generate_via_cli(prompt, max_tokens=max_tokens, temperature=temperature)
        yield resp.content
