#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
STEP 01: Requirements & Constraint Analysis.
Breaks down the input task into crisp requirements, constraints, I/O specs, and edge cases.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from core.config import AppConfig
from core.llm_client import LLMClient, LLMResponse, format_chatml
from core.pipeline_session import PipelineSession
from core.validator_base import BaseValidator, run_with_repair


class AnalyzeValidator(BaseValidator):
    """Validates that Step 1 contains all essential architectural requirements."""

    def __init__(self):
        super().__init__(required_sections=[
            "Problem Summary",
            "Functional Requirements",
            "Constraints",
            "Specification",
            "Edge Cases",
        ])


def build_analyze_prompt(task_text: str, rag_context: str = "") -> str:
    """Construct prompt for Step 1 using ChatML formatting."""
    system_prompt = (
        "You are an expert software architect and systems analyst.\n"
        "Analyze the following task and extract a rigorous, modular problem specification.\n"
        "Do NOT write final code yet; focus strictly on requirements, constraints, and architecture."
    )
    rag_block = f"RAG REFERENCE RULES:\n{rag_context}\n\n" if rag_context.strip() else ""

    user_prompt = f"""{rag_block}TASK DESCRIPTION:
{task_text}

OUTPUT FORMAT REQUIRED (Use these exact markdown headers):
## 1. Problem Summary & Objective
[Crisp summary of the core problem to solve]

## 2. Core Functional Requirements
- [Req 1]
- [Req 2]
- [Req 3]

## 3. Constraints & Invariants
- [Technical, architectural, or platform constraints]
- [Performance, memory, or dependency limits]

## 4. Input & Output Specification
- Input format / parameters:
- Expected output / deliverables:

## 5. Critical Edge Cases
- [Edge case 1 and how it must be handled]
- [Edge case 2 and how it must be handled]"""

    return format_chatml(system_prompt, user_prompt)


def run_step_01(
    session: PipelineSession,
    client: LLMClient,
    config: AppConfig,
    rag_context: str = "",
) -> LLMResponse:
    """Execute Step 1: Analyze & Constraints."""
    task_text = session.task_text
    if not task_text:
        raise ValueError("Cannot run Step 1: Task text in session is empty.")

    prompt = build_analyze_prompt(task_text, rag_context)
    validator = AnalyzeValidator()

    def _generator(p: str) -> LLMResponse:
        tokens = config.get_step_tokens("analyze")
        prompt_file = config.steps_dir / "prompt_01_analyze.txt"
        return client.generate(p, max_tokens=tokens, prompt_save_path=prompt_file)

    resp, val_res, history = run_with_repair(
        step_name="step_01_analyze",
        generator_fn=_generator,
        initial_prompt=prompt,
        validator=validator,
        task_text=task_text,
        max_attempts=config.max_repair_attempts,
    )

    # Persist in session
    session.record_step(
        step_name="analyze",
        content=resp.content,
        status="success" if val_res.is_valid else "failed",
        metadata={"history": history, "issues": [str(i) for i in val_res.issues]},
        latency_ms=resp.latency_ms,
        errors=[str(i) for i in val_res.issues if i.severity == "error"],
    )

    # Save to disk for user inspection
    config.steps_dir.mkdir(parents=True, exist_ok=True)
    out_file = config.steps_dir / "step_01_analyze.md"
    out_file.write_text(resp.content, encoding="utf-8")

    return resp
