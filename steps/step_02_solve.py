#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
STEP 02: Core Solution Generation & Code Synthesis.
Produces architectural design and complete, runnable implementation code.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

from core.config import AppConfig
from core.llm_client import LLMClient, LLMResponse
from core.pipeline_session import PipelineSession
from core.validator_base import BaseValidator, ValidationIssue, ValidationResult, run_with_repair


class SolveValidator(BaseValidator):
    """Validates that Step 2 contains sound architecture and well-formed code blocks."""

    def __init__(self):
        super().__init__(required_sections=[
            "Solution Architecture",
            "Implementation Code",
        ])

    def validate(self, content: str, session: Optional[PipelineSession] = None) -> ValidationResult:
        base_res = super().validate(content, session)
        if not base_res.is_valid:
            return base_res

        issues = list(base_res.issues)
        # Check presence of code block
        code_blocks = re.findall(r"```[a-zA-Z0-9_\-\.\+]*\s*\n(.*?)\n```", content, re.DOTALL)
        if not code_blocks:
            issues.append(ValidationIssue(
                code="NO_CODE_BLOCKS_FOUND",
                message="Step 2 must provide concrete implementation inside ``` code blocks.",
                severity="error",
                suggestion="Wrap code files in markdown ``` blocks with '# FILE: filename' comments.",
            ))

        is_valid = not any(i.severity == "error" for i in issues)
        return ValidationResult(is_valid=is_valid, issues=issues, cleaned_content=content)


def build_solve_prompt(task_text: str, analyze_output: str, rag_context: str = "") -> str:
    """Construct prompt for Step 2."""
    rag_block = f"\nRAG REFERENCE CONTEXT:\n{rag_context}\n" if rag_context.strip() else ""

    return f"""You are an elite principal engineer and solution implementer.
Using the problem analysis below, produce a complete, robust, production-grade implementation.
Write clean code with proper error handling, modularity, and zero placeholders (no TODOs).
{rag_block}
ORIGINAL TASK:
{task_text}

VERIFIED ARCHITECTURAL SPECIFICATION (STEP 1):
{analyze_output}

OUTPUT FORMAT REQUIRED (Use these exact markdown headers):
## 1. Solution Architecture & Design Decisions
[Explain technical design, modularity, and component responsibilities]

## 2. Implementation Code
For every code file, precede the code block with a heading or inside comment specifying the filename:
### File: path/to/file.py
```python
# FILE: path/to/file.py
[Full complete implementation]
```

## 3. Usage & Execution Instructions
[Commands to run, test, and verify the solution]
"""


def run_step_02(
    session: PipelineSession,
    client: LLMClient,
    config: AppConfig,
    rag_context: str = "",
) -> LLMResponse:
    """Execute Step 2: Core Solution Generation."""
    task_text = session.task_text
    analyze_output = session.get_step_content("analyze") or ""

    if not analyze_output.strip():
        raise ValueError("Cannot run Step 2: Prior step 'analyze' has no valid output in session.")

    prompt = build_solve_prompt(task_text, analyze_output, rag_context)
    validator = SolveValidator()

    def _generator(p: str) -> LLMResponse:
        tokens = config.get_step_tokens("solve")
        prompt_file = config.steps_dir / "prompt_02_solve.txt"
        return client.generate(p, max_tokens=tokens, prompt_save_path=prompt_file)

    resp, val_res, history = run_with_repair(
        step_name="step_02_solve",
        generator_fn=_generator,
        initial_prompt=prompt,
        validator=validator,
        task_text=task_text,
        max_attempts=config.max_repair_attempts,
    )

    # Persist in session
    session.record_step(
        step_name="solve",
        content=resp.content,
        status="success" if val_res.is_valid else "failed",
        metadata={"history": history, "issues": [str(i) for i in val_res.issues]},
        latency_ms=resp.latency_ms,
        errors=[str(i) for i in val_res.issues if i.severity == "error"],
    )

    # Save to disk for user inspection
    config.steps_dir.mkdir(parents=True, exist_ok=True)
    out_file = config.steps_dir / "step_02_solve.md"
    out_file.write_text(resp.content, encoding="utf-8")

    return resp
