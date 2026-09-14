#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
STEP 03: Verification, Constraint Validation & Final Assembly.
Performs end-to-end check against requirements, validates edge case coverage,
and compiles final clean deliverables for export.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

from core.config import AppConfig
from core.exporter import export_project
from core.llm_client import LLMClient, LLMResponse
from core.pipeline_session import PipelineSession
from core.validator_base import BaseValidator, ValidationIssue, ValidationResult, run_with_repair


class VerifyValidator(BaseValidator):
    """Validates that Step 3 contains a thorough verification audit and final files."""

    def __init__(self):
        super().__init__(required_sections=[
            "Verification Checklist",
            "Edge Case Handling",
            "Final Production-Ready Files",
        ])

    def validate(self, content: str, session: Optional[PipelineSession] = None) -> ValidationResult:
        base_res = super().validate(content, session)
        if not base_res.is_valid:
            return base_res

        issues = list(base_res.issues)
        # Check code block formatting
        code_blocks = re.findall(r"```[a-zA-Z0-9_\-\.\+]*\s*\n(.*?)\n```", content, re.DOTALL)
        if not code_blocks:
            issues.append(ValidationIssue(
                code="NO_FINAL_CODE_BLOCKS",
                message="Step 3 must assemble complete final production files in ``` blocks.",
                severity="error",
                suggestion="Include complete, finalized files in Section 3.",
            ))

        is_valid = not any(i.severity == "error" for i in issues)
        return ValidationResult(is_valid=is_valid, issues=issues, cleaned_content=content)


def build_verify_prompt(task_text: str, analyze_output: str, solve_output: str) -> str:
    """Construct prompt for Step 3."""
    return f"""You are a Lead QA Engineer and Software Verification Specialist.
Verify the solution generated in Step 2 against the constraints from Step 1 and the original task.
Ensure all edge cases are addressed, zero placeholder code remains, and present the final files.

ORIGINAL TASK:
{task_text}

STEP 1 REQUIREMENTS & CONSTRAINTS:
{analyze_output}

STEP 2 GENERATED SOLUTION:
{solve_output}

OUTPUT FORMAT REQUIRED (Use these exact markdown headers):
## 1. Verification Checklist & Constraint Matching
- [Constraint / Requirement 1]: [PASS/FAIL with brief explanation]
- [Constraint / Requirement 2]: [PASS/FAIL with brief explanation]

## 2. Edge Case Handling Review
- [Edge Case 1]: [Assessment of handling]
- [Edge Case 2]: [Assessment of handling]

## 3. Final Production-Ready Files
Assemble every final file needed for production. Make sure each file has a clean path marker:
### File: path/to/file.py
```python
# FILE: path/to/file.py
[Complete final source code]
```

## 4. Summary & Verification Status
- Status: [VERIFIED_PASSED / REQUIRES_FIX]
- Total files generated:
- Ready for immediate production run.
"""


def run_step_03(
    session: PipelineSession,
    client: LLMClient,
    config: AppConfig,
    auto_export: bool = True,
) -> LLMResponse:
    """Execute Step 3: Verification & Final Assembly."""
    task_text = session.task_text
    analyze_output = session.get_step_content("analyze") or ""
    solve_output = session.get_step_content("solve") or ""

    if not solve_output.strip():
        raise ValueError("Cannot run Step 3: Prior step 'solve' has no valid output in session.")

    prompt = build_verify_prompt(task_text, analyze_output, solve_output)
    validator = VerifyValidator()

    def _generator(p: str) -> LLMResponse:
        tokens = config.get_step_tokens("verify")
        prompt_file = config.steps_dir / "prompt_03_verify.txt"
        return client.generate(p, max_tokens=tokens, prompt_save_path=prompt_file)

    resp, val_res, history = run_with_repair(
        step_name="step_03_verify",
        generator_fn=_generator,
        initial_prompt=prompt,
        validator=validator,
        task_text=task_text,
        max_attempts=config.max_repair_attempts,
    )

    # Persist in session
    session.record_step(
        step_name="verify",
        content=resp.content,
        status="success" if val_res.is_valid else "failed",
        metadata={"history": history, "issues": [str(i) for i in val_res.issues]},
        latency_ms=resp.latency_ms,
        errors=[str(i) for i in val_res.issues if i.severity == "error"],
    )

    # Save to disk for user inspection
    config.steps_dir.mkdir(parents=True, exist_ok=True)
    out_file = config.steps_dir / "step_03_verify.md"
    out_file.write_text(resp.content, encoding="utf-8")

    # Automatically export final files
    if auto_export:
        export_project(session, config.final_project_dir)

    return resp
