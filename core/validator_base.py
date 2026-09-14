#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Extensible Syntax/Semantic Validator Base & Self-Repair Framework.
Provides structured diagnostic feedback and compact repair prompt generation for small 3B models.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from core.llm_client import LLMResponse
from core.pipeline_session import PipelineSession


@dataclass
class ValidationIssue:
    """Detailed information on a validation issue."""
    code: str
    message: str
    severity: str = "error"  # "error" or "warning"
    line: Optional[int] = None
    suggestion: Optional[str] = None

    def __str__(self) -> str:
        loc = f" (line {self.line})" if self.line is not None else ""
        sug = f" -> Suggestion: {self.suggestion}" if self.suggestion else ""
        return f"[{self.severity.upper()}] {self.code}{loc}: {self.message}{sug}"


@dataclass
class ValidationResult:
    """Outcome of validating step output."""
    is_valid: bool
    issues: List[ValidationIssue] = field(default_factory=list)
    cleaned_content: str = ""

    @property
    def has_errors(self) -> bool:
        return any(issue.severity == "error" for issue in self.issues)

    def error_summary(self) -> str:
        errors = [str(i) for i in self.issues if i.severity == "error"]
        return "\n".join(errors) if errors else "No errors"


class BaseValidator:
    """Base validator with common safety and format checks."""

    def __init__(self, required_sections: Optional[List[str]] = None):
        self.required_sections: List[str] = required_sections or []

    def validate(self, content: str, session: Optional[PipelineSession] = None) -> ValidationResult:
        """Run base checks. Subclasses should override and call super().validate()."""
        issues: List[ValidationIssue] = []
        text = content.strip()

        # 1. Empty content check
        if not text:
            issues.append(ValidationIssue(
                code="EMPTY_CONTENT",
                message="Model returned empty or whitespace-only response.",
                severity="error",
                suggestion="Regenerate with explicit structure requirements.",
            ))
            return ValidationResult(is_valid=False, issues=issues, cleaned_content=text)

        # 2. Check unclosed code blocks
        code_block_ticks = len(re.findall(r"```", text))
        if code_block_ticks % 2 != 0:
            issues.append(ValidationIssue(
                code="UNCLOSED_CODE_BLOCK",
                message="Markdown contains an unclosed code block (odd count of ```).",
                severity="error",
                suggestion="Close all markdown code blocks with ```.",
            ))

        # 3. Check required sections
        lower_text = text.lower()
        for sec in self.required_sections:
            sec_norm = sec.lower()
            if sec_norm not in lower_text:
                issues.append(ValidationIssue(
                    code="MISSING_REQUIRED_SECTION",
                    message=f"Required section '{sec}' was not found in the output.",
                    severity="error",
                    suggestion=f"Ensure your output includes a section header for '{sec}'.",
                ))

        is_valid = not any(i.severity == "error" for i in issues)
        return ValidationResult(is_valid=is_valid, issues=issues, cleaned_content=text)


def build_repair_prompt(
    task_text: str,
    step_name: str,
    previous_output: str,
    issues: List[ValidationIssue],
    extra_hints: Optional[str] = None,
) -> str:
    """
    Construct a compact, highly focused repair prompt.
    Does not overwhelm 3B models with full re-prompts; focuses strictly on fixing identified issues.
    """
    error_bullets = "\n".join(f"- {issue.message}" for issue in issues if issue.severity == "error")

    # Take a snippet of the previous output (up to 1200 characters) to show context
    snippet = previous_output[:1200]
    if len(previous_output) > 1200:
        snippet += "\n... [truncated for brevity] ..."

    hints_block = f"\nREPAIR GUIDANCE:\n{extra_hints}\n" if extra_hints else ""

    return f"""[CRITICAL REPAIR REQUEST]
Your previous output for step '{step_name}' failed validation checks.

ORIGINAL TASK:
{task_text}

PREVIOUS OUTPUT (EXCERPT):
```
{snippet}
```

ERRORS DETECTED:
{error_bullets}
{hints_block}
INSTRUCTIONS:
1. Fix all the errors listed above.
2. Maintain complete, correct markdown structure.
3. Output the entire corrected solution for this step now.
"""


def run_with_repair(
    step_name: str,
    generator_fn: Callable[[str], LLMResponse],
    initial_prompt: str,
    validator: BaseValidator,
    task_text: str,
    max_attempts: int = 2,
) -> Tuple[LLMResponse, ValidationResult, List[Dict[str, Any]]]:
    """
    Execute step generation with automatic validation and self-repair loop.
    Returns: (final_response, final_validation, repair_history)
    """
    repair_history: List[Dict[str, Any]] = []
    current_prompt = initial_prompt

    resp = generator_fn(current_prompt)
    val_res = validator.validate(resp.content)

    repair_history.append({
        "attempt": 1,
        "mode": resp.mode,
        "latency_ms": resp.latency_ms,
        "is_valid": val_res.is_valid,
        "issues": [str(i) for i in val_res.issues],
    })

    attempts = 1
    while not val_res.is_valid and attempts < max_attempts:
        attempts += 1
        print(f"[{step_name}] Validation failed on attempt {attempts-1}. Triggering self-repair loop (Attempt {attempts}/{max_attempts})...")

        repair_prompt = build_repair_prompt(
            task_text=task_text,
            step_name=step_name,
            previous_output=resp.content,
            issues=val_res.issues,
        )

        resp = generator_fn(repair_prompt)
        val_res = validator.validate(resp.content)

        repair_history.append({
            "attempt": attempts,
            "mode": resp.mode,
            "latency_ms": resp.latency_ms,
            "is_valid": val_res.is_valid,
            "issues": [str(i) for i in val_res.issues],
        })

    return resp, val_res, repair_history
