#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""LOCAL_AI_CORE pipeline steps package."""

from steps.step_01_analyze import run_step_01, AnalyzeValidator
from steps.step_02_solve import run_step_02, SolveValidator
from steps.step_03_verify import run_step_03, VerifyValidator

STEP_ORDER = ["analyze", "solve", "verify"]

STEP_RUNNERS = {
    "analyze": run_step_01,
    "step_01_analyze": run_step_01,
    "1": run_step_01,
    "solve": run_step_02,
    "step_02_solve": run_step_02,
    "2": run_step_02,
    "verify": run_step_03,
    "step_03_verify": run_step_03,
    "3": run_step_03,
}

__all__ = [
    "STEP_ORDER",
    "STEP_RUNNERS",
    "run_step_01",
    "run_step_02",
    "run_step_03",
    "AnalyzeValidator",
    "SolveValidator",
    "VerifyValidator",
]
