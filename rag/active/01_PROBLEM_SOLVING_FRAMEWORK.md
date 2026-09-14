# PROBLEM SOLVING & REASONING FRAMEWORK

## Overview
This framework governs how complex engineering tasks are solved through the 3-phase pipeline:
1. Analysis & Constraints
2. Core Solution Synthesis
3. Verification & Final Assembly

## Phase 1: Analysis & Constraints
- Extract explicit functional requirements.
- Identify implicit technical constraints (runtime, memory, platform, dependencies).
- Formulate input/output schemas and boundary conditions.
- Enumerate edge cases (empty collections, invalid types, network timeouts, special characters).

## Phase 2: Solution Synthesis
- Apply clean architectural patterns appropriate for the domain.
- Separate business domain logic from infrastructure and I/O.
- Provide clear comments explaining non-obvious algorithms or decisions.
- Format all files cleanly with explicit `# FILE: <path>` tags.

## Phase 3: Verification & Assembly
- Systematically cross-examine the generated code against the Phase 1 constraints.
- Verify that every specified edge case has defensive handling.
- Ensure all artifacts are packaged cleanly for export.
