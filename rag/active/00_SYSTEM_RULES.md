# LOCAL_AI_CORE - SYSTEM GOLDEN RULES

## 1. Modular Architecture & Small-Model Principles
- Maintain strict modularity across all components.
- Do NOT jump directly to code without first analyzing constraints, inputs, and outputs.
- Break down complex logic into small, testable, self-contained functions or classes.

## 2. Zero Hallucination & Code Completeness
- Never generate incomplete code, ellipsis (`...`), or placeholder comments (`// TODO`, `# Add logic here`).
- Every generated file must be 100% syntactically valid and runnable out of the box.
- All dependencies must be either Python standard library or standard language primitives unless explicitly requested.

## 3. Strict File Marker Formatting
- Every code block intended for export MUST explicitly declare its target relative file path using the standard comment format on the first or second line:
  ```python
  # FILE: src/main.py
  ```
  or for other languages:
  ```sql
  -- FILE: schema.sql
  ```
  ```json
  // FILE: config.json
  ```
- File paths must always be relative (no leading `/`, no drive letter `C:\`, no `..`).

## 4. Defensive Engineering & Edge Cases
- Always check for empty inputs, boundary values, null pointers, and missing files.
- Return informative, typed error messages rather than failing silently or crashing abruptly.
- When validating outputs, prioritize actionable feedback for the self-repair loop.
