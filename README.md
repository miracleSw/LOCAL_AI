# LOCAL_AI_CORE

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Inference Engine](https://img.shields.io/badge/Engine-llama.cpp%20(Vulkan)-red.svg)](https://github.com/ggerganov/llama.cpp)
[![Default Model](https://img.shields.io/badge/Model-Qwen2.5--Coder--3B--Q4__K__M-green.svg)](https://huggingface.co/Qwen/Qwen2.5-Coder-3B-Instruct-GGUF)
[![Platform](https://img.shields.io/badge/Platform-Windows%20x64-lightgrey.svg)]()
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A lightweight, deterministic pipeline framework designed to run 3B-class code models on resource-constrained hardware (e.g., 2 GB VRAM / 8 GB system RAM).

Built as an offline alternative to heavy agent frameworks, `LOCAL_AI_CORE` eliminates runtime bloat by combining pure-Python BM25 retrieval, an in-memory session coordinator, dual-mode execution (HTTP daemon + standalone CLI fallback), and automated validation with self-repair loops.

---

## Key Technical Decisions

* **Zero Heavyweight Dependencies:** Uses only Python's standard library for BM25 text search, text tokenization, diacritics stripping, and schema validation. Requires no external vector databases (Chroma, FAISS) or heavyweight runtimes (PyTorch, Transformers), saving 1.5–2.5 GB of system RAM on startup.
* **Dual-Mode Runtime (Zero Cold-Start):**
  * *Primary:* Connects to a persistent `llama-server.exe` daemon over local HTTP (`/completion`). The model weights stay resident in RAM/VRAM, reducing per-step model reload time from ~10–15s to zero and preserving KV-cache across prompts.
  * *Fallback:* Automatically fails over to standalone `llama-cli.exe` execution if the server process is offline.
* **Memory-Isolated Pipeline State:** Intermediate variables and outputs reside in an in-memory `PipelineSession`. This eliminates cross-run contamination caused by reading stale intermediate markdown files off the filesystem.
* **Deterministic Validation & Self-Repair:** Every generation step passes through strict syntax and structural validators. When outputs violate constraints, a targeted diagnostic prompt is constructed to trigger an automated self-repair loop without bloating the context window.
* **Vulkan Offline Backend:** Uses prebuilt `llama.cpp` Vulkan binaries (`ggml-vulkan.dll`), enabling hardware acceleration across NVIDIA, AMD (Radeon/RDNA), and Intel (Arc/Iris/iGPU) graphics cards on Windows without requiring CUDA SDK installations.

---

## Architecture Overview

```
                                    +-----------------------+
                                    | input/current_task.txt|
                                    +-----------+-----------+
                                                |
                                                v
+------------------------+          +-----------+-----------+          +-------------------------+
|      rag/active/       |          |      Orchestrator     |          |     PipelineSession     |
| (Rulebooks & Skeletons)| -------> |  (core/orchestrator)  | <------> |  (In-Memory State &     |
|   [Pure BM25 Engine]   |          +-----------+-----------+          |   Atomic Reset Engine)  |
+------------------------+                      |                      +-------------------------+
                                                v
                                    +-----------+-----------+
                                    |       LLMClient       |
                                    |    (core/llm_client)  |
                                    +-----+-----------+-----+
                                          |           |
                           [Server Online]|           |[Fallback Offline]
                                          v           v
                             +----------------+   +---------------+
                             |  llama-server  |   |   llama-cli   |
                             | (Port 8080)    |   |  (Standalone) |
                             +----------------+   +---------------+
                                          \           /
                                           v         v
                                    +-----------------------+
                                    |    BaseValidator      |
                                    | (Syntax/Schema Check) |
                                    +-----------+-----------+
                                          |           |
                                    [Fail]|           |[Pass]
                                          v           v
                                    +-----------+ +------------------------+
                                    |Self-Repair| |  output/final_project/ |
                                    |   Loop    | |   (Exported Source)    |
                                    +-----------+ +------------------------+
```

---

## Hardware Profiling & Resource Budget

The default configuration is specifically tuned for entry-level workstations (tested on Intel Core i7-12700, 8 GB RAM, AMD Radeon RX 6300 2 GB GDDR6):

### 1. VRAM Budget (2048 MB Total)

| Component | Allocation | Rationale |
| :--- | :--- | :--- |
| **Model Weights (20 Layers)** | ~1,140 MB | 20 out of 36 transformer blocks offloaded via Vulkan. |
| **Vulkan Compute Scratch** | ~305 MB | Execution graph buffers with 512 ubatch size. |
| **KV Cache (20 Layers @ 6144 ctx)** | ~120 MB | Lightweight due to Qwen 2.5 3B GQA (8:1 ratio, 2 KV heads). |
| **Windows DWM & Display Driver** | ~350–400 MB | OS desktop compositor baseline. |
| **Total VRAM Allocated** | **~1,915 MB** | **Optimal ~93% VRAM usage**, sustaining 17.5–18.2 tokens/s. |

> **Technical Warning:** The performance cliff occurs at `gpu_layers >= 23`. Exceeding 22 layers causes VRAM overflow, forcing Windows WDDM to page memory over the PCIe 4.0 x4 bus (bandwidth collapses from 64 GB/s down to <8 GB/s), which plunges token generation speed from **18.3 t/s to 11.1 t/s**. Setting `gpu_layers = 20` represents the maximum safe ceiling that prevents OS crashes and PCIe bottlenecks.

### 2. Flash Attention Optimization on Vulkan
* With Flash Attention enabled (`auto`/`on`): Prompt processing at 2048 ctx achieves only **143.42 tokens/s**.
* With Flash Attention disabled (`--flash-attn off`): Prompt processing jumps to **333.90 – 360 tokens/s** (**2.32x faster** / +133%). Standard Vulkan attention shaders execute with significantly higher parallelism on AMD RDNA2 compute units.

### 3. System RAM Budget (8192 MB Total)

| Component | Allocation |
| :--- | :--- |
| **Windows OS & Background Services** | ~3,400–3,800 MB |
| **CPU Model Layers (16 Layers)** | ~860 MB |
| **CPU KV Cache (16 Layers @ 6144 ctx)** | ~96 MB |
| **Python Runtime + BM25 Index** | ~80 MB |
| **Total System RAM Footprint** | **~4,650 MB** (~3.4 GB safe headroom) |

### 4. CPU Thread Allocation on Hybrid Architectures
On Intel Alder Lake / Raptor Lake processors (such as the i7-12700 with 8 Performance cores and 4 Efficient cores), set `threads = 8`. GGML matrix multiplications use synchronous barriers across all active threads per layer; setting `threads > 8` forces threads onto lower-IPC E-cores, causing fast P-cores to stall at synchronization barriers.

---

## Repository Structure

```text
LOCAL_AI_CORE/
├── config.ini                  # Central system & model configuration
├── 00_MENU.bat                 # UTF-8 interactive terminal dashboard
├── run_server.bat              # Background Vulkan llama-server launcher
├── stop_server.bat             # Clean process termination script
├── README.md                   # Technical documentation
│
├── core/                       # Framework modules
│   ├── config.py               # Typed configuration loader & path resolver
│   ├── bm25_rag.py             # Pure-Python BM25 search engine
│   ├── llm_client.py           # Dual-mode server/CLI inference client
│   ├── pipeline_session.py     # In-memory session manager & checkpoint serialiser
│   ├── validator_base.py       # Base validator interface & repair prompt builder
│   ├── exporter.py             # Code block extractor & project manifest generator
│   └── orchestrator.py         # CLI entrypoint and pipeline execution runner
│
├── steps/                      # Modular pipeline step definitions
│   ├── step_01_analyze.py      # Task decomposition, requirements, and edge cases
│   ├── step_02_solve.py        # Core technical solution & code generation
│   └── step_03_verify.py       # Verification, formatting, and file packaging
│
├── models/
│   └── qwen25-coder-3b-q4km.gguf   # Default 3B GGUF quantized model
│
├── llama-vulkan/               # llama.cpp binaries with Vulkan runtime
│   ├── llama-server.exe
│   ├── llama-cli.exe
│   └── ggml-vulkan.dll
│
├── rag/active/                 # Active knowledge base documents
│   ├── 00_SYSTEM_RULES.md
│   └── 01_PROBLEM_SOLVING_FRAMEWORK.md
│
├── input/
│   └── current_task.txt        # Input task specification file
│
└── output/
    ├── steps/                  # Step-by-step raw markdown logs
    ├── final_project/          # Final exported code files
    └── logs/                   # Execution metrics and session JSON checkpoints
```

---

## Quick Start

### Prerequisites
* **Windows 10 / 11 (64-bit)** or **Linux Mint / Ubuntu (x86_64)**
* Python 3.10+ (must be added to `PATH` or available via `python3`)
* Vulkan-compatible graphics driver (NVIDIA, AMD, or Intel)

### Option 1: Automated Setup (Recommended)
* **On Windows**: Double-click `SETUP.bat` to verify environment, download model, and launch `00_MENU.bat`.
* **On Linux Mint / Ubuntu**: Run `bash setup.sh` (or `chmod +x *.sh && ./setup.sh`) to auto-configure and launch `menu.sh`.

### Option 2: Interactive Menu
* **Windows**: Run `00_MENU.bat`.
* **Linux Mint / Ubuntu**: Run `./menu.sh`.
1. Press `[1]` to launch the **Llama Server** in the background or terminal.
2. Press `[15]` to chat directly in the interactive AGY CLI with real-time slash suggestions.
3. Press `[3]` to open `input/current_task.txt` and paste your task description.
4. Press `[6]` (**RUN ALL**) to execute the end-to-end pipeline.
5. Press `[13]` to open `output/final_project` and view the generated files.

### Option 2: Command-Line Interface (CLI)

```powershell
# Navigate to the project root
cd d:\Project\Model_Local_Test\LOCAL_AI_CORE

# 1. Verify system readiness and component integrity
python core/orchestrator.py --check

# 2. View current task and pipeline state
python core/orchestrator.py --status

# 3. Run the complete pipeline
python core/orchestrator.py --run-all

# 4. Alternatively, pass a task directly from the command line
python core/orchestrator.py --task "Implement a thread-safe LRU Cache with TTL in Python" --run-all

# 5. Execute an individual pipeline step
python core/orchestrator.py --step analyze
python core/orchestrator.py --step solve
python core/orchestrator.py --step verify

# 6. Extract source code files to output/final_project
python core/orchestrator.py --export

# 7. Reset workspace and clean intermediate files
python core/orchestrator.py --reset
```

---

## Configuration Reference (`config.ini`)

```ini
[paths]
llama_server = llama-vulkan/llama-server.exe
llama_cli    = llama-vulkan/llama-cli.exe
model        = models/qwen25-coder-3b-q4km.gguf
rag_dir      = rag/active
input_task   = input/current_task.txt
output_dir   = output

[server]
host                   = 127.0.0.1
port                   = 8080
endpoint               = http://127.0.0.1:8080/completion
request_timeout_seconds= 300

[llama]
ctx_size            = 6144      # Context size in tokens (multiple of 1024)
threads             = 8         # Optimal for 8 P-cores (leave E-cores free)
gpu_layers          = 12        # Safe for 2 GB VRAM (Navi 24 / RX 6300)
no_mmap             = true      # Pin model into RAM; prevents SSD page-fault stalls
no_kv_offload       = false     # Allow GPU to hold its 12-layer KV cache (~75 MB)
temperature         = 0.05      # Low temperature for deterministic code output
top_p               = 0.85
repeat_penalty      = 1.05
max_tokens          = 2048
max_runtime_seconds = 300

[rag]
top_k               = 5         # Number of BM25 chunks retrieved per step
chunk_chars         = 1200      # Target character length per chunk
overlap_chars       = 150       # Overlap between consecutive chunks
always_include_rules= true      # Always prepend core rulebooks matching prefix '00_'

[steps]
step_01_max_tokens  = 1200
step_02_max_tokens  = 2048
step_03_max_tokens  = 2048
max_repair_attempts = 2         # Max iterations for self-repair loop
```

---

## Extending the Pipeline (Custom Domains)

The framework is decoupled from any specific problem domain. To adapt it for tasks such as SQL generation, Java architecture, or REST API development:

### 1. Add Domain Knowledge
Place domain markdown rules, templates, or references in `rag/active/`. Prefix invariant rules with `00_` (e.g., `00_SQL_SERVER_2014_RULES.md`) so the BM25 engine includes them in every step prompt.

### 2. Implement a Custom Step
Create a step runner in `steps/` matching the standard signature:

```python
from core.llm_client import LLMClient
from core.pipeline_session import PipelineSession
from core.config import AppConfig

def run_custom_step(session: PipelineSession, client: LLMClient, config: AppConfig) -> bool:
    # 1. Retrieve previous outputs from the in-memory session
    previous_analysis = session.get_step_content("analyze")
    
    # 2. Build task-specific prompt
    prompt = f"Using this analysis:\n{previous_analysis}\nGenerate the SQL DDL."
    
    # 3. Generate completion (automatically routes to server or fallback CLI)
    response = client.generate(prompt, max_tokens=2048)
    
    # 4. Record step results
    session.record_step("custom_step", content=response.content, latency_ms=response.latency_ms)
    return True
```

### 3. Attach a Validator
Inherit from `BaseValidator` in `core/validator_base.py` and provide validation logic:

```python
from core.validator_base import BaseValidator, ValidationResult, ValidationIssue

class SQLSyntaxValidator(BaseValidator):
    def validate(self, output: str) -> ValidationResult:
        issues = []
        if "DROP TABLE IF EXISTS" in output.upper():
            issues.append(ValidationIssue(
                code="UNSUPPORTED_SYNTAX",
                message="Target database does not support DROP TABLE IF EXISTS.",
                severity="error",
                suggestion="Use IF OBJECT_ID('...') IS NOT NULL DROP TABLE instead."
            ))
        return ValidationResult(is_valid=(len(issues) == 0), issues=issues)
```

---

## Verification & Test Suite

The project includes an automated regression test suite using Python's standard `unittest`:

```powershell
python -m unittest discover -s tests
```

Test coverage includes:
* Strongly typed configuration loading and path resolution.
* BM25 search engine tokenization, diacritics stripping, and source-diverse scoring.
* In-memory session isolation and atomic directory cleanup.
* Structured validator diagnostic formatting and repair prompt generation.
* File export path sanitization (directory traversal prevention).

---

## License

This project is released under the [MIT License](LICENSE). Built for local, private, and deterministic offline code generation.
