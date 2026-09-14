#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Unit tests for LOCAL_AI_CORE components."""

import shutil
import tempfile
import unittest
from pathlib import Path

from unittest.mock import patch, MagicMock

from core.config import load_config
from core.bm25_rag import BM25Engine, normalize_text, tokenize, format_rag_context
from core.pipeline_session import PipelineSession
from core.validator_base import BaseValidator, build_repair_prompt, ValidationIssue, run_with_repair
from core.exporter import extract_files_from_markdown, export_project, sanitize_relpath
from core.llm_client import LLMClient, LLMResponse, format_chatml, clean_llm_output
from steps.step_01_analyze import build_analyze_prompt
from steps.step_02_solve import build_solve_prompt, SolveValidator
from steps.step_03_verify import build_verify_prompt, VerifyValidator


class TestCore(unittest.TestCase):

    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_config_loading(self):
        cfg = load_config()
        self.assertEqual(cfg.server_port, 8080)
        self.assertEqual(cfg.ctx_size, 6144)
        self.assertEqual(cfg.threads, 8)
        self.assertEqual(cfg.gpu_layers, 12)
        self.assertTrue(cfg.no_mmap)
        self.assertFalse(cfg.no_kv_offload)
        self.assertAlmostEqual(cfg.temperature, 0.05)
        self.assertEqual(cfg.get_step_tokens("analyze"), cfg.step_01_max_tokens)

    def test_bm25_vietnamese_and_english(self):
        engine = BM25Engine()
        # Test diacritics normalization
        self.assertEqual(normalize_text("Hệ thống Tìm kiếm Dữ liệu"), "he thong tim kiem du lieu")
        tokens = tokenize("Hệ thống lưu trữ phân tán")
        self.assertIn("phan", tokens)
        self.assertIn("tan", tokens)

        # Indexing test docs
        doc_dir = self.temp_dir / "rag_docs"
        doc_dir.mkdir()
        (doc_dir / "00_SYSTEM_RULES.md").write_text("# SYSTEM RULES\n\nMust use thread safe in memory storage.\n", encoding="utf-8")
        (doc_dir / "ttl_guide.md").write_text("# TTL GUIDE\n\nExpire keys using background cleanup thread.\n", encoding="utf-8")

        count = engine.index_documents_from_dir(doc_dir)
        self.assertGreaterEqual(count, 2)

        results = engine.retrieve("Làm sao để expire key và dọn dẹp?", top_k=2)
        self.assertGreaterEqual(len(results), 1)
        formatted = format_rag_context(results)
        self.assertIn("RAG CONTEXT", formatted)

    def test_pipeline_session_and_reset(self):
        sess = PipelineSession(task_text="Build Key Value store")
        sess.record_step("analyze", "Analysis result content", status="success")
        sess.record_step("solve", "Solution code content", status="success")

        self.assertEqual(sess.get_step_content("analyze"), "Analysis result content")
        acc = sess.get_accumulated_context()
        self.assertIn("OUTPUT FROM ANALYZE", acc)
        self.assertIn("OUTPUT FROM SOLVE", acc)

        # Checkpoint serialization
        cp_path = self.temp_dir / "test_session.json"
        sess.save_checkpoint(cp_path)
        self.assertTrue(cp_path.exists())

        restored = PipelineSession.load_checkpoint(cp_path)
        self.assertEqual(restored.task_text, "Build Key Value store")
        self.assertEqual(restored.get_step_content("analyze"), "Analysis result content")

        # Test atomic reset
        out_sub = self.temp_dir / "out"
        out_sub.mkdir()
        (out_sub / "stale.txt").write_text("old data")
        sess.atomic_reset(output_dirs=[out_sub])
        self.assertEqual(len(sess.steps), 0)
        self.assertFalse((out_sub / "stale.txt").exists())

    def test_validator_and_repair_prompt(self):
        validator = BaseValidator(required_sections=["Architecture", "Implementation"])

        # Invalid: missing sections
        bad_output = "Here is some code without required sections."
        res = validator.validate(bad_output)
        self.assertFalse(res.is_valid)
        self.assertTrue(res.has_errors)

        # Valid
        good_output = "## Architecture\nGood design.\n\n## Implementation\n```python\nprint('hello')\n```"
        res_good = validator.validate(good_output)
        self.assertTrue(res_good.is_valid)

        # Repair prompt construction
        repair_prompt = build_repair_prompt(
            task_text="Create something",
            step_name="test_step",
            previous_output=bad_output,
            issues=[ValidationIssue(code="MISSING_SECTION", message="Section 'Architecture' missing", severity="error")],
        )
        self.assertIn("CRITICAL REPAIR REQUEST", repair_prompt)
        self.assertIn("Section 'Architecture' missing", repair_prompt)

    def test_file_exporter(self):
        # Path sanitizer
        self.assertIsNone(sanitize_relpath("../evil.py"))
        self.assertEqual(str(sanitize_relpath("src/core/store.py")).replace("\\", "/"), "src/core/store.py")

        # Markdown extractor
        sample_markdown = """
## Implementation
### File: src/server.py
```python
# FILE: src/server.py
class Server:
    pass
```

```sql
-- FILE: db/schema.sql
CREATE TABLE items (id INT);
```
"""
        extracted = extract_files_from_markdown(sample_markdown)
        self.assertIn("src/server.py", extracted)
        self.assertIn("db/schema.sql", extracted)
        self.assertIn("class Server:", extracted["src/server.py"])

        # Project exporter
        sess = PipelineSession(task_text="Test export")
        sess.set_artifact("config.json", '{"port": 8080}')
        sess.record_step("solve", sample_markdown, status="success")

        target_dir = self.temp_dir / "final"
        exported = export_project(sess, target_dir)
        self.assertGreaterEqual(len(exported), 3)
        self.assertTrue((target_dir / "src" / "server.py").exists())
        self.assertTrue((target_dir / "config.json").exists())
        self.assertTrue((target_dir / "export_manifest.json").exists())

    def test_config_fallback_defaults(self):
        nonexistent = self.temp_dir / "nonexistent.ini"
        cfg = load_config(nonexistent)
        self.assertEqual(cfg.ctx_size, 6144)
        self.assertEqual(cfg.threads, 8)
        self.assertFalse(cfg.no_kv_offload)

    def test_format_chatml(self):
        # Basic formatting
        formatted = format_chatml("System instruction", "User query")
        expected = (
            "<|im_start|>system\nSystem instruction<|im_end|>\n"
            "<|im_start|>user\nUser query<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
        self.assertEqual(formatted, expected)

        # Empty system prompt
        no_sys = format_chatml("", "User query")
        self.assertEqual(no_sys, "<|im_start|>user\nUser query<|im_end|>\n<|im_start|>assistant\n")

        # Already formatted prompt should not be double-wrapped
        already = "<|im_start|>system\nCustom<|im_end|>\n<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n"
        self.assertEqual(format_chatml("ignored", already), already)

    def test_step_prompts_chatml(self):
        p1 = build_analyze_prompt("Analyze task", "Rule 1")
        self.assertIn("<|im_start|>system", p1)
        self.assertIn("<|im_start|>user", p1)
        self.assertIn("<|im_start|>assistant\n", p1)
        self.assertIn("Analyze task", p1)

        p2 = build_solve_prompt("Solve task", "Analyze output", "Context")
        self.assertIn("<|im_start|>system", p2)
        self.assertIn("<|im_start|>user", p2)
        self.assertIn("<|im_start|>assistant\n", p2)
        self.assertIn("Solve task", p2)

        p3 = build_verify_prompt("Verify task", "Analyze output", "Solve output")
        self.assertIn("<|im_start|>system", p3)
        self.assertIn("<|im_start|>user", p3)
        self.assertIn("<|im_start|>assistant\n", p3)
        self.assertIn("Verify task", p3)

        p_repair = build_repair_prompt("Repair task", "step_01_analyze", "prev output", [
            ValidationIssue(code="ERR", message="Broken section", severity="error")
        ])
        self.assertIn("<|im_start|>system", p_repair)
        self.assertIn("<|im_start|>user", p_repair)
        self.assertIn("<|im_start|>assistant\n", p_repair)
        self.assertIn("Broken section", p_repair)

    def test_validator_trailing_codeblocks_pruning(self):
        validator = BaseValidator(required_sections=["Summary"])
        # Output with repetitive empty backticks at the end (which would cause odd backtick count error if not pruned)
        raw_output = """## Summary
Valid analysis text.

```python
# code
print("done")
```
```
```
```"""
        # There are 5 ``` in raw_output (odd, unclosed). Pruning trailing empty backticks removes the extra ```\n```\n```
        res = validator.validate(raw_output)
        self.assertTrue(res.is_valid, f"Expected valid after pruning, got: {res.issues}")
        self.assertNotIn("```\n```\n```", res.cleaned_content)
        self.assertTrue(res.cleaned_content.endswith("```"))

        # SolveValidator also uses cleaned_content
        solve_val = SolveValidator()
        solve_output = """## Solution Architecture
Design notes.

## Implementation Code
```python
# FILE: main.py
def run(): pass
```
```
```"""
        s_res = solve_val.validate(solve_output)
        self.assertTrue(s_res.is_valid, f"Solve validation failed: {s_res.issues}")
        self.assertEqual(s_res.cleaned_content.count("```"), 2)

    def test_llm_client_prompt_saving_and_no_conversation(self):
        client = LLMClient()
        save_file = self.temp_dir / "saved_prompt.txt"

        # 1. generate_via_server saves prompt when prompt_save_path is passed
        mock_resp = MagicMock()
        mock_resp.read.return_value = b'{"content": "response", "tokens_evaluated": 10, "tokens_predicted": 5}'
        mock_resp.__enter__.return_value = mock_resp
        with patch("urllib.request.urlopen", return_value=mock_resp):
            client.generate_via_server("Test prompt for server", prompt_save_path=save_file)
        self.assertTrue(save_file.exists())
        self.assertEqual(save_file.read_text(encoding="utf-8"), "Test prompt for server")

        # 2. generate() passes prompt_save_path to server when server is online
        save_file2 = self.temp_dir / "saved_prompt2.txt"
        with patch.object(client, "is_server_available", return_value=True), \
             patch.object(client, "generate_via_server", wraps=client.generate_via_server) as mock_gen_server, \
             patch("urllib.request.urlopen", return_value=mock_resp):
            client.generate("Generate prompt", prompt_save_path=save_file2)
            mock_gen_server.assert_called_once()
            self.assertEqual(mock_gen_server.call_args.kwargs.get("prompt_save_path"), save_file2)
        self.assertTrue(save_file2.exists())

        # 3. CLI args must NOT contain --no-conversation
        with patch("subprocess.Popen") as mock_popen:
            mock_proc = MagicMock()
            mock_proc.communicate.return_value = ("CLI Output", "")
            mock_proc.returncode = 0
            mock_popen.return_value = mock_proc
            client.generate_via_cli("CLI prompt")
            cmd_args = mock_popen.call_args[0][0]
            self.assertNotIn("--no-conversation", cmd_args)
            self.assertIn("--single-turn", cmd_args)

        # 4. CLI mode creates parent directory for custom nested prompt_save_path
        nested_save = self.temp_dir / "deep" / "nested" / "path" / "cli_prompt.txt"
        with patch("subprocess.Popen") as mock_popen:
            mock_proc = MagicMock()
            mock_proc.communicate.return_value = ("CLI Output", "")
            mock_proc.returncode = 0
            mock_popen.return_value = mock_proc
            client.generate_via_cli("Nested prompt", prompt_save_path=nested_save)
            self.assertTrue(nested_save.exists())
            self.assertEqual(nested_save.read_text(encoding="utf-8"), "Nested prompt")

    def test_format_chatml_edge_cases(self):
        # Body containing <|im_start|> in user query must NOT bypass ChatML formatting
        prompt = format_chatml("System instructions", "How do I parse <|im_start|>user tokens?")
        self.assertTrue(prompt.startswith("<|im_start|>system\nSystem instructions<|im_end|>\n<|im_start|>user\n"))
        self.assertTrue(prompt.endswith("<|im_start|>assistant\n"))
        self.assertIn("How do I parse <|im_start|>user tokens?", prompt)

        # Defensive handling of None inputs
        self.assertEqual(format_chatml(None, "User"), "<|im_start|>user\nUser<|im_end|>\n<|im_start|>assistant\n")
        self.assertEqual(format_chatml("Sys", None), "<|im_start|>system\nSys<|im_end|>\n<|im_start|>user\n<|im_end|>\n<|im_start|>assistant\n")
        self.assertEqual(format_chatml(None, None), "<|im_start|>user\n<|im_end|>\n<|im_start|>assistant\n")

    def test_validator_with_trailing_im_end(self):
        validator = BaseValidator(required_sections=["Section"])
        # Output having trailing backticks followed by <|im_end|>
        raw = """## Section
Code here:
```python
def ok(): pass
```
```
```<|im_end|>"""
        res = validator.validate(raw)
        self.assertTrue(res.is_valid, f"Expected valid, got {res.issues}")
        self.assertNotIn("<|im_end|>", res.cleaned_content)
        self.assertEqual(res.cleaned_content.count("```"), 2)

    def test_clean_llm_output_chatml_tags(self):
        raw = "```python\nprint(123)\n```\n<|im_end|>\n[ Prompt: 150 t/s ]"
        cleaned = clean_llm_output(raw)
        self.assertNotIn("<|im_end|>", cleaned)
        self.assertNotIn("[ Prompt:", cleaned)
        self.assertEqual(cleaned.strip(), "```python\nprint(123)\n```")

        # Prompt echo followed by response containing chatml explanation
        raw_echo = (
            "<|im_start|>system\nYou are helpful.<|im_end|>\n"
            "<|im_start|>user\nHow do I prompt?<|im_end|>\n"
            "<|im_start|>assistant\n"
            "Use format:\n<|im_start|>user\nHello<|im_end|>"
        )
        cleaned_echo = clean_llm_output(raw_echo)
        self.assertIn("Use format:\n<|im_start|>user\nHello", cleaned_echo)

    def test_format_chatml_multilingual(self):
        sys_vi = "Bạn là chuyên gia lập trình Python."
        user_vi = "Viết hàm tính giai thừa bằng đệ quy."
        formatted = format_chatml(sys_vi, user_vi)
        self.assertIn(f"<|im_start|>system\n{sys_vi}<|im_end|>", formatted)
        self.assertIn(f"<|im_start|>user\n{user_vi}<|im_end|>", formatted)
        self.assertTrue(formatted.endswith("<|im_start|>assistant\n"))

    def test_run_with_repair_cleaned_content_sync(self):
        # Generator returning dirty text with trailing repetitive backticks
        dirty_output = "## Section\nContent\n```python\npass\n```\n```\n```"
        mock_gen = MagicMock(return_value=LLMResponse(
            content=dirty_output, mode="server", latency_ms=10.0
        ))
        validator = BaseValidator(required_sections=["Section"])
        resp, val_res, history = run_with_repair(
            step_name="test_sync",
            generator_fn=mock_gen,
            initial_prompt="do task",
            validator=validator,
            task_text="task",
        )
        self.assertTrue(val_res.is_valid)
        self.assertEqual(resp.content, val_res.cleaned_content)
        self.assertNotIn("```\n```\n```", resp.content)

    def test_validator_tuple_type_hints(self):
        import typing
        hints = typing.get_type_hints(run_with_repair)
        self.assertIn("return", hints)
        self.assertEqual(getattr(hints["return"], "__origin__", None), tuple)

    def test_resolve_bin_candidate_expansion(self):
        # Verify fallback to llama-linux/bin.exe and llama-vulkan/bin.exe
        fake_base = self.temp_dir / "fake_base"
        fake_base.mkdir()
        linux_dir = fake_base / "llama-linux"
        linux_dir.mkdir()
        dummy_exe = linux_dir / "llama-server.exe"
        dummy_exe.write_text("binary", encoding="utf-8")

        ini = fake_base / "test.ini"
        ini.write_text("[paths]\nllama_server = nonexistent/llama-server\n", encoding="utf-8")

        with patch("core.config.BASE_DIR", fake_base), patch("shutil.which", return_value=None):
            cfg = load_config(ini)
            self.assertEqual(cfg.llama_server_path, dummy_exe)

    def test_llm_client_cli_nonzero_exit_code_combined_error(self):
        client = LLMClient()

        # 1. Non-zero exit code with error in stdout only
        with patch("subprocess.Popen") as mock_popen:
            mock_proc = MagicMock()
            mock_proc.communicate.return_value = ("Failed to load the model: file corrupted", "")
            mock_proc.returncode = 1
            mock_popen.return_value = mock_proc
            with self.assertRaises(RuntimeError) as ctx:
                client.generate_via_cli("Test prompt")
            self.assertIn("llama-cli failed with exit code 1", str(ctx.exception))
            self.assertIn("Failed to load the model: file corrupted", str(ctx.exception))

        # 2. Non-zero exit code with error in both stdout and stderr
        with patch("subprocess.Popen") as mock_popen:
            mock_proc = MagicMock()
            mock_proc.communicate.return_value = ("stdout detail", "stderr fatal error")
            mock_proc.returncode = 2
            mock_popen.return_value = mock_proc
            with self.assertRaises(RuntimeError) as ctx:
                client.generate_via_cli("Test prompt")
            self.assertIn("llama-cli failed with exit code 2", str(ctx.exception))
            self.assertIn("stderr fatal error", str(ctx.exception))
            self.assertIn("stdout detail", str(ctx.exception))

    def test_pipeline_session_atomic_reset_preserves_gitkeep(self):
        sess = PipelineSession(task_text="Test gitkeep preservation")
        out_sub = self.temp_dir / "out_keep"
        out_sub.mkdir()
        keep_file = out_sub / ".gitkeep"
        keep_file.write_text("", encoding="utf-8")
        stale_file = out_sub / "stale_step.md"
        stale_file.write_text("stale output", encoding="utf-8")
        stale_dir = out_sub / "sub_artifacts"
        stale_dir.mkdir()
        (stale_dir / "old.txt").write_text("old", encoding="utf-8")

        sess.atomic_reset(output_dirs=[out_sub])
        self.assertTrue(keep_file.exists(), ".gitkeep was deleted during atomic_reset")
        self.assertFalse(stale_file.exists(), "stale file was not deleted during atomic_reset")
        self.assertFalse(stale_dir.exists(), "stale directory was not deleted during atomic_reset")

    def test_populate_session_from_disk_validation(self):
        from core.orchestrator import populate_session_from_disk
        sess = PipelineSession(task_text="Test disk populate validation")
        steps_dir = self.temp_dir / "test_steps"
        steps_dir.mkdir()

        cfg = MagicMock()
        cfg.steps_dir = steps_dir

        # 1. Empty step file should not be recorded
        (steps_dir / "step_01_analyze.md").write_text("   \n\t  ", encoding="utf-8")
        populate_session_from_disk(sess, cfg)
        self.assertIsNone(sess.get_step_content("analyze"))

        # 2. Fatal error marker "Failed to load the model" should not be recorded
        (steps_dir / "step_01_analyze.md").write_text("Failed to load the model: GGUF magic invalid", encoding="utf-8")
        populate_session_from_disk(sess, cfg)
        self.assertIsNone(sess.get_step_content("analyze"))

        # 3. Fatal error marker "llama-cli failed with exit code" should not be recorded
        (steps_dir / "step_01_analyze.md").write_text("llama-cli failed with exit code 1: out of memory", encoding="utf-8")
        populate_session_from_disk(sess, cfg)
        self.assertIsNone(sess.get_step_content("analyze"))

        # 4. Valid step file should be recorded
        (steps_dir / "step_01_analyze.md").write_text("## 1. Requirements\nValid analysis content.", encoding="utf-8")
        populate_session_from_disk(sess, cfg)
        self.assertEqual(sess.get_step_content("analyze"), "## 1. Requirements\nValid analysis content.")
        self.assertEqual(sess.steps["analyze"].status, "success")

        # 5. Fallback to candidate 2 if candidate 1 contains fatal error marker
        sess_fallback = PipelineSession(task_text="Test candidate fallback")
        (steps_dir / "step_02_solve.md").write_text("Failed to load the model", encoding="utf-8")
        (steps_dir / "step_solve.md").write_text("## Solution\nValid fallback content.", encoding="utf-8")
        populate_session_from_disk(sess_fallback, cfg)
        self.assertEqual(sess_fallback.get_step_content("solve"), "## Solution\nValid fallback content.")

        # 6. Existing session content should not be overwritten
        sess_existing = PipelineSession(task_text="Test existing preservation")
        sess_existing.record_step("analyze", "In-memory analysis", status="success")
        populate_session_from_disk(sess_existing, cfg)
        self.assertEqual(sess_existing.get_step_content("analyze"), "In-memory analysis")

        # 7. Valid markdown legitimately discussing "Failed to load the model" should be recorded
        sess_doc = PipelineSession(task_text="Test troubleshooting documentation")
        (steps_dir / "step_01_analyze.md").write_text(
            "## 1. Problem Summary\nFix the 'Failed to load the model' issue.\n\n## 2. Core Functional Requirements\n- Check VRAM",
            encoding="utf-8"
        )
        populate_session_from_disk(sess_doc, cfg)
        self.assertIsNotNone(sess_doc.get_step_content("analyze"))
        self.assertIn("Failed to load the model", sess_doc.get_step_content("analyze"))

        # 8. Valid code file legitimately handling error in step 3 should be recorded
        sess_code = PipelineSession(task_text="Test code error handling")
        (steps_dir / "step_03_verify.md").write_text(
            "## 1. Verification Checklist\n- Tested\n\n## 3. Final Production-Ready Files\n```python\n# FILE: app.py\nprint('error: failed to load')\n```",
            encoding="utf-8"
        )
        populate_session_from_disk(sess_code, cfg)
        self.assertIsNotNone(sess_code.get_step_content("verify"))
        self.assertIn("error: failed to load", sess_code.get_step_content("verify"))

    def test_is_fatal_error_content(self):
        from core.orchestrator import is_fatal_error_content
        # True cases (Fatal crashes / invalid dumps)
        self.assertTrue(is_fatal_error_content(""))
        self.assertTrue(is_fatal_error_content("   \n\t "))
        self.assertTrue(is_fatal_error_content("Failed to load the model: file corrupted"))
        self.assertTrue(is_fatal_error_content("llama-cli failed with exit code 1: OOM"))
        self.assertTrue(is_fatal_error_content("[LOG] error: failed to load model 'x.gguf'"))

        # False cases (Legitimate markdown documents)
        self.assertFalse(is_fatal_error_content("## 1. Problem Summary\nHow to solve failed to load the model"))
        self.assertFalse(is_fatal_error_content("# Guide\nIf you see error: failed to load, check path."))
        self.assertFalse(is_fatal_error_content("## 3. Implementation\n```python\nprint('llama-cli failed with exit code')\n```"))


if __name__ == "__main__":
    unittest.main()
