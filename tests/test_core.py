#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Unit tests for LOCAL_AI_CORE components."""

import shutil
import tempfile
import unittest
from pathlib import Path

from core.config import load_config
from core.bm25_rag import BM25Engine, normalize_text, tokenize, format_rag_context
from core.pipeline_session import PipelineSession
from core.validator_base import BaseValidator, build_repair_prompt, ValidationIssue
from core.exporter import extract_files_from_markdown, export_project, sanitize_relpath


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


if __name__ == "__main__":
    unittest.main()
