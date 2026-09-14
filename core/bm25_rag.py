#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pure-Python BM25 Retrieval-Augmented Generation (RAG) engine.
Zero external dependencies (no chromadb, no faiss, no torch).
Supports Vietnamese and English text tokenization, diacritics normalization,
heading-based document splitting, and source-diverse scoring.
"""
from __future__ import annotations

import math
import os
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

STOPWORDS: Set[str] = {
    # Vietnamese common stopwords
    "la", "va", "cua", "cho", "co", "cac", "mot", "nhung", "trong", "khi",
    "neu", "thi", "de", "ra", "hay", "viet", "tao", "duoc", "voi", "bang",
    "can", "the", "sao", "nao", "ve", "tu", "vao", "len", "xuong", "nhu",
    "nay", "do", "mot", "hai", "ba", "bon", "nam", "cac", "nhung", "cung",
    # English common stopwords
    "a", "an", "the", "and", "or", "in", "on", "at", "to", "for", "with",
    "by", "about", "against", "between", "into", "through", "during", "before",
    "after", "above", "below", "from", "up", "down", "is", "are", "was",
    "were", "be", "been", "being", "have", "has", "had", "do", "does", "did",
    "this", "that", "these", "those", "it", "its", "as", "if", "each",
}


@dataclass
class Chunk:
    id: str
    source: str
    title: str
    text: str
    tokens: List[str]


def normalize_text(text: str) -> str:
    """Normalize text: strip diacritics, lowercase, replace special chars with space."""
    text = unicodedata.normalize("NFD", text)
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
    text = text.lower().replace("đ", "d")
    text = re.sub(r"[^a-z0-9_+#.\s-]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def tokenize(text: str) -> List[str]:
    """Tokenize normalized string, skipping short tokens and stopwords."""
    norm = normalize_text(text)
    tokens: List[str] = []
    for t in norm.split():
        if len(t) < 2 or t in STOPWORDS:
            continue
        tokens.append(t)
    return tokens


def split_by_markdown_headings(text: str) -> List[Tuple[str, str]]:
    """Split markdown into sections based on # or ## headers."""
    sections: List[Tuple[str, List[str]]] = []
    current_title = "OVERVIEW"
    current_lines: List[str] = []

    for line in text.splitlines():
        if re.match(r"^\s{0,3}#{1,4}\s+", line):
            if current_lines:
                sections.append((current_title, current_lines))
            current_title = line.strip("# ").strip() or "SECTION"
            current_lines = [line]
        else:
            current_lines.append(line)

    if current_lines:
        sections.append((current_title, current_lines))

    result: List[Tuple[str, str]] = []
    for title, lines in sections:
        body = "\n".join(lines).strip()
        if body:
            result.append((title, body))
    return result


def chunk_text_by_chars(text: str, max_chars: int = 1200, overlap: int = 150) -> List[str]:
    """Split text into character windows with overlap."""
    if max_chars <= 0:
        raise ValueError("max_chars must be positive")
    if overlap >= max_chars:
        overlap = max_chars // 4

    clean = text.strip()
    if len(clean) <= max_chars:
        return [clean] if clean else []

    chunks: List[str] = []
    start = 0
    while start < len(clean):
        end = min(len(clean), start + max_chars)
        # Avoid splitting mid-word if possible
        if end < len(clean):
            space_idx = clean.rfind(" ", start, end)
            if space_idx > start + (max_chars // 2):
                end = space_idx
        chunk = clean[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(clean):
            break
        start = max(0, end - overlap)
    return chunks


class BM25Engine:
    """Pure Python BM25 index and retrieval engine."""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.chunks: List[Chunk] = []
        self.doc_freq: Dict[str, int] = {}
        self.avg_doc_len: float = 0.0

    def index_documents_from_dir(
        self,
        docs_dir: Path,
        chunk_chars: int = 1200,
        overlap_chars: int = 150,
        allowed_extensions: Optional[Set[str]] = None,
    ) -> int:
        """Load and index all document chunks from a directory."""
        if allowed_extensions is None:
            allowed_extensions = {".md", ".txt", ".json", ".py", ".sql"}

        if not docs_dir.exists() or not docs_dir.is_dir():
            return 0

        self.chunks = []
        files = sorted([p for p in docs_dir.rglob("*") if p.is_file() and p.suffix.lower() in allowed_extensions])

        for file in files:
            try:
                content = file.read_text(encoding="utf-8", errors="replace")
            except Exception:
                continue

            rel_path = file.name
            try:
                rel_path = str(file.relative_to(docs_dir.parent)).replace("\\", "/")
            except ValueError:
                pass

            headings = split_by_markdown_headings(content)
            for title, section_body in headings:
                pieces = chunk_text_by_chars(section_body, chunk_chars, overlap_chars)
                for piece in pieces:
                    cid = f"chk_{len(self.chunks)+1:04d}"
                    full_text = f"[{rel_path} > {title}]\n{piece}"
                    tokens = tokenize(f"{rel_path} {title} {piece}")
                    self.chunks.append(Chunk(cid, rel_path, title, full_text, tokens))

        self._build_index_statistics()
        return len(self.chunks)

    def _build_index_statistics(self) -> None:
        """Compute document frequencies and average document length."""
        self.doc_freq = {}
        total_len = 0
        n = len(self.chunks)

        for ch in self.chunks:
            total_len += len(ch.tokens)
            for token in set(ch.tokens):
                self.doc_freq[token] = self.doc_freq.get(token, 0) + 1

        self.avg_doc_len = total_len / max(1, n)

    def search(self, query: str, top_k: int = 5) -> List[Tuple[float, Chunk]]:
        """Score all chunks using BM25 and return top results."""
        q_tokens = tokenize(query)
        if not q_tokens or not self.chunks:
            return []

        n = len(self.chunks)
        results: List[Tuple[float, Chunk]] = []

        for ch in self.chunks:
            tf_map: Dict[str, int] = {}
            for t in ch.tokens:
                tf_map[t] = tf_map.get(t, 0) + 1

            dl = max(1, len(ch.tokens))
            score = 0.0
            title_norm = normalize_text(f"{ch.title} {ch.source}")

            for t in q_tokens:
                tf = tf_map.get(t, 0)
                if tf > 0:
                    df = self.doc_freq.get(t, 0)
                    idf = math.log(1.0 + (n - df + 0.5) / (df + 0.5))
                    bm25_term = idf * (tf * (self.k1 + 1.0)) / (tf + self.k1 * (1.0 - self.b + self.b * (dl / self.avg_doc_len)))
                    score += bm25_term

                if t in title_norm:
                    score += 1.5  # Title match bonus

            if score > 0:
                results.append((score, ch))

        results.sort(key=lambda x: x[0], reverse=True)
        return results[:top_k]

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        always_include_prefix: str = "00_",
        max_chunks_per_source: int = 2,
    ) -> List[Chunk]:
        """
        Smart retrieval:
        1. Always includes core rule chunks matching `always_include_prefix`.
        2. Fills remaining quota with top BM25 results with diversity limits.
        """
        selected: List[Chunk] = []
        selected_ids: Set[str] = set()
        source_counts: Dict[str, int] = {}

        # 1. Unconditionally include core system rules if present
        if always_include_prefix:
            for ch in self.chunks:
                file_name = Path(ch.source).name
                if file_name.startswith(always_include_prefix):
                    if ch.id not in selected_ids:
                        selected.append(ch)
                        selected_ids.add(ch.id)
                        source_counts[ch.source] = source_counts.get(ch.source, 0) + 1
                        if len(selected) >= 2:  # Cap initial core rules
                            break

        # 2. BM25 Search
        search_results = self.search(query, top_k=top_k * 3)
        for _, ch in search_results:
            if ch.id in selected_ids:
                continue
            count = source_counts.get(ch.source, 0)
            if count >= max_chunks_per_source:
                continue

            selected.append(ch)
            selected_ids.add(ch.id)
            source_counts[ch.source] = count + 1

            if len(selected) >= top_k:
                break

        return selected


def format_rag_context(chunks: List[Chunk]) -> str:
    """Format retrieved chunks into clean markdown for prompt injection."""
    if not chunks:
        return "(No additional RAG context found.)"

    blocks = []
    for i, ch in enumerate(chunks, 1):
        blocks.append(f"--- [RAG CONTEXT {i} | Source: {ch.source} | Section: {ch.title}] ---\n{ch.text}")
    return "\n\n".join(blocks)
