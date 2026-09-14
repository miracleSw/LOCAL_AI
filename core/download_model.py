#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Automated Model Downloader for LOCAL_AI_CORE.
Downloads Qwen 2.5 Coder 3B Q4_K_M GGUF from Hugging Face with progress display.
"""
import os
import sys
import time
import urllib.request
from pathlib import Path

MODEL_URL = "https://huggingface.co/Qwen/Qwen2.5-Coder-3B-Instruct-GGUF/resolve/main/qwen2.5-coder-3b-instruct-q4_k_m.gguf"
MODEL_FILENAME = "qwen25-coder-3b-q4km.gguf"


def download_model(target_dir: Path) -> bool:
    target_dir.mkdir(parents=True, exist_ok=True)
    target_path = target_dir / MODEL_FILENAME
    temp_path = target_dir / (MODEL_FILENAME + ".part")

    if target_path.exists() and target_path.stat().st_size > 1_500_000_000:
        size_gb = target_path.stat().st_size / (1024**3)
        print(f"[OK] Model file already exists: {target_path.name} ({size_gb:.2f} GB)")
        return True

    print("=" * 64)
    print("  LOCAL_AI_CORE - AUTOMATED MODEL DOWNLOADER")
    print(f"  Target: {target_path}")
    print(f"  Source: {MODEL_URL}")
    print("=" * 64)

    initial_size = 0
    headers = {"User-Agent": "LOCAL_AI_CORE/1.0"}
    if temp_path.exists():
        initial_size = temp_path.stat().st_size
        headers["Range"] = f"bytes={initial_size}-"
        print(f"[*] Resuming download from {initial_size / (1024**2):.1f} MB...")

    req = urllib.request.Request(MODEL_URL, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=30) as response:
            total_size = response.headers.get("Content-Length")
            if total_size is not None:
                total_size = int(total_size) + initial_size
            else:
                total_size = 2_104_932_800  # Approx ~2.1 GB

            mode = "ab" if initial_size > 0 else "wb"
            downloaded = initial_size
            start_time = time.time()
            chunk_size = 1024 * 1024  # 1 MB chunks

            with open(temp_path, mode) as f:
                while True:
                    chunk = response.read(chunk_size)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)

                    elapsed = max(0.1, time.time() - start_time)
                    speed_mb = ((downloaded - initial_size) / (1024**2)) / elapsed
                    percent = (downloaded / total_size * 100) if total_size else 0.0

                    bar_len = 30
                    filled = int(bar_len * percent / 100)
                    bar = "=" * filled + "-" * (bar_len - filled)

                    status = (
                        f"\r[{bar}] {percent:5.1f}% | "
                        f"{downloaded / (1024**2):.1f}/{total_size / (1024**2):.1f} MB | "
                        f"{speed_mb:5.2f} MB/s"
                    )
                    sys.stdout.write(status)
                    sys.stdout.flush()

        print("\n\n[+] Download completed! Finalizing file...")
        if target_path.exists():
            target_path.unlink()
        temp_path.rename(target_path)
        print(f"[SUCCESS] Model saved to: {target_path}")
        return True

    except Exception as exc:
        print(f"\n[ERROR] Download failed: {exc}")
        print("Tip: You can manually download the file from:")
        print(f"     {MODEL_URL}")
        print(f"     and place it at: {target_path}")
        return False


if __name__ == "__main__":
    repo_root = Path(__file__).resolve().parent.parent
    models_dir = repo_root / "models"
    success = download_model(models_dir)
    sys.exit(0 if success else 1)
