#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Automated llama.cpp binary installer for Linux Mint / Ubuntu.
Downloads prebuilt llama-server & llama-cli from official GitHub releases.
"""
import json
import os
import shutil
import stat
import sys
import urllib.request
import zipfile
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
TARGET_DIR = BASE_DIR / "llama-linux"
GITHUB_API_URL = "https://api.github.com/repos/ggerganov/llama.cpp/releases/latest"


def is_installed() -> bool:
    server = TARGET_DIR / "llama-server"
    cli = TARGET_DIR / "llama-cli"
    return server.exists() and os.access(server, os.X_OK) and cli.exists() and os.access(cli, os.X_OK)


def download_prebuilt() -> bool:
    TARGET_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 64)
    print("  LOCAL_AI_CORE - LINUX LLAMA BINARY INSTALLER")
    print(f"  Target directory: {TARGET_DIR}")
    print("=" * 64)

    req = urllib.request.Request(GITHUB_API_URL, headers={"User-Agent": "LOCAL_AI_CORE/1.0"})
    try:
        print("[*] Dang tim kiem ban phat hanh llama.cpp moi nhat tren GitHub...")
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except Exception as e:
        print(f"[!] Khong the ket noi den GitHub API: {e}")
        data = None

    download_url = None
    asset_name = None

    if data and "assets" in data:
        for asset in data["assets"]:
            name = asset.get("name", "")
            # Match ubuntu-x64 or linux-x64 zip
            if ("ubuntu-x64" in name or "bin-ubuntu" in name) and name.endswith(".zip"):
                download_url = asset.get("browser_download_url")
                asset_name = name
                break

    if not download_url:
        print("[!] Khong tim thay ban prebuilt tu dong.")
        return False

    print(f"[+] Tim thay goi nhi phan: {asset_name}")
    print(f"[*] Dang tai ve: {download_url}...")

    zip_path = TARGET_DIR / asset_name
    try:
        urllib.request.urlretrieve(download_url, zip_path)
        print("[+] Tai thanh cong! Dang giai nen...")

        with zipfile.ZipFile(zip_path, "r") as zf:
            for item in zf.namelist():
                base_item = Path(item).name
                if base_item in ("llama-server", "llama-cli", "libggml.so", "libllama.so") or base_item.startswith("lib"):
                    extracted_path = TARGET_DIR / base_item
                    with zf.open(item) as src, open(extracted_path, "wb") as dst:
                        shutil.copyfileobj(src, dst)
                    # Chmod +x
                    extracted_path.chmod(extracted_path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)

        if zip_path.exists():
            zip_path.unlink()

        print(f"[OK] Da cai dat xong llama-server va llama-cli vao {TARGET_DIR}!")
        return is_installed()
    except Exception as e:
        print(f"[!] Loi khi tai/giai nen: {e}")
        if zip_path.exists():
            zip_path.unlink()
        return False


def main() -> int:
    if is_installed():
        print(f"[OK] llama-server da co san tai: {TARGET_DIR / 'llama-server'}")
        return 0

    print("[*] llama-server chua co trong llama-linux/.")
    success = download_prebuilt()
    if success:
        return 0

    print("\n" + "=" * 64)
    print(" HUONG DAN BIEN DICH VULKAN CHO LINUX MINT / UBUNTU")
    print("=" * 64)
    print("De bien dich ban ho tro Vulkan toi uu cho GPU AMD / Intel tren Linux Mint:")
    print("1. Cai dat goi phat trien:")
    print("   sudo apt update && sudo apt install -y build-essential cmake libvulkan-dev vulkan-tools git")
    print("2. Clone va bien dich:")
    print("   git clone https://github.com/ggerganov/llama.cpp.git /tmp/llama.cpp")
    print("   cd /tmp/llama.cpp")
    print("   cmake -B build -DGGML_VULKAN=ON")
    print("   cmake --build build --config Release -j$(nproc)")
    print(f"3. Sao chep binary vao thu muc du an:")
    print(f"   cp build/bin/llama-server build/bin/llama-cli \"{TARGET_DIR}/\"")
    print("=" * 64 + "\n")
    return 1


if __name__ == "__main__":
    sys.exit(main())
