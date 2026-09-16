#!/usr/bin/env bash
# ================================================================
# LOCAL_AI_CORE - Persistent Llama Server (Linux Mint / Ubuntu)
# ================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

MODEL="models/qwen25-coder-3b-q4km.gguf"
PORT=8080
HOST="127.0.0.1"

# Resolve llama-server binary
SERVER_BIN=""
if [ -x "$SCRIPT_DIR/llama-linux/llama-server" ]; then
    SERVER_BIN="$SCRIPT_DIR/llama-linux/llama-server"
elif [ -x "$SCRIPT_DIR/llama-vulkan/llama-server" ]; then
    SERVER_BIN="$SCRIPT_DIR/llama-vulkan/llama-server"
elif command -v llama-server >/dev/null 2>&1; then
    SERVER_BIN="$(command -v llama-server)"
fi

if [ -z "$SERVER_BIN" ]; then
    echo -e "\033[1;31m[ERROR] Khong tim thay binary llama-server cho Linux!\033[0m"
    echo "Vui long chay: bash setup.sh de tu dong tai hoac huong dan cai dat llama-server."
    exit 1
fi

if [ ! -f "$MODEL" ]; then
    echo -e "\033[1;31m[ERROR] Khong tim thay file model: $MODEL\033[0m"
    echo "Vui long chay: python3 core/download_model.py de tai model."
    exit 1
fi

# Detect CPU threads (cap at 8 for hybrid P-cores)
THREADS=8
if command -v nproc >/dev/null 2>&1; then
    CPU_CORES=$(nproc)
    if [ "$CPU_CORES" -lt 8 ]; then
        THREADS=$CPU_CORES
    fi
fi

echo "================================================================"
echo " LOCAL_AI_CORE - PERSISTENT LLAMA SERVER (LINUX MINT / UBUNTU)"
echo "================================================================"
echo "Model:     $MODEL"
echo "Server:    $SERVER_BIN"
echo "Endpoint:  http://$HOST:$PORT/completion"
echo "Health:    http://$HOST:$PORT/health"
echo ""
echo "Cau hinh:"
echo "  - Context size:  6144 tokens"
echo "  - CPU Threads:   $THREADS"
echo "  - GPU Layers:    20 (Peak acceleration on 2GB VRAM without PCIe spill)"
echo "  - Flash Attn:    off (2.3x faster prompt processing on Vulkan shader)"
echo "  - Limits:        1 Slot (-np 1), 512MB Prompt Cache"
echo "================================================================"
echo "Nhan Ctrl+C de dung server hoac chay ./stop_server.sh tu terminal khac."
echo "================================================================"

exec "$SERVER_BIN" \
  -m "$MODEL" \
  -c 6144 \
  -t "$THREADS" \
  -ngl 20 \
  -fa off \
  -ub 512 \
  -b 2048 \
  -np 1 \
  -cram 512 \
  --no-mmap \
  --host "$HOST" \
  --port "$PORT"
