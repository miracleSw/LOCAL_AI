#!/usr/bin/env bash
# ================================================================
# LOCAL_AI_CORE - Automated Setup & Verification (Linux Mint)
# ================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "================================================================"
echo " LOCAL_AI_CORE - CAI DAT TU DONG CHO LINUX MINT / UBUNTU"
echo "================================================================"
echo ""

# 1. Check Python 3
echo "[*] Step 1/6: Kiem tra moi truong Python 3..."
if ! command -v python3 >/dev/null 2>&1; then
    echo -e "\033[1;31m[ERROR] Khong tim thay Python 3 tren may!\033[0m"
    echo "Vui long cai dat Python 3 bang lenh:"
    echo "  sudo apt update && sudo apt install -y python3 python3-pip python3-venv"
    exit 1
fi

PY_VER=$(python3 -c "import sys; print(sys.version.split()[0])")
echo -e "[+] Python Version: \033[1;32m$PY_VER [OK]\033[0m"
echo ""

# 2. Check & Create Directories
echo "[*] Step 2/6: Kiem tra cau truc thu muc du an..."
mkdir -p input output output/steps output/final_project output/logs models rag/active llama-linux

if [ ! -f "input/current_task.txt" ]; then
    echo "[Khoi tao] Tao file mau: input/current_task.txt"
    cat << 'EOF' > input/current_task.txt
# DE BAI / YEU CAU BAI TOAN [INPUT TASK]
# Dan yeu cau lap trinh hoac bai toan can giai vao day:

Xay dung ham tinh day so Fibonacci de quy co nho bang Python va viet unit test.
EOF
fi
echo -e "[+] Thu muc du an: \033[1;32mSan sang [OK]\033[0m"
echo ""

# 3. Check & Install Python Dependencies
echo "[*] Step 3/6: Kiem tra thu vien Python..."
if [ -f "requirements.txt" ]; then
    python3 -m pip install -q --disable-pip-version-check -r requirements.txt 2>/dev/null || true
    echo -e "[+] Python Dependencies: \033[1;32mSan sang [OK]\033[0m"
fi
echo ""

# 4. Check & Setup Linux Llama Binaries
echo "[*] Step 4/6: Kiem tra bo engine llama.cpp cho Linux..."
if [ -x "llama-linux/llama-server" ] || command -v llama-server >/dev/null 2>&1; then
    echo -e "[+] llama-server: \033[1;32mSan sang [OK]\033[0m"
else
    echo "[!] Chua co llama-server cho Linux. Dang khoi tao bo cai..."
    python3 "core/setup_llama_linux.py" || true
fi
echo ""

# 5. Check & Download Model File
echo "[*] Step 5/6: Kiem tra file Model GGUF..."
MODEL_FILE="models/qwen25-coder-3b-q4km.gguf"
if [ -f "$MODEL_FILE" ]; then
    MODEL_SIZE=$(du -h "$MODEL_FILE" | cut -f1)
    echo -e "[+] Model GGUF: \033[1;32mDa tim thay $MODEL_FILE ($MODEL_SIZE) [OK]\033[0m"
else
    echo "[!] Chua tim thay file model: $MODEL_FILE"
    echo "Ban co muon tu dong tai model Qwen 2.5 Coder 3B (~2.1 GB) ve khong?"
    read -r -p "Tai ngay bay gio? [Y/n]: " dl_choice
    dl_choice=${dl_choice:-Y}
    if [[ "$dl_choice" =~ ^[Yy]$ ]]; then
        python3 "core/download_model.py"
    else
        echo "[CHU Y] Ban can tai model ve truoc khi chay chuong trinh."
    fi
fi
echo ""

# 6. Grant execute permissions to shell scripts
echo "[*] Step 6/6: Cap quyen thuc thi cho cac file script..."
chmod +x *.sh 2>/dev/null || true
echo -e "[+] Quyen thuc thi: \033[1;32mHoan tat [OK]\033[0m"
echo ""

echo "================================================================"
echo " KIEM TRA CHAN DOAN TOAN DIEN HE THONG (--check)"
echo "================================================================"
python3 "core/orchestrator.py" --check || true
echo ""

echo "================================================================"
echo " CAI DAT HOAN TAT!"
echo "================================================================"
echo "De khoi dong giao dien quan ly, hay chay:"
echo "  ./menu.sh"
echo "================================================================"

read -r -p "Ban co muon mo menu ngay bay gio khong? [Y/n]: " launch_menu
launch_menu=${launch_menu:-Y}
if [[ "$launch_menu" =~ ^[Yy]$ ]]; then
    exec ./menu.sh
fi
