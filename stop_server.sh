#!/usr/bin/env bash
# ================================================================
# LOCAL_AI_CORE - Stop Llama Server (Linux Mint / Ubuntu)
# ================================================================
echo "================================================================"
echo " DUNG LOCAL_AI_CORE LLAMA SERVER"
echo "================================================================"

if pgrep -f "llama-server" >/dev/null 2>&1; then
    pkill -f "llama-server"
    echo "[OK] Da dung tien trinh llama-server thanh cong."
else
    echo "[*] Khong co tien trinh llama-server nao dang chay."
fi

exit 0
