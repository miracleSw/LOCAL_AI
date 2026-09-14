#!/usr/bin/env bash
# ================================================================
# LOCAL_AI_CORE - Interactive AGY CLI Menu (Linux Mint / Ubuntu)
# ================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON_CMD="python3"

# Colors
CYAN="\033[1;36m"
GREEN="\033[1;32m"
YELLOW="\033[1;33m"
BLUE="\033[1;34m"
RED="\033[1;31m"
BOLD="\033[1m"
RESET="\033[0m"

pause_prompt() {
    echo ""
    read -r -p "Nhan [Enter] de tiep tuc..."
}

open_file_or_dir() {
    local target="$1"
    if command -v xdg-open >/dev/null 2>&1; then
        xdg-open "$target" >/dev/null 2>&1 &
    elif command -v gio >/dev/null 2>&1; then
        gio open "$target" >/dev/null 2>&1 &
    else
        echo "[*] Vui long mo: $target"
    fi
}

edit_task() {
    local task_file="input/current_task.txt"
    if [ -n "$EDITOR" ]; then
        "$EDITOR" "$task_file"
    elif command -v nano >/dev/null 2>&1; then
        nano "$task_file"
    elif command -v xed >/dev/null 2>&1; then
        xed "$task_file" >/dev/null 2>&1 &
    elif command -v gedit >/dev/null 2>&1; then
        gedit "$task_file" >/dev/null 2>&1 &
    else
        open_file_or_dir "$task_file"
    fi
}

start_server() {
    if pgrep -f "llama-server" >/dev/null 2>&1; then
        echo -e "${YELLOW}[THONG BAO] Llama Server da dang chay tren Port 8080!${RESET}"
        pause_prompt
        return
    fi

    echo -e "${GREEN}[*] Dang khoi dong Llama Server...${RESET}"
    if command -v gnome-terminal >/dev/null 2>&1; then
        gnome-terminal --title="LOCAL_AI_CORE Server" -- ./run_server.sh &
    elif command -v x-terminal-emulator >/dev/null 2>&1; then
        x-terminal-emulator -T "LOCAL_AI_CORE Server" -e ./run_server.sh &
    elif command -v xterm >/dev/null 2>&1; then
        xterm -title "LOCAL_AI_CORE Server" -e ./run_server.sh &
    else
        mkdir -p output/logs
        nohup ./run_server.sh > output/logs/server.log 2>&1 &
        echo -e "${GREEN}[+] Server da khoi dong che do ngam (PID: $!).${RESET}"
        echo -e "    Xem log thoi gian thuc tai: ${BOLD}output/logs/server.log${RESET}"
    fi
    sleep 1
}

while true; do
    clear
    echo "================================================================"
    echo -e " ${BOLD}LOCAL_AI_CORE - MODULAR OFFLINE AI SOLVER FRAMEWORK${RESET}"
    echo -e " ${CYAN}Model: Qwen 2.5 Coder 3B (Q4_K_M) | Linux Mint Engine${RESET}"
    echo "================================================================"
    echo ""
    echo -e "${BOLD}[SERVER / INFERENCE ENGINE]${RESET}"
    echo -e "  [1]  Khoi dong Server nhanh (Persistent Llama Server - Port 8080)"
    echo -e "  [2]  Dung Server (Stop Llama Server)"
    echo ""
    echo -e "${BOLD}[CHAT TRUC TIEP VOI MODEL]${RESET}"
    echo -e "  [14] Mo Web Chat tren Trinh duyet (${BLUE}http://127.0.0.1:8080${RESET})"
    echo -e "  [15] ${GREEN}Chat truc tiep trong Terminal (AGY CLI Interactive Chat)${RESET}"
    echo ""
    echo -e "${BOLD}[QUAN LY DE BAI / INPUT]${RESET}"
    echo -e "  [3]  Soan / Chinh sua de bai hien tai (input/current_task.txt)"
    echo -e "  [4]  Xem trang thai hien tai (Pipeline Status)"
    echo -e "  [5]  Don dep ket qua cu (Atomic Reset)"
    echo ""
    echo -e "${BOLD}[CHAY PIPELINE TU DONG HOA]${RESET}"
    echo -e "  [6]  ${YELLOW}Chay toan bo quy trinh (RUN ALL - End to End Orchestrator)${RESET}"
    echo ""
    echo -e "${BOLD}[CHAY TUNG BUOC (STEP-BY-STEP)]${RESET}"
    echo -e "  [7]  STEP 01 - Phan tich yeu cau & Rang buoc (Analyze & Constraints)"
    echo -e "  [8]  STEP 02 - Tao giai phap & Ma nguon (Core Solution / Code)"
    echo -e "  [9]  STEP 03 - Kiem thu, Ra soat & Dong goi (Verify & Assembly)"
    echo ""
    echo -e "${BOLD}[TIEN ICH / XUAT BAN]${RESET}"
    echo -e "  [10] Xuat code ra thu muc du an (Export to output/final_project)"
    echo -e "  [11] Kiem tra toan dien he thong (Diagnostic Check --check)"
    echo -e "  [12] Mo thu muc ket qua tung buoc (output/steps)"
    echo -e "  [13] Mo thu muc san pham hoan chinh (output/final_project)"
    echo -e "  [0]  Thoat (Exit)"
    echo ""
    echo "================================================================"

    read -r -p "Chon chuc nang [0-15]: " choice

    case "$choice" in
        1)
            start_server
            ;;
        2)
            ./stop_server.sh
            pause_prompt
            ;;
        3)
            edit_task
            ;;
        4)
            $PYTHON_CMD "core/orchestrator.py" --status
            pause_prompt
            ;;
        5)
            $PYTHON_CMD "core/orchestrator.py" --reset
            pause_prompt
            ;;
        6)
            $PYTHON_CMD "core/orchestrator.py" --run-all
            pause_prompt
            ;;
        7)
            $PYTHON_CMD "core/orchestrator.py" --step analyze
            pause_prompt
            ;;
        8)
            $PYTHON_CMD "core/orchestrator.py" --step solve
            pause_prompt
            ;;
        9)
            $PYTHON_CMD "core/orchestrator.py" --step verify
            pause_prompt
            ;;
        10)
            $PYTHON_CMD "core/orchestrator.py" --export
            pause_prompt
            ;;
        11)
            $PYTHON_CMD "core/orchestrator.py" --check
            pause_prompt
            ;;
        12)
            open_file_or_dir "output/steps"
            ;;
        13)
            open_file_or_dir "output/final_project"
            ;;
        14)
            if ! pgrep -f "llama-server" >/dev/null 2>&1; then
                echo -e "${YELLOW}[CHU Y] Server chua bat. Dang tu dong khoi dong Server...${RESET}"
                start_server
                sleep 2
            fi
            open_file_or_dir "http://127.0.0.1:8080"
            ;;
        15)
            $PYTHON_CMD "core/chat.py"
            ;;
        0)
            echo -e "${GREEN}Tam biet!${RESET}"
            exit 0
            ;;
        *)
            echo -e "${RED}[LOI] Lua chon khong hop le. Vui long nhap tu 0 den 15.${RESET}"
            pause_prompt
            ;;
    esac
done
