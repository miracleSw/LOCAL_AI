@echo off
setlocal EnableExtensions
pushd "%~dp0"
set PYTHON_CMD=python -X utf8

:MENU
cls
echo ================================================================
echo  LOCAL_AI_CORE - MODULAR OFFLINE AI SOLVER FRAMEWORK
echo  Model: Qwen 2.5 Coder 3B (Q4_K_M) ^| Vulkan Offline Inference
echo ================================================================
echo.
echo [SERVER / INFERENCE ENGINE]
echo   [1]  Khoi dong Server nhanh (Persistent Llama Server - Port 8080)
echo   [2]  Dung Server (Stop Llama Server)
echo.
echo [CHAT TRUC TIEP VOI MODEL]
echo   [14] Mo Web Chat tren Trinh duyet (http://127.0.0.1:8080)
echo   [15] Chat truc tiep trong Terminal (CLI Interactive Chat)
echo.
echo [QUAN LY DE BAI / INPUT]
echo   [3]  Soan / Chinh sua de bai hien tai (input\current_task.txt)
echo   [4]  Xem trang thai hien tai (Pipeline Status)
echo   [5]  Don dep ket qua cu (Atomic Reset)
echo.
echo [CHAY PIPELINE TU DONG HOA]
echo   [6]  Chay toan bo quy trinh (RUN ALL - End to End Orchestrator)
echo.
echo [CHAY TUNG BUOC (STEP-BY-STEP)]
echo   [7]  STEP 01 - Phan tich yeu cau ^& Rang buoc (Analyze ^& Constraints)
echo   [8]  STEP 02 - Tao giai phap ^& Ma nguon (Core Solution / Code)
echo   [9]  STEP 03 - Kiem thu, Ra soat ^& Dong goi (Verify ^& Assembly)
echo.
echo [TIEN ICH / XUAT BAN]
echo   [10] Xuat code ra thu muc du an (Export to output\final_project)
echo   [11] Kiem tra toan dien he thong (Diagnostic Check --check)
echo   [12] Mo thu muc ket qua tung buoc (output\steps)
echo   [13] Mo thu muc san pham hoan chinh (output\final_project)
echo   [0]  Thoat (Exit)
echo.
echo ================================================================
set "choice="
set /p choice=Chon chuc nang [0-15]: 
if not defined choice goto MENU

if "%choice%"=="1" (
    tasklist /fi "imagename eq llama-server.exe" 2>nul | find /i "llama-server.exe" >nul
    if errorlevel 1 (
        start "LOCAL_AI_CORE Server" run_server.bat
    ) else (
        echo [THONG BAO] Llama Server da dang chay tren Port 8080!
        pause
    )
    goto MENU
)
if "%choice%"=="2" call stop_server.bat & pause & goto MENU
if "%choice%"=="3" start "" notepad "input\current_task.txt" & goto MENU
if "%choice%"=="4" %PYTHON_CMD% "core\orchestrator.py" --status & pause & goto MENU
if "%choice%"=="5" %PYTHON_CMD% "core\orchestrator.py" --reset & pause & goto MENU

if "%choice%"=="6" %PYTHON_CMD% "core\orchestrator.py" --run-all & pause & goto MENU

if "%choice%"=="7" %PYTHON_CMD% "core\orchestrator.py" --step analyze & pause & goto MENU
if "%choice%"=="8" %PYTHON_CMD% "core\orchestrator.py" --step solve & pause & goto MENU
if "%choice%"=="9" %PYTHON_CMD% "core\orchestrator.py" --step verify & pause & goto MENU

if "%choice%"=="10" %PYTHON_CMD% "core\orchestrator.py" --export & pause & goto MENU
if "%choice%"=="11" %PYTHON_CMD% "core\orchestrator.py" --check & pause & goto MENU
if "%choice%"=="12" start "" "output\steps" & goto MENU
if "%choice%"=="13" start "" "output\final_project" & goto MENU

if "%choice%"=="14" (
    tasklist /fi "imagename eq llama-server.exe" 2>nul | find /i "llama-server.exe" >nul
    if errorlevel 1 (
        echo [CHU Y] Server chua bat. Dang tu dong khoi dong Server...
        start "LOCAL_AI_CORE Server" run_server.bat
        ping 127.0.0.1 -n 3 >nul
    )
    start http://127.0.0.1:8080
    goto MENU
)
if "%choice%"=="15" %PYTHON_CMD% "core\chat.py" & goto MENU

if "%choice%"=="0" popd & exit /b 0

echo [LOI] Lua chon khong hop le. Vui long nhap tu 0 den 15.
pause
goto MENU
