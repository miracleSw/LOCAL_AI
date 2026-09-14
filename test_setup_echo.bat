@echo on
setlocal EnableExtensions
pushd "%~dp0"
title LOCAL_AI_CORE - Automated Setup & Environment Checker

echo ================================================================
echo  LOCAL_AI_CORE - ONE-CLICK AUTOMATED SETUP & VERIFICATION
echo ================================================================
echo.

:: 1. Check Python
echo [*] Step 1/6: Kiem tra moi truong Python...
where python >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    where py >nul 2>&1
    if %ERRORLEVEL% NEQ 0 (
        echo.
        echo [ERROR] Khong tim thay Python tren may tinh!
        echo Vui long cai dat Python 3.10 tro len tu: https://www.python.org/downloads/
        echo LUU Y: Khi cai dat nho tick chon "Add Python to PATH"!
        echo.
        pause
        exit /b 1
    )
    set "PYTHON_CMD=py -X utf8"
) else (
    set "PYTHON_CMD=python -X utf8"
)

for /f "tokens=*" %%i in ('python -c "import sys; print(sys.version.split()[0])"') do set PY_VER=%%i
echo [+] Python Version: %PY_VER% (OK)
echo.

:: 2. Check & Create Required Directories
echo [*] Step 2/6: Kiem tra cau truc thu muc du an...
if not exist "input" mkdir "input"
if not exist "output" mkdir "output"
if not exist "output\steps" mkdir "output\steps"
if not exist "output\final_project" mkdir "output\final_project"
if not exist "output\logs" mkdir "output\logs"
if not exist "models" mkdir "models"
if not exist "rag\active" mkdir "rag\active"

if not exist "input\current_task.txt" (
    echo [Khoi tao] Tao file mau: input\current_task.txt
    (
        echo # DE BAI / YEU CAU BAI TOAN (INPUT TASK)
        echo # Dan yeu cau lap trinh hoac bai toan can giai vao day:
        echo.
        echo Xay dung ham tinh day so Fibonacci de quy co nho bang Python va viet unit test.
    ) > "input\current_task.txt"
)
echo [+] Thu muc du an: San sang (OK)
echo.

:: 3. Check & Install Python Dependencies
echo [*] Step 3/6: Cai dat thu vien can thiet (requirements.txt)...
if exist "requirements.txt" (
    python -m pip install -q -r requirements.txt >nul 2>&1
    if %ERRORLEVEL% EQU 0 (
        echo [+] Python Dependencies: Da cai dat day du (OK)
    ) else (
        echo [*] Pip notice: Python standard library se duoc su dung mac dinh.
    )
)
echo.

:: 4. Check Llama Vulkan Binaries
echo [*] Step 4/6: Kiem tra bo engine suy luan llama.cpp Vulkan...
if not exist "llama-vulkan\llama-server.exe" (
    echo [CANH BAO] Khong tim thay llama-vulkan\llama-server.exe!
    echo Che do Server se khong the khoi dong neu thieu file nay.
) else (
    echo [+] llama-server.exe: San sang (OK)
)

if not exist "llama-vulkan\llama-cli.exe" (
    echo [CANH BAO] Khong tim thay llama-vulkan\llama-cli.exe!
) else (
    echo [+] llama-cli.exe: San sang (OK)
)
echo.

:: 5. Check & Download Model File
echo [*] Step 5/6: Kiem tra file Model GGUF...
set "MODEL_FILE=models\qwen25-coder-3b-q4km.gguf"
if exist "%MODEL_FILE%" (
    echo [+] Model GGUF: Da tim thay %MODEL_FILE% (OK)
) else (
    echo [!] Chua tim thay file model: %MODEL_FILE%
    echo.
    echo Ban co muon he thong TU DONG TAI model Qwen 2.5 Coder 3B (~2.1 GB)
    echo truc tiep tu HuggingFace ve khong?
    set "dl_choice="
    set /p dl_choice=Ban co muon tai ngay khong? [Y/N, mac dinh Y]: 
    if not defined dl_choice set dl_choice=Y
    if /i "%dl_choice%"=="Y" (
        echo.
        python core\download_model.py
        if %ERRORLEVEL% NEQ 0 (
            echo.
            echo [CANH BAO] Khong the tai tu dong. Ban co the copy file model .gguf
            echo vao thu muc models\ voi ten: qwen25-coder-3b-q4km.gguf
        )
    ) else (
        echo [*] Bo qua buoc tai model. Hay copy file .gguf vao thu muc models\ truoc khi chay.
    )
)
echo.

:: 6. Run Comprehensive Diagnostic Check
echo [*] Step 6/6: Chay kiem tra chan doan toan dien he thong...
echo.
%PYTHON_CMD% "core\orchestrator.py" --check
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [CANH BAO] Kiem tra chan doan phat hien mot so thanh phan chua san sang.
    echo Vui long xem thong bao phia tren.
    pause
    popd
    exit /b 1
)

echo.
echo ================================================================
echo  [SUCCESS] HE THONG DA CAI DAT VA SAN SANG 100%!
echo ================================================================
echo.
set "launch_choice="
set /p launch_choice=Ban co muon mo 00_MENU.bat ngay bay gio khong? [Y/N, mac dinh Y]: 
if not defined launch_choice set launch_choice=Y
if /i "%launch_choice%"=="Y" (
    start "" "00_MENU.bat"
)

popd
exit /b 0
