@echo off
setlocal EnableExtensions
title LOCAL_AI_CORE - Llama Server (Port 8080)
pushd "%~dp0"

set MODEL=models\qwen25-coder-3b-q4km.gguf
set SERVER_BIN=llama-vulkan\llama-server.exe
set PORT=8080
set HOST=127.0.0.1

echo ================================================================
echo  LOCAL_AI_CORE - PERSISTENT VULKAN LLAMA SERVER
echo ================================================================
echo.
echo Model:     %MODEL%
echo Server:    %SERVER_BIN%
echo Endpoint:  http://%HOST%:%PORT%/completion
echo Health:    http://%HOST%:%PORT%/health
echo.
echo Settings:
echo   - Context size:  6144 tokens
echo   - CPU Threads:   8 (Optimized for 8 P-cores)
echo   - GPU Layers:    12 (Vulkan acceleration on 2GB VRAM)
echo   - Limits:        1 Slot (-np 1), 512MB Prompt Cache (-cram 512)
echo.
echo NOTE: Keep this window running in the background while working.
echo       The client connects via fast HTTP with zero model reload latency.
echo       To stop the server, run stop_server.bat or press Ctrl+C here.
echo ================================================================
echo.

if not exist "%MODEL%" (
    echo [ERROR] Model file not found: %MODEL%
    pause
    exit /b 1
)

if not exist "%SERVER_BIN%" (
    echo [ERROR] llama-server binary not found: %SERVER_BIN%
    pause
    exit /b 1
)

"%SERVER_BIN%" ^
  -m "%MODEL%" ^
  -c 6144 ^
  -t 8 ^
  -ngl 12 ^
  -np 1 ^
  -cram 512 ^
  --no-mmap ^
  --host %HOST% ^
  --port %PORT%

popd
pause
