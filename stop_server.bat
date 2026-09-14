@echo off
setlocal EnableExtensions
title LOCAL_AI_CORE - Stop Server

echo ================================================================
echo  STOPPING LOCAL_AI_CORE LLAMA SERVER
echo ================================================================
echo.

taskkill /F /IM llama-server.exe >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo [OK] llama-server.exe process was terminated successfully.
) else (
    echo [*] No running llama-server.exe process was found.
)

echo.
ping 127.0.0.1 -n 2 >nul
exit /b 0
