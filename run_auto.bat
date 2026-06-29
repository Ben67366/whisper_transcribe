@echo off
cd /d "%~dp0"

echo ================================================
echo   Whisper Transcribe - Auto Batch
echo ================================================
echo.

if exist ".venv\Scripts\python.exe" (
    set "PYTHON_CMD=.venv\Scripts\python.exe"
    echo [INFO] Using virtual environment: .venv
) else (
    set "PYTHON_CMD=python"
    echo [INFO] No .venv found - using system Python
)

echo.
%PYTHON_CMD% whisper_transcribe.py --language auto
set "EXITCODE=%ERRORLEVEL%"

echo.
echo ================================================
if "%EXITCODE%"=="0" (
    echo   [SUCCESS] All tasks completed.
) else (
    echo   [ERROR] Completed with errors ^(exit code: %EXITCODE%^).
)
echo ================================================
echo.
pause
cmd /k
