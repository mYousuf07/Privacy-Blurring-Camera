@echo off
REM Start Privacy Camera application from workspace root
REM This will use the virtual environment in the workspace if available.
cd /d "%~dp0"

REM If a local virtual environment exists, use its python executable (relative path)
if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" "%~dp0main.py"
) else (
    REM Fallback to system python if no venv present
    python "%~dp0main.py"
)

pause
