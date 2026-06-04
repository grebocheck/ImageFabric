@echo off
REM ===========================================================================
REM  HFabric launcher  (double-click me)
REM
REM    run.bat          -> REAL mode: real models on the GPU (default)
REM    run.bat stub     -> STUB mode: full pipeline, no GPU/ML stack
REM
REM  Runs the backend (:8260) and the Vite frontend (:5173) together in THIS
REM  single window, frees any stale ports first, and opens the UI in your
REM  browser. Ctrl+C stops both. First run bootstraps the venv + npm deps.
REM ===========================================================================
setlocal
set "STUBFLAG="
if /I "%~1"=="stub" set "STUBFLAG=-Stub"

powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\run.ps1" %STUBFLAG%

echo.
echo [exit] servers stopped.
pause
endlocal
