@echo off
setlocal EnableExtensions
REM ===========================================================================
REM  ImageFabric launcher  (double-click me)
REM
REM    run.bat          -> REAL mode: loads actual models on the GPU
REM    run.bat stub     -> STUB mode: full pipeline, no GPU/ML stack
REM
REM  Opens the backend (:8260) and the Vite frontend (:5173) each in its own
REM  window, then points your browser at the app. First run bootstraps the venv
REM  and npm deps.
REM ===========================================================================

cd /d "%~dp0"

set "VENV_PY=%~dp0.venv\Scripts\python.exe"

REM --- mode ----------------------------------------------------------------
if /I "%~1"=="stub" (
    set "IMGFAB_STUB_MODE=true"
    echo [mode] STUB  - architectural pipeline only, no GPU/ML stack
) else (
    set "IMGFAB_STUB_MODE=false"
    echo [mode] REAL  - real models on the GPU ^(run "run.bat stub" for no-GPU mode^)
)

REM --- bootstrap backend venv ---------------------------------------------
if not exist "%VENV_PY%" (
    echo [setup] creating venv + installing foundation deps...
    python -m venv .venv
    "%VENV_PY%" -m pip install --upgrade pip
    "%VENV_PY%" -m pip install -r backend\requirements.txt
    echo.
    echo [setup] NOTE: REAL mode also needs the Blackwell GPU stack:
    echo         "%VENV_PY%" -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
    echo         "%VENV_PY%" -m pip install -r backend\requirements-gpu.txt
    echo.
)

REM --- bootstrap frontend deps -------------------------------------------
if not exist "%~dp0frontend\node_modules" (
    echo [setup] installing frontend deps...
    pushd "%~dp0frontend"
    call npm install
    popd
)

echo [run] backend  -^> http://127.0.0.1:8260
echo [run] frontend -^> http://localhost:5173

REM --- launch (each in its own window; env is inherited) ------------------
start "ImageFabric backend" /D "%~dp0backend" cmd /k ""%VENV_PY%" -m uvicorn app.main:app --host 127.0.0.1 --port 8260 --reload"
start "ImageFabric frontend" /D "%~dp0frontend" cmd /k "npm run dev"

REM give the dev servers a moment, then open the UI
timeout /t 6 /nobreak >nul
start "" http://localhost:5173

echo.
echo Both started in separate windows. UI: http://localhost:5173
echo Close those windows (or Ctrl+C in them) to stop.
endlocal
