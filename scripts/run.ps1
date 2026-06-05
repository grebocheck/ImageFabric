<#
  HFabric launcher — ONE window, both servers.

    .\scripts\run.ps1          # REAL mode (real models on the GPU)
    .\scripts\run.ps1 -Stub    # STUB mode (full pipeline, no GPU/ML stack)

  Before starting it frees the backend/frontend ports, killing stale instances
  left over from earlier runs. Those leftovers are the cause of the
  "WinError 10013 / socket forbidden" failure: a previous backend was still
  holding port 8260, so a new one could not bind. Bootstraps venv + npm on the
  first run, then runs the FastAPI backend and the Vite dev server in THIS
  console. Ctrl+C stops both.
#>
param(
    [switch]$Stub,
    [int]$Port = 0,
    [int]$FrontendPort = 0,
    [string]$BindHost = ""
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

$venvPy = Join-Path $root ".venv\Scripts\python.exe"

function Import-DotEnv([string]$Path) {
    if (-not (Test-Path $Path)) { return }
    foreach ($line in Get-Content $Path) {
        $trimmed = $line.Trim()
        if ([string]::IsNullOrWhiteSpace($trimmed) -or $trimmed.StartsWith("#")) { continue }

        $eq = $trimmed.IndexOf("=")
        if ($eq -lt 1) { continue }

        $key = $trimmed.Substring(0, $eq).Trim()
        $value = $trimmed.Substring($eq + 1).Trim()
        if (
            ($value.StartsWith('"') -and $value.EndsWith('"')) -or
            ($value.StartsWith("'") -and $value.EndsWith("'"))
        ) {
            $value = $value.Substring(1, $value.Length - 2)
        }

        if ([Environment]::GetEnvironmentVariable($key, "Process") -eq $null) {
            Set-Item -Path "Env:$key" -Value $value
        }
    }
}

function Get-EnvInt([string]$Name, [int]$Default) {
    $value = [Environment]::GetEnvironmentVariable($Name, "Process")
    if ([string]::IsNullOrWhiteSpace($value)) { return $Default }
    return [int]$value
}

function Test-Truthy([string]$Value) {
    if ([string]::IsNullOrWhiteSpace($Value)) { return $false }
    return @("1", "true", "yes", "on").Contains($Value.ToLowerInvariant())
}

Import-DotEnv (Join-Path $root ".env")

if ($Port -le 0) { $Port = Get-EnvInt "HFAB_PORT" 8260 }
if ($FrontendPort -le 0) { $FrontendPort = Get-EnvInt "HFAB_FRONTEND_PORT" 5173 }
if ([string]::IsNullOrWhiteSpace($BindHost)) {
    if ([string]::IsNullOrWhiteSpace($env:HFAB_HOST)) {
        $BindHost = "127.0.0.1"
    } else {
        $BindHost = $env:HFAB_HOST
    }
}

$env:HFAB_HOST = $BindHost
$env:HFAB_PORT = "$Port"

if ($Stub) {
    $env:HFAB_STUB_MODE = "true"
    Write-Host "[mode] STUB - architectural pipeline only, no GPU/ML stack" -ForegroundColor DarkYellow
} elseif (Test-Truthy $env:HFAB_STUB_MODE) {
    $env:HFAB_STUB_MODE = "true"
    Write-Host "[mode] STUB - architectural pipeline only, no GPU/ML stack" -ForegroundColor DarkYellow
} else {
    $env:HFAB_STUB_MODE = "false"
    Write-Host "[mode] REAL - real models on the GPU (use -Stub for no-GPU mode)" -ForegroundColor Green
}

# --- free ports held by stale instances of this app ---------------------------
function Stop-Port([int]$p) {
    $owners = Get-NetTCPConnection -LocalPort $p -ErrorAction SilentlyContinue |
        Select-Object -ExpandProperty OwningProcess -Unique
    foreach ($procId in $owners) {
        if ($procId -and $procId -ne 0) {
            try {
                $name = (Get-Process -Id $procId -ErrorAction Stop).ProcessName
                Write-Host "[ports] port $p busy -> stopping $name (pid $procId)" -ForegroundColor DarkGray
                Stop-Process -Id $procId -Force -ErrorAction Stop
            } catch {}
        }
    }
}
Stop-Port $Port
Stop-Port (Get-EnvInt "HFAB_LLAMA_PORT" 8261)          # llama-server (LLM)
Stop-Port (Get-EnvInt "HFAB_LLAMA_EMBED_PORT" 8262)    # llama-server (RAG embeddings)
Stop-Port $FrontendPort

# Safety net: a run closed via the window 'X' (not Ctrl+C) skips the finally
# block below, so its child llama processes can survive — orphaned, holding
# RAM/VRAM and shrinking the "available RAM" the pre-load guard checks. Sweep
# any strays so every launch starts from a clean slate.
foreach ($n in @("llama-server", "llama-tts", "llama-mtmd-cli")) {
    Get-Process -Name $n -ErrorAction SilentlyContinue | ForEach-Object {
        Write-Host "[ports] stray $($_.ProcessName) (pid $($_.Id)) -> stopping" -ForegroundColor DarkGray
        try { Stop-Process -Id $_.Id -Force -ErrorAction Stop } catch {}
    }
}
Start-Sleep -Milliseconds 400

# --- bootstrap backend venv ---------------------------------------------------
if (-not (Test-Path $venvPy)) {
    Write-Host "[setup] creating venv + installing foundation deps..." -ForegroundColor Cyan
    python -m venv .venv
    & $venvPy -m pip install --upgrade pip
    & $venvPy -m pip install -r backend\requirements.txt
    if (-not $Stub) {
        Write-Host "[setup] REAL mode also needs the Blackwell GPU stack:" -ForegroundColor Yellow
        Write-Host "        & '$venvPy' -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128" -ForegroundColor Yellow
        Write-Host "        & '$venvPy' -m pip install -r backend\requirements-gpu.txt" -ForegroundColor Yellow
    }
}

# --- bootstrap frontend deps --------------------------------------------------
if (-not (Test-Path (Join-Path $root "frontend\node_modules"))) {
    Write-Host "[setup] installing frontend deps..." -ForegroundColor Cyan
    Push-Location frontend; npm install; Pop-Location
}

Write-Host "[run] backend  -> http://${BindHost}:$Port"       -ForegroundColor Green
Write-Host "[run] frontend -> http://localhost:$FrontendPort" -ForegroundColor Green
Write-Host "[run] both run in THIS window; press Ctrl+C to stop.`n" -ForegroundColor Yellow

# Backend shares this console (one window). No --reload -> a single PID to manage.
$backend = Start-Process -FilePath $venvPy `
    -ArgumentList @("-m", "uvicorn", "app.main:app", "--host", "$BindHost", "--port", "$Port") `
    -WorkingDirectory (Join-Path $root "backend") `
    -NoNewWindow -PassThru

# Open the UI once the servers have had a moment to come up.
Start-Job -ScriptBlock {
    param($url)
    Start-Sleep -Seconds 6
    Start-Process $url
} -ArgumentList "http://localhost:$FrontendPort" | Out-Null

try {
    Push-Location frontend
    npm run dev          # foreground; blocks until Ctrl+C
} finally {
    Pop-Location
    if ($backend -and -not $backend.HasExited) {
        Write-Host "`n[stop] shutting down backend (pid $($backend.Id))..." -ForegroundColor DarkGray
        taskkill /PID $backend.Id /T /F 2>$null | Out-Null
    }
    Stop-Port $Port
    Get-Job | Remove-Job -Force -ErrorAction SilentlyContinue
}
