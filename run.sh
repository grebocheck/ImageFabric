#!/usr/bin/env bash
# =============================================================================
#  HFabric launcher — Linux / macOS
#
#    ./run.sh          REAL mode: real models on the GPU (default)
#    ./run.sh stub     STUB mode: full pipeline, no GPU/ML stack
#
#  Frees stale ports, bootstraps venv + npm on first run, then runs the FastAPI
#  backend (:8260) and the Vite frontend (:5173) in THIS terminal. Ctrl+C stops
#  both. Mirrors scripts/run.ps1.
# =============================================================================
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

PYBIN="$ROOT/.venv/bin/python"

load_env() {
  local file="$1"
  [ -f "$file" ] || return 0

  local line key value
  while IFS= read -r line || [ -n "$line" ]; do
    line="$(printf '%s' "$line" | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//')"
    case "$line" in ""|\#*) continue ;; esac

    key="${line%%=*}"
    value="${line#*=}"
    key="$(printf '%s' "$key" | tr -d '[:space:]')"
    value="$(printf '%s' "$value" | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//')"
    [[ "$key" =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]] || continue

    if { [ "${value:0:1}" = '"' ] && [ "${value: -1}" = '"' ]; } ||
       { [ "${value:0:1}" = "'" ] && [ "${value: -1}" = "'" ]; }; then
      value="${value:1:${#value}-2}"
    fi

    if [ -z "${!key+x}" ]; then
      export "$key=$value"
    fi
  done < "$file"
}

load_env "$ROOT/.env"

PORT="${HFAB_PORT:-8260}"
FPORT="${HFAB_FRONTEND_PORT:-5173}"
BIND_HOST="${HFAB_HOST:-127.0.0.1}"
LLAMA_PORT="${HFAB_LLAMA_PORT:-8261}"
LLAMA_EMBED_PORT="${HFAB_LLAMA_EMBED_PORT:-8262}"
export HFAB_HOST="$BIND_HOST"
export HFAB_PORT="$PORT"

if [ -t 1 ]; then
  C_GREEN=$'\033[32m'; C_YELLOW=$'\033[33m'; C_CYAN=$'\033[36m'; C_DIM=$'\033[2m'; C_RST=$'\033[0m'
else
  C_GREEN=""; C_YELLOW=""; C_CYAN=""; C_DIM=""; C_RST=""
fi
have() { command -v "$1" >/dev/null 2>&1; }

if [ "${1:-}" = "stub" ]; then
  export HFAB_STUB_MODE="true"
  printf '%s[mode] STUB — pipeline only, no GPU/ML stack%s\n' "$C_YELLOW" "$C_RST"
elif [ "${HFAB_STUB_MODE:-false}" = "true" ] || [ "${HFAB_STUB_MODE:-false}" = "1" ]; then
  export HFAB_STUB_MODE="true"
  printf '%s[mode] STUB - pipeline only, no GPU/ML stack%s\n' "$C_YELLOW" "$C_RST"
else
  export HFAB_STUB_MODE="false"
  printf '%s[mode] REAL — real models on the GPU (use "stub" for no-GPU mode)%s\n' "$C_GREEN" "$C_RST"
fi

# --- free ports held by stale instances --------------------------------------
free_port() {
  local p="$1"
  if have fuser; then
    fuser -k "${p}/tcp" >/dev/null 2>&1 || true
  elif have lsof; then
    local pids; pids="$(lsof -ti "tcp:${p}" 2>/dev/null || true)"
    [ -n "$pids" ] && kill -9 $pids >/dev/null 2>&1 || true
  fi
}
sweep_llama() {
  # A run closed by killing the terminal can orphan child llama processes that
  # keep holding RAM/VRAM; sweep them so every launch starts clean.
  for n in llama-server llama-tts llama-mtmd-cli; do
    pkill -9 -f "$n" >/dev/null 2>&1 || true
  done
}
for p in "$PORT" "$LLAMA_PORT" "$LLAMA_EMBED_PORT" "$FPORT"; do free_port "$p"; done
sweep_llama
sleep 0.4

# --- bootstrap backend venv --------------------------------------------------
if [ ! -x "$PYBIN" ]; then
  printf '%s[setup] creating venv + installing foundation deps...%s\n' "$C_CYAN" "$C_RST"
  PYHOST="python3"; have python3 || PYHOST="python"
  "$PYHOST" -m venv .venv
  "$PYBIN" -m pip install --upgrade pip >/dev/null
  "$PYBIN" -m pip install -r backend/requirements.txt >/dev/null
  if [ "${HFAB_STUB_MODE}" = "false" ]; then
    printf '%s[setup] REAL mode also needs the GPU stack — run ./setup.sh real%s\n' "$C_YELLOW" "$C_RST"
  fi
fi

# --- bootstrap frontend deps -------------------------------------------------
if [ ! -d "$ROOT/frontend/node_modules" ]; then
  printf '%s[setup] installing frontend deps...%s\n' "$C_CYAN" "$C_RST"
  ( cd frontend && npm install )
fi

printf '%s[run] backend  → http://%s:%s%s\n' "$C_GREEN" "$BIND_HOST" "$PORT" "$C_RST"
printf '%s[run] frontend → http://localhost:%s%s\n' "$C_GREEN" "$FPORT" "$C_RST"
printf '%s[run] both run in THIS terminal; press Ctrl+C to stop.%s\n\n' "$C_YELLOW" "$C_RST"

# --- start backend (background) ----------------------------------------------
( cd backend && exec "$PYBIN" -m uvicorn app.main:app --host "$BIND_HOST" --port "$PORT" ) &
BACKPID=$!

cleanup() {
  printf '\n%s[stop] shutting down...%s\n' "$C_DIM" "$C_RST"
  kill "$BACKPID" >/dev/null 2>&1 || true
  sweep_llama
  free_port "$PORT"
}
trap cleanup EXIT
trap 'exit 130' INT TERM

# --- open the UI once servers are up -----------------------------------------
( sleep 6
  if have xdg-open; then xdg-open "http://localhost:$FPORT"
  elif have open; then open "http://localhost:$FPORT"; fi ) >/dev/null 2>&1 &

# --- frontend (foreground; blocks until Ctrl+C) ------------------------------
( cd frontend && npm run dev )
