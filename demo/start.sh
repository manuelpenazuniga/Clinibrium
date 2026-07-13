#!/usr/bin/env bash
# One-command demo launcher (P1.7).
#
# Brings up the full Clinibrium stack from a clean clone:
#   - ml_engine   :8001  (Track B ML confidence layer, optional)
#   - backend     :8000  (Track A pipeline; DEMO_MODE enables the Kill-Claude toggle)
#   - frontend    :3000  (landing / demo / Dix-Hallpike)
#
# It sets up venvs / node_modules / the trained model if missing (idempotent),
# then launches everything and waits until all three are healthy. Ctrl-C stops all.
#
# Usage:
#   ANTHROPIC_API_KEY=sk-... ./demo/start.sh          # full demo (Claude reasoning live)
#   ./demo/start.sh                                   # works too; reasoner degrades gracefully
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PY_BIN="${PY_BIN:-/opt/homebrew/bin/python3.12}"
[[ -x "$PY_BIN" ]] || PY_BIN="$(command -v python3.12 || command -v python3)"

log()  { printf '\033[36m[demo]\033[0m %s\n' "$*"; }
die()  { printf '\033[31m[demo] ERROR:\033[0m %s\n' "$*" >&2; exit 1; }

# Load a local .env (e.g. ANTHROPIC_API_KEY) if present — never committed.
if [[ -f "$ROOT/.env" ]]; then set -a; . "$ROOT/.env"; set +a; fi

# Refuse to start if any demo port is already taken (codex-audit-3 Alta 5):
# we NEVER kill a process we did not start, so the ports must be ours alone.
# This also makes the port-based cleanup below safe — anything listening on
# these ports at exit was spawned by this run.
for port in 8000 8001 3000; do
  if lsof -ti "tcp:$port" >/dev/null 2>&1; then
    die "port $port is already in use — stop that process first (lsof -i tcp:$port) or free the port"
  fi
done

PIDS=()
cleanup() {
  log "stopping…"
  for pid in "${PIDS[@]:-}"; do
    pkill -P "$pid" 2>/dev/null || true   # children of the launcher subshells (uvicorn/npm)
    kill "$pid" 2>/dev/null || true
  done
  # Backstop for grandchildren (next dev workers, uvicorn reloaders): the
  # preflight guaranteed these ports were free at startup, so any listener
  # now belongs to this run.
  for port in 8000 8001 3000; do lsof -ti "tcp:$port" 2>/dev/null | xargs kill -9 2>/dev/null || true; done
}
trap cleanup EXIT INT TERM

wait_health() {  # <url> <name>
  for _ in $(seq 1 60); do
    if [[ "$(curl -s -o /dev/null -w '%{http_code}' "$1" 2>/dev/null)" == "200" ]]; then return 0; fi
    sleep 1
  done
  die "$2 did not become healthy at $1"
}

# ---------------------------------------------------------------- setup (idempotent)
log "setup: ml_engine venv + deps"
if [[ ! -d ml_engine/.venv ]]; then
  ( cd ml_engine && "$PY_BIN" -m venv .venv && .venv/bin/pip install -q -e ".[dev]" )
fi
if [[ ! -f ml_engine/ml_engine/artifacts/model/manifest.json ]]; then
  log "setup: training the synthetic ML model (one-time, ~20s)"
  ( cd ml_engine && .venv/bin/python -m ml_engine.train )
fi

log "setup: backend venv + deps"
if [[ ! -d backend/.venv ]]; then
  ( cd backend && "$PY_BIN" -m venv .venv && .venv/bin/pip install -q -e ".[dev]" )
fi

log "setup: frontend node_modules"
if [[ ! -d frontend/node_modules ]]; then
  ( cd frontend && npm install --no-audit --no-fund --silent )
fi

# ---------------------------------------------------------------- launch
log "launch: ml_engine → :8001"
( cd ml_engine && ML_ARTIFACTS_DIR="$ROOT/ml_engine/ml_engine/artifacts/model" \
    .venv/bin/python -m uvicorn ml_engine.service.app:app --port 8001 --log-level warning ) &
PIDS+=($!)
wait_health "http://localhost:8001/health" "ml_engine"

log "launch: backend → :8000 (DEMO_MODE on)"
( cd backend && ANTHROPIC_API_KEY="${ANTHROPIC_API_KEY:-}" DEMO_MODE=true RECORDING_MODE=true \
    ML_PREDICT_URL="http://localhost:8001" \
    .venv/bin/python -m uvicorn clinibrium.api:app --port 8000 --log-level warning ) &
PIDS+=($!)
wait_health "http://localhost:8000/health" "backend"

log "launch: frontend → :3000"
( cd frontend && NEXT_PUBLIC_API_URL="http://localhost:8000" npm run dev --silent ) &
PIDS+=($!)
wait_health "http://localhost:3000" "frontend"

printf '\n\033[32m[demo] ready\033[0m\n'
printf '  Landing   → http://localhost:3000\n'
printf '  Demo      → http://localhost:3000/demo\n'
printf '  Dix-Hallpike → http://localhost:3000/dix-hallpike\n'
[[ -n "${ANTHROPIC_API_KEY:-}" ]] || printf '  (no ANTHROPIC_API_KEY set — Claude reasoning will show as degraded; safety is unchanged)\n'
printf '\nPress Ctrl-C to stop everything.\n\n'
wait
