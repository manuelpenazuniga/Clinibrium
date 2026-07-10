#!/usr/bin/env bash
# Gate de Clinibrium. Sale 0 en verde.
# Backend: ruff + mypy + pytest. Frontend: typecheck + lint + build.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

PY_BIN="${PY_BIN:-/opt/homebrew/bin/python3.12}"
if [[ ! -x "$PY_BIN" ]]; then
  PY_BIN="$(command -v python3.12 || command -v python3)"
fi

step() { printf '\n=== %s ===\n' "$*"; }
ok()   { printf '  ok: %s\n' "$*"; }
fail() { printf '  FAIL: %s\n' "$*" >&2; exit 1; }

# ---------- BACKEND ----------
step "backend: setup"
cd "$REPO_ROOT/backend"
if [[ ! -d ".venv" ]]; then
  "$PY_BIN" -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate
python -m pip install --upgrade pip >/dev/null
pip install -e ".[dev]" >/dev/null
ok "venv listo, deps instaladas"

step "backend: ruff check"
ruff check clinibrium
ok "ruff"

step "backend: mypy"
mypy clinibrium
ok "mypy"

step "backend: pytest"
pytest -q
ok "pytest"

# ---------- FRONTEND ----------
step "frontend: setup"
cd "$REPO_ROOT/frontend"
if [[ ! -d "node_modules" ]]; then
  npm install --no-audit --no-fund --silent
fi
ok "node_modules listo"

step "frontend: typecheck"
npm run -s typecheck
ok "tsc --noEmit"

step "frontend: lint"
npm run -s lint
ok "next lint"

step "frontend: unit tests (vitest)"
npm run -s test
ok "vitest"

step "frontend: build"
npm run -s build
ok "next build"

# ---------- DOCKER COMPOSE VALIDATION ----------
step "docker-compose config"
if command -v docker >/dev/null 2>&1; then
  (cd "$REPO_ROOT" && docker compose config -q) && ok "docker compose config válido"
else
  echo "  skip: docker no disponible"
fi

step "GATE OK"
