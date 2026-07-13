#!/usr/bin/env bash
# Track B gate (ml_engine) — ISOLATED from A's gate.
# The root gate (A's check.sh) does NOT run this unless ml_engine/.venv exists
# (Codex/Gemini: a broken ML dep must NEVER take down A's gate · INV-6).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

PY_BIN="${PY_BIN:-/opt/homebrew/bin/python3.12}"
[[ -x "$PY_BIN" ]] || PY_BIN="$(command -v python3.12 || command -v python3)"

step() { printf '\n=== ml_engine: %s ===\n' "$*"; }
ok()   { printf '  ok: %s\n' "$*"; }

step "setup"
if [[ ! -d ".venv" ]]; then
  "$PY_BIN" -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate
python -m pip install --upgrade pip >/dev/null
pip install -e ".[dev]" >/dev/null
ok "ml_engine venv ready, ML deps installed"

step "ruff"
ruff check ml_engine
ok "ruff"

step "mypy"
mypy ml_engine
ok "mypy"

step "pytest"
pytest -q
ok "pytest"

printf '\n=== ml_engine GATE OK ===\n'
