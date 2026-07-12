#!/usr/bin/env bash
# Safety Harness (Codex audit2 §9.1) — hook PostToolUse.
#
# Cuando Claude Code toca un archivo SAFETY-CRITICAL (red flags, rieles,
# differential, reasoner, orchestrator, contracts), corre los tests de
# INVARIANTE relevantes (INV-1/2/4/5/7/8). Si fallan, exit 2 → Claude ve el
# error y NO sigue como si nada (el harness le devuelve stderr).
#
# Degrada elegante: ante cualquier duda (no es safety-critical, no hay venv,
# no se pudo parsear el input) NO bloquea (exit 0). NUNCA bloquea un edit que
# no toca la seguridad.
set -uo pipefail

# Raíz del repo derivada de la ubicación del script (.claude/hooks/ → ../../).
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BACKEND="$ROOT/backend"

# El PostToolUse pasa un JSON por stdin con tool_input.file_path.
INPUT="$(cat 2>/dev/null || true)"
FILE="$(printf '%s' "$INPUT" | python3 -c 'import sys,json;print(json.load(sys.stdin).get("tool_input",{}).get("file_path",""))' 2>/dev/null || true)"

# Elegir el subset de invariantes por el path tocado.
KEXPR=""
case "$FILE" in
  *clinibrium/redflag_engine/*|*clinibrium/rails/*|*clinibrium/differential_engine/*|*clinibrium/orchestrator/*|*clinibrium/contracts/*)
    TESTS="tests/test_redflag_engine.py tests/test_rails.py tests/test_orchestrator.py"
    INV="INV-1 (red flag⇒inmediata) · INV-4 (1 AuditEvent) · INV-5 (separación) · INV-7 (monotonía de rieles)"
    ;;
  *clinibrium/reasoner/*)
    TESTS="tests/test_reasoner.py"
    KEXPR="privacy or degrad or thinking or urgency or happy_path"  # evita los tests lentos de backoff
    INV="INV-2 (allowlist de privacidad) · INV-8 (reasoner degrada sin cambiar la seguridad)"
    ;;
  *)
    exit 0  # no es safety-critical → no-op silencioso
    ;;
esac

PY="$BACKEND/.venv/bin/python"
[ -x "$PY" ] || exit 0            # sin venv → no bloquear
cd "$BACKEND" 2>/dev/null || exit 0

# shellcheck disable=SC2086
if OUT="$("$PY" -m pytest -q $TESTS ${KEXPR:+-k "$KEXPR"} 2>&1)"; then
  echo "✓ Safety Harness — invariantes clínicos OK tras editar $(basename "$FILE"): $INV" >&2
  exit 0
else
  {
    echo "🚨 Safety Harness — INVARIANTES CLÍNICOS FALLARON tras editar $(basename "$FILE")."
    echo "   Verificados: $INV"
    echo "   El cambio puede haber roto una garantía de seguridad. NO continúes sin arreglarlo."
    echo "   ---- pytest (cola) ----"
    printf '%s\n' "$OUT" | tail -20
  } >&2
  exit 2
fi
