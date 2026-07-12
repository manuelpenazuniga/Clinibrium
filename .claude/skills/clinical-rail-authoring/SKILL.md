---
name: clinical-rail-authoring
description: Convierte una regla de seguridad APROBADA por el especialista en un cambio verificable del RedFlagEngine/rails de Clinibrium, con tests adversariales obligatorios y ejecución de invariantes. Claude NUNCA decide clínica ni declara validación; se detiene ante ambigüedad. Úsala cuando el especialista firma una nueva red flag / riel / umbral.
---

# Clinical Rail Authoring — Safety Harness

Guía para transformar una **regla de seguridad firmada por el especialista**
(T-CLIN) en código verificable de Clinibrium, sin que el modelo decida clínica.
Es el patrón central del proyecto: *Claude convierte expertise humano en
artefactos verificables; el runtime determinista los hace confiables.*

## Precondiciones — si falta ALGUNA, DETENTE y devolvé la pregunta al humano
- La regla viene **firmada por el especialista**, no inferida por el modelo.
- Está expresada como **condiciones → acción forzada**, usando SOLO:
  - campos del allowlist `CaseFeatures` (`backend/clinibrium/contracts/features.py`);
  - enums existentes (`contracts/enums.py`);
  - una o más `ForcedAction` (`DERIVAR_URGENTE`, `NO_BENIGNO`, `BLOQUEAR_EPLEY`,
    `PRECAUCION_EXAMEN`, `RED_SEGURIDAD`, `ESCALAR`).
- NO introduce un diagnóstico nuevo ni un umbral numérico que el especialista no fijó.

## Procedimiento (en orden)
1. **Mapear** la regla a la tabla determinista existente:
   - Red flag → una `RedFlagRule(id, label, severity, forced_actions, predicate)`
     en `redflag_engine/rules.py` (el `predicate` es una función pura de `CaseFeatures`).
   - Contraindicación / bloqueo de Epley / escalamiento → `rails/engine.py`.
   - Reusá helpers existentes (`_avs`, etc.); no dupliques lógica.
2. **Escribir tests adversariales OBLIGATORIOS** (antes o junto al código):
   - **positivo** (dispara), **negativo** (no dispara) y **borde**;
   - un test de **monotonía (INV-7)**: la regla nunca BAJA la urgencia;
   - si toca lo que cruza la red, un test de **allowlist (INV-2)**;
   - si es red flag central, que **fuerce `inmediata` (INV-1)** pese a ML/Claude.
3. **Correr el subset de invariantes** relevante. El hook
   `.claude/hooks/verify-clinical-invariants.sh` lo hace automático al guardar;
   manual: `cd backend && .venv/bin/python -m pytest -q tests/test_redflag_engine.py tests/test_rails.py`.
4. **Change report** (breve): qué regla, qué `RedFlagRule`/riel, qué tests
   nuevos, qué invariantes corridos y su resultado, y el diff.
5. **Detenerse ante ambigüedad clínica**: si la regla no es unívoca (umbral sin
   número, "según criterio", combinación no especificada) NO adivines — marcá
   `# TODO(clinical)` y devolvé la pregunta al especialista.
6. **NUNCA**: declarar validación clínica, firmar por el especialista, ni afirmar
   cumplimiento regulatorio. Lo provisional va etiquetado `# TODO(clinical)`.

## Invariantes que ningún cambio puede romper (el hook los verifica)
- **INV-1** — `red_flag_activa ⇒ urgencia == inmediata` (gana siempre, sobre ML y Claude).
- **INV-2** — solo campos del allowlist `CaseFeatures` cruzan la red (fail-closed).
- **INV-5** — `RedFlagEngine ⟂ DifferentialEngine` (no se acoplan por imports).
- **INV-7** — los rieles solo SUBEN urgencia (monotonía), idempotencia, totalidad.
- **INV-8** — reasoner caído NO cambia urgencia / red flags / forced_actions.

## Sesgo de seguridad (no negociable)
Los provisionales se calibran hacia **sobre-triaje**: un falso positivo (derivar
de más) es tolerable; un falso negativo (emergencia perdida) no. El especialista
puede *bajar* umbrales; el sistema nunca *sube* el riesgo de un miss por defecto.
**El médico decide siempre (Ley 21.719).**
