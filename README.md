# Clinibrium

> **Clinibrium** (producto) · **VertigoDx Engine** (motor clínico otoneurológico).
> Apoyo diagnóstico de vértigo que demuestra cómo **fallar de forma segura**:
> las capas deterministas fijan la seguridad, Claude explica y **el médico decide**.
> Prototipo de hackathon — **no destinado a uso clínico real**.

Monolito modular: backend FastAPI (async) + frontend Next.js/TypeScript + PostgreSQL 16 con
`pgvector`. El razonamiento peligroso **nunca** es criterio del LLM: reglas deterministas de
red flags + rieles duros deciden la urgencia; Claude concilia y explica. Cada evaluación emite
**exactamente 1 `AuditEvent`** (garantizado) y el artefacto FHIR es **tamper-evident** vía
hash SHA-256. El video ocular se procesa **on-device**; solo cruzan la red features
estructuradas desidentificadas.

## Divulgación de prior-art
> La lógica clínica (criterios ICVD, reglas de red flags, arquitectura de capas) proviene de
> investigación previa del equipo (roadmap v7.2, 105+ referencias auditadas) y de un prototipo
> anterior en el Gemma Hackathon. **Todo el código se construyó durante Built with Claude:
> Life Sciences (7–13 julio 2026) con Claude Code.**

## Arquitectura
```
CaseFeatures → RedFlagEngine ⟂ DifferentialEngine → ML?(track B) → grounding(RAG) →
  Reasoner(Claude, pick_model Opus/Haiku) → Rails(invariantes duros) → AuditEvent → PipelineResult + Bundle FHIR
```
- `RedFlagEngine` **físicamente separado** de `DifferentialEngine` (régimen regulatorio).
- **Rieles** (aplicados DESPUÉS de Claude, ganan siempre): `red_flag ⇒ urgencia inmediata`,
  monotonía de seguridad, bloqueo de Epley, incertidumbre → escalar.
- **Degradación elegante**: si el ML (track B) o Claude caen, el pipeline completa y la
  **seguridad no cambia** (la deciden las capas deterministas).
- Ver `ARCHITECTURE.md` (decisiones AD-1..14, invariantes INV-1..8).

## Estado
Backend + frontend + módulo Dix-Hallpike Tier 1 **implementados, testeados (INV-1..8) y
verificados end-to-end**. Demo funcional. Los **rieles de seguridad** (invariantes A–E: red
flags centrales, otras urgencias, contraindicaciones Dix-Hallpike, bloqueo de Epley, epistémicas
— incl. A9/A10 defensivas) fueron revisados y **aceptados por un otorrino super-especialista**
(T-CLIN ronda 1, `docs/CLINICAL_VALIDATION_PENDING.md`). **Pendiente (ronda 2)**: pesos
diagnósticos del *differential*, umbral A7 (edad + riesgo vascular), mensajes al médico y revisión
médico-legal. Track B (ML) opcional. El sistema **no está clínicamente validado como conjunto**
(pesos provisionales, sin validación prospectiva).

## Quick start
```bash
# Backend (:8000). Con ANTHROPIC_API_KEY el reasoner funciona; sin ella, degrada elegante.
cd backend && python3.12 -m venv .venv && .venv/bin/pip install -e ".[dev]"
ANTHROPIC_API_KEY=<key> .venv/bin/python -m uvicorn clinibrium.api:app --port 8000
# Frontend (:3000)
cd frontend && npm install && NEXT_PUBLIC_API_URL=http://localhost:8000 npm run dev
# Gate completo: ./check.sh   ·   Demo: ver docs/DEMO_GUION.md
```

## Privacidad
- El video/frames se procesan **localmente** (MediaPipe on-device); **0 frames se envían** al backend.
- A Claude solo llegan features estructuradas desidentificadas (validador *fail-closed* INV-2).
- Cada llamada ⇒ 1 `AuditEvent`; el Bundle FHIR tiene hash verificable.
- (El módulo carga assets de MediaPipe desde CDN; el *procesamiento* del video es local. No es un modo offline completo.)

## Limitaciones (honestas)
- **No aprobado para uso clínico.** Los rieles de seguridad (invariantes A–E) están **firmados por el otorrino super-especialista** (r1); los **pesos diagnósticos y el umbral A7 siguen provisionales** (r2). Sin validación prospectiva.
- El tracking de nistagmo es **experimental** (velocidades relativas, sin calibración a °/s validada); la torsión la **confirma el médico**.
- El artefacto es un **FHIR R4 Clinical Case Bundle** (perfiles CL Core donde existen), **no un IPS-CL** completo.
- La clasificación regulatoria (FDA) es una **hipótesis** dependiente del intended use, no una clasificación confirmada.

## Claude Code Safety Harness
Patrón Claude-native reutilizable (`.claude/`): cuando Claude Code edita un
archivo *safety-critical* (red flags, rieles, reasoner, orchestrator, contracts),
un **hook** (`hooks/verify-clinical-invariants.sh`) corre automáticamente los
tests de **invariante** relevantes (INV-1/2/4/5/7/8) y **bloquea** si se rompió
una garantía de seguridad. La **skill** `clinical-rail-authoring` guía la
conversión de una regla firmada por el especialista en código verificable — con
tests adversariales obligatorios, sin que el modelo decida clínica. *Claude
convierte expertise humano en artefactos verificables; el runtime determinista
los hace confiables.*

## Estructura
- `backend/clinibrium/` — engines, rails, reasoner, orchestrator, audit, storage, fhir, api, grounding, ml_client, contracts.
- `ml_engine/` — Track B: capa de confianza ML agnóstica de dominio (paquete aislado, experimental/sintético).
- `.claude/` — Safety Harness (hook de invariantes + skill de autoría de rieles).
- `frontend/` — Next.js: landing (`/`), pipeline demo (`/demo`, con onboarding) y módulo Dix-Hallpike (`/dix-hallpike`).
- `docker-compose.yml` — `pgvector/pgvector:pg16`. · `check.sh` — gate.
