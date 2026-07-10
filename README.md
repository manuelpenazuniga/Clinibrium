# Clinibrium

> Agente clínico otoneurológico (VertigoDx) — apoyo diagnóstico. El médico decide.

**Clinibrium** es un monolito modular de apoyo diagnóstico para vértigo: backend FastAPI
(async) + frontend Next.js/TypeScript + PostgreSQL 16 con `pgvector`. Las capas deterministas
clasifican; Claude (Anthropic) explica y concilia. Toda invocación emite 1 `AuditEvent`
inmutable. PII, texto libre con identificadores y video ocular nunca cruzan la red.

## Divulgación de prior-art

> La lógica clínica (criterios ICVD, reglas de red flags, arquitectura de capas) proviene
> de investigación previa del equipo (roadmap v7.2, 105+ referencias auditadas) y de un
> prototipo anterior en el Gemma Hackathon. Todo el código de este proyecto se construyó
> durante Built with Claude: Life Sciences (7–13 julio 2026) con Claude Code.

## Estructura

- `backend/` — FastAPI (Python 3.12). Layout en `backend/clinibrium/`.
- `frontend/` — Next.js 14 (App Router, TypeScript).
- `docker-compose.yml` — servicio `db` con `pgvector/pgvector:pg16`.
- `check.sh` — gate ejecutable (lint + typecheck + test + build).

## Estado actual

Scaffold (`spike/t1-scaffold`): estructura base, sin lógica clínica. Ver `docs/` y
`ARCHITECTURE.md` para el plan de ejecución completo.
