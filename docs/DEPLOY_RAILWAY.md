# Deploying to Railway (Hobby plan)

One Railway project with **3 services from this same repo** (monorepo: each service
points at the repo with its own *root directory*). Postgres/pgvector is optional:
without `DATABASE_URL` the backend degrades to `InlineGrounding` (AD-10) and the
demo is functionally identical.

The `railway.json` in each directory already pins the start command and healthcheck.
uvicorn binds to `0.0.0.0` (IPv4): Railway's healthchecks and public edge connect over
IPv4, uvicorn cannot dual-stack bind, and private networking is dual-stack (IPv4+IPv6)
in environments created after Oct 2025 — so IPv4-only works for all three paths.
The only things configured in the dashboard are root directory, variables and domains.

## 1. ml_engine

- **Root directory**: `ml_engine`
- **Public domain**: NO (private networking only)
- **Variables**:
  - `PORT=8001` — must match the port in the backend's `ML_PREDICT_URL`.
- The start command trains the synthetic model on boot (~20 s; the filesystem is
  ephemeral, so it retrains on every deploy; the healthcheck waits up to 120 s).

## 2. backend

- **Root directory**: `backend`
- **Public domain**: YES (the browser calls it directly)
- **Variables**:
  - `ANTHROPIC_API_KEY=sk-...`
  - `DEMO_MODE=true` (enables the Kill-Claude toggle)
  - `RECORDING_MODE=true` (forces Opus for ambulatory, as in `demo/start.sh`)
  - `ML_PREDICT_URL=http://ml-engine.railway.internal:8001` — adjust `ml-engine`
    to the actual service name in Railway.
  - `CORS_ORIGINS=https://<frontend>.up.railway.app` — the frontend's public URL,
    no trailing slash. Accepts a comma-separated list.
  - Do NOT set `DATABASE_URL` (unless deploying pgvector; see below).

## 3. frontend

- **Root directory**: `frontend`
- **Public domain**: YES
- **Variables** (⚠ build-time: set it BEFORE the first build; if it changes, redeploy):
  - `NEXT_PUBLIC_API_URL=https://<backend>.up.railway.app` — the backend's public URL.

## Startup order

There is no orchestration between services: if the backend boots before ml_engine,
`ml_client` degrades gracefully (INV-6) and the first request may show ML as
unavailable — it recovers on its own. For the demo, deploy ml_engine first.

## Postgres/pgvector (optional)

A fourth service running the `pgvector/pgvector:pg16` image + a volume (same setup
as `docker-compose.yml`), `DATABASE_URL` on the backend, and an embeddings seed step.
Without it, `InlineGrounding` covers the demo; the graceful degradation is part of
the narrative.

## Localhost is unaffected

`demo/start.sh` keeps working unchanged: `CORS_ORIGINS` defaults to
`http://localhost:3000`, `NEXT_PUBLIC_API_URL` falls back to
`http://localhost:8000`, and the `railway.json` files are inert outside Railway.
