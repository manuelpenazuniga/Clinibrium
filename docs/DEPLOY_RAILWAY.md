# Deploying to Railway (Hobby plan)

One Railway project with **3 services from this same repo** (monorepo: each service
points at the repo with its own *root directory*). Postgres/pgvector is optional:
without `DATABASE_URL` the backend degrades to `InlineGrounding` (AD-10) and the
demo is functionally identical.

The `railway.json` in each directory already pins the start command and healthcheck.
The only things configured in the dashboard are root directory, variables and domains.

## Hard-won gotchas (all hit in production — read before touching anything)

1. **The private-networking hostname keeps the service's ORIGINAL name.** Renaming a
   service to `ml-engine` does not update its internal domain — check the real one in
   the service's Settings → Networking (ours stayed `clinibrium.railway.internal`) and
   point `ML_PREDICT_URL` at that, or edit the internal domain to match. A wrong name
   fails as an instant `ConnectError` (DNS), indistinguishable from a refused bind.
2. **Config changes stage; they do NOT apply until a (re)deploy**, and a git-push
   rebuild does not pick up changes staged after it started. After editing variables
   or settings, hit Deploy in the dashboard and confirm no staged changes remain.
3. **ml_engine must serve dual-stack**: Railway healthchecks connect over IPv4, the
   backend reaches it over the private network (IPv6). uvicorn can only bind one
   stack, so ml_engine serves with hypercorn (`--bind [::]:$PORT` accepts both).
   The backend/frontend only receive edge traffic, so plain uvicorn IPv4 is fine there.

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
  - `ML_PREDICT_URL=http://<internal-domain>:8001` — the ml_engine service's
    internal domain as shown in its Settings → Networking (see gotcha 1: it is
    NOT necessarily the service's current name; ours is
    `clinibrium.railway.internal`).
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
