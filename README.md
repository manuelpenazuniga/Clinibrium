# Clinibrium

> **Clinibrium** (product) · **VertigoDx Engine** (otoneurological clinical engine).
> Vertigo diagnostic support that demonstrates how to **fail safely**:
> deterministic layers pin down safety, Claude explains, and **the physician decides**.
> Hackathon prototype — **not intended for real clinical use**.

Modular monolith: FastAPI backend (async) + Next.js/TypeScript frontend + PostgreSQL 16 with
`pgvector`. Dangerous reasoning is **never** left to the LLM: deterministic red-flag rules +
hard rails decide urgency; Claude reconciles and explains. Every evaluation emits
**exactly 1 `AuditEvent`** (guaranteed) and the FHIR artifact is **tamper-evident** via
SHA-256 hash. Ocular video is processed **on-device**; only de-identified structured
features ever cross the network.

> **Note on language**: the clinical UI is intentionally in Spanish — the product targets
> Chilean clinicians. Code, comments, and documentation are in English.

## Prior-art disclosure
> The clinical logic (ICVD criteria, red-flag rules, layered architecture) comes from the
> team's prior research (roadmap v7.2, 105+ audited references) and an earlier prototype
> built for the Gemma Hackathon. **All code was built during Built with Claude:
> Life Sciences (July 7–13, 2026) with Claude Code.**

## Architecture
```
CaseFeatures → RedFlagEngine ⟂ DifferentialEngine → ML?(track B) → grounding(RAG) →
  Reasoner(Claude, pick_model Opus/Haiku) → Rails(hard invariants) → AuditEvent → PipelineResult + FHIR Bundle
```
- `RedFlagEngine` is **physically separated** from `DifferentialEngine` (regulatory regime).
- **Rails** (applied AFTER Claude, always win): `red_flag ⇒ immediate urgency`,
  safety monotonicity, Epley blocking, uncertainty → escalate.
- **Graceful degradation**: if the ML (track B) or Claude goes down, the pipeline still
  completes and **safety does not change** (the deterministic layers own it).

## Status
Backend + frontend + Dix-Hallpike Tier 1 module **implemented, tested (INV-1..8) and
verified end-to-end**. Working demo. The **safety rails** (invariant families A–E: core red
flags, other urgencies, Dix-Hallpike contraindications, Epley blocking, epistemic — incl.
defensive A9/A10) were reviewed and **accepted by a subspecialist otolaryngologist**
(T-CLIN round 1). **Pending (round 2)**: differential diagnostic weights, A7 threshold
(age + vascular risk), physician-facing messages, and medico-legal review. Track B (ML) is
optional. The system is **not clinically validated as a whole** (provisional weights, no
prospective validation).

## Quick start
```bash
# One command — sets up venvs / model / node_modules if missing, then launches
# ml_engine (:8001) + backend (:8000, DEMO_MODE) + frontend (:3000). Ctrl-C stops all.
ANTHROPIC_API_KEY=<key> ./demo/start.sh   # key optional; reasoner degrades gracefully without it
# → http://localhost:3000/demo

# Or manually:
cd backend && python3.12 -m venv .venv && .venv/bin/pip install -e ".[dev]"
ANTHROPIC_API_KEY=<key> .venv/bin/python -m uvicorn clinibrium.api:app --port 8000
cd frontend && npm install && NEXT_PUBLIC_API_URL=http://localhost:8000 npm run dev
# Full gate: ./check.sh
```

## Privacy
- Video/frames are processed **locally** (MediaPipe on-device); **0 frames are sent** to the backend.
- Claude only receives de-identified structured features (*fail-closed* validator, INV-2).
- Every call ⇒ 1 `AuditEvent`; the FHIR Bundle carries a verifiable hash.
- (The module loads MediaPipe assets from a CDN; video *processing* is local. This is not a fully offline mode.)

## Limitations (honest)
- **Not approved for clinical use.** The safety rails (invariant families A–E) are **signed
  off by the subspecialist otolaryngologist** (round 1); the **diagnostic weights and the A7
  threshold remain provisional** (round 2). No prospective validation.
- Nystagmus tracking is **experimental** (relative velocities, no validated °/s calibration);
  torsion is **confirmed by the physician**.
- The artifact is a **FHIR R4 Clinical Case Bundle** (CL Core profiles where they exist),
  **not** a complete IPS-CL.
- The regulatory classification (FDA) is a **hypothesis** contingent on intended use, not a
  confirmed classification.

## Claude Code Safety Harness
A reusable Claude-native pattern used to build this repo: whenever Claude Code edits a
*safety-critical* file (red flags, rails, reasoner, orchestrator, contracts), a **hook**
automatically runs the relevant **invariant** tests (INV-1/2/4/5/7/8) and **blocks** the
edit if a safety guarantee broke. A companion **skill** guides turning a specialist-signed
rule into verifiable code — with mandatory adversarial tests, without the model ever making
the clinical call. *Claude turns human expertise into verifiable artifacts; the
deterministic runtime makes them trustworthy.*

## Layout
- `backend/clinibrium/` — engines, rails, reasoner, orchestrator, audit, storage, fhir, api, grounding, ml_client, contracts.
- `ml_engine/` — Track B: domain-agnostic ML confidence layer (isolated package, experimental/synthetic).
- `frontend/` — Next.js: landing (`/`), pipeline demo (`/demo`, with onboarding) and Dix-Hallpike module (`/dix-hallpike`).
- `docs/CONTRACT_predict.md` — frozen `POST /predict` contract (track A ↔ B boundary).
- `docker-compose.yml` — `pgvector/pgvector:pg16`. · `check.sh` — CI gate.

## License
Apache-2.0 — see [LICENSE](LICENSE) (explicit patent grant) and [NOTICE](NOTICE)
(third-party and prior-art attributions). Not for clinical use; see Limitations above.
