# `POST /predict` contract (track B — optional ML) — **FROZEN**

> **Contract frozen 2026-07-10.** Changes require explicit agreement from
> both people (hard rule, v7.3 §9). This is what person 2 (ML) can
> implement offline without blocking person 1 (A).

---

## Position in the system

```
RedFlagEngine → DifferentialEngine → ML? (optional, this contract) →
                Claude reasoner → rails + AuditEvent → the physician decides
```

- **B is ADDITIVE**: A calls it only if it is available.
- **A NEVER depends on B being alive.** If B is down, A still completes
  with `ml=None` (see `PredictResponse | None` in `PipelineResult`).
- **B does NOT decide urgency or red flags** — those are sealed by
  `RedFlagEngine` and the rails (INV-1, INV-3, INV-5). B is just one more
  input for the reasoner.

## Endpoint

| Field        | Value                                  |
|--------------|----------------------------------------|
| Method       | `POST`                                 |
| Path         | `/predict`                             |
| Content-Type | `application/json` (request and response)|
| Client timeout | `2.0 s` (configurable via `ML_PREDICT_TIMEOUT_S`) |

Base URL configurable via `ML_PREDICT_URL` in `Settings` (env var of the
same name). If it is `None`, the client degrades immediately to `None`
and pipeline A keeps working (INV-6).

## Request — JSON

Body: dump of `clinibrium.contracts.CaseFeatures` via
`model_dump(mode="json")`. The client is responsible for serializing
enums as strings and sets as lists.

```json
{
  "duration": "seconds",
  "onset": "sudden",
  "trigger": "positional_head",
  "timing_pattern": "episodic_triggered",
  "nystagmus_direction": "mixed",
  "nystagmus_direction_changing_gaze": false,
  "nystagmus_latency_s": 2.0,
  "nystagmus_duration_s": 20.0,
  "nystagmus_fatigable": true,
  "nystagmus_suppressed_by_fixation": true,
  "head_impulse": "normal",
  "skew_deviation": false,
  "hearing_loss": "none",
  "tinnitus": false,
  "aural_fullness": false,
  "focal_signs": [],
  "truncal_ataxia_severe": false,
  "headache_neck_pain_sudden_severe": false,
  "migrainous_features": false,
  "age_years": 60,
  "vascular_risk_factors": ["hypertension"],
  "fever": false,
  "neck_stiffness": false,
  "altered_consciousness": false,
  "presyncope_syncope": false,
  "palpitations": false,
  "chest_pain": false,
  "otitis_mastoiditis": false,
  "recent_head_neck_trauma": false,
  "cervical_pathology": false,
  "known_carotid_vertebrobasilar_disease": false,
  "cardiovascular_instability": false,
  "dix_hallpike": "right_positive",
  "torsion_confirmed_by_clinician": true,
  "episode_count": 3,
  "episode_duration": "hours",
  "worsening_during_flow": false
}
```

All fields come from the `NETWORK_SAFE_FIELDS` allowlist (see
`clinibrium/contracts/features.py`). The server must NOT expect nor
accept PII, free text, or video — the client guarantees the allowlist
and the reasoner's validator (INV-2) enforces it on the other end.

## Response — JSON (200 OK)

```json
{
  "probabilities": {
    "bppv_posterior": 0.62,
    "bppv_horizontal": 0.08,
    "vestibular_neuritis": 0.07,
    "vestibular_migraine": 0.06,
    "meniere": 0.05,
    "labyrinthitis": 0.04,
    "central_suspected": 0.03,
    "cardiogenic_suspected": 0.02,
    "undetermined": 0.03
  },
  "shap": {
    "dix_hallpike": 0.41,
    "nystagmus_mixed": 0.28,
    "nystagmus_latency_s": 0.12,
    "nystagmus_duration_s": 0.07,
    "trigger_positional_head": 0.05,
    "head_impulse_normal": -0.03,
    "hearing_loss_none": -0.02
  },
  "model_version": "catboost-v0.1"
}
```

| Field                       | Type                  | Notes                                                                                          |
|-----------------------------|-----------------------|------------------------------------------------------------------------------------------------|
| `probabilities`             | `dict[str, float]`    | **Keys = values of the `Diagnosis` enum** (table below). Sum ≈ 1.0; each value ∈ [0, 1]. |
| `shap`                      | `dict[str, float] \| null` | SHAP per feature (same vocabulary as `CaseFeatures`, kebab/snake free-form). `null` if B does not compute SHAP. |
| `model_version`             | `str`                 | Model version identifier. Appears in `AuditEvent` and logs.                        |

### Valid `probabilities` keys (`Diagnosis` enum)

```
bppv_posterior
bppv_horizontal
meniere
vestibular_migraine
vestibular_neuritis
labyrinthitis
central_suspected
cardiogenic_suspected
undetermined
```

B must emit at least one key; the client does not require all of them
(subsets are accepted) but the `PredictResponse` Pydantic model does not
validate the keys — this relies on the social contract between A and B.

## Error codes

| Status | Meaning for the client                                       |
|--------|--------------------------------------------------------------|
| 2xx    | Success. Body parsed as `PredictResponse`.                  |
| 4xx/5xx| Server error. Client degrades to `None` (INV-6).            |
| Timeout| Client degrades to `None` after `ML_PREDICT_TIMEOUT_S`.     |
| Network| Any exception (`httpx.HTTPError`, `ConnectionError`, etc.) → `None`. |

The client NEVER raises on B failures. If B is badly implemented and
returns 200 with invalid JSON, the client also degrades to `None`
(info log, not error — that is expected behavior of the degradation path).

## Semantics (what B can and canNOT do)

✅ **Can**:
- Return a probability distribution over the diagnoses defined in
  `Diagnosis`.
- Attach per-feature SHAP values to the prediction for explainability.
- Declare its `model_version` for reproducibility/auditing.

❌ **CanNOT**:
- Set `urgency` (sealed by `RedFlagEngine` + rails, INV-1).
- Override `red_flag_activa` (INV-1, INV-5).
- Request or receive PII, free text, or video (INV-2, INV-7).
- Assume the client will retry — the contract is fire-and-forget:
  failure ⇒ `None`, the pipeline continues.

## History

- **2026-07-10** — Contract frozen (v7.3 §9, AD-3). Client
  `ml_client.predict()` with graceful degradation implemented.
  Dev stub in `clinibrium.ml_client.stub_server` to validate the
  happy path without the real service.
