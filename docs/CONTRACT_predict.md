# Contrato `POST /predict` (track B — ML opcional) — **CONGELADO**

> **Contrato congelado 2026-07-10.** Cambios requieren acuerdo explícito de
> ambas personas (regla dura v7.3 §9). Esto es lo que persona 2 (ML) puede
> implementar offline sin bloquear a persona 1 (A).

---

## Posición en el sistema

```
RedFlagEngine → DifferentialEngine → ML? (opcional, este contrato) →
                Claude razonador → rails + AuditEvent → médico decide
```

- **B es ADITIVO**: A lo llama solo si está disponible.
- **A NUNCA depende de que B esté vivo.** Si B no está, A completa igual
  con `ml=None` (ver `PredictResponse | None` en `PipelineResult`).
- **B NO decide urgencia ni red flags** — esos los sellan `RedFlagEngine`
  y los rieles (INV-1, INV-3, INV-5). B es un input más para el razonador.

## Endpoint

| Campo        | Valor                                  |
|--------------|----------------------------------------|
| Método       | `POST`                                 |
| Ruta         | `/predict`                             |
| Content-Type | `application/json` (request y response)|
| Timeout cliente | `2.0 s` (configurable vía `ML_PREDICT_TIMEOUT_S`) |

Base URL configurable vía `ML_PREDICT_URL` en `Settings` (env var del
mismo nombre). Si es `None`, el cliente degrada inmediatamente a `None`
y el pipeline A sigue funcionando (INV-6).

## Request — JSON

Body: dump de `clinibrium.contracts.CaseFeatures` con
`model_dump(mode="json")`. El cliente se encarga de serializar enums
como string y sets como list.

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

Todos los campos son los del allowlist `NETWORK_SAFE_FIELDS` (ver
`clinibrium/contracts/features.py`). El servidor NO debe esperar ni
aceptar PII, texto libre, ni video — el cliente garantiza el allowlist
y el validador del reasoner (INV-2) lo refuerza en el otro extremo.

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

| Campo                       | Tipo                  | Notas                                                                                          |
|-----------------------------|-----------------------|------------------------------------------------------------------------------------------------|
| `probabilities`             | `dict[str, float]`    | **Claves = valores del enum `Diagnosis`** (ver tabla abajo). Suma ≈ 1.0; cada valor ∈ [0, 1]. |
| `shap`                      | `dict[str, float] \| null` | SHAP por feature (mismo vocabulario que `CaseFeatures`, kebab/snake libre). `null` si B no calcula SHAP. |
| `model_version`             | `str`                 | Identificador de la versión del modelo. Aparece en `AuditEvent` y logs.                        |

### Claves válidas de `probabilities` (enum `Diagnosis`)

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

B debe emitir al menos una clave; el cliente no exige que estén todas
(admite sub-sets) pero el modelo Pydantic `PredictResponse` no valida
las claves — se confía en el contrato social entre A y B.

## Códigos de error

| Status | Significado para el cliente                                  |
|--------|--------------------------------------------------------------|
| 2xx    | Éxito. Body parseado como `PredictResponse`.                 |
| 4xx/5xx| Error del servidor. Cliente degrada a `None` (INV-6).        |
| Timeout| Cliente degrada a `None` tras `ML_PREDICT_TIMEOUT_S`.        |
| Red    | Cualquier excepción (`httpx.HTTPError`, `ConnectionError`, etc.) → `None`. |

El cliente NUNCA levanta excepciones por fallas de B. Si B está mal
implementado y devuelve 200 con JSON inválido, también degrada a `None`
(log info, no error — es comportamiento esperado del degradation path).

## Semántica (lo que B puede y NO puede hacer)

✅ **Puede**:
- Devolver una distribución de probabilidad sobre los diagnósticos
  definidos en `Diagnosis`.
- Acompañar la predicción con valores SHAP por feature para explicabilidad.
- Declarar su `model_version` para reproducibilidad/auditoría.

❌ **NO puede**:
- Fijar `urgency` (lo sellan `RedFlagEngine` + rails, INV-1).
- Anular `red_flag_activa` (INV-1, INV-5).
- Pedir ni recibir PII, texto libre, ni video (INV-2, INV-7).
- Asumir que el cliente reintentará — el contrato es fire-and-forget:
  falla ⇒ `None`, el pipeline sigue.

## Historial

- **2026-07-10** — Contrato congelado (v7.3 §9, AD-3). Cliente
  `ml_client.predict()` con degradación elegante implementado.
  Stub dev en `clinibrium.ml_client.stub_server` para validar la
  ruta feliz sin el servicio real.
