"""Mapea un ``PipelineResult`` a un ``Bundle`` FHIR R4 (formato de salida, AD-9).

Función pura — sin I/O, sin red. Dado:

  - ``PipelineResult`` (output del pipeline)
  - ``CaseFeatures``   (input estructurado, sin PII)
  - ``AuditEvent``     (evento inmutable de la invocación, INV-4)

produce un ``dict`` JSON-serializable con un Bundle tipo ``collection``
que agrupa los recursos FHIR R4 auditables: ``Questionnaire`` (intake
adaptativo, SDC IG), ``QuestionnaireResponse`` (respuestas
desidentificadas), ``Observation`` (variables clínicas clave),
``DetectedIssue`` + ``Flag`` (uno por cada red flag disparada),
``ClinicalImpression`` (diferencial + razonamiento) y ``AuditEvent``
(perfil CL Core ``Auditoria``).

Grafo del módulo (regla dura del mapa de Clinibrium):
    fhir → contracts   ✓
    fhir → stdlib      ✓ (uuid, datetime, enum)

PROHIBIDO importar: ``engines``, ``reasoner``, ``rails``, ``orchestrator``,
``ml_client``, ``api``, ``storage``, ``audit``, ``grounding``, ``config``.

Perfiles CL Core IG 1.9.3 (R4):
    - ``AuditEvent`` (CL ``Auditoria``) — perfil nativo CL Core;
      referenciado en ``meta.profile``.

Perfiles Clinibrium propios (CL Core NO provee para Questionnaire /
QuestionnaireResponse / ClinicalImpression — registrados como
extensión):
    - ``cl-questionnaire``
    - ``cl-questionnaire-response``
    - ``cl-clinical-impression``

Extensiones propias (URLs canónicas declaradas en este módulo):
    - ``extension-questionnaire-version``: versión del cuestionario
      adjunta al ``QuestionnaireResponse``.
    - ``extension-reasoner-degraded``: flag que marca el
      ``ClinicalImpression`` cuando el razonador (Claude) no estuvo
      disponible.
    - ``extension-rail-triggered``: lista de rail-ids (R-INV1, R-EPLEY-D,
      R-E2, R-DIVERGENCIA) disparados, en ``ClinicalImpression`` y
      ``AuditEvent.entity.detail``.

IDs determinísticos (tests estables):
    ``uuid5(_URN_NAMESPACE, f"{case_id}/{kind}/{key}")``. Mismo
    ``case_id`` + mismo ``kind`` + mismo ``key`` ⇒ mismo id.
    NO se usa ``uuid4`` aleatorio.

Referencias internas:
    ``fullUrl = "urn:uuid:{id}"``. Las referencias entre recursos
    (p.ej. ``QuestionnaireResponse.questionnaire``) usan el mismo
    esquema. Cada referencia en el bundle resuelve a un recurso
    presente en el mismo bundle.

Codes SNOMED CT / LOINC:
    Los códigos de ``Observation.code`` y de ``ClinicalImpression.code``
    son **placeholders** marcados ``TODO(clinical)`` — confirmar con el
    superespecialista antes de producción. Los códigos SNOMED como
    reglas/hechos no son copyrightables; los códigos canónicos sí
    deben ser verificados.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from clinibrium.contracts.audit import AuditEvent
from clinibrium.contracts.enums import (
    DixHallpikeResult,
    FocalSign,
    HeadImpulse,
    HearingLoss,
    NystagmusDirection,
    Onset,
    SymptomDuration,
    Trigger,
)
from clinibrium.contracts.features import CaseFeatures
from clinibrium.contracts.results import (
    DifferentialCandidate,
    PipelineResult,
    RedFlagHit,
)

# ---------------------------------------------------------------------------
# Constantes — namespaces, URLs, codes
# ---------------------------------------------------------------------------

_URN_NAMESPACE: uuid.UUID = uuid.UUID("8d3a3b3e-7c14-5b9d-9b3a-3b3e7c145b9d")
"""Namespace fijo para ``uuid5``; garantiza ids determinísticos entre runs."""

# CL Core IG 1.9.3 (R4) — perfil CL ``Auditoria`` (perfil nativo).
_CLCORE_AUDIT_PROFILE: str = "http://hl7.cl/fhir/ig/clcore/StructureDefinition/Auditoria"
_CLCORE_PATIENT_PROFILE: str = (
    "http://hl7.cl/fhir/ig/clcore/StructureDefinition/CorePacienteCl"
)

# Perfiles Clinibrium (CL Core no provee Questionnaire /
# QuestionnaireResponse / ClinicalImpression → declaramos propios).
_CLINIBRIUM_BASE: str = "http://clinibrium.cl/fhir/StructureDefinition"
_PROFILE_QUESTIONNAIRE: str = f"{_CLINIBRIUM_BASE}/cl-questionnaire"
_PROFILE_QUESTIONNAIRE_RESPONSE: str = f"{_CLINIBRIUM_BASE}/cl-questionnaire-response"
_PROFILE_CLINICAL_IMPRESSION: str = f"{_CLINIBRIUM_BASE}/cl-clinical-impression"

# Extensiones propias (URLs canónicas).
_EXT_QUESTIONNAIRE_VERSION: str = (
    f"{_CLINIBRIUM_BASE}/extension-questionnaire-version"
)
_EXT_REASONER_DEGRADED: str = f"{_CLINIBRIUM_BASE}/extension-reasoner-degraded"
_EXT_RAIL_TRIGGERED: str = f"{_CLINIBRIUM_BASE}/extension-rail-triggered"

# Code systems locales (vocabulario del dominio Clinibrium / VertigoDx).
_CS_SYMPTOM_DURATION: str = "http://clinibrium.cl/fhir/CodeSystem/symptom-duration"
_CS_ONSET: str = "http://clinibrium.cl/fhir/CodeSystem/onset"
_CS_TRIGGER: str = "http://clinibrium.cl/fhir/CodeSystem/trigger"
_CS_NYSTAGMUS_DIRECTION: str = "http://clinibrium.cl/fhir/CodeSystem/nystagmus-direction"
_CS_HEAD_IMPULSE: str = "http://clinibrium.cl/fhir/CodeSystem/head-impulse"
_CS_HEARING_LOSS: str = "http://clinibrium.cl/fhir/CodeSystem/hearing-loss"
_CS_DIX_HALLPIKE: str = "http://clinibrium.cl/fhir/CodeSystem/dix-hallpike"
_CS_FOCAL_SIGN: str = "http://clinibrium.cl/fhir/CodeSystem/focal-sign"
_CS_DIAGNOSIS: str = "http://clinibrium.cl/fhir/CodeSystem/diagnosis"
_CS_RED_FLAG: str = "http://clinibrium.cl/fhir/CodeSystem/red-flag"
_CS_AUDIT_EVENT_TYPE: str = "http://clinibrium.cl/fhir/CodeSystem/audit-event-type"
_CS_OBSERVATION_CATEGORY: str = (
    "http://terminology.hl7.org/CodeSystem/observation-category"
)
_CS_FLAG_CATEGORY: str = "http://terminology.hl7.org/CodeSystem/flag-category"
_CS_AUDIT_AGENT_TYPE: str = (
    "http://terminology.hl7.org/CodeSystem/extra-security-role-type"
)
_CS_UNITSOFMEASURE: str = "http://unitsofmeasure.org"
_UCUM_SECOND: str = "s"

# Coding systems estándar (placeholders — ver TODO(clinical) al final del módulo).
_CS_SNOMED: str = "http://snomed.info/sct"

# TODO(clinical): confirmar los SNOMED CT canónicos con el superespecialista.
_SNOMED_NYSTAGMUS_OBSERVATION: str = "271925006"  # placeholder
_SNOMED_NYSTAGMUS_DURATION_S: str = "30714003"  # placeholder
_SNOMED_NYSTAGMUS_LATENCY_S: str = "271931008"  # placeholder
_SNOMED_DIX_HALLPIKE: str = "425444002"  # placeholder
_SNOMED_CLINICAL_IMPRESSION: str = "428321000124101"  # placeholder

# AuditEvent outcome (HL7 terminology).
_CS_AUDIT_OUTCOME: str = "http://terminology.hl7.org/CodeSystem/audit-event-outcome"
_AUDIT_OUTCOME_SUCCESS: str = "0"

# AuditEvent action (R4): C/R/U/D/E.
_AUDIT_ACTION_EXECUTE: str = "E"

# Questionnaire version (constante local — bumpear al cambiar el template).
_QUESTIONNAIRE_VERSION: str = "0.1.0"
_QUESTIONNAIRE_URL: str = "urn:clinibrium:questionnaire:vertigodx-intake"
_QUESTIONNAIRE_DATE: str = "2026-07-10"

# Display strings cortos.
_DISPLAY_EXAM: str = "Exam"
_DISPLAY_CLINICAL: str = "Clinical"
_DISPLAY_HUMANUSER: str = "Human User"
_DISPLAY_CASE: str = "Case"
_DISPLAY_DATA: str = "Data"
_DISPLAY_CLINICAL_IMPRESSION: str = "Clinical impression"


# ---------------------------------------------------------------------------
# Feature → Questionnaire item mapping (template, NO las 50 features)
# ---------------------------------------------------------------------------

# Cada item del template declara: linkId, text, type, code system, y las
# options (cuando es choice). Esta tabla es la fuente de verdad para
# construir el Questionnaire (template) Y el QuestionnaireResponse
# (instance) — al ser campos representativos del intake cubre los
# features estructurados principales; el QuestionnaireResponse incluye
# además items para features no modelados en el template (linkId 11+).

_FEATURE_TEMPLATE: list[dict[str, Any]] = [
    {
        "field": "duration",
        "linkId": "1",
        "text": "Duración del síntoma",
        "type": "choice",
        "system": _CS_SYMPTOM_DURATION,
        "options": [e.value for e in SymptomDuration],
    },
    {
        "field": "onset",
        "linkId": "2",
        "text": "Inicio",
        "type": "choice",
        "system": _CS_ONSET,
        "options": [e.value for e in Onset],
    },
    {
        "field": "trigger",
        "linkId": "3",
        "text": "Trigger",
        "type": "choice",
        "system": _CS_TRIGGER,
        "options": [e.value for e in Trigger],
    },
    {
        "field": "nystagmus_direction",
        "linkId": "4",
        "text": "Dirección del nistagmo (bedside / on-device)",
        "type": "choice",
        "system": _CS_NYSTAGMUS_DIRECTION,
        "options": [e.value for e in NystagmusDirection],
    },
    {
        "field": "nystagmus_latency_s",
        "linkId": "5",
        "text": "Latencia del nistagmo (s)",
        "type": "decimal",
    },
    {
        "field": "nystagmus_duration_s",
        "linkId": "6",
        "text": "Duración del nistagmo (s)",
        "type": "decimal",
    },
    {
        "field": "head_impulse",
        "linkId": "7",
        "text": "HINTS — head impulse test",
        "type": "choice",
        "system": _CS_HEAD_IMPULSE,
        "options": [e.value for e in HeadImpulse],
    },
    {
        "field": "hearing_loss",
        "linkId": "8",
        "text": "Pérdida auditiva",
        "type": "choice",
        "system": _CS_HEARING_LOSS,
        "options": [e.value for e in HearingLoss],
    },
    {
        "field": "dix_hallpike",
        "linkId": "9",
        "text": "Dix-Hallpike",
        "type": "choice",
        "system": _CS_DIX_HALLPIKE,
        "options": [e.value for e in DixHallpikeResult],
        # enableWhen: solo se muestra si trigger=positional_head
        # (Dix-Hallpike solo aplica a sospecha posicional).
        "enableWhen": [
            {
                "question": "3",
                "operator": "=",
                "answerCoding": {
                    "system": _CS_TRIGGER,
                    "code": Trigger.positional_head.value,
                },
            }
        ],
        "enableBehavior": "all",
    },
    {
        "field": "focal_signs",
        "linkId": "10",
        "text": "Signos focales (HINTS-neuro)",
        "type": "choice",
        "repeats": True,
        "system": _CS_FOCAL_SIGN,
        "options": [e.value for e in FocalSign],
        # enableWhen: focal signs solo se relevan si onset=sudden
        # (síndrome vestibular agudo).
        "enableWhen": [
            {
                "question": "2",
                "operator": "=",
                "answerCoding": {
                    "system": _CS_ONSET,
                    "code": Onset.sudden.value,
                },
            }
        ],
        "enableBehavior": "all",
    },
]

_FEATURE_TEMPLATE_BY_FIELD: dict[str, dict[str, Any]] = {
    m["field"]: m for m in _FEATURE_TEMPLATE
}


# ---------------------------------------------------------------------------
# Helpers genéricos
# ---------------------------------------------------------------------------


def _resource(resource_type: str, **fields: Any) -> dict[str, Any]:
    """Arma un recurso FHIR R4 como dict plano.

    Patrón: ``{"resourceType": <type>, ...fields}``. ``None`` se filtra
    para no emitir campos vacíos.  Los campos cuyo valor es un dict /
    list pasan tal cual.
    """
    out: dict[str, Any] = {"resourceType": resource_type}
    for k, v in fields.items():
        if v is None:
            continue
        out[k] = v
    return out


def _coding(
    system: str | None, code: str, display: str | None = None
) -> dict[str, Any]:
    """Coding FHIR R4 como dict."""
    out: dict[str, Any] = {"code": code}
    if system is not None:
        out["system"] = system
    if display is not None:
        out["display"] = display
    return out


def _id_for(case_id: str, kind: str, key: str = "") -> str:
    """ID determinístico para un recurso del bundle.

    ``uuid5(_URN_NAMESPACE, f"{case_id}/{kind}/{key}")``. ``key`` vacío
    sigue produciendo un id estable por ``(case_id, kind)``.
    """
    return str(uuid.uuid5(_URN_NAMESPACE, f"{case_id}/{kind}/{key}"))


def _urn(resource_id: str) -> str:
    """fullUrl / reference de un recurso del bundle (``urn:uuid:{id}``)."""
    return f"urn:uuid:{resource_id}"


def _datetime_iso(dt: datetime) -> str:
    """Serializa un datetime a ISO 8601 con tz."""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.isoformat()


def _annotation(text: str, time: datetime | None = None) -> dict[str, Any]:
    """Annotation FHIR R4: ``{text, time?}``."""
    out: dict[str, Any] = {"text": text}
    if time is not None:
        out["time"] = _datetime_iso(time)
    return out


# ---------------------------------------------------------------------------
# Builders por recurso
# ---------------------------------------------------------------------------


def _build_minimal_patient(case_id: str) -> tuple[dict[str, Any], str]:
    """Patient placeholder (sin PII) para que las referencias resuelvan.

    La política de privacidad (INV-2 / AD-7) prohíbe cualquier PII o
    identificador externo; el Patient es solo un anchor de referencia
    dentro del bundle. NO incluye nombre, RUT, DOB ni dirección.
    """
    pid = _id_for(case_id, "Patient")
    resource = _resource(
        "Patient",
        id=pid,
        meta={"profile": [_CLCORE_PATIENT_PROFILE]},
        # Marcamos el Patient como no-real (placeholder para referencias).
        active=False,
        text={
            "status": "generated",
            "div": (
                "<div xmlns=\"http://www.w3.org/1999/xhtml\">"
                "Subject placeholder (Clinibrium): NO contiene PII. "
                "La identidad real del paciente permanece on-device."
                "</div>"
            ),
        },
    )
    return resource, _urn(pid)


def _build_questionnaire(case_id: str) -> dict[str, Any]:
    """Questionnaire (SDC IG) — template del intake adaptativo.

    Contiene los items representativos (no las 50 features) e incluye
    ``enableWhen`` para dos branches de ejemplo (Dix-Hallpike si
    trigger=positional_head; focal_signs si onset=sudden).
    """
    qid = _id_for(case_id, "Questionnaire")
    items: list[dict[str, Any]] = []
    for meta in _FEATURE_TEMPLATE:
        item: dict[str, Any] = {
            "linkId": meta["linkId"],
            "text": meta["text"],
            "type": meta["type"],
        }
        if meta.get("repeats"):
            item["repeats"] = True
        if "system" in meta:
            item["answerOption"] = [
                {"valueCoding": _coding(meta["system"], code)}
                for code in meta["options"]
            ]
        if "enableWhen" in meta:
            item["enableWhen"] = meta["enableWhen"]
            item["enableBehavior"] = meta.get("enableBehavior", "all")
        items.append(item)

    return _resource(
        "Questionnaire",
        id=qid,
        meta={"profile": [_PROFILE_QUESTIONNAIRE]},
        url=_QUESTIONNAIRE_URL,
        version=_QUESTIONNAIRE_VERSION,
        name="VertigoDxIntakeQuestionnaire",
        title="Cuestionario clínico VertigoDx (intake)",
        status="draft",
        subjectType=["Patient"],
        date=_QUESTIONNAIRE_DATE,
        publisher="Clinibrium",
        description=(
            "Cuestionario estructurado desidentificado del flujo VertigoDx. "
            "Branching con enableWhen (SDC IG) para dos paths: "
            "Dix-Hallpike (si trigger=positional_head) y focal_signs "
            "(si onset=sudden)."
        ),
        item=items,
    )


def _build_questionnaire_response(
    case_id: str,
    features: CaseFeatures,
    subject_urn: str,
    now: datetime,
) -> dict[str, Any]:
    """QuestionnaireResponse — una respuesta por feature con valor.

    Itera por TODOS los fields del modelo ``CaseFeatures``; los
    presentes en el template usan su ``linkId``, el resto recibe
    ``linkId`` numérico correlativo (11+).
    """
    qid = _id_for(case_id, "Questionnaire")
    qrid = _id_for(case_id, "QuestionnaireResponse")
    items: list[dict[str, Any]] = []

    next_extra_link = 11

    for fname in CaseFeatures.model_fields:
        meta = _FEATURE_TEMPLATE_BY_FIELD.get(fname)
        value = getattr(features, fname)

        if meta is not None:
            link_id = meta["linkId"]
            text = meta["text"]
        else:
            link_id = str(next_extra_link)
            next_extra_link += 1
            text = fname

        if isinstance(value, Enum):
            answer: dict[str, Any] = {"valueCoding": _coding(None, code=value.value)}
        elif isinstance(value, set):
            if not value:
                continue  # set vacío → no responder
            if all(isinstance(v, Enum) for v in value):
                answer = {
                    "valueCoding": [
                        _coding(None, code=v.value) for v in value
                    ]
                }
            else:
                continue
        elif isinstance(value, bool):
            answer = {"valueBoolean": value}
        elif isinstance(value, int):
            answer = {"valueInteger": value}
        elif isinstance(value, float):
            answer = {"valueDecimal": value}
        elif value is None:
            continue  # feature opcional sin valor → no responder
        else:
            answer = {"valueString": str(value)}

        items.append(
            {
                "linkId": link_id,
                "text": text,
                "answer": [answer],
            }
        )

    return _resource(
        "QuestionnaireResponse",
        id=qrid,
        meta={
            "profile": [_PROFILE_QUESTIONNAIRE_RESPONSE],
        },
        questionnaire=_urn(qid),
        status="completed",
        subject={"reference": subject_urn},
        authored=_datetime_iso(now),
        item=items,
        extension=[
            {
                "url": _EXT_QUESTIONNAIRE_VERSION,
                "valueString": _QUESTIONNAIRE_VERSION,
            }
        ],
    )


def _build_observations(
    case_id: str,
    features: CaseFeatures,
    subject_urn: str,
    now: datetime,
) -> list[dict[str, Any]]:
    """Observations para variables clínicas clave.

    Solo emitimos Observations para features con valor; los códigos
    SNOMED CT son **placeholders** marcados ``TODO(clinical)``.
    """
    out: list[dict[str, Any]] = []

    def _emit_quantity(
        code: str,
        display: str,
        value: float,
        unit: str,
        ucum_code: str,
        suffix: str,
    ) -> None:
        obs_id = _id_for(case_id, "Observation", suffix)
        out.append(
            _resource(
                "Observation",
                id=obs_id,
                status="final",
                category=[
                    {
                        "coding": [
                            _coding(_CS_OBSERVATION_CATEGORY, "exam", _DISPLAY_EXAM)
                        ]
                    }
                ],
                code={"coding": [_coding(_CS_SNOMED, code, display)]},
                subject={"reference": subject_urn},
                effectiveDateTime=_datetime_iso(now),
                valueQuantity={
                    "value": value,
                    "unit": unit,
                    "system": _CS_UNITSOFMEASURE,
                    "code": ucum_code,
                },
            )
        )

    def _emit_codeable(
        code: str,
        display: str,
        coding_system: str,
        coding_code: str,
        suffix: str,
    ) -> None:
        obs_id = _id_for(case_id, "Observation", suffix)
        out.append(
            _resource(
                "Observation",
                id=obs_id,
                status="final",
                category=[
                    {
                        "coding": [
                            _coding(
                                _CS_OBSERVATION_CATEGORY, "exam", _DISPLAY_EXAM
                            )
                        ]
                    }
                ],
                code={"coding": [_coding(_CS_SNOMED, code, display)]},
                subject={"reference": subject_urn},
                effectiveDateTime=_datetime_iso(now),
                valueCodeableConcept={
                    "coding": [_coding(coding_system, coding_code)]
                },
            )
        )

    # Nistagmo (dirección)
    if features.nystagmus_direction != NystagmusDirection.none:
        _emit_codeable(
            _SNOMED_NYSTAGMUS_OBSERVATION,
            "Nystagmus (observation)",
            _CS_NYSTAGMUS_DIRECTION,
            features.nystagmus_direction.value,
            "nystagmus_direction",
        )

    # Latencia del nistagmo
    if features.nystagmus_latency_s is not None:
        _emit_quantity(
            _SNOMED_NYSTAGMUS_LATENCY_S,
            "Latency of nystagmus (s)",
            features.nystagmus_latency_s,
            _UCUM_SECOND,
            _UCUM_SECOND,
            "nystagmus_latency_s",
        )

    # Duración del nistagmo
    if features.nystagmus_duration_s is not None:
        _emit_quantity(
            _SNOMED_NYSTAGMUS_DURATION_S,
            "Duration of nystagmus (s)",
            features.nystagmus_duration_s,
            _UCUM_SECOND,
            _UCUM_SECOND,
            "nystagmus_duration_s",
        )

    # Dix-Hallpike (si fue hecho)
    if features.dix_hallpike != DixHallpikeResult.not_done:
        _emit_codeable(
            _SNOMED_DIX_HALLPIKE,
            "Dix-Hallpike maneuver (result)",
            _CS_DIX_HALLPIKE,
            features.dix_hallpike.value,
            "dix_hallpike",
        )

    return out


def _build_detected_issue(
    case_id: str,
    hit: RedFlagHit,
    hit_index: int,
    subject_urn: str,
    audit_urn: str,
    now: datetime,
) -> dict[str, Any]:
    """DetectedIssue por cada red flag disparada (``severity=high``)."""
    did = _id_for(case_id, "DetectedIssue", str(hit_index))
    forced_actions = [a.value for a in hit.forced_actions]
    return _resource(
        "DetectedIssue",
        id=did,
        status="final",
        severity="high",
        code={
            "coding": [
                _coding(_CS_RED_FLAG, hit.id, f"RedFlag {hit.id}: {hit.label}")
            ]
        },
        patient={"reference": subject_urn},
        identifiedDateTime=_datetime_iso(now),
        implicated=[{"reference": audit_urn}],
        detail=(
            f"id={hit.id}; label={hit.label}; forced_actions={forced_actions}"
        ),
    )


def _build_flag(
    case_id: str,
    hit: RedFlagHit,
    hit_index: int,
    subject_urn: str,
) -> dict[str, Any]:
    """Flag por cada red flag disparada (alerta clínica)."""
    fid = _id_for(case_id, "Flag", str(hit_index))
    forced_actions = [a.value for a in hit.forced_actions]
    return _resource(
        "Flag",
        id=fid,
        status="active",
        category=[
            {
                "coding": [
                    _coding(_CS_FLAG_CATEGORY, "clinical", _DISPLAY_CLINICAL)
                ]
            }
        ],
        code={
            "coding": [
                _coding(_CS_RED_FLAG, hit.id, f"RedFlag {hit.id}: {hit.label}")
            ]
        },
        subject={"reference": subject_urn},
        text=(
            f"Red flag {hit.id}: {hit.label}. "
            f"Forced actions: {forced_actions}"
        ),
    )


def _build_clinical_impression(
    case_id: str,
    result: PipelineResult,
    subject_urn: str,
    now: datetime,
) -> dict[str, Any]:
    """ClinicalImpression — el resultado: candidatos, urgency, razonamiento.

    Si ``result.reasoning`` es None, agrega una nota explícita
    "razonador no disponible (degradado)" y la extensión
    ``extension-reasoner-degraded`` para que la auditoría lo vea.
    """
    ciid = _id_for(case_id, "ClinicalImpression")
    findings: list[dict[str, Any]] = [
        _finding_for_candidate(c) for c in result.differential.candidates
    ]

    notes: list[dict[str, Any]] = []
    extensions: list[dict[str, Any]] = []

    if result.reasoning is None:
        notes.append(
            _annotation(
                "Razonador no disponible (degradado): la explicación y "
                "conciliación de Claude no fueron emitidas para este caso; "
                "el bundle refleja solo las capas deterministas "
                "(RedFlagEngine, DifferentialEngine, Rails).",
            )
        )
        extensions.append(
            {
                "url": _EXT_REASONER_DEGRADED,
                "valueBoolean": True,
            }
        )
    else:
        if result.reasoning.explanation:
            notes.append(_annotation(result.reasoning.explanation))
        if result.reasoning.reconciliation:
            notes.append(
                _annotation(
                    f"Reconciliation: {result.reasoning.reconciliation}"
                )
            )
        if result.reasoning.suggested_next_steps:
            notes.append(
                _annotation(
                    "Next steps: "
                    + "; ".join(result.reasoning.suggested_next_steps)
                )
            )

    if result.applied_rails:
        extensions.append(
            {
                "url": _EXT_RAIL_TRIGGERED,
                "valueString": ",".join(result.applied_rails),
            }
        )

    forced_actions = sorted(
        [a.value for a in result.forced_actions]
    )
    return _resource(
        "ClinicalImpression",
        id=ciid,
        meta={"profile": [_PROFILE_CLINICAL_IMPRESSION]},
        status="completed",
        code={
            "coding": [
                _coding(
                    _CS_SNOMED,
                    _SNOMED_CLINICAL_IMPRESSION,
                    _DISPLAY_CLINICAL_IMPRESSION,
                )
            ]
        },
        subject={"reference": subject_urn},
        effectiveDateTime=_datetime_iso(now),
        date=_datetime_iso(now),
        summary=(
            f"Urgencia final: {result.urgency.value}. "
            f"red_flag_activa={result.red_flag.red_flag_activa}. "
            f"forced_actions={forced_actions}. "
            f"applied_rails={result.applied_rails}."
        ),
        finding=findings,
        note=notes,
        extension=extensions if extensions else None,
    )


def _finding_for_candidate(c: DifferentialCandidate) -> dict[str, Any]:
    """Mapping ``DifferentialCandidate`` → ``ClinicalImpression.finding[]``."""
    rule_ids = f"; rule_ids={c.rule_ids}" if c.rule_ids else ""
    return {
        "itemCodeableConcept": {
            "coding": [_coding(_CS_DIAGNOSIS, c.diagnosis.value)]
        },
        "basis": f"score={c.score:.3f}{rule_ids}",
    }


def _build_audit_event(
    case_id: str,
    audit: AuditEvent,
    subject_urn: str,
) -> dict[str, Any]:
    """AuditEvent (perfil CL Core ``Auditoria``) — quién/qué/cuándo.

    El hash de features, la urgencia, ``red_flag_activa``, ``model_used``
    y el ``reasoner_status`` se adjuntan como ``entity.detail[]``.
    """
    aaid = _id_for(case_id, "AuditEvent")
    reasoner_status = "ok" if audit.model_used is not None else "degraded"

    agent_type_coding = _coding(
        _CS_AUDIT_AGENT_TYPE, "humanuser", _DISPLAY_HUMANUSER
    )

    return _resource(
        "AuditEvent",
        id=aaid,
        meta={"profile": [_CLCORE_AUDIT_PROFILE]},
        type={
            "system": _CS_AUDIT_EVENT_TYPE,
            "code": audit.event_type,
            "display": audit.event_type,
        },
        subtype=[
            {"system": _CS_AUDIT_EVENT_TYPE, "code": audit.event_type}
        ],
        action=_AUDIT_ACTION_EXECUTE,
        recorded=_datetime_iso(audit.occurred_at),
        outcome=_AUDIT_OUTCOME_SUCCESS,
        outcomeDesc=audit.outcome_summary,
        agent=[
            {
                "type": {"coding": [agent_type_coding]},
                "who": {"reference": subject_urn},
                "requestor": True,
            }
        ],
        source={
            "site": "Clinibrium backend (VertigoDx)",
            "observer": {"reference": subject_urn},
        },
        entity=[
            {
                "type": {
                    "system": _CS_AUDIT_EVENT_TYPE,
                    "code": "case",
                    "display": _DISPLAY_CASE,
                },
                "role": {
                    "system": _CS_AUDIT_EVENT_TYPE,
                    "code": "data",
                    "display": _DISPLAY_DATA,
                },
                # ``what`` es un Reference(Resource) — NO referenciamos un
                # "PipelineCase" inexistente (no es recurso FHIR). Usamos
                # ``description`` para describir el caso y los details
                # estructurados (hash, urgency, etc.) en ``detail[]``.
                "description": f"case_id={case_id}",
                "detail": [
                    {
                        "type": "input_features_hash",
                        "valueString": audit.input_features_hash,
                    },
                    {
                        "type": "urgency",
                        "valueString": audit.urgency.value,
                    },
                    {
                        "type": "red_flag_activa",
                        "valueString": str(audit.red_flag_activa).lower(),
                    },
                    {
                        "type": "model_used",
                        "valueString": audit.model_used or "n/a",
                    },
                    {
                        "type": "reasoner_status",
                        "valueString": reasoner_status,
                    },
                ],
            }
        ],
    )


# ---------------------------------------------------------------------------
# Bundle — orquestación
# ---------------------------------------------------------------------------


def to_bundle(
    result: PipelineResult,
    features: CaseFeatures,
    audit: AuditEvent,
) -> dict[str, Any]:
    """Produce un ``Bundle`` FHIR R4 (tipo ``collection``) con el artefacto auditable.

    Recursos incluidos:
      - ``Patient``              — placeholder sin PII (anchor de referencias).
      - ``Questionnaire``        — template SDC del intake adaptativo.
      - ``QuestionnaireResponse``— respuestas estructuradas de las CaseFeatures.
      - ``Observation`` × N      — variables clínicas clave (1 por feature relevante).
      - ``ClinicalImpression``   — diferencial + urgencia + razonamiento de Claude.
      - ``DetectedIssue`` × N    — uno por cada ``result.red_flag.hits`` (severity=high).
      - ``Flag`` × N             — uno por cada ``result.red_flag.hits``.
      - ``AuditEvent``           — evento inmutable de la invocación (perfil CL Auditoria).
      - ``Bundle``               — el contenedor tipo ``collection``.

    Determinismo: todos los IDs se derivan con ``uuid5(namespace, ...)``
    a partir de ``case_id``; el mismo input produce el mismo bundle.
    El ``timestamp`` del bundle se toma de ``audit.occurred_at`` para
    garantizar determinismo entre llamadas.  Sin I/O, sin red, sin
    estado mutable.
    """
    case_id = result.case_id
    bundle_id = _id_for(case_id, "Bundle")
    # Timestamp determinista: derivado del AuditEvent (no de ``now``),
    # para que ``to_bundle`` sea puro entre llamadas.
    now = audit.occurred_at

    # Subject (Patient placeholder)
    patient, subject_urn = _build_minimal_patient(case_id)

    # AuditEvent (necesitamos su urn para las referencias en DetectedIssue)
    audit_event = _build_audit_event(case_id, audit, subject_urn)
    audit_urn = _urn(audit_event["id"])

    # Questionnaires
    questionnaire = _build_questionnaire(case_id)
    questionnaire_response = _build_questionnaire_response(
        case_id, features, subject_urn, now
    )

    # Observations
    observations = _build_observations(case_id, features, subject_urn, now)

    # ClinicalImpression
    clinical_impression = _build_clinical_impression(
        case_id, result, subject_urn, now
    )

    # DetectedIssue + Flag por cada red flag
    red_flag_resources: list[dict[str, Any]] = []
    for i, hit in enumerate(result.red_flag.hits):
        red_flag_resources.append(
            _build_detected_issue(
                case_id, hit, i, subject_urn, audit_urn, now
            )
        )
        red_flag_resources.append(
            _build_flag(case_id, hit, i, subject_urn)
        )

    # entries del bundle
    entries: list[dict[str, Any]] = []
    for resource in (
        [patient, questionnaire, questionnaire_response, audit_event]
        + observations
        + [clinical_impression]
        + red_flag_resources
    ):
        entries.append(
            {"fullUrl": _urn(resource["id"]), "resource": resource}
        )

    return _resource(
        "Bundle",
        id=bundle_id,
        type="collection",
        timestamp=_datetime_iso(now),
        entry=entries,
    )
