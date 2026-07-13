"""Maps a ``PipelineResult`` to a FHIR R4 ``Bundle`` (output format, AD-9).

Pure function — no I/O, no network. Given:

  - ``PipelineResult`` (pipeline output)
  - ``CaseFeatures``   (structured input, no PII)
  - ``AuditEvent``     (immutable event of the invocation, INV-4)

produces a JSON-serializable ``dict`` with a ``collection``-type Bundle
grouping the auditable FHIR R4 resources: ``Questionnaire`` (adaptive
intake, SDC IG), ``QuestionnaireResponse`` (de-identified answers),
``Observation`` (key clinical variables), ``DetectedIssue`` + ``Flag``
(one per fired red flag), ``ClinicalImpression`` (differential +
reasoning) and ``AuditEvent`` (CL Core ``Auditoria`` profile).

Module graph (hard rule of the Clinibrium module map):
    fhir → contracts   ✓
    fhir → stdlib      ✓ (uuid, datetime, enum)

FORBIDDEN imports: ``engines``, ``reasoner``, ``rails``, ``orchestrator``,
``ml_client``, ``api``, ``storage``, ``audit``, ``grounding``, ``config``.

CL Core IG 1.9.3 (R4) profiles:
    - ``AuditEvent`` (CL ``Auditoria``) — native CL Core profile;
      referenced in ``meta.profile``.

Clinibrium's own profiles (CL Core does NOT provide Questionnaire /
QuestionnaireResponse / ClinicalImpression — registered as an
extension):
    - ``cl-questionnaire``
    - ``cl-questionnaire-response``
    - ``cl-clinical-impression``

Own extensions (canonical URLs declared in this module):
    - ``extension-questionnaire-version``: questionnaire version
      attached to the ``QuestionnaireResponse``.
    - ``extension-reasoner-degraded``: flag marking the
      ``ClinicalImpression`` when the reasoner (Claude) was not
      available.
    - ``extension-rail-triggered``: list of rail-ids (R-INV1, R-EPLEY-D,
      R-E2, R-DIVERGENCIA) that fired, in ``ClinicalImpression`` and
      ``AuditEvent.entity.detail``.

Deterministic IDs (stable tests):
    ``uuid5(_URN_NAMESPACE, f"{case_id}/{kind}/{key}")``. Same
    ``case_id`` + same ``kind`` + same ``key`` ⇒ same id.
    Random ``uuid4`` is NOT used.

Internal references:
    ``fullUrl = "urn:uuid:{id}"``. References between resources
    (e.g. ``QuestionnaireResponse.questionnaire``) use the same
    scheme. Every reference in the bundle resolves to a resource
    present in the same bundle.

SNOMED CT / LOINC codes:
    The ``Observation.code`` and ``ClinicalImpression.code`` codes are
    **placeholders** marked ``TODO(clinical)`` — confirm with the
    superspecialist before production. SNOMED codes as rules/facts are
    not copyrightable; the canonical codes do need to be verified.
"""
from __future__ import annotations

import hashlib
import json as _json
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
# Constants — namespaces, URLs, codes
# ---------------------------------------------------------------------------

_URN_NAMESPACE: uuid.UUID = uuid.UUID("8d3a3b3e-7c14-5b9d-9b3a-3b3e7c145b9d")
"""Fixed namespace for ``uuid5``; guarantees deterministic ids across runs."""

# CL Core IG 1.9.3 (R4) — CL ``Auditoria`` profile (native profile).
_CLCORE_AUDIT_PROFILE: str = "http://hl7.cl/fhir/ig/clcore/StructureDefinition/Auditoria"
_CLCORE_PATIENT_PROFILE: str = (
    "http://hl7.cl/fhir/ig/clcore/StructureDefinition/CorePacienteCl"
)

# Clinibrium profiles (CL Core does not provide Questionnaire /
# QuestionnaireResponse / ClinicalImpression → we declare our own).
_CLINIBRIUM_BASE: str = "http://clinibrium.cl/fhir/StructureDefinition"
_PROFILE_QUESTIONNAIRE: str = f"{_CLINIBRIUM_BASE}/cl-questionnaire"
_PROFILE_QUESTIONNAIRE_RESPONSE: str = f"{_CLINIBRIUM_BASE}/cl-questionnaire-response"
_PROFILE_CLINICAL_IMPRESSION: str = f"{_CLINIBRIUM_BASE}/cl-clinical-impression"

# Own extensions (canonical URLs).
_EXT_QUESTIONNAIRE_VERSION: str = (
    f"{_CLINIBRIUM_BASE}/extension-questionnaire-version"
)
_EXT_REASONER_DEGRADED: str = f"{_CLINIBRIUM_BASE}/extension-reasoner-degraded"
_EXT_RAIL_TRIGGERED: str = f"{_CLINIBRIUM_BASE}/extension-rail-triggered"

# Local code systems (Clinibrium / VertigoDx domain vocabulary).
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

# Standard coding systems (placeholders — see TODO(clinical) at the end of the module).
_CS_SNOMED: str = "http://snomed.info/sct"

# TODO(clinical): confirm the canonical SNOMED CT codes with the superspecialist.
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

# Questionnaire version (local constant — bump when the template changes).
_QUESTIONNAIRE_VERSION: str = "0.1.0"
_QUESTIONNAIRE_URL: str = "urn:clinibrium:questionnaire:vertigodx-intake"
_QUESTIONNAIRE_DATE: str = "2026-07-10"

# Short display strings.
_DISPLAY_EXAM: str = "Exam"
_DISPLAY_CLINICAL: str = "Clinical"
_DISPLAY_HUMANUSER: str = "Human User"
_DISPLAY_CASE: str = "Case"
_DISPLAY_DATA: str = "Data"
_DISPLAY_CLINICAL_IMPRESSION: str = "Clinical impression"


# ---------------------------------------------------------------------------
# Feature → Questionnaire item mapping (template, NOT all 50 features)
# ---------------------------------------------------------------------------

# Each template item declares: linkId, text, type, code system, and the
# options (when it is a choice). This table is the source of truth for
# building the Questionnaire (template) AND the QuestionnaireResponse
# (instance) — being representative intake fields it covers the main
# structured features; the QuestionnaireResponse additionally includes
# items for features not modeled in the template (linkId 11+).
# Item "text" values are clinician-facing (serialized into the bundle) — Spanish.

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
        # enableWhen: only shown if trigger=positional_head
        # (Dix-Hallpike only applies to positional suspicion).
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
        # enableWhen: focal signs are only collected if onset=sudden
        # (acute vestibular syndrome).
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
# Generic helpers
# ---------------------------------------------------------------------------


def _resource(resource_type: str, **fields: Any) -> dict[str, Any]:
    """Builds a FHIR R4 resource as a plain dict.

    Pattern: ``{"resourceType": <type>, ...fields}``. ``None`` is filtered
    out so empty fields are not emitted.  Fields whose value is a dict /
    list pass through as-is.
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
    """FHIR R4 Coding as a dict."""
    out: dict[str, Any] = {"code": code}
    if system is not None:
        out["system"] = system
    if display is not None:
        out["display"] = display
    return out


def _id_for(case_id: str, kind: str, key: str = "") -> str:
    """Deterministic ID for a bundle resource.

    ``uuid5(_URN_NAMESPACE, f"{case_id}/{kind}/{key}")``. An empty ``key``
    still produces a stable id per ``(case_id, kind)``.
    """
    return str(uuid.uuid5(_URN_NAMESPACE, f"{case_id}/{kind}/{key}"))


def _urn(resource_id: str) -> str:
    """fullUrl / reference of a bundle resource (``urn:uuid:{id}``)."""
    return f"urn:uuid:{resource_id}"


def _datetime_iso(dt: datetime) -> str:
    """Serializes a datetime to ISO 8601 with tz."""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.isoformat()


def _annotation(text: str, time: datetime | None = None) -> dict[str, Any]:
    """FHIR R4 Annotation: ``{text, time?}``."""
    out: dict[str, Any] = {"text": text}
    if time is not None:
        out["time"] = _datetime_iso(time)
    return out


# ---------------------------------------------------------------------------
# Per-resource builders
# ---------------------------------------------------------------------------


def _build_minimal_patient(case_id: str) -> tuple[dict[str, Any], str]:
    """Patient placeholder (no PII) so that references resolve.

    The privacy policy (INV-2 / AD-7) forbids any PII or external
    identifier; the Patient is only a reference anchor within the
    bundle. It does NOT include name, RUT, DOB or address.
    """
    pid = _id_for(case_id, "Patient")
    resource = _resource(
        "Patient",
        id=pid,
        meta={"profile": [_CLCORE_PATIENT_PROFILE]},
        # Mark the Patient as not-real (placeholder for references).
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
    """Questionnaire (SDC IG) — adaptive intake template.

    Contains the representative items (not the 50 features) and includes
    ``enableWhen`` for two example branches (Dix-Hallpike if
    trigger=positional_head; focal_signs if onset=sudden).
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
    """QuestionnaireResponse — one answer per feature with a value.

    Iterates over ALL fields of the ``CaseFeatures`` model; those
    present in the template use their ``linkId``, the rest get a
    sequential numeric ``linkId`` (11+).
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
                continue  # empty set → no answer
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
            continue  # optional feature without value → no answer
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
    """Observations for key clinical variables.

    We only emit Observations for features with a value; the SNOMED CT
    codes are **placeholders** marked ``TODO(clinical)``.
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

    # Nystagmus (direction)
    if features.nystagmus_direction != NystagmusDirection.none:
        _emit_codeable(
            _SNOMED_NYSTAGMUS_OBSERVATION,
            "Nystagmus (observation)",
            _CS_NYSTAGMUS_DIRECTION,
            features.nystagmus_direction.value,
            "nystagmus_direction",
        )

    # Nystagmus latency
    if features.nystagmus_latency_s is not None:
        _emit_quantity(
            _SNOMED_NYSTAGMUS_LATENCY_S,
            "Latency of nystagmus (s)",
            features.nystagmus_latency_s,
            _UCUM_SECOND,
            _UCUM_SECOND,
            "nystagmus_latency_s",
        )

    # Nystagmus duration
    if features.nystagmus_duration_s is not None:
        _emit_quantity(
            _SNOMED_NYSTAGMUS_DURATION_S,
            "Duration of nystagmus (s)",
            features.nystagmus_duration_s,
            _UCUM_SECOND,
            _UCUM_SECOND,
            "nystagmus_duration_s",
        )

    # Dix-Hallpike (if performed)
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
    """DetectedIssue for each fired red flag (``severity=high``)."""
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
    """Flag for each fired red flag (clinical alert)."""
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
    output_lang: str | None = None,
) -> dict[str, Any]:
    """ClinicalImpression — the result: candidates, urgency, reasoning.

    If ``result.reasoning`` is None, adds an explicit
    "reasoner unavailable (degraded)" note (Spanish, clinician-facing)
    and the ``extension-reasoner-degraded`` extension so the audit
    can see it.

    ``output_lang`` (AD-19 precision, codex-audit-4 Alta 1): the reasoner
    prose embedded in ``note`` keeps the language it was REQUESTED in.
    The deterministic content (summary, findings, extensions) is always
    canonical Spanish. When the prose was requested in English, the
    resource is tagged with FHIR ``language: "en"`` so the artifact is
    honest about it; the Spanish/default path adds NO key and stays
    byte-identical to the recorded demo.
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
        language=(
            "en" if output_lang == "en" and result.reasoning is not None else None
        ),
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
    """Maps ``DifferentialCandidate`` → ``ClinicalImpression.finding[]``."""
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
    """AuditEvent (CL Core ``Auditoria`` profile) — who/what/when.

    The features hash, urgency, ``red_flag_activa``, ``model_used``
    and ``reasoner_status`` are attached as ``entity.detail[]``.
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
                # ``what`` is a Reference(Resource) — we do NOT reference a
                # nonexistent "PipelineCase" (not a FHIR resource). We use
                # ``description`` to describe the case and the structured
                # details (hash, urgency, etc.) go in ``detail[]``.
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
# Bundle — orchestration
# ---------------------------------------------------------------------------


def to_bundle(
    result: PipelineResult,
    features: CaseFeatures,
    audit: AuditEvent,
) -> dict[str, Any]:
    """Produces a FHIR R4 ``Bundle`` (type ``collection``) with the auditable artifact.

    Included resources:
      - ``Patient``              — placeholder without PII (reference anchor).
      - ``Questionnaire``        — SDC template of the adaptive intake.
      - ``QuestionnaireResponse``— structured answers from the CaseFeatures.
      - ``Observation`` × N      — key clinical variables (1 per relevant feature).
      - ``ClinicalImpression``   — differential + urgency + Claude's reasoning.
      - ``DetectedIssue`` × N    — one per ``result.red_flag.hits`` (severity=high).
      - ``Flag`` × N             — one per ``result.red_flag.hits``.
      - ``AuditEvent``           — immutable event of the invocation (CL Auditoria profile).
      - ``Bundle``               — the ``collection``-type container.

    Determinism: all IDs are derived with ``uuid5(namespace, ...)``
    from ``case_id``; the same input produces the same bundle.
    The bundle ``timestamp`` is taken from ``audit.occurred_at`` to
    guarantee determinism across calls.  No I/O, no network, no
    mutable state.
    """
    case_id = result.case_id
    bundle_id = _id_for(case_id, "Bundle")
    # Deterministic timestamp: derived from the AuditEvent (not from ``now``),
    # so that ``to_bundle`` is pure across calls.
    now = audit.occurred_at

    # Subject (Patient placeholder)
    patient, subject_urn = _build_minimal_patient(case_id)

    # AuditEvent (we need its urn for the references in DetectedIssue)
    audit_event = _build_audit_event(case_id, audit, subject_urn)
    audit_urn = _urn(audit_event["id"])

    # Questionnaires
    questionnaire = _build_questionnaire(case_id)
    questionnaire_response = _build_questionnaire_response(
        case_id, features, subject_urn, now
    )

    # Observations
    observations = _build_observations(case_id, features, subject_urn, now)

    # ClinicalImpression. The reasoner prose in its notes keeps the language
    # it was requested in (audit.output_lang); the resource is tagged with
    # FHIR `language` when that was English. Spanish/None adds no key, so the
    # default bundle stays byte-identical (AD-19 precision).
    clinical_impression = _build_clinical_impression(
        case_id, result, subject_urn, now, output_lang=audit.output_lang
    )

    # DetectedIssue + Flag for each red flag
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

    # bundle entries
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


def _js_number(n: float) -> str:
    """Formats a number exactly like ECMAScript's ``JSON.stringify``.

    Needed so that the frontend (which recomputes the hash with
    ``JSON.stringify``) gets identical BYTES. The key difference with
    Python's ``json.dumps``: JS emits an integral float (``5.0``) as
    ``"5"`` (not ``"5.0"``), and ``NaN``/``Infinity`` as ``null``.
    """
    if n != n or n in (float("inf"), float("-inf")):  # NaN / ±Inf → null (JS)
        return "null"
    if n == int(n) and abs(n) < 1e21:
        return str(int(n))
    # For fractional values, Python's ``repr`` matches ECMAScript's
    # shortest-round-trip in the demo's value range.
    return repr(n)


def _canonical_json(obj: Any) -> str:
    """Canonical JSON byte-identical to the frontend's ``jsonCanonical``.

    Rules (RFC 8785 / JCS, subset used by the bundle):
      - object keys sorted (Unicode code point == UTF-16 for ASCII);
      - no whitespace between tokens;
      - strings with the same escaping as ``JSON.stringify`` (== ``json.dumps``
        with ``ensure_ascii=False``);
      - numbers in ECMAScript format (see ``_js_number``).
    """
    if obj is None:
        return "null"
    if obj is True:
        return "true"
    if obj is False:
        return "false"
    if isinstance(obj, str):
        return _json.dumps(obj, ensure_ascii=False)
    if isinstance(obj, bool):  # (already covered above, defensive)
        return "true" if obj else "false"
    if isinstance(obj, int):
        return str(obj)
    if isinstance(obj, float):
        return _js_number(obj)
    if isinstance(obj, (list, tuple)):
        return "[" + ",".join(_canonical_json(v) for v in obj) + "]"
    if isinstance(obj, dict):
        parts = [
            _json.dumps(str(k), ensure_ascii=False) + ":" + _canonical_json(v)
            for k, v in sorted(obj.items(), key=lambda kv: str(kv[0]))
        ]
        return "{" + ",".join(parts) + "}"
    # Fallback (e.g. a datetime not serialized upstream).
    return _json.dumps(str(obj), ensure_ascii=False)


def bundle_sha256(bundle: dict[str, Any]) -> str:
    """SHA-256 hex of the Bundle's canonical JSON (tamper-evident).

    Same bundle → same hash; alteration → different hash.
    The frontend recomputes the hash of the received bundle with the SAME
    canonicalization (sorted keys, no whitespace, ECMAScript number
    format) and compares it against this value to verify integrity
    independently (✓ intact / ✗ altered).
    """
    canonical = _canonical_json(bundle)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
