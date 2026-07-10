"""Tests del módulo ``fhir`` (artefacto auditable FHIR R4).

Cubre los criterios de aceptación de T10:

  1. ``to_bundle(...)`` con un PipelineResult de BPPV benigno → Bundle
     válido con Questionnaire, QuestionnaireResponse, Observation,
     ClinicalImpression, AuditEvent. SIN DetectedIssue/Flag si no hay
     red flags.

  2. ``to_bundle(...)`` con red flag activa (urgency=inmediata, hits) →
     DetectedIssue (severity=high) + Flag por cada hit, y
     ClinicalImpression con urgency inmediata.

  3. ``reasoning=None`` → ClinicalImpression nota "degradado", no
     rompe. La extensión ``extension-reasoner-degraded`` se emite.

  4. El Bundle serializa a JSON (``json.dumps``) y las referencias
     entre recursos resuelven: todo recurso referenciado está en el
     bundle (test automático de resolución de ``urn:uuid:``).

  5. Determinismo: mismo input → mismo bundle (IDs derivados del
     ``case_id``).

  6. Negativo: ``fhir`` SOLO importa ``contracts`` (regla dura del
     mapa — AST).

  7. Negativo: ``to_bundle`` es pura (no muta input, no hace I/O).
"""
from __future__ import annotations

import ast
import json
import re
from datetime import datetime, timezone
from pathlib import Path

from clinibrium.contracts import (
    ActorType,
    AuditEvent,
    CaseFeatures,
    Diagnosis,
    DifferentialCandidate,
    DifferentialResult,
    DixHallpikeResult,
    ForcedAction,
    HeadImpulse,
    HearingLoss,
    NystagmusDirection,
    Onset,
    PipelineResult,
    ReasonerOutput,
    RedFlagHit,
    RedFlagResult,
    SymptomDuration,
    Trigger,
    Urgency,
)
from clinibrium.fhir import to_bundle

# =========================================================================
# Helpers
# =========================================================================


def _audit(
    *,
    id: str = "evt-1",
    urgency: Urgency = Urgency.ambulatoria,
    red_flag_activa: bool = False,
    model_used: str | None = "claude-opus-4-8",
    forced_actions: list[ForcedAction] | None = None,
    outcome_summary: str = "BPPV posterior probable; sin red flags.",
) -> AuditEvent:
    return AuditEvent(
        id=id,
        occurred_at=datetime(2026, 7, 10, 12, 0, 0, tzinfo=timezone.utc),
        event_type="pipeline_evaluation",
        actor=ActorType.system,
        model_used=model_used,
        input_features_hash="sha256:" + "a" * 64,
        urgency=urgency,
        forced_actions=forced_actions or [],
        red_flag_activa=red_flag_activa,
        outcome_summary=outcome_summary,
    )


def _features_bppv() -> CaseFeatures:
    """Caso BPPV benigno: posicional, torsional, fatigable, <60s."""
    return CaseFeatures(
        duration=SymptomDuration.seconds,
        onset=Onset.sudden,
        trigger=Trigger.positional_head,
        timing_pattern=None,
        nystagmus_direction=NystagmusDirection.torsional_pure,
        nystagmus_direction_changing_gaze=False,
        nystagmus_latency_s=2.0,
        nystagmus_duration_s=20.0,
        nystagmus_fatigable=True,
        nystagmus_suppressed_by_fixation=True,
        head_impulse=HeadImpulse.normal,
        skew_deviation=False,
        hearing_loss=HearingLoss.none,
        tinnitus=False,
        aural_fullness=False,
        focal_signs=set(),
        truncal_ataxia_severe=False,
        headache_neck_pain_sudden_severe=False,
        migrainous_features=False,
        age_years=60,
        vascular_risk_factors=set(),
        fever=False,
        neck_stiffness=False,
        altered_consciousness=False,
        presyncope_syncope=False,
        palpitations=False,
        chest_pain=False,
        otitis_mastoiditis=False,
        recent_head_neck_trauma=False,
        cervical_pathology=False,
        known_carotid_vertebrobasilar_disease=False,
        cardiovascular_instability=False,
        dix_hallpike=DixHallpikeResult.right_positive,
        torsion_confirmed_by_clinician=True,
        episode_count=1,
        episode_duration=SymptomDuration.seconds,
        worsening_during_flow=False,
    )


def _result_bppv(
    *, case_id: str = "case-bppv-001", reasoning: ReasonerOutput | None = None
) -> PipelineResult:
    return PipelineResult(
        case_id=case_id,
        urgency=Urgency.ambulatoria,
        red_flag=RedFlagResult(red_flag_activa=False),
        differential=DifferentialResult(
            candidates=[
                DifferentialCandidate(
                    diagnosis=Diagnosis.bppv_posterior,
                    score=0.92,
                    rule_ids=["R-BPPV-1", "R-BPPV-2"],
                ),
                DifferentialCandidate(
                    diagnosis=Diagnosis.bppv_horizontal, score=0.10
                ),
            ]
        ),
        forced_actions=set(),
        applied_rails=[],
        reasoning=reasoning,
    )


def _result_red_flag(
    *, case_id: str = "case-avs-001", reasoning: ReasonerOutput | None = None
) -> PipelineResult:
    return PipelineResult(
        case_id=case_id,
        urgency=Urgency.inmediata,
        red_flag=RedFlagResult(
            red_flag_activa=True,
            hits=[
                RedFlagHit(
                    id="A1",
                    label="AVS con focal signs",
                    forced_actions=[
                        ForcedAction.DERIVAR_URGENTE,
                        ForcedAction.NO_BENIGNO,
                    ],
                    severity="high",
                ),
                RedFlagHit(
                    id="A8",
                    label="Headache + neck pain sudden severe",
                    forced_actions=[ForcedAction.ESCALAR],
                    severity="high",
                ),
            ],
            forced_actions={
                ForcedAction.DERIVAR_URGENTE,
                ForcedAction.NO_BENIGNO,
                ForcedAction.ESCALAR,
            },
        ),
        differential=DifferentialResult(
            candidates=[
                DifferentialCandidate(
                    diagnosis=Diagnosis.central_suspected,
                    score=0.55,
                    rule_ids=["R-CENTRAL-1"],
                ),
            ]
        ),
        forced_actions={
            ForcedAction.DERIVAR_URGENTE,
            ForcedAction.NO_BENIGNO,
            ForcedAction.ESCALAR,
        },
        applied_rails=["R-INV1"],
        reasoning=reasoning,
    )


def _resources_by_type(bundle: dict) -> dict[str, list[dict]]:
    """Agrupa los recursos del bundle por ``resourceType``."""
    out: dict[str, list[dict]] = {}
    for entry in bundle.get("entry", []):
        r = entry.get("resource", {})
        rt = r.get("resourceType")
        if rt is None:
            continue
        out.setdefault(rt, []).append(r)
    return out


def _collect_urn_refs(obj: object) -> set[str]:
    """Recorre un dict/list y devuelve todos los ``urn:uuid:...`` referenciados."""
    refs: set[str] = set()
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k == "reference" and isinstance(v, str):
                if v.startswith("urn:uuid:"):
                    refs.add(v)
            else:
                refs.update(_collect_urn_refs(v))
    elif isinstance(obj, list):
        for v in obj:
            refs.update(_collect_urn_refs(v))
    return refs


# =========================================================================
# 1. Caso benigno — BPPV sin red flags
# =========================================================================


class TestBundleBppvBenigno:
    """Bundle de un caso BPPV benigno: SIN DetectedIssue/Flag."""

    def test_bundle_tipo_collection(self) -> None:
        bundle = to_bundle(_result_bppv(), _features_bppv(), _audit())
        assert bundle["resourceType"] == "Bundle"
        assert bundle["type"] == "collection"
        assert "id" in bundle
        assert "timestamp" in bundle

    def test_bundle_serializa_a_json(self) -> None:
        bundle = to_bundle(_result_bppv(), _features_bppv(), _audit())
        s = json.dumps(bundle)
        # roundtrip
        again = json.loads(s)
        assert again["resourceType"] == "Bundle"
        assert again["id"] == bundle["id"]

    def test_recursos_esperados_presentes(self) -> None:
        bundle = to_bundle(_result_bppv(), _features_bppv(), _audit())
        grouped = _resources_by_type(bundle)
        # Recursos OBLIGATORIOS
        assert "Questionnaire" in grouped
        assert "QuestionnaireResponse" in grouped
        assert "ClinicalImpression" in grouped
        assert "AuditEvent" in grouped
        # Al menos 1 Observation (nistagmus_direction o duración)
        assert "Observation" in grouped
        assert len(grouped["Observation"]) >= 1
        # Patient placeholder (anchor de referencias)
        assert "Patient" in grouped

    def test_sin_detected_issue_sin_flag(self) -> None:
        bundle = to_bundle(_result_bppv(), _features_bppv(), _audit())
        grouped = _resources_by_type(bundle)
        assert "DetectedIssue" not in grouped
        assert "Flag" not in grouped

    def test_clinical_impression_bppv(self) -> None:
        bundle = to_bundle(_result_bppv(), _features_bppv(), _audit())
        ci = _resources_by_type(bundle)["ClinicalImpression"][0]
        assert ci["status"] == "completed"
        # summary menciona urgencia ambulatoria
        assert "ambulatoria" in ci["summary"]
        # findings: 2 candidatos
        assert len(ci["finding"]) == 2
        # score del primer candidato en el basis
        assert "score=0.920" in ci["finding"][0]["basis"]
        # coding del primer finding = bppv_posterior
        first_coding = ci["finding"][0]["itemCodeableConcept"]["coding"][0]
        assert first_coding["code"] == "bppv_posterior"
        # rule_ids aparecen
        assert "R-BPPV-1" in ci["finding"][0]["basis"]


# =========================================================================
# 2. Red flag activa — DetectedIssue + Flag
# =========================================================================


class TestBundleRedFlag:
    """Red flag activa: DetectedIssue(severity=high) + Flag por hit."""

    def test_detected_issue_y_flag_por_cada_hit(self) -> None:
        bundle = to_bundle(
            _result_red_flag(), _features_bppv(), _audit(urgency=Urgency.inmediata)
        )
        grouped = _resources_by_type(bundle)
        # 2 hits en el caso AVS
        assert len(grouped["DetectedIssue"]) == 2
        assert len(grouped["Flag"]) == 2

    def test_detected_issue_severity_high(self) -> None:
        bundle = to_bundle(
            _result_red_flag(), _features_bppv(), _audit(urgency=Urgency.inmediata)
        )
        issues = _resources_by_type(bundle)["DetectedIssue"]
        for di in issues:
            assert di["severity"] == "high"
            assert di["status"] == "final"
            # code = red flag id
            assert di["code"]["coding"][0]["system"].endswith("red-flag")
            # implicated = AuditEvent del bundle
            assert "implicated" in di
            assert di["implicated"][0]["reference"].startswith("urn:uuid:")

    def test_flag_status_y_code(self) -> None:
        bundle = to_bundle(
            _result_red_flag(), _features_bppv(), _audit(urgency=Urgency.inmediata)
        )
        flags = _resources_by_type(bundle)["Flag"]
        for f in flags:
            assert f["status"] == "active"
            assert f["code"]["coding"][0]["system"].endswith("red-flag")
            # text incluye forced_actions
            assert "Forced actions" in f["text"]

    def test_clinical_impression_urgency_inmediata(self) -> None:
        bundle = to_bundle(
            _result_red_flag(), _features_bppv(), _audit(urgency=Urgency.inmediata)
        )
        ci = _resources_by_type(bundle)["ClinicalImpression"][0]
        assert "inmediata" in ci["summary"]
        assert ci["code"]["coding"][0]["code"] == "428321000124101"

    def test_clinical_impression_ext_rail_triggered(self) -> None:
        """Si hay applied_rails, la extensión extension-rail-triggered se emite."""
        bundle = to_bundle(
            _result_red_flag(), _features_bppv(), _audit(urgency=Urgency.inmediata)
        )
        ci = _resources_by_type(bundle)["ClinicalImpression"][0]
        exts = ci.get("extension", [])
        rail_exts = [
            e for e in exts
            if e["url"].endswith("extension-rail-triggered")
        ]
        assert len(rail_exts) == 1
        assert "R-INV1" in rail_exts[0]["valueString"]


# =========================================================================
# 3. Reasoning=None → "degradado"
# =========================================================================


class TestBundleReasoningDegraded:
    """``reasoning=None`` no rompe; ClinicalImpression nota 'degradado'."""

    def test_clinical_impression_nota_degradado(self) -> None:
        bundle = to_bundle(
            _result_bppv(reasoning=None),
            _features_bppv(),
            _audit(model_used=None),
        )
        ci = _resources_by_type(bundle)["ClinicalImpression"][0]
        notes_text = " ".join(n["text"] for n in ci.get("note", []))
        assert "degradado" in notes_text.lower()

    def test_extension_reasoner_degraded_emitida(self) -> None:
        bundle = to_bundle(
            _result_bppv(reasoning=None),
            _features_bppv(),
            _audit(model_used=None),
        )
        ci = _resources_by_type(bundle)["ClinicalImpression"][0]
        exts = ci.get("extension", [])
        reasoner_exts = [
            e for e in exts if e["url"].endswith("extension-reasoner-degraded")
        ]
        assert len(reasoner_exts) == 1
        assert reasoner_exts[0]["valueBoolean"] is True

    def test_con_reasoning_no_hay_extension_degraded(self) -> None:
        r = ReasonerOutput(
            explanation="BPPV típico.",
            reconciliation="Concuerda con features.",
            model_used="claude-opus-4-8",
            reasoner_suggested_urgency=Urgency.ambulatoria,
        )
        bundle = to_bundle(
            _result_bppv(reasoning=r), _features_bppv(), _audit()
        )
        ci = _resources_by_type(bundle)["ClinicalImpression"][0]
        exts = ci.get("extension", [])
        reasoner_exts = [
            e for e in exts if e["url"].endswith("extension-reasoner-degraded")
        ]
        assert reasoner_exts == []
        # y la nota SÍ lleva la explicación
        notes_text = " ".join(n["text"] for n in ci.get("note", []))
        assert "BPPV típico" in notes_text


# =========================================================================
# 4. Referencias entre recursos resuelven
# =========================================================================


class TestBundleReferenciasResuelven:
    """Toda referencia ``urn:uuid:...`` apunta a un recurso del bundle."""

    def test_todas_las_referencias_resuelven(self) -> None:
        bundle = to_bundle(
            _result_red_flag(), _features_bppv(), _audit(urgency=Urgency.inmediata)
        )
        ids_in_bundle = {
            entry["resource"]["id"] for entry in bundle["entry"]
        }
        refs = _collect_urn_refs(bundle)
        for ref in refs:
            assert ref.startswith("urn:uuid:")
            target = ref.removeprefix("urn:uuid:")
            assert target in ids_in_bundle, (
                f"Referencia huérfana: {ref} (no hay recurso con id {target})"
            )

    def test_questionnaire_response_referencia_cuestionario(self) -> None:
        bundle = to_bundle(_result_bppv(), _features_bppv(), _audit())
        qr = _resources_by_type(bundle)["QuestionnaireResponse"][0]
        q = _resources_by_type(bundle)["Questionnaire"][0]
        assert qr["questionnaire"] == f"urn:uuid:{q['id']}"

    def test_todos_los_recursos_referencian_patient_placeholder(self) -> None:
        bundle = to_bundle(
            _result_red_flag(), _features_bppv(), _audit(urgency=Urgency.inmediata)
        )
        patient_id = _resources_by_type(bundle)["Patient"][0]["id"]
        # Todos los recursos que tienen `subject` o `patient` apuntan al placeholder
        for entry in bundle["entry"]:
            r = entry["resource"]
            rt = r["resourceType"]
            subj = r.get("subject", {}).get("reference")
            pat = r.get("patient", {}).get("reference")
            if rt in {
                "QuestionnaireResponse",
                "Observation",
                "ClinicalImpression",
                "Flag",
            }:
                assert subj == f"urn:uuid:{patient_id}", (
                    f"{rt}.subject referencia {subj}, no al patient placeholder"
                )
            if rt == "DetectedIssue":
                assert pat == f"urn:uuid:{patient_id}", (
                    f"DetectedIssue.patient referencia {pat}, no al patient placeholder"
                )


# =========================================================================
# 5. Determinismo
# =========================================================================


class TestBundleDeterminismo:
    """Mismo input → mismo bundle (IDs derivados del case_id)."""

    def test_mismo_case_id_mismo_bundle(self) -> None:
        result = _result_bppv(case_id="case-det-001")
        features = _features_bppv()
        audit = _audit()
        b1 = to_bundle(result, features, audit)
        b2 = to_bundle(result, features, audit)
        # Mismo id de bundle
        assert b1["id"] == b2["id"]
        # Misma cantidad de entries y mismos resource ids
        ids1 = sorted(e["resource"]["id"] for e in b1["entry"])
        ids2 = sorted(e["resource"]["id"] for e in b2["entry"])
        assert ids1 == ids2

    def test_distinto_case_id_distinto_bundle(self) -> None:
        b1 = to_bundle(_result_bppv(case_id="case-A"), _features_bppv(), _audit())
        b2 = to_bundle(_result_bppv(case_id="case-B"), _features_bppv(), _audit())
        assert b1["id"] != b2["id"]
        # No se solapan resource ids (urn:uuid: distintos)
        ids1 = {e["resource"]["id"] for e in b1["entry"]}
        ids2 = {e["resource"]["id"] for e in b2["entry"]}
        assert ids1.isdisjoint(ids2)

    def test_sin_uuid4_aleatorio(self) -> None:
        """Los IDs del bundle son uuid5 (formato uuid canónico con guiones)."""
        bundle = to_bundle(_result_bppv(), _features_bppv(), _audit())
        # Formato uuid canónico: 8-4-4-4-12 hex chars.
        uuid_re = re.compile(
            r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$"
        )
        for entry in bundle["entry"]:
            assert uuid_re.match(entry["resource"]["id"]), (
                f"ID {entry['resource']['id']!r} no parece uuid canónico"
            )


# =========================================================================
# 6. Negativo: pureza y aislamiento de imports
# =========================================================================


class TestFhirPureza:
    """``fhir`` es pura: no muta, no hace I/O, solo importa ``contracts``."""

    def test_to_bundle_no_muta_input(self) -> None:
        result = _result_bppv()
        features = _features_bppv()
        audit = _audit()
        # snapshots
        result_json = result.model_dump_json()
        features_json = features.model_dump_json()
        audit_json = audit.model_dump_json()
        to_bundle(result, features, audit)
        assert result.model_dump_json() == result_json
        assert features.model_dump_json() == features_json
        assert audit.model_dump_json() == audit_json

    def test_to_bundle_es_pura_llamadas_repetidas(self) -> None:
        """Mismo input, llamadas repetidas → mismo output."""
        result = _result_bppv(case_id="case-pure")
        features = _features_bppv()
        audit = _audit()
        b1 = to_bundle(result, features, audit)
        b2 = to_bundle(result, features, audit)
        assert b1 == b2

    def test_fhir_solo_importa_contracts(self) -> None:
        """Regla dura del mapa: ``fhir`` solo importa ``contracts`` (+ stdlib).

        Verificamos por AST que no haya imports de submódulos prohibidos
        de ``clinibrium``.
        """
        fhir_dir = Path(__file__).resolve().parents[1] / "clinibrium" / "fhir"
        forbidden_submodules = {
            "clinibrium.api",
            "clinibrium.audit",
            "clinibrium.config",
            "clinibrium.differential_engine",
            "clinibrium.grounding",
            "clinibrium.ml_client",
            "clinibrium.orchestrator",
            "clinibrium.rails",
            "clinibrium.reasoner",
            "clinibrium.redflag_engine",
            "clinibrium.storage",
        }
        for py_file in fhir_dir.glob("*.py"):
            tree = ast.parse(py_file.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom):
                    if node.module and node.module.startswith("clinibrium."):
                        mod_root = node.module.split(".")[1]
                        if mod_root != "fhir" and mod_root != "contracts":
                            # cualquier otro submódulo de clinibrium está prohibido
                            full = ".".join(node.module.split(".")[:2])
                            assert full not in forbidden_submodules, (
                                f"{py_file.name}: import prohibido "
                                f"de {node.module}"
                            )
                elif isinstance(node, ast.Import):
                    for alias in node.names:
                        if alias.name.startswith("clinibrium."):
                            mod_root = alias.name.split(".")[1]
                            assert mod_root in {"fhir", "contracts"}, (
                                f"{py_file.name}: import prohibido "
                                f"de {alias.name}"
                            )


# =========================================================================
# 7. Detalles estructurales del Questionnaire
# =========================================================================


class TestQuestionnaireEstructura:
    """Verifica que el Questionnaire tiene la forma R4 + SDC esperada."""

    def test_questionnaire_es_recurso_r4(self) -> None:
        bundle = to_bundle(_result_bppv(), _features_bppv(), _audit())
        q = _resources_by_type(bundle)["Questionnaire"][0]
        assert q["resourceType"] == "Questionnaire"
        assert q["status"] in {"draft", "active", "retired"}
        assert "url" in q
        assert "version" in q
        assert "item" in q
        assert len(q["item"]) >= 5  # representativo, no las 50

    def test_questionnaire_tiene_enable_when(self) -> None:
        """Al menos un branch con enableWhen (SDC IG)."""
        bundle = to_bundle(_result_bppv(), _features_bppv(), _audit())
        q = _resources_by_type(bundle)["Questionnaire"][0]
        items_with_branch = [
            it for it in q["item"] if "enableWhen" in it
        ]
        assert len(items_with_branch) >= 1
        # Cada item con enableWhen tiene enableBehavior
        for it in items_with_branch:
            assert "enableBehavior" in it

    def test_questionnaire_response_items_no_vacios(self) -> None:
        bundle = to_bundle(_result_bppv(), _features_bppv(), _audit())
        qr = _resources_by_type(bundle)["QuestionnaireResponse"][0]
        assert qr["status"] == "completed"
        # BPPV case: muchos features con valor → muchos items
        assert len(qr["item"]) >= 5
        # cada item tiene al menos un answer
        for it in qr["item"]:
            assert "answer" in it
            assert len(it["answer"]) >= 1


# =========================================================================
# 8. AuditEvent (perfil CL Auditoria)
# =========================================================================


class TestAuditEventClinibrium:
    """El AuditEvent tiene la forma R4 + perfil CL Auditoria."""

    def test_audit_event_meta_profile_cl(self) -> None:
        bundle = to_bundle(_result_bppv(), _features_bppv(), _audit())
        ae = _resources_by_type(bundle)["AuditEvent"][0]
        assert ae["resourceType"] == "AuditEvent"
        assert "meta" in ae
        assert any(
            p.endswith("/Auditoria")
            for p in ae["meta"].get("profile", [])
        )

    def test_audit_event_quien_que_cuando(self) -> None:
        bundle = to_bundle(_result_bppv(), _features_bppv(), _audit())
        ae = _resources_by_type(bundle)["AuditEvent"][0]
        # type + subtype
        assert ae["type"]["code"] == "pipeline_evaluation"
        # recorded = occurred_at del AuditEvent
        assert ae["recorded"].startswith("2026-07-10T12:00:00")
        # agent con who
        assert len(ae["agent"]) >= 1
        assert ae["agent"][0]["who"]["reference"].startswith("urn:uuid:")
        # entity con detail (input_features_hash, urgency, red_flag_activa,
        # model_used, reasoner_status)
        detail_types = {d["type"] for d in ae["entity"][0]["detail"]}
        assert "input_features_hash" in detail_types
        assert "urgency" in detail_types
        assert "red_flag_activa" in detail_types
        assert "model_used" in detail_types
        assert "reasoner_status" in detail_types

    def test_audit_event_reasoner_status_ok_con_reasoner(self) -> None:
        bundle = to_bundle(_result_bppv(), _features_bppv(), _audit())
        ae = _resources_by_type(bundle)["AuditEvent"][0]
        status = next(
            d for d in ae["entity"][0]["detail"] if d["type"] == "reasoner_status"
        )
        assert status["valueString"] == "ok"

    def test_audit_event_reasoner_status_degraded_sin_model(self) -> None:
        bundle = to_bundle(
            _result_bppv(reasoning=None),
            _features_bppv(),
            _audit(model_used=None),
        )
        ae = _resources_by_type(bundle)["AuditEvent"][0]
        status = next(
            d for d in ae["entity"][0]["detail"] if d["type"] == "reasoner_status"
        )
        assert status["valueString"] == "degraded"
