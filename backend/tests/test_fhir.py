"""Tests for the ``fhir`` module (auditable FHIR R4 artifact).

Covers the acceptance criteria of T10:

  1. ``to_bundle(...)`` with a benign-BPPV PipelineResult → valid Bundle
     with Questionnaire, QuestionnaireResponse, Observation,
     ClinicalImpression, AuditEvent. NO DetectedIssue/Flag when there
     are no red flags.

  2. ``to_bundle(...)`` with an active red flag (urgency=inmediata, hits) →
     DetectedIssue (severity=high) + one Flag per hit, and
     ClinicalImpression with urgency inmediata.

  3. ``reasoning=None`` → ClinicalImpression notes "degradado", does not
     break. The ``extension-reasoner-degraded`` extension is emitted.

  4. The Bundle serializes to JSON (``json.dumps``) and cross-resource
     references resolve: every referenced resource is in the bundle
     (automatic ``urn:uuid:`` resolution test).

  5. Determinism: same input → same bundle (IDs derived from
     ``case_id``).

  6. Negative: ``fhir`` ONLY imports ``contracts`` (hard rule of the
     module map — AST).

  7. Negative: ``to_bundle`` is pure (no input mutation, no I/O).
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
    """Benign BPPV case: positional, torsional, fatigable, <60s."""
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
    """Group the bundle's resources by ``resourceType``."""
    out: dict[str, list[dict]] = {}
    for entry in bundle.get("entry", []):
        r = entry.get("resource", {})
        rt = r.get("resourceType")
        if rt is None:
            continue
        out.setdefault(rt, []).append(r)
    return out


def _collect_urn_refs(obj: object) -> set[str]:
    """Walk a dict/list and return every referenced ``urn:uuid:...``."""
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
# 1. Benign case — BPPV without red flags
# =========================================================================


class TestBundleBppvBenign:
    """Bundle for a benign BPPV case: NO DetectedIssue/Flag."""

    def test_bundle_type_collection(self) -> None:
        bundle = to_bundle(_result_bppv(), _features_bppv(), _audit())
        assert bundle["resourceType"] == "Bundle"
        assert bundle["type"] == "collection"
        assert "id" in bundle
        assert "timestamp" in bundle

    def test_bundle_serializes_to_json(self) -> None:
        bundle = to_bundle(_result_bppv(), _features_bppv(), _audit())
        s = json.dumps(bundle)
        # roundtrip
        again = json.loads(s)
        assert again["resourceType"] == "Bundle"
        assert again["id"] == bundle["id"]

    def test_expected_resources_present(self) -> None:
        bundle = to_bundle(_result_bppv(), _features_bppv(), _audit())
        grouped = _resources_by_type(bundle)
        # MANDATORY resources
        assert "Questionnaire" in grouped
        assert "QuestionnaireResponse" in grouped
        assert "ClinicalImpression" in grouped
        assert "AuditEvent" in grouped
        # At least 1 Observation (nystagmus_direction or duration)
        assert "Observation" in grouped
        assert len(grouped["Observation"]) >= 1
        # Patient placeholder (reference anchor)
        assert "Patient" in grouped

    def test_no_detected_issue_no_flag(self) -> None:
        bundle = to_bundle(_result_bppv(), _features_bppv(), _audit())
        grouped = _resources_by_type(bundle)
        assert "DetectedIssue" not in grouped
        assert "Flag" not in grouped

    def test_clinical_impression_bppv(self) -> None:
        bundle = to_bundle(_result_bppv(), _features_bppv(), _audit())
        ci = _resources_by_type(bundle)["ClinicalImpression"][0]
        assert ci["status"] == "completed"
        # summary mentions ambulatory urgency
        assert "ambulatoria" in ci["summary"]
        # findings: 2 candidates
        assert len(ci["finding"]) == 2
        # first candidate's score in the basis
        assert "score=0.920" in ci["finding"][0]["basis"]
        # first finding's coding = bppv_posterior
        first_coding = ci["finding"][0]["itemCodeableConcept"]["coding"][0]
        assert first_coding["code"] == "bppv_posterior"
        # rule_ids appear
        assert "R-BPPV-1" in ci["finding"][0]["basis"]


# =========================================================================
# 2. Active red flag — DetectedIssue + Flag
# =========================================================================


class TestBundleRedFlag:
    """Active red flag: DetectedIssue(severity=high) + one Flag per hit."""

    def test_detected_issue_and_flag_per_hit(self) -> None:
        bundle = to_bundle(
            _result_red_flag(), _features_bppv(), _audit(urgency=Urgency.inmediata)
        )
        grouped = _resources_by_type(bundle)
        # 2 hits in the AVS case
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
            # implicated = the bundle's AuditEvent
            assert "implicated" in di
            assert di["implicated"][0]["reference"].startswith("urn:uuid:")

    def test_flag_status_and_code(self) -> None:
        bundle = to_bundle(
            _result_red_flag(), _features_bppv(), _audit(urgency=Urgency.inmediata)
        )
        flags = _resources_by_type(bundle)["Flag"]
        for f in flags:
            assert f["status"] == "active"
            assert f["code"]["coding"][0]["system"].endswith("red-flag")
            # text includes forced_actions
            assert "Forced actions" in f["text"]

    def test_clinical_impression_urgency_inmediata(self) -> None:
        bundle = to_bundle(
            _result_red_flag(), _features_bppv(), _audit(urgency=Urgency.inmediata)
        )
        ci = _resources_by_type(bundle)["ClinicalImpression"][0]
        assert "inmediata" in ci["summary"]
        assert ci["code"]["coding"][0]["code"] == "428321000124101"

    def test_clinical_impression_ext_rail_triggered(self) -> None:
        """If there are applied_rails, the extension-rail-triggered extension is emitted."""
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
    """``reasoning=None`` does not break; ClinicalImpression notes 'degradado'."""

    def test_clinical_impression_notes_degraded(self) -> None:
        bundle = to_bundle(
            _result_bppv(reasoning=None),
            _features_bppv(),
            _audit(model_used=None),
        )
        ci = _resources_by_type(bundle)["ClinicalImpression"][0]
        notes_text = " ".join(n["text"] for n in ci.get("note", []))
        assert "degradado" in notes_text.lower()

    def test_extension_reasoner_degraded_emitted(self) -> None:
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

    def test_with_reasoning_no_degraded_extension(self) -> None:
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
        # and the note DOES carry the explanation
        notes_text = " ".join(n["text"] for n in ci.get("note", []))
        assert "BPPV típico" in notes_text


# =========================================================================
# 4. Cross-resource references resolve
# =========================================================================


class TestBundleReferencesResolve:
    """Every ``urn:uuid:...`` reference points at a resource in the bundle."""

    def test_all_references_resolve(self) -> None:
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
                f"Orphan reference: {ref} (no resource with id {target})"
            )

    def test_questionnaire_response_references_questionnaire(self) -> None:
        bundle = to_bundle(_result_bppv(), _features_bppv(), _audit())
        qr = _resources_by_type(bundle)["QuestionnaireResponse"][0]
        q = _resources_by_type(bundle)["Questionnaire"][0]
        assert qr["questionnaire"] == f"urn:uuid:{q['id']}"

    def test_all_resources_reference_patient_placeholder(self) -> None:
        bundle = to_bundle(
            _result_red_flag(), _features_bppv(), _audit(urgency=Urgency.inmediata)
        )
        patient_id = _resources_by_type(bundle)["Patient"][0]["id"]
        # Every resource with `subject` or `patient` points at the placeholder
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
                    f"{rt}.subject references {subj}, not the patient placeholder"
                )
            if rt == "DetectedIssue":
                assert pat == f"urn:uuid:{patient_id}", (
                    f"DetectedIssue.patient references {pat}, not the patient placeholder"
                )


# =========================================================================
# 5. Determinism
# =========================================================================


class TestBundleDeterminism:
    """Same input → same bundle (IDs derived from case_id)."""

    def test_same_case_id_same_bundle(self) -> None:
        result = _result_bppv(case_id="case-det-001")
        features = _features_bppv()
        audit = _audit()
        b1 = to_bundle(result, features, audit)
        b2 = to_bundle(result, features, audit)
        # Same bundle id
        assert b1["id"] == b2["id"]
        # Same number of entries and same resource ids
        ids1 = sorted(e["resource"]["id"] for e in b1["entry"])
        ids2 = sorted(e["resource"]["id"] for e in b2["entry"])
        assert ids1 == ids2

    def test_different_case_id_different_bundle(self) -> None:
        b1 = to_bundle(_result_bppv(case_id="case-A"), _features_bppv(), _audit())
        b2 = to_bundle(_result_bppv(case_id="case-B"), _features_bppv(), _audit())
        assert b1["id"] != b2["id"]
        # Resource ids do not overlap (distinct urn:uuid:)
        ids1 = {e["resource"]["id"] for e in b1["entry"]}
        ids2 = {e["resource"]["id"] for e in b2["entry"]}
        assert ids1.isdisjoint(ids2)

    def test_no_random_uuid4(self) -> None:
        """Bundle IDs are uuid5 (canonical hyphenated uuid format)."""
        bundle = to_bundle(_result_bppv(), _features_bppv(), _audit())
        # Canonical uuid format: 8-4-4-4-12 hex chars.
        uuid_re = re.compile(
            r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$"
        )
        for entry in bundle["entry"]:
            assert uuid_re.match(entry["resource"]["id"]), (
                f"ID {entry['resource']['id']!r} does not look like a canonical uuid"
            )


# =========================================================================
# 6. Negative: purity and import isolation
# =========================================================================


class TestFhirPurity:
    """``fhir`` is pure: no mutation, no I/O, only imports ``contracts``."""

    def test_to_bundle_does_not_mutate_input(self) -> None:
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

    def test_to_bundle_is_pure_repeated_calls(self) -> None:
        """Same input, repeated calls → same output."""
        result = _result_bppv(case_id="case-pure")
        features = _features_bppv()
        audit = _audit()
        b1 = to_bundle(result, features, audit)
        b2 = to_bundle(result, features, audit)
        assert b1 == b2

    def test_fhir_only_imports_contracts(self) -> None:
        """Hard rule of the module map: ``fhir`` only imports ``contracts`` (+ stdlib).

        We verify via AST that there are no imports of forbidden
        ``clinibrium`` submodules.
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
                            # any other clinibrium submodule is forbidden
                            full = ".".join(node.module.split(".")[:2])
                            assert full not in forbidden_submodules, (
                                f"{py_file.name}: forbidden import "
                                f"of {node.module}"
                            )
                elif isinstance(node, ast.Import):
                    for alias in node.names:
                        if alias.name.startswith("clinibrium."):
                            mod_root = alias.name.split(".")[1]
                            assert mod_root in {"fhir", "contracts"}, (
                                f"{py_file.name}: forbidden import "
                                f"of {alias.name}"
                            )


# =========================================================================
# 7. Structural details of the Questionnaire
# =========================================================================


class TestQuestionnaireStructure:
    """Verifies the Questionnaire has the expected R4 + SDC shape."""

    def test_questionnaire_is_r4_resource(self) -> None:
        bundle = to_bundle(_result_bppv(), _features_bppv(), _audit())
        q = _resources_by_type(bundle)["Questionnaire"][0]
        assert q["resourceType"] == "Questionnaire"
        assert q["status"] in {"draft", "active", "retired"}
        assert "url" in q
        assert "version" in q
        assert "item" in q
        assert len(q["item"]) >= 5  # representative, not all 50

    def test_questionnaire_has_enable_when(self) -> None:
        """At least one branch with enableWhen (SDC IG)."""
        bundle = to_bundle(_result_bppv(), _features_bppv(), _audit())
        q = _resources_by_type(bundle)["Questionnaire"][0]
        items_with_branch = [
            it for it in q["item"] if "enableWhen" in it
        ]
        assert len(items_with_branch) >= 1
        # Every item with enableWhen has enableBehavior
        for it in items_with_branch:
            assert "enableBehavior" in it

    def test_questionnaire_response_items_not_empty(self) -> None:
        bundle = to_bundle(_result_bppv(), _features_bppv(), _audit())
        qr = _resources_by_type(bundle)["QuestionnaireResponse"][0]
        assert qr["status"] == "completed"
        # BPPV case: many features with values → many items
        assert len(qr["item"]) >= 5
        # every item has at least one answer
        for it in qr["item"]:
            assert "answer" in it
            assert len(it["answer"]) >= 1


# =========================================================================
# 8. AuditEvent (CL Auditoria profile)
# =========================================================================


class TestAuditEventClinibrium:
    """The AuditEvent has the R4 shape + CL Auditoria profile."""

    def test_audit_event_meta_profile_cl(self) -> None:
        bundle = to_bundle(_result_bppv(), _features_bppv(), _audit())
        ae = _resources_by_type(bundle)["AuditEvent"][0]
        assert ae["resourceType"] == "AuditEvent"
        assert "meta" in ae
        assert any(
            p.endswith("/Auditoria")
            for p in ae["meta"].get("profile", [])
        )

    def test_audit_event_who_what_when(self) -> None:
        bundle = to_bundle(_result_bppv(), _features_bppv(), _audit())
        ae = _resources_by_type(bundle)["AuditEvent"][0]
        # type + subtype
        assert ae["type"]["code"] == "pipeline_evaluation"
        # recorded = the AuditEvent's occurred_at
        assert ae["recorded"].startswith("2026-07-10T12:00:00")
        # agent with who
        assert len(ae["agent"]) >= 1
        assert ae["agent"][0]["who"]["reference"].startswith("urn:uuid:")
        # entity with detail (input_features_hash, urgency, red_flag_activa,
        # model_used, reasoner_status)
        detail_types = {d["type"] for d in ae["entity"][0]["detail"]}
        assert "input_features_hash" in detail_types
        assert "urgency" in detail_types
        assert "red_flag_activa" in detail_types
        assert "model_used" in detail_types
        assert "reasoner_status" in detail_types

    def test_audit_event_reasoner_status_ok_with_reasoner(self) -> None:
        bundle = to_bundle(_result_bppv(), _features_bppv(), _audit())
        ae = _resources_by_type(bundle)["AuditEvent"][0]
        status = next(
            d for d in ae["entity"][0]["detail"] if d["type"] == "reasoner_status"
        )
        assert status["valueString"] == "ok"

    def test_audit_event_reasoner_status_degraded_without_model(self) -> None:
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


# =========================================================================
# 9. Tamper-evident integrity — bundle_sha256
# =========================================================================


class TestBundleIntegrity:
    """``bundle_sha256``: canonical hash for integrity verification."""

    def test_same_bundle_same_hash(self) -> None:
        """Same bundle → same 64-char hex SHA-256 hash."""
        from clinibrium.fhir import bundle_sha256
        bundle = to_bundle(_result_bppv(), _features_bppv(), _audit())
        h1 = bundle_sha256(bundle)
        h2 = bundle_sha256(bundle)
        assert h1 == h2
        assert len(h1) == 64
        assert all(c in "0123456789abcdef" for c in h1)

    def test_tampered_bundle_different_hash(self) -> None:
        """Altering the bundle (modifying timestamp) → different hash."""
        from clinibrium.fhir import bundle_sha256
        bundle1 = to_bundle(_result_bppv(), _features_bppv(), _audit())
        bundle2 = to_bundle(_result_bppv(), _features_bppv(), _audit())
        bundle2["timestamp"] = "2020-01-01T00:00:00+00:00"
        assert bundle_sha256(bundle1) != bundle_sha256(bundle2)

    def test_hash_deterministic_across_calls(self) -> None:
        """Same inputs to to_bundle → same hash (determinism)."""
        from clinibrium.fhir import bundle_sha256
        result = _result_bppv(case_id="case-hash-det")
        features = _features_bppv()
        audit = _audit()
        b1 = to_bundle(result, features, audit)
        b2 = to_bundle(result, features, audit)
        assert bundle_sha256(b1) == bundle_sha256(b2)

    def test_canonical_matches_js_json_stringify(self) -> None:
        """The canonical form must be byte-identical to JS ``JSON.stringify``.

        This is the property that lets the frontend's "Verify" button
        recompute the SAME hash (✓ intact). The fragile spot: an integral
        float (``5.0``) — Python would emit ``"5.0"`` and JS ``"5"``.
        We replicate the client's ``jsonCanonical`` here and require a match.
        """
        import hashlib
        import json

        from clinibrium.fhir.mapping import _canonical_json, bundle_sha256

        # _js_number: integral float without ".0"; fractional as shortest-repr.
        assert _canonical_json(5.0) == "5"
        assert _canonical_json(5.5) == "5.5"
        assert _canonical_json({"b": 1, "a": 2.0}) == '{"a":2,"b":1}'

        def js_canonical(obj: object) -> str:
            # Exact replica of jsonCanonical (ClinicalCaseReceipt.tsx):
            # numbers via the JS representation (integers without ".0").
            if obj is None or not isinstance(obj, (dict, list)):
                if isinstance(obj, bool):
                    return "true" if obj else "false"
                if isinstance(obj, float) and obj.is_integer():
                    return str(int(obj))
                return json.dumps(obj, ensure_ascii=False)
            if isinstance(obj, list):
                return "[" + ",".join(js_canonical(v) for v in obj) + "]"
            keys = sorted(obj.keys())
            return (
                "{"
                + ",".join(
                    json.dumps(k, ensure_ascii=False) + ":" + js_canonical(obj[k])
                    for k in keys
                )
                + "}"
            )

        # Bundle with an integral float (nystagmus_latency_s=5.0 in the preset).
        bundle = to_bundle(_result_bppv(), _features_bppv(), _audit())
        client_hash = hashlib.sha256(
            js_canonical(bundle).encode("utf-8")
        ).hexdigest()
        assert client_hash == bundle_sha256(bundle)


# =========================================================================
# 8. output_lang → ClinicalImpression.language (AD-19 Decisión 3,
#    codex-audit-4 Alta 1: reasoner prose keeps its requested language and
#    the artifact must say so; the es/None bundle stays byte-identical)
# =========================================================================


def _reasoning_sentinel_en() -> ReasonerOutput:
    return ReasonerOutput(
        explanation="SENTINEL-EN explanation grounded in [icvd-bppv-1].",
        reconciliation="SENTINEL-EN reconciliation with the deterministic layers.",
        suggested_next_steps=["SENTINEL-EN next step"],
        model_used="claude-haiku-4-5-20251001",
        reasoner_suggested_urgency=None,
        grounding_refs=["icvd-bppv-1"],
    )


def test_bundle_en_reasoner_prose_is_tagged_with_language():
    """reasoner(en) → PipelineResult → FHIR: the prose enters the notes AND
    the ClinicalImpression is honestly tagged with FHIR ``language: "en"``."""
    audit = _audit().model_copy(update={"output_lang": "en"})
    bundle = to_bundle(
        _result_bppv(reasoning=_reasoning_sentinel_en()), _features_bppv(), audit
    )
    ci = _resources_by_type(bundle)["ClinicalImpression"][0]
    assert ci["language"] == "en"
    notes = " ".join(n["text"] for n in ci["note"])
    assert "SENTINEL-EN explanation" in notes
    assert "SENTINEL-EN reconciliation" in notes
    assert "SENTINEL-EN next step" in notes
    # Deterministic content stays canonical Spanish regardless of the tag.
    assert ci["summary"].startswith("Urgencia final:")


def test_bundle_es_and_none_add_no_language_key_and_are_identical():
    """Spanish/legacy paths add NO key → the default bundle is byte-identical
    (the recorded demo depends on this)."""
    reasoning = _reasoning_sentinel_en()
    bundles = []
    for lang in (None, "es"):
        audit = _audit().model_copy(update={"output_lang": lang})
        bundle = to_bundle(_result_bppv(reasoning=reasoning), _features_bppv(), audit)
        ci = _resources_by_type(bundle)["ClinicalImpression"][0]
        assert "language" not in ci
        bundles.append(bundle)
    assert json.dumps(bundles[0], sort_keys=True) == json.dumps(bundles[1], sort_keys=True)


def test_bundle_en_degraded_reasoner_is_not_tagged():
    """No reasoner prose → nothing English in the bundle → no language tag,
    even if the UI requested en."""
    audit = _audit(model_used=None).model_copy(update={"output_lang": "en"})
    bundle = to_bundle(_result_bppv(reasoning=None), _features_bppv(), audit)
    ci = _resources_by_type(bundle)["ClinicalImpression"][0]
    assert "language" not in ci
    assert any("degradado" in n["text"] for n in ci["note"])
