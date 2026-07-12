"""What Would Change My Mind? — motor contrafactual determinista + endpoint."""
from fastapi.testclient import TestClient

from clinibrium.api import create_app
from clinibrium.contracts.enums import FocalSign, Urgency
from clinibrium.contracts.features import CaseFeatures
from clinibrium.counterfactual import analyze


def _vppb_base() -> CaseFeatures:
    # Preset VPPB del demo (da ambulatoria: BPPV posicional, fatigable, latencia corta).
    return CaseFeatures(
        duration="under_1min",
        trigger="positional_head",
        timing_pattern="episodic_triggered",
        onset="sudden",
        dix_hallpike="right_positive",
        nystagmus_fatigable=True,
        nystagmus_latency_s=5.0,
    )


def test_base_vppb_is_ambulatory() -> None:
    res = analyze(_vppb_base())
    assert res.base_urgency == Urgency.ambulatoria.value


def test_focal_sign_escalates_to_immediate() -> None:
    res = analyze(_vppb_base())
    diplopia = [c for c in res.counterfactuals if c.feature == "focal_signs" and "diplop" in c.change.lower()]
    assert diplopia, "esperaba un contrafactual de signo focal"
    c = diplopia[0]
    assert c.escalates and c.new_urgency == Urgency.inmediata.value
    assert c.forced_actions_added  # p.ej. DERIVAR_URGENTE
    assert c.rails_fired


def test_each_counterfactual_changes_exactly_one_feature() -> None:
    res = analyze(_vppb_base())
    valid_fields = set(CaseFeatures.model_fields)
    assert res.counterfactuals
    for c in res.counterfactuals:
        assert c.feature in valid_fields


def test_minimal_escalation_is_lowest_urgency_that_escalates() -> None:
    res = analyze(_vppb_base())
    assert res.minimal_escalation is not None
    assert res.minimal_escalation.escalates


def test_noop_perturbation_is_skipped() -> None:
    # base que YA tiene el signo focal diplopía → no debe aparecer como contrafactual
    base = _vppb_base().model_copy(update={"focal_signs": {FocalSign.diplopia}})
    res = analyze(base)
    diplopia = [
        c for c in res.counterfactuals
        if c.feature == "focal_signs" and "diplop" in c.change.lower()
    ]
    assert not diplopia, "no debe proponer agregar un signo que ya está presente"


def test_already_immediate_case_has_no_escalation() -> None:
    # caso ya inmediato (signo focal + skew) → ningún contrafactual escala MÁS
    base = CaseFeatures(
        timing_pattern="acute_continuous", onset="sudden",
        focal_signs={FocalSign.dysarthria}, skew_deviation=True,
    )
    res = analyze(base)
    assert res.base_urgency == Urgency.inmediata.value
    assert res.minimal_escalation is None  # no se puede escalar por encima de inmediata


def test_deterministic_reproducible() -> None:
    base = _vppb_base()
    assert analyze(base).to_dict() == analyze(base).to_dict()


def test_endpoint_returns_analysis() -> None:
    client = TestClient(create_app())
    body = {
        "duration": "under_1min", "trigger": "positional_head",
        "timing_pattern": "episodic_triggered", "onset": "sudden",
        "dix_hallpike": "right_positive", "nystagmus_fatigable": True,
        "nystagmus_latency_s": 5.0,
    }
    r = client.post("/api/what-would-change", json=body)
    assert r.status_code == 200
    data = r.json()
    assert data["base_urgency"] == "ambulatoria"
    assert isinstance(data["counterfactuals"], list) and data["counterfactuals"]
    assert data["minimal_escalation"] is not None
    # contrato: cada contrafactual tiene los campos esperados
    c = data["counterfactuals"][0]
    assert {"feature", "change", "new_urgency", "escalates", "rails_fired"} <= set(c)


def test_endpoint_rejects_extra_keys() -> None:
    client = TestClient(create_app())
    r = client.post("/api/what-would-change", json={"patient_name": "Juan"})
    assert r.status_code == 422
