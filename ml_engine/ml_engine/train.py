"""CLI de entrenamiento (TB1.8/1.4/1.5): genera → entrena → calibra → abstención → evalúa → guarda.

Split 3-vías train/calibración/test (sin leakage): el modelo se ajusta en train,
la temperatura y el umbral de abstención en calibración, y las métricas (incl.
ECE) se reportan UNA vez sobre test intacto.

Las métricas miden RECUPERACIÓN DEL PROCESO GENERATIVO sintético (AD-17), NO
desempeño clínico. Escribe artifacts (modelos + calibración + manifest) + MODEL_CARD.
"""
from __future__ import annotations

import argparse
import dataclasses
import math
from pathlib import Path

from ml_engine.core.abstain import ConfidenceGate
from ml_engine.core.calibrate import TemperatureCalibrator, ece
from ml_engine.core.model import HierarchicalCatBoost, _danger_leaves
from ml_engine.core.spec import Domain
from ml_engine.core.synth import generate
from ml_engine.domains import vertigo

_DEFAULT_OUT = Path(__file__).parent / "artifacts" / "model"
_TARGET_COVERAGE = 0.90


@dataclasses.dataclass
class Metrics:
    n_test: int
    leaf_accuracy: float
    gate_accuracy: float
    danger_recall: float  # P(pred peligro | verdadero peligro) — el que importa
    ece_raw: float
    ece_calibrated: float
    abstain_rate: float
    temperature: float
    abstain_threshold: float
    # Stress set: mismos priors pero 40% de features faltantes (robustez a
    # inputs reales muy esparsos).
    stress_leaf_accuracy: float
    stress_abstain_rate: float


def _leaf_acc_abstain(model: HierarchicalCatBoost, df, domain: Domain) -> tuple[float, float]:
    """(leaf_accuracy, abstain_rate) sobre un DataFrame — reusable para stress."""
    rows = df.to_dict("records")
    y = [str(r["label"]) for r in rows]
    raw = model.predict_proba(rows)
    cal = model.calibrator.transform(raw) if model.calibrator else raw
    hit = abst = 0
    for yt, pcal in zip(y, cal, strict=True):
        hit += int(max(pcal, key=pcal.__getitem__) == yt)
        if model.abstainer and model.abstainer.abstains(pcal):
            abst += 1
    n = max(len(rows), 1)
    return hit / n, abst / n


def _evaluate(model: HierarchicalCatBoost, test_df, stress_df, domain: Domain) -> Metrics:
    leaves = domain.hierarchy.leaves
    danger = _danger_leaves(domain.hierarchy)
    rows = test_df.to_dict("records")
    y = [str(r["label"]) for r in rows]
    raw = model.predict_proba(rows)
    cal = model.calibrator.transform(raw) if model.calibrator else raw

    leaf_hit = gate_hit = 0
    d_total = d_caught = abst = 0
    for yt, pcal in zip(y, cal, strict=True):
        pred = max(pcal, key=pcal.__getitem__)
        leaf_hit += int(pred == yt)
        p_danger = sum(pcal[leaf] for leaf in danger)
        pred_danger = p_danger > 0.5
        true_danger = yt in danger
        gate_hit += int(pred_danger == true_danger)
        if true_danger:
            d_total += 1
            d_caught += int(pred_danger)
        if model.abstainer and model.abstainer.abstains(pcal):
            abst += 1

    n = len(rows)
    stress_leaf, stress_abst = _leaf_acc_abstain(model, stress_df, domain)
    return Metrics(
        n_test=n,
        leaf_accuracy=leaf_hit / n,
        gate_accuracy=gate_hit / n,
        danger_recall=(d_caught / d_total) if d_total else math.nan,
        ece_raw=ece(raw, y, leaves),
        ece_calibrated=ece(cal, y, leaves),
        abstain_rate=abst / n,
        temperature=model.calibrator.temperature if model.calibrator else 1.0,
        abstain_threshold=model.abstainer.threshold if model.abstainer else 0.0,
        stress_leaf_accuracy=stress_leaf,
        stress_abstain_rate=stress_abst,
    )


def train_domain(
    domain: Domain,
    *,
    seed: int = 20260711,
    n_samples: int | None = None,
    params: dict | None = None,
    splits: tuple[float, float, float] = (0.6, 0.2, 0.2),
) -> tuple[HierarchicalCatBoost, Metrics]:
    spec = domain.synthetic
    if n_samples is not None:
        spec = dataclasses.replace(spec, n_samples=n_samples)
    df = generate(spec, domain.features, seed=seed)
    n = len(df)
    n_test = int(n * splits[2])
    n_cal = int(n * splits[1])
    test_df = df.iloc[:n_test]
    cal_df = df.iloc[n_test : n_test + n_cal]
    train_df = df.iloc[n_test + n_cal :]

    model = HierarchicalCatBoost.train(domain, train_df, seed=seed, params=params)

    # calibración + abstención en el split de calibración (sin tocar test)
    leaves = domain.hierarchy.leaves
    cal_rows = cal_df.to_dict("records")
    raw_cal = model.predict_proba(cal_rows)
    y_cal = [str(r["label"]) for r in cal_rows]
    model.calibrator = TemperatureCalibrator.fit(raw_cal, y_cal, leaves)
    cal_calibrated = model.calibrator.transform(raw_cal)
    model.abstainer = ConfidenceGate.fit(
        cal_calibrated,
        target_coverage=_TARGET_COVERAGE,
        abstain_label=domain.hierarchy.abstain_label,
    )

    # stress set: mismos priors, 40% de features faltantes (robustez a esparsos)
    stress_spec = dataclasses.replace(
        spec, n_samples=min(1500, spec.n_samples), missing_rate=0.40, seed=spec.seed + 1
    )
    stress_df = generate(stress_spec, domain.features, seed=spec.seed + 1)

    return model, _evaluate(model, test_df, stress_df, domain)


def _write_model_card(out: Path, model: HierarchicalCatBoost, m: Metrics, domain: Domain) -> None:
    card = f"""# MODEL CARD — `ml_engine` capa de confianza ({domain.name})

> **EXPERIMENTAL · entrenado sobre datos 100% SINTÉTICOS · SIN validez clínica.**

- **model_version:** `{model.model_version}`
- **Arquitectura:** CatBoost jerárquico (LCPN). Gate raíz binario
  `dangerous vs peripheral` con `monotone_constraints=+1` en features de riesgo
  (INV-9: subir una feature de riesgo nunca baja P(dangerous), garantía dura del
  gate pre-abstención). Hijos: central/cardiogénico, periférico (5), BPPV (2).
- **Calibración:** temperature scaling del vector final, T={m.temperature:.3f}
  (ajustada en split de calibración por NLL). **Abstención:** gate de confianza
  τ={m.abstain_threshold:.3f} (cobertura objetivo {_TARGET_COVERAGE:.0%}) →
  `undetermined` como centinela (INV-10).
- **Features:** {len(domain.features.feature_names)} ({len(domain.features.categorical_names)} categóricas +
  {len(domain.features.numeric_feature_names)} numéricas, incl. {len(domain.features.derived)} derivadas).
- **Risk features (monótonas):** {", ".join(domain.features.risk_features)}.

## Métricas (recuperación del generador sintético, NO clínicas)
Sobre {m.n_test} casos sintéticos held-out (test intacto, evaluado una vez):

| Métrica | Valor |
|---|---|
| Leaf accuracy (8 clases) | {m.leaf_accuracy:.3f} |
| Gate accuracy (peligro vs periférico) | {m.gate_accuracy:.3f} |
| **Danger recall** (peligro capturado) | {m.danger_recall:.3f} |
| ECE crudo | {m.ece_raw:.4f} |
| ECE calibrado (T={m.temperature:.2f}) | {m.ece_calibrated:.4f} |
| Tasa de abstención (→ undetermined) | {m.abstain_rate:.3f} |
| **Stress** — leaf accuracy con 40% de features faltantes | {m.stress_leaf_accuracy:.3f} |
| **Stress** — tasa de abstención con 40% faltantes | {m.stress_abstain_rate:.3f} |

> ⚠️ Riesgo de circularidad conocido: el label genera las features y el modelo
> las reaprende → estas métricas NO implican utilidad clínica. Son "recuperación
> del generador". El ECE calibrado se REPORTA (no se impone como test — temperature
> minimiza NLL, no ECE). Priors provisionales pendientes de firma del especialista (T-CLIN).

## Límites (honestidad)
- El ML **nunca** fija urgencia (INV-11); solo aporta evidencia probabilística
  al razonador. Los rieles deterministas deciden.
- La abstención es EVIDENCIA (el ML sabe decir "no sé"); NO escala urgencia.
- SHAP (si presente) = atribución local no causal sobre el generador sintético.
"""
    (out / "MODEL_CARD.md").write_text(card)


def build_and_save(
    out_dir: str | Path = _DEFAULT_OUT, domain: Domain = vertigo.VERTIGO, **kw: object
) -> tuple[HierarchicalCatBoost, Metrics]:
    model, metrics = train_domain(domain, **kw)  # type: ignore[arg-type]
    out = Path(out_dir)
    model.save(out)
    _write_model_card(out, model, metrics, domain)
    return model, metrics


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(_DEFAULT_OUT))
    ap.add_argument("--n", type=int, default=None, help="n_samples (default: spec)")
    ap.add_argument("--iterations", type=int, default=None)
    ap.add_argument("--seed", type=int, default=20260711)
    args = ap.parse_args()
    params = {"iterations": args.iterations} if args.iterations else None
    _, m = build_and_save(args.out, seed=args.seed, n_samples=args.n, params=params)
    print(
        f"OK · leaf_acc={m.leaf_accuracy:.3f} gate_acc={m.gate_accuracy:.3f} "
        f"danger_recall={m.danger_recall:.3f} ECE {m.ece_raw:.3f}→{m.ece_calibrated:.3f} "
        f"abstain={m.abstain_rate:.3f} · stress(40%miss) leaf={m.stress_leaf_accuracy:.3f} "
        f"abstain={m.stress_abstain_rate:.3f} (n_test={m.n_test})"
    )


if __name__ == "__main__":
    main()
