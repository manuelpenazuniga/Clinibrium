"""CLI de entrenamiento (TB1.8): genera sintético → entrena → evalúa → guarda.

Uso:  python -m ml_engine.train  [--out DIR] [--n N] [--iterations I]

Las métricas miden RECUPERACIÓN DEL PROCESO GENERATIVO sintético (AD-17), NO
desempeño clínico. Escribe artifacts (modelos + manifest) + MODEL_CARD.md.
"""
from __future__ import annotations

import argparse
import dataclasses
from pathlib import Path

from ml_engine.core.model import HierarchicalCatBoost, _danger_leaves
from ml_engine.core.synth import generate
from ml_engine.core.spec import Domain
from ml_engine.domains import vertigo

_DEFAULT_OUT = Path(__file__).parent / "artifacts" / "model"


@dataclasses.dataclass
class Metrics:
    n_test: int
    leaf_accuracy: float
    gate_accuracy: float
    danger_recall: float  # P(pred peligro | verdadero peligro) — el que importa


def _evaluate(model: HierarchicalCatBoost, test_df, domain: Domain) -> Metrics:
    danger = _danger_leaves(domain.hierarchy)
    rows = test_df.to_dict("records")
    y_true = [str(r["label"]) for r in rows]
    preds = model.predict_proba(rows)

    leaf_hit = 0
    gate_hit = 0
    danger_total = 0
    danger_caught = 0
    for yt, p in zip(y_true, preds, strict=True):
        pred_leaf = max(p, key=p.__getitem__)
        leaf_hit += int(pred_leaf == yt)
        p_danger = sum(p[leaf] for leaf in danger)
        pred_danger = p_danger > 0.5
        true_danger = yt in danger
        gate_hit += int(pred_danger == true_danger)
        if true_danger:
            danger_total += 1
            danger_caught += int(pred_danger)

    n = len(rows)
    return Metrics(
        n_test=n,
        leaf_accuracy=leaf_hit / n,
        gate_accuracy=gate_hit / n,
        danger_recall=(danger_caught / danger_total) if danger_total else float("nan"),
    )


def train_domain(
    domain: Domain,
    *,
    seed: int = 20260711,
    n_samples: int | None = None,
    params: dict | None = None,
    test_frac: float = 0.2,
) -> tuple[HierarchicalCatBoost, Metrics]:
    spec = domain.synthetic
    if n_samples is not None:
        spec = dataclasses.replace(spec, n_samples=n_samples)
    df = generate(spec, domain.features, seed=seed)
    n_test = int(len(df) * test_frac)
    test_df, train_df = df.iloc[:n_test], df.iloc[n_test:]
    model = HierarchicalCatBoost.train(domain, train_df, seed=seed, params=params)
    metrics = _evaluate(model, test_df, domain)
    return model, metrics


def _write_model_card(out: Path, model: HierarchicalCatBoost, m: Metrics, domain: Domain) -> None:
    card = f"""# MODEL CARD — `ml_engine` capa de confianza ({domain.name})

> **EXPERIMENTAL · entrenado sobre datos 100% SINTÉTICOS · SIN validez clínica.**

- **model_version:** `{model.model_version}`
- **Arquitectura:** CatBoost jerárquico (LCPN). Gate raíz binario
  `dangerous vs peripheral` con `monotone_constraints=+1` en features de riesgo
  (INV-9: subir una feature de riesgo nunca baja P(dangerous), garantía dura del
  gate pre-abstención). Hijos: central/cardiogénico, periférico (5), BPPV (2).
- **Features:** {len(domain.features.feature_names)} ({len(domain.features.categorical_names)} categóricas +
  {len(domain.features.numeric_feature_names)} numéricas, incl. {len(domain.features.derived)} derivadas).
- **Risk features (monótonas):** {", ".join(domain.features.risk_features)}.

## Métricas (recuperación del generador sintético, NO clínicas)
Sobre {m.n_test} casos sintéticos held-out (mismo generador; miden cuán bien el
modelo reconstruye las reglas con que se fabricaron los datos):

| Métrica | Valor |
|---|---|
| Leaf accuracy (8 clases) | {m.leaf_accuracy:.3f} |
| Gate accuracy (peligro vs periférico) | {m.gate_accuracy:.3f} |
| **Danger recall** (peligro capturado) | {m.danger_recall:.3f} |

> ⚠️ Riesgo de circularidad conocido: el label genera las features y el modelo
> las reaprende → estas métricas NO implican utilidad clínica. Son "recuperación
> del generador". Priors provisionales pendientes de firma del especialista (T-CLIN).

## Límites (honestidad)
- El ML **nunca** fija urgencia (INV-11); solo aporta evidencia probabilística
  al razonador. Los rieles deterministas deciden.
- Sin calibración clínica; la calibración (temperature scaling) se mide con ECE
  sobre held-out sintético (ver TB1.4).
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
    print(f"OK · leaf_acc={m.leaf_accuracy:.3f} gate_acc={m.gate_accuracy:.3f} "
          f"danger_recall={m.danger_recall:.3f} (n_test={m.n_test})")


if __name__ == "__main__":
    main()
