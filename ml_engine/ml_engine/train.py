"""Training CLI (TB1.8/1.4/1.5): generate → train → calibrate → abstention → evaluate → save.

3-way train/calibration/test split (no leakage): the model is fit on train,
the temperature and the abstention threshold on calibration, and the metrics
(incl. ECE) are reported ONCE on untouched test data.

The metrics measure RECOVERY OF THE synthetic GENERATIVE PROCESS (AD-17), NOT
clinical performance. Writes artifacts (models + calibration + manifest) + MODEL_CARD.
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
    danger_recall: float  # P(pred danger | true danger) — the one that matters
    ece_raw: float
    ece_calibrated: float
    abstain_rate: float
    temperature: float
    abstain_threshold: float
    # Stress set: same priors but 40% of features missing (robustness to very
    # sparse real inputs).
    stress_leaf_accuracy: float
    stress_abstain_rate: float


def _leaf_acc_abstain(model: HierarchicalCatBoost, df, domain: Domain) -> tuple[float, float]:
    """(leaf_accuracy, abstain_rate) over a DataFrame — reusable for stress."""
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

    # calibration + abstention on the calibration split (test untouched)
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

    # stress set: same priors, 40% of features missing (robustness to sparse inputs)
    stress_spec = dataclasses.replace(
        spec, n_samples=min(1500, spec.n_samples), missing_rate=0.40, seed=spec.seed + 1
    )
    stress_df = generate(stress_spec, domain.features, seed=spec.seed + 1)

    return model, _evaluate(model, test_df, stress_df, domain)


def _write_model_card(out: Path, model: HierarchicalCatBoost, m: Metrics, domain: Domain) -> None:
    card = f"""# MODEL CARD — `ml_engine` confidence layer ({domain.name})

> **EXPERIMENTAL · trained on 100% SYNTHETIC data · NO clinical validity.**

- **model_version:** `{model.model_version}`
- **Architecture:** hierarchical CatBoost (LCPN). Binary root gate
  `dangerous vs peripheral` with `monotone_constraints=+1` on risk features
  (INV-9: raising a risk feature never lowers P(dangerous), hard guarantee of
  the pre-abstention gate). Children: central/cardiogenic, peripheral (5), BPPV (2).
- **Calibration:** temperature scaling of the final vector, T={m.temperature:.3f}
  (fit on the calibration split by NLL). **Abstention:** confidence gate
  τ={m.abstain_threshold:.3f} (target coverage {_TARGET_COVERAGE:.0%}) →
  `undetermined` as sentinel (INV-10).
- **Features:** {len(domain.features.feature_names)} ({len(domain.features.categorical_names)} categorical +
  {len(domain.features.numeric_feature_names)} numeric, incl. {len(domain.features.derived)} derived).
- **Risk features (monotone):** {", ".join(domain.features.risk_features)}.

## Metrics (synthetic generator recovery, NOT clinical)
Over {m.n_test} held-out synthetic cases (untouched test, evaluated once):

| Metric | Value |
|---|---|
| Leaf accuracy (8 classes) | {m.leaf_accuracy:.3f} |
| Gate accuracy (danger vs peripheral) | {m.gate_accuracy:.3f} |
| **Danger recall** (danger captured) | {m.danger_recall:.3f} |
| Raw ECE | {m.ece_raw:.4f} |
| Calibrated ECE (T={m.temperature:.2f}) | {m.ece_calibrated:.4f} |
| Abstention rate (→ undetermined) | {m.abstain_rate:.3f} |
| **Stress** — leaf accuracy with 40% of features missing | {m.stress_leaf_accuracy:.3f} |
| **Stress** — abstention rate with 40% missing | {m.stress_abstain_rate:.3f} |

> ⚠️ Known circularity risk: the label generates the features and the model
> re-learns them → these metrics do NOT imply clinical utility. They are
> "generator recovery". The calibrated ECE is REPORTED (not enforced as a test —
> temperature minimizes NLL, not ECE). Provisional priors pending specialist sign-off (T-CLIN).

## Limits (honesty)
- The ML **never** sets urgency (INV-11); it only contributes probabilistic
  evidence to the reasoner. The deterministic rails decide.
- Abstention is EVIDENCE (the ML can say "I don't know"); it does NOT escalate urgency.
- SHAP (if present) = local, non-causal attribution over the synthetic generator.
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
