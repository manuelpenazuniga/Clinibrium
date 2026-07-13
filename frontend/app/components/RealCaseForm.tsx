"use client";

import { useCallback, useId, type Dispatch, type SetStateAction } from "react";
import type { CaseFeatures, FocalSign, VascularRiskFactor } from "@/lib/types";
import type { Dict } from "@/lib/i18n";
import { useLanguage } from "./LanguageProvider";

/**
 * Structured intake for a REAL case: builds a `CaseFeatures` payload field by
 * field. Unset fields are OMITTED (never sent) — the backend fills safe
 * defaults and validates fail-closed against the INV-2 allowlist. There is no
 * free-text input by design: nothing identifiable can be typed here.
 */

// Keys of `realCase.checks` are exactly the boolean CaseFeatures fields.
type BoolKey = keyof Dict["realCase"]["checks"];

const EXAM_CHECKS: BoolKey[] = [
  "nystagmus_direction_changing_gaze",
  "nystagmus_fatigable",
  "nystagmus_suppressed_by_fixation",
  "skew_deviation",
  "torsion_confirmed_by_clinician",
  "truncal_ataxia_severe",
];

const HEARING_CHECKS: BoolKey[] = ["tinnitus", "aural_fullness"];

const ALARM_CHECKS: BoolKey[] = [
  "headache_neck_pain_sudden_severe",
  "fever",
  "neck_stiffness",
  "altered_consciousness",
  "presyncope_syncope",
  "palpitations",
  "chest_pain",
  "otitis_mastoiditis",
  "recent_head_neck_trauma",
  "cervical_pathology",
  "known_carotid_vertebrobasilar_disease",
  "cardiovascular_instability",
  "migrainous_features",
  "worsening_during_flow",
];

export default function RealCaseForm({
  features,
  onChange,
}: {
  features: CaseFeatures;
  /** setState dispatcher: updates MUST be functional (prev-based) so rapid
   *  consecutive changes (autofill, fast clicking) never lose a field. */
  onChange: Dispatch<SetStateAction<CaseFeatures>>;
}) {
  const { t } = useLanguage();
  const rc = t.realCase;
  const formId = useId();

  const set = useCallback(
    <K extends keyof CaseFeatures>(
      key: K,
      value: CaseFeatures[K] | undefined
    ) => {
      onChange((prev) => {
        const next = { ...prev };
        if (value === undefined) {
          delete next[key];
        } else {
          next[key] = value;
        }
        return next;
      });
    },
    [onChange]
  );

  const toggleBool = useCallback(
    (key: BoolKey, checked: boolean) => set(key, checked ? true : undefined),
    [set]
  );

  const toggleFocalSign = useCallback(
    (sign: FocalSign, checked: boolean) => {
      onChange((prev) => {
        const cur = new Set(prev.focal_signs ?? []);
        if (checked) cur.add(sign);
        else cur.delete(sign);
        const next = { ...prev };
        if (cur.size > 0) next.focal_signs = [...cur];
        else delete next.focal_signs;
        return next;
      });
    },
    [onChange]
  );

  const toggleVascularRisk = useCallback(
    (factor: VascularRiskFactor, checked: boolean) => {
      onChange((prev) => {
        const cur = new Set(prev.vascular_risk_factors ?? []);
        if (checked) cur.add(factor);
        else cur.delete(factor);
        const next = { ...prev };
        if (cur.size > 0) next.vascular_risk_factors = [...cur];
        else delete next.vascular_risk_factors;
        return next;
      });
    },
    [onChange]
  );

  const featureCount = Object.keys(features).length;

  const select = <K extends keyof CaseFeatures>(
    key: K,
    label: string,
    options: Record<string, string>
  ) => (
    <div className="rc-field">
      <label htmlFor={`${formId}-${key}`}>{label}</label>
      <select
        id={`${formId}-${key}`}
        value={(features[key] as string | undefined) ?? ""}
        onChange={(e) =>
          set(
            key,
            e.target.value === ""
              ? undefined
              : (e.target.value as CaseFeatures[K])
          )
        }
      >
        <option value="">{rc.notEvaluated}</option>
        {Object.entries(options).map(([value, text]) => (
          <option key={value} value={value}>
            {text}
          </option>
        ))}
      </select>
    </div>
  );

  const number = (
    key: "age_years" | "episode_count" | "nystagmus_latency_s" | "nystagmus_duration_s",
    label: string,
    opts: { min: number; max: number; step?: number }
  ) => (
    <div className="rc-field">
      <label htmlFor={`${formId}-${key}`}>{label}</label>
      <input
        id={`${formId}-${key}`}
        type="number"
        inputMode="numeric"
        min={opts.min}
        max={opts.max}
        step={opts.step ?? 1}
        value={features[key] ?? ""}
        onChange={(e) =>
          set(key, e.target.value === "" ? undefined : Number(e.target.value))
        }
      />
    </div>
  );

  const check = (key: BoolKey) => (
    <label key={key} className="rc-check">
      <input
        type="checkbox"
        checked={features[key] === true}
        onChange={(e) => toggleBool(key, e.target.checked)}
      />
      {rc.checks[key]}
    </label>
  );

  return (
    <div className="rc-form">
      <section className="rc-section">
        <h3>{rc.sections.history}</h3>
        <div className="rc-grid">
          {number("age_years", rc.fields.age, { min: 0, max: 120 })}
          {select("duration", rc.fields.duration, t.durationChip)}
          {select("onset", rc.fields.onset, rc.onsetOpt)}
          {select("trigger", rc.fields.trigger, t.triggerChip)}
          {select("timing_pattern", rc.fields.timingPattern, rc.timingOpt)}
          {number("episode_count", rc.fields.episodeCount, { min: 1, max: 999 })}
          {select("episode_duration", rc.fields.episodeDuration, t.durationChip)}
        </div>
      </section>

      <section className="rc-section">
        <h3>{rc.sections.exam}</h3>
        <div className="rc-grid">
          {select("nystagmus_direction", rc.fields.nystagmusDirection, rc.nystagmusOpt)}
          {select("head_impulse", rc.fields.headImpulse, rc.hitOpt)}
          {select("dix_hallpike", rc.fields.dixHallpike, rc.dixOpt)}
          {number("nystagmus_latency_s", rc.fields.nystagmusLatency, {
            min: 0,
            max: 120,
            step: 0.5,
          })}
          {number("nystagmus_duration_s", rc.fields.nystagmusDuration, {
            min: 0,
            max: 600,
            step: 0.5,
          })}
        </div>
        <div className="rc-check-grid">{EXAM_CHECKS.map(check)}</div>
      </section>

      <section className="rc-section">
        <h3>{rc.sections.hearing}</h3>
        <div className="rc-grid">
          {select("hearing_loss", rc.fields.hearingLoss, rc.hearingOpt)}
        </div>
        <div className="rc-check-grid">{HEARING_CHECKS.map(check)}</div>
      </section>

      <section className="rc-section">
        <h3>{rc.sections.alarm}</h3>
        <div className="rc-check-grid">{ALARM_CHECKS.map(check)}</div>

        <p className="rc-subhead">{rc.fields.focalSigns}</p>
        <div className="rc-check-grid">
          {(Object.keys(rc.focalOpt) as FocalSign[]).map((sign) => (
            <label key={sign} className="rc-check">
              <input
                type="checkbox"
                checked={features.focal_signs?.includes(sign) ?? false}
                onChange={(e) => toggleFocalSign(sign, e.target.checked)}
              />
              {rc.focalOpt[sign]}
            </label>
          ))}
        </div>

        <p className="rc-subhead">{rc.fields.vascularRisk}</p>
        <div className="rc-check-grid">
          {(Object.keys(rc.vascularOpt) as VascularRiskFactor[]).map((factor) => (
            <label key={factor} className="rc-check">
              <input
                type="checkbox"
                checked={features.vascular_risk_factors?.includes(factor) ?? false}
                onChange={(e) => toggleVascularRisk(factor, e.target.checked)}
              />
              {rc.vascularOpt[factor]}
            </label>
          ))}
        </div>
      </section>

      <div className="rc-footer">
        <span className="rc-count" aria-live="polite">
          {rc.featureCount(featureCount)}
        </span>
        <button
          type="button"
          className="btn-ghost"
          disabled={featureCount === 0}
          onClick={() => onChange({})}
        >
          {rc.clear}
        </button>
      </div>

      {featureCount > 0 && (
        <details className="rc-payload">
          <summary>{rc.payloadSummary}</summary>
          <pre>{JSON.stringify(features, null, 2)}</pre>
        </details>
      )}
    </div>
  );
}
