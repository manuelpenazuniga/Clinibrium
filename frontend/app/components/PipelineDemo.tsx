"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import type {
  CaseFeatures,
  CasePreset,
  PipelineResult,
  ReasonerOutput,
  StageName,
} from "@/lib/types";
import { CASE_PRESETS } from "@/lib/presets";
import { streamEvaluation } from "@/lib/api";
import { STAGE_ORDER } from "@/lib/labels";
import { featureChips, uiErrorText, type Dict, type UiError } from "@/lib/i18n";
import { useLanguage } from "./LanguageProvider";
import ClinicalCaseReceipt from "./ClinicalCaseReceipt";
import PipelineRail from "./PipelineRail";
import Onboarding from "./Onboarding";
import RealCaseForm from "./RealCaseForm";
import WelcomeOnboarding from "./WelcomeOnboarding";
import WhatWouldChange from "./WhatWouldChange";

const WELCOME_SEEN_KEY = "clinibrium.welcome.v1";

/** "presets" = guided demo cases; "real" = physician-entered real case. */
type CaseMode = "presets" | "real";

export default function PipelineDemo() {
  const { lang, t } = useLanguage();
  const [mode, setMode] = useState<CaseMode>("presets");
  const [selectedPreset, setSelectedPreset] = useState<CasePreset | null>(null);
  const [realFeatures, setRealFeatures] = useState<CaseFeatures>({});
  // Snapshot of what was actually evaluated — the receipt and counterfactuals
  // must refer to the sent payload, not to a form edited afterwards.
  const [evaluatedFeatures, setEvaluatedFeatures] =
    useState<CaseFeatures | null>(null);
  const [isStreaming, setIsStreaming] = useState(false);
  const [completedStages, setCompletedStages] = useState<Set<StageName>>(
    new Set()
  );
  const [activeStage, setActiveStage] = useState<StageName | null>(null);
  const [stageData, setStageData] = useState<Record<string, unknown>>({});
  const [result, setResult] = useState<PipelineResult | null>(null);
  const [error, setError] = useState<UiError | null>(null);
  const [killReasoner, setKillReasoner] = useState(false);
  const [tourOpen, setTourOpen] = useState(false);
  const [welcomeOpen, setWelcomeOpen] = useState(false);
  const abortRef = useRef<AbortController | null>(null);
  const pipelineSectionRef = useRef<HTMLElement | null>(null);

  // First run opens the full-screen welcome guide; the anchored spotlight
  // tour stays available on demand via "Ver guía".
  useEffect(() => {
    if (!window.localStorage.getItem(WELCOME_SEEN_KEY)) {
      setWelcomeOpen(true);
    }
  }, []);

  const closeWelcome = useCallback(() => {
    window.localStorage.setItem(WELCOME_SEEN_KEY, "1");
    setWelcomeOpen(false);
  }, []);

  const closeTour = useCallback(() => {
    setTourOpen(false);
  }, []);

  const resetRun = useCallback(() => {
    setCompletedStages(new Set());
    setActiveStage(null);
    setStageData({});
    setResult(null);
    setError(null);
  }, []);

  const handleSelectPreset = useCallback(
    (preset: CasePreset) => {
      setSelectedPreset(preset);
      resetRun();
    },
    [resetRun]
  );

  const handleSelectMode = useCallback(
    (next: CaseMode) => {
      if (next === mode) return;
      setMode(next);
      resetRun();
    },
    [mode, resetRun]
  );

  // The features the pipeline would evaluate right now, per mode. Null while
  // there is nothing to send (no preset picked / empty form).
  const activeFeatures: CaseFeatures | null =
    mode === "real"
      ? Object.keys(realFeatures).length > 0
        ? realFeatures
        : null
      : (selectedPreset?.features ?? null);

  const handleEvaluate = useCallback(async () => {
    if (!activeFeatures) return;

    abortRef.current?.abort();
    const controller = new AbortController();
    abortRef.current = controller;

    setIsStreaming(true);
    resetRun();
    setEvaluatedFeatures(activeFeatures);

    // Bring the live pipeline into view once it has rendered (next frame).
    requestAnimationFrame(() => {
      pipelineSectionRef.current?.scrollIntoView({
        behavior: window.matchMedia("(prefers-reduced-motion: reduce)").matches
          ? "auto"
          : "smooth",
        block: "start",
      });
    });

    try {
      const final = await streamEvaluation(activeFeatures, {
        killReasoner,
        lang,
        signal: controller.signal,
        onStage: (evt) => {
          if (evt.stage === "done") {
            setCompletedStages((prev) => new Set(prev).add("done"));
            setActiveStage(null);
          } else if (evt.stage !== "error") {
            setCompletedStages((prev) => new Set(prev).add(evt.stage));
            setActiveStage(evt.stage);
            setStageData((prev) => ({ ...prev, [evt.stage]: evt.data }));
          }
        },
      });
      setResult(final);
    } catch (err: unknown) {
      if (err instanceof DOMException && err.name === "AbortError") {
        return;
      }
      setError(
        err instanceof Error ? { message: err.message } : { key: "connectionError" }
      );
    } finally {
      setIsStreaming(false);
      setActiveStage(null);
    }
  }, [activeFeatures, killReasoner, lang, resetRun]);

  const hasResults = result !== null || Object.keys(stageData).length > 0;
  // Show the rail as soon as evaluation starts, with every stage pending —
  // not only after the first SSE event lands.
  const showPipeline = isStreaming || hasResults;

  return (
    <div className="demo">
      <WelcomeOnboarding open={welcomeOpen} onClose={closeWelcome} />
      <Onboarding open={tourOpen} onClose={closeTour} />

      <div className="demo-toolbar">
        <button
          type="button"
          className="btn-ghost"
          onClick={() => setTourOpen(true)}
        >
          {t.demo.seeGuide}
        </button>
      </div>

      <section className="demo-step">
        <h2 className="step-title">
          <span className="step-marker">{t.demo.step1}</span>{" "}
          {mode === "real" ? t.demo.step1TitleReal : t.demo.step1Title}
        </h2>

        <div className="case-mode-toggle" role="group" aria-label={t.demo.modeAria}>
          <button
            type="button"
            className={`case-mode-option${mode === "presets" ? " selected" : ""}`}
            aria-pressed={mode === "presets"}
            onClick={() => handleSelectMode("presets")}
          >
            {t.demo.modePresets}
          </button>
          <button
            type="button"
            className={`case-mode-option${mode === "real" ? " selected" : ""}`}
            aria-pressed={mode === "real"}
            onClick={() => handleSelectMode("real")}
          >
            {t.demo.modeReal}
          </button>
        </div>

        {mode === "presets" ? (
          <div className="preset-grid" data-tour="presets">
            {CASE_PRESETS.map((preset) => {
              const chips = featureChips(preset.features, t);
              const featureCount = Object.keys(preset.features).length;
              const presetCopy =
                t.presets[preset.id as keyof Dict["presets"]] ?? {
                  name: preset.name,
                  description: preset.description,
                };
              return (
                <button
                  key={preset.id}
                  className={`preset-card${selectedPreset?.id === preset.id ? " selected" : ""}`}
                  onClick={() => handleSelectPreset(preset)}
                  type="button"
                  aria-pressed={selectedPreset?.id === preset.id}
                >
                  <h3>{presetCopy.name}</h3>
                  <p>{presetCopy.description}</p>
                  <div className="preset-chips">
                    {chips.map((chip) => (
                      <span key={chip} className="chip">
                        {chip}
                      </span>
                    ))}
                  </div>
                  <span className="preset-meta">
                    {featureCount} {t.demo.presetFeatures}
                  </span>
                </button>
              );
            })}
          </div>
        ) : (
          <>
            <p className="rc-hint">{t.realCase.hint}</p>
            <RealCaseForm features={realFeatures} onChange={setRealFeatures} />
          </>
        )}

        <div className="evaluate-controls" data-tour="controls">
          <label className="switch" data-tour="kill">
            <input
              type="checkbox"
              checked={killReasoner}
              onChange={(e) => setKillReasoner(e.target.checked)}
            />
            <span className="switch-track" aria-hidden="true" />
            <span className="switch-label">{t.demo.killLabel}</span>
          </label>
          <button
            className="btn-primary"
            disabled={!activeFeatures || isStreaming}
            onClick={handleEvaluate}
            type="button"
          >
            {isStreaming ? (
              <>
                <span className="spinner" /> {t.common.spinnerEvaluating}
              </>
            ) : (
              t.demo.evaluate
            )}
          </button>
        </div>

        {!activeFeatures && !hasResults && (
          <p className="empty-hint">
            {mode === "real" ? t.realCase.noFeatures : t.demo.emptyHint}
          </p>
        )}

        {killReasoner && (
          <div className="notice notice-degraded" role="status">
            <strong>{t.demo.degradedTitle}</strong>
            {t.demo.degradedBody}
          </div>
        )}
      </section>

      {showPipeline && (
        <section className="demo-step" ref={pipelineSectionRef}>
          <h2 className="step-title">
            <span className="step-marker">{t.demo.step2}</span> {t.demo.step2Title}
          </h2>
          <PipelineRail
            completed={completedStages}
            active={activeStage}
            hasError={error !== null}
          />

          <div className="stage-details">
            {STAGE_ORDER.filter(
              ({ key }) => key !== "done" && stageData[key]
            ).map(({ key }) => (
              <StageDetailCard
                key={key}
                label={t.stages[key as keyof Dict["stages"]].label}
                stage={key}
                data={stageData[key]}
                t={t}
              />
            ))}
          </div>
        </section>
      )}

      {error && (
        <div className="notice notice-error" role="alert">
          <strong>{t.common.error}</strong> {uiErrorText(error, t)}
          <span className="notice-hint">
            {t.demo.backendHintPrefix}
            <code>uvicorn clinibrium.api:app --port 8000</code>.
          </span>
        </div>
      )}

      {result && (
        <section className="demo-step">
          <h2 className="step-title">
            <span className="step-marker">{t.demo.step3}</span> {t.demo.step3Title}
          </h2>

          {result.red_flag.red_flag_activa && (
            <div className="safety-banner" role="alert">
              <div className="safety-banner-head">
                <span className="safety-banner-kicker">{t.demo.safetyKicker}</span>
                <h3>{t.demo.safetyTitle}</h3>
              </div>
              <p>{t.demo.safetyBody}</p>
              <div className="safety-banner-hits">
                {result.red_flag.hits.map((hit) => (
                  <span key={hit.id} className="chip chip-danger">
                    {hit.id} · {hit.label}
                  </span>
                ))}
              </div>
            </div>
          )}

          <ClinicalCaseReceipt result={result} />

          {evaluatedFeatures && <WhatWouldChange features={evaluatedFeatures} />}
        </section>
      )}
    </div>
  );
}

function StageDetailCard({
  label,
  stage,
  data,
  t,
}: {
  label: string;
  stage: StageName;
  data: unknown;
  t: Dict;
}) {
  const s = t.demo.stage;
  const renderContent = () => {
    if (stage === "redflag") {
      const d = data as { red_flag_activa: boolean; hits_count: number };
      return (
        <div>
          <strong>{s.redflagActive}</strong> {d.red_flag_activa ? s.yes : s.no}
          {(d.hits_count ?? 0) > 0 && <span>{s.hitsSuffix(d.hits_count)}</span>}
        </div>
      );
    }
    if (stage === "differential") {
      const d = data as {
        top_candidates?: {
          diagnosis: string;
          score: number;
          rule_ids: string[];
        }[];
      };
      const cands = d.top_candidates ?? [];
      return (
        <div>
          {cands.length > 0
            ? cands
                .slice(0, 3)
                .map(
                  (c) =>
                    `${t.diagnosis[c.diagnosis as keyof Dict["diagnosis"]] ?? c.diagnosis}: ${(c.score * 100).toFixed(0)}%`
                )
                .join(" · ")
            : s.noCandidates}
        </div>
      );
    }
    if (stage === "ml") {
      const d = data as { available?: boolean };
      return <div>{d.available ? s.mlAvailable : s.mlUnavailable}</div>;
    }
    if (stage === "reasoning") {
      const d = data as ReasonerOutput;
      if (!d?.model_used) {
        return <div>{s.reasonerDegraded}</div>;
      }
      return (
        <div>
          {s.model} <code>{d.model_used}</code>
          {d.explanation && (
            <p className="stage-explanation">
              {d.explanation.slice(0, 200)}
              {d.explanation.length > 200 ? "…" : ""}
            </p>
          )}
        </div>
      );
    }
    if (stage === "rails") {
      const d = data as { applied_rails?: string[]; forced_actions?: string[] };
      return (
        <div>
          {d.applied_rails && d.applied_rails.length > 0 && (
            <span>
              {s.rails} {d.applied_rails.join(", ")}
            </span>
          )}
          {d.forced_actions && d.forced_actions.length > 0 && (
            <span>
              {" "}
              · {s.forcedActions}{" "}
              {d.forced_actions
                .map((a) => t.forcedAction[a as keyof Dict["forcedAction"]] ?? a)
                .join(", ")}
            </span>
          )}
          {!d.applied_rails?.length && !d.forced_actions?.length && (
            <span>{s.noRails}</span>
          )}
        </div>
      );
    }
    return <pre>{JSON.stringify(data, null, 2)}</pre>;
  };

  return (
    <div className="stage-detail">
      <h4>{label}</h4>
      {renderContent()}
    </div>
  );
}
