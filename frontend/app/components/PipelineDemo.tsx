"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import type {
  CasePreset,
  PipelineResult,
  ReasonerOutput,
  StageName,
} from "@/lib/types";
import { CASE_PRESETS } from "@/lib/presets";
import { streamEvaluation } from "@/lib/api";
import {
  DIAGNOSIS_LABELS,
  FORCED_ACTION_LABELS,
  STAGE_ORDER,
  featureChips,
} from "@/lib/labels";
import ClinicalCaseReceipt from "./ClinicalCaseReceipt";
import PipelineRail from "./PipelineRail";
import Onboarding from "./Onboarding";
import WhatWouldChange from "./WhatWouldChange";

const TOUR_SEEN_KEY = "clinibrium.tour.demo.v1";

export default function PipelineDemo() {
  const [selectedPreset, setSelectedPreset] = useState<CasePreset | null>(null);
  const [isStreaming, setIsStreaming] = useState(false);
  const [completedStages, setCompletedStages] = useState<Set<StageName>>(
    new Set()
  );
  const [activeStage, setActiveStage] = useState<StageName | null>(null);
  const [stageData, setStageData] = useState<Record<string, unknown>>({});
  const [result, setResult] = useState<PipelineResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [killReasoner, setKillReasoner] = useState(false);
  const [tourOpen, setTourOpen] = useState(false);
  const abortRef = useRef<AbortController | null>(null);

  useEffect(() => {
    if (!window.localStorage.getItem(TOUR_SEEN_KEY)) {
      setTourOpen(true);
    }
  }, []);

  const closeTour = useCallback(() => {
    window.localStorage.setItem(TOUR_SEEN_KEY, "1");
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

  const handleEvaluate = useCallback(async () => {
    if (!selectedPreset) return;

    abortRef.current?.abort();
    const controller = new AbortController();
    abortRef.current = controller;

    setIsStreaming(true);
    resetRun();

    try {
      const final = await streamEvaluation(selectedPreset.features, {
        killReasoner,
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
        err instanceof Error ? err.message : "Error de conexión con el backend"
      );
    } finally {
      setIsStreaming(false);
      setActiveStage(null);
    }
  }, [selectedPreset, killReasoner, resetRun]);

  const hasResults = result !== null || Object.keys(stageData).length > 0;

  return (
    <div className="demo">
      <Onboarding open={tourOpen} onClose={closeTour} />

      <div className="demo-toolbar">
        <button
          type="button"
          className="btn-ghost"
          onClick={() => setTourOpen(true)}
        >
          Ver guía
        </button>
      </div>

      <section className="demo-step">
        <h2 className="step-title">
          <span className="step-marker">Paso 1</span> Elige un caso clínico
        </h2>
        <div className="preset-grid" data-tour="presets">
          {CASE_PRESETS.map((preset) => {
            const chips = featureChips(preset.features);
            const featureCount = Object.keys(preset.features).length;
            return (
              <button
                key={preset.id}
                className={`preset-card${selectedPreset?.id === preset.id ? " selected" : ""}`}
                onClick={() => handleSelectPreset(preset)}
                type="button"
                aria-pressed={selectedPreset?.id === preset.id}
              >
                <h3>{preset.name}</h3>
                <p>{preset.description}</p>
                <div className="preset-chips">
                  {chips.map((chip) => (
                    <span key={chip} className="chip">
                      {chip}
                    </span>
                  ))}
                </div>
                <span className="preset-meta">
                  {featureCount} features desidentificadas · allowlist INV-2
                </span>
              </button>
            );
          })}
        </div>

        <div className="evaluate-controls" data-tour="controls">
          <label className="switch" data-tour="kill">
            <input
              type="checkbox"
              checked={killReasoner}
              onChange={(e) => setKillReasoner(e.target.checked)}
            />
            <span className="switch-track" aria-hidden="true" />
            <span className="switch-label">Kill Claude — simular caída del razonador</span>
          </label>
          <button
            className="btn-primary"
            disabled={!selectedPreset || isStreaming}
            onClick={handleEvaluate}
            type="button"
          >
            {isStreaming ? (
              <>
                <span className="spinner" /> Evaluando…
              </>
            ) : (
              "Evaluar caso"
            )}
          </button>
        </div>

        {!selectedPreset && !hasResults && (
          <p className="empty-hint">
            Selecciona un caso para habilitar la evaluación. El pipeline corre
            de verdad contra el backend.
          </p>
        )}

        {killReasoner && (
          <div className="notice notice-degraded" role="status">
            <strong>Modo degradado activo.</strong> El razonador se simulará
            caído: la urgencia y las red flags permanecerán idénticas — la
            seguridad no depende del LLM (INV-8).
          </div>
        )}
      </section>

      {hasResults && (
        <section className="demo-step">
          <h2 className="step-title">
            <span className="step-marker">Paso 2</span> Pipeline en tiempo real
          </h2>
          <PipelineRail
            completed={completedStages}
            active={activeStage}
            hasError={error !== null}
          />

          <div className="stage-details">
            {STAGE_ORDER.filter(
              ({ key }) => key !== "done" && stageData[key]
            ).map(({ key, label }) => (
              <StageDetailCard key={key} label={label} stage={key} data={stageData[key]} />
            ))}
          </div>
        </section>
      )}

      {error && (
        <div className="notice notice-error" role="alert">
          <strong>Error:</strong> {error}
          <span className="notice-hint">
            ¿Está corriendo el backend en el puerto 8000? Revisa{" "}
            <code>uvicorn clinibrium.api:app --port 8000</code>.
          </span>
        </div>
      )}

      {result && (
        <section className="demo-step">
          <h2 className="step-title">
            <span className="step-marker">Paso 3</span> Recibo clínico y
            decisión
          </h2>

          {result.red_flag.red_flag_activa && (
            <div className="safety-banner" role="alert">
              <div className="safety-banner-head">
                <span className="safety-banner-kicker">Seguridad probada</span>
                <h3>Red flag activa ⇒ urgencia inmediata</h3>
              </div>
              <p>
                Este veredicto viene del RedFlagEngine determinista. Ni el ML
                ni Claude pueden anularlo — el riel se aplica después y gana
                siempre (INV-1).
              </p>
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

          {selectedPreset && (
            <WhatWouldChange features={selectedPreset.features} />
          )}
        </section>
      )}
    </div>
  );
}

function StageDetailCard({
  label,
  stage,
  data,
}: {
  label: string;
  stage: StageName;
  data: unknown;
}) {
  const renderContent = () => {
    if (stage === "redflag") {
      const d = data as { red_flag_activa: boolean; hits_count: number };
      return (
        <div>
          <strong>Red flag activa:</strong> {d.red_flag_activa ? "SÍ" : "No"}
          {(d.hits_count ?? 0) > 0 && (
            <span> — {d.hits_count} hallazgo(s) de alarma</span>
          )}
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
                    `${DIAGNOSIS_LABELS[c.diagnosis] ?? c.diagnosis}: ${(c.score * 100).toFixed(0)}%`
                )
                .join(" · ")
            : "Sin candidatos"}
        </div>
      );
    }
    if (stage === "ml") {
      const d = data as { available?: boolean };
      return (
        <div>
          {d.available
            ? "Modelo ML disponible"
            : "ML no disponible (track B degradado) — el pipeline continúa"}
        </div>
      );
    }
    if (stage === "reasoning") {
      const d = data as ReasonerOutput;
      if (!d?.model_used) {
        return (
          <div>
            Razonador degradado — el pipeline continúa; la urgencia no depende
            del LLM (INV-8).
          </div>
        );
      }
      return (
        <div>
          Modelo: <code>{d.model_used}</code>
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
            <span>Rieles: {d.applied_rails.join(", ")}</span>
          )}
          {d.forced_actions && d.forced_actions.length > 0 && (
            <span>
              {" "}
              · Acciones forzadas:{" "}
              {d.forced_actions
                .map((a) => FORCED_ACTION_LABELS[a] ?? a)
                .join(", ")}
            </span>
          )}
          {!d.applied_rails?.length && !d.forced_actions?.length && (
            <span>Sin rieles aplicados</span>
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

