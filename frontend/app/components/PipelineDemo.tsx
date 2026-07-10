"use client";

import { useCallback, useRef, useState } from "react";
import type {
  CaseFeatures,
  CasePreset,
  PipelineResult,
  ReasonerOutput,
  StageEvent,
  StageName,
} from "@/lib/types";
import { CASE_PRESETS } from "@/lib/presets";
import ClinicalCaseReceipt from "./ClinicalCaseReceipt";

const API_URL =
  process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

const STAGE_ORDER: { key: StageName; label: string }[] = [
  { key: "redflag", label: "RedFlag" },
  { key: "differential", label: "Differential" },
  { key: "ml", label: "ML" },
  { key: "reasoning", label: "Reasoning" },
  { key: "rails", label: "Rails" },
  { key: "done", label: "Done" },
];

const DIAGNOSIS_LABELS: Record<string, string> = {
  bppv_posterior: "VPPB posterior",
  bppv_horizontal: "VPPB horizontal",
  meniere: "Ménière",
  vestibular_migraine: "Migraña vestibular",
  vestibular_neuritis: "Neuritis vestibular",
  labyrinthitis: "Laberintitis",
  central_suspected: "Central (sospecha)",
  cardiogenic_suspected: "Cardiogénico (sospecha)",
  undetermined: "Indeterminado",
};

const URGENCY_LABELS: Record<string, string> = {
  inmediata: "Inmediata",
  prioritaria: "Prioritaria",
  ambulatoria: "Ambulatoria",
};

function parseSSEEvents(
  buffer: string
): { events: StageEvent[]; remainder: string } {
  const events: StageEvent[] = [];
  let remaining = buffer;

  while (true) {
    const doubleNewline = remaining.indexOf("\n\n");
    if (doubleNewline === -1) break;

    const rawEvent = remaining.slice(0, doubleNewline);
    remaining = remaining.slice(doubleNewline + 2);

    let eventType = "";
    let dataStr = "";

    for (const line of rawEvent.split("\n")) {
      if (line.startsWith("event: ")) {
        eventType = line.slice(7).trim();
      } else if (line.startsWith("data: ")) {
        dataStr += line.slice(6);
      } else if (line.startsWith("data:")) {
        dataStr += line.slice(5);
      }
    }

    if (eventType && dataStr) {
      try {
        const data = JSON.parse(dataStr) as unknown;
        events.push({
          stage: eventType as StageName,
          data,
          timestamp: Date.now(),
        });
      } catch {
        // skip malformed JSON
      }
    }
  }

  return { events, remainder: remaining };
}

async function runEvaluation(
  features: CaseFeatures,
  killReasoner: boolean,
  signal: AbortSignal
): Promise<PipelineResult> {
  const url = killReasoner
    ? `${API_URL}/api/evaluate?debug_kill_reasoner=true`
    : `${API_URL}/api/evaluate`;

  const response = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(features),
    signal,
  });

  if (!response.ok) {
    throw new Error(`HTTP ${response.status}: ${response.statusText}`);
  }

  const reader = response.body?.getReader();
  if (!reader) {
    throw new Error("ReadableStream not supported");
  }

  const decoder = new TextDecoder();
  let buffer = "";
  let result: PipelineResult | null = null;

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });
    const { events, remainder } = parseSSEEvents(buffer);
    buffer = remainder;

    for (const evt of events) {
      if (evt.stage === "done") {
        result = evt.data as PipelineResult;
      } else if (evt.stage === "error") {
        const errData = evt.data as { error: string; message: string };
        throw new Error(`${errData.error}: ${errData.message}`);
      }
    }
  }

  if (!result) {
    throw new Error("No result received from pipeline");
  }
  return result;
}

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
  const [forkResult, setForkResult] = useState<{
    original: PipelineResult;
    forked: PipelineResult;
    changedFeature: string;
  } | null>(null);
  const [isForking, setIsForking] = useState(false);
  const abortRef = useRef<AbortController | null>(null);

  const handleSelectPreset = useCallback((preset: CasePreset) => {
    setSelectedPreset(preset);
    setCompletedStages(new Set());
    setActiveStage(null);
    setStageData({});
    setResult(null);
    setError(null);
    setForkResult(null);
  }, []);

  const handleEvaluate = useCallback(async () => {
    if (!selectedPreset) return;

    if (abortRef.current) {
      abortRef.current.abort();
    }
    const controller = new AbortController();
    abortRef.current = controller;

    setIsStreaming(true);
    setCompletedStages(new Set());
    setActiveStage(null);
    setStageData({});
    setResult(null);
    setError(null);
    setForkResult(null);

    try {
      const url = killReasoner
        ? `${API_URL}/api/evaluate?debug_kill_reasoner=true`
        : `${API_URL}/api/evaluate`;

      const response = await fetch(url, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(selectedPreset.features as CaseFeatures),
        signal: controller.signal,
      });

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }

      const reader = response.body?.getReader();
      if (!reader) {
        throw new Error("ReadableStream not supported");
      }

      const decoder = new TextDecoder();
      let buffer = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const { events, remainder } = parseSSEEvents(buffer);
        buffer = remainder;

        for (const evt of events) {
          if (evt.stage === "done") {
            setCompletedStages((prev) => new Set(prev).add("done"));
            setActiveStage(null);
            setResult(evt.data as PipelineResult);
          } else if (evt.stage === "error") {
            const errData = evt.data as { error: string; message: string };
            setError(`${errData.error}: ${errData.message}`);
            setActiveStage(null);
          } else {
            setCompletedStages((prev) => new Set(prev).add(evt.stage));
            setActiveStage(evt.stage);
            setStageData((prev) => ({ ...prev, [evt.stage]: evt.data }));
          }
        }
      }
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
  }, [selectedPreset, killReasoner]);

  const handleFork = useCallback(async () => {
    if (!selectedPreset || !result) return;

    setIsForking(true);
    setForkResult(null);

    const forkedFeatures: CaseFeatures = {
      ...selectedPreset.features,
      focal_signs: ["diplopia"],
      truncal_ataxia_severe: true,
    };

    const controller = new AbortController();

    try {
      const [originalResult, forkedResult] = await Promise.all([
        runEvaluation(selectedPreset.features, killReasoner, controller.signal),
        runEvaluation(forkedFeatures, killReasoner, controller.signal),
      ]);

      setForkResult({
        original: originalResult,
        forked: forkedResult,
        changedFeature: "focal_signs: [diplopia], truncal_ataxia_severe: true",
      });
    } catch (err: unknown) {
      if (err instanceof DOMException && err.name === "AbortError") return;
      setError(
        err instanceof Error ? err.message : "Error en fork clínico"
      );
    } finally {
      setIsForking(false);
    }
  }, [selectedPreset, result, killReasoner]);

  const hasResults = result !== null || Object.keys(stageData).length > 0;

  return (
    <div>
      <h2 className="section-title">1. Seleccioná un caso clínico</h2>
      <div className="preset-grid">
        {CASE_PRESETS.map((preset) => (
          <button
            key={preset.id}
            className={`preset-card${selectedPreset?.id === preset.id ? " selected" : ""}`}
            onClick={() => handleSelectPreset(preset)}
            type="button"
          >
            <h3>{preset.name}</h3>
            <p>{preset.description}</p>
          </button>
        ))}
      </div>

      <div className="evaluate-controls">
        <label className="kill-clause-toggle">
          <input
            type="checkbox"
            checked={killReasoner}
            onChange={(e) => setKillReasoner(e.target.checked)}
          />
          <span>Simular caída del razonador (Kill Claude)</span>
        </label>
        <button
          className="btn-primary"
          disabled={!selectedPreset || isStreaming}
          onClick={handleEvaluate}
          type="button"
        >
          {isStreaming ? (
            <>
              <span className="spinner" /> Evaluando...
            </>
          ) : (
            "Evaluar caso"
          )}
        </button>
      </div>

      {killReasoner && (
        <div className="kill-clause-info">
          La seguridad no depende del LLM — urgencia y red flags permanecen
          idénticas aunque el razonador degrade.
        </div>
      )}

      {hasResults && (
        <>
          <h2 className="section-title">2. Pipeline en tiempo real</h2>
          <div className="pipeline-stages">
            {STAGE_ORDER.map(({ key, label }) => {
              let cls = "stage-pill";
              if (completedStages.has(key)) cls += " completed";
              else if (activeStage === key) cls += " active";
              return (
                <span key={key} className={cls}>
                  {(activeStage === key ||
                    (key === "done" && completedStages.has("done"))) && (
                    <span className="spinner" />
                  )}
                  {completedStages.has(key) && key !== "done" && "✓"}
                  {label}
                </span>
              );
            })}
          </div>

          {STAGE_ORDER.filter(
            ({ key }) => key !== "done" && stageData[key]
          ).map(({ key, label }) => (
            <StageDetailCard
              key={key}
              label={label}
              data={stageData[key]}
            />
          ))}
        </>
      )}

      {error && (
        <div className="error-panel">
          <strong>Error:</strong> {error}
        </div>
      )}

      {result && (
        <>
          <ClinicalCaseReceipt result={result} />

          <div className="fork-section">
            <button
              className="btn-secondary"
              onClick={handleFork}
              disabled={isForking || !selectedPreset}
              type="button"
            >
              {isForking ? (
                <>
                  <span className="spinner" /> Forkeando...
                </>
              ) : (
                "Fork clínico: agregar signo focal"
              )}
            </button>
          </div>

          {forkResult && (
            <ClinicalForkDisplay
              original={forkResult.original}
              forked={forkResult.forked}
              changedFeature={forkResult.changedFeature}
            />
          )}
        </>
      )}
    </div>
  );
}

function StageDetailCard({ label, data }: { label: string; data: unknown }) {
  const renderContent = () => {
    if (label === "RedFlag") {
      const d = data as { red_flag_activa: boolean; hits_count: number };
      return (
        <div>
          <strong>Red flag activa:</strong>{" "}
          {d.red_flag_activa ? "SÍ" : "No"}
          {(d.hits_count ?? 0) > 0 && (
            <span> — {d.hits_count} hallazgo(s) de alarma</span>
          )}
        </div>
      );
    }
    if (label === "Differential") {
      const d = data as {
        top_candidates?: { diagnosis: string; score: number; rule_ids: string[] }[];
      };
      const cands = d.top_candidates ?? [];
      return (
        <div>
          {cands.length > 0
            ? cands
                .slice(0, 3)
                .map(
                  (c) =>
                    `${DIAGNOSIS_LABELS[c.diagnosis] ?? c.diagnosis}: ${(c.score * 100).toFixed(0)}%`,
                )
                .join(" · ")
            : "Sin candidatos"}
        </div>
      );
    }
    if (label === "ML") {
      const d = data as { available?: boolean };
      return (
        <div>
          {d.available
            ? "Modelo ML disponible"
            : "ML no disponible (track B degradado) — el pipeline continúa"}
        </div>
      );
    }
    if (label === "Reasoning") {
      const d = data as ReasonerOutput;
      return (
        <div>
          Modelo: {d.model_used}
          {d.explanation && (
            <p style={{ margin: "0.3rem 0 0", fontSize: "0.85rem" }}>
              {d.explanation.slice(0, 200)}
              {d.explanation.length > 200 ? "..." : ""}
            </p>
          )}
        </div>
      );
    }
    if (label === "Rails") {
      const d = data as { applied_rails?: string[]; forced_actions?: string[] };
      return (
        <div>
          {d.applied_rails && d.applied_rails.length > 0 && (
            <span>Rieles: {d.applied_rails.join(", ")}</span>
          )}
          {d.forced_actions && d.forced_actions.length > 0 && (
            <span>
              {" "}
              · Acciones forzadas: {d.forced_actions.join(", ")}
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

function ClinicalForkDisplay({
  original,
  forked,
  changedFeature,
}: {
  original: PipelineResult;
  forked: PipelineResult;
  changedFeature: string;
}) {
  const urgencyJumped = original.urgency !== forked.urgency;

  return (
    <div className="fork-display">
      <h3 className="fork-title">Clinical Fork — comparación lado a lado</h3>
      <p className="fork-changed">
        <strong>Feature cambiada:</strong> {changedFeature}
      </p>

      <div className="fork-grid">
        <div className="fork-card">
          <h4>Caso original</h4>
          <span className={`urgency-badge ${original.urgency}`}>
            {URGENCY_LABELS[original.urgency] ?? original.urgency}
          </span>
          {original.red_flag.red_flag_activa && (
            <div className="fork-flag">Red flag activa</div>
          )}
          {original.applied_rails.length > 0 && (
            <div className="fork-rails">
              Rieles: {original.applied_rails.join(", ")}
            </div>
          )}
          {original.differential.candidates[0] && (
            <div className="fork-top">
              Top:{" "}
              {DIAGNOSIS_LABELS[original.differential.candidates[0].diagnosis] ??
                original.differential.candidates[0].diagnosis}
            </div>
          )}
        </div>

        <div className="fork-arrow">→</div>

        <div className={`fork-card${urgencyJumped ? " fork-escalated" : ""}`}>
          <h4>Caso forkeado</h4>
          <span className={`urgency-badge ${forked.urgency}`}>
            {URGENCY_LABELS[forked.urgency] ?? forked.urgency}
          </span>
          {forked.red_flag.red_flag_activa && (
            <div className="fork-flag">Red flag activa</div>
          )}
          {forked.applied_rails.length > 0 && (
            <div className="fork-rails">
              Rieles: {forked.applied_rails.join(", ")}
            </div>
          )}
          {forked.differential.candidates[0] && (
            <div className="fork-top">
              Top:{" "}
              {DIAGNOSIS_LABELS[forked.differential.candidates[0].diagnosis] ??
                forked.differential.candidates[0].diagnosis}
            </div>
          )}
        </div>
      </div>

      {urgencyJumped && (
        <div className="fork-urgency-jump">
          Salto de urgencia: {URGENCY_LABELS[original.urgency] ?? original.urgency}
          {" → "}
          {URGENCY_LABELS[forked.urgency] ?? forked.urgency}. El riel disparó de
          verdad al cambiar una sola variable.
        </div>
      )}
    </div>
  );
}
