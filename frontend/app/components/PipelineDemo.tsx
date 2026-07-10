"use client";

import { useCallback, useRef, useState } from "react";
import type {
  CaseFeatures,
  CasePreset,
  DifferentialResult,
  PipelineResult,
  PredictResponse,
  ReasonerOutput,
  RedFlagResult,
  StageEvent,
  StageName,
} from "@/lib/types";
import { CASE_PRESETS } from "@/lib/presets";

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
  const [showFhirModal, setShowFhirModal] = useState(false);
  const abortRef = useRef<AbortController | null>(null);

  const handleSelectPreset = useCallback((preset: CasePreset) => {
    setSelectedPreset(preset);
    setCompletedStages(new Set());
    setActiveStage(null);
    setStageData({});
    setResult(null);
    setError(null);
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

    try {
      const response = await fetch(`${API_URL}/api/evaluate`, {
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
  }, [selectedPreset]);

  const handleDownloadFhir = useCallback(() => {
    if (!result?.fhir_bundle) return;
    const blob = new Blob([JSON.stringify(result.fhir_bundle, null, 2)], {
      type: "application/json",
    });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `clinibrium-fhir-${result.case_id}.json`;
    a.click();
    URL.revokeObjectURL(url);
  }, [result]);

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

      <div style={{ marginBottom: "1.5rem" }}>
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

      {result && <ResultPanel result={result} onShowFhir={() => setShowFhirModal(true)} />}

      {showFhirModal && result?.fhir_bundle && (
        <FhirModal
          bundle={result.fhir_bundle}
          onClose={() => setShowFhirModal(false)}
          onDownload={handleDownloadFhir}
        />
      )}
    </div>
  );
}

function StageDetailCard({ label, data }: { label: string; data: unknown }) {
  const renderContent = () => {
    if (label === "RedFlag") {
      // Payload observacional del stage: {red_flag_activa, hits_count}.
      // El detalle de los hits va en el panel de resultado final (done).
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
      // Payload observacional del stage: {top_candidates: [{diagnosis, score, rule_ids}]}.
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
      // Payload observacional del stage: {available: bool}.
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

function ResultPanel({
  result,
  onShowFhir,
}: {
  result: PipelineResult;
  onShowFhir: () => void;
}) {
  const safetyActive =
    result.red_flag.red_flag_activa && result.urgency === "inmediata";

  return (
    <div className="result-panel">
      <h2 className="section-title" style={{ marginTop: 0 }}>
        3. Resultado
      </h2>

      <div style={{ marginBottom: "1rem" }}>
        <span className={`urgency-badge ${result.urgency}`}>
          {URGENCY_LABELS[result.urgency] ?? result.urgency}
        </span>
      </div>

      {safetyActive && (
        <div className="safety-banner">
          <h3>
            🔴 Riel de seguridad activo
          </h3>
          <p>
            Aunque el ML/LLM sugieran un cuadro benigno, el guardián
            determinista fuerza derivación inmediata porque hay una red flag
            activa.
          </p>
          {result.red_flag.hits.length > 0 && (
            <>
              <strong>Red flags detectadas:</strong>
              <ul>
                {result.red_flag.hits.map((h) => (
                  <li key={h.id}>
                    <strong>{h.id}</strong> — {h.label}
                  </li>
                ))}
              </ul>
            </>
          )}
          {result.applied_rails.length > 0 && (
            <p style={{ marginTop: "0.5rem" }}>
              <strong>Rieles aplicados:</strong>{" "}
              {result.applied_rails.join(", ")}
            </p>
          )}
        </div>
      )}

      {result.forced_actions.length > 0 && (
        <div className="result-section">
          <h4>Acciones forzadas</h4>
          <div className="forced-actions">
            {result.forced_actions.map((a) => (
              <span key={a} className="forced-action-tag">
                {a}
              </span>
            ))}
          </div>
        </div>
      )}

      <div className="result-section">
        <h4>Diagnóstico diferencial</h4>
        <ul className="candidate-list">
          {result.differential.candidates.map((c) => (
            <li key={c.diagnosis} className="candidate-item">
              <span className="candidate-name">
                {DIAGNOSIS_LABELS[c.diagnosis] ?? c.diagnosis}
              </span>
              <div className="candidate-bar">
                <div
                  className="candidate-bar-fill"
                  style={{ width: `${(c.score * 100).toFixed(0)}%` }}
                />
              </div>
              <span className="candidate-score">
                {(c.score * 100).toFixed(0)}%
              </span>
            </li>
          ))}
          {result.differential.candidates.length === 0 && (
            <li style={{ color: "var(--color-text-muted)", fontSize: "0.9rem" }}>
              Sin candidatos
            </li>
          )}
        </ul>
      </div>

      <div className="result-section">
        <h4>Razonamiento clínico</h4>
        {result.reasoning ? (
          <div className="reasoning-block">
            <p>{result.reasoning.explanation}</p>
            {result.reasoning.reconciliation && (
              <p>
                <strong>Conciliación:</strong>{" "}
                {result.reasoning.reconciliation}
              </p>
            )}
            <p style={{ fontSize: "0.8rem", color: "var(--color-text-muted)" }}>
              Modelo: {result.reasoning.model_used}
            </p>
          </div>
        ) : (
          <div className="degraded-banner">
            Razonador no disponible (degradado) — la seguridad no depende de él.
          </div>
        )}
      </div>

      {result.applied_rails.length > 0 && !safetyActive && (
        <div className="result-section">
          <h4>Rieles aplicados</h4>
          <div className="applied-rails">
            {result.applied_rails.map((r) => (
              <code key={r} style={{ marginRight: "0.5rem" }}>
                {r}
              </code>
            ))}
          </div>
        </div>
      )}

      {result.fhir_bundle && (
        <div className="result-section">
          <h4>Artefacto auditable</h4>
          <p style={{ fontSize: "0.85rem", color: "var(--color-text-muted)", margin: "0 0 0.75rem" }}>
            Bundle FHIR — cada decisión trazable. Formato de salida auditable.
          </p>
          <button className="btn-secondary" onClick={onShowFhir} type="button">
            Ver artefacto auditable (FHIR)
          </button>
        </div>
      )}
    </div>
  );
}

function FhirModal({
  bundle,
  onClose,
  onDownload,
}: {
  bundle: Record<string, unknown>;
  onClose: () => void;
  onDownload: () => void;
}) {
  return (
    <div className="modal-overlay" onClick={onClose} role="dialog">
      <div className="modal-content" onClick={(e) => e.stopPropagation()}>
        <div className="modal-header">
          <h3>Artefacto auditable FHIR</h3>
          <button className="close-btn" onClick={onClose} type="button">
            ×
          </button>
        </div>
        <div className="modal-body">
          <pre>{JSON.stringify(bundle, null, 2)}</pre>
        </div>
        <div className="modal-footer">
          <button className="btn-secondary" onClick={onClose} type="button">
            Cerrar
          </button>
          <button className="btn-primary" onClick={onDownload} type="button">
            Descargar .json
          </button>
        </div>
      </div>
    </div>
  );
}
