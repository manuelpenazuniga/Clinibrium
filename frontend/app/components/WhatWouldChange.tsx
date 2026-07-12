"use client";

import { useCallback, useState } from "react";
import type { CaseFeatures } from "@/lib/types";
import { URGENCY_LABELS } from "@/lib/labels";

const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

interface Counterfactual {
  feature: string;
  change: string;
  base_urgency: string;
  new_urgency: string;
  urgency_changed: boolean;
  escalates: boolean;
  forced_actions_added: string[];
  rails_fired: string[];
}

interface WWCMResult {
  base_urgency: string;
  counterfactuals: Counterfactual[];
  minimal_escalation: Counterfactual | null;
}

function urgencyLabel(u: string): string {
  return (URGENCY_LABELS as Record<string, string>)[u] ?? u;
}

export default function WhatWouldChange({ features }: { features: CaseFeatures }) {
  const [result, setResult] = useState<WWCMResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const run = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const r = await fetch(`${API_URL}/api/what-would-change`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(features),
      });
      if (!r.ok) throw new Error(`HTTP ${r.status}: ${r.statusText}`);
      setResult((await r.json()) as WWCMResult);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Error de conexión con el backend");
    } finally {
      setLoading(false);
    }
  }, [features]);

  return (
    <div className="wwcm" data-tour="wwcm">
      <div className="wwcm-intro">
        <h3>¿Qué cambiaría el manejo?</h3>
        <p>
          Un solo hallazgo a la vez, corrido por el pipeline determinista real —
          qué buscar antes de tranquilizar al paciente. El LLM no decide qué es
          urgente; los rieles verifican cada contrafactual (INV-3).
        </p>
      </div>

      <button
        className="btn-secondary"
        onClick={run}
        disabled={loading}
        type="button"
      >
        {loading ? (
          <>
            <span className="spinner" /> Analizando contrafactuales…
          </>
        ) : (
          "Analizar: ¿qué cambiaría el manejo?"
        )}
      </button>

      {error && (
        <div className="notice notice-error" role="alert">
          <strong>Error:</strong> {error}
        </div>
      )}

      {result && (
        <div className="wwcm-result">
          {result.minimal_escalation ? (
            <div className="wwcm-minimal" role="status">
              <span className="wwcm-minimal-kicker">Cambio mínimo que escala</span>
              <p>
                <strong>{result.minimal_escalation.change}</strong> lleva el caso
                de{" "}
                <span className={`urgency-badge ${result.minimal_escalation.base_urgency}`}>
                  {urgencyLabel(result.minimal_escalation.base_urgency)}
                </span>{" "}
                a{" "}
                <span className={`urgency-badge ${result.minimal_escalation.new_urgency}`}>
                  {urgencyLabel(result.minimal_escalation.new_urgency)}
                </span>{" "}
                — dispara el riel{" "}
                <code>{result.minimal_escalation.rails_fired.join(", ")}</code>.
              </p>
            </div>
          ) : (
            <p className="wwcm-none">
              Ningún hallazgo único cambia el manejo (base:{" "}
              {urgencyLabel(result.base_urgency)}).
            </p>
          )}

          {result.counterfactuals.length > 0 && (
            <div className="wwcm-table-wrap">
              <table className="wwcm-table">
                <thead>
                  <tr>
                    <th>Hallazgo agregado (1 variable)</th>
                    <th>Urgencia</th>
                    <th>Riel</th>
                  </tr>
                </thead>
                <tbody>
                  {result.counterfactuals.map((c) => (
                    <tr
                      key={`${c.feature}:${c.change}`}
                      className={c.escalates ? "wwcm-escalates" : ""}
                    >
                      <td>{c.change}</td>
                      <td className="wwcm-urgency-cell">
                        <span className={`urgency-badge ${c.base_urgency}`}>
                          {urgencyLabel(c.base_urgency)}
                        </span>
                        <span className="wwcm-arrow" aria-hidden="true">
                          →
                        </span>
                        <span className={`urgency-badge ${c.new_urgency}`}>
                          {urgencyLabel(c.new_urgency)}
                        </span>
                      </td>
                      <td>
                        <code>{c.rails_fired.join(", ") || "—"}</code>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}

          <p className="wwcm-note">
            Cada fila cambia <strong>exactamente una</strong> variable y se corre
            por las capas deterministas (RedFlagEngine + rieles). Resultados
            verificables, no una opinión del modelo.
          </p>
        </div>
      )}
    </div>
  );
}
