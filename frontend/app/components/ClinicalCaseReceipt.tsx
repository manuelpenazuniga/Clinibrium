"use client";

import { useCallback, useState } from "react";
import type { AuditEvent, PipelineResult } from "@/lib/types";

const API_URL =
  process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

const URGENCY_LABELS: Record<string, string> = {
  inmediata: "Inmediata",
  prioritaria: "Prioritaria",
  ambulatoria: "Ambulatoria",
};

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

function jsonCanonical(obj: unknown): string {
  if (obj === null || typeof obj !== "object") {
    return JSON.stringify(obj);
  }
  if (Array.isArray(obj)) {
    return "[" + obj.map(jsonCanonical).join(",") + "]";
  }
  const keys = Object.keys(obj as Record<string, unknown>).sort();
  const parts = keys.map((k) => {
    const v = (obj as Record<string, unknown>)[k];
    return JSON.stringify(k) + ":" + jsonCanonical(v);
  });
  return "{" + parts.join(",") + "}";
}

async function computeSha256(text: string): Promise<string> {
  const encoder = new TextEncoder();
  const data = encoder.encode(text);
  const hashBuffer = await crypto.subtle.digest("SHA-256", data);
  const hashArray = Array.from(new Uint8Array(hashBuffer));
  return hashArray.map((b) => b.toString(16).padStart(2, "0")).join("");
}

export default function ClinicalCaseReceipt({
  result,
}: {
  result: PipelineResult;
}) {
  const [hashStatus, setHashStatus] = useState<
    "idle" | "verifying" | "match" | "mismatch" | "server-only"
  >("idle");
  const [decisionStatus, setDecisionStatus] = useState<
    "idle" | "submitting" | "accepted" | "rejected"
  >("idle");
  const [decisionReason, setDecisionReason] = useState("");
  const [decisionAuditEvent, setDecisionAuditEvent] = useState<AuditEvent | null>(null);

  const handleVerify = useCallback(async () => {
    if (!result.fhir_bundle || !result.bundle_sha256) return;
    setHashStatus("verifying");
    try {
      const canonical = jsonCanonical(result.fhir_bundle);
      const computed = await computeSha256(canonical);
      if (computed === result.bundle_sha256) {
        setHashStatus("match");
      } else {
        setHashStatus("server-only");
      }
    } catch {
      setHashStatus("server-only");
    }
  }, [result.fhir_bundle, result.bundle_sha256]);

  const handleDecision = useCallback(
    async (decision: "accept" | "reject") => {
      if (!result.audit_event_id) return;
      setDecisionStatus("submitting");
      try {
        const response = await fetch(`${API_URL}/api/decision`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            audit_event_id: result.audit_event_id,
            decision,
            reason: decisionReason || undefined,
          }),
        });
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        const auditEvent = (await response.json()) as AuditEvent;
        setDecisionAuditEvent(auditEvent);
        setDecisionStatus(decision === "accept" ? "accepted" : "rejected");
      } catch {
        setDecisionStatus("idle");
      }
    },
    [result.audit_event_id, decisionReason]
  );

  const handleDownloadFhir = useCallback(() => {
    if (!result.fhir_bundle) return;
    const blob = new Blob([JSON.stringify(result.fhir_bundle, null, 2)], {
      type: "application/json",
    });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `clinibrium-fhir-${result.case_id}.json`;
    a.click();
    URL.revokeObjectURL(url);
  }, [result.fhir_bundle, result.case_id]);

  const reasonerStatus = result.reasoning ? "ok" : "degraded";
  const modelUsed = result.reasoning?.model_used ?? "degradado";
  const groundingRefs = result.reasoning?.grounding_refs ?? [];
  const topDiagnosis = result.differential.candidates[0];

  return (
    <div className="clinical-receipt">
      <h3 className="receipt-title">Clinical Case Receipt</h3>

      <div className="receipt-section">
        <h4>Resumen clínico</h4>
        <div className="receipt-row">
          <span className="receipt-label">Urgencia:</span>
          <span className={`urgency-badge ${result.urgency}`}>
            {URGENCY_LABELS[result.urgency] ?? result.urgency}
          </span>
        </div>
        {result.forced_actions.length > 0 && (
          <div className="receipt-row">
            <span className="receipt-label">Acciones forzadas:</span>
            <span>{result.forced_actions.join(", ")}</span>
          </div>
        )}
        {result.applied_rails.length > 0 && (
          <div className="receipt-row">
            <span className="receipt-label">Rieles disparados:</span>
            <span>{result.applied_rails.join(", ")}</span>
          </div>
        )}
        {result.red_flag.red_flag_activa && result.red_flag.hits.length > 0 && (
          <div className="receipt-row">
            <span className="receipt-label">Red flags:</span>
            <span>
              {result.red_flag.hits.map((h) => `${h.id} (${h.label})`).join(", ")}
            </span>
          </div>
        )}
        {topDiagnosis && (
          <div className="receipt-row">
            <span className="receipt-label">Top diferencial:</span>
            <span>
              {DIAGNOSIS_LABELS[topDiagnosis.diagnosis] ?? topDiagnosis.diagnosis}
              {" "}({(topDiagnosis.score * 100).toFixed(0)}%)
            </span>
          </div>
        )}
      </div>

      <div className="receipt-section">
        <h4>Procedencia</h4>
        <div className="receipt-row">
          <span className="receipt-label">Modelo:</span>
          <span>{modelUsed}</span>
        </div>
        <div className="receipt-row">
          <span className="receipt-label">Reasoner:</span>
          <span className={reasonerStatus === "degraded" ? "reasoner-degraded" : ""}>
            {reasonerStatus}
          </span>
        </div>
        {groundingRefs.length > 0 && (
          <div className="receipt-row">
            <span className="receipt-label">Grounding:</span>
            <span>{groundingRefs.join(", ")}</span>
          </div>
        )}
      </div>

      {result.bundle_sha256 && (
        <div className="receipt-section">
          <h4>Integridad verificable</h4>
          <div className="receipt-row">
            <span className="receipt-label">SHA-256:</span>
            <code className="hash-display">{result.bundle_sha256}</code>
          </div>
          <div className="receipt-actions">
            <button
              className="btn-secondary"
              onClick={handleVerify}
              disabled={hashStatus === "verifying"}
              type="button"
            >
              {hashStatus === "verifying" ? (
                <>
                  <span className="spinner" /> Verificando...
                </>
              ) : (
                "Verificar"
              )}
            </button>
            {hashStatus === "match" && (
              <span className="hash-status hash-ok">✓ Íntegro (hash coincide)</span>
            )}
            {hashStatus === "mismatch" && (
              <span className="hash-status hash-fail">✗ Alterado</span>
            )}
            {hashStatus === "server-only" && (
              <span className="hash-status hash-info">
                Hash del servidor (canónico JS no coincide exacto — verificar con backend)
              </span>
            )}
          </div>
        </div>
      )}

      {result.audit_event_id && decisionStatus !== "accepted" && decisionStatus !== "rejected" && (
        <div className="receipt-section">
          <h4>Decisión del médico</h4>
          <p className="receipt-note">
            El médico decide — intervención humana (Ley 21.719).
          </p>
          <textarea
            className="receipt-textarea"
            placeholder="Justificación clínica (opcional)"
            value={decisionReason}
            onChange={(e) => setDecisionReason(e.target.value)}
            rows={2}
          />
          <div className="receipt-actions">
            <button
              className="btn-accept"
              onClick={() => handleDecision("accept")}
              disabled={decisionStatus === "submitting"}
              type="button"
            >
              {decisionStatus === "submitting" ? "Enviando..." : "Aceptar"}
            </button>
            <button
              className="btn-reject"
              onClick={() => handleDecision("reject")}
              disabled={decisionStatus === "submitting"}
              type="button"
            >
              {decisionStatus === "submitting" ? "Enviando..." : "Rechazar"}
            </button>
          </div>
        </div>
      )}

      {decisionAuditEvent && (
        <div className="receipt-section receipt-decision-recorded">
          <h4>Intervención registrada</h4>
          <div className="receipt-row">
            <span className="receipt-label">AuditEvent ID:</span>
            <code>{decisionAuditEvent.id}</code>
          </div>
          <div className="receipt-row">
            <span className="receipt-label">Tipo:</span>
            <span>{decisionAuditEvent.event_type}</span>
          </div>
          <div className="receipt-row">
            <span className="receipt-label">Registrado:</span>
            <span>{decisionAuditEvent.occurred_at}</span>
          </div>
        </div>
      )}

      {result.fhir_bundle && (
        <div className="receipt-section">
          <h4>Artefacto FHIR</h4>
          <button className="btn-secondary" onClick={handleDownloadFhir} type="button">
            Descargar .json
          </button>
        </div>
      )}
    </div>
  );
}
