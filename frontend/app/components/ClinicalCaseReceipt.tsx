"use client";

import { useCallback, useState } from "react";
import type { AuditEvent, PipelineResult } from "@/lib/types";
import { API_URL } from "@/lib/api";
import type { Dict } from "@/lib/i18n";
import { useLanguage } from "./LanguageProvider";

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
  const { lang, t } = useLanguage();
  const r = t.receipt;
  const [hashStatus, setHashStatus] = useState<
    "idle" | "verifying" | "match" | "mismatch" | "server-only"
  >("idle");
  const [decisionStatus, setDecisionStatus] = useState<
    "idle" | "submitting" | "accepted" | "rejected" | "failed"
  >("idle");
  const [decisionReason, setDecisionReason] = useState("");
  const [decisionAuditEvent, setDecisionAuditEvent] =
    useState<AuditEvent | null>(null);

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
            lang,
          }),
        });
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        const auditEvent = (await response.json()) as AuditEvent;
        setDecisionAuditEvent(auditEvent);
        setDecisionStatus(decision === "accept" ? "accepted" : "rejected");
      } catch {
        setDecisionStatus("failed");
      }
    },
    [result.audit_event_id, decisionReason, lang]
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

  const reasoning = result.reasoning;
  const reasonerStatus = reasoning ? "ok" : "degraded";
  const groundingRefs = reasoning?.grounding_refs ?? [];
  const topCandidates = result.differential.candidates.slice(0, 3);

  return (
    <div className="clinical-receipt">
      <div className="receipt-head">
        <span className="receipt-kicker">{r.kicker}</span>
        <code className="receipt-case-id">{result.case_id}</code>
      </div>

      <div className="receipt-urgency-row">
        <span className={`urgency-badge urgency-lg ${result.urgency}`}>
          {t.urgency[result.urgency] ?? result.urgency}
        </span>
        {result.forced_actions.length > 0 && (
          <div className="receipt-forced-actions">
            {result.forced_actions.map((action) => (
              <span key={action} className="chip chip-danger">
                {t.forcedAction[action as keyof Dict["forcedAction"]] ?? action}
              </span>
            ))}
          </div>
        )}
      </div>

      <div className="receipt-section">
        <h4>{r.summaryHeading}</h4>
        {result.applied_rails.length > 0 && (
          <div className="receipt-row">
            <span className="receipt-label">{r.railsFired}</span>
            <span className="receipt-value">
              {result.applied_rails.map((rail) => (
                <code key={rail} className="rail-code">
                  {rail}
                </code>
              ))}
            </span>
          </div>
        )}
        {result.red_flag.red_flag_activa && result.red_flag.hits.length > 0 && (
          <div className="receipt-row">
            <span className="receipt-label">{r.redFlags}</span>
            <span className="receipt-value">
              {result.red_flag.hits
                .map((h) => `${h.id} (${h.label})`)
                .join(", ")}
            </span>
          </div>
        )}
        {topCandidates.length > 0 && (
          <div className="receipt-candidates">
            <span className="receipt-label">{r.differential}</span>
            <ul className="candidate-list">
              {topCandidates.map((c) => (
                <li key={c.diagnosis} className="candidate-item">
                  <span className="candidate-name">
                    {t.diagnosis[c.diagnosis as keyof Dict["diagnosis"]] ?? c.diagnosis}
                  </span>
                  <span className="candidate-bar">
                    <span
                      className="candidate-bar-fill"
                      style={{ width: `${Math.round(c.score * 100)}%` }}
                    />
                  </span>
                  <span className="candidate-score">
                    {(c.score * 100).toFixed(0)}%
                  </span>
                </li>
              ))}
            </ul>
          </div>
        )}
      </div>

      <div className="receipt-section">
        <h4>{r.reasoningHeading}</h4>
        {reasoning ? (
          <>
            {reasoning.explanation && (
              <p className="receipt-prose">{reasoning.explanation}</p>
            )}
            {reasoning.reconciliation && (
              <p className="receipt-prose receipt-prose-secondary">
                <strong>{r.reconciliation}</strong> {reasoning.reconciliation}
              </p>
            )}
            {reasoning.suggested_next_steps.length > 0 && (
              <div className="receipt-row">
                <span className="receipt-label">{r.nextSteps}</span>
                <span className="receipt-value">
                  {reasoning.suggested_next_steps.join(" · ")}
                </span>
              </div>
            )}
          </>
        ) : (
          <p className="receipt-prose receipt-degraded-note">
            {r.reasonerDegradedNote}
          </p>
        )}
      </div>

      <div className="receipt-section">
        <h4>{r.provenanceHeading}</h4>
        <div className="receipt-row">
          <span className="receipt-label">{r.model}</span>
          <span className="receipt-value">
            <code>{reasoning?.model_used ?? "—"}</code>
          </span>
        </div>
        <div className="receipt-row">
          <span className="receipt-label">{r.reasoner}</span>
          <span
            className={`status-chip ${reasonerStatus === "degraded" ? "status-degraded" : "status-ok"}`}
          >
            {reasonerStatus}
          </span>
        </div>
        {groundingRefs.length > 0 && (
          <div className="receipt-row">
            <span className="receipt-label">{r.grounding}</span>
            <span className="receipt-value receipt-grounding">
              {groundingRefs.join(", ")}
              {/* The citations stay Spanish (ICVD corpus). Only in EN mode do we
                  flag that with a small note — es mode is byte-identical. */}
              {lang === "en" && (
                <span className="receipt-grounding-note"> · {r.groundingNote}</span>
              )}
            </span>
          </div>
        )}
      </div>

      {result.bundle_sha256 && (
        <div className="receipt-section">
          <h4>{r.integrityHeading}</h4>
          <div className="receipt-row">
            <span className="receipt-label">SHA-256</span>
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
                  <span className="spinner" /> {r.verifying}
                </>
              ) : (
                r.verify
              )}
            </button>
            {hashStatus === "match" && (
              <span className="hash-status hash-ok">{r.hashOk}</span>
            )}
            {hashStatus === "mismatch" && (
              <span className="hash-status hash-fail">{r.hashFail}</span>
            )}
            {hashStatus === "server-only" && (
              <span className="hash-status hash-info">{r.hashInfo}</span>
            )}
          </div>
        </div>
      )}

      {result.audit_event_id &&
        decisionStatus !== "accepted" &&
        decisionStatus !== "rejected" && (
          <div className="receipt-section">
            <h4>{r.decisionHeading}</h4>
            <p className="receipt-note">{r.decisionNote}</p>
            <textarea
              className="receipt-textarea"
              placeholder={r.decisionPlaceholder}
              value={decisionReason}
              onChange={(e) => setDecisionReason(e.target.value)}
              rows={2}
            />
            {decisionStatus === "failed" && (
              <p className="receipt-note receipt-error-note" role="alert">
                {r.decisionFailed}
              </p>
            )}
            <div className="receipt-actions">
              <button
                className="btn-accept"
                onClick={() => handleDecision("accept")}
                disabled={decisionStatus === "submitting"}
                type="button"
              >
                {decisionStatus === "submitting" ? r.registering : r.accept}
              </button>
              <button
                className="btn-reject"
                onClick={() => handleDecision("reject")}
                disabled={decisionStatus === "submitting"}
                type="button"
              >
                {decisionStatus === "submitting" ? r.registering : r.reject}
              </button>
            </div>
          </div>
        )}

      {decisionAuditEvent && (
        <div className="receipt-section receipt-decision-recorded">
          <h4>{r.interventionHeading}</h4>
          <div className="receipt-row">
            <span className="receipt-label">{r.auditEvent}</span>
            <code className="receipt-value">{decisionAuditEvent.id}</code>
          </div>
          <div className="receipt-row">
            <span className="receipt-label">{r.type}</span>
            <span className="receipt-value">
              {decisionAuditEvent.event_type}
            </span>
          </div>
          <div className="receipt-row">
            <span className="receipt-label">{r.registered}</span>
            <span className="receipt-value">
              {decisionAuditEvent.occurred_at}
            </span>
          </div>
        </div>
      )}

      {result.fhir_bundle && (
        <div className="receipt-section receipt-footer">
          <button
            className="btn-secondary"
            onClick={handleDownloadFhir}
            type="button"
          >
            {r.downloadFhir}
          </button>
          <span className="receipt-footnote">{r.fhirFootnote}</span>
        </div>
      )}
    </div>
  );
}
