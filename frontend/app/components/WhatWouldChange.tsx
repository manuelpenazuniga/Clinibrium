"use client";

import { useCallback, useState } from "react";
import type { CaseFeatures } from "@/lib/types";
import { uiErrorText, type UiError } from "@/lib/i18n";
import { useLanguage } from "./LanguageProvider";

const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

interface Counterfactual {
  feature: string;
  change: string;
  change_key?: string;
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

export default function WhatWouldChange({ features }: { features: CaseFeatures }) {
  const { lang, t } = useLanguage();
  const [result, setResult] = useState<WWCMResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<UiError | null>(null);

  const urgencyLabel = useCallback(
    (u: string): string =>
      (t.urgency as Record<string, string>)[u] ?? u,
    [t]
  );

  const run = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const r = await fetch(
        `${API_URL}/api/what-would-change?lang=${lang}`,
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(features),
        }
      );
      if (!r.ok) throw new Error(`HTTP ${r.status}: ${r.statusText}`);
      setResult((await r.json()) as WWCMResult);
    } catch (e) {
      setError(
        e instanceof Error ? { message: e.message } : { key: "connectionError" }
      );
    } finally {
      setLoading(false);
    }
  }, [features, lang]);

  return (
    <div className="wwcm" data-tour="wwcm">
      <div className="wwcm-intro">
        <h3>{t.wwc.heading}</h3>
        <p>{t.wwc.intro}</p>
      </div>

      <button
        className="btn-secondary"
        onClick={run}
        disabled={loading}
        type="button"
      >
        {loading ? (
          <>
            <span className="spinner" /> {t.wwc.analyzing}
          </>
        ) : (
          t.wwc.analyze
        )}
      </button>

      {error && (
        <div className="notice notice-error" role="alert">
          <strong>{t.common.error}</strong> {uiErrorText(error, t)}
        </div>
      )}

      {result && (
        <div className="wwcm-result">
          {result.minimal_escalation ? (
            <div className="wwcm-minimal" role="status">
              <span className="wwcm-minimal-kicker">{t.wwc.minimalKicker}</span>
              <p>
                <strong>{result.minimal_escalation.change}</strong>
                {t.wwc.leadsFrom}
                <span className={`urgency-badge ${result.minimal_escalation.base_urgency}`}>
                  {urgencyLabel(result.minimal_escalation.base_urgency)}
                </span>
                {t.wwc.to}
                <span className={`urgency-badge ${result.minimal_escalation.new_urgency}`}>
                  {urgencyLabel(result.minimal_escalation.new_urgency)}
                </span>
                {t.wwc.firesRail}
                <code>{result.minimal_escalation.rails_fired.join(", ")}</code>.
              </p>
            </div>
          ) : (
            <p className="wwcm-none">
              {t.wwc.noneChange}
              {urgencyLabel(result.base_urgency)}).
            </p>
          )}

          {result.counterfactuals.length > 0 && (
            <div className="wwcm-table-wrap">
              <table className="wwcm-table">
                <thead>
                  <tr>
                    <th>{t.wwc.tableFinding}</th>
                    <th>{t.wwc.tableUrgency}</th>
                    <th>{t.wwc.tableRail}</th>
                  </tr>
                </thead>
                <tbody>
                  {result.counterfactuals.map((c) => (
                    <tr
                      key={`${c.feature}:${c.change_key ?? c.change}`}
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
            {t.wwc.note.split(t.wwc.noteExactlyOne).map((part, i, arr) => (
              <span key={i}>
                {part}
                {i < arr.length - 1 && <strong>{t.wwc.noteExactlyOne}</strong>}
              </span>
            ))}
          </p>
        </div>
      )}
    </div>
  );
}
