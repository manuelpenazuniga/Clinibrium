"use client";

import type { CaseFeatures } from "@/lib/types";

function featureKeys(features: CaseFeatures): string[] {
  return Object.entries(features)
    .filter(([, v]) => v !== undefined && v !== null)
    .map(([k]) => k);
}

export default function PrivacyEgressMeter({
  framesProcessed,
  features,
}: {
  framesProcessed: number;
  features: CaseFeatures | null;
}) {
  const keys = features ? featureKeys(features) : [];

  return (
    <div className="privacy-egress-meter">
      <div className="egress-head">
        <h4 className="egress-title">Privacy Egress Meter</h4>
        <span className="egress-invariant">INV-2</span>
      </div>
      <div className="egress-stats">
        <div className="egress-stat">
          <span className="egress-value">{framesProcessed}</span>
          <span className="egress-label">frames procesados localmente</span>
        </div>
        <div className="egress-stat">
          <span className="egress-value egress-zero">0</span>
          <span className="egress-label">frames subidos a la red</span>
        </div>
        <div className="egress-stat egress-stat-wide">
          <span className="egress-label">features enviadas</span>
          {keys.length > 0 ? (
            <div className="egress-keys">
              {keys.map((k) => (
                <code key={k} className="egress-key">
                  {k}
                </code>
              ))}
            </div>
          ) : (
            <span className="egress-empty">ninguna aún</span>
          )}
        </div>
      </div>
      <p className="egress-note">
        Video procesado localmente; 0 frames a la red. Solo features numéricas
        desidentificadas se envían al backend.
      </p>
    </div>
  );
}
