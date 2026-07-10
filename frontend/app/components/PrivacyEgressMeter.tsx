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
      <h4 className="egress-title">Privacy Egress Meter (INV-2)</h4>
      <div className="egress-stats">
        <div className="egress-stat">
          <span className="egress-label">Frames procesados localmente:</span>
          <span className="egress-value">{framesProcessed}</span>
        </div>
        <div className="egress-stat">
          <span className="egress-label">Frames subidos a la red:</span>
          <span className="egress-value egress-zero">0</span>
        </div>
        <div className="egress-stat">
          <span className="egress-label">Features enviadas:</span>
          <span className="egress-value">
            {keys.length > 0 ? keys.join(", ") : "(ninguna aún)"}
          </span>
        </div>
      </div>
      <p className="egress-note">
        Video procesado localmente; 0 frames a la red. Solo features numéricas
        desidentificadas se envían al backend.
      </p>
    </div>
  );
}
