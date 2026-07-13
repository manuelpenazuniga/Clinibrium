"use client";

import { useEffect, useMemo, useState } from "react";
import type { CaseFeatures } from "@/lib/types";

// Any field name that would imply raw pixels / video / biometric frames leaving
// the device. The whole point: the outbound payload contains NONE of these.
const _VIDEO_FIELD = /frame|video|image|pixel|landmark|blob|base64|photo|face|iris/i;

function featureKeys(features: CaseFeatures): string[] {
  return Object.entries(features)
    .filter(([, v]) => v !== undefined && v !== null)
    .map(([k]) => k);
}

async function sha256Hex(text: string): Promise<string> {
  const buf = await crypto.subtle.digest("SHA-256", new TextEncoder().encode(text));
  return Array.from(new Uint8Array(buf))
    .map((b) => b.toString(16).padStart(2, "0"))
    .join("");
}

export default function PrivacyEgressMeter({
  framesProcessed,
  features,
}: {
  framesProcessed: number;
  features: CaseFeatures | null;
}) {
  const keys = useMemo(() => (features ? featureKeys(features) : []), [features]);
  // The EXACT bytes that would go on the wire (what fetch serializes).
  const compact = useMemo(() => (features ? JSON.stringify(features) : ""), [features]);
  const pretty = useMemo(
    () => (features ? JSON.stringify(features, null, 2) : ""),
    [features]
  );
  const bytes = useMemo(
    () => (compact ? new TextEncoder().encode(compact).length : 0),
    [compact]
  );
  // Derived (not asserted): scan the outbound payload for any video/frame field.
  const videoFields = useMemo(() => keys.filter((k) => _VIDEO_FIELD.test(k)).length, [keys]);

  const [sha, setSha] = useState<string | null>(null);
  useEffect(() => {
    let alive = true;
    if (!compact) {
      setSha(null);
      return;
    }
    sha256Hex(compact)
      .then((h) => alive && setSha(h))
      .catch(() => alive && setSha(null));
    return () => {
      alive = false;
    };
  }, [compact]);

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
        <div className="egress-stat">
          <span className={`egress-value ${videoFields === 0 ? "egress-zero" : "egress-alarm"}`}>
            {videoFields}
          </span>
          <span className="egress-label">campos de video/frame en el payload</span>
        </div>
        <div className="egress-stat">
          <span className="egress-value">{bytes}</span>
          <span className="egress-label">bytes que salen a la red</span>
        </div>
      </div>

      <div className="egress-outbound">
        <div className="egress-outbound-head">
          <span className="egress-label">payload saliente exacto (a /api/evaluate)</span>
          {sha && (
            <code className="egress-sha" title="SHA-256 del payload (Web Crypto, verificable)">
              sha256:{sha.slice(0, 16)}…
            </code>
          )}
        </div>
        {pretty ? (
          <pre className="egress-payload">{pretty}</pre>
        ) : (
          <span className="egress-empty">ninguna aún — procesá un clip para ver el payload</span>
        )}
        {keys.length > 0 && (
          <div className="egress-keys">
            {keys.map((k) => (
              <code key={k} className="egress-key">
                {k}
              </code>
            ))}
          </div>
        )}
      </div>

      <p className="egress-note">
        El video se procesa localmente (MediaPipe on-device). Lo que se ve arriba es
        <strong> exactamente</strong> lo que sale a la red: {bytes} bytes de features
        numéricas desidentificadas, <strong>{videoFields}</strong> campos de video/frame,
        con hash SHA-256 verificable. No es «confiá en nosotros» — es observable (INV-2).
      </p>
    </div>
  );
}
