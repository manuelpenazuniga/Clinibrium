"use client";

import { useLanguage } from "./LanguageProvider";
import type { Lang } from "@/lib/i18n";

const ORDER: Lang[] = ["es", "en"];

/**
 * Prominent segmented language selector ("Español / English", full words).
 *
 * The active pill is a DUPLICATED layer (brand background + white text)
 * clipped with `clip-path: inset(... round 999px)`. Animating the clip —
 * instead of crossfading colors on each button — slides one seamless surface
 * between the options, so text and background never de-sync mid-transition.
 * The overlay is `aria-hidden` + `pointer-events: none`; the real buttons
 * underneath keep focus, click and accessible state.
 */
export default function LanguageSwitch({ compact = false }: { compact?: boolean }) {
  const { lang, setLang, t } = useLanguage();
  const labels: Record<Lang, string> = {
    es: t.langSelect.es,
    en: t.langSelect.en,
  };

  const clipPath =
    lang === "es" ? "inset(0 50% 0 0 round 999px)" : "inset(0 0 0 50% round 999px)";

  return (
    <div
      className={`lang-switch${compact ? " compact" : ""}`}
      role="group"
      aria-label={t.langSelect.aria}
    >
      {ORDER.map((value) => (
        <button
          key={value}
          type="button"
          className="lang-switch-option"
          aria-pressed={lang === value}
          onClick={() => setLang(value)}
        >
          {labels[value]}
        </button>
      ))}
      <div className="lang-switch-thumb" style={{ clipPath }} aria-hidden="true">
        {ORDER.map((value) => (
          <span key={value}>{labels[value]}</span>
        ))}
      </div>
    </div>
  );
}
