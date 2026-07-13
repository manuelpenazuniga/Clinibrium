"use client";

import { useLanguage } from "./LanguageProvider";

/** Localized page header (eyebrow + heading + lede) for the /demo and
 * /dix-hallpike routes. The page-level Next `metadata` stays static (default
 * Spanish) since it is server-rendered head content. */
export default function PageHeader({ kind }: { kind: "demo" | "dix" }) {
  const { t } = useLanguage();
  const copy =
    kind === "demo"
      ? { eyebrow: t.demoPage.eyebrow, heading: t.demoPage.heading, lede: t.demoPage.lede }
      : { eyebrow: t.dix.pageEyebrow, heading: t.dix.pageHeading, lede: t.dix.pageLede };

  return (
    <header className="page-header">
      <p className="eyebrow">{copy.eyebrow}</p>
      <h1>{copy.heading}</h1>
      <p className="page-lede">{copy.lede}</p>
    </header>
  );
}
