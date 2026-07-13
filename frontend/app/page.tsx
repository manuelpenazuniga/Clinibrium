"use client";

import Link from "next/link";
import NystagmusTrace from "./components/NystagmusTrace";
import LanguageSwitch from "./components/LanguageSwitch";
import { useLanguage } from "./components/LanguageProvider";

// Language-independent structure paired with localized copy by index.
const PROP_META = [
  { number: "P1", invariant: "INV-1" },
  { number: "P2", invariant: "INV-6 · INV-8" },
  { number: "P3", invariant: "INV-4" },
];
const NODE_KINDS = [
  "input",
  "deterministic",
  "deterministic",
  "additive",
  "additive",
  "seal",
  "human",
];
const STAT_VALUES = ["0", "allowlist", "1", "SHA-256"];

export default function LandingPage() {
  const { t } = useLanguage();
  const L = t.landing;

  return (
    <main className="landing">
      <section className="hero">
        <div className="container">
          <div className="hero-lang">
            <LanguageSwitch />
          </div>
          <p className="eyebrow">{L.heroEyebrow}</p>
          <h1 className="hero-title">
            {L.heroTitle1}
            <br />
            {L.heroTitle2}
            <br />
            <em>{L.heroTitle3}</em>
          </h1>
          <p className="hero-lede">{L.heroLede}</p>
          <div className="hero-actions">
            <Link href="/demo" className="btn-primary btn-lg">
              {L.heroCtaDemo}
            </Link>
            <Link href="/dix-hallpike" className="btn-secondary btn-lg">
              {L.heroCtaDix}
            </Link>
          </div>
        </div>
        <figure className="hero-trace">
          <NystagmusTrace />
          <figcaption>{L.heroCaption}</figcaption>
        </figure>
      </section>

      <section className="landing-section">
        <div className="container">
          <p className="eyebrow">{L.propsEyebrow}</p>
          <h2 className="section-heading">{L.propsHeading}</h2>
          <div className="property-grid">
            {L.properties.map((p, i) => (
              <article key={PROP_META[i].number} className="property-card">
                <div className="property-head">
                  <span className="property-number">{PROP_META[i].number}</span>
                  <span className="property-invariant">{PROP_META[i].invariant}</span>
                </div>
                <h3>{p.title}</h3>
                <p>{p.body}</p>
              </article>
            ))}
          </div>
        </div>
      </section>

      <section className="landing-section landing-section-alt">
        <div className="container">
          <p className="eyebrow">{L.archEyebrow}</p>
          <h2 className="section-heading">{L.archHeading}</h2>
          <p className="section-lede">{L.archLede}</p>
          <ol className="rail-diagram">
            {L.nodes.map((node, i) => (
              <li key={node.name} className={`rail-node rail-${NODE_KINDS[i]}`}>
                <span className="rail-tag">{node.tag}</span>
                <span className="rail-name">{node.name}</span>
                <span className="rail-note">{node.note}</span>
              </li>
            ))}
          </ol>
        </div>
      </section>

      <section className="landing-section">
        <div className="container">
          <p className="eyebrow">{L.privacyEyebrow}</p>
          <h2 className="section-heading">{L.privacyHeading}</h2>
          <div className="privacy-grid">
            {L.privacyStats.map((stat, i) => (
              <div key={stat.label} className="privacy-stat">
                <span className="privacy-value">{STAT_VALUES[i]}</span>
                <span className="privacy-label">{stat.label}</span>
                <p className="privacy-note">{stat.note}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      <section className="landing-section landing-section-alt">
        <div className="container">
          <p className="eyebrow">{L.honestEyebrow}</p>
          <h2 className="section-heading">{L.honestHeading}</h2>
          <ul className="limitations-list">
            {L.limitations.map((item) => (
              <li key={item}>{item}</li>
            ))}
          </ul>
        </div>
      </section>

      <section className="landing-cta">
        <div className="container">
          <h2 className="cta-heading">{L.ctaHeading}</h2>
          <Link href="/demo" className="btn-primary btn-lg">
            {L.ctaButton}
          </Link>
        </div>
      </section>
    </main>
  );
}
