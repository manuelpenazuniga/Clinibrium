"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useLanguage } from "./LanguageProvider";
import type { Lang } from "@/lib/i18n";

export default function SiteHeader() {
  const pathname = usePathname();
  const { lang, setLang, t } = useLanguage();

  const navLinks = [
    { href: "/", label: t.header.nav.home },
    { href: "/demo", label: t.header.nav.demo },
    { href: "/dix-hallpike", label: t.header.nav.dix },
  ];

  const langButton = (value: Lang, label: string) => (
    <button
      type="button"
      className={`lang-option${lang === value ? " active" : ""}`}
      aria-pressed={lang === value}
      onClick={() => setLang(value)}
    >
      {label}
    </button>
  );

  return (
    <>
      <a href="#contenido" className="skip-link">
        {t.skipLink}
      </a>
      <header className="site-header">
        <div className="site-header-inner">
          <Link href="/" className="wordmark">
            <span className="wordmark-name">Clinibrium</span>
            <span className="wordmark-engine">VertigoDx</span>
          </Link>
          <nav className="site-nav" aria-label={t.header.navAria}>
            {navLinks.map(({ href, label }) => (
              <Link
                key={href}
                href={href}
                className={`site-nav-link${pathname === href ? " current" : ""}`}
                aria-current={pathname === href ? "page" : undefined}
              >
                {label}
              </Link>
            ))}
          </nav>
          <div className="site-header-aside">
            <div
              className="lang-toggle"
              role="group"
              aria-label={t.header.langToggleAria}
            >
              {langButton("es", t.header.langEs)}
              {langButton("en", t.header.langEn)}
            </div>
            <span className="prototype-chip" title={t.header.prototypeChipTitle}>
              {t.header.prototypeChip}
            </span>
          </div>
        </div>
      </header>
    </>
  );
}
