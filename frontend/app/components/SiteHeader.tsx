"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

const NAV_LINKS = [
  { href: "/", label: "Inicio" },
  { href: "/demo", label: "Demo" },
  { href: "/dix-hallpike", label: "Dix-Hallpike" },
];

export default function SiteHeader() {
  const pathname = usePathname();

  return (
    <header className="site-header">
      <div className="site-header-inner">
        <Link href="/" className="wordmark">
          <span className="wordmark-name">Clinibrium</span>
          <span className="wordmark-engine">VertigoDx</span>
        </Link>
        <nav className="site-nav" aria-label="Navegación principal">
          {NAV_LINKS.map(({ href, label }) => (
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
        <span className="prototype-chip" title="Prototipo de hackathon — no destinado a uso clínico real">
          Prototipo · no uso clínico
        </span>
      </div>
    </header>
  );
}
