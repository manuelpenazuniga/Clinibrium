"use client";

import { useLanguage } from "./LanguageProvider";

export default function SiteFooter() {
  const { t } = useLanguage();
  return (
    <footer className="site-footer">
      <div className="site-footer-inner">
        <p className="footer-thesis">{t.footer.thesis}</p>
        <p className="footer-meta">
          {t.footer.builtPrefix}
          <strong>{t.footer.builtStrong}</strong>
          {t.footer.builtSuffix}
        </p>
        <p className="footer-meta">{t.footer.disclaimer}</p>
      </div>
    </footer>
  );
}
