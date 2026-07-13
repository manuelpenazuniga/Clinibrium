"use client";

import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useState,
  type ReactNode,
} from "react";
import { flushSync } from "react-dom";
import { STRINGS, type Dict, type Lang } from "@/lib/i18n";

const STORAGE_KEY = "clinibrium.lang";

interface LanguageContextValue {
  lang: Lang;
  setLang: (lang: Lang) => void;
  t: Dict;
}

const LanguageContext = createContext<LanguageContextValue | null>(null);

/**
 * Client-side language context. Spanish is the DEFAULT and the value used for
 * the first (server-matching) render, so there is NO hydration mismatch: the
 * persisted choice from localStorage is only applied AFTER mount.
 *
 * Switching language re-renders copy from the dictionary keys — it does NOT
 * re-trigger any evaluation or emit a second AuditEvent. Backend labels already
 * fetched keep the language they were requested with until the next evaluation.
 */
export default function LanguageProvider({ children }: { children: ReactNode }) {
  const [lang, setLangState] = useState<Lang>("es");

  useEffect(() => {
    let stored: string | null = null;
    try {
      stored = window.localStorage.getItem(STORAGE_KEY);
    } catch {
      // localStorage unreadable (privacy policy / SecurityError) — degrade to
      // the Spanish default instead of failing the mount effect.
    }
    if (stored === "en" || stored === "es") {
      setLangState(stored);
    }
  }, []);

  useEffect(() => {
    document.documentElement.lang = lang;
  }, [lang]);

  const setLang = useCallback(
    (next: Lang) => {
      if (next === lang) return;

      const apply = () => {
        setLangState(next);
        try {
          window.localStorage.setItem(STORAGE_KEY, next);
        } catch {
          // localStorage unavailable (private mode) — keep in-memory choice.
        }
      };

      // Subtle whole-page crossfade on language switch (~200ms, see
      // ::view-transition rules in globals.css). flushSync makes React commit
      // the new copy inside the transition's snapshot callback. Falls back to
      // an instant swap when the API is missing or reduced motion is on.
      const doc = document as Document & {
        startViewTransition?: (cb: () => void) => {
          ready: Promise<void>;
          finished: Promise<void>;
        };
      };
      const reduceMotion = window.matchMedia(
        "(prefers-reduced-motion: reduce)"
      ).matches;
      if (doc.startViewTransition && !reduceMotion) {
        const transition = doc.startViewTransition(() => {
          flushSync(apply);
        });
        // The copy swap happens either way; a skipped transition (hidden tab,
        // concurrent transition) rejects these promises — don't surface it.
        transition.ready.catch(() => {});
        transition.finished.catch(() => {});
      } else {
        apply();
      }
    },
    [lang]
  );

  return (
    <LanguageContext.Provider value={{ lang, setLang, t: STRINGS[lang] }}>
      {children}
    </LanguageContext.Provider>
  );
}

export function useLanguage(): LanguageContextValue {
  const ctx = useContext(LanguageContext);
  if (!ctx) {
    throw new Error("useLanguage must be used within a LanguageProvider");
  }
  return ctx;
}
