"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { useLanguage } from "./LanguageProvider";
import LanguageSwitch from "./LanguageSwitch";

/**
 * Full-screen first-run onboarding (6 large steps). Shown once before the
 * Clinibrium app; the last step hands over control with "Entrar a Clinibrium".
 *
 * The progress indicator deliberately mirrors the product's own pipeline
 * rail: numbered nodes on a line that fills left → right as the guide
 * advances — the same metaphor the demo then shows for real.
 */
export default function WelcomeOnboarding({
  open,
  onClose,
}: {
  open: boolean;
  onClose: () => void;
}) {
  const { t } = useLanguage();
  const steps = t.welcome.steps;
  const [step, setStep] = useState(0);
  // "fwd" | "back" picks the slide direction of the step content.
  const [dir, setDir] = useState<"fwd" | "back">("fwd");
  const cardRef = useRef<HTMLDivElement>(null);

  const isLast = step === steps.length - 1;
  const current = steps[step];

  const close = useCallback(() => {
    setStep(0);
    setDir("fwd");
    onClose();
  }, [onClose]);

  const goTo = useCallback(
    (next: number) => {
      setDir(next > step ? "fwd" : "back");
      setStep(next);
    },
    [step]
  );

  useEffect(() => {
    if (open) cardRef.current?.focus();
  }, [open, step]);

  // The overlay owns the viewport while open: lock body scroll behind it.
  useEffect(() => {
    if (!open) return;
    const previous = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    return () => {
      document.body.style.overflow = previous;
    };
  }, [open]);

  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") close();
      if (e.key === "ArrowRight" && step < steps.length - 1) goTo(step + 1);
      if (e.key === "ArrowLeft" && step > 0) goTo(step - 1);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [open, step, steps.length, close, goTo]);

  if (!open) return null;

  return (
    <div className="welcome-root">
      <div className="welcome-backdrop" aria-hidden="true" />

      <div
        ref={cardRef}
        className="welcome-card"
        role="dialog"
        aria-modal="true"
        aria-label={current.title}
        tabIndex={-1}
      >
        <header className="welcome-top">
          <span className="welcome-wordmark">
            Clinibrium <span className="welcome-engine">beta version</span>
          </span>
          <LanguageSwitch compact />
        </header>

        {/* Re-keyed per step so the enter animation replays, sliding from
            the travel direction (fwd = from the right, back = from the left). */}
        <div key={step} className="welcome-body" data-dir={dir}>
          <p className="welcome-kicker">
            {current.kicker}
            <span className="welcome-count">
              {t.welcome.stepCount(step + 1, steps.length)}
            </span>
          </p>
          <h2 className="welcome-title">{current.title}</h2>
          <p className="welcome-text">{current.body}</p>
        </div>

        <div
          className="welcome-rail"
          role="progressbar"
          aria-label={t.welcome.progressAria}
          aria-valuemin={1}
          aria-valuemax={steps.length}
          aria-valuenow={step + 1}
        >
          <div className="welcome-rail-line" aria-hidden="true">
            <div
              className="welcome-rail-fill"
              style={{ width: `${(step / (steps.length - 1)) * 100}%` }}
            />
          </div>
          {steps.map((s, i) => (
            <button
              key={s.title}
              type="button"
              className={`welcome-node${i < step ? " done" : ""}${
                i === step ? " current" : ""
              }`}
              aria-label={t.welcome.goToStep(i + 1)}
              aria-current={i === step ? "step" : undefined}
              onClick={() => goTo(i)}
            >
              {i + 1}
            </button>
          ))}
        </div>

        <footer className="welcome-footer">
          {!isLast ? (
            <button type="button" className="welcome-skip" onClick={close}>
              {t.welcome.skip}
            </button>
          ) : (
            <span aria-hidden="true" />
          )}
          <div className="welcome-nav">
            {step > 0 && (
              <button
                type="button"
                className="btn-secondary"
                onClick={() => goTo(step - 1)}
              >
                {t.welcome.back}
              </button>
            )}
            {isLast ? (
              <button
                type="button"
                className="btn-primary btn-lg welcome-enter"
                onClick={close}
              >
                {t.welcome.enter}
              </button>
            ) : (
              <button
                type="button"
                className="btn-primary"
                onClick={() => goTo(step + 1)}
              >
                {t.welcome.next}
              </button>
            )}
          </div>
        </footer>
      </div>
    </div>
  );
}
