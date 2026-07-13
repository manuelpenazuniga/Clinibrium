"use client";

import {
  useCallback,
  useEffect,
  useRef,
  useState,
  type CSSProperties,
} from "react";
import { useLanguage } from "./LanguageProvider";

// data-tour target per step (language-independent structure); copy comes from
// the dictionary (t.onboarding.steps), paired by index. `undefined` = centered.
const STEP_TARGETS: (string | undefined)[] = ["presets", "controls", "kill", undefined];

interface SpotRect {
  top: number;
  left: number;
  width: number;
  height: number;
}

function rectsDiffer(a: SpotRect | null, b: SpotRect | null): boolean {
  if (!a || !b) return a !== b;
  return (
    Math.abs(a.top - b.top) > 1 ||
    Math.abs(a.left - b.left) > 1 ||
    Math.abs(a.width - b.width) > 1 ||
    Math.abs(a.height - b.height) > 1
  );
}

export default function Onboarding({
  open,
  onClose,
}: {
  open: boolean;
  onClose: () => void;
}) {
  const { t } = useLanguage();
  const steps = t.onboarding.steps;
  const [step, setStep] = useState(0);
  const [rect, setRect] = useState<SpotRect | null>(null);
  const cardRef = useRef<HTMLDivElement>(null);

  const current = steps[step];
  const currentTarget = STEP_TARGETS[step];
  const isLast = step === steps.length - 1;

  const close = useCallback(() => {
    setStep(0);
    onClose();
  }, [onClose]);

  // Follow the highlighted element (scroll/resize/reflow) with a rAF loop.
  useEffect(() => {
    if (!open) return;

    const target = currentTarget
      ? document.querySelector<HTMLElement>(`[data-tour="${currentTarget}"]`)
      : null;

    if (target) {
      target.scrollIntoView({ block: "center", behavior: "smooth" });
    } else {
      setRect(null);
    }

    let raf = 0;
    const track = () => {
      if (target) {
        const r = target.getBoundingClientRect();
        const next: SpotRect = {
          top: r.top,
          left: r.left,
          width: r.width,
          height: r.height,
        };
        setRect((prev) => (rectsDiffer(prev, next) ? next : prev));
      }
      raf = requestAnimationFrame(track);
    };
    raf = requestAnimationFrame(track);
    return () => cancelAnimationFrame(raf);
  }, [open, currentTarget]);

  useEffect(() => {
    if (open) cardRef.current?.focus();
  }, [open, step]);

  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") close();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [open, close]);

  if (!open) return null;

  const PAD = 8;
  const cardStyle: CSSProperties = rect
    ? {
        top: Math.min(
          rect.top + rect.height + PAD + 12,
          typeof window !== "undefined" ? window.innerHeight - 260 : 0
        ),
        left: Math.max(
          16,
          Math.min(
            rect.left,
            (typeof window !== "undefined" ? window.innerWidth : 1200) - 396
          )
        ),
      }
    : { top: "50%", left: "50%", transform: "translate(-50%, -50%)" };

  return (
    <div className="tour-root">
      {rect ? (
        <div
          className="tour-spotlight"
          style={{
            top: rect.top - PAD,
            left: rect.left - PAD,
            width: rect.width + PAD * 2,
            height: rect.height + PAD * 2,
          }}
          aria-hidden="true"
        />
      ) : (
        <div className="tour-dim" aria-hidden="true" />
      )}

      <div
        ref={cardRef}
        className="tour-card"
        style={cardStyle}
        role="dialog"
        aria-modal="true"
        aria-label={current.title}
        tabIndex={-1}
      >
        <h3 className="tour-title">{current.title}</h3>
        <p className="tour-body">{current.body}</p>
        <div className="tour-footer">
          <button type="button" className="tour-skip" onClick={close}>
            {t.onboarding.skip}
          </button>
          <div className="tour-dots" aria-hidden="true">
            {steps.map((s, i) => (
              <span
                key={s.title}
                className={`tour-dot${i === step ? " current" : ""}`}
              />
            ))}
          </div>
          <div className="tour-nav">
            {step > 0 && (
              <button
                type="button"
                className="btn-secondary btn-sm"
                onClick={() => setStep((s) => s - 1)}
              >
                {t.onboarding.back}
              </button>
            )}
            <button
              type="button"
              className="btn-primary btn-sm"
              onClick={() => (isLast ? close() : setStep((s) => s + 1))}
            >
              {isLast ? t.onboarding.start : t.onboarding.next}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
