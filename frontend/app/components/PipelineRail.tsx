"use client";

import type { StageName } from "@/lib/types";
import { STAGE_ORDER } from "@/lib/labels";
import type { Dict } from "@/lib/i18n";
import { useLanguage } from "./LanguageProvider";

/**
 * The pipeline as a rail: deterministic nodes solid, additive ones (ML/Claude)
 * dotted, the rails seal emphasized. The thesis, drawn.
 */
export default function PipelineRail({
  completed,
  active,
  hasError = false,
}: {
  completed: Set<StageName>;
  active: StageName | null;
  hasError?: boolean;
}) {
  const { t } = useLanguage();
  return (
    <ol className={`pipeline-rail${hasError ? " has-error" : ""}`}>
      {STAGE_ORDER.map(({ key, kind }) => {
        const { label, note } = t.stages[key as keyof Dict["stages"]];
        const isDone = completed.has(key);
        const isActive = active === key;
        let status = "pending";
        if (isDone) status = "done";
        else if (isActive) status = "active";
        return (
          <li
            key={key}
            className={`rail-stage rail-stage-${kind} is-${status}`}
          >
            <span className="rail-stage-dot" aria-hidden="true">
              {isDone ? "✓" : isActive ? <span className="spinner" /> : ""}
            </span>
            <span className="rail-stage-label">{label}</span>
            <span className="rail-stage-note">{note}</span>
          </li>
        );
      })}
    </ol>
  );
}
