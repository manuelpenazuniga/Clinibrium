import type { StageName } from "./types";

/**
 * Nature of each pipeline stage — the product thesis turned into data:
 * the deterministic layers set safety, ML/Claude are additive.
 *
 * Display labels/notes live in the i18n dictionary (`t.stages[key]`); this
 * array is the language-independent STRUCTURE (order + kind).
 */
export type StageKind = "deterministic" | "additive" | "seal" | "terminal";

export interface StageMeta {
  key: StageName;
  kind: StageKind;
}

export const STAGE_ORDER: StageMeta[] = [
  { key: "redflag", kind: "deterministic" },
  { key: "differential", kind: "deterministic" },
  { key: "ml", kind: "additive" },
  { key: "reasoning", kind: "additive" },
  { key: "rails", kind: "seal" },
  { key: "done", kind: "terminal" },
];
