import type { CaseFeatures, StageName, Urgency } from "./types";

export const URGENCY_LABELS: Record<Urgency, string> = {
  inmediata: "Inmediata",
  prioritaria: "Prioritaria",
  ambulatoria: "Ambulatoria",
};

export const DIAGNOSIS_LABELS: Record<string, string> = {
  bppv_posterior: "VPPB posterior",
  bppv_horizontal: "VPPB horizontal",
  meniere: "Ménière",
  vestibular_migraine: "Migraña vestibular",
  vestibular_neuritis: "Neuritis vestibular",
  labyrinthitis: "Laberintitis",
  central_suspected: "Central (sospecha)",
  cardiogenic_suspected: "Cardiogénico (sospecha)",
  undetermined: "Indeterminado",
};

export const FORCED_ACTION_LABELS: Record<string, string> = {
  DERIVAR_URGENTE: "Derivación urgente",
  NO_BENIGNO: "No asumir benigno",
  BLOQUEAR_EPLEY: "Epley bloqueado",
  PRECAUCION_EXAMEN: "Precaución en examen",
  RED_SEGURIDAD: "Red de seguridad",
  ESCALAR: "Escalar",
};

/**
 * Nature of each pipeline stage — the product thesis turned into data:
 * the deterministic layers set safety, ML/Claude are additive.
 */
export type StageKind = "deterministic" | "additive" | "seal" | "terminal";

export interface StageMeta {
  key: StageName;
  label: string;
  note: string;
  kind: StageKind;
}

const DURATION_CHIPS: Record<string, string> = {
  seconds: "segundos",
  under_1min: "< 1 min",
  minutes: "minutos",
  hours: "horas",
  over_24h_continuous: "> 24 h continuo",
  days: "días",
  recurrent_episodic: "episódico recurrente",
};

const TRIGGER_CHIPS: Record<string, string> = {
  positional_head: "posicional",
  spontaneous: "espontáneo",
  orthostatic: "ortostático",
  valsalva: "Valsalva",
  sound_pressure: "sonido/presión",
  none: "sin gatillo",
};

/** Readable chips for the case cards — an honest clinical at-a-glance view. */
export function featureChips(features: CaseFeatures): string[] {
  const chips: string[] = [];
  if (features.duration) chips.push(DURATION_CHIPS[features.duration] ?? features.duration);
  if (features.trigger) chips.push(TRIGGER_CHIPS[features.trigger] ?? features.trigger);
  if (features.head_impulse === "normal") chips.push("HIT normal");
  if (features.nystagmus_direction === "direction_changing")
    chips.push("nistagmo cambiante");
  if (features.skew_deviation) chips.push("skew deviation");
  if (features.focal_signs?.length) chips.push("signos focales");
  if (features.truncal_ataxia_severe) chips.push("ataxia troncal");
  if (features.hearing_loss === "fluctuating") chips.push("hipoacusia fluctuante");
  if (features.tinnitus) chips.push("tinnitus");
  if (features.aural_fullness) chips.push("plenitud aural");
  if (features.nystagmus_fatigable) chips.push("nistagmo fatigable");
  if (features.dix_hallpike && features.dix_hallpike !== "not_done")
    chips.push("Dix-Hallpike +");
  if (features.age_years) chips.push(`${features.age_years} años`);
  if (features.vascular_risk_factors?.length)
    chips.push(`riesgo vascular ×${features.vascular_risk_factors.length}`);
  return chips.slice(0, 6);
}

export const STAGE_ORDER: StageMeta[] = [
  {
    key: "redflag",
    label: "Red flags",
    note: "determinista",
    kind: "deterministic",
  },
  {
    key: "differential",
    label: "Diferencial",
    note: "reglas ICVD",
    kind: "deterministic",
  },
  { key: "ml", label: "ML", note: "opcional", kind: "additive" },
  { key: "reasoning", label: "Claude", note: "explica", kind: "additive" },
  { key: "rails", label: "Rieles", note: "ganan siempre", kind: "seal" },
  { key: "done", label: "Recibo", note: "AuditEvent", kind: "terminal" },
];
