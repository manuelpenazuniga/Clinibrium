import type { CasePreset } from "./types";

export const CASE_PRESETS: CasePreset[] = [
  {
    id: "bppv-benign",
    name: "VPPB benigno (ambulatorio)",
    description:
      "Vértigo posicional paroxístico benigno — episodio breve, desencadenado por posición, nistagmus fatigable. Manejo ambulatorio.",
    features: {
      duration: "under_1min",
      trigger: "positional_head",
      timing_pattern: "episodic_triggered",
      onset: "sudden",
      dix_hallpike: "right_positive",
      nystagmus_fatigable: true,
      nystagmus_latency_s: 5,
    },
  },
  {
    id: "stroke-hints",
    name: "Sospecha de stroke (HINTS central)",
    description:
      "Síndrome vestibular agudo continuo con signos HINTS centrales: head-impulse normal, nistagmus cambiante, skew deviation. Alta sospecha de stroke posterior.",
    features: {
      duration: "over_24h_continuous",
      trigger: "spontaneous",
      timing_pattern: "acute_continuous",
      onset: "sudden",
      head_impulse: "normal",
      nystagmus_direction: "direction_changing",
      skew_deviation: true,
      focal_signs: ["dysarthria"],
      truncal_ataxia_severe: true,
      age_years: 68,
      vascular_risk_factors: ["hypertension", "atrial_fibrillation"],
    },
  },
  {
    id: "meniere-episodic",
    name: "Ménière (episódico)",
    description:
      "Enfermedad de Ménière — episodios espontáneos de horas con hipoacusia fluctuante, tinnitus y plenitud aural.",
    features: {
      duration: "hours",
      trigger: "spontaneous",
      timing_pattern: "episodic_spontaneous",
      onset: "gradual",
      hearing_loss: "fluctuating",
      tinnitus: true,
      aural_fullness: true,
      episode_duration: "hours",
    },
  },
];
