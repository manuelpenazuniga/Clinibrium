export type SymptomDuration =
  | "seconds"
  | "under_1min"
  | "minutes"
  | "hours"
  | "over_24h_continuous"
  | "days"
  | "recurrent_episodic";

export type Onset = "sudden" | "gradual" | "unknown";

export type Trigger =
  | "positional_head"
  | "spontaneous"
  | "orthostatic"
  | "valsalva"
  | "sound_pressure"
  | "none";

export type TimingPattern =
  | "acute_continuous"
  | "episodic_triggered"
  | "episodic_spontaneous"
  | "chronic";

export type NystagmusDirection =
  | "none"
  | "horizontal"
  | "vertical_pure"
  | "torsional_pure"
  | "mixed"
  | "direction_changing";

export type HeadImpulse =
  | "normal"
  | "abnormal_corrective_saccade"
  | "not_done";

export type HearingLoss =
  | "none"
  | "sudden_unilateral"
  | "fluctuating"
  | "chronic";

export type FocalSign =
  | "dysarthria"
  | "dysphagia"
  | "diplopia"
  | "limb_weakness"
  | "facial_droop"
  | "numbness"
  | "hiccups"
  | "horner";

export type VascularRiskFactor =
  | "hypertension"
  | "diabetes"
  | "atrial_fibrillation"
  | "smoking"
  | "prior_stroke_tia";

export type DixHallpikeResult =
  | "right_positive"
  | "left_positive"
  | "bilateral_positive"
  | "negative"
  | "not_done";

export interface CaseFeatures {
  duration?: SymptomDuration;
  onset?: Onset;
  trigger?: Trigger;
  timing_pattern?: TimingPattern;
  nystagmus_direction?: NystagmusDirection;
  nystagmus_direction_changing_gaze?: boolean;
  nystagmus_latency_s?: number;
  nystagmus_duration_s?: number;
  nystagmus_fatigable?: boolean;
  nystagmus_suppressed_by_fixation?: boolean;
  head_impulse?: HeadImpulse;
  skew_deviation?: boolean;
  hearing_loss?: HearingLoss;
  tinnitus?: boolean;
  aural_fullness?: boolean;
  focal_signs?: FocalSign[];
  truncal_ataxia_severe?: boolean;
  headache_neck_pain_sudden_severe?: boolean;
  migrainous_features?: boolean;
  age_years?: number;
  vascular_risk_factors?: VascularRiskFactor[];
  fever?: boolean;
  neck_stiffness?: boolean;
  altered_consciousness?: boolean;
  presyncope_syncope?: boolean;
  palpitations?: boolean;
  chest_pain?: boolean;
  otitis_mastoiditis?: boolean;
  recent_head_neck_trauma?: boolean;
  cervical_pathology?: boolean;
  known_carotid_vertebrobasilar_disease?: boolean;
  cardiovascular_instability?: boolean;
  dix_hallpike?: DixHallpikeResult;
  torsion_confirmed_by_clinician?: boolean;
  episode_count?: number;
  episode_duration?: SymptomDuration;
  worsening_during_flow?: boolean;
}

export type Urgency = "inmediata" | "prioritaria" | "ambulatoria";

export type ForcedAction =
  | "DERIVAR_URGENTE"
  | "NO_BENIGNO"
  | "BLOQUEAR_EPLEY"
  | "PRECAUCION_EXAMEN"
  | "RED_SEGURIDAD"
  | "ESCALAR";

export type Diagnosis =
  | "bppv_posterior"
  | "bppv_horizontal"
  | "meniere"
  | "vestibular_migraine"
  | "vestibular_neuritis"
  | "labyrinthitis"
  | "central_suspected"
  | "cardiogenic_suspected"
  | "undetermined";

export interface RedFlagHit {
  id: string;
  label: string;
  forced_actions: ForcedAction[];
  severity: "high" | "medium";
}

export interface RedFlagResult {
  red_flag_activa: boolean;
  hits: RedFlagHit[];
  forced_actions: ForcedAction[];
}

export interface DifferentialCandidate {
  diagnosis: Diagnosis;
  score: number;
  rule_ids: string[];
}

export interface DifferentialResult {
  candidates: DifferentialCandidate[];
}

export interface PredictResponse {
  probabilities: Record<string, number>;
  shap: Record<string, number> | null;
  model_version: string;
}

export interface ReasonerOutput {
  explanation: string;
  reconciliation: string;
  suggested_next_steps: string[];
  model_used: string;
  reasoner_suggested_urgency: Urgency | null;
  grounding_refs: string[];
}

export interface AuditEvent {
  id: string;
  occurred_at: string;
  event_type: string;
  actor: string;
  model_used: string | null;
  input_features_hash: string;
  urgency: Urgency;
  forced_actions: ForcedAction[];
  red_flag_activa: boolean;
  outcome_summary: string;
  reasoner_status: "ok" | "degraded";
  outcome: string;
}

export interface PipelineResult {
  case_id: string;
  urgency: Urgency;
  red_flag: RedFlagResult;
  differential: DifferentialResult;
  ml: PredictResponse | null;
  reasoning: ReasonerOutput | null;
  forced_actions: ForcedAction[];
  applied_rails: string[];
  audit_event_id: string | null;
  audit_event: AuditEvent | null;
  fhir_bundle?: Record<string, unknown>;
}

export type StageName =
  | "redflag"
  | "differential"
  | "ml"
  | "reasoning"
  | "rails"
  | "done"
  | "error";

export interface StageEvent {
  stage: StageName;
  data: unknown;
  timestamp: number;
}

export interface CasePreset {
  id: string;
  name: string;
  description: string;
  features: CaseFeatures;
}
