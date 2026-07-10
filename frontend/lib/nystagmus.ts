export interface IrisPosition {
  x: number;
  y: number;
  time: number;
}

export interface Velocity {
  vx: number;
  vy: number;
}

export interface NystagmusMetrics {
  fps: number;
  velocityH: number;
  velocityV: number;
  frequency: number;
  direction: string;
  latency: number | null;
  beats: number;
  fatigable: string;
  duration: number | null;
}

export interface NystagmusFeatures {
  nystagmus_direction: "none" | "horizontal" | "vertical_pure" | "torsional_pure" | "mixed";
  nystagmus_latency_s: number | null;
  nystagmus_duration_s: number | null;
  nystagmus_fatigable: boolean | null;
}

export const LEFT_IRIS_INDICES = [468, 469, 470, 471, 472];
export const RIGHT_IRIS_INDICES = [473, 474, 475, 476, 477];
export const PX_TO_DEG = 0.05;
export const FAST_PHASE_THRESHOLD = 30;
export const MIN_BEAT_INTERVAL = 0.3;
export const MAX_HISTORY = 600;

export function getIrisCenter(
  landmarks: Array<{ x: number; y: number }>,
  indices: number[]
): { x: number; y: number } {
  let x = 0;
  let y = 0;
  for (const idx of indices) {
    x += landmarks[idx].x;
    y += landmarks[idx].y;
  }
  return { x: x / indices.length, y: y / indices.length };
}

export function calculateVelocity(
  currentPos: { x: number; y: number },
  prevPos: { x: number; y: number } | null,
  dt: number,
  videoWidth: number,
  videoHeight: number
): Velocity {
  if (!prevPos || dt === 0) return { vx: 0, vy: 0 };
  const dx = (currentPos.x - prevPos.x) * videoWidth;
  const dy = (currentPos.y - prevPos.y) * videoHeight;
  return {
    vx: (dx * PX_TO_DEG) / dt,
    vy: (dy * PX_TO_DEG) / dt,
  };
}

export interface BeatState {
  beatTimes: number[];
  firstBeatTime: number | null;
  amplitudes: number[];
}

export function createBeatState(): BeatState {
  return { beatTimes: [], firstBeatTime: null, amplitudes: [] };
}

export function detectBeat(
  velocity: Velocity,
  currentTime: number,
  state: BeatState,
  velocityHistory: Velocity[]
): boolean {
  const speed = Math.sqrt(velocity.vx ** 2 + velocity.vy ** 2);
  if (speed > FAST_PHASE_THRESHOLD) {
    const lastBeat = state.beatTimes[state.beatTimes.length - 1];
    if (!lastBeat || currentTime - lastBeat > MIN_BEAT_INTERVAL) {
      state.beatTimes.push(currentTime);
      if (state.firstBeatTime === null) {
        state.firstBeatTime = currentTime;
      }
      if (velocityHistory.length > 10) {
        const recent = velocityHistory.slice(-10);
        const avgSpeed =
          recent.reduce((s, v) => s + Math.sqrt(v.vx ** 2 + v.vy ** 2), 0) /
          10;
        state.amplitudes.push(avgSpeed);
      }
    }
  }
  return speed > FAST_PHASE_THRESHOLD;
}

export function computeDirection(
  velocityHistory: Velocity[]
): string {
  if (velocityHistory.length < 30) return "-";
  const recent = velocityHistory.slice(-30);
  const avgVx = recent.reduce((s, v) => s + v.vx, 0) / 30;
  const avgVy = recent.reduce((s, v) => s + v.vy, 0) / 30;
  if (Math.abs(avgVx) > Math.abs(avgVy)) {
    return avgVx > 0 ? "Derecha" : "Izquierda";
  }
  if (Math.abs(avgVy) > 5) {
    return avgVy > 0 ? "Abajo" : "Arriba";
  }
  return "-";
}

export function computeFatigability(amplitudes: number[]): string {
  if (amplitudes.length < 5) return "-";
  const mid = Math.floor(amplitudes.length / 2);
  const firstHalf = amplitudes.slice(0, mid);
  const secondHalf = amplitudes.slice(mid);
  const avgFirst = firstHalf.reduce((a, b) => a + b, 0) / firstHalf.length;
  const avgSecond = secondHalf.reduce((a, b) => a + b, 0) / secondHalf.length;
  return avgSecond / avgFirst < 0.7 ? "Sí" : "No";
}

export function computeFrequency(beatTimes: number[]): number {
  if (beatTimes.length < 2) return 0;
  const duration = beatTimes[beatTimes.length - 1] - beatTimes[0];
  return duration > 0 ? (beatTimes.length - 1) / duration : 0;
}

export function toNystagmusFeatures(
  beatState: BeatState,
  velocityHistory: Velocity[]
): NystagmusFeatures {
  let direction: NystagmusFeatures["nystagmus_direction"] = "none";
  if (velocityHistory.length >= 30) {
    const recent = velocityHistory.slice(-30);
    const avgVx = recent.reduce((s, v) => s + v.vx, 0) / 30;
    const avgVy = recent.reduce((s, v) => s + v.vy, 0) / 30;
    if (Math.abs(avgVx) > Math.abs(avgVy) && Math.abs(avgVx) > 5) {
      direction = "horizontal";
    } else if (Math.abs(avgVy) > Math.abs(avgVx) && Math.abs(avgVy) > 5) {
      direction = "vertical_pure";
    }
  }

  const latency = beatState.firstBeatTime;
  const duration =
    beatState.beatTimes.length >= 2
      ? beatState.beatTimes[beatState.beatTimes.length - 1] -
        beatState.beatTimes[0]
      : null;

  let fatigable: boolean | null = null;
  if (beatState.amplitudes.length >= 5) {
    const mid = Math.floor(beatState.amplitudes.length / 2);
    const firstHalf = beatState.amplitudes.slice(0, mid);
    const secondHalf = beatState.amplitudes.slice(mid);
    const avgFirst = firstHalf.reduce((a, b) => a + b, 0) / firstHalf.length;
    const avgSecond =
      secondHalf.reduce((a, b) => a + b, 0) / secondHalf.length;
    fatigable = avgSecond / avgFirst < 0.7;
  }

  return {
    nystagmus_direction: direction,
    nystagmus_latency_s: latency !== null ? Math.round(latency * 100) / 100 : null,
    nystagmus_duration_s: duration !== null ? Math.round(duration * 100) / 100 : null,
    nystagmus_fatigable: fatigable,
  };
}
