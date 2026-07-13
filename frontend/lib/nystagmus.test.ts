import { describe, it, expect } from "vitest";
import {
  calculateVelocity,
  computeDirection,
  computeFrequency,
  createBeatState,
  getIrisCenter,
  toNystagmusFeatures,
  type BeatState,
  type Velocity,
} from "./nystagmus";

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/** History of N samples with a constant velocity. */
function constantHistory(v: Velocity, n = 30): Velocity[] {
  return Array.from({ length: n }, () => ({ ...v }));
}

// ---------------------------------------------------------------------------
// calculateVelocity — division guards (dt<=0) and units
// ---------------------------------------------------------------------------

describe("calculateVelocity", () => {
  it("dt === 0 → zero velocity (no division by zero)", () => {
    const v = calculateVelocity({ x: 0.5, y: 0.5 }, { x: 0.4, y: 0.4 }, 0, 640, 480);
    expect(v).toEqual({ vx: 0, vy: 0 });
  });

  it("dt < 0 (non-monotonic timestamp) → zero velocity, never inverted sign", () => {
    const v = calculateVelocity({ x: 0.5, y: 0.5 }, { x: 0.4, y: 0.4 }, -0.033, 640, 480);
    expect(v).toEqual({ vx: 0, vy: 0 });
    // Explicit guarantee: it does not produce a negative/garbage value.
    expect(Number.isFinite(v.vx)).toBe(true);
    expect(Number.isFinite(v.vy)).toBe(true);
  });

  it("prevPos null (first frame) → zero velocity", () => {
    const v = calculateVelocity({ x: 0.5, y: 0.5 }, null, 0.033, 640, 480);
    expect(v).toEqual({ vx: 0, vy: 0 });
  });

  it("dt > 0 → finite velocity with the sign of the displacement", () => {
    // Moves to the right (dx>0) and downward (dy>0).
    const v = calculateVelocity({ x: 0.6, y: 0.6 }, { x: 0.5, y: 0.5 }, 0.1, 640, 480);
    expect(v.vx).toBeGreaterThan(0);
    expect(v.vy).toBeGreaterThan(0);
    expect(Number.isFinite(v.vx)).toBe(true);
    expect(Number.isFinite(v.vy)).toBe(true);
  });
});

// ---------------------------------------------------------------------------
// getIrisCenter — landmark averaging
// ---------------------------------------------------------------------------

describe("getIrisCenter", () => {
  it("averages the given landmarks", () => {
    const landmarks = [
      { x: 0, y: 0 },
      { x: 2, y: 4 },
      { x: 4, y: 8 },
    ];
    expect(getIrisCenter(landmarks, [0, 1, 2])).toEqual({ x: 2, y: 4 });
  });
});

// ---------------------------------------------------------------------------
// computeDirection — H / V / insufficient data
// ---------------------------------------------------------------------------

describe("computeDirection", () => {
  it("< 30 samples → '-' (insufficient data)", () => {
    expect(computeDirection(constantHistory({ vx: 50, vy: 0 }, 10))).toBe("-");
  });

  it("dominant horizontal component → Derecha/Izquierda", () => {
    expect(computeDirection(constantHistory({ vx: 40, vy: 1 }))).toBe("Derecha");
    expect(computeDirection(constantHistory({ vx: -40, vy: 1 }))).toBe("Izquierda");
  });

  it("dominant vertical component (>5) → Arriba/Abajo", () => {
    expect(computeDirection(constantHistory({ vx: 1, vy: 40 }))).toBe("Abajo");
    expect(computeDirection(constantHistory({ vx: 1, vy: -40 }))).toBe("Arriba");
  });
});

// ---------------------------------------------------------------------------
// computeFrequency
// ---------------------------------------------------------------------------

describe("computeFrequency", () => {
  it("< 2 beats → 0", () => {
    expect(computeFrequency([])).toBe(0);
    expect(computeFrequency([1.0])).toBe(0);
  });

  it("evenly spaced beats → beats/second", () => {
    // 5 beats in 2 s → 4 intervals / 2 s = 2 Hz
    expect(computeFrequency([0, 0.5, 1, 1.5, 2])).toBeCloseTo(2, 5);
  });
});

// ---------------------------------------------------------------------------
// toNystagmusFeatures — direction, latency/duration, none, and the egress CONTRACT
// ---------------------------------------------------------------------------

describe("toNystagmusFeatures", () => {
  it("empty history → direction 'none', latency/duration null, fatigable null", () => {
    const f = toNystagmusFeatures(createBeatState(), []);
    expect(f.nystagmus_direction).toBe("none");
    expect(f.nystagmus_latency_s).toBeNull();
    expect(f.nystagmus_duration_s).toBeNull();
    expect(f.nystagmus_fatigable).toBeNull();
  });

  it("dominant horizontal → 'horizontal'", () => {
    const f = toNystagmusFeatures(createBeatState(), constantHistory({ vx: 40, vy: 1 }));
    expect(f.nystagmus_direction).toBe("horizontal");
  });

  it("dominant vertical → 'vertical_pure'", () => {
    const f = toNystagmusFeatures(createBeatState(), constantHistory({ vx: 1, vy: 40 }));
    expect(f.nystagmus_direction).toBe("vertical_pure");
  });

  it("velocity below threshold (<5) → 'none' (does not force a direction)", () => {
    const f = toNystagmusFeatures(createBeatState(), constantHistory({ vx: 2, vy: 1 }));
    expect(f.nystagmus_direction).toBe("none");
  });

  it("latency = first beat; duration = last - first; rounded to 2 decimals", () => {
    const state: BeatState = {
      beatTimes: [1.234, 2.0, 3.789],
      firstBeatTime: 1.234,
      amplitudes: [],
    };
    const f = toNystagmusFeatures(state, []);
    expect(f.nystagmus_latency_s).toBe(1.23);
    expect(f.nystagmus_duration_s).toBe(2.56); // 3.789 - 1.234 = 2.555 → 2.56
  });

  it("single beat → duration null (no interval)", () => {
    const state: BeatState = { beatTimes: [1.5], firstBeatTime: 1.5, amplitudes: [] };
    const f = toNystagmusFeatures(state, []);
    expect(f.nystagmus_latency_s).toBe(1.5);
    expect(f.nystagmus_duration_s).toBeNull();
  });

  // INV-2 — the payload leaving the device contains NO video/coords/PII.
  it("INV-2: emits only the 4 de-identified features — no video, coords, or PII", () => {
    const state: BeatState = {
      beatTimes: [0.5, 1.0, 1.5, 2.0, 2.5, 3.0],
      firstBeatTime: 0.5,
      amplitudes: [10, 9, 8, 3, 2, 1],
    };
    const history = constantHistory({ vx: 40, vy: 1 });
    const f = toNystagmusFeatures(state, history);

    // EXACT allowlist of keys that cross the network.
    expect(Object.keys(f).sort()).toEqual(
      [
        "nystagmus_direction",
        "nystagmus_duration_s",
        "nystagmus_fatigable",
        "nystagmus_latency_s",
      ].sort()
    );

    // No raw coordinate / image / landmark / identifier key.
    const FORBIDDEN_KEYS = new Set([
      "x",
      "y",
      "frame",
      "frames",
      "image",
      "video",
      "landmark",
      "landmarks",
      "pixel",
      "iris",
      "name",
      "rut",
      "patient_id",
      "id",
    ]);
    for (const k of Object.keys(f)) {
      // All allowed keys live under the de-identified prefix.
      expect(k.startsWith("nystagmus_")).toBe(true);
      expect(FORBIDDEN_KEYS.has(k)).toBe(false);
    }

    // Values are serializable scalars (no objects/arrays with coords).
    for (const v of Object.values(f)) {
      expect(["string", "number", "boolean", "object"]).toContain(typeof v);
      if (typeof v === "object") expect(v).toBeNull(); // only null allowed
    }
  });
});
