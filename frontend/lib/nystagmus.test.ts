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

/** Historial de N muestras con una velocidad constante. */
function constantHistory(v: Velocity, n = 30): Velocity[] {
  return Array.from({ length: n }, () => ({ ...v }));
}

// ---------------------------------------------------------------------------
// calculateVelocity — guardas de división (dt<=0) y unidades
// ---------------------------------------------------------------------------

describe("calculateVelocity", () => {
  it("dt === 0 → velocidad nula (sin división por cero)", () => {
    const v = calculateVelocity({ x: 0.5, y: 0.5 }, { x: 0.4, y: 0.4 }, 0, 640, 480);
    expect(v).toEqual({ vx: 0, vy: 0 });
  });

  it("dt < 0 (timestamp no monótono) → velocidad nula, nunca signo invertido", () => {
    const v = calculateVelocity({ x: 0.5, y: 0.5 }, { x: 0.4, y: 0.4 }, -0.033, 640, 480);
    expect(v).toEqual({ vx: 0, vy: 0 });
    // Garantía explícita: no produce un valor negativo/garbage.
    expect(Number.isFinite(v.vx)).toBe(true);
    expect(Number.isFinite(v.vy)).toBe(true);
  });

  it("prevPos null (primer frame) → velocidad nula", () => {
    const v = calculateVelocity({ x: 0.5, y: 0.5 }, null, 0.033, 640, 480);
    expect(v).toEqual({ vx: 0, vy: 0 });
  });

  it("dt > 0 → velocidad finita y con el signo del desplazamiento", () => {
    // Se mueve a la derecha (dx>0) y hacia abajo (dy>0).
    const v = calculateVelocity({ x: 0.6, y: 0.6 }, { x: 0.5, y: 0.5 }, 0.1, 640, 480);
    expect(v.vx).toBeGreaterThan(0);
    expect(v.vy).toBeGreaterThan(0);
    expect(Number.isFinite(v.vx)).toBe(true);
    expect(Number.isFinite(v.vy)).toBe(true);
  });
});

// ---------------------------------------------------------------------------
// getIrisCenter — promedio de landmarks
// ---------------------------------------------------------------------------

describe("getIrisCenter", () => {
  it("promedia los landmarks indicados", () => {
    const landmarks = [
      { x: 0, y: 0 },
      { x: 2, y: 4 },
      { x: 4, y: 8 },
    ];
    expect(getIrisCenter(landmarks, [0, 1, 2])).toEqual({ x: 2, y: 4 });
  });
});

// ---------------------------------------------------------------------------
// computeDirection — H / V / insuficiente
// ---------------------------------------------------------------------------

describe("computeDirection", () => {
  it("< 30 muestras → '-' (datos insuficientes)", () => {
    expect(computeDirection(constantHistory({ vx: 50, vy: 0 }, 10))).toBe("-");
  });

  it("componente horizontal dominante → Derecha/Izquierda", () => {
    expect(computeDirection(constantHistory({ vx: 40, vy: 1 }))).toBe("Derecha");
    expect(computeDirection(constantHistory({ vx: -40, vy: 1 }))).toBe("Izquierda");
  });

  it("componente vertical dominante (>5) → Arriba/Abajo", () => {
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

  it("beats equiespaciados → beats/segundo", () => {
    // 5 beats en 2 s → 4 intervalos / 2 s = 2 Hz
    expect(computeFrequency([0, 0.5, 1, 1.5, 2])).toBeCloseTo(2, 5);
  });
});

// ---------------------------------------------------------------------------
// toNystagmusFeatures — dirección, latencia/duración, none, y CONTRATO de egress
// ---------------------------------------------------------------------------

describe("toNystagmusFeatures", () => {
  it("historial vacío → dirección 'none', latencia/duración null, fatigable null", () => {
    const f = toNystagmusFeatures(createBeatState(), []);
    expect(f.nystagmus_direction).toBe("none");
    expect(f.nystagmus_latency_s).toBeNull();
    expect(f.nystagmus_duration_s).toBeNull();
    expect(f.nystagmus_fatigable).toBeNull();
  });

  it("horizontal dominante → 'horizontal'", () => {
    const f = toNystagmusFeatures(createBeatState(), constantHistory({ vx: 40, vy: 1 }));
    expect(f.nystagmus_direction).toBe("horizontal");
  });

  it("vertical dominante → 'vertical_pure'", () => {
    const f = toNystagmusFeatures(createBeatState(), constantHistory({ vx: 1, vy: 40 }));
    expect(f.nystagmus_direction).toBe("vertical_pure");
  });

  it("velocidad por debajo del umbral (<5) → 'none' (no fuerza dirección)", () => {
    const f = toNystagmusFeatures(createBeatState(), constantHistory({ vx: 2, vy: 1 }));
    expect(f.nystagmus_direction).toBe("none");
  });

  it("latencia = primer beat; duración = último - primero; redondeadas a 2 decimales", () => {
    const state: BeatState = {
      beatTimes: [1.234, 2.0, 3.789],
      firstBeatTime: 1.234,
      amplitudes: [],
    };
    const f = toNystagmusFeatures(state, []);
    expect(f.nystagmus_latency_s).toBe(1.23);
    expect(f.nystagmus_duration_s).toBe(2.56); // 3.789 - 1.234 = 2.555 → 2.56
  });

  it("un solo beat → duración null (no hay intervalo)", () => {
    const state: BeatState = { beatTimes: [1.5], firstBeatTime: 1.5, amplitudes: [] };
    const f = toNystagmusFeatures(state, []);
    expect(f.nystagmus_latency_s).toBe(1.5);
    expect(f.nystagmus_duration_s).toBeNull();
  });

  // INV-2 — el payload que sale del dispositivo NO contiene video/coords/PII.
  it("INV-2: solo emite las 4 features desidentificadas — sin video, coords ni PII", () => {
    const state: BeatState = {
      beatTimes: [0.5, 1.0, 1.5, 2.0, 2.5, 3.0],
      firstBeatTime: 0.5,
      amplitudes: [10, 9, 8, 3, 2, 1],
    };
    const history = constantHistory({ vx: 40, vy: 1 });
    const f = toNystagmusFeatures(state, history);

    // Allowlist EXACTA de claves que cruzan la red.
    expect(Object.keys(f).sort()).toEqual(
      [
        "nystagmus_direction",
        "nystagmus_duration_s",
        "nystagmus_fatigable",
        "nystagmus_latency_s",
      ].sort()
    );

    // Ninguna clave cruda de coordenada / imagen / landmark / identificador.
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
      // Todas las claves permitidas viven bajo el prefijo desidentificado.
      expect(k.startsWith("nystagmus_")).toBe(true);
      expect(FORBIDDEN_KEYS.has(k)).toBe(false);
    }

    // Los valores son escalares serializables (no objetos/arrays con coords).
    for (const v of Object.values(f)) {
      expect(["string", "number", "boolean", "object"]).toContain(typeof v);
      if (typeof v === "object") expect(v).toBeNull(); // solo null permitido
    }
  });
});
