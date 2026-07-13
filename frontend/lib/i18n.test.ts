/**
 * Dictionary integrity tests (codex-audit-4 Media 2).
 *
 * `en: typeof es` already forces key parity at COMPILE time, but it cannot
 * see runtime content: empty strings, mismatched array lengths or a leaf
 * whose type diverges. These tests walk both trees and pin that down.
 */
import { describe, expect, it } from "vitest";
import { STRINGS, uiErrorText, type Dict, type UiError } from "./i18n";

interface Leaf {
  path: string;
  value: unknown;
}

/** Depth-first walk; arrays are treated as leaves (compared by length). */
function leaves(obj: unknown, prefix = ""): Leaf[] {
  if (obj !== null && typeof obj === "object" && !Array.isArray(obj)) {
    return Object.entries(obj as Record<string, unknown>).flatMap(([k, v]) =>
      leaves(v, prefix ? `${prefix}.${k}` : k)
    );
  }
  return [{ path: prefix, value: obj }];
}

const es = leaves(STRINGS.es);
const en = leaves(STRINGS.en);
const enByPath = new Map(en.map((l) => [l.path, l.value]));

describe("STRINGS es/en parity", () => {
  it("exposes exactly the same leaf paths in both languages", () => {
    expect(en.map((l) => l.path).sort()).toEqual(es.map((l) => l.path).sort());
  });

  it("every leaf has the same runtime type in both languages", () => {
    for (const { path, value } of es) {
      const other = enByPath.get(path);
      expect(typeof other, path).toBe(typeof value);
      expect(Array.isArray(other), path).toBe(Array.isArray(value));
    }
  });

  it("arrays have the same length in both languages", () => {
    for (const { path, value } of es) {
      if (Array.isArray(value)) {
        expect((enByPath.get(path) as unknown[]).length, path).toBe(value.length);
      }
    }
  });

  it("no string leaf is empty or whitespace-only, in either language", () => {
    for (const { path, value } of [...es, ...en]) {
      if (typeof value === "string") {
        expect(value.trim().length, path).toBeGreaterThan(0);
      }
    }
  });
});

describe("uiErrorText", () => {
  const t = STRINGS.es;
  const tEn = STRINGS.en;

  it("re-localizes dictionary-keyed errors when the language changes", () => {
    const err: UiError = { key: "connectionError" };
    expect(uiErrorText(err, t)).toBe(t.common.connectionError);
    expect(uiErrorText(err, tEn)).toBe(tEn.common.connectionError);
    expect(uiErrorText(err, t)).not.toBe(uiErrorText(err, tEn));
  });

  it("passes literal server messages through untouched", () => {
    const err: UiError = { message: "HTTP 500: Internal Server Error" };
    expect(uiErrorText(err, t)).toBe("HTTP 500: Internal Server Error");
    expect(uiErrorText(err, tEn)).toBe("HTTP 500: Internal Server Error");
  });
});

describe("dictionary functions", () => {
  it("parametrized chips render non-empty strings in both languages", () => {
    for (const lang of ["es", "en"] as const) {
      const c: Dict["chips"] = STRINGS[lang].chips;
      expect(c.years(63).trim().length).toBeGreaterThan(0);
      expect(c.vascularRisk(2).trim().length).toBeGreaterThan(0);
    }
  });
});
