/**
 * `streamEvaluation` transport tests (codex-audit-4 Media 2).
 *
 * Pin the AD-19 transport contract: `lang` travels ONLY as a query param
 * (default "es"), never inside the request body — so it can never reach
 * CaseFeatures or the ML engine.
 */
import { afterEach, describe, expect, it, vi } from "vitest";
import { streamEvaluation } from "./api";
import type { CaseFeatures } from "./types";

const DONE_SSE = 'event: done\ndata: {"case_id":"case-1"}\n\n';

function mockFetch(body: string = DONE_SSE) {
  const fetchMock = vi.fn(async () => new Response(body, { status: 200 }));
  vi.stubGlobal("fetch", fetchMock);
  return fetchMock;
}

function requestOf(fetchMock: ReturnType<typeof mockFetch>): {
  url: string;
  body: Record<string, unknown>;
} {
  const [url, init] = fetchMock.mock.calls[0] as unknown as [string, RequestInit];
  return { url, body: JSON.parse(init.body as string) };
}

const FEATURES = { duration: "under_1min" } as CaseFeatures;

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("streamEvaluation lang transport", () => {
  it("defaults to lang=es in the query string", async () => {
    const fetchMock = mockFetch();
    await streamEvaluation(FEATURES);
    const { url } = requestOf(fetchMock);
    expect(url).toContain("lang=es");
  });

  it("sends lang=en when requested", async () => {
    const fetchMock = mockFetch();
    await streamEvaluation(FEATURES, { lang: "en" });
    const { url } = requestOf(fetchMock);
    expect(url).toContain("lang=en");
  });

  it("never puts lang (or debug flags) in the body", async () => {
    const fetchMock = mockFetch();
    await streamEvaluation(FEATURES, { lang: "en", killReasoner: true });
    const { url, body } = requestOf(fetchMock);
    expect(url).toContain("debug_kill_reasoner=true");
    expect(body).not.toHaveProperty("lang");
    expect(body).not.toHaveProperty("debug_kill_reasoner");
    expect(body).toEqual(FEATURES);
  });

  it("resolves with the done-event PipelineResult", async () => {
    mockFetch();
    const result = await streamEvaluation(FEATURES);
    expect(result.case_id).toBe("case-1");
  });

  it("omits debug_kill_reasoner entirely by default", async () => {
    const fetchMock = mockFetch();
    await streamEvaluation(FEATURES);
    const { url } = requestOf(fetchMock);
    expect(url).not.toContain("debug_kill_reasoner");
  });
});
