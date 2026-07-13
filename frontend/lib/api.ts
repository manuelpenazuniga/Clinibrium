import type { CaseFeatures, PipelineResult, StageEvent, StageName } from "./types";
import { STRINGS, type Lang } from "./i18n";

export const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

export function parseSSEEvents(buffer: string): {
  events: StageEvent[];
  remainder: string;
} {
  const events: StageEvent[] = [];
  let remaining = buffer;

  while (true) {
    const doubleNewline = remaining.indexOf("\n\n");
    if (doubleNewline === -1) break;

    const rawEvent = remaining.slice(0, doubleNewline);
    remaining = remaining.slice(doubleNewline + 2);

    let eventType = "";
    let dataStr = "";

    for (const line of rawEvent.split("\n")) {
      if (line.startsWith("event: ")) {
        eventType = line.slice(7).trim();
      } else if (line.startsWith("data: ")) {
        dataStr += line.slice(6);
      } else if (line.startsWith("data:")) {
        dataStr += line.slice(5);
      }
    }

    if (eventType && dataStr) {
      try {
        events.push({
          stage: eventType as StageName,
          data: JSON.parse(dataStr) as unknown,
          timestamp: Date.now(),
        });
      } catch {
        // skip malformed JSON
      }
    }
  }

  return { events, remainder: remaining };
}

export interface StreamOptions {
  killReasoner?: boolean;
  /** UI language for backend-localized labels + reasoner output (default "es"). */
  lang?: Lang;
  signal?: AbortSignal;
  onStage?: (event: StageEvent) => void;
}

/**
 * POST /api/evaluate (SSE) — emits each stage via onStage and resolves with
 * the PipelineResult from the `done` event. Throws on `error` or if the
 * stream closes without a result.
 *
 * `lang` is passed as a query param so the backend localizes the labels it
 * produces (red-flag hits, reasoner prose) to match the UI. It NEVER goes in
 * the body, so it can never reach the ML engine or CaseFeatures.
 */
export async function streamEvaluation(
  features: CaseFeatures,
  { killReasoner = false, lang = "es", signal, onStage }: StreamOptions = {}
): Promise<PipelineResult> {
  const params = new URLSearchParams();
  if (killReasoner) params.set("debug_kill_reasoner", "true");
  params.set("lang", lang);
  const url = `${API_URL}/api/evaluate?${params.toString()}`;

  const response = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(features),
    signal,
  });

  if (!response.ok) {
    throw new Error(`HTTP ${response.status}: ${response.statusText}`);
  }

  const reader = response.body?.getReader();
  if (!reader) {
    throw new Error("ReadableStream not supported");
  }

  const decoder = new TextDecoder();
  let buffer = "";
  let result: PipelineResult | null = null;

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });
    const { events, remainder } = parseSSEEvents(buffer);
    buffer = remainder;

    for (const evt of events) {
      onStage?.(evt);
      if (evt.stage === "done") {
        result = evt.data as PipelineResult;
      } else if (evt.stage === "error") {
        const errData = evt.data as { error: string; message: string };
        throw new Error(`${errData.error}: ${errData.message}`);
      }
    }
  }

  if (!result) {
    throw new Error(STRINGS[lang].common.noPipelineResult);
  }
  return result;
}
