import type { CaseFeatures, PipelineResult, StageEvent, StageName } from "./types";

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
  signal?: AbortSignal;
  onStage?: (event: StageEvent) => void;
}

/**
 * POST /api/evaluate (SSE) — emite cada stage via onStage y resuelve con el
 * PipelineResult del evento `done`. Lanza en `error` o si el stream cierra
 * sin resultado.
 */
export async function streamEvaluation(
  features: CaseFeatures,
  { killReasoner = false, signal, onStage }: StreamOptions = {}
): Promise<PipelineResult> {
  const url = killReasoner
    ? `${API_URL}/api/evaluate?debug_kill_reasoner=true`
    : `${API_URL}/api/evaluate`;

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
    throw new Error("No se recibió resultado del pipeline");
  }
  return result;
}
