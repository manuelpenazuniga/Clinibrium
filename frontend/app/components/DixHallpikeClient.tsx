"use client";

import {
  useCallback,
  useEffect,
  useRef,
  useState,
  type ChangeEvent,
} from "react";
import {
  createBeatState,
  calculateVelocity,
  computeDirection,
  computeFatigability,
  computeFrequency,
  detectBeat,
  getIrisCenter,
  LEFT_IRIS_INDICES,
  MAX_HISTORY,
  RIGHT_IRIS_INDICES,
  toNystagmusFeatures,
  type BeatState,
  type Velocity,
} from "@/lib/nystagmus";
import type {
  CaseFeatures,
  DixHallpikeResult,
  PipelineResult,
  StageEvent,
  StageName,
} from "@/lib/types";

const API_URL =
  process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

type TorsionDirection = "right_ear" | "left_ear" | "none" | "";
type SourceMode = "webcam" | "file";

interface FaceLandmarkerInstance {
  detectForVideo: (
    video: HTMLVideoElement,
    timestampMs: number
  ) => { faceLandmarks: Array<Array<{ x: number; y: number }>> | undefined };
  close: () => void;
}

interface MediaPipeModules {
  FaceLandmarker: {
    createFromOptions: (
      filesetResolver: unknown,
      options: unknown
    ) => Promise<FaceLandmarkerInstance>;
  };
  FilesetResolver: new (basePath: string) => unknown;
}

async function loadMediaPipe(): Promise<MediaPipeModules> {
  const vision = await import("@mediapipe/tasks-vision");
  return {
    FaceLandmarker: vision.FaceLandmarker as unknown as MediaPipeModules["FaceLandmarker"],
    FilesetResolver: vision.FilesetResolver as unknown as MediaPipeModules["FilesetResolver"],
  };
}

function parseSSEEvents(
  buffer: string
): { events: StageEvent[]; remainder: string } {
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
      if (line.startsWith("event: ")) eventType = line.slice(7).trim();
      else if (line.startsWith("data: ")) dataStr += line.slice(6);
      else if (line.startsWith("data:")) dataStr += line.slice(5);
    }
    if (eventType && dataStr) {
      try {
        events.push({
          stage: eventType as StageName,
          data: JSON.parse(dataStr) as unknown,
          timestamp: Date.now(),
        });
      } catch {
        /* skip */
      }
    }
  }
  return { events, remainder: remaining };
}

export default function DixHallpikeClient() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const overlayRef = useRef<HTMLCanvasElement>(null);
  const graphRef = useRef<HTMLCanvasElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const faceLandmarkerRef = useRef<FaceLandmarkerInstance | null>(null);
  const runningRef = useRef(false);
  const rafRef = useRef<number>(0);
  const startTimeRef = useRef(0);
  const frameCountRef = useRef(0);
  const irisHistoryRef = useRef<Array<{ pos: { x: number; y: number }; time: number }>>([]);
  const velocityHistoryRef = useRef<Velocity[]>([]);
  const beatStateRef = useRef<BeatState>(createBeatState());

  const [mpReady, setMpReady] = useState(false);
  const [mpError, setMpError] = useState<string | null>(null);
  const [sourceMode, setSourceMode] = useState<SourceMode>("webcam");
  const [tracking, setTracking] = useState(false);
  const [fps, setFps] = useState(0);
  const [velH, setVelH] = useState(0);
  const [velV, setVelV] = useState(0);
  const [freq, setFreq] = useState(0);
  const [direction, setDirection] = useState("-");
  const [latency, setLatency] = useState<number | null>(null);
  const [beats, setBeats] = useState(0);
  const [fatigable, setFatigable] = useState("-");
  const [duration, setDuration] = useState<number | null>(null);

  const [torsion, setTorsion] = useState<TorsionDirection>("");
  const [dixResult, setDixResult] = useState<DixHallpikeResult | "">("");
  const [confirmError, setConfirmError] = useState<string | null>(null);

  const [isStreaming, setIsStreaming] = useState(false);
  const [pipelineResult, setPipelineResult] = useState<PipelineResult | null>(null);
  const [pipelineError, setPipelineError] = useState<string | null>(null);
  const [completedStages, setCompletedStages] = useState<Set<StageName>>(new Set());
  const [activeStage, setActiveStage] = useState<StageName | null>(null);
  const abortRef = useRef<AbortController | null>(null);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const mp = await loadMediaPipe();
        if (cancelled) return;
        const filesetResolver = new mp.FilesetResolver(
          "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.18/wasm"
        );
        faceLandmarkerRef.current = await mp.FaceLandmarker.createFromOptions(
          filesetResolver,
          {
            baseOptions: {
              modelAssetPath:
                "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
              delegate: "GPU",
            },
            runningMode: "VIDEO",
            numFaces: 1,
            outputFaceBlendshapes: false,
            outputFacialTransformationMatrixes: false,
          }
        );
        setMpReady(true);
      } catch (err) {
        if (!cancelled) setMpError(err instanceof Error ? err.message : String(err));
      }
    })();
    return () => {
      cancelled = true;
      faceLandmarkerRef.current?.close();
      faceLandmarkerRef.current = null;
    };
  }, []);

  const drawOverlay = useCallback(
    (landmarks: Array<{ x: number; y: number }>, vw: number, vh: number) => {
      const canvas = overlayRef.current;
      if (!canvas) return;
      const ctx = canvas.getContext("2d");
      if (!ctx) return;
      ctx.clearRect(0, 0, canvas.width, canvas.height);

      const allIris = [...LEFT_IRIS_INDICES, ...RIGHT_IRIS_INDICES];
      ctx.fillStyle = "#00ff00";
      for (const idx of allIris) {
        const lm = landmarks[idx];
        ctx.beginPath();
        ctx.arc(lm.x * vw, lm.y * vh, 3, 0, 2 * Math.PI);
        ctx.fill();
      }

      const lc = getIrisCenter(landmarks, LEFT_IRIS_INDICES);
      const rc = getIrisCenter(landmarks, RIGHT_IRIS_INDICES);
      ctx.fillStyle = "#ff0000";
      ctx.beginPath();
      ctx.arc(lc.x * vw, lc.y * vh, 5, 0, 2 * Math.PI);
      ctx.fill();
      ctx.beginPath();
      ctx.arc(rc.x * vw, rc.y * vh, 5, 0, 2 * Math.PI);
      ctx.fill();
    },
    []
  );

  const drawGraph = useCallback(() => {
    const canvas = graphRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.fillStyle = "#1a1a2e";
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    const hist = velocityHistoryRef.current;
    if (hist.length < 2) return;

    const midY = canvas.height / 2;
    const scale = 2;
    const step = canvas.width / MAX_HISTORY;

    ctx.strokeStyle = "#333";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(0, midY);
    ctx.lineTo(canvas.width, midY);
    ctx.stroke();

    ctx.strokeStyle = "#ff4a4a";
    ctx.lineWidth = 2;
    ctx.beginPath();
    for (let i = 0; i < hist.length; i++) {
      const x = i * step;
      const y = midY - hist[i].vx * scale;
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();

    ctx.strokeStyle = "#4a9eff";
    ctx.beginPath();
    for (let i = 0; i < hist.length; i++) {
      const x = i * step;
      const y = midY - hist[i].vy * scale;
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();

    ctx.fillStyle = "#ff4a4a";
    ctx.font = "12px monospace";
    ctx.fillText("H", 10, 16);
    ctx.fillStyle = "#4a9eff";
    ctx.fillText("V", 30, 16);
  }, []);

  const processFrame = useCallback(() => {
    if (!runningRef.current) return;
    const video = videoRef.current;
    const fl = faceLandmarkerRef.current;
    if (!video || !fl) return;

    const currentTime = (performance.now() - startTimeRef.current) / 1000;

    if (video.readyState >= 2) {
      const timestampMs =
        video.currentTime > 0 ? video.currentTime * 1000 : performance.now();
      const result = fl.detectForVideo(video, timestampMs);

      if (result.faceLandmarks && result.faceLandmarks.length > 0) {
        const face = result.faceLandmarks[0];
        const leftIris = getIrisCenter(face, LEFT_IRIS_INDICES);
        const rightIris = getIrisCenter(face, RIGHT_IRIS_INDICES);
        const avgIris = {
          x: (leftIris.x + rightIris.x) / 2,
          y: (leftIris.y + rightIris.y) / 2,
        };

        const hist = irisHistoryRef.current;
        const prev = hist[hist.length - 1];
        const dt = prev ? currentTime - prev.time : 0;
        const velocity = calculateVelocity(
          avgIris,
          prev?.pos ?? null,
          dt,
          video.videoWidth,
          video.videoHeight
        );

        hist.push({ pos: avgIris, time: currentTime });
        velocityHistoryRef.current.push(velocity);
        if (hist.length > MAX_HISTORY) {
          hist.shift();
          velocityHistoryRef.current.shift();
        }

        detectBeat(velocity, currentTime, beatStateRef.current, velocityHistoryRef.current);
        drawOverlay(face, video.videoWidth, video.videoHeight);
      }
      frameCountRef.current++;
    }

    const elapsed = (performance.now() - startTimeRef.current) / 1000;
    setFps(elapsed > 0 ? frameCountRef.current / elapsed : 0);

    const vh = velocityHistoryRef.current;
    if (vh.length > 0) {
      const last = vh[vh.length - 1];
      setVelH(last.vx);
      setVelV(last.vy);
    }

    const bs = beatStateRef.current;
    setFreq(computeFrequency(bs.beatTimes));
    setDirection(computeDirection(vh));
    setLatency(bs.firstBeatTime);
    setBeats(bs.beatTimes.length);
    setFatigable(computeFatigability(bs.amplitudes));
    setDuration(
      bs.beatTimes.length >= 2
        ? bs.beatTimes[bs.beatTimes.length - 1] - bs.beatTimes[0]
        : null
    );

    drawGraph();
    rafRef.current = requestAnimationFrame(processFrame);
  }, [drawOverlay, drawGraph]);

  const stopTracking = useCallback(() => {
    runningRef.current = false;
    if (rafRef.current) cancelAnimationFrame(rafRef.current);
    const video = videoRef.current;
    if (video?.srcObject) {
      (video.srcObject as MediaStream).getTracks().forEach((t) => t.stop());
      video.srcObject = null;
    }
    if (video?.src && video.src.startsWith("blob:")) {
      URL.revokeObjectURL(video.src);
      video.src = "";
    }
    const canvas = overlayRef.current;
    if (canvas) {
      const ctx = canvas.getContext("2d");
      ctx?.clearRect(0, 0, canvas.width, canvas.height);
    }
    setTracking(false);
  }, []);

  const startWebcam = useCallback(async () => {
    const video = videoRef.current;
    const fl = faceLandmarkerRef.current;
    if (!video || !fl) return;
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: 640, height: 480, facingMode: "user" },
      });
      video.srcObject = stream;
      await video.play();
      const overlay = overlayRef.current;
      if (overlay) {
        overlay.width = video.videoWidth;
        overlay.height = video.videoHeight;
      }
      runningRef.current = true;
      startTimeRef.current = performance.now();
      frameCountRef.current = 0;
      irisHistoryRef.current = [];
      velocityHistoryRef.current = [];
      beatStateRef.current = createBeatState();
      setTracking(true);
      setPipelineResult(null);
      setPipelineError(null);
      processFrame();
    } catch (err) {
      setMpError(err instanceof Error ? err.message : String(err));
    }
  }, [processFrame]);

  const handleFileLoad = useCallback(
    (e: ChangeEvent<HTMLInputElement>) => {
      const file = e.target.files?.[0];
      if (!file) return;
      const video = videoRef.current;
      if (!video) return;
      const url = URL.createObjectURL(file);
      video.src = url;
      video.onloadedmetadata = () => {
        const overlay = overlayRef.current;
        if (overlay) {
          overlay.width = video.videoWidth;
          overlay.height = video.videoHeight;
        }
      };
      video.play().then(() => {
        runningRef.current = true;
        startTimeRef.current = performance.now();
        frameCountRef.current = 0;
        irisHistoryRef.current = [];
        velocityHistoryRef.current = [];
        beatStateRef.current = createBeatState();
        setTracking(true);
        setPipelineResult(null);
        setPipelineError(null);
        processFrame();
      });
    },
    [processFrame]
  );

  const buildCaseFeatures = useCallback((): CaseFeatures => {
    const nystagmusFeatures = toNystagmusFeatures(
      beatStateRef.current,
      velocityHistoryRef.current
    );

    const features: CaseFeatures = {
      ...nystagmusFeatures,
      trigger: "positional_head",
      timing_pattern: "episodic_triggered",
      duration: "under_1min",
      onset: "sudden",
      torsion_confirmed_by_clinician: torsion !== "",
      nystagmus_latency_s: nystagmusFeatures.nystagmus_latency_s ?? undefined,
      nystagmus_duration_s: nystagmusFeatures.nystagmus_duration_s ?? undefined,
      nystagmus_fatigable: nystagmusFeatures.nystagmus_fatigable ?? undefined,
    };

    if (dixResult) {
      features.dix_hallpike = dixResult as DixHallpikeResult;
    }

    if (
      nystagmusFeatures.nystagmus_direction === "torsional_pure" ||
      torsion === "right_ear" ||
      torsion === "left_ear"
    ) {
      features.nystagmus_direction = "mixed";
    }

    return features;
  }, [torsion, dixResult]);

  const handleSendToPipeline = useCallback(async () => {
    if (!torsion) {
      setConfirmError("La confirmación de torsión es obligatoria antes de enviar.");
      return;
    }
    if (!dixResult) {
      setConfirmError("El resultado del Dix-Hallpike es obligatorio.");
      return;
    }
    setConfirmError(null);

    if (abortRef.current) abortRef.current.abort();
    const controller = new AbortController();
    abortRef.current = controller;

    const features = buildCaseFeatures();

    setIsStreaming(true);
    setCompletedStages(new Set());
    setActiveStage(null);
    setPipelineResult(null);
    setPipelineError(null);

    try {
      const response = await fetch(`${API_URL}/api/evaluate`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(features),
        signal: controller.signal,
      });
      if (!response.ok) throw new Error(`HTTP ${response.status}`);

      const reader = response.body?.getReader();
      if (!reader) throw new Error("ReadableStream not supported");

      const decoder = new TextDecoder();
      let buffer = "";
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const { events, remainder } = parseSSEEvents(buffer);
        buffer = remainder;
        for (const evt of events) {
          if (evt.stage === "done") {
            setCompletedStages((p) => new Set(p).add("done"));
            setActiveStage(null);
            setPipelineResult(evt.data as PipelineResult);
          } else if (evt.stage === "error") {
            const errData = evt.data as { error: string; message: string };
            setPipelineError(`${errData.error}: ${errData.message}`);
            setActiveStage(null);
          } else {
            setCompletedStages((p) => new Set(p).add(evt.stage));
            setActiveStage(evt.stage);
          }
        }
      }
    } catch (err: unknown) {
      if (err instanceof DOMException && err.name === "AbortError") return;
      setPipelineError(
        err instanceof Error ? err.message : "Error de conexión con el backend"
      );
    } finally {
      setIsStreaming(false);
      setActiveStage(null);
    }
  }, [buildCaseFeatures, torsion, dixResult]);

  const handleReset = useCallback(() => {
    stopTracking();
    irisHistoryRef.current = [];
    velocityHistoryRef.current = [];
    beatStateRef.current = createBeatState();
    setFps(0);
    setVelH(0);
    setVelV(0);
    setFreq(0);
    setDirection("-");
    setLatency(null);
    setBeats(0);
    setFatigable("-");
    setDuration(null);
    setTorsion("");
    setDixResult("");
    setConfirmError(null);
    setPipelineResult(null);
    setPipelineError(null);
    if (fileInputRef.current) fileInputRef.current.value = "";
  }, [stopTracking]);

  const STAGE_ORDER: { key: StageName; label: string }[] = [
    { key: "redflag", label: "RedFlag" },
    { key: "differential", label: "Differential" },
    { key: "ml", label: "ML" },
    { key: "reasoning", label: "Reasoning" },
    { key: "rails", label: "Rails" },
    { key: "done", label: "Done" },
  ];

  return (
    <div className="dix-hallpike">
      <div className="dh-header">
        <h2>Maniobra de Dix-Hallpike — Tier 1</h2>
        <p className="dh-privacy">
          <strong>Privacidad:</strong> Todo el procesamiento es on-device.
          Solo features numéricas desidentificadas se envían al backend.
          El video nunca sale del dispositivo.
        </p>
      </div>

      {mpError && (
        <div className="error-panel">
          <strong>Error MediaPipe:</strong> {mpError}
        </div>
      )}

      {!mpReady && !mpError && (
        <div className="status-info">
          <span className="spinner" /> Cargando MediaPipe FaceLandmarker...
        </div>
      )}

      {mpReady && (
        <>
          <div className="dh-controls">
            <div className="dh-source-toggle">
              <button
                className={`btn-secondary${sourceMode === "webcam" ? " active" : ""}`}
                onClick={() => setSourceMode("webcam")}
                type="button"
                disabled={tracking}
              >
                Webcam
              </button>
              <button
                className={`btn-secondary${sourceMode === "file" ? " active" : ""}`}
                onClick={() => setSourceMode("file")}
                type="button"
                disabled={tracking}
              >
                Video local
              </button>
            </div>

            {sourceMode === "webcam" && (
              <button
                className="btn-primary"
                onClick={startWebcam}
                type="button"
                disabled={tracking}
              >
                Iniciar webcam
              </button>
            )}

            {sourceMode === "file" && (
              <label className="btn-primary" style={{ cursor: "pointer" }}>
                Cargar video
                <input
                  ref={fileInputRef}
                  type="file"
                  accept="video/*"
                  onChange={handleFileLoad}
                  style={{ display: "none" }}
                />
              </label>
            )}

            <button
              className="btn-secondary"
              onClick={stopTracking}
              type="button"
              disabled={!tracking}
            >
              Detener
            </button>

            <button
              className="btn-secondary"
              onClick={handleReset}
              type="button"
            >
              Reset
            </button>
          </div>

          <div className="dh-video-area">
            <div className="dh-video-container">
              <video ref={videoRef} playsInline muted style={{ maxWidth: "100%", borderRadius: 8 }} />
              <canvas
                ref={overlayRef}
                style={{ position: "absolute", top: 0, left: 0, pointerEvents: "none" }}
              />
              {tracking && (
                <div className="dh-fps-badge">
                  {fps.toFixed(1)} FPS
                </div>
              )}
            </div>

            <canvas
              ref={graphRef}
              width={600}
              height={150}
              style={{ borderRadius: 8, maxWidth: "100%", marginTop: 12 }}
            />
          </div>

          <div className="dh-metrics">
            <div className="dh-metric">
              <span className="dh-metric-label">FPS</span>
              <span className="dh-metric-value">{fps.toFixed(1)}</span>
            </div>
            <div className="dh-metric">
              <span className="dh-metric-label">Vel. H (°/s)</span>
              <span className="dh-metric-value">{velH.toFixed(1)}</span>
            </div>
            <div className="dh-metric">
              <span className="dh-metric-label">Vel. V (°/s)</span>
              <span className="dh-metric-value">{velV.toFixed(1)}</span>
            </div>
            <div className="dh-metric">
              <span className="dh-metric-label">Frecuencia (Hz)</span>
              <span className="dh-metric-value">{freq.toFixed(2)}</span>
            </div>
            <div className="dh-metric">
              <span className="dh-metric-label">Dirección</span>
              <span className="dh-metric-value">{direction}</span>
            </div>
            <div className="dh-metric">
              <span className="dh-metric-label">Latencia (s)</span>
              <span className="dh-metric-value">{latency !== null ? latency.toFixed(2) : "-"}</span>
            </div>
            <div className="dh-metric">
              <span className="dh-metric-label">Batidas</span>
              <span className="dh-metric-value">{beats}</span>
            </div>
            <div className="dh-metric">
              <span className="dh-metric-label">Duración (s)</span>
              <span className="dh-metric-value">{duration !== null ? duration.toFixed(2) : "-"}</span>
            </div>
            <div className="dh-metric">
              <span className="dh-metric-label">Fatigable</span>
              <span className="dh-metric-value">{fatigable}</span>
            </div>
          </div>

          <div className="dh-confirm-section">
            <h3>Confirmación del médico (obligatoria)</h3>
            <p className="dh-confirm-note">
              La torsión NO se auto-trackea — el médico confirma la dirección torsional
              observada (intervención humana, Ley 21.719).
            </p>

            <div className="dh-confirm-group">
              <label>Dirección torsional observada:</label>
              <div className="dh-radio-group">
                <label className="dh-radio">
                  <input
                    type="radio"
                    name="torsion"
                    value="right_ear"
                    checked={torsion === "right_ear"}
                    onChange={() => setTorsion("right_ear")}
                  />
                  Torsión hacia oído derecho
                </label>
                <label className="dh-radio">
                  <input
                    type="radio"
                    name="torsion"
                    value="left_ear"
                    checked={torsion === "left_ear"}
                    onChange={() => setTorsion("left_ear")}
                  />
                  Torsión hacia oído izquierdo
                </label>
                <label className="dh-radio">
                  <input
                    type="radio"
                    name="torsion"
                    value="none"
                    checked={torsion === "none"}
                    onChange={() => setTorsion("none")}
                  />
                  No observada
                </label>
              </div>
            </div>

            <div className="dh-confirm-group">
              <label>Resultado Dix-Hallpike:</label>
              <select
                className="dh-select"
                value={dixResult}
                onChange={(e) => setDixResult(e.target.value as DixHallpikeResult)}
              >
                <option value="">— Seleccionar —</option>
                <option value="right_positive">Derecho positivo</option>
                <option value="left_positive">Izquierdo positivo</option>
                <option value="bilateral_positive">Bilateral positivo</option>
                <option value="negative">Negativo</option>
              </select>
            </div>

            {confirmError && (
              <div className="error-panel" style={{ marginTop: 8 }}>
                {confirmError}
              </div>
            )}

            <button
              className="btn-primary"
              onClick={handleSendToPipeline}
              type="button"
              disabled={isStreaming || !tracking}
              style={{ marginTop: 12 }}
            >
              {isStreaming ? (
                <>
                  <span className="spinner" /> Evaluando...
                </>
              ) : (
                "Confirmar y evaluar"
              )}
            </button>
          </div>

          {(isStreaming || pipelineResult || pipelineError || completedStages.size > 0) && (
            <div className="dh-pipeline-section">
              <h3>Pipeline de evaluación</h3>

              <div className="pipeline-stages">
                {STAGE_ORDER.map(({ key, label }) => {
                  let cls = "stage-pill";
                  if (completedStages.has(key)) cls += " completed";
                  else if (activeStage === key) cls += " active";
                  return (
                    <span key={key} className={cls}>
                      {(activeStage === key ||
                        (key === "done" && completedStages.has("done"))) && (
                        <span className="spinner" />
                      )}
                      {completedStages.has(key) && key !== "done" && "✓"}
                      {label}
                    </span>
                  );
                })}
              </div>

              {pipelineError && (
                <div className="error-panel">
                  <strong>Error:</strong> {pipelineError}
                </div>
              )}

              {pipelineResult && <PipelineResultPanel result={pipelineResult} />}
            </div>
          )}
        </>
      )}
    </div>
  );
}

const DIAGNOSIS_LABELS: Record<string, string> = {
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

const URGENCY_LABELS: Record<string, string> = {
  inmediata: "Inmediata",
  prioritaria: "Prioritaria",
  ambulatoria: "Ambulatoria",
};

function PipelineResultPanel({ result }: { result: PipelineResult }) {
  const safetyActive =
    result.red_flag.red_flag_activa && result.urgency === "inmediata";

  return (
    <div className="result-panel">
      <div style={{ marginBottom: "1rem" }}>
        <span className={`urgency-badge ${result.urgency}`}>
          {URGENCY_LABELS[result.urgency] ?? result.urgency}
        </span>
      </div>

      {safetyActive && (
        <div className="safety-banner">
          <h3>Riel de seguridad activo</h3>
          <p>
            Red flag activa — el guardián determinista fuerza la acción de
            seguridad independientemente del ML/LLM.
          </p>
        </div>
      )}

      <div className="result-section">
        <h4>Diagnóstico diferencial</h4>
        <ul className="candidate-list">
          {result.differential.candidates.map((c) => (
            <li key={c.diagnosis} className="candidate-item">
              <span className="candidate-name">
                {DIAGNOSIS_LABELS[c.diagnosis] ?? c.diagnosis}
              </span>
              <div className="candidate-bar">
                <div
                  className="candidate-bar-fill"
                  style={{ width: `${(c.score * 100).toFixed(0)}%` }}
                />
              </div>
              <span className="candidate-score">
                {(c.score * 100).toFixed(0)}%
              </span>
            </li>
          ))}
        </ul>
      </div>

      {result.reasoning && (
        <div className="result-section">
          <h4>Razonamiento clínico</h4>
          <div className="reasoning-block">
            <p>{result.reasoning.explanation}</p>
            {result.reasoning.reconciliation && (
              <p>
                <strong>Conciliación:</strong> {result.reasoning.reconciliation}
              </p>
            )}
            <p style={{ fontSize: "0.8rem", color: "var(--color-text-muted)" }}>
              Modelo: {result.reasoning.model_used}
            </p>
          </div>
        </div>
      )}

      {result.fhir_bundle && (
        <div className="result-section">
          <h4>Artefacto auditable (FHIR)</h4>
          <details>
            <summary>Ver bundle FHIR</summary>
            <pre style={{ maxHeight: 300, overflow: "auto", fontSize: "0.75rem" }}>
              {JSON.stringify(result.fhir_bundle, null, 2)}
            </pre>
          </details>
        </div>
      )}
    </div>
  );
}
