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
  StageName,
} from "@/lib/types";
import { streamEvaluation } from "@/lib/api";
import { uiErrorText, type Dict, type UiError } from "@/lib/i18n";
import { useLanguage } from "./LanguageProvider";
import ClinicalCaseReceipt from "./ClinicalCaseReceipt";
import PipelineRail from "./PipelineRail";
import PrivacyEgressMeter from "./PrivacyEgressMeter";

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
  FilesetResolver: {
    forVisionTasks: (basePath: string) => Promise<unknown>;
  };
}

async function loadMediaPipe(): Promise<MediaPipeModules> {
  const vision = await import("@mediapipe/tasks-vision");
  return {
    FaceLandmarker: vision.FaceLandmarker as unknown as MediaPipeModules["FaceLandmarker"],
    FilesetResolver: vision.FilesetResolver as unknown as MediaPipeModules["FilesetResolver"],
  };
}

export default function DixHallpikeClient() {
  const { lang, t } = useLanguage();
  const d = t.dix;
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
  // Stored as a dictionary KEY so a mid-session language toggle re-localizes
  // the visible alert (codex-audit-4 Baja 1), not the already-translated text.
  const [confirmError, setConfirmError] = useState<
    "errTorsionRequired" | "errDixRequired" | null
  >(null);

  const [isStreaming, setIsStreaming] = useState(false);
  const [pipelineResult, setPipelineResult] = useState<PipelineResult | null>(null);
  const [pipelineError, setPipelineError] = useState<UiError | null>(null);
  const [completedStages, setCompletedStages] = useState<Set<StageName>>(new Set());
  const [activeStage, setActiveStage] = useState<StageName | null>(null);
  const [framesProcessed, setFramesProcessed] = useState(0);
  const [sentFeatures, setSentFeatures] = useState<CaseFeatures | null>(null);
  const abortRef = useRef<AbortController | null>(null);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const mp = await loadMediaPipe();
        if (cancelled) return;
        // The CDN wasm must match the installed version of the JS package.
        const filesetResolver = await mp.FilesetResolver.forVisionTasks(
          "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.35/wasm"
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
    ctx.fillStyle = "#12202B";
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    const hist = velocityHistoryRef.current;
    if (hist.length < 2) return;

    const midY = canvas.height / 2;
    const scale = 2;
    const step = canvas.width / MAX_HISTORY;

    ctx.strokeStyle = "rgba(255, 255, 255, 0.14)";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(0, midY);
    ctx.lineTo(canvas.width, midY);
    ctx.stroke();

    // H = teal (typical peripheral pattern) · V = red (pure vertical = central sign)
    ctx.strokeStyle = "#4FC1BC";
    ctx.lineWidth = 2;
    ctx.beginPath();
    for (let i = 0; i < hist.length; i++) {
      const x = i * step;
      const y = midY - hist[i].vx * scale;
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();

    ctx.strokeStyle = "#F0837A";
    ctx.beginPath();
    for (let i = 0; i < hist.length; i++) {
      const x = i * step;
      const y = midY - hist[i].vy * scale;
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();

    ctx.fillStyle = "#4FC1BC";
    ctx.font = "12px monospace";
    ctx.fillText("H", 10, 16);
    ctx.fillStyle = "#F0837A";
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
      if (frameCountRef.current % 10 === 0) {
        setFramesProcessed(frameCountRef.current);
      }
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
      setFramesProcessed(0);
      setSentFeatures(null);
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
        setFramesProcessed(0);
        setSentFeatures(null);
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
      torsion_confirmed_by_clinician:
        torsion === "right_ear" || torsion === "left_ear",
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
      setConfirmError("errTorsionRequired");
      return;
    }
    if (!dixResult) {
      setConfirmError("errDixRequired");
      return;
    }
    setConfirmError(null);

    if (abortRef.current) abortRef.current.abort();
    const controller = new AbortController();
    abortRef.current = controller;

    const features = buildCaseFeatures();
    setSentFeatures(features);

    setIsStreaming(true);
    setCompletedStages(new Set());
    setActiveStage(null);
    setPipelineResult(null);
    setPipelineError(null);

    try {
      const final = await streamEvaluation(features, {
        lang,
        signal: controller.signal,
        onStage: (evt) => {
          if (evt.stage === "done") {
            setCompletedStages((p) => new Set(p).add("done"));
            setActiveStage(null);
          } else if (evt.stage !== "error") {
            setCompletedStages((p) => new Set(p).add(evt.stage));
            setActiveStage(evt.stage);
          }
        },
      });
      setPipelineResult(final);
    } catch (err: unknown) {
      if (err instanceof DOMException && err.name === "AbortError") return;
      setPipelineError(
        err instanceof Error ? { message: err.message } : { key: "connectionError" }
      );
    } finally {
      setIsStreaming(false);
      setActiveStage(null);
    }
  }, [buildCaseFeatures, torsion, dixResult, lang]);

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
    setFramesProcessed(0);
    setSentFeatures(null);
    if (fileInputRef.current) fileInputRef.current.value = "";
  }, [stopTracking]);

  const sessionStarted =
    tracking || framesProcessed > 0 || pipelineResult !== null;

  return (
    <div className="dix-hallpike">
      {mpError && (
        <div className="notice notice-error" role="alert">
          <strong>{d.mpError}</strong> {mpError}
        </div>
      )}

      {!mpReady && !mpError && (
        <div className="status-info">
          <span className="spinner" /> {d.loadingMp}
        </div>
      )}

      {mpReady && (
        <>
          {!sessionStarted && (
            <div className="dh-intro">
              <div className="dh-intro-step">
                <span className="dh-intro-number">1</span>
                <h3>{d.intro1Title}</h3>
                <p>{d.intro1Body}</p>
              </div>
              <div className="dh-intro-step">
                <span className="dh-intro-number">2</span>
                <h3>{d.intro2Title}</h3>
                <p>{d.intro2Body}</p>
              </div>
              <div className="dh-intro-step">
                <span className="dh-intro-number">3</span>
                <h3>{d.intro3Title}</h3>
                <p>{d.intro3Body}</p>
              </div>
            </div>
          )}

          <PrivacyEgressMeter
            framesProcessed={framesProcessed}
            features={sentFeatures}
          />

          <div className="dh-controls">
            <div className="dh-source-toggle">
              <button
                className={`btn-secondary${sourceMode === "webcam" ? " active" : ""}`}
                onClick={() => setSourceMode("webcam")}
                type="button"
                disabled={tracking}
              >
                {d.webcam}
              </button>
              <button
                className={`btn-secondary${sourceMode === "file" ? " active" : ""}`}
                onClick={() => setSourceMode("file")}
                type="button"
                disabled={tracking}
              >
                {d.localVideo}
              </button>
            </div>

            {sourceMode === "webcam" && (
              <button
                className="btn-primary"
                onClick={startWebcam}
                type="button"
                disabled={tracking}
              >
                {d.startWebcam}
              </button>
            )}

            {sourceMode === "file" && (
              <label className="btn-primary btn-file">
                {d.loadVideo}
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
              {d.stop}
            </button>

            <button
              className="btn-secondary"
              onClick={handleReset}
              type="button"
            >
              {d.reset}
            </button>
          </div>

          <div className="dh-video-area">
            <div className="dh-video-container">
              <video ref={videoRef} playsInline muted className="dh-video" />
              <canvas ref={overlayRef} className="dh-overlay" />
              {tracking && (
                <div className="dh-fps-badge">
                  {fps.toFixed(1)} FPS
                </div>
              )}
            </div>

            <canvas ref={graphRef} width={600} height={150} className="dh-graph" />
          </div>

          <div className="dh-metrics">
            <div className="dh-metric">
              <span className="dh-metric-label">{d.metricFps}</span>
              <span className="dh-metric-value">{fps.toFixed(1)}</span>
            </div>
            <div className="dh-metric">
              <span className="dh-metric-label">{d.metricVelH}</span>
              <span className="dh-metric-value">{velH.toFixed(1)}</span>
            </div>
            <div className="dh-metric">
              <span className="dh-metric-label">{d.metricVelV}</span>
              <span className="dh-metric-value">{velV.toFixed(1)}</span>
            </div>
            <div className="dh-metric">
              <span className="dh-metric-label">{d.metricFreq}</span>
              <span className="dh-metric-value">{freq.toFixed(2)}</span>
            </div>
            <div className="dh-metric">
              <span className="dh-metric-label">{d.metricDirection}</span>
              <span className="dh-metric-value">
                {t.direction[direction as keyof Dict["direction"]] ?? direction}
              </span>
            </div>
            <div className="dh-metric">
              <span className="dh-metric-label">{d.metricLatency}</span>
              <span className="dh-metric-value">{latency !== null ? latency.toFixed(2) : "-"}</span>
            </div>
            <div className="dh-metric">
              <span className="dh-metric-label">{d.metricBeats}</span>
              <span className="dh-metric-value">{beats}</span>
            </div>
            <div className="dh-metric">
              <span className="dh-metric-label">{d.metricDuration}</span>
              <span className="dh-metric-value">{duration !== null ? duration.toFixed(2) : "-"}</span>
            </div>
            <div className="dh-metric">
              <span className="dh-metric-label">{d.metricFatigable}</span>
              <span className="dh-metric-value">
                {t.fatigable[fatigable as keyof Dict["fatigable"]] ?? fatigable}
              </span>
            </div>
          </div>

          <p className="dh-experimental-note">{d.experimentalNote}</p>

          <div className="dh-confirm-section">
            <h3>{d.confirmHeading}</h3>
            <p className="dh-confirm-note">{d.confirmNote}</p>

            <div className="dh-confirm-group">
              <label>{d.torsionLabel}</label>
              <div className="dh-radio-group">
                <label className="dh-radio">
                  <input
                    type="radio"
                    name="torsion"
                    value="right_ear"
                    checked={torsion === "right_ear"}
                    onChange={() => setTorsion("right_ear")}
                  />
                  {d.torsionRight}
                </label>
                <label className="dh-radio">
                  <input
                    type="radio"
                    name="torsion"
                    value="left_ear"
                    checked={torsion === "left_ear"}
                    onChange={() => setTorsion("left_ear")}
                  />
                  {d.torsionLeft}
                </label>
                <label className="dh-radio">
                  <input
                    type="radio"
                    name="torsion"
                    value="none"
                    checked={torsion === "none"}
                    onChange={() => setTorsion("none")}
                  />
                  {d.torsionNone}
                </label>
              </div>
            </div>

            <div className="dh-confirm-group">
              <label>{d.dixLabel}</label>
              <select
                className="dh-select"
                value={dixResult}
                onChange={(e) => setDixResult(e.target.value as DixHallpikeResult)}
              >
                <option value="">{d.dixSelect}</option>
                <option value="right_positive">{d.dixRight}</option>
                <option value="left_positive">{d.dixLeft}</option>
                <option value="bilateral_positive">{d.dixBilateral}</option>
                <option value="negative">{d.dixNegative}</option>
              </select>
            </div>

            {confirmError && (
              <div className="notice notice-error dh-confirm-error" role="alert">
                {d[confirmError]}
              </div>
            )}

            <button
              className="btn-primary dh-confirm-submit"
              onClick={handleSendToPipeline}
              type="button"
              disabled={isStreaming || !tracking}
            >
              {isStreaming ? (
                <>
                  <span className="spinner" /> {d.evaluatingShort}
                </>
              ) : (
                d.confirmSubmit
              )}
            </button>
          </div>

          {(isStreaming || pipelineResult || pipelineError || completedStages.size > 0) && (
            <div className="dh-pipeline-section">
              <h3>{d.pipelineHeading}</h3>

              <PipelineRail
                completed={completedStages}
                active={activeStage}
                hasError={pipelineError !== null}
              />

              {pipelineError && (
                <div className="notice notice-error" role="alert">
                  <strong>{t.common.error}</strong> {uiErrorText(pipelineError, t)}
                </div>
              )}

              {pipelineResult && <ClinicalCaseReceipt result={pipelineResult} />}
            </div>
          )}
        </>
      )}
    </div>
  );
}
