"use client";

import { useEffect, useRef } from "react";

/**
 * Trazo tipo nistagmo: deriva lenta + corrección rápida (sawtooth), como un
 * registro de video-oculografía. Puramente ilustrativo — no son datos.
 */
export default function NystagmusTrace() {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const reduced = window.matchMedia(
      "(prefers-reduced-motion: reduce)"
    ).matches;

    let raf = 0;
    let width = 0;
    let height = 0;

    const resize = () => {
      const dpr = window.devicePixelRatio || 1;
      width = canvas.clientWidth;
      height = canvas.clientHeight;
      canvas.width = Math.floor(width * dpr);
      canvas.height = Math.floor(height * dpr);
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    };
    resize();
    const observer = new ResizeObserver(() => {
      resize();
      if (reduced) draw(STATIC_T);
    });
    observer.observe(canvas);

    // Componente horizontal: fase lenta (82% del ciclo) / fase rápida (18%),
    // ~0.9 Hz, amplitud modulada para que respire como señal biológica.
    const sampleH = (t: number) => {
      const beat = 1.15;
      const p = ((t % beat) + beat) % beat / beat;
      const saw = p < 0.82 ? p / 0.82 : 1 - (p - 0.82) / 0.18;
      const amp = 0.55 + 0.35 * Math.sin(t * 0.31) * Math.sin(t * 0.13);
      const noise = 0.04 * Math.sin(t * 7.3) + 0.03 * Math.sin(t * 11.7);
      return (saw - 0.5) * amp + noise;
    };
    const sampleV = (t: number) =>
      0.1 * Math.sin(t * 2.1) + 0.05 * Math.sin(t * 5.7);

    const SECONDS_VISIBLE = 9;
    const STATIC_T = 4200;

    const trace = (
      fn: (t: number) => number,
      color: string,
      lineWidth: number,
      scale: number,
      t0: number
    ) => {
      ctx.strokeStyle = color;
      ctx.lineWidth = lineWidth;
      ctx.lineJoin = "round";
      ctx.beginPath();
      const steps = Math.max(140, Math.floor(width / 3));
      for (let i = 0; i <= steps; i++) {
        const x = (i / steps) * width;
        const t = t0 - SECONDS_VISIBLE + (i / steps) * SECONDS_VISIBLE;
        const y = height / 2 - fn(t) * scale;
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }
      ctx.stroke();
    };

    const draw = (now: number) => {
      const t0 = now / 1000;
      ctx.clearRect(0, 0, width, height);
      trace(sampleV, "rgba(24, 38, 52, 0.26)", 1.25, height * 0.3, t0);
      trace(sampleH, "#0B6B70", 1.75, height * 0.34, t0);
    };

    if (reduced) {
      draw(STATIC_T);
    } else {
      const loop = (now: number) => {
        draw(now);
        raf = requestAnimationFrame(loop);
      };
      raf = requestAnimationFrame(loop);
    }

    return () => {
      cancelAnimationFrame(raf);
      observer.disconnect();
    };
  }, []);

  return (
    <canvas ref={canvasRef} className="nystagmus-trace" aria-hidden="true" />
  );
}
