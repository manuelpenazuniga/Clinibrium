import type { Metadata } from "next";
import DixHallpikeClient from "../components/DixHallpikeClient";

export const metadata: Metadata = {
  title: "Dix-Hallpike — Clinibrium",
  description:
    "Maniobra de Dix-Hallpike con tracking on-device de nistagmo (MediaPipe). El video nunca sale del dispositivo.",
};

export default function DixHallpikePage() {
  return (
    <main className="app-container">
      <header className="page-header">
        <p className="eyebrow">Módulo Dix-Hallpike · Tier 1</p>
        <h1>Medición on-device de nistagmo</h1>
        <p className="page-lede">
          El video se procesa localmente con MediaPipe — 0 frames cruzan la
          red. Solo features numéricas desidentificadas llegan al backend, y la
          torsión la confirma el médico.
        </p>
      </header>
      <DixHallpikeClient />
    </main>
  );
}
