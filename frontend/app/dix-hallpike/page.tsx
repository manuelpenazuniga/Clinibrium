import type { Metadata } from "next";
import DixHallpikeClient from "../components/DixHallpikeClient";
import PageHeader from "../components/PageHeader";

export const metadata: Metadata = {
  title: "Dix-Hallpike — Clinibrium",
  description:
    "Maniobra de Dix-Hallpike con tracking on-device de nistagmo (MediaPipe). El video nunca sale del dispositivo.",
};

export default function DixHallpikePage() {
  return (
    <main className="app-container">
      <PageHeader kind="dix" />
      <DixHallpikeClient />
    </main>
  );
}
