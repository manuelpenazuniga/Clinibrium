import DixHallpikeClient from "../components/DixHallpikeClient";

export const metadata = {
  title: "Dix-Hallpike — Clinibrium",
  description:
    "Maniobra de Dix-Hallpike con tracking on-device de nistagmus (MediaPipe). El video nunca sale del dispositivo.",
};

export default function DixHallpikePage() {
  return (
    <main className="app-container">
      <header className="header">
        <h1>Clinibrium — Dix-Hallpike</h1>
        <p>
          Medición determinista on-device de nistagmus ·{" "}
          <a href="/">Volver al pipeline</a>
        </p>
        <div className="disclaimer">
          El médico decide — apoyo diagnóstico, no diagnóstico autónomo.
        </div>
      </header>
      <DixHallpikeClient />
    </main>
  );
}
