import type { Metadata } from "next";
import PipelineDemo from "../components/PipelineDemo";

export const metadata: Metadata = {
  title: "Demo — Clinibrium",
  description:
    "Pipeline de evaluación en vivo: red flags deterministas, diferencial ICVD, ML opcional, Claude explica, rieles sellan y el médico decide.",
};

export default function DemoPage() {
  return (
    <main className="app-container">
      <header className="page-header">
        <p className="eyebrow">Demo interactiva</p>
        <h1>Pipeline de evaluación en vivo</h1>
        <p className="page-lede">
          Elige un caso, corre el pipeline real por SSE y comprueba las tres
          propiedades: los rieles ganan, la degradación es segura y todo queda
          en un recibo auditable.
        </p>
      </header>
      <PipelineDemo />
    </main>
  );
}
