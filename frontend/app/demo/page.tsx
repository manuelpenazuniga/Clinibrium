import type { Metadata } from "next";
import PipelineDemo from "../components/PipelineDemo";
import PageHeader from "../components/PageHeader";

export const metadata: Metadata = {
  title: "Demo — Clinibrium",
  description:
    "Pipeline de evaluación en vivo: red flags deterministas, diferencial ICVD, ML opcional, Claude explica, rieles sellan y el médico decide.",
};

export default function DemoPage() {
  return (
    <main className="app-container">
      <PageHeader kind="demo" />
      <PipelineDemo />
    </main>
  );
}
