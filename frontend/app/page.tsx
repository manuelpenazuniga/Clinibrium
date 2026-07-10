import PipelineDemo from "./components/PipelineDemo";
import Link from "next/link";

export default function Home() {
  return (
    <main className="app-container">
      <header className="header">
        <h1>Clinibrium</h1>
        <p>Apoyo diagnóstico otoneurológico — VertigoDx</p>
        <div className="disclaimer">
          El médico decide — apoyo diagnóstico, no diagnóstico autónomo.
        </div>
        <nav style={{ marginTop: "1rem" }}>
          <Link href="/dix-hallpike" className="btn-secondary">
            Dix-Hallpike (Tier 1)
          </Link>
        </nav>
      </header>
      <PipelineDemo />
    </main>
  );
}
