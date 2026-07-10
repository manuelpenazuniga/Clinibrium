import PipelineDemo from "./components/PipelineDemo";

export default function Home() {
  return (
    <main className="app-container">
      <header className="header">
        <h1>Clinibrium</h1>
        <p>Apoyo diagnóstico otoneurológico — VertigoDx</p>
        <div className="disclaimer">
          El médico decide — apoyo diagnóstico, no diagnóstico autónomo.
        </div>
      </header>
      <PipelineDemo />
    </main>
  );
}
