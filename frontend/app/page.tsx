import Link from "next/link";
import NystagmusTrace from "./components/NystagmusTrace";

const PROPERTIES = [
  {
    number: "P1",
    invariant: "INV-1",
    title: "Una red flag nunca puede ser anulada",
    body: "Si el motor determinista de red flags dispara, la urgencia es inmediata — sin importar lo que digan el ML o Claude. El riel se aplica después del LLM y gana siempre.",
  },
  {
    number: "P2",
    invariant: "INV-6 · INV-8",
    title: "La seguridad no depende de los modelos",
    body: "Si el servicio de ML o la API de Claude caen, el pipeline completa igual y la urgencia no cambia. En la demo puedes matarlos en vivo y comprobarlo.",
  },
  {
    number: "P3",
    invariant: "INV-4",
    title: "El médico revisa, interviene y verifica",
    body: "Cada evaluación emite exactamente un AuditEvent y un recibo clínico con hash SHA-256 verificable. La decisión final — aceptar o rechazar — queda registrada.",
  },
];

const PIPELINE_NODES = [
  {
    tag: "entrada",
    name: "Features desidentificadas",
    note: "Solo campos del allowlist cruzan la red — ni PII, ni texto libre, ni video (INV-2).",
    kind: "input",
  },
  {
    tag: "determinista",
    name: "RedFlagEngine",
    note: "¿Es emergencia? Funciones puras, físicamente separado por régimen regulatorio.",
    kind: "deterministic",
  },
  {
    tag: "determinista",
    name: "DifferentialEngine",
    note: "Reglas ICVD con scoring determinista — el pool de candidatos.",
    kind: "deterministic",
  },
  {
    tag: "aditivo",
    name: "ML · track B",
    note: "CatBoost jerárquico con gate monótono de peligro. Si cae, nada cambia.",
    kind: "additive",
  },
  {
    tag: "aditivo",
    name: "Claude",
    note: "Explica y concilia con criterios ICVD parafraseados (RAG). No clasifica ni fija urgencia.",
    kind: "additive",
  },
  {
    tag: "sello",
    name: "Rieles",
    note: "Invariantes duros aplicados después de Claude. Solo suben la urgencia, nunca la bajan.",
    kind: "seal",
  },
  {
    tag: "decisión",
    name: "Médico",
    note: "Acepta o rechaza con justificación — intervención humana registrada en el AuditEvent.",
    kind: "human",
  },
];

const PRIVACY_STATS = [
  {
    value: "0",
    label: "frames de video a la red",
    note: "El tracking ocular corre on-device con MediaPipe; el video nunca sale del dispositivo.",
  },
  {
    value: "allowlist",
    label: "features que cruzan la red",
    note: "Validador fail-closed (INV-2): cualquier campo fuera del allowlist se rechaza.",
  },
  {
    value: "1",
    label: "AuditEvent por evaluación",
    note: "Exactamente uno, garantizado por diseño — incluso si el pipeline falla.",
  },
  {
    value: "SHA-256",
    label: "integridad del artefacto FHIR",
    note: "Bundle R4 tamper-evident; el hash se verifica en tu propio browser.",
  },
];

const LIMITATIONS = [
  "No es un dispositivo médico ni está aprobado para uso clínico.",
  "Los umbrales y pesos clínicos son provisionales, pendientes de firma del otoneurólogo validador.",
  "El tracking de nistagmo es experimental: velocidades relativas, sin calibración validada a °/s. La torsión la confirma el médico.",
  "El artefacto es un FHIR R4 Clinical Case Bundle (perfiles CL Core donde existen), no un IPS-CL completo.",
];

export default function LandingPage() {
  return (
    <main className="landing">
      <section className="hero">
        <div className="container">
          <p className="eyebrow">
            VertigoDx Engine · Apoyo diagnóstico otoneurológico
          </p>
          <h1 className="hero-title">
            El modelo explica.
            <br />
            Los rieles protegen.
            <br />
            <em>El médico decide.</em>
          </h1>
          <p className="hero-lede">
            Clinibrium es un agente clínico para vértigo agudo que demuestra
            cómo fallar de forma segura: reglas deterministas fijan la
            urgencia, Claude la explica con criterios ICVD parafraseados, y
            cada evaluación deja un recibo auditable que el médico acepta o
            rechaza.
          </p>
          <div className="hero-actions">
            <Link href="/demo" className="btn-primary btn-lg">
              Explorar la demo
            </Link>
            <Link href="/dix-hallpike" className="btn-secondary btn-lg">
              Maniobra Dix-Hallpike
            </Link>
          </div>
        </div>
        <figure className="hero-trace">
          <NystagmusTrace />
          <figcaption>
            Trazo tipo nistagmo — deriva lenta, corrección rápida: el signo que
            el sistema ayuda a documentar. Ilustrativo.
          </figcaption>
        </figure>
      </section>

      <section className="landing-section">
        <div className="container">
          <p className="eyebrow">Propiedades verificables</p>
          <h2 className="section-heading">
            Lo que la demo prueba, no lo que afirma
          </h2>
          <div className="property-grid">
            {PROPERTIES.map((p) => (
              <article key={p.number} className="property-card">
                <div className="property-head">
                  <span className="property-number">{p.number}</span>
                  <span className="property-invariant">{p.invariant}</span>
                </div>
                <h3>{p.title}</h3>
                <p>{p.body}</p>
              </article>
            ))}
          </div>
        </div>
      </section>

      <section className="landing-section landing-section-alt">
        <div className="container">
          <p className="eyebrow">Arquitectura</p>
          <h2 className="section-heading">
            Un pipeline donde el LLM no puede fijar la urgencia
          </h2>
          <p className="section-lede">
            Las capas deterministas corren primero y sellan al final. ML y
            Claude son aditivos: aportan contexto, no deciden seguridad.
          </p>
          <ol className="rail-diagram">
            {PIPELINE_NODES.map((node) => (
              <li key={node.name} className={`rail-node rail-${node.kind}`}>
                <span className="rail-tag">{node.tag}</span>
                <span className="rail-name">{node.name}</span>
                <span className="rail-note">{node.note}</span>
              </li>
            ))}
          </ol>
        </div>
      </section>

      <section className="landing-section">
        <div className="container">
          <p className="eyebrow">Privacidad verificable</p>
          <h2 className="section-heading">
            Lo que cruza la red se puede contar
          </h2>
          <div className="privacy-grid">
            {PRIVACY_STATS.map((stat) => (
              <div key={stat.label} className="privacy-stat">
                <span className="privacy-value">{stat.value}</span>
                <span className="privacy-label">{stat.label}</span>
                <p className="privacy-note">{stat.note}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      <section className="landing-section landing-section-alt">
        <div className="container">
          <p className="eyebrow">Claims honestos</p>
          <h2 className="section-heading">Lo que este prototipo no es</h2>
          <ul className="limitations-list">
            {LIMITATIONS.map((item) => (
              <li key={item}>{item}</li>
            ))}
          </ul>
        </div>
      </section>

      <section className="landing-cta">
        <div className="container">
          <h2 className="cta-heading">
            Mata a Claude en vivo y mira que la urgencia no se mueve.
          </h2>
          <Link href="/demo" className="btn-primary btn-lg">
            Explorar la demo
          </Link>
        </div>
      </section>
    </main>
  );
}
