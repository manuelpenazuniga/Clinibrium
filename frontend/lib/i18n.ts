/**
 * Hand-authored bilingual dictionary (single source of truth for UI copy).
 *
 * - Spanish (`es`) is the DEFAULT and its values are byte-identical to the
 *   pre-toggle copy (the recorded demo depends on this — do not reword `es`).
 * - English (`en`) is hand-written for hackathon judges. NO runtime machine
 *   translation, no external service.
 * - `en` is typed as `typeof es`, so the compiler enforces that both languages
 *   expose exactly the same keys (missing/extra keys fail `tsc`).
 *
 * Enum values (Urgency/ForcedAction/Diagnosis) and computed direction strings
 * are CANONICAL — we translate only their DISPLAY here, never the value.
 */
import type { CaseFeatures } from "./types";

export type Lang = "es" | "en";

const es = {
  common: {
    error: "Error:",
    connectionError: "Error de conexión con el backend",
    noPipelineResult: "No se recibió resultado del pipeline",
    spinnerEvaluating: "Evaluando…",
  },

  header: {
    navAria: "Navegación principal",
    nav: { home: "Inicio", demo: "Demo", dix: "Dix-Hallpike" },
    prototypeChip: "Prototipo · no uso clínico",
    prototypeChipTitle: "Prototipo de hackathon — no destinado a uso clínico real",
    langToggleAria: "Idioma / Language",
    langEs: "ES",
    langEn: "EN",
  },

  skipLink: "Saltar al contenido",

  // Big landing selector + welcome-card toggle. The option labels stay in
  // their OWN language on purpose ("Español" / "English" never translate).
  langSelect: {
    aria: "Idioma / Language",
    es: "Español",
    en: "English",
  },

  welcome: {
    skip: "Saltar introducción",
    back: "Atrás",
    next: "Siguiente",
    enter: "Entrar a Clinibrium",
    progressAria: "Progreso de la introducción",
    stepCount: (i: number, n: number) => `Paso ${i} de ${n}`,
    goToStep: (i: number) => `Ir al paso ${i}`,
    steps: [
      {
        kicker: "Bienvenida",
        title: "El modelo explica. Los rieles protegen. El médico decide.",
        body:
          "Clinibrium es un agente clínico para vértigo agudo construido para fallar de forma segura. Antes de empezar, esta guía te muestra qué hace cada pieza — y qué puedes verificar tú mismo, sin confiar en nadie.",
      },
      {
        kicker: "El caso",
        title: "Todo empieza con un caso clínico desidentificado",
        body:
          "Eliges uno de tres presets reales: VPPB benigno, sospecha de stroke (HINTS central) y Ménière. Cada tarjeta muestra exactamente las features que se envían — nada más cruza la red: ni PII, ni texto libre, ni video.",
      },
      {
        kicker: "El pipeline",
        title: "Las capas deterministas deciden; las demás explican",
        body:
          "La evaluación corre en vivo, etapa por etapa: red flags y diferencial deterministas primero, ML y Claude como capas aditivas que aportan contexto. La urgencia la fijan las reglas — nunca el modelo.",
      },
      {
        kicker: "La prueba",
        title: "Puedes matar a Claude y la seguridad no se mueve",
        body:
          "Un interruptor simula la caída del razonador. Re-evalúa y compruébalo: la urgencia y las red flags quedan idénticas. La seguridad no depende del LLM — esa es la tesis de todo el proyecto.",
      },
      {
        kicker: "El recibo",
        title: "Cada evaluación deja un recibo auditable",
        body:
          "Urgencia, rieles disparados, hash SHA-256 verificable en tu propio browser y exactamente un AuditEvent. La decisión final — aceptar o rechazar, con justificación — la toma y la firma el médico.",
      },
      {
        kicker: "La privacidad",
        title: "El video nunca sale de tu dispositivo",
        body:
          "El módulo Dix-Hallpike mide el nistagmo on-device con MediaPipe: 0 frames cruzan la red, y el Privacy Egress Meter te deja contar cada byte que sale. Todo listo — la aplicación te espera.",
      },
    ],
  },

  footer: {
    thesis: "El modelo explica. Los rieles protegen. El médico decide.",
    builtPrefix: "Construido durante ",
    builtStrong: "Built with Claude: Life Sciences",
    builtSuffix:
      " (julio 2026) con Claude Code. La lógica clínica proviene de investigación previa del equipo, divulgada como prior-art.",
    disclaimer:
      "Prototipo de investigación — no aprobado para uso clínico. Umbrales y pesos provisionales, pendientes de validación del especialista.",
  },

  landing: {
    heroEyebrow: "VertigoDx Engine · Apoyo diagnóstico otoneurológico",
    heroTitle1: "El modelo explica.",
    heroTitle2: "Los rieles protegen.",
    heroTitle3: "El médico decide.",
    heroLede:
      "Clinibrium es un agente clínico para vértigo agudo que demuestra cómo fallar de forma segura: reglas deterministas fijan la urgencia, Claude la explica con criterios ICVD parafraseados, y cada evaluación deja un recibo auditable que el médico acepta o rechaza.",
    heroCtaDemo: "Explorar la demo",
    heroCtaDix: "Maniobra Dix-Hallpike",
    heroCaption:
      "Trazo tipo nistagmo — deriva lenta, corrección rápida: el signo que el sistema ayuda a documentar. Ilustrativo.",
    propsEyebrow: "Propiedades verificables",
    propsHeading: "Lo que la demo prueba, no lo que afirma",
    properties: [
      {
        title: "Una red flag nunca puede ser anulada",
        body:
          "Si el motor determinista de red flags dispara, la urgencia es inmediata — sin importar lo que digan el ML o Claude. El riel se aplica después del LLM y gana siempre.",
      },
      {
        title: "La seguridad no depende de los modelos",
        body:
          "Si el servicio de ML o la API de Claude caen, el pipeline completa igual y la urgencia no cambia. En la demo puedes matarlos en vivo y comprobarlo.",
      },
      {
        title: "El médico revisa, interviene y verifica",
        body:
          "Cada evaluación emite exactamente un AuditEvent y un recibo clínico con hash SHA-256 verificable. La decisión final — aceptar o rechazar — queda registrada.",
      },
    ],
    archEyebrow: "Arquitectura",
    archHeading: "Un pipeline donde el LLM no puede fijar la urgencia",
    archLede:
      "Las capas deterministas corren primero y sellan al final. ML y Claude son aditivos: aportan contexto, no deciden seguridad.",
    nodes: [
      {
        tag: "entrada",
        name: "Features desidentificadas",
        note: "Solo campos del allowlist cruzan la red — ni PII, ni texto libre, ni video (INV-2).",
      },
      {
        tag: "determinista",
        name: "RedFlagEngine",
        note: "¿Es emergencia? Funciones puras, físicamente separado por régimen regulatorio.",
      },
      {
        tag: "determinista",
        name: "DifferentialEngine",
        note: "Reglas ICVD con scoring determinista — el pool de candidatos.",
      },
      {
        tag: "aditivo",
        name: "ML · track B",
        note: "CatBoost jerárquico con gate monótono de peligro. Si cae, nada cambia.",
      },
      {
        tag: "aditivo",
        name: "Claude",
        note: "Explica y concilia con criterios ICVD parafraseados (RAG). No clasifica ni fija urgencia.",
      },
      {
        tag: "sello",
        name: "Rieles",
        note: "Invariantes duros aplicados después de Claude. Solo suben la urgencia, nunca la bajan.",
      },
      {
        tag: "decisión",
        name: "Médico",
        note: "Acepta o rechaza con justificación — intervención humana registrada en el AuditEvent.",
      },
    ],
    privacyEyebrow: "Privacidad verificable",
    privacyHeading: "Lo que cruza la red se puede contar",
    privacyStats: [
      {
        label: "frames de video a la red",
        note: "El tracking ocular corre on-device con MediaPipe; el video nunca sale del dispositivo.",
      },
      {
        label: "features que cruzan la red",
        note: "Validador fail-closed (INV-2): cualquier campo fuera del allowlist se rechaza.",
      },
      {
        label: "AuditEvent por evaluación",
        note: "Exactamente uno, garantizado por diseño — incluso si el pipeline falla.",
      },
      {
        label: "integridad del artefacto FHIR",
        note: "Bundle R4 tamper-evident; el hash se verifica en tu propio browser.",
      },
    ],
    honestEyebrow: "Claims honestos",
    honestHeading: "Lo que este prototipo no es",
    limitations: [
      "No es un dispositivo médico ni está aprobado para uso clínico.",
      "Los umbrales y pesos clínicos son provisionales, pendientes de firma del otoneurólogo validador.",
      "El tracking de nistagmo es experimental: velocidades relativas, sin calibración validada a °/s. La torsión la confirma el médico.",
      "El artefacto es un FHIR R4 Clinical Case Bundle (perfiles CL Core donde existen), no un IPS-CL completo.",
    ],
    ctaHeading: "Mata a Claude en vivo y mira que la urgencia no se mueve.",
    ctaButton: "Explorar la demo",
  },

  demoPage: {
    eyebrow: "Demo interactiva",
    heading: "Pipeline de evaluación en vivo",
    lede:
      "Elige un caso, corre el pipeline real por SSE y comprueba las tres propiedades: los rieles ganan, la degradación es segura y todo queda en un recibo auditable.",
  },

  demo: {
    seeGuide: "Ver guía",
    step1: "Paso 1",
    step1Title: "Elige un caso clínico",
    step1TitleReal: "Ingresa el caso clínico real",
    modeAria: "Modo de ingreso del caso",
    modePresets: "Casos de ejemplo",
    modeReal: "Caso real",
    presetFeatures: "features desidentificadas · allowlist INV-2",
    killLabel: "Kill Claude — simular caída del razonador",
    evaluate: "Evaluar caso",
    emptyHint:
      "Selecciona un caso para habilitar la evaluación. El pipeline corre de verdad contra el backend.",
    degradedTitle: "Modo degradado activo.",
    degradedBody:
      " El razonador se simulará caído: la urgencia y las red flags permanecerán idénticas — la seguridad no depende del LLM (INV-8).",
    step2: "Paso 2",
    step2Title: "Pipeline en tiempo real",
    backendHintPrefix: "¿Está corriendo el backend en el puerto 8000? Revisa ",
    step3: "Paso 3",
    step3Title: "Recibo clínico y decisión",
    safetyKicker: "Seguridad probada",
    safetyTitle: "Red flag activa ⇒ urgencia inmediata",
    safetyBody:
      "Este veredicto viene del RedFlagEngine determinista. Ni el ML ni Claude pueden anularlo — el riel se aplica después y gana siempre (INV-1).",
    stage: {
      redflagActive: "Red flag activa:",
      yes: "SÍ",
      no: "No",
      hitsSuffix: (n: number) => ` — ${n} hallazgo(s) de alarma`,
      noCandidates: "Sin candidatos",
      mlAvailable: "Modelo ML disponible",
      mlUnavailable: "ML no disponible (track B degradado) — el pipeline continúa",
      reasonerDegraded:
        "Razonador degradado — el pipeline continúa; la urgencia no depende del LLM (INV-8).",
      model: "Modelo:",
      rails: "Rieles:",
      forcedActions: "Acciones forzadas:",
      noRails: "Sin rieles aplicados",
    },
  },

  realCase: {
    hint:
      "Modo aplicación real: ingresa las features estructuradas de tu propio caso y córrelas por el mismo pipeline que la demo. Solo los campos que completes cruzan la red — desidentificados y validados fail-closed contra el allowlist INV-2. Nunca ingreses nombre, RUT ni texto libre.",
    clear: "Limpiar formulario",
    featureCount: (n: number) => `${n} feature(s) estructuradas listas para enviar`,
    noFeatures: "Completa al menos un campo del caso para habilitar la evaluación.",
    payloadSummary: "Ver payload exacto que cruzará la red",
    notEvaluated: "— no evaluado —",
    sections: {
      history: "Historia",
      exam: "Examen vestibular",
      hearing: "Audición",
      alarm: "Signos de alarma y contexto",
    },
    fields: {
      age: "Edad (años)",
      duration: "Duración del síntoma",
      onset: "Inicio",
      trigger: "Desencadenante",
      timingPattern: "Patrón temporal",
      episodeCount: "N° de episodios",
      episodeDuration: "Duración de cada episodio",
      nystagmusDirection: "Dirección del nistagmo",
      nystagmusLatency: "Latencia del nistagmo (s)",
      nystagmusDuration: "Duración del nistagmo (s)",
      headImpulse: "Head impulse test (HIT)",
      dixHallpike: "Dix-Hallpike",
      hearingLoss: "Hipoacusia",
      focalSigns: "Signos focales",
      vascularRisk: "Factores de riesgo vascular",
    },
    onsetOpt: { sudden: "Súbito", gradual: "Gradual", unknown: "Desconocido" },
    timingOpt: {
      acute_continuous: "Agudo continuo",
      episodic_triggered: "Episódico gatillado",
      episodic_spontaneous: "Episódico espontáneo",
      chronic: "Crónico",
    },
    nystagmusOpt: {
      none: "Sin nistagmo",
      horizontal: "Horizontal",
      vertical_pure: "Vertical puro",
      torsional_pure: "Torsional puro",
      mixed: "Mixto",
      direction_changing: "Cambiante con la mirada",
    },
    hitOpt: {
      normal: "Normal",
      abnormal_corrective_saccade: "Anormal (sacada correctiva)",
      not_done: "No realizado",
    },
    hearingOpt: {
      none: "Sin hipoacusia",
      sudden_unilateral: "Súbita unilateral",
      fluctuating: "Fluctuante",
      chronic: "Crónica",
    },
    dixOpt: {
      right_positive: "Derecho positivo",
      left_positive: "Izquierdo positivo",
      bilateral_positive: "Bilateral positivo",
      negative: "Negativo",
      not_done: "No realizado",
    },
    focalOpt: {
      dysarthria: "Disartria",
      dysphagia: "Disfagia",
      diplopia: "Diplopía",
      limb_weakness: "Debilidad de extremidades",
      facial_droop: "Paresia facial",
      numbness: "Entumecimiento / parestesias",
      hiccups: "Hipo persistente",
      horner: "Síndrome de Horner",
    },
    vascularOpt: {
      hypertension: "Hipertensión",
      diabetes: "Diabetes",
      atrial_fibrillation: "Fibrilación auricular",
      smoking: "Tabaquismo",
      prior_stroke_tia: "ACV / TIA previo",
    },
    checks: {
      nystagmus_direction_changing_gaze: "Nistagmo cambia con la mirada",
      nystagmus_fatigable: "Nistagmo fatigable",
      nystagmus_suppressed_by_fixation: "Suprime con fijación visual",
      skew_deviation: "Skew deviation",
      torsion_confirmed_by_clinician: "Torsión confirmada por el médico",
      truncal_ataxia_severe: "Ataxia troncal severa",
      tinnitus: "Tinnitus",
      aural_fullness: "Plenitud aural",
      headache_neck_pain_sudden_severe: "Cefalea / cervicalgia súbita severa",
      migrainous_features: "Rasgos migrañosos",
      fever: "Fiebre",
      neck_stiffness: "Rigidez de nuca",
      altered_consciousness: "Compromiso de conciencia",
      presyncope_syncope: "Presíncope / síncope",
      palpitations: "Palpitaciones",
      chest_pain: "Dolor torácico",
      otitis_mastoiditis: "Otitis / mastoiditis",
      recent_head_neck_trauma: "Trauma craneocervical reciente",
      cervical_pathology: "Patología cervical",
      known_carotid_vertebrobasilar_disease:
        "Enfermedad carotídea / vertebrobasilar conocida",
      cardiovascular_instability: "Inestabilidad cardiovascular",
      worsening_during_flow: "Empeoramiento durante la evolución",
    },
  },

  receipt: {
    kicker: "Clinical Case Receipt",
    summaryHeading: "Resumen clínico",
    railsFired: "Rieles disparados",
    redFlags: "Red flags",
    differential: "Diferencial",
    reasoningHeading: "Razonamiento — Claude explica, no decide",
    reconciliation: "Conciliación:",
    nextSteps: "Próximos pasos sugeridos",
    reasonerDegradedNote:
      "Razonador degradado — el pipeline completó sin Claude. Urgencia, red flags y acciones forzadas son idénticas a la corrida con razonador (INV-8).",
    provenanceHeading: "Procedencia",
    model: "Modelo",
    reasoner: "Reasoner",
    grounding: "Grounding",
    groundingNote: "paráfrasis fuente en español",
    integrityHeading: "Integridad verificable",
    verify: "Verificar en este browser",
    verifying: "Verificando…",
    hashOk: "✓ Íntegro — el hash coincide",
    hashFail: "✗ Alterado",
    hashInfo:
      "Hash del servidor (canónico JS no coincide exacto — verificar con backend)",
    decisionHeading: "Decisión del médico",
    decisionNote:
      "La decisión queda registrada en el AuditEvent — intervención humana (Ley 21.719).",
    decisionPlaceholder: "Justificación clínica (opcional)",
    decisionFailed:
      "No se pudo registrar la decisión — revisa la conexión con el backend e intenta de nuevo.",
    accept: "Aceptar",
    reject: "Rechazar",
    registering: "Registrando…",
    interventionHeading: "Intervención registrada",
    auditEvent: "AuditEvent",
    type: "Tipo",
    registered: "Registrado",
    downloadFhir: "Descargar Bundle FHIR R4 (.json)",
    fhirFootnote: "FHIR R4 Clinical Case Bundle — perfiles CL Core donde existen.",
  },

  wwc: {
    heading: "¿Qué cambiaría el manejo?",
    intro:
      "Un solo hallazgo a la vez, corrido por el pipeline determinista real — qué buscar antes de tranquilizar al paciente. El LLM no decide qué es urgente; los rieles verifican cada contrafactual (INV-3).",
    analyzing: "Analizando contrafactuales…",
    analyze: "Analizar: ¿qué cambiaría el manejo?",
    minimalKicker: "Cambio mínimo que escala",
    leadsFrom: " lleva el caso de ",
    to: " a ",
    firesRail: " — dispara el riel ",
    noneChange: "Ningún hallazgo único cambia el manejo (base: ",
    tableFinding: "Hallazgo agregado (1 variable)",
    tableUrgency: "Urgencia",
    tableRail: "Riel",
    note:
      "Cada fila cambia exactamente una variable y se corre por las capas deterministas (RedFlagEngine + rieles). Resultados verificables, no una opinión del modelo.",
    noteExactlyOne: "exactamente una",
  },

  onboarding: {
    skip: "Saltar guía",
    back: "Atrás",
    next: "Siguiente",
    start: "Empezar",
    steps: [
      {
        title: "1 · Elige un caso clínico",
        body:
          "Tres presets reales: VPPB benigno, sospecha de stroke (HINTS central) y Ménière. Cada tarjeta muestra las features desidentificadas que se envían — nada más cruza la red.",
      },
      {
        title: "2 · Corre el pipeline en vivo",
        body:
          "La evaluación llega por streaming, etapa por etapa: red flags y diferencial deterministas primero, ML y Claude como capas aditivas, y los rieles sellando al final.",
      },
      {
        title: "3 · Mata a Claude",
        body:
          "Activa este interruptor y re-evalúa: la urgencia y las red flags no cambian aunque el razonador caiga. La seguridad no depende del LLM — esa es la tesis.",
      },
      {
        title: "4 · Recibo clínico y decisión",
        body:
          "Al final obtienes un recibo auditable: urgencia, rieles disparados, hash SHA-256 verificable en tu browser, y la decisión del médico registrada en el AuditEvent. Prueba también el fork clínico: una variable cambia y el riel dispara de verdad.",
      },
    ],
  },

  egress: {
    title: "Privacy Egress Meter",
    framesLocal: "frames procesados localmente",
    framesUploaded: "frames subidos a la red",
    videoFields: "campos de video/frame en el payload",
    bytes: "bytes que salen a la red",
    outboundHead: "payload saliente exacto (a /api/evaluate)",
    shaTitle: "SHA-256 del payload (Web Crypto, verificable)",
    empty: "ninguna aún — procesá un clip para ver el payload",
    notePrefix: "El video se procesa localmente (MediaPipe on-device). Lo que se ve arriba es",
    noteExactly: " exactamente",
    noteMid: " lo que sale a la red: ",
    noteBytes: " bytes de features numéricas desidentificadas, ",
    noteFieldsSuffix: " campos de video/frame, con hash SHA-256 verificable. No es «confiá en nosotros» — es observable (INV-2).",
  },

  dix: {
    pageEyebrow: "Módulo Dix-Hallpike · Tier 1",
    pageHeading: "Medición on-device de nistagmo",
    pageLede:
      "El video se procesa localmente con MediaPipe — 0 frames cruzan la red. Solo features numéricas desidentificadas llegan al backend, y la torsión la confirma el médico.",
    mpError: "Error MediaPipe:",
    loadingMp: "Cargando MediaPipe FaceLandmarker...",
    intro1Title: "Fuente local",
    intro1Body: "Usa la webcam o carga un clip de la maniobra. El video no sale de este dispositivo.",
    intro2Title: "Tracking on-device",
    intro2Body:
      "MediaPipe sigue el iris y estima velocidades relativas, frecuencia y fatigabilidad del nistagmo. Experimental.",
    intro3Title: "El médico confirma",
    intro3Body:
      "La torsión no se auto-trackea: la confirmas tú, junto con el resultado de la maniobra, antes de evaluar.",
    webcam: "Webcam",
    localVideo: "Video local",
    startWebcam: "Iniciar webcam",
    loadVideo: "Cargar video",
    stop: "Detener",
    reset: "Reset",
    metricFps: "FPS",
    metricVelH: "Vel. H (rel.)",
    metricVelV: "Vel. V (rel.)",
    metricFreq: "Frecuencia (Hz)",
    metricDirection: "Dirección",
    metricLatency: "Latencia (s)",
    metricBeats: "Batidas",
    metricDuration: "Duración (s)",
    metricFatigable: "Fatigable",
    experimentalNote:
      "Tracking experimental on-device. Las velocidades son relativas (sin calibración a °/s validada); no es un instrumento de medición clínica certificado. El médico confirma el patrón observado.",
    confirmHeading: "Confirmación del médico (obligatoria)",
    confirmNote:
      "La torsión NO se auto-trackea — el médico confirma la dirección torsional observada (intervención humana, Ley 21.719).",
    torsionLabel: "Dirección torsional observada:",
    torsionRight: "Torsión hacia oído derecho",
    torsionLeft: "Torsión hacia oído izquierdo",
    torsionNone: "No observada",
    dixLabel: "Resultado Dix-Hallpike:",
    dixSelect: "— Seleccionar —",
    dixRight: "Derecho positivo",
    dixLeft: "Izquierdo positivo",
    dixBilateral: "Bilateral positivo",
    dixNegative: "Negativo",
    errTorsionRequired: "La confirmación de torsión es obligatoria antes de enviar.",
    errDixRequired: "El resultado del Dix-Hallpike es obligatorio.",
    confirmSubmit: "Confirmar y evaluar",
    evaluatingShort: "Evaluando...",
    pipelineHeading: "Pipeline de evaluación",
  },

  stages: {
    redflag: { label: "Red flags", note: "determinista" },
    differential: { label: "Diferencial", note: "reglas ICVD" },
    ml: { label: "ML", note: "opcional" },
    reasoning: { label: "Claude", note: "explica" },
    rails: { label: "Rieles", note: "ganan siempre" },
    done: { label: "Recibo", note: "AuditEvent" },
  },

  urgency: {
    inmediata: "Inmediata",
    prioritaria: "Prioritaria",
    ambulatoria: "Ambulatoria",
  },

  diagnosis: {
    bppv_posterior: "VPPB posterior",
    bppv_horizontal: "VPPB horizontal",
    meniere: "Ménière",
    vestibular_migraine: "Migraña vestibular",
    vestibular_neuritis: "Neuritis vestibular",
    labyrinthitis: "Laberintitis",
    central_suspected: "Central (sospecha)",
    cardiogenic_suspected: "Cardiogénico (sospecha)",
    undetermined: "Indeterminado",
  },

  forcedAction: {
    DERIVAR_URGENTE: "Derivación urgente",
    NO_BENIGNO: "No asumir benigno",
    BLOQUEAR_EPLEY: "Epley bloqueado",
    PRECAUCION_EXAMEN: "Precaución en examen",
    RED_SEGURIDAD: "Red de seguridad",
    ESCALAR: "Escalar",
  },

  durationChip: {
    seconds: "segundos",
    under_1min: "< 1 min",
    minutes: "minutos",
    hours: "horas",
    over_24h_continuous: "> 24 h continuo",
    days: "días",
    recurrent_episodic: "episódico recurrente",
  },

  triggerChip: {
    positional_head: "posicional",
    spontaneous: "espontáneo",
    orthostatic: "ortostático",
    valsalva: "Valsalva",
    sound_pressure: "sonido/presión",
    none: "sin gatillo",
  },

  // Display map for lib/nystagmus.computeDirection() return values. The
  // function's RETURN value stays Spanish (tests pin it); we only localize the
  // rendered text keyed by that value.
  direction: {
    Derecha: "Derecha",
    Izquierda: "Izquierda",
    Arriba: "Arriba",
    Abajo: "Abajo",
    "-": "-",
  },

  // Display map for lib/nystagmus.computeFatigability() return values.
  fatigable: { Sí: "Sí", No: "No", "-": "-" },

  // featureChips tokens (see featureChips()).
  chips: {
    hitNormal: "HIT normal",
    directionChanging: "nistagmo cambiante",
    skewDeviation: "skew deviation",
    focalSigns: "signos focales",
    truncalAtaxia: "ataxia troncal",
    fluctuatingHearing: "hipoacusia fluctuante",
    tinnitus: "tinnitus",
    auralFullness: "plenitud aural",
    fatigableNystagmus: "nistagmo fatigable",
    dixPositive: "Dix-Hallpike +",
    years: (n: number) => `${n} años`,
    vascularRisk: (n: number) => `riesgo vascular ×${n}`,
  },

  presets: {
    "bppv-benign": {
      name: "VPPB benigno (ambulatorio)",
      description:
        "Vértigo posicional paroxístico benigno — episodio breve, desencadenado por posición, nistagmus fatigable. Manejo ambulatorio.",
    },
    "stroke-hints": {
      name: "Sospecha de stroke (HINTS central)",
      description:
        "Síndrome vestibular agudo continuo con signos HINTS centrales: head-impulse normal, nistagmus cambiante, skew deviation. Alta sospecha de stroke posterior.",
    },
    "meniere-episodic": {
      name: "Ménière (episódico)",
      description:
        "Enfermedad de Ménière — episodios espontáneos de horas con hipoacusia fluctuante, tinnitus y plenitud aural.",
    },
  },
};

const en: typeof es = {
  common: {
    error: "Error:",
    connectionError: "Connection error with the backend",
    noPipelineResult: "No result received from the pipeline",
    spinnerEvaluating: "Evaluating…",
  },

  header: {
    navAria: "Main navigation",
    nav: { home: "Home", demo: "Demo", dix: "Dix-Hallpike" },
    prototypeChip: "Prototype · not for clinical use",
    prototypeChipTitle: "Hackathon prototype — not intended for real clinical use",
    langToggleAria: "Idioma / Language",
    langEs: "ES",
    langEn: "EN",
  },

  skipLink: "Skip to content",

  langSelect: {
    aria: "Idioma / Language",
    es: "Español",
    en: "English",
  },

  welcome: {
    skip: "Skip intro",
    back: "Back",
    next: "Next",
    enter: "Enter Clinibrium",
    progressAria: "Introduction progress",
    stepCount: (i: number, n: number) => `Step ${i} of ${n}`,
    goToStep: (i: number) => `Go to step ${i}`,
    steps: [
      {
        kicker: "Welcome",
        title: "The model explains. The rails protect. The physician decides.",
        body:
          "Clinibrium is a clinical agent for acute vertigo built to fail safely. Before you start, this guide shows you what each piece does — and what you can verify yourself, without trusting anyone.",
      },
      {
        kicker: "The case",
        title: "Everything starts with a de-identified clinical case",
        body:
          "You pick one of three real presets: benign BPPV, suspected stroke (central HINTS) and Ménière. Each card shows exactly the features that are sent — nothing else crosses the network: no PII, no free text, no video.",
      },
      {
        kicker: "The pipeline",
        title: "The deterministic layers decide; the others explain",
        body:
          "The evaluation runs live, stage by stage: deterministic red flags and differential first, ML and Claude as additive layers that add context. Urgency is set by the rules — never by the model.",
      },
      {
        kicker: "The proof",
        title: "You can kill Claude and safety does not move",
        body:
          "A switch simulates the reasoner going down. Re-evaluate and check it: urgency and red flags stay identical. Safety does not depend on the LLM — that is the thesis of the whole project.",
      },
      {
        kicker: "The receipt",
        title: "Every evaluation leaves an auditable receipt",
        body:
          "Urgency, fired rails, a SHA-256 hash verifiable in your own browser and exactly one AuditEvent. The final decision — accept or reject, with justification — is made and signed by the physician.",
      },
      {
        kicker: "The privacy",
        title: "The video never leaves your device",
        body:
          "The Dix-Hallpike module measures nystagmus on-device with MediaPipe: 0 frames cross the network, and the Privacy Egress Meter lets you count every byte that leaves. All set — the application awaits.",
      },
    ],
  },

  footer: {
    thesis: "The model explains. The rails protect. The physician decides.",
    builtPrefix: "Built during ",
    builtStrong: "Built with Claude: Life Sciences",
    builtSuffix:
      " (July 2026) with Claude Code. The clinical logic comes from the team's prior research, disclosed as prior art.",
    disclaimer:
      "Research prototype — not approved for clinical use. Provisional thresholds and weights, pending specialist validation.",
  },

  landing: {
    heroEyebrow: "VertigoDx Engine · Otoneurological diagnostic support",
    heroTitle1: "The model explains.",
    heroTitle2: "The rails protect.",
    heroTitle3: "The physician decides.",
    heroLede:
      "Clinibrium is a clinical agent for acute vertigo that demonstrates how to fail safely: deterministic rules set urgency, Claude explains it with paraphrased ICVD criteria, and every evaluation leaves an auditable receipt the physician accepts or rejects.",
    heroCtaDemo: "Explore the demo",
    heroCtaDix: "Dix-Hallpike maneuver",
    heroCaption:
      "Nystagmus-like trace — slow drift, fast correction: the sign the system helps document. Illustrative.",
    propsEyebrow: "Verifiable properties",
    propsHeading: "What the demo proves, not what it claims",
    properties: [
      {
        title: "A red flag can never be overridden",
        body:
          "If the deterministic red-flag engine fires, urgency is immediate — regardless of what ML or Claude say. The rail is applied after the LLM and always wins.",
      },
      {
        title: "Safety does not depend on the models",
        body:
          "If the ML service or the Claude API go down, the pipeline still completes and urgency does not change. In the demo you can kill them live and check it.",
      },
      {
        title: "The physician reviews, intervenes and verifies",
        body:
          "Every evaluation emits exactly one AuditEvent and a clinical receipt with a verifiable SHA-256 hash. The final decision — accept or reject — is recorded.",
      },
    ],
    archEyebrow: "Architecture",
    archHeading: "A pipeline where the LLM cannot set urgency",
    archLede:
      "The deterministic layers run first and seal at the end. ML and Claude are additive: they add context, they do not decide safety.",
    nodes: [
      {
        tag: "input",
        name: "De-identified features",
        note: "Only allowlisted fields cross the network — no PII, no free text, no video (INV-2).",
      },
      {
        tag: "deterministic",
        name: "RedFlagEngine",
        note: "Is it an emergency? Pure functions, physically separated by regulatory regime.",
      },
      {
        tag: "deterministic",
        name: "DifferentialEngine",
        note: "ICVD rules with deterministic scoring — the candidate pool.",
      },
      {
        tag: "additive",
        name: "ML · track B",
        note: "Hierarchical CatBoost with a monotone danger gate. If it falls, nothing changes.",
      },
      {
        tag: "additive",
        name: "Claude",
        note: "Explains and reconciles with paraphrased ICVD criteria (RAG). It does not classify or set urgency.",
      },
      {
        tag: "seal",
        name: "Rails",
        note: "Hard invariants applied after Claude. They only raise urgency, never lower it.",
      },
      {
        tag: "decision",
        name: "Physician",
        note: "Accepts or rejects with justification — human intervention recorded in the AuditEvent.",
      },
    ],
    privacyEyebrow: "Verifiable privacy",
    privacyHeading: "What crosses the network can be counted",
    privacyStats: [
      {
        label: "video frames to the network",
        note: "Eye tracking runs on-device with MediaPipe; the video never leaves the device.",
      },
      {
        label: "features that cross the network",
        note: "Fail-closed validator (INV-2): any field outside the allowlist is rejected.",
      },
      {
        label: "AuditEvent per evaluation",
        note: "Exactly one, guaranteed by design — even if the pipeline fails.",
      },
      {
        label: "FHIR artifact integrity",
        note: "Tamper-evident R4 Bundle; the hash is verified in your own browser.",
      },
    ],
    honestEyebrow: "Honest claims",
    honestHeading: "What this prototype is not",
    limitations: [
      "It is not a medical device and is not approved for clinical use.",
      "The clinical thresholds and weights are provisional, pending sign-off by the validating otoneurologist.",
      "Nystagmus tracking is experimental: relative velocities, without calibration validated to °/s. Torsion is confirmed by the physician.",
      "The artifact is a FHIR R4 Clinical Case Bundle (CL Core profiles where they exist), not a complete IPS-CL.",
    ],
    ctaHeading: "Kill Claude live and watch urgency stay put.",
    ctaButton: "Explore the demo",
  },

  demoPage: {
    eyebrow: "Interactive demo",
    heading: "Live evaluation pipeline",
    lede:
      "Pick a case, run the real pipeline over SSE and check the three properties: the rails win, degradation is safe, and everything ends up in an auditable receipt.",
  },

  demo: {
    seeGuide: "View guide",
    step1: "Step 1",
    step1Title: "Pick a clinical case",
    step1TitleReal: "Enter the real clinical case",
    modeAria: "Case input mode",
    modePresets: "Example cases",
    modeReal: "Real case",
    presetFeatures: "de-identified features · INV-2 allowlist",
    killLabel: "Kill Claude — simulate a reasoner outage",
    evaluate: "Evaluate case",
    emptyHint:
      "Select a case to enable evaluation. The pipeline really runs against the backend.",
    degradedTitle: "Degraded mode active.",
    degradedBody:
      " The reasoner will be simulated as down: urgency and red flags stay identical — safety does not depend on the LLM (INV-8).",
    step2: "Step 2",
    step2Title: "Real-time pipeline",
    backendHintPrefix: "Is the backend running on port 8000? Check ",
    step3: "Step 3",
    step3Title: "Clinical receipt and decision",
    safetyKicker: "Safety proven",
    safetyTitle: "Active red flag ⇒ immediate urgency",
    safetyBody:
      "This verdict comes from the deterministic RedFlagEngine. Neither ML nor Claude can override it — the rail is applied afterward and always wins (INV-1).",
    stage: {
      redflagActive: "Red flag active:",
      yes: "YES",
      no: "No",
      hitsSuffix: (n: number) => ` — ${n} alarm finding(s)`,
      noCandidates: "No candidates",
      mlAvailable: "ML model available",
      mlUnavailable: "ML unavailable (track B degraded) — the pipeline continues",
      reasonerDegraded:
        "Reasoner degraded — the pipeline continues; urgency does not depend on the LLM (INV-8).",
      model: "Model:",
      rails: "Rails:",
      forcedActions: "Forced actions:",
      noRails: "No rails applied",
    },
  },

  realCase: {
    hint:
      "Real application mode: enter the structured features of your own case and run them through the same pipeline as the demo. Only the fields you fill in cross the network — de-identified and validated fail-closed against the INV-2 allowlist. Never enter a name, national ID or free text.",
    clear: "Clear form",
    featureCount: (n: number) => `${n} structured feature(s) ready to send`,
    noFeatures: "Fill in at least one field of the case to enable evaluation.",
    payloadSummary: "See the exact payload that will cross the network",
    notEvaluated: "— not assessed —",
    sections: {
      history: "History",
      exam: "Vestibular exam",
      hearing: "Hearing",
      alarm: "Alarm signs and context",
    },
    fields: {
      age: "Age (years)",
      duration: "Symptom duration",
      onset: "Onset",
      trigger: "Trigger",
      timingPattern: "Timing pattern",
      episodeCount: "Episode count",
      episodeDuration: "Duration of each episode",
      nystagmusDirection: "Nystagmus direction",
      nystagmusLatency: "Nystagmus latency (s)",
      nystagmusDuration: "Nystagmus duration (s)",
      headImpulse: "Head impulse test (HIT)",
      dixHallpike: "Dix-Hallpike",
      hearingLoss: "Hearing loss",
      focalSigns: "Focal signs",
      vascularRisk: "Vascular risk factors",
    },
    onsetOpt: { sudden: "Sudden", gradual: "Gradual", unknown: "Unknown" },
    timingOpt: {
      acute_continuous: "Acute continuous",
      episodic_triggered: "Episodic triggered",
      episodic_spontaneous: "Episodic spontaneous",
      chronic: "Chronic",
    },
    nystagmusOpt: {
      none: "No nystagmus",
      horizontal: "Horizontal",
      vertical_pure: "Pure vertical",
      torsional_pure: "Pure torsional",
      mixed: "Mixed",
      direction_changing: "Direction-changing with gaze",
    },
    hitOpt: {
      normal: "Normal",
      abnormal_corrective_saccade: "Abnormal (corrective saccade)",
      not_done: "Not done",
    },
    hearingOpt: {
      none: "No hearing loss",
      sudden_unilateral: "Sudden unilateral",
      fluctuating: "Fluctuating",
      chronic: "Chronic",
    },
    dixOpt: {
      right_positive: "Right positive",
      left_positive: "Left positive",
      bilateral_positive: "Bilateral positive",
      negative: "Negative",
      not_done: "Not done",
    },
    focalOpt: {
      dysarthria: "Dysarthria",
      dysphagia: "Dysphagia",
      diplopia: "Diplopia",
      limb_weakness: "Limb weakness",
      facial_droop: "Facial droop",
      numbness: "Numbness / paresthesia",
      hiccups: "Persistent hiccups",
      horner: "Horner's syndrome",
    },
    vascularOpt: {
      hypertension: "Hypertension",
      diabetes: "Diabetes",
      atrial_fibrillation: "Atrial fibrillation",
      smoking: "Smoking",
      prior_stroke_tia: "Prior stroke / TIA",
    },
    checks: {
      nystagmus_direction_changing_gaze: "Nystagmus changes with gaze",
      nystagmus_fatigable: "Fatigable nystagmus",
      nystagmus_suppressed_by_fixation: "Suppressed by visual fixation",
      skew_deviation: "Skew deviation",
      torsion_confirmed_by_clinician: "Torsion confirmed by the physician",
      truncal_ataxia_severe: "Severe truncal ataxia",
      tinnitus: "Tinnitus",
      aural_fullness: "Aural fullness",
      headache_neck_pain_sudden_severe: "Sudden severe headache / neck pain",
      migrainous_features: "Migrainous features",
      fever: "Fever",
      neck_stiffness: "Neck stiffness",
      altered_consciousness: "Altered consciousness",
      presyncope_syncope: "Presyncope / syncope",
      palpitations: "Palpitations",
      chest_pain: "Chest pain",
      otitis_mastoiditis: "Otitis / mastoiditis",
      recent_head_neck_trauma: "Recent head/neck trauma",
      cervical_pathology: "Cervical pathology",
      known_carotid_vertebrobasilar_disease:
        "Known carotid / vertebrobasilar disease",
      cardiovascular_instability: "Cardiovascular instability",
      worsening_during_flow: "Worsening during the course",
    },
  },

  receipt: {
    kicker: "Clinical Case Receipt",
    summaryHeading: "Clinical summary",
    railsFired: "Rails fired",
    redFlags: "Red flags",
    differential: "Differential",
    reasoningHeading: "Reasoning — Claude explains, does not decide",
    reconciliation: "Reconciliation:",
    nextSteps: "Suggested next steps",
    reasonerDegradedNote:
      "Reasoner degraded — the pipeline completed without Claude. Urgency, red flags and forced actions are identical to the run with the reasoner (INV-8).",
    provenanceHeading: "Provenance",
    model: "Model",
    reasoner: "Reasoner",
    grounding: "Grounding",
    groundingNote: "source paraphrase in Spanish",
    integrityHeading: "Verifiable integrity",
    verify: "Verify in this browser",
    verifying: "Verifying…",
    hashOk: "✓ Intact — the hash matches",
    hashFail: "✗ Altered",
    hashInfo:
      "Server hash (JS canonical does not match exactly — verify with backend)",
    decisionHeading: "Physician's decision",
    decisionNote:
      "The decision is recorded in the AuditEvent — human intervention (Law 21.719).",
    decisionPlaceholder: "Clinical justification (optional)",
    decisionFailed:
      "Could not record the decision — check the connection with the backend and try again.",
    accept: "Accept",
    reject: "Reject",
    registering: "Recording…",
    interventionHeading: "Intervention recorded",
    auditEvent: "AuditEvent",
    type: "Type",
    registered: "Recorded",
    downloadFhir: "Download FHIR R4 Bundle (.json)",
    fhirFootnote: "FHIR R4 Clinical Case Bundle — CL Core profiles where they exist.",
  },

  wwc: {
    heading: "What would change the management?",
    intro:
      "One finding at a time, run through the real deterministic pipeline — what to look for before reassuring the patient. The LLM does not decide what is urgent; the rails verify each counterfactual (INV-3).",
    analyzing: "Analyzing counterfactuals…",
    analyze: "Analyze: what would change the management?",
    minimalKicker: "Minimal change that escalates",
    leadsFrom: " takes the case from ",
    to: " to ",
    firesRail: " — fires rail ",
    noneChange: "No single finding changes the management (base: ",
    tableFinding: "Added finding (1 variable)",
    tableUrgency: "Urgency",
    tableRail: "Rail",
    note:
      "Each row changes exactly one variable and runs through the deterministic layers (RedFlagEngine + rails). Verifiable results, not a model opinion.",
    noteExactlyOne: "exactly one",
  },

  onboarding: {
    skip: "Skip guide",
    back: "Back",
    next: "Next",
    start: "Start",
    steps: [
      {
        title: "1 · Pick a clinical case",
        body:
          "Three real presets: benign BPPV, suspected stroke (central HINTS) and Ménière. Each card shows the de-identified features that are sent — nothing else crosses the network.",
      },
      {
        title: "2 · Run the pipeline live",
        body:
          "The evaluation arrives as a stream, stage by stage: deterministic red flags and differential first, ML and Claude as additive layers, and the rails sealing at the end.",
      },
      {
        title: "3 · Kill Claude",
        body:
          "Flip this switch and re-evaluate: urgency and red flags do not change even if the reasoner falls. Safety does not depend on the LLM — that is the thesis.",
      },
      {
        title: "4 · Clinical receipt and decision",
        body:
          "At the end you get an auditable receipt: urgency, fired rails, a SHA-256 hash verifiable in your browser, and the physician's decision recorded in the AuditEvent. Try the clinical fork too: one variable changes and the rail really fires.",
      },
    ],
  },

  egress: {
    title: "Privacy Egress Meter",
    framesLocal: "frames processed locally",
    framesUploaded: "frames uploaded to the network",
    videoFields: "video/frame fields in the payload",
    bytes: "bytes leaving to the network",
    outboundHead: "exact outbound payload (to /api/evaluate)",
    shaTitle: "SHA-256 of the payload (Web Crypto, verifiable)",
    empty: "none yet — process a clip to see the payload",
    notePrefix: "The video is processed locally (MediaPipe on-device). What you see above is",
    noteExactly: " exactly",
    noteMid: " what leaves to the network: ",
    noteBytes: " bytes of de-identified numeric features, ",
    noteFieldsSuffix: " video/frame fields, with a verifiable SHA-256 hash. It is not «trust us» — it is observable (INV-2).",
  },

  dix: {
    pageEyebrow: "Dix-Hallpike module · Tier 1",
    pageHeading: "On-device nystagmus measurement",
    pageLede:
      "The video is processed locally with MediaPipe — 0 frames cross the network. Only de-identified numeric features reach the backend, and torsion is confirmed by the physician.",
    mpError: "MediaPipe error:",
    loadingMp: "Loading MediaPipe FaceLandmarker...",
    intro1Title: "Local source",
    intro1Body: "Use the webcam or load a clip of the maneuver. The video does not leave this device.",
    intro2Title: "On-device tracking",
    intro2Body:
      "MediaPipe tracks the iris and estimates relative velocities, frequency and fatigability of the nystagmus. Experimental.",
    intro3Title: "The physician confirms",
    intro3Body:
      "Torsion is not auto-tracked: you confirm it, along with the maneuver result, before evaluating.",
    webcam: "Webcam",
    localVideo: "Local video",
    startWebcam: "Start webcam",
    loadVideo: "Load video",
    stop: "Stop",
    reset: "Reset",
    metricFps: "FPS",
    metricVelH: "Vel. H (rel.)",
    metricVelV: "Vel. V (rel.)",
    metricFreq: "Frequency (Hz)",
    metricDirection: "Direction",
    metricLatency: "Latency (s)",
    metricBeats: "Beats",
    metricDuration: "Duration (s)",
    metricFatigable: "Fatigable",
    experimentalNote:
      "Experimental on-device tracking. Velocities are relative (without validated °/s calibration); it is not a certified clinical measurement instrument. The physician confirms the observed pattern.",
    confirmHeading: "Physician confirmation (required)",
    confirmNote:
      "Torsion is NOT auto-tracked — the physician confirms the observed torsional direction (human intervention, Law 21.719).",
    torsionLabel: "Observed torsional direction:",
    torsionRight: "Torsion toward right ear",
    torsionLeft: "Torsion toward left ear",
    torsionNone: "Not observed",
    dixLabel: "Dix-Hallpike result:",
    dixSelect: "— Select —",
    dixRight: "Right positive",
    dixLeft: "Left positive",
    dixBilateral: "Bilateral positive",
    dixNegative: "Negative",
    errTorsionRequired: "Torsion confirmation is required before sending.",
    errDixRequired: "The Dix-Hallpike result is required.",
    confirmSubmit: "Confirm and evaluate",
    evaluatingShort: "Evaluating...",
    pipelineHeading: "Evaluation pipeline",
  },

  stages: {
    redflag: { label: "Red flags", note: "deterministic" },
    differential: { label: "Differential", note: "ICVD rules" },
    ml: { label: "ML", note: "optional" },
    reasoning: { label: "Claude", note: "explains" },
    rails: { label: "Rails", note: "always win" },
    done: { label: "Receipt", note: "AuditEvent" },
  },

  urgency: {
    inmediata: "Immediate",
    prioritaria: "Priority",
    ambulatoria: "Outpatient",
  },

  diagnosis: {
    bppv_posterior: "Posterior BPPV",
    bppv_horizontal: "Horizontal BPPV",
    meniere: "Ménière",
    vestibular_migraine: "Vestibular migraine",
    vestibular_neuritis: "Vestibular neuritis",
    labyrinthitis: "Labyrinthitis",
    central_suspected: "Central (suspected)",
    cardiogenic_suspected: "Cardiogenic (suspected)",
    undetermined: "Undetermined",
  },

  forcedAction: {
    DERIVAR_URGENTE: "Urgent referral",
    NO_BENIGNO: "Do not assume benign",
    BLOQUEAR_EPLEY: "Epley blocked",
    PRECAUCION_EXAMEN: "Exam caution",
    RED_SEGURIDAD: "Safety net",
    ESCALAR: "Escalate",
  },

  durationChip: {
    seconds: "seconds",
    under_1min: "< 1 min",
    minutes: "minutes",
    hours: "hours",
    over_24h_continuous: "> 24 h continuous",
    days: "days",
    recurrent_episodic: "recurrent episodic",
  },

  triggerChip: {
    positional_head: "positional",
    spontaneous: "spontaneous",
    orthostatic: "orthostatic",
    valsalva: "Valsalva",
    sound_pressure: "sound/pressure",
    none: "no trigger",
  },

  direction: {
    Derecha: "Right",
    Izquierda: "Left",
    Arriba: "Up",
    Abajo: "Down",
    "-": "-",
  },

  fatigable: { Sí: "Yes", No: "No", "-": "-" },

  chips: {
    hitNormal: "HIT normal",
    directionChanging: "direction-changing nystagmus",
    skewDeviation: "skew deviation",
    focalSigns: "focal signs",
    truncalAtaxia: "truncal ataxia",
    fluctuatingHearing: "fluctuating hearing loss",
    tinnitus: "tinnitus",
    auralFullness: "aural fullness",
    fatigableNystagmus: "fatigable nystagmus",
    dixPositive: "Dix-Hallpike +",
    years: (n: number) => `${n} yrs`,
    vascularRisk: (n: number) => `vascular risk ×${n}`,
  },

  presets: {
    "bppv-benign": {
      name: "Benign BPPV (outpatient)",
      description:
        "Benign paroxysmal positional vertigo — brief episode, triggered by position, fatigable nystagmus. Outpatient management.",
    },
    "stroke-hints": {
      name: "Suspected stroke (central HINTS)",
      description:
        "Acute continuous vestibular syndrome with central HINTS signs: normal head-impulse, direction-changing nystagmus, skew deviation. High suspicion of posterior stroke.",
    },
    "meniere-episodic": {
      name: "Ménière (episodic)",
      description:
        "Ménière's disease — spontaneous episodes lasting hours with fluctuating hearing loss, tinnitus and aural fullness.",
    },
  },
};

export const STRINGS: Record<Lang, typeof es> = { es, en };

export type Dict = typeof es;

/**
 * Error state that survives a mid-session language toggle: dictionary-sourced
 * errors are stored as a KEY (re-localized on every render), while messages
 * from the server/HTTP layer are literal data and keep the language they were
 * received in — same rule as backend labels (AD-19, Decisión 2).
 */
export type UiError =
  | { key: keyof Dict["common"] }
  | { message: string };

export function uiErrorText(error: UiError, t: Dict): string {
  return "key" in error ? t.common[error.key] : error.message;
}

/** Readable chips for the case cards — an honest clinical at-a-glance view. */
export function featureChips(features: CaseFeatures, t: Dict): string[] {
  const c = t.chips;
  const chips: string[] = [];
  if (features.duration)
    chips.push(t.durationChip[features.duration as keyof Dict["durationChip"]] ?? features.duration);
  if (features.trigger)
    chips.push(t.triggerChip[features.trigger as keyof Dict["triggerChip"]] ?? features.trigger);
  if (features.head_impulse === "normal") chips.push(c.hitNormal);
  if (features.nystagmus_direction === "direction_changing") chips.push(c.directionChanging);
  if (features.skew_deviation) chips.push(c.skewDeviation);
  if (features.focal_signs?.length) chips.push(c.focalSigns);
  if (features.truncal_ataxia_severe) chips.push(c.truncalAtaxia);
  if (features.hearing_loss === "fluctuating") chips.push(c.fluctuatingHearing);
  if (features.tinnitus) chips.push(c.tinnitus);
  if (features.aural_fullness) chips.push(c.auralFullness);
  if (features.nystagmus_fatigable) chips.push(c.fatigableNystagmus);
  if (features.dix_hallpike && features.dix_hallpike !== "not_done") chips.push(c.dixPositive);
  if (features.age_years) chips.push(c.years(features.age_years));
  if (features.vascular_risk_factors?.length)
    chips.push(c.vascularRisk(features.vascular_risk_factors.length));
  return chips.slice(0, 6);
}
