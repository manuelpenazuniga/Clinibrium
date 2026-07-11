# DATA CARD — dataset sintético de `ml_engine` (Track B)

> **EXPERIMENTAL · 100% SINTÉTICO · SIN pacientes reales · SIN validez clínica.**

## Qué es
Dataset generado por `ml_engine.core.synth.generate` a partir de priors por
diagnóstico (`ml_engine.domains.vertigo.SYNTHETIC`). Cada fila = un caso
sintético con features desidentificadas + un `label` (uno de los 8 diagnósticos
entrenables). **No proviene de ningún paciente, historia clínica ni registro
real.** No cruza PII de ningún tipo (es data fabricada por muestreo).

## Cómo se genera
- **Muestreo condicional por label:** para cada diagnóstico, distribuciones
  provisionales de features derivadas de los criterios ICVD (categóricas por
  tabla de probabilidad, booleanas por prevalencia, numéricas por normal
  truncada). Features no especificadas por un label → default neutral.
- **Prevalencia:** por label (provisional, documentada en `vertigo.SYNTHETIC`),
  no uniforme.
- **Tamaño / seed:** `n_samples=8000`, `seed=20260711` (fijo). Splits aguas
  abajo: train/calibración/test (60/20/20), todo sintético.
- **Reproducibilidad:** mismo seed ⇒ mismo DataFrame **bajo el entorno
  bloqueado** (`requirements.lock`) y con tolerancia numérica; NO se promete
  identidad byte-a-byte entre plataformas/threads.

## Qué NO es (límites de honestidad)
- **No es evidencia clínica.** Las métricas que se calculen sobre estos datos
  (F1, AUC, ECE, calibración) miden **recuperación del proceso generativo**
  sintético — cuán bien el modelo reconstruye las reglas con que se fabricaron
  los datos —, **no** desempeño diagnóstico en pacientes.
- **Los priors son PROVISIONALES.** Prevalencias, distribuciones y umbrales los
  debe revisar/firmar el especialista (T-CLIN). Hasta entonces: "supuestos del
  prototipo, pendientes de revisión clínica".
- **Riesgo de circularidad conocido:** el label genera las features y el modelo
  las reaprende; métricas altas NO implican utilidad clínica. Por eso se
  reportan como recuperación del generador y se acompañan de stress sets.

## Uso previsto
Demostrar el **mecanismo** de la capa de confianza (jerarquía, monotonía de
peligro, calibración, abstención, SHAP) de forma reproducible y honesta. No
para inferencia clínica.
