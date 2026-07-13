"""`InlineGrounding` — deterministic, DB-free implementation of the
criteria RAG (reliable demo path, AD-10).

The `CORPUS` here is the **source of truth** of the grounding for the
demo and for the `rag_inline` path (when pgvector is unavailable). The
`PgvectorGrounding` implementation (in `pgvector.py`) ingests this same
corpus into the vector table.

AD-5 / hard rule 3 — CORPUS IS OUR OWN AUTHORSHIP
-------------------------------------------------
The ICVD criteria (International Classification of Vestibular Disorders,
published by Sage) are CC BY-**NC** (non-commercial). The **text** of
the criteria is restricted for commercial use; the **rules as facts**
(thresholds, booleans) are not copyrightable. Therefore, this corpus is
an **ORIGINAL structured rewrite by the Clinibrium team**, NOT a copy of
the PDF or ICVD tables. Each chunk is a paraphrase written by the team
from the concepts the superspecialist considers relevant for the agent's
reasoning. The final review of each paraphrase against the original
criterion remains a clinical task (`T-CLIN`).

NOTE: the chunk texts below are intentionally kept in Spanish — they are
RAG content fed to Claude and potentially surfaced to the clinician.

What the reasoner consumes from here is **reasoning context**,
NOT a bibliographic citation.
"""
from __future__ import annotations

from clinibrium.contracts import CaseFeatures, Diagnosis, DifferentialResult
from clinibrium.grounding.base import GroundingChunk

# ---------------------------------------------------------------------------
# CORPUS — 1-3 chunks per diagnosis (8 diagnoses). Each `source_id`
# is traceable: `clinibrium-paraphrase:<dx>-<n>`. Text: our own paraphrase.
# ---------------------------------------------------------------------------


def _chunk(
    diagnosis: Diagnosis,
    n: int,
    text: str,
) -> GroundingChunk:
    return GroundingChunk(
        text=text,
        diagnosis=diagnosis,
        source_id=f"clinibrium-paraphrase:{diagnosis.value}-{n}",
    )


# Each diagnosis has a "core" chunk (typical presentation) and, where
# applicable, additional chunks on trigger, relevant nystagmus/findings,
# and management. This gives the reasoner options to ground its
# explanation without reusing verbatim text.

_BPPV_POSTERIOR: list[GroundingChunk] = [
    _chunk(
        Diagnosis.bppv_posterior,
        1,
        "El VPPB de canal posterior se presenta como vértigo breve "
        "(típicamente menos de un minuto) desencadenado por cambios de "
        "posición de la cabeza — girarse en la cama, mirar hacia "
        "arriba, agacharse — con nistagmo torsional-vertical que aparece "
        "tras una latencia corta y se fatiga con la repetición de la "
        "maniobra. Es la causa más frecuente de vértigo periférico y la "
        "más tratable en sitio, mediante la maniobra de Epley.",
    ),
    _chunk(
        Diagnosis.bppv_posterior,
        2,
        "El signo de cabecera confirmatorio es la maniobra de Dix-Hallpike "
        "positiva del lado sintomático, con nistagmo upbeating y componente "
        "torsional hacia el oído afectado. La confirmación clínica de la "
        "componente torsional es el gold-standard para distinguir el canal "
        "posterior del horizontal. El cuadro no asocia hipoacusia ni "
        "síntomas neurovegetativos marcados.",
    ),
    _chunk(
        Diagnosis.bppv_posterior,
        3,
        "Manejo: maniobra de Epley en la consulta como primera línea, con "
        "tasas de resolución del 60-80% en una sola sesión. La maniobra "
        "de Semont es una alternativa. Si fracasan, se consideran los "
        "programas domiciliario de Brandt-Daroff. No requiere imagen ni "
        "estudio de laboratorio en el cuadro típico.",
    ),
]


_BPPV_HORIZONTAL: list[GroundingChunk] = [
    _chunk(
        Diagnosis.bppv_horizontal,
        1,
        "El VPPB de canal horizontal se sospecha cuando el vértigo "
        "posicional se acompaña de nistagmo horizontal puro desencadenado "
        "por la prueba de supine roll (Pagnini-McClure). A diferencia del "
        "canal posterior, el nistagmo puede no fatigarse tan rápidamente y "
        "puede cambiar de dirección según el lado evaluado.",
    ),
    _chunk(
        Diagnosis.bppv_horizontal,
        2,
        "La maniobra diagnóstica es el supine roll test: con el paciente "
        "acostado y la cabeza elevada 30°, se rota la cabeza a cada lado y "
        "se observa nistagmo horizontal geotrópico (hacia el suelo) o "
        "apogeotrópico (hacia el techo). El patrón geotrópico se asocia a "
        "canalitiasis y el apogeotrópico a cupulolitiasis.",
    ),
    _chunk(
        Diagnosis.bppv_horizontal,
        3,
        "Manejo: maniobra de Lempert (BBQ roll) o de Gufoni según la "
        "variante (geotrópica vs apogeotrópica). La maniobra de Epley "
        "del canal posterior NO aplica — usar la maniobra del canal "
        "correcto es crítico para la resolución.",
    ),
]


_MENIERE: list[GroundingChunk] = [
    _chunk(
        Diagnosis.meniere,
        1,
        "La enfermedad de Ménière se caracteriza por episodios espontáneos "
        "y recurrentes de vértigo rotatorio de minutos a horas, asociados "
        "a hipoacusia neurosensorial fluctuante, acúfeno y sensación de "
        "plenitud aural en el oído afectado. La tríada vértigo + "
        "hipoacusia fluctuante + acúfeno es la firma clínica.",
    ),
    _chunk(
        Diagnosis.meniere,
        2,
        "Los episodios duran típicamente entre 20 minutos y 12 horas, "
        "con un componente neurovegetativo marcado (náuseas, vómitos, "
        "inestabilidad). Entre episodios el paciente puede estar "
        "asintomático durante semanas a meses. La hipoacusia fluctúa y "
        "progresa a frecuencias bajas en las primeras etapas.",
    ),
    _chunk(
        Diagnosis.meniere,
        3,
        "Manejo en crisis: sedantes vestibulares y antieméticos. Manejo "
        "de fondo: dieta baja en sodio, diuréticos (hidroclorotiazida), "
        "y rehabilitación vestibular. En casos refractarios, inyecciones "
        "intratimpánicas de corticoide o gentamicina, y quirúrgico como "
        "última línea.",
    ),
]


_VESTIBULAR_MIGRAINE: list[GroundingChunk] = [
    _chunk(
        Diagnosis.vestibular_migraine,
        1,
        "La migraña vestibular se manifiesta como episodios de vértigo o "
        "mareo asociados a antecedentes o features migrañosos (cefalea, "
        "fotofobia, fonofobia, aura visual). Puede ocurrir con o sin "
        "cefalea concurrente — el vértigo es la manifestación predominante.",
    ),
    _chunk(
        Diagnosis.vestibular_migraine,
        2,
        "Los episodios duran de minutos a días (frecuentemente horas), y "
        "el gatillo típico incluye estrés, falta de sueño, ciclo menstrual "
        "y alimentos desencadenantes. El examen entre episodios suele ser "
        "normal, lo que la distingue de lesiones estructurales.",
    ),
    _chunk(
        Diagnosis.vestibular_migraine,
        3,
        "Manejo: identificar y evitar gatillos, profilaxis con "
        "betabloqueadores, topiramato o flunarizina según el caso, y "
        "abortivos tipo triptanos si hay cefalea. La rehabilitación "
        "vestibular puede ayudar entre episodios. El cuadro es benigno "
        "pero muy limitante en frecuencia.",
    ),
]


_VESTIBULAR_NEURITIS: list[GroundingChunk] = [
    _chunk(
        Diagnosis.vestibular_neuritis,
        1,
        "La neuritis vestibular se presenta como un síndrome vestibular "
        "agudo (AVS) espontáneo, continuo, de horas a días, con vértigo "
        "rotatorio intenso, náuseas, vómitos e inestabilidad postural. "
        "Se distingue por el test de head-impulse anormal (signo "
        "periférico) y la **ausencia de hipoacusia**.",
    ),
    _chunk(
        Diagnosis.vestibular_neuritis,
        2,
        "El nistagmo es típicamente horizontal unidireccional, "
        "suprimible con la fijación visual, y sigue la ley de Alexander "
        "(hacia el lado sano). En el HINTS, head-impulse anormal + "
        "nistagmo que suprime con fijación + sin skew deviation = patrón "
        "periférico.",
    ),
    _chunk(
        Diagnosis.vestibular_neuritis,
        3,
        "Manejo agudo: corticoides orales en las primeras 72 horas "
        "(evidencia controvertida), sedantes vestibulares cortos, y "
        "rehabilitación vestibular temprana (clave para la recuperación "
        "funcional). La compensación central completa puede tomar semanas.",
    ),
]


_LABYRINTHITIS: list[GroundingChunk] = [
    _chunk(
        Diagnosis.labyrinthitis,
        1,
        "La laberintitis combina un AVS continuo con **hipoacusia** "
        "neurosensorial del mismo oído. Se diferencia de la neuritis "
        "vestibular por la presencia de compromiso auditivo, lo que "
        "sugiere inflamación del laberinto completo (no solo del nervio).",
    ),
    _chunk(
        Diagnosis.labyrinthitis,
        2,
        "La hipoacusia suele ser unilateral, aguda y puede ser parcial o "
        "completa. El vértigo y la hipoacusia pueden instalarse en forma "
        "simultánea o secuencial. **No debe confundirse con un AICA**: "
        "ante hipoacusia súbita, siempre evaluar el escenario vascular "
        "antes de asumir causa inflamatoria.",
    ),
    _chunk(
        Diagnosis.labyrinthitis,
        3,
        "Manejo: corticoides orales, sedantes vestibulares cortos, "
        "rehabilitación vestibular. La hipoacusia puede recuperarse "
        "parcial o totalmente. Si la hipoacusia no se recupera, evaluar "
        "audífono o implante según el grado de pérdida residual.",
    ),
]


_CENTRAL_SUSPECTED: list[GroundingChunk] = [
    _chunk(
        Diagnosis.central_suspected,
        1,
        "El vértigo de causa central (stroke cerebeloso o de tronco "
        "encefálico, típicamente AICA/PICA) puede mimetizar un cuadro "
        "periférico. Las banderas rojas para sospecharlo son: AVS con "
        "head-impulse **normal** (en periferia sería anormal), nistagmo "
        "**vertical puro** o de dirección cambiante con la mirada, "
        "skew deviation, signos focales asociados, y ataxia troncal severa.",
    ),
    _chunk(
        Diagnosis.central_suspected,
        2,
        "El trío HINTS (Head-Impulse, Nystagmus, Test of Skew) en AVS "
        "tiene mayor sensibilidad para stroke que la RMN en las primeras "
        "24-48 horas. Un patrón periférico (head-impulse anormal + "
        "nistagmo suprimible + sin skew) hace menos probable el stroke; "
        "un patrón central (head-impulse normal + nistagmo que NO "
        "suprime + skew) lo hace más probable.",
    ),
    _chunk(
        Diagnosis.central_suspected,
        3,
        "Manejo: **derivación urgente** a centro con neuroimagen y "
        "stroke unit. No aplicar maniobra de Epley ni sedantes "
        "vestibulares que retrasen la evaluación. Tiempo-dependiente: "
        "cada minuto cuenta para tratamiento reperfusor si aplica.",
    ),
]


_CARDIOGENIC_SUSPECTED: list[GroundingChunk] = [
    _chunk(
        Diagnosis.cardiogenic_suspected,
        1,
        "El mareo de origen cardiogénico se sospecha cuando el vértigo o "
        "presíncope se asocia a palpitaciones, dolor torácico, disnea, o "
        "factores de riesgo cardiovascular (arritmia, valvulopatía, "
        "hipotensión). La presentación puede ser continua o "
        "sincopal/transitoria.",
    ),
    _chunk(
        Diagnosis.cardiogenic_suspected,
        2,
        "El examen neurológico suele ser normal entre eventos, sin "
        "nistagmo espontáneo ni déficit vestibular. El electrocardiograma "
        "y el monitoreo Holter son clave para el diagnóstico; en "
        "presentación aguda, evaluación cardiovascular con troponinas.",
    ),
    _chunk(
        Diagnosis.cardiogenic_suspected,
        3,
        "Manejo: escalamiento a cardiología, estudio electrofisiológico "
        "si hay sospecha de arritmia, corrección de factores de riesgo. "
        "No tratar como vértigo vestibular primario — el riesgo es "
        "sincope arrítmico y muerte súbita.",
    ),
]


CORPUS: dict[Diagnosis, list[GroundingChunk]] = {
    Diagnosis.bppv_posterior: _BPPV_POSTERIOR,
    Diagnosis.bppv_horizontal: _BPPV_HORIZONTAL,
    Diagnosis.meniere: _MENIERE,
    Diagnosis.vestibular_migraine: _VESTIBULAR_MIGRAINE,
    Diagnosis.vestibular_neuritis: _VESTIBULAR_NEURITIS,
    Diagnosis.labyrinthitis: _LABYRINTHITIS,
    Diagnosis.central_suspected: _CENTRAL_SUSPECTED,
    Diagnosis.cardiogenic_suspected: _CARDIOGENIC_SUSPECTED,
}


# We cover exactly the 8 diagnoses from the task spec
# (excluding `undetermined`, the pool's fallback when nothing matches).
SUPPORTED_DIAGNOSES: frozenset[Diagnosis] = frozenset(CORPUS.keys())


class InlineGrounding:
    """Deterministic, DB-free `Grounding` — the reliable demo path.

    Strategy:
      - Takes the diagnoses from `DifferentialResult.candidates` (already
        sorted desc by score by the `DifferentialEngine`).
      - For each top-N candidate (up to `k` chunks in total), emits the
        associated `CORPUS` chunks in corpus order.
      - Truncates to `k` chunks in total.

    Deterministic: same (candidates, features, k) ⇒ same list of chunks,
    in the same order. No randomness, no I/O, no network, no LLM.
    """

    def retrieve(
        self,
        candidates: DifferentialResult,
        features: CaseFeatures,  # noqa: ARG002 — Protocol signature
        k: int = 4,
    ) -> list[GroundingChunk]:
        out: list[GroundingChunk] = []
        for cand in candidates.candidates:
            chunks_for_dx = CORPUS.get(cand.diagnosis, [])
            for chunk in chunks_for_dx:
                if len(out) >= k:
                    return out
                out.append(chunk)
            if len(out) >= k:
                return out
        return out
