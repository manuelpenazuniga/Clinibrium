"""Reasoner engine: calls Claude, assembles ReasonerOutput.

NOTE: the system/user prompts below are intentionally in Spanish — they
instruct Claude to produce clinician-facing output in Spanish. Do not
translate them.
"""
from __future__ import annotations

import asyncio
import logging
from textwrap import dedent
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel

from clinibrium.contracts import (
    CaseFeatures,
    DifferentialResult,
    PredictResponse,
    ReasonerOutput,
    RedFlagResult,
    Urgency,
)
from clinibrium.grounding import GroundingChunk
from clinibrium.reasoner.pick_model import OPUS, pick_model
from clinibrium.reasoner.privacy import build_network_payload

if TYPE_CHECKING:
    from anthropic import AsyncAnthropic

logger = logging.getLogger(__name__)

MAX_RETRIES = 2
RETRY_BACKOFF_S = (1.0, 3.0)


class _LLMReasoning(BaseModel):
    """Structured LLM output. Explanation and reconciliation only."""

    explanation: str
    reconciliation: str
    suggested_next_steps: list[str] = []
    reasoner_suggested_urgency: Urgency | None = None


# Output-language directive appended ONLY for lang == "en". The Spanish
# (default) prompt is left byte-identical to the recorded-demo version; the
# whole prompt is written in Spanish, which implicitly yields Spanish output.
_EN_OUTPUT_DIRECTIVE = (
    "6. OUTPUT LANGUAGE: write your entire response (explanation, reconciliation,\n"
    "   suggested_next_steps) in ENGLISH. The ICVD criteria chunks are provided in\n"
    "   Spanish (source paraphrase); ground your reasoning in them and cite their\n"
    "   chunk IDs verbatim, but do NOT translate the IDs."
)


def _build_system_prompt(lang: str = "es") -> str:
    base = dedent("""\
        Eres un asistente clínico otoneurológico (VertigoDx). Tu función es EXPLICAR
        y CONCILIAR los hallazgos de los motores deterministas (RedFlagEngine,
        DifferentialEngine) con el contexto clínico estructurado provisto. NO eres un
        clasificador diagnóstico y NO fijas la urgencia vinculante del caso.

        REGLAS DURAS:
        1. Solo SUGIERES urgencia (campo reasoner_suggested_urgency). La urgencia
           vinculante la sellan los rieles (hard invariants) después de ti, y tu
           sugerencia puede ser sobrescrita.
        2. TU explicación (explanation) debe fundamentarse en los chunks de criterios
           ICVD (paráfrasis propia) que se te proveen. Citá los IDs de los chunks.
        3. TU conciliación (reconciliation) debe explicar cómo se alinean (o no) los
           hallazgos deterministas con los chunks de grounding y, si está disponible,
           la predicción ML.
        4. NO inventes diagnósticos, criterios ni hallazgos. Si la evidencia es
           insuficiente, indicalo.
        5. NO incluyas PII ni texto libre del paciente en tu respuesta.""")
    if lang == "en":
        return base + "\n" + _EN_OUTPUT_DIRECTIVE
    return base


def _build_user_prompt(
    safe_payload: dict,
    grounding_chunks: list[GroundingChunk],
    differential: DifferentialResult,
    ml: PredictResponse | None,
) -> str:
    chunks_text = "\n\n".join(
        f"[{c.source_id}] ({c.diagnosis.value if c.diagnosis else 'general'}): {c.text}"
        for c in grounding_chunks
    ) or "(sin chunks de grounding disponibles)"

    differential_text = "\n".join(
        f"  - {c.diagnosis.value}: {c.score:.3f}" for c in differential.candidates[:5]
    ) or "(sin candidatos diferenciales)"

    ml_text = ""
    if ml is not None:
        ml_text = (
            "\n\nPredicción ML (CatBoost, track B opcional):\n"
            + "\n".join(f"  - {dx}: {prob:.3f}" for dx, prob in ml.probabilities.items())
            + f"\n  model_version: {ml.model_version}"
        )

    features_text = "\n".join(
        f"  {k}: {v}" for k, v in safe_payload.items() if v is not None and v != [] and v != {}
    )

    return dedent(f"""\
        CASO CLÍNICO (features estructuradas desidentificadas):
        {features_text}

        RESULTADO DIFERENCIAL (DifferentialEngine, ordenado por score):
        {differential_text}
        {ml_text}

        CHUNKS DE GROUNDING (criterios ICVD en paráfrasis propia):
        {chunks_text}

        Con base en la evidencia provista, emití tu razonamiento estructurado.""")


async def reason(
    features: CaseFeatures,
    red_flag: RedFlagResult,
    differential: DifferentialResult,
    ml: PredictResponse | None,
    grounding_chunks: list[GroundingChunk],
    *,
    recording_mode: bool = False,
    lang: str = "es",
    client: AsyncAnthropic | None = None,  # pyright: ignore[reportInvalidTypeForm]
    timeout_s: float = 60.0,
) -> ReasonerOutput | None:
    model = pick_model(red_flag, recording_mode=recording_mode)

    safe = build_network_payload(features)

    system_prompt = _build_system_prompt(lang)
    user_prompt = _build_user_prompt(safe, grounding_chunks, differential, ml)

    messages = [{"role": "user", "content": user_prompt}]

    from anthropic import APIConnectionError, APIStatusError, AsyncAnthropic, RateLimitError

    if client is None:
        from clinibrium.config import get_settings

        api_key = get_settings().ANTHROPIC_API_KEY
        client = AsyncAnthropic(api_key=api_key)

    # `thinking` only applies to Opus 4.8 (adaptive). For Haiku 4.5 the
    # parameter is OMITTED entirely — passing an explicit `thinking=None`
    # is rejected by the API with 400 ("thinking: Input should be an object").
    extra_kwargs: dict[str, Any] = {}
    if model == OPUS:
        extra_kwargs["thinking"] = {"type": "adaptive"}

    for attempt in range(1 + MAX_RETRIES):
        try:
            async with asyncio.timeout(timeout_s):
                resp = await client.messages.parse(
                    model=model,
                    # max_tokens caps thinking + text combined; Opus 4.8 in
                    # adaptive-thinking mode needs headroom or the structured
                    # JSON truncates mid-string (ValidationError → degrade).
                    max_tokens=16000,
                    messages=messages,  # type: ignore[arg-type]
                    system=system_prompt,
                    output_format=_LLMReasoning,
                    **extra_kwargs,
                )
            parsed = resp.parsed_output
            if parsed is None:
                raise RuntimeError("Claude returned a response without parsed_output")
            return ReasonerOutput(
                explanation=parsed.explanation,
                reconciliation=parsed.reconciliation,
                suggested_next_steps=parsed.suggested_next_steps,
                model_used=model,
                reasoner_suggested_urgency=parsed.reasoner_suggested_urgency,
                grounding_refs=[c.source_id for c in grounding_chunks],
            )
        except RateLimitError:
            if attempt < MAX_RETRIES:
                delay = RETRY_BACKOFF_S[attempt]
                logger.warning(
                    "Reasoner rate limited (attempt %d/%d), retrying in %.1fs",
                    attempt + 1,
                    1 + MAX_RETRIES,
                    delay,
                )
                await asyncio.sleep(delay)
            else:
                logger.exception("Reasoner rate limited after %d retries", MAX_RETRIES)
        except (APIConnectionError, APIStatusError, asyncio.TimeoutError) as e:
            # Retryable: connection, timeout, or server 5xx. A client 4xx
            # (400 bad request, etc.) does NOT improve on retry → break.
            retryable = isinstance(e, (APIConnectionError, asyncio.TimeoutError)) or (
                isinstance(e, APIStatusError) and e.status_code >= 500
            )
            if retryable and attempt < MAX_RETRIES:
                delay = RETRY_BACKOFF_S[attempt]
                logger.warning(
                    "Reasoner retryable error %s (attempt %d/%d), retrying in %.1fs",
                    type(e).__name__,
                    attempt + 1,
                    1 + MAX_RETRIES,
                    delay,
                )
                await asyncio.sleep(delay)
            else:
                logger.exception("Reasoner failed: %s", type(e).__name__)
                break  # non-retryable (4xx) or retries exhausted → degrade now
        except Exception:
            logger.exception("Reasoner unexpected error")
            break

    logger.info("Reasoner degraded (INV-8): returning None; pipeline continues.")
    return None
