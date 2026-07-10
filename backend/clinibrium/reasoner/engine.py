"""Motor del reasoner: llama a Claude, assembla ReasonerOutput."""
from __future__ import annotations

import asyncio
import logging
from textwrap import dedent
from typing import TYPE_CHECKING

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
    """Salida estructurada del LLM. Solo explicación y conciliación."""

    explanation: str
    reconciliation: str
    suggested_next_steps: list[str] = []
    reasoner_suggested_urgency: Urgency | None = None


def _build_system_prompt() -> str:
    return dedent("""\
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
    client: AsyncAnthropic | None = None,  # pyright: ignore[reportInvalidTypeForm]
    timeout_s: float = 20.0,
) -> ReasonerOutput | None:
    model = pick_model(red_flag, recording_mode=recording_mode)

    safe = build_network_payload(features)

    system_prompt = _build_system_prompt()
    user_prompt = _build_user_prompt(safe, grounding_chunks, differential, ml)

    messages = [{"role": "user", "content": user_prompt}]

    from anthropic import APIConnectionError, APIStatusError, AsyncAnthropic, RateLimitError

    if client is None:
        from clinibrium.config import get_settings

        api_key = get_settings().ANTHROPIC_API_KEY
        client = AsyncAnthropic(api_key=api_key)

    thinking: dict | None = {"type": "adaptive"} if model == OPUS else None

    for attempt in range(1 + MAX_RETRIES):
        try:
            async with asyncio.timeout(timeout_s):
                resp = await client.messages.parse(
                    model=model,
                    max_tokens=1200,
                    thinking=thinking,  # type: ignore[arg-type]
                    messages=messages,  # type: ignore[arg-type]
                    system=system_prompt,
                    output_format=_LLMReasoning,
                )
            parsed = resp.parsed_output
            if parsed is None:
                raise RuntimeError("Claude devolvió respuesta sin parsed_output")
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
            is_server_error = isinstance(e, APIStatusError) and e.status_code >= 500
            if is_server_error and attempt < MAX_RETRIES:
                delay = RETRY_BACKOFF_S[attempt]
                logger.warning(
                    "Reasoner server error %s (attempt %d/%d), retrying in %.1fs",
                    e.status_code if isinstance(e, APIStatusError) else "connection",
                    attempt + 1,
                    1 + MAX_RETRIES,
                    delay,
                )
                await asyncio.sleep(delay)
            elif isinstance(e, asyncio.TimeoutError) and attempt < MAX_RETRIES:
                delay = RETRY_BACKOFF_S[attempt]
                logger.warning(
                    "Reasoner timeout (attempt %d/%d), retrying in %.1fs",
                    attempt + 1,
                    1 + MAX_RETRIES,
                    delay,
                )
                await asyncio.sleep(delay)
            else:
                logger.exception("Reasoner failed: %s", type(e).__name__)
        except Exception:
            logger.exception("Reasoner unexpected error")
            break

    logger.info("Reasoner degraded (INV-8): returning None; pipeline continues.")
    return None
