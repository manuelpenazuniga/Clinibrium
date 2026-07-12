"""What Would Change My Mind? — análisis contrafactual determinista.

Explicabilidad contrafactual clínica: ¿qué ÚNICO hallazgo cambiaría el manejo de
este paciente? El LLM NO decide — el núcleo determinista (RedFlagEngine + rails)
verifica cada contrafactual y Claude (opcional) solo lo explica (INV-3).
"""
from clinibrium.counterfactual.engine import (
    Counterfactual,
    WhatWouldChangeResult,
    analyze,
)

__all__ = ["Counterfactual", "WhatWouldChangeResult", "analyze"]
