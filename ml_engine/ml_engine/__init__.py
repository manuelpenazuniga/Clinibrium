"""Clinibrium Track B — motor ML (capa de confianza agnóstica de dominio).

EXPERIMENTAL · datos SINTÉTICOS · sin validez clínica.

Aislado de Track A por diseño (AD-15/AD-16): este paquete tiene su propio
venv y sus deps ML; el core ``clinibrium`` NUNCA importa ``ml_engine`` ni
instala sus dependencias (INV-6). La comunicación A↔B es solo por el
contrato HTTP congelado (``ML_PREDICT_URL`` → ``POST /predict``).
"""

__version__ = "0.1.0"
