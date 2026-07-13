"""Clinibrium Track B — ML engine (domain-agnostic confidence layer).

EXPERIMENTAL · SYNTHETIC data · no clinical validity.

Isolated from Track A by design (AD-15/AD-16): this package has its own venv
and its own ML deps; the ``clinibrium`` core NEVER imports ``ml_engine`` nor
installs its dependencies (INV-6). A↔B communication happens only through the
frozen HTTP contract (``ML_PREDICT_URL`` → ``POST /predict``).
"""

__version__ = "0.1.0"
