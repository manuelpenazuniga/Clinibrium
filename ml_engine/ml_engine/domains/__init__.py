"""Domain configurations (data) that instantiate the ML platform.

Each domain is written as a ``Domain`` (FeatureSpec + LabelHierarchy +
SyntheticSpec) WITHOUT touching ``ml_engine.core``. Vertigo = instance #1;
``toy`` = agnosticism proof (INV-12).
"""
