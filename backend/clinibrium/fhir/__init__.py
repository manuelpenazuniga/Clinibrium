"""Mapea resultados a artefactos FHIR (formato de salida, AD-9).

Función pura: ``to_bundle(result, features, audit)`` → ``dict`` (Bundle
FHIR R4 tipo ``collection``). Sin I/O, sin red. Hoja del grafo: solo
importa ``contracts`` (regla dura del mapa de Clinibrium).

``bundle_sha256(bundle)`` → SHA-256 del JSON canónico del bundle,
para integridad tamper-evident.
"""
from __future__ import annotations

from clinibrium.fhir.mapping import bundle_sha256, to_bundle

__all__ = ["bundle_sha256", "to_bundle"]
