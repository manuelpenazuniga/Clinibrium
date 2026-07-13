"""Maps results to FHIR artifacts (output format, AD-9).

Pure function: ``to_bundle(result, features, audit)`` → ``dict`` (FHIR R4
Bundle of type ``collection``). No I/O, no network. Leaf of the graph:
only imports ``contracts`` (hard rule of the Clinibrium module map).

``bundle_sha256(bundle)`` → SHA-256 of the bundle's canonical JSON,
for tamper-evident integrity.
"""
from __future__ import annotations

from clinibrium.fhir.mapping import bundle_sha256, to_bundle

__all__ = ["bundle_sha256", "to_bundle"]
