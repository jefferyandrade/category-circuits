"""Shared response-version interfaces without executable numbered workflows.

The numbered v2--v4 workflow modules are deliberately not imported here.
Their reusable response-geometry capabilities live under
``mechanistic_systems.measurement.response_geometry`` and their exact workflow
carriers are handled only by the supersession archive protocol.  Importing
this package therefore cannot make a reserved workflow executable as an
incidental side effect.
"""

from .base import GeneratedCandidateSet, ResponseVersion, VersionState

__all__ = [
    "GeneratedCandidateSet",
    "ResponseVersion",
    "VersionState",
]
