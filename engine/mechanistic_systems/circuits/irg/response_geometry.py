"""Compatibility facade for workflow-neutral response-measurement geometry.

The implementation lives in
``mechanistic_systems.measurement.response_geometry``.  This facade preserves
the established import path for existing consumers; it defines no
workflow semantics.
"""

from mechanistic_systems.measurement.response_geometry import *  # noqa: F401,F403
from mechanistic_systems.measurement.response_geometry import __all__
from mechanistic_systems.measurement.response_geometry.tensors import (
    _batched_symmetric_psd,
    _symmetric_psd,
)
