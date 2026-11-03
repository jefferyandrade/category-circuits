"""Structural views over canonical complete hard-mask interventions.

This module does not construct masks or authorize commits.  It defines the
minimal read-only capability that measurement providers consume after a
workflow owner has resolved a complete physical mask.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class HardMaskView(Protocol):
    """Read-only complete-mask surface required by measurement adapters."""

    route_count: int
    mask_hash: str

    def to_bool_array(self) -> np.ndarray:
        """Return the complete Boolean gate vector on the declared route axis."""


__all__ = ["HardMaskView"]
