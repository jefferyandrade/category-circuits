"""
Skeleton package for future Hessian-related utilities.

The concrete implementations will live in:
    * hvp.py   – Gauss–Legendre 2-point Hessian-vector products.
    * lift.py  – Helpers to lift activation-space HVPs into weight space.
    * lowrank.py – Low-rank sketches / power iterations for Hessian factors.
"""

from . import hvp, lift, lowrank  # noqa: F401 (imported for side effects / namespace)

__all__ = ["hvp", "lift", "lowrank"]

