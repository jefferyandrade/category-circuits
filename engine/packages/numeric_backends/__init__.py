"""Shared optimized numeric solver core.

Concept-free numeric primitives (projected-Newton calibration solve, fused
trust-region step) that BOTH the mechanistic_systems backends and the thermobayes
engine draw down onto. Depends only on torch/numpy + contracts; nothing above.
"""
