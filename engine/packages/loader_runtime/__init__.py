"""Shared loader / throughput runtime.

Concept-free data-loading primitives (pinned prefetch, background iteration,
device-resident banks, uint8 decode params, copy streams, execution-ordered
bundles, saturating-batch measurement) that the mechanistic_systems measurement
and circuits concept layers draw down onto. Depends only on torch/numpy/stdlib
(+ deferred torchvision); nothing above.
"""
