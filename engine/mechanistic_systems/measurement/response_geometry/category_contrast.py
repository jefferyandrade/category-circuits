"""Compatibility facade for category-contrast response geometry.

The implementation owner is :mod:`mechanistic_systems.geometry.category_contrast`.
This module preserves the established measurement submodule coordinate and exact
public object identities without taking ownership or widening authority.
"""

from mechanistic_systems.geometry.category_contrast import (
    CategoryContrastJacobian,
    CategoryContrastResponse,
    CategoryContrastResponseSpec,
    ExactMaskResponseComparison,
    ExactMaskResponseReceipt,
    build_exact_mask_response_receipt,
    category_contrast_jacobian,
    category_contrast_response,
    compare_exact_mask_logits,
)

__all__ = [
    "CategoryContrastJacobian",
    "CategoryContrastResponse",
    "CategoryContrastResponseSpec",
    "ExactMaskResponseComparison",
    "ExactMaskResponseReceipt",
    "build_exact_mask_response_receipt",
    "category_contrast_jacobian",
    "category_contrast_response",
    "compare_exact_mask_logits",
]
