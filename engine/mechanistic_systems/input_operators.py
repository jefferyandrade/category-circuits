"""Canonical pixel-valued operators shared by measurement and identification."""

from __future__ import annotations

from typing import Any


CANONICAL_IMAGENET_INPUT_OPERATOR_V1 = (
    "imagenet_pil_rgb_bilinear_resize256_center_crop224_uint8_v1"
)


def canonical_imagenet_input_operator_contract() -> dict[str, Any]:
    """Return the complete scientific identity of the ImageNet v1 operator."""

    return {
        "schema_name": "mechanistic_image_preprocessing_v1",
        "preprocessing_id": CANONICAL_IMAGENET_INPUT_OPERATOR_V1,
        "resize_size": [256],
        "center_crop_size": [224, 224],
        "interpolation": "bilinear",
        "antialias": True,
        "scale_uint8_to_unit": True,
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
        "output_dtype": "float32",
        "notes": {
            "physical_decoder": "pil_rgb",
            "spatial_output": "round_clamp_uint8_v1",
            "online_decode": "uint8_to_float32_imagenet_normalize",
            "scientific_scope": (
                "shared_measurement_identification_execution_operator"
            ),
        },
    }


def build_canonical_imagenet_input_transform() -> Any:
    """Build the executable PIL transform represented by the frozen contract."""

    from torchvision import transforms

    return transforms.Compose(
        [
            transforms.Resize(
                256,
                interpolation=transforms.InterpolationMode.BILINEAR,
                antialias=True,
            ),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        ]
    )


__all__ = [
    "CANONICAL_IMAGENET_INPUT_OPERATOR_V1",
    "build_canonical_imagenet_input_transform",
    "canonical_imagenet_input_operator_contract",
]
