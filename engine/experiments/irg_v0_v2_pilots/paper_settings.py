"""The settings the paper's circuits were extracted with, used by the entry points as defaults.

Each circuit's receipt in the circuit library records these values. ``apply`` fills in every setting the
environment does not already give, so a value set explicitly still takes precedence. Speed switches are not
listed: they do not change the circuits, and keep the engine's own defaults.
"""
from __future__ import annotations

from collections.abc import MutableMapping

# The same for every network and pruning method.
SHARED = {
    "IRG_ANALYTIC_SIGN_PARTITION": "1",
    "IRG_ANALYTIC_SIGN_SCOPE": "all_supported_routes",
    "IRG_CALIBRATION_PENALTY": "firth",
    "IRG_CENSUS_DECISION_REFERENCED": "1",
    "IRG_CENSUS_EPSILON": "0.005",
    "IRG_CENSUS_MODE": "per_category",
    "IRG_CENSUS_PRECISION_EPS": "0",
    "IRG_CENSUS_SAMPLE_DESIGN": "importance_batches_v1",
    "IRG_CONSEQUENCE_PRICE_SHORT_CIRCUIT": "1",
    # The walk prunes layers deep to shallow (AlexNet: fc7 first, conv1 last): "forward" follows the profile's
    # layer list, which is ordered from the output side.
    "IRG_LAYER_EXECUTION_ORDER": "forward",
    "IRG_LOCAL_COVARIANCE_SKETCH_RANK": "0",
    "IRG_LOSS_WEIGHTING": "balanced",
    "IRG_READOUT_GUARD_MODE": "full",
    "IRG_RESPONSE_STABLE_LOOKS": "1",
    "IRG_SINGLE_PAIR_WALK": "1",
    "IRG_STEP_MATRIX_LANES": "production_v1",
    "IRG_SUBSTITUTED_LOSS_WEIGHTING": "per_image",
    "IRG_TERMINAL_CALIBRATION_RULE": "walk_resolution",
    "IRG_UINT8_FOLDED_STEM": "1",
    "IRG_V6_RUNTIME_ITEMS": "1",
}

# Per network: the loss threshold tau and the census batch size. AlexNet's threshold is also the one the
# all-layers comparison of pruning methods runs at.
PER_NETWORK = {
    "torchvision.alexnet.conv_kernel_weight.imagenet1k_v1": {
        "IRG_SENSITIVITY_TAU": "1.5e-07",
        "IRG_REGISTERED_SENSITIVITY_TAU": "1.5e-07",
        "IRG_CENSUS_BATCH_SIZE": "1024",
        "IRG_CENSUS_CAPTURE_ROWS": "1024",
    },
    "torchvision.vgg11.conv_kernel_weight.imagenet1k_v1": {
        "IRG_SENSITIVITY_TAU": "1.768e-07",
        "IRG_CENSUS_BATCH_SIZE": "512",
        "IRG_CENSUS_CAPTURE_ROWS": "512",
    },
    "torchvision.resnet18.conv_kernel_weight.imagenet1k_v1": {
        "IRG_SENSITIVITY_TAU": "5e-07",
        "IRG_CENSUS_BATCH_SIZE": "512",
        "IRG_CENSUS_CAPTURE_ROWS": "512",
    },
    "torchvision.resnet50.conv_kernel_weight.imagenet1k_v2": {
        "IRG_SENSITIVITY_TAU": "4.033e-07",
        "IRG_CENSUS_BATCH_SIZE": "256",
        "IRG_CENSUS_CAPTURE_ROWS": "256",
    },
    "torchvision.resnet50.conv_kernel_weight.imagenet1k_v1": {
        "IRG_SENSITIVITY_TAU": "4.033e-07",
        "IRG_CENSUS_BATCH_SIZE": "256",
        "IRG_CENSUS_CAPTURE_ROWS": "256",
    },
}


def apply(environment: MutableMapping[str, str], model_id: str | None = None) -> list[str]:
    """Give every setting ``environment`` leaves unset the paper's value; return the names filled in."""

    settings = dict(SHARED)
    if model_id is not None:
        settings.update(PER_NETWORK.get(str(model_id), {}))
    filled = [name for name in settings if name not in environment]
    for name in filled:
        environment[name] = settings[name]
    return filled
