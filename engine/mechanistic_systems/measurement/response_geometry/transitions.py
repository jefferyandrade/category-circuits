"""Workflow-neutral finite-transition response measurement capability.

The capability augments an :class:`IntrinsicResponseField` with measured
directional second response, finite-dose summaries, and exact hard-mask
transition receipts.  It remains a proposal and model-criticism surface: an
exact receipt can replace the local prediction for its *same* canonical
physical action, but this object does not select a workflow or authorize a
pruning commit.

``capability_hash`` is the live workflow-neutral identity.  The explicit
``legacy_v4_*`` helpers reproduce historical state identities only so archived
receipts remain verifiable; they carry no current workflow authority.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from functools import cached_property
import math
from typing import Any

import numpy as np

from mechanistic_systems.contracts.content_address import content_hash

from .contracts import readonly_vector
from .intrinsic import IntrinsicResponseField
from .tensors import ResponseActionMoments, ResponseTransitionReceipt


def readonly_optional_matrix(
    value: np.ndarray | None,
    *,
    name: str,
) -> np.ndarray | None:
    """Return a finite, C-contiguous, read-only matrix when present."""

    if value is None:
        return None
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 2 or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a finite matrix")
    array = np.array(array, copy=True, order="C")
    array.setflags(write=False)
    return array


def readonly_optional_tensor3(
    value: np.ndarray | None,
    *,
    name: str,
) -> np.ndarray | None:
    """Return a finite, C-contiguous, read-only rank-three tensor when present."""

    if value is None:
        return None
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 3 or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a finite rank-three tensor")
    array = np.array(array, copy=True, order="C")
    array.setflags(write=False)
    return array


def readonly_optional_psd(
    value: np.ndarray | None,
    *,
    name: str,
    tolerance: float = 1e-7,
) -> np.ndarray | None:
    """Return a read-only PSD matrix, repairing roundoff-scale negativity only."""

    if value is None:
        return None
    array = np.asarray(value, dtype=np.float64)
    if (
        array.ndim != 2
        or array.shape[0] != array.shape[1]
        or not np.isfinite(array).all()
    ):
        raise ValueError(f"{name} must be a finite square matrix")
    array = 0.5 * (array + array.T)
    eigenvalues = np.linalg.eigvalsh(array)
    scale = max(1.0, float(np.max(np.abs(eigenvalues)))) if eigenvalues.size else 1.0
    minimum = float(np.min(eigenvalues)) if eigenvalues.size else 0.0
    if minimum < -float(tolerance) * scale:
        raise ValueError(f"{name} must be positive semidefinite")
    if minimum < 0.0:
        array = array + (-minimum + np.finfo(np.float64).eps * scale) * np.eye(
            array.shape[0]
        )
    array = np.array(array, copy=True, order="C")
    array.setflags(write=False)
    return array


def positive_semidefinite_part(matrix: np.ndarray) -> np.ndarray:
    """Return the positive-spectral part of one finite symmetric matrix."""

    value = np.asarray(matrix, dtype=np.float64)
    if (
        value.ndim != 2
        or value.shape[0] != value.shape[1]
        or not np.isfinite(value).all()
    ):
        raise ValueError("PSD-part input must be a finite square matrix")
    symmetric = 0.5 * (value + value.T)
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    result = (eigenvectors * np.maximum(eigenvalues, 0.0)) @ eigenvectors.T
    result = np.array(0.5 * (result + result.T), copy=True, order="C")
    result.setflags(write=False)
    return result


def psd_common_upper_envelope(
    left: np.ndarray | None,
    right: np.ndarray,
) -> np.ndarray:
    """Construct a PSD matrix dominating both inputs in Loewner order.

    This construction is a conservative common upper envelope.  It makes no
    claim to be a least upper bound: the Loewner order on PSD matrices is not a
    lattice in general.
    """

    candidate = readonly_optional_psd(
        right,
        name="response residual model-error candidate",
    )
    assert candidate is not None
    if left is None:
        return candidate
    base = readonly_optional_psd(
        left,
        name="response residual model-error envelope",
    )
    assert base is not None
    if base.shape != candidate.shape:
        raise ValueError("PSD common-upper-envelope inputs use different axes")
    joined = base + positive_semidefinite_part(candidate - base)
    result = readonly_optional_psd(
        0.5 * (joined + joined.T),
        name="response residual model-error common upper envelope",
    )
    assert result is not None
    return result


def uniform_sample_mean_variance_from_mean_rms(
    mean: np.ndarray,
    rms: np.ndarray,
    sample_count: int,
) -> np.ndarray:
    """Recover sample-mean variance for one fixed uniformly weighted panel.

    ``mean`` and ``rms`` must summarize the same finite panel with equal
    weights, where ``rms`` is ``sqrt(mean(x**2))``.  The result
    ``(rms**2 - mean**2)/(n - 1)`` is the usual finite-panel estimate of the
    variance of that uniform sample mean.  It is not, by itself, a
    superpopulation-generalization or adaptively sampled-panel guarantee.
    """

    count = int(sample_count)
    if count <= 1:
        raise ValueError(
            "uniform sample-mean variance requires at least two panel members"
        )
    mean_array = np.asarray(mean, dtype=np.float64)
    rms_array = np.asarray(rms, dtype=np.float64)
    if (
        mean_array.shape != rms_array.shape
        or not np.isfinite(mean_array).all()
        or not np.isfinite(rms_array).all()
        or np.any(rms_array < 0.0)
    ):
        raise ValueError("uniform panel mean and RMS must be finite and aligned")
    numerator = rms_array * rms_array - mean_array * mean_array
    scale = np.maximum(1.0, np.maximum(rms_array * rms_array, mean_array * mean_array))
    if np.any(numerator < -1e-10 * scale):
        raise ValueError("RMS cannot be materially smaller than the absolute mean")
    result = np.maximum(numerator, 0.0) / float(count - 1)
    result = np.array(result, copy=True, order="C")
    result.setflags(write=False)
    return result


def _string_identity_axis(values: Sequence[str], *, name: str) -> tuple[str, ...]:
    identifiers = tuple(str(value) for value in values)
    if any(not value for value in identifiers) or len(set(identifiers)) != len(
        identifiers
    ):
        raise ValueError(f"{name} must contain unique nonempty identities")
    return identifiers


@dataclass(frozen=True)
class FiniteTransitionResponseField:
    """Intrinsic response field augmented by finite-transition evidence."""

    intrinsic_field: IntrinsicResponseField
    directional_second_response: np.ndarray | None = field(
        default=None,
        repr=False,
        compare=False,
    )
    directional_sample_count: int = 0
    transition_outcome_hashes: tuple[str, ...] = ()
    control_probe_response: np.ndarray | None = field(
        default=None,
        repr=False,
        compare=False,
    )
    control_probe_estimator_variance: np.ndarray | None = field(
        default=None,
        repr=False,
        compare=False,
    )
    control_probe_ids: tuple[str, ...] = ()
    control_probe_hash: str | None = None
    finite_dose_nonlinear_response: np.ndarray | None = field(
        default=None,
        repr=False,
        compare=False,
    )
    finite_dose_exact_response: np.ndarray | None = field(
        default=None,
        repr=False,
        compare=False,
    )
    finite_dose_exact_rms: np.ndarray | None = field(
        default=None,
        repr=False,
        compare=False,
    )
    finite_dose_nonlinear_rms: np.ndarray | None = field(
        default=None,
        repr=False,
        compare=False,
    )
    finite_dose_levels: tuple[float, ...] = ()
    finite_dose_route_indices: tuple[int, ...] = ()
    finite_dose_measurement_hash: str | None = None
    finite_dose_sample_count: int = 0
    residual_model_error_envelope: np.ndarray | None = field(
        default=None,
        repr=False,
        compare=False,
    )
    transition_receipts: tuple[ResponseTransitionReceipt, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.intrinsic_field, IntrinsicResponseField):
            raise TypeError(
                "finite-transition geometry requires IntrinsicResponseField"
            )
        factor = self.intrinsic_field.intrinsic_response_factor
        if factor is None:
            raise ValueError(
                "finite-transition geometry requires an intrinsic response factor"
            )
        route_count = len(self.intrinsic_field.route_ids)
        response_dimension = factor.response_dimension

        directional = readonly_optional_matrix(
            self.directional_second_response,
            name="directional_second_response",
        )
        directional_count = int(self.directional_sample_count)
        if directional_count < 0:
            raise ValueError("directional_sample_count must be nonnegative")
        if directional is not None and directional.shape != (
            response_dimension,
            route_count,
        ):
            raise ValueError(
                "directional_second_response must have [response,physical_route] axes"
            )

        outcome_hashes = _string_identity_axis(
            self.transition_outcome_hashes,
            name="transition_outcome_hashes",
        )

        control = readonly_optional_tensor3(
            self.control_probe_response,
            name="control_probe_response",
        )
        control_variance = readonly_optional_tensor3(
            self.control_probe_estimator_variance,
            name="control_probe_estimator_variance",
        )
        probe_ids = _string_identity_axis(
            self.control_probe_ids,
            name="control_probe_ids",
        )
        if control is not None:
            if (
                control.shape != (len(probe_ids), response_dimension, route_count)
                or not probe_ids
                or not self.control_probe_hash
            ):
                raise ValueError(
                    "control probes require aligned probe, response, and physical-route axes"
                )
        elif probe_ids or self.control_probe_hash is not None:
            raise ValueError("control-probe identity requires measured responses")
        if control_variance is not None and (
            control is None
            or control_variance.shape != control.shape
            or np.any(control_variance < 0.0)
        ):
            raise ValueError(
                "control-probe estimator variance must be nonnegative and aligned"
            )

        finite_nonlinear = readonly_optional_tensor3(
            self.finite_dose_nonlinear_response,
            name="finite_dose_nonlinear_response",
        )
        finite_exact = readonly_optional_tensor3(
            self.finite_dose_exact_response,
            name="finite_dose_exact_response",
        )
        finite_exact_rms = readonly_optional_tensor3(
            self.finite_dose_exact_rms,
            name="finite_dose_exact_rms",
        )
        finite_nonlinear_rms = readonly_optional_tensor3(
            self.finite_dose_nonlinear_rms,
            name="finite_dose_nonlinear_rms",
        )
        doses = tuple(float(value) for value in self.finite_dose_levels)
        finite_routes = tuple(int(value) for value in self.finite_dose_route_indices)
        finite_count = int(self.finite_dose_sample_count)
        if finite_count < 0:
            raise ValueError("finite_dose_sample_count must be nonnegative")
        if finite_nonlinear is not None:
            expected_shape = (len(doses), response_dimension, len(finite_routes))
            if (
                finite_nonlinear.shape != expected_shape
                or finite_exact is None
                or finite_exact_rms is None
                or finite_nonlinear_rms is None
                or finite_exact.shape != expected_shape
                or finite_exact_rms.shape != expected_shape
                or finite_nonlinear_rms.shape != expected_shape
                or not doses
                or tuple(sorted(set(doses))) != doses
                or any(
                    not math.isfinite(value) or not 0.0 < value <= 1.0
                    for value in doses
                )
                or not finite_routes
                or len(set(finite_routes)) != len(finite_routes)
                or any(value < 0 or value >= route_count for value in finite_routes)
                or not self.finite_dose_measurement_hash
                or finite_count <= 1
            ):
                raise ValueError(
                    "finite-dose field requires aligned exact/nonlinear mean/RMS summaries, "
                    "valid dose/route axes, a measurement identity, and at least two contexts"
                )
            # Refuse impossible retained summaries at construction rather than
            # silently clipping material violations when variance is requested.
            uniform_sample_mean_variance_from_mean_rms(
                finite_exact,
                finite_exact_rms,
                finite_count,
            )
            uniform_sample_mean_variance_from_mean_rms(
                finite_nonlinear,
                finite_nonlinear_rms,
                finite_count,
            )
        elif (
            finite_exact is not None
            or finite_exact_rms is not None
            or finite_nonlinear_rms is not None
            or doses
            or finite_routes
            or self.finite_dose_measurement_hash is not None
            or finite_count
        ):
            raise ValueError(
                "finite-dose identity and summaries require measured responses"
            )

        envelope = readonly_optional_psd(
            self.residual_model_error_envelope,
            name="residual_model_error_envelope",
        )
        if envelope is not None and envelope.shape != (
            response_dimension,
            response_dimension,
        ):
            raise ValueError(
                "residual model-error envelope uses an incompatible response axis"
            )

        receipts = tuple(self.transition_receipts)
        receipt_hashes = tuple(item.content_hash for item in receipts)
        if len(set(receipt_hashes)) != len(receipt_hashes):
            raise ValueError("finite-transition receipts must be unique")
        allowed_state_hashes = {
            self.intrinsic_field.state_hash,
            self.intrinsic_field.capability_hash,
        }
        atlas = self.intrinsic_field.functional_atlas
        if atlas is not None:
            allowed_state_hashes.update(
                str(value)
                for value in dict(getattr(atlas, "metadata", {})).get(
                    "linearization_equivalent_parent_state_hashes",
                    (),
                )
            )
        for receipt in receipts:
            if receipt.state_hash not in allowed_state_hashes:
                raise ValueError(
                    "finite-transition receipt was measured under a different intrinsic field"
                )
            if receipt.validity.exact.response_dimension != response_dimension:
                raise ValueError(
                    "finite-transition receipt uses an incompatible response axis"
                )
            if any(
                index < 0 or index >= route_count for index in receipt.action_indices
            ):
                raise ValueError(
                    "finite-transition receipt uses an invalid physical route index"
                )

        object.__setattr__(self, "directional_second_response", directional)
        object.__setattr__(self, "directional_sample_count", directional_count)
        object.__setattr__(self, "transition_outcome_hashes", outcome_hashes)
        object.__setattr__(self, "control_probe_response", control)
        object.__setattr__(self, "control_probe_estimator_variance", control_variance)
        object.__setattr__(self, "control_probe_ids", probe_ids)
        object.__setattr__(self, "finite_dose_nonlinear_response", finite_nonlinear)
        object.__setattr__(self, "finite_dose_exact_response", finite_exact)
        object.__setattr__(self, "finite_dose_exact_rms", finite_exact_rms)
        object.__setattr__(self, "finite_dose_nonlinear_rms", finite_nonlinear_rms)
        object.__setattr__(self, "finite_dose_levels", doses)
        object.__setattr__(self, "finite_dose_route_indices", finite_routes)
        object.__setattr__(self, "finite_dose_sample_count", finite_count)
        object.__setattr__(self, "residual_model_error_envelope", envelope)
        object.__setattr__(self, "transition_receipts", receipts)

    @property
    def route_ids(self) -> tuple[str, ...]:
        return self.intrinsic_field.route_ids

    @property
    def route_costs(self) -> np.ndarray:
        return self.intrinsic_field.route_costs

    @property
    def protected(self) -> np.ndarray:
        return self.intrinsic_field.protected

    @property
    def measured_mask_hash(self) -> str:
        return self.intrinsic_field.measured_mask_hash

    @property
    def finite_dose_exact_estimator_variance(self) -> np.ndarray | None:
        if self.finite_dose_exact_response is None:
            return None
        assert self.finite_dose_exact_rms is not None
        return uniform_sample_mean_variance_from_mean_rms(
            self.finite_dose_exact_response,
            self.finite_dose_exact_rms,
            self.finite_dose_sample_count,
        )

    @property
    def finite_dose_nonlinear_estimator_variance(self) -> np.ndarray | None:
        if self.finite_dose_nonlinear_response is None:
            return None
        assert self.finite_dose_nonlinear_rms is not None
        return uniform_sample_mean_variance_from_mean_rms(
            self.finite_dose_nonlinear_response,
            self.finite_dose_nonlinear_rms,
            self.finite_dose_sample_count,
        )

    def canonical_action(
        self,
        indices: Sequence[int],
        coefficients: Sequence[float] | None = None,
    ) -> tuple[tuple[int, ...], tuple[float, ...]]:
        """Canonicalize one action on the complete physical route axis."""

        raw_indices = tuple(int(index) for index in indices)
        raw_values = (
            tuple(-1.0 for _ in raw_indices)
            if coefficients is None
            else tuple(float(value) for value in coefficients)
        )
        if len(raw_values) != len(raw_indices):
            raise ValueError("action requires one coefficient per route")
        pairs = tuple(sorted(zip(raw_indices, raw_values), key=lambda item: item[0]))
        selected = tuple(item[0] for item in pairs)
        values = tuple(item[1] for item in pairs)
        if len(set(selected)) != len(selected):
            raise ValueError("action indices must be unique")
        if any(index < 0 or index >= len(self.route_ids) for index in selected):
            raise ValueError("action indices fall outside the physical route axis")
        if any(not math.isfinite(value) for value in values):
            raise ValueError("action coefficients must be finite")
        return selected, values

    def transition_receipt_for(
        self,
        indices: Sequence[int],
        coefficients: Sequence[float] | None = None,
    ) -> ResponseTransitionReceipt | None:
        """Return the exact receipt for the same canonical action, if measured."""

        key = self.canonical_action(indices, coefficients)
        for receipt in self.transition_receipts:
            if receipt.action_key == key:
                return receipt
        return None

    def action_moments(
        self,
        indices: Sequence[int],
        coefficients: Sequence[float] | None = None,
        *,
        prefer_exact_receipt: bool = True,
    ) -> ResponseActionMoments:
        """Evaluate a physical action with exact override and conservative jets."""

        selected, values = self.canonical_action(indices, coefficients)
        receipt = (
            self.transition_receipt_for(selected, values)
            if prefer_exact_receipt and selected
            else None
        )
        if receipt is not None:
            return receipt.validity.exact

        # IntrinsicResponseField performs the explicit physical-to-screened
        # route map and refuses unmeasured physical routes.
        moments = self.intrinsic_field.intrinsic_action_moments(selected, values)
        if self.directional_second_response is not None and selected:
            chosen = np.asarray(selected, dtype=np.int64)
            coefficient_array = np.asarray(values, dtype=np.float64)
            correction = 0.5 * (
                self.directional_second_response[:, chosen]
                @ (coefficient_array * coefficient_array)
            )
            moments = moments.with_mean_shift(
                correction,
                approximation="intrinsic_response_plus_route_directional_second_jet",
                metadata={
                    "directional_second_response_retained": True,
                    "cross_route_curvature_retained": False,
                },
            )
        if self.residual_model_error_envelope is not None and selected:
            scale = float(np.sum(np.abs(values))) ** 2
            moments = moments.inflate_model_error_covariance(
                scale * self.residual_model_error_envelope,
                approximation="finite_transition_response_with_residual_envelope",
                metadata={
                    "residual_model_error_envelope_scale": scale,
                    "residual_envelope_normalization": "squared_l1_action_magnitude",
                },
            )
        return moments

    @cached_property
    def capability_hash(self) -> str:
        """Workflow-neutral identity of this finite-transition capability."""

        return content_hash(
            {
                "capability": "finite_transition_response_field_v1",
                "intrinsic_field_capability_hash": self.intrinsic_field.capability_hash,
                "directional_second_response": self.directional_second_response,
                "directional_sample_count": self.directional_sample_count,
                "transition_outcome_hashes": self.transition_outcome_hashes,
                "control_probe_response": self.control_probe_response,
                "control_probe_estimator_variance": self.control_probe_estimator_variance,
                "control_probe_ids": self.control_probe_ids,
                "control_probe_hash": self.control_probe_hash,
                "finite_dose_nonlinear_response": self.finite_dose_nonlinear_response,
                "finite_dose_exact_response": self.finite_dose_exact_response,
                "finite_dose_exact_rms": self.finite_dose_exact_rms,
                "finite_dose_nonlinear_rms": self.finite_dose_nonlinear_rms,
                "finite_dose_levels": self.finite_dose_levels,
                "finite_dose_route_indices": self.finite_dose_route_indices,
                "finite_dose_measurement_hash": self.finite_dose_measurement_hash,
                "finite_dose_sample_count": self.finite_dose_sample_count,
                "residual_model_error_envelope": self.residual_model_error_envelope,
                "transition_receipt_hashes": tuple(
                    receipt.content_hash for receipt in self.transition_receipts
                ),
            },
            prefix="finite-transition-response-field-",
        )


def legacy_v4_state_payload(
    field: FiniteTransitionResponseField,
    *,
    legacy_directional_curvature: np.ndarray,
    legacy_learned_route_residual: np.ndarray,
    legacy_atlas_transition_receipt_hashes: Sequence[str] = (),
) -> dict[str, Any]:
    """Serialize the exact historical v4 state payload for receipt verification.

    The two scalar route vectors and atlas-receipt hashes existed only in the
    historical workflow carrier.  Requiring them explicitly prevents this
    neutral capability from treating them as current semantics.
    """

    if not isinstance(field, FiniteTransitionResponseField):
        raise TypeError("legacy v4 serialization requires a finite-transition field")
    curvature = readonly_vector(
        legacy_directional_curvature,
        name="legacy_directional_curvature",
    )
    residual = readonly_vector(
        legacy_learned_route_residual,
        name="legacy_learned_route_residual",
    )
    if curvature.size != len(field.route_ids) or residual.size != len(field.route_ids):
        raise ValueError(
            "legacy v4 scalar vectors must align with the physical route axis"
        )
    atlas_hashes = _string_identity_axis(
        legacy_atlas_transition_receipt_hashes,
        name="legacy_atlas_transition_receipt_hashes",
    )
    return {
        "version": "v4",
        "intrinsic_state_hash": field.intrinsic_field.state_hash,
        "directional_curvature": curvature,
        "learned_route_residual": residual,
        "curvature_sample_count": field.directional_sample_count,
        "directional_second_response": field.directional_second_response,
        "control_probe_response": field.control_probe_response,
        "control_probe_estimator_variance": field.control_probe_estimator_variance,
        "control_probe_ids": field.control_probe_ids,
        "control_probe_hash": field.control_probe_hash,
        "finite_dose_nonlinear_response": field.finite_dose_nonlinear_response,
        "finite_dose_exact_response": field.finite_dose_exact_response,
        "finite_dose_exact_rms": field.finite_dose_exact_rms,
        "finite_dose_nonlinear_rms": field.finite_dose_nonlinear_rms,
        "finite_dose_levels": field.finite_dose_levels,
        "finite_dose_route_indices": field.finite_dose_route_indices,
        "finite_dose_measurement_hash": field.finite_dose_measurement_hash,
        "finite_dose_sample_count": field.finite_dose_sample_count,
        "residual_model_error_envelope": field.residual_model_error_envelope,
        "transition_receipt_hashes": tuple(
            receipt.content_hash for receipt in field.transition_receipts
        ),
        "transition_outcome_hashes": field.transition_outcome_hashes,
        "atlas_transition_receipt_hashes": atlas_hashes,
    }


def legacy_v4_state_hash(
    field: FiniteTransitionResponseField,
    *,
    legacy_directional_curvature: np.ndarray,
    legacy_learned_route_residual: np.ndarray,
    legacy_atlas_transition_receipt_hashes: Sequence[str] = (),
) -> str:
    """Return the historical v4 content address without granting authority."""

    return content_hash(
        legacy_v4_state_payload(
            field,
            legacy_directional_curvature=legacy_directional_curvature,
            legacy_learned_route_residual=legacy_learned_route_residual,
            legacy_atlas_transition_receipt_hashes=(
                legacy_atlas_transition_receipt_hashes
            ),
        ),
        prefix="irg-v4-state-",
    )


__all__ = [
    "FiniteTransitionResponseField",
    "legacy_v4_state_hash",
    "legacy_v4_state_payload",
    "positive_semidefinite_part",
    "psd_common_upper_envelope",
    "readonly_optional_matrix",
    "readonly_optional_psd",
    "readonly_optional_tensor3",
    "uniform_sample_mean_variance_from_mean_rms",
]
