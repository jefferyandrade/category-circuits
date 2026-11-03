"""Per-context logit intervals using the retained analytic-envelope primitives.

These bounds are conditional on the retained storage/execution error model.
They are not population certificates. Work/J projection enclosures and hardware
qualification remain separate obligations. The evaluator preserves actual
circuit biases and evaluates one context/dose at a time with tiled linear rows.
"""
from types import SimpleNamespace
from dataclasses import replace
from pathlib import Path
import hashlib

import numpy as np
import torch
import torch.nn.functional as F

from mechanistic_systems.contracts.content_address import content_hash
from .. import modern_v1_alexnet_envelope as envelope
from .alexnet_factors import MaskedAlexNet
from . import masked_alexnet_paired_bce as leases

NATIVE_EXECUTION = 'native_storage_execution_family'
STORED_REAL_FUNCTION = 'exact_stored_coefficient_real_function'


class RawTargetLogitIntervalEvaluator:
    """A process-local numerical evaluator, with no image-source authority.

    Native scope includes the retained model's native execution error family.
    Stored-real scope encloses the real-arithmetic function of the exact stored
    coefficients, realized mask and converted input. It uses the same FP64
    bound engine, without adding hypothetical native FP32 errors at every cut.
    It does not certify FP32 forwards. Scope is immutable and identity-bound.
    """
    def __init__(self, net: MaskedAlexNet, *, linear_row_chunk: int = 64,
                 bound_scope: str = NATIVE_EXECUTION):
        if type(net) is not MaskedAlexNet:
            raise TypeError("point intervals require exact MaskedAlexNet execution")
        if type(linear_row_chunk) is not int or linear_row_chunk <= 0:
            raise ValueError("positive interval linear-row chunk required")
        if type(bound_scope) is not str or bound_scope not in (NATIVE_EXECUTION, STORED_REAL_FUNCTION):
            raise ValueError("explicit supported logit interval bound scope required")
        self.net = net
        self.linear_row_chunk = linear_row_chunk
        self.bound_scope = bound_scope
        # Tensor/geometry views supply the retained arithmetic model. They do
        # not create another model, alter weights, or claim graph admission.
        state_view = SimpleNamespace(state_dict=lambda: dict(leases._model_tensor_items(net)))
        self.native_profile = envelope._arithmetic_profile(state_view)
        self.profile = self.native_profile
        if bound_scope == STORED_REAL_FUNCTION:
            # FP32 -> FP64 coefficient/input conversion is exact. The retained
            # FP64 profile conservatively covers bound-engine contractions and
            # effective weight*mask products (including FP64 underflow). No
            # whole-model FP64 copy is needed: existing affine tiles convert
            # the actual source coefficients as they are used. storage_dtype
            # here describes the reference engine; native_profile retains the
            # actual model's execution dtype and roundoff law separately.
            self.profile = replace(self.native_profile, storage_dtype='torch.float64',
                multiplicand_unit_roundoff=2.**-53, accumulation_unit_roundoff=2.**-53,
                bound_engine_unit_roundoff=2.**-53,
                numerical_ceiling=float(torch.finfo(torch.float64).max)/16.,
                tf32_multiplicand_rounding_admitted=False)
        self._leases = leases._model_leases(net)
        self._structure = leases._structural_fingerprint(net)
        self._profile_hash = self.profile.profile_hash
        self._native_profile_hash = self.native_profile.profile_hash
        self._scope = bound_scope
        self._chunk = linear_row_chunk
        self.evaluator_hash = content_hash(dict(
            schema="v1_actual_point_logit_interval_evaluator_v1",
            structure=self._structure, arithmetic_profile_hash=self._profile_hash,
            native_arithmetic_profile_hash=self._native_profile_hash, bound_scope=bound_scope,
            linear_row_chunk=linear_row_chunk,
            envelope_source_sha256=envelope._source_sha256(),
            point_adapter_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            input_law="exact_stored_values_after_forward_dtype_conversion",
            scope="conditional_retained_execution_error_model_no_population_authority"),
            prefix="v1-point-logit-interval-evaluator-")

    def assert_live(self):
        leases._check_model_leases(self.net, self._leases)
        if (leases._structural_fingerprint(self.net) != self._structure
                or self.profile.profile_hash != self._profile_hash
                or self.native_profile.profile_hash != self._native_profile_hash
                or self.bound_scope != self._scope
                or self.linear_row_chunk != self._chunk):
            raise ValueError("point interval model/profile/configuration changed")

    def _affine_views(self, *, use_dense_biases=False):
        views = []
        biases = self.net._dense_biases if use_dense_biases else self.net.biases
        for geometry, weight, bias in zip(self.net.axis.layers, self.net.weights, biases, strict=True):
            fields = dict(weight=weight, bias=bias)
            if geometry.kind == 'conv':
                kernel, stride, padding = geometry.conv_params
                fields.update(in_channels=weight.shape[1], groups=1, kernel_size=(kernel,kernel),
                              stride=stride, padding=padding, dilation=1)
            else:
                fields.update(in_features=weight.shape[1], out_features=weight.shape[0])
            views.append(SimpleNamespace(**fields))
        readout = SimpleNamespace(weight=self.net.readout_weight,
                                  bias=self.net._dense_readout_bias if use_dense_biases else self.net.readout_bias,
                                  in_features=self.net.readout_weight.shape[1],
                                  out_features=self.net.readout_weight.shape[0])
        return views, readout

    @torch.no_grad()
    def evaluate(self, images: torch.Tensor, *, masks: tuple[torch.Tensor, ...],
                 target_class: int, upper_images: torch.Tensor | None = None,
                 use_dense_biases: tuple[bool, ...] | None = None) -> tuple[np.ndarray, np.ndarray]:
        self.assert_live()
        if (type(images) is not torch.Tensor or images.ndim != 4
                or tuple(images.shape[1:]) != (3,224,224) or images.shape[0] == 0
                or not bool(torch.isfinite(images).all())):
            raise ValueError("finite nonempty canonical image batch required")
        if upper_images is not None and (
            type(upper_images) is not torch.Tensor or upper_images.shape != images.shape
            or upper_images.device != images.device or not bool(torch.isfinite(upper_images).all())
            or bool((images > upper_images).any())
        ):
            raise ValueError("finite ordered input-box tensors must match lower images")
        if type(target_class) is not int or not 0 <= target_class < self.net.readout_bias.numel():
            raise ValueError("point interval target outside actual readout")
        if type(masks) is not tuple or not masks:
            raise ValueError("nonempty tuple of finite-dose masks required")
        if use_dense_biases is None:
            use_dense_biases = (False,) * len(masks)
        if (type(use_dense_biases) is not tuple or len(use_dense_biases) != len(masks)
                or any(type(value) is not bool for value in use_dense_biases)):
            raise ValueError('one exact dense-bias flag per interval mask required')
        image_lease = leases._tensor_lease(images, name="point interval images")
        upper_lease = (None if upper_images is None else
                       leases._tensor_lease(upper_images, name="interval upper images"))
        realized_masks = tuple(self.net._forward_mask(mask) for mask in masks)
        mask_leases = tuple(leases._tensor_lease(mask, name=f"point mask {i}")
                            for i,mask in enumerate(realized_masks))
        lane_views = tuple(self._affine_views(use_dense_biases=value) for value in use_dense_biases)
        result_lower = np.empty((len(images),len(masks)))
        result_upper = np.empty_like(result_lower)
        for context, source_image in enumerate(images):
            # Match forward conversion while retaining only one FP64 input
            # context, not an additional complete image batch on the device.
            image = source_image.to(self.net.device, self.net.dtype).to(torch.float64)
            image_upper = (image if upper_images is None else
                           upper_images[context].to(self.net.device, self.net.dtype).to(torch.float64))
            for dose, mask in enumerate(realized_masks):
                views, readout = lane_views[dose]
                lower, upper = image, image_upper
                for geometry, view, gate in zip(self.net.axis.layers, views,
                                               self.net.axis.split(mask), strict=True):
                    if geometry.kind == 'conv':
                        lower, upper, _ = envelope._conv_forward_interval_bound(
                            lower, upper, view, gate.reshape(view.weight.shape), self.profile)
                    else:
                        if lower.ndim == 3:
                            if tuple(lower.shape[-2:]) != (6,6):
                                raise ValueError("point interval avgpool identity shape differs")
                            lower, upper = lower.flatten(), upper.flatten()
                        lower, upper, _ = envelope._linear_forward_interval_bound(
                            lower, upper, view, gate.reshape(view.weight.shape), self.profile,
                            output_row_chunk=self.linear_row_chunk)
                    lower, upper = lower.clamp_min(0.), upper.clamp_min(0.)
                    if geometry.name in self.net.pool_after:
                        lower = F.max_pool2d(lower.unsqueeze(0), 3, 2)[0]
                        upper = F.max_pool2d(upper.unsqueeze(0), 3, 2)[0]
                    envelope._assert_interval_tensors(lower, upper, profile=self.profile,
                                                      name="point ReLU/pool")
                lower, upper, _ = envelope._linear_forward_interval_bound(
                    lower, upper, readout, None, self.profile, output_row_chunk=1,
                    output_row_start=target_class, output_row_stop=target_class+1)
                result_lower[context,dose] = float(lower[0])
                result_upper[context,dose] = float(upper[0])
        leases._check_tensor_lease(images, image_lease)
        if upper_images is not None:
            leases._check_tensor_lease(upper_images, upper_lease)
        for mask,lease in zip(realized_masks,mask_leases,strict=True):
            leases._check_tensor_lease(mask,lease)
        self.assert_live()
        return result_lower, result_upper

    def evaluate_registered_box(self, box: envelope.V1RegisteredNormalizedInputBox, *,
                                masks: tuple[torch.Tensor, ...], target_class: int,
                                use_dense_biases: tuple[bool, ...] | None = None):
        if type(box) is not envelope.V1RegisteredNormalizedInputBox:
            raise TypeError("population support requires the retained registered input box")
        box.assert_integrity()
        if box.spatial_shape != (224,224):
            raise ValueError("registered input box differs from canonical forward shape")
        lower = torch.tensor(box.lower_by_channel, dtype=torch.float64,
                             device=self.net.device).reshape(1,3,1,1).expand(1,3,224,224)
        upper = torch.tensor(box.upper_by_channel, dtype=torch.float64,
                             device=self.net.device).reshape(1,3,1,1).expand_as(lower)
        return self.evaluate(lower, upper_images=upper, masks=masks, target_class=target_class,
                             use_dense_biases=use_dense_biases)
