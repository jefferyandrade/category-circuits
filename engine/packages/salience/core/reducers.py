from __future__ import annotations

from typing import Dict, Iterable, List, Mapping, Optional

import torch

from packages.salience.ops.abs_mode import is_frontier_aggregate_abs_mode, normalize_abs_aggregation_mode


class Reducer:
	"""Minimal reducer interface used by streaming scoring paths."""

	def reset(self, config: Optional[Mapping[str, object]] = None) -> None:  # pragma: no cover - interface
		raise NotImplementedError

	def update(self, slice_data: Mapping[str, Optional[torch.Tensor]]) -> None:  # pragma: no cover - interface
		raise NotImplementedError

	def finalize(self) -> Mapping[str, Optional[torch.Tensor]]:  # pragma: no cover - interface
		raise NotImplementedError


class KernelScoreReducer(Reducer):
	"""
	Accumulate running kernel statistics in float32 on the target device.

	The reducer keeps the rolling act×grad sums alongside optional neuron-
	aware products so layer hooks can stream slices without materializing
	full-batch tensors.
	"""

	_BASE_FIELDS: tuple[str, ...] = (
		"kernel_scores_sum",
		"activating_scores_sum",
		"inhibiting_scores_sum",
		"scores_signed_acts_sum",
		"abs_scores_sum",
		"gradients_sum",
		"abs_gradients_sum",
	)
	_NEURON_FIELDS: tuple[str, ...] = (
		"product_scores_sum",
		"abs_product_scores_sum",
		"neuron_scores_sum",
		"abs_neuron_scores_sum",
		"neuron_scores_sq_sum",
		"abs_neuron_scores_sq_sum",
	)

	def __init__(self, *, device: torch.device, compute_neuron_metrics: bool = True) -> None:
		self.device = torch.device(device)
		self.compute_neuron_metrics = bool(compute_neuron_metrics)
		self._state: Dict[str, Optional[torch.Tensor]] = {}
		self._corr_values: List[torch.Tensor] = []
		self.reset()

	def reset(self, config: Optional[Mapping[str, object]] = None) -> None:
		if config is not None and "compute_neuron_metrics" in config:
			self.compute_neuron_metrics = bool(config["compute_neuron_metrics"])
		self._state = {name: None for name in self._BASE_FIELDS}
		if self.compute_neuron_metrics:
			for name in self._NEURON_FIELDS:
				self._state[name] = None
		else:
			for name in self._NEURON_FIELDS:
				self._state.pop(name, None)
		self._corr_values = []

	def update(self, slice_data: Mapping[str, Optional[torch.Tensor]]) -> None:
		for name in self._BASE_FIELDS:
			self._accumulate(name, slice_data.get(name))
		if self.compute_neuron_metrics:
			for name in self._NEURON_FIELDS:
				self._accumulate(name, slice_data.get(name))

		corr_value = slice_data.get("corr")
		if torch.is_tensor(corr_value):
			self._corr_values.append(corr_value.detach().to(device=self.device, dtype=torch.float32))

	def finalize(self) -> Mapping[str, Optional[torch.Tensor]]:
		result: Dict[str, Optional[torch.Tensor]] = {}
		for name, tensor in self._state.items():
			if tensor is None:
				result[name] = None
			else:
				result[name] = tensor.detach().cpu()

		if self._corr_values:
			stacked = torch.stack(self._corr_values).mean().detach().cpu()
			result["corr"] = stacked
		else:
			result["corr"] = None
		return result

	def _accumulate(self, name: str, value: Optional[torch.Tensor]) -> None:
		if value is None:
			return
		tensor = value.detach()
		if tensor.dtype != torch.float32:
			tensor = tensor.to(dtype=torch.float32)
		if tensor.device != self.device:
			tensor = tensor.to(device=self.device)
		existing = self._state.get(name)
		if existing is None:
			self._state[name] = tensor.clone()
		else:
			existing.add_(tensor)


class LossDeltaReducer(Reducer):
	"""Collect running loss deltas emitted per slice."""

	_FIELDS: tuple[str, ...] = ("loss_change_sum", "abs_loss_change_sum")

	def __init__(self, *, device: torch.device) -> None:
		self.device = torch.device(device)
		self._state: Dict[str, Optional[torch.Tensor]] = {}
		self.reset()

	def reset(self, config: Optional[Mapping[str, object]] = None) -> None:
		self._state = {name: None for name in self._FIELDS}

	def update(self, slice_data: Mapping[str, Optional[torch.Tensor]]) -> None:
		for name in self._FIELDS:
			self._accumulate(name, slice_data.get(name))

	def finalize(self) -> Mapping[str, Optional[torch.Tensor]]:
		return {
			name: (tensor.detach().cpu() if tensor is not None else None)
			for name, tensor in self._state.items()
		}

	def _accumulate(self, name: str, value: Optional[torch.Tensor]) -> None:
		if value is None:
			return
		tensor = value.detach()
		if tensor.dtype != torch.float32:
			tensor = tensor.to(dtype=torch.float32)
		if tensor.device != self.device:
			tensor = tensor.to(device=self.device)
		existing = self._state.get(name)
		if existing is None:
			self._state[name] = tensor.clone()
		else:
			existing.add_(tensor)


class BitpackReducer(Reducer):
	"""Helper reducer that streams bitpacked rows directly to CPU memory."""

	def __init__(self) -> None:
		self._rows: List[torch.Tensor] = []

	def reset(self, config: Optional[Mapping[str, object]] = None) -> None:
		self._rows = []

	def update(self, slice_data: Mapping[str, Optional[torch.Tensor]]) -> None:
		rows = slice_data.get("bitpacked")
		if rows is None:
			return
		if isinstance(rows, torch.Tensor):
			self._rows.append(rows.detach().cpu())
		elif isinstance(rows, Iterable):
			for row in rows:
				if torch.is_tensor(row):
					self._rows.append(row.detach().cpu())

	def finalize(self) -> List[torch.Tensor]:
		return list(self._rows)


class BieReducer(Reducer):
	"""Accumulate boundary integral estimates (base or boundary correction)."""

	def __init__(self, *, device: torch.device, prefix: str) -> None:
		self.device = torch.device(device)
		self.prefix = prefix
		self._sum_key = f"{prefix}_sum"
		self._abs_key = f"{prefix}_abs_sum"
		self._state: Dict[str, Optional[torch.Tensor]] = {}
		self.reset()

	def reset(self, config: Optional[Mapping[str, object]] = None) -> None:
		self._state = {
			self._sum_key: None,
			self._abs_key: None,
		}

	def update(self, slice_data: Mapping[str, Optional[torch.Tensor]]) -> None:
		self._accumulate(self._sum_key, slice_data.get(self._sum_key))
		self._accumulate(self._abs_key, slice_data.get(self._abs_key))

	def finalize(self) -> Mapping[str, Optional[torch.Tensor]]:
		result: Dict[str, Optional[torch.Tensor]] = {}
		sum_tensor = self._state.get(self._sum_key)
		if torch.is_tensor(sum_tensor):
			result[self._sum_key] = sum_tensor.detach().cpu()
		else:
			result[self._sum_key] = None
		abs_tensor = self._state.get(self._abs_key)
		if torch.is_tensor(abs_tensor):
			result[self._abs_key] = abs_tensor.detach().cpu()
		else:
			result[self._abs_key] = None
		return result

	def _accumulate(self, name: str, value: Optional[torch.Tensor]) -> None:
		if value is None:
			return
		tensor = value.detach()
		if tensor.dtype != torch.float32:
			tensor = tensor.to(dtype=torch.float32)
		if tensor.device != self.device:
			tensor = tensor.to(device=self.device)
		current = self._state.get(name)
		if current is None:
			self._state[name] = tensor.clone()
		else:
			current.add_(tensor)


class ReducerBundle:
	"""
	Coordinate the reducer set held by a frontier layer.

	The bundle keeps per-layer metadata (e.g., weights, chunk sizes) alongside
	the streaming reducers so hooks can publish consolidated results in a
	single pass.
	"""

	def __init__(
		self,
		*,
		device: torch.device,
		track_categories: bool,
		compute_neuron_metrics: bool,
		collect_bitpack: bool = True,
		collect_bie: bool = False,
	) -> None:
		self.device = torch.device(device)
		self.track_categories = bool(track_categories)
		self.compute_neuron_metrics = bool(compute_neuron_metrics)
		self.collect_bitpack = bool(collect_bitpack)
		self.collect_bie = bool(collect_bie)

		self.kernel_all = KernelScoreReducer(device=self.device, compute_neuron_metrics=self.compute_neuron_metrics)
		self.kernel_categories = (
			KernelScoreReducer(device=self.device, compute_neuron_metrics=self.compute_neuron_metrics)
			if self.track_categories
			else None
		)

		self.loss_all = LossDeltaReducer(device=self.device)
		self.loss_categories = LossDeltaReducer(device=self.device) if self.track_categories else None

		self.bitpack = BitpackReducer() if self.collect_bitpack else None
		self.bie_base_all = BieReducer(device=self.device, prefix="bie_base") if self.collect_bie else None
		self.bie_boundary_all = BieReducer(device=self.device, prefix="bie_boundary") if self.collect_bie else None
		self.bie_base_categories = (
			BieReducer(device=self.device, prefix="bie_base") if (self.collect_bie and self.track_categories) else None
		)
		self.bie_boundary_categories = (
			BieReducer(device=self.device, prefix="bie_boundary") if (self.collect_bie and self.track_categories) else None
		)
		self.metadata: Dict[str, object] = {}

	def reset(self, config: Optional[Mapping[str, object]] = None) -> None:
		cfg = dict(config or {})
		cfg.setdefault("compute_neuron_metrics", self.compute_neuron_metrics)
		self.kernel_all.reset(cfg)
		if self.kernel_categories is not None:
			self.kernel_categories.reset(cfg)

		self.loss_all.reset(cfg)
		if self.loss_categories is not None:
			self.loss_categories.reset(cfg)

		if self.bie_base_all is not None:
			self.bie_base_all.reset(cfg)
		if self.bie_boundary_all is not None:
			self.bie_boundary_all.reset(cfg)
		if self.bie_base_categories is not None:
			self.bie_base_categories.reset(cfg)
		if self.bie_boundary_categories is not None:
			self.bie_boundary_categories.reset(cfg)

		if self.bitpack is not None:
			self.bitpack.reset(cfg)

		self.metadata = {}

	def update_all(self, slice_data: Mapping[str, Optional[torch.Tensor]]) -> None:
		self.kernel_all.update(slice_data)
		self.loss_all.update(slice_data)
		if self.bie_base_all is not None:
			self.bie_base_all.update(slice_data)
		if self.bie_boundary_all is not None:
			self.bie_boundary_all.update(slice_data)
		if self.bitpack is not None:
			self.bitpack.update(slice_data)

	def update_categories(self, slice_data: Mapping[str, Optional[torch.Tensor]]) -> None:
		if self.kernel_categories is not None:
			self.kernel_categories.update(slice_data)
		if self.loss_categories is not None:
			self.loss_categories.update(slice_data)
		if self.bie_base_categories is not None:
			self.bie_base_categories.update(slice_data)
		if self.bie_boundary_categories is not None:
			self.bie_boundary_categories.update(slice_data)

	def record_metadata(self, **kwargs: object) -> None:
		self.metadata.update(kwargs)

	def finalize(self) -> Dict[str, object]:
		all_payload = {**self.kernel_all.finalize(), **self.loss_all.finalize()}
		if self.bie_base_all is not None:
			all_payload.update(self.bie_base_all.finalize())
		if self.bie_boundary_all is not None:
			all_payload.update(self.bie_boundary_all.finalize())
		cat_payload: Optional[Mapping[str, Optional[torch.Tensor]]]
		if self.kernel_categories is not None and self.loss_categories is not None:
			cat_payload = {**self.kernel_categories.finalize(), **self.loss_categories.finalize()}
			if self.bie_base_categories is not None:
				cat_payload.update(self.bie_base_categories.finalize())
			if self.bie_boundary_categories is not None:
				cat_payload.update(self.bie_boundary_categories.finalize())
		else:
			cat_payload = None

		abs_mode = normalize_abs_aggregation_mode(self.metadata.get("abs_aggregation_mode", "exact_abs"))
		if is_frontier_aggregate_abs_mode(abs_mode):
			_apply_frontier_abs_aggregation(all_payload, self.metadata)
			if isinstance(cat_payload, dict):
				_apply_frontier_abs_aggregation(cat_payload, self.metadata)

		return {
			"all": all_payload,
			"categories": cat_payload,
			"bitpack": self.bitpack.finalize() if self.bitpack is not None else [],
			"metadata": dict(self.metadata),
		}


__all__ = [
	"Reducer",
	"KernelScoreReducer",
	"LossDeltaReducer",
	"BitpackReducer",
	"BieReducer",
	"ReducerBundle",
]


def _apply_frontier_abs_aggregation(
	payload: Mapping[str, Optional[torch.Tensor]],
	metadata: Mapping[str, object],
) -> None:
	if not isinstance(payload, dict):
		return
	kernel_scores = payload.get("kernel_scores_sum")
	if torch.is_tensor(kernel_scores):
		payload["abs_scores_sum"] = kernel_scores.abs()
		payload["activating_scores_sum"] = torch.relu(kernel_scores)
		payload["inhibiting_scores_sum"] = -torch.relu(-kernel_scores)

	gradients = payload.get("gradients_sum")
	if torch.is_tensor(gradients):
		payload["abs_gradients_sum"] = gradients.abs()

	score_basis = str(metadata.get("score_basis", "") or "").strip().lower()
	if torch.is_tensor(kernel_scores):
		if score_basis == "channel_virtual":
			payload["scores_signed_acts_sum"] = kernel_scores.abs()
		else:
			weight_out = metadata.get("weight_out")
			if torch.is_tensor(weight_out) and torch.is_tensor(gradients):
				scale_value = float(metadata.get("scale_value", 1.0) or 1.0)
				scale_sign = -1.0 if scale_value < 0 else 1.0
				weight_cpu = weight_out.detach().to(dtype=torch.float32, device=gradients.device)
				payload["scores_signed_acts_sum"] = weight_cpu * gradients.abs() * scale_sign
			else:
				payload["scores_signed_acts_sum"] = kernel_scores.abs()

	neuron_scores = payload.get("neuron_scores_sum")
	if torch.is_tensor(neuron_scores):
		abs_neuron = neuron_scores.abs()
		payload["abs_neuron_scores_sum"] = abs_neuron
		payload["abs_neuron_scores_sq_sum"] = abs_neuron.square()

	product_scores = payload.get("product_scores_sum")
	if torch.is_tensor(product_scores):
		payload["abs_product_scores_sum"] = product_scores.abs()
