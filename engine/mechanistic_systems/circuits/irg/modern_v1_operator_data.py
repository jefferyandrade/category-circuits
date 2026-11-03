"""Bounded native measurement batches from the verified operator image source.

This adapter transports requested contexts; it never selects statistical draws.
The retained calibration/oracle owners still validate role and epoch authority.
"""
from dataclasses import dataclass, field
from threading import Lock
from types import MappingProxyType
from collections.abc import Sequence

import torch

from mechanistic_systems.contracts.content_address import content_hash
from mechanistic_systems.measurement.contexts import ContextAxis
from mechanistic_systems.measurement.torch_adapter import MeasurementInputBatch
from .modern_v1_neutral_runtime import build_class_balanced_sampling_population
from .tractable_v1.genuine_issuer_bridge import (
    BoundedImageContextSource, _bounded_source_identity,
)


class _BoundedContextPanelTransport:
    """Shared requested-ID transport; concrete owners validate source roles."""
    @property
    def expected_labels(self):
        return self._labels

    def stats(self):
        return {**self._stats, 'batch_rows': self.batch_rows,
            'retained_normalized_cache_bytes': 0,
            'memory_scope': 'adapter output plus one stratum tensor; source decoding/prefetch/cache and caller retention excluded'}

    def panel_batches(self, selected, *, role):
        # Validate the entire request before creating the outcome iterator.
        self.assert_source(self.context_source)
        if isinstance(selected, (str, bytes)) or not isinstance(selected, Sequence):
            raise TypeError('native panel requires an ordered context sequence')
        contexts = tuple(selected)
        if (not contexts or any(type(c) is not str or c not in self._labels for c in contexts)
                or len(contexts) != len(set(contexts))):
            raise ValueError('native panel requires nonempty unique known context IDs')
        if type(role) is not str or not role.strip():
            raise ValueError('native panel requires an explicit role label')
        return self._batches(contexts, role)

    def _batches(self, contexts, role):
        if not self._lock.acquire(blocking=False):
            raise RuntimeError('native image data plane already has an active iterator')
        try:
            self.assert_source(self.context_source)
            self._stats['panels_started'] += 1
            for start in range(0, len(contexts), self.batch_rows):
                self.assert_source(self.context_source)
                ids = contexts[start:start + self.batch_rows]
                output = None
                receipts = []
                for stratum in ('target', 'complement'):
                    positions = tuple(i for i, c in enumerate(ids)
                        if (self._labels[c] == self.target_class_id) == (stratum == 'target'))
                    if not positions:
                        continue
                    group = self.context_source(stratum, tuple(ids[i] for i in positions))
                    self.assert_source(self.context_source)
                    if output is None:
                        output = group.images.new_empty((len(ids), *group.images.shape[1:]))
                    index = torch.tensor(positions, dtype=torch.long, device=output.device)
                    output.index_copy_(0, index, group.images)
                    peak = output.numel()*output.element_size() + group.images.numel()*group.images.element_size()
                    self._stats['adapter_normalized_peak_bytes'] = max(self._stats['adapter_normalized_peak_bytes'], peak)
                    receipts.append(group.source_batch_receipt_hash)
                    del group, index
                batch = MeasurementInputBatch(context_ids=ids, inputs=output,
                    labels=torch.tensor([self._labels[c] for c in ids], dtype=torch.long),
                    input_identity_hash=content_hash(dict(binding_hash=self.binding_hash,
                        role=role, prefix_start=start, context_ids=ids,
                        source_batch_receipt_hashes=tuple(receipts)),
                        prefix='v1-operator-native-image-batch-'))
                batch.validate()
                self._stats['batches_yielded'] += 1
                self._stats['rows_yielded'] += len(ids)
                yield batch
                del batch, output
        finally:
            self._lock.release()


@dataclass(frozen=True, slots=True)
class OperatorNativeImageDataPlane(_BoundedContextPanelTransport):
    """One physical batch at a time, with original class labels and exact IDs.

    Only immutable population metadata persists. Normalized image tensors are
    yielded and released on iterator advance/close; the underlying source owns
    its separately bounded uint8 cache, decoding and prefetch. Caller-retained
    batches are outside this adapter's memory bound.
    """

    context_source: BoundedImageContextSource
    context_axis: ContextAxis
    target_class_id: int
    batch_rows: int = 64
    source_role: str = 'optimization'
    binding_hash: str = field(init=False)
    population_hash: str = field(init=False)
    _labels: object = field(init=False, repr=False, compare=False)
    _source_coordinates: tuple = field(init=False, repr=False, compare=False)
    _source_objects: tuple = field(init=False, repr=False, compare=False)
    _lock: object = field(init=False, repr=False, compare=False)
    _stats: dict = field(init=False, repr=False, compare=False)

    def __post_init__(self):
        if type(self.context_source) is not BoundedImageContextSource:
            raise TypeError('native data plane requires the exact bounded context source')
        if type(self.context_axis) is not ContextAxis:
            raise TypeError('native data plane requires the retained ContextAxis')
        if type(self.batch_rows) is not int or self.batch_rows <= 0:
            raise ValueError('batch_rows must be a positive integer')
        if self.batch_rows > self.context_source.max_batch_rows:
            raise ValueError('native batch exceeds the bounded context source capacity')
        self.context_axis.validate()
        population = build_class_balanced_sampling_population(self.context_axis,
            target_class_id=self.target_class_id, source_role=self.source_role)
        owner = self.context_source.source_owner
        if (population.population_hash != owner.source_population_hash
                or population.sampling_law_hash != owner.sampling_law_hash
                or self.context_axis.preprocessing.content_hash != owner.preprocessing_hash
                or population.on_target_context_ids != owner.target_context_ids
                or population.off_target_context_ids != owner.complement_context_ids):
            raise ValueError('native population/labels differ from the opened source owner')
        object.__setattr__(self, 'population_hash', population.population_hash)
        object.__setattr__(self, '_labels', MappingProxyType(dict(population.label_by_context_id)))
        object.__setattr__(self, '_source_coordinates', self._coordinates())
        source = self.context_source
        object.__setattr__(self, '_source_objects', (source.source_owner, source.image_source,
            source._context_set_by_stratum, source._listing_index_by_context_id))
        object.__setattr__(self, '_lock', Lock())
        object.__setattr__(self, '_stats', dict(panels_started=0, batches_yielded=0,
            rows_yielded=0, adapter_normalized_peak_bytes=0))
        object.__setattr__(self, 'binding_hash', content_hash(dict(
            schema='v1_operator_native_image_data_plane_v1',
            source_binding_hash=source.actual_source_binding_hash,
            population_hash=population.population_hash,
            source_context_axis_hash=self.context_axis.content_hash,
            source_label_map_hash=self.context_axis.label_map_hash,
            preprocessing_hash=owner.preprocessing_hash,
            target_class_id=self.target_class_id, batch_rows=self.batch_rows,
            source_role=self.source_role, sampling_authority=False,
            dataset_bytes_certified=False, production_authority=False),
            prefix='v1-operator-native-image-plane-'))
        self.assert_source(source)

    def _coordinates(self):
        source = self.context_source
        return (source.actual_source_binding_hash, source.source_identity_hash,
            source.source_context_axis_hash, source.source_label_map_hash,
            source.preprocessing_hash, source.max_batch_rows,
            self.batch_rows, self.target_class_id, self.source_role)

    def assert_source(self, source):
        if source is not self.context_source or self._coordinates() != self._source_coordinates:
            raise ValueError('native image data plane source binding changed')
        actual_objects = (source.source_owner, source.image_source,
            source._context_set_by_stratum, source._listing_index_by_context_id)
        if any(a is not b for a, b in zip(actual_objects, self._source_objects, strict=True)):
            raise ValueError('native image data plane source custody changed')
        identity, _ = _bounded_source_identity(source.image_source)
        if identity != source.source_identity_hash:
            raise ValueError('native image source identity changed')

