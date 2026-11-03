"""Data-only frozen circuit export and exact verified-evaluator reconstruction.

An export preserves an extraction result. It grants no acceptance, population,
terminal-reveal or production-readiness authority. Reload requires the original
verified model/source realization; model weights are referenced, not duplicated.
"""
from collections.abc import Mapping
from dataclasses import dataclass, fields, is_dataclass
import hashlib
import json
from pathlib import Path
import tempfile

import torch

from mechanistic_systems.backends.readout_projection import RuntimeReadoutProjectionSpec
from mechanistic_systems.contracts.content_address import content_hash
from mechanistic_systems.contracts.hard_mask_actions import HardMask
from .modern_v1_controller import V1FrozenAffineCalibration
from .modern_v1_neutral_runtime import FixedAffineBCECalibration, fixed_calibration_bce_rows
from .modern_v1_operator_checkpoint_wire import canonical_json
from .modern_v1_operator_model import OperatorNativeModelEvaluator
from .modern_v1_operator_run_binding import _source_payload, validate_bound_operator_sources
from .modern_v1_operator_terminal_model import OperatorTerminalModelEvaluator
from .modern_v1_receipts import V1TerminalFreezeReceipt

LEGACY_SCHEMA = 'v1_operator_frozen_circuit_export_v1'
SCHEMA = 'v1_operator_frozen_circuit_export_v2'
_FIELDS = frozenset(('schema', 'source', 'query_hash', 'execution_settings', 'mask',
    'frozen_calibration', 'terminal_freeze', 'production_authority', 'export_hash'))


def _plain(value):
    if is_dataclass(value):
        return {f.name:_plain(getattr(value, f.name)) for f in fields(value)}
    if isinstance(value, Mapping):
        return {k:_plain(v) for k,v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(v) for v in value]
    return value


def _fixed(frozen):
    names = ('query_id', 'query_hash', 'target_class_id', 'scale', 'offset',
        'calibration_state_id', 'calibration_source_hash', 'calibration_context_axis_hash',
        'source_label_map_hash', 'model_state_hash')
    calibration = FixedAffineBCECalibration(**{n:getattr(frozen, n) for n in names},
        schema=frozen.calibration_schema)
    if calibration.content_hash != frozen.calibration_hash:
        raise ValueError('export affine state differs from its frozen calibration hash')
    return calibration


def _decode(raw):
    if type(raw) is not bytes:
        raise TypeError('circuit export requires immutable bytes')
    value = json.loads(raw)
    legacy = type(value) is dict and value.get('schema') == LEGACY_SCHEMA
    expected_fields = _FIELDS | {'runtime_state_hash', 'terminal_validation_receipt_hash'} if legacy else _FIELDS
    if (type(value) is not dict or set(value) != expected_fields or value['schema'] not in (SCHEMA, LEGACY_SCHEMA)
            or canonical_json(value) != raw or value['production_authority'] is not False):
        raise ValueError('circuit export canonical schema or authority differs')
    if value['export_hash'] != content_hash({k:v for k,v in value.items() if k != 'export_hash'},
            prefix='v1-operator-circuit-export-'):
        raise ValueError('circuit export content identity differs')
    mask_value = value['mask']
    if set(mask_value) != {'route_axis_hash', 'route_count', 'packed_bits', 'mask_hash'}:
        raise ValueError('circuit export mask schema differs')
    mask = HardMask(mask_value['route_axis_hash'], mask_value['route_count'],
        bytes.fromhex(mask_value['packed_bits']))
    frozen = V1FrozenAffineCalibration(**value['frozen_calibration'])
    freeze = V1TerminalFreezeReceipt(**value['terminal_freeze'])
    calibration = _fixed(frozen)
    if (mask.mask_hash != mask_value['mask_hash'] or mask.mask_hash != frozen.mask_hash
            or mask.mask_hash != freeze.terminal_mask_hash
            or calibration.content_hash != freeze.extraction_calibration_hash
            or value['query_hash'] != calibration.query_hash):
        raise ValueError('circuit export mask/calibration/terminal freeze join differs')
    source_fit = frozen.fit_receipt.get('source_fit_receipt')
    if (not isinstance(source_fit, Mapping) or frozen.fit_receipt.get('source_fit_receipt_hash')
            != content_hash(source_fit, prefix='partition-free-v1-source-calibration-fit-receipt-')):
        raise ValueError('circuit export calibration source fit receipt differs')
    if (not isinstance(value['source'], dict) or set(value['source']) != {
            'source_opening_hash', 'model_checkpoint_binding_hash', 'actual_source_binding_hash',
            'image_content_manifest_hash', 'circuit_biases', 'source_versions'}):
        raise ValueError('circuit export requires the complete verified source binding')
    # Biases are data, never a pickled executable tensor or arbitrary dtype.
    for bias in value['source']['circuit_biases'].values():
        if (type(bias) is not dict or set(bias) != {'dtype', 'values'}
                or bias['dtype'] not in ('torch.float32', 'torch.float64')
                or type(bias['values']) is not list or not bias['values']
                or any(type(x) not in (int, float) for x in bias['values'])):
            raise ValueError('circuit export bias schema differs')
        tensor = torch.tensor(bias['values'], dtype=getattr(torch, bias['dtype'].split('.')[1]))
        if tensor.ndim != 1 or not bool(torch.isfinite(tensor).all()):
            raise ValueError('circuit export biases must be finite vectors')
    return value, mask, calibration


@dataclass(frozen=True, slots=True)
class OperatorCircuitExport:
    raw: bytes

    def __post_init__(self):
        _decode(self.raw)

    @property
    def sha256(self):
        return hashlib.sha256(self.raw).hexdigest()

    @property
    def mask(self):
        return _decode(self.raw)[1]

    @property
    def calibration(self):
        return _decode(self.raw)[2]

    def circuit_bias_tensors(self):
        """Fresh tensors for constructing the verified source before reload."""
        value = _decode(self.raw)[0]
        return {name:torch.tensor(b['values'], dtype=getattr(torch, b['dtype'].split('.')[1]))
            for name,b in value['source']['circuit_biases'].items()}


def export_operator_circuit(*, state, factory):
    """Snapshot an already frozen result without acquisition or calibration."""
    from .modern_v1_operator_factory import OperatorAcquisitionFactory
    from .modern_v1_runtime import V1RuntimeState
    if type(state) is not V1RuntimeState or type(factory) is not OperatorAcquisitionFactory:
        raise TypeError('circuit export requires exact runtime and operator factory')
    state.__post_init__()
    if (state.terminal_freeze is None or state.controller.phase != 'stopped'
            or state.controller.accepted_step_count <= 0):
        raise ValueError('circuit export requires an accepted circuit and terminal freeze')
    model = factory._model
    model.assert_live()
    validate_bound_operator_sources(state.config, model.sources, factory=factory,
        current_mask=state.controller.current_mask)
    installer = factory.native_evaluator.calibration_installer
    mask = state.controller.current_mask
    # Never call export_frozen_calibration here: its uncached path can fit.
    frozen = installer._frozen_exports.get(mask.mask_hash)
    if type(frozen) is not V1FrozenAffineCalibration:
        raise ValueError('circuit export requires the existing accepted frozen calibration')
    installer.verify_frozen_calibration_installed(frozen)
    payload = dict(schema=SCHEMA, source=_source_payload(model.sources),
        query_hash=model.query.content_hash, execution_settings=model._settings,
        mask=dict(route_axis_hash=mask.route_axis_hash, route_count=mask.route_count,
            packed_bits=mask.packed_bits.hex(), mask_hash=mask.mask_hash),
        frozen_calibration=_plain(frozen), terminal_freeze=_plain(state.terminal_freeze),
        production_authority=False)
    payload['export_hash'] = content_hash(payload, prefix='v1-operator-circuit-export-')
    return OperatorCircuitExport(canonical_json(payload))


def write_operator_circuit_export(path, artifact):
    """Publish complete bytes atomically, accepting only identical retries."""
    from .modern_v1_production import _atomic_materialize_exact_file
    if type(artifact) is not OperatorCircuitExport:
        raise TypeError('circuit export write requires the exact artifact')
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.operator-export-', dir=destination.parent) as temporary:
        source = Path(temporary) / 'circuit.json'
        source.write_bytes(artifact.raw)
        _atomic_materialize_exact_file(source, destination, label='operator circuit export')
    return artifact.sha256


def publish_operator_circuit_for_run(*, state, factory, destination, source=None):
    """Derive identical frozen bytes on every restart; verify prior publication.

    A stopped run with no accepted step has no exported pruned circuit. Missing
    files can be reconstructed from the verified checkpoint and accepted fit.
    A present source or destination file must agree exactly and is never replaced.
    """
    if state.controller.accepted_step_count <= 0:
        if any(p is not None and (Path(p).exists() or Path(p).is_symlink())
                for p in (source, destination)):
            raise ValueError('run without an accepted circuit has an unexpected export')
        return None
    artifact = export_operator_circuit(state=state, factory=factory)
    if source is not None and (Path(source).exists() or Path(source).is_symlink()):
        from .modern_v1_production import _require_nonsymlink_regular_file
        _require_nonsymlink_regular_file(Path(source), label='resumed operator circuit export')
        read_operator_circuit_export(source, expected_sha256=artifact.sha256)
    write_operator_circuit_export(destination, artifact)
    return artifact


def read_operator_circuit_export(path, *, expected_sha256):
    raw = Path(path).read_bytes()
    if type(expected_sha256) is not str or hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError('circuit export differs from the expected file SHA256')
    return OperatorCircuitExport(raw)


@dataclass(frozen=True, slots=True)
class ReloadedOperatorCircuit:
    artifact: OperatorCircuitExport
    evaluator: object
    compiled_mask: object

    def __post_init__(self):
        _verify_evaluator(self.artifact, self.evaluator)
        expected = self.evaluator.compile_hard_mask(self.artifact.mask)
        if type(self.compiled_mask) is not type(expected) or self.compiled_mask != expected:
            raise ValueError('circuit reload compiled mask differs from the exported mask')

    def readout(self, batches):
        # Recheck source custody and realization before the first input is opened.
        _verify_evaluator(self.artifact, self.evaluator)
        calibration = self.artifact.calibration
        response, = self.evaluator.readout_responses_from_batches(batches,
            compiled_masks=(self.compiled_mask,), lane_keys=('exported-circuit',),
            readout_projection=RuntimeReadoutProjectionSpec.coordinate(
                calibration.target_class_id, retain_host_values=True))
        return response, fixed_calibration_bce_rows(response.values, response.labels,
            calibration=calibration)


def _verify_evaluator(artifact, evaluator):
    if type(artifact) is not OperatorCircuitExport or type(evaluator) not in (
            OperatorNativeModelEvaluator, OperatorTerminalModelEvaluator):
        raise TypeError('circuit reload requires exact export and verified operator evaluator')
    evaluator.assert_live()
    value = _decode(artifact.raw)[0]
    if (content_hash(value['source']) != content_hash(_source_payload(evaluator.sources))
            or value['query_hash'] != evaluator.query.content_hash
            or tuple(value['execution_settings']) != evaluator._settings):
        raise ValueError('circuit reload source/model/bias/query/numerical realization differs')


def reload_operator_circuit(artifact, *, evaluator):
    """Compile the exported mask on a fresh verified source, without fitting."""
    _verify_evaluator(artifact, evaluator)
    return ReloadedOperatorCircuit(artifact, evaluator, evaluator.compile_hard_mask(artifact.mask))
