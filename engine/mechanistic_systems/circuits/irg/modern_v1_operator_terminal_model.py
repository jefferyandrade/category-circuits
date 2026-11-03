"""Terminal contexts over the same verified functional model and mask ABI.

This is a readout adapter, not a terminal reveal oracle. The retained terminal
validator and frozen population/family request retain reveal authority.
"""
from dataclasses import dataclass, field
from pathlib import Path
import hashlib

from mechanistic_systems.contracts.content_address import content_hash
from .modern_v1_operator_model import OperatorNativeModelEvaluator
from .modern_v1_operator_terminal_data import OperatorTerminalImageDataPlane


@dataclass(frozen=True, slots=True)
class OperatorTerminalModelEvaluator(OperatorNativeModelEvaluator):
    _terminal_content_manifest: str = field(init=False, repr=False, compare=False)
    _terminal_source_sha256: str = field(init=False, repr=False, compare=False)

    def __post_init__(self):
        if type(self.data_plane) is not OperatorTerminalImageDataPlane:
            raise TypeError('terminal operator requires the exact bounded image data plane')
        terminal = self.data_plane.context_source
        images = terminal.image_source
        if images.stats()['decode_requests_submitted']:
            raise ValueError('terminal model binding must precede terminal image outcomes')
        if self.data_plane.source_role != 'terminal' or self.data_plane.context_axis.split not in ('val', 'validation'):
            raise ValueError('terminal operator requires a declared terminal validation source')
        optimization = self.sources.context_source
        if (terminal is optimization or images is optimization.image_source
                or terminal.source_context_axis_hash == optimization.source_context_axis_hash):
            raise ValueError('terminal operator cannot reuse the optimization source')
        if (terminal.source_label_map_hash != optimization.source_label_map_hash
                or terminal.preprocessing_hash != optimization.preprocessing_hash):
            raise ValueError('terminal labels/preprocessing differ from the model source contract')
        optimization_ids = set(optimization.source_owner.target_context_ids) | set(optimization.source_owner.complement_context_ids)
        if optimization_ids.intersection(self.data_plane.expected_labels):
            raise ValueError('terminal and optimization context IDs overlap')
        optimization_paths = {str(Path(p).resolve()) for paths in optimization.image_source._paths.values() for p in paths}
        terminal_paths = {str(Path(p).resolve()) for paths in images._paths.values() for p in paths}
        if optimization_paths.intersection(terminal_paths):
            raise ValueError('terminal and optimization image paths overlap')
        object.__setattr__(self, '_terminal_content_manifest', images.freeze_content_manifest())
        object.__setattr__(self, '_terminal_source_sha256', hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
        OperatorNativeModelEvaluator.__post_init__(self)

    def _validate_data_plane(self):
        if type(self.data_plane) is not OperatorTerminalImageDataPlane:
            raise TypeError('terminal operator requires its exact terminal data plane')

    def assert_live(self):
        self._assert_model_and_query()
        source = self.data_plane.context_source
        self.data_plane.assert_source(source)
        if (self.data_plane.source_role != 'terminal'
                or source.image_source.content_manifest_hash != self._terminal_content_manifest):
            raise ValueError('terminal operator source identity changed')

    @property
    def binding_hash(self):
        # Retain the native mask/forward ABI while separating role identity.
        return content_hash(dict(schema='v1_operator_terminal_model_v1',
            functional_model_binding_hash=OperatorNativeModelEvaluator.binding_hash.fget(self),
            terminal_content_manifest_hash=self._terminal_content_manifest,
            terminal_source_sha256=self._terminal_source_sha256,
            reveal_authority=False, population_independence_certified=False),
            prefix='v1-operator-terminal-model-')
