"""Source-bound JPEG dataset and operator role identities, before acquisition.

These describe the declared finite population and execution roles. They confer
no ImageNet completeness, independent sampling, physical or commit authority.
"""
from dataclasses import asdict, dataclass
import hashlib

from mechanistic_systems.contracts.content_address import content_hash
from mechanistic_systems.measurement.contexts import ContextAxis
from .modern_v1_operator_checkpoint_wire import canonical_json
from .modern_v1_neutral_runtime import build_class_balanced_sampling_population
from .modern_v1_physical_metric import BALANCED_MEASURE
from .tractable_v1.genuine_issuer_bridge import BoundedImageContextSource, _bounded_source_identity
from .tractable_v1.image_stream import BoundedImageSource


@dataclass(frozen=True, slots=True)
class OperatorDatasetRegistration:
    source_context_axis_hash: str
    source_label_map_hash: str
    preprocessing_hash: str
    realized_preprocessing_hash: str
    population_hash: str
    sampling_law_hash: str
    target_class_id: int
    target_count: int
    complement_count: int
    target_membership_hash: str
    complement_membership_hash: str
    context_content_sha256: str

    def __post_init__(self):
        for name in ('source_context_axis_hash', 'source_label_map_hash', 'preprocessing_hash',
                'realized_preprocessing_hash', 'population_hash', 'sampling_law_hash',
                'target_membership_hash', 'complement_membership_hash'):
            if type(getattr(self,name)) is not str or not getattr(self,name).strip():
                raise ValueError(f'dataset registration requires {name}')
        for name in ('target_count', 'complement_count'):
            if type(getattr(self,name)) is not int or getattr(self,name) <= 0:
                raise ValueError('dataset registration requires positive stratum counts')
        if type(self.target_class_id) is not int or self.target_class_id < 0:
            raise ValueError('dataset registration requires a nonnegative target class')
        digest = self.context_content_sha256
        if type(digest) is not str or len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest):
            raise ValueError('dataset content digest must be SHA256')

    def to_json(self):
        return dict(schema='v1_operator_declared_jpeg_dataset_v1', **asdict(self),
            population_measure=BALANCED_MEASURE, context_content_encoding='canonical_json_rows_with_newline_v1',
            bounded_listing_complete=True, full_imagenet_population_qualified=False,
            sampling_authority=False, production_authority=False)

    @property
    def dataset_hash(self):
        return content_hash(self.to_json(), prefix='v1-operator-jpeg-dataset-')

    def source_role_payload(self):
        return dict(schema='v1_operator_population_roles_v1', dataset_hash=self.dataset_hash,
            population_hash=self.population_hash, population_measure=BALANCED_MEASURE,
            source_role='optimization_train_only',
            discovery_role='raw_target_logit_operator_chart_and_binary_action_freeze',
            physical_validation_role='frozen_action_parent_and_original_dense_finite_replay',
            calibration_role='mask_specific_affine_fit_at_registered_calibration_epochs',
            oracle_role='known_K_plus_conditionally_sampled_residual_U_minus_K_paired_BCE',
            role_order='D_freeze_action_C_then_registered_K_O_epochs_decide_commit_or_refuse_relinearize',
            static_role_partition_authority=False, incidental_identity_overlap_permitted=True,
            roles_nontransferable=True, bootstrap_population_authority=False,
            legacy_KKT_FQR_authority=False, sampled_doses_certify_entire_path=False,
            terminal_population_included=False, sampling_authority=False, commit_authority=False)

    @property
    def source_role_hash(self):
        return content_hash(self.source_role_payload(), prefix='v1-operator-population-roles-')


def register_operator_dataset(*, context_source, context_axis, target_class_id):
    """Hash file metadata in registered context order, without decoded outcomes.

    This registration deliberately excludes memory/batch settings and incidental
    source-listing permutations. Numerical execution remains separately bound.
    The existing content manifest hashes actual bytes; streaming its verified
    records here adds only constant-size hashing state, not another image cache.
    """
    if type(context_source) is not BoundedImageContextSource or type(context_axis) is not ContextAxis:
        raise TypeError('operator dataset requires exact bounded source and context axis')
    images, owner = context_source.image_source, context_source.source_owner
    if type(images) is not BoundedImageSource:
        raise TypeError('operator dataset requires its exact bounded image source')
    if images.stats()['decode_requests_submitted'] or images.stats()['closed']:
        raise ValueError('operator dataset registration requires an unopened live image source')
    if context_axis.split != 'train':
        raise ValueError('operator dataset excludes terminal population')
    owner.__post_init__()
    identity_hash, identity = _bounded_source_identity(images)
    population = build_class_balanced_sampling_population(context_axis,
        target_class_id=target_class_id, source_role='optimization')
    if (identity_hash != owner.bounded_image_source_identity_hash
            or owner.source_context_axis_hash != context_axis.content_hash
            or owner.source_label_map_hash != context_axis.label_map_hash
            or owner.preprocessing_hash != context_axis.preprocessing.content_hash
            or owner.source_population_hash != population.population_hash
            or owner.sampling_law_hash != population.sampling_law_hash
            or owner.target_context_ids != population.on_target_context_ids
            or owner.complement_context_ids != population.off_target_context_ids):
        raise ValueError('operator dataset source differs from registered axis/population')
    digest = hashlib.sha256()
    for stratum in ('target','complement'):
        contexts = owner.context_ids_by_stratum[stratum]
        indices = owner.listing_indices_by_stratum[stratum]
        # A unique in-range crosswalk with equal cardinality covers the listing.
        if len(contexts) != len(images._paths[stratum]):
            raise ValueError('operator dataset must cover the complete bounded listing')
        if len(indices) != len(contexts):
            raise ValueError('operator dataset crosswalk lengths differ')
    images.freeze_content_manifest()
    index_iters = {s:iter(owner.listing_indices_by_stratum[s]) for s in ('target','complement')}
    for record in context_axis.records:
        stratum = 'target' if record.label == target_class_id else 'complement'
        size, file_hash = images.frozen_content_record(stratum,next(index_iters[stratum]))
        if size != record.source_size_bytes:
            raise ValueError('operator dataset file size differs from the registered context record')
        digest.update(canonical_json((stratum,record.context_id,size,file_hash)))
        digest.update(b'\n')
    return OperatorDatasetRegistration(source_context_axis_hash=context_axis.content_hash,
        source_label_map_hash=context_axis.label_map_hash,
        preprocessing_hash=context_axis.preprocessing.content_hash,
        realized_preprocessing_hash=content_hash(identity['preprocessing'],
            prefix='tractable-v1-bounded-image-source-preprocessing-'),
        population_hash=population.population_hash, sampling_law_hash=population.sampling_law_hash,
        target_class_id=target_class_id, target_count=len(owner.target_context_ids),
        complement_count=len(owner.complement_context_ids),
        target_membership_hash=content_hash(owner.target_context_ids,
            prefix='partition-free-v1-alexnet-fq-target-membership-'),
        complement_membership_hash=content_hash(owner.complement_context_ids,
            prefix='partition-free-v1-alexnet-fq-complement-membership-'),
        context_content_sha256=digest.hexdigest())


def validate_operator_fq_dataset(*, dataset, fq_registration):
    from .modern_v1_alexnet_fq_registration import V1AlexNetFQAcquisitionRegistration
    if type(dataset) is not OperatorDatasetRegistration or type(fq_registration) is not V1AlexNetFQAcquisitionRegistration:
        raise TypeError('operator source join requires exact dataset and F/Q registrations')
    dataset.__post_init__()
    fq_registration.assert_integrity()
    expected = dict(train_dataset_identity_hash=dataset.dataset_hash,
        train_split_identity_hash=dataset.population_hash, source_role_registration_hash=dataset.source_role_hash,
        source_context_axis_hash=dataset.source_context_axis_hash,
        source_label_map_hash=dataset.source_label_map_hash, preprocessing_hash=dataset.preprocessing_hash,
        target_population_count=dataset.target_count, complement_population_count=dataset.complement_count,
        target_population_membership_hash=dataset.target_membership_hash,
        complement_population_membership_hash=dataset.complement_membership_hash)
    for name,value in expected.items():
        if getattr(fq_registration,name) != value:
            raise ValueError(f'operator F/Q dataset join differs: {name}')
