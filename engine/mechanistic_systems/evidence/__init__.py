"""Statistical kernels for the exploratory and shadow evidence foundations.

The legacy v1/continuous-v2 estimators use planned-look Student-t,
empirical-Bernstein, and empirical-Bayes moderation.  They remain exploratory:
their intervals are not the claim-bearing primitive under adaptive allocation
or optional stopping.

The shadow ``evidence_foundation_glocal_v3`` path instead owns bounded
finite-population sampling-without-replacement confidence sequences,
predictable proposal registration, campaign-global alpha/distortion/resource
ledgers, and one-time terminal-audit contracts.  Empirical-Bayes quantities may
guide allocation but never narrow a glocal-v3 safety bound.  This package
deliberately imports neither measurement nor circuit runtimes so both can
consume the contracts without dependency cycles.
"""

from .bounds import (
    EVIDENCE_FOUNDATION_V1,
    EvidenceInterval,
    empirical_bernstein_interval,
    eb_confidence_sequence,
    staged_alpha,
    student_t_interval,
)
from .campaign import (
    CampaignEvidenceState,
    DecisionEvidenceReceipt,
    ResourceLedger,
    ResourceUsage,
)
from .deployment import (
    GLOCAL_DEPLOYMENT_CERTIFICATE_SCHEMA,
    GlocalAcceptedTransition,
    GlocalDeploymentAuthorityHandle,
    GlocalDeploymentMaskCertificate,
    build_glocal_deployment_mask_certificate,
    build_live_glocal_deployment_authority,
)
from .finite_population import (
    BoundedWoRBettingCS,
    FinitePopulationCSSnapshot,
    StratifiedCSSnapshot,
    StratifiedPairedEProcess,
    bounded_wor_betting_confidence_sequence,
)
from .glocal_contracts import (
    EVIDENCE_FOUNDATION_GLOCAL_V3,
    GLOCAL_EXTERNAL_ENTROPY_SOURCE_V1,
    GLOCAL_IID_FIXED_N_RANDOMIZATION_METHOD_V1,
    GLOCAL_IID_RUNTIME_RANDOMIZATION_METHOD_V2,
    GLOCAL_SEALED_PERMUTATION_ALGORITHM_V1,
    GLOCAL_SYSTEM_ENTROPY_SOURCE_V1,
    IIDDrawRecord,
    IIDSamplingReceipt,
    IIDSupportDistribution,
    JOINT_RISK_QUERY_ID,
    CalibrationAlgorithmSpec,
    CalibrationFitState,
    EvidenceRole,
    ObservationSetRef,
    PredictableAllocationLookReceipt,
    PredictableProposalRegistration,
    PredictionRuleSpec,
    RiskEstimandSpec,
    SamplingPermutationReceipt,
    SamplingDesignSpec,
    SamplingStratumSpec,
    TerminalAuditDesignKind,
    TerminalAuditEvidenceReceipt,
    TerminalAuditPrecommitment,
    TerminalParityEstimandSpec,
    TerminalParityCertificate,
    ValidationLock,
    binary_brier_loss,
    class_balanced_observation_weights,
    normalized_multiclass_brier_loss,
    paired_brier_increment,
    iid_physical_executor_identity_hash,
    iid_physical_provider_identity_hash,
    validate_role_separation,
)
from .ledger import AlphaLedgerEntry, AlphaWealthLedger
from .moderation import (
    VarianceModerationPrior,
    fit_variance_prior,
    moderated_variances,
)
from .roles_exact import ExactRoleEvidence, exact_simultaneous_roles

__all__ = [
    "EVIDENCE_FOUNDATION_GLOCAL_V3",
    "EVIDENCE_FOUNDATION_V1",
    "GLOCAL_EXTERNAL_ENTROPY_SOURCE_V1",
    "GLOCAL_SEALED_PERMUTATION_ALGORITHM_V1",
    "GLOCAL_SYSTEM_ENTROPY_SOURCE_V1",
    "JOINT_RISK_QUERY_ID",
    "AlphaLedgerEntry",
    "AlphaWealthLedger",
    "BoundedWoRBettingCS",
    "CalibrationAlgorithmSpec",
    "CalibrationFitState",
    "CampaignEvidenceState",
    "DecisionEvidenceReceipt",
    "EvidenceRole",
    "EvidenceInterval",
    "ExactRoleEvidence",
    "FinitePopulationCSSnapshot",
    "GLOCAL_DEPLOYMENT_CERTIFICATE_SCHEMA",
    "GlocalAcceptedTransition",
    "GlocalDeploymentAuthorityHandle",
    "GlocalDeploymentMaskCertificate",
    "GLOCAL_IID_FIXED_N_RANDOMIZATION_METHOD_V1",
    "GLOCAL_IID_RUNTIME_RANDOMIZATION_METHOD_V2",
    "IIDDrawRecord",
    "IIDSamplingReceipt",
    "IIDSupportDistribution",
    "ObservationSetRef",
    "PredictableAllocationLookReceipt",
    "PredictableProposalRegistration",
    "PredictionRuleSpec",
    "ResourceLedger",
    "ResourceUsage",
    "RiskEstimandSpec",
    "SamplingPermutationReceipt",
    "SamplingDesignSpec",
    "SamplingStratumSpec",
    "StratifiedCSSnapshot",
    "StratifiedPairedEProcess",
    "TerminalAuditDesignKind",
    "TerminalAuditEvidenceReceipt",
    "TerminalAuditPrecommitment",
    "TerminalParityEstimandSpec",
    "TerminalParityCertificate",
    "ValidationLock",
    "VarianceModerationPrior",
    "binary_brier_loss",
    "build_glocal_deployment_mask_certificate",
    "build_live_glocal_deployment_authority",
    "bounded_wor_betting_confidence_sequence",
    "class_balanced_observation_weights",
    "eb_confidence_sequence",
    "empirical_bernstein_interval",
    "exact_simultaneous_roles",
    "fit_variance_prior",
    "iid_physical_executor_identity_hash",
    "iid_physical_provider_identity_hash",
    "moderated_variances",
    "normalized_multiclass_brier_loss",
    "paired_brier_increment",
    "staged_alpha",
    "student_t_interval",
    "validate_role_separation",
]
