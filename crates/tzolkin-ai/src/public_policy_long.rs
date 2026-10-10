//! Closed long-run v2 policy production. A matching count1 v1 cohort is the
//! one-way entry; later steps accept only this version's immediate parent.
//! The all-seat actor-SUM math is shared with v1. This is neither a runner nor
//! a loader: stored hashes establish consistency, not producer authentication.
use std::collections::BTreeSet;

use serde::{Deserialize, Serialize};

use crate::dataset::{DatasetSplit, split_for_family};
use crate::features::PUBLIC_FEATURE_SCHEMA;
use crate::public_model::{self, PublicPolicyModel};
use crate::public_policy_episode::TARGET_CONTRACT;
use crate::public_policy_likelihood::{CORRECTION_VERSION, LIKELIHOOD_VERSION};
use crate::public_policy_pullback::PULLBACK_VERSION;
use crate::public_policy_repeat::propose_cohort;
use crate::public_policy_update::{
    Deltas, MAX_ALL_LOG_DELTA, NORMALIZER_VERSION, NumericReport, OneStepConfig,
    UpdatedPublicRlPolicy, hash,
};
use crate::public_rl_policy_cohort::{COHORT_CONTRACT, ValidatedRlCohort};
use crate::public_stochastic::{RL_SAMPLING_VERSION, RNG_VERSION};
use crate::public_stochastic_native::numerical_target;
use crate::replay::{RULES_BASELINE, RULES_VERSION, SeatPolicy, catalog_hash};

pub const ARTIFACT_SCHEMA: &str = "tzolkin-public-rl-long-v2";
pub const UPDATE_VERSION: &str = "public-current-parent-long-ascent-f64-to-f32-v2";
pub const TASK: &str = "policyOnlyRlLongV2";
pub const POLICY_VERSION: &str = "learned-public-rl-long-v2";
pub const COHORT_VERSION: &str = "four-train-games-all-seat-rl-long-actor-sum-v2";
pub const SAMPLING_VERSION: &str = "public-stochastic-rl-long-uniform-tick53-v2";
// An implementation safety ceiling, not an approved experiment budget.
pub const MAX_UPDATES: u64 = 1_000_000;

/// Explicit per-lineage limits. No default silently selects a long experiment.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct LongRunLimits {
    max_updates: u64,
    max_families: usize,
}
impl LongRunLimits {
    pub fn new(max_updates: u64, max_families: usize) -> Result<Self, String> {
        let updates = usize::try_from(max_updates).map_err(|_| "Long update limit overflow")?;
        let ceiling = updates
            .checked_mul(4)
            .and_then(|n| n.checked_add(crate::policy_dataset::MAX_FILES))
            .ok_or("Long family limit overflow")?;
        if !(2..=MAX_UPDATES).contains(&max_updates) || max_families == 0 || max_families > ceiling
        {
            return Err("Invalid explicit long-run update/family limits".into());
        }
        Ok(Self {
            max_updates,
            max_families,
        })
    }
    pub fn max_updates(self) -> u64 {
        self.max_updates
    }
    pub fn max_families(self) -> usize {
        self.max_families
    }
    fn next_count(self, parent: u64) -> Result<u64, String> {
        if parent == 0 || parent >= self.max_updates {
            return Err("Long parent has reached the configured update limit".into());
        }
        parent
            .checked_add(1)
            .ok_or_else(|| "Long update count overflow".into())
    }
}

/// Closed production inputs, never an arbitrary parameter or stored DTO route.
#[derive(Clone, Copy)]
pub enum LongRlParent<'a> {
    Count1(&'a UpdatedPublicRlPolicy),
    Long(&'a LongRlPolicy),
}
impl<'a> LongRlParent<'a> {
    pub(crate) fn checksum(self) -> &'a str {
        match self {
            Self::Count1(p) => p.artifact().checksum(),
            Self::Long(p) => p.artifact().checksum(),
        }
    }
    pub(crate) fn count(self) -> u64 {
        match self {
            Self::Count1(p) => p.artifact().update_count(),
            Self::Long(p) => p.artifact().update_count(),
        }
    }
    fn model(self) -> &'a PublicPolicyModel {
        match self {
            Self::Count1(p) => p.model(),
            Self::Long(p) => p.model(),
        }
    }
    fn root_init(self) -> &'a str {
        match self {
            Self::Count1(p) => p.artifact().report().parent_init_checksum(),
            Self::Long(p) => &p.artifact.report.root_init_checksum,
        }
    }
    fn root_count1(self) -> &'a str {
        match self {
            Self::Count1(p) => p.artifact().checksum(),
            Self::Long(p) => &p.artifact.report.root_count1_checksum,
        }
    }
    fn bc_source(self) -> &'a SeatPolicy {
        match self {
            Self::Count1(p) => p.artifact().report().bc_source(),
            Self::Long(p) => &p.artifact.report.bc_source,
        }
    }
    fn family_count(self) -> usize {
        match self {
            Self::Count1(p) => p.artifact().report().family_closure().len(),
            Self::Long(p) => p.families.len(),
        }
    }
    fn contains_family(self, family: &str) -> bool {
        match self {
            Self::Count1(p) => p
                .artifact()
                .report()
                .family_closure()
                .iter()
                .any(|f| f == family),
            Self::Long(p) => p.families.contains(family),
        }
    }
    fn families(self) -> BTreeSet<String> {
        match self {
            Self::Count1(p) => p
                .artifact()
                .report()
                .family_closure()
                .iter()
                .cloned()
                .collect(),
            Self::Long(p) => p.families.clone(),
        }
    }
    fn lineage(self) -> Result<String, String> {
        match self {
            Self::Count1(p) => hash(
                b"tzolkin-public-rl-long-root-lineage-v2\0",
                &(
                    p.artifact().checksum(),
                    p.artifact().report().family_closure(),
                ),
            ),
            Self::Long(p) => Ok(p.artifact.report.lineage_checksum.clone()),
        }
    }
    fn source(self) -> (&'static str, &'static str) {
        match self {
            Self::Count1(_) => (COHORT_CONTRACT, RL_SAMPLING_VERSION),
            Self::Long(_) => (COHORT_VERSION, SAMPLING_VERSION),
        }
    }
    pub(crate) fn validate(self, limits: LongRunLimits) -> Result<(), String> {
        match self {
            Self::Count1(p) => {
                p.validate_for_inference()?;
                let families = p.artifact().report().family_closure();
                if families.is_empty()
                    || families.len() > crate::policy_dataset::MAX_FILES + 4
                    || families.windows(2).any(|w| w[0] >= w[1])
                    || families
                        .iter()
                        .any(|f| !crate::public_policy_update::checkpoint_digest(f))
                {
                    return Err("Invalid Count1 long bootstrap closure".into());
                }
                for f in families {
                    if split_for_family(f)? == DatasetSplit::Test {
                        return Err("Count1 long bootstrap includes Test".into());
                    }
                }
                Ok(())
            }
            Self::Long(p) => {
                p.validate_for_sampling()?;
                if p.limits() != limits {
                    return Err("Long lineage limits cannot change".into());
                }
                Ok(())
            }
        }
    }
}

#[derive(Debug, Serialize, PartialEq)]
#[serde(rename_all = "camelCase")]
struct Contract {
    schema: &'static str,
    task: &'static str,
    policy_version: &'static str,
    update_version: &'static str,
    normalizer_version: &'static str,
    likelihood_version: &'static str,
    correction_version: &'static str,
    pullback_version: &'static str,
    target_contract: &'static str,
    rng_version: &'static str,
    model_version: &'static str,
    input_contract: &'static str,
    feature_schema: u32,
    parameter_count: usize,
    rules_version: u32,
    rules_baseline: &'static str,
    catalog_hash: String,
    numerical_target: String,
    backend: &'static str,
    max_all_log_delta: f64,
    limits: LongRunLimits,
}
fn contract(limits: LongRunLimits) -> Contract {
    Contract {
        schema: ARTIFACT_SCHEMA,
        task: TASK,
        policy_version: POLICY_VERSION,
        update_version: UPDATE_VERSION,
        normalizer_version: NORMALIZER_VERSION,
        likelihood_version: LIKELIHOOD_VERSION,
        correction_version: CORRECTION_VERSION,
        pullback_version: PULLBACK_VERSION,
        target_contract: TARGET_CONTRACT,
        rng_version: RNG_VERSION,
        model_version: public_model::MODEL_VERSION,
        input_contract: public_model::INPUT_CONTRACT,
        feature_schema: PUBLIC_FEATURE_SCHEMA,
        parameter_count: public_model::PARAMETER_COUNT,
        rules_version: RULES_VERSION,
        rules_baseline: RULES_BASELINE,
        catalog_hash: catalog_hash(),
        numerical_target: numerical_target(),
        backend: "scalar",
        max_all_log_delta: MAX_ALL_LOG_DELTA,
        limits,
    }
}

/// A constant-size per-batch receipt: no cumulative family list or parent model.
/// NoChange reports the attempted families but keeps the parent's lineage.
#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct LongStepReport {
    contract: Contract,
    parent_artifact_checksum: String,
    parent_update_count: u64,
    attempted_update_count: u64,
    root_init_checksum: String,
    root_count1_checksum: String,
    bc_source: SeatPolicy,
    cohort_contract: &'static str,
    sampling_version: &'static str,
    cohort_plan_checksum: String,
    cohort_receipt_checksum: String,
    episode_checksums: [String; 4],
    rollout_families: [String; 4],
    parent_lineage_checksum: String,
    lineage_checksum: String,
    family_count: usize,
    config: OneStepConfig,
    counts: [usize; 3],
    deltas: Deltas,
    numeric: NumericReport,
    checksum: String,
}
impl LongStepReport {
    pub fn checksum(&self) -> &str {
        &self.checksum
    }
    pub fn parent_checksum(&self) -> &str {
        &self.parent_artifact_checksum
    }
    pub fn parent_update_count(&self) -> u64 {
        self.parent_update_count
    }
    pub fn attempted_update_count(&self) -> u64 {
        self.attempted_update_count
    }
    pub fn root_init_checksum(&self) -> &str {
        &self.root_init_checksum
    }
    pub fn root_count1_checksum(&self) -> &str {
        &self.root_count1_checksum
    }
    pub fn lineage_checksum(&self) -> &str {
        &self.lineage_checksum
    }
    pub fn family_count(&self) -> usize {
        self.family_count
    }
    pub fn rollout_families(&self) -> &[String; 4] {
        &self.rollout_families
    }
    pub fn counts(&self) -> [usize; 3] {
        self.counts
    }
    pub fn changed_parameters(&self) -> usize {
        self.numeric.changed_parameters
    }
    pub fn config(&self) -> &OneStepConfig {
        &self.config
    }
    pub(crate) fn bc_source(&self) -> &SeatPolicy {
        &self.bc_source
    }
    pub(crate) fn plan_checksum(&self) -> &str {
        &self.cohort_plan_checksum
    }
    pub(crate) fn parent_lineage(&self) -> &str {
        &self.parent_lineage_checksum
    }
    fn expected_checksum(&self) -> Result<String, String> {
        hash(
            b"tzolkin-public-rl-long-report-v2\0",
            &(
                &self.contract,
                (
                    &self.parent_artifact_checksum,
                    self.parent_update_count,
                    self.attempted_update_count,
                ),
                (
                    &self.root_init_checksum,
                    &self.root_count1_checksum,
                    &self.bc_source,
                ),
                (
                    self.cohort_contract,
                    self.sampling_version,
                    &self.cohort_plan_checksum,
                    &self.cohort_receipt_checksum,
                ),
                (&self.episode_checksums, &self.rollout_families),
                (
                    &self.parent_lineage_checksum,
                    &self.lineage_checksum,
                    self.family_count,
                ),
                (self.config, self.counts, &self.deltas, &self.numeric),
            ),
        )
    }
}

/// Serialize-only numeric artifact. Family membership is held by the sealed
/// owner, reconstructed only through controlled production in this unit.
#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct LongRlArtifact {
    update_count: u64,
    report: LongStepReport,
    model: PublicPolicyModel,
    checksum: String,
}
impl LongRlArtifact {
    pub fn checksum(&self) -> &str {
        &self.checksum
    }
    pub fn update_count(&self) -> u64 {
        self.update_count
    }
    pub fn report(&self) -> &LongStepReport {
        &self.report
    }
}
pub struct LongRlPolicy {
    artifact: LongRlArtifact,
    families: BTreeSet<String>,
}
impl LongRlPolicy {
    pub fn artifact(&self) -> &LongRlArtifact {
        &self.artifact
    }
    pub fn limits(&self) -> LongRunLimits {
        self.artifact.report.contract.limits
    }
    pub(crate) fn model(&self) -> &PublicPolicyModel {
        &self.artifact.model
    }
    pub(crate) fn contains_family(&self, family: &str) -> bool {
        self.families.contains(family)
    }
    pub(crate) fn family_set(&self) -> &BTreeSet<String> {
        &self.families
    }
    pub(crate) fn validate_for_sampling(&self) -> Result<(), String> {
        let a = &self.artifact;
        let r = &a.report;
        let limits = LongRunLimits::new(self.limits().max_updates, self.limits().max_families)?;
        let source = if r.parent_update_count == 1 {
            (COHORT_CONTRACT, RL_SAMPLING_VERSION)
        } else {
            (COHORT_VERSION, SAMPLING_VERSION)
        };
        if r.contract != contract(limits)
            || a.update_count < 2
            || a.update_count != limits.next_count(r.parent_update_count)?
            || r.attempted_update_count != a.update_count
            || r.numeric.changed_parameters == 0
            || (r.cohort_contract, r.sampling_version) != source
            || r.family_count != self.families.len()
            || r.family_count > limits.max_families
        {
            return Err("Incompatible sealed long-run owner".into());
        }
        r.config.validate()?;
        r.bc_source.validate()?;
        if !matches!(&r.bc_source, SeatPolicy::PublicLearned { inference_backend, .. } if inference_backend == "scalar")
            || [
                &r.parent_artifact_checksum,
                &r.root_init_checksum,
                &r.root_count1_checksum,
                &r.cohort_plan_checksum,
                &r.cohort_receipt_checksum,
                &r.parent_lineage_checksum,
                &r.lineage_checksum,
                &r.checksum,
                &a.checksum,
            ]
            .into_iter()
            .chain(r.episode_checksums.iter())
            .any(|s| !crate::public_policy_update::checkpoint_digest(s))
            || (r.parent_update_count == 1 && r.parent_artifact_checksum != r.root_count1_checksum)
            || r.lineage_checksum
                != extend_lineage(&r.parent_lineage_checksum, &r.rollout_families)?
        {
            return Err("Invalid Long source/root/lineage metadata".into());
        }
        for (i, f) in r.rollout_families.iter().enumerate() {
            if !crate::public_policy_update::checkpoint_digest(f)
                || split_for_family(f)? != DatasetSplit::Train
                || r.rollout_families[..i].contains(f)
                || !self.families.contains(f)
            {
                return Err("Invalid Long latest whole-four membership".into());
            }
        }
        self.model().validate()?;
        if r.checksum != r.expected_checksum()?
            || a.checksum != artifact_checksum(a.update_count, r, &a.model)?
        {
            return Err("Long owner checksum mismatch".into());
        }
        Ok(())
    }
}
pub enum LongStepOutcome {
    Updated(LongRlPolicy),
    NoChange(LongStepReport),
}

/// Parent/source/freshness/cap guards run before NN work. The same shared ordered
/// member→actor→own-decision ascent is used by v1; no clipping or optimizer is added.
pub fn ascent_long(
    parent: LongRlParent<'_>,
    cohort: &ValidatedRlCohort,
    config: &OneStepConfig,
    limits: LongRunLimits,
) -> Result<LongStepOutcome, String> {
    let prepared = prepare_long_step(parent, cohort, config, limits)?;
    match prepared {
        PreparedLongStep::NoChange(report) => Ok(LongStepOutcome::NoChange(report)),
        updated => Ok(updated.commit(parent.families())),
    }
}

// Only sealed parents/cohorts create this object. Every fallible calculation is
// complete before a session moves its owned family set into commit.
pub(crate) enum PreparedLongStep {
    Updated(LongRlArtifact),
    NoChange(LongStepReport),
}
impl PreparedLongStep {
    pub(crate) fn artifact(&self) -> Option<&LongRlArtifact> {
        match self {
            Self::Updated(a) => Some(a),
            Self::NoChange(_) => None,
        }
    }
    pub(crate) fn report(&self) -> &LongStepReport {
        match self {
            Self::Updated(a) => &a.report,
            Self::NoChange(r) => r,
        }
    }
    pub(crate) fn commit(self, mut families: BTreeSet<String>) -> LongStepOutcome {
        match self {
            Self::NoChange(r) => LongStepOutcome::NoChange(r),
            Self::Updated(artifact) => {
                families.extend(artifact.report.rollout_families.iter().cloned());
                LongStepOutcome::Updated(LongRlPolicy { artifact, families })
            }
        }
    }
}
pub(crate) fn take_families(owner: LongRlPolicy) -> BTreeSet<String> {
    owner.families
}
pub(crate) fn prepare_long_step(
    parent: LongRlParent<'_>,
    cohort: &ValidatedRlCohort,
    config: &OneStepConfig,
    limits: LongRunLimits,
) -> Result<PreparedLongStep, String> {
    config.validate()?;
    LongRunLimits::new(limits.max_updates, limits.max_families)?;
    parent.validate(limits)?;
    let count = limits.next_count(parent.count())?;
    let source = parent.source();
    if parent.checksum() != cohort.plan().parent_artifact_checksum()
        || parent.count() != cohort.plan().parent_update_count()
        || cohort.plan().numerical_target() != numerical_target()
        || cohort.plan().cohort_contract() != source.0
        || cohort.total_candidate_rows() > config.max_candidate_rows()
        || cohort.episodes().iter().any(|e| {
            e.parent_artifact_checksum() != parent.checksum()
                || e.parent_update_count() != parent.count()
                || e.sampling_version() != source.1
        })
    {
        return Err("Long update parent/cohort/source/target/budget mismatch".into());
    }
    let families = cohort.plan().rollout_families();
    let size = parent
        .family_count()
        .checked_add(4)
        .ok_or("Long family count overflow")?;
    if size > limits.max_families {
        return Err("Long family limit exceeded before forward".into());
    }
    for (i, f) in families.iter().enumerate() {
        if split_for_family(f)? != DatasetSplit::Train
            || parent.contains_family(f)
            || families[..i].contains(f)
        {
            return Err("Long update requires four fresh distinct Train families".into());
        }
    }
    let parent_lineage = parent.lineage()?;
    let (proposal, counts, deltas) = propose_cohort(parent.model(), cohort, config)?;
    let changed = proposal.numeric.changed_parameters != 0;
    let lineage = if changed {
        extend_lineage(&parent_lineage, families)?
    } else {
        parent_lineage.clone()
    };
    let mut report = LongStepReport {
        contract: contract(limits),
        parent_artifact_checksum: parent.checksum().into(),
        parent_update_count: parent.count(),
        attempted_update_count: count,
        root_init_checksum: parent.root_init().into(),
        root_count1_checksum: parent.root_count1().into(),
        bc_source: parent.bc_source().clone(),
        cohort_contract: source.0,
        sampling_version: source.1,
        cohort_plan_checksum: cohort.plan().checksum().into(),
        cohort_receipt_checksum: cohort.receipt_checksum().into(),
        episode_checksums: cohort
            .episodes()
            .each_ref()
            .map(|e| e.canonical_record_checksum().into()),
        rollout_families: families.clone(),
        parent_lineage_checksum: parent_lineage,
        lineage_checksum: lineage,
        family_count: if changed { size } else { parent.family_count() },
        config: *config,
        counts,
        deltas,
        numeric: proposal.numeric,
        checksum: String::new(),
    };
    report.checksum = report.expected_checksum()?;
    if !changed {
        return Ok(PreparedLongStep::NoChange(report));
    }
    let model = PublicPolicyModel {
        parameters: proposal.parameters,
    };
    model.validate()?;
    let checksum = artifact_checksum(count, &report, &model)?;
    Ok(PreparedLongStep::Updated(LongRlArtifact {
        update_count: count,
        report,
        model,
        checksum,
    }))
}

// A bounded pinned checkpoint may reconstruct content, never attest its
// optimization history. Exact trained membership is independently derived from
// the session's ordered journal, not accepted from the artifact JSON.
pub(crate) fn restore_checkpoint_owner(
    value: serde_json::Value,
    families: BTreeSet<String>,
    limits: LongRunLimits,
) -> Result<LongRlPolicy, String> {
    #[derive(Deserialize)]
    #[serde(rename_all = "camelCase", deny_unknown_fields)]
    struct Report {
        contract: serde_json::Value,
        parent_artifact_checksum: String,
        parent_update_count: u64,
        attempted_update_count: u64,
        root_init_checksum: String,
        root_count1_checksum: String,
        bc_source: SeatPolicy,
        cohort_contract: String,
        sampling_version: String,
        cohort_plan_checksum: String,
        cohort_receipt_checksum: String,
        episode_checksums: [String; 4],
        rollout_families: [String; 4],
        parent_lineage_checksum: String,
        lineage_checksum: String,
        family_count: usize,
        config: serde_json::Value,
        counts: [usize; 3],
        deltas: Deltas,
        numeric: NumericReport,
        checksum: String,
    }
    #[derive(Deserialize)]
    #[serde(rename_all = "camelCase", deny_unknown_fields)]
    struct Artifact {
        update_count: u64,
        report: Report,
        model: PublicPolicyModel,
        checksum: String,
    }
    let decoded: Artifact = serde_json::from_value(value.clone()).map_err(|e| e.to_string())?;
    let r = decoded.report;
    let source = if r.parent_update_count == 1 {
        (COHORT_CONTRACT, RL_SAMPLING_VERSION)
    } else {
        (COHORT_VERSION, SAMPLING_VERSION)
    };
    if r.contract != serde_json::to_value(contract(limits)).map_err(|e| e.to_string())?
        || (r.cohort_contract.as_str(), r.sampling_version.as_str()) != source
    {
        return Err("Long checkpoint contract mismatch".into());
    }
    let config = OneStepConfig::from_checkpoint(r.config)?;
    let mut latest = r.rollout_families.to_vec();
    latest.sort();
    crate::public_policy_update::validate_checkpoint_report(
        &r.counts,
        &r.deltas,
        &r.numeric,
        &config,
        (
            [
                &r.root_init_checksum,
                &r.cohort_plan_checksum,
                &r.cohort_receipt_checksum,
                &r.checksum,
            ],
            &r.episode_checksums,
        ),
        &latest,
        &r.bc_source,
    )?;
    let owner = LongRlPolicy {
        artifact: LongRlArtifact {
            update_count: decoded.update_count,
            model: decoded.model,
            checksum: decoded.checksum,
            report: LongStepReport {
                contract: contract(limits),
                parent_artifact_checksum: r.parent_artifact_checksum,
                parent_update_count: r.parent_update_count,
                attempted_update_count: r.attempted_update_count,
                root_init_checksum: r.root_init_checksum,
                root_count1_checksum: r.root_count1_checksum,
                bc_source: r.bc_source,
                cohort_contract: source.0,
                sampling_version: source.1,
                cohort_plan_checksum: r.cohort_plan_checksum,
                cohort_receipt_checksum: r.cohort_receipt_checksum,
                episode_checksums: r.episode_checksums,
                rollout_families: r.rollout_families,
                parent_lineage_checksum: r.parent_lineage_checksum,
                lineage_checksum: r.lineage_checksum,
                family_count: r.family_count,
                config,
                counts: r.counts,
                deltas: r.deltas,
                numeric: r.numeric,
                checksum: r.checksum,
            },
        },
        families,
    };
    owner.validate_for_sampling()?;
    if serde_json::to_value(owner.artifact()).map_err(|e| e.to_string())? != value {
        return Err("Noncanonical Long checkpoint fields".into());
    }
    Ok(owner)
}
pub(crate) fn extend_lineage(parent: &str, families: &[String; 4]) -> Result<String, String> {
    hash(
        b"tzolkin-public-rl-long-lineage-step-v2\0",
        &(parent, families),
    )
}
fn artifact_checksum(
    count: u64,
    report: &LongStepReport,
    model: &PublicPolicyModel,
) -> Result<String, String> {
    hash(
        b"tzolkin-public-rl-long-artifact-v2\0",
        &(count, report, model),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn explicit_limits_allow_eleven_then_reject_final_and_overflow() {
        let limits = LongRunLimits::new(11, 64).unwrap();
        assert_eq!(limits.next_count(10).unwrap(), 11);
        assert!(limits.next_count(11).is_err());
        assert!(LongRunLimits::new(u64::MAX, usize::MAX).is_err());
        assert!(LongRunLimits::new(1, 64).is_err());
        assert!(LongRunLimits::new(11, crate::policy_dataset::MAX_FILES + 45).is_err());
    }
    #[test]
    fn lineage_binds_immediate_parent_and_ordered_whole_batch() {
        let batch = ["a".into(), "b".into(), "c".into(), "d".into()];
        let a = extend_lineage("parent-a", &batch).unwrap();
        let mut swapped = batch.clone();
        swapped.swap(0, 1);
        assert_ne!(a, extend_lineage("parent-b", &batch).unwrap());
        assert_ne!(a, extend_lineage("parent-a", &swapped).unwrap());
    }
}
