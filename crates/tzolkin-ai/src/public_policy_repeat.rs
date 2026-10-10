//! One plain ascent step from the sealed current RL parent and a whole fresh
//! four-game cohort. Count2..10 share one owner and update contract; count1's
//! owner/contract remain separate. Each next update requires a fresh closed
//! rollout cohort bound to the immediate parent's artifact and source version.
//! The objective is an independent-actor best-response surrogate, not improving
//! the constant all-seat winner-share mean. No loader, inference or I/O exists.
//! Checksums bind consistency, not producer/time/independent-origin proof.
use std::collections::BTreeSet;

use serde::{Deserialize, Serialize};

use crate::dataset::{DatasetSplit, split_for_family};
use crate::features::PUBLIC_FEATURE_SCHEMA;
use crate::public_model::{self, PARAMETER_COUNT, PublicPolicyModel};
use crate::public_policy_episode::TARGET_CONTRACT;
use crate::public_policy_likelihood::{CORRECTION_VERSION, LIKELIHOOD_VERSION};
use crate::public_policy_pullback::PULLBACK_VERSION;
use crate::public_policy_update::{
    Deltas, MAX_ALL_LOG_DELTA, NORMALIZER_VERSION, NumericReport, OneStepConfig,
    UpdatedPublicRlPolicy, accumulate_step, add_actor, hash, propose_parameters,
};
use crate::public_rl_policy_cohort::{COHORT_CONTRACT, ValidatedRlCohort};
use crate::public_stochastic::{RL_SAMPLING_VERSION, RNG_VERSION};
use crate::public_stochastic_native::numerical_target;
use crate::replay::{RULES_BASELINE, RULES_VERSION, SeatPolicy, catalog_hash};

pub const UPDATE_VERSION: &str = "public-current-parent-repeat-ascent-f64-to-f32-v1";
pub const ARTIFACT_SCHEMA: &str = "tzolkin-public-rl-repeat-v1";
pub const TASK: &str = "policyOnlyRlRepeat";
pub const POLICY_VERSION: &str = "learned-public-rl-repeat-v1";
pub const MAX_UPDATE_COUNT: u64 = 10;
// Distinct contracts for repeated collection/cohort adapters. Count1 sources
// cannot carry these identities.
pub const REPEATED_COHORT_CONTRACT: &str = "four-train-games-all-seat-rl-repeat-actor-sum-v1";
pub const REPEATED_SAMPLING_VERSION: &str = "public-stochastic-rl-repeat-uniform-tick53-v1";
const MAX_FAMILIES: usize = crate::policy_dataset::MAX_FILES + 4 * MAX_UPDATE_COUNT as usize;

#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
enum ParentKind {
    Count1,
    Repeated,
}
impl ParentKind {
    fn source_contracts(self) -> (&'static str, &'static str) {
        match self {
            Self::Count1 => (COHORT_CONTRACT, RL_SAMPLING_VERSION),
            Self::Repeated => (REPEATED_COHORT_CONTRACT, REPEATED_SAMPLING_VERSION),
        }
    }
}

/// The caller supplies only an immutable sealed owner, never parameters or a
/// claimed artifact identity. Repeated count10 cannot produce an eleventh update.
#[derive(Clone, Copy)]
pub enum RlUpdateParent<'a> {
    Count1(&'a UpdatedPublicRlPolicy),
    Repeated(&'a RepeatedPublicRlPolicy),
}
impl RlUpdateParent<'_> {
    pub fn checksum(&self) -> &str {
        match self {
            Self::Count1(policy) => policy.artifact().checksum(),
            Self::Repeated(policy) => policy.artifact().checksum(),
        }
    }
    pub fn update_count(&self) -> u64 {
        match self {
            Self::Count1(policy) => policy.artifact().update_count(),
            Self::Repeated(policy) => policy.artifact().update_count(),
        }
    }
    pub fn family_closure(&self) -> &[String] {
        match self {
            Self::Count1(policy) => policy.artifact().report().family_closure(),
            Self::Repeated(policy) => policy.artifact().report().family_closure(),
        }
    }
    fn kind(&self) -> ParentKind {
        match self {
            Self::Count1(_) => ParentKind::Count1,
            Self::Repeated(_) => ParentKind::Repeated,
        }
    }
    fn model(&self) -> &PublicPolicyModel {
        match self {
            Self::Count1(policy) => policy.model(),
            Self::Repeated(policy) => policy.model(),
        }
    }
    pub(crate) fn root_init_checksum(&self) -> &str {
        match self {
            Self::Count1(policy) => policy.artifact().report().parent_init_checksum(),
            Self::Repeated(policy) => &policy.artifact.report.root_init_checksum,
        }
    }
    pub(crate) fn bc_source(&self) -> &SeatPolicy {
        match self {
            Self::Count1(policy) => policy.artifact().report().bc_source(),
            Self::Repeated(policy) => &policy.artifact.report.bc_source,
        }
    }
    fn validate(&self) -> Result<(), String> {
        match self {
            Self::Count1(policy) => policy.validate_for_inference(),
            Self::Repeated(policy) => policy.validate_for_handle(),
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
    max_update_count: u64,
}
fn contract() -> Contract {
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
        parameter_count: PARAMETER_COUNT,
        rules_version: RULES_VERSION,
        rules_baseline: RULES_BASELINE,
        catalog_hash: catalog_hash(),
        numerical_target: numerical_target(),
        backend: "scalar",
        max_all_log_delta: MAX_ALL_LOG_DELTA,
        max_update_count: MAX_UPDATE_COUNT,
    }
}
#[derive(Debug, Deserialize, Serialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct ParentIdentity {
    kind: ParentKind,
    artifact_checksum: String,
    update_count: u64,
}
/// Serialize-only receipt. NoChange retains its attempted next count in the
/// report, but creates no updated owner and does not increment the parent.
#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct RepeatedStepReport {
    contract: Contract,
    parent: ParentIdentity,
    attempted_update_count: u64,
    root_init_checksum: String,
    bc_source: SeatPolicy,
    cohort_contract: &'static str,
    sampling_version: &'static str,
    cohort_plan_checksum: String,
    cohort_receipt_checksum: String,
    episode_checksums: [String; 4],
    config: OneStepConfig,
    counts: [usize; 3],
    deltas: Deltas,
    numeric: NumericReport,
    family_closure: Vec<String>,
    checksum: String,
}
impl RepeatedStepReport {
    pub(crate) fn checkpoint_plan_checksum(&self) -> &str {
        &self.cohort_plan_checksum
    }
    pub fn checksum(&self) -> &str {
        &self.checksum
    }
    pub fn parent_checksum(&self) -> &str {
        &self.parent.artifact_checksum
    }
    pub fn parent_update_count(&self) -> u64 {
        self.parent.update_count
    }
    pub fn attempted_update_count(&self) -> u64 {
        self.attempted_update_count
    }
    pub fn root_init_checksum(&self) -> &str {
        &self.root_init_checksum
    }
    pub fn config(&self) -> &OneStepConfig {
        &self.config
    }
    pub fn counts(&self) -> [usize; 3] {
        self.counts
    }
    pub fn changed_parameters(&self) -> usize {
        self.numeric.changed_parameters
    }
    pub fn family_closure(&self) -> &[String] {
        &self.family_closure
    }
    fn expected_checksum(&self) -> Result<String, String> {
        hash(
            b"tzolkin-public-rl-repeat-report-v1\0",
            &(
                &self.contract,
                &self.parent,
                self.attempted_update_count,
                &self.root_init_checksum,
                &self.bc_source,
                (self.cohort_contract, self.sampling_version),
                (&self.cohort_plan_checksum, &self.cohort_receipt_checksum),
                &self.episode_checksums,
                self.config,
                self.counts,
                &self.deltas,
                &self.numeric,
                &self.family_closure,
            ),
        )
    }
}

/// The immediate parent is linked by checksum, not embedded with another model.
/// There is no Deserialize, caller-supplied weights or arbitrary reseal route.
#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct RepeatedPublicRlArtifact {
    update_count: u64,
    report: RepeatedStepReport,
    model: PublicPolicyModel,
    checksum: String,
}
impl RepeatedPublicRlArtifact {
    pub fn checksum(&self) -> &str {
        &self.checksum
    }
    pub fn update_count(&self) -> u64 {
        self.update_count
    }
    pub fn report(&self) -> &RepeatedStepReport {
        &self.report
    }
}
pub struct RepeatedPublicRlPolicy {
    artifact: RepeatedPublicRlArtifact,
}
impl RepeatedPublicRlPolicy {
    pub fn artifact(&self) -> &RepeatedPublicRlArtifact {
        &self.artifact
    }
    pub(crate) fn model(&self) -> &PublicPolicyModel {
        &self.artifact.model
    }
    /// For a future typed repeated handle, not raw-content qualification or
    /// producer authentication. The dedicated pinned-checkpoint session can
    /// restore the same content without replaying the optimization history.
    pub(crate) fn validate_for_handle(&self) -> Result<(), String> {
        let artifact = &self.artifact;
        let report = &artifact.report;
        let expected_count = next_count(report.parent.update_count)?;
        let source = report.parent.kind.source_contracts();
        if report.contract != contract()
            || !(2..=MAX_UPDATE_COUNT).contains(&artifact.update_count)
            || artifact.update_count != expected_count
            || report.attempted_update_count != expected_count
            || (report.parent.kind == ParentKind::Count1) != (report.parent.update_count == 1)
            || (report.cohort_contract, report.sampling_version) != source
            || report.numeric.changed_parameters == 0
            || !digest_id(&report.parent.artifact_checksum)
            || !digest_id(&report.root_init_checksum)
        {
            return Err("Incompatible repeated RL owner contract".into());
        }
        report.config.validate()?;
        report.bc_source.validate()?;
        if !matches!(&report.bc_source, SeatPolicy::PublicLearned { inference_backend, .. } if inference_backend == "scalar")
        {
            return Err("Repeated RL ancestry requires Scalar BC provenance".into());
        }
        validate_closure(&report.family_closure)?;
        artifact.model.validate()?;
        if report.checksum != report.expected_checksum()?
            || artifact.checksum
                != artifact_checksum(artifact.update_count, report, &artifact.model)?
        {
            return Err("Repeated RL owner checksum mismatch".into());
        }
        Ok(())
    }
}
pub(crate) fn restore_checkpoint_owner(
    value: serde_json::Value,
) -> Result<RepeatedPublicRlPolicy, String> {
    #[derive(Deserialize)]
    #[serde(rename_all = "camelCase", deny_unknown_fields)]
    struct Report {
        contract: serde_json::Value,
        parent: ParentIdentity,
        attempted_update_count: u64,
        root_init_checksum: String,
        bc_source: SeatPolicy,
        cohort_contract: String,
        sampling_version: String,
        cohort_plan_checksum: String,
        cohort_receipt_checksum: String,
        episode_checksums: [String; 4],
        config: serde_json::Value,
        counts: [usize; 3],
        deltas: Deltas,
        numeric: NumericReport,
        family_closure: Vec<String>,
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
    let source = r.parent.kind.source_contracts();
    if r.contract != serde_json::to_value(contract()).map_err(|e| e.to_string())?
        || r.cohort_contract != source.0
        || r.sampling_version != source.1
    {
        return Err("Checkpoint repeated contract mismatch".into());
    }
    let config = OneStepConfig::from_checkpoint(r.config)?;
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
        &r.family_closure,
        &r.bc_source,
    )?;
    let owner = RepeatedPublicRlPolicy {
        artifact: RepeatedPublicRlArtifact {
            update_count: decoded.update_count,
            report: RepeatedStepReport {
                contract: contract(),
                parent: r.parent,
                attempted_update_count: r.attempted_update_count,
                root_init_checksum: r.root_init_checksum,
                bc_source: r.bc_source,
                cohort_contract: source.0,
                sampling_version: source.1,
                cohort_plan_checksum: r.cohort_plan_checksum,
                cohort_receipt_checksum: r.cohort_receipt_checksum,
                episode_checksums: r.episode_checksums,
                config,
                counts: r.counts,
                deltas: r.deltas,
                numeric: r.numeric,
                family_closure: r.family_closure,
                checksum: r.checksum,
            },
            model: decoded.model,
            checksum: decoded.checksum,
        },
    };
    owner.validate_for_handle()?;
    if serde_json::to_value(owner.artifact()).map_err(|e| e.to_string())? != value {
        return Err("Noncanonical checkpoint repeated fields".into());
    }
    Ok(owner)
}
pub enum RepeatedStepOutcome {
    Updated(RepeatedPublicRlPolicy),
    NoChange(RepeatedStepReport),
}

/// The parent/source/freshness/row-cap checks precede every forward. All four
/// episodes and all own decisions complete before publishing one new owner.
/// Future repeated collection must supply its distinct source contracts; old
/// count1 records cannot be reused as a cohort for a repeated parent.
pub fn ascent_repeated(
    parent: RlUpdateParent<'_>,
    cohort: &ValidatedRlCohort,
    config: &OneStepConfig,
) -> Result<RepeatedStepOutcome, String> {
    config.validate()?;
    parent.validate()?;
    let count = next_count(parent.update_count())?;
    let source = parent.kind().source_contracts();
    if parent.checksum() != cohort.plan().parent_artifact_checksum()
        || parent.update_count() != cohort.plan().parent_update_count()
        || cohort.plan().numerical_target() != numerical_target()
        || cohort.plan().cohort_contract() != source.0
        || cohort.total_candidate_rows() > config.max_candidate_rows()
        || cohort.episodes().iter().any(|episode| {
            episode.parent_artifact_checksum() != parent.checksum()
                || episode.parent_update_count() != parent.update_count()
                || episode.sampling_version() != source.1
        })
    {
        return Err("Repeated update parent/cohort/sampler/target/budget mismatch".into());
    }
    let families = extend_closure(parent.family_closure(), cohort.plan().rollout_families())?;
    let mut gradient = vec![0.0; PARAMETER_COUNT];
    let mut counts = [0usize; 3];
    let mut deltas = Deltas::default();
    for (member, episode) in cohort.episodes().iter().enumerate() {
        let players = episode.source_config().players();
        for actor in 0..players {
            let mut own = vec![0.0; PARAMETER_COUNT];
            for step in episode.steps().filter(|step| step.actor() == actor) {
                let index = step.global_callback_index();
                accumulate_step(parent.model(), step, players, config, &mut own, &mut counts, &mut deltas)
                    .map_err(|error| format!("Whole repeated cohort rejected: member {member} actor {actor} callback {index}: {error}"))?;
            }
            add_actor(
                &mut gradient,
                &own,
                cohort.actor_coefficient(member, actor)?,
            )?;
        }
    }
    if counts
        != [
            cohort.total_callbacks(),
            cohort.total_candidate_rows(),
            cohort.total_singletons(),
        ]
    {
        return Err("Reconstructed repeated cohort counts differ from sealed receipt".into());
    }
    let proposal = propose_parameters(parent.model().parameters(), &gradient, config)?;
    let mut report = RepeatedStepReport {
        contract: contract(),
        parent: ParentIdentity {
            kind: parent.kind(),
            artifact_checksum: parent.checksum().into(),
            update_count: parent.update_count(),
        },
        attempted_update_count: count,
        root_init_checksum: parent.root_init_checksum().into(),
        bc_source: parent.bc_source().clone(),
        cohort_contract: source.0,
        sampling_version: source.1,
        cohort_plan_checksum: cohort.plan().checksum().into(),
        cohort_receipt_checksum: cohort.receipt_checksum().into(),
        episode_checksums: cohort
            .episodes()
            .each_ref()
            .map(|episode| episode.canonical_record_checksum().into()),
        config: *config,
        counts,
        deltas,
        numeric: proposal.numeric,
        family_closure: families,
        checksum: String::new(),
    };
    report.checksum = report.expected_checksum()?;
    if report.numeric.changed_parameters == 0 {
        return Ok(RepeatedStepOutcome::NoChange(report));
    }
    let model = PublicPolicyModel {
        parameters: proposal.parameters,
    };
    model.validate()?;
    let checksum = artifact_checksum(count, &report, &model)?;
    Ok(RepeatedStepOutcome::Updated(RepeatedPublicRlPolicy {
        artifact: RepeatedPublicRlArtifact {
            update_count: count,
            report,
            model,
            checksum,
        },
    }))
}
fn next_count(parent_count: u64) -> Result<u64, String> {
    if !(1..MAX_UPDATE_COUNT).contains(&parent_count) {
        return Err("Repeated update requires parent count1..9; count10 is final".into());
    }
    Ok(parent_count + 1)
}
fn validate_closure(families: &[String]) -> Result<(), String> {
    if families.is_empty()
        || families.len() > MAX_FAMILIES
        || families.windows(2).any(|pair| pair[0] >= pair[1])
    {
        return Err("Invalid sorted bounded repeated family closure".into());
    }
    for family in families {
        if split_for_family(family)? == DatasetSplit::Test {
            return Err("Repeated lineage includes Test".into());
        }
    }
    Ok(())
}
fn extend_closure(inherited: &[String], rollout: &[String; 4]) -> Result<Vec<String>, String> {
    validate_closure(inherited)?;
    let mut families = inherited.iter().cloned().collect::<BTreeSet<_>>();
    for family in rollout {
        if split_for_family(family)? != DatasetSplit::Train || !families.insert(family.clone()) {
            return Err(
                "Repeated rollout requires distinct Train families outside inherited lineage"
                    .into(),
            );
        }
    }
    let result = families.into_iter().collect::<Vec<_>>();
    validate_closure(&result)?;
    Ok(result)
}
fn artifact_checksum(
    count: u64,
    report: &RepeatedStepReport,
    model: &PublicPolicyModel,
) -> Result<String, String> {
    hash(
        b"tzolkin-public-rl-repeat-artifact-v1\0",
        &(count, report, model),
    )
}
fn digest_id(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}

#[cfg(test)]
pub(crate) fn checkpoint_test_owner(
    parent: RlUpdateParent<'_>,
    plan: &str,
    families: Vec<String>,
    counts: [usize; 3],
    config: &OneStepConfig,
) -> RepeatedPublicRlPolicy {
    // Private format/numeric fixture. No rollout, likelihood or forward occurs.
    let mut gradient = vec![0.0; PARAMETER_COUNT];
    gradient[2] = -3.0;
    let proposal = propose_parameters(parent.model().parameters(), &gradient, config).unwrap();
    let source = parent.kind().source_contracts();
    let count = next_count(parent.update_count()).unwrap();
    let mut report = RepeatedStepReport {
        contract: contract(),
        parent: ParentIdentity {
            kind: parent.kind(),
            artifact_checksum: parent.checksum().into(),
            update_count: parent.update_count(),
        },
        attempted_update_count: count,
        root_init_checksum: parent.root_init_checksum().into(),
        bc_source: parent.bc_source().clone(),
        cohort_contract: source.0,
        sampling_version: source.1,
        cohort_plan_checksum: plan.into(),
        cohort_receipt_checksum: "b".repeat(64),
        episode_checksums: std::array::from_fn(|i| format!("{:064x}", i + 1)),
        config: *config,
        counts,
        deltas: Deltas::default(),
        numeric: proposal.numeric,
        family_closure: families,
        checksum: String::new(),
    };
    report.checksum = report.expected_checksum().unwrap();
    let model = PublicPolicyModel {
        parameters: proposal.parameters,
    };
    let checksum = artifact_checksum(count, &report, &model).unwrap();
    RepeatedPublicRlPolicy {
        artifact: RepeatedPublicRlArtifact {
            update_count: count,
            report,
            model,
            checksum,
        },
    }
}
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lineage_union_keeps_validation_but_rejects_test_overlap_and_wrong_order() {
        // Synthetic digest partitions only: no game, producer or owner is forged.
        let family = |first: u8| format!("{first:02x}{}", "0".repeat(62));
        let inherited = vec![family(0), family(2)];
        let rollout = [family(3), family(4), family(5), family(6)];
        let expected = (0..=6)
            .filter(|value| *value != 1)
            .map(family)
            .collect::<Vec<_>>();
        assert_eq!(extend_closure(&inherited, &rollout).unwrap(), expected);
        for first in [0, 1, 2, 4] {
            let mut invalid = rollout.clone();
            invalid[0] = family(first);
            assert!(extend_closure(&inherited, &invalid).is_err());
        }
        assert!(extend_closure(&[family(1)], &rollout).is_err());
        assert!(extend_closure(&[family(2), family(0)], &rollout).is_err());
    }
}
