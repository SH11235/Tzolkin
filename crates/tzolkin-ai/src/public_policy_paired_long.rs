//! Joint actor/residual owners for current-parent, whole-four training.
//! This task cannot be constructed by relabeling a Plain or Long checkpoint.
use crate::dataset::{DatasetSplit, split_for_family};
use crate::features::PUBLIC_FEATURE_SCHEMA;
use crate::public_model::{self, PublicPolicyModel};
use crate::public_policy_baseline::{self, InitializedResidualBaseline, ResidualFitConfig};
use crate::public_policy_paired::{self, PairedStepOutcome};
use crate::public_policy_update::{
    self, Deltas, NumericReport, OneStepConfig, UpdatedPublicRlPolicy,
    accumulate_step_with_baseline, add_actor, checkpoint_digest, hash, propose_parameters,
};
use crate::public_rl_artifact::InitializedPublicRlPolicy;
use crate::public_rl_paired_policy_cohort::ValidatedPairedCohort;
use crate::public_rl_policy_cohort::ValidatedRlCohort;
use crate::public_state_critic::{self, PublicStateContext};
use crate::public_stochastic_native::numerical_target;
use crate::replay::{SeatPolicy, catalog_hash};
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

pub const ARTIFACT_SCHEMA: &str = "tzolkin-public-rl-paired-long-v1";
pub const TASK: &str = "jointActorResidualCurrentParentV1";
pub const POLICY_VERSION: &str = "learned-public-rl-paired-long-v1";
pub const SAMPLING_VERSION: &str = "public-stochastic-rl-paired-uniform-tick53-v1";
pub const COHORT_VERSION: &str = "four-train-games-all-seat-paired-actor-sum-v1";
pub const EPISODE_VERSION: &str = "applied-paired-actor-gamma1-terminal-winner-share-v1";
pub const MAX_UPDATES: u64 = 1_000_000;
#[derive(Clone, Copy)]
pub enum PairedRlParent<'a> {
    Count1(&'a UpdatedPublicRlPolicy),
    Paired(&'a PairedRlPolicy),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct PairedRunLimits {
    max_updates: u64,
    max_families: usize,
}
impl PairedRunLimits {
    pub fn new(max_updates: u64, max_families: usize) -> Result<Self, String> {
        let ceiling = usize::try_from(max_updates)
            .ok()
            .and_then(|n| n.checked_mul(4))
            .and_then(|n| n.checked_add(crate::policy_dataset::MAX_FILES))
            .ok_or("Paired limits overflow")?;
        if !(2..=MAX_UPDATES).contains(&max_updates) || max_families == 0 || max_families > ceiling
        {
            return Err("Invalid explicit paired update/family limits".into());
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
}
fn contract(limits: PairedRunLimits) -> serde_json::Value {
    serde_json::json!({
        "schema": ARTIFACT_SCHEMA, "task": TASK, "policyVersion": POLICY_VERSION,
        "updateVersion": public_policy_paired::UPDATE_VERSION,
        "modelVersion": public_model::MODEL_VERSION, "inputContract": public_model::INPUT_CONTRACT,
        "featureSchema": PUBLIC_FEATURE_SCHEMA, "actorHidden": public_model::HIDDEN, "actorParameterCount": public_model::PARAMETER_COUNT,
        "criticModelVersion": public_state_critic::MODEL_VERSION, "contextContract": public_state_critic::CONTEXT_CONTRACT,
        "contextSchema": public_state_critic::CONTEXT_SCHEMA, "contextCount": public_state_critic::CONTEXT_COUNT,
        "criticHidden": public_state_critic::HIDDEN, "criticParameterCount": public_state_critic::PARAMETER_COUNT,
        "baselineVersion": public_policy_baseline::BASELINE_VERSION, "fitVersion": public_policy_baseline::FIT_VERSION,
        "normalizerVersion": public_policy_update::NORMALIZER_VERSION,
        "likelihoodVersion": crate::public_policy_likelihood::LIKELIHOOD_VERSION,
        "correctionVersion": crate::public_policy_likelihood::CORRECTION_VERSION,
        "pullbackVersion": crate::public_policy_pullback::PULLBACK_VERSION,
        "targetContract": crate::public_policy_episode::TARGET_CONTRACT,
        "episodeContract": EPISODE_VERSION, "cohortContract": COHORT_VERSION,
        "samplingVersion": SAMPLING_VERSION, "rngVersion": crate::public_stochastic::RNG_VERSION,
        "denominatorBits": crate::public_stochastic::DENOMINATOR_BITS,
        "uniformMixtureBits": crate::public_stochastic_record::hex64(crate::public_stochastic::UNIFORM_MIXTURE.to_bits()),
        "observationSchema": tzolkin_core::observation::OBSERVATION_SCHEMA, "moveSchema": tzolkin_core::observation::MOVE_SCHEMA,
        "rulesVersion": crate::replay::RULES_VERSION, "rulesBaseline": crate::replay::RULES_BASELINE, "catalogHash": catalog_hash(),
        "numericalTarget": numerical_target(), "backend": "scalar", "maxAllLogDelta": public_policy_update::MAX_ALL_LOG_DELTA,
        "limits": limits
    })
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct PairedLongStepReport {
    contract: serde_json::Value,
    parent_artifact_checksum: String,
    parent_update_count: u64,
    attempted_update_count: u64,
    root_init_checksum: String,
    root_count1_checksum: String,
    residual_seed: u64,
    root_residual_checksum: String,
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
    actor_config: serde_json::Value,
    critic_config: serde_json::Value,
    counts: [usize; 3],
    deltas: Deltas,
    numeric: NumericReport,
    mean_raw_residual_mse: f64,
    critic_changed_parameters: usize,
    checksum: String,
}
impl PairedLongStepReport {
    pub fn checksum(&self) -> &str {
        &self.checksum
    }
    pub fn parent_checksum(&self) -> &str {
        &self.parent_artifact_checksum
    }
    pub fn parent_update_count(&self) -> u64 {
        self.parent_update_count
    }
    pub fn root_init_checksum(&self) -> &str {
        &self.root_init_checksum
    }
    pub fn root_count1_checksum(&self) -> &str {
        &self.root_count1_checksum
    }
    pub fn residual_seed(&self) -> u64 {
        self.residual_seed
    }
    pub fn root_residual_checksum(&self) -> &str {
        &self.root_residual_checksum
    }
    pub fn bc_source(&self) -> &SeatPolicy {
        &self.bc_source
    }
    pub fn counts(&self) -> [usize; 3] {
        self.counts
    }
    pub fn changed_parameters(&self) -> [usize; 2] {
        [
            self.numeric.changed_parameters,
            self.critic_changed_parameters,
        ]
    }
    pub fn mean_raw_residual_mse(&self) -> f64 {
        self.mean_raw_residual_mse
    }
    pub fn plan_checksum(&self) -> &str {
        &self.cohort_plan_checksum
    }
    pub fn lineage_checksum(&self) -> &str {
        &self.lineage_checksum
    }
    pub(crate) fn parent_lineage(&self) -> &str {
        &self.parent_lineage_checksum
    }
    pub fn rollout_families(&self) -> &[String; 4] {
        &self.rollout_families
    }
    pub fn family_count(&self) -> usize {
        self.family_count
    }
    pub fn configs_match(
        &self,
        actor: &OneStepConfig,
        critic: &ResidualFitConfig,
    ) -> Result<bool, String> {
        Ok(
            self.actor_config == serde_json::to_value(actor).map_err(|e| e.to_string())?
                && self.critic_config == serde_json::to_value(critic).map_err(|e| e.to_string())?,
        )
    }
    fn expected_checksum(&self) -> Result<String, String> {
        let mut value = serde_json::to_value(self).map_err(|e| e.to_string())?;
        value
            .as_object_mut()
            .ok_or("Paired report object")?
            .remove("checksum");
        hash(b"tzolkin-public-rl-paired-long-report-v1\0", &value)
    }
}
#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct PairedLongArtifact {
    update_count: u64,
    report: PairedLongStepReport,
    actor: PublicPolicyModel,
    residual_parameters: Vec<f32>,
    checksum: String,
}
impl PairedLongArtifact {
    pub fn checksum(&self) -> &str {
        &self.checksum
    }
    pub fn update_count(&self) -> u64 {
        self.update_count
    }
    pub fn report(&self) -> &PairedLongStepReport {
        &self.report
    }
    fn expected_checksum(&self) -> Result<String, String> {
        hash(
            b"tzolkin-public-rl-paired-long-artifact-v1\0",
            &(
                self.update_count,
                &self.report,
                &self.actor,
                &self.residual_parameters,
            ),
        )
    }
}
/// Only a joint producer or internally journal-checked restore constructs owners.
pub struct PairedRlPolicy {
    artifact: PairedLongArtifact,
    families: BTreeSet<String>,
    limits: PairedRunLimits,
}
impl PairedRlPolicy {
    pub fn artifact(&self) -> &PairedLongArtifact {
        &self.artifact
    }
    pub fn limits(&self) -> PairedRunLimits {
        self.limits
    }
    pub fn actor_model(&self) -> &PublicPolicyModel {
        &self.artifact.actor
    }
    pub fn residual_parameters(&self) -> &[f32] {
        &self.artifact.residual_parameters
    }
    pub(crate) fn family_set(&self) -> &BTreeSet<String> {
        &self.families
    }
    pub(crate) fn contains_family(&self, family: &str) -> bool {
        self.families.contains(family)
    }
    pub(crate) fn validate_for_sampling(&self) -> Result<(), String> {
        PairedRunLimits::new(self.limits.max_updates, self.limits.max_families)?;
        let a = &self.artifact;
        let r = &a.report;
        let actor = OneStepConfig::from_checkpoint(r.actor_config.clone())?;
        let critic = ResidualFitConfig::from_checkpoint(r.critic_config.clone())?;
        let source = if r.parent_update_count == 1 {
            (
                crate::public_rl_policy_cohort::COHORT_CONTRACT,
                crate::public_stochastic::RL_SAMPLING_VERSION,
            )
        } else {
            (COHORT_VERSION, SAMPLING_VERSION)
        };
        if r.contract != contract(self.limits)
            || a.update_count != r.attempted_update_count
            || a.update_count
                != r.parent_update_count
                    .checked_add(1)
                    .ok_or("Paired count overflow")?
            || !(2..=self.limits.max_updates).contains(&a.update_count)
            || (r.cohort_contract.as_str(), r.sampling_version.as_str()) != source
            || r.family_count != self.families.len()
            || self.families.len() > self.limits.max_families
            || self.families.is_empty()
            || self.families.len()
                < usize::try_from(a.update_count)
                    .map_err(|_| "Paired count shape overflow")?
                    .checked_mul(4)
                    .ok_or("Paired closure size overflow")?
            || (r.parent_update_count == 1 && r.parent_artifact_checksum != r.root_count1_checksum)
            || r.root_residual_checksum
                != public_policy_baseline::zero_checksum(&r.root_init_checksum, r.residual_seed)?
            || r.changed_parameters() == [0, 0]
            || r.critic_changed_parameters > public_state_critic::PARAMETER_COUNT
            || !r.mean_raw_residual_mse.is_finite()
            || r.mean_raw_residual_mse < 0.0
            || a.residual_parameters.len() != public_state_critic::PARAMETER_COUNT
            || a.residual_parameters.iter().any(|x| !x.is_finite())
            || ![
                &r.root_count1_checksum,
                &r.root_residual_checksum,
                &r.parent_artifact_checksum,
                &r.parent_lineage_checksum,
                &r.lineage_checksum,
            ]
            .iter()
            .all(|x| checkpoint_digest(x))
            || r.expected_checksum()? != r.checksum
            || a.expected_checksum()? != a.checksum
            || extend_lineage(&r.parent_lineage_checksum, &r.rollout_families)?
                != r.lineage_checksum
        {
            return Err("Invalid paired owner contract/numeric/content/lineage".into());
        }
        for f in &self.families {
            if split_for_family(f)? == DatasetSplit::Test {
                return Err("Paired owner includes Test family".into());
            }
        }
        for (i, f) in r.rollout_families.iter().enumerate() {
            if split_for_family(f)? != DatasetSplit::Train
                || !self.families.contains(f)
                || r.rollout_families[..i].contains(f)
            {
                return Err("Paired rollout/closure mismatch".into());
            }
        }
        critic.check_callbacks(r.counts[0])?;
        let mut latest = r.rollout_families.to_vec();
        latest.sort();
        public_policy_update::validate_joint_checkpoint_report(
            &r.counts,
            &r.deltas,
            &r.numeric,
            &actor,
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
        a.actor.validate()
    }
}
pub enum PairedStepOutcomeLong {
    Updated(PairedRlPolicy),
    NoChange(PairedLongStepReport),
}
pub(crate) enum PreparedPairedStep {
    Updated(PairedLongArtifact),
    NoChange(PairedLongStepReport),
}
impl PreparedPairedStep {
    pub(crate) fn artifact(&self) -> Option<&PairedLongArtifact> {
        match self {
            Self::Updated(a) => Some(a),
            Self::NoChange(_) => None,
        }
    }
    pub(crate) fn report(&self) -> &PairedLongStepReport {
        match self {
            Self::Updated(a) => &a.report,
            Self::NoChange(r) => r,
        }
    }
    pub(crate) fn commit(
        self,
        mut families: BTreeSet<String>,
        limits: PairedRunLimits,
    ) -> PairedStepOutcomeLong {
        match self {
            Self::NoChange(r) => PairedStepOutcomeLong::NoChange(r),
            Self::Updated(artifact) => {
                families.extend(artifact.report.rollout_families.iter().cloned());
                PairedStepOutcomeLong::Updated(PairedRlPolicy {
                    artifact,
                    families,
                    limits,
                })
            }
        }
    }
}
pub(crate) fn take_families(owner: PairedRlPolicy) -> BTreeSet<String> {
    owner.families
}
pub(crate) fn root_lineage(count1: &UpdatedPublicRlPolicy) -> Result<String, String> {
    hash(
        b"tzolkin-public-rl-paired-root-lineage-v1\0",
        &(
            count1.artifact().checksum(),
            count1.artifact().report().family_closure(),
        ),
    )
}
pub(crate) fn extend_lineage(parent: &str, families: &[String; 4]) -> Result<String, String> {
    hash(
        b"tzolkin-public-rl-paired-lineage-step-v1\0",
        &(parent, families),
    )
}
fn finish(
    mut report: PairedLongStepReport,
    actor: PublicPolicyModel,
    residual_parameters: Vec<f32>,
) -> Result<PreparedPairedStep, String> {
    report.checksum = report.expected_checksum()?;
    if report.changed_parameters() == [0, 0] {
        return Ok(PreparedPairedStep::NoChange(report));
    }
    let mut artifact = PairedLongArtifact {
        update_count: report.attempted_update_count,
        report,
        actor,
        residual_parameters,
        checksum: String::new(),
    };
    artifact.actor.validate()?;
    artifact.checksum = artifact.expected_checksum()?;
    Ok(PreparedPairedStep::Updated(artifact))
}
pub(crate) fn prepare_count1_step(
    parent: &UpdatedPublicRlPolicy,
    initial: &InitializedPublicRlPolicy,
    seed: u64,
    cohort: &ValidatedRlCohort,
    actor_config: &OneStepConfig,
    critic_config: &ResidualFitConfig,
    limits: PairedRunLimits,
) -> Result<PreparedPairedStep, String> {
    if parent
        .artifact()
        .report()
        .family_closure()
        .len()
        .checked_add(4)
        .is_none_or(|n| n > limits.max_families)
        || limits.max_updates < 2
    {
        return Err("Paired bootstrap capacity".into());
    }
    let residual = InitializedResidualBaseline::new(initial, seed)?;
    let (old, actor, parameters) = match public_policy_paired::ascent_paired_from_count1(
        parent,
        &residual,
        cohort,
        actor_config,
        critic_config,
    )? {
        PairedStepOutcome::Updated(p) => p.into_parts()?,
        PairedStepOutcome::NoChange(r) => {
            (r, parent.model().clone(), residual.parameters().to_vec())
        }
    };
    let changed = old.changed_parameters() != [0, 0];
    let parent_lineage = root_lineage(parent)?;
    let families = cohort.plan().rollout_families().clone();
    let report = PairedLongStepReport {
        contract: contract(limits),
        parent_artifact_checksum: parent.artifact().checksum().into(),
        parent_update_count: 1,
        attempted_update_count: 2,
        root_init_checksum: initial.artifact().checksum().into(),
        root_count1_checksum: parent.artifact().checksum().into(),
        residual_seed: residual.seed(),
        root_residual_checksum: residual.checksum().into(),
        bc_source: initial.artifact().bc_source().clone(),
        cohort_contract: crate::public_rl_policy_cohort::COHORT_CONTRACT.into(),
        sampling_version: crate::public_stochastic::RL_SAMPLING_VERSION.into(),
        cohort_plan_checksum: cohort.plan().checksum().into(),
        cohort_receipt_checksum: cohort.receipt_checksum().into(),
        episode_checksums: cohort
            .episodes()
            .each_ref()
            .map(|e| e.canonical_record_checksum().into()),
        rollout_families: families.clone(),
        parent_lineage_checksum: parent_lineage.clone(),
        lineage_checksum: if changed {
            extend_lineage(&parent_lineage, &families)?
        } else {
            parent_lineage
        },
        family_count: parent.artifact().report().family_closure().len() + usize::from(changed) * 4,
        actor_config: serde_json::to_value(actor_config).map_err(|e| e.to_string())?,
        critic_config: serde_json::to_value(critic_config).map_err(|e| e.to_string())?,
        counts: old.counts(),
        mean_raw_residual_mse: old.mean_raw_residual_mse(),
        critic_changed_parameters: old.changed_parameters()[1],
        deltas: old.actor_deltas,
        numeric: old.actor_numeric,
        checksum: String::new(),
    };
    finish(report, actor, parameters)
}
pub(crate) fn prepare_paired_step(
    parent: &PairedRlPolicy,
    cohort: &ValidatedPairedCohort,
    actor_config: &OneStepConfig,
    critic_config: &ResidualFitConfig,
    limits: PairedRunLimits,
) -> Result<PreparedPairedStep, String> {
    parent.validate_for_sampling()?;
    actor_config.validate()?;
    let prior = parent.artifact().report();
    let count = parent
        .artifact()
        .update_count()
        .checked_add(1)
        .ok_or("Paired update overflow")?;
    let counts = [
        cohort.total_callbacks(),
        cohort.total_candidate_rows(),
        cohort.total_singletons(),
    ];
    critic_config.check_callbacks(counts[0])?;
    if limits != parent.limits
        || count > limits.max_updates
        || !prior.configs_match(actor_config, critic_config)?
        || parent
            .family_set()
            .len()
            .checked_add(4)
            .is_none_or(|n| n > limits.max_families)
        || cohort.plan().parent_artifact_checksum() != parent.artifact().checksum()
        || cohort.plan().parent_update_count() != parent.artifact().update_count()
        || cohort.plan().numerical_target() != numerical_target()
        || counts[1] > actor_config.max_candidate_rows()
    {
        return Err("Paired continuation immediate parent/config/target/cap mismatch".into());
    }
    for (i, f) in cohort.plan().rollout_families().iter().enumerate() {
        if split_for_family(f)? != DatasetSplit::Train
            || parent.contains_family(f)
            || cohort.plan().rollout_families()[..i].contains(f)
        {
            return Err("Paired continuation requires four fresh distinct Train families".into());
        }
    }
    let mut gradient = vec![0.0; public_model::PARAMETER_COUNT];
    let mut observed = [0; 3];
    let mut deltas = Deltas::default();
    for (member, episode) in cohort.episodes().iter().enumerate() {
        for actor in 0..episode.source_config().players() {
            let mut own = vec![0.0; public_model::PARAMETER_COUNT];
            for step in episode.steps().filter(|s| s.actor() == actor) {
                accumulate_step_with_baseline(
                    parent.actor_model(),
                    step,
                    actor_config,
                    &mut own,
                    &mut observed,
                    &mut deltas,
                    |o| {
                        public_policy_baseline::baseline(
                            parent.residual_parameters(),
                            &PublicStateContext::from_observation(o)?,
                        )
                    },
                )?;
            }
            add_actor(
                &mut gradient,
                &own,
                cohort.actor_coefficient(member, actor)?,
            )?;
        }
    }
    if observed != counts {
        return Err("Paired continuation applied count differs".into());
    }
    let actor = propose_parameters(parent.actor_model().parameters(), &gradient, actor_config)?;
    let critic = public_policy_baseline::fit_applied(
        parent.residual_parameters(),
        cohort.episodes().iter().flat_map(|e| e.steps()),
        counts[0],
        critic_config,
    )?;
    let changed = [actor.numeric.changed_parameters, critic.changed_parameters] != [0, 0];
    let families = cohort.plan().rollout_families().clone();
    let lineage = prior.lineage_checksum.clone();
    finish(
        PairedLongStepReport {
            contract: contract(limits),
            parent_artifact_checksum: parent.artifact().checksum().into(),
            parent_update_count: parent.artifact().update_count(),
            attempted_update_count: count,
            root_init_checksum: prior.root_init_checksum.clone(),
            root_count1_checksum: prior.root_count1_checksum.clone(),
            residual_seed: prior.residual_seed,
            root_residual_checksum: prior.root_residual_checksum.clone(),
            bc_source: prior.bc_source.clone(),
            cohort_contract: COHORT_VERSION.into(),
            sampling_version: SAMPLING_VERSION.into(),
            cohort_plan_checksum: cohort.plan().checksum().into(),
            cohort_receipt_checksum: cohort.receipt_checksum().into(),
            episode_checksums: cohort
                .episodes()
                .each_ref()
                .map(|e| e.canonical_record_checksum().into()),
            rollout_families: families.clone(),
            parent_lineage_checksum: lineage.clone(),
            lineage_checksum: if changed {
                extend_lineage(&lineage, &families)?
            } else {
                lineage
            },
            family_count: parent.family_set().len() + usize::from(changed) * 4,
            actor_config: serde_json::to_value(actor_config).map_err(|e| e.to_string())?,
            critic_config: serde_json::to_value(critic_config).map_err(|e| e.to_string())?,
            counts,
            deltas,
            numeric: actor.numeric,
            mean_raw_residual_mse: critic.mean_loss,
            critic_changed_parameters: critic.changed_parameters,
            checksum: String::new(),
        },
        PublicPolicyModel {
            parameters: actor.parameters,
        },
        critic.parameters,
    )
}
/// Internal closed reconstruction. The session independently derives membership.
pub(crate) fn restore_checkpoint_owner(
    value: serde_json::Value,
    families: BTreeSet<String>,
    limits: PairedRunLimits,
) -> Result<PairedRlPolicy, String> {
    let artifact: PairedLongArtifact =
        serde_json::from_value(value.clone()).map_err(|e| e.to_string())?;
    let owner = PairedRlPolicy {
        artifact,
        families,
        limits,
    };
    owner.validate_for_sampling()?;
    if serde_json::to_value(owner.artifact()).map_err(|e| e.to_string())? != value {
        return Err("Noncanonical paired owner".into());
    }
    Ok(owner)
}
