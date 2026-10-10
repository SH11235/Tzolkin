//! A paired research step from a sealed Count1 actor and its root's zero residual.
//! Actor ascent uses the immutable parent critic; raw residual fitting follows.
//! The joint owner is a distinct task, with no Plain/Long wire or loader conversion.
//! Content bindings do not authenticate producers or optimization history.
use std::collections::BTreeSet;

use serde::Serialize;

use crate::dataset::{DatasetSplit, split_for_family};
use crate::public_model::{self, PublicPolicyModel};
use crate::public_policy_baseline::{
    BASELINE_VERSION, FIT_VERSION, InitializedResidualBaseline, ResidualFitConfig, fit_applied,
};
use crate::public_policy_episode::TARGET_CONTRACT;
use crate::public_policy_likelihood::{CORRECTION_VERSION, LIKELIHOOD_VERSION};
use crate::public_policy_pullback::PULLBACK_VERSION;
use crate::public_policy_update::{
    Deltas, NORMALIZER_VERSION, NumericReport, OneStepConfig, UpdatedPublicRlPolicy,
    accumulate_step_with_baseline, add_actor, checkpoint_digest, hash, propose_parameters,
};
use crate::public_rl_policy_cohort::{COHORT_CONTRACT, ValidatedRlCohort};
use crate::public_rl_policy_episode::EPISODE_CONTRACT;
use crate::public_state_critic::{CONTEXT_CONTRACT, CONTEXT_SCHEMA, PublicStateContext};
use crate::public_stochastic::{RL_SAMPLING_VERSION, RNG_VERSION};
use crate::public_stochastic_native::numerical_target;
use crate::public_stochastic_record::Config;
use crate::replay::{RULES_VERSION, SeatPolicy, catalog_hash};

pub const ARTIFACT_SCHEMA: &str = "tzolkin-public-rl-paired-residual-v1";
pub const TASK: &str = "pairedActorResidualResearch";
pub const POLICY_VERSION: &str = "learned-public-rl-paired-residual-v1";
pub const UPDATE_VERSION: &str = "frozen-residual-actor-ascent-then-raw-mse-v1";

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct Contract {
    schema: &'static str,
    task: &'static str,
    policy_version: &'static str,
    update_version: &'static str,
    actor_versions: [&'static str; 5],
    residual_versions: [&'static str; 2],
    context_contract: &'static str,
    context_schema: u32,
    model_version: &'static str,
    input_contract: &'static str,
    numerical_target: String,
    rules_version: u32,
    catalog_hash: String,
    backend: &'static str,
}
fn contract() -> Contract {
    Contract {
        schema: ARTIFACT_SCHEMA,
        task: TASK,
        policy_version: POLICY_VERSION,
        update_version: UPDATE_VERSION,
        actor_versions: [
            NORMALIZER_VERSION,
            LIKELIHOOD_VERSION,
            CORRECTION_VERSION,
            PULLBACK_VERSION,
            TARGET_CONTRACT,
        ],
        residual_versions: [BASELINE_VERSION, FIT_VERSION],
        context_contract: CONTEXT_CONTRACT,
        context_schema: CONTEXT_SCHEMA,
        model_version: public_model::MODEL_VERSION,
        input_contract: public_model::INPUT_CONTRACT,
        numerical_target: numerical_target(),
        rules_version: RULES_VERSION,
        catalog_hash: catalog_hash(),
        backend: "scalar",
    }
}

/// Both zero means NoChange and leaves count1 untouched. A one-sided numerical
/// change still produces one joint count2 owner, explicitly recording both counts.
#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct PairedStepReport {
    contract: Contract,
    parent_actor_checksum: String,
    parent_residual_checksum: String,
    parent_update_count: u64,
    attempted_update_count: u64,
    root_init_checksum: String,
    bc_source: SeatPolicy,
    cohort_contract: &'static str,
    sampling_version: &'static str,
    rng_version: &'static str,
    cohort_plan_checksum: String,
    cohort_receipt_checksum: String,
    episode_checksums: [String; 4],
    pub(crate) configurations: [Config; 4],
    pub(crate) actor_config: OneStepConfig,
    pub(crate) critic_config: ResidualFitConfig,
    counts: [usize; 3],
    pub(crate) actor_deltas: Deltas,
    pub(crate) actor_numeric: NumericReport,
    mean_raw_residual_mse: f64,
    critic_changed_parameters: usize,
    family_closure: Vec<String>,
    checksum: String,
}
impl PairedStepReport {
    pub fn checksum(&self) -> &str {
        &self.checksum
    }
    pub fn parent_actor_checksum(&self) -> &str {
        &self.parent_actor_checksum
    }
    pub fn parent_residual_checksum(&self) -> &str {
        &self.parent_residual_checksum
    }
    pub fn root_init_checksum(&self) -> &str {
        &self.root_init_checksum
    }
    pub fn cohort_receipt_checksum(&self) -> &str {
        &self.cohort_receipt_checksum
    }
    pub fn counts(&self) -> [usize; 3] {
        self.counts
    }
    pub fn changed_parameters(&self) -> [usize; 2] {
        [
            self.actor_numeric.changed_parameters,
            self.critic_changed_parameters,
        ]
    }
    pub fn mean_raw_residual_mse(&self) -> f64 {
        self.mean_raw_residual_mse
    }
    pub fn family_closure(&self) -> &[String] {
        &self.family_closure
    }
    fn expected_checksum(&self) -> Result<String, String> {
        hash(
            b"tzolkin-public-rl-paired-residual-report-v1\0",
            &(
                &self.contract,
                (
                    &self.parent_actor_checksum,
                    &self.parent_residual_checksum,
                    self.parent_update_count,
                ),
                self.attempted_update_count,
                (&self.root_init_checksum, &self.bc_source),
                (
                    self.cohort_contract,
                    self.sampling_version,
                    self.rng_version,
                ),
                (&self.cohort_plan_checksum, &self.cohort_receipt_checksum),
                (&self.episode_checksums, &self.configurations),
                (self.actor_config, self.critic_config),
                self.counts,
                (&self.actor_deltas, &self.actor_numeric),
                (self.mean_raw_residual_mse, self.critic_changed_parameters),
                &self.family_closure,
            ),
        )
    }
}

/// Serialize-only joint content. It cannot be loaded as a BC, Plain or Long owner.
#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct PairedPublicRlArtifact {
    update_count: u64,
    report: PairedStepReport,
    actor: PublicPolicyModel,
    residual_parameters: Vec<f32>,
    checksum: String,
}
impl PairedPublicRlArtifact {
    pub fn checksum(&self) -> &str {
        &self.checksum
    }
    pub fn update_count(&self) -> u64 {
        self.update_count
    }
    pub fn report(&self) -> &PairedStepReport {
        &self.report
    }
}

/// Only the paired producer constructs this immutable owner. Numeric borrows
/// grant no old-task sampling, deployment or training-history qualification.
/// Future iteration requires a distinct paired collection/cohort contract.
pub struct PairedPublicRlPolicy {
    artifact: PairedPublicRlArtifact,
}
impl PairedPublicRlPolicy {
    pub(crate) fn into_parts(
        self,
    ) -> Result<(PairedStepReport, PublicPolicyModel, Vec<f32>), String> {
        let a = self.artifact;
        let r = &a.report;
        if a.update_count != 2
            || r.parent_update_count != 1
            || r.attempted_update_count != 2
            || serde_json::to_value(&r.contract).map_err(|e| e.to_string())?
                != serde_json::to_value(contract()).map_err(|e| e.to_string())?
            || r.expected_checksum()? != r.checksum
            || r.changed_parameters() == [0, 0]
            || r.family_closure.len() > crate::policy_dataset::MAX_FILES + 8
            || r.family_closure.windows(2).any(|p| p[0] >= p[1])
            || r.family_closure
                .iter()
                .any(|f| split_for_family(f) == Ok(DatasetSplit::Test))
            || a.residual_parameters.len() != crate::public_state_critic::PARAMETER_COUNT
            || a.residual_parameters.iter().any(|p| !p.is_finite())
            || hash(
                b"tzolkin-public-rl-paired-residual-artifact-v1\0",
                &(a.update_count, &a.report, &a.actor, &a.residual_parameters),
            )? != a.checksum
        {
            return Err("Paired bootstrap content/shape/lineage mismatch".into());
        }
        a.actor.validate()?;
        for family in &r.family_closure {
            split_for_family(family)?;
        }
        Ok((a.report, a.actor, a.residual_parameters))
    }
    pub fn artifact(&self) -> &PairedPublicRlArtifact {
        &self.artifact
    }
    pub fn actor_model(&self) -> &PublicPolicyModel {
        &self.artifact.actor
    }
    pub fn residual_parameters(&self) -> &[f32] {
        &self.artifact.residual_parameters
    }
}
pub enum PairedStepOutcome {
    Updated(PairedPublicRlPolicy),
    NoChange(PairedStepReport),
}

/// Bootstrap a distinct count2 joint owner from Count1, not an old Long artifact.
/// All identity, split and budget checks precede forwards. Parent critic output
/// is promoted to f64 and clamped only for actor advantages. The later critic
/// objective is raw MSE against G - f64(1/p), averaged across every callback,
/// including singletons. Actor coefficients/order stay own-SUM, 1/(4p).
/// Any error, including rejected critic descent, returns neither component.
pub fn ascent_paired_from_count1(
    parent: &UpdatedPublicRlPolicy,
    residual: &InitializedResidualBaseline<'_>,
    cohort: &ValidatedRlCohort,
    actor_config: &OneStepConfig,
    critic_config: &ResidualFitConfig,
) -> Result<PairedStepOutcome, String> {
    actor_config.validate()?;
    parent.validate_for_inference()?;
    let initial = residual.initial().artifact();
    initial.validate()?;
    let parent_report = parent.artifact().report();
    let families = bootstrap_lineage(
        (initial.checksum(), initial.bc_source()),
        (
            parent_report.parent_init_checksum(),
            parent_report.bc_source(),
        ),
        parent_report.family_closure(),
        initial
            .family_closure()
            .iter()
            .map(|family| family.family_id()),
        cohort.plan().rollout_families(),
    )?;
    let counts = [
        cohort.total_callbacks(),
        cohort.total_candidate_rows(),
        cohort.total_singletons(),
    ];
    critic_config.check_callbacks(counts[0])?;
    if parent.artifact().checksum() != cohort.plan().parent_artifact_checksum()
        || cohort.plan().parent_update_count() != 1
        || cohort.plan().cohort_contract() != COHORT_CONTRACT
        || cohort.plan().numerical_target() != numerical_target()
        || counts[1] > actor_config.max_candidate_rows()
        || !checkpoint_digest(cohort.plan().checksum())
        || !checkpoint_digest(cohort.receipt_checksum())
        || !checkpoint_digest(residual.checksum())
        || cohort.episodes().iter().any(|episode| {
            episode.parent_artifact_checksum() != parent.artifact().checksum()
                || episode.parent_update_count() != 1
                || episode.episode_contract() != EPISODE_CONTRACT
                || episode.sampling_version() != RL_SAMPLING_VERSION
        })
    {
        return Err("Paired bootstrap parent/cohort/receipt/source/target/budget mismatch".into());
    }
    let mut gradient = vec![0.0; public_model::PARAMETER_COUNT];
    let mut observed = [0usize; 3];
    let mut deltas = Deltas::default();
    // Exact Plain accumulation order; the only changed input is the frozen baseline.
    for (member, episode) in cohort.episodes().iter().enumerate() {
        let players = episode.source_config().players();
        for actor in 0..players {
            let mut own = vec![0.0; public_model::PARAMETER_COUNT];
            for step in episode.steps().filter(|step| step.actor() == actor) {
                if step.observation().players.len() != players {
                    return Err("Paired actor observation/player count mismatch".into());
                }
                accumulate_step_with_baseline(
                    parent.model(),
                    step,
                    actor_config,
                    &mut own,
                    &mut observed,
                    &mut deltas,
                    |observation| {
                        residual.baseline(&PublicStateContext::from_observation(observation)?)
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
        return Err("Paired actor applied counts differ from the sealed receipt".into());
    }
    let actor = propose_parameters(parent.model().parameters(), &gradient, actor_config)?;
    // All actor work precedes fitting, even when its rounded proposal is NoChange.
    let critic = fit_applied(
        residual.parameters(),
        cohort.episodes().iter().flat_map(|episode| episode.steps()),
        counts[0],
        critic_config,
    )?;
    let mut report = PairedStepReport {
        contract: contract(),
        parent_actor_checksum: parent.artifact().checksum().into(),
        parent_residual_checksum: residual.checksum().into(),
        parent_update_count: 1,
        attempted_update_count: 2,
        root_init_checksum: initial.checksum().into(),
        bc_source: initial.bc_source().clone(),
        cohort_contract: COHORT_CONTRACT,
        sampling_version: RL_SAMPLING_VERSION,
        rng_version: RNG_VERSION,
        cohort_plan_checksum: cohort.plan().checksum().into(),
        cohort_receipt_checksum: cohort.receipt_checksum().into(),
        episode_checksums: cohort
            .episodes()
            .each_ref()
            .map(|episode| episode.canonical_record_checksum().into()),
        configurations: cohort
            .plan()
            .configurations()
            .each_ref()
            .map(Config::from_config),
        actor_config: *actor_config,
        critic_config: *critic_config,
        counts,
        actor_deltas: deltas,
        actor_numeric: actor.numeric,
        mean_raw_residual_mse: critic.mean_loss,
        critic_changed_parameters: critic.changed_parameters,
        family_closure: families,
        checksum: String::new(),
    };
    report.checksum = report.expected_checksum()?;
    if report.changed_parameters() == [0, 0] {
        return Ok(PairedStepOutcome::NoChange(report));
    }
    let actor = PublicPolicyModel {
        parameters: actor.parameters,
    };
    actor.validate()?;
    let checksum = hash(
        b"tzolkin-public-rl-paired-residual-artifact-v1\0",
        &(2u64, &report, &actor, &critic.parameters),
    )?;
    Ok(PairedStepOutcome::Updated(PairedPublicRlPolicy {
        artifact: PairedPublicRlArtifact {
            update_count: 2,
            report,
            actor,
            residual_parameters: critic.parameters,
            checksum,
        },
    }))
}

fn bootstrap_lineage<'a>(
    initial: (&str, &SeatPolicy),
    parent: (&str, &SeatPolicy),
    inherited: &[String],
    initial_families: impl Iterator<Item = &'a str>,
    rollout: &[String; 4],
) -> Result<Vec<String>, String> {
    if !checkpoint_digest(initial.0)
        || initial.0 != parent.0
        || initial.1 != parent.1
        || inherited.is_empty()
        || inherited.len() > crate::policy_dataset::MAX_FILES + 4
        || inherited.windows(2).any(|pair| pair[0] >= pair[1])
    {
        return Err("Paired bootstrap root/BC/bounded lineage mismatch".into());
    }
    let mut families = inherited.iter().cloned().collect::<BTreeSet<_>>();
    for family in inherited {
        if split_for_family(family)? == DatasetSplit::Test {
            return Err("Paired bootstrap inherited Test family".into());
        }
    }
    for family in initial_families {
        if !families.contains(family) {
            return Err("Paired bootstrap parent omits initialization BC closure".into());
        }
    }
    for family in rollout {
        if split_for_family(family)? != DatasetSplit::Train || !families.insert(family.clone()) {
            return Err("Paired bootstrap requires four distinct fresh Train families".into());
        }
    }
    Ok(families.into_iter().collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dataset::seed_family_id;
    #[test]
    fn bootstrap_lineage_rejects_wrong_root_bc_closure_and_rollout() {
        let model = crate::public_native::integration_fixture::model(false);
        let source = crate::public_native::integration_fixture::handle(&model)
            .provenance()
            .clone();
        let root = "a".repeat(64);
        let bc = [seed_family_id(0), seed_family_id(3)];
        let inherited = bc
            .iter()
            .cloned()
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect::<Vec<_>>();
        let rollout: [String; 4] = (1..100)
            .map(seed_family_id)
            .filter(|f| !bc.contains(f) && split_for_family(f).unwrap() == DatasetSplit::Train)
            .take(4)
            .collect::<Vec<_>>()
            .try_into()
            .unwrap();
        let check = |parent_root: &str,
                     parent_source: &SeatPolicy,
                     closure: &[String],
                     rollout: &[String; 4]| {
            bootstrap_lineage(
                (&root, &source),
                (parent_root, parent_source),
                closure,
                bc.iter().map(String::as_str),
                rollout,
            )
        };
        assert_eq!(
            check(&root, &source, &inherited, &rollout).unwrap().len(),
            6
        );
        assert!(check(&"b".repeat(64), &source, &inherited, &rollout).is_err());
        let mut wrong_source = source.clone();
        if let SeatPolicy::PublicLearned { model_checksum, .. } = &mut wrong_source {
            *model_checksum = "c".repeat(64);
        }
        assert!(check(&root, &wrong_source, &inherited, &rollout).is_err());
        assert!(check(&root, &source, &inherited[..1], &rollout).is_err());
        for replacement in [bc[0].clone(), seed_family_id(3), rollout[1].clone()] {
            let mut bad = rollout.clone();
            bad[0] = replacement;
            assert!(check(&root, &source, &inherited, &bad).is_err());
        }
    }
}
