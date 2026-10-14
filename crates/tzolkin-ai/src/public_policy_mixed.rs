//! One sealed mixed learner joint update; no durable continuation admission.
use crate::dataset::{DatasetSplit, split_for_family};
use crate::human_policy_dataset::ConditionalFeatureProfile;
use crate::public_model::{self, PublicPolicyModel};
use crate::public_policy_baseline::{self, InitializedResidualBaseline, ResidualFitConfig};
use crate::public_policy_episode::AppliedActorStep;
use crate::public_policy_paired_long::PairedRlParent;
use crate::public_policy_update::{
    self, Deltas, NumericActorStep, NumericReport, OneStepConfig, PhaseTimer, TimedPhase, Untimed,
    hash, timed,
};
use crate::public_rl_artifact::InitializedPublicRlPolicy;
use crate::public_rl_mixed_cohort::{COHORT_CONTRACT, ValidatedMixedRlCohort};
use crate::public_rl_mixed_episode::EPISODE_CONTRACT;
use crate::public_rl_mixed_session::ImportedMixedRlParent;
use crate::public_state_critic::PublicStateContext;
use crate::public_stochastic_native::numerical_target;
use crate::replay::SeatPolicy;
use crate::semantic_playing_rl_collection::SemanticPlayingActorStep;
use serde::{Deserialize, Serialize};

pub const ARTIFACT_SCHEMA: &str = "tzolkin-public-rl-mixed-joint-one-batch-v1";
pub const TASK: &str = "mixedLearnerOwnSumQuarterJointOneBatchV1";
pub const NORMALIZER_VERSION: &str = "four-mixed-games-learner-own-sum-quarter-v1";

fn contract() -> serde_json::Value {
    serde_json::json!({"schema":ARTIFACT_SCHEMA,"task":TASK,"normalizer":NORMALIZER_VERSION,
        "model":public_model::MODEL_VERSION,"input":public_model::INPUT_CONTRACT,
        "critic":crate::public_state_critic::MODEL_VERSION,"context":crate::public_state_critic::CONTEXT_CONTRACT,
        "update":crate::public_policy_paired::UPDATE_VERSION,"fit":public_policy_baseline::FIT_VERSION,
        "likelihood":crate::public_policy_likelihood::LIKELIHOOD_VERSION,
        "correction":crate::public_policy_likelihood::CORRECTION_VERSION,
        "pullback":crate::public_policy_pullback::PULLBACK_VERSION,
        "target":crate::public_policy_episode::TARGET_CONTRACT,"episode":EPISODE_CONTRACT,
        "cohort":COHORT_CONTRACT,"numericalTarget":numerical_target(),"catalog":crate::replay::catalog_hash(),
        "rules":crate::replay::RULES_VERSION,"featureSchema":crate::features::PUBLIC_FEATURE_SCHEMA,
        "rng":crate::public_stochastic::RNG_VERSION,"backend":"scalar","continuationAvailable":false})
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub(crate) struct MixedRoot {
    pub(crate) import_identity: serde_json::Value,
    pub(crate) parent_checksum: String,
    pub(crate) parent_count: u64,
    root_init_checksum: String,
    root_count1_checksum: String,
    imported_lineage_checksum: String,
    parent_residual_checksum: String,
    residual_seed: u64,
    bc_source: SeatPolicy,
    trained_families: Vec<String>,
}
pub(crate) fn root(
    imported: &ImportedMixedRlParent,
    initial: &InitializedPublicRlPolicy,
    seed: u64,
) -> Result<(MixedRoot, Vec<f32>), String> {
    initial.artifact().validate()?;
    let (checksum, count, root_init, count1, lineage, used_seed, source, families, residual) =
        match imported.parent() {
            PairedRlParent::Count1(p) => {
                p.validate_for_inference()?;
                let r = p.artifact().report();
                (
                    p.artifact().checksum(),
                    1,
                    r.parent_init_checksum(),
                    p.artifact().checksum(),
                    hash(
                        b"tzolkin-mixed-imported-count1-lineage-v1\0",
                        &(p.artifact().checksum(), r.family_closure()),
                    )?,
                    seed,
                    r.bc_source(),
                    r.family_closure().to_vec(),
                    InitializedResidualBaseline::new(initial, seed)?
                        .parameters()
                        .to_vec(),
                )
            }
            PairedRlParent::Paired(p) => {
                p.validate_for_sampling()?;
                let r = p.artifact().report();
                if seed != r.residual_seed() {
                    return Err("Mixed spec residual seed differs from imported Paired root".into());
                }
                (
                    p.artifact().checksum(),
                    p.artifact().update_count(),
                    r.root_init_checksum(),
                    r.root_count1_checksum(),
                    r.lineage_checksum().to_owned(),
                    r.residual_seed(),
                    r.bc_source(),
                    p.family_set().iter().cloned().collect(),
                    p.residual_parameters().to_vec(),
                )
            }
        };
    if root_init != initial.artifact().checksum() || source != initial.artifact().bc_source() {
        return Err("Mixed update initialization/root source mismatch".into());
    }
    Ok((
        MixedRoot {
            import_identity: imported.import_identity().clone(),
            parent_checksum: checksum.into(),
            parent_count: count,
            root_init_checksum: root_init.into(),
            root_count1_checksum: count1.into(),
            imported_lineage_checksum: lineage,
            parent_residual_checksum: hash(b"tzolkin-mixed-frozen-residual-v1\0", &residual)?,
            residual_seed: used_seed,
            bc_source: source.clone(),
            trained_families: families,
        },
        residual,
    ))
}
pub(crate) fn parent_model(imported: &ImportedMixedRlParent) -> &PublicPolicyModel {
    match imported.parent() {
        PairedRlParent::Count1(p) => p.model(),
        PairedRlParent::Paired(p) => p.actor_model(),
    }
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct MixedJointReport {
    contract: serde_json::Value,
    root: MixedRoot,
    pub(crate) plan_checksum: String,
    pub(crate) cohort_checksum: String,
    pub(crate) episode_checksums: [String; 4],
    pub(crate) learner_seats: [usize; 4],
    pub(crate) configurations: [crate::public_stochastic_record::Config; 4],
    pub(crate) roster_checksums: [String; 4],
    pub(crate) families: [String; 4],
    pub(crate) counts: [usize; 3],
    pub(crate) observed_callbacks: usize,
    actor_config: serde_json::Value,
    critic_config: serde_json::Value,
    deltas: Deltas,
    numeric: NumericReport,
    critic_changed_parameters: usize,
    mean_raw_residual_mse: f64,
    pub(crate) checksum: String,
}
impl MixedJointReport {
    pub fn checksum(&self) -> &str {
        &self.checksum
    }
    pub fn changed_parameters(&self) -> [usize; 2] {
        [
            self.numeric.changed_parameters,
            self.critic_changed_parameters,
        ]
    }
    pub fn learner_counts(&self) -> [usize; 3] {
        self.counts
    }
    fn expected_checksum(&self) -> Result<String, String> {
        let mut v = serde_json::to_value(self).map_err(|e| e.to_string())?;
        v.as_object_mut()
            .ok_or("Mixed report shape")?
            .remove("checksum");
        hash(b"tzolkin-mixed-joint-report-v1\0", &v)
    }
    pub(crate) fn validate(
        &self,
        root: &MixedRoot,
        actor: &OneStepConfig,
        critic: &ResidualFitConfig,
    ) -> Result<(), String> {
        let mut families = self.families.to_vec();
        families.sort();
        if self.contract != contract()
            || &self.root != root
            || self.expected_checksum()? != self.checksum
            || self.actor_config != serde_json::to_value(actor).map_err(|e| e.to_string())?
            || self.critic_config != serde_json::to_value(critic).map_err(|e| e.to_string())?
            || self.observed_callbacks < self.counts[0]
            || self.observed_callbacks > 4 * crate::public_stochastic::MAX_SAMPLES
            || self.critic_changed_parameters > crate::public_state_critic::PARAMETER_COUNT
            || !self.mean_raw_residual_mse.is_finite()
            || self.mean_raw_residual_mse < 0.0
            || self
                .roster_checksums
                .iter()
                .any(|s| !public_policy_update::checkpoint_digest(s))
        {
            return Err("Mixed report contract/root/numeric mismatch".into());
        }
        critic.check_callbacks(self.counts[0])?;
        public_policy_update::validate_joint_checkpoint_report(
            &self.counts,
            &self.deltas,
            &self.numeric,
            actor,
            (
                [
                    &root.root_init_checksum,
                    &self.plan_checksum,
                    &self.cohort_checksum,
                    &self.checksum,
                ],
                &self.episode_checksums,
            ),
            &families,
            &root.bc_source,
        )?;
        let configs = self
            .configurations
            .iter()
            .map(|c| c.checked_config())
            .collect::<Result<Vec<_>, _>>()?;
        if configs.iter().filter(|c| c.players() == 3).count() != 2
            || configs.iter().filter(|c| c.players() == 4).count() != 2
        {
            return Err("Mixed report whole-four shape".into());
        }
        for (i, c) in configs.iter().enumerate() {
            if self.learner_seats[i] >= c.players()
                || crate::dataset::seed_family_id(c.environment_seed()) != self.families[i]
                || split_for_family(&self.families[i])? != DatasetSplit::Train
                || root.trained_families.contains(&self.families[i])
                || self.families[..i].contains(&self.families[i])
            {
                return Err("Mixed report learner/config/family mismatch".into());
            }
        }
        Ok(())
    }
}
/// Content DTO alone admits no owner; only journal-checked restore can mint one.
#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct MixedJointArtifact {
    imported_update_count: u64,
    mixed_update_count: u64,
    report: MixedJointReport,
    actor: PublicPolicyModel,
    residual_parameters: Vec<f32>,
    checksum: String,
}
impl MixedJointArtifact {
    pub fn checksum(&self) -> &str {
        &self.checksum
    }
    pub fn report(&self) -> &MixedJointReport {
        &self.report
    }
    fn expected_checksum(&self) -> Result<String, String> {
        hash(
            b"tzolkin-mixed-joint-artifact-v1\0",
            &(
                self.imported_update_count,
                self.mixed_update_count,
                &self.report,
                &self.actor,
                &self.residual_parameters,
            ),
        )
    }
}
pub struct MixedJointOwner {
    artifact: MixedJointArtifact,
}
impl MixedJointOwner {
    pub(crate) fn continued_root_reference(
        &self,
    ) -> Result<crate::public_policy_mixed_continued::MixedRootReference, String> {
        Ok(crate::public_policy_mixed_continued::MixedRootReference {
            original_root_checksum: hash(
                b"tzolkin-continued-mixed-original-root-v2\0",
                &self.artifact.report.root,
            )?,
            first_mixed_checksum: self.artifact.checksum.clone(),
            root_init_checksum: self.artifact.report.root.root_init_checksum.clone(),
            imported_update_count: self.imported_update_count(),
            residual_seed: self.artifact.report.root.residual_seed,
            bc_source: self.artifact.report.root.bc_source.clone(),
        })
    }
    pub fn artifact(&self) -> &MixedJointArtifact {
        &self.artifact
    }
    pub fn actor_model(&self) -> &PublicPolicyModel {
        &self.artifact.actor
    }
    pub fn residual_parameters(&self) -> &[f32] {
        &self.artifact.residual_parameters
    }
    pub fn mixed_update_count(&self) -> u64 {
        self.artifact.mixed_update_count
    }
    pub fn imported_update_count(&self) -> u64 {
        self.artifact.imported_update_count
    }
    pub(crate) fn family_closure(&self) -> impl Iterator<Item = &String> {
        self.artifact
            .report
            .root
            .trained_families
            .iter()
            .chain(self.artifact.report.families.iter())
    }
    pub(crate) fn validate_for_inference(&self) -> Result<(), String> {
        let a = &self.artifact;
        a.actor.validate()?;
        if a.report.contract != contract()
            || a.imported_update_count != a.report.root.parent_count
            || !(1..=crate::public_policy_paired_long::MAX_UPDATES)
                .contains(&a.imported_update_count)
            || a.mixed_update_count != 1
            || a.report.changed_parameters() == [0, 0]
            || a.report.expected_checksum()? != a.report.checksum
            || a.expected_checksum()? != a.checksum
            || a.residual_parameters.len() != crate::public_state_critic::PARAMETER_COUNT
            || a.residual_parameters.iter().any(|x| !x.is_finite())
        {
            return Err("Mixed inference requires the genuine one-batch joint owner".into());
        }
        Ok(())
    }
}
pub(crate) enum MixedProposal {
    Updated(MixedJointOwner),
    NoChange(MixedJointReport),
}
impl MixedProposal {
    pub(crate) fn report(&self) -> &MixedJointReport {
        match self {
            Self::Updated(p) => p.artifact().report(),
            Self::NoChange(r) => r,
        }
    }
}
pub(crate) fn restore_owner(
    artifact: MixedJointArtifact,
    root: &MixedRoot,
    actor: &OneStepConfig,
    critic: &ResidualFitConfig,
    old_actor: &PublicPolicyModel,
    old_residual: &[f32],
) -> Result<MixedJointOwner, String> {
    artifact.report.validate(root, actor, critic)?;
    artifact.actor.validate()?;
    if artifact.imported_update_count != root.parent_count
        || artifact.mixed_update_count != 1
        || artifact.report.changed_parameters() == [0, 0]
        || artifact.expected_checksum()? != artifact.checksum
        || artifact.residual_parameters.len() != crate::public_state_critic::PARAMETER_COUNT
        || artifact.residual_parameters.iter().any(|x| !x.is_finite())
    {
        return Err("Mixed joint owner identity/content mismatch".into());
    }
    let actor_meta = serde_json::to_value(actor).map_err(|e| e.to_string())?;
    let critic_meta = serde_json::to_value(critic).map_err(|e| e.to_string())?;
    let numeric = serde_json::to_value(&artifact.report.numeric).map_err(|e| e.to_string())?;
    let (changed, l2, max) = rounded_delta(
        old_actor.parameters(),
        artifact.actor.parameters(),
        actor_meta["maxAbsActualDelta"]
            .as_f64()
            .ok_or("Mixed actor delta cap")?,
    )?;
    let (critic_changed, _, _) = rounded_delta(
        old_residual,
        &artifact.residual_parameters,
        critic_meta["maxAbsActualDelta"]
            .as_f64()
            .ok_or("Mixed critic delta cap")?,
    )?;
    if changed != artifact.report.numeric.changed_parameters
        || critic_changed != artifact.report.critic_changed_parameters
        || numeric["actualDeltaL2"].as_f64() != Some(l2)
        || numeric["actualDeltaMaxAbs"].as_f64() != Some(max)
    {
        return Err("Mixed restored candidate actual rounded deltas differ".into());
    }
    Ok(MixedJointOwner { artifact })
}
pub(crate) fn rounded_delta(
    old: &[f32],
    new: &[f32],
    cap: f64,
) -> Result<(usize, f64, f64), String> {
    if old.len() != new.len() || old.is_empty() || !cap.is_finite() || cap < 0.0 {
        return Err("Mixed delta shape".into());
    }
    let mut changed = 0;
    let mut sum = 0.0;
    let mut max: f64 = 0.0;
    for (&a, &b) in old.iter().zip(new) {
        let d = f64::from(b) - f64::from(a);
        if !a.is_finite()
            || !b.is_finite()
            || d.abs() > cap
            || (a == b && a.to_bits() != b.to_bits())
        {
            return Err("Mixed actual rounded delta exceeds bound".into());
        }
        changed += usize::from(a.to_bits() != b.to_bits());
        sum += d * d;
        max = max.max(d.abs());
    }
    if !sum.is_finite() {
        return Err("Mixed nonfinite rounded delta norm".into());
    }
    Ok((changed, sum.sqrt(), max))
}
pub(crate) fn prepare(
    imported: &ImportedMixedRlParent,
    root: MixedRoot,
    residual: &[f32],
    cohort: &ValidatedMixedRlCohort,
    actor_config: &OneStepConfig,
    critic_config: &ResidualFitConfig,
) -> Result<MixedProposal, String> {
    actor_config.validate()?;
    let counts = [
        cohort.total_learner_callbacks(),
        cohort.total_learner_candidate_rows(),
        cohort.total_learner_singletons(),
    ];
    critic_config.check_callbacks(counts[0])?;
    if cohort.plan().parent_artifact_checksum() != root.parent_checksum
        || cohort.plan().parent_update_count() != root.parent_count
        || counts[1] > actor_config.max_candidate_rows()
        || hash(b"tzolkin-mixed-frozen-residual-v1\0", &residual)? != root.parent_residual_checksum
    {
        return Err("Mixed update immediate parent/frozen residual/count mismatch".into());
    }
    for e in cohort.episodes() {
        if e.numerical_target() != numerical_target()
            || e.episode_contract() != EPISODE_CONTRACT
            || split_for_family(e.family_id())? != DatasetSplit::Train
            || root.trained_families.iter().any(|f| f == e.family_id())
        {
            return Err("Mixed update source target/family mismatch".into());
        }
    }
    let model = parent_model(imported);
    let members = cohort
        .episodes()
        .iter()
        .enumerate()
        .map(|(i, e)| {
            Ok(MixedNumericMember {
                learner: e.learner_seat(),
                players: e.source_config().players(),
                coefficient: cohort.actor_coefficient(i, e.learner_seat())?,
                steps: e.steps().collect(),
            })
        })
        .collect::<Result<Vec<_>, String>>()?
        .try_into()
        .map_err(|_| "Mixed numeric whole-four shape")?;
    let (deltas, actor, critic) = prepare_numeric(
        model,
        residual,
        &members,
        counts,
        actor_config,
        critic_config,
    )?;
    let mut report = MixedJointReport {
        contract: contract(),
        root,
        plan_checksum: cohort.plan().checksum().into(),
        cohort_checksum: cohort.receipt_checksum().into(),
        episode_checksums: cohort
            .episodes()
            .each_ref()
            .map(|e| e.canonical_record_checksum().into()),
        learner_seats: cohort.plan().learner_seats(),
        configurations: cohort
            .plan()
            .configurations()
            .each_ref()
            .map(crate::public_stochastic_record::Config::from_config),
        roster_checksums: cohort
            .episodes()
            .each_ref()
            .map(|e| e.frozen_roster_checksum().into()),
        families: cohort.plan().rollout_families().clone(),
        counts,
        observed_callbacks: cohort.total_observed_callbacks(),
        actor_config: serde_json::to_value(actor_config).map_err(|e| e.to_string())?,
        critic_config: serde_json::to_value(critic_config).map_err(|e| e.to_string())?,
        deltas,
        numeric: actor.numeric,
        critic_changed_parameters: critic.changed_parameters,
        mean_raw_residual_mse: critic.mean_loss,
        checksum: String::new(),
    };
    report.checksum = report.expected_checksum()?;
    report.validate(&report.root, actor_config, critic_config)?;
    if report.changed_parameters() == [0, 0] {
        return Ok(MixedProposal::NoChange(report));
    }
    let mut artifact = MixedJointArtifact {
        imported_update_count: report.root.parent_count,
        mixed_update_count: 1,
        report,
        actor: PublicPolicyModel {
            parameters: actor.parameters,
        },
        residual_parameters: critic.parameters,
        checksum: String::new(),
    };
    artifact.checksum = artifact.expected_checksum()?;
    let bound_root = artifact.report.root.clone();
    Ok(MixedProposal::Updated(restore_owner(
        artifact,
        &bound_root,
        actor_config,
        critic_config,
        model,
        residual,
    )?))
}
pub(crate) struct NumericMember<S> {
    pub(crate) learner: usize,
    pub(crate) players: usize,
    pub(crate) coefficient: f64,
    pub(crate) steps: Vec<S>,
}
pub(crate) type MixedNumericMember<'a> = NumericMember<AppliedActorStep<'a>>;
pub(crate) type SemanticNumericMember<'a> = NumericMember<SemanticPlayingActorStep<'a>>;
pub(crate) fn prepare_numeric(
    model: &PublicPolicyModel,
    residual: &[f32],
    members: &[MixedNumericMember<'_>; 4],
    counts: [usize; 3],
    actor_config: &OneStepConfig,
    critic_config: &ResidualFitConfig,
) -> Result<
    (
        Deltas,
        public_policy_update::Proposal,
        public_policy_baseline::NumericResidualFit,
    ),
    String,
> {
    prepare_numeric_timed(
        model,
        residual,
        members,
        counts,
        actor_config,
        critic_config,
        &mut Untimed,
    )
}

pub(crate) fn prepare_numeric_timed<T: PhaseTimer>(
    model: &PublicPolicyModel,
    residual: &[f32],
    members: &[MixedNumericMember<'_>; 4],
    counts: [usize; 3],
    actor_config: &OneStepConfig,
    critic_config: &ResidualFitConfig,
    timer: &mut T,
) -> Result<
    (
        Deltas,
        public_policy_update::Proposal,
        public_policy_baseline::NumericResidualFit,
    ),
    String,
> {
    prepare_numeric_timed_for_profile(
        ConditionalFeatureProfile::Public2,
        model,
        residual,
        members,
        counts,
        actor_config,
        critic_config,
        timer,
    )
}

/// A numeric proposal only. The profile does not admit an original source or
/// alter the actor-own SUM/4, frozen public384 critic or rounded-delta contract.
#[allow(clippy::too_many_arguments)]
pub(crate) fn prepare_numeric_timed_for_profile<T: PhaseTimer, S: NumericActorStep>(
    profile: ConditionalFeatureProfile,
    model: &PublicPolicyModel,
    residual: &[f32],
    members: &[NumericMember<S>; 4],
    counts: [usize; 3],
    actor_config: &OneStepConfig,
    critic_config: &ResidualFitConfig,
    timer: &mut T,
) -> Result<
    (
        Deltas,
        public_policy_update::Proposal,
        public_policy_baseline::NumericResidualFit,
    ),
    String,
> {
    actor_config.validate()?;
    critic_config.check_callbacks(counts[0])?;
    model.validate()?;
    if counts[1] > actor_config.max_candidate_rows()
        || residual.len() != crate::public_state_critic::PARAMETER_COUNT
        || residual.iter().any(|x| !x.is_finite())
    {
        return Err("Mixed numeric current actor/residual/count bound".into());
    }
    let mut gradient = vec![0.0; public_model::PARAMETER_COUNT];
    let mut observed = [0; 3];
    let mut deltas = Deltas::default();
    for (index, e) in members.iter().enumerate() {
        timer.actor_member(index, e.learner, e.players);
        let mut own = vec![0.0; public_model::PARAMETER_COUNT];
        for step in &e.steps {
            if step.actor() != e.learner || step.observation().players.len() != e.players {
                return Err("Mixed update opponent/absolute learner actor mismatch".into());
            }
            public_policy_update::accumulate_step_with_baseline_timed_for_profile(
                profile,
                model,
                step,
                actor_config,
                &mut own,
                &mut observed,
                &mut deltas,
                |o| {
                    public_policy_baseline::baseline(
                        residual,
                        &PublicStateContext::from_observation(o)?,
                    )
                },
                timer,
            )?;
        }
        timed(timer, TimedPhase::ActorAggregate, |_| {
            add_learner_actor(
                &mut gradient,
                &own,
                e.learner,
                e.players,
                e.learner,
                e.coefficient,
            )
        })?;
    }
    if observed != counts {
        return Err("Mixed update sealed learner count mismatch".into());
    }
    let actor = timed(timer, TimedPhase::ActorProposal, |_| {
        public_policy_update::propose_parameters(model.parameters(), &gradient, actor_config)
    })?;
    let critic = timed(timer, TimedPhase::CriticFit, |_| {
        public_policy_baseline::fit_applied(
            residual,
            members.iter().flat_map(|e| e.steps.iter()),
            counts[0],
            critic_config,
        )
    })?;
    Ok((deltas, actor, critic))
}
fn add_learner_actor(
    total: &mut [f64],
    own: &[f64],
    learner: usize,
    players: usize,
    actor: usize,
    coefficient: f64,
) -> Result<(), String> {
    if ![3, 4].contains(&players) || learner >= players || actor != learner || coefficient != 0.25 {
        return Err("Mixed aggregation requires absolute learner own-SUM/4".into());
    }
    public_policy_update::add_actor(total, own, coefficient)
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn absolute_learner_sum_quarter_and_restored_rounded_boundaries() {
        let mut total = [0.0, 0.0];
        for (players, seat, own) in [
            (3, 2, [4.0, 8.0]),
            (3, 1, [8.0, 0.0]),
            (4, 3, [0.0, 4.0]),
            (4, 2, [4.0, 4.0]),
        ] {
            add_learner_actor(&mut total, &own, seat, players, seat, 0.25).unwrap();
            assert!(
                add_learner_actor(&mut total, &own, seat, players, (seat + 1) % players, 0.25)
                    .is_err()
            );
        }
        assert_eq!(total, [4.0, 4.0]);
        assert!(add_learner_actor(&mut total, &[1.0, 1.0], 2, 4, 2, 1.0 / 16.0).is_err());
        assert_eq!(
            rounded_delta(&[1.0, 2.0], &[1.0, 2.0], 0.01).unwrap(),
            (0, 0.0, 0.0)
        );
        assert!(rounded_delta(&[1.0], &[1.02], 0.01).is_err());
        assert!(rounded_delta(&[0.0], &[-0.0], 0.01).is_err());
        // Neither layout may turn absent applied steps into an admitted update.
        let model = crate::public_model::PublicPolicyArtifact::new(17)
            .unwrap()
            .model;
        let residual = public_policy_baseline::zero_output_parameters(17);
        let members = std::array::from_fn(|i| MixedNumericMember {
            learner: i % if i < 2 { 3 } else { 4 },
            players: if i < 2 { 3 } else { 4 },
            coefficient: 0.25,
            steps: Vec::new(),
        });
        let actor = OneStepConfig::new(1e-4, 0.01, 4).unwrap();
        let critic = ResidualFitConfig::new(1e-4, 0.01, 4).unwrap();
        for profile in [
            ConditionalFeatureProfile::Public2,
            ConditionalFeatureProfile::SemanticPlaying4,
        ] {
            assert_eq!(
                prepare_numeric_timed_for_profile(
                    profile,
                    &model,
                    &residual,
                    &members,
                    [4, 4, 0],
                    &actor,
                    &critic,
                    &mut Untimed
                )
                .err()
                .as_deref(),
                Some("Mixed update sealed learner count mismatch")
            );
        }
    }
}
