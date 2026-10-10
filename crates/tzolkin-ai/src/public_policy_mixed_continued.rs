//! Count-neutral mixed proposals and immutable current-parent inference.
//! A proposal or artifact DTO never admits an executable continued owner.
//! Durable promotion and journal-checked restoration are separate boundaries.
use crate::Decision;
use crate::dataset::{DatasetSplit, split_for_family};
use crate::features::FeatureEncoder;
use crate::kernel::{Kernel, ResolvedKernel};
use crate::public_model::{self, PublicPolicyModel, ValueValidity};
use crate::public_policy_baseline::ResidualFitConfig;
use crate::public_policy_mixed::{
    MixedJointOwner, MixedNumericMember, prepare_numeric, rounded_delta,
};
use crate::public_policy_update::{
    self, Deltas, NumericReport, OneStepConfig, checkpoint_digest, hash,
};
use crate::public_rl_current_mixed_source::{
    self as source, CurrentMixedParentIdentity, ValidatedCurrentMixedCohort,
};
use crate::public_rl_mixed_handle::MixedPolicyHandle;
use crate::public_stochastic_native::numerical_target;
use crate::replay::SeatPolicy;
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;
use tzolkin_core::observation::Observation;

pub const ARTIFACT_SCHEMA: &str = "tzolkin-public-rl-continued-mixed-joint-v2";
pub const REPORT_SCHEMA: &str = "tzolkin-public-rl-continued-mixed-joint-report-v2";
pub const NORMALIZER_VERSION: &str = "four-current-mixed-games-learner-own-sum-quarter-v2";

fn contract() -> serde_json::Value {
    serde_json::json!({"schema":REPORT_SCHEMA,"artifact":ARTIFACT_SCHEMA,"task":source::CONTINUED_TASK,
        "normalizer":NORMALIZER_VERSION,"model":public_model::MODEL_VERSION,"input":public_model::INPUT_CONTRACT,
        "critic":crate::public_state_critic::MODEL_VERSION,"context":crate::public_state_critic::CONTEXT_CONTRACT,
        "update":crate::public_policy_paired::UPDATE_VERSION,"fit":crate::public_policy_baseline::FIT_VERSION,
        "likelihood":crate::public_policy_likelihood::LIKELIHOOD_VERSION,"correction":crate::public_policy_likelihood::CORRECTION_VERSION,
        "pullback":crate::public_policy_pullback::PULLBACK_VERSION,"target":crate::public_policy_episode::TARGET_CONTRACT,
        "episode":source::EPISODE_CONTRACT,"cohort":source::COHORT_CONTRACT,"numericalTarget":numerical_target(),
        "catalog":crate::replay::catalog_hash(),"rules":crate::replay::RULES_VERSION,
        "featureSchema":crate::features::PUBLIC_FEATURE_SCHEMA,"rng":crate::public_stochastic::RNG_VERSION,"backend":"scalar",
        "numericProposalOnly":true,"durableOwnerAdmissionAvailable":false})
}

/// Constant root references; no growing family/history list is serialized here.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub(crate) struct MixedRootReference {
    pub(crate) original_root_checksum: String,
    pub(crate) first_mixed_checksum: String,
    pub(crate) root_init_checksum: String,
    pub(crate) imported_update_count: u64,
    pub(crate) residual_seed: u64,
    pub(crate) bc_source: SeatPolicy,
}
fn next_count(parent: &CurrentMixedParentIdentity) -> Result<u64, String> {
    parent.validate()?;
    parent
        .mixed_update_count()
        .checked_add(1)
        .filter(|&n| (2..=source::MAX_MIXED_UPDATES).contains(&n))
        .ok_or_else(|| "Continued mixed successful-update ceiling".into())
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct ContinuedMixedReport {
    contract: serde_json::Value,
    root: MixedRootReference,
    parent: CurrentMixedParentIdentity,
    proposed_mixed_update_count: u64,
    plan_checksum: String,
    cohort_checksum: String,
    episode_checksums: [String; 4],
    learner_seats: [usize; 4],
    configurations: [crate::public_stochastic_record::Config; 4],
    roster_checksums: [String; 4],
    families: [String; 4],
    counts: [usize; 3],
    observed_callbacks: usize,
    actor_config: serde_json::Value,
    critic_config: serde_json::Value,
    deltas: Deltas,
    numeric: NumericReport,
    critic_changed_parameters: usize,
    mean_raw_residual_mse: f64,
    checksum: String,
}
impl ContinuedMixedReport {
    pub fn checksum(&self) -> &str {
        &self.checksum
    }
    pub fn parent_identity(&self) -> &CurrentMixedParentIdentity {
        &self.parent
    }
    pub fn proposed_mixed_update_count(&self) -> u64 {
        self.proposed_mixed_update_count
    }
    pub fn learner_counts(&self) -> [usize; 3] {
        self.counts
    }
    pub fn changed_parameters(&self) -> [usize; 2] {
        [
            self.numeric.changed_parameters,
            self.critic_changed_parameters,
        ]
    }
    pub fn families(&self) -> &[String; 4] {
        &self.families
    }
    pub(crate) fn plan_checksum(&self) -> &str {
        &self.plan_checksum
    }
    pub(crate) fn require_saved_cohort(
        &self,
        plan: &serde_json::Value,
        cohort: &serde_json::Value,
        actor: &OneStepConfig,
        critic: &ResidualFitConfig,
    ) -> Result<(), String> {
        self.validate_content()?;
        if plan["checksum"] != self.plan_checksum
            || plan["families"]
                != serde_json::to_value(&self.families).map_err(|e| e.to_string())?
            || cohort["checksum"] != self.cohort_checksum
            || cohort["planChecksum"] != self.plan_checksum
            || cohort["episodeChecksums"]
                != serde_json::to_value(&self.episode_checksums).map_err(|e| e.to_string())?
            || cohort["learnerCounts"]
                != serde_json::to_value(self.counts).map_err(|e| e.to_string())?
            || cohort["observedCallbacks"] != self.observed_callbacks
            || self.actor_config != serde_json::to_value(actor).map_err(|e| e.to_string())?
            || self.critic_config != serde_json::to_value(critic).map_err(|e| e.to_string())?
        {
            return Err("Continued report differs from saved plan/cohort/configuration".into());
        }
        let headers = plan["headers"]
            .as_array()
            .filter(|h| h.len() == 4)
            .ok_or("Saved current mixed planned headers")?;
        for (i, h) in headers.iter().enumerate() {
            if h["config"]
                != serde_json::to_value(&self.configurations[i]).map_err(|e| e.to_string())?
                || h["basePolicy"]["parent"]
                    != serde_json::to_value(&self.parent).map_err(|e| e.to_string())?
                || h["basePolicy"]["learnerSeat"] != self.learner_seats[i]
                || h["basePolicy"]["rosterChecksum"] != self.roster_checksums[i]
            {
                return Err("Continued report full immediate-parent planned headers differ".into());
            }
        }
        Ok(())
    }
    fn expected_checksum(&self) -> Result<String, String> {
        let mut value = serde_json::to_value(self).map_err(|e| e.to_string())?;
        value
            .as_object_mut()
            .ok_or("Continued mixed report shape")?
            .remove("checksum");
        hash(b"tzolkin-continued-mixed-joint-report-v2\0", &value)
    }
    fn validate_content(&self) -> Result<(), String> {
        let actor = OneStepConfig::from_checkpoint(self.actor_config.clone())?;
        let critic = ResidualFitConfig::from_checkpoint(self.critic_config.clone())?;
        critic.check_callbacks(self.counts[0])?;
        let first = match &self.parent {
            CurrentMixedParentIdentity::InitialMixedV1 { policy } => policy.artifact_checksum(),
            CurrentMixedParentIdentity::ContinuedMixedV2 { policy } => {
                policy.first_mixed_checksum()
            }
        };
        if self.contract != contract()
            || self.expected_checksum()? != self.checksum
            || self.proposed_mixed_update_count != next_count(&self.parent)?
            || self.root.imported_update_count != self.parent.imported_update_count()
            || self.root.first_mixed_checksum != first
            || [
                &self.root.original_root_checksum,
                &self.root.first_mixed_checksum,
                &self.root.root_init_checksum,
            ]
            .into_iter()
            .any(|s| !checkpoint_digest(s))
            || self.roster_checksums.iter().any(|s| !checkpoint_digest(s))
            || self.observed_callbacks < self.counts[0]
            || self.observed_callbacks > 4 * crate::public_stochastic::MAX_SAMPLES
            || self.critic_changed_parameters > crate::public_state_critic::PARAMETER_COUNT
            || !self.mean_raw_residual_mse.is_finite()
            || self.mean_raw_residual_mse < 0.0
        {
            return Err("Continued mixed report parent/contract/count/finite mismatch".into());
        }
        let mut families = self.families.to_vec();
        families.sort();
        public_policy_update::validate_joint_checkpoint_report(
            &self.counts,
            &self.deltas,
            &self.numeric,
            &actor,
            (
                [
                    &self.root.root_init_checksum,
                    &self.plan_checksum,
                    &self.cohort_checksum,
                    &self.checksum,
                ],
                &self.episode_checksums,
            ),
            &families,
            &self.root.bc_source,
        )?;
        let configs = self
            .configurations
            .iter()
            .map(|c| c.checked_config())
            .collect::<Result<Vec<_>, _>>()?;
        if configs.iter().filter(|c| c.players() == 3).count() != 2
            || configs.iter().filter(|c| c.players() == 4).count() != 2
        {
            return Err("Continued mixed report whole-four shape".into());
        }
        for (i, c) in configs.iter().enumerate() {
            if self.learner_seats[i] >= c.players()
                || crate::dataset::seed_family_id(c.environment_seed()) != self.families[i]
                || split_for_family(&self.families[i])? != DatasetSplit::Train
                || self.families[..i].contains(&self.families[i])
            {
                return Err("Continued mixed report absolute learner/family mismatch".into());
            }
        }
        Ok(())
    }
}

/// Closed content DTO only. Deserialize or a valid checksum cannot mint an owner.
#[derive(Debug, Deserialize, Serialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct ContinuedMixedArtifact {
    schema: String,
    imported_update_count: u64,
    mixed_update_count: u64,
    report: ContinuedMixedReport,
    actor: PublicPolicyModel,
    residual_parameters: Vec<f32>,
    checksum: String,
}
impl ContinuedMixedArtifact {
    pub fn checksum(&self) -> &str {
        &self.checksum
    }
    pub fn report(&self) -> &ContinuedMixedReport {
        &self.report
    }
    pub fn imported_update_count(&self) -> u64 {
        self.imported_update_count
    }
    pub fn mixed_update_count(&self) -> u64 {
        self.mixed_update_count
    }
    fn expected_checksum(&self) -> Result<String, String> {
        hash(
            b"tzolkin-continued-mixed-joint-artifact-v2\0",
            &(
                &self.schema,
                self.imported_update_count,
                self.mixed_update_count,
                &self.report,
                &self.actor,
                &self.residual_parameters,
            ),
        )
    }
    fn validate_content(&self) -> Result<(), String> {
        self.report.validate_content()?;
        self.actor.validate()?;
        if self.schema != ARTIFACT_SCHEMA
            || self.expected_checksum()? != self.checksum
            || self.imported_update_count != self.report.root.imported_update_count
            || self.mixed_update_count != self.report.proposed_mixed_update_count
            || self.report.changed_parameters() == [0, 0]
            || self.residual_parameters.len() != crate::public_state_critic::PARAMETER_COUNT
            || self.residual_parameters.iter().any(|x| !x.is_finite())
        {
            return Err("Continued mixed artifact joint/count/finite mismatch".into());
        }
        Ok(())
    }
    fn validate_against(&self, parent: &CurrentMixedPolicyHandle<'_>) -> Result<(), String> {
        self.validate_content()?;
        if &self.report.parent != parent.identity()
            || self.report.root != parent.root
            || self
                .report
                .families
                .iter()
                .any(|f| parent.contains_family(f))
        {
            return Err(
                "Continued mixed artifact differs from the genuine immediate parent".into(),
            );
        }
        let actor = self.report.actor_config["maxAbsActualDelta"]
            .as_f64()
            .ok_or("Continued actor delta cap")?;
        let critic = self.report.critic_config["maxAbsActualDelta"]
            .as_f64()
            .ok_or("Continued critic delta cap")?;
        let (changed, l2, max) = rounded_delta(
            parent.actor_model().parameters(),
            self.actor.parameters(),
            actor,
        )?;
        let (critic_changed, _, _) = rounded_delta(
            parent.residual_parameters(),
            &self.residual_parameters,
            critic,
        )?;
        let numeric = serde_json::to_value(&self.report.numeric).map_err(|e| e.to_string())?;
        if changed != self.report.numeric.changed_parameters
            || critic_changed != self.report.critic_changed_parameters
            || numeric["actualDeltaL2"].as_f64() != Some(l2)
            || numeric["actualDeltaMaxAbs"].as_f64() != Some(max)
        {
            return Err("Continued mixed actual rounded joint deltas differ".into());
        }
        Ok(())
    }
}

/// No public constructor, Clone, Deserialize or artifact-to-owner conversion.
/// Durable Resolve admission must supply this owner in a separate session unit.
pub struct ContinuedMixedOwner {
    artifact: ContinuedMixedArtifact,
    families: BTreeSet<String>,
}
impl ContinuedMixedOwner {
    pub fn artifact(&self) -> &ContinuedMixedArtifact {
        &self.artifact
    }
    pub fn actor_model(&self) -> &PublicPolicyModel {
        &self.artifact.actor
    }
    pub fn residual_parameters(&self) -> &[f32] {
        &self.artifact.residual_parameters
    }
    pub fn imported_update_count(&self) -> u64 {
        self.artifact.imported_update_count
    }
    pub fn mixed_update_count(&self) -> u64 {
        self.artifact.mixed_update_count
    }
    pub fn first_mixed_checksum(&self) -> &str {
        &self.artifact.report.root.first_mixed_checksum
    }
    pub(crate) fn family_set(&self) -> &BTreeSet<String> {
        &self.families
    }
    pub(crate) fn validate_for_inference(&self) -> Result<(), String> {
        self.artifact.validate_content()?;
        if self
            .artifact
            .report
            .families
            .iter()
            .any(|f| !self.families.contains(f))
        {
            return Err("Continued mixed owner successful family closure differs".into());
        }
        Ok(())
    }
}
enum CurrentMixedParent<'a> {
    Initial(&'a MixedJointOwner),
    Continued(&'a ContinuedMixedOwner),
}

/// Only actual sealed owners construct this borrowed handle, never identity DTOs.
pub struct CurrentMixedPolicyHandle<'a> {
    parent: CurrentMixedParent<'a>,
    identity: CurrentMixedParentIdentity,
    root: MixedRootReference,
    kernel: ResolvedKernel,
}
impl<'a> CurrentMixedPolicyHandle<'a> {
    pub fn from_initial(owner: &'a MixedJointOwner) -> Result<Self, String> {
        let handle = MixedPolicyHandle::new(owner)?;
        Ok(Self {
            parent: CurrentMixedParent::Initial(owner),
            identity: CurrentMixedParentIdentity::InitialMixedV1 {
                policy: handle.identity().clone(),
            },
            root: owner.continued_root_reference()?,
            kernel: Kernel::Scalar.resolve()?,
        })
    }
    pub fn from_continued(owner: &'a ContinuedMixedOwner) -> Result<Self, String> {
        let identity = source::ContinuedMixedIdentity::from_owner(owner)?;
        Ok(Self {
            parent: CurrentMixedParent::Continued(owner),
            identity: CurrentMixedParentIdentity::ContinuedMixedV2 { policy: identity },
            root: owner.artifact.report.root.clone(),
            kernel: Kernel::Scalar.resolve()?,
        })
    }
    pub fn identity(&self) -> &CurrentMixedParentIdentity {
        &self.identity
    }
    pub fn sampling_version(&self) -> &'static str {
        self.identity.sampling_version()
    }
    pub fn backend(&self) -> &'static str {
        self.kernel.backend()
    }
    pub(crate) fn actor_model(&self) -> &PublicPolicyModel {
        match self.parent {
            CurrentMixedParent::Initial(o) => o.actor_model(),
            CurrentMixedParent::Continued(o) => o.actor_model(),
        }
    }
    pub(crate) fn residual_parameters(&self) -> &[f32] {
        match self.parent {
            CurrentMixedParent::Initial(o) => o.residual_parameters(),
            CurrentMixedParent::Continued(o) => o.residual_parameters(),
        }
    }
    pub(crate) fn contains_family(&self, family: &str) -> bool {
        match self.parent {
            CurrentMixedParent::Initial(o) => o.family_closure().any(|f| f == family),
            CurrentMixedParent::Continued(o) => o.family_set().contains(family),
        }
    }
    pub(crate) fn family_closure(&self) -> Box<dyn Iterator<Item = &String> + '_> {
        match self.parent {
            CurrentMixedParent::Initial(o) => Box::new(o.family_closure()),
            CurrentMixedParent::Continued(o) => Box::new(o.family_set().iter()),
        }
    }
    pub fn provenance(&self) -> SeatPolicy {
        match &self.identity {
            CurrentMixedParentIdentity::InitialMixedV1 { policy } => SeatPolicy::PublicRlMixed {
                identity: policy.clone(),
                selection_version: crate::replay::RL_ARGMAX_SELECTION_VERSION.into(),
                guard: (),
            },
            CurrentMixedParentIdentity::ContinuedMixedV2 { policy } => {
                SeatPolicy::PublicRlContinuedMixed {
                    identity: Box::new(policy.clone()),
                    selection_version: crate::replay::RL_ARGMAX_SELECTION_VERSION.into(),
                    guard: (),
                }
            }
        }
    }
    pub fn distribution(
        &self,
        observation: &Observation,
    ) -> Result<CurrentMixedDistribution, String> {
        public_model::validate_contract(observation)?;
        let encoder = FeatureEncoder::new_public(observation)?;
        let rows = (0..observation.legal_actions.len())
            .map(|i| encoder.encode_legal_tagged(i))
            .collect::<Result<Vec<_>, _>>()?;
        let numeric = public_model::predict_numeric(self.actor_model(), self.kernel, &rows)?;
        Ok(CurrentMixedDistribution {
            identity: self.identity.clone(),
            logits: numeric.logits,
            probabilities: numeric.probabilities,
            value: (),
            value_validity: ValueValidity::UnavailablePolicyOnly,
        })
    }
    pub fn choose_move(&self, observation: &Observation) -> Result<Decision, String> {
        let p = self.distribution(observation)?;
        let best = (1..p.logits.len()).fold(0, |best, i| {
            if p.logits[i] > p.logits[best] {
                i
            } else {
                best
            }
        });
        Ok(Decision {
            actor: observation.actor,
            observation_key: observation.observation_key.clone(),
            policy_version: self.identity.policy_version().into(),
            r#move: observation.legal_actions[best].r#move.clone(),
            score: f64::from(p.logits[best]),
        })
    }
}
#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct CurrentMixedDistribution {
    identity: CurrentMixedParentIdentity,
    logits: Vec<f32>,
    probabilities: Vec<f32>,
    value: (),
    value_validity: ValueValidity,
}
impl CurrentMixedDistribution {
    pub fn identity(&self) -> &CurrentMixedParentIdentity {
        &self.identity
    }
    pub fn logits(&self) -> &[f32] {
        &self.logits
    }
    pub fn probabilities(&self) -> &[f32] {
        &self.probabilities
    }
    pub fn value_validity(&self) -> ValueValidity {
        self.value_validity
    }
}
enum ProposalOutcome {
    Updated(ContinuedMixedArtifact),
    NoChange(ContinuedMixedReport),
}
/// Opaque genuine numeric result; publication DTOs remain non-executable.
pub struct ContinuedMixedProposal {
    outcome: ProposalOutcome,
}
/// Candidate and initial union are checked before any durable Resolve.
pub(crate) struct PreparedContinuedPromotion {
    artifact: ContinuedMixedArtifact,
    initial_families: Option<BTreeSet<String>>,
}
impl PreparedContinuedPromotion {
    pub(crate) fn artifact(&self) -> &ContinuedMixedArtifact {
        &self.artifact
    }
}
pub(crate) fn admit_resolved(
    proof: crate::public_rl_current_mixed_durable::DurableMixedResolve,
    current: &mut Option<ContinuedMixedOwner>,
) {
    let staged = proof.into_promotion();
    let mut families = match staged.initial_families {
        Some(s) => s,
        None => {
            current
                .take()
                .expect("sealed immediate continued owner")
                .families
        }
    };
    families.extend(staged.artifact.report.families.iter().cloned());
    *current = Some(ContinuedMixedOwner {
        artifact: staged.artifact,
        families,
    });
}
impl ContinuedMixedProposal {
    pub fn report(&self) -> &ContinuedMixedReport {
        match &self.outcome {
            ProposalOutcome::Updated(a) => a.report(),
            ProposalOutcome::NoChange(r) => r,
        }
    }
    pub fn artifact_for_publication(&self) -> Option<&ContinuedMixedArtifact> {
        match &self.outcome {
            ProposalOutcome::Updated(a) => Some(a),
            ProposalOutcome::NoChange(_) => None,
        }
    }
    pub(crate) fn require_parent(
        &self,
        parent: &CurrentMixedPolicyHandle<'_>,
    ) -> Result<(), String> {
        match &self.outcome {
            ProposalOutcome::Updated(a) => a.validate_against(parent),
            ProposalOutcome::NoChange(r) => {
                r.validate_content()?;
                if &r.parent != parent.identity()
                    || r.root != parent.root
                    || r.changed_parameters() != [0, 0]
                    || r.families.iter().any(|f| parent.contains_family(f))
                {
                    return Err("Continued NoChange differs from genuine parent".into());
                }
                Ok(())
            }
        }
    }
    pub(crate) fn restored(
        artifact: ContinuedMixedArtifact,
        parent: &CurrentMixedPolicyHandle<'_>,
    ) -> Result<Self, String> {
        artifact.validate_against(parent)?;
        Ok(Self {
            outcome: ProposalOutcome::Updated(artifact),
        })
    }
    pub(crate) fn restored_no_change(
        report: ContinuedMixedReport,
        parent: &CurrentMixedPolicyHandle<'_>,
    ) -> Result<Self, String> {
        let proposal = Self {
            outcome: ProposalOutcome::NoChange(report),
        };
        proposal.require_parent(parent)?;
        Ok(proposal)
    }
    pub(crate) fn stage(
        self,
        parent: &CurrentMixedPolicyHandle<'_>,
    ) -> Result<PreparedContinuedPromotion, String> {
        let artifact = match self.outcome {
            ProposalOutcome::Updated(a) => a,
            ProposalOutcome::NoChange(_) => {
                return Err("NoChange cannot promote a continued owner".into());
            }
        };
        artifact.validate_against(parent)?;
        let initial_families = match parent.parent {
            CurrentMixedParent::Initial(o) => Some(o.family_closure().cloned().collect()),
            CurrentMixedParent::Continued(_) => None,
        };
        Ok(PreparedContinuedPromotion {
            artifact,
            initial_families,
        })
    }
}
/// Exactly one joint calculation from a genuine current parent and four sealed
/// learner-only episodes. No file I/O, owner promotion, loop or retry is provided.
pub fn prepare_current(
    parent: &CurrentMixedPolicyHandle<'_>,
    cohort: &ValidatedCurrentMixedCohort,
    actor_config: &OneStepConfig,
    critic_config: &ResidualFitConfig,
) -> Result<ContinuedMixedProposal, String> {
    let proposed_count = next_count(parent.identity())?;
    if cohort.plan().parent_identity() != parent.identity() {
        return Err(
            "Continued mixed cohort differs from genuine current parent before math".into(),
        );
    }
    let counts = [
        cohort.total_learner_callbacks(),
        cohort.total_learner_candidate_rows(),
        cohort.total_learner_singletons(),
    ];
    actor_config.validate()?;
    critic_config.check_callbacks(counts[0])?;
    if counts[1] > actor_config.max_candidate_rows() {
        return Err("Continued mixed candidate row cap before math".into());
    }
    for (i, e) in cohort.episodes().iter().enumerate() {
        let family = &cohort.plan().rollout_families()[i];
        if e.parent_identity() != parent.identity()
            || e.numerical_target() != numerical_target()
            || split_for_family(family)? != DatasetSplit::Train
            || parent.contains_family(family)
        {
            return Err("Continued mixed sealed source/target/family differs before math".into());
        }
    }
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
        .map_err(|_| "Continued mixed numeric whole-four shape")?;
    let (deltas, actor, critic) = prepare_numeric(
        parent.actor_model(),
        parent.residual_parameters(),
        &members,
        counts,
        actor_config,
        critic_config,
    )?;
    let mut report = ContinuedMixedReport {
        contract: contract(),
        root: parent.root.clone(),
        parent: parent.identity.clone(),
        proposed_mixed_update_count: proposed_count,
        plan_checksum: cohort.plan().checksum().into(),
        cohort_checksum: cohort.receipt_checksum().into(),
        episode_checksums: cohort
            .episodes()
            .each_ref()
            .map(|e| e.canonical_record_checksum().into()),
        learner_seats: cohort.episodes().each_ref().map(|e| e.learner_seat()),
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
    report.validate_content()?;
    if report.changed_parameters() == [0, 0] {
        return Ok(ContinuedMixedProposal {
            outcome: ProposalOutcome::NoChange(report),
        });
    }
    let mut artifact = ContinuedMixedArtifact {
        schema: ARTIFACT_SCHEMA.into(),
        imported_update_count: parent.identity.imported_update_count(),
        mixed_update_count: proposed_count,
        report,
        actor: PublicPolicyModel {
            parameters: actor.parameters,
        },
        residual_parameters: critic.parameters,
        checksum: String::new(),
    };
    artifact.checksum = artifact.expected_checksum()?;
    artifact.validate_against(parent)?;
    Ok(ContinuedMixedProposal {
        outcome: ProposalOutcome::Updated(artifact),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn continued_mixed_count_neutral_binding_and_evaluation_remain_distinct() {
        let initial = source::tests::initial();
        let old = CurrentMixedParentIdentity::InitialMixedV1 {
            policy: initial.clone(),
        };
        assert_eq!(next_count(&old).unwrap(), 2);
        assert_eq!(old.binding_key().unwrap(), initial.binding_key().unwrap());
        let mut wire = serde_json::to_value(&initial).unwrap();
        for (key, value) in [
            ("schema", source::CONTINUED_IDENTITY_SCHEMA),
            ("policyVersion", source::CONTINUED_POLICY_VERSION),
            ("samplingVersion", source::CONTINUED_SAMPLING_VERSION),
            ("task", source::CONTINUED_TASK),
        ] {
            wire[key] = value.into();
        }
        wire["firstMixedChecksum"] = serde_json::json!("d".repeat(64));
        wire["parentArtifactChecksum"] = serde_json::json!("e".repeat(64));
        wire["mixedUpdateCount"] = 2.into();
        let identity: source::ContinuedMixedIdentity =
            serde_json::from_value(wire.clone()).unwrap();
        let parent = CurrentMixedParentIdentity::ContinuedMixedV2 {
            policy: identity.clone(),
        };
        assert_eq!(next_count(&parent).unwrap(), 3);
        assert_eq!(parent.imported_update_count(), 1);
        assert_ne!(parent.binding_key().unwrap(), old.binding_key().unwrap());
        for key in [
            "actorParametersChecksum",
            "residualParametersChecksum",
            "parentArtifactChecksum",
            "firstMixedChecksum",
        ] {
            let mut changed = wire.clone();
            changed[key] = serde_json::json!("f".repeat(64));
            let p = CurrentMixedParentIdentity::ContinuedMixedV2 {
                policy: serde_json::from_value(changed).unwrap(),
            };
            assert_ne!(p.binding_key().unwrap(), parent.binding_key().unwrap());
        }
        wire["mixedUpdateCount"] = source::MAX_MIXED_UPDATES.into();
        let ceiling = CurrentMixedParentIdentity::ContinuedMixedV2 {
            policy: serde_json::from_value(wire.clone()).unwrap(),
        };
        assert!(next_count(&ceiling).is_err());
        assert!(
            serde_json::from_value::<crate::public_rl_mixed_handle::MixedInferenceIdentity>(
                wire.clone()
            )
            .is_err()
        );
        wire["acceptedOwner"] = true.into();
        assert!(serde_json::from_value::<source::ContinuedMixedIdentity>(wire).is_err());
        let provenance = SeatPolicy::PublicRlContinuedMixed {
            identity: Box::new(identity),
            selection_version: crate::replay::RL_ARGMAX_SELECTION_VERSION.into(),
            guard: (),
        };
        provenance.validate().unwrap();
        assert!(provenance.is_rl_evaluation());
        assert!(!crate::public_rl_mixed_native::legacy_opponent_identity_allowed(&provenance));
        let mut wrong = serde_json::to_value(&provenance).unwrap();
        wrong["kind"] = "publicRlMixed".into();
        assert!(serde_json::from_value::<SeatPolicy>(wrong).is_err());
        assert!(
            serde_json::from_value::<crate::public_rl_mixed_native::MixedLearnerIdentity>(
                serde_json::json!({"kind":"continuedMixed","policy":parent})
            )
            .is_err()
        );
    }
}
