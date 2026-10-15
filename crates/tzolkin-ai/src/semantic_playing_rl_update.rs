//! Recurrent, in-memory whole-four semantic Playing joint updates.
//! Ownership follows a successfully loaded fresh origin, not a parameter file.
//! Private staged hooks also serve the journal and its checked continuation.
//! No legacy owner conversion or Adam state is admitted here.
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

use crate::conditional_human_rl_update::{ConditionalFrozenPool, ConditionalNativeExclusions};
use crate::conditional_semantic_playing_rl::{
    SemanticCollectionScope, SemanticFreshOwner, SemanticPlayingRlOwner, SemanticTrainingState,
};
use crate::human_policy_dataset::ConditionalFeatureProfile;
use crate::public_model::PublicPolicyModel;
use crate::public_native::PreparedPublicPolicy;
use crate::public_policy_baseline::ResidualFitConfig;
use crate::public_policy_mixed::{SemanticNumericMember, prepare_numeric_timed_for_profile};
use crate::public_policy_update::{NumericReport, OneStepConfig, PhaseTimer, TimedPhase, hash};
use crate::public_stochastic::SamplingSeed;
use crate::public_stochastic_native::CollectionLimits;
use crate::semantic_playing_rl_collection::{
    AuditedSemanticPlayingCohort, DeclaredSamplingHistory, PlannedSemanticReservation, PoolRef,
    SemanticCollectionComposition, SemanticCollectionLedger, SemanticPlayingCollectedGame,
    SemanticReservation,
};
use crate::semantic_playing_rl_pool::{FrozenSemanticSnapshot, SemanticFrozenMixture};

pub const UPDATE_VERSION: &str = "semantic-playing-own-sum-quarter-recurrent-scalar-v1";
pub const GUARDED_UPDATE_VERSION: &str = "semantic-playing-guarded-proposal-sum-quarter-scalar-v2";

/// Same-pass observations only; never serialized into an owned report or identity.
#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct SemanticSignalDiagnostics {
    report_checksum: String,
    parent: String,
    next: String,
    cohort: String,
    actor: NumericReport,
    members: [MemberSignal; 4],
}
#[derive(Debug, Default, Serialize)]
#[serde(rename_all = "camelCase")]
struct SignalSummary {
    min: f64,
    max: f64,
    mean: f64,
    positive: usize,
    negative: usize,
    zero: usize,
    one: usize,
}
impl SignalSummary {
    fn observe(&mut self, value: f64, count: usize) {
        self.min = if count == 1 {
            value
        } else {
            self.min.min(value)
        };
        self.max = if count == 1 {
            value
        } else {
            self.max.max(value)
        };
        self.mean += value; // Converted from ordered sum exactly once at finish.
        self.positive += usize::from(value > 0.0);
        self.negative += usize::from(value < 0.0);
        self.zero += usize::from(value == 0.0);
        self.one += usize::from(value == 1.0);
    }
}
#[derive(Debug, Default, Serialize)]
#[serde(rename_all = "camelCase")]
struct MemberSignal {
    learner: usize,
    players: usize,
    callbacks: usize,
    candidates: usize,
    singletons: usize,
    terminal_winner_share: f64,
    baseline: SignalSummary,
    advantage: SignalSummary,
}
#[derive(Default)]
struct SignalCollector {
    member: usize,
    inconsistent_target: bool,
    members: [MemberSignal; 4],
}
impl PhaseTimer for SignalCollector {
    type Span = ();
    fn enter(&mut self, _: TimedPhase) {}
    fn returned(&mut self, _: (), _: bool) {}
    fn actor_member(&mut self, index: usize, learner: usize, players: usize) {
        self.member = index;
        self.members[index].learner = learner;
        self.members[index].players = players;
    }
    fn actor_signal(&mut self, target: f64, baseline: f64, candidates: usize) {
        let member = &mut self.members[self.member];
        if member.callbacks == 0 {
            member.terminal_winner_share = target;
        } else {
            self.inconsistent_target |= member.terminal_winner_share != target;
        }
        member.callbacks += 1;
        member.candidates += candidates;
        member.singletons += usize::from(candidates == 1);
        member.baseline.observe(baseline, member.callbacks);
        member
            .advantage
            .observe(target - baseline, member.callbacks);
    }
}
impl SignalCollector {
    fn finish(
        mut self,
        actor: NumericReport,
        counts: [usize; 3],
    ) -> Result<SemanticSignalDiagnostics, String> {
        let observed = self.members.iter().fold([0; 3], |mut sums, m| {
            for (sum, n) in sums
                .iter_mut()
                .zip([m.callbacks, m.candidates, m.singletons])
            {
                *sum += n;
            }
            sums
        });
        if self.inconsistent_target
            || observed != counts
            || self.members.iter().any(|m| m.callbacks == 0)
        {
            return Err("Semantic signal summary differs from sealed learner steps".into());
        }
        for m in &mut self.members {
            m.baseline.mean /= m.callbacks as f64;
            m.advantage.mean /= m.callbacks as f64;
        }
        Ok(SemanticSignalDiagnostics {
            report_checksum: String::new(),
            parent: String::new(),
            next: String::new(),
            cohort: String::new(),
            actor,
            members: self.members,
        })
    }
}

/// Descriptive, non-deserializable identity of this session's current state.
/// Counts belong only to complete admitted updates from its own fresh root.
/// A later failed attempt changes the session's burned allocation ledger, not
/// these last-admitted weights/counters. Snapshot capture binds both separately.
#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct SemanticPlayingCurrentIdentity {
    root: String,
    previous: String,
    actor: String,
    critic: String,
    updates: u64,
    credit: u64,
    attempts: u64,
    next: u64,
    checksum: String,
    #[serde(skip)]
    cohort: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    collection_composition: Option<SemanticCollectionComposition>,
    #[serde(skip_serializing_if = "Option::is_none")]
    trade_guard: Option<tzolkin_inference::trade_guard::TradeGuardConfig>,
}
impl SemanticPlayingCurrentIdentity {
    pub fn checksum(&self) -> &str {
        &self.checksum
    }
    pub fn root_checksum(&self) -> &str {
        &self.root
    }
    pub fn immediate_parent_checksum(&self) -> &str {
        &self.previous
    }
    pub fn actor_checksum(&self) -> &str {
        &self.actor
    }
    pub fn critic_checksum(&self) -> &str {
        &self.critic
    }
    pub fn completed_updates(&self) -> u64 {
        self.updates
    }
    pub fn credited_playing_decisions(&self) -> u64 {
        self.credit
    }
    pub fn attempted_whole_four(&self) -> u64 {
        self.attempts
    }
    pub fn next_episode_ordinal(&self) -> u64 {
        self.next
    }
    pub fn policy_version(&self) -> &'static str {
        if self.composition() == SemanticCollectionComposition::LegacyV1 {
            UPDATE_VERSION
        } else {
            GUARDED_UPDATE_VERSION
        }
    }
    pub fn composition(&self) -> SemanticCollectionComposition {
        self.collection_composition.unwrap_or_default()
    }
    pub fn feature_schema(&self) -> u32 {
        crate::features::SEMANTIC_PLAYING_FEATURE_SCHEMA
    }
    pub fn critic_feature_schema(&self) -> u32 {
        crate::features::PUBLIC_FEATURE_SCHEMA
    }
    pub fn source_authenticated(&self) -> bool {
        false
    }
    pub fn source_qualified(&self) -> bool {
        false
    }
    pub fn verified_native_policy(&self) -> bool {
        false
    }
}

/// A closed outcome, not a checkpoint or a constructor for an owner.
#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct SemanticPlayingJointReport {
    #[serde(skip)]
    signals: Option<SemanticSignalDiagnostics>,
    parent: String,
    next: String,
    cohort: String,
    families: [String; 4],
    counts: [usize; 3],
    changes: [usize; 2],
    attempts: usize,
    next_ordinal: u64,
    credit: u64,
    residual_loss: f64,
    mixture: Option<String>,
    opponent_samples: usize,
    total_rows: usize,
    actor_draws: [[u64; 4]; 4],
    checksum: String,
    /// Learner proposals/overrides, then sampled opponent proposals/overrides.
    /// Probabilities and credit continue to describe full-legal latent draws.
    #[serde(skip_serializing_if = "Option::is_none")]
    guarded_proposals: Option<[usize; 4]>,
}
impl SemanticPlayingJointReport {
    pub fn parent_checksum(&self) -> &str {
        &self.parent
    }
    pub fn next_identity_checksum(&self) -> &str {
        &self.next
    }
    pub fn cohort_checksum(&self) -> &str {
        &self.cohort
    }
    pub fn families(&self) -> &[String; 4] {
        &self.families
    }
    /// Playing callbacks, full legal candidate rows, singleton callbacks.
    pub fn counts(&self) -> [usize; 3] {
        self.counts
    }
    pub fn changed_parameters(&self) -> [usize; 2] {
        self.changes
    }
    pub fn updated(&self) -> bool {
        self.changes != [0, 0]
    }
    pub fn attempted_whole_four(&self) -> usize {
        self.attempts
    }
    pub fn next_episode_ordinal(&self) -> u64 {
        self.next_ordinal
    }
    pub fn credited_playing_decisions(&self) -> u64 {
        self.credit
    }
    pub fn mean_raw_residual_mse(&self) -> f64 {
        self.residual_loss
    }
    pub fn checksum(&self) -> &str {
        &self.checksum
    }
    pub fn mixture_checksum(&self) -> Option<&str> {
        self.mixture.as_deref()
    }
    pub fn opponent_playing_samples(&self) -> usize {
        self.opponent_samples
    }
    pub fn total_sampled_candidate_rows(&self) -> usize {
        self.total_rows
    }
    pub fn actor_draws(&self) -> &[[u64; 4]; 4] {
        &self.actor_draws
    }
}

/// One fresh root, a recurrent current state and one persistent allocation ledger.
/// All source checks and both proposals finish before either parameter is moved.
/// NoChange stops further attempts while retaining all family/stream burns.
pub struct SemanticPlayingJointSession<'a> {
    state: SemanticTrainingState,
    identity: SemanticPlayingCurrentIdentity,
    ledger: SemanticCollectionLedger<'a>,
    actor_config: OneStepConfig,
    critic_config: ResidualFitConfig,
    attempt_base: u64,
}
/// Private numerical payload. It is never a fresh owner or an actor-step DTO.
#[derive(Clone, Deserialize, Serialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub(crate) struct SemanticCheckpointState {
    root: String,
    current: serde_json::Value,
    cohort: String,
    actor: PublicPolicyModel,
    critic: Vec<f32>,
}
impl SemanticCheckpointState {
    /// Numeric borrow only; the checkpoint reader validates its owned identity.
    /// This does not expose a training owner or a checkpoint constructor.
    pub(crate) fn actor_for_policy_export(&self) -> &[f32] {
        self.actor.parameters()
    }

    pub(crate) fn validate(
        &self,
        actor: &OneStepConfig,
        critic: &ResidualFitConfig,
    ) -> Result<SemanticPlayingCurrentIdentity, String> {
        self.actor.validate()?;
        if self.critic.len() != crate::public_state_critic::PARAMETER_COUNT
            || self.critic.iter().any(|v| !v.is_finite())
        {
            return Err("Semantic checkpoint critic shape/finite values".into());
        }
        let u = |key: &str| {
            self.current[key]
                .as_u64()
                .ok_or("Semantic checkpoint counter".to_string())
        };
        let previous = self.current["previous"]
            .as_str()
            .ok_or("Semantic checkpoint parent")?;
        let identity = make_identity(
            &self.root,
            previous,
            self.actor.parameters(),
            &self.critic,
            u("updates")?,
            u("credit")?,
            u("attempts")?,
            u("next")?,
            &self.cohort,
            actor,
            critic,
            self.composition()?,
        )?;
        if serde_json::to_value(&identity).map_err(|e| e.to_string())? != self.current
            || identity.updates == 0
            || identity.credit == 0
            || identity.attempts < identity.updates
        {
            return Err("Semantic checkpoint identity differs".into());
        }
        Ok(identity)
    }
    pub(crate) fn composition(&self) -> Result<SemanticCollectionComposition, String> {
        match self.current.get("collectionComposition") {
            None => Ok(SemanticCollectionComposition::LegacyV1),
            Some(value) => {
                let composition: SemanticCollectionComposition =
                    serde_json::from_value(value.clone()).map_err(|e| e.to_string())?;
                if composition != SemanticCollectionComposition::GuardedProposalV2 {
                    return Err("Semantic checkpoint explicit composition differs".into());
                }
                Ok(composition)
            }
        }
    }
}
impl<'a> SemanticPlayingJointSession<'a> {
    pub(crate) fn checkpoint_state(&self) -> Result<SemanticCheckpointState, String> {
        self.journal_recheck()?;
        if self.stopped()
            || self.attempts() == 0
            || self.identity.attempts != self.attempt_base + self.attempts() as u64
        {
            return Err("Semantic checkpoint requires completed admitted work".into());
        }
        Ok(SemanticCheckpointState {
            root: self.identity.root.clone(),
            current: serde_json::to_value(&self.identity).map_err(|e| e.to_string())?,
            cohort: self.identity.cohort.clone(),
            actor: PublicPolicyModel {
                parameters: self.state.actor_parameters().to_vec(),
            },
            critic: self.state.critic_parameters().to_vec(),
        })
    }
    #[allow(clippy::too_many_arguments)]
    #[cfg(test)]
    pub(crate) fn from_checkpoint(
        owner: SemanticPlayingRlOwner,
        saved: SemanticCheckpointState,
        external: &'a ConditionalNativeExclusions,
        history: &'a DeclaredSamplingHistory,
        seed: SamplingSeed,
        first: u64,
        replicate: u64,
        actor_config: OneStepConfig,
        critic_config: ResidualFitConfig,
    ) -> Result<Self, String> {
        Self::from_checkpoint_owner(
            SemanticFreshOwner::Human(Box::new(owner)),
            saved,
            external,
            history,
            seed,
            first,
            replicate,
            actor_config,
            critic_config,
        )
    }
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn from_checkpoint_owner(
        owner: SemanticFreshOwner,
        saved: SemanticCheckpointState,
        external: &'a ConditionalNativeExclusions,
        history: &'a DeclaredSamplingHistory,
        seed: SamplingSeed,
        first: u64,
        replicate: u64,
        actor_config: OneStepConfig,
        critic_config: ResidualFitConfig,
    ) -> Result<Self, String> {
        actor_config.validate()?;
        critic_config.check_callbacks(1)?;
        let identity = saved.validate(&actor_config, &critic_config)?;
        if first < identity.next
            || (first > identity.next
                && (3..=4).any(|players| {
                    (0..players).any(|actor| {
                        !history.covers(seed, replicate, players, actor, identity.next, first)
                    })
                }))
        {
            return Err("Semantic checkpoint identity or uncovered ordinal gap differs".into());
        }
        let state =
            SemanticTrainingState::from_checkpoint(owner, &saved.root, saved.actor, saved.critic)?;
        let mut ledger = SemanticCollectionLedger::new_composed(
            external,
            history,
            seed,
            first,
            replicate,
            identity.composition(),
        )?;
        ledger.commit_admitted(BTreeSet::from([identity.checksum().to_string()]));
        ledger.recheck(&state.view(identity.checksum()))?;
        Ok(Self {
            attempt_base: identity.attempts,
            state,
            identity,
            ledger,
            actor_config,
            critic_config,
        })
    }
    /// Borrow the admitted current source for saved-record auditing only.
    /// No restore, reservation, actor-step or ownership conversion is exposed.
    pub fn record_auditor(
        &self,
    ) -> Result<crate::semantic_playing_rl_record::SemanticPlayingRecordAuditor<'_, 'a>, String>
    {
        self.ledger
            .record_auditor(self.state.view(self.identity.checksum()))
    }
    #[allow(clippy::too_many_arguments)]
    pub fn from_fresh(
        owner: SemanticPlayingRlOwner,
        external: &'a ConditionalNativeExclusions,
        history: &'a DeclaredSamplingHistory,
        seed: SamplingSeed,
        first: u64,
        replicate: u64,
        actor_config: OneStepConfig,
        critic_config: ResidualFitConfig,
    ) -> Result<Self, String> {
        Self::from_fresh_composed(
            owner,
            external,
            history,
            seed,
            first,
            replicate,
            actor_config,
            critic_config,
            SemanticCollectionComposition::LegacyV1,
        )
    }
    /// A distinct fresh guarded-proposal session from the same checked Human4
    /// source. It does not migrate or relabel an earlier unguarded checkpoint.
    #[allow(clippy::too_many_arguments)]
    pub fn from_fresh_guarded(
        owner: SemanticPlayingRlOwner,
        external: &'a ConditionalNativeExclusions,
        history: &'a DeclaredSamplingHistory,
        seed: SamplingSeed,
        first: u64,
        replicate: u64,
        actor_config: OneStepConfig,
        critic_config: ResidualFitConfig,
    ) -> Result<Self, String> {
        Self::from_fresh_composed(
            owner,
            external,
            history,
            seed,
            first,
            replicate,
            actor_config,
            critic_config,
            SemanticCollectionComposition::GuardedProposalV2,
        )
    }
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn from_fresh_composed(
        owner: SemanticPlayingRlOwner,
        external: &'a ConditionalNativeExclusions,
        history: &'a DeclaredSamplingHistory,
        seed: SamplingSeed,
        first: u64,
        replicate: u64,
        actor_config: OneStepConfig,
        critic_config: ResidualFitConfig,
        composition: SemanticCollectionComposition,
    ) -> Result<Self, String> {
        Self::from_fresh_owner(
            SemanticFreshOwner::Human(Box::new(owner)),
            external,
            history,
            seed,
            first,
            replicate,
            actor_config,
            critic_config,
            composition,
        )
    }
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn from_fresh_owner(
        owner: SemanticFreshOwner,
        external: &'a ConditionalNativeExclusions,
        history: &'a DeclaredSamplingHistory,
        seed: SamplingSeed,
        first: u64,
        replicate: u64,
        actor_config: OneStepConfig,
        critic_config: ResidualFitConfig,
        composition: SemanticCollectionComposition,
    ) -> Result<Self, String> {
        actor_config.validate()?;
        critic_config.check_callbacks(1)?;
        let state = SemanticTrainingState::from_fresh(owner)?;
        let mut ledger = SemanticCollectionLedger::new_composed(
            external,
            history,
            seed,
            first,
            replicate,
            composition,
        )?;
        let root = state.root().checksum().to_string();
        let (identity, admitted) = ledger.with_checked(&state.view(&root), |scope| {
            let identity = make_identity(
                &root,
                &root,
                scope.actor().parameters(),
                scope.critic(),
                0,
                0,
                0,
                first,
                "fresh-root",
                &actor_config,
                &critic_config,
                composition,
            )?;
            let admitted = BTreeSet::from([identity.checksum().to_string()]);
            Ok((identity, admitted))
        })?;
        ledger.commit_admitted(admitted);
        Ok(Self {
            state,
            identity,
            ledger,
            actor_config,
            critic_config,
            attempt_base: 0,
        })
    }
    pub fn identity(&self) -> Result<&SemanticPlayingCurrentIdentity, String> {
        self.ledger
            .recheck(&self.state.view(self.identity.checksum()))?;
        Ok(&self.identity)
    }
    /// Numeric observations only; these slices cannot reconstruct this owner.
    pub fn actor_parameters(&self) -> Result<&[f32], String> {
        self.ledger
            .recheck(&self.state.view(self.identity.checksum()))?;
        Ok(self.state.actor_parameters())
    }
    pub fn critic_parameters(&self) -> Result<&[f32], String> {
        self.ledger
            .recheck(&self.state.view(self.identity.checksum()))?;
        Ok(self.state.critic_parameters())
    }
    pub fn attempts(&self) -> usize {
        self.ledger.attempts()
    }
    pub fn next_episode_ordinal(&self) -> u64 {
        self.ledger.next_episode_ordinal()
    }
    pub fn consumed_families(&self) -> &BTreeSet<String> {
        self.ledger.consumed_families()
    }
    pub fn consumed_actor_streams(&self) -> usize {
        self.ledger.consumed_actor_streams()
    }
    pub fn stopped(&self) -> bool {
        self.ledger.stopped()
    }
    pub fn failed_game(&self) -> Option<&SemanticPlayingCollectedGame> {
        self.ledger.failed_game()
    }
    #[cfg(test)]
    pub(crate) fn fixture_lanes(
        &mut self,
        lanes: crate::semantic_playing_rl_collection::CollectionLanes,
    ) {
        self.ledger.lanes = lanes;
    }
    #[cfg(test)]
    pub(crate) fn fixture_admitted(&self) -> &BTreeSet<String> {
        self.ledger.admitted_checksums()
    }
    pub fn freeze_snapshot(&self) -> Result<FrozenSemanticSnapshot<'a>, String> {
        self.ledger
            .freeze_snapshot(&self.state.view(self.identity.checksum()), &self.identity)
    }

    pub fn collect_and_update_mixed_whole_four(
        &mut self,
        seeds: [u32; 4],
        mixture: &SemanticFrozenMixture<'_>,
        limits: CollectionLimits,
    ) -> Result<SemanticPlayingJointReport, String> {
        self.check_headroom(limits)?;
        let attempt = self
            .ledger
            .attempts()
            .checked_add(1)
            .ok_or("Semantic attempt overflow")?;
        let proposal = self.ledger.collect_mixture_using(
            &self.state.view(self.identity.checksum()),
            seeds,
            mixture,
            limits,
            |scope, cohort| {
                prepare(
                    scope,
                    &self.identity,
                    &cohort,
                    &self.actor_config,
                    &self.critic_config,
                    attempt,
                )
            },
        )?;
        let result = self.admit_with_pool(proposal, Some(mixture));
        if result.is_err() {
            self.ledger.stop();
        }
        result
    }

    pub fn collect_and_update_whole_four(
        &mut self,
        seeds: [u32; 4],
        pool: &ConditionalFrozenPool,
        prepared_opponent: Option<&PreparedPublicPolicy>,
        limits: CollectionLimits,
    ) -> Result<SemanticPlayingJointReport, String> {
        self.check_headroom(limits)?;
        let attempt = self
            .ledger
            .attempts()
            .checked_add(1)
            .ok_or("Semantic attempt overflow")?;
        let proposal = self.ledger.collect_whole_four_using(
            &self.state.view(self.identity.checksum()),
            seeds,
            pool,
            prepared_opponent,
            limits,
            |scope, cohort| {
                prepare(
                    scope,
                    &self.identity,
                    &cohort,
                    &self.actor_config,
                    &self.critic_config,
                    attempt,
                )
            },
        )?;
        let result = self.admit(proposal);
        if result.is_err() {
            self.ledger.stop();
        }
        result
    }
    fn check_headroom(&self, limits: CollectionLimits) -> Result<(), String> {
        let rows = limits
            .max_candidate_rows()
            .checked_mul(4)
            .ok_or("Semantic joint row headroom overflow")?;
        let callbacks = limits
            .max_callbacks()
            .checked_mul(4)
            .ok_or("Semantic joint callback headroom overflow")?;
        if rows > self.actor_config.max_candidate_rows() {
            return Err("Semantic whole-four prospective rows exceed numerical cap".into());
        }
        self.critic_config.check_callbacks(callbacks)
    }
    fn admit(&mut self, proposal: JointProposal) -> Result<SemanticPlayingJointReport, String> {
        self.admit_with_pool(proposal, None)
    }
    fn prepare_admission(
        &self,
        mut proposal: JointProposal,
        mixture: Option<&SemanticFrozenMixture<'_>>,
    ) -> Result<PreparedSemanticAdmission, String> {
        let admitted = Self::prepare_admission_metadata(
            &self.identity,
            &self.ledger,
            self.attempt_base,
            &mut proposal,
            mixture,
        )?;
        let owner = self.state.view(self.identity.checksum());
        if let Some(mixture) = mixture {
            self.ledger
                .recheck_with_pool(&owner, PoolRef::Mixture(mixture))?;
        } else {
            self.ledger.recheck(&owner)?;
        }
        Ok(PreparedSemanticAdmission { proposal, admitted })
    }
    fn prepare_admission_metadata(
        identity: &SemanticPlayingCurrentIdentity,
        ledger: &SemanticCollectionLedger<'_>,
        attempt_base: u64,
        proposal: &mut JointProposal,
        mixture: Option<&SemanticFrozenMixture<'_>>,
    ) -> Result<BTreeSet<String>, String> {
        if proposal.report.parent != identity.checksum()
            || proposal
                .report
                .families
                .iter()
                .any(|f| !ledger.consumed_families().contains(f))
            || proposal.identity.previous != identity.checksum()
            || proposal.identity.attempts != attempt_base + ledger.attempts() as u64
            || proposal.identity.next != ledger.next_episode_ordinal()
            || proposal.report.mixture.as_deref() != mixture.map(SemanticFrozenMixture::checksum)
        {
            return Err("Semantic joint proposal differs from the current reserved parent".into());
        }
        // This only prepares metadata. Source-checked callers retain custody
        // until their POST, and no identity is admitted until the later commit.
        let admitted =
            ledger.prepare_admitted(identity.checksum(), proposal.identity.checksum())?;
        let report = &mut proposal.report;
        report.attempts = usize::try_from(proposal.identity.attempts)
            .map_err(|_| "Semantic cumulative report attempt overflow")?;
        report.next_ordinal = ledger.next_episode_ordinal();
        report.next = proposal.identity.checksum().into();
        report.checksum = hash(
            b"semantic-playing-joint-report-v1\0",
            &(
                (
                    &report.parent,
                    &report.next,
                    &report.cohort,
                    &report.families,
                ),
                (
                    report.counts,
                    report.changes,
                    report.attempts,
                    report.next_ordinal,
                    report.credit,
                ),
                report.residual_loss,
            ),
        )?;
        if report.mixture.is_some() {
            report.checksum = hash(
                b"semantic-playing-fixed-mixture-report-v1\0",
                &(
                    &report.checksum,
                    &report.mixture,
                    report.opponent_samples,
                    report.total_rows,
                    report.actor_draws,
                ),
            )?;
        }
        if let Some(counts) = report.guarded_proposals {
            report.checksum = hash(
                b"semantic-playing-guarded-proposal-report-v2\0",
                &(
                    &report.checksum,
                    counts,
                    identity.composition(),
                    tzolkin_inference::trade_guard::TradeGuardConfig::default(),
                ),
            )?;
        }
        if let Some(signals) = &mut report.signals {
            signals.report_checksum = report.checksum.clone();
            signals.parent = report.parent.clone();
            signals.next = report.next.clone();
            signals.cohort = report.cohort.clone();
        }
        Ok(admitted)
    }
    fn admit_with_pool(
        &mut self,
        proposal: JointProposal,
        mixture: Option<&SemanticFrozenMixture<'_>>,
    ) -> Result<SemanticPlayingJointReport, String> {
        let prepared = self.prepare_admission(proposal, mixture)?;
        Ok(self.commit_prepared(prepared))
    }
    /// Prepared by the same source-postchecked numerical path. Nothing here
    /// hashes, allocates, performs IO or can return an error after Resolve.
    pub(crate) fn commit_prepared(
        &mut self,
        prepared: PreparedSemanticAdmission,
    ) -> SemanticPlayingJointReport {
        let PreparedSemanticAdmission { proposal, admitted } = prepared;
        if let Some((actor, critic)) = proposal.parameters {
            self.state.commit(actor, critic);
        } else {
            self.ledger.stop();
        }
        self.identity = proposal.identity;
        self.ledger.commit_admitted(admitted);
        proposal.report
    }
    pub(crate) fn plan_reserved(
        &self,
        seeds: [u32; 4],
        pool: PoolRef<'_>,
        limits: CollectionLimits,
    ) -> Result<PlannedSemanticReservation, String> {
        self.check_headroom(limits)?;
        self.ledger.plan(
            &self.state.view(self.identity.checksum()),
            seeds,
            pool,
            limits,
        )
    }
    pub(crate) fn burn_reserved(
        &mut self,
        plan: PlannedSemanticReservation,
    ) -> SemanticReservation {
        self.ledger.reserve(plan)
    }
    pub(crate) fn stage_reserved_using(
        &mut self,
        batch: &SemanticReservation,
        pool: PoolRef<'_>,
        create: impl Fn(
            &crate::public_stochastic_native::NativeStochasticConfig,
        ) -> Result<tzolkin_core::GameState, String>
        + Sync,
    ) -> Result<PreparedSemanticAdmission, String> {
        self.stage_records_using(batch, pool, None, create)
    }
    pub(crate) fn stage_saved_records_using(
        &mut self,
        batch: &SemanticReservation,
        pool: PoolRef<'_>,
        saved: &[Vec<u8>; 4],
        create: impl Fn(
            &crate::public_stochastic_native::NativeStochasticConfig,
        ) -> Result<tzolkin_core::GameState, String>
        + Sync,
    ) -> Result<PreparedSemanticAdmission, String> {
        self.stage_records_using(batch, pool, Some(saved), create)
    }
    fn stage_records_using(
        &mut self,
        batch: &SemanticReservation,
        pool: PoolRef<'_>,
        saved: Option<&[Vec<u8>; 4]>,
        create: impl Fn(
            &crate::public_stochastic_native::NativeStochasticConfig,
        ) -> Result<tzolkin_core::GameState, String>
        + Sync,
    ) -> Result<PreparedSemanticAdmission, String> {
        let attempt = self.ledger.attempts();
        let identity = &self.identity;
        let result = self.ledger.collect_reserved_records(
            &self.state.view(identity.checksum()),
            batch,
            pool,
            saved,
            |scope, cohort, ledger| {
                let records = crate::semantic_playing_rl_record::encode_records(&cohort)?;
                if saved.is_some_and(|saved| *saved != records) {
                    return Err("Semantic restored records differ after fresh source audit".into());
                }
                let mut proposal = prepare(
                    scope,
                    identity,
                    &cohort,
                    &self.actor_config,
                    &self.critic_config,
                    attempt,
                )?;
                proposal.records = Some(records);
                let admitted = Self::prepare_admission_metadata(
                    identity,
                    ledger,
                    self.attempt_base,
                    &mut proposal,
                    match pool {
                        PoolRef::Mixture(m) => Some(m),
                        _ => None,
                    },
                )?;
                Ok(PreparedSemanticAdmission { proposal, admitted })
            },
            create,
        );
        if result.is_err() {
            self.ledger.stop();
        }
        result
    }
    pub(crate) fn journal_recheck(&self) -> Result<(), String> {
        self.ledger
            .recheck(&self.state.view(self.identity.checksum()))
    }
    pub(crate) fn journal_recheck_with_pool(&self, pool: PoolRef<'_>) -> Result<(), String> {
        self.ledger
            .recheck_with_pool(&self.state.view(self.identity.checksum()), pool)
    }
    pub(crate) fn journal_source(&self) -> Result<serde_json::Value, String> {
        Ok(
            serde_json::json!({"sources":self.ledger.journal_source(&self.state.view(self.identity.checksum()))?,
            "freshRoot":{"checksum":self.state.root().checksum(),"criticSeed":self.state.root().critic_seed()},
            "actorConfig":self.actor_config,"criticConfig":self.critic_config,"current":self.identity}),
        )
    }
    pub(crate) fn current_unchecked(&self) -> &SemanticPlayingCurrentIdentity {
        &self.identity
    }
    pub(crate) fn capture_ledger(&self) -> serde_json::Value {
        self.ledger.capture_ledger()
    }
    pub(crate) fn stop(&mut self) {
        self.ledger.stop();
    }
}

pub(crate) struct PreparedSemanticAdmission {
    proposal: JointProposal,
    admitted: BTreeSet<String>,
}
impl PreparedSemanticAdmission {
    pub(crate) fn report(&self) -> &SemanticPlayingJointReport {
        &self.proposal.report
    }
    pub(crate) fn take_signals(&mut self) -> Option<SemanticSignalDiagnostics> {
        self.proposal.report.signals.take()
    }
    pub(crate) fn records(&self) -> Result<&[Vec<u8>; 4], String> {
        self.proposal
            .records
            .as_ref()
            .ok_or_else(|| "Semantic staged records missing".into())
    }
}
struct JointProposal {
    records: Option<[Vec<u8>; 4]>,
    identity: SemanticPlayingCurrentIdentity,
    parameters: Option<(PublicPolicyModel, Vec<f32>)>,
    report: SemanticPlayingJointReport,
}
fn prepare(
    scope: &SemanticCollectionScope<'_>,
    parent: &SemanticPlayingCurrentIdentity,
    cohort: &AuditedSemanticPlayingCohort,
    actor_config: &OneStepConfig,
    critic_config: &ResidualFitConfig,
    attempt: usize,
) -> Result<JointProposal, String> {
    if scope.parent_checksum() != parent.checksum()
        || cohort.parent_checksum() != parent.checksum()
        || cohort
            .episodes()
            .iter()
            .any(|e| e.composition() != parent.composition())
    {
        return Err("Semantic joint numerical parent mismatch".into());
    }
    let episodes = cohort.episodes();
    if episodes
        .iter()
        .enumerate()
        .any(|(i, e)| e.config().players() != if i < 2 { 3 } else { 4 })
    {
        return Err("Semantic joint requires the exact whole-four 3,3,4,4 order".into());
    }
    let members = episodes
        .iter()
        .map(|episode| {
            let steps = (0..episode.playing_steps())
                .map(|i| {
                    episode
                        .step(i)
                        .ok_or("Semantic sealed numerical step missing".to_string())
                })
                .collect::<Result<Vec<_>, _>>()?;
            let learner = steps
                .first()
                .ok_or("Semantic joint has no learner Playing steps")?
                .actor();
            Ok(SemanticNumericMember {
                learner,
                players: episode.config().players(),
                coefficient: 0.25,
                steps,
            })
        })
        .collect::<Result<Vec<_>, String>>()?;
    let members = members
        .try_into()
        .map_err(|_| "Semantic numeric whole-four shape")?;
    let counts = cohort.numeric_counts()?;
    let mut signals = SignalCollector::default();
    let (_, actor, critic) = prepare_numeric_timed_for_profile(
        ConditionalFeatureProfile::SemanticPlaying4,
        scope.actor(),
        scope.critic(),
        &members,
        counts,
        actor_config,
        critic_config,
        &mut signals,
    )?;
    let changes = [actor.numeric.changed_parameters, critic.changed_parameters];
    let updated = changes != [0, 0];
    let credit = if updated { counts[0] as u64 } else { 0 };
    let updates = parent
        .updates
        .checked_add(u64::from(updated))
        .ok_or("Semantic update count overflow")?;
    let total_credit = parent
        .credit
        .checked_add(credit)
        .ok_or("Semantic Playing credit overflow")?;
    let cohort_checksum = cohort.seal_checksum()?;
    let next_ordinal = episodes[3]
        .config()
        .sampling_identity()
        .episode_ordinal()
        .checked_add(1)
        .ok_or("Semantic next ordinal overflow")?;
    let identity = make_identity(
        &parent.root,
        parent.checksum(),
        &actor.parameters,
        &critic.parameters,
        updates,
        total_credit,
        parent
            .attempts
            .checked_add(1)
            .ok_or("Semantic cumulative attempt overflow")?,
        next_ordinal,
        &cohort_checksum,
        actor_config,
        critic_config,
        parent.composition(),
    )?;
    let report = SemanticPlayingJointReport {
        signals: Some(signals.finish(actor.numeric, counts)?),
        parent: parent.checksum().into(),
        next: String::new(),
        cohort: cohort_checksum,
        families: cohort.families().clone(),
        counts,
        changes,
        attempts: attempt,
        next_ordinal: 0,
        credit,
        residual_loss: critic.mean_loss,
        mixture: cohort.mixture_checksum().map(str::to_string),
        opponent_samples: episodes.iter().map(|e| e.opponent_playing_samples()).sum(),
        total_rows: episodes
            .iter()
            .map(|e| e.total_sampled_candidate_rows())
            .sum(),
        actor_draws: episodes.each_ref().map(|e| *e.actor_draws()),
        checksum: String::new(),
        guarded_proposals: cohort.guarded_proposals(),
    };
    let parameters = if updated {
        let model = PublicPolicyModel {
            parameters: actor.parameters,
        };
        model.validate()?;
        Some((model, critic.parameters))
    } else {
        None
    };
    Ok(JointProposal {
        records: None,
        identity,
        parameters,
        report,
    })
}

#[allow(clippy::too_many_arguments)]
fn make_identity(
    root: &str,
    previous: &str,
    actor: &[f32],
    critic: &[f32],
    updates: u64,
    credit: u64,
    attempts: u64,
    next: u64,
    cohort: &str,
    actor_config: &OneStepConfig,
    critic_config: &ResidualFitConfig,
    composition: SemanticCollectionComposition,
) -> Result<SemanticPlayingCurrentIdentity, String> {
    let actor = crate::conditional_semantic_playing_rl::parameter_checksum(
        b"semantic-current-actor-bits-v1\0",
        actor,
    );
    let critic = crate::conditional_semantic_playing_rl::parameter_checksum(
        b"semantic-current-critic-bits-v1\0",
        critic,
    );
    let checksum = hash(
        b"semantic-playing-current-owned-identity-v1\0",
        &(
            (
                UPDATE_VERSION,
                crate::features::SEMANTIC_PLAYING_FEATURE_SCHEMA,
                crate::features::PUBLIC_FEATURE_SCHEMA,
                crate::public_state_critic::CONTEXT_CONTRACT,
            ),
            (
                root, previous, &actor, &critic, updates, credit, attempts, next, cohort,
            ),
            (
                actor_config,
                critic_config,
                crate::semantic_playing_rl_collection::TARGET_CONTRACT,
                "scalar",
                false,
                false,
            ),
            (
                crate::public_policy_likelihood::LIKELIHOOD_VERSION,
                crate::public_policy_likelihood::CORRECTION_VERSION,
                crate::public_policy_pullback::PULLBACK_VERSION,
                crate::public_policy_baseline::FIT_VERSION,
                crate::semantic_playing_rl_collection::SAMPLING_VERSION,
                crate::public_stochastic::RNG_VERSION,
            ),
        ),
    )?;
    Ok(SemanticPlayingCurrentIdentity {
        root: root.into(),
        previous: previous.into(),
        actor,
        critic,
        updates,
        credit,
        attempts,
        next,
        checksum: composition.bind(checksum)?,
        cohort: cohort.into(),
        collection_composition: (composition != SemanticCollectionComposition::LegacyV1)
            .then_some(composition),
        trade_guard: (composition != SemanticCollectionComposition::LegacyV1)
            .then(tzolkin_inference::trade_guard::TradeGuardConfig::default),
    })
}

/// Extends the existing loader/numeric fixture, with no full-game test added.
#[cfg(test)]
pub(crate) fn joint_fixture() {
    use crate::dataset::{DatasetSplit, seed_family_id, split_for_family};
    use crate::human_initial_wealth::ContentPin;
    use crate::model::digest;
    use crate::policy::HeuristicWeights;
    use serde_json::json;
    use std::{cell::Cell, fs};
    mod temp_root {
        include!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/support/temp_root.rs"
        ));
    }
    struct Files(std::path::PathBuf);
    impl Drop for Files {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }
    let files = Files(temp_root::create("semantic-joint-original-fixture").unwrap());
    let prepared = loaded_fixture(&files.0);
    let root_owner = SemanticPlayingRlOwner::from_prepared(&prepared, 11).unwrap();
    let fresh = root_owner.identity().unwrap().clone();
    let excluded = seed_family_id(42);
    let raw = serde_json::to_vec(
        &json!({"schema":"tzolkin-conditional-native-exclusions-v1","families":[excluded]}),
    )
    .unwrap();
    let path = files.0.join("external.json");
    fs::write(&path, &raw).unwrap();
    let pin = ContentPin {
        path: path.to_str().unwrap().into(),
        bytes: raw.len() as u64,
        sha256: digest(&raw),
    };
    let external = ConditionalNativeExclusions::load(
        &path,
        &pin,
        1,
        &digest(format!("{excluded}\n").as_bytes()),
    )
    .unwrap();
    let history = DeclaredSamplingHistory::declare_complete(
        vec![
            crate::semantic_playing_rl_collection::DeclaredSamplingRange::new(
                SamplingSeed::new(17),
                0,
                3,
                0,
                90,
                94,
            )
            .unwrap(),
        ],
        vec![pin.clone()],
    )
    .unwrap();
    let mut session = SemanticPlayingJointSession::from_fresh(
        root_owner,
        &external,
        &history,
        SamplingSeed::new(17),
        100,
        0,
        OneStepConfig::default(),
        ResidualFitConfig::new(0.001, 0.01, 256).unwrap(),
    )
    .unwrap();
    assert_eq!(fresh.completed_updates(), 0);
    assert_eq!(session.identity().unwrap().completed_updates(), 0);
    let mut seeds = (0..100).filter(|s| {
        *s != 42 && split_for_family(&seed_family_id(*s)).unwrap() == DatasetSplit::Train
    });
    let opponent = json!({"kind":"heuristic"});
    let pool = ConditionalFrozenPool::from_json(&serde_json::to_vec(&json!({
        "schema":"tzolkin-conditional-human-mixed-frozen-pool-v1", "weights":HeuristicWeights::default(),
        "rosters":[[null,opponent,opponent],[null,opponent,opponent],
            [null,opponent,opponent,opponent],[null,opponent,opponent,opponent]]
    })).unwrap()).unwrap();
    let limits = CollectionLimits::new(64, 65536, 8 * 1024 * 1024, 1024 * 1024).unwrap();
    crate::semantic_playing_rl_journal::journal_fixture(
        &files.0, &prepared, &external, &history, &pool, limits,
    );
    assert!(session.check_headroom(CollectionLimits::default()).is_err());
    assert_eq!(
        (session.attempts(), session.consumed_actor_streams()),
        (0, 0)
    );
    let mut credit = 0;
    let create = |config: &crate::public_stochastic_native::NativeStochasticConfig| {
        let mut state = crate::conditional_human_rl_update::tests::terminal_start(
            config.players(),
            config.environment_seed(),
        );
        state.turn.mode = tzolkin_core::TurnMode::None;
        state.turn.count = 0;
        Ok(state)
    };
    // The same checked source can start an explicitly distinct guarded session;
    // legacy identity bytes have neither the mode nor a guard field.
    let legacy_identity = serde_json::to_value(session.identity().unwrap()).unwrap();
    assert!(legacy_identity.get("collectionComposition").is_none());
    assert!(legacy_identity.get("tradeGuard").is_none());
    let mut guarded = SemanticPlayingJointSession::from_fresh_guarded(
        SemanticPlayingRlOwner::from_prepared(&prepared, 11).unwrap(),
        &external,
        &history,
        SamplingSeed::new(29),
        100,
        0,
        OneStepConfig::default(),
        ResidualFitConfig::new(0.001, 0.01, 256).unwrap(),
    )
    .unwrap();
    assert_eq!(
        guarded.actor_parameters().unwrap(),
        session.actor_parameters().unwrap()
    );
    assert_eq!(
        guarded.critic_parameters().unwrap(),
        session.critic_parameters().unwrap()
    );
    assert_ne!(
        guarded.identity().unwrap().checksum(),
        session.identity().unwrap().checksum()
    );
    let guard_parent = guarded.identity.checksum().to_string();
    let mut guarded_seeds = (700..800)
        .filter(|seed| split_for_family(&seed_family_id(*seed)).unwrap() == DatasetSplit::Train);
    let families = std::array::from_fn(|_| guarded_seeds.next().unwrap());
    let proposed = guarded
        .ledger
        .collect_fixture_using(
            &guarded.state.view(&guard_parent),
            families,
            &pool,
            limits,
            |scope, cohort| {
                prepare(
                    scope,
                    &guarded.identity,
                    &cohort,
                    &guarded.actor_config,
                    &guarded.critic_config,
                    1,
                )
            },
            create,
        )
        .unwrap();
    let mut report = guarded.admit(proposed).unwrap();
    assert_eq!(report.guarded_proposals.unwrap()[0], report.counts[0]);
    let old_report_wire = serde_json::to_vec(&report).unwrap();
    let signals = report.signals.take().unwrap();
    assert_eq!(old_report_wire, serde_json::to_vec(&report).unwrap());
    assert_eq!(signals.report_checksum, report.checksum);
    assert_eq!(signals.actor.changed_parameters, report.changes[0]);
    assert_eq!(
        signals.members.iter().map(|m| m.callbacks).sum::<usize>(),
        report.counts[0]
    );
    assert_eq!(
        signals.members.iter().map(|m| m.singletons).sum::<usize>(),
        report.counts[2]
    );
    for m in &signals.members {
        assert!(m.baseline.min >= 0.0 && m.baseline.max <= 1.0);
        assert_eq!(
            m.advantage.positive + m.advantage.negative + m.advantage.zero,
            m.callbacks
        );
        assert!((m.advantage.mean - (m.terminal_winner_share - m.baseline.mean)).abs() < 1e-12);
    }
    let saved = guarded.checkpoint_state().unwrap();
    {
        // Explicitly unqualified source fixture: the real native loader rejects
        // it. Exercise native numerical custody/Reserve ordering without a new
        // full-game CI fixture or claiming a successful source qualification.
        let initial = crate::conditional_human_rl_update::tests::terminal_start(3, 1);
        let observation =
            tzolkin_core::observation::observe(&initial, initial.current_player).unwrap();
        let directory = files.0.join("native-origin-cache");
        let native_prepared = std::sync::Arc::new(
            crate::arena::teacher::PreparedSemanticPlayingTeacher::unqualified_rl_fixture(
                &directory,
                observation,
            ),
        );
        let families: Vec<_> = native_prepared
            .rl_checkpoint()
            .dataset
            .manifest
            .families
            .iter()
            .map(|f| f.family_id.clone())
            .collect();
        let exclusions = |name: &str, families: &[String]| {
            let raw =
                serde_json::to_vec(&json!({"schema":"tzolkin-conditional-native-exclusions-v1",
                "families":families}))
                .unwrap();
            let path = files.0.join(name);
            fs::write(&path, &raw).unwrap();
            let pin = ContentPin {
                path: path.to_str().unwrap().into(),
                bytes: raw.len() as u64,
                sha256: digest(&raw),
            };
            let excluded = ConditionalNativeExclusions::load(
                &path,
                &pin,
                families.len(),
                &digest(format!("{}\n", families.join("\n")).as_bytes()),
            )
            .unwrap();
            (excluded, pin)
        };
        let (all, all_pin) = exclusions("native-all.json", &families);
        let (missing, _) = exclusions("native-missing-one.json", &families[..families.len() - 1]);
        let history = DeclaredSamplingHistory::declare_complete(vec![], vec![all_pin]).unwrap();
        let owner =
            || SemanticFreshOwner::native(std::sync::Arc::clone(&native_prepared), 11).unwrap();
        let state = SemanticTrainingState::from_fresh(owner()).unwrap();
        assert_eq!(
            state.root().policy_version(),
            crate::conditional_semantic_playing_rl::NATIVE_POLICY_VERSION
        );
        let refused = SemanticCollectionLedger::new_composed(
            &missing,
            &history,
            SamplingSeed::new(991),
            0,
            0,
            SemanticCollectionComposition::GuardedProposalV2,
        )
        .unwrap();
        let seeds: [u32; 4] = (10_000..)
            .filter(|s| split_for_family(&seed_family_id(*s)).unwrap() == DatasetSplit::Train)
            .take(4)
            .collect::<Vec<_>>()
            .try_into()
            .unwrap();
        let error = refused
            .plan(
                &state.view(state.root().checksum()),
                seeds,
                PoolRef::Legacy(&pool, None),
                limits,
            )
            .err()
            .unwrap();
        assert!(error.contains("omits a retained teacher family"));
        assert_eq!(
            (
                refused.attempts(),
                refused.next_episode_ordinal(),
                refused.consumed_actor_streams()
            ),
            (0, 0, 0)
        );
        assert!(refused.consumed_families().is_empty());
        let mut native = SemanticPlayingJointSession::from_fresh_owner(
            owner(),
            &all,
            &history,
            SamplingSeed::new(991),
            0,
            0,
            OneStepConfig::default(),
            ResidualFitConfig::new(0.001, 0.01, 256).unwrap(),
            SemanticCollectionComposition::GuardedProposalV2,
        )
        .unwrap();
        assert_eq!(native.identity().unwrap().completed_updates(), 0);
        assert_eq!(native.identity().unwrap().credited_playing_decisions(), 0);
        assert!(
            native
                .actor_parameters()
                .unwrap()
                .iter()
                .zip(native_prepared.rl_checkpoint().model.parameters())
                .all(|(a, b)| a.to_bits() == b.to_bits())
        );
        assert!(native.check_headroom(CollectionLimits::default()).is_err());
        let snapshot = native.freeze_snapshot().unwrap();
        for family in &families {
            assert!(snapshot.contains_family(family));
        }
        let parent = native.identity().unwrap().checksum().to_string();
        let proposed = native
            .ledger
            .collect_fixture_using(
                &native.state.view(&parent),
                seeds,
                &pool,
                limits,
                |scope, cohort| {
                    prepare(
                        scope,
                        &native.identity,
                        &cohort,
                        &native.actor_config,
                        &native.critic_config,
                        1,
                    )
                },
                create,
            )
            .unwrap();
        let report = native.admit(proposed).unwrap();
        assert!(report.updated());
        let native_saved = native.checkpoint_state().unwrap();
        assert!(
            SemanticPlayingJointSession::from_checkpoint_owner(
                owner(),
                saved.clone(),
                &all,
                &history,
                SamplingSeed::new(993),
                saved.current["next"].as_u64().unwrap(),
                0,
                OneStepConfig::default(),
                ResidualFitConfig::new(0.001, 0.01, 256).unwrap()
            )
            .is_err()
        );
        assert!(
            SemanticPlayingJointSession::from_checkpoint_owner(
                SemanticFreshOwner::Human(Box::new(
                    SemanticPlayingRlOwner::from_prepared(&prepared, 11).unwrap()
                )),
                native_saved.clone(),
                &all,
                &history,
                SamplingSeed::new(993),
                native.next_episode_ordinal(),
                0,
                OneStepConfig::default(),
                ResidualFitConfig::new(0.001, 0.01, 256).unwrap()
            )
            .is_err()
        );
        let resumed = SemanticPlayingJointSession::from_checkpoint_owner(
            owner(),
            native_saved,
            &all,
            &history,
            SamplingSeed::new(993),
            native.next_episode_ordinal(),
            0,
            OneStepConfig::default(),
            ResidualFitConfig::new(0.001, 0.01, 256).unwrap(),
        )
        .unwrap();
        assert_eq!(
            resumed.identity().unwrap().checksum(),
            native.identity().unwrap().checksum()
        );
        let source =
            directory.join(&native_prepared.rl_checkpoint().dataset.manifest.games[0].source_file);
        let before = fs::read(&source).unwrap();
        let parent = native.identity().unwrap().checksum().to_string();
        let error = native
            .ledger
            .with_checked(&native.state.view(&parent), |_| {
                fs::write(&source, b"changed unqualified native source").unwrap();
                Err::<(), String>("synthetic native early error".into())
            })
            .unwrap_err();
        assert!(
            error.contains("synthetic native early error") && error.contains("source postcheck")
        );
        assert_eq!(
            (
                native.attempts(),
                native.consumed_families().len(),
                native.consumed_actor_streams()
            ),
            (1, 4, 14)
        );
        fs::write(&source, before).unwrap();
    }
    assert_eq!(
        saved.composition().unwrap(),
        SemanticCollectionComposition::GuardedProposalV2
    );
    let bytes = serde_json::to_vec(&saved).unwrap();
    let saved: SemanticCheckpointState = serde_json::from_slice(&bytes).unwrap();
    let restored_guard = SemanticPlayingJointSession::from_checkpoint(
        SemanticPlayingRlOwner::from_prepared(&prepared, 11).unwrap(),
        saved,
        &external,
        &history,
        SamplingSeed::new(31),
        guarded.identity.next,
        0,
        OneStepConfig::default(),
        ResidualFitConfig::new(0.001, 0.01, 256).unwrap(),
    )
    .unwrap();
    assert_eq!(
        restored_guard.identity().unwrap().composition(),
        SemanticCollectionComposition::GuardedProposalV2
    );
    assert_eq!(
        restored_guard.actor_parameters().unwrap(),
        guarded.actor_parameters().unwrap()
    );
    assert_eq!(
        restored_guard.critic_parameters().unwrap(),
        guarded.critic_parameters().unwrap()
    );
    for update in 1..=2 {
        let saved = (update == 2).then(|| session.checkpoint_state().unwrap());
        let mut warm_records = None;
        let families = std::array::from_fn(|_| seeds.next().unwrap());
        let previous = session.identity().unwrap().checksum().to_string();
        let proposal = session
            .ledger
            .collect_fixture_using(
                &session.state.view(&previous),
                families,
                &pool,
                limits,
                |scope, cohort| {
                    assert_eq!(cohort.parent_checksum(), previous);
                    let records =
                        crate::semantic_playing_rl_record::encode_records(&cohort).unwrap();
                    warm_records = Some(records.clone());
                    let stored = crate::semantic_playing_rl_record::store_records(&cohort).unwrap();
                    for (raw, saved) in records.iter().zip(&stored) {
                        assert_eq!(
                            crate::public_rl_record_storage::decode(
                                saved.stored_bytes(),
                                saved.raw_bytes(),
                                saved.raw_sha256(),
                            )
                            .unwrap(),
                            *raw,
                        );
                    }
                    assert!(
                        cohort
                            .episodes()
                            .iter()
                            .all(|e| e.learner_setup_choices() == 0)
                    );
                    prepare(
                        scope,
                        &session.identity,
                        &cohort,
                        &session.actor_config,
                        &session.critic_config,
                        update,
                    )
                },
                create,
            )
            .unwrap();
        let report = session.admit(proposal).unwrap();
        if let Some(saved) = saved {
            let raw = serde_json::to_vec(&saved).unwrap();
            let saved: SemanticCheckpointState = serde_json::from_slice(&raw).unwrap();
            let resume = |saved, first, config| {
                SemanticPlayingJointSession::from_checkpoint(
                    SemanticPlayingRlOwner::from_prepared(&prepared, 11).unwrap(),
                    saved,
                    &external,
                    &history,
                    SamplingSeed::new(17),
                    first,
                    0,
                    config,
                    ResidualFitConfig::new(0.001, 0.01, 256).unwrap(),
                )
            };
            let mut bad = saved.clone();
            bad.actor.parameters[0] = f32::NAN;
            assert!(resume(bad, 104, OneStepConfig::default()).is_err());
            let mut bad = saved.clone();
            bad.root = "0".repeat(64);
            assert!(resume(bad, 104, OneStepConfig::default()).is_err());
            assert!(resume(saved.clone(), 105, OneStepConfig::default()).is_err());
            assert!(resume(saved.clone(), 103, OneStepConfig::default()).is_err());
            let changed = OneStepConfig::new(0.0002, 0.01, 4_000_000).unwrap();
            assert!(resume(saved.clone(), 104, changed).is_err());
            let mut resumed = resume(saved, 104, OneStepConfig::default()).unwrap();
            assert_eq!(resumed.attempts(), 0);
            let proposal = resumed
                .ledger
                .collect_fixture_using(
                    &resumed.state.view(&previous),
                    families,
                    &pool,
                    limits,
                    |scope, cohort| {
                        assert_eq!(
                            crate::semantic_playing_rl_record::encode_records(&cohort).unwrap(),
                            warm_records.clone().unwrap()
                        );
                        prepare(
                            scope,
                            &resumed.identity,
                            &cohort,
                            &resumed.actor_config,
                            &resumed.critic_config,
                            1,
                        )
                    },
                    create,
                )
                .unwrap();
            let resumed_report = resumed.admit(proposal).unwrap();
            assert_eq!(
                serde_json::to_value(&resumed_report).unwrap(),
                serde_json::to_value(&report).unwrap()
            );
            assert_eq!(
                resumed.identity().unwrap().checksum(),
                session.identity().unwrap().checksum()
            );
            assert_eq!(
                resumed
                    .actor_parameters()
                    .unwrap()
                    .iter()
                    .map(|v| v.to_bits())
                    .collect::<Vec<_>>(),
                session
                    .actor_parameters()
                    .unwrap()
                    .iter()
                    .map(|v| v.to_bits())
                    .collect::<Vec<_>>()
            );
            assert_eq!(
                resumed
                    .critic_parameters()
                    .unwrap()
                    .iter()
                    .map(|v| v.to_bits())
                    .collect::<Vec<_>>(),
                session
                    .critic_parameters()
                    .unwrap()
                    .iter()
                    .map(|v| v.to_bits())
                    .collect::<Vec<_>>()
            );
            assert_eq!(
                (
                    resumed.attempts(),
                    resumed.identity().unwrap().attempted_whole_four()
                ),
                (1, 2)
            );
            resumed.stop();
            assert!(resumed.checkpoint_state().is_err());
        }
        assert!(report.updated());
        assert_eq!(report.parent_checksum(), previous);
        assert!(report.changed_parameters()[0] > 0);
        assert!(report.changed_parameters()[1] > 0);
        credit += report.counts()[0] as u64;
        let current = session.identity().unwrap();
        assert_eq!(current.completed_updates(), update as u64);
        assert_eq!(current.credited_playing_decisions(), credit);
        assert_eq!(current.root_checksum(), fresh.checksum());
        assert_eq!(current.immediate_parent_checksum(), previous);
        assert_eq!(session.next_episode_ordinal(), 100 + 4 * update as u64);
        assert_eq!(session.consumed_actor_streams(), 14 * update);
        assert_eq!(session.consumed_families().len(), 4 * update);
    }
    // A real captured actor is independent of the learner's next update.
    // Every preselected snapshot keeps the full unloaded Test reservation too.
    use crate::semantic_playing_rl_pool::SnapshotSlot;
    let snapshot = session.freeze_snapshot().unwrap();
    let frozen_actor = snapshot.identity().unwrap().actor_checksum().to_string();
    let frozen_identity = snapshot.identity().unwrap().checksum().to_string();
    assert_eq!(
        snapshot.human_table_reservations().unwrap()["125"],
        DatasetSplit::Test
    );
    let slots = || {
        vec![
            SnapshotSlot::new(0, 1, &snapshot).unwrap(),
            SnapshotSlot::new(1, 2, &snapshot).unwrap(),
            SnapshotSlot::new(2, 1, &snapshot).unwrap(),
            SnapshotSlot::new(3, 3, &snapshot).unwrap(),
        ]
    };
    let mixture = SemanticFrozenMixture::new(&pool, slots()).unwrap();
    // Each restriction is exercised before a game, attempt or new stream burn.
    // The only snapshot is in member3, not the first member created.
    let lineage_pool =
        SemanticFrozenMixture::new(&pool, vec![SnapshotSlot::new(3, 3, &snapshot).unwrap()])
            .unwrap();
    let empty_history = DeclaredSamplingHistory::declare_complete(vec![], vec![pin]).unwrap();
    let other_families = std::array::from_fn(|_| seeds.next().unwrap());
    let reject = |mut other: SemanticPlayingJointSession<'_>,
                  mixture: &SemanticFrozenMixture<'_>,
                  expected: &str| {
        let entered = std::sync::atomic::AtomicBool::new(false);
        let result = other.ledger.collect_mixture_fixture_using(
            &other.state.view(other.identity.checksum()),
            other_families,
            mixture,
            limits,
            |_, _| Ok(()),
            |_| {
                entered.store(true, std::sync::atomic::Ordering::Relaxed);
                Err("Lineage refusal must precede game creation".into())
            },
        );
        assert_eq!(result.unwrap_err(), expected);
        assert!(!entered.load(std::sync::atomic::Ordering::Relaxed));
        assert_eq!(
            (
                other.attempts(),
                other.next_episode_ordinal(),
                other.consumed_actor_streams()
            ),
            (0, 108, 0)
        );
    };
    for (critic_seed, declaration, expected) in [
        (12, &history, "Semantic snapshot has a foreign fresh root"),
        (
            11,
            &empty_history,
            "Semantic snapshot source declaration differs",
        ),
        (
            11,
            &history,
            "Semantic snapshot prior burns are not retained",
        ),
    ] {
        let other = SemanticPlayingJointSession::from_fresh(
            SemanticPlayingRlOwner::from_prepared(&prepared, critic_seed).unwrap(),
            &external,
            declaration,
            SamplingSeed::new(17),
            108,
            0,
            OneStepConfig::default(),
            ResidualFitConfig::new(0.001, 0.01, 256).unwrap(),
        )
        .unwrap();
        reject(other, &lineage_pool, expected);
    }
    // Same root, source declarations and empty burn sets are insufficient:
    // another real session's different configuration makes a nonancestor ID.
    let fork = SemanticPlayingJointSession::from_fresh(
        SemanticPlayingRlOwner::from_prepared(&prepared, 11).unwrap(),
        &external,
        &history,
        SamplingSeed::new(17),
        108,
        0,
        OneStepConfig::new(0.0002, 0.01, OneStepConfig::default().max_candidate_rows()).unwrap(),
        ResidualFitConfig::new(0.001, 0.01, 256).unwrap(),
    )
    .unwrap();
    let fork_snapshot = fork.freeze_snapshot().unwrap();
    let fork_pool = SemanticFrozenMixture::new(
        &pool,
        vec![SnapshotSlot::new(3, 3, &fork_snapshot).unwrap()],
    )
    .unwrap();
    let other = SemanticPlayingJointSession::from_fresh(
        SemanticPlayingRlOwner::from_prepared(&prepared, 11).unwrap(),
        &external,
        &history,
        SamplingSeed::new(17),
        108,
        0,
        OneStepConfig::default(),
        ResidualFitConfig::new(0.001, 0.01, 256).unwrap(),
    )
    .unwrap();
    assert_eq!(
        fork_snapshot.identity().unwrap().root_checksum(),
        other.identity().unwrap().root_checksum()
    );
    assert!(fork_snapshot.consumed_families().is_empty());
    reject(
        other,
        &fork_pool,
        "Semantic snapshot is not an admitted ancestor",
    );
    assert!(
        SemanticFrozenMixture::new(&pool, vec![SnapshotSlot::new(0, 0, &snapshot).unwrap()])
            .is_err()
    );
    assert!(
        SemanticFrozenMixture::new(
            &pool,
            vec![
                SnapshotSlot::new(0, 1, &snapshot).unwrap(),
                SnapshotSlot::new(0, 1, &snapshot).unwrap()
            ]
        )
        .is_err()
    );
    for kind in [
        json!({"kind":"preparedBc"}),
        json!({"kind":"preparedBcTradeGuard","guard":crate::public_trade_guard::TradeGuardConfig::default()}),
    ] {
        let mut wire = json!({"schema":"tzolkin-conditional-human-mixed-frozen-pool-v1","weights":HeuristicWeights::default(),
            "rosters":[[null,opponent,opponent],[null,opponent,opponent],
                [null,opponent,opponent,opponent],[null,opponent,opponent,opponent]]});
        wire["rosters"][0][1] = kind;
        let forbidden =
            ConditionalFrozenPool::from_json(&serde_json::to_vec(&wire).unwrap()).unwrap();
        assert!(SemanticFrozenMixture::new(&forbidden, slots()).is_err());
    }
    for update in 3..=4 {
        // The second mixed update uses the actual re-frozen first mixed result.
        let current_snapshot = session.freeze_snapshot().unwrap();
        let active_mixture = SemanticFrozenMixture::new(
            &pool,
            vec![
                SnapshotSlot::new(0, 1, &current_snapshot).unwrap(),
                SnapshotSlot::new(1, 2, &current_snapshot).unwrap(),
                SnapshotSlot::new(2, 1, &current_snapshot).unwrap(),
                SnapshotSlot::new(3, 3, &current_snapshot).unwrap(),
            ],
        )
        .unwrap();
        let before_mixed = session.identity().unwrap().checksum().to_string();
        let proposal = session
            .ledger
            .collect_mixture_fixture_using(
                &session.state.view(&before_mixed),
                std::array::from_fn(|_| seeds.next().unwrap()),
                &active_mixture,
                limits,
                |scope, cohort| {
                    crate::semantic_playing_rl_collection::mixed_trace_fixture(
                        scope,
                        &active_mixture,
                        cohort.episodes()[0].config(),
                    );
                    assert_eq!(cohort.mixture_checksum(), Some(active_mixture.checksum()));
                    assert!(
                        cohort
                            .episodes()
                            .iter()
                            .all(|e| e.opponent_playing_samples() > 0)
                    );
                    prepare(
                        scope,
                        &session.identity,
                        &cohort,
                        &session.actor_config,
                        &session.critic_config,
                        update,
                    )
                },
                |config| {
                    let mut state = crate::conditional_human_rl_update::tests::terminal_start(
                        config.players(),
                        config.environment_seed(),
                    );
                    state.turn.mode = tzolkin_core::TurnMode::None;
                    state.turn.count = 0;
                    Ok(state)
                },
            )
            .unwrap();
        let report = session
            .admit_with_pool(proposal, Some(&active_mixture))
            .unwrap();
        assert!(report.updated() && report.opponent_playing_samples() > 0);
        assert!(report.total_sampled_candidate_rows() > report.counts()[1]);
        assert_eq!(
            report.actor_draws().iter().flatten().sum::<u64>(),
            report.counts()[0] as u64 + report.opponent_playing_samples() as u64
        );
        credit += report.counts()[0] as u64;
        assert_eq!(
            session.identity().unwrap().credited_playing_decisions(),
            credit
        );
        assert_eq!(
            session.identity().unwrap().completed_updates(),
            update as u64
        );
        assert!(
            session
                .ledger
                .admitted_checksums()
                .contains(session.identity().unwrap().checksum())
        );
        assert_eq!(session.ledger.admitted_checksums().len(), update + 1);
    }
    assert_eq!(snapshot.identity().unwrap().checksum(), frozen_identity);
    assert_eq!(snapshot.identity().unwrap().actor_checksum(), frozen_actor);
    assert_ne!(session.identity().unwrap().actor_checksum(), frozen_actor);
    let before_failure_snapshot = session.freeze_snapshot().unwrap();
    let before_failure_capture = before_failure_snapshot
        .identity()
        .unwrap()
        .capture_ledger_checksum()
        .to_string();
    // A metadata refusal and raw-byte mutation after BOTH numerical proposals
    // still reach POST. No actor/critic/current identity can be admitted.
    let before = session.identity.clone();
    let actor = session.state.actor_parameters().to_vec();
    let critic = session.state.critic_parameters().to_vec();
    let admitted = session.ledger.admitted_checksums().clone();
    let checkpoint_path = prepared.loaded_semantic_playing_sources().unwrap()[0]
        .path
        .clone();
    let checkpoint = fs::read(&checkpoint_path).unwrap();
    let entered = Cell::new(false);
    let plan = session
        .plan_reserved(
            std::array::from_fn(|_| seeds.next().unwrap()),
            PoolRef::Mixture(&mixture),
            limits,
        )
        .unwrap();
    let batch = session.burn_reserved(plan);
    let result = session.ledger.collect_reserved_records(
        &session.state.view(before.checksum()),
        &batch,
        PoolRef::Mixture(&mixture),
        None,
        |scope, cohort, ledger| {
            let mut proposed = prepare(
                scope,
                &before,
                &cohort,
                &session.actor_config,
                &session.critic_config,
                5,
            )?;
            entered.set(true);
            fs::write(&checkpoint_path, [checkpoint.as_slice(), b" "].concat()).unwrap();
            proposed.report.parent = "0".repeat(64);
            SemanticPlayingJointSession::prepare_admission_metadata(
                &before,
                ledger,
                session.attempt_base,
                &mut proposed,
                Some(&mixture),
            )
        },
        |config| {
            Ok(crate::conditional_human_rl_update::tests::terminal_start(
                config.players(),
                config.environment_seed(),
            ))
        },
    );
    assert!(entered.get() && result.is_err());
    let error = result.unwrap_err();
    assert!(error.contains("Semantic joint proposal differs"));
    assert!(error.contains("semantic source postcheck"));
    fs::write(&checkpoint_path, checkpoint).unwrap();
    assert_eq!(session.identity().unwrap().checksum(), before.checksum());
    assert_eq!(session.state.actor_parameters(), actor);
    assert_eq!(session.state.critic_parameters(), critic);
    assert_eq!(session.ledger.admitted_checksums(), &admitted);
    assert!(session.stopped());
    assert_eq!(
        (
            session.attempts(),
            session.next_episode_ordinal(),
            session.consumed_actor_streams()
        ),
        (5, 120, 70)
    );
    let after_failure = session.freeze_snapshot().unwrap();
    assert_eq!(
        after_failure
            .identity()
            .unwrap()
            .current_identity_checksum(),
        before.checksum()
    );
    assert_eq!(
        after_failure.identity().unwrap().actor_checksum(),
        before.actor_checksum()
    );
    assert_ne!(
        after_failure.identity().unwrap().capture_ledger_checksum(),
        before_failure_capture
    );
    assert_eq!(after_failure.consumed_families().len(), 20);
}

/// Two one-decision synthetic original records exercise the ACTUAL schema-4
/// dataset/checkpoint loader. They are not retained human games or source proof.
#[cfg(test)]
fn loaded_fixture(
    root: &std::path::Path,
) -> crate::human_policy_training::PreparedConditionalHumanPolicy {
    use crate::human_policy_dataset::{CAPPED_EXTRACTION_LABEL_SCOPE, INPUT_SCHEMA_V8};
    use crate::human_policy_training::{
        PreparedConditionalHumanPolicy, SEMANTIC_PLAYING_COLLECTION_SCHEMA,
        train_semantic_playing_collection_manifest,
    };
    use serde_json::{Value, json};
    use std::fs;
    use tzolkin_core::observation::{TypedAction, observe};
    use tzolkin_core::{Phase, apply_move, create_game};
    fn write(dir: &std::path::Path, name: &str, value: &Value) -> Value {
        let bytes = serde_json::to_vec(value).unwrap();
        fs::write(dir.join(name), &bytes).unwrap();
        json!({"path":name,"bytes":bytes.len(),"sha256":crate::model::digest(&bytes)})
    }
    let mut state = create_game(vec!["P0".into(), "P1".into(), "P2".into()], 42, false).unwrap();
    while state.phase == Phase::Setup {
        let observation = observe(&state, state.current_player).unwrap();
        state = apply_move(&state, observation.legal_actions[0].r#move.clone()).unwrap();
    }
    let observation = observe(&state, state.current_player).unwrap();
    let row = observation
        .legal_actions
        .iter()
        .find(|a| {
            matches!(
                a.action,
                TypedAction::Place {
                    discount: false,
                    ..
                }
            )
        })
        .unwrap();
    let TypedAction::Place {
        gear, corn_cost, ..
    } = row.action
    else {
        unreachable!()
    };
    let gear = match gear {
        tzolkin_core::GearId::Palenque => "Palenque",
        tzolkin_core::GearId::Yaxchilan => "Yaxchilan",
        tzolkin_core::GearId::Tikal => "Tikal",
        tzolkin_core::GearId::Uxmal => "Uxmal",
        tzolkin_core::GearId::ChichenItza => "Chichen Itza",
    };
    let caption = format!(
        "P{}は{gear}の歯車に{corn_cost} を支払ってワーカーを置いた",
        state.current_player
    );
    let mut inputs = Vec::new();
    for table in ["123", "124"] {
        let dir = root.join(table);
        fs::create_dir(&dir).unwrap();
        let url = format!("https://boardgamearena.com/gamereview?table={table}");
        let source = write(
            &dir,
            "raw.json",
            &json!({"schema_version":"1.0","source_type":"public_ui_visible_log",
            "export_scope":"all_observed_log_entries","table_id":table,"source_url":url,"entry_count":1,
            "metadata":{"table_id":table,"players":[{"text":"P0","url":"https://boardgamearena.com/player?id=101"},
                {"text":"P1","url":"https://boardgamearena.com/player?id=102"},{"text":"P2","url":"https://boardgamearena.com/player?id=103"}]},
            "entries":[{"action_id":10,"timestamp_display":"12:00","raw_text":caption}],
            "dom_entries":[{"action_id":10,"timestamp_display":"12:00","messages":[{"raw_text":caption}]}]}),
        );
        let record = write(
            &dir,
            "record.json",
            &json!({"schema":tzolkin_core::public_replay::PUBLIC_REPLAY_SCHEMA,
            "rulesVersion":tzolkin_core::public_replay::PUBLIC_RULES_VERSION,"catalogHash":tzolkin_core::public_replay::public_catalog_hash(),
            "market":"unlimited","source":{"reference":url,"actionIds":[]},
            "initial":tzolkin_core::public_replay::PublicState::from_game_state(&state).unwrap(),
            "initialCheckpoint":null,"steps":[{"actor":state.current_player,"move":row.r#move,
                "sourceActionIds":[10],"refills":{},"checkpoint":null}],"terminalCheckpoint":null,"terminalDisplayCheckpoint":null}),
        );
        let audit = write(
            &dir,
            "audit.json",
            &json!({"tableId":table,"sourceSha256":source["sha256"],
            "conditionalMechanicalProfile":{"schema":"tzolkin-bga-declared-mechanical-profile-v1","rules":"basic","market":"unlimited","sourceOptions":"unknown"},
            "observedOptions":{"tribes":null,"prophecies":null,"additionalBuildings":null,"quickActions":null},
            "audit":[{"step":1,"actionId":10,"actor":state.current_player,"action":row.action,"derivedFromVisibleBoundary":false,"equivalentChoices":1}]}),
        );
        let nominations = write(
            &dir,
            "nominations.json",
            &json!({"schema":"tzolkin-human-player-nomination-v1",
            "basis":"userNominatedHistoricalArenaTop5","playerIds":["101","102","103"]}),
        );
        let reservations = write(
            &dir,
            "reservations.json",
            &json!({"schema":"tzolkin-human-source-table-reservations-v1",
            "entries":[{"tableId":"123","split":"train","basis":"synthetic fixture"},
                {"tableId":"124","split":"validation","basis":"synthetic fixture"},
                {"tableId":"125","split":"test","basis":"unloaded synthetic fixture"}]}),
        );
        let manifest = write(
            &dir,
            "manifest.json",
            &json!({"schema":INPUT_SCHEMA_V8,"basis":"ConditionalBasicHumanPolicy",
            "featureProfile":"semanticPlaying4","labelScope":CAPPED_EXTRACTION_LABEL_SCOPE,"tableId":table,
            "observedOptions":{"tribes":null,"prophecies":null,"additionalBuildings":null,"quickActions":null},
            "record":record,"sourceLog":source,"decoderAudit":audit,"nominations":nominations,
            "reservations":reservations,"reconstructionEvidence":[]}),
        );
        inputs.push(json!({"manifestPath":dir.join("manifest.json").to_str().unwrap(),"manifestSha256":manifest["sha256"],
            "nominationsSha256":nominations["sha256"],"reservationsSha256":reservations["sha256"]}));
    }
    let collection = root.join("collection.json");
    fs::write(
        &collection,
        serde_json::to_vec(&json!({"schema":SEMANTIC_PLAYING_COLLECTION_SCHEMA,
        "config":{"epochs":1,"batchSize":1,"learningRate":0.001,"seed":7},"datasets":inputs}))
        .unwrap(),
    )
    .unwrap();
    let trained = train_semantic_playing_collection_manifest(&collection).unwrap();
    let output = root.join("checkpoint");
    trained
        .save_new_semantic_playing_directory(&output)
        .unwrap();
    PreparedConditionalHumanPolicy::load_semantic_playing(
        &output.join("checkpoint.json"),
        &collection,
    )
    .unwrap()
}
