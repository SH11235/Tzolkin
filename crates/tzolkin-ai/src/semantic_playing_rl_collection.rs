//! Separate in-memory schema-4 collection and fresh mechanical audit.
//! Setup is an applied Corn-first teacher choice, never a learner draw or
//! gradient step. No wire, journal, update, export or legacy-owner conversion.
use std::collections::{BTreeMap, BTreeSet};

use tzolkin_core::observation::{LegalAction, Observation, observe};
use tzolkin_core::{
    FinalScore, GameOptions, GameState, Phase, apply_move, create_game_with_options,
};

use crate::Decision;
use crate::conditional_human_rl_update::{
    ConditionalFrozenMixRoster, ConditionalFrozenPool, ConditionalNativeExclusions,
};
use crate::conditional_semantic_playing_rl::{
    SemanticCollectionScope, SemanticOwnerView, SemanticPlayingRlOwner,
};
use crate::dataset::{DatasetSplit, seed_family_id, split_for_family};
use crate::human_initial_wealth::ContentPin;
use crate::policy::HeuristicWeights;
use crate::public_native::PreparedPublicPolicy;
use crate::public_policy_update::hash;
use crate::public_stochastic::{
    SamplingSeed, SamplingStreamIdentity, TickDistribution, draw_tick53, full_logits_digest,
    legal_digest,
};
use crate::public_stochastic_native::{CollectionLimits, NativeStochasticConfig};
use crate::semantic_playing_rl_pool::{
    FrozenSemanticSnapshot, OPPONENT_SAMPLING_VERSION, SemanticFrozenMixture,
    SemanticMixtureRoster, SnapshotCaptureLedger,
};
use crate::semantic_playing_rl_update::SemanticPlayingCurrentIdentity;

pub const SAMPLING_VERSION: &str = "semantic-playing-only-uniform-tick53-v1";
pub const TARGET_CONTRACT: &str = "semantic-playing-only-gamma1-terminal-winner-share-v1";
const MAX_ATTEMPTS: usize = 64;
const MAX_RANGES: usize = 8192;

/// One fixed-domain five-u64 actor-stream interval. No policy label or native
/// environment seed is part of this sampling identity.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct DeclaredSamplingRange {
    prefix: [u64; 4], // sampling seed, replicate, players, absolute actor
    first: u64,
    next: u64,
}
impl DeclaredSamplingRange {
    pub fn new(
        seed: SamplingSeed,
        replicate: u64,
        players: usize,
        actor: usize,
        first: u64,
        next: u64,
    ) -> Result<Self, String> {
        if !(3..=4).contains(&players) || actor >= players || first >= next {
            return Err("Invalid declared sampling interval".into());
        }
        Ok(Self {
            prefix: [seed.value(), replicate, players as u64, actor as u64],
            first,
            next,
        })
    }
    fn contains(&self, identity: SamplingStreamIdentity, actor: usize) -> bool {
        self.prefix
            == [
                identity.sampling_seed().value(),
                identity.replicate_ordinal(),
                identity.players() as u64,
                actor as u64,
            ]
            && (self.first..self.next).contains(&identity.episode_ordinal())
    }
}

/// An explicit complete-history declaration bound to unchanged raw evidence.
/// Pins bind bytes; they do not authenticate the declaration or prove that the
/// caller inventoried all history. No new wire/receipt schema is interpreted.
pub struct DeclaredSamplingHistory {
    ranges: Vec<DeclaredSamplingRange>,
    pins: Vec<ContentPin>,
    checksum: String,
}
impl DeclaredSamplingHistory {
    pub(crate) fn recheck(&self) -> Result<(), String> {
        for pin in &self.pins {
            crate::human_policy_dataset::recheck_absolute_pin(pin)?;
        }
        Ok(())
    }
    pub(crate) fn pins(&self) -> &[ContentPin] {
        &self.pins
    }
    pub(crate) fn checksum(&self) -> &str {
        &self.checksum
    }
    pub fn declare_complete(
        ranges: Vec<DeclaredSamplingRange>,
        pins: Vec<ContentPin>,
    ) -> Result<Self, String> {
        if ranges.len() > MAX_RANGES
            || pins.is_empty()
            || pins.len() > MAX_RANGES
            || ranges
                .windows(2)
                .any(|p| p[0] >= p[1] || (p[0].prefix == p[1].prefix && p[0].next > p[1].first))
            || pins
                .iter()
                .enumerate()
                .any(|(i, p)| pins[..i].iter().any(|q| q.path == p.path))
        {
            return Err(
                "Sampling history requires a bounded sorted nonoverlapping complete declaration"
                    .into(),
            );
        }
        for pin in &pins {
            crate::human_policy_dataset::recheck_absolute_pin(pin)?;
        }
        let tuples = ranges
            .iter()
            .map(|r| (r.prefix, r.first, r.next))
            .collect::<Vec<_>>();
        let checksum = hash(
            b"tzolkin-semantic-playing-declared-sampling-history-v1\0",
            &(tuples, &pins),
        )?;
        Ok(Self {
            ranges,
            pins,
            checksum,
        })
    }
    fn contains(&self, identity: SamplingStreamIdentity, actor: usize) -> bool {
        self.ranges.iter().any(|r| r.contains(identity, actor))
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SemanticCallbackRole {
    LearnerSetup,
    LearnerPlaying,
    Opponent,
}

#[derive(Debug, PartialEq)]
struct SemanticSample {
    chosen: LegalAction,
    decision: Decision,
    index: usize,
    sample_index: usize,
    global_sample_index: usize,
    before: u64,
    after: u64,
    draws_before: u64,
    raw: u64,
    u53: u64,
    mass: u64,
    nominal_bits: u64,
    logp_bits: u64,
    logit_bits: u32,
    rows_before: usize,
    rows_after: usize,
    legal_digest: String,
    logits_digest: String,
    distribution_digest: String,
}
#[derive(Debug, PartialEq)]
enum Choice {
    LearnerSetup {
        decision: Decision,
        chosen: LegalAction,
    },
    LearnerPlaying(Box<SemanticSample>),
    OpponentSample {
        sample: Box<SemanticSample>,
        snapshot: String,
    },
    Opponent {
        decision: Decision,
        chosen: LegalAction,
    },
}
impl Choice {
    fn role(&self) -> SemanticCallbackRole {
        match self {
            Self::LearnerSetup { .. } => SemanticCallbackRole::LearnerSetup,
            Self::LearnerPlaying(_) => SemanticCallbackRole::LearnerPlaying,
            Self::Opponent { .. } | Self::OpponentSample { .. } => SemanticCallbackRole::Opponent,
        }
    }
    fn decision(&self) -> &Decision {
        match self {
            Self::LearnerSetup { decision, .. } | Self::Opponent { decision, .. } => decision,
            Self::LearnerPlaying(s) | Self::OpponentSample { sample: s, .. } => &s.decision,
        }
    }
}
#[derive(Debug, PartialEq)]
struct Callback {
    index: usize,
    observation: Observation,
    before: String,
    choice: Option<Choice>,
    apply_attempted: bool,
    after: Option<String>,
}
#[derive(Debug, PartialEq)]
struct Record {
    binding: String,
    callbacks: Vec<Callback>,
    final_state: Option<String>,
    scores: Vec<FinalScore>,
    failure: Option<String>,
    draws: [u64; 4],
    samples: usize,
    opponent_samples: usize,
    rows: usize,
}
impl Record {
    fn complete(&self) -> bool {
        self.failure.is_none() && self.final_state.is_some()
    }
}

/// Failed records remain diagnostics only, with no actor-step constructor.
pub struct SemanticPlayingCollectedGame {
    record: Record,
}
impl SemanticPlayingCollectedGame {
    pub fn complete(&self) -> bool {
        self.record.complete()
    }
    pub fn failure(&self) -> Option<&str> {
        self.record.failure.as_deref()
    }
    pub fn observed_callbacks(&self) -> usize {
        self.record.callbacks.len()
    }
    pub fn playing_samples(&self) -> usize {
        self.record.samples
    }
    pub fn opponent_playing_samples(&self) -> usize {
        self.record.opponent_samples
    }
    pub fn total_reserved_candidate_rows(&self) -> usize {
        self.record.rows
    }
    pub fn actor_draws(&self) -> &[u64; 4] {
        &self.record.draws
    }
    pub fn callback_role(&self, index: usize) -> Option<SemanticCallbackRole> {
        self.record
            .callbacks
            .get(index)?
            .choice
            .as_ref()
            .map(Choice::role)
    }
    pub fn applied_choices(&self) -> usize {
        self.record
            .callbacks
            .iter()
            .filter(|c| c.after.is_some())
            .count()
    }
}

/// A fresh replay, not raw metadata, constructs this complete episode.
pub struct AuditedSemanticPlayingEpisode {
    record: Record,
    config: NativeStochasticConfig,
    playing: Vec<usize>,
    rank: usize,
    winners: usize,
}
impl AuditedSemanticPlayingEpisode {
    pub fn config(&self) -> &NativeStochasticConfig {
        &self.config
    }
    pub fn target_contract(&self) -> &'static str {
        TARGET_CONTRACT
    }
    pub fn observed_callbacks(&self) -> usize {
        self.record.callbacks.len()
    }
    pub fn playing_steps(&self) -> usize {
        self.playing.len()
    }
    pub fn opponent_playing_samples(&self) -> usize {
        self.record.opponent_samples
    }
    pub fn total_sampled_candidate_rows(&self) -> usize {
        self.record.rows
    }
    pub fn learner_setup_choices(&self) -> usize {
        self.record
            .callbacks
            .iter()
            .filter(|c| matches!(c.choice, Some(Choice::LearnerSetup { .. })))
            .count()
    }
    pub fn callback_role(&self, index: usize) -> Option<SemanticCallbackRole> {
        self.record
            .callbacks
            .get(index)?
            .choice
            .as_ref()
            .map(Choice::role)
    }
    pub fn actor_draws(&self) -> &[u64; 4] {
        &self.record.draws
    }
    pub fn training_admission_available(&self) -> bool {
        false
    }
    pub fn producer_authenticated(&self) -> bool {
        false
    }
    pub fn independent_seed_origins_verified(&self) -> bool {
        false
    }
    pub fn step(&self, index: usize) -> Option<SemanticPlayingActorStep<'_>> {
        let &global = self.playing.get(index)?;
        let next = self.playing.get(index + 1).copied();
        let callback = &self.record.callbacks[global];
        let Some(Choice::LearnerPlaying(sample)) = &callback.choice else {
            return None;
        };
        Some(SemanticPlayingActorStep {
            callback,
            sample,
            ordinal: index,
            next,
            gap: next.unwrap_or(self.record.callbacks.len()) - global - 1,
            share: if self.rank == 1 {
                1.0 / self.winners as f64
            } else {
                0.0
            },
        })
    }
}
/// Borrowed Playing-only target. No Deserialize, public constructor, Setup
/// projection or conversion to the old AppliedActorStep exists.
pub struct SemanticPlayingActorStep<'a> {
    callback: &'a Callback,
    sample: &'a SemanticSample,
    ordinal: usize,
    next: Option<usize>,
    gap: usize,
    share: f64,
}
impl SemanticPlayingActorStep<'_> {
    pub fn observation(&self) -> &Observation {
        &self.callback.observation
    }
    pub fn chosen(&self) -> &LegalAction {
        &self.sample.chosen
    }
    pub fn chosen_index(&self) -> usize {
        self.sample.index
    }
    pub fn global_callback_index(&self) -> usize {
        self.callback.index
    }
    pub fn playing_step_index(&self) -> usize {
        self.ordinal
    }
    pub fn actor(&self) -> usize {
        self.callback.observation.actor
    }
    pub fn actor_draw_ordinal(&self) -> u64 {
        self.sample.draws_before
    }
    pub fn session_sample_index(&self) -> usize {
        self.sample.sample_index
    }
    pub fn state_before(&self) -> &str {
        &self.callback.before
    }
    pub fn state_after(&self) -> &str {
        self.callback
            .after
            .as_deref()
            .expect("freshly audited applied Playing step")
    }
    pub fn next_playing_global_index(&self) -> Option<usize> {
        self.next
    }
    pub fn intervening_callbacks(&self) -> usize {
        self.gap
    }
    pub fn behavior_mass_ticks(&self) -> u64 {
        self.sample.mass
    }
    pub fn behavior_probability(&self) -> f64 {
        self.sample.mass as f64 / crate::public_stochastic::TICKS as f64
    }
    pub fn nominal_probability(&self) -> f64 {
        f64::from_bits(self.sample.nominal_bits)
    }
    pub fn behavior_logp(&self) -> f64 {
        f64::from_bits(self.sample.logp_bits)
    }
    pub fn logits_digest(&self) -> &str {
        &self.sample.logits_digest
    }
    pub fn distribution_digest(&self) -> &str {
        &self.sample.distribution_digest
    }
    pub fn return_target(&self) -> f64 {
        self.share
    }
    pub fn reward(&self) -> f64 {
        if self.next.is_none() { self.share } else { 0.0 }
    }
}

pub struct AuditedSemanticPlayingCohort {
    episodes: [AuditedSemanticPlayingEpisode; 4],
    parent: String,
    families: [String; 4],
    human_reservations: BTreeMap<String, DatasetSplit>,
    mixture: Option<String>,
}
impl AuditedSemanticPlayingCohort {
    pub fn episodes(&self) -> &[AuditedSemanticPlayingEpisode; 4] {
        &self.episodes
    }
    pub fn parent_checksum(&self) -> &str {
        &self.parent
    }
    pub fn families(&self) -> &[String; 4] {
        &self.families
    }
    pub fn human_table_reservations(&self) -> &BTreeMap<String, DatasetSplit> {
        &self.human_reservations
    }
    pub fn credited_decisions(&self) -> u64 {
        0
    }
    pub fn mixture_checksum(&self) -> Option<&str> {
        self.mixture.as_deref()
    }
    pub(crate) fn numeric_counts(&self) -> Result<[usize; 3], String> {
        let mut counts = [0_usize; 3];
        for episode in &self.episodes {
            for i in 0..episode.playing_steps() {
                let step = episode
                    .step(i)
                    .ok_or("Semantic sealed Playing step missing")?;
                for (count, add) in counts.iter_mut().zip([
                    1,
                    step.observation().legal_actions.len(),
                    usize::from(step.observation().legal_actions.len() == 1),
                ]) {
                    *count = count
                        .checked_add(add)
                        .ok_or("Semantic numeric count overflow")?;
                }
            }
        }
        Ok(counts)
    }
    pub(crate) fn seal_checksum(&self) -> Result<String, String> {
        let mut records = Vec::new();
        for episode in &self.episodes {
            let mut callbacks = Vec::new();
            for callback in &episode.record.callbacks {
                let choice = callback
                    .choice
                    .as_ref()
                    .ok_or("Semantic sealed choice missing")?;
                let sample = if let Choice::LearnerPlaying(s)
                | Choice::OpponentSample { sample: s, .. } = choice
                {
                    Some(hash(
                        b"semantic-joint-applied-sample-v1\0",
                        &(
                            (&s.chosen, &s.decision, s.index, s.sample_index),
                            (s.before, s.after, s.draws_before, s.raw, s.u53, s.mass),
                            (
                                s.nominal_bits,
                                s.logp_bits,
                                s.logit_bits,
                                s.rows_before,
                                s.rows_after,
                            ),
                            (&s.legal_digest, &s.logits_digest, &s.distribution_digest),
                        ),
                    )?)
                } else {
                    None
                };
                let role = match choice.role() {
                    SemanticCallbackRole::LearnerSetup => 0,
                    SemanticCallbackRole::LearnerPlaying => 1,
                    SemanticCallbackRole::Opponent => 2,
                };
                let callback_checksum = hash(
                    b"semantic-joint-applied-callback-v1\0",
                    &(
                        callback.index,
                        &callback.observation,
                        &callback.before,
                        role,
                        choice.decision(),
                        callback.apply_attempted,
                        &callback.after,
                        sample,
                    ),
                )?;
                callbacks.push(
                    if let Choice::OpponentSample { sample, snapshot } = choice {
                        hash(
                            b"semantic-joint-sampled-opponent-callback-v1\0",
                            &(callback_checksum, snapshot, sample.global_sample_index),
                        )?
                    } else {
                        callback_checksum
                    },
                );
            }
            let identity = episode.config.sampling_identity();
            records.push(hash(
                b"semantic-joint-audited-episode-v1\0",
                &(
                    (
                        episode.config.players(),
                        episode.config.environment_seed(),
                        identity.sampling_seed().value(),
                        identity.episode_ordinal(),
                        identity.replicate_ordinal(),
                    ),
                    (
                        &episode.record.binding,
                        callbacks,
                        &episode.record.final_state,
                        &episode.record.scores,
                    ),
                    (
                        episode.record.draws,
                        episode.record.samples,
                        episode.record.rows,
                        episode.rank,
                        episode.winners,
                    ),
                ),
            )?);
        }
        hash(
            b"semantic-joint-audited-whole-four-v1\0",
            &(
                &self.parent,
                &self.families,
                &self.human_reservations,
                records,
            ),
        )
    }
}

/// Fixed fresh owner and explicit new allocation. This in-memory ledger burns
/// whole-four families and ALL configured actor streams before game creation.
/// It is not a durable reservation or proof of global history completeness.
pub struct SemanticPlayingCollectionSession<'a> {
    owner: &'a SemanticPlayingRlOwner,
    ledger: SemanticCollectionLedger<'a>,
}
impl<'a> SemanticPlayingCollectionSession<'a> {
    pub fn new(
        owner: &'a SemanticPlayingRlOwner,
        external: &'a ConditionalNativeExclusions,
        history: &'a DeclaredSamplingHistory,
        seed: SamplingSeed,
        first: u64,
        replicate: u64,
    ) -> Result<Self, String> {
        Ok(Self {
            owner,
            ledger: SemanticCollectionLedger::new(external, history, seed, first, replicate)?,
        })
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
    pub fn collect_whole_four(
        &mut self,
        seeds: [u32; 4],
        pool: &ConditionalFrozenPool,
        prepared_opponent: Option<&PreparedPublicPolicy>,
        limits: CollectionLimits,
    ) -> Result<AuditedSemanticPlayingCohort, String> {
        self.ledger.collect_whole_four_using(
            &self.owner.collection_view(),
            seeds,
            pool,
            prepared_opponent,
            limits,
            |_, cohort| Ok(cohort),
        )
    }
}

/// Shared allocation state, retained across recurrent semantic parameter updates.
/// Only the closed collector creates episodes; the continuation cannot inject steps.
#[derive(Clone, Copy)]
enum PoolRef<'a> {
    Legacy(&'a ConditionalFrozenPool, Option<&'a PreparedPublicPolicy>),
    Mixture(&'a SemanticFrozenMixture<'a>),
}
impl PoolRef<'_> {
    fn recheck(&self) -> Result<(), String> {
        match self {
            Self::Legacy(..) => Ok(()),
            Self::Mixture(p) => p.recheck(),
        }
    }
    fn binding(&self) -> Result<serde_json::Value, String> {
        match self {
            Self::Legacy(p, prepared) => p.binding(*prepared),
            Self::Mixture(p) => Ok(serde_json::json!({"fixedSemanticMixture":p.checksum()})),
        }
    }
    fn freeze(&self) -> Result<FrozenRosters<'_>, String> {
        match self {
            Self::Legacy(p, prepared) => Ok(FrozenRosters::Legacy(p.freeze(*prepared)?)),
            Self::Mixture(p) => Ok(FrozenRosters::Mixture(p.freeze()?)),
        }
    }
}
enum FrozenRosters<'a> {
    Legacy([ConditionalFrozenMixRoster<'a>; 4]),
    Mixture([SemanticMixtureRoster<'a>; 4]),
}
impl FrozenRosters<'_> {
    fn get(&self, index: usize) -> RosterRef<'_> {
        match self {
            Self::Legacy(r) => RosterRef::Legacy(&r[index]),
            Self::Mixture(r) => RosterRef::Mixture(&r[index]),
        }
    }
}
#[derive(Clone, Copy)]
enum RosterRef<'a> {
    Legacy(&'a ConditionalFrozenMixRoster<'a>),
    Mixture(&'a SemanticMixtureRoster<'a>),
}
impl<'a> RosterRef<'a> {
    fn learner_seat(self) -> usize {
        match self {
            Self::Legacy(r) => r.learner_seat(),
            Self::Mixture(r) => r.legacy.learner_seat(),
        }
    }
    fn checksum(self) -> &'a str {
        match self {
            Self::Legacy(r) => r.checksum(),
            Self::Mixture(r) => r.checksum(),
        }
    }
    fn contains_family(self, family: &str) -> bool {
        match self {
            Self::Legacy(r) => r.contains_family(family),
            Self::Mixture(r) => r.contains_family(family),
        }
    }
    fn opponent(
        self,
        actor: usize,
    ) -> Result<&'a crate::arena::FrozenReactiveOpponent<'a>, String> {
        match self {
            Self::Legacy(r) => r.opponent(actor),
            Self::Mixture(r) => r.legacy.opponent(actor),
        }
    }
    fn snapshot(self, actor: usize) -> Option<&'a FrozenSemanticSnapshot<'a>> {
        match self {
            Self::Legacy(_) => None,
            Self::Mixture(r) => r.snapshot(actor),
        }
    }
    fn common_setup(self) -> bool {
        matches!(self, Self::Mixture(_))
    }
    fn sampled_opponents(self, players: usize) -> [bool; 4] {
        std::array::from_fn(|actor| actor < players && self.snapshot(actor).is_some())
    }
}

pub(crate) struct SemanticCollectionLedger<'a> {
    external: &'a ConditionalNativeExclusions,
    history: &'a DeclaredSamplingHistory,
    seed: SamplingSeed,
    replicate: u64,
    next: u64,
    attempts: usize,
    spent: BTreeSet<String>,
    streams: BTreeSet<[u64; 5]>,
    pins: Vec<ContentPin>,
    stopped: bool,
    failed: Option<SemanticPlayingCollectedGame>,
}
impl<'a> SemanticCollectionLedger<'a> {
    pub(crate) fn freeze_snapshot(
        &self,
        owner: &SemanticOwnerView<'_>,
        current: &SemanticPlayingCurrentIdentity,
    ) -> Result<FrozenSemanticSnapshot<'a>, String> {
        FrozenSemanticSnapshot::capture(
            owner,
            current,
            SnapshotCaptureLedger {
                external: self.external,
                history: self.history,
                families: self.spent.clone(),
                streams: self.streams.clone(),
                sampling_seed: self.seed.value(),
                replicate: self.replicate,
                next: self.next,
                attempts: self.attempts,
                stopped: self.stopped,
            },
        )
    }
    pub(crate) fn with_checked<R>(
        &self,
        owner: &SemanticOwnerView<'_>,
        run: impl for<'scope> FnOnce(&SemanticCollectionScope<'scope>) -> Result<R, String>,
    ) -> Result<R, String> {
        owner.with_checked(self.external, &self.pins, run)
    }
    pub(crate) fn recheck(&self, owner: &SemanticOwnerView<'_>) -> Result<(), String> {
        owner.with_checked_episode(self.external, &self.pins, || Ok(()))
    }
    pub(crate) fn new(
        external: &'a ConditionalNativeExclusions,
        history: &'a DeclaredSamplingHistory,
        seed: SamplingSeed,
        first: u64,
        replicate: u64,
    ) -> Result<Self, String> {
        first.checked_add(4).ok_or("Semantic ordinal overflow")?;
        let pins = history.pins.clone();
        Ok(Self {
            external,
            history,
            seed,
            replicate,
            next: first,
            attempts: 0,
            spent: BTreeSet::new(),
            streams: BTreeSet::new(),
            pins,
            stopped: false,
            failed: None,
        })
    }
    pub(crate) fn attempts(&self) -> usize {
        self.attempts
    }
    pub(crate) fn next_episode_ordinal(&self) -> u64 {
        self.next
    }
    pub(crate) fn consumed_families(&self) -> &BTreeSet<String> {
        &self.spent
    }
    pub(crate) fn consumed_actor_streams(&self) -> usize {
        self.streams.len()
    }
    pub(crate) fn stopped(&self) -> bool {
        self.stopped
    }
    pub(crate) fn failed_game(&self) -> Option<&SemanticPlayingCollectedGame> {
        self.failed.as_ref()
    }
    pub(crate) fn stop(&mut self) {
        self.stopped = true;
    }
    pub(crate) fn collect_whole_four_using<R>(
        &mut self,
        owner: &SemanticOwnerView<'_>,
        seeds: [u32; 4],
        pool: &ConditionalFrozenPool,
        prepared_opponent: Option<&PreparedPublicPolicy>,
        limits: CollectionLimits,
        finish: impl for<'scope> FnOnce(
            &SemanticCollectionScope<'scope>,
            AuditedSemanticPlayingCohort,
        ) -> Result<R, String>,
    ) -> Result<R, String> {
        self.collect_using(
            owner,
            seeds,
            pool,
            prepared_opponent,
            limits,
            finish,
            create,
        )
    }
    pub(crate) fn collect_mixture_using<R>(
        &mut self,
        owner: &SemanticOwnerView<'_>,
        seeds: [u32; 4],
        mixture: &SemanticFrozenMixture<'_>,
        limits: CollectionLimits,
        finish: impl for<'scope> FnOnce(
            &SemanticCollectionScope<'scope>,
            AuditedSemanticPlayingCohort,
        ) -> Result<R, String>,
    ) -> Result<R, String> {
        self.collect_pool(
            owner,
            seeds,
            PoolRef::Mixture(mixture),
            limits,
            finish,
            create,
        )
    }
    #[allow(clippy::too_many_arguments)]
    fn collect_using<R>(
        &mut self,
        owner: &SemanticOwnerView<'_>,
        seeds: [u32; 4],
        pool: &ConditionalFrozenPool,
        prepared_opponent: Option<&PreparedPublicPolicy>,
        limits: CollectionLimits,
        finish: impl for<'scope> FnOnce(
            &SemanticCollectionScope<'scope>,
            AuditedSemanticPlayingCohort,
        ) -> Result<R, String>,
        create: impl Fn(&NativeStochasticConfig) -> Result<GameState, String>,
    ) -> Result<R, String> {
        self.collect_pool(
            owner,
            seeds,
            PoolRef::Legacy(pool, prepared_opponent),
            limits,
            finish,
            create,
        )
    }
    #[allow(clippy::too_many_arguments)]
    fn collect_pool<R>(
        &mut self,
        owner: &SemanticOwnerView<'_>,
        seeds: [u32; 4],
        pool: PoolRef<'_>,
        limits: CollectionLimits,
        finish: impl for<'scope> FnOnce(
            &SemanticCollectionScope<'scope>,
            AuditedSemanticPlayingCohort,
        ) -> Result<R, String>,
        create: impl Fn(&NativeStochasticConfig) -> Result<GameState, String>,
    ) -> Result<R, String> {
        if self.stopped || self.attempts >= MAX_ATTEMPTS {
            return Err("Semantic collection stopped or attempt bound".into());
        }
        let result = (|| {
            pool.recheck()?;
            let result = self.collect(owner, seeds, pool, limits, finish, create);
            match (result, pool.recheck()) {
                (Ok(value), Ok(())) => Ok(value),
                (Err(error), Ok(())) => Err(error),
                (Ok(_), Err(post)) => Err(format!("Frozen pool source postcheck: {post}")),
                (Err(error), Err(post)) => {
                    Err(format!("{error}; frozen pool source postcheck: {post}"))
                }
            }
        })();
        if result.is_err() {
            self.stopped = true;
        }
        result
    }
    #[cfg(test)]
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn collect_fixture_using<R>(
        &mut self,
        owner: &SemanticOwnerView<'_>,
        seeds: [u32; 4],
        pool: &ConditionalFrozenPool,
        limits: CollectionLimits,
        finish: impl for<'scope> FnOnce(
            &SemanticCollectionScope<'scope>,
            AuditedSemanticPlayingCohort,
        ) -> Result<R, String>,
        create: impl Fn(&NativeStochasticConfig) -> Result<GameState, String>,
    ) -> Result<R, String> {
        self.collect_using(owner, seeds, pool, None, limits, finish, create)
    }
    #[cfg(test)]
    pub(crate) fn collect_mixture_fixture_using<R>(
        &mut self,
        owner: &SemanticOwnerView<'_>,
        seeds: [u32; 4],
        mixture: &SemanticFrozenMixture<'_>,
        limits: CollectionLimits,
        finish: impl for<'scope> FnOnce(
            &SemanticCollectionScope<'scope>,
            AuditedSemanticPlayingCohort,
        ) -> Result<R, String>,
        create: impl Fn(&NativeStochasticConfig) -> Result<GameState, String>,
    ) -> Result<R, String> {
        self.collect_pool(
            owner,
            seeds,
            PoolRef::Mixture(mixture),
            limits,
            finish,
            create,
        )
    }
    #[allow(clippy::too_many_arguments)]
    fn collect<R>(
        &mut self,
        owner: &SemanticOwnerView<'_>,
        seeds: [u32; 4],
        pool: PoolRef<'_>,
        limits: CollectionLimits,
        finish: impl for<'scope> FnOnce(
            &SemanticCollectionScope<'scope>,
            AuditedSemanticPlayingCohort,
        ) -> Result<R, String>,
        create: impl Fn(&NativeStochasticConfig) -> Result<GameState, String>,
    ) -> Result<R, String> {
        let next = self
            .next
            .checked_add(4)
            .ok_or("Semantic ordinal overflow")?;
        let families = seeds.map(seed_family_id);
        let mut configs = Vec::new();
        let mut streams = Vec::new();
        for (i, &seed) in seeds.iter().enumerate() {
            if split_for_family(&families[i])? != DatasetSplit::Train
                || families[..i].contains(&families[i])
                || self.spent.contains(&families[i])
                || self.external.contains_family(&families[i])
            {
                return Err("Semantic family is held-out, excluded or already consumed".into());
            }
            let players = if i < 2 { 3 } else { 4 };
            let identity = SamplingStreamIdentity::new(
                self.seed,
                self.next + i as u64,
                self.replicate,
                players,
            )?;
            for actor in 0..players {
                let key = [
                    self.seed.value(),
                    identity.episode_ordinal(),
                    self.replicate,
                    players as u64,
                    actor as u64,
                ];
                if self.history.contains(identity, actor) || self.streams.contains(&key) {
                    return Err("Semantic sampling stream was permanently excluded".into());
                }
                streams.push(key);
            }
            configs.push(NativeStochasticConfig::new(
                players, seed, identity, limits,
            )?);
        }
        let rosters = pool.freeze()?;
        if (0..4).any(|i| families.iter().any(|f| rosters.get(i).contains_family(f))) {
            return Err("Semantic family overlaps frozen opponent lineage".into());
        }
        let (identity, reservations) = owner.metadata();
        let parent = identity.to_string();
        let human_reservations = reservations.clone();
        let binding = hash(
            b"tzolkin-semantic-playing-in-memory-collection-v1\0",
            &(
                &parent,
                self.external.descriptor(),
                &self.history.checksum,
                pool.binding()?,
                &human_reservations,
                crate::setup_policy::POLICY_VERSION,
                HeuristicWeights::default(),
            ),
        )?;
        let pins = self.pins.clone();
        owner.with_checked(self.external, &pins, |scope| {
            if scope.parent_checksum() != parent {
                return Err("Semantic checked owner differs from collection parent".into());
            }
            self.attempts += 1;
            self.next = next;
            self.spent.extend(families.iter().cloned());
            self.streams.extend(streams);
            let mut episodes = Vec::new();
            for i in 0..4 {
                let roster = rosters.get(i);
                let episode = owner.with_checked_episode(self.external, &pins, || {
                    let collected = run(&configs[i], roster, scope, &binding, None, || {
                        create(&configs[i])
                    })?;
                    if !collected.complete() {
                        let error = collected
                            .failure
                            .clone()
                            .unwrap_or_else(|| "Semantic game incomplete".into());
                        self.failed = Some(SemanticPlayingCollectedGame { record: collected });
                        return Err(error);
                    }
                    let audited = run(
                        &configs[i],
                        roster,
                        scope,
                        &binding,
                        Some(&collected),
                        || create(&configs[i]),
                    )?;
                    seal(
                        audited,
                        configs[i].clone(),
                        roster.learner_seat(),
                        roster.sampled_opponents(configs[i].players()),
                    )
                })?;
                episodes.push(episode);
            }
            let cohort = AuditedSemanticPlayingCohort {
                episodes: episodes
                    .try_into()
                    .map_err(|_| "Semantic whole-four shape")?,
                parent,
                families,
                human_reservations,
                mixture: match pool {
                    PoolRef::Mixture(p) => Some(p.checksum().into()),
                    PoolRef::Legacy(..) => None,
                },
            };
            finish(scope, cohort)
        })
    }
}

fn create(config: &NativeStochasticConfig) -> Result<GameState, String> {
    create_game_with_options(
        (0..config.players())
            .map(|p| format!("Semantic P{p}"))
            .collect(),
        config.environment_seed(),
        GameOptions::default(),
    )
}
fn validate(state: &GameState) -> Result<(), String> {
    if !tzolkin_core::validation::validate_game_state(
        &serde_json::to_value(state).map_err(|e| e.to_string())?,
    ) {
        return Err("Semantic core state invariant failed".into());
    }
    Ok(())
}
fn legal_choice(
    o: &Observation,
    decision: Decision,
    version: &str,
) -> Result<(Decision, LegalAction), String> {
    let matches = o
        .legal_actions
        .iter()
        .filter(|a| a.r#move == decision.r#move)
        .collect::<Vec<_>>();
    if decision.actor != o.actor
        || decision.observation_key != o.observation_key
        || decision.policy_version != version
        || !decision.score.is_finite()
        || matches.len() != 1
    {
        return Err("Semantic decision/full legal binding differs".into());
    }
    Ok((decision, matches[0].clone()))
}
fn run(
    config: &NativeStochasticConfig,
    roster: RosterRef<'_>,
    scope: &SemanticCollectionScope<'_>,
    binding: &str,
    expected: Option<&Record>,
    start: impl FnOnce() -> Result<GameState, String>,
) -> Result<Record, String> {
    let binding = hash(
        b"tzolkin-semantic-playing-episode-binding-v1\0",
        &(
            binding,
            scope.parent_checksum(),
            roster.checksum(),
            config.players(),
            config.environment_seed(),
            config.sampling_identity().sampling_seed().value(),
            config.sampling_identity().episode_ordinal(),
            config.sampling_identity().replicate_ordinal(),
            SAMPLING_VERSION,
            crate::public_stochastic::RNG_VERSION,
            crate::public_stochastic::DENOMINATOR_BITS,
            crate::public_stochastic::UNIFORM_MIXTURE.to_bits(),
            crate::public_stochastic_native::numerical_target(),
            (
                config.limits().max_callbacks(),
                config.limits().max_candidate_rows(),
                config.limits().max_source_bytes(),
                config.limits().max_trace_bytes(),
            ),
        ),
    )?;
    let mut record = Record {
        binding,
        callbacks: vec![],
        final_state: None,
        scores: vec![],
        failure: None,
        draws: [0; 4],
        samples: 0,
        opponent_samples: 0,
        rows: 0,
    };
    if expected.is_some_and(|e| e.binding != record.binding) {
        return Err("Semantic audit source/configuration binding differs".into());
    }
    let mut states = [0; 4];
    for (actor, state) in states.iter_mut().enumerate().take(config.players()) {
        *state = config.sampling_identity().initial_actor_state(actor)?;
    }
    let mut guards = (0..config.players())
        .map(|actor| {
            if actor == roster.learner_seat() || roster.snapshot(actor).is_some() {
                Ok(None)
            } else {
                roster.opponent(actor)?.trade_guard_session()
            }
        })
        .collect::<Result<Vec<_>, String>>()?;
    let mut source_bytes = 16 * 1024usize;
    let mut trace_bytes = 0usize;
    let result = (|| {
        let mut state = start()?;
        validate(&state)?;
        if state.additional_buildings || state.expansion.is_some() {
            return Err("Semantic collection requires actual default Basic options".into());
        }
        while state.phase != Phase::Finished {
            let index = record.callbacks.len();
            if index >= config.limits().max_callbacks() {
                return Err("Semantic callback limit".into());
            }
            let o = observe(&state, state.current_player)?;
            let before = record
                .callbacks
                .last()
                .and_then(|c| c.after.clone())
                .map_or_else(|| crate::replay::state_key(&state), Ok)?;
            if let Some(e) = expected {
                let c = e
                    .callbacks
                    .get(index)
                    .ok_or("Semantic audit extra callback")?;
                if c.observation != o || c.before != before {
                    return Err("Semantic audit observation/state differs before scoring".into());
                }
            }
            let bytes = serde_json::to_vec(&o).map_err(|e| e.to_string())?.len();
            source_bytes = source_bytes
                .checked_add(
                    bytes
                        .checked_mul(3)
                        .and_then(|n| n.checked_add(4096))
                        .ok_or("Semantic source accounting overflow")?,
                )
                .ok_or("Semantic source accounting overflow")?;
            if source_bytes > config.limits().max_source_bytes() {
                return Err("Semantic source payload cap before choice".into());
            }
            if !(1..=4096).contains(&o.legal_actions.len()) {
                return Err("Semantic full legal mask bound".into());
            }
            for guard in guards.iter_mut().flatten() {
                guard.on_callback(index, &o)?;
            }
            record.callbacks.push(Callback {
                index,
                observation: o,
                before,
                choice: None,
                apply_attempted: false,
                after: None,
            });
            let o = &record.callbacks[index].observation;
            let learner = o.actor == roster.learner_seat();
            let snapshot = roster.snapshot(o.actor);
            let choice = if o.phase == Phase::Setup && (learner || roster.common_setup()) {
                let decision =
                    crate::setup_policy::choose_move_with_weights(o, &HeuristicWeights::default())?;
                let (decision, chosen) =
                    legal_choice(o, decision, crate::setup_policy::POLICY_VERSION)?;
                if learner {
                    Choice::LearnerSetup { decision, chosen }
                } else {
                    Choice::Opponent { decision, chosen }
                }
            } else if !learner && snapshot.is_none() {
                let opponent = roster.opponent(o.actor)?;
                let decision = opponent.choose_indexed(index, o, guards[o.actor].as_mut())?;
                let (decision, chosen) = legal_choice(o, decision, opponent.policy_version())?;
                Choice::Opponent { decision, chosen }
            } else if o.phase == Phase::Playing {
                let sampled_snapshot = if learner {
                    None
                } else {
                    Some(snapshot.ok_or("Missing frozen sampled actor")?)
                };
                let rows_before = record.rows;
                let rows = record
                    .rows
                    .checked_add(o.legal_actions.len())
                    .ok_or("Semantic row overflow")?;
                trace_bytes = trace_bytes
                    .checked_add(4096)
                    .ok_or("Semantic trace overflow")?;
                if rows > config.limits().max_candidate_rows()
                    || trace_bytes > config.limits().max_trace_bytes()
                {
                    return Err("Semantic Playing row/trace cap before forward or draw".into());
                }
                record.rows = rows;
                let prediction = if let Some(snapshot) = sampled_snapshot {
                    snapshot.distribution(o, scope.kernel())?
                } else {
                    scope.distribution(o)?
                };
                if prediction.logits.len() != o.legal_actions.len()
                    || prediction.probabilities.len() != prediction.logits.len()
                {
                    return Err("Semantic numeric mask differs".into());
                }
                let grid = TickDistribution::new(&prediction.logits)?;
                let legal_digest = legal_digest(&o.legal_actions)?;
                let logits_digest = full_logits_digest(&prediction.logits);
                let distribution_digest = grid.digest();
                let draw = draw_tick53(states[o.actor], record.draws[o.actor], &grid)?;
                let chosen = o.legal_actions[draw.index].clone();
                let logit = prediction.logits[draw.index];
                let nominal_bits = grid.nominal_probability(draw.index)?.to_bits();
                let sample = SemanticSample {
                    decision: Decision {
                        actor: o.actor,
                        observation_key: o.observation_key.clone(),
                        policy_version: if learner {
                            SAMPLING_VERSION
                        } else {
                            OPPONENT_SAMPLING_VERSION
                        }
                        .into(),
                        r#move: chosen.r#move.clone(),
                        score: f64::from(logit),
                    },
                    chosen,
                    index: draw.index,
                    sample_index: if learner {
                        record.samples
                    } else {
                        record.opponent_samples
                    },
                    global_sample_index: record
                        .samples
                        .checked_add(record.opponent_samples)
                        .ok_or("Semantic sampled index overflow")?,
                    before: draw.state_before,
                    after: draw.state_after,
                    draws_before: draw.draws_before,
                    raw: draw.raw,
                    u53: draw.u53,
                    mass: draw.mass,
                    nominal_bits,
                    logp_bits: draw.logp.to_bits(),
                    logit_bits: logit.to_bits(),
                    rows_before,
                    rows_after: rows,
                    legal_digest,
                    logits_digest,
                    distribution_digest,
                };
                states[o.actor] = draw.state_after;
                record.draws[o.actor] = draw.draws_after;
                if learner {
                    record.samples += 1;
                } else {
                    record.opponent_samples += 1;
                }
                record.rows = rows;
                if let Some(snapshot) = sampled_snapshot {
                    Choice::OpponentSample {
                        sample: Box::new(sample),
                        snapshot: snapshot.binding().into(),
                    }
                } else {
                    Choice::LearnerPlaying(Box::new(sample))
                }
            } else {
                return Err("Semantic callback phase is neither Setup nor Playing".into());
            };
            let operation = choice.decision().r#move.clone();
            record.callbacks[index].choice = Some(choice);
            record.callbacks[index].apply_attempted = true;
            state = apply_move(&state, operation)?;
            validate(&state)?;
            record.callbacks[index].after = Some(crate::replay::state_key(&state)?);
            if expected.is_some_and(|e| e.callbacks.get(index) != record.callbacks.get(index)) {
                return Err("Semantic audit role/choice/trace/apply chain differs".into());
            }
        }
        validate(&state)?;
        crate::state_mc_dataset::winner_shares(&state.final_scores, config.players())?;
        record.final_state = Some(crate::replay::state_key(&state)?);
        record.scores = state.final_scores;
        Ok::<_, String>(())
    })();
    if let Err(error) = result {
        record.failure = Some(error.chars().take(1024).collect());
    }
    if expected.is_some_and(|e| *e != record) {
        return Err("Semantic fresh audit complete record differs".into());
    }
    Ok(record)
}
fn seal(
    record: Record,
    config: NativeStochasticConfig,
    learner: usize,
    sampled_opponents: [bool; 4],
) -> Result<AuditedSemanticPlayingEpisode, String> {
    if !record.complete() {
        return Err("Semantic episode requires completed fresh audit".into());
    }
    let playing = record
        .callbacks
        .iter()
        .enumerate()
        .filter_map(|(i, c)| matches!(c.choice, Some(Choice::LearnerPlaying(_))).then_some(i))
        .collect::<Vec<_>>();
    let mut draws = [0_u64; 4];
    let mut own = 0;
    let mut opponents = 0;
    let mut rows = 0;
    for callback in &record.callbacks {
        let actor = callback.observation.actor;
        if actor >= config.players() {
            return Err("Semantic audited actor out of range".into());
        }
        let sample = match callback
            .choice
            .as_ref()
            .ok_or("Missing audited semantic choice")?
        {
            Choice::LearnerPlaying(sample)
                if actor == learner && callback.observation.phase == Phase::Playing =>
            {
                if sample.sample_index != own {
                    return Err("Learner sample ordinal differs".into());
                }
                own += 1;
                Some(sample)
            }
            Choice::OpponentSample { sample, .. }
                if actor != learner
                    && sampled_opponents[actor]
                    && callback.observation.phase == Phase::Playing =>
            {
                if sample.sample_index != opponents {
                    return Err("Opponent sample ordinal differs".into());
                }
                opponents += 1;
                Some(sample)
            }
            Choice::LearnerSetup { .. }
                if actor == learner && callback.observation.phase == Phase::Setup =>
            {
                None
            }
            Choice::Opponent { .. }
                if actor != learner
                    && (!sampled_opponents[actor]
                        || callback.observation.phase == Phase::Setup) =>
            {
                None
            }
            _ => return Err("Semantic audited phase/slot role differs".into()),
        };
        if let Some(sample) = sample {
            if sample.global_sample_index != own + opponents - 1
                || sample.draws_before != draws[actor]
                || sample.rows_before != rows
            {
                return Err("Semantic all-actor sampled coverage differs".into());
            }
            draws[actor] += 1;
            rows += callback.observation.legal_actions.len();
            if sample.rows_after != rows {
                return Err("Semantic sampled row coverage differs".into());
            }
        }
    }
    if playing.is_empty()
        || playing.len() != record.samples
        || own != record.samples
        || opponents != record.opponent_samples
        || rows != record.rows
        || draws != record.draws
        || record.draws[learner] != playing.len() as u64
    {
        return Err("Semantic Playing-only draw coverage differs".into());
    }
    let (_, ranks, winners) =
        crate::state_mc_dataset::winner_shares(&record.scores, config.players())?;
    Ok(AuditedSemanticPlayingEpisode {
        record,
        config,
        playing,
        rank: ranks[learner],
        winners,
    })
}

#[cfg(test)]
pub(crate) fn mixed_trace_fixture(
    scope: &SemanticCollectionScope<'_>,
    mixture: &SemanticFrozenMixture<'_>,
    config: &NativeStochasticConfig,
) {
    let rosters = mixture.freeze().unwrap();
    let roster = RosterRef::Mixture(&rosters[0]);
    let setup_config = NativeStochasticConfig::new(
        config.players(),
        config.environment_seed(),
        *config.sampling_identity(),
        CollectionLimits::new(2, 4096, 1024 * 1024, 65536).unwrap(),
    )
    .unwrap();
    let prefix = run(
        &setup_config,
        roster,
        scope,
        "synthetic-all-seat-corn-first",
        None,
        || create(&setup_config),
    )
    .unwrap();
    assert!(!prefix.complete());
    assert_eq!(
        (prefix.samples, prefix.opponent_samples, prefix.draws),
        (0, 0, [0; 4])
    );
    assert_eq!(prefix.callbacks.len(), 2);
    assert!(prefix.callbacks.iter().any(|c| c.observation.actor == 1));
    for callback in &prefix.callbacks {
        assert_eq!(callback.observation.phase, Phase::Setup);
        assert_eq!(
            *callback.choice.as_ref().unwrap().decision(),
            crate::setup_policy::choose_move_with_weights(
                &callback.observation,
                &HeuristicWeights::default()
            )
            .unwrap()
        );
    }
    let start = || {
        let mut state = crate::conditional_human_rl_update::tests::terminal_start(
            config.players(),
            config.environment_seed(),
        );
        state.turn.mode = tzolkin_core::TurnMode::None;
        state.turn.count = 0;
        Ok(state)
    };
    let mut complete = run(config, roster, scope, "synthetic-mixed-trace", None, start).unwrap();
    assert!(complete.complete(), "{:?}", complete.failure);
    assert!(complete.samples > 0 && complete.opponent_samples > 0);
    assert_eq!(
        complete.draws.iter().sum::<u64>(),
        (complete.samples + complete.opponent_samples) as u64
    );
    let audited = run(
        config,
        roster,
        scope,
        "synthetic-mixed-trace",
        Some(&complete),
        start,
    )
    .unwrap();
    let episode = seal(
        audited,
        config.clone(),
        roster.learner_seat(),
        roster.sampled_opponents(config.players()),
    )
    .unwrap();
    assert_eq!(episode.playing_steps(), complete.samples);
    assert_eq!(episode.actor_draws(), &complete.draws);
    for index in 0..episode.playing_steps() {
        assert_eq!(episode.step(index).unwrap().actor(), roster.learner_seat());
    }
    let sample = complete
        .callbacks
        .iter_mut()
        .find_map(|c| match c.choice.as_mut() {
            Some(Choice::OpponentSample { sample, .. }) => Some(sample),
            _ => None,
        })
        .unwrap();
    sample.raw ^= 1;
    assert!(
        run(
            config,
            roster,
            scope,
            "synthetic-mixed-trace",
            Some(&complete),
            start
        )
        .is_err()
    );
    assert!(seal(complete, config.clone(), roster.learner_seat(), [false; 4]).is_err());

    // The first actor is a snapshot opponent. Its quota must reject before
    // that actor's first forward/draw, not merely at learner-gradient admission.
    let tiny = NativeStochasticConfig::new(
        config.players(),
        config.environment_seed(),
        *config.sampling_identity(),
        CollectionLimits::new(64, 1, 8 * 1024 * 1024, 1024 * 1024).unwrap(),
    )
    .unwrap();
    let refused = run(
        &tiny,
        roster,
        scope,
        "synthetic-opponent-row-cap",
        None,
        || {
            let mut state = start()?;
            state.current_player = 1;
            state.turn_index = 1;
            assert!(observe(&state, 1)?.legal_actions.len() > 1);
            Ok(state)
        },
    )
    .unwrap();
    assert!(!refused.complete());
    assert!(
        refused
            .failure
            .as_deref()
            .unwrap()
            .contains("cap before forward or draw")
    );
    assert_eq!(
        (refused.samples, refused.opponent_samples, refused.draws),
        (0, 0, [0; 4])
    );
}

#[cfg(test)]
pub(crate) fn phase_collection_fixture(
    prepared: &crate::human_policy_training::PreparedConditionalHumanPolicy,
) {
    use serde_json::json;
    crate::conditional_semantic_playing_rl::with_collection_numeric_fixture(prepared, |scope| {
        let opponent = json!({"kind":"heuristic"});
        let pool = ConditionalFrozenPool::from_json(
            &serde_json::to_vec(&json!({
                "schema":"tzolkin-conditional-human-mixed-frozen-pool-v1",
                "weights":HeuristicWeights::default(),
                "rosters":[[null,opponent,opponent],[null,opponent,opponent],
                    [null,opponent,opponent,opponent],[null,opponent,opponent,opponent]]
            }))
            .unwrap(),
        )
        .unwrap();
        let rosters = pool.freeze(None).unwrap();
        let config = NativeStochasticConfig::new(
            3,
            0,
            SamplingStreamIdentity::new(SamplingSeed::new(17), 0, 0, 3).unwrap(),
            CollectionLimits::new(1, 4096, 1024 * 1024, 65536).unwrap(),
        )
        .unwrap();
        let prefix = run(
            &config,
            RosterRef::Legacy(&rosters[0]),
            scope,
            "synthetic-unit",
            None,
            || create(&config),
        )
        .unwrap();
        assert!(!prefix.complete());
        assert_eq!((prefix.samples, prefix.rows, prefix.draws), (0, 0, [0; 4]));
        assert_eq!(prefix.callbacks.len(), 1);
        let callback = &prefix.callbacks[0];
        assert!(callback.apply_attempted && callback.after.is_some());
        let Some(Choice::LearnerSetup { decision, chosen }) = &callback.choice else {
            panic!("Setup must have its own applied role")
        };
        assert_eq!(
            *decision,
            crate::setup_policy::choose_move_with_weights(
                &callback.observation,
                &HeuristicWeights::default()
            )
            .unwrap()
        );
        assert_eq!(chosen.r#move, decision.r#move);
        let replayed = run(
            &config,
            RosterRef::Legacy(&rosters[0]),
            scope,
            "synthetic-unit",
            Some(&prefix),
            || create(&config),
        )
        .unwrap();
        assert!(seal(replayed, config.clone(), 0, [false; 4]).is_err());

        // Existing synthetic last-round Core oracle; no full game or score
        // prediction is being certified as a native source.
        let last = crate::conditional_human_rl_update::tests::terminal_start(3, 0);
        let config = NativeStochasticConfig::new(
            3,
            0,
            *config.sampling_identity(),
            CollectionLimits::new(64, 262144, 8 * 1024 * 1024, 1024 * 1024).unwrap(),
        )
        .unwrap();
        let complete = run(
            &config,
            RosterRef::Legacy(&rosters[0]),
            scope,
            "synthetic-unit",
            None,
            || Ok(last.clone()),
        )
        .unwrap();
        assert!(complete.complete(), "{:?}", complete.failure);
        let audited = run(
            &config,
            RosterRef::Legacy(&rosters[0]),
            scope,
            "synthetic-unit",
            Some(&complete),
            || Ok(last.clone()),
        )
        .unwrap();
        let episode = seal(audited, config.clone(), 0, [false; 4]).unwrap();
        assert_eq!(episode.learner_setup_choices(), 0);
        assert!(episode.playing_steps() > 0);
        assert_eq!(episode.actor_draws()[0], episode.playing_steps() as u64);
        for i in 0..episode.playing_steps() {
            let step = episode.step(i).unwrap();
            assert_eq!(step.observation().phase, Phase::Playing);
            assert_eq!(step.actor_draw_ordinal(), i as u64);
            assert_eq!(step.return_target(), 1.0);
            assert_eq!(
                step.reward(),
                if i + 1 == episode.playing_steps() {
                    1.0
                } else {
                    0.0
                }
            );
            assert_eq!(
                step.intervening_callbacks(),
                step.next_playing_global_index()
                    .unwrap_or(episode.observed_callbacks())
                    - step.global_callback_index()
                    - 1
            );
        }
        let mut bad = complete;
        let first = bad
            .callbacks
            .iter_mut()
            .find(|c| matches!(c.choice, Some(Choice::LearnerPlaying(_))))
            .unwrap();
        let Some(Choice::LearnerPlaying(sample)) = first.choice.take() else {
            unreachable!()
        };
        first.choice = Some(Choice::LearnerSetup {
            decision: sample.decision,
            chosen: sample.chosen,
        });
        assert!(
            run(
                &config,
                RosterRef::Legacy(&rosters[0]),
                scope,
                "synthetic-unit",
                Some(&bad),
                || Ok(last)
            )
            .is_err()
        );
        Ok(())
    })
    .unwrap();
}
