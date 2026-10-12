//! In-memory, on-policy Setup learning with immutable guarded Playing parameters.
//! Attempts are reserved before collection. This owner is not a durable checkpoint.
use std::collections::BTreeSet;

use serde::{Deserialize, Serialize};
use tzolkin_core::observation::Observation;
use tzolkin_inference::deployment::LoadedDeployment;

use crate::dataset::{DatasetSplit, seed_family_id, split_for_family};
use crate::own_setup_model::{
    BorrowedInitializedOwnSetup, InitializedOwnSetupArtifact, OwnSetupModel, SetupCandidates,
};
use crate::own_setup_native::{
    OwnSetupCollector, OwnSetupNativeConfig, TrainedOwnSetupCollector, ValidatedOwnSetupEpisode,
    ValidatedTrainedOwnSetupEpisode, fixed_playing_identity, playing_binding_checksum,
};
use crate::own_setup_pullback::OwnSetupScalarPullback;
use crate::own_setup_record::{self, FixedPlayingIdentity, OwnSetupCounts, SetupSample};
use crate::own_setup_trained_model::{
    self as trained, BorrowedTrainedOwnSetup, SetupUpdateSource, TrainedOwnSetupArtifact,
    TrainedOwnSetupIdentity,
};
use crate::public_model::PARAMETER_COUNT;
use crate::public_policy_likelihood::BehaviorLikelihood;
use crate::public_policy_update::{MAX_ALL_LOG_DELTA, checkpoint_digest, hash};
use crate::public_stochastic::{
    SamplingSeed, TICKS, full_logits_digest, legal_digest, next_splitmix,
};
use crate::public_stochastic_record::{MAX_RECORD_BYTES, hex64};

pub const TRAINING_VERSION: &str = "own-setup-onpolicy-winner-share-adam-f64-v1";
pub const EXCLUSION_DOMAIN: &[u8] = b"tzolkin-own-setup-declared-train-exclusions-v1\0";
const MAX_ATTEMPTED_EPISODES: u64 = 100_000;
const MAX_EXCLUSIONS: usize = 200_000;
const MAX_BATCH: usize = 4;

/// The caller must provide the complete teacher/opponent/held-out family union.
/// Pins verify that declaration's consistency; they do not prove its completeness.
pub struct DeclaredOwnSetupExclusions {
    families: BTreeSet<String>,
    digest: String,
}
impl DeclaredOwnSetupExclusions {
    pub fn from_families(
        families: Vec<String>,
        expected_count: usize,
        expected_digest: &str,
    ) -> Result<Self, String> {
        if families.len() > MAX_EXCLUSIONS
            || families.len() != expected_count
            || !checkpoint_digest(expected_digest)
            || families.iter().any(|f| !checkpoint_digest(f))
            || families.windows(2).any(|pair| pair[0] >= pair[1])
            || Self::digest_for_families(&families)? != expected_digest
        {
            return Err("Invalid sorted complete exclusion declaration/pins".into());
        }
        Ok(Self {
            families: families.into_iter().collect(),
            digest: expected_digest.into(),
        })
    }
    pub fn digest_for_families(families: &[String]) -> Result<String, String> {
        if families.len() > MAX_EXCLUSIONS {
            return Err("Exclusion declaration exceeds cap".into());
        }
        hash(EXCLUSION_DOMAIN, &families)
    }
    pub fn digest(&self) -> &str {
        &self.digest
    }
    pub(crate) fn families(&self) -> &BTreeSet<String> {
        &self.families
    }
    pub fn count(&self) -> usize {
        self.families.len()
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct OwnSetupAdamConfig {
    pub learning_rate: f64,
    pub beta1: f64,
    pub beta2: f64,
    pub epsilon: f64,
    pub max_gradient_norm: f64,
    pub max_parameter_delta: f64,
}
impl OwnSetupAdamConfig {
    pub fn validate(&self) -> Result<(), String> {
        if !self.learning_rate.is_finite()
            || !(0.0..=0.01).contains(&self.learning_rate)
            || self.learning_rate == 0.0
            || !self.beta1.is_finite()
            || !(0.0..1.0).contains(&self.beta1)
            || !self.beta2.is_finite()
            || !(0.0..1.0).contains(&self.beta2)
            || !self.epsilon.is_finite()
            || !(1.0e-12..=1.0).contains(&self.epsilon)
            || !self.max_gradient_norm.is_finite()
            || !(1.0e-12..=1000.0).contains(&self.max_gradient_norm)
            || !self.max_parameter_delta.is_finite()
            || !(1.0e-12..=0.1).contains(&self.max_parameter_delta)
        {
            return Err("Invalid bounded Own Setup Adam configuration".into());
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct OwnSetupRunConfig {
    pub sampling_seed: u64,
    pub replicate_ordinal: u64,
    pub first_episode_ordinal: u64,
    pub max_attempted_episodes: u64,
}

enum SetupState {
    Initialized(Box<InitializedOwnSetupArtifact>),
    Trained(Box<TrainedOwnSetupArtifact>),
}
impl SetupState {
    fn model(&self) -> &OwnSetupModel {
        match self {
            Self::Initialized(a) => &a.model,
            Self::Trained(a) => a.model(),
        }
    }
    fn checksum(&self) -> &str {
        match self {
            Self::Initialized(a) => &a.identity.checksum,
            Self::Trained(a) => &a.identity().checksum,
        }
    }
    fn identity_bytes(&self) -> Result<Vec<u8>, String> {
        match self {
            Self::Initialized(a) => json(&a.identity),
            Self::Trained(a) => json(a.identity()),
        }
    }
    fn initial(&self) -> &crate::own_setup_model::InitializedOwnSetupIdentity {
        match self {
            Self::Initialized(a) => &a.identity,
            Self::Trained(a) => &a.identity().initial,
        }
    }
    fn count(&self) -> u64 {
        match self {
            Self::Initialized(_) => 0,
            Self::Trained(a) => a.identity().update_count,
        }
    }
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(
    deny_unknown_fields,
    tag = "kind",
    rename_all = "camelCase",
    rename_all_fields = "camelCase"
)]
pub enum OwnSetupAttemptOutcome {
    Updated {
        update_count: u64,
        changed_parameters: usize,
        max_parameter_delta: f64,
    },
    NoChange,
    Rejected {
        reason: String,
    },
}
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct OwnSetupEpisodeReceipt {
    pub record_checksum: String,
    pub episode_id: Option<String>,
    pub counts: OwnSetupCounts,
}
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct OwnSetupAttemptReport {
    pub first_episode_ordinal: u64,
    pub reserved_episodes: usize,
    pub next_episode_ordinal: Option<u64>,
    pub attempted_episodes: u64,
    pub families: Vec<String>,
    pub setup_before: String,
    pub setup_after: String,
    pub receipts: Vec<OwnSetupEpisodeReceipt>,
    pub outcome: OwnSetupAttemptOutcome,
}

#[derive(Clone)]
struct AdamState {
    step: u64,
    first: Vec<f64>,
    second: Vec<f64>,
}
struct AdamProposal {
    model: OwnSetupModel,
    optimizer: AdamState,
    changed: usize,
    max_delta: f64,
}

/// One live in-memory owner. Only its own reserved, freshly audited games update it.
/// No external episode, deserialized owner or restored optimizer is admitted here.
pub struct OwnSetupTrainingOwner {
    setup: SetupState,
    playing: Option<LoadedDeployment>,
    playing_binding: FixedPlayingIdentity,
    playing_checksum: String,
    adam_config: OwnSetupAdamConfig,
    run_config: OwnSetupRunConfig,
    exclusions: DeclaredOwnSetupExclusions,
    config_checksum: String,
    optimizer: AdamState,
    next_episode: Option<u64>,
    attempted: u64,
    spent_families: BTreeSet<String>,
    credited_ids: BTreeSet<String>,
    credited_records: BTreeSet<String>,
    ledger: Vec<OwnSetupAttemptReport>,
}
impl OwnSetupTrainingOwner {
    pub fn new(
        initial: InitializedOwnSetupArtifact,
        playing: LoadedDeployment,
        adam: OwnSetupAdamConfig,
        run: OwnSetupRunConfig,
        exclusions: DeclaredOwnSetupExclusions,
    ) -> Result<Self, String> {
        initial.validate()?;
        adam.validate()?;
        if !(1..=MAX_ATTEMPTED_EPISODES).contains(&run.max_attempted_episodes) {
            return Err("Own Setup attempted episode budget outside cap".into());
        }
        let binding = fixed_playing_identity(&playing)?;
        let playing_checksum = playing_binding_checksum(&binding)?;
        let config_checksum = hash(
            b"tzolkin-own-setup-training-config-v1\0",
            &(
                TRAINING_VERSION,
                &adam,
                &run,
                exclusions.count(),
                exclusions.digest(),
                &initial.identity,
                &playing_checksum,
            ),
        )?;
        Ok(Self {
            setup: SetupState::Initialized(Box::new(initial)),
            playing: Some(playing),
            playing_binding: binding,
            playing_checksum,
            adam_config: adam,
            next_episode: Some(run.first_episode_ordinal),
            run_config: run,
            exclusions,
            config_checksum,
            optimizer: AdamState {
                step: 0,
                first: vec![0.0; PARAMETER_COUNT],
                second: vec![0.0; PARAMETER_COUNT],
            },
            attempted: 0,
            spent_families: BTreeSet::new(),
            credited_ids: BTreeSet::new(),
            credited_records: BTreeSet::new(),
            ledger: Vec::new(),
        })
    }
    pub fn update_count(&self) -> u64 {
        self.setup.count()
    }
    pub fn attempted_episodes(&self) -> u64 {
        self.attempted
    }
    pub fn next_episode_ordinal(&self) -> Option<u64> {
        self.next_episode
    }
    pub fn setup_checksum(&self) -> &str {
        self.setup.checksum()
    }
    pub fn trained_artifact(&self) -> Option<&TrainedOwnSetupArtifact> {
        match &self.setup {
            SetupState::Trained(a) => Some(a),
            _ => None,
        }
    }
    pub fn playing(&self) -> Result<&LoadedDeployment, String> {
        self.playing
            .as_ref()
            .ok_or_else(|| "Own Setup fixed Playing storage unavailable".into())
    }
    pub fn ledger(&self) -> &[OwnSetupAttemptReport] {
        &self.ledger
    }

    pub(crate) fn config_checksum(&self) -> &str {
        &self.config_checksum
    }

    pub(crate) fn exclusion_families(&self) -> Vec<String> {
        self.exclusions.families.iter().cloned().collect()
    }

    pub(crate) fn exclusion_digest(&self) -> &str {
        self.exclusions.digest()
    }

    /// Only the durable source reader can extend a completed successful owner.
    /// This changes the prospective binding, never the existing trained identity,
    /// parameters, Adam moments, cumulative cursor, ledger or learning credit.
    pub(crate) fn continue_budget(
        &mut self,
        additional_episodes: u64,
        exclusions: DeclaredOwnSetupExclusions,
    ) -> Result<OwnSetupRunConfig, String> {
        let budget = self
            .attempted
            .checked_add(additional_episodes)
            .ok_or("Own Setup continuation budget overflow")?;
        if !(1..=256).contains(&additional_episodes)
            || budget > MAX_ATTEMPTED_EPISODES
            || self.attempted != self.run_config.max_attempted_episodes
            || self.setup.count() == 0
            || self.optimizer.step != self.setup.count()
            || self.ledger.len() as u64 != self.setup.count()
            || self
                .ledger
                .iter()
                .any(|r| !matches!(r.outcome, OwnSetupAttemptOutcome::Updated { .. }))
            || self
                .next_episode
                .and_then(|n| n.checked_add(additional_episodes - 1))
                .is_none()
            || !self.exclusions.families.is_subset(&exclusions.families)
            || !self.spent_families.is_subset(&exclusions.families)
        {
            return Err("Own Setup continuation requires a complete successful owner, cursor and full monotone baseline".into());
        }
        let mut run = self.run_config.clone();
        run.max_attempted_episodes = budget;
        let checksum = hash(
            b"tzolkin-own-setup-continuation-training-config-v1\0",
            &(
                TRAINING_VERSION,
                &self.adam_config,
                &run,
                exclusions.count(),
                exclusions.digest(),
                self.setup.initial(),
                &self.playing_checksum,
                &self.config_checksum,
                self.setup.checksum(),
                self.attempted,
                self.setup.count(),
                self.ledger.len(),
            ),
        )?;
        self.run_config = run.clone();
        self.exclusions = exclusions;
        self.config_checksum = checksum;
        Ok(run)
    }

    /// Reserves all requested slots first. Post-reservation refusals, failed games
    /// and no-change updates consume those slots and families without rewind.
    /// A batch uses one frozen Setup identity; it stops at the first failed game.
    pub fn attempt(
        &mut self,
        configs: &[OwnSetupNativeConfig],
    ) -> Result<OwnSetupAttemptReport, String> {
        let reservation = self.reserve(configs)?;
        self.resolve(reservation, configs, BatchRecords::Collect(None))
    }
    pub(crate) fn preview(
        &self,
        configs: &[OwnSetupNativeConfig],
    ) -> Result<OwnSetupAttemptReport, String> {
        if !(1..=MAX_BATCH).contains(&configs.len())
            || self
                .attempted
                .checked_add(configs.len() as u64)
                .is_none_or(|n| n > self.run_config.max_attempted_episodes)
        {
            return Err("Own Setup batch/attempt budget exhausted".into());
        }
        let first = self
            .next_episode
            .ok_or("Own Setup u64 episode cursor exhausted")?;
        let last = first
            .checked_add(configs.len() as u64 - 1)
            .ok_or("Own Setup batch exceeds remaining u64 cursor")?;
        let families: Vec<_> = configs
            .iter()
            .map(|c| seed_family_id(c.native().environment_seed()))
            .collect();
        let report = OwnSetupAttemptReport {
            first_episode_ordinal: first,
            reserved_episodes: configs.len(),
            next_episode_ordinal: last.checked_add(1),
            attempted_episodes: self.attempted + configs.len() as u64,
            families,
            setup_before: self.setup.checksum().into(),
            setup_after: self.setup.checksum().into(),
            receipts: Vec::new(),
            outcome: OwnSetupAttemptOutcome::Rejected {
                reason: "Reserved attempt did not finish".into(),
            },
        };
        Ok(report)
    }
    pub(crate) fn reserve(
        &mut self,
        configs: &[OwnSetupNativeConfig],
    ) -> Result<ReservedOwnSetupAttempt, String> {
        let report = self.preview(configs)?;
        let reused = report
            .families
            .iter()
            .any(|f| self.spent_families.contains(f));
        self.next_episode = report.next_episode_ordinal;
        self.attempted = report.attempted_episodes;
        self.spent_families.extend(report.families.iter().cloned());
        self.ledger.push(report.clone());
        Ok(ReservedOwnSetupAttempt {
            report,
            reused,
            ledger_index: self.ledger.len() - 1,
        })
    }
    pub(crate) fn resolve(
        &mut self,
        reservation: ReservedOwnSetupAttempt,
        configs: &[OwnSetupNativeConfig],
        mode: BatchRecords<'_>,
    ) -> Result<OwnSetupAttemptReport, String> {
        let ReservedOwnSetupAttempt {
            mut report,
            reused,
            ledger_index,
        } = reservation;
        self.check_reservation(&report, ledger_index, configs)?;
        let result = self
            .preflight(configs, report.first_episode_ordinal, reused)
            .and_then(|()| self.execute(configs, &mut report.receipts, mode));
        report.outcome = match result {
            Ok(outcome) => outcome,
            Err(reason) => OwnSetupAttemptOutcome::Rejected { reason },
        };
        report.setup_after = self.setup.checksum().into();
        self.ledger[ledger_index] = report.clone();
        Ok(report)
    }
    pub(crate) fn abandon(
        &mut self,
        reservation: ReservedOwnSetupAttempt,
        configs: &[OwnSetupNativeConfig],
    ) -> Result<OwnSetupAttemptReport, String> {
        let mut report = reservation.report;
        self.check_reservation(&report, reservation.ledger_index, configs)?;
        report.outcome = OwnSetupAttemptOutcome::Rejected {
            reason: "Interrupted durable reservation consumed without model work".into(),
        };
        self.ledger[reservation.ledger_index] = report.clone();
        Ok(report)
    }
    fn check_reservation(
        &self,
        report: &OwnSetupAttemptReport,
        index: usize,
        configs: &[OwnSetupNativeConfig],
    ) -> Result<(), String> {
        if index + 1 != self.ledger.len()
            || report.setup_before != self.setup.checksum()
            || report.attempted_episodes != self.attempted
            || report.next_episode_ordinal != self.next_episode
            || report.reserved_episodes != configs.len()
            || report.families
                != configs
                    .iter()
                    .map(|c| seed_family_id(c.native().environment_seed()))
                    .collect::<Vec<_>>()
        {
            return Err("Own Setup reservation/current owner mismatch".into());
        }
        Ok(())
    }
    fn preflight(
        &self,
        configs: &[OwnSetupNativeConfig],
        first: u64,
        reused: bool,
    ) -> Result<(), String> {
        if reused
            || self.optimizer.step != self.setup.count()
            || self.setup.count() >= trained::MAX_UPDATES
        {
            return Err("Spent family or incompatible Own Setup update count".into());
        }
        let mut tuples = BTreeSet::new();
        let mut total_bytes = 0usize;
        for (offset, c) in configs.iter().enumerate() {
            let native = c.native();
            let stream = native.sampling_identity();
            let family = seed_family_id(native.environment_seed());
            if stream.episode_ordinal() != first + offset as u64
                || stream.sampling_seed() != SamplingSeed::new(self.run_config.sampling_seed)
                || stream.replicate_ordinal() != self.run_config.replicate_ordinal
                || split_for_family(&family)? != DatasetSplit::Train
                || self.exclusions.families.contains(&family)
                || !tuples.insert((family, native.players(), c.learner_seat()))
            {
                return Err("Own Setup requires fresh Train families, unique seats and exact reserved streams".into());
            }
            total_bytes = total_bytes
                .checked_add(native.limits().max_source_bytes())
                .and_then(|n| n.checked_add(native.limits().max_trace_bytes()))
                .ok_or("Own Setup batch payload sum overflow")?;
        }
        if total_bytes > MAX_RECORD_BYTES {
            return Err("Own Setup batch declared payload exceeds aggregate cap".into());
        }
        if playing_binding_checksum(&fixed_playing_identity(self.playing()?)?)?
            != self.playing_checksum
        {
            return Err("Own Setup immutable Playing identity changed".into());
        }
        Ok(())
    }
    fn execute(
        &mut self,
        configs: &[OwnSetupNativeConfig],
        receipts: &mut Vec<OwnSetupEpisodeReceipt>,
        mode: BatchRecords<'_>,
    ) -> Result<OwnSetupAttemptOutcome, String> {
        let episodes = collect_batch(&self.setup, &mut self.playing, configs, receipts, mode)?;
        let model = self.setup.model();
        let mut gradient = vec![0.0; PARAMETER_COUNT];
        let identity = self.setup.identity_bytes()?;
        let mut ids = BTreeSet::new();
        let mut records = BTreeSet::new();
        let mut sources = Vec::new();
        for (episode, config) in episodes.iter().zip(configs) {
            let e = episode.view()?;
            if e.identity != identity
                || json(e.playing)? != json(&self.playing_binding)?
                || json(&e.config)?
                    != json(&crate::public_stochastic_record::Config::from_config(
                        config.native(),
                    ))?
                || e.learner_seat != config.learner_seat()
                || e.family != seed_family_id(config.native().environment_seed())
                || self.credited_ids.contains(e.id)
                || self.credited_records.contains(e.checksum)
                || !ids.insert(e.id.to_owned())
                || !records.insert(e.checksum.to_owned())
            {
                return Err("Stale, duplicate or mismatched Own Setup sealed episode".into());
            }
            let candidates = SetupCandidates::from_observation(e.observation)?;
            let pullback = OwnSetupScalarPullback::new(model, &candidates)?;
            let likelihood = checked_sample(pullback.logits(), e.sample, e.observation, config)?;
            let advantage = e.winner_share - 1.0 / config.native().players() as f64;
            let mut dlogits = [0.0; 6];
            likelihood.weighted_logit_gradient(likelihood.smooth(), advantage, &mut dlogits)?;
            pullback.accumulate_parameter_gradient(&dlogits, &mut gradient)?;
            sources.push(SetupUpdateSource {
                episode_id: e.id.into(),
                record_checksum: e.checksum.into(),
            });
        }
        if episodes.len() != configs.len() {
            return Err("Own Setup frozen batch is incomplete".into());
        }
        let proposal = adam_proposal(
            model,
            &self.optimizer,
            &gradient,
            episodes.len(),
            &self.adam_config,
        )?;
        if proposal.changed == 0 {
            return Ok(OwnSetupAttemptOutcome::NoChange);
        }
        let identity = TrainedOwnSetupIdentity {
            schema: trained::ARTIFACT_SCHEMA.into(),
            policy_version: trained::POLICY_VERSION.into(),
            sampling_version: trained::SAMPLING_VERSION.into(),
            task: trained::TASK.into(),
            initial: self.setup.initial().clone(),
            parent_checksum: self.setup.checksum().into(),
            update_count: proposal.optimizer.step,
            training_config_checksum: self.config_checksum.clone(),
            playing_binding_checksum: self.playing_checksum.clone(),
            numerical_target: crate::public_stochastic_native::numerical_target(),
            parameter_words_checksum: String::new(),
            sources,
            checksum: String::new(),
        };
        // Construct and validate everything before mutating parameters/moments/credit.
        let artifact = TrainedOwnSetupArtifact::from_update(identity, proposal.model)?;
        if playing_binding_checksum(&fixed_playing_identity(self.playing()?)?)?
            != self.playing_checksum
        {
            return Err("Own Setup Playing changed during collection".into());
        }
        self.setup = SetupState::Trained(Box::new(artifact));
        self.optimizer = proposal.optimizer;
        self.credited_ids.extend(ids);
        self.credited_records.extend(records);
        Ok(OwnSetupAttemptOutcome::Updated {
            update_count: self.setup.count(),
            changed_parameters: proposal.changed,
            max_parameter_delta: proposal.max_delta,
        })
    }
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(
    tag = "kind",
    rename_all = "camelCase",
    rename_all_fields = "camelCase",
    deny_unknown_fields
)]
pub(crate) enum SetupCheckpoint {
    Initialized {
        identity: Box<crate::own_setup_model::InitializedOwnSetupIdentity>,
        parameter_bits: Vec<String>,
    },
    Trained {
        identity: Box<TrainedOwnSetupIdentity>,
        parameter_bits: Vec<String>,
    },
}
#[derive(Debug, Deserialize, Serialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub(crate) struct OwnSetupCheckpointState {
    schema: String,
    setup: SetupCheckpoint,
    optimizer_step: u64,
    first_moment_bits: Vec<String>,
    second_moment_bits: Vec<String>,
    config_checksum: String,
    playing_binding: FixedPlayingIdentity,
    adam_config: OwnSetupAdamConfig,
    run_config: OwnSetupRunConfig,
    exclusion_count: usize,
    exclusion_digest: String,
    next_episode_ordinal: Option<u64>,
    attempted_episodes: u64,
    spent_families: Vec<String>,
    credited_episode_ids: Vec<String>,
    credited_record_checksums: Vec<String>,
}
impl OwnSetupTrainingOwner {
    pub(crate) fn checkpoint(&self) -> Result<OwnSetupCheckpointState, String> {
        let parameter_bits = self
            .setup
            .model()
            .parameters()
            .iter()
            .map(|v| format!("{:08x}", v.to_bits()))
            .collect();
        let setup = match &self.setup {
            SetupState::Initialized(a) => SetupCheckpoint::Initialized {
                identity: Box::new(a.identity.clone()),
                parameter_bits,
            },
            SetupState::Trained(a) => SetupCheckpoint::Trained {
                identity: Box::new(a.identity().clone()),
                parameter_bits,
            },
        };
        Ok(OwnSetupCheckpointState {
            schema: "tzolkin-own-setup-checkpoint-state-v1".into(),
            setup,
            optimizer_step: self.optimizer.step,
            first_moment_bits: self
                .optimizer
                .first
                .iter()
                .map(|v| hex64(v.to_bits()))
                .collect(),
            second_moment_bits: self
                .optimizer
                .second
                .iter()
                .map(|v| hex64(v.to_bits()))
                .collect(),
            config_checksum: self.config_checksum.clone(),
            playing_binding: crate::own_setup_native::fixed_playing_identity(self.playing()?)?,
            adam_config: self.adam_config.clone(),
            run_config: self.run_config.clone(),
            exclusion_count: self.exclusions.count(),
            exclusion_digest: self.exclusions.digest().into(),
            next_episode_ordinal: self.next_episode,
            attempted_episodes: self.attempted,
            spent_families: self.spent_families.iter().cloned().collect(),
            credited_episode_ids: self.credited_ids.iter().cloned().collect(),
            credited_record_checksums: self.credited_records.iter().cloned().collect(),
        })
    }
}
impl OwnSetupCheckpointState {
    // Numeric/shape refusal only. The durable reader must rebuild the live owner
    // through all reservations and real source audits, then compare every word.
    pub(crate) fn validate_claim(&self) -> Result<(), String> {
        if self.schema != "tzolkin-own-setup-checkpoint-state-v1"
            || !checkpoint_digest(&self.config_checksum)
            || !checkpoint_digest(&self.exclusion_digest)
            || self.first_moment_bits.len() != PARAMETER_COUNT
            || self.second_moment_bits.len() != PARAMETER_COUNT
            || !(1..=MAX_ATTEMPTED_EPISODES).contains(&self.run_config.max_attempted_episodes)
            || self.attempted_episodes > self.run_config.max_attempted_episodes
            || self.optimizer_step > self.attempted_episodes
            || self.next_episode_ordinal
                != self
                    .run_config
                    .first_episode_ordinal
                    .checked_add(self.attempted_episodes)
            || self.exclusion_count > MAX_EXCLUSIONS
            || self.spent_families.len() > self.attempted_episodes as usize
            || self.credited_episode_ids.len() > self.attempted_episodes as usize
            || self.credited_record_checksums.len() != self.credited_episode_ids.len()
        {
            return Err("Invalid closed Own Setup checkpoint shape/count/digest".into());
        }
        self.adam_config.validate()?;
        for list in [
            &self.spent_families,
            &self.credited_episode_ids,
            &self.credited_record_checksums,
        ] {
            if list.iter().any(|s| !checkpoint_digest(s)) || list.windows(2).any(|p| p[0] >= p[1]) {
                return Err("Own Setup checkpoint set closure differs".into());
            }
        }
        for (first, second) in self.first_moment_bits.iter().zip(&self.second_moment_bits) {
            let m = f64::from_bits(crate::public_stochastic_record::parse_hex64(first)?);
            let v = f64::from_bits(crate::public_stochastic_record::parse_hex64(second)?);
            if !m.is_finite()
                || !v.is_finite()
                || v < 0.0
                || (self.optimizer_step == 0 && (m != 0.0 || v != 0.0))
            {
                return Err("Nonfinite/incompatible Own Setup moment claim".into());
            }
        }
        let (bits, count) = match &self.setup {
            SetupCheckpoint::Initialized { parameter_bits, .. } => (parameter_bits, 0),
            SetupCheckpoint::Trained {
                identity,
                parameter_bits,
            } => (parameter_bits, identity.update_count),
        };
        if bits.len() != PARAMETER_COUNT || count != self.optimizer_step {
            return Err("Own Setup checkpoint parameter/count shape".into());
        }
        let mut parameters = Vec::with_capacity(PARAMETER_COUNT);
        for word in bits {
            if word.len() != 8
                || !word
                    .bytes()
                    .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
            {
                return Err("Invalid f32 checkpoint word".into());
            }
            parameters.push(f32::from_bits(
                u32::from_str_radix(word, 16).map_err(|e| e.to_string())?,
            ));
        }
        let model = OwnSetupModel {
            own_setup_parameters: parameters,
        };
        model.validate()?;
        match &self.setup {
            SetupCheckpoint::Initialized { identity, .. } => {
                let actual = InitializedOwnSetupArtifact::new(identity.seed)?;
                if identity.as_ref() != &actual.identity
                    || model
                        .parameters()
                        .iter()
                        .zip(actual.model.parameters())
                        .any(|(a, b)| a.to_bits() != b.to_bits())
                {
                    return Err("Initialized checkpoint words/identity changed".into());
                }
            }
            SetupCheckpoint::Trained { identity, .. } => {
                trained::validate_numeric_claim(identity, &model)?
            }
        }
        Ok(())
    }
}

fn json<T: Serialize>(v: &T) -> Result<Vec<u8>, String> {
    serde_json::to_vec(v).map_err(|e| e.to_string())
}

enum SealedEpisode {
    Initialized(Box<ValidatedOwnSetupEpisode>),
    Trained(Box<ValidatedTrainedOwnSetupEpisode>),
}
struct EpisodeView<'a> {
    id: &'a str,
    checksum: &'a str,
    family: &'a str,
    learner_seat: usize,
    config: crate::public_stochastic_record::Config,
    identity: Vec<u8>,
    playing: &'a FixedPlayingIdentity,
    observation: &'a Observation,
    sample: &'a SetupSample,
    winner_share: f64,
}
impl SealedEpisode {
    fn view(&self) -> Result<EpisodeView<'_>, String> {
        macro_rules! view {
            ($e:expr) => {{
                let e = $e;
                Ok(EpisodeView {
                    id: e.episode_id(),
                    checksum: e.record_checksum(),
                    family: e.family_id(),
                    learner_seat: e.learner_seat(),
                    config: crate::public_stochastic_record::Config::from_config(
                        &e.native_config()?,
                    ),
                    identity: json(e.setup_identity())?,
                    playing: e.playing_binding(),
                    observation: e.setup_observation(),
                    sample: e.sample(),
                    winner_share: e.winner_share(),
                })
            }};
        }
        match self {
            Self::Initialized(e) => view!(e),
            Self::Trained(e) => view!(e),
        }
    }
}
pub(crate) struct ReservedOwnSetupAttempt {
    report: OwnSetupAttemptReport,
    reused: bool,
    ledger_index: usize,
}
impl ReservedOwnSetupAttempt {
    pub(crate) fn report(&self) -> &OwnSetupAttemptReport {
        &self.report
    }
}
pub(crate) enum BatchRecords<'a> {
    Collect(Option<&'a mut Vec<Vec<u8>>>),
    Audit(&'a [Vec<u8>]),
}
trait BatchSource {
    fn one(
        &mut self,
        config: &OwnSetupNativeConfig,
        receipts: &mut Vec<OwnSetupEpisodeReceipt>,
        expected: Option<&[u8]>,
        captured: Option<&mut Vec<Vec<u8>>>,
    ) -> Result<SealedEpisode, String>;
}
macro_rules! batch_source {
    ($collector:ident,$encode:ident,$decode:ident,$variant:ident) => {
        impl BatchSource for $collector<'_> {
            fn one(
                &mut self,
                config: &OwnSetupNativeConfig,
                receipts: &mut Vec<OwnSetupEpisodeReceipt>,
                expected: Option<&[u8]>,
                captured: Option<&mut Vec<Vec<u8>>>,
            ) -> Result<SealedEpisode, String> {
                match expected {
                    None => {
                        let game = self.collect(config)?;
                        receipts.push(OwnSetupEpisodeReceipt {
                            record_checksum: game.checksum().into(),
                            episode_id: None,
                            counts: game.counts().clone(),
                        });
                        let bytes = own_setup_record::$encode(&game)?;
                        let audited = self.audit(&bytes);
                        if let Some(captured) = captured {
                            captured.push(bytes);
                        }
                        let episode = audited?.into_episode()?;
                        receipts
                            .last_mut()
                            .ok_or("Missing Own Setup receipt")?
                            .episode_id = Some(episode.episode_id().into());
                        Ok(SealedEpisode::$variant(Box::new(episode)))
                    }
                    Some(bytes) => {
                        let record = own_setup_record::$decode(bytes)?;
                        if json(&record.header.config)?
                            != json(&crate::public_stochastic_record::Config::from_config(
                                config.native(),
                            ))?
                            || record.header.learner_seat != config.learner_seat()
                        {
                            return Err("Restored Own Setup source config mismatch".into());
                        }
                        receipts.push(OwnSetupEpisodeReceipt {
                            record_checksum: record.checksum.clone(),
                            episode_id: None,
                            counts: record.counts.clone(),
                        });
                        let episode = self.audit(bytes)?.into_episode()?;
                        receipts
                            .last_mut()
                            .ok_or("Missing restored Own Setup receipt")?
                            .episode_id = Some(episode.episode_id().into());
                        Ok(SealedEpisode::$variant(Box::new(episode)))
                    }
                }
            }
        }
    };
}
batch_source!(OwnSetupCollector, encode_record, decode, Initialized);
batch_source!(
    TrainedOwnSetupCollector,
    encode_trained_record,
    decode_trained,
    Trained
);
fn collect_batch(
    setup: &SetupState,
    playing: &mut Option<LoadedDeployment>,
    configs: &[OwnSetupNativeConfig],
    receipts: &mut Vec<OwnSetupEpisodeReceipt>,
    mode: BatchRecords<'_>,
) -> Result<Vec<SealedEpisode>, String> {
    fn collect<C: BatchSource>(
        c: &mut C,
        configs: &[OwnSetupNativeConfig],
        receipts: &mut Vec<OwnSetupEpisodeReceipt>,
        mode: BatchRecords<'_>,
    ) -> Result<Vec<SealedEpisode>, String> {
        let mut episodes = Vec::new();
        match mode {
            BatchRecords::Collect(mut captured) => {
                for cfg in configs {
                    episodes.push(c.one(cfg, receipts, None, captured.as_deref_mut())?);
                }
            }
            BatchRecords::Audit(records) => {
                for (i, cfg) in configs.iter().enumerate() {
                    let bytes = records
                        .get(i)
                        .ok_or("Missing retained Own Setup source record")?;
                    episodes.push(c.one(cfg, receipts, Some(bytes), None)?);
                }
                if records.len() != configs.len() {
                    return Err("Extra retained Own Setup source records".into());
                }
            }
        }
        Ok(episodes)
    }
    macro_rules! batch {
        ($binding:expr,$collector:ident) => {{
            let binding = $binding?;
            let loaded = playing
                .take()
                .ok_or("Own Setup Playing storage unavailable")?;
            let mut collector = $collector::from_parts(binding, loaded);
            let result = collect(&mut collector, configs, receipts, mode);
            *playing = Some(collector.into_playing()?);
            result
        }};
    }
    match setup {
        SetupState::Initialized(a) => {
            batch!(BorrowedInitializedOwnSetup::new(a), OwnSetupCollector)
        }
        SetupState::Trained(a) => batch!(BorrowedTrainedOwnSetup::new(a), TrainedOwnSetupCollector),
    }
}

fn checked_sample(
    logits: &[f32],
    sample: &SetupSample,
    observation: &Observation,
    config: &OwnSetupNativeConfig,
) -> Result<BehaviorLikelihood, String> {
    let likelihood = BehaviorLikelihood::from_logits(logits, sample.chosen_index)?;
    let before = config
        .native()
        .sampling_identity()
        .initial_actor_state(config.learner_seat())?;
    let (after, raw) = next_splitmix(before);
    let u53 = raw >> 11;
    let mut cumulative = 0;
    let chosen = likelihood
        .mass_ticks()
        .iter()
        .position(|mass| {
            cumulative += mass;
            u53 < cumulative
        })
        .ok_or("Own Setup draw outside reconstructed masses")?;
    if logits.len() != 6
        || sample.logits_bits
            != logits
                .iter()
                .map(|z| format!("{:08x}", z.to_bits()))
                .collect::<Vec<_>>()
        || sample.logits_digest != full_logits_digest(logits)
        || sample.ordered_legal_digest != legal_digest(&observation.legal_actions)?
        || sample.nominal_probability_bits
            != likelihood
                .smooth()
                .probabilities()
                .iter()
                .map(|p| hex64(p.to_bits()))
                .collect::<Vec<_>>()
        || sample.mass_ticks != likelihood.mass_ticks()
        || sample.distribution_digest != likelihood.distribution_digest()
        || sample.chosen_nominal_probability_bits
            != hex64(likelihood.smooth().probabilities()[sample.chosen_index].to_bits())
        || sample.chosen_behavior_probability_bits
            != hex64(likelihood.behavior_probability().to_bits())
        || sample.chosen_behavior_logp_bits
            != hex64(likelihood.behavior_log_probability().to_bits())
        || sample.rng_state_before != hex64(before)
        || sample.rng_state_after != hex64(after)
        || sample.raw_u64 != hex64(raw)
        || sample.u53 != u53
        || sample.chosen_index != chosen
        || sample.draws_before != hex64(0)
        || sample.draws_after != hex64(1)
        || likelihood.mass_ticks().iter().sum::<u64>() != TICKS
        || likelihood.max_abs_log_delta() > MAX_ALL_LOG_DELTA
    {
        return Err("Own Setup scalar/full-mask/p/q/RNG reconstruction mismatch".into());
    }
    Ok(likelihood)
}

fn adam_proposal(
    model: &OwnSetupModel,
    state: &AdamState,
    gradient: &[f64],
    episodes: usize,
    config: &OwnSetupAdamConfig,
) -> Result<AdamProposal, String> {
    model.validate()?;
    config.validate()?;
    if !(1..=MAX_BATCH).contains(&episodes)
        || gradient.len() != PARAMETER_COUNT
        || gradient.iter().any(|v| !v.is_finite())
        || state.first.len() != PARAMETER_COUNT
        || state.second.len() != PARAMETER_COUNT
        || state.step >= trained::MAX_UPDATES
        || state.first.iter().any(|v| !v.is_finite())
        || state.second.iter().any(|v| !v.is_finite() || *v < 0.0)
    {
        return Err("Invalid Own Setup gradient/moment shape or values".into());
    }
    let norm = gradient
        .iter()
        .fold(0.0_f64, |n, g| n.hypot(*g / episodes as f64));
    if !norm.is_finite() {
        return Err("Nonfinite Own Setup gradient norm".into());
    }
    let scale = if norm > config.max_gradient_norm {
        config.max_gradient_norm / norm
    } else {
        1.0
    };
    let mut optimizer = state.clone();
    optimizer.step += 1;
    let correction1 = 1.0 - config.beta1.powf(optimizer.step as f64);
    let correction2 = 1.0 - config.beta2.powf(optimizer.step as f64);
    if correction1 <= 0.0 || correction2 <= 0.0 {
        return Err("Invalid Own Setup Adam bias correction".into());
    }
    let mut parameters = Vec::with_capacity(PARAMETER_COUNT);
    let mut changed = 0;
    let mut max_delta = 0.0_f64;
    for (i, &old) in model.parameters().iter().enumerate() {
        let g = (gradient[i] / episodes as f64) * scale;
        optimizer.first[i] = config.beta1 * state.first[i] + (1.0 - config.beta1) * g;
        optimizer.second[i] = config.beta2 * state.second[i] + (1.0 - config.beta2) * g * g;
        let step = config.learning_rate * (optimizer.first[i] / correction1)
            / ((optimizer.second[i] / correction2).sqrt() + config.epsilon);
        let proposed = (f64::from(old) + step) as f32;
        let new = if proposed == old { old } else { proposed };
        let delta = (f64::from(new) - f64::from(old)).abs();
        if !optimizer.first[i].is_finite()
            || !optimizer.second[i].is_finite()
            || !step.is_finite()
            || !new.is_finite()
            || !delta.is_finite()
            || delta > config.max_parameter_delta
        {
            return Err("Own Setup Adam proposal exceeds finite rounded delta bound".into());
        }
        changed += usize::from(new.to_bits() != old.to_bits());
        max_delta = max_delta.max(delta);
        parameters.push(new);
    }
    Ok(AdamProposal {
        model: OwnSetupModel {
            own_setup_parameters: parameters,
        },
        optimizer,
        changed,
        max_delta,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::public_stochastic::{SamplingStreamIdentity, UNIFORM_MIXTURE};
    use crate::public_stochastic_native::{CollectionLimits, NativeStochasticConfig};

    fn adam() -> OwnSetupAdamConfig {
        OwnSetupAdamConfig {
            learning_rate: 0.0001,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1.0e-8,
            max_gradient_norm: 2.0,
            max_parameter_delta: 0.001,
        }
    }
    fn exclusions(families: Vec<String>) -> DeclaredOwnSetupExclusions {
        let digest = DeclaredOwnSetupExclusions::digest_for_families(&families).unwrap();
        DeclaredOwnSetupExclusions::from_families(families.clone(), families.len(), &digest)
            .unwrap()
    }
    fn train_seed() -> u32 {
        (0..100)
            .find(|seed| split_for_family(&seed_family_id(*seed)).unwrap() == DatasetSplit::Train)
            .unwrap()
    }
    fn config(seed: u32, ordinal: u64, seat: usize) -> OwnSetupNativeConfig {
        OwnSetupNativeConfig::new(
            NativeStochasticConfig::new(
                3,
                seed,
                SamplingStreamIdentity::new(SamplingSeed::new(u64::MAX), ordinal, 7, 3).unwrap(),
                CollectionLimits::new(4, 1024, 1024 * 1024, 1024 * 1024).unwrap(),
            )
            .unwrap(),
            seat,
        )
        .unwrap()
    }
    fn owner(first: u64, excluded: Vec<String>) -> OwnSetupTrainingOwner {
        OwnSetupTrainingOwner::new(
            InitializedOwnSetupArtifact::new(7).unwrap(),
            crate::own_setup_native::tests::playing(),
            adam(),
            OwnSetupRunConfig {
                sampling_seed: u64::MAX,
                replicate_ordinal: 7,
                first_episode_ordinal: first,
                max_attempted_episodes: 4,
            },
            exclusions(excluded),
        )
        .unwrap()
    }

    #[test]
    fn corrected_mixture_gradient_and_atomic_adam_match_independent_oracles() {
        let logits = [0.2_f32, -0.4, 1.1, 0.03, -0.7, 0.9];
        let chosen = 2;
        let likelihood = BehaviorLikelihood::from_logits(&logits, chosen).unwrap();
        let advantage = 2.0 / 3.0;
        let mut derivative = [0.0; 6];
        likelihood
            .weighted_logit_gradient(likelihood.smooth(), advantage, &mut derivative)
            .unwrap();
        // Independent real-valued objective. The old p/q factor stays fixed.
        let objective = |z: &[f64]| {
            let exps: Vec<_> = z.iter().map(|x| x.exp()).collect();
            let sum = exps.iter().sum::<f64>();
            let p = (1.0 - UNIFORM_MIXTURE) * exps[chosen] / sum + UNIFORM_MIXTURE / 6.0;
            likelihood.importance_weight() * advantage * p.ln()
        };
        let shadow: Vec<_> = logits.iter().map(|v| f64::from(*v)).collect();
        for index in 0..6 {
            let mut plus = shadow.clone();
            let mut minus = shadow.clone();
            plus[index] += 1.0e-5;
            minus[index] -= 1.0e-5;
            let finite_difference = (objective(&plus) - objective(&minus)) / 2.0e-5;
            assert!((finite_difference - derivative[index]).abs() < 1.0e-9);
        }
        let initial = InitializedOwnSetupArtifact::new(17).unwrap();
        let mut state = AdamState {
            step: 0,
            first: vec![0.0; PARAMETER_COUNT],
            second: vec![0.0; PARAMETER_COUNT],
        };
        let mut model = initial.model;
        let cfg = adam();
        let mut shadow_first = [0.0_f64; 2];
        let mut shadow_second = [0.0_f64; 2];
        let mut carried: Option<OwnSetupTrainingOwner> = None;
        for (step, g) in [(1, [6.0_f64, -8.0]), (2, [-4.0, 3.0])] {
            let mut gradient = vec![0.0; PARAMETER_COUNT];
            gradient[0] = g[0];
            gradient[1] = g[1];
            let proposal = adam_proposal(&model, &state, &gradient, 2, &cfg).unwrap();
            let average = [g[0] / 2.0, g[1] / 2.0];
            let norm = average[0].hypot(average[1]);
            let scale = (cfg.max_gradient_norm / norm).min(1.0);
            for index in 0..2 {
                let clipped = average[index] * scale;
                shadow_first[index] = 0.9 * shadow_first[index] + 0.1 * clipped;
                shadow_second[index] = 0.999 * shadow_second[index] + 0.001 * clipped.powi(2);
                let delta = cfg.learning_rate * (shadow_first[index] / (1.0 - 0.9_f64.powi(step)))
                    / ((shadow_second[index] / (1.0 - 0.999_f64.powi(step))).sqrt() + cfg.epsilon);
                let expected = (f64::from(model.parameters()[index]) + delta) as f32;
                assert_eq!(
                    proposal.model.parameters()[index].to_bits(),
                    expected.to_bits()
                );
                assert!((proposal.optimizer.first[index] - shadow_first[index]).abs() < 1.0e-14);
                assert!((proposal.optimizer.second[index] - shadow_second[index]).abs() < 1.0e-14);
            }
            assert_eq!(proposal.optimizer.step, step as u64);
            assert_eq!(proposal.changed, 2);
            if let Some(owner) = &carried {
                let next = adam_proposal(owner.setup.model(), &owner.optimizer, &gradient, 2, &cfg)
                    .unwrap();
                assert_eq!(next.optimizer.first, proposal.optimizer.first);
                assert_eq!(next.optimizer.second, proposal.optimizer.second);
                assert_eq!(next.optimizer.step, proposal.optimizer.step);
                assert_eq!(
                    next.model
                        .parameters()
                        .iter()
                        .map(|v| v.to_bits())
                        .collect::<Vec<_>>(),
                    proposal
                        .model
                        .parameters()
                        .iter()
                        .map(|v| v.to_bits())
                        .collect::<Vec<_>>()
                );
            } else {
                // Mathematical fixture only: this grants no native source credit.
                let mut owner = OwnSetupTrainingOwner::new(
                    InitializedOwnSetupArtifact::new(17).unwrap(),
                    crate::own_setup_native::tests::playing(),
                    cfg.clone(),
                    OwnSetupRunConfig {
                        sampling_seed: u64::MAX,
                        replicate_ordinal: 7,
                        first_episode_ordinal: 0,
                        max_attempted_episodes: 2,
                    },
                    exclusions(vec![]),
                )
                .unwrap();
                let family = seed_family_id(train_seed());
                let identity = TrainedOwnSetupIdentity {
                    schema: trained::ARTIFACT_SCHEMA.into(),
                    policy_version: trained::POLICY_VERSION.into(),
                    sampling_version: trained::SAMPLING_VERSION.into(),
                    task: trained::TASK.into(),
                    initial: owner.setup.initial().clone(),
                    parent_checksum: owner.setup_checksum().into(),
                    update_count: 1,
                    training_config_checksum: owner.config_checksum.clone(),
                    playing_binding_checksum: owner.playing_checksum.clone(),
                    numerical_target: crate::public_stochastic_native::numerical_target(),
                    parameter_words_checksum: String::new(),
                    checksum: String::new(),
                    sources: vec![SetupUpdateSource {
                        episode_id: "a".repeat(64),
                        record_checksum: "b".repeat(64),
                    }],
                };
                owner.setup = SetupState::Trained(Box::new(
                    TrainedOwnSetupArtifact::from_update(
                        identity,
                        OwnSetupModel {
                            own_setup_parameters: proposal.model.parameters().to_vec(),
                        },
                    )
                    .unwrap(),
                ));
                owner.optimizer = proposal.optimizer.clone();
                owner.attempted = 2;
                owner.next_episode = Some(2);
                owner.spent_families.insert(family.clone());
                owner.credited_ids.insert("a".repeat(64));
                owner.credited_records.insert("b".repeat(64));
                owner.ledger.push(OwnSetupAttemptReport {
                    first_episode_ordinal: 0,
                    reserved_episodes: 2,
                    next_episode_ordinal: Some(2),
                    attempted_episodes: 2,
                    families: vec![family.clone()],
                    setup_before: String::new(),
                    setup_after: owner.setup_checksum().into(),
                    receipts: vec![],
                    outcome: OwnSetupAttemptOutcome::Updated {
                        update_count: 1,
                        changed_parameters: 2,
                        max_parameter_delta: proposal.max_delta,
                    },
                });
                let before = owner.checkpoint().unwrap();
                let old_identity = owner.setup.identity_bytes().unwrap();
                let old_ledger = json(&owner.ledger).unwrap();
                assert!(owner.continue_budget(2, exclusions(vec![])).is_err());
                assert_eq!(
                    json(&before).unwrap(),
                    json(&owner.checkpoint().unwrap()).unwrap()
                );
                owner.continue_budget(2, exclusions(vec![family])).unwrap();
                let after = owner.checkpoint().unwrap();
                assert_eq!(json(&before.setup).unwrap(), json(&after.setup).unwrap());
                assert_eq!(before.first_moment_bits, after.first_moment_bits);
                assert_eq!(before.second_moment_bits, after.second_moment_bits);
                assert_eq!(before.optimizer_step, after.optimizer_step);
                assert_eq!(before.next_episode_ordinal, after.next_episode_ordinal);
                assert_eq!(before.attempted_episodes, after.attempted_episodes);
                assert_eq!(before.spent_families, after.spent_families);
                assert_eq!(before.credited_episode_ids, after.credited_episode_ids);
                assert_eq!(
                    before.credited_record_checksums,
                    after.credited_record_checksums
                );
                assert_eq!(
                    before.run_config.first_episode_ordinal,
                    after.run_config.first_episode_ordinal
                );
                assert_eq!(after.run_config.max_attempted_episodes, 4);
                assert_ne!(before.config_checksum, after.config_checksum);
                assert_eq!(old_identity, owner.setup.identity_bytes().unwrap());
                assert_eq!(old_ledger, json(&owner.ledger).unwrap());
                carried = Some(owner);
            }
            model = proposal.model;
            state = proposal.optimizer;
        }
        let old_bits = model
            .parameters()
            .iter()
            .map(|v| v.to_bits())
            .collect::<Vec<_>>();
        let old_state = state.clone();
        let mut gradient = vec![0.0; PARAMETER_COUNT];
        gradient[0] = f64::INFINITY;
        assert!(adam_proposal(&model, &state, &gradient, 1, &cfg).is_err());
        gradient[0] = 1.0;
        let mut bounded = cfg.clone();
        bounded.max_parameter_delta = 1.0e-12;
        assert!(adam_proposal(&model, &state, &gradient, 1, &bounded).is_err());
        assert_eq!(
            model
                .parameters()
                .iter()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>(),
            old_bits
        );
        assert_eq!(state.first, old_state.first);
        assert_eq!(state.second, old_state.second);
        assert_eq!(state.step, old_state.step);
        let zero = AdamState {
            step: 0,
            first: vec![0.0; PARAMETER_COUNT],
            second: vec![0.0; PARAMETER_COUNT],
        };
        let no_change = adam_proposal(&model, &zero, &vec![0.0; PARAMETER_COUNT], 1, &cfg).unwrap();
        assert_eq!(no_change.changed, 0); // The live owner discards these proposed moments/step.
    }

    #[test]
    fn actual_failed_prefix_consumes_reserved_u64_slots_without_update_or_rewind() {
        let seed = train_seed();
        let mut learner = owner(u64::MAX - 1, vec![]);
        let before = learner.setup_checksum().to_owned();
        let playing = learner.playing().unwrap().checksum().to_owned();
        let report = learner
            .attempt(&[config(seed, u64::MAX - 1, 0), config(seed, u64::MAX, 1)])
            .unwrap();
        assert!(matches!(
            report.outcome,
            OwnSetupAttemptOutcome::Rejected { .. }
        ));
        assert_eq!(report.receipts.len(), 1);
        assert_eq!(report.receipts[0].episode_id, None);
        assert_eq!(report.receipts[0].counts.observed_callbacks, 4);
        assert_eq!(report.receipts[0].counts.guard_notifications, 4);
        assert_eq!(report.receipts[0].counts.setup_applied, 1);
        assert_eq!(
            report.receipts[0].counts.actor_draws,
            vec![hex64(1), hex64(0), hex64(0)]
        );
        assert_eq!(learner.attempted_episodes(), 2);
        assert_eq!(learner.next_episode_ordinal(), None);
        assert_eq!(learner.setup_checksum(), before);
        assert_eq!(learner.playing().unwrap().checksum(), playing);
        assert_eq!(learner.update_count(), 0);
        assert_eq!(learner.optimizer.step, 0);
        assert!(learner.optimizer.first.iter().all(|v| *v == 0.0));
        assert_eq!(learner.ledger().len(), 1);
        assert!(learner.trained_artifact().is_none());
        assert!(learner.attempt(&[config(seed, 0, 0)]).is_err());
        assert_eq!(learner.attempted_episodes(), 2);
        let mut learner = owner(0, vec![]);
        let first = learner.attempt(&[config(seed, 0, 0)]).unwrap();
        assert_eq!(first.receipts.len(), 1);
        let retry = learner.attempt(&[config(seed, 1, 1)]).unwrap();
        assert!(retry.receipts.is_empty());
        assert!(matches!(
            retry.outcome,
            OwnSetupAttemptOutcome::Rejected { .. }
        ));
        assert_eq!(learner.next_episode_ordinal(), Some(2));
    }

    #[test]
    fn exclusions_and_wrong_reserved_stream_refuse_before_model_work() {
        let seed = train_seed();
        let family = seed_family_id(seed);
        let mut excluded = owner(0, vec![family.clone()]);
        let report = excluded.attempt(&[config(seed, 0, 0)]).unwrap();
        assert!(report.receipts.is_empty());
        assert_eq!(excluded.attempted_episodes(), 1);
        assert_eq!(excluded.update_count(), 0);
        let mut learner = owner(0, vec![]);
        let report = learner.attempt(&[config(seed, 1, 0)]).unwrap();
        assert!(report.receipts.is_empty());
        assert_eq!(learner.next_episode_ordinal(), Some(1));
        assert!(
            DeclaredOwnSetupExclusions::from_families(
                vec![family.clone(), family.clone()],
                2,
                "a".repeat(64).as_str()
            )
            .is_err()
        );
        let digest =
            DeclaredOwnSetupExclusions::digest_for_families(std::slice::from_ref(&family)).unwrap();
        assert!(DeclaredOwnSetupExclusions::from_families(vec![family], 0, &digest).is_err());
        let initial = InitializedOwnSetupArtifact::new(7).unwrap();
        let mut collector =
            OwnSetupCollector::new(&initial, crate::own_setup_native::tests::playing()).unwrap();
        let prefix = collector.collect(&config(seed, 0, 0)).unwrap();
        let bytes = own_setup_record::encode_record(&prefix).unwrap();
        assert!(own_setup_record::decode_trained(&bytes).is_err());
        let mut changed = serde_json::from_slice::<serde_json::Value>(&bytes).unwrap();
        changed["schema"] = own_setup_record::TRAINED_RECORD_SCHEMA.into();
        let changed = serde_json::to_vec(&changed).unwrap();
        assert!(own_setup_record::decode(&changed).is_err());
        assert!(own_setup_record::decode_trained(&changed).is_err());
    }
}
