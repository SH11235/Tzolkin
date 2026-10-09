//! Qualified native V2 policy execution. Preparation checks consistency, not authenticity/strength.
use std::collections::BTreeSet;
use std::path::Path;

use crate::Decision;
use crate::dataset::{DatasetSplit, seed_family_id, split_for_family};
use crate::kernel::{Kernel, ResolvedKernel};
use crate::policy_dataset::{ValidatedPolicyDataset, load_policy_dataset};
use crate::policy_training::{PolicyTrainingCheckpoint, validate_checkpoint_dataset};
use crate::public_model::{LoadedPublicPolicy, PublicPolicyArtifact, PublicPolicyDistribution};
use crate::public_trade_guard::{self, PublicLearnedSource, TradeGuardConfig, TradeGuardSession};
use crate::replay::{self, GameReplay, ReplaySource, SeatPolicy};
use tzolkin_core::observation::Observation;
use tzolkin_core::{GameOptions, GameState};

/// Owns audited immutable checkpoint/model storage. Borrowed handles are separate, not self-referential.
pub struct PreparedPublicPolicy {
    checkpoint: Box<PolicyTrainingCheckpoint>,
    kernel: ResolvedKernel,
    non_test_families: BTreeSet<String>,
}
impl PreparedPublicPolicy {
    pub fn load(checkpoint: &Path, dataset: &Path, kernel: Kernel) -> Result<Self, String> {
        let checkpoint = PolicyTrainingCheckpoint::load(checkpoint)?;
        let dataset = load_policy_dataset(dataset)?;
        Self::new(checkpoint, &dataset, kernel)
    }
    pub fn new(
        checkpoint: PolicyTrainingCheckpoint,
        dataset: &ValidatedPolicyDataset,
        kernel: Kernel,
    ) -> Result<Self, String> {
        let kernel = kernel.resolve()?;
        validate_checkpoint_dataset(dataset, &checkpoint)?;
        let non_test_families = dataset
            .manifest()
            .games
            .iter()
            .filter(|game| game.split != DatasetSplit::Test)
            .map(|game| game.family_id.clone())
            .collect();
        Ok(Self {
            checkpoint: Box::new(checkpoint),
            kernel,
            non_test_families,
        })
    }
    pub fn model(&self) -> &PublicPolicyArtifact {
        &self.checkpoint.model
    }
    pub fn backend(&self) -> &'static str {
        self.kernel.backend()
    }
    pub fn handle(&self) -> Result<PublicPolicyHandle<'_>, String> {
        let checkpoint = &self.checkpoint;
        let loaded = LoadedPublicPolicy::with_resolved(&checkpoint.model, self.kernel)?;
        let provenance = SeatPolicy::PublicLearned {
            policy_version: checkpoint.model.policy_version.clone(),
            model_version: checkpoint.model.model_version.clone(),
            training_version: checkpoint.training_version.clone(),
            feature_schema: checkpoint.feature_schema,
            input_contract: checkpoint.model.input_contract.clone(),
            task: checkpoint.task.clone(),
            value_validity: checkpoint.value_validity,
            model_checksum: checkpoint.model.checksum.clone(),
            training_checkpoint_checksum: checkpoint.checksum.clone(),
            dataset_fingerprint: checkpoint.dataset.fingerprint.clone(),
            inference_backend: loaded.backend().into(),
        };
        provenance.validate()?;
        Ok(PublicPolicyHandle { loaded, provenance })
    }
    /// Arena applies this to candidate, reference and every opponent policy before playing.
    pub(crate) fn require_test_families(&self, seeds: &[u32]) -> Result<(), String> {
        for seed in seeds {
            let family = seed_family_id(*seed);
            if split_for_family(&family)? != DatasetSplit::Test
                || self.non_test_families.contains(&family)
            {
                return Err(format!(
                    "Seed {seed} is not a held-out public-policy test family"
                ));
            }
        }
        Ok(())
    }
}

pub struct PublicPolicyHandle<'a> {
    loaded: LoadedPublicPolicy<'a>,
    provenance: SeatPolicy,
}
impl PublicPolicyHandle<'_> {
    pub fn backend(&self) -> &'static str {
        self.loaded.backend()
    }
    pub fn provenance(&self) -> &SeatPolicy {
        &self.provenance
    }
    pub fn choose_move(&self, observation: &Observation) -> Result<Decision, String> {
        self.loaded.choose_move(observation)
    }
    pub(crate) fn distribution(
        &self,
        observation: &Observation,
    ) -> Result<PublicPolicyDistribution, String> {
        self.loaded.distribution(observation)
    }
}

pub fn parse_kernel(value: &str) -> Result<Kernel, String> {
    match value {
        "scalar" => Ok(Kernel::Scalar),
        "auto" => Ok(Kernel::Auto),
        "avx2" => Ok(Kernel::Avx2),
        "sse2" => Ok(Kernel::Sse2),
        "neon" => Ok(Kernel::Neon),
        "simd128" => Ok(Kernel::Simd128),
        _ => Err("Unknown public-policy inference kernel".into()),
    }
}
pub fn validate_seats(
    players: usize,
    options: &GameOptions,
    seats: &[usize],
) -> Result<(), String> {
    if !(3..=4).contains(&players)
        || *options != GameOptions::default()
        || seats.is_empty()
        || seats.len() > players
        || seats.iter().any(|seat| *seat >= players)
        || seats
            .iter()
            .enumerate()
            .any(|(i, seat)| seats[..i].contains(seat))
    {
        return Err(
            "Public-policy selfplay requires base 3/4p and distinct nonempty absolute seats".into(),
        );
    }
    Ok(())
}
/// Uses only each decision actor's core Observation. Model errors remain errors, without fallback.
pub fn play_game(
    policy: &PublicPolicyHandle<'_>,
    players: usize,
    seed: u32,
    options: GameOptions,
    seats: &[usize],
    record: bool,
) -> Result<(GameState, usize, Option<GameReplay>), String> {
    validate_seats(players, &options, seats)?;
    let policies = (0..players)
        .map(|seat| {
            if seats.contains(&seat) {
                policy.provenance.clone()
            } else {
                crate::experiment::heuristic_seat()
            }
        })
        .collect();
    let result = replay::play_game_using_fast(
        players,
        seed,
        options,
        record,
        ReplaySource::PolicySelfPlay { policies },
        |observation| {
            if seats.contains(&observation.actor) {
                policy.choose_move(observation)
            } else {
                crate::choose_move(observation)
            }
        },
    )?;
    if let Some(record) = &result.2 {
        replay::verify_replay(record)?;
    }
    Ok(result)
}

/// Opt-in composite algorithm, with one fresh session and all-seat notifications.
/// Source consistency is audited; neither this metadata nor replay traces authenticate a producer.
pub fn play_game_guarded(
    policy: &PublicPolicyHandle<'_>,
    players: usize,
    seed: u32,
    options: GameOptions,
    seats: &[usize],
    record: bool,
    guard: TradeGuardConfig,
) -> Result<(GameState, usize, Option<GameReplay>), String> {
    validate_seats(players, &options, seats)?;
    let base = PublicLearnedSource::from_pure(policy.provenance())?;
    let configuration_key = guard.configuration_key()?;
    let provenance = SeatPolicy::PublicLearnedTradeGuard {
        policy_version: public_trade_guard::POLICY_VERSION.into(),
        base,
        guard: guard.clone(),
        configuration_key,
    };
    provenance.validate()?;
    let policies = (0..players)
        .map(|seat| {
            if seats.contains(&seat) {
                provenance.clone()
            } else {
                crate::experiment::heuristic_seat()
            }
        })
        .collect();
    let mut sessions = (0..players)
        .map(|seat| {
            if seats.contains(&seat) {
                TradeGuardSession::new(guard.clone()).map(Some)
            } else {
                Ok(None)
            }
        })
        .collect::<Result<Vec<_>, _>>()?;
    let result = replay::play_game_using_trade_guard(
        players,
        seed,
        options,
        record,
        ReplaySource::PolicySelfPlay { policies },
        true,
        |index, observation| {
            for session in sessions.iter_mut().flatten() {
                session.on_callback(index, observation)?;
            }
            if let Some(session) = sessions.get_mut(observation.actor).and_then(Option::as_mut) {
                session.choose_with_distribution(index, observation, || {
                    policy.distribution(observation)
                })
            } else {
                Ok((crate::choose_move(observation)?, None))
            }
        },
    )?;
    if let Some(record) = &result.2 {
        replay::verify_replay(record)?;
    }
    Ok(result)
}
