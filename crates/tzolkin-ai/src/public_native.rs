//! Qualified native V2 policy execution. Preparation checks consistency, not authenticity/strength.
use std::collections::BTreeSet;
use std::path::Path;

use crate::Decision;
use crate::dataset::{DatasetSplit, seed_family_id, split_for_family};
use crate::kernel::{Kernel, ResolvedKernel};
use crate::policy_dataset::{ValidatedPolicyDataset, load_policy_dataset};
use crate::policy_training::{PolicyTrainingCheckpoint, validate_checkpoint_dataset};
use crate::public_model::{LoadedPublicPolicy, PublicPolicyArtifact, PublicPolicyDistribution};
use crate::public_trade_guard::{
    self, PublicLearnedSource, TradeGuardCapture, TradeGuardConfig, TradeGuardSession,
    TradeGuardSummary,
};
use crate::replay::{self, GameReplay, ReplaySource, SeatPolicy};
use tzolkin_core::observation::Observation;
use tzolkin_core::{GameOptions, GameState};

/// Owns audited immutable checkpoint/model storage. Borrowed handles are separate, not self-referential.
pub struct PreparedPublicPolicy {
    checkpoint: Box<PolicyTrainingCheckpoint>,
    kernel: ResolvedKernel,
    non_test_families: BTreeSet<String>,
    train_families: BTreeSet<String>,
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
        let train_families = dataset
            .manifest()
            .games
            .iter()
            .filter(|game| game.split == DatasetSplit::Train)
            .map(|game| game.family_id.clone())
            .collect();
        Ok(Self {
            checkpoint: Box::new(checkpoint),
            kernel,
            non_test_families,
            train_families,
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
    /// Development comparisons may reuse Validation families; this is not a fresh Test gate.
    pub(crate) fn require_validation_families(&self, seeds: &[u32]) -> Result<(), String> {
        check_validation_families(&self.train_families, seeds)
    }
}

fn check_validation_families(
    train_families: &BTreeSet<String>,
    seeds: &[u32],
) -> Result<(), String> {
    for seed in seeds {
        let family = seed_family_id(*seed);
        if split_for_family(&family)? != DatasetSplit::Validation
            || train_families.contains(&family)
        {
            return Err(format!(
                "Seed {seed} is not a non-training public-policy validation family"
            ));
        }
    }
    Ok(())
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
    pub fn guarded_provenance(&self, guard: &TradeGuardConfig) -> Result<SeatPolicy, String> {
        let provenance = SeatPolicy::PublicLearnedTradeGuard {
            policy_version: public_trade_guard::POLICY_VERSION.into(),
            base: PublicLearnedSource::from_pure(self.provenance())?,
            guard: guard.clone(),
            configuration_key: guard.configuration_key()?,
        };
        provenance.validate()?;
        Ok(provenance)
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
    play_game_guarded_with_summary(policy, players, seed, options, seats, record, guard)?.game
}

/// In-game errors retain observed diagnostics; outer errors happen before a game.
pub struct PublicTradeGuardGameResult {
    pub game: Result<(GameState, usize, Option<GameReplay>), String>,
    /// Absolute-seat order. None denotes a non-guarded seat.
    pub trade_guard: Vec<Option<TradeGuardSummary>>,
}
pub fn play_game_guarded_with_summary(
    policy: &PublicPolicyHandle<'_>,
    players: usize,
    seed: u32,
    options: GameOptions,
    seats: &[usize],
    record: bool,
    guard: TradeGuardConfig,
) -> Result<PublicTradeGuardGameResult, String> {
    validate_seats(players, &options, seats)?;
    let provenance = policy.guarded_provenance(&guard)?;
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
    let mut trade_guard = (0..players)
        .map(|seat| {
            if seats.contains(&seat) {
                TradeGuardSummary::new(&provenance).map(Some)
            } else {
                Ok(None)
            }
        })
        .collect::<Result<Vec<_>, _>>()?;
    let mut capture = TradeGuardCapture::new(public_trade_guard::MAX_GAME_EXAMPLES);
    let mut run_capture = TradeGuardCapture::new(public_trade_guard::MAX_GAME_EXAMPLES);
    let game = replay::play_game_using_trade_guard(
        players,
        seed,
        options,
        record,
        ReplaySource::PolicySelfPlay { policies },
        true,
        |index, observation| {
            for (session, summary) in sessions.iter_mut().zip(&mut trade_guard) {
                if let Some(session) = session {
                    session.on_callback(index, observation)?;
                    summary
                        .as_mut()
                        .expect("matching guarded seat")
                        .notified(index, observation);
                }
            }
            if let Some(session) = sessions.get_mut(observation.actor).and_then(Option::as_mut) {
                let summary = trade_guard[observation.actor]
                    .as_mut()
                    .expect("matching guarded seat");
                summary.choice_started();
                let result = session.choose_with_distribution(index, observation, || {
                    policy.distribution(observation)
                });
                match result {
                    Ok((decision, trace)) => {
                        summary.returned(trace.as_ref(), &mut capture, &mut run_capture)?;
                        Ok((decision, trace))
                    }
                    Err(error) => {
                        summary.choice_failed();
                        Err(error)
                    }
                }
            } else {
                Ok((crate::choose_move(observation)?, None))
            }
        },
    )
    .and_then(|result| {
        if let Some(record) = &result.2 {
            replay::verify_replay(record)?;
        }
        Ok(result)
    });
    Ok(PublicTradeGuardGameResult { game, trade_guard })
}

#[cfg(test)]
pub(crate) mod integration_fixture {
    use super::*;
    use sha2::{Digest, Sha256};

    /// Untrained, synthetic parameters and format-valid metadata. This private
    /// fixture never bypasses the production PreparedPublicPolicy qualification.
    pub(crate) fn model(overflow: bool) -> PublicPolicyArtifact {
        let mut wire = serde_json::to_value(PublicPolicyArtifact::new(17).unwrap()).unwrap();
        let p = wire["model"]["parameters"].as_array_mut().unwrap();
        p.fill(serde_json::json!(0.0));
        p[384 + 5] = 2.0.into(); // end the turn when legally available
        p[384 + 8] = 1.0.into(); // voluntary pending-task Skip
        p[512 * 32 + 32] = 1.0.into();
        if overflow {
            p[231] = serde_json::json!(f32::MAX); // Setup phase feature
            p[512 * 32] = serde_json::json!(f32::MAX);
        }
        let mut artifact: PublicPolicyArtifact = serde_json::from_value(wire).unwrap();
        artifact.checksum.clear();
        artifact.checksum = format!(
            "{:x}",
            Sha256::digest(serde_json::to_vec(&artifact).unwrap())
        );
        artifact.validate().unwrap();
        artifact
    }
    pub(crate) fn handle(model: &PublicPolicyArtifact) -> PublicPolicyHandle<'_> {
        let provenance = SeatPolicy::PublicLearned {
            policy_version: model.policy_version.clone(),
            model_version: model.model_version.clone(),
            training_version: crate::policy_training::TRAINING_VERSION.into(),
            feature_schema: 2,
            input_contract: model.input_contract.clone(),
            task: "policyOnlyBc".into(),
            value_validity: model.value_validity,
            model_checksum: model.checksum.clone(),
            training_checkpoint_checksum: "b".repeat(64),
            dataset_fingerprint: "c".repeat(64),
            inference_backend: "scalar".into(),
        };
        provenance.validate().unwrap();
        PublicPolicyHandle {
            loaded: LoadedPublicPolicy::new(model).unwrap(),
            provenance,
        }
    }
}

#[cfg(test)]
mod integration_tests {
    use super::*;

    #[test]
    fn development_families_allow_validation_reuse_but_reject_train_test_and_overlap() {
        let train = BTreeSet::from([seed_family_id(0)]);
        assert_eq!(
            split_for_family(&seed_family_id(3)).unwrap(),
            DatasetSplit::Validation
        );
        check_validation_families(&train, &[3]).unwrap();
        check_validation_families(&train, &[3]).unwrap();
        for seeds in [&[0][..], &[10][..], &[3, 0][..]] {
            assert!(check_validation_families(&train, seeds).is_err());
        }
        let overlap = BTreeSet::from([seed_family_id(3)]);
        assert!(check_validation_families(&overlap, &[3]).is_err());
    }

    #[test]
    fn g2_native_outcome_and_old_adapter_match_and_sessions_restart() {
        let model = integration_fixture::model(false);
        let policy = integration_fixture::handle(&model);
        let original = serde_json::to_vec(&model).unwrap();
        for players in [3, 4] {
            let seats = (0..players).collect::<Vec<_>>();
            let a = play_game_guarded_with_summary(
                &policy,
                players,
                17,
                Default::default(),
                &seats,
                true,
                Default::default(),
            )
            .unwrap();
            let b = play_game_guarded(
                &policy,
                players,
                17,
                Default::default(),
                &seats,
                true,
                Default::default(),
            )
            .unwrap();
            let c = play_game_guarded_with_summary(
                &policy,
                players,
                17,
                Default::default(),
                &seats,
                true,
                Default::default(),
            )
            .unwrap();
            let (state, count, record) = a.game.unwrap();
            println!(
                "G2 synthetic native fixture seed17,{players}p three trajectories callbacks{count}"
            );
            assert_eq!(
                crate::replay::state_key(&state).unwrap(),
                crate::replay::state_key(&b.0).unwrap()
            );
            assert_eq!(
                serde_json::to_vec(&record).unwrap(),
                serde_json::to_vec(&b.2).unwrap()
            );
            assert_eq!(
                serde_json::to_vec(&record).unwrap(),
                serde_json::to_vec(&c.game.unwrap().2).unwrap()
            );
            assert_eq!(a.trade_guard, c.trade_guard);
            let summaries = a.trade_guard.iter().flatten().collect::<Vec<_>>();
            assert!(summaries.iter().all(
                |summary| summary.global_notifications == count && summary.failed_choices == 0
            ));
            assert_eq!(
                summaries
                    .iter()
                    .map(|summary| summary.choices_returned)
                    .sum::<usize>(),
                count
            );
            assert_eq!(serde_json::to_vec(&model).unwrap(), original);
        }
    }
    #[test]
    fn g2_native_failed_choice_retains_notification_and_null_game_result() {
        let model = integration_fixture::model(true);
        let policy = integration_fixture::handle(&model);
        let a = play_game_guarded_with_summary(
            &policy,
            3,
            17,
            Default::default(),
            &[0, 1],
            true,
            Default::default(),
        )
        .unwrap();
        assert!(a.game.unwrap_err().contains("hidden activation"));
        assert!(a.trade_guard[2].is_none());
        assert_eq!(
            a.trade_guard
                .iter()
                .flatten()
                .map(|s| s.guarded_decisions_observed)
                .sum::<usize>(),
            1
        );
        assert_eq!(
            a.trade_guard
                .iter()
                .flatten()
                .map(|s| s.failed_choices)
                .sum::<usize>(),
            1
        );
        assert!(
            a.trade_guard
                .iter()
                .flatten()
                .all(|s| s.global_notifications == 1 && s.choices_returned == 0)
        );
        assert!(a.trade_guard.iter().flatten().all(|s| {
            let last = s.last_callback.as_ref().unwrap();
            last.index == 0 && last.phase == tzolkin_core::Phase::Setup && !last.trade
        }));
    }
}
