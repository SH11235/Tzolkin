//! Paired, seat-rotated experiments. Policies receive only actor observations.
use crate::dataset::{DatasetSplit, load_dataset, split_for_family};
// Keep the original arena API while all callers use the dataset's family identity.
pub use crate::dataset::seed_family_id as family_id;
use crate::kernel::Kernel;
use crate::model::{LoadedPolicy, ModelArtifact};
use crate::policy::HeuristicWeights;
use crate::replay::{self, ReplaySource, SeatPolicy};
use crate::training::TrainingCheckpoint;
use crate::{Decision, choose_move_with_weights};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::path::{Path, PathBuf};
use std::time::Instant;
use tzolkin_core::observation::{Observation, PublicPlayer, observe};
use tzolkin_core::{FinalScore, GameOptions, Phase};

pub const ARENA_SCHEMA: u32 = 1;
const MAX_SEEDS: usize = 1024;
const BOOTSTRAP_DRAWS: usize = 2048;

#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub enum Partition {
    Pilot,
    Validation,
    Test,
}
impl Partition {
    fn split(self) -> DatasetSplit {
        match self {
            Self::Pilot => DatasetSplit::Train,
            Self::Validation => DatasetSplit::Validation,
            Self::Test => DatasetSplit::Test,
        }
    }
}
pub fn seed_partition(seed: u32) -> Result<DatasetSplit, String> {
    split_for_family(&family_id(seed))
}
/// Choose a reproducible list without touching games, models or private state.
pub fn partition_seeds(partition: Partition, start: u32, count: usize) -> Result<Vec<u32>, String> {
    if count == 0 || count > MAX_SEEDS {
        return Err("Arena needs 1..1024 seed blocks".into());
    }
    let mut selected = Vec::with_capacity(count);
    for offset in 0..1_000_000u32 {
        let seed = start.checked_add(offset).ok_or("Seed selection overflow")?;
        if seed_partition(seed)? == partition.split() {
            selected.push(seed);
            if selected.len() == count {
                return Ok(selected);
            }
        }
    }
    Err("Not enough seed families in the bounded selection window".into())
}
fn scalar() -> String {
    "scalar".into()
}
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(
    tag = "kind",
    rename_all = "camelCase",
    rename_all_fields = "camelCase",
    deny_unknown_fields
)]
pub enum PolicyConfig {
    Heuristic {
        weights: HeuristicWeights,
    },
    Learned {
        checkpoint: PathBuf,
        dataset: PathBuf,
        #[serde(default = "scalar")]
        kernel: String,
    },
}
impl Default for PolicyConfig {
    fn default() -> Self {
        Self::Heuristic {
            weights: HeuristicWeights::default(),
        }
    }
}
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct ArenaConfig {
    pub schema: u32,
    pub players: usize,
    pub partition: Partition,
    pub seeds: Vec<u32>,
    pub candidate: PolicyConfig,
    pub reference: PolicyConfig,
    /// Absolute-seat assignments remain identical between each paired game.
    pub opponent_pool: Vec<PolicyConfig>,
    pub bootstrap_seed: u64,
}
impl ArenaConfig {
    pub fn validate(&self) -> Result<(), String> {
        if self.schema != ARENA_SCHEMA
            || !(3..=4).contains(&self.players)
            || self.opponent_pool.len() != self.players
            || self.seeds.is_empty()
            || self.seeds.len() > MAX_SEEDS
        {
            return Err(
                "Arena requires schema1, base rules, 3..4 players and 1..1024 seed blocks".into(),
            );
        }
        for (index, seed) in self.seeds.iter().enumerate() {
            if self.seeds[..index].contains(seed)
                || seed_partition(*seed)? != self.partition.split()
            {
                return Err("Repeated seed or wrong predeclared seed partition".into());
            }
        }
        Ok(())
    }
}
enum PreparedPolicy {
    Heuristic(HeuristicWeights),
    Learned {
        model: Box<ModelArtifact>,
        kernel: Kernel,
        dataset_fingerprint: String,
    },
}
impl PreparedPolicy {
    fn load(config: &PolicyConfig, arena: &ArenaConfig, base: &Path) -> Result<Self, String> {
        match config {
            PolicyConfig::Heuristic { weights } => {
                weights.validate()?;
                Ok(Self::Heuristic(weights.clone()))
            }
            PolicyConfig::Learned {
                checkpoint,
                dataset,
                kernel,
            } => {
                if arena.partition != Partition::Test {
                    return Err("Learned arena policies require the held-out test partition".into());
                }
                let checkpoint = TrainingCheckpoint::load(&base.join(checkpoint))?;
                let dataset = load_dataset(&base.join(dataset))?;
                if checkpoint.dataset_fingerprint != dataset.manifest().fingerprint {
                    return Err("Arena checkpoint/dataset fingerprint mismatch".into());
                }
                for seed in &arena.seeds {
                    let family = family_id(*seed);
                    if dataset
                        .manifest()
                        .games
                        .iter()
                        .any(|game| game.family_id == family && game.split != DatasetSplit::Test)
                    {
                        return Err("Arena seed overlaps model training or validation".into());
                    }
                }
                let kernel = match kernel.as_str() {
                    "scalar" => Kernel::Scalar,
                    "auto" => Kernel::Auto,
                    "avx2" => Kernel::Avx2,
                    "sse2" => Kernel::Sse2,
                    "neon" => Kernel::Neon,
                    "simd128" => Kernel::Simd128,
                    _ => return Err("Unknown arena inference kernel".into()),
                };
                Ok(Self::Learned {
                    model: Box::new(checkpoint.model),
                    kernel,
                    dataset_fingerprint: dataset.manifest().fingerprint.clone(),
                })
            }
        }
    }
}
#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct PolicyDescription {
    pub provenance: SeatPolicy,
    pub dataset_fingerprint: Option<String>,
}
enum PolicyChooser<'a> {
    Heuristic(&'a HeuristicWeights),
    Learned(LoadedPolicy<'a>),
}
/// Borrowing the prepared policies keeps coefficients/model bytes immutable for
/// the entire run. Validation, backend resolution and provenance happen before
/// game clocks start, rather than once per seat and arm.
struct PolicyHandle<'a> {
    chooser: PolicyChooser<'a>,
    description: PolicyDescription,
}
impl<'a> PolicyHandle<'a> {
    fn new(policy: &'a PreparedPolicy) -> Result<Self, String> {
        let (chooser, provenance, dataset_fingerprint) = match policy {
            PreparedPolicy::Heuristic(weights) => (
                PolicyChooser::Heuristic(weights),
                SeatPolicy::Heuristic {
                    policy_version: crate::POLICY_VERSION.into(),
                    weights: weights.clone(),
                },
                None,
            ),
            PreparedPolicy::Learned {
                model,
                kernel,
                dataset_fingerprint,
            } => {
                let loaded = LoadedPolicy::with_kernel(model, *kernel)?;
                let provenance = SeatPolicy::Learned {
                    policy_version: crate::model::LEARNED_POLICY_VERSION.into(),
                    model_checksum: model.checksum.clone(),
                    inference_backend: Some(loaded.backend().into()),
                };
                (
                    PolicyChooser::Learned(loaded),
                    provenance,
                    Some(dataset_fingerprint.clone()),
                )
            }
        };
        Ok(Self {
            chooser,
            description: PolicyDescription {
                provenance,
                dataset_fingerprint,
            },
        })
    }
    fn choose(&self, observation: &Observation) -> Result<Decision, String> {
        match &self.chooser {
            PolicyChooser::Heuristic(weights) => choose_move_with_weights(observation, weights),
            PolicyChooser::Learned(policy) => policy.choose_move(observation),
        }
    }
}
#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ArmResult {
    pub decisions: Option<usize>,
    pub elapsed_ms: f64,
    pub winner_utility: Option<f64>,
    pub score: Option<f64>,
    pub rank: Option<usize>,
    /// This runner does not collect complete feeding events; never infer from truncated UI logs.
    pub unfed_workers: Option<usize>,
    pub final_scores: Vec<FinalScore>,
    /// Allowlisted terminal public players only; never private offers or decks.
    pub terminal_players: Option<Vec<PublicPlayer>>,
    pub start_of_play: Option<InitialConditions>,
    pub error: Option<String>,
}
/// Public state after all seats have selected their initial wealth. Offers stay private.
#[derive(Clone, Debug, Serialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct InitialConditions {
    pub round: i64,
    pub first_player: usize,
    pub turn_order: Vec<usize>,
    pub resources: Vec<[i64; 5]>,
    pub wealth: Vec<Vec<String>>,
    pub workers: Vec<i64>,
    pub technologies: Vec<[i64; 4]>,
    pub temples: Vec<[i64; 3]>,
    pub buildings: Vec<String>,
    pub monuments: Vec<String>,
}
impl InitialConditions {
    fn from_observation(observation: &Observation) -> Self {
        Self {
            round: observation.round,
            first_player: observation.first_player,
            turn_order: observation.turn_order.clone(),
            resources: observation
                .players
                .iter()
                .map(|player| player.resources)
                .collect(),
            wealth: observation
                .players
                .iter()
                .map(|player| player.wealth.clone())
                .collect(),
            workers: observation
                .players
                .iter()
                .map(|player| player.workers)
                .collect(),
            technologies: observation
                .players
                .iter()
                .map(|player| player.technologies)
                .collect(),
            temples: observation
                .players
                .iter()
                .map(|player| player.temples)
                .collect(),
            buildings: observation.buildings.clone(),
            monuments: observation.monuments.clone(),
        }
    }
}
#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct SeatPair {
    pub seat: usize,
    pub candidate: ArmResult,
    pub reference: ArmResult,
}
#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct SeedBlock {
    pub seed: u32,
    pub family_id: String,
    pub pairs: Vec<SeatPair>,
    pub complete: bool,
    pub mean_utility_delta: Option<f64>,
    pub mean_score_delta: Option<f64>,
    pub mean_rank_improvement: Option<f64>,
}
fn play_arm(
    seed: u32,
    seat: usize,
    focal: &PolicyHandle<'_>,
    pool: &[PolicyHandle<'_>],
) -> ArmResult {
    let started = Instant::now();
    let mut start_of_play = None;
    let played = (|| {
        let policies = pool
            .iter()
            .enumerate()
            .map(|(index, opponent)| {
                if index == seat {
                    focal.description.provenance.clone()
                } else {
                    opponent.description.provenance.clone()
                }
            })
            .collect();
        let (state, decisions, _) = replay::play_game_using_fast(
            pool.len(),
            seed,
            GameOptions::default(),
            false,
            ReplaySource::PolicySelfPlay { policies },
            |observation| {
                if start_of_play.is_none() && observation.phase == Phase::Playing {
                    start_of_play = Some(InitialConditions::from_observation(observation));
                }
                if observation.actor == seat {
                    focal.choose(observation)
                } else {
                    pool[observation.actor].choose(observation)
                }
            },
        )?;
        let terminal_players = observe(&state, state.current_player)?.players;
        Ok::<_, String>((state, decisions, terminal_players))
    })();
    match played {
        Ok((state, decisions, terminal_players)) => {
            let winners = state
                .final_scores
                .iter()
                .filter(|score| score.rank == 1)
                .count();
            let focal = state
                .final_scores
                .iter()
                .find(|score| score.player_id == seat)
                .expect("Validated native terminal contains each seat");
            ArmResult {
                decisions: Some(decisions),
                elapsed_ms: started.elapsed().as_secs_f64() * 1000.0,
                winner_utility: Some(if focal.rank == 1 {
                    1.0 / winners as f64
                } else {
                    0.0
                }),
                score: Some(focal.total),
                rank: Some(focal.rank),
                unfed_workers: None,
                final_scores: state.final_scores,
                terminal_players: Some(terminal_players),
                start_of_play,
                error: None,
            }
        }
        Err(error) => ArmResult {
            decisions: None,
            elapsed_ms: started.elapsed().as_secs_f64() * 1000.0,
            winner_utility: None,
            score: None,
            rank: None,
            unfed_workers: None,
            final_scores: vec![],
            terminal_players: None,
            start_of_play,
            error: Some(error),
        },
    }
}
fn block(seed: u32, pairs: Vec<SeatPair>) -> SeedBlock {
    let complete = pairs
        .iter()
        .all(|pair| pair.candidate.error.is_none() && pair.reference.error.is_none());
    let mean = |value: fn(&SeatPair) -> f64| {
        complete.then(|| pairs.iter().map(value).sum::<f64>() / pairs.len() as f64)
    };
    SeedBlock {
        seed,
        family_id: family_id(seed),
        mean_utility_delta: mean(|pair| {
            pair.candidate.winner_utility.unwrap() - pair.reference.winner_utility.unwrap()
        }),
        mean_score_delta: mean(|pair| {
            pair.candidate.score.unwrap() - pair.reference.score.unwrap()
        }),
        mean_rank_improvement: mean(|pair| {
            pair.reference.rank.unwrap() as f64 - pair.candidate.rank.unwrap() as f64
        }),
        pairs,
        complete,
    }
}
#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct PairedStatistics {
    pub complete_blocks: usize,
    pub incomplete_blocks: usize,
    pub planned_games: usize,
    pub completed_games: usize,
    pub failed_games: usize,
    pub candidate_failed_games: usize,
    pub reference_failed_games: usize,
    pub candidate_mean_utility: Option<f64>,
    pub reference_mean_utility: Option<f64>,
    pub mean_utility_delta: Option<f64>,
    pub mean_score_delta: Option<f64>,
    pub mean_rank_improvement: Option<f64>,
    pub utility_delta_bootstrap95: Option<[f64; 2]>,
    pub confidence_method: String,
    pub bootstrap_draws: usize,
    pub bootstrap_seed: u64,
    pub failures_preclude_adoption: bool,
    pub strength_improvement_declared: bool,
}
/// Percentile interval resamples entire seed blocks, retaining all seat correlations.
fn summarize(blocks: &[SeedBlock], bootstrap_seed: u64) -> PairedStatistics {
    let complete: Vec<_> = blocks.iter().filter(|block| block.complete).collect();
    let mean = |value: fn(&SeedBlock) -> f64| {
        (!complete.is_empty())
            .then(|| complete.iter().map(|block| value(block)).sum::<f64>() / complete.len() as f64)
    };
    let utility_delta = mean(|block| block.mean_utility_delta.unwrap());
    let interval = if complete.len() < 2 {
        None
    } else {
        let deltas: Vec<_> = complete
            .iter()
            .map(|block| block.mean_utility_delta.unwrap())
            .collect();
        let mut random = bootstrap_seed ^ 0x9e3779b97f4a7c15;
        let mut means = Vec::with_capacity(BOOTSTRAP_DRAWS);
        for _ in 0..BOOTSTRAP_DRAWS {
            let mut sum = 0.0;
            for _ in 0..deltas.len() {
                // SplitMix64; unbiased bounded rejection, independent of game RNG and policies.
                let bound = deltas.len() as u64;
                let threshold = bound.wrapping_neg() % bound;
                let index = loop {
                    random = random.wrapping_add(0x9e3779b97f4a7c15);
                    let mut value = random;
                    value = (value ^ (value >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
                    value = (value ^ (value >> 27)).wrapping_mul(0x94d049bb133111eb);
                    value ^= value >> 31;
                    if value >= threshold {
                        break (value % bound) as usize;
                    }
                };
                sum += deltas[index];
            }
            means.push(sum / deltas.len() as f64);
        }
        means.sort_by(f64::total_cmp);
        Some([
            means[(BOOTSTRAP_DRAWS - 1) * 25 / 1000],
            means[(BOOTSTRAP_DRAWS - 1) * 975 / 1000],
        ])
    };
    let planned_games = blocks
        .iter()
        .map(|block| block.pairs.len() * 2)
        .sum::<usize>();
    let candidate_failed_games = blocks
        .iter()
        .flat_map(|block| &block.pairs)
        .filter(|pair| pair.candidate.error.is_some())
        .count();
    let reference_failed_games = blocks
        .iter()
        .flat_map(|block| &block.pairs)
        .filter(|pair| pair.reference.error.is_some())
        .count();
    let failed_games = candidate_failed_games + reference_failed_games;
    PairedStatistics {
        complete_blocks: complete.len(),
        incomplete_blocks: blocks.len() - complete.len(),
        planned_games,
        completed_games: planned_games - failed_games,
        failed_games,
        candidate_failed_games,
        reference_failed_games,
        candidate_mean_utility: mean(|block| {
            block
                .pairs
                .iter()
                .map(|pair| pair.candidate.winner_utility.unwrap())
                .sum::<f64>()
                / block.pairs.len() as f64
        }),
        reference_mean_utility: mean(|block| {
            block
                .pairs
                .iter()
                .map(|pair| pair.reference.winner_utility.unwrap())
                .sum::<f64>()
                / block.pairs.len() as f64
        }),
        mean_utility_delta: utility_delta,
        mean_score_delta: mean(|block| block.mean_score_delta.unwrap()),
        mean_rank_improvement: mean(|block| block.mean_rank_improvement.unwrap()),
        utility_delta_bootstrap95: interval,
        confidence_method: "percentile bootstrap of paired seed blocks, 95%, complete blocks only"
            .into(),
        bootstrap_draws: BOOTSTRAP_DRAWS,
        bootstrap_seed,
        failures_preclude_adoption: failed_games != 0,
        strength_improvement_declared: false,
    }
}
#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ArenaReport {
    pub schema: u32,
    pub config_sha256: String,
    pub players: usize,
    pub options: GameOptions,
    pub partition: Partition,
    pub candidate: PolicyDescription,
    pub reference: PolicyDescription,
    pub opponent_pool: Vec<PolicyDescription>,
    pub blocks: Vec<SeedBlock>,
    pub statistics: PairedStatistics,
    pub rules_baseline: String,
    pub catalog_hash: String,
    pub policy_input: String,
    pub feeding_events_collected: bool,
    pub build_profile: String,
    pub target_arch: String,
    pub target_os: String,
}
impl ArenaReport {
    pub fn save_new(&self, path: &Path) -> Result<(), String> {
        crate::model::write_new_json_bounded(path, self, 32 * 1024 * 1024)
    }
}
pub fn run_arena(config: &ArenaConfig, relative_to: &Path) -> Result<ArenaReport, String> {
    config.validate()?;
    let candidate = PreparedPolicy::load(&config.candidate, config, relative_to)?;
    let reference = PreparedPolicy::load(&config.reference, config, relative_to)?;
    let pool = config
        .opponent_pool
        .iter()
        .map(|policy| PreparedPolicy::load(policy, config, relative_to))
        .collect::<Result<Vec<_>, _>>()?;
    let candidate = PolicyHandle::new(&candidate)?;
    let reference = PolicyHandle::new(&reference)?;
    let pool = pool
        .iter()
        .map(PolicyHandle::new)
        .collect::<Result<Vec<_>, _>>()?;
    let mut blocks = Vec::with_capacity(config.seeds.len());
    for (index, seed) in config.seeds.iter().enumerate() {
        let mut pairs = Vec::with_capacity(config.players);
        for seat in 0..config.players {
            // Alternate paired run order to avoid measuring one arm systematically later.
            let (candidate_result, reference_result) = if (index + seat) % 2 == 0 {
                (
                    play_arm(*seed, seat, &candidate, &pool),
                    play_arm(*seed, seat, &reference, &pool),
                )
            } else {
                let reference_result = play_arm(*seed, seat, &reference, &pool);
                (play_arm(*seed, seat, &candidate, &pool), reference_result)
            };
            pairs.push(SeatPair {
                seat,
                candidate: candidate_result,
                reference: reference_result,
            });
        }
        blocks.push(block(*seed, pairs));
    }
    let statistics = summarize(&blocks, config.bootstrap_seed);
    Ok(ArenaReport {
        schema: ARENA_SCHEMA,
        config_sha256: format!(
            "{:x}",
            Sha256::digest(serde_json::to_vec(config).map_err(|error| error.to_string())?)
        ),
        players: config.players,
        options: GameOptions::default(),
        partition: config.partition,
        candidate: candidate.description,
        reference: reference.description,
        opponent_pool: pool
            .iter()
            .map(|policy| policy.description.clone())
            .collect(),
        blocks,
        statistics,
        rules_baseline: replay::RULES_BASELINE.into(),
        catalog_hash: replay::catalog_hash(),
        policy_input: "current actor allowlisted Observation only; no runner seed or hidden state"
            .into(),
        feeding_events_collected: false,
        build_profile: if cfg!(debug_assertions) {
            "debug"
        } else {
            "release"
        }
        .into(),
        target_arch: std::env::consts::ARCH.into(),
        target_os: std::env::consts::OS.into(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn arm(utility: f64) -> ArmResult {
        ArmResult {
            decisions: Some(100),
            elapsed_ms: 1.0,
            winner_utility: Some(utility),
            score: Some(utility * 10.0),
            rank: Some(if utility > 0.0 { 1 } else { 3 }),
            unfed_workers: None,
            final_scores: vec![],
            terminal_players: None,
            start_of_play: None,
            error: None,
        }
    }
    fn pairs(utility: f64) -> Vec<SeatPair> {
        (0..3)
            .map(|seat| SeatPair {
                seat,
                candidate: arm(utility),
                reference: arm(0.0),
            })
            .collect()
    }
    #[test]
    fn bootstrap_resamples_seed_blocks_and_retains_missing_failed_games() {
        // Perfect correlation across seats: treating six seats as independent would narrow this interval.
        let blocks = vec![block(0, pairs(0.0)), block(1, pairs(1.0))];
        let result = summarize(&blocks, 11235);
        assert_eq!(result.mean_utility_delta, Some(0.5));
        assert_eq!(result.utility_delta_bootstrap95, Some([0.0, 1.0]));
        assert_eq!(
            serde_json::to_value(&result).unwrap(),
            serde_json::to_value(summarize(&blocks, 11235)).unwrap()
        );
        assert_eq!(result.planned_games, 12);
        assert_eq!(result.completed_games, 12);
        assert_eq!(result.candidate_failed_games, 0);
        assert_eq!(result.reference_failed_games, 0);
        assert!(!result.strength_improvement_declared);
        assert_eq!(summarize(&blocks[..1], 0).utility_delta_bootstrap95, None);
        let mut broken = pairs(1.0);
        broken[0].candidate.error = Some("candidate failed".into());
        broken[0].candidate.winner_utility = None;
        broken[0].candidate.score = None;
        broken[0].candidate.rank = None;
        broken[1].candidate.error = Some("engine failed".into());
        broken[1].candidate.winner_utility = None;
        broken[1].candidate.score = None;
        broken[1].candidate.rank = None;
        broken[2].reference.error = Some("reference failed".into());
        broken[2].reference.winner_utility = None;
        broken[2].reference.score = None;
        broken[2].reference.rank = None;
        let failed = block(2, broken);
        assert!(!failed.complete);
        assert_eq!(failed.mean_utility_delta, None);
        let result = summarize(&[blocks[0].clone(), failed], 0);
        assert_eq!(
            (
                result.complete_blocks,
                result.incomplete_blocks,
                result.completed_games,
                result.failed_games
            ),
            (1, 1, 9, 3)
        );
        assert_eq!(result.candidate_failed_games, 2);
        assert_eq!(result.reference_failed_games, 1);
        assert_eq!(result.mean_utility_delta, Some(0.0));
        assert!(result.failures_preclude_adoption);
        assert!(!result.strength_improvement_declared);
        assert_eq!(summarize(&[], 0).mean_utility_delta, None);
    }

    #[test]
    fn cached_handles_preserve_distinct_weights_models_and_content_provenance() {
        let weights = HeuristicWeights {
            worker: 3.0,
            technology_step: 8.0,
            ..HeuristicWeights::default()
        };
        let prepared = [
            PreparedPolicy::Heuristic(weights.clone()),
            PreparedPolicy::Learned {
                model: Box::new(ModelArtifact::new(replay::catalog_hash(), 0).unwrap()),
                kernel: Kernel::Scalar,
                dataset_fingerprint: "a".repeat(64),
            },
            PreparedPolicy::Learned {
                model: Box::new(ModelArtifact::new(replay::catalog_hash(), 1).unwrap()),
                kernel: Kernel::Scalar,
                dataset_fingerprint: "b".repeat(64),
            },
        ];
        let handles = prepared
            .iter()
            .map(PolicyHandle::new)
            .collect::<Result<Vec<_>, _>>()
            .unwrap();
        let mut changed_weighted_decision = false;
        let mut changed_model_decision = false;
        for state in replay::benchmark_states(3, 0, GameOptions::default()).unwrap() {
            let observation =
                tzolkin_core::observation::observe(&state, state.current_player).unwrap();
            let weighted = handles[0].choose(&observation).unwrap();
            assert_eq!(
                weighted,
                choose_move_with_weights(&observation, &weights).unwrap()
            );
            changed_weighted_decision |=
                weighted.r#move != crate::choose_move(&observation).unwrap().r#move;
            for index in 1..=2 {
                let PreparedPolicy::Learned { model, .. } = &prepared[index] else {
                    unreachable!()
                };
                assert_eq!(
                    handles[index].choose(&observation).unwrap(),
                    model.choose_move(&observation).unwrap()
                );
                let SeatPolicy::Learned {
                    model_checksum,
                    inference_backend,
                    ..
                } = &handles[index].description.provenance
                else {
                    unreachable!()
                };
                assert_eq!(model_checksum, &model.checksum);
                assert_eq!(inference_backend.as_deref(), Some("scalar"));
            }
            changed_model_decision |= handles[1].choose(&observation).unwrap().r#move
                != handles[2].choose(&observation).unwrap().r#move;
        }
        assert!(changed_weighted_decision);
        assert!(changed_model_decision);
        assert_eq!(
            handles[1].description.dataset_fingerprint,
            Some("a".repeat(64))
        );
        assert_eq!(
            handles[2].description.dataset_fingerprint,
            Some("b".repeat(64))
        );
        let mut corrupt = ModelArtifact::new(replay::catalog_hash(), 0).unwrap();
        corrupt.checksum = "0".repeat(64);
        assert!(
            PolicyHandle::new(&PreparedPolicy::Learned {
                model: Box::new(corrupt),
                kernel: Kernel::Scalar,
                dataset_fingerprint: "a".repeat(64),
            })
            .is_err()
        );
    }
}
