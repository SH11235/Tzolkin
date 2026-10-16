//! Bounded public-information root Monte Carlo search with heuristic rollouts.
//! Each candidate uses the same sampled worlds; only whole completed blocks
//! contribute scores. This is an uncalibrated score-margin planner, not MCTS,
//! an omniscient policy, or a demonstrated strength improvement.
use crate::{
    Decision,
    policy::{HeuristicRevision, HeuristicWeights},
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use tzolkin_core::catalog::CATALOG;
use tzolkin_core::observation::Observation;
use tzolkin_core::rollout::{RolloutRoot, RolloutWorld, SAMPLING_VERSION};
use tzolkin_core::{GameMove, Phase, TEMPLE_IDS};

pub const SEARCH_POLICY_VERSION: &str = "public-root-search-v1";
pub const LEAF_VERSION: &str = "public-score-potential-v1";

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct SearchConfig {
    pub worlds_per_action: u32,
    pub horizon_days: i64,
    pub max_rollout_steps: usize,
    pub max_total_steps: usize,
    pub min_completed_worlds: u32,
    /// Public, reproducible search salt. Never populated from a game seed.
    pub sampling_salt: u32,
}
impl Default for SearchConfig {
    fn default() -> Self {
        Self {
            worlds_per_action: 4,
            horizon_days: 6,
            max_rollout_steps: 256,
            max_total_steps: 16_384,
            min_completed_worlds: 2,
            sampling_salt: 0,
        }
    }
}
impl SearchConfig {
    pub fn validate(&self) -> Result<(), String> {
        if !(1..=32).contains(&self.worlds_per_action)
            || !(1..=27).contains(&self.horizon_days)
            || !(1..=2048).contains(&self.max_rollout_steps)
            || !(1..=262_144).contains(&self.max_total_steps)
            || self.min_completed_worlds == 0
            || self.min_completed_worlds > self.worlds_per_action
        {
            return Err("Invalid bounded search configuration".into());
        }
        Ok(())
    }
    /// Freeze every behavioral input, including catalog, fixed rollout weights,
    /// objective, leaf/sampling versions and budgets. No authority metadata is used.
    pub fn configuration_key(&self) -> Result<String, String> {
        self.configuration_key_for_revision(HeuristicRevision::V2)
    }
    pub(crate) fn configuration_key_for_revision(
        &self,
        revision: HeuristicRevision,
    ) -> Result<String, String> {
        self.validate()?;
        let bytes = serde_json::to_vec(&(
            SEARCH_POLICY_VERSION,
            Objective::ScoreMargin,
            crate::replay::RULES_VERSION,
            crate::replay::RULES_BASELINE,
            LEAF_VERSION,
            SAMPLING_VERSION,
            revision.policy_version(),
            HeuristicWeights::default(),
            crate::replay::catalog_hash(),
            self,
        ))
        .map_err(|e| e.to_string())?;
        Ok(format!("{:x}", Sha256::digest(bytes)))
    }
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub enum Objective {
    ScoreMargin,
}
#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub enum DecisionScoreKind {
    RolloutMargin,
    HeuristicActionPriority,
}
#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(rename_all = "camelCase")]
pub enum FallbackReason {
    Setup,
    PendingContinuationUnavailable,
    UnsupportedPlayerCount,
    UnsupportedRules,
    InsufficientCompletedWorlds,
}
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(tag = "type", rename_all = "camelCase", deny_unknown_fields)]
pub enum SearchStatus {
    Searched,
    Fallback { reason: FallbackReason },
}
#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(rename_all = "camelCase")]
pub enum DiscardReason {
    RolloutStepCap,
    TotalStepCap,
    SimulationError,
    NonFiniteScore,
}
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct DiscardedWorld {
    pub world_index: u32,
    pub failed_candidate: usize,
    pub reason: DiscardReason,
}
#[derive(Clone, Debug, Default, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct SearchStats {
    pub attempted_worlds: u32,
    pub completed_worlds: u32,
    pub atomic_steps: usize,
    pub max_reached_round: Option<i64>,
    /// Successful attempted rollouts, including those in a subsequently discarded block.
    pub terminal_rollouts: usize,
    pub cutoff_rollouts: usize,
    pub discarded_worlds: Vec<DiscardedWorld>,
}
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct CandidateResult {
    pub legal_index: usize,
    pub visits: u32,
    pub mean_score: Option<f64>,
    pub minimum_score: Option<f64>,
    pub maximum_score: Option<f64>,
    pub terminal_samples: u32,
    pub cutoff_samples: u32,
    pub minimum_scored_round: Option<i64>,
    pub maximum_scored_round: Option<i64>,
}
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct SearchOutcome {
    pub decision: Decision,
    pub status: SearchStatus,
    pub objective: Objective,
    pub score_kind: DecisionScoreKind,
    pub configuration_key: String,
    pub stats: SearchStats,
    /// Indexes refer to the complete authoritative root legal array.
    pub candidates: Vec<CandidateResult>,
}

/// Terminal-only diagnostic ranking. `None` means keep the caller's original
/// policy; a heuristic fallback or partially completed block is never selected.
#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct TerminalCandidateRanking {
    pub selected_legal_index: Option<usize>,
    pub stats: SearchStats,
    pub candidates: Vec<CandidateResult>,
}

/// Compare every root move through the actual final scoring, using the existing
/// common-world rollout loop. Each candidate/world gets a fresh continuation
/// policy so game-local guards cannot leak between hypothetical branches.
///
/// This is an opt-in research API, not an application policy or trained model.
/// The caller supplies all seats' continuation policy and keeps its normal move
/// when ranking is unavailable. The root must be a late, non-pending decision;
/// no real hidden deck, game seed or prior opponent moves enter the search.
pub fn rank_terminal_candidates<Factory, Policy>(
    o: &Observation,
    config: &SearchConfig,
    mut make_policy: Factory,
) -> Result<TerminalCandidateRanking, String>
where
    Factory: FnMut() -> Result<Policy, String>,
    Policy: FnMut(&Observation) -> Result<Decision, String>,
{
    config.validate()?;
    if !(23..=27).contains(&o.round) || o.round + config.horizon_days < 27 {
        return Err("Terminal ranking requires day23+ and a horizon reaching day27".into());
    }
    let outcome = choose_prepared_with_for_revision(
        o,
        config,
        "terminal-candidate-ranking-v1",
        HeuristicRevision::V2,
        crate::choose_move,
        |world, first, actor, target, config, stats, _| {
            let mut policy = make_policy().map_err(|_| DiscardReason::SimulationError)?;
            simulate(world, first, actor, target, config, stats, &mut policy)
        },
    )?;
    // Require every requested world, not merely the historical search's minimum.
    // Failed or budget-truncated comparisons retain the caller's normal move.
    let complete = outcome.status == SearchStatus::Searched
        && outcome.stats.completed_worlds == config.worlds_per_action
        && outcome.stats.discarded_worlds.is_empty()
        && outcome.stats.cutoff_rollouts == 0
        && outcome.candidates.iter().all(|candidate| {
            candidate.terminal_samples == config.worlds_per_action && candidate.cutoff_samples == 0
        });
    let selected_legal_index = complete.then(|| {
        o.legal_actions
            .iter()
            .position(|action| action.r#move == outcome.decision.r#move)
            .expect("searched decision belongs to the root legal set")
    });
    Ok(TerminalCandidateRanking {
        selected_legal_index,
        stats: outcome.stats,
        candidates: outcome.candidates,
    })
}

pub fn choose_move(o: &Observation, config: &SearchConfig) -> Result<SearchOutcome, String> {
    PreparedSearch::new(config)?.choose(o)
}

/// Immutable native handle: validate and freeze the public configuration once.
#[derive(Clone)]
pub struct PreparedSearch {
    config: SearchConfig,
    configuration_key: String,
    revision: HeuristicRevision,
}
impl PreparedSearch {
    pub fn new(config: &SearchConfig) -> Result<Self, String> {
        Self::for_revision(config, HeuristicRevision::V2)
    }
    pub(crate) fn for_revision(
        config: &SearchConfig,
        revision: HeuristicRevision,
    ) -> Result<Self, String> {
        Ok(Self {
            configuration_key: config.configuration_key_for_revision(revision)?,
            config: config.clone(),
            revision,
        })
    }
    pub fn config(&self) -> &SearchConfig {
        &self.config
    }
    pub fn configuration_key(&self) -> &str {
        &self.configuration_key
    }
    pub fn choose(&self, o: &Observation) -> Result<SearchOutcome, String> {
        match self.revision {
            HeuristicRevision::V2 => {
                choose_prepared(o, &self.config, &self.configuration_key, crate::choose_move)
            }
            HeuristicRevision::V1 => choose_prepared_for_revision(
                o,
                &self.config,
                &self.configuration_key,
                self.revision,
                |o| {
                    tzolkin_bot::choose_move_with_weights_for_revision(
                        o,
                        &HeuristicWeights::default(),
                        HeuristicRevision::V1,
                    )
                },
            ),
        }
    }
}

pub(crate) fn unsupported_reason(o: &Observation) -> Option<FallbackReason> {
    if !(3..=4).contains(&o.players.len()) {
        Some(FallbackReason::UnsupportedPlayerCount)
    } else if o.additional_buildings || o.expansion.is_some() {
        Some(FallbackReason::UnsupportedRules)
    } else if o.phase == Phase::Setup {
        Some(FallbackReason::Setup)
    } else if o.pending_task.is_some() {
        Some(FallbackReason::PendingContinuationUnavailable)
    } else {
        None
    }
}

#[cfg(test)]
fn choose_using<F>(
    o: &Observation,
    config: &SearchConfig,
    rollout_policy: F,
) -> Result<SearchOutcome, String>
where
    F: FnMut(&Observation) -> Result<Decision, String>,
{
    let configuration_key = config.configuration_key()?;
    choose_prepared(o, config, &configuration_key, rollout_policy)
}
fn choose_prepared<F>(
    o: &Observation,
    config: &SearchConfig,
    configuration_key: &str,
    rollout_policy: F,
) -> Result<SearchOutcome, String>
where
    F: FnMut(&Observation) -> Result<Decision, String>,
{
    choose_prepared_for_revision(
        o,
        config,
        configuration_key,
        HeuristicRevision::V2,
        rollout_policy,
    )
}
fn choose_prepared_for_revision<F>(
    o: &Observation,
    config: &SearchConfig,
    configuration_key: &str,
    revision: HeuristicRevision,
    rollout_policy: F,
) -> Result<SearchOutcome, String>
where
    F: FnMut(&Observation) -> Result<Decision, String>,
{
    choose_prepared_with_for_revision(
        o,
        config,
        configuration_key,
        revision,
        rollout_policy,
        simulate::<F>,
    )
}
// Static function-item dispatch in production; the historical loop is injected
// only by unit tests. The closed revision changes the heuristic delegate only.
#[cfg(test)]
fn choose_prepared_with<F, S>(
    o: &Observation,
    config: &SearchConfig,
    configuration_key: &str,
    rollout_policy: F,
    simulation: S,
) -> Result<SearchOutcome, String>
where
    F: FnMut(&Observation) -> Result<Decision, String>,
    S: FnMut(
        RolloutWorld,
        GameMove,
        usize,
        i64,
        &SearchConfig,
        &mut SearchStats,
        &mut F,
    ) -> Result<Sample, DiscardReason>,
{
    choose_prepared_with_for_revision(
        o,
        config,
        configuration_key,
        HeuristicRevision::V2,
        rollout_policy,
        simulation,
    )
}
fn choose_prepared_with_for_revision<F, S>(
    o: &Observation,
    config: &SearchConfig,
    configuration_key: &str,
    revision: HeuristicRevision,
    mut rollout_policy: F,
    mut simulation: S,
) -> Result<SearchOutcome, String>
where
    F: FnMut(&Observation) -> Result<Decision, String>,
    S: FnMut(
        RolloutWorld,
        GameMove,
        usize,
        i64,
        &SearchConfig,
        &mut SearchStats,
        &mut F,
    ) -> Result<Sample, DiscardReason>,
{
    let configuration_key = configuration_key.to_owned();
    let unsupported = unsupported_reason(o);
    if let Some(reason) = unsupported {
        // Fallback consumes only the supplied observation, too. Feature shape
        // checks protect the fixed heuristic's indexing on malformed inputs.
        crate::features::FeatureEncoder::new(o)?;
        let decision = fallback_decision(o, revision)?;
        return Ok(SearchOutcome {
            decision,
            status: SearchStatus::Fallback { reason },
            objective: Objective::ScoreMargin,
            score_kind: DecisionScoreKind::HeuristicActionPriority,
            configuration_key,
            stats: SearchStats::default(),
            candidates: vec![],
        });
    }
    let root = RolloutRoot::from_observation(o)?;
    let baseline = fallback_decision(o, revision)?;
    let mut stats = SearchStats::default();
    let mut candidates: Vec<_> = (0..o.legal_actions.len())
        .map(|legal_index| CandidateResult {
            legal_index,
            visits: 0,
            mean_score: None,
            minimum_score: None,
            maximum_score: None,
            terminal_samples: 0,
            cutoff_samples: 0,
            minimum_scored_round: None,
            maximum_scored_round: None,
        })
        .collect();
    let mut sums = vec![0.0; candidates.len()];
    for world_index in 0..config.worlds_per_action {
        if stats.atomic_steps >= config.max_total_steps {
            break;
        }
        stats.attempted_worlds += 1;
        let common = root.sample(world_index, config.sampling_salt);
        let mut block = Vec::with_capacity(candidates.len());
        let mut failed = None;
        for (index, action) in o.legal_actions.iter().enumerate() {
            match simulation(
                common.clone(),
                action.r#move.clone(),
                o.actor,
                o.round + config.horizon_days,
                config,
                &mut stats,
                &mut rollout_policy,
            ) {
                Ok(sample) => {
                    if sample.terminal {
                        stats.terminal_rollouts += 1;
                    } else {
                        stats.cutoff_rollouts += 1;
                    }
                    block.push(sample);
                }
                Err(reason) => {
                    failed = Some((index, reason));
                    break;
                }
            }
        }
        if let Some((failed_candidate, reason)) = failed {
            stats.discarded_worlds.push(DiscardedWorld {
                world_index,
                failed_candidate,
                reason,
            });
            if reason == DiscardReason::TotalStepCap {
                break;
            }
            continue;
        }
        stats.completed_worlds += 1;
        for ((candidate, sum), sample) in candidates.iter_mut().zip(&mut sums).zip(block) {
            candidate.visits += 1;
            *sum += sample.score;
            candidate.mean_score = Some(*sum / f64::from(candidate.visits));
            candidate.minimum_score = Some(
                candidate
                    .minimum_score
                    .map_or(sample.score, |v| v.min(sample.score)),
            );
            candidate.maximum_score = Some(
                candidate
                    .maximum_score
                    .map_or(sample.score, |v| v.max(sample.score)),
            );
            candidate.minimum_scored_round = Some(
                candidate
                    .minimum_scored_round
                    .map_or(sample.round, |v| v.min(sample.round)),
            );
            candidate.maximum_scored_round = Some(
                candidate
                    .maximum_scored_round
                    .map_or(sample.round, |v| v.max(sample.round)),
            );
            if sample.terminal {
                candidate.terminal_samples += 1;
            } else {
                candidate.cutoff_samples += 1;
            }
        }
    }
    if stats.completed_worlds < config.min_completed_worlds {
        return Ok(SearchOutcome {
            decision: baseline,
            status: SearchStatus::Fallback {
                reason: FallbackReason::InsufficientCompletedWorlds,
            },
            objective: Objective::ScoreMargin,
            score_kind: DecisionScoreKind::HeuristicActionPriority,
            configuration_key,
            stats,
            candidates,
        });
    }
    let mut best = 0;
    for index in 1..candidates.len() {
        if candidates[index].mean_score.unwrap() > candidates[best].mean_score.unwrap() {
            best = index;
        }
    }
    let decision = Decision {
        actor: o.actor,
        observation_key: o.observation_key.clone(),
        policy_version: SEARCH_POLICY_VERSION.into(),
        r#move: o.legal_actions[best].r#move.clone(),
        score: candidates[best].mean_score.unwrap(),
    };
    Ok(SearchOutcome {
        decision,
        status: SearchStatus::Searched,
        objective: Objective::ScoreMargin,
        score_kind: DecisionScoreKind::RolloutMargin,
        configuration_key,
        stats,
        candidates,
    })
}

fn fallback_decision(o: &Observation, revision: HeuristicRevision) -> Result<Decision, String> {
    let mut decision = match revision {
        HeuristicRevision::V2 => crate::choose_move(o)?,
        HeuristicRevision::V1 => tzolkin_bot::choose_move_with_weights_for_revision(
            o,
            &HeuristicWeights::default(),
            revision,
        )?,
    };
    decision.policy_version = SEARCH_POLICY_VERSION.into();
    Ok(decision)
}
struct Sample {
    score: f64,
    terminal: bool,
    round: i64,
}
fn simulate<F>(
    mut world: RolloutWorld,
    first: GameMove,
    actor: usize,
    target_round: i64,
    config: &SearchConfig,
    stats: &mut SearchStats,
    policy: &mut F,
) -> Result<Sample, DiscardReason>
where
    F: FnMut(&Observation) -> Result<Decision, String>,
{
    let mut local_steps = 0;
    count_attempt(stats, &mut local_steps, config)?;
    // The initial candidate still uses the existing opaque world's checked API.
    world
        .apply(first)
        .map_err(|_| DiscardReason::SimulationError)?;
    loop {
        stats.max_reached_round = Some(
            stats
                .max_reached_round
                .map_or(world.round(), |v| v.max(world.round())),
        );
        if world.finished() {
            return Ok(Sample {
                score: margin(&world.score_projection(), actor)?,
                terminal: true,
                round: world.round(),
            });
        }
        let guard = world
            .decision()
            .map_err(|_| DiscardReason::SimulationError)?;
        let observation = guard.observation();
        // At a horizon reaching the final day, resolve the actual terminal
        // score rather than guessing which seats have a final action left.
        if target_round < 27 && guard.settled_for(actor, target_round) {
            let scores = potential_scores(observation, guard.score_projection())?;
            return Ok(Sample {
                score: margin(&scores, actor)?,
                terminal: false,
                round: observation.round,
            });
        }
        let decision = policy(observation).map_err(|_| DiscardReason::SimulationError)?;
        if decision.actor != observation.actor
            || decision.observation_key != observation.observation_key
            || !decision.score.is_finite()
            || !observation
                .legal_actions
                .iter()
                .any(|a| a.r#move == decision.r#move)
        {
            return Err(DiscardReason::SimulationError);
        }
        // Preserve legacy precedence: observe, compute and validate the next
        // policy decision even when the successful previous apply reached a cap.
        count_attempt(stats, &mut local_steps, config)?;
        guard
            .apply(decision.r#move)
            .map_err(|_| DiscardReason::SimulationError)?;
    }
}
fn count_attempt(
    stats: &mut SearchStats,
    local_steps: &mut usize,
    config: &SearchConfig,
) -> Result<(), DiscardReason> {
    if stats.atomic_steps >= config.max_total_steps {
        return Err(DiscardReason::TotalStepCap);
    }
    if *local_steps >= config.max_rollout_steps {
        return Err(DiscardReason::RolloutStepCap);
    }
    stats.atomic_steps += 1;
    *local_steps += 1;
    Ok(())
}
fn margin(scores: &[f64], actor: usize) -> Result<f64, DiscardReason> {
    if actor >= scores.len() || scores.len() < 2 || scores.iter().any(|v| !v.is_finite()) {
        return Err(DiscardReason::NonFiniteScore);
    }
    let opposition = scores
        .iter()
        .enumerate()
        .filter(|(id, _)| *id != actor)
        .map(|(_, score)| *score)
        .fold(f64::NEG_INFINITY, f64::max);
    let result = scores[actor] - opposition;
    if result.is_finite() {
        Ok(result)
    } else {
        Err(DiscardReason::NonFiniteScore)
    }
}

// Explicit fixed heuristic, in points. This is not a calibrated final score or
// win probability. Mechanical projection supplies current score/inventory/owned
// monuments; future investment weights fade to zero at day27. Future temple
// awards are projected at the current public levels/leadership, not predicted.
fn potential_scores(o: &Observation, mut scores: Vec<f64>) -> Result<Vec<f64>, DiscardReason> {
    let weights = HeuristicWeights::default();
    let future = (27 - o.round).max(0) as f64 / 26.0;
    for (id, player) in o.players.iter().enumerate() {
        let investment = (player.workers - 3) as f64 * weights.worker
            + player.technologies.iter().sum::<i64>() as f64 * weights.technology_step
            + (weights.corn_base - 0.25) * player.resources[0] as f64
            + (weights.material_values[0] - 0.5) * player.resources[1] as f64
            + (weights.material_values[1] - 0.75) * player.resources[2] as f64
            + (weights.material_values[2] - 1.0) * player.resources[3] as f64;
        scores[id] += future * investment;
        // Public maturity prior for workers still on gears. It is deliberately
        // small and does not pretend that an unclaimed action is already paid.
        scores[id] += o
            .gears
            .values()
            .flat_map(|slots| slots.iter().enumerate())
            .filter(|(_, worker)| {
                worker
                    .as_ref()
                    .is_some_and(|w| !w.dummy && w.player_id == id as i64)
            })
            .map(|(position, _)| position.min(7) as f64 * 0.5)
            .sum::<f64>();
        for (at, temple) in TEMPLE_IDS.iter().enumerate() {
            let track = &CATALOG.temple_tracks[temple];
            let highest = o.players.iter().map(|p| p.temples[at]).max().unwrap();
            let leaders = o
                .players
                .iter()
                .filter(|p| p.temples[at] == highest)
                .count();
            for day in [14, 27]
                .into_iter()
                .filter(|day| !o.food_days.contains(day))
            {
                let level = usize::try_from(player.temples[at] + 1)
                    .map_err(|_| DiscardReason::SimulationError)?;
                let points = *track
                    .points
                    .get(level)
                    .ok_or(DiscardReason::SimulationError)? as f64;
                let bonus = if day == 14 {
                    track.age1_bonus
                } else {
                    track.age2_bonus
                } as f64;
                scores[id] += points
                    + if player.temples[at] == highest {
                        bonus / if leaders > 1 { 2.0 } else { 1.0 }
                    } else {
                        0.0
                    };
            }
        }
    }
    if scores.iter().any(|v| !v.is_finite()) {
        return Err(DiscardReason::NonFiniteScore);
    }
    Ok(scores)
}

#[cfg(test)]
#[path = "search_decision_tests.rs"]
mod decision_cache_tests;

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;
    use tzolkin_core::{create_game, observation::observe};

    fn terminal_fixture() -> Observation {
        let mut state = create_game(
            vec!["A".into(), "B".into(), "C".into(), "D".into()],
            11235,
            false,
        )
        .unwrap();
        while state.phase == Phase::Setup {
            let o = observe(&state, state.current_player).unwrap();
            state =
                tzolkin_core::apply_move(&state, crate::choose_move(&o).unwrap().r#move).unwrap();
        }
        state.round = 27;
        state.age = 2;
        state.food_days = vec![8, 14, 21];
        state.buildings = state.age2_deck.drain(..6).collect();
        state.building_deck = std::mem::take(&mut state.age2_deck);
        observe(&state, state.current_player).unwrap()
    }

    #[test]
    fn terminal_ranking_completes_every_candidate_with_fresh_branch_policies() {
        let o = terminal_fixture();
        let config = SearchConfig {
            horizon_days: 27,
            ..SearchConfig::default()
        };
        let branches = std::cell::Cell::new(0);
        let ranked = rank_terminal_candidates(&o, &config, || {
            branches.set(branches.get() + 1);
            Ok(crate::choose_move)
        })
        .unwrap();
        assert!(ranked.selected_legal_index.is_some());
        assert_eq!(ranked.stats.cutoff_rollouts, 0);
        assert_eq!(
            branches.get(),
            o.legal_actions.len() * config.worlds_per_action as usize
        );
        assert!(
            ranked
                .candidates
                .iter()
                .all(|c| c.terminal_samples == config.worlds_per_action)
        );
    }

    #[test]
    fn terminal_ranking_never_selects_from_an_incomplete_budget() {
        let config = SearchConfig {
            horizon_days: 27,
            max_total_steps: 1,
            ..SearchConfig::default()
        };
        let ranked =
            rank_terminal_candidates(&terminal_fixture(), &config, || Ok(crate::choose_move))
                .unwrap();
        assert_eq!(ranked.selected_legal_index, None);
        assert_eq!(ranked.stats.atomic_steps, 1);
        assert_eq!(ranked.stats.completed_worlds, 0);
    }

    fn initial() -> Observation {
        let mut state =
            create_game(vec!["A".into(), "B".into(), "C".into()], 11235, false).unwrap();
        while state.phase == Phase::Setup {
            let o = observe(&state, state.current_player).unwrap();
            state =
                tzolkin_core::apply_move(&state, crate::choose_move(&o).unwrap().r#move).unwrap();
        }
        observe(&state, state.current_player).unwrap()
    }
    #[test]
    fn multiplayer_margin_uses_absolute_seats_without_sign_flips_or_probability_mix() {
        assert_eq!(margin(&[87.0, 120.5, 67.0, 95.5], 1), Ok(25.0));
        assert_eq!(margin(&[87.0, 120.5, 67.0, 95.5], 3), Ok(-25.0));
        assert_eq!(margin(&[10.0, 10.0, 4.0], 0), Ok(0.0));
        assert_eq!(
            margin(&[f64::NAN, 1.0], 0),
            Err(DiscardReason::NonFiniteScore)
        );
        let config = SearchConfig {
            worlds_per_action: 1,
            horizon_days: 1,
            max_rollout_steps: 1,
            max_total_steps: 1,
            min_completed_worlds: 1,
            sampling_salt: 0,
        };
        // Fixed historical producer key; no model, world or decision is run.
        let old = PreparedSearch::for_revision(&config, HeuristicRevision::V1).unwrap();
        assert_eq!(
            old.configuration_key(),
            "c40b96c933a0145678602a2820b6ca2200818424f3c80bea67f9b332708a42e3"
        );
        assert_ne!(old.configuration_key(), config.configuration_key().unwrap());
        let mut provenance = crate::search_native::provenance(&old);
        assert!(provenance.validate().is_err());
        provenance.validate_historical_restore().unwrap();
        let crate::replay::SeatPolicy::Search { config, .. } = &mut provenance else {
            unreachable!()
        };
        config.sampling_salt = 1;
        assert!(provenance.validate_historical_restore().is_err());
    }
    #[test]
    fn rollout_policy_gets_only_each_actors_observation_including_rotation_actor() {
        let o = initial();
        let mut actors = HashSet::new();
        let mut changed_actor = false;
        let config = SearchConfig {
            worlds_per_action: 1,
            min_completed_worlds: 1,
            horizon_days: 2,
            ..SearchConfig::default()
        };
        let outcome = choose_using(&o, &config, |observed| {
            actors.insert(observed.actor);
            changed_actor |= observed.actor != observed.turn_player;
            if observed.actor == o.actor {
                assert_eq!(observed.private.wealth_offer, o.private.wealth_offer);
            } else {
                assert!(observed.private.wealth_offer.is_empty());
            }
            crate::choose_move(observed)
        })
        .unwrap();
        assert_eq!(outcome.status, SearchStatus::Searched);
        assert_eq!(actors.len(), 3);
        assert!(
            changed_actor,
            "first-player branch must resolve rotation as its actual actor"
        );
    }
    #[test]
    fn stale_wrong_actor_illegal_and_nonfinite_policy_decisions_discard_worlds() {
        let o = initial();
        let config = SearchConfig {
            worlds_per_action: 1,
            min_completed_worlds: 1,
            horizon_days: 1,
            ..SearchConfig::default()
        };
        for mutation in 0..4 {
            let outcome = choose_using(&o, &config, |observed| {
                let mut decision = crate::choose_move(observed)?;
                match mutation {
                    0 => decision.actor = (decision.actor + 1) % observed.players.len(),
                    1 => decision.observation_key = "stale".into(),
                    2 => {
                        decision.r#move = GameMove::Remove {
                            gear: tzolkin_core::GearId::Tikal,
                            position: 99,
                        }
                    }
                    _ => decision.score = f64::NAN,
                }
                Ok(decision)
            })
            .unwrap();
            assert_eq!(outcome.stats.completed_worlds, 0);
            assert_eq!(
                outcome.stats.discarded_worlds[0].reason,
                DiscardReason::SimulationError
            );
            assert!(
                outcome
                    .candidates
                    .iter()
                    .all(|c| c.visits == 0 && c.mean_score.is_none())
            );
            assert!(matches!(outcome.status, SearchStatus::Fallback { .. }));
        }
    }
}
