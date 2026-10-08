//! Native Search diagnostics and controlled Observation-only selfplay.
//! Verification checks rules and trace/provenance consistency, not source authentication.
use crate::Decision;
use crate::replay::{self, GameReplay, ReplaySource, SeatPolicy};
use crate::search::{
    DecisionScoreKind, DiscardReason, FallbackReason, PreparedSearch, SEARCH_POLICY_VERSION,
    SearchConfig, SearchOutcome, SearchStats, SearchStatus, unsupported_reason,
};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use tzolkin_core::observation::Observation;
use tzolkin_core::{GameOptions, GameState};

/// Compact trace: no candidate, sampled card, world or private-state arrays.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct SearchTrace {
    pub policy_version: String,
    pub configuration_key: String,
    pub status: SearchStatus,
    pub score_kind: DecisionScoreKind,
    pub score: f64,
    pub stats: SearchStats,
}
impl SearchTrace {
    pub fn from_outcome(outcome: &SearchOutcome) -> Self {
        Self {
            policy_version: outcome.decision.policy_version.clone(),
            configuration_key: outcome.configuration_key.clone(),
            status: outcome.status.clone(),
            score_kind: outcome.score_kind,
            score: outcome.decision.score,
            stats: outcome.stats.clone(),
        }
    }
    pub(crate) fn validate(
        &self,
        observation: &Observation,
        config: &SearchConfig,
        key: &str,
    ) -> Result<(), String> {
        config.validate()?;
        let invalid = || Err("Invalid Search trace/configuration/phase consistency".into());
        if self.policy_version != SEARCH_POLICY_VERSION
            || self.configuration_key != key
            || !self.score.is_finite()
        {
            return invalid();
        }
        let unsupported = unsupported_reason(observation);
        if let Some(reason) = unsupported {
            return if self.status == (SearchStatus::Fallback { reason })
                && self.score_kind == DecisionScoreKind::HeuristicActionPriority
                && self.stats == SearchStats::default()
            {
                Ok(())
            } else {
                invalid()
            };
        }
        if observation.phase != tzolkin_core::Phase::Playing
            || observation.legal_actions.is_empty()
            || observation.legal_actions.len() > tzolkin_core::rollout::MAX_ROOT_ACTIONS
        {
            return invalid();
        }
        let s = &self.stats;
        if s.attempted_worlds == 0
            || s.atomic_steps == 0
            || s.attempted_worlds > config.worlds_per_action
            || s.completed_worlds > s.attempted_worlds
            || s.discarded_worlds.len() > config.worlds_per_action as usize
            || s.completed_worlds as usize + s.discarded_worlds.len() != s.attempted_worlds as usize
            || s.atomic_steps > config.max_total_steps
            || s.atomic_steps
                > config.max_rollout_steps
                    * observation.legal_actions.len()
                    * s.attempted_worlds as usize
            || s.max_reached_round
                .is_some_and(|round| !(observation.round..=27).contains(&round))
        {
            return invalid();
        }
        let mut previous = None;
        let mut successes = s.completed_worlds as usize * observation.legal_actions.len();
        for discarded in &s.discarded_worlds {
            if discarded.world_index >= s.attempted_worlds
                || previous.is_some_and(|index| index >= discarded.world_index)
                || discarded.failed_candidate >= observation.legal_actions.len()
                || (discarded.reason == DiscardReason::TotalStepCap
                    && (s.atomic_steps != config.max_total_steps
                        || discarded.world_index + 1 != s.attempted_worlds))
                || (discarded.reason == DiscardReason::RolloutStepCap
                    && s.atomic_steps < config.max_rollout_steps)
            {
                return invalid();
            }
            previous = Some(discarded.world_index);
            successes += discarded.failed_candidate;
        }
        if s.terminal_rollouts.checked_add(s.cutoff_rollouts) != Some(successes)
            || successes > s.atomic_steps
            || (successes > 0 && s.max_reached_round.is_none())
        {
            return invalid();
        }
        match self.status {
            SearchStatus::Searched
                if s.completed_worlds >= config.min_completed_worlds
                    && self.score_kind == DecisionScoreKind::RolloutMargin =>
            {
                Ok(())
            }
            SearchStatus::Fallback {
                reason: FallbackReason::InsufficientCompletedWorlds,
            } if s.completed_worlds < config.min_completed_worlds
                && self.score_kind == DecisionScoreKind::HeuristicActionPriority =>
            {
                Ok(())
            }
            _ => invalid(),
        }
    }
}

/// Attempted work survives a failed game. Zero means measured no work, not a missing result.
#[derive(Clone, Debug, Default, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct SearchSummary {
    pub decisions: u64,
    pub searched: u64,
    pub fallbacks: u64,
    pub attempted_worlds: u64,
    pub completed_worlds: u64,
    pub discarded_worlds: u64,
    pub atomic_steps: u64,
    pub terminal_rollouts: u64,
    pub cutoff_rollouts: u64,
    pub fallback_reasons: BTreeMap<FallbackReason, u64>,
    pub discard_reasons: BTreeMap<DiscardReason, u64>,
}
impl SearchSummary {
    pub(crate) fn add(&mut self, trace: &SearchTrace) {
        self.decisions += 1;
        match trace.status {
            SearchStatus::Searched => self.searched += 1,
            SearchStatus::Fallback { reason } => {
                self.fallbacks += 1;
                *self.fallback_reasons.entry(reason).or_default() += 1;
            }
        }
        self.attempted_worlds += u64::from(trace.stats.attempted_worlds);
        self.completed_worlds += u64::from(trace.stats.completed_worlds);
        self.discarded_worlds += trace.stats.discarded_worlds.len() as u64;
        self.atomic_steps += trace.stats.atomic_steps as u64;
        self.terminal_rollouts += trace.stats.terminal_rollouts as u64;
        self.cutoff_rollouts += trace.stats.cutoff_rollouts as u64;
        for discarded in &trace.stats.discarded_worlds {
            *self.discard_reasons.entry(discarded.reason).or_default() += 1;
        }
    }
    pub(crate) fn merge(&mut self, other: &Self) {
        self.decisions += other.decisions;
        self.searched += other.searched;
        self.fallbacks += other.fallbacks;
        self.attempted_worlds += other.attempted_worlds;
        self.completed_worlds += other.completed_worlds;
        self.discarded_worlds += other.discarded_worlds;
        self.atomic_steps += other.atomic_steps;
        self.terminal_rollouts += other.terminal_rollouts;
        self.cutoff_rollouts += other.cutoff_rollouts;
        for (reason, count) in &other.fallback_reasons {
            *self.fallback_reasons.entry(*reason).or_default() += count;
        }
        for (reason, count) in &other.discard_reasons {
            *self.discard_reasons.entry(*reason).or_default() += count;
        }
    }
}

pub fn provenance(policy: &PreparedSearch) -> SeatPolicy {
    SeatPolicy::Search {
        policy_version: SEARCH_POLICY_VERSION.into(),
        config: policy.config().clone(),
        configuration_key: policy.configuration_key().into(),
    }
}
pub fn decide(
    policy: &PreparedSearch,
    observation: &Observation,
) -> Result<(Decision, Option<SearchTrace>), String> {
    let outcome = policy.choose(observation)?;
    let trace = SearchTrace::from_outcome(&outcome);
    Ok((outcome.decision, Some(trace)))
}

pub struct SearchGameResult {
    pub game: Result<(GameState, usize, Option<GameReplay>), String>,
    /// Actual absolute seats; None is a non-Search seat. Simulated heuristic
    /// decisions count inside the root actor's atomic_steps, not as new summaries.
    pub search: Vec<Option<SearchSummary>>,
}
pub fn play_game(
    players: usize,
    seed: u32,
    options: GameOptions,
    search_seats: &[usize],
    record: bool,
    fast: bool,
    policy: &PreparedSearch,
) -> Result<SearchGameResult, String> {
    if !(3..=4).contains(&players)
        || options != GameOptions::default()
        || search_seats.is_empty()
        || search_seats.iter().any(|seat| *seat >= players)
        || search_seats
            .iter()
            .enumerate()
            .any(|(i, seat)| search_seats[..i].contains(seat))
    {
        return Err("Native Search requires base 3..4p and unique selected seats".into());
    }
    let policies = (0..players)
        .map(|seat| {
            if search_seats.contains(&seat) {
                provenance(policy)
            } else {
                crate::experiment::heuristic_seat()
            }
        })
        .collect();
    let mut summary = (0..players)
        .map(|seat| search_seats.contains(&seat).then(SearchSummary::default))
        .collect::<Vec<_>>();
    let game = replay::play_game_using_diagnostics(
        players,
        seed,
        options,
        record,
        ReplaySource::PolicySelfPlay { policies },
        fast,
        |observation| {
            if search_seats.contains(&observation.actor) {
                let (decision, trace) = decide(policy, observation)?;
                summary[observation.actor]
                    .as_mut()
                    .unwrap()
                    .add(trace.as_ref().unwrap());
                Ok((decision, trace))
            } else {
                Ok((crate::choose_move(observation)?, None))
            }
        },
    );
    Ok(SearchGameResult {
        game,
        search: summary,
    })
}
