//! Versioned, validated replay and corpus tools. Training shards are a separate future layer.
use crate::public_trade_guard::{
    PublicLearnedSource, TradeGuardConfig, TradeGuardSession, TradeGuardTrace,
};
use crate::search::{SEARCH_POLICY_VERSION, SearchConfig};
use crate::search_native::SearchTrace;
use crate::{POLICY_VERSION, choose_move, policy::HeuristicWeights};
use serde::{Deserialize, Serialize};
use tzolkin_core::observation::{
    LegalAction, MOVE_SCHEMA, OBSERVATION_SCHEMA, Observation, fingerprint, observe,
};
use tzolkin_core::validation::validate_game_state;
use tzolkin_core::*;

pub const REPLAY_SCHEMA: u32 = 1;
pub const RULES_VERSION: u32 = 1;
pub const RULES_BASELINE: &str = "1ec8fdbb61f671ca2cbf0dbac7f3662c160be677";
pub const MAX_DECISIONS: usize = 4000;
pub const RL_ARGMAX_SELECTION_VERSION: &str = "public-rl-argmax-first-legal-tie-v1";
fn required_null<'de, D: serde::Deserializer<'de>>(deserializer: D) -> Result<(), D::Error> {
    if serde_json::Value::deserialize(deserializer)?.is_null() {
        Ok(())
    } else {
        Err(serde::de::Error::custom(
            "RL evaluation guard must be explicit null",
        ))
    }
}
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(
    tag = "kind",
    rename_all = "camelCase",
    rename_all_fields = "camelCase",
    deny_unknown_fields
)]
pub enum ReplaySource {
    SelfPlay {
        policy_version: String,
        weights: HeuristicWeights,
    },
    PolicySelfPlay {
        policies: Vec<SeatPolicy>,
    },
    Human {
        provider: String,
        reference: String,
        skill_rating: Option<f64>,
    },
}
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(
    tag = "kind",
    rename_all = "camelCase",
    rename_all_fields = "camelCase",
    deny_unknown_fields
)]
pub enum SeatPolicy {
    /// Evaluation provenance only. Metadata validation is not an RL owner constructor.
    PublicRl {
        policy_version: String,
        artifact_checksum: String,
        update_count: u64,
        task: String,
        feature_schema: u32,
        input_contract: String,
        numerical_target: String,
        inference_backend: String,
        selection_version: String,
        #[serde(deserialize_with = "required_null")]
        guard: (),
    },
    PublicLearnedTradeGuard {
        policy_version: String,
        base: PublicLearnedSource,
        guard: TradeGuardConfig,
        configuration_key: String,
    },
    PublicLearned {
        policy_version: String,
        model_version: String,
        training_version: String,
        feature_schema: u32,
        input_contract: String,
        task: String,
        value_validity: crate::public_model::ValueValidity,
        model_checksum: String,
        training_checkpoint_checksum: String,
        dataset_fingerprint: String,
        inference_backend: String,
    },
    Search {
        policy_version: String,
        config: SearchConfig,
        configuration_key: String,
    },
    Heuristic {
        policy_version: String,
        weights: HeuristicWeights,
    },
    CornFirstSetup {
        policy_version: String,
        weights: HeuristicWeights,
    },
    CornFirstUxmalOpening {
        policy_version: String,
        weights: HeuristicWeights,
    },
    Learned {
        policy_version: String,
        model_checksum: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        inference_backend: Option<String>,
    },
}

impl SeatPolicy {
    pub fn validate(&self) -> Result<(), String> {
        match self {
            Self::PublicRl {
                policy_version,
                artifact_checksum,
                update_count,
                task,
                feature_schema,
                input_contract,
                numerical_target,
                inference_backend,
                selection_version,
                guard: (),
            } if ((*update_count == 1
                && policy_version == crate::public_rl_native::POLICY_VERSION
                && task == crate::public_policy_update::TASK)
                || ((2..=crate::public_policy_repeat::MAX_UPDATE_COUNT)
                    .contains(update_count)
                    && policy_version == crate::public_policy_repeat::POLICY_VERSION
                    && task == crate::public_policy_repeat::TASK)
                || ((2..=crate::public_policy_long::MAX_UPDATES).contains(update_count)
                    && policy_version == crate::public_policy_long::POLICY_VERSION
                    && task == crate::public_policy_long::TASK))
                && *feature_schema == crate::features::PUBLIC_FEATURE_SCHEMA
                && input_contract == crate::public_model::INPUT_CONTRACT
                && numerical_target == &crate::public_stochastic_native::numerical_target()
                && inference_backend == "scalar"
                && selection_version == RL_ARGMAX_SELECTION_VERSION
                && artifact_checksum.len() == 64
                && artifact_checksum
                    .bytes()
                    .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b)) =>
            {
                Ok(())
            }
            Self::PublicLearnedTradeGuard {
                policy_version,
                base,
                guard,
                configuration_key,
            } => {
                if policy_version != crate::public_trade_guard::POLICY_VERSION
                    || *configuration_key != guard.configuration_key()?
                {
                    return Err("Unsupported guarded public learned provenance".into());
                }
                base.pure_policy().validate()
            }
            Self::PublicLearned {
                policy_version,
                model_version,
                training_version,
                feature_schema,
                input_contract,
                task,
                value_validity,
                model_checksum,
                training_checkpoint_checksum,
                dataset_fingerprint,
                inference_backend,
            } if policy_version == crate::public_model::POLICY_VERSION
                && model_version == crate::public_model::MODEL_VERSION
                && training_version == crate::policy_training::TRAINING_VERSION
                && *feature_schema == crate::features::PUBLIC_FEATURE_SCHEMA
                && input_contract == crate::public_model::INPUT_CONTRACT
                && task == "policyOnlyBc"
                && *value_validity == crate::public_model::ValueValidity::UnavailablePolicyOnly
                && ["scalar", "avx2", "sse2", "neon", "simd128"]
                    .contains(&inference_backend.as_str())
                && [
                    model_checksum,
                    training_checkpoint_checksum,
                    dataset_fingerprint,
                ]
                .iter()
                .all(|id| {
                    id.len() == 64
                        && id
                            .bytes()
                            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
                }) =>
            {
                Ok(())
            }
            Self::Search {
                policy_version,
                config,
                configuration_key,
            } if policy_version == SEARCH_POLICY_VERSION
                && config.configuration_key()? == *configuration_key =>
            {
                Ok(())
            }
            Self::Heuristic {
                policy_version,
                weights,
            } if policy_version == POLICY_VERSION => weights.validate(),
            Self::CornFirstSetup {
                policy_version,
                weights,
            } if policy_version == crate::setup_policy::POLICY_VERSION => weights.validate(),
            Self::CornFirstUxmalOpening {
                policy_version,
                weights,
            } if policy_version == crate::setup_policy::UXMAL_OPENING_POLICY_VERSION => {
                weights.validate()
            }
            Self::Learned {
                policy_version,
                model_checksum,
                inference_backend,
            } if policy_version == crate::model::LEARNED_POLICY_VERSION
                && inference_backend.as_ref().is_none_or(|backend| {
                    ["scalar", "avx2", "sse2", "neon", "simd128"].contains(&backend.as_str())
                })
                && model_checksum.len() == 64
                && model_checksum
                    .bytes()
                    .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte)) =>
            {
                Ok(())
            }
            _ => Err("Unsupported selfplay seat-policy provenance".into()),
        }
    }
}
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct ReplayHeader {
    pub replay_schema: u32,
    pub rules_version: u32,
    pub rules_baseline: String,
    pub catalog_hash: String,
    pub move_schema: u32,
    pub observation_schema: u32,
    pub source: ReplaySource,
    pub names: Vec<String>,
    /// Trusted runner metadata. Never passed into `choose_move`.
    pub seed: u32,
    pub options: GameOptions,
}
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct ReplayStep {
    pub index: usize,
    pub actor: usize,
    pub turn_player: usize,
    pub observation: Observation,
    pub chosen: LegalAction,
    pub state_before: String,
    pub state_after: String,
    pub validated: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub search: Option<SearchTrace>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub trade_guard: Option<TradeGuardTrace>,
}
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct GameReplay {
    pub header: ReplayHeader,
    pub steps: Vec<ReplayStep>,
    pub final_scores: Vec<FinalScore>,
    pub final_state: String,
    pub verified_complete: bool,
}
#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct RunSummary {
    pub games: usize,
    pub decisions: usize,
    pub illegal_operations: usize,
    pub unfinished_games: usize,
}
pub fn catalog_hash() -> String {
    format!(
        "{:016x}",
        fingerprint(include_bytes!("../../tzolkin-core/data/catalog.json"))
    )
}
pub fn state_key(state: &GameState) -> Result<String, String> {
    let mut value = serde_json::to_value(state).map_err(|e| e.to_string())?;
    value
        .as_object_mut()
        .ok_or("Expected game object")?
        .remove("log");
    Ok(format!(
        "{:016x}",
        fingerprint(&serde_json::to_vec(&value).map_err(|e| e.to_string())?)
    ))
}
pub fn options_from_mask(mask: u8) -> GameOptions {
    GameOptions {
        additional_buildings: mask & 1 != 0,
        tribes: mask & 2 != 0,
        prophecies: mask & 4 != 0,
        quick_actions: mask & 8 != 0,
    }
}
fn check_state(state: &GameState) -> Result<(), String> {
    let value = serde_json::to_value(state).map_err(|e| e.to_string())?;
    if !validate_game_state(&value) {
        return Err("Core returned an invalid saved game".into());
    }
    let restored: GameState = serde_json::from_value(value).map_err(|e| e.to_string())?;
    if *state != restored {
        return Err("Saved game roundtrip mismatch".into());
    }
    Ok(())
}
pub fn play_game(
    players: usize,
    seed: u32,
    options: GameOptions,
    record: bool,
) -> Result<(GameState, usize, Option<GameReplay>), String> {
    play_game_using(
        players,
        seed,
        options,
        record,
        ReplaySource::SelfPlay {
            policy_version: POLICY_VERSION.into(),
            weights: HeuristicWeights::default(),
        },
        choose_move,
    )
}
/// Controlled native runner. The policy receives only the current actor's redacted view.
/// Source describes every generating seat; verification validates legality, not policy quality.
pub fn play_game_using(
    players: usize,
    seed: u32,
    options: GameOptions,
    record: bool,
    source: ReplaySource,
    mut decide: impl FnMut(&Observation) -> Result<crate::Decision, String>,
) -> Result<(GameState, usize, Option<GameReplay>), String> {
    play_game_internal(
        players,
        seed,
        options,
        record,
        source,
        |o| Ok((decide(o)?, None, None)),
        true,
    )
}
/// Native games start from a validated seed/options and use checked legal operations.
/// Avoids per-transition saved-JSON roundtrips; terminal validation remains mandatory.
/// Published records must still pass the unchanged full independent replay verifier.
pub fn play_game_fast(
    players: usize,
    seed: u32,
    options: GameOptions,
    record: bool,
) -> Result<(GameState, usize, Option<GameReplay>), String> {
    play_game_using_fast(
        players,
        seed,
        options,
        record,
        ReplaySource::SelfPlay {
            policy_version: POLICY_VERSION.into(),
            weights: HeuristicWeights::default(),
        },
        choose_move,
    )
}
pub fn play_game_using_fast(
    players: usize,
    seed: u32,
    options: GameOptions,
    record: bool,
    source: ReplaySource,
    mut decide: impl FnMut(&Observation) -> Result<crate::Decision, String>,
) -> Result<(GameState, usize, Option<GameReplay>), String> {
    play_game_internal(
        players,
        seed,
        options,
        record,
        source,
        |o| Ok((decide(o)?, None, None)),
        false,
    )
}
/// Additive controlled runner. Search diagnostics never receive authority state.
pub fn play_game_using_diagnostics(
    players: usize,
    seed: u32,
    options: GameOptions,
    record: bool,
    source: ReplaySource,
    fast: bool,
    mut decide: impl FnMut(&Observation) -> Result<(crate::Decision, Option<SearchTrace>), String>,
) -> Result<(GameState, usize, Option<GameReplay>), String> {
    play_game_internal(
        players,
        seed,
        options,
        record,
        source,
        |o| {
            let (decision, search) = decide(o)?;
            Ok((decision, search, None))
        },
        !fast,
    )
}

/// Additive guarded runner. Callback indices include every seat and every phase.
pub fn play_game_using_trade_guard(
    players: usize,
    seed: u32,
    options: GameOptions,
    record: bool,
    source: ReplaySource,
    fast: bool,
    mut decide: impl FnMut(
        usize,
        &Observation,
    ) -> Result<(crate::Decision, Option<TradeGuardTrace>), String>,
) -> Result<(GameState, usize, Option<GameReplay>), String> {
    play_game_using_all_diagnostics(players, seed, options, record, source, fast, |index, o| {
        let (decision, guard) = decide(index, o)?;
        Ok((decision, None, guard))
    })
}

/// One global callback index for mixed Search/Guard rosters. The policy still
/// receives only the current actor's Observation; both traces use the existing
/// independent checks. Indices include Setup, pending tasks and every seat.
pub fn play_game_using_all_diagnostics(
    players: usize,
    seed: u32,
    options: GameOptions,
    record: bool,
    source: ReplaySource,
    fast: bool,
    mut decide: impl FnMut(
        usize,
        &Observation,
    ) -> Result<
        (
            crate::Decision,
            Option<SearchTrace>,
            Option<TradeGuardTrace>,
        ),
        String,
    >,
) -> Result<(GameState, usize, Option<GameReplay>), String> {
    let mut index = 0;
    play_game_internal(
        players,
        seed,
        options,
        record,
        source,
        |o| {
            let result = decide(index, o)?;
            index += 1;
            Ok(result)
        },
        !fast,
    )
}

struct GuardReplay {
    sessions: Vec<Option<TradeGuardSession>>,
}
impl GuardReplay {
    fn new(source: &ReplaySource) -> Result<Self, String> {
        let sessions = match source {
            ReplaySource::PolicySelfPlay { policies } => policies
                .iter()
                .map(|policy| match policy {
                    SeatPolicy::PublicLearnedTradeGuard { guard, .. } => {
                        TradeGuardSession::new(guard.clone()).map(Some)
                    }
                    _ => Ok(None),
                })
                .collect::<Result<Vec<_>, _>>()?,
            _ => Vec::new(),
        };
        Ok(Self { sessions })
    }
    fn check(
        &mut self,
        index: usize,
        o: &Observation,
        chosen: &LegalAction,
        trace: &Option<TradeGuardTrace>,
        decision: Option<&crate::Decision>,
    ) -> Result<(), String> {
        for session in self.sessions.iter_mut().flatten() {
            session.on_callback(index, o)?;
        }
        let session = self.sessions.get_mut(o.actor).and_then(Option::as_mut);
        if let Some(session) = session {
            if decision.is_some_and(|d| {
                d.policy_version != crate::public_trade_guard::POLICY_VERSION
                    || !d.score.is_finite()
            }) {
                return Err("Guarded decision policy version/finite score mismatch".into());
            }
            if crate::public_trade_guard::is_trade_observation(o) {
                let trace = trace
                    .as_ref()
                    .ok_or("Missing public Trade guard diagnostics")?;
                session.verify_trace(index, o, chosen, trace)?;
                if decision.is_some_and(|d| d.score != f64::from(trace.effective.logit)) {
                    return Err("Guarded decision/effective logit mismatch".into());
                }
                return Ok(());
            }
        }
        if trace.is_some() {
            return Err("Public Trade guard diagnostics on a non-guarded Trade callback".into());
        }
        Ok(())
    }
}

fn validate_search_trace(
    source: &ReplaySource,
    observation: &Observation,
    trace: &Option<SearchTrace>,
) -> Result<(), String> {
    let policy = match source {
        ReplaySource::PolicySelfPlay { policies } => policies.get(observation.actor),
        _ => None,
    };
    match (policy, trace) {
        (
            Some(SeatPolicy::Search {
                config,
                configuration_key,
                ..
            }),
            Some(trace),
        ) => trace.validate(observation, config, configuration_key),
        (Some(SeatPolicy::Search { .. }), None) => Err("Missing Search seat diagnostics".into()),
        (_, Some(_)) => Err("Search diagnostics on a non-Search seat".into()),
        (_, None) => Ok(()),
    }
}
fn play_game_internal(
    players: usize,
    seed: u32,
    mut options: GameOptions,
    record: bool,
    source: ReplaySource,
    mut decide: impl FnMut(
        &Observation,
    ) -> Result<
        (
            crate::Decision,
            Option<SearchTrace>,
            Option<TradeGuardTrace>,
        ),
        String,
    >,
    check_every_step: bool,
) -> Result<(GameState, usize, Option<GameReplay>), String> {
    if !(2..=5).contains(&players) {
        return Err("Players must be 2..5".into());
    }
    if let ReplaySource::PolicySelfPlay { policies } = &source {
        if policies.len() != players {
            return Err("Selfplay policy/seat count mismatch".into());
        }
        for policy in policies {
            policy.validate()?;
        }
        if policies
            .iter()
            .any(|policy| matches!(policy, SeatPolicy::Search { .. }))
            && (!(3..=4).contains(&players) || options != GameOptions::default())
        {
            return Err("Search provenance requires base 3..4p".into());
        }
        if policies
            .iter()
            .any(|policy| matches!(policy, SeatPolicy::PublicLearned { .. }))
            && (!(3..=4).contains(&players) || options != GameOptions::default())
        {
            return Err("Public learned provenance requires base 3..4p".into());
        }
        if policies
            .iter()
            .any(|policy| matches!(policy, SeatPolicy::PublicRl { .. }))
            && (!(3..=4).contains(&players) || options != GameOptions::default())
        {
            return Err("RL evaluation provenance requires base 3..4p".into());
        }
        if policies
            .iter()
            .any(|policy| matches!(policy, SeatPolicy::PublicLearnedTradeGuard { .. }))
            && (!(3..=4).contains(&players) || options != GameOptions::default())
        {
            return Err("Guarded public learned provenance requires base 3..4p".into());
        }
    }
    if players == 5 {
        options.quick_actions = true;
    }
    let names = (0..players)
        .map(|i| format!("CPU {}", i + 1))
        .collect::<Vec<_>>();
    let mut state = create_game_with_options(names.clone(), seed, options.clone())?;
    check_state(&state)?;
    let mut steps = Vec::new();
    let mut decisions = 0;
    let mut guard_replay = GuardReplay::new(&source)?;
    while state.phase != Phase::Finished {
        if decisions >= MAX_DECISIONS {
            return Err(format!(
                "Game did not finish: players={players}, seed={seed}, options={options:?}"
            ));
        }
        let observation = observe(&state, state.current_player)?;
        let (decision, search, trade_guard) = decide(&observation)?;
        if let ReplaySource::PolicySelfPlay { policies } = &source
            && let SeatPolicy::PublicRl { policy_version, .. } = &policies[observation.actor]
            && (decision.policy_version != *policy_version || !decision.score.is_finite())
        {
            return Err("RL evaluation decision/provenance mismatch".into());
        }
        validate_search_trace(&source, &observation, &search)?;
        if search.as_ref().is_some_and(|trace| {
            decision.policy_version != trace.policy_version || decision.score != trace.score
        }) {
            return Err("Search decision/diagnostics mismatch".into());
        }
        if decision.actor != observation.actor
            || decision.observation_key != observation.observation_key
        {
            return Err("Policy returned stale or wrong-actor decision".into());
        }
        let chosen = observation
            .legal_actions
            .iter()
            .find(|a| a.r#move == decision.r#move)
            .cloned()
            .ok_or("Policy returned a non-legal action")?;
        guard_replay.check(
            decisions,
            &observation,
            &chosen,
            &trade_guard,
            Some(&decision),
        )?;
        let before = if record {
            state_key(&state)?
        } else {
            String::new()
        };
        state = apply_move(&state, decision.r#move).map_err(|e| format!("Legal-operation failure at decision {decisions} (players={players},seed={seed}): {e}; {chosen:?}"))?;
        if check_every_step {
            check_state(&state)?;
        }
        if record {
            steps.push(ReplayStep {
                index: decisions,
                actor: observation.actor,
                turn_player: observation.turn_player,
                observation,
                chosen,
                state_before: before,
                state_after: state_key(&state)?,
                validated: true,
                search,
                trade_guard,
            });
        }
        decisions += 1;
    }
    check_state(&state)?;
    if state.final_scores.len() != players {
        return Err("Missing final score entries".into());
    }
    let replay = if record {
        Some(GameReplay {
            header: ReplayHeader {
                replay_schema: REPLAY_SCHEMA,
                rules_version: RULES_VERSION,
                rules_baseline: RULES_BASELINE.into(),
                catalog_hash: catalog_hash(),
                move_schema: MOVE_SCHEMA,
                observation_schema: OBSERVATION_SCHEMA,
                source,
                names,
                seed,
                options,
            },
            steps,
            final_scores: state.final_scores.clone(),
            final_state: state_key(&state)?,
            verified_complete: true,
        })
    } else {
        None
    };
    Ok((state, decisions, replay))
}
/// Recompute every actor, legal set, private view, transition and terminal result.
/// Human moves are checked exactly the same way; reproducing the heuristic is not required.
pub fn verify_replay(replay: &GameReplay) -> Result<GameState, String> {
    let h = &replay.header;
    if h.replay_schema != REPLAY_SCHEMA
        || h.rules_version != RULES_VERSION
        || h.rules_baseline != RULES_BASELINE
        || h.catalog_hash != catalog_hash()
        || h.move_schema != MOVE_SCHEMA
        || h.observation_schema != OBSERVATION_SCHEMA
    {
        return Err("Unsupported replay rules/catalog/schema".into());
    }
    if !replay.verified_complete {
        return Err("Partial logs cannot be verified as complete games".into());
    }
    if replay.steps.len() > MAX_DECISIONS {
        return Err("Replay exceeds decision limit".into());
    }
    if let ReplaySource::PolicySelfPlay { policies } = &h.source {
        if policies.len() != h.names.len() {
            return Err("Replay policy/seat count mismatch".into());
        }
        for policy in policies {
            policy.validate()?;
        }
        if policies
            .iter()
            .any(|policy| matches!(policy, SeatPolicy::Search { .. }))
            && (!(3..=4).contains(&h.names.len()) || h.options != GameOptions::default())
        {
            return Err("Search provenance requires base 3..4p".into());
        }
        if policies
            .iter()
            .any(|policy| matches!(policy, SeatPolicy::PublicLearned { .. }))
            && (!(3..=4).contains(&h.names.len()) || h.options != GameOptions::default())
        {
            return Err("Public learned provenance requires base 3..4p".into());
        }
        if policies
            .iter()
            .any(|policy| matches!(policy, SeatPolicy::PublicRl { .. }))
            && (!(3..=4).contains(&h.names.len()) || h.options != GameOptions::default())
        {
            return Err("RL evaluation provenance requires base 3..4p".into());
        }
        if policies
            .iter()
            .any(|policy| matches!(policy, SeatPolicy::PublicLearnedTradeGuard { .. }))
            && (!(3..=4).contains(&h.names.len()) || h.options != GameOptions::default())
        {
            return Err("Guarded public learned provenance requires base 3..4p".into());
        }
    }
    let mut state = create_game_with_options(h.names.clone(), h.seed, h.options.clone())?;
    check_state(&state)?;
    let mut guard_replay = GuardReplay::new(&h.source)?;
    for (index, step) in replay.steps.iter().enumerate() {
        if state.phase == Phase::Finished {
            return Err("Replay contains operations after the game ended".into());
        }
        let observation = observe(&state, state.current_player)?;
        validate_search_trace(&h.source, &observation, &step.search)?;
        if step.index != index
            || step.actor != observation.actor
            || step.turn_player != observation.turn_player
            || step.observation != observation
            || step.state_before != state_key(&state)?
            || !step.validated
            || !observation.legal_actions.contains(&step.chosen)
        {
            return Err(format!(
                "Replay observation/actor/legal/state mismatch at step {index}"
            ));
        }
        guard_replay.check(index, &observation, &step.chosen, &step.trade_guard, None)?;
        state = apply_move(&state, step.chosen.r#move.clone())?;
        check_state(&state)?;
        if step.state_after != state_key(&state)? {
            return Err(format!("Replay transition mismatch at step {index}"));
        }
    }
    if state.phase != Phase::Finished
        || replay.final_scores != state.final_scores
        || replay.final_state != state_key(&state)?
    {
        return Err("Replay terminal result mismatch".into());
    }
    Ok(state)
}
/// Independently verify, then publish a new bounded replay without replacing files.
pub fn save_replay_new(path: &std::path::Path, replay: &GameReplay) -> Result<(), String> {
    verify_replay(replay)?;
    crate::model::write_new_json_bounded(path, replay, crate::dataset::MAX_REPLAY_BYTES as usize)
}
/// All 56 effective configurations, including the forced quick action setting for five seats.
pub fn verify_corpus(seeds: u32) -> Result<RunSummary, String> {
    if seeds == 0 {
        return Err("At least one seed is required".into());
    }
    let mut summary = RunSummary {
        games: 0,
        decisions: 0,
        illegal_operations: 0,
        unfinished_games: 0,
    };
    for players in 2..=5 {
        let masks = if players == 5 { 8..16 } else { 0..16 };
        for mask in masks {
            for seed in 0..seeds {
                let (_, decisions, _) = play_game(players, seed, options_from_mask(mask), false)?;
                summary.games += 1;
                summary.decisions += decisions;
            }
        }
    }
    Ok(summary)
}
/// Reachable states used as the baseline corpus for legal generation, transitions and policy.
pub fn benchmark_states(
    players: usize,
    seed: u32,
    options: GameOptions,
) -> Result<Vec<GameState>, String> {
    let mut state = create_game_with_options(
        (0..players).map(|i| format!("CPU {}", i + 1)).collect(),
        seed,
        options,
    )?;
    let mut samples = Vec::new();
    for index in 0..MAX_DECISIONS {
        if state.phase == Phase::Finished {
            break;
        }
        if index % 7 == 0
            || state.current_player != state.turn_order[state.turn_index]
            || matches!(
                state.pending.as_ref().map(|p| &p.task),
                Some(Task::Build { .. } | Task::ProphecyGain { .. } | Task::Rotation | Task::Trade)
            )
        {
            samples.push(state.clone());
        }
        let observation = observe(&state, state.current_player)?;
        state = apply_move(&state, choose_move(&observation)?.r#move)?;
    }
    if state.phase != Phase::Finished {
        return Err("Benchmark corpus did not finish".into());
    }
    Ok(samples)
}
