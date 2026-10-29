//! Deterministic, untrained policy operating exclusively on an actor observation.
pub mod policy;
pub mod setup_policy;
use serde::{Deserialize, Serialize};
use tzolkin_core::GameMove;
use tzolkin_core::observation::{MOVE_SCHEMA, OBSERVATION_SCHEMA, Observation, observation_key};

pub const POLICY_VERSION: &str = "heuristic-v2";
pub const FOOD_ALIGNED_POLICY_VERSION: &str = "heuristic-food-aligned-v1";
pub const FOOD_ALIGNED_V2_POLICY_VERSION: &str = "heuristic-food-aligned-v2";
pub const MONUMENT_ALIGNED_POLICY_VERSION: &str = "heuristic-monument-aligned-v1";
pub const MONUMENT_ALIGNED_V2_POLICY_VERSION: &str = "heuristic-monument-aligned-v2";
pub const RESOURCE_ALIGNED_POLICY_VERSION: &str = "heuristic-resource-aligned-v1";
pub const RESOURCE_ALIGNED_V2_POLICY_VERSION: &str = "heuristic-resource-aligned-v2";
pub const RESOURCE_ALIGNED_V3_POLICY_VERSION: &str = "heuristic-resource-aligned-v3";
pub const TIMED_CORN_POLICY_VERSION: &str = "heuristic-timed-corn-v1";
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct Decision {
    pub actor: usize,
    pub observation_key: String,
    pub policy_version: String,
    pub r#move: GameMove,
    pub score: f64,
}
pub fn choose_move(observation: &Observation) -> Result<Decision, String> {
    choose_move_with_weights(observation, &policy::HeuristicWeights::default())
}
/// Experimental coefficients are explicit; the UI/default CPU keeps its frozen weights.
pub fn choose_move_with_weights(
    observation: &Observation,
    weights: &policy::HeuristicWeights,
) -> Result<Decision, String> {
    choose_move_with_weights_for_revision(observation, weights, policy::HeuristicRevision::V2)
}
/// Historical selection executes its scoring revision, retaining validation,
/// finite scores and the first maximum in original legal order.
pub fn choose_move_with_weights_for_revision(
    observation: &Observation,
    weights: &policy::HeuristicWeights,
    revision: policy::HeuristicRevision,
) -> Result<Decision, String> {
    choose_scored(observation, weights, revision.policy_version(), |action| {
        policy::score_action_for_revision(observation, action, weights, revision)
    })
}

/// V1 food estimates follow the core's full-feeding cost. Historical
/// selectors keep their original versions, scoring and legal-order tie breaks.
pub fn choose_move_food_aligned(observation: &Observation) -> Result<Decision, String> {
    let weights = policy::HeuristicWeights::default();
    choose_scored(
        observation,
        &weights,
        FOOD_ALIGNED_POLICY_VERSION,
        |action| policy::score_action_food_aligned(observation, action, &weights),
    )
}

/// V2 estimates selected harvests and capped marginal food savings.
pub fn choose_move_food_aligned_v2(observation: &Observation) -> Result<Decision, String> {
    let weights = policy::HeuristicWeights::default();
    choose_scored(
        observation,
        &weights,
        FOOD_ALIGNED_V2_POLICY_VERSION,
        |action| policy::score_action_food_aligned_v2(observation, action, &weights),
    )
}

/// Current App also values current monument points and separately estimates
/// recurring effects lost to renovation, without forecasting future growth.
pub fn choose_move_monument_aligned(observation: &Observation) -> Result<Decision, String> {
    let weights = policy::HeuristicWeights::default();
    choose_scored(
        observation,
        &weights,
        MONUMENT_ALIGNED_POLICY_VERSION,
        |action| policy::score_action_monument_aligned(observation, action, &weights),
    )
}

/// V2 also counts exact untaxed rewards when previewing an owned Yaxchilan
/// worker at positions 1..5. Existing App and historical selectors are retained.
pub fn choose_move_monument_aligned_v2(observation: &Observation) -> Result<Decision, String> {
    let weights = policy::HeuristicWeights::default();
    choose_scored(
        observation,
        &weights,
        MONUMENT_ALIGNED_V2_POLICY_VERSION,
        |action| policy::score_action_monument_aligned_v2(observation, action, &weights),
    )
}

/// Explicit, unactivated resource-investment hypothesis; old choosers stay frozen.
pub fn choose_move_resource_aligned(observation: &Observation) -> Result<Decision, String> {
    let weights = policy::HeuristicWeights::default();
    choose_scored(
        observation,
        &weights,
        RESOURCE_ALIGNED_POLICY_VERSION,
        |action| policy::score_action_resource_aligned(observation, action, &weights),
    )
}

/// Current App resource routes use a shared bounded observation context.
pub fn choose_move_resource_aligned_v2(observation: &Observation) -> Result<Decision, String> {
    let weights = policy::HeuristicWeights::default();
    let context = std::cell::OnceCell::new();
    choose_scored(
        observation,
        &weights,
        RESOURCE_ALIGNED_V2_POLICY_VERSION,
        |action| {
            context
                .get_or_init(|| policy::ResourceRouteContext::new(observation, &weights))
                .score(observation, action, &weights)
        },
    )
}

/// Opt-in lifetime investment forecast; the current App continues to use V2.
pub fn choose_move_resource_aligned_v3(observation: &Observation) -> Result<Decision, String> {
    let weights = policy::HeuristicWeights::default();
    let context = std::cell::OnceCell::new();
    choose_scored(
        observation,
        &weights,
        RESOURCE_ALIGNED_V3_POLICY_VERSION,
        |action| {
            context
                .get_or_init(|| policy::ResourceRouteContext::new_v3(observation, &weights))
                .score(observation, action, &weights)
        },
    )
}

pub fn choose_move_timed_corn(observation: &Observation) -> Result<Decision, String> {
    let weights = policy::HeuristicWeights::default();
    let context = std::cell::OnceCell::new();
    choose_scored(observation, &weights, TIMED_CORN_POLICY_VERSION, |action| {
        context
            .get_or_init(|| policy::TimedCornContext::new(observation, &weights))
            .score(observation, action, &weights)
    })
}

fn choose_scored(
    observation: &Observation,
    weights: &policy::HeuristicWeights,
    policy_version: &str,
    score_action: impl Fn(&tzolkin_core::observation::TypedAction) -> f64,
) -> Result<Decision, String> {
    weights.validate()?;
    if observation.schema != OBSERVATION_SCHEMA || observation.move_schema != MOVE_SCHEMA {
        return Err("Unsupported observation or move schema".into());
    }
    if !(2..=5).contains(&observation.players.len())
        || observation.actor >= observation.players.len()
        || observation.turn_player >= observation.players.len()
        || !(1..=27).contains(&observation.round)
        || !(1..=2).contains(&observation.age)
        || observation
            .players
            .iter()
            .enumerate()
            .any(|(index, player)| player.id != index)
    {
        return Err("Invalid observation actor, players or day".into());
    }
    let key = observation_key(observation)?;
    if key != observation.observation_key {
        return Err("Observation key mismatch".into());
    }
    let mut best = None;
    for action in &observation.legal_actions {
        let score = score_action(&action.action);
        if !score.is_finite() {
            return Err("Non-finite policy score".into());
        }
        if best.as_ref().is_none_or(|(_, value)| score > *value) {
            best = Some((action, score));
        }
    }
    let (action, score) = best.ok_or("No legal action")?;
    Ok(Decision {
        actor: observation.actor,
        observation_key: key,
        policy_version: policy_version.into(),
        r#move: action.r#move.clone(),
        score,
    })
}
/// This endpoint accepts only the redacted observation, including the complete legal set.
pub fn dispatch_cpu(request: &str) -> Result<String, String> {
    dispatch_with(request, choose_move)
}

pub fn dispatch_cpu_food_aligned(request: &str) -> Result<String, String> {
    dispatch_with(request, choose_move_food_aligned)
}

pub fn dispatch_cpu_food_aligned_v2(request: &str) -> Result<String, String> {
    dispatch_with(request, choose_move_food_aligned_v2)
}

pub fn dispatch_cpu_monument_aligned(request: &str) -> Result<String, String> {
    dispatch_with(request, choose_move_monument_aligned)
}

pub fn dispatch_cpu_monument_aligned_v2(request: &str) -> Result<String, String> {
    dispatch_with(request, choose_move_monument_aligned_v2)
}

pub fn dispatch_cpu_resource_aligned_v2(request: &str) -> Result<String, String> {
    dispatch_with(request, choose_move_resource_aligned_v2)
}

pub fn dispatch_cpu_resource_aligned_v3(request: &str) -> Result<String, String> {
    dispatch_with(request, choose_move_resource_aligned_v3)
}

fn dispatch_with(
    request: &str,
    choose: fn(&Observation) -> Result<Decision, String>,
) -> Result<String, String> {
    let observation: Observation = serde_json::from_str(request).map_err(|e| e.to_string())?;
    serde_json::to_string(&choose(&observation)?).map_err(|e| e.to_string())
}
