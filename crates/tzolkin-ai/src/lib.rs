//! Deterministic, untrained policy operating exclusively on an actor observation.
pub mod arena;
pub mod dataset;
pub mod experiment;
pub mod features;
pub mod kernel;
pub mod model;
pub mod policy;
pub mod policy_dataset;
pub mod policy_training;
pub mod public_model;
pub mod public_native;
pub mod public_state_critic;
pub mod replay;
pub mod search;
pub mod search_native;
pub mod selfplay_batch;
pub mod state_mc_dataset;
pub mod state_mc_training;
pub mod training;
use serde::{Deserialize, Serialize};
use tzolkin_core::GameMove;
use tzolkin_core::observation::{MOVE_SCHEMA, OBSERVATION_SCHEMA, Observation, observation_key};

pub const POLICY_VERSION: &str = "heuristic-v1";
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
        let score = policy::score_action(observation, &action.action, weights);
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
        policy_version: POLICY_VERSION.into(),
        r#move: action.r#move.clone(),
        score,
    })
}
/// This endpoint accepts only the redacted observation, including the complete legal set.
pub fn dispatch_cpu(request: &str) -> Result<String, String> {
    let observation: Observation = serde_json::from_str(request).map_err(|e| e.to_string())?;
    serde_json::to_string(&choose_move(&observation)?).map_err(|e| e.to_string())
}
