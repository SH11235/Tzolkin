//! Experiment tooling around the game: learned policies, datasets, search, arenas and replays.
//!
//! The policy the application ships lives in `tzolkin-bot`; its public items are re-exported
//! here so existing `tzolkin_ai::` paths keep working.
pub mod arena;
pub mod dataset;
pub mod experiment;
pub mod features;
pub mod kernel;
pub mod model;
pub mod policy_dataset;
pub mod policy_training;
pub mod public_model;
pub mod public_native;
pub mod public_policy_cohort;
pub mod public_policy_episode;
pub mod public_policy_likelihood;
pub mod public_policy_pullback;
pub mod public_policy_update;
pub mod public_rl_artifact;
pub mod public_state_critic;
pub mod public_stochastic;
pub mod public_stochastic_native;
pub mod public_stochastic_record;
pub mod public_trade_guard;
pub mod replay;
pub mod search;
pub mod search_native;
pub mod selfplay_batch;
pub mod setup_policy;
pub mod state_mc_dataset;
pub mod state_mc_training;
pub mod training;
pub use tzolkin_bot::{
    Decision, POLICY_VERSION, choose_move, choose_move_with_weights, dispatch_cpu, policy,
};
