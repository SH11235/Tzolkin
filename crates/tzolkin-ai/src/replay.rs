//! Versioned, validated replay and corpus tools. Training shards are a separate future layer.
use crate::public_trade_guard::{
    PublicLearnedSource, TradeGuardConfig, TradeGuardSession, TradeGuardTrace,
};
use crate::search::{SEARCH_POLICY_VERSION, SearchConfig};
use crate::search_native::SearchTrace;
use crate::{POLICY_VERSION, choose_move, policy::HeuristicWeights};
use serde::{Deserialize, Serialize};
use tzolkin_core::observation::{
    LegalAction, MOVE_SCHEMA, OBSERVATION_SCHEMA, Observation, TypedAction, fingerprint, observe,
};
use tzolkin_core::validation::validate_game_state;
use tzolkin_core::*;

pub const REPLAY_SCHEMA: u32 = 1;
pub const RULES_VERSION: u32 = 2;
pub const RULES_BASELINE: &str = "1ec8fdbb61f671ca2cbf0dbac7f3662c160be677";
pub const MAX_DECISIONS: usize = 4000;
pub const RL_ARGMAX_SELECTION_VERSION: &str = "public-rl-argmax-first-legal-tie-v1";
pub const CORN_FIRST_V5_TEACHER_POLICY_VERSION: &str =
    "corn-first-setup-resource-cycle-v5-playing-teacher-v1";
pub const CORN_FIRST_V7_TEACHER_POLICY_VERSION: &str =
    "corn-first-setup-resource-cycle-v7-playing-teacher-v1";
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
    /// One fixed teacher for every seat. This describes generation, not dataset,
    /// model, Human origin or source-authentication authority.
    CornFirstV5Teacher {
        policy_version: String,
        setup_policy_version: String,
        setup_weights: HeuristicWeights,
        playing_configuration: crate::arena::resource_cycle::ResourceCycleConfig,
    },
    /// Separate prospective V7 teacher; never a V5/Human source conversion.
    CornFirstV7Teacher {
        policy_version: String,
        setup_policy_version: String,
        setup_weights: HeuristicWeights,
        playing_configuration: crate::arena::resource_cycle::ResourceCycleConfig,
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
    /// Guarded current joint evaluation only; this DTO admits no owner or Train source.
    PublicRlCurrentMixedTradeGuard {
        policy_version: String,
        identity: Box<crate::public_rl_current_mixed_source::CurrentMixedParentIdentity>,
        selection_version: String,
        guard: TradeGuardConfig,
        configuration_key: String,
    },
    /// Count-neutral continued mixed evaluation metadata, never owner admission.
    PublicRlContinuedMixed {
        identity: Box<crate::public_rl_current_mixed_source::ContinuedMixedIdentity>,
        selection_version: String,
        #[serde(deserialize_with = "required_null")]
        guard: (),
    },
    /// Mixed joint owner evaluation only; identity DTOs mint no executable owner.
    PublicRlMixed {
        identity: crate::public_rl_mixed_handle::MixedInferenceIdentity,
        selection_version: String,
        #[serde(deserialize_with = "required_null")]
        guard: (),
    },
    /// Joint owner identity for deterministic actor-only development evaluation.
    /// This metadata is not a joint owner constructor or training source.
    PublicRlPaired {
        policy_version: String,
        artifact_checksum: String,
        update_count: u64,
        task: String,
        actor_model_version: String,
        critic_model_version: String,
        feature_schema: u32,
        input_contract: String,
        context_schema: u32,
        context_contract: String,
        root_init_checksum: String,
        root_count1_checksum: String,
        root_residual_checksum: String,
        residual_seed: u64,
        lineage_checksum: String,
        numerical_target: String,
        inference_backend: String,
        selection_version: String,
        #[serde(deserialize_with = "required_null")]
        guard: (),
    },
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
    ResourceCycle {
        configuration: crate::arena::resource_cycle::ResourceCycleConfig,
    },
    Learned {
        policy_version: String,
        model_checksum: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        inference_backend: Option<String>,
    },
}

impl SeatPolicy {
    /// Only saved-plan restoration may use this closed historical vocabulary.
    /// A recognized description is not an executable policy or owner admission.
    // Ordinary source validators deliberately keep calling `validate` below.
    pub(crate) fn validate_historical_restore(&self) -> Result<(), String> {
        match self {
            Self::Heuristic {
                policy_version,
                weights,
            } if policy_version == tzolkin_bot::policy::HeuristicRevision::V1.policy_version() => {
                weights.validate()
            }
            Self::CornFirstSetup {
                policy_version,
                weights,
            } if policy_version == crate::setup_policy::LEGACY_POLICY_VERSION => weights.validate(),
            Self::CornFirstUxmalOpening {
                policy_version,
                weights,
            } if policy_version == crate::setup_policy::LEGACY_UXMAL_OPENING_POLICY_VERSION => {
                weights.validate()
            }
            Self::Search {
                policy_version,
                config,
                configuration_key,
            } if policy_version == SEARCH_POLICY_VERSION
                && *configuration_key
                    == config.configuration_key_for_revision(
                        tzolkin_bot::policy::HeuristicRevision::V1,
                    )? =>
            {
                Ok(())
            }
            _ => self.validate(),
        }
    }
    pub(crate) fn is_rl_evaluation(&self) -> bool {
        self.rl_evaluation_policy_version().is_some()
    }
    pub(crate) fn rl_evaluation_policy_version(&self) -> Option<&str> {
        match self {
            Self::PublicRl { policy_version, .. } | Self::PublicRlPaired { policy_version, .. } => {
                Some(policy_version)
            }
            Self::PublicRlCurrentMixedTradeGuard { policy_version, .. } => Some(policy_version),
            Self::PublicRlMixed { identity, .. } => Some(identity.policy_version()),
            Self::PublicRlContinuedMixed { identity, .. } => Some(identity.policy_version()),
            _ => None,
        }
    }
    pub fn validate(&self) -> Result<(), String> {
        match self {
            Self::PublicRlCurrentMixedTradeGuard {
                policy_version,
                identity,
                selection_version,
                guard,
                configuration_key,
            } => {
                if policy_version != crate::public_trade_guard::CURRENT_MIXED_POLICY_VERSION
                    || selection_version != RL_ARGMAX_SELECTION_VERSION
                    || *configuration_key != guard.configuration_key()?
                {
                    return Err("Unsupported guarded current mixed provenance".into());
                }
                identity.validate()
            }
            Self::PublicRlContinuedMixed {
                identity,
                selection_version,
                guard: (),
            } => {
                if selection_version != RL_ARGMAX_SELECTION_VERSION {
                    return Err("Unsupported continued mixed evaluation selection version".into());
                }
                identity.validate()
            }
            Self::PublicRlMixed {
                identity,
                selection_version,
                guard: (),
            } => {
                if selection_version != RL_ARGMAX_SELECTION_VERSION {
                    return Err("Unsupported mixed evaluation selection version".into());
                }
                identity.validate()
            }
            Self::PublicRlPaired {
                policy_version,
                artifact_checksum,
                update_count,
                task,
                actor_model_version,
                critic_model_version,
                feature_schema,
                input_contract,
                context_schema,
                context_contract,
                root_init_checksum,
                root_count1_checksum,
                root_residual_checksum,
                residual_seed,
                lineage_checksum,
                numerical_target,
                inference_backend,
                selection_version,
                guard: (),
            } if (2..=crate::public_policy_paired_long::MAX_UPDATES).contains(update_count)
                && policy_version == crate::public_policy_paired_long::POLICY_VERSION
                && task == crate::public_policy_paired_long::TASK
                && actor_model_version == crate::public_model::MODEL_VERSION
                && critic_model_version == crate::public_state_critic::MODEL_VERSION
                && *feature_schema == crate::features::PUBLIC_FEATURE_SCHEMA
                && input_contract == crate::public_model::INPUT_CONTRACT
                && *context_schema == crate::public_state_critic::CONTEXT_SCHEMA
                && context_contract == crate::public_state_critic::CONTEXT_CONTRACT
                && [
                    artifact_checksum,
                    root_init_checksum,
                    root_count1_checksum,
                    root_residual_checksum,
                    lineage_checksum,
                ]
                .iter()
                .all(|s| crate::public_policy_update::checkpoint_digest(s))
                && *root_residual_checksum
                    == crate::public_policy_baseline::zero_checksum(
                        root_init_checksum,
                        *residual_seed,
                    )?
                && numerical_target == &crate::public_stochastic_native::numerical_target()
                && inference_backend == "scalar"
                && selection_version == RL_ARGMAX_SELECTION_VERSION =>
            {
                Ok(())
            }
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
            Self::ResourceCycle { configuration } => configuration.validate(),
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
        Some(source),
        |o| Ok((decide(o)?, None, None)),
        true,
    )
}

pub(crate) fn corn_first_v5_teacher_provenance() -> ReplaySource {
    ReplaySource::CornFirstV5Teacher {
        policy_version: CORN_FIRST_V5_TEACHER_POLICY_VERSION.into(),
        setup_policy_version: crate::setup_policy::POLICY_VERSION.into(),
        setup_weights: HeuristicWeights::default(),
        playing_configuration: crate::arena::resource_cycle::ResourceCycleConfig::v5(),
    }
}
fn require_corn_first_v5_teacher_source(
    source: &ReplaySource,
    players: usize,
    options: &GameOptions,
) -> Result<bool, String> {
    if !matches!(source, ReplaySource::CornFirstV5Teacher { .. }) {
        return Ok(false);
    }
    // Compare the entire closed declaration, including the effective floating
    // config bytes. No alternate version, weights, per-seat choice or options.
    if !(3..=4).contains(&players)
        || *options != GameOptions::default()
        || serde_json::to_vec(source).map_err(|e| e.to_string())?
            != serde_json::to_vec(&corn_first_v5_teacher_provenance()).map_err(|e| e.to_string())?
    {
        return Err("Corn-first V5 teacher requires fixed all-seat base 3/4p provenance".into());
    }
    Ok(true)
}
fn corn_first_v5_teacher_choice(observation: &Observation) -> Result<crate::Decision, String> {
    match observation.phase {
        Phase::Setup => {
            crate::setup_policy::choose_move_with_weights(observation, &HeuristicWeights::default())
        }
        Phase::Playing => crate::arena::resource_cycle::choose_move(
            observation,
            &crate::arena::resource_cycle::ResourceCycleConfig::v5(),
        ),
        _ => Err("Corn-first V5 teacher requires an actual Setup or Playing callback".into()),
    }
}
pub(crate) fn corn_first_v7_teacher_provenance() -> ReplaySource {
    ReplaySource::CornFirstV7Teacher {
        policy_version: CORN_FIRST_V7_TEACHER_POLICY_VERSION.into(),
        setup_policy_version: crate::setup_policy::POLICY_VERSION.into(),
        setup_weights: HeuristicWeights::default(),
        playing_configuration: crate::arena::resource_cycle::ResourceCycleConfig::v7(),
    }
}
fn require_corn_first_v7_teacher_source(
    source: &ReplaySource,
    players: usize,
    options: &GameOptions,
) -> Result<bool, String> {
    if !matches!(source, ReplaySource::CornFirstV7Teacher { .. }) {
        return Ok(false);
    }
    if !(3..=4).contains(&players)
        || *options != GameOptions::default()
        || serde_json::to_vec(source).map_err(|e| e.to_string())?
            != serde_json::to_vec(&corn_first_v7_teacher_provenance()).map_err(|e| e.to_string())?
    {
        return Err("Corn-first V7 teacher requires fixed all-seat base 3/4p provenance".into());
    }
    Ok(true)
}
fn corn_first_v7_teacher_choice(observation: &Observation) -> Result<crate::Decision, String> {
    match observation.phase {
        Phase::Setup => {
            crate::setup_policy::choose_move_with_weights(observation, &HeuristicWeights::default())
        }
        Phase::Playing => crate::arena::resource_cycle::choose_move(
            observation,
            &crate::arena::resource_cycle::ResourceCycleConfig::v7(),
        ),
        _ => Err("Corn-first V7 teacher requires an actual Setup or Playing callback".into()),
    }
}
/// Generate a complete all-seat fixed teacher replay from a real base-game
/// initial state. This grants no split reservation, dataset or model admission.
pub fn play_corn_first_v5_teacher(players: usize, seed: u32) -> Result<GameReplay, String> {
    let source = corn_first_v5_teacher_provenance();
    require_corn_first_v5_teacher_source(&source, players, &GameOptions::default())?;
    let (state, _, record) = play_game_using(
        players,
        seed,
        GameOptions::default(),
        true,
        source,
        corn_first_v5_teacher_choice,
    )?;
    let record = record.ok_or("Corn-first V5 teacher did not produce a replay")?;
    if verify_corn_first_v5_teacher_replay(&record)? != state {
        return Err("Corn-first V5 teacher terminal state differs".into());
    }
    Ok(record)
}
/// First replay every Core operation/view/result, then regenerate every actual
/// phase's teacher choice for every actor. Consistency is not authentication or
/// eligibility for any existing native/Human training route.
pub fn verify_corn_first_v5_teacher_replay(replay: &GameReplay) -> Result<GameState, String> {
    if !require_corn_first_v5_teacher_source(
        &replay.header.source,
        replay.header.names.len(),
        &replay.header.options,
    )? {
        return Err("Corn-first V5 teacher requires its dedicated replay source".into());
    }
    let terminal = verify_replay(replay)?;
    for step in &replay.steps {
        let decision = corn_first_v5_teacher_choice(&step.observation)?;
        if decision.actor != step.actor
            || !decision.score.is_finite()
            || decision.r#move != step.chosen.r#move
            || decision.policy_version
                != match step.observation.phase {
                    Phase::Setup => crate::setup_policy::POLICY_VERSION,
                    Phase::Playing => crate::arena::resource_cycle::V5_POLICY_VERSION,
                    _ => return Err("Unexpected teacher callback phase".into()),
                }
        {
            return Err(format!(
                "Corn-first V5 teacher choice differs at step {}",
                step.index
            ));
        }
    }
    Ok(terminal)
}
/// Separate V7 source and full Core/all-seat/all-phase teacher re-generation.
/// Generic replay verification or a policy-version header alone is insufficient.
pub fn verify_corn_first_v7_teacher_replay(replay: &GameReplay) -> Result<GameState, String> {
    if !require_corn_first_v7_teacher_source(
        &replay.header.source,
        replay.header.names.len(),
        &replay.header.options,
    )? {
        return Err("Corn-first V7 teacher requires its dedicated replay source".into());
    }
    let terminal = verify_replay(replay)?;
    for step in &replay.steps {
        let decision = corn_first_v7_teacher_choice(&step.observation)?;
        if decision.actor != step.actor
            || !decision.score.is_finite()
            || decision.r#move != step.chosen.r#move
            || decision.policy_version
                != match step.observation.phase {
                    Phase::Setup => crate::setup_policy::POLICY_VERSION,
                    Phase::Playing => crate::arena::resource_cycle::V7_POLICY_VERSION,
                    _ => return Err("Unexpected teacher callback phase".into()),
                }
        {
            return Err(format!(
                "Corn-first V7 teacher choice differs at step {}",
                step.index
            ));
        }
    }
    Ok(terminal)
}

#[cfg(test)]
pub(crate) fn corn_first_v5_teacher_fixture() {
    let source = corn_first_v5_teacher_provenance();
    let v7_source = corn_first_v7_teacher_provenance();
    assert!(!require_corn_first_v5_teacher_source(&v7_source, 3, &GameOptions::default()).unwrap());
    assert!(!require_corn_first_v7_teacher_source(&source, 3, &GameOptions::default()).unwrap());
    let mut mislabeled = serde_json::to_value(&v7_source).unwrap();
    mislabeled["playingConfiguration"] =
        serde_json::to_value(crate::arena::resource_cycle::ResourceCycleConfig::v5()).unwrap();
    assert!(
        require_corn_first_v7_teacher_source(
            &serde_json::from_value(mislabeled).unwrap(),
            3,
            &GameOptions::default()
        )
        .is_err()
    );
    for players in [2, 5] {
        assert!(play_corn_first_v5_teacher(players, 42).is_err());
    }
    for key in [
        "policyVersion",
        "setupPolicyVersion",
        "setupWeights",
        "playingConfiguration",
    ] {
        let mut bad = serde_json::to_value(&source).unwrap();
        match key {
            "setupWeights" => bad[key]["worker"] = serde_json::json!(999.0),
            "playingConfiguration" => {
                bad[key] = serde_json::to_value(
                    crate::arena::resource_cycle::ResourceCycleConfig::default(),
                )
                .unwrap();
            }
            _ => bad[key] = serde_json::json!("wrong-version"),
        }
        let bad: ReplaySource = serde_json::from_value(bad).unwrap();
        assert!(require_corn_first_v5_teacher_source(&bad, 3, &GameOptions::default()).is_err());
    }
    let mut unknown = serde_json::to_value(&source).unwrap();
    unknown["policies"] = serde_json::json!([]);
    assert!(serde_json::from_value::<ReplaySource>(unknown).is_err());
    assert!(require_corn_first_v5_teacher_source(&source, 3, &options_from_mask(1)).is_err());
    for players in [3, 4] {
        assert!(
            require_corn_first_v7_teacher_source(&v7_source, players, &GameOptions::default())
                .unwrap()
        );
        assert!(
            require_corn_first_v5_teacher_source(&source, players, &GameOptions::default())
                .unwrap()
        );
        let names = (0..players).map(|i| format!("P{i}")).collect();
        let mut state = create_game_with_options(names, 42, GameOptions::default()).unwrap();
        let mut actors = std::collections::BTreeSet::new();
        let mut setup_callbacks = 0;
        while state.phase == Phase::Setup {
            let observation = observe(&state, state.current_player).unwrap();
            let actual = corn_first_v5_teacher_choice(&observation).unwrap();
            assert_eq!(corn_first_v7_teacher_choice(&observation).unwrap(), actual);
            let expected = crate::setup_policy::choose_move_with_weights(
                &observation,
                &HeuristicWeights::default(),
            )
            .unwrap();
            assert_eq!(actual.policy_version, crate::setup_policy::POLICY_VERSION);
            assert_eq!(actual.actor, observation.actor);
            assert_eq!(actual.r#move, expected.r#move);
            assert_eq!(actual.score.to_bits(), expected.score.to_bits());
            assert!(
                observation
                    .legal_actions
                    .iter()
                    .any(|a| a.r#move == actual.r#move)
            );
            actors.insert(observation.actor);
            state = apply_move(&state, actual.r#move).unwrap();
            setup_callbacks += 1;
            assert!(setup_callbacks <= 64);
        }
        assert_eq!(actors.len(), players);
        assert_eq!(state.phase, Phase::Playing);
        let observation = observe(&state, state.current_player).unwrap();
        let actual = corn_first_v5_teacher_choice(&observation).unwrap();
        let expected = crate::arena::resource_cycle::choose_move(
            &observation,
            &crate::arena::resource_cycle::ResourceCycleConfig::v5(),
        )
        .unwrap();
        assert_eq!(
            actual.policy_version,
            crate::arena::resource_cycle::V5_POLICY_VERSION
        );
        assert_eq!(actual.actor, observation.actor);
        assert_eq!(actual.r#move, expected.r#move);
        assert_eq!(actual.score.to_bits(), expected.score.to_bits());
        let v7 = corn_first_v7_teacher_choice(&observation).unwrap();
        let expected_v7 = crate::arena::resource_cycle::choose_move(
            &observation,
            &crate::arena::resource_cycle::ResourceCycleConfig::v7(),
        )
        .unwrap();
        assert_eq!(v7, expected_v7);
        assert_eq!(
            v7.policy_version,
            crate::arena::resource_cycle::V7_POLICY_VERSION
        );
    }
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
        Some(source),
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
        Some(source),
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
        Some(source),
        |o| {
            let result = decide(index, o)?;
            index += 1;
            Ok(result)
        },
        !fast,
    )
}

/// Inference-only sibling: the same legal simulator and terminal checks, without
/// a recorded source or a training-compatible policy provenance.
pub(crate) fn play_unrecorded_inference(
    players: usize,
    seed: u32,
    mut decide: impl FnMut(usize, &Observation) -> Result<crate::Decision, String>,
) -> Result<(GameState, usize), String> {
    if !(3..=4).contains(&players) {
        return Err("Numeric inference evaluation requires base 3/4p".into());
    }
    let mut index = 0;
    let (state, decisions, _) = play_game_internal(
        players,
        seed,
        GameOptions::default(),
        false,
        None,
        |o| {
            let decision = decide(index, o)?;
            index += 1;
            Ok((decision, None, None))
        },
        false,
    )?;
    Ok((state, decisions))
}

/// Output-only actual Feeding events. The local buffer is returned only after
/// the same simulator completes and validates the entire game successfully.
pub(crate) struct IndexedFeedingEvent {
    pub move_index: usize,
    pub event: FeedingEvent,
}

#[derive(Clone, Debug, Default, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct AppliedInvestmentCounters {
    pub player_id: usize,
    pub paid_technology: [u64; 4],
    pub paid_tikal1: [u64; 4],
    pub paid_tikal3: [u64; 4],
    pub paid_unknown_origin: [u64; 4],
    pub free_technology_choices: [u64; 4],
    pub technology_level_increments: [i64; 4],
    pub capped_technology_actions: [u64; 4],
    pub uxmal3_actions: u64,
    pub uxmal3_worker_increments: i64,
    pub uxmal2_entries: u64,
    pub trade_buys: [u64; 3],
    pub trade_sells: [u64; 3],
    pub uxmal2_buys: [u64; 3],
    pub uxmal2_sells: [u64; 3],
    pub unknown_origin_buys: [u64; 3],
    pub unknown_origin_sells: [u64; 3],
    pub yaxchilan5_actions: u64,
    /// Net actual held-resource change, including any backward-action corn fee.
    pub yaxchilan5_resource_delta: [i64; 5],
}

#[derive(Clone, Copy)]
enum TechnologyOrigin {
    Tikal1,
    Tikal3,
}
#[derive(Default)]
struct InvestmentContext {
    technology: Option<TechnologyOrigin>,
    uxmal_trade: bool,
}
pub(crate) struct IndexedFeedingTechnology {
    pub move_index: usize,
    pub snapshot: FeedingTechnologySnapshot,
}
pub(crate) struct AppliedGameInvestments {
    pub players: Vec<AppliedInvestmentCounters>,
    pub feeding: Vec<IndexedFeedingTechnology>,
    contexts: Vec<InvestmentContext>,
}
impl AppliedGameInvestments {
    pub(crate) fn new(players: usize) -> Self {
        Self {
            players: (0..players)
                .map(|player_id| AppliedInvestmentCounters {
                    player_id,
                    ..AppliedInvestmentCounters::default()
                })
                .collect(),
            feeding: Vec::new(),
            contexts: (0..players).map(|_| InvestmentContext::default()).collect(),
        }
    }

    // Called only after exact legal-move matching and a successful Core apply.
    // Context follows the actual pending chain, never a presumed strategy.
    pub(crate) fn record_applied(
        &mut self,
        before: &GameState,
        after: &GameState,
        chosen: &TypedAction,
    ) {
        if before.phase != Phase::Playing {
            return;
        }
        let actor = before.current_player;
        let previous = &before.players[actor];
        let next = &after.players[actor];
        let counts = &mut self.players[actor];
        let context = &mut self.contexts[actor];
        for (i, technology) in TECHNOLOGY_IDS.into_iter().enumerate() {
            counts.technology_level_increments[i] +=
                (next.technologies[&technology] - previous.technologies[&technology]).max(0);
        }
        let resolved = match (chosen, before.pending.as_ref().map(|p| &p.task)) {
            (TypedAction::Payment { .. }, Some(Task::PayTechnology { technology, .. })) => {
                Some((*technology, false))
            }
            (TypedAction::Technology { technology }, Some(Task::Technology { free: true, .. })) => {
                Some((*technology, true))
            }
            _ => None,
        };
        if let Some((technology, free)) = resolved {
            let i = TECHNOLOGY_IDS
                .iter()
                .position(|t| *t == technology)
                .unwrap();
            if free {
                counts.free_technology_choices[i] += 1;
            } else {
                counts.paid_technology[i] += 1;
                match context.technology {
                    Some(TechnologyOrigin::Tikal1) => counts.paid_tikal1[i] += 1,
                    Some(TechnologyOrigin::Tikal3) => counts.paid_tikal3[i] += 1,
                    None => counts.paid_unknown_origin[i] += 1,
                }
            }
            if previous.technologies[&technology] == 3 {
                counts.capped_technology_actions[i] += 1;
            }
        }
        if let TypedAction::UseAction { gear, position, .. } = chosen {
            context.technology = match (*gear, *position) {
                (GearId::Tikal, 1) => Some(TechnologyOrigin::Tikal1),
                (GearId::Tikal, 3) => Some(TechnologyOrigin::Tikal3),
                _ => None,
            };
            context.uxmal_trade = *gear == GearId::Uxmal && *position == 2;
            if *gear == GearId::Uxmal && *position == 3 {
                counts.uxmal3_actions += 1;
                counts.uxmal3_worker_increments += next.workers - previous.workers;
            }
            if context.uxmal_trade {
                counts.uxmal2_entries += 1;
            }
            if *gear == GearId::Yaxchilan && *position == 5 {
                counts.yaxchilan5_actions += 1;
                for (i, resource) in RESOURCE_IDS.into_iter().enumerate() {
                    counts.yaxchilan5_resource_delta[i] +=
                        next.resources[&resource] - previous.resources[&resource];
                }
            }
        }
        if let TypedAction::Trade { resource, buy } = chosen
            && let Some(i) = MATERIALS.iter().position(|r| r == resource)
        {
            let (all, known, unknown) = if *buy {
                (
                    &mut counts.trade_buys,
                    &mut counts.uxmal2_buys,
                    &mut counts.unknown_origin_buys,
                )
            } else {
                (
                    &mut counts.trade_sells,
                    &mut counts.uxmal2_sells,
                    &mut counts.unknown_origin_sells,
                )
            };
            all[i] += 1;
            if context.uxmal_trade {
                known[i] += 1;
            } else {
                unknown[i] += 1;
            }
        }
        let pending = after.pending.as_ref();
        let same_actor = after.current_player == actor;
        let has_task = |predicate: fn(&Task) -> bool| {
            same_actor
                && pending.is_some_and(|p| predicate(&p.task) || p.after.iter().any(predicate))
        };
        if !has_task(|t| {
            matches!(
                t,
                Task::Technology { remaining: 1.., .. } | Task::PayTechnology { .. }
            )
        }) {
            context.technology = None;
        }
        if !has_task(|t| matches!(t, Task::Trade)) {
            context.uxmal_trade = false;
        }
    }
}

/// Output-only applied counters and exact Feeding snapshots, discarded if the
/// full game fails. This route creates no replay source or training authority.
pub(crate) fn play_unrecorded_inference_with_investments(
    players: usize,
    seed: u32,
    mut decide: impl FnMut(usize, &Observation) -> Result<crate::Decision, String>,
) -> Result<(GameState, usize, AppliedGameInvestments), String> {
    if !(3..=4).contains(&players) {
        return Err("Unrecorded inference supports base 3/4p only".into());
    }
    let mut actual = AppliedGameInvestments::new(players);
    let mut index = 0;
    let (state, decisions, _) = play_game_internal_with_diagnostics(
        players,
        seed,
        GameOptions::default(),
        false,
        None,
        |o| {
            let decision = decide(index, o)?;
            index += 1;
            Ok((decision, None, None))
        },
        false,
        None,
        Some(&mut actual),
    )?;
    Ok((state, decisions, actual))
}

pub(crate) fn play_unrecorded_inference_with_feeding(
    players: usize,
    seed: u32,
    mut decide: impl FnMut(usize, &Observation) -> Result<crate::Decision, String>,
) -> Result<(GameState, usize, Vec<IndexedFeedingEvent>), String> {
    if !(3..=4).contains(&players) {
        return Err("Unrecorded inference supports base 3/4p only".into());
    }
    let mut events = Vec::new();
    let mut index = 0;
    let (state, decisions, _) = play_game_internal_with_feeding(
        players,
        seed,
        GameOptions::default(),
        false,
        None,
        |o| {
            let decision = decide(index, o)?;
            index += 1;
            Ok((decision, None, None))
        },
        false,
        Some(&mut events),
    )?;
    Ok((state, decisions, events))
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
                    SeatPolicy::PublicLearnedTradeGuard { .. }
                    | SeatPolicy::PublicRlCurrentMixedTradeGuard { .. } => {
                        TradeGuardSession::from_provenance(policy).map(Some)
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
                d.policy_version != session.policy_version() || !d.score.is_finite()
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
    options: GameOptions,
    record: bool,
    source: Option<ReplaySource>,
    decide: impl FnMut(
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
    play_game_internal_with_feeding(
        players,
        seed,
        options,
        record,
        source,
        decide,
        check_every_step,
        None,
    )
}

#[allow(clippy::too_many_arguments)]
fn play_game_internal_with_feeding(
    players: usize,
    seed: u32,
    options: GameOptions,
    record: bool,
    source: Option<ReplaySource>,
    decide: impl FnMut(
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
    feeding_events: Option<&mut Vec<IndexedFeedingEvent>>,
) -> Result<(GameState, usize, Option<GameReplay>), String> {
    play_game_internal_with_diagnostics(
        players,
        seed,
        options,
        record,
        source,
        decide,
        check_every_step,
        feeding_events,
        None,
    )
}

#[allow(clippy::too_many_arguments)]
fn play_game_internal_with_diagnostics(
    players: usize,
    seed: u32,
    mut options: GameOptions,
    record: bool,
    source: Option<ReplaySource>,
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
    mut feeding_events: Option<&mut Vec<IndexedFeedingEvent>>,
    mut investments: Option<&mut AppliedGameInvestments>,
) -> Result<(GameState, usize, Option<GameReplay>), String> {
    if !(2..=5).contains(&players) {
        return Err("Players must be 2..5".into());
    }
    if source.is_none() && record {
        return Err("Unrecorded inference has no replay source".into());
    }
    if let Some(source) = &source {
        require_corn_first_v5_teacher_source(source, players, &options)?;
        require_corn_first_v7_teacher_source(source, players, &options)?;
    }
    if let Some(ReplaySource::PolicySelfPlay { policies }) = &source {
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
        if policies.iter().any(SeatPolicy::is_rl_evaluation)
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
        if policies
            .iter()
            .any(|policy| matches!(policy, SeatPolicy::ResourceCycle { .. }))
            && (!(3..=4).contains(&players) || options != GameOptions::default())
        {
            return Err("ResourceCycle provenance requires base 3..4p".into());
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
    let mut guard_replay = source.as_ref().map(GuardReplay::new).transpose()?;
    while state.phase != Phase::Finished {
        if decisions >= MAX_DECISIONS {
            return Err(format!(
                "Game did not finish: players={players}, seed={seed}, options={options:?}"
            ));
        }
        let observation = observe(&state, state.current_player)?;
        let (decision, search, trade_guard) = decide(&observation)?;
        if let Some(ReplaySource::PolicySelfPlay { policies }) = &source
            && let Some(policy_version) = policies[observation.actor].rl_evaluation_policy_version()
            && (decision.policy_version != policy_version || !decision.score.is_finite())
        {
            return Err("RL evaluation decision/provenance mismatch".into());
        }
        if let Some(source) = &source {
            validate_search_trace(source, &observation, &search)?;
        } else if search.is_some() || trade_guard.is_some() {
            return Err("Unrecorded inference diagnostics cannot claim replay provenance".into());
        }
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
        if let Some(guard_replay) = &mut guard_replay {
            guard_replay.check(
                decisions,
                &observation,
                &chosen,
                &trade_guard,
                Some(&decision),
            )?;
        }
        let before = if record {
            state_key(&state)?
        } else {
            String::new()
        };
        let applied = if let Some(actual) = investments.as_mut() {
            apply_move_with_feeding_technologies(&state, decision.r#move).map(
                |(next, snapshots)| {
                    actual.record_applied(&state, &next, &chosen.action);
                    actual.feeding.extend(snapshots.into_iter().map(|snapshot| {
                        IndexedFeedingTechnology {
                            move_index: decisions,
                            snapshot,
                        }
                    }));
                    next
                },
            )
        } else if let Some(events) = feeding_events.as_mut() {
            apply_move_with_feeding_events(&state, decision.r#move).map(|(next, actual)| {
                events.extend(actual.into_iter().map(|event| IndexedFeedingEvent {
                    move_index: decisions,
                    event,
                }));
                next
            })
        } else {
            apply_move(&state, decision.r#move)
        };
        state = applied.map_err(|e| format!("Legal-operation failure at decision {decisions} (players={players},seed={seed}): {e}; {chosen:?}"))?;
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
                source: source.ok_or("Recorded game requires its replay source")?,
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
    require_corn_first_v5_teacher_source(&h.source, h.names.len(), &h.options)?;
    require_corn_first_v7_teacher_source(&h.source, h.names.len(), &h.options)?;
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
        if policies.iter().any(SeatPolicy::is_rl_evaluation)
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
        if policies
            .iter()
            .any(|policy| matches!(policy, SeatPolicy::ResourceCycle { .. }))
            && (!(3..=4).contains(&h.names.len()) || h.options != GameOptions::default())
        {
            return Err("ResourceCycle provenance requires base 3..4p".into());
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
