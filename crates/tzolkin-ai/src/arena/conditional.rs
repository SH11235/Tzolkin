//! Conditional Human origin and explicit Setup/Playing composition for Dev only.
pub use super::numeric::{
    run_conditional_count_zero_validation, run_conditional_mixed_validation,
    run_conditional_validation,
};
use crate::Decision;
use crate::conditional_human_rl_update::{
    ConditionalMixedIdentity, ConditionalMixedOwner, ConditionalMixedPolicyHandle,
};
use crate::dataset::DatasetSplit;
use crate::human_policy_dataset::{
    CAPPED_EXTRACTION_LABEL_SCOPE, ConditionalFeatureProfile, EXPLICIT_MARKET_ANYWHERE_LABEL_SCOPE,
    FACTORIZED_LABEL_SCOPE, OBSERVED_CORN_BEG_LABEL_SCOPE, OBSERVED_PENDING_CHOICE_LABEL_SCOPE,
    PAID_TECHNOLOGY_LABEL_SCOPE,
};
use crate::human_policy_training::{ConditionalDatasetIdentity, PreparedConditionalHumanPolicy};
use crate::policy::HeuristicWeights;
use crate::semantic_playing_teacher_dataset::TeacherProfile;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use tzolkin_core::Phase;
use tzolkin_core::observation::{ActionSource, Observation, TypedAction, observation_key};
use tzolkin_inference::deployment::{GuardedTradeTrace, MAX_GAME_CALLBACKS};
use tzolkin_inference::policy::{NumericPolicyDistribution, validate_contract};
use tzolkin_inference::trade_guard::{
    TradeGuardConfig, TradeHistory, is_trade, public_bytes, skip_index,
};

pub const POLICY_VERSION: &str = "conditional-human-corn-setup-guarded-playing-v1";
pub const ARENA_SCHEMA: &str = "tzolkin-conditional-human-policy-development-arena-v1";
pub const SEMANTIC_PLAYING_POLICY_VERSION: &str =
    "conditional-human-semantic-playing-corn-setup-guarded-v1";
pub const MIXED_POLICY_VERSION: &str = "conditional-human-current-allphase-guarded-argmax-v1";
pub const MIXED_ARENA_SCHEMA: &str = "tzolkin-conditional-human-mixed-development-arena-v1";
pub const COUNT_ZERO_POLICY_VERSION: &str =
    "conditional-human-countzero-allphase-guarded-argmax-v1";
pub const COUNT_ZERO_ARENA_SCHEMA: &str =
    "tzolkin-conditional-human-countzero-development-arena-v1";

/// The algorithm and weights are the existing frozen Corn-first Setup policy.
#[derive(Clone, Copy, Debug, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub enum ConditionalSetup {
    CornFirst,
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ConditionalIdentity {
    pub policy_version: String,
    pub task: serde_json::Value,
    pub input_contract: String,
    pub checkpoint_checksum: String,
    pub model_checksum: String,
    pub dataset: serde_json::Value,
    pub reservations: BTreeMap<String, DatasetSplit>,
    pub backend: String,
    pub setup: ConditionalSetup,
    pub setup_policy_version: String,
    pub setup_weights: HeuristicWeights,
    pub guard: TradeGuardConfig,
    pub configuration_key: String,
    pub source_qualified: bool,
    pub verified_native_policy: bool,
    pub value_validity: serde_json::Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub choice_diagnostics: Option<&'static str>,
}

/// Borrowed immutable conditional storage; no native artifact or owner conversion.
pub struct ConditionalPolicy<'a> {
    prepared: &'a PreparedConditionalHumanPolicy,
    identity: ConditionalIdentity,
}
impl<'a> ConditionalPolicy<'a> {
    pub fn new(
        prepared: &'a PreparedConditionalHumanPolicy,
        setup: ConditionalSetup,
        guard: TradeGuardConfig,
    ) -> Result<Self, String> {
        // This constructor retains the legacy schema-2 policy identity.
        prepared.model().validate()?;
        let configuration_key = guard.configuration_key()?;
        Ok(Self {
            prepared,
            identity: ConditionalIdentity {
                policy_version: POLICY_VERSION.into(),
                task: serde_json::to_value(&prepared.model().task).map_err(|e| e.to_string())?,
                input_contract: prepared.model().input_contract.clone(),
                checkpoint_checksum: prepared.checkpoint_checksum().into(),
                model_checksum: prepared.model().checksum.clone(),
                dataset: serde_json::to_value(prepared.identity()).map_err(|e| e.to_string())?,
                reservations: prepared.reservations().clone(),
                backend: prepared.backend().into(),
                setup,
                setup_policy_version: crate::setup_policy::POLICY_VERSION.into(),
                setup_weights: HeuristicWeights::default(),
                guard,
                configuration_key,
                source_qualified: false,
                verified_native_policy: false,
                value_validity: serde_json::to_value(prepared.model().value_validity)
                    .map_err(|e| e.to_string())?,
                choice_diagnostics: None,
            },
        })
    }
    pub fn identity(&self) -> &ConditionalIdentity {
        &self.identity
    }
    /// Opt in to bounded context counts without changing choices or teacher scope.
    pub fn with_choice_diagnostics(mut self) -> Self {
        self.identity.choice_diagnostics = Some(CHOICE_DIAGNOSTICS_SCHEMA);
        self
    }
    pub(super) fn teacher_grammar_contains(&self, action: &TypedAction) -> bool {
        teacher_grammar_contains(self.prepared.identity().label_scope.as_deref(), action)
    }
}

/// Explicit schema-4 Playing with the original source-loaded conditional origin.
/// This type cannot enter the legacy Arena, native deployment or learner APIs.
pub struct SemanticPlayingConditionalPolicy<'a> {
    prepared: &'a PreparedConditionalHumanPolicy,
    identity: ConditionalIdentity,
}
impl<'a> SemanticPlayingConditionalPolicy<'a> {
    pub fn new(
        prepared: &'a PreparedConditionalHumanPolicy,
        setup: ConditionalSetup,
        guard: TradeGuardConfig,
    ) -> Result<Self, String> {
        prepared.model().validate_semantic_playing()?;
        if prepared.identity().feature_profile()? != ConditionalFeatureProfile::SemanticPlaying4 {
            return Err("Semantic Playing Arena requires its own schema-4 origin".into());
        }
        prepared.recheck_sources()?;
        let configuration_key = guard.configuration_key()?;
        Ok(Self {
            prepared,
            identity: ConditionalIdentity {
                policy_version: SEMANTIC_PLAYING_POLICY_VERSION.into(),
                task: serde_json::to_value(&prepared.model().task).map_err(|e| e.to_string())?,
                input_contract: prepared.model().input_contract.clone(),
                checkpoint_checksum: prepared.checkpoint_checksum().into(),
                model_checksum: prepared.model().checksum.clone(),
                dataset: serde_json::to_value(prepared.identity()).map_err(|e| e.to_string())?,
                reservations: prepared.reservations().clone(),
                backend: prepared.backend().into(),
                setup,
                setup_policy_version: crate::setup_policy::POLICY_VERSION.into(),
                setup_weights: HeuristicWeights::default(),
                guard,
                configuration_key,
                source_qualified: false,
                verified_native_policy: false,
                value_validity: serde_json::to_value(prepared.model().value_validity)
                    .map_err(|e| e.to_string())?,
                choice_diagnostics: None,
            },
        })
    }
    pub fn identity(&self) -> &ConditionalIdentity {
        &self.identity
    }
    pub fn with_choice_diagnostics(mut self) -> Self {
        self.identity.choice_diagnostics = Some(CHOICE_DIAGNOSTICS_SCHEMA);
        self
    }
    pub(super) fn prepared(&self) -> &'a PreparedConditionalHumanPolicy {
        self.prepared
    }
    pub(super) fn teacher_grammar_contains(&self, action: &TypedAction) -> bool {
        teacher_grammar_contains(self.prepared.identity().label_scope.as_deref(), action)
    }
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ConditionalMixedArenaIdentity {
    pub policy_version: String,
    pub input_contract: String,
    pub origin_checksum: String,
    pub dataset: ConditionalDatasetIdentity,
    pub reservations: BTreeMap<String, DatasetSplit>,
    pub current: ConditionalMixedIdentity,
    pub backend: String,
    pub selection: String,
    pub guard: TradeGuardConfig,
    pub configuration_key: String,
    pub source_qualified: bool,
    pub verified_native_policy: bool,
    pub value_validity: crate::public_model::ValueValidity,
}

/// A genuine updated actor borrowed for all-phase inference, not a learning or
/// close certificate. A journal caller obtains it only through a Ready owner;
/// a restarted caller still needs the journal's positive-close restore boundary.
pub struct ConditionalMixedArenaPolicy<'a> {
    handle: ConditionalMixedPolicyHandle<'a>,
    identity: ConditionalMixedArenaIdentity,
}
impl<'a> ConditionalMixedArenaPolicy<'a> {
    pub fn new(owner: &'a ConditionalMixedOwner, guard: TradeGuardConfig) -> Result<Self, String> {
        let handle = owner.handle()?;
        if handle.identity().mixed_update_count() == 0 {
            return Err("Conditional mixed Arena requires a positive update count".into());
        }
        let configuration_key = guard.configuration_key()?;
        Ok(Self {
            identity: ConditionalMixedArenaIdentity {
                policy_version: MIXED_POLICY_VERSION.into(),
                input_contract: crate::conditional_human_rl::INPUT_CONTRACT.into(),
                origin_checksum: owner.origin().checksum().into(),
                dataset: owner.origin().dataset().clone(),
                reservations: owner.origin().table_reservations().clone(),
                current: handle.identity().clone(),
                backend: "scalar".into(),
                selection: "orderedFirstMaxSetupPlaying".into(),
                guard,
                configuration_key,
                source_qualified: false,
                verified_native_policy: false,
                value_validity: crate::public_model::ValueValidity::UnavailablePolicyOnly,
            },
            handle,
        })
    }
    pub fn identity(&self) -> &ConditionalMixedArenaIdentity {
        &self.identity
    }
    pub(super) fn teacher_grammar_contains(&self, action: &TypedAction) -> bool {
        teacher_grammar_contains(self.identity.dataset.label_scope.as_deref(), action)
    }
}

/// Fresh BC control identity. This is neither an updated deployment nor a
/// training, producing-process, value or source-authentication certificate.
#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ConditionalCountZeroArenaIdentity {
    pub policy_version: String,
    pub input_contract: String,
    pub origin_checksum: String,
    pub dataset: ConditionalDatasetIdentity,
    pub reservations: BTreeMap<String, DatasetSplit>,
    pub count_zero: ConditionalMixedIdentity,
    pub backend: String,
    pub selection: String,
    pub guard: TradeGuardConfig,
    pub configuration_key: String,
    pub source_qualified: bool,
    pub verified_native_policy: bool,
    pub value_validity: crate::public_model::ValueValidity,
}

/// Read-only all-phase control borrowed from a genuine, freshly loaded BC
/// origin. No sampler, reservation, counter mutation or portable export exists.
pub struct ConditionalCountZeroArenaPolicy<'a> {
    handle: ConditionalMixedPolicyHandle<'a>,
    identity: ConditionalCountZeroArenaIdentity,
}
impl<'a> ConditionalCountZeroArenaPolicy<'a> {
    pub fn new_bc_control(
        owner: &'a ConditionalMixedOwner,
        guard: TradeGuardConfig,
    ) -> Result<Self, String> {
        if owner.origin().source_pins().is_empty() {
            return Err("Count0 BC control requires original loaded source custody".into());
        }
        // Lifecycle IO only; prediction and notifications borrow unchanged bits.
        owner.origin().recheck_sources()?;
        Self::from_owner(owner, guard)
    }
    fn from_owner(
        owner: &'a ConditionalMixedOwner,
        guard: TradeGuardConfig,
    ) -> Result<Self, String> {
        let handle = owner.handle()?;
        require_fresh_count_zero(handle.identity())?;
        let configuration_key = guard.configuration_key()?;
        Ok(Self {
            identity: ConditionalCountZeroArenaIdentity {
                policy_version: COUNT_ZERO_POLICY_VERSION.into(),
                input_contract: crate::conditional_human_rl::INPUT_CONTRACT.into(),
                origin_checksum: owner.origin().checksum().into(),
                dataset: owner.origin().dataset().clone(),
                reservations: owner.origin().table_reservations().clone(),
                count_zero: handle.identity().clone(),
                backend: "scalar".into(),
                selection: "orderedFirstMaxSetupPlaying".into(),
                guard,
                configuration_key,
                source_qualified: false,
                verified_native_policy: false,
                value_validity: crate::public_model::ValueValidity::UnavailablePolicyOnly,
            },
            handle,
        })
    }
    pub fn identity(&self) -> &ConditionalCountZeroArenaIdentity {
        &self.identity
    }
    pub(super) fn teacher_grammar_contains(&self, action: &TypedAction) -> bool {
        teacher_grammar_contains(self.identity.dataset.label_scope.as_deref(), action)
    }
    #[cfg(test)]
    pub(super) fn fixture(
        owner: &'a ConditionalMixedOwner,
        guard: TradeGuardConfig,
    ) -> Result<Self, String> {
        // Existing tiny synthetic owners intentionally have no loaded custody.
        // The public constructor's refusal is tested separately.
        Self::from_owner(owner, guard)
    }
}
fn require_fresh_count_zero(identity: &ConditionalMixedIdentity) -> Result<(), String> {
    if identity.mixed_update_count() != 0 || identity.next_ordinal() != 0 {
        return Err("Count0 BC control requires fresh zero-update/zero-ordinal owner".into());
    }
    Ok(())
}

// Category membership is diagnostic only: it does not check observed captions,
// an uncapped technology/pending payment, or positive coverage in actual Train rows.
fn teacher_grammar_contains(label_scope: Option<&str>, action: &TypedAction) -> bool {
    match label_scope {
        None
        | Some(FACTORIZED_LABEL_SCOPE)
        | Some(
            PAID_TECHNOLOGY_LABEL_SCOPE
            | EXPLICIT_MARKET_ANYWHERE_LABEL_SCOPE
            | OBSERVED_CORN_BEG_LABEL_SCOPE
            | OBSERVED_PENDING_CHOICE_LABEL_SCOPE
            | CAPPED_EXTRACTION_LABEL_SCOPE,
        ) => {
            matches!(
                action,
                TypedAction::Place { .. }
                    | TypedAction::Remove { .. }
                    | TypedAction::FirstPlayer { .. }
            ) || (label_scope.is_some()
                && matches!(
                    action,
                    TypedAction::UseAction {
                        source: ActionSource::CurrentGear,
                        ..
                    }
                ))
                || (matches!(
                    label_scope,
                    Some(
                        PAID_TECHNOLOGY_LABEL_SCOPE
                            | EXPLICIT_MARKET_ANYWHERE_LABEL_SCOPE
                            | OBSERVED_CORN_BEG_LABEL_SCOPE
                            | OBSERVED_PENDING_CHOICE_LABEL_SCOPE
                            | CAPPED_EXTRACTION_LABEL_SCOPE
                    )
                ) && match action {
                    TypedAction::Technology { .. } => true,
                    TypedAction::Payment { resources } => {
                        resources[0] == 0
                            && resources[4] == 0
                            && resources[1..4].iter().all(|&value| value >= 0)
                            && resources[1..4].iter().any(|&value| value > 0)
                    }
                    _ => false,
                })
                || (matches!(
                    label_scope,
                    Some(
                        EXPLICIT_MARKET_ANYWHERE_LABEL_SCOPE
                            | OBSERVED_CORN_BEG_LABEL_SCOPE
                            | OBSERVED_PENDING_CHOICE_LABEL_SCOPE
                            | CAPPED_EXTRACTION_LABEL_SCOPE
                    )
                ) && matches!(
                    action,
                    TypedAction::Trade {
                        resource: tzolkin_core::Resource::Wood
                            | tzolkin_core::Resource::Stone
                            | tzolkin_core::Resource::Gold,
                        buy: false
                    } | TypedAction::UseAction {
                        source: ActionSource::Anywhere,
                        corn_cost: 1,
                        ..
                    }
                ))
                || (matches!(
                    label_scope,
                    Some(
                        OBSERVED_CORN_BEG_LABEL_SCOPE
                            | OBSERVED_PENDING_CHOICE_LABEL_SCOPE
                            | CAPPED_EXTRACTION_LABEL_SCOPE
                    )
                ) && matches!(
                    action,
                    TypedAction::Beg | TypedAction::Temple { direction: -1, .. }
                ))
                || (matches!(
                    label_scope,
                    Some(OBSERVED_PENDING_CHOICE_LABEL_SCOPE | CAPPED_EXTRACTION_LABEL_SCOPE)
                ) && matches!(
                    action,
                    TypedAction::Temple { direction: 1, .. }
                        | TypedAction::Monument {
                            renovation: None,
                            ..
                        }
                ))
        }
        Some(_) => false,
    }
}

#[derive(Clone, Debug, Default, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ConditionalDiagnostics {
    pub setup_choices: usize,
    pub playing_choices: usize,
    /// Counts effective choices after the guard, including guard-forced Skip.
    /// Counts categories even when the training dataset had no such labels.
    /// A grammar-supported category need not occur in the actual Train tables.
    pub selected_action_kinds: BTreeMap<String, usize>,
    /// Uses the sealed scope: v1 primitives, v2 also CurrentGear UseAction,
    /// v3 also Technology and positive material-only Payment categories.
    /// v4 also material Sell and one-corn Anywhere categories.
    /// Counts effective choices after the guard, so guard-forced Skip is outside.
    /// Membership does not establish positive Train coverage of that category.
    /// For a mixed RL actor this still refers to its original BC teacher grammar,
    /// not the action categories it subsequently encountered during RL.
    pub outside_teacher_grammar_choices: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub choice_contexts: Option<ConditionalChoiceDiagnostics>,
}

pub const CHOICE_DIAGNOSTICS_SCHEMA: &str = "tzolkin-conditional-returned-choice-context-counts-v1";
const MAX_CHOICE_CONTEXTS: usize = 256;
const ROUND_BINS: [[u8; 2]; 4] = [[1, 7], [8, 14], [15, 21], [22, 27]];
const ACTION_KINDS: [&str; 27] = [
    "place",
    "remove",
    "firstPlayer",
    "beg",
    "quickAction",
    "endTurn",
    "chooseTribe",
    "chooseWealth",
    "skip",
    "selectSpace",
    "technologyBonus",
    "prophecyGain",
    "prophecyTemple",
    "useAction",
    "technology",
    "payment",
    "temple",
    "resource",
    "build",
    "monument",
    "technologyExchange",
    "trade",
    "harvest",
    "offering",
    "rotate",
    "tribeSell",
    "tribeSkipSpace",
];

fn action_kind(action: &TypedAction) -> usize {
    match action {
        TypedAction::Place { .. } => 0,
        TypedAction::Remove { .. } => 1,
        TypedAction::FirstPlayer { .. } => 2,
        TypedAction::Beg => 3,
        TypedAction::QuickAction { .. } => 4,
        TypedAction::EndTurn { .. } => 5,
        TypedAction::ChooseTribe { .. } => 6,
        TypedAction::ChooseWealth { .. } => 7,
        TypedAction::Skip => 8,
        TypedAction::SelectSpace { .. } => 9,
        TypedAction::TechnologyBonus { .. } => 10,
        TypedAction::ProphecyGain { .. } => 11,
        TypedAction::ProphecyTemple { .. } => 12,
        TypedAction::UseAction { .. } => 13,
        TypedAction::Technology { .. } => 14,
        TypedAction::Payment { .. } => 15,
        TypedAction::Temple { .. } => 16,
        TypedAction::Resource { .. } => 17,
        TypedAction::Build { .. } => 18,
        TypedAction::Monument { .. } => 19,
        TypedAction::TechnologyExchange { .. } => 20,
        TypedAction::Trade { .. } => 21,
        TypedAction::Harvest { .. } => 22,
        TypedAction::Offering { .. } => 23,
        TypedAction::Rotate { .. } => 24,
        TypedAction::TribeSell { .. } => 25,
        TypedAction::TribeSkipSpace => 26,
    }
}

/// Categories only. These counts establish neither positive Train coverage nor OOD.
/// Choices returned before core apply are counted; successful apply is not inferred.
#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ConditionalChoiceDiagnostics {
    schema: &'static str,
    max_contexts: usize,
    max_choices: usize,
    round_bins: [[u8; 2]; 4],
    choices: usize,
    overflow_choices: usize,
    requested_outside_teacher_grammar_choices: usize,
    effective_outside_teacher_grammar_choices: usize,
    contexts: Vec<ChoiceContextCounts>,
    #[serde(skip)]
    indices: BTreeMap<(u8, u32, usize, usize, bool, bool), usize>,
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct ChoiceContextCounts {
    round_bin: u8,
    legal_kinds: Vec<&'static str>,
    requested_kind: &'static str,
    effective_kind: &'static str,
    requested_outside_teacher_grammar: bool,
    effective_outside_teacher_grammar: bool,
    choices: usize,
    guard_overrides: usize,
    repeated_trade_exits: usize,
    trade_budget_exits: usize,
}

impl ConditionalChoiceDiagnostics {
    pub(super) fn new() -> Self {
        Self {
            schema: CHOICE_DIAGNOSTICS_SCHEMA,
            max_contexts: MAX_CHOICE_CONTEXTS,
            max_choices: MAX_GAME_CALLBACKS,
            round_bins: ROUND_BINS,
            choices: 0,
            overflow_choices: 0,
            requested_outside_teacher_grammar_choices: 0,
            effective_outside_teacher_grammar_choices: 0,
            contexts: Vec::new(),
            indices: BTreeMap::new(),
        }
    }

    pub(super) fn record(
        &mut self,
        observation: &Observation,
        requested: usize,
        effective: usize,
        trade: Option<&GuardedTradeTrace>,
        in_grammar: impl Fn(&TypedAction) -> bool,
    ) -> Result<(), String> {
        if observation.phase != Phase::Playing
            || !(1..=27).contains(&observation.round)
            || observation.legal_actions.is_empty()
            || observation.legal_actions.len() > super::numeric::MAX_ROWS_PER_CALLBACK
            || self.choices >= MAX_GAME_CALLBACKS
        {
            return Err("Conditional choice context exceeds fixed Playing bounds".into());
        }
        let requested = &observation
            .legal_actions
            .get(requested)
            .ok_or("Requested conditional diagnostic row is not legal")?
            .action;
        let effective = &observation
            .legal_actions
            .get(effective)
            .ok_or("Effective conditional diagnostic row is not legal")?
            .action;
        let requested_kind = action_kind(requested);
        let effective_kind = action_kind(effective);
        let requested_outside = !in_grammar(requested);
        let effective_outside = !in_grammar(effective);
        let mask = observation
            .legal_actions
            .iter()
            .fold(0u32, |mask, row| mask | (1 << action_kind(&row.action)));
        let round_bin = ((observation.round - 1) / 7) as u8;
        let key = (
            round_bin,
            mask,
            requested_kind,
            effective_kind,
            requested_outside,
            effective_outside,
        );
        self.choices += 1;
        self.requested_outside_teacher_grammar_choices += usize::from(requested_outside);
        self.effective_outside_teacher_grammar_choices += usize::from(effective_outside);
        let index = if let Some(index) = self.indices.get(&key) {
            *index
        } else if self.contexts.len() < MAX_CHOICE_CONTEXTS {
            let index = self.contexts.len();
            self.indices.insert(key, index);
            self.contexts.push(ChoiceContextCounts {
                round_bin,
                legal_kinds: ACTION_KINDS
                    .iter()
                    .enumerate()
                    .filter_map(|(i, kind)| (mask & (1 << i) != 0).then_some(*kind))
                    .collect(),
                requested_kind: ACTION_KINDS[requested_kind],
                effective_kind: ACTION_KINDS[effective_kind],
                requested_outside_teacher_grammar: requested_outside,
                effective_outside_teacher_grammar: effective_outside,
                choices: 0,
                guard_overrides: 0,
                repeated_trade_exits: 0,
                trade_budget_exits: 0,
            });
            index
        } else {
            // Keep totals exact without allocating another context or changing the move.
            self.overflow_choices += 1;
            return Ok(());
        };
        let counts = &mut self.contexts[index];
        counts.choices += 1;
        if let Some(trade) = trade {
            counts.guard_overrides += usize::from(trade.overridden);
            counts.repeated_trade_exits += usize::from(trade.overridden && trade.repeat_seen);
            counts.trade_budget_exits += usize::from(trade.overridden && trade.budget_reached);
        }
        Ok(())
    }
}

pub(super) struct ConditionalDecision {
    pub decision: Decision,
    pub trade: Option<GuardedTradeTrace>,
    pub requested_index: usize,
    pub opening: Option<OpeningSelection>,
}
pub(super) struct OpeningSelection {
    pub placement_count: usize,
    pub changed_move: bool,
}

// Keep the legacy public Setup composition separate from all-phase actor inference.
#[derive(Clone, Copy)]
enum SetupRoute {
    CornFirst,
    SemanticPlayingCornFirst,
    NativeTeacherPlayingCornFirst(TeacherProfile),
    NativeV7FundedUxmalPlayingCornFirst,
    CurrentSnapshotPlayingCornFirst,
    CheckpointPolicyPlayingCornFirst,
    NumericAllPhase(&'static str),
    PhaseRoutedDevelopment,
}
impl SetupRoute {
    fn policy_version(self) -> &'static str {
        match self {
            Self::CornFirst => POLICY_VERSION,
            Self::SemanticPlayingCornFirst => SEMANTIC_PLAYING_POLICY_VERSION,
            Self::NativeTeacherPlayingCornFirst(profile) => match profile {
                TeacherProfile::V5 => super::teacher::SEMANTIC_PLAYING_POLICY_VERSION,
                TeacherProfile::V7 => super::teacher::SEMANTIC_PLAYING_V7_POLICY_VERSION,
            },
            Self::NativeV7FundedUxmalPlayingCornFirst => {
                super::teacher::SEMANTIC_PLAYING_V7_FUNDED_UXMAL_POLICY_VERSION
            }
            Self::CurrentSnapshotPlayingCornFirst => super::semantic_snapshot::POLICY_VERSION,
            Self::CheckpointPolicyPlayingCornFirst => {
                super::semantic_snapshot::CHECKPOINT_POLICY_VERSION
            }
            Self::NumericAllPhase(version) => version,
            Self::PhaseRoutedDevelopment => super::phase_routed::POLICY_VERSION,
        }
    }
}
/// Fresh callback ownership and bounded Trade history for one arm. Weights stay borrowed.
pub(super) struct GameSession {
    setup: SetupRoute,
    history: TradeHistory,
    next_callback: usize,
    notified: Option<(usize, usize, bool, String)>,
    chosen: bool,
    failed: bool,
    actor: Option<usize>,
}
impl GameSession {
    pub(super) fn new(policy: &ConditionalPolicy<'_>) -> Self {
        Self::with_config(policy.identity.setup, &policy.identity.guard)
            .expect("conditional constructor validated guard configuration")
    }
    pub(super) fn new_semantic_playing(policy: &SemanticPlayingConditionalPolicy<'_>) -> Self {
        Self::with_route(SetupRoute::SemanticPlayingCornFirst, &policy.identity.guard)
            .expect("semantic Playing constructor validated guard configuration")
    }
    pub(super) fn new_native_playing_teacher(
        policy: &super::teacher::SemanticPlayingTeacherArenaPolicy<'_>,
    ) -> Self {
        Self::with_route(
            if policy.funded_uxmal_opening {
                SetupRoute::NativeV7FundedUxmalPlayingCornFirst
            } else {
                SetupRoute::NativeTeacherPlayingCornFirst(policy.prepared.profile())
            },
            &policy.identity().guard,
        )
        .expect("native teacher constructor validated guard configuration")
    }
    pub(super) fn new_semantic_snapshot(
        policy: &super::semantic_snapshot::SemanticPlayingSnapshotArenaPolicy<'_>,
    ) -> Self {
        Self::with_route(
            SetupRoute::CurrentSnapshotPlayingCornFirst,
            &policy.identity().guard,
        )
        .expect("semantic snapshot constructor validated guard configuration")
    }
    pub(super) fn new_checkpoint_policy(
        policy: &super::semantic_snapshot::SemanticPlayingCheckpointArenaPolicy,
    ) -> Self {
        Self::with_route(
            SetupRoute::CheckpointPolicyPlayingCornFirst,
            &policy.identity().guard,
        )
        .expect("semantic checkpoint constructor validated the default guard")
    }
    pub(super) fn new_mixed(policy: &ConditionalMixedArenaPolicy<'_>) -> Self {
        Self::with_route(
            SetupRoute::NumericAllPhase(MIXED_POLICY_VERSION),
            &policy.identity.guard,
        )
        .expect("conditional mixed constructor validated guard configuration")
    }
    pub(super) fn new_count_zero(policy: &ConditionalCountZeroArenaPolicy<'_>) -> Self {
        Self::with_route(
            SetupRoute::NumericAllPhase(COUNT_ZERO_POLICY_VERSION),
            &policy.identity.guard,
        )
        .expect("conditional count0 constructor validated guard configuration")
    }
    pub(super) fn new_teacher(
        policy: &super::teacher::ResourceCycleTeacherArenaPolicy<'_>,
    ) -> Self {
        Self::with_route(
            SetupRoute::NumericAllPhase(super::teacher::POLICY_VERSION),
            &policy.identity().guard,
        )
        .expect("teacher constructor validated guard configuration")
    }
    pub(super) fn new_semantic_setup_teacher(
        policy: &super::teacher::SemanticSetupResourceCycleTeacherArenaPolicy<'_>,
    ) -> Self {
        Self::with_route(
            SetupRoute::NumericAllPhase(super::teacher::SEMANTIC_SETUP_POLICY_VERSION),
            &policy.identity().teacher.guard,
        )
        .expect("semantic Setup teacher constructor validated guard configuration")
    }
    pub(super) fn new_phase_routed(
        policy: &super::phase_routed::PhaseRoutedDevelopmentPolicy<'_>,
    ) -> Self {
        Self::with_route(SetupRoute::PhaseRoutedDevelopment, &policy.identity().guard)
            .expect("phase-routed constructor validated the default guard configuration")
    }
    fn with_config(setup: ConditionalSetup, guard: &TradeGuardConfig) -> Result<Self, String> {
        let route = match setup {
            ConditionalSetup::CornFirst => SetupRoute::CornFirst,
        };
        Self::with_route(route, guard)
    }
    fn with_route(setup: SetupRoute, guard: &TradeGuardConfig) -> Result<Self, String> {
        guard.configuration_key()?;
        Ok(Self {
            setup,
            history: TradeHistory::new(guard.max_trades_per_episode, guard.max_public_state_bytes)?,
            next_callback: 0,
            notified: None,
            chosen: false,
            failed: false,
            actor: None,
        })
    }
    fn reset_episode(&mut self) {
        self.actor = None;
        self.history.reset();
    }
    fn invalidate(&mut self) {
        self.reset_episode();
        self.failed = true;
    }
    pub(super) fn on_callback(&mut self, index: usize, o: &Observation) -> Result<(), String> {
        let result = (|| {
            if self.failed || index != self.next_callback || index >= MAX_GAME_CALLBACKS {
                return Err("Stale/out-of-order conditional callback".into());
            }
            validate_contract(o)?;
            if o.actor >= o.players.len()
                || o.observation_key != observation_key(o)?
                || o.legal_actions.is_empty()
                || o.legal_actions.len() > super::numeric::MAX_ROWS_PER_CALLBACK
            {
                return Err("Conditional callback key/row bound mismatch".into());
            }
            let trade = is_trade(o);
            if !trade || self.actor.is_some_and(|actor| actor != o.actor) {
                self.reset_episode();
            }
            self.next_callback += 1;
            self.notified = Some((index, o.actor, trade, o.observation_key.clone()));
            self.chosen = false;
            Ok(())
        })();
        if result.is_err() {
            self.invalidate();
        }
        result
    }
    pub(super) fn choose_move(
        &mut self,
        index: usize,
        o: &Observation,
        policy: &ConditionalPolicy<'_>,
    ) -> Result<ConditionalDecision, String> {
        self.choose_using(index, o, |o| policy.prepared.distribution(o))
    }
    pub(super) fn choose_semantic_playing(
        &mut self,
        index: usize,
        o: &Observation,
        policy: &SemanticPlayingConditionalPolicy<'_>,
    ) -> Result<ConditionalDecision, String> {
        self.choose_using(index, o, |o| {
            policy.prepared.distribution_semantic_playing(o)
        })
    }
    pub(super) fn choose_mixed(
        &mut self,
        index: usize,
        o: &Observation,
        policy: &ConditionalMixedArenaPolicy<'_>,
    ) -> Result<ConditionalDecision, String> {
        self.choose_using(index, o, |o| policy.handle.numeric_distribution(o))
    }
    pub(super) fn choose_count_zero(
        &mut self,
        index: usize,
        o: &Observation,
        policy: &ConditionalCountZeroArenaPolicy<'_>,
    ) -> Result<ConditionalDecision, String> {
        self.choose_using(index, o, |o| policy.handle.numeric_distribution(o))
    }
    pub(super) fn choose_teacher(
        &mut self,
        index: usize,
        o: &Observation,
        policy: &super::teacher::ResourceCycleTeacherArenaPolicy<'_>,
    ) -> Result<ConditionalDecision, String> {
        self.choose_using(index, o, |o| policy.prepared.distribution(o))
    }
    pub(super) fn choose_semantic_setup_teacher(
        &mut self,
        index: usize,
        o: &Observation,
        policy: &super::teacher::SemanticSetupResourceCycleTeacherArenaPolicy<'_>,
    ) -> Result<ConditionalDecision, String> {
        self.choose_using(index, o, |o| policy.prepared.distribution(o))
    }
    pub(super) fn choose_native_playing_teacher(
        &mut self,
        index: usize,
        o: &Observation,
        policy: &super::teacher::SemanticPlayingTeacherArenaPolicy<'_>,
    ) -> Result<ConditionalDecision, String> {
        self.choose_using(index, o, |o| policy.prepared.distribution(o))
    }
    pub(super) fn choose_semantic_snapshot(
        &mut self,
        index: usize,
        o: &Observation,
        policy: &super::semantic_snapshot::SemanticPlayingSnapshotArenaPolicy<'_>,
    ) -> Result<ConditionalDecision, String> {
        self.choose_using(index, o, |o| policy.distribution(o))
    }
    pub(super) fn choose_checkpoint_policy(
        &mut self,
        index: usize,
        o: &Observation,
        policy: &super::semantic_snapshot::SemanticPlayingCheckpointArenaPolicy,
    ) -> Result<ConditionalDecision, String> {
        self.choose_using(index, o, |o| policy.distribution(o))
    }
    pub(super) fn choose_phase_routed(
        &mut self,
        index: usize,
        o: &Observation,
        policy: &super::phase_routed::PhaseRoutedDevelopmentPolicy<'_>,
    ) -> Result<ConditionalDecision, String> {
        self.choose_using(index, o, |o| policy.distribution(o))
    }
    fn choose_using(
        &mut self,
        index: usize,
        o: &Observation,
        predict: impl FnOnce(&Observation) -> Result<NumericPolicyDistribution, String>,
    ) -> Result<ConditionalDecision, String> {
        let result = (|| {
            if self.failed
                || self.chosen
                || !self
                    .notified
                    .as_ref()
                    .is_some_and(|(seen, actor, trade, key)| {
                        *seen == index
                            && *actor == o.actor
                            && *trade == is_trade(o)
                            && key == &o.observation_key
                    })
                || observation_key(o)? != o.observation_key
            {
                return Err("Conditional choice requires one current callback".into());
            }
            let (raw, scores) = match o.phase {
                Phase::Setup
                    if matches!(
                        self.setup,
                        SetupRoute::CornFirst
                            | SetupRoute::SemanticPlayingCornFirst
                            | SetupRoute::NativeTeacherPlayingCornFirst(_)
                            | SetupRoute::NativeV7FundedUxmalPlayingCornFirst
                            | SetupRoute::CurrentSnapshotPlayingCornFirst
                            | SetupRoute::CheckpointPolicyPlayingCornFirst
                    ) =>
                {
                    let decision = crate::setup_policy::choose_move_with_weights(
                        o,
                        &HeuristicWeights::default(),
                    )?;
                    let index = o
                        .legal_actions
                        .iter()
                        .position(|a| a.r#move == decision.r#move)
                        .ok_or("Conditional Setup choice is not legal")?;
                    (index, vec![decision.score; o.legal_actions.len()])
                }
                Phase::Setup | Phase::Playing => {
                    let prediction = predict(o)?;
                    if prediction.logits.len() != o.legal_actions.len()
                        || prediction.probabilities.len() != o.legal_actions.len()
                        || prediction.logits.iter().any(|v| !v.is_finite())
                        || prediction.probabilities.iter().any(|p| !p.is_finite())
                    {
                        return Err("Conditional prediction shape/finite mismatch".into());
                    }
                    let raw = (1..prediction.logits.len()).fold(0, |best, i| {
                        if prediction.logits[i] > prediction.logits[best] {
                            i
                        } else {
                            best
                        }
                    });
                    (raw, prediction.logits.into_iter().map(f64::from).collect())
                }
                _ => return Err("Conditional policy requires Setup or Playing".into()),
            };
            let mut effective = raw;
            let trade = if is_trade(o) {
                let skip = skip_index(o)?;
                let bytes = public_bytes(o)?;
                let state_sha256 = {
                    use sha2::{Digest, Sha256};
                    format!("{:x}", Sha256::digest(&bytes))
                };
                let transition = self.history.transition(index, bytes, raw == skip)?;
                if transition.force_exit {
                    effective = skip;
                }
                self.actor = Some(o.actor);
                if effective == skip {
                    self.reset_episode();
                }
                Some(GuardedTradeTrace {
                    public_state_sha256: state_sha256,
                    first_seen_callback_index: transition.first_seen_index,
                    repeat_seen: transition.repeat_seen,
                    budget_reached: transition.budget_reached,
                    overridden: transition.force_exit,
                    trade_count_before: transition.count_before,
                    trade_count_after: transition.count_after,
                    episode_closed: effective == skip,
                })
            } else {
                None
            };
            // Keep the NN's single forward/raw index and Trade history intact.
            // The existing helper accepts only a wholly funded, non-pending
            // first-player prefix. No post-session move replacement is used.
            let opening = if matches!(self.setup, SetupRoute::NativeV7FundedUxmalPlayingCornFirst)
                && !is_trade(o)
            {
                crate::setup_policy::first_uxmal_opening(o)
                    .map(|action| {
                        effective = o
                            .legal_actions
                            .iter()
                            .position(|a| a.r#move == action.r#move)
                            .ok_or("Funded Uxmal selector returned no legal row")?;
                        Ok::<_, String>(OpeningSelection {
                            placement_count: usize::try_from(o.turn.count)
                                .map_err(|_| "Funded Uxmal placement count")?,
                            changed_move: effective != raw,
                        })
                    })
                    .transpose()?
            } else {
                None
            };
            if !scores[effective].is_finite() {
                return Err("Nonfinite conditional decision".into());
            }
            self.chosen = true;
            Ok(ConditionalDecision {
                decision: Decision {
                    actor: o.actor,
                    observation_key: o.observation_key.clone(),
                    r#move: o.legal_actions[effective].r#move.clone(),
                    policy_version: self.setup.policy_version().into(),
                    score: scores[effective],
                },
                trade,
                requested_index: raw,
                opening,
            })
        })();
        if result.is_err() {
            self.invalidate();
        }
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tzolkin_core::observation::observe;
    use tzolkin_core::{GearId, Pending, Resource, Task, TechnologyId, apply_move, create_game};

    fn session() -> GameSession {
        GameSession::with_config(ConditionalSetup::CornFirst, &TradeGuardConfig::default()).unwrap()
    }
    #[test]
    fn phase_routed_experts_keep_numeric_playing_and_trade_parity() {
        use super::super::{numeric, phase_routed, teacher};
        let mut parameters = vec![0.0; tzolkin_inference::policy::PARAMETER_COUNT];
        parameters[480] = 1.0;
        parameters[512 * 32 + 32] = 1.0;
        let model = serde_json::from_value::<
            crate::resource_cycle_teacher_training::SemanticSetupTeacherModel,
        >(serde_json::json!({"semanticSetupParameters": parameters}))
        .unwrap();
        model.validate().unwrap();
        let scalar = crate::kernel::Kernel::Scalar.resolve().unwrap();
        let numeric = numeric::tests::synthetic_policy();
        let mut guarded = numeric::tests::synthetic_guard();
        let mut game = GameSession::with_route(
            SetupRoute::PhaseRoutedDevelopment,
            &TradeGuardConfig::default(),
        )
        .unwrap();
        let predict = |o: &Observation| {
            phase_routed::route_distribution(
                o,
                |o| teacher::semantic_setup_distribution(&model, scalar, o),
                |o| numeric.distribution(o),
            )
        };
        let mut state = create_game(vec!["A".into(), "B".into(), "C".into()], 17, false).unwrap();
        let mut index = 0;
        while state.phase == Phase::Setup {
            let o = observe(&state, state.current_player).unwrap();
            game.on_callback(index, &o).unwrap();
            guarded.on_callback(index, &o).unwrap();
            let selected = if o.actor == 0 {
                let expected = teacher::semantic_setup_distribution(&model, scalar, &o).unwrap();
                let routed = phase_routed::route_distribution(
                    &o,
                    |_| {
                        Ok(NumericPolicyDistribution {
                            logits: expected.logits.clone(),
                            probabilities: expected.probabilities.clone(),
                        })
                    },
                    |_| panic!("Setup must not invoke numeric Playing"),
                )
                .unwrap();
                assert_eq!(routed.logits, expected.logits);
                let chosen = game.choose_using(index, &o, predict).unwrap().decision;
                assert_eq!(chosen.policy_version, phase_routed::POLICY_VERSION);
                chosen.r#move
            } else {
                o.legal_actions[0].r#move.clone()
            };
            state = apply_move(&state, selected).unwrap();
            index += 1;
        }
        state.current_player = 0;
        for resource in [Resource::Corn, Resource::Wood] {
            state.players[0].resources.insert(resource, 3);
        }
        state.pending = Some(Pending {
            title: "Trade".into(),
            task: Task::Trade,
            after: vec![],
        });
        let trade = observe(&state, 0).unwrap();
        let actual = phase_routed::route_distribution(
            &trade,
            |_| panic!("Playing must not invoke semantic Setup"),
            |o| numeric.distribution(o),
        )
        .unwrap();
        assert_eq!(actual.logits, numeric.distribution(&trade).unwrap().logits);
        // Exact repeats trigger the same exit, and another actor's callback
        // resets both episodes without restarting the game callback index.
        for other_actor in [false, false, true, false] {
            if other_actor {
                state.current_player = 1;
                let other = observe(&state, 1).unwrap();
                game.on_callback(index, &other).unwrap();
                guarded.on_callback(index, &other).unwrap();
                index += 1;
                state.current_player = 0;
            }
            game.on_callback(index, &trade).unwrap();
            guarded.on_callback(index, &trade).unwrap();
            let routed = game.choose_using(index, &trade, predict).unwrap();
            let reference = guarded.choose_move(index, &trade).unwrap();
            assert_eq!(routed.decision.r#move, reference.decision.r#move);
            assert_eq!(routed.decision.score, reference.decision.score);
            assert_eq!(
                serde_json::to_value(routed.trade).unwrap(),
                serde_json::to_value(reference.trade).unwrap()
            );
            index += 1;
        }
        assert!(
            game.choose_using(index - 1, &trade, |_| unreachable!())
                .is_err()
        );
        assert!(game.on_callback(index, &trade).is_err());
        let mut fresh = GameSession::with_route(
            SetupRoute::PhaseRoutedDevelopment,
            &TradeGuardConfig::default(),
        )
        .unwrap();
        fresh.on_callback(0, &trade).unwrap();
        assert!(
            !fresh
                .choose_using(0, &trade, predict)
                .unwrap()
                .trade
                .unwrap()
                .repeat_seen
        );
        let mut too_many = trade.clone();
        too_many.legal_actions =
            vec![trade.legal_actions[0].clone(); numeric::MAX_ROWS_PER_CALLBACK + 1];
        too_many.observation_key = observation_key(&too_many).unwrap();
        assert!(
            GameSession::with_route(
                SetupRoute::PhaseRoutedDevelopment,
                &TradeGuardConfig::default()
            )
            .unwrap()
            .on_callback(0, &too_many)
            .is_err()
        );
        let mut finished = trade.clone();
        finished.phase = Phase::Finished;
        assert!(
            phase_routed::route_distribution(&finished, |_| unreachable!(), |_| unreachable!())
                .is_err()
        );
    }
    #[test]
    fn conditional_callbacks_and_setup_are_explicit_and_fail_closed() {
        for scope in [
            None,
            Some(FACTORIZED_LABEL_SCOPE),
            Some(PAID_TECHNOLOGY_LABEL_SCOPE),
            Some(EXPLICIT_MARKET_ANYWHERE_LABEL_SCOPE),
            Some(OBSERVED_CORN_BEG_LABEL_SCOPE),
            Some(OBSERVED_PENDING_CHOICE_LABEL_SCOPE),
            Some(CAPPED_EXTRACTION_LABEL_SCOPE),
        ] {
            for action in [
                TypedAction::Place {
                    gear: GearId::Tikal,
                    corn_cost: 0,
                    discount: false,
                },
                TypedAction::Remove {
                    gear: GearId::Tikal,
                    position: 1,
                },
                TypedAction::FirstPlayer { corn_cost: 0 },
            ] {
                assert!(teacher_grammar_contains(scope, &action));
            }
            for source in [
                ActionSource::CurrentGear,
                ActionSource::PairedGear,
                ActionSource::TheologyAhead,
                ActionSource::TribeAhead,
                ActionSource::Anywhere,
            ] {
                let action = TypedAction::UseAction {
                    gear: GearId::Tikal,
                    position: 1,
                    source,
                    corn_cost: 0,
                };
                assert_eq!(
                    teacher_grammar_contains(scope, &action),
                    scope.is_some() && source == ActionSource::CurrentGear
                );
            }
            assert!(!teacher_grammar_contains(scope, &TypedAction::Skip));
            for action in [
                TypedAction::Technology {
                    technology: TechnologyId::Extraction,
                },
                TypedAction::Payment {
                    resources: [0, 1, 0, 0, 0],
                },
                TypedAction::Payment {
                    resources: [0, 1, 2, 3, 0],
                },
            ] {
                assert_eq!(
                    teacher_grammar_contains(scope, &action),
                    matches!(
                        scope,
                        Some(
                            PAID_TECHNOLOGY_LABEL_SCOPE
                                | EXPLICIT_MARKET_ANYWHERE_LABEL_SCOPE
                                | OBSERVED_CORN_BEG_LABEL_SCOPE
                                | OBSERVED_PENDING_CHOICE_LABEL_SCOPE
                                | CAPPED_EXTRACTION_LABEL_SCOPE
                        )
                    )
                );
                assert!(!teacher_grammar_contains(Some("unknown"), &action));
            }
            for resources in [[0; 5], [1, 1, 0, 0, 0], [0, 1, 0, 0, 1], [0, -1, 0, 0, 0]] {
                assert!(!teacher_grammar_contains(
                    scope,
                    &TypedAction::Payment { resources }
                ));
            }
            for resource in [
                Resource::Wood,
                Resource::Stone,
                Resource::Gold,
                Resource::Corn,
                Resource::Skull,
            ] {
                for buy in [false, true] {
                    assert_eq!(
                        teacher_grammar_contains(scope, &TypedAction::Trade { resource, buy }),
                        matches!(
                            scope,
                            Some(
                                EXPLICIT_MARKET_ANYWHERE_LABEL_SCOPE
                                    | OBSERVED_CORN_BEG_LABEL_SCOPE
                                    | OBSERVED_PENDING_CHOICE_LABEL_SCOPE
                                    | CAPPED_EXTRACTION_LABEL_SCOPE
                            )
                        ) && !buy
                            && matches!(
                                resource,
                                Resource::Wood | Resource::Stone | Resource::Gold
                            )
                    );
                }
            }
            for corn_cost in [0, 1, 2] {
                assert_eq!(
                    teacher_grammar_contains(
                        scope,
                        &TypedAction::UseAction {
                            gear: GearId::Uxmal,
                            position: 3,
                            source: ActionSource::Anywhere,
                            corn_cost
                        }
                    ),
                    matches!(
                        scope,
                        Some(
                            EXPLICIT_MARKET_ANYWHERE_LABEL_SCOPE
                                | OBSERVED_CORN_BEG_LABEL_SCOPE
                                | OBSERVED_PENDING_CHOICE_LABEL_SCOPE
                                | CAPPED_EXTRACTION_LABEL_SCOPE
                        )
                    ) && corn_cost == 1
                );
            }
            assert_eq!(
                teacher_grammar_contains(scope, &TypedAction::Beg),
                matches!(
                    scope,
                    Some(
                        OBSERVED_CORN_BEG_LABEL_SCOPE
                            | OBSERVED_PENDING_CHOICE_LABEL_SCOPE
                            | CAPPED_EXTRACTION_LABEL_SCOPE
                    )
                )
            );
            for direction in [-1, 1] {
                assert_eq!(
                    teacher_grammar_contains(
                        scope,
                        &TypedAction::Temple {
                            temple: tzolkin_core::TempleId::Chaac,
                            direction
                        }
                    ),
                    (matches!(
                        scope,
                        Some(
                            OBSERVED_CORN_BEG_LABEL_SCOPE
                                | OBSERVED_PENDING_CHOICE_LABEL_SCOPE
                                | CAPPED_EXTRACTION_LABEL_SCOPE
                        )
                    ) && direction == -1)
                        || (matches!(
                            scope,
                            Some(
                                OBSERVED_PENDING_CHOICE_LABEL_SCOPE | CAPPED_EXTRACTION_LABEL_SCOPE
                            )
                        ) && direction == 1)
                );
            }
            for renovation in [None, Some("synthetic".into())] {
                assert_eq!(
                    teacher_grammar_contains(
                        scope,
                        &TypedAction::Monument {
                            id: "synthetic".into(),
                            cost: [0, 1, 0, 0, 0],
                            renovation: renovation.clone(),
                        }
                    ),
                    matches!(
                        scope,
                        Some(OBSERVED_PENDING_CHOICE_LABEL_SCOPE | CAPPED_EXTRACTION_LABEL_SCOPE)
                    ) && renovation.is_none()
                );
            }
            assert!(!teacher_grammar_contains(
                scope,
                &TypedAction::Build {
                    id: "synthetic".into(),
                    cost: [0; 5],
                    architecture: false,
                    discount_resource: None,
                    renovation: None,
                }
            ));
        }
        assert!(!teacher_grammar_contains(
            Some("unknown"),
            &TypedAction::FirstPlayer { corn_cost: 0 }
        ));
        let legacy = serde_json::json!({
            "task": "ConditionalBasicHumanPolicy", "inputContract": "synthetic",
            "featureSchema": 1, "featureCount": 512, "rulesVersion": 1,
            "catalogHash": "synthetic", "sourceAuthenticated": false,
            "sourceQualified": false, "verifiedNativePolicy": false, "tables": [],
            "samples": [0, 0, 0], "candidates": [0, 0, 0], "fingerprint": "synthetic"
        });
        // Structural serialization only; this is not a qualified dataset identity.
        let mut identity: crate::human_policy_training::ConditionalDatasetIdentity =
            serde_json::from_value(legacy.clone()).unwrap();
        assert!(identity.label_scope.is_none());
        assert_eq!(serde_json::to_value(&identity).unwrap(), legacy);
        identity.label_scope = Some(FACTORIZED_LABEL_SCOPE.into());
        let expanded = serde_json::to_value(&identity).unwrap();
        assert_eq!(expanded["labelScope"], FACTORIZED_LABEL_SCOPE);
        assert_eq!(
            serde_json::from_value::<crate::human_policy_training::ConditionalDatasetIdentity>(
                expanded
            )
            .unwrap(),
            identity
        );
        let invalid = TradeGuardConfig {
            max_trades_per_episode: 0,
            ..TradeGuardConfig::default()
        };
        assert!(GameSession::with_config(ConditionalSetup::CornFirst, &invalid).is_err());
        let state = create_game(vec!["A".into(), "B".into(), "C".into()], 17, false).unwrap();
        let o = observe(&state, state.current_player).unwrap();
        let mut semantic_setup = GameSession::with_route(
            SetupRoute::SemanticPlayingCornFirst,
            &TradeGuardConfig::default(),
        )
        .unwrap();
        semantic_setup.on_callback(0, &o).unwrap();
        let semantic_setup_choice = semantic_setup
            .choose_using(0, &o, |_| {
                panic!("Schema-4 Playing inference must not score Setup")
            })
            .unwrap()
            .decision;
        let old_setup_choice =
            crate::setup_policy::choose_move_with_weights(&o, &HeuristicWeights::default())
                .unwrap();
        assert_eq!(semantic_setup_choice.r#move, old_setup_choice.r#move);
        assert_eq!(semantic_setup_choice.score, old_setup_choice.score);
        assert_eq!(
            semantic_setup_choice.policy_version,
            SEMANTIC_PLAYING_POLICY_VERSION
        );
        // Exercise the actual candidate encoder and numeric borrow without
        // constructing an audited checkpoint/source or granting an Arena role.
        use tzolkin_inference::policy::{
            BorrowedPolicy, BorrowedSemanticSetupPolicy, PARAMETER_COUNT,
        };
        let mut parameters = vec![0.0; PARAMETER_COUNT];
        parameters[480] = 1.0;
        parameters[512 * 32 + 32] = 1.0;
        let model = serde_json::from_value::<
            crate::resource_cycle_teacher_training::SemanticSetupTeacherModel,
        >(serde_json::json!({"semanticSetupParameters": parameters}))
        .unwrap();
        model.validate().unwrap();
        assert!(
            serde_json::from_value::<
                crate::resource_cycle_teacher_training::SemanticSetupTeacherModel,
            >(serde_json::json!({"parameters": model.parameters()}))
            .is_err()
        );
        let scalar = crate::kernel::Kernel::Scalar.resolve().unwrap();
        let encoded = |encoder: crate::features::FeatureEncoder<'_>| {
            (0..o.legal_actions.len())
                .map(|i| encoder.encode_legal_tagged(i).unwrap())
                .collect::<Vec<_>>()
        };
        let semantic_rows =
            encoded(crate::features::FeatureEncoder::new_semantic_setup(&o).unwrap());
        let public_rows = encoded(crate::features::FeatureEncoder::new_public(&o).unwrap());
        let semantic =
            BorrowedSemanticSetupPolicy::from_validated_parameters(model.parameters()).unwrap();
        let public = BorrowedPolicy::from_validated_parameters(model.parameters()).unwrap();
        assert!(semantic.predict(scalar, &public_rows).is_err());
        assert!(public.predict(scalar, &semantic_rows).is_err());
        let prediction =
            super::super::teacher::semantic_setup_distribution(&model, scalar, &o).unwrap();
        let expected = semantic.predict(scalar, &semantic_rows).unwrap();
        assert_eq!(prediction.logits, expected.logits);
        assert_eq!(prediction.probabilities, expected.probabilities);
        assert_ne!(
            prediction.logits,
            public.predict(scalar, &public_rows).unwrap().logits
        );
        let mut semantic_game = GameSession::with_route(
            SetupRoute::NumericAllPhase(super::super::teacher::SEMANTIC_SETUP_POLICY_VERSION),
            &TradeGuardConfig::default(),
        )
        .unwrap();
        semantic_game.on_callback(0, &o).unwrap();
        let decision = semantic_game
            .choose_using(0, &o, |o| {
                super::super::teacher::semantic_setup_distribution(&model, scalar, o)
            })
            .unwrap()
            .decision;
        let first_max = (1..prediction.logits.len()).fold(0, |best, i| {
            if prediction.logits[i] > prediction.logits[best] {
                i
            } else {
                best
            }
        });
        assert_eq!(decision.r#move, o.legal_actions[first_max].r#move);
        assert_eq!(decision.score, f64::from(prediction.logits[first_max]));
        assert_eq!(
            decision.policy_version,
            super::super::teacher::SEMANTIC_SETUP_POLICY_VERSION
        );
        assert_eq!(
            serde_json::to_value(
                super::super::teacher::SemanticSetupTeacherInputProfile::SemanticSetupAllPhase
            )
            .unwrap(),
            "semanticSetupAllPhase"
        );
        let mut too_many = o.clone();
        too_many.legal_actions =
            vec![o.legal_actions[0].clone(); super::super::numeric::MAX_ROWS_PER_CALLBACK + 1];
        assert_eq!(
            super::super::teacher::semantic_setup_distribution(&model, scalar, &too_many)
                .err()
                .unwrap(),
            "Semantic Setup teacher candidate-row cap exceeded before prediction"
        );
        let mut game = session();
        game.on_callback(0, &o).unwrap();
        let selected = game
            .choose_using(0, &o, |_| panic!("Setup must not predict Human policy"))
            .unwrap()
            .decision;
        assert_eq!(
            selected.r#move,
            crate::setup_policy::choose_move_with_weights(&o, &HeuristicWeights::default())
                .unwrap()
                .r#move
        );
        assert_eq!(selected.policy_version, POLICY_VERSION);
        let mut native = GameSession::with_route(
            SetupRoute::NativeTeacherPlayingCornFirst(TeacherProfile::V5),
            &TradeGuardConfig::default(),
        )
        .unwrap();
        native.on_callback(0, &o).unwrap();
        let selected_native = native
            .choose_using(0, &o, |_| {
                panic!("Native Playing teacher must not predict Setup")
            })
            .unwrap()
            .decision;
        assert_eq!(selected_native.r#move, selected.r#move);
        assert_eq!(
            selected_native.policy_version,
            super::super::teacher::SEMANTIC_PLAYING_POLICY_VERSION
        );
        let mut composite_setup = GameSession::with_route(
            SetupRoute::NativeV7FundedUxmalPlayingCornFirst,
            &TradeGuardConfig::default(),
        )
        .unwrap();
        composite_setup.on_callback(0, &o).unwrap();
        let selected_composite = composite_setup
            .choose_using(0, &o, |_| panic!("Composite must not predict Setup"))
            .unwrap();
        assert_eq!(selected_composite.decision.r#move, selected.r#move);
        assert_eq!(selected_composite.decision.score, selected.score);
        assert!(selected_composite.opening.is_none());
        assert!(
            super::super::teacher::semantic_playing_distribution(model.parameters(), scalar, &o)
                .is_err()
        );
        let corn_index = o
            .legal_actions
            .iter()
            .position(|a| a.r#move == selected.r#move)
            .unwrap();
        assert!(o.legal_actions.len() > 1);
        let target = (corn_index + 1) % o.legal_actions.len();
        for version in [
            MIXED_POLICY_VERSION,
            COUNT_ZERO_POLICY_VERSION,
            super::super::teacher::POLICY_VERSION,
            super::super::teacher::SEMANTIC_SETUP_POLICY_VERSION,
        ] {
            let mut numeric = GameSession::with_route(
                SetupRoute::NumericAllPhase(version),
                &TradeGuardConfig::default(),
            )
            .unwrap();
            numeric.on_callback(0, &o).unwrap();
            let mixed = numeric
                .choose_using(0, &o, |o| {
                    let mut logits = vec![0.; o.legal_actions.len()];
                    logits[target] = 1.;
                    let mut probabilities = vec![0.; o.legal_actions.len()];
                    probabilities[target] = 1.;
                    Ok(NumericPolicyDistribution {
                        logits,
                        probabilities,
                    })
                })
                .unwrap()
                .decision;
            assert_eq!(mixed.r#move, o.legal_actions[target].r#move);
            assert_eq!(mixed.policy_version, version);
            assert!(numeric.choose_using(0, &o, |_| unreachable!()).is_err());
        }
        let zero = crate::conditional_human_rl_update::tests::owner();
        require_fresh_count_zero(zero.identity()).unwrap();
        let no_change_tip = crate::conditional_human_rl_update::checked_tip_identity(
            zero.identity(),
            zero.actor_parameters(),
            zero.critic_parameters(),
            4,
            false,
        )
        .unwrap();
        assert_eq!(no_change_tip.mixed_update_count(), 0);
        assert!(require_fresh_count_zero(&no_change_tip).is_err());
        assert!(ConditionalCountZeroArenaPolicy::fixture(&zero, invalid).is_err());
        assert!(game.choose_using(0, &o, |_| unreachable!()).is_err());
        assert!(game.on_callback(1, &o).is_err());
        let mut stale = session();
        assert!(stale.on_callback(1, &o).is_err());
        let mut altered = session();
        altered.on_callback(0, &o).unwrap();
        let mut changed = o.clone();
        changed.players[o.actor].resources[0] += 1;
        assert!(
            altered
                .choose_using(0, &changed, |_| unreachable!())
                .is_err()
        );
        assert!(altered.choose_using(0, &o, |_| unreachable!()).is_err());
        let mut unsupported = o.clone();
        unsupported.additional_buildings = true;
        assert!(session().on_callback(0, &unsupported).is_err());

        // A shared outer Setup selector notifies this session without asking
        // it to predict or choose. Playing must still accept the next exact
        // callback and enforce one choice, retaining its own guard history.
        let mut state = create_game(vec!["A".into(), "B".into(), "C".into()], 17, false).unwrap();
        let mut routed = session();
        for index in 0..3 {
            let o = observe(&state, state.current_player).unwrap();
            assert_eq!(o.phase, Phase::Setup);
            routed.on_callback(index, &o).unwrap();
            let decision =
                crate::setup_policy::choose_move_with_weights(&o, &HeuristicWeights::default())
                    .unwrap();
            state = apply_move(&state, decision.r#move).unwrap();
        }
        let o = observe(&state, state.current_player).unwrap();
        assert_eq!(o.phase, Phase::Playing);
        routed.on_callback(3, &o).unwrap();
        let decision = routed
            .choose_using(3, &o, |o| {
                Ok(NumericPolicyDistribution {
                    logits: vec![0.; o.legal_actions.len()],
                    probabilities: vec![1. / o.legal_actions.len() as f32; o.legal_actions.len()],
                })
            })
            .unwrap();
        assert_eq!(decision.decision.policy_version, POLICY_VERSION);
        assert_eq!(decision.decision.r#move, o.legal_actions[0].r#move);
        assert!(routed.choose_using(3, &o, |_| unreachable!()).is_err());
        // Pure numeric/callback parity only; no loaded checkpoint or source admission.
        let playing =
            tzolkin_inference::policy::BorrowedSemanticPlayingPolicy::from_validated_parameters(
                model.parameters(),
            )
            .unwrap();
        let encoder = crate::features::FeatureEncoder::new_semantic_playing(&o).unwrap();
        let rows = (0..o.legal_actions.len())
            .map(|i| encoder.encode_legal_tagged(i).unwrap())
            .collect::<Vec<_>>();
        let prediction = playing.predict(scalar, &rows).unwrap();
        assert!(public.predict(scalar, &rows).is_err());
        let expected_index = (1..prediction.logits.len()).fold(0, |best, i| {
            if prediction.logits[i] > prediction.logits[best] {
                i
            } else {
                best
            }
        });
        let mut semantic_game = GameSession::with_route(
            SetupRoute::SemanticPlayingCornFirst,
            &TradeGuardConfig::default(),
        )
        .unwrap();
        semantic_game.on_callback(0, &o).unwrap();
        let decision = semantic_game
            .choose_using(0, &o, |_| Ok(prediction))
            .unwrap()
            .decision;
        assert_eq!(decision.r#move, o.legal_actions[expected_index].r#move);
        assert_eq!(decision.policy_version, SEMANTIC_PLAYING_POLICY_VERSION);
        assert!(
            semantic_game
                .choose_using(0, &o, |_| unreachable!())
                .is_err()
        );
        let native_prediction =
            super::super::teacher::semantic_playing_distribution(model.parameters(), scalar, &o)
                .unwrap();
        let expected = playing.predict(scalar, &rows).unwrap();
        assert_eq!(native_prediction.logits, expected.logits);
        assert_eq!(native_prediction.probabilities, expected.probabilities);
        let mut native = GameSession::with_route(
            SetupRoute::NativeTeacherPlayingCornFirst(TeacherProfile::V5),
            &TradeGuardConfig::default(),
        )
        .unwrap();
        native.on_callback(0, &o).unwrap();
        let decision = native
            .choose_using(0, &o, |_| Ok(native_prediction))
            .unwrap()
            .decision;
        assert_eq!(decision.r#move, o.legal_actions[expected_index].r#move);
        assert_eq!(
            decision.policy_version,
            super::super::teacher::SEMANTIC_PLAYING_POLICY_VERSION
        );
        assert!(native.choose_using(0, &o, |_| unreachable!()).is_err());

        // Bounded real Core prefixes, with a deliberately contrary finite NN
        // response. Both routes invoke the predictor once and keep its raw row.
        let playing_with = |players: usize, seed, pair: &str| {
            let mut state = create_game(
                (0..players).map(|i| format!("Seat {i}")).collect(),
                seed,
                false,
            )
            .unwrap();
            state = apply_move(
                &state,
                tzolkin_core::GameMove::Choose {
                    choice_id: pair.into(),
                },
            )
            .unwrap();
            while state.phase == Phase::Setup {
                let o = observe(&state, state.current_player).unwrap();
                state = apply_move(
                    &state,
                    crate::setup_policy::choose_move_with_weights(&o, &HeuristicWeights::default())
                        .unwrap()
                        .r#move,
                )
                .unwrap();
            }
            state
        };
        let predict = |o: &Observation| {
            let target = o
                .legal_actions
                .iter()
                .position(|a| {
                    matches!(
                        a.r#move,
                        tzolkin_core::GameMove::Place {
                            gear: GearId::Yaxchilan
                        }
                    )
                })
                .unwrap_or(0);
            let mut logits = vec![-2.5; o.legal_actions.len()];
            logits[target] = 1.25;
            let mut probabilities = vec![0.; o.legal_actions.len()];
            probabilities[target] = 1.;
            Ok(NumericPolicyDistribution {
                logits,
                probabilities,
            })
        };
        let choose = |o: &Observation, composite| {
            let route = if composite {
                SetupRoute::NativeV7FundedUxmalPlayingCornFirst
            } else {
                SetupRoute::NativeTeacherPlayingCornFirst(TeacherProfile::V7)
            };
            let mut game = GameSession::with_route(route, &TradeGuardConfig::default()).unwrap();
            game.on_callback(0, o).unwrap();
            let calls = std::cell::Cell::new(0);
            let selected = game
                .choose_using(0, o, |o| {
                    calls.set(calls.get() + 1);
                    predict(o)
                })
                .unwrap();
            assert_eq!(calls.get(), 1);
            assert!(game.choose_using(0, o, |_| unreachable!()).is_err());
            selected
        };
        for players in [3, 4] {
            for (seed, pair, workers) in [(3, "wealth:w10:w11", 3), (1, "wealth:w15:w21", 4)] {
                let mut state = playing_with(players, seed, pair);
                assert_eq!(state.players[0].workers, workers);
                let mut counts = super::super::numeric::FundedUxmalOpeningDiagnostics::default();
                for count in 0..=3 {
                    let o = observe(&state, state.current_player).unwrap();
                    let raw = choose(&o, false);
                    let composite = choose(&o, true);
                    assert_eq!(raw.requested_index, composite.requested_index);
                    assert!(raw.opening.is_none());
                    let opening = composite.opening.unwrap();
                    counts.record(&opening).unwrap();
                    assert_eq!(opening.placement_count, count);
                    assert_eq!(
                        opening.changed_move,
                        raw.decision.r#move != composite.decision.r#move
                    );
                    assert!(composite.trade.is_none());
                    assert_eq!(
                        composite.decision.policy_version,
                        super::super::teacher::SEMANTIC_PLAYING_V7_FUNDED_UXMAL_POLICY_VERSION
                    );
                    assert_eq!(
                        composite.decision.score,
                        if opening.changed_move { -2.5 } else { 1.25 }
                    );
                    assert_eq!(
                        composite.decision.r#move,
                        if count < 3 {
                            tzolkin_core::GameMove::Place {
                                gear: GearId::Uxmal,
                            }
                        } else {
                            tzolkin_core::GameMove::EndTurn {
                                double_advance: None,
                            }
                        }
                    );
                    state = apply_move(&state, composite.decision.r#move).unwrap();
                }
                assert_eq!(counts.eligible_choices, 4);
                assert_eq!(counts.eligible_by_placement_count, [1; 4]);
                assert_eq!(
                    counts.changed_moves,
                    counts.changed_by_placement_count.iter().sum::<usize>()
                );
                assert_eq!(tzolkin_core::available_workers(&state, 0), workers - 3);
                let later = observe(&state, state.current_player).unwrap();
                assert!(choose(&later, true).opening.is_none());
            }
        }
        let funded = playing_with(3, 3, "wealth:w10:w11");
        let mut poor = funded.clone();
        poor.players[0].resources.insert(Resource::Corn, 0);
        let nonprefix = apply_move(
            &funded,
            tzolkin_core::GameMove::Place {
                gear: GearId::Yaxchilan,
            },
        )
        .unwrap();
        let mut pending = funded.clone();
        pending.pending = Some(Pending {
            title: "Resource".into(),
            task: Task::Resource { remaining: 1 },
            after: vec![],
        });
        for state in [
            poor,
            nonprefix,
            pending,
            playing_with(3, 17, "wealth:w01:w19"),
        ] {
            let o = observe(&state, state.current_player).unwrap();
            let raw = choose(&o, false);
            let composite = choose(&o, true);
            assert!(composite.opening.is_none());
            assert_eq!(raw.requested_index, composite.requested_index);
            assert_eq!(raw.decision.r#move, composite.decision.r#move);
            assert_eq!(raw.decision.score, composite.decision.score);
            assert_eq!(
                serde_json::to_value(raw.trade).unwrap(),
                serde_json::to_value(composite.trade).unwrap()
            );
        }
        let mut trading = funded;
        trading.players[0].resources.insert(Resource::Wood, 3);
        trading.pending = Some(Pending {
            title: "Trade".into(),
            task: Task::Trade,
            after: vec![],
        });
        let o = observe(&trading, trading.current_player).unwrap();
        let mut raw = GameSession::with_route(
            SetupRoute::NativeTeacherPlayingCornFirst(TeacherProfile::V7),
            &TradeGuardConfig::default(),
        )
        .unwrap();
        let mut composite = GameSession::with_route(
            SetupRoute::NativeV7FundedUxmalPlayingCornFirst,
            &TradeGuardConfig::default(),
        )
        .unwrap();
        for index in 0..2 {
            raw.on_callback(index, &o).unwrap();
            composite.on_callback(index, &o).unwrap();
            let a = raw.choose_using(index, &o, predict).unwrap();
            let b = composite.choose_using(index, &o, predict).unwrap();
            assert!(b.opening.is_none());
            assert_eq!(a.requested_index, b.requested_index);
            assert_eq!(a.decision.r#move, b.decision.r#move);
            assert_eq!(
                serde_json::to_value(a.trade).unwrap(),
                serde_json::to_value(b.trade).unwrap()
            );
        }
        let legacy =
            serde_json::to_value(super::super::numeric::GuardDiagnostics::default()).unwrap();
        assert!(legacy.get("fundedUxmalOpening").is_none());
    }
    #[test]
    fn conditional_trade_uses_shared_history_and_resets_on_other_actor() {
        let mut state = create_game(vec!["A".into(), "B".into(), "C".into()], 17, false).unwrap();
        while state.phase == Phase::Setup {
            let o = observe(&state, state.current_player).unwrap();
            state = apply_move(&state, o.legal_actions[0].r#move.clone()).unwrap();
        }
        state.players[state.current_player]
            .resources
            .insert(Resource::Corn, 3);
        state.players[state.current_player]
            .resources
            .insert(Resource::Wood, 3);
        state.pending = Some(Pending {
            title: "Trade".into(),
            task: Task::Trade,
            after: vec![],
        });
        let o = observe(&state, state.current_player).unwrap();
        let predict = |o: &Observation| {
            Ok(NumericPolicyDistribution {
                logits: vec![0.; o.legal_actions.len()],
                probabilities: vec![1. / o.legal_actions.len() as f32; o.legal_actions.len()],
            })
        };
        for route in [
            SetupRoute::CornFirst,
            SetupRoute::SemanticPlayingCornFirst,
            SetupRoute::NativeTeacherPlayingCornFirst(TeacherProfile::V5),
            SetupRoute::NumericAllPhase(MIXED_POLICY_VERSION),
            SetupRoute::NumericAllPhase(COUNT_ZERO_POLICY_VERSION),
            SetupRoute::NumericAllPhase(super::super::teacher::POLICY_VERSION),
            SetupRoute::NumericAllPhase(super::super::teacher::SEMANTIC_SETUP_POLICY_VERSION),
        ] {
            state.current_player = o.actor;
            let mut game = GameSession::with_route(route, &TradeGuardConfig::default()).unwrap();
            game.on_callback(0, &o).unwrap();
            let first = game.choose_using(0, &o, predict).unwrap();
            assert_eq!(first.decision.r#move, o.legal_actions[0].r#move);
            let mut contexts = ConditionalChoiceDiagnostics::new();
            contexts
                .record(
                    &o,
                    first.requested_index,
                    0,
                    first.trade.as_ref(),
                    |action| teacher_grammar_contains(Some(CAPPED_EXTRACTION_LABEL_SCOPE), action),
                )
                .unwrap();
            assert!(!first.trade.unwrap().overridden);
            game.on_callback(1, &o).unwrap();
            let repeated = game.choose_using(1, &o, predict).unwrap();
            assert_eq!(
                repeated.decision.r#move,
                o.legal_actions[skip_index(&o).unwrap()].r#move
            );
            contexts
                .record(
                    &o,
                    repeated.requested_index,
                    skip_index(&o).unwrap(),
                    repeated.trade.as_ref(),
                    |action| teacher_grammar_contains(Some(CAPPED_EXTRACTION_LABEL_SCOPE), action),
                )
                .unwrap();
            assert_eq!(contexts.choices, 2);
            assert_eq!(
                contexts
                    .contexts
                    .iter()
                    .map(|row| row.choices)
                    .sum::<usize>(),
                2
            );
            let forced = contexts
                .contexts
                .iter()
                .find(|row| row.guard_overrides == 1)
                .unwrap();
            assert_eq!(forced.requested_kind, "trade");
            assert_eq!(forced.effective_kind, "skip");
            assert_eq!(forced.repeated_trade_exits, 1);
            assert_eq!(forced.round_bin, 0);
            assert!(forced.legal_kinds.contains(&"trade") && forced.legal_kinds.contains(&"skip"));
            assert!(forced.effective_outside_teacher_grammar);
            let mut invalid = o.clone();
            invalid.round = 28;
            assert!(contexts.record(&invalid, 0, 0, None, |_| false).is_err());
            assert!(
                contexts
                    .record(&o, o.legal_actions.len(), 0, None, |_| false)
                    .is_err()
            );
            assert_eq!(contexts.choices, 2);
            assert!(repeated.trade.unwrap().repeat_seen);
            game.on_callback(2, &o).unwrap();
            assert!(
                !game
                    .choose_using(2, &o, predict)
                    .unwrap()
                    .trade
                    .unwrap()
                    .repeat_seen
            );
            state.current_player = (state.current_player + 1) % 3;
            let other = observe(&state, state.current_player).unwrap();
            game.on_callback(3, &other).unwrap();
            game.on_callback(4, &o).unwrap();
            assert!(
                !game
                    .choose_using(4, &o, predict)
                    .unwrap()
                    .trade
                    .unwrap()
                    .repeat_seen
            );
            let mut fresh = GameSession::with_route(route, &TradeGuardConfig::default()).unwrap();
            fresh.on_callback(0, &o).unwrap();
            assert!(
                !fresh
                    .choose_using(0, &o, predict)
                    .unwrap()
                    .trade
                    .unwrap()
                    .repeat_seen
            );
            let bound = TradeGuardConfig {
                max_trades_per_episode: 1,
                ..TradeGuardConfig::default()
            };
            let mut bounded = GameSession::with_route(route, &bound).unwrap();
            bounded.on_callback(0, &o).unwrap();
            assert!(
                !bounded
                    .choose_using(0, &o, predict)
                    .unwrap()
                    .trade
                    .unwrap()
                    .budget_reached
            );
            bounded.on_callback(1, &o).unwrap();
            let forced = bounded.choose_using(1, &o, predict).unwrap();
            assert!(forced.trade.unwrap().budget_reached);
            assert_eq!(
                forced.decision.r#move,
                o.legal_actions[skip_index(&o).unwrap()].r#move
            );
        }
    }
}
