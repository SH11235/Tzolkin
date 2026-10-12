//! Conditional Human origin and explicit Setup/Playing composition for Dev only.
pub use super::numeric::{run_conditional_mixed_validation, run_conditional_validation};
use crate::Decision;
use crate::conditional_human_rl_update::{
    ConditionalMixedIdentity, ConditionalMixedOwner, ConditionalMixedPolicyHandle,
};
use crate::dataset::DatasetSplit;
use crate::human_policy_dataset::{FACTORIZED_LABEL_SCOPE, PAID_TECHNOLOGY_LABEL_SCOPE};
use crate::human_policy_training::{ConditionalDatasetIdentity, PreparedConditionalHumanPolicy};
use crate::policy::HeuristicWeights;
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
pub const MIXED_POLICY_VERSION: &str = "conditional-human-current-allphase-guarded-argmax-v1";
pub const MIXED_ARENA_SCHEMA: &str = "tzolkin-conditional-human-mixed-development-arena-v1";

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
            },
        })
    }
    pub fn identity(&self) -> &ConditionalIdentity {
        &self.identity
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

// Category membership is diagnostic only: it does not check observed captions,
// an uncapped technology/pending payment, or positive coverage in actual Train rows.
fn teacher_grammar_contains(label_scope: Option<&str>, action: &TypedAction) -> bool {
    match label_scope {
        None | Some(FACTORIZED_LABEL_SCOPE) | Some(PAID_TECHNOLOGY_LABEL_SCOPE) => {
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
                || (label_scope == Some(PAID_TECHNOLOGY_LABEL_SCOPE)
                    && match action {
                        TypedAction::Technology { .. } => true,
                        TypedAction::Payment { resources } => {
                            resources[0] == 0
                                && resources[4] == 0
                                && resources[1..4].iter().all(|&value| value >= 0)
                                && resources[1..4].iter().any(|&value| value > 0)
                        }
                        _ => false,
                    })
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
    /// Counts effective choices after the guard, so guard-forced Skip is outside.
    /// Membership does not establish positive Train coverage of that category.
    /// For a mixed RL actor this still refers to its original BC teacher grammar,
    /// not the action categories it subsequently encountered during RL.
    pub outside_teacher_grammar_choices: usize,
}

pub(super) struct ConditionalDecision {
    pub decision: Decision,
    pub trade: Option<GuardedTradeTrace>,
}

// Keep the legacy public Setup composition separate from all-phase actor inference.
#[derive(Clone, Copy)]
enum SetupRoute {
    CornFirst,
    NumericAllPhase,
}
impl SetupRoute {
    fn policy_version(self) -> &'static str {
        match self {
            Self::CornFirst => POLICY_VERSION,
            Self::NumericAllPhase => MIXED_POLICY_VERSION,
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
    pub(super) fn new_mixed(policy: &ConditionalMixedArenaPolicy<'_>) -> Self {
        Self::with_route(SetupRoute::NumericAllPhase, &policy.identity.guard)
            .expect("conditional mixed constructor validated guard configuration")
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
    pub(super) fn choose_mixed(
        &mut self,
        index: usize,
        o: &Observation,
        policy: &ConditionalMixedArenaPolicy<'_>,
    ) -> Result<ConditionalDecision, String> {
        self.choose_using(index, o, |o| policy.handle.numeric_distribution(o))
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
                Phase::Setup if matches!(self.setup, SetupRoute::CornFirst) => {
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
    fn conditional_callbacks_and_setup_are_explicit_and_fail_closed() {
        for scope in [
            None,
            Some(FACTORIZED_LABEL_SCOPE),
            Some(PAID_TECHNOLOGY_LABEL_SCOPE),
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
                    scope == Some(PAID_TECHNOLOGY_LABEL_SCOPE)
                );
                assert!(!teacher_grammar_contains(Some("unknown"), &action));
            }
            for resources in [[0; 5], [1, 1, 0, 0, 0], [0, 1, 0, 0, 1], [0, -1, 0, 0, 0]] {
                assert!(!teacher_grammar_contains(
                    scope,
                    &TypedAction::Payment { resources }
                ));
            }
            assert!(!teacher_grammar_contains(
                scope,
                &TypedAction::Trade {
                    resource: Resource::Wood,
                    buy: false,
                }
            ));
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
        let corn_index = o
            .legal_actions
            .iter()
            .position(|a| a.r#move == selected.r#move)
            .unwrap();
        assert!(o.legal_actions.len() > 1);
        let target = (corn_index + 1) % o.legal_actions.len();
        let mut numeric =
            GameSession::with_route(SetupRoute::NumericAllPhase, &TradeGuardConfig::default())
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
        assert_eq!(mixed.policy_version, MIXED_POLICY_VERSION);
        assert!(numeric.choose_using(0, &o, |_| unreachable!()).is_err());
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
        let mut game = session();
        game.on_callback(0, &o).unwrap();
        let first = game.choose_using(0, &o, predict).unwrap();
        assert_eq!(first.decision.r#move, o.legal_actions[0].r#move);
        assert!(!first.trade.unwrap().overridden);
        game.on_callback(1, &o).unwrap();
        let repeated = game.choose_using(1, &o, predict).unwrap();
        assert_eq!(
            repeated.decision.r#move,
            o.legal_actions[skip_index(&o).unwrap()].r#move
        );
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
        let mut fresh = session();
        fresh.on_callback(0, &o).unwrap();
        assert!(
            !fresh
                .choose_using(0, &o, predict)
                .unwrap()
                .trade
                .unwrap()
                .repeat_seen
        );
    }
}
