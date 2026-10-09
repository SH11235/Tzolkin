//! Opt-in, game-local public Trade exit guard. This is not a producer authenticator.
use std::collections::BTreeMap;
use std::io::{self, Write};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use tzolkin_core::observation::{
    LegalAction, Observation, PublicExpansion, PublicPlayer, TypedAction,
};
use tzolkin_core::tribes::TribeId;
use tzolkin_core::{GameMove, GearId, GearWorker, JungleBox, Phase, Resource, Task, Turn};

use crate::Decision;
use crate::public_model::{LoadedPublicPolicy, PublicPolicyDistribution, ValueValidity};
use crate::replay::SeatPolicy;

pub const GUARD_VERSION: &str = "public-trade-cycle-guard-v1";
pub const POLICY_VERSION: &str = "learned-public-policy-trade-guard-v1";
pub const PROJECTION_VERSION: &str = "public-trading-state-v1";
pub const MAX_PUBLIC_STATE_BYTES: usize = 131_072;
pub const MAX_TRADES: usize = 64;

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct TradeGuardConfig {
    pub schema: u32,
    pub algorithm_version: String,
    pub state_projection_version: String,
    pub repeat_state_policy: String,
    pub max_trades_per_episode: usize,
    pub max_public_state_bytes: usize,
}
impl Default for TradeGuardConfig {
    fn default() -> Self {
        Self {
            schema: 1,
            algorithm_version: GUARD_VERSION.into(),
            state_projection_version: PROJECTION_VERSION.into(),
            repeat_state_policy: "secondVisitExit".into(),
            max_trades_per_episode: MAX_TRADES,
            max_public_state_bytes: MAX_PUBLIC_STATE_BYTES,
        }
    }
}
impl TradeGuardConfig {
    pub fn validate(&self) -> Result<(), String> {
        if self.schema != 1
            || self.algorithm_version != GUARD_VERSION
            || self.state_projection_version != PROJECTION_VERSION
            || self.repeat_state_policy != "secondVisitExit"
            || !(1..=MAX_TRADES).contains(&self.max_trades_per_episode)
            || self.max_public_state_bytes != MAX_PUBLIC_STATE_BYTES
        {
            return Err("Unsupported public Trade guard configuration".into());
        }
        Ok(())
    }
    pub fn configuration_key(&self) -> Result<String, String> {
        self.validate()?;
        let mut hash = Sha256::new();
        hash.update(b"tzolkin-public-trade-guard-config-v1\0");
        hash.update(serde_json::to_vec(self).map_err(|e| e.to_string())?);
        Ok(format!("{:x}", hash.finalize()))
    }
}

/// Closed base-NN metadata. Its validation reuses the unchanged pure variant.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct PublicLearnedSource {
    pub policy_version: String,
    pub model_version: String,
    pub training_version: String,
    pub feature_schema: u32,
    pub input_contract: String,
    pub task: String,
    pub value_validity: ValueValidity,
    pub model_checksum: String,
    pub training_checkpoint_checksum: String,
    pub dataset_fingerprint: String,
    pub inference_backend: String,
}
impl PublicLearnedSource {
    pub fn from_pure(policy: &SeatPolicy) -> Result<Self, String> {
        policy.validate()?;
        let SeatPolicy::PublicLearned {
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
        } = policy
        else {
            return Err("Trade guard requires a pure public learned base".into());
        };
        Ok(Self {
            policy_version: policy_version.clone(),
            model_version: model_version.clone(),
            training_version: training_version.clone(),
            feature_schema: *feature_schema,
            input_contract: input_contract.clone(),
            task: task.clone(),
            value_validity: *value_validity,
            model_checksum: model_checksum.clone(),
            training_checkpoint_checksum: training_checkpoint_checksum.clone(),
            dataset_fingerprint: dataset_fingerprint.clone(),
            inference_backend: inference_backend.clone(),
        })
    }
    pub fn pure_policy(&self) -> SeatPolicy {
        SeatPolicy::PublicLearned {
            policy_version: self.policy_version.clone(),
            model_version: self.model_version.clone(),
            training_version: self.training_version.clone(),
            feature_schema: self.feature_schema,
            input_contract: self.input_contract.clone(),
            task: self.task.clone(),
            value_validity: self.value_validity,
            model_checksum: self.model_checksum.clone(),
            training_checkpoint_checksum: self.training_checkpoint_checksum.clone(),
            dataset_fingerprint: self.dataset_fingerprint.clone(),
            inference_backend: self.inference_backend.clone(),
        }
    }
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub enum ExitReason {
    RepeatedPublicState,
    EpisodeTradeBudget,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct GuardCandidate {
    pub legal_index: usize,
    pub legal: LegalAction,
    pub logit: f32,
}
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct TradeGuardTrace {
    pub schema: u32,
    pub policy_version: String,
    pub raw_policy_version: String,
    pub guard_version: String,
    pub projection_version: String,
    pub configuration_key: String,
    pub callback_index: usize,
    pub actor: usize,
    pub public_state_sha256: String,
    pub episode_ordinal: usize,
    /// Counts effective Trade choices committed by this session, not callbacks.
    /// The caller must discard the session after a downstream apply/game error.
    pub trade_count_before: usize,
    pub trade_count_after: usize,
    pub first_seen_callback_index: Option<usize>,
    pub repeat_seen: bool,
    pub budget_reached: bool,
    pub force_exit: bool,
    pub overridden: bool,
    pub reason: Option<ExitReason>,
    pub episode_closed: bool,
    pub raw: GuardCandidate,
    pub skip: GuardCandidate,
    pub effective: GuardCandidate,
}

// Destructure every Observation field: additions require an explicit decision.
// All nested referenced types are core's public projection, never GameState.
#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct TradingPlayer<'a> {
    id: usize,
    resources: &'a [i64; 5],
    score_quarters: i64,
    workers: i64,
    available_workers: i64,
    temples: &'a [i64; 3],
    technologies: &'a [i64; 4],
    buildings: &'a [String],
    monuments: &'a [String],
    wealth: &'a [String],
    tribe: Option<TribeId>,
    feed_workers: i64,
    feed_all: bool,
    feed_discount: i64,
    corn_tiles: i64,
    wood_tiles: i64,
    skulls_placed: i64,
    building_skulls: i64,
    double_advance_available: bool,
    temple_points: i64,
}
impl<'a> From<&'a PublicPlayer> for TradingPlayer<'a> {
    fn from(p: &'a PublicPlayer) -> Self {
        let PublicPlayer {
            id,
            resources,
            score_quarters,
            workers,
            available_workers,
            temples,
            technologies,
            buildings,
            monuments,
            wealth,
            tribe,
            feed_workers,
            feed_all,
            feed_discount,
            corn_tiles,
            wood_tiles,
            skulls_placed,
            building_skulls,
            double_advance_available,
            temple_points,
        } = p;
        Self {
            id: *id,
            resources,
            score_quarters: *score_quarters,
            workers: *workers,
            available_workers: *available_workers,
            temples,
            technologies,
            buildings,
            monuments,
            wealth,
            tribe: *tribe,
            feed_workers: *feed_workers,
            feed_all: *feed_all,
            feed_discount: *feed_discount,
            corn_tiles: *corn_tiles,
            wood_tiles: *wood_tiles,
            skulls_placed: *skulls_placed,
            building_skulls: *building_skulls,
            double_advance_available: *double_advance_available,
            temple_points: *temple_points,
        }
    }
}
#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct PublicTradingState<'a> {
    schema: u32,
    move_schema: u32,
    actor: usize,
    turn_player: usize,
    phase: Phase,
    round: i64,
    age: i64,
    additional_buildings: bool,
    players: Vec<TradingPlayer<'a>>,
    first_player: usize,
    turn_order: &'a [usize],
    turn_index: usize,
    turn: &'a Turn,
    gears: &'a BTreeMap<GearId, Vec<Option<GearWorker>>>,
    jungle: &'a BTreeMap<i64, JungleBox>,
    skull_supply: i64,
    skull_spaces: &'a [Option<usize>],
    first_player_claimed: Option<usize>,
    accumulated_corn: i64,
    buildings: &'a [String],
    building_deck_count: usize,
    age2_deck_count: usize,
    monuments: &'a [String],
    pending_task: &'a Option<Task>,
    food_days: &'a [i64],
    expansion: &'a Option<PublicExpansion>,
    legal_actions: &'a [LegalAction],
}
struct BoundedBytes(Vec<u8>);
impl Write for BoundedBytes {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        if self
            .0
            .len()
            .checked_add(bytes.len())
            .is_none_or(|n| n > MAX_PUBLIC_STATE_BYTES)
        {
            return Err(io::Error::other("Public Trade state exceeds 128 KiB"));
        }
        self.0.extend_from_slice(bytes);
        Ok(bytes.len())
    }
    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}
fn public_bytes(o: &Observation) -> Result<Vec<u8>, String> {
    let Observation {
        observation_key: _,
        private: _,
        schema,
        move_schema,
        actor,
        turn_player,
        phase,
        round,
        age,
        additional_buildings,
        players,
        first_player,
        turn_order,
        turn_index,
        turn,
        gears,
        jungle,
        skull_supply,
        skull_spaces,
        first_player_claimed,
        accumulated_corn,
        buildings,
        building_deck_count,
        age2_deck_count,
        monuments,
        pending_task,
        food_days,
        expansion,
        legal_actions,
    } = o;
    let public = PublicTradingState {
        schema: *schema,
        move_schema: *move_schema,
        actor: *actor,
        turn_player: *turn_player,
        phase: *phase,
        round: *round,
        age: *age,
        additional_buildings: *additional_buildings,
        players: players.iter().map(TradingPlayer::from).collect(),
        first_player: *first_player,
        turn_order,
        turn_index: *turn_index,
        turn,
        gears,
        jungle,
        skull_supply: *skull_supply,
        skull_spaces,
        first_player_claimed: *first_player_claimed,
        accumulated_corn: *accumulated_corn,
        buildings,
        building_deck_count: *building_deck_count,
        age2_deck_count: *age2_deck_count,
        monuments,
        pending_task,
        food_days,
        expansion,
        legal_actions,
    };
    let mut output = BoundedBytes(Vec::new());
    serde_json::to_writer(&mut output, &public).map_err(|e| e.to_string())?;
    Ok(output.0)
}
fn is_trade(o: &Observation) -> bool {
    o.phase == Phase::Playing && o.pending_task == Some(Task::Trade)
}
fn skip_index(o: &Observation) -> Result<usize, String> {
    let mut skip = None;
    let mut trades = Vec::new();
    for (i, legal) in o.legal_actions.iter().enumerate() {
        match (&legal.action, &legal.r#move) {
            (TypedAction::Skip, GameMove::Choose { choice_id })
                if choice_id == "skip" && skip.is_none() =>
            {
                skip = Some(i)
            }
            (TypedAction::Trade { resource, buy }, GameMove::Choose { choice_id })
                if matches!(resource, Resource::Wood | Resource::Stone | Resource::Gold)
                    && *choice_id
                        == format!("{}:{resource}", if *buy { "buy" } else { "sell" })
                    && !trades.contains(&(*resource, *buy)) =>
            {
                trades.push((*resource, *buy))
            }
            _ => return Err("Malformed public Trade legal actions".into()),
        }
    }
    skip.ok_or_else(|| "Missing unique legal public Trade exit".into())
}
struct SeenState {
    bytes: Vec<u8>,
    first_index: usize,
}

/// Fresh per game/arm. No Clone, Deserialize, state/deck access or shared mutable handle.
/// Notify *every* seat's callback once, then choose at most once for that callback.
/// A callback/choice error invalidates this session; construct a fresh one to restart.
/// A returned choice/trace does not attest successful core application. Discard
/// this session after a downstream apply/game error, rather than retrying it.
/// ```compile_fail
/// # use tzolkin_ai::public_trade_guard::TradeGuardSession;
/// let session = TradeGuardSession::new(Default::default()).unwrap();
/// session.clone();
/// ```
/// ```compile_fail
/// # use tzolkin_ai::public_trade_guard::TradeGuardSession;
/// serde_json::from_str::<TradeGuardSession>("{}").unwrap();
/// ```
/// ```compile_fail
/// # use tzolkin_ai::public_trade_guard::TradeGuardSession;
/// let session = TradeGuardSession::new(Default::default()).unwrap();
/// session.seen;
/// ```
pub struct TradeGuardSession {
    config: TradeGuardConfig,
    configuration_key: String,
    next_callback: usize,
    notified: Option<(usize, usize, bool)>,
    chosen: bool,
    failed: bool,
    actor: Option<usize>,
    next_episode: usize,
    episode: usize,
    trades: usize,
    seen: Vec<SeenState>,
}
impl TradeGuardSession {
    pub fn new(config: TradeGuardConfig) -> Result<Self, String> {
        let configuration_key = config.configuration_key()?;
        Ok(Self {
            config,
            configuration_key,
            next_callback: 0,
            notified: None,
            chosen: false,
            failed: false,
            actor: None,
            next_episode: 0,
            episode: 0,
            trades: 0,
            seen: Vec::new(),
        })
    }
    fn reset_episode(&mut self) {
        self.actor = None;
        self.trades = 0;
        self.seen.clear();
    }
    fn invalidate(&mut self) {
        self.reset_episode();
        self.failed = true;
    }
    pub fn on_callback(&mut self, index: usize, o: &Observation) -> Result<(), String> {
        if self.failed || index != self.next_callback || index >= crate::replay::MAX_DECISIONS {
            self.invalidate();
            return Err("Stale/out-of-order public Trade callback or invalidated session".into());
        }
        self.next_callback += 1;
        let trade = is_trade(o);
        if !trade || self.actor.is_some_and(|actor| actor != o.actor) {
            self.reset_episode();
        }
        self.notified = Some((index, o.actor, trade));
        self.chosen = false;
        Ok(())
    }
    /// Uses exactly one existing NN distribution. This standalone entry does not
    /// qualify a checkpoint, native source, training run or producer.
    pub fn choose_loaded(
        &mut self,
        index: usize,
        o: &Observation,
        policy: &LoadedPublicPolicy<'_>,
    ) -> Result<(Decision, Option<TradeGuardTrace>), String> {
        self.choose_with_distribution(index, o, || policy.distribution(o))
    }
    pub(crate) fn choose_with_distribution(
        &mut self,
        index: usize,
        o: &Observation,
        distribution: impl FnOnce() -> Result<PublicPolicyDistribution, String>,
    ) -> Result<(Decision, Option<TradeGuardTrace>), String> {
        let result = (|| {
            self.check_callback(index, o)?;
            let output = distribution()?;
            if output.policy_version != crate::public_model::POLICY_VERSION
                || output.feature_schema != crate::features::PUBLIC_FEATURE_SCHEMA
                || output.value_validity != ValueValidity::UnavailablePolicyOnly
                || output.logits.len() != o.legal_actions.len()
                || output.logits.is_empty()
                || output.logits.len() > crate::model::MAX_CANDIDATES
                || output.logits.iter().any(|x| !x.is_finite())
            {
                return Err("Invalid base NN distribution for public Trade guard".into());
            }
            let raw = (1..output.logits.len()).fold(0, |best, i| {
                if output.logits[i] > output.logits[best] {
                    i
                } else {
                    best
                }
            });
            if !is_trade(o) {
                self.chosen = true;
                return Ok((decision(o, raw, output.logits[raw]), None));
            }
            let skip = skip_index(o)?;
            let trace = self.transition(index, o, raw, output.logits[raw], output.logits[skip])?;
            Ok((
                decision(o, trace.effective.legal_index, trace.effective.logit),
                Some(trace),
            ))
        })();
        if result.is_err() {
            self.invalidate();
        }
        result
    }
    fn check_callback(&self, index: usize, o: &Observation) -> Result<(), String> {
        if self.failed || self.chosen || self.notified != Some((index, o.actor, is_trade(o))) {
            return Err("Public Trade choice requires its single current callback".into());
        }
        Ok(())
    }
    fn transition(
        &mut self,
        index: usize,
        o: &Observation,
        raw: usize,
        raw_logit: f32,
        skip_logit: f32,
    ) -> Result<TradeGuardTrace, String> {
        self.check_callback(index, o)?;
        let skip = skip_index(o)?;
        if raw >= o.legal_actions.len()
            || !raw_logit.is_finite()
            || !skip_logit.is_finite()
            || (raw == skip && raw_logit.to_bits() != skip_logit.to_bits())
        {
            return Err("Invalid reported Trade proposal/logits".into());
        }
        let bytes = public_bytes(o)?;
        let first_seen = self
            .seen
            .iter()
            .find(|s| s.bytes == bytes)
            .map(|s| s.first_index);
        let repeat_seen = first_seen.is_some();
        let budget_reached = self.trades >= self.config.max_trades_per_episode;
        let force_exit = raw != skip && (repeat_seen || budget_reached);
        let effective = if force_exit { skip } else { raw };
        let episode = if self.actor.is_none() {
            self.next_episode
        } else {
            self.episode
        };
        let count_after = self.trades + usize::from(effective != skip);
        if effective != skip && self.seen.len() >= MAX_TRADES {
            return Err("Public Trade cache budget exceeded".into());
        }
        let candidate = |legal_index, logit| GuardCandidate {
            legal_index,
            legal: o.legal_actions[legal_index].clone(),
            logit,
        };
        let trace = TradeGuardTrace {
            schema: 1,
            policy_version: POLICY_VERSION.into(),
            raw_policy_version: crate::public_model::POLICY_VERSION.into(),
            guard_version: GUARD_VERSION.into(),
            projection_version: PROJECTION_VERSION.into(),
            configuration_key: self.configuration_key.clone(),
            callback_index: index,
            actor: o.actor,
            public_state_sha256: format!("{:x}", Sha256::digest(&bytes)),
            episode_ordinal: episode,
            trade_count_before: self.trades,
            trade_count_after: count_after,
            first_seen_callback_index: first_seen,
            repeat_seen,
            budget_reached,
            force_exit,
            overridden: force_exit,
            reason: if force_exit {
                Some(if repeat_seen {
                    ExitReason::RepeatedPublicState
                } else {
                    ExitReason::EpisodeTradeBudget
                })
            } else {
                None
            },
            episode_closed: effective == skip,
            raw: candidate(raw, raw_logit),
            skip: candidate(skip, skip_logit),
            effective: candidate(
                effective,
                if effective == skip {
                    skip_logit
                } else {
                    raw_logit
                },
            ),
        };
        // Commit only after every projection/legal/logit/trace operation succeeds.
        if self.actor.is_none() {
            self.episode = episode;
            self.next_episode += 1;
            self.actor = Some(o.actor);
        }
        self.chosen = true;
        if effective == skip {
            self.reset_episode();
        } else {
            self.seen.push(SeenState {
                bytes,
                first_index: index,
            });
            self.trades = count_after;
        }
        Ok(trace)
    }
    pub(crate) fn verify_trace(
        &mut self,
        index: usize,
        o: &Observation,
        chosen: &LegalAction,
        trace: &TradeGuardTrace,
    ) -> Result<(), String> {
        let result = (|| {
            let expected = self.transition(
                index,
                o,
                trace.raw.legal_index,
                trace.raw.logit,
                trace.skip.logit,
            )?;
            if expected.effective.legal != *chosen
                || serde_json::to_vec(&expected).map_err(|e| e.to_string())?
                    != serde_json::to_vec(trace).map_err(|e| e.to_string())?
            {
                return Err("Public Trade guard history/diagnostics mismatch".into());
            }
            Ok(())
        })();
        if result.is_err() {
            self.invalidate();
        }
        result
    }
}
fn decision(o: &Observation, index: usize, logit: f32) -> Decision {
    Decision {
        actor: o.actor,
        observation_key: o.observation_key.clone(),
        policy_version: POLICY_VERSION.into(),
        r#move: o.legal_actions[index].r#move.clone(),
        score: f64::from(logit),
    }
}

pub(crate) fn is_trade_observation(o: &Observation) -> bool {
    is_trade(o)
}

#[cfg(test)]
mod tests {
    use super::{MAX_PUBLIC_STATE_BYTES, public_bytes};

    #[test]
    fn public_projection_bound_and_private_exclusion_do_not_relax_nn_input_guards() {
        let state =
            tzolkin_core::create_game(vec!["A".into(), "B".into(), "C".into()], 17, false).unwrap();
        let original = tzolkin_core::observation::observe(&state, state.current_player).unwrap();
        let mut private = original.clone();
        private.private.wealth_offer = vec!["unknown".repeat(MAX_PUBLIC_STATE_BYTES)];
        private.observation_key = "excluded-key".into();
        assert_eq!(
            public_bytes(&original).unwrap(),
            public_bytes(&private).unwrap()
        );
        let mut oversized_public = original;
        oversized_public
            .monuments
            .push("x".repeat(MAX_PUBLIC_STATE_BYTES));
        assert!(
            public_bytes(&oversized_public)
                .unwrap_err()
                .contains("128 KiB")
        );
        // These direct serializer tests make no claim that either mutated
        // observation is accepted by the unchanged NN/card/key contract.
    }
}
