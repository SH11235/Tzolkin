//! Opt-in, game-local public Trade exit guard. This is not a producer authenticator.
use std::collections::BTreeMap;
use std::io::{self, Read, Write};
use std::path::{Component, Path, PathBuf};

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
pub const MAX_CONFIG_BYTES: usize = 64 * 1024;
pub const MAX_GAME_EXAMPLES: usize = 64;
pub const MAX_ARENA_EXAMPLES: usize = 4096;

fn network_or_device_path(path: &Path) -> bool {
    let text = path.to_string_lossy();
    let bytes = text.as_bytes();
    (bytes.len() >= 2
        && matches!(bytes[0], b'/' | b'\\')
        && matches!(bytes[1], b'/' | b'\\'))
        || path.components().any(|component| {
            matches!(component, Component::Prefix(prefix) if !matches!(prefix.kind(), std::path::Prefix::Disk(_)))
        })
}

/// Purpose-specific, closed local configuration input. No URL, network, device,
/// symlink or junction hierarchy is accepted. Publication has its own authority.
pub fn load_config(path: &Path) -> Result<TradeGuardConfig, String> {
    let text = path.to_string_lossy();
    if text.is_empty()
        || text.contains('\0')
        || text.contains("://")
        || network_or_device_path(path)
        || path.components().any(|c| matches!(c, Component::ParentDir))
        || (!path.is_absolute() && matches!(path.components().next(), Some(Component::Prefix(_))))
    {
        return Err("Trade guard config requires a local regular file".into());
    }
    let absolute = if path.is_absolute() {
        path.to_path_buf()
    } else {
        std::env::current_dir()
            .map_err(|e| e.to_string())?
            .join(path)
    };
    if network_or_device_path(&absolute) {
        return Err("Network Trade guard config hierarchy rejected".into());
    }
    let mut current = PathBuf::new();
    let mut leaf_metadata = None;
    for component in absolute.components() {
        current.push(component.as_os_str());
        if matches!(component, Component::Prefix(_)) {
            continue;
        }
        let metadata = std::fs::symlink_metadata(&current).map_err(|e| e.to_string())?;
        #[cfg(windows)]
        let reparse = {
            use std::os::windows::fs::MetadataExt;
            metadata.file_attributes() & 0x400 != 0
        };
        #[cfg(not(windows))]
        let reparse = false;
        if metadata.file_type().is_symlink() || reparse {
            return Err("Trade guard config symlink/junction hierarchy rejected".into());
        }
        leaf_metadata = Some(metadata);
    }
    // Reject stationary FIFOs/devices before opening: a FIFO read-only open
    // can block before the bounded read or post-open type check is reached.
    let metadata = leaf_metadata.ok_or("Trade guard config requires a local regular file")?;
    if !metadata.is_file() || metadata.len() > MAX_CONFIG_BYTES as u64 {
        return Err("Trade guard config must be a regular file of at most 64 KiB".into());
    }
    let file = std::fs::File::open(&absolute).map_err(|e| e.to_string())?;
    // Retain handle metadata and the bounded read for changes after preflight.
    let metadata = file.metadata().map_err(|e| e.to_string())?;
    if !metadata.is_file() || metadata.len() > MAX_CONFIG_BYTES as u64 {
        return Err("Trade guard config must be a regular file of at most 64 KiB".into());
    }
    let mut bytes = Vec::new();
    file.take((MAX_CONFIG_BYTES + 1) as u64)
        .read_to_end(&mut bytes)
        .map_err(|e| e.to_string())?;
    if bytes.len() > MAX_CONFIG_BYTES {
        return Err("Trade guard config exceeds 64 KiB".into());
    }
    let config: TradeGuardConfig = serde_json::from_slice(&bytes).map_err(|e| e.to_string())?;
    config.validate()?;
    Ok(config)
}

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

/// Fixed Trade/Skip tokens only, rather than full legal-choice strings or views.
#[derive(Clone, Debug, Serialize, PartialEq, Eq)]
#[serde(tag = "kind", rename_all = "camelCase")]
pub enum CompactTradeChoice {
    Trade { resource: Resource, buy: bool },
    Skip,
}
impl CompactTradeChoice {
    fn from_candidate(candidate: &GuardCandidate) -> Result<Self, String> {
        match &candidate.legal.action {
            TypedAction::Trade { resource, buy } => Ok(Self::Trade {
                resource: *resource,
                buy: *buy,
            }),
            TypedAction::Skip => Ok(Self::Skip),
            _ => Err("Compact Trade example requires Trade/Skip actions".into()),
        }
    }
}
#[derive(Clone, Debug, Serialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct TradeGuardExample {
    pub callback_index: usize,
    pub actor: usize,
    pub episode_ordinal: usize,
    pub public_state_sha256: String,
    pub trade_count_before: usize,
    pub trade_count_after: usize,
    pub raw_index: usize,
    pub effective_index: usize,
    pub skip_index: usize,
    pub raw: CompactTradeChoice,
    pub effective: CompactTradeChoice,
    pub raw_logit: f32,
    pub effective_logit: f32,
    pub skip_logit: f32,
    pub repeat_seen: bool,
    pub budget_reached: bool,
    pub reason: ExitReason,
}
impl TradeGuardExample {
    fn from_trace(t: &TradeGuardTrace) -> Result<Self, String> {
        if !t.overridden
            || t.callback_index >= crate::replay::MAX_DECISIONS
            || t.actor >= 4
            || t.episode_ordinal >= crate::replay::MAX_DECISIONS
            || t.trade_count_before > MAX_TRADES
            || t.trade_count_after > MAX_TRADES
            || t.public_state_sha256.len() != 64
            || !t
                .public_state_sha256
                .bytes()
                .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
            || [&t.raw, &t.effective, &t.skip]
                .iter()
                .any(|c| c.legal_index >= crate::model::MAX_CANDIDATES || !c.logit.is_finite())
        {
            return Err("Invalid bounded Trade intervention example".into());
        }
        Ok(Self {
            callback_index: t.callback_index,
            actor: t.actor,
            episode_ordinal: t.episode_ordinal,
            public_state_sha256: t.public_state_sha256.clone(),
            trade_count_before: t.trade_count_before,
            trade_count_after: t.trade_count_after,
            raw_index: t.raw.legal_index,
            effective_index: t.effective.legal_index,
            skip_index: t.skip.legal_index,
            raw: CompactTradeChoice::from_candidate(&t.raw)?,
            effective: CompactTradeChoice::from_candidate(&t.effective)?,
            raw_logit: t.raw.logit,
            effective_logit: t.effective.logit,
            skip_logit: t.skip.logit,
            repeat_seen: t.repeat_seen,
            budget_reached: t.budget_reached,
            reason: t.reason.ok_or("Intervention example missing reason")?,
        })
    }
}
/// Observation counters precede authoritative apply. Failed-game applied totals
/// are unavailable. These summaries are diagnostics, not training qualification.
#[derive(Clone, Debug, Serialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct TradeGuardCallback {
    pub index: usize,
    pub actor: usize,
    pub phase: Phase,
    pub round: i64,
    pub trade: bool,
}
#[derive(Clone, Debug, Serialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct TradeGuardSummary {
    pub provenance: SeatPolicy,
    pub global_notifications: usize,
    /// The last successful notification, not an apply acknowledgement.
    pub last_callback: Option<TradeGuardCallback>,
    pub notified_trade_callbacks: usize,
    pub guarded_decisions_observed: usize,
    pub choices_returned: usize,
    pub failed_choices: usize,
    pub completed_trade_traces: usize,
    pub episodes_observed: usize,
    pub raw_trade_choices: usize,
    pub effective_trade_choices: usize,
    pub voluntary_skip_choices: usize,
    pub repeated_state_conditions: usize,
    pub trade_budget_conditions: usize,
    pub repeated_state_overrides: usize,
    pub trade_budget_overrides: usize,
    pub max_episode_trade_count: usize,
    pub intervention_examples: Vec<TradeGuardExample>,
    pub omitted_intervention_examples: usize,
    #[serde(skip)]
    last_episode: Option<usize>,
}
/// Shared capture budget affects examples only. Fresh guard histories are separate.
pub(crate) struct TradeGuardCapture {
    remaining: usize,
}
impl TradeGuardCapture {
    pub(crate) fn new(limit: usize) -> Self {
        Self { remaining: limit }
    }
}
impl TradeGuardSummary {
    pub(crate) fn new(provenance: &SeatPolicy) -> Result<Self, String> {
        provenance.validate()?;
        if !matches!(provenance, SeatPolicy::PublicLearnedTradeGuard { .. }) {
            return Err("Trade guard summary requires composite provenance".into());
        }
        Ok(Self {
            provenance: provenance.clone(),
            global_notifications: 0,
            last_callback: None,
            notified_trade_callbacks: 0,
            guarded_decisions_observed: 0,
            choices_returned: 0,
            failed_choices: 0,
            completed_trade_traces: 0,
            episodes_observed: 0,
            raw_trade_choices: 0,
            effective_trade_choices: 0,
            voluntary_skip_choices: 0,
            repeated_state_conditions: 0,
            trade_budget_conditions: 0,
            repeated_state_overrides: 0,
            trade_budget_overrides: 0,
            max_episode_trade_count: 0,
            intervention_examples: Vec::new(),
            omitted_intervention_examples: 0,
            last_episode: None,
        })
    }
    pub(crate) fn notified(&mut self, index: usize, o: &Observation) {
        self.global_notifications += 1;
        self.notified_trade_callbacks += usize::from(is_trade(o));
        self.last_callback = Some(TradeGuardCallback {
            index,
            actor: o.actor,
            phase: o.phase,
            round: o.round,
            trade: is_trade(o),
        });
    }
    pub(crate) fn choice_started(&mut self) {
        self.guarded_decisions_observed += 1;
    }
    pub(crate) fn choice_failed(&mut self) {
        self.failed_choices += 1;
    }
    pub(crate) fn returned(
        &mut self,
        trace: Option<&TradeGuardTrace>,
        arm: &mut TradeGuardCapture,
        run: &mut TradeGuardCapture,
    ) -> Result<(), String> {
        self.choices_returned += 1;
        let Some(t) = trace else {
            return Ok(());
        };
        let SeatPolicy::PublicLearnedTradeGuard {
            configuration_key, ..
        } = &self.provenance
        else {
            unreachable!()
        };
        if t.configuration_key != *configuration_key {
            return Err("Trade summary configuration mismatch".into());
        }
        self.completed_trade_traces += 1;
        if self.last_episode != Some(t.episode_ordinal) {
            self.episodes_observed += 1;
            self.last_episode = Some(t.episode_ordinal);
        }
        self.raw_trade_choices +=
            usize::from(matches!(t.raw.legal.action, TypedAction::Trade { .. }));
        self.effective_trade_choices += usize::from(matches!(
            t.effective.legal.action,
            TypedAction::Trade { .. }
        ));
        self.voluntary_skip_choices +=
            usize::from(!t.overridden && matches!(t.raw.legal.action, TypedAction::Skip));
        self.repeated_state_conditions += usize::from(t.repeat_seen);
        self.trade_budget_conditions += usize::from(t.budget_reached);
        self.repeated_state_overrides +=
            usize::from(t.reason == Some(ExitReason::RepeatedPublicState));
        self.trade_budget_overrides +=
            usize::from(t.reason == Some(ExitReason::EpisodeTradeBudget));
        self.max_episode_trade_count = self.max_episode_trade_count.max(t.trade_count_after);
        if t.overridden {
            let example = TradeGuardExample::from_trace(t)?;
            if arm.remaining > 0 && run.remaining > 0 {
                arm.remaining -= 1;
                run.remaining -= 1;
                self.intervention_examples.push(example);
            } else {
                self.omitted_intervention_examples += 1;
            }
        }
        Ok(())
    }
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
            || (raw != skip && (skip_logit > raw_logit || (skip_logit == raw_logit && skip < raw)))
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
    use super::{MAX_PUBLIC_STATE_BYTES, TradeGuardSession, public_bytes};
    use sha2::{Digest, Sha256};
    use tzolkin_core::observation::{observation_key, observe};
    use tzolkin_core::{Pending, Phase, Task, apply_move};

    #[test]
    fn g2_capture_budgets_omit_examples_without_losing_observed_counts() {
        use super::*;
        let mut state =
            tzolkin_core::create_game(vec!["A".into(), "B".into(), "C".into()], 17, false).unwrap();
        while state.phase == Phase::Setup {
            let o = observe(&state, state.current_player).unwrap();
            state = apply_move(&state, crate::choose_move(&o).unwrap().r#move).unwrap();
        }
        state.players[state.current_player]
            .resources
            .insert(Resource::Corn, 3);
        state.pending = Some(Pending {
            title: "Trade capture fixture".into(),
            task: Task::Trade,
            after: vec![],
        });
        let o = observe(&state, state.current_player).unwrap();
        let guard = TradeGuardConfig::default();
        let model = crate::public_native::integration_fixture::model(false);
        let handle = crate::public_native::integration_fixture::handle(&model);
        let provenance = handle.guarded_provenance(&guard).unwrap();
        let mut session = TradeGuardSession::new(guard).unwrap();
        session.on_callback(0, &o).unwrap();
        session.transition(0, &o, 0, 1.0, 0.0).unwrap();
        session.on_callback(1, &o).unwrap();
        let intervention = session.transition(1, &o, 0, 1.0, 0.0).unwrap();
        assert!(intervention.overridden && intervention.repeat_seen);
        let mut run = TradeGuardCapture::new(2);
        let mut arm = TradeGuardCapture::new(MAX_GAME_EXAMPLES);
        let mut summary = TradeGuardSummary::new(&provenance).unwrap();
        // Counter/capture unit fixture, not extra game callbacks. The production
        // controller and selection do not receive either capture budget.
        for _ in 0..3 {
            summary
                .returned(Some(&intervention), &mut arm, &mut run)
                .unwrap();
        }
        assert_eq!(summary.intervention_examples.len(), 2);
        assert_eq!(summary.omitted_intervention_examples, 1);
        assert_eq!(summary.repeated_state_overrides, 3);
        assert_eq!(summary.completed_trade_traces, 3);
        assert_eq!(summary.episodes_observed, 1);
        assert_eq!(summary.raw_trade_choices, 3);
        assert_eq!(summary.effective_trade_choices, 0);
        let mut next_arm = TradeGuardCapture::new(MAX_GAME_EXAMPLES);
        let mut next = TradeGuardSummary::new(&provenance).unwrap();
        next.returned(Some(&intervention), &mut next_arm, &mut run)
            .unwrap();
        assert!(next.intervention_examples.is_empty());
        assert_eq!(next.omitted_intervention_examples, 1);
        let mut run = TradeGuardCapture::new(MAX_ARENA_EXAMPLES);
        let mut arm = TradeGuardCapture::new(0);
        next.returned(Some(&intervention), &mut arm, &mut run)
            .unwrap();
        assert_eq!(next.omitted_intervention_examples, 2);
        let mut hostile = intervention;
        hostile.raw.logit = f32::NAN;
        assert!(TradeGuardExample::from_trace(&hostile).is_err());
        hostile.raw.logit = 1.0;
        hostile.public_state_sha256 = "x".repeat(64);
        assert!(TradeGuardExample::from_trace(&hostile).is_err());
        let wire = serde_json::to_string(&summary).unwrap();
        assert!(
            !wire.contains("legalActions")
                && !wire.contains("wealthOffer")
                && !wire.contains("buy:wood")
        );
    }

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

    #[test]
    fn reported_skip_logits_cannot_contradict_pairwise_first_maximum() {
        let mut state =
            tzolkin_core::create_game(vec!["A".into(), "B".into(), "C".into()], 17, false).unwrap();
        while state.phase == Phase::Setup {
            let o = observe(&state, state.current_player).unwrap();
            state = apply_move(&state, crate::choose_move(&o).unwrap().r#move).unwrap();
        }
        state.players[state.current_player].resources = [
            (tzolkin_core::Resource::Corn, 3),
            (tzolkin_core::Resource::Wood, 3),
            (tzolkin_core::Resource::Stone, 1),
            (tzolkin_core::Resource::Gold, 0),
            (tzolkin_core::Resource::Skull, 0),
        ]
        .into_iter()
        .collect();
        state.pending = Some(Pending {
            title: "Trade".into(),
            task: Task::Trade,
            after: Vec::new(),
        });
        assert!(tzolkin_core::validation::validate_game_state(
            &serde_json::to_value(&state).unwrap()
        ));
        let original = observe(&state, state.current_player).unwrap();
        let mut session = TradeGuardSession::new(Default::default()).unwrap();
        session.on_callback(0, &original).unwrap();
        let valid = session.transition(0, &original, 0, 0.0, 0.0).unwrap();
        assert!(valid.skip.legal_index > valid.raw.legal_index);
        for earlier_skip in [false, true] {
            let mut o = original.clone();
            let mut trace = valid.clone();
            if earlier_skip {
                // Native replay validation also preserves its authoritative order.
                // This directly audits the standalone ordered-mask trace boundary.
                o.legal_actions.rotate_right(1);
                o.observation_key = observation_key(&o).unwrap();
                trace.raw.legal_index = 1;
                trace.raw.legal = o.legal_actions[1].clone();
                trace.skip.legal_index = 0;
                trace.skip.legal = o.legal_actions[0].clone();
                trace.skip.logit = -0.0; // numeric tie obeys first-max even across signed zero
                trace.public_state_sha256 =
                    format!("{:x}", Sha256::digest(public_bytes(&o).unwrap()));
            } else {
                trace.skip.logit = 1.0;
            }
            trace.effective = trace.raw.clone();
            let mut session = TradeGuardSession::new(Default::default()).unwrap();
            session.on_callback(0, &o).unwrap();
            assert!(
                session
                    .verify_trace(0, &o, &trace.effective.legal, &trace)
                    .unwrap_err()
                    .contains("Invalid reported Trade proposal/logits")
            );
            assert!(session.on_callback(1, &original).is_err());
        }
        let mut earlier_skip = original;
        earlier_skip.legal_actions.rotate_right(1);
        let mut session = TradeGuardSession::new(Default::default()).unwrap();
        session.on_callback(0, &earlier_skip).unwrap();
        assert!(session.transition(0, &earlier_skip, 1, 1.0, 0.0).is_ok());
    }
}
