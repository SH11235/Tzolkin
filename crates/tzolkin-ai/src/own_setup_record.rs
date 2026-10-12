//! Closed wire format for one Setup sample and deterministic native continuation.
//! Decoded JSON is a claim; only a fresh native audit can create an episode seal.
use std::io::{self, Write};

use serde::ser::SerializeStruct;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use tzolkin_core::observation::{LegalAction, Observation};
use tzolkin_inference::deployment::GuardedDeploymentDecision;

use crate::own_setup_model::InitializedOwnSetupIdentity;
use crate::own_setup_native::{
    AuditedOwnSetupGame, AuditedTrainedOwnSetupGame, CollectedOwnSetupGame,
    CollectedTrainedOwnSetupGame, OwnSetupCollector, TrainedOwnSetupCollector,
};
use crate::own_setup_trained_model::TrainedOwnSetupIdentity;
use crate::public_stochastic_record::{
    Config, DecisionWire, Failure, MAX_RECORD_BYTES, Terminal, hex64, serialized_size, unique_json,
};

pub const RECORD_SCHEMA: &str = "tzolkin-own-setup-native-record-v1";
pub const SOURCE_KIND: &str = "ownSetupSampleFixedGuardedPlayingV5";
pub const SAMPLING_VERSION: &str = "own-setup-semantic-scalar-uniform-tick53-v1";
const CHECKSUM_DOMAIN: &[u8] = b"tzolkin-own-setup-native-record-v1\0";
pub const TRAINED_RECORD_SCHEMA: &str = "tzolkin-own-setup-trained-native-record-v2";
pub const TRAINED_SOURCE_KIND: &str = "trainedOwnSetupSampleFixedGuardedPlayingV5";
const TRAINED_CHECKSUM_DOMAIN: &[u8] = b"tzolkin-own-setup-trained-native-record-v2\0";

#[derive(Debug, Deserialize, Serialize, PartialEq)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub(crate) struct Record<S = InitializedOwnSetupIdentity> {
    pub schema: String,
    pub header: Header<S>,
    pub callbacks: Vec<Callback>,
    pub terminal: Option<Terminal>,
    pub failure: Option<Failure>,
    pub counts: OwnSetupCounts,
    pub checksum: String,
}
#[derive(Debug, Deserialize, Serialize, PartialEq)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub(crate) struct Header<S = InitializedOwnSetupIdentity> {
    pub source_kind: String,
    pub config: Config,
    pub learner_seat: usize,
    pub native_family_id: String,
    pub setup: S,
    pub playing: FixedPlayingIdentity,
    pub roles: Vec<SeatRole>,
    pub opponent: crate::arena::resource_cycle::ResourceCycleConfig,
    pub sampling_version: String,
    pub rng_version: String,
    pub numerical_target: String,
    pub denominator_bits: u32,
    pub uniform_mixture_bits: String,
    pub producer_authenticated: bool,
    pub independent_seed_origins_verified: bool,
    pub training_admission: String,
}
pub(crate) use tzolkin_inference::own_setup_deployment::FixedPlayingIdentity;
#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub(crate) enum SeatRole {
    LearnerSetupFixedPlaying,
    V5Opponent,
}
#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub(crate) enum CallbackRole {
    SetupSample,
    FixedPlaying,
    V5Opponent,
}
#[derive(Debug, Deserialize, Serialize, PartialEq)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub(crate) struct Callback {
    pub index: usize,
    pub actor: usize,
    pub turn_player: usize,
    pub observation_key: String,
    pub observation: Option<Observation>,
    pub state_before: String,
    pub role: CallbackRole,
    pub guard_notification_attempted: bool,
    pub guard_notified: bool,
    pub choice_attempted: bool,
    pub choice: Option<Choice>,
    pub apply_attempted: bool,
    pub apply_succeeded: bool,
    pub state_after: Option<String>,
    pub state_validated: bool,
    pub failure: Option<Failure>,
}
#[derive(Debug, Deserialize, Serialize, PartialEq)]
#[serde(
    tag = "kind",
    rename_all = "camelCase",
    rename_all_fields = "camelCase",
    deny_unknown_fields
)]
pub(crate) enum Choice {
    SetupSample {
        sample: SetupSample,
        chosen: LegalAction,
    },
    FixedPlaying {
        guarded: GuardedWire,
        chosen: LegalAction,
    },
    V5Opponent {
        decision: DecisionWire,
        chosen: LegalAction,
    },
}
impl Choice {
    pub(crate) fn chosen(&self) -> &LegalAction {
        match self {
            Self::SetupSample { chosen, .. }
            | Self::FixedPlaying { chosen, .. }
            | Self::V5Opponent { chosen, .. } => chosen,
        }
    }
    pub(crate) fn role(&self) -> CallbackRole {
        match self {
            Self::SetupSample { .. } => CallbackRole::SetupSample,
            Self::FixedPlaying { .. } => CallbackRole::FixedPlaying,
            Self::V5Opponent { .. } => CallbackRole::V5Opponent,
        }
    }
}
#[derive(Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub(crate) struct SetupSample {
    pub chosen_index: usize,
    pub logits_bits: Vec<String>,
    pub logits_digest: String,
    pub ordered_legal_digest: String,
    pub nominal_probability_bits: Vec<String>,
    pub mass_ticks: Vec<u64>,
    pub distribution_digest: String,
    pub chosen_nominal_probability_bits: String,
    pub chosen_behavior_probability_bits: String,
    pub chosen_behavior_logp_bits: String,
    pub rng_state_before: String,
    pub rng_state_after: String,
    pub raw_u64: String,
    pub u53: u64,
    pub draws_before: String,
    pub draws_after: String,
}
#[derive(Debug, Deserialize, Serialize, PartialEq)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub(crate) struct GuardedWire {
    pub decision: DecisionWire,
    pub raw_policy_version: String,
    pub deployment_checksum: String,
    pub guard_version: String,
    pub projection_version: String,
    pub configuration_key: String,
    pub callback_index: usize,
    pub raw_legal_index: usize,
    pub effective_legal_index: usize,
    pub trade: Option<TradeWire>,
}
#[derive(Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub(crate) struct TradeWire {
    pub public_state_sha256: String,
    pub first_seen_callback_index: Option<usize>,
    pub repeat_seen: bool,
    pub budget_reached: bool,
    pub overridden: bool,
    pub trade_count_before: usize,
    pub trade_count_after: usize,
    pub episode_closed: bool,
}
impl GuardedWire {
    pub(crate) fn from_decision(d: GuardedDeploymentDecision) -> Self {
        Self {
            decision: DecisionWire {
                actor: d.decision.actor,
                observation_key: d.decision.observation_key,
                policy_version: d.decision.policy_version,
                r#move: d.decision.r#move,
                score_bits: hex64(d.decision.score.to_bits()),
            },
            raw_policy_version: d.raw_policy_version,
            deployment_checksum: d.deployment_checksum,
            guard_version: d.guard_version,
            projection_version: d.projection_version,
            configuration_key: d.configuration_key,
            callback_index: d.callback_index,
            raw_legal_index: d.raw_legal_index,
            effective_legal_index: d.effective_legal_index,
            trade: d.trade.map(|t| TradeWire {
                public_state_sha256: t.public_state_sha256,
                first_seen_callback_index: t.first_seen_callback_index,
                repeat_seen: t.repeat_seen,
                budget_reached: t.budget_reached,
                overridden: t.overridden,
                trade_count_before: t.trade_count_before,
                trade_count_after: t.trade_count_after,
                episode_closed: t.episode_closed,
            }),
        }
    }
}

/// Row counters are admission/work reservations, not hardware forward measurements.
/// Source bytes count serialized source fragments; trace bytes count full choices.
#[derive(Clone, Debug, Default, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct OwnSetupCounts {
    pub observed_callbacks: usize,
    pub guard_notification_attempts: usize,
    pub guard_notifications: usize,
    pub setup_attempts: usize,
    pub setup_accepted: usize,
    pub setup_applied: usize,
    pub setup_reserved_rows: usize,
    pub playing_attempts: usize,
    pub playing_choices: usize,
    pub opponent_attempts: usize,
    pub opponent_choices: usize,
    pub all_role_reserved_rows: usize,
    pub apply_attempts: usize,
    pub apply_successes: usize,
    pub validated_transitions: usize,
    pub actor_draws: Vec<String>,
    pub source_reserved_bytes: usize,
    pub trace_reserved_bytes: usize,
    pub actual_source_bytes: usize,
    pub actual_trace_bytes: usize,
    pub actual_record_bytes: usize,
    pub stopped: bool,
}

// This serialization differs only in checksum content; it avoids cloning a game.
struct ChecksumPayload<'a, S>(&'a Record<S>);
impl<S: Serialize> Serialize for ChecksumPayload<'_, S> {
    fn serialize<T: serde::Serializer>(&self, s: T) -> Result<T::Ok, T::Error> {
        let r = self.0;
        let mut s = s.serialize_struct("Record", 7)?;
        s.serialize_field("schema", &r.schema)?;
        s.serialize_field("header", &r.header)?;
        s.serialize_field("callbacks", &r.callbacks)?;
        s.serialize_field("terminal", &r.terminal)?;
        s.serialize_field("failure", &r.failure)?;
        s.serialize_field("counts", &r.counts)?;
        s.serialize_field("checksum", "")?;
        s.end()
    }
}
struct HashWriter {
    hash: Sha256,
    bytes: usize,
}
impl Write for HashWriter {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        self.bytes = self
            .bytes
            .checked_add(bytes.len())
            .filter(|n| *n <= MAX_RECORD_BYTES)
            .ok_or_else(|| io::Error::other("Own Setup record byte bound"))?;
        self.hash.update(bytes);
        Ok(bytes.len())
    }
    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}
pub(crate) fn checksum<S: Serialize>(record: &Record<S>) -> Result<String, String> {
    let mut out = HashWriter {
        hash: Sha256::new(),
        bytes: 0,
    };
    out.hash.update(match record.schema.as_str() {
        RECORD_SCHEMA => CHECKSUM_DOMAIN,
        TRAINED_RECORD_SCHEMA => TRAINED_CHECKSUM_DOMAIN,
        _ => return Err("Unsupported Own Setup checksum domain".into()),
    });
    serde_json::to_writer(&mut out, &ChecksumPayload(record)).map_err(|e| e.to_string())?;
    Ok(format!("{:x}", out.hash.finalize()))
}
fn decode_kind<S: serde::de::DeserializeOwned + Serialize>(
    bytes: &[u8],
    schema: &str,
) -> Result<Record<S>, String> {
    let raw = unique_json(bytes, MAX_RECORD_BYTES)?;
    let record: Record<S> = serde_json::from_slice(bytes).map_err(|e| e.to_string())?;
    if raw != serde_json::to_value(&record).map_err(|e| e.to_string())? || record.schema != schema {
        return Err("Own Setup record closed schema/field mismatch".into());
    }
    if record.checksum.len() != 64 || record.checksum != checksum(&record)? {
        return Err("Own Setup record checksum mismatch".into());
    }
    Ok(record)
}
pub(crate) fn decode(bytes: &[u8]) -> Result<Record, String> {
    decode_kind(bytes, RECORD_SCHEMA)
}
pub(crate) fn decode_trained(bytes: &[u8]) -> Result<Record<TrainedOwnSetupIdentity>, String> {
    decode_kind(bytes, TRAINED_RECORD_SCHEMA)
}
/// Encode only an actual collection result, including a closed failure prefix.
pub fn encode_record(game: &CollectedOwnSetupGame) -> Result<Vec<u8>, String> {
    let record = game.record();
    serialized_size(record, MAX_RECORD_BYTES)?;
    serde_json::to_vec(record).map_err(|e| e.to_string())
}
/// The same immutable models are used again with fresh native state and guard history.
pub fn decode_and_audit(
    collector: &mut OwnSetupCollector<'_>,
    bytes: &[u8],
) -> Result<AuditedOwnSetupGame, String> {
    collector.audit(bytes)
}
/// Encode only a real trained-model collection; the initialized codec stays closed.
pub fn encode_trained_record(game: &CollectedTrainedOwnSetupGame) -> Result<Vec<u8>, String> {
    let record = game.record();
    serialized_size(record, MAX_RECORD_BYTES)?;
    serde_json::to_vec(record).map_err(|e| e.to_string())
}
pub fn decode_and_audit_trained(
    collector: &mut TrainedOwnSetupCollector<'_>,
    bytes: &[u8],
) -> Result<AuditedTrainedOwnSetupGame, String> {
    collector.audit(bytes)
}
