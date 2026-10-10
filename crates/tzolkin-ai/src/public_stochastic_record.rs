//! Closed stochastic record codec. This is neither GameReplay nor ML admission.
//!
//! Parsing untrusted bytes yields a receipt only after native transitions and
//! the caller's immutable Scalar policy have been rechecked. A record cannot
//! select a model path. Logical model/checkpoint/dataset identities are bound;
//! original artifact bytes and producer/seed independence are not authenticated.
use std::fmt;
use std::io::{self, Write};

use serde::de::{self, MapAccess, SeqAccess, Visitor};
use serde::{Deserialize, Serialize};
use tzolkin_core::observation::{LegalAction, Observation};
use tzolkin_core::{FinalScore, GameMove};

use crate::public_native::PublicPolicyHandle;
use crate::public_stochastic::{SampledDecision, SamplingStreamIdentity};
use crate::public_stochastic_native::{
    AuditedStochasticGame, CollectedStochasticGame, NativeStochasticConfig, audit_record,
};
use crate::replay::SeatPolicy;

pub const RECORD_SCHEMA: &str = "tzolkin-public-stochastic-native-record-v1";
pub const SOURCE_KIND: &str = "publicStochasticBc";
pub const MAX_SOURCE_BYTES: usize = 64 * 1024 * 1024;
pub const MAX_TRACE_BYTES: usize = 32 * 1024 * 1024;
pub const MAX_RECORD_BYTES: usize = MAX_SOURCE_BYTES + MAX_TRACE_BYTES;
pub(crate) const ERROR_BYTES: usize = 1024;
pub(crate) const FINAL_RESERVE: usize = 16 * 1024;
pub(crate) const CALLBACK_RESERVE: usize = 4 * 1024;
pub(crate) const TRACE_RESERVE: usize = 4 * 1024;

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub(crate) struct Record<P = SeatPolicy> {
    pub schema: String,
    pub header: Header<P>,
    pub callbacks: Vec<Callback>,
    pub terminal: Option<Terminal>,
    pub failure: Option<Failure>,
    pub counts: Counts,
    pub source_payload_reserved_bytes: usize,
    pub trace_payload_reserved_bytes: usize,
    pub training_admission: String,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub(crate) struct Header<P = SeatPolicy> {
    pub source_kind: String,
    pub rules_version: u32,
    pub rules_baseline: String,
    pub catalog_hash: String,
    pub observation_schema: u32,
    pub move_schema: u32,
    pub config: Config,
    pub base_policy: P,
    pub sampling_version: String,
    pub rng_version: String,
    pub denominator_bits: u32,
    pub uniform_mixture_bits: String,
    pub backend: String,
    pub numerical_target: String,
    pub native_family_id: String,
    pub raw_artifact_identity_available: bool,
    pub producer_authenticated: bool,
    pub independent_seed_origins_verified: bool,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub(crate) struct Config {
    pub players: usize,
    /// Trusted replay metadata; not a sampler input.
    pub environment_seed: u32,
    pub options: Options,
    pub sampling_seed: String,
    pub episode_ordinal: String,
    pub replicate_ordinal: String,
    pub max_callbacks: usize,
    pub max_candidate_rows: usize,
    pub max_source_bytes: usize,
    pub max_trace_bytes: usize,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub(crate) struct Options {
    pub additional_buildings: bool,
    pub tribes: bool,
    pub prophecies: bool,
    pub quick_actions: bool,
}
impl Options {
    pub(crate) fn base() -> Self {
        Self {
            additional_buildings: false,
            tribes: false,
            prophecies: false,
            quick_actions: false,
        }
    }
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub(crate) struct Callback {
    pub global_callback_index: usize,
    pub actor: usize,
    pub turn_player: usize,
    pub observation_key: String,
    /// Null only when the current observation refused payload admission.
    pub observation: Option<Observation>,
    pub state_before: String,
    pub sample: Option<Sample>,
    /// Only the distinct mixed schema permits this field. Old wire bytes omit it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub opponent_choice: Option<OpponentChoice>,
    #[serde(default, skip_serializing_if = "is_false")]
    pub opponent_attempted: bool,
    pub sampler_attempted: bool,
    pub apply_attempted: bool,
    pub apply_succeeded: bool,
    pub state_after: Option<String>,
    pub failure: Option<Failure>,
}

fn is_false(value: &bool) -> bool {
    !value
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub(crate) struct OpponentChoice {
    pub decision: DecisionWire,
    pub chosen: LegalAction,
}
impl OpponentChoice {
    pub(crate) fn from_decision(decision: &crate::Decision, chosen: LegalAction) -> Self {
        Self {
            decision: DecisionWire {
                actor: decision.actor,
                observation_key: decision.observation_key.clone(),
                policy_version: decision.policy_version.clone(),
                r#move: decision.r#move.clone(),
                score_bits: hex64(decision.score.to_bits()),
            },
            chosen,
        }
    }
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub(crate) struct Sample {
    pub decision: DecisionWire,
    pub chosen: LegalAction,
    pub trace: Trace,
}
impl Sample {
    pub(crate) fn from_sample(sample: &SampledDecision) -> Self {
        let d = sample.decision();
        Self {
            decision: DecisionWire {
                actor: d.actor,
                observation_key: d.observation_key.clone(),
                policy_version: d.policy_version.clone(),
                r#move: d.r#move.clone(),
                score_bits: hex64(d.score.to_bits()),
            },
            chosen: sample.legal_action().clone(),
            trace: Trace::from_sample(sample),
        }
    }
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub(crate) struct DecisionWire {
    pub actor: usize,
    pub observation_key: String,
    pub policy_version: String,
    pub r#move: GameMove,
    pub score_bits: String,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub(crate) struct Trace {
    pub policy_binding_key: String,
    /// Successful samples across this session, not callback or per-seat index.
    pub sample_index: usize,
    pub actor: usize,
    pub legal_count: usize,
    pub chosen_index: usize,
    pub state_before: String,
    pub state_after: String,
    pub draws_before: String,
    pub draws_after: String,
    pub raw_u64: String,
    pub u53: u64,
    pub mass_ticks: String,
    pub denominator_bits: u32,
    pub nominal_probability_bits: String,
    pub behavior_probability_bits: String,
    pub behavior_logp_bits: String,
    pub chosen_logit_bits: String,
    pub ordered_legal_digest: String,
    pub logits_digest: String,
    pub distribution_digest: String,
    pub reserved_rows_before: usize,
    pub reserved_rows_after: usize,
}
impl Trace {
    fn from_sample(sample: &SampledDecision) -> Self {
        let t = sample.trace();
        Self {
            policy_binding_key: t.policy_binding_key().into(),
            sample_index: t.sample_index(),
            actor: t.actor(),
            legal_count: t.legal_count(),
            chosen_index: t.chosen_index(),
            state_before: hex64(t.state_before()),
            state_after: hex64(t.state_after()),
            draws_before: hex64(t.draws_before()),
            draws_after: hex64(t.draws_after()),
            raw_u64: hex64(t.raw_u64()),
            u53: t.u53(),
            mass_ticks: t.mass_ticks().to_string(),
            denominator_bits: t.denominator_bits(),
            nominal_probability_bits: hex64(t.nominal_probability().to_bits()),
            behavior_probability_bits: hex64(t.behavior_probability().to_bits()),
            behavior_logp_bits: hex64(t.behavior_logp().to_bits()),
            chosen_logit_bits: format!("{:08x}", t.chosen_logit().to_bits()),
            ordered_legal_digest: t.ordered_legal_digest().into(),
            logits_digest: t.logits_digest().into(),
            distribution_digest: t.distribution_digest().into(),
            reserved_rows_before: t.reserved_rows_before(),
            reserved_rows_after: t.reserved_rows_after(),
        }
    }
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub(crate) struct Counts {
    pub observed_callbacks: usize,
    pub sampler_attempts: usize,
    pub accepted_samples: usize,
    pub candidate_rows_reserved: usize,
    pub actor_draws: Vec<String>,
    pub apply_attempts: usize,
    pub apply_successes: usize,
    pub validated_applied_steps: usize,
    pub session_stopped: bool,
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub enum FailureReason {
    CreationError,
    ObservationError,
    CallbackLimit,
    CandidateLimit,
    SourcePayloadLimit,
    TracePayloadLimit,
    ModelOrSamplingError,
    DecisionMismatch,
    CoreApplyError,
    CoreValidationError,
}
#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub enum AttemptStage {
    Startup,
    BeforeObservation,
    BeforeSample,
    BeforeApply,
    AfterApply,
    TerminalValidation,
}
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub(crate) struct Failure {
    pub reason: FailureReason,
    pub stage: AttemptStage,
    pub global_callback_index: Option<usize>,
    pub message: String,
    pub message_truncated: bool,
}
impl Failure {
    pub(crate) fn new(
        reason: FailureReason,
        stage: AttemptStage,
        index: Option<usize>,
        message: &str,
    ) -> Self {
        let mut end = message.len().min(ERROR_BYTES);
        while !message.is_char_boundary(end) {
            end -= 1;
        }
        Self {
            reason,
            stage,
            global_callback_index: index,
            message: message[..end].into(),
            message_truncated: end != message.len(),
        }
    }
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub(crate) struct Terminal {
    pub final_state: String,
    pub final_scores: Vec<FinalScoreWire>,
}
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub(crate) struct FinalScoreWire {
    pub player_id: usize,
    pub points_before_final_bits: String,
    pub resource_points_bits: String,
    pub skull_points_bits: String,
    pub monument_points_bits: String,
    pub total_bits: String,
    pub workers_on_gears: i64,
    pub rank: usize,
}
impl FinalScoreWire {
    pub(crate) fn from_score(s: &FinalScore) -> Self {
        Self {
            player_id: s.player_id,
            points_before_final_bits: hex64(s.points_before_final.to_bits()),
            resource_points_bits: hex64(s.resource_points.to_bits()),
            skull_points_bits: hex64(s.skull_points.to_bits()),
            monument_points_bits: hex64(s.monument_points.to_bits()),
            total_bits: hex64(s.total.to_bits()),
            workers_on_gears: s.workers_on_gears,
            rank: s.rank,
        }
    }
}

pub(crate) fn hex64(value: u64) -> String {
    format!("{value:016x}")
}
pub(crate) fn parse_hex64(value: &str) -> Result<u64, String> {
    if value.len() != 16
        || !value
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
    {
        return Err("Expected fixed-width lowercase u64 bits".into());
    }
    u64::from_str_radix(value, 16).map_err(|e| e.to_string())
}

fn parse_hex32(value: &str) -> Result<u32, String> {
    if value.len() != 8 || !lowercase_hex(value) {
        return Err("Expected fixed-width lowercase f32 bits".into());
    }
    u32::from_str_radix(value, 16).map_err(|e| e.to_string())
}

fn lowercase_hex(value: &str) -> bool {
    value
        .bytes()
        .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}

fn finite64(value: &str) -> Result<(), String> {
    if !f64::from_bits(parse_hex64(value)?).is_finite() {
        return Err("Nonfinite f64 wire bits".into());
    }
    Ok(())
}

fn sha_digest(value: &str) -> Result<(), String> {
    if value.len() != 64 || !lowercase_hex(value) {
        return Err("Expected fixed-width lowercase SHA256 digest".into());
    }
    Ok(())
}

/// Validate representation before any reconstruction/model work. Existing
/// FNV fingerprints are sixteen hex digits; SHA digests are sixty-four.
/// These shape checks neither authenticate a source nor validate likelihood.
fn validate_wire<P>(record: &Record<P>) -> Result<(), String> {
    let header = &record.header;
    parse_hex64(&header.catalog_hash)?;
    finite64(&header.uniform_mixture_bits)?;
    sha_digest(&header.native_family_id)?;
    for value in [
        &header.config.sampling_seed,
        &header.config.episode_ordinal,
        &header.config.replicate_ordinal,
    ] {
        parse_hex64(value)?;
    }
    for value in &record.counts.actor_draws {
        parse_hex64(value)?;
    }
    for callback in &record.callbacks {
        if record.schema != crate::public_mixed_native::RECORD_SCHEMA
            && (callback.opponent_choice.is_some() || callback.opponent_attempted)
        {
            return Err("Opponent fields are forbidden in an all-seat stochastic schema".into());
        }
        if let Some(opponent) = &callback.opponent_choice {
            if !callback.opponent_attempted
                || callback.sampler_attempted
                || callback.sample.is_some()
            {
                return Err("Opponent and learner choices must be distinct".into());
            }
            parse_hex64(&opponent.decision.observation_key)?;
            finite64(&opponent.decision.score_bits)?;
        }
        if callback.opponent_attempted && callback.sampler_attempted {
            return Err("A callback cannot attempt both roster policies".into());
        }
        parse_hex64(&callback.observation_key)?;
        parse_hex64(&callback.state_before)?;
        if let Some(after) = &callback.state_after {
            parse_hex64(after)?;
        }
        if let Some(observation) = &callback.observation {
            parse_hex64(&observation.observation_key)?;
        }
        if let Some(sample) = &callback.sample {
            let trace = &sample.trace;
            parse_hex64(&sample.decision.observation_key)?;
            finite64(&sample.decision.score_bits)?;
            for value in [
                &trace.state_before,
                &trace.state_after,
                &trace.draws_before,
                &trace.draws_after,
                &trace.raw_u64,
            ] {
                parse_hex64(value)?;
            }
            for value in [
                &trace.policy_binding_key,
                &trace.ordered_legal_digest,
                &trace.logits_digest,
                &trace.distribution_digest,
            ] {
                sha_digest(value)?;
            }
            for value in [
                &trace.nominal_probability_bits,
                &trace.behavior_probability_bits,
                &trace.behavior_logp_bits,
            ] {
                finite64(value)?;
            }
            if !f32::from_bits(parse_hex32(&trace.chosen_logit_bits)?).is_finite() {
                return Err("Nonfinite f32 wire bits".into());
            }
            let ticks = trace
                .mass_ticks
                .parse::<u64>()
                .map_err(|_| "Expected canonical positive mass ticks")?;
            if ticks == 0
                || ticks > (1u64 << 53)
                || trace.mass_ticks != ticks.to_string()
                || trace.u53 >= (1u64 << 53)
            {
                return Err("Noncanonical/out-of-range tick or u53 representation".into());
            }
        }
    }
    if let Some(terminal) = &record.terminal {
        parse_hex64(&terminal.final_state)?;
        for score in &terminal.final_scores {
            for value in [
                &score.points_before_final_bits,
                &score.resource_points_bits,
                &score.skull_points_bits,
                &score.monument_points_bits,
                &score.total_bits,
            ] {
                finite64(value)?;
            }
        }
    }
    Ok(())
}

/// Value's normal object parser accepts duplicate keys. This private visitor
/// rejects them at every nesting level, including escaped-equivalent keys,
/// before a last-wins map can discard evidence. Serde JSON's recursion limit
/// and strict trailing-input check remain enabled.
struct UniqueValue(serde_json::Value);

impl<'de> Deserialize<'de> for UniqueValue {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct UniqueVisitor;
        impl<'de> Visitor<'de> for UniqueVisitor {
            type Value = UniqueValue;
            fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                formatter.write_str("JSON with unique object keys")
            }
            fn visit_bool<E: de::Error>(self, value: bool) -> Result<Self::Value, E> {
                Ok(UniqueValue(value.into()))
            }
            fn visit_i64<E: de::Error>(self, value: i64) -> Result<Self::Value, E> {
                Ok(UniqueValue(value.into()))
            }
            fn visit_u64<E: de::Error>(self, value: u64) -> Result<Self::Value, E> {
                Ok(UniqueValue(value.into()))
            }
            fn visit_f64<E: de::Error>(self, value: f64) -> Result<Self::Value, E> {
                let number = serde_json::Number::from_f64(value)
                    .ok_or_else(|| E::custom("Nonfinite JSON number"))?;
                Ok(UniqueValue(serde_json::Value::Number(number)))
            }
            fn visit_str<E: de::Error>(self, value: &str) -> Result<Self::Value, E> {
                Ok(UniqueValue(value.into()))
            }
            fn visit_string<E: de::Error>(self, value: String) -> Result<Self::Value, E> {
                Ok(UniqueValue(value.into()))
            }
            fn visit_unit<E: de::Error>(self) -> Result<Self::Value, E> {
                Ok(UniqueValue(serde_json::Value::Null))
            }
            fn visit_none<E: de::Error>(self) -> Result<Self::Value, E> {
                self.visit_unit()
            }
            fn visit_seq<A: SeqAccess<'de>>(
                self,
                mut sequence: A,
            ) -> Result<Self::Value, A::Error> {
                let mut values = Vec::new();
                while let Some(value) = sequence.next_element::<UniqueValue>()? {
                    values.push(value.0);
                }
                Ok(UniqueValue(serde_json::Value::Array(values)))
            }
            fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> Result<Self::Value, A::Error> {
                let mut values = serde_json::Map::new();
                while let Some(key) = map.next_key::<String>()? {
                    if values.contains_key(&key) {
                        return Err(de::Error::custom("Duplicate JSON object key"));
                    }
                    let value = map.next_value::<UniqueValue>()?;
                    values.insert(key, value.0);
                }
                Ok(UniqueValue(serde_json::Value::Object(values)))
            }
        }
        deserializer.deserialize_any(UniqueVisitor)
    }
}

fn decode_record(bytes: &[u8]) -> Result<Record, String> {
    decode_typed_record(bytes)
}
pub(crate) fn decode_typed_record<P: serde::de::DeserializeOwned + Serialize>(
    bytes: &[u8],
) -> Result<Record<P>, String> {
    if bytes.is_empty() || bytes.len() > MAX_RECORD_BYTES {
        return Err("Stochastic record byte bound".into());
    }
    let raw: UniqueValue = serde_json::from_slice(bytes).map_err(|e| e.to_string())?;
    // Extending the private DTO must not broaden any old source's closed shape,
    // including explicit null/false forms that serde could otherwise omit.
    if raw.0.get("schema").and_then(serde_json::Value::as_str)
        != Some(crate::public_mixed_native::RECORD_SCHEMA)
        && raw
            .0
            .get("callbacks")
            .and_then(serde_json::Value::as_array)
            .is_some_and(|callbacks| {
                callbacks.iter().any(|c| {
                    c.get("opponentChoice").is_some() || c.get("opponentAttempted").is_some()
                })
            })
    {
        return Err("Opponent fields are forbidden in an all-seat stochastic schema".into());
    }
    let record: Record<P> = serde_json::from_slice(bytes).map_err(|e| e.to_string())?;
    if raw.0 != serde_json::to_value(&record).map_err(|e| e.to_string())? {
        return Err("Nested unknown, omitted nullable, or noncanonical wire fields".into());
    }
    validate_wire(&record)?;
    Ok(record)
}
impl Config {
    pub(crate) fn from_config(c: &NativeStochasticConfig) -> Self {
        let i = c.sampling_identity();
        let q = c.limits();
        Self {
            players: c.players(),
            environment_seed: c.environment_seed(),
            options: Options::base(),
            sampling_seed: hex64(i.sampling_seed().value()),
            episode_ordinal: hex64(i.episode_ordinal()),
            replicate_ordinal: hex64(i.replicate_ordinal()),
            max_callbacks: q.max_callbacks(),
            max_candidate_rows: q.max_candidate_rows(),
            max_source_bytes: q.max_source_bytes(),
            max_trace_bytes: q.max_trace_bytes(),
        }
    }
    pub(crate) fn checked_config(&self) -> Result<NativeStochasticConfig, String> {
        if self.options != Options::base() {
            return Err("Stochastic native record requires four explicit false options".into());
        }
        let identity = SamplingStreamIdentity::new(
            crate::public_stochastic::SamplingSeed::new(parse_hex64(&self.sampling_seed)?),
            parse_hex64(&self.episode_ordinal)?,
            parse_hex64(&self.replicate_ordinal)?,
            self.players,
        )?;
        let limits = crate::public_stochastic_native::CollectionLimits::new(
            self.max_callbacks,
            self.max_candidate_rows,
            self.max_source_bytes,
            self.max_trace_bytes,
        )?;
        NativeStochasticConfig::new(self.players, self.environment_seed, identity, limits)
    }
}

struct Counter {
    used: usize,
    maximum: usize,
}
impl Write for Counter {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        self.used = self
            .used
            .checked_add(bytes.len())
            .ok_or_else(|| io::Error::other("Serialization count overflow"))?;
        if self.used > self.maximum {
            return Err(io::Error::other("Serialized payload limit"));
        }
        Ok(bytes.len())
    }
    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}
pub(crate) fn serialized_size<T: Serialize>(value: &T, max: usize) -> Result<usize, String> {
    let mut counter = Counter {
        used: 0,
        maximum: max,
    };
    serde_json::to_writer(&mut counter, value).map_err(|e| e.to_string())?;
    Ok(counter.used)
}

/// Encode only a collected sealed record. No filesystem publication occurs here.
pub fn encode_record(game: &CollectedStochasticGame) -> Result<Vec<u8>, String> {
    encode_typed_record(game.record())
}
pub(crate) fn encode_typed_record<P: Serialize>(record: &Record<P>) -> Result<Vec<u8>, String> {
    validate_wire(record)?;
    serialized_size(record, MAX_RECORD_BYTES)?;
    let bytes = serde_json::to_vec(record).map_err(|e| e.to_string())?;
    if bytes.len() > MAX_RECORD_BYTES {
        return Err("Record exceeds byte bound".into());
    }
    Ok(bytes)
}

/// Strict EOF and nested shape checks precede a fresh native/Scalar audit.
/// No path/model selection or existing dataset admission is provided.
pub fn audit_record_bytes(
    bytes: &[u8],
    policy: &PublicPolicyHandle<'_>,
) -> Result<AuditedStochasticGame, String> {
    let record = decode_record(bytes)?;
    audit_record(&record, policy)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::public_native::integration_fixture;
    use crate::public_stochastic::{SamplingSeed, SamplingStreamIdentity};
    use crate::public_stochastic_native::{CollectionLimits, collect_native};

    fn collected(policy: &PublicPolicyHandle<'_>, players: usize) -> CollectedStochasticGame {
        let identity = SamplingStreamIdentity::new(SamplingSeed::new(17), 0, 0, players).unwrap();
        let limits = CollectionLimits::new(2, 100_000, MAX_SOURCE_BYTES, MAX_TRACE_BYTES).unwrap();
        let config = NativeStochasticConfig::new(players, 17, identity, limits).unwrap();
        collect_native(&config, policy).unwrap()
    }

    #[test]
    fn sealed_native_prefix_roundtrip_and_fresh_scalar_audit() {
        let model = integration_fixture::model(false);
        let policy = integration_fixture::handle(&model);
        for players in [3, 4] {
            let game = collected(&policy, players);
            assert!(!game.complete());
            assert_eq!(game.accepted_samples(), 2);
            assert_eq!(game.applied_choices(), 2);
            let bytes = encode_record(&game).unwrap();
            let wire: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
            for callback in wire["callbacks"].as_array().unwrap() {
                assert!(callback.get("opponentChoice").is_none());
                assert!(callback.get("opponentAttempted").is_none());
            }
            let checked = audit_record_bytes(&bytes, &policy).unwrap();
            assert!(!checked.complete());
            assert_eq!(checked.accepted_samples(), 2);
            assert_eq!(checked.applied_choices(), 2);
            assert_eq!(checked.failure_reason(), Some(FailureReason::CallbackLimit));
            assert!(!checked.training_admission_available());
            assert!(!checked.producer_authenticated());
            assert!(!checked.independent_seed_origins_verified());
            assert_eq!(encode_record(&game).unwrap(), bytes);
        }
    }

    #[test]
    fn duplicate_keys_are_rejected_even_when_equal_nested_or_escaped() {
        for bytes in [
            br#"{"schema":1,"schema":1}"#.as_slice(),
            br#"{"header":{"config":{"players":3,"players":3}}}"#.as_slice(),
            br#"{"callbacks":[{"observation":{"private":{"x":null,"x":null}}}]}"#.as_slice(),
            br#"{"schema":1,"\u0073chema":1}"#.as_slice(),
        ] {
            let error = serde_json::from_slice::<UniqueValue>(bytes)
                .err()
                .unwrap()
                .to_string();
            assert!(error.contains("Duplicate JSON object key"), "{error}");
        }
        let model = integration_fixture::model(false);
        let policy = integration_fixture::handle(&model);
        let bytes = encode_record(&collected(&policy, 3)).unwrap();
        let mut duplicate = br#"{"schema":"tzolkin-public-stochastic-native-record-v1","#.to_vec();
        duplicate.extend_from_slice(&bytes[1..]);
        let error = audit_record_bytes(&duplicate, &policy).err().unwrap();
        assert!(error.contains("Duplicate JSON object key"), "{error}");
    }

    #[test]
    fn nested_unknown_and_missing_required_null_cannot_be_discarded() {
        let model = integration_fixture::model(false);
        let policy = integration_fixture::handle(&model);
        let bytes = encode_record(&collected(&policy, 3)).unwrap();
        let original: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        for (key, value) in [
            ("opponentChoice", serde_json::Value::Null),
            ("opponentAttempted", serde_json::Value::Null),
            ("opponentAttempted", serde_json::Value::Bool(false)),
        ] {
            let mut corrupt = original.clone();
            corrupt["callbacks"][0][key] = value;
            let error = decode_record(&serde_json::to_vec(&corrupt).unwrap())
                .err()
                .unwrap();
            assert!(error.contains("Opponent fields are forbidden"), "{error}");
        }
        for pointer in [
            "",
            "/header",
            "/header/config",
            "/header/config/options",
            "/counts",
            "/callbacks/0",
            "/callbacks/0/observation",
            "/callbacks/0/observation/private",
            "/callbacks/0/sample",
            "/callbacks/0/sample/decision",
            "/callbacks/0/sample/trace",
            "/callbacks/0/sample/chosen",
            "/callbacks/0/sample/decision/move",
        ] {
            let mut corrupt = original.clone();
            corrupt
                .pointer_mut(pointer)
                .unwrap()
                .as_object_mut()
                .unwrap()
                .insert("unexpected".into(), true.into());
            assert!(
                decode_record(&serde_json::to_vec(&corrupt).unwrap()).is_err(),
                "{pointer}"
            );
        }
        for (parent, key) in [
            ("", "terminal"),
            ("/callbacks/0", "failure"),
            ("/callbacks/0/observation", "pendingTask"),
        ] {
            let mut corrupt = original.clone();
            let removed = corrupt
                .pointer_mut(parent)
                .unwrap()
                .as_object_mut()
                .unwrap()
                .remove(key)
                .unwrap();
            assert!(removed.is_null(), "fixture nullable value: {parent}/{key}");
            let error = decode_record(&serde_json::to_vec(&corrupt).unwrap())
                .err()
                .unwrap();
            assert!(error.contains("omitted nullable"), "{error}");
        }
    }

    #[test]
    fn strict_eof_recursion_and_nonfinite_json_before_audit() {
        let model = integration_fixture::model(false);
        let policy = integration_fixture::handle(&model);
        let bytes = encode_record(&collected(&policy, 3)).unwrap();
        for suffix in [b"{}".as_slice(), b"null".as_slice(), b"x".as_slice()] {
            let mut corrupt = bytes.clone();
            corrupt.extend_from_slice(suffix);
            assert!(decode_record(&corrupt).is_err());
        }
        let mut whitespace = bytes.clone();
        whitespace.extend_from_slice(b" \r\n\t");
        assert!(decode_record(&whitespace).is_ok());
        for invalid in [
            b"".as_slice(),
            b" ".as_slice(),
            b"NaN".as_slice(),
            b"1e999".as_slice(),
        ] {
            assert!(decode_record(invalid).is_err());
        }
        let deep = format!("{}0{}", "[".repeat(140), "]".repeat(140));
        assert!(serde_json::from_slice::<UniqueValue>(deep.as_bytes()).is_err());
    }

    #[test]
    fn wire_bit_strings_decimal_ticks_and_integer_identities_are_canonical() {
        let model = integration_fixture::model(false);
        let policy = integration_fixture::handle(&model);
        let bytes = encode_record(&collected(&policy, 3)).unwrap();
        let original: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        for (pointer, replacement) in [
            ("/header/config/samplingSeed", "000000000000001A"),
            ("/header/config/episodeOrdinal", "0"),
            ("/header/config/replicateOrdinal", "10000000000000000"),
            ("/header/uniformMixtureBits", "7ff0000000000000"),
            ("/callbacks/0/sample/decision/scoreBits", "7ff8000000000000"),
            ("/callbacks/0/sample/trace/chosenLogitBits", "7f800000"),
            ("/callbacks/0/sample/trace/chosenLogitBits", "0000000A"),
            ("/callbacks/0/sample/trace/rawU64", "0x00000000000001"),
            ("/callbacks/0/sample/trace/logitsDigest", "0"),
            ("/callbacks/0/sample/trace/massTicks", "01"),
            ("/callbacks/0/sample/trace/massTicks", "+1"),
            ("/callbacks/0/sample/trace/massTicks", "0"),
            ("/callbacks/0/sample/trace/massTicks", "9007199254740993"),
            ("/counts/actorDraws/0", "000000000000000G"),
        ] {
            let mut corrupt = original.clone();
            *corrupt.pointer_mut(pointer).unwrap() = replacement.into();
            assert!(
                decode_record(&serde_json::to_vec(&corrupt).unwrap()).is_err(),
                "{pointer}: {replacement}"
            );
        }
        let mut corrupt = original;
        corrupt["callbacks"][0]["sample"]["trace"]["u53"] = serde_json::json!(1u64 << 53);
        assert!(decode_record(&serde_json::to_vec(&corrupt).unwrap()).is_err());
        for value in [0, 1, u64::MAX] {
            assert_eq!(parse_hex64(&hex64(value)).unwrap(), value);
        }
        // Signed zero has a canonical, distinct bit representation. Audit,
        // rather than a numeric-equality coercion, decides whether it matches.
        assert!(finite64("8000000000000000").is_ok());
    }

    #[test]
    fn valid_shape_tampering_still_requires_fresh_likelihood_and_actor_audit() {
        let model = integration_fixture::model(false);
        let policy = integration_fixture::handle(&model);
        let bytes = encode_record(&collected(&policy, 3)).unwrap();
        let original: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        for pointer in [
            "/callbacks/0/sample/trace/behaviorLogpBits",
            "/callbacks/0/sample/trace/logitsDigest",
        ] {
            let mut corrupt = original.clone();
            let text = corrupt.pointer_mut(pointer).unwrap().as_str().unwrap();
            let mut replacement = text.to_owned();
            let last = replacement.pop().unwrap();
            replacement.push(if last == '0' { '1' } else { '0' });
            *corrupt.pointer_mut(pointer).unwrap() = replacement.into();
            let changed = serde_json::to_vec(&corrupt).unwrap();
            assert!(decode_record(&changed).is_ok(), "shape remains canonical");
            assert!(audit_record_bytes(&changed, &policy).is_err(), "{pointer}");
        }
        let mut corrupt = original;
        let actor = corrupt["callbacks"][0]["actor"].as_u64().unwrap();
        corrupt["callbacks"][0]["actor"] = serde_json::json!((actor + 1) % 3);
        let changed = serde_json::to_vec(&corrupt).unwrap();
        assert!(decode_record(&changed).is_ok());
        assert!(audit_record_bytes(&changed, &policy).is_err());
    }

    #[test]
    fn actual_record_96mib_boundary_and_counted_serialization_fail_closed() {
        let model = integration_fixture::model(false);
        let policy = integration_fixture::handle(&model);
        let mut bytes = encode_record(&collected(&policy, 3)).unwrap();
        bytes.resize(MAX_RECORD_BYTES, b' ');
        assert!(
            decode_record(&bytes).is_ok(),
            "exact bound admits valid JSON with whitespace"
        );
        bytes.push(b' ');
        let error = audit_record_bytes(&bytes, &policy).err().unwrap();
        assert_eq!(error, "Stochastic record byte bound");
        let escaped = serde_json::json!(["\n\"\\", "日本語"]);
        let actual = serde_json::to_vec(&escaped).unwrap();
        assert_eq!(
            serialized_size(&escaped, actual.len()).unwrap(),
            actual.len()
        );
        assert!(serialized_size(&escaped, actual.len() - 1).is_err());
        assert!(serialized_size(&escaped, 0).is_err());
        let mut counter = Counter {
            used: usize::MAX,
            maximum: usize::MAX,
        };
        assert!(counter.write(b"x").is_err());
    }
}
