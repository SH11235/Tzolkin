//! Closed schema-4 records, not journals, checkpoints or training admission.
//!
//! A decoded payload stays private. Only a fresh Core/Scalar replay against a
//! borrowed source-owned session and its typed pool produces an audit summary.
//! Mixed records name snapshots; no bytes construct a snapshot or an owner.
use serde::{Deserialize, Serialize, de::DeserializeOwned};

use crate::conditional_human_rl_update::ConditionalFrozenPool;
use crate::conditional_semantic_playing_rl::SemanticOwnerView;
use crate::public_stochastic_native::NativeStochasticConfig;
use crate::public_stochastic_record::{Config, serialized_size, unique_json};
use crate::semantic_playing_rl_collection::{
    AuditedSemanticPlayingCohort, AuditedSemanticPlayingEpisode, SemanticCollectionLedger,
};
use crate::semantic_playing_rl_pool::SemanticFrozenMixture;

pub const RECORD_SCHEMA: &str = "tzolkin-semantic-playing-native-record-v1";

// The new wire stores numeric decision bits explicitly. This does not change
// Decision's existing public JSON or any old record, scoring or digest path.
pub(crate) mod decision_bits {
    use serde::{Deserialize, Deserializer, Serialize, Serializer, de::Error};
    use tzolkin_core::GameMove;

    #[derive(Serialize)]
    #[serde(rename_all = "camelCase")]
    struct Ref<'a> {
        actor: usize,
        observation_key: &'a str,
        policy_version: &'a str,
        r#move: &'a GameMove,
        score_bits: u64,
    }
    #[derive(Deserialize)]
    #[serde(rename_all = "camelCase", deny_unknown_fields)]
    struct Owned {
        actor: usize,
        observation_key: String,
        policy_version: String,
        r#move: GameMove,
        score_bits: u64,
    }
    pub(crate) fn serialize<S: Serializer>(
        decision: &crate::Decision,
        serializer: S,
    ) -> Result<S::Ok, S::Error> {
        Ref {
            actor: decision.actor,
            observation_key: &decision.observation_key,
            policy_version: &decision.policy_version,
            r#move: &decision.r#move,
            score_bits: decision.score.to_bits(),
        }
        .serialize(serializer)
    }
    pub(crate) fn deserialize<'de, D: Deserializer<'de>>(
        deserializer: D,
    ) -> Result<crate::Decision, D::Error> {
        let value = Owned::deserialize(deserializer)?;
        let score = f64::from_bits(value.score_bits);
        if !score.is_finite() {
            return Err(D::Error::custom("Semantic decision score must be finite"));
        }
        Ok(crate::Decision {
            actor: value.actor,
            observation_key: value.observation_key,
            policy_version: value.policy_version,
            r#move: value.r#move,
            score,
        })
    }
}

#[derive(Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub(crate) struct Header {
    schema: String,
    feature_schema: u32,
    rules_version: u32,
    rules_baseline: String,
    catalog_hash: String,
    config: Config,
    parent: String,
    member: usize,
    mixture: Option<String>,
    sampling_version: String,
    target_contract: String,
    numerical_target: String,
}
impl Header {
    pub(crate) fn new(
        parent: &str,
        member: usize,
        mixture: Option<&str>,
        config: &NativeStochasticConfig,
    ) -> Result<Self, String> {
        if member >= 4 || config.players() != if member < 2 { 3 } else { 4 } {
            return Err("Semantic record member/player count differs".into());
        }
        Ok(Self {
            schema: RECORD_SCHEMA.into(),
            feature_schema: crate::features::SEMANTIC_PLAYING_FEATURE_SCHEMA,
            rules_version: crate::replay::RULES_VERSION,
            rules_baseline: crate::replay::RULES_BASELINE.into(),
            catalog_hash: crate::replay::catalog_hash(),
            config: Config::from_config(config),
            parent: parent.into(),
            member,
            mixture: mixture.map(str::to_owned),
            sampling_version: crate::semantic_playing_rl_collection::SAMPLING_VERSION.into(),
            target_contract: crate::semantic_playing_rl_collection::TARGET_CONTRACT.into(),
            numerical_target: crate::public_stochastic_native::numerical_target(),
        })
    }
}

#[derive(Deserialize, Serialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct Wire<R> {
    header: Header,
    record: R,
}
#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct WireRef<'a, R> {
    header: &'a Header,
    record: &'a R,
}

fn maximum(config: &NativeStochasticConfig) -> Result<usize, String> {
    config
        .limits()
        .max_source_bytes()
        .checked_add(config.limits().max_trace_bytes())
        .filter(|n| *n <= crate::public_stochastic_record::MAX_RECORD_BYTES)
        .ok_or_else(|| "Semantic record byte cap overflow".into())
}
pub(crate) fn encode<R: Serialize>(
    header: &Header,
    record: &R,
    config: &NativeStochasticConfig,
) -> Result<Vec<u8>, String> {
    let wire = WireRef { header, record };
    serialized_size(&wire, maximum(config)?)?;
    serde_json::to_vec(&wire).map_err(|e| e.to_string())
}
pub(crate) fn decode<R: DeserializeOwned + Serialize>(
    raw: &[u8],
    expected: &Header,
    config: &NativeStochasticConfig,
) -> Result<R, String> {
    let wire: Wire<R> =
        serde_json::from_value(unique_json(raw, maximum(config)?)?).map_err(|e| e.to_string())?;
    if wire.header != *expected || wire.header.config.checked_config()? != *config {
        return Err("Semantic record header/source/profile differs".into());
    }
    // The sole producer emits this canonical encoding. This also refuses any
    // ignored fields inside nested Core DTOs, alternate floats and trailing data.
    if encode(&wire.header, &wire.record, config)? != raw {
        return Err("Semantic record is not the exact closed canonical wire".into());
    }
    Ok(wire.record)
}

/// Encode all four already audited episodes in their original member order.
/// This does not export an actor, construct ownership or publish any file.
pub fn encode_records(cohort: &AuditedSemanticPlayingCohort) -> Result<[Vec<u8>; 4], String> {
    crate::semantic_playing_rl_collection::encode_records(cohort)
}

/// Lossless bounded storage; a stored payload itself carries no audit authority.
pub struct StoredSemanticPlayingRecord {
    stored: Vec<u8>,
    raw_bytes: usize,
    raw_sha256: String,
}
impl StoredSemanticPlayingRecord {
    pub fn stored_bytes(&self) -> &[u8] {
        &self.stored
    }
    pub fn raw_bytes(&self) -> usize {
        self.raw_bytes
    }
    pub fn raw_sha256(&self) -> &str {
        &self.raw_sha256
    }
}
pub fn store_records(
    cohort: &AuditedSemanticPlayingCohort,
) -> Result<[StoredSemanticPlayingRecord; 4], String> {
    encode_records(cohort)?
        .into_iter()
        .map(|raw| {
            Ok(StoredSemanticPlayingRecord {
                stored: crate::public_rl_record_storage::encode(&raw)?,
                raw_bytes: raw.len(),
                raw_sha256: crate::model::digest(&raw),
            })
        })
        .collect::<Result<Vec<_>, String>>()?
        .try_into()
        .map_err(|_| "Semantic stored whole-four shape".into())
}

/// A descriptive fresh-audit result, never an ActorStep/owner/continuation token.
#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct SemanticPlayingRecordAudit {
    raw_sha256: String,
    callbacks: usize,
    learner_setup_choices: usize,
    learner_playing_samples: usize,
    opponent_playing_samples: usize,
    candidate_rows: usize,
    actor_draws: [u64; 4],
    training_admission_available: bool,
    source_authenticated: bool,
    source_qualified: bool,
    verified_native_policy: bool,
}
impl SemanticPlayingRecordAudit {
    pub(crate) fn new(raw: &[u8], episode: &AuditedSemanticPlayingEpisode) -> Self {
        Self {
            raw_sha256: crate::model::digest(raw),
            callbacks: episode.observed_callbacks(),
            learner_setup_choices: episode.learner_setup_choices(),
            learner_playing_samples: episode.playing_steps(),
            opponent_playing_samples: episode.opponent_playing_samples(),
            candidate_rows: episode.total_sampled_candidate_rows(),
            actor_draws: *episode.actor_draws(),
            training_admission_available: false,
            source_authenticated: false,
            source_qualified: false,
            verified_native_policy: false,
        }
    }
}

/// Borrowed current source/ledger; only a genuine collection or joint session
/// can supply it. Auditing does not reserve, draw from the live ledger or update.
pub struct SemanticPlayingRecordAuditor<'state, 'inputs> {
    pub(crate) owner: SemanticOwnerView<'state>,
    pub(crate) ledger: &'state SemanticCollectionLedger<'inputs>,
}
impl SemanticPlayingRecordAuditor<'_, '_> {
    pub fn audit_procedural(
        &self,
        raw: &[u8],
        config: &NativeStochasticConfig,
        member: usize,
        pool: &ConditionalFrozenPool,
    ) -> Result<SemanticPlayingRecordAudit, String> {
        self.ledger
            .audit_procedural_record(&self.owner, raw, config, member, pool)
    }
    pub fn audit_mixed(
        &self,
        raw: &[u8],
        config: &NativeStochasticConfig,
        member: usize,
        mixture: &SemanticFrozenMixture<'_>,
    ) -> Result<SemanticPlayingRecordAudit, String> {
        self.ledger
            .audit_mixed_record(&self.owner, raw, config, member, mixture)
    }
    #[allow(clippy::too_many_arguments)]
    pub fn audit_stored_procedural(
        &self,
        stored: &[u8],
        raw_bytes: usize,
        raw_sha256: &str,
        config: &NativeStochasticConfig,
        member: usize,
        pool: &ConditionalFrozenPool,
    ) -> Result<SemanticPlayingRecordAudit, String> {
        if raw_bytes > maximum(config)? {
            return Err("Semantic stored raw length exceeds this configuration".into());
        }
        let raw = crate::public_rl_record_storage::decode(stored, raw_bytes, raw_sha256)?;
        self.audit_procedural(&raw, config, member, pool)
    }
    #[allow(clippy::too_many_arguments)]
    pub fn audit_stored_mixed(
        &self,
        stored: &[u8],
        raw_bytes: usize,
        raw_sha256: &str,
        config: &NativeStochasticConfig,
        member: usize,
        mixture: &SemanticFrozenMixture<'_>,
    ) -> Result<SemanticPlayingRecordAudit, String> {
        if raw_bytes > maximum(config)? {
            return Err("Semantic stored raw length exceeds this configuration".into());
        }
        let raw = crate::public_rl_record_storage::decode(stored, raw_bytes, raw_sha256)?;
        self.audit_mixed(&raw, config, member, mixture)
    }
}
