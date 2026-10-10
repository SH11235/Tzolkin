//! Native collection from a sealed Scalar count1 RL owner. BC remains a
//! separate API/schema; these results cannot become a BC episode or GameReplay.
//! Freshness is a Train split/lineage check, not authentication of registration
//! time, producer identity or independent environment/sampling seed origins.
use tzolkin_core::observation::{MOVE_SCHEMA, OBSERVATION_SCHEMA};

use crate::dataset::{DatasetSplit, seed_family_id, split_for_family};
use crate::public_rl_native::UpdatedPublicRlHandle;
use crate::public_stochastic::{
    MAX_SAMPLES, RL_SAMPLING_VERSION, RNG_VERSION, RlSamplingPolicy, RlStochasticSession,
    UNIFORM_MIXTURE,
};
use crate::public_stochastic_native::{
    NativePolicy, NativeStochasticConfig, numerical_target, run_shared,
};
use crate::public_stochastic_record::{
    AttemptStage, Config, FailureReason, Header, Record, decode_typed_record, encode_typed_record,
    hex64,
};
use crate::replay::{RULES_BASELINE, RULES_VERSION, catalog_hash};

pub const RECORD_SCHEMA: &str = "tzolkin-public-stochastic-rl-count1-native-record-v1";
pub const SOURCE_KIND: &str = "publicStochasticRlCount1";

/// Sealed collection, including failed prefixes. No Deserialize, raw constructor
/// or conversion into an existing BC/native training source is provided.
pub struct CollectedRlStochasticGame {
    record: Record<RlSamplingPolicy>,
}
impl CollectedRlStochasticGame {
    pub fn complete(&self) -> bool {
        self.record.terminal.is_some() && self.record.failure.is_none()
    }
    pub fn observed_callbacks(&self) -> usize {
        self.record.counts.observed_callbacks
    }
    pub fn sampler_attempts(&self) -> usize {
        self.record.counts.sampler_attempts
    }
    pub fn accepted_samples(&self) -> usize {
        self.record.counts.accepted_samples
    }
    pub fn candidate_rows_reserved(&self) -> usize {
        self.record.counts.candidate_rows_reserved
    }
    pub fn applied_choices(&self) -> usize {
        self.record.counts.apply_successes
    }
    pub fn terminal_state_key(&self) -> Option<&str> {
        self.record
            .terminal
            .as_ref()
            .map(|t| t.final_state.as_str())
    }
    pub fn failure_reason(&self) -> Option<FailureReason> {
        self.record.failure.as_ref().map(|f| f.reason)
    }
    pub fn failure_stage(&self) -> Option<AttemptStage> {
        self.record.failure.as_ref().map(|f| f.stage)
    }
    pub fn failure_callback_index(&self) -> Option<usize> {
        self.record
            .failure
            .as_ref()
            .and_then(|f| f.global_callback_index)
    }
    pub fn failure_message(&self) -> Option<&str> {
        self.record.failure.as_ref().map(|f| f.message.as_str())
    }
}

/// A fresh same-target mechanical audit, not a stored completion flag or
/// permission to train. Both complete games and reproducible prefixes survive.
pub struct AuditedRlStochasticGame {
    record: Record<RlSamplingPolicy>,
}
impl AuditedRlStochasticGame {
    pub fn complete(&self) -> bool {
        let record = self.record();
        record.terminal.is_some() && record.failure.is_none()
    }
    pub fn observed_callbacks(&self) -> usize {
        self.record.counts.observed_callbacks
    }
    pub fn sampler_attempts(&self) -> usize {
        self.record.counts.sampler_attempts
    }
    pub fn accepted_samples(&self) -> usize {
        self.record.counts.accepted_samples
    }
    pub fn candidate_rows_reserved(&self) -> usize {
        self.record.counts.candidate_rows_reserved
    }
    pub fn applied_choices(&self) -> usize {
        self.record.counts.apply_successes
    }
    pub fn terminal_state_key(&self) -> Option<&str> {
        self.record
            .terminal
            .as_ref()
            .map(|t| t.final_state.as_str())
    }
    pub fn failure_reason(&self) -> Option<FailureReason> {
        self.record.failure.as_ref().map(|f| f.reason)
    }
    pub fn failure_stage(&self) -> Option<AttemptStage> {
        self.record.failure.as_ref().map(|f| f.stage)
    }
    pub fn failure_callback_index(&self) -> Option<usize> {
        self.record
            .failure
            .as_ref()
            .and_then(|f| f.global_callback_index)
    }
    pub fn failure_message(&self) -> Option<&str> {
        self.record.failure.as_ref().map(|f| f.message.as_str())
    }
    pub fn training_admission_available(&self) -> bool {
        false
    }
    pub fn producer_authenticated(&self) -> bool {
        false
    }
    pub fn independent_seed_origins_verified(&self) -> bool {
        false
    }
    pub(crate) fn record(&self) -> &Record<RlSamplingPolicy> {
        &self.record
    }
}

pub(crate) fn header(
    config: &NativeStochasticConfig,
    policy: &UpdatedPublicRlHandle<'_>,
) -> Result<Header<RlSamplingPolicy>, String> {
    let family = seed_family_id(config.environment_seed());
    if split_for_family(&family)? != DatasetSplit::Train
        || policy.family_closure().contains(&family)
    {
        return Err("RL native collection requires Train outside inherited family closure".into());
    }
    let base_policy = RlSamplingPolicy::from_handle(policy)?;
    // No forward/draw occurs here. The shared run creates a fresh session only
    // after header comparison and source payload admission, exactly as BC does.
    RlStochasticSession::new(policy, *config.sampling_identity())?;
    Ok(Header {
        source_kind: SOURCE_KIND.into(),
        rules_version: RULES_VERSION,
        rules_baseline: RULES_BASELINE.into(),
        catalog_hash: catalog_hash(),
        observation_schema: OBSERVATION_SCHEMA,
        move_schema: MOVE_SCHEMA,
        config: Config::from_config(config),
        base_policy,
        sampling_version: RL_SAMPLING_VERSION.into(),
        rng_version: RNG_VERSION.into(),
        denominator_bits: 53,
        uniform_mixture_bits: hex64(UNIFORM_MIXTURE.to_bits()),
        backend: policy.backend().into(),
        numerical_target: numerical_target(),
        native_family_id: family,
        raw_artifact_identity_available: false,
        producer_authenticated: false,
        independent_seed_origins_verified: false,
    })
}

/// All actual decision actors use a single fresh RL session. It retains every
/// accepted draw, including an unapplied choice; no retry/fallback is possible.
pub fn collect_rl_native(
    config: &NativeStochasticConfig,
    policy: &UpdatedPublicRlHandle<'_>,
) -> Result<CollectedRlStochasticGame, String> {
    let record = run_shared(
        config,
        NativePolicy::Rl(policy),
        RECORD_SCHEMA,
        header(config, policy)?,
        None,
    )?;
    Ok(CollectedRlStochasticGame { record })
}

/// Only a collected sealed result is serializable through this entrypoint.
pub fn encode_rl_record(game: &CollectedRlStochasticGame) -> Result<Vec<u8>, String> {
    encode_typed_record(&game.record)
}

/// Strict duplicate/shape/required-null/EOF checks precede a fresh run.
/// Source observations are compared before NN use; they never enter the model.
pub fn audit_rl_record_bytes(
    bytes: &[u8],
    policy: &UpdatedPublicRlHandle<'_>,
) -> Result<AuditedRlStochasticGame, String> {
    let record: Record<RlSamplingPolicy> = decode_typed_record(bytes)?;
    audit_rl_record(&record, policy)
}

/// Internal path for a record already strictly decoded from pinned saved bytes.
/// The source/header checks and fresh replay are shared with the public codec.
pub(crate) fn audit_rl_record(
    record: &Record<RlSamplingPolicy>,
    policy: &UpdatedPublicRlHandle<'_>,
) -> Result<AuditedRlStochasticGame, String> {
    if record.schema != RECORD_SCHEMA
        || record.header.source_kind != SOURCE_KIND
        || record.callbacks.len() > MAX_SAMPLES
        || record.training_admission != "unavailable-distinct-codec-required"
    {
        return Err("Unsupported RL stochastic record schema/kind/admission/count".into());
    }
    let config = record.header.config.checked_config()?;
    if record.callbacks.len() > config.limits().max_callbacks() {
        return Err("RL callback count exceeds declared cap".into());
    }
    let checked = run_shared(
        &config,
        NativePolicy::Rl(policy),
        RECORD_SCHEMA,
        header(&config, policy)?,
        Some(record),
    )?;
    Ok(AuditedRlStochasticGame { record: checked })
}
