//! Long-v2 all-seat collection and fresh same-target mechanical audit. This
//! distinct source cannot enter BC/v1 replay or authorize stored completion.
//! Train/lineage exclusion is checked before world creation or NN work.
use tzolkin_core::observation::{MOVE_SCHEMA, OBSERVATION_SCHEMA};

use crate::dataset::{DatasetSplit, seed_family_id, split_for_family};
use crate::public_policy_long::SAMPLING_VERSION;
use crate::public_rl_native::LongRlHandle;
use crate::public_stochastic::{
    LongStochasticSession, MAX_SAMPLES, RNG_VERSION, RlSamplingPolicy, UNIFORM_MIXTURE,
};
use crate::public_stochastic_native::{
    NativePolicy, NativeStochasticConfig, numerical_target, run_shared,
};
use crate::public_stochastic_record::{
    AttemptStage, Config, FailureReason, Header, Record, decode_typed_record, encode_typed_record,
    hex64,
};
use crate::replay::{RULES_BASELINE, RULES_VERSION, catalog_hash};

pub const RECORD_SCHEMA: &str = "tzolkin-public-stochastic-rl-long-native-record-v2";
pub const SOURCE_KIND: &str = "publicStochasticRlLongV2";

pub struct CollectedLongRlStochasticGame {
    record: Record<RlSamplingPolicy>,
}
pub struct AuditedLongRlStochasticGame {
    record: Record<RlSamplingPolicy>,
}
macro_rules! summary {
    ($ty:ty) => {
        impl $ty {
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
    };
}
summary!(CollectedLongRlStochasticGame);
summary!(AuditedLongRlStochasticGame);
impl AuditedLongRlStochasticGame {
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

fn header(
    config: &NativeStochasticConfig,
    policy: &LongRlHandle<'_>,
) -> Result<Header<RlSamplingPolicy>, String> {
    policy.check_rollout_capacity(4)?;
    let family = seed_family_id(config.environment_seed());
    if split_for_family(&family)? != DatasetSplit::Train || policy.contains_family(&family) {
        return Err("Long native collection requires Train outside inherited lineage".into());
    }
    let base_policy = RlSamplingPolicy::from_long_handle(policy)?;
    LongStochasticSession::new(policy, *config.sampling_identity())?;
    Ok(Header {
        source_kind: SOURCE_KIND.into(),
        rules_version: RULES_VERSION,
        rules_baseline: RULES_BASELINE.into(),
        catalog_hash: catalog_hash(),
        observation_schema: OBSERVATION_SCHEMA,
        move_schema: MOVE_SCHEMA,
        config: Config::from_config(config),
        base_policy,
        sampling_version: SAMPLING_VERSION.into(),
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
/// Failed prefixes and accepted-but-unapplied draws are retained by the shared
/// actual loop; there is no retry or heuristic fallback.
pub fn collect_long_rl_native(
    config: &NativeStochasticConfig,
    policy: &LongRlHandle<'_>,
) -> Result<CollectedLongRlStochasticGame, String> {
    let record = run_shared(
        config,
        NativePolicy::Long(policy),
        RECORD_SCHEMA,
        header(config, policy)?,
        None,
    )?;
    Ok(CollectedLongRlStochasticGame { record })
}
pub fn encode_long_rl_record(game: &CollectedLongRlStochasticGame) -> Result<Vec<u8>, String> {
    encode_typed_record(&game.record)
}
/// Closed schema/EOF/required-null checks and exact actual header comparison
/// precede fresh replay. Only actual observations enter the model.
pub fn audit_long_rl_record_bytes(
    bytes: &[u8],
    policy: &LongRlHandle<'_>,
) -> Result<AuditedLongRlStochasticGame, String> {
    let record: Record<RlSamplingPolicy> = decode_typed_record(bytes)?;
    if record.schema != RECORD_SCHEMA
        || record.header.source_kind != SOURCE_KIND
        || record.callbacks.len() > MAX_SAMPLES
        || record.training_admission != "unavailable-distinct-codec-required"
    {
        return Err("Unsupported Long-v2 stochastic schema/kind/admission/count".into());
    }
    let config = record.header.config.checked_config()?;
    if record.callbacks.len() > config.limits().max_callbacks() {
        return Err("Long callback count exceeds declared cap".into());
    }
    let checked = run_shared(
        &config,
        NativePolicy::Long(policy),
        RECORD_SCHEMA,
        header(&config, policy)?,
        Some(&record),
    )?;
    Ok(AuditedLongRlStochasticGame { record: checked })
}
