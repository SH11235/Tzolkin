//! Sealed, complete RL actor trajectories from a trusted fresh native audit.
//! Targets and own-actor links share the BC numeric implementation; source,
//! identity and checksum domains remain distinct. This is no update authority.
use crate::dataset::seed_family_id;
use crate::public_policy_episode::{
    ActorLink, AppliedActorStep, TARGET_CONTRACT, applied_plan, applied_step,
    canonical_record_checksum,
};
use crate::public_rl_stochastic_native::{AuditedRlStochasticGame, RECORD_SCHEMA, SOURCE_KIND};
use crate::public_stochastic::{
    DENOMINATOR_BITS, RL_SAMPLING_VERSION, RNG_VERSION, RlSamplingPolicy, UNIFORM_MIXTURE,
};
use crate::public_stochastic_native::{NativeStochasticConfig, numerical_target};
use crate::public_stochastic_record::hex64;
use crate::replay::{RULES_BASELINE, RULES_VERSION, catalog_hash};
use tzolkin_core::observation::{MOVE_SCHEMA, OBSERVATION_SCHEMA};

pub const EPISODE_CONTRACT: &str = "applied-rl-count1-actor-gamma1-terminal-winner-share-v1";

/// Owns only a freshly audited RL record. No raw constructor, deserialization,
/// BC episode conversion or model selection is exposed.
pub struct ValidatedRlPolicyEpisode {
    source: AuditedRlStochasticGame,
    config: NativeStochasticConfig,
    links: Vec<ActorLink>,
    ranks: Vec<usize>,
    winners: usize,
    checksum: String,
}
impl ValidatedRlPolicyEpisode {
    pub fn from_audited(source: AuditedRlStochasticGame) -> Result<Self, String> {
        let record = source.record();
        let header = &record.header;
        let config = header.config.checked_config()?;
        if record.schema != RECORD_SCHEMA
            || header.source_kind != SOURCE_KIND
            || record.training_admission != "unavailable-distinct-codec-required"
            || record.failure.is_some()
            || header.rules_version != RULES_VERSION
            || header.rules_baseline != RULES_BASELINE
            || header.catalog_hash != catalog_hash()
            || header.observation_schema != OBSERVATION_SCHEMA
            || header.move_schema != MOVE_SCHEMA
            || header.backend != "scalar"
            || header.sampling_version != RL_SAMPLING_VERSION
            || header.rng_version != RNG_VERSION
            || header.denominator_bits != DENOMINATOR_BITS
            || header.uniform_mixture_bits != hex64(UNIFORM_MIXTURE.to_bits())
            || header.native_family_id != seed_family_id(config.environment_seed())
            || header.numerical_target != numerical_target()
            || header.base_policy.update_count() != 1
            || record.callbacks.len() > config.limits().max_callbacks()
        {
            return Err("Incompatible completed RL actor episode source".into());
        }
        // The owned audited source already binds its full DTO to the sealed
        // count1 owner. Here, the same link checks use that distinct RL binding.
        let (links, ranks, winners) = applied_plan(record, &config, RL_SAMPLING_VERSION, || {
            header.base_policy.binding_key()
        })?;
        let checksum =
            canonical_record_checksum(record, b"tzolkin-applied-rl-count1-actor-episode-v1\0")?;
        Ok(Self {
            source,
            config,
            links,
            ranks,
            winners,
            checksum,
        })
    }
    /// Canonical audited content, not original file bytes or producer proof.
    pub fn canonical_record_checksum(&self) -> &str {
        &self.checksum
    }
    pub fn episode_contract(&self) -> &'static str {
        EPISODE_CONTRACT
    }
    pub fn target_contract(&self) -> &'static str {
        TARGET_CONTRACT
    }
    pub fn source_config(&self) -> &NativeStochasticConfig {
        &self.config
    }
    pub fn family_id(&self) -> &str {
        &self.source.record().header.native_family_id
    }
    pub fn parent_artifact_checksum(&self) -> &str {
        self.source.record().header.base_policy.artifact_checksum()
    }
    pub fn parent_update_count(&self) -> u64 {
        self.source.record().header.base_policy.update_count()
    }
    pub fn catalog_hash(&self) -> &str {
        &self.source.record().header.catalog_hash
    }
    pub fn numerical_target(&self) -> &str {
        &self.source.record().header.numerical_target
    }
    pub(crate) fn sampling_version(&self) -> &str {
        &self.source.record().header.sampling_version
    }
    pub fn terminal_state_key(&self) -> &str {
        &self
            .source
            .record()
            .terminal
            .as_ref()
            .expect("validated RL terminal")
            .final_state
    }
    pub fn len(&self) -> usize {
        self.links.len()
    }
    pub fn is_empty(&self) -> bool {
        self.links.is_empty()
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
    pub fn step(&self, index: usize) -> Option<AppliedActorStep<'_>> {
        let link = self.links.get(index)?;
        let callback = &self.source.record().callbacks[index];
        Some(applied_step(
            callback,
            link,
            self.ranks[callback.actor],
            self.winners,
        ))
    }
    pub fn steps(&self) -> impl ExactSizeIterator<Item = AppliedActorStep<'_>> {
        (0..self.len()).map(|index| self.step(index).expect("validated RL actor step"))
    }
    pub(crate) fn parent_policy(&self) -> &RlSamplingPolicy {
        &self.source.record().header.base_policy
    }
}
