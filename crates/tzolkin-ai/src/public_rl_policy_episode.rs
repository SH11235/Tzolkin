//! Sealed, complete RL actor trajectories from a trusted fresh native audit.
//! Targets and own-actor links share the BC numeric implementation; source,
//! identity and checksum domains remain distinct. This is no update authority.
use crate::dataset::seed_family_id;
use crate::public_policy_episode::{
    ActorLink, AppliedActorStep, TARGET_CONTRACT, applied_plan, applied_step,
    canonical_record_checksum,
};
use crate::public_policy_repeat::{MAX_UPDATE_COUNT, REPEATED_SAMPLING_VERSION};
use crate::public_rl_repeated_stochastic_native::{
    AuditedRepeatedRlStochasticGame, RECORD_SCHEMA as REPEATED_RECORD_SCHEMA,
    SOURCE_KIND as REPEATED_SOURCE_KIND,
};
use crate::public_rl_stochastic_native::{AuditedRlStochasticGame, RECORD_SCHEMA, SOURCE_KIND};
use crate::public_stochastic::{
    DENOMINATOR_BITS, RL_SAMPLING_VERSION, RNG_VERSION, RlSamplingPolicy, UNIFORM_MIXTURE,
};
use crate::public_stochastic_native::{NativeStochasticConfig, numerical_target};
use crate::public_stochastic_record::Record;
use crate::public_stochastic_record::hex64;
use crate::replay::{RULES_BASELINE, RULES_VERSION, catalog_hash};
use tzolkin_core::observation::{MOVE_SCHEMA, OBSERVATION_SCHEMA};

pub const EPISODE_CONTRACT: &str = "applied-rl-count1-actor-gamma1-terminal-winner-share-v1";
pub const REPEATED_EPISODE_CONTRACT: &str =
    "applied-rl-repeat-actor-gamma1-terminal-winner-share-v1";

pub const LONG_EPISODE_CONTRACT: &str = "applied-rl-long-actor-gamma1-terminal-winner-share-v2";

enum EpisodeSource {
    Count1(AuditedRlStochasticGame),
    Repeated(AuditedRepeatedRlStochasticGame),
    Long(crate::public_rl_long_stochastic_native::AuditedLongRlStochasticGame),
}
struct SourceContract {
    schema: &'static str,
    kind: &'static str,
    sampling: &'static str,
    episode: &'static str,
    domain: &'static [u8],
    counts: std::ops::RangeInclusive<u64>,
}
impl EpisodeSource {
    fn record(&self) -> &Record<RlSamplingPolicy> {
        match self {
            Self::Count1(source) => source.record(),
            Self::Repeated(source) => source.record(),
            Self::Long(source) => source.record(),
        }
    }
    fn contract(&self) -> SourceContract {
        match self {
            Self::Count1(_) => SourceContract {
                schema: RECORD_SCHEMA,
                kind: SOURCE_KIND,
                sampling: RL_SAMPLING_VERSION,
                episode: EPISODE_CONTRACT,
                domain: b"tzolkin-applied-rl-count1-actor-episode-v1\0",
                counts: 1..=1,
            },
            Self::Repeated(_) => SourceContract {
                schema: REPEATED_RECORD_SCHEMA,
                kind: REPEATED_SOURCE_KIND,
                sampling: REPEATED_SAMPLING_VERSION,
                episode: REPEATED_EPISODE_CONTRACT,
                domain: b"tzolkin-applied-rl-repeat-actor-episode-v1\0",
                counts: 2..=MAX_UPDATE_COUNT,
            },
            Self::Long(_) => SourceContract {
                schema: crate::public_rl_long_stochastic_native::RECORD_SCHEMA,
                kind: crate::public_rl_long_stochastic_native::SOURCE_KIND,
                sampling: crate::public_policy_long::SAMPLING_VERSION,
                episode: LONG_EPISODE_CONTRACT,
                domain: b"tzolkin-applied-rl-long-actor-episode-v2\0",
                counts: 2..=crate::public_policy_long::MAX_UPDATES,
            },
        }
    }
    fn binding_key(&self) -> Result<String, String> {
        let source = &self.record().header.base_policy;
        match self {
            Self::Count1(_) => source.binding_key(),
            Self::Repeated(_) => source.repeated_binding_key(),
            Self::Long(_) => source.long_binding_key(),
        }
    }
}

/// Owns only a freshly audited RL record. No raw constructor, deserialization,
/// BC episode conversion or model selection is exposed.
pub struct ValidatedRlPolicyEpisode {
    source: EpisodeSource,
    config: NativeStochasticConfig,
    links: Vec<ActorLink>,
    ranks: Vec<usize>,
    winners: usize,
    checksum: String,
}
impl ValidatedRlPolicyEpisode {
    pub fn from_audited(source: AuditedRlStochasticGame) -> Result<Self, String> {
        Self::from_source(EpisodeSource::Count1(source))
    }
    pub fn from_repeated_audited(source: AuditedRepeatedRlStochasticGame) -> Result<Self, String> {
        Self::from_source(EpisodeSource::Repeated(source))
    }
    pub fn from_long_audited(
        source: crate::public_rl_long_stochastic_native::AuditedLongRlStochasticGame,
    ) -> Result<Self, String> {
        Self::from_source(EpisodeSource::Long(source))
    }
    fn from_source(source: EpisodeSource) -> Result<Self, String> {
        let contract = source.contract();
        let record = source.record();
        let header = &record.header;
        let config = header.config.checked_config()?;
        if record.schema != contract.schema
            || header.source_kind != contract.kind
            || record.training_admission != "unavailable-distinct-codec-required"
            || record.failure.is_some()
            || header.rules_version != RULES_VERSION
            || header.rules_baseline != RULES_BASELINE
            || header.catalog_hash != catalog_hash()
            || header.observation_schema != OBSERVATION_SCHEMA
            || header.move_schema != MOVE_SCHEMA
            || header.backend != "scalar"
            || header.sampling_version != contract.sampling
            || header.rng_version != RNG_VERSION
            || header.denominator_bits != DENOMINATOR_BITS
            || header.uniform_mixture_bits != hex64(UNIFORM_MIXTURE.to_bits())
            || header.native_family_id != seed_family_id(config.environment_seed())
            || header.numerical_target != numerical_target()
            || !contract.counts.contains(&header.base_policy.update_count())
            || record.callbacks.len() > config.limits().max_callbacks()
        {
            return Err("Incompatible completed RL actor episode source".into());
        }
        // The private source variant is chosen only by a sealed audit factory,
        // never from untrusted metadata. Each role retains its distinct domain.
        let (links, ranks, winners) =
            applied_plan(record, &config, contract.sampling, || source.binding_key())?;
        let checksum = canonical_record_checksum(record, contract.domain)?;
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
        self.source.contract().episode
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
