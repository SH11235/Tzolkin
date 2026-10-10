//! Complete paired trajectories come only from a fresh actual native audit.
use crate::public_policy_episode::{
    ActorLink, AppliedActorStep, applied_plan, applied_step, canonical_record_checksum,
};
use crate::public_policy_paired_long::{EPISODE_VERSION, SAMPLING_VERSION};
use crate::public_rl_paired_stochastic_native::{
    AuditedPairedRlStochasticGame, RECORD_SCHEMA, SOURCE_KIND,
};
use crate::public_stochastic::{DENOMINATOR_BITS, RNG_VERSION, UNIFORM_MIXTURE};
use crate::public_stochastic_native::{NativeStochasticConfig, numerical_target};
use crate::public_stochastic_record::hex64;
use crate::replay::{RULES_BASELINE, RULES_VERSION, catalog_hash};
use tzolkin_core::observation::{MOVE_SCHEMA, OBSERVATION_SCHEMA};

pub struct ValidatedPairedEpisode {
    source: AuditedPairedRlStochasticGame,
    config: NativeStochasticConfig,
    links: Vec<ActorLink>,
    ranks: Vec<usize>,
    winners: usize,
    checksum: String,
}
impl ValidatedPairedEpisode {
    pub fn from_audited(source: AuditedPairedRlStochasticGame) -> Result<Self, String> {
        let record = source.record();
        let h = &record.header;
        let config = h.config.checked_config()?;
        if record.schema != RECORD_SCHEMA
            || h.source_kind != SOURCE_KIND
            || record.failure.is_some()
            || record.training_admission != "unavailable-distinct-codec-required"
            || h.rules_version != RULES_VERSION
            || h.rules_baseline != RULES_BASELINE
            || h.catalog_hash != catalog_hash()
            || h.observation_schema != OBSERVATION_SCHEMA
            || h.move_schema != MOVE_SCHEMA
            || h.backend != "scalar"
            || h.sampling_version != SAMPLING_VERSION
            || h.rng_version != RNG_VERSION
            || h.denominator_bits != DENOMINATOR_BITS
            || h.uniform_mixture_bits != hex64(UNIFORM_MIXTURE.to_bits())
            || h.numerical_target != numerical_target()
            || h.native_family_id != crate::dataset::seed_family_id(config.environment_seed())
            || record.callbacks.len() > config.limits().max_callbacks()
            || !(2..=crate::public_policy_paired_long::MAX_UPDATES)
                .contains(&h.base_policy.update_count())
        {
            return Err("Incompatible complete paired episode".into());
        }
        let (links, ranks, winners) = applied_plan(record, &config, SAMPLING_VERSION, || {
            h.base_policy.binding_key()
        })?;
        let checksum =
            canonical_record_checksum(record, b"tzolkin-applied-paired-actor-episode-v1\0")?;
        Ok(Self {
            source,
            config,
            links,
            ranks,
            winners,
            checksum,
        })
    }
    pub fn canonical_record_checksum(&self) -> &str {
        &self.checksum
    }
    pub fn episode_contract(&self) -> &'static str {
        EPISODE_VERSION
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
    pub(crate) fn parent_policy(&self) -> &crate::public_stochastic::PairedSamplingPolicy {
        &self.source.record().header.base_policy
    }
    pub fn numerical_target(&self) -> &str {
        &self.source.record().header.numerical_target
    }
    pub fn len(&self) -> usize {
        self.links.len()
    }
    pub fn is_empty(&self) -> bool {
        self.links.is_empty()
    }
    pub fn steps(&self) -> impl ExactSizeIterator<Item = AppliedActorStep<'_>> {
        (0..self.len()).map(|i| {
            let c = &self.source.record().callbacks[i];
            applied_step(c, &self.links[i], self.ranks[c.actor], self.winners)
        })
    }
}
