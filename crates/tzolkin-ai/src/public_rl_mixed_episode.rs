//! Sealed learner-only terminal targets from a complete fresh native audit.
use crate::public_policy_episode::{
    ActorLink, AppliedActorStep, applied_mixed_plan, applied_step, canonical_record_checksum,
};
use crate::public_rl_mixed_native::{
    AuditedMixedRlGame, MixedRlSource, RECORD_SCHEMA, ROSTER_CONTRACT, SOURCE_KIND,
};
use crate::public_stochastic_native::NativeStochasticConfig;

pub const EPISODE_CONTRACT: &str = "applied-mixed-learner-gamma1-terminal-winner-share-v1";
pub struct ValidatedMixedRlEpisode {
    source: AuditedMixedRlGame,
    config: NativeStochasticConfig,
    indices: Vec<usize>,
    links: Vec<ActorLink>,
    rank: usize,
    winners: usize,
    checksum: String,
}
impl ValidatedMixedRlEpisode {
    pub fn from_audited(source: AuditedMixedRlGame) -> Result<Self, String> {
        let record = source.record();
        let h = &record.header;
        let config = h.config.checked_config()?;
        let role = &h.base_policy;
        if !source.complete()
            || record.schema != RECORD_SCHEMA
            || h.source_kind != SOURCE_KIND
            || role.contract != ROSTER_CONTRACT
            || role.learner_seat >= config.players()
            || role.opponents.len() != config.players()
            || role
                .opponents
                .iter()
                .enumerate()
                .any(|(i, p)| p.is_none() != (i == role.learner_seat))
        {
            return Err("Incomplete or incorrectly masked mixed RL episode".into());
        }
        let (links, ranks, winners) = applied_mixed_plan(
            record,
            &config,
            role.learner.sampling_version(),
            || role.learner.binding_key(),
            role.learner_seat,
        )?;
        let indices = record
            .callbacks
            .iter()
            .enumerate()
            .filter(|(_, c)| c.actor == role.learner_seat)
            .map(|(i, _)| i)
            .collect::<Vec<_>>();
        if links.len() != indices.len() {
            return Err("Mixed learner link coverage differs".into());
        }
        let rank = ranks[role.learner_seat];
        let checksum = canonical_record_checksum(record, b"tzolkin-applied-mixed-rl-episode-v1\0")?;
        Ok(Self {
            source,
            config,
            indices,
            links,
            rank,
            winners,
            checksum,
        })
    }
    pub fn episode_contract(&self) -> &'static str {
        EPISODE_CONTRACT
    }
    pub fn canonical_record_checksum(&self) -> &str {
        &self.checksum
    }
    pub fn source_config(&self) -> &NativeStochasticConfig {
        &self.config
    }
    pub fn learner_seat(&self) -> usize {
        self.source_identity().learner_seat
    }
    pub fn frozen_roster_checksum(&self) -> &str {
        &self.source_identity().roster_checksum
    }
    pub fn parent_artifact_checksum(&self) -> &str {
        self.source_identity().learner.checksum()
    }
    pub fn parent_update_count(&self) -> u64 {
        self.source_identity().learner.count()
    }
    pub fn family_id(&self) -> &str {
        &self.source.record().header.native_family_id
    }
    pub fn numerical_target(&self) -> &str {
        &self.source.record().header.numerical_target
    }
    pub fn observed_callbacks(&self) -> usize {
        self.source.observed_callbacks()
    }
    pub fn len(&self) -> usize {
        self.links.len()
    }
    pub fn is_empty(&self) -> bool {
        self.links.is_empty()
    }
    pub(crate) fn source_identity(&self) -> &MixedRlSource {
        &self.source.record().header.base_policy
    }
    pub(crate) fn source_header(&self) -> &crate::public_stochastic_record::Header<MixedRlSource> {
        &self.source.record().header
    }
    pub fn steps(&self) -> impl ExactSizeIterator<Item = AppliedActorStep<'_>> {
        self.indices.iter().zip(&self.links).map(|(&index, link)| {
            applied_step(
                &self.source.record().callbacks[index],
                link,
                self.rank,
                self.winners,
            )
        })
    }
}
