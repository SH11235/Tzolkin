//! Sealed whole-four admission for one immutable immediate joint parent.
use crate::dataset::{DatasetSplit, seed_family_id, split_for_family};
use crate::public_policy_paired_long::{COHORT_VERSION, EPISODE_VERSION, SAMPLING_VERSION};
use crate::public_policy_update::{checkpoint_digest, hash};
use crate::public_rl_native::PairedRlHandle;
use crate::public_rl_paired_policy_episode::ValidatedPairedEpisode;
use crate::public_stochastic::PairedSamplingPolicy;
use crate::public_stochastic_native::{NativeStochasticConfig, numerical_target};
use crate::public_stochastic_record::Config;

pub struct PlannedPairedCohort {
    source: PairedSamplingPolicy,
    configurations: [NativeStochasticConfig; 4],
    families: [String; 4],
    target: String,
    checksum: String,
}
impl PlannedPairedCohort {
    pub fn new(
        parent: &PairedRlHandle<'_>,
        configurations: [NativeStochasticConfig; 4],
    ) -> Result<Self, String> {
        parent.check_rollout_capacity(4)?;
        if configurations.iter().filter(|c| c.players() == 3).count() != 2
            || configurations.iter().filter(|c| c.players() == 4).count() != 2
        {
            return Err("Paired cohort requires two 3p and two 4p games".into());
        }
        let source = PairedSamplingPolicy::from_handle(parent)?;
        let families =
            std::array::from_fn(|i| seed_family_id(configurations[i].environment_seed()));
        for (i, c) in configurations.iter().enumerate() {
            if split_for_family(&families[i])? != DatasetSplit::Train
                || parent.contains_family(&families[i])
                || families[..i].contains(&families[i])
                || configurations[..i]
                    .iter()
                    .any(|old| old.sampling_identity() == c.sampling_identity())
            {
                return Err(
                    "Paired cohort requires fresh Train families and distinct streams".into(),
                );
            }
        }
        let target = numerical_target();
        let checksum = hash(
            b"tzolkin-planned-paired-cohort-v1\0",
            &(
                COHORT_VERSION,
                EPISODE_VERSION,
                SAMPLING_VERSION,
                crate::public_policy_episode::TARGET_CONTRACT,
                crate::replay::RULES_VERSION,
                crate::replay::catalog_hash(),
                &target,
                &source,
                configurations.each_ref().map(Config::from_config),
                &families,
            ),
        )?;
        Ok(Self {
            source,
            configurations,
            families,
            target,
            checksum,
        })
    }
    pub fn checksum(&self) -> &str {
        &self.checksum
    }
    pub fn parent_artifact_checksum(&self) -> &str {
        self.source.artifact_checksum()
    }
    pub fn parent_update_count(&self) -> u64 {
        self.source.update_count()
    }
    pub fn configurations(&self) -> &[NativeStochasticConfig; 4] {
        &self.configurations
    }
    pub fn rollout_families(&self) -> &[String; 4] {
        &self.families
    }
    pub fn numerical_target(&self) -> &str {
        &self.target
    }
    pub fn finish(
        self,
        results: [Result<ValidatedPairedEpisode, String>; 4],
    ) -> Result<ValidatedPairedCohort, String> {
        let [a, b, c, d] = results;
        let episodes = [a?, b?, c?, d?];
        let mut counts = [0usize; 3];
        for (i, e) in episodes.iter().enumerate() {
            if e.parent_policy() != &self.source
                || e.source_config() != &self.configurations[i]
                || e.family_id() != self.families[i]
                || e.episode_contract() != EPISODE_VERSION
                || e.numerical_target() != self.target
                || !checkpoint_digest(e.canonical_record_checksum())
                || e.is_empty()
            {
                return Err("Whole paired cohort member/source/parent differs".into());
            }
            let mut n = [0usize; 3];
            for s in e.steps() {
                n[0] += 1;
                n[1] += s.observation().legal_actions.len();
                n[2] += usize::from(s.observation().legal_actions.len() == 1);
            }
            for (total, amount) in counts.iter_mut().zip(n) {
                *total = total
                    .checked_add(amount)
                    .ok_or("Paired cohort count overflow")?;
            }
        }
        if counts[0] > 4 * crate::public_stochastic::MAX_SAMPLES
            || counts[1] > 4 * crate::public_stochastic::MAX_RESERVED_CANDIDATE_ROWS
        {
            return Err("Paired receipt cap".into());
        }
        let checksum = hash(
            b"tzolkin-completed-paired-cohort-v1\0",
            &(
                &self.checksum,
                episodes.each_ref().map(|e| e.canonical_record_checksum()),
                counts,
            ),
        )?;
        Ok(ValidatedPairedCohort {
            plan: self,
            episodes,
            counts,
            checksum,
        })
    }
}
pub struct ValidatedPairedCohort {
    plan: PlannedPairedCohort,
    episodes: [ValidatedPairedEpisode; 4],
    counts: [usize; 3],
    checksum: String,
}
impl ValidatedPairedCohort {
    pub fn plan(&self) -> &PlannedPairedCohort {
        &self.plan
    }
    pub fn episodes(&self) -> &[ValidatedPairedEpisode; 4] {
        &self.episodes
    }
    pub fn receipt_checksum(&self) -> &str {
        &self.checksum
    }
    pub fn total_callbacks(&self) -> usize {
        self.counts[0]
    }
    pub fn total_candidate_rows(&self) -> usize {
        self.counts[1]
    }
    pub fn total_singletons(&self) -> usize {
        self.counts[2]
    }
    pub fn actor_coefficient(&self, member: usize, actor: usize) -> Result<f64, String> {
        let c = self
            .plan
            .configurations
            .get(member)
            .ok_or("Paired member index")?;
        if actor >= c.players() {
            return Err("Paired actor index".into());
        }
        Ok(1.0 / (4 * c.players()) as f64)
    }
}
