//! Sealed mixed-data whole-four admission. This cannot enter self-play producers.
use crate::dataset::seed_family_id;
use crate::public_policy_update::hash;
use crate::public_rl_mixed_episode::{EPISODE_CONTRACT, ValidatedMixedRlEpisode};
use crate::public_rl_mixed_native::{
    FrozenMixedRoster, MixedRlLearner, MixedRlSource, ROSTER_CONTRACT,
};
use crate::public_stochastic_native::NativeStochasticConfig;
use crate::public_stochastic_record::Header;
use serde_json::json;

pub const COHORT_CONTRACT: &str = "whole4-frozen-mixed-learner-own-sum-quarter-v1";
/// Construct before collection from the genuine learner and frozen opponents.
/// Serialized plan/receipts do not construct this sealed admission object.
pub struct PlannedMixedRlCohort {
    configs: [NativeStochasticConfig; 4],
    headers: [Header<MixedRlSource>; 4],
    families: [String; 4],
    checksum: String,
}
impl PlannedMixedRlCohort {
    pub(crate) fn new(
        learner: &MixedRlLearner<'_>,
        configs: [NativeStochasticConfig; 4],
        rosters: [&FrozenMixedRoster<'_>; 4],
    ) -> Result<Self, String> {
        if configs.iter().filter(|c| c.players() == 3).count() != 2
            || configs.iter().filter(|c| c.players() == 4).count() != 2
        {
            return Err("Mixed cohort requires two 3p and two 4p games".into());
        }
        let families = std::array::from_fn(|i| seed_family_id(configs[i].environment_seed()));
        for (i, c) in configs.iter().enumerate() {
            if families[..i].contains(&families[i])
                || configs[..i]
                    .iter()
                    .any(|p| p.sampling_identity() == c.sampling_identity())
            {
                return Err("Mixed cohort repeats a family or sampling stream".into());
            }
        }
        let headers = configs
            .iter()
            .zip(rosters)
            .map(|(c, r)| learner.header(c, r))
            .collect::<Result<Vec<_>, _>>()?
            .try_into()
            .map_err(|_| "Mixed whole4 header shape")?;
        let checksum = hash(
            b"tzolkin-planned-mixed-rl-cohort-v1\0",
            &(COHORT_CONTRACT, EPISODE_CONTRACT, &headers),
        )?;
        Ok(Self {
            configs,
            headers,
            families,
            checksum,
        })
    }
    pub fn checksum(&self) -> &str {
        &self.checksum
    }
    pub fn configurations(&self) -> &[NativeStochasticConfig; 4] {
        &self.configs
    }
    pub fn rollout_families(&self) -> &[String; 4] {
        &self.families
    }
    pub fn learner_seats(&self) -> [usize; 4] {
        self.headers.each_ref().map(|h| h.base_policy.learner_seat)
    }
    pub fn parent_artifact_checksum(&self) -> &str {
        self.headers[0].base_policy.learner.checksum()
    }
    pub fn parent_update_count(&self) -> u64 {
        self.headers[0].base_policy.learner.count()
    }
    pub fn receipt(&self) -> serde_json::Value {
        json!({"schema":"tzolkin-planned-mixed-rl-cohort-v1","cohortContract":COHORT_CONTRACT,
            "episodeContract":EPISODE_CONTRACT,"rosterContract":ROSTER_CONTRACT,"checksum":self.checksum,
            "headers":self.headers,"families":self.families,"actorNormalization":"learnerOwnSumTimes1/4",
            "criticTargets":"learnerCallbacksOnlyIncludingSingletons","producerAuthenticated":false,"independentSeedOriginsVerified":false})
    }
    pub(crate) fn finish(
        self,
        results: [Result<ValidatedMixedRlEpisode, String>; 4],
    ) -> Result<ValidatedMixedRlCohort, String> {
        let [a, b, c, d] = results;
        let episodes = [a?, b?, c?, d?];
        let mut counts = [0usize; 3];
        let mut observed = 0usize;
        for (i, e) in episodes.iter().enumerate() {
            if e.source_header() != &self.headers[i]
                || e.family_id() != self.families[i]
                || e.source_config() != &self.configs[i]
                || e.is_empty()
                || e.episode_contract() != EPISODE_CONTRACT
            {
                return Err("Mixed whole4 source/roster/seat/parent/config differs".into());
            }
            for s in e.steps() {
                if s.actor() != e.learner_seat() {
                    return Err("Opponent leaked into mixed targets".into());
                }
                let rows = s.observation().legal_actions.len();
                for (n, k) in counts.iter_mut().zip([1, rows, usize::from(rows == 1)]) {
                    *n = n.checked_add(k).ok_or("Mixed target count overflow")?;
                }
            }
            observed = observed
                .checked_add(e.observed_callbacks())
                .ok_or("Mixed transition count overflow")?;
        }
        if counts[0] > 4 * crate::public_stochastic::MAX_SAMPLES
            || counts[1] > 4 * crate::public_stochastic::MAX_RESERVED_CANDIDATE_ROWS
        {
            return Err("Mixed learner cohort target cap".into());
        }
        let checksum = hash(
            b"tzolkin-completed-mixed-rl-cohort-v1\0",
            &(
                &self.checksum,
                episodes.each_ref().map(|e| e.canonical_record_checksum()),
                counts,
                observed,
            ),
        )?;
        Ok(ValidatedMixedRlCohort {
            plan: self,
            episodes,
            counts,
            observed,
            checksum,
        })
    }
}
pub struct ValidatedMixedRlCohort {
    plan: PlannedMixedRlCohort,
    episodes: [ValidatedMixedRlEpisode; 4],
    counts: [usize; 3],
    observed: usize,
    checksum: String,
}
impl ValidatedMixedRlCohort {
    pub fn plan(&self) -> &PlannedMixedRlCohort {
        &self.plan
    }
    pub fn episodes(&self) -> &[ValidatedMixedRlEpisode; 4] {
        &self.episodes
    }
    pub fn receipt_checksum(&self) -> &str {
        &self.checksum
    }
    pub fn total_learner_callbacks(&self) -> usize {
        self.counts[0]
    }
    pub fn total_learner_candidate_rows(&self) -> usize {
        self.counts[1]
    }
    pub fn total_learner_singletons(&self) -> usize {
        self.counts[2]
    }
    pub fn total_observed_callbacks(&self) -> usize {
        self.observed
    }
    /// A one-seat learner contributes own-SUM/4. Opponent actors are rejected,
    /// rather than acquiring the all-seat self-play coefficient 1/(4p).
    pub fn actor_coefficient(&self, member: usize, actor: usize) -> Result<f64, String> {
        if self
            .plan
            .headers
            .get(member)
            .is_none_or(|h| h.base_policy.learner_seat != actor)
        {
            return Err("Mixed coefficient requires its declared learner actor".into());
        }
        Ok(0.25)
    }
    pub fn receipt(&self) -> serde_json::Value {
        json!({"schema":"tzolkin-validated-mixed-rl-cohort-v1","cohortContract":COHORT_CONTRACT,
            "planChecksum":self.plan.checksum,"checksum":self.checksum,
            "episodeChecksums":self.episodes.each_ref().map(|e|e.canonical_record_checksum()),
            "learnerCounts":self.counts,"observedCallbacks":self.observed,"learnerSeats":self.plan.learner_seats(),
            "actorNormalization":"learnerOwnSumTimes1/4","criticTargets":"learnerCallbacksOnlyIncludingSingletons",
            "trainingDataAdmission":true,"selfPlayProducerAdmission":false,"mixedOptimizerAvailable":false})
    }
}
