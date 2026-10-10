//! Fixed four-game, all-seat RL rollout admission from a sealed immediate parent.
//! No collection, forwarding, update or chronology authentication occurs here.
//! Count1 is the current closed route; repeated count2..10 needs a new owner
//! contract rather than fabricated BC metadata or a type for every count.
//! Freshness here is Train plus inherited-lineage exclusion; full external
//! experiment history and chronological reservation remain caller responsibilities.
use crate::dataset::{DatasetSplit, seed_family_id, split_for_family};
use crate::public_policy_episode::TARGET_CONTRACT;
use crate::public_rl_native::UpdatedPublicRlHandle;
use crate::public_rl_policy_episode::{EPISODE_CONTRACT, ValidatedRlPolicyEpisode};
use crate::public_stochastic::{MAX_RESERVED_CANDIDATE_ROWS, MAX_SAMPLES, RlSamplingPolicy};
use crate::public_stochastic_native::{NativeStochasticConfig, numerical_target};
use crate::public_stochastic_record::Config;
use crate::replay::{RULES_BASELINE, RULES_VERSION, catalog_hash};
use serde::Serialize;
use sha2::{Digest, Sha256};

pub const COHORT_CONTRACT: &str = "four-train-games-all-seat-rl-count1-actor-sum-v1";
const GAMES: usize = 4;
const MAX_RECEIPT_BYTES: usize = 64 * 1024;

/// Content-bound full plan. Call before collecting all members; no raw or
/// Deserialize constructor. External reservation/time evidence is separate.
pub struct PlannedRlCohort {
    source: RlSamplingPolicy,
    configurations: [NativeStochasticConfig; GAMES],
    families: [String; GAMES],
    catalog: String,
    target: String,
    checksum: String,
}
impl PlannedRlCohort {
    pub fn new(
        parent: &UpdatedPublicRlHandle<'_>,
        configurations: [NativeStochasticConfig; GAMES],
    ) -> Result<Self, String> {
        plan_parts(
            RlSamplingPolicy::from_handle(parent)?,
            parent.family_closure(),
            configurations,
        )
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
    pub fn configurations(&self) -> &[NativeStochasticConfig; GAMES] {
        &self.configurations
    }
    pub fn rollout_families(&self) -> &[String; GAMES] {
        &self.families
    }
    pub fn numerical_target(&self) -> &str {
        &self.target
    }
    /// Any failed, unattempted or incomplete member rejects the whole cohort.
    /// This consumer cannot silently remove a member or replace it with a retry.
    pub fn finish(
        self,
        results: [Result<ValidatedRlPolicyEpisode, String>; GAMES],
    ) -> Result<ValidatedRlCohort, String> {
        let [a, b, c, d] = results;
        let episodes = [
            completed(0, a)?,
            completed(1, b)?,
            completed(2, c)?,
            completed(3, d)?,
        ];
        let [a, b, c, d] = episodes.each_ref().map(member);
        let members = [a?, b?, c?, d?];
        let counts = check_members(&self, &members)?;
        let identities = members.map(|m| (m.checksum, m.callbacks, m.rows, m.singletons));
        let checksum = hash(
            b"tzolkin-completed-rl-count1-cohort-v1\0",
            &(&self.checksum, identities, counts),
        )?;
        Ok(ValidatedRlCohort {
            plan: self,
            episodes,
            counts,
            checksum,
        })
    }
}

/// Owns all four complete RL episodes. This is not existing BC/native dataset
/// admission, and no mutable/filter/subset or arbitrary stored-record route exists.
pub struct ValidatedRlCohort {
    plan: PlannedRlCohort,
    episodes: [ValidatedRlPolicyEpisode; GAMES],
    counts: [usize; 3],
    checksum: String,
}
impl ValidatedRlCohort {
    pub fn plan(&self) -> &PlannedRlCohort {
        &self.plan
    }
    pub fn episodes(&self) -> &[ValidatedRlPolicyEpisode; GAMES] {
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
    /// Sum own decisions, average absolute actors, average four member families.
    /// 3p and 4p each have half the mass. Singletons keep the actor denominator.
    pub fn actor_coefficient(&self, member: usize, actor: usize) -> Result<f64, String> {
        let config = self
            .plan
            .configurations
            .get(member)
            .ok_or("Invalid RL cohort member")?;
        if actor >= config.players() {
            return Err("Invalid RL cohort absolute actor".into());
        }
        Ok(1.0 / (GAMES * config.players()) as f64)
    }
}
fn completed(
    index: usize,
    result: Result<ValidatedRlPolicyEpisode, String>,
) -> Result<ValidatedRlPolicyEpisode, String> {
    result.map_err(|error| {
        format!(
            "Whole RL cohort rejected: member {index} failed: {}",
            error.chars().take(1024).collect::<String>()
        )
    })
}
fn plan_parts(
    source: RlSamplingPolicy,
    inherited_closure: &[String],
    configurations: [NativeStochasticConfig; GAMES],
) -> Result<PlannedRlCohort, String> {
    if source.update_count() != 1
        || !digest_id(source.artifact_checksum())
        || configurations.iter().filter(|c| c.players() == 3).count() != 2
        || configurations.iter().filter(|c| c.players() == 4).count() != 2
    {
        return Err("RL cohort requires a sealed count1 parent and two 3p/two 4p games".into());
    }
    let families = std::array::from_fn(|i| seed_family_id(configurations[i].environment_seed()));
    for (index, config) in configurations.iter().enumerate() {
        if split_for_family(&families[index])? != DatasetSplit::Train
            || inherited_closure.contains(&families[index])
            || families[..index].contains(&families[index])
            || configurations[..index]
                .iter()
                .any(|old| old.sampling_identity() == config.sampling_identity())
        {
            return Err(
                "RL cohort requires fresh distinct Train families and sampling streams".into(),
            );
        }
    }
    let catalog = catalog_hash();
    let target = numerical_target();
    let encoded = configurations.each_ref().map(Config::from_config);
    // The parent checksum already seals its entire non-Test lineage; the plan
    // checks exclusion against that owner without copying/hashing the model.
    let checksum = hash(
        b"tzolkin-planned-rl-count1-cohort-v1\0",
        &(
            COHORT_CONTRACT,
            EPISODE_CONTRACT,
            TARGET_CONTRACT,
            RULES_VERSION,
            RULES_BASELINE,
            &catalog,
            &target,
            &source,
            encoded,
            &families,
        ),
    )?;
    Ok(PlannedRlCohort {
        source,
        configurations,
        families,
        catalog,
        target,
        checksum,
    })
}

#[derive(Clone, Copy)]
struct Member<'a> {
    config: &'a NativeStochasticConfig,
    family: &'a str,
    source: &'a RlSamplingPolicy,
    catalog: &'a str,
    target: &'a str,
    checksum: &'a str,
    callbacks: usize,
    rows: usize,
    singletons: usize,
}
fn member(episode: &ValidatedRlPolicyEpisode) -> Result<Member<'_>, String> {
    let mut rows = 0usize;
    let mut singletons = 0usize;
    for step in episode.steps() {
        let legal = step.observation().legal_actions.len();
        rows = rows
            .checked_add(legal)
            .ok_or("RL cohort candidate count overflow")?;
        singletons += usize::from(legal == 1);
    }
    Ok(Member {
        config: episode.source_config(),
        family: episode.family_id(),
        source: episode.parent_policy(),
        catalog: episode.catalog_hash(),
        target: episode.numerical_target(),
        checksum: episode.canonical_record_checksum(),
        callbacks: episode.len(),
        rows,
        singletons,
    })
}
fn check_members(
    plan: &PlannedRlCohort,
    members: &[Member<'_>; GAMES],
) -> Result<[usize; 3], String> {
    let mut counts = [0usize; 3];
    for (index, member) in members.iter().enumerate() {
        let limits = plan.configurations[index].limits();
        if member.config != &plan.configurations[index]
            || member.family != plan.families[index]
            || member.source != &plan.source
            || member.catalog != plan.catalog
            || member.target != plan.target
            || !digest_id(member.checksum)
            || member.callbacks == 0
            || member.callbacks > limits.max_callbacks()
            || member.rows < member.callbacks
            || member.rows > limits.max_candidate_rows()
            || member.singletons > member.callbacks
        {
            return Err(format!(
                "Whole RL cohort rejected: member {index} identity/count mismatch"
            ));
        }
        for (total, amount) in
            counts
                .iter_mut()
                .zip([member.callbacks, member.rows, member.singletons])
        {
            *total = total
                .checked_add(amount)
                .ok_or("RL cohort count overflow")?;
        }
    }
    if counts[0] > GAMES * MAX_SAMPLES || counts[1] > GAMES * MAX_RESERVED_CANDIDATE_ROWS {
        return Err("Whole RL cohort exceeds callback/candidate caps".into());
    }
    Ok(counts)
}
fn digest_id(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}
fn hash(domain: &[u8], value: &impl Serialize) -> Result<String, String> {
    let bytes = serde_json::to_vec(value).map_err(|e| e.to_string())?;
    if bytes.len() > MAX_RECEIPT_BYTES {
        return Err("RL cohort identity exceeds receipt byte bound".into());
    }
    let mut digest = Sha256::new();
    digest.update(domain);
    digest.update(bytes);
    Ok(format!("{:x}", digest.finalize()))
}
