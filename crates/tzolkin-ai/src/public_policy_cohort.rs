//! A fixed four-game, all-seat BC rollout cohort for one future update.
//! Content binding is not proof of registration time, producer authenticity or
//! independent seed origins. No collection, forwarding or update occurs here.
use serde::Serialize;
use sha2::{Digest, Sha256};

use crate::dataset::{DatasetSplit, seed_family_id, split_for_family};
use crate::public_policy_episode::{TARGET_CONTRACT, ValidatedPolicyEpisode};
use crate::public_rl_artifact::InitializedPublicRlPolicy;
use crate::public_stochastic::{MAX_RESERVED_CANDIDATE_ROWS, MAX_SAMPLES};
use crate::public_stochastic_native::{NativeStochasticConfig, numerical_target};
use crate::public_stochastic_record::Config;
use crate::replay::{RULES_BASELINE, RULES_VERSION, SeatPolicy, catalog_hash};

pub const COHORT_CONTRACT: &str = "four-train-games-all-seat-bc-actor-sum-v1";
const GAMES: usize = 4;
const MAX_RECEIPT_BYTES: usize = 64 * 1024;

/// Construct before collecting every listed member. There is no raw/Deserialize
/// constructor; external chronological registration evidence remains separate.
pub struct PlannedBcCohort {
    init_checksum: String,
    source: SeatPolicy,
    configurations: [NativeStochasticConfig; GAMES],
    families: [String; GAMES],
    catalog: String,
    target: String,
    checksum: String,
}
impl PlannedBcCohort {
    pub fn new(
        initial: &InitializedPublicRlPolicy,
        configurations: [NativeStochasticConfig; GAMES],
    ) -> Result<Self, String> {
        initial.artifact().validate()?;
        plan_parts(
            initial.artifact().checksum(),
            initial.artifact().bc_source().clone(),
            configurations,
        )
    }
    pub fn checksum(&self) -> &str {
        &self.checksum
    }
    pub fn init_checksum(&self) -> &str {
        &self.init_checksum
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
    /// Missing or unattempted members must be supplied as Err; any Err rejects
    /// the whole cohort. A failed audit/prefix cannot become an actor episode.
    pub fn finish(
        self,
        results: [Result<ValidatedPolicyEpisode, String>; GAMES],
    ) -> Result<ValidatedBcCohort, String> {
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
            b"tzolkin-completed-bc-cohort-v1\0",
            &(&self.checksum, identities, counts),
        )?;
        Ok(ValidatedBcCohort {
            plan: self,
            episodes,
            counts,
            checksum,
        })
    }
}

/// Owns all four complete episodes; no mutable, filtered or partial input route.
pub struct ValidatedBcCohort {
    plan: PlannedBcCohort,
    episodes: [ValidatedPolicyEpisode; GAMES],
    counts: [usize; 3],
    checksum: String,
}
impl ValidatedBcCohort {
    pub fn plan(&self) -> &PlannedBcCohort {
        &self.plan
    }
    pub fn episodes(&self) -> &[ValidatedPolicyEpisode; GAMES] {
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
    /// The 3p and 4p games each have half the mass, then families and actors are equal.
    /// Sum every actor decision; singleton gradients are zero, not removed rows.
    pub fn actor_coefficient(&self, member: usize, actor: usize) -> Result<f64, String> {
        let config = self
            .plan
            .configurations
            .get(member)
            .ok_or("Invalid cohort member")?;
        coefficient(config, actor)
    }
}
fn coefficient(config: &NativeStochasticConfig, actor: usize) -> Result<f64, String> {
    if actor >= config.players() {
        return Err("Invalid cohort absolute actor".into());
    }
    Ok(1.0 / (4 * config.players()) as f64)
}
fn completed(
    index: usize,
    result: Result<ValidatedPolicyEpisode, String>,
) -> Result<ValidatedPolicyEpisode, String> {
    result.map_err(|error| {
        format!(
            "Whole cohort rejected: member {index} failed: {}",
            error.chars().take(1024).collect::<String>()
        )
    })
}
fn plan_parts(
    init_checksum: &str,
    source: SeatPolicy,
    configurations: [NativeStochasticConfig; GAMES],
) -> Result<PlannedBcCohort, String> {
    source.validate()?;
    if !digest_id(init_checksum)
        || !matches!(&source, SeatPolicy::PublicLearned { inference_backend, .. } if inference_backend == "scalar")
    {
        return Err("Cohort requires an initialized Scalar BC identity".into());
    }
    if configurations.iter().filter(|c| c.players() == 3).count() != 2
        || configurations.iter().filter(|c| c.players() == 4).count() != 2
    {
        return Err("Cohort requires two 3p and two 4p games".into());
    }
    let families = std::array::from_fn(|i| seed_family_id(configurations[i].environment_seed()));
    for (index, config) in configurations.iter().enumerate() {
        if split_for_family(&families[index])? != DatasetSplit::Train
            || families[..index].contains(&families[index])
            || configurations[..index]
                .iter()
                .any(|old| old.sampling_identity() == config.sampling_identity())
        {
            return Err("Cohort requires distinct Train families and sampling streams".into());
        }
    }
    let catalog = catalog_hash();
    let target = numerical_target();
    let encoded = configurations.each_ref().map(Config::from_config);
    let checksum = hash(
        b"tzolkin-planned-bc-cohort-v1\0",
        &(
            COHORT_CONTRACT,
            TARGET_CONTRACT,
            RULES_VERSION,
            RULES_BASELINE,
            &catalog,
            &target,
            init_checksum,
            &source,
            encoded,
            &families,
        ),
    )?;
    Ok(PlannedBcCohort {
        init_checksum: init_checksum.into(),
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
    source: &'a SeatPolicy,
    catalog: &'a str,
    target: &'a str,
    checksum: &'a str,
    callbacks: usize,
    rows: usize,
    singletons: usize,
}
fn member(episode: &ValidatedPolicyEpisode) -> Result<Member<'_>, String> {
    let mut rows = 0usize;
    let mut singletons = 0;
    for step in episode.steps() {
        let legal = step.observation().legal_actions.len();
        rows = rows
            .checked_add(legal)
            .ok_or("Cohort candidate count overflow")?;
        singletons += usize::from(legal == 1);
    }
    Ok(Member {
        config: episode.source_config(),
        family: episode.family_id(),
        source: episode.policy_provenance(),
        catalog: episode.catalog_hash(),
        target: episode.numerical_target(),
        checksum: episode.canonical_record_checksum(),
        callbacks: episode.len(),
        rows,
        singletons,
    })
}
fn check_members(
    plan: &PlannedBcCohort,
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
                "Whole cohort rejected: member {index} identity/count mismatch"
            ));
        }
        for (total, amount) in
            counts
                .iter_mut()
                .zip([member.callbacks, member.rows, member.singletons])
        {
            *total = total.checked_add(amount).ok_or("Cohort count overflow")?;
        }
    }
    if counts[0] > GAMES * MAX_SAMPLES || counts[1] > GAMES * MAX_RESERVED_CANDIDATE_ROWS {
        return Err("Whole cohort exceeds callback/candidate caps".into());
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
    let bytes = serde_json::to_vec(value).map_err(|error| error.to_string())?;
    if bytes.len() > MAX_RECEIPT_BYTES {
        return Err("Cohort identity exceeds receipt byte bound".into());
    }
    let mut digest = Sha256::new();
    digest.update(domain);
    digest.update(bytes);
    Ok(format!("{:x}", digest.finalize()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::public_native::integration_fixture;
    use crate::public_stochastic::{SamplingSeed, SamplingStreamIdentity};
    use crate::public_stochastic_native::CollectionLimits;
    fn parts() -> (SeatPolicy, [NativeStochasticConfig; GAMES]) {
        let model = integration_fixture::model(false);
        let source = integration_fixture::handle(&model).provenance().clone();
        let seeds = (0..1000)
            .filter(|seed| split_for_family(&seed_family_id(*seed)).unwrap() == DatasetSplit::Train)
            .take(4)
            .collect::<Vec<_>>();
        let configs = std::array::from_fn(|i| {
            let players = if i < 2 { 3 } else { 4 };
            NativeStochasticConfig::new(
                players,
                seeds[i],
                SamplingStreamIdentity::new(SamplingSeed::new(17), i as u64, 0, players).unwrap(),
                CollectionLimits::default(),
            )
            .unwrap()
        });
        (source, configs)
    }
    fn summaries(plan: &PlannedBcCohort) -> [Member<'_>; GAMES] {
        std::array::from_fn(|i| Member {
            config: &plan.configurations[i],
            family: &plan.families[i],
            source: &plan.source,
            catalog: &plan.catalog,
            target: &plan.target,
            checksum: "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
            callbacks: 5,
            rows: 9,
            singletons: 2,
        })
    }
    #[test]
    fn full_plan_and_summary_preserve_order_counts_and_fixed_weights() {
        let (source, configs) = parts();
        let plan = plan_parts(&"a".repeat(64), source.clone(), configs.clone()).unwrap();
        assert_eq!(
            plan.checksum(),
            plan_parts(&"a".repeat(64), source, configs)
                .unwrap()
                .checksum()
        );
        assert_eq!(
            check_members(&plan, &summaries(&plan)).unwrap(),
            [20, 36, 8]
        );
        assert_eq!(coefficient(&plan.configurations[0], 0).unwrap(), 1.0 / 12.0);
        assert_eq!(coefficient(&plan.configurations[2], 3).unwrap(), 1.0 / 16.0);
        assert!(coefficient(&plan.configurations[0], 3).is_err());
        let rejected = plan.finish(std::array::from_fn(|_| Err("unattempted".into())));
        assert!(rejected.is_err());
    }
    #[test]
    fn whole_cohort_rejects_nontrain_duplicates_and_any_member_mismatch() {
        let (source, configs) = parts();
        let mut duplicate = configs.clone();
        duplicate[1] = duplicate[0].clone();
        assert!(plan_parts(&"a".repeat(64), source.clone(), duplicate).is_err());
        let mut stream = configs.clone();
        stream[1] = NativeStochasticConfig::new(
            3,
            stream[1].environment_seed(),
            *stream[0].sampling_identity(),
            CollectionLimits::default(),
        )
        .unwrap();
        assert!(plan_parts(&"a".repeat(64), source.clone(), stream).is_err());
        for (seed, split) in [(3, DatasetSplit::Validation), (10, DatasetSplit::Test)] {
            assert_eq!(split_for_family(&seed_family_id(seed)).unwrap(), split);
            let mut bad = configs.clone();
            bad[0] = NativeStochasticConfig::new(
                3,
                seed,
                *bad[0].sampling_identity(),
                CollectionLimits::default(),
            )
            .unwrap();
            assert!(plan_parts(&"a".repeat(64), source.clone(), bad).is_err());
        }
        let plan = plan_parts(&"a".repeat(64), source, configs).unwrap();
        let valid = summaries(&plan);
        for index in 0..GAMES {
            let mut bad = valid;
            bad[index].callbacks = 0;
            assert!(check_members(&plan, &bad).is_err());
            let mut bad = valid;
            bad[index].config = &plan.configurations[(index + 1) % GAMES];
            assert!(check_members(&plan, &bad).is_err());
        }
        let mut bad = valid;
        bad[0].target = "other-target";
        assert!(check_members(&plan, &bad).is_err());
        let mut bad = valid;
        bad[0].catalog = "old-catalog";
        assert!(check_members(&plan, &bad).is_err());
        let mut bad = valid;
        bad[0].rows = MAX_RESERVED_CANDIDATE_ROWS + 1;
        assert!(check_members(&plan, &bad).is_err());
        let mut other = plan.source.clone();
        if let SeatPolicy::PublicLearned { model_checksum, .. } = &mut other {
            *model_checksum = "c".repeat(64);
        }
        let mut bad = valid;
        bad[0].source = &other;
        assert!(check_members(&plan, &bad).is_err());
    }
}
