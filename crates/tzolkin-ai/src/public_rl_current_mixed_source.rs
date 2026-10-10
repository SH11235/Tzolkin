//! Distinct current-mixed data codec. Only a genuine initial mixed handle is
//! executable here; continued metadata cannot construct an owner or producer.
use crate::dataset::{DatasetSplit, seed_family_id, split_for_family};
use crate::public_policy_episode::{
    ActorLink, AppliedActorStep, applied_mixed_plan, applied_step, canonical_record_checksum,
};
use crate::public_policy_update::{checkpoint_digest, hash};
use crate::public_rl_mixed_handle::{MixedInferenceIdentity, MixedPolicyHandle};
use crate::public_rl_mixed_native::{FrozenMixedRoster, FrozenOpponentIdentity};
use crate::public_stochastic::{RNG_VERSION, UNIFORM_MIXTURE};
use crate::public_stochastic_native::{
    NativeStochasticConfig, numerical_target, run_current_mixed_shared,
};
use crate::public_stochastic_record::{
    Config, Header, Record, decode_typed_record, encode_typed_record, hex64,
};
use crate::replay::{RULES_BASELINE, RULES_VERSION, catalog_hash};
use serde::{Deserialize, Serialize};
use serde_json::json;
use tzolkin_core::observation::{MOVE_SCHEMA, OBSERVATION_SCHEMA};

pub const RECORD_SCHEMA: &str = "tzolkin-public-stochastic-rl-current-mixed-native-record-v2";
pub const SOURCE_KIND: &str = "publicStochasticRlCurrentMixedV2";
pub const ROSTER_CONTRACT: &str = "frozen-current-mixed-absolute-reactive-opponents-one-learner-v2";
pub const EPISODE_CONTRACT: &str = "applied-current-mixed-learner-gamma1-terminal-winner-share-v2";
pub const COHORT_CONTRACT: &str = "whole4-frozen-current-mixed-learner-own-sum-quarter-v2";
pub const CONTINUED_IDENTITY_SCHEMA: &str =
    "tzolkin-public-rl-continued-mixed-inference-identity-v2";
pub const CONTINUED_POLICY_VERSION: &str = "learned-public-rl-continued-mixed-joint-argmax-v2";
pub const CONTINUED_SAMPLING_VERSION: &str =
    "public-stochastic-rl-continued-mixed-joint-uniform-tick53-v2";
pub const CONTINUED_TASK: &str = "currentMixedLearnerOwnSumQuarterJointV2";
pub const MAX_MIXED_UPDATES: u64 = 1_000_000;

/// Count-neutral, closed future-parent metadata. Validation is content-only;
/// this unit provides no continued owner, handle, sampler or artifact restore.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct ContinuedMixedIdentity {
    schema: String,
    policy_version: String,
    sampling_version: String,
    task: String,
    artifact_checksum: String,
    first_mixed_checksum: String,
    parent_artifact_checksum: String,
    imported_update_count: u64,
    mixed_update_count: u64,
    actor_model_version: String,
    critic_model_version: String,
    actor_parameters_checksum: String,
    residual_parameters_checksum: String,
    feature_schema: u32,
    input_contract: String,
    context_schema: u32,
    context_contract: String,
    backend: String,
    numerical_target: String,
}
impl ContinuedMixedIdentity {
    pub fn validate(&self) -> Result<(), String> {
        if self.schema != CONTINUED_IDENTITY_SCHEMA
            || self.policy_version != CONTINUED_POLICY_VERSION
            || self.sampling_version != CONTINUED_SAMPLING_VERSION
            || self.task != CONTINUED_TASK
            || !(1..=crate::public_policy_paired_long::MAX_UPDATES)
                .contains(&self.imported_update_count)
            || !(2..=MAX_MIXED_UPDATES).contains(&self.mixed_update_count)
            || self.actor_model_version != crate::public_model::MODEL_VERSION
            || self.critic_model_version != crate::public_state_critic::MODEL_VERSION
            || [
                &self.artifact_checksum,
                &self.first_mixed_checksum,
                &self.parent_artifact_checksum,
                &self.actor_parameters_checksum,
                &self.residual_parameters_checksum,
            ]
            .into_iter()
            .any(|s| !checkpoint_digest(s))
            || self.feature_schema != crate::features::PUBLIC_FEATURE_SCHEMA
            || self.input_contract != crate::public_model::INPUT_CONTRACT
            || self.context_schema != crate::public_state_critic::CONTEXT_SCHEMA
            || self.context_contract != crate::public_state_critic::CONTEXT_CONTRACT
            || self.backend != "scalar"
            || self.numerical_target != numerical_target()
        {
            return Err("Unsupported continued-mixed metadata/count/target".into());
        }
        Ok(())
    }
}

/// V1 bridge and V2 continuation retain their own identities. In particular,
/// the initial sampler binding hashes the original inner V1 identity verbatim.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(tag = "kind", rename_all = "camelCase", deny_unknown_fields)]
pub enum CurrentMixedParentIdentity {
    InitialMixedV1 { policy: MixedInferenceIdentity },
    ContinuedMixedV2 { policy: ContinuedMixedIdentity },
}
impl CurrentMixedParentIdentity {
    pub fn validate(&self) -> Result<(), String> {
        match self {
            Self::InitialMixedV1 { policy } => policy.validate(),
            Self::ContinuedMixedV2 { policy } => policy.validate(),
        }
    }
    pub fn imported_update_count(&self) -> u64 {
        match self {
            Self::InitialMixedV1 { policy } => policy.imported_update_count(),
            Self::ContinuedMixedV2 { policy } => policy.imported_update_count,
        }
    }
    pub fn mixed_update_count(&self) -> u64 {
        match self {
            Self::InitialMixedV1 { policy } => policy.mixed_update_count(),
            Self::ContinuedMixedV2 { policy } => policy.mixed_update_count,
        }
    }
    pub fn artifact_checksum(&self) -> &str {
        match self {
            Self::InitialMixedV1 { policy } => policy.artifact_checksum(),
            Self::ContinuedMixedV2 { policy } => &policy.artifact_checksum,
        }
    }
    pub(crate) fn require_initial(&self, actual: &MixedInferenceIdentity) -> Result<(), String> {
        self.validate()?;
        match self {
            Self::InitialMixedV1 { policy } if policy == actual => Ok(()),
            Self::InitialMixedV1 { .. } => {
                Err("Current mixed parent differs from genuine handle".into())
            }
            Self::ContinuedMixedV2 { .. } => {
                Err("Continued mixed producer is unavailable before work".into())
            }
        }
    }
    fn binding_key(&self) -> Result<String, String> {
        match self {
            Self::InitialMixedV1 { policy } => policy.binding_key(),
            Self::ContinuedMixedV2 { .. } => Err("Continued mixed sampler is unavailable".into()),
        }
    }
    fn sampling_version(&self) -> &'static str {
        match self {
            Self::InitialMixedV1 { .. } => crate::public_rl_mixed_handle::SAMPLING_VERSION,
            Self::ContinuedMixedV2 { .. } => CONTINUED_SAMPLING_VERSION,
        }
    }
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub(crate) struct CurrentMixedSource {
    pub contract: String,
    pub learner_seat: usize,
    pub parent: CurrentMixedParentIdentity,
    pub opponents: Vec<Option<FrozenOpponentIdentity>>,
    pub roster_checksum: String,
    pub max_opponent_action_rows: usize,
    pub max_opponent_search_steps: usize,
}
pub(crate) fn header(
    config: &NativeStochasticConfig,
    handle: &MixedPolicyHandle<'_>,
    roster: &FrozenMixedRoster<'_>,
    consumed: &std::collections::BTreeSet<String>,
) -> Result<Header<CurrentMixedSource>, String> {
    let family = seed_family_id(config.environment_seed());
    if config.players() != roster.players()
        || split_for_family(&family)? != DatasetSplit::Train
        || consumed.contains(&family)
        || handle.family_closure().any(|f| f == &family)
        || roster.contains_family(&family)
    {
        return Err("Current mixed Train family overlaps closure or roster size".into());
    }
    Ok(Header {
        source_kind: SOURCE_KIND.into(),
        rules_version: RULES_VERSION,
        rules_baseline: RULES_BASELINE.into(),
        catalog_hash: catalog_hash(),
        observation_schema: OBSERVATION_SCHEMA,
        move_schema: MOVE_SCHEMA,
        config: Config::from_config(config),
        base_policy: CurrentMixedSource {
            contract: ROSTER_CONTRACT.into(),
            learner_seat: roster.learner_seat(),
            parent: CurrentMixedParentIdentity::InitialMixedV1 {
                policy: handle.identity().clone(),
            },
            opponents: roster.descriptions().to_vec(),
            roster_checksum: roster.checksum().into(),
            max_opponent_action_rows: config
                .limits()
                .max_callbacks()
                .checked_mul(crate::features::MAX_LEGAL_ACTIONS)
                .ok_or("Current mixed opponent row overflow")?,
            max_opponent_search_steps: roster
                .max_opponent_search_steps(config.limits().max_callbacks())?,
        },
        sampling_version: crate::public_rl_mixed_handle::SAMPLING_VERSION.into(),
        rng_version: RNG_VERSION.into(),
        denominator_bits: 53,
        uniform_mixture_bits: hex64(UNIFORM_MIXTURE.to_bits()),
        backend: "scalar".into(),
        numerical_target: numerical_target(),
        native_family_id: family,
        raw_artifact_identity_available: false,
        producer_authenticated: false,
        independent_seed_origins_verified: false,
    })
}
pub(crate) fn collect(
    config: &NativeStochasticConfig,
    handle: &MixedPolicyHandle<'_>,
    roster: &FrozenMixedRoster<'_>,
    header: Header<CurrentMixedSource>,
) -> Result<Record<CurrentMixedSource>, String> {
    run_current_mixed_shared(config, handle, roster, header, None)
}
pub(crate) fn encode(record: &Record<CurrentMixedSource>) -> Result<Vec<u8>, String> {
    encode_typed_record(record)
}
pub(crate) fn check_record(
    raw: &[u8],
    header: &Header<CurrentMixedSource>,
) -> Result<Record<CurrentMixedSource>, String> {
    let record: Record<CurrentMixedSource> = decode_typed_record(raw)?;
    if record.schema != RECORD_SCHEMA
        || record.header.source_kind != SOURCE_KIND
        || record.training_admission != "unavailable-distinct-codec-required"
        || record.header != *header
    {
        return Err("Current mixed record/schema/full planned header differs before work".into());
    }
    let config = record.header.config.checked_config()?;
    let seat = header.base_policy.learner_seat;
    if record.callbacks.len() > config.limits().max_callbacks()
        || seat >= config.players()
        || header.base_policy.opponents.len() != config.players()
        || header
            .base_policy
            .opponents
            .iter()
            .enumerate()
            .any(|(i, p)| p.is_none() != (i == seat))
        || record.callbacks.iter().any(|c| {
            c.actor >= config.players()
                || (c.actor == seat && (c.opponent_attempted || c.opponent_choice.is_some()))
                || (c.actor != seat && (c.sampler_attempted || c.sample.is_some()))
        })
    {
        return Err("Current mixed role/absolute-seat partition differs before work".into());
    }
    Ok(record)
}
pub(crate) fn audit(
    raw: &[u8],
    handle: &MixedPolicyHandle<'_>,
    roster: &FrozenMixedRoster<'_>,
    header: Header<CurrentMixedSource>,
) -> Result<ValidatedCurrentMixedEpisode, String> {
    header
        .base_policy
        .parent
        .require_initial(handle.identity())?;
    let expected = check_record(raw, &header)?;
    let config = header.config.checked_config()?;
    let actual = run_current_mixed_shared(&config, handle, roster, header, Some(&expected))?;
    ValidatedCurrentMixedEpisode::from_audited(actual, config)
}

/// Only a successful fresh native replay constructs terminal learner targets.
pub struct ValidatedCurrentMixedEpisode {
    record: Record<CurrentMixedSource>,
    config: NativeStochasticConfig,
    indices: Vec<usize>,
    links: Vec<ActorLink>,
    rank: usize,
    winners: usize,
    checksum: String,
}
impl ValidatedCurrentMixedEpisode {
    fn from_audited(
        record: Record<CurrentMixedSource>,
        config: NativeStochasticConfig,
    ) -> Result<Self, String> {
        if record.terminal.is_none()
            || record.failure.is_some()
            || record.header.base_policy.contract != ROSTER_CONTRACT
        {
            return Err("Incomplete current mixed episode cannot be sealed".into());
        }
        let parent = &record.header.base_policy.parent;
        let learner = record.header.base_policy.learner_seat;
        let (links, ranks, winners) = applied_mixed_plan(
            &record,
            &config,
            parent.sampling_version(),
            || parent.binding_key(),
            learner,
        )?;
        let indices = record
            .callbacks
            .iter()
            .enumerate()
            .filter(|(_, c)| c.actor == learner)
            .map(|(i, _)| i)
            .collect::<Vec<_>>();
        if links.len() != indices.len() {
            return Err("Current mixed own-link coverage differs".into());
        }
        let rank = ranks[learner];
        let checksum =
            canonical_record_checksum(&record, b"tzolkin-applied-current-mixed-episode-v2\0")?;
        Ok(Self {
            record,
            config,
            indices,
            links,
            rank,
            winners,
            checksum,
        })
    }
    pub fn learner_seat(&self) -> usize {
        self.record.header.base_policy.learner_seat
    }
    pub fn canonical_record_checksum(&self) -> &str {
        &self.checksum
    }
    pub fn source_config(&self) -> &NativeStochasticConfig {
        &self.config
    }
    pub fn parent_identity(&self) -> &CurrentMixedParentIdentity {
        &self.record.header.base_policy.parent
    }
    pub fn steps(&self) -> impl ExactSizeIterator<Item = AppliedActorStep<'_>> {
        self.indices.iter().zip(&self.links).map(|(&index, link)| {
            applied_step(&self.record.callbacks[index], link, self.rank, self.winners)
        })
    }
}
/// An actual handle, legal Train partition and all four configurations create
/// this plan. Serialized receipts never construct it or a validated cohort.
pub struct PlannedCurrentMixedCohort {
    configs: [NativeStochasticConfig; 4],
    pub(crate) headers: [Header<CurrentMixedSource>; 4],
    families: [String; 4],
    checksum: String,
}
impl PlannedCurrentMixedCohort {
    pub(crate) fn new(
        configs: [NativeStochasticConfig; 4],
        headers: [Header<CurrentMixedSource>; 4],
    ) -> Result<Self, String> {
        if configs.iter().filter(|c| c.players() == 3).count() != 2
            || configs.iter().filter(|c| c.players() == 4).count() != 2
        {
            return Err("Current mixed whole-four requires two 3p and two 4p".into());
        }
        let families = configs
            .each_ref()
            .map(|c| seed_family_id(c.environment_seed()));
        for i in 0..4 {
            if families[..i].contains(&families[i])
                || configs[..i]
                    .iter()
                    .any(|c| c.sampling_identity() == configs[i].sampling_identity())
                || headers[i].base_policy.parent != headers[0].base_policy.parent
                || headers[i].config != Config::from_config(&configs[i])
                || headers[i].native_family_id != families[i]
            {
                return Err("Current mixed whole-four family/RNG/parent/config differs".into());
            }
        }
        let checksum = hash(
            b"tzolkin-planned-current-mixed-cohort-v2\0",
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
    pub fn rollout_families(&self) -> &[String; 4] {
        &self.families
    }
    pub fn configurations(&self) -> &[NativeStochasticConfig; 4] {
        &self.configs
    }
    pub fn parent_identity(&self) -> &CurrentMixedParentIdentity {
        &self.headers[0].base_policy.parent
    }
    pub fn receipt(&self) -> serde_json::Value {
        json!({"schema":"tzolkin-planned-current-mixed-cohort-v2","cohortContract":COHORT_CONTRACT,
            "episodeContract":EPISODE_CONTRACT,"rosterContract":ROSTER_CONTRACT,"checksum":self.checksum,
            "headers":self.headers,"families":self.families,"actorNormalization":"learnerOwnSumTimes1/4",
            "criticTargets":"learnerCallbacksOnlyIncludingSingletons","producerAuthenticated":false,"independentSeedOriginsVerified":false})
    }
    pub(crate) fn finish(
        self,
        results: [Result<ValidatedCurrentMixedEpisode, String>; 4],
    ) -> Result<ValidatedCurrentMixedCohort, String> {
        let [a, b, c, d] = results;
        let episodes = [a?, b?, c?, d?];
        let mut counts = [0usize; 3];
        let mut observed = 0usize;
        for (i, e) in episodes.iter().enumerate() {
            if e.record.header != self.headers[i]
                || e.config != self.configs[i]
                || e.indices.is_empty()
            {
                return Err("Current mixed whole-four audited member differs".into());
            }
            for s in e.steps() {
                if s.actor() != e.learner_seat() {
                    return Err("Current mixed opponent leaked into targets".into());
                }
                let n = s.observation().legal_actions.len();
                for (sum, k) in counts.iter_mut().zip([1, n, usize::from(n == 1)]) {
                    *sum = sum
                        .checked_add(k)
                        .ok_or("Current mixed target count overflow")?;
                }
            }
            observed = observed
                .checked_add(e.record.counts.observed_callbacks)
                .ok_or("Current mixed callback count overflow")?;
        }
        if counts[0] > 4 * crate::public_stochastic::MAX_SAMPLES
            || counts[1] > 4 * crate::public_stochastic::MAX_RESERVED_CANDIDATE_ROWS
        {
            return Err("Current mixed cohort target cap".into());
        }
        let checksum = hash(
            b"tzolkin-completed-current-mixed-cohort-v2\0",
            &(
                &self.checksum,
                episodes.each_ref().map(|e| e.canonical_record_checksum()),
                counts,
                observed,
            ),
        )?;
        Ok(ValidatedCurrentMixedCohort {
            plan: self,
            episodes,
            counts,
            observed,
            checksum,
        })
    }
}
pub struct ValidatedCurrentMixedCohort {
    plan: PlannedCurrentMixedCohort,
    episodes: [ValidatedCurrentMixedEpisode; 4],
    counts: [usize; 3],
    observed: usize,
    checksum: String,
}
impl ValidatedCurrentMixedCohort {
    pub fn plan(&self) -> &PlannedCurrentMixedCohort {
        &self.plan
    }
    pub fn episodes(&self) -> &[ValidatedCurrentMixedEpisode; 4] {
        &self.episodes
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
    pub fn receipt_checksum(&self) -> &str {
        &self.checksum
    }
    pub fn actor_coefficient(&self, member: usize, actor: usize) -> Result<f64, String> {
        if self
            .plan
            .headers
            .get(member)
            .is_none_or(|h| h.base_policy.learner_seat != actor)
        {
            return Err("Current mixed coefficient requires its absolute learner actor".into());
        }
        Ok(0.25)
    }
    pub fn receipt(&self) -> serde_json::Value {
        json!({"schema":"tzolkin-validated-current-mixed-cohort-v2","cohortContract":COHORT_CONTRACT,
            "planChecksum":self.plan.checksum,"checksum":self.checksum,"episodeChecksums":self.episodes.each_ref().map(|e|e.canonical_record_checksum()),
            "learnerCounts":self.counts,"observedCallbacks":self.observed,"actorNormalization":"learnerOwnSumTimes1/4",
            "criticTargets":"learnerCallbacksOnlyIncludingSingletons","trainingDataAdmission":true,"selfPlayProducerAdmission":false,"mixedOptimizerAvailable":false})
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    pub(crate) fn initial() -> MixedInferenceIdentity {
        serde_json::from_value(json!({"schema":crate::public_rl_mixed_handle::IDENTITY_SCHEMA,
            "policyVersion":crate::public_rl_mixed_handle::POLICY_VERSION,
            "samplingVersion":crate::public_rl_mixed_handle::SAMPLING_VERSION,
            "task":crate::public_policy_mixed::TASK,"artifactChecksum":"a".repeat(64),
            "importedUpdateCount":1,"mixedUpdateCount":1,
            "actorModelVersion":crate::public_model::MODEL_VERSION,"criticModelVersion":crate::public_state_critic::MODEL_VERSION,
            "actorParametersChecksum":"b".repeat(64),"residualParametersChecksum":"c".repeat(64),
            "featureSchema":crate::features::PUBLIC_FEATURE_SCHEMA,"inputContract":crate::public_model::INPUT_CONTRACT,
            "contextSchema":crate::public_state_critic::CONTEXT_SCHEMA,"contextContract":crate::public_state_critic::CONTEXT_CONTRACT,
            "backend":"scalar","numericalTarget":numerical_target()})).unwrap()
    }
    pub(crate) fn plan() -> PlannedCurrentMixedCohort {
        use crate::public_stochastic::{SamplingSeed, SamplingStreamIdentity};
        use crate::public_stochastic_native::CollectionLimits;
        let seeds = (0..100)
            .filter(|&s| split_for_family(&seed_family_id(s)).unwrap() == DatasetSplit::Train)
            .take(4)
            .collect::<Vec<_>>();
        let configs = std::array::from_fn(|i| {
            NativeStochasticConfig::new(
                if i < 2 { 3 } else { 4 },
                seeds[i],
                SamplingStreamIdentity::new(
                    SamplingSeed::new(7),
                    9 + i as u64,
                    11,
                    if i < 2 { 3 } else { 4 },
                )
                .unwrap(),
                CollectionLimits::default(),
            )
            .unwrap()
        });
        let headers = std::array::from_fn(|i| Header {
            source_kind: SOURCE_KIND.into(),
            rules_version: RULES_VERSION,
            rules_baseline: RULES_BASELINE.into(),
            catalog_hash: catalog_hash(),
            observation_schema: OBSERVATION_SCHEMA,
            move_schema: MOVE_SCHEMA,
            config: Config::from_config(&configs[i]),
            base_policy: CurrentMixedSource {
                contract: ROSTER_CONTRACT.into(),
                learner_seat: 1,
                parent: CurrentMixedParentIdentity::InitialMixedV1 { policy: initial() },
                opponents: (0..configs[i].players())
                    .map(|actor| {
                        if actor == 1 {
                            None
                        } else {
                            Some(FrozenOpponentIdentity::test_heuristic())
                        }
                    })
                    .collect(),
                roster_checksum: "d".repeat(64),
                max_opponent_action_rows: 4000 * crate::features::MAX_LEGAL_ACTIONS,
                max_opponent_search_steps: 0,
            },
            sampling_version: crate::public_rl_mixed_handle::SAMPLING_VERSION.into(),
            rng_version: RNG_VERSION.into(),
            denominator_bits: 53,
            uniform_mixture_bits: hex64(UNIFORM_MIXTURE.to_bits()),
            backend: "scalar".into(),
            numerical_target: numerical_target(),
            native_family_id: seed_family_id(seeds[i]),
            raw_artifact_identity_available: false,
            producer_authenticated: false,
            independent_seed_origins_verified: false,
        });
        PlannedCurrentMixedCohort::new(configs, headers).unwrap()
    }
    #[test]
    fn current_mixed_codec_keeps_initial_binding_and_refuses_continued_producer() {
        let old = initial();
        let old_raw = serde_json::to_vec(&old).unwrap();
        let tagged = CurrentMixedParentIdentity::InitialMixedV1 {
            policy: old.clone(),
        };
        assert_eq!(tagged.binding_key().unwrap(), old.binding_key().unwrap());
        assert_eq!(
            serde_json::to_vec(
                &serde_json::from_slice::<MixedInferenceIdentity>(&old_raw).unwrap()
            )
            .unwrap(),
            old_raw
        );
        assert!(
            serde_json::from_value::<crate::public_rl_mixed_native::MixedLearnerIdentity>(
                serde_json::to_value(&tagged).unwrap()
            )
            .is_err()
        );
        let mut wire = serde_json::to_value(&old).unwrap();
        for (key, value) in [
            ("schema", CONTINUED_IDENTITY_SCHEMA),
            ("policyVersion", CONTINUED_POLICY_VERSION),
            ("samplingVersion", CONTINUED_SAMPLING_VERSION),
            ("task", CONTINUED_TASK),
        ] {
            wire[key] = value.into();
        }
        wire["firstMixedChecksum"] = "d".repeat(64).into();
        wire["parentArtifactChecksum"] = "e".repeat(64).into();
        for count in [2, 3, MAX_MIXED_UPDATES] {
            wire["mixedUpdateCount"] = count.into();
            let parent = CurrentMixedParentIdentity::ContinuedMixedV2 {
                policy: serde_json::from_value(wire.clone()).unwrap(),
            };
            parent.validate().unwrap();
            assert!(parent.require_initial(&old).is_err());
            assert!(parent.binding_key().is_err());
            assert_eq!(parent.imported_update_count(), 1);
            assert_eq!(parent.mixed_update_count(), count);
        }
        for count in [0, 1, MAX_MIXED_UPDATES + 1] {
            wire["mixedUpdateCount"] = count.into();
            assert!(
                serde_json::from_value::<ContinuedMixedIdentity>(wire.clone())
                    .unwrap()
                    .validate()
                    .is_err()
            );
        }
        wire["mixedUpdateCount"] = 2.into();
        assert!(serde_json::from_value::<MixedInferenceIdentity>(wire).is_err());
        let mut unknown = serde_json::to_value(tagged).unwrap();
        unknown["accepted"] = true.into();
        assert!(serde_json::from_value::<CurrentMixedParentIdentity>(unknown).is_err());
    }
    #[test]
    fn current_mixed_record_namespace_and_absolute_role_guard_precede_work() {
        let plan = plan();
        let header = plan.headers[0].clone();
        let record = Record {
            schema: RECORD_SCHEMA.into(),
            header: header.clone(),
            callbacks: Vec::new(),
            terminal: None,
            failure: None,
            counts: crate::public_stochastic_record::Counts {
                observed_callbacks: 0,
                sampler_attempts: 0,
                accepted_samples: 0,
                candidate_rows_reserved: 0,
                actor_draws: vec![hex64(0); 4],
                apply_attempts: 0,
                apply_successes: 0,
                validated_applied_steps: 0,
                session_stopped: false,
            },
            source_payload_reserved_bytes: 0,
            trace_payload_reserved_bytes: 0,
            training_admission: "unavailable-distinct-codec-required".into(),
        };
        let raw = encode(&record).unwrap();
        assert!(check_record(&raw, &header).is_ok());
        assert!(decode_typed_record::<crate::public_rl_mixed_native::MixedRlSource>(&raw).is_err());
        let mut wrong = record.clone();
        wrong.schema = crate::public_rl_mixed_native::RECORD_SCHEMA.into();
        assert!(check_record(&encode(&wrong).unwrap(), &header).is_err());
        let mut wrong_header = header.clone();
        wrong_header.base_policy.learner_seat = 0;
        assert!(check_record(&raw, &wrong_header).is_err());
        let mut wrong = record;
        wrong.header.base_policy.opponents[0] = None;
        let bad_header = wrong.header.clone();
        assert!(check_record(&encode(&wrong).unwrap(), &bad_header).is_err());
    }
}
