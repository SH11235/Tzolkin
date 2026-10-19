//! Explicit, unauthenticated research actors using the sealed Playing updater.
//! This entrance grants no teacher qualification, adoption or optimizer history.
use std::collections::{BTreeMap, BTreeSet};
use std::io::Read;
use std::sync::Arc;

use crate::conditional_human_rl_update::{ConditionalFrozenPool, ConditionalNativeExclusions};
use crate::conditional_semantic_playing_rl::{SemanticFreshOwner, SemanticSourceCheckPass};
use crate::dataset::DatasetSplit;
use crate::human_initial_wealth::ContentPin;
use crate::public_model::PublicPolicyModel;
use crate::public_native::PreparedPublicPolicy;
use crate::public_policy_baseline::ResidualFitConfig;
use crate::public_policy_update::{OneStepConfig, hash};
use crate::public_stochastic::SamplingSeed;
use crate::public_stochastic_native::CollectionLimits;
use crate::semantic_playing_rl_collection::{
    DeclaredSamplingHistory, SemanticCollectionComposition,
};
use crate::semantic_playing_rl_update::{
    SemanticCheckpointState, SemanticPlayingJointReport, SemanticPlayingJointSession,
};
use serde::{Deserialize, Serialize};

pub const POLICY_VERSION: &str = "research-actor-semantic-playing-fresh-scalar-v1";
pub const INPUT_CONTRACT: &str = "research-actor-fresh-start-base-3-4p-schema4-v1";
const MAX_BYTES: usize = 16 * 1024 * 1024;

/// The caller declares the artifact format and schema; neither is inferred from
/// a filename or treated as evidence of human/native provenance.
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct ResearchActorInput {
    pub schema: String,
    pub feature_schema: u32,
    pub actor: ContentPin,
    pub artifact_schema: String,
    pub critic_seed: u64,
}

pub(crate) struct ResearchOrigin {
    input: ResearchActorInput,
    required_external: ConditionalNativeExclusions,
    pub(crate) description: serde_json::Value,
    pub(crate) checksum: String,
    pub(crate) tables: BTreeMap<String, DatasetSplit>,
}
impl ResearchOrigin {
    pub(crate) fn recheck(&self) -> Result<(), String> {
        crate::human_policy_dataset::recheck_absolute_pin(&self.input.actor)?;
        self.required_external.recheck()
    }
    pub(crate) fn recheck_in(
        &self,
        checked: &mut SemanticSourceCheckPass<'_>,
    ) -> Result<(), String> {
        checked.pin(&self.input.actor)?;
        checked.external(&self.required_external)
    }
    pub(crate) fn check_external(
        &self,
        external: &ConditionalNativeExclusions,
    ) -> Result<(), String> {
        if !self.required_external.is_subset_of(external) {
            return Err("Research exclusions omit a previously declared family".into());
        }
        Ok(())
    }
}

pub struct PreparedResearchActor {
    origin: Arc<ResearchOrigin>,
    actor: PublicPolicyModel,
}
impl PreparedResearchActor {
    pub fn load(
        input: ResearchActorInput,
        external: &ConditionalNativeExclusions,
    ) -> Result<Self, String> {
        if input.schema != "researchActorFreshStartV1"
            || input.feature_schema != 4
            || input.artifact_schema.is_empty()
        {
            return Err("Research fresh-start input schema differs".into());
        }
        crate::human_policy_dataset::recheck_absolute_pin(&input.actor)?;
        external.recheck()?;
        let mut bytes = Vec::new();
        std::fs::File::open(&input.actor.path)
            .map_err(|e| e.to_string())?
            .take((MAX_BYTES + 1) as u64)
            .read_to_end(&mut bytes)
            .map_err(|e| e.to_string())?;
        if bytes.len() > MAX_BYTES
            || bytes.len() as u64 != input.actor.bytes
            || crate::model::digest(&bytes) != input.actor.sha256
        {
            return Err("Research actor pin differs".into());
        }
        let raw = crate::public_stochastic_record::unique_json(&bytes, MAX_BYTES)?;
        if raw["schema"].as_str() != Some(&input.artifact_schema)
            || raw
                .get("featureSchema")
                .is_some_and(|v| v.as_u64() != Some(4))
            || [
                "sourceAuthenticated",
                "sourceQualified",
                "verifiedNativePolicy",
                "adoptionGranted",
                "originalCheckpoint",
            ]
            .iter()
            .any(|key| raw.get(key).is_some_and(|v| v.as_bool() != Some(false)))
        {
            return Err("Research actor schema or provenance claim differs".into());
        }
        let bits = raw["parameterBits"]
            .as_array()
            .ok_or("Research actor bits missing")?;
        if bits.len() != crate::public_model::PARAMETER_COUNT {
            return Err("Research actor parameter count differs".into());
        }
        let parameters = bits
            .iter()
            .map(|v| {
                let s = v.as_str().ok_or("Research parameter is not a bit string")?;
                if s.len() != 8 || !s.bytes().all(|c| c.is_ascii_hexdigit()) {
                    return Err("Research parameter bits differ".to_string());
                }
                let f = f32::from_bits(u32::from_str_radix(s, 16).map_err(|e| e.to_string())?);
                if !f.is_finite() {
                    return Err("Research actor parameter is not finite".into());
                }
                Ok(f)
            })
            .collect::<Result<Vec<_>, String>>()?;
        let actor = PublicPolicyModel { parameters };
        actor.validate()?;
        let description = serde_json::json!({"schema":"research-semantic-playing-origin-v1", "input":input,
            "external":external.descriptor(), "sourceAuthenticated":false,"sourceQualified":false,
            "verifiedNativePolicy":false,"adoptionGranted":false});
        let checksum = hash(b"research-semantic-playing-origin-v1\0", &description)?;
        let origin = Arc::new(ResearchOrigin {
            input,
            required_external: external.clone(),
            description,
            checksum,
            tables: BTreeMap::new(),
        });
        origin.recheck()?;
        Ok(Self { origin, actor })
    }
    fn owner(&self) -> Result<SemanticFreshOwner, String> {
        SemanticFreshOwner::research(
            Arc::clone(&self.origin),
            self.actor.clone(),
            self.origin.input.critic_seed,
        )
    }
    pub fn origin_checksum(&self) -> &str {
        &self.origin.checksum
    }
    pub fn actor_parameters(&self) -> Result<&[f32], String> {
        self.origin.recheck()?;
        Ok(self.actor.parameters())
    }
    /// Reuses the existing guarded proposal collector and the existing SGD
    /// actor/critic update. No raw actor-step or test fixture is accepted.
    #[allow(clippy::too_many_arguments)]
    pub fn fresh_session<'a>(
        &self,
        external: &'a ConditionalNativeExclusions,
        history: &'a DeclaredSamplingHistory,
        seed: SamplingSeed,
        first: u64,
        replicate: u64,
        actor: OneStepConfig,
        critic: ResidualFitConfig,
    ) -> Result<ResearchSession<'a>, String> {
        self.origin.check_external(external)?;
        let session = SemanticPlayingJointSession::from_fresh_owner(
            self.owner()?,
            external,
            history,
            seed,
            first,
            replicate,
            actor,
            critic,
            SemanticCollectionComposition::GuardedProposalV2,
        )?;
        Ok(ResearchSession {
            session,
            history,
            external,
            origin: Arc::clone(&self.origin),
            actor_config: actor,
            critic_config: critic,
            stream: StreamRange {
                seed: seed.value(),
                replicate,
                first,
                next: first,
            },
            past_ranges: Vec::new(),
            past_families: BTreeSet::new(),
            restored: None,
        })
    }

    /// This is a research-only checkpoint, not a normal runner initialization.
    /// Caller-pinned bytes bind integrity, not authenticity of training claims.
    #[allow(clippy::too_many_arguments)]
    pub fn restore_session<'a>(
        &self,
        checkpoint: &ContentPin,
        external: &'a ConditionalNativeExclusions,
        history: &'a DeclaredSamplingHistory,
        seed: SamplingSeed,
        first: u64,
        replicate: u64,
        actor: OneStepConfig,
        critic: ResidualFitConfig,
    ) -> Result<ResearchSession<'a>, String> {
        let raw = read_pin(checkpoint)?;
        let saved: ResearchCheckpoint = serde_json::from_value(
            crate::public_stochastic_record::unique_json(&raw, MAX_BYTES)?,
        )
        .map_err(|e| e.to_string())?;
        self.origin.recheck()?;
        self.origin.check_external(external)?;
        external.recheck()?;
        history.recheck()?;
        if saved.schema != "research-semantic-playing-checkpoint-v2"
            || saved.origin != self.origin.checksum
            || saved.source_authenticated
            || saved.source_qualified
            || saved.verified_native_policy
            || saved.adoption_granted
            || saved.actor_config != serde_json::to_value(actor).map_err(|e| e.to_string())?
            || saved.critic_config != serde_json::to_value(critic).map_err(|e| e.to_string())?
            || saved.families.iter().any(|f| !external.contains_family(f))
            || saved.ranges.len() > 8192
            || saved.source_history.ranges.len() > 8192
            || saved.source_history.pins.len() > 8192
            || saved
                .source_history
                .pins
                .iter()
                .any(|p| !history.pins().contains(p))
        {
            return Err("Research checkpoint origin, configuration or exclusions differ".into());
        }
        let retained_external = saved.effective_external.load()?;
        if !retained_external.is_subset_of(external) {
            return Err(
                "Research checkpoint external exclusions omit a previously declared family".into(),
            );
        }
        for r in &saved.source_history.ranges {
            if !(3..=4).contains(&r[2])
                || r[3] >= r[2]
                || r[4] >= r[5]
                || !history.covers(
                    SamplingSeed::new(r[0]),
                    r[1],
                    r[2] as usize,
                    r[3] as usize,
                    r[4],
                    r[5],
                )
            {
                return Err(
                    "Research checkpoint omits previously declared sampling history".into(),
                );
            }
        }
        // Every prior interval must be excluded for every possible actor. The
        // common ledger separately rejects overlap with this new stream.
        for r in &saved.ranges {
            if r.first >= r.next
                || (3..=4).any(|p| {
                    (0..p).any(|a| {
                        !history.covers(
                            SamplingSeed::new(r.seed),
                            r.replicate,
                            p,
                            a,
                            r.first,
                            r.next,
                        )
                    })
                })
            {
                return Err("Research checkpoint sampling history is not excluded".into());
            }
        }
        let session = match &saved.numerics {
            ResearchNumerics::Fresh { current } => {
                if !saved.families.is_empty() || !saved.ranges.is_empty() {
                    return Err("Fresh research checkpoint claims consumed work".into());
                }
                let fresh = SemanticPlayingJointSession::from_fresh_owner(
                    self.owner()?,
                    external,
                    history,
                    seed,
                    first,
                    replicate,
                    actor,
                    critic,
                    SemanticCollectionComposition::GuardedProposalV2,
                )?;
                if serde_json::to_value(fresh.identity()?).map_err(|e| e.to_string())? != *current {
                    return Err("Fresh research checkpoint identity differs".into());
                }
                fresh
            }
            ResearchNumerics::Updated { state } => {
                let identity = state.validate(&actor, &critic)?;
                if identity.composition() != SemanticCollectionComposition::GuardedProposalV2
                    || !history.pins().contains(checkpoint)
                    || saved.ranges.is_empty()
                    || saved.families.is_empty()
                    || identity.attempted_whole_four() != identity.completed_updates()
                    || identity.attempted_whole_four().checked_mul(4)
                        != Some(saved.families.len() as u64)
                    || saved
                        .ranges
                        .iter()
                        .try_fold(0_u64, |n, r| n.checked_add(r.next - r.first))
                        != identity.attempted_whole_four().checked_mul(4)
                {
                    return Err("Research checkpoint requires guarded admitted work".into());
                }
                for (i, r) in saved.ranges.iter().enumerate() {
                    if saved.ranges[..i].iter().any(|old| {
                        old.seed == r.seed
                            && old.replicate == r.replicate
                            && old.first < r.next
                            && r.first < old.next
                    }) {
                        return Err("Research checkpoint repeats a sampling interval".into());
                    }
                }
                SemanticPlayingJointSession::from_checkpoint_owner(
                    self.owner()?,
                    state.clone(),
                    external,
                    history,
                    seed,
                    first,
                    replicate,
                    actor,
                    critic,
                )?
            }
        };
        retained_external.recheck()?;
        crate::human_policy_dataset::recheck_absolute_pin(checkpoint)?;
        Ok(ResearchSession {
            session,
            history,
            external,
            origin: Arc::clone(&self.origin),
            actor_config: actor,
            critic_config: critic,
            stream: StreamRange {
                seed: seed.value(),
                replicate,
                first,
                next: first,
            },
            past_ranges: saved.ranges,
            past_families: saved.families,
            restored: Some(saved.numerics),
        })
    }
}

fn read_pin(pin: &ContentPin) -> Result<Vec<u8>, String> {
    crate::human_policy_dataset::recheck_absolute_pin(pin)?;
    let mut bytes = Vec::new();
    std::fs::File::open(&pin.path)
        .map_err(|e| e.to_string())?
        .take((MAX_BYTES + 1) as u64)
        .read_to_end(&mut bytes)
        .map_err(|e| e.to_string())?;
    if bytes.len() > MAX_BYTES
        || bytes.len() as u64 != pin.bytes
        || crate::model::digest(&bytes) != pin.sha256
    {
        return Err("Research checkpoint pin differs".into());
    }
    Ok(bytes)
}

#[derive(Clone, Deserialize, Serialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct StreamRange {
    seed: u64,
    replicate: u64,
    first: u64,
    next: u64,
}
#[derive(Clone, Deserialize, Serialize)]
#[serde(tag = "kind", rename_all = "camelCase", deny_unknown_fields)]
enum ResearchNumerics {
    Fresh { current: serde_json::Value },
    Updated { state: SemanticCheckpointState },
}
#[derive(Deserialize, Serialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct ResearchHistory {
    ranges: Vec<[u64; 6]>,
    pins: Vec<ContentPin>,
}
/// A descriptor alone grants no exclusion custody. Restoration resolves its
/// complete raw file through the typed loader before comparing family sets.
#[derive(Deserialize, Serialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct ResearchExclusions {
    pin: ContentPin,
    count: usize,
    canonical_digest: String,
}
impl ResearchExclusions {
    fn capture(external: &ConditionalNativeExclusions) -> Result<Self, String> {
        serde_json::from_value(
            serde_json::to_value(external.descriptor()).map_err(|e| e.to_string())?,
        )
        .map_err(|e| e.to_string())
    }
    fn load(&self) -> Result<ConditionalNativeExclusions, String> {
        ConditionalNativeExclusions::load(
            std::path::Path::new(&self.pin.path),
            &self.pin,
            self.count,
            &self.canonical_digest,
        )
    }
}
#[derive(Deserialize, Serialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct ResearchCheckpoint {
    schema: String,
    origin: String,
    actor_config: serde_json::Value,
    critic_config: serde_json::Value,
    numerics: ResearchNumerics,
    families: BTreeSet<String>,
    ranges: Vec<StreamRange>,
    source_history: ResearchHistory,
    effective_external: ResearchExclusions,
    source_authenticated: bool,
    source_qualified: bool,
    verified_native_policy: bool,
    adoption_granted: bool,
}

/// Thin custody wrapper: all actual collection, auditing and numeric updates
/// remain in SemanticPlayingJointSession. Only checked research restoration is added.
pub struct ResearchSession<'a> {
    session: SemanticPlayingJointSession<'a>,
    history: &'a DeclaredSamplingHistory,
    external: &'a ConditionalNativeExclusions,
    origin: Arc<ResearchOrigin>,
    actor_config: OneStepConfig,
    critic_config: ResidualFitConfig,
    stream: StreamRange,
    past_ranges: Vec<StreamRange>,
    past_families: BTreeSet<String>,
    restored: Option<ResearchNumerics>,
}
impl<'a> ResearchSession<'a> {
    pub fn session(&self) -> &SemanticPlayingJointSession<'a> {
        &self.session
    }
    pub fn collect_and_update_whole_four(
        &mut self,
        seeds: [u32; 4],
        pool: &ConditionalFrozenPool,
        prepared_opponent: Option<&PreparedPublicPolicy>,
        limits: CollectionLimits,
    ) -> Result<SemanticPlayingJointReport, String> {
        self.session
            .collect_and_update_whole_four(seeds, pool, prepared_opponent, limits)
    }
    /// Only admitted state can be saved. A failed or burned-but-unadmitted
    /// attempt is rejected, rather than dropping consumption on restoration.
    pub fn checkpoint_bytes(&self) -> Result<Vec<u8>, String> {
        self.session.journal_recheck()?;
        if self.session.stopped() {
            return Err("Stopped research session cannot be checkpointed".into());
        }
        let mut ranges = self.past_ranges.clone();
        let mut families = self.past_families.clone();
        families.extend(self.session.consumed_families().iter().cloned());
        let numerics = if self.session.attempts() == 0 {
            self.restored.clone().unwrap_or(ResearchNumerics::Fresh {
                current: serde_json::to_value(self.session.identity()?)
                    .map_err(|e| e.to_string())?,
            })
        } else {
            let mut range = self.stream.clone();
            range.next = self.session.next_episode_ordinal();
            ranges.push(range);
            ResearchNumerics::Updated {
                state: self.session.checkpoint_state()?,
            }
        };
        if ranges.len() > 8192 {
            return Err("Research checkpoint sampling range bound".into());
        }
        let saved = ResearchCheckpoint {
            schema: "research-semantic-playing-checkpoint-v2".into(),
            origin: self.origin.checksum.clone(),
            actor_config: serde_json::to_value(self.actor_config).map_err(|e| e.to_string())?,
            critic_config: serde_json::to_value(self.critic_config).map_err(|e| e.to_string())?,
            numerics,
            families,
            ranges,
            source_history: ResearchHistory {
                ranges: self.history.declared_ranges(),
                pins: self.history.pins().to_vec(),
            },
            effective_external: ResearchExclusions::capture(self.external)?,
            source_authenticated: false,
            source_qualified: false,
            verified_native_policy: false,
            adoption_granted: false,
        };
        let bytes = serde_json::to_vec(&saved).map_err(|e| e.to_string())?;
        if bytes.len() > MAX_BYTES {
            return Err("Research checkpoint byte bound".into());
        }
        self.session.journal_recheck()?;
        Ok(bytes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dataset::{seed_family_id, split_for_family};
    use crate::model::digest;
    use crate::semantic_playing_rl_collection::{DeclaredSamplingRange, PoolRef};
    use serde_json::json;
    use std::path::{Path, PathBuf};
    mod temp_root {
        include!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/support/temp_root.rs"
        ));
    }
    struct Files(PathBuf);
    impl Files {
        fn new() -> Self {
            Self(temp_root::create("research-playing-fresh").unwrap())
        }
        fn write(&self, name: &str, bytes: &[u8]) -> ContentPin {
            let path = self.0.join(name);
            std::fs::write(&path, bytes).unwrap();
            ContentPin {
                path: path.to_str().unwrap().into(),
                bytes: bytes.len() as u64,
                sha256: digest(bytes),
            }
        }
        fn json(&self, name: &str, value: &serde_json::Value) -> ContentPin {
            self.write(name, &serde_json::to_vec(value).unwrap())
        }
        fn exclusions(
            &self,
            name: &str,
            families: BTreeSet<String>,
        ) -> (ConditionalNativeExclusions, ContentPin) {
            let list = families.into_iter().collect::<Vec<_>>();
            let pin = self.json(
                name,
                &json!({"schema":"tzolkin-conditional-native-exclusions-v1","families":list}),
            );
            let external = ConditionalNativeExclusions::load(
                Path::new(&pin.path),
                &pin,
                list.len(),
                &digest(format!("{}\n", list.join("\n")).as_bytes()),
            )
            .unwrap();
            (external, pin)
        }
        fn actor(&self) -> (ResearchActorInput, Vec<f32>) {
            let mut parameters = crate::public_model::initialized_model(31).parameters;
            parameters[0] = -0.0;
            let bits: Vec<_> = parameters
                .iter()
                .map(|f| format!("{:08x}", f.to_bits()))
                .collect();
            let actor = self.json(
                "actor.json",
                &json!({"schema":"synthetic-research-actor-v1","parameterBits":bits}),
            );
            (
                ResearchActorInput {
                    schema: "researchActorFreshStartV1".into(),
                    feature_schema: 4,
                    actor,
                    artifact_schema: "synthetic-research-actor-v1".into(),
                    critic_seed: 101,
                },
                parameters,
            )
        }
    }
    impl Drop for Files {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }
    fn critic() -> ResidualFitConfig {
        ResidualFitConfig::new(1e-4, 0.01, 16000).unwrap()
    }
    fn bits(p: &[f32]) -> Vec<u32> {
        p.iter().map(|f| f.to_bits()).collect()
    }
    fn pool() -> ConditionalFrozenPool {
        let opponent = json!({"kind":"heuristic"});
        ConditionalFrozenPool::from_json(&serde_json::to_vec(&json!({
            "schema":"tzolkin-conditional-human-mixed-frozen-pool-v1", "weights":crate::policy::HeuristicWeights::default(),
            "rosters":[[null,opponent,opponent],[opponent,null,opponent],
                [null,opponent,opponent,opponent],[opponent,null,opponent,opponent]]})).unwrap()).unwrap()
    }
    #[test]
    fn research_checkpoint_retains_effective_unconsumed_exclusions() {
        let f = Files::new();
        let initial = seed_family_id(42);
        let extra: Vec<_> = (0..1000)
            .filter(|s| {
                *s != 42 && split_for_family(&seed_family_id(*s)).unwrap() == DatasetSplit::Train
            })
            .take(3)
            .map(seed_family_id)
            .collect();
        let (e0, pin0) = f.exclusions("e0.json", BTreeSet::from([initial.clone()]));
        let families = |n| {
            std::iter::once(initial.clone())
                .chain(extra[..n].iter().cloned())
                .collect()
        };
        let (e1, _) = f.exclusions("e1.json", families(1));
        let (e2, _) = f.exclusions("e2.json", families(2));
        let (e3, _) = f.exclusions("e3.json", families(3));
        let history = DeclaredSamplingHistory::declare_complete(vec![], vec![pin0]).unwrap();
        let (input, _) = f.actor();
        let prepared = PreparedResearchActor::load(input, &e0).unwrap();
        let session = prepared
            .fresh_session(
                &e1,
                &history,
                SamplingSeed::new(19),
                0,
                0,
                OneStepConfig::default(),
                critic(),
            )
            .unwrap();
        let checkpoint1 = f.write("checkpoint1.json", &session.checkpoint_bytes().unwrap());
        let restore = |checkpoint: &ContentPin, external| {
            prepared.restore_session(
                checkpoint,
                external,
                &history,
                SamplingSeed::new(19),
                0,
                0,
                OneStepConfig::default(),
                critic(),
            )
        };
        assert!(session.session.consumed_families().is_empty());
        assert!(
            restore(&checkpoint1, &e0)
                .err()
                .unwrap()
                .contains("external exclusions")
        );
        assert!(restore(&checkpoint1, &e1).is_ok());
        let resumed = restore(&checkpoint1, &e2).unwrap();
        assert_eq!(
            session.session.identity().unwrap().checksum(),
            resumed.session.identity().unwrap().checksum()
        );
        // The second save must capture E2, rather than replaying E1 from the
        // restored payload or only checking the original preparation's E0.
        let checkpoint2 = f.write("checkpoint2.json", &resumed.checkpoint_bytes().unwrap());
        assert!(resumed.session.consumed_families().is_empty());
        assert!(
            restore(&checkpoint2, &e1)
                .err()
                .unwrap()
                .contains("external exclusions")
        );
        assert!(restore(&checkpoint2, &e2).is_ok());
        assert!(restore(&checkpoint2, &e3).is_ok());
        let raw: serde_json::Value =
            serde_json::from_slice(&std::fs::read(&checkpoint2.path).unwrap()).unwrap();
        assert_eq!(raw["schema"], "research-semantic-playing-checkpoint-v2");
        assert_eq!(raw["effectiveExternal"]["count"], 3);
        assert_eq!(raw["numerics"]["current"]["updates"], 0);
        assert!(raw["families"].as_array().unwrap().is_empty());
        let mut legacy = raw.clone();
        legacy["schema"] = json!("research-semantic-playing-checkpoint-v1");
        legacy.as_object_mut().unwrap().remove("effectiveExternal");
        assert!(restore(&f.json("legacy.json", &legacy), &e3).is_err());
        let mut wrong_descriptor = raw;
        wrong_descriptor["effectiveExternal"]["canonicalDigest"] = json!("0".repeat(64));
        assert!(restore(&f.json("wrong-descriptor.json", &wrong_descriptor), &e3).is_err());
        // The descriptor's raw input must actually be read and checked, even
        // though it is not one of the sampling-history evidence pins.
        f.write("e2.json", b"changed");
        assert!(
            restore(&checkpoint2, &e3)
                .err()
                .unwrap()
                .contains("External exclusions raw pin differs")
        );
    }
    #[test]
    fn research_fresh_bits_and_pinned_input_refusals() {
        let f = Files::new();
        let (external, pin) = f.exclusions("external.json", BTreeSet::from([seed_family_id(42)]));
        let history = DeclaredSamplingHistory::declare_complete(vec![], vec![pin]).unwrap();
        let (input, expected) = f.actor();
        let prepared = PreparedResearchActor::load(input.clone(), &external).unwrap();
        let owner = prepared.owner().unwrap();
        let state =
            crate::conditional_semantic_playing_rl::SemanticTrainingState::from_fresh(owner)
                .unwrap();
        assert_eq!(state.root().policy_version(), POLICY_VERSION);
        assert_eq!(state.root().input_contract(), INPUT_CONTRACT);
        assert!(
            !state.root().source_authenticated()
                && !state.root().source_qualified()
                && !state.root().verified_native_policy()
        );
        let session = prepared
            .fresh_session(
                &external,
                &history,
                SamplingSeed::new(19),
                0,
                0,
                OneStepConfig::default(),
                critic(),
            )
            .unwrap();
        assert_eq!(
            bits(session.session.actor_parameters().unwrap()),
            bits(&expected)
        );
        assert_eq!(
            bits(session.session.critic_parameters().unwrap()),
            bits(&crate::public_policy_baseline::zero_output_parameters(101))
        );
        assert_eq!(session.session.critic_parameters().unwrap().len(), 12353);
        for players in [3, 4] {
            let state = crate::conditional_human_rl_update::tests::terminal_start(players, 1);
            let observation =
                tzolkin_core::observation::observe(&state, state.current_player).unwrap();
            let context =
                crate::public_state_critic::PublicStateContext::from_observation(&observation)
                    .unwrap();
            assert_eq!(
                crate::public_policy_baseline::baseline(
                    session.session.critic_parameters().unwrap(),
                    &context
                )
                .unwrap(),
                1.0 / players as f64
            );
        }

        let id = session.session.identity().unwrap();
        assert_eq!(
            (
                id.completed_updates(),
                id.attempted_whole_four(),
                id.credited_playing_decisions()
            ),
            (0, 0, 0)
        );
        assert!(!id.source_authenticated() && !id.source_qualified());
        let saved = f.write("fresh.json", &session.checkpoint_bytes().unwrap());
        let restored = prepared
            .restore_session(
                &saved,
                &external,
                &history,
                SamplingSeed::new(19),
                0,
                0,
                OneStepConfig::default(),
                critic(),
            )
            .unwrap();
        assert_eq!(
            id.checksum(),
            restored.session.identity().unwrap().checksum()
        );
        assert_eq!(
            bits(restored.session.actor_parameters().unwrap()),
            bits(&expected)
        );
        assert_eq!(
            session.checkpoint_bytes().unwrap(),
            restored.checkpoint_bytes().unwrap()
        );
        let mut wrong = input.clone();
        wrong.feature_schema = 2;
        assert!(PreparedResearchActor::load(wrong, &external).is_err());
        let mut wrong = input.clone();
        wrong.actor.sha256 = "0".repeat(64);
        assert!(PreparedResearchActor::load(wrong, &external).is_err());
        let mut raw: serde_json::Value =
            serde_json::from_slice(&std::fs::read(&input.actor.path).unwrap()).unwrap();
        for replacement in [json!("7f800000"), json!("7fc00000"), json!("xyz"), json!(1)] {
            let mut bad = raw.clone();
            bad["parameterBits"][0] = replacement;
            let mut i = input.clone();
            i.actor = f.json("invalid.json", &bad);
            assert!(PreparedResearchActor::load(i, &external).is_err());
        }
        let mut bad = raw.clone();
        bad["parameterBits"].as_array_mut().unwrap().pop();
        let mut i = input.clone();
        i.actor = f.json("short.json", &bad);
        assert!(PreparedResearchActor::load(i, &external).is_err());
        for key in [
            "sourceAuthenticated",
            "sourceQualified",
            "verifiedNativePolicy",
            "adoptionGranted",
            "originalCheckpoint",
        ] {
            let mut bad = raw.clone();
            bad[key] = json!(true);
            let mut i = input.clone();
            i.actor = f.json("claim.json", &bad);
            assert!(PreparedResearchActor::load(i, &external).is_err());
        }
        raw["schema"] = json!("wrong");
        let mut i = input.clone();
        i.actor = f.json("schema.json", &raw);
        assert!(PreparedResearchActor::load(i, &external).is_err());
        // Successful load does not turn a later raw-file mutation into custody.
        std::fs::write(&input.actor.path, b"changed").unwrap();
        assert!(prepared.actor_parameters().is_err());
        assert!(session.checkpoint_bytes().is_err());
    }

    #[test]
    fn research_common_update_and_research_only_checkpoint_roundtrip() {
        let f = Files::new();
        let initial_family = seed_family_id(42);
        let (external, pin) =
            f.exclusions("external.json", BTreeSet::from([initial_family.clone()]));
        let older = DeclaredSamplingRange::new(SamplingSeed::new(17), 0, 3, 0, 90, 94).unwrap();
        let history =
            DeclaredSamplingHistory::declare_complete(vec![older.clone()], vec![pin.clone()])
                .unwrap();
        let (input, _) = f.actor();
        let prepared = PreparedResearchActor::load(input.clone(), &external).unwrap();
        let mut session = prepared
            .fresh_session(
                &external,
                &history,
                SamplingSeed::new(19),
                0,
                0,
                OneStepConfig::default(),
                critic(),
            )
            .unwrap();
        let before = session.session.identity().unwrap().clone();
        let pool = pool();
        let limits = CollectionLimits::new(64, 65536, 8 * 1024 * 1024, 1024 * 1024).unwrap();
        let seeds: Vec<_> = (0..1000)
            .filter(|s| {
                *s != 42 && split_for_family(&seed_family_id(*s)).unwrap() == DatasetSplit::Train
            })
            .take(4)
            .collect();
        let plan = session
            .session
            .plan_reserved(
                seeds.clone().try_into().unwrap(),
                PoolRef::Legacy(&pool, None),
                limits,
            )
            .unwrap();
        let reservation = session.session.burn_reserved(plan);
        // This small synthetic terminal fixture is test-only. The public
        // entrance delegates normal production game creation to the collector.
        let proposal = session
            .session
            .stage_reserved_using(&reservation, PoolRef::Legacy(&pool, None), |config| {
                let mut state = crate::conditional_human_rl_update::tests::terminal_start(
                    config.players(),
                    config.environment_seed(),
                );
                state.turn.mode = tzolkin_core::TurnMode::None;
                state.turn.count = 0;
                Ok(state)
            })
            .unwrap();
        session.session.commit_prepared(proposal);
        let after = session.session.identity().unwrap();
        assert_eq!(
            (
                after.completed_updates(),
                after.attempted_whole_four(),
                after.next_episode_ordinal()
            ),
            (1, 1, 4)
        );
        assert!(after.credited_playing_decisions() > 0);
        assert_ne!(before.actor_checksum(), after.actor_checksum());
        assert_ne!(before.critic_checksum(), after.critic_checksum());
        assert_eq!(before.root_checksum(), after.root_checksum());
        let saved = f.write("updated.json", &session.checkpoint_bytes().unwrap());
        let (expanded, expanded_pin) = f.exclusions(
            "expanded.json",
            std::iter::once(initial_family)
                .chain(seeds.iter().map(|s| seed_family_id(*s)))
                .collect(),
        );
        let ranges = || {
            std::iter::once(older.clone())
                .chain((3..=4).flat_map(|p| {
                    (0..p).map(move |a| {
                        DeclaredSamplingRange::new(SamplingSeed::new(19), 0, p, a, 0, 4).unwrap()
                    })
                }))
                .collect()
        };
        let covered = DeclaredSamplingHistory::declare_complete(
            ranges(),
            vec![pin, expanded_pin, saved.clone()],
        )
        .unwrap();
        let restore = |checkpoint: &ContentPin, ext, hist| {
            prepared.restore_session(
                checkpoint,
                ext,
                hist,
                SamplingSeed::new(19),
                4,
                0,
                OneStepConfig::default(),
                critic(),
            )
        };
        assert!(restore(&saved, &external, &covered).is_err());
        assert!(restore(&saved, &expanded, &history).is_err());
        let own_only = DeclaredSamplingHistory::declare_complete(
            ranges().into_iter().skip(1).collect(),
            covered.pins().to_vec(),
        )
        .unwrap();
        assert!(
            restore(&saved, &expanded, &own_only)
                .err()
                .unwrap()
                .contains("previously declared sampling history")
        );
        let restored = restore(&saved, &expanded, &covered).unwrap();
        assert_eq!(
            after.checksum(),
            restored.session.identity().unwrap().checksum()
        );
        assert_eq!(
            bits(session.session.actor_parameters().unwrap()),
            bits(restored.session.actor_parameters().unwrap())
        );
        assert_eq!(
            bits(session.session.critic_parameters().unwrap()),
            bits(restored.session.critic_parameters().unwrap())
        );
        let again: serde_json::Value =
            serde_json::from_slice(&restored.checkpoint_bytes().unwrap()).unwrap();
        let original: serde_json::Value =
            serde_json::from_slice(&session.checkpoint_bytes().unwrap()).unwrap();
        assert_eq!(again["numerics"], original["numerics"]);
        assert!(
            restored
                .session
                .plan_reserved(
                    seeds.try_into().unwrap(),
                    PoolRef::Legacy(&pool, None),
                    limits
                )
                .is_err()
        );
        let raw: serde_json::Value =
            serde_json::from_slice(&std::fs::read(&saved.path).unwrap()).unwrap();
        for key in ["origin", "schema"] {
            let mut bad = raw.clone();
            bad[key] = json!("wrong");
            let bad = f.json("bad-checkpoint.json", &bad);
            assert!(restore(&bad, &expanded, &covered).is_err());
        }
        let mut bad = raw.clone();
        bad["families"].as_array_mut().unwrap().pop();
        let bad = f.json("false-history.json", &bad);
        let mut retained = covered.pins().to_vec();
        retained.push(bad.clone());
        let false_history = DeclaredSamplingHistory::declare_complete(ranges(), retained).unwrap();
        assert!(
            restore(&bad, &expanded, &false_history)
                .err()
                .unwrap()
                .contains("requires guarded admitted work")
        );
        let mut bad = raw.clone();
        bad["numerics"]["state"]["current"]["updates"] = json!(2);
        let bad = f.json("false-counter.json", &bad);
        assert!(
            restore(&bad, &expanded, &covered)
                .err()
                .unwrap()
                .contains("identity differs")
        );
        let mut bad = raw.clone();
        bad["numerics"]["state"]["actor"]["parameters"][0] = json!(0.25);
        let bad = f.json("false-actor.json", &bad);
        assert!(
            restore(&bad, &expanded, &covered)
                .err()
                .unwrap()
                .contains("identity differs")
        );
        let mut changed_origin = input;
        changed_origin.critic_seed += 1;
        let other = PreparedResearchActor::load(changed_origin, &external).unwrap();
        assert!(
            other
                .restore_session(
                    &saved,
                    &expanded,
                    &covered,
                    SamplingSeed::new(19),
                    4,
                    0,
                    OneStepConfig::default(),
                    critic()
                )
                .is_err()
        );
    }
}
