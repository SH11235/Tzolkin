//! Conditional Human-origin whole-four updates. This in-memory route is not a
//! native qualification, durable journal, saved-owner restore or Ready certificate.
use crate::arena::{
    BorrowedArenaPolicy, FrozenReactiveOpponent, resource_cycle::ResourceCycleConfig,
};
use crate::conditional_human_rl::{ConditionalRlOrigin, ConditionalRlOwner, INPUT_CONTRACT};
use crate::dataset::{DatasetSplit, seed_family_id, split_for_family};
use crate::features::FeatureEncoder;
use crate::human_initial_wealth::ContentPin;
use crate::kernel::{Kernel, ResolvedKernel};
use crate::public_model::{PublicPolicyModel, ValueValidity, predict_numeric, validate_contract};
use crate::public_native::PreparedPublicPolicy;
use crate::public_policy_baseline::ResidualFitConfig;
use crate::public_policy_episode::{
    ActorLink, AppliedActorStep, applied_mixed_plan, applied_step, canonical_record_checksum,
};
use crate::public_policy_mixed::{MixedNumericMember, prepare_numeric_timed, rounded_delta};
use crate::public_policy_update::{
    Deltas, NumericReport, OneStepConfig, Untimed, checkpoint_digest, hash,
};
use crate::public_stochastic::{
    RNG_VERSION, SamplingSeed, SamplingStreamIdentity, UNIFORM_MIXTURE,
};
use crate::public_stochastic_native::{
    CollectionLimits, NativeStochasticConfig, numerical_target, run_conditional_mixed_shared,
};
use crate::public_stochastic_record::{
    Config, Header, Record, decode_typed_record, encode_typed_record, hex64, unique_json,
};
use crate::public_trade_guard::TradeGuardConfig;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{collections::BTreeSet, fs, io::Read, path::Path};
use tzolkin_core::{
    GameOptions,
    observation::{MOVE_SCHEMA, OBSERVATION_SCHEMA, Observation},
};

pub const RECORD_SCHEMA: &str = "tzolkin-conditional-human-mixed-native-record-v1";
pub const SOURCE_KIND: &str = "conditionalHumanMixedNativeV1";
pub const POLICY_VERSION: &str = "conditional-human-current-mixed-scalar-v1";
pub const SAMPLING_VERSION: &str = "conditional-human-current-mixed-uniform-tick53-v1";
pub const REPORT_SCHEMA: &str = "tzolkin-conditional-human-mixed-joint-report-v1";
pub const ARTIFACT_SCHEMA: &str = "tzolkin-conditional-human-mixed-joint-artifact-v1";
pub const MAX_EXTERNAL_FAMILIES: usize = 262_144;
pub const MAX_EXTERNAL_BYTES: usize = 32 * 1024 * 1024;
pub const MAX_ATTEMPTS: u64 = 64;
pub const MAX_SPENT_FAMILIES: usize = 4096;

/// Human tables and native environment families use distinct typed namespaces.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct TypedHumanTableReservation {
    table_id: String,
    split: DatasetSplit,
}
#[derive(Clone, Debug, Serialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(rename_all = "camelCase")]
pub struct NativeSeedFamily {
    digest: String,
}
impl NativeSeedFamily {
    fn train(seed: u32) -> Result<Self, String> {
        let digest = seed_family_id(seed);
        if split_for_family(&digest)? != DatasetSplit::Train {
            return Err("Conditional native collection requires a Train seed family".into());
        }
        Ok(Self { digest })
    }
    pub fn digest(&self) -> &str {
        &self.digest
    }
}
#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct NativeExclusionDescriptor {
    pin: ContentPin,
    count: usize,
    canonical_digest: String,
}
/// Only checked raw input emits this closure; a descriptor cannot reserve work.
pub struct ConditionalNativeExclusions {
    descriptor: NativeExclusionDescriptor,
    families: BTreeSet<String>,
}
impl ConditionalNativeExclusions {
    pub fn load(
        path: &Path,
        expected: &ContentPin,
        expected_count: usize,
        expected_digest: &str,
    ) -> Result<Self, String> {
        if !path.is_absolute()
            || path.to_str() != Some(expected.path.as_str())
            || !(1..=MAX_EXTERNAL_FAMILIES).contains(&expected_count)
            || !checkpoint_digest(expected_digest)
        {
            return Err("Conditional external exclusion input binding/bound".into());
        }
        let metadata = fs::symlink_metadata(path).map_err(|e| e.to_string())?;
        if !metadata.is_file() || metadata.len() as usize > MAX_EXTERNAL_BYTES {
            return Err("External exclusions require a bounded regular raw file".into());
        }
        let mut raw = Vec::new();
        fs::File::open(path)
            .map_err(|e| e.to_string())?
            .take((MAX_EXTERNAL_BYTES + 1) as u64)
            .read_to_end(&mut raw)
            .map_err(|e| e.to_string())?;
        if raw.len() > MAX_EXTERNAL_BYTES
            || expected.bytes != raw.len() as u64
            || expected.sha256 != crate::model::digest(&raw)
        {
            return Err("External exclusions raw pin differs".into());
        }
        #[derive(Deserialize)]
        #[serde(rename_all = "camelCase", deny_unknown_fields)]
        struct Wire {
            schema: String,
            families: Vec<String>,
        }
        let wire: Wire = serde_json::from_value(unique_json(&raw, MAX_EXTERNAL_BYTES)?)
            .map_err(|e| e.to_string())?;
        if wire.schema != "tzolkin-conditional-native-exclusions-v1"
            || wire.families.len() != expected_count
            || wire.families.iter().any(|f| !checkpoint_digest(f))
            || wire.families.windows(2).any(|p| p[0] >= p[1])
        {
            return Err(
                "External exclusions require the complete declared sorted unique union".into(),
            );
        }
        let canonical_digest = exclusion_digest(&wire.families);
        if canonical_digest != expected_digest {
            return Err("External exclusion union digest differs".into());
        }
        Ok(Self {
            descriptor: NativeExclusionDescriptor {
                pin: expected.clone(),
                count: expected_count,
                canonical_digest,
            },
            families: wire.families.into_iter().collect(),
        })
    }
    pub fn descriptor(&self) -> &NativeExclusionDescriptor {
        &self.descriptor
    }
    fn recheck(&self) -> Result<(), String> {
        let checked = Self::load(
            Path::new(&self.descriptor.pin.path),
            &self.descriptor.pin,
            self.descriptor.count,
            &self.descriptor.canonical_digest,
        )?;
        if checked.families != self.families {
            return Err("External family closure changed".into());
        }
        Ok(())
    }
}
// The declared union uses sorted UTF-8 lines, including the final LF. Its
// descriptor and source identity are hashed separately in their own domains.
fn exclusion_digest(families: &[String]) -> String {
    let mut digest = Sha256::new();
    for family in families {
        digest.update(family.as_bytes());
        digest.update(b"\n");
    }
    format!("{:x}", digest.finalize())
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(tag = "kind", rename_all = "camelCase", deny_unknown_fields)]
pub enum ConditionalOpponentSpec {
    Heuristic,
    CornFirstSetup,
    CornFirstUxmalOpening,
    ResourceCycle { configuration: ResourceCycleConfig },
    PreparedBc,
    PreparedBcTradeGuard { guard: TradeGuardConfig },
    Search,
    ParentArgmax,
}
#[derive(Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct PoolWire {
    schema: String,
    weights: crate::policy::HeuristicWeights,
    rosters: [Vec<Option<ConditionalOpponentSpec>>; 4],
}
pub struct ConditionalFrozenPool {
    wire: PoolWire,
    checksum: String,
}
impl ConditionalFrozenPool {
    pub(crate) fn binding(
        &self,
        prepared: Option<&PreparedPublicPolicy>,
    ) -> Result<serde_json::Value, String> {
        let rosters = self.freeze(prepared)?;
        Ok(serde_json::json!({"checksum":self.checksum,
            "rosters":rosters.iter().map(|r|serde_json::json!({"learner":r.learner,
                "opponents":r.descriptions,"checksum":r.checksum})).collect::<Vec<_>>()}))
    }
    pub fn from_json(raw: &[u8]) -> Result<Self, String> {
        if raw.len() > 64 * 1024 {
            return Err("Conditional pool byte bound".into());
        }
        let wire: PoolWire =
            serde_json::from_value(unique_json(raw, 64 * 1024)?).map_err(|e| e.to_string())?;
        if wire.schema != "tzolkin-conditional-human-mixed-frozen-pool-v1"
            || wire.rosters.iter().enumerate().any(|(i, r)| {
                r.len() != if i < 2 { 3 } else { 4 }
                    || r.iter().filter(|p| p.is_none()).count() != 1
                    || r.iter().flatten().any(|p| {
                        matches!(
                            p,
                            ConditionalOpponentSpec::Search | ConditionalOpponentSpec::ParentArgmax
                        )
                    })
            })
        {
            return Err(
                "Conditional pool requires 2x3p/2x4p, one learner and no Search/ParentArgmax"
                    .into(),
            );
        }
        let checksum = hash(
            b"tzolkin-conditional-human-mixed-frozen-pool-v1\0",
            &(&wire.schema, &wire.weights, &wire.rosters),
        )?;
        Ok(Self { wire, checksum })
    }
    fn freeze<'a>(
        &'a self,
        prepared: Option<&'a PreparedPublicPolicy>,
    ) -> Result<[ConditionalFrozenMixRoster<'a>; 4], String> {
        self.wire
            .rosters
            .iter()
            .map(|seats| {
                let opponents = seats
                    .iter()
                    .map(|p| {
                        p.as_ref()
                            .map(|p| {
                                let borrowed = match p {
                                    ConditionalOpponentSpec::Heuristic => {
                                        BorrowedArenaPolicy::Heuristic(&self.wire.weights)
                                    }
                                    ConditionalOpponentSpec::CornFirstSetup => {
                                        BorrowedArenaPolicy::CornFirstSetup(&self.wire.weights)
                                    }
                                    ConditionalOpponentSpec::CornFirstUxmalOpening => {
                                        BorrowedArenaPolicy::CornFirstUxmalOpening(
                                            &self.wire.weights,
                                        )
                                    }
                                    ConditionalOpponentSpec::ResourceCycle { configuration } => {
                                        BorrowedArenaPolicy::ResourceCycle(configuration)
                                    }
                                    ConditionalOpponentSpec::PreparedBc => {
                                        BorrowedArenaPolicy::PublicBc(
                                            prepared.ok_or("Missing genuine native BC opponent")?,
                                        )
                                    }
                                    ConditionalOpponentSpec::PreparedBcTradeGuard { guard } => {
                                        BorrowedArenaPolicy::PublicBcTradeGuard(
                                            prepared.ok_or("Missing genuine native BC opponent")?,
                                            guard,
                                        )
                                    }
                                    ConditionalOpponentSpec::Search
                                    | ConditionalOpponentSpec::ParentArgmax => {
                                        return Err("Forbidden conditional opponent".into());
                                    }
                                };
                                FrozenReactiveOpponent::new(borrowed)
                            })
                            .transpose()
                    })
                    .collect::<Result<Vec<_>, String>>()?;
                let learner = seats
                    .iter()
                    .position(Option::is_none)
                    .ok_or("Missing learner")?;
                let descriptions = opponents
                    .iter()
                    .map(|p| {
                        p.as_ref()
                            .map(|p| {
                                serde_json::to_value(p.description()).map_err(|e| e.to_string())
                            })
                            .transpose()
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                let checksum = hash(
                    b"tzolkin-conditional-frozen-absolute-roster-v1\0",
                    &(learner, &descriptions),
                )?;
                Ok(ConditionalFrozenMixRoster {
                    learner,
                    opponents,
                    descriptions,
                    checksum,
                })
            })
            .collect::<Result<Vec<_>, String>>()?
            .try_into()
            .map_err(|_| "Conditional roster shape".into())
    }
}
pub(crate) struct ConditionalFrozenMixRoster<'a> {
    learner: usize,
    opponents: Vec<Option<FrozenReactiveOpponent<'a>>>,
    descriptions: Vec<Option<serde_json::Value>>,
    checksum: String,
}
impl<'a> ConditionalFrozenMixRoster<'a> {
    pub(crate) fn learner_seat(&self) -> usize {
        self.learner
    }
    pub(crate) fn opponent(&self, actor: usize) -> Result<&FrozenReactiveOpponent<'a>, String> {
        self.opponents
            .get(actor)
            .and_then(Option::as_ref)
            .ok_or_else(|| "Missing conditional opponent".into())
    }
    fn contains_family(&self, family: &str) -> bool {
        self.opponents
            .iter()
            .flatten()
            .any(|p| p.contains_family(family))
    }
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct ConditionalMixedIdentity {
    policy_version: String,
    input_contract: String,
    origin_checksum: String,
    root_identity_checksum: String,
    parent_identity_checksum: String,
    actor_checksum: String,
    critic_checksum: String,
    mixed_update_count: u64,
    sampling_seed: u64,
    next_episode_ordinal: u64,
    game_options: GameOptions,
    uxmal_corn_limit: bool,
    source_qualified: bool,
    verified_native_policy: bool,
    checksum: String,
}
impl ConditionalMixedIdentity {
    pub(crate) fn next_ordinal(&self) -> u64 {
        self.next_episode_ordinal
    }
    pub(crate) fn origin_checksum(&self) -> &str {
        &self.origin_checksum
    }
    pub fn checksum(&self) -> &str {
        &self.checksum
    }
    pub fn mixed_update_count(&self) -> u64 {
        self.mixed_update_count
    }
    pub(crate) fn compatible(&self) -> bool {
        self.policy_version == POLICY_VERSION
            && self.input_contract == INPUT_CONTRACT
            && self.game_options == GameOptions::default()
            && !self.uxmal_corn_limit
            && !self.source_qualified
            && !self.verified_native_policy
            && self.mixed_update_count <= MAX_ATTEMPTS
    }
}
pub struct ConditionalMixedOwner {
    origin: ConditionalRlOrigin,
    actor: PublicPolicyModel,
    critic: Vec<f32>,
    identity: ConditionalMixedIdentity,
}
/// Portable summary of a checked Ready journal; raw custody paths stay private.
#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct ConditionalInferenceJournal<'a> {
    pub(crate) head_sha256: &'a str,
    pub(crate) head_bytes: u64,
    pub(crate) attempts: u64,
    pub(crate) credited_own_decisions: u64,
    pub(crate) spent_native_families: Vec<&'a str>,
    pub(crate) external_native_count: usize,
    pub(crate) external_native_digest: &'a str,
    pub(crate) external_native_sha256: &'a str,
}
impl ConditionalMixedOwner {
    pub fn from_count_zero(owner: ConditionalRlOwner) -> Result<Self, String> {
        owner.origin().recheck_sources()?;
        let (origin, actor, critic, root) = owner.into_mixed_parts();
        let identity = make_identity(
            &origin,
            &actor,
            &critic,
            root.checksum(),
            root.checksum(),
            0,
            root.sampling_seed(),
            0,
        )?;
        Ok(Self {
            origin,
            actor,
            critic,
            identity,
        })
    }
    pub fn identity(&self) -> &ConditionalMixedIdentity {
        &self.identity
    }
    pub fn origin(&self) -> &ConditionalRlOrigin {
        &self.origin
    }
    pub fn actor_parameters(&self) -> &[f32] {
        self.actor.parameters()
    }
    pub fn critic_parameters(&self) -> &[f32] {
        &self.critic
    }
    pub fn handle(&self) -> Result<ConditionalMixedPolicyHandle<'_>, String> {
        self.actor.validate()?;
        if !self.identity.compatible()
            || self.critic.len() != crate::public_state_critic::PARAMETER_COUNT
            || self.critic.iter().any(|x| !x.is_finite())
        {
            return Err("Conditional current owner contract".into());
        }
        Ok(ConditionalMixedPolicyHandle {
            owner: self,
            kernel: Kernel::Scalar.resolve()?,
        })
    }
    pub(crate) fn export_numeric_inference(
        &self,
        journal: ConditionalInferenceJournal<'_>,
    ) -> Result<Vec<u8>, String> {
        use tzolkin_inference::{conditional_deployment as output, deployment};
        self.handle()?;
        check_identity_bits(&self.identity, self.actor.parameters(), &self.critic)?;
        if self.identity.mixed_update_count == 0
            || self.origin.table_reservations().len() > 4096
            || journal.spent_native_families.len() > MAX_SPENT_FAMILIES
        {
            return Err("Conditional numeric export requires positive genuine updates".into());
        }
        #[derive(Serialize)]
        #[serde(rename_all = "camelCase")]
        struct Contract {
            model_version: &'static str,
            input_contract: &'static str,
            feature_schema: u32,
            feature_count: usize,
            hidden_count: usize,
            parameter_count: usize,
            rules_version: u32,
            rules_baseline: &'static str,
            observation_schema: u32,
            move_schema: u32,
            catalog_hash: String,
            backend: &'static str,
            numeric_contract: &'static str,
            scope: &'static str,
            value_validity: &'static str,
        }
        #[derive(Serialize)]
        #[serde(rename_all = "camelCase")]
        struct Source<'a> {
            task: &'static str,
            origin_checksum: &'a str,
            checkpoint_checksum: &'a str,
            model_checksum: &'a str,
            dataset: &'a crate::human_policy_training::ConditionalDatasetIdentity,
            table_reservations: &'a std::collections::BTreeMap<String, DatasetSplit>,
            current: &'a ConditionalMixedIdentity,
            journal: ConditionalInferenceJournal<'a>,
            source_authenticated: bool,
            source_qualified: bool,
            verified_native_policy: bool,
            close_scope: &'static str,
        }
        #[derive(Serialize)]
        #[serde(rename_all = "camelCase")]
        struct Wire<'a> {
            schema: &'static str,
            contract: Contract,
            source: Source<'a>,
            source_numerical_target: String,
            parameter_bits: Vec<String>,
            checksum: String,
        }
        let (checkpoint_checksum, model_checksum) = self.origin.deployment_checksums();
        let mut wire = Wire {
            schema: output::DEPLOYMENT_SCHEMA,
            contract: Contract {
                model_version: crate::public_model::MODEL_VERSION,
                input_contract: crate::public_model::INPUT_CONTRACT,
                feature_schema: crate::features::PUBLIC_FEATURE_SCHEMA,
                feature_count: crate::features::FEATURE_COUNT,
                hidden_count: crate::public_model::HIDDEN,
                parameter_count: crate::public_model::PARAMETER_COUNT,
                rules_version: crate::replay::RULES_VERSION,
                rules_baseline: crate::replay::RULES_BASELINE,
                observation_schema: OBSERVATION_SCHEMA,
                move_schema: MOVE_SCHEMA,
                catalog_hash: crate::replay::catalog_hash(),
                backend: "scalar",
                numeric_contract: deployment::NUMERIC_CONTRACT,
                scope: deployment::SCOPE,
                value_validity: deployment::VALUE_VALIDITY,
            },
            source: Source {
                task: "ConditionalHumanCurrentMixedPolicyInference",
                origin_checksum: self.origin.checksum(),
                checkpoint_checksum,
                model_checksum,
                dataset: self.origin.dataset(),
                table_reservations: self.origin.table_reservations(),
                current: &self.identity,
                journal,
                source_authenticated: false,
                source_qualified: false,
                verified_native_policy: false,
                close_scope: "returnedLibraryReadyOnly",
            },
            source_numerical_target: numerical_target(),
            parameter_bits: self
                .actor
                .parameters()
                .iter()
                .map(|v| format!("{:08x}", v.to_bits()))
                .collect(),
            checksum: String::new(),
        };
        // Finite owner shapes and explicit metadata bounds are checked by the
        // separate closed decoder; no conversion to a native/BC wire occurs.
        wire.checksum =
            output::numeric_checksum(&serde_json::to_vec(&wire).map_err(|e| e.to_string())?)?;
        let bytes = serde_json::to_vec(&wire).map_err(|e| e.to_string())?;
        output::LoadedConditionalDeployment::load(&bytes)?;
        Ok(bytes)
    }
}
#[allow(clippy::too_many_arguments)]
fn make_identity(
    origin: &ConditionalRlOrigin,
    actor: &PublicPolicyModel,
    critic: &[f32],
    root: &str,
    parent: &str,
    count: u64,
    sampling_seed: u64,
    next: u64,
) -> Result<ConditionalMixedIdentity, String> {
    identity_from_checksum(
        origin.checksum(),
        actor,
        critic,
        root,
        parent,
        count,
        sampling_seed,
        next,
    )
}
#[allow(clippy::too_many_arguments)]
fn identity_from_checksum(
    origin: &str,
    actor: &PublicPolicyModel,
    critic: &[f32],
    root: &str,
    parent: &str,
    count: u64,
    sampling_seed: u64,
    next: u64,
) -> Result<ConditionalMixedIdentity, String> {
    actor.validate()?;
    if critic.len() != crate::public_state_critic::PARAMETER_COUNT
        || critic.iter().any(|x| !x.is_finite())
        || count > MAX_ATTEMPTS
    {
        return Err("Conditional joint identity parameter/count bound".into());
    }
    let bits = |values: &[f32]| values.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
    let mut identity = ConditionalMixedIdentity {
        policy_version: POLICY_VERSION.into(),
        input_contract: INPUT_CONTRACT.into(),
        origin_checksum: origin.into(),
        root_identity_checksum: root.into(),
        parent_identity_checksum: parent.into(),
        actor_checksum: hash(
            b"tzolkin-conditional-current-actor-bits-v1\0",
            &bits(actor.parameters()),
        )?,
        critic_checksum: hash(
            b"tzolkin-conditional-current-critic-bits-v1\0",
            &bits(critic),
        )?,
        mixed_update_count: count,
        sampling_seed,
        next_episode_ordinal: next,
        game_options: GameOptions::default(),
        uxmal_corn_limit: false,
        source_qualified: false,
        verified_native_policy: false,
        checksum: String::new(),
    };
    identity.checksum = hash(b"tzolkin-conditional-current-identity-v1\0", &identity)?;
    Ok(identity)
}
pub(crate) fn checked_tip_identity(
    parent: &ConditionalMixedIdentity,
    actor: &[f32],
    critic: &[f32],
    next: u64,
    updated: bool,
) -> Result<ConditionalMixedIdentity, String> {
    let model = PublicPolicyModel {
        parameters: actor.to_vec(),
    };
    identity_from_checksum(
        &parent.origin_checksum,
        &model,
        critic,
        &parent.root_identity_checksum,
        parent.checksum(),
        parent.mixed_update_count + u64::from(updated),
        parent.sampling_seed,
        next,
    )
}
pub(crate) fn check_identity_bits(
    identity: &ConditionalMixedIdentity,
    actor: &[f32],
    critic: &[f32],
) -> Result<(), String> {
    let model = PublicPolicyModel {
        parameters: actor.to_vec(),
    };
    if !identity.compatible()
        || identity_from_checksum(
            &identity.origin_checksum,
            &model,
            critic,
            &identity.root_identity_checksum,
            &identity.parent_identity_checksum,
            identity.mixed_update_count,
            identity.sampling_seed,
            identity.next_episode_ordinal,
        )? != *identity
    {
        return Err("Conditional identity/actor/critic bits differ".into());
    }
    Ok(())
}
pub struct ConditionalMixedPolicyHandle<'a> {
    owner: &'a ConditionalMixedOwner,
    kernel: ResolvedKernel,
}
impl ConditionalMixedPolicyHandle<'_> {
    pub fn identity(&self) -> &ConditionalMixedIdentity {
        &self.owner.identity
    }
    pub(crate) fn sampling_prefix_matches(&self, i: SamplingStreamIdentity) -> bool {
        i.sampling_seed().value() == self.owner.identity.sampling_seed
            && i.replicate_ordinal() == 0
            && i.episode_ordinal() >= self.owner.identity.next_episode_ordinal
            && i.episode_ordinal() < self.owner.identity.next_episode_ordinal.saturating_add(4)
    }
    pub fn distribution(
        &self,
        observation: &Observation,
    ) -> Result<ConditionalMixedDistribution, String> {
        let numeric = self.numeric_distribution(observation)?;
        Ok(ConditionalMixedDistribution {
            identity: self.owner.identity.clone(),
            logits: numeric.logits,
            probabilities: numeric.probabilities,
        })
    }
    pub(crate) fn numeric_distribution(
        &self,
        observation: &Observation,
    ) -> Result<tzolkin_inference::policy::NumericPolicyDistribution, String> {
        validate_contract(observation)?;
        let encoder = FeatureEncoder::new_public(observation)?;
        let rows = (0..observation.legal_actions.len())
            .map(|i| encoder.encode_legal_tagged(i))
            .collect::<Result<Vec<_>, _>>()?;
        predict_numeric(&self.owner.actor, self.kernel, &rows)
    }
}
pub struct ConditionalMixedDistribution {
    identity: ConditionalMixedIdentity,
    logits: Vec<f32>,
    probabilities: Vec<f32>,
}
impl ConditionalMixedDistribution {
    pub fn identity(&self) -> &ConditionalMixedIdentity {
        &self.identity
    }
    pub fn logits(&self) -> &[f32] {
        &self.logits
    }
    pub fn probabilities(&self) -> &[f32] {
        &self.probabilities
    }
    pub fn value_validity(&self) -> ValueValidity {
        ValueValidity::UnavailablePolicyOnly
    }
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub(crate) struct ConditionalMixedSource {
    parent: ConditionalMixedIdentity,
    learner_seat: usize,
    opponents: Vec<Option<serde_json::Value>>,
    roster_checksum: String,
    pool_checksum: String,
    external_exclusions: serde_json::Value,
    human_table_reservations: Vec<TypedHumanTableReservation>,
    uxmal_corn_limit: bool,
}
impl ConditionalMixedSource {
    pub(crate) fn require_handle(
        &self,
        handle: &ConditionalMixedPolicyHandle<'_>,
        roster: &ConditionalFrozenMixRoster<'_>,
    ) -> Result<(), String> {
        if self.parent != *handle.identity()
            || !self.parent.compatible()
            || self.uxmal_corn_limit
            || self.learner_seat != roster.learner
            || self.roster_checksum != roster.checksum
            || self.opponents != roster.descriptions
        {
            return Err(
                "Conditional source differs from actual parent/options/frozen roster".into(),
            );
        }
        Ok(())
    }
}
fn header(
    config: &NativeStochasticConfig,
    handle: &ConditionalMixedPolicyHandle<'_>,
    roster: &ConditionalFrozenMixRoster<'_>,
    pool: &str,
    external: &ConditionalNativeExclusions,
) -> Result<Header<ConditionalMixedSource>, String> {
    let family = NativeSeedFamily::train(config.environment_seed())?;
    if config.players() != roster.opponents.len()
        || !handle.sampling_prefix_matches(*config.sampling_identity())
        || external.families.contains(family.digest())
        || roster.contains_family(family.digest())
    {
        return Err("Conditional native family/roster/RNG prefix differs before work".into());
    }
    let human_table_reservations = handle
        .owner
        .origin
        .table_reservations()
        .iter()
        .map(|(table, split)| TypedHumanTableReservation {
            table_id: table.clone(),
            split: *split,
        })
        .collect();
    Ok(Header {
        source_kind: SOURCE_KIND.into(),
        rules_version: crate::replay::RULES_VERSION,
        rules_baseline: crate::replay::RULES_BASELINE.into(),
        catalog_hash: crate::replay::catalog_hash(),
        observation_schema: OBSERVATION_SCHEMA,
        move_schema: MOVE_SCHEMA,
        config: Config::from_config(config),
        base_policy: ConditionalMixedSource {
            parent: handle.identity().clone(),
            learner_seat: roster.learner,
            opponents: roster.descriptions.clone(),
            roster_checksum: roster.checksum.clone(),
            pool_checksum: pool.into(),
            external_exclusions: serde_json::to_value(external.descriptor())
                .map_err(|e| e.to_string())?,
            human_table_reservations,
            uxmal_corn_limit: false,
        },
        sampling_version: SAMPLING_VERSION.into(),
        rng_version: RNG_VERSION.into(),
        denominator_bits: 53,
        uniform_mixture_bits: hex64(UNIFORM_MIXTURE.to_bits()),
        backend: "scalar".into(),
        numerical_target: numerical_target(),
        native_family_id: family.digest,
        raw_artifact_identity_available: true,
        producer_authenticated: false,
        independent_seed_origins_verified: false,
    })
}
struct ConditionalEpisode {
    record: Record<ConditionalMixedSource>,
    config: NativeStochasticConfig,
    indices: Vec<usize>,
    links: Vec<ActorLink>,
    rank: usize,
    winners: usize,
    checksum: String,
}
impl ConditionalEpisode {
    fn from_audited(
        record: Record<ConditionalMixedSource>,
        config: NativeStochasticConfig,
    ) -> Result<Self, String> {
        if record.schema != RECORD_SCHEMA
            || record.failure.is_some()
            || record.terminal.is_none()
            || record.header.base_policy.uxmal_corn_limit
        {
            return Err("Conditional episode requires complete fresh native audit".into());
        }
        let learner = record.header.base_policy.learner_seat;
        let (links, ranks, winners) = applied_mixed_plan(
            &record,
            &config,
            SAMPLING_VERSION,
            || Ok(record.header.base_policy.parent.checksum.clone()),
            learner,
        )?;
        let indices = record
            .callbacks
            .iter()
            .enumerate()
            .filter(|(_, c)| c.actor == learner)
            .map(|(i, _)| i)
            .collect::<Vec<_>>();
        if links.len() != indices.len() || indices.is_empty() {
            return Err("Conditional learner coverage".into());
        }
        let rank = ranks[learner];
        let checksum =
            canonical_record_checksum(&record, b"tzolkin-conditional-mixed-episode-v1\0")?;
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
    fn steps(&self) -> impl Iterator<Item = AppliedActorStep<'_>> {
        self.indices
            .iter()
            .zip(&self.links)
            .map(|(&i, l)| applied_step(&self.record.callbacks[i], l, self.rank, self.winners))
    }
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ConditionalMixedReport {
    schema: &'static str,
    artifact_schema: &'static str,
    input_contract: &'static str,
    parent: ConditionalMixedIdentity,
    proposed_update_count: u64,
    pool_checksum: String,
    families: [NativeSeedFamily; 4],
    episodes: [String; 4],
    counts: [usize; 3],
    actor_config: OneStepConfig,
    critic_config: ResidualFitConfig,
    deltas: Deltas,
    numeric: NumericReport,
    critic_changed_parameters: usize,
    mean_raw_residual_mse: f64,
    next_episode_ordinal: u64,
    attempted_whole_four: u64,
    credited_own_decisions: u64,
    source_qualified: bool,
    verified_native_policy: bool,
    durable_admission: bool,
    checksum: String,
}
impl ConditionalMixedReport {
    pub fn counts(&self) -> [usize; 3] {
        self.counts
    }
    pub fn checksum(&self) -> &str {
        &self.checksum
    }
    pub fn changed_parameters(&self) -> [usize; 2] {
        [
            self.numeric.changed_parameters,
            self.critic_changed_parameters,
        ]
    }
}
#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ConditionalMixedArtifact {
    schema: &'static str,
    identity: ConditionalMixedIdentity,
    report_checksum: String,
    actor_parameters: Vec<f32>,
    critic_parameters: Vec<f32>,
    checksum: String,
}
pub struct ConditionalMixedOutcome {
    report: ConditionalMixedReport,
    artifact: Option<ConditionalMixedArtifact>,
    records: [Vec<u8>; 4],
}
impl ConditionalMixedOutcome {
    pub fn report(&self) -> &ConditionalMixedReport {
        &self.report
    }
    pub fn artifact(&self) -> Option<&ConditionalMixedArtifact> {
        self.artifact.as_ref()
    }
    /// Canonical bytes of the four freshly audited distinct conditional records.
    pub fn records(&self) -> &[Vec<u8>; 4] {
        &self.records
    }
}

/// Local in-memory reservations are permanent while this handle exists. Dropping
/// it grants no resume/retry proof; durable preservation belongs to a later API.
pub struct ConditionalMixedSession {
    owner: ConditionalMixedOwner,
    external: ConditionalNativeExclusions,
    attempts: u64,
    next: u64,
    credited: u64,
    spent: BTreeSet<NativeSeedFamily>,
    stopped: bool,
}
impl ConditionalMixedSession {
    pub fn new(
        owner: ConditionalMixedOwner,
        external: ConditionalNativeExclusions,
    ) -> Result<Self, String> {
        owner.origin.recheck_sources()?;
        external.recheck()?;
        owner.handle()?;
        if owner.identity.mixed_update_count != 0 || owner.identity.next_episode_ordinal != 0 {
            return Err("Initial conditional in-memory session requires load-backed count0".into());
        }
        Ok(Self {
            owner,
            external,
            attempts: 0,
            next: 0,
            credited: 0,
            spent: BTreeSet::new(),
            stopped: false,
        })
    }
    pub fn owner(&self) -> &ConditionalMixedOwner {
        &self.owner
    }
    pub fn attempts(&self) -> u64 {
        self.attempts
    }
    pub fn next_episode_ordinal(&self) -> u64 {
        self.next
    }
    pub fn credited_own_decisions(&self) -> u64 {
        self.credited
    }
    pub fn consumed_families(&self) -> &BTreeSet<NativeSeedFamily> {
        &self.spent
    }
    pub fn stopped(&self) -> bool {
        self.stopped
    }
    /// One fixed-parent collect -> byte codec -> fresh audit -> joint numeric
    /// proposal -> atomic in-memory admission. No automatic loop or retry.
    pub fn update_whole_four(
        &mut self,
        seeds: [u32; 4],
        pool: &ConditionalFrozenPool,
        prepared_opponent: Option<&PreparedPublicPolicy>,
        limits: CollectionLimits,
        actor_config: OneStepConfig,
        critic_config: ResidualFitConfig,
    ) -> Result<ConditionalMixedOutcome, String> {
        if self.stopped {
            return Err("Conditional session stopped or reservation bound".into());
        }
        if let Err(e) = self.full_source_check() {
            self.stop();
            return Err(e);
        }
        let plan = self.plan(
            seeds,
            pool,
            prepared_opponent,
            limits,
            actor_config,
            critic_config,
        )?;
        self.reserve(&plan)?;
        let proposal = match self.stage(&plan, pool, prepared_opponent, None) {
            Ok(proposal) => proposal,
            Err(e) => {
                self.stop();
                return Err(e);
            }
        };
        if let Err(e) = self.full_source_check() {
            self.stop();
            return Err(e);
        }
        self.admit(proposal).inspect_err(|_| self.stopped = true)
    }
    pub(crate) fn full_source_check(&self) -> Result<(), String> {
        self.owner
            .origin
            .recheck_sources()
            .and_then(|()| self.external.recheck())
    }
    pub(crate) fn stop(&mut self) {
        self.stopped = true;
    }
    pub(crate) fn plan(
        &self,
        seeds: [u32; 4],
        pool: &ConditionalFrozenPool,
        prepared_opponent: Option<&PreparedPublicPolicy>,
        limits: CollectionLimits,
        actor_config: OneStepConfig,
        critic_config: ResidualFitConfig,
    ) -> Result<ConditionalReservation, String> {
        if self.stopped
            || self.attempts >= MAX_ATTEMPTS
            || self
                .spent
                .len()
                .checked_add(4)
                .is_none_or(|n| n > MAX_SPENT_FAMILIES)
        {
            return Err("Conditional session stopped or reservation bound".into());
        }
        actor_config.validate()?;
        let rosters = pool.freeze(prepared_opponent)?;
        let families = seeds
            .map(NativeSeedFamily::train)
            .into_iter()
            .collect::<Result<Vec<_>, _>>()?;
        if families
            .iter()
            .enumerate()
            .any(|(i, f)| families[..i].contains(f) || self.spent.contains(f))
        {
            return Err("Conditional whole-four repeats a spent native family".into());
        }
        let handle = self.owner.handle()?;
        let configs: [NativeStochasticConfig; 4] = seeds
            .iter()
            .enumerate()
            .map(|(i, seed)| {
                let players = if i < 2 { 3 } else { 4 };
                NativeStochasticConfig::new(
                    players,
                    *seed,
                    SamplingStreamIdentity::new(
                        SamplingSeed::new(self.owner.identity.sampling_seed),
                        self.next
                            .checked_add(i as u64)
                            .ok_or("Conditional ordinal overflow")?,
                        0,
                        players,
                    )?,
                    limits,
                )
            })
            .collect::<Result<Vec<_>, String>>()?
            .try_into()
            .map_err(|_| "Conditional config shape")?;
        let headers = (0..4)
            .map(|i| {
                header(
                    &configs[i],
                    &handle,
                    &rosters[i],
                    &pool.checksum,
                    &self.external,
                )
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(ConditionalReservation {
            parent: self.owner.identity.clone(),
            previous_next: self.next,
            previous_attempts: self.attempts,
            seeds,
            families: families
                .try_into()
                .map_err(|_| "Conditional family shape")?,
            configs,
            headers: headers.try_into().map_err(|_| "Conditional header shape")?,
            actor_config,
            critic_config,
        })
    }
    pub(crate) fn reserve(&mut self, plan: &ConditionalReservation) -> Result<(), String> {
        self.reserve_inner(plan, false)
    }
    pub(crate) fn reserve_after_publish_error(
        &mut self,
        plan: &ConditionalReservation,
    ) -> Result<(), String> {
        self.reserve_inner(plan, true)
    }
    fn reserve_inner(
        &mut self,
        plan: &ConditionalReservation,
        dead_commit: bool,
    ) -> Result<(), String> {
        if (!dead_commit && self.stopped)
            || plan.parent != self.owner.identity
            || plan.previous_next != self.next
            || plan.previous_attempts != self.attempts
            || plan.families.iter().any(|f| self.spent.contains(f))
        {
            return Err("Conditional reservation differs from live immediate parent".into());
        }
        self.next = self
            .next
            .checked_add(4)
            .ok_or("Conditional ordinal overflow")?;
        self.attempts += 1;
        self.spent.extend(plan.families.iter().cloned());
        Ok(())
    }
    pub(crate) fn proposed_identity(
        &self,
        proposal: &JointProposal,
    ) -> Result<ConditionalMixedIdentity, String> {
        match &proposal.updated {
            Some((_, _, artifact)) => Ok(artifact.identity.clone()),
            None => checked_tip_identity(
                &self.owner.identity,
                self.owner.actor.parameters(),
                &self.owner.critic,
                self.next,
                false,
            ),
        }
    }
    pub(crate) fn stage(
        &self,
        plan: &ConditionalReservation,
        pool: &ConditionalFrozenPool,
        prepared_opponent: Option<&PreparedPublicPolicy>,
        saved: Option<&[Vec<u8>; 4]>,
    ) -> Result<JointProposal, String> {
        if self.stopped
            || plan.parent != self.owner.identity
            || self.next != plan.previous_next + 4
            || self.attempts != plan.previous_attempts + 1
            || plan.families.iter().any(|f| !self.spent.contains(f))
        {
            return Err("Conditional stage has no exact reserved parent".into());
        }
        let handle = self.owner.handle()?;
        let rosters = pool.freeze(prepared_opponent)?;
        let mut episodes = Vec::new();
        for i in 0..4 {
            let actual_header = header(
                &plan.configs[i],
                &handle,
                &rosters[i],
                &pool.checksum,
                &self.external,
            )?;
            if actual_header != plan.headers[i] {
                return Err("Conditional frozen stage binding differs".into());
            }
            let raw = match saved {
                Some(records) => records[i].clone(),
                None => encode_typed_record(&run_conditional_mixed_shared(
                    &plan.configs[i],
                    &handle,
                    &rosters[i],
                    actual_header.clone(),
                    None,
                )?)?,
            };
            let expected: Record<ConditionalMixedSource> = decode_typed_record(&raw)?;
            if expected.schema != RECORD_SCHEMA
                || expected.header != actual_header
                || expected.training_admission != "unavailable-distinct-codec-required"
            {
                return Err("Conditional full source/header differs before audit".into());
            }
            let audited = run_conditional_mixed_shared(
                &plan.configs[i],
                &handle,
                &rosters[i],
                actual_header,
                Some(&expected),
            )?;
            episodes.push(ConditionalEpisode::from_audited(
                audited,
                plan.configs[i].clone(),
            )?);
        }
        prepare(
            &self.owner,
            &episodes,
            plan.families.clone(),
            pool,
            plan.actor_config,
            plan.critic_config,
            self.attempts,
            self.next,
            self.credited,
        )
    }
    pub(crate) fn admit(
        &mut self,
        proposal: JointProposal,
    ) -> Result<ConditionalMixedOutcome, String> {
        if proposal.parent != self.owner.identity
            || proposal.report.next_episode_ordinal != self.next
            || proposal.report.attempted_whole_four != self.attempts
            || proposal
                .report
                .families
                .iter()
                .any(|f| !self.spent.contains(f))
        {
            return Err(
                "Conditional staged joint proposal differs from reserved immediate parent".into(),
            );
        }
        let Some((actor, critic, artifact)) = proposal.updated else {
            // NoChange still consumes four ordinals; advance the non-weight identity.
            self.owner.identity = make_identity(
                &self.owner.origin,
                &self.owner.actor,
                &self.owner.critic,
                &self.owner.identity.root_identity_checksum,
                self.owner.identity.checksum(),
                self.owner.identity.mixed_update_count,
                self.owner.identity.sampling_seed,
                self.next,
            )?;
            return Ok(ConditionalMixedOutcome {
                report: proposal.report,
                artifact: None,
                records: proposal.records,
            });
        };
        actor.validate()?;
        let expected = make_identity(
            &self.owner.origin,
            &actor,
            &critic,
            &self.owner.identity.root_identity_checksum,
            self.owner.identity.checksum(),
            self.owner.identity.mixed_update_count + 1,
            self.owner.identity.sampling_seed,
            self.next,
        )?;
        if artifact.identity != expected
            || artifact.actor_parameters != actor.parameters()
            || artifact.critic_parameters != critic
            || artifact.report_checksum != proposal.report.checksum
        {
            return Err("Conditional joint actor/critic/identity admission differs".into());
        }
        let credited = self
            .credited
            .checked_add(proposal.report.counts[0] as u64)
            .ok_or("Conditional credit overflow")?;
        // All fallible checks precede this single joint state commit.
        (
            self.owner.actor,
            self.owner.critic,
            self.owner.identity,
            self.credited,
        ) = (actor, critic, expected, credited);
        Ok(ConditionalMixedOutcome {
            report: proposal.report,
            artifact: Some(artifact),
            records: proposal.records,
        })
    }
}
pub(crate) struct ConditionalReservation {
    parent: ConditionalMixedIdentity,
    previous_next: u64,
    previous_attempts: u64,
    seeds: [u32; 4],
    families: [NativeSeedFamily; 4],
    configs: [NativeStochasticConfig; 4],
    headers: [Header<ConditionalMixedSource>; 4],
    actor_config: OneStepConfig,
    critic_config: ResidualFitConfig,
}
impl ConditionalReservation {
    pub(crate) fn seeds(&self) -> [u32; 4] {
        self.seeds
    }
    pub(crate) fn families(&self) -> [String; 4] {
        self.families.each_ref().map(|f| f.digest.clone())
    }
    pub(crate) fn parent(&self) -> &ConditionalMixedIdentity {
        &self.parent
    }
}
pub(crate) struct JointProposal {
    parent: ConditionalMixedIdentity,
    report: ConditionalMixedReport,
    updated: Option<(PublicPolicyModel, Vec<f32>, ConditionalMixedArtifact)>,
    records: [Vec<u8>; 4],
}
impl JointProposal {
    pub(crate) fn payload(&self) -> Result<(Vec<u8>, Option<Vec<u8>>), String> {
        let report = crate::policy_training::bounded_json(&self.report)?;
        let artifact = self
            .updated
            .as_ref()
            .map(|(_, _, artifact)| crate::policy_training::bounded_json(artifact))
            .transpose()?;
        Ok((report, artifact))
    }
    pub(crate) fn records(&self) -> &[Vec<u8>; 4] {
        &self.records
    }
    pub(crate) fn is_updated(&self) -> bool {
        self.updated.is_some()
    }
    pub(crate) fn own_credit(&self) -> u64 {
        self.report.credited_own_decisions
    }
}
#[cfg(test)]
pub(crate) fn test_short_stage(
    session: &ConditionalMixedSession,
    plan: &ConditionalReservation,
    pool: &ConditionalFrozenPool,
    winning: bool,
    saved: Option<&[Vec<u8>; 4]>,
) -> Result<JointProposal, String> {
    let handle = session.owner.handle()?;
    let rosters = pool.freeze(None)?;
    let mut episodes = Vec::new();
    for (i, roster) in rosters.iter().enumerate() {
        let config = &plan.configs[i];
        let mut start = tests::terminal_start(config.players(), plan.seeds[i]);
        if !winning {
            start.players[0].score = 0.0;
            for player in start.players.iter_mut().skip(1) {
                player.score = 100.0;
            }
        }
        let expected = match saved {
            Some(raw) => Some(decode_typed_record::<ConditionalMixedSource>(&raw[i])?),
            None => None,
        };
        let record = crate::public_stochastic_native::conditional_test_start(
            config,
            &handle,
            roster,
            plan.headers[i].clone(),
            expected.as_ref(),
            start.clone(),
        )?;
        let raw = encode_typed_record(&record)?;
        let expected = decode_typed_record::<ConditionalMixedSource>(&raw)?;
        let audited = crate::public_stochastic_native::conditional_test_start(
            config,
            &handle,
            roster,
            plan.headers[i].clone(),
            Some(&expected),
            start,
        )?;
        episodes.push(ConditionalEpisode::from_audited(audited, config.clone())?);
    }
    prepare(
        &session.owner,
        &episodes,
        plan.families.clone(),
        pool,
        plan.actor_config,
        plan.critic_config,
        session.attempts,
        session.next,
        session.credited,
    )
}
#[allow(clippy::too_many_arguments)]
fn prepare(
    owner: &ConditionalMixedOwner,
    episodes: &[ConditionalEpisode],
    families: [NativeSeedFamily; 4],
    pool: &ConditionalFrozenPool,
    actor_config: OneStepConfig,
    critic_config: ResidualFitConfig,
    attempts: u64,
    next: u64,
    credited: u64,
) -> Result<JointProposal, String> {
    if episodes.len() != 4
        || episodes.iter().enumerate().any(|(i, e)| {
            e.record.header.base_policy.parent != owner.identity
                || e.config.players() != if i < 2 { 3 } else { 4 }
                || e.record.header.native_family_id != families[i].digest
                || e.record.header.numerical_target != numerical_target()
        })
    {
        return Err("Conditional sealed cohort immediate parent/config/family differs".into());
    }
    let mut counts = [0usize; 3];
    for e in episodes {
        for s in e.steps() {
            for (total, n) in counts.iter_mut().zip([
                1,
                s.observation().legal_actions.len(),
                usize::from(s.observation().legal_actions.len() == 1),
            ]) {
                *total = total
                    .checked_add(n)
                    .ok_or("Conditional cohort count overflow")?;
            }
        }
    }
    let members: [MixedNumericMember<'_>; 4] = episodes
        .iter()
        .map(|e| MixedNumericMember {
            learner: e.record.header.base_policy.learner_seat,
            players: e.config.players(),
            coefficient: 0.25,
            steps: e.steps().collect(),
        })
        .collect::<Vec<_>>()
        .try_into()
        .map_err(|_| "Conditional numeric whole-four shape")?;
    let (deltas, actor, critic) = prepare_numeric_timed(
        &owner.actor,
        &owner.critic,
        &members,
        counts,
        &actor_config,
        &critic_config,
        &mut Untimed,
    )?;
    let episode_checksums: [String; 4] = episodes
        .iter()
        .map(|e| e.checksum.clone())
        .collect::<Vec<_>>()
        .try_into()
        .map_err(|_| "Conditional episode shape")?;
    crate::public_policy_update::validate_conditional_joint_report(
        &counts,
        &deltas,
        &actor.numeric,
        &actor_config,
        (
            [
                owner.identity.checksum(),
                owner.origin.checksum(),
                &pool.checksum,
                owner.identity.root_identity_checksum.as_str(),
            ],
            &episode_checksums,
        ),
        &families.each_ref().map(|f| f.digest.clone()),
    )?;
    let changed = actor.numeric.changed_parameters != 0 || critic.changed_parameters != 0;
    let proposed_count = owner.identity.mixed_update_count + u64::from(changed);
    let credit = if changed {
        credited
            .checked_add(counts[0] as u64)
            .ok_or("Conditional report credit overflow")?
    } else {
        credited
    };
    let mut report = ConditionalMixedReport {
        schema: REPORT_SCHEMA,
        artifact_schema: ARTIFACT_SCHEMA,
        input_contract: INPUT_CONTRACT,
        parent: owner.identity.clone(),
        proposed_update_count: proposed_count,
        pool_checksum: pool.checksum.clone(),
        families,
        episodes: episode_checksums,
        counts,
        actor_config,
        critic_config,
        deltas,
        numeric: actor.numeric,
        critic_changed_parameters: critic.changed_parameters,
        mean_raw_residual_mse: critic.mean_loss,
        next_episode_ordinal: next,
        attempted_whole_four: attempts,
        credited_own_decisions: credit,
        source_qualified: false,
        verified_native_policy: false,
        durable_admission: false,
        checksum: String::new(),
    };
    report.checksum = hash(b"tzolkin-conditional-joint-report-v1\0", &report)?;
    let updated = if changed {
        let model = PublicPolicyModel {
            parameters: actor.parameters,
        };
        let actor_cap =
            serde_json::to_value(actor_config).map_err(|e| e.to_string())?["maxAbsActualDelta"]
                .as_f64()
                .ok_or("Actor cap")?;
        let critic_cap =
            serde_json::to_value(critic_config).map_err(|e| e.to_string())?["maxAbsActualDelta"]
                .as_f64()
                .ok_or("Critic cap")?;
        if rounded_delta(owner.actor.parameters(), model.parameters(), actor_cap)?.0
            != report.numeric.changed_parameters
            || rounded_delta(&owner.critic, &critic.parameters, critic_cap)?.0
                != report.critic_changed_parameters
        {
            return Err("Conditional rounded joint delta metadata differs".into());
        }
        let identity = make_identity(
            &owner.origin,
            &model,
            &critic.parameters,
            &owner.identity.root_identity_checksum,
            owner.identity.checksum(),
            proposed_count,
            owner.identity.sampling_seed,
            next,
        )?;
        let mut artifact = ConditionalMixedArtifact {
            schema: ARTIFACT_SCHEMA,
            identity,
            report_checksum: report.checksum.clone(),
            actor_parameters: model.parameters().to_vec(),
            critic_parameters: critic.parameters.clone(),
            checksum: String::new(),
        };
        artifact.checksum = hash(b"tzolkin-conditional-joint-artifact-v1\0", &artifact)?;
        Some((model, critic.parameters, artifact))
    } else {
        None
    };
    Ok(JointProposal {
        parent: owner.identity.clone(),
        report,
        updated,
        records: episodes
            .iter()
            .map(|e| encode_typed_record(&e.record))
            .collect::<Result<Vec<_>, _>>()?
            .try_into()
            .map_err(|_| "Conditional source record shape")?,
    })
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::human_policy_dataset::ConditionalHumanPolicyDataset;
    use crate::human_policy_training::{
        PreparedConditionalHumanPolicy, train_conditional_datasets,
    };
    use crate::policy_training::PolicyBcConfig;
    use crate::public_stochastic::ConditionalMixedStochasticSession;
    use serde_json::json;
    mod temp_root {
        include!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/support/temp_root.rs"
        ));
    }
    struct Files(std::path::PathBuf);
    impl Drop for Files {
        fn drop(&mut self) {
            let _ = fs::remove_file(self.0.join("exclusions.json"));
            let _ = fs::remove_dir(&self.0);
        }
    }
    fn external(files: &Files) -> ConditionalNativeExclusions {
        let families = vec![seed_family_id(42)];
        let raw = serde_json::to_vec(
            &json!({"schema":"tzolkin-conditional-native-exclusions-v1","families":families}),
        )
        .unwrap();
        let path = files.0.join("exclusions.json");
        fs::write(&path, &raw).unwrap();
        let pin = ContentPin {
            path: path.to_str().unwrap().into(),
            bytes: raw.len() as u64,
            sha256: crate::model::digest(&raw),
        };
        let digest = crate::model::digest(format!("{}\n", families[0]).as_bytes());
        assert!(ConditionalNativeExclusions::load(&path, &pin, 2, &digest).is_err());
        let checked = ConditionalNativeExclusions::load(&path, &pin, 1, &digest).unwrap();
        fs::write(&path, [raw.as_slice(), b" "].concat()).unwrap();
        assert!(checked.recheck().is_err());
        fs::write(&path, raw).unwrap();
        checked
    }
    fn pool() -> serde_json::Value {
        let opponent = json!({"kind":"heuristic"});
        json!({"schema":"tzolkin-conditional-human-mixed-frozen-pool-v1",
            "weights":crate::policy::HeuristicWeights::default(),
            "rosters":[[null,opponent,opponent],[null,opponent,opponent],
                [null,opponent,opponent,opponent],[null,opponent,opponent,opponent]]})
    }
    pub(crate) fn owner() -> ConditionalMixedOwner {
        let datasets = [
            ConditionalHumanPolicyDataset::training_fixture("123", DatasetSplit::Train, 0),
            ConditionalHumanPolicyDataset::training_fixture("124", DatasetSplit::Validation, 1),
        ];
        let result = train_conditional_datasets(
            &datasets,
            &PolicyBcConfig {
                epochs: 1,
                batch_size: 1,
                learning_rate: 0.001,
                seed: 7,
            },
            None,
        )
        .unwrap();
        let prepared = PreparedConditionalHumanPolicy::new(result.checkpoint, &datasets).unwrap();
        assert!(ConditionalRlOwner::from_prepared(&prepared, 11, SamplingSeed::new(17)).is_err());
        ConditionalMixedOwner::from_count_zero(ConditionalRlOwner::synthetic(&prepared).unwrap())
            .unwrap()
    }
    pub(crate) fn updated_owner() -> ConditionalMixedOwner {
        let files = Files(temp_root::create("conditional-arena-owner").unwrap());
        let mut session = ConditionalMixedSession::new(owner(), external(&files)).unwrap();
        let pool = ConditionalFrozenPool::from_json(&serde_json::to_vec(&pool()).unwrap()).unwrap();
        let plan = session
            .plan(
                seeds(),
                &pool,
                None,
                CollectionLimits::new(64, 262144, 8 * 1024 * 1024, 1024 * 1024).unwrap(),
                OneStepConfig::default(),
                ResidualFitConfig::new(0.001, 0.01, 16_000).unwrap(),
            )
            .unwrap();
        session.reserve(&plan).unwrap();
        let staged = test_short_stage(&session, &plan, &pool, true, None).unwrap();
        assert!(session.admit(staged).unwrap().artifact().is_some());
        assert_eq!(session.owner.identity().mixed_update_count(), 1);
        session.owner
    }
    fn seeds() -> [u32; 4] {
        (0..32)
            .filter(|s| NativeSeedFamily::train(*s).is_ok())
            .take(4)
            .collect::<Vec<_>>()
            .try_into()
            .unwrap()
    }
    #[test]
    fn conditional_pool_and_failed_prefix_keep_typed_reservations_and_stop() {
        for forbidden in ["search", "parentArgmax"] {
            let mut wire = pool();
            wire["rosters"][0][1] = json!({"kind":forbidden});
            assert!(ConditionalFrozenPool::from_json(&serde_json::to_vec(&wire).unwrap()).is_err());
        }
        let files = Files(temp_root::create("conditional-mixed-bounds").unwrap());
        let mut session = ConditionalMixedSession::new(owner(), external(&files)).unwrap();
        let pool = ConditionalFrozenPool::from_json(&serde_json::to_vec(&pool()).unwrap()).unwrap();
        let before = session.owner().identity().checksum().to_owned();
        let seeds = seeds();
        let actor = OneStepConfig::default();
        let critic = ResidualFitConfig::new(0.001, 0.01, 16_000).unwrap();
        assert!(
            session
                .update_whole_four(
                    [seeds[0]; 4],
                    &pool,
                    None,
                    CollectionLimits::default(),
                    actor,
                    critic
                )
                .is_err()
        );
        assert_eq!(session.attempts(), 0);
        let tiny = CollectionLimits::new(1, 4096, 1024 * 1024, 64 * 1024).unwrap();
        assert!(
            session
                .update_whole_four(seeds, &pool, None, tiny, actor, critic)
                .is_err()
        );
        assert_eq!(
            (
                session.attempts(),
                session.next_episode_ordinal(),
                session.consumed_families().len()
            ),
            (1, 4, 4)
        );
        assert!(session.stopped());
        assert_eq!(session.credited_own_decisions(), 0);
        assert_eq!(session.owner().identity().checksum(), before);
        assert!(
            session
                .update_whole_four(seeds, &pool, None, tiny, actor, critic)
                .is_err()
        );
        assert_eq!(session.attempts(), 1);
    }
    pub(crate) fn terminal_start(players: usize, seed: u32) -> tzolkin_core::GameState {
        let mut state = tzolkin_core::create_game_with_options(
            (0..players).map(|p| format!("P{p}")).collect(),
            seed,
            GameOptions::default(),
        )
        .unwrap();
        while state.phase == tzolkin_core::Phase::Setup {
            let o = tzolkin_core::observation::observe(&state, state.current_player).unwrap();
            state = tzolkin_core::apply_move(&state, o.legal_actions[0].r#move.clone()).unwrap();
        }
        // Synthetic last round only; no full-game fixture or terminal score injection.
        state.round = 27;
        state.age = 2;
        let mut age_two = std::mem::take(&mut state.age2_deck);
        state.building_deck = age_two.split_off(6);
        state.buildings = age_two;
        state.food_days = vec![8, 14, 21, 27];
        state.first_player = 0;
        state.turn_order = (0..players).collect();
        state.turn_index = 0;
        state.current_player = 0;
        state.turn.mode = tzolkin_core::TurnMode::Remove;
        state.turn.count = 1;
        state.players[0].score = 100.0;
        assert!(tzolkin_core::validation::validate_game_state(
            &serde_json::to_value(&state).unwrap()
        ));
        state
    }
    #[test]
    fn conditional_short_native_audit_joint_admission_and_updated_prefix() {
        let files = Files(temp_root::create("conditional-mixed-joint").unwrap());
        let mut session = ConditionalMixedSession::new(owner(), external(&files)).unwrap();
        let pool = ConditionalFrozenPool::from_json(&serde_json::to_vec(&pool()).unwrap()).unwrap();
        let rosters = pool.freeze(None).unwrap();
        let families = seeds().map(|s| NativeSeedFamily::train(s).unwrap());
        let before = session.owner.identity.clone();
        let before_actor = session.owner.actor.parameters().to_vec();
        let handle = session.owner.handle().unwrap();
        let limits = CollectionLimits::new(64, 262144, 8 * 1024 * 1024, 1024 * 1024).unwrap();
        let mut episodes = Vec::new();
        for (i, roster) in rosters.iter().enumerate() {
            let players = if i < 2 { 3 } else { 4 };
            let seed = seeds()[i];
            let config = NativeStochasticConfig::new(
                players,
                seed,
                SamplingStreamIdentity::new(SamplingSeed::new(17), i as u64, 0, players).unwrap(),
                limits,
            )
            .unwrap();
            let h = header(&config, &handle, roster, &pool.checksum, &session.external).unwrap();
            let start = terminal_start(players, seed);
            let record = crate::public_stochastic_native::conditional_test_start(
                &config,
                &handle,
                roster,
                h.clone(),
                None,
                start.clone(),
            )
            .unwrap();
            assert!(
                record.terminal.is_some() && record.failure.is_none(),
                "short native fixture: {:?}",
                record.failure
            );
            let raw = encode_typed_record(&record).unwrap();
            let expected: Record<ConditionalMixedSource> = decode_typed_record(&raw).unwrap();
            let audited = crate::public_stochastic_native::conditional_test_start(
                &config,
                &handle,
                roster,
                h.clone(),
                Some(&expected),
                start.clone(),
            )
            .unwrap();
            let mut changed = expected;
            changed.header.base_policy.uxmal_corn_limit = true;
            assert!(
                crate::public_stochastic_native::conditional_test_start(
                    &config,
                    &handle,
                    roster,
                    h,
                    Some(&changed),
                    start
                )
                .is_err()
            );
            episodes.push(ConditionalEpisode::from_audited(audited, config).unwrap());
        }
        session.attempts = 1;
        session.next = 4;
        session.spent.extend(families.iter().cloned());
        let actor = OneStepConfig::default();
        let critic = ResidualFitConfig::new(0.001, 0.01, 16_000).unwrap();
        let mut bad = prepare(
            &session.owner,
            &episodes,
            families.clone(),
            &pool,
            actor,
            critic,
            1,
            4,
            0,
        )
        .unwrap();
        bad.updated.as_mut().unwrap().2.actor_parameters[0] = f32::NAN;
        assert!(session.admit(bad).is_err());
        assert_eq!(session.owner.identity, before);
        assert_eq!(session.owner.actor.parameters(), before_actor);
        assert_eq!(session.credited, 0);
        let good = prepare(
            &session.owner,
            &episodes,
            families,
            &pool,
            actor,
            critic,
            1,
            4,
            0,
        )
        .unwrap();
        let outcome = session.admit(good).unwrap();
        assert!(outcome.artifact().is_some());
        assert_eq!(session.owner.identity.mixed_update_count(), 1);
        assert_eq!(session.credited, outcome.report.counts[0] as u64);
        assert_eq!(session.next, 4);
        assert_eq!(
            session.owner.identity.parent_identity_checksum,
            before.checksum
        );
        let updated = session.owner.handle().unwrap();
        let state = tzolkin_core::create_game_with_options(
            vec!["A".into(), "B".into(), "C".into()],
            42,
            GameOptions::default(),
        )
        .unwrap();
        let o = tzolkin_core::observation::observe(&state, state.current_player).unwrap();
        assert!(
            ConditionalMixedStochasticSession::new(
                &updated,
                SamplingStreamIdentity::new(SamplingSeed::new(17), 0, 0, 3).unwrap()
            )
            .is_err()
        );
        let mut sampler = ConditionalMixedStochasticSession::new(
            &updated,
            SamplingStreamIdentity::new(SamplingSeed::new(17), 4, 0, 3).unwrap(),
        )
        .unwrap();
        assert_eq!(
            sampler.sample(&o).unwrap().decision().policy_version,
            SAMPLING_VERSION
        );
        assert!(!updated.identity().source_qualified && !updated.identity().verified_native_policy);
    }
}
