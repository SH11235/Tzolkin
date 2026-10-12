//! Conditional Human-origin actor inference only. Checksums establish content
//! consistency, not source authenticity, critic bits, journal custody or OS exit.
//! Loading never constructs a learning owner or native-qualified policy.
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use tzolkin_core::observation::{MOVE_SCHEMA, OBSERVATION_SCHEMA, Observation};

use crate::deployment::{self, DeploymentDecision};
use crate::features::{FEATURE_COUNT, FeatureEncoder, PUBLIC_FEATURE_SCHEMA};
use crate::kernel::{Kernel, ResolvedKernel};
use crate::policy::{self, BorrowedPolicy, HIDDEN, NumericPolicyDistribution, PARAMETER_COUNT};

pub const DEPLOYMENT_SCHEMA: &str = "tzolkin-conditional-human-current-numeric-inference-v1";
pub const MAX_DEPLOYMENT_BYTES: usize = 1024 * 1024;
pub const POLICY_VERSION: &str = "conditional-human-current-mixed-scalar-v1";
pub const GUARDED_POLICY_VERSION: &str = "numeric-conditional-human-trade-guard-v1";
const INPUT_CONTRACT: &str =
    "conditional-human-origin-base-default-3-4p-public512-native-setup-playing-v1";
const CHECKSUM_DOMAIN: &[u8] = b"tzolkin-conditional-human-current-numeric-inference-v1\0";

fn digest(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}
fn hash(domain: &[u8], value: &impl Serialize) -> Result<String, String> {
    let mut h = Sha256::new();
    h.update(domain);
    h.update(serde_json::to_vec(value).map_err(|e| e.to_string())?);
    Ok(format!("{:x}", h.finalize()))
}
fn table_id(value: &str) -> bool {
    !value.is_empty()
        && value.len() <= 16
        && !value.starts_with('0')
        && value.bytes().all(|b| b.is_ascii_digit())
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
enum Split {
    Train,
    Validation,
    Test,
}
impl Split {
    fn index(self) -> usize {
        match self {
            Self::Train => 0,
            Self::Validation => 1,
            Self::Test => 2,
        }
    }
}
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct Table {
    table_id: String,
    family_id: String,
    split: Split,
    manifest_sha256: String,
    sealed_report_sha256: String,
    samples: usize,
    candidates: usize,
}
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct Dataset {
    task: String,
    input_contract: String,
    feature_schema: u32,
    feature_count: usize,
    rules_version: u32,
    catalog_hash: String,
    source_authenticated: bool,
    source_qualified: bool,
    verified_native_policy: bool,
    tables: Vec<Table>,
    samples: [usize; 3],
    candidates: [usize; 3],
    fingerprint: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    label_scope: Option<String>,
}
impl Dataset {
    fn validate(&self, reservations: &BTreeMap<String, Split>) -> Result<(), String> {
        if self.task != "ConditionalBasicHumanPolicy"
            || self.input_contract != "conditional-basic-unlimited-3-4p-public512-playing-v1"
            || self.feature_schema != PUBLIC_FEATURE_SCHEMA
            || self.feature_count != FEATURE_COUNT
            || self.rules_version != deployment::RULES_VERSION
            || self.catalog_hash != deployment::catalog_hash()
            || self.source_authenticated
            || self.source_qualified
            || self.verified_native_policy
            || self.tables.is_empty()
            || self.tables.len() > 128
            || self.label_scope.as_deref().is_some_and(|scope| {
                !matches!(
                    scope,
                    "PrimitivesAndRemovalPairedCurrentGearActionsV2"
                        | "PrimitivesRemovalActionAndObservedUncappedPaidTechnologyV3"
                )
            })
            || reservations.is_empty()
            || reservations.len() > 4096
            || reservations.keys().any(|id| !table_id(id))
        {
            return Err("Invalid conditional inference dataset/reservation metadata".into());
        }
        let mut counts = ([0usize; 3], [0usize; 3]);
        let mut previous: Option<&str> = None;
        let mut families = BTreeSet::new();
        for table in &self.tables {
            let i = table.split.index();
            let mut h = Sha256::new();
            h.update(b"tzolkin-human-initial-table-family-v1\0");
            h.update(table.table_id.as_bytes());
            if !table_id(&table.table_id)
                || i == 2
                || table.samples == 0
                || table.samples > 65_536
                || table.candidates < table.samples
                || table.candidates > 262_144
                || !digest(&table.manifest_sha256)
                || !digest(&table.sealed_report_sha256)
                || format!("{:x}", h.finalize()) != table.family_id
                || !families.insert(&table.family_id)
                || previous.is_some_and(|p| p >= table.table_id.as_str())
                || reservations.get(&table.table_id) != Some(&table.split)
            {
                return Err("Conditional inference table/split/count mismatch".into());
            }
            previous = Some(&table.table_id);
            counts.0[i] = counts.0[i]
                .checked_add(table.samples)
                .ok_or("Sample overflow")?;
            counts.1[i] = counts.1[i]
                .checked_add(table.candidates)
                .ok_or("Candidate overflow")?;
        }
        let mut unhashed = self.clone();
        unhashed.fingerprint.clear();
        if counts != (self.samples, self.candidates)
            || self.samples[0] == 0
            || self.samples[1] == 0
            || self.samples[2] != 0
            || self.candidates[2] != 0
            || self.samples.iter().sum::<usize>() > 65_536
            || self.candidates.iter().sum::<usize>() > 262_144
            || hash(b"", &unhashed)? != self.fingerprint
        {
            return Err("Conditional inference dataset fingerprint mismatch".into());
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct BasicOptions {
    additional_buildings: bool,
    tribes: bool,
    prophecies: bool,
    quick_actions: bool,
}
impl BasicOptions {
    fn basic(&self) -> bool {
        !self.additional_buildings && !self.tribes && !self.prophecies && !self.quick_actions
    }
}
/// Closed claimed joint identity. Only actor bits are present and independently
/// validated. The critic checksum, ancestry and training history remain claims.
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct ConditionalCurrentIdentity {
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
    game_options: BasicOptions,
    uxmal_corn_limit: bool,
    source_qualified: bool,
    verified_native_policy: bool,
    checksum: String,
}
impl ConditionalCurrentIdentity {
    pub fn checksum(&self) -> &str {
        &self.checksum
    }
    pub fn mixed_update_count(&self) -> u64 {
        self.mixed_update_count
    }
    pub fn actor_checksum(&self) -> &str {
        &self.actor_checksum
    }
    /// Checksum-bound metadata only: this loader receives no critic weights.
    pub fn claimed_critic_checksum(&self) -> &str {
        &self.critic_checksum
    }
    fn validate(&self, values: &[f32]) -> Result<(), String> {
        let mut unhashed = self.clone();
        unhashed.checksum.clear();
        let bits = values.iter().map(|v| v.to_bits()).collect::<Vec<_>>();
        if self.policy_version != POLICY_VERSION
            || self.input_contract != INPUT_CONTRACT
            || !(1..=64).contains(&self.mixed_update_count)
            || !self.game_options.basic()
            || self.uxmal_corn_limit
            || self.source_qualified
            || self.verified_native_policy
            || [
                &self.origin_checksum,
                &self.root_identity_checksum,
                &self.parent_identity_checksum,
                &self.critic_checksum,
                &self.checksum,
            ]
            .iter()
            .any(|v| !digest(v))
            || hash(b"tzolkin-conditional-current-actor-bits-v1\0", &bits)? != self.actor_checksum
            || hash(b"tzolkin-conditional-current-identity-v1\0", &unhashed)? != self.checksum
        {
            return Err("Conditional inference current identity/actor bits mismatch".into());
        }
        Ok(())
    }
}
#[derive(Deserialize, Serialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct Journal {
    head_sha256: String,
    head_bytes: u64,
    attempts: u64,
    credited_own_decisions: u64,
    spent_native_families: Vec<String>,
    external_native_count: usize,
    external_native_digest: String,
    external_native_sha256: String,
}
#[derive(Deserialize, Serialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct Source {
    task: String,
    origin_checksum: String,
    checkpoint_checksum: String,
    model_checksum: String,
    dataset: Dataset,
    #[serde(deserialize_with = "reservations")]
    table_reservations: BTreeMap<String, Split>,
    current: ConditionalCurrentIdentity,
    journal: Journal,
    source_authenticated: bool,
    source_qualified: bool,
    verified_native_policy: bool,
    close_scope: String,
}
fn reservations<'de, D: serde::Deserializer<'de>>(
    decoder: D,
) -> Result<BTreeMap<String, Split>, D::Error> {
    struct Visitor;
    impl<'de> serde::de::Visitor<'de> for Visitor {
        type Value = BTreeMap<String, Split>;
        fn expecting(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.write_str("bounded unique Human table reservations")
        }
        fn visit_map<M: serde::de::MapAccess<'de>>(
            self,
            mut map: M,
        ) -> Result<Self::Value, M::Error> {
            let mut result = BTreeMap::new();
            while let Some((id, split)) = map.next_entry::<String, Split>()? {
                if result.len() >= 4096 || !table_id(&id) || result.insert(id, split).is_some() {
                    return Err(serde::de::Error::custom(
                        "Duplicate/invalid/bounded table reservation",
                    ));
                }
            }
            Ok(result)
        }
    }
    decoder.deserialize_map(Visitor)
}
impl Source {
    fn validate(&self, values: &[f32]) -> Result<(), String> {
        self.dataset.validate(&self.table_reservations)?;
        self.current.validate(values)?;
        let j = &self.journal;
        if self.task != "ConditionalHumanCurrentMixedPolicyInference"
            || self.close_scope != "returnedLibraryReadyOnly"
            || self.source_authenticated
            || self.source_qualified
            || self.verified_native_policy
            || self.origin_checksum != self.current.origin_checksum
            || [
                &self.origin_checksum,
                &self.checkpoint_checksum,
                &self.model_checksum,
                &j.head_sha256,
                &j.external_native_digest,
                &j.external_native_sha256,
            ]
            .iter()
            .any(|v| !digest(v))
            || j.head_bytes == 0
            || j.head_bytes > 512 * 1024
            || !(1..=64).contains(&j.attempts)
            || self.current.mixed_update_count > j.attempts
            || j.credited_own_decisions < self.current.mixed_update_count
            || self.current.next_episode_ordinal != j.attempts * 4
            || j.spent_native_families.len() != (j.attempts * 4) as usize
            || j.spent_native_families.len() > 4096
            || j.spent_native_families.iter().any(|f| {
                !digest(f) || u8::from_str_radix(&f[..2], 16).is_ok_and(|first| first % 10 < 2)
            })
            || j.spent_native_families.windows(2).any(|f| f[0] >= f[1])
            || !(1..=262_144).contains(&j.external_native_count)
        {
            return Err("Conditional inference origin/journal metadata mismatch".into());
        }
        Ok(())
    }
}
#[derive(Deserialize, Serialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct Contract {
    model_version: String,
    input_contract: String,
    feature_schema: u32,
    feature_count: usize,
    hidden_count: usize,
    parameter_count: usize,
    rules_version: u32,
    rules_baseline: String,
    observation_schema: u32,
    move_schema: u32,
    catalog_hash: String,
    backend: String,
    numeric_contract: String,
    scope: String,
    value_validity: String,
}
#[derive(Deserialize, Serialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct Wire {
    schema: String,
    contract: Contract,
    source: Source,
    source_numerical_target: String,
    parameter_bits: Vec<String>,
    checksum: String,
}
impl Wire {
    fn parameters(&self) -> Result<Vec<f32>, String> {
        let c = &self.contract;
        if self.schema != DEPLOYMENT_SCHEMA
            || c.model_version != deployment::MODEL_VERSION
            || c.input_contract != deployment::INPUT_CONTRACT
            || c.feature_schema != PUBLIC_FEATURE_SCHEMA
            || c.feature_count != FEATURE_COUNT
            || c.hidden_count != HIDDEN
            || c.parameter_count != PARAMETER_COUNT
            || c.rules_version != deployment::RULES_VERSION
            || c.rules_baseline != deployment::RULES_BASELINE
            || c.observation_schema != OBSERVATION_SCHEMA
            || c.move_schema != MOVE_SCHEMA
            || c.catalog_hash != deployment::catalog_hash()
            || c.backend != "scalar"
            || c.numeric_contract != deployment::NUMERIC_CONTRACT
            || c.scope != deployment::SCOPE
            || c.value_validity != deployment::VALUE_VALIDITY
            || self.parameter_bits.len() != PARAMETER_COUNT
        {
            return Err("Incompatible conditional numeric contract/shape".into());
        }
        let component = |s: &str| {
            !s.is_empty()
                && s.len() <= 32
                && s.bytes()
                    .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'_')
        };
        let mut parts = self.source_numerical_target.split(':');
        if self.source_numerical_target.len() > 128
            || !matches!(
            (parts.next(), parts.next(), parts.next(), parts.next(), parts.next()),
            (Some("scalar-f32-f64-tick53-v1"), Some(arch), Some(os), Some("32" | "64"), None)
            if component(arch) && component(os))
        {
            return Err("Invalid conditional export numerical target".into());
        }
        let values = self
            .parameter_bits
            .iter()
            .map(|token| {
                if token.len() != 8
                    || !token
                        .bytes()
                        .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
                {
                    return Err("Conditional actor requires canonical lowercase f32 bits".into());
                }
                Ok(f32::from_bits(
                    u32::from_str_radix(token, 16).map_err(|e| e.to_string())?,
                ))
            })
            .collect::<Result<Vec<_>, String>>()?;
        policy::validate_parameters(&values)?;
        self.source.validate(&values)?;
        Ok(values)
    }
    fn expected_checksum(&mut self) -> Result<String, String> {
        self.checksum.clear();
        hash(CHECKSUM_DOMAIN, self)
    }
}
fn parse(bytes: &[u8]) -> Result<Wire, String> {
    if bytes.len() > MAX_DEPLOYMENT_BYTES {
        return Err("Conditional deployment exceeds 1 MiB".into());
    }
    serde_json::from_slice(bytes).map_err(|_| "Invalid closed conditional deployment JSON".into())
}
/// Numeric content checksum, not an authenticated source or Ready signer.
pub fn numeric_checksum(bytes: &[u8]) -> Result<String, String> {
    let mut wire = parse(bytes)?;
    wire.parameters()?;
    wire.expected_checksum()
}
/// Immutable actor-only inference. No raw weight constructor, parameter getter,
/// filesystem access, native conversion or learning-owner restoration exists.
pub struct LoadedConditionalDeployment {
    parameters: Vec<f32>,
    source: Source,
    source_numerical_target: String,
    checksum: String,
    kernel: ResolvedKernel,
}
impl LoadedConditionalDeployment {
    pub fn load(bytes: &[u8]) -> Result<Self, String> {
        let mut wire = parse(bytes)?;
        let parameters = wire.parameters()?;
        let checksum = wire.checksum.clone();
        if !digest(&checksum) || checksum != wire.expected_checksum()? {
            return Err("Conditional numeric checksum mismatch".into());
        }
        Ok(Self {
            parameters,
            source: wire.source,
            source_numerical_target: wire.source_numerical_target,
            checksum,
            kernel: Kernel::Scalar.resolve()?,
        })
    }
    pub fn checksum(&self) -> &str {
        &self.checksum
    }
    pub fn identity(&self) -> &ConditionalCurrentIdentity {
        &self.source.current
    }
    pub fn origin_checksum(&self) -> &str {
        &self.source.origin_checksum
    }
    pub fn dataset_fingerprint(&self) -> &str {
        &self.source.dataset.fingerprint
    }
    pub fn source_numerical_target(&self) -> &str {
        &self.source_numerical_target
    }
    pub fn backend(&self) -> &'static str {
        self.kernel.backend()
    }
    pub fn policy_version(&self) -> &'static str {
        POLICY_VERSION
    }
    /// Checks the native observation/key/ordered-menu contract. Observations
    /// contain no market-option field, so callers still bind actual Basic and
    /// unlimited GameState provenance. Scalar cross-target bit equality is not
    /// promised by a source numerical-target string.
    pub fn distribution(
        &self,
        observation: &Observation,
    ) -> Result<NumericPolicyDistribution, String> {
        policy::validate_contract(observation)?;
        let encoder = FeatureEncoder::new_public(observation)?;
        let rows = (0..observation.legal_actions.len())
            .map(|i| encoder.encode_legal_tagged(i))
            .collect::<Result<Vec<_>, _>>()?;
        BorrowedPolicy::from_validated_parameters(&self.parameters)?.predict(self.kernel, &rows)
    }
    /// Raw all-phase argmax. Stateful trade guarding belongs to the caller and
    /// must receive every actor callback; no guard history is stored here.
    pub fn choose_move(&self, observation: &Observation) -> Result<DeploymentDecision, String> {
        let prediction = self.distribution(observation)?;
        let best = (1..prediction.logits.len()).fold(0, |best, i| {
            if prediction.logits[i] > prediction.logits[best] {
                i
            } else {
                best
            }
        });
        Ok(DeploymentDecision {
            actor: observation.actor,
            observation_key: observation.observation_key.clone(),
            r#move: observation.legal_actions[best].r#move.clone(),
            policy_version: POLICY_VERSION.into(),
            score: f64::from(prediction.logits[best]),
        })
    }
}

/// The same all-actor Trade lifecycle as native numeric deployments, with a
/// distinct conditional identity. New games move the immutable actor once;
/// neither this adapter nor its metadata grants learning ownership.
pub struct GuardedConditionalDeployment {
    policy: LoadedConditionalDeployment,
    game: crate::deployment::guarded::GuardedGame,
}
impl GuardedConditionalDeployment {
    pub fn new(
        policy: LoadedConditionalDeployment,
        config: crate::trade_guard::TradeGuardConfig,
    ) -> Result<Self, String> {
        Ok(Self {
            policy,
            game: crate::deployment::guarded::GuardedGame::new(config)?,
        })
    }
    pub fn policy(&self) -> &LoadedConditionalDeployment {
        &self.policy
    }
    pub fn into_policy(self) -> LoadedConditionalDeployment {
        self.policy
    }
    pub fn configuration_key(&self) -> &str {
        self.game.configuration_key()
    }
    pub fn on_callback(&mut self, index: usize, o: &Observation) -> Result<(), String> {
        self.game.on_callback(index, o)
    }
    pub fn choose_move(
        &mut self,
        index: usize,
        o: &Observation,
    ) -> Result<crate::deployment::GuardedDeploymentDecision, String> {
        self.game.choose_move(
            index,
            o,
            || self.policy.distribution(o),
            self.policy.policy_version(),
            self.policy.checksum(),
            GUARDED_POLICY_VERSION,
        )
    }
}
