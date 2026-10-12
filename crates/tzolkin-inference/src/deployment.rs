//! Bounded numeric deployment, separate from BC/RL learning owners.
//! A valid checksum establishes content consistency, never training history,
//! source authenticity, adoption, calibrated values or playing strength.
//! Private immutable weights are loaded once; prediction borrows them without
//! filesystem access, weight copies or per-decision model hashing.
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use tzolkin_core::GameMove;
use tzolkin_core::observation::{MOVE_SCHEMA, OBSERVATION_SCHEMA, Observation, fingerprint};

use crate::features::{FEATURE_COUNT, FeatureEncoder, PUBLIC_FEATURE_SCHEMA};
use crate::kernel::{Kernel, ResolvedKernel};
use crate::policy::{self, BorrowedPolicy, HIDDEN, NumericPolicyDistribution, PARAMETER_COUNT};

mod current_mixed;
pub use current_mixed::CurrentMixedIdentity;
pub(crate) mod guarded;
pub use guarded::{
    BorrowedGuardedDeployment, GUARDED_DEPLOYMENT_VERSION, GuardedDeployment,
    GuardedDeploymentDecision, GuardedTradeTrace, MAX_GAME_CALLBACKS,
};

pub const DEPLOYMENT_SCHEMA: &str = "tzolkin-numeric-policy-deployment-v1";
pub const MAX_DEPLOYMENT_BYTES: usize = 1024 * 1024;
pub const MODEL_VERSION: &str = "tiny-public-policy-mlp-v1";
pub const INPUT_CONTRACT: &str = "base-3-4p-native-setup-public-playing-v2";
pub const RULES_VERSION: u32 = 1;
pub const RULES_BASELINE: &str = "1ec8fdbb61f671ca2cbf0dbac7f3662c160be677";
pub const NUMERIC_CONTRACT: &str = "ordered-f32-affine-tanh-scalar-first-max-v1";
pub const SCOPE: &str = "base-3-4p-setup-playing";
pub const VALUE_VALIDITY: &str = "unavailablePolicyOnly";
// Source export target is metadata, not a cross-target prediction guarantee.
const MAX_SOURCE_TARGET_BYTES: usize = 128;
pub(crate) fn valid_source_target(value: &str) -> bool {
    if value.len() > MAX_SOURCE_TARGET_BYTES {
        return false;
    }
    let component = |token: &str| {
        !token.is_empty()
            && token.len() <= 32
            && token
                .bytes()
                .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'_')
    };
    let mut parts = value.split(':');
    matches!((parts.next(), parts.next(), parts.next(), parts.next(), parts.next()),
        (Some("scalar-f32-f64-tick53-v1"), Some(arch), Some(os), Some("32" | "64"), None)
        if component(arch) && component(os))
}
const CHECKSUM_DOMAIN: &[u8] = b"tzolkin-numeric-policy-deployment-v1\0";

/// Same raw catalog bytes and FNV domain used by the native artifact boundary.
pub fn catalog_hash() -> String {
    format!(
        "{:016x}",
        fingerprint(include_bytes!("../../tzolkin-core/data/catalog.json"))
    )
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
#[serde(
    tag = "kind",
    rename_all = "camelCase",
    rename_all_fields = "camelCase",
    deny_unknown_fields
)]
enum Source {
    BcPrepared {
        policy_version: String,
        task: String,
        model_checksum: String,
        training_checkpoint_checksum: String,
    },
    RlCount1 {
        policy_version: String,
        task: String,
        artifact_checksum: String,
        update_count: u64,
    },
    RlRepeated {
        policy_version: String,
        task: String,
        artifact_checksum: String,
        update_count: u64,
    },
    RlCurrentMixed {
        identity: Box<CurrentMixedIdentity>,
    },
}
pub(crate) fn digest(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}
impl Source {
    fn validate(&self) -> Result<(), String> {
        let valid = match self {
            Self::BcPrepared {
                policy_version,
                task,
                model_checksum,
                training_checkpoint_checksum,
            } => {
                policy_version == "learned-public-policy-v1"
                    && task == "policyOnlyBc"
                    && digest(model_checksum)
                    && digest(training_checkpoint_checksum)
            }
            Self::RlCount1 {
                policy_version,
                task,
                artifact_checksum,
                update_count,
            } => {
                policy_version == "learned-public-rl-one-step-v1"
                    && task == "policyOnlyRlOneStep"
                    && *update_count == 1
                    && digest(artifact_checksum)
            }
            Self::RlRepeated {
                policy_version,
                task,
                artifact_checksum,
                update_count,
            } => {
                policy_version == "learned-public-rl-repeat-v1"
                    && task == "policyOnlyRlRepeat"
                    && (2..=10).contains(update_count)
                    && digest(artifact_checksum)
            }
            Self::RlCurrentMixed { identity } => identity.valid(),
        };
        if valid {
            Ok(())
        } else {
            Err("Incompatible deployment source role/count/identity".into())
        }
    }
    fn role(&self) -> &'static str {
        match self {
            Self::BcPrepared { .. } => "bcPrepared",
            Self::RlCount1 { .. } => "rlCount1",
            Self::RlRepeated { .. } => "rlRepeated",
            Self::RlCurrentMixed { .. } => "rlCurrentMixed",
        }
    }
    fn policy_version(&self) -> &str {
        match self {
            Self::BcPrepared { policy_version, .. }
            | Self::RlCount1 { policy_version, .. }
            | Self::RlRepeated { policy_version, .. } => policy_version,
            Self::RlCurrentMixed { identity } => identity.policy_version(),
        }
    }
    fn source_checksum(&self) -> &str {
        match self {
            Self::BcPrepared { model_checksum, .. } => model_checksum,
            Self::RlCount1 {
                artifact_checksum, ..
            }
            | Self::RlRepeated {
                artifact_checksum, ..
            } => artifact_checksum,
            Self::RlCurrentMixed { identity } => identity.artifact_checksum(),
        }
    }
    fn update_count(&self) -> Option<u64> {
        match self {
            Self::BcPrepared { .. } => None,
            Self::RlCount1 { update_count, .. } | Self::RlRepeated { update_count, .. } => {
                Some(*update_count)
            }
            Self::RlCurrentMixed { identity } => Some(identity.mixed_update_count()),
        }
    }
}

#[derive(Deserialize, Serialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub(crate) struct Wire {
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
            || c.model_version != MODEL_VERSION
            || c.input_contract != INPUT_CONTRACT
            || c.feature_schema != PUBLIC_FEATURE_SCHEMA
            || c.feature_count != FEATURE_COUNT
            || c.hidden_count != HIDDEN
            || c.parameter_count != PARAMETER_COUNT
            || c.rules_version != RULES_VERSION
            || c.rules_baseline != RULES_BASELINE
            || c.observation_schema != OBSERVATION_SCHEMA
            || c.move_schema != MOVE_SCHEMA
            || c.catalog_hash != catalog_hash()
            || c.backend != "scalar"
            || c.numeric_contract != NUMERIC_CONTRACT
            || c.scope != SCOPE
            || c.value_validity != VALUE_VALIDITY
            || self.parameter_bits.len() != PARAMETER_COUNT
        {
            return Err("Incompatible numeric deployment contract/shape".into());
        }
        self.source.validate()?;
        if !valid_source_target(&self.source_numerical_target) {
            return Err("Invalid deployment source numerical target".into());
        }
        let values = decode_parameter_bits(&self.parameter_bits)?;
        if let Source::RlCurrentMixed { identity } = &self.source {
            identity.validate_actor(&values, &self.source_numerical_target)?;
        }
        Ok(values)
    }
    fn expected_checksum(&mut self) -> Result<String, String> {
        self.checksum.clear();
        let bytes = serde_json::to_vec(self).map_err(|error| error.to_string())?;
        let mut hash = Sha256::new();
        hash.update(CHECKSUM_DOMAIN);
        hash.update(bytes);
        Ok(format!("{:x}", hash.finalize()))
    }
}
pub(crate) fn decode_parameter_bits(tokens: &[String]) -> Result<Vec<f32>, String> {
    let values = tokens
        .iter()
        .map(|token| {
            if token.len() != 8
                || !token
                    .bytes()
                    .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
            {
                return Err("Deployment parameters require canonical lowercase f32 bits".into());
            }
            let bits = u32::from_str_radix(token, 16).map_err(|error| error.to_string())?;
            Ok(f32::from_bits(bits))
        })
        .collect::<Result<Vec<_>, String>>()?;
    policy::validate_parameters(&values)?;
    Ok(values)
}
pub(crate) fn parse(bytes: &[u8]) -> Result<Wire, String> {
    if bytes.len() > MAX_DEPLOYMENT_BYTES {
        return Err("Numeric deployment exceeds 1 MiB".into());
    }
    // Typed required fields/unknown keys/duplicates and trailing input all reject.
    serde_json::from_slice(bytes).map_err(|_| "Invalid closed numeric deployment JSON".into())
}

/// Compute a numeric content checksum after closed contract/bit/finite validation.
/// The checksum field is excluded. This is not a signer, owner constructor or
/// qualification boundary; anybody can calculate a checksum of numeric content.
pub fn numeric_checksum(bytes: &[u8]) -> Result<String, String> {
    let mut wire = parse(bytes)?;
    wire.parameters()?;
    wire.expected_checksum()
}

/// Immutable numeric storage. No raw-array constructor, Deserialize, parameter
/// setter/getter, learning-owner conversion or filesystem access is exposed.
pub struct LoadedDeployment {
    parameters: Vec<f32>,
    source: Source,
    source_numerical_target: String,
    checksum: String,
    kernel: ResolvedKernel,
}
impl LoadedDeployment {
    pub fn load(bytes: &[u8]) -> Result<Self, String> {
        Self::from_wire(parse(bytes)?)
    }
    pub(crate) fn from_wire(mut wire: Wire) -> Result<Self, String> {
        let parameters = wire.parameters()?;
        let checksum = wire.checksum.clone();
        if !digest(&checksum) || checksum != wire.expected_checksum()? {
            return Err("Numeric deployment checksum mismatch".into());
        }
        Ok(Self {
            parameters,
            source: wire.source,
            source_numerical_target: wire.source_numerical_target,
            checksum,
            kernel: Kernel::Scalar.resolve()?,
        })
    }
    /// Content identity; it does not attest source history or adoption.
    pub fn checksum(&self) -> &str {
        &self.checksum
    }
    /// Closed claimed role, checked for consistency rather than authenticated.
    pub fn source_role(&self) -> &'static str {
        self.source.role()
    }
    /// Closed numeric metadata, not a joint owner. The actor fingerprint is
    /// checked against loaded bits; the residual fingerprint remains a claim.
    pub fn current_mixed_identity(&self) -> Option<&CurrentMixedIdentity> {
        match &self.source {
            Source::RlCurrentMixed { identity } => Some(identity),
            _ => None,
        }
    }
    /// Claimed BC model/RL artifact identity, retained without recreating it.
    pub fn source_checksum(&self) -> &str {
        self.source.source_checksum()
    }
    /// Native export target claimed in the checksum-bound source metadata.
    /// BC records the export target, not authenticated training provenance;
    /// it may differ from this runtime's target. Backend is a separate getter.
    pub fn source_numerical_target(&self) -> &str {
        &self.source_numerical_target
    }
    pub fn update_count(&self) -> Option<u64> {
        self.source.update_count()
    }
    pub fn policy_version(&self) -> &str {
        self.source.policy_version()
    }
    pub fn backend(&self) -> &'static str {
        self.kernel.backend()
    }
    pub fn distribution(
        &self,
        observation: &Observation,
    ) -> Result<NumericPolicyDistribution, String> {
        policy::validate_contract(observation)?;
        let encoder = FeatureEncoder::new_public(observation)?;
        let rows = (0..observation.legal_actions.len())
            .map(|index| encoder.encode_legal_tagged(index))
            .collect::<Result<Vec<_>, _>>()?;
        BorrowedPolicy::from_validated_parameters(&self.parameters)?.predict(self.kernel, &rows)
    }
    /// The caller still owns legal-set provenance and must recheck the returned
    /// move against its current authority. Cross-target transcendental bit
    /// equality is not guaranteed by this Scalar deployment contract.
    pub fn choose_move(&self, observation: &Observation) -> Result<DeploymentDecision, String> {
        let prediction = self.distribution(observation)?;
        let best = (1..prediction.logits.len()).fold(0, |best, index| {
            if prediction.logits[index] > prediction.logits[best] {
                index
            } else {
                best
            }
        });
        Ok(DeploymentDecision {
            actor: observation.actor,
            observation_key: observation.observation_key.clone(),
            r#move: observation.legal_actions[best].r#move.clone(),
            policy_version: self.policy_version().into(),
            score: f64::from(prediction.logits[best]),
        })
    }
}
/// A numeric decision, not BC/RL training provenance or value supervision.
#[derive(Debug, Serialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct DeploymentDecision {
    pub actor: usize,
    pub observation_key: String,
    pub r#move: GameMove,
    pub policy_version: String,
    pub score: f64,
}

#[cfg(test)]
mod tests {
    use super::*;
    fn wire() -> Wire {
        Wire {
            schema: DEPLOYMENT_SCHEMA.into(),
            contract: Contract {
                model_version: MODEL_VERSION.into(),
                input_contract: INPUT_CONTRACT.into(),
                feature_schema: PUBLIC_FEATURE_SCHEMA,
                feature_count: FEATURE_COUNT,
                hidden_count: HIDDEN,
                parameter_count: PARAMETER_COUNT,
                rules_version: RULES_VERSION,
                rules_baseline: RULES_BASELINE.into(),
                observation_schema: OBSERVATION_SCHEMA,
                move_schema: MOVE_SCHEMA,
                catalog_hash: catalog_hash(),
                backend: "scalar".into(),
                numeric_contract: NUMERIC_CONTRACT.into(),
                scope: SCOPE.into(),
                value_validity: VALUE_VALIDITY.into(),
            },
            source: Source::BcPrepared {
                policy_version: "learned-public-policy-v1".into(),
                task: "policyOnlyBc".into(),
                model_checksum: "a".repeat(64),
                training_checkpoint_checksum: "b".repeat(64),
            },
            source_numerical_target: "scalar-f32-f64-tick53-v1:x86_64:windows:64".into(),
            parameter_bits: vec!["00000000".into(); PARAMETER_COUNT],
            checksum: String::new(),
        }
    }
    fn encoded(mut wire: Wire) -> Vec<u8> {
        wire.checksum = wire.expected_checksum().unwrap();
        serde_json::to_vec(&wire).unwrap()
    }
    #[test]
    fn guarded_deployment_exits_repeated_trade_and_binds_all_callbacks() {
        use tzolkin_core::observation::observe;
        use tzolkin_core::{Pending, Phase, Resource, Task, apply_move};
        let mut state =
            tzolkin_core::create_game(vec!["A".into(), "B".into(), "C".into()], 17, false).unwrap();
        while state.phase == Phase::Setup {
            let o = observe(&state, state.current_player).unwrap();
            state = apply_move(&state, o.legal_actions[0].r#move.clone()).unwrap();
        }
        state.players[state.current_player]
            .resources
            .insert(Resource::Corn, 3);
        state.players[state.current_player]
            .resources
            .insert(Resource::Wood, 3);
        state.pending = Some(Pending {
            title: "Trade".into(),
            task: Task::Trade,
            after: vec![],
        });
        let o = observe(&state, state.current_player).unwrap();
        let loaded = LoadedDeployment::load(&encoded(wire())).unwrap();
        let raw = loaded.choose_move(&o).unwrap();
        let (borrowed_first, borrowed_repeat) = {
            let mut a = BorrowedGuardedDeployment::new(&loaded, Default::default()).unwrap();
            let mut b = BorrowedGuardedDeployment::new(&loaded, Default::default()).unwrap();
            assert!(std::ptr::eq(a.policy(), b.policy()));
            assert!(std::ptr::eq(a.policy(), &loaded));
            assert_eq!(a.configuration_key(), b.configuration_key());
            a.on_callback(0, &o).unwrap();
            let first = serde_json::to_value(a.choose_move(0, &o).unwrap()).unwrap();
            a.on_callback(1, &o).unwrap();
            let repeated = serde_json::to_value(a.choose_move(1, &o).unwrap()).unwrap();
            assert!(a.choose_move(1, &o).is_err());
            assert!(a.on_callback(2, &o).is_err());
            // The other view starts with no Trade history despite a's repeated
            // trade and terminal failure, then independently sees its repeat.
            b.on_callback(0, &o).unwrap();
            assert_eq!(
                serde_json::to_value(b.choose_move(0, &o).unwrap()).unwrap(),
                first
            );
            b.on_callback(1, &o).unwrap();
            assert_eq!(
                serde_json::to_value(b.choose_move(1, &o).unwrap()).unwrap(),
                repeated
            );
            b.on_callback(2, &o).unwrap();
            let mut changed = o.clone();
            changed.players[o.actor].resources[0] += 1;
            assert!(b.choose_move(2, &changed).is_err());
            assert!(b.choose_move(2, &o).is_err());
            let mut stale = BorrowedGuardedDeployment::new(&loaded, Default::default()).unwrap();
            assert!(stale.on_callback(1, &o).is_err());
            assert!(stale.on_callback(0, &o).is_err());
            (first, repeated)
        };
        let mut session = GuardedDeployment::new(loaded, Default::default()).unwrap();
        session.on_callback(0, &o).unwrap();
        let first = session.choose_move(0, &o).unwrap();
        assert_eq!(serde_json::to_value(&first).unwrap(), borrowed_first);
        assert_eq!(first.decision.r#move, raw.r#move);
        assert!(!first.trade.unwrap().overridden);
        session.on_callback(1, &o).unwrap();
        let repeated = session.choose_move(1, &o).unwrap();
        assert_eq!(serde_json::to_value(&repeated).unwrap(), borrowed_repeat);
        let skip = crate::trade_guard::skip_index(&o).unwrap();
        assert_eq!(repeated.decision.r#move, o.legal_actions[skip].r#move);
        assert!(repeated.trade.unwrap().repeat_seen);
        assert_ne!(repeated.raw_legal_index, repeated.effective_legal_index);
        session.on_callback(2, &o).unwrap();
        assert!(
            !session
                .choose_move(2, &o)
                .unwrap()
                .trade
                .unwrap()
                .overridden
        );
        // Every seat notification breaks an interrupted actor's episode even
        // when this model did not choose that other seat's action.
        state.current_player = (state.current_player + 1) % 3;
        let other = observe(&state, state.current_player).unwrap();
        session.on_callback(3, &other).unwrap();
        session.on_callback(4, &o).unwrap();
        assert!(
            !session
                .choose_move(4, &o)
                .unwrap()
                .trade
                .unwrap()
                .overridden
        );
        assert!(session.choose_move(4, &o).is_err());
        assert!(session.on_callback(5, &o).is_err());
        let mut stale = GuardedDeployment::new(
            LoadedDeployment::load(&encoded(wire())).unwrap(),
            Default::default(),
        )
        .unwrap();
        assert!(stale.on_callback(1, &o).is_err());
        assert!(stale.on_callback(0, &o).is_err());
        let mut altered = GuardedDeployment::new(
            LoadedDeployment::load(&encoded(wire())).unwrap(),
            Default::default(),
        )
        .unwrap();
        altered.on_callback(0, &o).unwrap();
        let mut changed = o.clone();
        changed.players[o.actor].resources[0] += 1;
        assert!(altered.choose_move(0, &changed).is_err());
        assert!(altered.choose_move(0, &o).is_err());
        // A fresh explicit game moves the same immutable model and no history.
        let checksum = altered.policy().checksum().to_owned();
        let mut fresh = GuardedDeployment::new(altered.into_policy(), Default::default()).unwrap();
        assert_eq!(fresh.policy().checksum(), checksum);
        fresh.on_callback(0, &o).unwrap();
        assert!(!fresh.choose_move(0, &o).unwrap().trade.unwrap().overridden);
    }
    #[test]
    fn preserves_bits_and_distinct_source_roles() {
        let mut model = wire();
        model.parameter_bits[0] = "80000000".into();
        let loaded = LoadedDeployment::load(&encoded(model)).unwrap();
        assert_eq!(loaded.parameters[0].to_bits(), (-0.0f32).to_bits());
        assert_eq!(loaded.source_role(), "bcPrepared");
        assert_eq!(loaded.update_count(), None);
        assert_eq!(
            loaded.source_numerical_target(),
            "scalar-f32-f64-tick53-v1:x86_64:windows:64"
        );
        // A source export target is not required to equal the loader runtime.
        let mut other_target = wire();
        other_target.source_numerical_target = "scalar-f32-f64-tick53-v1:aarch64:linux:64".into();
        assert!(LoadedDeployment::load(&encoded(other_target)).is_ok());
        for count in [1, 2, 10] {
            let mut model = wire();
            model.source = if count == 1 {
                Source::RlCount1 {
                    policy_version: "learned-public-rl-one-step-v1".into(),
                    task: "policyOnlyRlOneStep".into(),
                    artifact_checksum: "c".repeat(64),
                    update_count: count,
                }
            } else {
                Source::RlRepeated {
                    policy_version: "learned-public-rl-repeat-v1".into(),
                    task: "policyOnlyRlRepeat".into(),
                    artifact_checksum: "d".repeat(64),
                    update_count: count,
                }
            };
            let loaded = LoadedDeployment::load(&encoded(model)).unwrap();
            assert_eq!(loaded.update_count(), Some(count));
            assert_ne!(loaded.source_role(), "bcPrepared");
        }
    }
    #[test]
    fn current_mixed_identity_roundtrip_binds_generation_counts_and_actor_bits() {
        // Numeric metadata fixtures establish no trained owner or admission.
        for (initial, count, imported) in [
            (true, 1, 1),
            (false, 2, 1),
            (false, 49, 7),
            (false, 1_000_000, 1_000_000),
        ] {
            let mut model = wire();
            model.parameter_bits[0] = "80000000".into();
            model.parameter_bits[1] = "3eaaaaab".into();
            let parameters = model.parameters().unwrap();
            let domain: &[u8] = if initial {
                b"tzolkin-mixed-inference-actor-v1\0"
            } else {
                b"tzolkin-continued-mixed-inference-actor-v2\0"
            };
            let mut hash = Sha256::new();
            hash.update(domain);
            hash.update(serde_json::to_vec(&parameters).unwrap());
            let fingerprint = format!("{:x}", hash.finalize());
            let mut policy = serde_json::json!({
                "schema": if initial {"tzolkin-public-rl-mixed-joint-inference-identity-v1"}
                    else {"tzolkin-public-rl-continued-mixed-inference-identity-v2"},
                "policyVersion": if initial {"learned-public-rl-mixed-joint-argmax-v1"}
                    else {"learned-public-rl-continued-mixed-joint-argmax-v2"},
                "samplingVersion": if initial {"public-stochastic-rl-mixed-joint-uniform-tick53-v1"}
                    else {"public-stochastic-rl-continued-mixed-joint-uniform-tick53-v2"},
                "task": if initial {"mixedLearnerOwnSumQuarterJointOneBatchV1"}
                    else {"currentMixedLearnerOwnSumQuarterJointV2"},
                "artifactChecksum":"c".repeat(64), "importedUpdateCount":imported,
                "mixedUpdateCount":count, "actorModelVersion":MODEL_VERSION,
                "criticModelVersion":"tiny-public-state-critic-mlp-v1",
                "actorParametersChecksum":fingerprint,
                "residualParametersChecksum":"d".repeat(64),
                "featureSchema":PUBLIC_FEATURE_SCHEMA,"inputContract":INPUT_CONTRACT,
                "contextSchema":1,"contextContract":"public-v2-state-prefix384-v1",
                "backend":"scalar","numericalTarget":model.source_numerical_target,
            });
            if !initial {
                policy["firstMixedChecksum"] = "e".repeat(64).into();
                policy["parentArtifactChecksum"] = "f".repeat(64).into();
            }
            let identity: CurrentMixedIdentity = serde_json::from_value(serde_json::json!({
                "kind":if initial {"initialMixedV1"} else {"continuedMixedV2"},"policy":policy,
            }))
            .unwrap();
            model.source = Source::RlCurrentMixed {
                identity: Box::new(identity),
            };
            let bytes = encoded(model);
            let loaded = LoadedDeployment::load(&bytes).unwrap();
            let identity = loaded.current_mixed_identity().unwrap();
            assert_eq!(loaded.source_role(), "rlCurrentMixed");
            assert_eq!(loaded.source_checksum(), "c".repeat(64));
            assert_eq!(loaded.update_count(), Some(count));
            assert_eq!(identity.mixed_update_count(), count);
            assert_eq!(identity.imported_update_count(), imported);
            assert_eq!(loaded.parameters[0].to_bits(), (-0.0f32).to_bits());
            assert_eq!(loaded.parameters[1].to_bits(), 0x3eaaaaab);
            // Actor identity uses native generation domains and serde f32
            // numbers. Recomputing the outer content checksum cannot bypass it.
            let original: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
            for (field, bad) in [
                ("actorParametersChecksum", serde_json::json!("0".repeat(64))),
                (
                    "residualParametersChecksum",
                    serde_json::json!("not-a-digest"),
                ),
                ("actorModelVersion", serde_json::json!("different")),
                ("criticModelVersion", serde_json::json!("different")),
                ("schema", serde_json::json!("different")),
                ("task", serde_json::json!("different")),
                ("samplingVersion", serde_json::json!("different")),
                ("contextSchema", serde_json::json!(2)),
                ("importedUpdateCount", serde_json::json!(0)),
                (
                    "mixedUpdateCount",
                    serde_json::json!(if initial { 2 } else { 1 }),
                ),
                (
                    "numericalTarget",
                    serde_json::json!("scalar-f32-f64-tick53-v1:aarch64:linux:64"),
                ),
            ] {
                let mut changed = original.clone();
                changed["source"]["identity"]["policy"][field] = bad;
                assert!(
                    numeric_checksum(&serde_json::to_vec(&changed).unwrap()).is_err(),
                    "{field}"
                );
            }
            for field in ["mixedUpdateCount", "importedUpdateCount"] {
                let mut changed = original.clone();
                changed["source"]["identity"]["policy"][field] = 1_000_001.into();
                assert!(numeric_checksum(&serde_json::to_vec(&changed).unwrap()).is_err());
            }
            let mut changed = original.clone();
            changed["parameterBits"][1] = "3eaaaaaa".into();
            assert!(numeric_checksum(&serde_json::to_vec(&changed).unwrap()).is_err());
            let mut changed = original.clone();
            changed["source"]["identity"]["policy"]["extra"] = true.into();
            assert!(numeric_checksum(&serde_json::to_vec(&changed).unwrap()).is_err());
            let mut changed = original.clone();
            if initial {
                changed["source"]["identity"]["policy"]["firstMixedChecksum"] =
                    "e".repeat(64).into();
            } else {
                changed["source"]["identity"]["policy"]
                    .as_object_mut()
                    .unwrap()
                    .remove("parentArtifactChecksum");
            }
            assert!(numeric_checksum(&serde_json::to_vec(&changed).unwrap()).is_err());
        }
    }
    #[test]
    fn rejects_tampered_bits_contract_counts_and_wire() {
        let valid = encoded(wire());
        let mut changed: serde_json::Value = serde_json::from_slice(&valid).unwrap();
        changed["parameterBits"][0] = "3f800000".into();
        assert!(LoadedDeployment::load(&serde_json::to_vec(&changed).unwrap()).is_err());
        let mut changed_target: serde_json::Value = serde_json::from_slice(&valid).unwrap();
        changed_target["sourceNumericalTarget"] =
            "scalar-f32-f64-tick53-v1:aarch64:linux:64".into();
        assert!(LoadedDeployment::load(&serde_json::to_vec(&changed_target).unwrap()).is_err());
        for target in [
            "",
            "scalar-f32-f64-tick53-v2:x86_64:windows:64",
            "scalar-f32-f64-tick53-v1:X86_64:windows:64",
            "scalar-f32-f64-tick53-v1::linux:64",
            "scalar-f32-f64-tick53-v1:x86_64:linux:128",
            "scalar-f32-f64-tick53-v1:x86_64:linux:64:extra",
            "scalar-f32-f64-tick53-v1:x86_64:日本:64",
        ] {
            let mut wrong = wire();
            wrong.source_numerical_target = target.into();
            assert!(numeric_checksum(&serde_json::to_vec(&wrong).unwrap()).is_err());
        }
        let mut wrong = wire();
        wrong.source_numerical_target = "a".repeat(MAX_SOURCE_TARGET_BYTES + 1);
        assert!(numeric_checksum(&serde_json::to_vec(&wrong).unwrap()).is_err());
        for token in ["7f800000", "7fc00000", "8000000", "ABCDEF00"] {
            let mut model = wire();
            model.parameter_bits[0] = token.into();
            assert!(numeric_checksum(&serde_json::to_vec(&model).unwrap()).is_err());
        }
        let mut wrong = wire();
        wrong.contract.feature_schema = 1;
        assert!(numeric_checksum(&serde_json::to_vec(&wrong).unwrap()).is_err());
        let mut wrong = wire();
        wrong.contract.catalog_hash = "0".repeat(16);
        assert!(numeric_checksum(&serde_json::to_vec(&wrong).unwrap()).is_err());
        let mut wrong = wire();
        wrong.source = Source::RlCount1 {
            policy_version: "learned-public-policy-v1".into(),
            task: "policyOnlyBc".into(),
            artifact_checksum: "d".repeat(64),
            update_count: 1,
        };
        assert!(numeric_checksum(&serde_json::to_vec(&wrong).unwrap()).is_err());
        for count in [0, 1, 11, u64::MAX] {
            let mut wrong = wire();
            wrong.source = Source::RlRepeated {
                policy_version: "learned-public-rl-repeat-v1".into(),
                task: "policyOnlyRlRepeat".into(),
                artifact_checksum: "d".repeat(64),
                update_count: count,
            };
            assert!(numeric_checksum(&serde_json::to_vec(&wrong).unwrap()).is_err());
        }
        for tail in [b" {}".as_slice(), b"x".as_slice()] {
            let mut bytes = valid.clone();
            bytes.extend_from_slice(tail);
            assert!(LoadedDeployment::load(&bytes).is_err());
        }
        let text = String::from_utf8(valid).unwrap();
        for text in [
            text.replacen("\"checksum\":", "\"unknown\":0,\"checksum\":", 1),
            text.replacen(
                "\"backend\":\"scalar\"",
                "\"backend\":\"scalar\",\"backend\":\"scalar\"",
                1,
            ),
            text.replacen(
                "\"modelChecksum\":",
                "\"extra\":false,\"modelChecksum\":",
                1,
            ),
        ] {
            assert!(LoadedDeployment::load(text.as_bytes()).is_err());
        }
        let error = LoadedDeployment::load(&vec![b' '; MAX_DEPLOYMENT_BYTES + 1])
            .err()
            .unwrap();
        assert!(error.contains("1 MiB"));
    }
}
