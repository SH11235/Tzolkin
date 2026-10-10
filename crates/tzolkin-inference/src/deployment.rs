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
fn valid_source_target(value: &str) -> bool {
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
}
fn digest(value: &str) -> bool {
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
        }
    }
    fn policy_version(&self) -> &str {
        match self {
            Self::BcPrepared { policy_version, .. }
            | Self::RlCount1 { policy_version, .. }
            | Self::RlRepeated { policy_version, .. } => policy_version,
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
        }
    }
    fn update_count(&self) -> Option<u64> {
        match self {
            Self::BcPrepared { .. } => None,
            Self::RlCount1 { update_count, .. } | Self::RlRepeated { update_count, .. } => {
                Some(*update_count)
            }
        }
    }
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
        let values = self
            .parameter_bits
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
    fn expected_checksum(&mut self) -> Result<String, String> {
        self.checksum.clear();
        let bytes = serde_json::to_vec(self).map_err(|error| error.to_string())?;
        let mut hash = Sha256::new();
        hash.update(CHECKSUM_DOMAIN);
        hash.update(bytes);
        Ok(format!("{:x}", hash.finalize()))
    }
}
fn parse(bytes: &[u8]) -> Result<Wire, String> {
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
        let mut wire = parse(bytes)?;
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
