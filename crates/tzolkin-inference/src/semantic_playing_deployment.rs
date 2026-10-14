//! Actor-only numeric deployment from a closed semantic training checkpoint.
//! Loading establishes content integrity, not training ownership, source
//! authentication, calibrated values, application adoption or playing strength.
//! No critic, journal, corpus or filesystem input enters prediction.
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use tzolkin_core::observation::{MOVE_SCHEMA, OBSERVATION_SCHEMA, Observation};

use crate::deployment::{self, DeploymentDecision};
use crate::features::{FEATURE_COUNT, FeatureEncoder, SEMANTIC_PLAYING_FEATURE_SCHEMA};
use crate::kernel::{Kernel, ResolvedKernel};
use crate::policy::{
    self, BorrowedSemanticPlayingPolicy, HIDDEN, NumericPolicyDistribution, PARAMETER_COUNT,
};

pub const DEPLOYMENT_SCHEMA: &str = "tzolkin-semantic-playing-checkpoint-policy-inference-v1";
pub const MAX_DEPLOYMENT_BYTES: usize = 1024 * 1024;
pub const POLICY_VERSION: &str = "semantic-playing-checkpoint-policy-scalar-v1";
pub const INPUT_CONTRACT: &str = "semantic-playing-checkpoint-actor-base-3-4p-v1";
pub const SOURCE_ROLE: &str = "semanticPlayingCheckpoint";
pub const TASK: &str = "SemanticPlayingCheckpointPolicyInference";
pub const SCOPE: &str = "base-3-4p-playing-only";
const CHECKSUM_DOMAIN: &[u8] = b"tzolkin-semantic-playing-checkpoint-policy-inference-v1\0";
const ACTOR_DOMAIN: &[u8] = b"semantic-current-actor-bits-v1\0";

fn digest(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}

fn actor_checksum(values: &[f32]) -> String {
    let mut hash = Sha256::new();
    hash.update(ACTOR_DOMAIN);
    for value in values {
        hash.update(value.to_bits().to_le_bytes());
    }
    format!("{:x}", hash.finalize())
}

/// Digest links describe the checked export source. They do not authenticate
/// its history, restore a training owner or make critic values available.
/// Source paths, raw Human metadata and critic parameters have no wire fields.
#[derive(Debug, Deserialize, Serialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct SemanticCheckpointActorIdentity {
    source_role: String,
    task: String,
    checkpoint_sha256: String,
    close_declaration_sha256: String,
    root_checksum: String,
    previous_identity_checksum: String,
    current_identity_checksum: String,
    actor_checksum: String,
    updates: u64,
    playing_credit: u64,
    source_authenticated: bool,
    source_qualified: bool,
    verified_native_policy: bool,
    strength_measured: bool,
}

impl SemanticCheckpointActorIdentity {
    fn validate(&self, values: &[f32]) -> Result<(), String> {
        if self.source_role != SOURCE_ROLE
            || self.task != TASK
            || self.updates == 0
            || self.playing_credit == 0
            || self.source_authenticated
            || self.source_qualified
            || self.verified_native_policy
            || self.strength_measured
            || [
                &self.checkpoint_sha256,
                &self.close_declaration_sha256,
                &self.root_checksum,
                &self.previous_identity_checksum,
                &self.current_identity_checksum,
                &self.actor_checksum,
            ]
            .iter()
            .any(|value| !digest(value))
            || self.actor_checksum != actor_checksum(values)
        {
            return Err("Incompatible semantic checkpoint actor identity".into());
        }
        Ok(())
    }

    pub fn checkpoint_sha256(&self) -> &str {
        &self.checkpoint_sha256
    }

    pub fn current_identity_checksum(&self) -> &str {
        &self.current_identity_checksum
    }

    pub fn actor_checksum(&self) -> &str {
        &self.actor_checksum
    }

    pub fn updates(&self) -> u64 {
        self.updates
    }

    pub fn playing_credit(&self) -> u64 {
        self.playing_credit
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
    identity: SemanticCheckpointActorIdentity,
    source_numerical_target: String,
    parameter_bits: Vec<String>,
    checksum: String,
}

impl Wire {
    fn parameters(&self) -> Result<Vec<f32>, String> {
        let c = &self.contract;
        if self.schema != DEPLOYMENT_SCHEMA
            || c.model_version != deployment::MODEL_VERSION
            || c.input_contract != INPUT_CONTRACT
            || c.feature_schema != SEMANTIC_PLAYING_FEATURE_SCHEMA
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
            || c.scope != SCOPE
            || c.value_validity != deployment::VALUE_VALIDITY
            || !deployment::valid_source_target(&self.source_numerical_target)
            || self.parameter_bits.len() != PARAMETER_COUNT
        {
            return Err("Incompatible semantic Playing deployment contract/shape".into());
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
                    return Err("Semantic actor requires canonical lowercase f32 bits".into());
                }
                Ok(f32::from_bits(
                    u32::from_str_radix(token, 16).map_err(|e| e.to_string())?,
                ))
            })
            .collect::<Result<Vec<_>, String>>()?;
        // The borrowed fast kernel only checks shape. This immutable boundary
        // performs the one complete finite-value scan before any prediction.
        policy::validate_parameters(&values)?;
        self.identity.validate(&values)?;
        Ok(values)
    }

    fn expected_checksum(&mut self) -> Result<String, String> {
        self.checksum.clear();
        let mut hash = Sha256::new();
        hash.update(CHECKSUM_DOMAIN);
        hash.update(serde_json::to_vec(self).map_err(|e| e.to_string())?);
        Ok(format!("{:x}", hash.finalize()))
    }
}

fn parse(bytes: &[u8]) -> Result<Wire, String> {
    if bytes.len() > MAX_DEPLOYMENT_BYTES {
        return Err("Semantic Playing deployment exceeds 1 MiB".into());
    }
    // Derived closed structs reject duplicate and unknown fields at each level.
    serde_json::from_slice(bytes).map_err(|_| "Invalid closed semantic deployment JSON".into())
}

/// Computes numeric content identity for the checked exporter. This is not a
/// source signature, a process-close verifier or a learning-owner constructor.
pub fn numeric_checksum(bytes: &[u8]) -> Result<String, String> {
    let mut wire = parse(bytes)?;
    wire.parameters()?;
    wire.expected_checksum()
}

/// Immutable actor-only storage, with no raw-parameter constructor or getter.
pub struct LoadedSemanticPlayingDeployment {
    parameters: Vec<f32>,
    identity: SemanticCheckpointActorIdentity,
    source_numerical_target: String,
    checksum: String,
    kernel: ResolvedKernel,
}

impl LoadedSemanticPlayingDeployment {
    pub fn load(bytes: &[u8]) -> Result<Self, String> {
        let mut wire = parse(bytes)?;
        let parameters = wire.parameters()?;
        let checksum = wire.checksum.clone();
        if !digest(&checksum) || checksum != wire.expected_checksum()? {
            return Err("Semantic Playing deployment checksum mismatch".into());
        }
        Ok(Self {
            parameters,
            identity: wire.identity,
            source_numerical_target: wire.source_numerical_target,
            checksum,
            kernel: Kernel::Scalar.resolve()?,
        })
    }

    pub fn checksum(&self) -> &str {
        &self.checksum
    }

    pub fn identity(&self) -> &SemanticCheckpointActorIdentity {
        &self.identity
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

    /// Setup is deliberately refused. The later App composite must use the
    /// shared CornFirst selector and a fresh game-local TradeGuard lifecycle.
    /// Redacted observations cannot authenticate the originating game options
    /// or guarantee cross-target bit equality from source-target metadata.
    pub fn distribution_playing(
        &self,
        observation: &Observation,
    ) -> Result<NumericPolicyDistribution, String> {
        let encoder = FeatureEncoder::new_semantic_playing(observation)?;
        let rows = (0..observation.legal_actions.len())
            .map(|i| encoder.encode_legal_tagged(i))
            .collect::<Result<Vec<_>, _>>()?;
        BorrowedSemanticPlayingPolicy::from_validated_parameters(&self.parameters)?
            .predict(self.kernel, &rows)
    }

    /// Raw Playing-only first-maximum selection, without a trade-history claim.
    pub fn choose_move_playing(
        &self,
        observation: &Observation,
    ) -> Result<DeploymentDecision, String> {
        let prediction = self.distribution_playing(observation)?;
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
