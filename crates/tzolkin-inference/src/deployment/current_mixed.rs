//! Closed inference metadata only. No joint owner or learning admission is
//! reconstructed; residual identity is retained without residual weights.
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use super::{INPUT_CONTRACT, MODEL_VERSION, digest, valid_source_target};
use crate::features::PUBLIC_FEATURE_SCHEMA;

const MAX_UPDATES: u64 = 1_000_000;
macro_rules! policy_identity {
    ($name:ident $(, $lineage:ident)*) => {
        #[derive(Debug, Deserialize, Serialize, PartialEq, Eq)]
        #[serde(rename_all = "camelCase", deny_unknown_fields)]
        pub struct $name {
            schema: String,
            policy_version: String,
            sampling_version: String,
            task: String,
            artifact_checksum: String,
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
            $($lineage: String,)*
        }
        impl $name {
            fn valid_common(&self) -> bool {
                (1..=MAX_UPDATES).contains(&self.imported_update_count)
                    && self.actor_model_version == MODEL_VERSION
                    && self.critic_model_version == "tiny-public-state-critic-mlp-v1"
                    && digest(&self.artifact_checksum)
                    && digest(&self.actor_parameters_checksum)
                    && digest(&self.residual_parameters_checksum)
                    && self.feature_schema == PUBLIC_FEATURE_SCHEMA
                    && self.input_contract == INPUT_CONTRACT
                    && self.context_schema == 1
                    && self.context_contract == "public-v2-state-prefix384-v1"
                    && self.backend == "scalar"
                    && valid_source_target(&self.numerical_target)
            }
        }
    };
}
policy_identity!(InitialMixedIdentity);
policy_identity!(
    ContinuedMixedIdentity,
    first_mixed_checksum,
    parent_artifact_checksum
);

/// The original generation remains explicit; count1 and continued identities
/// keep their separate schemas, versions, tasks and fingerprint hash domains.
#[derive(Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(tag = "kind", rename_all = "camelCase", deny_unknown_fields)]
pub enum CurrentMixedIdentity {
    InitialMixedV1 { policy: InitialMixedIdentity },
    ContinuedMixedV2 { policy: ContinuedMixedIdentity },
}
macro_rules! identity_getter {
    ($method:ident, $field:ident, $result:ty) => {
        pub fn $method(&self) -> $result {
            match self {
                Self::InitialMixedV1 { policy } => &policy.$field,
                Self::ContinuedMixedV2 { policy } => &policy.$field,
            }
        }
    };
}
impl CurrentMixedIdentity {
    pub(super) fn valid(&self) -> bool {
        match self {
            Self::InitialMixedV1 { policy: p } => {
                p.valid_common()
                    && p.schema == "tzolkin-public-rl-mixed-joint-inference-identity-v1"
                    && p.policy_version == "learned-public-rl-mixed-joint-argmax-v1"
                    && p.sampling_version == "public-stochastic-rl-mixed-joint-uniform-tick53-v1"
                    && p.task == "mixedLearnerOwnSumQuarterJointOneBatchV1"
                    && p.mixed_update_count == 1
            }
            Self::ContinuedMixedV2 { policy: p } => {
                p.valid_common()
                    && p.schema == "tzolkin-public-rl-continued-mixed-inference-identity-v2"
                    && p.policy_version == "learned-public-rl-continued-mixed-joint-argmax-v2"
                    && p.sampling_version
                        == "public-stochastic-rl-continued-mixed-joint-uniform-tick53-v2"
                    && p.task == "currentMixedLearnerOwnSumQuarterJointV2"
                    && (2..=MAX_UPDATES).contains(&p.mixed_update_count)
                    && digest(&p.first_mixed_checksum)
                    && digest(&p.parent_artifact_checksum)
            }
        }
    }
    pub(super) fn validate_actor(&self, parameters: &[f32], target: &str) -> Result<(), String> {
        if !self.valid() || target != self.numerical_target() {
            return Err("Incompatible current Mixed deployment identity/count/target".into());
        }
        let (domain, expected): (&[u8], &str) = match self {
            Self::InitialMixedV1 { policy } => (
                b"tzolkin-mixed-inference-actor-v1\0",
                &policy.actor_parameters_checksum,
            ),
            Self::ContinuedMixedV2 { policy } => (
                b"tzolkin-continued-mixed-inference-actor-v2\0",
                &policy.actor_parameters_checksum,
            ),
        };
        // Match the native identity hash: generation domain followed by the
        // canonical serde f32 array, reconstructed from the exported f32 bits.
        let mut hash = Sha256::new();
        hash.update(domain);
        hash.update(serde_json::to_vec(parameters).map_err(|error| error.to_string())?);
        if format!("{:x}", hash.finalize()) != expected {
            return Err("Current Mixed deployment actor fingerprint mismatch".into());
        }
        Ok(())
    }
    identity_getter!(artifact_checksum, artifact_checksum, &str);
    identity_getter!(policy_version, policy_version, &str);
    identity_getter!(numerical_target, numerical_target, &str);
    /// Actual claimed counts, separate from legacy policy-only RL roles.
    pub fn imported_update_count(&self) -> u64 {
        match self {
            Self::InitialMixedV1 { policy } => policy.imported_update_count,
            Self::ContinuedMixedV2 { policy } => policy.imported_update_count,
        }
    }
    pub fn mixed_update_count(&self) -> u64 {
        match self {
            Self::InitialMixedV1 { policy } => policy.mixed_update_count,
            Self::ContinuedMixedV2 { policy } => policy.mixed_update_count,
        }
    }
}
