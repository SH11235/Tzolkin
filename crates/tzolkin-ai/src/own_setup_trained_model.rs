//! Updated Setup parameters with their own source lineage and Scalar identity.
//! A serialized identity is not a learning owner or a restore certificate.
use serde::Serialize;
use tzolkin_inference::own_setup_contract::trained as numeric_contract;
pub use tzolkin_inference::own_setup_contract::trained::{
    ARTIFACT_SCHEMA, MAX_BATCH_EPISODES, MAX_UPDATES, POLICY_VERSION, SAMPLING_VERSION,
    SetupUpdateSource, TASK, TrainedOwnSetupIdentity,
};
use tzolkin_inference::policy::{BorrowedSemanticSetupPolicy, NumericPolicyDistribution};

use crate::kernel::{Kernel, ResolvedKernel};
use crate::own_setup_model::{InitializedOwnSetupArtifact, OwnSetupModel, SetupCandidates};

/// Only the Setup updater creates this artifact; JSON cannot deserialize it.
/// It is separate from both initialized artifacts and any saved learning owner.
#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct TrainedOwnSetupArtifact {
    pub(crate) identity: TrainedOwnSetupIdentity,
    pub(crate) model: OwnSetupModel,
}
impl TrainedOwnSetupArtifact {
    pub fn identity(&self) -> &TrainedOwnSetupIdentity {
        &self.identity
    }
    pub fn model(&self) -> &OwnSetupModel {
        &self.model
    }
    pub(crate) fn from_update(
        mut identity: TrainedOwnSetupIdentity,
        model: OwnSetupModel,
    ) -> Result<Self, String> {
        model.validate()?;
        identity.parameter_words_checksum = word_checksum(model.parameters());
        identity.checksum.clear();
        let mut artifact = Self { identity, model };
        artifact.identity.checksum = artifact.expected_checksum()?;
        artifact.validate()?;
        Ok(artifact)
    }
    fn expected_checksum(&self) -> Result<String, String> {
        numeric_contract::expected_checksum(&self.identity, self.model.parameters())
    }
    pub fn validate(&self) -> Result<(), String> {
        validate_numeric_claim(&self.identity, &self.model)
    }
}
pub(crate) fn validate_numeric_claim(
    identity: &TrainedOwnSetupIdentity,
    model: &OwnSetupModel,
) -> Result<(), String> {
    model.validate()?;
    let initial = InitializedOwnSetupArtifact::new(identity.initial.seed)?;
    // Source reconstruction and current-host equality remain native-only.
    if identity.initial != initial.identity
        || identity.numerical_target != crate::public_stochastic_native::numerical_target()
    {
        return Err("Incompatible trained Own Setup identity/parameters/lineage".into());
    }
    numeric_contract::validate_numeric_claim(identity, model.parameters())
}
pub(crate) use numeric_contract::word_checksum;

/// One validated immutable updated Setup borrow; no raw rows or model conversion.
pub struct BorrowedTrainedOwnSetup<'a> {
    artifact: &'a TrainedOwnSetupArtifact,
    numeric: BorrowedSemanticSetupPolicy<'a>,
    kernel: ResolvedKernel,
}
impl<'a> BorrowedTrainedOwnSetup<'a> {
    pub fn new(artifact: &'a TrainedOwnSetupArtifact) -> Result<Self, String> {
        artifact.validate()?;
        Ok(Self {
            artifact,
            numeric: BorrowedSemanticSetupPolicy::from_validated_parameters(
                artifact.model.parameters(),
            )?,
            kernel: Kernel::Scalar.resolve()?,
        })
    }
    pub fn identity(&self) -> &TrainedOwnSetupIdentity {
        &self.artifact.identity
    }
    pub fn predict(
        &self,
        candidates: &SetupCandidates,
    ) -> Result<NumericPolicyDistribution, String> {
        self.numeric.predict(self.kernel, candidates.rows())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pure_numeric_claim_does_not_bypass_native_source_reconstruction() {
        // A checksum-consistent fixture is not a successful update or owner.
        let initialized = InitializedOwnSetupArtifact::new(u64::MAX).unwrap();
        let mut identity = TrainedOwnSetupIdentity {
            schema: ARTIFACT_SCHEMA.into(),
            policy_version: POLICY_VERSION.into(),
            sampling_version: SAMPLING_VERSION.into(),
            task: TASK.into(),
            parent_checksum: initialized.identity.checksum.clone(),
            initial: initialized.identity,
            update_count: 1,
            training_config_checksum: "a".repeat(64),
            playing_binding_checksum: "b".repeat(64),
            numerical_target: "scalar-f32-f64-tick53-v1:test_arch:test_os:64".into(),
            parameter_words_checksum: word_checksum(initialized.model.parameters()),
            sources: vec![SetupUpdateSource {
                episode_id: "c".repeat(64),
                record_checksum: "d".repeat(64),
            }],
            checksum: String::new(),
        };
        identity.checksum =
            numeric_contract::expected_checksum(&identity, initialized.model.parameters()).unwrap();
        numeric_contract::validate_numeric_claim(&identity, initialized.model.parameters())
            .unwrap();
        assert!(validate_numeric_claim(&identity, &initialized.model).is_err());
        let bytes = serde_json::to_vec(&identity).unwrap();
        assert!(
            String::from_utf8(bytes.clone())
                .unwrap()
                .contains("\"seed\":18446744073709551615")
        );
        let decoded: TrainedOwnSetupIdentity = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(decoded, identity);

        identity.numerical_target = crate::public_stochastic_native::numerical_target();
        identity.initial.checksum = "e".repeat(64);
        identity.parent_checksum = identity.initial.checksum.clone();
        identity.checksum =
            numeric_contract::expected_checksum(&identity, initialized.model.parameters()).unwrap();
        numeric_contract::validate_numeric_claim(&identity, initialized.model.parameters())
            .unwrap();
        assert!(validate_numeric_claim(&identity, &initialized.model).is_err());

        identity.initial.feature_schema = 2;
        identity.checksum =
            numeric_contract::expected_checksum(&identity, initialized.model.parameters()).unwrap();
        assert!(
            numeric_contract::validate_numeric_claim(&identity, initialized.model.parameters())
                .is_err()
        );
    }
}
