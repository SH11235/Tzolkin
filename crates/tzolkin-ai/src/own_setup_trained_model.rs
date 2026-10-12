//! Updated Setup parameters with their own source lineage and Scalar identity.
//! A serialized identity is not a learning owner or a restore certificate.
use serde::{Deserialize, Serialize};
use tzolkin_inference::policy::{BorrowedSemanticSetupPolicy, NumericPolicyDistribution};

use crate::kernel::{Kernel, ResolvedKernel};
use crate::model::digest;
use crate::own_setup_model::{
    InitializedOwnSetupArtifact, InitializedOwnSetupIdentity, OwnSetupModel, SetupCandidates,
};
use crate::public_policy_update::checkpoint_digest;

pub const ARTIFACT_SCHEMA: &str = "tzolkin-own-setup-trained-model-v1";
pub const POLICY_VERSION: &str = "own-setup-trained-semantic-scalar-argmax-v1";
pub const SAMPLING_VERSION: &str = "own-setup-trained-semantic-scalar-uniform-tick53-v2";
pub const TASK: &str = "ownSetupWinnerShareBaselineUniformCorrectedAdamV1";
pub const MAX_UPDATES: u64 = 1_000_000;
pub const MAX_BATCH_EPISODES: usize = 32;

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct SetupUpdateSource {
    pub episode_id: String,
    pub record_checksum: String,
}
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct TrainedOwnSetupIdentity {
    pub schema: String,
    pub policy_version: String,
    pub sampling_version: String,
    pub task: String,
    pub initial: InitializedOwnSetupIdentity,
    pub parent_checksum: String,
    pub update_count: u64,
    pub training_config_checksum: String,
    pub playing_binding_checksum: String,
    pub numerical_target: String,
    pub parameter_words_checksum: String,
    pub sources: Vec<SetupUpdateSource>,
    pub checksum: String,
}

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
        expected_checksum(&self.identity, &self.model)
    }
    pub fn validate(&self) -> Result<(), String> {
        validate_numeric_claim(&self.identity, &self.model)
    }
}
fn expected_checksum(
    identity: &TrainedOwnSetupIdentity,
    model: &OwnSetupModel,
) -> Result<String, String> {
    let mut identity = identity.clone();
    identity.checksum.clear();
    let mut bytes = b"tzolkin-own-setup-trained-model-v1\0".to_vec();
    bytes.extend(serde_json::to_vec(&identity).map_err(|e| e.to_string())?);
    bytes.push(0);
    bytes.extend(model.parameters().iter().flat_map(|p| p.to_le_bytes()));
    Ok(digest(&bytes))
}

pub(crate) fn validate_numeric_claim(
    i: &TrainedOwnSetupIdentity,
    model: &OwnSetupModel,
) -> Result<(), String> {
    model.validate()?;
    let initial = InitializedOwnSetupArtifact::new(i.initial.seed)?;
    if i.schema != ARTIFACT_SCHEMA
        || i.policy_version != POLICY_VERSION
        || i.sampling_version != SAMPLING_VERSION
        || i.task != TASK
        || !(1..=MAX_UPDATES).contains(&i.update_count)
        || i.initial != initial.identity
        || !checkpoint_digest(&i.parent_checksum)
        || (i.update_count == 1 && i.parent_checksum != i.initial.checksum)
        || !checkpoint_digest(&i.training_config_checksum)
        || !checkpoint_digest(&i.playing_binding_checksum)
        || i.numerical_target != crate::public_stochastic_native::numerical_target()
        || i.parameter_words_checksum != word_checksum(model.parameters())
        || !(1..=MAX_BATCH_EPISODES).contains(&i.sources.len())
        || i.sources
            .iter()
            .any(|s| !checkpoint_digest(&s.episode_id) || !checkpoint_digest(&s.record_checksum))
        || i.sources.iter().enumerate().any(|(index, s)| {
            i.sources[..index].iter().any(|old| {
                old.episode_id == s.episode_id || old.record_checksum == s.record_checksum
            })
        })
        || i.checksum != expected_checksum(i, model)?
    {
        return Err("Incompatible trained Own Setup identity/parameters/lineage".into());
    }
    Ok(())
}
pub(crate) fn word_checksum(parameters: &[f32]) -> String {
    digest(
        &parameters
            .iter()
            .flat_map(|p| p.to_le_bytes())
            .collect::<Vec<_>>(),
    )
}

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
