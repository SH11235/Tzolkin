//! Pure schema-3 Setup contracts and canonical observation candidates.
//! Numeric metadata/checksums do not authenticate initialization/episodes or
//! grant trained ownership, optimizer state, resume, deployment or app admission.
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use tzolkin_core::observation::{MOVE_SCHEMA, OBSERVATION_SCHEMA, Observation, TypedAction};
use tzolkin_core::{GameMove, Phase};

use crate::features::{
    EncodedCandidate, FEATURE_COUNT, FeatureEncoder, SEMANTIC_SETUP_FEATURE_SCHEMA,
};
use crate::policy::{self, HIDDEN, PARAMETER_COUNT};

pub const ARTIFACT_SCHEMA: &str = "tzolkin-own-setup-initialized-model-v1";
pub const MODEL_VERSION: &str = "own-setup-quantity-512x32-tanh-v1";
pub const INPUT_CONTRACT: &str = "base-3-4p-setup-only-six-wealth-pairs-semantic-v3";
pub const TASK: &str = "initializedOwnSetupPolicyOnly";
pub const INIT_VERSION: &str = "splitmix64-glorot-quantity-weight-div17-f32-v1";
const PAIR_COUNT: usize = 6;

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct InitializedOwnSetupIdentity {
    pub schema: String,
    pub model_version: String,
    pub input_contract: String,
    pub task: String,
    pub init_version: String,
    pub seed: u64,
    pub completed_updates: u64,
    pub feature_schema: u32,
    pub feature_count: usize,
    pub hidden_count: usize,
    pub parameter_count: usize,
    pub backend: String,
    pub rules_version: u32,
    pub rules_baseline: String,
    pub observation_schema: u32,
    pub move_schema: u32,
    pub catalog_hash: String,
    pub checksum: String,
}
/// Check initialized metadata layout, not seeded-word authenticity.
/// Seed regeneration and admission remain in AI.
pub fn validate_initialized_contract(id: &InitializedOwnSetupIdentity) -> Result<(), String> {
    if id.schema != ARTIFACT_SCHEMA
        || id.model_version != MODEL_VERSION
        || id.input_contract != INPUT_CONTRACT
        || id.task != TASK
        || id.init_version != INIT_VERSION
        || id.completed_updates != 0
        || id.feature_schema != SEMANTIC_SETUP_FEATURE_SCHEMA
        || id.feature_count != FEATURE_COUNT
        || id.hidden_count != HIDDEN
        || id.parameter_count != PARAMETER_COUNT
        || id.backend != "scalar"
        || id.rules_version != crate::deployment::RULES_VERSION
        || id.rules_baseline != crate::deployment::RULES_BASELINE
        || id.observation_schema != OBSERVATION_SCHEMA
        || id.move_schema != MOVE_SCHEMA
        || id.catalog_hash != crate::deployment::catalog_hash()
    {
        return Err("Incompatible initialized own Setup contract".into());
    }
    Ok(())
}

/// Content checksum of supplied metadata/words only, without source admission.
pub fn initialized_checksum(
    identity: &InitializedOwnSetupIdentity,
    parameters: &[f32],
) -> Result<String, String> {
    let mut identity = identity.clone();
    identity.checksum.clear();
    let mut bytes = serde_json::to_vec(&identity).map_err(|error| error.to_string())?;
    // An explicit byte boundary precedes little-endian f32 words. Checksums
    // bind actual parameter bits, including signed zero, not float formatting.
    bytes.push(0);
    bytes.extend(parameters.iter().flat_map(|value| value.to_le_bytes()));
    Ok(hash(&bytes))
}

/// Immutable full six-pair numeric batch from one self-consistent observation.
/// This checks the native Setup input/order, not source audit or learning admission.
/// No constructor accepts arbitrary rows or a caller's claimed source identity.
pub struct SetupCandidates {
    observation_key: String,
    rows: Vec<EncodedCandidate>,
}
impl SetupCandidates {
    pub fn from_observation(observation: &Observation) -> Result<Self, String> {
        if observation.phase != Phase::Setup
            || observation.pending_task.is_some()
            || !observation.private.selected_wealth.is_empty()
            || observation.legal_actions.len() != PAIR_COUNT
        {
            return Err("Own Setup requires an unselected six-pair Setup observation".into());
        }
        let encoder = FeatureEncoder::new_semantic_setup(observation)?;
        let offer = &observation.private.wealth_offer;
        let mut index = 0;
        for a in 0..offer.len() {
            for b in a + 1..offer.len() {
                let legal = &observation.legal_actions[index];
                let ids = vec![offer[a].clone(), offer[b].clone()];
                if legal.action != (TypedAction::ChooseWealth { ids })
                    || legal.r#move
                        != (GameMove::Choose {
                            choice_id: format!("wealth:{}:{}", offer[a], offer[b]),
                        })
                {
                    return Err("Own Setup pair order/action/move mismatch".into());
                }
                index += 1;
            }
        }
        let rows = (0..PAIR_COUNT)
            .map(|index| {
                let row = encoder.encode_legal_tagged(index)?;
                policy::validated_semantic_setup_values(&row)?;
                Ok(row)
            })
            .collect::<Result<Vec<_>, String>>()?;
        Ok(Self {
            observation_key: observation.observation_key.clone(),
            rows,
        })
    }
    pub fn observation_key(&self) -> &str {
        &self.observation_key
    }
    pub fn rows(&self) -> &[EncodedCandidate] {
        &self.rows
    }
}

fn hash(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

/// Updated metadata is a numeric claim only. This module cannot construct the
/// AI's Serialize-only trained artifact or any learning/restore owner.
pub mod trained {
    use super::{InitializedOwnSetupIdentity, hash, policy, validate_initialized_contract};
    use crate::deployment::digest;
    use serde::{Deserialize, Serialize};

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

    /// Compute content identity only; this does not seal a trained artifact.
    pub fn expected_checksum(
        identity: &TrainedOwnSetupIdentity,
        parameters: &[f32],
    ) -> Result<String, String> {
        let mut identity = identity.clone();
        identity.checksum.clear();
        let mut bytes = b"tzolkin-own-setup-trained-model-v1\0".to_vec();
        bytes.extend(serde_json::to_vec(&identity).map_err(|e| e.to_string())?);
        bytes.push(0);
        bytes.extend(parameters.iter().flat_map(|p| p.to_le_bytes()));
        Ok(hash(&bytes))
    }

    /// Check metadata/word consistency. The seed and source lineage are claims;
    /// native reconstruction, target equality and owner admission remain in AI.
    pub fn validate_numeric_claim(
        i: &TrainedOwnSetupIdentity,
        parameters: &[f32],
    ) -> Result<(), String> {
        policy::validate_parameters(parameters)?;
        validate_initialized_contract(&i.initial)?;
        if i.schema != ARTIFACT_SCHEMA
            || i.policy_version != POLICY_VERSION
            || i.sampling_version != SAMPLING_VERSION
            || i.task != TASK
            || !(1..=MAX_UPDATES).contains(&i.update_count)
            || !digest(&i.initial.checksum)
            || !digest(&i.parent_checksum)
            || (i.update_count == 1 && i.parent_checksum != i.initial.checksum)
            || !digest(&i.training_config_checksum)
            || !digest(&i.playing_binding_checksum)
            || !crate::deployment::valid_source_target(&i.numerical_target)
            || i.parameter_words_checksum != word_checksum(parameters)
            || !(1..=MAX_BATCH_EPISODES).contains(&i.sources.len())
            || i.sources
                .iter()
                .any(|s| !digest(&s.episode_id) || !digest(&s.record_checksum))
            || i.sources.iter().enumerate().any(|(index, s)| {
                i.sources[..index].iter().any(|old| {
                    old.episode_id == s.episode_id || old.record_checksum == s.record_checksum
                })
            })
            || i.checksum != expected_checksum(i, parameters)?
        {
            return Err("Incompatible trained Own Setup identity/parameters/lineage".into());
        }
        Ok(())
    }
    pub fn word_checksum(parameters: &[f32]) -> String {
        hash(
            &parameters
                .iter()
                .flat_map(|p| p.to_le_bytes())
                .collect::<Vec<_>>(),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernel::Kernel;
    use crate::policy::BorrowedSemanticSetupPolicy;

    #[test]
    fn moved_private_batch_still_refuses_schema_two_rows() {
        let parameters = vec![0.0; PARAMETER_COUNT];
        let numeric = BorrowedSemanticSetupPolicy::from_validated_parameters(&parameters).unwrap();
        for players in [3, 4] {
            let state = tzolkin_core::create_game(
                (0..players).map(|seat| format!("P{seat}")).collect(),
                17,
                false,
            )
            .unwrap();
            let observation =
                tzolkin_core::observation::observe(&state, state.current_player).unwrap();
            let mut candidates = SetupCandidates::from_observation(&observation).unwrap();
            candidates.rows[0] = FeatureEncoder::new_public(&observation)
                .unwrap()
                .encode_legal_tagged(0)
                .unwrap();
            assert!(policy::validated_semantic_setup_values(&candidates.rows[0]).is_err());
            assert!(
                numeric
                    .predict(Kernel::Scalar.resolve().unwrap(), candidates.rows())
                    .is_err()
            );
        }
    }
}
