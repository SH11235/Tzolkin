//! Fresh, initialized schema-3 Setup inference. No training or deployment owner.
//! Features retain their raw quantities; only initialization scales those weights.
use serde::{Deserialize, Serialize};
use tzolkin_core::observation::{MOVE_SCHEMA, OBSERVATION_SCHEMA, Observation, TypedAction};
use tzolkin_core::{GameMove, Phase};
use tzolkin_inference::policy::{self, BorrowedSemanticSetupPolicy, NumericPolicyDistribution};

use crate::features::{
    EncodedCandidate, FEATURE_COUNT, FeatureEncoder, SEMANTIC_SETUP_FEATURE_SCHEMA,
    SETUP_PAIR_FEATURE_COUNT, SETUP_PAIR_FEATURE_START,
};
use crate::kernel::{Kernel, ResolvedKernel};
use crate::model::{Random, digest};
use crate::public_model::{B1, BP, HIDDEN, PARAMETER_COUNT, WP};

pub const ARTIFACT_SCHEMA: &str = "tzolkin-own-setup-initialized-model-v1";
pub const MODEL_VERSION: &str = "own-setup-quantity-512x32-tanh-v1";
pub const INPUT_CONTRACT: &str = "base-3-4p-setup-only-six-wealth-pairs-semantic-v3";
pub const TASK: &str = "initializedOwnSetupPolicyOnly";
pub const INIT_VERSION: &str = "splitmix64-glorot-quantity-weight-div17-f32-v1";
const PAIR_COUNT: usize = 6;

/// Numerical parameters only; deserialization grants no initialized/trained identity.
/// The distinct serialized field cannot deserialize an old schema-2 policy model.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct OwnSetupModel {
    pub(crate) own_setup_parameters: Vec<f32>,
}
impl OwnSetupModel {
    pub fn parameters(&self) -> &[f32] {
        &self.own_setup_parameters
    }
    pub fn validate(&self) -> Result<(), String> {
        policy::validate_parameters(self.parameters())
    }
}

fn fresh_model(seed: u64) -> OwnSetupModel {
    let mut random = Random { state: seed };
    let mut parameters = vec![0.0; PARAMETER_COUNT];
    let input_scale = (6.0 / (FEATURE_COUNT + HIDDEN) as f32).sqrt();
    for (index, value) in parameters[..B1].iter_mut().enumerate() {
        let scale = if (SETUP_PAIR_FEATURE_START
            ..SETUP_PAIR_FEATURE_START + SETUP_PAIR_FEATURE_COUNT)
            .contains(&(index % FEATURE_COUNT))
        {
            input_scale / 17.0
        } else {
            input_scale
        };
        *value = (random.unit() * 2.0 - 1.0) * scale;
    }
    let policy_scale = (6.0 / (HIDDEN + 1) as f32).sqrt();
    for value in &mut parameters[WP..BP] {
        *value = (random.unit() * 2.0 - 1.0) * policy_scale;
    }
    OwnSetupModel {
        own_setup_parameters: parameters,
    }
}

/// Fresh initialization, not a checkpoint. Its only permitted update count is zero.
/// Validation reconstructs the seeded words; a resealed update is not this artifact.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct InitializedOwnSetupArtifact {
    pub identity: InitializedOwnSetupIdentity,
    pub model: OwnSetupModel,
}
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
impl InitializedOwnSetupArtifact {
    pub fn new(seed: u64) -> Result<Self, String> {
        let mut artifact = Self {
            identity: InitializedOwnSetupIdentity {
                schema: ARTIFACT_SCHEMA.into(),
                model_version: MODEL_VERSION.into(),
                input_contract: INPUT_CONTRACT.into(),
                task: TASK.into(),
                init_version: INIT_VERSION.into(),
                seed,
                completed_updates: 0,
                feature_schema: SEMANTIC_SETUP_FEATURE_SCHEMA,
                feature_count: FEATURE_COUNT,
                hidden_count: HIDDEN,
                parameter_count: PARAMETER_COUNT,
                backend: "scalar".into(),
                rules_version: crate::replay::RULES_VERSION,
                rules_baseline: crate::replay::RULES_BASELINE.into(),
                observation_schema: OBSERVATION_SCHEMA,
                move_schema: MOVE_SCHEMA,
                catalog_hash: crate::replay::catalog_hash(),
                checksum: String::new(),
            },
            model: fresh_model(seed),
        };
        artifact.identity.checksum = artifact.expected_checksum()?;
        artifact.validate()?;
        Ok(artifact)
    }
    fn expected_checksum(&self) -> Result<String, String> {
        let mut identity = self.identity.clone();
        identity.checksum.clear();
        let mut bytes = serde_json::to_vec(&identity).map_err(|error| error.to_string())?;
        // An explicit byte boundary precedes little-endian f32 words. Checksums
        // bind actual parameter bits, including signed zero, not float formatting.
        bytes.push(0);
        bytes.extend(
            self.model
                .parameters()
                .iter()
                .flat_map(|value| value.to_le_bytes()),
        );
        Ok(digest(&bytes))
    }
    pub fn validate(&self) -> Result<(), String> {
        let id = &self.identity;
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
            || id.rules_version != crate::replay::RULES_VERSION
            || id.rules_baseline != crate::replay::RULES_BASELINE
            || id.observation_schema != OBSERVATION_SCHEMA
            || id.move_schema != MOVE_SCHEMA
            || id.catalog_hash != crate::replay::catalog_hash()
        {
            return Err("Incompatible initialized own Setup contract".into());
        }
        self.model.validate()?;
        if self
            .model
            .parameters()
            .iter()
            .zip(fresh_model(id.seed).parameters())
            .any(|(a, b)| a.to_bits() != b.to_bits())
        {
            return Err("Own Setup parameters do not match fresh seeded initialization".into());
        }
        if id.checksum.len() != 64 || id.checksum != self.expected_checksum()? {
            return Err("Initialized own Setup checksum mismatch".into());
        }
        Ok(())
    }
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

/// Validate fresh identity once and retain a borrow; inference is always Scalar.
/// No trained owner, continuation policy, fallback, or application adoption is provided.
pub struct BorrowedInitializedOwnSetup<'a> {
    artifact: &'a InitializedOwnSetupArtifact,
    numeric: BorrowedSemanticSetupPolicy<'a>,
    kernel: ResolvedKernel,
}
impl<'a> BorrowedInitializedOwnSetup<'a> {
    pub fn new(artifact: &'a InitializedOwnSetupArtifact) -> Result<Self, String> {
        artifact.validate()?;
        Ok(Self {
            artifact,
            numeric: BorrowedSemanticSetupPolicy::from_validated_parameters(
                artifact.model.parameters(),
            )?,
            kernel: Kernel::Scalar.resolve()?,
        })
    }
    pub fn identity(&self) -> &InitializedOwnSetupIdentity {
        &self.artifact.identity
    }
    pub fn predict(
        &self,
        candidates: &SetupCandidates,
    ) -> Result<NumericPolicyDistribution, String> {
        self.numeric.predict(self.kernel, candidates.rows())
    }
    pub fn distribution(
        &self,
        observation: &Observation,
    ) -> Result<NumericPolicyDistribution, String> {
        self.predict(&SetupCandidates::from_observation(observation)?)
    }
}

#[cfg(test)]
pub(crate) fn setup_fixture(players: usize, offer: [&str; 4]) -> Observation {
    let mut state = tzolkin_core::create_game(
        (0..players).map(|seat| format!("P{seat}")).collect(),
        17,
        false,
    )
    .unwrap();
    state.players[state.current_player].wealth_offer = offer.map(String::from).to_vec();
    tzolkin_core::observation::observe(&state, state.current_player).unwrap()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::own_setup_pullback::OwnSetupScalarPullback;
    use crate::public_model::{LoadedPublicPolicy, PublicPolicyArtifact};
    use crate::public_policy_pullback::ScalarPolicyPullback;
    use tzolkin_core::observation::observation_key;

    #[test]
    fn seeded_words_checksum_and_untrained_identity_are_reproducible() {
        let artifact = InitializedOwnSetupArtifact::new(11235).unwrap();
        let parameters = artifact.model.parameters();
        // Independent SplitMix64/f32 word fixture, including quantity columns.
        for (index, bits) in [
            (0, 3171472333),
            (480, 987376115),
            (493, 995808018),
            (494, 3165438780),
            (992, 998770453),
            (WP, 3173514069),
            (BP - 1, 3201331259),
            (BP, 0),
        ] {
            assert_eq!(parameters[index].to_bits(), bits);
        }
        let words: Vec<u8> = parameters
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect();
        assert_eq!(
            digest(&words),
            "4ebce8eb3f8182741e3b0afb087e85258b7b73a294796d27b14acf163c58c827"
        );
        assert_eq!(artifact, InitializedOwnSetupArtifact::new(11235).unwrap());
        assert_ne!(
            artifact.identity.checksum,
            InitializedOwnSetupArtifact::new(11236)
                .unwrap()
                .identity
                .checksum
        );
        let roundtrip: InitializedOwnSetupArtifact =
            serde_json::from_slice(&serde_json::to_vec(&artifact).unwrap()).unwrap();
        roundtrip.validate().unwrap();
        assert_eq!(roundtrip.identity.checksum, artifact.identity.checksum);
        for seed in [0, u64::MAX] {
            let fresh = InitializedOwnSetupArtifact::new(seed).unwrap();
            assert!(
                fresh
                    .model
                    .parameters()
                    .iter()
                    .all(|value| value.is_finite())
            );
            assert!(
                fresh.model.parameters()[..B1]
                    .iter()
                    .any(|value| *value != 0.0)
            );
            assert!(
                fresh.model.parameters()[WP..BP]
                    .iter()
                    .any(|value| *value != 0.0)
            );
            assert_eq!(fresh.identity.completed_updates, 0);
        }
        let mut changed = artifact.clone();
        changed.model.own_setup_parameters[480] += 0.001;
        changed.identity.checksum = changed.expected_checksum().unwrap();
        assert!(BorrowedInitializedOwnSetup::new(&changed).is_err());
        let mut changed = artifact.clone();
        changed.model.own_setup_parameters[B1] = -0.0;
        assert!(changed.validate().is_err());
        for field in ["featureSchema", "parameterCount", "completedUpdates"] {
            let mut json = serde_json::to_value(&artifact).unwrap();
            json["identity"][field] = serde_json::json!(1);
            let changed: InitializedOwnSetupArtifact = serde_json::from_value(json).unwrap();
            assert!(changed.validate().is_err());
        }
        let mut changed = artifact.clone();
        changed.identity.catalog_hash = "0".repeat(64);
        assert!(changed.validate().is_err());
        let mut changed = artifact.clone();
        changed.model.own_setup_parameters.pop();
        assert!(changed.validate().is_err());
        let mut changed = artifact.clone();
        changed.model.own_setup_parameters[0] = f32::NAN;
        assert!(changed.validate().is_err());
        let old = PublicPolicyArtifact::new(11235).unwrap();
        assert!(
            serde_json::from_slice::<OwnSetupModel>(&serde_json::to_vec(&old.model).unwrap())
                .is_err()
        );
        assert!(
            serde_json::from_slice::<PublicPolicyArtifact>(&serde_json::to_vec(&artifact).unwrap())
                .is_err()
        );
    }

    #[test]
    fn setup_factory_preserves_full_pair_order_and_rejects_other_contracts() {
        let artifact = InitializedOwnSetupArtifact::new(7).unwrap();
        let loaded = BorrowedInitializedOwnSetup::new(&artifact).unwrap();
        let old = PublicPolicyArtifact::new(7).unwrap();
        let old_loaded = LoadedPublicPolicy::new(&old).unwrap();
        for players in [3, 4] {
            let observation = setup_fixture(players, ["w01", "w14", "w15", "w17"]);
            let candidates = SetupCandidates::from_observation(&observation).unwrap();
            assert_eq!(candidates.observation_key(), observation.observation_key);
            assert_eq!(candidates.rows().len(), PAIR_COUNT);
            let predicted = loaded.predict(&candidates).unwrap();
            assert_eq!(predicted.logits.len(), PAIR_COUNT);
            assert!(predicted.logits.iter().all(|value| value.is_finite()));
            assert!(predicted.probabilities.iter().all(|value| *value > 0.0));
            assert!(old_loaded.predict(candidates.rows()).is_err());
            assert!(ScalarPolicyPullback::new(&old.model, candidates.rows()).is_err());
            for phase in [Phase::Playing, Phase::Finished] {
                let mut invalid = observation.clone();
                invalid.phase = phase;
                invalid.observation_key = observation_key(&invalid).unwrap();
                assert!(loaded.distribution(&invalid).is_err());
            }
            for alteration in 0..6 {
                let mut invalid = observation.clone();
                match alteration {
                    0 => invalid.additional_buildings = true,
                    1 => invalid.schema += 1,
                    2 => invalid.private.wealth_offer[0] = "unknown".into(),
                    3 => invalid.legal_actions.reverse(),
                    4 => {
                        invalid.legal_actions.pop();
                    }
                    _ => invalid.legal_actions[0].r#move = invalid.legal_actions[1].r#move.clone(),
                }
                invalid.observation_key = observation_key(&invalid).unwrap();
                assert!(SetupCandidates::from_observation(&invalid).is_err());
            }
            let mut stale = observation.clone();
            stale.observation_key.clear();
            assert!(SetupCandidates::from_observation(&stale).is_err());
            // Mutate only this module's private test batch to exercise tag refusal.
            let encoder = FeatureEncoder::new_public(&observation).unwrap();
            let mut wrong_tag = candidates;
            wrong_tag.rows[0] = encoder.encode_legal_tagged(0).unwrap();
            assert!(loaded.predict(&wrong_tag).is_err());
            assert!(OwnSetupScalarPullback::new(&artifact.model, &wrong_tag).is_err());
        }
    }
}
