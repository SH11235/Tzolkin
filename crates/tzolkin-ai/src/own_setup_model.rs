//! Fresh, initialized schema-3 Setup inference. No training or deployment owner.
//! Features retain their raw quantities; only initialization scales those weights.
use serde::{Deserialize, Serialize};
use tzolkin_core::observation::{MOVE_SCHEMA, OBSERVATION_SCHEMA, Observation};
use tzolkin_inference::policy::{self, BorrowedSemanticSetupPolicy, NumericPolicyDistribution};

use crate::features::{
    FEATURE_COUNT, SEMANTIC_SETUP_FEATURE_SCHEMA, SETUP_PAIR_FEATURE_COUNT,
    SETUP_PAIR_FEATURE_START,
};
use crate::kernel::{Kernel, ResolvedKernel};
use crate::model::Random;
use crate::public_model::{B1, BP, HIDDEN, PARAMETER_COUNT, WP};

pub use tzolkin_inference::own_setup_contract::{
    ARTIFACT_SCHEMA, INIT_VERSION, INPUT_CONTRACT, InitializedOwnSetupIdentity, MODEL_VERSION,
    SetupCandidates, TASK,
};
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
        tzolkin_inference::own_setup_contract::initialized_checksum(
            &self.identity,
            self.model.parameters(),
        )
    }
    pub fn validate(&self) -> Result<(), String> {
        let id = &self.identity;
        tzolkin_inference::own_setup_contract::validate_initialized_contract(id)?;
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
    use crate::model::digest;
    use tzolkin_core::Phase;
    const PAIR_COUNT: usize = 6;
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
            // Private batch tag refusal lives with the inference-owned factory.
        }
    }
}
