//! Explicit schema-2 policy inference. Setup uses the actor's legitimate offers;
//! Playing excludes all private features. This model has no value head.
use std::path::Path;

use serde::{Deserialize, Serialize};
use tzolkin_core::observation::{MOVE_SCHEMA, OBSERVATION_SCHEMA, Observation};

use crate::Decision;
use crate::features::{EncodedCandidate, FEATURE_COUNT, FeatureEncoder, PUBLIC_FEATURE_SCHEMA};
use crate::kernel::{Kernel, ResolvedKernel};
use crate::model::{Random, digest, read_json, write_new_json};
use tzolkin_inference::policy::{self, BorrowedPolicy};
pub use tzolkin_inference::policy::{HIDDEN, MAX_OBSERVATION_BYTES, PARAMETER_COUNT};

pub const MODEL_SCHEMA: &str = "tzolkin-public-policy-model-v1";
pub const MODEL_VERSION: &str = "tiny-public-policy-mlp-v1";
pub const POLICY_VERSION: &str = "learned-public-policy-v1";
pub const INPUT_CONTRACT: &str = "base-3-4p-native-setup-public-playing-v2";
pub(crate) const B1: usize = FEATURE_COUNT * HIDDEN;
pub(crate) const WP: usize = B1 + HIDDEN;
pub(crate) const BP: usize = WP + HIDDEN;
#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub enum ValueValidity {
    UnavailablePolicyOnly,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct PublicPolicyModel {
    pub(crate) parameters: Vec<f32>,
}
impl PublicPolicyModel {
    pub fn parameters(&self) -> &[f32] {
        &self.parameters
    }
    pub fn validate(&self) -> Result<(), String> {
        policy::validate_parameters(&self.parameters)
    }
    pub(crate) fn validated_values(
        row: &EncodedCandidate,
    ) -> Result<&[f32; FEATURE_COUNT], String> {
        policy::validated_values(row)
    }
    fn logit(&self, row: &EncodedCandidate, kernel: ResolvedKernel) -> Result<f32, String> {
        BorrowedPolicy::from_validated_parameters(&self.parameters)?.logit(row, kernel)
    }
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct PublicPolicyArtifact {
    pub schema: String,
    pub model_version: String,
    pub policy_version: String,
    pub input_contract: String,
    pub feature_schema: u32,
    pub feature_count: usize,
    pub hidden_count: usize,
    pub parameter_count: usize,
    pub task: String,
    pub value_validity: ValueValidity,
    pub rules_version: u32,
    pub rules_baseline: String,
    pub observation_schema: u32,
    pub move_schema: u32,
    pub catalog_hash: String,
    pub model: PublicPolicyModel,
    pub checksum: String,
}
impl PublicPolicyArtifact {
    /// An initialized, untrained policy. This constructor does not train BC.
    pub fn new(seed: u64) -> Result<Self, String> {
        let mut random = Random { state: seed };
        let mut parameters = vec![0.0; PARAMETER_COUNT];
        let input_scale = (6.0 / (FEATURE_COUNT + HIDDEN) as f32).sqrt();
        for value in &mut parameters[..B1] {
            *value = (random.unit() * 2.0 - 1.0) * input_scale;
        }
        let policy_scale = (6.0 / (HIDDEN + 1) as f32).sqrt();
        for value in &mut parameters[WP..BP] {
            *value = (random.unit() * 2.0 - 1.0) * policy_scale;
        }
        let mut artifact = Self {
            schema: MODEL_SCHEMA.into(),
            model_version: MODEL_VERSION.into(),
            policy_version: POLICY_VERSION.into(),
            input_contract: INPUT_CONTRACT.into(),
            feature_schema: PUBLIC_FEATURE_SCHEMA,
            feature_count: FEATURE_COUNT,
            hidden_count: HIDDEN,
            parameter_count: PARAMETER_COUNT,
            task: "policyOnlyBc".into(),
            value_validity: ValueValidity::UnavailablePolicyOnly,
            rules_version: crate::replay::RULES_VERSION,
            rules_baseline: crate::replay::RULES_BASELINE.into(),
            observation_schema: OBSERVATION_SCHEMA,
            move_schema: MOVE_SCHEMA,
            catalog_hash: crate::replay::catalog_hash(),
            model: PublicPolicyModel { parameters },
            checksum: String::new(),
        };
        artifact.checksum = artifact.expected_checksum()?;
        artifact.validate()?;
        Ok(artifact)
    }
    fn expected_checksum(&self) -> Result<String, String> {
        let mut payload = self.clone();
        payload.checksum.clear();
        Ok(digest(
            &serde_json::to_vec(&payload).map_err(|error| error.to_string())?,
        ))
    }
    /// Training's private boundary after finite parameter updates; not a raw public constructor.
    pub(crate) fn reseal(&mut self) -> Result<(), String> {
        self.model.validate()?;
        self.checksum = self.expected_checksum()?;
        self.validate()
    }
    pub fn validate(&self) -> Result<(), String> {
        if self.schema != MODEL_SCHEMA
            || self.model_version != MODEL_VERSION
            || self.policy_version != POLICY_VERSION
            || self.input_contract != INPUT_CONTRACT
            || self.feature_schema != PUBLIC_FEATURE_SCHEMA
            || self.feature_count != FEATURE_COUNT
            || self.hidden_count != HIDDEN
            || self.parameter_count != PARAMETER_COUNT
            || self.task != "policyOnlyBc"
            || self.value_validity != ValueValidity::UnavailablePolicyOnly
            || self.rules_version != crate::replay::RULES_VERSION
            || self.rules_baseline != crate::replay::RULES_BASELINE
            || self.observation_schema != OBSERVATION_SCHEMA
            || self.move_schema != MOVE_SCHEMA
            || self.catalog_hash != crate::replay::catalog_hash()
        {
            return Err("Incompatible public policy model/features/rules contract".into());
        }
        self.model.validate()?;
        if self.checksum.len() != 64 || self.checksum != self.expected_checksum()? {
            return Err("Public policy model checksum mismatch".into());
        }
        Ok(())
    }
    pub fn load(path: &Path) -> Result<Self, String> {
        let artifact: Self = read_json(path)?;
        artifact.validate()?;
        Ok(artifact)
    }
    pub fn save_new(&self, path: &Path) -> Result<(), String> {
        self.validate()?;
        write_new_json(path, self)
    }
}

/// No utilities, value parameters or calibrated-critic conversion are available.
#[derive(Clone, Debug, Serialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct PublicPolicyDistribution {
    pub policy_version: String,
    pub feature_schema: u32,
    pub logits: Vec<f32>,
    pub probabilities: Vec<f32>,
    /// Unit serializes as JSON null and cannot hold a numeric value target.
    pub value: (),
    pub value_validity: ValueValidity,
}

/// Validate once and retain an immutable borrow. Scalar is the default.
///
/// Raw feature slices deliberately have no public prediction entrypoint.
/// ```compile_fail
/// # use tzolkin_ai::public_model::{LoadedPublicPolicy, PublicPolicyArtifact};
/// # let model = PublicPolicyArtifact::new(1).unwrap();
/// # let loaded = LoadedPublicPolicy::new(&model).unwrap();
/// loaded.predict(&[0.0_f32; 512]).unwrap();
/// ```
/// The loaded artifact cannot be mutated while its handle is used.
/// ```compile_fail
/// # use tzolkin_ai::public_model::{LoadedPublicPolicy, PublicPolicyArtifact};
/// let mut model = PublicPolicyArtifact::new(1).unwrap();
/// let loaded = LoadedPublicPolicy::new(&model).unwrap();
/// model.feature_schema = 1;
/// loaded.backend();
/// ```
pub struct LoadedPublicPolicy<'a> {
    artifact: &'a PublicPolicyArtifact,
    kernel: ResolvedKernel,
}
impl<'a> LoadedPublicPolicy<'a> {
    pub fn new(artifact: &'a PublicPolicyArtifact) -> Result<Self, String> {
        Self::with_kernel(artifact, Kernel::Scalar)
    }
    pub fn with_kernel(artifact: &'a PublicPolicyArtifact, kernel: Kernel) -> Result<Self, String> {
        artifact.validate()?;
        Ok(Self {
            artifact,
            kernel: kernel.resolve()?,
        })
    }
    /// The native prepared owner has resolved CPU support and audited its immutable model.
    /// Shape/checksum validation stays at this boundary; no raw parameter constructor is exposed.
    pub(crate) fn with_resolved(
        artifact: &'a PublicPolicyArtifact,
        kernel: ResolvedKernel,
    ) -> Result<Self, String> {
        artifact.validate()?;
        Ok(Self { artifact, kernel })
    }
    pub fn backend(&self) -> &'static str {
        self.kernel.backend()
    }
    pub fn policy_logit(&self, row: &EncodedCandidate) -> Result<f32, String> {
        self.artifact.model.logit(row, self.kernel)
    }
    /// Classify a supplied ordered candidate batch, not an authenticated legal set.
    pub fn predict(&self, rows: &[EncodedCandidate]) -> Result<PublicPolicyDistribution, String> {
        let numeric = predict_numeric(&self.artifact.model, self.kernel, rows)?;
        Ok(PublicPolicyDistribution {
            policy_version: POLICY_VERSION.into(),
            feature_schema: PUBLIC_FEATURE_SCHEMA,
            logits: numeric.logits,
            probabilities: numeric.probabilities,
            value: (),
            value_validity: ValueValidity::UnavailablePolicyOnly,
        })
    }
    pub fn distribution(
        &self,
        observation: &Observation,
    ) -> Result<PublicPolicyDistribution, String> {
        validate_contract(observation)?;
        let encoder = FeatureEncoder::new_public(observation)?;
        let rows = (0..observation.legal_actions.len())
            .map(|index| encoder.encode_legal_tagged(index))
            .collect::<Result<Vec<_>, _>>()?;
        self.predict(&rows)
    }
    pub fn choose_move(&self, observation: &Observation) -> Result<Decision, String> {
        let prediction = self.distribution(observation)?;
        let best = (1..prediction.logits.len()).fold(0, |best, index| {
            if prediction.logits[index] > prediction.logits[best] {
                index
            } else {
                best
            }
        });
        Ok(Decision {
            actor: observation.actor,
            observation_key: observation.observation_key.clone(),
            policy_version: POLICY_VERSION.into(),
            r#move: observation.legal_actions[best].r#move.clone(),
            score: f64::from(prediction.logits[best]),
        })
    }
}

// Keep AI-only callers on the same numeric signature and concrete shared result.
pub(crate) use tzolkin_inference::policy::NumericPolicyDistribution;
pub(crate) fn predict_numeric(
    model: &PublicPolicyModel,
    kernel: ResolvedKernel,
    rows: &[EncodedCandidate],
) -> Result<NumericPolicyDistribution, String> {
    BorrowedPolicy::from_validated_parameters(&model.parameters)?.predict(kernel, rows)
}

pub(crate) fn validate_contract(o: &Observation) -> Result<(), String> {
    policy::validate_contract(o)
}
