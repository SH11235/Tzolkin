//! Scalar inference from a sealed count1 RL update, with a distinct policy
//! identity. No BC resealing, loading, sampling, training admission or file I/O.
use serde::Serialize;
use tzolkin_core::observation::Observation;

use crate::Decision;
use crate::features::{FeatureEncoder, PUBLIC_FEATURE_SCHEMA};
use crate::kernel::{Kernel, ResolvedKernel};
use crate::public_model::{self, ValueValidity};
use crate::public_policy_update::UpdatedPublicRlPolicy;

pub const POLICY_VERSION: &str = "learned-public-rl-one-step-v1";

/// Prediction metadata is distinct from BC. The null value cannot carry a
/// critic estimate; this distribution does not authenticate a stored record.
#[derive(Debug, Serialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct PublicRlPolicyDistribution {
    policy_version: &'static str,
    artifact_checksum: String,
    update_count: u64,
    feature_schema: u32,
    backend: &'static str,
    logits: Vec<f32>,
    probabilities: Vec<f32>,
    value: (),
    value_validity: ValueValidity,
}
impl PublicRlPolicyDistribution {
    pub fn policy_version(&self) -> &'static str {
        self.policy_version
    }
    pub fn artifact_checksum(&self) -> &str {
        &self.artifact_checksum
    }
    pub fn update_count(&self) -> u64 {
        self.update_count
    }
    pub fn feature_schema(&self) -> u32 {
        self.feature_schema
    }
    pub fn backend(&self) -> &'static str {
        self.backend
    }
    pub fn logits(&self) -> &[f32] {
        &self.logits
    }
    pub fn probabilities(&self) -> &[f32] {
        &self.probabilities
    }
    pub fn value_validity(&self) -> ValueValidity {
        self.value_validity
    }
}

/// Only the controlled update owner can construct this immutable handle.
/// No arbitrary model, artifact DTO or supplied feature-row constructor exists.
/// The owner is checked once; every call validates and encodes its full
/// Observation legal order. Scalar and same numerical target are required.
pub struct UpdatedPublicRlHandle<'a> {
    owner: &'a UpdatedPublicRlPolicy,
    kernel: ResolvedKernel,
}
impl<'a> UpdatedPublicRlHandle<'a> {
    pub fn new(owner: &'a UpdatedPublicRlPolicy) -> Result<Self, String> {
        owner.validate_for_inference()?;
        Ok(Self {
            owner,
            kernel: Kernel::Scalar.resolve()?,
        })
    }
    pub fn artifact_checksum(&self) -> &str {
        self.owner.artifact().checksum()
    }
    pub fn update_count(&self) -> u64 {
        self.owner.artifact().update_count()
    }
    pub fn backend(&self) -> &'static str {
        self.kernel.backend()
    }
    pub(crate) fn task(&self) -> &str {
        self.owner.artifact().task()
    }
    pub(crate) fn family_closure(&self) -> &[String] {
        self.owner.artifact().report().family_closure()
    }
    pub fn distribution(
        &self,
        observation: &Observation,
    ) -> Result<PublicRlPolicyDistribution, String> {
        public_model::validate_contract(observation)?;
        let encoder = FeatureEncoder::new_public(observation)?;
        let rows = (0..observation.legal_actions.len())
            .map(|index| encoder.encode_legal_tagged(index))
            .collect::<Result<Vec<_>, _>>()?;
        let numeric = public_model::predict_numeric(self.owner.model(), self.kernel, &rows)?;
        Ok(PublicRlPolicyDistribution {
            policy_version: POLICY_VERSION,
            artifact_checksum: self.artifact_checksum().into(),
            update_count: self.update_count(),
            feature_schema: PUBLIC_FEATURE_SCHEMA,
            backend: self.backend(),
            logits: numeric.logits,
            probabilities: numeric.probabilities,
            value: (),
            value_validity: ValueValidity::UnavailablePolicyOnly,
        })
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
