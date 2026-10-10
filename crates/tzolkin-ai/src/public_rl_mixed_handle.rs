//! Inference-only view of a genuine mixed actor/residual owner. Metadata DTOs
//! and standalone samples grant no collection or durable training admission.
use crate::Decision;
use crate::features::{FeatureEncoder, PUBLIC_FEATURE_SCHEMA};
use crate::kernel::{Kernel, ResolvedKernel};
use crate::public_model::{self, ValueValidity};
use crate::public_policy_mixed::MixedJointOwner;
use crate::public_policy_update::{checkpoint_digest, hash};
use crate::public_state_critic;
use crate::public_stochastic_native::numerical_target;
use crate::replay::SeatPolicy;
use serde::{Deserialize, Serialize};
use tzolkin_core::observation::Observation;

pub const POLICY_VERSION: &str = "learned-public-rl-mixed-joint-argmax-v1";
pub const SAMPLING_VERSION: &str = "public-stochastic-rl-mixed-joint-uniform-tick53-v1";
pub const IDENTITY_SCHEMA: &str = "tzolkin-public-rl-mixed-joint-inference-identity-v1";

/// Closed content identity, never an executable owner or sampler constructor.
/// Original bytes, producer and independent RNG origin remain caller concerns.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct MixedInferenceIdentity {
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
}
impl MixedInferenceIdentity {
    fn from_owner(owner: &MixedJointOwner) -> Result<Self, String> {
        owner.validate_for_inference()?;
        let identity = Self {
            schema: IDENTITY_SCHEMA.into(),
            policy_version: POLICY_VERSION.into(),
            sampling_version: SAMPLING_VERSION.into(),
            task: crate::public_policy_mixed::TASK.into(),
            artifact_checksum: owner.artifact().checksum().into(),
            imported_update_count: owner.imported_update_count(),
            mixed_update_count: owner.mixed_update_count(),
            actor_model_version: public_model::MODEL_VERSION.into(),
            critic_model_version: public_state_critic::MODEL_VERSION.into(),
            actor_parameters_checksum: hash(
                b"tzolkin-mixed-inference-actor-v1\0",
                &owner.actor_model().parameters(),
            )?,
            residual_parameters_checksum: hash(
                b"tzolkin-mixed-inference-residual-v1\0",
                &owner.residual_parameters(),
            )?,
            feature_schema: PUBLIC_FEATURE_SCHEMA,
            input_contract: public_model::INPUT_CONTRACT.into(),
            context_schema: public_state_critic::CONTEXT_SCHEMA,
            context_contract: public_state_critic::CONTEXT_CONTRACT.into(),
            backend: "scalar".into(),
            numerical_target: numerical_target(),
        };
        identity.validate()?;
        Ok(identity)
    }
    /// Validates content consistency only; it cannot authenticate or mint an owner.
    pub fn validate(&self) -> Result<(), String> {
        if self.schema != IDENTITY_SCHEMA
            || self.policy_version != POLICY_VERSION
            || self.sampling_version != SAMPLING_VERSION
            || self.task != crate::public_policy_mixed::TASK
            || !(1..=crate::public_policy_paired_long::MAX_UPDATES)
                .contains(&self.imported_update_count)
            || self.mixed_update_count != 1
            || self.actor_model_version != public_model::MODEL_VERSION
            || self.critic_model_version != public_state_critic::MODEL_VERSION
            || ![
                &self.artifact_checksum,
                &self.actor_parameters_checksum,
                &self.residual_parameters_checksum,
            ]
            .into_iter()
            .all(|s| checkpoint_digest(s))
            || self.feature_schema != PUBLIC_FEATURE_SCHEMA
            || self.input_contract != public_model::INPUT_CONTRACT
            || self.context_schema != public_state_critic::CONTEXT_SCHEMA
            || self.context_contract != public_state_critic::CONTEXT_CONTRACT
            || self.backend != "scalar"
            || self.numerical_target != numerical_target()
        {
            return Err("Unsupported mixed inference identity/version/count/target".into());
        }
        Ok(())
    }
    pub fn artifact_checksum(&self) -> &str {
        &self.artifact_checksum
    }
    pub fn imported_update_count(&self) -> u64 {
        self.imported_update_count
    }
    pub fn mixed_update_count(&self) -> u64 {
        self.mixed_update_count
    }
    pub fn policy_version(&self) -> &str {
        &self.policy_version
    }
    pub fn actor_parameters_checksum(&self) -> &str {
        &self.actor_parameters_checksum
    }
    pub fn residual_parameters_checksum(&self) -> &str {
        &self.residual_parameters_checksum
    }
    pub fn binding_key(&self) -> Result<String, String> {
        self.validate()?;
        hash(b"tzolkin-mixed-inference-sampling-binding-v1\0", self)
    }
}

/// The residual is retained in identity, never emitted as a game value.
#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct MixedPolicyDistribution {
    identity: MixedInferenceIdentity,
    logits: Vec<f32>,
    probabilities: Vec<f32>,
    value: (),
    value_validity: ValueValidity,
}
impl MixedPolicyDistribution {
    pub fn identity(&self) -> &MixedInferenceIdentity {
        &self.identity
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

/// Only an immutable genuine joint owner constructs this handle. There is no
/// raw artifact/model/identity constructor, Deserialize, or continuation token.
pub struct MixedPolicyHandle<'a> {
    owner: &'a MixedJointOwner,
    identity: MixedInferenceIdentity,
    kernel: ResolvedKernel,
}
impl<'a> MixedPolicyHandle<'a> {
    pub fn new(owner: &'a MixedJointOwner) -> Result<Self, String> {
        Ok(Self {
            owner,
            identity: MixedInferenceIdentity::from_owner(owner)?,
            kernel: Kernel::Scalar.resolve()?,
        })
    }
    pub fn identity(&self) -> &MixedInferenceIdentity {
        &self.identity
    }
    pub fn backend(&self) -> &'static str {
        self.kernel.backend()
    }
    pub(crate) fn family_closure(&self) -> impl Iterator<Item = &String> {
        self.owner.family_closure()
    }
    pub fn provenance(&self) -> SeatPolicy {
        SeatPolicy::PublicRlMixed {
            identity: self.identity.clone(),
            selection_version: crate::replay::RL_ARGMAX_SELECTION_VERSION.into(),
            guard: (),
        }
    }
    pub fn distribution(
        &self,
        observation: &Observation,
    ) -> Result<MixedPolicyDistribution, String> {
        public_model::validate_contract(observation)?;
        let encoder = FeatureEncoder::new_public(observation)?;
        let rows = (0..observation.legal_actions.len())
            .map(|index| encoder.encode_legal_tagged(index))
            .collect::<Result<Vec<_>, _>>()?;
        let numeric = public_model::predict_numeric(self.owner.actor_model(), self.kernel, &rows)?;
        Ok(MixedPolicyDistribution {
            identity: self.identity.clone(),
            logits: numeric.logits,
            probabilities: numeric.probabilities,
            value: (),
            value_validity: ValueValidity::UnavailablePolicyOnly,
        })
    }
    /// Frozen ParentArgmax/evaluation follows the first maximum in legal order.
    pub fn choose_move(&self, observation: &Observation) -> Result<Decision, String> {
        let p = self.distribution(observation)?;
        let best = (1..p.logits.len()).fold(0, |best, index| {
            if p.logits[index] > p.logits[best] {
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
            score: f64::from(p.logits[best]),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn metadata() -> MixedInferenceIdentity {
        MixedInferenceIdentity {
            schema: IDENTITY_SCHEMA.into(),
            policy_version: POLICY_VERSION.into(),
            sampling_version: SAMPLING_VERSION.into(),
            task: crate::public_policy_mixed::TASK.into(),
            artifact_checksum: "a".repeat(64),
            imported_update_count: 1,
            mixed_update_count: 1,
            actor_model_version: public_model::MODEL_VERSION.into(),
            critic_model_version: public_state_critic::MODEL_VERSION.into(),
            actor_parameters_checksum: "b".repeat(64),
            residual_parameters_checksum: "c".repeat(64),
            feature_schema: PUBLIC_FEATURE_SCHEMA,
            input_contract: public_model::INPUT_CONTRACT.into(),
            context_schema: public_state_critic::CONTEXT_SCHEMA,
            context_contract: public_state_critic::CONTEXT_CONTRACT.into(),
            backend: "scalar".into(),
            numerical_target: numerical_target(),
        }
    }
    #[test]
    fn mixed_identity_is_closed_and_binds_both_counts_and_actual_parameter_fingerprints() {
        let identity = metadata();
        identity.validate().unwrap();
        let original = identity.binding_key().unwrap();
        for field in [
            "artifactChecksum",
            "actorParametersChecksum",
            "residualParametersChecksum",
        ] {
            let mut wire = serde_json::to_value(&identity).unwrap();
            wire[field] = serde_json::json!("d".repeat(64));
            let changed: MixedInferenceIdentity = serde_json::from_value(wire).unwrap();
            assert_ne!(changed.binding_key().unwrap(), original);
        }
        for (imported, mixed) in [
            (0, 1),
            (1, 0),
            (1, 2),
            (crate::public_policy_paired_long::MAX_UPDATES + 1, 1),
        ] {
            let mut invalid = metadata();
            invalid.imported_update_count = imported;
            invalid.mixed_update_count = mixed;
            assert!(invalid.validate().is_err());
        }
        let mut paired = metadata();
        paired.imported_update_count = 2;
        assert_ne!(paired.binding_key().unwrap(), original);
        for (field, value) in [
            ("schema", "old-v1"),
            ("policyVersion", crate::public_rl_native::POLICY_VERSION),
            (
                "samplingVersion",
                crate::public_stochastic::RL_SAMPLING_VERSION,
            ),
            ("task", crate::public_policy_paired_long::TASK),
            ("backend", "auto"),
        ] {
            let mut wire = serde_json::to_value(&identity).unwrap();
            wire[field] = value.into();
            assert!(
                serde_json::from_value::<MixedInferenceIdentity>(wire)
                    .unwrap()
                    .validate()
                    .is_err()
            );
        }
        let mut wire = serde_json::to_value(&identity).unwrap();
        wire["allowContinuation"] = true.into();
        assert!(serde_json::from_value::<MixedInferenceIdentity>(wire).is_err());
    }
    #[test]
    fn mixed_evaluation_metadata_never_enters_legacy_learner_or_training_identity() {
        let policy = SeatPolicy::PublicRlMixed {
            identity: metadata(),
            selection_version: crate::replay::RL_ARGMAX_SELECTION_VERSION.into(),
            guard: (),
        };
        policy.validate().unwrap();
        assert!(policy.is_rl_evaluation());
        let wire = serde_json::to_value(&policy).unwrap();
        assert_eq!(wire["kind"], "publicRlMixed");
        let decoded: SeatPolicy = serde_json::from_value(wire.clone()).unwrap();
        assert_eq!(decoded, policy);
        assert!(!crate::public_rl_mixed_native::legacy_opponent_identity_allowed(&policy));
        let mut legacy = wire.clone();
        legacy["kind"] = "publicRlPaired".into();
        assert!(serde_json::from_value::<SeatPolicy>(legacy).is_err());
        assert!(
            serde_json::from_value::<crate::public_rl_mixed_native::MixedLearnerIdentity>(
                serde_json::json!({"kind":"mixed", "policy":metadata()})
            )
            .is_err()
        );
        let mut invalid = wire;
        invalid["guard"] = false.into();
        assert!(serde_json::from_value::<SeatPolicy>(invalid).is_err());
    }
}
