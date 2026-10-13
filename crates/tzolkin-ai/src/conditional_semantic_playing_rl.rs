//! Fresh schema-4 Playing ownership from a successfully loaded conditional origin.
//! This sealed type has no sampler, collector, update, journal, export or resume
//! conversion. Numeric consistency and raw custody do not authenticate sources.
use std::collections::BTreeMap;

use sha2::{Digest, Sha256};
use tzolkin_core::observation::Observation;
use tzolkin_inference::policy::NumericPolicyDistribution;

use crate::dataset::DatasetSplit;
use crate::features::{FeatureEncoder, PUBLIC_FEATURE_SCHEMA, SEMANTIC_PLAYING_FEATURE_SCHEMA};
use crate::human_initial_wealth::ContentPin;
use crate::human_policy_dataset::ConditionalFeatureProfile;
use crate::human_policy_training::{ConditionalDatasetIdentity, PreparedConditionalHumanPolicy};
use crate::kernel::{Kernel, ResolvedKernel};
use crate::public_model::{PublicPolicyModel, ValueValidity, predict_semantic_playing_numeric};
use crate::public_policy_update::hash;
use crate::public_state_critic as critic;

pub const POLICY_VERSION: &str = "conditional-human-semantic-playing-fresh-scalar-v1";
pub const INPUT_CONTRACT: &str =
    "conditional-human-semantic-playing-origin-base-3-4p-schema4-numeric-v1";
pub const CRITIC_INIT_VERSION: &str = "semantic-playing-fresh-public384-zero-residual-v1";

/// Only the successful schema-4 production loader supplies this captured closure.
/// No raw constructor, Deserialize or legacy-owner conversion is available.
pub struct SemanticPlayingRlOrigin {
    checkpoint_checksum: String,
    model_checksum: String,
    dataset: ConditionalDatasetIdentity,
    table_reservations: BTreeMap<String, DatasetSplit>,
    source_pins: Vec<ContentPin>,
    checksum: String,
}
impl SemanticPlayingRlOrigin {
    fn from_prepared(prepared: &PreparedConditionalHumanPolicy) -> Result<Self, String> {
        if prepared.identity().feature_profile()? != ConditionalFeatureProfile::SemanticPlaying4 {
            return Err("Semantic Playing RL requires original schema-4 preparation".into());
        }
        prepared.model().validate_semantic_playing()?;
        let source_pins = prepared.loaded_semantic_playing_sources()?.to_vec();
        prepared.recheck_sources()?;
        let mut origin = Self {
            checkpoint_checksum: prepared.checkpoint_checksum().into(),
            model_checksum: prepared.model().checksum.clone(),
            dataset: prepared.identity().clone(),
            table_reservations: prepared.reservations().clone(),
            source_pins,
            checksum: String::new(),
        };
        origin.checksum = hash(
            b"tzolkin-semantic-playing-rl-loaded-origin-v1\0",
            &(
                &origin.checkpoint_checksum,
                &origin.model_checksum,
                &origin.dataset,
                &origin.table_reservations,
                &origin.source_pins,
            ),
        )?;
        Ok(origin)
    }
    pub fn recheck_sources(&self) -> Result<(), String> {
        for pin in &self.source_pins {
            crate::human_policy_dataset::recheck_absolute_pin(pin)?;
        }
        Ok(())
    }
    pub fn checksum(&self) -> Result<&str, String> {
        self.recheck_sources()?;
        Ok(&self.checksum)
    }
    pub fn dataset(&self) -> Result<&ConditionalDatasetIdentity, String> {
        self.recheck_sources()?;
        Ok(&self.dataset)
    }
    /// Includes all unloaded, zero-row, Validation and Test table declarations.
    /// These Human table IDs are not native environment-family or stream IDs.
    pub fn table_reservations(&self) -> Result<&BTreeMap<String, DatasetSplit>, String> {
        self.recheck_sources()?;
        Ok(&self.table_reservations)
    }
    pub fn source_pins(&self) -> Result<&[ContentPin], String> {
        self.recheck_sources()?;
        Ok(&self.source_pins)
    }
}

/// Descriptive metadata only; it cannot be deserialized or used to create an owner.
/// The actor is schema 4. Its independently initialized critic still uses the
/// unchanged schema-2 public384 context, with no calibrated value claim.
#[derive(Clone, Debug)]
pub struct SemanticPlayingRlIdentity {
    origin_checksum: String,
    actor_checksum: String,
    critic_checksum: String,
    critic_seed: u64,
    checksum: String,
}
impl SemanticPlayingRlIdentity {
    pub fn checksum(&self) -> &str {
        &self.checksum
    }
    pub fn policy_version(&self) -> &'static str {
        POLICY_VERSION
    }
    pub fn input_contract(&self) -> &'static str {
        INPUT_CONTRACT
    }
    pub fn feature_schema(&self) -> u32 {
        SEMANTIC_PLAYING_FEATURE_SCHEMA
    }
    pub fn origin_checksum(&self) -> &str {
        &self.origin_checksum
    }
    pub fn actor_checksum(&self) -> &str {
        &self.actor_checksum
    }
    pub fn critic_checksum(&self) -> &str {
        &self.critic_checksum
    }
    pub fn critic_seed(&self) -> u64 {
        self.critic_seed
    }
    pub fn critic_feature_schema(&self) -> u32 {
        PUBLIC_FEATURE_SCHEMA
    }
    pub fn critic_context_contract(&self) -> &'static str {
        critic::CONTEXT_CONTRACT
    }
    pub fn completed_updates(&self) -> u64 {
        0
    }
    pub fn source_authenticated(&self) -> bool {
        false
    }
    pub fn source_qualified(&self) -> bool {
        false
    }
    pub fn verified_native_policy(&self) -> bool {
        false
    }
    pub fn value_validity(&self) -> ValueValidity {
        ValueValidity::UnavailablePolicyOnly
    }
}

/// Fresh ownership only: no BC optimizer/RNG/epochs, old RL credit or critic is
/// imported. There is no From/AsRef/Deref/into-parts route to a legacy owner.
pub struct SemanticPlayingRlOwner {
    origin: SemanticPlayingRlOrigin,
    actor: PublicPolicyModel,
    critic: Vec<f32>,
    identity: SemanticPlayingRlIdentity,
}
impl SemanticPlayingRlOwner {
    pub fn from_prepared(
        prepared: &PreparedConditionalHumanPolicy,
        critic_seed: u64,
    ) -> Result<Self, String> {
        let origin = SemanticPlayingRlOrigin::from_prepared(prepared)?;
        let actor = prepared.model().model.clone();
        actor.validate()?;
        let critic = crate::public_policy_baseline::zero_output_parameters(critic_seed);
        let actor_checksum = parameter_checksum(
            b"tzolkin-semantic-playing-rl-actor-bits-v1\0",
            actor.parameters(),
        );
        let critic_checksum = hash(
            b"tzolkin-semantic-playing-rl-fresh-critic-v1\0",
            &(
                &origin.checksum,
                CRITIC_INIT_VERSION,
                critic::MODEL_VERSION,
                critic::CONTEXT_CONTRACT,
                (
                    critic::CONTEXT_SCHEMA,
                    PUBLIC_FEATURE_SCHEMA,
                    critic::CONTEXT_COUNT,
                ),
                critic_seed,
                parameter_checksum(b"tzolkin-semantic-playing-rl-critic-bits-v1\0", &critic),
            ),
        )?;
        let checksum = hash(
            b"tzolkin-semantic-playing-rl-fresh-identity-v1\0",
            &(
                (
                    POLICY_VERSION,
                    INPUT_CONTRACT,
                    SEMANTIC_PLAYING_FEATURE_SCHEMA,
                ),
                (
                    &origin.checksum,
                    &actor_checksum,
                    &critic_checksum,
                    critic_seed,
                ),
                (
                    "scalar",
                    0_u64,
                    false,
                    false,
                    false,
                    ValueValidity::UnavailablePolicyOnly,
                ),
            ),
        )?;
        origin.recheck_sources()?;
        Ok(Self {
            identity: SemanticPlayingRlIdentity {
                origin_checksum: origin.checksum.clone(),
                actor_checksum,
                critic_checksum,
                critic_seed,
                checksum,
            },
            origin,
            actor,
            critic,
        })
    }
    pub fn origin(&self) -> Result<&SemanticPlayingRlOrigin, String> {
        self.origin.recheck_sources()?;
        Ok(&self.origin)
    }
    pub fn identity(&self) -> Result<&SemanticPlayingRlIdentity, String> {
        self.origin.recheck_sources()?;
        Ok(&self.identity)
    }
    pub fn actor_parameters(&self) -> Result<&[f32], String> {
        self.origin.recheck_sources()?;
        Ok(self.actor.parameters())
    }
    pub fn critic_parameters(&self) -> Result<&[f32], String> {
        self.origin.recheck_sources()?;
        Ok(&self.critic)
    }
    pub fn handle(&self) -> Result<SemanticPlayingRlHandle<'_>, String> {
        self.origin.recheck_sources()?;
        Ok(SemanticPlayingRlHandle {
            owner: self,
            kernel: Kernel::Scalar.resolve()?,
        })
    }
}
fn parameter_checksum(domain: &[u8], parameters: &[f32]) -> String {
    let mut hash = Sha256::new();
    hash.update(domain);
    for value in parameters {
        hash.update(value.to_bits().to_le_bytes());
    }
    format!("{:x}", hash.finalize())
}

/// A separate immutable borrow. Scoring supplied observations is numeric only;
/// it neither samples/chooses/applies actions nor grants a native actor role.
pub struct SemanticPlayingRlHandle<'a> {
    owner: &'a SemanticPlayingRlOwner,
    kernel: ResolvedKernel,
}
impl SemanticPlayingRlHandle<'_> {
    pub fn identity(&self) -> Result<&SemanticPlayingRlIdentity, String> {
        self.owner.identity()
    }
    pub fn backend(&self) -> &'static str {
        self.kernel.backend()
    }
    pub fn distribution(
        &self,
        observation: &Observation,
    ) -> Result<NumericPolicyDistribution, String> {
        self.owner.origin.recheck_sources()?;
        let result = (|| {
            let encoder = FeatureEncoder::new_semantic_playing(observation)?;
            let rows = (0..observation.legal_actions.len())
                .map(|i| encoder.encode_legal_tagged(i))
                .collect::<Result<Vec<_>, _>>()?;
            predict_semantic_playing_numeric(&self.owner.actor, self.kernel, &rows)
        })();
        self.owner.origin.recheck_sources()?;
        result
    }
}
