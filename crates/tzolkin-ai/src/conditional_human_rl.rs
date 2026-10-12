//! Load-backed conditional Human initialization, separate from native RL lineage.
//! This unit grants all-phase inference and sampling, not updates or source admission.
use crate::dataset::DatasetSplit;
use crate::features::FeatureEncoder;
use crate::human_initial_wealth::ContentPin;
use crate::human_policy_training::{ConditionalDatasetIdentity, PreparedConditionalHumanPolicy};
use crate::kernel::{Kernel, ResolvedKernel};
use crate::model::digest;
use crate::public_model::{PublicPolicyModel, ValueValidity, predict_numeric, validate_contract};
use crate::public_stochastic::{SamplingSeed, SamplingStreamIdentity};
use serde::Serialize;
use std::collections::BTreeMap;
use tzolkin_core::GameOptions;
use tzolkin_core::observation::Observation;

pub const INPUT_CONTRACT: &str =
    "conditional-human-origin-base-default-3-4p-public512-native-setup-playing-v1";
pub const POLICY_VERSION: &str = "conditional-human-origin-count0-scalar-v1";
pub const SAMPLING_VERSION: &str = "conditional-human-origin-count0-uniform-tick53-v1";

/// Only successful production loading emits these raw bindings. No Deserialize,
/// report conversion or public constructor can mint an origin.
#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ConditionalRlOrigin {
    checkpoint_checksum: String,
    model_checksum: String,
    dataset: ConditionalDatasetIdentity,
    table_reservations: BTreeMap<String, DatasetSplit>,
    source_pins: Vec<ContentPin>,
    checksum: String,
}
impl ConditionalRlOrigin {
    fn from_prepared(prepared: &PreparedConditionalHumanPolicy) -> Result<Self, String> {
        let mut origin = Self {
            checkpoint_checksum: prepared.checkpoint_checksum().into(),
            model_checksum: prepared.model().checksum.clone(),
            dataset: prepared.identity().clone(),
            table_reservations: prepared.reservations().clone(),
            source_pins: prepared.loaded_sources()?.to_vec(),
            checksum: String::new(),
        };
        origin.recheck_sources()?;
        let mut bytes = b"tzolkin-conditional-human-rl-loaded-origin-v1\0".to_vec();
        bytes.extend(crate::policy_training::bounded_json(&origin)?);
        origin.checksum = digest(&bytes);
        Ok(origin)
    }
    pub fn checksum(&self) -> &str {
        &self.checksum
    }
    pub fn dataset(&self) -> &ConditionalDatasetIdentity {
        &self.dataset
    }
    /// Includes reservations for unloaded held-out and zero-row tables.
    pub fn table_reservations(&self) -> &BTreeMap<String, DatasetSplit> {
        &self.table_reservations
    }
    pub fn source_pins(&self) -> &[ContentPin] {
        &self.source_pins
    }
    /// Full raw closure check belongs at lifecycle boundaries, never a decision callback.
    pub fn recheck_sources(&self) -> Result<(), String> {
        for pin in &self.source_pins {
            crate::human_policy_dataset::recheck_absolute_pin(pin)?;
        }
        Ok(())
    }
}

/// Metadata from a sealed owner; deserializing metadata cannot produce an owner.
#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ConditionalRlIdentity {
    policy_version: &'static str,
    input_contract: &'static str,
    native_input_contract: &'static str,
    game_options: GameOptions,
    uxmal_corn_limit: bool,
    origin_checksum: String,
    actor_checksum: String,
    critic_checksum: String,
    critic_seed: u64,
    sampling_seed: u64,
    mixed_update_count: u64,
    source_authenticated: bool,
    source_qualified: bool,
    verified_native_policy: bool,
    checksum: String,
}
impl ConditionalRlIdentity {
    pub fn checksum(&self) -> &str {
        &self.checksum
    }
    pub fn mixed_update_count(&self) -> u64 {
        self.mixed_update_count
    }
    pub fn input_contract(&self) -> &str {
        self.input_contract
    }
    pub fn origin_checksum(&self) -> &str {
        &self.origin_checksum
    }
    pub fn critic_checksum(&self) -> &str {
        &self.critic_checksum
    }
}

/// Fresh conditional count0 ownership. BC optimizer, epochs and RNG are not copied.
/// There is no native SeatPolicy conversion, update/import or durable API here.
pub struct ConditionalRlOwner {
    origin: ConditionalRlOrigin,
    actor: PublicPolicyModel,
    critic: Vec<f32>,
    identity: ConditionalRlIdentity,
}
impl ConditionalRlOwner {
    pub fn from_prepared(
        prepared: &PreparedConditionalHumanPolicy,
        critic_seed: u64,
        sampling_seed: SamplingSeed,
    ) -> Result<Self, String> {
        let origin = ConditionalRlOrigin::from_prepared(prepared)?;
        let actor = prepared.model().model.clone();
        actor.validate()?;
        let critic = crate::public_policy_baseline::zero_output_parameters(critic_seed);
        // A distinct domain binds fresh critic bits to this conditional source.
        let mut critic_bytes = b"tzolkin-conditional-human-rl-fresh-critic-v1\0".to_vec();
        critic_bytes.extend_from_slice(origin.checksum().as_bytes());
        critic_bytes.extend_from_slice(crate::public_state_critic::MODEL_VERSION.as_bytes());
        critic_bytes.extend_from_slice(crate::public_state_critic::CONTEXT_CONTRACT.as_bytes());
        critic_bytes.extend_from_slice(&critic_seed.to_le_bytes());
        for value in &critic {
            critic_bytes.extend_from_slice(&value.to_bits().to_le_bytes());
        }
        let mut identity = ConditionalRlIdentity {
            policy_version: POLICY_VERSION,
            input_contract: INPUT_CONTRACT,
            native_input_contract: crate::public_model::INPUT_CONTRACT,
            game_options: GameOptions::default(),
            uxmal_corn_limit: false,
            origin_checksum: origin.checksum().into(),
            actor_checksum: prepared.model().checksum.clone(),
            critic_checksum: digest(&critic_bytes),
            critic_seed,
            sampling_seed: sampling_seed.value(),
            mixed_update_count: 0,
            source_authenticated: false,
            source_qualified: false,
            verified_native_policy: false,
            checksum: String::new(),
        };
        let mut bytes = b"tzolkin-conditional-human-rl-count0-identity-v1\0".to_vec();
        bytes.extend(crate::policy_training::bounded_json(&identity)?);
        identity.checksum = digest(&bytes);
        Ok(Self {
            origin,
            actor,
            critic,
            identity,
        })
    }
    pub fn origin(&self) -> &ConditionalRlOrigin {
        &self.origin
    }
    pub fn identity(&self) -> &ConditionalRlIdentity {
        &self.identity
    }
    pub fn actor_parameters(&self) -> &[f32] {
        self.actor.parameters()
    }
    pub fn critic_parameters(&self) -> &[f32] {
        &self.critic
    }
    pub fn next_episode_ordinal(&self) -> u64 {
        0
    }
    pub fn attempts(&self) -> u64 {
        0
    }
    pub fn credited_own_decisions(&self) -> u64 {
        0
    }
    pub fn consumed_native_families(&self) -> usize {
        0
    }
    pub fn handle(&self) -> Result<ConditionalRlPolicyHandle<'_>, String> {
        Ok(ConditionalRlPolicyHandle {
            owner: self,
            kernel: Kernel::Scalar.resolve()?,
        })
    }
}

/// Borrowed immutable model; validates native inputs before encoding all legal rows.
pub struct ConditionalRlPolicyHandle<'a> {
    owner: &'a ConditionalRlOwner,
    kernel: ResolvedKernel,
}
impl ConditionalRlPolicyHandle<'_> {
    pub fn identity(&self) -> &ConditionalRlIdentity {
        self.owner.identity()
    }
    pub fn backend(&self) -> &'static str {
        self.kernel.backend()
    }
    pub(crate) fn sampling_prefix_matches(&self, identity: SamplingStreamIdentity) -> bool {
        identity.sampling_seed().value() == self.owner.identity.sampling_seed
            && identity.replicate_ordinal() == 0
    }
    pub fn distribution(
        &self,
        observation: &Observation,
    ) -> Result<ConditionalRlDistribution, String> {
        validate_contract(observation)?;
        let encoder = FeatureEncoder::new_public(observation)?;
        let rows = (0..observation.legal_actions.len())
            .map(|i| encoder.encode_legal_tagged(i))
            .collect::<Result<Vec<_>, _>>()?;
        let numeric = predict_numeric(&self.owner.actor, self.kernel, &rows)?;
        Ok(ConditionalRlDistribution {
            identity: self.owner.identity.clone(),
            logits: numeric.logits,
            probabilities: numeric.probabilities,
        })
    }
}
pub struct ConditionalRlDistribution {
    identity: ConditionalRlIdentity,
    logits: Vec<f32>,
    probabilities: Vec<f32>,
}
impl ConditionalRlDistribution {
    pub fn identity(&self) -> &ConditionalRlIdentity {
        &self.identity
    }
    pub fn logits(&self) -> &[f32] {
        &self.logits
    }
    pub fn probabilities(&self) -> &[f32] {
        &self.probabilities
    }
    pub fn value_validity(&self) -> ValueValidity {
        ValueValidity::UnavailablePolicyOnly
    }
}
