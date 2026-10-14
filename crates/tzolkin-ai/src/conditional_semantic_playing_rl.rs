//! Fresh schema-4 Playing ownership and private checked numerical custody.
//! The public fresh identity never becomes an updated or legacy owner. Separate
//! in-memory routes do not provide a journal/export/resume conversion or source authentication.
use std::collections::BTreeMap;

use sha2::{Digest, Sha256};
use tzolkin_core::observation::Observation;
use tzolkin_inference::policy::NumericPolicyDistribution;

use crate::conditional_human_rl_update::ConditionalNativeExclusions;
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
    pub(crate) fn collection_view(&self) -> SemanticOwnerView<'_> {
        SemanticOwnerView {
            origin: &self.origin,
            actor: &self.actor,
            critic: &self.critic,
            parent: self.identity.checksum(),
            root: self.identity.checksum(),
        }
    }

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

/// An owned, private numerical state obtained only by consuming a fresh owner.
/// It cannot be exposed as that fresh owner after either parameter vector changes.
pub(crate) struct SemanticTrainingState {
    origin: SemanticPlayingRlOrigin,
    actor: PublicPolicyModel,
    critic: Vec<f32>,
    root: SemanticPlayingRlIdentity,
}
impl SemanticTrainingState {
    pub(crate) fn from_fresh(owner: SemanticPlayingRlOwner) -> Result<Self, String> {
        owner.origin.recheck_sources()?;
        let current = Self {
            origin: owner.origin,
            actor: owner.actor,
            critic: owner.critic,
            root: owner.identity,
        };
        current.origin.recheck_sources()?;
        Ok(current)
    }
    pub(crate) fn root(&self) -> &SemanticPlayingRlIdentity {
        &self.root
    }
    pub(crate) fn actor_parameters(&self) -> &[f32] {
        self.actor.parameters()
    }
    pub(crate) fn critic_parameters(&self) -> &[f32] {
        &self.critic
    }
    pub(crate) fn view<'a>(&'a self, parent: &'a str) -> SemanticOwnerView<'a> {
        SemanticOwnerView {
            origin: &self.origin,
            actor: &self.actor,
            critic: &self.critic,
            parent,
            root: self.root.checksum(),
        }
    }
    /// All proposal validation and every-outcome source checks precede this move.
    pub(crate) fn commit(&mut self, actor: PublicPolicyModel, critic: Vec<f32>) {
        (self.actor, self.critic) = (actor, critic);
    }
}

/// Closed borrowed source: only fresh/current sealed owners can construct it.
/// Neither models nor a caller predictor can construct a numerical scope.
pub(crate) struct SemanticOwnerView<'a> {
    origin: &'a SemanticPlayingRlOrigin,
    actor: &'a PublicPolicyModel,
    critic: &'a [f32],
    parent: &'a str,
    root: &'a str,
}
impl SemanticOwnerView<'_> {
    pub(crate) fn root_checksum(&self) -> &str {
        self.root
    }
    pub(crate) fn origin_custody(&self) -> SemanticOriginCustody {
        SemanticOriginCustody {
            checksum: self.origin.checksum.clone(),
            pins: self.origin.source_pins.clone(),
            reservations: self.origin.table_reservations.clone(),
        }
    }
    pub(crate) fn metadata(&self) -> (&str, &BTreeMap<String, DatasetSplit>) {
        (self.parent, &self.origin.table_reservations)
    }
    pub(crate) fn with_checked<R>(
        &self,
        external: &ConditionalNativeExclusions,
        pins: &[ContentPin],
        run: impl for<'scope> FnOnce(&SemanticCollectionScope<'scope>) -> Result<R, String>,
    ) -> Result<R, String> {
        self.with_checked_episode(external, pins, || {
            let scope = SemanticCollectionScope {
                actor: self.actor,
                critic: self.critic,
                parent: self.parent,
                kernel: Kernel::Scalar.resolve()?,
            };
            run(&scope)
        })
    }
    pub(crate) fn with_checked_episode<R>(
        &self,
        external: &ConditionalNativeExclusions,
        pins: &[ContentPin],
        run: impl FnOnce() -> Result<R, String>,
    ) -> Result<R, String> {
        let check = || {
            self.origin.recheck_sources()?;
            // This typed loader admits up to 32MiB. Do not narrow it through
            // the generic 16MiB metadata-pin rechecker.
            external.recheck()?;
            for pin in pins {
                crate::human_policy_dataset::recheck_absolute_pin(pin)?;
            }
            Ok::<_, String>(())
        };
        check()?;
        let result = run();
        match (result, check()) {
            (Ok(value), Ok(())) => Ok(value),
            (Err(error), Ok(())) => Err(error),
            (Ok(_), Err(error)) => Err(format!("Semantic source postcheck: {error}")),
            (Err(error), Err(post)) => Err(format!("{error}; semantic source postcheck: {post}")),
        }
    }
}

/// Copied source metadata, never a numerical owner or a constructor for one.
pub(crate) struct SemanticOriginCustody {
    checksum: String,
    pins: Vec<ContentPin>,
    reservations: BTreeMap<String, DatasetSplit>,
}
impl SemanticOriginCustody {
    pub(crate) fn recheck(&self) -> Result<(), String> {
        for pin in &self.pins {
            crate::human_policy_dataset::recheck_absolute_pin(pin)?;
        }
        Ok(())
    }
    pub(crate) fn binding(&self) -> Result<String, String> {
        hash(
            b"semantic-frozen-origin-custody-v1\0",
            &(&self.checksum, &self.pins, &self.reservations),
        )
    }
    pub(crate) fn reservations(&self) -> &BTreeMap<String, DatasetSplit> {
        &self.reservations
    }
}

/// Private scoped numerical access; copied values do not carry source ownership.
/// Only checked owners create a scope, and scoring performs no filesystem reads.
pub(crate) struct SemanticCollectionScope<'a> {
    actor: &'a PublicPolicyModel,
    critic: &'a [f32],
    parent: &'a str,
    kernel: ResolvedKernel,
}
impl SemanticCollectionScope<'_> {
    pub(crate) fn parent_checksum(&self) -> &str {
        self.parent
    }
    pub(crate) fn actor(&self) -> &PublicPolicyModel {
        self.actor
    }
    pub(crate) fn critic(&self) -> &[f32] {
        self.critic
    }
    pub(crate) fn kernel(&self) -> ResolvedKernel {
        self.kernel
    }
    pub(crate) fn distribution(
        &self,
        observation: &Observation,
    ) -> Result<NumericPolicyDistribution, String> {
        let encoder = FeatureEncoder::new_semantic_playing(observation)?;
        let rows = (0..observation.legal_actions.len())
            .map(|i| encoder.encode_legal_tagged(i))
            .collect::<Result<Vec<_>, _>>()?;
        predict_semantic_playing_numeric(self.actor, self.kernel, &rows)
    }
}

/// Numeric/Core fixture context only, not an owner or successful-load claim.
/// Production collection can obtain this borrow only through source checks.
#[cfg(test)]
pub(crate) fn with_collection_numeric_fixture<R>(
    prepared: &PreparedConditionalHumanPolicy,
    run: impl for<'scope> FnOnce(&SemanticCollectionScope<'scope>) -> Result<R, String>,
) -> Result<R, String> {
    prepared.model().validate_semantic_playing()?;
    let critic = crate::public_policy_baseline::zero_output_parameters(11);
    let identity = SemanticPlayingRlIdentity {
        origin_checksum: "synthetic-numeric-unit-only".into(),
        actor_checksum: parameter_checksum(b"synthetic-actor", prepared.model().model.parameters()),
        critic_checksum: parameter_checksum(b"synthetic-critic", &critic),
        critic_seed: 11,
        checksum: "synthetic-no-source-qualification".into(),
    };
    let scope = SemanticCollectionScope {
        actor: &prepared.model().model,
        critic: &critic,
        parent: identity.checksum(),
        kernel: Kernel::Scalar.resolve()?,
    };
    run(&scope)
}
pub(crate) fn parameter_checksum(domain: &[u8], parameters: &[f32]) -> String {
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
