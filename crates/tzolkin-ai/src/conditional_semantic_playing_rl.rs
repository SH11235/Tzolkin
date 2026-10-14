//! Fresh schema-4 Playing ownership and private checked numerical custody.
//! The public fresh identity never becomes an updated or legacy owner. Separate
//! in-memory routes do not provide a journal/export/resume conversion or source authentication.
use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use crate::arena::teacher::PreparedSemanticPlayingTeacher;

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
use crate::semantic_playing_teacher_dataset::TeacherProfile;

/// Invocation-only custody for already-owned values from the complete V7 loader.
/// It is deliberately absent from origin, numerical and snapshot identities.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Deserialize, serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) enum SemanticNativeSourceCustody {
    NativeCommandBoundaryV1,
}

#[cfg(test)]
thread_local! {
    static NATIVE_BACKING_CHECKS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}
#[cfg(test)]
pub(crate) fn native_backing_checks() -> usize {
    NATIVE_BACKING_CHECKS.get()
}

/// One checked boundary pass, never retained across work or publication. Generic
/// origin/history pins and typed exclusions keep their distinct read contracts.
pub(crate) struct SemanticSourceCheckPass<'a> {
    pins: BTreeMap<String, ContentPin>,
    exclusions: BTreeSet<Vec<u8>>,
    native: BTreeSet<String>,
    _boundary: &'a (),
}
pub(crate) fn with_source_check_pass<R>(
    run: impl for<'check> FnOnce(&mut SemanticSourceCheckPass<'check>) -> Result<R, String>,
) -> Result<R, String> {
    let boundary = ();
    run(&mut SemanticSourceCheckPass {
        pins: BTreeMap::new(),
        exclusions: BTreeSet::new(),
        native: BTreeSet::new(),
        _boundary: &boundary,
    })
}
impl SemanticSourceCheckPass<'_> {
    pub(crate) fn pin(&mut self, pin: &ContentPin) -> Result<(), String> {
        if let Some(previous) = self.pins.get(&pin.path) {
            return if previous == pin {
                Ok(())
            } else {
                Err("Conflicting semantic source pins in one boundary pass".into())
            };
        }
        crate::human_policy_dataset::recheck_absolute_pin(pin)?;
        self.pins.insert(pin.path.clone(), pin.clone());
        Ok(())
    }
    pub(crate) fn external(
        &mut self,
        external: &ConditionalNativeExclusions,
    ) -> Result<(), String> {
        let descriptor = external.descriptor();
        let raw = serde_json::to_vec(descriptor).map_err(|e| e.to_string())?;
        if self.exclusions.contains(&raw) {
            return Ok(());
        }
        // Keep the typed 32MiB loader separate from the generic 16MiB pin path,
        // including when the same file is declared in both roles.
        external.recheck()?;
        self.exclusions.insert(raw);
        Ok(())
    }
}

/// Native custody retains the successful complete-source loader, including its
/// typed dataset caps. It cannot be constructed from an actor parameter slice.
pub(crate) struct NativeOrigin {
    prepared: Arc<PreparedSemanticPlayingTeacher>,
    source_custody: Option<SemanticNativeSourceCustody>,
    description: serde_json::Value,
    checksum: String,
    tables: BTreeMap<String, DatasetSplit>,
}
impl NativeOrigin {
    fn from_prepared(
        prepared: Arc<PreparedSemanticPlayingTeacher>,
        source_custody: Option<SemanticNativeSourceCustody>,
    ) -> Result<Self, String> {
        if source_custody.is_some() && prepared.profile() != TeacherProfile::V7 {
            return Err("Command-boundary source custody requires a V7 native origin".into());
        }
        Self::recheck_prepared(&prepared, source_custody)?;
        let checkpoint = prepared.rl_checkpoint();
        let (schema, domain, task): (&str, &[u8], &str) = match prepared.profile() {
            TeacherProfile::V5 => (
                "native-corn-first-v5-semantic-playing-rl-origin-v1",
                b"native-corn-first-v5-semantic-playing-rl-origin-v1\0",
                crate::semantic_playing_teacher_training::TASK,
            ),
            TeacherProfile::V7 => (
                "native-corn-first-v7-semantic-playing-rl-origin-v1",
                b"native-corn-first-v7-semantic-playing-rl-origin-v1\0",
                crate::semantic_playing_teacher_training::V7_TASK,
            ),
        };
        let description = serde_json::json!({
            "schema":schema,
            "sourceKind":prepared.profile().source_kind(),
            "task":task,
            "checkpoint":checkpoint.checksum,
            "actor":parameter_checksum(b"tzolkin-semantic-playing-rl-actor-bits-v1\0",
                checkpoint.model.parameters()),
            "dataset":checkpoint.dataset,"locations":prepared.rl_source_locations()});
        let checksum = hash(domain, &description)?;
        Self::recheck_prepared(&prepared, source_custody)?;
        Ok(Self {
            prepared,
            source_custody,
            description,
            checksum,
            tables: BTreeMap::new(),
        })
    }
    fn recheck_prepared(
        prepared: &PreparedSemanticPlayingTeacher,
        source_custody: Option<SemanticNativeSourceCustody>,
    ) -> Result<(), String> {
        if source_custody.is_some() {
            return Ok(());
        }
        #[cfg(test)]
        NATIVE_BACKING_CHECKS.update(|n| n + 1);
        prepared.recheck_sources()
    }
    fn recheck_backing(&self) -> Result<(), String> {
        Self::recheck_prepared(&self.prepared, self.source_custody)
    }
    fn recheck_in(&self, checked: &mut SemanticSourceCheckPass<'_>) -> Result<(), String> {
        // A skipped backing read must never satisfy a separately supplied legacy
        // origin with the same data checksum in this boundary pass.
        if self.source_custody.is_some() {
            return Ok(());
        }
        if !checked.native.contains(&self.checksum) {
            self.recheck_backing()?;
            checked.native.insert(self.checksum.clone());
        }
        Ok(())
    }
    fn check_external(&self, external: &ConditionalNativeExclusions) -> Result<(), String> {
        if self
            .prepared
            .rl_checkpoint()
            .dataset
            .manifest
            .families
            .iter()
            .any(|f| !external.contains_family(&f.family_id))
        {
            return Err("Native4 RL prior omits a retained teacher family".into());
        }
        Ok(())
    }
}
enum TrainingOrigin {
    Human(Box<SemanticPlayingRlOrigin>),
    Native(Arc<NativeOrigin>),
}
#[derive(Clone, Copy)]
enum OriginRef<'a> {
    Human(&'a SemanticPlayingRlOrigin),
    Native(&'a Arc<NativeOrigin>),
}
impl TrainingOrigin {
    fn view(&self) -> OriginRef<'_> {
        match self {
            Self::Human(o) => OriginRef::Human(o),
            Self::Native(o) => OriginRef::Native(o),
        }
    }
}
impl OriginRef<'_> {
    fn recheck(&self) -> Result<(), String> {
        match self {
            Self::Human(o) => o.recheck_sources(),
            Self::Native(o) => o.recheck_backing(),
        }
    }
    fn tables(&self) -> &BTreeMap<String, DatasetSplit> {
        match self {
            Self::Human(o) => &o.table_reservations,
            Self::Native(o) => &o.tables,
        }
    }
    fn recheck_in(&self, checked: &mut SemanticSourceCheckPass<'_>) -> Result<(), String> {
        match self {
            Self::Human(o) => {
                for pin in &o.source_pins {
                    checked.pin(pin)?;
                }
                Ok(())
            }
            Self::Native(o) => o.recheck_in(checked),
        }
    }
}
/// Closed internal initialization; only successful typed loaders supply owners.
pub(crate) enum SemanticFreshOwner {
    Human(Box<SemanticPlayingRlOwner>),
    Native {
        origin: Arc<NativeOrigin>,
        actor: PublicPolicyModel,
        critic: Vec<f32>,
        identity: SemanticPlayingRlIdentity,
    },
}
impl SemanticFreshOwner {
    pub(crate) fn native(
        prepared: Arc<PreparedSemanticPlayingTeacher>,
        seed: u64,
    ) -> Result<Self, String> {
        Self::native_with_custody(prepared, seed, None)
    }
    pub(crate) fn native_with_custody(
        prepared: Arc<PreparedSemanticPlayingTeacher>,
        seed: u64,
        source_custody: Option<SemanticNativeSourceCustody>,
    ) -> Result<Self, String> {
        let origin = Arc::new(NativeOrigin::from_prepared(prepared, source_custody)?);
        let actor = PublicPolicyModel {
            parameters: origin.prepared.rl_checkpoint().model.parameters().to_vec(),
        };
        actor.validate()?;
        let (critic, identity) = fresh_identity(
            &origin.checksum,
            &actor,
            seed,
            Some(origin.prepared.profile()),
        )?;
        origin.recheck_backing()?;
        Ok(Self::Native {
            origin,
            actor,
            critic,
            identity,
        })
    }
    fn parts(
        self,
    ) -> (
        TrainingOrigin,
        PublicPolicyModel,
        Vec<f32>,
        SemanticPlayingRlIdentity,
    ) {
        match self {
            Self::Human(o) => {
                let o = *o;
                (
                    TrainingOrigin::Human(Box::new(o.origin)),
                    o.actor,
                    o.critic,
                    o.identity,
                )
            }
            Self::Native {
                origin,
                actor,
                critic,
                identity,
            } => (TrainingOrigin::Native(origin), actor, critic, identity),
        }
    }
}

pub const POLICY_VERSION: &str = "conditional-human-semantic-playing-fresh-scalar-v1";
pub const INPUT_CONTRACT: &str =
    "conditional-human-semantic-playing-origin-base-3-4p-schema4-numeric-v1";
pub const NATIVE_POLICY_VERSION: &str = "native-corn-first-v5-semantic-playing-fresh-scalar-v1";
pub const NATIVE_INPUT_CONTRACT: &str =
    "native-corn-first-v5-semantic-playing-origin-base-3-4p-schema4-numeric-v1";
pub const NATIVE_V7_POLICY_VERSION: &str = "native-corn-first-v7-semantic-playing-fresh-scalar-v1";
pub const NATIVE_V7_INPUT_CONTRACT: &str =
    "native-corn-first-v7-semantic-playing-origin-base-3-4p-schema4-numeric-v1";
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
    native: Option<TeacherProfile>,
}
impl SemanticPlayingRlIdentity {
    pub fn checksum(&self) -> &str {
        &self.checksum
    }
    pub fn policy_version(&self) -> &'static str {
        match self.native {
            None => POLICY_VERSION,
            Some(TeacherProfile::V5) => NATIVE_POLICY_VERSION,
            Some(TeacherProfile::V7) => NATIVE_V7_POLICY_VERSION,
        }
    }
    pub fn input_contract(&self) -> &'static str {
        match self.native {
            None => INPUT_CONTRACT,
            Some(TeacherProfile::V5) => NATIVE_INPUT_CONTRACT,
            Some(TeacherProfile::V7) => NATIVE_V7_INPUT_CONTRACT,
        }
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
            origin: OriginRef::Human(&self.origin),
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
        let (critic, identity) = fresh_identity(&origin.checksum, &actor, critic_seed, None)?;
        origin.recheck_sources()?;
        Ok(Self {
            identity,
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

fn fresh_identity(
    origin_checksum: &str,
    actor: &PublicPolicyModel,
    critic_seed: u64,
    native: Option<TeacherProfile>,
) -> Result<(Vec<f32>, SemanticPlayingRlIdentity), String> {
    let critic = crate::public_policy_baseline::zero_output_parameters(critic_seed);
    let actor_checksum = parameter_checksum(
        b"tzolkin-semantic-playing-rl-actor-bits-v1\0",
        actor.parameters(),
    );
    let critic_checksum = hash(
        match native {
            None => b"tzolkin-semantic-playing-rl-fresh-critic-v1\0".as_slice(),
            Some(TeacherProfile::V5) => b"native-corn-first-v5-fresh-critic-v1\0".as_slice(),
            Some(TeacherProfile::V7) => b"native-corn-first-v7-fresh-critic-v1\0".as_slice(),
        },
        &(
            origin_checksum,
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
        match native {
            None => b"tzolkin-semantic-playing-rl-fresh-identity-v1\0".as_slice(),
            Some(TeacherProfile::V5) => b"native-corn-first-v5-fresh-identity-v1\0".as_slice(),
            Some(TeacherProfile::V7) => b"native-corn-first-v7-fresh-identity-v1\0".as_slice(),
        },
        &(
            (
                match native {
                    None => POLICY_VERSION,
                    Some(TeacherProfile::V5) => NATIVE_POLICY_VERSION,
                    Some(TeacherProfile::V7) => NATIVE_V7_POLICY_VERSION,
                },
                match native {
                    None => INPUT_CONTRACT,
                    Some(TeacherProfile::V5) => NATIVE_INPUT_CONTRACT,
                    Some(TeacherProfile::V7) => NATIVE_V7_INPUT_CONTRACT,
                },
                SEMANTIC_PLAYING_FEATURE_SCHEMA,
            ),
            (
                origin_checksum,
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
    Ok((
        critic,
        SemanticPlayingRlIdentity {
            origin_checksum: origin_checksum.into(),
            actor_checksum,
            critic_checksum,
            critic_seed,
            checksum,
            native,
        },
    ))
}

/// An owned, private numerical state obtained only by consuming a fresh owner.
/// It cannot be exposed as that fresh owner after either parameter vector changes.
pub(crate) struct SemanticTrainingState {
    origin: TrainingOrigin,
    actor: PublicPolicyModel,
    critic: Vec<f32>,
    root: SemanticPlayingRlIdentity,
}
impl SemanticTrainingState {
    pub(crate) fn from_checkpoint(
        owner: SemanticFreshOwner,
        root: &str,
        actor: PublicPolicyModel,
        critic: Vec<f32>,
    ) -> Result<Self, String> {
        let (origin, _, _, identity) = owner.parts();
        origin.view().recheck()?;
        actor.validate()?;
        if identity.checksum() != root
            || critic.len() != crate::public_state_critic::PARAMETER_COUNT
            || critic.iter().any(|v| !v.is_finite())
        {
            return Err("Semantic checkpoint root/critic differs".into());
        }
        let current = Self {
            origin,
            actor,
            critic,
            root: identity,
        };
        current.origin.view().recheck()?;
        Ok(current)
    }
    pub(crate) fn from_fresh(owner: SemanticFreshOwner) -> Result<Self, String> {
        let (origin, actor, critic, root) = owner.parts();
        origin.view().recheck()?;
        let current = Self {
            origin,
            actor,
            critic,
            root,
        };
        current.origin.view().recheck()?;
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
            origin: self.origin.view(),
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
    origin: OriginRef<'a>,
    actor: &'a PublicPolicyModel,
    critic: &'a [f32],
    parent: &'a str,
    root: &'a str,
}
impl SemanticOwnerView<'_> {
    pub(crate) fn journal_origin(&self) -> serde_json::Value {
        match self.origin {
            OriginRef::Human(origin) => serde_json::json!({"checksum":origin.checksum,
                "checkpoint":origin.checkpoint_checksum,"model":origin.model_checksum,
                "dataset":origin.dataset,"reservations":origin.table_reservations,
                "pins":origin.source_pins}),
            OriginRef::Native(origin) => {
                serde_json::json!({"checksum":origin.checksum,"nativeTeacher":origin.description})
            }
        }
    }
    pub(crate) fn root_checksum(&self) -> &str {
        self.root
    }
    pub(crate) fn origin_custody(&self) -> SemanticOriginCustody {
        match self.origin {
            OriginRef::Human(origin) => SemanticOriginCustody {
                checksum: origin.checksum.clone(),
                pins: origin.source_pins.clone(),
                reservations: origin.table_reservations.clone(),
                native: None,
            },
            OriginRef::Native(origin) => SemanticOriginCustody {
                checksum: origin.checksum.clone(),
                pins: Vec::new(),
                reservations: BTreeMap::new(),
                native: Some(Arc::clone(origin)),
            },
        }
    }
    pub(crate) fn metadata(&self) -> (&str, &BTreeMap<String, DatasetSplit>) {
        (self.parent, self.origin.tables())
    }
    pub(crate) fn with_checked<R>(
        &self,
        external: &ConditionalNativeExclusions,
        pins: &[ContentPin],
        run: impl for<'scope> FnOnce(&SemanticCollectionScope<'scope>) -> Result<R, String>,
    ) -> Result<R, String> {
        self.with_checked_pool(external, pins, |_| Ok(()), run)
    }
    pub(crate) fn with_checked_pool<R>(
        &self,
        external: &ConditionalNativeExclusions,
        pins: &[ContentPin],
        check_pool: impl for<'check> Fn(&mut SemanticSourceCheckPass<'check>) -> Result<(), String>,
        run: impl for<'scope> FnOnce(&SemanticCollectionScope<'scope>) -> Result<R, String>,
    ) -> Result<R, String> {
        let check = || self.recheck_with_pool(external, pins, &check_pool);
        check()?;
        let result = (|| {
            let scope = SemanticCollectionScope {
                actor: self.actor,
                critic: self.critic,
                parent: self.parent,
                kernel: Kernel::Scalar.resolve()?,
            };
            run(&scope)
        })();
        match (result, check()) {
            (Ok(value), Ok(())) => Ok(value),
            (Err(error), Ok(())) => Err(error),
            (Ok(_), Err(error)) => Err(format!("Semantic source postcheck: {error}")),
            (Err(error), Err(post)) => Err(format!("{error}; semantic source postcheck: {post}")),
        }
    }
    pub(crate) fn recheck_with_pool(
        &self,
        external: &ConditionalNativeExclusions,
        pins: &[ContentPin],
        check_pool: impl for<'check> Fn(&mut SemanticSourceCheckPass<'check>) -> Result<(), String>,
    ) -> Result<(), String> {
        with_source_check_pass(|checked| {
            let source = (|| {
                self.origin.recheck_in(checked)?;
                checked.external(external)?;
                if let OriginRef::Native(origin) = self.origin {
                    origin.check_external(external)?;
                }
                for pin in pins {
                    checked.pin(pin)?;
                }
                Ok::<_, String>(())
            })();
            // Attempt pool validation even when the owner check failed. Each
            // POST gets a new pass, so no successful PRE entry can mask mutation.
            match (source, check_pool(checked)) {
                (Ok(()), Ok(())) => Ok(()),
                (Err(error), Ok(())) => Err(error),
                (Ok(()), Err(error)) => Err(format!("Semantic pool source check: {error}")),
                (Err(error), Err(pool)) => {
                    Err(format!("{error}; semantic pool source check: {pool}"))
                }
            }
        })
    }
}

/// Copied source metadata, never a numerical owner or a constructor for one.
pub(crate) struct SemanticOriginCustody {
    checksum: String,
    pins: Vec<ContentPin>,
    reservations: BTreeMap<String, DatasetSplit>,
    native: Option<Arc<NativeOrigin>>,
}
impl SemanticOriginCustody {
    pub(crate) fn recheck_in(
        &self,
        checked: &mut SemanticSourceCheckPass<'_>,
    ) -> Result<(), String> {
        for pin in &self.pins {
            checked.pin(pin)?;
        }
        if let Some(native) = &self.native {
            native.recheck_in(checked)?;
        }
        Ok(())
    }
    pub(crate) fn recheck(&self) -> Result<(), String> {
        for pin in &self.pins {
            crate::human_policy_dataset::recheck_absolute_pin(pin)?;
        }
        if let Some(native) = &self.native {
            native.recheck_backing()?;
        }
        Ok(())
    }
    pub(crate) fn binding(&self) -> Result<String, String> {
        if let Some(native) = &self.native {
            hash(
                match native.prepared.profile() {
                    TeacherProfile::V5 => b"native-corn-first-v5-frozen-origin-custody-v1\0",
                    TeacherProfile::V7 => b"native-corn-first-v7-frozen-origin-custody-v1\0",
                },
                &(&self.checksum, &native.description),
            )
        } else {
            hash(
                b"semantic-frozen-origin-custody-v1\0",
                &(&self.checksum, &self.pins, &self.reservations),
            )
        }
    }
    pub(crate) fn contains_native_family(&self, family: &str) -> bool {
        self.native.as_ref().is_some_and(|n| {
            n.prepared
                .rl_checkpoint()
                .dataset
                .manifest
                .families
                .iter()
                .any(|f| f.family_id == family)
        })
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
        native: None,
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
