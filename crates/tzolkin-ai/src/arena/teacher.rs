//! Audited V5-distillation storage and a distinct, borrowed Development Arena role.
//! Source consistency does not authenticate training history or measure strength.
pub use super::numeric::{
    run_resource_cycle_teacher_validation, run_semantic_setup_resource_cycle_teacher_validation,
};
use crate::dataset::{DatasetSplit, seed_family_id, split_for_family};
use crate::features::FeatureEncoder;
use crate::kernel::{Kernel, ResolvedKernel};
use crate::policy_dataset::{
    ValidatedPolicyDataset, ValidatedSemanticSetupTeacherDataset,
    load_resource_cycle_v5_semantic_setup_teacher_dataset, load_resource_cycle_v5_teacher_dataset,
};
use crate::public_model::{NumericPolicyDistribution, ValueValidity, predict_numeric};
use crate::resource_cycle_teacher_training::{
    SemanticSetupTeacherModel, SemanticSetupTeacherTrainingCheckpoint, TeacherDatasetIdentity,
    TeacherTrainingCheckpoint, audit_dataset, audit_semantic_setup_dataset,
    validate_checkpoint_dataset, validate_semantic_setup_checkpoint_dataset,
};
use serde::Serialize;
use sha2::{Digest, Sha256};
use std::path::Path;
use tzolkin_core::observation::Observation;
use tzolkin_inference::policy::{BorrowedSemanticSetupPolicy, validate_contract};
use tzolkin_inference::trade_guard::TradeGuardConfig;

pub const POLICY_VERSION: &str = "resource-cycle-v5-learned-teacher-allphase-guarded-argmax-v1";
pub const ARENA_SCHEMA: &str = "tzolkin-resource-cycle-v5-learned-teacher-development-arena-v1";
pub const SEMANTIC_SETUP_POLICY_VERSION: &str =
    "resource-cycle-v5-semantic-setup-learned-teacher-allphase-guarded-argmax-v1";
pub const SEMANTIC_SETUP_ARENA_SCHEMA: &str =
    "tzolkin-resource-cycle-v5-semantic-setup-learned-teacher-development-arena-v1";

/// Full typed checkpoint plus its dedicated regenerated source. No native
/// artifact, raw-weight constructor, training owner or roster conversion exists.
pub struct PreparedResourceCycleTeacher {
    checkpoint: Box<TeacherTrainingCheckpoint>,
    dataset: ValidatedPolicyDataset,
    manifest_pin: (u64, String),
    kernel: ResolvedKernel,
    parameter_checksum: String,
}
impl PreparedResourceCycleTeacher {
    pub fn load(checkpoint: &Path, dataset: &Path, kernel: Kernel) -> Result<Self, String> {
        Self::new(
            TeacherTrainingCheckpoint::load(checkpoint)?,
            load_resource_cycle_v5_teacher_dataset(dataset)?,
            kernel,
        )
    }
    pub fn new(
        checkpoint: TeacherTrainingCheckpoint,
        dataset: ValidatedPolicyDataset,
        kernel: Kernel,
    ) -> Result<Self, String> {
        let kernel = kernel.resolve()?;
        let manifest_pin = dataset.recheck_resource_cycle_v5_teacher_manifest()?;
        validate_checkpoint_dataset(&dataset, &checkpoint)?;
        if dataset.recheck_resource_cycle_v5_teacher_manifest()? != manifest_pin {
            return Err("Teacher manifest changed during preparation".into());
        }
        let mut hash = Sha256::new();
        hash.update(b"resource-cycle-v5-teacher-parameters-f32le-v1\0");
        for parameter in checkpoint.model.parameters() {
            hash.update(parameter.to_bits().to_le_bytes());
        }
        Ok(Self {
            checkpoint: Box::new(checkpoint),
            dataset,
            manifest_pin,
            kernel,
            parameter_checksum: format!("{:x}", hash.finalize()),
        })
    }
    pub fn arena_handle(
        &self,
        guard: TradeGuardConfig,
    ) -> Result<ResourceCycleTeacherArenaPolicy<'_>, String> {
        let configuration_key = guard.configuration_key()?;
        let checkpoint = &self.checkpoint;
        Ok(ResourceCycleTeacherArenaPolicy {
            prepared: self,
            identity: ResourceCycleTeacherArenaIdentity {
                policy_version: POLICY_VERSION.into(),
                source_kind: checkpoint.source_kind.clone(),
                task: checkpoint.task.clone(),
                training_version: checkpoint.training_version.clone(),
                model_version: checkpoint.model_version.clone(),
                input_contract: checkpoint.input_contract.clone(),
                feature_schema: checkpoint.feature_schema,
                feature_count: checkpoint.feature_count,
                checkpoint_checksum: checkpoint.checksum.clone(),
                parameter_checksum: self.parameter_checksum.clone(),
                dataset: checkpoint.dataset.clone(),
                teacher: checkpoint.teacher.clone(),
                backend: self.kernel.backend().into(),
                guard,
                configuration_key,
                value_validity: checkpoint.value_validity,
                producer_authenticated: false,
                source_qualified: false,
                verified_native_policy: false,
                strength_measured: false,
            },
        })
    }
    /// Ordered full legal-set scores in Setup and Playing; no value head.
    /// This is numeric inference only, without source I/O or identity copies.
    pub fn distribution(
        &self,
        observation: &Observation,
    ) -> Result<NumericPolicyDistribution, String> {
        validate_contract(observation)?;
        if observation.legal_actions.len() > super::numeric::MAX_ROWS_PER_CALLBACK {
            return Err("Teacher candidate-row cap exceeded before prediction".into());
        }
        let encoder = FeatureEncoder::new_public(observation)?;
        let rows = (0..observation.legal_actions.len())
            .map(|index| encoder.encode_legal_tagged(index))
            .collect::<Result<Vec<_>, _>>()?;
        predict_numeric(&self.checkpoint.model, self.kernel, &rows)
    }
    pub(super) fn recheck_sources(&self) -> Result<(), String> {
        if self.dataset.recheck_resource_cycle_v5_teacher_manifest()? != self.manifest_pin
            || audit_dataset(&self.dataset)? != self.checkpoint.dataset
            || self.dataset.recheck_resource_cycle_v5_teacher_manifest()? != self.manifest_pin
        {
            return Err("Teacher source changed before Arena publication".into());
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ResourceCycleTeacherArenaIdentity {
    pub policy_version: String,
    pub source_kind: String,
    pub task: String,
    pub training_version: String,
    pub model_version: String,
    pub input_contract: String,
    pub feature_schema: u32,
    pub feature_count: usize,
    pub checkpoint_checksum: String,
    pub parameter_checksum: String,
    pub dataset: TeacherDatasetIdentity,
    pub teacher: super::resource_cycle::ResourceCycleConfig,
    pub backend: String,
    pub guard: TradeGuardConfig,
    pub configuration_key: String,
    pub value_validity: ValueValidity,
    pub producer_authenticated: bool,
    pub source_qualified: bool,
    pub verified_native_policy: bool,
    pub strength_measured: bool,
}

/// Immutable borrow of the audited teacher; serialized identity alone cannot construct it.
pub struct ResourceCycleTeacherArenaPolicy<'a> {
    pub(super) prepared: &'a PreparedResourceCycleTeacher,
    identity: ResourceCycleTeacherArenaIdentity,
}
impl ResourceCycleTeacherArenaPolicy<'_> {
    pub fn identity(&self) -> &ResourceCycleTeacherArenaIdentity {
        &self.identity
    }
    /// Read-only preflight, also used by the paired evaluator. No game or
    /// inference is attempted, and this grants no training or Test authority.
    pub fn validate_arena_config(
        &self,
        config: &super::numeric::NumericArenaConfig,
    ) -> Result<(), String> {
        config.validate()?;
        for seed in &config.seeds {
            let family = seed_family_id(*seed);
            if split_for_family(&family)? != DatasetSplit::Validation
                || self
                    .identity
                    .dataset
                    .families
                    .iter()
                    .any(|f| f.family_id == family)
            {
                return Err("Teacher Arena rejects all retained source families".into());
            }
        }
        self.prepared.recheck_sources()
    }
}

/// The semantic encoder applies in both Setup and Playing. This profile cannot
/// select a legacy encoder or a heuristic Setup composition.
#[derive(Clone, Copy, Debug, Serialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub enum SemanticSetupTeacherInputProfile {
    SemanticSetupAllPhase,
}

/// Schema-3 teacher storage, distinct from the schema-2 teacher and all native
/// artifacts. Preparation consumes the dedicated typed source and checkpoint.
pub struct PreparedSemanticSetupResourceCycleTeacher {
    checkpoint: Box<SemanticSetupTeacherTrainingCheckpoint>,
    dataset: ValidatedSemanticSetupTeacherDataset,
    manifest_pin: (u64, String),
    kernel: ResolvedKernel,
    parameter_checksum: String,
}
impl PreparedSemanticSetupResourceCycleTeacher {
    pub fn load(checkpoint: &Path, dataset: &Path, kernel: Kernel) -> Result<Self, String> {
        Self::new(
            SemanticSetupTeacherTrainingCheckpoint::load(checkpoint)?,
            load_resource_cycle_v5_semantic_setup_teacher_dataset(dataset)?,
            kernel,
        )
    }
    /// Legacy teacher checkpoints and datasets cannot enter this role.
    /// ```compile_fail
    /// use tzolkin_ai::arena::teacher::PreparedSemanticSetupResourceCycleTeacher;
    /// use tzolkin_ai::kernel::Kernel;
    /// use tzolkin_ai::policy_dataset::ValidatedPolicyDataset;
    /// use tzolkin_ai::resource_cycle_teacher_training::TeacherTrainingCheckpoint;
    /// fn legacy(checkpoint: TeacherTrainingCheckpoint, dataset: ValidatedPolicyDataset) {
    ///     let _ = PreparedSemanticSetupResourceCycleTeacher::new(checkpoint, dataset, Kernel::Scalar);
    /// }
    /// ```
    pub fn new(
        checkpoint: SemanticSetupTeacherTrainingCheckpoint,
        dataset: ValidatedSemanticSetupTeacherDataset,
        kernel: Kernel,
    ) -> Result<Self, String> {
        let kernel = kernel.resolve()?;
        let manifest_pin = dataset.recheck_teacher_bytes()?;
        validate_semantic_setup_checkpoint_dataset(&dataset, &checkpoint)?;
        if dataset.recheck_teacher_bytes()? != manifest_pin {
            return Err("Semantic Setup teacher manifest changed during preparation".into());
        }
        let mut hash = Sha256::new();
        hash.update(b"resource-cycle-v5-semantic-setup-teacher-parameters-f32le-v1\0");
        for parameter in checkpoint.model.parameters() {
            hash.update(parameter.to_bits().to_le_bytes());
        }
        Ok(Self {
            checkpoint: Box::new(checkpoint),
            dataset,
            manifest_pin,
            kernel,
            parameter_checksum: format!("{:x}", hash.finalize()),
        })
    }
    pub fn arena_handle(
        &self,
        guard: TradeGuardConfig,
    ) -> Result<SemanticSetupResourceCycleTeacherArenaPolicy<'_>, String> {
        let configuration_key = guard.configuration_key()?;
        let checkpoint = &self.checkpoint;
        Ok(SemanticSetupResourceCycleTeacherArenaPolicy {
            prepared: self,
            identity: SemanticSetupTeacherArenaIdentity {
                input_profile: SemanticSetupTeacherInputProfile::SemanticSetupAllPhase,
                teacher: ResourceCycleTeacherArenaIdentity {
                    policy_version: SEMANTIC_SETUP_POLICY_VERSION.into(),
                    source_kind: checkpoint.source_kind.clone(),
                    task: checkpoint.task.clone(),
                    training_version: checkpoint.training_version.clone(),
                    model_version: checkpoint.model_version.clone(),
                    input_contract: checkpoint.input_contract.clone(),
                    feature_schema: checkpoint.feature_schema,
                    feature_count: checkpoint.feature_count,
                    checkpoint_checksum: checkpoint.checksum.clone(),
                    parameter_checksum: self.parameter_checksum.clone(),
                    dataset: checkpoint.dataset.clone(),
                    teacher: checkpoint.teacher.clone(),
                    backend: self.kernel.backend().into(),
                    guard,
                    configuration_key,
                    value_validity: checkpoint.value_validity,
                    producer_authenticated: false,
                    source_qualified: false,
                    verified_native_policy: false,
                    strength_measured: false,
                },
            },
        })
    }
    /// Numeric-only inference over the ordered full legal set, with schema-3
    /// rows in both phases. Source I/O and identity copies stay at boundaries.
    pub fn distribution(
        &self,
        observation: &Observation,
    ) -> Result<NumericPolicyDistribution, String> {
        semantic_setup_distribution(&self.checkpoint.model, self.kernel, observation)
    }
    pub(super) fn recheck_sources(&self) -> Result<(), String> {
        if self.dataset.recheck_teacher_bytes()? != self.manifest_pin
            || audit_semantic_setup_dataset(&self.dataset)? != self.checkpoint.dataset
            || self.dataset.recheck_teacher_bytes()? != self.manifest_pin
        {
            return Err("Semantic Setup teacher source changed before Arena publication".into());
        }
        Ok(())
    }
}

pub(super) fn semantic_setup_distribution(
    model: &SemanticSetupTeacherModel,
    kernel: ResolvedKernel,
    observation: &Observation,
) -> Result<NumericPolicyDistribution, String> {
    validate_contract(observation)?;
    if observation.legal_actions.len() > super::numeric::MAX_ROWS_PER_CALLBACK {
        return Err("Semantic Setup teacher candidate-row cap exceeded before prediction".into());
    }
    let encoder = FeatureEncoder::new_semantic_setup(observation)?;
    let rows = (0..observation.legal_actions.len())
        .map(|index| encoder.encode_legal_tagged(index))
        .collect::<Result<Vec<_>, _>>()?;
    BorrowedSemanticSetupPolicy::from_validated_parameters(model.parameters())?
        .predict(kernel, &rows)
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct SemanticSetupTeacherArenaIdentity {
    pub input_profile: SemanticSetupTeacherInputProfile,
    #[serde(flatten)]
    pub teacher: ResourceCycleTeacherArenaIdentity,
}

/// Immutable schema-3 borrow. A serialized report identity cannot construct it.
pub struct SemanticSetupResourceCycleTeacherArenaPolicy<'a> {
    pub(super) prepared: &'a PreparedSemanticSetupResourceCycleTeacher,
    identity: SemanticSetupTeacherArenaIdentity,
}
impl SemanticSetupResourceCycleTeacherArenaPolicy<'_> {
    pub fn identity(&self) -> &SemanticSetupTeacherArenaIdentity {
        &self.identity
    }
    pub fn validate_arena_config(
        &self,
        config: &super::numeric::NumericArenaConfig,
    ) -> Result<(), String> {
        config.validate()?;
        for seed in &config.seeds {
            let family = seed_family_id(*seed);
            if split_for_family(&family)? != DatasetSplit::Validation
                || self
                    .identity
                    .teacher
                    .dataset
                    .families
                    .iter()
                    .any(|f| f.family_id == family)
            {
                return Err(
                    "Semantic Setup teacher Arena rejects all retained source families".into(),
                );
            }
        }
        self.prepared.recheck_sources()
    }
}

/// Existing validated baselines retain their own encoder and admission rules.
/// Neither variant accepts a semantic teacher or an unvalidated artifact.
pub enum SemanticSetupTeacherReference<'a> {
    GuardedNumeric(Box<super::numeric::NumericPolicy>),
    ConditionalHumanCountZeroAllPhase(Box<super::conditional::ConditionalCountZeroArenaPolicy<'a>>),
}

pub const SEMANTIC_PLAYING_POLICY_VERSION: &str =
    "native-v5-teacher-playing4-corn-first-guarded-argmax-v1";
pub const SEMANTIC_PLAYING_ARENA_SCHEMA: &str =
    "tzolkin-native-playing4-teacher-paired-development-arena-v1";

/// Only a dedicated checkpoint and its fully regenerated original dataset can
/// construct this inference role. Raw parameters and report identities cannot.
pub struct PreparedSemanticPlayingTeacher {
    checkpoint: Box<crate::semantic_playing_teacher_training::SemanticPlayingTeacherCheckpoint>,
    dataset: crate::semantic_playing_teacher_dataset::ValidatedSemanticPlayingTeacherDataset,
    dataset_path: std::path::PathBuf,
    checkpoint_path: std::path::PathBuf,
    checkpoint_pin: (usize, String),
    kernel: ResolvedKernel,
}
impl PreparedSemanticPlayingTeacher {
    pub fn load(checkpoint: &Path, dataset_path: &Path) -> Result<Self, String> {
        use crate::semantic_playing_teacher_dataset as data;
        use crate::semantic_playing_teacher_training as training;
        let before = data::read(checkpoint, crate::model::MAX_ARTIFACT_BYTES as u64)?;
        let checkpoint_pin = (before.len(), crate::model::digest(&before));
        let result = (|| {
            let loaded = training::SemanticPlayingTeacherCheckpoint::load(checkpoint)?;
            let dataset = data::load_semantic_playing_teacher_dataset(dataset_path)?;
            // Exactly one audited no-op outside callbacks. This checks current
            // source/model/metrics consistency, not training-history authenticity.
            training::validate_checkpoint_dataset(&dataset, &loaded)?;
            Ok(Self {
                checkpoint: Box::new(loaded),
                dataset,
                dataset_path: dataset_path.to_path_buf(),
                checkpoint_path: checkpoint.to_path_buf(),
                checkpoint_pin: checkpoint_pin.clone(),
                kernel: Kernel::Scalar.resolve()?,
            })
        })();
        let after = data::read(checkpoint, crate::model::MAX_ARTIFACT_BYTES as u64)?;
        if (after.len(), crate::model::digest(&after)) != checkpoint_pin {
            return Err("Native Playing teacher checkpoint changed during load".into());
        }
        result
    }
    pub fn arena_handle(&self) -> Result<SemanticPlayingTeacherArenaPolicy<'_>, String> {
        self.recheck_sources()?;
        let guard = TradeGuardConfig::default();
        Ok(SemanticPlayingTeacherArenaPolicy {
            prepared: self,
            identity: SemanticPlayingTeacherArenaIdentity {
                policy_version: SEMANTIC_PLAYING_POLICY_VERSION.into(),
                model_version: crate::semantic_playing_teacher_training::MODEL_VERSION.into(),
                training_version: self.checkpoint.training_version.clone(),
                task: crate::semantic_playing_teacher_training::TASK.into(),
                input_contract: crate::semantic_playing_teacher_dataset::INPUT_CONTRACT.into(),
                source_kind: crate::semantic_playing_teacher_dataset::SOURCE_KIND.into(),
                checkpoint_checksum: self.checkpoint.checksum.clone(),
                checkpoint_bytes_sha256: self.checkpoint_pin.1.clone(),
                dataset: self.checkpoint.dataset.clone(),
                feature_schema: crate::features::SEMANTIC_PLAYING_FEATURE_SCHEMA,
                feature_count: crate::features::FEATURE_COUNT,
                setup_policy_version: crate::setup_policy::POLICY_VERSION.into(),
                setup_weights: crate::policy::HeuristicWeights::default(),
                backend: self.kernel.backend().into(),
                configuration_key: guard.configuration_key()?,
                guard,
                value_validity: ValueValidity::UnavailablePolicyOnly,
                source_authenticated: false,
                strength_measured: false,
            },
        })
    }
    pub(super) fn distribution(
        &self,
        o: &Observation,
    ) -> Result<NumericPolicyDistribution, String> {
        semantic_playing_distribution(self.checkpoint.model.parameters(), self.kernel, o)
    }
    pub fn recheck_sources(&self) -> Result<(), String> {
        let result = self.dataset.recheck();
        let bytes = crate::semantic_playing_teacher_dataset::read(
            &self.checkpoint_path,
            crate::model::MAX_ARTIFACT_BYTES as u64,
        )?;
        if (bytes.len(), crate::model::digest(&bytes)) != self.checkpoint_pin {
            return Err("Native Playing teacher checkpoint changed after load".into());
        }
        result
    }
    pub(crate) fn rl_checkpoint(
        &self,
    ) -> &crate::semantic_playing_teacher_training::SemanticPlayingTeacherCheckpoint {
        &self.checkpoint
    }
    pub(crate) fn rl_source_locations(&self) -> serde_json::Value {
        serde_json::json!({"checkpoint":{"path":self.checkpoint_path,
            "bytes":self.checkpoint_pin.0,"sha256":self.checkpoint_pin.1},
            "datasetDirectory":self.dataset_path})
    }
    #[cfg(test)]
    pub(crate) fn unqualified_rl_fixture(directory: &Path, observation: Observation) -> Self {
        use crate::semantic_playing_teacher_dataset as data;
        use crate::semantic_playing_teacher_training as training;
        // A bounded numerical/custody oracle, NOT a complete-source loader
        // success. Production must continue to reject this synthetic dataset.
        let chosen = observation.legal_actions[0].clone();
        let first_playing = observation.players.len();
        let dataset =
            data::training_cache_fixture(directory, &[(observation, chosen, first_playing)], None)
                .unwrap();
        let outcome = training::train_dataset(
            &dataset,
            &crate::policy_training::PolicyBcConfig {
                epochs: 1,
                batch_size: 2,
                learning_rate: 0.001,
                seed: 7,
            },
            None,
        )
        .unwrap();
        let checkpoint_path = directory.with_extension("checkpoint.json");
        outcome.checkpoint.save_new(&checkpoint_path).unwrap();
        assert!(Self::load(&checkpoint_path, directory).is_err());
        let raw = data::read(&checkpoint_path, crate::model::MAX_ARTIFACT_BYTES as u64).unwrap();
        Self {
            checkpoint: Box::new(outcome.checkpoint),
            dataset,
            dataset_path: directory.to_path_buf(),
            checkpoint_path,
            checkpoint_pin: (raw.len(), crate::model::digest(&raw)),
            kernel: Kernel::Scalar.resolve().unwrap(),
        }
    }
}
pub(super) fn semantic_playing_distribution(
    parameters: &[f32],
    kernel: ResolvedKernel,
    o: &Observation,
) -> Result<NumericPolicyDistribution, String> {
    validate_contract(o)?;
    if o.phase != tzolkin_core::Phase::Playing
        || o.legal_actions.len() > super::numeric::MAX_ROWS_PER_CALLBACK
    {
        return Err("Native Playing teacher requires bounded Playing rows".into());
    }
    let encoder = FeatureEncoder::new_semantic_playing(o)?;
    let rows = (0..o.legal_actions.len())
        .map(|i| encoder.encode_legal_tagged(i))
        .collect::<Result<Vec<_>, _>>()?;
    tzolkin_inference::policy::BorrowedSemanticPlayingPolicy::from_validated_parameters(parameters)?
        .predict(kernel, &rows)
}
#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct SemanticPlayingTeacherArenaIdentity {
    pub policy_version: String,
    pub model_version: String,
    pub training_version: String,
    pub task: String,
    pub input_contract: String,
    pub source_kind: String,
    pub checkpoint_checksum: String,
    pub checkpoint_bytes_sha256: String,
    pub dataset: crate::semantic_playing_teacher_training::NativeTeacherDatasetIdentity,
    pub feature_schema: u32,
    pub feature_count: usize,
    pub setup_policy_version: String,
    pub setup_weights: crate::policy::HeuristicWeights,
    pub backend: String,
    pub guard: TradeGuardConfig,
    pub configuration_key: String,
    pub value_validity: ValueValidity,
    pub source_authenticated: bool,
    pub strength_measured: bool,
}
pub struct SemanticPlayingTeacherArenaPolicy<'a> {
    pub(super) prepared: &'a PreparedSemanticPlayingTeacher,
    identity: SemanticPlayingTeacherArenaIdentity,
}
impl SemanticPlayingTeacherArenaPolicy<'_> {
    pub fn identity(&self) -> &SemanticPlayingTeacherArenaIdentity {
        &self.identity
    }
    pub(super) fn exclude_source_families(
        &self,
        config: &super::numeric::NumericArenaConfig,
    ) -> Result<(), String> {
        if config.seeds.iter().any(|seed| {
            self.identity
                .dataset
                .manifest
                .families
                .iter()
                .any(|g| g.family_id == seed_family_id(*seed))
        }) {
            return Err("Native Playing Arena rejects every retained teacher source family".into());
        }
        Ok(())
    }
}
/// Explicit fixed reference; no raw weights, Human origin or Search fallback.
pub enum SemanticPlayingTeacherReference {
    ResourceCycleV5,
    GuardedNumeric(Box<super::numeric::NumericPolicy>),
}
