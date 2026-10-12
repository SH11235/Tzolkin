//! Audited V5-distillation storage and a distinct, borrowed Development Arena role.
//! Source consistency does not authenticate training history or measure strength.
pub use super::numeric::run_resource_cycle_teacher_validation;
use crate::dataset::{DatasetSplit, seed_family_id, split_for_family};
use crate::features::FeatureEncoder;
use crate::kernel::{Kernel, ResolvedKernel};
use crate::policy_dataset::{ValidatedPolicyDataset, load_resource_cycle_v5_teacher_dataset};
use crate::public_model::{NumericPolicyDistribution, ValueValidity, predict_numeric};
use crate::resource_cycle_teacher_training::{
    TeacherDatasetIdentity, TeacherTrainingCheckpoint, audit_dataset, validate_checkpoint_dataset,
};
use serde::Serialize;
use sha2::{Digest, Sha256};
use std::path::Path;
use tzolkin_core::observation::Observation;
use tzolkin_inference::policy::validate_contract;
use tzolkin_inference::trade_guard::TradeGuardConfig;

pub const POLICY_VERSION: &str = "resource-cycle-v5-learned-teacher-allphase-guarded-argmax-v1";
pub const ARENA_SCHEMA: &str = "tzolkin-resource-cycle-v5-learned-teacher-development-arena-v1";

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
