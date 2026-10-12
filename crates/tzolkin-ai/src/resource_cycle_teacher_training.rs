//! Policy-only distillation from the dedicated, regenerated all-seat V5 route.
//! Consistency is checked; teacher quality and producer authenticity are not claimed.
use crate::arena::resource_cycle::ResourceCycleConfig;
use crate::dataset::{DatasetSplit, split_for_family};
use crate::features::{FEATURE_COUNT, PUBLIC_FEATURE_SCHEMA};
use crate::model::{Random, digest};
use crate::policy_dataset::{
    MAX_FILES, MAX_PARTITION_CANDIDATES, MAX_PARTITION_SAMPLES,
    RESOURCE_CYCLE_V5_TEACHER_SOURCE_KIND, RESOURCE_CYCLE_V5_TEACHER_TASK, ValidatedPolicyDataset,
    resource_cycle_v5_teacher_policy_id,
};
use crate::policy_training::{
    MAX_CANDIDATE_EVALUATIONS, MAX_UPDATES, PolicyAdamState, PolicyBcConfig, PolicyDatasetIdentity,
    PolicyEpochState, PolicyLossReport, PolicyTrainingMetrics, Scratch, bounded_json, evaluate,
    publish, read_local_json, train_epochs, validate_new_output_directory,
};
use crate::public_model::{INPUT_CONTRACT, MODEL_VERSION, PublicPolicyModel, ValueValidity};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::fs;
use std::path::Path;

pub const CHECKPOINT_SCHEMA: &str = "tzolkin-resource-cycle-v5-teacher-checkpoint-v1";
pub const TRAINING_VERSION: &str = "resource-cycle-v5-policy-distillation-scalar-adam-v1";
pub const TASK: &str = "resourceCycleV5PolicyOnlyDistillation";

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct TeacherFamily {
    pub family_id: String,
    pub split: DatasetSplit,
    pub games: usize,
    pub samples: usize,
    pub candidates: usize,
}
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct TeacherDatasetIdentity {
    pub dataset: PolicyDatasetIdentity,
    /// Every source family, including Test integrity metadata; not optimization targets.
    pub families: Vec<TeacherFamily>,
}
fn split_index(split: DatasetSplit) -> usize {
    match split {
        DatasetSplit::Train => 0,
        DatasetSplit::Validation => 1,
        DatasetSplit::Test => 2,
    }
}
impl TeacherDatasetIdentity {
    fn from_dataset(dataset: &ValidatedPolicyDataset) -> Result<Self, String> {
        dataset.require_resource_cycle_v5_teacher_route()?;
        let mut families = BTreeMap::<String, TeacherFamily>::new();
        for game in &dataset.manifest().games {
            let f = families
                .entry(game.family_id.clone())
                .or_insert(TeacherFamily {
                    family_id: game.family_id.clone(),
                    split: game.split,
                    games: 0,
                    samples: 0,
                    candidates: 0,
                });
            if f.split != game.split {
                return Err("Teacher family split changed".into());
            }
            f.games += 1;
            f.samples += game.samples;
            f.candidates += game.candidates;
        }
        let identity = Self {
            dataset: PolicyDatasetIdentity::from_dataset(dataset),
            families: families.into_values().collect(),
        };
        identity.validate(false)?;
        Ok(identity)
    }
    fn validate(&self, require_training: bool) -> Result<(), String> {
        let d = &self.dataset;
        if d.source_kind != RESOURCE_CYCLE_V5_TEACHER_SOURCE_KIND
            || d.task != RESOURCE_CYCLE_V5_TEACHER_TASK
            || self.families.is_empty()
            || self.families.len() > MAX_FILES
        {
            return Err("Incompatible V5 teacher dataset identity".into());
        }
        d.validate_structure(require_training)?;
        for stratum in &d.strata {
            if stratum.policy_id != resource_cycle_v5_teacher_policy_id(stratum.players)? {
                return Err("Teacher strata require the exact all-seat V5 configuration".into());
            }
        }
        let mut samples = [0usize; 3];
        let mut candidates = [0usize; 3];
        let mut games = 0usize;
        let mut previous = None;
        for family in &self.families {
            if previous.is_some_and(|p: &str| p >= family.family_id.as_str())
                || split_for_family(&family.family_id)? != family.split
                || family.games == 0
                || family.games > MAX_FILES
                || family.samples == 0
                || family.samples > MAX_PARTITION_SAMPLES
                || family.candidates < family.samples
                || family.candidates > MAX_PARTITION_CANDIDATES
            {
                return Err("Invalid V5 teacher family closure".into());
            }
            previous = Some(family.family_id.as_str());
            let i = split_index(family.split);
            samples[i] = samples[i]
                .checked_add(family.samples)
                .ok_or("Teacher sample count overflow")?;
            candidates[i] = candidates[i]
                .checked_add(family.candidates)
                .ok_or("Teacher candidate count overflow")?;
            games = games
                .checked_add(family.games)
                .ok_or("Teacher game count overflow")?;
        }
        if samples != d.samples
            || candidates != d.candidates
            || games > MAX_FILES
            || games != d.strata.iter().map(|s| s.games).sum::<usize>()
        {
            return Err("Teacher family closure/count mismatch".into());
        }
        Ok(())
    }
}
fn audit_dataset(dataset: &ValidatedPolicyDataset) -> Result<TeacherDatasetIdentity, String> {
    let identity = TeacherDatasetIdentity::from_dataset(dataset)?;
    let mut counts = [0usize; 2];
    for sample in dataset.iter() {
        let sample = sample?;
        if sample.value_target().is_some() {
            return Err("V5 distillation cannot consume value labels".into());
        }
        counts[0] += 1;
        counts[1] += sample.candidates().len();
    }
    if counts
        != [
            identity.dataset.samples.iter().sum::<usize>(),
            identity.dataset.candidates.iter().sum::<usize>(),
        ]
        || TeacherDatasetIdentity::from_dataset(dataset)? != identity
    {
        return Err("Teacher source changed during full integrity audit".into());
    }
    Ok(identity)
}

/// A distinct teacher checkpoint, never a native BC artifact or learning owner.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct TeacherTrainingCheckpoint {
    pub schema: String,
    pub training_version: String,
    pub source_kind: String,
    pub task: String,
    pub model_version: String,
    pub input_contract: String,
    pub feature_schema: u32,
    pub feature_count: usize,
    pub backend: String,
    pub value_validity: ValueValidity,
    pub teacher: ResourceCycleConfig,
    pub dataset: TeacherDatasetIdentity,
    pub config: PolicyBcConfig,
    pub completed_epochs: usize,
    pub random_state: u64,
    pub model: PublicPolicyModel,
    pub optimizer: PolicyAdamState,
    pub metrics: PolicyTrainingMetrics,
    pub checksum: String,
}
impl TeacherTrainingCheckpoint {
    fn expected_checksum(&self) -> Result<String, String> {
        let mut copy = self.clone();
        copy.checksum.clear();
        Ok(digest(
            &serde_json::to_vec(&copy).map_err(|e| e.to_string())?,
        ))
    }
    pub fn validate(&self) -> Result<(), String> {
        self.config.validate()?;
        self.dataset.validate(true)?;
        self.model.validate()?;
        self.optimizer.validate()?;
        if self.schema != CHECKPOINT_SCHEMA
            || self.training_version != TRAINING_VERSION
            || self.source_kind != RESOURCE_CYCLE_V5_TEACHER_SOURCE_KIND
            || self.task != TASK
            || self.model_version != MODEL_VERSION
            || self.input_contract != INPUT_CONTRACT
            || self.feature_schema != PUBLIC_FEATURE_SCHEMA
            || self.feature_count != FEATURE_COUNT
            || self.backend != "scalar"
            || self.value_validity != ValueValidity::UnavailablePolicyOnly
            || self.teacher != ResourceCycleConfig::v5()
            || self.completed_epochs == 0
            || self.completed_epochs != self.config.epochs
            || self.metrics.completed_epochs != self.completed_epochs
            || self.metrics.strength_measured
            || self.optimizer.step
                != self.dataset.dataset.samples[0].div_ceil(self.config.batch_size) as u64
                    * self.completed_epochs as u64
        {
            return Err("Invalid V5 teacher checkpoint contract".into());
        }
        let d = &self.dataset.dataset;
        for r in [&self.metrics.initial_train, &self.metrics.final_train] {
            r.validate(d.samples[0], d.candidates[0])?;
        }
        for r in [
            &self.metrics.initial_validation,
            &self.metrics.final_validation,
        ] {
            r.validate(d.samples[1], d.candidates[1])?;
        }
        if self.checksum.len() != 64 || self.checksum != self.expected_checksum()? {
            return Err("V5 teacher checkpoint checksum mismatch".into());
        }
        Ok(())
    }
    pub fn load(path: &Path) -> Result<Self, String> {
        let checkpoint: Self = read_local_json(path)?;
        checkpoint.validate()?;
        Ok(checkpoint)
    }
    pub fn save_new(&self, path: &Path) -> Result<(), String> {
        self.validate()?;
        publish(path, self)
    }
}
#[derive(Clone, Debug, Serialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct TeacherTrainingOutcome {
    pub checkpoint: TeacherTrainingCheckpoint,
    pub metrics: PolicyTrainingMetrics,
}
impl TeacherTrainingOutcome {
    pub fn save_new_directory(&self, directory: &Path) -> Result<(), String> {
        self.checkpoint.validate()?;
        if self.metrics != self.checkpoint.metrics {
            return Err("Inconsistent V5 teacher outcome".into());
        }
        validate_new_output_directory(directory)?;
        bounded_json(&self.checkpoint)?;
        bounded_json(&self.metrics)?;
        fs::create_dir(directory).map_err(|e| e.to_string())?;
        publish(&directory.join("checkpoint.json"), &self.checkpoint)?;
        // No generic/native model.json is emitted; source identity remains in the checkpoint.
        publish(&directory.join("metrics.json"), &self.metrics)
    }
}
/// Fresh or same-route deterministic resume. Test is integrity-only, never a loss target.
pub fn train_dataset(
    dataset: &ValidatedPolicyDataset,
    config: &PolicyBcConfig,
    resume: Option<&TeacherTrainingCheckpoint>,
) -> Result<TeacherTrainingOutcome, String> {
    config.validate()?;
    let completed = resume.map_or(0, |p| p.completed_epochs);
    if let Some(p) = resume {
        p.validate()?;
        if !config.compatible(&p.config) || config.epochs < completed {
            return Err("Teacher resume dataset/config/epoch mismatch".into());
        }
    }
    let identity = audit_dataset(dataset)?;
    identity.validate(true)?;
    if resume.is_some_and(|p| p.dataset != identity) {
        return Err("Teacher resume dataset/config/epoch mismatch".into());
    }
    let d = &identity.dataset;
    let remaining = config.epochs - completed;
    let evaluations = d.candidates[0]
        .checked_mul(remaining + 2)
        .and_then(|n| {
            d.candidates[1]
                .checked_mul(2)
                .and_then(|v| n.checked_add(v))
        })
        .ok_or("Teacher evaluation budget overflow")?;
    let updates = d.samples[0].div_ceil(config.batch_size) as u64 * config.epochs as u64;
    if evaluations > MAX_CANDIDATE_EVALUATIONS || updates > MAX_UPDATES {
        return Err("Teacher candidate/update budget exceeded".into());
    }
    let mut model = resume.map_or_else(
        || crate::public_model::initialized_model(config.seed),
        |p| p.model.clone(),
    );
    model.validate()?;
    let mut optimizer = resume.map_or_else(PolicyAdamState::new, |p| p.optimizer.clone());
    let mut random = Random {
        state: resume.map_or(config.seed ^ 0xd1b54a32d192ed03, |p| p.random_state),
    };
    let mut scratch = Scratch::default();
    let train = evaluate(dataset, d, &model, DatasetSplit::Train, &mut scratch)?;
    let validation = evaluate(dataset, d, &model, DatasetSplit::Validation, &mut scratch)?;
    let mut metrics = if let Some(p) = resume {
        if train != p.metrics.final_train || validation != p.metrics.final_validation {
            return Err("Teacher resume metric/model/source mismatch".into());
        }
        p.metrics.clone()
    } else {
        PolicyTrainingMetrics {
            completed_epochs: 0,
            initial_train: train.clone(),
            initial_validation: validation.clone(),
            final_train: train,
            final_validation: validation,
            strength_measured: false,
        }
    };
    train_epochs(
        dataset,
        config,
        completed,
        d.samples[0],
        &mut model,
        PolicyEpochState {
            optimizer: &mut optimizer,
            random: &mut random,
            scratch: &mut scratch,
        },
    )?;
    metrics.completed_epochs = config.epochs;
    metrics.final_train = evaluate(dataset, d, &model, DatasetSplit::Train, &mut scratch)?;
    metrics.final_validation =
        evaluate(dataset, d, &model, DatasetSplit::Validation, &mut scratch)?;
    if audit_dataset(dataset)? != identity {
        return Err("Teacher dataset changed before completion".into());
    }
    let mut checkpoint = TeacherTrainingCheckpoint {
        schema: CHECKPOINT_SCHEMA.into(),
        training_version: TRAINING_VERSION.into(),
        source_kind: RESOURCE_CYCLE_V5_TEACHER_SOURCE_KIND.into(),
        task: TASK.into(),
        model_version: MODEL_VERSION.into(),
        input_contract: INPUT_CONTRACT.into(),
        feature_schema: PUBLIC_FEATURE_SCHEMA,
        feature_count: FEATURE_COUNT,
        backend: "scalar".into(),
        value_validity: ValueValidity::UnavailablePolicyOnly,
        teacher: ResourceCycleConfig::v5(),
        dataset: identity,
        config: config.clone(),
        completed_epochs: config.epochs,
        random_state: random.state,
        model,
        optimizer,
        metrics: metrics.clone(),
        checksum: String::new(),
    };
    checkpoint.checksum = checkpoint.expected_checksum()?;
    checkpoint.validate()?;
    Ok(TeacherTrainingOutcome {
        checkpoint,
        metrics,
    })
}
/// Full source/EOF checks plus exact final-loss recomputation; no training-history authentication.
pub fn validate_checkpoint_dataset(
    dataset: &ValidatedPolicyDataset,
    checkpoint: &TeacherTrainingCheckpoint,
) -> Result<(), String> {
    let audited = train_dataset(dataset, &checkpoint.config, Some(checkpoint))?;
    if audited.checkpoint != *checkpoint {
        return Err("Teacher checkpoint differs from audited no-op resume".into());
    }
    Ok(())
}
/// Explicit partition evaluation from the same complete teacher source, without updates.
pub fn evaluate_dataset(
    dataset: &ValidatedPolicyDataset,
    checkpoint: &TeacherTrainingCheckpoint,
    split: DatasetSplit,
) -> Result<PolicyLossReport, String> {
    checkpoint.validate()?;
    let identity = audit_dataset(dataset)?;
    if identity != checkpoint.dataset {
        return Err("Teacher evaluation dataset mismatch".into());
    }
    let report = evaluate(
        dataset,
        &identity.dataset,
        &checkpoint.model,
        split,
        &mut Scratch::default(),
    )?;
    if audit_dataset(dataset)? != identity {
        return Err("Teacher evaluation source changed".into());
    }
    Ok(report)
}
