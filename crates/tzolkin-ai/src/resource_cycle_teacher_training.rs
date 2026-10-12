//! Policy-only distillation from the dedicated, regenerated all-seat V5 route.
//! Consistency is checked; teacher quality and producer authenticity are not claimed.
use crate::arena::resource_cycle::ResourceCycleConfig;
use crate::dataset::{DatasetSplit, split_for_family};
use crate::features::{FEATURE_COUNT, PUBLIC_FEATURE_SCHEMA, SEMANTIC_SETUP_FEATURE_SCHEMA};
use crate::model::{Random, digest};
use crate::policy_dataset::{
    MAX_FILES, MAX_PARTITION_CANDIDATES, MAX_PARTITION_SAMPLES,
    RESOURCE_CYCLE_V5_SEMANTIC_SETUP_TEACHER_SOURCE_KIND,
    RESOURCE_CYCLE_V5_SEMANTIC_SETUP_TEACHER_TASK, RESOURCE_CYCLE_V5_TEACHER_SOURCE_KIND,
    RESOURCE_CYCLE_V5_TEACHER_TASK, ValidatedPolicyDataset, ValidatedPolicySample,
    ValidatedSemanticSetupTeacherDataset, resource_cycle_v5_teacher_policy_id,
};
use crate::policy_training::{
    MAX_CANDIDATE_EVALUATIONS, MAX_UPDATES, PolicyAdamState, PolicyBcConfig, PolicyDatasetIdentity,
    PolicyEpochState, PolicyFeatureProfile, PolicyLossReport, PolicyTrainingMetrics, Scratch,
    bounded_json, evaluate, evaluate_samples_using, publish, read_local_json,
    train_epochs_using_profile, validate_new_output_directory,
};
use crate::public_model::{INPUT_CONTRACT, MODEL_VERSION, PublicPolicyModel, ValueValidity};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::fs;
use std::path::Path;

pub const CHECKPOINT_SCHEMA: &str = "tzolkin-resource-cycle-v5-teacher-checkpoint-v1";
pub const TRAINING_VERSION: &str = "resource-cycle-v5-policy-distillation-scalar-adam-v1";
pub const TASK: &str = "resourceCycleV5PolicyOnlyDistillation";
pub const SEMANTIC_SETUP_CHECKPOINT_SCHEMA: &str =
    "tzolkin-resource-cycle-v5-semantic-setup-teacher-checkpoint-v1";
pub const SEMANTIC_SETUP_TRAINING_VERSION: &str =
    "resource-cycle-v5-semantic-setup-policy-distillation-scalar-adam-v1";
pub const SEMANTIC_SETUP_TASK: &str = "resourceCycleV5SemanticSetupPolicyOnlyDistillation";
pub const SEMANTIC_SETUP_MODEL_VERSION: &str = "tiny-semantic-setup-teacher-policy-mlp-v1";
pub const SEMANTIC_SETUP_INPUT_CONTRACT: &str =
    PolicyFeatureProfile::SemanticSetup.input_contract();
// Account owned Vec/String allocations, not allocator overhead or process RSS.
const MAX_ENCODED_CACHE_BYTES: usize = 256 * 1024 * 1024;

// Private, closed dispatch. No caller can pass another dataset profile or
// convert the semantic handle into the old Native/schema-2 teacher handle.
#[derive(Clone, Copy)]
enum TeacherRoute<'a> {
    Public(&'a ValidatedPolicyDataset),
    SemanticSetup(&'a ValidatedSemanticSetupTeacherDataset),
}
impl<'a> TeacherRoute<'a> {
    fn profile(self) -> PolicyFeatureProfile {
        match self {
            Self::Public(_) => PolicyFeatureProfile::Public,
            Self::SemanticSetup(_) => PolicyFeatureProfile::SemanticSetup,
        }
    }
    fn manifest(self) -> &'a crate::policy_dataset::PolicyDatasetManifest {
        match self {
            Self::Public(d) => d.manifest(),
            Self::SemanticSetup(d) => d.manifest(),
        }
    }
    fn iter(self, split: Option<DatasetSplit>) -> TeacherRows<'a> {
        match self {
            Self::Public(d) => TeacherRows::Public(match split {
                Some(s) => d.iter_split(s),
                None => d.iter(),
            }),
            Self::SemanticSetup(d) => TeacherRows::SemanticSetup(match split {
                Some(s) => d.iter_split(s),
                None => d.iter(),
            }),
        }
    }
    fn recheck(self) -> Result<(u64, String), String> {
        match self {
            Self::Public(d) => d.recheck_resource_cycle_v5_teacher_bytes(),
            Self::SemanticSetup(d) => d.recheck_teacher_bytes(),
        }
    }
}
enum TeacherRows<'a> {
    Public(crate::policy_dataset::PolicyDatasetIter),
    SemanticSetup(crate::policy_dataset::SemanticSetupTeacherDatasetIter<'a>),
}
impl Iterator for TeacherRows<'_> {
    type Item = Result<ValidatedPolicySample, String>;
    fn next(&mut self) -> Option<Self::Item> {
        match self {
            Self::Public(rows) => rows.next(),
            Self::SemanticSetup(rows) => rows.next(),
        }
    }
}

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
    fn from_route(route: TeacherRoute<'_>) -> Result<Self, String> {
        if let TeacherRoute::Public(dataset) = route {
            dataset.require_resource_cycle_v5_teacher_route()?;
        }
        let mut families = BTreeMap::<String, TeacherFamily>::new();
        for game in &route.manifest().games {
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
            dataset: PolicyDatasetIdentity::from_manifest(route.manifest(), route.profile()),
            families: families.into_values().collect(),
        };
        identity.validate_using(false, route.profile())?;
        Ok(identity)
    }
    fn validate(&self, require_training: bool) -> Result<(), String> {
        self.validate_using(require_training, PolicyFeatureProfile::Public)
    }
    fn validate_using(
        &self,
        require_training: bool,
        profile: PolicyFeatureProfile,
    ) -> Result<(), String> {
        let d = &self.dataset;
        let (kind, task) = match profile {
            PolicyFeatureProfile::Public => (
                RESOURCE_CYCLE_V5_TEACHER_SOURCE_KIND,
                RESOURCE_CYCLE_V5_TEACHER_TASK,
            ),
            PolicyFeatureProfile::SemanticSetup => (
                RESOURCE_CYCLE_V5_SEMANTIC_SETUP_TEACHER_SOURCE_KIND,
                RESOURCE_CYCLE_V5_SEMANTIC_SETUP_TEACHER_TASK,
            ),
        };
        if d.source_kind != kind
            || d.task != task
            || self.families.is_empty()
            || self.families.len() > MAX_FILES
        {
            return Err("Incompatible V5 teacher dataset identity".into());
        }
        d.validate_structure_using(require_training, profile)?;
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
pub(crate) fn audit_dataset(
    dataset: &ValidatedPolicyDataset,
) -> Result<TeacherDatasetIdentity, String> {
    audit_route(TeacherRoute::Public(dataset))
}
fn audit_route(route: TeacherRoute<'_>) -> Result<TeacherDatasetIdentity, String> {
    let identity = TeacherDatasetIdentity::from_route(route)?;
    let mut counts = [0usize; 2];
    for sample in route.iter(None) {
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
        || TeacherDatasetIdentity::from_route(route)? != identity
    {
        return Err("Teacher source changed during full integrity audit".into());
    }
    Ok(identity)
}

struct TeacherSamples {
    train: Vec<ValidatedPolicySample>,
    validation: Vec<ValidatedPolicySample>,
    manifest_pin: (u64, String),
    allocation_bytes: usize,
}
impl TeacherSamples {
    #[cfg(test)]
    fn load(
        dataset: &ValidatedPolicyDataset,
        identity: &TeacherDatasetIdentity,
    ) -> Result<Self, String> {
        Self::load_route(TeacherRoute::Public(dataset), identity)
    }
    fn load_route(
        route: TeacherRoute<'_>,
        identity: &TeacherDatasetIdentity,
    ) -> Result<Self, String> {
        let d = &identity.dataset;
        let samples = d.samples[0]
            .checked_add(d.samples[1])
            .ok_or("Teacher cache sample count overflow")?;
        let candidates = d.candidates[0]
            .checked_add(d.candidates[1])
            .ok_or("Teacher cache candidate count overflow")?;
        let minimum = samples
            .checked_mul(std::mem::size_of::<ValidatedPolicySample>())
            .and_then(|n| {
                candidates
                    .checked_mul(std::mem::size_of::<crate::features::EncodedCandidate>())
                    .and_then(|c| n.checked_add(c))
            })
            .ok_or("Teacher cache byte count overflow")?;
        if minimum > MAX_ENCODED_CACHE_BYTES {
            return Err("Teacher encoded cache exceeds 256 MiB".into());
        }
        let manifest_pin = route.recheck()?;
        let mut cache = Self {
            train: Vec::new(),
            validation: Vec::new(),
            manifest_pin,
            allocation_bytes: 0,
        };
        cache
            .train
            .try_reserve_exact(d.samples[0])
            .map_err(|e| e.to_string())?;
        cache
            .validation
            .try_reserve_exact(d.samples[1])
            .map_err(|e| e.to_string())?;
        cache.allocation_bytes = cache
            .train
            .capacity()
            .checked_add(cache.validation.capacity())
            .and_then(|n| n.checked_mul(std::mem::size_of::<ValidatedPolicySample>()))
            .and_then(|n| n.checked_add(cache.manifest_pin.1.capacity()))
            .ok_or("Teacher cache byte count overflow")?;
        if cache.allocation_bytes > MAX_ENCODED_CACHE_BYTES {
            return Err("Teacher encoded cache exceeds 256 MiB".into());
        }
        let splits = identity
            .families
            .iter()
            .map(|f| (f.family_id.as_str(), f.split))
            .collect::<BTreeMap<_, _>>();
        let mut counts = [[0usize; 2]; 3];
        // Consuming every partition to EOF retains full source replay, fixed
        // teacher-choice regeneration and shard checks. Test rows are discarded.
        for sample in route.iter(None) {
            let sample = sample?;
            let split = *splits
                .get(sample.family_id())
                .ok_or("Teacher sample family changed")?;
            let index = split_index(split);
            counts[index][0] += 1;
            counts[index][1] += sample.candidates().len();
            if sample.value_target().is_some() {
                return Err("V5 distillation cannot consume value labels".into());
            }
            let rows = match split {
                DatasetSplit::Train => &mut cache.train,
                DatasetSplit::Validation => &mut cache.validation,
                DatasetSplit::Test => continue,
            };
            cache.allocation_bytes = cache
                .allocation_bytes
                .checked_add(sample.owned_allocation_bytes()?)
                .ok_or("Teacher cache byte count overflow")?;
            if cache.allocation_bytes > MAX_ENCODED_CACHE_BYTES || rows.len() == rows.capacity() {
                return Err("Teacher encoded cache exceeds 256 MiB/count budget".into());
            }
            rows.push(sample);
        }
        for (index, count) in counts.iter().enumerate() {
            if *count != [d.samples[index], d.candidates[index]] {
                return Err("Teacher cache partition count mismatch".into());
            }
        }
        if TeacherDatasetIdentity::from_route(route)? != *identity {
            return Err("Teacher source changed during cache validation".into());
        }
        cache.recheck(route)?;
        Ok(cache)
    }
    fn recheck(&self, route: TeacherRoute<'_>) -> Result<(), String> {
        if route.recheck()? != self.manifest_pin {
            return Err("Teacher cache manifest bytes changed".into());
        }
        Ok(())
    }
    fn evaluate(
        &self,
        identity: &PolicyDatasetIdentity,
        model: &PublicPolicyModel,
        split: DatasetSplit,
        scratch: &mut Scratch,
        profile: PolicyFeatureProfile,
    ) -> Result<PolicyLossReport, String> {
        let rows = match split {
            DatasetSplit::Train => &self.train,
            DatasetSplit::Validation => &self.validation,
            DatasetSplit::Test => return Err("Teacher cache does not retain Test loss rows".into()),
        };
        evaluate_samples_using(
            rows.iter().map(Ok),
            identity,
            model,
            split,
            scratch,
            profile,
        )
    }
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
    train_dataset_using::<true>(dataset, config, resume)
}
fn train_dataset_using<const CACHED: bool>(
    dataset: &ValidatedPolicyDataset,
    config: &PolicyBcConfig,
    resume: Option<&TeacherTrainingCheckpoint>,
) -> Result<TeacherTrainingOutcome, String> {
    train_dataset_after_cache::<CACHED>(dataset, config, resume, || Ok(()))
}
fn train_dataset_after_cache<const CACHED: bool>(
    dataset: &ValidatedPolicyDataset,
    config: &PolicyBcConfig,
    resume: Option<&TeacherTrainingCheckpoint>,
    after_cache: impl FnOnce() -> Result<(), String>,
) -> Result<TeacherTrainingOutcome, String> {
    config.validate()?;
    if let Some(p) = resume {
        p.validate()?;
    }
    train_route_after_cache::<CACHED, _>(
        TeacherRoute::Public(dataset),
        config,
        resume.map(TeacherResume::public),
        after_cache,
        |run| {
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
                dataset: run.identity,
                config: config.clone(),
                completed_epochs: config.epochs,
                random_state: run.random_state,
                model: run.model,
                optimizer: run.optimizer,
                metrics: run.metrics.clone(),
                checksum: String::new(),
            };
            checkpoint.checksum = checkpoint.expected_checksum()?;
            checkpoint.validate()?;
            Ok(TeacherTrainingOutcome {
                checkpoint,
                metrics: run.metrics,
            })
        },
    )
}
struct TeacherResume<'a> {
    dataset: &'a TeacherDatasetIdentity,
    config: &'a PolicyBcConfig,
    completed_epochs: usize,
    random_state: u64,
    parameters: &'a [f32],
    optimizer: &'a PolicyAdamState,
    metrics: &'a PolicyTrainingMetrics,
}
impl<'a> TeacherResume<'a> {
    fn public(p: &'a TeacherTrainingCheckpoint) -> Self {
        Self {
            dataset: &p.dataset,
            config: &p.config,
            completed_epochs: p.completed_epochs,
            random_state: p.random_state,
            parameters: p.model.parameters(),
            optimizer: &p.optimizer,
            metrics: &p.metrics,
        }
    }
}
struct TeacherRun {
    identity: TeacherDatasetIdentity,
    random_state: u64,
    model: PublicPolicyModel,
    optimizer: PolicyAdamState,
    metrics: PolicyTrainingMetrics,
}
fn train_route_after_cache<const CACHED: bool, O>(
    route: TeacherRoute<'_>,
    config: &PolicyBcConfig,
    resume: Option<TeacherResume<'_>>,
    after_cache: impl FnOnce() -> Result<(), String>,
    finish: impl FnOnce(TeacherRun) -> Result<O, String>,
) -> Result<O, String> {
    let completed = resume.as_ref().map_or(0, |p| p.completed_epochs);
    if let Some(p) = &resume
        && (!config.compatible(p.config) || config.epochs < completed)
    {
        return Err("Teacher resume dataset/config/epoch mismatch".into());
    }
    let identity = if CACHED {
        TeacherDatasetIdentity::from_route(route)?
    } else {
        audit_route(route)?
    };
    identity.validate_using(true, route.profile())?;
    if resume.as_ref().is_some_and(|p| *p.dataset != identity) {
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
    let cache = if CACHED {
        Some(TeacherSamples::load_route(route, &identity)?)
    } else {
        None
    };
    let evaluate = |model: &PublicPolicyModel, split, scratch: &mut Scratch| {
        if let Some(cache) = &cache {
            cache.evaluate(d, model, split, scratch, route.profile())
        } else {
            evaluate_samples_using(
                route.iter(Some(split)),
                d,
                model,
                split,
                scratch,
                route.profile(),
            )
        }
    };
    let result = (|| {
        after_cache()?;
        let mut model = resume.as_ref().map_or_else(
            || crate::public_model::initialized_model(config.seed),
            |p| PublicPolicyModel {
                parameters: p.parameters.to_vec(),
            },
        );
        model.validate()?;
        let mut optimizer = resume
            .as_ref()
            .map_or_else(PolicyAdamState::new, |p| (*p.optimizer).clone());
        let mut random = Random {
            state: resume
                .as_ref()
                .map_or(config.seed ^ 0xd1b54a32d192ed03, |p| p.random_state),
        };
        let mut scratch = Scratch::default();
        let train = evaluate(&model, DatasetSplit::Train, &mut scratch)?;
        let validation = evaluate(&model, DatasetSplit::Validation, &mut scratch)?;
        let mut metrics = if let Some(p) = &resume {
            if train != p.metrics.final_train || validation != p.metrics.final_validation {
                return Err("Teacher resume metric/model/source mismatch".into());
            }
            (*p.metrics).clone()
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
        let state = PolicyEpochState {
            optimizer: &mut optimizer,
            random: &mut random,
            scratch: &mut scratch,
        };
        if let Some(cache) = &cache {
            train_epochs_using_profile(
                || cache.train.iter().map(Ok),
                config,
                completed,
                d.samples[0],
                &mut model,
                state,
                route.profile(),
            )?;
        } else {
            train_epochs_using_profile(
                || route.iter(Some(DatasetSplit::Train)),
                config,
                completed,
                d.samples[0],
                &mut model,
                state,
                route.profile(),
            )?;
        }
        metrics.completed_epochs = config.epochs;
        metrics.final_train = evaluate(&model, DatasetSplit::Train, &mut scratch)?;
        metrics.final_validation = evaluate(&model, DatasetSplit::Validation, &mut scratch)?;
        if audit_route(route)? != identity {
            return Err("Teacher dataset changed before completion".into());
        }
        finish(TeacherRun {
            identity: identity.clone(),
            random_state: random.state,
            model,
            optimizer,
            metrics,
        })
    })();
    if let Some(cache) = &cache {
        // Also run on numerical failure. No changed-source cached outcome is returned.
        cache.recheck(route)?;
    }
    result
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

/// Parameters for the explicit semantic teacher checkpoint only. There is no
/// public constructor/conversion from an old policy model or unqualified rows.
/// The distinct serialized key also rejects an old model under retagged metadata.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct SemanticSetupTeacherModel {
    semantic_setup_parameters: Vec<f32>,
}
impl SemanticSetupTeacherModel {
    pub fn parameters(&self) -> &[f32] {
        &self.semantic_setup_parameters
    }
    pub fn validate(&self) -> Result<(), String> {
        tzolkin_inference::policy::validate_parameters(self.parameters())
    }
}

/// A schema-3 teacher checkpoint with its own model and resume contract.
/// Checksums establish consistency, not producer or training-history authenticity.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct SemanticSetupTeacherTrainingCheckpoint {
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
    pub model: SemanticSetupTeacherModel,
    pub optimizer: PolicyAdamState,
    pub metrics: PolicyTrainingMetrics,
    pub checksum: String,
}
impl SemanticSetupTeacherTrainingCheckpoint {
    fn expected_checksum(&self) -> Result<String, String> {
        let mut copy = self.clone();
        copy.checksum.clear();
        Ok(digest(
            &serde_json::to_vec(&copy).map_err(|e| e.to_string())?,
        ))
    }
    pub fn validate(&self) -> Result<(), String> {
        self.config.validate()?;
        self.dataset
            .validate_using(true, PolicyFeatureProfile::SemanticSetup)?;
        self.model.validate()?;
        self.optimizer.validate()?;
        if self.schema != SEMANTIC_SETUP_CHECKPOINT_SCHEMA
            || self.training_version != SEMANTIC_SETUP_TRAINING_VERSION
            || self.source_kind != RESOURCE_CYCLE_V5_SEMANTIC_SETUP_TEACHER_SOURCE_KIND
            || self.task != SEMANTIC_SETUP_TASK
            || self.model_version != SEMANTIC_SETUP_MODEL_VERSION
            || self.input_contract != SEMANTIC_SETUP_INPUT_CONTRACT
            || self.feature_schema != SEMANTIC_SETUP_FEATURE_SCHEMA
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
            return Err("Invalid semantic Setup V5 teacher checkpoint contract".into());
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
            return Err("Semantic Setup teacher checkpoint checksum mismatch".into());
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
pub struct SemanticSetupTeacherTrainingOutcome {
    pub checkpoint: SemanticSetupTeacherTrainingCheckpoint,
    pub metrics: PolicyTrainingMetrics,
}
impl SemanticSetupTeacherTrainingOutcome {
    pub fn save_new_directory(&self, directory: &Path) -> Result<(), String> {
        self.checkpoint.validate()?;
        if self.metrics != self.checkpoint.metrics {
            return Err("Inconsistent semantic Setup V5 teacher outcome".into());
        }
        validate_new_output_directory(directory)?;
        bounded_json(&self.checkpoint)?;
        bounded_json(&self.metrics)?;
        fs::create_dir(directory).map_err(|e| e.to_string())?;
        publish(&directory.join("checkpoint.json"), &self.checkpoint)?;
        publish(&directory.join("metrics.json"), &self.metrics)
    }
}
/// Fresh seeded schema-3 training or same-type resume only; no old weights input.
/// ```compile_fail
/// use tzolkin_ai::policy_dataset::ValidatedSemanticSetupTeacherDataset;
/// use tzolkin_ai::policy_training::PolicyBcConfig;
/// use tzolkin_ai::resource_cycle_teacher_training::{self as teacher, TeacherTrainingCheckpoint};
/// fn old_resume(d: &ValidatedSemanticSetupTeacherDataset, old: &TeacherTrainingCheckpoint) {
///     let _ = teacher::train_semantic_setup_dataset(d, &PolicyBcConfig::default(), Some(old));
/// }
/// ```
pub fn train_semantic_setup_dataset(
    dataset: &ValidatedSemanticSetupTeacherDataset,
    config: &PolicyBcConfig,
    resume: Option<&SemanticSetupTeacherTrainingCheckpoint>,
) -> Result<SemanticSetupTeacherTrainingOutcome, String> {
    train_semantic_setup_after_cache::<true>(dataset, config, resume, || Ok(()))
}
fn train_semantic_setup_after_cache<const CACHED: bool>(
    dataset: &ValidatedSemanticSetupTeacherDataset,
    config: &PolicyBcConfig,
    resume: Option<&SemanticSetupTeacherTrainingCheckpoint>,
    after_cache: impl FnOnce() -> Result<(), String>,
) -> Result<SemanticSetupTeacherTrainingOutcome, String> {
    config.validate()?;
    if let Some(p) = resume {
        p.validate()?;
    }
    let resume = resume.map(|p| TeacherResume {
        dataset: &p.dataset,
        config: &p.config,
        completed_epochs: p.completed_epochs,
        random_state: p.random_state,
        parameters: p.model.parameters(),
        optimizer: &p.optimizer,
        metrics: &p.metrics,
    });
    train_route_after_cache::<CACHED, _>(
        TeacherRoute::SemanticSetup(dataset),
        config,
        resume,
        after_cache,
        |run| {
            let mut checkpoint = SemanticSetupTeacherTrainingCheckpoint {
                schema: SEMANTIC_SETUP_CHECKPOINT_SCHEMA.into(),
                training_version: SEMANTIC_SETUP_TRAINING_VERSION.into(),
                source_kind: RESOURCE_CYCLE_V5_SEMANTIC_SETUP_TEACHER_SOURCE_KIND.into(),
                task: SEMANTIC_SETUP_TASK.into(),
                model_version: SEMANTIC_SETUP_MODEL_VERSION.into(),
                input_contract: SEMANTIC_SETUP_INPUT_CONTRACT.into(),
                feature_schema: SEMANTIC_SETUP_FEATURE_SCHEMA,
                feature_count: FEATURE_COUNT,
                backend: "scalar".into(),
                value_validity: ValueValidity::UnavailablePolicyOnly,
                teacher: ResourceCycleConfig::v5(),
                dataset: run.identity,
                config: config.clone(),
                completed_epochs: config.epochs,
                random_state: run.random_state,
                model: SemanticSetupTeacherModel {
                    semantic_setup_parameters: run.model.parameters,
                },
                optimizer: run.optimizer,
                metrics: run.metrics.clone(),
                checksum: String::new(),
            };
            checkpoint.checksum = checkpoint.expected_checksum()?;
            checkpoint.validate()?;
            Ok(SemanticSetupTeacherTrainingOutcome {
                checkpoint,
                metrics: run.metrics,
            })
        },
    )
}
pub fn validate_semantic_setup_checkpoint_dataset(
    dataset: &ValidatedSemanticSetupTeacherDataset,
    checkpoint: &SemanticSetupTeacherTrainingCheckpoint,
) -> Result<(), String> {
    let audited = train_semantic_setup_dataset(dataset, &checkpoint.config, Some(checkpoint))?;
    if audited.checkpoint != *checkpoint {
        return Err("Semantic Setup teacher checkpoint differs from audited no-op resume".into());
    }
    Ok(())
}
pub fn evaluate_semantic_setup_dataset(
    dataset: &ValidatedSemanticSetupTeacherDataset,
    checkpoint: &SemanticSetupTeacherTrainingCheckpoint,
    split: DatasetSplit,
) -> Result<PolicyLossReport, String> {
    checkpoint.validate()?;
    let route = TeacherRoute::SemanticSetup(dataset);
    let identity = audit_route(route)?;
    if identity != checkpoint.dataset {
        return Err("Semantic Setup teacher evaluation dataset mismatch".into());
    }
    let model = PublicPolicyModel {
        parameters: checkpoint.model.parameters().to_vec(),
    };
    let result = evaluate_samples_using(
        route.iter(Some(split)),
        &identity.dataset,
        &model,
        split,
        &mut Scratch::default(),
        route.profile(),
    );
    if audit_route(route)? != identity {
        return Err("Semantic Setup teacher evaluation source changed".into());
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dataset;
    use crate::features::FeatureEncoder;
    use crate::policy_dataset::{
        self, export_resource_cycle_v5_teacher_files, load_resource_cycle_v5_teacher_dataset,
    };
    use crate::public_model::PublicPolicyArtifact;
    use crate::replay::{self, GameReplay, ReplaySource};
    use std::path::PathBuf;
    use tzolkin_core::{GameOptions, Phase};
    mod test_temp_root {
        include!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/support/temp_root.rs"
        ));
    }
    struct Temp(PathBuf);
    impl Temp {
        fn new() -> Self {
            Self(test_temp_root::create("tzolkin-teacher-cache").unwrap())
        }
        fn source(&self, name: &str, record: &GameReplay) -> PathBuf {
            let path = self.0.join(name);
            fs::write(&path, serde_json::to_vec_pretty(record).unwrap()).unwrap();
            path
        }
        fn dataset(&self) -> PathBuf {
            self.0.join("dataset")
        }
    }
    impl Drop for Temp {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }
    fn hash(bytes: &[u8]) -> String {
        digest(bytes)
    }
    fn resource_cycle_v5_record(players: usize, seed: u32) -> GameReplay {
        use crate::arena::resource_cycle;
        use crate::replay::SeatPolicy;
        let configuration = ResourceCycleConfig::v5();
        replay::play_game_using_fast(
            players,
            seed,
            GameOptions::default(),
            true,
            ReplaySource::PolicySelfPlay {
                policies: vec![
                    SeatPolicy::ResourceCycle {
                        configuration: configuration.clone()
                    };
                    players
                ],
            },
            |o| resource_cycle::choose_move(o, &configuration),
        )
        .unwrap()
        .2
        .unwrap()
    }

    #[test]
    #[ignore = "Runs policy training and full source audits; execute locally when the teacher trainer changes."]
    fn resource_cycle_v5_teacher_training_resume_and_source_contract() {
        use crate::arena::resource_cycle::ResourceCycleConfig;
        use crate::policy_training::{PolicyBcConfig, PolicyTrainingCheckpoint};
        use crate::resource_cycle_teacher_training::{self as teacher, TeacherTrainingCheckpoint};
        let temp = Temp::new();
        let records = [3, 4].map(|players| resource_cycle_v5_record(players, 42));
        let sources = records
            .iter()
            .enumerate()
            .map(|(i, record)| temp.source(&format!("teacher-{i}.json"), record))
            .collect::<Vec<_>>();
        let manifest = export_resource_cycle_v5_teacher_files(&sources, &temp.dataset()).unwrap();
        let config = PolicyBcConfig {
            epochs: 2,
            batch_size: 256,
            learning_rate: 0.001,
            seed: 7,
        };
        assert_eq!(manifest.games[0].split, DatasetSplit::Train);
        assert_eq!(
            dataset::split_for_family(&dataset::seed_family_id(10005)).unwrap(),
            DatasetSplit::Validation
        );
        let validation = resource_cycle_v5_record(3, 10005);
        let mut training_sources = sources.clone();
        training_sources.push(temp.source("teacher-validation.json", &validation));
        let training_directory = temp.0.join("teacher-training-dataset");
        let training_manifest =
            export_resource_cycle_v5_teacher_files(&training_sources, &training_directory).unwrap();
        let training = load_resource_cycle_v5_teacher_dataset(&training_directory).unwrap();
        let semantic_directory = temp.0.join("semantic-teacher-training-dataset");
        policy_dataset::export_resource_cycle_v5_semantic_setup_teacher_files(
            &training_sources,
            &semantic_directory,
        )
        .unwrap();
        let semantic = policy_dataset::load_resource_cycle_v5_semantic_setup_teacher_dataset(
            &semantic_directory,
        )
        .unwrap();
        let semantic_continuous = train_semantic_setup_dataset(&semantic, &config, None).unwrap();
        let semantic_streaming =
            train_semantic_setup_after_cache::<false>(&semantic, &config, None, || Ok(())).unwrap();
        assert_eq!(
            serde_json::to_vec(&semantic_continuous).unwrap(),
            serde_json::to_vec(&semantic_streaming).unwrap()
        );
        let semantic_first = train_semantic_setup_dataset(
            &semantic,
            &PolicyBcConfig {
                epochs: 1,
                ..config.clone()
            },
            None,
        )
        .unwrap();
        let semantic_resumed =
            train_semantic_setup_dataset(&semantic, &config, Some(&semantic_first.checkpoint))
                .unwrap();
        assert_eq!(
            serde_json::to_vec(&semantic_continuous).unwrap(),
            serde_json::to_vec(&semantic_resumed).unwrap()
        );
        assert_eq!(
            train_semantic_setup_dataset(&semantic, &config, Some(&semantic_resumed.checkpoint))
                .unwrap(),
            semantic_resumed
        );
        validate_semantic_setup_checkpoint_dataset(&semantic, &semantic_resumed.checkpoint)
            .unwrap();
        let semantic_streaming_resume = train_semantic_setup_after_cache::<false>(
            &semantic,
            &config,
            Some(&semantic_first.checkpoint),
            || Ok(()),
        )
        .unwrap();
        assert_eq!(
            serde_json::to_vec(&semantic_resumed).unwrap(),
            serde_json::to_vec(&semantic_streaming_resume).unwrap()
        );
        assert_eq!(
            evaluate_semantic_setup_dataset(
                &semantic,
                &semantic_resumed.checkpoint,
                DatasetSplit::Validation
            )
            .unwrap(),
            semantic_resumed.metrics.final_validation
        );
        assert!(
            evaluate_semantic_setup_dataset(
                &semantic,
                &semantic_resumed.checkpoint,
                DatasetSplit::Test
            )
            .is_err()
        );
        let semantic_cache = TeacherSamples::load_route(
            TeacherRoute::SemanticSetup(&semantic),
            &semantic_continuous.checkpoint.dataset,
        )
        .unwrap();
        assert!(semantic_cache.allocation_bytes <= MAX_ENCODED_CACHE_BYTES);
        assert_eq!(
            semantic
                .iter_split(DatasetSplit::Train)
                .map(|s| s.unwrap().sample_id().to_owned())
                .collect::<Vec<_>>(),
            semantic_cache
                .train
                .iter()
                .map(|s| s.sample_id().to_owned())
                .collect::<Vec<_>>()
        );
        let mut too_large_semantic = semantic_continuous.checkpoint.dataset.clone();
        too_large_semantic.dataset.candidates[0] =
            MAX_ENCODED_CACHE_BYTES / std::mem::size_of::<crate::features::EncodedCandidate>() + 1;
        assert!(
            TeacherSamples::load_route(TeacherRoute::SemanticSetup(&semantic), &too_large_semantic)
                .err()
                .unwrap()
                .contains("256 MiB")
        );
        assert_eq!(
            semantic_continuous.checkpoint.feature_schema,
            SEMANTIC_SETUP_FEATURE_SCHEMA
        );
        assert_eq!(
            semantic_continuous
                .checkpoint
                .dataset
                .dataset
                .input_contract,
            SEMANTIC_SETUP_INPUT_CONTRACT
        );
        assert_eq!(semantic_continuous.metrics.final_train.value_loss, ());
        assert!(!semantic_continuous.metrics.strength_measured);
        let semantic_output = temp.0.join("semantic-teacher-checkpoint");
        semantic_resumed
            .save_new_directory(&semantic_output)
            .unwrap();
        assert_eq!(
            SemanticSetupTeacherTrainingCheckpoint::load(&semantic_output.join("checkpoint.json"))
                .unwrap(),
            semantic_resumed.checkpoint
        );
        assert!(!semantic_output.join("model.json").exists());
        assert!(
            semantic_resumed
                .save_new_directory(&semantic_output)
                .is_err()
        );
        let semantic_bytes = serde_json::to_vec(&semantic_resumed.checkpoint).unwrap();
        assert!(serde_json::from_slice::<TeacherTrainingCheckpoint>(&semantic_bytes).is_err());
        assert!(serde_json::from_slice::<PolicyTrainingCheckpoint>(&semantic_bytes).is_err());
        assert!(serde_json::from_slice::<PublicPolicyArtifact>(&semantic_bytes).is_err());
        for field in [
            "featureSchema",
            "sourceKind",
            "inputContract",
            "modelVersion",
            "trainingVersion",
            "teacher",
            "family",
            "policyId",
        ] {
            let mut bad = semantic_resumed.checkpoint.clone();
            match field {
                "featureSchema" => bad.feature_schema = PUBLIC_FEATURE_SCHEMA,
                "sourceKind" => bad.source_kind = RESOURCE_CYCLE_V5_TEACHER_SOURCE_KIND.into(),
                "inputContract" => bad.input_contract = INPUT_CONTRACT.into(),
                "modelVersion" => bad.model_version = MODEL_VERSION.into(),
                "trainingVersion" => bad.training_version = TRAINING_VERSION.into(),
                "teacher" => bad.teacher = ResourceCycleConfig::default(),
                "family" => {
                    bad.dataset.families.pop().unwrap();
                }
                "policyId" => bad.dataset.dataset.strata[0].policy_id = "0".repeat(64),
                _ => unreachable!(),
            }
            bad.checksum = bad.expected_checksum().unwrap();
            assert!(bad.validate().is_err());
            assert!(train_semantic_setup_dataset(&semantic, &config, Some(&bad)).is_err());
        }
        // JSON retains full u64 RNG/seed values; no f64 roundtrip is involved.
        let mut large_integer = semantic_resumed.checkpoint.clone();
        large_integer.config.seed = u64::MAX;
        large_integer.random_state = u64::MAX - 1;
        large_integer.checksum = large_integer.expected_checksum().unwrap();
        large_integer.validate().unwrap();
        assert_eq!(
            serde_json::from_slice::<SemanticSetupTeacherTrainingCheckpoint>(
                &serde_json::to_vec(&large_integer).unwrap()
            )
            .unwrap(),
            large_integer
        );
        assert!(train_semantic_setup_dataset(&semantic, &config, Some(&large_integer)).is_err());
        let continuous = teacher::train_dataset(&training, &config, None).unwrap();
        assert!(
            serde_json::from_slice::<SemanticSetupTeacherTrainingCheckpoint>(
                &serde_json::to_vec(&continuous.checkpoint).unwrap()
            )
            .is_err()
        );
        assert!(
            serde_json::from_slice::<SemanticSetupTeacherModel>(
                &serde_json::to_vec(&continuous.checkpoint.model).unwrap()
            )
            .is_err()
        );
        assert!(
            serde_json::from_slice::<PublicPolicyModel>(
                &serde_json::to_vec(&semantic_continuous.checkpoint.model).unwrap()
            )
            .is_err()
        );
        // Re-signing every outer route field still cannot reinterpret the old
        // inner model's serialized parameters as a semantic model.
        let mut retag = serde_json::to_value(&continuous.checkpoint).unwrap();
        for field in [
            "schema",
            "trainingVersion",
            "sourceKind",
            "task",
            "modelVersion",
            "inputContract",
            "featureSchema",
            "dataset",
        ] {
            retag[field] =
                serde_json::to_value(&semantic_continuous.checkpoint).unwrap()[field].clone();
        }
        retag["checksum"] = serde_json::Value::String(String::new());
        retag["checksum"] = serde_json::Value::String(hash(&serde_json::to_vec(&retag).unwrap()));
        assert!(serde_json::from_value::<SemanticSetupTeacherTrainingCheckpoint>(retag).is_err());
        let streaming = train_dataset_using::<false>(&training, &config, None).unwrap();
        // Includes every parameter/moment, RNG, metric, identity and checksum byte.
        assert_eq!(
            serde_json::to_vec(&continuous).unwrap(),
            serde_json::to_vec(&streaming).unwrap()
        );
        let cached = TeacherSamples::load(&training, &continuous.checkpoint.dataset).unwrap();
        assert!(cached.allocation_bytes <= MAX_ENCODED_CACHE_BYTES);
        let ordered_ids = training
            .iter_split(DatasetSplit::Train)
            .map(|sample| sample.unwrap().sample_id().to_owned())
            .collect::<Vec<_>>();
        assert_eq!(
            ordered_ids,
            cached
                .train
                .iter()
                .map(|s| s.sample_id().to_owned())
                .collect::<Vec<_>>()
        );
        let mut too_large = continuous.checkpoint.dataset.clone();
        too_large.dataset.candidates[0] =
            MAX_ENCODED_CACHE_BYTES / std::mem::size_of::<crate::features::EncodedCandidate>() + 1;
        assert!(
            TeacherSamples::load(&training, &too_large)
                .err()
                .unwrap()
                .contains("256 MiB")
        );

        let first = teacher::train_dataset(
            &training,
            &PolicyBcConfig {
                epochs: 1,
                ..config.clone()
            },
            None,
        )
        .unwrap();
        let resumed = teacher::train_dataset(&training, &config, Some(&first.checkpoint)).unwrap();
        assert_eq!(continuous, resumed);
        assert_eq!(
            serde_json::to_vec(&continuous).unwrap(),
            serde_json::to_vec(&resumed).unwrap()
        );
        assert_eq!(
            teacher::train_dataset(&training, &config, Some(&resumed.checkpoint)).unwrap(),
            resumed
        );
        teacher::validate_checkpoint_dataset(&training, &resumed.checkpoint).unwrap();
        let streaming_resume =
            train_dataset_using::<false>(&training, &config, Some(&first.checkpoint)).unwrap();
        assert_eq!(
            serde_json::to_vec(&resumed).unwrap(),
            serde_json::to_vec(&streaming_resume).unwrap()
        );

        assert_eq!(resumed.checkpoint.dataset.families.len(), 2);
        assert_eq!(resumed.checkpoint.dataset.dataset.samples[2], 0);
        assert!(!resumed.metrics.strength_measured);
        assert_eq!(resumed.metrics.final_train.value_samples, 0);
        assert_eq!(resumed.metrics.final_validation.value_loss, ());
        assert_eq!(
            teacher::evaluate_dataset(&training, &resumed.checkpoint, DatasetSplit::Validation)
                .unwrap(),
            resumed.metrics.final_validation
        );
        assert!(
            teacher::evaluate_dataset(&training, &resumed.checkpoint, DatasetSplit::Test).is_err()
        );
        assert!(crate::policy_training::train_dataset(&training, &config, None).is_err());
        let bytes = serde_json::to_vec(&resumed.checkpoint).unwrap();
        assert!(serde_json::from_slice::<PolicyTrainingCheckpoint>(&bytes).is_err());
        assert!(serde_json::from_slice::<PublicPolicyArtifact>(&bytes).is_err());
        let output = temp.0.join("teacher-checkpoint");
        resumed.save_new_directory(&output).unwrap();
        assert_eq!(
            TeacherTrainingCheckpoint::load(&output.join("checkpoint.json")).unwrap(),
            resumed.checkpoint
        );
        assert!(!output.join("model.json").exists());
        assert!(resumed.save_new_directory(&output).is_err());
        for field in ["teacher", "sourceKind", "family", "policyId"] {
            let mut bad = resumed.checkpoint.clone();
            match field {
                "teacher" => bad.teacher = ResourceCycleConfig::default(),
                "sourceKind" => bad.source_kind = "verifiedNativePolicy".into(),
                "family" => bad.dataset.families.pop().map(|_| ()).unwrap(),
                "policyId" => bad.dataset.dataset.strata[0].policy_id = "0".repeat(64),
                _ => unreachable!(),
            }
            bad.checksum.clear();
            bad.checksum = hash(&serde_json::to_vec(&bad).unwrap());
            assert!(bad.validate().is_err());
            assert!(teacher::train_dataset(&training, &config, Some(&bad)).is_err());
        }
        assert!(
            teacher::train_dataset(
                &training,
                &PolicyBcConfig {
                    seed: 8,
                    ..config.clone()
                },
                Some(&resumed.checkpoint)
            )
            .is_err()
        );
        assert!(
            teacher::train_dataset(
                &training,
                &PolicyBcConfig {
                    epochs: 1,
                    ..config
                },
                Some(&resumed.checkpoint)
            )
            .is_err()
        );
        // Audited teacher inference retains its source route; no native conversion.
        use crate::arena::numeric::{ExternalExclusions, NumericArenaConfig};
        use crate::arena::teacher::PreparedResourceCycleTeacher;
        use crate::kernel::Kernel;
        use tzolkin_inference::policy::BorrowedPolicy;
        use tzolkin_inference::trade_guard::TradeGuardConfig;
        let prepared = PreparedResourceCycleTeacher::new(
            resumed.checkpoint.clone(),
            load_resource_cycle_v5_teacher_dataset(&training_directory).unwrap(),
            Kernel::Scalar,
        )
        .unwrap();
        let handle = prepared.arena_handle(TradeGuardConfig::default()).unwrap();
        let identity = serde_json::to_value(handle.identity()).unwrap();
        assert_eq!(
            identity["sourceKind"],
            policy_dataset::RESOURCE_CYCLE_V5_TEACHER_SOURCE_KIND
        );
        assert_eq!(identity["checkpointChecksum"], resumed.checkpoint.checksum);
        assert_eq!(identity["dataset"]["families"].as_array().unwrap().len(), 2);
        assert_eq!(identity["valueValidity"], "unavailablePolicyOnly");
        for field in [
            "producerAuthenticated",
            "sourceQualified",
            "verifiedNativePolicy",
            "strengthMeasured",
        ] {
            assert_eq!(identity[field], false);
        }
        for phase in [Phase::Setup, Phase::Playing] {
            let mut state =
                tzolkin_core::create_game(vec!["A".into(), "B".into(), "C".into()], 17, false)
                    .unwrap();
            while state.phase != phase {
                let o = tzolkin_core::observation::observe(&state, state.current_player).unwrap();
                state =
                    tzolkin_core::apply_move(&state, o.legal_actions[0].r#move.clone()).unwrap();
            }
            let o = tzolkin_core::observation::observe(&state, state.current_player).unwrap();
            let encoder = FeatureEncoder::new_public(&o).unwrap();
            let rows = (0..o.legal_actions.len())
                .map(|i| encoder.encode_legal_tagged(i).unwrap())
                .collect::<Vec<_>>();
            let expected =
                BorrowedPolicy::from_validated_parameters(resumed.checkpoint.model.parameters())
                    .unwrap()
                    .predict(Kernel::Scalar.resolve().unwrap(), &rows)
                    .unwrap();
            let actual = prepared.distribution(&o).unwrap();
            assert_eq!(
                actual
                    .logits
                    .iter()
                    .map(|v| v.to_bits())
                    .collect::<Vec<_>>(),
                expected
                    .logits
                    .iter()
                    .map(|v| v.to_bits())
                    .collect::<Vec<_>>()
            );
            assert_eq!(
                actual
                    .probabilities
                    .iter()
                    .map(|v| v.to_bits())
                    .collect::<Vec<_>>(),
                expected
                    .probabilities
                    .iter()
                    .map(|v| v.to_bits())
                    .collect::<Vec<_>>()
            );
        }
        let arena_config = NumericArenaConfig {
            schema: crate::arena::numeric::SCHEMA.into(),
            players: 3,
            seeds: vec![10006],
            bootstrap_seed: 7,
            external_exclusions: ExternalExclusions {
                evidence_sha256: "e".repeat(64),
                excluded_family_ids: vec![],
            },
        };
        handle.validate_arena_config(&arena_config).unwrap();
        let mut overlap = arena_config.clone();
        overlap.seeds = vec![10005];
        assert!(handle.validate_arena_config(&overlap).is_err());
        let mut excluded = arena_config.clone();
        excluded
            .external_exclusions
            .excluded_family_ids
            .push(dataset::seed_family_id(10006));
        assert!(handle.validate_arena_config(&excluded).is_err());
        let manifest_path = training_directory.join("manifest.json");
        let original_manifest = fs::read(&manifest_path).unwrap();
        let changed_after_cache = original_manifest.clone();
        let error = train_dataset_after_cache::<true>(
            &training,
            &config,
            Some(&resumed.checkpoint),
            || {
                let mut changed = changed_after_cache;
                changed.push(b' ');
                fs::write(&manifest_path, changed).map_err(|e| e.to_string())
            },
        )
        .unwrap_err();
        assert!(error.contains("manifest bytes changed"));
        fs::write(&manifest_path, &original_manifest).unwrap();

        let mut changed_bytes = original_manifest.clone();
        changed_bytes.push(b' ');
        fs::write(&manifest_path, changed_bytes).unwrap();
        assert!(handle.validate_arena_config(&arena_config).is_err());
        fs::write(manifest_path, original_manifest).unwrap();
        let training_source = training_directory.join(&training_manifest.games[0].source_file);
        let original_training_source = fs::read(&training_source).unwrap();
        fs::write(&training_source, b"{}").unwrap();
        assert!(teacher::validate_checkpoint_dataset(&training, &resumed.checkpoint).is_err());
        assert!(handle.validate_arena_config(&arena_config).is_err());
        fs::write(training_source, original_training_source).unwrap();
        // All raw source classes are rechecked even when the numeric work fails
        // after cache creation; no cached outcome is returned for changed bytes.
        for path in [
            semantic_directory.join("manifest.json"),
            semantic_directory.join(&semantic.manifest().games[0].source_file),
            semantic_directory.join(&semantic.manifest().shards[0].file),
        ] {
            let original = fs::read(&path).unwrap();
            let error = train_semantic_setup_after_cache::<true>(
                &semantic,
                &config,
                Some(&semantic_resumed.checkpoint),
                || {
                    let mut changed = original.clone();
                    changed.push(b' ');
                    fs::write(&path, changed).map_err(|e| e.to_string())?;
                    Err("injected numeric failure".into())
                },
            )
            .unwrap_err();
            assert_ne!(error, "injected numeric failure");
            fs::write(&path, &original).unwrap();
        }
    }
}
