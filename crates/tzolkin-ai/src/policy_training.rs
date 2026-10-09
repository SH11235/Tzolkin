//! Scalar policy-only BC on fully source-validated V2 native datasets. No value head or labels.
use crate::dataset::DatasetSplit;
use crate::features::{EncodedCandidate, FEATURE_COUNT, MAX_LEGAL_ACTIONS, PUBLIC_FEATURE_SCHEMA};
use crate::model::{MAX_ARTIFACT_BYTES, Random, digest, dot, policy_softmax, read_json};
use crate::policy_dataset::{
    DATASET_SCHEMA, PolicyDatasetStratum, ValidatedPolicyDataset, ValidatedPolicySample,
};
use crate::public_model::{
    B1, BP, HIDDEN, INPUT_CONTRACT, MODEL_VERSION, PARAMETER_COUNT, PublicPolicyArtifact,
    PublicPolicyModel, ValueValidity, WP,
};
use serde::{Deserialize, Serialize};
use std::fs::{self, File};
use std::io::Write;
use std::path::{Component, Path, PathBuf};

pub const CHECKPOINT_SCHEMA: &str = "tzolkin-public-policy-checkpoint-v1";
pub const TRAINING_VERSION: &str = "public-policy-bc-scalar-adam-v1";
pub const MAX_CANDIDATE_EVALUATIONS: usize = 10_000_000;
pub const MAX_UPDATES: u64 = 1_000_000;
const SHUFFLE_BUFFER: usize = 8;
const GRADIENT_CLIP: f64 = 5.0;

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct PolicyBcConfig {
    /// Total target epochs; only this field may change at resume.
    pub epochs: usize,
    pub batch_size: usize,
    pub learning_rate: f32,
    pub seed: u64,
}
impl Default for PolicyBcConfig {
    fn default() -> Self {
        Self {
            epochs: 3,
            batch_size: 16,
            learning_rate: 0.001,
            seed: 7,
        }
    }
}
impl PolicyBcConfig {
    pub fn validate(&self) -> Result<(), String> {
        if !(1..=1000).contains(&self.epochs)
            || !(1..=256).contains(&self.batch_size)
            || !self.learning_rate.is_finite()
            || self.learning_rate <= 0.0
            || self.learning_rate > 1.0
        {
            return Err("Invalid/bounded public policy BC config".into());
        }
        Ok(())
    }
    fn compatible(&self, other: &Self) -> bool {
        self.batch_size == other.batch_size
            && self.learning_rate.to_bits() == other.learning_rate.to_bits()
            && self.seed == other.seed
    }
}
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct PolicyLossReport {
    pub samples: usize,
    pub candidates: usize,
    pub policy_loss: f64,
    pub correct_top1: usize,
    pub value_loss: (),
    pub value_samples: usize,
    pub value_validity: ValueValidity,
}
impl PolicyLossReport {
    fn validate(&self, samples: usize, candidates: usize) -> Result<(), String> {
        if samples == 0
            || self.samples != samples
            || self.candidates != candidates
            || self.correct_top1 > samples
            || !self.policy_loss.is_finite()
            || self.policy_loss < 0.0
            || self.value_samples != 0
            || self.value_validity != ValueValidity::UnavailablePolicyOnly
        {
            return Err("Invalid public policy loss report".into());
        }
        Ok(())
    }
}
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct PolicyDatasetIdentity {
    pub schema: String,
    pub fingerprint: String,
    pub feature_schema: u32,
    pub feature_count: usize,
    pub rules_version: u32,
    pub rules_baseline: String,
    pub catalog_hash: String,
    pub move_schema: u32,
    pub observation_schema: u32,
    pub source_kind: String,
    pub task: String,
    pub input_contract: String,
    /// Train, validation, test. Test counts are integrity metadata, not training targets.
    pub samples: [usize; 3],
    pub candidates: [usize; 3],
    pub strata: Vec<PolicyDatasetStratum>,
}
fn split_index(split: DatasetSplit) -> usize {
    match split {
        DatasetSplit::Train => 0,
        DatasetSplit::Validation => 1,
        DatasetSplit::Test => 2,
    }
}
fn digest_id(id: &str) -> bool {
    id.len() == 64
        && id
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}
impl PolicyDatasetIdentity {
    fn from_dataset(dataset: &ValidatedPolicyDataset) -> Self {
        let m = dataset.manifest();
        let mut samples = [0; 3];
        let mut candidates = [0; 3];
        for game in &m.games {
            let i = split_index(game.split);
            samples[i] += game.samples;
            candidates[i] += game.candidates;
        }
        Self {
            schema: m.schema.clone(),
            fingerprint: m.fingerprint.clone(),
            feature_schema: m.feature_schema,
            feature_count: m.feature_count,
            rules_version: m.rules_version,
            rules_baseline: m.rules_baseline.clone(),
            catalog_hash: m.catalog_hash.clone(),
            move_schema: m.move_schema,
            observation_schema: m.observation_schema,
            source_kind: m.source_kind.clone(),
            task: m.task.clone(),
            input_contract: INPUT_CONTRACT.into(),
            samples,
            candidates,
            strata: m.strata.clone(),
        }
    }
    fn validate(&self, require_training: bool) -> Result<(), String> {
        if self.schema != DATASET_SCHEMA
            || !digest_id(&self.fingerprint)
            || self.feature_schema != PUBLIC_FEATURE_SCHEMA
            || self.feature_count != FEATURE_COUNT
            || self.rules_version != crate::replay::RULES_VERSION
            || self.rules_baseline != crate::replay::RULES_BASELINE
            || self.catalog_hash != crate::replay::catalog_hash()
            || self.move_schema != tzolkin_core::observation::MOVE_SCHEMA
            || self.observation_schema != tzolkin_core::observation::OBSERVATION_SCHEMA
            || self.source_kind != "verifiedNativePolicy"
            || self.task != "policyOnlyBc"
            || self.input_contract != INPUT_CONTRACT
            || self.strata.is_empty()
            || self.strata.len() > crate::policy_dataset::MAX_FILES
        {
            return Err("Incompatible public BC dataset identity".into());
        }
        for i in 0..3 {
            if self.samples[i] > crate::policy_dataset::MAX_PARTITION_SAMPLES
                || self.candidates[i] > crate::policy_dataset::MAX_PARTITION_CANDIDATES
                || self.candidates[i] < self.samples[i]
                || (self.samples[i] == 0 && self.candidates[i] != 0)
            {
                return Err("Invalid BC dataset partition counts".into());
            }
        }
        if require_training && (self.samples[0] == 0 || self.samples[1] == 0) {
            return Err("Nonempty isolated train and validation families required".into());
        }
        let mut totals = (0usize, 0usize);
        let mut games = 0usize;
        let mut previous = None;
        for stratum in &self.strata {
            let key = (stratum.players, stratum.policy_id.as_str());
            if !(3..=4).contains(&stratum.players)
                || !digest_id(&stratum.policy_id)
                || stratum.games == 0
                || stratum.games > crate::policy_dataset::MAX_FILES
                || stratum.samples == 0
                || stratum.samples > 3 * crate::policy_dataset::MAX_PARTITION_SAMPLES
                || stratum.candidates < stratum.samples
                || stratum.candidates > 3 * crate::policy_dataset::MAX_PARTITION_CANDIDATES
                || previous.is_some_and(|p| p >= key)
            {
                return Err("Invalid BC content strata".into());
            }
            previous = Some(key);
            totals.0 = totals
                .0
                .checked_add(stratum.samples)
                .ok_or("BC stratum count overflow")?;
            totals.1 = totals
                .1
                .checked_add(stratum.candidates)
                .ok_or("BC stratum count overflow")?;
            games = games
                .checked_add(stratum.games)
                .ok_or("BC stratum game overflow")?;
        }
        if games > crate::policy_dataset::MAX_FILES
            || totals != (self.samples.iter().sum(), self.candidates.iter().sum())
        {
            return Err("BC identity strata/count mismatch".into());
        }
        Ok(())
    }
}
fn audit_dataset(dataset: &ValidatedPolicyDataset) -> Result<PolicyDatasetIdentity, String> {
    let identity = PolicyDatasetIdentity::from_dataset(dataset);
    identity.validate(false)?;
    let mut counts = (0usize, 0usize);
    for sample in dataset.iter() {
        let sample = sample?;
        counts.0 += 1;
        counts.1 += sample.candidates().len();
        if sample.value_target().is_some() {
            return Err("Policy BC cannot consume value labels".into());
        }
    }
    if counts
        != (
            identity.samples.iter().sum(),
            identity.candidates.iter().sum(),
        )
    {
        return Err("BC source counts changed during full integrity audit".into());
    }
    Ok(identity)
}
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct PolicyAdamState {
    pub step: u64,
    pub first_moment: Vec<f32>,
    pub second_moment: Vec<f32>,
}
impl PolicyAdamState {
    fn new() -> Self {
        Self {
            step: 0,
            first_moment: vec![0.0; PARAMETER_COUNT],
            second_moment: vec![0.0; PARAMETER_COUNT],
        }
    }
    pub fn validate(&self) -> Result<(), String> {
        if self.step > MAX_UPDATES
            || self.first_moment.len() != PARAMETER_COUNT
            || self.second_moment.len() != PARAMETER_COUNT
            || self.first_moment.iter().any(|v| !v.is_finite())
            || self
                .second_moment
                .iter()
                .any(|v| !v.is_finite() || *v < 0.0)
        {
            return Err("Invalid policy-only Adam state".into());
        }
        Ok(())
    }
    fn update(
        &mut self,
        model: &mut PublicPolicyModel,
        gradient: &mut [f32],
        batch: usize,
        rate: f32,
    ) -> Result<(), String> {
        if self.step >= MAX_UPDATES
            || gradient.len() != PARAMETER_COUNT
            || batch == 0
            || gradient.iter().any(|v| !v.is_finite())
        {
            return Err("Invalid policy Adam update".into());
        }
        for value in gradient.iter_mut() {
            *value /= batch as f32;
        }
        let norm = gradient
            .iter()
            .map(|v| f64::from(*v).powi(2))
            .sum::<f64>()
            .sqrt();
        if !norm.is_finite() {
            return Err("Nonfinite gradient norm".into());
        }
        let scale = if norm > GRADIENT_CLIP {
            (GRADIENT_CLIP / norm) as f32
        } else {
            1.0
        };
        self.step += 1;
        let first = 1.0 - 0.9_f32.powf(self.step as f32);
        let second = 1.0 - 0.999_f32.powf(self.step as f32);
        for (i, parameter) in model.parameters.iter_mut().enumerate() {
            let g = gradient[i] * scale;
            self.first_moment[i] = 0.9 * self.first_moment[i] + 0.1 * g;
            self.second_moment[i] = 0.999 * self.second_moment[i] + 0.001 * g * g;
            *parameter -= rate * (self.first_moment[i] / first)
                / ((self.second_moment[i] / second).sqrt() + 1.0e-8);
        }
        gradient.fill(0.0);
        model.validate()?;
        self.validate()
    }
}
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct PolicyTrainingMetrics {
    pub completed_epochs: usize,
    pub initial_train: PolicyLossReport,
    pub initial_validation: PolicyLossReport,
    pub final_train: PolicyLossReport,
    pub final_validation: PolicyLossReport,
    pub strength_measured: bool,
}
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct PolicyTrainingCheckpoint {
    pub schema: String,
    pub training_version: String,
    pub model_version: String,
    pub feature_schema: u32,
    pub task: String,
    pub backend: String,
    pub value_validity: ValueValidity,
    pub dataset: PolicyDatasetIdentity,
    pub config: PolicyBcConfig,
    pub completed_epochs: usize,
    pub random_state: u64,
    pub model: PublicPolicyArtifact,
    pub optimizer: PolicyAdamState,
    pub metrics: PolicyTrainingMetrics,
    pub checksum: String,
}
impl PolicyTrainingCheckpoint {
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
            || self.model_version != MODEL_VERSION
            || self.feature_schema != PUBLIC_FEATURE_SCHEMA
            || self.task != "policyOnlyBc"
            || self.backend != "scalar"
            || self.value_validity != ValueValidity::UnavailablePolicyOnly
            || self.completed_epochs == 0
            || self.completed_epochs != self.config.epochs
            || self.metrics.completed_epochs != self.completed_epochs
            || self.metrics.strength_measured
            || self.optimizer.step
                != self.dataset.samples[0].div_ceil(self.config.batch_size) as u64
                    * self.completed_epochs as u64
        {
            return Err("Invalid BC checkpoint schema/epoch/optimizer metadata".into());
        }
        self.metrics
            .initial_train
            .validate(self.dataset.samples[0], self.dataset.candidates[0])?;
        self.metrics
            .final_train
            .validate(self.dataset.samples[0], self.dataset.candidates[0])?;
        self.metrics
            .initial_validation
            .validate(self.dataset.samples[1], self.dataset.candidates[1])?;
        self.metrics
            .final_validation
            .validate(self.dataset.samples[1], self.dataset.candidates[1])?;
        if !digest_id(&self.checksum) || self.checksum != self.expected_checksum()? {
            return Err("BC checkpoint checksum mismatch".into());
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
/// Bounded local artifact loading for the dedicated Scalar evaluation command.
pub fn load_evaluation_model(path: &Path) -> Result<PublicPolicyArtifact, String> {
    let model: PublicPolicyArtifact = read_local_json(path)?;
    model.validate()?;
    Ok(model)
}
/// Early local-path check, not publication authority: create_dir still refuses races.
pub fn validate_new_output_directory(path: &Path) -> Result<(), String> {
    local_path(path)?;
    match fs::symlink_metadata(path) {
        Ok(_) => return Err("Policy training output must be a new directory".into()),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
        Err(e) => return Err(e.to_string()),
    }
    let parent = path
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    if !fs::metadata(parent).map_err(|e| e.to_string())?.is_dir() {
        return Err("Policy training output parent must be an existing local directory".into());
    }
    Ok(())
}
#[derive(Clone, Debug, Serialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct PolicyTrainingOutcome {
    pub model: PublicPolicyArtifact,
    pub checkpoint: PolicyTrainingCheckpoint,
    pub metrics: PolicyTrainingMetrics,
}
impl PolicyTrainingOutcome {
    pub fn save_new_directory(&self, directory: &Path) -> Result<(), String> {
        self.checkpoint.validate()?;
        if self.model != self.checkpoint.model || self.metrics != self.checkpoint.metrics {
            return Err("Inconsistent policy training output".into());
        }
        validate_new_output_directory(directory)?;
        // Bound all payloads before reserving a new output; metrics is the completion marker.
        bounded_json(&self.model)?;
        bounded_json(&self.checkpoint)?;
        bounded_json(&self.metrics)?;
        fs::create_dir(directory).map_err(|e| format!("Cannot create new training output: {e}"))?;
        publish(&directory.join("model.json"), &self.model)?;
        publish(&directory.join("checkpoint.json"), &self.checkpoint)?;
        publish(&directory.join("metrics.json"), &self.metrics)
    }
}
#[derive(Default)]
struct Scratch {
    hidden: Vec<f32>,
    logits: Vec<f32>,
    probabilities: Vec<f32>,
}
fn policy_loss(
    model: &PublicPolicyModel,
    rows: &[EncodedCandidate],
    chosen: usize,
    scratch: &mut Scratch,
    gradient: Option<&mut [f32]>,
) -> Result<(f64, bool), String> {
    if rows.is_empty() || rows.len() > MAX_LEGAL_ACTIONS || chosen >= rows.len() {
        return Err("Invalid BC complete-legal candidate batch".into());
    }
    let context = &rows[0].values_for_schema(PUBLIC_FEATURE_SCHEMA)?[..384];
    scratch.hidden.resize(rows.len() * HIDDEN, 0.0);
    scratch.logits.resize(rows.len(), 0.0);
    scratch.probabilities.resize(rows.len(), 0.0);
    for (i, row) in rows.iter().enumerate() {
        let values = row.values_for_schema(PUBLIC_FEATURE_SCHEMA)?;
        if &values[..384] != context || values.iter().any(|v| !v.is_finite() || v.abs() > 1024.0) {
            return Err("Invalid BC tagged/context/finite features".into());
        }
        let hidden = &mut scratch.hidden[i * HIDDEN..(i + 1) * HIDDEN];
        for (unit, value) in hidden.iter_mut().enumerate() {
            let affine = dot(
                &model.parameters[unit * FEATURE_COUNT..(unit + 1) * FEATURE_COUNT],
                values,
            ) + model.parameters[B1 + unit];
            if !affine.is_finite() {
                return Err("Nonfinite BC hidden affine".into());
            }
            *value = affine.tanh();
        }
        let logit = dot(&model.parameters[WP..BP], hidden) + model.parameters[BP];
        if !logit.is_finite() {
            return Err("Nonfinite BC policy affine".into());
        }
        scratch.logits[i] = logit;
    }
    policy_softmax(&scratch.logits, &mut scratch.probabilities)?;
    let maximum = scratch
        .logits
        .iter()
        .copied()
        .fold(f32::NEG_INFINITY, f32::max) as f64;
    let loss = maximum - f64::from(scratch.logits[chosen])
        + scratch
            .logits
            .iter()
            .map(|v| (f64::from(*v) - maximum).exp())
            .sum::<f64>()
            .ln();
    if !loss.is_finite() || loss < 0.0 {
        return Err("Nonfinite BC categorical loss".into());
    }
    let best =
        scratch.logits.iter().enumerate().fold(
            0,
            |best, (i, v)| if *v > scratch.logits[best] { i } else { best },
        );
    if let Some(gradient) = gradient {
        if gradient.len() != PARAMETER_COUNT {
            return Err("Invalid policy-only gradient shape".into());
        }
        for (candidate, row) in rows.iter().enumerate() {
            let hidden = &scratch.hidden[candidate * HIDDEN..(candidate + 1) * HIDDEN];
            let delta =
                scratch.probabilities[candidate] - if candidate == chosen { 1.0 } else { 0.0 };
            gradient[BP] += delta;
            for unit in 0..HIDDEN {
                gradient[WP + unit] += delta * hidden[unit];
                let activation =
                    delta * model.parameters[WP + unit] * (1.0 - hidden[unit] * hidden[unit]);
                gradient[B1 + unit] += activation;
                for (feature, value) in row
                    .values_for_schema(PUBLIC_FEATURE_SCHEMA)?
                    .iter()
                    .enumerate()
                {
                    gradient[unit * FEATURE_COUNT + feature] += activation * value;
                }
            }
        }
        if gradient.iter().any(|v| !v.is_finite()) {
            return Err("Nonfinite policy BC gradient".into());
        }
    }
    Ok((loss, best == chosen))
}
fn evaluate(
    dataset: &ValidatedPolicyDataset,
    identity: &PolicyDatasetIdentity,
    model: &PublicPolicyModel,
    split: DatasetSplit,
    scratch: &mut Scratch,
) -> Result<PolicyLossReport, String> {
    let mut report = PolicyLossReport {
        samples: 0,
        candidates: 0,
        policy_loss: 0.0,
        correct_top1: 0,
        value_loss: (),
        value_samples: 0,
        value_validity: ValueValidity::UnavailablePolicyOnly,
    };
    for sample in dataset.iter_split(split) {
        let sample = sample?;
        let (loss, correct) =
            policy_loss(model, sample.candidates(), sample.chosen(), scratch, None)?;
        report.samples += 1;
        report.candidates += sample.candidates().len();
        report.policy_loss += loss;
        report.correct_top1 += usize::from(correct);
    }
    let index = split_index(split);
    if report.samples == 0 {
        return Err("Requested evaluation partition is empty".into());
    }
    report.policy_loss /= report.samples as f64;
    report.validate(identity.samples[index], identity.candidates[index])?;
    Ok(report)
}
/// Evaluates exactly the named partition with Scalar; does not modify an artifact or train.
pub fn evaluate_dataset(
    dataset: &ValidatedPolicyDataset,
    model: &PublicPolicyArtifact,
    split: DatasetSplit,
) -> Result<PolicyLossReport, String> {
    model.validate()?;
    let identity = audit_dataset(dataset)?;
    let report = evaluate(
        dataset,
        &identity,
        &model.model,
        split,
        &mut Scratch::default(),
    )?;
    if audit_dataset(dataset)? != identity {
        return Err("BC evaluation dataset changed before completion".into());
    }
    Ok(report)
}
/// Epoch-boundary, deterministic Scalar training. No external unqualified/raw sample API.
pub fn train_dataset(
    dataset: &ValidatedPolicyDataset,
    config: &PolicyBcConfig,
    resume: Option<&PolicyTrainingCheckpoint>,
) -> Result<PolicyTrainingOutcome, String> {
    config.validate()?;
    let completed = resume.map_or(0, |p| p.completed_epochs);
    if let Some(previous) = resume {
        previous.validate()?;
        if !config.compatible(&previous.config) || config.epochs < completed {
            return Err("BC resume dataset/config/epoch mismatch".into());
        }
    }
    let identity = audit_dataset(dataset)?;
    identity.validate(true)?;
    if resume.is_some_and(|previous| previous.dataset != identity) {
        return Err("BC resume dataset/config/epoch mismatch".into());
    }
    let remaining = config.epochs - completed;
    let evaluations = identity.candidates[0]
        .checked_mul(remaining + 2)
        .and_then(|n| {
            identity.candidates[1]
                .checked_mul(2)
                .and_then(|v| n.checked_add(v))
        })
        .ok_or("BC evaluation budget overflow")?;
    let updates = identity.samples[0].div_ceil(config.batch_size) as u64 * config.epochs as u64;
    if evaluations > MAX_CANDIDATE_EVALUATIONS || updates > MAX_UPDATES {
        return Err("BC candidate/update budget exceeded".into());
    }
    let mut artifact = match resume {
        Some(previous) => previous.model.clone(),
        None => PublicPolicyArtifact::new(config.seed)?,
    };
    let mut optimizer = resume.map_or_else(PolicyAdamState::new, |p| p.optimizer.clone());
    let mut random = Random {
        state: resume.map_or(config.seed ^ 0xd1b54a32d192ed03, |p| p.random_state),
    };
    let mut scratch = Scratch::default();
    let current_train = evaluate(
        dataset,
        &identity,
        &artifact.model,
        DatasetSplit::Train,
        &mut scratch,
    )?;
    let current_validation = evaluate(
        dataset,
        &identity,
        &artifact.model,
        DatasetSplit::Validation,
        &mut scratch,
    )?;
    let mut metrics = if let Some(previous) = resume {
        if current_train != previous.metrics.final_train
            || current_validation != previous.metrics.final_validation
        {
            return Err("BC resume metric/model/source mismatch".into());
        }
        previous.metrics.clone()
    } else {
        PolicyTrainingMetrics {
            completed_epochs: 0,
            initial_train: current_train.clone(),
            initial_validation: current_validation.clone(),
            final_train: current_train,
            final_validation: current_validation,
            strength_measured: false,
        }
    };
    let mut gradient = vec![0.0; PARAMETER_COUNT];
    let mut buffer: Vec<ValidatedPolicySample> = Vec::with_capacity(SHUFFLE_BUFFER);
    for _ in completed..config.epochs {
        let mut seen = 0;
        let mut batch = 0;
        let mut consume = |sample: ValidatedPolicySample| -> Result<(), String> {
            policy_loss(
                &artifact.model,
                sample.candidates(),
                sample.chosen(),
                &mut scratch,
                Some(&mut gradient),
            )?;
            seen += 1;
            batch += 1;
            if batch == config.batch_size {
                optimizer.update(
                    &mut artifact.model,
                    &mut gradient,
                    batch,
                    config.learning_rate,
                )?;
                batch = 0;
            }
            Ok(())
        };
        for sample in dataset.iter_split(DatasetSplit::Train) {
            let sample = sample?;
            if buffer.len() < SHUFFLE_BUFFER {
                buffer.push(sample);
            } else {
                let i = random.index(buffer.len());
                consume(std::mem::replace(&mut buffer[i], sample))?;
            }
        }
        while !buffer.is_empty() {
            let i = random.index(buffer.len());
            consume(buffer.swap_remove(i))?;
        }
        if seen != identity.samples[0] {
            return Err("BC train count changed".into());
        }
        if batch > 0 {
            optimizer.update(
                &mut artifact.model,
                &mut gradient,
                batch,
                config.learning_rate,
            )?;
        }
    }
    metrics.completed_epochs = config.epochs;
    metrics.final_train = evaluate(
        dataset,
        &identity,
        &artifact.model,
        DatasetSplit::Train,
        &mut scratch,
    )?;
    metrics.final_validation = evaluate(
        dataset,
        &identity,
        &artifact.model,
        DatasetSplit::Validation,
        &mut scratch,
    )?;
    if audit_dataset(dataset)? != identity {
        return Err("BC dataset changed before completion".into());
    }
    artifact.reseal()?;
    let mut checkpoint = PolicyTrainingCheckpoint {
        schema: CHECKPOINT_SCHEMA.into(),
        training_version: TRAINING_VERSION.into(),
        model_version: MODEL_VERSION.into(),
        feature_schema: PUBLIC_FEATURE_SCHEMA,
        task: "policyOnlyBc".into(),
        backend: "scalar".into(),
        value_validity: ValueValidity::UnavailablePolicyOnly,
        dataset: identity,
        config: config.clone(),
        completed_epochs: config.epochs,
        random_state: random.state,
        model: artifact.clone(),
        optimizer,
        metrics: metrics.clone(),
        checksum: String::new(),
    };
    checkpoint.checksum = checkpoint.expected_checksum()?;
    checkpoint.validate()?;
    Ok(PolicyTrainingOutcome {
        model: artifact,
        checkpoint,
        metrics,
    })
}
fn local_path(path: &Path) -> Result<(), String> {
    let text = path.to_string_lossy();
    if text.is_empty()
        || text.contains('\0')
        || text.contains("://")
        || text.starts_with("\\\\")
        || text.starts_with("//")
        || (!path.is_absolute() && matches!(path.components().next(), Some(Component::Prefix(_))))
    {
        return Err("Expected local policy training path".into());
    }
    let absolute = if path.is_absolute() {
        path.to_path_buf()
    } else {
        std::env::current_dir()
            .map_err(|e| e.to_string())?
            .join(path)
    };
    if absolute.to_string_lossy().starts_with("\\\\")
        || absolute.to_string_lossy().starts_with("//")
    {
        return Err("Network output hierarchy rejected".into());
    }
    let mut current = PathBuf::new();
    for component in absolute.components() {
        current.push(component.as_os_str());
        if matches!(component, Component::Prefix(_)) {
            continue;
        }
        match fs::symlink_metadata(&current) {
            Ok(m) if m.file_type().is_symlink() => {
                return Err("Policy training symlink/junction hierarchy rejected".into());
            }
            Ok(_) => {}
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
            Err(e) => return Err(e.to_string()),
        }
    }
    Ok(())
}
fn bounded_json(value: &impl Serialize) -> Result<Vec<u8>, String> {
    let bytes = serde_json::to_vec(value).map_err(|e| e.to_string())?;
    if bytes.len() > MAX_ARTIFACT_BYTES {
        return Err("Policy checkpoint/output exceeds 8 MiB".into());
    }
    Ok(bytes)
}
fn read_local_json<T: serde::de::DeserializeOwned>(path: &Path) -> Result<T, String> {
    local_path(path)?;
    if !fs::symlink_metadata(path)
        .map_err(|e| e.to_string())?
        .is_file()
    {
        return Err("Policy training input must be a local regular file".into());
    }
    read_json(path)
}
fn publish(path: &Path, value: &impl Serialize) -> Result<(), String> {
    local_path(path)?;
    let bytes = bounded_json(value)?;
    let parent = path
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    let name = path
        .file_name()
        .ok_or("Policy output requires filename")?
        .to_string_lossy();
    let temp = parent.join(format!(".{name}.{}.tmp", std::process::id()));
    let mut created = false;
    let result = (|| {
        let mut file = File::options()
            .create_new(true)
            .write(true)
            .open(&temp)
            .map_err(|e| e.to_string())?;
        created = true;
        file.write_all(&bytes)
            .and_then(|_| file.sync_all())
            .map_err(|e| e.to_string())?;
        drop(file);
        fs::hard_link(&temp, path).map_err(|e| e.to_string())
    })();
    if created {
        let _ = fs::remove_file(&temp);
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::features::FeatureEncoder;
    use crate::public_model::LoadedPublicPolicy;
    use std::sync::OnceLock;
    use tzolkin_core::GameOptions;
    use tzolkin_core::observation::observe;

    fn rows() -> &'static [EncodedCandidate] {
        static ROWS: OnceLock<Vec<EncodedCandidate>> = OnceLock::new();
        ROWS.get_or_init(|| {
            let state = crate::replay::benchmark_states(3, 42, GameOptions::default())
                .unwrap()
                .into_iter()
                .find(|s| observe(s, s.current_player).unwrap().legal_actions.len() >= 7)
                .unwrap();
            let observation = observe(&state, state.current_player).unwrap();
            let encoder = FeatureEncoder::new_public(&observation).unwrap();
            (0..observation.legal_actions.len())
                .map(|i| encoder.encode_legal_tagged(i).unwrap())
                .collect()
        })
    }

    #[test]
    fn ragged_categorical_loss_forward_and_singleton_match_policy_only_inference() {
        let artifact = PublicPolicyArtifact::new(11235).unwrap();
        let loaded = LoadedPublicPolicy::new(&artifact).unwrap();
        // These subsets exercise the private mathematical primitive's ragged dimensions.
        // Production training accepts only the source-verified complete legal batch.
        for size in [1, 2, 3, 7] {
            let rows = &rows()[..size];
            let prediction = loaded.predict(rows).unwrap();
            let mut scratch = Scratch::default();
            let mut gradient = vec![0.0; PARAMETER_COUNT];
            let (loss, _) = policy_loss(
                &artifact.model,
                rows,
                size - 1,
                &mut scratch,
                Some(&mut gradient),
            )
            .unwrap();
            assert_eq!(scratch.logits, prediction.logits);
            assert_eq!(scratch.probabilities, prediction.probabilities);
            let maximum = prediction
                .logits
                .iter()
                .copied()
                .fold(f32::NEG_INFINITY, f32::max) as f64;
            let expected = (prediction
                .logits
                .iter()
                .map(|v| (f64::from(*v) - maximum).exp())
                .sum::<f64>())
            .ln()
                + maximum
                - f64::from(prediction.logits[size - 1]);
            assert!((loss - expected).abs() < 1.0e-12);
            if size == 1 {
                assert_eq!(loss, 0.0);
                assert!(gradient.iter().all(|v| *v == 0.0));
            }
            assert!(gradient[BP].abs() < 2.0e-7); // shared policy bias cancels under softmax.
        }
    }

    #[test]
    fn full_legal_cross_entropy_backprop_matches_finite_differences_in_all_parameter_blocks() {
        let artifact = PublicPolicyArtifact::new(7).unwrap();
        let mut scratch = Scratch::default();
        let mut gradient = vec![0.0; PARAMETER_COUNT];
        policy_loss(
            &artifact.model,
            rows(),
            1,
            &mut scratch,
            Some(&mut gradient),
        )
        .unwrap();
        for range in [0..B1, B1..WP, WP..BP, BP..PARAMETER_COUNT] {
            let index = range
                .max_by(|a, b| gradient[*a].abs().total_cmp(&gradient[*b].abs()))
                .unwrap();
            let mut plus = artifact.model.clone();
            let mut minus = artifact.model.clone();
            let epsilon = 0.01_f32;
            plus.parameters[index] += epsilon;
            minus.parameters[index] -= epsilon;
            let high = policy_loss(&plus, rows(), 1, &mut scratch, None).unwrap().0;
            let low = policy_loss(&minus, rows(), 1, &mut scratch, None)
                .unwrap()
                .0;
            let numerical =
                (high - low) / f64::from(plus.parameters[index] - minus.parameters[index]);
            // f32 affine/tanh forward arithmetic, compared to a f64 loss difference.
            let tolerance = 5.0e-5 + 0.01 * f64::from(gradient[index].abs());
            assert!(
                (numerical - f64::from(gradient[index])).abs() < tolerance,
                "index={index}, analytical={}, numerical={numerical}",
                gradient[index]
            );
        }
    }

    #[test]
    fn categorical_logits_share_a_bias_shift_but_not_value_parameters() {
        let mut artifact = PublicPolicyArtifact::new(17).unwrap();
        let original =
            policy_loss(&artifact.model, rows(), 0, &mut Scratch::default(), None).unwrap();
        artifact.model.parameters[BP] += 10.0;
        let shifted =
            policy_loss(&artifact.model, rows(), 0, &mut Scratch::default(), None).unwrap();
        assert!((original.0 - shifted.0).abs() < 2.0e-6);
        assert_eq!(original.1, shifted.1);
        assert_eq!(PARAMETER_COUNT, 16449);
        assert_eq!(BP + 1, PARAMETER_COUNT);
    }

    #[test]
    fn adam_clips_batch_mean_and_refuses_invalid_or_nonfinite_updates() {
        let mut artifact = PublicPolicyArtifact::new(7).unwrap();
        let mut optimizer = PolicyAdamState::new();
        let mut gradient = vec![0.0; PARAMETER_COUNT];
        gradient[0] = 120.0;
        gradient[1] = 160.0;
        optimizer
            .update(&mut artifact.model, &mut gradient, 2, 0.001)
            .unwrap();
        assert_eq!(optimizer.step, 1);
        assert!((optimizer.first_moment[0] - 0.3).abs() < 1.0e-7);
        assert!((optimizer.first_moment[1] - 0.4).abs() < 1.0e-7);
        assert!((optimizer.second_moment[0] - 0.009).abs() < 1.0e-8);
        assert!(gradient.iter().all(|v| *v == 0.0));
        gradient[0] = f32::NAN;
        assert!(
            optimizer
                .update(&mut artifact.model, &mut gradient, 1, 0.001)
                .is_err()
        );
        assert_eq!(optimizer.step, 1);
        gradient[0] = 0.0;
        assert!(
            optimizer
                .update(&mut artifact.model, &mut gradient, 0, 0.001)
                .is_err()
        );
        optimizer.step = MAX_UPDATES;
        assert!(
            optimizer
                .update(&mut artifact.model, &mut gradient, 1, 0.001)
                .is_err()
        );
        optimizer.second_moment[1] = -1.0;
        assert!(optimizer.validate().is_err());
    }

    #[test]
    fn policy_loss_rejects_legacy_tags_empty_wrong_choice_and_finite_overflow() {
        let mut artifact = PublicPolicyArtifact::new(7).unwrap();
        let mut scratch = Scratch::default();
        assert!(policy_loss(&artifact.model, &[], 0, &mut scratch, None).is_err());
        assert!(policy_loss(&artifact.model, rows(), rows().len(), &mut scratch, None).is_err());
        let state =
            tzolkin_core::create_game(vec!["A".into(), "B".into(), "C".into()], 42, false).unwrap();
        let observation = observe(&state, state.current_player).unwrap();
        let row = FeatureEncoder::new(&observation)
            .unwrap()
            .encode_legal_tagged(0)
            .unwrap();
        assert!(policy_loss(&artifact.model, &[row], 0, &mut scratch, None).is_err());
        artifact.model.parameters[..B1].fill(f32::MAX);
        assert!(artifact.model.validate().is_ok());
        assert!(policy_loss(&artifact.model, rows(), 0, &mut scratch, None).is_err());
    }

    #[test]
    fn publisher_never_overwrites_a_destination_or_foreign_temporary_file() {
        let dir = std::env::temp_dir().join(format!("tzolkin-bc-publish-{}", std::process::id()));
        fs::create_dir(&dir).unwrap();
        let path = dir.join("checkpoint.json");
        fs::write(&path, b"original").unwrap();
        assert!(publish(&path, &7).is_err());
        assert_eq!(fs::read(&path).unwrap(), b"original");
        let temporary = dir.join(format!(".checkpoint.json.{}.tmp", std::process::id()));
        assert!(!temporary.exists());
        fs::write(&temporary, b"foreign").unwrap();
        assert!(publish(&path, &8).is_err());
        assert_eq!(fs::read(&temporary).unwrap(), b"foreign");
        assert_eq!(fs::read(&path).unwrap(), b"original");
        fs::remove_dir_all(&dir).unwrap();
    }
}
