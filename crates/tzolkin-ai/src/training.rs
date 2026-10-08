//! Bounded CPU Adam training; datasets are streamed, never collected wholesale.
use std::borrow::Cow;
use std::collections::{HashMap, HashSet};
use std::io::Write;
use std::path::Path;

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::dataset::{DatasetSplit, TrainingSample, ValidatedDataset, validate_sample};
use crate::features::{FEATURE_COUNT, FEATURE_SCHEMA};
use crate::model::{
    B1, BP, BV, HIDDEN, MAX_CANDIDATES, ModelArtifact, PARAMETER_COUNT, Random, TinyModel,
    VALUE_SIDES, W1, WP, WV, digest, policy_softmax, read_json, value_softmax, write_new_json,
};

pub const CHECKPOINT_SCHEMA: u32 = 1;
pub const MAX_SAMPLES: usize = 100_000;
pub const MAX_TOTAL_CANDIDATES: usize = 1_000_000;
pub const MAX_CANDIDATE_EVALUATIONS: usize = 10_000_000;
const SHUFFLE_BUFFER: usize = 8;
const GRADIENT_CLIP: f64 = 5.0;
const MAX_UPDATES: u64 = 1_000_000;

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(default, rename_all = "camelCase", deny_unknown_fields)]
pub struct TrainingConfig {
    /// Total target epochs. Resume may extend it without changing other fields.
    pub epochs: usize,
    pub batch_size: usize,
    pub learning_rate: f32,
    pub seed: u64,
    pub value_weight: f32,
}
impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            epochs: 3,
            batch_size: 16,
            learning_rate: 0.001,
            seed: 7,
            value_weight: 1.0,
        }
    }
}
impl TrainingConfig {
    pub fn validate(&self) -> Result<(), String> {
        if !(1..=1000).contains(&self.epochs)
            || !(1..=256).contains(&self.batch_size)
            || !self.learning_rate.is_finite()
            || !(0.0..=1.0).contains(&self.learning_rate)
            || self.learning_rate == 0.0
            || !self.value_weight.is_finite()
            || !(0.0..=10.0).contains(&self.value_weight)
        {
            return Err("Invalid/big training configuration".into());
        }
        Ok(())
    }
    fn compatible(&self, previous: &Self) -> bool {
        self.batch_size == previous.batch_size
            && self.learning_rate == previous.learning_rate
            && self.seed == previous.seed
            && self.value_weight == previous.value_weight
    }
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct LossReport {
    pub samples: usize,
    pub policy_loss: f64,
    pub value_loss: f64,
    pub total_loss: f64,
}
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct TrainingMetrics {
    pub completed_epochs: usize,
    pub train_samples: usize,
    pub validation_samples: usize,
    pub initial_train: LossReport,
    pub initial_validation: LossReport,
    pub final_train: LossReport,
    pub final_validation: LossReport,
    pub strength_measured: bool,
}
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct AdamState {
    pub step: u64,
    pub first_moment: Vec<f32>,
    pub second_moment: Vec<f32>,
}
impl AdamState {
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
            || self.first_moment.iter().any(|value| !value.is_finite())
            || self
                .second_moment
                .iter()
                .any(|value| !value.is_finite() || *value < 0.0)
        {
            return Err("Invalid Adam step/moment shape or values".into());
        }
        Ok(())
    }
    fn update(
        &mut self,
        model: &mut TinyModel,
        gradient: &mut [f32],
        batch: usize,
        rate: f32,
    ) -> Result<(), String> {
        if self.step >= MAX_UPDATES
            || gradient.len() != PARAMETER_COUNT
            || batch == 0
            || gradient.iter().any(|value| !value.is_finite())
        {
            return Err("Invalid/big optimizer update".into());
        }
        for value in gradient.iter_mut() {
            *value /= batch as f32;
        }
        let norm = gradient
            .iter()
            .map(|value| (*value as f64).powi(2))
            .sum::<f64>()
            .sqrt();
        let scale = if norm > GRADIENT_CLIP {
            (GRADIENT_CLIP / norm) as f32
        } else {
            1.0
        };
        self.step += 1;
        let first_correction = 1.0 - 0.9_f32.powf(self.step as f32);
        let second_correction = 1.0 - 0.999_f32.powf(self.step as f32);
        for (index, parameter) in model.parameters.iter_mut().enumerate() {
            let g = gradient[index] * scale;
            self.first_moment[index] = 0.9 * self.first_moment[index] + 0.1 * g;
            self.second_moment[index] = 0.999 * self.second_moment[index] + 0.001 * g * g;
            let first = self.first_moment[index] / first_correction;
            let second = self.second_moment[index] / second_correction;
            *parameter -= rate * first / (second.sqrt() + 1.0e-8);
        }
        gradient.fill(0.0);
        model.validate()?;
        self.validate()
    }
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct TrainingCheckpoint {
    pub schema: u32,
    pub dataset_fingerprint: String,
    pub config: TrainingConfig,
    pub completed_epochs: usize,
    pub random_state: u64,
    pub model: ModelArtifact,
    pub optimizer: AdamState,
    pub metrics: TrainingMetrics,
    pub checksum: String,
}
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct TrainingOutcome {
    pub model: ModelArtifact,
    pub checkpoint: TrainingCheckpoint,
    pub metrics: TrainingMetrics,
}

impl TrainingCheckpoint {
    fn expected_checksum(&self) -> Result<String, String> {
        let mut payload = self.clone();
        payload.checksum.clear();
        Ok(digest(
            &serde_json::to_vec(&payload).map_err(|error| error.to_string())?,
        ))
    }
    pub fn validate(&self) -> Result<(), String> {
        self.config.validate()?;
        self.model.validate()?;
        self.optimizer.validate()?;
        if self.schema != CHECKPOINT_SCHEMA
            || self.dataset_fingerprint.len() != 64
            || !self
                .dataset_fingerprint
                .bytes()
                .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
            || self.completed_epochs > self.config.epochs
            || self.completed_epochs != self.metrics.completed_epochs
            || self.metrics.strength_measured
            || self.metrics.train_samples == 0
            || self.metrics.validation_samples == 0
            || self.metrics.train_samples > MAX_SAMPLES
            || self.metrics.validation_samples > MAX_SAMPLES
        {
            return Err("Invalid checkpoint metadata/metrics".into());
        }
        let expected_updates = self.metrics.train_samples.div_ceil(self.config.batch_size) as u64
            * self.completed_epochs as u64;
        if self.optimizer.step != expected_updates {
            return Err("Checkpoint epoch/Adam step mismatch".into());
        }
        for (report, expected) in [
            (&self.metrics.initial_train, self.metrics.train_samples),
            (
                &self.metrics.initial_validation,
                self.metrics.validation_samples,
            ),
            (&self.metrics.final_train, self.metrics.train_samples),
            (
                &self.metrics.final_validation,
                self.metrics.validation_samples,
            ),
        ] {
            if report.samples != expected
                || [report.policy_loss, report.value_loss, report.total_loss]
                    .iter()
                    .any(|value| !value.is_finite() || *value < 0.0)
            {
                return Err("Invalid checkpoint losses".into());
            }
        }
        if self.checksum.len() != 64 || self.checksum != self.expected_checksum()? {
            return Err("Training checkpoint checksum mismatch".into());
        }
        Ok(())
    }
    pub fn load(path: &Path) -> Result<Self, String> {
        let checkpoint: Self = read_json(path)?;
        checkpoint.validate()?;
        Ok(checkpoint)
    }
    pub fn save_new(&self, path: &Path) -> Result<(), String> {
        self.validate()?;
        write_new_json(path, self)
    }
}

enum Source<'a> {
    Dataset(&'a ValidatedDataset),
    Slices {
        train: &'a [TrainingSample],
        validation: &'a [TrainingSample],
    },
}
impl<'a> Source<'a> {
    fn iter(
        &self,
        split: DatasetSplit,
    ) -> Box<dyn Iterator<Item = Result<Cow<'a, TrainingSample>, String>> + 'a> {
        match self {
            Self::Dataset(dataset) => Box::new(
                dataset
                    .iter_split(split)
                    .map(|sample| sample.map(Cow::Owned)),
            ),
            Self::Slices { train, validation } => {
                let samples = match split {
                    DatasetSplit::Train => *train,
                    DatasetSplit::Validation => *validation,
                    DatasetSplit::Test => &[],
                };
                Box::new(samples.iter().map(|sample| Ok(Cow::Borrowed(sample))))
            }
        }
    }
    fn fingerprint(&self) -> Result<String, String> {
        match self {
            Self::Dataset(dataset) => Ok(dataset.manifest().fingerprint.clone()),
            Self::Slices { train, validation } => {
                struct HashWriter(Sha256);
                impl Write for HashWriter {
                    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
                        self.0.update(bytes);
                        Ok(bytes.len())
                    }
                    fn flush(&mut self) -> std::io::Result<()> {
                        Ok(())
                    }
                }
                let mut writer = HashWriter(Sha256::new());
                serde_json::to_writer(&mut writer, &(train, validation))
                    .map_err(|error| error.to_string())?;
                Ok(format!("{:x}", writer.0.finalize()))
            }
        }
    }
    fn audit(&self) -> Result<SourceInfo, String> {
        if let Self::Dataset(dataset) = self {
            let manifest = dataset.manifest();
            if manifest.schema != 1
                || manifest.feature_schema != FEATURE_SCHEMA
                || manifest.feature_count != FEATURE_COUNT
                || manifest.catalog_hash != crate::replay::catalog_hash()
                || manifest.samples > MAX_SAMPLES
            {
                return Err("Incompatible/big training dataset manifest".into());
            }
            let mut games = HashSet::new();
            let mut families = HashMap::new();
            for game in &manifest.games {
                if !games.insert(&game.game_id) {
                    return Err("Duplicate dataset game ID".into());
                }
                if let Some(previous) = families.insert(&game.family_id, &game.split)
                    && previous != &game.split
                {
                    return Err("Dataset family crosses split partitions".into());
                }
            }
        }
        let mut train_ids = HashSet::new();
        let mut train_families = HashSet::new();
        let mut counts = [0_usize; 2];
        let mut candidates = [0_usize; 2];
        for (side, split) in [DatasetSplit::Train, DatasetSplit::Validation]
            .into_iter()
            .enumerate()
        {
            for sample in self.iter(split) {
                let sample = sample?;
                validate_sample(&sample)?;
                if sample.features.is_empty() || sample.features.len() > MAX_CANDIDATES {
                    return Err("Invalid/big sample legal candidate set".into());
                }
                if side == 0 {
                    train_ids.insert(sample.game_id.clone());
                    train_families.insert(sample.family_id.clone());
                } else if train_ids.contains(&sample.game_id)
                    || train_families.contains(&sample.family_id)
                {
                    return Err("Training/validation game or family overlap".into());
                }
                counts[side] += 1;
                candidates[side] = candidates[side]
                    .checked_add(sample.features.len())
                    .ok_or("Dataset candidate overflow")?;
                if counts[side] > MAX_SAMPLES || candidates[side] > MAX_TOTAL_CANDIDATES {
                    return Err("Dataset exceeds bounded sample/candidate budget".into());
                }
            }
        }
        if counts.contains(&0) {
            return Err("Both training and isolated validation samples are required".into());
        }
        Ok(SourceInfo {
            counts,
            candidates,
            fingerprint: self.fingerprint()?,
        })
    }
}
struct SourceInfo {
    counts: [usize; 2],
    candidates: [usize; 2],
    fingerprint: String,
}

#[derive(Default)]
struct Scratch {
    hidden: Vec<f32>,
    logits: Vec<f32>,
    probabilities: Vec<f32>,
}
fn sample_loss(
    model: &TinyModel,
    sample: &TrainingSample,
    value_weight: f32,
    scratch: &mut Scratch,
    gradient: Option<&mut [f32]>,
) -> Result<(f64, f64), String> {
    validate_sample(sample)?;
    let count = sample.features.len();
    if count == 0 || count > MAX_CANDIDATES {
        return Err("Invalid/big candidate count".into());
    }
    scratch.hidden.resize(count * HIDDEN, 0.0);
    scratch.logits.resize(count, 0.0);
    scratch.probabilities.resize(count, 0.0);
    for (index, features) in sample.features.iter().enumerate() {
        let hidden = &mut scratch.hidden[index * HIDDEN..(index + 1) * HIDDEN];
        model.hidden_into(features, hidden)?;
        scratch.logits[index] = model.policy_logit(hidden)?;
    }
    policy_softmax(&scratch.logits, &mut scratch.probabilities)?;
    let chosen_hidden = &scratch.hidden[sample.chosen * HIDDEN..(sample.chosen + 1) * HIDDEN];
    let value_logits = model.value_logits(chosen_hidden);
    let value = value_softmax(value_logits, sample.active)?;
    // f64 log-sum-exp metrics avoid clipped/underflowed cross-entropy. Forward
    // predictions and the trained parameters remain f32.
    let policy_max = scratch
        .logits
        .iter()
        .copied()
        .fold(f32::NEG_INFINITY, f32::max) as f64;
    let policy_normalizer = scratch
        .logits
        .iter()
        .map(|logit| (*logit as f64 - policy_max).exp())
        .sum::<f64>()
        .ln();
    let policy_loss = policy_max - scratch.logits[sample.chosen] as f64 + policy_normalizer;
    let value_max = value_logits
        .iter()
        .zip(sample.active)
        .filter(|(_, active)| *active)
        .map(|(logit, _)| *logit)
        .fold(f32::NEG_INFINITY, f32::max) as f64;
    let value_normalizer = value_logits
        .iter()
        .zip(sample.active)
        .filter(|(_, active)| *active)
        .map(|(logit, _)| (*logit as f64 - value_max).exp())
        .sum::<f64>()
        .ln();
    let value_loss = value_logits
        .iter()
        .zip(sample.utilities)
        .zip(sample.active)
        .filter(|(_, active)| *active)
        .map(|((logit, target), _)| target as f64 * (value_max - *logit as f64 + value_normalizer))
        .sum::<f64>();
    if let Some(gradient) = gradient {
        if gradient.len() != PARAMETER_COUNT {
            return Err("Invalid gradient shape".into());
        }
        let value_delta: [f32; VALUE_SIDES] = std::array::from_fn(|side| {
            if sample.active[side] {
                value_weight * (value[side] - sample.utilities[side])
            } else {
                0.0
            }
        });
        for side in 0..VALUE_SIDES {
            gradient[BV + side] += value_delta[side];
            for unit in 0..HIDDEN {
                gradient[WV + side * HIDDEN + unit] += value_delta[side] * chosen_hidden[unit];
            }
        }
        for candidate in 0..count {
            let hidden = &scratch.hidden[candidate * HIDDEN..(candidate + 1) * HIDDEN];
            let policy_delta = scratch.probabilities[candidate]
                - if candidate == sample.chosen { 1.0 } else { 0.0 };
            gradient[BP] += policy_delta;
            for unit in 0..HIDDEN {
                gradient[WP + unit] += policy_delta * hidden[unit];
                let mut hidden_delta = policy_delta * model.parameters[WP + unit];
                if candidate == sample.chosen {
                    for (side, delta) in value_delta.iter().enumerate() {
                        hidden_delta += delta * model.parameters[WV + side * HIDDEN + unit];
                    }
                }
                let activation_delta = hidden_delta * (1.0 - hidden[unit] * hidden[unit]);
                gradient[B1 + unit] += activation_delta;
                for feature in 0..FEATURE_COUNT {
                    gradient[W1 + unit * FEATURE_COUNT + feature] +=
                        activation_delta * sample.features[candidate][feature];
                }
            }
        }
    }
    Ok((policy_loss, value_loss))
}

fn evaluate(
    source: &Source<'_>,
    split: DatasetSplit,
    model: &TinyModel,
    weight: f32,
    expected: usize,
    scratch: &mut Scratch,
) -> Result<LossReport, String> {
    let mut report = LossReport {
        samples: 0,
        policy_loss: 0.0,
        value_loss: 0.0,
        total_loss: 0.0,
    };
    for sample in source.iter(split) {
        let sample = sample?;
        let (policy, value) = sample_loss(model, sample.as_ref(), weight, scratch, None)?;
        report.samples += 1;
        report.policy_loss += policy;
        report.value_loss += value;
    }
    if report.samples != expected || expected == 0 {
        return Err("Dataset count changed during evaluation".into());
    }
    report.policy_loss /= expected as f64;
    report.value_loss /= expected as f64;
    report.total_loss = report.policy_loss + weight as f64 * report.value_loss;
    Ok(report)
}

fn train_source(
    source: Source<'_>,
    config: &TrainingConfig,
    resume: Option<&TrainingCheckpoint>,
) -> Result<TrainingOutcome, String> {
    config.validate()?;
    let info = source.audit()?;
    let (mut model, mut optimizer, mut random, completed, initial_metrics) =
        if let Some(previous) = resume {
            previous.validate()?;
            if previous.dataset_fingerprint != info.fingerprint
                || !config.compatible(&previous.config)
                || config.epochs < previous.completed_epochs
                || previous.metrics.train_samples != info.counts[0]
                || previous.metrics.validation_samples != info.counts[1]
            {
                return Err("Resume dataset/config/epoch mismatch".into());
            }
            (
                previous.model.model.clone(),
                previous.optimizer.clone(),
                Random {
                    state: previous.random_state,
                },
                previous.completed_epochs,
                Some(previous.metrics.clone()),
            )
        } else {
            (
                ModelArtifact::new(crate::replay::catalog_hash(), config.seed)?.model,
                AdamState::new(),
                Random {
                    state: config.seed ^ 0xd1b54a32d192ed03,
                },
                0,
                None,
            )
        };
    let remaining = config.epochs - completed;
    let evaluations = info.candidates[0]
        .checked_mul(remaining + 2)
        .and_then(|train| {
            info.candidates[1]
                .checked_mul(2)
                .and_then(|validation| train.checked_add(validation))
        })
        .ok_or("Training candidate budget overflow")?;
    let updates = info.counts[0].div_ceil(config.batch_size) as u64 * config.epochs as u64;
    if evaluations > MAX_CANDIDATE_EVALUATIONS || updates > MAX_UPDATES {
        return Err("Training exceeds bounded candidate/update budget".into());
    }
    let mut scratch = Scratch::default();
    let mut metrics = if let Some(metrics) = initial_metrics {
        metrics
    } else {
        let train = evaluate(
            &source,
            DatasetSplit::Train,
            &model,
            config.value_weight,
            info.counts[0],
            &mut scratch,
        )?;
        let validation = evaluate(
            &source,
            DatasetSplit::Validation,
            &model,
            config.value_weight,
            info.counts[1],
            &mut scratch,
        )?;
        TrainingMetrics {
            completed_epochs: 0,
            train_samples: info.counts[0],
            validation_samples: info.counts[1],
            initial_train: train.clone(),
            initial_validation: validation.clone(),
            final_train: train,
            final_validation: validation,
            strength_measured: false,
        }
    };
    let mut gradient = vec![0.0; PARAMETER_COUNT];
    let mut buffer = Vec::with_capacity(SHUFFLE_BUFFER);
    for _epoch in completed..config.epochs {
        let mut seen = 0;
        let mut batch = 0;
        let mut consume = |sample: Cow<'_, TrainingSample>| -> Result<(), String> {
            sample_loss(
                &model,
                &sample,
                config.value_weight,
                &mut scratch,
                Some(&mut gradient),
            )?;
            seen += 1;
            batch += 1;
            if batch == config.batch_size {
                optimizer.update(&mut model, &mut gradient, batch, config.learning_rate)?;
                batch = 0;
            }
            Ok(())
        };
        for sample in source.iter(DatasetSplit::Train) {
            let sample = sample?;
            if buffer.len() < SHUFFLE_BUFFER {
                buffer.push(sample);
            } else {
                let index = random.index(buffer.len());
                consume(std::mem::replace(&mut buffer[index], sample))?;
            }
        }
        while !buffer.is_empty() {
            let index = random.index(buffer.len());
            consume(buffer.swap_remove(index))?;
        }
        if seen != info.counts[0] {
            return Err("Dataset count changed during training".into());
        }
        if batch > 0 {
            optimizer.update(&mut model, &mut gradient, batch, config.learning_rate)?;
        }
    }
    metrics.completed_epochs = config.epochs;
    metrics.final_train = evaluate(
        &source,
        DatasetSplit::Train,
        &model,
        config.value_weight,
        info.counts[0],
        &mut scratch,
    )?;
    metrics.final_validation = evaluate(
        &source,
        DatasetSplit::Validation,
        &model,
        config.value_weight,
        info.counts[1],
        &mut scratch,
    )?;
    let artifact = ModelArtifact::from_model(crate::replay::catalog_hash(), model)?;
    let mut checkpoint = TrainingCheckpoint {
        schema: CHECKPOINT_SCHEMA,
        dataset_fingerprint: info.fingerprint,
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
    Ok(TrainingOutcome {
        model: artifact,
        checkpoint,
        metrics,
    })
}

/// Production entry point: stream manifest-isolated shards with a fixed small shuffle window.
pub fn train_dataset(
    dataset: &ValidatedDataset,
    config: &TrainingConfig,
    resume: Option<&TrainingCheckpoint>,
) -> Result<TrainingOutcome, String> {
    train_source(Source::Dataset(dataset), config, resume)
}

/// Small fixture helper. The same isolation, forward, gradients and Adam path are used.
pub fn train(
    training: &[TrainingSample],
    validation: &[TrainingSample],
    config: &TrainingConfig,
    resume: Option<&TrainingCheckpoint>,
) -> Result<TrainingOutcome, String> {
    train_source(
        Source::Slices {
            train: training,
            validation,
        },
        config,
        resume,
    )
}
