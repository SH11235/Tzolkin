//! Portable scalar policy/value reference. Inputs are redacted observations only.
use std::fs::{self, File, OpenOptions};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use tzolkin_core::observation::{MOVE_SCHEMA, OBSERVATION_SCHEMA, Observation, observation_key};

use crate::Decision;
use crate::features::{FEATURE_COUNT, FEATURE_SCHEMA, FeatureEncoder};
use crate::kernel::{Kernel, ResolvedKernel};

pub const MODEL_SCHEMA: u32 = 1;
pub const MODEL_VERSION: &str = "tiny-policy-value-mlp-v1";
pub const LEARNED_POLICY_VERSION: &str = "learned-policy-v1";
pub const HIDDEN: usize = 32;
pub const VALUE_SIDES: usize = 5;
pub const MAX_CANDIDATES: usize = 4096;
pub const MAX_ARTIFACT_BYTES: usize = 8 * 1024 * 1024;
pub(crate) const W1: usize = 0;
pub(crate) const B1: usize = FEATURE_COUNT * HIDDEN;
pub(crate) const WP: usize = B1 + HIDDEN;
pub(crate) const BP: usize = WP + HIDDEN;
pub(crate) const WV: usize = BP + 1;
pub(crate) const BV: usize = WV + HIDDEN * VALUE_SIDES;
pub const PARAMETER_COUNT: usize = BV + VALUE_SIDES;

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct TinyModel {
    pub(crate) parameters: Vec<f32>,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct ModelArtifact {
    pub schema: u32,
    pub model_version: String,
    pub policy_version: String,
    pub feature_schema: u32,
    pub feature_count: usize,
    pub hidden_count: usize,
    pub value_sides: usize,
    pub catalog_hash: String,
    pub model: TinyModel,
    pub checksum: String,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct Prediction {
    pub policy_logit: f32,
    /// Actor-relative winner utility; inactive sides are exactly zero.
    pub utilities: [f32; VALUE_SIDES],
}

/// Immutable borrowed model validated once for repeated native selfplay inference.
/// The borrow prevents artifact mutation while this handle is in use.
pub struct LoadedPolicy<'a> {
    artifact: &'a ModelArtifact,
    kernel: ResolvedKernel,
}
impl<'a> LoadedPolicy<'a> {
    pub fn new(artifact: &'a ModelArtifact) -> Result<Self, String> {
        Self::with_kernel(artifact, Kernel::Scalar)
    }
    pub fn with_kernel(artifact: &'a ModelArtifact, kernel: Kernel) -> Result<Self, String> {
        artifact.validate()?;
        Ok(Self {
            artifact,
            kernel: kernel.resolve()?,
        })
    }
    pub fn backend(&self) -> &'static str {
        self.kernel.backend()
    }
    pub fn predict(
        &self,
        features: &[f32],
        active: [bool; VALUE_SIDES],
    ) -> Result<Prediction, String> {
        self.artifact
            .predict_validated(features, active, self.kernel)
    }
    pub fn choose_move(&self, observation: &Observation) -> Result<Decision, String> {
        self.artifact.choose_validated(observation, self.kernel)
    }
}

/// Deliberately simple, ordered f32 sum: the reference for optimized kernels.
pub fn scalar_dot(left: &[f32], right: &[f32]) -> Result<f32, String> {
    if left.len() != right.len() || left.iter().chain(right).any(|value| !value.is_finite()) {
        return Err("Invalid scalar dot operands".into());
    }
    let result = dot(left, right);
    if result.is_finite() {
        Ok(result)
    } else {
        Err("Non-finite scalar dot result".into())
    }
}

pub(crate) fn dot(left: &[f32], right: &[f32]) -> f32 {
    left.iter()
        .zip(right)
        .fold(0.0_f32, |sum, (a, b)| sum + a * b)
}

pub(crate) fn digest(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

pub(crate) fn check_active(active: [bool; VALUE_SIDES]) -> Result<(), String> {
    let count = active.iter().filter(|side| **side).count();
    if !(2..=VALUE_SIDES).contains(&count)
        || active
            .iter()
            .enumerate()
            .any(|(side, value)| *value != (side < count))
    {
        return Err("Expected contiguous actor-relative active player mask".into());
    }
    Ok(())
}

pub(crate) fn value_softmax(
    logits: [f32; VALUE_SIDES],
    active: [bool; VALUE_SIDES],
) -> Result<[f32; VALUE_SIDES], String> {
    check_active(active)?;
    if logits.iter().any(|value| !value.is_finite()) {
        return Err("Non-finite value logit".into());
    }
    let maximum = logits
        .iter()
        .zip(active)
        .filter(|(_, mask)| *mask)
        .map(|(value, _)| *value)
        .fold(f32::NEG_INFINITY, f32::max);
    let mut probabilities = [0.0; VALUE_SIDES];
    let mut total = 0.0;
    for side in 0..VALUE_SIDES {
        if active[side] {
            probabilities[side] = (logits[side] - maximum).exp();
            total += probabilities[side];
        }
    }
    if !total.is_finite() || total <= 0.0 {
        return Err("Invalid masked value softmax".into());
    }
    for value in &mut probabilities {
        *value /= total;
    }
    Ok(probabilities)
}

pub(crate) fn policy_softmax(logits: &[f32], output: &mut [f32]) -> Result<(), String> {
    if logits.is_empty()
        || logits.len() > MAX_CANDIDATES
        || output.len() != logits.len()
        || logits.iter().any(|value| !value.is_finite())
    {
        return Err("Invalid ragged policy logits".into());
    }
    let maximum = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut total = 0.0;
    for (probability, logit) in output.iter_mut().zip(logits) {
        *probability = (*logit - maximum).exp();
        total += *probability;
    }
    if !total.is_finite() || total <= 0.0 {
        return Err("Invalid policy softmax".into());
    }
    for value in output {
        *value /= total;
    }
    Ok(())
}

#[derive(Clone, Copy)]
pub(crate) struct Random {
    pub state: u64,
}

impl Random {
    pub fn next(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9e3779b97f4a7c15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
        z ^ (z >> 31)
    }
    pub fn unit(&mut self) -> f32 {
        (self.next() >> 40) as f32 / 16_777_216.0
    }
    pub fn index(&mut self, maximum: usize) -> usize {
        // Rejection avoids modulo bias, and always consumes deterministic draws.
        let limit = u64::MAX - u64::MAX % maximum as u64;
        loop {
            let value = self.next();
            if value < limit {
                return (value % maximum as u64) as usize;
            }
        }
    }
}

impl TinyModel {
    fn initialized(seed: u64) -> Self {
        let mut random = Random { state: seed };
        let mut parameters = vec![0.0; PARAMETER_COUNT];
        let input_scale = (6.0 / (FEATURE_COUNT + HIDDEN) as f32).sqrt();
        for value in &mut parameters[W1..B1] {
            *value = (random.unit() * 2.0 - 1.0) * input_scale;
        }
        let output_scale = (6.0 / (HIDDEN + VALUE_SIDES) as f32).sqrt();
        for value in &mut parameters[WP..BP] {
            *value = (random.unit() * 2.0 - 1.0) * output_scale;
        }
        for value in &mut parameters[WV..BV] {
            *value = (random.unit() * 2.0 - 1.0) * output_scale;
        }
        Self { parameters }
    }
    pub fn validate(&self) -> Result<(), String> {
        if self.parameters.len() != PARAMETER_COUNT
            || self.parameters.iter().any(|value| !value.is_finite())
        {
            return Err("Invalid model parameter shape/finite values".into());
        }
        Ok(())
    }
    pub fn parameters(&self) -> &[f32] {
        &self.parameters
    }
    pub(crate) fn hidden_into(&self, features: &[f32], hidden: &mut [f32]) -> Result<(), String> {
        if features.len() != FEATURE_COUNT
            || hidden.len() != HIDDEN
            || features.iter().any(|value| !value.is_finite())
        {
            return Err("Invalid model feature/hidden shape or values".into());
        }
        for (unit, value) in hidden.iter_mut().enumerate() {
            let start = W1 + unit * FEATURE_COUNT;
            let linear = dot(&self.parameters[start..start + FEATURE_COUNT], features)
                + self.parameters[B1 + unit];
            if !linear.is_finite() {
                return Err("Non-finite hidden activation".into());
            }
            *value = linear.tanh();
        }
        Ok(())
    }
    pub(crate) fn policy_logit(&self, hidden: &[f32]) -> Result<f32, String> {
        let value = dot(&self.parameters[WP..BP], hidden) + self.parameters[BP];
        if value.is_finite() {
            Ok(value)
        } else {
            Err("Non-finite policy logit".into())
        }
    }
    pub(crate) fn value_logits(&self, hidden: &[f32]) -> [f32; VALUE_SIDES] {
        std::array::from_fn(|side| {
            dot(
                &self.parameters[WV + side * HIDDEN..WV + (side + 1) * HIDDEN],
                hidden,
            ) + self.parameters[BV + side]
        })
    }
    fn hidden_with_kernel(
        &self,
        features: &[f32],
        hidden: &mut [f32],
        kernel: ResolvedKernel,
    ) -> Result<(), String> {
        if features.len() != FEATURE_COUNT
            || hidden.len() != HIDDEN
            || features.iter().any(|value| !value.is_finite())
        {
            return Err("Invalid model feature/hidden shape or values".into());
        }
        kernel.dot_rows_validated(&self.parameters[W1..B1], features, hidden);
        for (unit, value) in hidden.iter_mut().enumerate() {
            *value += self.parameters[B1 + unit];
            if !value.is_finite() {
                return Err("Non-finite hidden activation".into());
            }
            *value = value.tanh();
        }
        Ok(())
    }
    fn policy_with_kernel(&self, hidden: &[f32], kernel: ResolvedKernel) -> Result<f32, String> {
        let value = kernel.dot_validated(&self.parameters[WP..BP], hidden) + self.parameters[BP];
        if value.is_finite() {
            Ok(value)
        } else {
            Err("Non-finite policy logit".into())
        }
    }
    fn values_with_kernel(&self, hidden: &[f32], kernel: ResolvedKernel) -> [f32; VALUE_SIDES] {
        let mut values = [0.0; VALUE_SIDES];
        kernel.dot_rows_validated(&self.parameters[WV..BV], hidden, &mut values);
        for (side, value) in values.iter_mut().enumerate() {
            *value += self.parameters[BV + side];
        }
        values
    }
}

impl ModelArtifact {
    pub fn new(catalog_hash: String, seed: u64) -> Result<Self, String> {
        Self::from_model(catalog_hash, TinyModel::initialized(seed))
    }
    pub(crate) fn from_model(catalog_hash: String, model: TinyModel) -> Result<Self, String> {
        let mut artifact = Self {
            schema: MODEL_SCHEMA,
            model_version: MODEL_VERSION.into(),
            policy_version: LEARNED_POLICY_VERSION.into(),
            feature_schema: FEATURE_SCHEMA,
            feature_count: FEATURE_COUNT,
            hidden_count: HIDDEN,
            value_sides: VALUE_SIDES,
            catalog_hash,
            model,
            checksum: String::new(),
        };
        artifact.checksum = artifact.expected_checksum()?;
        artifact.validate()?;
        Ok(artifact)
    }
    fn expected_checksum(&self) -> Result<String, String> {
        let mut payload = self.clone();
        payload.checksum.clear();
        Ok(digest(
            &serde_json::to_vec(&payload).map_err(|error| error.to_string())?,
        ))
    }
    pub fn validate(&self) -> Result<(), String> {
        if self.schema != MODEL_SCHEMA
            || self.model_version != MODEL_VERSION
            || self.policy_version != LEARNED_POLICY_VERSION
            || self.feature_schema != FEATURE_SCHEMA
            || self.feature_count != FEATURE_COUNT
            || self.hidden_count != HIDDEN
            || self.value_sides != VALUE_SIDES
            || self.catalog_hash != crate::replay::catalog_hash()
        {
            return Err("Incompatible model/catalog/features/policy version".into());
        }
        self.model.validate()?;
        if self.checksum.len() != 64 || self.checksum != self.expected_checksum()? {
            return Err("Model checksum mismatch".into());
        }
        Ok(())
    }
    pub fn predict(
        &self,
        features: &[f32],
        active: [bool; VALUE_SIDES],
    ) -> Result<Prediction, String> {
        self.validate()?;
        self.predict_validated(features, active, Kernel::Scalar.resolve()?)
    }
    fn predict_validated(
        &self,
        features: &[f32],
        active: [bool; VALUE_SIDES],
        kernel: ResolvedKernel,
    ) -> Result<Prediction, String> {
        let mut hidden = [0.0; HIDDEN];
        self.model
            .hidden_with_kernel(features, &mut hidden, kernel)?;
        Ok(Prediction {
            policy_logit: self.model.policy_with_kernel(&hidden, kernel)?,
            utilities: value_softmax(self.model.values_with_kernel(&hidden, kernel), active)?,
        })
    }
    pub fn choose_move(&self, observation: &Observation) -> Result<Decision, String> {
        self.validate()?;
        self.choose_validated(observation, Kernel::Scalar.resolve()?)
    }
    fn choose_validated(
        &self,
        observation: &Observation,
        kernel: ResolvedKernel,
    ) -> Result<Decision, String> {
        if observation.schema != OBSERVATION_SCHEMA
            || observation.move_schema != MOVE_SCHEMA
            || !(2..=5).contains(&observation.players.len())
            || observation.actor >= observation.players.len()
            || observation.legal_actions.is_empty()
            || observation.legal_actions.len() > MAX_CANDIDATES
        {
            return Err("Invalid learned-policy observation/legal set".into());
        }
        let key = observation_key(observation)?;
        if key != observation.observation_key {
            return Err("Observation key mismatch".into());
        }
        let mut hidden = [0.0; HIDDEN];
        let mut best = None;
        let encoder = FeatureEncoder::new(observation)?;
        for (index, legal) in observation.legal_actions.iter().enumerate() {
            let features = encoder.encode_legal(index)?;
            self.model
                .hidden_with_kernel(&features, &mut hidden, kernel)?;
            let score = self.model.policy_with_kernel(&hidden, kernel)?;
            if best.as_ref().is_none_or(|(_, value)| score > *value) {
                best = Some((legal, score));
            }
        }
        let (legal, score) = best.ok_or("No legal learned-policy candidate")?;
        Ok(Decision {
            actor: observation.actor,
            observation_key: key,
            policy_version: LEARNED_POLICY_VERSION.into(),
            r#move: legal.r#move.clone(),
            score: score as f64,
        })
    }
    pub fn load(path: &Path) -> Result<Self, String> {
        let artifact: Self = read_json(path)?;
        artifact.validate()?;
        Ok(artifact)
    }
    pub fn save_new(&self, path: &Path) -> Result<(), String> {
        self.validate()?;
        write_new_json(path, self)
    }
}

pub(crate) fn read_json<T: serde::de::DeserializeOwned>(path: &Path) -> Result<T, String> {
    let mut bytes = Vec::new();
    File::open(path)
        .map_err(|error| error.to_string())?
        .take(MAX_ARTIFACT_BYTES as u64 + 1)
        .read_to_end(&mut bytes)
        .map_err(|error| error.to_string())?;
    if bytes.len() > MAX_ARTIFACT_BYTES {
        return Err("Artifact exceeds bounded 8 MiB input".into());
    }
    serde_json::from_slice(&bytes).map_err(|error| error.to_string())
}

static TEMPORARY_SERIAL: AtomicU64 = AtomicU64::new(0);
/// Publish with an atomic create-new hard link; existing files are never replaced.
pub(crate) fn write_new_json<T: Serialize>(path: &Path, value: &T) -> Result<(), String> {
    write_new_json_bounded(path, value, MAX_ARTIFACT_BYTES)
}
pub(crate) fn write_new_json_bounded<T: Serialize>(
    path: &Path,
    value: &T,
    limit: usize,
) -> Result<(), String> {
    let bytes = serde_json::to_vec(value).map_err(|error| error.to_string())?;
    if bytes.len() > limit {
        return Err(format!("Artifact exceeds bounded {limit} byte output"));
    }
    let parent = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    let name = path.file_name().ok_or("Artifact output needs a filename")?;
    fs::create_dir_all(parent).map_err(|error| error.to_string())?;
    let mut temporary: Option<(PathBuf, File)> = None;
    for _ in 0..100 {
        let serial = TEMPORARY_SERIAL.fetch_add(1, Ordering::Relaxed);
        let candidate = parent.join(format!(
            ".{}.{}.{serial}.tmp",
            name.to_string_lossy(),
            std::process::id()
        ));
        match OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&candidate)
        {
            Ok(file) => {
                temporary = Some((candidate, file));
                break;
            }
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => continue,
            Err(error) => return Err(error.to_string()),
        }
    }
    let (temporary_path, mut file) = temporary.ok_or("Could not reserve artifact staging file")?;
    let result = (|| {
        file.write_all(&bytes).map_err(|error| error.to_string())?;
        file.sync_all().map_err(|error| error.to_string())?;
        drop(file);
        fs::hard_link(&temporary_path, path).map_err(|error| error.to_string())
    })();
    let cleanup = fs::remove_file(&temporary_path).map_err(|error| error.to_string());
    result.and(cleanup)
}
