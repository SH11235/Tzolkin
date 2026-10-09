//! Independent, unqualified state-return estimates from sealed public V2 context.
//! This module supplies inference, not value training, PPO, or a calibrated win probability.
use std::fs::{self, File};
use std::io::{self, Read, Write};
use std::path::{Component, Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use serde::{Deserialize, Serialize};
use tzolkin_core::Phase;
use tzolkin_core::observation::{MOVE_SCHEMA, OBSERVATION_SCHEMA, Observation};

use crate::features::{FeatureEncoder, PUBLIC_FEATURE_SCHEMA};
use crate::kernel::{Kernel, ResolvedKernel};
use crate::model::{MAX_ARTIFACT_BYTES, Random, digest};

pub const MODEL_SCHEMA: &str = "tzolkin-public-state-critic-model-v1";
pub const MODEL_VERSION: &str = "tiny-public-state-critic-mlp-v1";
pub const CONTEXT_CONTRACT: &str = "public-v2-state-prefix384-v1";
pub const CONTEXT_SCHEMA: u32 = 1;
pub const CONTEXT_COUNT: usize = 384;
pub const HIDDEN: usize = 32;
pub const PARAMETER_COUNT: usize = CONTEXT_COUNT * HIDDEN + HIDDEN + HIDDEN + 1;
const B1: usize = CONTEXT_COUNT * HIDDEN;
const WV: usize = B1 + HIDDEN;
const BV: usize = WV + HIDDEN;
const TASK: &str = "actorWinnerShareStateCritic";

/// Only the Observation constructor can create this context. It validates base 3/4p
/// Setup/Playing and uses the candidate-independent V2 prefix, including legal count.
/// A matching key does not authenticate source provenance or the supplied legal set.
///
/// No arbitrary array or chosen candidate conversion is available.
/// ```compile_fail
/// use tzolkin_ai::public_state_critic::PublicStateContext;
/// let _ = PublicStateContext::from_raw([0.0_f32; 384]);
/// ```
/// ```compile_fail
/// use tzolkin_ai::public_state_critic::PublicStateContext;
/// let _: PublicStateContext = serde_json::from_str("{}").unwrap();
/// ```
/// ```compile_fail
/// use tzolkin_ai::features::EncodedCandidate;
/// use tzolkin_ai::public_state_critic::PublicStateContext;
/// fn from_chosen(row: EncodedCandidate) -> PublicStateContext { row.into() }
/// ```
/// ```compile_fail
/// use tzolkin_ai::public_state_critic::PublicStateContext;
/// let _ = PublicStateContext {
///     context_schema: 1, feature_schema: 2, values: [0.0_f32; 384],
///     actor: 0, player_count: 3, phase: tzolkin_core::Phase::Playing,
/// };
/// ```
#[derive(Clone, Debug, PartialEq)]
pub struct PublicStateContext {
    context_schema: u32,
    feature_schema: u32,
    values: [f32; CONTEXT_COUNT],
    actor: usize,
    player_count: usize,
    phase: Phase,
}
impl PublicStateContext {
    pub fn from_observation(observation: &Observation) -> Result<Self, String> {
        crate::public_model::validate_contract(observation)?;
        let encoder = FeatureEncoder::new_public(observation)?;
        Ok(Self {
            context_schema: CONTEXT_SCHEMA,
            feature_schema: PUBLIC_FEATURE_SCHEMA,
            values: encoder.public_state_prefix()?,
            actor: observation.actor,
            player_count: observation.players.len(),
            phase: observation.phase,
        })
    }
    pub fn context_schema(&self) -> u32 {
        self.context_schema
    }
    pub fn feature_schema(&self) -> u32 {
        self.feature_schema
    }
    pub fn actor(&self) -> usize {
        self.actor
    }
    pub fn player_count(&self) -> usize {
        self.player_count
    }
    pub fn phase(&self) -> Phase {
        self.phase
    }
    pub(crate) fn context_checksum(&self) -> String {
        let mut bytes = Vec::with_capacity(CONTEXT_COUNT * 4 + 128);
        bytes.extend_from_slice(b"tzolkin-public-state-context-v1\0");
        bytes.extend_from_slice(CONTEXT_CONTRACT.as_bytes());
        bytes.push(0);
        for value in [
            self.context_schema,
            self.feature_schema,
            CONTEXT_COUNT as u32,
            self.actor as u32,
            self.player_count as u32,
        ] {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        bytes.push(match self.phase {
            Phase::Setup => 0,
            Phase::Playing => 1,
            Phase::Finished => 2,
        });
        for value in &self.values {
            bytes.extend_from_slice(&value.to_bits().to_le_bytes());
        }
        digest(&bytes)
    }
}

/// Finite inference alone establishes neither training history nor calibration.
#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub enum EstimateValidity {
    UnqualifiedStateEstimate,
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub enum CriticQualification {
    Unqualified,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct PublicStateCriticModel {
    parameters: Vec<f32>,
}
impl PublicStateCriticModel {
    pub fn parameters(&self) -> &[f32] {
        &self.parameters
    }
    fn validate(&self) -> Result<(), String> {
        if self.parameters.len() != PARAMETER_COUNT
            || self.parameters.iter().any(|value| !value.is_finite())
        {
            return Err("Invalid state critic parameter shape/finite values".into());
        }
        Ok(())
    }
}

/// A plain inference artifact. Checksum integrity is not source/training authentication.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct PublicStateCriticArtifact {
    pub schema: String,
    pub model_version: String,
    pub context_contract: String,
    pub feature_schema: u32,
    pub context_schema: u32,
    pub input_size: usize,
    pub hidden_size: usize,
    pub output_size: usize,
    pub parameter_count: usize,
    pub task: String,
    pub output_transform: String,
    pub qualification: CriticQualification,
    pub rules_version: u32,
    pub rules_baseline: String,
    pub observation_schema: u32,
    pub move_schema: u32,
    pub catalog_hash: String,
    pub model: PublicStateCriticModel,
    pub checksum: String,
}
impl PublicStateCriticArtifact {
    /// Deterministically initialized and untrained. This does not run value training.
    pub fn new(seed: u64) -> Result<Self, String> {
        let mut random = Random { state: seed };
        let mut parameters = vec![0.0; PARAMETER_COUNT];
        let input_scale = (6.0 / (CONTEXT_COUNT + HIDDEN) as f32).sqrt();
        for value in &mut parameters[..B1] {
            *value = (random.unit() * 2.0 - 1.0) * input_scale;
        }
        let output_scale = (6.0 / (HIDDEN + 1) as f32).sqrt();
        for value in &mut parameters[WV..BV] {
            *value = (random.unit() * 2.0 - 1.0) * output_scale;
        }
        let mut artifact = Self {
            schema: MODEL_SCHEMA.into(),
            model_version: MODEL_VERSION.into(),
            context_contract: CONTEXT_CONTRACT.into(),
            feature_schema: PUBLIC_FEATURE_SCHEMA,
            context_schema: CONTEXT_SCHEMA,
            input_size: CONTEXT_COUNT,
            hidden_size: HIDDEN,
            output_size: 1,
            parameter_count: PARAMETER_COUNT,
            task: TASK.into(),
            output_transform: "linear".into(),
            qualification: CriticQualification::Unqualified,
            rules_version: crate::replay::RULES_VERSION,
            rules_baseline: crate::replay::RULES_BASELINE.into(),
            observation_schema: OBSERVATION_SCHEMA,
            move_schema: MOVE_SCHEMA,
            catalog_hash: crate::replay::catalog_hash(),
            model: PublicStateCriticModel { parameters },
            checksum: String::new(),
        };
        artifact.checksum = artifact.expected_checksum()?;
        artifact.validate()?;
        Ok(artifact)
    }
    fn expected_checksum(&self) -> Result<String, String> {
        let mut payload = self.clone();
        payload.checksum.clear();
        Ok(digest(&bounded_json(&payload)?))
    }
    pub fn validate(&self) -> Result<(), String> {
        if self.schema != MODEL_SCHEMA
            || self.model_version != MODEL_VERSION
            || self.context_contract != CONTEXT_CONTRACT
            || self.feature_schema != PUBLIC_FEATURE_SCHEMA
            || self.context_schema != CONTEXT_SCHEMA
            || self.input_size != CONTEXT_COUNT
            || self.hidden_size != HIDDEN
            || self.output_size != 1
            || self.parameter_count != PARAMETER_COUNT
            || self.task != TASK
            || self.output_transform != "linear"
            || self.qualification != CriticQualification::Unqualified
            || self.rules_version != crate::replay::RULES_VERSION
            || self.rules_baseline != crate::replay::RULES_BASELINE
            || self.observation_schema != OBSERVATION_SCHEMA
            || self.move_schema != MOVE_SCHEMA
            || self.catalog_hash != crate::replay::catalog_hash()
        {
            return Err("Incompatible state critic model/context/rules contract".into());
        }
        self.model.validate()?;
        if self.checksum.len() != 64 || self.checksum != self.expected_checksum()? {
            return Err("State critic checksum mismatch".into());
        }
        Ok(())
    }
    pub fn load(path: &Path) -> Result<Self, String> {
        let path = local_path(path)?;
        let metadata = fs::symlink_metadata(&path).map_err(|error| error.to_string())?;
        if !metadata.is_file() || metadata.len() > MAX_ARTIFACT_BYTES as u64 {
            return Err("State critic requires a local regular input bounded to 8 MiB".into());
        }
        let mut bytes = Vec::new();
        File::open(&path)
            .map_err(|error| error.to_string())?
            .take(MAX_ARTIFACT_BYTES as u64 + 1)
            .read_to_end(&mut bytes)
            .map_err(|error| error.to_string())?;
        if bytes.len() > MAX_ARTIFACT_BYTES {
            return Err("State critic input exceeds 8 MiB".into());
        }
        let artifact: Self = serde_json::from_slice(&bytes).map_err(|error| error.to_string())?;
        artifact.validate()?;
        Ok(artifact)
    }
    pub fn save_new(&self, path: &Path) -> Result<(), String> {
        self.validate()?;
        let path = local_path(path)?;
        let bytes = bounded_json(self)?;
        publish_new(&path, &bytes)
    }
}

/// This finite raw estimate can be outside [0,1]. It is not a calibrated probability.
#[derive(Clone, Debug, Serialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct StateCriticEstimate {
    pub actor: usize,
    pub player_count: usize,
    pub raw_return_estimate: f32,
    pub validity: EstimateValidity,
    pub backend: String,
}

/// Immutable model and backend validated once; no raw-array inference entrypoint.
/// ```compile_fail
/// use tzolkin_ai::public_state_critic::{LoadedPublicStateCritic, PublicStateCriticArtifact};
/// let model = PublicStateCriticArtifact::new(1).unwrap();
/// let loaded = LoadedPublicStateCritic::new(&model).unwrap();
/// loaded.estimate(&[0.0_f32; 384]).unwrap();
/// ```
/// ```compile_fail
/// use tzolkin_ai::public_state_critic::{LoadedPublicStateCritic, PublicStateCriticArtifact};
/// let mut model = PublicStateCriticArtifact::new(1).unwrap();
/// let loaded = LoadedPublicStateCritic::new(&model).unwrap();
/// model.feature_schema = 1;
/// loaded.backend();
/// ```
pub struct LoadedPublicStateCritic<'a> {
    artifact: &'a PublicStateCriticArtifact,
    kernel: ResolvedKernel,
}
impl<'a> LoadedPublicStateCritic<'a> {
    pub fn new(artifact: &'a PublicStateCriticArtifact) -> Result<Self, String> {
        Self::with_kernel(artifact, Kernel::Scalar)
    }
    pub fn with_kernel(
        artifact: &'a PublicStateCriticArtifact,
        kernel: Kernel,
    ) -> Result<Self, String> {
        artifact.validate()?;
        Ok(Self {
            artifact,
            kernel: kernel.resolve()?,
        })
    }
    pub fn backend(&self) -> &'static str {
        self.kernel.backend()
    }
    pub fn estimate(&self, context: &PublicStateContext) -> Result<StateCriticEstimate, String> {
        if context.context_schema != CONTEXT_SCHEMA
            || context.feature_schema != PUBLIC_FEATURE_SCHEMA
        {
            return Err("State critic context schema mismatch".into());
        }
        let parameters = &self.artifact.model.parameters;
        let mut hidden = [0.0; HIDDEN];
        self.kernel
            .dot_rows_validated(&parameters[..B1], &context.values, &mut hidden);
        for (unit, value) in hidden.iter_mut().enumerate() {
            *value += parameters[B1 + unit];
            if !value.is_finite() {
                return Err("Non-finite state critic hidden activation".into());
            }
            *value = value.tanh();
        }
        let raw_return_estimate =
            self.kernel.dot_validated(&parameters[WV..BV], &hidden) + parameters[BV];
        if !raw_return_estimate.is_finite() {
            return Err("Non-finite state critic return estimate".into());
        }
        Ok(StateCriticEstimate {
            actor: context.actor,
            player_count: context.player_count,
            raw_return_estimate,
            validity: EstimateValidity::UnqualifiedStateEstimate,
            backend: self.backend().into(),
        })
    }
}

fn is_link(metadata: &fs::Metadata) -> bool {
    #[cfg(windows)]
    {
        use std::os::windows::fs::MetadataExt;
        // Reject all reparse points, including directory junctions.
        metadata.file_type().is_symlink() || metadata.file_attributes() & 0x400 != 0
    }
    #[cfg(not(windows))]
    {
        metadata.file_type().is_symlink()
    }
}
fn local_path(path: &Path) -> Result<PathBuf, String> {
    let text = path.to_string_lossy();
    if text.is_empty()
        || text.contains('\0')
        || text.contains("://")
        || text.starts_with("\\\\")
        || text.starts_with("//")
        || (!path.is_absolute() && matches!(path.components().next(), Some(Component::Prefix(_))))
    {
        return Err("State critic requires a local path".into());
    }
    let absolute = if path.is_absolute() {
        path.to_path_buf()
    } else {
        std::env::current_dir()
            .map_err(|error| error.to_string())?
            .join(path)
    };
    if absolute.to_string_lossy().starts_with("\\\\")
        || absolute.to_string_lossy().starts_with("//")
    {
        return Err("State critic network hierarchy rejected".into());
    }
    // Inspect root to leaf so a parent link is rejected before any traversal.
    let mut current = PathBuf::new();
    for component in absolute.components() {
        current.push(component.as_os_str());
        if matches!(component, Component::Prefix(_)) {
            continue;
        }
        match fs::symlink_metadata(&current) {
            Ok(metadata) if is_link(&metadata) => {
                return Err("State critic symlink/junction hierarchy rejected".into());
            }
            Ok(_) => {}
            Err(error) if error.kind() == io::ErrorKind::NotFound => {}
            Err(error) => return Err(error.to_string()),
        }
    }
    Ok(absolute)
}

struct BoundedBytes(Vec<u8>);
impl Write for BoundedBytes {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        if self
            .0
            .len()
            .checked_add(bytes.len())
            .is_none_or(|length| length > MAX_ARTIFACT_BYTES)
        {
            return Err(io::Error::other("State critic output exceeds 8 MiB"));
        }
        self.0.extend_from_slice(bytes);
        Ok(bytes.len())
    }
    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}
fn bounded_json(value: &impl Serialize) -> Result<Vec<u8>, String> {
    let mut writer = BoundedBytes(Vec::new());
    serde_json::to_writer(&mut writer, value).map_err(|error| error.to_string())?;
    Ok(writer.0)
}
static TEMPORARY_SERIAL: AtomicU64 = AtomicU64::new(0);
fn publish_new(path: &Path, bytes: &[u8]) -> Result<(), String> {
    let parent = path.parent().ok_or("State critic output needs a parent")?;
    let name = path
        .file_name()
        .ok_or("State critic output needs a filename")?;
    fs::create_dir_all(parent).map_err(|error| error.to_string())?;
    let mut staging = None;
    for _ in 0..100 {
        let serial = TEMPORARY_SERIAL.fetch_add(1, Ordering::Relaxed);
        let candidate = parent.join(format!(
            ".{}.{}.{serial}.tmp",
            name.to_string_lossy(),
            std::process::id()
        ));
        match File::options()
            .write(true)
            .create_new(true)
            .open(&candidate)
        {
            Ok(file) => {
                staging = Some((candidate, file));
                break;
            }
            Err(error) if error.kind() == io::ErrorKind::AlreadyExists => {}
            Err(error) => return Err(error.to_string()),
        }
    }
    let (temporary, mut file) = staging.ok_or("Could not reserve state critic staging file")?;
    let result = (|| {
        file.write_all(bytes)
            .and_then(|_| file.flush())
            .and_then(|_| file.sync_all())
            .map_err(|error| error.to_string())?;
        drop(file);
        fs::hard_link(&temporary, path).map_err(|error| error.to_string())
    })();
    // Cleanup failure cannot turn a successfully published final into a reported failure.
    // On publication failure only this invocation's staging file is removed.
    let _ = fs::remove_file(&temporary);
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use tzolkin_core::{create_game, observation::observe};

    #[test]
    fn only_schema_two_candidate_independent_prefix_is_exposed_inside_crate() {
        for players in [3, 4] {
            let state =
                create_game((0..players).map(|p| format!("P{p}")).collect(), 42, false).unwrap();
            let observation = observe(&state, state.current_player).unwrap();
            assert!(
                FeatureEncoder::new(&observation)
                    .unwrap()
                    .public_state_prefix()
                    .is_err()
            );
            let context = PublicStateContext::from_observation(&observation).unwrap();
            let encoder = FeatureEncoder::new_public(&observation).unwrap();
            for index in 0..observation.legal_actions.len() {
                let row = encoder.encode_legal_tagged(index).unwrap();
                let values = row.values_for_schema(PUBLIC_FEATURE_SCHEMA).unwrap();
                assert_eq!(context.values, values[..CONTEXT_COUNT]);
            }
        }
    }
    #[test]
    fn finite_parameter_and_bounded_serialization_errors_are_rejected() {
        let mut artifact = PublicStateCriticArtifact::new(7).unwrap();
        for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            artifact.model.parameters[0] = value;
            assert!(artifact.validate().unwrap_err().contains("finite"));
        }
        let oversized = "x".repeat(MAX_ARTIFACT_BYTES);
        assert!(bounded_json(&oversized).unwrap_err().contains("8 MiB"));
    }
}
