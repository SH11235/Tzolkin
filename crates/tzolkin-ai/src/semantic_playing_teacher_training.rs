//! Separate native schema-4 policy-only BC. Typed complete teacher data is the
//! only source; checkpoints establish consistency, not authenticated history.
//! Fixed-teacher imitation metrics are not mixed-roster or general strength.
use crate::dataset::DatasetSplit;
use crate::features::{FEATURE_COUNT, SEMANTIC_PLAYING_FEATURE_SCHEMA};
use crate::model::{MAX_ARTIFACT_BYTES, Random, digest};
use crate::policy_training::{
    MAX_CANDIDATE_EVALUATIONS, MAX_UPDATES, PolicyAdamState, PolicyBcConfig, PolicyLossReport,
    Scratch, bounded_json, publish, semantic_playing_policy_loss, validate_new_output_directory,
};
use crate::public_model::{PARAMETER_COUNT, PublicPolicyModel, ValueValidity, initialized_model};
use crate::semantic_playing_teacher_dataset::{
    self as data, SemanticPlayingTeacherManifest, ValidatedSemanticPlayingTeacherDataset,
    ValidatedSemanticPlayingTeacherSample,
};
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::Path;

pub const MODEL_SCHEMA: &str = "tzolkin-native-semantic-playing-teacher-model-v1";
pub const MODEL_VERSION: &str = "native-corn-first-v5-playing-512x32x1-scalar-v1";
pub const CHECKPOINT_SCHEMA: &str = "tzolkin-native-semantic-playing-teacher-checkpoint-v1";
pub const TRAINING_VERSION: &str = "native-playing4-teacher-scalar-adam-fisher-yates-v1";
pub const TASK: &str = "nativeCornFirstV5SemanticPlayingPolicyOnlyBc";
pub const V7_MODEL_VERSION: &str = "native-corn-first-v7-playing-512x32x1-scalar-v1";
pub const V7_TASK: &str = "nativeCornFirstV7SemanticPlayingPolicyOnlyBc";
const SHUFFLE: &str = "descending-fisher-yates-random-index-v1";
const INIT_DOMAIN: &[u8] = b"tzolkin-native-playing4-teacher-init-v1\0";
const SHUFFLE_DOMAIN: &[u8] = b"tzolkin-native-playing4-teacher-shuffle-v1\0";
impl data::TeacherProfile {
    fn model_version(self) -> &'static str {
        match self {
            Self::V5 => MODEL_VERSION,
            Self::V7 => V7_MODEL_VERSION,
        }
    }
    fn bc_task(self) -> &'static str {
        match self {
            Self::V5 => TASK,
            Self::V7 => V7_TASK,
        }
    }
}
fn seeded(domain: &[u8], seed: u64) -> u64 {
    let mut h = Sha256::new();
    h.update(domain);
    h.update(seed.to_le_bytes());
    u64::from_le_bytes(h.finalize()[..8].try_into().expect("eight digest bytes"))
}
fn checksum<T: Serialize>(value: &T) -> Result<String, String> {
    Ok(digest(&bounded_json(value)?))
}
fn read<T: DeserializeOwned>(path: &Path) -> Result<T, String> {
    let bytes = data::read(path, MAX_ARTIFACT_BYTES as u64)?;
    serde_json::from_value(crate::public_stochastic_record::unique_json(
        &bytes,
        MAX_ARTIFACT_BYTES,
    )?)
    .map_err(|e| e.to_string())
}
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct NativeTeacherDatasetIdentity {
    pub manifest: SemanticPlayingTeacherManifest,
    pub raw_manifest_bytes: u64,
    pub raw_manifest_sha256: String,
}
impl NativeTeacherDatasetIdentity {
    fn validate(&self) -> Result<(), String> {
        data::validate_manifest(&self.manifest)?;
        if self.raw_manifest_bytes == 0
            || self.raw_manifest_bytes > crate::policy_dataset::MAX_MANIFEST_BYTES
            || !crate::policy_dataset::digest_id(&self.raw_manifest_sha256)
            || self
                .manifest
                .stored_bytes
                .checked_add(self.raw_manifest_bytes)
                .is_none_or(|n| n > data::MAX_STORED_BYTES)
        {
            return Err("Invalid native4 teacher raw dataset identity".into());
        }
        Ok(())
    }
    fn count(&self, split: DatasetSplit) -> (usize, usize) {
        self.manifest
            .games
            .iter()
            .filter(|g| g.split == split)
            .fold((0, 0), |(r, c), g| {
                (r + g.counts.rows, c + g.counts.candidates)
            })
    }
}
/// Own wire and parameter key; no constructor/conversion from old or Human models.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct SemanticPlayingTeacherModel {
    schema: String,
    model_version: String,
    source_kind: String,
    task: String,
    input_contract: String,
    feature_schema: u32,
    feature_count: usize,
    dataset_fingerprint: String,
    value_validity: ValueValidity,
    native_playing_teacher_parameters: Vec<f32>,
    checksum: String,
}
impl SemanticPlayingTeacherModel {
    pub fn parameters(&self) -> &[f32] {
        &self.native_playing_teacher_parameters
    }
    fn expected_checksum(&self) -> Result<String, String> {
        let mut m = self.clone();
        m.checksum.clear();
        checksum(&m)
    }
    fn profile(&self) -> Result<data::TeacherProfile, String> {
        [data::TeacherProfile::V5, data::TeacherProfile::V7]
            .into_iter()
            .find(|p| {
                self.model_version == p.model_version()
                    && self.source_kind == p.source_kind()
                    && self.task == p.bc_task()
                    && self.input_contract == p.input_contract()
            })
            .ok_or_else(|| "Invalid native4 teacher model contract/checksum".into())
    }
    pub fn validate(&self) -> Result<(), String> {
        tzolkin_inference::policy::validate_parameters(self.parameters())?;
        self.profile()?;
        if self.schema != MODEL_SCHEMA
            || self.feature_schema != SEMANTIC_PLAYING_FEATURE_SCHEMA
            || self.feature_count != FEATURE_COUNT
            || self.value_validity != ValueValidity::UnavailablePolicyOnly
            || !crate::policy_dataset::digest_id(&self.dataset_fingerprint)
            || self.checksum != self.expected_checksum()?
        {
            return Err("Invalid native4 teacher model contract/checksum".into());
        }
        Ok(())
    }
    pub fn load(path: &Path) -> Result<Self, String> {
        let value: Self = read(path)?;
        value.validate()?;
        Ok(value)
    }
}
#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(rename_all = "camelCase")]
pub enum RoundBand {
    Early,
    Middle,
    Late,
}
fn band(round: i64) -> RoundBand {
    if round <= 9 {
        RoundBand::Early
    } else if round <= 18 {
        RoundBand::Middle
    } else {
        RoundBand::Late
    }
}
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct ValidationStratum {
    pub players: usize,
    pub round_band: RoundBand,
    #[serde(deserialize_with = "required_loss")]
    pub loss: Option<PolicyLossReport>,
}
fn required_loss<'de, D: serde::Deserializer<'de>>(
    d: D,
) -> Result<Option<PolicyLossReport>, D::Error> {
    Option::<PolicyLossReport>::deserialize(d)
}
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct TeacherEvaluation {
    pub overall: PolicyLossReport,
    /// Exactly six groups. Empty groups have None, never an imputed zero loss.
    pub validation_strata: Vec<ValidationStratum>,
}
fn empty_loss() -> PolicyLossReport {
    PolicyLossReport {
        samples: 0,
        candidates: 0,
        policy_loss: 0.0,
        correct_top1: 0,
        value_loss: (),
        value_samples: 0,
        value_validity: ValueValidity::UnavailablePolicyOnly,
    }
}
fn finish_loss(mut r: PolicyLossReport) -> Result<PolicyLossReport, String> {
    r.policy_loss /= r.samples as f64;
    r.validate(r.samples, r.candidates)?;
    Ok(r)
}
fn evaluate(
    rows: &[&ValidatedSemanticPlayingTeacherSample],
    model: &PublicPolicyModel,
    validation: bool,
) -> Result<TeacherEvaluation, String> {
    evaluate_with_fit(rows, model, validation, None)
}
fn evaluate_with_fit(
    rows: &[&ValidatedSemanticPlayingTeacherSample],
    model: &PublicPolicyModel,
    validation: bool,
    mut fit: Option<&mut Vec<NativeTeacherSampleFit>>,
) -> Result<TeacherEvaluation, String> {
    let mut total = empty_loss();
    let mut groups: BTreeMap<(usize, RoundBand), PolicyLossReport> = BTreeMap::new();
    let mut scratch = Scratch::default();
    for s in rows {
        // One real scalar forward per row contributes to overall and its group.
        let (loss, correct) =
            semantic_playing_policy_loss(model, s.candidates(), s.chosen(), &mut scratch, None)?;
        if let Some(fit) = fit.as_deref_mut() {
            if fit.len() >= 2 * crate::policy_dataset::MAX_PARTITION_SAMPLES || !loss.is_finite() {
                return Err("Native teacher fit row bound/finite mismatch".into());
            }
            fit.push(NativeTeacherSampleFit {
                sample_id: s.sample_id().into(),
                split: if validation {
                    DatasetSplit::Validation
                } else {
                    DatasetSplit::Train
                },
                players: s.players(),
                round: s.round(),
                nll: loss,
                correct_top1: correct,
            });
        }
        for r in std::iter::once(&mut total).chain(if validation {
            Some(
                groups
                    .entry((s.players(), band(s.round())))
                    .or_insert_with(empty_loss),
            )
        } else {
            None
        }) {
            r.samples += 1;
            r.candidates += s.candidates().len();
            r.policy_loss += loss;
            r.correct_top1 += usize::from(correct);
        }
    }
    let overall = finish_loss(total)?;
    let mut validation_strata = Vec::new();
    if validation {
        for players in [3, 4] {
            for round_band in [RoundBand::Early, RoundBand::Middle, RoundBand::Late] {
                validation_strata.push(ValidationStratum {
                    players,
                    round_band,
                    loss: groups
                        .remove(&(players, round_band))
                        .map(finish_loss)
                        .transpose()?,
                });
            }
        }
    }
    Ok(TeacherEvaluation {
        overall,
        validation_strata,
    })
}
#[derive(Clone, Debug, Default, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct AliasDiagnostics {
    pub rows: usize,
    pub candidates: usize,
    pub rows_with_candidate_aliases: usize,
    pub chosen_class_aliased_rows: usize,
    pub repeated_input_groups: usize,
    /// Same ordered input bits, different chosen feature classes; not intent.
    pub conflicting_chosen_feature_class_groups: usize,
    pub index_only_within_same_class_groups: usize,
    pub differing_ordered_legal_hash_groups: usize,
}
#[derive(Default)]
struct AliasGroup {
    rows: usize,
    chosen_classes: BTreeSet<String>,
    chosen_indices: BTreeSet<usize>,
    legal_hashes: BTreeSet<String>,
}
fn feature_key(row: &crate::features::EncodedCandidate) -> Result<String, String> {
    let mut h = Sha256::new();
    h.update(b"native-playing4-feature-class-v1\0");
    h.update((FEATURE_COUNT as u64).to_le_bytes());
    h.update(SEMANTIC_PLAYING_FEATURE_SCHEMA.to_le_bytes());
    for x in row.values_for_schema(SEMANTIC_PLAYING_FEATURE_SCHEMA)? {
        h.update(x.to_bits().to_le_bytes());
    }
    Ok(format!("{:x}", h.finalize()))
}
fn aliases(rows: &[&ValidatedSemanticPlayingTeacherSample]) -> Result<AliasDiagnostics, String> {
    let mut report = AliasDiagnostics::default();
    let mut groups: BTreeMap<String, AliasGroup> = BTreeMap::new();
    for s in rows {
        let mut matrix = Sha256::new();
        matrix.update(b"native-playing4-ordered-input-v1\0");
        matrix.update((s.candidates().len() as u64).to_le_bytes());
        matrix.update((FEATURE_COUNT as u64).to_le_bytes());
        matrix.update(SEMANTIC_PLAYING_FEATURE_SCHEMA.to_le_bytes());
        let mut classes: BTreeMap<String, usize> = BTreeMap::new();
        let mut chosen = None;
        for (i, row) in s.candidates().iter().enumerate() {
            for v in row.values_for_schema(SEMANTIC_PLAYING_FEATURE_SCHEMA)? {
                matrix.update(v.to_bits().to_le_bytes());
            }
            let key = feature_key(row)?;
            *classes.entry(key.clone()).or_default() += 1;
            if i == s.chosen() {
                chosen = Some(key);
            }
        }
        let chosen = chosen.ok_or("Invalid teacher collision chosen index")?;
        report.rows += 1;
        report.candidates += s.candidates().len();
        report.rows_with_candidate_aliases += usize::from(classes.values().any(|n| *n > 1));
        report.chosen_class_aliased_rows += usize::from(classes[&chosen] > 1);
        let g = groups
            .entry(format!("{:x}", matrix.finalize()))
            .or_default();
        g.rows += 1;
        g.chosen_classes.insert(chosen);
        g.chosen_indices.insert(s.chosen());
        g.legal_hashes.insert(s.ordered_legal_sha256().into());
    }
    for g in groups.values().filter(|g| g.rows > 1) {
        report.repeated_input_groups += 1;
        report.conflicting_chosen_feature_class_groups += usize::from(g.chosen_classes.len() > 1);
        report.index_only_within_same_class_groups +=
            usize::from(g.chosen_classes.len() == 1 && g.chosen_indices.len() > 1);
        report.differing_ordered_legal_hash_groups += usize::from(g.legal_hashes.len() > 1);
    }
    Ok(report)
}
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct SemanticPlayingTeacherMetrics {
    pub completed_epochs: usize,
    pub initial_train: TeacherEvaluation,
    pub initial_validation: TeacherEvaluation,
    pub final_train: TeacherEvaluation,
    pub final_validation: TeacherEvaluation,
    pub train_aliases: AliasDiagnostics,
    pub validation_aliases: AliasDiagnostics,
    pub strength_measured: bool,
}
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct SemanticPlayingTeacherCheckpoint {
    pub schema: String,
    pub training_version: String,
    pub dataset: NativeTeacherDatasetIdentity,
    pub config: PolicyBcConfig,
    pub completed_epochs: usize,
    pub shuffle_algorithm: String,
    pub random_state: u64,
    pub model: SemanticPlayingTeacherModel,
    pub optimizer: PolicyAdamState,
    pub metrics: SemanticPlayingTeacherMetrics,
    pub checksum: String,
}
fn validate_evaluation(
    e: &TeacherEvaluation,
    counts: (usize, usize),
    validation: bool,
) -> Result<(), String> {
    e.overall.validate(counts.0, counts.1)?;
    if !validation {
        if !e.validation_strata.is_empty() {
            return Err("Unexpected Train strata".into());
        }
        return Ok(());
    }
    let mut i = 0;
    let mut total = (0, 0, 0, 0.0);
    for players in [3, 4] {
        for round_band in [RoundBand::Early, RoundBand::Middle, RoundBand::Late] {
            let s = e
                .validation_strata
                .get(i)
                .ok_or("Missing validation stratum")?;
            i += 1;
            if s.players != players || s.round_band != round_band {
                return Err("Validation stratum order mismatch".into());
            }
            if let Some(r) = &s.loss {
                r.validate(r.samples, r.candidates)?;
                total.0 += r.samples;
                total.1 += r.candidates;
                total.2 += r.correct_top1;
                total.3 += r.policy_loss * r.samples as f64;
            }
        }
    }
    if e.validation_strata.len() != 6
        || (total.0, total.1, total.2) != (counts.0, counts.1, e.overall.correct_top1)
        || !total.3.is_finite()
        || (total.3 / counts.0 as f64 - e.overall.policy_loss).abs()
            > 1e-10 * (1.0 + e.overall.policy_loss.abs())
    {
        return Err("Validation strata totals differ".into());
    }
    Ok(())
}
impl SemanticPlayingTeacherCheckpoint {
    fn expected_checksum(&self) -> Result<String, String> {
        let mut c = self.clone();
        c.checksum.clear();
        checksum(&c)
    }
    pub fn validate(&self) -> Result<(), String> {
        self.dataset.validate()?;
        self.config.validate()?;
        self.model.validate()?;
        self.optimizer.validate()?;
        let train = self.dataset.count(DatasetSplit::Train);
        let validation = self.dataset.count(DatasetSplit::Validation);
        let steps = train.0.div_ceil(self.config.batch_size) as u64;
        if train.0 == 0
            || validation.0 == 0
            || self.schema != CHECKPOINT_SCHEMA
            || self.training_version != TRAINING_VERSION
            || self.shuffle_algorithm != SHUFFLE
            || self.completed_epochs == 0
            || self.completed_epochs != self.config.epochs
            || self.metrics.completed_epochs != self.completed_epochs
            || self.metrics.strength_measured
            || self.model.dataset_fingerprint != self.dataset.manifest.fingerprint
            || self.model.profile()? != self.dataset.manifest.profile()?
            || steps.checked_mul(self.completed_epochs as u64) != Some(self.optimizer.step)
            || self.optimizer.step > MAX_UPDATES
            || self.checksum != self.expected_checksum()?
        {
            return Err(
                "Invalid native4 teacher checkpoint epoch/source/optimizer/checksum".into(),
            );
        }
        for e in [&self.metrics.initial_train, &self.metrics.final_train] {
            validate_evaluation(e, train, false)?;
        }
        for e in [
            &self.metrics.initial_validation,
            &self.metrics.final_validation,
        ] {
            validate_evaluation(e, validation, true)?;
        }
        for (a, n) in [
            (&self.metrics.train_aliases, train),
            (&self.metrics.validation_aliases, validation),
        ] {
            if (a.rows, a.candidates) != n
                || a.rows_with_candidate_aliases > a.rows
                || a.chosen_class_aliased_rows > a.rows_with_candidate_aliases
                || a.repeated_input_groups > a.rows / 2
                || a.conflicting_chosen_feature_class_groups > a.repeated_input_groups
                || a.index_only_within_same_class_groups > a.repeated_input_groups
                || a.differing_ordered_legal_hash_groups > a.repeated_input_groups
            {
                return Err("Invalid teacher alias counts".into());
            }
        }
        Ok(())
    }
    pub fn load(path: &Path) -> Result<Self, String> {
        let c: Self = read(path)?;
        c.validate()?;
        Ok(c)
    }
    pub fn save_new(&self, path: &Path) -> Result<(), String> {
        self.validate()?;
        data::local(path)?;
        publish(path, self)
    }
}
#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct SemanticPlayingTeacherOutcome {
    pub model: SemanticPlayingTeacherModel,
    pub checkpoint: SemanticPlayingTeacherCheckpoint,
    pub metrics: SemanticPlayingTeacherMetrics,
}
impl SemanticPlayingTeacherOutcome {
    pub fn save_new_directory(&self, directory: &Path) -> Result<(), String> {
        self.checkpoint.validate()?;
        if self.model != self.checkpoint.model || self.metrics != self.checkpoint.metrics {
            return Err("Inconsistent native4 teacher outcome".into());
        }
        data::local(directory)?;
        validate_new_output_directory(directory)?;
        bounded_json(&self.model)?;
        bounded_json(&self.checkpoint)?;
        bounded_json(&self.metrics)?;
        fs::create_dir(directory).map_err(|e| e.to_string())?;
        publish(&directory.join("model.json"), &self.model)?;
        publish(&directory.join("checkpoint.json"), &self.checkpoint)?;
        publish(&directory.join("metrics.json"), &self.metrics)
    }
}
fn selected<'a>(
    samples: &'a [ValidatedSemanticPlayingTeacherSample],
    m: &SemanticPlayingTeacherManifest,
    split: DatasetSplit,
) -> Vec<&'a ValidatedSemanticPlayingTeacherSample> {
    samples
        .iter()
        .filter(|s| {
            m.families
                .iter()
                .any(|f| f.family_id == s.family_id() && f.split == split)
        })
        .collect()
}
fn train_after_cache(
    dataset: &ValidatedSemanticPlayingTeacherDataset,
    config: &PolicyBcConfig,
    resume: Option<&SemanticPlayingTeacherCheckpoint>,
    after_cache: impl FnOnce() -> Result<(), String>,
) -> Result<SemanticPlayingTeacherOutcome, String> {
    train_after_cache_with_fit(dataset, config, resume, after_cache, None)
}
fn train_after_cache_with_fit(
    dataset: &ValidatedSemanticPlayingTeacherDataset,
    config: &PolicyBcConfig,
    resume: Option<&SemanticPlayingTeacherCheckpoint>,
    after_cache: impl FnOnce() -> Result<(), String>,
    mut fit: Option<&mut Vec<NativeTeacherSampleFit>>,
) -> Result<SemanticPlayingTeacherOutcome, String> {
    dataset.with_checked_cache(|samples, raw_pin| {
        config.validate()?;
        let identity = NativeTeacherDatasetIdentity {
            manifest: dataset.manifest().clone(),
            raw_manifest_bytes: raw_pin.0,
            raw_manifest_sha256: raw_pin.1.clone(),
        };
        identity.validate()?;
        let profile = identity.manifest.profile()?;
        // Test is removed before ALL loss, top1, alias and round-band operations.
        let train = selected(samples, &identity.manifest, DatasetSplit::Train);
        let validation = selected(samples, &identity.manifest, DatasetSplit::Validation);
        if train.is_empty() || validation.is_empty() {
            return Err("Nonempty native4 Train/Validation required".into());
        }
        let completed = resume.map_or(0, |p| p.completed_epochs);
        if let Some(p) = resume {
            p.validate()?;
            if p.dataset != identity || !config.compatible(&p.config) || config.epochs < completed {
                return Err("Native4 teacher resume source/order/config mismatch".into());
            }
        }
        let remaining = config.epochs - completed;
        if fit.is_some()
            && (profile != data::TeacherProfile::V7 || resume.is_none() || remaining != 0)
        {
            return Err("Fit diagnostics require a V7 no-op resume".into());
        }
        let metric_passes = 1 + usize::from(resume.is_some()) + usize::from(remaining > 0);
        let tc = identity.count(DatasetSplit::Train);
        let vc = identity.count(DatasetSplit::Validation);
        if (
            train.len(),
            train.iter().map(|s| s.candidates().len()).sum::<usize>(),
        ) != tc
            || (
                validation.len(),
                validation
                    .iter()
                    .map(|s| s.candidates().len())
                    .sum::<usize>(),
            ) != vc
        {
            return Err("Native4 teacher cached partition counts differ".into());
        }
        let evaluations =
            tc.1.checked_mul(remaining + metric_passes)
                .and_then(|n| {
                    vc.1.checked_mul(metric_passes)
                        .and_then(|v| n.checked_add(v))
                })
                .ok_or("Native4 teacher evaluation overflow")?;
        if evaluations > MAX_CANDIDATE_EVALUATIONS
            || (tc.0.div_ceil(config.batch_size) as u64)
                .checked_mul(config.epochs as u64)
                .is_none_or(|n| n > MAX_UPDATES)
        {
            return Err("Native4 teacher candidate/update budget exceeded".into());
        }
        after_cache()?;
        let initial = initialized_model(seeded(INIT_DOMAIN, config.seed));
        let initial_train = evaluate(&train, &initial, false)?;
        let initial_validation = evaluate(&validation, &initial, true)?;
        let train_aliases = aliases(&train)?;
        let validation_aliases = aliases(&validation)?;
        let mut model = PublicPolicyModel {
            parameters: resume
                .map_or_else(|| initial.parameters, |p| p.model.parameters().to_vec()),
        };
        model.validate()?;
        let mut optimizer = resume.map_or_else(PolicyAdamState::new, |p| p.optimizer.clone());
        let mut random = Random {
            state: resume.map_or_else(|| seeded(SHUFFLE_DOMAIN, config.seed), |p| p.random_state),
        };
        let (current_train, current_validation) = if let Some(p) = resume {
            let t = evaluate_with_fit(&train, &model, false, fit.as_deref_mut())?;
            let v = evaluate_with_fit(&validation, &model, true, fit.as_deref_mut())?;
            if p.metrics.initial_train != initial_train
                || p.metrics.initial_validation != initial_validation
                || p.metrics.final_train != t
                || p.metrics.final_validation != v
                || p.metrics.train_aliases != train_aliases
                || p.metrics.validation_aliases != validation_aliases
            {
                return Err("Native4 teacher resume model/metric/collision mismatch".into());
            }
            (t, v)
        } else {
            (initial_train.clone(), initial_validation.clone())
        };
        let mut gradient = vec![0.0; PARAMETER_COUNT];
        let mut scratch = Scratch::default();
        for _ in completed..config.epochs {
            // Canonical indices are reset each epoch; no pending partial buffer.
            let mut order = (0..train.len()).collect::<Vec<_>>();
            for i in (1..order.len()).rev() {
                let j = random.index(i + 1);
                order.swap(i, j);
            }
            let mut batch = 0;
            for index in order {
                let s = train[index];
                semantic_playing_policy_loss(
                    &model,
                    s.candidates(),
                    s.chosen(),
                    &mut scratch,
                    Some(&mut gradient),
                )?;
                batch += 1;
                if batch == config.batch_size {
                    optimizer.update(&mut model, &mut gradient, batch, config.learning_rate)?;
                    batch = 0;
                }
            }
            if batch > 0 {
                optimizer.update(&mut model, &mut gradient, batch, config.learning_rate)?;
            }
        }
        let (final_train, final_validation) = if remaining > 0 {
            (
                evaluate(&train, &model, false)?,
                evaluate(&validation, &model, true)?,
            )
        } else {
            (current_train, current_validation)
        };
        let metrics = SemanticPlayingTeacherMetrics {
            completed_epochs: config.epochs,
            initial_train,
            initial_validation,
            final_train,
            final_validation,
            train_aliases,
            validation_aliases,
            strength_measured: false,
        };
        let mut artifact = SemanticPlayingTeacherModel {
            schema: MODEL_SCHEMA.into(),
            model_version: profile.model_version().into(),
            source_kind: profile.source_kind().into(),
            task: profile.bc_task().into(),
            input_contract: profile.input_contract().into(),
            feature_schema: SEMANTIC_PLAYING_FEATURE_SCHEMA,
            feature_count: FEATURE_COUNT,
            dataset_fingerprint: identity.manifest.fingerprint.clone(),
            value_validity: ValueValidity::UnavailablePolicyOnly,
            native_playing_teacher_parameters: model.parameters,
            checksum: String::new(),
        };
        artifact.checksum = artifact.expected_checksum()?;
        artifact.validate()?;
        let mut checkpoint = SemanticPlayingTeacherCheckpoint {
            schema: CHECKPOINT_SCHEMA.into(),
            training_version: TRAINING_VERSION.into(),
            dataset: identity,
            config: config.clone(),
            completed_epochs: config.epochs,
            shuffle_algorithm: SHUFFLE.into(),
            random_state: random.state,
            model: artifact.clone(),
            optimizer,
            metrics: metrics.clone(),
            checksum: String::new(),
        };
        checkpoint.checksum = checkpoint.expected_checksum()?;
        checkpoint.validate()?;
        Ok(SemanticPlayingTeacherOutcome {
            model: artifact,
            checkpoint,
            metrics,
        })
    })
}
// Advisory selected-model rows, kept outside every serialized BC core type.
const FIT_SCHEMA: &str = "tzolkin-native-v7-selected-bc-fit-diagnostics-v1";
const MAX_FIT_BYTES: usize = 32 * 1024 * 1024;
#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct NativeTeacherSampleFit {
    sample_id: String,
    split: DatasetSplit,
    players: usize,
    round: i64,
    nll: f64,
    correct_top1: bool,
}
#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct NativeTeacherFitDiagnostics {
    schema: String,
    dataset_fingerprint: String,
    raw_manifest_sha256: String,
    checkpoint_checksum: String,
    model_checksum: String,
    config: PolicyBcConfig,
    completed_epochs: usize,
    existing_metric_passes_per_row: usize,
    diagnostic_additional_forward_passes: usize,
    training_updates: usize,
    strength_claimed: bool,
    rows: Vec<NativeTeacherSampleFit>,
    checksum: String,
}
impl NativeTeacherFitDiagnostics {
    fn expected_checksum(&self) -> Result<String, String> {
        let mut value = serde_json::to_value(self).map_err(|e| e.to_string())?;
        value["checksum"] = "".into();
        let mut hash = Sha256::new();
        hash.update(b"tzolkin-native-v7-selected-bc-fit-diagnostics-v1\0");
        hash.update(serde_json::to_vec(&value).map_err(|e| e.to_string())?);
        Ok(format!("{:x}", hash.finalize()))
    }
    /// Caller must finish source POST before this final advisory publication;
    /// an actual successful CLI exit is still required for acceptance.
    pub fn save_new(&self, directory: &Path) -> Result<(), String> {
        data::local(directory)?;
        if self.checksum != self.expected_checksum()? {
            return Err("Fit checksum".into());
        }
        let bytes = serde_json::to_vec(self).map_err(|e| e.to_string())?;
        if bytes.len() > MAX_FIT_BYTES {
            return Err("Fit diagnostics byte cap".into());
        }
        crate::policy_dataset::publish_bytes(directory, "fit-diagnostics.json", &bytes)
    }
}
/// Reuses only the selected model's existing no-op resume metric pass. The
/// existing initialized-model checks still run; no optimizer/RNG step is added.
pub fn train_dataset_with_fit_diagnostics(
    dataset: &ValidatedSemanticPlayingTeacherDataset,
    config: &PolicyBcConfig,
    checkpoint: &SemanticPlayingTeacherCheckpoint,
) -> Result<(SemanticPlayingTeacherOutcome, NativeTeacherFitDiagnostics), String> {
    let mut rows = Vec::new();
    let outcome = train_after_cache_with_fit(
        dataset,
        config,
        Some(checkpoint),
        || Ok(()),
        Some(&mut rows),
    )?;
    if bounded_json(&outcome.checkpoint)? != bounded_json(checkpoint)? {
        return Err("Fit no-op changed checkpoint".into());
    }
    let mut report = NativeTeacherFitDiagnostics {
        schema: FIT_SCHEMA.into(),
        dataset_fingerprint: checkpoint.dataset.manifest.fingerprint.clone(),
        raw_manifest_sha256: checkpoint.dataset.raw_manifest_sha256.clone(),
        checkpoint_checksum: checkpoint.checksum.clone(),
        model_checksum: checkpoint.model.checksum.clone(),
        config: config.clone(),
        completed_epochs: checkpoint.completed_epochs,
        existing_metric_passes_per_row: 2,
        diagnostic_additional_forward_passes: 0,
        training_updates: 0,
        strength_claimed: false,
        rows,
        checksum: String::new(),
    };
    report.checksum = report.expected_checksum()?;
    Ok((outcome, report))
}

pub fn train_dataset(
    dataset: &ValidatedSemanticPlayingTeacherDataset,
    config: &PolicyBcConfig,
    resume: Option<&SemanticPlayingTeacherCheckpoint>,
) -> Result<SemanticPlayingTeacherOutcome, String> {
    train_after_cache(dataset, config, resume, || Ok(()))
}
/// Audited no-op consistency, not proof of the original training history.
pub fn validate_checkpoint_dataset(
    dataset: &ValidatedSemanticPlayingTeacherDataset,
    checkpoint: &SemanticPlayingTeacherCheckpoint,
) -> Result<(), String> {
    let audited = train_dataset(dataset, &checkpoint.config, Some(checkpoint))?;
    if bounded_json(&audited.checkpoint)? != bounded_json(checkpoint)? {
        return Err("Native4 teacher audited no-op differs".into());
    }
    Ok(())
}
pub fn evaluate_validation(
    dataset: &ValidatedSemanticPlayingTeacherDataset,
    checkpoint: &SemanticPlayingTeacherCheckpoint,
) -> Result<TeacherEvaluation, String> {
    // No public split argument can request Test forwards.
    validate_checkpoint_dataset(dataset, checkpoint)?;
    Ok(checkpoint.metrics.final_validation.clone())
}

#[cfg(test)]
pub(crate) fn fixture(
    parent: &Path,
    examples: &[(
        tzolkin_core::observation::Observation,
        tzolkin_core::observation::LegalAction,
        usize,
    )],
) {
    assert_eq!(band(9), RoundBand::Early);
    assert_eq!(band(10), RoundBand::Middle);
    assert_eq!(band(18), RoundBand::Middle);
    assert_eq!(band(19), RoundBand::Late);
    assert_eq!(band(28), RoundBand::Late);
    let root = parent.join("native4-training-cache");
    // Invalid cached Test profile ensures no Test loss/top1/aliases/bands are read.
    // This private synthetic cache cannot pass the production complete-source loader.
    let dataset = data::training_cache_fixture(&root, examples, Some(DatasetSplit::Test)).unwrap();
    let config = |epochs| PolicyBcConfig {
        epochs,
        batch_size: 3,
        learning_rate: 0.001,
        seed: 7,
    };
    let full = train_dataset(&dataset, &config(3), None).unwrap();
    let first = train_dataset(&dataset, &config(1), None).unwrap();
    let resumed = train_dataset(&dataset, &config(3), Some(&first.checkpoint)).unwrap();
    assert_eq!(
        bounded_json(&full.checkpoint).unwrap(),
        bounded_json(&resumed.checkpoint).unwrap()
    );
    assert_eq!(full.checkpoint.optimizer.step, 6);
    validate_checkpoint_dataset(&dataset, &full.checkpoint).unwrap();
    let noop = train_dataset(&dataset, &config(3), Some(&full.checkpoint)).unwrap();
    assert_eq!(
        bounded_json(&noop.checkpoint).unwrap(),
        bounded_json(&full.checkpoint).unwrap()
    );
    let v7_root = parent.join("native4-v7-training-cache");
    let v7_data = data::training_cache_fixture_for_profile(
        &v7_root,
        examples,
        Some(DatasetSplit::Test),
        data::TeacherProfile::V7,
    )
    .unwrap();
    let v7_full = train_dataset(&v7_data, &config(3), None).unwrap();
    let v7_first = train_dataset(&v7_data, &config(1), None).unwrap();
    let v7_resumed = train_dataset(&v7_data, &config(3), Some(&v7_first.checkpoint)).unwrap();
    assert_eq!(
        bounded_json(&v7_full.checkpoint).unwrap(),
        bounded_json(&v7_resumed.checkpoint).unwrap()
    );
    validate_checkpoint_dataset(&v7_data, &v7_full.checkpoint).unwrap();
    let (fit_noop, fit) =
        train_dataset_with_fit_diagnostics(&v7_data, &config(3), &v7_full.checkpoint).unwrap();
    assert_eq!(
        bounded_json(&fit_noop.checkpoint).unwrap(),
        bounded_json(&v7_full.checkpoint).unwrap()
    );
    assert_eq!(
        fit.rows.len(),
        v7_full.metrics.final_train.overall.samples
            + v7_full.metrics.final_validation.overall.samples
    );
    assert!(
        fit.rows
            .iter()
            .all(|r| r.split != DatasetSplit::Test && r.nll.is_finite())
    );
    for (split, expected) in [
        (DatasetSplit::Train, &v7_full.metrics.final_train.overall),
        (
            DatasetSplit::Validation,
            &v7_full.metrics.final_validation.overall,
        ),
    ] {
        let rows: Vec<_> = fit.rows.iter().filter(|r| r.split == split).collect();
        assert_eq!(
            rows.iter().filter(|r| r.correct_top1).count(),
            expected.correct_top1
        );
        assert!(
            (rows.iter().map(|r| r.nll).sum::<f64>() / rows.len() as f64 - expected.policy_loss)
                .abs()
                < 1e-12
        );
    }
    assert_eq!(fit.checksum, fit.expected_checksum().unwrap());
    let fit_output = parent.join("native4-v7-fit-diagnostics");
    fs::create_dir(&fit_output).unwrap();
    fit.save_new(&fit_output).unwrap();
    let saved = fs::read(fit_output.join("fit-diagnostics.json")).unwrap();
    assert_eq!(saved, serde_json::to_vec(&fit).unwrap());
    assert!(fit.save_new(&fit_output).is_err());
    assert_eq!(
        fs::read(fit_output.join("fit-diagnostics.json")).unwrap(),
        saved
    );
    assert!(train_dataset_with_fit_diagnostics(&v7_data, &config(4), &v7_full.checkpoint).is_err());
    assert!(train_dataset_with_fit_diagnostics(&dataset, &config(3), &full.checkpoint).is_err());
    assert_eq!(v7_full.model.source_kind, data::V7_SOURCE_KIND);
    assert_eq!(v7_full.model.task, V7_TASK);
    assert_eq!(v7_full.model.input_contract, data::V7_INPUT_CONTRACT);
    assert_eq!(v7_full.model.model_version, V7_MODEL_VERSION);
    // Identical synthetic features/labels retain exact numerical math; teacher
    // metadata/fingerprints differ. These caches are not qualified replays.
    assert_eq!(
        full.model
            .parameters()
            .iter()
            .map(|v| v.to_bits())
            .collect::<Vec<_>>(),
        v7_full
            .model
            .parameters()
            .iter()
            .map(|v| v.to_bits())
            .collect::<Vec<_>>()
    );
    assert_eq!(
        bounded_json(&full.checkpoint.optimizer).unwrap(),
        bounded_json(&v7_full.checkpoint.optimizer).unwrap()
    );
    assert_eq!(
        full.checkpoint.random_state,
        v7_full.checkpoint.random_state
    );
    assert!(train_dataset(&v7_data, &config(3), Some(&first.checkpoint)).is_err());
    assert!(train_dataset(&dataset, &config(3), Some(&v7_first.checkpoint)).is_err());
    let mut mismatched = first.checkpoint.clone();
    mismatched.model.source_kind = data::V7_SOURCE_KIND.into();
    mismatched.model.task = V7_TASK.into();
    mismatched.model.input_contract = data::V7_INPUT_CONTRACT.into();
    mismatched.model.model_version = V7_MODEL_VERSION.into();
    mismatched.model.checksum = mismatched.model.expected_checksum().unwrap();
    mismatched.model.validate().unwrap();
    mismatched.checksum = mismatched.expected_checksum().unwrap();
    assert!(mismatched.validate().is_err());
    let v7_checkpoint = v7_root.with_extension("checkpoint.json");
    v7_full.checkpoint.save_new(&v7_checkpoint).unwrap();
    assert!(
        crate::arena::teacher::PreparedSemanticPlayingTeacher::load(&v7_checkpoint, &v7_root)
            .is_err()
    );
    assert_eq!(
        evaluate_validation(&dataset, &full.checkpoint).unwrap(),
        full.metrics.final_validation
    );
    assert_eq!(
        full.metrics
            .final_validation
            .validation_strata
            .iter()
            .filter(|g| g.loss.is_none())
            .count(),
        4
    );
    for field in ["family", "raw", "order", "profile"] {
        let mut wrong = first.checkpoint.clone();
        match field {
            "family" => wrong.dataset.manifest.families[0].family_id = "0".repeat(64),
            "raw" => wrong.dataset.raw_manifest_sha256 = "0".repeat(64),
            "order" => wrong.dataset.manifest.games.swap(0, 1),
            _ => wrong.model.feature_schema = 2,
        }
        if field == "profile" {
            wrong.model.checksum = wrong.model.expected_checksum().unwrap();
        }
        wrong.checksum = wrong.expected_checksum().unwrap();
        assert!(train_dataset(&dataset, &config(3), Some(&wrong)).is_err());
    }
    let mut changed = config(3);
    changed.learning_rate = 0.002;
    assert!(train_dataset(&dataset, &changed, Some(&first.checkpoint)).is_err());
    let mut changed = config(3);
    changed.batch_size = 2;
    assert!(train_dataset(&dataset, &changed, Some(&first.checkpoint)).is_err());
    let bad = data::training_cache_fixture(
        &parent.join("native4-bad-train-cache"),
        examples,
        Some(DatasetSplit::Train),
    )
    .unwrap();
    assert!(train_dataset(&bad, &config(1), None).is_err());
    let aliases_rows = data::collision_fixture(&examples[0]);
    let same_class = aliases(&aliases_rows[..2].iter().collect::<Vec<_>>()).unwrap();
    assert_eq!(same_class.index_only_within_same_class_groups, 1);
    assert_eq!(same_class.conflicting_chosen_feature_class_groups, 0);
    assert_eq!(same_class.chosen_class_aliased_rows, 2);
    let different = aliases(&aliases_rows.iter().collect::<Vec<_>>()).unwrap();
    assert_eq!(different.conflicting_chosen_feature_class_groups, 1);
    assert_eq!(different.index_only_within_same_class_groups, 0);
    assert_eq!(different.differing_ordered_legal_hash_groups, 1);
    assert_eq!(different.chosen_class_aliased_rows, 2);
    // Source mutation is observed by the whole-operation POST even on Err.
    let file = root.join("sources/source-000000.json");
    let before = fs::read(&file).unwrap();
    for numerical_error in [false, true] {
        let failed = train_after_cache(&dataset, &config(1), None, || {
            fs::write(&file, b"changed").unwrap();
            if numerical_error {
                Err("injected numerical failure".into())
            } else {
                Ok(())
            }
        });
        assert!(failed.unwrap_err().contains("changed after load"));
        fs::write(&file, &before).unwrap();
    }
    dataset.recheck().unwrap();
    let raw = bounded_json(&full.checkpoint).unwrap();
    let mut wrong: serde_json::Value = serde_json::from_slice(&raw).unwrap();
    wrong["schema"] =
        serde_json::json!(crate::human_policy_training::SEMANTIC_PLAYING_CHECKPOINT_SCHEMA);
    assert!(
        serde_json::from_value::<SemanticPlayingTeacherCheckpoint>(wrong)
            .unwrap()
            .validate()
            .is_err()
    );
}
