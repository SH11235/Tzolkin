//! Separate policy-only BC for sealed conditional Human inputs, never native admission.
use crate::dataset::DatasetSplit;
#[cfg(test)]
use crate::features::FeatureEncoder;
use crate::features::{FEATURE_COUNT, MAX_LEGAL_ACTIONS, PUBLIC_FEATURE_SCHEMA};
use crate::human_initial_wealth::ContentPin;
use crate::human_initial_wealth::table_family_id;
use crate::human_policy_dataset::{
    CAPPED_EXTRACTION_LABEL_SCOPE, ConditionalFeatureProfile, ConditionalHumanPolicyDataset,
    EXPLICIT_MARKET_ANYWHERE_LABEL_SCOPE, FACTORIZED_LABEL_SCOPE, HumanPolicySample,
    OBSERVED_CORN_BEG_LABEL_SCOPE, OBSERVED_PENDING_CHOICE_LABEL_SCOPE,
    PAID_TECHNOLOGY_LABEL_SCOPE, TASK,
};
use crate::kernel::{Kernel, ResolvedKernel};
use crate::model::{Random, digest};
use crate::policy_training::{
    MAX_CANDIDATE_EVALUATIONS, MAX_UPDATES, PolicyAdamState, PolicyBcConfig, PolicyLossReport,
    PolicyTrainingMetrics, Scratch, bounded_json, policy_loss, publish, read_local_json,
    read_local_json_pinned, semantic_playing_policy_loss, validate_new_output_directory,
};
use crate::public_model::{
    MODEL_VERSION, PublicPolicyArtifact, PublicPolicyModel, ValueValidity, predict_numeric,
    predict_semantic_playing_numeric, validate_contract,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::Path;
use tzolkin_core::Phase;
use tzolkin_core::observation::Observation;
use tzolkin_inference::policy::NumericPolicyDistribution;

pub const CHECKPOINT_SCHEMA: &str = "tzolkin-conditional-human-policy-checkpoint-v1";
pub const MODEL_SCHEMA: &str = "tzolkin-conditional-human-policy-model-v1";
pub const TRAINING_VERSION: &str = "conditional-human-policy-bc-scalar-adam-v1";
pub const INPUT_CONTRACT: &str = "conditional-basic-unlimited-3-4p-public512-playing-v1";
pub const MAX_TABLES: usize = 128;
pub const MAX_SAMPLES: usize = 65_536;
pub const MAX_CANDIDATES: usize = 262_144;
pub const COLLECTION_SCHEMA: &str = "tzolkin-conditional-human-policy-training-input-v1";
pub const SEMANTIC_PLAYING_COLLECTION_SCHEMA: &str =
    "tzolkin-conditional-human-semantic-playing-training-input-v2";
pub const SEMANTIC_PLAYING_INPUT_CONTRACT: &str =
    "conditional-basic-unlimited-3-4p-semantic-playing512-v4";
pub const SEMANTIC_PLAYING_MODEL_SCHEMA: &str =
    "tzolkin-conditional-human-semantic-playing-model-v2";
pub const SEMANTIC_PLAYING_MODEL_VERSION: &str = "conditional-semantic-playing-512x32x1-scalar-v1";
pub const SEMANTIC_PLAYING_CHECKPOINT_SCHEMA: &str =
    "tzolkin-conditional-human-semantic-playing-checkpoint-v2";
pub const SEMANTIC_PLAYING_TRAINING_VERSION: &str =
    "conditional-human-semantic-playing-bc-scalar-adam-v2";

impl ConditionalFeatureProfile {
    fn input_contract(self) -> &'static str {
        match self {
            Self::Public2 => INPUT_CONTRACT,
            Self::SemanticPlaying4 => SEMANTIC_PLAYING_INPUT_CONTRACT,
        }
    }
    fn collection_schema(self) -> &'static str {
        match self {
            Self::Public2 => COLLECTION_SCHEMA,
            Self::SemanticPlaying4 => SEMANTIC_PLAYING_COLLECTION_SCHEMA,
        }
    }
    fn model_schema(self) -> &'static str {
        match self {
            Self::Public2 => MODEL_SCHEMA,
            Self::SemanticPlaying4 => SEMANTIC_PLAYING_MODEL_SCHEMA,
        }
    }
    fn model_version(self) -> &'static str {
        match self {
            Self::Public2 => MODEL_VERSION,
            Self::SemanticPlaying4 => SEMANTIC_PLAYING_MODEL_VERSION,
        }
    }
    fn checkpoint_schema(self) -> &'static str {
        match self {
            Self::Public2 => CHECKPOINT_SCHEMA,
            Self::SemanticPlaying4 => SEMANTIC_PLAYING_CHECKPOINT_SCHEMA,
        }
    }
    fn training_version(self) -> &'static str {
        match self {
            Self::Public2 => TRAINING_VERSION,
            Self::SemanticPlaying4 => SEMANTIC_PLAYING_TRAINING_VERSION,
        }
    }
}

#[derive(Deserialize, Serialize, PartialEq)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct CollectionInput {
    schema: String,
    config: PolicyBcConfig,
    datasets: Vec<DatasetInput>,
}
#[derive(Deserialize, Serialize, PartialEq)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct DatasetInput {
    manifest_path: String,
    manifest_sha256: String,
    nominations_sha256: String,
    reservations_sha256: String,
}
/// Bounded multi-table producer loading; published float-row reports are not accepted.
pub fn train_collection_manifest(
    path: &Path,
    resume: Option<&ConditionalHumanPolicyCheckpoint>,
) -> Result<ConditionalHumanPolicyOutcome, String> {
    let (input, datasets) = load_collection(path)?;
    let outcome = train_conditional_datasets(&datasets, &input.config, resume)?;
    if read_local_json::<CollectionInput>(path)? != input {
        return Err("Conditional collection input changed before completion".into());
    }
    Ok(outcome)
}
/// Fresh BC only: neither an old checkpoint nor its optimizer can seed this profile.
pub fn train_semantic_playing_collection_manifest(
    path: &Path,
) -> Result<ConditionalHumanPolicyOutcome, String> {
    let input: CollectionInput = read_local_json(path)?;
    let (input, datasets) =
        load_collection_input_using(path, input, ConditionalFeatureProfile::SemanticPlaying4)?;
    let outcome = train_using_profile(
        &datasets,
        &input.config,
        None,
        ConditionalFeatureProfile::SemanticPlaying4,
    )?;
    if read_local_json::<CollectionInput>(path)? != input {
        return Err("Conditional collection input changed before completion".into());
    }
    Ok(outcome)
}
fn load_collection(
    path: &Path,
) -> Result<(CollectionInput, Vec<ConditionalHumanPolicyDataset>), String> {
    let input: CollectionInput = read_local_json(path)?;
    load_collection_input(path, input)
}
fn load_collection_input(
    path: &Path,
    input: CollectionInput,
) -> Result<(CollectionInput, Vec<ConditionalHumanPolicyDataset>), String> {
    load_collection_input_using(path, input, ConditionalFeatureProfile::Public2)
}
fn load_collection_input_using(
    path: &Path,
    input: CollectionInput,
    profile: ConditionalFeatureProfile,
) -> Result<(CollectionInput, Vec<ConditionalHumanPolicyDataset>), String> {
    if input.schema != profile.collection_schema()
        || input.datasets.is_empty()
        || input.datasets.len() > MAX_TABLES
    {
        return Err("Invalid bounded conditional training collection".into());
    }
    input.config.validate()?;
    let base = path.parent().unwrap_or(Path::new("."));
    let mut datasets = Vec::with_capacity(input.datasets.len());
    let mut totals = (0usize, 0usize);
    for source in &input.datasets {
        let load = match profile {
            ConditionalFeatureProfile::Public2 => ConditionalHumanPolicyDataset::load,
            ConditionalFeatureProfile::SemanticPlaying4 => {
                ConditionalHumanPolicyDataset::load_semantic_playing
            }
        };
        let dataset = load(
            &base.join(&source.manifest_path),
            &source.manifest_sha256,
            &source.nominations_sha256,
            &source.reservations_sha256,
        )?;
        totals.0 = totals
            .0
            .checked_add(dataset.counts().samples)
            .ok_or("Sample overflow")?;
        totals.1 = totals
            .1
            .checked_add(dataset.counts().candidates)
            .ok_or("Candidate overflow")?;
        if totals.0 > MAX_SAMPLES || totals.1 > MAX_CANDIDATES {
            return Err("Conditional collection sample/candidate bound exceeded".into());
        }
        datasets.push(dataset);
    }
    Ok((input, datasets))
}

fn checksum(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}
fn split_index(split: DatasetSplit) -> usize {
    match split {
        DatasetSplit::Train => 0,
        DatasetSplit::Validation => 1,
        DatasetSplit::Test => 2,
    }
}
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct ConditionalTableIdentity {
    pub table_id: String,
    pub family_id: String,
    pub split: DatasetSplit,
    pub manifest_sha256: String,
    pub sealed_report_sha256: String,
    pub samples: usize,
    pub candidates: usize,
}
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct ConditionalDatasetIdentity {
    pub task: String,
    pub input_contract: String,
    pub feature_schema: u32,
    pub feature_count: usize,
    pub rules_version: u32,
    pub catalog_hash: String,
    pub source_authenticated: bool,
    pub source_qualified: bool,
    pub verified_native_policy: bool,
    pub tables: Vec<ConditionalTableIdentity>,
    pub samples: [usize; 3],
    pub candidates: [usize; 3],
    pub fingerprint: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub label_scope: Option<String>,
}
impl ConditionalDatasetIdentity {
    fn expected_fingerprint(&self) -> Result<String, String> {
        let mut identity = self.clone();
        identity.fingerprint.clear();
        Ok(digest(&bounded_json(&identity)?))
    }
    fn validate(&self) -> Result<(), String> {
        self.validate_partition(false)
    }
    fn validate_partition(&self, validation_only: bool) -> Result<(), String> {
        self.validate_partition_using(validation_only, ConditionalFeatureProfile::Public2)
    }
    pub fn feature_profile(&self) -> Result<ConditionalFeatureProfile, String> {
        match (self.input_contract.as_str(), self.feature_schema) {
            (INPUT_CONTRACT, PUBLIC_FEATURE_SCHEMA) => Ok(ConditionalFeatureProfile::Public2),
            (SEMANTIC_PLAYING_INPUT_CONTRACT, crate::features::SEMANTIC_PLAYING_FEATURE_SCHEMA) => {
                Ok(ConditionalFeatureProfile::SemanticPlaying4)
            }
            _ => Err("Conditional input contract/feature profile mismatch".into()),
        }
    }
    fn validate_partition_using(
        &self,
        validation_only: bool,
        profile: ConditionalFeatureProfile,
    ) -> Result<(), String> {
        if self.task != TASK
            || self.input_contract != profile.input_contract()
            || self.label_scope.as_deref().is_some_and(|scope| {
                scope != FACTORIZED_LABEL_SCOPE
                    && scope != PAID_TECHNOLOGY_LABEL_SCOPE
                    && scope != EXPLICIT_MARKET_ANYWHERE_LABEL_SCOPE
                    && scope != OBSERVED_CORN_BEG_LABEL_SCOPE
                    && scope != OBSERVED_PENDING_CHOICE_LABEL_SCOPE
                    && scope != CAPPED_EXTRACTION_LABEL_SCOPE
            })
            || self.feature_schema != profile.schema()
            || (profile == ConditionalFeatureProfile::SemanticPlaying4
                && self.label_scope.as_deref() != Some(CAPPED_EXTRACTION_LABEL_SCOPE))
            || self.feature_count != FEATURE_COUNT
            || self.rules_version != tzolkin_core::public_replay::PUBLIC_RULES_VERSION
            || self.catalog_hash != tzolkin_core::public_replay::public_catalog_hash()
            || self.source_authenticated
            || self.source_qualified
            || self.verified_native_policy
            || self.tables.is_empty()
            || self.tables.len() > MAX_TABLES
            || if validation_only {
                self.samples[0] != 0 || self.candidates[0] != 0
            } else {
                self.samples[0] == 0
            }
            || self.samples[1] == 0
            || self.samples[2] != 0
            || self.candidates[2] != 0
        {
            return Err("Invalid isolated conditional Human dataset identity".into());
        }
        let mut counts = ([0usize; 3], [0usize; 3]);
        let mut previous: Option<&str> = None;
        let mut families = BTreeSet::new();
        for table in &self.tables {
            let i = split_index(table.split);
            if i == 2
                || table.samples == 0
                || table.samples > MAX_SAMPLES
                || table.candidates < table.samples
                || table.candidates > MAX_CANDIDATES
                || !checksum(&table.manifest_sha256)
                || !checksum(&table.sealed_report_sha256)
                || table_family_id(&table.table_id)? != table.family_id
                || !families.insert(&table.family_id)
                || previous.is_some_and(|p| p >= table.table_id.as_str())
            {
                return Err("Duplicate/conflicting conditional table family or counts".into());
            }
            previous = Some(&table.table_id);
            counts.0[i] = counts.0[i]
                .checked_add(table.samples)
                .ok_or("Sample overflow")?;
            counts.1[i] = counts.1[i]
                .checked_add(table.candidates)
                .ok_or("Candidate overflow")?;
        }
        if counts != (self.samples, self.candidates)
            || self.samples.iter().sum::<usize>() > MAX_SAMPLES
            || self.candidates.iter().sum::<usize>() > MAX_CANDIDATES
            || !checksum(&self.fingerprint)
            || self.expected_fingerprint()? != self.fingerprint
        {
            return Err("Conditional identity fingerprint/count mismatch".into());
        }
        Ok(())
    }
}
#[cfg(test)]
fn audit(datasets: &[ConditionalHumanPolicyDataset]) -> Result<ConditionalDatasetIdentity, String> {
    audit_partition(datasets, false)
}
#[cfg(test)]
fn audit_partition(
    datasets: &[ConditionalHumanPolicyDataset],
    validation_only: bool,
) -> Result<ConditionalDatasetIdentity, String> {
    audit_partition_using(
        datasets,
        validation_only,
        ConditionalFeatureProfile::Public2,
    )
}
fn audit_partition_using(
    datasets: &[ConditionalHumanPolicyDataset],
    validation_only: bool,
    profile: ConditionalFeatureProfile,
) -> Result<ConditionalDatasetIdentity, String> {
    if datasets.is_empty() || datasets.len() > MAX_TABLES {
        return Err("Conditional collection table bound exceeded".into());
    }
    let label_scope = datasets[0].label_scope();
    if datasets
        .iter()
        .any(|dataset| dataset.label_scope() != label_scope || dataset.feature_profile() != profile)
    {
        return Err("Conditional collections cannot mix label scopes".into());
    }
    let mut identity = ConditionalDatasetIdentity {
        task: TASK.into(),
        input_contract: profile.input_contract().into(),
        feature_schema: profile.schema(),
        feature_count: FEATURE_COUNT,
        rules_version: tzolkin_core::public_replay::PUBLIC_RULES_VERSION,
        catalog_hash: tzolkin_core::public_replay::public_catalog_hash(),
        source_authenticated: false,
        source_qualified: false,
        verified_native_policy: false,
        tables: Vec::new(),
        samples: [0; 3],
        candidates: [0; 3],
        fingerprint: String::new(),
        label_scope: label_scope.map(str::to_owned),
    };
    let mut reservations = BTreeMap::new();
    let mut sample_ids = BTreeSet::new();
    for dataset in datasets {
        // Test is neither evaluated nor used for selection by this training entry point.
        if dataset.split() == DatasetSplit::Test {
            return Err("Test tables are excluded from conditional training and selection".into());
        }
        if validation_only && dataset.split() != DatasetSplit::Validation {
            return Err("Shared evaluation requires Validation-only tables".into());
        }
        dataset.recheck_sources()?;
        for (table, split) in dataset.reservations() {
            if reservations
                .insert(table, *split)
                .is_some_and(|old| old != *split)
            {
                return Err("Conflicting source-table split reservations across datasets".into());
            }
        }
        let family = table_family_id(dataset.table_id())?;
        let i = split_index(dataset.split());
        let mut candidates = 0usize;
        for sample in dataset.samples() {
            if sample.family_id() != family
                || sample.split() != dataset.split()
                || !sample_ids.insert(sample.sample_id())
                || sample.value_target().is_some()
                || sample.candidates().is_empty()
                || sample.candidates().len() > MAX_LEGAL_ACTIONS
                || sample.chosen() >= sample.candidates().len()
            {
                return Err("Invalid/duplicate sealed conditional sample".into());
            }
            let context = &sample.candidates()[0].values_for_schema(profile.schema())?[..384];
            for row in sample.candidates() {
                let values = row.values_for_schema(profile.schema())?;
                if &values[..384] != context
                    || values
                        .iter()
                        .any(|value| !value.is_finite() || value.abs() > 1024.0)
                {
                    return Err("Conditional features exceed finite1024/context bound".into());
                }
            }
            candidates = candidates
                .checked_add(sample.candidates().len())
                .ok_or("Candidate overflow")?;
        }
        if dataset.samples().len() != dataset.counts().samples
            || candidates != dataset.counts().candidates
        {
            return Err("Conditional sealed dataset counts differ".into());
        }
        identity.samples[i] = identity.samples[i]
            .checked_add(dataset.samples().len())
            .ok_or("Sample overflow")?;
        identity.candidates[i] = identity.candidates[i]
            .checked_add(candidates)
            .ok_or("Candidate overflow")?;
        if identity.samples.iter().sum::<usize>() > MAX_SAMPLES
            || identity.candidates.iter().sum::<usize>() > MAX_CANDIDATES
        {
            return Err("Conditional collection sample/candidate bound exceeded".into());
        }
        identity.tables.push(ConditionalTableIdentity {
            table_id: dataset.table_id().into(),
            family_id: family,
            split: dataset.split(),
            manifest_sha256: dataset.manifest_sha256().into(),
            sealed_report_sha256: dataset.fingerprint()?,
            samples: dataset.samples().len(),
            candidates,
        });
    }
    identity.tables.sort_by(|a, b| a.table_id.cmp(&b.table_id));
    identity.fingerprint = identity.expected_fingerprint()?;
    identity.validate_partition_using(validation_only, profile)?;
    Ok(identity)
}
/// Conditional provenance is separate from the native public-policy artifact wire format.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct ConditionalHumanPolicyArtifact {
    pub schema: String,
    pub model_version: String,
    pub task: String,
    pub input_contract: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub feature_profile: Option<ConditionalFeatureProfile>,
    pub dataset_fingerprint: String,
    pub value_validity: ValueValidity,
    pub model: PublicPolicyModel,
    pub checksum: String,
}
impl ConditionalHumanPolicyArtifact {
    fn expected_checksum(&self) -> Result<String, String> {
        let mut artifact = self.clone();
        artifact.checksum.clear();
        Ok(digest(&bounded_json(&artifact)?))
    }
    fn reseal(&mut self) -> Result<(), String> {
        self.reseal_using(ConditionalFeatureProfile::Public2)
    }
    fn reseal_using(&mut self, profile: ConditionalFeatureProfile) -> Result<(), String> {
        self.checksum = self.expected_checksum()?;
        self.validate_using(profile)
    }
    pub fn validate(&self) -> Result<(), String> {
        self.validate_using(ConditionalFeatureProfile::Public2)
    }
    pub fn validate_semantic_playing(&self) -> Result<(), String> {
        self.validate_using(ConditionalFeatureProfile::SemanticPlaying4)
    }
    fn validate_using(&self, profile: ConditionalFeatureProfile) -> Result<(), String> {
        self.model.validate()?;
        if self.schema != profile.model_schema()
            || self.model_version != profile.model_version()
            || self.task != TASK
            || self.input_contract != profile.input_contract()
            || self.feature_profile
                != (profile == ConditionalFeatureProfile::SemanticPlaying4).then_some(profile)
            || !checksum(&self.dataset_fingerprint)
            || self.value_validity != ValueValidity::UnavailablePolicyOnly
            || !checksum(&self.checksum)
            || self.checksum != self.expected_checksum()?
        {
            return Err("Invalid conditional policy artifact".into());
        }
        Ok(())
    }
}
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct ConditionalHumanPolicyCheckpoint {
    pub schema: String,
    pub training_version: String,
    pub dataset: ConditionalDatasetIdentity,
    pub config: PolicyBcConfig,
    pub completed_epochs: usize,
    pub random_state: u64,
    pub model: ConditionalHumanPolicyArtifact,
    pub optimizer: PolicyAdamState,
    pub metrics: PolicyTrainingMetrics,
    pub checksum: String,
}
impl ConditionalHumanPolicyCheckpoint {
    fn expected_checksum(&self) -> Result<String, String> {
        let mut checkpoint = self.clone();
        checkpoint.checksum.clear();
        Ok(digest(&bounded_json(&checkpoint)?))
    }
    pub fn validate(&self) -> Result<(), String> {
        self.validate_using(ConditionalFeatureProfile::Public2)
    }
    pub fn validate_semantic_playing(&self) -> Result<(), String> {
        self.validate_using(ConditionalFeatureProfile::SemanticPlaying4)
    }
    fn validate_using(&self, profile: ConditionalFeatureProfile) -> Result<(), String> {
        match profile {
            ConditionalFeatureProfile::Public2 => self.dataset.validate()?,
            ConditionalFeatureProfile::SemanticPlaying4 => {
                self.dataset.validate_partition_using(false, profile)?
            }
        }
        self.config.validate()?;
        self.model.validate_using(profile)?;
        self.optimizer.validate()?;
        if self.schema != profile.checkpoint_schema()
            || self.training_version != profile.training_version()
            || self.completed_epochs == 0
            || self.completed_epochs != self.config.epochs
            || self.metrics.completed_epochs != self.completed_epochs
            || self.metrics.strength_measured
            || self.model.dataset_fingerprint != self.dataset.fingerprint
            || self.optimizer.step
                != self.dataset.samples[0].div_ceil(self.config.batch_size) as u64
                    * self.completed_epochs as u64
        {
            return Err("Conditional checkpoint epoch/source/optimizer mismatch".into());
        }
        for report in [&self.metrics.initial_train, &self.metrics.final_train] {
            report.validate(self.dataset.samples[0], self.dataset.candidates[0])?;
        }
        for report in [
            &self.metrics.initial_validation,
            &self.metrics.final_validation,
        ] {
            report.validate(self.dataset.samples[1], self.dataset.candidates[1])?;
        }
        if !checksum(&self.checksum) || self.checksum != self.expected_checksum()? {
            return Err("Conditional checkpoint checksum mismatch".into());
        }
        Ok(())
    }
    pub fn load(path: &Path) -> Result<Self, String> {
        let checkpoint: Self = read_local_json(path)?;
        checkpoint.validate()?;
        Ok(checkpoint)
    }
    pub fn load_semantic_playing(path: &Path) -> Result<Self, String> {
        let checkpoint: Self = read_local_json(path)?;
        checkpoint.validate_semantic_playing()?;
        Ok(checkpoint)
    }
}

/// Immutable conditional Human policy with source-checked table identity.
/// Preparation preserves the scalar 512→32→1 parameters; it grants no native
/// policy qualification, RL ownership, value supervision or measured strength.
pub struct PreparedConditionalHumanPolicy {
    checkpoint: Box<ConditionalHumanPolicyCheckpoint>,
    reservations: BTreeMap<String, DatasetSplit>,
    kernel: ResolvedKernel,
    loaded_sources: Option<Vec<ContentPin>>,
    loaded_boundary_sources: Option<Vec<ContentPin>>,
    feature_profile: ConditionalFeatureProfile,
    source_labels_by_split: [String; 3],
}
impl PreparedConditionalHumanPolicy {
    pub fn load(checkpoint_path: &Path, collection_path: &Path) -> Result<Self, String> {
        Self::load_using(
            checkpoint_path,
            collection_path,
            ConditionalFeatureProfile::Public2,
        )
    }
    pub fn load_semantic_playing(
        checkpoint_path: &Path,
        collection_path: &Path,
    ) -> Result<Self, String> {
        Self::load_using(
            checkpoint_path,
            collection_path,
            ConditionalFeatureProfile::SemanticPlaying4,
        )
    }
    fn load_using(
        checkpoint_path: &Path,
        collection_path: &Path,
        profile: ConditionalFeatureProfile,
    ) -> Result<Self, String> {
        let (checkpoint, checkpoint_pin): (ConditionalHumanPolicyCheckpoint, _) =
            read_local_json_pinned(checkpoint_path)?;
        checkpoint.validate_using(profile)?;
        let (input, collection_pin) = read_local_json_pinned(collection_path)?;
        let (input, datasets) = load_collection_input_using(collection_path, input, profile)?;
        if input.config != checkpoint.config {
            return Err("Conditional prepared collection/checkpoint config mismatch".into());
        }
        let mut prepared = Self::new_using(checkpoint, &datasets, profile)?;
        let mut boundaries = vec![checkpoint_pin.clone(), collection_pin.clone()];
        let mut pins = vec![checkpoint_pin, collection_pin];
        for dataset in &datasets {
            let sources = dataset.loaded_source_pins()?;
            boundaries.push(
                sources
                    .first()
                    .ok_or("Missing loaded manifest binding")?
                    .clone(),
            );
            pins.extend(sources);
        }
        prepared.loaded_sources = Some(seal_loaded_sources(pins)?);
        prepared.loaded_boundary_sources = Some(seal_loaded_sources(boundaries)?);
        Ok(prepared)
    }

    pub fn new(
        checkpoint: ConditionalHumanPolicyCheckpoint,
        datasets: &[ConditionalHumanPolicyDataset],
    ) -> Result<Self, String> {
        Self::new_using(checkpoint, datasets, ConditionalFeatureProfile::Public2)
    }
    fn new_using(
        checkpoint: ConditionalHumanPolicyCheckpoint,
        datasets: &[ConditionalHumanPolicyDataset],
        profile: ConditionalFeatureProfile,
    ) -> Result<Self, String> {
        checkpoint.validate_using(profile)?;
        let identity = audit_partition_using(datasets, false, profile)?;
        if checkpoint.dataset != identity {
            return Err("Conditional prepared checkpoint/source identity mismatch".into());
        }
        let mut reservations = BTreeMap::new();
        for dataset in datasets {
            for (table, split) in dataset.reservations() {
                reservations.insert(table.clone(), *split);
            }
            for sample in dataset.samples() {
                for row in sample.candidates() {
                    match profile {
                        ConditionalFeatureProfile::Public2 => {
                            PublicPolicyModel::validated_values(row)?;
                        }
                        ConditionalFeatureProfile::SemanticPlaying4 => {
                            tzolkin_inference::policy::validated_semantic_playing_values(row)?;
                        }
                    }
                }
            }
        }
        let mut ordered = datasets.iter().collect::<Vec<_>>();
        ordered.sort_by(|a, b| a.table_id().cmp(b.table_id()));
        let train = ordered
            .iter()
            .filter(|d| d.split() == DatasetSplit::Train)
            .flat_map(|d| d.samples())
            .collect::<Vec<_>>();
        let validation = ordered
            .iter()
            .filter(|d| d.split() == DatasetSplit::Validation)
            .flat_map(|d| d.samples())
            .collect::<Vec<_>>();
        let source_labels_by_split = [
            ordered_source_labels(&train)?,
            ordered_source_labels(&validation)?,
            ordered_source_labels(&[])?,
        ];
        let initial = PublicPolicyArtifact::new(checkpoint.config.seed)?.model;
        let mut scratch = Scratch::default();
        if evaluate_using(&train, &initial, &mut scratch, profile)?
            != checkpoint.metrics.initial_train
            || evaluate_using(&validation, &initial, &mut scratch, profile)?
                != checkpoint.metrics.initial_validation
            || evaluate_using(&train, &checkpoint.model.model, &mut scratch, profile)?
                != checkpoint.metrics.final_train
            || evaluate_using(&validation, &checkpoint.model.model, &mut scratch, profile)?
                != checkpoint.metrics.final_validation
        {
            return Err("Conditional prepared metric/model/source mismatch".into());
        }
        if audit_partition_using(datasets, false, profile)? != identity {
            return Err("Conditional sources changed during preparation".into());
        }
        Ok(Self {
            checkpoint: Box::new(checkpoint),
            reservations,
            kernel: Kernel::Scalar.resolve()?,
            loaded_sources: None,
            loaded_boundary_sources: None,
            feature_profile: profile,
            source_labels_by_split,
        })
    }
    pub(crate) fn loaded_sources(&self) -> Result<&[ContentPin], String> {
        if self.feature_profile != ConditionalFeatureProfile::Public2 {
            return Err("Legacy conditional RL origin requires public2 preparation".into());
        }
        self.loaded_sources
            .as_deref()
            .ok_or_else(|| "Conditional RL origin requires successful source load".into())
    }
    pub(crate) fn loaded_boundary_sources(&self) -> Result<&[ContentPin], String> {
        if self.feature_profile != ConditionalFeatureProfile::Public2 {
            return Err(
                "Legacy conditional journal/deployment requires public2 preparation".into(),
            );
        }
        self.loaded_boundary_sources
            .as_deref()
            .ok_or_else(|| "Conditional boundary pins require successful source load".into())
    }

    /// Dedicated schema-4 raw custody; this does not open legacy RL getters.
    pub(crate) fn loaded_semantic_playing_sources(&self) -> Result<&[ContentPin], String> {
        if self.feature_profile != ConditionalFeatureProfile::SemanticPlaying4 {
            return Err("Semantic Playing origin requires schema-4 preparation".into());
        }
        self.loaded_sources
            .as_deref()
            .filter(|pins| !pins.is_empty())
            .ok_or_else(|| "Semantic Playing origin requires successful source load".into())
    }

    /// Recheck the immutable pins captured by a successful original source load.
    /// This is content consistency, not source authentication or native admission.
    pub fn recheck_sources(&self) -> Result<(), String> {
        recheck_prepared_origin(self)
    }

    pub fn model(&self) -> &ConditionalHumanPolicyArtifact {
        &self.checkpoint.model
    }
    pub fn identity(&self) -> &ConditionalDatasetIdentity {
        &self.checkpoint.dataset
    }
    pub fn checkpoint_checksum(&self) -> &str {
        &self.checkpoint.checksum
    }
    /// Includes declared held-out reservations as well as the loaded tables.
    pub fn reservations(&self) -> &BTreeMap<String, DatasetSplit> {
        &self.reservations
    }
    pub fn backend(&self) -> &'static str {
        self.kernel.backend()
    }

    /// Score every supplied legal row in its original order. Only the producer's
    /// observed Playing Place/Remove/FirstPlayer labels were supervised. Opt-in
    /// v2 also factorizes a joint Remove/CurrentGear UseAction caption; these are
    /// two App labels, not two independent observed Human choices. Scoring other
    /// legal actions does not establish observed Human labels for them.
    /// The core observation/key is checked, without authenticating its source.
    pub fn distribution(
        &self,
        observation: &Observation,
    ) -> Result<NumericPolicyDistribution, String> {
        if self.feature_profile != ConditionalFeatureProfile::Public2 {
            return Err(
                "Legacy conditional distribution cannot score semantic Playing weights".into(),
            );
        }
        self.distribution_using(observation, ConditionalFeatureProfile::Public2)
    }
    /// Explicit input layout only; Arena/deployment admission is a separate boundary.
    pub fn distribution_semantic_playing(
        &self,
        observation: &Observation,
    ) -> Result<NumericPolicyDistribution, String> {
        if self.feature_profile != ConditionalFeatureProfile::SemanticPlaying4 {
            return Err("Semantic Playing distribution requires its own loaded profile".into());
        }
        self.distribution_using(observation, ConditionalFeatureProfile::SemanticPlaying4)
    }
    fn distribution_using(
        &self,
        observation: &Observation,
        profile: ConditionalFeatureProfile,
    ) -> Result<NumericPolicyDistribution, String> {
        validate_contract(observation)?;
        if observation.phase != Phase::Playing {
            return Err("Conditional Human policy requires Playing observations".into());
        }
        let encoder = profile.encoder(observation)?;
        let rows = (0..observation.legal_actions.len())
            .map(|index| encoder.encode_legal_tagged(index))
            .collect::<Result<Vec<_>, _>>()?;
        match profile {
            ConditionalFeatureProfile::Public2 => {
                predict_numeric(&self.checkpoint.model.model, self.kernel, &rows)
            }
            ConditionalFeatureProfile::SemanticPlaying4 => {
                predict_semantic_playing_numeric(&self.checkpoint.model.model, self.kernel, &rows)
            }
        }
    }
}
fn seal_loaded_sources(pins: Vec<ContentPin>) -> Result<Vec<ContentPin>, String> {
    // Each dataset has at most 32 evidence leaves, five direct leaves and one manifest.
    if pins.len() > MAX_TABLES * 38 + 2 {
        return Err("Conditional origin source count bound".into());
    }
    let mut unique = BTreeMap::<String, ContentPin>::new();
    for pin in pins {
        crate::human_policy_dataset::recheck_absolute_pin(&pin)?;
        let canonical = fs::canonicalize(&pin.path).map_err(|e| e.to_string())?;
        let key = canonical.to_string_lossy().into_owned();
        #[cfg(windows)]
        let key = key.to_lowercase();
        if let Some(previous) = unique.get(&key) {
            if previous.bytes != pin.bytes || previous.sha256 != pin.sha256 {
                return Err("Conflicting conditional origin pins for one physical source".into());
            }
        } else {
            unique.insert(key, pin);
        }
    }
    Ok(unique.into_values().collect())
}

pub const VALIDATION_COMPARISON_SCHEMA: &str = "tzolkin-conditional-human-shared-validation-v1";
pub const SEMANTIC_PLAYING_VALIDATION_SCHEMA: &str =
    "tzolkin-conditional-human-semantic-playing-validation-v2";

fn ordered_source_labels(samples: &[&HumanPolicySample]) -> Result<String, String> {
    let mut hash = Sha256::new();
    hash.update(b"tzolkin-conditional-source-label-order-v1\0");
    for sample in samples {
        hash.update(sample.source_label_digest()?.as_bytes());
    }
    Ok(format!("{:x}", hash.finalize()))
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct ValidationActorReport {
    checkpoint_checksum: String,
    model_checksum: String,
    origin: ConditionalDatasetIdentity,
    config: PolicyBcConfig,
    backend: &'static str,
    all_rows: PolicyLossReport,
    multi_candidate_rows: Option<PolicyLossReport>,
}

/// Policy-only agreement on one shared sealed Validation partition. This is
/// neither native admission nor a win-rate, value or strength measurement.
#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ConditionalValidationComparison {
    schema: &'static str,
    task: &'static str,
    input_contract: &'static str,
    validation_collection: Option<ContentPin>,
    validation_fingerprint: String,
    validation_tables: Vec<ConditionalTableIdentity>,
    validation_label_scope: Option<String>,
    ordered_samples_sha256: String,
    samples: usize,
    candidates: usize,
    single_candidate_samples: usize,
    action_counts: BTreeMap<String, usize>,
    candidate_count_samples: BTreeMap<usize, usize>,
    actors: [ValidationActorReport; 2],
    source_authenticated: bool,
    source_qualified: bool,
    verified_native_policy: bool,
    value_validity: ValueValidity,
    strength_measured: bool,
}
impl ConditionalValidationComparison {
    pub fn samples(&self) -> usize {
        self.samples
    }
    pub fn candidates(&self) -> usize {
        self.candidates
    }
    pub fn save_new_directory(&self, directory: &Path) -> Result<(), String> {
        validate_new_output_directory(directory)?;
        bounded_json(self)?;
        fs::create_dir(directory).map_err(|e| e.to_string())?;
        publish(&directory.join("report.json"), self)
    }
}

fn recheck_prepared_origin(policy: &PreparedConditionalHumanPolicy) -> Result<(), String> {
    let pins = policy
        .loaded_sources
        .as_deref()
        .ok_or("Conditional source recheck requires successful source load")?;
    for pin in pins {
        crate::human_policy_dataset::recheck_absolute_pin(pin)?;
    }
    Ok(())
}

/// Each actor must first load its own original checkpoint and collection.
/// The evaluation collection is independently loaded and must be Validation
/// reserved by both origins, with no overlap with either origin's Train families.
pub fn compare_validation_collection(
    actors: [&PreparedConditionalHumanPolicy; 2],
    validation_collection: &Path,
) -> Result<ConditionalValidationComparison, String> {
    if actors
        .iter()
        .any(|actor| actor.feature_profile != ConditionalFeatureProfile::Public2)
    {
        return Err("Legacy shared Validation requires public2 origins".into());
    }
    for actor in actors {
        recheck_prepared_origin(actor)?;
    }
    let (input, collection_pin): (CollectionInput, _) =
        read_local_json_pinned(validation_collection)?;
    let (input, datasets) = load_collection_input(validation_collection, input)?;
    let mut comparison = compare_validation_datasets(actors, &datasets)?;
    if read_local_json::<CollectionInput>(validation_collection)? != input {
        return Err("Conditional Validation collection changed during evaluation".into());
    }
    crate::human_policy_dataset::recheck_absolute_pin(&collection_pin)?;
    for dataset in &datasets {
        dataset.recheck_sources()?;
    }
    for actor in actors {
        recheck_prepared_origin(actor)?;
    }
    comparison.validation_collection = Some(collection_pin);
    Ok(comparison)
}

fn compare_validation_datasets(
    actors: [&PreparedConditionalHumanPolicy; 2],
    datasets: &[ConditionalHumanPolicyDataset],
) -> Result<ConditionalValidationComparison, String> {
    compare_validation_datasets_using(actors, datasets, ConditionalFeatureProfile::Public2)
}
fn validate_shared_reservations(
    actors: [&PreparedConditionalHumanPolicy; 2],
    datasets: &[ConditionalHumanPolicyDataset],
) -> Result<(), String> {
    let mut reservations = BTreeMap::new();
    let mut train_families = BTreeSet::new();
    for actor in actors {
        for (table, split) in actor.reservations() {
            if reservations
                .insert(table, *split)
                .is_some_and(|previous| previous != *split)
            {
                return Err("Conditional actor origins disagree on split reservations".into());
            }
            if *split == DatasetSplit::Train {
                train_families.insert(table_family_id(table)?);
            }
        }
        train_families.extend(
            actor
                .identity()
                .tables
                .iter()
                .filter(|table| table.split == DatasetSplit::Train)
                .map(|table| table.family_id.clone()),
        );
    }
    for dataset in datasets {
        for actor in actors {
            if actor.reservations().get(dataset.table_id()) != Some(&DatasetSplit::Validation) {
                return Err("Shared Validation table is not reserved by both actor origins".into());
            }
        }
        if train_families.contains(&table_family_id(dataset.table_id())?) {
            return Err("Shared Validation overlaps an actor's Train family".into());
        }
        for (table, split) in dataset.reservations() {
            if reservations
                .insert(table, *split)
                .is_some_and(|previous| previous != *split)
            {
                return Err("Shared Validation conflicts with origin split reservations".into());
            }
        }
    }
    Ok(())
}
fn compare_validation_datasets_using(
    actors: [&PreparedConditionalHumanPolicy; 2],
    datasets: &[ConditionalHumanPolicyDataset],
    profile: ConditionalFeatureProfile,
) -> Result<ConditionalValidationComparison, String> {
    if actors.iter().any(|actor| actor.feature_profile != profile) {
        return Err("Validation features must match each actor's loaded profile".into());
    }
    let identity = audit_partition_using(datasets, true, profile)?;
    validate_shared_reservations(actors, datasets)?;
    let mut ordered = datasets.iter().collect::<Vec<_>>();
    ordered.sort_by(|a, b| a.table_id().cmp(b.table_id()));
    let samples = ordered
        .iter()
        .flat_map(|dataset| dataset.samples())
        .collect::<Vec<_>>();
    let multiple = samples
        .iter()
        .copied()
        .filter(|sample| sample.candidates().len() > 1)
        .collect::<Vec<_>>();
    let mut hasher = Sha256::new();
    let mut action_counts = BTreeMap::new();
    let mut candidate_count_samples = BTreeMap::new();
    match profile {
        ConditionalFeatureProfile::Public2 => {
            hasher.update(b"tzolkin-conditional-human-shared-validation-ordered-public512-v1\0")
        }
        ConditionalFeatureProfile::SemanticPlaying4 => hasher
            .update(b"tzolkin-conditional-human-shared-validation-ordered-semantic-playing4-v1\0"),
    }
    for sample in &samples {
        let action = serde_json::to_value(sample.chosen_action()?).map_err(|e| e.to_string())?;
        let kind = action["type"]
            .as_str()
            .ok_or("Missing conditional chosen action type")?;
        *action_counts.entry(kind.to_owned()).or_default() += 1;
        *candidate_count_samples
            .entry(sample.candidates().len())
            .or_default() += 1;
        let id = sample.sample_id().as_bytes();
        hasher.update((id.len() as u64).to_le_bytes());
        hasher.update(id);
        hasher.update((sample.chosen() as u64).to_le_bytes());
        hasher.update((sample.candidates().len() as u64).to_le_bytes());
        for row in sample.candidates() {
            for value in row.values_for_schema(profile.schema())? {
                hasher.update(value.to_bits().to_le_bytes());
            }
        }
    }
    let actor_reports = actors
        .map(|actor| {
            let mut scratch = Scratch::default();
            Ok(ValidationActorReport {
                checkpoint_checksum: actor.checkpoint_checksum().into(),
                model_checksum: actor.model().checksum.clone(),
                origin: actor.identity().clone(),
                config: actor.checkpoint.config.clone(),
                backend: actor.backend(),
                all_rows: evaluate_using(&samples, &actor.model().model, &mut scratch, profile)?,
                multi_candidate_rows: if multiple.is_empty() {
                    None
                } else {
                    Some(evaluate_using(
                        &multiple,
                        &actor.model().model,
                        &mut scratch,
                        profile,
                    )?)
                },
            })
        })
        .into_iter()
        .collect::<Result<Vec<_>, String>>()?
        .try_into()
        .map_err(|_| "Shared Validation actor pair shape")?;
    if audit_partition_using(datasets, true, profile)? != identity {
        return Err("Shared Validation sources changed during evaluation".into());
    }
    Ok(ConditionalValidationComparison {
        schema: match profile {
            ConditionalFeatureProfile::Public2 => VALIDATION_COMPARISON_SCHEMA,
            ConditionalFeatureProfile::SemanticPlaying4 => SEMANTIC_PLAYING_VALIDATION_SCHEMA,
        },
        task: TASK,
        input_contract: profile.input_contract(),
        validation_collection: None,
        validation_fingerprint: identity.fingerprint,
        validation_tables: identity.tables,
        validation_label_scope: identity.label_scope,
        ordered_samples_sha256: format!("{:x}", hasher.finalize()),
        samples: samples.len(),
        candidates: identity.candidates[1],
        single_candidate_samples: samples.len() - multiple.len(),
        action_counts,
        candidate_count_samples,
        actors: actor_reports,
        source_authenticated: false,
        source_qualified: false,
        verified_native_policy: false,
        value_validity: ValueValidity::UnavailablePolicyOnly,
        strength_measured: false,
    })
}

/// Same original Train labels and same Validation source decisions, independently
/// re-encoded for each profile. Numerical feature digests are deliberately separate.
#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct SemanticPlayingValidationComparison {
    schema: &'static str,
    training_selection: &'static str,
    common_train_source_labels_sha256: String,
    ordered_validation_source_labels_sha256: String,
    validation_collections: [ContentPin; 2],
    validation_fingerprints: [String; 2],
    ordered_features_sha256: [String; 2],
    validation_tables: [Vec<ConditionalTableIdentity>; 2],
    samples: usize,
    candidates: usize,
    single_candidate_samples: usize,
    action_counts: BTreeMap<String, usize>,
    candidate_count_samples: BTreeMap<usize, usize>,
    semantic_feature_eligibility: Vec<crate::human_policy_dataset::FeatureEligibilityComparison>,
    actors: [ValidationActorReport; 2],
    source_authenticated: bool,
    source_qualified: bool,
    verified_native_policy: bool,
    value_validity: ValueValidity,
    strength_measured: bool,
}
impl SemanticPlayingValidationComparison {
    pub fn samples(&self) -> usize {
        self.samples
    }
    pub fn candidates(&self) -> usize {
        self.candidates
    }
    pub fn save_new_directory(&self, directory: &Path) -> Result<(), String> {
        validate_new_output_directory(directory)?;
        bounded_json(self)?;
        fs::create_dir(directory).map_err(|e| e.to_string())?;
        publish(&directory.join("report.json"), self)
    }
}

pub fn compare_semantic_playing_validation_collections(
    public: &PreparedConditionalHumanPolicy,
    semantic: &PreparedConditionalHumanPolicy,
    public_validation: &Path,
    semantic_validation: &Path,
) -> Result<SemanticPlayingValidationComparison, String> {
    if public.feature_profile != ConditionalFeatureProfile::Public2
        || semantic.feature_profile != ConditionalFeatureProfile::SemanticPlaying4
        || public.source_labels_by_split[0] != semantic.source_labels_by_split[0]
    {
        return Err("Feature comparison requires the same original Train labels and distinct explicit profiles".into());
    }
    let actors = [public, semantic];
    for actor in actors {
        recheck_prepared_origin(actor)?;
    }
    let (left_input, left_pin): (CollectionInput, _) = read_local_json_pinned(public_validation)?;
    let (right_input, right_pin): (CollectionInput, _) =
        read_local_json_pinned(semantic_validation)?;
    let (left_input, left) = load_collection_input_using(
        public_validation,
        left_input,
        ConditionalFeatureProfile::Public2,
    )?;
    let (right_input, right) = load_collection_input_using(
        semantic_validation,
        right_input,
        ConditionalFeatureProfile::SemanticPlaying4,
    )?;
    validate_shared_reservations(actors, &left)?;
    validate_shared_reservations(actors, &right)?;
    let labels = |datasets: &[ConditionalHumanPolicyDataset]| {
        let mut ordered = datasets.iter().collect::<Vec<_>>();
        ordered.sort_by(|a, b| a.table_id().cmp(b.table_id()));
        let rows = ordered
            .iter()
            .flat_map(|dataset| dataset.samples())
            .collect::<Vec<_>>();
        ordered_source_labels(&rows)
    };
    let source_labels = labels(&left)?;
    if labels(&right)? != source_labels {
        return Err("Validation feature profiles differ in source labels/legal order".into());
    }
    let l = compare_validation_datasets_using(
        [public, public],
        &left,
        ConditionalFeatureProfile::Public2,
    )?;
    let r = compare_validation_datasets_using(
        [semantic, semantic],
        &right,
        ConditionalFeatureProfile::SemanticPlaying4,
    )?;
    if l.samples != r.samples
        || l.candidates != r.candidates
        || l.single_candidate_samples != r.single_candidate_samples
        || l.action_counts != r.action_counts
        || l.candidate_count_samples != r.candidate_count_samples
    {
        return Err("Shared source decisions differ between feature profiles".into());
    }
    let eligibility = right
        .iter()
        .map(|dataset| {
            dataset
                .feature_eligibility()
                .cloned()
                .ok_or_else(|| "Missing semantic common/full eligibility comparison".to_string())
        })
        .collect::<Result<Vec<_>, _>>()?;
    if read_local_json::<CollectionInput>(public_validation)? != left_input
        || read_local_json::<CollectionInput>(semantic_validation)? != right_input
    {
        return Err("Profile Validation collections changed during evaluation".into());
    }
    for pin in [&left_pin, &right_pin] {
        crate::human_policy_dataset::recheck_absolute_pin(pin)?;
    }
    for dataset in left.iter().chain(&right) {
        dataset.recheck_sources()?;
    }
    for actor in actors {
        recheck_prepared_origin(actor)?;
    }
    let [left_actor, _] = l.actors;
    let [right_actor, _] = r.actors;
    Ok(SemanticPlayingValidationComparison {
        schema: SEMANTIC_PLAYING_VALIDATION_SCHEMA,
        training_selection: "schema2EligibleCommonSubset",
        common_train_source_labels_sha256: public.source_labels_by_split[0].clone(),
        ordered_validation_source_labels_sha256: source_labels,
        validation_collections: [left_pin, right_pin],
        validation_fingerprints: [l.validation_fingerprint, r.validation_fingerprint],
        ordered_features_sha256: [l.ordered_samples_sha256, r.ordered_samples_sha256],
        validation_tables: [l.validation_tables, r.validation_tables],
        samples: l.samples,
        candidates: l.candidates,
        single_candidate_samples: l.single_candidate_samples,
        action_counts: l.action_counts,
        candidate_count_samples: l.candidate_count_samples,
        semantic_feature_eligibility: eligibility,
        actors: [left_actor, right_actor],
        source_authenticated: false,
        source_qualified: false,
        verified_native_policy: false,
        value_validity: ValueValidity::UnavailablePolicyOnly,
        strength_measured: false,
    })
}
#[derive(Clone, Debug, Serialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct ConditionalHumanPolicyOutcome {
    pub model: ConditionalHumanPolicyArtifact,
    pub checkpoint: ConditionalHumanPolicyCheckpoint,
    pub metrics: PolicyTrainingMetrics,
}
impl ConditionalHumanPolicyOutcome {
    pub fn save_new_directory(&self, directory: &Path) -> Result<(), String> {
        self.save_using(directory, ConditionalFeatureProfile::Public2)
    }
    pub fn save_new_semantic_playing_directory(&self, directory: &Path) -> Result<(), String> {
        self.save_using(directory, ConditionalFeatureProfile::SemanticPlaying4)
    }
    fn save_using(
        &self,
        directory: &Path,
        profile: ConditionalFeatureProfile,
    ) -> Result<(), String> {
        self.checkpoint.validate_using(profile)?;
        if self.model != self.checkpoint.model || self.metrics != self.checkpoint.metrics {
            return Err("Inconsistent conditional training output".into());
        }
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
fn conditional_policy_loss(
    model: &PublicPolicyModel,
    sample: &HumanPolicySample,
    scratch: &mut Scratch,
    gradient: Option<&mut [f32]>,
    profile: ConditionalFeatureProfile,
) -> Result<(f64, bool), String> {
    match profile {
        ConditionalFeatureProfile::Public2 => policy_loss(
            model,
            sample.candidates(),
            sample.chosen(),
            scratch,
            gradient,
        ),
        ConditionalFeatureProfile::SemanticPlaying4 => semantic_playing_policy_loss(
            model,
            sample.candidates(),
            sample.chosen(),
            scratch,
            gradient,
        ),
    }
}
fn evaluate_using(
    samples: &[&HumanPolicySample],
    model: &PublicPolicyModel,
    scratch: &mut Scratch,
    profile: ConditionalFeatureProfile,
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
    for sample in samples {
        let (loss, correct) = conditional_policy_loss(model, sample, scratch, None, profile)?;
        report.samples += 1;
        report.candidates += sample.candidates().len();
        report.policy_loss += loss;
        report.correct_top1 += usize::from(correct);
    }
    if report.samples == 0 {
        return Err("Empty conditional evaluation partition".into());
    }
    report.policy_loss /= report.samples as f64;
    report.validate(samples.len(), report.candidates)?;
    Ok(report)
}
/// Train and Validation must come from distinct sealed tables. Test is rejected.
/// Validation losses are for external selection; this call has no Test or strength evaluation.
pub fn train_conditional_datasets(
    datasets: &[ConditionalHumanPolicyDataset],
    config: &PolicyBcConfig,
    resume: Option<&ConditionalHumanPolicyCheckpoint>,
) -> Result<ConditionalHumanPolicyOutcome, String> {
    train_using_profile(datasets, config, resume, ConditionalFeatureProfile::Public2)
}
fn train_using_profile(
    datasets: &[ConditionalHumanPolicyDataset],
    config: &PolicyBcConfig,
    resume: Option<&ConditionalHumanPolicyCheckpoint>,
    profile: ConditionalFeatureProfile,
) -> Result<ConditionalHumanPolicyOutcome, String> {
    config.validate()?;
    let identity = audit_partition_using(datasets, false, profile)?;
    let completed = resume.map_or(0, |checkpoint| checkpoint.completed_epochs);
    if let Some(previous) = resume {
        previous.validate_using(profile)?;
        if previous.dataset != identity
            || !config.compatible(&previous.config)
            || config.epochs < completed
        {
            return Err("Conditional resume dataset/config mismatch".into());
        }
    }
    let remaining = config.epochs - completed;
    let evaluations = identity.candidates[0]
        .checked_mul(remaining + 2)
        .and_then(|n| {
            identity.candidates[1]
                .checked_mul(2)
                .and_then(|v| n.checked_add(v))
        })
        .ok_or("Conditional evaluation overflow")?;
    if evaluations > MAX_CANDIDATE_EVALUATIONS
        || identity.samples[0].div_ceil(config.batch_size) as u64 * config.epochs as u64
            > MAX_UPDATES
    {
        return Err("Conditional candidate/update budget exceeded".into());
    }
    let mut ordered = datasets.iter().collect::<Vec<_>>();
    ordered.sort_by(|a, b| a.table_id().cmp(b.table_id()));
    let train = ordered
        .iter()
        .filter(|d| d.split() == DatasetSplit::Train)
        .flat_map(|d| d.samples())
        .collect::<Vec<_>>();
    let validation = ordered
        .iter()
        .filter(|d| d.split() == DatasetSplit::Validation)
        .flat_map(|d| d.samples())
        .collect::<Vec<_>>();
    let mut model = resume.map_or_else(
        || PublicPolicyArtifact::new(config.seed).map(|a| a.model),
        |p| Ok(p.model.model.clone()),
    )?;
    let mut optimizer = resume.map_or_else(PolicyAdamState::new, |p| p.optimizer.clone());
    let mut random = Random {
        state: resume.map_or(config.seed ^ 0xd1b54a32d192ed03, |p| p.random_state),
    };
    let mut scratch = Scratch::default();
    let current_train = evaluate_using(&train, &model, &mut scratch, profile)?;
    let current_validation = evaluate_using(&validation, &model, &mut scratch, profile)?;
    let mut metrics = if let Some(previous) = resume {
        if current_train != previous.metrics.final_train
            || current_validation != previous.metrics.final_validation
        {
            return Err("Conditional resume metric/model/source mismatch".into());
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
    let mut gradient = vec![0.0; crate::public_model::PARAMETER_COUNT];
    for _ in completed..config.epochs {
        let mut order = (0..train.len()).collect::<Vec<_>>();
        for i in (1..order.len()).rev() {
            let j = random.index(i + 1);
            order.swap(i, j);
        }
        let mut batch = 0;
        let mut consume = |sample: &HumanPolicySample| -> Result<(), String> {
            conditional_policy_loss(&model, sample, &mut scratch, Some(&mut gradient), profile)?;
            batch += 1;
            if batch == config.batch_size {
                optimizer.update(&mut model, &mut gradient, batch, config.learning_rate)?;
                batch = 0;
            }
            Ok(())
        };
        for index in order {
            consume(train[index])?;
        }
        if batch > 0 {
            optimizer.update(&mut model, &mut gradient, batch, config.learning_rate)?;
        }
    }
    metrics.completed_epochs = config.epochs;
    metrics.final_train = evaluate_using(&train, &model, &mut scratch, profile)?;
    metrics.final_validation = evaluate_using(&validation, &model, &mut scratch, profile)?;
    if audit_partition_using(datasets, false, profile)? != identity {
        return Err("Conditional source bytes changed before completion".into());
    }
    let mut artifact = ConditionalHumanPolicyArtifact {
        schema: profile.model_schema().into(),
        model_version: profile.model_version().into(),
        task: TASK.into(),
        input_contract: profile.input_contract().into(),
        feature_profile: (profile == ConditionalFeatureProfile::SemanticPlaying4)
            .then_some(profile),
        dataset_fingerprint: identity.fingerprint.clone(),
        value_validity: ValueValidity::UnavailablePolicyOnly,
        model,
        checksum: String::new(),
    };
    match profile {
        ConditionalFeatureProfile::Public2 => artifact.reseal()?,
        ConditionalFeatureProfile::SemanticPlaying4 => artifact.reseal_using(profile)?,
    }
    let mut checkpoint = ConditionalHumanPolicyCheckpoint {
        schema: profile.checkpoint_schema().into(),
        training_version: profile.training_version().into(),
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
    checkpoint.validate_using(profile)?;
    Ok(ConditionalHumanPolicyOutcome {
        model: artifact,
        checkpoint,
        metrics,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    fn datasets(train: usize, validation: usize) -> Vec<ConditionalHumanPolicyDataset> {
        vec![
            ConditionalHumanPolicyDataset::training_fixture("123", DatasetSplit::Train, train),
            ConditionalHumanPolicyDataset::training_fixture(
                "124",
                DatasetSplit::Validation,
                validation,
            ),
        ]
    }
    fn config(epochs: usize) -> PolicyBcConfig {
        PolicyBcConfig {
            epochs,
            batch_size: 1,
            learning_rate: 0.003,
            seed: 7,
        }
    }
    fn playing_observation() -> Observation {
        let mut state =
            tzolkin_core::create_game(vec!["A".into(), "B".into(), "C".into()], 42, false).unwrap();
        state.phase = Phase::Playing;
        tzolkin_core::observation::observe(&state, state.current_player).unwrap()
    }
    #[test]
    fn prepared_conditional_preserves_bits_identity_and_ordered_scalar_scores() {
        let datasets = datasets(0, 1);
        let outcome = train_conditional_datasets(&datasets, &config(1), None).unwrap();
        let bits = outcome
            .model
            .model
            .parameters()
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>();
        let identity = outcome.checkpoint.dataset.clone();
        let checkpoint_checksum = outcome.checkpoint.checksum.clone();
        let prepared = PreparedConditionalHumanPolicy::new(outcome.checkpoint, &datasets).unwrap();
        assert_eq!(
            prepared
                .model()
                .model
                .parameters()
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            bits
        );
        assert_eq!(prepared.identity(), &identity);
        assert_eq!(prepared.checkpoint_checksum(), checkpoint_checksum);
        assert_eq!(prepared.model().task, TASK);
        assert!(!prepared.identity().verified_native_policy);
        assert!(!prepared.identity().source_qualified);
        assert_eq!(prepared.reservations()["123"], DatasetSplit::Train);
        assert_eq!(prepared.reservations()["124"], DatasetSplit::Validation);
        assert_eq!(prepared.backend(), "scalar");
        let mut observation = playing_observation();
        let prediction = prepared.distribution(&observation).unwrap();
        assert_eq!(prediction.logits.len(), observation.legal_actions.len());
        let encoder = FeatureEncoder::new_public(&observation).unwrap();
        let rows = (0..observation.legal_actions.len())
            .map(|index| encoder.encode_legal_tagged(index).unwrap())
            .collect::<Vec<_>>();
        let (loss, _) = policy_loss(
            &prepared.model().model,
            &rows,
            0,
            &mut Scratch::default(),
            None,
        )
        .unwrap();
        let maximum = prediction
            .logits
            .iter()
            .copied()
            .fold(f32::NEG_INFINITY, f32::max) as f64;
        let expected_loss = maximum - f64::from(prediction.logits[0])
            + prediction
                .logits
                .iter()
                .map(|value| (f64::from(*value) - maximum).exp())
                .sum::<f64>()
                .ln();
        assert_eq!(loss, expected_loss);
        observation.legal_actions.reverse();
        observation.observation_key =
            tzolkin_core::observation::observation_key(&observation).unwrap();
        let reversed = prepared.distribution(&observation).unwrap();
        assert_eq!(
            reversed
                .logits
                .iter()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>(),
            prediction
                .logits
                .iter()
                .rev()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>()
        );
    }
    #[test]
    fn prepared_conditional_rejects_wrong_sources_test_and_resealed_metrics() {
        let datasets = datasets(0, 1);
        let outcome = train_conditional_datasets(&datasets, &config(1), None).unwrap();
        assert!(
            PreparedConditionalHumanPolicy::new(outcome.checkpoint.clone(), &self::datasets(1, 1))
                .is_err()
        );
        let mut test = self::datasets(0, 1);
        test.push(ConditionalHumanPolicyDataset::training_fixture(
            "125",
            DatasetSplit::Test,
            0,
        ));
        assert!(PreparedConditionalHumanPolicy::new(outcome.checkpoint.clone(), &test).is_err());
        let left =
            PreparedConditionalHumanPolicy::new(outcome.checkpoint.clone(), &datasets).unwrap();
        let mut right =
            PreparedConditionalHumanPolicy::new(outcome.checkpoint.clone(), &datasets).unwrap();
        let heldout = vec![ConditionalHumanPolicyDataset::training_fixture(
            "124",
            DatasetSplit::Validation,
            1,
        )];
        let before = left.checkpoint.clone();
        let comparison = compare_validation_datasets([&left, &right], &heldout).unwrap();
        assert_eq!(comparison.samples, 1);
        assert_eq!(comparison.candidates, heldout[0].counts().candidates);
        assert_eq!(
            comparison.actors[0].all_rows,
            outcome.metrics.final_validation
        );
        assert_eq!(comparison.actors[0].all_rows, comparison.actors[1].all_rows);
        assert_eq!(comparison.single_candidate_samples, 0);
        assert_eq!(comparison.action_counts.values().sum::<usize>(), 1);
        assert_eq!(
            comparison.candidate_count_samples.values().sum::<usize>(),
            1
        );
        assert!(!comparison.source_qualified && !comparison.strength_measured);
        assert_eq!(left.checkpoint, before);
        assert!(
            compare_validation_collection([&left, &right], Path::new("unbound-validation.json"))
                .is_err()
        );
        right.reservations.insert("124".into(), DatasetSplit::Train);
        assert!(compare_validation_datasets([&left, &right], &heldout).is_err());
        assert!(compare_validation_datasets([&right, &left], &heldout).is_err());
        right
            .reservations
            .insert("124".into(), DatasetSplit::Validation);
        for split in [DatasetSplit::Train, DatasetSplit::Test] {
            let prohibited = vec![ConditionalHumanPolicyDataset::training_fixture(
                "124", split, 1,
            )];
            assert!(compare_validation_datasets([&left, &right], &prohibited).is_err());
        }
        let undeclared = vec![ConditionalHumanPolicyDataset::training_fixture(
            "125",
            DatasetSplit::Validation,
            1,
        )];
        assert!(compare_validation_datasets([&left, &right], &undeclared).is_err());
        let duplicate = vec![
            ConditionalHumanPolicyDataset::training_fixture("124", DatasetSplit::Validation, 1),
            ConditionalHumanPolicyDataset::training_fixture("124", DatasetSplit::Validation, 1),
        ];
        assert!(compare_validation_datasets([&left, &right], &duplicate).is_err());
        assert!(left.recheck_sources().is_err());
        let mut wrong = outcome.checkpoint;
        wrong.metrics.final_validation.policy_loss += 0.5;
        wrong.checksum = wrong.expected_checksum().unwrap();
        wrong.validate().unwrap();
        assert!(PreparedConditionalHumanPolicy::new(wrong, &datasets).is_err());
        let semantic_datasets = vec![
            ConditionalHumanPolicyDataset::training_fixture_using(
                "123",
                DatasetSplit::Train,
                0,
                ConditionalFeatureProfile::SemanticPlaying4,
                true,
            ),
            ConditionalHumanPolicyDataset::training_fixture_using(
                "124",
                DatasetSplit::Validation,
                1,
                ConditionalFeatureProfile::SemanticPlaying4,
                true,
            ),
        ];
        let semantic = train_using_profile(
            &semantic_datasets,
            &config(1),
            None,
            ConditionalFeatureProfile::SemanticPlaying4,
        )
        .unwrap();
        semantic.checkpoint.validate_semantic_playing().unwrap();
        assert!(semantic.checkpoint.validate().is_err());
        assert!(semantic.model.validate().is_err());
        assert_eq!(
            semantic.model.feature_profile,
            Some(ConditionalFeatureProfile::SemanticPlaying4)
        );
        assert!(
            PreparedConditionalHumanPolicy::new(semantic.checkpoint.clone(), &semantic_datasets)
                .is_err()
        );
        let prepared = PreparedConditionalHumanPolicy::new_using(
            semantic.checkpoint.clone(),
            &semantic_datasets,
            ConditionalFeatureProfile::SemanticPlaying4,
        )
        .unwrap();
        assert_eq!(
            prepared.loaded_semantic_playing_sources().err().as_deref(),
            Some("Semantic Playing origin requires successful source load")
        );
        assert_eq!(
            crate::conditional_semantic_playing_rl::SemanticPlayingRlOwner::from_prepared(
                &prepared, 11,
            )
            .err()
            .as_deref(),
            Some("Semantic Playing origin requires successful source load")
        );
        crate::semantic_playing_rl_collection::phase_collection_fixture(&prepared);
        let playing = playing_observation();
        assert!(prepared.distribution(&playing).is_err());
        assert_eq!(
            prepared
                .distribution_semantic_playing(&playing)
                .unwrap()
                .logits
                .len(),
            playing.legal_actions.len()
        );
        let public_playing = ConditionalHumanPolicyDataset::training_fixture_using(
            "123",
            DatasetSplit::Train,
            0,
            ConditionalFeatureProfile::Public2,
            true,
        );
        assert_eq!(
            public_playing.samples()[0].source_label_digest().unwrap(),
            semantic_datasets[0].samples()[0]
                .source_label_digest()
                .unwrap()
        );
        assert!(
            tzolkin_inference::policy::validated_semantic_playing_values(
                &public_playing.samples()[0].candidates()[0]
            )
            .is_err()
        );
        let mut resealed = semantic.checkpoint;
        resealed.metrics.final_validation.policy_loss += 0.5;
        resealed.checksum = resealed.expected_checksum().unwrap();
        resealed.validate_semantic_playing().unwrap();
        assert!(
            PreparedConditionalHumanPolicy::new_using(
                resealed,
                &semantic_datasets,
                ConditionalFeatureProfile::SemanticPlaying4
            )
            .is_err()
        );
    }
    #[test]
    fn prepared_conditional_checks_native_contract_and_playing_key_before_features() {
        let datasets = datasets(0, 1);
        let outcome = train_conditional_datasets(&datasets, &config(1), None).unwrap();
        let prepared = PreparedConditionalHumanPolicy::new(outcome.checkpoint, &datasets).unwrap();
        let state =
            tzolkin_core::create_game(vec!["A".into(), "B".into(), "C".into()], 42, false).unwrap();
        let setup = tzolkin_core::observation::observe(&state, state.current_player).unwrap();
        assert!(
            prepared
                .distribution(&setup)
                .err()
                .unwrap()
                .contains("Playing")
        );
        let mut observation = playing_observation();
        observation.additional_buildings = true;
        observation.observation_key.clear();
        assert!(
            prepared
                .distribution(&observation)
                .err()
                .unwrap()
                .contains("base 3-4p")
        );
        observation.additional_buildings = false;
        assert!(
            prepared
                .distribution(&observation)
                .err()
                .unwrap()
                .contains("Observation key mismatch")
        );
    }
    #[test]
    fn conditional_heldout_labels_do_not_update_policy_but_train_labels_do() {
        let first = train_conditional_datasets(&datasets(0, 0), &config(1), None).unwrap();
        let heldout = train_conditional_datasets(&datasets(0, 1), &config(1), None).unwrap();
        let changed = train_conditional_datasets(&datasets(1, 0), &config(1), None).unwrap();
        assert_eq!(
            first.model.model.parameters(),
            heldout.model.model.parameters()
        );
        assert_eq!(first.checkpoint.optimizer, heldout.checkpoint.optimizer);
        assert_ne!(
            first.model.model.parameters(),
            changed.model.model.parameters()
        );
        assert_ne!(
            first.checkpoint.dataset.fingerprint,
            heldout.checkpoint.dataset.fingerprint
        );
        assert_eq!(first.metrics.final_train.value_samples, 0);
    }
    #[test]
    fn conditional_resume_matches_continuous_and_rejects_identity_or_config_changes() {
        let mut datasets = datasets(0, 1);
        datasets.extend([
            ConditionalHumanPolicyDataset::training_fixture("125", DatasetSplit::Train, 1),
            ConditionalHumanPolicyDataset::training_fixture("126", DatasetSplit::Train, 0),
        ]);
        let first = train_conditional_datasets(&datasets, &config(1), None).unwrap();
        let legacy_bytes = bounded_json(&first.checkpoint).unwrap();
        let legacy: ConditionalHumanPolicyCheckpoint =
            serde_json::from_slice(&legacy_bytes).unwrap();
        assert!(legacy.dataset.label_scope.is_none());
        assert!(
            serde_json::to_value(&legacy.dataset)
                .unwrap()
                .get("labelScope")
                .is_none()
        );
        legacy.validate().unwrap();
        assert_eq!(legacy.checksum, first.checkpoint.checksum);
        assert_eq!(bounded_json(&legacy).unwrap(), legacy_bytes);
        let expanded = vec![
            ConditionalHumanPolicyDataset::factorized_training_fixture(
                "123",
                DatasetSplit::Train,
                0,
            ),
            ConditionalHumanPolicyDataset::factorized_training_fixture(
                "124",
                DatasetSplit::Validation,
                1,
            ),
            ConditionalHumanPolicyDataset::factorized_training_fixture(
                "125",
                DatasetSplit::Train,
                1,
            ),
            ConditionalHumanPolicyDataset::factorized_training_fixture(
                "126",
                DatasetSplit::Train,
                0,
            ),
        ];
        let expanded_identity = audit(&expanded).unwrap();
        assert_eq!(
            expanded_identity.label_scope.as_deref(),
            Some(FACTORIZED_LABEL_SCOPE)
        );
        assert_ne!(
            expanded_identity.fingerprint,
            first.checkpoint.dataset.fingerprint
        );
        assert!(
            train_conditional_datasets(&expanded, &config(2), Some(&first.checkpoint))
                .unwrap_err()
                .contains("dataset/config mismatch")
        );
        let mixed = vec![
            ConditionalHumanPolicyDataset::factorized_training_fixture(
                "123",
                DatasetSplit::Train,
                0,
            ),
            ConditionalHumanPolicyDataset::training_fixture("124", DatasetSplit::Validation, 1),
        ];
        assert!(audit(&mixed).unwrap_err().contains("mix label scopes"));
        let continuous = train_conditional_datasets(&datasets, &config(2), None).unwrap();
        let resumed =
            train_conditional_datasets(&datasets, &config(2), Some(&first.checkpoint)).unwrap();
        assert_eq!(continuous, resumed);
        assert_eq!(
            train_conditional_datasets(&datasets, &config(2), Some(&continuous.checkpoint))
                .unwrap(),
            continuous
        );
        let mut bad = first.checkpoint.clone();
        bad.schema = crate::policy_training::CHECKPOINT_SCHEMA.into();
        assert!(train_conditional_datasets(&datasets, &config(2), Some(&bad)).is_err());
        let mut cfg = config(2);
        cfg.seed += 1;
        assert!(train_conditional_datasets(&datasets, &cfg, Some(&first.checkpoint)).is_err());
    }
    #[test]
    fn conditional_collection_rejects_family_split_conflicts_duplicates_and_test() {
        let conflict = vec![
            ConditionalHumanPolicyDataset::training_fixture("123", DatasetSplit::Train, 0),
            ConditionalHumanPolicyDataset::training_fixture("123", DatasetSplit::Validation, 0),
        ];
        assert!(train_conditional_datasets(&conflict, &config(1), None).is_err());
        let mut duplicate = datasets(0, 0);
        duplicate.push(ConditionalHumanPolicyDataset::training_fixture(
            "123",
            DatasetSplit::Train,
            0,
        ));
        assert!(train_conditional_datasets(&duplicate, &config(1), None).is_err());
        let mut test = datasets(0, 0);
        test.push(ConditionalHumanPolicyDataset::training_fixture(
            "125",
            DatasetSplit::Test,
            0,
        ));
        assert!(train_conditional_datasets(&test, &config(1), None).is_err());
        let only_heldout = vec![ConditionalHumanPolicyDataset::training_fixture(
            "124",
            DatasetSplit::Validation,
            0,
        )];
        assert!(train_conditional_datasets(&only_heldout, &config(1), None).is_err());
    }
    #[test]
    fn conditional_checkpoint_publication_is_distinct_and_create_only() {
        mod temp {
            include!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/tests/support/temp_root.rs"
            ));
        }
        let root = temp::create("conditional-policy-training").unwrap();
        let output = root.join("new");
        let result = train_conditional_datasets(&datasets(0, 1), &config(1), None).unwrap();
        result.save_new_directory(&output).unwrap();
        let bytes = fs::read(output.join("checkpoint.json")).unwrap();
        assert_eq!(
            ConditionalHumanPolicyCheckpoint::load(&output.join("checkpoint.json")).unwrap(),
            result.checkpoint
        );
        assert!(
            serde_json::from_slice::<crate::policy_training::PolicyTrainingCheckpoint>(&bytes)
                .is_err()
        );
        assert!(
            serde_json::from_slice::<PublicPolicyArtifact>(
                &fs::read(output.join("model.json")).unwrap()
            )
            .is_err()
        );
        assert!(result.save_new_directory(&output).is_err());
        assert_eq!(fs::read(output.join("checkpoint.json")).unwrap(), bytes);
        fs::remove_dir_all(root).unwrap();
    }
}
