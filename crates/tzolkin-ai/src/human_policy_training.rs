//! Separate policy-only BC for sealed conditional Human inputs, never native admission.
use crate::dataset::DatasetSplit;
use crate::features::{FEATURE_COUNT, FeatureEncoder, MAX_LEGAL_ACTIONS, PUBLIC_FEATURE_SCHEMA};
use crate::human_initial_wealth::table_family_id;
use crate::human_policy_dataset::{
    ConditionalHumanPolicyDataset, FACTORIZED_LABEL_SCOPE, HumanPolicySample, TASK,
};
use crate::kernel::{Kernel, ResolvedKernel};
use crate::model::{Random, digest};
use crate::policy_training::{
    MAX_CANDIDATE_EVALUATIONS, MAX_UPDATES, PolicyAdamState, PolicyBcConfig, PolicyLossReport,
    PolicyTrainingMetrics, Scratch, bounded_json, policy_loss, publish, read_local_json,
    validate_new_output_directory,
};
use crate::public_model::{
    MODEL_VERSION, PublicPolicyArtifact, PublicPolicyModel, ValueValidity, predict_numeric,
    validate_contract,
};
use serde::{Deserialize, Serialize};
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
fn load_collection(
    path: &Path,
) -> Result<(CollectionInput, Vec<ConditionalHumanPolicyDataset>), String> {
    let input: CollectionInput = read_local_json(path)?;
    if input.schema != COLLECTION_SCHEMA
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
        let dataset = ConditionalHumanPolicyDataset::load(
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
        if self.task != TASK
            || self.input_contract != INPUT_CONTRACT
            || self
                .label_scope
                .as_deref()
                .is_some_and(|scope| scope != FACTORIZED_LABEL_SCOPE)
            || self.feature_schema != PUBLIC_FEATURE_SCHEMA
            || self.feature_count != FEATURE_COUNT
            || self.rules_version != tzolkin_core::public_replay::PUBLIC_RULES_VERSION
            || self.catalog_hash != tzolkin_core::public_replay::public_catalog_hash()
            || self.source_authenticated
            || self.source_qualified
            || self.verified_native_policy
            || self.tables.is_empty()
            || self.tables.len() > MAX_TABLES
            || self.samples[0] == 0
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
fn audit(datasets: &[ConditionalHumanPolicyDataset]) -> Result<ConditionalDatasetIdentity, String> {
    if datasets.is_empty() || datasets.len() > MAX_TABLES {
        return Err("Conditional collection table bound exceeded".into());
    }
    let label_scope = datasets[0].label_scope();
    if datasets
        .iter()
        .any(|dataset| dataset.label_scope() != label_scope)
    {
        return Err("Conditional collections cannot mix label scopes".into());
    }
    let mut identity = ConditionalDatasetIdentity {
        task: TASK.into(),
        input_contract: INPUT_CONTRACT.into(),
        feature_schema: PUBLIC_FEATURE_SCHEMA,
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
            let context = &sample.candidates()[0].values_for_schema(PUBLIC_FEATURE_SCHEMA)?[..384];
            for row in sample.candidates() {
                let values = row.values_for_schema(PUBLIC_FEATURE_SCHEMA)?;
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
    identity.validate()?;
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
        self.checksum = self.expected_checksum()?;
        self.validate()
    }
    pub fn validate(&self) -> Result<(), String> {
        self.model.validate()?;
        if self.schema != MODEL_SCHEMA
            || self.model_version != MODEL_VERSION
            || self.task != TASK
            || self.input_contract != INPUT_CONTRACT
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
        self.dataset.validate()?;
        self.config.validate()?;
        self.model.validate()?;
        self.optimizer.validate()?;
        if self.schema != CHECKPOINT_SCHEMA
            || self.training_version != TRAINING_VERSION
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
}

/// Immutable conditional Human policy with source-checked table identity.
/// Preparation preserves the scalar 512→32→1 parameters; it grants no native
/// policy qualification, RL ownership, value supervision or measured strength.
pub struct PreparedConditionalHumanPolicy {
    checkpoint: Box<ConditionalHumanPolicyCheckpoint>,
    reservations: BTreeMap<String, DatasetSplit>,
    kernel: ResolvedKernel,
}
impl PreparedConditionalHumanPolicy {
    pub fn load(checkpoint_path: &Path, collection_path: &Path) -> Result<Self, String> {
        let checkpoint = ConditionalHumanPolicyCheckpoint::load(checkpoint_path)?;
        let (input, datasets) = load_collection(collection_path)?;
        if input.config != checkpoint.config {
            return Err("Conditional prepared collection/checkpoint config mismatch".into());
        }
        let prepared = Self::new(checkpoint, &datasets)?;
        if read_local_json::<CollectionInput>(collection_path)? != input
            || ConditionalHumanPolicyCheckpoint::load(checkpoint_path)? != *prepared.checkpoint
        {
            return Err("Conditional preparation inputs changed before completion".into());
        }
        Ok(prepared)
    }

    pub fn new(
        checkpoint: ConditionalHumanPolicyCheckpoint,
        datasets: &[ConditionalHumanPolicyDataset],
    ) -> Result<Self, String> {
        checkpoint.validate()?;
        let identity = audit(datasets)?;
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
                    PublicPolicyModel::validated_values(row)?;
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
        let initial = PublicPolicyArtifact::new(checkpoint.config.seed)?.model;
        let mut scratch = Scratch::default();
        if evaluate(&train, &initial, &mut scratch)? != checkpoint.metrics.initial_train
            || evaluate(&validation, &initial, &mut scratch)?
                != checkpoint.metrics.initial_validation
            || evaluate(&train, &checkpoint.model.model, &mut scratch)?
                != checkpoint.metrics.final_train
            || evaluate(&validation, &checkpoint.model.model, &mut scratch)?
                != checkpoint.metrics.final_validation
        {
            return Err("Conditional prepared metric/model/source mismatch".into());
        }
        if audit(datasets)? != identity {
            return Err("Conditional sources changed during preparation".into());
        }
        Ok(Self {
            checkpoint: Box::new(checkpoint),
            reservations,
            kernel: Kernel::Scalar.resolve()?,
        })
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
        validate_contract(observation)?;
        if observation.phase != Phase::Playing {
            return Err("Conditional Human policy requires Playing observations".into());
        }
        let encoder = FeatureEncoder::new_public(observation)?;
        let rows = (0..observation.legal_actions.len())
            .map(|index| encoder.encode_legal_tagged(index))
            .collect::<Result<Vec<_>, _>>()?;
        predict_numeric(&self.checkpoint.model.model, self.kernel, &rows)
    }
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
        self.checkpoint.validate()?;
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
fn evaluate(
    samples: &[&HumanPolicySample],
    model: &PublicPolicyModel,
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
    for sample in samples {
        let (loss, correct) =
            policy_loss(model, sample.candidates(), sample.chosen(), scratch, None)?;
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
    config.validate()?;
    let identity = audit(datasets)?;
    let completed = resume.map_or(0, |checkpoint| checkpoint.completed_epochs);
    if let Some(previous) = resume {
        previous.validate()?;
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
    let current_train = evaluate(&train, &model, &mut scratch)?;
    let current_validation = evaluate(&validation, &model, &mut scratch)?;
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
            policy_loss(
                &model,
                sample.candidates(),
                sample.chosen(),
                &mut scratch,
                Some(&mut gradient),
            )?;
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
    metrics.final_train = evaluate(&train, &model, &mut scratch)?;
    metrics.final_validation = evaluate(&validation, &model, &mut scratch)?;
    if audit(datasets)? != identity {
        return Err("Conditional source bytes changed before completion".into());
    }
    let mut artifact = ConditionalHumanPolicyArtifact {
        schema: MODEL_SCHEMA.into(),
        model_version: MODEL_VERSION.into(),
        task: TASK.into(),
        input_contract: INPUT_CONTRACT.into(),
        dataset_fingerprint: identity.fingerprint.clone(),
        value_validity: ValueValidity::UnavailablePolicyOnly,
        model,
        checksum: String::new(),
    };
    artifact.reseal()?;
    let mut checkpoint = ConditionalHumanPolicyCheckpoint {
        schema: CHECKPOINT_SCHEMA.into(),
        training_version: TRAINING_VERSION.into(),
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
        let mut wrong = outcome.checkpoint;
        wrong.metrics.final_validation.policy_loss += 0.5;
        wrong.checksum = wrong.expected_checksum().unwrap();
        wrong.validate().unwrap();
        assert!(PreparedConditionalHumanPolicy::new(wrong, &datasets).is_err());
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
