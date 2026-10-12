//! Portable, finite Own Setup runs. Only the durable source API owns learning.
//! The caller supplies the complete historical family union and one writer, and
//! retains an external state pin. A watchdog must bound synchronous native work.
use std::collections::BTreeSet;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

use serde::{Deserialize, Serialize, de::DeserializeOwned};
use tzolkin_inference::deployment::{LoadedDeployment, MAX_DEPLOYMENT_BYTES};

use crate::dataset::{DatasetSplit, seed_family_id, split_for_family};
use crate::model::digest;
use crate::own_setup_durable::{
    DurableOwnSetupOwner, OwnSetupArchiveLimits, OwnSetupArchiveSource, OwnSetupArchiveTip,
    OwnSetupRestoreWork,
};
use crate::own_setup_model::InitializedOwnSetupArtifact;
use crate::own_setup_native::OwnSetupNativeConfig;
use crate::own_setup_training::{
    DeclaredOwnSetupExclusions, OwnSetupAdamConfig, OwnSetupAttemptOutcome, OwnSetupAttemptReport,
    OwnSetupRunConfig, OwnSetupTrainingOwner,
};
use crate::public_policy_update::{checkpoint_digest, hash};
use crate::public_rl_paired_session::{local, publish_mixed_metadata, read_bounded};
use crate::public_stochastic::{SamplingSeed, SamplingStreamIdentity};
use crate::public_stochastic_native::{CollectionLimits, NativeStochasticConfig};
use crate::public_stochastic_record::{MAX_RECORD_BYTES, hex64, parse_hex64, unique_json};

pub const SPEC_SCHEMA: &str = "tzolkin-own-setup-run-spec-v1";
pub const EXCLUSIONS_SCHEMA: &str = "tzolkin-own-setup-run-exclusions-v1";
pub const STATE_SCHEMA: &str = "tzolkin-own-setup-run-state-v1";
pub const CONTINUE_SPEC_SCHEMA: &str = "tzolkin-own-setup-continue-spec-v1";
pub const LINEAGE_SCHEMA: &str = "tzolkin-own-setup-run-lineage-v1";
const CONTINUE_STATE_SCHEMA: &str = "tzolkin-own-setup-continue-state-v1";
pub const MAX_SPEC_BYTES: usize = 256 * 1024;
pub const MAX_EXCLUSIONS_BYTES: usize = 16 * 1024 * 1024;
pub const MAX_STATE_BYTES: usize = 128 * 1024;
const MAX_REPORT_BYTES: usize = 2 * 1024 * 1024;
const MANIFEST: &str = "planned-run.json";
const RESERVATION: &str = "reservation.json";
const INITIAL_STATE: &str = "initial-state.json";
const TERMINAL_STOP: &str = "terminal-stop.json";
const ARCHIVE: &str = "archive";

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct ContentPin {
    pub bytes: usize,
    pub sha256: String,
}
impl ContentPin {
    fn of(bytes: &[u8]) -> Self {
        Self {
            bytes: bytes.len(),
            sha256: digest(bytes),
        }
    }
    fn validate(&self, cap: usize) -> Result<(), String> {
        if self.bytes == 0 || self.bytes > cap || !checkpoint_digest(&self.sha256) {
            return Err("Invalid bounded content pin".into());
        }
        Ok(())
    }
    fn check(&self, bytes: &[u8], cap: usize) -> Result<(), String> {
        self.validate(cap)?;
        if *self != Self::of(bytes) {
            return Err("Run input content pin mismatch".into());
        }
        Ok(())
    }
}

/// Completeness is the caller's declaration; its full bytes are pinned by the plan.
#[derive(Debug, Deserialize, Serialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct ExclusionDeclaration {
    pub schema: String,
    pub count: usize,
    pub digest: String,
    pub families: Vec<String>,
}
impl ExclusionDeclaration {
    fn checked(&self) -> Result<DeclaredOwnSetupExclusions, String> {
        if self.schema != EXCLUSIONS_SCHEMA {
            return Err("Unknown Own Setup exclusion schema".into());
        }
        DeclaredOwnSetupExclusions::from_families(self.families.clone(), self.count, &self.digest)
    }
}

/// Hex strings preserve all u64 words across JSON consumers. Episodes are derived
/// in plan order, never selected after observing outcomes.
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct RunStreams {
    pub sampling_seed_hex: String,
    pub replicate_ordinal_hex: String,
    pub first_episode_ordinal_hex: String,
    pub max_attempted_episodes: u64,
}
impl RunStreams {
    fn native(&self) -> Result<OwnSetupRunConfig, String> {
        Ok(OwnSetupRunConfig {
            sampling_seed: parse_hex64(&self.sampling_seed_hex)?,
            replicate_ordinal: parse_hex64(&self.replicate_ordinal_hex)?,
            first_episode_ordinal: parse_hex64(&self.first_episode_ordinal_hex)?,
            max_attempted_episodes: self.max_attempted_episodes,
        })
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct CollectionSpec {
    pub max_callbacks: usize,
    pub max_candidate_rows: usize,
    pub max_source_bytes: usize,
    pub max_trace_bytes: usize,
}
impl CollectionSpec {
    fn native(&self) -> Result<CollectionLimits, String> {
        CollectionLimits::new(
            self.max_callbacks,
            self.max_candidate_rows,
            self.max_source_bytes,
            self.max_trace_bytes,
        )
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct PlannedGame {
    pub players: usize,
    pub environment_seed: u32,
    pub learner_seat: usize,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct OwnSetupRunSpec {
    pub schema: String,
    pub initial_seed_hex: String,
    pub playing: ContentPin,
    pub exclusions: ContentPin,
    pub exclusion_count: usize,
    pub exclusion_digest: String,
    pub adam: OwnSetupAdamConfig,
    pub run: RunStreams,
    pub archive_limits: OwnSetupArchiveLimits,
    pub collection: CollectionSpec,
    pub update_budget: u64,
    /// Per invocation, including input validation and cold restore. An external
    /// watchdog is needed while a synchronous collection/restore call runs.
    pub max_seconds: u64,
    pub batches: Vec<Vec<PlannedGame>>,
    pub ordered_configs_digest: String,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct OrderedConfig<'a> {
    game: &'a PlannedGame,
    sampling_seed_hex: &'a str,
    replicate_ordinal_hex: &'a str,
    episode_ordinal_hex: String,
    limits: &'a CollectionSpec,
}

impl OwnSetupRunSpec {
    pub fn from_json(bytes: &[u8]) -> Result<Self, String> {
        decode(bytes, MAX_SPEC_BYTES)
    }
    /// Planning helper only; this creates no reservation or model owner.
    pub fn ordered_digest(&self) -> Result<String, String> {
        let run = self.run.native()?;
        let mut ordinal = run.first_episode_ordinal;
        let mut expanded = Vec::new();
        for batch in &self.batches {
            let mut configs = Vec::new();
            for game in batch {
                configs.push(OrderedConfig {
                    game,
                    sampling_seed_hex: &self.run.sampling_seed_hex,
                    replicate_ordinal_hex: &self.run.replicate_ordinal_hex,
                    episode_ordinal_hex: hex64(ordinal),
                    limits: &self.collection,
                });
                ordinal = ordinal.checked_add(1).ok_or("Episode ordinal overflow")?;
            }
            expanded.push(configs);
        }
        hash(b"tzolkin-own-setup-run-ordered-configs-v1\0", &expanded)
    }
    fn validate(
        &self,
        declaration: &ExclusionDeclaration,
    ) -> Result<Vec<Vec<OwnSetupNativeConfig>>, String> {
        if self.schema != SPEC_SCHEMA
            || !(1..=64).contains(&self.batches.len())
            || !(1..=self.batches.len() as u64).contains(&self.update_budget)
            || !(1..=86400).contains(&self.max_seconds)
            || !(self.batches.len() as u64..=64).contains(&self.archive_limits.max_attempts)
            || !(32 * 1024 * 1024..=8 * 1024 * 1024 * 1024)
                .contains(&self.archive_limits.max_disk_bytes)
        {
            return Err("Own Setup run exceeds finite attempt/time/archive bounds".into());
        }
        parse_hex64(&self.initial_seed_hex)?;
        self.playing.validate(MAX_DEPLOYMENT_BYTES)?;
        self.exclusions.validate(MAX_EXCLUSIONS_BYTES)?;
        self.adam.validate()?;
        let exclusions = declaration.checked()?;
        if exclusions.count() != self.exclusion_count
            || exclusions.digest() != self.exclusion_digest
            || self.ordered_digest()? != self.ordered_configs_digest
        {
            return Err("Run exclusion or ordered configuration digest mismatch".into());
        }
        let run = self.run.native()?;
        let limits = self.collection.native()?;
        let payload = self
            .collection
            .max_source_bytes
            .checked_add(self.collection.max_trace_bytes)
            .ok_or("Batch payload overflow")?;
        let mut ordinal = run.first_episode_ordinal;
        let mut unique = BTreeSet::new();
        let mut native = Vec::new();
        let mut episodes = 0;
        for batch in &self.batches {
            if !(1..=4).contains(&batch.len())
                || payload
                    .checked_mul(batch.len())
                    .is_none_or(|n| n > MAX_RECORD_BYTES)
            {
                return Err("Run batch exceeds native size/payload bounds".into());
            }
            let mut configs = Vec::new();
            for game in batch {
                let family = seed_family_id(game.environment_seed);
                if split_for_family(&family)? != DatasetSplit::Train
                    || exclusions.families().contains(&family)
                    || !unique.insert(family)
                {
                    return Err(
                        "Run requires globally unique fresh Train families outside the full union"
                            .into(),
                    );
                }
                configs.push(OwnSetupNativeConfig::new(
                    NativeStochasticConfig::new(
                        game.players,
                        game.environment_seed,
                        SamplingStreamIdentity::new(
                            SamplingSeed::new(run.sampling_seed),
                            ordinal,
                            run.replicate_ordinal,
                            game.players,
                        )?,
                        limits,
                    )?,
                    game.learner_seat,
                )?);
                ordinal = ordinal.checked_add(1).ok_or("Episode ordinal overflow")?;
                episodes += 1;
            }
            native.push(configs);
        }
        if episodes == 0 || episodes > 256 || run.max_attempted_episodes != episodes {
            return Err("Run episode budget must equal the complete schedule, at most 256".into());
        }
        Ok(native)
    }
}

/// A new prospective segment, never an edit of a completed original schedule.
#[derive(Debug, Deserialize, Serialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct OwnSetupContinueSpec {
    pub schema: String,
    pub lineage: ContentPin,
    pub segment: OwnSetupRunSpec,
}
impl OwnSetupContinueSpec {
    pub fn from_json(bytes: &[u8]) -> Result<Self, String> {
        let spec: Self = decode(bytes, MAX_SPEC_BYTES)?;
        if spec.schema != CONTINUE_SPEC_SCHEMA {
            return Err("Unknown Own Setup continuation specification".into());
        }
        spec.lineage.validate(MAX_SPEC_BYTES)?;
        Ok(spec)
    }
}

/// Explicit locations relative to the declaration's directory. Each source has
/// its own original plan, full baseline and externally retained exact state.
#[derive(Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(
    tag = "kind",
    rename_all = "camelCase",
    rename_all_fields = "camelCase",
    deny_unknown_fields
)]
pub enum OriginalRunSource {
    Train {
        output: String,
        spec: String,
        exclusions: String,
        state: String,
        expected_sha256: String,
    },
    Continue {
        output: String,
        spec: String,
        exclusions: String,
        lineage: String,
        state: String,
        expected_sha256: String,
    },
}
#[derive(Debug, Deserialize, Serialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct OwnSetupRunLineage {
    pub schema: String,
    pub sources: Vec<OriginalRunSource>,
}

#[derive(Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct Reservation {
    schema: String,
    spec: ContentPin,
    ordered_configs_digest: String,
    families: Vec<String>,
    /// Every family is spent once this receipt is published, including unentered
    /// suffixes and failures. These are distinct originals, not all-seat samples.
    all_families_spent: bool,
}
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct Manifest {
    schema: String,
    spec: ContentPin,
    playing: ContentPin,
    exclusions: ContentPin,
    ordered_configs_digest: String,
    reservation: ContentPin,
}
#[derive(Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct SegmentOffsets {
    attempted_episodes: u64,
    updates: u64,
    ledger_entries: usize,
}
#[derive(Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct ContinuationManifest {
    schema: String,
    plan: Manifest,
    lineage: ContentPin,
    parent_state: ContentPin,
    offsets: SegmentOffsets,
}
/// An externally retained byte pin of this receipt binds the original complete
/// plan and source-produced root/tip. It is custody metadata, not a model owner.
#[derive(Debug, Deserialize, Serialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct OwnSetupRunState {
    pub schema: String,
    pub manifest: ContentPin,
    pub root: ContentPin,
    pub tip: OwnSetupArchiveTip,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct TerminalStop {
    schema: String,
    manifest: ContentPin,
    reason: String,
}
#[derive(Debug, Deserialize, Serialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct BatchIntent {
    schema: String,
    manifest: ContentPin,
    batch: usize,
    before: OwnSetupArchiveTip,
}

pub struct RunInputs<'a> {
    pub spec: &'a Path,
    pub playing: &'a Path,
    pub exclusions: &'a Path,
    /// New for train; existing for resume. Its archive child is source-owned.
    pub output: &'a Path,
    /// Fresh leaf names in output, outside the closed source archive.
    pub artifact: &'a str,
    pub report: &'a str,
    pub tip: &'a str,
    /// Optional exact-byte composite export while the actual trained owner lives.
    pub deployment: Option<&'a str>,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct OwnSetupRunReport {
    schema: &'static str,
    pub stop_reason: String,
    pub exit_code: i32,
    /// None means publication left the live owner inaccessible; no zero claims.
    pub update_count: Option<u64>,
    pub attempted_episodes: Option<u64>,
    next_episode_ordinal_hex: Option<String>,
    closed_attempts: Option<usize>,
    original_collections: Option<usize>,
    fresh_audit_seals: Option<usize>,
    planned_reserved_families: usize,
    source_ledger_terminal: Option<bool>,
    training_playing_loads: usize,
    export_validation_loads: usize,
    cold_restore: Option<OwnSetupRestoreWork>,
    artifact: Option<ContentPin>,
    deployment: Option<ContentPin>,
    state: ContentPin,
    elapsed_seconds: f64,
    error: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    continuation: Option<SegmentOffsets>,
}

struct Prepared {
    spec: OwnSetupRunSpec,
    declaration: ExclusionDeclaration,
    playing_bytes: Vec<u8>,
    configs: Vec<Vec<OwnSetupNativeConfig>>,
    reservation: Reservation,
    manifest: Manifest,
}

struct Ancestor {
    directory: PathBuf,
    prepared: Prepared,
    state: OwnSetupRunState,
}
struct PreparedContinuation {
    prepared: Prepared,
    ancestors: Vec<Ancestor>,
    manifest: ContinuationManifest,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct ContinuationRootBindings {
    schema: String,
    config: RootBindings,
    parent: ContinuationParentBinding,
    offsets: DurableOffsetBindings,
    additional_episodes: u64,
}
#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct ContinuationParentBinding {
    root: RootPinBinding,
    tip: OwnSetupArchiveTip,
}
#[derive(Deserialize)]
struct RootPinBinding {
    name: String,
    bytes: usize,
    sha256: String,
}
#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct DurableOffsetBindings {
    segments: usize,
    attempted_episodes: u64,
    updates: u64,
    ledger_entries: usize,
}

// Reads only original-plan bindings from the source's persisted root. It cannot
// import an owner, optimizer or learning credit. Full root/history validation
// remains exclusively DurableOwnSetupOwner::restore's job.
#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct RootBindings {
    schema: String,
    initial: InitialBinding,
    adam: OwnSetupAdamConfig,
    run: OwnSetupRunConfig,
    limits: OwnSetupArchiveLimits,
    excluded_families: Vec<String>,
    exclusion_digest: String,
}
#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct InitialBinding {
    seed: u64,
    completed_updates: u64,
}
impl RootBindings {
    fn check(
        &self,
        spec: &OwnSetupRunSpec,
        declaration: &ExclusionDeclaration,
    ) -> Result<(), String> {
        let run = spec.run.native()?;
        self.check_run(spec, declaration, &run)
    }
    fn check_run(
        &self,
        spec: &OwnSetupRunSpec,
        declaration: &ExclusionDeclaration,
        run: &OwnSetupRunConfig,
    ) -> Result<(), String> {
        let adam_words = |a: &OwnSetupAdamConfig| {
            [
                a.learning_rate.to_bits(),
                a.beta1.to_bits(),
                a.beta2.to_bits(),
                a.epsilon.to_bits(),
                a.max_gradient_norm.to_bits(),
                a.max_parameter_delta.to_bits(),
            ]
        };
        let run_words = |r: &OwnSetupRunConfig| {
            [
                r.sampling_seed,
                r.replicate_ordinal,
                r.first_episode_ordinal,
                r.max_attempted_episodes,
            ]
        };
        if self.schema != "tzolkin-own-setup-durable-root-v1"
            || self.initial.seed != parse_hex64(&spec.initial_seed_hex)?
            || self.initial.completed_updates != 0
            || adam_words(&self.adam) != adam_words(&spec.adam)
            || run_words(&self.run) != run_words(run)
            || self.limits.max_attempts != spec.archive_limits.max_attempts
            || self.limits.max_disk_bytes != spec.archive_limits.max_disk_bytes
            || self.excluded_families != declaration.families
            || self.exclusion_digest != declaration.digest
        {
            return Err("Actual durable root differs from the original initialization/run/Adam/limits/full exclusions".into());
        }
        Ok(())
    }
}
impl Prepared {
    fn read(inputs: &RunInputs<'_>) -> Result<Self, String> {
        let spec_bytes = read_bounded(inputs.spec, MAX_SPEC_BYTES)?;
        let spec = OwnSetupRunSpec::from_json(&spec_bytes)?;
        Self::read_spec(
            spec,
            ContentPin::of(&spec_bytes),
            inputs.playing,
            inputs.exclusions,
        )
    }
    fn read_spec(
        spec: OwnSetupRunSpec,
        spec_pin: ContentPin,
        playing: &Path,
        exclusions: &Path,
    ) -> Result<Self, String> {
        let exclusion_bytes = read_bounded(exclusions, MAX_EXCLUSIONS_BYTES)?;
        spec.exclusions
            .check(&exclusion_bytes, MAX_EXCLUSIONS_BYTES)?;
        let declaration: ExclusionDeclaration = decode(&exclusion_bytes, MAX_EXCLUSIONS_BYTES)?;
        let configs = spec.validate(&declaration)?;
        let playing_bytes = read_bounded(playing, MAX_DEPLOYMENT_BYTES)?;
        spec.playing.check(&playing_bytes, MAX_DEPLOYMENT_BYTES)?;
        let reservation = Reservation {
            schema: "tzolkin-own-setup-run-reservation-v1".into(),
            spec: spec_pin.clone(),
            ordered_configs_digest: spec.ordered_configs_digest.clone(),
            families: spec
                .batches
                .iter()
                .flatten()
                .map(|g| seed_family_id(g.environment_seed))
                .collect(),
            all_families_spent: true,
        };
        let manifest = Manifest {
            schema: "tzolkin-own-setup-planned-run-v1".into(),
            spec: spec_pin,
            playing: spec.playing.clone(),
            exclusions: spec.exclusions.clone(),
            ordered_configs_digest: spec.ordered_configs_digest.clone(),
            reservation: ContentPin::of(&json(&reservation, MAX_STATE_BYTES)?),
        };
        Ok(Self {
            spec,
            declaration,
            playing_bytes,
            configs,
            reservation,
            manifest,
        })
    }
    fn verify_saved(&self, directory: &Path, state: &OwnSetupRunState) -> Result<(), String> {
        let reservation_bytes = read_bounded(&directory.join(RESERVATION), MAX_STATE_BYTES)?;
        self.manifest
            .reservation
            .check(&reservation_bytes, MAX_STATE_BYTES)?;
        if decode::<Reservation>(&reservation_bytes, MAX_STATE_BYTES)? != self.reservation {
            return Err("Original complete reservation mismatch".into());
        }
        let manifest_bytes = read_bounded(&directory.join(MANIFEST), MAX_STATE_BYTES)?;
        state.manifest.check(&manifest_bytes, MAX_STATE_BYTES)?;
        if decode::<Manifest>(&manifest_bytes, MAX_STATE_BYTES)? != self.manifest {
            return Err(
                "Resume requires the exact original plan, Playing and full exclusions".into(),
            );
        }
        let root_bytes = check_root(directory, state)?;
        decode::<RootBindings>(&root_bytes, MAX_EXCLUSIONS_BYTES)?
            .check(&self.spec, &self.declaration)?;
        self.check_pending_configs(directory, state.tip.sequence)?;
        read_stop(directory, &state.manifest)?;
        Ok(())
    }
    fn check_pending_configs(&self, directory: &Path, sequence: u64) -> Result<(), String> {
        if sequence > 128 || sequence.div_ceil(2) as usize > self.configs.len() {
            return Err("Archive exceeds the original complete schedule".into());
        }
        for batch in 0..sequence.div_ceil(2) as usize {
            let sequence = 2 * batch as u64 + 1;
            let bytes = read_bounded(
                &directory
                    .join(ARCHIVE)
                    .join(format!("head-{sequence:08}.json")),
                MAX_STATE_BYTES,
            )?;
            let head = unique_json(&bytes, MAX_STATE_BYTES)?;
            let expected: Vec<_> = self.configs[batch].iter().map(config_value).collect();
            // These are integer-only Rust JSON values, retaining exact u64s.
            // Source restore subsequently checks the full closed DTO/checksum.
            if head["schema"] != "tzolkin-own-setup-durable-head-v1"
                || head["sequence"] != sequence
                || head["event"]["kind"] != "pending"
                || head["event"]["configs"] != serde_json::Value::Array(expected)
            {
                return Err("Saved source configs differ from original ordered players/seats/streams/limits".into());
            }
        }
        Ok(())
    }
}

impl PreparedContinuation {
    fn read(inputs: &RunInputs<'_>, lineage_path: &Path) -> Result<Self, String> {
        let bytes = read_bounded(inputs.spec, MAX_SPEC_BYTES)?;
        let spec = OwnSetupContinueSpec::from_json(&bytes)?;
        let lineage_bytes = read_bounded(lineage_path, MAX_SPEC_BYTES)?;
        spec.lineage.check(&lineage_bytes, MAX_SPEC_BYTES)?;
        let lineage: OwnSetupRunLineage = decode(&lineage_bytes, MAX_SPEC_BYTES)?;
        if lineage.schema != LINEAGE_SCHEMA || !(1..=15).contains(&lineage.sources.len()) {
            return Err(
                "Continuation requires one original source and at most14 successors".into(),
            );
        }
        let prepared = Prepared::read_spec(
            spec.segment,
            ContentPin::of(&bytes),
            inputs.playing,
            inputs.exclusions,
        )?;
        if prepared.spec.update_budget != prepared.configs.len() as u64 {
            return Err("A prospective continuation must budget every original batch".into());
        }
        let lineage_directory = local(lineage_path, false)?;
        let lineage_directory = lineage_directory
            .parent()
            .ok_or("Lineage lacks a directory")?;
        let full = prepared.declaration.checked()?;
        let mut ancestors = Vec::<Ancestor>::new();
        let mut directories = BTreeSet::new();
        let mut offsets = SegmentOffsets {
            attempted_episodes: 0,
            updates: 0,
            ledger_entries: 0,
        };
        let mut parent_state = None;
        for (index, source) in lineage.sources.iter().enumerate() {
            let (output, original_spec, exclusions, state, expected, original_lineage) =
                match source {
                    OriginalRunSource::Train {
                        output,
                        spec,
                        exclusions,
                        state,
                        expected_sha256,
                    } if index == 0 => (output, spec, exclusions, state, expected_sha256, None),
                    OriginalRunSource::Continue {
                        output,
                        spec,
                        exclusions,
                        state,
                        expected_sha256,
                        lineage,
                    } if index != 0 => (
                        output,
                        spec,
                        exclusions,
                        state,
                        expected_sha256,
                        Some(lineage),
                    ),
                    _ => {
                        return Err(
                            "Lineage must begin with train followed only by continuations".into(),
                        );
                    }
                };
            let resolve = |p: &str| -> Result<PathBuf, String> {
                if p.is_empty() || p.len() > 4096 {
                    return Err("Invalid original source location".into());
                }
                local(&lineage_directory.join(p), false)
            };
            let directory = resolve(output)?;
            let canonical = fs::canonicalize(directory.join(ARCHIVE)).map_err(|e| e.to_string())?;
            if !directories.insert(canonical) {
                return Err("Lineage repeats a source archive".into());
            }
            let state_path = resolve(state)?;
            let original_state = pinned_state(&state_path, expected)?;
            let state_pin = ContentPin::of(&read_bounded(&state_path, MAX_STATE_BYTES)?);
            if state_pin.sha256 != *expected {
                return Err("Original external state changed during validation".into());
            }
            let original_spec_path = resolve(original_spec)?;
            let original_bytes = read_bounded(&original_spec_path, MAX_SPEC_BYTES)?;
            let original_exclusions = resolve(exclusions)?;
            let (original, original_lineage_pin) = if let Some(location) = original_lineage {
                let original_spec = OwnSetupContinueSpec::from_json(&original_bytes)?;
                let original_lineage_bytes = read_bounded(&resolve(location)?, MAX_SPEC_BYTES)?;
                original_spec
                    .lineage
                    .check(&original_lineage_bytes, MAX_SPEC_BYTES)?;
                let original_declaration: OwnSetupRunLineage =
                    decode(&original_lineage_bytes, MAX_SPEC_BYTES)?;
                if original_declaration.schema != LINEAGE_SCHEMA
                    || original_declaration.sources != lineage.sources[..index]
                {
                    return Err(
                        "Original successor lineage differs from the declared complete prefix"
                            .into(),
                    );
                }
                (original_spec.segment, Some(original_spec.lineage))
            } else {
                (OwnSetupRunSpec::from_json(&original_bytes)?, None)
            };
            let original = Prepared::read_spec(
                original,
                ContentPin::of(&original_bytes),
                inputs.playing,
                &original_exclusions,
            )?;
            if let Some(original_lineage_pin) = original_lineage_pin {
                let expected_manifest = ContinuationManifest {
                    schema: "tzolkin-own-setup-continue-manifest-v1".into(),
                    plan: original.manifest.clone(),
                    lineage: original_lineage_pin,
                    parent_state: parent_state
                        .clone()
                        .ok_or("Successor lacks its original parent state")?,
                    offsets: SegmentOffsets {
                        attempted_episodes: offsets.attempted_episodes,
                        updates: offsets.updates,
                        ledger_entries: offsets.ledger_entries,
                    },
                };
                verify_continuation_saved(
                    &original,
                    &expected_manifest,
                    &directory,
                    &original_state,
                    &ancestors,
                )?;
            } else {
                original.verify_saved(&directory, &original_state)?;
            }
            if original_state.tip.sequence != 2 * original.configs.len() as u64
                || DurableOwnSetupOwner::inspect_tip(&directory.join(ARCHIVE))?
                    != original_state.tip
                || read_stop(&directory, &original_state.manifest)?.is_some()
            {
                return Err(
                    "Continuation requires exact latest complete successful original schedules"
                        .into(),
                );
            }
            let reports = saved_reports(&directory, original.configs.len())?;
            if terminal(&reports).is_some()
                || check_intents(
                    &directory,
                    &original_state.manifest,
                    &reports,
                    original.configs.len(),
                )?
            {
                return Err("Original source has rejected/NoChange/unfinished intent work".into());
            }
            check_segment_reports(
                &reports,
                &original.spec,
                offsets.attempted_episodes,
                offsets.updates,
            )?;
            if !original
                .declaration
                .checked()?
                .families()
                .is_subset(full.families())
                || original
                    .reservation
                    .families
                    .iter()
                    .any(|f| !full.families().contains(f))
            {
                return Err(
                    "New full baseline omits an ancestral exclusion or permanently reserved family"
                        .into(),
                );
            }
            offsets.attempted_episodes = offsets
                .attempted_episodes
                .checked_add(original.spec.run.max_attempted_episodes)
                .ok_or("Ancestral original count overflow")?;
            offsets.updates += reports.len() as u64;
            offsets.ledger_entries += reports.len();
            parent_state = Some(state_pin);
            ancestors.push(Ancestor {
                directory,
                prepared: original,
                state: original_state,
            });
        }
        check_successor_stream(&prepared.spec, &ancestors, &offsets)?;
        let manifest = ContinuationManifest {
            schema: "tzolkin-own-setup-continue-manifest-v1".into(),
            plan: prepared.manifest.clone(),
            lineage: spec.lineage,
            parent_state: parent_state.ok_or("Continuation lacks original sources")?,
            offsets,
        };
        Ok(Self {
            prepared,
            ancestors,
            manifest,
        })
    }
}

fn saved_reports(directory: &Path, batches: usize) -> Result<Vec<OwnSetupAttemptReport>, String> {
    (0..batches)
        .map(|batch| {
            let bytes = read_bounded(
                &directory
                    .join(ARCHIVE)
                    .join(format!("head-{:08}.json", 2 * batch + 2)),
                MAX_STATE_BYTES,
            )?;
            let value = unique_json(&bytes, MAX_STATE_BYTES)?;
            if value["event"]["kind"] != "closed" || value["event"]["abandoned"] != false {
                return Err("Original source lacks a nonabandoned Closed report".into());
            }
            serde_json::from_value(value["event"]["report"].clone()).map_err(|e| e.to_string())
        })
        .collect()
}

fn check_successor_stream(
    spec: &OwnSetupRunSpec,
    ancestors: &[Ancestor],
    offsets: &SegmentOffsets,
) -> Result<(), String> {
    let original = &ancestors
        .first()
        .ok_or("Missing original train source")?
        .prepared
        .spec;
    let origin = original.run.native()?;
    let successor = spec.run.native()?;
    let expected_ordinal = origin
        .first_episode_ordinal
        .checked_add(offsets.attempted_episodes)
        .ok_or("Cumulative episode ordinal overflow")?;
    if spec.initial_seed_hex != original.initial_seed_hex
        || successor.sampling_seed != origin.sampling_seed
        || successor.replicate_ordinal != origin.replicate_ordinal
        || successor.first_episode_ordinal != expected_ordinal
        || offsets
            .attempted_episodes
            .checked_add(successor.max_attempted_episodes)
            .is_none_or(|n| n > 4096)
    {
        return Err(
            "Continuation changes initialization/stream or exceeds cumulative originals".into(),
        );
    }
    let bits = |a: &OwnSetupAdamConfig| {
        [
            a.learning_rate.to_bits(),
            a.beta1.to_bits(),
            a.beta2.to_bits(),
            a.epsilon.to_bits(),
            a.max_gradient_norm.to_bits(),
            a.max_parameter_delta.to_bits(),
        ]
    };
    if bits(&spec.adam) != bits(&original.adam) {
        return Err("Continuation changes exact Adam configuration".into());
    }
    Ok(())
}

fn verify_continuation_saved(
    prepared: &Prepared,
    expected: &ContinuationManifest,
    directory: &Path,
    state: &OwnSetupRunState,
    ancestors: &[Ancestor],
) -> Result<(), String> {
    if state.schema != CONTINUE_STATE_SCHEMA {
        return Err("Unknown continuation state".into());
    }
    let reservation = read_bounded(&directory.join(RESERVATION), MAX_STATE_BYTES)?;
    expected
        .plan
        .reservation
        .check(&reservation, MAX_STATE_BYTES)?;
    if decode::<Reservation>(&reservation, MAX_STATE_BYTES)? != prepared.reservation {
        return Err("Continuation original reservation differs".into());
    }
    let manifest_bytes = read_bounded(&directory.join(MANIFEST), MAX_STATE_BYTES)?;
    state.manifest.check(&manifest_bytes, MAX_STATE_BYTES)?;
    if decode::<ContinuationManifest>(&manifest_bytes, MAX_STATE_BYTES)? != *expected {
        return Err("Continuation requires its exact original ancestry/spec/full baseline".into());
    }
    let root_bytes = read_bounded(
        &directory.join(ARCHIVE).join("root.json"),
        MAX_EXCLUSIONS_BYTES,
    )?;
    state.root.check(&root_bytes, MAX_EXCLUSIONS_BYTES)?;
    let root: ContinuationRootBindings = decode(&root_bytes, MAX_EXCLUSIONS_BYTES)?;
    let parent = ancestors.last().ok_or("Continuation lacks a predecessor")?;
    let mut run = prepared.spec.run.native()?;
    run.first_episode_ordinal = ancestors[0]
        .prepared
        .spec
        .run
        .native()?
        .first_episode_ordinal;
    run.max_attempted_episodes = expected
        .offsets
        .attempted_episodes
        .checked_add(run.max_attempted_episodes)
        .ok_or("Cumulative run budget overflow")?;
    if root.schema != "tzolkin-own-setup-durable-continuation-root-v1"
        || root.parent.root.name != "root.json"
        || root.parent.root.bytes != parent.state.root.bytes
        || root.parent.root.sha256 != parent.state.root.sha256
        || root.parent.tip != parent.state.tip
        || root.offsets.segments != ancestors.len() + 1
        || root.offsets.attempted_episodes != expected.offsets.attempted_episodes
        || root.offsets.updates != expected.offsets.updates
        || root.offsets.ledger_entries != expected.offsets.ledger_entries
        || root.additional_episodes != prepared.spec.run.max_attempted_episodes
    {
        return Err("Continuation actual source parent/offset/schedule binding differs".into());
    }
    root.config
        .check_run(&prepared.spec, &prepared.declaration, &run)?;
    prepared.check_pending_configs(directory, state.tip.sequence)?;
    read_stop(directory, &state.manifest)?;
    Ok(())
}

fn check_segment_reports(
    reports: &[OwnSetupAttemptReport],
    spec: &OwnSetupRunSpec,
    attempted_offset: u64,
    update_offset: u64,
) -> Result<(), String> {
    if reports.len() > spec.batches.len() {
        return Err("Segment exceeds its original batches".into());
    }
    let mut ordinal = spec.run.native()?.first_episode_ordinal;
    let mut episodes = attempted_offset;
    for (index, (report, batch)) in reports.iter().zip(&spec.batches).enumerate() {
        episodes = episodes
            .checked_add(batch.len() as u64)
            .ok_or("Cumulative attempt overflow")?;
        if report.first_episode_ordinal != ordinal
            || report.reserved_episodes != batch.len()
            || report.families
                != batch
                    .iter()
                    .map(|g| seed_family_id(g.environment_seed))
                    .collect::<Vec<_>>()
            || report.attempted_episodes != episodes
            || matches!(report.outcome, OwnSetupAttemptOutcome::Updated { update_count, .. } if update_count != update_offset + index as u64 + 1)
        {
            return Err(
                "Segment ledger differs from its original ordered reservation and offsets".into(),
            );
        }
        ordinal = ordinal
            .checked_add(batch.len() as u64)
            .ok_or("Episode ordinal overflow")?;
        if report.next_episode_ordinal != Some(ordinal) {
            return Err("Segment report cursor differs".into());
        }
    }
    Ok(())
}

fn config_value(config: &OwnSetupNativeConfig) -> serde_json::Value {
    let native = config.native();
    let stream = native.sampling_identity();
    let limits = native.limits();
    serde_json::json!({
        "players": native.players(), "environmentSeed": native.environment_seed(),
        "samplingSeed": stream.sampling_seed().value(), "episodeOrdinal": stream.episode_ordinal(),
        "replicateOrdinal": stream.replicate_ordinal(), "learnerSeat": config.learner_seat(),
        "maxCallbacks": limits.max_callbacks(), "maxCandidateRows": limits.max_candidate_rows(),
        "maxSourceBytes": limits.max_source_bytes(), "maxTraceBytes": limits.max_trace_bytes(),
    })
}

/// Publishes the complete immutable reservation before any model is loaded.
pub fn train(inputs: RunInputs<'_>) -> Result<OwnSetupRunReport, String> {
    let began = Instant::now();
    let prepared = Prepared::read(&inputs)?;
    let directory = output(&inputs, true)?;
    fs::create_dir(&directory).map_err(|e| e.to_string())?;
    publish(
        &directory,
        RESERVATION,
        &prepared.reservation,
        MAX_STATE_BYTES,
    )?;
    let manifest = publish(&directory, MANIFEST, &prepared.manifest, MAX_STATE_BYTES)?;
    if began.elapsed().as_secs() >= prepared.spec.max_seconds {
        return Err(
            "Input preparation exhausted the fixed time budget; all reservations remain spent"
                .into(),
        );
    }
    let playing = LoadedDeployment::load(&prepared.playing_bytes)?;
    let mut durable = DurableOwnSetupOwner::create(
        &directory.join(ARCHIVE),
        InitializedOwnSetupArtifact::new(parse_hex64(&prepared.spec.initial_seed_hex)?)?,
        playing,
        prepared.spec.adam.clone(),
        prepared.spec.run.native()?,
        prepared.declaration.checked()?,
        prepared.spec.archive_limits.clone(),
    )?;
    let root = ContentPin::of(&read_bounded(
        &directory.join(ARCHIVE).join("root.json"),
        MAX_EXCLUSIONS_BYTES,
    )?);
    let state = OwnSetupRunState {
        schema: STATE_SCHEMA.into(),
        manifest,
        root,
        tip: durable.tip().clone(),
    };
    publish(&directory, INITIAL_STATE, &state, MAX_STATE_BYTES)?;
    finish(
        inputs,
        &directory,
        prepared,
        &mut durable,
        state,
        None,
        began,
    )
}

/// Cold restore accepts no trained-artifact bootstrap. The exact original
/// inputs and an external raw state pin are checked before the one Playing load.
pub fn resume(
    inputs: RunInputs<'_>,
    state_path: &Path,
    expected_sha256: &str,
) -> Result<OwnSetupRunReport, String> {
    let began = Instant::now();
    let prepared = Prepared::read(&inputs)?;
    let directory = output(&inputs, false)?;
    let state = pinned_state(state_path, expected_sha256)?;
    prepared.verify_saved(&directory, &state)?;
    if began.elapsed().as_secs() >= prepared.spec.max_seconds {
        return Err("Input preparation exhausted the fixed time budget".into());
    }
    let playing = LoadedDeployment::load(&prepared.playing_bytes)?;
    let (mut durable, work) = DurableOwnSetupOwner::restore(
        &directory.join(ARCHIVE),
        &state.tip,
        playing,
        prepared.declaration.checked()?,
    )?;
    finish(
        inputs,
        &directory,
        prepared,
        &mut durable,
        state,
        Some(work),
        began,
    )
}

/// Reserves the entire new segment before one Playing load and ancestral replay.
/// Only the source's reconstructed live owner can create the child archive.
pub fn continue_run(inputs: RunInputs<'_>, lineage: &Path) -> Result<OwnSetupRunReport, String> {
    let began = Instant::now();
    let mut prepared = PreparedContinuation::read(&inputs, lineage)?;
    let directory = output(&inputs, true)?;
    outside_original_archives(&directory, &prepared.ancestors)?;
    fs::create_dir(&directory).map_err(|e| e.to_string())?;
    publish(
        &directory,
        RESERVATION,
        &prepared.prepared.reservation,
        MAX_STATE_BYTES,
    )?;
    let manifest = publish(&directory, MANIFEST, &prepared.manifest, MAX_STATE_BYTES)?;
    if began.elapsed().as_secs() >= prepared.prepared.spec.max_seconds {
        return Err(
            "Continuation preparation exhausted time; all new reservations remain spent".into(),
        );
    }
    let playing = LoadedDeployment::load(&prepared.prepared.playing_bytes)?;
    let directories: Vec<_> = prepared
        .ancestors
        .iter()
        .map(|a| a.directory.join(ARCHIVE))
        .collect();
    let tips: Vec<_> = prepared
        .ancestors
        .iter()
        .map(|a| a.state.tip.clone())
        .collect();
    let sources = original_sources(&mut prepared.ancestors, &directories, &tips)?;
    let (parent, work) = DurableOwnSetupOwner::restore_chain(sources, playing)?;
    let offsets = &prepared.manifest.offsets;
    let owner = parent.owner()?;
    if owner.attempted_episodes() != offsets.attempted_episodes
        || owner.update_count() != offsets.updates
        || owner.ledger().len() != offsets.ledger_entries
        || owner.next_episode_ordinal()
            != Some(prepared.prepared.spec.run.native()?.first_episode_ordinal)
    {
        return Err(
            "Reconstructed parent differs from the exact cumulative schedule offsets".into(),
        );
    }
    if began.elapsed().as_secs() >= prepared.prepared.spec.max_seconds {
        return Err(
            "Ancestral cold audit exhausted time; all new reservations remain spent".into(),
        );
    }
    let mut durable = parent.continue_in(
        &directory.join(ARCHIVE),
        prepared.prepared.spec.run.max_attempted_episodes,
        prepared.prepared.declaration.checked()?,
        prepared.prepared.spec.archive_limits.clone(),
    )?;
    let state = OwnSetupRunState {
        schema: CONTINUE_STATE_SCHEMA.into(),
        manifest,
        root: ContentPin::of(&read_bounded(
            &directory.join(ARCHIVE).join("root.json"),
            MAX_EXCLUSIONS_BYTES,
        )?),
        tip: durable.tip().clone(),
    };
    publish(&directory, INITIAL_STATE, &state, MAX_STATE_BYTES)?;
    finish_segment(
        inputs,
        &directory,
        prepared.prepared,
        &mut durable,
        state,
        Some(work),
        began,
        Some(prepared.manifest.offsets),
    )
}

/// Reuses only the exact original child plan and full named source chain.
pub fn resume_continuation(
    inputs: RunInputs<'_>,
    lineage: &Path,
    state_path: &Path,
    expected_sha256: &str,
) -> Result<OwnSetupRunReport, String> {
    let began = Instant::now();
    let mut prepared = PreparedContinuation::read(&inputs, lineage)?;
    let directory = output(&inputs, false)?;
    let state = pinned_state(state_path, expected_sha256)?;
    verify_continuation_saved(
        &prepared.prepared,
        &prepared.manifest,
        &directory,
        &state,
        &prepared.ancestors,
    )?;
    if DurableOwnSetupOwner::inspect_tip(&directory.join(ARCHIVE))? != state.tip {
        return Err("Continuation requires the exact latest externally pinned source tip".into());
    }
    if began.elapsed().as_secs() >= prepared.prepared.spec.max_seconds {
        return Err("Continuation validation exhausted time".into());
    }
    let playing = LoadedDeployment::load(&prepared.prepared.playing_bytes)?;
    let mut directories: Vec<_> = prepared
        .ancestors
        .iter()
        .map(|a| a.directory.join(ARCHIVE))
        .collect();
    let mut tips: Vec<_> = prepared
        .ancestors
        .iter()
        .map(|a| a.state.tip.clone())
        .collect();
    directories.push(directory.join(ARCHIVE));
    tips.push(state.tip.clone());
    let mut sources = original_sources(
        &mut prepared.ancestors,
        &directories[..directories.len() - 1],
        &tips[..tips.len() - 1],
    )?;
    sources.push(OwnSetupArchiveSource {
        directory: directories.last().ok_or("Missing child source")?,
        expected_tip: tips.last().ok_or("Missing child tip")?,
        exclusions: prepared.prepared.declaration.checked()?,
    });
    let (mut durable, work) = DurableOwnSetupOwner::restore_chain(sources, playing)?;
    finish_segment(
        inputs,
        &directory,
        prepared.prepared,
        &mut durable,
        state,
        Some(work),
        began,
        Some(prepared.manifest.offsets),
    )
}

fn original_sources<'a>(
    ancestors: &mut [Ancestor],
    directories: &'a [PathBuf],
    tips: &'a [OwnSetupArchiveTip],
) -> Result<Vec<OwnSetupArchiveSource<'a>>, String> {
    if ancestors.len() != directories.len() || ancestors.len() != tips.len() {
        return Err("Original source list differs".into());
    }
    ancestors
        .iter_mut()
        .zip(directories)
        .zip(tips)
        .map(|((a, directory), expected_tip)| {
            // The metadata has already been checked. Transfer its large baseline
            // strings rather than retaining a second copy during source cold replay.
            let families = std::mem::take(&mut a.prepared.declaration.families);
            Ok(OwnSetupArchiveSource {
                directory,
                expected_tip,
                exclusions: DeclaredOwnSetupExclusions::from_families(
                    families,
                    a.prepared.declaration.count,
                    &a.prepared.declaration.digest,
                )?,
            })
        })
        .collect()
}

fn outside_original_archives(directory: &Path, ancestors: &[Ancestor]) -> Result<(), String> {
    let parent = directory
        .parent()
        .ok_or("Continuation output lacks parent")?;
    let name = directory
        .file_name()
        .ok_or("Continuation output lacks leaf")?;
    let canonical = fs::canonicalize(parent)
        .map_err(|e| e.to_string())?
        .join(name);
    for ancestor in ancestors {
        let source =
            fs::canonicalize(ancestor.directory.join(ARCHIVE)).map_err(|e| e.to_string())?;
        if canonical.starts_with(source) {
            return Err("Continuation output must be outside every original source archive".into());
        }
    }
    Ok(())
}

/// Metadata-only candidate recovery; no owner or model authority is returned.
pub fn inspect_continuation_tip(
    output: &Path,
    state_path: &Path,
    expected_sha256: &str,
    new_tip: &str,
) -> Result<ContentPin, String> {
    let directory = local(output, false)?;
    leaf(new_tip)?;
    absent(&directory.join(new_tip))?;
    let mut state = pinned_state(state_path, expected_sha256)?;
    if state.schema != CONTINUE_STATE_SCHEMA {
        return Err("Unknown continuation state".into());
    }
    let manifest_bytes = read_bounded(&directory.join(MANIFEST), MAX_STATE_BYTES)?;
    state.manifest.check(&manifest_bytes, MAX_STATE_BYTES)?;
    let manifest: ContinuationManifest = decode(&manifest_bytes, MAX_STATE_BYTES)?;
    if manifest.schema != "tzolkin-own-setup-continue-manifest-v1" {
        return Err("Unknown continuation manifest".into());
    }
    manifest.plan.reservation.check(
        &read_bounded(&directory.join(RESERVATION), MAX_STATE_BYTES)?,
        MAX_STATE_BYTES,
    )?;
    state.root.check(
        &read_bounded(
            &directory.join(ARCHIVE).join("root.json"),
            MAX_EXCLUSIONS_BYTES,
        )?,
        MAX_EXCLUSIONS_BYTES,
    )?;
    state.tip = DurableOwnSetupOwner::inspect_tip(&directory.join(ARCHIVE))?;
    publish(&directory, new_tip, &state, MAX_STATE_BYTES)
}

/// Metadata-only recovery after an uncertain publication. The returned tip is
/// a candidate; only a later full source restore can confer an owner.
pub fn inspect_tip(
    output: &Path,
    state_path: &Path,
    expected_sha256: &str,
    new_tip: &str,
) -> Result<ContentPin, String> {
    let directory = local(output, false)?;
    leaf(new_tip)?;
    absent(&directory.join(new_tip))?;
    let mut state = pinned_state(state_path, expected_sha256)?;
    let manifest_bytes = read_bounded(&directory.join(MANIFEST), MAX_STATE_BYTES)?;
    state.manifest.check(&manifest_bytes, MAX_STATE_BYTES)?;
    let manifest: Manifest = decode(&manifest_bytes, MAX_STATE_BYTES)?;
    manifest.reservation.check(
        &read_bounded(&directory.join(RESERVATION), MAX_STATE_BYTES)?,
        MAX_STATE_BYTES,
    )?;
    check_root(&directory, &state)?;
    state.tip = DurableOwnSetupOwner::inspect_tip(&directory.join(ARCHIVE))?;
    publish(&directory, new_tip, &state, MAX_STATE_BYTES)
}

fn finish(
    inputs: RunInputs<'_>,
    directory: &Path,
    prepared: Prepared,
    durable: &mut DurableOwnSetupOwner,
    state: OwnSetupRunState,
    work: Option<OwnSetupRestoreWork>,
    began: Instant,
) -> Result<OwnSetupRunReport, String> {
    finish_segment(
        inputs, directory, prepared, durable, state, work, began, None,
    )
}

#[allow(clippy::too_many_arguments)]
fn finish_segment(
    inputs: RunInputs<'_>,
    directory: &Path,
    prepared: Prepared,
    durable: &mut DurableOwnSetupOwner,
    mut state: OwnSetupRunState,
    work: Option<OwnSetupRestoreWork>,
    began: Instant,
    continuation: Option<SegmentOffsets>,
) -> Result<OwnSetupRunReport, String> {
    let mut error = None;
    let ledger_offset = continuation.as_ref().map_or(0, |s| s.ledger_entries);
    let attempted_offset = continuation.as_ref().map_or(0, |s| s.attempted_episodes);
    let update_offset = continuation.as_ref().map_or(0, |s| s.updates);
    if durable.ledger_offset() != ledger_offset
        || durable.attempted_episode_offset() != attempted_offset
        || durable.update_offset() != update_offset
    {
        return Err("Actual carried owner segment offsets differ from original manifest".into());
    }
    let ledger = durable
        .owner()?
        .ledger()
        .get(ledger_offset..)
        .ok_or("Invalid cumulative ledger offset")?;
    let saved_stop = read_stop(directory, &state.manifest)?;
    let unfinished_intent =
        check_intents(directory, &state.manifest, ledger, prepared.configs.len())?;
    let reason = loop {
        let owner = durable.owner()?;
        let consumed = if continuation.is_some() {
            validate_segment_ledger(
                owner,
                &prepared,
                ledger_offset,
                attempted_offset,
                update_offset,
            )?
        } else {
            validate_ledger(owner, &prepared)?
        };
        let ledger = owner
            .ledger()
            .get(ledger_offset..)
            .ok_or("Invalid segment ledger slice")?;
        if let Some(reason) = terminal(ledger) {
            break reason.to_owned();
        }
        if unfinished_intent {
            break "unfinishedIntent".into();
        }
        if let Some(reason) = &saved_stop {
            break reason.clone();
        }
        if owner
            .update_count()
            .checked_sub(update_offset)
            .ok_or("Update offset exceeds owner")?
            >= prepared.spec.update_budget
        {
            break "updateBudgetReached".into();
        }
        if consumed == prepared.configs.len() {
            break "scheduleExhausted".into();
        }
        if began.elapsed().as_secs() >= prepared.spec.max_seconds {
            break "wallBudgetReached".into();
        }
        publish(
            directory,
            &intent_name(consumed),
            &BatchIntent {
                schema: "tzolkin-own-setup-run-batch-intent-v1".into(),
                manifest: state.manifest.clone(),
                batch: consumed,
                before: durable.tip().clone(),
            },
            MAX_STATE_BYTES,
        )?;
        // Source API publishes Pending before collecting, then audits every
        // callback and closes the actual report/checkpoint before returning.
        if let Err(e) = durable.attempt(&prepared.configs[consumed]) {
            error = Some(e);
            break "sourceAttemptError".into();
        }
    };
    if matches!(
        reason.as_str(),
        "sourceNoChange" | "sourceRejected" | "sourceAttemptError" | "unfinishedIntent"
    ) && saved_stop.is_none()
    {
        publish(
            directory,
            TERMINAL_STOP,
            &TerminalStop {
                schema: "tzolkin-own-setup-run-terminal-stop-v1".into(),
                manifest: state.manifest.clone(),
                reason: reason.clone(),
            },
            MAX_STATE_BYTES,
        )?;
    }
    let owner = durable.owner().ok();
    let mut artifact = None;
    let mut deployment = None;
    if let Some(trained) = owner.and_then(|o| o.trained_artifact()) {
        trained.validate()?;
        artifact = Some(publish(
            directory,
            inputs.artifact,
            trained,
            MAX_REPORT_BYTES,
        )?);
        if let Some(name) = inputs.deployment {
            let bytes = crate::public_deployment::export_own_setup_composite(
                trained,
                &prepared.playing_bytes,
            )?;
            deployment = Some(publish_bytes(
                directory,
                name,
                &bytes,
                MAX_DEPLOYMENT_BYTES,
            )?);
        }
    }
    // Inaccessible live state is not treated as zero successful work. A caller
    // can inspect a new exact published tip and cold-restore it explicitly.
    state.tip = durable.tip().clone();
    let state_pin = publish(directory, inputs.tip, &state, MAX_STATE_BYTES)?;
    let report = OwnSetupRunReport {
        schema: if continuation.is_some() {
            "tzolkin-own-setup-continue-report-v1"
        } else {
            "tzolkin-own-setup-run-report-v1"
        },
        exit_code: match reason.as_str() {
            "updateBudgetReached" => 0,
            "sourceRejected" | "sourceAttemptError" | "unfinishedIntent" => 1,
            _ => 2,
        },
        stop_reason: reason,
        update_count: owner.map(|o| o.update_count()),
        attempted_episodes: owner.map(|o| o.attempted_episodes()),
        next_episode_ordinal_hex: owner.and_then(|o| o.next_episode_ordinal()).map(hex64),
        closed_attempts: owner.map(|o| o.ledger().len() - ledger_offset),
        original_collections: owner.map(|o| {
            o.ledger()[ledger_offset..]
                .iter()
                .map(|r| r.receipts.len())
                .sum()
        }),
        fresh_audit_seals: owner.map(|o| {
            o.ledger()[ledger_offset..]
                .iter()
                .flat_map(|r| &r.receipts)
                .filter(|r| r.episode_id.is_some())
                .count()
        }),
        planned_reserved_families: prepared.reservation.families.len(),
        source_ledger_terminal: owner.map(|o| terminal(&o.ledger()[ledger_offset..]).is_some()),
        training_playing_loads: 1,
        // The reviewed exporter validates the Playing binding, numeric checksum
        // and final composite by constructing three additional Playing loads.
        export_validation_loads: if deployment.is_some() { 3 } else { 0 },
        cold_restore: work,
        artifact,
        deployment,
        state: state_pin,
        elapsed_seconds: began.elapsed().as_secs_f64(),
        error,
        continuation,
    };
    publish(directory, inputs.report, &report, MAX_REPORT_BYTES)?;
    Ok(report)
}

fn terminal(ledger: &[OwnSetupAttemptReport]) -> Option<&'static str> {
    ledger.iter().find_map(|r| match r.outcome {
        OwnSetupAttemptOutcome::NoChange => Some("sourceNoChange"),
        OwnSetupAttemptOutcome::Rejected { .. } => Some("sourceRejected"),
        OwnSetupAttemptOutcome::Updated { .. } => None,
    })
}

fn intent_name(batch: usize) -> String {
    format!("intent-{batch:02}.json")
}
fn intent_closed(ledger: &[OwnSetupAttemptReport], batch: usize) -> bool {
    ledger
        .get(batch)
        .is_some_and(|r| matches!(r.outcome, OwnSetupAttemptOutcome::Updated { .. }))
}
fn check_intents(
    directory: &Path,
    manifest: &ContentPin,
    ledger: &[OwnSetupAttemptReport],
    batches: usize,
) -> Result<bool, String> {
    let mut unfinished = false;
    // This bounded metadata check runs once per invocation. Source cold restore
    // has already validated the archive; live updates never replay old records.
    for batch in 0..batches {
        let path = directory.join(intent_name(batch));
        match fs::symlink_metadata(&path) {
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                if batch < ledger.len() {
                    return Err("Source ledger lacks its original pre-call intent".into());
                }
            }
            Err(e) => return Err(e.to_string()),
            Ok(_) => {
                let intent: BatchIntent =
                    decode(&read_bounded(&path, MAX_STATE_BYTES)?, MAX_STATE_BYTES)?;
                let sequence = 2 * batch as u64;
                if intent.schema != "tzolkin-own-setup-run-batch-intent-v1"
                    || intent.manifest != *manifest
                    || intent.batch != batch
                    || intent.before.sequence != sequence
                    || intent.before.name != format!("head-{sequence:08}.json")
                {
                    return Err("Intent differs from the original plan/source ancestry".into());
                }
                ContentPin {
                    bytes: intent.before.bytes,
                    sha256: intent.before.sha256,
                }
                .check(
                    &read_bounded(
                        &directory.join(ARCHIVE).join(intent.before.name),
                        MAX_STATE_BYTES,
                    )?,
                    MAX_STATE_BYTES,
                )?;
                unfinished |= !intent_closed(ledger, batch);
            }
        }
    }
    Ok(unfinished)
}

fn validate_ledger(owner: &OwnSetupTrainingOwner, prepared: &Prepared) -> Result<usize, String> {
    let ledger = owner.ledger();
    if ledger.len() > prepared.configs.len() {
        return Err("Archive exceeds original planned batches".into());
    }
    let run = prepared.spec.run.native()?;
    let mut ordinal = run.first_episode_ordinal;
    let mut episodes = 0;
    for (report, batch) in ledger.iter().zip(&prepared.spec.batches) {
        let families: Vec<_> = batch
            .iter()
            .map(|g| seed_family_id(g.environment_seed))
            .collect();
        episodes += batch.len() as u64;
        if report.first_episode_ordinal != ordinal
            || report.reserved_episodes != batch.len()
            || report.families != families
            || report.attempted_episodes != episodes
        {
            return Err("Source ledger differs from original ordered reservation".into());
        }
        ordinal = ordinal
            .checked_add(batch.len() as u64)
            .ok_or("Episode ordinal overflow")?;
    }
    if owner.attempted_episodes() != episodes || owner.next_episode_ordinal() != Some(ordinal) {
        return Err("Source cursor differs from complete original plan".into());
    }
    Ok(ledger.len())
}

fn validate_segment_ledger(
    owner: &OwnSetupTrainingOwner,
    prepared: &Prepared,
    ledger_offset: usize,
    attempted_offset: u64,
    update_offset: u64,
) -> Result<usize, String> {
    let ledger = owner
        .ledger()
        .get(ledger_offset..)
        .ok_or("Segment ledger offset exceeds source")?;
    check_segment_reports(ledger, &prepared.spec, attempted_offset, update_offset)?;
    let consumed = ledger
        .iter()
        .map(|r| r.reserved_episodes as u64)
        .sum::<u64>();
    let cursor = prepared
        .spec
        .run
        .native()?
        .first_episode_ordinal
        .checked_add(consumed);
    if owner.attempted_episodes()
        != attempted_offset
            .checked_add(consumed)
            .ok_or("Cumulative source attempted overflow")?
        || owner.next_episode_ordinal() != cursor
    {
        return Err("Cumulative source cursor differs from the new segment schedule".into());
    }
    Ok(ledger.len())
}

fn output(inputs: &RunInputs<'_>, new: bool) -> Result<PathBuf, String> {
    let directory = local(inputs.output, new)?;
    if new {
        absent(&directory)?;
    }
    let mut names = BTreeSet::new();
    for name in [inputs.artifact, inputs.report, inputs.tip]
        .into_iter()
        .chain(inputs.deployment)
    {
        leaf(name)?;
        if !names.insert(name) {
            return Err("Output leaf names must be distinct".into());
        }
        if !new {
            absent(&directory.join(name))?;
        }
    }
    Ok(directory)
}
fn leaf(name: &str) -> Result<(), String> {
    let stem = name.split('.').next().unwrap_or("");
    let device = matches!(stem, "con" | "prn" | "aux" | "nul")
        || (stem.len() == 4
            && (stem.starts_with("com") || stem.starts_with("lpt"))
            && matches!(stem.as_bytes()[3], b'1'..=b'9'));
    if name.is_empty()
        || name.len() > 96
        || name.starts_with('.')
        || name.ends_with('.')
        || device
        || name.starts_with("intent-")
        || !name.bytes().all(|b| {
            b.is_ascii_digit() || b.is_ascii_lowercase() || matches!(b, b'-' | b'_' | b'.')
        })
        || matches!(
            name,
            MANIFEST | RESERVATION | INITIAL_STATE | TERMINAL_STOP | ARCHIVE
        )
    {
        return Err("Use a fresh ordinary output leaf outside the source archive".into());
    }
    Ok(())
}
fn read_stop(directory: &Path, manifest: &ContentPin) -> Result<Option<String>, String> {
    let path = directory.join(TERMINAL_STOP);
    match fs::symlink_metadata(&path) {
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(e) => Err(e.to_string()),
        Ok(_) => {
            let stop: TerminalStop =
                decode(&read_bounded(&path, MAX_STATE_BYTES)?, MAX_STATE_BYTES)?;
            if stop.schema != "tzolkin-own-setup-run-terminal-stop-v1"
                || stop.manifest != *manifest
                || !matches!(
                    stop.reason.as_str(),
                    "sourceNoChange" | "sourceRejected" | "sourceAttemptError" | "unfinishedIntent"
                )
            {
                return Err("Invalid immutable original-run terminal stop".into());
            }
            Ok(Some(stop.reason))
        }
    }
}
fn absent(path: &Path) -> Result<(), String> {
    match fs::symlink_metadata(path) {
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Ok(_) => Err("Own Setup output already exists; no overwrite/retry".into()),
        Err(e) => Err(e.to_string()),
    }
}
fn check_root(directory: &Path, state: &OwnSetupRunState) -> Result<Vec<u8>, String> {
    if state.schema != STATE_SCHEMA {
        return Err("Unknown Own Setup run state".into());
    }
    let bytes = read_bounded(
        &directory.join(ARCHIVE).join("root.json"),
        MAX_EXCLUSIONS_BYTES,
    )?;
    state.root.check(&bytes, MAX_EXCLUSIONS_BYTES)?;
    Ok(bytes)
}
fn pinned_state(path: &Path, expected: &str) -> Result<OwnSetupRunState, String> {
    if !checkpoint_digest(expected) {
        return Err("Expected external lowercase raw SHA256".into());
    }
    let bytes = read_bounded(path, MAX_STATE_BYTES)?;
    if digest(&bytes) != expected {
        return Err("External original-run/state pin mismatch".into());
    }
    decode(&bytes, MAX_STATE_BYTES)
}
fn decode<T: DeserializeOwned>(bytes: &[u8], cap: usize) -> Result<T, String> {
    unique_json(bytes, cap)?;
    serde_json::from_slice(bytes).map_err(|e| e.to_string())
}
fn json(value: &impl Serialize, cap: usize) -> Result<Vec<u8>, String> {
    let bytes = serde_json::to_vec(value).map_err(|e| e.to_string())?;
    if bytes.is_empty() || bytes.len() > cap {
        return Err("Own Setup metadata exceeds byte cap".into());
    }
    Ok(bytes)
}
fn publish(
    directory: &Path,
    name: &str,
    value: &impl Serialize,
    cap: usize,
) -> Result<ContentPin, String> {
    let bytes = json(value, cap)?;
    publish_bytes(directory, name, &bytes, cap)
}
fn publish_bytes(
    directory: &Path,
    name: &str,
    bytes: &[u8],
    cap: usize,
) -> Result<ContentPin, String> {
    if bytes.is_empty() || bytes.len() > cap {
        return Err("Own Setup output exceeds byte cap".into());
    }
    publish_mixed_metadata(directory, name, bytes)?;
    Ok(ContentPin::of(bytes))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn declaration(families: Vec<String>) -> ExclusionDeclaration {
        ExclusionDeclaration {
            schema: EXCLUSIONS_SCHEMA.into(),
            count: families.len(),
            digest: DeclaredOwnSetupExclusions::digest_for_families(&families).unwrap(),
            families,
        }
    }
    fn spec() -> OwnSetupRunSpec {
        let mut spec = OwnSetupRunSpec {
            schema: SPEC_SCHEMA.into(),
            initial_seed_hex: "ffffffffffffffff".into(),
            playing: ContentPin::of(b"metadata fixture only"),
            exclusions: ContentPin::of(b"[]"),
            exclusion_count: 0,
            exclusion_digest: declaration(vec![]).digest,
            adam: OwnSetupAdamConfig {
                learning_rate: 1e-4,
                beta1: 0.9,
                beta2: 0.999,
                epsilon: 1e-8,
                max_gradient_norm: 2.0,
                max_parameter_delta: 0.001,
            },
            run: RunStreams {
                sampling_seed_hex: "ffffffffffffffff".into(),
                replicate_ordinal_hex: "8000000000000001".into(),
                first_episode_ordinal_hex: "fffffffffffffffd".into(),
                max_attempted_episodes: 1,
            },
            archive_limits: OwnSetupArchiveLimits {
                max_attempts: 1,
                max_disk_bytes: 1024 * 1024 * 1024,
            },
            collection: CollectionSpec {
                max_callbacks: 4,
                max_candidate_rows: 1024,
                max_source_bytes: 1024 * 1024,
                max_trace_bytes: 1024 * 1024,
            },
            update_budget: 1,
            max_seconds: 1,
            batches: vec![vec![PlannedGame {
                players: 3,
                environment_seed: 0,
                learner_seat: 0,
            }]],
            ordered_configs_digest: String::new(),
        };
        // Existing source-test metadata range only; no collection or reservation.
        spec.batches[0][0].environment_seed = (0..100)
            .find(|s| split_for_family(&seed_family_id(*s)).unwrap() == DatasetSplit::Train)
            .unwrap();
        spec.ordered_configs_digest = spec.ordered_digest().unwrap();
        spec
    }

    #[test]
    fn complete_plan_preserves_wide_stream_words_and_rejects_reuse_or_changed_order() {
        let mut spec = spec();
        let configs = spec.validate(&declaration(vec![])).unwrap();
        let identity = configs[0][0].native().sampling_identity();
        assert_eq!(identity.sampling_seed().value(), u64::MAX);
        assert_eq!(identity.replicate_ordinal(), (1 << 63) + 1);
        assert_eq!(identity.episode_ordinal(), u64::MAX - 2);
        let value = config_value(&configs[0][0]);
        let serialized = serde_json::to_vec(&value).unwrap();
        let decoded = unique_json(&serialized, MAX_STATE_BYTES).unwrap();
        assert_eq!(decoded["samplingSeed"].as_u64(), Some(u64::MAX));
        assert_eq!(decoded["replicateOrdinal"].as_u64(), Some((1 << 63) + 1));
        assert_eq!(decoded["episodeOrdinal"].as_u64(), Some(u64::MAX - 2));
        spec.batches[0][0].learner_seat = 1;
        assert!(spec.validate(&declaration(vec![])).is_err());
        spec.ordered_configs_digest = spec.ordered_digest().unwrap();
        let duplicate = spec.batches[0][0].clone();
        spec.batches[0].push(duplicate);
        spec.run.max_attempted_episodes = 2;
        spec.ordered_configs_digest = spec.ordered_digest().unwrap();
        assert!(spec.validate(&declaration(vec![])).is_err());
        spec.batches[0].pop();
        spec.run.max_attempted_episodes = 1;
        let excluded = declaration(vec![seed_family_id(spec.batches[0][0].environment_seed)]);
        spec.exclusion_count = excluded.count;
        spec.exclusion_digest = excluded.digest.clone();
        spec.ordered_configs_digest = spec.ordered_digest().unwrap();
        assert!(spec.validate(&excluded).is_err());
    }

    #[test]
    fn actual_root_bindings_require_exact_adam_run_limits_and_complete_exclusions() {
        let spec = spec();
        let declaration = declaration(vec![]);
        let mut bindings = RootBindings {
            schema: "tzolkin-own-setup-durable-root-v1".into(),
            initial: InitialBinding {
                seed: u64::MAX,
                completed_updates: 0,
            },
            adam: spec.adam.clone(),
            run: spec.run.native().unwrap(),
            limits: spec.archive_limits.clone(),
            excluded_families: vec![],
            exclusion_digest: declaration.digest.clone(),
        };
        bindings.check(&spec, &declaration).unwrap();
        bindings.adam.beta2 = f64::from_bits(bindings.adam.beta2.to_bits() + 1);
        assert!(bindings.check(&spec, &declaration).is_err());
        bindings.adam = spec.adam.clone();
        bindings.run.replicate_ordinal -= 1;
        assert!(bindings.check(&spec, &declaration).is_err());
        bindings.run = spec.run.native().unwrap();
        bindings.limits.max_attempts += 1;
        assert!(bindings.check(&spec, &declaration).is_err());
        bindings.limits = spec.archive_limits.clone();
        bindings.excluded_families.push("0".repeat(64));
        assert!(bindings.check(&spec, &declaration).is_err());
    }

    #[test]
    fn terminal_ledger_and_unfinished_intent_do_not_depend_on_cli_stop_receipts() {
        let report = |outcome| OwnSetupAttemptReport {
            first_episode_ordinal: u64::MAX,
            reserved_episodes: 1,
            next_episode_ordinal: None,
            attempted_episodes: 1,
            families: vec![],
            setup_before: String::new(),
            setup_after: String::new(),
            receipts: vec![],
            outcome,
        };
        let no_change = [report(OwnSetupAttemptOutcome::NoChange)];
        assert_eq!(terminal(&no_change), Some("sourceNoChange"));
        assert!(!intent_closed(&no_change, 0));
        let rejected = [report(OwnSetupAttemptOutcome::Rejected {
            reason: "Interrupted durable reservation consumed without model work".into(),
        })];
        assert_eq!(terminal(&rejected), Some("sourceRejected"));
        assert!(!intent_closed(&rejected, 0));
        assert!(!intent_closed(&[], 0));
        let updated = [report(OwnSetupAttemptOutcome::Updated {
            update_count: 1,
            changed_parameters: 1,
            max_parameter_delta: 0.001,
        })];
        assert!(terminal(&updated).is_none());
        assert!(intent_closed(&updated, 0));
        assert!(!intent_closed(&updated, 1));
        // Counter/plan metadata only, with no source owner or learning credit.
        let spec = spec();
        let first = spec.run.native().unwrap().first_episode_ordinal;
        let mut continued = report(OwnSetupAttemptOutcome::Updated {
            update_count: 61,
            changed_parameters: 1,
            max_parameter_delta: 0.001,
        });
        continued.first_episode_ordinal = first;
        continued.next_episode_ordinal = first.checked_add(1);
        continued.attempted_episodes = 241;
        continued.families = vec![seed_family_id(spec.batches[0][0].environment_seed)];
        check_segment_reports(std::slice::from_ref(&continued), &spec, 240, 60).unwrap();
        assert!(check_segment_reports(std::slice::from_ref(&continued), &spec, 0, 60).is_err());
        assert!(check_segment_reports(std::slice::from_ref(&continued), &spec, 240, 0).is_err());
        continued.next_episode_ordinal = None;
        assert!(check_segment_reports(&[continued], &spec, 240, 60).is_err());
    }

    #[test]
    fn closed_json_and_output_names_fail_before_model_work() {
        assert!(
            decode::<ContentPin>(br#"{"bytes":1,"bytes":2,"sha256":"x"}"#, MAX_SPEC_BYTES).is_err()
        );
        let mut value = serde_json::to_value(spec()).unwrap();
        value["unrecognized"] = true.into();
        assert!(OwnSetupRunSpec::from_json(&serde_json::to_vec(&value).unwrap()).is_err());
        let continuation = OwnSetupContinueSpec {
            schema: CONTINUE_SPEC_SCHEMA.into(),
            lineage: ContentPin::of(b"metadata-only"),
            segment: spec(),
        };
        let bytes = serde_json::to_vec(&continuation).unwrap();
        let decoded = OwnSetupContinueSpec::from_json(&bytes).unwrap();
        assert_eq!(decoded.segment.run.sampling_seed_hex, "ffffffffffffffff");
        let mut value = serde_json::to_value(&continuation).unwrap();
        value["unknown"] = true.into();
        assert!(OwnSetupContinueSpec::from_json(&serde_json::to_vec(&value).unwrap()).is_err());
        for name in [
            "archive",
            "terminal-stop.json",
            "intent-00.json",
            "../tip",
            "TIP.json",
            "con.json",
            "tip.",
        ] {
            assert!(leaf(name).is_err());
        }
        leaf("tip-1.json").unwrap();
    }
}
