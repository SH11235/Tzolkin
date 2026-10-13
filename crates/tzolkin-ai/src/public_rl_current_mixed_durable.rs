//! One explicitly requested current-mixed whole-four operation per call.
//! Live checks cover original controls, the immediate owner and the active
//! batch. Complete archives are checked on close/restore, never every phase.
//! A pending restore cannot replay, reseal, recollect or repeat an optimizer.
use crate::arena::BorrowedArenaPolicy;
use crate::dataset::{DatasetSplit, seed_family_id, split_for_family};
use crate::public_policy_baseline::ResidualFitConfig;
use crate::public_policy_mixed_continued::{
    self as mixed, ContinuedMixedOwner, CurrentMixedPolicyHandle,
};
use crate::public_policy_update::{
    OneStepConfig, PhaseTimer, TimedPhase, Untimed, checkpoint_digest, hash, timed,
};
use crate::public_rl_current_mixed_session::{
    AdmittedCurrentMixedRoot, CurrentMixedCollectionSpec, OriginalLivePins,
};
use crate::public_rl_current_mixed_source::{
    self as source, CurrentMixedParentIdentity, PlannedCurrentMixedCohort,
};
use crate::public_rl_mixed_native::{FrozenMixedRoster, SavedRosterProfile};
use crate::public_rl_mixed_session::{FilePin, OpponentSpec, read_pin};
use crate::public_rl_paired_session::{
    local, publish_mixed_metadata, publish_mixed_stored_record, read_bounded,
    read_mixed_stored_record,
};
use crate::public_rl_record_storage::{self as record_storage, MAX_STORED_RECORD_BYTES};
use crate::public_stochastic_record::{Config, MAX_RECORD_BYTES, unique_json};
use crate::search::PreparedSearch;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    borrow::Cow,
    collections::{BTreeMap, BTreeSet},
    fs,
    path::{Path, PathBuf},
    time::Instant,
};

#[path = "public_rl_current_mixed_sampling_consumption.rs"]
mod sampling;
pub use sampling::CurrentMixedSamplingConsumptionSpec;

#[path = "public_rl_current_mixed_restore.rs"]
mod restore;
pub use restore::{CurrentMixedRestoreStep, CurrentMixedRestoreTip};
#[path = "public_rl_current_mixed_segment.rs"]
mod segment;
pub use segment::{
    CurrentMixedSegmentSpec, GLOBAL_CONTRACT_SCHEMA, SEGMENT_JOURNAL_SCHEMA, SEGMENT_SPEC_SCHEMA,
    StoppedCurrentMixedForkSource,
};

#[path = "public_rl_current_mixed_campaign.rs"]
mod campaign;
pub use campaign::{
    CurrentMixedCampaignSpec, CurrentMixedLivePlanSpec, current_mixed_campaign,
    current_mixed_live_plan,
};

pub const SPEC_SCHEMA: &str = "tzolkin-current-mixed-durable-spec-v2";
pub const JOURNAL_SCHEMA: &str = "tzolkin-current-mixed-durable-journal-v2";
pub const COMPRESSED_SPEC_SCHEMA: &str = "tzolkin-current-mixed-durable-spec-v3";
pub const COMPRESSED_JOURNAL_SCHEMA: &str = "tzolkin-current-mixed-durable-journal-v3";
const SPEC_CAP: usize = 64 * 1024;
const HEAD_CAP: usize = 512 * 1024;
const ARTIFACT_CAP: usize = 16 * 1024 * 1024;
const EVENTS_PER_ATTEMPT: u64 = 21;
const MAX_ATTEMPTS: u64 = 64;
const MAX_FAMILIES: usize = 4096;
const METADATA_HEADROOM: u64 =
    EVENTS_PER_ATTEMPT * HEAD_CAP as u64 + ARTIFACT_CAP as u64 + 2 * HEAD_CAP as u64;
const MAX_STORAGE: u64 = 64 * 1024 * 1024 * 1024;
const ACTIVE_RECORD_CACHE_BYTES: usize = MAX_RECORD_BYTES;
fn sha(raw: &[u8]) -> String {
    format!("{:x}", Sha256::digest(raw))
}
fn bytes(v: &impl Serialize, cap: usize) -> Result<Vec<u8>, String> {
    let raw = serde_json::to_vec(v).map_err(|e| e.to_string())?;
    if raw.is_empty() || raw.len() > cap {
        return Err("Current mixed metadata byte cap".into());
    }
    Ok(raw)
}
fn closed<T: serde::de::DeserializeOwned>(raw: &[u8], cap: usize) -> Result<T, String> {
    serde_json::from_value(unique_json(raw, cap)?).map_err(|e| e.to_string())
}
fn file_pin(name: &str, raw: &[u8]) -> FilePin {
    FilePin {
        name: name.into(),
        bytes: raw.len(),
        sha256: sha(raw),
    }
}
fn leaf(pin: &FilePin, name: &str, cap: usize) -> Result<(), String> {
    if pin.name != name || pin.bytes == 0 || pin.bytes > cap || !checkpoint_digest(&pin.sha256) {
        return Err("Current mixed pinned leaf/bytes/SHA shape".into());
    }
    Ok(())
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct RecordStorageSpec {
    schema: String,
    max_stored_bytes: u64,
    max_temporary_bytes: u64,
    max_disk_bytes: u64,
}
impl RecordStorageSpec {
    fn validate(&self) -> Result<(), String> {
        if self.schema != record_storage::STORAGE_SCHEMA
            || !(4 * MAX_STORED_RECORD_BYTES as u64..=MAX_STORAGE).contains(&self.max_stored_bytes)
            || !(MAX_STORED_RECORD_BYTES as u64..=512 * 1024 * 1024)
                .contains(&self.max_temporary_bytes)
            || !(4 * MAX_STORED_RECORD_BYTES as u64 + METADATA_HEADROOM + self.max_temporary_bytes
                ..=MAX_STORAGE)
                .contains(&self.max_disk_bytes)
        {
            return Err("Current mixed explicit compressed storage limits".into());
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct CompressedRecordPin {
    schema: String,
    stored: FilePin,
    raw_bytes: usize,
    raw_sha256: String,
}
// The raw alternative retains the exact original v2 FilePin wire shape.
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(untagged)]
enum RecordPin {
    Raw(FilePin),
    Compressed(CompressedRecordPin),
}
impl RecordPin {
    fn file(&self) -> &FilePin {
        match self {
            Self::Raw(pin) => pin,
            Self::Compressed(pin) => &pin.stored,
        }
    }
    fn raw_bytes(&self) -> usize {
        match self {
            Self::Raw(pin) => pin.bytes,
            Self::Compressed(pin) => pin.raw_bytes,
        }
    }
    fn validate(&self, member: usize, spec: &CurrentMixedRunSpec) -> Result<(), String> {
        match (self, &spec.record_storage) {
            (Self::Raw(pin), None) => leaf(pin, &record_name(member, spec), MAX_RECORD_BYTES),
            (Self::Compressed(pin), Some(_))
                if pin.schema == record_storage::PIN_SCHEMA
                    && checkpoint_digest(&pin.raw_sha256) =>
            {
                leaf(
                    &pin.stored,
                    &record_name(member, spec),
                    record_storage::maximum_stored_bytes(pin.raw_bytes)?,
                )
            }
            _ => Err("Current mixed record pin/storage mode differs".into()),
        }
    }
}
fn record_name(member: usize, spec: &CurrentMixedRunSpec) -> String {
    let suffix = if spec.record_storage.is_some() {
        ".gz"
    } else {
        ""
    };
    format!("record-{member}.json{suffix}")
}
fn encode_record<'r>(
    raw: &'r [u8],
    member: usize,
    spec: &CurrentMixedRunSpec,
) -> Result<(Cow<'r, [u8]>, RecordPin), String> {
    if member >= 4 || raw.is_empty() || raw.len() > MAX_RECORD_BYTES {
        return Err("Current mixed raw record member/byte cap".into());
    }
    let name = record_name(member, spec);
    match spec.record_storage {
        None => Ok((Cow::Borrowed(raw), RecordPin::Raw(file_pin(&name, raw)))),
        Some(_) => {
            let stored = record_storage::encode(raw)?;
            let pin = RecordPin::Compressed(CompressedRecordPin {
                schema: record_storage::PIN_SCHEMA.into(),
                stored: file_pin(&name, &stored),
                raw_bytes: raw.len(),
                raw_sha256: sha(raw),
            });
            Ok((Cow::Owned(stored), pin))
        }
    }
}
fn publish_record_file(
    directory: &Path,
    stored: &[u8],
    raw: &[u8],
    pin: &RecordPin,
    member: usize,
    spec: &CurrentMixedRunSpec,
) -> Result<Option<VerifiedRecordBytes>, String> {
    pin.validate(member, spec)?;
    if stored.len() != pin.file().bytes || sha(stored) != pin.file().sha256 {
        return Err("Current mixed publication differs from stored record pin".into());
    }
    if spec.record_storage.is_some() {
        publish_mixed_stored_record(directory, &pin.file().name, stored)?;
    } else {
        publish_mixed_metadata(directory, &pin.file().name, stored)?;
    }
    if read_record(directory, pin, member, spec)? != raw {
        return Err("Current mixed record readback differs from original raw bytes".into());
    }
    Ok(
        (stored.len() <= ACTIVE_RECORD_CACHE_BYTES).then(|| VerifiedRecordBytes {
            directory: directory.to_owned(),
            pin: pin.file().clone(),
            compressed: spec.record_storage.is_some(),
            bytes: stored.to_vec(),
        }),
    )
}

// Only exact publication/readback supplies these bytes. This process-local
// cache never replaces a current-path read or an archive/restore SHA check.
struct VerifiedRecordBytes {
    directory: PathBuf,
    pin: FilePin,
    compressed: bool,
    bytes: Vec<u8>,
}
#[derive(Default)]
struct ActiveRecordByteCache {
    attempt: Option<u64>,
    records: Vec<VerifiedRecordBytes>,
    bytes: usize,
}
impl ActiveRecordByteCache {
    fn remember(&mut self, attempt: u64, record: Option<VerifiedRecordBytes>) {
        if self.attempt != Some(attempt) {
            *self = Self {
                attempt: Some(attempt),
                ..Self::default()
            };
        }
        if let Some(record) = record
            && self.records.len() < 4
            && let Some(total) = self
                .bytes
                .checked_add(record.bytes.len())
                .filter(|total| *total <= ACTIVE_RECORD_CACHE_BYTES)
        {
            self.bytes = total;
            self.records.push(record);
        }
    }
    fn lookup(
        &self,
        attempt: u64,
        directory: &Path,
        pin: &FilePin,
        spec: &CurrentMixedRunSpec,
    ) -> Option<&[u8]> {
        if self.attempt != Some(attempt) {
            return None;
        }
        self.records
            .iter()
            .find(|record| {
                record.directory == directory
                    && record.pin == *pin
                    && record.compressed == spec.record_storage.is_some()
            })
            .map(|record| record.bytes.as_slice())
    }
    fn require(
        &self,
        attempt: u64,
        directory: &Path,
        pin: &FilePin,
        spec: &CurrentMixedRunSpec,
    ) -> Result<(), String> {
        if let Some(expected) = self.lookup(attempt, directory, pin, spec) {
            let path = directory.join(&pin.name);
            let actual = if spec.record_storage.is_some() {
                read_mixed_stored_record(&path, pin.bytes)?
            } else {
                read_bounded(&path, pin.bytes)?
            };
            if actual.len() != pin.bytes || actual.as_slice() != expected {
                return Err("Current mixed active record bytes changed".into());
            }
            Ok(())
        } else {
            read_physical_pin(directory, pin, spec).map(|_| ())
        }
    }
}
fn read_record(
    directory: &Path,
    pin: &RecordPin,
    member: usize,
    spec: &CurrentMixedRunSpec,
) -> Result<Vec<u8>, String> {
    pin.validate(member, spec)?;
    match pin {
        RecordPin::Raw(pin) => read_pin(directory, pin, MAX_RECORD_BYTES),
        RecordPin::Compressed(pin) => {
            let stored = read_physical_pin(directory, &pin.stored, spec)?;
            record_storage::decode(&stored, pin.raw_bytes, &pin.raw_sha256)
        }
    }
}
fn read_physical_pin(
    directory: &Path,
    pin: &FilePin,
    spec: &CurrentMixedRunSpec,
) -> Result<Vec<u8>, String> {
    if pin.name.starts_with("record-")
        && !(0..4).any(|member| pin.name == record_name(member, spec))
    {
        return Err("Current mixed record filename/storage mode differs".into());
    }
    if pin.name.starts_with("record-") && spec.record_storage.is_some() {
        if !pin.name.ends_with(".json.gz") {
            return Err("Current mixed compressed record filename differs".into());
        }
        leaf(pin, &pin.name, MAX_STORED_RECORD_BYTES)?;
        let raw = read_mixed_stored_record(&directory.join(&pin.name), pin.bytes)?;
        if raw.len() != pin.bytes || sha(&raw) != pin.sha256 {
            return Err("Current mixed stored bytes/SHA differ".into());
        }
        Ok(raw)
    } else {
        read_pin(directory, pin, pin.bytes)
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct CurrentMixedRunSpec {
    schema: String,
    actor: serde_json::Value,
    critic: serde_json::Value,
    max_attempts: u64,
    max_successful_updates: u64,
    max_families: usize,
    max_journal_entries: u64,
    max_metadata_bytes: u64,
    max_output_bytes: u64,
    max_callbacks: u64,
    max_candidate_rows: u64,
    max_search_steps: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    record_storage: Option<RecordStorageSpec>,
}
impl CurrentMixedRunSpec {
    pub fn from_json(raw: &[u8]) -> Result<Self, String> {
        let value = unique_json(raw, SPEC_CAP)?;
        if value
            .get("recordStorage")
            .is_some_and(serde_json::Value::is_null)
        {
            return Err("Current mixed explicit null storage mode".into());
        }
        let s: Self = serde_json::from_value(value).map_err(|e| e.to_string())?;
        s.validate()?;
        Ok(s)
    }
    fn actor(&self) -> Result<OneStepConfig, String> {
        OneStepConfig::from_checkpoint(self.actor.clone())
    }
    fn critic(&self) -> Result<ResidualFitConfig, String> {
        ResidualFitConfig::from_checkpoint(self.critic.clone())
    }
    fn checksum(&self) -> Result<String, String> {
        hash(
            if self.record_storage.is_some() {
                b"tzolkin-current-mixed-durable-spec-v3\0"
            } else {
                b"tzolkin-current-mixed-durable-spec-v2\0"
            },
            self,
        )
    }
    fn journal_schema(&self) -> &'static str {
        if self.record_storage.is_some() {
            COMPRESSED_JOURNAL_SCHEMA
        } else {
            JOURNAL_SCHEMA
        }
    }
    fn check_storage(&self, raw: u64, stored: u64, metadata: u64) -> Result<(), String> {
        if raw
            .checked_add(metadata)
            .is_none_or(|n| n > self.max_output_bytes)
            || metadata > self.max_metadata_bytes
        {
            return Err("Current mixed logical/metadata storage budget consumed".into());
        }
        if let Some(storage) = &self.record_storage {
            if stored > storage.max_stored_bytes
                || stored
                    .checked_add(metadata)
                    .and_then(|n| n.checked_add(storage.max_temporary_bytes))
                    .is_none_or(|n| n > storage.max_disk_bytes)
            {
                return Err("Current mixed stored/temporary/disk storage budget consumed".into());
            }
        } else if raw != stored {
            return Err("Current mixed raw storage counters differ".into());
        }
        Ok(())
    }
    fn check_whole_four_storage(&self, raw: u64, stored: u64, metadata: u64) -> Result<(), String> {
        if self.record_storage.is_some() {
            self.check_storage(
                raw.checked_add(4 * MAX_RECORD_BYTES as u64)
                    .ok_or("Current mixed logical reserve overflow")?,
                stored
                    .checked_add(4 * MAX_STORED_RECORD_BYTES as u64)
                    .ok_or("Current mixed stored reserve overflow")?,
                metadata
                    .checked_add(METADATA_HEADROOM)
                    .ok_or("Current mixed metadata reserve overflow")?,
            )?;
        }
        Ok(())
    }
    fn validate(&self) -> Result<(), String> {
        self.actor()?;
        self.critic()?;
        match (&self.record_storage, self.schema.as_str()) {
            (None, SPEC_SCHEMA) => {}
            (Some(storage), COMPRESSED_SPEC_SCHEMA) => storage.validate()?,
            _ => return Err("Current mixed run schema/storage mode differs".into()),
        }
        if !(1..=MAX_ATTEMPTS).contains(&self.max_attempts)
            || !(1..=self.max_attempts).contains(&self.max_successful_updates)
            || !(4..=MAX_FAMILIES).contains(&self.max_families)
            || self.max_journal_entries < 1 + EVENTS_PER_ATTEMPT * self.max_attempts
            || self.max_journal_entries > 1 + EVENTS_PER_ATTEMPT * MAX_ATTEMPTS
            || !(32 * 1024 * 1024..=2 * 1024 * 1024 * 1024).contains(&self.max_metadata_bytes)
            || !(512 * 1024 * 1024..=64 * 1024 * 1024 * 1024).contains(&self.max_output_bytes)
            || self.max_callbacks == 0
            || self.max_candidate_rows == 0
        {
            return Err("Current mixed finite pilot limits".into());
        }
        Ok(())
    }
}
#[derive(Clone, Copy, Debug, Default, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct Work {
    callbacks: u64,
    rows: u64,
    search: u64,
    bytes: u64,
}
impl Work {
    fn add(self, other: Self) -> Result<Self, String> {
        Ok(Self {
            callbacks: self
                .callbacks
                .checked_add(other.callbacks)
                .ok_or("Callback budget overflow")?,
            rows: self
                .rows
                .checked_add(other.rows)
                .ok_or("Candidate budget overflow")?,
            search: self
                .search
                .checked_add(other.search)
                .ok_or("Search budget overflow")?,
            bytes: self
                .bytes
                .checked_add(other.bytes)
                .ok_or("Output budget overflow")?,
        })
    }
    fn checked(self, s: &CurrentMixedRunSpec) -> Result<(), String> {
        if self.callbacks > s.max_callbacks
            || self.rows > s.max_candidate_rows
            || self.search > s.max_search_steps
            || self.bytes > s.max_output_bytes
        {
            return Err("Current mixed reserved aggregate work exceeded".into());
        }
        Ok(())
    }
}
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct Reservation {
    attempt: u64,
    parent: CurrentMixedParentIdentity,
    collection_spec: FilePin,
    plan: FilePin,
    plan_checksum: String,
    configurations: [Config; 4],
    families: [String; 4],
    work: Work,
}
#[derive(Clone, Copy, Debug, Serialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
enum Phase {
    Ready,
    Reserved,
    Collecting(usize),
    Collected(usize),
    Auditing(usize),
    Audited(usize),
    Updating,
    FailedReturned,
    Stopped,
}
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(tag = "kind", rename_all = "camelCase", deny_unknown_fields)]
enum Resolution {
    Updated {
        report: FilePin,
        artifact: FilePin,
        report_checksum: String,
        artifact_checksum: String,
    },
    NoChange {
        report: FilePin,
        report_checksum: String,
    },
    Failed {
        error: String,
    },
    Abandoned {
        reason: String,
    },
}
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(tag = "kind", rename_all = "camelCase", deny_unknown_fields)]
enum Event {
    Start {
        run_spec: FilePin,
        parent: Box<CurrentMixedParentIdentity>,
        prefix: [u64; 2],
        next: u64,
        consumed: Vec<String>,
    },
    SegmentStart {
        run_spec: FilePin,
        binding: Box<segment::SegmentBinding>,
    },
    ReserveWhole4 {
        reservation: Box<Reservation>,
    },
    CollectStart {
        member: usize,
    },
    CollectReturn {
        member: usize,
        record: Option<RecordPin>,
        complete: bool,
        error: Option<String>,
    },
    AuditStart {
        member: usize,
    },
    AuditReturn {
        member: usize,
        passed: bool,
        episode_checksum: Option<String>,
        error: Option<String>,
    },
    GradientStart {
        cohort: FilePin,
        cohort_checksum: String,
        episode_checksums: [String; 4],
        counts: [usize; 3],
    },
    Resolve {
        resolution: Resolution,
    },
}
#[derive(Deserialize, Serialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct Entry {
    schema: String,
    sequence: u64,
    previous: String,
    root: String,
    spec: String,
    pool: String,
    event: Event,
}
struct BatchPlan {
    plan: PlannedCurrentMixedCohort,
    collection_raw: Vec<u8>,
    plan_raw: Vec<u8>,
    reservation: Reservation,
}

/// Sealed learner-only counts from one validated GradientStart. Candidate rows
/// and singletons are separate diagnostics; neither is an extra callback. These
/// counts do not measure audit work, all-seat decisions or native forward calls.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct CurrentMixedOwnCounts {
    pub callbacks: u64,
    pub candidate_rows: u64,
    pub singletons: u64,
}

/// Resolved whole-four cohorts. Missing counts mean that
/// no GradientStart was saved, not that the cohort had zero learner callbacks.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct CurrentMixedResolvedOwnCounts {
    pub cohorts: u64,
    pub cohorts_without_counts: u64,
    pub own: CurrentMixedOwnCounts,
}
impl CurrentMixedResolvedOwnCounts {
    fn add(&mut self, counts: Option<CurrentMixedOwnCounts>) {
        // At most 64 local attempts, each bounded by the actor/critic configs.
        self.cohorts += 1;
        if let Some(counts) = counts {
            self.own.callbacks += counts.callbacks;
            self.own.candidate_rows += counts.candidate_rows;
            self.own.singletons += counts.singletons;
        } else {
            self.cohorts_without_counts += 1;
        }
    }
}

/// Output-only resolved counters. Only Updated contributes to the updated bucket;
/// NoChange, failed and abandoned cohorts never imply an admitted numeric step.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct CurrentMixedOwnDecisionCounters {
    pub updated: CurrentMixedResolvedOwnCounts,
    pub no_change: CurrentMixedResolvedOwnCounts,
    pub failed: CurrentMixedResolvedOwnCounts,
    pub abandoned: CurrentMixedResolvedOwnCounts,
}
impl CurrentMixedOwnDecisionCounters {
    fn checked_add(self, other: Self) -> Result<Self, String> {
        fn bucket(
            a: CurrentMixedResolvedOwnCounts,
            b: CurrentMixedResolvedOwnCounts,
        ) -> Result<CurrentMixedResolvedOwnCounts, String> {
            let sum = |a: u64, b: u64| {
                a.checked_add(b)
                    .ok_or_else(|| "Cumulative current mixed own counters overflow".to_owned())
            };
            Ok(CurrentMixedResolvedOwnCounts {
                cohorts: sum(a.cohorts, b.cohorts)?,
                cohorts_without_counts: sum(a.cohorts_without_counts, b.cohorts_without_counts)?,
                own: CurrentMixedOwnCounts {
                    callbacks: sum(a.own.callbacks, b.own.callbacks)?,
                    candidate_rows: sum(a.own.candidate_rows, b.own.candidate_rows)?,
                    singletons: sum(a.own.singletons, b.own.singletons)?,
                },
            })
        }
        Ok(Self {
            updated: bucket(self.updated, other.updated)?,
            no_change: bucket(self.no_change, other.no_change)?,
            failed: bucket(self.failed, other.failed)?,
            abandoned: bucket(self.abandoned, other.abandoned)?,
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "camelCase")]
pub enum CurrentMixedOwnResolution {
    Updated,
    NoChange,
    Failed,
    Abandoned,
}

/// Borrows no policy authority. Counts remain known for an abandoned in-flight
/// gradient, but do not establish whether its numeric work returned or applied.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct CurrentMixedResolvedOwnReceipt {
    pub attempt: u64,
    pub outcome: CurrentMixedOwnResolution,
    pub own: Option<CurrentMixedOwnCounts>,
}

struct Progress {
    started: bool,
    phase: Phase,
    base: u64,
    segment_start: Option<segment::SegmentBinding>,
    prefix: [u64; 2],
    next: u64,
    consumed: BTreeSet<String>,
    external_unavailable: BTreeSet<String>,
    attempts: u64,
    successful: u64,
    imported: u64,
    current_checksum: String,
    work: Work,
    active: Option<Reservation>,
    collect: [Option<RecordPin>; 4],
    audited: [Option<String>; 4],
    cohort: Option<FilePin>,
    unknown_work: bool,
    outcome: Option<String>,
    own_decisions: CurrentMixedOwnDecisionCounters,
    active_own: Option<CurrentMixedOwnCounts>,
    last_resolved_own: Option<CurrentMixedResolvedOwnReceipt>,
}
impl Progress {
    fn parent_count(&self) -> Result<u64, String> {
        if !(1..=source::MAX_MIXED_UPDATES).contains(&self.base) {
            return Err("Current mixed base count is outside the owner ceiling".into());
        }
        self.base
            .checked_add(self.successful)
            .filter(|&n| n <= source::MAX_MIXED_UPDATES)
            .ok_or_else(|| "Current mixed parent count overflow or owner ceiling".into())
    }
    fn next_parent_count(&self) -> Result<u64, String> {
        self.parent_count()?
            .checked_add(1)
            .filter(|&n| n <= source::MAX_MIXED_UPDATES)
            .ok_or_else(|| "Current mixed successful-update ceiling".into())
    }
    fn require_ready_continuation(&self, blocked: bool, has_head: bool) -> Result<(), String> {
        if !self.started
            || !has_head
            || self.attempts == 0
            || blocked
            || self.phase != Phase::Ready
            || self.unknown_work
            || !matches!(self.outcome.as_deref(), Some("updated" | "noChange"))
        {
            return Err("Current mixed continuation requires a resolved Ready session".into());
        }
        Ok(())
    }
    fn require_stopped_pool_fork(&self, blocked: bool, has_head: bool) -> Result<(), String> {
        if !self.started
            || !has_head
            || !blocked
            || self.phase != Phase::Stopped
            || self.attempts == 0
            || self.unknown_work
            || self.outcome.as_deref() != Some("failed")
            || self
                .active
                .as_ref()
                .is_none_or(|r| r.attempt.checked_add(1) != Some(self.attempts))
        {
            return Err("Pool fork requires the final known Failed stopped tip".into());
        }
        self.parent_count()?;
        Ok(())
    }
    fn initial(
        parent: &CurrentMixedParentIdentity,
        prefix: [u64; 2],
        next: u64,
        consumed: &BTreeSet<String>,
    ) -> Self {
        Self {
            started: false,
            phase: Phase::Ready,
            base: 1,
            segment_start: None,
            prefix,
            next,
            consumed: consumed.clone(),
            external_unavailable: BTreeSet::new(),
            attempts: 0,
            successful: 0,
            imported: parent.imported_update_count(),
            current_checksum: parent.artifact_checksum().into(),
            work: Work::default(),
            active: None,
            collect: std::array::from_fn(|_| None),
            audited: std::array::from_fn(|_| None),
            cohort: None,
            unknown_work: false,
            outcome: None,
            own_decisions: CurrentMixedOwnDecisionCounters::default(),
            active_own: None,
            last_resolved_own: None,
        }
    }
    fn check(&self, e: &Event, s: &CurrentMixedRunSpec) -> Result<(), String> {
        let parent_count = self.parent_count()?;
        let fail = || Err("Invalid current mixed journal transition".into());
        match e {
            Event::Start {
                run_spec,
                parent,
                prefix,
                next,
                consumed,
            } => {
                leaf(run_spec, "run-spec.json", SPEC_CAP)?;
                parent.validate()?;
                if self.started
                    || self.segment_start.is_some()
                    || self.attempts != 0
                    || self.phase != Phase::Ready
                    || parent_count != 1 // Legacy Start remains genuine Count1 only.
                    || parent.mixed_update_count() != parent_count
                    || parent.imported_update_count() != self.imported
                    || parent.artifact_checksum() != self.current_checksum
                    || *prefix != self.prefix
                    || *next != self.next
                    || consumed.windows(2).any(|v| v[0] >= v[1])
                    || consumed.iter().cloned().collect::<BTreeSet<_>>() != self.consumed
                {
                    return fail();
                }
            }
            Event::SegmentStart { run_spec, binding } => {
                leaf(run_spec, "run-spec.json", SPEC_CAP)?;
                if self.started
                    || self.attempts != 0
                    || self.phase != Phase::Ready
                    || self.segment_start.as_ref() != Some(binding.as_ref())
                    || binding.parent.mixed_update_count() != parent_count
                    || binding.parent.imported_update_count() != self.imported
                    || binding.parent.artifact_checksum() != self.current_checksum
                    || binding.prefix != self.prefix
                    || binding.next != self.next
                    || binding.consumed.iter().cloned().collect::<BTreeSet<_>>() != self.consumed
                    || binding.external_families() != self.external_unavailable
                {
                    return fail();
                }
            }
            Event::ReserveWhole4 { reservation: r } => {
                self.next_parent_count()?;
                r.parent.validate()?;
                leaf(&r.collection_spec, "collection-spec.json", SPEC_CAP)?;
                leaf(&r.plan, "plan.json", HEAD_CAP)?;
                if !self.started
                    || self.phase != Phase::Ready
                    || self.attempts >= s.max_attempts
                    || self.successful >= s.max_successful_updates
                    || r.attempt != self.attempts
                    || r.parent.artifact_checksum() != self.current_checksum
                    || r.parent.mixed_update_count() != parent_count
                    || r.parent.imported_update_count() != self.imported
                    || !checkpoint_digest(&r.plan_checksum)
                    || self
                        .consumed
                        .len()
                        .checked_add(self.external_unavailable.len())
                        .and_then(|n| n.checked_add(4))
                        .is_none_or(|n| n > s.max_families)
                    || self.next.checked_add(4).is_none()
                {
                    return fail();
                }
                let configs = r
                    .configurations
                    .iter()
                    .map(Config::checked_config)
                    .collect::<Result<Vec<_>, _>>()?;
                if configs.iter().filter(|c| c.players() == 3).count() != 2
                    || configs.iter().filter(|c| c.players() == 4).count() != 2
                {
                    return fail();
                }
                for (i, c) in r.configurations.iter().enumerate() {
                    if c.sampling_seed != format!("{:016x}", self.prefix[0])
                        || c.replicate_ordinal != format!("{:016x}", self.prefix[1])
                        || c.episode_ordinal != format!("{:016x}", self.next + i as u64)
                        || seed_family_id(c.environment_seed) != r.families[i]
                        || self.consumed.contains(&r.families[i])
                        || self.external_unavailable.contains(&r.families[i])
                        || r.families[..i].contains(&r.families[i])
                        || split_for_family(&r.families[i])? != DatasetSplit::Train
                    {
                        return fail();
                    }
                }
                self.work.add(r.work)?.checked(s)?;
            }
            Event::CollectStart { member: i }
                if *i < 4
                    && (self.phase == Phase::Reserved && *i == 0
                        || *i > 0 && self.phase == Phase::Collected(i - 1)) => {}
            Event::CollectReturn {
                member: i,
                record,
                complete,
                error,
            } if self.phase == Phase::Collecting(*i) => {
                if let Some(p) = record {
                    p.validate(*i, s)?;
                }
                if *complete != error.is_none()
                    || *complete && record.is_none()
                    || error
                        .as_ref()
                        .is_some_and(|e| e.is_empty() || e.len() > 1024)
                {
                    return fail();
                }
            }
            Event::AuditStart { member: i }
                if *i < 4
                    && (self.phase == Phase::Collected(3) && *i == 0
                        || *i > 0 && self.phase == Phase::Audited(i - 1)) => {}
            Event::AuditReturn {
                member: i,
                passed,
                episode_checksum,
                error,
            } if self.phase == Phase::Auditing(*i) => {
                if *passed != error.is_none()
                    || *passed != episode_checksum.is_some()
                    || episode_checksum
                        .as_ref()
                        .is_some_and(|h| !checkpoint_digest(h))
                    || error
                        .as_ref()
                        .is_some_and(|e| e.is_empty() || e.len() > 1024)
                {
                    return fail();
                }
            }
            Event::GradientStart {
                cohort,
                cohort_checksum,
                episode_checksums,
                counts,
            } if self.phase == Phase::Audited(3) => {
                leaf(cohort, "cohort.json", HEAD_CAP)?;
                if !checkpoint_digest(cohort_checksum)
                    || counts[0] == 0
                    || counts[2] > counts[0]
                    || episode_checksums
                        .iter()
                        .enumerate()
                        .any(|(i, h)| self.audited[i].as_ref() != Some(h))
                {
                    return fail();
                }
                s.critic()?.check_callbacks(counts[0])?;
                if counts[1] > s.actor()?.max_candidate_rows() {
                    return fail();
                }
            }
            Event::Resolve { resolution } => match resolution {
                Resolution::Updated {
                    report,
                    artifact,
                    report_checksum,
                    artifact_checksum,
                } if self.phase == Phase::Updating => {
                    self.next_parent_count()?;
                    leaf(report, "report.json", HEAD_CAP)?;
                    leaf(artifact, "artifact.json", ARTIFACT_CAP)?;
                    if !checkpoint_digest(report_checksum)
                        || !checkpoint_digest(artifact_checksum)
                        || artifact_checksum == &self.current_checksum
                    {
                        return fail();
                    }
                }
                Resolution::NoChange {
                    report,
                    report_checksum,
                } if self.phase == Phase::Updating => {
                    leaf(report, "report.json", HEAD_CAP)?;
                    if !checkpoint_digest(report_checksum) {
                        return fail();
                    }
                }
                Resolution::Failed { error }
                    if matches!(
                        self.phase,
                        Phase::FailedReturned | Phase::Updating | Phase::Audited(3)
                    ) =>
                {
                    if error.is_empty() || error.len() > 1024 {
                        return fail();
                    }
                }
                Resolution::Abandoned { reason }
                    if !matches!(self.phase, Phase::Ready | Phase::Stopped) =>
                {
                    if reason.is_empty() || reason.len() > 1024 {
                        return fail();
                    }
                }
                _ => return fail(),
            },
            _ => return fail(),
        }
        Ok(())
    }
    fn apply(&mut self, e: &Event, s: &CurrentMixedRunSpec) -> Result<(), String> {
        self.check(e, s)?;
        if let Event::Resolve { resolution } = e {
            let (outcome, bucket) = match resolution {
                Resolution::Updated { .. } => (
                    CurrentMixedOwnResolution::Updated,
                    &mut self.own_decisions.updated,
                ),
                Resolution::NoChange { .. } => (
                    CurrentMixedOwnResolution::NoChange,
                    &mut self.own_decisions.no_change,
                ),
                Resolution::Failed { .. } => (
                    CurrentMixedOwnResolution::Failed,
                    &mut self.own_decisions.failed,
                ),
                Resolution::Abandoned { .. } => (
                    CurrentMixedOwnResolution::Abandoned,
                    &mut self.own_decisions.abandoned,
                ),
            };
            bucket.add(self.active_own);
            self.last_resolved_own = Some(CurrentMixedResolvedOwnReceipt {
                attempt: self.attempts - 1,
                outcome,
                own: self.active_own,
            });
        }
        match e {
            Event::Start { .. } | Event::SegmentStart { .. } => {
                self.started = true;
            }
            Event::ReserveWhole4 { reservation: r } => {
                self.attempts += 1;
                self.next += 4;
                self.consumed.extend(r.families.iter().cloned());
                self.work = self.work.add(r.work)?;
                self.active = Some(r.as_ref().clone());
                self.phase = Phase::Reserved;
                self.collect = std::array::from_fn(|_| None);
                self.audited = std::array::from_fn(|_| None);
                self.cohort = None;
                self.outcome = None;
                self.active_own = None;
            }
            Event::CollectStart { member } => {
                self.phase = Phase::Collecting(*member);
                self.unknown_work = true;
            }
            Event::CollectReturn {
                member,
                record,
                complete,
                ..
            } => {
                self.collect[*member] = record.clone();
                self.unknown_work = false;
                self.phase = if *complete {
                    Phase::Collected(*member)
                } else {
                    Phase::FailedReturned
                };
            }
            Event::AuditStart { member } => {
                self.phase = Phase::Auditing(*member);
                self.unknown_work = true;
            }
            Event::AuditReturn {
                member,
                passed,
                episode_checksum,
                ..
            } => {
                self.audited[*member] = episode_checksum.clone();
                self.unknown_work = false;
                self.phase = if *passed {
                    Phase::Audited(*member)
                } else {
                    Phase::FailedReturned
                };
            }
            Event::GradientStart { cohort, counts, .. } => {
                self.cohort = Some(cohort.clone());
                self.active_own = Some(CurrentMixedOwnCounts {
                    callbacks: counts[0] as u64,
                    candidate_rows: counts[1] as u64,
                    singletons: counts[2] as u64,
                });
                self.phase = Phase::Updating;
                self.unknown_work = true;
            }
            Event::Resolve { resolution } => match resolution {
                Resolution::Updated {
                    artifact_checksum, ..
                } => {
                    self.successful += 1;
                    self.current_checksum = artifact_checksum.clone();
                    self.outcome = Some("updated".into());
                    self.phase = Phase::Ready;
                    self.unknown_work = false;
                }
                Resolution::NoChange { .. } => {
                    self.outcome = Some("noChange".into());
                    self.phase = Phase::Ready;
                    self.unknown_work = false;
                }
                Resolution::Failed { .. } => {
                    self.outcome = Some("failed".into());
                    self.phase = Phase::Stopped;
                    self.unknown_work = false;
                }
                Resolution::Abandoned { .. } => {
                    self.outcome = Some("abandoned".into());
                    self.phase = Phase::Stopped;
                }
            },
        }
        Ok(())
    }
}

/// Only this module can construct a proof after the verified durable Resolve.
/// Crate-private consumers cannot mint it from a wire receipt or artifact DTO.
pub(crate) struct DurableMixedResolve {
    promotion: mixed::PreparedContinuedPromotion,
}
impl DurableMixedResolve {
    pub(crate) fn into_promotion(self) -> mixed::PreparedContinuedPromotion {
        self.promotion
    }
}

/// Owns original admission once and one current owner. Single-writer local
/// persistence does not authenticate seed origins or prevent global forks.
pub struct CurrentMixedTrainingSession<'a> {
    root: AdmittedCurrentMixedRoot<'a>,
    original: OriginalLivePins,
    prepared: &'a crate::public_native::PreparedPublicPolicy,
    search: PreparedSearch,
    pool: String,
    spec: CurrentMixedRunSpec,
    root_checksum: String,
    progress: Progress,
    continued: Option<ContinuedMixedOwner>,
    directory: PathBuf,
    sequence: u64,
    head: Option<FilePin>,
    spec_pin: Option<FilePin>,
    owner_pins: Option<(PathBuf, FilePin, FilePin)>,
    references: BTreeMap<u64, Vec<FilePin>>,
    active_record_bytes: ActiveRecordByteCache,
    output_bytes: u64,
    metadata_bytes: u64,
    raw_record_bytes: u64,
    stored_record_bytes: u64,
    blocked: bool,
    segment: Option<segment::SegmentContext>,
}

/// Borrows only archival state; it cannot admit or expose a policy owner.
struct ArchiveView<'s> {
    directory: &'s Path,
    spec: &'s CurrentMixedRunSpec,
    journal_schema: &'static str,
    root_checksum: &'s str,
    pool: &'s str,
    sequence: u64,
    head: Option<&'s FilePin>,
    spec_pin: Option<&'s FilePin>,
    references: &'s BTreeMap<u64, Vec<FilePin>>,
    output_bytes: u64,
    metadata_bytes: u64,
    raw_record_bytes: u64,
    stored_record_bytes: u64,
}
impl ArchiveView<'_> {
    fn batch(&self, attempt: u64) -> PathBuf {
        self.directory.join(format!("batch-{attempt:016x}"))
    }
}

/// Consumes a fully closed Ready session without changing its actual owner,
/// reservations, cursor, specifications or spent budgets. No constructor from
/// wire metadata is provided. A successor consumes this token after preflight.
pub struct ReadyCurrentMixedContinuation<'a> {
    session: CurrentMixedTrainingSession<'a>,
}
impl ReadyCurrentMixedContinuation<'_> {
    pub fn current_policy(&self) -> Result<CurrentMixedPolicyHandle<'_>, String> {
        self.session.current_policy()
    }
    pub fn status(&self) -> Result<CurrentMixedSessionStatus, String> {
        self.current_policy()?;
        Ok(self.session.status())
    }
    /// All consumed reservations, including NoChange, remain unavailable.
    pub fn reserved_families(&self) -> &BTreeSet<String> {
        self.session.reserved_families()
    }
    pub fn last_resolved_own_receipt(&self) -> Option<&CurrentMixedResolvedOwnReceipt> {
        self.session.last_resolved_own_receipt()
    }
    pub fn current_policy_for_arena(&self) -> Result<BorrowedArenaPolicy<'_>, String> {
        self.session.current_policy_for_arena()
    }
    pub fn original_policy_for_arena(&self) -> Result<BorrowedArenaPolicy<'_>, String> {
        self.session.original_policy_for_arena()
    }
}

#[derive(Debug, Default, Serialize)]
#[serde(rename_all = "camelCase")]
struct StageTiming {
    entered: u64,
    returned: u64,
    returned_errors: u64,
    wall_seconds: f64,
}

#[derive(Debug, Default, Serialize)]
#[serde(rename_all = "camelCase")]
struct AttemptStages {
    whole_call_inclusive: StageTiming,
    preflight: StageTiming,
    reservation_publication: StageTiming,
    collection_journal_inclusive: [StageTiming; 4],
    collection_guards_and_preparation: [StageTiming; 4],
    collection_native: [StageTiming; 4],
    record_publication: [StageTiming; 4],
    audit_record_read_and_check: [StageTiming; 4],
    audit_journal_inclusive: [StageTiming; 4],
    audit_guards_and_preparation: [StageTiming; 4],
    audit_native: [StageTiming; 4],
    cohort_publication: StageTiming,
    gradient_start: StageTiming,
    numeric_guards_and_preparation: StageTiming,
    numeric_preparation_inclusive: StageTiming,
    actor_encode: StageTiming,
    actor_scalar_forward: StageTiming,
    actor_trace_and_cotangent: StageTiming,
    actor_frozen_baseline: StageTiming,
    actor_backward: StageTiming,
    actor_aggregation: StageTiming,
    actor_rounded_proposal: StageTiming,
    critic_fit: StageTiming,
    proposal_publication_readback: StageTiming,
    resolve_and_admit: StageTiming,
    final_guards: StageTiming,
}

/// Current-process diagnostic output only. It is never journaled or restored.
/// Whole-call and numerical preparation spans include their named inner spans;
/// collection/audit include their own inference and replay work. Do not sum them.
/// The collection/audit journal, preparation and record-read groups cover
/// separate surrounding calls, outside the existing native/publication spans.
/// Journal groups include live verification and metadata serialization/readback;
/// preparation groups include live verification and handle/roster/config work.
/// These are inclusive operation groups, not pure disk or hash measurements.
/// Zero entered calls mean unobserved work, not a measured zero-duration stage.
/// Returned-error counts describe `Result::Err`; a returned stopped status is
/// recorded separately in the resolution and stopped fields. The whole-call
/// span covers the update body, excluding the profile's before/after snapshots.
#[derive(Debug, Default, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct CurrentMixedAttemptTimings {
    call_observed: bool,
    attempts_before: u64,
    attempts_after: u64,
    head_sha256_before: String,
    head_sha256_after: String,
    reservation_charged: bool,
    new_resolution: Option<CurrentMixedOwnResolution>,
    phase_after: Option<Phase>,
    stopped_after: bool,
    unknown_work_after: bool,
    returned_error: bool,
    stages: AttemptStages,
}
impl CurrentMixedAttemptTimings {
    fn stage(&mut self, phase: TimedPhase) -> &mut StageTiming {
        match phase {
            TimedPhase::WholeCall => &mut self.stages.whole_call_inclusive,
            TimedPhase::Preflight => &mut self.stages.preflight,
            TimedPhase::ReservationPublication => &mut self.stages.reservation_publication,
            TimedPhase::CollectionJournal(i) => &mut self.stages.collection_journal_inclusive[i],
            TimedPhase::CollectionPreparation(i) => {
                &mut self.stages.collection_guards_and_preparation[i]
            }
            TimedPhase::Collection(i) => &mut self.stages.collection_native[i],
            TimedPhase::RecordPublication(i) => &mut self.stages.record_publication[i],
            TimedPhase::AuditRecordRead(i) => &mut self.stages.audit_record_read_and_check[i],
            TimedPhase::AuditJournal(i) => &mut self.stages.audit_journal_inclusive[i],
            TimedPhase::AuditPreparation(i) => &mut self.stages.audit_guards_and_preparation[i],
            TimedPhase::Audit(i) => &mut self.stages.audit_native[i],
            TimedPhase::CohortPublication => &mut self.stages.cohort_publication,
            TimedPhase::GradientStart => &mut self.stages.gradient_start,
            TimedPhase::NumericPreparationControl => {
                &mut self.stages.numeric_guards_and_preparation
            }
            TimedPhase::NumericPreparation => &mut self.stages.numeric_preparation_inclusive,
            TimedPhase::ActorEncode => &mut self.stages.actor_encode,
            TimedPhase::ActorForward => &mut self.stages.actor_scalar_forward,
            TimedPhase::ActorTrace => &mut self.stages.actor_trace_and_cotangent,
            TimedPhase::ActorBaseline => &mut self.stages.actor_frozen_baseline,
            TimedPhase::ActorBackward => &mut self.stages.actor_backward,
            TimedPhase::ActorAggregate => &mut self.stages.actor_aggregation,
            TimedPhase::ActorProposal => &mut self.stages.actor_rounded_proposal,
            TimedPhase::CriticFit => &mut self.stages.critic_fit,
            TimedPhase::ProposalReadback => &mut self.stages.proposal_publication_readback,
            TimedPhase::Resolve => &mut self.stages.resolve_and_admit,
            TimedPhase::FinalGuards => &mut self.stages.final_guards,
        }
    }
}
impl PhaseTimer for CurrentMixedAttemptTimings {
    type Span = (TimedPhase, Instant);
    fn enter(&mut self, phase: TimedPhase) -> Self::Span {
        self.stage(phase).entered += 1;
        (phase, Instant::now())
    }
    fn returned(&mut self, (phase, started): Self::Span, error: bool) {
        let seconds = started.elapsed().as_secs_f64();
        let stage = self.stage(phase);
        stage.returned += 1;
        stage.returned_errors += u64::from(error);
        stage.wall_seconds += seconds;
    }
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct CurrentMixedSessionStatus {
    schema: &'static str,
    head_file: String,
    head_sha256: String,
    imported_update_count: u64,
    mixed_update_count: u64,
    attempts: u64,
    next_episode_ordinal: u64,
    reserved_family_count: usize,
    phase: Phase,
    stopped: bool,
    unknown_work_possible: bool,
    outcome: Option<String>,
    own_decisions: CurrentMixedOwnDecisionCounters,
    pending_own: Option<CurrentMixedOwnCounts>,
    #[serde(skip_serializing_if = "Option::is_none")]
    storage_usage: Option<StorageUsage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    segment_usage: Option<segment::SegmentUsage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    external_unavailable_family_count: Option<usize>,
}
#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct StorageUsage {
    logical_raw_record_bytes: u64,
    stored_record_bytes: u64,
    metadata_bytes: u64,
    retained_bytes: u64,
    temporary_reserve_bytes: u64,
}
impl CurrentMixedSessionStatus {
    pub fn head_file(&self) -> &str {
        &self.head_file
    }
    pub fn head_sha256(&self) -> &str {
        &self.head_sha256
    }
    pub fn mixed_update_count(&self) -> u64 {
        self.mixed_update_count
    }
    pub fn imported_update_count(&self) -> u64 {
        self.imported_update_count
    }
    pub fn next_episode_ordinal(&self) -> u64 {
        self.next_episode_ordinal
    }
    pub fn attempts(&self) -> u64 {
        self.attempts
    }
    /// Local-segment totals rebuilt by the same validated live/restore reducer.
    pub fn own_decisions(&self) -> &CurrentMixedOwnDecisionCounters {
        &self.own_decisions
    }
    /// Known sealed counts for the unresolved attempt; not credited to totals.
    pub fn pending_own(&self) -> Option<&CurrentMixedOwnCounts> {
        self.pending_own.as_ref()
    }
}
impl<'a> CurrentMixedTrainingSession<'a> {
    fn journal_schema(&self) -> &'static str {
        if self.segment.is_some() {
            SEGMENT_JOURNAL_SCHEMA
        } else {
            self.spec.journal_schema()
        }
    }
    fn archive_view(&self) -> ArchiveView<'_> {
        ArchiveView {
            directory: &self.directory,
            spec: &self.spec,
            journal_schema: self.journal_schema(),
            root_checksum: &self.root_checksum,
            pool: &self.pool,
            sequence: self.sequence,
            head: self.head.as_ref(),
            spec_pin: self.spec_pin.as_ref(),
            references: &self.references,
            output_bytes: self.output_bytes,
            metadata_bytes: self.metadata_bytes,
            raw_record_bytes: self.raw_record_bytes,
            stored_record_bytes: self.stored_record_bytes,
        }
    }
    fn initial(
        root: AdmittedCurrentMixedRoot<'a>,
        spec: CurrentMixedRunSpec,
        pool: &CurrentMixedCollectionSpec,
        directory: PathBuf,
    ) -> Result<Self, String> {
        spec.validate()?;
        root.require_current()?; // Full original archive admission once.
        root.require_output(&directory)?;
        if root
            .consumed_families()
            .len()
            .checked_add(4)
            .is_none_or(|n| n > spec.max_families)
        {
            return Err("Current mixed inherited reservation ceiling".into());
        }
        let prepared = root.prepared()?;
        let original = root.live_pins()?;
        let parent = CurrentMixedPolicyHandle::from_initial(root.initial_owner()?)?;
        let progress = Progress::initial(
            parent.identity(),
            root.prefix(),
            root.next_episode_ordinal(),
            root.consumed_families(),
        );
        let root_checksum = hash(
            b"tzolkin-current-mixed-durable-root-v2\0",
            &(&original, parent.identity()),
        )?;
        Ok(Self {
            search: PreparedSearch::new(&pool.inner.wire.search)?,
            pool: pool.pool_checksum()?,
            root,
            original,
            prepared,
            spec,
            root_checksum,
            progress,
            continued: None,
            directory,
            sequence: 0,
            head: None,
            spec_pin: None,
            owner_pins: None,
            references: BTreeMap::new(),
            active_record_bytes: ActiveRecordByteCache::default(),
            output_bytes: 0,
            metadata_bytes: 0,
            raw_record_bytes: 0,
            stored_record_bytes: 0,
            blocked: false,
            segment: None,
        })
    }
    pub fn new(
        root: AdmittedCurrentMixedRoot<'a>,
        spec: CurrentMixedRunSpec,
        pool: &CurrentMixedCollectionSpec,
        output: &Path,
    ) -> Result<Self, String> {
        let directory = local(output, true)?;
        if directory.exists() {
            return Err("Current mixed durable output must be new".into());
        }
        let mut s = Self::initial(root, spec, pool, directory)?;
        let raw = bytes(&s.spec, SPEC_CAP)?;
        let initial = Event::Start {
            run_spec: file_pin("run-spec.json", &raw),
            parent: Box::new(s.handle()?.identity().clone()),
            prefix: s.progress.prefix,
            next: s.progress.next,
            consumed: s.progress.consumed.iter().cloned().collect(),
        };
        bytes(&initial, HEAD_CAP)?;
        fs::create_dir(&s.directory).map_err(|e| e.to_string())?;
        s.spec_pin = Some(s.publish_root("run-spec.json", &raw, SPEC_CAP)?);
        s.save(initial)?;
        Ok(s)
    }
    fn handle(&self) -> Result<CurrentMixedPolicyHandle<'_>, String> {
        let h = match &self.continued {
            Some(o) => CurrentMixedPolicyHandle::from_continued(o)?,
            None => CurrentMixedPolicyHandle::from_initial(self.root.initial_owner()?)?,
        };
        if h.identity().artifact_checksum() != self.progress.current_checksum
            || h.identity().mixed_update_count() != self.progress.parent_count()?
            || h.identity().imported_update_count() != self.progress.imported
        {
            return Err("Current mixed owner is not the actual reduced journal parent".into());
        }
        Ok(h)
    }
    pub fn current_policy(&self) -> Result<CurrentMixedPolicyHandle<'_>, String> {
        if self.blocked || self.progress.phase != Phase::Ready {
            return Err("Stopped/unresolved current mixed session cannot infer or continue".into());
        }
        self.require_live()?;
        self.handle()
    }
    /// Borrows the actual current owner for a development Arena comparison.
    /// The borrow prevents updating or consuming this session during evaluation.
    pub fn current_policy_for_arena(&self) -> Result<BorrowedArenaPolicy<'_>, String> {
        self.current_policy()?;
        match &self.continued {
            Some(owner) => Ok(BorrowedArenaPolicy::ContinuedMixed(owner)),
            None => Ok(BorrowedArenaPolicy::Mixed(self.root.initial_owner()?)),
        }
    }
    /// Borrows the unchanged original Mixed count1 baseline, including its critic
    /// lineage. Uses the same readiness and current-owner guards as evaluation
    /// of the current policy; stopped or unresolved sessions expose neither.
    /// The returned policy cannot outlive this session or overlap its mutation.
    pub fn original_policy_for_arena(&self) -> Result<BorrowedArenaPolicy<'_>, String> {
        self.current_policy()?;
        Ok(BorrowedArenaPolicy::Mixed(self.root.initial_owner()?))
    }
    pub fn continued_owner(&self) -> Option<&ContinuedMixedOwner> {
        self.continued.as_ref()
    }
    pub fn reserved_families(&self) -> &BTreeSet<String> {
        &self.progress.consumed
    }
    /// Caller-declared unavailable families, not native executions or update credit.
    pub fn external_unavailable_families(&self) -> &BTreeSet<String> {
        &self.progress.external_unavailable
    }
    /// Resolved learner-own totals across this session's genuine journal chain.
    /// Imported original training and unresolved counts are not credited. Semantic
    /// restore rebuilds the totals from predecessor reducers; reading them performs
    /// no I/O or policy admission and does not change local status serialization.
    pub fn cumulative_own_decisions(&self) -> Result<CurrentMixedOwnDecisionCounters, String> {
        self.segment
            .as_ref()
            .map_or(CurrentMixedOwnDecisionCounters::default(), |s| {
                s.inherited_own
            })
            .checked_add(self.progress.own_decisions)
    }
    /// Latest resolved local receipt, retained while a later attempt is pending.
    /// Reading this output metadata performs no archive scan or policy admission.
    pub fn last_resolved_own_receipt(&self) -> Option<&CurrentMixedResolvedOwnReceipt> {
        self.progress.last_resolved_own.as_ref()
    }
    pub fn status(&self) -> CurrentMixedSessionStatus {
        CurrentMixedSessionStatus {
            schema: self.journal_schema(),
            head_file: self
                .head
                .as_ref()
                .map(|p| p.name.clone())
                .unwrap_or_default(),
            head_sha256: self
                .head
                .as_ref()
                .map(|p| p.sha256.clone())
                .unwrap_or_default(),
            imported_update_count: self.progress.imported,
            mixed_update_count: self
                .progress
                .parent_count()
                .expect("validated private current mixed count"),
            attempts: self.progress.attempts,
            next_episode_ordinal: self.progress.next,
            reserved_family_count: self.progress.consumed.len(),
            phase: self.progress.phase,
            stopped: self.blocked || self.progress.phase == Phase::Stopped,
            unknown_work_possible: self.progress.unknown_work,
            outcome: self.progress.outcome.clone(),
            own_decisions: self.progress.own_decisions,
            pending_own: self
                .progress
                .outcome
                .is_none()
                .then_some(self.progress.active_own)
                .flatten(),
            storage_usage: self
                .spec
                .record_storage
                .as_ref()
                .map(|storage| StorageUsage {
                    logical_raw_record_bytes: self.raw_record_bytes,
                    stored_record_bytes: self.stored_record_bytes,
                    metadata_bytes: self.metadata_bytes,
                    retained_bytes: self.output_bytes,
                    temporary_reserve_bytes: storage.max_temporary_bytes,
                }),
            segment_usage: self.segment_usage(),
            external_unavailable_family_count: (!self.progress.external_unavailable.is_empty())
                .then_some(self.progress.external_unavailable.len()),
        }
    }
    fn batch(&self, attempt: u64) -> PathBuf {
        self.directory.join(format!("batch-{attempt:016x}"))
    }
    fn require_live(&self) -> Result<(), String> {
        for p in &self.original.files {
            read_pin(&self.original.directory, p, p.bytes)?;
        }
        if let Some(p) = &self.spec_pin {
            read_pin(&self.directory, p, SPEC_CAP)?;
        }
        if let Some(p) = &self.head {
            read_pin(&self.directory, p, HEAD_CAP)?;
        }
        if let Some(pins) = &self.owner_pins {
            require_owner_pins(pins)?;
        }
        if let Some(r) = &self.progress.active {
            for p in self.references.get(&r.attempt).into_iter().flatten() {
                self.active_record_bytes.require(
                    r.attempt,
                    &self.batch(r.attempt),
                    p,
                    &self.spec,
                )?;
            }
        }
        Ok(())
    }
    fn publish_root(&mut self, name: &str, raw: &[u8], cap: usize) -> Result<FilePin, String> {
        if raw.is_empty() || raw.len() > cap {
            return Err("Current mixed root publication cap".into());
        }
        let output = self
            .output_bytes
            .checked_add(raw.len() as u64)
            .ok_or("Current mixed storage overflow")?;
        let metadata = self
            .metadata_bytes
            .checked_add(raw.len() as u64)
            .ok_or("Current mixed metadata overflow")?;
        self.spec
            .check_storage(self.raw_record_bytes, self.stored_record_bytes, metadata)?;
        publish_mixed_metadata(&self.directory, name, raw)?;
        let p = file_pin(name, raw);
        read_pin(&self.directory, &p, cap)?;
        self.output_bytes = output;
        self.metadata_bytes = metadata;
        Ok(p)
    }
    fn publish_batch(
        &mut self,
        attempt: u64,
        name: &str,
        raw: &[u8],
        cap: usize,
    ) -> Result<FilePin, String> {
        self.require_live()?;
        let output = self
            .output_bytes
            .checked_add(raw.len() as u64)
            .ok_or("Current mixed storage overflow")?;
        let metadata = self
            .metadata_bytes
            .checked_add(raw.len() as u64)
            .ok_or("Current mixed metadata overflow")?;
        self.spec
            .check_storage(self.raw_record_bytes, self.stored_record_bytes, metadata)?;
        let refs = self.references.entry(attempt).or_default();
        if raw.is_empty()
            || raw.len() > cap
            || name.starts_with("record-")
            || refs.len() >= 9
            || refs.iter().any(|p| p.name == name)
        {
            return Err("Current mixed bounded batch publication".into());
        }
        let dir = self.batch(attempt);
        publish_mixed_metadata(&dir, name, raw)?;
        let p = file_pin(name, raw);
        read_pin(&dir, &p, cap)?;
        self.references.entry(attempt).or_default().push(p.clone());
        self.output_bytes = output;
        self.metadata_bytes = metadata;
        self.require_live()?;
        Ok(p)
    }
    fn publish_record(
        &mut self,
        attempt: u64,
        member: usize,
        raw: &[u8],
    ) -> Result<RecordPin, String> {
        self.require_live()?;
        let (stored, pin) = encode_record(raw, member, &self.spec)?;
        let logical = self
            .raw_record_bytes
            .checked_add(raw.len() as u64)
            .ok_or("Current mixed logical storage overflow")?;
        let physical = self
            .stored_record_bytes
            .checked_add(stored.len() as u64)
            .ok_or("Current mixed stored storage overflow")?;
        self.spec
            .check_storage(logical, physical, self.metadata_bytes)?;
        let refs = self.references.entry(attempt).or_default();
        if refs.len() >= 9 || refs.iter().any(|p| p.name == pin.file().name) {
            return Err("Current mixed repeated record output".into());
        }
        let directory = self.batch(attempt);
        let verified = publish_record_file(&directory, &stored, raw, &pin, member, &self.spec)?;
        self.references
            .entry(attempt)
            .or_default()
            .push(pin.file().clone());
        self.output_bytes = physical
            .checked_add(self.metadata_bytes)
            .ok_or("Current mixed retained storage overflow")?;
        self.raw_record_bytes = logical;
        self.stored_record_bytes = physical;
        self.active_record_bytes.remember(attempt, verified);
        self.require_live()?;
        Ok(pin)
    }
    fn save(&mut self, event: Event) -> Result<(), String> {
        self.progress.check(&event, &self.spec)?;
        self.require_live()?;
        let sequence = self
            .sequence
            .checked_add(1)
            .filter(|n| *n <= self.spec.max_journal_entries)
            .ok_or("Current mixed journal exhausted")?;
        let entry = Entry {
            schema: self.journal_schema().into(),
            sequence,
            previous: self
                .head
                .as_ref()
                .map(|p| p.sha256.clone())
                .unwrap_or_default(),
            root: self.root_checksum.clone(),
            spec: self.spec.checksum()?,
            pool: self.pool.clone(),
            event,
        };
        let raw = bytes(&entry, HEAD_CAP)?;
        let reserve = matches!(entry.event, Event::ReserveWhole4 { .. });
        if reserve {
            self.progress.apply(&entry.event, &self.spec)?;
        }
        let p = self.publish_root(&format!("head-{sequence:016x}.json"), &raw, HEAD_CAP)?;
        if !reserve {
            self.progress.apply(&entry.event, &self.spec)?;
        }
        self.sequence = sequence;
        self.head = Some(p);
        self.require_live()
    }
    fn rosters<'s>(
        &'s self,
        spec: &'s CurrentMixedCollectionSpec,
    ) -> Result<[FrozenMixedRoster<'s>; 4], String> {
        self.rosters_with_profile(spec, None)
    }
    fn rosters_with_profile<'s>(
        &'s self,
        spec: &'s CurrentMixedCollectionSpec,
        profile: Option<&SavedRosterProfile>,
    ) -> Result<[FrozenMixedRoster<'s>; 4], String> {
        let parent = self.handle()?;
        spec.inner
            .wire
            .rosters
            .iter()
            .enumerate()
            .map(|(member, seats)| {
                let seats = seats
                    .iter()
                    .map(|p| {
                        p.as_ref().map(|p| match p {
                            OpponentSpec::Heuristic => {
                                BorrowedArenaPolicy::Heuristic(&spec.inner.wire.weights)
                            }
                            OpponentSpec::CornFirstSetup => {
                                BorrowedArenaPolicy::CornFirstSetup(&spec.inner.wire.weights)
                            }
                            OpponentSpec::CornFirstUxmalOpening => {
                                BorrowedArenaPolicy::CornFirstUxmalOpening(&spec.inner.wire.weights)
                            }
                            OpponentSpec::ResourceCycle { configuration } => {
                                BorrowedArenaPolicy::ResourceCycle(configuration)
                            }
                            OpponentSpec::PreparedBc => {
                                BorrowedArenaPolicy::PublicBc(self.prepared)
                            }
                            OpponentSpec::PreparedBcTradeGuard { guard } => {
                                BorrowedArenaPolicy::PublicBcTradeGuard(self.prepared, guard)
                            }
                            OpponentSpec::Search => BorrowedArenaPolicy::Search(&self.search),
                            OpponentSpec::ParentArgmax => match &self.continued {
                                Some(o) => BorrowedArenaPolicy::ContinuedMixed(o),
                                None => BorrowedArenaPolicy::Mixed(
                                    self.root.initial_owner().expect("genuine initial owner"),
                                ),
                            },
                        })
                    })
                    .collect();
                match profile {
                    Some(profile) => FrozenMixedRoster::for_saved_plan(
                        seats,
                        Some(parent.identity()),
                        profile,
                        member,
                    ),
                    None => FrozenMixedRoster::for_current(seats, parent.identity()),
                }
            })
            .collect::<Result<Vec<_>, _>>()?
            .try_into()
            .map_err(|_| "Current mixed roster shape".into())
    }
    fn plan(&self, spec: &CurrentMixedCollectionSpec) -> Result<PlannedCurrentMixedCohort, String> {
        self.plan_with_profile(spec, None)
    }
    fn plan_for_saved(
        &self,
        spec: &CurrentMixedCollectionSpec,
        expected: &serde_json::Value,
    ) -> Result<PlannedCurrentMixedCohort, String> {
        let profile = SavedRosterProfile::from_plan(expected)?;
        let plan = self.plan_with_profile(spec, Some(&profile))?;
        profile.require_full_plan(&plan.receipt(), expected)?;
        Ok(plan)
    }
    fn plan_with_profile(
        &self,
        spec: &CurrentMixedCollectionSpec,
        profile: Option<&SavedRosterProfile>,
    ) -> Result<PlannedCurrentMixedCohort, String> {
        if spec.pool_checksum()? != self.pool {
            return Err("Current mixed immutable pool/rosters changed".into());
        }
        let handle = self.handle()?;
        let configs = spec.inner.configs()?;
        if configs.iter().any(|c| {
            self.progress
                .external_unavailable
                .contains(&seed_family_id(c.environment_seed()))
        }) {
            return Err("Current mixed external-unavailable family cannot be collected".into());
        }
        let rosters = self.rosters_with_profile(spec, profile)?;
        let headers = configs
            .iter()
            .zip(&rosters)
            .map(|(c, r)| source::header(c, &handle, r, &self.progress.consumed))
            .collect::<Result<Vec<_>, _>>()?
            .try_into()
            .map_err(|_| "Current mixed plan shape")?;
        PlannedCurrentMixedCohort::new(configs, headers)
    }
    fn declared_work(&self, spec: &CurrentMixedCollectionSpec) -> Result<Work, String> {
        let configs = spec.inner.configs()?;
        let rosters = self.rosters(spec)?;
        let mut w = Work {
            bytes: spec
                .inner
                .wire
                .max_output_bytes
                .checked_add(METADATA_HEADROOM)
                .ok_or("Current mixed allocation overflow")?,
            ..Work::default()
        };
        for (c, r) in configs.iter().zip(&rosters) {
            let callbacks = c.limits().max_callbacks();
            w = w.add(Work {
                callbacks: 2 * callbacks as u64,
                rows: 2
                    * (c.limits().max_candidate_rows() as u64
                        + callbacks as u64 * crate::features::MAX_LEGAL_ACTIONS as u64),
                search: 2 * r.max_opponent_search_steps(callbacks)? as u64,
                bytes: 0,
            })?;
        }
        w.callbacks = w
            .callbacks
            .checked_add(
                self.spec.critic["maxCallbacks"]
                    .as_u64()
                    .ok_or("Current mixed critic callback cap")?,
            )
            .ok_or("Current mixed callback overflow")?;
        w.rows = w
            .rows
            .checked_add(self.spec.actor()?.max_candidate_rows() as u64)
            .ok_or("Current mixed row overflow")?;
        Ok(w)
    }
    fn preflight(&self, collection: &CurrentMixedCollectionSpec) -> Result<BatchPlan, String> {
        self.require_live()?;
        if self.blocked
            || self.progress.phase != Phase::Ready
            || self.progress.attempts >= self.spec.max_attempts
            || self.progress.successful >= self.spec.max_successful_updates
        {
            return Err("Current mixed session stopped or budget consumed".into());
        }
        let work = self.declared_work(collection)?;
        self.progress.work.add(work)?.checked(&self.spec)?;
        if self
            .sequence
            .checked_add(EVENTS_PER_ATTEMPT)
            .is_none_or(|n| n > self.spec.max_journal_entries)
            || self
                .metadata_bytes
                .checked_add(METADATA_HEADROOM)
                .is_none_or(|n| n > self.spec.max_metadata_bytes)
        {
            return Err("Current mixed terminal/failure publication headroom".into());
        }
        self.spec.check_whole_four_storage(
            self.raw_record_bytes,
            self.stored_record_bytes,
            self.metadata_bytes,
        )?;
        let plan = self.plan(collection)?;
        let collection_raw = collection.inner.to_json()?;
        let plan_raw = bytes(&plan.receipt(), HEAD_CAP)?;
        let reservation = Reservation {
            attempt: self.progress.attempts,
            parent: plan.parent_identity().clone(),
            collection_spec: file_pin("collection-spec.json", &collection_raw),
            plan: file_pin("plan.json", &plan_raw),
            plan_checksum: plan.checksum().into(),
            configurations: plan.configurations().each_ref().map(Config::from_config),
            families: plan.rollout_families().clone(),
            work,
        };
        self.progress.check(
            &Event::ReserveWhole4 {
                reservation: Box::new(reservation.clone()),
            },
            &self.spec,
        )?;
        Ok(BatchPlan {
            plan,
            collection_raw,
            plan_raw,
            reservation,
        })
    }
    fn stopped_error(&mut self, error: String) -> Result<CurrentMixedSessionStatus, String> {
        self.require_batch_inventory(
            self.progress
                .active
                .as_ref()
                .ok_or("Missing failed current mixed attempt")?
                .attempt,
        )?;
        self.save(Event::Resolve {
            resolution: Resolution::Failed {
                error: bounded_error(&error),
            },
        })?;
        self.blocked = true;
        Ok(self.status())
    }
    /// No automatic next batch or retry; every call supplies a checked new four.
    pub fn update_whole_four(
        &mut self,
        collection: CurrentMixedCollectionSpec,
    ) -> Result<CurrentMixedSessionStatus, String> {
        self.update_whole_four_timed(collection, &mut Untimed)
    }
    /// Opt-in, output-only timings for exactly this call, including partial errors.
    /// The caller owns the profile and may save it after the charged outcome;
    /// diagnostic serialization never participates in journal or owner admission.
    pub fn update_whole_four_profiled(
        &mut self,
        collection: CurrentMixedCollectionSpec,
        profile: &mut CurrentMixedAttemptTimings,
    ) -> Result<CurrentMixedSessionStatus, String> {
        *profile = CurrentMixedAttemptTimings::default();
        profile.call_observed = true;
        profile.attempts_before = self.progress.attempts;
        profile.head_sha256_before = self
            .head
            .as_ref()
            .map(|p| p.sha256.clone())
            .unwrap_or_default();
        let result = timed(profile, TimedPhase::WholeCall, |timer| {
            self.update_whole_four_timed(collection, timer)
        });
        profile.attempts_after = self.progress.attempts;
        profile.head_sha256_after = self
            .head
            .as_ref()
            .map(|p| p.sha256.clone())
            .unwrap_or_default();
        profile.reservation_charged = self.progress.attempts > profile.attempts_before;
        profile.new_resolution = self
            .progress
            .last_resolved_own
            .as_ref()
            .filter(|r| profile.reservation_charged && r.attempt == profile.attempts_before)
            .map(|r| r.outcome);
        profile.phase_after = Some(self.progress.phase);
        profile.stopped_after = self.blocked || self.progress.phase == Phase::Stopped;
        profile.unknown_work_after = self.progress.unknown_work;
        profile.returned_error = result.is_err();
        result
    }
    fn update_whole_four_timed<T: PhaseTimer>(
        &mut self,
        collection: CurrentMixedCollectionSpec,
        timer: &mut T,
    ) -> Result<CurrentMixedSessionStatus, String> {
        let batch = timed(timer, TimedPhase::Preflight, |_| {
            self.preflight(&collection)
        })?;
        self.blocked = true;
        let result = self.run_whole_four(collection, batch, timer);
        if result.is_err() {
            self.blocked = true;
        }
        result
    }
    fn run_whole_four<T: PhaseTimer>(
        &mut self,
        collection: CurrentMixedCollectionSpec,
        batch: BatchPlan,
        timer: &mut T,
    ) -> Result<CurrentMixedSessionStatus, String> {
        let BatchPlan {
            plan,
            collection_raw,
            plan_raw,
            reservation,
        } = batch;
        let attempt = reservation.attempt;
        let dir = self.batch(attempt);
        let plan_value = timed(timer, TimedPhase::ReservationPublication, |_| {
            local(&dir, true)?;
            fs::create_dir(&dir).map_err(|e| e.to_string())?;
            if self.publish_batch(attempt, "collection-spec.json", &collection_raw, SPEC_CAP)?
                != reservation.collection_spec
                || self.publish_batch(attempt, "plan.json", &plan_raw, HEAD_CAP)?
                    != reservation.plan
            {
                return Err("Current mixed original planned pins differ".into());
            }
            let plan_value = plan.receipt();
            self.save(Event::ReserveWhole4 {
                reservation: Box::new(reservation),
            })?;
            Ok(plan_value)
        })?;
        for member in 0..4 {
            timed(timer, TimedPhase::CollectionJournal(member), |_| {
                self.save(Event::CollectStart { member })
            })?;
            timed(timer, TimedPhase::CollectionPreparation(member), |_| {
                self.require_live()
            })?;
            let returned = {
                let (handle, rosters) =
                    timed(timer, TimedPhase::CollectionPreparation(member), |_| {
                        Ok((self.handle()?, self.rosters(&collection)?))
                    })?;
                timed(timer, TimedPhase::Collection(member), |_| {
                    source::collect(
                        &plan.configurations()[member],
                        &handle,
                        &rosters[member],
                        plan.headers[member].clone(),
                    )
                })
            };
            timed(timer, TimedPhase::CollectionPreparation(member), |_| {
                self.require_live()
            })?;
            let (record, complete, error) = match returned {
                Ok(game) => {
                    let complete = game.terminal.is_some() && game.failure.is_none();
                    let p = timed(timer, TimedPhase::RecordPublication(member), |_| {
                        let raw = source::encode(&game)?;
                        source::check_record(&raw, &plan.headers[member])?;
                        self.publish_record(attempt, member, &raw)
                    })?;
                    let error = (!complete).then(|| {
                        bounded_error(
                            &game
                                .failure
                                .as_ref()
                                .map(|f| f.message.clone())
                                .unwrap_or_else(|| {
                                    "Incomplete current mixed native collection".into()
                                }),
                        )
                    });
                    (Some(p), complete, error)
                }
                Err(e) => (None, false, Some(bounded_error(&e))),
            };
            timed(timer, TimedPhase::CollectionJournal(member), |_| {
                self.save(Event::CollectReturn {
                    member,
                    record,
                    complete,
                    error: error.clone(),
                })
            })?;
            if let Some(e) = error {
                return timed(timer, TimedPhase::Resolve, |_| self.stopped_error(e));
            }
        }
        let mut episodes = Vec::new();
        for member in 0..4 {
            let record = self.progress.collect[member]
                .as_ref()
                .ok_or("Missing returned current mixed record")?;
            let checked = timed(timer, TimedPhase::AuditRecordRead(member), |_| {
                let raw = read_record(&dir, record, member, &self.spec)?;
                source::check_for_audit(&raw, &plan.headers[member])
            })?;
            timed(timer, TimedPhase::AuditJournal(member), |_| {
                self.save(Event::AuditStart { member })
            })?;
            timed(timer, TimedPhase::AuditPreparation(member), |_| {
                self.require_live()
            })?;
            let returned = {
                let (handle, rosters) = timed(timer, TimedPhase::AuditPreparation(member), |_| {
                    Ok((self.handle()?, self.rosters(&collection)?))
                })?;
                timed(timer, TimedPhase::Audit(member), |_| {
                    source::audit_checked(
                        checked,
                        &handle,
                        &rosters[member],
                        plan.headers[member].clone(),
                    )
                })
            };
            timed(timer, TimedPhase::AuditPreparation(member), |_| {
                self.require_live()
            })?;
            timed(timer, TimedPhase::AuditJournal(member), |_| {
                self.save(Event::AuditReturn {
                    member,
                    passed: returned.is_ok(),
                    episode_checksum: returned
                        .as_ref()
                        .ok()
                        .map(|e| e.canonical_record_checksum().into()),
                    error: returned.as_ref().err().map(|e| bounded_error(e)),
                })
            })?;
            match returned {
                Ok(e) => episodes.push(Ok(e)),
                Err(e) => return timed(timer, TimedPhase::Resolve, |_| self.stopped_error(e)),
            }
        }
        let episodes = episodes
            .try_into()
            .map_err(|_| "Current mixed sealed whole-four shape")?;
        let cohort = match timed(timer, TimedPhase::CohortPublication, |_| {
            plan.finish(episodes)
        }) {
            Ok(c) => c,
            Err(e) => return timed(timer, TimedPhase::Resolve, |_| self.stopped_error(e)),
        };
        let cohort_value = cohort.receipt();
        let cohort_pin = timed(timer, TimedPhase::CohortPublication, |_| {
            self.publish_batch(
                attempt,
                "cohort.json",
                &bytes(&cohort_value, HEAD_CAP)?,
                HEAD_CAP,
            )
        })?;
        timed(timer, TimedPhase::GradientStart, |_| {
            self.save(Event::GradientStart {
                cohort: cohort_pin,
                cohort_checksum: cohort.receipt_checksum().into(),
                episode_checksums: cohort
                    .episodes()
                    .each_ref()
                    .map(|e| e.canonical_record_checksum().into()),
                counts: [
                    cohort.total_learner_callbacks(),
                    cohort.total_learner_candidate_rows(),
                    cohort.total_learner_singletons(),
                ],
            })
        })?;
        let (handle, actor_config, critic_config) =
            timed(timer, TimedPhase::NumericPreparationControl, |_| {
                self.require_live()?;
                Ok((self.handle()?, self.spec.actor()?, self.spec.critic()?))
            })?;
        let result = timed(timer, TimedPhase::NumericPreparation, |timer| {
            mixed::prepare_current_timed(&handle, &cohort, &actor_config, &critic_config, timer)
        });
        timed(timer, TimedPhase::NumericPreparationControl, |_| {
            self.require_live()
        })?;
        let proposal = match result {
            Ok(p) => p,
            Err(e) => return timed(timer, TimedPhase::Resolve, |_| self.stopped_error(e)),
        };
        let (report, saved_report) = timed(timer, TimedPhase::ProposalReadback, |_| {
            proposal.require_parent(&self.handle()?)?;
            proposal.report().require_saved_cohort(
                &plan_value,
                &cohort_value,
                &self.spec.actor()?,
                &self.spec.critic()?,
            )?;
            let report = self.publish_batch(
                attempt,
                "report.json",
                &bytes(proposal.report(), HEAD_CAP)?,
                HEAD_CAP,
            )?;
            let saved_report: mixed::ContinuedMixedReport =
                closed(&read_pin(&dir, &report, HEAD_CAP)?, HEAD_CAP)?;
            saved_report.require_saved_cohort(
                &plan_value,
                &cohort_value,
                &self.spec.actor()?,
                &self.spec.critic()?,
            )?;
            if saved_report.checksum() != proposal.report().checksum() {
                return Err("Current mixed report readback differs".into());
            }
            Ok((report, saved_report))
        })?;
        if proposal.artifact_for_publication().is_some() {
            let (promotion, artifact) = timed(timer, TimedPhase::ProposalReadback, |_| {
                self.require_reserved_report(proposal.report())?;
                let promotion = proposal.stage(&self.handle()?)?;
                let a = promotion.artifact();
                let artifact = self.publish_batch(
                    attempt,
                    "artifact.json",
                    &bytes(a, ARTIFACT_CAP)?,
                    ARTIFACT_CAP,
                )?;
                let saved: mixed::ContinuedMixedArtifact =
                    closed(&read_pin(&dir, &artifact, ARTIFACT_CAP)?, ARTIFACT_CAP)?;
                if saved.checksum() != a.checksum()
                    || bytes(saved.report(), HEAD_CAP)? != bytes(&saved_report, HEAD_CAP)?
                {
                    return Err("Current mixed joint artifact/report readback differs".into());
                }
                mixed::ContinuedMixedProposal::restored(saved, &self.handle()?)?;
                self.require_batch_inventory(attempt)?;
                Ok((promotion, artifact))
            })?;
            timed(timer, TimedPhase::Resolve, |_| {
                let a = promotion.artifact();
                self.save(Event::Resolve {
                    resolution: Resolution::Updated {
                        report: report.clone(),
                        artifact: artifact.clone(),
                        report_checksum: a.report().checksum().into(),
                        artifact_checksum: a.checksum().into(),
                    },
                })?;
                // Publication/readback completed; no prior owner is taken on failure.
                mixed::admit_resolved(DurableMixedResolve { promotion }, &mut self.continued);
                self.owner_pins = Some((dir, report, artifact));
                Ok(())
            })?;
        } else {
            timed(timer, TimedPhase::Resolve, |_| {
                self.require_batch_inventory(attempt)?;
                self.save(Event::Resolve {
                    resolution: Resolution::NoChange {
                        report,
                        report_checksum: proposal.report().checksum().into(),
                    },
                })
            })?;
        }
        timed(timer, TimedPhase::FinalGuards, |_| {
            self.require_live()?;
            self.handle()?;
            self.blocked = false;
            Ok(self.status())
        })
    }
    fn require_reserved_report(&self, report: &mixed::ContinuedMixedReport) -> Result<(), String> {
        let r = self
            .progress
            .active
            .as_ref()
            .ok_or("Missing current mixed reserved four")?;
        if report.parent_identity() != &r.parent
            || report.plan_checksum() != r.plan_checksum
            || report.families() != &r.families
            || report.proposed_mixed_update_count() != self.progress.next_parent_count()?
        {
            return Err("Current mixed Resolve candidate differs from reserved four".into());
        }
        Ok(())
    }
    pub fn abandon_unresolved(
        &mut self,
        reason: &str,
    ) -> Result<CurrentMixedSessionStatus, String> {
        self.require_layout()?;
        self.save(Event::Resolve {
            resolution: Resolution::Abandoned {
                reason: bounded_error(reason),
            },
        })?;
        self.blocked = true;
        Ok(self.status())
    }
    fn require_batch_inventory(&self, attempt: u64) -> Result<(), String> {
        let dir = self.batch(attempt);
        local(&dir, false)?;
        let pins = self
            .references
            .get(&attempt)
            .ok_or("Missing current mixed original file references")?;
        let mut names = Vec::new();
        for e in fs::read_dir(&dir).map_err(|e| e.to_string())? {
            if names.len() >= 9 {
                return Err("Current mixed batch inventory count cap".into());
            }
            names.push(
                e.map_err(|e| e.to_string())?
                    .file_name()
                    .into_string()
                    .map_err(|_| "Current mixed leaf is not UTF8")?,
            );
        }
        names.sort();
        let mut expected = pins.iter().map(|p| p.name.clone()).collect::<Vec<_>>();
        expected.sort();
        if names != expected {
            return Err("Current mixed extra/missing batch outputs".into());
        }
        for p in pins {
            read_physical_pin(&dir, p, &self.spec)?;
        }
        Ok(())
    }
}
fn require_owner_pins(pins: &(PathBuf, FilePin, FilePin)) -> Result<(), String> {
    read_pin(&pins.0, &pins.1, HEAD_CAP)?;
    read_pin(&pins.0, &pins.2, ARTIFACT_CAP)?;
    Ok(())
}
fn names(directory: &Path, cap: usize) -> Result<Vec<String>, String> {
    local(directory, false)?;
    let mut result = Vec::new();
    for e in fs::read_dir(directory).map_err(|e| e.to_string())? {
        if result.len() >= cap {
            return Err("Current mixed inventory count cap".into());
        }
        let e = e.map_err(|e| e.to_string())?;
        if e.file_type().map_err(|e| e.to_string())?.is_symlink() {
            return Err("Current mixed inventory link".into());
        }
        result.push(
            e.file_name()
                .into_string()
                .map_err(|_| "Current mixed inventory UTF8")?,
        );
    }
    result.sort();
    Ok(result)
}
fn bounded_error(error: &str) -> String {
    let text = if error.is_empty() {
        "Current mixed returned an empty error"
    } else {
        error
    };
    let mut n = text.len().min(1024);
    while !text.is_char_boundary(n) {
        n -= 1;
    }
    text[..n].into()
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    fn spec() -> CurrentMixedRunSpec {
        CurrentMixedRunSpec::from_json(&serde_json::to_vec(&json!({"schema":SPEC_SCHEMA,
            "actor":OneStepConfig::default(),"critic":ResidualFitConfig::new(1e-4,0.01,100).unwrap(),
            "maxAttempts":2,"maxSuccessfulUpdates":2,"maxFamilies":32,"maxJournalEntries":43,
            "maxMetadataBytes":33554432,"maxOutputBytes":1073741824,"maxCallbacks":1000,
            "maxCandidateRows":10000000,"maxSearchSteps":0})).unwrap()).unwrap()
    }
    fn compressed_spec() -> CurrentMixedRunSpec {
        let mut value = serde_json::to_value(spec()).unwrap();
        value["schema"] = COMPRESSED_SPEC_SCHEMA.into();
        value["recordStorage"] = json!({"schema":record_storage::STORAGE_SCHEMA,
            "maxStoredBytes":1073741824u64,"maxTemporaryBytes":MAX_STORED_RECORD_BYTES,
            "maxDiskBytes":2147483648u64});
        CurrentMixedRunSpec::from_json(&serde_json::to_vec(&value).unwrap()).unwrap()
    }
    #[test]
    fn current_mixed_storage_preserves_raw_wire_and_rejects_mode_mixing() {
        let raw = spec();
        let expected = br#"{"schema":"tzolkin-current-mixed-durable-spec-v2","actor":{"learningRate":0.0001,"maxAbsActualDelta":0.01,"maxCandidateRows":4000000},"critic":{"learningRate":0.0001,"maxAbsActualDelta":0.01,"maxCallbacks":100},"maxAttempts":2,"maxSuccessfulUpdates":2,"maxFamilies":32,"maxJournalEntries":43,"maxMetadataBytes":33554432,"maxOutputBytes":1073741824,"maxCallbacks":1000,"maxCandidateRows":10000000,"maxSearchSteps":0}"#;
        assert_eq!(bytes(&raw, SPEC_CAP).unwrap(), expected);
        assert_eq!(
            raw.checksum().unwrap(),
            "3d25e45ea5068d04866cbb18d17b689699ff930cc6b8c3c1a02a23b325e10205"
        );
        assert_eq!(raw.journal_schema(), JOURNAL_SCHEMA);
        let pin = file_pin("record-0.json", b"record");
        assert_eq!(
            bytes(&RecordPin::Raw(pin.clone()), HEAD_CAP).unwrap(),
            bytes(&pin, HEAD_CAP).unwrap()
        );
        let compressed = compressed_spec();
        assert_eq!(compressed.journal_schema(), COMPRESSED_JOURNAL_SCHEMA);
        assert_ne!(raw.checksum().unwrap(), compressed.checksum().unwrap());
        let mut value = serde_json::to_value(&raw).unwrap();
        value["recordStorage"] = serde_json::Value::Null;
        assert!(CurrentMixedRunSpec::from_json(&serde_json::to_vec(&value).unwrap()).is_err());
        value["schema"] = COMPRESSED_SPEC_SCHEMA.into();
        value.as_object_mut().unwrap().remove("recordStorage");
        assert!(CurrentMixedRunSpec::from_json(&serde_json::to_vec(&value).unwrap()).is_err());
        let mut value = serde_json::to_value(&compressed).unwrap();
        value["schema"] = SPEC_SCHEMA.into();
        assert!(CurrentMixedRunSpec::from_json(&serde_json::to_vec(&value).unwrap()).is_err());
        value["schema"] = COMPRESSED_SPEC_SCHEMA.into();
        value["recordStorage"]["schema"] = "unknown".into();
        assert!(CurrentMixedRunSpec::from_json(&serde_json::to_vec(&value).unwrap()).is_err());
        let (_, gzip) = encode_record(b"record", 0, &compressed).unwrap();
        assert!(gzip.validate(0, &raw).is_err());
        assert!(RecordPin::Raw(pin).validate(0, &compressed).is_err());
        assert!(gzip.validate(1, &compressed).is_err());
        let (mut progress, _, reservation) = reserved();
        progress
            .apply(
                &Event::ReserveWhole4 {
                    reservation: Box::new(reservation),
                },
                &compressed,
            )
            .unwrap();
        progress
            .apply(&Event::CollectStart { member: 0 }, &compressed)
            .unwrap();
        let event = Event::CollectReturn {
            member: 0,
            record: Some(gzip),
            complete: true,
            error: None,
        };
        let wire = bytes(&event, HEAD_CAP).unwrap();
        let restored: Event = closed(&wire, HEAD_CAP).unwrap();
        progress.apply(&restored, &compressed).unwrap();
        assert_eq!(progress.phase, Phase::Collected(0));
        assert_eq!(progress.collect[0].as_ref().unwrap().raw_bytes(), 6);
    }
    #[test]
    fn current_mixed_storage_io_binds_stored_and_original_bytes_without_overwrite() {
        let raw_spec = spec();
        let spec = compressed_spec();
        let raw = b" {\n\"literal\":\"\\u0041\",\"numbers\":[1.0,1e0]}\n";
        let (stored, pin) = encode_record(raw, 0, &spec).unwrap();
        assert_eq!(encode_record(raw, 0, &spec).unwrap().0, stored);
        let directory = std::env::temp_dir().join(format!(
            "tzolkin-mixed-storage-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir(&directory).unwrap();
        publish_record_file(&directory, &stored, raw, &pin, 0, &spec).unwrap();
        assert_eq!(read_record(&directory, &pin, 0, &spec).unwrap(), raw);
        assert!(publish_record_file(&directory, &stored, raw, &pin, 0, &spec).is_err());
        assert_eq!(names(&directory, 9).unwrap(), vec!["record-0.json.gz"]);
        assert!(read_record(&directory, &pin, 0, &raw_spec).is_err());
        let mut wrong = pin.clone();
        if let RecordPin::Compressed(p) = &mut wrong {
            p.stored.sha256 = "f".repeat(64);
        }
        assert!(read_record(&directory, &wrong, 0, &spec).is_err());
        let mut wrong = pin.clone();
        if let RecordPin::Compressed(p) = &mut wrong {
            p.raw_sha256 = "f".repeat(64);
        }
        assert!(read_record(&directory, &wrong, 0, &spec).is_err());
        assert!(
            read_mixed_stored_record(
                &directory.join("record-0.json.gz"),
                MAX_STORED_RECORD_BYTES + 1
            )
            .is_err()
        );
        fs::write(directory.join("record-1.json.gz"), b"corrupt").unwrap();
        let corrupt = RecordPin::Compressed(CompressedRecordPin {
            schema: record_storage::PIN_SCHEMA.into(),
            stored: file_pin("record-1.json.gz", b"corrupt"),
            raw_bytes: raw.len(),
            raw_sha256: sha(raw),
        });
        assert!(read_record(&directory, &corrupt, 1, &spec).is_err());
        fs::remove_dir_all(directory).unwrap();
    }
    #[test]
    fn current_mixed_active_record_cache_keeps_current_bytes_and_scope_checks() {
        for spec in [spec(), compressed_spec()] {
            let directory = test_directory("active-record-cache");
            let raw = br#"{"record":"original"}"#;
            let (stored, pin) = encode_record(raw, 0, &spec).unwrap();
            let verified = publish_record_file(&directory, &stored, raw, &pin, 0, &spec).unwrap();
            let mut cache = ActiveRecordByteCache::default();
            cache.remember(7, verified);
            assert_eq!(
                cache.lookup(7, &directory, pin.file(), &spec),
                Some(stored.as_ref())
            );
            for _ in 0..2 {
                cache.require(7, &directory, pin.file(), &spec).unwrap();
            }
            let path = directory.join(&pin.file().name);
            let modified = fs::metadata(&path).unwrap().modified().unwrap();
            let mut changed = stored.to_vec();
            changed[0] ^= 1;
            fs::write(&path, &changed).unwrap();
            fs::OpenOptions::new()
                .write(true)
                .open(&path)
                .unwrap()
                .set_times(fs::FileTimes::new().set_modified(modified))
                .unwrap();
            assert_eq!(fs::metadata(&path).unwrap().modified().unwrap(), modified);
            assert!(cache.require(7, &directory, pin.file(), &spec).is_err());
            assert!(read_record(&directory, &pin, 0, &spec).is_err());
            fs::write(&path, &stored[..stored.len() - 1]).unwrap();
            assert!(cache.require(7, &directory, pin.file(), &spec).is_err());
            fs::write(&path, &stored).unwrap();
            cache.require(7, &directory, pin.file(), &spec).unwrap();

            let mut wrong = pin.file().clone();
            wrong.sha256 = "f".repeat(64);
            assert!(cache.lookup(7, &directory, &wrong, &spec).is_none());
            assert!(cache.require(7, &directory, &wrong, &spec).is_err());
            assert!(cache.lookup(8, &directory, pin.file(), &spec).is_none());
            let other = directory.join("other");
            fs::create_dir(&other).unwrap();
            fs::write(other.join(&pin.file().name), &changed).unwrap();
            assert!(cache.lookup(7, &other, pin.file(), &spec).is_none());
            assert!(cache.require(7, &other, pin.file(), &spec).is_err());
            let other_spec = if spec.record_storage.is_some() {
                self::spec()
            } else {
                compressed_spec()
            };
            assert!(
                cache
                    .lookup(7, &directory, pin.file(), &other_spec)
                    .is_none()
            );
            assert!(
                cache
                    .require(7, &directory, pin.file(), &other_spec)
                    .is_err()
            );

            let (next_stored, next_pin) = encode_record(raw, 1, &spec).unwrap();
            let next =
                publish_record_file(&directory, &next_stored, raw, &next_pin, 1, &spec).unwrap();
            cache.remember(8, next);
            assert_eq!(cache.records.len(), 1);
            assert_eq!(cache.bytes, next_stored.len());
            assert!(cache.lookup(7, &directory, pin.file(), &spec).is_none());
            cache
                .require(8, &directory, next_pin.file(), &spec)
                .unwrap();
            let next_path = directory.join(&next_pin.file().name);
            fs::remove_file(&next_path).unwrap();
            fs::create_dir(&next_path).unwrap();
            assert!(
                cache
                    .require(8, &directory, next_pin.file(), &spec)
                    .is_err()
            );
            fs::remove_dir_all(directory).unwrap();
        }
    }
    #[test]
    fn current_mixed_storage_reserves_logical_physical_and_temporary_headroom() {
        let raw = spec();
        assert!(raw.check_storage(10, 11, 0).is_err());
        assert!(raw.check_storage(10, 10, raw.max_output_bytes).is_err());
        assert!(raw.check_whole_four_storage(0, 0, 0).is_ok());
        let mut gzip = compressed_spec();
        assert!(gzip.check_storage(10, 12, 100).is_ok());
        assert!(gzip.check_whole_four_storage(0, 0, 0).is_ok());
        assert!(gzip.check_storage(u64::MAX, 0, 1).is_err());
        assert!(gzip.check_storage(0, u64::MAX, 1).is_err());
        assert!(gzip.check_whole_four_storage(u64::MAX, 0, 0).is_err());
        assert!(
            gzip.check_storage(0, 0, gzip.max_metadata_bytes + 1)
                .is_err()
        );
        let storage = gzip.record_storage.as_mut().unwrap();
        storage.max_stored_bytes = 4 * MAX_STORED_RECORD_BYTES as u64;
        storage.max_disk_bytes =
            storage.max_stored_bytes + METADATA_HEADROOM + storage.max_temporary_bytes;
        assert!(gzip.check_whole_four_storage(0, 0, 0).is_ok());
        assert!(gzip.check_whole_four_storage(0, 1, 0).is_err());
        assert!(gzip.check_whole_four_storage(0, 0, 1).is_err());
        assert_eq!(MAX_STORED_RECORD_BYTES, 108 * 1024 * 1024 + 1024);
        assert_eq!(METADATA_HEADROOM, 55 * 1024 * 1024 / 2);
    }
    pub(super) fn reserved() -> (Progress, CurrentMixedRunSpec, Reservation) {
        let s = spec();
        let plan = source::tests::plan();
        let parent = plan.parent_identity().clone();
        let mut progress = Progress::initial(&parent, [7, 11], 9, &BTreeSet::new());
        progress
            .apply(
                &Event::Start {
                    run_spec: file_pin("run-spec.json", b"spec"),
                    parent: Box::new(parent.clone()),
                    prefix: [7, 11],
                    next: 9,
                    consumed: vec![],
                },
                &s,
            )
            .unwrap();
        let r = Reservation {
            attempt: 0,
            parent,
            collection_spec: file_pin("collection-spec.json", b"spec"),
            plan: file_pin("plan.json", b"plan"),
            plan_checksum: plan.checksum().into(),
            configurations: plan.configurations().each_ref().map(Config::from_config),
            families: plan.rollout_families().clone(),
            work: Work {
                callbacks: 4,
                rows: 4,
                search: 0,
                bytes: 64,
            },
        };
        (progress, s, r)
    }
    fn gradient_events(counts: [usize; 3]) -> Vec<Event> {
        let mut events = Vec::new();
        for member in 0..4 {
            events.push(Event::CollectStart { member });
            events.push(Event::CollectReturn {
                member,
                record: Some(RecordPin::Raw(file_pin(
                    &format!("record-{member}.json"),
                    b"record",
                ))),
                complete: true,
                error: None,
            });
        }
        for member in 0..4 {
            events.push(Event::AuditStart { member });
            events.push(Event::AuditReturn {
                member,
                passed: true,
                episode_checksum: Some(format!("{member:064x}")),
                error: None,
            });
        }
        events.push(Event::GradientStart {
            cohort: file_pin("cohort.json", b"cohort"),
            cohort_checksum: "a".repeat(64),
            episode_checksums: std::array::from_fn(|i| format!("{i:064x}")),
            counts,
        });
        events
    }
    pub(super) fn gradient(p: &mut Progress, s: &CurrentMixedRunSpec) {
        for event in gradient_events([4, 4, 4]) {
            p.apply(&event, s).unwrap();
        }
    }
    // Content-only reducer fixtures: no sealed owner or continuation admission.
    fn continued_parent(p: &Progress) -> CurrentMixedParentIdentity {
        let mut wire = serde_json::to_value(source::tests::initial()).unwrap();
        for (key, value) in [
            ("schema", source::CONTINUED_IDENTITY_SCHEMA),
            ("policyVersion", source::CONTINUED_POLICY_VERSION),
            ("samplingVersion", source::CONTINUED_SAMPLING_VERSION),
            ("task", source::CONTINUED_TASK),
        ] {
            wire[key] = value.into();
        }
        wire["firstMixedChecksum"] = "d".repeat(64).into();
        wire["parentArtifactChecksum"] = "e".repeat(64).into();
        wire["artifactChecksum"] = p.current_checksum.clone().into();
        wire["mixedUpdateCount"] = p.parent_count().unwrap().into();
        let parent = CurrentMixedParentIdentity::ContinuedMixedV2 {
            policy: serde_json::from_value(wire).unwrap(),
        };
        parent.validate().unwrap();
        parent
    }
    pub(super) fn later_reservation(p: &Progress, template: &Reservation) -> Reservation {
        let mut r = template.clone();
        r.attempt = p.attempts;
        r.parent = continued_parent(p);
        let seeds = (0..100)
            .filter(|&seed| {
                let family = seed_family_id(seed);
                split_for_family(&family).unwrap() == DatasetSplit::Train
                    && !p.consumed.contains(&family)
            })
            .take(4)
            .collect::<Vec<_>>();
        assert_eq!(seeds.len(), 4);
        for (i, seed) in seeds.into_iter().enumerate() {
            r.configurations[i].environment_seed = seed;
            r.configurations[i].episode_ordinal = format!("{:016x}", p.next + i as u64);
            r.families[i] = seed_family_id(seed);
        }
        r
    }
    fn cumulative_own_fixture() -> CurrentMixedOwnDecisionCounters {
        let bucket = |n| CurrentMixedResolvedOwnCounts {
            cohorts: n + 1,
            cohorts_without_counts: n,
            own: CurrentMixedOwnCounts {
                callbacks: n * 10,
                candidate_rows: n * 20,
                singletons: n * 2,
            },
        };
        CurrentMixedOwnDecisionCounters {
            updated: bucket(1),
            no_change: bucket(2),
            failed: bucket(3),
            abandoned: bucket(4),
        }
    }
    #[test]
    fn current_mixed_cumulative_own_decisions_keep_all_buckets_and_missing_counts() {
        let inherited = cumulative_own_fixture();
        let mut local = CurrentMixedOwnDecisionCounters::default();
        let counts = CurrentMixedOwnCounts {
            callbacks: 7,
            candidate_rows: 21,
            singletons: 2,
        };
        local.updated.add(Some(counts));
        local.no_change.add(None);
        local.failed.add(Some(counts));
        local.abandoned.add(None);
        let total = inherited.checked_add(local).unwrap();
        assert_eq!(total.updated.cohorts, 3);
        assert_eq!(total.updated.cohorts_without_counts, 1);
        assert_eq!(total.updated.own.callbacks, 17);
        assert_eq!(total.updated.own.candidate_rows, 41);
        assert_eq!(total.updated.own.singletons, 4);
        assert_eq!(total.no_change.cohorts, 4);
        assert_eq!(total.no_change.cohorts_without_counts, 3);
        assert_eq!(total.no_change.own, inherited.no_change.own);
        assert_eq!(total.failed.cohorts, 5);
        assert_eq!(total.failed.cohorts_without_counts, 3);
        assert_eq!(total.failed.own.callbacks, 37);
        assert_eq!(total.failed.own.candidate_rows, 81);
        assert_eq!(total.failed.own.singletons, 8);
        assert_eq!(total.abandoned.cohorts, 6);
        assert_eq!(total.abandoned.cohorts_without_counts, 5);
        assert_eq!(total.abandoned.own, inherited.abandoned.own);
        assert_eq!(local.checked_add(inherited).unwrap(), total);
        assert_eq!(total.checked_add(Default::default()).unwrap(), total);
    }
    #[test]
    fn current_mixed_cumulative_own_decisions_refuse_each_counter_overflow() {
        for bucket_index in 0..4 {
            for field_index in 0..5 {
                let mut inherited = cumulative_own_fixture();
                let bucket = [
                    &mut inherited.updated,
                    &mut inherited.no_change,
                    &mut inherited.failed,
                    &mut inherited.abandoned,
                ]
                .into_iter()
                .nth(bucket_index)
                .unwrap();
                *[
                    &mut bucket.cohorts,
                    &mut bucket.cohorts_without_counts,
                    &mut bucket.own.callbacks,
                    &mut bucket.own.candidate_rows,
                    &mut bucket.own.singletons,
                ][field_index] = u64::MAX;
                assert!(inherited.checked_add(cumulative_own_fixture()).is_err());
                assert!(cumulative_own_fixture().checked_add(inherited).is_err());
                assert_eq!(
                    inherited.checked_add(Default::default()).unwrap(),
                    inherited
                );
            }
        }
    }
    #[test]
    fn current_mixed_own_decision_counts_follow_resolve_and_restore() {
        let (_, mut spec, template) = reserved();
        spec.max_attempts = 3;
        spec.max_successful_updates = 3;
        spec.max_journal_entries = 64;
        spec.validate().unwrap();
        let initial = || Progress::initial(&template.parent, [7, 11], 9, &BTreeSet::new());
        let own = |callbacks, candidate_rows, singletons| CurrentMixedOwnCounts {
            callbacks,
            candidate_rows,
            singletons,
        };
        // The third attempt may stop before counts exist, or after a sealed
        // cohort. Both kinds of pending work must leave resolved totals alone.
        for (kind, has_counts, resolved) in [
            (CurrentMixedOwnResolution::Failed, false, false),
            (CurrentMixedOwnResolution::Failed, true, false),
            (CurrentMixedOwnResolution::Failed, false, true),
            (CurrentMixedOwnResolution::Failed, true, true),
            (CurrentMixedOwnResolution::Abandoned, false, true),
            (CurrentMixedOwnResolution::Abandoned, true, true),
        ] {
            let mut progress = initial();
            let mut events = Vec::new();
            let apply = |p: &mut Progress, events: &mut Vec<Event>, event: Event| {
                p.apply(&event, &spec).unwrap();
                events.push(event);
            };
            apply(
                &mut progress,
                &mut events,
                Event::Start {
                    run_spec: file_pin("run-spec.json", b"spec"),
                    parent: Box::new(template.parent.clone()),
                    prefix: [7, 11],
                    next: 9,
                    consumed: vec![],
                },
            );
            for (attempt, counts) in [[4, 8, 1], [7, 14, 3]].into_iter().enumerate() {
                let reservation = if attempt == 0 {
                    template.clone()
                } else {
                    later_reservation(&progress, &template)
                };
                apply(
                    &mut progress,
                    &mut events,
                    Event::ReserveWhole4 {
                        reservation: Box::new(reservation),
                    },
                );
                for event in gradient_events(counts) {
                    apply(&mut progress, &mut events, event);
                }
                assert_eq!(progress.own_decisions.no_change.cohorts, 0);
                let resolution = if attempt == 0 {
                    Resolution::Updated {
                        report: file_pin("report.json", b"report"),
                        artifact: file_pin("artifact.json", b"artifact"),
                        report_checksum: "b".repeat(64),
                        artifact_checksum: "c".repeat(64),
                    }
                } else {
                    Resolution::NoChange {
                        report: file_pin("report.json", b"no change report"),
                        report_checksum: "d".repeat(64),
                    }
                };
                apply(&mut progress, &mut events, Event::Resolve { resolution });
            }
            let expected = CurrentMixedOwnDecisionCounters {
                updated: CurrentMixedResolvedOwnCounts {
                    cohorts: 1,
                    own: own(4, 8, 1),
                    ..Default::default()
                },
                no_change: CurrentMixedResolvedOwnCounts {
                    cohorts: 1,
                    own: own(7, 14, 3),
                    ..Default::default()
                },
                ..Default::default()
            };
            assert_eq!(progress.own_decisions, expected);
            assert_eq!(progress.last_resolved_own.unwrap().own, Some(own(7, 14, 3)));
            let reservation = later_reservation(&progress, &template);
            apply(
                &mut progress,
                &mut events,
                Event::ReserveWhole4 {
                    reservation: Box::new(reservation),
                },
            );
            assert_eq!(progress.active_own, None); // Never the prior owner's report.
            for event in if has_counts {
                gradient_events([5, 20, 2])
            } else {
                vec![Event::CollectStart { member: 0 }]
            } {
                apply(&mut progress, &mut events, event);
            }
            assert_eq!(progress.own_decisions, expected);
            assert_eq!(progress.active_own, has_counts.then_some(own(5, 20, 2)));
            if resolved {
                let resolution = if kind == CurrentMixedOwnResolution::Failed {
                    if !has_counts {
                        apply(
                            &mut progress,
                            &mut events,
                            Event::CollectReturn {
                                member: 0,
                                record: None,
                                complete: false,
                                error: Some("returned failure".into()),
                            },
                        );
                    }
                    Resolution::Failed {
                        error: "returned failure".into(),
                    }
                } else {
                    Resolution::Abandoned {
                        reason: "explicit stop".into(),
                    }
                };
                apply(&mut progress, &mut events, Event::Resolve { resolution });
                let bucket = if kind == CurrentMixedOwnResolution::Failed {
                    progress.own_decisions.failed
                } else {
                    progress.own_decisions.abandoned
                };
                assert_eq!(bucket.cohorts, 1);
                assert_eq!(bucket.cohorts_without_counts, u64::from(!has_counts));
                assert_eq!(
                    bucket.own,
                    if has_counts {
                        own(5, 20, 2)
                    } else {
                        own(0, 0, 0)
                    }
                );
                assert_eq!(progress.last_resolved_own.unwrap().attempt, 2);
                assert_eq!(progress.last_resolved_own.unwrap().outcome, kind);
            } else {
                assert_eq!(progress.own_decisions, expected);
                assert_eq!(progress.last_resolved_own.unwrap().attempt, 1);
            }
            assert_eq!(progress.own_decisions.updated, expected.updated);
            assert_eq!(progress.own_decisions.no_change, expected.no_change);
            assert_eq!(progress.successful, 1);
            assert_eq!(
                progress.unknown_work,
                !resolved || kind == CurrentMixedOwnResolution::Abandoned
            );
            // Restore consumes the unchanged serialized events through apply,
            // just as the production restore does after validating each body.
            let mut restored = initial();
            for event in events {
                let decoded: Event = closed(&bytes(&event, HEAD_CAP).unwrap(), HEAD_CAP).unwrap();
                restored.apply(&decoded, &spec).unwrap();
            }
            assert_eq!(restored.own_decisions, progress.own_decisions);
            assert_eq!(restored.active_own, progress.active_own);
            assert_eq!(restored.last_resolved_own, progress.last_resolved_own);
            assert_eq!(restored.phase, progress.phase);
            assert_eq!(restored.unknown_work, progress.unknown_work);
            assert_eq!(restored.consumed, progress.consumed);
            assert_eq!(restored.next, progress.next);
            let inherited = cumulative_own_fixture();
            let cumulative = inherited.checked_add(progress.own_decisions).unwrap();
            assert_eq!(
                inherited.checked_add(restored.own_decisions).unwrap(),
                cumulative
            );
            // A successor starts a new local reducer; its inherited totals are
            // taken from the resolved predecessor, never from its pending counts.
            let mut next = Progress::initial(
                &continued_parent(&progress),
                progress.prefix,
                progress.next,
                &progress.consumed,
            );
            next.base = progress.parent_count().unwrap();
            assert_eq!(
                next.own_decisions,
                CurrentMixedOwnDecisionCounters::default()
            );
            assert_eq!(
                cumulative.checked_add(next.own_decisions).unwrap(),
                cumulative
            );
            assert_eq!(
                initial().own_decisions,
                CurrentMixedOwnDecisionCounters::default()
            );
        }
    }
    #[test]
    fn output_only_timings_preserve_numeric_result_and_retain_failed_spans() {
        use crate::public_policy_update::propose_parameters;
        let mut empty = CurrentMixedAttemptTimings::default();
        let surrounding = [
            TimedPhase::CollectionJournal(2),
            TimedPhase::CollectionPreparation(2),
            TimedPhase::AuditRecordRead(2),
            TimedPhase::AuditJournal(2),
            TimedPhase::AuditPreparation(2),
            TimedPhase::NumericPreparationControl,
        ];
        for phase in surrounding {
            let stage = empty.stage(phase);
            assert_eq!(
                (stage.entered, stage.returned, stage.returned_errors),
                (0, 0, 0)
            );
            assert_eq!(stage.wall_seconds, 0.0);
        }
        let mut old = vec![0.0; crate::public_model::PARAMETER_COUNT];
        old[0] = -0.0;
        old[1] = 1.0;
        let mut gradient = vec![0.0; old.len()];
        gradient[1] = 2.0;
        gradient[2] = -3.0;
        let config = OneStepConfig::default();
        let mut profile = CurrentMixedAttemptTimings::default();
        let calculate = |timer: &mut CurrentMixedAttemptTimings, gradient: &[f64]| {
            timed(timer, TimedPhase::WholeCall, |timer| {
                timed(timer, TimedPhase::ActorProposal, |_| {
                    propose_parameters(&old, gradient, &config)
                })
            })
        };
        let plain = timed(&mut Untimed, TimedPhase::ActorProposal, |_| {
            propose_parameters(&old, &gradient, &config)
        })
        .unwrap();
        let observed = calculate(&mut profile, &gradient).unwrap();
        assert_eq!(
            plain
                .parameters
                .iter()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>(),
            observed
                .parameters
                .iter()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>()
        );
        assert_eq!(
            serde_json::to_value(plain.numeric).unwrap(),
            serde_json::to_value(observed.numeric).unwrap()
        );
        gradient[1] = f64::NAN;
        let plain_error = propose_parameters(&old, &gradient, &config).err().unwrap();
        assert_eq!(
            calculate(&mut profile, &gradient).err().unwrap(),
            plain_error
        );
        for stage in [
            &profile.stages.whole_call_inclusive,
            &profile.stages.actor_rounded_proposal,
        ] {
            assert_eq!(
                (stage.entered, stage.returned, stage.returned_errors),
                (2, 2, 1)
            );
        }
        assert_eq!(profile.stages.collection_native[0].entered, 0);
        assert!(!profile.reservation_charged);
        assert!(profile.new_resolution.is_none());
        for phase in surrounding {
            let plain = timed(&mut Untimed, phase, |_| Ok(7_u64)).unwrap();
            let observed = timed(&mut profile, phase, |_| Ok(7_u64)).unwrap();
            assert_eq!(observed, plain);
            let failed = timed(&mut profile, phase, |_| {
                Err::<(), _>("synthetic surrounding-stage error".into())
            });
            assert_eq!(failed.unwrap_err(), "synthetic surrounding-stage error");
            let stage = profile.stage(phase);
            assert_eq!(
                (stage.entered, stage.returned, stage.returned_errors),
                (2, 2, 1)
            );
        }
        serde_json::to_value(profile).unwrap();
    }

    #[test]
    fn current_mixed_private_base_preserves_no_change_consumption() {
        let (mut p, mut s, template) = reserved();
        p.base = 49; // Private synthetic state, never a legacy Start or real owner.
        s.max_attempts = 3;
        s.max_successful_updates = 3;
        s.max_journal_entries = 64;
        s.validate().unwrap();
        assert_eq!(p.parent_count().unwrap(), 49);
        for (attempt, updated) in [true, false, true].into_iter().enumerate() {
            let r = later_reservation(&p, &template);
            p.apply(
                &Event::ReserveWhole4 {
                    reservation: Box::new(r),
                },
                &s,
            )
            .unwrap();
            gradient(&mut p, &s);
            let resolution = if updated {
                Resolution::Updated {
                    report: file_pin("report.json", b"report"),
                    artifact: file_pin("artifact.json", b"artifact"),
                    report_checksum: "b".repeat(64),
                    artifact_checksum: if attempt == 0 { "c" } else { "d" }.repeat(64),
                }
            } else {
                Resolution::NoChange {
                    report: file_pin("report.json", b"report"),
                    report_checksum: "b".repeat(64),
                }
            };
            p.apply(&Event::Resolve { resolution }, &s).unwrap();
            assert_eq!(p.parent_count().unwrap(), [50, 50, 51][attempt]);
            let attempts = attempt as u64 + 1;
            assert_eq!(
                (p.phase, p.attempts, p.imported),
                (Phase::Ready, attempts, 1)
            );
            assert_eq!(p.next, 9 + 4 * attempts);
            assert_eq!(p.consumed.len(), 4 * attempts as usize);
            assert_eq!(
                p.work,
                Work {
                    callbacks: 4 * attempts,
                    rows: 4 * attempts,
                    search: 0,
                    bytes: 64 * attempts,
                }
            );
            assert_eq!(
                p.current_checksum,
                if attempt < 2 { "c" } else { "d" }.repeat(64)
            );
        }
        assert_eq!(p.successful, 2);
    }
    #[test]
    fn current_mixed_legacy_start_refuses_later_private_base() {
        let (mut p, s, _) = reserved();
        p.started = false;
        p.base = 49;
        let parent = continued_parent(&p);
        let start = Event::Start {
            run_spec: file_pin("run-spec.json", b"spec"),
            parent: Box::new(parent),
            prefix: p.prefix,
            next: p.next,
            consumed: vec![],
        };
        for (base, successful) in [(1, 0), (49, 0), (1, 48)] {
            p.base = base;
            p.successful = successful;
            assert!(p.apply(&start, &s).is_err());
            assert!(!p.started);
        }
    }
    #[test]
    fn current_mixed_private_counts_refuse_overflow_and_ceiling_before_reserve() {
        let (mut p, s, template) = reserved();
        p.successful = u64::MAX;
        assert!(p.parent_count().is_err());
        p.successful = 0;
        for base in [0, source::MAX_MIXED_UPDATES + 1] {
            p.base = base;
            assert!(p.parent_count().is_err());
        }
        p.base = source::MAX_MIXED_UPDATES - 1;
        let r = later_reservation(&p, &template);
        p.apply(
            &Event::ReserveWhole4 {
                reservation: Box::new(r),
            },
            &s,
        )
        .unwrap();
        gradient(&mut p, &s);
        p.apply(
            &Event::Resolve {
                resolution: Resolution::Updated {
                    report: file_pin("report.json", b"report"),
                    artifact: file_pin("artifact.json", b"artifact"),
                    report_checksum: "b".repeat(64),
                    artifact_checksum: "c".repeat(64),
                },
            },
            &s,
        )
        .unwrap();
        assert_eq!(p.parent_count().unwrap(), source::MAX_MIXED_UPDATES);
        assert!(p.next_parent_count().is_err());
        let r = later_reservation(&p, &template);
        let before = (p.attempts, p.next, p.consumed.clone(), p.work);
        assert!(
            p.apply(
                &Event::ReserveWhole4 {
                    reservation: Box::new(r)
                },
                &s
            )
            .is_err()
        );
        assert_eq!((p.attempts, p.next, p.consumed.clone(), p.work), before);
    }
    fn test_directory(label: &str) -> PathBuf {
        let directory = std::env::temp_dir().join(format!(
            "tzolkin-mixed-{label}-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir(&directory).unwrap();
        directory
    }
    #[test]
    fn current_mixed_owner_pins_keep_their_absolute_batch_directory() {
        let directory = test_directory("owner-pins");
        let ancestor = directory.join("ancestor");
        let active = directory.join("active");
        fs::create_dir(&ancestor).unwrap();
        fs::create_dir(&active).unwrap();
        fs::write(ancestor.join("report.json"), b"old report").unwrap();
        fs::write(ancestor.join("artifact.json"), b"old artifact").unwrap();
        fs::write(active.join("report.json"), b"new report").unwrap();
        fs::write(active.join("artifact.json"), b"new artifact").unwrap();
        let mut pins = (
            ancestor,
            file_pin("report.json", b"old report"),
            file_pin("artifact.json", b"old artifact"),
        );
        assert!(pins.0.is_absolute());
        require_owner_pins(&pins).unwrap();
        pins.0 = active;
        assert!(require_owner_pins(&pins).is_err());
        fs::remove_dir_all(directory).unwrap();
    }
    #[test]
    fn current_mixed_archive_view_checks_layout_chain_and_byte_counters() {
        for s in [spec(), compressed_spec()] {
            let directory = test_directory("archive-view");
            let raw_spec = bytes(&s, SPEC_CAP).unwrap();
            let spec_pin = file_pin("run-spec.json", &raw_spec);
            let raw = bytes(
                &Entry {
                    schema: s.journal_schema().into(),
                    sequence: 1,
                    previous: String::new(),
                    root: "root".into(),
                    spec: s.checksum().unwrap(),
                    pool: "pool".into(),
                    event: Event::Start {
                        run_spec: spec_pin.clone(),
                        parent: Box::new(CurrentMixedParentIdentity::InitialMixedV1 {
                            policy: source::tests::initial(),
                        }),
                        prefix: [7, 11],
                        next: 9,
                        consumed: vec![],
                    },
                },
                HEAD_CAP,
            )
            .unwrap();
            let head = file_pin("head-0000000000000001.json", &raw);
            fs::write(directory.join(&spec_pin.name), &raw_spec).unwrap();
            fs::write(directory.join(&head.name), &raw).unwrap();
            let references = BTreeMap::new();
            let metadata = (raw_spec.len() + raw.len()) as u64;
            let mut view = ArchiveView {
                directory: &directory,
                spec: &s,
                journal_schema: s.journal_schema(),
                root_checksum: "root",
                pool: "pool",
                sequence: 1,
                head: Some(&head),
                spec_pin: Some(&spec_pin),
                references: &references,
                output_bytes: metadata,
                metadata_bytes: metadata,
                raw_record_bytes: 0,
                stored_record_bytes: 0,
            };
            restore::require_archive_files(&view).unwrap();
            view.metadata_bytes += 1;
            assert!(restore::require_archive_files(&view).is_err());
            view.metadata_bytes = metadata;
            view.raw_record_bytes = 1;
            assert!(restore::require_archive_files(&view).is_err());
            view.raw_record_bytes = 0;
            fs::write(directory.join("orphan.tmp"), b"orphan").unwrap();
            assert!(restore::require_archive_files(&view).is_err());
            fs::remove_file(directory.join("orphan.tmp")).unwrap();
            let mut changed: Entry = closed(&raw, HEAD_CAP).unwrap();
            if let Event::Start { next, .. } = &mut changed.event {
                *next += 1;
            }
            // A syntactically valid replacement still refuses the final raw
            // barrier; semantic state from an earlier read cannot repin it.
            fs::write(
                directory.join(&head.name),
                bytes(&changed, HEAD_CAP).unwrap(),
            )
            .unwrap();
            assert!(restore::require_archive_files(&view).is_err());
            fs::write(directory.join(&head.name), &raw).unwrap();
            restore::require_archive_files(&view).unwrap();
            fs::write(directory.join(&head.name), b"changed").unwrap();
            assert!(restore::require_archive_files(&view).is_err());
            fs::remove_dir_all(directory).unwrap();
        }
    }
    #[test]
    fn current_mixed_ready_continuation_preserves_updated_and_no_change_reservations() {
        for updated in [false, true] {
            let (mut p, s, r) = reserved();
            p.apply(
                &Event::ReserveWhole4 {
                    reservation: Box::new(r),
                },
                &s,
            )
            .unwrap();
            gradient(&mut p, &s);
            let resolution = if updated {
                Resolution::Updated {
                    report: file_pin("report.json", b"report"),
                    artifact: file_pin("artifact.json", b"artifact"),
                    report_checksum: "b".repeat(64),
                    artifact_checksum: "c".repeat(64),
                }
            } else {
                Resolution::NoChange {
                    report: file_pin("report.json", b"report"),
                    report_checksum: "b".repeat(64),
                }
            };
            p.apply(&Event::Resolve { resolution }, &s).unwrap();
            assert_eq!(
                (p.phase, p.attempts, p.successful, p.imported, p.next),
                (Phase::Ready, 1, u64::from(updated), 1, 13)
            );
            assert_eq!(p.consumed.len(), 4);
            assert!(p.active.is_some());
            assert_eq!(
                p.current_checksum,
                if updated { "c" } else { "a" }.repeat(64)
            );
            let reserved_before = p.consumed.clone();
            let active_before = bytes(&p.active, HEAD_CAP).unwrap();
            let work_before = p.work;
            p.require_ready_continuation(false, true).unwrap();
            assert_eq!(p.consumed, reserved_before);
            assert_eq!(bytes(&p.active, HEAD_CAP).unwrap(), active_before);
            assert_eq!(p.work, work_before);
            assert!(p.require_ready_continuation(true, true).is_err());
            assert!(p.require_ready_continuation(false, false).is_err());
            p.unknown_work = true;
            assert!(p.require_ready_continuation(false, true).is_err());
        }
    }
    #[test]
    fn current_mixed_ready_continuation_refuses_empty_pending_and_stopped_states() {
        let (mut p, s, r) = reserved();
        assert!(p.require_ready_continuation(false, true).is_err());
        p.apply(
            &Event::ReserveWhole4 {
                reservation: Box::new(r),
            },
            &s,
        )
        .unwrap();
        assert!(p.require_ready_continuation(false, true).is_err());
        for member in 0..4 {
            p.apply(&Event::CollectStart { member }, &s).unwrap();
            assert!(p.require_ready_continuation(false, true).is_err());
            p.apply(
                &Event::CollectReturn {
                    member,
                    record: Some(RecordPin::Raw(file_pin(
                        &format!("record-{member}.json"),
                        b"record",
                    ))),
                    complete: true,
                    error: None,
                },
                &s,
            )
            .unwrap();
            assert!(p.require_ready_continuation(false, true).is_err());
        }
        for member in 0..4 {
            p.apply(&Event::AuditStart { member }, &s).unwrap();
            assert!(p.require_ready_continuation(false, true).is_err());
            p.apply(
                &Event::AuditReturn {
                    member,
                    passed: true,
                    episode_checksum: Some(format!("{member:064x}")),
                    error: None,
                },
                &s,
            )
            .unwrap();
            assert!(p.require_ready_continuation(false, true).is_err());
        }
        p.apply(
            &Event::GradientStart {
                cohort: file_pin("cohort.json", b"cohort"),
                cohort_checksum: "a".repeat(64),
                episode_checksums: std::array::from_fn(|i| format!("{i:064x}")),
                counts: [4, 4, 4],
            },
            &s,
        )
        .unwrap();
        assert!(p.require_ready_continuation(false, true).is_err());
        p.apply(
            &Event::Resolve {
                resolution: Resolution::Abandoned {
                    reason: "explicit stop".into(),
                },
            },
            &s,
        )
        .unwrap();
        assert!(p.unknown_work);
        assert!(p.require_ready_continuation(false, true).is_err());
        let (mut failed, s, r) = reserved();
        failed
            .apply(
                &Event::ReserveWhole4 {
                    reservation: Box::new(r),
                },
                &s,
            )
            .unwrap();
        failed
            .apply(&Event::CollectStart { member: 0 }, &s)
            .unwrap();
        failed
            .apply(
                &Event::CollectReturn {
                    member: 0,
                    record: None,
                    complete: false,
                    error: Some("returned failure".into()),
                },
                &s,
            )
            .unwrap();
        assert!(failed.require_ready_continuation(false, true).is_err());
        failed
            .apply(
                &Event::Resolve {
                    resolution: Resolution::Failed {
                        error: "returned failure".into(),
                    },
                },
                &s,
            )
            .unwrap();
        assert!(!failed.unknown_work);
        assert!(failed.require_ready_continuation(false, true).is_err());
        assert_eq!(
            (failed.next, failed.consumed.len(), failed.attempts),
            (13, 4, 1)
        );
    }
    #[test]
    fn current_mixed_durable_reservation_refuses_reuse_and_retains_known_failure() {
        let (mut p, s, r) = reserved();
        let mut wrong = r.clone();
        wrong.configurations[2].episode_ordinal = "0000000000000000".into();
        assert!(
            p.check(
                &Event::ReserveWhole4 {
                    reservation: Box::new(wrong)
                },
                &s
            )
            .is_err()
        );
        assert_eq!(p.next, 9);
        assert!(p.consumed.is_empty());
        p.apply(
            &Event::ReserveWhole4 {
                reservation: Box::new(r.clone()),
            },
            &s,
        )
        .unwrap();
        assert_eq!(
            (p.next, p.attempts, p.successful, p.imported),
            (13, 1, 0, 1)
        );
        assert_eq!(p.consumed.len(), 4);
        assert!(
            p.check(
                &Event::ReserveWhole4 {
                    reservation: Box::new(r)
                },
                &s
            )
            .is_err()
        );
        p.apply(&Event::CollectStart { member: 0 }, &s).unwrap();
        p.apply(
            &Event::CollectReturn {
                member: 0,
                record: None,
                complete: false,
                error: Some("returned error".into()),
            },
            &s,
        )
        .unwrap();
        p.apply(
            &Event::Resolve {
                resolution: Resolution::Failed {
                    error: "returned error".into(),
                },
            },
            &s,
        )
        .unwrap();
        assert_eq!(p.phase, Phase::Stopped);
        assert!(!p.unknown_work);
        assert_eq!((p.next, p.consumed.len(), p.successful), (13, 4, 0));
        assert!(p.check(&Event::CollectStart { member: 1 }, &s).is_err());
    }
    #[test]
    fn current_mixed_durable_only_resolved_update_advances_model_count() {
        let (mut p, s, r) = reserved();
        p.apply(
            &Event::ReserveWhole4 {
                reservation: Box::new(r),
            },
            &s,
        )
        .unwrap();
        gradient(&mut p, &s);
        p.apply(
            &Event::Resolve {
                resolution: Resolution::NoChange {
                    report: file_pin("report.json", b"report"),
                    report_checksum: "b".repeat(64),
                },
            },
            &s,
        )
        .unwrap();
        assert_eq!((p.successful, p.imported, p.phase), (0, 1, Phase::Ready));
        assert_eq!(p.current_checksum, "a".repeat(64));
        let (mut updated, s, r) = reserved();
        updated
            .apply(
                &Event::ReserveWhole4 {
                    reservation: Box::new(r),
                },
                &s,
            )
            .unwrap();
        gradient(&mut updated, &s);
        updated
            .apply(
                &Event::Resolve {
                    resolution: Resolution::Updated {
                        report: file_pin("report.json", b"report"),
                        artifact: file_pin("artifact.json", b"artifact"),
                        report_checksum: "b".repeat(64),
                        artifact_checksum: "c".repeat(64),
                    },
                },
                &s,
            )
            .unwrap();
        assert_eq!(
            (updated.successful, updated.imported, updated.next),
            (1, 1, 13)
        );
        assert_eq!(updated.current_checksum, "c".repeat(64));
        let (mut pending, s, r) = reserved();
        pending
            .apply(
                &Event::ReserveWhole4 {
                    reservation: Box::new(r),
                },
                &s,
            )
            .unwrap();
        pending
            .apply(&Event::CollectStart { member: 0 }, &s)
            .unwrap();
        pending
            .apply(
                &Event::Resolve {
                    resolution: Resolution::Abandoned {
                        reason: "explicit stop".into(),
                    },
                },
                &s,
            )
            .unwrap();
        assert!(pending.unknown_work);
        assert_eq!(
            (pending.successful, pending.consumed.len(), pending.next),
            (0, 4, 13)
        );
        assert!(
            pending
                .check(
                    &Event::CollectReturn {
                        member: 0,
                        record: None,
                        complete: false,
                        error: Some("late return".into())
                    },
                    &s
                )
                .is_err()
        );
    }
    #[test]
    fn current_mixed_durable_wire_is_closed_and_has_no_executable_owner_route() {
        let raw = bytes(&spec(), SPEC_CAP).unwrap();
        let text = String::from_utf8(raw.clone()).unwrap();
        let duplicate = text.replacen(
            "\"maxAttempts\":2",
            "\"maxAttempts\":2,\"maxAttempts\":1",
            1,
        );
        assert!(CurrentMixedRunSpec::from_json(duplicate.as_bytes()).is_err());
        let mut value: serde_json::Value = serde_json::from_slice(&raw).unwrap();
        value["acceptedOwner"] = true.into();
        assert!(CurrentMixedRunSpec::from_json(&serde_json::to_vec(&value).unwrap()).is_err());
        let (p, s, r) = reserved();
        let mut wrong = r;
        wrong.parent = CurrentMixedParentIdentity::InitialMixedV1 {
            policy: source::tests::initial(),
        };
        if let CurrentMixedParentIdentity::InitialMixedV1 { policy } = &mut wrong.parent {
            let mut v = serde_json::to_value(&*policy).unwrap();
            v["artifactChecksum"] = "f".repeat(64).into();
            *policy = serde_json::from_value(v).unwrap();
        }
        assert!(
            p.check(
                &Event::ReserveWhole4 {
                    reservation: Box::new(wrong)
                },
                &s
            )
            .is_err()
        );
        assert!(closed::<Entry>(br#"{"schema":"a","schema":"b"}"#, HEAD_CAP).is_err());
        let (_, _, r) = reserved();
        assert!(
            serde_json::from_value::<crate::public_rl_mixed_native::MixedLearnerIdentity>(
                serde_json::to_value(r.parent).unwrap()
            )
            .is_err()
        );
    }
}
