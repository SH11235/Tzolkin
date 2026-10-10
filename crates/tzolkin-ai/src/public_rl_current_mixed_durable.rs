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
use crate::public_policy_update::{OneStepConfig, checkpoint_digest, hash};
use crate::public_rl_current_mixed_session::{
    AdmittedCurrentMixedRoot, CurrentMixedCollectionSpec, OriginalLivePins,
};
use crate::public_rl_current_mixed_source::{
    self as source, CurrentMixedParentIdentity, PlannedCurrentMixedCohort,
};
use crate::public_rl_mixed_native::FrozenMixedRoster;
use crate::public_rl_mixed_session::{FilePin, OpponentSpec, read_pin};
use crate::public_rl_paired_session::{local, publish_mixed_metadata, read_bounded};
use crate::public_stochastic_record::{Config, MAX_RECORD_BYTES, unique_json};
use crate::search::PreparedSearch;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    path::{Path, PathBuf},
};

#[path = "public_rl_current_mixed_restore.rs"]
mod restore;

pub const SPEC_SCHEMA: &str = "tzolkin-current-mixed-durable-spec-v2";
pub const JOURNAL_SCHEMA: &str = "tzolkin-current-mixed-durable-journal-v2";
const SPEC_CAP: usize = 64 * 1024;
const HEAD_CAP: usize = 512 * 1024;
const ARTIFACT_CAP: usize = 16 * 1024 * 1024;
const EVENTS_PER_ATTEMPT: u64 = 21;
const MAX_ATTEMPTS: u64 = 64;
const MAX_FAMILIES: usize = 4096;
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
}
impl CurrentMixedRunSpec {
    pub fn from_json(raw: &[u8]) -> Result<Self, String> {
        let s: Self = closed(raw, SPEC_CAP)?;
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
        hash(b"tzolkin-current-mixed-durable-spec-v2\0", self)
    }
    fn validate(&self) -> Result<(), String> {
        self.actor()?;
        self.critic()?;
        if self.schema != SPEC_SCHEMA
            || !(1..=MAX_ATTEMPTS).contains(&self.max_attempts)
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
    ReserveWhole4 {
        reservation: Box<Reservation>,
    },
    CollectStart {
        member: usize,
    },
    CollectReturn {
        member: usize,
        record: Option<FilePin>,
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
struct Progress {
    started: bool,
    phase: Phase,
    prefix: [u64; 2],
    next: u64,
    consumed: BTreeSet<String>,
    attempts: u64,
    successful: u64,
    imported: u64,
    current_checksum: String,
    work: Work,
    active: Option<Reservation>,
    collect: [Option<FilePin>; 4],
    audited: [Option<String>; 4],
    cohort: Option<FilePin>,
    unknown_work: bool,
    outcome: Option<String>,
}
impl Progress {
    fn initial(
        parent: &CurrentMixedParentIdentity,
        prefix: [u64; 2],
        next: u64,
        consumed: &BTreeSet<String>,
    ) -> Self {
        Self {
            started: false,
            phase: Phase::Ready,
            prefix,
            next,
            consumed: consumed.clone(),
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
        }
    }
    fn check(&self, e: &Event, s: &CurrentMixedRunSpec) -> Result<(), String> {
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
                    || self.attempts != 0
                    || self.phase != Phase::Ready
                    || parent.mixed_update_count() != 1
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
            Event::ReserveWhole4 { reservation: r } => {
                r.parent.validate()?;
                leaf(&r.collection_spec, "collection-spec.json", SPEC_CAP)?;
                leaf(&r.plan, "plan.json", HEAD_CAP)?;
                if !self.started
                    || self.phase != Phase::Ready
                    || self.attempts >= s.max_attempts
                    || self.successful >= s.max_successful_updates
                    || r.attempt != self.attempts
                    || r.parent.artifact_checksum() != self.current_checksum
                    || r.parent.mixed_update_count() != 1 + self.successful
                    || r.parent.imported_update_count() != self.imported
                    || !checkpoint_digest(&r.plan_checksum)
                    || self
                        .consumed
                        .len()
                        .checked_add(4)
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
                    leaf(p, &format!("record-{i}.json"), MAX_RECORD_BYTES)?;
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
        match e {
            Event::Start { .. } => {
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
            Event::GradientStart { cohort, .. } => {
                self.cohort = Some(cohort.clone());
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
    owner_pins: Option<(u64, FilePin, FilePin)>,
    references: BTreeMap<u64, Vec<FilePin>>,
    output_bytes: u64,
    metadata_bytes: u64,
    blocked: bool,
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
}
impl<'a> CurrentMixedTrainingSession<'a> {
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
            output_bytes: 0,
            metadata_bytes: 0,
            blocked: false,
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
            || h.identity().mixed_update_count() != self.progress.successful + 1
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
    pub fn continued_owner(&self) -> Option<&ContinuedMixedOwner> {
        self.continued.as_ref()
    }
    pub fn reserved_families(&self) -> &BTreeSet<String> {
        &self.progress.consumed
    }
    pub fn status(&self) -> CurrentMixedSessionStatus {
        CurrentMixedSessionStatus {
            schema: JOURNAL_SCHEMA,
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
            mixed_update_count: 1 + self.progress.successful,
            attempts: self.progress.attempts,
            next_episode_ordinal: self.progress.next,
            reserved_family_count: self.progress.consumed.len(),
            phase: self.progress.phase,
            stopped: self.blocked || self.progress.phase == Phase::Stopped,
            unknown_work_possible: self.progress.unknown_work,
            outcome: self.progress.outcome.clone(),
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
        if let Some((attempt, report, artifact)) = &self.owner_pins {
            read_pin(&self.batch(*attempt), report, HEAD_CAP)?;
            read_pin(&self.batch(*attempt), artifact, ARTIFACT_CAP)?;
        }
        if let Some(r) = &self.progress.active {
            for p in self.references.get(&r.attempt).into_iter().flatten() {
                read_pin(&self.batch(r.attempt), p, p.bytes)?;
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
        if output > self.spec.max_output_bytes || metadata > self.spec.max_metadata_bytes {
            return Err("Current mixed storage budget consumed".into());
        }
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
            .checked_add(if name.starts_with("record-") {
                0
            } else {
                raw.len() as u64
            })
            .ok_or("Current mixed metadata overflow")?;
        let refs = self.references.entry(attempt).or_default();
        if raw.is_empty()
            || raw.len() > cap
            || output > self.spec.max_output_bytes
            || metadata > self.spec.max_metadata_bytes
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
    fn save(&mut self, event: Event) -> Result<(), String> {
        self.progress.check(&event, &self.spec)?;
        self.require_live()?;
        let sequence = self
            .sequence
            .checked_add(1)
            .filter(|n| *n <= self.spec.max_journal_entries)
            .ok_or("Current mixed journal exhausted")?;
        let entry = Entry {
            schema: JOURNAL_SCHEMA.into(),
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
        let parent = self.handle()?;
        spec.inner
            .wire
            .rosters
            .iter()
            .map(|seats| {
                FrozenMixedRoster::for_current(
                    seats
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
                                    BorrowedArenaPolicy::CornFirstUxmalOpening(
                                        &spec.inner.wire.weights,
                                    )
                                }
                                OpponentSpec::PreparedBc => {
                                    BorrowedArenaPolicy::PublicBc(self.prepared)
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
                        .collect(),
                    parent.identity(),
                )
            })
            .collect::<Result<Vec<_>, _>>()?
            .try_into()
            .map_err(|_| "Current mixed roster shape".into())
    }
    fn plan(&self, spec: &CurrentMixedCollectionSpec) -> Result<PlannedCurrentMixedCohort, String> {
        if spec.pool_checksum()? != self.pool {
            return Err("Current mixed immutable pool/rosters changed".into());
        }
        let handle = self.handle()?;
        let configs = spec.inner.configs()?;
        let rosters = self.rosters(spec)?;
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
                .checked_add(
                    EVENTS_PER_ATTEMPT * HEAD_CAP as u64
                        + ARTIFACT_CAP as u64
                        + 2 * HEAD_CAP as u64,
                )
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
                .checked_add(
                    EVENTS_PER_ATTEMPT * HEAD_CAP as u64
                        + ARTIFACT_CAP as u64
                        + 2 * HEAD_CAP as u64,
                )
                .is_none_or(|n| n > self.spec.max_metadata_bytes)
        {
            return Err("Current mixed terminal/failure publication headroom".into());
        }
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
        let batch = self.preflight(&collection)?;
        self.blocked = true;
        let result = self.run_whole_four(collection, batch);
        if result.is_err() {
            self.blocked = true;
        }
        result
    }
    fn run_whole_four(
        &mut self,
        collection: CurrentMixedCollectionSpec,
        batch: BatchPlan,
    ) -> Result<CurrentMixedSessionStatus, String> {
        let BatchPlan {
            plan,
            collection_raw,
            plan_raw,
            reservation,
        } = batch;
        let attempt = reservation.attempt;
        let dir = self.batch(attempt);
        local(&dir, true)?;
        fs::create_dir(&dir).map_err(|e| e.to_string())?;
        if self.publish_batch(attempt, "collection-spec.json", &collection_raw, SPEC_CAP)?
            != reservation.collection_spec
            || self.publish_batch(attempt, "plan.json", &plan_raw, HEAD_CAP)? != reservation.plan
        {
            return Err("Current mixed original planned pins differ".into());
        }
        let plan_value = plan.receipt();
        self.save(Event::ReserveWhole4 {
            reservation: Box::new(reservation),
        })?;
        for member in 0..4 {
            self.save(Event::CollectStart { member })?;
            self.require_live()?;
            let returned = {
                let handle = self.handle()?;
                let rosters = self.rosters(&collection)?;
                source::collect(
                    &plan.configurations()[member],
                    &handle,
                    &rosters[member],
                    plan.headers[member].clone(),
                )
            };
            self.require_live()?;
            let (record, complete, error) = match returned {
                Ok(game) => {
                    let complete = game.terminal.is_some() && game.failure.is_none();
                    let raw = source::encode(&game)?;
                    source::check_record(&raw, &plan.headers[member])?;
                    let p = self.publish_batch(
                        attempt,
                        &format!("record-{member}.json"),
                        &raw,
                        MAX_RECORD_BYTES,
                    )?;
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
            self.save(Event::CollectReturn {
                member,
                record,
                complete,
                error: error.clone(),
            })?;
            if let Some(e) = error {
                return self.stopped_error(e);
            }
        }
        let mut episodes = Vec::new();
        for member in 0..4 {
            let record = self.progress.collect[member]
                .as_ref()
                .ok_or("Missing returned current mixed record")?;
            let raw = read_pin(&dir, record, MAX_RECORD_BYTES)?;
            source::check_record(&raw, &plan.headers[member])?;
            self.save(Event::AuditStart { member })?;
            self.require_live()?;
            let returned = {
                let handle = self.handle()?;
                let rosters = self.rosters(&collection)?;
                source::audit(
                    &raw,
                    &handle,
                    &rosters[member],
                    plan.headers[member].clone(),
                )
            };
            self.require_live()?;
            self.save(Event::AuditReturn {
                member,
                passed: returned.is_ok(),
                episode_checksum: returned
                    .as_ref()
                    .ok()
                    .map(|e| e.canonical_record_checksum().into()),
                error: returned.as_ref().err().map(|e| bounded_error(e)),
            })?;
            match returned {
                Ok(e) => episodes.push(Ok(e)),
                Err(e) => return self.stopped_error(e),
            }
        }
        let cohort = match plan.finish(
            episodes
                .try_into()
                .map_err(|_| "Current mixed sealed whole-four shape")?,
        ) {
            Ok(c) => c,
            Err(e) => return self.stopped_error(e),
        };
        let cohort_value = cohort.receipt();
        let cohort_pin = self.publish_batch(
            attempt,
            "cohort.json",
            &bytes(&cohort_value, HEAD_CAP)?,
            HEAD_CAP,
        )?;
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
        })?;
        self.require_live()?;
        let result = mixed::prepare_current(
            &self.handle()?,
            &cohort,
            &self.spec.actor()?,
            &self.spec.critic()?,
        );
        self.require_live()?;
        let proposal = match result {
            Ok(p) => p,
            Err(e) => return self.stopped_error(e),
        };
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
        if proposal.artifact_for_publication().is_some() {
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
            self.owner_pins = Some((attempt, report, artifact));
        } else {
            self.require_batch_inventory(attempt)?;
            self.save(Event::Resolve {
                resolution: Resolution::NoChange {
                    report,
                    report_checksum: proposal.report().checksum().into(),
                },
            })?;
        }
        self.require_live()?;
        self.handle()?;
        self.blocked = false;
        Ok(self.status())
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
            || report.proposed_mixed_update_count() != 2 + self.progress.successful
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
            read_pin(&dir, p, p.bytes)?;
        }
        Ok(())
    }
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
    fn reserved() -> (Progress, CurrentMixedRunSpec, Reservation) {
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
    fn gradient(p: &mut Progress, s: &CurrentMixedRunSpec) {
        for member in 0..4 {
            p.apply(&Event::CollectStart { member }, s).unwrap();
            p.apply(
                &Event::CollectReturn {
                    member,
                    record: Some(file_pin(&format!("record-{member}.json"), b"record")),
                    complete: true,
                    error: None,
                },
                s,
            )
            .unwrap();
        }
        for member in 0..4 {
            p.apply(&Event::AuditStart { member }, s).unwrap();
            p.apply(
                &Event::AuditReturn {
                    member,
                    passed: true,
                    episode_checksum: Some(format!("{member:064x}")),
                    error: None,
                },
                s,
            )
            .unwrap();
        }
        p.apply(
            &Event::GradientStart {
                cohort: file_pin("cohort.json", b"cohort"),
                cohort_checksum: "a".repeat(64),
                episode_checksums: std::array::from_fn(|i| format!("{i:064x}")),
                counts: [4, 4, 4],
            },
            s,
        )
        .unwrap();
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
