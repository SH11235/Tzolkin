//! Bounded successor admission consumes a genuine Ready joint owner. The ledger
//! covers one lineage; restoring an older predecessor can still create a fork.
use super::*;
use crate::public_rl_mixed_session::require_output_outside_directory;

pub const SEGMENT_SPEC_SCHEMA: &str = "tzolkin-current-mixed-segment-spec-v1";
pub const SEGMENT_JOURNAL_SCHEMA: &str = "tzolkin-current-mixed-segment-journal-v1";
pub const GLOBAL_CONTRACT_SCHEMA: &str = "tzolkin-current-mixed-global-contract-v1";
const MAX_SEGMENTS: usize = 64;
const MAX_PROTECTED_DIRECTORIES: usize = 16;

/// A finite local run and immutable global limits. This is metadata, never a
/// constructor for a policy owner. All amounts include the predecessor's spent
/// ledger of the first durable run and its successors. Borrowed original input
/// files remain external. Unused local capacity is not carried forward.
/// `maxSegments` includes the first durable run. Optional `protectedDirectories`
/// fixes at most sixteen existing absolute source/cache directories for this
/// lineage, independently of policy authentication.
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct CurrentMixedSegmentSpec {
    schema: String,
    run_spec: CurrentMixedRunSpec,
    global_contract: GlobalContract,
}
impl CurrentMixedSegmentSpec {
    pub fn from_json(raw: &[u8]) -> Result<Self, String> {
        let value = unique_json(raw, SPEC_CAP)?;
        let run_raw = serde_json::to_vec(&value["runSpec"]).map_err(|e| e.to_string())?;
        CurrentMixedRunSpec::from_json(&run_raw)?;
        let s: Self = serde_json::from_value(value).map_err(|e| e.to_string())?;
        s.validate()?;
        Ok(s)
    }
    fn validate(&self) -> Result<(), String> {
        if self.schema != SEGMENT_SPEC_SCHEMA {
            return Err("Current mixed segment specification schema".into());
        }
        self.run_spec.validate()?;
        self.global_contract.validate()?;
        bytes(self, SPEC_CAP)?;
        Ok(())
    }
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct GlobalContract {
    schema: String,
    max_segments: usize,
    max_attempts: u64,
    max_successful_updates: u64,
    max_families: usize,
    max_callbacks: u64,
    max_candidate_rows: u64,
    max_search_steps: u64,
    max_declared_work_bytes: u64,
    max_logical_record_bytes: u64,
    max_stored_record_bytes: u64,
    max_metadata_bytes: u64,
    max_temporary_bytes: u64,
    max_disk_bytes: u64,
    // Caller-declared restrictions protect source/cache directories that the
    // borrowed Prepared policy does not remember. They do not authenticate it.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    protected_directories: Vec<PathBuf>,
}
impl GlobalContract {
    fn validate(&self) -> Result<(), String> {
        if self.protected_directories.len() > MAX_PROTECTED_DIRECTORIES
            || self.protected_directories.iter().any(|p| {
                !p.is_absolute() || p.to_str().is_none_or(|s| s.is_empty() || s.len() > 1024)
            })
        {
            return Err("Current mixed finite protected-directory restrictions".into());
        }
        if self.schema != GLOBAL_CONTRACT_SCHEMA
            || !(2..=MAX_SEGMENTS).contains(&self.max_segments)
            || self.max_attempts == 0
            || self.max_successful_updates == 0
            || self.max_successful_updates > self.max_attempts
            || self.max_successful_updates >= source::MAX_MIXED_UPDATES
            || !(4..=MAX_FAMILIES).contains(&self.max_families)
            || self.max_callbacks == 0
            || self.max_candidate_rows == 0
            || self.max_declared_work_bytes == 0
            || self.max_logical_record_bytes == 0
            || self.max_stored_record_bytes == 0
            || self.max_metadata_bytes == 0
            || self.max_stored_record_bytes > self.max_disk_bytes
            || self.max_metadata_bytes > self.max_disk_bytes
            || !(MAX_RECORD_BYTES as u64..=512 * 1024 * 1024).contains(&self.max_temporary_bytes)
            || !(512 * 1024 * 1024..=MAX_STORAGE).contains(&self.max_disk_bytes)
        {
            return Err("Current mixed finite global contract limits".into());
        }
        Ok(())
    }
    fn require_output(&self, output: &Path) -> Result<(), String> {
        for directory in &self.protected_directories {
            let directory = local(directory, false)?;
            if !directory.is_dir() {
                return Err("Current mixed protected directory is not a directory".into());
            }
            require_output_outside_directory(&directory, output)?;
        }
        Ok(())
    }
    fn checksum(&self) -> Result<String, String> {
        hash(b"tzolkin-current-mixed-global-contract-v1\0", self)
    }
    fn envelope(&self, spent: Spent, local: &CurrentMixedRunSpec) -> Result<(), String> {
        self.validate()?;
        local.validate()?;
        let work = spent.reserved_work.add(Work {
            callbacks: local.max_callbacks,
            rows: local.max_candidate_rows,
            search: local.max_search_steps,
            bytes: local.max_output_bytes,
        })?;
        let (stored, temporary, disk) = match &local.record_storage {
            Some(s) => (s.max_stored_bytes, s.max_temporary_bytes, s.max_disk_bytes),
            None => (
                local.max_output_bytes,
                MAX_RECORD_BYTES as u64,
                sum(local.max_output_bytes, MAX_RECORD_BYTES as u64)?,
            ),
        };
        if sum(spent.attempts, local.max_attempts)? > self.max_attempts
            || sum(spent.successful_updates, local.max_successful_updates)?
                > self.max_successful_updates
            || local.max_families > self.max_families
            || work.callbacks > self.max_callbacks
            || work.rows > self.max_candidate_rows
            || work.search > self.max_search_steps
            || work.bytes > self.max_declared_work_bytes
            || sum(spent.logical_record_bytes, local.max_output_bytes)?
                > self.max_logical_record_bytes
            || sum(spent.stored_record_bytes, stored)? > self.max_stored_record_bytes
            || sum(spent.metadata_bytes, local.max_metadata_bytes)? > self.max_metadata_bytes
            || temporary > self.max_temporary_bytes
            || sum(sum(spent.stored_record_bytes, spent.metadata_bytes)?, disk)?
                > self.max_disk_bytes
        {
            return Err("Current mixed successor admission exceeds the global envelope".into());
        }
        Ok(())
    }
}
fn sum(a: u64, b: u64) -> Result<u64, String> {
    a.checked_add(b)
        .ok_or_else(|| "Current mixed global ledger overflow".into())
}

#[derive(Clone, Copy, Debug, Default, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct Spent {
    attempts: u64,
    successful_updates: u64,
    reserved_work: Work,
    logical_record_bytes: u64,
    stored_record_bytes: u64,
    metadata_bytes: u64,
}
impl Spent {
    fn with_session(self, s: &CurrentMixedTrainingSession<'_>) -> Result<Self, String> {
        self.with_progress(
            &s.progress,
            s.raw_record_bytes,
            s.stored_record_bytes,
            s.metadata_bytes,
        )
    }
    fn with_progress(
        self,
        p: &Progress,
        raw: u64,
        stored: u64,
        metadata: u64,
    ) -> Result<Self, String> {
        Ok(Self {
            attempts: sum(self.attempts, p.attempts)?,
            successful_updates: sum(self.successful_updates, p.successful)?,
            reserved_work: self.reserved_work.add(p.work)?,
            logical_record_bytes: sum(self.logical_record_bytes, raw)?,
            stored_record_bytes: sum(self.stored_record_bytes, stored)?,
            metadata_bytes: sum(self.metadata_bytes, metadata)?,
        })
    }
}

/// Bound only by a genuine final stopped session; this metadata cannot admit an owner.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct PoolForkBinding {
    schema: String,
    predecessor_pool: String,
    predecessor_outcome: String,
    failed_attempt: u64,
    // Fixed snapshot of [cohorts, missing counts, callbacks, rows, singletons]
    // for Updated, NoChange, Failed and Abandoned, including known missingness.
    predecessor_own_counters: [[u64; 5]; 4],
    predecessor_pending_own: Option<[u64; 3]>,
}
fn own_snapshot(c: CurrentMixedOwnDecisionCounters) -> [[u64; 5]; 4] {
    [c.updated, c.no_change, c.failed, c.abandoned].map(|v| {
        [
            v.cohorts,
            v.cohorts_without_counts,
            v.own.callbacks,
            v.own.candidate_rows,
            v.own.singletons,
        ]
    })
}
fn require_guarded_pool_replacement(
    old: &CurrentMixedCollectionSpec,
    new: &CurrentMixedCollectionSpec,
    old_checksum: &str,
) -> Result<String, String> {
    if old.pool_checksum()? != old_checksum
        || bytes(&old.inner.wire.weights, SPEC_CAP)? != bytes(&new.inner.wire.weights, SPEC_CAP)?
        || bytes(&old.inner.wire.search, SPEC_CAP)? != bytes(&new.inner.wire.search, SPEC_CAP)?
    {
        return Err("Pool fork cannot replace weights, Search or the actual old pool".into());
    }
    let mut changed = 0usize;
    for (before, after) in old.inner.wire.rosters.iter().zip(&new.inner.wire.rosters) {
        if before.len() != after.len() {
            return Err("Pool fork absolute roster lengths changed".into());
        }
        for (before, after) in before.iter().zip(after) {
            match (before, after) {
                (
                    Some(OpponentSpec::PreparedBc),
                    Some(OpponentSpec::PreparedBcTradeGuard { guard }),
                ) => {
                    guard.validate()?;
                    changed += 1;
                }
                (Some(OpponentSpec::PreparedBc), _) => {
                    return Err("Pool fork must guard every declared PreparedBC seat".into());
                }
                _ if bytes(before, SPEC_CAP)? == bytes(after, SPEC_CAP)? => (),
                _ => return Err("Pool fork permits only PreparedBC to guarded PreparedBC".into()),
            }
        }
    }
    let checksum = new.pool_checksum()?;
    if changed == 0 || checksum == old_checksum {
        return Err("Pool fork requires an explicit different guarded pool".into());
    }
    Ok(checksum)
}

/// Owns the complete known-failed tip and its last genuinely resolved joint owner.
/// It cannot be built from an artifact, metadata, an older checkpoint or a Ready token.
/// The predecessor remains stopped in its immutable saved archive.
pub struct StoppedCurrentMixedForkSource<'a> {
    session: CurrentMixedTrainingSession<'a>,
}
impl StoppedCurrentMixedForkSource<'_> {
    /// Read-only checks leave this token owned by the caller on predictable errors.
    pub fn preflight_pool_fork(
        &self,
        spec: &CurrentMixedSegmentSpec,
        old_pool: &CurrentMixedCollectionSpec,
        new_pool: &CurrentMixedCollectionSpec,
        output: &Path,
    ) -> Result<(), String> {
        let directory = local(output, true)?;
        if directory.exists() {
            return Err("Current mixed pool fork output must be new".into());
        }
        self.session
            .pool_fork_binding(spec, old_pool, new_pool, &directory)?;
        Ok(())
    }
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub(super) struct SegmentBinding {
    chain_root: String,
    index: usize,
    predecessor_directory: PathBuf,
    predecessor_schema: String,
    predecessor_root: String,
    predecessor_spec: String,
    predecessor_head: String,
    predecessor_sequence: u64,
    pool: String,
    pub(super) parent: CurrentMixedParentIdentity,
    pub(super) prefix: [u64; 2],
    pub(super) next: u64,
    pub(super) consumed: Vec<String>,
    inherited: Spent,
    contract: GlobalContract,
    contract_checksum: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pool_fork: Option<PoolForkBinding>,
}
impl SegmentBinding {
    fn checksum(&self) -> Result<String, String> {
        hash(b"tzolkin-current-mixed-segment-root-v1\0", self)
    }
}

pub(super) struct SegmentContext {
    binding: SegmentBinding,
    archives: Vec<ClosedArchive>,
}
#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub(super) struct SegmentUsage {
    index: usize,
    contract_checksum: String,
    total_attempts: u64,
    total_successful_updates: u64,
    reserved_work: Work,
    logical_record_bytes: u64,
    stored_record_bytes: u64,
    metadata_bytes: u64,
    retained_bytes: u64,
}
struct ClosedArchive {
    directory: PathBuf,
    spec: CurrentMixedRunSpec,
    journal_schema: &'static str,
    root_checksum: String,
    pool: String,
    sequence: u64,
    head: Option<FilePin>,
    spec_pin: Option<FilePin>,
    references: BTreeMap<u64, Vec<FilePin>>,
    output_bytes: u64,
    metadata_bytes: u64,
    raw_record_bytes: u64,
    stored_record_bytes: u64,
}
impl ClosedArchive {
    fn view(&self) -> ArchiveView<'_> {
        ArchiveView {
            directory: &self.directory,
            spec: &self.spec,
            journal_schema: self.journal_schema,
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
}
impl SegmentContext {
    pub(super) fn require_archives(&self) -> Result<(), String> {
        for archive in &self.archives {
            restore::require_archive_files(&archive.view())?;
        }
        Ok(())
    }
}

impl ReadyCurrentMixedContinuation<'_> {
    /// Read-only checks before ownership is consumed. No directory is created,
    /// no work is reserved, and the token remains usable if a check fails.
    pub fn preflight_segment(
        &self,
        spec: &CurrentMixedSegmentSpec,
        output: &Path,
    ) -> Result<(), String> {
        let directory = local(output, true)?;
        if directory.exists() {
            return Err("Current mixed successor output must be new".into());
        }
        self.session.segment_binding(spec, &directory)?;
        Ok(())
    }
}
impl<'a> CurrentMixedTrainingSession<'a> {
    pub(super) fn segment_usage(&self) -> Option<SegmentUsage> {
        self.segment.as_ref().map(|c| {
            let spent = c
                .binding
                .inherited
                .with_session(self)
                .expect("validated private global envelope");
            SegmentUsage {
                index: c.binding.index,
                contract_checksum: c.binding.contract_checksum.clone(),
                total_attempts: spent.attempts,
                total_successful_updates: spent.successful_updates,
                reserved_work: spent.reserved_work,
                logical_record_bytes: spent.logical_record_bytes,
                stored_record_bytes: spent.stored_record_bytes,
                metadata_bytes: spent.metadata_bytes,
                retained_bytes: sum(spent.stored_record_bytes, spent.metadata_bytes)
                    .expect("validated private retained envelope"),
            }
        })
    }
    fn segment_binding(
        &self,
        spec: &CurrentMixedSegmentSpec,
        directory: &Path,
    ) -> Result<SegmentBinding, String> {
        self.progress
            .require_ready_continuation(self.blocked, self.head.is_some())?;
        self.current_policy()?;
        self.successor_binding(spec, directory, self.pool.clone(), None)
    }
    fn require_stopped_fork_source(&self) -> Result<(), String> {
        self.progress
            .require_stopped_pool_fork(self.blocked, self.head.is_some())?;
        if self.continued.is_none() {
            return Err("Pool fork requires a genuinely resolved continued joint owner".into());
        }
        self.handle()?;
        Ok(())
    }
    /// Consumes only the final known Failed tip, never reviving its old training authority.
    pub fn into_stopped_pool_fork(self) -> Result<StoppedCurrentMixedForkSource<'a>, String> {
        self.require_stopped_fork_source()?;
        self.require_full_archives()?;
        self.require_stopped_fork_source()?;
        Ok(StoppedCurrentMixedForkSource { session: self })
    }
    fn pool_fork_binding(
        &self,
        spec: &CurrentMixedSegmentSpec,
        old_pool: &CurrentMixedCollectionSpec,
        new_pool: &CurrentMixedCollectionSpec,
        directory: &Path,
    ) -> Result<SegmentBinding, String> {
        self.require_stopped_fork_source()?;
        let pool = require_guarded_pool_replacement(old_pool, new_pool, &self.pool)?;
        let fork = PoolForkBinding {
            schema: "tzolkin-current-mixed-stopped-pool-fork-v1".into(),
            predecessor_pool: self.pool.clone(),
            predecessor_outcome: "failed".into(),
            failed_attempt: self
                .progress
                .active
                .as_ref()
                .expect("checked failure")
                .attempt,
            predecessor_own_counters: own_snapshot(self.progress.own_decisions),
            predecessor_pending_own: self
                .progress
                .active_own
                .map(|c| [c.callbacks, c.candidate_rows, c.singletons]),
        };
        self.successor_binding(spec, directory, pool, Some(fork))
    }
    fn successor_binding(
        &self,
        spec: &CurrentMixedSegmentSpec,
        directory: &Path,
        pool: String,
        pool_fork: Option<PoolForkBinding>,
    ) -> Result<SegmentBinding, String> {
        spec.validate()?;
        self.handle()?;
        self.require_full_archives()?;
        self.root.require_output(directory)?;
        spec.global_contract.require_output(directory)?;
        require_output_outside_directory(&self.directory, directory)?;
        let (chain_root, index, inherited) = match &self.segment {
            Some(c) => {
                if c.binding.contract != spec.global_contract {
                    return Err("Current mixed global contract is immutable".into());
                }
                for archive in &c.archives {
                    require_output_outside_directory(&archive.directory, directory)?;
                }
                (
                    c.binding.chain_root.clone(),
                    c.binding.index + 1,
                    c.binding.inherited,
                )
            }
            None => (self.root_checksum.clone(), 1, Spent::default()),
        };
        let inherited = inherited.with_session(self)?;
        if index >= spec.global_contract.max_segments
            || spec.run_spec.actor != self.spec.actor
            || spec.run_spec.critic != self.spec.critic
            || self
                .progress
                .consumed
                .len()
                .checked_add(4)
                .is_none_or(|n| n > spec.run_spec.max_families)
            || sum(
                self.progress.parent_count()?,
                spec.run_spec.max_successful_updates,
            )? > source::MAX_MIXED_UPDATES
        {
            return Err("Current mixed successor count/configuration/family envelope".into());
        }
        spec.global_contract.envelope(inherited, &spec.run_spec)?;
        let binding = SegmentBinding {
            chain_root,
            index,
            predecessor_directory: self.directory.clone(),
            predecessor_schema: self.journal_schema().into(),
            predecessor_root: self.root_checksum.clone(),
            predecessor_spec: self.spec.checksum()?,
            predecessor_head: self
                .head
                .as_ref()
                .ok_or("Missing predecessor head")?
                .sha256
                .clone(),
            predecessor_sequence: self.sequence,
            pool: pool.clone(),
            parent: self.handle()?.identity().clone(),
            prefix: self.progress.prefix,
            next: self.progress.next,
            consumed: self.progress.consumed.iter().cloned().collect(),
            inherited,
            contract: spec.global_contract.clone(),
            contract_checksum: spec.global_contract.checksum()?,
            pool_fork,
        };
        let raw = bytes(&spec.run_spec, SPEC_CAP)?;
        let event = Event::SegmentStart {
            run_spec: file_pin("run-spec.json", &raw),
            binding: Box::new(binding.clone()),
        };
        bytes(
            &Entry {
                schema: SEGMENT_JOURNAL_SCHEMA.into(),
                sequence: 1,
                previous: String::new(),
                root: binding.checksum()?,
                spec: spec.run_spec.checksum()?,
                pool,
                event,
            },
            HEAD_CAP,
        )?;
        Ok(binding)
    }
    fn initial_segment(
        predecessor: ReadyCurrentMixedContinuation<'a>,
        spec: CurrentMixedSegmentSpec,
        directory: PathBuf,
    ) -> Result<Self, String> {
        let binding = predecessor.session.segment_binding(&spec, &directory)?;
        Self::from_predecessor(predecessor.session, spec, binding, directory)
    }
    fn from_predecessor(
        s: CurrentMixedTrainingSession<'a>,
        spec: CurrentMixedSegmentSpec,
        binding: SegmentBinding,
        directory: PathBuf,
    ) -> Result<Self, String> {
        let root_checksum = binding.checksum()?;
        let mut progress = Progress::initial(
            &binding.parent,
            binding.prefix,
            binding.next,
            &binding.consumed.iter().cloned().collect(),
        );
        progress.base = binding.parent.mixed_update_count();
        progress.segment_start = Some(binding.clone());
        let journal_schema = s.journal_schema();
        let mut archives = s.segment.map(|c| c.archives).unwrap_or_default();
        archives.push(ClosedArchive {
            directory: s.directory,
            spec: s.spec,
            journal_schema,
            root_checksum: s.root_checksum,
            pool: s.pool.clone(),
            sequence: s.sequence,
            head: s.head,
            spec_pin: s.spec_pin,
            references: s.references,
            output_bytes: s.output_bytes,
            metadata_bytes: s.metadata_bytes,
            raw_record_bytes: s.raw_record_bytes,
            stored_record_bytes: s.stored_record_bytes,
        });
        Ok(Self {
            root: s.root,
            original: s.original,
            prepared: s.prepared,
            search: s.search,
            pool: binding.pool.clone(),
            spec: spec.run_spec,
            root_checksum,
            progress,
            continued: s.continued,
            directory,
            sequence: 0,
            head: None,
            spec_pin: None,
            owner_pins: s.owner_pins,
            references: BTreeMap::new(),
            output_bytes: 0,
            metadata_bytes: 0,
            raw_record_bytes: 0,
            stored_record_bytes: 0,
            blocked: false,
            segment: Some(SegmentContext { binding, archives }),
        })
    }
    /// Start one bounded successor. Call the token's read-only preflight first
    /// to retain ownership on predictable validation failures. I/O failures can
    /// still consume the token; restore uses a freshly genuine predecessor.
    pub fn new_segment(
        predecessor: ReadyCurrentMixedContinuation<'a>,
        spec: CurrentMixedSegmentSpec,
        output: &Path,
    ) -> Result<Self, String> {
        let directory = local(output, true)?;
        if directory.exists() {
            return Err("Current mixed successor output must be new".into());
        }
        Self::publish_successor(Self::initial_segment(predecessor, spec, directory)?)
    }
    fn initial_pool_fork(
        predecessor: StoppedCurrentMixedForkSource<'a>,
        spec: CurrentMixedSegmentSpec,
        old_pool: &CurrentMixedCollectionSpec,
        new_pool: &CurrentMixedCollectionSpec,
        directory: PathBuf,
    ) -> Result<Self, String> {
        let binding = predecessor
            .session
            .pool_fork_binding(&spec, old_pool, new_pool, &directory)?;
        Self::from_predecessor(predecessor.session, spec, binding, directory)
    }
    /// Starts a distinct guarded-pool successor from the full final failed archive.
    /// Preflight first; predictable errors preserve the token, I/O errors can consume it.
    pub fn new_pool_fork(
        predecessor: StoppedCurrentMixedForkSource<'a>,
        spec: CurrentMixedSegmentSpec,
        old_pool: &CurrentMixedCollectionSpec,
        new_pool: &CurrentMixedCollectionSpec,
        output: &Path,
    ) -> Result<Self, String> {
        let directory = local(output, true)?;
        if directory.exists() {
            return Err("Current mixed pool fork output must be new".into());
        }
        Self::publish_successor(Self::initial_pool_fork(
            predecessor,
            spec,
            old_pool,
            new_pool,
            directory,
        )?)
    }
    /// Requires a freshly genuine complete failed predecessor, not a saved artifact DTO.
    pub fn restore_pool_fork(
        predecessor: StoppedCurrentMixedForkSource<'a>,
        spec: CurrentMixedSegmentSpec,
        old_pool: &CurrentMixedCollectionSpec,
        new_pool: &CurrentMixedCollectionSpec,
        head: (&[u8], &str),
        directory: &Path,
    ) -> Result<Self, String> {
        Self::check_latest_head(head.0, head.1)?;
        Self::initial_pool_fork(
            predecessor,
            spec,
            old_pool,
            new_pool,
            local(directory, false)?,
        )?
        .restore_entries(head.0)
    }
    fn publish_successor(mut s: Self) -> Result<Self, String> {
        let raw = bytes(&s.spec, SPEC_CAP)?;
        let initial = Event::SegmentStart {
            run_spec: file_pin("run-spec.json", &raw),
            binding: Box::new(
                s.segment
                    .as_ref()
                    .expect("private segment admission")
                    .binding
                    .clone(),
            ),
        };
        fs::create_dir(&s.directory).map_err(|e| e.to_string())?;
        s.spec_pin = Some(s.publish_root("run-spec.json", &raw, SPEC_CAP)?);
        s.save(initial)?;
        Ok(s)
    }
    /// Restore only against the same freshly reconstructed genuine predecessor.
    /// The saved SegmentStart is compared to owner-derived admission metadata;
    /// a saved head alone cannot admit an actor or critic.
    pub fn restore_segment(
        predecessor: ReadyCurrentMixedContinuation<'a>,
        spec: CurrentMixedSegmentSpec,
        head_raw: &[u8],
        expected_sha256: &str,
        directory: &Path,
    ) -> Result<Self, String> {
        Self::check_latest_head(head_raw, expected_sha256)?;
        let s = Self::initial_segment(predecessor, spec, local(directory, false)?)?;
        s.restore_entries(head_raw)
    }
}

#[cfg(test)]
#[path = "public_rl_current_mixed_segment_tests.rs"]
mod tests;
