//! Bounded successor admission consumes a genuine Ready joint owner. The ledger
//! covers one lineage; restoring an older predecessor can still create a fork.
use super::*;
use crate::public_rl_mixed_session::require_output_outside_directory;

pub const SEGMENT_SPEC_SCHEMA: &str = "tzolkin-current-mixed-segment-spec-v1";
pub const SEGMENT_JOURNAL_SCHEMA: &str = "tzolkin-current-mixed-segment-journal-v1";
pub const GLOBAL_CONTRACT_SCHEMA: &str = "tzolkin-current-mixed-global-contract-v1";
const BUDGET_AMENDMENT_SCHEMA: &str = "tzolkin-current-mixed-ready-budget-amendment-v1";
const SEAT_ROTATION_SCHEMA: &str = "tzolkin-current-mixed-ready-seat-rotation-v1";
pub(super) const MAX_SEGMENTS: usize = 64;
const MAX_PROTECTED_DIRECTORIES: usize = 16;

/// A finite local run and global limits. Ordinary successors keep those limits
/// immutable; an explicit Ready budget amendment may only increase them.
/// This is metadata, never a constructor for a policy owner. All amounts include
/// the predecessor's spent ledger of the first durable run and its successors. Borrowed original input
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
    fn require_monotonic_extension(&self, next: &Self) -> Result<(), String> {
        self.validate()?;
        next.validate()?;
        let old = [
            self.max_attempts,
            self.max_successful_updates,
            self.max_callbacks,
            self.max_candidate_rows,
            self.max_search_steps,
            self.max_declared_work_bytes,
            self.max_logical_record_bytes,
            self.max_stored_record_bytes,
            self.max_metadata_bytes,
            self.max_temporary_bytes,
            self.max_disk_bytes,
        ];
        let new = [
            next.max_attempts,
            next.max_successful_updates,
            next.max_callbacks,
            next.max_candidate_rows,
            next.max_search_steps,
            next.max_declared_work_bytes,
            next.max_logical_record_bytes,
            next.max_stored_record_bytes,
            next.max_metadata_bytes,
            next.max_temporary_bytes,
            next.max_disk_bytes,
        ];
        if self.schema != next.schema
            || !self
                .protected_directories
                .iter()
                .map(|p| p.as_os_str())
                .eq(next.protected_directories.iter().map(|p| p.as_os_str()))
            || self.max_segments > next.max_segments
            || self.max_families > next.max_families
            || old.into_iter().zip(new).any(|(a, b)| a > b)
            || self == next
        {
            return Err(
                "Current mixed budget amendment must strictly extend unchanged restrictions".into(),
            );
        }
        Ok(())
    }
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

fn require_rotation_shifts(shifts: [u8; 2]) -> Result<(), String> {
    if shifts[0] >= 3 || shifts[1] >= 4 || shifts == [0, 0] {
        return Err(
            "Current mixed cyclic shifts require 3p 0..2 and 4p 0..3, not both zero".into(),
        );
    }
    Ok(())
}
fn require_cyclic_seat_rotation(
    old: &CurrentMixedCollectionSpec,
    new: &CurrentMixedCollectionSpec,
    old_checksum: &str,
    shifts: [u8; 2],
) -> Result<String, String> {
    require_rotation_shifts(shifts)?;
    if old.pool_checksum()? != old_checksum
        || bytes(&old.inner.wire.weights, SPEC_CAP)? != bytes(&new.inner.wire.weights, SPEC_CAP)?
        || bytes(&old.inner.wire.search, SPEC_CAP)? != bytes(&new.inner.wire.search, SPEC_CAP)?
    {
        return Err("Seat rotation cannot replace weights, Search or the actual old pool".into());
    }
    for (before, after) in old.inner.wire.rosters.iter().zip(&new.inner.wire.rosters) {
        let shift = match before.len() {
            3 => shifts[0] as usize,
            4 => shifts[1] as usize,
            _ => return Err("Seat rotation requires 3p or 4p roster members".into()),
        };
        if before.len() != after.len() {
            return Err("Seat rotation cannot reorder player-count members".into());
        }
        for (seat, role) in before.iter().enumerate() {
            if bytes(role, SPEC_CAP)? != bytes(&after[(seat + shift) % before.len()], SPEC_CAP)? {
                return Err("Seat rotation must move every complete role cyclically".into());
            }
        }
    }
    let checksum = new.pool_checksum()?;
    if checksum == old_checksum {
        return Err("Seat rotation requires a different immediate predecessor pool".into());
    }
    Ok(checksum)
}
impl Progress {
    fn require_updated_seat_rotation(&self, blocked: bool, has_head: bool) -> Result<(), String> {
        self.require_ready_continuation(blocked, has_head)?;
        if self.outcome.as_deref() != Some("updated") {
            return Err("Seat rotation requires the latest resolved Updated outcome".into());
        }
        Ok(())
    }
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
    #[serde(default, skip_serializing_if = "Option::is_none")]
    budget_amendment: Option<BudgetAmendmentBinding>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    seat_rotation: Option<SeatRotationBinding>,
}
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct BudgetAmendmentBinding {
    schema: String,
    predecessor_contract_checksum: String,
    ordinal: u64,
}
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct SeatRotationBinding {
    schema: String,
    predecessor_pool: String,
    new_pool: String,
    shifts: [u8; 2],
    predecessor_outcome: String,
    // Cumulative Updated/NoChange/Failed/Abandoned rows, including missingness.
    // Campaign callers compare their declared non-updated baseline; historical
    // failures are not erased or required to be zero by this transition.
    predecessor_own_counters: [[u64; 5]; 4],
}
impl SegmentBinding {
    fn checksum(&self) -> Result<String, String> {
        if let Some(rotation) = &self.seat_rotation {
            require_rotation_shifts(rotation.shifts)?;
            if rotation.schema != SEAT_ROTATION_SCHEMA
                || rotation.predecessor_outcome != "updated"
                || rotation.new_pool != self.pool
                || rotation.new_pool == rotation.predecessor_pool
                || self.pool_fork.is_some()
                || self.budget_amendment.is_some()
            {
                return Err(
                    "Current mixed seat rotation binding must be exclusive and Updated".into(),
                );
            }
        }
        hash(b"tzolkin-current-mixed-segment-root-v1\0", self)
    }
    fn require_start_bytes(&self, spec: &CurrentMixedRunSpec) -> Result<(), String> {
        let raw = bytes(spec, SPEC_CAP)?;
        let event = Event::SegmentStart {
            run_spec: file_pin("run-spec.json", &raw),
            binding: Box::new(self.clone()),
        };
        bytes(
            &Entry {
                schema: SEGMENT_JOURNAL_SCHEMA.into(),
                sequence: 1,
                previous: String::new(),
                root: self.checksum()?,
                spec: spec.checksum()?,
                pool: self.pool.clone(),
                event,
            },
            HEAD_CAP,
        )?;
        Ok(())
    }
}

pub(super) struct SegmentContext {
    binding: SegmentBinding,
    archives: Vec<ClosedArchive>,
    pub(super) inherited_own: CurrentMixedOwnDecisionCounters,
    amendment_ordinal: u64,
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
    fn next_budget_amendment(
        &self,
        contract: &GlobalContract,
    ) -> Result<BudgetAmendmentBinding, String> {
        self.binding
            .contract
            .require_monotonic_extension(contract)?;
        Ok(BudgetAmendmentBinding {
            schema: BUDGET_AMENDMENT_SCHEMA.into(),
            predecessor_contract_checksum: self.binding.contract.checksum()?,
            ordinal: sum(self.amendment_ordinal, 1)?,
        })
    }
    pub(super) fn require_archives(&self) -> Result<(), String> {
        for archive in &self.archives {
            restore::require_archive_files(&archive.view())?;
        }
        Ok(())
    }
}

impl ReadyCurrentMixedContinuation<'_> {
    /// Preflight an explicit cyclic roster successor after the latest Updated.
    /// `shifts` is [3p, 4p]; every role moves with the learner. This checks no
    /// campaign-wide seat coverage or non-updated baseline. No output is written.
    pub fn preflight_seat_rotation(
        &self,
        spec: &CurrentMixedSegmentSpec,
        old_pool: &CurrentMixedCollectionSpec,
        new_pool: &CurrentMixedCollectionSpec,
        shifts: [u8; 2],
        output: &Path,
    ) -> Result<(), String> {
        let directory = local(output, true)?;
        if directory.exists() {
            return Err("Current mixed successor output must be new".into());
        }
        self.session.seat_rotation_binding(
            spec,
            old_pool,
            new_pool,
            shifts,
            &directory,
            restore::ArchiveValidation::Immediate,
        )?;
        Ok(())
    }
    /// Read-only preflight for an explicit monotonic contract amendment. The
    /// established contract, actual Ready owner and all spent work are required.
    /// Predictable refusal leaves this token owned by the caller.
    pub fn preflight_budget_amendment(
        &self,
        spec: &CurrentMixedSegmentSpec,
        output: &Path,
    ) -> Result<(), String> {
        let directory = local(output, true)?;
        if directory.exists() {
            return Err("Current mixed successor output must be new".into());
        }
        self.session.segment_binding_with_validation(
            spec,
            &directory,
            restore::ArchiveValidation::Immediate,
            true,
        )?;
        Ok(())
    }
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
        self.segment_binding_with_validation(
            spec,
            directory,
            restore::ArchiveValidation::Immediate,
            false,
        )
    }
    fn segment_binding_with_validation(
        &self,
        spec: &CurrentMixedSegmentSpec,
        directory: &Path,
        validation: restore::ArchiveValidation,
        amend_budget: bool,
    ) -> Result<SegmentBinding, String> {
        self.progress
            .require_ready_continuation(self.blocked, self.head.is_some())?;
        self.current_policy()?;
        self.successor_binding(
            spec,
            directory,
            self.pool.clone(),
            None,
            validation,
            amend_budget,
        )
    }
    fn seat_rotation_binding(
        &self,
        spec: &CurrentMixedSegmentSpec,
        old_pool: &CurrentMixedCollectionSpec,
        new_pool: &CurrentMixedCollectionSpec,
        shifts: [u8; 2],
        directory: &Path,
        validation: restore::ArchiveValidation,
    ) -> Result<SegmentBinding, String> {
        self.progress
            .require_updated_seat_rotation(self.blocked, self.head.is_some())?;
        if self.continued.is_none() {
            return Err("Seat rotation requires a genuinely resolved joint owner".into());
        }
        self.current_policy()?;
        let pool = require_cyclic_seat_rotation(old_pool, new_pool, &self.pool, shifts)?;
        let rotation = SeatRotationBinding {
            schema: SEAT_ROTATION_SCHEMA.into(),
            predecessor_pool: self.pool.clone(),
            new_pool: pool.clone(),
            shifts,
            predecessor_outcome: "updated".into(),
            predecessor_own_counters: own_snapshot(self.cumulative_own_decisions()?),
        };
        let mut binding = self.successor_binding(spec, directory, pool, None, validation, false)?;
        binding.seat_rotation = Some(rotation);
        binding.require_start_bytes(&spec.run_spec)?;
        Ok(binding)
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
        self.pool_fork_binding_with_validation(
            spec,
            old_pool,
            new_pool,
            directory,
            restore::ArchiveValidation::Immediate,
        )
    }
    fn pool_fork_binding_with_validation(
        &self,
        spec: &CurrentMixedSegmentSpec,
        old_pool: &CurrentMixedCollectionSpec,
        new_pool: &CurrentMixedCollectionSpec,
        directory: &Path,
        validation: restore::ArchiveValidation,
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
        };
        self.successor_binding(spec, directory, pool, Some(fork), validation, false)
    }
    fn successor_binding(
        &self,
        spec: &CurrentMixedSegmentSpec,
        directory: &Path,
        pool: String,
        pool_fork: Option<PoolForkBinding>,
        validation: restore::ArchiveValidation,
        amend_budget: bool,
    ) -> Result<SegmentBinding, String> {
        spec.validate()?;
        self.handle()?;
        validation.require(self)?;
        self.root.require_output(directory)?;
        spec.global_contract.require_output(directory)?;
        require_output_outside_directory(&self.directory, directory)?;
        let (chain_root, index, inherited, budget_amendment) = match &self.segment {
            Some(c) => {
                let amendment = if amend_budget {
                    Some(c.next_budget_amendment(&spec.global_contract)?)
                } else if c.binding.contract != spec.global_contract {
                    return Err("Current mixed global contract is immutable".into());
                } else {
                    None
                };
                for archive in &c.archives {
                    require_output_outside_directory(&archive.directory, directory)?;
                }
                (
                    c.binding.chain_root.clone(),
                    c.binding.index + 1,
                    c.binding.inherited,
                    amendment,
                )
            }
            None => {
                if amend_budget {
                    return Err(
                        "Current mixed budget amendment requires an established global contract"
                            .into(),
                    );
                }
                (self.root_checksum.clone(), 1, Spent::default(), None)
            }
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
            budget_amendment,
            seat_rotation: None,
        };
        binding.require_start_bytes(&spec.run_spec)?;
        Ok(binding)
    }
    fn initial_segment(
        predecessor: ReadyCurrentMixedContinuation<'a>,
        spec: CurrentMixedSegmentSpec,
        directory: PathBuf,
        amend_budget: bool,
    ) -> Result<Self, String> {
        let binding = predecessor.session.segment_binding_with_validation(
            &spec,
            &directory,
            restore::ArchiveValidation::Immediate,
            amend_budget,
        )?;
        Self::from_predecessor(predecessor.session, spec, binding, directory)
    }
    fn from_predecessor(
        s: CurrentMixedTrainingSession<'a>,
        spec: CurrentMixedSegmentSpec,
        binding: SegmentBinding,
        directory: PathBuf,
    ) -> Result<Self, String> {
        let root_checksum = binding.checksum()?;
        let inherited_own = s.cumulative_own_decisions()?;
        let amendment_ordinal = binding.budget_amendment.as_ref().map_or_else(
            || s.segment.as_ref().map_or(0, |c| c.amendment_ordinal),
            |amendment| amendment.ordinal,
        );
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
            active_record_bytes: ActiveRecordByteCache::default(),
            output_bytes: 0,
            metadata_bytes: 0,
            raw_record_bytes: 0,
            stored_record_bytes: 0,
            blocked: false,
            segment: Some(SegmentContext {
                binding,
                archives,
                inherited_own,
                amendment_ordinal,
            }),
        })
    }
    pub(super) fn restore_chain_segment(
        self,
        spec: CurrentMixedSegmentSpec,
        head_raw: &[u8],
        directory: &Path,
    ) -> Result<Self, String> {
        let directory = local(directory, false)?;
        let binding = self.segment_binding_with_validation(
            &spec,
            &directory,
            restore::ArchiveValidation::AtChainClose,
            false,
        )?;
        Self::from_predecessor(self, spec, binding, directory)?.restore_entries_in_chain(head_raw)
    }
    pub(super) fn restore_chain_seat_rotation(
        self,
        spec: CurrentMixedSegmentSpec,
        old_pool: &CurrentMixedCollectionSpec,
        new_pool: &CurrentMixedCollectionSpec,
        shifts: [u8; 2],
        tip: restore::CurrentMixedRestoreTip<'_>,
    ) -> Result<Self, String> {
        let directory = local(tip.directory, false)?;
        let binding = self.seat_rotation_binding(
            &spec,
            old_pool,
            new_pool,
            shifts,
            &directory,
            restore::ArchiveValidation::AtChainClose,
        )?;
        Self::from_predecessor(self, spec, binding, directory)?
            .restore_entries_in_chain(tip.head_raw)
    }
    pub(super) fn restore_chain_budget_amendment(
        self,
        spec: CurrentMixedSegmentSpec,
        head_raw: &[u8],
        directory: &Path,
    ) -> Result<Self, String> {
        let directory = local(directory, false)?;
        let binding = self.segment_binding_with_validation(
            &spec,
            &directory,
            restore::ArchiveValidation::AtChainClose,
            true,
        )?;
        Self::from_predecessor(self, spec, binding, directory)?.restore_entries_in_chain(head_raw)
    }
    pub(super) fn restore_chain_pool_fork(
        self,
        spec: CurrentMixedSegmentSpec,
        old_pool: &CurrentMixedCollectionSpec,
        new_pool: &CurrentMixedCollectionSpec,
        head_raw: &[u8],
        directory: &Path,
    ) -> Result<Self, String> {
        let directory = local(directory, false)?;
        let binding = self.pool_fork_binding_with_validation(
            &spec,
            old_pool,
            new_pool,
            &directory,
            restore::ArchiveValidation::AtChainClose,
        )?;
        Self::from_predecessor(self, spec, binding, directory)?.restore_entries_in_chain(head_raw)
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
        Self::publish_successor(Self::initial_segment(predecessor, spec, directory, false)?)
    }
    /// Consumes only a genuine latest-Updated Ready owner. Every complete role
    /// rotates by `shifts` [3p, 4p]; the actor, critic and stream are retained.
    /// The caller must declare seat coverage and stop on new non-updated work.
    /// Preflight first; I/O errors may consume the token.
    pub fn new_seat_rotation_segment(
        predecessor: ReadyCurrentMixedContinuation<'a>,
        spec: CurrentMixedSegmentSpec,
        old_pool: &CurrentMixedCollectionSpec,
        new_pool: &CurrentMixedCollectionSpec,
        shifts: [u8; 2],
        output: &Path,
    ) -> Result<Self, String> {
        let directory = local(output, true)?;
        if directory.exists() {
            return Err("Current mixed successor output must be new".into());
        }
        let binding = predecessor.session.seat_rotation_binding(
            &spec,
            old_pool,
            new_pool,
            shifts,
            &directory,
            restore::ArchiveValidation::Immediate,
        )?;
        Self::publish_successor(Self::from_predecessor(
            predecessor.session,
            spec,
            binding,
            directory,
        )?)
    }
    /// Reconstructs rotation admission from the genuine predecessor and exact
    /// externally pinned latest child tip. Saved rotation metadata is not an owner.
    pub fn restore_seat_rotation_segment(
        predecessor: ReadyCurrentMixedContinuation<'a>,
        spec: CurrentMixedSegmentSpec,
        old_pool: &CurrentMixedCollectionSpec,
        new_pool: &CurrentMixedCollectionSpec,
        shifts: [u8; 2],
        tip: restore::CurrentMixedRestoreTip<'_>,
    ) -> Result<Self, String> {
        Self::check_latest_head(tip.head_raw, tip.expected_sha256)?;
        let directory = local(tip.directory, false)?;
        let binding = predecessor.session.seat_rotation_binding(
            &spec,
            old_pool,
            new_pool,
            shifts,
            &directory,
            restore::ArchiveValidation::Immediate,
        )?;
        Self::from_predecessor(predecessor.session, spec, binding, directory)?
            .restore_entries(tip.head_raw)
    }
    /// Consumes an actual Ready continuation to start a finite successor under
    /// strictly extended global limits. Pool/configuration, prior spend, families
    /// and owner lineage remain unchanged. Preflight first; I/O can consume the token.
    pub fn new_segment_with_budget_amendment(
        predecessor: ReadyCurrentMixedContinuation<'a>,
        spec: CurrentMixedSegmentSpec,
        output: &Path,
    ) -> Result<Self, String> {
        let directory = local(output, true)?;
        if directory.exists() {
            return Err("Current mixed successor output must be new".into());
        }
        Self::publish_successor(Self::initial_segment(predecessor, spec, directory, true)?)
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
        let s = Self::initial_segment(predecessor, spec, local(directory, false)?, false)?;
        s.restore_entries(head_raw)
    }
    /// Regenerates amendment metadata from the same freshly genuine Ready
    /// predecessor. Saved contract/checksum/ordinal DTOs cannot admit an owner.
    pub fn restore_segment_with_budget_amendment(
        predecessor: ReadyCurrentMixedContinuation<'a>,
        spec: CurrentMixedSegmentSpec,
        head_raw: &[u8],
        expected_sha256: &str,
        directory: &Path,
    ) -> Result<Self, String> {
        Self::check_latest_head(head_raw, expected_sha256)?;
        Self::initial_segment(predecessor, spec, local(directory, false)?, true)?
            .restore_entries(head_raw)
    }
}

#[cfg(test)]
#[path = "public_rl_current_mixed_segment_tests.rs"]
mod tests;
