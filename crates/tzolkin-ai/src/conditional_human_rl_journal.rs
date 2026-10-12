//! Finite conditional joint journal. Local source custody is part of identity;
//! stored records grant neither portable/native qualification nor process-close proof.
use crate::conditional_human_rl::ConditionalRlOwner;
use crate::conditional_human_rl_update::{
    ConditionalFrozenPool, ConditionalMixedIdentity, ConditionalMixedOwner,
    ConditionalMixedSession, ConditionalNativeExclusions, ConditionalReservation, JointProposal,
    MAX_ATTEMPTS, MAX_SPENT_FAMILIES,
};
use crate::human_initial_wealth::ContentPin;
use crate::human_policy_training::PreparedConditionalHumanPolicy;
use crate::public_native::PreparedPublicPolicy;
use crate::public_policy_baseline::ResidualFitConfig;
use crate::public_policy_update::{OneStepConfig, checkpoint_digest, hash};
use crate::public_rl_paired_session::{
    local, publish_mixed_metadata, publish_mixed_stored_record, read_bounded,
    read_mixed_stored_record,
};
use crate::public_rl_record_storage as storage;
use crate::public_stochastic::SamplingSeed;
use crate::public_stochastic_native::CollectionLimits;
use crate::public_stochastic_record::{MAX_RECORD_BYTES, unique_json};
use serde::{Deserialize, Serialize};
use std::{
    collections::BTreeSet,
    fs,
    path::{Path, PathBuf},
};

pub const JOURNAL_SCHEMA: &str = "tzolkin-conditional-human-joint-journal-v1";
pub const SPEC_SCHEMA: &str = "tzolkin-conditional-human-joint-journal-spec-v1";
pub const CLOSE_AUTHORITY_SCHEMA: &str = "tzolkin-conditional-human-journal-close-authority-v1";
pub const PROCESS_CLOSE_SCHEMA: &str =
    "tzolkin-conditional-human-journal-observed-process-close-v1";
const HEAD_BYTES: usize = 512 * 1024;
const JOINT_BYTES: usize = 2 * 1024 * 1024;
const MAX_FILES: usize = 1024;

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct ConditionalJournalSpec {
    schema: String,
    max_attempts: u64,
    max_spent_families: usize,
    max_callbacks: usize,
    max_candidate_rows: usize,
    max_source_bytes: usize,
    max_trace_bytes: usize,
    max_logical_bytes: u64,
    max_stored_bytes: u64,
    max_metadata_bytes: u64,
    max_temporary_bytes: u64,
    max_disk_bytes: u64,
    actor: serde_json::Value,
    critic: serde_json::Value,
}
impl ConditionalJournalSpec {
    pub fn from_json(raw: &[u8]) -> Result<Self, String> {
        let spec: Self =
            serde_json::from_value(unique_json(raw, 64 * 1024)?).map_err(|e| e.to_string())?;
        spec.validate()?;
        Ok(spec)
    }
    fn limits(&self) -> Result<CollectionLimits, String> {
        CollectionLimits::new(
            self.max_callbacks,
            self.max_candidate_rows,
            self.max_source_bytes,
            self.max_trace_bytes,
        )
    }
    fn validate(&self) -> Result<(), String> {
        self.limits()?;
        self.actor()?;
        self.critic()?;
        let four_stored = 4 * storage::MAX_STORED_RECORD_BYTES as u64;
        if self.schema != SPEC_SCHEMA
            || !(1..=MAX_ATTEMPTS).contains(&self.max_attempts)
            || !(4..=MAX_SPENT_FAMILIES).contains(&self.max_spent_families)
            || !(4 * MAX_RECORD_BYTES as u64..=64 * 1024 * 1024 * 1024)
                .contains(&self.max_logical_bytes)
            || !(four_stored..=48 * 1024 * 1024 * 1024).contains(&self.max_stored_bytes)
            || !(8 * JOINT_BYTES as u64..=1024 * 1024 * 1024).contains(&self.max_metadata_bytes)
            || !(storage::MAX_STORED_RECORD_BYTES as u64..=512 * 1024 * 1024)
                .contains(&self.max_temporary_bytes)
            || self.max_disk_bytes > 56 * 1024 * 1024 * 1024
            || self.max_disk_bytes
                < four_stored + self.max_metadata_bytes + self.max_temporary_bytes
        {
            return Err("Invalid finite conditional journal specification".into());
        }
        Ok(())
    }
    fn actor(&self) -> Result<OneStepConfig, String> {
        OneStepConfig::from_checkpoint(self.actor.clone())
    }
    fn critic(&self) -> Result<ResidualFitConfig, String> {
        ResidualFitConfig::from_checkpoint(self.critic.clone())
    }
}

/// Paths and declared external custody are not an origin constructor. Every
/// lifecycle load goes through the production BC loader and its four losses.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct ConditionalJournalInput {
    checkpoint_path: String,
    collection_path: String,
    critic_seed: u64,
    sampling_seed: u64,
    external_pin: ContentPin,
    external_count: usize,
    external_digest: String,
}
impl ConditionalJournalInput {
    pub fn new(
        checkpoint: &Path,
        collection: &Path,
        critic_seed: u64,
        sampling_seed: u64,
        external_pin: ContentPin,
        external_count: usize,
        external_digest: String,
    ) -> Result<Self, String> {
        let checkpoint_path = local(checkpoint, false)?.to_string_lossy().into_owned();
        let collection_path = local(collection, false)?.to_string_lossy().into_owned();
        let input = Self {
            checkpoint_path,
            collection_path,
            critic_seed,
            sampling_seed,
            external_pin,
            external_count,
            external_digest,
        };
        // Validate syntax/custody here; only load() can emit the owner.
        local(Path::new(&input.external_pin.path), false)?;
        if !checkpoint_digest(&input.external_digest) {
            return Err("External digest shape".into());
        }
        Ok(input)
    }
    fn load(&self) -> Result<(ConditionalMixedSession, Vec<ContentPin>), String> {
        let prepared = PreparedConditionalHumanPolicy::load(
            Path::new(&self.checkpoint_path),
            Path::new(&self.collection_path),
        )?;
        let boundaries = prepared.loaded_boundary_sources()?.to_vec();
        let owner = ConditionalMixedOwner::from_count_zero(ConditionalRlOwner::from_prepared(
            &prepared,
            self.critic_seed,
            SamplingSeed::new(self.sampling_seed),
        )?)?;
        let external = ConditionalNativeExclusions::load(
            Path::new(&self.external_pin.path),
            &self.external_pin,
            self.external_count,
            &self.external_digest,
        )?;
        Ok((ConditionalMixedSession::new(owner, external)?, boundaries))
    }
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct Pin {
    name: String,
    bytes: u64,
    sha256: String,
}
impl Pin {
    fn new(name: &str, raw: &[u8]) -> Self {
        Self {
            name: name.into(),
            bytes: raw.len() as u64,
            sha256: crate::model::digest(raw),
        }
    }
}
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct SourceSeal {
    input: ConditionalJournalInput,
    origin: serde_json::Value,
    root: ConditionalMixedIdentity,
    boundary_pins: Vec<ContentPin>,
    #[cfg(test)]
    #[serde(skip)]
    test_outcomes: Option<Vec<bool>>,
}
impl SourceSeal {
    fn capture(
        input: ConditionalJournalInput,
        session: &ConditionalMixedSession,
        boundary_pins: Vec<ContentPin>,
    ) -> Result<Self, String> {
        Ok(Self {
            input,
            origin: serde_json::to_value(session.owner().origin()).map_err(|e| e.to_string())?,
            root: session.owner().identity().clone(),
            boundary_pins,
            #[cfg(test)]
            test_outcomes: None,
        })
    }
    fn small_check(&self) -> Result<(), String> {
        for pin in &self.boundary_pins {
            crate::human_policy_dataset::recheck_absolute_pin(pin)?;
        }
        Ok(())
    }
    fn full_load(&self) -> Result<ConditionalMixedSession, String> {
        #[cfg(test)]
        if self.test_outcomes.is_some() {
            self.small_check()?;
            return tests::session(&self.input);
        }
        let (session, boundaries) = self.input.load()?;
        if self.root != *session.owner().identity()
            || boundaries != self.boundary_pins
            || self.origin
                != serde_json::to_value(session.owner().origin()).map_err(|e| e.to_string())?
        {
            return Err("Conditional loaded origin/root/raw closure differs".into());
        }
        Ok(session)
    }
}
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct StoredRecord {
    stored: Pin,
    raw_bytes: usize,
    raw_sha256: String,
}
#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub enum ConditionalResolution {
    Updated,
    NoChange,
    Failed,
    Unknown,
}
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(tag = "kind", rename_all = "camelCase", deny_unknown_fields)]
enum Event {
    Init {
        source: SourceSeal,
        spec: ConditionalJournalSpec,
        pool: serde_json::Value,
        opponent_binding: serde_json::Value,
        root_parameters: Pin,
    },
    Reserve {
        attempt: u64,
        first_ordinal: u64,
        seeds: [u32; 4],
        families: [String; 4],
        parent: ConditionalMixedIdentity,
    },
    Resolve {
        attempt: u64,
        resolution: ConditionalResolution,
        report: Option<Pin>,
        joint: Option<Pin>,
        records: Option<[StoredRecord; 4]>,
        tip: ConditionalMixedIdentity,
        credited_own: u64,
        error: Option<String>,
    },
    Closed {
        attempt: u64,
        resolve: Pin,
        tip_checksum: String,
    },
    CloseFailed {
        attempt: u64,
        resolve: Pin,
        error: String,
    },
    CloseUnknown {
        attempt: u64,
        resolve: Pin,
        error: String,
    },
}
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct Entry {
    schema: String,
    sequence: u64,
    previous: Option<Pin>,
    event: Event,
    checksum: String,
}
fn entry_bytes(sequence: u64, previous: Option<Pin>, event: Event) -> Result<Vec<u8>, String> {
    let mut entry = Entry {
        schema: JOURNAL_SCHEMA.into(),
        sequence,
        previous,
        event,
        checksum: String::new(),
    };
    entry.checksum = hash(b"tzolkin-conditional-journal-entry-v1\0", &entry)?;
    bounded(&entry, HEAD_BYTES)
}
fn decode_entry(raw: &[u8]) -> Result<Entry, String> {
    let mut entry: Entry =
        serde_json::from_value(unique_json(raw, HEAD_BYTES)?).map_err(|e| e.to_string())?;
    let checksum = std::mem::take(&mut entry.checksum);
    if entry.schema != JOURNAL_SCHEMA
        || entry.sequence > 3 * MAX_ATTEMPTS
        || hash(b"tzolkin-conditional-journal-entry-v1\0", &entry)? != checksum
    {
        return Err("Conditional journal entry hash/domain differs".into());
    }
    entry.checksum = checksum;
    Ok(entry)
}
fn bounded(value: &impl Serialize, max: usize) -> Result<Vec<u8>, String> {
    let bytes = serde_json::to_vec(value).map_err(|e| e.to_string())?;
    if bytes.is_empty() || bytes.len() > max {
        return Err("Conditional metadata byte bound".into());
    }
    Ok(bytes)
}
fn read_pin(directory: &Path, pin: &Pin, max: usize) -> Result<Vec<u8>, String> {
    if pin.name.contains(['/', '\\'])
        || pin.name.contains("..")
        || pin.bytes == 0
        || pin.bytes > max as u64
        || !checkpoint_digest(&pin.sha256)
    {
        return Err("Conditional owned pin shape/bound".into());
    }
    let path = local(&directory.join(&pin.name), false)?;
    let raw = if max > MAX_RECORD_BYTES {
        read_mixed_stored_record(&path, max)?
    } else {
        read_bounded(&path, max)?
    };
    if raw.len() as u64 != pin.bytes || crate::model::digest(&raw) != pin.sha256 {
        return Err("Conditional owned pin changed".into());
    }
    Ok(raw)
}
fn root_parameters(session: &ConditionalMixedSession) -> Result<Vec<u8>, String> {
    bounded(
        &serde_json::json!({"schema":"tzolkin-conditional-journal-root-parameters-v1",
        "identity":session.owner().identity(),
        "actorBits":session.owner().actor_parameters().iter().map(|x|x.to_bits()).collect::<Vec<_>>(),
        "criticBits":session.owner().critic_parameters().iter().map(|x|x.to_bits()).collect::<Vec<_>>() }),
        JOINT_BYTES,
    )
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub enum ConditionalJournalPhase {
    Ready,
    Reserved,
    ResolvedAwaitingClose,
    Stopped,
}
/// Accounting view only. It cannot emit a sampler/owner or prove an OS exit.
#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ConditionalSavedTip {
    phase: ConditionalJournalPhase,
    attempts: u64,
    next_episode_ordinal: u64,
    successful_updates: u64,
    credited_own: u64,
    spent_families: BTreeSet<String>,
    tip: ConditionalMixedIdentity,
    head: ContentPin,
    last_resolution: Option<ConditionalResolution>,
    actual_process_close_verified: bool,
    authority_producer_authenticated: bool,
    source_qualified: bool,
    verified_native_policy: bool,
}
impl ConditionalSavedTip {
    pub fn successful_updates(&self) -> u64 {
        self.successful_updates
    }
    pub fn credited_own_decisions(&self) -> u64 {
        self.credited_own
    }
    pub fn next_episode_ordinal(&self) -> u64 {
        self.next_episode_ordinal
    }
    pub fn attempts(&self) -> u64 {
        self.attempts
    }
    pub fn phase(&self) -> ConditionalJournalPhase {
        self.phase
    }
}

pub struct ConditionalHumanJournal {
    directory: PathBuf,
    source: SourceSeal,
    spec: ConditionalJournalSpec,
    pool: ConditionalFrozenPool,
    opponent_binding: serde_json::Value,
    session: ConditionalMixedSession,
    head: Pin,
    sequence: u64,
    phase: ConditionalJournalPhase,
    members: BTreeSet<String>,
    logical_bytes: u64,
    #[cfg(test)]
    fail_after_publish: Option<&'static str>,
    #[cfg(test)]
    change_source_after_resolve: Option<PathBuf>,
}
/// Emitted only after the library's close publication returned successfully.
/// A serialized token does not authenticate an OS process exit.
#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ConditionalReturnedClose {
    head: ContentPin,
    tip_checksum: String,
    attempt: u64,
    resolution: ConditionalResolution,
    library_close_returned: bool,
    actual_process_close_verified: bool,
}
struct AppendFailure {
    error: String,
    committed: bool,
}
impl From<String> for AppendFailure {
    fn from(error: String) -> Self {
        Self {
            error,
            committed: false,
        }
    }
}
impl From<&str> for AppendFailure {
    fn from(error: &str) -> Self {
        Self {
            error: error.into(),
            committed: false,
        }
    }
}
#[derive(Default)]
struct Usage {
    stored: u64,
    metadata: u64,
    temporary: u64,
}
fn owned_usage(
    directory: &Path,
    spec: &ConditionalJournalSpec,
    members: &BTreeSet<String>,
    extras: &BTreeSet<String>,
    final_membership: bool,
) -> Result<Usage, String> {
    let mut allowed = members.clone();
    allowed.extend(extras.iter().cloned());
    let mut seen = BTreeSet::new();
    let mut usage = Usage::default();
    let mut count = 0;
    for entry in fs::read_dir(local(directory, false)?).map_err(|e| e.to_string())? {
        count += 1;
        if count > MAX_FILES {
            return Err("Conditional owned file count".into());
        }
        let entry = entry.map_err(|e| e.to_string())?;
        let name = entry
            .file_name()
            .into_string()
            .map_err(|_| "Non-UTF8 journal leaf")?;
        let path = local(&entry.path(), false)?;
        let metadata = fs::symlink_metadata(path).map_err(|e| e.to_string())?;
        if !metadata.is_file() {
            return Err("Conditional owned output must be regular leaves".into());
        }
        if allowed.contains(&name) {
            seen.insert(name.clone());
            let target = if name.ends_with(".json.gz") {
                &mut usage.stored
            } else {
                &mut usage.metadata
            };
            *target = target
                .checked_add(metadata.len())
                .ok_or("Owned byte overflow")?;
        } else if !final_membership
            && name.ends_with(".tmp")
            && allowed.iter().any(|a| name.starts_with(&format!("{a}.")))
        {
            usage.temporary = usage
                .temporary
                .checked_add(metadata.len())
                .ok_or("Temporary byte overflow")?;
        } else {
            return Err("Unexpected conditional owned output".into());
        }
    }
    if usage.stored > spec.max_stored_bytes
        || usage.metadata > spec.max_metadata_bytes
        || usage.temporary > spec.max_temporary_bytes
        || usage.stored + usage.metadata + usage.temporary > spec.max_disk_bytes
        || (final_membership && seen != *members)
    {
        return Err("Conditional stored/metadata/temporary/disk/member bound".into());
    }
    Ok(usage)
}
impl ConditionalHumanJournal {
    pub fn start(
        directory: &Path,
        input: ConditionalJournalInput,
        spec: ConditionalJournalSpec,
        pool_raw: &[u8],
        opponent: Option<&PreparedPublicPolicy>,
    ) -> Result<Self, String> {
        spec.validate()?;
        let pool = ConditionalFrozenPool::from_json(pool_raw)?;
        let opponent_binding = pool.binding(opponent)?;
        let (session, boundaries) = input.load()?;
        let source = SourceSeal::capture(input, &session, boundaries)?;
        Self::start_loaded(
            directory,
            source,
            spec,
            pool,
            serde_json::from_slice(pool_raw).map_err(|e| e.to_string())?,
            opponent_binding,
            session,
        )
    }
    fn start_loaded(
        directory: &Path,
        source: SourceSeal,
        spec: ConditionalJournalSpec,
        pool: ConditionalFrozenPool,
        pool_value: serde_json::Value,
        opponent_binding: serde_json::Value,
        session: ConditionalMixedSession,
    ) -> Result<Self, String> {
        let directory = local(directory, true)?;
        if fs::symlink_metadata(&directory).is_ok() {
            return Err("Conditional journal requires fresh output".into());
        }
        for pin in session
            .owner()
            .origin()
            .source_pins()
            .iter()
            .chain(std::iter::once(&source.input.external_pin))
        {
            if local(Path::new(&pin.path), false)?.starts_with(&directory) {
                return Err("Journal output intersects original source custody".into());
            }
        }
        let root = root_parameters(&session)?;
        let root_pin = Pin::new("root-parameters.json", &root);
        let initial = entry_bytes(
            0,
            None,
            Event::Init {
                source: source.clone(),
                spec: spec.clone(),
                pool: pool_value,
                opponent_binding: opponent_binding.clone(),
                root_parameters: root_pin.clone(),
            },
        )?;
        fs::create_dir(&directory).map_err(|e| e.to_string())?;
        publish_mixed_metadata(&directory, &root_pin.name, &root)?;
        publish_mixed_metadata(&directory, "head-0000.json", &initial)?;
        let head = Pin::new("head-0000.json", &initial);
        Ok(Self {
            directory,
            source,
            spec,
            pool,
            opponent_binding,
            session,
            head,
            sequence: 0,
            phase: ConditionalJournalPhase::Ready,
            members: BTreeSet::from([root_pin.name, "head-0000.json".into()]),
            logical_bytes: 0,
            #[cfg(test)]
            fail_after_publish: None,
            #[cfg(test)]
            change_source_after_resolve: None,
        })
    }
    /// Standalone scoring grants no reservation or journal continuation proof.
    pub fn owner(&self) -> Result<&ConditionalMixedOwner, String> {
        if self.phase != ConditionalJournalPhase::Ready || self.session.stopped() {
            return Err("Stopped/unclosed conditional journal exposes no owner".into());
        }
        Ok(self.session.owner())
    }
    pub fn head(&self) -> ContentPin {
        absolute_pin(&self.directory, &self.head)
    }
    pub fn phase(&self) -> ConditionalJournalPhase {
        self.phase
    }
    fn usage(&self, extras: &BTreeSet<String>, final_membership: bool) -> Result<Usage, String> {
        owned_usage(
            &self.directory,
            &self.spec,
            &self.members,
            extras,
            final_membership,
        )
    }
    fn require_head(&self) -> Result<(), String> {
        read_pin(&self.directory, &self.head, HEAD_BYTES).map(|_| ())
    }
    fn preflight(&mut self) -> Result<(), String> {
        if self.phase != ConditionalJournalPhase::Ready
            || self.session.stopped()
            || self.session.attempts() >= self.spec.max_attempts
            || self.session.consumed_families().len() + 4 > self.spec.max_spent_families
        {
            return Err("Conditional journal is stopped/unclosed/exhausted".into());
        }
        if let Err(e) = self.require_head().and_then(|()| self.source.small_check()) {
            self.stop();
            return Err(e);
        }
        let usage = match self.usage(&BTreeSet::new(), true) {
            Ok(u) => u,
            Err(e) => {
                self.stop();
                return Err(e);
            }
        };
        let logical = 4 * MAX_RECORD_BYTES as u64;
        let stored = 4 * storage::MAX_STORED_RECORD_BYTES as u64;
        let metadata = 2 * JOINT_BYTES as u64 + 3 * HEAD_BYTES as u64;
        if self.logical_bytes + logical > self.spec.max_logical_bytes
            || usage.stored + stored > self.spec.max_stored_bytes
            || usage.metadata + metadata > self.spec.max_metadata_bytes
            || usage.stored + usage.metadata + stored + metadata + self.spec.max_temporary_bytes
                > self.spec.max_disk_bytes
        {
            return Err("Conditional complete attempt cannot fit before Reserve".into());
        }
        Ok(())
    }
    fn stop(&mut self) {
        self.phase = ConditionalJournalPhase::Stopped;
        self.session.stop();
    }
    fn append(&mut self, event: Event) -> Result<Pin, AppendFailure> {
        #[cfg(test)]
        let label = match &event {
            Event::Resolve { .. } => "resolve",
            Event::Closed { .. } => "closed",
            _ => "head",
        };
        let operation = (|| {
            self.require_head()?;
            let sequence = self
                .sequence
                .checked_add(1)
                .ok_or("Journal sequence overflow")?;
            if sequence > 3 * self.spec.max_attempts {
                return Err("Conditional journal entry cap".into());
            }
            let name = format!("head-{sequence:04}.json");
            let raw = entry_bytes(sequence, Some(self.head.clone()), event)?;
            let pin = Pin::new(&name, &raw);
            self.usage(&BTreeSet::from([name.clone()]), false)?;
            let result = publish_mixed_metadata(&self.directory, &name, &raw);
            #[cfg(test)]
            let result = if self.fail_after_publish == Some(label) {
                self.fail_after_publish = None;
                Err("Synthetic publisher error after destination".into())
            } else {
                result
            };
            match result {
                Ok(()) => {
                    self.members.insert(name);
                    self.head = pin.clone();
                    self.sequence = sequence;
                    Ok(pin)
                }
                Err(error) => {
                    // Err may follow hard-link/readback. Inspect this exact attempted
                    // destination only; never append another resolution or infer close.
                    let committed = read_pin(&self.directory, &pin, HEAD_BYTES).is_ok();
                    if committed {
                        self.members.insert(name);
                        self.head = pin.clone();
                        self.sequence = sequence;
                    }
                    Err(AppendFailure { error, committed })
                }
            }
        })();
        match operation {
            Ok(v) => Ok(v),
            Err(e) => {
                self.stop();
                Err(e)
            }
        }
    }
    fn publish_payload(&mut self, name: &str, raw: &[u8], record: bool) -> Result<Pin, String> {
        let max = if record {
            storage::MAX_STORED_RECORD_BYTES
        } else {
            JOINT_BYTES
        };
        if raw.is_empty() || raw.len() > max {
            return Err("Conditional payload bound".into());
        }
        self.usage(&BTreeSet::from([name.into()]), false)?;
        if record {
            publish_mixed_stored_record(&self.directory, name, raw)?
        } else {
            publish_mixed_metadata(&self.directory, name, raw)?
        }
        #[cfg(test)]
        if self.fail_after_publish == Some("payload") {
            self.fail_after_publish = None;
            return Err("Synthetic payload publisher error after destination".into());
        }
        let pin = Pin::new(name, raw);
        read_pin(&self.directory, &pin, max)?;
        self.members.insert(name.into());
        Ok(pin)
    }
    pub fn update_whole_four(
        &mut self,
        seeds: [u32; 4],
        opponent: Option<&PreparedPublicPolicy>,
    ) -> Result<ConditionalReturnedClose, String> {
        self.preflight()?;
        if self.pool.binding(opponent)? != self.opponent_binding {
            return Err("Frozen conditional opponents changed".into());
        }
        let plan = self.session.plan(
            seeds,
            &self.pool,
            opponent,
            self.spec.limits()?,
            self.spec.actor()?,
            self.spec.critic()?,
        )?;
        let event = Event::Reserve {
            attempt: self.session.attempts() + 1,
            first_ordinal: self.session.next_episode_ordinal(),
            seeds: plan.seeds(),
            families: plan.families(),
            parent: plan.parent().clone(),
        };
        match self.append(event) {
            Ok(_) => {
                if let Err(e) = self.session.reserve(&plan) {
                    self.stop();
                    return Err(e);
                }
            }
            Err(failure) => {
                if failure.committed {
                    self.session.reserve_after_publish_error(&plan)?;
                }
                self.stop();
                return Err(failure.error);
            }
        }
        self.phase = ConditionalJournalPhase::Reserved;
        match self.stage(&plan, opponent, None) {
            Ok(proposal) => self.resolve_proposal(proposal),
            Err(error) => self.resolve_failure(error),
        }
    }
    fn stage(
        &self,
        plan: &ConditionalReservation,
        opponent: Option<&PreparedPublicPolicy>,
        saved: Option<&[Vec<u8>; 4]>,
    ) -> Result<JointProposal, String> {
        #[cfg(test)]
        if let Some(outcomes) = &self.source.test_outcomes {
            let winning = *outcomes
                .get((self.session.attempts() - 1) as usize)
                .ok_or("Missing tiny fixture case")?;
            return crate::conditional_human_rl_update::test_short_stage(
                &self.session,
                plan,
                &self.pool,
                winning,
                saved,
            );
        }
        self.session.stage(plan, &self.pool, opponent, saved)
    }
    fn resolve_failure(&mut self, error: String) -> Result<ConditionalReturnedClose, String> {
        let error = bounded_error(&error);
        let event = Event::Resolve {
            attempt: self.session.attempts(),
            resolution: ConditionalResolution::Failed,
            report: None,
            joint: None,
            records: None,
            tip: self.session.owner().identity().clone(),
            credited_own: self.session.credited_own_decisions(),
            error: Some(error.clone()),
        };
        let resolve = match self.append(event) {
            Ok(pin) => pin,
            Err(e) => {
                self.stop();
                return Err(e.error);
            }
        };
        self.stop();
        let _ = self.append(Event::CloseFailed {
            attempt: self.session.attempts(),
            resolve,
            error: error.clone(),
        });
        self.stop();
        Err(error)
    }
    fn resolve_proposal(
        &mut self,
        proposal: JointProposal,
    ) -> Result<ConditionalReturnedClose, String> {
        let stage = (|| {
            let (report_raw, joint_raw) = proposal.payload()?;
            let attempt = self.session.attempts();
            let report =
                self.publish_payload(&format!("report-{attempt:04}.json"), &report_raw, false)?;
            let joint = joint_raw
                .as_ref()
                .map(|raw| self.publish_payload(&format!("joint-{attempt:04}.json"), raw, false))
                .transpose()?;
            let mut records = Vec::new();
            for (i, raw) in proposal.records().iter().enumerate() {
                let encoded = storage::encode(raw)?;
                let stored = self.publish_payload(
                    &format!("record-{attempt:04}-{i}.json.gz"),
                    &encoded,
                    true,
                )?;
                records.push(StoredRecord {
                    stored,
                    raw_bytes: raw.len(),
                    raw_sha256: crate::model::digest(raw),
                });
            }
            let records: [StoredRecord; 4] = records
                .try_into()
                .map_err(|_| "Conditional stored whole-four shape")?;
            let tip = self.session.proposed_identity(&proposal)?;
            let resolution = if proposal.is_updated() {
                ConditionalResolution::Updated
            } else {
                ConditionalResolution::NoChange
            };
            Ok::<_, String>((report, joint, records, tip, resolution))
        })();
        let (report, joint, records, tip, resolution) = match stage {
            Ok(v) => v,
            Err(e) => {
                self.stop();
                return Err(e);
            }
        };
        let event = Event::Resolve {
            attempt: self.session.attempts(),
            resolution,
            report: Some(report),
            joint,
            records: Some(records.clone()),
            tip: tip.clone(),
            credited_own: proposal.own_credit(),
            error: None,
        };
        let resolve = match self.append(event) {
            Ok(pin) => pin,
            Err(failure) => {
                if failure.committed {
                    self.session.admit(proposal)?;
                }
                self.stop();
                return Err(failure.error);
            }
        };
        // Verified durable resolution precedes joint live admission and every
        // trailing original-source check. Its credit is never reclassified.
        if let Err(e) = self.session.admit(proposal) {
            self.stop();
            return Err(e);
        }
        #[cfg(test)]
        if let Some(path) = self.change_source_after_resolve.take()
            && let Err(e) = fs::write(path, b"changed after verified Resolve")
        {
            self.stop();
            return Err(e.to_string());
        }
        self.logical_bytes += records.iter().map(|r| r.raw_bytes as u64).sum::<u64>();
        self.phase = ConditionalJournalPhase::ResolvedAwaitingClose;
        if let Err(error) = self
            .source
            .full_load()
            .and_then(|_| self.require_head())
            .and_then(|()| inspect_archive(&self.directory, &self.head()).map(|_| ()))
            .and_then(|()| self.usage(&BTreeSet::new(), true).map(|_| ()))
        {
            self.stop();
            let error = bounded_error(&error);
            let _ = self.append(Event::CloseFailed {
                attempt: self.session.attempts(),
                resolve,
                error: error.clone(),
            });
            self.stop();
            return Err(error);
        }
        let closed = self
            .append(Event::Closed {
                attempt: self.session.attempts(),
                resolve,
                tip_checksum: self.session.owner().identity().checksum().into(),
            })
            .map_err(|e| e.error)?;
        // No fallible operation follows successful library finalization.
        self.phase = ConditionalJournalPhase::Ready;
        Ok(ConditionalReturnedClose {
            head: absolute_pin(&self.directory, &closed),
            tip_checksum: tip.checksum().into(),
            attempt: self.session.attempts(),
            resolution,
            library_close_returned: true,
            actual_process_close_verified: false,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    mod temp_root {
        include!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/support/temp_root.rs"
        ));
    }
    struct Files {
        root: PathBuf,
    }
    impl Files {
        fn new(label: &str) -> Self {
            Self {
                root: temp_root::create(label).unwrap(),
            }
        }
        fn write(&self, name: &str, raw: &[u8]) -> ContentPin {
            let path = self.root.join(name);
            fs::write(&path, raw).unwrap();
            ContentPin {
                path: path.to_string_lossy().into_owned(),
                bytes: raw.len() as u64,
                sha256: crate::model::digest(raw),
            }
        }
    }
    impl Drop for Files {
        fn drop(&mut self) {
            // Only this fixture's regular leaves and direct owned child folders.
            let Ok(root) = fs::canonicalize(&self.root) else {
                return;
            };
            if let Ok(entries) = fs::read_dir(&self.root) {
                for entry in entries.take(MAX_FILES).flatten() {
                    let path = entry.path();
                    let Ok(resolved) = fs::canonicalize(&path) else {
                        continue;
                    };
                    if !resolved.starts_with(&root) {
                        continue;
                    }
                    if entry.file_type().is_ok_and(|t| t.is_file()) {
                        let _ = fs::remove_file(path);
                    } else if entry.file_type().is_ok_and(|t| t.is_dir()) {
                        if let Ok(leaves) = fs::read_dir(&path) {
                            for leaf in leaves.take(MAX_FILES).flatten() {
                                if leaf.file_type().is_ok_and(|t| t.is_file())
                                    && fs::canonicalize(leaf.path())
                                        .is_ok_and(|p| p.starts_with(&root))
                                {
                                    let _ = fs::remove_file(leaf.path());
                                }
                            }
                        }
                        let _ = fs::remove_dir(path);
                    }
                }
            }
            let _ = fs::remove_dir(&self.root);
        }
    }
    fn input(files: &Files) -> (ConditionalJournalInput, ContentPin) {
        let marker = files.write("source-marker.json", b"unchanged synthetic source");
        let family = crate::dataset::seed_family_id(42);
        let external = files.write(
            "external.json",
            &serde_json::to_vec(&json!({
            "schema":"tzolkin-conditional-native-exclusions-v1","families":[family]}))
            .unwrap(),
        );
        let input = ConditionalJournalInput::new(
            Path::new(&marker.path),
            Path::new(&marker.path),
            11,
            17,
            external,
            1,
            crate::model::digest(format!("{family}\n").as_bytes()),
        )
        .unwrap();
        (input, marker)
    }
    pub(super) fn session(
        input: &ConditionalJournalInput,
    ) -> Result<ConditionalMixedSession, String> {
        let external = ConditionalNativeExclusions::load(
            Path::new(&input.external_pin.path),
            &input.external_pin,
            input.external_count,
            &input.external_digest,
        )?;
        ConditionalMixedSession::new(crate::conditional_human_rl_update::tests::owner(), external)
    }
    fn spec(rate: f64) -> ConditionalJournalSpec {
        ConditionalJournalSpec::from_json(&serde_json::to_vec(&json!({
            "schema":SPEC_SCHEMA,"maxAttempts":4,"maxSpentFamilies":16,
            "maxCallbacks":64,"maxCandidateRows":262144,"maxSourceBytes":8388608,"maxTraceBytes":1048576,
            "maxLogicalBytes":1073741824u64,"maxStoredBytes":1073741824u64,"maxMetadataBytes":67108864,
            "maxTemporaryBytes":storage::MAX_STORED_RECORD_BYTES,"maxDiskBytes":2147483648u64,
            "actor":{"learningRate":rate,"maxAbsActualDelta":0.01,"maxCandidateRows":1048576},
            "critic":{"learningRate":rate,"maxAbsActualDelta":0.01,"maxCallbacks":256}
        })).unwrap()).unwrap()
    }
    fn pool() -> serde_json::Value {
        let opponent = json!({"kind":"heuristic"});
        json!({"schema":"tzolkin-conditional-human-mixed-frozen-pool-v1",
            "weights":crate::policy::HeuristicWeights::default(),
            "rosters":[[null,opponent,opponent],[null,opponent,opponent],
                [null,opponent,opponent,opponent],[null,opponent,opponent,opponent]]})
    }
    fn start(files: &Files, name: &str, rate: f64) -> ConditionalHumanJournal {
        let (input, marker) = input(files);
        let session = session(&input).unwrap();
        let pool_value = pool();
        let pool =
            ConditionalFrozenPool::from_json(&serde_json::to_vec(&pool_value).unwrap()).unwrap();
        let binding = pool.binding(None).unwrap();
        let mut source = SourceSeal::capture(input, &session, vec![marker]).unwrap();
        source.test_outcomes = Some(vec![true; 4]);
        ConditionalHumanJournal::start_loaded(
            &files.root.join(name),
            source,
            spec(rate),
            pool,
            pool_value,
            binding,
            session,
        )
        .unwrap()
    }
    fn seeds() -> [[u32; 4]; 2] {
        let values = (0..64)
            .filter(|s| {
                *s != 42
                    && crate::dataset::split_for_family(&crate::dataset::seed_family_id(*s))
                        == Ok(crate::dataset::DatasetSplit::Train)
            })
            .take(8)
            .collect::<Vec<_>>();
        [
            values[..4].try_into().unwrap(),
            values[4..].try_into().unwrap(),
        ]
    }
    fn authority(
        files: &Files,
        name: &str,
        returned: &ConditionalReturnedClose,
        exit: i32,
    ) -> ContentPin {
        let library = files.write(
            &format!("{name}-library.json"),
            &serde_json::to_vec(returned).unwrap(),
        );
        let process = files.write(
            &format!("{name}-process.json"),
            &serde_json::to_vec(&json!({
            "schema":PROCESS_CLOSE_SCHEMA,"actualClose":true,"exitCode":exit,
            "journalHead":returned.head,"libraryReturnedClose":library}))
            .unwrap(),
        );
        files.write(&format!("{name}-authority.json"),&serde_json::to_vec(&json!({
            "schema":CLOSE_AUTHORITY_SCHEMA,"rootAdmitted":true,"rootObserved":true,
            "journalHead":returned.head,"libraryReturnedClose":library,"observedProcessClose":process})).unwrap())
    }
    fn restore_test(
        directory: &Path,
        head: &ContentPin,
        authority: &ContentPin,
    ) -> Result<ConditionalHumanJournal, String> {
        let mut parsed = inspect_archive(directory, head)?;
        if !parsed.logically_closed {
            return Err("Test archive is not closed".into());
        }
        require_authority(authority, &parsed.saved)?;
        parsed.source.test_outcomes = Some(vec![true; 4]);
        ConditionalHumanJournal::restore_verified(directory, parsed, None, Some(authority))
    }
    #[test]
    fn conditional_closed_restore_keeps_joint_bits_credit_rng_and_rounded_nochange() {
        for (name, rate, expected_updates) in [("updated", 0.001, 2), ("rounded", 1e-300, 0)] {
            let files = Files::new("conditional-journal-restore");
            let mut continuous = start(&files, &format!("{name}-continuous"), rate);
            let mut journal = start(&files, &format!("{name}-restart"), rate);
            continuous.update_whole_four(seeds()[0], None).unwrap();
            let first = journal.update_whole_four(seeds()[0], None).unwrap();
            let head = journal.head();
            let directory = journal.directory.clone();
            let saved = ConditionalHumanJournal::inspect_saved_tip(&directory, &head).unwrap();
            assert_eq!(saved.phase(), ConditionalJournalPhase::Stopped);
            let positive = authority(&files, "positive", &first, 0);
            let negative = authority(&files, "negative", &first, 1);
            assert!(restore_test(&directory, &head, &negative).is_err());
            drop(journal);
            let mut restored = restore_test(&directory, &head, &positive).unwrap();
            continuous.update_whole_four(seeds()[1], None).unwrap();
            restored.update_whole_four(seeds()[1], None).unwrap();
            assert_eq!(
                restored.session.owner().actor_parameters(),
                continuous.session.owner().actor_parameters()
            );
            assert_eq!(
                restored.session.owner().critic_parameters(),
                continuous.session.owner().critic_parameters()
            );
            assert_eq!(
                restored.session.owner().identity(),
                continuous.session.owner().identity()
            );
            assert_eq!(
                restored.session.credited_own_decisions(),
                continuous.session.credited_own_decisions()
            );
            let second_head = restored.head();
            let second = inspect_archive(&restored.directory, &second_head).unwrap();
            assert_eq!(second.saved.successful_updates(), expected_updates);
            assert_eq!(second.saved.next_episode_ordinal(), 8);
            assert_eq!(second.saved.spent_families.len(), 8);
            if expected_updates == 0 {
                assert_eq!(second.saved.credited_own_decisions(), 0);
                assert_eq!(first.resolution, ConditionalResolution::NoChange);
                // Nonzero residual targets may still yield zero rounded f32 change.
            } else {
                assert!(second.saved.credited_own_decisions() > 0);
            }
        }
    }
    #[test]
    fn conditional_late_source_failure_preserves_updated_credit_and_stops() {
        let files = Files::new("conditional-journal-late-source");
        let mut journal = start(&files, "journal", 0.001);
        journal.change_source_after_resolve = Some(files.root.join("source-marker.json"));
        assert!(journal.update_whole_four(seeds()[0], None).is_err());
        assert!(journal.owner().is_err());
        assert_eq!(journal.session.owner().identity().mixed_update_count(), 1);
        let saved = ConditionalHumanJournal::inspect_saved_tip(&journal.directory, &journal.head())
            .unwrap();
        assert_eq!(saved.successful_updates(), 1);
        assert!(saved.credited_own_decisions() > 0);
        assert_eq!(saved.next_episode_ordinal(), 4);
        assert_eq!(saved.spent_families.len(), 4);
        assert!(
            !inspect_archive(&journal.directory, &journal.head())
                .unwrap()
                .logically_closed
        );
        assert!(journal.update_whole_four(seeds()[1], None).is_err());
        assert_eq!(journal.session.attempts(), 1);
    }
    #[test]
    fn conditional_published_error_missing_close_and_unknown_reserve_stay_stopped() {
        for label in ["resolve", "closed"] {
            let files = Files::new("conditional-journal-publish-error");
            let mut journal = start(&files, "journal", 0.001);
            journal.fail_after_publish = Some(label);
            assert!(journal.update_whole_four(seeds()[0], None).is_err());
            assert!(journal.owner().is_err());
            let saved =
                ConditionalHumanJournal::inspect_saved_tip(&journal.directory, &journal.head())
                    .unwrap();
            assert_eq!(saved.phase(), ConditionalJournalPhase::Stopped);
            assert_eq!(saved.successful_updates(), 1);
            assert_eq!(saved.next_episode_ordinal(), 4);
            assert!(saved.credited_own_decisions() > 0);
        }
        let files = Files::new("conditional-journal-orphan-payload");
        let mut journal = start(&files, "journal", 0.001);
        journal.fail_after_publish = Some("payload");
        assert!(journal.update_whole_four(seeds()[0], None).is_err());
        assert!(journal.directory.join("report-0001.json").is_file());
        assert!(journal.owner().is_err());
        let saved = ConditionalHumanJournal::inspect_saved_tip(&journal.directory, &journal.head())
            .unwrap();
        assert_eq!(saved.last_resolution, Some(ConditionalResolution::Unknown));
        assert_eq!(saved.attempts(), 1);
        assert_eq!(saved.next_episode_ordinal(), 4);
        assert_eq!(saved.successful_updates(), 0);
        assert_eq!(saved.spent_families.len(), 4);
        fs::write(journal.directory.join("unrelated.json"), b"{}").unwrap();
        assert!(
            ConditionalHumanJournal::inspect_saved_tip(&journal.directory, &journal.head())
                .is_err()
        );
        let files = Files::new("conditional-journal-unknown");
        let mut journal = start(&files, "journal", 0.001);
        journal.fail_after_publish = Some("head");
        assert!(journal.update_whole_four(seeds()[0], None).is_err());
        assert!(journal.owner().is_err());
        let saved = ConditionalHumanJournal::inspect_saved_tip(&journal.directory, &journal.head())
            .unwrap();
        assert_eq!(saved.last_resolution, Some(ConditionalResolution::Unknown));
        assert_eq!(saved.next_episode_ordinal(), 4);
        assert_eq!(saved.successful_updates(), 0);
        assert_eq!(saved.spent_families.len(), 4);
    }
}
fn absolute_pin(directory: &Path, pin: &Pin) -> ContentPin {
    ContentPin {
        path: directory.join(&pin.name).to_string_lossy().into_owned(),
        bytes: pin.bytes,
        sha256: pin.sha256.clone(),
    }
}
fn bounded_error(error: &str) -> String {
    error.chars().take(2048).collect()
}

struct Parsed {
    entries: Vec<(Pin, Entry)>,
    source: SourceSeal,
    spec: ConditionalJournalSpec,
    pool: serde_json::Value,
    opponent_binding: serde_json::Value,
    root_parameters: Pin,
    saved: ConditionalSavedTip,
    members: BTreeSet<String>,
    logical_bytes: u64,
    logically_closed: bool,
}
fn field<'a>(value: &'a serde_json::Value, key: &str) -> Result<&'a serde_json::Value, String> {
    value
        .get(key)
        .ok_or_else(|| format!("Missing conditional metadata field {key}"))
}
fn number(value: &serde_json::Value, key: &str) -> Result<u64, String> {
    field(value, key)?
        .as_u64()
        .ok_or_else(|| format!("Invalid conditional counter {key}"))
}
fn parameters(value: &serde_json::Value, key: &str) -> Result<Vec<f32>, String> {
    let result: Vec<f32> =
        serde_json::from_value(field(value, key)?.clone()).map_err(|e| e.to_string())?;
    if result.iter().any(|v| !v.is_finite()) {
        return Err("Nonfinite conditional joint parameters".into());
    }
    Ok(result)
}
fn inspect_archive(directory: &Path, expected: &ContentPin) -> Result<Parsed, String> {
    let directory = local(directory, false)?;
    let expected_path = local(Path::new(&expected.path), false)?;
    if expected_path.parent() != Some(directory.as_path()) {
        return Err("Expected journal head is not directly owned".into());
    }
    let mut pin = Pin {
        name: expected_path
            .file_name()
            .and_then(|n| n.to_str())
            .ok_or("Invalid expected head")?
            .into(),
        bytes: expected.bytes,
        sha256: expected.sha256.clone(),
    };
    let mut entries = Vec::new();
    loop {
        if entries.len() > 3 * MAX_ATTEMPTS as usize {
            return Err("Conditional chain entry cap".into());
        }
        let entry = decode_entry(&read_pin(&directory, &pin, HEAD_BYTES)?)?;
        if pin.name != format!("head-{:04}.json", entry.sequence) {
            return Err("Conditional head sequence/name differs".into());
        }
        let previous = entry.previous.clone();
        entries.push((pin, entry));
        match previous {
            Some(p) => pin = p,
            None => break,
        }
    }
    entries.reverse();
    let (source, spec, pool, opponent_binding, root_pin) = match &entries[0].1.event {
        Event::Init {
            source,
            spec,
            pool,
            opponent_binding,
            root_parameters,
        } if entries[0].1.sequence == 0 => (
            source.clone(),
            spec.clone(),
            pool.clone(),
            opponent_binding.clone(),
            root_parameters.clone(),
        ),
        _ => return Err("Conditional chain has no count0 Init".into()),
    };
    spec.validate()?;
    if entries.len() > 3 * spec.max_attempts as usize + 1 {
        return Err("Declared journal entry cap".into());
    }
    let initial = read_pin(&directory, &root_pin, JOINT_BYTES)?;
    let value = unique_json(&initial, JOINT_BYTES)?;
    if field(&value, "schema")? != "tzolkin-conditional-journal-root-parameters-v1"
        || field(&value, "identity")?
            != &serde_json::to_value(&source.root).map_err(|e| e.to_string())?
        || source.root.mixed_update_count() != 0
        || source.root.next_ordinal() != 0
        || source.root.origin_checksum()
            != field(&source.origin, "checksum")?
                .as_str()
                .ok_or("Origin checksum shape")?
    {
        return Err("Conditional initial identity/origin differs".into());
    }
    let bits = |key| -> Result<Vec<f32>, String> {
        let words: Vec<u32> =
            serde_json::from_value(field(&value, key)?.clone()).map_err(|e| e.to_string())?;
        Ok(words.into_iter().map(f32::from_bits).collect())
    };
    let mut actor = bits("actorBits")?;
    let mut critic = bits("criticBits")?;
    if critic != crate::public_policy_baseline::zero_output_parameters(source.input.critic_seed) {
        return Err("Conditional Init critic is not the fresh seeded initializer".into());
    }
    crate::conditional_human_rl_update::check_identity_bits(&source.root, &actor, &critic)?;
    let mut tip = source.root.clone();
    let mut attempts = 0u64;
    let mut next = 0u64;
    let mut credited = 0u64;
    let mut spent = BTreeSet::new();
    let mut phase = ConditionalJournalPhase::Ready;
    let mut pending: Option<([String; 4], Pin)> = None;
    let mut last_resolve = None;
    let mut resolution = None;
    let mut members = BTreeSet::from([root_pin.name.clone()]);
    let mut logical = 0u64;
    for (index, (entry_pin, entry)) in entries.iter().enumerate() {
        if entry.sequence != index as u64
            || (index > 0 && entry.previous.as_ref() != Some(&entries[index - 1].0))
        {
            return Err("Conditional journal hash-chain predecessor differs".into());
        }
        members.insert(entry_pin.name.clone());
        if index == 0 {
            continue;
        }
        match &entry.event {
            Event::Init { .. } => return Err("Conditional duplicate Init".into()),
            Event::Reserve {
                attempt,
                first_ordinal,
                seeds,
                families,
                parent,
            } => {
                if phase != ConditionalJournalPhase::Ready
                    || *attempt != attempts + 1
                    || *first_ordinal != next
                    || *parent != tip
                    || *attempt > spec.max_attempts
                    || spent.len() + 4 > spec.max_spent_families
                {
                    return Err("Conditional Reserve parent/accounting/phase differs".into());
                }
                for (i, seed) in seeds.iter().enumerate() {
                    let family = crate::dataset::seed_family_id(*seed);
                    if family != families[i]
                        || crate::dataset::split_for_family(&family)?
                            != crate::dataset::DatasetSplit::Train
                        || !spent.insert(family)
                    {
                        return Err("Conditional Reserve repeats/nonTrain family".into());
                    }
                }
                attempts = *attempt;
                next += 4;
                pending = Some((families.clone(), entry_pin.clone()));
                phase = ConditionalJournalPhase::Reserved;
                resolution = None;
            }
            Event::Resolve {
                attempt,
                resolution: kind,
                report,
                joint,
                records,
                tip: new_tip,
                credited_own,
                error,
            } => {
                let (families, _) = pending
                    .as_ref()
                    .ok_or("Conditional Resolve without Reserve")?;
                if phase != ConditionalJournalPhase::Reserved || *attempt != attempts {
                    return Err("Conditional duplicate/out-of-phase Resolve".into());
                }
                match kind {
                    ConditionalResolution::Failed | ConditionalResolution::Unknown => {
                        if report.is_some()
                            || joint.is_some()
                            || records.is_some()
                            || *new_tip != tip
                            || *credited_own != credited
                            || error.as_ref().is_none_or(|s| s.len() > 8192)
                        {
                            return Err("Failed/Unknown resolution changes joint credit/tip".into());
                        }
                        phase = ConditionalJournalPhase::Stopped;
                    }
                    ConditionalResolution::Updated | ConditionalResolution::NoChange => {
                        if error.is_some() {
                            return Err("Successful resolution has error".into());
                        }
                        let report = report.as_ref().ok_or("Missing joint report")?;
                        let report_raw = read_pin(&directory, report, JOINT_BYTES)?;
                        members.insert(report.name.clone());
                        let r = unique_json(&report_raw, JOINT_BYTES)?;
                        let counts: [usize; 3] =
                            serde_json::from_value(field(&r, "counts")?.clone())
                                .map_err(|e| e.to_string())?;
                        let expected_families = families
                            .iter()
                            .map(|f| serde_json::json!({"digest":f}))
                            .collect::<Vec<_>>();
                        let updated = *kind == ConditionalResolution::Updated;
                        let expected_credit = credited
                            .checked_add(if updated { counts[0] as u64 } else { 0 })
                            .ok_or("Credit overflow")?;
                        if field(&r, "schema")? != crate::conditional_human_rl_update::REPORT_SCHEMA
                            || field(&r, "parent")?
                                != &serde_json::to_value(&tip).map_err(|e| e.to_string())?
                            || field(&r, "families")?
                                != &serde_json::to_value(expected_families)
                                    .map_err(|e| e.to_string())?
                            || number(&r, "proposedUpdateCount")?
                                != tip.mixed_update_count() + u64::from(updated)
                            || number(&r, "attemptedWholeFour")? != attempts
                            || number(&r, "nextEpisodeOrdinal")? != next
                            || number(&r, "creditedOwnDecisions")? != expected_credit
                            || *credited_own != expected_credit
                            || counts[0] == 0
                            || counts[0] > 4 * spec.max_callbacks
                            || counts[2] > counts[0]
                            || field(&r, "sourceQualified")? != false
                            || field(&r, "verifiedNativePolicy")? != false
                            || field(&r, "durableAdmission")? != false
                        {
                            return Err("Conditional Resolve report/count/domain differs".into());
                        }
                        if updated {
                            let pin = joint
                                .as_ref()
                                .ok_or("Updated resolution has no joint artifact")?;
                            let raw = read_pin(&directory, pin, JOINT_BYTES)?;
                            members.insert(pin.name.clone());
                            let artifact = unique_json(&raw, JOINT_BYTES)?;
                            if field(&artifact, "schema")?
                                != crate::conditional_human_rl_update::ARTIFACT_SCHEMA
                                || field(&artifact, "identity")?
                                    != &serde_json::to_value(new_tip).map_err(|e| e.to_string())?
                                || field(&artifact, "reportChecksum")? != field(&r, "checksum")?
                            {
                                return Err(
                                    "Conditional joint artifact/report identity differs".into()
                                );
                            }
                            actor = parameters(&artifact, "actorParameters")?;
                            critic = parameters(&artifact, "criticParameters")?;
                        } else if joint.is_some() {
                            return Err("NoChange has a new joint artifact".into());
                        }
                        if crate::conditional_human_rl_update::checked_tip_identity(
                            &tip, &actor, &critic, next, updated,
                        )? != *new_tip
                        {
                            return Err("Conditional joint bit identity differs".into());
                        }
                        for record in records
                            .as_ref()
                            .ok_or("Missing complete whole-four records")?
                        {
                            let stored = read_pin(
                                &directory,
                                &record.stored,
                                storage::MAX_STORED_RECORD_BYTES,
                            )?;
                            storage::decode(&stored, record.raw_bytes, &record.raw_sha256)?;
                            members.insert(record.stored.name.clone());
                            logical = logical
                                .checked_add(record.raw_bytes as u64)
                                .ok_or("Logical byte overflow")?;
                        }
                        if logical > spec.max_logical_bytes {
                            return Err("Conditional logical storage cap".into());
                        }
                        tip = new_tip.clone();
                        credited = *credited_own;
                        phase = ConditionalJournalPhase::ResolvedAwaitingClose;
                    }
                }
                resolution = Some(*kind);
                last_resolve = Some(entry_pin.clone());
                pending = None;
            }
            Event::Closed {
                attempt,
                resolve,
                tip_checksum,
            } => {
                if phase != ConditionalJournalPhase::ResolvedAwaitingClose
                    || *attempt != attempts
                    || last_resolve.as_ref() != Some(resolve)
                    || tip.checksum() != tip_checksum
                    || entry.previous.as_ref() != Some(resolve)
                {
                    return Err("Conditional close is not the exact resolved joint tip".into());
                }
                phase = ConditionalJournalPhase::Ready;
            }
            Event::CloseFailed {
                attempt,
                resolve,
                error,
            }
            | Event::CloseUnknown {
                attempt,
                resolve,
                error,
            } => {
                if !matches!(
                    phase,
                    ConditionalJournalPhase::ResolvedAwaitingClose
                        | ConditionalJournalPhase::Stopped
                ) || *attempt != attempts
                    || last_resolve.as_ref() != Some(resolve)
                    || error.len() > 8192
                    || entry.previous.as_ref() != Some(resolve)
                {
                    return Err("Conditional failed close changes/repeats resolution".into());
                }
                phase = ConditionalJournalPhase::Stopped;
            }
        }
    }
    let logically_closed = phase == ConditionalJournalPhase::Ready
        && attempts > 0
        && matches!(
            entries.last().map(|(_, e)| &e.event),
            Some(Event::Closed { .. })
        );
    if phase == ConditionalJournalPhase::Reserved {
        resolution = Some(ConditionalResolution::Unknown);
    }
    // Data-only inspection keeps a failed attempt charged even if a known
    // payload destination was published but not referenced by a Resolve.
    // These bounded leaves never grant a Ready owner or a successful close.
    let mut attempted_leaves = BTreeSet::from([format!("head-{:04}.json", entries.len())]);
    if attempts > 0 {
        attempted_leaves.insert(format!("report-{attempts:04}.json"));
        attempted_leaves.insert(format!("joint-{attempts:04}.json"));
        for i in 0..4 {
            attempted_leaves.insert(format!("record-{attempts:04}-{i}.json.gz"));
        }
    }
    owned_usage(&directory, &spec, &members, &attempted_leaves, false)?;
    let head = expected.clone();
    let saved = ConditionalSavedTip {
        phase: ConditionalJournalPhase::Stopped,
        attempts,
        next_episode_ordinal: next,
        successful_updates: tip.mixed_update_count(),
        credited_own: credited,
        spent_families: spent,
        tip,
        head,
        last_resolution: resolution,
        actual_process_close_verified: false,
        authority_producer_authenticated: false,
        source_qualified: false,
        verified_native_policy: false,
    };
    Ok(Parsed {
        entries,
        source,
        spec,
        pool,
        opponent_binding,
        root_parameters: root_pin,
        saved,
        members,
        logical_bytes: logical,
        logically_closed,
    })
}
impl ConditionalHumanJournal {
    pub fn inspect_saved_tip(
        directory: &Path,
        expected_head: &ContentPin,
    ) -> Result<ConditionalSavedTip, String> {
        Ok(inspect_archive(directory, expected_head)?.saved)
    }
}
#[derive(Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct Authority {
    schema: String,
    journal_head: ContentPin,
    library_returned_close: ContentPin,
    observed_process_close: ContentPin,
    root_admitted: bool,
    root_observed: bool,
}
#[derive(Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct Returned {
    head: ContentPin,
    tip_checksum: String,
    attempt: u64,
    resolution: ConditionalResolution,
    library_close_returned: bool,
    actual_process_close_verified: bool,
}
#[derive(Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct ProcessClose {
    schema: String,
    actual_close: bool,
    exit_code: i32,
    journal_head: ContentPin,
    library_returned_close: ContentPin,
}
fn external_raw(pin: &ContentPin) -> Result<Vec<u8>, String> {
    let path = local(Path::new(&pin.path), false)?;
    if !path.is_absolute()
        || pin.bytes == 0
        || pin.bytes > JOINT_BYTES as u64
        || !checkpoint_digest(&pin.sha256)
    {
        return Err("Invalid bounded external close custody".into());
    }
    let raw = read_bounded(&path, JOINT_BYTES)?;
    if raw.len() as u64 != pin.bytes || crate::model::digest(&raw) != pin.sha256 {
        return Err("External close binding changed".into());
    }
    Ok(raw)
}
fn require_authority(pin: &ContentPin, saved: &ConditionalSavedTip) -> Result<(), String> {
    let authority: Authority =
        serde_json::from_value(unique_json(&external_raw(pin)?, JOINT_BYTES)?)
            .map_err(|e| e.to_string())?;
    if authority.schema != CLOSE_AUTHORITY_SCHEMA
        || !authority.root_admitted
        || !authority.root_observed
        || authority.journal_head != saved.head
    {
        return Err("Missing independently pinned close admission".into());
    }
    let returned: Returned = serde_json::from_value(unique_json(
        &external_raw(&authority.library_returned_close)?,
        JOINT_BYTES,
    )?)
    .map_err(|e| e.to_string())?;
    let observed: ProcessClose = serde_json::from_value(unique_json(
        &external_raw(&authority.observed_process_close)?,
        JOINT_BYTES,
    )?)
    .map_err(|e| e.to_string())?;
    if !returned.library_close_returned
        || returned.actual_process_close_verified
        || returned.head != saved.head
        || returned.tip_checksum != saved.tip.checksum()
        || returned.attempt != saved.attempts
        || Some(returned.resolution) != saved.last_resolution
        || observed.schema != PROCESS_CLOSE_SCHEMA
        || !observed.actual_close
        || observed.exit_code != 0
        || observed.journal_head != saved.head
        || observed.library_returned_close != authority.library_returned_close
    {
        return Err("Negative/mismatched actual close dominates saved positive metadata".into());
    }
    Ok(())
}
impl ConditionalHumanJournal {
    /// The caller supplies an independently trusted pin for its actual observed
    /// successful process-close admission. This library checks bindings, not the
    /// authenticity of that observer. Default inspection never emits Ready.
    pub fn restore_ready(
        directory: &Path,
        expected_head: &ContentPin,
        authority_pin: &ContentPin,
        opponent: Option<&PreparedPublicPolicy>,
    ) -> Result<Self, String> {
        let parsed = inspect_archive(directory, expected_head)?;
        if !parsed.logically_closed {
            return Err("Conditional saved tip is not successfully closed".into());
        }
        require_authority(authority_pin, &parsed.saved)?;
        Self::restore_verified(directory, parsed, opponent, Some(authority_pin))
    }
    fn restore_verified(
        directory: &Path,
        parsed: Parsed,
        opponent: Option<&PreparedPublicPolicy>,
        authority_pin: Option<&ContentPin>,
    ) -> Result<Self, String> {
        let directory = local(directory, false)?;
        let mut session = parsed.source.full_load()?;
        if root_parameters(&session)? != read_pin(&directory, &parsed.root_parameters, JOINT_BYTES)?
        {
            return Err("Conditional root actor/fresh critic bits differ on restore".into());
        }
        let pool = ConditionalFrozenPool::from_json(&bounded(&parsed.pool, 64 * 1024)?)?;
        if pool.binding(opponent)? != parsed.opponent_binding {
            return Err("Frozen opponents differ on restore".into());
        }
        let mut plan: Option<ConditionalReservation> = None;
        for (_, entry) in parsed.entries.iter().skip(1) {
            match &entry.event {
                Event::Reserve {
                    seeds,
                    families,
                    parent,
                    first_ordinal,
                    attempt,
                } => {
                    let current = session.plan(
                        *seeds,
                        &pool,
                        opponent,
                        parsed.spec.limits()?,
                        parsed.spec.actor()?,
                        parsed.spec.critic()?,
                    )?;
                    if current.families() != *families
                        || current.parent() != parent
                        || session.next_episode_ordinal() != *first_ordinal
                        || session.attempts() + 1 != *attempt
                    {
                        return Err(
                            "Saved Reserve differs from actual loaded owner/families".into()
                        );
                    }
                    session.reserve(&current)?;
                    plan = Some(current);
                }
                Event::Resolve {
                    resolution,
                    report,
                    joint,
                    records,
                    tip,
                    credited_own,
                    ..
                } => {
                    if !matches!(
                        resolution,
                        ConditionalResolution::Updated | ConditionalResolution::NoChange
                    ) {
                        return Err(
                            "Failed/Unknown saved lineage has no automatic continuation".into()
                        );
                    }
                    let records = records.as_ref().ok_or("Missing saved conditional cohort")?;
                    let raw: [Vec<u8>; 4] = records
                        .iter()
                        .map(|r| {
                            storage::decode(
                                &read_pin(&directory, &r.stored, storage::MAX_STORED_RECORD_BYTES)?,
                                r.raw_bytes,
                                &r.raw_sha256,
                            )
                        })
                        .collect::<Result<Vec<_>, _>>()?
                        .try_into()
                        .map_err(|_| "Saved cohort shape")?;
                    #[cfg(test)]
                    let staged = if let Some(outcomes) = &parsed.source.test_outcomes {
                        crate::conditional_human_rl_update::test_short_stage(
                            &session,
                            plan.as_ref().ok_or("Missing saved Reserve")?,
                            &pool,
                            *outcomes
                                .get((session.attempts() - 1) as usize)
                                .ok_or("Missing tiny fixture case")?,
                            Some(&raw),
                        )?
                    } else {
                        session.stage(
                            plan.as_ref().ok_or("Missing saved Reserve")?,
                            &pool,
                            opponent,
                            Some(&raw),
                        )?
                    };
                    #[cfg(not(test))]
                    let staged = session.stage(
                        plan.as_ref().ok_or("Missing saved Reserve")?,
                        &pool,
                        opponent,
                        Some(&raw),
                    )?;
                    let (actual_report, actual_joint) = staged.payload()?;
                    if actual_report
                        != read_pin(
                            &directory,
                            report.as_ref().ok_or("Missing saved report")?,
                            JOINT_BYTES,
                        )?
                        || actual_joint
                            != joint
                                .as_ref()
                                .map(|p| read_pin(&directory, p, JOINT_BYTES))
                                .transpose()?
                        || session.proposed_identity(&staged)? != *tip
                        || staged.own_credit() != *credited_own
                        || staged.is_updated() != (*resolution == ConditionalResolution::Updated)
                    {
                        return Err(
                            "Saved joint report/artifact differs from genuine fresh audit/SGD"
                                .into(),
                        );
                    }
                    session.admit(staged)?;
                    plan = None;
                }
                Event::Closed { .. } => {}
                _ => return Err("Unclosed/failed journal cannot restore a usable owner".into()),
            }
        }
        if plan.is_some()
            || *session.owner().identity() != parsed.saved.tip
            || session.attempts() != parsed.saved.attempts
            || session.next_episode_ordinal() != parsed.saved.next_episode_ordinal
            || session.credited_own_decisions() != parsed.saved.credited_own
            || session
                .consumed_families()
                .iter()
                .map(|f| f.digest().to_owned())
                .collect::<BTreeSet<_>>()
                != parsed.saved.spent_families
        {
            return Err("Conditional restored joint owner/accounting/RNG differs".into());
        }
        let (head, entry) = parsed.entries.last().ok_or("Empty conditional archive")?;
        let journal = Self {
            directory,
            source: parsed.source,
            spec: parsed.spec,
            pool,
            opponent_binding: parsed.opponent_binding,
            session,
            head: head.clone(),
            sequence: entry.sequence,
            phase: ConditionalJournalPhase::Ready,
            members: parsed.members,
            logical_bytes: parsed.logical_bytes,
            #[cfg(test)]
            fail_after_publish: None,
            #[cfg(test)]
            change_source_after_resolve: None,
        };
        journal.usage(&BTreeSet::new(), true)?;
        journal.source.full_load()?;
        journal.require_head()?;
        if let Some(pin) = authority_pin {
            require_authority(pin, &parsed.saved)?;
        }
        Ok(journal)
    }
}
