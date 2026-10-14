//! Single-writer semantic-4 durability and source-owned cold reconstruction.
//! Only a fresh loaded owner enters. Saved descriptors/records never construct
//! an owner, snapshot or actor step. Library Closed permits warm continuation;
//! cold reconstruction additionally requires a pinned external close declaration.
//! That declaration is evidence supplied by the caller, not OS authentication.
//! Warm checks cover output membership/lengths, not historical payload bytes:
//! only the new destination is read back by the shared publisher. Historical
//! outputs are never warm learning inputs. Cold restore must hash and replay.
use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    path::{Path, PathBuf},
};

use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::conditional_human_rl_update::ConditionalFrozenPool;
use crate::human_initial_wealth::ContentPin;
use crate::public_policy_update::hash;
use crate::public_rl_paired_session::{
    local, publish_mixed_metadata, publish_mixed_stored_record, read_bounded,
    read_mixed_stored_record,
};
use crate::public_rl_record_storage as storage;
use crate::public_stochastic_native::{CollectionLimits, NativeStochasticConfig};
use crate::semantic_playing_rl_collection::PoolRef;
use crate::semantic_playing_rl_pool::{FrozenSemanticSnapshot, SemanticFrozenMixture};
use crate::semantic_playing_rl_update::{
    PreparedSemanticAdmission, SemanticPlayingCurrentIdentity, SemanticPlayingJointReport,
    SemanticPlayingJointSession,
};

pub const JOURNAL_SCHEMA: &str = "tzolkin-semantic-playing-joint-journal-v1";
#[path = "semantic_playing_rl_restore.rs"]
mod restore;
pub use restore::{
    CLOSE_DECLARATION_SCHEMA, PROCESS_CLOSE_SCHEMA, SemanticJournalRestoration,
    restore_observed_closed,
};
#[path = "semantic_playing_rl_run.rs"]
mod runner;
pub use runner::{
    SemanticJournalAudit, SemanticJournalAuditSpec, SemanticJournalInvocation,
    SemanticJournalRunSpec, audit_semantic_playing_journal, run_semantic_playing_journal,
};
const META_BYTES: usize = 16 * 1024 * 1024;
const MAX_FILES: usize = 640;

#[derive(Clone, Copy, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct SemanticJournalBudget {
    max_attempts: usize,
    max_logical_bytes: u64,
    max_stored_bytes: u64,
    max_metadata_bytes: u64,
    max_disk_bytes: u64,
}
impl SemanticJournalBudget {
    pub fn new(
        attempts: usize,
        logical: u64,
        stored: u64,
        metadata: u64,
        disk: u64,
    ) -> Result<Self, String> {
        if !(1..=64).contains(&attempts)
            || !(1..=(64u64 << 30)).contains(&logical)
            || !(1..=(64u64 << 30)).contains(&stored)
            || !(META_BYTES as u64..=(1u64 << 30)).contains(&metadata)
            || !(stored.saturating_add(metadata)..=(66u64 << 30)).contains(&disk)
        {
            return Err("Semantic journal finite budget differs".into());
        }
        Ok(Self {
            max_attempts: attempts,
            max_logical_bytes: logical,
            max_stored_bytes: stored,
            max_metadata_bytes: metadata,
            max_disk_bytes: disk,
        })
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "camelCase")]
pub enum SemanticJournalPhase {
    Closed,
    Reserved,
    ResolvedAwaitingClose,
    Stopped,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "camelCase")]
pub enum SemanticJournalResolution {
    Updated,
    NoChange,
    AbortedBeforeAdmission,
}

/// Accounting only, including after an error. No source/OS-close authentication.
#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct SemanticJournalStatus {
    phase: SemanticJournalPhase,
    current: SemanticPlayingCurrentIdentity,
    attempts: usize,
    next_ordinal: u64,
    spent_families: usize,
    spent_actor_streams: usize,
    reservation_publication_confirmed: bool,
    last_resolution: Option<SemanticJournalResolution>,
    actual_process_close_verified: bool,
    source_qualified: bool,
    verified_native_policy: bool,
    historical_archive_bytes_verified: bool,
}
/// A returned library close is not an observed producing-process exit.
#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct SemanticJournalReturnedClose {
    head: ContentPin,
    report: ContentPin,
    current: String,
    attempt: usize,
    resolution: SemanticJournalResolution,
    library_close_returned: bool,
    actual_process_close_verified: bool,
    historical_archive_bytes_verified: bool,
}
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct Pin {
    name: String,
    bytes: usize,
    sha256: String,
}
impl Pin {
    fn new(name: &str, raw: &[u8]) -> Self {
        Self {
            name: name.into(),
            bytes: raw.len(),
            sha256: crate::model::digest(raw),
        }
    }
    fn absolute(&self, directory: &Path) -> ContentPin {
        ContentPin {
            path: directory.join(&self.name).to_string_lossy().into_owned(),
            bytes: self.bytes as u64,
            sha256: self.sha256.clone(),
        }
    }
}
struct Publication {
    pin: Pin,
    raw: Vec<u8>,
    members: BTreeMap<String, Pin>,
    sequence: u64,
}
struct PublishError {
    error: String,
    published: bool,
}
#[derive(Deserialize, Serialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct StoredRecord {
    stored: Pin,
    raw_bytes: usize,
    raw_sha256: String,
}

/// No Deserialize, parameter import/export or caller predictor.
pub struct SemanticPlayingJournal<'a> {
    directory: PathBuf,
    session: SemanticPlayingJointSession<'a>,
    limits: CollectionLimits,
    budget: SemanticJournalBudget,
    phase: SemanticJournalPhase,
    head: Pin,
    sequence: u64,
    members: BTreeMap<String, Pin>,
    captures: BTreeSet<String>,
    logical_bytes: u64,
    reservation_confirmed: bool,
    resolution: Option<SemanticJournalResolution>,
    #[cfg(test)]
    fail_after_publish: Option<&'static str>,
    #[cfg(test)]
    change_source_after_resolve: Option<PathBuf>,
}
impl<'a> SemanticPlayingJournal<'a> {
    pub fn start(
        directory: &Path,
        session: SemanticPlayingJointSession<'a>,
        limits: CollectionLimits,
        budget: SemanticJournalBudget,
    ) -> Result<Self, String> {
        if session.attempts() != 0
            || session.stopped()
            || session.identity()?.completed_updates() != 0
        {
            return Err("Semantic journal requires its genuine fresh root".into());
        }
        let source = session.journal_source()?;
        let directory = local(directory, true)?;
        if fs::symlink_metadata(&directory).is_ok() {
            return Err("Semantic journal needs fresh output".into());
        }
        reject_input_namespace(&directory, &source)?;
        let raw = entry(
            0,
            None,
            &json!({"kind":"init","source":source,"budget":budget,
            "limits":[limits.max_callbacks(),limits.max_candidate_rows(),limits.max_source_bytes(),limits.max_trace_bytes()],
            "actualProcessCloseVerified":false,"sourceQualified":false,"verifiedNativePolicy":false}),
        )?;
        if raw.len() as u64 > budget.max_metadata_bytes
            || 2 * raw.len() as u64 > budget.max_disk_bytes
        {
            return Err("Semantic init metadata budget".into());
        }
        let head = Pin::new("entry-0000.json", &raw);
        let members = BTreeMap::from([(head.name.clone(), head.clone())]);
        fs::create_dir(&directory).map_err(|e| e.to_string())?;
        publish_mixed_metadata(&directory, &head.name, &raw)?;
        let journal = Self {
            directory,
            session,
            limits,
            budget,
            phase: SemanticJournalPhase::Closed,
            head,
            sequence: 0,
            members,
            captures: BTreeSet::new(),
            logical_bytes: 0,
            reservation_confirmed: false,
            resolution: None,
            #[cfg(test)]
            fail_after_publish: None,
            #[cfg(test)]
            change_source_after_resolve: None,
        };
        journal.session.journal_recheck()?;
        journal.verify_files()?;
        Ok(journal)
    }
    pub fn status(&self) -> SemanticJournalStatus {
        SemanticJournalStatus {
            phase: self.phase,
            current: self.session.current_unchecked().clone(),
            attempts: self.session.attempts(),
            next_ordinal: self.session.next_episode_ordinal(),
            spent_families: self.session.consumed_families().len(),
            spent_actor_streams: self.session.consumed_actor_streams(),
            reservation_publication_confirmed: self.reservation_confirmed,
            last_resolution: self.resolution,
            actual_process_close_verified: false,
            source_qualified: false,
            verified_native_policy: false,
            historical_archive_bytes_verified: false,
        }
    }
    pub fn head(&self) -> ContentPin {
        self.head.absolute(&self.directory)
    }
    /// Explicit full byte-integrity check for an outer/final or cold boundary.
    /// This checks saved pins only, not replay, source custody, an OS close or
    /// restoration authority. It does not change the live accounting state.
    pub fn verify_archive_bytes(&self) -> Result<(), String> {
        self.verify_files()?;
        for pin in self.members.values() {
            self.read_pin(pin)?;
        }
        self.verify_files()
    }
    fn stop(&mut self) {
        self.phase = SemanticJournalPhase::Stopped;
        self.session.stop();
    }
    fn require_closed(&self) -> Result<(), String> {
        if self.phase != SemanticJournalPhase::Closed || self.session.stopped() {
            return Err("Semantic journal is pending, unclosed or stopped".into());
        }
        self.session.journal_recheck()?;
        self.verify_files()
    }
    fn evaluation_recheck(&self) -> Result<(), String> {
        // Closed checks source custody and membership/lengths; archive bytes
        // need one separate full hash pass, even if the source check fails.
        match (self.require_closed(), self.verify_archive_bytes()) {
            (Ok(()), Ok(())) => Ok(()),
            (Err(error), Ok(())) => Err(error),
            (Ok(()), Err(post)) => Err(format!("Semantic evaluation archive: {post}")),
            (Err(error), Err(post)) => Err(format!("{error}; semantic evaluation archive: {post}")),
        }
    }
    /// Borrow one owned current actor for read-only evaluation. The scoped
    /// borrow cannot escape; no capture is published or allocation consumed.
    /// Errors also run the source/archive postcheck and never stop the owner.
    pub fn with_evaluation_snapshot<R>(
        &self,
        run: impl for<'eval> FnOnce(&'eval FrozenSemanticSnapshot<'a>) -> Result<R, String>,
    ) -> Result<R, String> {
        let result = (|| {
            self.evaluation_recheck()?;
            let snapshot = self.session.freeze_snapshot()?;
            run(&snapshot)
        })();
        match (result, self.evaluation_recheck()) {
            (Ok(value), Ok(())) => Ok(value),
            (Err(error), Ok(())) => Err(error),
            (Ok(_), Err(post)) => Err(format!("Semantic evaluation postcheck: {post}")),
            (Err(error), Err(post)) => {
                Err(format!("{error}; semantic evaluation postcheck: {post}"))
            }
        }
    }
    /// Warm, internally closed capture only. Its actor stays typed in memory;
    /// this metadata is a replay reference, never a weight/snapshot constructor.
    pub fn freeze_snapshot(&mut self) -> Result<FrozenSemanticSnapshot<'a>, String> {
        let result = (|| {
            self.require_closed()?;
            let snapshot = self.session.freeze_snapshot()?;
            let id = snapshot.identity()?.checksum().to_string();
            let mut captures = self.captures.clone();
            if captures.len() > self.budget.max_attempts || !captures.insert(id) {
                return Err("Semantic capture must be unique at a closed boundary".into());
            }
            let publication = self.prepare_entry(
                &json!({"kind":"capture","snapshot":snapshot.journal_identity()?,
                "ledger":self.session.capture_ledger()}),
            )?;
            self.publish(publication, "capture").map_err(|e| e.error)?;
            self.session.journal_recheck()?;
            self.verify_files()?;
            self.captures = captures;
            Ok(snapshot)
        })();
        if result.is_err() {
            self.stop();
        }
        result
    }
    pub fn update_procedural_whole_four(
        &mut self,
        seeds: [u32; 4],
        pool: &ConditionalFrozenPool,
    ) -> Result<SemanticJournalReturnedClose, String> {
        self.update_using(
            seeds,
            PoolRef::Legacy(pool, None),
            crate::semantic_playing_rl_collection::create,
        )
    }
    pub fn update_mixed_whole_four(
        &mut self,
        seeds: [u32; 4],
        mixture: &SemanticFrozenMixture<'_>,
    ) -> Result<SemanticJournalReturnedClose, String> {
        self.update_using(
            seeds,
            PoolRef::Mixture(mixture),
            crate::semantic_playing_rl_collection::create,
        )
    }
    fn preflight(&self) -> Result<(), String> {
        self.require_closed()?;
        self.check_attempt_capacity()
    }
    fn check_attempt_capacity(&self) -> Result<(), String> {
        if self.session.attempts() >= self.budget.max_attempts {
            return Err("Semantic journal attempt cap".into());
        }
        let raw = (self.limits.max_source_bytes() + self.limits.max_trace_bytes()) as u64;
        let stored = storage::maximum_stored_bytes(raw as usize)? as u64;
        let (used_stored, used_meta) = self.usage()?;
        if self.logical_bytes + 4 * raw > self.budget.max_logical_bytes
            || used_stored + 4 * stored > self.budget.max_stored_bytes
            || used_meta + 4 * META_BYTES as u64 > self.budget.max_metadata_bytes
            || used_stored + used_meta + 5 * stored + 5 * META_BYTES as u64
                > self.budget.max_disk_bytes
        {
            return Err("A complete semantic attempt cannot fit before Reserve".into());
        }
        Ok(())
    }
    fn update_using(
        &mut self,
        seeds: [u32; 4],
        pool: PoolRef<'_>,
        create: impl Fn(&NativeStochasticConfig) -> Result<tzolkin_core::GameState, String>,
    ) -> Result<SemanticJournalReturnedClose, String> {
        let result = (|| {
            self.preflight()?;
            let pool_description = pool.journal_binding()?; // Reject legacy neural/search slots before a game.
            if let Some(slots) = pool_description.get("snapshots").and_then(Value::as_array)
                && slots.iter().any(|s| {
                    s.get("snapshot")
                        .and_then(Value::as_str)
                        .is_none_or(|id| !self.captures.contains(id))
                })
            {
                return Err("Semantic mixture capture was not published by this journal".into());
            }
            let plan = self.session.plan_reserved(seeds, pool, self.limits)?;
            let reserve = self
                .prepare_entry(&json!({"kind":"reserve","plan":plan.description(),
                "fixedPool":pool_description}))?;
            let publication = self.publish(reserve, "reserve");
            // Even an uncertain publisher result permanently consumes this exact
            // allocation in the held session; no game starts on that error.
            let batch = self.session.burn_reserved(plan);
            self.phase = SemanticJournalPhase::Reserved;
            self.reservation_confirmed =
                publication.as_ref().map_or_else(|e| e.published, |_| true);
            self.resolution = None;
            publication.map_err(|e| e.error)?;
            let staged = self.session.stage_reserved_using(&batch, pool, create);
            let prepared = match staged {
                Ok(p) => p,
                Err(error) => {
                    // This states only that admission did not occur. It does not
                    // classify a source/numeric/IO error as a failed training game.
                    let abort = self.prepare_entry(
                        &json!({"kind":"resolve","reservation":batch.description(),
                        "resolution":SemanticJournalResolution::AbortedBeforeAdmission,
                        "error":error.chars().take(2048).collect::<String>(),"creditedPlaying":0}),
                    )?;
                    let publication = self.publish(abort, "abort");
                    if publication.as_ref().map_or_else(|e| e.published, |_| true) {
                        self.resolution = Some(SemanticJournalResolution::AbortedBeforeAdmission);
                    }
                    return Err(match publication {
                        Ok(()) => error,
                        Err(post) => format!("{error}; abort publication: {}", post.error),
                    });
                }
            };
            self.resolve(prepared, pool)
        })();
        if result.is_err() {
            self.stop();
        }
        result
    }
    fn resolve(
        &mut self,
        prepared: PreparedSemanticAdmission,
        pool: PoolRef<'_>,
    ) -> Result<SemanticJournalReturnedClose, String> {
        let attempt = self.session.attempts();
        let mut records = Vec::with_capacity(4);
        let mut raw_total = 0;
        for (i, raw) in prepared.records()?.iter().enumerate() {
            let stored = storage::encode(raw)?;
            let pin = self.payload(&format!("record-{attempt:04}-{i}.json.gz"), &stored, true)?;
            records.push(StoredRecord {
                stored: pin,
                raw_bytes: raw.len(),
                raw_sha256: crate::model::digest(raw),
            });
            raw_total += raw.len() as u64;
        }
        let report_raw = bounded(prepared.report())?;
        let report = self.payload(&format!("report-{attempt:04}.json"), &report_raw, false)?;
        let resolution = if prepared.report().updated() {
            SemanticJournalResolution::Updated
        } else {
            SemanticJournalResolution::NoChange
        };
        let next = prepared.report().next_identity_checksum().to_string();
        let resolve = self.prepare_entry(&json!({"kind":"resolve","attempt":attempt,
            "resolution":resolution,"report":report,"records":records,"current":next,
            "creditedPlaying":prepared.report().credited_playing_decisions()}))?;
        let closed = self.prepare_following(
            &resolve,
            &json!({"kind":"closed","attempt":attempt,
            "resolve":resolve.pin,"current":next,"actualProcessCloseVerified":false}),
        )?;
        let returned = SemanticJournalReturnedClose {
            head: closed.pin.absolute(&self.directory),
            report: report.absolute(&self.directory),
            current: next,
            attempt,
            resolution,
            library_close_returned: true,
            actual_process_close_verified: false,
            historical_archive_bytes_verified: false,
        };
        // All semantic proposal/report/identity/admitted-set allocations and
        // every prepublication source/file check precede durable Resolve.
        self.session.journal_recheck()?;
        pool.recheck()?;
        self.verify_files()?;
        let publication = self.publish(resolve, "resolve");
        if publication.as_ref().map_or_else(|e| e.published, |_| true) {
            let _report: SemanticPlayingJointReport = self.session.commit_prepared(prepared);
            self.logical_bytes += raw_total;
            self.resolution = Some(resolution);
            self.phase = SemanticJournalPhase::ResolvedAwaitingClose;
        }
        publication.map_err(|e| e.error)?;
        #[cfg(test)]
        if let Some(path) = self.change_source_after_resolve.take() {
            fs::write(path, b"changed after durable semantic Resolve")
                .map_err(|e| e.to_string())?;
        }
        // Errors here or after Closed stop the session, retaining accepted credit.
        self.session.journal_recheck()?;
        pool.recheck()?;
        self.verify_files()?;
        self.publish(closed, "closed").map_err(|e| e.error)?;
        self.session.journal_recheck()?;
        pool.recheck()?;
        self.verify_files()?;
        self.phase = if self.session.stopped() {
            SemanticJournalPhase::Stopped
        } else {
            SemanticJournalPhase::Closed
        };
        Ok(returned)
    }
    fn payload(&mut self, name: &str, raw: &[u8], record: bool) -> Result<Pin, String> {
        let pin = Pin::new(name, raw);
        let answer = pin.clone();
        let mut members = self.members.clone();
        if members.insert(name.into(), pin).is_some() {
            return Err("Semantic payload already named".into());
        }
        let (stored, metadata) = totals(&members)?;
        self.check_totals(stored, metadata)?;
        if stored + metadata + raw.len() as u64 > self.budget.max_disk_bytes {
            return Err("Semantic payload temporary byte bound".into());
        }
        self.verify_files()?;
        let result = if record {
            publish_mixed_stored_record(&self.directory, name, raw)
        } else {
            publish_mixed_metadata(&self.directory, name, raw)
        };
        if result.is_ok() || self.read_pin(&answer).is_ok() {
            self.members = members;
        }
        result?;
        Ok(answer)
    }
    fn prepare_entry(&self, event: &Value) -> Result<Publication, String> {
        self.verify_files()?;
        self.prepare_from(self.sequence, Some(&self.head), &self.members, event)
    }
    fn prepare_following(
        &self,
        previous: &Publication,
        event: &Value,
    ) -> Result<Publication, String> {
        self.prepare_from(
            previous.sequence,
            Some(&previous.pin),
            &previous.members,
            event,
        )
    }
    fn prepare_from(
        &self,
        sequence: u64,
        previous: Option<&Pin>,
        members: &BTreeMap<String, Pin>,
        event: &Value,
    ) -> Result<Publication, String> {
        let sequence = sequence
            .checked_add(1)
            .ok_or("Semantic journal sequence overflow")?;
        if sequence > 4 * self.budget.max_attempts as u64 + 1 {
            return Err("Semantic entry cap".into());
        }
        let raw = entry(sequence, previous, event)?;
        let pin = Pin::new(&format!("entry-{sequence:04}.json"), &raw);
        let mut members = members.clone();
        if members.insert(pin.name.clone(), pin.clone()).is_some() {
            return Err("Semantic entry already named".into());
        }
        let (stored, metadata) = totals(&members)?;
        self.check_totals(stored, metadata)?;
        if stored + metadata + raw.len() as u64 > self.budget.max_disk_bytes {
            return Err("Semantic entry temporary byte bound".into());
        }
        Ok(Publication {
            pin,
            raw,
            members,
            sequence,
        })
    }
    fn publish(&mut self, publication: Publication, label: &str) -> Result<(), PublishError> {
        let result =
            publish_mixed_metadata(&self.directory, &publication.pin.name, &publication.raw);
        #[cfg(test)]
        let result = if self.fail_after_publish == Some(label) {
            self.fail_after_publish = None;
            Err("Synthetic error after exact semantic destination".into())
        } else {
            result
        };
        #[cfg(not(test))]
        let _ = label;
        let published = result.is_ok() || self.read_pin(&publication.pin).is_ok();
        if published {
            self.head = publication.pin;
            self.sequence = publication.sequence;
            self.members = publication.members;
        }
        result.map_err(|error| PublishError { error, published })
    }
    fn read_pin(&self, pin: &Pin) -> Result<(), String> {
        let max = if pin.name.ends_with(".json.gz") {
            storage::MAX_STORED_RECORD_BYTES
        } else {
            META_BYTES
        };
        if pin.bytes == 0 || pin.bytes > max {
            return Err("Semantic pin byte cap".into());
        }
        let raw = if pin.name.ends_with(".json.gz") {
            read_mixed_stored_record(&self.directory.join(&pin.name), pin.bytes)?
        } else {
            read_bounded(&self.directory.join(&pin.name), pin.bytes)?
        };
        if raw.len() != pin.bytes || crate::model::digest(&raw) != pin.sha256 {
            return Err("Semantic journal pin changed".into());
        }
        Ok(())
    }
    fn verify_files(&self) -> Result<(), String> {
        let mut seen = BTreeSet::new();
        for item in fs::read_dir(local(&self.directory, false)?).map_err(|e| e.to_string())? {
            let item = item.map_err(|e| e.to_string())?;
            let name = item
                .file_name()
                .into_string()
                .map_err(|_| "Semantic owned non-UTF8 leaf")?;
            if !seen.insert(name.clone()) || seen.len() > MAX_FILES {
                return Err("Semantic owned file cap".into());
            }
            let pin = self
                .members
                .get(&name)
                .ok_or("Unexpected semantic journal output")?;
            let path = local(&item.path(), false)?;
            let metadata = fs::symlink_metadata(path).map_err(|e| e.to_string())?;
            if !metadata.is_file() || metadata.len() != pin.bytes as u64 {
                return Err("Semantic owned output type/length differs".into());
            }
        }
        if seen.len() != self.members.len() {
            return Err("Semantic journal member missing".into());
        }
        let (stored, metadata) = self.usage()?;
        self.check_totals(stored, metadata)
    }
    fn usage(&self) -> Result<(u64, u64), String> {
        totals(&self.members)
    }
    fn check_totals(&self, stored: u64, metadata: u64) -> Result<(), String> {
        if self.members.len() > MAX_FILES
            || stored > self.budget.max_stored_bytes
            || metadata > self.budget.max_metadata_bytes
            || stored + metadata > self.budget.max_disk_bytes
        {
            return Err("Semantic journal stored/metadata/disk bound".into());
        }
        Ok(())
    }
}
fn totals(members: &BTreeMap<String, Pin>) -> Result<(u64, u64), String> {
    let mut sums = [0u64; 2];
    for pin in members.values() {
        let index = usize::from(!pin.name.ends_with(".json.gz"));
        sums[index] = sums[index]
            .checked_add(pin.bytes as u64)
            .ok_or("Semantic archive size overflow")?;
    }
    Ok((sums[0], sums[1]))
}
fn bounded(value: &impl Serialize) -> Result<Vec<u8>, String> {
    let raw = serde_json::to_vec(value).map_err(|e| e.to_string())?;
    if raw.is_empty() || raw.len() > META_BYTES {
        return Err("Semantic journal metadata bound".into());
    }
    Ok(raw)
}
fn entry(sequence: u64, previous: Option<&Pin>, event: &Value) -> Result<Vec<u8>, String> {
    let mut value = json!({"schema":JOURNAL_SCHEMA,"sequence":sequence,"previous":previous,"event":event,"checksum":""});
    value["checksum"] = hash(b"semantic-playing-journal-entry-v1\0", &value)?.into();
    bounded(&value)
}
fn reject_input_namespace(output: &Path, value: &Value) -> Result<(), String> {
    match value {
        Value::Object(object) => {
            if let (Some(path), Some(_), Some(_)) = (
                object.get("path").and_then(Value::as_str),
                object.get("bytes"),
                object.get("sha256"),
            ) && local(Path::new(path), false)?.starts_with(output)
            {
                return Err("Semantic journal overlaps source custody".into());
            }
            for child in object.values() {
                reject_input_namespace(output, child)?;
            }
        }
        Value::Array(array) => {
            for child in array {
                reject_input_namespace(output, child)?;
            }
        }
        _ => {}
    }
    Ok(())
}

/// Invoked only by the existing loaded-origin fixture; games begin at its tiny
/// literal terminal turn, not a campaign/full-game or retained human record.
#[cfg(test)]
pub(crate) fn journal_fixture(
    root: &Path,
    prepared: &crate::human_policy_training::PreparedConditionalHumanPolicy,
    external: &crate::conditional_human_rl_update::ConditionalNativeExclusions,
    history: &crate::semantic_playing_rl_collection::DeclaredSamplingHistory,
    pool: &ConditionalFrozenPool,
    limits: CollectionLimits,
) {
    use crate::conditional_semantic_playing_rl::SemanticPlayingRlOwner;
    use crate::dataset::{DatasetSplit, seed_family_id, split_for_family};
    use crate::public_policy_baseline::ResidualFitConfig;
    use crate::public_policy_update::OneStepConfig;
    use crate::public_stochastic::SamplingSeed;
    use crate::semantic_playing_rl_pool::SnapshotSlot;
    let make_with_seed = |critic_seed| {
        SemanticPlayingJointSession::from_fresh(
            SemanticPlayingRlOwner::from_prepared(prepared, critic_seed).unwrap(),
            external,
            history,
            SamplingSeed::new(17),
            200,
            0,
            OneStepConfig::default(),
            ResidualFitConfig::new(0.001, 0.01, 256).unwrap(),
        )
        .unwrap()
    };
    let make = || make_with_seed(11);
    let budget = SemanticJournalBudget::new(3, 1 << 30, 1 << 30, 512 << 20, 2 << 30).unwrap();
    let seeds: [u32; 4] = (400..500)
        .filter(|s| split_for_family(&seed_family_id(*s)).unwrap() == DatasetSplit::Train)
        .take(4)
        .collect::<Vec<_>>()
        .try_into()
        .unwrap();
    let create = |config: &NativeStochasticConfig| {
        let mut state = crate::conditional_human_rl_update::tests::terminal_start(
            config.players(),
            config.environment_seed(),
        );
        state.turn.mode = tzolkin_core::TurnMode::None;
        state.turn.count = 0;
        Ok(state)
    };
    let mut journal =
        SemanticPlayingJournal::start(&root.join("journal"), make(), limits, budget).unwrap();
    let close = journal
        .update_using(seeds, PoolRef::Legacy(pool, None), create)
        .unwrap();
    assert_eq!(journal.phase, SemanticJournalPhase::Closed);
    assert!(!close.actual_process_close_verified);
    assert_eq!(journal.status().current.completed_updates(), 1);
    assert_eq!(
        (
            journal.status().attempts,
            journal.status().spent_actor_streams
        ),
        (1, 14)
    );
    let credit = journal.status().current.credited_playing_decisions();
    assert!(credit > 0);
    let reserve_raw = read_bounded(&root.join("journal/entry-0001.json"), META_BYTES).unwrap();
    let reserve = crate::public_stochastic_record::unique_json(&reserve_raw, META_BYTES).unwrap();
    for (i, config) in reserve["event"]["plan"]["members"]
        .as_array()
        .unwrap()
        .iter()
        .enumerate()
    {
        assert_eq!(config["environmentSeed"], seeds[i]);
        assert_eq!(config["episodeOrdinal"], format!("{:016x}", 200 + i));
        assert_eq!(config["players"], if i < 2 { 3 } else { 4 });
    }
    let snapshot = journal.freeze_snapshot().unwrap(); // Warm capture needs no OS-close claim.
    let mixture =
        SemanticFrozenMixture::new(pool, vec![SnapshotSlot::new(3, 3, &snapshot).unwrap()])
            .unwrap();
    let next: [u32; 4] = (500..600)
        .filter(|s| split_for_family(&seed_family_id(*s)).unwrap() == DatasetSplit::Train)
        .take(4)
        .collect::<Vec<_>>()
        .try_into()
        .unwrap();
    let mixed_close = journal
        .update_using(next, PoolRef::Mixture(&mixture), create)
        .unwrap();
    assert_eq!(journal.status().current.completed_updates(), 2);
    assert_eq!(journal.status().spent_actor_streams, 28);
    assert!(journal.status().current.credited_playing_decisions() > credit);
    // Evaluation borrows the genuine updated actor but publishes no capture,
    // including callback errors and source/archive postcheck failures.
    let evaluation_status = serde_json::to_value(journal.status()).unwrap();
    let evaluation_head = serde_json::to_value(journal.head()).unwrap();
    let evaluation_sequence = journal.sequence;
    let evaluation_captures = journal.captures.clone();
    let evaluation_files = journal
        .members
        .keys()
        .map(|name| {
            (
                name.clone(),
                fs::read(journal.directory.join(name)).unwrap(),
            )
        })
        .collect::<BTreeMap<_, _>>();
    let actor = journal
        .with_evaluation_snapshot(|snapshot| {
            use crate::arena::numeric::{ExternalExclusions, NumericArenaConfig, SCHEMA};
            use crate::arena::semantic_snapshot::{
                POLICY_VERSION, SemanticPlayingSnapshotArenaPolicy,
                SemanticPlayingSnapshotReference,
            };
            let policy = SemanticPlayingSnapshotArenaPolicy::new(snapshot)?;
            assert_eq!(policy.identity().policy_version, POLICY_VERSION);
            assert_eq!(policy.identity().backend, "scalar");
            assert!(!policy.identity().source_authenticated);
            assert!(!policy.identity().strength_measured);
            for table in snapshot.human_table_reservations()?.keys() {
                assert!(
                    snapshot.contains_family(&crate::human_initial_wealth::table_family_id(table)?)
                );
            }
            // Native spent and external closure refusal precedes any arm;
            // these synthetic families need not themselves be Validation.
            for seed in [seeds[0], next[0], 42] {
                let config = NumericArenaConfig {
                    schema: SCHEMA.into(),
                    players: 3,
                    seeds: vec![seed],
                    bootstrap_seed: 1,
                    external_exclusions: ExternalExclusions {
                        evidence_sha256: "0".repeat(64),
                        excluded_family_ids: vec![],
                    },
                };
                let error = crate::arena::numeric::run_semantic_playing_snapshot_validation(
                    &config,
                    SemanticPlayingSnapshotArenaPolicy::new(snapshot)?,
                    SemanticPlayingSnapshotReference::ResourceCycleV5,
                    vec![],
                )
                .unwrap_err();
                assert!(error.contains("human/native family closure"));
            }
            Ok(policy.identity().actor_checksum.clone())
        })
        .unwrap();
    assert_eq!(actor, journal.status().current.actor_checksum());
    let callback_error = journal
        .with_evaluation_snapshot::<()>(|_| Err("Synthetic evaluation callback".into()))
        .unwrap_err();
    assert_eq!(callback_error, "Synthetic evaluation callback");
    let source_path = PathBuf::from(&prepared.loaded_semantic_playing_sources().unwrap()[0].path);
    let source_bytes = fs::read(&source_path).unwrap();
    let source_error = journal
        .with_evaluation_snapshot::<()>(|_| {
            fs::write(&source_path, b"changed inside read-only evaluation fixture")
                .map_err(|e| e.to_string())?;
            Err("Synthetic evaluation callback".into())
        })
        .unwrap_err();
    fs::write(&source_path, source_bytes).unwrap();
    assert!(source_error.contains("Synthetic evaluation callback"));
    assert!(source_error.contains("evaluation postcheck"));
    let archive_name = journal.head.name.clone();
    let mut changed_bytes = evaluation_files[&archive_name].clone();
    changed_bytes[0] ^= 1; // Same-length change must be caught by byte hashing.
    let archive_error = journal
        .with_evaluation_snapshot::<()>(|_| {
            fs::write(journal.directory.join(&archive_name), &changed_bytes)
                .map_err(|e| e.to_string())?;
            Err("Synthetic evaluation callback".into())
        })
        .unwrap_err();
    fs::write(
        journal.directory.join(&archive_name),
        &evaluation_files[&archive_name],
    )
    .unwrap();
    assert!(archive_error.contains("Synthetic evaluation callback"));
    assert!(archive_error.contains("Semantic journal pin changed"));
    assert_eq!(
        serde_json::to_value(journal.status()).unwrap(),
        evaluation_status
    );
    assert_eq!(
        serde_json::to_value(journal.head()).unwrap(),
        evaluation_head
    );
    assert_eq!(journal.sequence, evaluation_sequence);
    assert_eq!(journal.captures, evaluation_captures);
    for (name, bytes) in evaluation_files {
        assert_eq!(fs::read(journal.directory.join(name)).unwrap(), bytes);
    }
    journal.verify_archive_bytes().unwrap();
    let final_capture = journal.freeze_snapshot().unwrap();
    // Synthetic external evidence tests the closed schema, not an actual OS
    // observation. No serialized declaration upgrades authentication flags.
    let save = |name: &str, value: &Value| {
        let raw = bounded(value).unwrap();
        let path = root.join(name);
        let mut file = fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&path)
            .unwrap();
        std::io::Write::write_all(&mut file, &raw).unwrap();
        Pin::new(name, &raw).absolute(root)
    };
    let declaration = |name: &str, journal: &SemanticPlayingJournal<'_>, exit_code: i32| {
        let returned = save(
            &format!("{name}-returned.json"),
            &serde_json::to_value(&mixed_close).unwrap(),
        );
        let process = save(
            &format!("{name}-process.json"),
            &json!({
            "schema":PROCESS_CLOSE_SCHEMA,"actualClose":true,"pid":1,"exitCode":exit_code,
            "signal":null,"stdoutEof":true,"stderrEof":true,"allPostchecksPassed":true,
            "observedHead":journal.head(),"returnedClose":returned}),
        );
        save(
            &format!("{name}-declaration.json"),
            &json!({
                "schema":CLOSE_DECLARATION_SCHEMA,"observedHead":journal.head(),
                "processClose":process,"returnedClose":returned,"completeConsumptionDeclared":true,
                "consumption":journal.session.capture_ledger(),"status":journal.status(),
                "inventory":journal.members.values().map(|p|p.absolute(&journal.directory)).collect::<Vec<_>>()
            }),
        )
    };
    let declared = declaration("closed", &journal, 0);
    let restored = restore::restore_using(
        &journal.directory,
        &journal.head(),
        &declared,
        make(),
        &[pool],
        create,
    )
    .unwrap();
    let (mut restored, captures) = restored.into_journal_and_snapshots();
    assert_eq!(
        serde_json::to_value(restored.status()).unwrap(),
        serde_json::to_value(journal.status()).unwrap()
    );
    assert_eq!(
        restored.session.capture_ledger(),
        journal.session.capture_ledger()
    );
    assert_eq!(captures.len(), 2);
    assert_eq!(
        captures[0].journal_identity().unwrap(),
        snapshot.journal_identity().unwrap()
    );
    assert_eq!(
        captures[1].journal_identity().unwrap(),
        final_capture.journal_identity().unwrap()
    );
    let audit = runner::audit_report("0".repeat(64), &declared, &restored, &captures).unwrap();
    let audit = serde_json::to_value(audit).unwrap();
    assert_eq!(
        audit["status"],
        serde_json::to_value(journal.status()).unwrap()
    );
    assert_eq!(
        audit["completeSessionConsumption"],
        journal.session.capture_ledger()
    );
    assert_eq!(audit["captureIds"].as_array().unwrap().len(), 2);
    assert_eq!(audit["newFamilies"], 0);
    assert_eq!(audit["newActorStreams"], 0);
    assert_eq!(audit["newLogicalUpdates"], 0);
    assert_eq!(audit["historicalArchiveBytesVerified"], true);
    assert_eq!(audit["actualProcessCloseVerified"], false);
    assert_eq!(audit["strengthClaimed"], false);
    assert!(
        restore::restore_using(
            &journal.directory,
            &journal.head(),
            &declared,
            make_with_seed(12),
            &[pool],
            create
        )
        .is_err()
    );
    let failed_close = declaration("failed-close", &journal, 1);
    assert!(
        restore::restore_using(
            &journal.directory,
            &journal.head(),
            &failed_close,
            make(),
            &[pool],
            create
        )
        .is_err()
    );
    journal.verify_archive_bytes().unwrap();
    let report_path = root.join("journal/report-0001.json");
    let original_report = fs::read(&report_path).unwrap();
    let mut corrupt = original_report.clone();
    corrupt[0] ^= 1;
    fs::write(&report_path, &corrupt).unwrap();
    journal.verify_files().unwrap(); // Warm outputs are not learning inputs.
    assert!(journal.verify_archive_bytes().is_err());
    assert!(runner::audit_report("0".repeat(64), &declared, &restored, &captures).is_err());
    assert!(
        restore::restore_using(
            &journal.directory,
            &journal.head(),
            &declared,
            make(),
            &[pool],
            create
        )
        .is_err()
    );
    fs::write(&report_path, [original_report.as_slice(), b" "].concat()).unwrap();
    assert!(journal.verify_files().is_err());
    fs::write(&report_path, &original_report).unwrap();
    let extra = root.join("journal/unexpected.json");
    fs::write(&extra, b"{}").unwrap();
    assert!(journal.verify_files().is_err());
    fs::remove_file(extra).unwrap();
    journal.verify_archive_bytes().unwrap();
    // A valid old Closed does not rule out a later attempted Reserve. Even a
    // fully declared/pinned later tip cannot continue without Resolve+Closed.
    let later: [u32; 4] = (600..700)
        .filter(|s| split_for_family(&seed_family_id(*s)).unwrap() == DatasetSplit::Train)
        .take(4)
        .collect::<Vec<_>>()
        .try_into()
        .unwrap();
    let plan = journal
        .session
        .plan_reserved(later, PoolRef::Legacy(pool, None), limits)
        .unwrap();
    let raw = entry(
        journal.sequence + 1,
        Some(&journal.head),
        &json!({
        "kind":"reserve","plan":plan.description(),"fixedPool":pool.procedural_binding().unwrap()}),
    )
    .unwrap();
    let name = format!("entry-{:04}.json", journal.sequence + 1);
    let pin = Pin::new(&name, &raw);
    fs::write(journal.directory.join(&name), raw).unwrap();
    let old_head = std::mem::replace(&mut journal.head, pin.clone());
    journal.members.insert(name.clone(), pin);
    let pending = declaration("pending", &journal, 0);
    assert!(
        restore::restore_using(
            &journal.directory,
            &journal.head(),
            &pending,
            make(),
            &[pool],
            create
        )
        .is_err()
    );
    journal.head = old_head;
    journal.members.remove(&name);
    fs::remove_file(journal.directory.join(name)).unwrap();
    // Exact destination despite publisher Err: Reserve burns but enters no game;
    // Resolve admits the prepared update, retains credit and cannot claim close.
    for label in ["reserve", "resolve"] {
        let mut other = SemanticPlayingJournal::start(
            &root.join(format!("journal-{label}")),
            make(),
            limits,
            budget,
        )
        .unwrap();
        other.fail_after_publish = Some(label);
        let entered = std::cell::Cell::new(false);
        assert!(
            other
                .update_using(seeds, PoolRef::Legacy(pool, None), |config| {
                    entered.set(true);
                    create(config)
                })
                .is_err()
        );
        assert_eq!(other.phase, SemanticJournalPhase::Stopped);
        assert_eq!(
            (other.status().attempts, other.status().spent_actor_streams),
            (1, 14)
        );
        assert_eq!(entered.get(), label == "resolve");
        assert_eq!(
            other.status().current.completed_updates(),
            u64::from(label == "resolve")
        );
        assert!(other.freeze_snapshot().is_err());
    }
    let mut changed =
        SemanticPlayingJournal::start(&root.join("journal-source-post"), make(), limits, budget)
            .unwrap();
    let path = PathBuf::from(&prepared.loaded_semantic_playing_sources().unwrap()[0].path);
    let original = fs::read(&path).unwrap();
    changed.change_source_after_resolve = Some(path.clone());
    assert!(
        changed
            .update_using(seeds, PoolRef::Legacy(pool, None), create)
            .is_err()
    );
    fs::write(path, original).unwrap();
    assert_eq!(changed.phase, SemanticJournalPhase::Stopped);
    assert_eq!(changed.status().current.completed_updates(), 1);
    assert_eq!(
        changed.status().last_resolution,
        Some(SemanticJournalResolution::Updated)
    );
    assert!(changed.status().current.credited_playing_decisions() > 0);
    // A reconstructed typed capture can drive the next warm batch on the same
    // held learner; no parameter import or per-batch process restart is needed.
    let mixture =
        SemanticFrozenMixture::new(pool, vec![SnapshotSlot::new(3, 3, &captures[0]).unwrap()])
            .unwrap();
    let continued = restored
        .update_using(later, PoolRef::Mixture(&mixture), create)
        .unwrap();
    assert_eq!(continued.attempt, 3);
    assert_eq!(restored.status().current.completed_updates(), 3);
    assert_eq!(restored.status().next_ordinal, 212);
    assert_eq!(restored.status().spent_actor_streams, 42);
    restored.verify_archive_bytes().unwrap();
    runner::control_fixture(root, limits, later);
}
