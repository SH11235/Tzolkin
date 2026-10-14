//! Cold reconstruction from a caller's genuinely fresh source-owned session.
//! Every historical record is freshly Core-audited before the same joint SGD
//! preparation/admission path. Neither journal JSON nor the external declaration
//! authenticates a process, creates an origin, imports weights or refunds burns.
use super::*;
use crate::public_policy_update::checkpoint_digest;
use crate::public_stochastic_record::unique_json;
use crate::semantic_playing_rl_pool::SnapshotSlot;

pub const CLOSE_DECLARATION_SCHEMA: &str = "tzolkin-semantic-playing-journal-close-declaration-v1";
pub const PROCESS_CLOSE_SCHEMA: &str = "tzolkin-semantic-playing-journal-observed-process-close-v1";

/// Only successfully replayed, source-owned values can leave reconstruction.
/// Captures retain the actual past actors, not just their serialized identities.
/// The journal's accounting flags remain non-authenticating after restoration.
pub struct SemanticJournalRestoration<'a> {
    journal: SemanticPlayingJournal<'a>,
    snapshots: Vec<FrozenSemanticSnapshot<'a>>,
}
impl<'a> SemanticJournalRestoration<'a> {
    pub fn into_journal_and_snapshots(
        self,
    ) -> (SemanticPlayingJournal<'a>, Vec<FrozenSemanticSnapshot<'a>>) {
        (self.journal, self.snapshots)
    }
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct CloseDeclaration {
    schema: String,
    observed_head: ContentPin,
    process_close: ContentPin,
    returned_close: ContentPin,
    complete_consumption_declared: bool,
    consumption: Value,
    status: Value,
    inventory: Vec<ContentPin>,
}
#[derive(Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct ProcessClose {
    schema: String,
    actual_close: bool,
    pid: u32,
    exit_code: i32,
    signal: Value,
    stdout_eof: bool,
    stderr_eof: bool,
    all_postchecks_passed: bool,
    observed_head: ContentPin,
    returned_close: ContentPin,
}
#[derive(Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct ReturnedClose {
    head: ContentPin,
    report: ContentPin,
    current: String,
    attempt: usize,
    resolution: Resolution,
    library_close_returned: bool,
    actual_process_close_verified: bool,
    historical_archive_bytes_verified: bool,
}
#[derive(Clone, Copy, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
enum Resolution {
    Updated,
    NoChange,
}
impl Resolution {
    fn live(self) -> SemanticJournalResolution {
        match self {
            Self::Updated => SemanticJournalResolution::Updated,
            Self::NoChange => SemanticJournalResolution::NoChange,
        }
    }
}
#[derive(Deserialize, Serialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct Budget {
    max_attempts: usize,
    max_logical_bytes: u64,
    max_stored_bytes: u64,
    max_metadata_bytes: u64,
    max_disk_bytes: u64,
}
#[derive(Deserialize, Serialize)]
#[serde(
    tag = "kind",
    rename_all = "camelCase",
    rename_all_fields = "camelCase",
    deny_unknown_fields
)]
enum Event {
    Init {
        source: Value,
        budget: Budget,
        limits: [usize; 4],
        actual_process_close_verified: bool,
        source_qualified: bool,
        verified_native_policy: bool,
    },
    Reserve {
        plan: Value,
        fixed_pool: Value,
    },
    Resolve {
        attempt: usize,
        resolution: Resolution,
        report: Pin,
        records: [StoredRecord; 4],
        current: String,
        credited_playing: u64,
    },
    Closed {
        attempt: usize,
        resolve: Pin,
        current: String,
        actual_process_close_verified: bool,
    },
    Capture {
        snapshot: Value,
        ledger: Value,
    },
}
#[derive(Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct Entry {
    schema: String,
    sequence: u64,
    previous: Option<Pin>,
    event: Event,
    checksum: String,
}

/// The source session must still be its fresh original root, with the original
/// external/history pins. A later host-wide exclusion overlay is not a substitute
/// for those inputs. The close declaration must bind the exact producing tip,
/// complete consumption and all archive leaves; an earlier Closed is insufficient.
/// No files are published here and no extra update/stream is allocated.
pub fn restore_observed_closed<'a>(
    directory: &Path,
    expected_tip: &ContentPin,
    close_declaration: &ContentPin,
    fresh: SemanticPlayingJointSession<'a>,
    procedural_pools: &[&ConditionalFrozenPool],
) -> Result<SemanticJournalRestoration<'a>, String> {
    restore_using(
        directory,
        expected_tip,
        close_declaration,
        fresh,
        procedural_pools,
        crate::semantic_playing_rl_collection::create,
    )
}

pub(super) fn restore_using<'a>(
    directory: &Path,
    expected_tip: &ContentPin,
    close_pin: &ContentPin,
    fresh: SemanticPlayingJointSession<'a>,
    pools: &[&ConditionalFrozenPool],
    create: impl Fn(&NativeStochasticConfig) -> Result<tzolkin_core::GameState, String>,
) -> Result<SemanticJournalRestoration<'a>, String> {
    let directory = local(directory, false)?;
    if fresh.attempts() != 0 || fresh.stopped() || fresh.identity()?.completed_updates() != 0 {
        return Err("Semantic restore needs a genuinely fresh source root".into());
    }
    let source = fresh.journal_source()?;
    reject_input_namespace(&directory, &source)?;
    let declaration: CloseDeclaration = metadata(close_pin, &directory)?;
    if declaration.schema != CLOSE_DECLARATION_SCHEMA
        || !declaration.complete_consumption_declared
        || !same_pin(&declaration.observed_head, expected_tip)?
    {
        return Err("Semantic exact-tip/full-consumption declaration differs".into());
    }
    let process: ProcessClose = metadata(&declaration.process_close, &directory)?;
    let returned: ReturnedClose = metadata(&declaration.returned_close, &directory)?;
    if process.schema != PROCESS_CLOSE_SCHEMA
        || !process.actual_close
        || process.pid == 0
        || process.exit_code != 0
        || !process.signal.is_null()
        || !process.stdout_eof
        || !process.stderr_eof
        || !process.all_postchecks_passed
        || !same_pin(&process.observed_head, expected_tip)?
        || !same_pin(&process.returned_close, &declaration.returned_close)?
        || !returned.library_close_returned
        || returned.actual_process_close_verified
        || returned.historical_archive_bytes_verified
    {
        return Err("Semantic declared producing-process close is unsuccessful/incomplete".into());
    }
    let bindings = pool_bindings(pools)?;
    let archive = archive(&directory, &declaration.inventory)?;
    let tip_name = direct_name(expected_tip, &directory)?;
    if !same_pin(
        &archive
            .get(&tip_name)
            .ok_or("Semantic producing tip missing")?
            .absolute(&directory),
        expected_tip,
    )? {
        return Err("Semantic expected producing tip is not in the full archive".into());
    }
    let entries = read_entries(&directory, &archive)?;
    if entries.last().is_none_or(|(p, _)| p.name != tip_name) {
        return Err("Semantic observed tip is not the final journal entry".into());
    }
    let (first_pin, first) = entries.first().ok_or("Semantic journal Init missing")?;
    let Event::Init {
        source: saved_source,
        budget,
        limits,
        actual_process_close_verified,
        source_qualified,
        verified_native_policy,
    } = &first.event
    else {
        return Err("Semantic journal must begin with Init".into());
    };
    if *saved_source != source
        || *actual_process_close_verified
        || *source_qualified
        || *verified_native_policy
    {
        return Err("Semantic original source/root/config differs".into());
    }
    let budget = SemanticJournalBudget::new(
        budget.max_attempts,
        budget.max_logical_bytes,
        budget.max_stored_bytes,
        budget.max_metadata_bytes,
        budget.max_disk_bytes,
    )?;
    let limits = CollectionLimits::new(limits[0], limits[1], limits[2], limits[3])?;
    let (stored, metadata) = totals(&archive)?;
    if entries.len() > 4 * budget.max_attempts + 2
        || 2 * first_pin.bytes as u64 > budget.max_disk_bytes
    {
        return Err("Semantic restore entry/Init budget".into());
    }
    if stored > budget.max_stored_bytes
        || metadata > budget.max_metadata_bytes
        || stored + metadata > budget.max_disk_bytes
    {
        return Err("Semantic complete archive exceeds its Init budget".into());
    }
    let mut journal = SemanticPlayingJournal {
        directory,
        session: fresh,
        limits,
        budget,
        phase: SemanticJournalPhase::Closed,
        head: first_pin.clone(),
        sequence: 0,
        members: BTreeMap::from([(first_pin.name.clone(), first_pin.clone())]),
        captures: BTreeSet::new(),
        logical_bytes: 0,
        reservation_confirmed: false,
        resolution: None,
        #[cfg(test)]
        fail_after_publish: None,
        #[cfg(test)]
        change_source_after_resolve: None,
    };
    let mut snapshots = Vec::new();
    let mut latest_close = None;
    let mut index = 1;
    while index < entries.len() {
        let (pin, entry) = &entries[index];
        if journal.phase != SemanticJournalPhase::Closed || journal.session.stopped() {
            return Err("Semantic stopped/pending journal has a later operation".into());
        }
        match &entry.event {
            Event::Capture { snapshot, ledger } => {
                let actual = journal.session.freeze_snapshot()?;
                let id = actual.identity()?.checksum().to_string();
                if actual.journal_identity()? != *snapshot
                    || journal.session.capture_ledger() != *ledger
                    || journal.captures.len() > budget.max_attempts
                    || !journal.captures.insert(id)
                {
                    return Err("Semantic past capture/ledger differs".into());
                }
                snapshots.push(actual);
                add(&mut journal, pin)?;
                index += 1;
            }
            Event::Reserve { plan, fixed_pool } => {
                journal.check_attempt_capacity()?;
                let procedural = fixed_pool.get("procedural").unwrap_or(fixed_pool);
                let pool = pools[*bindings
                    .get(&binding_key(procedural)?)
                    .ok_or("Semantic procedural pool not supplied")?];
                let mixture = if fixed_pool.get("snapshots").is_some() {
                    Some(rebuild_mixture(pool, fixed_pool, &snapshots)?)
                } else {
                    None
                };
                let pool_ref = mixture
                    .as_ref()
                    .map_or(PoolRef::Legacy(pool, None), PoolRef::Mixture);
                if pool_ref.journal_binding()? != *fixed_pool {
                    return Err("Semantic fixed procedural/mixture binding differs".into());
                }
                let members = plan
                    .get("members")
                    .and_then(Value::as_array)
                    .filter(|m| m.len() == 4)
                    .ok_or("Semantic Reserve needs four exact members")?;
                let seeds: [u32; 4] = members
                    .iter()
                    .map(|m| {
                        m.get("environmentSeed")
                            .and_then(Value::as_u64)
                            .and_then(|n| u32::try_from(n).ok())
                            .ok_or_else(|| "Semantic Reserve seed differs".to_string())
                    })
                    .collect::<Result<Vec<_>, _>>()?
                    .try_into()
                    .map_err(|_| "Semantic seed count")?;
                let planned = journal.session.plan_reserved(seeds, pool_ref, limits)?;
                if planned.description() != plan {
                    return Err("Semantic exact Reserve parent/tuple/config/limits differs".into());
                }
                add(&mut journal, pin)?;
                let batch = journal.session.burn_reserved(planned);
                journal.phase = SemanticJournalPhase::Reserved;
                journal.reservation_confirmed = true;
                let (resolve_pin, resolve) =
                    entries.get(index + 1).ok_or("Semantic pending Reserve")?;
                let Event::Resolve {
                    attempt,
                    resolution,
                    report,
                    records,
                    current,
                    credited_playing,
                } = &resolve.event
                else {
                    return Err("Semantic Reserve is not successfully resolved".into());
                };
                if *attempt != journal.session.attempts()
                    || report.name != format!("report-{attempt:04}.json")
                {
                    return Err("Semantic resolved attempt/report differs".into());
                }
                let mut raw = Vec::with_capacity(4);
                let mut raw_total = 0u64;
                for (i, record) in records.iter().enumerate() {
                    if record.stored.name != format!("record-{attempt:04}-{i}.json.gz")
                        || record.raw_bytes > limits.max_source_bytes() + limits.max_trace_bytes()
                    {
                        return Err("Semantic saved record member/length differs".into());
                    }
                    require_member(&archive, &record.stored)?;
                    let stored = archive_raw(&journal.directory, &record.stored)?;
                    raw.push(storage::decode(
                        &stored,
                        record.raw_bytes,
                        &record.raw_sha256,
                    )?);
                    raw_total += record.raw_bytes as u64;
                    add(&mut journal, &record.stored)?;
                }
                let raw: [Vec<u8>; 4] =
                    raw.try_into().map_err(|_| "Semantic saved record count")?;
                let prepared = journal
                    .session
                    .stage_saved_records_using(&batch, pool_ref, &raw, &create)?;
                let report_raw = bounded(prepared.report())?;
                require_member(&archive, report)?;
                if report_raw != archive_raw(&journal.directory, report)?
                    || prepared.report().next_identity_checksum() != current
                    || prepared.report().credited_playing_decisions() != *credited_playing
                    || prepared.report().updated() != (*resolution == Resolution::Updated)
                    || journal.logical_bytes + raw_total > budget.max_logical_bytes
                {
                    return Err(
                        "Semantic freshly prepared numeric report/identity/credit differs".into(),
                    );
                }
                add(&mut journal, report)?;
                add(&mut journal, resolve_pin)?;
                let (closed_pin, closed) = entries
                    .get(index + 2)
                    .ok_or("Semantic Resolve lacks library close")?;
                let Event::Closed {
                    attempt: closed_attempt,
                    resolve,
                    current: closed_current,
                    actual_process_close_verified,
                } = &closed.event
                else {
                    return Err("Semantic Resolve lacks exact Closed".into());
                };
                if *closed_attempt != *attempt
                    || resolve != resolve_pin
                    || closed_current != current
                    || *actual_process_close_verified
                {
                    return Err("Semantic library close binding differs".into());
                }
                journal.session.journal_recheck()?;
                pool_ref.recheck()?;
                journal.session.commit_prepared(prepared);
                journal.logical_bytes += raw_total;
                journal.resolution = Some(resolution.live());
                add(&mut journal, closed_pin)?;
                journal.phase = if journal.session.stopped() {
                    SemanticJournalPhase::Stopped
                } else {
                    SemanticJournalPhase::Closed
                };
                latest_close = Some((closed_pin, report, *attempt, *resolution, current));
                index += 3;
            }
            _ => return Err("Semantic unexpected Init/Resolve/Closed operation".into()),
        }
    }
    let (closed, report, attempt, resolution, current) =
        latest_close.ok_or("Semantic no returned library close")?;
    if !same_pin(&returned.head, &closed.absolute(&journal.directory))?
        || !same_pin(&returned.report, &report.absolute(&journal.directory))?
        || returned.attempt != attempt
        || returned.resolution != resolution
        || returned.current != *current
        || declaration.consumption != journal.session.capture_ledger()
        || declaration.status
            != serde_json::to_value(journal.status()).map_err(|e| e.to_string())?
        || journal.members != archive
    {
        return Err("Semantic terminal close/status/full consumption/archive differs".into());
    }
    journal.verify_archive_bytes()?;
    for pin in [
        close_pin,
        &declaration.process_close,
        &declaration.returned_close,
    ] {
        external_raw(pin, &journal.directory)?;
    }
    journal.session.journal_recheck()?;
    if pool_bindings(pools)? != bindings {
        return Err("Semantic supplied pools changed".into());
    }
    Ok(SemanticJournalRestoration { journal, snapshots })
}

fn binding_key(value: &Value) -> Result<String, String> {
    serde_json::to_string(value).map_err(|e| e.to_string())
}
fn pool_bindings(pools: &[&ConditionalFrozenPool]) -> Result<BTreeMap<String, usize>, String> {
    if pools.is_empty() || pools.len() > 64 {
        return Err("Semantic supplied pool cap".into());
    }
    let mut bindings = BTreeMap::new();
    for (i, pool) in pools.iter().enumerate() {
        if bindings
            .insert(binding_key(&pool.procedural_binding()?)?, i)
            .is_some()
        {
            return Err("Semantic duplicate supplied pool".into());
        }
    }
    Ok(bindings)
}
fn rebuild_mixture<'b, 'a: 'b>(
    pool: &'b ConditionalFrozenPool,
    saved: &Value,
    snapshots: &'b [FrozenSemanticSnapshot<'a>],
) -> Result<SemanticFrozenMixture<'b>, String> {
    let slots = saved
        .get("snapshots")
        .and_then(Value::as_array)
        .filter(|s| !s.is_empty() && s.len() <= 10)
        .ok_or("Semantic snapshot slot cap")?;
    let mut actual = Vec::with_capacity(slots.len());
    for slot in slots {
        let id = slot
            .get("snapshot")
            .and_then(Value::as_str)
            .ok_or("Semantic capture ID missing")?;
        let snapshot = snapshots
            .iter()
            .find(|s| s.binding() == id)
            .ok_or("Semantic capture not replayed at its event")?;
        let member = slot
            .get("member")
            .and_then(Value::as_u64)
            .and_then(|n| usize::try_from(n).ok())
            .ok_or("Semantic capture member")?;
        let actor = slot
            .get("actor")
            .and_then(Value::as_u64)
            .and_then(|n| usize::try_from(n).ok())
            .ok_or("Semantic capture actor")?;
        actual.push(SnapshotSlot::new(member, actor, snapshot)?);
    }
    let mixture = SemanticFrozenMixture::new(pool, actual)?;
    if mixture.journal_binding()? != *saved {
        return Err("Semantic reconstructed capture roster differs".into());
    }
    Ok(mixture)
}
fn add(journal: &mut SemanticPlayingJournal<'_>, pin: &Pin) -> Result<(), String> {
    if journal
        .members
        .insert(pin.name.clone(), pin.clone())
        .is_some()
    {
        return Err("Semantic journal duplicate reference".into());
    }
    let (stored, metadata) = journal.usage()?;
    journal.check_totals(stored, metadata)?;
    if stored + metadata + pin.bytes as u64 > journal.budget.max_disk_bytes {
        return Err("Semantic replay publication temporary bound".into());
    }
    if pin.name.starts_with("entry-") {
        journal.head = pin.clone();
        journal.sequence += 1;
    }
    Ok(())
}
fn require_member(archive: &BTreeMap<String, Pin>, pin: &Pin) -> Result<(), String> {
    if archive.get(&pin.name) != Some(pin) {
        return Err("Semantic referenced file pin differs".into());
    }
    Ok(())
}
fn same_pin(a: &ContentPin, b: &ContentPin) -> Result<bool, String> {
    if !Path::new(&a.path).is_absolute()
        || !Path::new(&b.path).is_absolute()
        || a.bytes == 0
        || b.bytes == 0
        || !checkpoint_digest(&a.sha256)
        || !checkpoint_digest(&b.sha256)
    {
        return Err("Semantic absolute nonempty pin differs".into());
    }
    Ok(a.bytes == b.bytes
        && a.sha256 == b.sha256
        && local(Path::new(&a.path), false)? == local(Path::new(&b.path), false)?)
}
fn direct_name(pin: &ContentPin, directory: &Path) -> Result<String, String> {
    let path = local(Path::new(&pin.path), false)?;
    if !Path::new(&pin.path).is_absolute() || path.parent() != Some(directory) {
        return Err("Semantic archive pin is not an absolute direct leaf".into());
    }
    path.file_name()
        .and_then(|s| s.to_str())
        .map(str::to_string)
        .ok_or("Semantic non-UTF8 archive pin".into())
}
fn external_raw(pin: &ContentPin, directory: &Path) -> Result<Vec<u8>, String> {
    let path = local(Path::new(&pin.path), false)?;
    if !Path::new(&pin.path).is_absolute()
        || path.starts_with(directory)
        || pin.bytes == 0
        || pin.bytes > META_BYTES as u64
        || !checkpoint_digest(&pin.sha256)
    {
        return Err("Semantic external declaration pin namespace/size differs".into());
    }
    let raw = read_bounded(&path, pin.bytes as usize)?;
    if raw.len() as u64 != pin.bytes || crate::model::digest(&raw) != pin.sha256 {
        return Err("Semantic external declaration pin changed".into());
    }
    Ok(raw)
}
fn metadata<T: serde::de::DeserializeOwned>(
    pin: &ContentPin,
    directory: &Path,
) -> Result<T, String> {
    serde_json::from_value(unique_json(&external_raw(pin, directory)?, META_BYTES)?)
        .map_err(|e| e.to_string())
}
fn archive(directory: &Path, inventory: &[ContentPin]) -> Result<BTreeMap<String, Pin>, String> {
    if inventory.is_empty() || inventory.len() > MAX_FILES {
        return Err("Semantic inventory cap".into());
    }
    let mut expected = BTreeMap::new();
    for pin in inventory {
        let name = direct_name(pin, directory)?;
        let size = usize::try_from(pin.bytes).map_err(|_| "Semantic archive length overflow")?;
        if size == 0
            || !checkpoint_digest(&pin.sha256)
            || !valid_name(&name)
            || size
                > if name.ends_with(".gz") {
                    storage::MAX_STORED_RECORD_BYTES
                } else {
                    META_BYTES
                }
            || expected
                .insert(
                    name.clone(),
                    Pin {
                        name,
                        bytes: size,
                        sha256: pin.sha256.clone(),
                    },
                )
                .is_some()
        {
            return Err("Semantic inventory name/length/digest/uniqueness differs".into());
        }
    }
    let (stored, metadata) = totals(&expected)?;
    if stored > (64u64 << 30) || metadata > (1u64 << 30) || stored + metadata > (66u64 << 30) {
        return Err("Semantic complete archive absolute byte cap".into());
    }
    let mut actual = BTreeSet::new();
    for item in fs::read_dir(directory).map_err(|e| e.to_string())? {
        let item = item.map_err(|e| e.to_string())?;
        let name = item
            .file_name()
            .into_string()
            .map_err(|_| "Semantic archive non-UTF8 leaf")?;
        let pin = expected
            .get(&name)
            .ok_or("Semantic archive has undeclared leaf")?;
        let path = local(&item.path(), false)?;
        let meta = fs::symlink_metadata(&path).map_err(|e| e.to_string())?;
        if !meta.is_file() || meta.len() != pin.bytes as u64 || !actual.insert(name.clone()) {
            return Err("Semantic archive leaf type/length differs".into());
        }
        archive_raw(directory, pin)?;
    }
    if actual.len() != expected.len() {
        return Err("Semantic archive member missing".into());
    }
    Ok(expected)
}
fn valid_name(name: &str) -> bool {
    let number = |s: &str, max: usize| {
        s.len() == 4
            && s.bytes().all(|c| c.is_ascii_digit())
            && s.parse::<usize>().is_ok_and(|n| n <= max)
    };
    if let Some(s) = name
        .strip_prefix("entry-")
        .and_then(|s| s.strip_suffix(".json"))
    {
        return number(s, 257);
    }
    if let Some(s) = name
        .strip_prefix("report-")
        .and_then(|s| s.strip_suffix(".json"))
    {
        return number(s, 64) && s != "0000";
    }
    name.strip_prefix("record-")
        .and_then(|s| s.strip_suffix(".json.gz"))
        .and_then(|s| s.split_once('-'))
        .is_some_and(|(n, m)| number(n, 64) && n != "0000" && matches!(m, "0" | "1" | "2" | "3"))
}
fn read_entries(
    directory: &Path,
    archive: &BTreeMap<String, Pin>,
) -> Result<Vec<(Pin, Entry)>, String> {
    let mut entries = Vec::new();
    let mut previous = None;
    for pin in archive.values().filter(|p| p.name.starts_with("entry-")) {
        let raw = archive_raw(directory, pin)?;
        let entry: Entry =
            serde_json::from_value(unique_json(&raw, META_BYTES)?).map_err(|e| e.to_string())?;
        let sequence = entries.len() as u64;
        if entry.schema != JOURNAL_SCHEMA
            || entry.sequence != sequence
            || entry.previous != previous
            || pin.name != format!("entry-{sequence:04}.json")
            || !checkpoint_digest(&entry.checksum)
            || super::entry(
                sequence,
                previous.as_ref(),
                &serde_json::to_value(&entry.event).map_err(|e| e.to_string())?,
            )? != raw
        {
            return Err("Semantic canonical journal sequence/chain/checksum differs".into());
        }
        previous = Some(pin.clone());
        entries.push((pin.clone(), entry));
    }
    Ok(entries)
}
fn archive_raw(directory: &Path, pin: &Pin) -> Result<Vec<u8>, String> {
    let path = directory.join(&pin.name);
    let raw = if pin.name.ends_with(".gz") {
        read_mixed_stored_record(&path, pin.bytes)?
    } else {
        read_bounded(&path, pin.bytes)?
    };
    if raw.len() != pin.bytes || crate::model::digest(&raw) != pin.sha256 {
        return Err("Semantic full archive bytes differ".into());
    }
    Ok(raw)
}
