//! A finite concrete loader/loop over the journal, not another learner or host.
//! Original inputs never change on resume. A newer declared global PRIOR checks
//! only prospective work; the separately pinned full allocation is host-consumed
//! even when its later batches never enter. Neither metadata authenticates an OS.
use super::*;
use crate::conditional_human_rl_update::ConditionalNativeExclusions;
use crate::conditional_semantic_playing_rl::SemanticPlayingRlOwner;
use crate::dataset::{DatasetSplit, seed_family_id, split_for_family};
use crate::human_policy_training::PreparedConditionalHumanPolicy;
use crate::public_policy_baseline::ResidualFitConfig;
use crate::public_policy_update::{OneStepConfig, checkpoint_digest};
use crate::public_stochastic::{SamplingSeed, SamplingStreamIdentity};
use crate::public_stochastic_record::{Config, unique_json};
use crate::semantic_playing_rl_collection::{DeclaredSamplingHistory, DeclaredSamplingRange};
use crate::semantic_playing_rl_pool::SnapshotSlot;

const RUN_SCHEMA: &str = "tzolkin-semantic-playing-journal-run-v1";
const ALLOCATION_SCHEMA: &str = "tzolkin-semantic-playing-invocation-allocation-v1";
const MAX_CONTROL: usize = 1024 * 1024;

#[derive(Deserialize, Serialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct External {
    pin: ContentPin,
    count: usize,
    canonical_digest: String,
}
impl External {
    fn load(&self) -> Result<ConditionalNativeExclusions, String> {
        ConditionalNativeExclusions::load(
            Path::new(&self.pin.path),
            &self.pin,
            self.count,
            &self.canonical_digest,
        )
    }
}
#[derive(Deserialize, Serialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct History {
    // sampling seed, replicate, players, absolute actor, first, next; all exact.
    ranges: Vec<[u64; 6]>,
    pins: Vec<ContentPin>,
}
impl History {
    fn load(&self) -> Result<DeclaredSamplingHistory, String> {
        if self.pins.len() > 64 || self.ranges.len() > 8192 {
            return Err("Runner history cap".into());
        }
        let ranges = self
            .ranges
            .iter()
            .map(|r| {
                DeclaredSamplingRange::new(
                    SamplingSeed::new(r[0]),
                    r[1],
                    usize::try_from(r[2]).map_err(|_| "History players")?,
                    usize::try_from(r[3]).map_err(|_| "History actor")?,
                    r[4],
                    r[5],
                )
            })
            .collect::<Result<Vec<_>, String>>()?;
        DeclaredSamplingHistory::declare_complete(ranges, self.pins.clone())
    }
}
#[derive(Deserialize, Serialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct Original {
    checkpoint: ContentPin,
    collection: ContentPin,
    external: External,
    history: History,
    critic_seed: u64,
    sampling_seed: u64,
    replicate: u64,
    first_ordinal: u64,
    actor_config: Value,
    critic_config: Value,
    limits: [usize; 4],
    // attempts, logical, stored, metadata, disk; exactly the original budget.
    budget: [u64; 5],
}
impl Original {
    fn limits(&self) -> Result<CollectionLimits, String> {
        CollectionLimits::new(
            self.limits[0],
            self.limits[1],
            self.limits[2],
            self.limits[3],
        )
    }
    fn budget(&self) -> Result<SemanticJournalBudget, String> {
        SemanticJournalBudget::new(
            usize::try_from(self.budget[0]).map_err(|_| "Attempt cap")?,
            self.budget[1],
            self.budget[2],
            self.budget[3],
            self.budget[4],
        )
    }
}
#[derive(Deserialize, Serialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct Prior {
    external: External,
    history: History,
    // Unknown historical replicates are not filled with zero: forbid the seed.
    forbidden_sampling_seeds: Vec<u64>,
}
#[derive(Deserialize, Serialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct Restore {
    tip: ContentPin,
    close_declaration: ContentPin,
}
#[derive(Deserialize, Serialize)]
#[serde(tag = "kind", rename_all = "camelCase", deny_unknown_fields)]
enum SnapshotRef {
    Existing { checksum: String },
    Local { alias: String },
}
#[derive(Deserialize, Serialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct Slot {
    member: usize,
    actor: usize,
    snapshot: SnapshotRef,
}
#[derive(Deserialize, Serialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct Batch {
    seeds: [u32; 4],
    // Capture the genuine current actor at this closed boundary, once.
    capture_before: Option<String>,
    slots: Vec<Slot>,
}
#[derive(Deserialize, Serialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct Wire {
    schema: String,
    mode: String,
    journal_directory: String,
    original: Original,
    procedural_pool: ContentPin,
    restore: Option<Restore>,
    global_prior: Prior,
    permanent_allocation: ContentPin,
    batches: Vec<Batch>,
}

/// Closed pinned control; no learner, snapshot or raw parameter constructor.
pub struct SemanticJournalRunSpec {
    wire: Wire,
    checksum: String,
}
impl SemanticJournalRunSpec {
    pub fn from_json(raw: &[u8], expected_sha256: &str) -> Result<Self, String> {
        if !checkpoint_digest(expected_sha256) || crate::model::digest(raw) != expected_sha256 {
            return Err("Semantic run control pin differs".into());
        }
        let wire: Wire =
            serde_json::from_value(unique_json(raw, MAX_CONTROL)?).map_err(|e| e.to_string())?;
        if wire.schema != RUN_SCHEMA
            || !matches!(wire.mode.as_str(), "start" | "resume")
            || (wire.mode == "resume") != wire.restore.is_some()
            || wire.batches.is_empty()
            || wire.batches.len() > 64
            || wire.global_prior.forbidden_sampling_seeds.len() > 8192
            || wire
                .global_prior
                .forbidden_sampling_seeds
                .windows(2)
                .any(|p| p[0] >= p[1])
        {
            return Err("Semantic finite run mode/schedule/prior differs".into());
        }
        wire.original.limits()?;
        wire.original.budget()?;
        OneStepConfig::from_checkpoint(wire.original.actor_config.clone())?;
        ResidualFitConfig::from_checkpoint(wire.original.critic_config.clone())?;
        let mut aliases = BTreeSet::new();
        for batch in &wire.batches {
            if batch.slots.len() > 10
                || batch
                    .slots
                    .windows(2)
                    .any(|p| (p[0].member, p[0].actor) >= (p[1].member, p[1].actor))
            {
                return Err("Semantic sorted unique roster slots differ".into());
            }
            if let Some(alias) = &batch.capture_before
                && (!valid_alias(alias) || !aliases.insert(alias.clone()))
            {
                return Err("Semantic capture alias differs/repeats".into());
            }
            for slot in &batch.slots {
                if slot.member >= 4 || slot.actor >= if slot.member < 2 { 3 } else { 4 } {
                    return Err("Semantic fixed roster member/seat differs".into());
                }
                match &slot.snapshot {
                    SnapshotRef::Existing { checksum } if checkpoint_digest(checksum) => {}
                    SnapshotRef::Local { alias } if aliases.contains(alias) => {}
                    _ => return Err("Semantic snapshot reference is unknown or forward".into()),
                }
            }
        }
        Ok(Self {
            wire,
            checksum: expected_sha256.into(),
        })
    }
    pub fn command(&self) -> &str {
        &self.wire.mode
    }
}

/// Complete descriptive return, including on errors after the journal was opened.
/// Serialized consumption and a successful library return do not observe an OS.
#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct SemanticJournalInvocation {
    schema: &'static str,
    control_sha256: String,
    disposition: String,
    error: Option<String>,
    head: ContentPin,
    status: SemanticJournalStatus,
    complete_session_consumption: Value,
    archive_inventory: Vec<ContentPin>,
    capture_aliases: BTreeMap<String, String>,
    capture_identities: Vec<Value>,
    returned_closes: Vec<ContentPin>,
    last_library_return: Option<SemanticJournalReturnedClose>,
    last_returned_close: Option<ContentPin>,
    declared_permanent_allocation: ContentPin,
    declared_families: usize,
    declared_actor_streams: usize,
    historical_archive_bytes_checked_at_return: bool,
    actual_process_close_verified: bool,
    source_authenticated: bool,
    source_qualified: bool,
    verified_native_policy: bool,
}
impl SemanticJournalInvocation {
    pub fn exit_code(&self) -> i32 {
        match self.disposition.as_str() {
            "completed" => 0,
            "stoppedNoChange" => 2,
            _ => 1,
        }
    }
}

/// Loads only the declared original source through its genuine Human4 loader.
/// Resume replays before NEW work is admitted. A full fixed allocation is checked
/// first, but is not confused with the subset actually burned by this session.
pub fn run_semantic_playing_journal(
    spec: SemanticJournalRunSpec,
    output: &Path,
) -> Result<SemanticJournalInvocation, String> {
    let w = &spec.wire;
    let directory = local(Path::new(&w.journal_directory), w.mode == "start")?;
    let output = local(output, true)?;
    if output == directory
        || output.starts_with(&directory)
        || directory.starts_with(&output)
        || output.to_string_lossy().len() > 1024
        || directory.to_string_lossy().len() > 1024
        || fs::symlink_metadata(&output).is_ok()
    {
        return Err("Semantic invocation needs a separate fresh output namespace".into());
    }
    if w.mode == "start" && fs::symlink_metadata(&directory).is_ok()
        || w.mode == "resume"
            && !fs::metadata(&directory)
                .map_err(|e| e.to_string())?
                .is_dir()
        || !output.parent().is_some_and(Path::is_dir)
    {
        return Err("Semantic journal/invocation directory state differs".into());
    }
    let source = serde_json::to_value(w).map_err(|e| e.to_string())?;
    reject_input_namespace(&output, &source)?;
    reject_input_namespace(
        &directory,
        &json!({"original":w.original,
        "pool":w.procedural_pool,"allocation":w.permanent_allocation,
        "closeDeclaration":w.restore.as_ref().map(|r|&r.close_declaration)}),
    )?;
    let original_external = w.original.external.load()?;
    let original_history = w.original.history.load()?;
    let prior = w.global_prior.external.load()?;
    let prior_history = w.global_prior.history.load()?;
    check_original_prior(w, &prior)?;
    let pool_raw = pinned(&w.procedural_pool, MAX_CONTROL)?;
    let pool = ConditionalFrozenPool::from_json(&pool_raw)?;
    pool.procedural_binding()?; // Explicitly refuse neural/search legacy slots.
    let allocation_raw = pinned(&w.permanent_allocation, MAX_CONTROL)?;
    // This is a plan checksum, not a control-pin cycle or an authentication claim.
    let plan_checksum = plan_checksum(w)?;
    let limits = w.original.limits()?;
    let expected_first = match &w.restore {
        None => w.original.first_ordinal,
        Some(_) => allocation_first(&allocation_raw)?,
    };
    let expected = allocation(w, &plan_checksum, expected_first, limits)?;
    if unique_json(&allocation_raw, MAX_CONTROL)? != expected {
        return Err("Semantic complete permanent-allocation declaration differs".into());
    }
    check_new(w, expected_first, &prior, &prior_history, limits)?;
    check_new(
        w,
        expected_first,
        &original_external,
        &original_history,
        limits,
    )?;
    // All control/allocation/prior checks above precede model loading or Game.
    pinned(&w.original.checkpoint, 32 * 1024 * 1024)?;
    pinned(&w.original.collection, 16 * 1024 * 1024)?;
    let prepared = PreparedConditionalHumanPolicy::load_semantic_playing(
        Path::new(&w.original.checkpoint.path),
        Path::new(&w.original.collection.path),
    )?;
    pinned(&w.original.checkpoint, 32 * 1024 * 1024)?;
    pinned(&w.original.collection, 16 * 1024 * 1024)?;
    let fresh = SemanticPlayingJointSession::from_fresh(
        SemanticPlayingRlOwner::from_prepared(&prepared, w.original.critic_seed)?,
        &original_external,
        &original_history,
        SamplingSeed::new(w.original.sampling_seed),
        w.original.first_ordinal,
        w.original.replicate,
        OneStepConfig::from_checkpoint(w.original.actor_config.clone())?,
        ResidualFitConfig::from_checkpoint(w.original.critic_config.clone())?,
    )?;
    let (mut journal, mut snapshots) = if let Some(restore) = &w.restore {
        restore_observed_closed(
            &directory,
            &restore.tip,
            &restore.close_declaration,
            fresh,
            &[&pool],
        )?
        .into_journal_and_snapshots()
    } else {
        (
            SemanticPlayingJournal::start(&directory, fresh, limits, w.original.budget()?)?,
            Vec::new(),
        )
    };
    if journal.limits != limits
        || bounded(&journal.budget)? != bounded(&w.original.budget()?)?
        || journal.session.next_episode_ordinal() != expected_first
    {
        return Err("Semantic restored original limits/budget/next cursor differ".into());
    }
    check_prior(
        &journal,
        &prior,
        &prior_history,
        &w.global_prior.forbidden_sampling_seeds,
    )?;
    if journal.session.attempts() + w.batches.len() > journal.budget.max_attempts {
        return Err("Semantic complete invocation exceeds original attempt headroom".into());
    }
    // Missing restored capture IDs and learner-seat replacements refuse before a new Reserve.
    let rosters = pool.freeze_procedural()?;
    for slot in w.batches.iter().flat_map(|b| &b.slots) {
        if slot.actor == rosters[slot.member].learner_seat() {
            return Err("Snapshot cannot replace learner".into());
        }
        if let SnapshotRef::Existing { checksum } = &slot.snapshot
            && !snapshots.iter().any(|s| s.binding() == checksum)
        {
            return Err("Requested capture was not actually restored".into());
        }
    }
    fs::create_dir(&output).map_err(|e| e.to_string())?;
    let mut aliases = BTreeMap::new();
    let mut returns = Vec::new();
    let mut last_library_return = None;
    let mut last_returned_pin = None;
    // Lookup metadata is cached once from genuine captures. Warm alias lookup
    // must not hash every old snapshot's complete origin closure per batch.
    let mut capture_identities = snapshots
        .iter()
        .map(|s| s.journal_identity())
        .collect::<Result<Vec<_>, _>>()?;
    let result = (|| {
        for batch in &w.batches {
            recheck_inputs(w, &prior, &prior_history)?;
            if let Some(alias) = &batch.capture_before {
                let current = journal.session.current_unchecked().checksum();
                let existing = capture_identities
                    .iter()
                    .find(|s| s["current"].as_str() == Some(current));
                let id = match existing {
                    Some(identity) => identity["checksum"]
                        .as_str()
                        .ok_or("Typed capture checksum")?
                        .to_string(),
                    None => {
                        let snapshot = journal.freeze_snapshot()?;
                        let identity = snapshot.journal_identity()?;
                        let id = snapshot.binding().to_string();
                        snapshots.push(snapshot);
                        capture_identities.push(identity);
                        id
                    }
                };
                aliases.insert(alias.clone(), id);
            }
            let close = if batch.slots.is_empty() {
                journal.update_procedural_whole_four(batch.seeds, &pool)?
            } else {
                let slots = batch
                    .slots
                    .iter()
                    .map(|slot| {
                        let id = match &slot.snapshot {
                            SnapshotRef::Existing { checksum } => checksum,
                            SnapshotRef::Local { alias } => {
                                aliases.get(alias).ok_or("Capture alias not produced")?
                            }
                        };
                        let snapshot = snapshots
                            .iter()
                            .find(|s| s.binding() == id)
                            .ok_or("Actual capture missing")?;
                        SnapshotSlot::new(slot.member, slot.actor, snapshot)
                    })
                    .collect::<Result<Vec<_>, String>>()?;
                let mixture = SemanticFrozenMixture::new(&pool, slots)?;
                journal.update_mixed_whole_four(batch.seeds, &mixture)?
            };
            let no_change = close.resolution == SemanticJournalResolution::NoChange;
            let name = format!("returned-{:04}.json", close.attempt);
            last_library_return = Some(close);
            last_returned_pin = None;
            let raw = bounded(&last_library_return)?;
            publish_mixed_metadata(&output, &name, &raw)?;
            let pin = Pin::new(&name, &raw).absolute(&output);
            last_returned_pin = Some(pin.clone());
            returns.push(pin);
            // Source/global-prior/output failures stop before another attempt.
            recheck_inputs(w, &prior, &prior_history)?;
            if no_change {
                return Ok(false);
            }
        }
        Ok(true)
    })();
    let mut error = result.as_ref().err().cloned();
    let full_bytes = journal.verify_archive_bytes();
    if let Err(post) = &full_bytes {
        error = Some(format!(
            "{}; final archive: {post}",
            error.unwrap_or_default()
        ));
    }
    if let Err(post) = journal
        .session
        .journal_recheck()
        .and_then(|_| recheck_inputs(w, &prior, &prior_history))
    {
        error = Some(format!(
            "{}; final source/prior: {post}",
            error.unwrap_or_default()
        ));
    }
    let disposition = if error.is_some() {
        "refused"
    } else if result == Ok(true) {
        "completed"
    } else {
        "stoppedNoChange"
    };
    let report = SemanticJournalInvocation {
        schema: "tzolkin-semantic-playing-journal-invocation-v1",
        control_sha256: spec.checksum,
        disposition: disposition.into(),
        error,
        head: journal.head(),
        status: journal.status(),
        complete_session_consumption: journal.session.capture_ledger(),
        archive_inventory: journal
            .members
            .values()
            .map(|p| p.absolute(&directory))
            .collect(),
        capture_aliases: aliases,
        capture_identities,
        last_returned_close: last_returned_pin,
        last_library_return,
        returned_closes: returns,
        declared_permanent_allocation: w.permanent_allocation.clone(),
        declared_families: 4 * w.batches.len(),
        declared_actor_streams: 14 * w.batches.len(),
        historical_archive_bytes_checked_at_return: full_bytes.is_ok(),
        actual_process_close_verified: false,
        source_authenticated: false,
        source_qualified: false,
        verified_native_policy: false,
    };
    let raw = bounded(&report)?;
    // On a late IO failure the CLI still receives exact accounting to print.
    if let Err(error) = publish_mixed_metadata(&output, "invocation.json", &raw) {
        return Ok(SemanticJournalInvocation {
            disposition: "refused".into(),
            error: Some(format!("Invocation publication: {error}")),
            ..report
        });
    }
    if let Err(error) = journal
        .session
        .journal_recheck()
        .and_then(|_| recheck_inputs(w, &prior, &prior_history))
    {
        return Ok(SemanticJournalInvocation {
            disposition: "refused".into(),
            error: Some(format!("Invocation postpublication: {error}")),
            ..report
        });
    }
    Ok(report)
}

fn valid_alias(alias: &str) -> bool {
    !alias.is_empty()
        && alias.len() <= 32
        && alias
            .bytes()
            .all(|c| c.is_ascii_alphanumeric() || c == b'-')
}
fn pinned(pin: &ContentPin, cap: usize) -> Result<Vec<u8>, String> {
    if !Path::new(&pin.path).is_absolute()
        || pin.bytes == 0
        || pin.bytes > cap as u64
        || !checkpoint_digest(&pin.sha256)
    {
        return Err("Semantic runner bounded absolute pin differs".into());
    }
    let raw = read_bounded(&local(Path::new(&pin.path), false)?, pin.bytes as usize)?;
    if raw.len() as u64 != pin.bytes || crate::model::digest(&raw) != pin.sha256 {
        return Err("Semantic runner input changed".into());
    }
    Ok(raw)
}
fn plan_checksum(w: &Wire) -> Result<String, String> {
    hash(
        b"semantic-playing-invocation-plan-v1\0",
        &json!({"mode":w.mode,"journalDirectory":w.journal_directory,
        "original":w.original,"proceduralPool":w.procedural_pool,"restore":w.restore,"globalPrior":w.global_prior,"batches":w.batches}),
    )
}
fn allocation_first(raw: &[u8]) -> Result<u64, String> {
    unique_json(raw, MAX_CONTROL)?
        .get("firstOrdinal")
        .and_then(Value::as_u64)
        .ok_or("Semantic allocation first ordinal".into())
}
fn allocation(w: &Wire, plan: &str, first: u64, limits: CollectionLimits) -> Result<Value, String> {
    let next = first
        .checked_add(4 * w.batches.len() as u64)
        .ok_or("Semantic invocation ordinal overflow")?;
    let mut members = Vec::new();
    let mut families = BTreeSet::new();
    let mut streams = BTreeSet::new();
    for (batch, b) in w.batches.iter().enumerate() {
        for (member, &seed) in b.seeds.iter().enumerate() {
            let family = seed_family_id(seed);
            if split_for_family(&family)? != DatasetSplit::Train || !families.insert(family.clone())
            {
                return Err("Semantic allocation needs distinct Train families".into());
            }
            let players = if member < 2 { 3 } else { 4 };
            let ordinal = first + 4 * batch as u64 + member as u64;
            let identity = SamplingStreamIdentity::new(
                SamplingSeed::new(w.original.sampling_seed),
                ordinal,
                w.original.replicate,
                players,
            )?;
            let config = NativeStochasticConfig::new(players, seed, identity, limits)?;
            for actor in 0..players {
                streams.insert([
                    w.original.sampling_seed,
                    ordinal,
                    w.original.replicate,
                    players as u64,
                    actor as u64,
                ]);
            }
            members.push(json!({"batch":batch,"member":member,"family":family,"config":Config::from_config(&config)}));
        }
    }
    Ok(
        json!({"schema":ALLOCATION_SCHEMA,"planChecksum":plan,"globalPrior":w.global_prior,
        "priorHead":w.restore.as_ref().map(|r|&r.tip),"permanent":true,"allUnenteredRemainSpent":true,
        "samplingSeed":w.original.sampling_seed,"replicate":w.original.replicate,"firstOrdinal":first,"nextOrdinal":next,
        "families":families,"actorStreams":streams,"members":members}),
    )
}
fn check_new(
    w: &Wire,
    first: u64,
    prior: &ConditionalNativeExclusions,
    history: &DeclaredSamplingHistory,
    limits: CollectionLimits,
) -> Result<(), String> {
    if w.global_prior
        .forbidden_sampling_seeds
        .binary_search(&w.original.sampling_seed)
        .is_ok()
    {
        return Err("Prospective sampling seed is wholly forbidden".into());
    }
    for (batch, b) in w.batches.iter().enumerate() {
        for (member, &seed) in b.seeds.iter().enumerate() {
            if prior.contains_family(&seed_family_id(seed)) {
                return Err("Allocation intersects GLOBAL PRIOR (including a self-union)".into());
            }
            let players = if member < 2 { 3 } else { 4 };
            let identity = SamplingStreamIdentity::new(
                SamplingSeed::new(w.original.sampling_seed),
                first + 4 * batch as u64 + member as u64,
                w.original.replicate,
                players,
            )?;
            NativeStochasticConfig::new(players, seed, identity, limits)?;
            if (0..players).any(|actor| history.contains(identity, actor)) {
                return Err("New allocation intersects GLOBAL PRIOR streams".into());
            }
        }
    }
    Ok(())
}
fn check_prior(
    journal: &SemanticPlayingJournal<'_>,
    prior: &ConditionalNativeExclusions,
    history: &DeclaredSamplingHistory,
    forbidden: &[u64],
) -> Result<(), String> {
    if journal
        .session
        .consumed_families()
        .iter()
        .any(|f| !prior.contains_family(f))
    {
        return Err("GLOBAL PRIOR omits earlier journal family burns".into());
    }
    let ledger = journal.session.capture_ledger();
    for row in ledger["streams"]
        .as_array()
        .ok_or("Session stream ledger")?
    {
        let r: [u64; 5] = serde_json::from_value(row.clone()).map_err(|e| e.to_string())?;
        let identity = SamplingStreamIdentity::new(
            SamplingSeed::new(r[0]),
            r[1],
            r[2],
            usize::try_from(r[3]).map_err(|_| "Ledger players")?,
        )?;
        if forbidden.binary_search(&r[0]).is_err() && !history.contains(identity, r[4] as usize) {
            return Err("GLOBAL PRIOR omits earlier journal stream burns".into());
        }
    }
    Ok(())
}
fn check_original_prior(w: &Wire, prior: &ConditionalNativeExclusions) -> Result<(), String> {
    // The existing typed loader has already validated this exact pinned wire.
    let original = unique_json(
        &pinned(&w.original.external.pin, 32 * 1024 * 1024)?,
        32 * 1024 * 1024,
    )?;
    if original["families"]
        .as_array()
        .ok_or("Original family closure")?
        .iter()
        .any(|f| f.as_str().is_none_or(|f| !prior.contains_family(f)))
    {
        return Err("GLOBAL PRIOR omits original external families".into());
    }
    for r in &w.original.history.ranges {
        if w.global_prior
            .forbidden_sampling_seeds
            .binary_search(&r[0])
            .is_ok()
        {
            continue;
        }
        let mut cursor = r[4];
        for q in &w.global_prior.history.ranges {
            if q[..4] == r[..4] && q[4] <= cursor && cursor < q[5] {
                cursor = q[5];
            }
        }
        if cursor < r[5] {
            return Err("GLOBAL PRIOR omits original declared sampling history".into());
        }
    }
    Ok(())
}
fn recheck_inputs(
    w: &Wire,
    prior: &ConditionalNativeExclusions,
    history: &DeclaredSamplingHistory,
) -> Result<(), String> {
    prior.recheck()?;
    history.recheck()?;
    pinned(&w.procedural_pool, MAX_CONTROL)?;
    pinned(&w.permanent_allocation, MAX_CONTROL)?;
    Ok(())
}

// Called by the existing loaded-origin fixture. These are control/tuple
// assertions, not an execution authorization or an alternative game runner.
#[cfg(test)]
pub(super) fn control_fixture(root: &Path, limits: CollectionLimits, seeds: [u32; 4]) {
    let pin = |path: PathBuf| {
        let raw = fs::read(&path).unwrap();
        ContentPin {
            path: path.to_str().unwrap().into(),
            bytes: raw.len() as u64,
            sha256: crate::model::digest(&raw),
        }
    };
    let external_pin = pin(root.join("external.json"));
    let family = seed_family_id(42);
    let external = json!({"pin":external_pin,"count":1,
        "canonicalDigest":crate::model::digest(format!("{family}\n").as_bytes())});
    let history = json!({"ranges":[[17,0,3,0,90,94]],"pins":[external_pin]});
    let mut value = json!({"schema":RUN_SCHEMA,"mode":"start",
        "journalDirectory":root.join("new-cli-journal"),
        "original":{"checkpoint":pin(root.join("checkpoint/checkpoint.json")),
            "collection":pin(root.join("collection.json")),"external":external,"history":history,
            "criticSeed":11,"samplingSeed":17,"replicate":0,"firstOrdinal":200,
            "actorConfig":OneStepConfig::default(),"criticConfig":ResidualFitConfig::new(0.001,0.01,256).unwrap(),
            "limits":[limits.max_callbacks(),limits.max_candidate_rows(),limits.max_source_bytes(),limits.max_trace_bytes()],
            "budget":[3,1u64<<30,1u64<<30,512u64<<20,2u64<<30]},
        // These pins test closed control parsing only; run() would independently
        // require a genuine pool and exact permanent allocation before loading.
        "proceduralPool":external_pin,"permanentAllocation":external_pin,"restore":null,
        "globalPrior":{"external":external,"history":history,"forbiddenSamplingSeeds":[]},
        "batches":[{"seeds":seeds,"captureBefore":"current","slots":[
            {"member":3,"actor":3,"snapshot":{"kind":"local","alias":"current"}}]}]});
    let parse = |value: &Value| {
        let raw = serde_json::to_vec(value).unwrap();
        SemanticJournalRunSpec::from_json(&raw, &crate::model::digest(&raw))
    };
    let spec = parse(&value).unwrap();
    let w = &spec.wire;
    let declared = allocation(w, &plan_checksum(w).unwrap(), 200, limits).unwrap();
    let streams = declared["actorStreams"].as_array().unwrap();
    assert_eq!(streams.len(), 14);
    assert_eq!(streams[0], json!([17, 200, 0, 3, 0]));
    assert_eq!(streams[13], json!([17, 203, 0, 4, 3]));
    assert_eq!(declared["nextOrdinal"], 204);
    let prior = w.global_prior.external.load().unwrap();
    let history = w.global_prior.history.load().unwrap();
    check_original_prior(w, &prior).unwrap();
    check_new(w, 200, &prior, &history, limits).unwrap();
    value["batches"][0]["slots"][0]["snapshot"]["alias"] = "future".into();
    assert!(parse(&value).is_err());
    value["batches"][0]["slots"][0]["snapshot"]["alias"] = "current".into();
    value["batches"][0]["seeds"][0] = 42.into();
    let self_union = parse(&value).unwrap();
    assert!(check_new(&self_union.wire, 200, &prior, &history, limits).is_err());
}
