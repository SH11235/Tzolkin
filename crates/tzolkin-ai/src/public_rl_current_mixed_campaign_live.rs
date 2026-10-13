//! A separately bounded live plan; native owners and physical archive barriers
//! remain authoritative. Wrapper returns do not certify OS process closure.
use super::*;

const LIVE_SCHEMA: &str = "tzolkin-current-mixed-live-plan-spec-v1";
const SEGMENT_PLAN_SCHEMA: &str = "tzolkin-current-mixed-live-segment-plan-v1";
const RESERVATION_SCHEMA: &str = "tzolkin-current-mixed-live-reservation-v1";
const CALL_CAP: usize = 64 * 1024;
const INPUT_CAP: usize = 128 * 1024 * 1024;
const WRAPPER_CAP: u64 = 128 * 1024 * 1024;
const MAX_LIVE_UPDATES: usize = 1000;
const MAX_LIVE_SEGMENTS: usize = 24;

#[derive(Clone, Deserialize, Serialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct LiveWire {
    schema: String,
    checkpoint: InputPin,
    dataset_manifest: InputPin,
    bootstrap: Bootstrap,
    original_mixed: OriginalMixed,
    initial: InitialDurable,
    steps: Vec<SavedStep>,
    exclusions: Exclusions,
    non_updated_baseline: [[u64; 5]; 3],
    segments: Vec<InputPin>,
    max_aggregate_input_bytes: usize,
    max_wrapper_bytes: u64,
    max_invocation_updates: usize,
    max_invocation_segments: usize,
    max_invocation_wall_seconds: u64,
    max_invocation_new_retained_bytes: u64,
}
#[derive(Clone, Deserialize, Serialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct SegmentPlan {
    schema: String,
    segment_spec: Value,
    pool: Value,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    rotation: Option<CampaignSeatRotation>,
    collections: Vec<Value>,
}
struct PlannedSegment {
    wire: SegmentPlan,
    raw: Vec<u8>,
}
/// A closed, SHA-pinned finite schedule. It cannot import a numerical owner.
pub struct CurrentMixedLivePlanSpec {
    wire: LiveWire,
    raw: Vec<u8>,
    raw_sha256: String,
    segments: Vec<PlannedSegment>,
    first: CurrentMixedCampaignSpec,
    total: usize,
}
fn add(a: u64, b: u64) -> Result<u64, String> {
    a.checked_add(b)
        .ok_or_else(|| "Live plan accounting overflow".into())
}
fn number(value: &Value, key: &str) -> Result<u64, String> {
    value[key]
        .as_u64()
        .ok_or_else(|| format!("Live plan integer field {key}"))
}
impl CurrentMixedLivePlanSpec {
    pub fn max_invocation_updates(&self) -> usize {
        self.wire.max_invocation_updates
    }
    pub fn from_json(raw: &[u8], expected_sha256: &str) -> Result<Self, String> {
        if !checkpoint_digest(expected_sha256) || sha(raw) != expected_sha256 {
            return Err("Live plan expected raw SHA256 differs".into());
        }
        let wire: LiveWire = closed(raw, CAMPAIGN_CAP)?;
        if wire.schema != LIVE_SCHEMA
            || wire.steps.is_empty()
            || matches!(wire.steps.first(), Some(SavedStep::SeatRotation { .. }))
            || !(1..=MAX_LIVE_SEGMENTS).contains(&wire.segments.len())
            || !(1..=INPUT_CAP).contains(&wire.max_aggregate_input_bytes)
            || wire.max_wrapper_bytes == 0
            || wire.max_wrapper_bytes > WRAPPER_CAP
            || !(1..=MAX_LIVE_UPDATES).contains(&wire.max_invocation_updates)
            || !(1..=MAX_LIVE_SEGMENTS).contains(&wire.max_invocation_segments)
            || !(1..=86400).contains(&wire.max_invocation_wall_seconds)
            || wire.max_invocation_new_retained_bytes == 0
            || wire.max_invocation_new_retained_bytes > MAX_STORAGE
        {
            return Err("Live plan schema/finite declared limits/established lineage".into());
        }
        let mut aggregate = raw.len();
        let mut segments = Vec::new();
        let mut total = 0usize;
        let mut paths = BTreeSet::new();
        let mut pool = collection(&wire.initial.pool)?.pool_checksum()?;
        for step in &wire.steps {
            if let SavedStep::GuardedPoolFork { new_pool, .. }
            | SavedStep::SeatRotation { new_pool, .. } = step
            {
                pool = collection(new_pool)?.pool_checksum()?;
            }
        }
        for pin in &wire.segments {
            if !paths.insert(local(&pin.path, false)?) {
                return Err("Live plan repeated segment input file".into());
            }
            let raw = pin.read(CAMPAIGN_CAP)?;
            aggregate = aggregate
                .checked_add(raw.len())
                .ok_or("Live plan input overflow")?;
            let plan: SegmentPlan = closed(&raw, CAMPAIGN_CAP)?;
            let spec = segment_spec(&plan.segment_spec)?;
            let run =
                run_spec(&serde_json::to_value(&spec).map_err(|e| e.to_string())?["runSpec"])?;
            if plan.schema != SEGMENT_PLAN_SCHEMA
                || !(1..=MAX_ATTEMPTS as usize).contains(&plan.collections.len())
                || plan.collections.len() as u64 > run.max_attempts
                || plan.collections.len() as u64 > run.max_successful_updates
                || run.max_journal_entries
                    < add(1, EVENTS_PER_ATTEMPT * plan.collections.len() as u64)?
            {
                return Err("Live plan bounded local segment/cohort capacity".into());
            }
            if let Some(rotation) = &plan.rotation {
                pool = rotation.checked_pools(&pool)?.1.pool_checksum()?;
            }
            if collection(&plan.pool)?.pool_checksum()? != pool {
                return Err("Live plan segment pool differs from cyclic predecessor".into());
            }
            for value in &plan.collections {
                if collection(value)?.pool_checksum()? != pool {
                    return Err("Live plan cohort changes the declared pool".into());
                }
            }
            total = total
                .checked_add(plan.collections.len())
                .ok_or("Live plan update overflow")?;
            segments.push(PlannedSegment { wire: plan, raw });
        }
        if total > MAX_LIVE_UPDATES
            || aggregate > wire.max_aggregate_input_bytes
            || (total as u64)
                .checked_mul((CALL_CAP + 1024) as u64)
                .and_then(|n| n.checked_add(4 * HEAD_CAP as u64))
                .is_none_or(|n| n > wire.max_wrapper_bytes)
        {
            return Err("Live plan aggregate updates/input/report capacity".into());
        }
        let first = &segments[0].wire;
        let original = CampaignWire {
            schema: CAMPAIGN_SCHEMA.into(),
            checkpoint: wire.checkpoint.clone(),
            dataset_manifest: wire.dataset_manifest.clone(),
            bootstrap: wire.bootstrap.clone(),
            original_mixed: wire.original_mixed.clone(),
            initial: wire.initial.clone(),
            steps: wire.steps.clone(),
            segment_spec: first.segment_spec.clone(),
            amend_budget: false,
            pool: first.pool.clone(),
            collections: first.collections.clone(),
            exclusions: wire.exclusions.clone(),
            seat_campaign: Some(SeatCampaign {
                schema: SEAT_CAMPAIGN_SCHEMA.into(),
                non_updated_baseline: wire.non_updated_baseline,
                rotation: first.rotation.clone(),
            }),
        };
        let first_raw = encoded(&original)?;
        let first = CurrentMixedCampaignSpec::from_json(&first_raw, &sha(&first_raw))?;
        Ok(Self {
            wire,
            raw: raw.to_vec(),
            raw_sha256: expected_sha256.into(),
            segments,
            first,
            total,
        })
    }
    fn require_pins(&self) -> Result<(), String> {
        self.first.require_input_pins()?;
        let mut aggregate = self.raw.len();
        for (pin, segment) in self.wire.segments.iter().zip(&self.segments) {
            if pin.read(CAMPAIGN_CAP)? != segment.raw {
                return Err("Live plan pinned segment bytes changed".into());
            }
            aggregate = aggregate
                .checked_add(pin.bytes)
                .ok_or("Live plan input overflow")?;
        }
        let mut pins = vec![
            &self.wire.checkpoint,
            &self.wire.dataset_manifest,
            &self.wire.exclusions.file,
            &self.wire.original_mixed.tip.head,
            &self.wire.initial.tip.head,
        ];
        pins.extend(self.wire.steps.iter().map(|step| &step.tip().head));
        match &self.wire.bootstrap {
            Bootstrap::Count1 { checkpoint } => pins.push(checkpoint),
            Bootstrap::Paired { head, .. } => pins.push(head),
        }
        for pin in pins {
            aggregate = aggregate
                .checked_add(pin.bytes)
                .ok_or("Live plan input overflow")?;
        }
        if aggregate > self.wire.max_aggregate_input_bytes {
            return Err("Live plan all input pins exceed aggregate byte cap".into());
        }
        Ok(())
    }
    fn preflight(&self) -> Result<LivePreflight, String> {
        self.require_pins()?;
        let original = self.first.preflight()?;
        let latest = original.steps.last().unwrap_or(&original.initial);
        let binding = serde_json::to_value(
            latest
                .binding
                .as_ref()
                .ok_or("Live plan established contract missing")?,
        )
        .map_err(|e| e.to_string())?;
        let contract = &binding["contract"];
        let inherited = &binding["inherited"];
        let mut budget = Budget {
            attempts: add(number(inherited, "attempts")?, latest.progress.attempts)?,
            updates: add(
                number(inherited, "successfulUpdates")?,
                latest.progress.successful,
            )?,
            work: Work {
                callbacks: add(
                    number(&inherited["reservedWork"], "callbacks")?,
                    latest.progress.work.callbacks,
                )?,
                rows: add(
                    number(&inherited["reservedWork"], "rows")?,
                    latest.progress.work.rows,
                )?,
                search: add(
                    number(&inherited["reservedWork"], "search")?,
                    latest.progress.work.search,
                )?,
                bytes: add(
                    number(&inherited["reservedWork"], "bytes")?,
                    latest.progress.work.bytes,
                )?,
            },
            raw: add(
                number(inherited, "logicalRecordBytes")?,
                latest.archive_usage[0],
            )?,
            stored: add(
                number(inherited, "storedRecordBytes")?,
                latest.archive_usage[1],
            )?,
            metadata: add(number(inherited, "metadataBytes")?, latest.archive_usage[2])?,
        };
        let index = number(&binding, "index")?;
        if add(add(index, 1)?, self.segments.len() as u64)? > number(contract, "maxSegments")? {
            return Err("Live plan exceeds actual established segment envelope".into());
        }
        let mut used = latest.progress.consumed.clone();
        let original_run = run_spec(
            &self
                .wire
                .steps
                .last()
                .ok_or("Live plan established predecessor missing")?
                .spec()["runSpec"],
        )?;
        let mut update_envelope = latest.progress.parent_count()?;
        let mut next = latest.progress.next;
        let mut configurations = Vec::new();
        let mut families = Vec::new();
        for segment in &self.segments {
            require_established_seat_contract(latest, &segment.wire.segment_spec)?;
            let normalized = serde_json::to_value(segment_spec(&segment.wire.segment_spec)?)
                .map_err(|e| e.to_string())?;
            let run = run_spec(&normalized["runSpec"])?;
            if run.actor != original_run.actor || run.critic != original_run.critic {
                return Err(
                    "Live plan future actor/critic controls differ from predecessor".into(),
                );
            }
            update_envelope = add(update_envelope, run.max_successful_updates)?;
            if update_envelope > source::MAX_MIXED_UPDATES {
                return Err("Live plan future mixed-update count exceeds native capacity".into());
            }
            budget.reserve(contract, &run)?;
            let mut segment_configs = Vec::new();
            let mut segment_families = Vec::new();
            for value in &segment.wire.collections {
                let configs = collection(value)?
                    .inner
                    .configs()?
                    .each_ref()
                    .map(Config::from_config);
                let family = configs
                    .each_ref()
                    .map(|c| seed_family_id(c.environment_seed));
                for (offset, (c, f)) in configs.iter().zip(&family).enumerate() {
                    if c.sampling_seed != format!("{:016x}", latest.progress.prefix[0])
                        || c.replicate_ordinal != format!("{:016x}", latest.progress.prefix[1])
                        || c.episode_ordinal != format!("{:016x}", add(next, offset as u64)?)
                        || split_for_family(f)? != DatasetSplit::Train
                        || original.exclusions.contains(f)
                        || !used.insert(f.clone())
                    {
                        return Err(
                            "Live plan all future Train families/streams must be new and disjoint"
                                .into(),
                        );
                    }
                }
                next = add(next, 4)?;
                segment_configs.push(configs);
                segment_families.push(family);
            }
            if used.len() > MAX_FAMILIES
                || used.len() > run.max_families
                || used.len() as u64 > number(contract, "maxFamilies")?
            {
                return Err("Live plan aggregate genuine family capacity".into());
            }
            configurations.push(segment_configs);
            families.push(segment_families);
        }
        Ok(LivePreflight {
            original,
            configurations,
            families,
            inherited_index: index,
        })
    }
    fn reservation(&self, pre: &LivePreflight) -> Result<Vec<u8>, String> {
        bytes(
            &serde_json::json!({"schema":RESERVATION_SCHEMA,"planRawSha256":self.raw_sha256,
            "allDeclaredFamiliesRemainSpent":true,"segmentInputs":self.wire.segments,
            "configurations":pre.configurations,"families":pre.families,"externalExclusions":self.wire.exclusions,
            "producerAuthenticated":false,"freshTestAdmission":false,"strengthOrAdoptionAdmission":false}),
            INPUT_CAP,
        )
    }
}
struct Budget {
    attempts: u64,
    updates: u64,
    work: Work,
    raw: u64,
    stored: u64,
    metadata: u64,
}
impl Budget {
    fn reserve(&mut self, contract: &Value, run: &CurrentMixedRunSpec) -> Result<(), String> {
        self.attempts = add(self.attempts, run.max_attempts)?;
        self.updates = add(self.updates, run.max_successful_updates)?;
        self.work = self.work.add(Work {
            callbacks: run.max_callbacks,
            rows: run.max_candidate_rows,
            search: run.max_search_steps,
            bytes: run.max_output_bytes,
        })?;
        self.raw = add(self.raw, run.max_output_bytes)?;
        let (stored, temp, disk) = match &run.record_storage {
            Some(s) => (s.max_stored_bytes, s.max_temporary_bytes, s.max_disk_bytes),
            None => (
                run.max_output_bytes,
                MAX_RECORD_BYTES as u64,
                add(run.max_output_bytes, MAX_RECORD_BYTES as u64)?,
            ),
        };
        let disk = add(add(self.stored, self.metadata)?, disk)?;
        self.stored = add(self.stored, stored)?;
        self.metadata = add(self.metadata, run.max_metadata_bytes)?;
        for (actual, key) in [
            (self.attempts, "maxAttempts"),
            (self.updates, "maxSuccessfulUpdates"),
            (self.work.callbacks, "maxCallbacks"),
            (self.work.rows, "maxCandidateRows"),
            (self.work.search, "maxSearchSteps"),
            (self.work.bytes, "maxDeclaredWorkBytes"),
            (self.raw, "maxLogicalRecordBytes"),
            (self.stored, "maxStoredRecordBytes"),
            (self.metadata, "maxMetadataBytes"),
            (temp, "maxTemporaryBytes"),
            (disk, "maxDiskBytes"),
        ] {
            if actual > number(contract, key)? {
                return Err(format!("Live plan exceeds established {key}"));
            }
        }
        Ok(())
    }
}
struct LivePreflight {
    original: Preflight,
    configurations: Vec<Vec<[Config; 4]>>,
    families: Vec<Vec<[String; 4]>>,
    inherited_index: u64,
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct Cursor {
    segment: usize,
    updates_in_segment: usize,
    total_updates: usize,
}
impl Cursor {
    fn initial() -> Self {
        Self {
            segment: 0,
            updates_in_segment: 0,
            total_updates: 0,
        }
    }
    fn require(&self, plan: &CurrentMixedLivePlanSpec) -> Result<(), String> {
        if self.segment >= plan.segments.len()
            || self.updates_in_segment == 0
            || self.updates_in_segment > plan.segments[self.segment].wire.collections.len()
            || self.total_updates
                != plan
                    .segments
                    .iter()
                    .take(self.segment)
                    .map(|s| s.wire.collections.len())
                    .sum::<usize>()
                    + self.updates_in_segment
        {
            return Err("Live plan saved cursor is not an exact successful prefix".into());
        }
        Ok(())
    }
}
#[derive(Deserialize, Serialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct LiveIntent {
    schema: String,
    plan_raw_sha256: String,
    before: Cursor,
    max_new_updates: usize,
}
#[derive(Deserialize, Serialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct LiveClose {
    schema: String,
    plan_raw_sha256: String,
    before: Cursor,
    after: Cursor,
    tip: SavedTip,
    segment_tips: Vec<SavedTip>,
    stop_reason: String,
    report: FilePin,
    calls: Vec<FilePin>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    deployment: Option<FilePin>,
}
#[derive(Deserialize, Serialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct LiveInspection {
    schema: String,
    plan_raw_sha256: String,
    cursor_claims: Cursor,
    numeric_owner_admission: bool,
    new_reservations: u64,
    new_originals: u64,
    new_updates: u64,
    process_closure_authenticated: bool,
}
fn segment_directory(output: &Path, index: usize) -> PathBuf {
    output.join("segments").join(format!("seg-{index:04}"))
}
fn live_name(prefix: usize, suffix: &str) -> String {
    format!("live-{prefix:016x}-{suffix}.json")
}
fn require_leaf(name: &str) -> Result<(), String> {
    leaf_name(name)?;
    let folded = name.to_ascii_lowercase();
    if matches!(folded.as_str(), "segments" | "live-plan.json")
        || folded.starts_with("live-")
        || folded.starts_with("call-")
    {
        return Err("Live plan output leaf aliases its controls".into());
    }
    Ok(())
}
fn require_closed_prefix(
    plan: &CurrentMixedLivePlanSpec,
    cursor: Cursor,
    intent: &LiveIntent,
    close: &LiveClose,
) -> Result<(), String> {
    if intent.schema != "tzolkin-current-mixed-live-intent-v1"
        || close.schema != "tzolkin-current-mixed-live-close-v1"
        || intent.plan_raw_sha256 != plan.raw_sha256
        || close.plan_raw_sha256 != plan.raw_sha256
        || intent.before != cursor
        || close.before != cursor
        || intent.max_new_updates == 0
        || intent.max_new_updates > plan.wire.max_invocation_updates
        || close.after.total_updates <= cursor.total_updates
        || close.after.total_updates - cursor.total_updates > intent.max_new_updates
        || !matches!(
            close.stop_reason.as_str(),
            "invocationUpdateLimit"
                | "invocationSegmentLimit"
                | "invocationWallLimit"
                | "invocationRetainedLimit"
                | "declaredPlanComplete"
        )
        || (close.stop_reason == "declaredPlanComplete")
            != (close.after.total_updates == plan.total)
    {
        return Err("Live plan previous invocation has no closed successful exact prefix".into());
    }
    close.after.require(plan)?;
    let new_starts = if cursor.total_updates == 0 {
        close.after.segment + 1
    } else {
        close
            .after
            .segment
            .checked_sub(cursor.segment)
            .ok_or("Live plan closed cursor moves backward")?
    };
    if close.after.segment < cursor.segment
        || new_starts > plan.wire.max_invocation_segments
        || close.segment_tips.len() != close.after.segment - cursor.segment + 1
        || encoded(
            close
                .segment_tips
                .last()
                .ok_or("Live plan closed native tip missing")?,
        )? != encoded(&close.tip)?
        || close.calls.len() != close.after.total_updates - cursor.total_updates
    {
        return Err("Live plan complete native segment-return inventory differs".into());
    }
    Ok(())
}
fn check_native_prefix(
    plan: &CurrentMixedLivePlanSpec,
    pre: &LivePreflight,
    index: usize,
    tip: &MetadataTip,
) -> Result<usize, String> {
    let collections = &plan.segments[index].wire.collections;
    if tip.progress.phase != Phase::Ready
        || tip.progress.unknown_work
        || tip.progress.attempts != tip.progress.successful
        || tip.progress.attempts == 0
        || tip.progress.outcome.as_deref() != Some("updated")
        || tip.progress.attempts as usize > collections.len()
        || tip.reservations.len() != tip.progress.attempts as usize
        || tip.resolved_heads.len() != tip.reservations.len()
    {
        return Err("Live plan failed/NoChange/unresolved native prefix".into());
    }
    for (i, r) in tip.reservations.iter().enumerate() {
        if r.configurations != pre.configurations[index][i]
            || r.families != pre.families[index][i]
            || read_pin(
                &tip.directory.join(format!("batch-{:016x}", r.attempt)),
                &r.collection_spec,
                SPEC_CAP,
            )? != collection(&collections[i])?.inner.to_json()?
        {
            return Err("Live plan native prefix differs from whole campaign reservation".into());
        }
    }
    Ok(tip.reservations.len())
}
fn checked_prior(
    plan: &CurrentMixedLivePlanSpec,
    pre: &LivePreflight,
    output: &Path,
    head: (&Path, &str),
) -> Result<(Cursor, Vec<MetadataTip>, Vec<SavedTip>), String> {
    let root_names = names(output, 6 * MAX_LIVE_UPDATES + 16)?;
    if root_names.iter().any(|name| name.ends_with(".tmp")) {
        return Err("Live plan uncertain wrapper publication".into());
    }
    let mut cursor = Cursor::initial();
    let mut allowed = BTreeSet::from([
        "live-plan.json".to_string(),
        "reservation.json".to_string(),
        "segments".to_string(),
    ]);
    let mut closes = Vec::new();
    let mut returned_calls = Vec::new();
    loop {
        let intent_name = live_name(cursor.total_updates, "intent");
        if !output.join(&intent_name).exists() {
            break;
        }
        let close_name = live_name(cursor.total_updates, "close");
        let intent: LiveIntent = closed(
            &read_bounded(&output.join(&intent_name), HEAD_CAP)?,
            HEAD_CAP,
        )?;
        let close: LiveClose = closed(
            &read_bounded(&output.join(&close_name), HEAD_CAP)?,
            HEAD_CAP,
        )?;
        require_closed_prefix(plan, cursor, &intent, &close)?;
        require_leaf(&close.report.name)?;
        read_pin(output, &close.report, HEAD_CAP)?;
        for (offset, tip) in close.segment_tips.iter().enumerate() {
            if local(&tip.directory, false)?
                != local(&segment_directory(output, cursor.segment + offset), false)?
            {
                return Err("Live plan closed segment namespace differs".into());
            }
        }
        if let Some(pin) = &close.deployment {
            require_leaf(&pin.name)?;
            read_pin(output, pin, ARTIFACT_CAP)?;
            allowed.insert(pin.name.clone());
        }
        if close.calls.len() != close.after.total_updates - cursor.total_updates {
            return Err("Live plan call-return inventory differs".into());
        }
        for (offset, pin) in close.calls.iter().enumerate() {
            if pin.name != format!("call-{:016x}.json", cursor.total_updates + offset) {
                return Err("Live plan call pin sequence differs".into());
            }
            let call: Value = unique_json(&read_pin(output, pin, CALL_CAP)?, CALL_CAP)?;
            if call["schema"] != "tzolkin-current-mixed-live-call-v1"
                || call["planRawSha256"] != plan.raw_sha256
                || call["totalUpdates"].as_u64() != Some((cursor.total_updates + offset + 1) as u64)
                || call["status"]["phase"] != "ready"
                || call["status"]["unknownWorkPossible"] != false
                || call["status"]["outcome"] != "updated"
            {
                return Err("Live plan saved native call did not return Updated".into());
            }
            let call_cursor: Cursor =
                serde_json::from_value(call["cursor"].clone()).map_err(|e| e.to_string())?;
            call_cursor.require(plan)?;
            if call_cursor.total_updates as u64 != number(&call, "totalUpdates")?
                || number(&call["status"], "attempts")? != call_cursor.updates_in_segment as u64
            {
                return Err("Live plan call cursor differs from its prefix".into());
            }
            let digest = call["status"]["headSha256"]
                .as_str()
                .ok_or("Live plan call head missing")?;
            if !checkpoint_digest(digest) {
                return Err("Live plan call head shape".into());
            }
            returned_calls.push((call_cursor, digest.to_string()));
            allowed.insert(pin.name.clone());
        }
        allowed.insert(intent_name);
        allowed.insert(close_name);
        allowed.insert(close.report.name.clone());
        cursor = close.after;
        closes.push(close);
        if cursor.total_updates == plan.total {
            break;
        }
    }
    if closes.is_empty() {
        return Err("Live plan missing closed invocation".into());
    }
    for name in root_names.iter().filter(|name| !allowed.contains(*name)) {
        // Metadata-only inspection is inert, never a substitute for a close or
        // a native tip. A dangling control/foreign output remains a refusal.
        require_leaf(name)?;
        let inspection: LiveInspection =
            closed(&read_bounded(&output.join(name), HEAD_CAP)?, HEAD_CAP)?;
        inspection.cursor_claims.require(plan)?;
        if inspection.schema != "tzolkin-current-mixed-live-inspection-v1"
            || inspection.plan_raw_sha256 != plan.raw_sha256
            || inspection.cursor_claims.total_updates > cursor.total_updates
            || inspection.numeric_owner_admission
            || inspection.process_closure_authenticated
            || inspection.new_reservations != 0
            || inspection.new_originals != 0
            || inspection.new_updates != 0
        {
            return Err("Live plan foreign output is not inert metadata inspection".into());
        }
    }
    let last = closes.last().ok_or("Live plan return missing")?;
    if local(head.0, false)? != local(&last.tip.head.path, false)?
        || head.1 != last.tip.head.sha256
        || local(&last.tip.directory, false)?
            != local(&segment_directory(output, cursor.segment), false)?
    {
        return Err("Live plan external latest head differs from closed return".into());
    }
    let segment_names = names(&output.join("segments"), MAX_LIVE_SEGMENTS)?;
    if segment_names.len() != cursor.segment + 1
        || (0..=cursor.segment).any(|i| !segment_names.contains(&format!("seg-{i:04}")))
    {
        return Err("Live plan unclosed or foreign SegmentStart directory".into());
    }
    let mut metadata = Vec::new();
    let mut saved = Vec::new();
    let mut cumulative = pre.original.cumulative_own;
    for index in 0..=cursor.segment {
        let directory = local(&segment_directory(output, index), false)?;
        let tip = closes
            .iter()
            .rev()
            .flat_map(|c| c.segment_tips.iter().rev())
            .find(|tip| local(&tip.directory, false).ok().as_ref() == Some(&directory))
            .ok_or("Live plan closed segment tip missing")?;
        if local(&tip.directory, false)? != local(&segment_directory(output, index), false)? {
            return Err("Live plan saved tip directory differs".into());
        }
        let segment = &plan.segments[index].wire;
        let actual = metadata_tip(
            tip,
            &segment.segment_spec,
            &collection(&segment.pool)?.pool_checksum()?,
            true,
        )?;
        let previous = metadata
            .last()
            .unwrap_or_else(|| pre.original.steps.last().unwrap_or(&pre.original.initial));
        successor_metadata(previous, &actual, false, segment.rotation.is_some())?;
        require_established_seat_contract(previous, &segment.segment_spec)?;
        if let Some(rotation) = &segment.rotation {
            rotation_metadata(
                previous,
                &actual,
                &previous.entry.pool,
                &collection(&segment.pool)?.pool_checksum()?,
                rotation.shifts,
                cumulative,
            )?;
        }
        let completed = check_native_prefix(plan, pre, index, &actual)?;
        if completed
            != if index == cursor.segment {
                cursor.updates_in_segment
            } else {
                segment.collections.len()
            }
        {
            return Err("Live plan native Updated suffix exceeds closed cursor".into());
        }
        for close in closes.iter().filter(|c| c.after.segment == index) {
            if close.tip.head.sha256 != actual.resolved_heads[close.after.updates_in_segment - 1] {
                return Err("Live plan wrapper head is not the exact native Updated prefix".into());
            }
        }
        for (call_cursor, digest) in &returned_calls {
            if call_cursor.segment == index
                && (call_cursor.updates_in_segment > completed
                    || digest != &actual.resolved_heads[call_cursor.updates_in_segment - 1])
            {
                return Err("Live plan source return is not its native exact Updated head".into());
            }
        }
        cumulative = cumulative.checked_add(actual.progress.own_decisions)?;
        plan.first.require_seat_baseline(cumulative)?;
        metadata.push(actual);
        saved.push(tip.clone());
    }
    Ok((cursor, metadata, saved))
}
fn require_live_prefix(
    plan: &CurrentMixedLivePlanSpec,
    pre: &LivePreflight,
    session: &CurrentMixedTrainingSession<'_>,
    cursor: Cursor,
    started: bool,
) -> Result<(), String> {
    let latest = pre.original.steps.last().unwrap_or(&pre.original.initial);
    let expected = latest
        .progress
        .consumed
        .iter()
        .cloned()
        .chain(
            pre.families
                .iter()
                .flatten()
                .take(cursor.total_updates)
                .flatten()
                .cloned(),
        )
        .collect::<BTreeSet<_>>();
    let expected_count = add(latest.progress.parent_count()?, cursor.total_updates as u64)?;
    if session.progress.phase != Phase::Ready
        || session.progress.unknown_work
        || session.progress.outcome.as_deref() != Some("updated")
            && (!started || cursor.updates_in_segment > 0)
        || session.reserved_families() != &expected
        || session.progress.prefix != latest.progress.prefix
        || session.progress.imported != latest.progress.imported
        || session.progress.next != add(latest.progress.next, 4 * cursor.total_updates as u64)?
        || session.progress.parent_count() != Ok(expected_count)
        || started
            && (session.pool
                != collection(&plan.segments[cursor.segment].wire.pool)?.pool_checksum()?
                || session.progress.attempts != cursor.updates_in_segment as u64
                || session.progress.successful != cursor.updates_in_segment as u64)
    {
        return Err("Live plan genuine owner differs from the exact declared prefix".into());
    }
    plan.first.require_live_seat_baseline(session)
}
fn start_segment<'a>(
    plan: &CurrentMixedLivePlanSpec,
    index: usize,
    session: CurrentMixedTrainingSession<'a>,
    output: &Path,
) -> Result<CurrentMixedTrainingSession<'a>, String> {
    if session.progress.outcome.as_deref() != Some("updated")
        || session.progress.phase != Phase::Ready
        || session.progress.unknown_work
    {
        return Err("Live plan transition requires latest genuine Updated".into());
    }
    plan.require_pins()?;
    plan.first.require_live_seat_baseline(&session)?;
    let own_before = session.cumulative_own_decisions()?;
    let wire = &plan.segments[index].wire;
    let spec = segment_spec(&wire.segment_spec)?;
    let ready = session.into_ready_continuation()?;
    // Constructors retain their full Immediate physical archive barrier.
    let next = if let Some(rotation) = &wire.rotation {
        let old = collection(&rotation.old_pool)?;
        let new = collection(&rotation.new_pool)?;
        CurrentMixedTrainingSession::new_seat_rotation_segment(
            ready,
            spec,
            &old,
            &new,
            rotation.shifts,
            &segment_directory(output, index),
        )?
    } else {
        CurrentMixedTrainingSession::new_segment(ready, spec, &segment_directory(output, index))?
    };
    if next.cumulative_own_decisions()? != own_before {
        return Err("Live plan transition changes genuine cumulative own counters".into());
    }
    Ok(next)
}

fn wrapper_bytes(output: &Path) -> Result<u64, String> {
    if !output.exists() {
        return Ok(0);
    }
    let mut total = 0u64;
    for name in names(output, 6 * MAX_LIVE_UPDATES + 16)? {
        let path = local(&output.join(&name), false)?;
        let metadata = fs::symlink_metadata(&path).map_err(|e| e.to_string())?;
        if name == "segments" && metadata.is_dir() {
            continue;
        }
        if !metadata.is_file() || metadata.file_type().is_symlink() {
            return Err("Live plan wrapper inventory is not regular".into());
        }
        total = add(total, metadata.len())?;
    }
    Ok(total)
}
fn publish_live(
    output: &Path,
    name: &str,
    raw: &[u8],
    limit: u64,
    used: &mut u64,
) -> Result<(), String> {
    let next = add(*used, raw.len() as u64)?;
    if next > limit {
        return Err("Live plan aggregate wrapper byte capacity".into());
    }
    publish_mixed_metadata(output, name, raw)?;
    *used = next;
    Ok(())
}
fn actual_tip(session: &CurrentMixedTrainingSession<'_>) -> Result<SavedTip, String> {
    let status = session.status();
    let path = session.directory.join(status.head_file());
    let raw = read_bounded(&path, HEAD_CAP)?;
    if sha(&raw) != status.head_sha256 {
        return Err("Live plan source head changed before close".into());
    }
    Ok(SavedTip {
        head: InputPin {
            path,
            bytes: raw.len(),
            sha256: status.head_sha256,
        },
        directory: session.directory.clone(),
    })
}
/// Finite whole-campaign reservation, one genuine cold restore, then live
/// latest-Updated boundaries. An external single-writer wall/disk watchdog is
/// required; wrapper return metadata never authenticates process closure.
pub fn current_mixed_live_plan(
    plan: CurrentMixedLivePlanSpec,
    command: &str,
    output: &Path,
    head: Option<(&Path, &str)>,
    max_new_updates: usize,
    report: &str,
    deployment: Option<&str>,
) -> Result<i32, String> {
    if !matches!(command, "live-continue" | "live-resume" | "live-inspect")
        || max_new_updates == 0
        || max_new_updates > plan.wire.max_invocation_updates
    {
        return Err("Live plan command/invocation cap".into());
    }
    require_leaf(report)?;
    if let Some(name) = deployment {
        require_leaf(name)?;
        if name.eq_ignore_ascii_case(report) {
            return Err("Live plan output leaves alias".into());
        }
    }
    let began = Instant::now();
    let pre = plan.preflight()?;
    let output = local(output, command == "live-continue")?;
    plan.first.protect_output(&output)?;
    for segment in &plan.segments {
        let normalized = serde_json::to_value(segment_spec(&segment.wire.segment_spec)?)
            .map_err(|e| e.to_string())?;
        if let Some(dirs) = normalized["globalContract"]["protectedDirectories"].as_array() {
            for directory in dirs {
                let path: PathBuf =
                    serde_json::from_value(directory.clone()).map_err(|e| e.to_string())?;
                require_output_outside_directory(&local(&path, false)?, &output)?;
            }
        }
    }
    let reservation = plan.reservation(&pre)?;
    let mut wrapper_used = wrapper_bytes(&output)?;
    if wrapper_used > plan.wire.max_wrapper_bytes {
        return Err("Live plan existing wrapper capacity exceeded".into());
    }
    if reservation.len() as u64 > plan.wire.max_wrapper_bytes {
        return Err("Live plan reservation exceeds wrapper capacity".into());
    }
    let (mut cursor, metadata, saved) = if command == "live-continue" {
        if head.is_some() || output.exists() {
            return Err("Live plan continue requires a new output".into());
        }
        (Cursor::initial(), Vec::new(), Vec::new())
    } else {
        if read_bounded(&output.join("live-plan.json"), CAMPAIGN_CAP)? != plan.raw
            || read_bounded(&output.join("reservation.json"), INPUT_CAP)? != reservation
        {
            return Err("Live plan original full reservation differs".into());
        }
        checked_prior(
            &plan,
            &pre,
            &output,
            head.ok_or("Live plan resume requires pinned latest native head")?,
        )?
    };
    if output.join(report).exists() || deployment.is_some_and(|name| output.join(name).exists()) {
        return Err("Live plan outputs must be new".into());
    }
    if command == "live-inspect" {
        publish_live(
            &output,
            report,
            &bytes(
                &LiveInspection {
                    schema: "tzolkin-current-mixed-live-inspection-v1".into(),
                    plan_raw_sha256: plan.raw_sha256.clone(),
                    cursor_claims: cursor,
                    numeric_owner_admission: false,
                    new_reservations: 0,
                    new_originals: 0,
                    new_updates: 0,
                    process_closure_authenticated: false,
                },
                HEAD_CAP,
            )?,
            plan.wire.max_wrapper_bytes,
            &mut wrapper_used,
        )?;
        return Ok(0);
    }
    if cursor.total_updates == plan.total {
        return Err("Live plan already completed; no new invocation".into());
    }
    if command == "live-continue" {
        fs::create_dir(&output).map_err(|e| e.to_string())?;
        publish_live(
            &output,
            "live-plan.json",
            &plan.raw,
            plan.wire.max_wrapper_bytes,
            &mut wrapper_used,
        )?;
        publish_live(
            &output,
            "reservation.json",
            &reservation,
            plan.wire.max_wrapper_bytes,
            &mut wrapper_used,
        )?;
        fs::create_dir(output.join("segments")).map_err(|e| e.to_string())?;
    }
    let before = cursor;
    publish_live(
        &output,
        &live_name(before.total_updates, "intent"),
        &bytes(
            &LiveIntent {
                schema: "tzolkin-current-mixed-live-intent-v1".into(),
                plan_raw_sha256: plan.raw_sha256.clone(),
                before,
                max_new_updates,
            },
            HEAD_CAP,
        )?,
        plan.wire.max_wrapper_bytes,
        &mut wrapper_used,
    )?;
    let qualification = Instant::now();
    let prepared = PreparedPublicPolicy::load(
        &plan.wire.checkpoint.path,
        plan.wire
            .dataset_manifest
            .path
            .parent()
            .ok_or("Live plan dataset directory")?,
        Kernel::Scalar,
    )?;
    plan.require_pins()?;
    let qualification_seconds = qualification.elapsed().as_secs_f64();
    let pools = plan
        .segments
        .iter()
        .map(|s| {
            s.wire
                .rotation
                .as_ref()
                .map(|r| Ok((collection(&r.old_pool)?, collection(&r.new_pool)?)))
                .transpose()
        })
        .collect::<Result<Vec<_>, String>>()?;
    let extra = saved
        .iter()
        .zip(&metadata)
        .enumerate()
        .map(|(i, (tip, m))| {
            let spec = segment_spec(&plan.segments[i].wire.segment_spec)?;
            Ok(if let Some(rotation) = &plan.segments[i].wire.rotation {
                let (old_pool, new_pool) = pools[i].as_ref().ok_or("Live plan restore pools")?;
                CurrentMixedRestoreStep::SeatRotation {
                    spec,
                    old_pool,
                    new_pool,
                    shifts: rotation.shifts,
                    tip: tip.tip(&m.raw),
                }
            } else {
                CurrentMixedRestoreStep::Segment {
                    spec,
                    tip: tip.tip(&m.raw),
                }
            })
        })
        .collect::<Result<Vec<_>, String>>()?;
    let restore = Instant::now();
    let (mut session, _) = restore_campaign_owner(&plan.first, &pre.original, &prepared, extra)?;
    let restore_seconds = restore.elapsed().as_secs_f64();
    let mut started = command != "live-continue";
    require_live_prefix(&plan, &pre, &session, cursor, started)?;
    let retained_before=serde_json::to_value(session.status()).map_err(|e|e.to_string())?["segmentUsage"]["retainedBytes"].as_u64().ok_or("Live plan retained baseline missing")?;
    let mut new_segments = 0usize;
    let mut calls = Vec::new();
    let mut segment_tips = Vec::new();
    let mut failure = None;
    let mut stop;
    let mut native_calls = 0usize;
    let mut failed_timing = None;
    loop {
        if started
            && cursor.updates_in_segment == plan.segments[cursor.segment].wire.collections.len()
        {
            if cursor.total_updates == plan.total {
                stop = "declaredPlanComplete";
                break;
            }
            segment_tips.push(actual_tip(&session)?);
            cursor.segment += 1;
            cursor.updates_in_segment = 0;
            started = false;
        }
        // Every cooperative pause follows a genuine Updated and precedes any
        // next Start/reserve. The first invocation must complete at least one.
        if !calls.is_empty() {
            let retained =
                serde_json::to_value(session.status()).map_err(|e| e.to_string())?["segmentUsage"]
                    ["retainedBytes"]
                    .as_u64()
                    .ok_or("Live plan retained usage missing")?;
            stop = if calls.len() >= max_new_updates {
                "invocationUpdateLimit"
            } else if !started && new_segments >= plan.wire.max_invocation_segments {
                "invocationSegmentLimit"
            } else if began.elapsed().as_secs() >= plan.wire.max_invocation_wall_seconds {
                "invocationWallLimit"
            } else if retained
                .checked_sub(retained_before)
                .ok_or("Live plan retained ledger decreased")?
                >= plan.wire.max_invocation_new_retained_bytes
            {
                "invocationRetainedLimit"
            } else {
                "running"
            };
            if stop != "running" {
                if !started {
                    cursor.segment -= 1;
                    cursor.updates_in_segment =
                        plan.segments[cursor.segment].wire.collections.len();
                }
                break;
            }
        }
        if !started {
            session = start_segment(&plan, cursor.segment, session, &output)?;
            started = true;
            new_segments += 1;
            require_live_prefix(&plan, &pre, &session, cursor, true)?;
            let value = serde_json::to_value(session.status()).map_err(|e| e.to_string())?;
            if number(&value["segmentUsage"], "index")?
                != add(pre.inherited_index, cursor.segment as u64 + 1)?
                || session.sequence != 1
            {
                return Err("Live plan Start index/head differs".into());
            }
        }
        let prior = session.status();
        let mut timing = CurrentMixedAttemptTimings::default();
        native_calls += 1;
        let result = session.update_whole_four_profiled(
            collection(&plan.segments[cursor.segment].wire.collections[cursor.updates_in_segment])?,
            &mut timing,
        );
        match result {
            Ok(status) => {
                if let Err(error) = plan.first.require_live_seat_baseline(&session) {
                    stop = "nonUpdatedBaselineChanged";
                    failure = Some(bounded_error(&error));
                    failed_timing = Some(timing);
                    break;
                }
                if status.phase != Phase::Ready
                    || status.unknown_work_possible
                    || status.outcome.as_deref() != Some("updated")
                    || status.mixed_update_count != add(prior.mixed_update_count, 1)?
                    || status.imported_update_count != prior.imported_update_count
                    || status.attempts != add(prior.attempts, 1)?
                    || !timing.call_observed
                    || timing.head_sha256_before != prior.head_sha256
                    || timing.head_sha256_after != status.head_sha256
                    || timing.attempts_before != prior.attempts
                    || timing.attempts_after != status.attempts
                    || timing.new_resolution != Some(CurrentMixedOwnResolution::Updated)
                {
                    stop = "failedNoChangeOrUnknown";
                    failure =
                        Some("Live plan call did not return a genuine exact Updated".to_string());
                    failed_timing = Some(timing);
                    break;
                }
                cursor.updates_in_segment += 1;
                cursor.total_updates += 1;
                if let Err(error) = require_live_prefix(&plan, &pre, &session, cursor, true) {
                    stop = "sourcePrefixOrBaselineChanged";
                    failure = Some(bounded_error(&error));
                    failed_timing = Some(timing);
                    break;
                }
                let name = format!("call-{:016x}.json", cursor.total_updates - 1);
                let raw = bytes(
                    &serde_json::json!({"schema":"tzolkin-current-mixed-live-call-v1","planRawSha256":plan.raw_sha256,
                    "totalUpdates":cursor.total_updates,"cursor":cursor,"headSha256Before":prior.head_sha256,"status":status,
                    "cumulativeOwnCounters":own_rows(session.cumulative_own_decisions()?),"timings":timing}),
                    CALL_CAP,
                )?;
                publish_live(
                    &output,
                    &name,
                    &raw,
                    plan.wire.max_wrapper_bytes,
                    &mut wrapper_used,
                )?;
                calls.push(file_pin(&name, &raw));
            }
            Err(error) => {
                stop = "errorOrUnknownWork";
                failure = Some(bounded_error(&error));
                failed_timing = Some(timing);
                break;
            }
        }
    }
    plan.require_pins()?;
    let status = session.status();
    let complete = stop == "declaredPlanComplete" && failure.is_none();
    let mut deployment_pin = None;
    if failure.is_none() {
        require_live_prefix(&plan, &pre, &session, cursor, true)?;
        session.require_full_archives()?;
        if let Some(name) = deployment {
            let raw = crate::public_deployment::export_current_mixed(&session.current_policy()?)?;
            publish_live(
                &output,
                name,
                &raw,
                plan.wire.max_wrapper_bytes,
                &mut wrapper_used,
            )?;
            deployment_pin = Some(file_pin(name, &raw));
        }
    }
    let report_raw = bytes(
        &serde_json::json!({"schema":"tzolkin-current-mixed-live-report-v1","planRawSha256":plan.raw_sha256,
        "command":command,"cursorBefore":before,"cursorAfter":cursor,"complete":complete,"stopReason":stop,"error":failure,
        "nativeCallsThisInvocation":native_calls,"successfulReturnedCalls":calls.len(),"failedCallTiming":failed_timing,
        "newSegments":new_segments,"qualificationCalls":1,"qualificationSeconds":qualification_seconds,
        "restoreSeconds":restore_seconds,"invocationWallSeconds":began.elapsed().as_secs_f64(),"status":status,
        "cumulativeOwnCounters":own_rows(session.cumulative_own_decisions()?),"allDeclaredFamiliesRemainSpent":true,
        "processClosureAuthenticated":false,"freshTestAdmission":false,"strengthOrAdoptionAdmission":false}),
        HEAD_CAP,
    )?;
    publish_live(
        &output,
        report,
        &report_raw,
        plan.wire.max_wrapper_bytes,
        &mut wrapper_used,
    )?;
    let tip = actual_tip(&session)?;
    if segment_tips
        .last()
        .is_some_and(|t: &SavedTip| t.directory == tip.directory)
    {
        segment_tips.pop();
    }
    segment_tips.push(tip.clone());
    publish_live(
        &output,
        &live_name(before.total_updates, "close"),
        &bytes(
            &LiveClose {
                schema: "tzolkin-current-mixed-live-close-v1".into(),
                plan_raw_sha256: plan.raw_sha256,
                before,
                after: cursor,
                tip,
                segment_tips,
                stop_reason: stop.into(),
                report: file_pin(report, &report_raw),
                calls,
                deployment: deployment_pin,
            },
            HEAD_CAP,
        )?,
        plan.wire.max_wrapper_bytes,
        &mut wrapper_used,
    )?;
    Ok(if complete {
        0
    } else if failure.is_none() {
        2
    } else {
        1
    })
}

#[cfg(test)]
pub(super) fn metadata_boundaries(original: &CampaignWire, next_spec: &Value) {
    // Synthetic metadata only; the original/null inputs cannot admit an owner.
    let pin = original.checkpoint.clone();
    let segment = SegmentPlan {
        schema: SEGMENT_PLAN_SCHEMA.into(),
        segment_spec: next_spec.clone(),
        pool: Value::Null,
        rotation: None,
        collections: vec![Value::Null, Value::Null],
    };
    let raw = encoded(original).unwrap();
    let plan = CurrentMixedLivePlanSpec {
        wire: LiveWire {
            schema: LIVE_SCHEMA.into(),
            checkpoint: pin.clone(),
            dataset_manifest: pin.clone(),
            bootstrap: original.bootstrap.clone(),
            original_mixed: original.original_mixed.clone(),
            initial: original.initial.clone(),
            steps: original.steps.clone(),
            exclusions: original.exclusions.clone(),
            non_updated_baseline: [[0; 5]; 3],
            segments: vec![pin.clone(), pin],
            max_aggregate_input_bytes: INPUT_CAP,
            max_wrapper_bytes: WRAPPER_CAP,
            max_invocation_updates: 4,
            max_invocation_segments: 2,
            max_invocation_wall_seconds: 60,
            max_invocation_new_retained_bytes: 1024,
        },
        raw: Vec::new(),
        raw_sha256: "a".repeat(64),
        segments: vec![
            PlannedSegment {
                wire: segment.clone(),
                raw: Vec::new(),
            },
            PlannedSegment {
                wire: segment,
                raw: Vec::new(),
            },
        ],
        first: CurrentMixedCampaignSpec {
            wire: original.clone(),
            raw: raw.clone(),
            raw_sha256: sha(&raw),
        },
        total: 4,
    };
    assert!(CurrentMixedLivePlanSpec::from_json(&raw, &sha(&raw)).is_err());
    assert!(
        CurrentMixedCampaignSpec::from_json(
            &encoded(&plan.wire).unwrap(),
            &sha(&encoded(&plan.wire).unwrap())
        )
        .is_err()
    );
    let mut cursor = Cursor {
        segment: 0,
        updates_in_segment: 2,
        total_updates: 2,
    };
    cursor.require(&plan).unwrap(); // Exact boundary remains in the closed old segment.
    cursor.segment = 1;
    cursor.updates_in_segment = 0;
    assert!(cursor.require(&plan).is_err()); // Unclosed new Start cannot be a pause.
    cursor.updates_in_segment = 1;
    cursor.total_updates = 3;
    cursor.require(&plan).unwrap();
    cursor.total_updates = 4;
    assert!(cursor.require(&plan).is_err());
    let before = Cursor::initial();
    let intent = LiveIntent {
        schema: "tzolkin-current-mixed-live-intent-v1".into(),
        plan_raw_sha256: plan.raw_sha256.clone(),
        before,
        max_new_updates: 2,
    };
    let tip = original.initial.tip.clone();
    let mut close = LiveClose {
        schema: "tzolkin-current-mixed-live-close-v1".into(),
        plan_raw_sha256: plan.raw_sha256.clone(),
        before,
        after: Cursor {
            segment: 0,
            updates_in_segment: 1,
            total_updates: 1,
        },
        tip: tip.clone(),
        segment_tips: vec![tip],
        stop_reason: "invocationUpdateLimit".into(),
        report: file_pin("report.json", b"metadata"),
        calls: vec![file_pin("call-0000000000000000.json", b"metadata")],
        deployment: None,
    };
    require_closed_prefix(&plan, before, &intent, &close).unwrap();
    for stop in ["noChange", "failed", "unknownWork", "declaredPlanComplete"] {
        close.stop_reason = stop.into();
        assert!(require_closed_prefix(&plan, before, &intent, &close).is_err());
    }
    close.stop_reason = "invocationWallLimit".into();
    require_closed_prefix(&plan, before, &intent, &close).unwrap();
    close.segment_tips.clear();
    assert!(require_closed_prefix(&plan, before, &intent, &close).is_err());
    for leaf in [
        "segments",
        "LIVE-plan.json",
        "live-0000-close.json",
        "Call-00.json",
        "NUL.json",
    ] {
        assert!(require_leaf(leaf).is_err());
    }
    require_leaf("report-01.json").unwrap();
    let run = run_spec(&next_spec["runSpec"]).unwrap();
    let mut contract = next_spec["globalContract"].clone();
    contract["maxAttempts"] = serde_json::json!(1000);
    contract["maxSuccessfulUpdates"] = serde_json::json!(1000);
    contract["maxDiskBytes"] = serde_json::json!(8589934592u64);
    contract["maxStoredRecordBytes"] = serde_json::json!(8589934592u64);
    contract["maxMetadataBytes"] = serde_json::json!(1073741824);
    let mut budget = Budget {
        attempts: 900,
        updates: 900,
        work: Work::default(),
        raw: 0,
        stored: 0,
        metadata: 0,
    };
    budget.reserve(&contract, &run).unwrap();
    assert!(
        budget
            .reserve(&contract, &run)
            .unwrap_err()
            .contains("maxAttempts")
    );
    assert!(add(u64::MAX, 1).is_err());
    // Wrapper capacity fails before publishing a new file.
    let mut used = WRAPPER_CAP;
    assert!(
        publish_live(
            Path::new("not-created"),
            "report.json",
            b"x",
            WRAPPER_CAP,
            &mut used
        )
        .is_err()
    );
    assert_eq!(used, WRAPPER_CAP);
}
