//! A fixed-pool, externally reserved campaign over the genuine durable owner.
//! The campaign layer adds no optimizer, record, artifact or owner wire format.
use super::*;
use crate::kernel::Kernel;
use crate::public_native::PreparedPublicPolicy;
use crate::public_rl_mixed_session::{ImportedMixedRlParent, require_output_outside_directory};
use crate::public_rl_mixed_training_session::MixedOneBatchSpec;
use crate::public_rl_paired_session::PairedRunSpec;
use serde_json::Value;

const CAMPAIGN_SCHEMA: &str = "tzolkin-current-mixed-campaign-spec-v1";
const CAMPAIGN_CAP: usize = 1024 * 1024;
const EXCLUSION_CAP: usize = 16 * 1024 * 1024;
const MAX_EXCLUSIONS: usize = 200_000;
const PREFLIGHT_CAP: usize = 256 * 1024 * 1024;

#[derive(Clone, Deserialize, Serialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct InputPin {
    path: PathBuf,
    bytes: usize,
    sha256: String,
}
impl InputPin {
    fn read(&self, cap: usize) -> Result<Vec<u8>, String> {
        if self.bytes == 0 || self.bytes > cap || !checkpoint_digest(&self.sha256) {
            return Err("Campaign input pin shape/cap".into());
        }
        let raw = read_bounded(&local(&self.path, false)?, self.bytes)?;
        if raw.len() != self.bytes || sha(&raw) != self.sha256 {
            return Err("Campaign input raw bytes/SHA256 differ".into());
        }
        Ok(raw)
    }
}

#[derive(Clone, Deserialize, Serialize)]
#[serde(tag = "kind", rename_all = "camelCase", deny_unknown_fields)]
enum Bootstrap {
    Count1 {
        checkpoint: InputPin,
    },
    Paired {
        spec: Value,
        head: InputPin,
        directory: PathBuf,
    },
}
#[derive(Clone, Deserialize, Serialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct SavedTip {
    head: InputPin,
    directory: PathBuf,
}
impl SavedTip {
    fn tip<'s>(&'s self, raw: &'s [u8]) -> CurrentMixedRestoreTip<'s> {
        CurrentMixedRestoreTip {
            head_raw: raw,
            expected_sha256: &self.head.sha256,
            directory: &self.directory,
        }
    }
}
#[derive(Clone, Deserialize, Serialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct OriginalMixed {
    spec: Value,
    tip: SavedTip,
}
#[derive(Clone, Deserialize, Serialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct InitialDurable {
    run_spec: Value,
    pool: Value,
    tip: SavedTip,
}
#[derive(Clone, Deserialize, Serialize)]
#[serde(tag = "kind", rename_all = "camelCase", deny_unknown_fields)]
enum SavedStep {
    Segment {
        spec: Value,
        tip: SavedTip,
    },
    BudgetAmendment {
        spec: Value,
        tip: SavedTip,
    },
    GuardedPoolFork {
        spec: Value,
        old_pool: Value,
        new_pool: Value,
        tip: SavedTip,
    },
}
impl SavedStep {
    fn spec(&self) -> &Value {
        match self {
            Self::Segment { spec, .. }
            | Self::BudgetAmendment { spec, .. }
            | Self::GuardedPoolFork { spec, .. } => spec,
        }
    }
    fn tip(&self) -> &SavedTip {
        match self {
            Self::Segment { tip, .. }
            | Self::BudgetAmendment { tip, .. }
            | Self::GuardedPoolFork { tip, .. } => tip,
        }
    }
}

#[derive(Clone, Deserialize, Serialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct Exclusions {
    file: InputPin,
    count: usize,
    family_digest: String,
}
#[derive(Clone, Deserialize, Serialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct CampaignWire {
    schema: String,
    checkpoint: InputPin,
    dataset_manifest: InputPin,
    bootstrap: Bootstrap,
    original_mixed: OriginalMixed,
    initial: InitialDurable,
    steps: Vec<SavedStep>,
    segment_spec: Value,
    amend_budget: bool,
    pool: Value,
    collections: Vec<Value>,
    exclusions: Exclusions,
}

/// Checked campaign metadata. It cannot deserialize or construct a policy owner.
pub struct CurrentMixedCampaignSpec {
    wire: CampaignWire,
    raw: Vec<u8>,
    raw_sha256: String,
}
fn encoded(value: &impl Serialize) -> Result<Vec<u8>, String> {
    serde_json::to_vec(value).map_err(|e| e.to_string())
}
fn collection(value: &Value) -> Result<CurrentMixedCollectionSpec, String> {
    CurrentMixedCollectionSpec::from_json(&encoded(value)?)
}
fn run_spec(value: &Value) -> Result<CurrentMixedRunSpec, String> {
    CurrentMixedRunSpec::from_json(&encoded(value)?)
}
fn segment_spec(value: &Value) -> Result<CurrentMixedSegmentSpec, String> {
    CurrentMixedSegmentSpec::from_json(&encoded(value)?)
}
impl CurrentMixedCampaignSpec {
    pub fn from_json(raw: &[u8], expected_sha256: &str) -> Result<Self, String> {
        if !checkpoint_digest(expected_sha256) || sha(raw) != expected_sha256 {
            return Err("Campaign expected raw SHA256 differs".into());
        }
        let wire: CampaignWire = closed(raw, CAMPAIGN_CAP)?;
        if wire.schema != CAMPAIGN_SCHEMA
            || wire.steps.len() + 2 > segment::MAX_SEGMENTS
            || !(1..=MAX_ATTEMPTS as usize).contains(&wire.collections.len())
        {
            return Err("Campaign schema/finite segment/cohort limits".into());
        }
        run_spec(&wire.initial.run_spec)?;
        MixedOneBatchSpec::from_json(&encoded(&wire.original_mixed.spec)?)?;
        if let Bootstrap::Paired { spec, .. } = &wire.bootstrap {
            PairedRunSpec::from_json(&encoded(spec)?)?;
        }
        let initial_pool = collection(&wire.initial.pool)?;
        let mut pool_checksum = initial_pool.pool_checksum()?;
        for step in &wire.steps {
            segment_spec(step.spec())?;
            if let SavedStep::GuardedPoolFork {
                old_pool, new_pool, ..
            } = step
            {
                if collection(old_pool)?.pool_checksum()? != pool_checksum {
                    return Err("Campaign saved fork old pool differs".into());
                }
                pool_checksum = collection(new_pool)?.pool_checksum()?;
            }
        }
        if collection(&wire.pool)?.pool_checksum()? != pool_checksum {
            return Err("Campaign fixed pool differs from the saved lineage".into());
        }
        segment_spec(&wire.segment_spec)?;
        let next = run_spec(&wire.segment_spec["runSpec"])?;
        if wire.collections.len() as u64 > next.max_attempts
            || wire.collections.len() as u64 > next.max_successful_updates
        {
            return Err("Campaign complete chunk exceeds local attempt/update capacity".into());
        }
        for value in &wire.collections {
            if collection(value)?.pool_checksum()? != pool_checksum {
                return Err("Campaign cohort changes the fixed pool".into());
            }
        }
        Ok(Self {
            wire,
            raw: raw.to_vec(),
            raw_sha256: expected_sha256.into(),
        })
    }
}

struct MetadataTip {
    raw: Vec<u8>,
    progress: Progress,
    reservations: Vec<Reservation>,
    resolved_heads: Vec<String>,
    binding: Option<segment::SegmentBinding>,
    entry: Entry,
    directory: PathBuf,
}
/// Same journal reducer, without records, gradients or a model. These claims are
/// only a pre-load refusal gate; restore_chain still proves the actual owner.
fn metadata_tip(
    tip: &SavedTip,
    expected_spec: &Value,
    pool: &str,
    is_segment: bool,
) -> Result<MetadataTip, String> {
    let spec = if is_segment {
        run_spec(&expected_spec["runSpec"])?
    } else {
        run_spec(expected_spec)?
    };
    let directory = local(&tip.directory, false)?;
    let raw = tip.head.read(HEAD_CAP)?;
    let latest: Entry = closed(&raw, HEAD_CAP)?;
    if local(&tip.head.path, false)?
        != directory.join(format!("head-{:016x}.json", latest.sequence))
        || latest.sequence == 0
        || latest.sequence > spec.max_journal_entries
    {
        return Err("Campaign latest native head path/sequence/metadata capacity".into());
    }
    let native_spec = read_bounded(&directory.join("run-spec.json"), SPEC_CAP)?;
    let expected = if is_segment {
        serde_json::to_value(segment_spec(expected_spec)?).map_err(|e| e.to_string())?
    } else {
        serde_json::to_value(&spec).map_err(|e| e.to_string())?
    };
    let expected_run = if is_segment {
        &expected["runSpec"]
    } else {
        &expected
    };
    if unique_json(&native_spec, SPEC_CAP)? != *expected_run {
        return Err("Campaign native run specification differs".into());
    }
    let mut previous = String::new();
    let mut progress = None;
    let mut reservations = Vec::new();
    let mut resolved_heads = Vec::new();
    let mut first_binding = None;
    let mut references = BTreeMap::<u64, Vec<FilePin>>::new();
    let mut total_bytes = 0usize;
    for sequence in 1..=latest.sequence {
        let bytes = read_bounded(
            &directory.join(format!("head-{sequence:016x}.json")),
            HEAD_CAP,
        )?;
        total_bytes = total_bytes
            .checked_add(bytes.len())
            .ok_or("Campaign metadata overflow")?;
        if total_bytes > PREFLIGHT_CAP {
            return Err("Campaign metadata byte capacity".into());
        }
        let entry: Entry = closed(&bytes, HEAD_CAP)?;
        if entry.sequence != sequence
            || entry.previous != previous
            || entry.spec != spec.checksum()?
            || entry.root != latest.root
            || entry.pool != pool
            || entry.schema != latest.schema
        {
            return Err("Campaign native metadata chain differs".into());
        }
        if sequence == 1 {
            let (parent, prefix, next, consumed, binding) = match &entry.event {
                Event::Start {
                    parent,
                    prefix,
                    next,
                    consumed,
                    run_spec,
                } => {
                    if latest.schema != spec.journal_schema()
                        || run_spec.sha256 != sha(&native_spec)
                        || run_spec.bytes != native_spec.len()
                    {
                        return Err("Campaign native Start pin/schema".into());
                    }
                    (parent.as_ref(), *prefix, *next, consumed, None)
                }
                Event::SegmentStart { binding, run_spec } => {
                    if latest.schema != SEGMENT_JOURNAL_SCHEMA
                        || run_spec.sha256 != sha(&native_spec)
                        || run_spec.bytes != native_spec.len()
                    {
                        return Err("Campaign native SegmentStart pin/schema".into());
                    }
                    if serde_json::to_value(binding).map_err(|e| e.to_string())?["contract"]
                        != expected["globalContract"]
                    {
                        return Err("Campaign original segment contract differs".into());
                    }
                    (
                        &binding.parent,
                        binding.prefix,
                        binding.next,
                        &binding.consumed,
                        Some(binding.as_ref().clone()),
                    )
                }
                _ => return Err("Campaign journal has no original Start".into()),
            };
            if consumed.len() > MAX_FAMILIES
                || consumed.windows(2).any(|p| p[0] >= p[1])
                || consumed.iter().any(|s| !checkpoint_digest(s))
            {
                return Err("Campaign inherited family metadata".into());
            }
            parent.validate()?;
            let mut p =
                Progress::initial(parent, prefix, next, &consumed.iter().cloned().collect());
            if let Some(binding) = binding {
                p.base = parent.mixed_update_count();
                first_binding = Some(binding.clone());
                p.segment_start = Some(binding);
            }
            progress = Some(p);
        }
        let p = progress.as_mut().ok_or("Campaign native Start missing")?;
        let reference = match &entry.event {
            Event::ReserveWhole4 { reservation: r } => {
                Some((r.attempt, vec![r.collection_spec.clone(), r.plan.clone()]))
            }
            Event::CollectReturn {
                record: Some(record),
                ..
            } => Some((
                p.active
                    .as_ref()
                    .ok_or("Campaign active reservation missing")?
                    .attempt,
                vec![record.file().clone()],
            )),
            Event::GradientStart { cohort, .. } => Some((
                p.active
                    .as_ref()
                    .ok_or("Campaign active reservation missing")?
                    .attempt,
                vec![cohort.clone()],
            )),
            Event::Resolve {
                resolution:
                    Resolution::Updated {
                        report, artifact, ..
                    },
            } => Some((
                p.active
                    .as_ref()
                    .ok_or("Campaign active reservation missing")?
                    .attempt,
                vec![report.clone(), artifact.clone()],
            )),
            Event::Resolve {
                resolution: Resolution::NoChange { report, .. },
            } => Some((
                p.active
                    .as_ref()
                    .ok_or("Campaign active reservation missing")?
                    .attempt,
                vec![report.clone()],
            )),
            _ => None,
        };
        if let Some((attempt, pins)) = reference {
            let refs = references.entry(attempt).or_default();
            for pin in pins {
                if refs.iter().any(|p| p.name == pin.name) {
                    return Err("Campaign native repeated archive reference".into());
                }
                let path = local(
                    &directory
                        .join(format!("batch-{attempt:016x}"))
                        .join(&pin.name),
                    false,
                )?;
                if !fs::symlink_metadata(path)
                    .map_err(|e| e.to_string())?
                    .is_file()
                {
                    return Err("Campaign native archive reference is not a regular file".into());
                }
                refs.push(pin);
            }
        }
        if let Event::ReserveWhole4 { reservation } = &entry.event {
            reservations.push(reservation.as_ref().clone());
        }
        p.apply(&entry.event, &spec)?;
        previous = sha(&bytes);
        if matches!(
            entry.event,
            Event::Resolve {
                resolution: Resolution::Updated { .. }
            }
        ) {
            resolved_heads.push(previous.clone());
        }
    }
    if previous != tip.head.sha256 {
        return Err("Campaign latest head differs from complete chain".into());
    }
    let spec_pin = file_pin("run-spec.json", &native_spec);
    let head_pin = file_pin(&format!("head-{:016x}.json", latest.sequence), &raw);
    restore::require_archive_layout(&ArchiveView {
        directory: &directory,
        spec: &spec,
        journal_schema: if is_segment {
            SEGMENT_JOURNAL_SCHEMA
        } else {
            spec.journal_schema()
        },
        root_checksum: &latest.root,
        pool,
        sequence: latest.sequence,
        head: Some(&head_pin),
        spec_pin: Some(&spec_pin),
        references: &references,
        output_bytes: 0,
        metadata_bytes: 0,
        raw_record_bytes: 0,
        stored_record_bytes: 0,
    })?;
    Ok(MetadataTip {
        raw,
        progress: progress.ok_or("Campaign native Start missing")?,
        reservations,
        resolved_heads,
        binding: first_binding,
        entry: latest,
        directory,
    })
}

fn successor_metadata(
    previous: &MetadataTip,
    next: &MetadataTip,
    fork: bool,
) -> Result<(), String> {
    let binding = next
        .binding
        .as_ref()
        .ok_or("Campaign successor native binding missing")?;
    let value = serde_json::to_value(binding).map_err(|e| e.to_string())?;
    let predecessor: PathBuf =
        serde_json::from_value(value["predecessorDirectory"].clone()).map_err(|e| e.to_string())?;
    if local(&predecessor, false)? != previous.directory
        || value["predecessorSchema"] != previous.entry.schema
        || value["predecessorRoot"] != previous.entry.root
        || value["predecessorSpec"] != previous.entry.spec
        || value["predecessorHead"] != sha(&previous.raw)
        || value["predecessorSequence"].as_u64() != Some(previous.entry.sequence)
        || binding.parent.artifact_checksum() != previous.progress.current_checksum
        || binding.parent.mixed_update_count() != previous.progress.parent_count()?
        || binding.parent.imported_update_count() != previous.progress.imported
        || binding.prefix != previous.progress.prefix
        || binding.next != previous.progress.next
        || binding.consumed.iter().cloned().collect::<BTreeSet<_>>() != previous.progress.consumed
        || if fork {
            previous.progress.phase != Phase::Stopped
                || previous.progress.unknown_work
                || previous.progress.outcome.as_deref() != Some("failed")
        } else {
            previous.progress.phase != Phase::Ready
                || previous.progress.unknown_work
                || !matches!(
                    previous.progress.outcome.as_deref(),
                    Some("updated" | "noChange")
                )
        }
    {
        return Err("Campaign successor ancestry/baseline metadata differs".into());
    }
    Ok(())
}

struct Preflight {
    initial: MetadataTip,
    steps: Vec<MetadataTip>,
    exclusions: BTreeSet<String>,
    configurations: Vec<[Config; 4]>,
    families: Vec<[String; 4]>,
}
impl CurrentMixedCampaignSpec {
    fn require_input_pins(&self) -> Result<(), String> {
        self.wire.checkpoint.read(16 * 1024 * 1024)?;
        self.wire.dataset_manifest.read(16 * 1024 * 1024)?;
        self.wire.exclusions.file.read(EXCLUSION_CAP)?;
        match &self.wire.bootstrap {
            Bootstrap::Count1 { checkpoint } => {
                checkpoint.read(crate::public_rl_session::MAX_CHECKPOINT_BYTES)?;
            }
            Bootstrap::Paired { head, .. } => {
                head.read(crate::public_rl_paired_session::MAX_CHECKPOINT_BYTES)?;
            }
        }
        self.wire
            .original_mixed
            .tip
            .head
            .read(crate::public_rl_mixed_training_session::MAX_HEAD_BYTES)?;
        self.wire.initial.tip.head.read(HEAD_CAP)?;
        for step in &self.wire.steps {
            step.tip().head.read(HEAD_CAP)?;
        }
        Ok(())
    }
    fn preflight(&self) -> Result<Preflight, String> {
        self.wire.checkpoint.read(16 * 1024 * 1024)?;
        self.wire.dataset_manifest.read(16 * 1024 * 1024)?;
        if self
            .wire
            .dataset_manifest
            .path
            .file_name()
            .is_none_or(|n| n != "manifest.json")
        {
            return Err("Campaign dataset manifest requires its original leaf".into());
        }
        match &self.wire.bootstrap {
            Bootstrap::Count1 { checkpoint } => {
                checkpoint.read(crate::public_rl_session::MAX_CHECKPOINT_BYTES)?;
            }
            Bootstrap::Paired {
                head, directory, ..
            } => {
                head.read(crate::public_rl_paired_session::MAX_CHECKPOINT_BYTES)?;
                if local(&head.path, false)?.parent() != Some(local(directory, false)?.as_path()) {
                    return Err("Campaign paired head directory differs".into());
                }
            }
        }
        self.wire
            .original_mixed
            .tip
            .head
            .read(crate::public_rl_mixed_training_session::MAX_HEAD_BYTES)?;
        if local(&self.wire.original_mixed.tip.head.path, false)?.parent()
            != Some(local(&self.wire.original_mixed.tip.directory, false)?.as_path())
        {
            return Err("Campaign original mixed head directory differs".into());
        }
        let mut pool = collection(&self.wire.initial.pool)?.pool_checksum()?;
        let initial = metadata_tip(
            &self.wire.initial.tip,
            &self.wire.initial.run_spec,
            &pool,
            false,
        )?;
        let mut steps = Vec::new();
        for step in &self.wire.steps {
            if let SavedStep::GuardedPoolFork { new_pool, .. } = step {
                pool = collection(new_pool)?.pool_checksum()?;
            }
            let metadata = metadata_tip(step.tip(), step.spec(), &pool, true)?;
            successor_metadata(
                steps.last().unwrap_or(&initial),
                &metadata,
                matches!(step, SavedStep::GuardedPoolFork { .. }),
            )?;
            steps.push(metadata);
        }
        let latest = steps.last().unwrap_or(&initial);
        if latest.progress.phase != Phase::Ready
            || latest.progress.unknown_work
            || latest.progress.outcome.as_deref() != Some("updated")
        {
            return Err("Campaign predecessor must be the resolved updated Ready tip".into());
        }
        let exclusions_raw = self.wire.exclusions.file.read(EXCLUSION_CAP)?;
        let families: Vec<String> = closed(&exclusions_raw, EXCLUSION_CAP)?;
        if families.len() != self.wire.exclusions.count
            || families.len() > MAX_EXCLUSIONS
            || families.iter().any(|s| !checkpoint_digest(s))
            || families.windows(2).any(|p| p[0] >= p[1])
            || hash(b"tzolkin-current-mixed-campaign-exclusions-v1\0", &families)?
                != self.wire.exclusions.family_digest
        {
            return Err("Campaign full external exclusion declaration differs".into());
        }
        let exclusions = families.into_iter().collect::<BTreeSet<_>>();
        let mut used = latest.progress.consumed.clone();
        let mut configurations = Vec::new();
        let mut families = Vec::new();
        let mut next = latest.progress.next;
        for value in &self.wire.collections {
            let configs = collection(value)?
                .inner
                .configs()?
                .each_ref()
                .map(Config::from_config);
            let f = configs
                .each_ref()
                .map(|c| seed_family_id(c.environment_seed));
            for (i, (c, family)) in configs.iter().zip(&f).enumerate() {
                if c.sampling_seed != format!("{:016x}", latest.progress.prefix[0])
                    || c.replicate_ordinal != format!("{:016x}", latest.progress.prefix[1])
                    || c.episode_ordinal
                        != format!(
                            "{:016x}",
                            next.checked_add(i as u64)
                                .ok_or("Campaign stream overflow")?
                        )
                    || split_for_family(family)? != DatasetSplit::Train
                    || exclusions.contains(family)
                    || !used.insert(family.clone())
                {
                    return Err(
                        "Campaign all Train families/streams must be new and disjoint".into(),
                    );
                }
            }
            next = next.checked_add(4).ok_or("Campaign full stream overflow")?;
            configurations.push(configs);
            families.push(f);
        }
        let next_spec = run_spec(&self.wire.segment_spec["runSpec"])?;
        if used.len() > MAX_FAMILIES || used.len() > next_spec.max_families {
            return Err(
                "Campaign whole declared chunk exceeds genuine lineage family capacity".into(),
            );
        }
        Ok(Preflight {
            initial,
            steps,
            exclusions,
            configurations,
            families,
        })
    }
    fn protect_output(&self, output: &Path) -> Result<(), String> {
        let mut directories = vec![
            self.wire.original_mixed.tip.directory.as_path(),
            self.wire.initial.tip.directory.as_path(),
            self.wire
                .dataset_manifest
                .path
                .parent()
                .ok_or("Campaign dataset directory")?,
        ];
        directories.extend(self.wire.steps.iter().map(|s| s.tip().directory.as_path()));
        if let Bootstrap::Paired { directory, .. } = &self.wire.bootstrap {
            directories.push(directory);
        }
        for directory in directories {
            require_output_outside_directory(directory, output)?;
        }
        Ok(())
    }
    fn reservation(&self, pre: &Preflight) -> Result<Vec<u8>, String> {
        bytes(
            &serde_json::json!({"schema":"tzolkin-current-mixed-campaign-reservation-v1",
            "campaignRawSha256":self.raw_sha256,"allDeclaredFamiliesRemainSpent":true,
            "configurations":pre.configurations,"families":pre.families,"externalExclusions":self.wire.exclusions,
            "producerAuthenticated":false,"freshTestAdmission":false,"strengthOrAdoptionAdmission":false}),
            CAMPAIGN_CAP,
        )
    }
}

fn check_prefix(
    pre: &Preflight,
    child: &MetadataTip,
    spec: &CurrentMixedCampaignSpec,
    directory: &Path,
) -> Result<usize, String> {
    if child.progress.phase != Phase::Ready
        || child.progress.unknown_work
        || child.progress.attempts != child.progress.successful
        || child.progress.attempts as usize > spec.wire.collections.len()
        || child.reservations.len() != child.progress.attempts as usize
        || child.resolved_heads.len() != child.progress.attempts as usize
        || child.progress.attempts > 0 && child.progress.outcome.as_deref() != Some("updated")
    {
        return Err("Campaign unresolved/failed/NoChange prefix cannot resume or recollect".into());
    }
    for (i, r) in child.reservations.iter().enumerate() {
        if r.configurations != pre.configurations[i]
            || r.families != pre.families[i]
            || read_pin(
                &directory.join(format!("batch-{:016x}", r.attempt)),
                &r.collection_spec,
                SPEC_CAP,
            )? != collection(&spec.wire.collections[i])?.inner.to_json()?
        {
            return Err(
                "Campaign saved native prefix differs from the complete reservation".into(),
            );
        }
    }
    Ok(child.progress.attempts as usize)
}

#[derive(Deserialize, Serialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct InvocationIntent {
    schema: String,
    campaign_raw_sha256: String,
    prefix_before: usize,
    max_new_attempts: usize,
}
#[derive(Deserialize, Serialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct InvocationClose {
    schema: String,
    campaign_raw_sha256: String,
    prefix_before: usize,
    prefix_after: usize,
    native_head_sha256: String,
    stop_reason: String,
    report: FilePin,
}
fn invocation_name(prefix: usize, suffix: &str) -> String {
    format!("invocation-{prefix:016x}-{suffix}.json")
}
fn leaf_name(name: &str) -> Result<(), String> {
    let folded = name.to_ascii_lowercase();
    let stem = folded.split('.').next().unwrap_or_default();
    let device = matches!(stem, "con" | "prn" | "aux" | "nul")
        || stem.len() == 4
            && (stem.starts_with("com") || stem.starts_with("lpt"))
            && matches!(stem.as_bytes()[3], b'1'..=b'9');
    if name.is_empty()
        || name.len() > 64
        || !name
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'.' | b'-'))
        || matches!(
            folded.as_str(),
            "." | ".." | "archive" | "campaign-spec.json" | "reservation.json"
        )
        || folded.starts_with("invocation-")
        || folded.ends_with(".tmp")
        || name.ends_with('.')
        || device
    {
        return Err("Campaign output requires a distinct simple leaf".into());
    }
    Ok(())
}
fn output_leaves(report: &str, deployment: Option<&str>) -> Result<(), String> {
    leaf_name(report)?;
    if let Some(name) = deployment {
        leaf_name(name)?;
        if name.eq_ignore_ascii_case(report) {
            return Err("Campaign report and deployment leaves alias".into());
        }
    }
    Ok(())
}
fn prior_invocations(
    output: &Path,
    campaign_raw_sha256: &str,
    child: &MetadataTip,
) -> Result<(), String> {
    // A crash between a native Updated publication and the wrapper return is
    // still unknown to this campaign. It cannot silently authorize a new suffix.
    if names(output, 4 * MAX_ATTEMPTS as usize + 8)?
        .iter()
        .any(|n| n.ends_with(".tmp"))
    {
        return Err("Campaign uncertain metadata publication refuses resume".into());
    }
    let completed = child.progress.attempts as usize;
    let mut prefix = 0usize;
    while prefix < completed {
        let intent: InvocationIntent = closed(
            &read_bounded(&output.join(invocation_name(prefix, "intent")), HEAD_CAP)?,
            HEAD_CAP,
        )?;
        let close: InvocationClose = closed(
            &read_bounded(&output.join(invocation_name(prefix, "close")), HEAD_CAP)?,
            HEAD_CAP,
        )?;
        if intent.schema != "tzolkin-current-mixed-campaign-invocation-intent-v1"
            || close.schema != "tzolkin-current-mixed-campaign-invocation-close-v1"
            || intent.campaign_raw_sha256 != campaign_raw_sha256
            || close.campaign_raw_sha256 != campaign_raw_sha256
            || intent.prefix_before != prefix
            || close.prefix_before != prefix
            || !(1..=MAX_ATTEMPTS as usize).contains(&intent.max_new_attempts)
            || close.prefix_after <= prefix
            || close.prefix_after > completed
            || close.prefix_after > prefix + intent.max_new_attempts
            || close.native_head_sha256 != child.resolved_heads[close.prefix_after - 1]
            || !matches!(
                close.stop_reason.as_str(),
                "invocationAttemptLimit" | "declaredChunkComplete"
            )
        {
            return Err(
                "Campaign previous invocation did not close a successful exact prefix".into(),
            );
        }
        leaf_name(&close.report.name)?;
        read_pin(output, &close.report, HEAD_CAP)?;
        prefix = close.prefix_after;
    }
    Ok(())
}

/// One finite invocation. `continue` publishes the whole chunk reservation before
/// any model/owner load. `resume` requires its exact successful prefix; unknown
/// work is never recollected. An external single-writer wall watchdog is required.
pub fn current_mixed_campaign(
    spec: CurrentMixedCampaignSpec,
    command: &str,
    output: &Path,
    head: Option<(&Path, &str)>,
    max_new_attempts: usize,
    report: &str,
    deployment: Option<&str>,
) -> Result<i32, String> {
    if !matches!(command, "continue" | "resume" | "inspect-tip")
        || max_new_attempts == 0
        || max_new_attempts > MAX_ATTEMPTS as usize
    {
        return Err("Campaign command/invocation attempt cap".into());
    }
    output_leaves(report, deployment)?;
    let began = Instant::now();
    let pre = spec.preflight()?;
    let output = local(output, command == "continue")?;
    spec.protect_output(&output)?;
    let archive = output.join("archive");
    let expected_reservation = spec.reservation(&pre)?;
    let child_spec = segment_spec(&spec.wire.segment_spec)?;
    let pool = collection(&spec.wire.pool)?;
    let child = if command == "continue" {
        if head.is_some() || output.exists() {
            return Err("Campaign continue output must be new and has no child head".into());
        }
        None
    } else {
        if read_bounded(&output.join("campaign-spec.json"), CAMPAIGN_CAP)? != spec.raw
            || read_bounded(&output.join("reservation.json"), CAMPAIGN_CAP)? != expected_reservation
        {
            return Err("Campaign original whole-chunk reservation differs".into());
        }
        let (path, digest) =
            head.ok_or("Campaign resume/inspect-tip requires pinned latest child head")?;
        let bytes = read_bounded(path, HEAD_CAP)?;
        let tip = SavedTip {
            head: InputPin {
                path: path.into(),
                bytes: bytes.len(),
                sha256: digest.into(),
            },
            directory: archive.clone(),
        };
        Some(metadata_tip(
            &tip,
            &spec.wire.segment_spec,
            &pool.pool_checksum()?,
            true,
        )?)
    };
    if let Some(child) = &child {
        successor_metadata(pre.steps.last().unwrap_or(&pre.initial), child, false)?;
    }
    let completed = child
        .as_ref()
        .map(|c| check_prefix(&pre, c, &spec, &archive))
        .transpose()?
        .unwrap_or(0);
    let report_path = output.join(report);
    if report_path.exists() {
        return Err("Campaign report output must be new".into());
    }
    if deployment.is_some_and(|name| output.join(name).exists()) {
        return Err("Campaign deployment output must be new and distinct".into());
    }
    if command == "inspect-tip" {
        publish_mixed_metadata(
            report_path.parent().ok_or("Campaign report parent")?,
            report_path
                .file_name()
                .and_then(|s| s.to_str())
                .ok_or("Campaign report leaf")?,
            &bytes(
                &serde_json::json!({"schema":"tzolkin-current-mixed-campaign-inspection-v1","campaignRawSha256":spec.raw_sha256,
                "completedPrefixClaims":completed,"declaredCohorts":spec.wire.collections.len(),"numericOwnerAdmission":false,
                "successfulWrapperPrefixEvaluated":false,"resumeReadinessEvaluated":false,
                "newReservations":0,"newOriginals":0,"newUpdates":0}),
                HEAD_CAP,
            )?,
        )?;
        return Ok(0);
    }
    if command == "continue" {
        fs::create_dir(&output).map_err(|e| e.to_string())?;
        publish_mixed_metadata(&output, "campaign-spec.json", &spec.raw)?;
        publish_mixed_metadata(&output, "reservation.json", &expected_reservation)?;
    } else {
        prior_invocations(
            &output,
            &spec.raw_sha256,
            child.as_ref().ok_or("Campaign saved child missing")?,
        )?;
    }
    let intent_name = invocation_name(completed, "intent");
    publish_mixed_metadata(
        &output,
        &intent_name,
        &bytes(
            &InvocationIntent {
                schema: "tzolkin-current-mixed-campaign-invocation-intent-v1".into(),
                campaign_raw_sha256: spec.raw_sha256.clone(),
                prefix_before: completed,
                max_new_attempts,
            },
            HEAD_CAP,
        )?,
    )?;
    let qualification = Instant::now();
    let prepared = PreparedPublicPolicy::load(
        &spec.wire.checkpoint.path,
        spec.wire
            .dataset_manifest
            .path
            .parent()
            .ok_or("Campaign dataset directory")?,
        Kernel::Scalar,
    )?;
    spec.require_input_pins()?;
    let qualification_seconds = qualification.elapsed().as_secs_f64();
    let parent = match &spec.wire.bootstrap {
        Bootstrap::Count1 { checkpoint } => ImportedMixedRlParent::from_count1(
            &prepared,
            &checkpoint.read(crate::public_rl_session::MAX_CHECKPOINT_BYTES)?,
            &checkpoint.sha256,
        )?,
        Bootstrap::Paired {
            spec: paired,
            head,
            directory,
        } => ImportedMixedRlParent::from_paired(
            &prepared,
            PairedRunSpec::from_json(&encoded(paired)?)?,
            &head.read(crate::public_rl_paired_session::MAX_CHECKPOINT_BYTES)?,
            &head.sha256,
            directory,
        )?,
    };
    let original = &spec.wire.original_mixed;
    let root = AdmittedCurrentMixedRoot::restore_one_batch(
        parent,
        &prepared,
        MixedOneBatchSpec::from_json(&encoded(&original.spec)?)?,
        &original
            .tip
            .head
            .read(crate::public_rl_mixed_training_session::MAX_HEAD_BYTES)?,
        &original.tip.head.sha256,
        &original.tip.directory,
    )?;
    let initial_pool = collection(&spec.wire.initial.pool)?;
    let fork_pools = spec
        .wire
        .steps
        .iter()
        .map(|step| match step {
            SavedStep::GuardedPoolFork {
                old_pool, new_pool, ..
            } => Ok(Some((collection(old_pool)?, collection(new_pool)?))),
            _ => Ok(None),
        })
        .collect::<Result<Vec<_>, String>>()?;
    let mut steps = spec
        .wire
        .steps
        .iter()
        .zip(&pre.steps)
        .zip(&fork_pools)
        .map(|((saved, metadata), pools)| {
            let tip = saved.tip().tip(&metadata.raw);
            let s = segment_spec(saved.spec())?;
            Ok(match saved {
                SavedStep::Segment { .. } => CurrentMixedRestoreStep::Segment { spec: s, tip },
                SavedStep::BudgetAmendment { .. } => {
                    CurrentMixedRestoreStep::BudgetAmendment { spec: s, tip }
                }
                SavedStep::GuardedPoolFork { .. } => {
                    let (old_pool, new_pool) =
                        pools.as_ref().ok_or("Campaign fork pool missing")?;
                    CurrentMixedRestoreStep::GuardedPoolFork {
                        spec: s,
                        old_pool,
                        new_pool,
                        tip,
                    }
                }
            })
        })
        .collect::<Result<Vec<_>, String>>()?;
    if let Some(child) = &child {
        let digest = head.ok_or("Campaign child tip missing")?.1;
        let tip = CurrentMixedRestoreTip {
            head_raw: &child.raw,
            expected_sha256: digest,
            directory: &archive,
        };
        steps.push(if spec.wire.amend_budget {
            CurrentMixedRestoreStep::BudgetAmendment {
                spec: child_spec.clone(),
                tip,
            }
        } else {
            CurrentMixedRestoreStep::Segment {
                spec: child_spec.clone(),
                tip,
            }
        });
    }
    let restored = Instant::now();
    let mut session = CurrentMixedTrainingSession::restore_chain(
        root,
        run_spec(&spec.wire.initial.run_spec)?,
        &initial_pool,
        spec.wire.initial.tip.tip(&pre.initial.raw),
        steps,
    )?;
    let restore_seconds = restored.elapsed().as_secs_f64();
    let latest = pre.steps.last().unwrap_or(&pre.initial);
    let expected_consumed = latest
        .progress
        .consumed
        .iter()
        .cloned()
        .chain(pre.families.iter().take(completed).flatten().cloned())
        .collect::<BTreeSet<_>>();
    if session.reserved_families() != &expected_consumed
        || session.progress.prefix != latest.progress.prefix
        || session.progress.next
            != latest
                .progress
                .next
                .checked_add(4 * completed as u64)
                .ok_or("Campaign owner stream overflow")?
        || session.progress.phase != Phase::Ready
        || session.progress.unknown_work
        || session.progress.outcome.as_deref() != Some("updated") && completed > 0
        || pre
            .families
            .iter()
            .skip(completed)
            .flatten()
            .any(|f| session.reserved_families().contains(f) || pre.exclusions.contains(f))
    {
        return Err("Campaign genuine restored owner differs from the reserved prefix".into());
    }
    if command == "continue" {
        let ready = session.into_ready_continuation()?;
        session = if spec.wire.amend_budget {
            CurrentMixedTrainingSession::new_segment_with_budget_amendment(
                ready, child_spec, &archive,
            )?
        } else {
            CurrentMixedTrainingSession::new_segment(ready, child_spec, &archive)?
        };
    }
    let mut timings = Vec::new();
    let mut failure = None;
    let mut stop = "invocationAttemptLimit";
    for value in spec
        .wire
        .collections
        .iter()
        .skip(completed)
        .take(max_new_attempts)
    {
        let mut timing = CurrentMixedAttemptTimings::default();
        match session.update_whole_four_profiled(collection(value)?, &mut timing) {
            Ok(status) => {
                timings.push(timing);
                if status.phase != Phase::Ready
                    || status.unknown_work_possible
                    || status.outcome.as_deref() != Some("updated")
                {
                    stop = if status.unknown_work_possible {
                        "unknownWork"
                    } else if status.outcome.as_deref() == Some("noChange") {
                        "noChange"
                    } else {
                        "failed"
                    };
                    break;
                }
            }
            Err(error) => {
                timings.push(timing);
                stop = "errorOrUnknownWork";
                failure = Some(bounded_error(&error));
                break;
            }
        }
    }
    let status = session.status();
    let complete = status.attempts as usize == spec.wire.collections.len()
        && status.phase == Phase::Ready
        && !status.unknown_work_possible
        && status.outcome.as_deref() == Some("updated")
        && failure.is_none();
    if complete {
        stop = "declaredChunkComplete";
    }
    spec.require_input_pins()?;
    if failure.is_none()
        && status.phase == Phase::Ready
        && !status.unknown_work_possible
        && status.outcome.as_deref() == Some("updated")
    {
        session.require_full_archives()?;
        if let Some(name) = deployment {
            let raw = crate::public_deployment::export_current_mixed(&session.current_policy()?)?;
            publish_mixed_metadata(&output, name, &raw)?;
        }
    }
    let report_raw = bytes(
        &serde_json::json!({"schema":"tzolkin-current-mixed-campaign-report-v1","command":command,
        "campaignRawSha256":spec.raw_sha256,"allDeclaredFamiliesRemainSpent":true,"declaredCohorts":spec.wire.collections.len(),
        "prefixCohortsBefore":completed,"nativeCallsThisInvocation":timings.len(),"complete":complete,"stopReason":stop,"error":failure,
        "qualification":{"calls":1,"seconds":qualification_seconds},"restoreSeconds":restore_seconds,
        "invocationWallSeconds":began.elapsed().as_secs_f64(),"status":status,"attemptTimings":timings,
        "phaseDecisionTimingAvailable":false,"freshTestAdmission":false,"strengthOrAdoptionAdmission":false}),
        HEAD_CAP,
    )?;
    publish_mixed_metadata(&output, report, &report_raw)?;
    publish_mixed_metadata(
        &output,
        &invocation_name(completed, "close"),
        &bytes(
            &InvocationClose {
                schema: "tzolkin-current-mixed-campaign-invocation-close-v1".into(),
                campaign_raw_sha256: spec.raw_sha256,
                prefix_before: completed,
                prefix_after: status.attempts as usize,
                native_head_sha256: status.head_sha256,
                stop_reason: stop.into(),
                report: file_pin(report, &report_raw),
            },
            HEAD_CAP,
        )?,
    )?;
    Ok(if complete {
        0
    } else if stop == "invocationAttemptLimit" {
        2
    } else {
        1
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn campaign_refuses_unaccounted_updated_suffix_and_forged_return() {
        // Only fabricated metadata: no Game, model, collection or fresh owner.
        for name in [
            "run_1.json",
            "run 1.json",
            "Report.",
            "NUL.json",
            "CON",
            "prn.txt",
            "Aux.log",
            "COM1.json",
            "Lpt9",
            "Archive",
            "Campaign-Spec.JSON",
            "Reservation.JSON",
            "Invocation-0000000000000000-close.json",
            "Report.TMP",
        ] {
            assert!(output_leaves(name, None).is_err(), "unsafe leaf {name}");
            assert!(
                output_leaves("report.json", Some(name)).is_err(),
                "unsafe deployment {name}"
            );
        }
        assert!(output_leaves(&"a".repeat(65), None).is_err());
        assert!(output_leaves("report.json", Some("REPORT.JSON")).is_err());
        output_leaves("Report-01.json", Some("Deployment-01.JSON")).unwrap();
        let longest = "a".repeat(64);
        output_leaves(&longest, None).unwrap();
        let parent = CurrentMixedParentIdentity::InitialMixedV1 {
            policy: source::tests::initial(),
        };
        let mut progress = Progress::initial(
            &parent,
            [u64::MAX, u64::MAX - 1],
            u64::MAX - 8,
            &BTreeSet::new(),
        );
        progress.attempts = 1;
        let child = MetadataTip {
            raw: Vec::new(),
            progress,
            reservations: Vec::new(),
            resolved_heads: vec!["b".repeat(64)],
            binding: None,
            entry: Entry {
                schema: JOURNAL_SCHEMA.into(),
                sequence: 1,
                previous: String::new(),
                root: "c".repeat(64),
                spec: "d".repeat(64),
                pool: "e".repeat(64),
                event: Event::CollectStart { member: 0 },
            },
            directory: PathBuf::new(),
        };
        let directory = std::env::temp_dir().join(format!(
            "tzolkin-campaign-metadata-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir(&directory).unwrap();
        // Exercise both unchanged persistence APIs at the admitted name boundary.
        publish_mixed_metadata(&directory, &longest, b"metadata boundary").unwrap();
        assert_eq!(
            read_pin(
                &directory,
                &file_pin(&longest, b"metadata boundary"),
                HEAD_CAP
            )
            .unwrap(),
            b"metadata boundary"
        );
        let digest = "a".repeat(64);
        let intent = InvocationIntent {
            schema: "tzolkin-current-mixed-campaign-invocation-intent-v1".into(),
            campaign_raw_sha256: digest.clone(),
            prefix_before: 0,
            max_new_attempts: 1,
        };
        fs::write(
            directory.join(invocation_name(0, "intent")),
            encoded(&intent).unwrap(),
        )
        .unwrap();
        assert!(prior_invocations(&directory, &digest, &child).is_err()); // native Updated alone is insufficient
        let report_raw = b"synthetic report";
        fs::write(directory.join("report.json"), report_raw).unwrap();
        let mut close = InvocationClose {
            schema: "tzolkin-current-mixed-campaign-invocation-close-v1".into(),
            campaign_raw_sha256: digest.clone(),
            prefix_before: 0,
            prefix_after: 1,
            native_head_sha256: "f".repeat(64),
            stop_reason: "invocationAttemptLimit".into(),
            report: file_pin("report.json", report_raw),
        };
        let close_path = directory.join(invocation_name(0, "close"));
        fs::write(&close_path, encoded(&close).unwrap()).unwrap();
        assert!(prior_invocations(&directory, &digest, &child).is_err()); // correct count, foreign native head
        close.native_head_sha256 = child.resolved_heads[0].clone();
        fs::write(&close_path, encoded(&close).unwrap()).unwrap();
        prior_invocations(&directory, &digest, &child).unwrap();
        close.stop_reason = "noChange".into();
        fs::write(&close_path, encoded(&close).unwrap()).unwrap();
        assert!(prior_invocations(&directory, &digest, &child).is_err());
        close.stop_reason = "invocationAttemptLimit".into();
        fs::write(&close_path, encoded(&close).unwrap()).unwrap();
        fs::write(directory.join("report.json"), b"substituted report").unwrap();
        assert!(prior_invocations(&directory, &digest, &child).is_err());
        // Directory is exactly this fixture's newly created local temporary root.
        assert_eq!(directory.parent(), Some(std::env::temp_dir().as_path()));
        fs::remove_dir_all(&directory).unwrap();
    }
}
