//! Bounded single-writer Setup checkpoints with reservations persisted before work.
//! Cold restore audits retained native sources and rebuilds updates once. Saved
//! reports and numeric checkpoint claims cannot independently confer learning credit.
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

use serde::{Deserialize, Serialize, de::DeserializeOwned};
use tzolkin_inference::deployment::LoadedDeployment;

use crate::model::digest;
use crate::own_setup_model::{InitializedOwnSetupArtifact, InitializedOwnSetupIdentity};
use crate::own_setup_native::{OwnSetupNativeConfig, fixed_playing_identity};
use crate::own_setup_record::FixedPlayingIdentity;
use crate::own_setup_training::{
    BatchRecords, DeclaredOwnSetupExclusions, OwnSetupAdamConfig, OwnSetupAttemptReport,
    OwnSetupCheckpointState, OwnSetupRunConfig, OwnSetupTrainingOwner, ReservedOwnSetupAttempt,
};
use crate::public_policy_update::{checkpoint_digest, hash};
use crate::public_rl_mixed_session::{FilePin, read_pin};
use crate::public_rl_paired_session::{
    local, publish_mixed_metadata, publish_mixed_stored_record, read_bounded,
    read_mixed_stored_record,
};
use crate::public_rl_record_storage as storage;
use crate::public_stochastic::{SamplingSeed, SamplingStreamIdentity};
use crate::public_stochastic_native::{CollectionLimits, NativeStochasticConfig, numerical_target};
use crate::public_stochastic_record::{MAX_RECORD_BYTES, unique_json};

const ROOT_SCHEMA: &str = "tzolkin-own-setup-durable-root-v1";
const HEAD_SCHEMA: &str = "tzolkin-own-setup-durable-head-v1";
const MAX_ATTEMPTS: u64 = 64;
const ROOT_CAP: usize = 16 * 1024 * 1024;
const HEAD_CAP: usize = 128 * 1024;
const CHECKPOINT_CAP: usize = 2 * 1024 * 1024;
const MAX_FILES: usize = 2048;
const MAX_ORPHANS: usize = 16;

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct OwnSetupArchiveLimits {
    pub max_attempts: u64,
    pub max_disk_bytes: u64,
}
impl OwnSetupArchiveLimits {
    fn validate(&self) -> Result<(), String> {
        if !(1..=MAX_ATTEMPTS).contains(&self.max_attempts)
            || !(32 * 1024 * 1024..=8 * 1024 * 1024 * 1024).contains(&self.max_disk_bytes)
        {
            return Err("Own Setup archive limits outside bounded range".into());
        }
        Ok(())
    }
}

/// An external latest-head pin is required on restore; it is not model authority.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct OwnSetupArchiveTip {
    pub sequence: u64,
    pub name: String,
    pub bytes: usize,
    pub sha256: String,
}
impl OwnSetupArchiveTip {
    fn pin(&self) -> Result<FilePin, String> {
        if self.name != head_name(self.sequence)
            || !checkpoint_digest(&self.sha256)
            || self.bytes == 0
            || self.bytes > HEAD_CAP
            || self.sequence > 2 * MAX_ATTEMPTS
        {
            return Err("Invalid exact Own Setup latest-head pin".into());
        }
        Ok(FilePin {
            name: self.name.clone(),
            bytes: self.bytes,
            sha256: self.sha256.clone(),
        })
    }
}

/// Explicit cold-rebuild work, separate from collection/update latency.
#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct OwnSetupRestoreWork {
    pub closed_attempts: u64,
    pub audited_records: usize,
    pub audited_raw_bytes: u64,
    pub abandoned_pending: bool,
    pub elapsed_seconds: f64,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct Root {
    schema: String,
    initial: InitializedOwnSetupIdentity,
    playing: FixedPlayingIdentity,
    numerical_target: String,
    adam: OwnSetupAdamConfig,
    run: OwnSetupRunConfig,
    excluded_families: Vec<String>,
    exclusion_digest: String,
    limits: OwnSetupArchiveLimits,
    checksum: String,
}
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct Config {
    players: usize,
    environment_seed: u32,
    sampling_seed: u64,
    episode_ordinal: u64,
    replicate_ordinal: u64,
    learner_seat: usize,
    max_callbacks: usize,
    max_candidate_rows: usize,
    max_source_bytes: usize,
    max_trace_bytes: usize,
}
impl Config {
    fn from_native(c: &OwnSetupNativeConfig) -> Self {
        let n = c.native();
        let s = n.sampling_identity();
        let l = n.limits();
        Self {
            players: n.players(),
            environment_seed: n.environment_seed(),
            sampling_seed: s.sampling_seed().value(),
            episode_ordinal: s.episode_ordinal(),
            replicate_ordinal: s.replicate_ordinal(),
            learner_seat: c.learner_seat(),
            max_callbacks: l.max_callbacks(),
            max_candidate_rows: l.max_candidate_rows(),
            max_source_bytes: l.max_source_bytes(),
            max_trace_bytes: l.max_trace_bytes(),
        }
    }
    fn native(&self) -> Result<OwnSetupNativeConfig, String> {
        OwnSetupNativeConfig::new(
            NativeStochasticConfig::new(
                self.players,
                self.environment_seed,
                SamplingStreamIdentity::new(
                    SamplingSeed::new(self.sampling_seed),
                    self.episode_ordinal,
                    self.replicate_ordinal,
                    self.players,
                )?,
                CollectionLimits::new(
                    self.max_callbacks,
                    self.max_candidate_rows,
                    self.max_source_bytes,
                    self.max_trace_bytes,
                )?,
            )?,
            self.learner_seat,
        )
    }
}
#[derive(Debug, Deserialize, Serialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct RecordPin {
    stored: FilePin,
    raw_bytes: usize,
    raw_sha256: String,
}
#[derive(Debug, Deserialize, Serialize)]
#[serde(
    tag = "kind",
    rename_all = "camelCase",
    rename_all_fields = "camelCase",
    deny_unknown_fields
)]
enum Event {
    Genesis {
        checkpoint: FilePin,
    },
    Pending {
        configs: Vec<Config>,
        reservation: OwnSetupAttemptReport,
    },
    Closed {
        abandoned: bool,
        report: OwnSetupAttemptReport,
        records: Vec<RecordPin>,
        checkpoint: FilePin,
        ignored_orphans: Vec<FilePin>,
    },
}
#[derive(Debug, Deserialize, Serialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct Head {
    schema: String,
    sequence: u64,
    root: FilePin,
    previous: Option<FilePin>,
    event: Event,
    checksum: String,
}

/// Process-crash recovery for one writer. Publication uses synced regular files
/// and immutable names. It does not promise power-loss durability or hostile-FS safety.
pub struct DurableOwnSetupOwner {
    directory: PathBuf,
    root: Root,
    root_pin: FilePin,
    tip: OwnSetupArchiveTip,
    owner: OwnSetupTrainingOwner,
    disk_bytes: u64,
    ready: bool,
}
impl DurableOwnSetupOwner {
    pub fn create(
        directory: &Path,
        initial: InitializedOwnSetupArtifact,
        playing: LoadedDeployment,
        adam: OwnSetupAdamConfig,
        run: OwnSetupRunConfig,
        exclusions: DeclaredOwnSetupExclusions,
        limits: OwnSetupArchiveLimits,
    ) -> Result<Self, String> {
        limits.validate()?;
        if run.max_attempted_episodes > limits.max_attempts * 4 {
            return Err("Own Setup run exceeds durable attempt archive".into());
        }
        let mut root = Root {
            schema: ROOT_SCHEMA.into(),
            initial: initial.identity.clone(),
            playing: fixed_playing_identity(&playing)?,
            numerical_target: numerical_target(),
            adam: adam.clone(),
            run: run.clone(),
            excluded_families: exclusions.families().iter().cloned().collect(),
            exclusion_digest: exclusions.digest().into(),
            limits,
            checksum: String::new(),
        };
        let owner = OwnSetupTrainingOwner::new(initial, playing, adam, run, exclusions)?;
        root.checksum = root_checksum(&root)?;
        let directory = local(directory, true)?;
        fs::create_dir(&directory).map_err(|e| e.to_string())?;
        let root_bytes = bounded_json(&root, ROOT_CAP)?;
        let root_pin = publish_metadata(&directory, "root.json", &root_bytes)?;
        let state = owner.checkpoint()?;
        let bytes = bounded_json(&state, CHECKPOINT_CAP)?;
        let checkpoint = publish_metadata(&directory, &checkpoint_name(0, false), &bytes)?;
        let (tip, head_bytes) = publish_head(
            &directory,
            0,
            &root_pin,
            None,
            Event::Genesis { checkpoint },
        )?;
        let disk_bytes = (root_bytes.len() + bytes.len() + head_bytes) as u64;
        Ok(Self {
            directory,
            root,
            root_pin,
            tip,
            owner,
            disk_bytes,
            ready: true,
        })
    }
    /// Locates a published latest head after an uncertain I/O result. This only
    /// returns a file pin; `restore` still validates its complete source ancestry.
    pub fn inspect_tip(directory: &Path) -> Result<OwnSetupArchiveTip, String> {
        let directory = local(directory, false)?;
        let files = inventory(&directory)?;
        let sequence = files
            .keys()
            .filter_map(|name| head_sequence(name))
            .max()
            .ok_or("Own Setup archive has no published head")?;
        if sequence > 2 * MAX_ATTEMPTS {
            return Err("Own Setup inspected head exceeds cap".into());
        }
        let name = head_name(sequence);
        let raw = read_bounded(&directory.join(&name), HEAD_CAP)?;
        let head: Head = decode(&raw, HEAD_CAP)?;
        if head.schema != HEAD_SCHEMA
            || head.sequence != sequence
            || head.checksum != head_checksum(&head)?
        {
            return Err("Own Setup inspected head is not exact".into());
        }
        Ok(OwnSetupArchiveTip {
            sequence,
            name,
            bytes: raw.len(),
            sha256: digest(&raw),
        })
    }
    pub fn tip(&self) -> &OwnSetupArchiveTip {
        &self.tip
    }
    pub fn owner(&self) -> Result<&OwnSetupTrainingOwner, String> {
        if !self.ready {
            return Err(
                "Own Setup archive requires cold recovery after incomplete publication".into(),
            );
        }
        Ok(&self.owner)
    }
    pub fn attempt(
        &mut self,
        configs: &[OwnSetupNativeConfig],
    ) -> Result<OwnSetupAttemptReport, String> {
        let reservation = self.reserve_on_disk(configs)?;
        let mut records = Vec::new();
        let report = self.owner.resolve(
            reservation,
            configs,
            BatchRecords::Collect(Some(&mut records)),
        )?;
        self.close(report, records, false, Vec::new())
    }
    fn reserve_on_disk(
        &mut self,
        configs: &[OwnSetupNativeConfig],
    ) -> Result<ReservedOwnSetupAttempt, String> {
        self.owner()?;
        let preview = self.owner.preview(configs)?;
        self.require_space(configs.len())?;
        // Freeze the reservation on disk before either sampling or optimizer work.
        self.ready = false;
        let pending = self.tip.sequence + 1;
        let event = Event::Pending {
            configs: configs.iter().map(Config::from_native).collect(),
            reservation: preview.clone(),
        };
        let result = publish_head(
            &self.directory,
            pending,
            &self.root_pin,
            Some(self.tip.pin()?),
            event,
        );
        let (tip, bytes) = match result {
            Ok(v) => v,
            Err(e) => {
                return Err(format!(
                    "Own Setup reservation publication uncertain; cold recovery required: {e}"
                ));
            }
        };
        self.disk_bytes += bytes as u64;
        self.tip = tip;
        let reservation = self.owner.reserve(configs)?;
        if json(reservation.report())? != json(&preview)? {
            return Err("Own Setup published/live reservation differs".into());
        }
        Ok(reservation)
    }
    fn require_space(&self, batch: usize) -> Result<(), String> {
        if self.tip.sequence / 2 >= self.root.limits.max_attempts {
            return Err("Own Setup durable attempt cap reached".into());
        }
        let reserve = (2 * CHECKPOINT_CAP + 4 * HEAD_CAP) as u64
            + (2 * storage::MAX_STORED_RECORD_BYTES) as u64 * batch as u64;
        if self
            .disk_bytes
            .checked_add(reserve)
            .is_none_or(|n| n > self.root.limits.max_disk_bytes)
        {
            return Err("Own Setup durable disk budget cannot reserve bounded publication".into());
        }
        Ok(())
    }
    fn close(
        &mut self,
        report: OwnSetupAttemptReport,
        records: Vec<Vec<u8>>,
        abandoned: bool,
        ignored_orphans: Vec<FilePin>,
    ) -> Result<OwnSetupAttemptReport, String> {
        if self.tip.sequence % 2 != 1
            || records.len() != report.receipts.len()
            || (abandoned && !records.is_empty())
        {
            return Err("Own Setup close has incompatible pending records".into());
        }
        let mut pins = Vec::new();
        let mut total_raw = 0usize;
        for (i, raw) in records.into_iter().enumerate() {
            total_raw = total_raw
                .checked_add(raw.len())
                .ok_or("Own Setup retained batch bytes overflow")?;
            if total_raw > MAX_RECORD_BYTES {
                return Err("Own Setup retained batch record byte cap".into());
            }
            let name = record_name(self.tip.sequence, i);
            let compressed = storage::encode(&raw)?;
            publish_mixed_stored_record(&self.directory, &name, &compressed)?;
            let stored = pin(&name, &compressed);
            read_stored(&self.directory, &stored)?;
            self.disk_bytes += compressed.len() as u64;
            pins.push(RecordPin {
                stored,
                raw_bytes: raw.len(),
                raw_sha256: digest(&raw),
            });
        }
        let sequence = self.tip.sequence + 1;
        let state = self.owner.checkpoint()?;
        let bytes = bounded_json(&state, CHECKPOINT_CAP)?;
        let name = checkpoint_name(sequence, abandoned);
        let checkpoint = if abandoned && self.directory.join(&name).exists() {
            let p = pin(&name, &bytes);
            read_pin(&self.directory, &p, CHECKPOINT_CAP)?;
            p
        } else {
            publish_metadata(&self.directory, &name, &bytes)?
        };
        self.disk_bytes += bytes.len() as u64;
        let event = Event::Closed {
            abandoned,
            report: report.clone(),
            records: pins,
            checkpoint,
            ignored_orphans,
        };
        let (tip, bytes) = publish_head(
            &self.directory,
            sequence,
            &self.root_pin,
            Some(self.tip.pin()?),
            event,
        )?;
        self.disk_bytes += bytes as u64;
        self.tip = tip;
        self.ready = true;
        Ok(report)
    }

    /// Replays retained *sources*, not saved learning claims. All updates are
    /// rebuilt atomically through their exact reserved configs and original order.
    /// The last pending attempt is consumed without sampling or optimizer retry.
    pub fn restore(
        directory: &Path,
        expected_tip: &OwnSetupArchiveTip,
        playing: LoadedDeployment,
        exclusions: DeclaredOwnSetupExclusions,
    ) -> Result<(Self, OwnSetupRestoreWork), String> {
        let start = Instant::now();
        let directory = local(directory, false)?;
        let inventory = inventory(&directory)?;
        let latest = expected_tip.pin()?;
        if inventory
            .keys()
            .filter_map(|name| head_sequence(name))
            .max()
            != Some(expected_tip.sequence)
        {
            return Err("Own Setup expected tip is not latest archived head".into());
        }
        let root_bytes = read_bounded(&directory.join("root.json"), ROOT_CAP)?;
        let root: Root = decode(&root_bytes, ROOT_CAP)?;
        root.limits.validate()?;
        if root.schema != ROOT_SCHEMA
            || root.numerical_target != numerical_target()
            || root.checksum != root_checksum(&root)?
            || root.run.max_attempted_episodes > root.limits.max_attempts * 4
            || json(&root.playing)? != json(&fixed_playing_identity(&playing)?)?
            || root.excluded_families != exclusions.families().iter().cloned().collect::<Vec<_>>()
            || root.exclusion_digest != exclusions.digest()
        {
            return Err("Own Setup root/config/Playing/exclusion identity mismatch".into());
        }
        let disk_bytes = inventory.values().try_fold(0u64, |n, bytes| {
            n.checked_add(*bytes as u64)
                .ok_or("Own Setup archive size overflow")
        })?;
        if disk_bytes > root.limits.max_disk_bytes
            || expected_tip.sequence > root.limits.max_attempts * 2
        {
            return Err("Own Setup archive disk/head budget exceeded".into());
        }
        let root_pin = pin("root.json", &root_bytes);
        let mut chain = Vec::new();
        let mut current = Some(latest);
        while let Some(p) = current {
            let raw = read_pin(&directory, &p, HEAD_CAP)?;
            let head: Head = decode(&raw, HEAD_CAP)?;
            if head.schema != HEAD_SCHEMA
                || head.root != root_pin
                || p.name != head_name(head.sequence)
                || head.checksum != head_checksum(&head)?
                || chain.len() > 2 * MAX_ATTEMPTS as usize
            {
                return Err("Own Setup head ancestry/identity/checksum differs".into());
            }
            current = head.previous.clone();
            chain.push((p, head));
        }
        chain.reverse();
        if chain.len() != expected_tip.sequence as usize + 1
            || chain
                .iter()
                .enumerate()
                .any(|(i, (_, h))| h.sequence != i as u64)
        {
            return Err("Own Setup archive head chain is incomplete or unordered".into());
        }
        let initial = InitializedOwnSetupArtifact::new(root.initial.seed)?;
        if initial.identity != root.initial {
            return Err("Own Setup initial root is not reproducible".into());
        }
        let owner = OwnSetupTrainingOwner::new(
            initial,
            playing,
            root.adam.clone(),
            root.run.clone(),
            exclusions,
        )?;
        let mut restored = Self {
            directory,
            root,
            root_pin,
            tip: expected_tip.clone(),
            owner,
            disk_bytes,
            ready: false,
        };
        let mut expected_names = BTreeSet::from(["root.json".to_owned()]);
        let mut pending: Option<(ReservedOwnSetupAttempt, Vec<OwnSetupNativeConfig>, u64)> = None;
        let mut work = OwnSetupRestoreWork {
            closed_attempts: 0,
            audited_records: 0,
            audited_raw_bytes: 0,
            abandoned_pending: false,
            elapsed_seconds: 0.0,
        };
        for (head_pin, head) in chain {
            expected_names.insert(head_pin.name);
            match head.event {
                Event::Genesis { checkpoint } => {
                    if head.sequence != 0 || head.previous.is_some() || pending.is_some() {
                        return Err("Invalid Own Setup genesis".into());
                    }
                    restored.check_checkpoint(&checkpoint, 0, false, &mut expected_names)?;
                }
                Event::Pending {
                    configs,
                    reservation,
                } => {
                    if head.sequence % 2 != 1
                        || pending.is_some()
                        || !(1..=4).contains(&configs.len())
                    {
                        return Err("Invalid Own Setup pending transition".into());
                    }
                    let configs = configs
                        .iter()
                        .map(Config::native)
                        .collect::<Result<Vec<_>, _>>()?;
                    let actual = restored.owner.reserve(&configs)?;
                    if json(actual.report())? != json(&reservation)? {
                        return Err(
                            "Own Setup saved reservation is not exact native cursor/config".into(),
                        );
                    }
                    pending = Some((actual, configs, head.sequence));
                }
                Event::Closed {
                    abandoned,
                    report,
                    records,
                    checkpoint,
                    ignored_orphans,
                } => {
                    let (reservation, configs, sequence) =
                        pending.take().ok_or("Own Setup close lacks reservation")?;
                    if head.sequence != sequence + 1
                        || records.len() != report.receipts.len()
                        || records.len() > configs.len()
                        || ignored_orphans.len() > MAX_ORPHANS
                        || (!abandoned && !ignored_orphans.is_empty())
                    {
                        return Err("Invalid Own Setup close/record closure".into());
                    }
                    let actual = if abandoned {
                        if !records.is_empty() {
                            return Err("Abandoned Own Setup attempt cannot audit records".into());
                        }
                        for p in ignored_orphans {
                            if !orphan_name(&p.name, sequence, configs.len()) {
                                return Err("Invalid Own Setup ignored orphan name".into());
                            }
                            read_any_pin(&restored.directory, &p)?;
                            if !expected_names.insert(p.name) {
                                return Err("Duplicate Own Setup ignored orphan".into());
                            }
                        }
                        restored.owner.abandon(reservation, &configs)?
                    } else {
                        let mut raw_records = Vec::new();
                        let mut raw_bytes = 0usize;
                        for (i, p) in records.into_iter().enumerate() {
                            if p.stored.name != record_name(sequence, i) {
                                return Err("Own Setup retained record order differs".into());
                            }
                            let stored = read_stored(&restored.directory, &p.stored)?;
                            let raw = storage::decode(&stored, p.raw_bytes, &p.raw_sha256)?;
                            raw_bytes = raw_bytes
                                .checked_add(raw.len())
                                .ok_or("Own Setup restored batch bytes overflow")?;
                            if raw_bytes > MAX_RECORD_BYTES {
                                return Err("Own Setup restored batch record cap".into());
                            }
                            expected_names.insert(p.stored.name);
                            raw_records.push(raw);
                        }
                        work.audited_records += raw_records.len();
                        work.audited_raw_bytes += raw_bytes as u64;
                        restored.owner.resolve(
                            reservation,
                            &configs,
                            BatchRecords::Audit(&raw_records),
                        )?
                    };
                    if json(&actual)? != json(&report)? {
                        return Err(
                            "Own Setup fresh audit/update differs from complete saved report"
                                .into(),
                        );
                    }
                    restored.check_checkpoint(
                        &checkpoint,
                        head.sequence,
                        abandoned,
                        &mut expected_names,
                    )?;
                    work.closed_attempts += 1;
                }
            }
        }
        if let Some((reservation, configs, sequence)) = pending {
            let mut orphans = Vec::new();
            for name in inventory.keys() {
                if !expected_names.contains(name) {
                    if matching_published_temporary(&restored.directory, name, &expected_names)? {
                        continue;
                    }
                    if !orphan_name(name, sequence, configs.len()) || orphans.len() >= MAX_ORPHANS {
                        return Err("Unknown/unbounded pending Own Setup payload".into());
                    }
                    let raw = read_mixed_stored_record(
                        &restored.directory.join(name),
                        storage::MAX_STORED_RECORD_BYTES,
                    )?;
                    if name != &checkpoint_name(sequence + 1, true) {
                        orphans.push(pin(name, &raw));
                    }
                    expected_names.insert(name.clone());
                }
            }
            require_inventory(&restored.directory, &inventory, &expected_names)?;
            let reserve = (2 * CHECKPOINT_CAP + 4 * HEAD_CAP) as u64;
            if restored
                .disk_bytes
                .checked_add(reserve)
                .is_none_or(|n| n > restored.root.limits.max_disk_bytes)
            {
                return Err("Own Setup pending recovery publication budget exhausted".into());
            }
            let report = restored.owner.abandon(reservation, &configs)?;
            restored.close(report, Vec::new(), true, orphans)?;
            work.abandoned_pending = true;
            work.closed_attempts += 1;
        } else {
            restored.recover_unpublished_pending(&inventory, &expected_names)?;
            restored.ready = true;
        }
        work.elapsed_seconds = start.elapsed().as_secs_f64();
        Ok((restored, work))
    }
    fn recover_unpublished_pending(
        &mut self,
        inventory: &BTreeMap<String, usize>,
        names: &BTreeSet<String>,
    ) -> Result<(), String> {
        let next = head_name(self.tip.sequence + 1);
        let mut remaining = inventory.clone();
        let mut paths = Vec::new();
        for (name, bytes) in inventory {
            if temporary_base(name) == Some(next.as_str()) {
                if paths.len() >= MAX_ORPHANS
                    || *bytes > HEAD_CAP
                    || self.tip.sequence / 2 >= self.root.limits.max_attempts
                {
                    return Err("Unpublished Own Setup pending temporary exceeds bounds".into());
                }
                let path = local(&self.directory.join(name), false)?;
                let meta = fs::symlink_metadata(&path).map_err(|e| e.to_string())?;
                if path.parent() != Some(self.directory.as_path())
                    || !meta.is_file()
                    || meta.len() != *bytes as u64
                {
                    return Err("Unpublished Own Setup pending temporary custody differs".into());
                }
                paths.push((path, *bytes));
                remaining.remove(name);
            }
        }
        // Validate every published/pinned file and reject foreign files before
        // removing anything. Unlinked pre-work bytes confer no reservation.
        require_inventory(&self.directory, &remaining, names)?;
        if !paths.is_empty() {
            match fs::symlink_metadata(self.directory.join(&next)) {
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
                Err(e) => return Err(e.to_string()),
                Ok(_) => {
                    return Err("Own Setup pending destination appeared during recovery".into());
                }
            }
        }
        for (path, bytes) in paths {
            fs::remove_file(path).map_err(|e| e.to_string())?;
            self.disk_bytes = self
                .disk_bytes
                .checked_sub(bytes as u64)
                .ok_or("Own Setup temporary byte accounting differs")?;
        }
        Ok(())
    }
    fn check_checkpoint(
        &self,
        p: &FilePin,
        sequence: u64,
        abandoned: bool,
        names: &mut BTreeSet<String>,
    ) -> Result<(), String> {
        if p.name != checkpoint_name(sequence, abandoned) || !names.insert(p.name.clone()) {
            return Err("Own Setup checkpoint name/uniqueness differs".into());
        }
        let raw = read_pin(&self.directory, p, CHECKPOINT_CAP)?;
        let claim: OwnSetupCheckpointState = decode(&raw, CHECKPOINT_CAP)?;
        claim.validate_claim()?;
        if json(&claim)? != json(&self.owner.checkpoint()?)? {
            return Err(
                "Own Setup exact parameter/moment/cursor/config/credit checkpoint differs".into(),
            );
        }
        Ok(())
    }
}

fn json<T: Serialize>(v: &T) -> Result<Vec<u8>, String> {
    serde_json::to_vec(v).map_err(|e| e.to_string())
}
fn bounded_json<T: Serialize>(v: &T, cap: usize) -> Result<Vec<u8>, String> {
    let bytes = json(v)?;
    if bytes.len() > cap {
        return Err("Own Setup metadata exceeds cap".into());
    }
    Ok(bytes)
}
fn decode<T: DeserializeOwned + Serialize>(raw: &[u8], cap: usize) -> Result<T, String> {
    let value = unique_json(raw, cap)?;
    let decoded: T = serde_json::from_value(value.clone()).map_err(|e| e.to_string())?;
    if serde_json::to_value(&decoded).map_err(|e| e.to_string())? != value {
        return Err("Own Setup metadata has missing or noncanonical typed fields".into());
    }
    Ok(decoded)
}
fn root_checksum(root: &Root) -> Result<String, String> {
    hash(
        b"tzolkin-own-setup-durable-root-v1\0",
        &(
            &root.schema,
            &root.initial,
            &root.playing,
            &root.numerical_target,
            &root.adam,
            &root.run,
            &root.excluded_families,
            &root.exclusion_digest,
            &root.limits,
        ),
    )
}
fn head_checksum(head: &Head) -> Result<String, String> {
    hash(
        b"tzolkin-own-setup-durable-head-v1\0",
        &(
            &head.schema,
            head.sequence,
            &head.root,
            &head.previous,
            &head.event,
        ),
    )
}
fn pin(name: &str, raw: &[u8]) -> FilePin {
    FilePin {
        name: name.into(),
        bytes: raw.len(),
        sha256: digest(raw),
    }
}
fn head_name(sequence: u64) -> String {
    format!("head-{sequence:08}.json")
}
fn checkpoint_name(sequence: u64, abandoned: bool) -> String {
    format!(
        "checkpoint-{sequence:08}{}.json",
        if abandoned { "-aborted" } else { "" }
    )
}
fn record_name(sequence: u64, member: usize) -> String {
    format!("record-{sequence:08}-{member}.json.gz")
}
fn head_sequence(name: &str) -> Option<u64> {
    let n = name
        .strip_prefix("head-")?
        .strip_suffix(".json")?
        .parse()
        .ok()?;
    if name == head_name(n) { Some(n) } else { None }
}
fn publish_metadata(directory: &Path, name: &str, raw: &[u8]) -> Result<FilePin, String> {
    publish_mixed_metadata(directory, name, raw)?;
    let p = pin(name, raw);
    read_pin(directory, &p, raw.len())?;
    Ok(p)
}
fn publish_head(
    directory: &Path,
    sequence: u64,
    root: &FilePin,
    previous: Option<FilePin>,
    event: Event,
) -> Result<(OwnSetupArchiveTip, usize), String> {
    let mut head = Head {
        schema: HEAD_SCHEMA.into(),
        sequence,
        root: root.clone(),
        previous,
        event,
        checksum: String::new(),
    };
    head.checksum = head_checksum(&head)?;
    let raw = bounded_json(&head, HEAD_CAP)?;
    let p = publish_metadata(directory, &head_name(sequence), &raw)?;
    Ok((
        OwnSetupArchiveTip {
            sequence,
            name: p.name,
            bytes: p.bytes,
            sha256: p.sha256,
        },
        raw.len(),
    ))
}
fn read_stored(directory: &Path, p: &FilePin) -> Result<Vec<u8>, String> {
    if p.name
        != Path::new(&p.name)
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("")
        || !checkpoint_digest(&p.sha256)
        || p.bytes > storage::MAX_STORED_RECORD_BYTES
    {
        return Err("Invalid Own Setup stored leaf/pin".into());
    }
    let raw = read_mixed_stored_record(&directory.join(&p.name), storage::MAX_STORED_RECORD_BYTES)?;
    if raw.len() != p.bytes || digest(&raw) != p.sha256 {
        return Err("Own Setup stored bytes/SHA differs".into());
    }
    Ok(raw)
}
fn read_any_pin(directory: &Path, p: &FilePin) -> Result<(), String> {
    read_stored(directory, p).map(|_| ())
}
fn inventory(directory: &Path) -> Result<BTreeMap<String, usize>, String> {
    let mut files = BTreeMap::new();
    for entry in fs::read_dir(directory).map_err(|e| e.to_string())? {
        if files.len() >= MAX_FILES {
            return Err("Own Setup archive file cap exceeded".into());
        }
        let entry = entry.map_err(|e| e.to_string())?;
        let name = entry
            .file_name()
            .to_str()
            .ok_or("Non-UTF8 Own Setup archive leaf")?
            .to_owned();
        if name.is_empty()
            || name.len() > 64
            || !name
                .bytes()
                .all(|b| b.is_ascii_alphanumeric() || b == b'-' || b == b'.')
        {
            return Err("Invalid Own Setup archive leaf".into());
        }
        let path = local(&entry.path(), false)?;
        let metadata = fs::symlink_metadata(path).map_err(|e| e.to_string())?;
        if !metadata.is_file() || metadata.len() > storage::MAX_STORED_RECORD_BYTES as u64 {
            return Err("Own Setup archive file type/size differs".into());
        }
        files.insert(name, metadata.len() as usize);
    }
    Ok(files)
}
fn orphan_name(name: &str, sequence: u64, batch: usize) -> bool {
    let mut allowed = vec![
        checkpoint_name(sequence + 1, false),
        checkpoint_name(sequence + 1, true),
        head_name(sequence + 1),
    ];
    allowed.extend((0..batch).map(|i| record_name(sequence, i)));
    allowed.iter().any(|base| {
        name == base
            || name
                .strip_prefix(&format!("{base}."))
                .and_then(|s| s.strip_suffix(".tmp"))
                .is_some_and(|s| {
                    s.split_once('-').is_some_and(|(a, b)| {
                        !a.is_empty()
                            && !b.is_empty()
                            && a.bytes().chain(b.bytes()).all(|c| c.is_ascii_digit())
                    })
                })
    })
}
fn temporary_base(name: &str) -> Option<&str> {
    let (base, suffix) = name.strip_suffix(".tmp")?.rsplit_once('.')?;
    let (pid, serial) = suffix.split_once('-')?;
    if pid.is_empty()
        || serial.is_empty()
        || !pid
            .bytes()
            .chain(serial.bytes())
            .all(|b| b.is_ascii_digit())
    {
        return None;
    }
    Some(base)
}
fn matching_published_temporary(
    directory: &Path,
    name: &str,
    names: &BTreeSet<String>,
) -> Result<bool, String> {
    let Some(base) = temporary_base(name).filter(|base| names.contains(*base)) else {
        return Ok(false);
    };
    // A crashed hard-link publisher may leave a duplicate temporary file after
    // publishing its destination. The duplicate cannot change the pinned source.
    let raw = read_mixed_stored_record(&directory.join(name), storage::MAX_STORED_RECORD_BYTES)?;
    let published =
        read_mixed_stored_record(&directory.join(base), storage::MAX_STORED_RECORD_BYTES)?;
    if raw.len() != published.len() || digest(&raw) != digest(&published) {
        return Err("Own Setup published temporary differs from its pinned destination".into());
    }
    Ok(true)
}
fn require_inventory(
    directory: &Path,
    inventory: &BTreeMap<String, usize>,
    names: &BTreeSet<String>,
) -> Result<(), String> {
    if names.iter().any(|n| !inventory.contains_key(n)) {
        return Err("Own Setup archive referenced file is absent".into());
    }
    for name in inventory.keys() {
        if !names.contains(name) && !matching_published_temporary(directory, name, names)? {
            return Err("Own Setup archive includes unreferenced files".into());
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dataset::{DatasetSplit, seed_family_id, split_for_family};
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::time::{SystemTime, UNIX_EPOCH};

    struct Temporary(PathBuf);
    impl Temporary {
        fn new() -> Self {
            static SERIAL: AtomicU64 = AtomicU64::new(0);
            let nonce = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos();
            let p = std::env::temp_dir().join(format!(
                "tzolkin-own-setup-durable-{}-{nonce}-{}",
                std::process::id(),
                SERIAL.fetch_add(1, Ordering::Relaxed)
            ));
            fs::create_dir(&p).unwrap();
            Self(p)
        }
    }
    impl Drop for Temporary {
        fn drop(&mut self) {
            fs::remove_dir_all(&self.0).unwrap();
        }
    }
    fn exclusions(families: Vec<String>) -> DeclaredOwnSetupExclusions {
        let d = DeclaredOwnSetupExclusions::digest_for_families(&families).unwrap();
        DeclaredOwnSetupExclusions::from_families(families.clone(), families.len(), &d).unwrap()
    }
    fn create(directory: &Path, first: u64) -> DurableOwnSetupOwner {
        DurableOwnSetupOwner::create(
            directory,
            InitializedOwnSetupArtifact::new(7).unwrap(),
            crate::own_setup_native::tests::playing(),
            OwnSetupAdamConfig {
                learning_rate: 1.0e-4,
                beta1: 0.9,
                beta2: 0.999,
                epsilon: 1.0e-8,
                max_gradient_norm: 2.0,
                max_parameter_delta: 0.001,
            },
            OwnSetupRunConfig {
                sampling_seed: u64::MAX,
                replicate_ordinal: 7,
                first_episode_ordinal: first,
                max_attempted_episodes: 4,
            },
            exclusions(vec![]),
            OwnSetupArchiveLimits {
                max_attempts: 2,
                max_disk_bytes: 1024 * 1024 * 1024,
            },
        )
        .unwrap()
    }
    fn config(seed: u32, ordinal: u64) -> OwnSetupNativeConfig {
        OwnSetupNativeConfig::new(
            NativeStochasticConfig::new(
                3,
                seed,
                SamplingStreamIdentity::new(SamplingSeed::new(u64::MAX), ordinal, 7, 3).unwrap(),
                CollectionLimits::new(4, 1024, 1024 * 1024, 1024 * 1024).unwrap(),
            )
            .unwrap(),
            0,
        )
        .unwrap()
    }
    fn seeds() -> Vec<u32> {
        (0..100)
            .filter(|s| split_for_family(&seed_family_id(*s)).unwrap() == DatasetSplit::Train)
            .take(2)
            .collect()
    }

    #[test]
    fn failed_prefix_split_resume_preserves_exact_state_and_refuses_changed_inputs() {
        let temp = Temporary::new();
        let seeds = seeds();
        let first = u64::MAX - 2;
        let a = config(seeds[0], first);
        let b = config(seeds[1], first + 1);
        let mut continuous = create(&temp.0.join("continuous"), first);
        let mut split = create(&temp.0.join("split"), first);
        let first_report = split.attempt(std::slice::from_ref(&a)).unwrap();
        assert_eq!(first_report.receipts.len(), 1);
        assert_eq!(first_report.receipts[0].episode_id, None);
        assert_eq!(first_report.receipts[0].counts.observed_callbacks, 4);
        continuous.attempt(std::slice::from_ref(&a)).unwrap();
        let tip = split.tip().clone();
        drop(split);
        let (mut split, work) = DurableOwnSetupOwner::restore(
            &temp.0.join("split"),
            &tip,
            crate::own_setup_native::tests::playing(),
            exclusions(vec![]),
        )
        .unwrap();
        assert_eq!(work.closed_attempts, 1);
        assert_eq!(work.audited_records, 1);
        assert!(work.audited_raw_bytes > 0);
        assert!(!work.abandoned_pending);
        let second = split.attempt(std::slice::from_ref(&b)).unwrap();
        let reference = continuous.attempt(std::slice::from_ref(&b)).unwrap();
        assert_eq!(json(&second).unwrap(), json(&reference).unwrap());
        assert_eq!(
            json(&split.owner().unwrap().checkpoint().unwrap()).unwrap(),
            json(&continuous.owner().unwrap().checkpoint().unwrap()).unwrap()
        );
        assert_eq!(split.owner().unwrap().update_count(), 0);
        assert_eq!(split.owner().unwrap().attempted_episodes(), 2);
        assert_eq!(
            split.owner().unwrap().next_episode_ordinal(),
            Some(u64::MAX)
        );
        assert!(
            DurableOwnSetupOwner::restore(
                &split.directory,
                &tip,
                crate::own_setup_native::tests::playing(),
                exclusions(vec![])
            )
            .is_err()
        );
        let mut wrong = vec![seed_family_id(999)];
        wrong.sort();
        assert!(
            DurableOwnSetupOwner::restore(
                &split.directory,
                split.tip(),
                crate::own_setup_native::tests::playing(),
                exclusions(wrong)
            )
            .is_err()
        );
        let name = record_name(1, 0);
        let original = fs::read(split.directory.join(&name)).unwrap();
        let mut corrupt = original.clone();
        corrupt.push(0);
        fs::write(split.directory.join(&name), corrupt).unwrap();
        assert!(
            DurableOwnSetupOwner::restore(
                &split.directory,
                split.tip(),
                crate::own_setup_native::tests::playing(),
                exclusions(vec![])
            )
            .is_err()
        );
        fs::write(split.directory.join(&name), original).unwrap();
        let checkpoint = split.owner().unwrap().checkpoint().unwrap();
        let mut value = serde_json::to_value(&checkpoint).unwrap();
        value["firstMomentBits"][0] = "7ff0000000000000".into();
        let claim: OwnSetupCheckpointState =
            decode(&json(&value).unwrap(), CHECKPOINT_CAP).unwrap();
        assert!(claim.validate_claim().is_err());
        value = serde_json::to_value(&checkpoint).unwrap();
        value["secondMomentBits"].as_array_mut().unwrap().pop();
        let claim: OwnSetupCheckpointState =
            decode(&json(&value).unwrap(), CHECKPOINT_CAP).unwrap();
        assert!(claim.validate_claim().is_err());
        value = serde_json::to_value(&checkpoint).unwrap();
        value.as_object_mut().unwrap().remove("nextEpisodeOrdinal");
        assert!(decode::<OwnSetupCheckpointState>(&json(&value).unwrap(), CHECKPOINT_CAP).is_err());
        let mut root = serde_json::to_value(&split.root).unwrap();
        root["unexpected"] = true.into();
        assert!(decode::<Root>(&json(&root).unwrap(), ROOT_CAP).is_err());
        assert!(decode::<OwnSetupArchiveTip>(br#"{"sequence":0,"sequence":1,"name":"head-00000000.json","bytes":1,"sha256":"a"}"#,HEAD_CAP).is_err());
        let exact_tip = split.tip().clone();
        drop(split);
        let (restored, work) = DurableOwnSetupOwner::restore(
            &temp.0.join("split"),
            &exact_tip,
            crate::own_setup_native::tests::playing(),
            exclusions(vec![]),
        )
        .unwrap();
        assert_eq!(work.audited_records, 2);
        assert_eq!(work.closed_attempts, 2);
        assert_eq!(
            json(&restored.owner().unwrap().checkpoint().unwrap()).unwrap(),
            json(&continuous.owner().unwrap().checkpoint().unwrap()).unwrap()
        );
    }

    #[test]
    fn pending_u64_max_is_consumed_without_audit_or_optimizer_retry() {
        let temp = Temporary::new();
        let directory = temp.0.join("pending");
        let mut live = create(&directory, u64::MAX);
        let config = config(seeds()[0], u64::MAX);
        let genesis = live.tip().clone();
        let initial = json(&live.owner().unwrap().checkpoint().unwrap()).unwrap();
        let mut head = Head {
            schema: HEAD_SCHEMA.into(),
            sequence: 1,
            root: live.root_pin.clone(),
            previous: Some(genesis.pin().unwrap()),
            event: Event::Pending {
                configs: vec![Config::from_native(&config)],
                reservation: live.owner.preview(std::slice::from_ref(&config)).unwrap(),
            },
            checksum: String::new(),
        };
        head.checksum = head_checksum(&head).unwrap();
        // Reproduce the publisher's synced write-before-hard-link crash window.
        let unpublished = directory.join(format!("{}.123-0.tmp", head_name(1)));
        let partial = directory.join(format!("{}.123-1.tmp", head_name(1)));
        for (path, bytes) in [(&unpublished, json(&head).unwrap()), (&partial, vec![b'{'])] {
            use std::io::Write;
            let mut file = fs::OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(path)
                .unwrap();
            file.write_all(&bytes).unwrap();
            file.sync_all().unwrap();
        }
        live.ready = false;
        assert!(live.owner().is_err());
        drop(live);
        assert_eq!(
            DurableOwnSetupOwner::inspect_tip(&directory).unwrap(),
            genesis
        );
        // A foreign next-next temporary must refuse before deleting known ones.
        let foreign = directory.join(format!("{}.123-0.tmp", head_name(3)));
        fs::write(&foreign, b"foreign").unwrap();
        assert!(
            DurableOwnSetupOwner::restore(
                &directory,
                &genesis,
                crate::own_setup_native::tests::playing(),
                exclusions(vec![])
            )
            .is_err()
        );
        assert!(unpublished.exists());
        assert!(partial.exists());
        fs::remove_file(foreign).unwrap();
        let (mut live, work) = DurableOwnSetupOwner::restore(
            &directory,
            &genesis,
            crate::own_setup_native::tests::playing(),
            exclusions(vec![]),
        )
        .unwrap();
        assert!(!unpublished.exists());
        assert!(!partial.exists());
        assert!(!work.abandoned_pending);
        assert_eq!(work.closed_attempts, 0);
        assert_eq!(work.audited_records, 0);
        assert_eq!(live.owner().unwrap().attempted_episodes(), 0);
        assert_eq!(live.owner().unwrap().next_episode_ordinal(), Some(u64::MAX));
        assert_eq!(
            initial,
            json(&live.owner().unwrap().checkpoint().unwrap()).unwrap()
        );
        let reservation = live.reserve_on_disk(std::slice::from_ref(&config)).unwrap();
        assert_eq!(reservation.report().first_episode_ordinal, u64::MAX);
        assert_eq!(reservation.report().next_episode_ordinal, None);
        assert!(live.owner().is_err());
        drop(reservation);
        let before = live.owner.checkpoint().unwrap();
        // Inert interrupted payloads are retained without being passed to the audit.
        publish_mixed_metadata(
            &directory,
            &record_name(1, 0),
            b"incomplete untrusted source",
        )
        .unwrap();
        publish_metadata(
            &directory,
            &checkpoint_name(2, false),
            &json(&before).unwrap(),
        )
        .unwrap();
        publish_metadata(
            &directory,
            &checkpoint_name(2, true),
            &json(&before).unwrap(),
        )
        .unwrap();
        let pending = live.tip().clone();
        fs::copy(
            directory.join(&pending.name),
            directory.join(format!("{}.1-1.tmp", pending.name)),
        )
        .unwrap();
        drop(live);
        assert_eq!(
            DurableOwnSetupOwner::inspect_tip(&directory).unwrap(),
            pending
        );
        let (restored, work) = DurableOwnSetupOwner::restore(
            &directory,
            &pending,
            crate::own_setup_native::tests::playing(),
            exclusions(vec![]),
        )
        .unwrap();
        assert!(work.abandoned_pending);
        assert_eq!(work.audited_records, 0);
        assert_eq!(work.audited_raw_bytes, 0);
        assert_eq!(work.closed_attempts, 1);
        assert_eq!(restored.tip().sequence, 2);
        assert_eq!(restored.owner().unwrap().attempted_episodes(), 1);
        assert_eq!(restored.owner().unwrap().next_episode_ordinal(), None);
        assert_eq!(restored.owner().unwrap().update_count(), 0);
        assert_eq!(
            json(&before).unwrap(),
            json(&restored.owner().unwrap().checkpoint().unwrap()).unwrap()
        );
        let tip = restored.tip().clone();
        fs::copy(
            directory.join(&tip.name),
            directory.join(format!("{}.1-2.tmp", tip.name)),
        )
        .unwrap();
        drop(restored);
        let (mut again, work) = DurableOwnSetupOwner::restore(
            &directory,
            &tip,
            crate::own_setup_native::tests::playing(),
            exclusions(vec![]),
        )
        .unwrap();
        assert!(!work.abandoned_pending);
        assert_eq!(work.audited_records, 0);
        assert_eq!(work.closed_attempts, 1);
        assert!(again.attempt(std::slice::from_ref(&config)).is_err());
        assert_eq!(again.owner().unwrap().attempted_episodes(), 1);
        assert!(
            DurableOwnSetupOwner::restore(
                &directory,
                &pending,
                crate::own_setup_native::tests::playing(),
                exclusions(vec![])
            )
            .is_err()
        );
    }
}
