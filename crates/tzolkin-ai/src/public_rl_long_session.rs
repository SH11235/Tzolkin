//! Bounded Long-v2 driving from a pinned, resolved Count1-v1 session.
//! Immutable small checkpoint entries form a delta journal; model files are
//! written only on updates. Restoration checks content, not producer/history
//! authenticity. Evaluation thresholds are notices, never Arena results.
use std::collections::BTreeSet;
use std::fs::{self, OpenOptions};
use std::io::{Read, Write};
use std::path::{Component, Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};

use crate::dataset::{DatasetSplit, seed_family_id, split_for_family};
use crate::public_native::PreparedPublicPolicy;
use crate::public_policy_long::{self, LongRlParent, LongRlPolicy, LongRunLimits, LongStepOutcome};
use crate::public_policy_repeat::RlUpdateParent;
use crate::public_policy_update::{OneStepConfig, UpdatedPublicRlPolicy, checkpoint_digest, hash};
use crate::public_rl_artifact::InitializedPublicRlPolicy;
use crate::public_rl_long_stochastic_native::{
    audit_long_rl_record_bytes, collect_long_rl_native, encode_long_rl_record,
};
use crate::public_rl_native::{LongRlHandle, UpdatedPublicRlHandle};
use crate::public_rl_policy_cohort::{PlannedRlCohort, ValidatedRlCohort};
use crate::public_rl_policy_episode::ValidatedRlPolicyEpisode;
use crate::public_rl_session::RlTrainingSession;
use crate::public_rl_stochastic_native::{
    audit_rl_record_bytes, collect_rl_native, encode_rl_record,
};
use crate::public_stochastic::{RlSamplingPolicy, SamplingSeed, SamplingStreamIdentity};
use crate::public_stochastic_native::{CollectionLimits, NativeStochasticConfig};
use crate::public_stochastic_record::{
    Config, MAX_RECORD_BYTES, Record, decode_typed_record, unique_json,
};

pub const SPEC_SCHEMA: &str = "tzolkin-public-rl-long-run-spec-v2";
pub const MAX_SPEC_BYTES: usize = 64 * 1024;
pub const MAX_CHECKPOINT_BYTES: usize = 64 * 1024;
const OWNER_BYTES: usize = 16 * 1024 * 1024;
const MAX_STORAGE: u64 = 8 * 1024 * 1024 * 1024;

/// Fixed metadata allocation and capacities. Safety limits do not authorize an
/// experiment. Decision work is the new Long segment, excluding the bootstrap.
#[derive(Clone, Deserialize, Serialize, PartialEq)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct LongRunSpec {
    schema: String,
    sampling_seed: u64,
    replicate_ordinal: u64,
    environment_seed_start: u32,
    max_seed_candidates: u32,
    decision_budget: u64,
    evaluation_interval: u64,
    max_updates: u64,
    max_families: usize,
    max_batches: u64,
    max_journal_entries: u64,
    learning_rate: f64,
    max_abs_actual_delta: f64,
    max_gradient_rows: usize,
    max_callbacks: usize,
    max_candidate_rows: usize,
    max_source_bytes: usize,
    max_trace_bytes: usize,
    max_metadata_bytes: u64,
    max_record_bytes: u64,
    excluded_families: Vec<String>,
}
impl LongRunSpec {
    pub fn from_json(bytes: &[u8]) -> Result<Self, String> {
        let spec: Self = closed(bytes, MAX_SPEC_BYTES)?;
        spec.validate()?;
        Ok(spec)
    }
    fn validate(&self) -> Result<(), String> {
        self.limits()?;
        self.step()?;
        self.collection_limits()?;
        if self.schema != SPEC_SCHEMA
            || self.decision_budget == 0
            || self.evaluation_interval == 0
            || !(1..=1_000_000).contains(&self.max_seed_candidates)
            || self.end_seed() > u32::MAX as u64 + 1
            || self.max_batches == 0
            || self.max_batches > public_policy_long::MAX_UPDATES
            || self.max_journal_entries
                < self
                    .max_batches
                    .checked_mul(19)
                    .and_then(|n| n.checked_add(1))
                    .ok_or("Journal count overflow")?
            || self.max_journal_entries > public_policy_long::MAX_UPDATES * 24
            || self.excluded_families.len() > 512
            || self.excluded_families.iter().any(|f| !checkpoint_digest(f))
            || self.excluded_families.windows(2).any(|w| w[0] >= w[1])
            || !(OWNER_BYTES as u64 * 2..=MAX_STORAGE).contains(&self.max_metadata_bytes)
            || !(4 * MAX_RECORD_BYTES as u64..=MAX_STORAGE).contains(&self.max_record_bytes)
        {
            return Err("Invalid explicit Long run specification".into());
        }
        Ok(())
    }
    fn end_seed(&self) -> u64 {
        self.environment_seed_start as u64 + self.max_seed_candidates as u64
    }
    fn limits(&self) -> Result<LongRunLimits, String> {
        LongRunLimits::new(self.max_updates, self.max_families)
    }
    fn step(&self) -> Result<OneStepConfig, String> {
        OneStepConfig::new(
            self.learning_rate,
            self.max_abs_actual_delta,
            self.max_gradient_rows,
        )
    }
    fn collection_limits(&self) -> Result<CollectionLimits, String> {
        CollectionLimits::new(
            self.max_callbacks,
            self.max_candidate_rows,
            self.max_source_bytes,
            self.max_trace_bytes,
        )
    }
    fn checksum(&self) -> Result<String, String> {
        hash(b"tzolkin-rl-long-run-spec-v2\0", self)
    }
    fn configs(
        &self,
        cursor: &mut u64,
        ordinal: u64,
        consumed: &BTreeSet<String>,
    ) -> Result<[Config; 4], String> {
        let mut configs = Vec::with_capacity(4);
        let mut chosen = BTreeSet::new();
        ordinal
            .checked_add(4)
            .ok_or("Long episode ordinal overflow")?;
        while configs.len() < 4 && *cursor < self.end_seed() {
            let seed = *cursor as u32;
            *cursor += 1;
            let f = seed_family_id(seed);
            if split_for_family(&f)? == DatasetSplit::Train
                && !consumed.contains(&f)
                && self.excluded_families.binary_search(&f).is_err()
                && chosen.insert(f)
            {
                let players = if configs.len() < 2 { 3 } else { 4 };
                configs.push(Config::from_config(&NativeStochasticConfig::new(
                    players,
                    seed,
                    SamplingStreamIdentity::new(
                        SamplingSeed::new(self.sampling_seed),
                        ordinal + configs.len() as u64,
                        self.replicate_ordinal,
                        players,
                    )?,
                    self.collection_limits()?,
                )?));
            }
        }
        configs
            .try_into()
            .map_err(|_| "Long fixed metadata seed range exhausted".into())
    }
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct Pin {
    name: String,
    bytes: usize,
    sha256: String,
}
impl Pin {
    fn valid(&self, maximum: usize) -> bool {
        self.bytes > 0
            && self.bytes <= maximum
            && checkpoint_digest(&self.sha256)
            && !self.name.is_empty()
            && self.name.len() < 80
            && self
                .name
                .bytes()
                .all(|b| b.is_ascii_alphanumeric() || b == b'-' || b == b'.')
    }
}
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct OwnerPin {
    file: Pin,
    count: u64,
    artifact: String,
    lineage: String,
}
#[derive(Clone, Default, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct Work {
    callbacks: u64,
    attempts: u64,
    accepted: u64,
    applied: u64,
    reserved_rows: u64,
    accepted_rows: u64,
    singletons: u64,
}
impl Work {
    fn add(&mut self, w: &Self) -> Result<(), String> {
        for (a, b) in [
            &mut self.callbacks,
            &mut self.attempts,
            &mut self.accepted,
            &mut self.applied,
            &mut self.reserved_rows,
            &mut self.accepted_rows,
            &mut self.singletons,
        ]
        .into_iter()
        .zip([
            w.callbacks,
            w.attempts,
            w.accepted,
            w.applied,
            w.reserved_rows,
            w.accepted_rows,
            w.singletons,
        ]) {
            *a = a.checked_add(b).ok_or("Long logical work overflow")?;
        }
        Ok(())
    }
    fn valid(&self, c: &Config) -> bool {
        self.applied <= self.accepted
            && self.accepted <= self.attempts
            && self.attempts <= self.callbacks
            && self.callbacks <= c.max_callbacks as u64
            && self.reserved_rows <= c.max_candidate_rows as u64
            && self.accepted_rows <= self.reserved_rows
            && self.accepted_rows >= self.accepted
            && self.singletons <= self.accepted
    }
}
#[derive(Clone, Default, Debug, Deserialize, Serialize, PartialEq)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct Phase {
    started: u64,
    returned: u64,
    returned_seconds: f64,
}
impl Phase {
    fn returned(&mut self, seconds: f64) -> Result<(), String> {
        if !seconds.is_finite() || seconds < 0.0 || self.returned >= self.started {
            return Err("Invalid Long phase return".into());
        }
        self.returned += 1;
        self.returned_seconds += seconds;
        if !self.returned_seconds.is_finite() {
            return Err("Long elapsed time overflow".into());
        }
        Ok(())
    }
}
#[derive(Clone, Default, Debug, Deserialize, Serialize, PartialEq)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct Member {
    started: bool,
    returned: bool,
    record: Option<Pin>,
    work: Option<Work>,
    error: Option<String>,
    audit_started: bool,
    audit_passed: bool,
}
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct Batch {
    configs: [Config; 4],
    plan: String,
    parent: String,
    parent_count: u64,
    members: [Member; 4],
    gradient_unknown: bool,
}
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct Progress {
    owner: OwnerPin,
    cursor: u64,
    next_ordinal: u64,
    reservations: u64,
    pending: Option<Batch>,
    known: Work,
    complete_counts: [u64; 3],
    unknown_collections: u64,
    collect: Phase,
    audit: Phase,
    gradient: Phase,
    record_bytes: u64,
    metadata_bytes: u64,
    due: u64,
    last_outcome: Option<String>,
    last_error: Option<String>,
}
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(rename_all = "camelCase", deny_unknown_fields, tag = "kind")]
enum Event {
    Start,
    Reserve {
        configs: Box<[Config; 4]>,
        plan: String,
        cursor: u64,
    },
    CollectStart {
        member: usize,
    },
    CollectReturn {
        member: usize,
        record: Option<Pin>,
        work: Option<Work>,
        error: Option<String>,
        seconds: f64,
    },
    AuditStart {
        member: usize,
    },
    AuditReturn {
        member: usize,
        passed: bool,
        error: Option<String>,
        seconds: f64,
    },
    GradientStart,
    Resolve {
        outcome: String,
        error: Option<String>,
        counts: Option<[usize; 3]>,
        report: Option<String>,
        owner: Option<OwnerPin>,
        seconds: Option<f64>,
    },
}
#[derive(Deserialize, Serialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct Checkpoint {
    schema: String,
    root: Pin,
    spec_checksum: String,
    sequence: u64,
    previous: String,
    event: Event,
    progress: Progress,
}

enum Current {
    Count1(Box<UpdatedPublicRlPolicy>),
    Long(Box<LongRlPolicy>),
}
impl Current {
    fn parent(&self) -> LongRlParent<'_> {
        match self {
            Self::Count1(p) => LongRlParent::Count1(p),
            Self::Long(p) => LongRlParent::Long(p),
        }
    }
    fn value(&self) -> Result<Value, String> {
        match self {
            Self::Count1(p) => serde_json::to_value(p.artifact()),
            Self::Long(p) => serde_json::to_value(p.artifact()),
        }
        .map_err(|e| e.to_string())
    }
}
/// Sealed in-memory current policy. No raw numeric/provider/Deserialize route.
/// Mutating collection/update methods require exclusive access; an evaluation
/// caller can borrow the parent and finish before resuming the session.
pub struct LongTrainingSession {
    spec: LongRunSpec,
    root: Pin,
    root_init: String,
    root_count1: String,
    current: Option<Current>,
    consumed: BTreeSet<String>,
    trained: BTreeSet<String>,
    progress: Progress,
    sequence: u64,
    tip: String,
    bootstrap_complete_counts: [u64; 3],
    directory: PathBuf,
}
impl LongTrainingSession {
    pub fn current_parent(&self) -> LongRlParent<'_> {
        self.current
            .as_ref()
            .expect("session always retains its owner")
            .parent()
    }
    fn bootstrap(
        spec: LongRunSpec,
        prepared: &PreparedPublicPolicy,
        bytes: &[u8],
        expected: &str,
        directory: PathBuf,
    ) -> Result<Self, String> {
        spec.validate()?;
        let initial = InitializedPublicRlPolicy::from_bc(prepared)?;
        let root_init = initial.artifact().checksum().to_owned();
        let old = RlTrainingSession::restore(initial, bytes, expected)?;
        let progress = old.progress()?;
        if progress.unresolved_batch() {
            return Err("Long bootstrap requires a resolved Count1 session".into());
        }
        let config = serde_json::to_value(old.config()).map_err(|e| e.to_string())?;
        if config["step"] != serde_json::to_value(spec.step()?).map_err(|e| e.to_string())?
            || config["samplingSeed"] != spec.sampling_seed
            || config["replicateOrdinal"] != spec.replicate_ordinal
        {
            return Err("Long bootstrap cannot change step/sampling identity".into());
        }
        let count1 = match old.parent() {
            Some(RlUpdateParent::Count1(p)) => p,
            _ => return Err("Long entry requires Count1-v1 only".into()),
        };
        let owner = crate::public_policy_update::restore_checkpoint_owner(
            serde_json::to_value(count1.artifact()).map_err(|e| e.to_string())?,
        )?;
        LongRlParent::Count1(&owner).validate(spec.limits()?)?;
        let root_count1 = owner.artifact().checksum().to_owned();
        let trained: BTreeSet<_> = owner
            .artifact()
            .report()
            .family_closure()
            .iter()
            .cloned()
            .collect();
        // Full Count1 closure/root/source validation happens once at bootstrap.
        if owner.artifact().report().parent_init_checksum() != root_init
            || owner.artifact().report().config() != &spec.step()?
        {
            return Err("Long bootstrap root/config mismatch".into());
        }
        let consumed = old.consumed_families()?;
        if consumed.len() > spec.max_families || !trained.is_subset(&consumed) {
            return Err("Long bootstrap exceeds consumed family capacity".into());
        }
        progress
            .next_episode_ordinal()
            .checked_add(4 * spec.max_batches)
            .ok_or("Long reserved ordinal capacity overflow")?;
        let root = Pin {
            name: "bootstrap.json".into(),
            bytes: bytes.len(),
            sha256: expected.into(),
        };
        let raw = json_bytes(
            &serde_json::to_value(owner.artifact()).map_err(|e| e.to_string())?,
            OWNER_BYTES,
        )?;
        let lineage = hash(
            b"tzolkin-public-rl-long-root-lineage-v2\0",
            &(
                owner.artifact().checksum(),
                owner.artifact().report().family_closure(),
            ),
        )?;
        let owner_pin = OwnerPin {
            file: pin("owner-1.json", &raw),
            count: 1,
            artifact: root_count1.clone(),
            lineage,
        };
        let state = Progress {
            owner: owner_pin,
            cursor: spec.environment_seed_start as u64,
            next_ordinal: progress.next_episode_ordinal(),
            reservations: 0,
            pending: None,
            known: Work::default(),
            complete_counts: [0; 3],
            unknown_collections: 0,
            collect: Phase::default(),
            audit: Phase::default(),
            gradient: Phase::default(),
            record_bytes: 0,
            metadata_bytes: (bytes.len() + raw.len()) as u64,
            due: 0,
            last_outcome: None,
            last_error: None,
        };
        Ok(Self {
            spec,
            root,
            root_init,
            root_count1,
            current: Some(Current::Count1(Box::new(owner))),
            consumed,
            trained,
            progress: state,
            sequence: 0,
            tip: expected.into(),
            bootstrap_complete_counts: progress.complete_cohort_counts(),
            directory,
        })
    }
    fn save(&mut self, event: Event) -> Result<(), String> {
        let sequence = self
            .sequence
            .checked_add(1)
            .ok_or("Long journal sequence overflow")?;
        if sequence > self.spec.max_journal_entries {
            return Err("Long journal entry capacity reached".into());
        }
        let mut next = self.progress.clone();
        apply(&self.spec, &mut next, &event, &self.consumed)?;
        let checkpoint = Checkpoint {
            schema: "tzolkin-rl-long-checkpoint-v2".into(),
            root: self.root.clone(),
            spec_checksum: self.spec.checksum()?,
            sequence,
            previous: self.tip.clone(),
            event: event.clone(),
            progress: next.clone(),
        };
        let bytes = json_bytes(&checkpoint, MAX_CHECKPOINT_BYTES)?;
        next.metadata_bytes = next
            .metadata_bytes
            .checked_add(bytes.len() as u64)
            .ok_or("Long metadata capacity overflow")?;
        if next.metadata_bytes > self.spec.max_metadata_bytes {
            return Err("Long metadata storage byte cap reached".into());
        }
        // metadataBytes excludes this checkpoint until its publication commits.
        let published = publish(&self.directory, &head_name(sequence), &bytes)?;
        if let Event::Reserve { configs, .. } = event {
            self.consumed
                .extend(configs.iter().map(|c| seed_family_id(c.environment_seed)));
        }
        self.progress = next;
        self.sequence = sequence;
        self.tip = published.sha256;
        Ok(())
    }
    fn reserve(&mut self) -> Result<(), String> {
        let mut cursor = self.progress.cursor;
        let configs = self
            .spec
            .configs(&mut cursor, self.progress.next_ordinal, &self.consumed)?;
        let checked = checked_configs(&configs)?;
        let plan = match self.current.as_ref().ok_or("Missing Long current owner")? {
            Current::Count1(p) => PlannedRlCohort::new(&UpdatedPublicRlHandle::new(p)?, checked)?,
            Current::Long(p) => PlannedRlCohort::new_long(&LongRlHandle::new(p)?, checked)?,
        };
        self.save(Event::Reserve {
            configs: Box::new(configs),
            plan: plan.checksum().into(),
            cursor,
        })
    }
    fn collect(&mut self) -> Result<(), String> {
        for i in 0..4 {
            let batch = self
                .progress
                .pending
                .as_ref()
                .ok_or("Missing Long reservation")?;
            if batch.members[i].started {
                return Err("Unknown saved collection cannot be recollected".into());
            }
            let config = batch.configs[i].checked_config()?;
            // Reserve the whole four maximum byte payloads before any native call.
            self.save(Event::CollectStart { member: i })?;
            let began = Instant::now();
            let returned = match self.current.as_ref().ok_or("Missing Long current owner")? {
                Current::Count1(p) => {
                    match collect_rl_native(&config, &UpdatedPublicRlHandle::new(p)?) {
                        Ok(game) => Ok(encode_rl_record(&game)?),
                        Err(error) => Err(error),
                    }
                }
                Current::Long(p) => match collect_long_rl_native(&config, &LongRlHandle::new(p)?) {
                    Ok(game) => Ok(encode_long_rl_record(&game)?),
                    Err(error) => Err(error),
                },
            };
            let seconds = began.elapsed().as_secs_f64();
            match returned {
                Ok(bytes) => {
                    let work = record_work(&bytes)?;
                    let record = publish(
                        &self.directory,
                        &record_name(self.progress.reservations, i),
                        &bytes,
                    )?;
                    self.save(Event::CollectReturn {
                        member: i,
                        record: Some(record),
                        work: Some(work),
                        error: None,
                        seconds,
                    })?;
                }
                Err(error) => self.save(Event::CollectReturn {
                    member: i,
                    record: None,
                    work: None,
                    error: Some(capped(&error)),
                    seconds,
                })?,
            }
        }
        Ok(())
    }
    fn preflight_records(&self) -> Result<[Vec<u8>; 4], String> {
        let batch = self
            .progress
            .pending
            .as_ref()
            .ok_or("No Long pending reservation")?;
        let mut bytes = Vec::with_capacity(4);
        let expected_policy = match self.current.as_ref().ok_or("Missing current owner")? {
            Current::Count1(p) => RlSamplingPolicy::from_handle(&UpdatedPublicRlHandle::new(p)?)?,
            Current::Long(p) => RlSamplingPolicy::from_long_handle(&LongRlHandle::new(p)?)?,
        };
        for (i, m) in batch.members.iter().enumerate() {
            let raw = read_pin(
                &self.directory,
                m.record.as_ref().ok_or(
                    "Pending batch lacks all four saved records; explicit abandon required",
                )?,
                MAX_RECORD_BYTES,
            )?;
            let record: Record<RlSamplingPolicy> = decode_typed_record(&raw)?;
            if record.header.config != batch.configs[i]
                || record.header.base_policy != expected_policy
            {
                return Err(
                    "Saved record configuration/current policy differs before audit".into(),
                );
            }
            bytes.push(raw);
        }
        bytes
            .try_into()
            .map_err(|_| "Expected four saved records".into())
    }
    fn audit(&mut self, bytes: [Vec<u8>; 4]) -> Result<Result<ValidatedRlCohort, String>, String> {
        let configs = checked_configs(
            &self
                .progress
                .pending
                .as_ref()
                .ok_or("Missing Long batch")?
                .configs,
        )?;
        let plan = match self.current.as_ref().ok_or("Missing Long current owner")? {
            Current::Count1(p) => PlannedRlCohort::new(&UpdatedPublicRlHandle::new(p)?, configs)?,
            Current::Long(p) => PlannedRlCohort::new_long(&LongRlHandle::new(p)?, configs)?,
        };
        if plan.checksum()
            != self
                .progress
                .pending
                .as_ref()
                .ok_or("Missing Long batch")?
                .plan
        {
            return Err("Restored Long plan mismatch".into());
        }
        let mut episodes = Vec::with_capacity(4);
        for (i, bytes) in bytes.iter().enumerate() {
            self.save(Event::AuditStart { member: i })?;
            let began = Instant::now();
            // Handle/infrastructure errors are outer errors, never failed cohorts.
            let result = match self.current.as_ref().ok_or("Missing Long current owner")? {
                Current::Count1(p) => {
                    let handle = UpdatedPublicRlHandle::new(p)?;
                    audit_rl_record_bytes(bytes, &handle)
                        .and_then(ValidatedRlPolicyEpisode::from_audited)
                }
                Current::Long(p) => {
                    let handle = LongRlHandle::new(p)?;
                    audit_long_rl_record_bytes(bytes, &handle)
                        .and_then(ValidatedRlPolicyEpisode::from_long_audited)
                }
            };
            self.save(Event::AuditReturn {
                member: i,
                passed: result.is_ok(),
                error: result.as_ref().err().map(|s| capped(s)),
                seconds: began.elapsed().as_secs_f64(),
            })?;
            episodes.push(result);
        }
        Ok(plan.finish(
            episodes
                .try_into()
                .map_err(|_| "Expected four audited episodes")?,
        ))
    }
    fn update(&mut self, cohort: &ValidatedRlCohort) -> Result<(), String> {
        self.save(Event::GradientStart)?;
        let began = Instant::now();
        let result = public_policy_long::prepare_long_step(
            self.current_parent(),
            cohort,
            &self.spec.step()?,
            self.spec.limits()?,
        );
        let seconds = began.elapsed().as_secs_f64();
        let counts = Some([
            cohort.total_callbacks(),
            cohort.total_candidate_rows(),
            cohort.total_singletons(),
        ]);
        match result {
            Err(error) => self.save(Event::Resolve {
                outcome: "failed".into(),
                error: Some(capped(&error)),
                counts,
                report: None,
                owner: None,
                seconds: Some(seconds),
            }),
            Ok(step) => {
                let report = Some(step.report().checksum().into());
                if let Some(artifact) = step.artifact() {
                    let raw = json_bytes(
                        &serde_json::to_value(artifact).map_err(|e| e.to_string())?,
                        OWNER_BYTES,
                    )?;
                    let owner = OwnerPin {
                        file: pin(&format!("owner-{}.json", artifact.update_count()), &raw),
                        count: artifact.update_count(),
                        artifact: artifact.checksum().into(),
                        lineage: artifact.report().lineage_checksum().into(),
                    };
                    if self
                        .progress
                        .metadata_bytes
                        .checked_add(raw.len() as u64)
                        .ok_or("Long metadata overflow")?
                        > self.spec.max_metadata_bytes
                    {
                        return Err(
                            "Long model storage byte cap reached with pending gradient".into()
                        );
                    }
                    publish(&self.directory, &owner.file.name, &raw)?;
                    self.save(Event::Resolve {
                        outcome: "updated".into(),
                        error: None,
                        counts,
                        report,
                        owner: Some(owner),
                        seconds: Some(seconds),
                    })?;
                    // Publication commits before the infallible sealed-owner swap.
                    // A failed Resolve leaves the retained parent unchanged.
                    let old = self.current.take().expect("session retains its owner");
                    let families = match old {
                        Current::Count1(_) => std::mem::take(&mut self.trained),
                        Current::Long(p) => public_policy_long::take_families(*p),
                    };
                    match step.commit(families) {
                        LongStepOutcome::Updated(p) => {
                            self.current = Some(Current::Long(Box::new(p)));
                        }
                        LongStepOutcome::NoChange(_) => unreachable!(),
                    }
                    Ok(())
                } else {
                    self.save(Event::Resolve {
                        outcome: "noChange".into(),
                        error: None,
                        counts,
                        report,
                        owner: None,
                        seconds: Some(seconds),
                    })
                }
            }
        }
    }
    fn execute(&mut self, resume_pending: bool, abandon: Option<&str>) -> Result<String, String> {
        if let Some(reason) = abandon {
            if self.progress.pending.is_none() || reason.is_empty() || reason.chars().count() > 1024
            {
                return Err("Explicit abandon requires a pending batch and bounded reason".into());
            }
            self.save(Event::Resolve {
                outcome: "abandoned".into(),
                error: Some(reason.into()),
                counts: None,
                report: None,
                owner: None,
                seconds: None,
            })?;
            return Ok("abandoned".into());
        }
        let mut saved_only = resume_pending;
        loop {
            if self.progress.pending.is_none() {
                if self.progress.known.accepted >= self.spec.decision_budget {
                    return Ok("decisionBudget".into());
                }
                if self.progress.owner.count >= self.spec.max_updates {
                    return Ok("maxUpdates".into());
                }
                if self.progress.reservations >= self.spec.max_batches {
                    return Ok("batchCapacity".into());
                }
                if self
                    .consumed
                    .len()
                    .checked_add(4)
                    .ok_or("Consumed capacity overflow")?
                    > self.spec.max_families
                {
                    return Ok("consumedFamilyCapacity".into());
                }
                let record_reserve = 4 * MAX_RECORD_BYTES as u64;
                if self
                    .progress
                    .record_bytes
                    .checked_add(record_reserve)
                    .ok_or("Record capacity overflow")?
                    > self.spec.max_record_bytes
                {
                    return Ok("storageByteCap".into());
                }
                let metadata_reserve = OWNER_BYTES as u64 + 19 * MAX_CHECKPOINT_BYTES as u64;
                if self
                    .progress
                    .metadata_bytes
                    .checked_add(metadata_reserve)
                    .ok_or("Metadata reserve overflow")?
                    > self.spec.max_metadata_bytes
                    || self
                        .sequence
                        .checked_add(19)
                        .ok_or("Journal capacity overflow")?
                        > self.spec.max_journal_entries
                {
                    return Ok("storageByteCap".into());
                }
                if self.progress.cursor >= self.spec.end_seed() {
                    return Ok("seedRange".into());
                }
                self.reserve()?;
                saved_only = false;
            }
            if self
                .progress
                .pending
                .as_ref()
                .is_some_and(|b| b.gradient_unknown)
            {
                return Ok("gradientUnknown".into());
            }
            if !saved_only {
                self.collect()?;
            }
            let batch = self.progress.pending.as_ref().ok_or("Missing Long batch")?;
            if !saved_only && batch.members.iter().any(|m| m.error.is_some()) {
                self.save(Event::Resolve {
                    outcome: "failed".into(),
                    error: Some("A whole-four collection call returned an error".into()),
                    counts: None,
                    report: None,
                    owner: None,
                    seconds: None,
                })?;
                return Ok("batchFailed".into());
            }
            if saved_only {
                resume_headroom(&self.spec, &self.progress, self.sequence)?;
            }
            let bytes = self.preflight_records()?;
            match self.audit(bytes)? {
                Ok(cohort) => self.update(&cohort)?,
                Err(error) => {
                    self.save(Event::Resolve {
                        outcome: "failed".into(),
                        error: Some(capped(&error)),
                        counts: None,
                        report: None,
                        owner: None,
                        seconds: None,
                    })?;
                    return Ok("batchFailed".into());
                }
            }
            if let Some(outcome) = &self.progress.last_outcome {
                if outcome == "noChange" {
                    return Ok("NoChange".into());
                }
                if outcome == "failed" {
                    return Ok("batchFailed".into());
                }
            }
            saved_only = false;
        }
    }
    fn report(&self, reason: String, wall_seconds: f64) -> LongRunReport {
        LongRunReport {
            schema: "tzolkin-public-rl-long-report-v2",
            decision_budget_reached: reason == "decisionBudget",
            stop_reason: reason,
            consumed_families: self.consumed.len(),
            progress: self.progress.clone(),
            bootstrap_complete_cohort_counts: self.bootstrap_complete_counts,
            bootstrap_failed_prefix_work: None,
            instrumented_forward_calls: None,
            evaluations_performed: 0,
            invocation_wall_seconds: wall_seconds,
            checkpoint_file: head_name(self.sequence),
            checkpoint_sha256: self.tip.clone(),
            producer_authenticated: false,
        }
    }
    /// Start a new single-writer directory, retaining the exact v1 bootstrap
    /// bytes. No collection runs until `run` is explicitly called.
    pub fn from_count1(
        spec: LongRunSpec,
        prepared: &PreparedPublicPolicy,
        count1_bytes: &[u8],
        count1_sha256: &str,
        directory: &Path,
    ) -> Result<Self, String> {
        let directory = local(directory, true)?;
        let mut session = Self::bootstrap(spec, prepared, count1_bytes, count1_sha256, directory)?;
        absent(&session.directory)?;
        fs::create_dir(&session.directory).map_err(|e| e.to_string())?;
        publish(&session.directory, "bootstrap.json", count1_bytes)?;
        let raw = json_bytes(
            &session
                .current
                .as_ref()
                .ok_or("Missing Count1 owner")?
                .value()?,
            OWNER_BYTES,
        )?;
        publish(&session.directory, "owner-1.json", &raw)?;
        session.save(Event::Start)?;
        Ok(session)
    }
    /// Content-consistent restoration from an externally pinned head. No native
    /// audit or gradient is executed during restoration.
    pub fn restore(
        spec: LongRunSpec,
        prepared: &PreparedPublicPolicy,
        checkpoint_bytes: &[u8],
        expected_sha256: &str,
        directory: &Path,
    ) -> Result<Self, String> {
        restore(spec, prepared, checkpoint_bytes, expected_sha256, directory)
    }
    /// Resume pending work only from four saved pinned records. An unknown
    /// gradient stops; explicit abandon resolves it without reusing families.
    pub fn run(&mut self, abandon: Option<&str>) -> Result<LongRunReport, String> {
        let began = Instant::now();
        let reason = self.execute(self.progress.pending.is_some(), abandon)?;
        Ok(self.report(reason, began.elapsed().as_secs_f64()))
    }
}

/// Known returned collection work includes failed prefixes. A stopped pending
/// call has unknown work, not zero. Timings are measured synchronous phases;
/// no deadline, statistical quality, calibration or strength is implied.
#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct LongRunReport {
    schema: &'static str,
    stop_reason: String,
    decision_budget_reached: bool,
    consumed_families: usize,
    progress: Progress,
    bootstrap_complete_cohort_counts: [u64; 3],
    bootstrap_failed_prefix_work: Option<()>,
    instrumented_forward_calls: Option<u64>,
    evaluations_performed: u64,
    invocation_wall_seconds: f64,
    checkpoint_file: String,
    checkpoint_sha256: String,
    producer_authenticated: bool,
}
impl LongRunReport {
    pub fn decision_budget_reached(&self) -> bool {
        self.decision_budget_reached
    }
    pub fn stop_reason(&self) -> &str {
        &self.stop_reason
    }
}

fn apply(
    spec: &LongRunSpec,
    p: &mut Progress,
    event: &Event,
    consumed: &BTreeSet<String>,
) -> Result<(), String> {
    match event {
        Event::Start => {
            if p.reservations != 0 || p.pending.is_some() || p.owner.count != 1 {
                return Err("Start requires the Count1 root".into());
            }
        }
        Event::Reserve {
            configs,
            plan,
            cursor,
        } => {
            if p.pending.is_some()
                || p.reservations >= spec.max_batches
                || p.owner.count >= spec.max_updates
                || consumed
                    .len()
                    .checked_add(4)
                    .ok_or("Long capacity overflow")?
                    > spec.max_families
                || !checkpoint_digest(plan)
                || *cursor < p.cursor
            {
                return Err("Invalid Long reservation transition".into());
            }
            let mut expected_cursor = p.cursor;
            let expected = spec.configs(&mut expected_cursor, p.next_ordinal, consumed)?;
            if configs.as_ref() != &expected || *cursor != expected_cursor {
                return Err("Saved reservation differs from the fixed metadata allocator".into());
            }
            p.next_ordinal = p
                .next_ordinal
                .checked_add(4)
                .ok_or("Long ordinal overflow")?;
            p.reservations += 1;
            p.cursor = *cursor;
            p.last_outcome = None;
            p.last_error = None;
            p.pending = Some(Batch {
                configs: configs.as_ref().clone(),
                plan: plan.clone(),
                parent: p.owner.artifact.clone(),
                parent_count: p.owner.count,
                members: std::array::from_fn(|_| Member::default()),
                gradient_unknown: false,
            });
        }
        Event::CollectStart { member } => {
            let b = p.pending.as_mut().ok_or("No Long batch")?;
            if *member >= 4
                || b.members[*member].started
                || b.gradient_unknown
                || b.members[..*member].iter().any(|m| !m.returned)
            {
                return Err("Invalid collection start".into());
            }
            let m = &mut b.members[*member];
            m.started = true;
            p.collect.started += 1;
            p.unknown_collections += 1;
        }
        Event::CollectReturn {
            member,
            record,
            work,
            error,
            seconds,
        } => {
            let b = p.pending.as_mut().ok_or("No Long batch")?;
            let m = b.members.get_mut(*member).ok_or("Long member index")?;
            if !m.started
                || m.returned
                || record.is_some() != work.is_some()
                || record.is_some() == error.is_some()
                || error
                    .as_ref()
                    .is_some_and(|e| e.is_empty() || e.chars().count() > 1024)
            {
                return Err("Invalid collection return".into());
            }
            if let (Some(pin), Some(w)) = (record, work) {
                if !pin.valid(MAX_RECORD_BYTES)
                    || pin.name != record_name(p.reservations, *member)
                    || !w.valid(&b.configs[*member])
                {
                    return Err("Invalid saved collection pin/work".into());
                }
                p.record_bytes = p
                    .record_bytes
                    .checked_add(pin.bytes as u64)
                    .ok_or("Long record bytes overflow")?;
                if p.record_bytes > spec.max_record_bytes {
                    return Err("Long record storage cap".into());
                }
                p.known.add(w)?;
                p.unknown_collections -= 1;
            }
            p.collect.returned(*seconds)?;
            m.returned = true;
            m.record = record.clone();
            m.work = work.clone();
            m.error = error.clone();
        }
        Event::AuditStart { member } => {
            let b = p.pending.as_mut().ok_or("No Long batch")?;
            if b.gradient_unknown || b.members.iter().any(|m| m.record.is_none()) {
                return Err("Audit requires all four pinned records".into());
            }
            let m = b.members.get_mut(*member).ok_or("Long member index")?;
            m.audit_started = true;
            m.audit_passed = false;
            p.audit.started += 1;
        }
        Event::AuditReturn {
            member,
            passed,
            error,
            seconds,
        } => {
            let m = p
                .pending
                .as_mut()
                .ok_or("No Long batch")?
                .members
                .get_mut(*member)
                .ok_or("Long member index")?;
            if !m.audit_started
                || *passed == error.is_some()
                || error.as_ref().is_some_and(|e| e.chars().count() > 1024)
            {
                return Err("Invalid audit return".into());
            }
            m.audit_started = false;
            m.audit_passed = *passed;
            p.audit.returned(*seconds)?;
        }
        Event::GradientStart => {
            let b = p.pending.as_mut().ok_or("No Long batch")?;
            if b.gradient_unknown || b.members.iter().any(|m| !m.audit_passed) {
                return Err("Gradient requires the whole audited batch".into());
            }
            b.gradient_unknown = true;
            p.gradient.started += 1;
        }
        Event::Resolve {
            outcome,
            error,
            counts,
            report,
            owner,
            seconds,
        } => {
            let b = p.pending.as_ref().ok_or("No Long reservation to resolve")?;
            if !matches!(
                outcome.as_str(),
                "updated" | "noChange" | "failed" | "abandoned"
            ) || error
                .as_ref()
                .is_some_and(|e| e.is_empty() || e.chars().count() > 1024)
                || matches!(outcome.as_str(), "failed" | "abandoned") != error.is_some()
                || matches!(outcome.as_str(), "updated" | "noChange") != report.is_some()
                || report.as_ref().is_some_and(|r| !checkpoint_digest(r))
                || (seconds.is_some() && !b.gradient_unknown)
                || (outcome == "updated") != owner.is_some()
                || (matches!(outcome.as_str(), "updated" | "noChange")
                    && (counts.is_none() || !b.gradient_unknown))
            {
                return Err("Invalid Long resolution".into());
            }
            if let Some(c) = counts {
                let mut actual = [0usize; 3];
                for m in &b.members {
                    let w = m.work.as_ref().ok_or("Cohort lacks returned work")?;
                    for (a, n) in actual
                        .iter_mut()
                        .zip([w.accepted, w.accepted_rows, w.singletons])
                    {
                        *a = a.checked_add(n as usize).ok_or("Cohort count overflow")?;
                    }
                }
                if c != &actual || b.members.iter().any(|m| !m.audit_passed) {
                    return Err("Whole-cohort counts/audits differ".into());
                }
                for (a, n) in p.complete_counts.iter_mut().zip(c) {
                    *a = a.checked_add(*n as u64).ok_or("Complete count overflow")?;
                }
            }
            if let Some(o) = owner {
                if !o.file.valid(OWNER_BYTES)
                    || !checkpoint_digest(&o.artifact)
                    || o.count != p.owner.count + 1
                    || o.count > spec.max_updates
                    || o.file.name != format!("owner-{}.json", o.count)
                    || o.lineage
                        != public_policy_long::extend_lineage(
                            &p.owner.lineage,
                            &b.configs
                                .each_ref()
                                .map(|c| seed_family_id(c.environment_seed)),
                        )?
                {
                    return Err("Invalid immediate Long owner transition".into());
                }
                p.metadata_bytes = p
                    .metadata_bytes
                    .checked_add(o.file.bytes as u64)
                    .ok_or("Model byte overflow")?;
                p.owner = o.clone();
            }
            if let Some(seconds) = seconds {
                p.gradient.returned(*seconds)?;
            }
            p.due = p.known.accepted / spec.evaluation_interval;
            p.last_outcome = Some(outcome.clone());
            p.last_error = error.clone();
            p.pending = None;
        }
    }
    Ok(())
}

fn restore(
    spec: LongRunSpec,
    prepared: &PreparedPublicPolicy,
    bytes: &[u8],
    expected: &str,
    directory: &Path,
) -> Result<LongTrainingSession, String> {
    if !checkpoint_digest(expected) || sha(bytes) != expected {
        return Err("Long checkpoint external raw SHA mismatch".into());
    }
    let supplied: Checkpoint = closed(bytes, MAX_CHECKPOINT_BYTES)?;
    let directory = local(directory, false)?;
    let original = read_pin(&directory, &supplied.root, OWNER_BYTES)?;
    let mut session = LongTrainingSession::bootstrap(
        spec,
        prepared,
        &original,
        &supplied.root.sha256,
        directory,
    )?;
    if supplied.root != session.root
        || supplied.spec_checksum != session.spec.checksum()?
        || supplied.sequence == 0
        || supplied.sequence > session.spec.max_journal_entries
    {
        return Err("Long checkpoint root/spec/journal mismatch".into());
    }
    let root_source = match session.current.as_ref().ok_or("Missing bootstrap owner")? {
        Current::Count1(p) => p.artifact().report().bc_source().clone(),
        Current::Long(_) => unreachable!(),
    };
    let root_owner_raw = read_pin(
        &session.directory,
        &session.progress.owner.file,
        OWNER_BYTES,
    )?;
    if unique_json(&root_owner_raw, OWNER_BYTES)?
        != session
            .current
            .as_ref()
            .ok_or("Missing Count1 root")?
            .value()?
    {
        return Err("Stored Count1 root differs from pinned bootstrap".into());
    }
    let mut latest_report = None;
    let mut names = BTreeSet::from(["bootstrap.json".to_owned(), "owner-1.json".to_owned()]);
    for sequence in 1..=supplied.sequence {
        let name = head_name(sequence);
        let raw = read_bounded(&session.directory.join(&name), MAX_CHECKPOINT_BYTES)?;
        let entry: Checkpoint = closed(&raw, MAX_CHECKPOINT_BYTES)?;
        if entry.schema != "tzolkin-rl-long-checkpoint-v2"
            || matches!(entry.event, Event::Start) != (sequence == 1)
            || entry.sequence != sequence
            || entry.previous != session.tip
            || entry.root != session.root
            || entry.spec_checksum != supplied.spec_checksum
        {
            return Err("Long journal chain differs".into());
        }
        let parent_lineage = session.progress.owner.lineage.clone();
        let batch_before = session.progress.pending.clone();
        apply(
            &session.spec,
            &mut session.progress,
            &entry.event,
            &session.consumed,
        )?;
        if entry.progress != session.progress {
            return Err("Long journal logical progress differs".into());
        }
        match &entry.event {
            Event::Reserve { configs, .. } => {
                session
                    .consumed
                    .extend(configs.iter().map(|c| seed_family_id(c.environment_seed)));
            }
            Event::CollectReturn {
                record: Some(p),
                work: Some(w),
                ..
            } => {
                let raw_record = read_pin(&session.directory, p, MAX_RECORD_BYTES)?;
                if record_work(&raw_record)? != *w {
                    return Err("Saved record logical counters differ".into());
                }
                names.insert(p.name.clone());
            }
            Event::Resolve {
                outcome,
                report,
                owner: Some(o),
                counts,
                ..
            } if outcome == "updated" => {
                let batch = batch_before.ok_or("Updated journal lacks reservation")?;
                session.trained.extend(
                    batch
                        .configs
                        .iter()
                        .map(|c| seed_family_id(c.environment_seed)),
                );
                read_pin(&session.directory, &o.file, OWNER_BYTES)?;
                latest_report = Some((
                    report.clone().ok_or("Updated report pin missing")?,
                    batch,
                    parent_lineage,
                    counts.ok_or("Updated counts missing")?,
                ));
                names.insert(o.file.name.clone());
            }
            _ => {}
        }
        session.progress.metadata_bytes = session
            .progress
            .metadata_bytes
            .checked_add(raw.len() as u64)
            .ok_or("Long metadata overflow")?;
        if session.progress.metadata_bytes > session.spec.max_metadata_bytes {
            return Err("Long metadata storage cap".into());
        }
        session.sequence = sequence;
        session.tip = sha(&raw);
        names.insert(name);
        if sequence == supplied.sequence && (raw != bytes || session.tip != expected) {
            return Err("Supplied Long head is not the pinned stored head".into());
        }
    }
    // A stale head never skips already published history or temporary evidence.
    for item in fs::read_dir(&session.directory).map_err(|e| e.to_string())? {
        let item = item.map_err(|e| e.to_string())?;
        local(&item.path(), false)?;
        let name = item
            .file_name()
            .to_str()
            .ok_or("Non-UTF8 Long file name")?
            .to_owned();
        if !names.contains(&name) {
            return Err(
                "Uncommitted/future/stale Long files require explicit investigation".into(),
            );
        }
    }
    let owner_value = unique_json(
        &read_pin(
            &session.directory,
            &session.progress.owner.file,
            OWNER_BYTES,
        )?,
        OWNER_BYTES,
    )?;
    if session.progress.owner.count == 1 {
        if owner_value
            != session
                .current
                .as_ref()
                .ok_or("Missing Count1 root")?
                .value()?
        {
            return Err("Current Count1 bytes differ from bootstrap".into());
        }
    } else {
        let owner = public_policy_long::restore_checkpoint_owner(
            owner_value,
            std::mem::take(&mut session.trained),
            session.spec.limits()?,
        )?;
        let report = owner.artifact().report();
        let (checksum, batch, parent_lineage, counts) =
            latest_report.ok_or("Long current owner lacks an updated journal receipt")?;
        if report.root_init_checksum() != session.root_init
            || report.root_count1_checksum() != session.root_count1
            || report.parent_checksum() != batch.parent
            || report.parent_update_count() != batch.parent_count
            || report.checksum() != checksum
            || report.config() != &session.spec.step()?
            || report.bc_source() != &root_source
            || report.plan_checksum() != batch.plan
            || report.counts() != counts
            || report.parent_lineage() != parent_lineage
            || report.lineage_checksum() != session.progress.owner.lineage
            || report.rollout_families()
                != &batch
                    .configs
                    .each_ref()
                    .map(|c| seed_family_id(c.environment_seed))
            || owner.artifact().checksum() != session.progress.owner.artifact
            || owner.artifact().update_count() != session.progress.owner.count
        {
            return Err("Current Long owner/root/journal bindings differ".into());
        }
        session.current = Some(Current::Long(Box::new(owner)));
    }
    Ok(session)
}
fn resume_headroom(spec: &LongRunSpec, progress: &Progress, sequence: u64) -> Result<(), String> {
    // Re-auditing four records may append eight events, then gradient/resolve.
    // Refuse before native work, retaining pending and remaining headroom.
    let metadata = OWNER_BYTES as u64 + 10 * MAX_CHECKPOINT_BYTES as u64;
    if sequence
        .checked_add(10)
        .is_none_or(|n| n > spec.max_journal_entries)
        || progress
            .metadata_bytes
            .checked_add(metadata)
            .is_none_or(|n| n > spec.max_metadata_bytes)
    {
        return Err("Insufficient Long headroom before resume audit; pending unchanged".into());
    }
    Ok(())
}

fn checked_configs(configs: &[Config; 4]) -> Result<[NativeStochasticConfig; 4], String> {
    configs
        .iter()
        .map(Config::checked_config)
        .collect::<Result<Vec<_>, _>>()?
        .try_into()
        .map_err(|_| "Expected four configurations".into())
}
fn record_work(bytes: &[u8]) -> Result<Work, String> {
    let value = unique_json(bytes, MAX_RECORD_BYTES)?;
    let get = |name| {
        value["counts"][name]
            .as_u64()
            .ok_or("Missing returned record counter")
    };
    let mut rows = 0u64;
    let mut singletons = 0u64;
    for c in value["callbacks"].as_array().ok_or("Missing callbacks")? {
        if !c["sample"].is_null() {
            let n = c["sample"]["trace"]["legalCount"]
                .as_u64()
                .ok_or("Missing legal count")?;
            rows = rows.checked_add(n).ok_or("Returned row overflow")?;
            singletons += u64::from(n == 1);
        }
    }
    Ok(Work {
        callbacks: get("observedCallbacks")?,
        attempts: get("samplerAttempts")?,
        accepted: get("acceptedSamples")?,
        applied: get("applySuccesses")?,
        reserved_rows: get("candidateRowsReserved")?,
        accepted_rows: rows,
        singletons,
    })
}
fn closed<T: serde::de::DeserializeOwned + Serialize>(
    bytes: &[u8],
    max: usize,
) -> Result<T, String> {
    let value = unique_json(bytes, max)?;
    let typed = serde_json::from_value::<T>(value.clone()).map_err(|e| e.to_string())?;
    if serde_json::to_value(&typed).map_err(|e| e.to_string())? != value {
        return Err("Unknown or omitted Long fields".into());
    }
    Ok(typed)
}
fn json_bytes(value: &impl Serialize, max: usize) -> Result<Vec<u8>, String> {
    crate::public_stochastic_record::serialized_size(value, max)?;
    let bytes = serde_json::to_vec(value).map_err(|e| e.to_string())?;
    if bytes.len() > max {
        return Err("Long serialized byte cap exceeded".into());
    }
    Ok(bytes)
}
fn sha(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}
fn capped(s: &str) -> String {
    s.chars().take(1024).collect()
}
fn pin(name: &str, bytes: &[u8]) -> Pin {
    Pin {
        name: name.into(),
        bytes: bytes.len(),
        sha256: sha(bytes),
    }
}
fn head_name(sequence: u64) -> String {
    format!("checkpoint-{sequence:016}.json")
}
fn record_name(batch: u64, member: usize) -> String {
    format!("batch-{batch:016}-member-{member}.json")
}
fn read_pin(directory: &Path, p: &Pin, max: usize) -> Result<Vec<u8>, String> {
    if !p.valid(max) {
        return Err("Invalid Long file pin".into());
    }
    let bytes = read_bounded(&directory.join(&p.name), max)?;
    if bytes.len() != p.bytes || sha(&bytes) != p.sha256 {
        return Err("Long file bytes/SHA mismatch".into());
    }
    Ok(bytes)
}
/// Local regular-file reading only. This is a single-writer persistence boundary,
/// not a hostile-filesystem or producer-authentication claim.
pub fn read_bounded(path: &Path, max: usize) -> Result<Vec<u8>, String> {
    if max == 0 || max > MAX_RECORD_BYTES {
        return Err("Invalid Long read byte limit".into());
    }
    let path = local(path, false)?;
    let meta = fs::symlink_metadata(&path).map_err(|e| e.to_string())?;
    if !meta.is_file() || meta.len() > max as u64 {
        return Err("Long input is not a bounded regular file".into());
    }
    let mut bytes = Vec::new();
    fs::File::open(path)
        .map_err(|e| e.to_string())?
        .take(max as u64 + 1)
        .read_to_end(&mut bytes)
        .map_err(|e| e.to_string())?;
    if bytes.len() > max {
        return Err("Long file grew beyond its byte cap".into());
    }
    Ok(bytes)
}
fn local(path: &Path, allow_missing: bool) -> Result<PathBuf, String> {
    let text = path.to_string_lossy();
    if text.starts_with('/') && text.get(1..2).is_some_and(|s| s == "/" || s == "\\")
        || text.starts_with('\\')
        || text.contains("://")
        || path.components().any(|c| matches!(c, Component::ParentDir))
        || (text.as_bytes().get(1) == Some(&b':')
            && !matches!(text.as_bytes().get(2), Some(b'/' | b'\\')))
    {
        return Err("Long paths must be local without parent traversal".into());
    }
    let path = if path.is_absolute() {
        path.to_owned()
    } else {
        std::env::current_dir()
            .map_err(|e| e.to_string())?
            .join(path)
    };
    let mut part = PathBuf::new();
    for component in path.components() {
        part.push(component);
        match fs::symlink_metadata(&part) {
            Ok(m) => {
                if m.is_symlink() {
                    return Err("Long path contains a link".into());
                }
                #[cfg(windows)]
                {
                    use std::os::windows::fs::MetadataExt;
                    if m.file_attributes() & 0x400 != 0 {
                        return Err("Long path contains a reparse point".into());
                    }
                }
            }
            Err(e) if allow_missing && e.kind() == std::io::ErrorKind::NotFound && part == path => {
            }
            Err(e) => return Err(e.to_string()),
        }
    }
    Ok(path)
}
fn absent(path: &Path) -> Result<(), String> {
    match fs::symlink_metadata(path) {
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Ok(_) => Err("Long destination already exists".into()),
        Err(e) => Err(e.to_string()),
    }
}
fn publish(directory: &Path, name: &str, bytes: &[u8]) -> Result<Pin, String> {
    static TEMP: AtomicU64 = AtomicU64::new(0);
    local(directory, false)?;
    let destination = local(&directory.join(name), true)?;
    absent(&destination)?;
    let temporary = directory.join(format!(
        "{name}.{}-{}.tmp",
        std::process::id(),
        TEMP.fetch_add(1, Ordering::Relaxed)
    ));
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&temporary)
        .map_err(|e| e.to_string())?;
    file.write_all(bytes)
        .and_then(|_| file.sync_all())
        .map_err(|e| e.to_string())?;
    drop(file);
    fs::hard_link(&temporary, &destination).map_err(|e| e.to_string())?;
    let expected = pin(name, bytes);
    read_pin(directory, &expected, bytes.len())?;
    fs::remove_file(&temporary).map_err(|e| e.to_string())?;
    Ok(expected)
}

#[cfg(test)]
mod tests {
    use super::*;
    mod temp_root {
        include!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/support/temp_root.rs"
        ));
    }
    fn spec() -> LongRunSpec {
        serde_json::from_value(serde_json::json!({
            "schema": SPEC_SCHEMA, "samplingSeed": u64::MAX, "replicateOrdinal": 7,
            "environmentSeedStart": 17, "maxSeedCandidates": 1000, "decisionBudget": 100,
            "evaluationInterval": 10, "maxUpdates": 100, "maxFamilies": 512, "maxBatches": 10,
            "maxJournalEntries": 240, "learningRate": 1e-4, "maxAbsActualDelta": 0.01,
            "maxGradientRows": 1000, "maxCallbacks": 8, "maxCandidateRows": 1000,
            "maxSourceBytes": 1048576, "maxTraceBytes": 1048576, "maxMetadataBytes": 67108864,
            "maxRecordBytes": MAX_STORAGE, "excludedFamilies": []
        }))
        .unwrap()
    }
    fn progress(spec: &LongRunSpec) -> Progress {
        // Metadata predicate fixture only: no sealed owner/episode is created.
        serde_json::from_value(serde_json::json!({
            "owner": {"file": pin("owner-1.json", b"fixture"), "count": 1,
                "artifact": "a".repeat(64), "lineage": "b".repeat(64)},
            "cursor": spec.environment_seed_start, "nextOrdinal": 40, "reservations": 0,
            "pending": null, "known": Work::default(), "completeCounts": [0,0,0],
            "unknownCollections": 0, "collect": Phase::default(), "audit": Phase::default(),
            "gradient": Phase::default(), "recordBytes": 0, "metadataBytes": 0,
            "due": 0, "lastOutcome": null, "lastError": null
        }))
        .unwrap()
    }
    #[test]
    fn fixed_allocator_pending_and_consumed_boundaries() {
        let spec = spec();
        spec.validate().unwrap();
        let mut p = progress(&spec);
        let mut consumed = BTreeSet::new();
        let mut cursor = p.cursor;
        let configs = spec
            .configs(&mut cursor, p.next_ordinal, &consumed)
            .unwrap();
        let event = Event::Reserve {
            configs: Box::new(configs.clone()),
            plan: "c".repeat(64),
            cursor,
        };
        for bad in [0, 1] {
            let mut e = event.clone();
            if let Event::Reserve { configs, .. } = &mut e {
                if bad == 0 {
                    configs[0].max_callbacks += 1;
                } else {
                    configs[0].episode_ordinal = format!("{:016x}", 39);
                }
            }
            assert!(apply(&spec, &mut p.clone(), &e, &consumed).is_err());
        }
        let mut excluded = spec.clone();
        excluded
            .excluded_families
            .push(seed_family_id(configs[0].environment_seed));
        assert!(apply(&excluded, &mut p.clone(), &event, &consumed).is_err());
        apply(&spec, &mut p, &event, &consumed).unwrap();
        consumed.extend(configs.iter().map(|c| seed_family_id(c.environment_seed)));
        assert_eq!(p.next_ordinal, 44);
        assert!(apply(&spec, &mut p.clone(), &event, &consumed).is_err());
        apply(
            &spec,
            &mut p,
            &Event::Resolve {
                outcome: "abandoned".into(),
                error: Some("explicit".into()),
                counts: None,
                report: None,
                owner: None,
                seconds: None,
            },
            &consumed,
        )
        .unwrap();
        assert_eq!(p.owner.count, 1);
        assert_eq!(consumed.len(), 4);
        let next = spec
            .configs(&mut cursor, p.next_ordinal, &consumed)
            .unwrap();
        assert!(
            next.iter()
                .all(|c| !consumed.contains(&seed_family_id(c.environment_seed)))
        );
        let mut cap = spec.clone();
        cap.max_families = 7;
        assert!(
            apply(
                &cap,
                &mut p,
                &Event::Reserve {
                    configs: Box::new(next),
                    plan: "d".repeat(64),
                    cursor
                },
                &consumed
            )
            .is_err()
        );
    }
    #[test]
    fn closed_nested_null_duplicate_and_eof() {
        let m = serde_json::to_value(Member::default()).unwrap();
        closed::<Member>(&serde_json::to_vec(&m).unwrap(), 1024).unwrap();
        for field in ["record", "work", "error"] {
            let mut bad = m.clone();
            bad.as_object_mut().unwrap().remove(field);
            assert!(closed::<Member>(&serde_json::to_vec(&bad).unwrap(), 1024).is_err());
        }
        for bad in [
            b"{\"started\":false,\"started\":false}".as_slice(),
            b"{} {}",
        ] {
            assert!(closed::<Member>(bad, 1024).is_err());
        }
        let mut bad = serde_json::to_value(spec()).unwrap();
        bad["extra"] = Value::Null;
        assert!(LongRunSpec::from_json(&serde_json::to_vec(&bad).unwrap()).is_err());
        let mut nested = m;
        nested["record"] =
            serde_json::json!({"name":"x.json","bytes":1,"sha256":"a".repeat(64),"extra":null});
        assert!(closed::<Member>(&serde_json::to_vec(&nested).unwrap(), 1024).is_err());
    }
    #[test]
    fn new_only_pin_io_preserves_collisions_and_partial_evidence() {
        let directory = temp_root::create("rl-long-pins").unwrap();
        fs::write(directory.join("foreign.tmp"), b"retained").unwrap();
        let p = publish(&directory, "checkpoint.json", b"original").unwrap();
        assert!(publish(&directory, "checkpoint.json", b"replacement").is_err());
        assert_eq!(read_pin(&directory, &p, 32).unwrap(), b"original");
        assert_eq!(
            fs::read(directory.join("foreign.tmp")).unwrap(),
            b"retained"
        );
        let mut bad = p.clone();
        bad.sha256 = "0".repeat(64);
        assert!(read_pin(&directory, &bad, 32).is_err());
        bad.name = "missing.json".into();
        assert!(read_pin(&directory, &bad, 32).is_err());
        assert!(read_bounded(&directory, 32).is_err());
        assert!(local(Path::new("C:relative"), true).is_err());
        fs::remove_dir_all(directory).unwrap();
    }
}
