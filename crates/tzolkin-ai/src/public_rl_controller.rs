//! Fixed-plan batch-boundary driving of the existing count0..10 contracts.
//! Evaluation intervals are notices, not evaluation results. Saved content pins
//! establish consistency, not producer, chronology or optimization authenticity.
use std::collections::BTreeSet;
use std::fs::{self, OpenOptions};
use std::io::{Read, Write};
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::dataset::{DatasetSplit, seed_family_id, split_for_family};
use crate::public_native::PreparedPublicPolicy;
use crate::public_policy_episode::ValidatedPolicyEpisode;
use crate::public_policy_repeat::{MAX_UPDATE_COUNT, RlUpdateParent};
use crate::public_policy_update::OneStepConfig;
use crate::public_rl_artifact::InitializedPublicRlPolicy;
use crate::public_rl_native::{RepeatedPublicRlHandle, UpdatedPublicRlHandle};
use crate::public_rl_policy_episode::ValidatedRlPolicyEpisode;
use crate::public_rl_repeated_stochastic_native::{
    audit_repeated_rl_record_bytes, collect_repeated_rl_native, encode_repeated_rl_record,
};
use crate::public_rl_session::{RlTrainingSession, SessionConfig, SessionProgress};
use crate::public_rl_stochastic_native::{
    audit_rl_record_bytes, collect_rl_native, encode_rl_record,
};
use crate::public_stochastic::{SamplingSeed, SamplingStreamIdentity};
use crate::public_stochastic_native::{CollectionLimits, NativeStochasticConfig, collect_native};
use crate::public_stochastic_record::{
    Config, MAX_RECORD_BYTES, audit_record_bytes, encode_record, unique_json,
};

pub const RUN_SPEC_SCHEMA: &str = "tzolkin-public-rl-run-spec-v1";
pub const STATE_SCHEMA: &str = "tzolkin-public-rl-controller-checkpoint-v1";
pub const MAX_SPEC_BYTES: usize = 64 * 1024;
pub const MAX_STATE_BYTES: usize = 40 * 1024 * 1024;

/// One fixed stream/run. A decision budget stops at a whole-batch boundary;
/// the last batch can overshoot it. No wall deadline interrupts synchronous
/// collection or math: a caller may supervise the process and resume its state.
#[derive(Clone, Deserialize, Serialize, PartialEq)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct RunSpec {
    schema: String,
    sampling_seed: u64,
    replicate_ordinal: u64,
    first_episode_ordinal: u64,
    environment_seed_start: u32,
    max_seed_candidates: u32,
    decision_budget: u64,
    evaluation_interval: u64,
    max_batches: usize,
    learning_rate: f64,
    max_abs_actual_delta: f64,
    max_gradient_rows: usize,
    max_callbacks: usize,
    max_candidate_rows: usize,
    max_source_bytes: usize,
    max_trace_bytes: usize,
    excluded_families: Vec<String>,
}
impl RunSpec {
    pub fn from_json(bytes: &[u8]) -> Result<Self, String> {
        let spec: Self = closed(bytes, MAX_SPEC_BYTES)?;
        spec.validate()?;
        Ok(spec)
    }
    fn validate(&self) -> Result<(), String> {
        if self.schema != RUN_SPEC_SCHEMA
            || self.decision_budget == 0
            || self.evaluation_interval == 0
            || !(1..=10).contains(&self.max_batches)
            || !(1..=1_000_000).contains(&self.max_seed_candidates)
            || self.end_seed() > u32::MAX as u64 + 1
            || self.excluded_families.len() > 512
            || self.excluded_families.iter().any(|s| !digest(s))
            || self.excluded_families.windows(2).any(|w| w[0] >= w[1])
        {
            return Err("Invalid bounded RL run specification".into());
        }
        self.first_episode_ordinal
            .checked_add(4 * self.max_batches as u64)
            .ok_or("Run episode ordinal overflow")?;
        self.session_config()?;
        self.limits()?;
        Ok(())
    }
    fn end_seed(&self) -> u64 {
        self.environment_seed_start as u64 + self.max_seed_candidates as u64
    }
    fn limits(&self) -> Result<CollectionLimits, String> {
        CollectionLimits::new(
            self.max_callbacks,
            self.max_candidate_rows,
            self.max_source_bytes,
            self.max_trace_bytes,
        )
    }
    fn session_config(&self) -> Result<SessionConfig, String> {
        SessionConfig::new(
            OneStepConfig::new(
                self.learning_rate,
                self.max_abs_actual_delta,
                self.max_gradient_rows,
            )?,
            SamplingSeed::new(self.sampling_seed),
            self.replicate_ordinal,
            self.first_episode_ordinal,
        )
    }
    fn allocate(
        &self,
        cursor: &mut u64,
        ordinal: u64,
        mut used: BTreeSet<String>,
    ) -> Result<[NativeStochasticConfig; 4], String> {
        used.extend(self.excluded_families.iter().cloned());
        let mut configs = Vec::with_capacity(4);
        while configs.len() < 4 && *cursor < self.end_seed() {
            let seed = *cursor as u32;
            *cursor += 1;
            let family = seed_family_id(seed);
            if split_for_family(&family)? == DatasetSplit::Train && used.insert(family) {
                let players = if configs.len() < 2 { 3 } else { 4 };
                configs.push(NativeStochasticConfig::new(
                    players,
                    seed,
                    SamplingStreamIdentity::new(
                        SamplingSeed::new(self.sampling_seed),
                        ordinal + configs.len() as u64,
                        self.replicate_ordinal,
                        players,
                    )?,
                    self.limits()?,
                )?);
            }
        }
        configs
            .try_into()
            .map_err(|_| "Run exhausted its fixed metadata seed range".into())
    }
}

#[derive(Clone, Default, Deserialize, Serialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct Phase {
    started_calls: u64,
    returned_calls: u64,
    returned_seconds: f64,
}
impl Phase {
    fn returned(&mut self, began: Instant) {
        self.returned_calls += 1;
        self.returned_seconds += began.elapsed().as_secs_f64();
    }
    fn valid(&self) -> bool {
        self.returned_calls <= self.started_calls
            && self.returned_seconds.is_finite()
            && self.returned_seconds >= 0.0
    }
}
#[derive(Clone, Deserialize, Serialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct Pin {
    bytes: usize,
    sha256: String,
}
#[derive(Clone, Default, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct Work {
    observed_callbacks: u64,
    sampler_attempts: u64,
    accepted_samples: u64,
    applied_choices: u64,
    candidate_rows_reserved: u64,
    accepted_candidate_rows: u64,
    singleton_samples: u64,
}
#[derive(Clone, Default, Deserialize, Serialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct Member {
    collection_started: bool,
    record: Option<Pin>,
    known_work: Option<Work>,
    collection_error: Option<String>,
    audit_passed: bool,
    audit_error: Option<String>,
}
#[derive(Clone, Deserialize, Serialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct Batch {
    configurations: [Config; 4],
    members: [Member; 4],
    outcome: Option<String>,
    error: Option<String>,
}
#[derive(Deserialize, Serialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct State {
    schema: String,
    spec: RunSpec,
    session_json: String,
    session_sha256: String,
    seed_cursor: u64,
    seed_range_exhausted: bool,
    sequence: u64,
    batches: Vec<Batch>,
    collect: Phase,
    audit: Phase,
    gradient: Phase,
    evaluation_due: u64,
}

/// Counts from returned collections include failed prefixes, independently of
/// the session's whole-cohort progress. Unknown calls have no invented counts.
/// Forward work is not instrumented; audit and qualification may also infer.
#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ControllerReport {
    schema: &'static str,
    stop_reason: String,
    error: Option<String>,
    decision_budget_reached: bool,
    session_progress: SessionProgress,
    known_collection_work: Work,
    accepted_multi_samples: u64,
    accepted_multi_candidate_rows: u64,
    unknown_collection_calls: u64,
    native_forward_calls: Option<u64>,
    collect: Phase,
    audit: Phase,
    gradient: Phase,
    invocation_io: Phase,
    evaluation_due_notice_count: u64,
    evaluation_interval: u64,
    evaluations_performed: u64,
    checkpoint_file: String,
    checkpoint_sha256: String,
}
impl ControllerReport {
    pub fn decision_budget_reached(&self) -> bool {
        self.decision_budget_reached
    }
    pub fn stopped_with_error(&self) -> bool {
        self.error.is_some() || self.stop_reason == "unknownPendingWork"
    }
}

struct Driver<'a> {
    prepared: &'a PreparedPublicPolicy,
    session: RlTrainingSession,
    store: Store,
}

struct Store {
    state: State,
    io: Phase,
    last_pin: Option<Pin>,
}

/// The new directory is acquired exclusively. Parent directories must exist.
pub fn train(
    spec: RunSpec,
    prepared: &PreparedPublicPolicy,
    directory: &Path,
) -> Result<ControllerReport, String> {
    spec.validate()?;
    let session = RlTrainingSession::start(
        InitializedPublicRlPolicy::from_bc(prepared)?,
        spec.session_config()?,
    )?;
    fs::create_dir(directory).map_err(|e| e.to_string())?;
    let cursor = spec.environment_seed_start as u64;
    let state = State {
        schema: STATE_SCHEMA.into(),
        spec,
        session_json: String::new(),
        session_sha256: String::new(),
        seed_cursor: cursor,
        seed_range_exhausted: false,
        sequence: 0,
        batches: vec![],
        collect: Phase::default(),
        audit: Phase::default(),
        gradient: Phase::default(),
        evaluation_due: 0,
    };
    let mut driver = Driver {
        prepared,
        session,
        store: Store {
            state,
            io: Phase::default(),
            last_pin: None,
        },
    };
    driver.store.refresh_session(&driver.session)?;
    driver.store.save(directory)?;
    driver.run(directory, true, None)
}

/// A pending batch uses only pinned saved records, never new collection. An
/// explicit abandon consumes its reserved families and streams and stops.
pub fn resume(
    spec: RunSpec,
    prepared: &PreparedPublicPolicy,
    bytes: &[u8],
    expected_sha256: &str,
    directory: &Path,
    abandon: Option<&str>,
) -> Result<ControllerReport, String> {
    spec.validate()?;
    if !digest(expected_sha256) || sha(bytes) != expected_sha256 {
        return Err("Controller raw checkpoint SHA mismatch".into());
    }
    let state: State = closed(bytes, MAX_STATE_BYTES)?;
    if state.schema != STATE_SCHEMA || state.spec != spec {
        return Err("Controller run specification changed on resume".into());
    }
    let session = RlTrainingSession::restore(
        InitializedPublicRlPolicy::from_bc(prepared)?,
        state.session_json.as_bytes(),
        &state.session_sha256,
    )?;
    if session.config() != &spec.session_config()? {
        return Err("Controller session configuration differs".into());
    }
    validate_state(&state, &session)?;
    let driver = Driver {
        prepared,
        session,
        store: Store {
            state,
            io: Phase::default(),
            last_pin: Some(Pin {
                bytes: bytes.len(),
                sha256: expected_sha256.into(),
            }),
        },
    };
    driver.run(directory, false, abandon)
}

impl Store {
    fn refresh_session(&mut self, session: &RlTrainingSession) -> Result<(), String> {
        self.state.session_json =
            String::from_utf8(session.checkpoint_bytes()?).map_err(|e| e.to_string())?;
        self.state.session_sha256 = sha(self.state.session_json.as_bytes());
        Ok(())
    }
    fn save(&mut self, directory: &Path) -> Result<(), String> {
        let sequence = self
            .state
            .sequence
            .checked_add(1)
            .filter(|n| *n <= 1000)
            .ok_or("Controller sequence bound")?;
        let previous = self.state.sequence;
        self.state.sequence = sequence;
        let encoded = serde_json::to_vec(&self.state);
        self.state.sequence = previous;
        let bytes = encoded.map_err(|e| e.to_string())?;
        if bytes.len() > MAX_STATE_BYTES {
            return Err("Controller checkpoint byte bound".into());
        }
        let pin = self.publish(directory, &state_name(sequence), &bytes)?;
        self.state.sequence = sequence;
        self.last_pin = Some(pin);
        Ok(())
    }
    fn publish(&mut self, directory: &Path, name: &str, bytes: &[u8]) -> Result<Pin, String> {
        self.io.started_calls += 1;
        let began = Instant::now();
        let result = publish(directory, name, bytes);
        self.io.returned(began);
        result
    }
    fn read_record(
        &mut self,
        directory: &Path,
        batch: usize,
        member: usize,
    ) -> Result<Vec<u8>, String> {
        let pin = self.state.batches[batch].members[member]
            .record
            .as_ref()
            .ok_or("No pinned collection record")?
            .clone();
        self.io.started_calls += 1;
        let began = Instant::now();
        let result = read_bounded(
            &directory.join(record_name(batch, member)),
            MAX_RECORD_BYTES,
        )
        .and_then(|bytes| {
            if bytes.len() == pin.bytes
                && sha(&bytes) == pin.sha256
                && self.state.batches[batch].members[member]
                    .known_work
                    .as_ref()
                    == Some(&work(&bytes)?)
            {
                Ok(bytes)
            } else {
                Err("Saved collection record identity differs".into())
            }
        });
        self.io.returned(began);
        result
    }
    fn read_records(&mut self, directory: &Path, batch: usize) -> Result<[Vec<u8>; 4], String> {
        let mut records = Vec::with_capacity(4);
        for member in 0..4 {
            records.push(self.read_record(directory, batch, member)?);
        }
        records
            .try_into()
            .map_err(|_| "Controller record member count".into())
    }
    fn collect_member(
        &mut self,
        directory: &Path,
        batch: usize,
        member: usize,
        collect: impl FnOnce() -> Result<Vec<u8>, String>,
    ) -> Result<(), String> {
        self.state.batches[batch].members[member].collection_started = true;
        self.state.collect.started_calls += 1;
        if let Err(error) = self.save(directory) {
            self.state.batches[batch].members[member].collection_started = false;
            self.state.collect.started_calls -= 1;
            return Err(error);
        }
        let began = Instant::now();
        let result = collect();
        self.state.collect.returned(began);
        match result {
            Ok(bytes) => {
                self.state.batches[batch].members[member].known_work = Some(work(&bytes)?);
                let pin = self.publish(directory, &record_name(batch, member), &bytes)?;
                self.state.batches[batch].members[member].record = Some(pin);
            }
            Err(error) => {
                self.state.batches[batch].members[member].collection_error = Some(capped(&error))
            }
        }
        self.save(directory)
    }
    fn audit_members<T>(
        &mut self,
        directory: &Path,
        batch: usize,
        records: [Vec<u8>; 4],
        mut audit: impl FnMut(&[u8]) -> Result<T, String>,
    ) -> Result<[Result<T, String>; 4], String> {
        let mut results = Vec::with_capacity(4);
        for (member, bytes) in records.into_iter().enumerate() {
            self.state.audit.started_calls += 1;
            if let Err(error) = self.save(directory) {
                self.state.audit.started_calls -= 1;
                return Err(error);
            }
            let began = Instant::now();
            let result = audit(&bytes);
            self.state.audit.returned(began);
            self.state.batches[batch].members[member].audit_passed = result.is_ok();
            self.state.batches[batch].members[member].audit_error =
                result.as_ref().err().map(|e| capped(e));
            results.push(result);
            self.save(directory)?;
        }
        results
            .try_into()
            .map_err(|_| "Controller audit member count".into())
    }
    fn before_gradient(&mut self, directory: &Path, batch: usize) -> Result<(), String> {
        let previous = self.state.batches[batch]
            .outcome
            .replace("gradientUnknown".into());
        self.state.gradient.started_calls += 1;
        if let Err(error) = self.save(directory) {
            self.state.batches[batch].outcome = previous;
            self.state.gradient.started_calls -= 1;
            return Err(error);
        }
        Ok(())
    }
}

impl Driver<'_> {
    fn run(
        mut self,
        directory: &Path,
        mut fresh: bool,
        abandon: Option<&str>,
    ) -> Result<ControllerReport, String> {
        if let Some(reason) = abandon {
            self.session.abandon_pending(reason)?;
            let batch = self
                .store
                .state
                .batches
                .last_mut()
                .ok_or("No pending controller batch")?;
            batch.outcome = Some("abandoned".into());
            batch.error = Some(capped(reason));
            self.store.refresh_session(&self.session)?;
            self.store.save(directory)?;
            return self.report("explicitAbandon", None);
        }
        loop {
            let progress = self.session.progress()?;
            if !progress.unresolved_batch() {
                if self.store.state.seed_range_exhausted {
                    return self.report(
                        "seedRangeExhausted",
                        Some("Run exhausted its fixed metadata seed range".into()),
                    );
                }
                if self.known_decisions() >= self.store.state.spec.decision_budget {
                    return self.report("decisionBudget", None);
                }
                if progress.completed_updates() >= MAX_UPDATE_COUNT {
                    return self.report("v1Count10Limit", None);
                }
                if progress.reserved_batches() >= self.store.state.spec.max_batches {
                    return self.report("batchLimit", None);
                }
                let configs = match self.store.state.spec.allocate(
                    &mut self.store.state.seed_cursor,
                    progress.next_episode_ordinal(),
                    self.session.consumed_families()?,
                ) {
                    Ok(c) => c,
                    Err(e) => {
                        if self.store.state.seed_cursor == self.store.state.spec.end_seed() {
                            self.store.state.seed_range_exhausted = true;
                            self.store.save(directory)?;
                            return self.report("seedRangeExhausted", Some(e));
                        }
                        return Err(e);
                    }
                };
                if self.session.parent().is_some() {
                    self.session.plan_next(configs.clone())?;
                } else {
                    self.session.plan_first(configs.clone())?;
                }
                self.store.state.batches.push(Batch {
                    configurations: configs.each_ref().map(Config::from_config),
                    members: std::array::from_fn(|_| Member::default()),
                    outcome: None,
                    error: None,
                });
                self.store.refresh_session(&self.session)?;
                self.store.save(directory)?; // The reservation exists before any collection.
                fresh = true;
            }
            let index = self.store.state.batches.len() - 1;
            if fresh {
                for member in 0..4 {
                    self.collect_member(directory, index, member)?;
                }
            } else if self.store.state.batches[index].outcome.as_deref() == Some("gradientUnknown")
                || self.store.state.batches[index]
                    .members
                    .iter()
                    .any(|m| m.record.is_none())
            {
                return self.report("unknownPendingWork", None);
            }
            // Infrastructure failure must not resolve/abandon pending work.
            let result = self.audit_and_apply(directory, index)?;
            match result {
                Ok(changed) => {
                    self.store.state.batches[index].outcome =
                        Some(if changed { "updated" } else { "noChange" }.into());
                    self.notices();
                    self.store.save(directory)?;
                    if !changed {
                        return self.report("noChange", None);
                    }
                }
                Err(error) => {
                    if self.session.progress()?.unresolved_batch() {
                        self.session.abandon_pending(&error)?;
                    }
                    self.store.state.batches[index].outcome = Some("failed".into());
                    self.store.state.batches[index].error = Some(capped(&error));
                    self.store.refresh_session(&self.session)?;
                    self.store.save(directory)?;
                    return self.report("batchFailed", Some(capped(&error)));
                }
            }
            fresh = true;
        }
    }
    fn collect_member(
        &mut self,
        directory: &Path,
        batch: usize,
        member: usize,
    ) -> Result<(), String> {
        let config = self.store.state.batches[batch].configurations[member].checked_config()?;
        match self.session.parent() {
            None => {
                let handle = self.prepared.handle()?;
                self.store.collect_member(directory, batch, member, || {
                    collect_native(&config, &handle).and_then(|g| encode_record(&g))
                })
            }
            Some(RlUpdateParent::Count1(p)) => {
                let handle = UpdatedPublicRlHandle::new(p)?;
                self.store.collect_member(directory, batch, member, || {
                    collect_rl_native(&config, &handle).and_then(|g| encode_rl_record(&g))
                })
            }
            Some(RlUpdateParent::Repeated(p)) => {
                let handle = RepeatedPublicRlHandle::new(p)?;
                self.store.collect_member(directory, batch, member, || {
                    collect_repeated_rl_native(&config, &handle)
                        .and_then(|g| encode_repeated_rl_record(&g))
                })
            }
        }
    }
    // Outer errors are storage/handle failures. Only a finished cohort or
    // attempted update can produce an inner rejection that resolves a batch.
    fn audit_and_apply(
        &mut self,
        directory: &Path,
        batch: usize,
    ) -> Result<Result<bool, String>, String> {
        if self.store.state.batches[batch]
            .members
            .iter()
            .any(|m| m.record.is_none())
        {
            return Ok(Err("Cohort contains a failed collection".into()));
        }
        // Verify every pin before any audit, metadata save or session mutation.
        let records = self.store.read_records(directory, batch)?;
        let result = if self.session.parent().is_none() {
            let plan = self.session.pending_first_plan()?;
            let handle = self.prepared.handle()?;
            let results = self.store.audit_members(directory, batch, records, |b| {
                ValidatedPolicyEpisode::from_audited(audit_record_bytes(b, &handle)?)
            })?;
            let cohort = match plan.finish(results) {
                Ok(cohort) => cohort,
                Err(error) => return Ok(Err(error)),
            };
            self.store.before_gradient(directory, batch)?;
            let began = Instant::now();
            let result = self.session.apply_first(&cohort);
            self.store.state.gradient.returned(began);
            result
        } else {
            let plan = self.session.pending_next_plan()?;
            let results = match self.session.parent().ok_or("Missing RL parent")? {
                RlUpdateParent::Count1(p) => {
                    let handle = UpdatedPublicRlHandle::new(p)?;
                    self.store.audit_members(directory, batch, records, |b| {
                        ValidatedRlPolicyEpisode::from_audited(audit_rl_record_bytes(b, &handle)?)
                    })?
                }
                RlUpdateParent::Repeated(p) => {
                    let handle = RepeatedPublicRlHandle::new(p)?;
                    self.store.audit_members(directory, batch, records, |b| {
                        ValidatedRlPolicyEpisode::from_repeated_audited(
                            audit_repeated_rl_record_bytes(b, &handle)?,
                        )
                    })?
                }
            };
            let cohort = match plan.finish(results) {
                Ok(cohort) => cohort,
                Err(error) => return Ok(Err(error)),
            };
            self.store.before_gradient(directory, batch)?;
            let began = Instant::now();
            let result = self.session.apply_next(&cohort);
            self.store.state.gradient.returned(began);
            result
        };
        self.store.refresh_session(&self.session)?;
        Ok(result)
    }
    fn known_decisions(&self) -> u64 {
        self.store
            .state
            .batches
            .iter()
            .flat_map(|b| &b.members)
            .filter_map(|m| m.known_work.as_ref())
            .map(|w| w.accepted_samples)
            .sum()
    }
    fn notices(&mut self) {
        self.store.state.evaluation_due =
            self.known_decisions() / self.store.state.spec.evaluation_interval;
    }
    fn report(self, reason: &str, error: Option<String>) -> Result<ControllerReport, String> {
        let decisions = self.known_decisions();
        let mut counts = [0u64; 7];
        let mut unknown = 0;
        for m in self.store.state.batches.iter().flat_map(|b| &b.members) {
            if let Some(w) = &m.known_work {
                for (sum, value) in counts.iter_mut().zip([
                    w.observed_callbacks,
                    w.sampler_attempts,
                    w.accepted_samples,
                    w.applied_choices,
                    w.candidate_rows_reserved,
                    w.accepted_candidate_rows,
                    w.singleton_samples,
                ]) {
                    *sum += value;
                }
            } else if m.collection_started {
                unknown += 1;
            }
        }
        Ok(ControllerReport {
            schema: "tzolkin-public-rl-controller-report-v1",
            stop_reason: reason.into(),
            decision_budget_reached: reason == "decisionBudget"
                && error.is_none()
                && decisions >= self.store.state.spec.decision_budget,
            error,
            session_progress: self.session.progress()?,
            known_collection_work: Work {
                observed_callbacks: counts[0],
                sampler_attempts: counts[1],
                accepted_samples: counts[2],
                applied_choices: counts[3],
                candidate_rows_reserved: counts[4],
                accepted_candidate_rows: counts[5],
                singleton_samples: counts[6],
            },
            accepted_multi_samples: counts[2] - counts[6],
            accepted_multi_candidate_rows: counts[5] - counts[6],
            unknown_collection_calls: unknown,
            native_forward_calls: None,
            collect: self.store.state.collect,
            audit: self.store.state.audit,
            gradient: self.store.state.gradient,
            invocation_io: self.store.io,
            evaluation_due_notice_count: self.store.state.evaluation_due,
            evaluation_interval: self.store.state.spec.evaluation_interval,
            evaluations_performed: 0,
            checkpoint_file: state_name(self.store.state.sequence),
            checkpoint_sha256: self
                .store
                .last_pin
                .ok_or("Controller checkpoint unavailable")?
                .sha256,
        })
    }
}

fn validate_state(state: &State, session: &RlTrainingSession) -> Result<(), String> {
    let progress = session.progress()?;
    let saved: serde_json::Value =
        serde_json::from_str(&state.session_json).map_err(|e| e.to_string())?;
    let reservations = saved["reservations"]
        .as_array()
        .ok_or("Controller missing reservations")?;
    validate_metadata(state, reservations, progress.reserved_batches())
}
fn validate_metadata(
    state: &State,
    reservations: &[serde_json::Value],
    reserved: usize,
) -> Result<(), String> {
    if state.sequence == 0
        || state.sequence > 1000
        || state.batches.len() != reserved
        || reservations.len() != reserved
        || state.batches.len() > state.spec.max_batches
        || state.seed_cursor < state.spec.environment_seed_start as u64
        || state.seed_cursor > state.spec.end_seed()
        || !state.collect.valid()
        || !state.audit.valid()
        || !state.gradient.valid()
        || [&state.collect, &state.audit, &state.gradient]
            .iter()
            .any(|p| p.started_calls > state.sequence)
    {
        return Err("Invalid controller cursor/progress/phase accounting".into());
    }
    let mut started = 0;
    let mut returned = 0;
    let mut passed = 0;
    let mut decisions = 0;
    for (i, batch) in state.batches.iter().enumerate() {
        if serde_json::to_value(&batch.configurations).map_err(|e| e.to_string())?
            != reservations[i]["configurations"]
        {
            return Err("Controller planned configurations differ from session".into());
        }
        let status = reservations[i]["status"]
            .as_str()
            .ok_or("Missing reservation outcome")?;
        let outcome_matches = match status {
            "reserved" => {
                batch.outcome.is_none() || batch.outcome.as_deref() == Some("gradientUnknown")
            }
            "updated" => batch.outcome.as_deref() == Some("updated"),
            "noChange" => batch.outcome.as_deref() == Some("noChange"),
            "failed" => matches!(batch.outcome.as_deref(), Some("failed" | "abandoned")),
            _ => false,
        };
        if !outcome_matches
            || batch
                .error
                .as_ref()
                .is_some_and(|e| e.chars().count() > 1024)
            || (matches!(status, "updated" | "noChange")
                && batch.members.iter().any(|m| !m.audit_passed))
        {
            return Err("Controller/session batch outcomes differ".into());
        }
        for (member, (m, c)) in batch.members.iter().zip(&batch.configurations).enumerate() {
            let seed = c.environment_seed as u64;
            if seed < state.spec.environment_seed_start as u64
                || seed >= state.spec.end_seed()
                || seed >= state.seed_cursor
                || c.players != if member < 2 { 3 } else { 4 }
                || c.checked_config()?.limits() != &state.spec.limits()?
                || state
                    .spec
                    .excluded_families
                    .binary_search(&seed_family_id(c.environment_seed))
                    .is_ok()
            {
                return Err(
                    "Controller reserved configuration differs from fixed range/limits/exclusions"
                        .into(),
                );
            }
            if m.record
                .as_ref()
                .is_some_and(|p| !digest(&p.sha256) || p.bytes == 0 || p.bytes > MAX_RECORD_BYTES)
                || m.record.is_some() != m.known_work.is_some()
                || (m.record.is_some() && !m.collection_started)
                || (m.audit_passed && (m.record.is_none() || m.audit_error.is_some()))
                || (m.collection_error.is_some() && (!m.collection_started || m.record.is_some()))
                || [&m.collection_error, &m.audit_error]
                    .into_iter()
                    .flatten()
                    .any(|s| s.chars().count() > 1024)
            {
                return Err("Invalid controller member outcome".into());
            }
            if let Some(w) = &m.known_work {
                if w.accepted_samples > w.sampler_attempts
                    || w.applied_choices > w.accepted_samples
                    || w.sampler_attempts > w.observed_callbacks
                    || w.observed_callbacks > c.max_callbacks as u64
                    || w.candidate_rows_reserved > c.max_candidate_rows as u64
                    || w.accepted_candidate_rows > w.candidate_rows_reserved
                    || w.singleton_samples > w.accepted_samples
                    || w.accepted_candidate_rows < w.accepted_samples
                {
                    return Err("Invalid controller logical collection counts".into());
                }
                decisions += w.accepted_samples;
            }
            started += u64::from(m.collection_started);
            returned += u64::from(m.record.is_some() || m.collection_error.is_some());
            passed += u64::from(m.audit_passed);
        }
        if !reservations[i]["counts"].is_null() {
            let mut counts = [0u64; 3];
            for m in &batch.members {
                let w = m
                    .known_work
                    .as_ref()
                    .ok_or("Complete cohort lacks collection counters")?;
                counts[0] += w.accepted_samples;
                counts[1] += w.accepted_candidate_rows;
                counts[2] += w.singleton_samples;
            }
            if serde_json::to_value(counts).map_err(|e| e.to_string())? != reservations[i]["counts"]
            {
                return Err("Controller complete-cohort counters differ from session".into());
            }
        }
    }
    if started != state.collect.started_calls
        || returned != state.collect.returned_calls
        || passed > state.audit.returned_calls
        || state.evaluation_due > decisions / state.spec.evaluation_interval
        || (state.seed_range_exhausted && state.seed_cursor != state.spec.end_seed())
    {
        return Err("Controller phase/member accounting differs".into());
    }
    Ok(())
}
fn work(bytes: &[u8]) -> Result<Work, String> {
    let v = unique_json(bytes, MAX_RECORD_BYTES)?;
    let get = |name| {
        v["counts"][name]
            .as_u64()
            .ok_or("Missing collected counter")
    };
    let mut rows = 0;
    let mut singleton = 0;
    for c in v["callbacks"]
        .as_array()
        .ok_or("Missing collected callbacks")?
    {
        if !c["sample"].is_null() {
            let n = c["sample"]["trace"]["legalCount"]
                .as_u64()
                .ok_or("Missing collected legal count")?;
            rows += n;
            singleton += u64::from(n == 1);
        }
    }
    Ok(Work {
        observed_callbacks: get("observedCallbacks")?,
        sampler_attempts: get("samplerAttempts")?,
        accepted_samples: get("acceptedSamples")?,
        applied_choices: get("applySuccesses")?,
        candidate_rows_reserved: get("candidateRowsReserved")?,
        accepted_candidate_rows: rows,
        singleton_samples: singleton,
    })
}
fn closed<T: serde::de::DeserializeOwned + Serialize>(
    bytes: &[u8],
    maximum: usize,
) -> Result<T, String> {
    let value = unique_json(bytes, maximum)?;
    let typed: T = serde_json::from_value(value.clone()).map_err(|e| e.to_string())?;
    if value != serde_json::to_value(&typed).map_err(|e| e.to_string())? {
        return Err("Unknown or omitted controller fields".into());
    }
    Ok(typed)
}
fn digest(s: &str) -> bool {
    s.len() == 64
        && s.bytes()
            .all(|c| c.is_ascii_digit() || (b'a'..=b'f').contains(&c))
}
fn sha(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}
fn capped(s: &str) -> String {
    s.chars().take(1024).collect()
}
fn state_name(sequence: u64) -> String {
    format!("checkpoint-{sequence:04}.json")
}
fn record_name(batch: usize, member: usize) -> String {
    format!("batch-{batch:04}-member-{member}.json")
}

/// Ordinary trusted-directory persistence, not a hostile filesystem sandbox.
/// Existing destinations/temporary files are never reused or removed on failure.
fn publish(directory: &Path, name: &str, bytes: &[u8]) -> Result<Pin, String> {
    static NEXT_TEMP: AtomicU64 = AtomicU64::new(0);
    let path = directory.join(name);
    require_absent(&path)?;
    let mut acquired = None;
    for _ in 0..32 {
        let temp = directory.join(format!(
            "{name}.{}-{}.tmp",
            std::process::id(),
            NEXT_TEMP.fetch_add(1, Ordering::Relaxed)
        ));
        match OpenOptions::new().write(true).create_new(true).open(&temp) {
            Ok(file) => {
                acquired = Some((temp, file));
                break;
            }
            Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => continue,
            Err(e) => return Err(e.to_string()),
        }
    }
    let (temp, mut file) = acquired.ok_or("Controller temporary file collision bound")?;
    file.write_all(bytes)
        .and_then(|_| file.sync_all())
        .map_err(|e| e.to_string())?;
    drop(file);
    require_absent(&path)?;
    fs::rename(&temp, &path).map_err(|e| e.to_string())?;
    Ok(Pin {
        bytes: bytes.len(),
        sha256: sha(bytes),
    })
}
fn require_absent(path: &Path) -> Result<(), String> {
    match fs::symlink_metadata(path) {
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Ok(_) => Err("Controller destination already exists".into()),
        Err(e) => Err(e.to_string()),
    }
}
pub fn read_bounded(path: &Path, maximum: usize) -> Result<Vec<u8>, String> {
    if maximum == 0 || maximum > MAX_RECORD_BYTES {
        return Err("Invalid controller read bound".into());
    }
    let metadata = fs::symlink_metadata(path).map_err(|e| e.to_string())?;
    if !metadata.is_file() || metadata.file_type().is_symlink() || metadata.len() > maximum as u64 {
        return Err("Controller input must be bounded regular file".into());
    }
    let file = fs::File::open(path).map_err(|e| e.to_string())?;
    let mut bytes = Vec::new();
    file.take(maximum as u64 + 1)
        .read_to_end(&mut bytes)
        .map_err(|e| e.to_string())?;
    if bytes.len() > maximum {
        return Err("Controller input byte bound".into());
    }
    Ok(bytes)
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
    fn spec() -> RunSpec {
        RunSpec {
            schema: RUN_SPEC_SCHEMA.into(),
            sampling_seed: 17,
            replicate_ordinal: 0,
            first_episode_ordinal: 0,
            environment_seed_start: 0,
            max_seed_candidates: 1000,
            decision_budget: 100,
            evaluation_interval: 50,
            max_batches: 2,
            learning_rate: 1e-4,
            max_abs_actual_delta: 0.01,
            max_gradient_rows: 1000,
            max_callbacks: 10,
            max_candidate_rows: 1000,
            max_source_bytes: 65536,
            max_trace_bytes: 65536,
            excluded_families: vec![],
        }
    }
    fn state(spec: RunSpec, configs: [NativeStochasticConfig; 4], cursor: u64) -> State {
        State {
            schema: STATE_SCHEMA.into(),
            spec,
            session_json: String::new(),
            session_sha256: String::new(),
            seed_cursor: cursor,
            seed_range_exhausted: false,
            sequence: 10,
            batches: vec![Batch {
                configurations: configs.each_ref().map(Config::from_config),
                members: std::array::from_fn(|_| Member::default()),
                outcome: None,
                error: None,
            }],
            collect: Phase::default(),
            audit: Phase::default(),
            gradient: Phase::default(),
            evaluation_due: 0,
        }
    }
    #[test]
    fn fixed_plan_and_resume_metadata_reject_inconsistency() {
        let spec = spec();
        let wire = serde_json::to_vec(&spec).unwrap();
        assert!(RunSpec::from_json(&wire).is_ok());
        let mut value = serde_json::to_value(&spec).unwrap();
        value["unknown"] = true.into();
        assert!(RunSpec::from_json(&serde_json::to_vec(&value).unwrap()).is_err());
        let mut cursor = 0;
        let excluded = BTreeSet::from([seed_family_id(0), seed_family_id(3)]);
        let configs = spec.allocate(&mut cursor, 0, excluded.clone()).unwrap();
        assert_eq!(
            configs.each_ref().map(NativeStochasticConfig::players),
            [3, 3, 4, 4]
        );
        assert!(configs.iter().all(|c| {
            split_for_family(&seed_family_id(c.environment_seed())).unwrap() == DatasetSplit::Train
                && !excluded.contains(&seed_family_id(c.environment_seed()))
        }));
        let mut replay_cursor = 0;
        assert_eq!(
            configs,
            spec.allocate(&mut replay_cursor, 0, excluded).unwrap()
        );
        assert_eq!(cursor, replay_cursor);
        let mut state = state(spec, configs, cursor);
        // Two known returned prefixes and an interrupted third call retain
        // work separately from any completed-cohort session progress.
        for member in &mut state.batches[0].members[..2] {
            member.collection_started = true;
            member.record = Some(Pin {
                bytes: 3,
                sha256: sha(b"abc"),
            });
            member.known_work = Some(Work {
                observed_callbacks: 2,
                sampler_attempts: 2,
                accepted_samples: 2,
                applied_choices: 1,
                candidate_rows_reserved: 4,
                accepted_candidate_rows: 4,
                singleton_samples: 1,
            });
        }
        state.batches[0].members[2].collection_started = true;
        state.collect.started_calls = 3;
        state.collect.returned_calls = 2;
        let reservations = vec![
            serde_json::json!({"configurations": state.batches[0].configurations, "status": "reserved"}),
        ];
        assert!(validate_metadata(&state, &reservations, 1).is_ok());
        let cursor = state.seed_cursor;
        state.seed_cursor = state.batches[0].configurations[3].environment_seed as u64;
        assert!(validate_metadata(&state, &reservations, 1).is_err());
        state.seed_cursor = cursor;
        state.batches[0].outcome = Some("updated".into());
        assert!(validate_metadata(&state, &reservations, 1).is_err());
        state.batches[0].outcome = None;
        state.collect.returned_calls = 3;
        assert!(validate_metadata(&state, &reservations, 1).is_err());
        state.collect.returned_calls = 2;
        let original = state.batches[0].configurations[0].clone();
        state.batches[0].configurations[0].max_callbacks -= 1;
        let mut matching = reservations.clone();
        matching[0]["configurations"] =
            serde_json::to_value(&state.batches[0].configurations).unwrap();
        assert!(validate_metadata(&state, &matching, 1).is_err());
        state.batches[0].configurations[0] = original;
        state.spec.excluded_families.push(seed_family_id(
            state.batches[0].configurations[0].environment_seed,
        ));
        assert!(validate_metadata(&state, &reservations, 1).is_err());
    }
    #[test]
    fn new_only_io_preserves_collisions_and_rejects_corruption() {
        let root = temp_root::create("rl-controller").unwrap();
        let pin = publish(&root, "record.json", b"abc").unwrap();
        assert_eq!(pin.sha256, sha(b"abc"));
        assert!(publish(&root, "record.json", b"different").is_err());
        assert_eq!(read_bounded(&root.join("record.json"), 3).unwrap(), b"abc");
        assert!(read_bounded(&root.join("record.json"), 2).is_err());
        fs::write(root.join("other.json.tmp"), b"preserved").unwrap();
        assert!(publish(&root, "other.json", b"abc").is_ok());
        assert_eq!(fs::read(root.join("other.json.tmp")).unwrap(), b"preserved");
        assert_eq!(fs::read(root.join("other.json")).unwrap(), b"abc");
        assert!(
            closed::<Phase>(
                br#"{"startedCalls":1,"startedCalls":1,"returnedCalls":0,"returnedSeconds":0}"#,
                1024
            )
            .is_err()
        );
        assert!(closed::<Member>(br#"{"collectionStarted":false}"#, 1024).is_err());
        let spec = spec();
        let mut cursor = 0;
        let configs = spec.allocate(&mut cursor, 0, BTreeSet::new()).unwrap();
        let mut store = Store {
            state: state(spec, configs, cursor),
            io: Phase::default(),
            last_pin: None,
        };
        store.state.sequence = 0;
        // Synthetic counters exercise storage only, not native audit or a policy.
        let record = serde_json::to_vec(&serde_json::json!({
            "counts": {"observedCallbacks":0, "samplerAttempts":0, "acceptedSamples":0,
                "applySuccesses":0, "candidateRowsReserved":0}, "callbacks":[]
        }))
        .unwrap();
        for member in 0..4 {
            store.state.batches[0].members[member].record =
                Some(publish(&root, &record_name(0, member), &record).unwrap());
            store.state.batches[0].members[member].known_work = Some(work(&record).unwrap());
        }
        store.save(&root).unwrap();
        let unchanged = serde_json::to_vec(&store.state).unwrap();
        fs::remove_file(root.join(record_name(0, 3))).unwrap();
        assert!(store.read_records(&root, 0).is_err());
        assert_eq!(serde_json::to_vec(&store.state).unwrap(), unchanged);
        fs::write(root.join(record_name(0, 3)), b"corrupt").unwrap();
        assert!(store.read_records(&root, 0).is_err());
        assert_eq!(serde_json::to_vec(&store.state).unwrap(), unchanged);
        fs::write(root.join(record_name(0, 3)), &record).unwrap();
        let records = store.read_records(&root, 0).unwrap();
        fs::write(root.join(state_name(2)), b"existing checkpoint").unwrap();
        let mut audit_called = false;
        assert!(
            store
                .audit_members(&root, 0, records, |_| {
                    audit_called = true;
                    Ok(())
                })
                .is_err()
        );
        assert!(!audit_called);
        assert!(store.before_gradient(&root, 0).is_err());
        assert_eq!(serde_json::to_vec(&store.state).unwrap(), unchanged);
        assert_eq!(
            fs::read(root.join(state_name(2))).unwrap(),
            b"existing checkpoint"
        );
        assert!(!root.join(state_name(3)).exists());
        store.state.sequence = 1000;
        assert!(store.save(&root).is_err());
        assert_eq!(store.state.sequence, 1000);
        fs::remove_dir_all(root).unwrap();
    }
}
