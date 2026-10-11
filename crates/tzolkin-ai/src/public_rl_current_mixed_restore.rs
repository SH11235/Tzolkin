//! Metadata-only journal restoration. Applied-link checks do not seal episodes
//! or confer fresh audit/learning capability on serialized records.
use super::*;
use crate::public_policy_episode::{applied_mixed_plan, canonical_record_checksum};
use serde_json::{Value, json};

struct MemberSummary {
    checksum: String,
    counts: [usize; 3],
    observed: usize,
}
impl MemberSummary {
    fn read(
        record: &crate::public_stochastic_record::Record<source::CurrentMixedSource>,
        plan: &PlannedCurrentMixedCohort,
        member: usize,
    ) -> Result<Self, String> {
        if record.terminal.is_none() || record.failure.is_some() {
            return Err("Current mixed saved member is incomplete".into());
        }
        let parent = &record.header.base_policy.parent;
        let learner = record.header.base_policy.learner_seat;
        let (links, _, _) = applied_mixed_plan(
            record,
            &plan.configurations()[member],
            parent.sampling_version(),
            || parent.binding_key(),
            learner,
        )?;
        let mut counts = [0usize; 3];
        for c in record.callbacks.iter().filter(|c| c.actor == learner) {
            let n = c
                .observation
                .as_ref()
                .ok_or("Saved learner observation missing")?
                .legal_actions
                .len();
            for (sum, k) in counts.iter_mut().zip([1, n, usize::from(n == 1)]) {
                *sum = sum.checked_add(k).ok_or("Saved learner count overflow")?;
            }
        }
        if links.len() != counts[0] {
            return Err("Saved learner link coverage differs".into());
        }
        Ok(Self {
            checksum: canonical_record_checksum(
                record,
                b"tzolkin-applied-current-mixed-episode-v2\0",
            )?,
            counts,
            observed: record.counts.observed_callbacks,
        })
    }
}
fn cohort_receipt(
    plan: &PlannedCurrentMixedCohort,
    summaries: &[Option<MemberSummary>; 4],
) -> Result<Value, String> {
    let mut counts = [0usize; 3];
    let mut observed = 0usize;
    let mut episodes = Vec::new();
    for s in summaries {
        let s = s
            .as_ref()
            .ok_or("Saved current mixed whole-four coverage missing")?;
        for (sum, k) in counts.iter_mut().zip(s.counts) {
            *sum = sum.checked_add(k).ok_or("Saved cohort target overflow")?;
        }
        observed = observed
            .checked_add(s.observed)
            .ok_or("Saved cohort callback overflow")?;
        episodes.push(s.checksum.clone());
    }
    if counts[0] > 4 * crate::public_stochastic::MAX_SAMPLES
        || counts[1] > 4 * crate::public_stochastic::MAX_RESERVED_CANDIDATE_ROWS
    {
        return Err("Saved whole-four target cap".into());
    }
    let checksum = hash(
        b"tzolkin-completed-current-mixed-cohort-v2\0",
        &(plan.checksum(), &episodes, counts, observed),
    )?;
    Ok(
        json!({"schema":"tzolkin-validated-current-mixed-cohort-v2","cohortContract":source::COHORT_CONTRACT,
        "planChecksum":plan.checksum(),"checksum":checksum,"episodeChecksums":episodes,"learnerCounts":counts,
        "observedCallbacks":observed,"actorNormalization":"learnerOwnSumTimes1/4","criticTargets":"learnerCallbacksOnlyIncludingSingletons",
        "trainingDataAdmission":true,"selfPlayProducerAdmission":false,"mixedOptimizerAvailable":false}),
    )
}
impl<'a> CurrentMixedTrainingSession<'a> {
    /// Requires an externally pinned exact latest head. No collection, replay,
    /// forward, optimizer or wire-to-sealed-cohort conversion occurs here.
    /// Extra orphan/temp outputs refuse restore and require external handling.
    pub fn restore(
        root: AdmittedCurrentMixedRoot<'a>,
        spec: CurrentMixedRunSpec,
        pool: &CurrentMixedCollectionSpec,
        head_raw: &[u8],
        expected_sha256: &str,
        directory: &Path,
    ) -> Result<Self, String> {
        if !checkpoint_digest(expected_sha256)
            || head_raw.is_empty()
            || head_raw.len() > HEAD_CAP
            || sha(head_raw) != expected_sha256
        {
            return Err("Current mixed externally pinned latest head differs".into());
        }
        let tip: Entry = closed(head_raw, HEAD_CAP)?;
        let mut s = Self::initial(root, spec, pool, local(directory, false)?)?;
        if tip.sequence == 0 || tip.sequence > s.spec.max_journal_entries {
            return Err("Current mixed bounded restore sequence".into());
        }
        let mut previous = String::new();
        let mut plan: Option<PlannedCurrentMixedCohort> = None;
        let mut summaries: [Option<MemberSummary>; 4] = std::array::from_fn(|_| None);
        let mut cohort: Option<Value> = None;
        for sequence in 1..=tip.sequence {
            let name = format!("head-{sequence:016x}.json");
            let raw = read_bounded(&s.directory.join(&name), HEAD_CAP)?;
            if sequence == tip.sequence && raw != head_raw {
                return Err("Current mixed saved latest head differs".into());
            }
            let entry: Entry = closed(&raw, HEAD_CAP)?;
            s.require_entry(&entry, sequence, &previous)?;
            s.progress.check(&entry.event, &s.spec)?;
            let mut promotion = None;
            let mut owner_pins = None;
            match &entry.event {
                Event::Start {
                    run_spec, parent, ..
                } => {
                    parent.require_current(&s.handle()?)?;
                    let raw = read_pin(&s.directory, run_spec, SPEC_CAP)?;
                    let stored = CurrentMixedRunSpec::from_json(&raw)?;
                    if stored.checksum()? != s.spec.checksum()? {
                        return Err("Current mixed original run specification differs".into());
                    }
                    s.account(run_spec, true)?;
                    s.spec_pin = Some(run_spec.clone());
                }
                Event::ReserveWhole4 { reservation: r } => {
                    r.parent.require_current(&s.handle()?)?;
                    let raw = read_pin(&s.batch(r.attempt), &r.collection_spec, SPEC_CAP)?;
                    let collection = CurrentMixedCollectionSpec::from_json(&raw)?;
                    let expected = s.plan(&collection)?;
                    let saved =
                        unique_json(&read_pin(&s.batch(r.attempt), &r.plan, HEAD_CAP)?, HEAD_CAP)?;
                    if saved != expected.receipt()
                        || r.plan_checksum != expected.checksum()
                        || r.families != *expected.rollout_families()
                        || r.configurations
                            != expected
                                .configurations()
                                .each_ref()
                                .map(Config::from_config)
                        || r.work != s.declared_work(&collection)?
                    {
                        return Err("Current mixed original reserved plan/work differs".into());
                    }
                    s.add_reference(r.attempt, &r.collection_spec)?;
                    s.add_reference(r.attempt, &r.plan)?;
                    plan = Some(expected);
                    summaries = std::array::from_fn(|_| None);
                    cohort = None;
                }
                Event::CollectReturn {
                    member,
                    record: Some(pin),
                    complete,
                    error,
                } => {
                    let r = s
                        .progress
                        .active
                        .as_ref()
                        .ok_or("Saved current mixed active reservation missing")?;
                    let raw = read_record(&s.batch(r.attempt), pin, *member, &s.spec)?;
                    let p = plan.as_ref().ok_or("Saved current mixed plan missing")?;
                    let record = source::check_record(&raw, &p.headers[*member])?;
                    crate::public_stochastic_native::require_payload_reservations(
                        &record,
                        &p.configurations()[*member],
                    )?;
                    if *complete != (record.terminal.is_some() && record.failure.is_none()) {
                        return Err("Saved collection return differs from record outcome".into());
                    }
                    if *complete {
                        summaries[*member] = Some(MemberSummary::read(&record, p, *member)?);
                    } else if error.as_deref()
                        != Some(
                            bounded_error(
                                record
                                    .failure
                                    .as_ref()
                                    .map(|f| f.message.as_str())
                                    .unwrap_or("Incomplete current mixed native collection"),
                            )
                            .as_str(),
                        )
                    {
                        return Err("Saved failed collection error differs".into());
                    }
                    let attempt = r.attempt;
                    s.raw_record_bytes = s
                        .raw_record_bytes
                        .checked_add(pin.raw_bytes() as u64)
                        .ok_or("Current mixed restored logical record overflow")?;
                    s.stored_record_bytes = s
                        .stored_record_bytes
                        .checked_add(pin.file().bytes as u64)
                        .ok_or("Current mixed restored stored record overflow")?;
                    s.add_reference(attempt, pin.file())?;
                }
                Event::AuditReturn {
                    member,
                    passed: true,
                    episode_checksum,
                    ..
                } if summaries[*member].as_ref().map(|v| &v.checksum)
                    != episode_checksum.as_ref() =>
                {
                    return Err("Saved audit canonical member checksum differs".into());
                }
                Event::GradientStart {
                    cohort: pin,
                    cohort_checksum,
                    episode_checksums,
                    counts,
                } => {
                    let expected = cohort_receipt(
                        plan.as_ref().ok_or("Saved current mixed plan missing")?,
                        &summaries,
                    )?;
                    let r = s
                        .progress
                        .active
                        .as_ref()
                        .ok_or("Saved current mixed reservation missing")?;
                    let actual =
                        unique_json(&read_pin(&s.batch(r.attempt), pin, HEAD_CAP)?, HEAD_CAP)?;
                    if actual != expected
                        || expected["checksum"] != *cohort_checksum
                        || expected["episodeChecksums"] != json!(episode_checksums)
                        || expected["learnerCounts"] != json!(counts)
                    {
                        return Err("Saved whole-four receipt/counts differ".into());
                    }
                    let attempt = r.attempt;
                    s.add_reference(attempt, pin)?;
                    cohort = Some(expected);
                }
                Event::Resolve {
                    resolution:
                        Resolution::Updated {
                            report,
                            artifact,
                            report_checksum,
                            artifact_checksum,
                        },
                } => {
                    let attempt = s
                        .progress
                        .active
                        .as_ref()
                        .ok_or("Saved current mixed reservation missing")?
                        .attempt;
                    let saved_report = s.saved_report(
                        attempt,
                        report,
                        report_checksum,
                        plan.as_ref(),
                        cohort.as_ref(),
                    )?;
                    let raw = read_pin(&s.batch(attempt), artifact, ARTIFACT_CAP)?;
                    let saved: mixed::ContinuedMixedArtifact = closed(&raw, ARTIFACT_CAP)?;
                    if saved.checksum() != artifact_checksum
                        || bytes(saved.report(), HEAD_CAP)? != bytes(&saved_report, HEAD_CAP)?
                    {
                        return Err("Saved current mixed joint artifact/report differs".into());
                    }
                    promotion = Some(
                        mixed::ContinuedMixedProposal::restored(saved, &s.handle()?)?
                            .stage(&s.handle()?)?,
                    );
                    s.add_reference(attempt, report)?;
                    s.add_reference(attempt, artifact)?;
                    owner_pins = Some((s.batch(attempt), report.clone(), artifact.clone()));
                }
                Event::Resolve {
                    resolution:
                        Resolution::NoChange {
                            report,
                            report_checksum,
                        },
                } => {
                    let attempt = s
                        .progress
                        .active
                        .as_ref()
                        .ok_or("Saved current mixed reservation missing")?
                        .attempt;
                    let saved = s.saved_report(
                        attempt,
                        report,
                        report_checksum,
                        plan.as_ref(),
                        cohort.as_ref(),
                    )?;
                    mixed::ContinuedMixedProposal::restored_no_change(saved, &s.handle()?)?;
                    s.add_reference(attempt, report)?;
                }
                _ => {}
            }
            s.progress.apply(&entry.event, &s.spec)?;
            if let Some(promotion) = promotion {
                // Entire candidate/union/parent validation preceded Resolve.
                mixed::admit_resolved(DurableMixedResolve { promotion }, &mut s.continued);
                s.owner_pins = owner_pins;
            }
            let pin = file_pin(&name, &raw);
            s.account(&pin, true)?;
            previous = pin.sha256.clone();
            s.head = Some(pin);
            s.sequence = sequence;
        }
        s.require_layout()?;
        s.require_live()?;
        s.handle()?;
        s.blocked = s.progress.phase != Phase::Ready;
        Ok(s)
    }
    fn saved_report(
        &self,
        attempt: u64,
        pin: &FilePin,
        checksum: &str,
        plan: Option<&PlannedCurrentMixedCohort>,
        cohort: Option<&Value>,
    ) -> Result<mixed::ContinuedMixedReport, String> {
        let raw = read_pin(&self.batch(attempt), pin, HEAD_CAP)?;
        let report: mixed::ContinuedMixedReport = closed(&raw, HEAD_CAP)?;
        if report.checksum() != checksum {
            return Err("Saved current mixed report checksum differs".into());
        }
        self.require_reserved_report(&report)?;
        report.require_saved_cohort(
            &plan.ok_or("Saved current mixed plan missing")?.receipt(),
            cohort.ok_or("Saved current mixed cohort missing")?,
            &self.spec.actor()?,
            &self.spec.critic()?,
        )?;
        Ok(report)
    }
    fn account(&mut self, pin: &FilePin, metadata: bool) -> Result<(), String> {
        self.output_bytes = self
            .output_bytes
            .checked_add(pin.bytes as u64)
            .ok_or("Current mixed restore output overflow")?;
        self.metadata_bytes = self
            .metadata_bytes
            .checked_add(if metadata { pin.bytes as u64 } else { 0 })
            .ok_or("Current mixed restore metadata overflow")?;
        self.spec.check_storage(
            self.raw_record_bytes,
            self.stored_record_bytes,
            self.metadata_bytes,
        )?;
        if self.output_bytes
            != self
                .stored_record_bytes
                .checked_add(self.metadata_bytes)
                .ok_or("Current mixed restored retained byte overflow")?
        {
            return Err("Current mixed restored retained byte counters differ".into());
        }
        Ok(())
    }
    fn add_reference(&mut self, attempt: u64, pin: &FilePin) -> Result<(), String> {
        let refs = self.references.entry(attempt).or_default();
        if refs.len() >= 9 || refs.iter().any(|v| v.name == pin.name) {
            return Err("Current mixed repeated/unbounded saved reference".into());
        }
        refs.push(pin.clone());
        self.account(pin, !pin.name.starts_with("record-"))
    }
    fn require_entry(&self, entry: &Entry, sequence: u64, previous: &str) -> Result<(), String> {
        require_archive_entry(&self.archive_view(), entry, sequence, previous)
    }
    pub(super) fn require_layout(&self) -> Result<(), String> {
        require_archive_layout(&self.archive_view())
    }
    fn require_full_archives(&self) -> Result<(), String> {
        self.root.require_current()?;
        require_archive_files(&self.archive_view())?;
        self.require_live()
    }
    /// Consume the session only after full archival checks, returning unchanged
    /// genuine admission for a separately pinned metadata restore.
    pub fn into_original_root(self) -> Result<AdmittedCurrentMixedRoot<'a>, String> {
        self.require_full_archives()?;
        Ok(self.root)
    }

    /// Consumes only a resolved Ready session, retaining the actual joint owner,
    /// original admission and every reservation/work/storage counter. The token
    /// borrows policies; it cannot start another training segment.
    pub fn into_ready_continuation(self) -> Result<ReadyCurrentMixedContinuation<'a>, String> {
        self.progress
            .require_ready_continuation(self.blocked, self.head.is_some())?;
        self.current_policy()?;
        self.require_full_archives()?;
        self.current_policy()?;
        Ok(ReadyCurrentMixedContinuation { session: self })
    }
}

fn require_archive_entry(
    archive: &ArchiveView<'_>,
    entry: &Entry,
    sequence: u64,
    previous: &str,
) -> Result<(), String> {
    if entry.schema != archive.journal_schema
        || entry.sequence != sequence
        || entry.previous != previous
        || entry.root != archive.root_checksum
        || entry.spec != archive.spec.checksum()?
        || entry.pool != archive.pool
    {
        return Err("Current mixed journal original chain/spec/root/pool differs".into());
    }
    Ok(())
}

fn require_archive_layout(archive: &ArchiveView<'_>) -> Result<(), String> {
    let mut expected = vec!["run-spec.json".into()];
    expected.extend((1..=archive.sequence).map(|n| format!("head-{n:016x}.json")));
    expected.extend(archive.references.keys().map(|n| format!("batch-{n:016x}")));
    expected.sort();
    if names(
        archive.directory,
        1 + archive.spec.max_journal_entries as usize + archive.spec.max_attempts as usize,
    )? != expected
    {
        return Err("Current mixed stale head or extra/missing root outputs".into());
    }
    for (&attempt, refs) in archive.references {
        let mut expected = refs.iter().map(|p| p.name.clone()).collect::<Vec<_>>();
        expected.sort();
        if names(&archive.batch(attempt), 9)? != expected {
            return Err(
                "Current mixed unreferenced/temp batch outputs refuse restore/close".into(),
            );
        }
    }
    Ok(())
}

pub(super) fn require_archive_files(archive: &ArchiveView<'_>) -> Result<(), String> {
    require_archive_layout(archive)?;
    let spec = archive
        .spec_pin
        .ok_or("Current mixed original run pin missing")?;
    read_pin(archive.directory, spec, SPEC_CAP)?;
    let mut previous = String::new();
    let mut output = spec.bytes as u64;
    let mut metadata = output;
    let mut logical = 0u64;
    let mut stored = 0u64;
    for n in 1..=archive.sequence {
        let name = format!("head-{n:016x}.json");
        let raw = read_bounded(&archive.directory.join(name), HEAD_CAP)?;
        let entry: Entry = closed(&raw, HEAD_CAP)?;
        require_archive_entry(archive, &entry, n, &previous)?;
        if let Event::CollectReturn {
            member,
            record: Some(pin),
            ..
        } = &entry.event
        {
            pin.validate(*member, archive.spec)?;
            logical = logical
                .checked_add(pin.raw_bytes() as u64)
                .ok_or("Current mixed archive logical byte overflow")?;
            stored = stored
                .checked_add(pin.file().bytes as u64)
                .ok_or("Current mixed archive stored byte overflow")?;
        }
        previous = sha(&raw);
        output += raw.len() as u64;
        metadata += raw.len() as u64;
    }
    if archive.head.is_none_or(|p| p.sha256 != previous) {
        return Err("Current mixed original latest head changed".into());
    }
    for (&attempt, refs) in archive.references {
        for p in refs {
            read_physical_pin(&archive.batch(attempt), p, archive.spec)?;
            output += p.bytes as u64;
            if !p.name.starts_with("record-") {
                metadata += p.bytes as u64;
            }
        }
    }
    archive.spec.check_storage(logical, stored, metadata)?;
    if output != archive.output_bytes
        || metadata != archive.metadata_bytes
        || logical != archive.raw_record_bytes
        || stored != archive.stored_record_bytes
        || output
            != stored
                .checked_add(metadata)
                .ok_or("Current mixed archive retained overflow")?
    {
        return Err("Current mixed original archive byte accounting differs".into());
    }
    Ok(())
}
