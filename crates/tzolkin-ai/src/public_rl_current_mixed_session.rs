//! Data-only current-mixed collection. A reusable genuine root borrows qualified
//! BC once. Local reservations are durable before work; no owner update,
//! continuation restoration, retry or global anti-fork registry is provided.
use crate::arena::BorrowedArenaPolicy;
use crate::public_native::PreparedPublicPolicy;
use crate::public_rl_current_mixed_source::{
    self as source, PlannedCurrentMixedCohort, ValidatedCurrentMixedCohort,
};
use crate::public_rl_mixed_handle::MixedPolicyHandle;
use crate::public_rl_mixed_native::FrozenMixedRoster;
use crate::public_rl_mixed_session::{
    FilePin, ImportedMixedRlParent, MixedRlCollectionSpec, OpponentSpec, Writer, read_pin,
    require_output_outside_directory,
};
use crate::public_rl_mixed_training_session::{MixedOneBatchSession, MixedOneBatchSpec};
use crate::public_rl_paired_session::{local, read_bounded};
use crate::public_stochastic_record::{MAX_RECORD_BYTES, unique_json};
use crate::search::PreparedSearch;
use serde::{Deserialize, Serialize};
use serde_json::json;
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeSet,
    fs,
    path::{Path, PathBuf},
    time::Instant,
};

pub const SPEC_SCHEMA: &str = "tzolkin-current-mixed-data-collection-spec-v2";
const META_CAP: usize = 64 * 1024;
const MAX_FAMILIES: usize = 65_536;
const MAX_BATCHES: usize = 10_000;
fn sha(raw: &[u8]) -> String {
    format!("{:x}", Sha256::digest(raw))
}

/// Restoring the old genuinely updated journal is the only public admission.
/// Its old count1 artifact/status remains unchanged, including continuation=false.
pub struct AdmittedCurrentMixedRoot<'a> {
    session: MixedOneBatchSession<'a>,
    snapshot: serde_json::Value,
    prefix: [u64; 2],
    next: u64,
    consumed: BTreeSet<String>,
}
impl<'a> AdmittedCurrentMixedRoot<'a> {
    pub fn restore_one_batch(
        parent: ImportedMixedRlParent,
        prepared: &'a PreparedPublicPolicy,
        spec: MixedOneBatchSpec,
        head_raw: &[u8],
        expected_sha256: &str,
        directory: &Path,
    ) -> Result<Self, String> {
        let session = MixedOneBatchSession::restore(
            parent,
            prepared,
            spec,
            head_raw,
            expected_sha256,
            directory,
        )?;
        let (_, prefix) = session.current_data_context()?;
        let snapshot = session.current_data_snapshot()?;
        let status = session.status();
        let next = status.next_episode_ordinal();
        let consumed = status
            .reserved_families()
            .iter()
            .cloned()
            .collect::<BTreeSet<_>>();
        if consumed.len() != status.reserved_families().len() || consumed.len() > MAX_FAMILIES {
            return Err("Current mixed root reservation closure bound".into());
        }
        Ok(Self {
            session,
            snapshot,
            prefix,
            next,
            consumed,
        })
    }
    pub fn handle(&self) -> Result<MixedPolicyHandle<'_>, String> {
        MixedPolicyHandle::new(self.session.owner().ok_or("Missing admitted mixed owner")?)
    }
    pub fn next_episode_ordinal(&self) -> u64 {
        self.next
    }
    pub fn consumed_families(&self) -> &BTreeSet<String> {
        &self.consumed
    }
    fn require_current(&self) -> Result<(), String> {
        if self.session.current_data_snapshot()? != self.snapshot {
            return Err("Admitted current mixed root bytes changed".into());
        }
        Ok(())
    }
}

#[derive(Clone)]
pub struct CurrentMixedCollectionSpec {
    inner: MixedRlCollectionSpec,
}
impl CurrentMixedCollectionSpec {
    pub fn from_json(raw: &[u8]) -> Result<Self, String> {
        Ok(Self {
            inner: MixedRlCollectionSpec::from_json_schema(raw, SPEC_SCHEMA)?,
        })
    }
    fn pool_checksum(&self) -> Result<String, String> {
        crate::public_policy_update::hash(
            b"tzolkin-current-mixed-frozen-pool-v2\0",
            &(
                &self.inner.wire.weights,
                &self.inner.wire.search,
                &self.inner.wire.rosters,
            ),
        )
    }
}
/// These budgets count reserved attempts, not successful mixed updates.
#[derive(Clone, Copy)]
pub struct CurrentMixedDataLimits {
    max_batches: usize,
    max_families: usize,
}
impl CurrentMixedDataLimits {
    pub fn new(max_batches: usize, max_families: usize) -> Result<Self, String> {
        if !(1..=MAX_BATCHES).contains(&max_batches) || !(4..=MAX_FAMILIES).contains(&max_families)
        {
            return Err("Current mixed data reservation budget".into());
        }
        Ok(Self {
            max_batches,
            max_families,
        })
    }
}
/// The root and search are admitted once. Each method performs at most one
/// fixed whole-four batch. A dropped/failed batch retains all its reservations.
pub struct CurrentMixedDataSession<'a> {
    root: AdmittedCurrentMixedRoot<'a>,
    search: PreparedSearch,
    pool: String,
    limits: CurrentMixedDataLimits,
    batches: usize,
    blocked: bool,
    reservations: Vec<ReservedBatch>,
}
struct ReservedBatch {
    directory: PathBuf,
    spec: CurrentMixedCollectionSpec,
    plan: serde_json::Value,
    reservation: FilePin,
    inventory: Option<Vec<FilePin>>,
    audit_attempted: bool,
    audit_pins: Vec<FilePin>,
}
impl<'a> CurrentMixedDataSession<'a> {
    pub fn new(
        root: AdmittedCurrentMixedRoot<'a>,
        pool: &CurrentMixedCollectionSpec,
        limits: CurrentMixedDataLimits,
    ) -> Result<Self, String> {
        root.require_current()?;
        if root
            .consumed
            .len()
            .checked_add(4)
            .is_none_or(|n| n > limits.max_families)
        {
            return Err("Current mixed inherited family budget".into());
        }
        Ok(Self {
            root,
            search: PreparedSearch::new(&pool.inner.wire.search)?,
            pool: pool.pool_checksum()?,
            limits,
            batches: 0,
            blocked: false,
            reservations: Vec::new(),
        })
    }
    pub fn root(&self) -> &AdmittedCurrentMixedRoot<'a> {
        &self.root
    }
    fn rosters<'s>(
        &'s self,
        spec: &'s CurrentMixedCollectionSpec,
    ) -> Result<[FrozenMixedRoster<'s>; 4], String> {
        let owner = self
            .root
            .session
            .owner()
            .ok_or("Missing current mixed parent")?;
        let parent = MixedPolicyHandle::new(owner)?;
        let (prepared, _) = self.root.session.current_data_context()?;
        spec.inner
            .wire
            .rosters
            .iter()
            .map(|seats| {
                FrozenMixedRoster::for_current(
                    seats
                        .iter()
                        .map(|p| {
                            p.as_ref().map(|p| match p {
                                OpponentSpec::Heuristic => {
                                    BorrowedArenaPolicy::Heuristic(&spec.inner.wire.weights)
                                }
                                OpponentSpec::CornFirstSetup => {
                                    BorrowedArenaPolicy::CornFirstSetup(&spec.inner.wire.weights)
                                }
                                OpponentSpec::CornFirstUxmalOpening => {
                                    BorrowedArenaPolicy::CornFirstUxmalOpening(
                                        &spec.inner.wire.weights,
                                    )
                                }
                                OpponentSpec::PreparedBc => BorrowedArenaPolicy::PublicBc(prepared),
                                OpponentSpec::ParentArgmax => BorrowedArenaPolicy::Mixed(owner),
                                OpponentSpec::Search => BorrowedArenaPolicy::Search(&self.search),
                            })
                        })
                        .collect(),
                    parent.identity(),
                )
            })
            .collect::<Result<Vec<_>, _>>()?
            .try_into()
            .map_err(|_| "Current mixed roster shape".into())
    }
    fn plan(
        &self,
        spec: &CurrentMixedCollectionSpec,
        consumed: &BTreeSet<String>,
    ) -> Result<PlannedCurrentMixedCohort, String> {
        if spec.pool_checksum()? != self.pool {
            return Err("Current mixed frozen pool changed".into());
        }
        let configs = spec.inner.configs()?;
        let rosters = self.rosters(spec)?;
        let handle = self.root.handle()?;
        let headers = configs
            .iter()
            .zip(&rosters)
            .map(|(c, r)| source::header(c, &handle, r, consumed))
            .collect::<Result<Vec<_>, _>>()?
            .try_into()
            .map_err(|_| "Current mixed header shape")?;
        PlannedCurrentMixedCohort::new(configs, headers)
    }
    fn require_prior_pins(&self) -> Result<(), String> {
        self.root.require_current()?;
        for r in &self.reservations {
            read_pin(&r.directory, &r.reservation, META_CAP)?;
            if let Some(pins) = &r.inventory
                && inventory(&r.directory, r.spec.inner.wire.max_output_bytes)? != *pins
            {
                return Err("Current mixed prior collected bytes changed".into());
            }
            for pin in &r.audit_pins {
                read_pin(&r.directory.join("audit"), pin, META_CAP)?;
            }
            if !r.audit_pins.is_empty()
                && audit_inventory(&r.directory.join("audit"))? != r.audit_pins
            {
                return Err("Current mixed prior audit inventory changed".into());
            }
        }
        Ok(())
    }
    /// Reserve every family and RNG ordinal in a synced new-only receipt before
    /// the first world. An interrupted attempt blocks this live session; no retry.
    pub fn collect_new(
        &mut self,
        spec: CurrentMixedCollectionSpec,
        output: &Path,
    ) -> Result<usize, String> {
        if self.blocked || self.batches >= self.limits.max_batches {
            return Err("Current mixed data session stopped or batch budget consumed".into());
        }
        self.require_prior_pins()?;
        let directory = local(output, true)?;
        self.root.session.require_current_data_output(&directory)?;
        for r in &self.reservations {
            require_output_outside_directory(&r.directory, &directory)?;
        }
        let plan = self.plan(&spec, &self.root.consumed)?;
        let (next, used) = reserve(
            &plan,
            self.root.prefix,
            self.root.next,
            &self.root.consumed,
            self.limits.max_families,
        )?;
        let batch = self.batches;
        let mut writer = Writer::new(&directory, spec.inner.wire.max_output_bytes)?;
        // Conservatively consume locally even if persisting the reservation fails.
        self.blocked = true;
        self.batches += 1;
        self.root.next = next;
        self.root.consumed = used;
        let spec_pin = writer.raw("collection-spec.json", &spec.inner.to_json()?, META_CAP)?;
        let reservation=writer.save("reservation.json",&json!({"schema":"tzolkin-current-mixed-whole-four-reservation-v2",
            "batchOrdinal":batch,"parent":plan.parent_identity(),"rootPins":self.root.snapshot,
            "maxBatches":self.limits.max_batches,"maxFamilies":self.limits.max_families,
            "collectionSpec":spec_pin,"poolChecksum":self.pool,
            "samplingPrefix":self.root.prefix,"nextEpisodeOrdinal":next,"families":plan.rollout_families(),
            "planChecksum":plan.checksum(),"unknownUntilWholeReturned":true,"globalSeedNoveltyVerified":false}))?;
        self.reservations.push(ReservedBatch {
            directory: directory.clone(),
            spec: spec.clone(),
            plan: plan.receipt(),
            reservation: reservation.clone(),
            inventory: None,
            audit_attempted: false,
            audit_pins: Vec::new(),
        });
        let plan_pin = writer.save("plan.json", &plan.receipt())?;
        let handle = self.root.handle()?;
        let rosters = self.rosters(&spec)?;
        let began = Instant::now();
        let mut members = Vec::new();
        for (member, roster) in rosters.iter().enumerate() {
            self.require_prior_pins()?;
            writer.save(
                &format!("collect-started-{member}.json"),
                &started(member, &plan_pin),
            )?;
            writer.require_current()?;
            let game = source::collect(
                &plan.configurations()[member],
                &handle,
                roster,
                plan.headers[member].clone(),
            )?;
            writer.require_current()?;
            let raw = source::encode(&game)?;
            let record = writer.raw(&format!("record-{member}.json"), &raw, MAX_RECORD_BYTES)?;
            let returned = MemberReturn {
                member,
                record,
                complete: game.terminal.is_some() && game.failure.is_none(),
                observed_callbacks: game.counts.observed_callbacks,
                learner_callbacks: game.counts.accepted_samples,
                learner_candidate_rows: game.counts.candidate_rows_reserved,
                opponent_attempts: game
                    .callbacks
                    .iter()
                    .filter(|c| c.opponent_attempted)
                    .count(),
            };
            writer.save(&format!("collect-returned-{member}.json"), &returned)?;
            members.push(returned);
        }
        self.require_prior_pins()?;
        writer.require_current()?;
        let all_complete = members.iter().all(|m| m.complete);
        let report = CollectionReceipt {
            schema: "tzolkin-current-mixed-collected-batch-v2".into(),
            batch_ordinal: batch,
            reservation,
            plan: plan_pin,
            members,
            seconds: began.elapsed().as_secs_f64(),
            training_data_admission: false,
            audits_required: 4,
            updated_owners: 0,
            recollection_available: false,
        };
        writer.save("collection.json", &report)?;
        let pins = retain_written_inventory(
            &writer,
            inventory(&directory, spec.inner.wire.max_output_bytes)?,
        )?;
        self.reservations[batch].inventory = Some(pins);
        self.require_prior_pins()?;
        writer.require_current()?;
        self.blocked = !all_complete;
        Ok(batch)
    }
    /// Caller must pin the manifest externally. The live reservation is also
    /// required; saved JSON or a copied directory alone cannot mint admission.
    pub fn audit_saved(
        &mut self,
        batch: usize,
        collection_raw: &[u8],
        expected_sha256: &str,
    ) -> Result<ValidatedCurrentMixedCohort, String> {
        if self.blocked {
            return Err("Current mixed stopped/Unknown work cannot be audited or retried".into());
        }
        self.require_prior_pins()?;
        let r = self
            .reservations
            .get(batch)
            .ok_or("Missing current mixed reserved batch")?;
        if r.audit_attempted || r.inventory.is_none() || sha(collection_raw) != expected_sha256 {
            return Err(
                "Current mixed audit already attempted or external manifest pin differs".into(),
            );
        }
        let receipt: CollectionReceipt =
            serde_json::from_value(unique_json(collection_raw, META_CAP)?)
                .map_err(|e| e.to_string())?;
        if receipt.schema != "tzolkin-current-mixed-collected-batch-v2"
            || receipt.batch_ordinal != batch
            || receipt.reservation != r.reservation
            || receipt.plan.name != "plan.json"
            || receipt.members.len() != 4
            || receipt.training_data_admission
            || receipt.audits_required != 4
            || receipt.updated_owners != 0
            || receipt.recollection_available
            || read_bounded(&r.directory.join("collection.json"), META_CAP)? != collection_raw
        {
            return Err("Current mixed closed collection receipt differs".into());
        }
        if unique_json(&read_pin(&r.directory, &receipt.plan, META_CAP)?, META_CAP)? != r.plan {
            return Err("Current mixed full saved plan differs".into());
        }
        let mut used = self.root.consumed.clone();
        for family in r
            .plan
            .get("families")
            .and_then(|v| v.as_array())
            .ok_or("Missing reserved current mixed families")?
        {
            used.remove(family.as_str().ok_or("Current mixed family string")?);
        }
        let plan = self.plan(&r.spec, &used)?;
        if plan.receipt() != r.plan {
            return Err("Current mixed genuine current plan differs".into());
        }
        let mut raw = Vec::new();
        for (i, m) in receipt.members.iter().enumerate() {
            if m.member != i
                || m.record.name != format!("record-{i}.json")
                || !m.complete
                || unique_json(
                    &read_bounded(
                        &r.directory.join(format!("collect-started-{i}.json")),
                        META_CAP,
                    )?,
                    META_CAP,
                )? != started(i, &receipt.plan)
                || unique_json(
                    &read_bounded(
                        &r.directory.join(format!("collect-returned-{i}.json")),
                        META_CAP,
                    )?,
                    META_CAP,
                )? != serde_json::to_value(m).map_err(|e| e.to_string())?
            {
                return Err("Current mixed saved member/start/return differs before audit".into());
            }
            let bytes = read_pin(&r.directory, &m.record, MAX_RECORD_BYTES)?;
            let record = source::check_record(&bytes, &plan.headers[i])?;
            if record.terminal.is_some() != m.complete
                || record.failure.is_some()
                || record.counts.observed_callbacks != m.observed_callbacks
                || record.counts.accepted_samples != m.learner_callbacks
                || record.counts.candidate_rows_reserved != m.learner_candidate_rows
                || record
                    .callbacks
                    .iter()
                    .filter(|c| c.opponent_attempted)
                    .count()
                    != m.opponent_attempts
            {
                return Err("Current mixed saved counters differ before fresh audit".into());
            }
            raw.push(bytes);
        }
        let directory = r.directory.clone();
        let spec = r.spec.clone();
        self.blocked = true;
        self.reservations[batch].audit_attempted = true;
        let output = directory.join("audit");
        let mut writer = Writer::new(&output, 2 * 1024 * 1024)?;
        writer.save("plan.json", &plan.receipt())?;
        let handle = self.root.handle()?;
        let rosters = self.rosters(&spec)?;
        let mut results = Vec::new();
        for (member, (bytes, roster)) in raw.iter().zip(&rosters).enumerate() {
            self.require_prior_pins()?;
            writer.save(&format!("audit-started-{member}.json"),&json!({"schema":"tzolkin-current-mixed-audit-started-v2","member":member,"record":receipt.members[member].record,"unknownUntilReturned":true}))?;
            writer.require_current()?;
            let result = source::audit(bytes, &handle, roster, plan.headers[member].clone());
            writer.require_current()?;
            writer.save(&format!("audit-returned-{member}.json"),&json!({"schema":"tzolkin-current-mixed-audit-returned-v2","member":member,"passed":result.is_ok(),"error":result.as_ref().err()}))?;
            results.push(result);
        }
        self.require_prior_pins()?;
        writer.require_current()?;
        let cohort = plan.finish(
            results
                .try_into()
                .map_err(|_| "Current mixed whole-four result shape")?,
        )?;
        writer.save("cohort.json", &cohort.receipt())?;
        self.reservations[batch].audit_pins =
            retain_written_inventory(&writer, audit_inventory(&output)?)?;
        self.require_prior_pins()?;
        writer.require_current()?;
        self.blocked = false;
        Ok(cohort)
    }
}
fn retain_written_inventory(
    writer: &Writer<'_>,
    observed: Vec<FilePin>,
) -> Result<Vec<FilePin>, String> {
    if observed != writer.pinned_inventory() {
        return Err("Current mixed final inventory differs from originally written bytes".into());
    }
    writer.require_current()?;
    Ok(observed)
}
fn reserve(
    plan: &PlannedCurrentMixedCohort,
    prefix: [u64; 2],
    next: u64,
    used: &BTreeSet<String>,
    cap: usize,
) -> Result<(u64, BTreeSet<String>), String> {
    let new_next = next
        .checked_add(4)
        .ok_or("Current mixed ordinal overflow")?;
    let mut reserved = used.clone();
    for (i, c) in plan.headers.iter().enumerate() {
        if c.config.sampling_seed != format!("{:016x}", prefix[0])
            || c.config.replicate_ordinal != format!("{:016x}", prefix[1])
            || c.config.episode_ordinal != format!("{:016x}", next + i as u64)
            || !reserved.insert(c.native_family_id.clone())
        {
            return Err(
                "Current mixed reservation cannot reset/reuse family or sampler prefix".into(),
            );
        }
    }
    if reserved.len() > cap {
        return Err("Current mixed reserved family budget".into());
    }
    Ok((new_next, reserved))
}
fn started(member: usize, plan: &FilePin) -> serde_json::Value {
    json!({"schema":"tzolkin-current-mixed-collect-started-v2","member":member,"planSha256":plan.sha256,"unknownUntilReturned":true})
}
#[derive(Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct MemberReturn {
    member: usize,
    record: FilePin,
    complete: bool,
    observed_callbacks: usize,
    learner_callbacks: usize,
    learner_candidate_rows: usize,
    opponent_attempts: usize,
}
#[derive(Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct CollectionReceipt {
    schema: String,
    batch_ordinal: usize,
    reservation: FilePin,
    plan: FilePin,
    members: Vec<MemberReturn>,
    seconds: f64,
    training_data_admission: bool,
    audits_required: usize,
    updated_owners: usize,
    recollection_available: bool,
}
fn inventory(dir: &Path, cap: u64) -> Result<Vec<FilePin>, String> {
    let mut names = vec![
        "reservation.json".to_owned(),
        "collection-spec.json".into(),
        "plan.json".into(),
        "collection.json".into(),
    ];
    for i in 0..4 {
        for stem in ["collect-started", "collect-returned", "record"] {
            names.push(format!("{stem}-{i}.json"));
        }
    }
    inventory_named(dir, &names, cap, true)
}
fn audit_inventory(dir: &Path) -> Result<Vec<FilePin>, String> {
    let mut names = vec!["plan.json".to_owned(), "cohort.json".into()];
    for i in 0..4 {
        for stem in ["audit-started", "audit-returned"] {
            names.push(format!("{stem}-{i}.json"));
        }
    }
    inventory_named(dir, &names, 2 * 1024 * 1024, false)
}
fn inventory_named(
    dir: &Path,
    names: &[String],
    cap: u64,
    allow_audit: bool,
) -> Result<Vec<FilePin>, String> {
    let mut actual = Vec::new();
    for e in fs::read_dir(dir).map_err(|e| e.to_string())? {
        if actual.len() > names.len() {
            return Err("Current mixed inventory count cap".into());
        }
        let e = e.map_err(|e| e.to_string())?;
        if allow_audit && e.file_name() == "audit" {
            let m = fs::symlink_metadata(e.path()).map_err(|e| e.to_string())?;
            if !m.is_dir() || m.is_symlink() {
                return Err("Current mixed audit must be regular directory".into());
            }
            continue;
        }
        actual.push(
            e.file_name()
                .into_string()
                .map_err(|_| "Current mixed non-UTF8 leaf")?,
        );
    }
    let mut expected = names.to_vec();
    expected.sort();
    actual.sort();
    if actual != expected {
        return Err("Current mixed inventory missing/unreferenced files".into());
    }
    let mut total = 0u64;
    expected
        .into_iter()
        .map(|name| {
            let limit = if name.starts_with("record-") {
                MAX_RECORD_BYTES
            } else {
                META_CAP
            };
            let raw = read_bounded(&dir.join(&name), limit)?;
            total = total
                .checked_add(raw.len() as u64)
                .ok_or("Current mixed inventory byte overflow")?;
            if total > cap {
                return Err("Current mixed inventory byte cap".into());
            }
            Ok(FilePin {
                name,
                bytes: raw.len(),
                sha256: sha(&raw),
            })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn current_mixed_whole_four_reservations_preserve_prefix_and_consumption() {
        let plan = source::tests::plan();
        let used = BTreeSet::from(["f".repeat(64)]);
        let (next, reserved) = reserve(&plan, [7, 11], 9, &used, 8).unwrap();
        assert_eq!(next, 13);
        assert_eq!(reserved.len(), 5);
        assert!(used.is_subset(&reserved));
        assert_eq!(plan.parent_identity().mixed_update_count(), 1);
        assert!(reserve(&plan, [7, 11], 9, &reserved, 9).is_err());
        assert!(reserve(&plan, [7, 11], 13, &used, 8).is_err());
        assert!(reserve(&plan, [8, 11], 9, &used, 8).is_err());
        assert!(reserve(&plan, [7, 12], 9, &used, 8).is_err());
        assert!(reserve(&plan, [7, 11], 9, &used, 4).is_err());
        assert!(reserve(&plan, [7, 11], u64::MAX - 2, &used, 8).is_err());
        assert!(CurrentMixedDataLimits::new(0, 8).is_err());
        assert!(CurrentMixedDataLimits::new(1, 3).is_err());

        let opponent = json!({"kind":"heuristic"});
        let mut spec = json!({"schema":SPEC_SCHEMA,"catalogHash":crate::replay::catalog_hash(),
            "configurations":plan.headers.each_ref().map(|h|&h.config),
            "rosters":[[null,opponent,opponent],[opponent,null,opponent],
                [null,opponent,opponent,opponent],[opponent,opponent,opponent,null]],
            "weights":crate::policy::HeuristicWeights::default(),"search":crate::search::SearchConfig::default(),
            "maxOutputBytes":4*MAX_RECORD_BYTES+2*1024*1024});
        let checksum = CurrentMixedCollectionSpec::from_json(&serde_json::to_vec(&spec).unwrap())
            .unwrap()
            .pool_checksum()
            .unwrap();
        for (i, c) in spec["configurations"]
            .as_array_mut()
            .unwrap()
            .iter_mut()
            .enumerate()
        {
            c["environmentSeed"] = json!(100 + i);
            c["episodeOrdinal"] = json!(format!("{:016x}", 13 + i));
        }
        let next =
            CurrentMixedCollectionSpec::from_json(&serde_json::to_vec(&spec).unwrap()).unwrap();
        assert_eq!(next.pool_checksum().unwrap(), checksum);
        spec["rosters"][0][1]["kind"] = json!("preparedBc");
        let changed =
            CurrentMixedCollectionSpec::from_json(&serde_json::to_vec(&spec).unwrap()).unwrap();
        assert_ne!(changed.pool_checksum().unwrap(), checksum);
    }

    #[test]
    fn current_mixed_final_inventory_retains_original_pins_on_local_paths() {
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let base = std::env::temp_dir().join(format!(
            "tzolkin-current-mixed-pins-{}-{nonce}",
            std::process::id()
        ));
        fs::create_dir(&base).unwrap();
        let directory = local(&base.join("collection"), true).unwrap();
        let mut writer = Writer::new(&directory, 4096).unwrap();
        for name in ["reservation.json", "collection-spec.json", "plan.json"] {
            writer.raw(name, b"{}", META_CAP).unwrap();
        }
        for i in 0..4 {
            for stem in ["collect-started", "collect-returned", "record"] {
                writer
                    .raw(&format!("{stem}-{i}.json"), b"{}", META_CAP)
                    .unwrap();
            }
        }
        writer.raw("collection.json", b"{}", META_CAP).unwrap();
        let pins = writer.pinned_inventory();
        assert_eq!(pins.len(), 16);
        assert!(pins.iter().any(|p| p.name == "collection.json"));
        assert!(retain_written_inventory(&writer, inventory(&directory, 4096).unwrap()).is_ok());
        #[cfg(windows)]
        assert!(local(&fs::canonicalize(&directory).unwrap(), false).is_err());
        fs::write(directory.join("record-0.json"), b"changed").unwrap();
        assert!(retain_written_inventory(&writer, inventory(&directory, 4096).unwrap()).is_err());
        assert!(writer.pinned_inventory() == pins);

        let output = directory.join("audit");
        let mut audit = Writer::new(&output, 4096).unwrap();
        audit.raw("plan.json", b"{}", META_CAP).unwrap();
        for i in 0..4 {
            for stem in ["audit-started", "audit-returned"] {
                audit
                    .raw(&format!("{stem}-{i}.json"), b"{}", META_CAP)
                    .unwrap();
            }
        }
        audit.raw("cohort.json", b"{}", META_CAP).unwrap();
        let pins = audit.pinned_inventory();
        assert_eq!(pins.len(), 10);
        assert!(pins.iter().any(|p| p.name == "cohort.json"));
        assert!(retain_written_inventory(&audit, audit_inventory(&output).unwrap()).is_ok());
        fs::write(output.join("audit-returned-0.json"), b"changed").unwrap();
        assert!(retain_written_inventory(&audit, audit_inventory(&output).unwrap()).is_err());
        assert!(audit.pinned_inventory() == pins);
        fs::remove_dir_all(base).unwrap();
    }
}
