//! Bounded whole-four mixed collection and saved-only fresh training-data audit.
//! This data session emits no updated owner and never resumes a lost collection.
use crate::arena::BorrowedArenaPolicy;
use crate::policy::HeuristicWeights;
use crate::public_native::PreparedPublicPolicy;
use crate::public_policy_paired_long::PairedRlParent;
use crate::public_rl_mixed_cohort::{PlannedMixedRlCohort, ValidatedMixedRlCohort};
use crate::public_rl_mixed_episode::ValidatedMixedRlEpisode;
use crate::public_rl_mixed_native::{
    FrozenMixedRoster, MixedRlLearner, audit_mixed_rl_record_bytes, collect_mixed_rl_native,
    encode_mixed_rl_record,
};
use crate::public_rl_paired_session::read_bounded;
use crate::public_stochastic_native::NativeStochasticConfig;
use crate::public_stochastic_record::{Config, MAX_RECORD_BYTES, unique_json};
use crate::search::{PreparedSearch, SearchConfig};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{fs, path::Path, time::Instant};

pub const SPEC_SCHEMA: &str = "tzolkin-mixed-rl-collection-spec-v1";
pub const MAX_SPEC_BYTES: usize = 64 * 1024;
const RECEIPT_CAP: usize = 64 * 1024;
const METADATA_RESERVE: u64 = 2 * 1024 * 1024;
fn sha(raw: &[u8]) -> String {
    format!("{:x}", Sha256::digest(raw))
}

enum ImportedOwner {
    Count1(Box<crate::public_rl_session::RlTrainingSession>),
    Paired(Box<crate::public_rl_paired_session::PairedTrainingSession>),
}
/// A read-only import preserves the original owner/task/checksum and all
/// reservation exclusions. It is never a mixed-updated owner or root reset.
pub struct ImportedMixedRlParent {
    owner: ImportedOwner,
    sampling_seed: u64,
    replicate: u64,
    next_ordinal: u64,
}
fn integer(v: &serde_json::Value, key: &str) -> Result<u64, String> {
    v[key]
        .as_u64()
        .ok_or_else(|| format!("Missing imported stream integer {key}"))
}
impl ImportedMixedRlParent {
    pub fn from_count1(
        prepared: &PreparedPublicPolicy,
        raw: &[u8],
        expected_sha256: &str,
    ) -> Result<Self, String> {
        let initial = crate::public_rl_artifact::InitializedPublicRlPolicy::from_bc(prepared)?;
        let owner =
            crate::public_rl_session::RlTrainingSession::restore(initial, raw, expected_sha256)?;
        let progress = owner.progress()?;
        if progress.unresolved_batch()
            || progress.completed_updates() != 1
            || !matches!(
                owner.parent(),
                Some(crate::public_policy_repeat::RlUpdateParent::Count1(_))
            )
        {
            return Err("Mixed import requires genuine resolved Count1".into());
        }
        let config = serde_json::to_value(owner.config()).map_err(|e| e.to_string())?;
        Ok(Self {
            sampling_seed: integer(&config, "samplingSeed")?,
            replicate: integer(&config, "replicateOrdinal")?,
            next_ordinal: progress.next_episode_ordinal(),
            owner: ImportedOwner::Count1(Box::new(owner)),
        })
    }
    pub fn from_paired(
        prepared: &PreparedPublicPolicy,
        spec: crate::public_rl_paired_session::PairedRunSpec,
        raw: &[u8],
        expected_sha256: &str,
        directory: &Path,
    ) -> Result<Self, String> {
        let config = serde_json::to_value(&spec).map_err(|e| e.to_string())?;
        let owner = crate::public_rl_paired_session::PairedTrainingSession::restore(
            spec,
            prepared,
            raw,
            expected_sha256,
            directory,
        )?;
        owner.require_evaluation_ready()?;
        if !matches!(owner.current_parent(), PairedRlParent::Paired(_)) {
            return Err("Paired mixed import requires a genuine joint owner".into());
        }
        let report = serde_json::to_value(owner.snapshot()).map_err(|e| e.to_string())?;
        Ok(Self {
            sampling_seed: integer(&config, "samplingSeed")?,
            replicate: integer(&config, "replicateOrdinal")?,
            next_ordinal: integer(&report["progress"], "nextOrdinal")?,
            owner: ImportedOwner::Paired(Box::new(owner)),
        })
    }
    fn parent(&self) -> PairedRlParent<'_> {
        match &self.owner {
            ImportedOwner::Count1(s) => match s.parent() {
                Some(crate::public_policy_repeat::RlUpdateParent::Count1(p)) => {
                    PairedRlParent::Count1(p)
                }
                _ => unreachable!("checked Count1"),
            },
            ImportedOwner::Paired(s) => s.current_parent(),
        }
    }
    pub fn collection_session<'a>(
        &'a self,
        spec: MixedRlCollectionSpec,
        prepared: &'a PreparedPublicPolicy,
    ) -> Result<MixedRlCollectionSession<'a>, String> {
        for (i, c) in spec.configurations.iter().enumerate() {
            let ordinal = self
                .next_ordinal
                .checked_add(i as u64)
                .ok_or("Mixed imported ordinal overflow")?;
            if c.sampling_seed != format!("{:016x}", self.sampling_seed)
                || c.replicate_ordinal != format!("{:016x}", self.replicate)
                || c.episode_ordinal != format!("{ordinal:016x}")
            {
                return Err("Mixed import cannot reset or relabel parent sampler prefix".into());
            }
        }
        let seeds = spec.configurations.each_ref().map(|c| c.environment_seed);
        match &self.owner {
            ImportedOwner::Count1(s) => s.require_unused_training_families(&seeds)?,
            ImportedOwner::Paired(s) => s.require_unused_training_families(&seeds)?,
        }
        MixedRlCollectionSession::new(spec, self.parent(), prepared)
    }
}

#[derive(Clone, Deserialize, Serialize)]
#[serde(tag = "kind", rename_all = "camelCase", deny_unknown_fields)]
enum OpponentSpec {
    Heuristic,
    CornFirstSetup,
    CornFirstUxmalOpening,
    PreparedBc,
    ParentArgmax,
    Search,
}
/// Every member names its absolute learner slot (`null`) and frozen opponents.
/// Configurations include exact environment/sampling/replicate/episode identities.
#[derive(Clone, Deserialize, Serialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct MixedRlCollectionSpec {
    schema: String,
    catalog_hash: String,
    configurations: [Config; 4],
    rosters: [Vec<Option<OpponentSpec>>; 4],
    weights: HeuristicWeights,
    search: SearchConfig,
    max_output_bytes: u64,
}
impl MixedRlCollectionSpec {
    pub fn from_json(raw: &[u8]) -> Result<Self, String> {
        let value = unique_json(raw, MAX_SPEC_BYTES)?;
        let s: Self = serde_json::from_value(value).map_err(|e| e.to_string())?;
        s.weights.validate()?;
        s.search.validate()?;
        if s.schema != SPEC_SCHEMA
            || s.catalog_hash != crate::replay::catalog_hash()
            || !(4 * MAX_RECORD_BYTES as u64 + METADATA_RESERVE..=512 * 1024 * 1024)
                .contains(&s.max_output_bytes)
        {
            return Err(
                "Mixed spec requires canonical catalog and whole4 bounded output reserve".into(),
            );
        }
        let configs = s.configs()?;
        if configs.iter().filter(|c| c.players() == 3).count() != 2
            || configs.iter().filter(|c| c.players() == 4).count() != 2
        {
            return Err("Mixed spec requires two 3p and two 4p members".into());
        }
        for (c, r) in configs.iter().zip(&s.rosters) {
            if r.len() != c.players() || r.iter().filter(|p| p.is_none()).count() != 1 {
                return Err("Mixed spec absolute roster/learner coverage differs".into());
            }
        }
        Ok(s)
    }
    fn configs(&self) -> Result<[NativeStochasticConfig; 4], String> {
        self.configurations
            .iter()
            .map(Config::checked_config)
            .collect::<Result<Vec<_>, _>>()?
            .try_into()
            .map_err(|_| "Mixed spec configuration shape".into())
    }
}
/// Borrows the same immutable qualified BC and genuine parent for all four
/// worlds and all audits. Frozen Search/weights and source identities are owned.
pub struct MixedRlCollectionSession<'a> {
    spec: MixedRlCollectionSpec,
    learner: MixedRlLearner<'a>,
    prepared: &'a PreparedPublicPolicy,
    parent: PairedRlParent<'a>,
    search: PreparedSearch,
}
impl<'a> MixedRlCollectionSession<'a> {
    pub fn new(
        spec: MixedRlCollectionSpec,
        learner: PairedRlParent<'a>,
        prepared: &'a PreparedPublicPolicy,
    ) -> Result<Self, String> {
        let search = PreparedSearch::new(&spec.search)?;
        Ok(Self {
            learner: MixedRlLearner::new(learner)?,
            parent: learner,
            prepared,
            search,
            spec,
        })
    }
    fn rosters(&self) -> Result<[FrozenMixedRoster<'_>; 4], String> {
        self.spec
            .rosters
            .iter()
            .map(|seats| {
                FrozenMixedRoster::new(
                    seats
                        .iter()
                        .map(|s| {
                            s.as_ref().map(|p| match p {
                                OpponentSpec::Heuristic => {
                                    BorrowedArenaPolicy::Heuristic(&self.spec.weights)
                                }
                                OpponentSpec::CornFirstSetup => {
                                    BorrowedArenaPolicy::CornFirstSetup(&self.spec.weights)
                                }
                                OpponentSpec::CornFirstUxmalOpening => {
                                    BorrowedArenaPolicy::CornFirstUxmalOpening(&self.spec.weights)
                                }
                                OpponentSpec::PreparedBc => {
                                    BorrowedArenaPolicy::PublicBc(self.prepared)
                                }
                                OpponentSpec::ParentArgmax => match self.parent {
                                    PairedRlParent::Count1(p) => BorrowedArenaPolicy::Count1(p),
                                    PairedRlParent::Paired(p) => BorrowedArenaPolicy::Paired(p),
                                },
                                OpponentSpec::Search => BorrowedArenaPolicy::Search(&self.search),
                            })
                        })
                        .collect(),
                )
            })
            .collect::<Result<Vec<_>, _>>()?
            .try_into()
            .map_err(|_| "Mixed roster shape".into())
    }
    fn plan(&self, rosters: &[FrozenMixedRoster<'_>; 4]) -> Result<PlannedMixedRlCohort, String> {
        PlannedMixedRlCohort::new(&self.learner, self.spec.configs()?, rosters.each_ref())
    }
    /// Save the entire plan before any world. Each started receipt remains an
    /// unresolved/Unknown attempt if killed before its returned receipt. Every
    /// fixed member is retained, including failed prefixes; no success subset.
    pub fn collect_new(self, output: &Path) -> Result<MixedRlCollectionReport, String> {
        let rosters = self.rosters()?;
        let plan = self.plan(&rosters)?;
        let mut writer = Writer::new(output, self.spec.max_output_bytes)?;
        let plan_pin = writer.save("plan.json", &plan.receipt())?;
        let mut records = Vec::new();
        let started = Instant::now();
        for (member, roster) in rosters.iter().enumerate() {
            writer.save(&format!("collect-started-{member}.json"),&serde_json::json!({"schema":"tzolkin-mixed-rl-collect-started-v1","member":member,"planSha256":plan_pin.sha256,"unknownUntilReturned":true}))?;
            let began = Instant::now();
            let game =
                collect_mixed_rl_native(&plan.configurations()[member], &self.learner, roster)?;
            let raw = encode_mixed_rl_record(&game)?;
            let pin = writer.raw(&format!("record-{member}.json"), &raw, MAX_RECORD_BYTES)?;
            let report = MemberReport {
                member,
                record: pin,
                complete: game.complete(),
                observed_callbacks: game.observed_callbacks(),
                learner_callbacks: game.learner_callbacks(),
                learner_candidate_rows: game.learner_candidate_rows(),
                opponent_attempts: game.opponent_attempts(),
                seconds: began.elapsed().as_secs_f64(),
            };
            writer.save(&format!("collect-returned-{member}.json"), &report)?;
            records.push(report);
        }
        let report = MixedRlCollectionReport {
            schema: "tzolkin-mixed-rl-collected-batch-v1".into(),
            plan: plan_pin,
            records,
            seconds: started.elapsed().as_secs_f64(),
            training_data_admission: false,
            audits_required: 4,
            updated_owners: 0,
            recollection_available: false,
        };
        writer.save("collection.json", &report)?;
        Ok(report)
    }
    /// Read all four externally pinned saved records, then fresh-audit all four
    /// against the same actual core/learner/opponents. Never collect or write to
    /// the source directory. Only the complete whole returns consumable steps.
    pub fn audit_saved(
        &self,
        source: &Path,
        collection_raw: &[u8],
        expected_sha256: &str,
        output: &Path,
    ) -> Result<ValidatedMixedRlCohort, String> {
        require_separate_audit_output(source, output)?;
        if sha(collection_raw) != expected_sha256 {
            return Err("Mixed collection external raw pin differs".into());
        }
        let value = unique_json(collection_raw, RECEIPT_CAP)?;
        let receipt: MixedRlCollectionReport =
            serde_json::from_value(value).map_err(|e| e.to_string())?;
        if receipt.schema != "tzolkin-mixed-rl-collected-batch-v1"
            || receipt.records.len() != 4
            || receipt.plan.name != "plan.json"
            || receipt.training_data_admission
            || receipt.audits_required != 4
            || receipt.updated_owners != 0
            || receipt.recollection_available
        {
            return Err("Mixed saved collection contract differs".into());
        }
        let rosters = self.rosters()?;
        let plan = self.plan(&rosters)?;
        let saved_plan = read_pin(source, &receipt.plan, RECEIPT_CAP)?;
        if unique_json(&saved_plan, RECEIPT_CAP)? != plan.receipt() {
            return Err("Mixed saved full plan differs from genuine current source".into());
        }
        let raw = receipt
            .records
            .iter()
            .enumerate()
            .map(|(i, m)| {
                if m.member != i || m.record.name != format!("record-{i}.json") {
                    return Err("Mixed saved member ordering differs".into());
                }
                read_pin(source, &m.record, MAX_RECORD_BYTES)
            })
            .collect::<Result<Vec<_>, String>>()?;
        let before = collection_inventory(source, self.spec.max_output_bytes)?;
        if read_bounded(&source.join("collection.json"), RECEIPT_CAP)? != collection_raw {
            return Err("Mixed saved collection bytes differ from pinned manifest".into());
        }
        for (member, m) in receipt.records.iter().enumerate() {
            let returned = unique_json(
                &read_bounded(
                    &source.join(format!("collect-returned-{member}.json")),
                    RECEIPT_CAP,
                )?,
                RECEIPT_CAP,
            )?;
            let started = unique_json(
                &read_bounded(
                    &source.join(format!("collect-started-{member}.json")),
                    RECEIPT_CAP,
                )?,
                RECEIPT_CAP,
            )?;
            if returned != serde_json::to_value(m).map_err(|e| e.to_string())?
                || started
                    != serde_json::json!({"schema":"tzolkin-mixed-rl-collect-started-v1","member":member,"planSha256":receipt.plan.sha256,"unknownUntilReturned":true})
            {
                return Err("Mixed saved started/returned receipt differs".into());
            }
        }
        let mut writer = Writer::new(output, METADATA_RESERVE)?;
        writer.save("plan.json", &plan.receipt())?;
        let mut results = Vec::new();
        for (member, (bytes, roster)) in raw.iter().zip(&rosters).enumerate() {
            writer.save(&format!("audit-started-{member}.json"),&serde_json::json!({"schema":"tzolkin-mixed-rl-audit-started-v1","member":member,"record":receipt.records[member].record,"unknownUntilReturned":true}))?;
            let began = Instant::now();
            let result =
                audit_mixed_rl_record_bytes(bytes, &self.learner, roster).and_then(|game| {
                    let m = &receipt.records[member];
                    if game.complete() != m.complete
                        || game.observed_callbacks() != m.observed_callbacks
                        || game.learner_callbacks() != m.learner_callbacks
                        || game.learner_candidate_rows() != m.learner_candidate_rows
                        || game.opponent_attempts() != m.opponent_attempts
                    {
                        return Err("Mixed fresh source counters differ from saved return".into());
                    }
                    ValidatedMixedRlEpisode::from_audited(game)
                });
            writer.save(&format!("audit-returned-{member}.json"),&serde_json::json!({"schema":"tzolkin-mixed-rl-audit-returned-v1","member":member,"passed":result.is_ok(),"error":result.as_ref().err(),"seconds":began.elapsed().as_secs_f64()}))?;
            results.push(result);
        }
        if collection_inventory(source, self.spec.max_output_bytes)? != before {
            return Err("Mixed saved source changed during fresh audit".into());
        }
        let cohort = plan.finish(
            results
                .try_into()
                .map_err(|_| "Mixed audited result shape")?,
        )?;
        writer.save("cohort.json", &cohort.receipt())?;
        Ok(cohort)
    }
}
#[derive(Clone, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct FilePin {
    name: String,
    bytes: usize,
    sha256: String,
}
fn expected_collection_names() -> Vec<String> {
    let mut expected = vec!["plan.json".to_owned(), "collection.json".to_owned()];
    for member in 0..4 {
        for stem in ["collect-started", "collect-returned", "record"] {
            expected.push(format!("{stem}-{member}.json"));
        }
    }
    expected.sort();
    expected
}
fn bounded_inventory_names(
    entries: impl Iterator<Item = Result<std::ffi::OsString, String>>,
) -> Result<Vec<String>, String> {
    let expected = expected_collection_names();
    let mut names = Vec::with_capacity(expected.len());
    for entry in entries {
        if names.len() == expected.len() {
            return Err("Mixed saved source exceeds fourteen outputs".into());
        }
        let entry = entry?;
        let name = entry.to_str().ok_or("Mixed saved name is not UTF8")?;
        if !expected.iter().any(|allowed| allowed == name) {
            return Err("Mixed saved source has an unreferenced output".into());
        }
        names.push(name.to_owned());
    }
    names.sort();
    if names != expected {
        return Err("Mixed saved source has missing/unreferenced outputs".into());
    }
    Ok(names)
}
fn collection_inventory(dir: &Path, cap: u64) -> Result<Vec<FilePin>, String> {
    let entries = fs::read_dir(dir)
        .map_err(|e| e.to_string())?
        .map(|entry| entry.map(|e| e.file_name()).map_err(|e| e.to_string()));
    let names = bounded_inventory_names(entries)?;
    let mut total = 0u64;
    names
        .into_iter()
        .map(|name| {
            let limit = if name.starts_with("record-") {
                MAX_RECORD_BYTES
            } else {
                RECEIPT_CAP
            };
            let raw = read_bounded(&dir.join(&name), limit)?;
            total = total
                .checked_add(raw.len() as u64)
                .ok_or("Mixed saved inventory overflow")?;
            if total > cap {
                return Err("Mixed saved inventory content cap".into());
            }
            Ok(FilePin {
                name,
                bytes: raw.len(),
                sha256: sha(&raw),
            })
        })
        .collect()
}
fn require_separate_audit_output(source: &Path, output: &Path) -> Result<(), String> {
    let source = fs::canonicalize(source).map_err(|e| e.to_string())?;
    let resolved_output = match fs::canonicalize(output) {
        Ok(resolved) => resolved,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            let leaf = output
                .file_name()
                .ok_or("Mixed audit output needs a leaf")?;
            let parent = output
                .parent()
                .filter(|p| !p.as_os_str().is_empty())
                .unwrap_or_else(|| Path::new("."));
            fs::canonicalize(parent)
                .map_err(|e| e.to_string())?
                .join(leaf)
        }
        Err(e) => return Err(e.to_string()),
    };
    if resolved_output.starts_with(source) {
        return Err("Mixed audit output must remain outside the saved source directory".into());
    }
    Ok(())
}
#[derive(Deserialize, Serialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct MemberReport {
    member: usize,
    record: FilePin,
    complete: bool,
    observed_callbacks: usize,
    learner_callbacks: usize,
    learner_candidate_rows: usize,
    opponent_attempts: usize,
    seconds: f64,
}
#[derive(Deserialize, Serialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct MixedRlCollectionReport {
    schema: String,
    plan: FilePin,
    records: Vec<MemberReport>,
    seconds: f64,
    training_data_admission: bool,
    audits_required: usize,
    updated_owners: usize,
    recollection_available: bool,
}
fn read_pin(dir: &Path, pin: &FilePin, cap: usize) -> Result<Vec<u8>, String> {
    if pin.name.is_empty()
        || pin.name.len() > 64
        || !pin
            .name
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b == b'-' || b == b'.')
        || pin.bytes > cap
    {
        return Err("Mixed saved leaf/count bound".into());
    }
    let raw = read_bounded(&dir.join(&pin.name), cap)?;
    if raw.len() != pin.bytes || sha(&raw) != pin.sha256 {
        return Err("Mixed saved record raw bytes/SHA differ".into());
    }
    Ok(raw)
}
struct Writer<'a> {
    dir: &'a Path,
    bytes: u64,
    cap: u64,
    files: usize,
}
impl<'a> Writer<'a> {
    fn new(dir: &'a Path, cap: u64) -> Result<Self, String> {
        fs::create_dir(dir).map_err(|e| e.to_string())?;
        Ok(Self {
            dir,
            bytes: 0,
            cap,
            files: 0,
        })
    }
    fn raw(&mut self, name: &str, raw: &[u8], per_file: usize) -> Result<FilePin, String> {
        use std::io::Write;
        let total = self
            .bytes
            .checked_add(raw.len() as u64)
            .ok_or("Mixed output byte overflow")?;
        if raw.len() > per_file || total > self.cap || self.files >= 16 {
            return Err("Mixed output content/count cap".into());
        }
        let mut file = fs::OpenOptions::new()
            .create_new(true)
            .write(true)
            .open(self.dir.join(name))
            .map_err(|e| e.to_string())?;
        file.write_all(raw)
            .and_then(|_| file.sync_all())
            .map_err(|e| e.to_string())?;
        self.bytes = total;
        self.files += 1;
        Ok(FilePin {
            name: name.into(),
            bytes: raw.len(),
            sha256: sha(raw),
        })
    }
    fn save(&mut self, name: &str, value: &impl Serialize) -> Result<FilePin, String> {
        self.raw(
            name,
            &serde_json::to_vec(value).map_err(|e| e.to_string())?,
            RECEIPT_CAP,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::Cell;
    use std::ffi::OsString;

    #[test]
    fn saved_inventory_rejects_unexpected_or_fifteenth_entry_without_draining() {
        let visited = Cell::new(0);
        let unexpected = std::iter::from_fn(|| {
            let n = visited.get() + 1;
            visited.set(n);
            assert_eq!(n, 1, "must stop at first unreferenced entry");
            Some(Ok(OsString::from("unreferenced.json")))
        });
        assert!(bounded_inventory_names(unexpected).is_err());
        assert_eq!(visited.get(), 1);
        visited.set(0);
        let excessive = std::iter::from_fn(|| {
            let n = visited.get() + 1;
            visited.set(n);
            assert!(n <= 15, "must stop at the fifteenth directory entry");
            Some(Ok(OsString::from("plan.json")))
        });
        assert!(bounded_inventory_names(excessive).is_err());
        assert_eq!(visited.get(), 15);
        let valid = expected_collection_names()
            .into_iter()
            .map(|n| Ok(n.into()));
        assert_eq!(bounded_inventory_names(valid).unwrap().len(), 14);
    }

    #[test]
    fn audit_output_rejects_saved_source_descendants_and_canonical_parent_aliases() {
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let directory = std::env::temp_dir().join(format!(
            "tzolkin-mixed-output-{}-{nonce}",
            std::process::id()
        ));
        fs::create_dir(&directory).unwrap();
        let source = directory.join("saved");
        fs::create_dir(&source).unwrap();
        assert!(require_separate_audit_output(&source, &source).is_err());
        assert!(require_separate_audit_output(&source, &source.join("audit")).is_err());
        let alias = source.join("..").join("saved").join("audit");
        assert!(require_separate_audit_output(&source, &alias).is_err());
        assert!(require_separate_audit_output(&source, &directory.join("audit")).is_ok());
        assert_eq!(fs::read_dir(&source).unwrap().count(), 0);
        assert!(!directory.join("audit").exists());
        fs::remove_dir(&source).unwrap();
        fs::remove_dir(&directory).unwrap();
    }
}
