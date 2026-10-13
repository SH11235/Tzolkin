//! Saved whole-four mixed joint update with a bounded single-writer journal.
//! Content consistency is not producer authentication. Continuation is unavailable.
use crate::dataset::{DatasetSplit, seed_family_id, split_for_family};
use crate::public_native::PreparedPublicPolicy;
use crate::public_policy_baseline::ResidualFitConfig;
use crate::public_policy_mixed::{
    self, MixedJointArtifact, MixedJointOwner, MixedJointReport, MixedProposal, MixedRoot,
};
use crate::public_policy_update::{OneStepConfig, checkpoint_digest, hash};
use crate::public_rl_artifact::InitializedPublicRlPolicy;
use crate::public_rl_mixed_session::{
    ImportedMixedRlParent, MixedRlCollectionSpec, require_output_outside_directory,
};
use crate::public_rl_paired_session::{local, publish_mixed_metadata, read_bounded};
use crate::public_stochastic_record::{Config, unique_json};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeSet,
    fs,
    path::{Path, PathBuf},
};

pub const SPEC_SCHEMA: &str = "tzolkin-mixed-saved-one-batch-spec-v1";
pub const JOURNAL_SCHEMA: &str = "tzolkin-mixed-saved-one-batch-journal-v1";
pub const MAX_SPEC_BYTES: usize = 64 * 1024;
pub const MAX_HEAD_BYTES: usize = 2 * 1024 * 1024;
const ARTIFACT_BYTES: usize = 16 * 1024 * 1024;
const MAX_ENTRIES: u64 = 6;
fn sha(raw: &[u8]) -> String {
    format!("{:x}", Sha256::digest(raw))
}
fn bytes(v: &impl Serialize, cap: usize) -> Result<Vec<u8>, String> {
    let b = serde_json::to_vec(v).map_err(|e| e.to_string())?;
    if b.len() > cap {
        return Err("Mixed metadata byte cap".into());
    }
    Ok(b)
}
fn closed<T: serde::de::DeserializeOwned>(raw: &[u8], cap: usize) -> Result<T, String> {
    serde_json::from_value(unique_json(raw, cap)?).map_err(|e| e.to_string())
}

#[derive(Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct MixedOneBatchSpec {
    schema: String,
    actor: serde_json::Value,
    critic: serde_json::Value,
    residual_seed: u64,
    max_output_bytes: u64,
    excluded_families: Vec<String>,
}
impl MixedOneBatchSpec {
    pub fn from_json(raw: &[u8]) -> Result<Self, String> {
        let s: Self = closed(raw, MAX_SPEC_BYTES)?;
        s.validate()?;
        Ok(s)
    }
    fn validate(&self) -> Result<(), String> {
        self.actor()?;
        self.critic()?;
        if self.schema != SPEC_SCHEMA
            || !(32 * 1024 * 1024..=64 * 1024 * 1024).contains(&self.max_output_bytes)
            || self.excluded_families.len() > 512
            || self.excluded_families.iter().any(|s| !checkpoint_digest(s))
            || self.excluded_families.windows(2).any(|p| p[0] >= p[1])
        {
            return Err("Invalid bounded mixed one-batch spec".into());
        }
        Ok(())
    }
    fn actor(&self) -> Result<OneStepConfig, String> {
        OneStepConfig::from_checkpoint(self.actor.clone())
    }
    fn critic(&self) -> Result<ResidualFitConfig, String> {
        ResidualFitConfig::from_checkpoint(self.critic.clone())
    }
}
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct Pin {
    name: String,
    bytes: usize,
    sha256: String,
}
fn pin(name: &str, b: &[u8]) -> Pin {
    Pin {
        name: name.into(),
        bytes: b.len(),
        sha256: sha(b),
    }
}
fn read_pin(dir: &Path, p: &Pin, name: &str, cap: usize) -> Result<Vec<u8>, String> {
    if p.name != name || p.bytes == 0 || p.bytes > cap || !checkpoint_digest(&p.sha256) {
        return Err("Mixed journal file pin shape".into());
    }
    let b = read_bounded(&dir.join(name), p.bytes)?;
    if b.len() != p.bytes || sha(&b) != p.sha256 {
        return Err("Mixed journal file bytes/SHA mismatch".into());
    }
    Ok(b)
}
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
enum Phase {
    Initial,
    Reserved,
    Auditing,
    Audited,
    Updating,
    Resolved,
}
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct Progress {
    phase: Phase,
    next_ordinal: u64,
    consumed_families: Vec<String>,
    plan: Option<serde_json::Value>,
    source_directory: Option<PathBuf>,
    source_spec: Option<Pin>,
    source_receipt: Option<Pin>,
    source_inventory: Option<serde_json::Value>,
    audit: Option<Pin>,
    audit_inventory: Vec<Pin>,
    outcome: Option<String>,
    report: Option<Pin>,
    owner: Option<Pin>,
    error: Option<String>,
    unknown_work: bool,
}
#[derive(Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "camelCase", deny_unknown_fields)]
enum Event {
    Start,
    AdoptSaved {
        plan: serde_json::Value,
        source_directory: PathBuf,
        source_inventory: serde_json::Value,
        spec: Pin,
        receipt: Pin,
    },
    AuditStart,
    AuditReturn {
        cohort: Pin,
        inventory: Vec<Pin>,
    },
    GradientStart,
    Resolve {
        outcome: String,
        report: Option<Pin>,
        owner: Option<Pin>,
        error: Option<String>,
    },
}
#[derive(Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct Entry {
    schema: String,
    sequence: u64,
    previous: String,
    spec_checksum: String,
    root: MixedRoot,
    sampling_prefix: [u64; 3],
    initial_consumed: Vec<String>,
    event: Event,
    progress: Progress,
}
fn reserve_families(
    plan: &serde_json::Value,
    prefix: [u64; 3],
    next: u64,
    used: &[String],
) -> Result<[String; 4], String> {
    if plan.get("schema").and_then(|v| v.as_str()) != Some("tzolkin-planned-mixed-rl-cohort-v1")
        || !plan
            .get("checksum")
            .and_then(|v| v.as_str())
            .is_some_and(checkpoint_digest)
    {
        return Err("Wrong mixed reservation plan".into());
    }
    let headers = plan
        .get("headers")
        .and_then(|v| v.as_array())
        .ok_or("Missing mixed planned headers")?;
    if headers.len() != 4 {
        return Err("Mixed reservation needs four headers".into());
    }
    let mut families = Vec::new();
    for (i, h) in headers.iter().enumerate() {
        let c: Config =
            serde_json::from_value(h.get("config").ok_or("Mixed reservation config")?.clone())
                .map_err(|e| e.to_string())?;
        c.checked_config()?;
        let ordinal = next
            .checked_add(i as u64)
            .ok_or("Mixed reserved ordinal overflow")?;
        let f = seed_family_id(c.environment_seed);
        if c.sampling_seed != format!("{:016x}", prefix[0])
            || c.replicate_ordinal != format!("{:016x}", prefix[1])
            || c.episode_ordinal != format!("{ordinal:016x}")
            || used.contains(&f)
            || families.contains(&f)
            || split_for_family(&f)? != DatasetSplit::Train
        {
            return Err(
                "Mixed reservation repeats family/ordinal/RNG or resets imported prefix".into(),
            );
        }
        families.push(f);
    }
    families
        .try_into()
        .map_err(|_| "Mixed families shape".into())
}
fn apply(p: &mut Progress, e: &Event, prefix: [u64; 3]) -> Result<(), String> {
    match e {
        Event::Start if p.phase == Phase::Initial => {}
        Event::AdoptSaved {
            plan,
            source_directory,
            source_inventory,
            spec,
            receipt,
        } if p.phase == Phase::Initial => {
            let families = reserve_families(plan, prefix, p.next_ordinal, &p.consumed_families)?;
            p.next_ordinal = p
                .next_ordinal
                .checked_add(4)
                .ok_or("Mixed reservation ordinal overflow")?;
            let mut all = p.consumed_families.iter().cloned().collect::<BTreeSet<_>>();
            all.extend(families);
            p.consumed_families = all.into_iter().collect();
            p.plan = Some(plan.clone());
            p.source_directory = Some(source_directory.clone());
            p.source_inventory = Some(source_inventory.clone());
            p.source_spec = Some(spec.clone());
            p.source_receipt = Some(receipt.clone());
            p.phase = Phase::Reserved;
        }
        Event::AuditStart if p.phase == Phase::Reserved => {
            p.phase = Phase::Auditing;
            p.unknown_work = true;
        }
        Event::AuditReturn { cohort, inventory } if p.phase == Phase::Auditing => {
            if inventory.iter().map(|p| p.name.clone()).collect::<Vec<_>>() != audit_names()
                || !inventory.contains(cohort)
                || cohort.name != "audit/cohort.json"
                || inventory.iter().any(|p| {
                    p.bytes == 0 || p.bytes > MAX_SPEC_BYTES || !checkpoint_digest(&p.sha256)
                })
            {
                return Err("Mixed audit must bind the complete returned inventory".into());
            }
            p.audit = Some(cohort.clone());
            p.audit_inventory = inventory.clone();
            p.phase = Phase::Audited;
            p.unknown_work = false;
        }
        Event::GradientStart if p.phase == Phase::Audited => {
            p.phase = Phase::Updating;
            p.unknown_work = true;
        }
        Event::Resolve {
            outcome,
            report,
            owner,
            error,
        } => {
            let success = matches!(outcome.as_str(), "updated" | "noChange");
            if matches!(p.phase, Phase::Initial | Phase::Resolved)
                || (success
                    && (p.phase != Phase::Updating
                        || p.audit.is_none()
                        || report.is_none()
                        || error.is_some()))
                || (outcome == "updated") != owner.is_some()
                || (!success
                    && (!matches!(outcome.as_str(), "failed" | "abandoned")
                        || report.is_some()
                        || owner.is_some()
                        || error.is_none()))
                || error
                    .as_ref()
                    .is_some_and(|s| s.is_empty() || s.chars().count() > 1024)
            {
                return Err("Invalid mixed joint resolution; no partial promotion/retry".into());
            }
            p.phase = Phase::Resolved;
            p.outcome = Some(outcome.clone());
            p.report = report.clone();
            p.owner = owner.clone();
            p.error = error.clone();
            if success {
                p.unknown_work = false;
            }
        }
        _ => return Err("Mixed one-batch phase already used or unresolved".into()),
    }
    Ok(())
}

/// Genuine imported owners and journal replay alone mint this session/owner.
/// One batch only: restoring pending work never retries audit or optimizer.
pub struct MixedOneBatchSession<'a> {
    parent: ImportedMixedRlParent,
    prepared: &'a PreparedPublicPolicy,
    spec: MixedOneBatchSpec,
    root: MixedRoot,
    residual: Vec<f32>,
    prefix: [u64; 3],
    initial_consumed: Vec<String>,
    progress: Progress,
    directory: PathBuf,
    sequence: u64,
    tip: String,
    owner: Option<MixedJointOwner>,
    restored_source: Option<RestoredSourceBinding>,
}
// Constructed only after the entire externally pinned original journal has
// restored successfully. No wire description can import this capability.
struct RestoredSourceBinding {
    spec: MixedRlCollectionSpec,
    plan: serde_json::Value,
    directory: PathBuf,
    receipt: Pin,
    inventory: serde_json::Value,
    heads: Vec<Pin>,
    pins: Vec<(Pin, usize)>,
}
impl<'a> MixedOneBatchSession<'a> {
    fn initial(
        parent: ImportedMixedRlParent,
        prepared: &'a PreparedPublicPolicy,
        spec: MixedOneBatchSpec,
        directory: PathBuf,
    ) -> Result<Self, String> {
        spec.validate()?;
        let initial = InitializedPublicRlPolicy::from_bc(prepared)?;
        let (root, residual) = public_policy_mixed::root(&parent, &initial, spec.residual_seed)?;
        let (a, b, c) = parent.sampling_prefix();
        c.checked_add(4).ok_or("Mixed next ordinal overflow")?;
        let mut used = parent.reserved_families()?;
        used.extend(spec.excluded_families.iter().cloned());
        let initial_consumed = used.into_iter().collect::<Vec<_>>();
        let progress = Progress {
            phase: Phase::Initial,
            next_ordinal: c,
            consumed_families: initial_consumed.clone(),
            plan: None,
            source_directory: None,
            source_spec: None,
            source_receipt: None,
            source_inventory: None,
            audit: None,
            audit_inventory: Vec::new(),
            outcome: None,
            report: None,
            owner: None,
            error: None,
            unknown_work: false,
        };
        Ok(Self {
            parent,
            prepared,
            spec,
            root,
            residual,
            prefix: [a, b, c],
            initial_consumed,
            progress,
            directory,
            sequence: 0,
            tip: String::new(),
            owner: None,
            restored_source: None,
        })
    }
    pub fn from_imported(
        parent: ImportedMixedRlParent,
        prepared: &'a PreparedPublicPolicy,
        spec: MixedOneBatchSpec,
        output: &Path,
    ) -> Result<Self, String> {
        if let Some(source) = parent.protected_directory() {
            require_output_outside_directory(source, output)?;
        }
        let directory = local(output, true)?;
        if directory.exists() {
            return Err("Mixed owner output must be new".into());
        }
        let mut s = Self::initial(parent, prepared, spec, directory)?;
        // Check bounded root/spec before any output is created.
        bytes(&s.root, MAX_HEAD_BYTES)?;
        bytes(&s.spec, MAX_SPEC_BYTES)?;
        fs::create_dir(&s.directory).map_err(|e| e.to_string())?;
        s.publish(
            "training-spec.json",
            &bytes(&s.spec, MAX_SPEC_BYTES)?,
            MAX_SPEC_BYTES,
        )?;
        s.save(Event::Start)?;
        Ok(s)
    }
    fn spec_checksum(&self) -> Result<String, String> {
        hash(b"tzolkin-mixed-one-batch-spec-v1\0", &self.spec)
    }
    fn usage(&self) -> Result<u64, String> {
        let mut total = 0u64;
        let mut count = 0usize;
        for entry in fs::read_dir(&self.directory).map_err(|e| e.to_string())? {
            let entry = entry.map_err(|e| e.to_string())?;
            count += 1;
            if count > 20 {
                return Err("Mixed owner file count cap".into());
            }
            let meta = fs::symlink_metadata(entry.path()).map_err(|e| e.to_string())?;
            if meta.is_symlink() {
                return Err("Mixed owner rejects links".into());
            }
            if meta.is_file() {
                total = total.checked_add(meta.len()).ok_or("Mixed byte overflow")?;
            } else if meta.is_dir() && entry.file_name() == "audit" {
                let mut n = 0;
                for item in fs::read_dir(entry.path()).map_err(|e| e.to_string())? {
                    n += 1;
                    if n > 10 {
                        return Err("Mixed audit output count cap".into());
                    }
                    let item = item.map_err(|e| e.to_string())?;
                    let m = fs::symlink_metadata(item.path()).map_err(|e| e.to_string())?;
                    if !m.is_file() || m.is_symlink() {
                        return Err("Mixed audit output is not regular".into());
                    }
                    total = total.checked_add(m.len()).ok_or("Mixed byte overflow")?;
                }
            } else {
                return Err("Unexpected mixed owner directory entry".into());
            }
        }
        if total > self.spec.max_output_bytes {
            return Err("Mixed total output byte cap".into());
        }
        Ok(total)
    }
    fn publish(&self, name: &str, raw: &[u8], cap: usize) -> Result<Pin, String> {
        if raw.is_empty()
            || raw.len() > cap
            || self
                .usage()?
                .checked_add(raw.len() as u64)
                .is_none_or(|n| n > self.spec.max_output_bytes)
        {
            return Err("Mixed publication storage cap".into());
        }
        publish_mixed_metadata(&self.directory, name, raw)?;
        let p = pin(name, raw);
        read_pin(&self.directory, &p, name, cap)?;
        Ok(p)
    }
    fn save(&mut self, event: Event) -> Result<(), String> {
        let sequence = self
            .sequence
            .checked_add(1)
            .ok_or("Mixed journal sequence overflow")?;
        if sequence > MAX_ENTRIES {
            return Err("Mixed one-batch journal cap".into());
        }
        let mut next = self.progress.clone();
        apply(&mut next, &event, self.prefix)?;
        let entry = Entry {
            schema: JOURNAL_SCHEMA.into(),
            sequence,
            previous: self.tip.clone(),
            spec_checksum: self.spec_checksum()?,
            root: self.root.clone(),
            sampling_prefix: self.prefix,
            initial_consumed: self.initial_consumed.clone(),
            event,
            progress: next.clone(),
        };
        let p = self.publish(
            &format!("head-{sequence}.json"),
            &bytes(&entry, MAX_HEAD_BYTES)?,
            MAX_HEAD_BYTES,
        )?;
        self.progress = next;
        self.sequence = sequence;
        self.tip = p.sha256;
        Ok(())
    }
    /// Adopts prior saved attempts explicitly; this journal did not precede their
    /// original collection. All four identities are reserved before fresh audit.
    pub fn update_saved(
        &mut self,
        spec: MixedRlCollectionSpec,
        source: &Path,
        collection_raw: &[u8],
        expected_sha256: &str,
    ) -> Result<MixedOneBatchStatus, String> {
        if self.progress.phase != Phase::Initial {
            return Err("Mixed one-batch source already reserved; no reuse/retry".into());
        }
        require_output_outside_directory(source, &self.directory)?;
        let source = local(source, false)?;
        if !checkpoint_digest(expected_sha256) || sha(collection_raw) != expected_sha256 {
            return Err("Mixed saved collection external raw pin differs".into());
        }
        unique_json(collection_raw, MAX_SPEC_BYTES)?;
        let plan = self
            .parent
            .collection_session(spec.clone(), self.prepared)?
            .training_plan()?
            .receipt();
        let source_inventory = self
            .parent
            .collection_session(spec.clone(), self.prepared)?
            .verify_saved(&source, collection_raw, expected_sha256)?;
        reserve_families(
            &plan,
            self.prefix,
            self.progress.next_ordinal,
            &self.progress.consumed_families,
        )?;
        let spec_pin = self.publish("collection-spec.json", &spec.to_json()?, MAX_SPEC_BYTES)?;
        let receipt = self.publish("source-receipt.json", collection_raw, MAX_SPEC_BYTES)?;
        self.save(Event::AdoptSaved {
            plan,
            source_directory: source.clone(),
            source_inventory,
            spec: spec_pin,
            receipt,
        })?;
        if let Err(e) = self.require_live_inputs(&spec, &source) {
            self.fail(&e)?;
            return Ok(self.status());
        }
        self.save(Event::AuditStart)?;
        let result = self
            .parent
            .collection_session(spec.clone(), self.prepared)?
            .audit_saved(
                &source,
                collection_raw,
                expected_sha256,
                &self.directory.join("audit"),
            );
        let cohort = match result {
            Ok(c) => c,
            Err(e) => {
                self.fail(&e)?;
                return Ok(self.status());
            }
        };
        if let Err(e) = self.require_live_inputs(&spec, &source) {
            self.fail(&e)?;
            return Ok(self.status());
        }
        let audit_raw = read_bounded(&self.directory.join("audit/cohort.json"), MAX_SPEC_BYTES)?;
        if unique_json(&audit_raw, MAX_SPEC_BYTES)? != cohort.receipt() {
            return Err("Mixed saved audit receipt differs before gradient".into());
        }
        self.save(Event::AuditReturn {
            cohort: pin("audit/cohort.json", &audit_raw),
            inventory: audit_inventory(&self.directory)?,
        })?;
        // Reserve enough disk for the joint artifact/report and all remaining
        // journal entries before forwards. No mathematical retry on failure.
        if self
            .usage()?
            .checked_add(ARTIFACT_BYTES as u64 + 3 * MAX_HEAD_BYTES as u64)
            .is_none_or(|n| n > self.spec.max_output_bytes)
        {
            self.fail("Mixed update output headroom")?;
            return Ok(self.status());
        }
        self.save(Event::GradientStart)?;
        let result = public_policy_mixed::prepare(
            &self.parent,
            self.root.clone(),
            &self.residual,
            &cohort,
            &self.spec.actor()?,
            &self.spec.critic()?,
        );
        match result {
            Err(e) => self.fail(&e)?,
            Ok(proposal) => {
                if let Err(e) = self.require_live_inputs(&spec, &source) {
                    self.fail(&e)?;
                    return Ok(self.status());
                }
                let report = self.publish(
                    "report.json",
                    &bytes(proposal.report(), MAX_HEAD_BYTES)?,
                    MAX_HEAD_BYTES,
                )?;
                match proposal {
                    MixedProposal::NoChange(_) => {
                        if let Err(e) = self.require_live_inputs(&spec, &source) {
                            self.fail(&e)?;
                            return Ok(self.status());
                        }
                        self.save(Event::Resolve {
                            outcome: "noChange".into(),
                            report: Some(report),
                            owner: None,
                            error: None,
                        })?;
                    }
                    MixedProposal::Updated(owner) => {
                        let artifact = self.publish(
                            "artifact.json",
                            &bytes(owner.artifact(), ARTIFACT_BYTES)?,
                            ARTIFACT_BYTES,
                        )?;
                        if let Err(e) = self.require_live_inputs(&spec, &source) {
                            self.fail(&e)?;
                            return Ok(self.status());
                        }
                        self.save(Event::Resolve {
                            outcome: "updated".into(),
                            report: Some(report),
                            owner: Some(artifact),
                            error: None,
                        })?;
                        self.owner = Some(owner);
                    }
                }
            }
        }
        self.usage()?;
        Ok(self.status())
    }
    fn require_live_inputs(
        &self,
        spec: &MixedRlCollectionSpec,
        source: &Path,
    ) -> Result<(), String> {
        let reserved = self
            .progress
            .source_inventory
            .as_ref()
            .and_then(|v| v.get("inventory"))
            .ok_or("Missing reserved mixed source inventory")?;
        let current = self
            .parent
            .collection_session(spec.clone(), self.prepared)?
            .current_saved_inventory(source)?;
        if &current != reserved {
            return Err("Mixed saved source changed after reservation".into());
        }
        for (p, name) in [
            (&self.progress.source_spec, "collection-spec.json"),
            (&self.progress.source_receipt, "source-receipt.json"),
        ] {
            read_pin(
                &self.directory,
                p.as_ref().ok_or("Missing adopted mixed input pin")?,
                name,
                MAX_SPEC_BYTES,
            )?;
        }
        if self.progress.audit.is_some()
            && audit_inventory(&self.directory)? != self.progress.audit_inventory
        {
            return Err("Mixed returned audit files changed before resolution".into());
        }
        Ok(())
    }
    fn fail(&mut self, error: &str) -> Result<(), String> {
        self.save(Event::Resolve {
            outcome: "failed".into(),
            report: None,
            owner: None,
            error: Some(error.chars().take(1024).collect()),
        })
    }
    pub fn abandon(&mut self, reason: &str) -> Result<MixedOneBatchStatus, String> {
        self.save(Event::Resolve {
            outcome: "abandoned".into(),
            report: None,
            owner: None,
            error: Some(reason.into()),
        })?;
        Ok(self.status())
    }
    pub fn owner(&self) -> Option<&MixedJointOwner> {
        self.owner.as_ref()
    }
    pub(crate) fn current_data_directory(&self) -> &Path {
        &self.directory
    }
    pub(crate) fn current_data_context(
        &self,
    ) -> Result<(&'a PreparedPublicPolicy, [u64; 2]), String> {
        if self.progress.phase != Phase::Resolved
            || self.progress.outcome.as_deref() != Some("updated")
            || self.progress.unknown_work
            || self.progress.error.is_some()
            || self.owner.is_none()
        {
            return Err("Current mixed data requires a genuinely resolved updated owner".into());
        }
        self.owner
            .as_ref()
            .expect("checked owner")
            .validate_for_inference()?;
        Ok((self.prepared, [self.prefix[0], self.prefix[1]]))
    }
    pub(crate) fn require_current_data_output(&self, output: &Path) -> Result<(), String> {
        require_output_outside_directory(&self.directory, output)?;
        require_output_outside_directory(
            self.progress
                .source_directory
                .as_deref()
                .ok_or("Missing mixed adopted source directory")?,
            output,
        )?;
        if let Some(parent) = self.parent.protected_directory() {
            require_output_outside_directory(parent, output)?;
        }
        Ok(())
    }
    pub(crate) fn current_data_snapshot(&self) -> Result<serde_json::Value, String> {
        self.current_data_context()?;
        self.require_current_inventory()?;
        self.usage()?;
        let source_spec = read_pin(
            &self.directory,
            self.progress
                .source_spec
                .as_ref()
                .ok_or("Missing mixed collection spec pin")?,
            "collection-spec.json",
            MAX_SPEC_BYTES,
        )?;
        let spec = MixedRlCollectionSpec::from_json(&source_spec)?;
        let source = crate::public_rl_mixed_session::collection_inventory(
            self.progress
                .source_directory
                .as_deref()
                .ok_or("Missing mixed source")?,
            spec.wire.max_output_bytes,
        )?;
        if serde_json::to_value(&source).map_err(|e| e.to_string())?
            != *self
                .progress
                .source_inventory
                .as_ref()
                .and_then(|v| v.get("inventory"))
                .ok_or("Missing reserved mixed source inventory")?
            || audit_inventory(&self.directory)? != self.progress.audit_inventory
        {
            return Err("Current mixed root source/audit inventory changed".into());
        }
        let mut files = Vec::new();
        let mut previous = String::new();
        for sequence in 1..=self.sequence {
            let name = format!("head-{sequence}.json");
            let raw = read_bounded(&self.directory.join(&name), MAX_HEAD_BYTES)?;
            let entry: Entry = closed(&raw, MAX_HEAD_BYTES)?;
            if entry.sequence != sequence || entry.previous != previous {
                return Err("Current mixed root raw journal chain changed".into());
            }
            previous = sha(&raw);
            if sequence == self.sequence && previous != self.tip {
                return Err("Current mixed latest raw head changed".into());
            }
            files.push(pin(&name, &raw));
        }
        let raw = read_bounded(&self.directory.join("training-spec.json"), MAX_SPEC_BYTES)?;
        let stored: MixedOneBatchSpec = closed(&raw, MAX_SPEC_BYTES)?;
        if hash(b"tzolkin-mixed-one-batch-spec-v1\0", &stored)? != self.spec_checksum()? {
            return Err("Current mixed training spec changed".into());
        }
        files.push(pin("training-spec.json", &raw));
        for (p, name, cap) in [
            (
                &self.progress.source_spec,
                "collection-spec.json",
                MAX_SPEC_BYTES,
            ),
            (
                &self.progress.source_receipt,
                "source-receipt.json",
                MAX_SPEC_BYTES,
            ),
            (&self.progress.report, "report.json", MAX_HEAD_BYTES),
            (&self.progress.owner, "artifact.json", ARTIFACT_BYTES),
        ] {
            let p = p.as_ref().ok_or("Missing resolved mixed root file pin")?;
            read_pin(&self.directory, p, name, cap)?;
            files.push(p.clone());
        }
        Ok(
            serde_json::json!({"rootFiles":files,"auditFiles":self.progress.audit_inventory,"sourceFiles":source}),
        )
    }
    pub fn status(&self) -> MixedOneBatchStatus {
        MixedOneBatchStatus {
            schema: JOURNAL_SCHEMA.into(),
            head_file: format!("head-{}.json", self.sequence),
            head_sha256: self.tip.clone(),
            imported_update_count: self.root.parent_count,
            mixed_update_count: u64::from(self.owner.is_some()),
            next_episode_ordinal: self.progress.next_ordinal,
            reserved_families: self.progress.consumed_families.clone(),
            unresolved_work: !matches!(self.progress.phase, Phase::Initial | Phase::Resolved),
            outcome: self.progress.outcome.clone(),
            error: self.progress.error.clone(),
            unknown_work_possible: self.progress.unknown_work,
            continuation_available: false,
        }
    }
    /// Metadata/raw-pin restoration only; no native rerun and no optimizer retry.
    /// The original genuine import and exact current head SHA are mandatory.
    pub fn restore(
        parent: ImportedMixedRlParent,
        prepared: &'a PreparedPublicPolicy,
        spec: MixedOneBatchSpec,
        head_raw: &[u8],
        expected_sha256: &str,
        directory: &Path,
    ) -> Result<Self, String> {
        if !checkpoint_digest(expected_sha256) || sha(head_raw) != expected_sha256 {
            return Err("Mixed head external raw SHA differs".into());
        }
        let supplied: Entry = closed(head_raw, MAX_HEAD_BYTES)?;
        if supplied.sequence == 0 || supplied.sequence > MAX_ENTRIES {
            return Err("Mixed head sequence cap".into());
        }
        if let Some(protected) = parent.protected_directory() {
            require_output_outside_directory(protected, directory)?;
        }
        let mut s = Self::initial(parent, prepared, spec, local(directory, false)?)?;
        let spec_raw = read_bounded(&s.directory.join("training-spec.json"), MAX_SPEC_BYTES)?;
        let stored_spec: MixedOneBatchSpec = closed(&spec_raw, MAX_SPEC_BYTES)?;
        if hash(b"tzolkin-mixed-one-batch-spec-v1\0", &stored_spec)? != s.spec_checksum()? {
            return Err("Mixed full stored spec differs".into());
        }
        let mut source_binding = None;
        let mut heads = Vec::with_capacity(supplied.sequence as usize);
        let mut pins = vec![(pin("training-spec.json", &spec_raw), MAX_SPEC_BYTES)];
        for sequence in 1..=supplied.sequence {
            let raw = read_bounded(
                &s.directory.join(format!("head-{sequence}.json")),
                MAX_HEAD_BYTES,
            )?;
            let entry: Entry = closed(&raw, MAX_HEAD_BYTES)?;
            if entry.schema != JOURNAL_SCHEMA
                || entry.sequence != sequence
                || entry.previous != s.tip
                || entry.spec_checksum != s.spec_checksum()?
                || entry.root != s.root
                || entry.sampling_prefix != s.prefix
                || entry.initial_consumed != s.initial_consumed
                || matches!(entry.event, Event::Start) != (sequence == 1)
            {
                return Err("Mixed journal root/spec/chain mismatch".into());
            }
            match &entry.event {
                Event::AdoptSaved {
                    plan,
                    source_directory,
                    source_inventory,
                    spec,
                    receipt,
                } => {
                    require_output_outside_directory(source_directory, &s.directory)?;
                    pins.push((spec.clone(), MAX_SPEC_BYTES));
                    pins.push((receipt.clone(), MAX_SPEC_BYTES));
                    let spec = MixedRlCollectionSpec::from_json(&read_pin(
                        &s.directory,
                        spec,
                        "collection-spec.json",
                        MAX_SPEC_BYTES,
                    )?)?;
                    let source_raw =
                        read_pin(&s.directory, receipt, "source-receipt.json", MAX_SPEC_BYTES)?;
                    let session = s
                        .parent
                        .collection_session(spec.clone(), s.prepared)?
                        .for_restored_plan(plan)?;
                    if session.training_plan()?.receipt() != *plan
                        || session.verify_saved(source_directory, &source_raw, &receipt.sha256)?
                            != *source_inventory
                    {
                        return Err(
                            "Mixed restored source plan or full saved raw inventory differs".into(),
                        );
                    }
                    source_binding = Some(RestoredSourceBinding {
                        spec,
                        plan: plan.clone(),
                        directory: source_directory.clone(),
                        receipt: receipt.clone(),
                        inventory: source_inventory.clone(),
                        heads: Vec::new(),
                        pins: Vec::new(),
                    });
                }
                Event::AuditReturn { cohort, inventory } => {
                    if audit_inventory(&s.directory)? != *inventory {
                        return Err("Mixed complete audit output raw pins differ".into());
                    }
                    let v = unique_json(
                        &read_pin(&s.directory, cohort, "audit/cohort.json", MAX_SPEC_BYTES)?,
                        MAX_SPEC_BYTES,
                    )?;
                    check_audit(
                        &v,
                        s.progress.plan.as_ref().ok_or("Mixed audit missing plan")?,
                    )?;
                    pins.extend(inventory.iter().cloned().map(|p| (p, MAX_SPEC_BYTES)));
                }
                Event::Resolve {
                    outcome,
                    report,
                    owner,
                    ..
                } if matches!(outcome.as_str(), "updated" | "noChange") => {
                    let r: MixedJointReport = closed(
                        &read_pin(
                            &s.directory,
                            report.as_ref().ok_or("Mixed missing report")?,
                            "report.json",
                            MAX_HEAD_BYTES,
                        )?,
                        MAX_HEAD_BYTES,
                    )?;
                    r.validate(&s.root, &s.spec.actor()?, &s.spec.critic()?)?;
                    let audit = unique_json(
                        &read_pin(
                            &s.directory,
                            s.progress.audit.as_ref().ok_or("Mixed missing audit")?,
                            "audit/cohort.json",
                            MAX_SPEC_BYTES,
                        )?,
                        MAX_SPEC_BYTES,
                    )?;
                    check_report(
                        &r,
                        s.progress.plan.as_ref().ok_or("Mixed missing plan")?,
                        &audit,
                    )?;
                    if let Some(p) = owner {
                        let a: MixedJointArtifact = closed(
                            &read_pin(&s.directory, p, "artifact.json", ARTIFACT_BYTES)?,
                            ARTIFACT_BYTES,
                        )?;
                        if a.report().checksum() != r.checksum() {
                            return Err("Mixed joint artifact/report differs".into());
                        }
                        s.owner = Some(public_policy_mixed::restore_owner(
                            a,
                            &s.root,
                            &s.spec.actor()?,
                            &s.spec.critic()?,
                            public_policy_mixed::parent_model(&s.parent),
                            &s.residual,
                        )?);
                        pins.push((p.clone(), ARTIFACT_BYTES));
                    } else if r.changed_parameters() != [0, 0] {
                        return Err("Mixed NoChange report contains an update".into());
                    }
                    pins.push((
                        report.as_ref().ok_or("Mixed missing report")?.clone(),
                        MAX_HEAD_BYTES,
                    ));
                }
                _ => {}
            }
            apply(&mut s.progress, &entry.event, s.prefix)?;
            if entry.progress != s.progress {
                return Err("Mixed derived journal progress differs".into());
            }
            s.sequence = sequence;
            s.tip = sha(&raw);
            heads.push(pin(&format!("head-{sequence}.json"), &raw));
        }
        if s.tip != expected_sha256
            || read_bounded(
                &s.directory.join(format!("head-{}.json", s.sequence)),
                MAX_HEAD_BYTES,
            )? != head_raw
        {
            return Err("Mixed final head raw bytes differ".into());
        }
        s.require_current_inventory()?;
        s.usage()?;
        if let Some(mut binding) = source_binding {
            binding.heads = heads;
            binding.pins = pins;
            s.restored_source = Some(binding);
        }
        Ok(s)
    }
    /// Freshly verify the original saved callbacks/actions through this genuine
    /// restored journal. This writes nothing and returns no learning capability,
    /// owner, credit or invitation to recollect an already spent family.
    pub fn reaudit_restored_source(&self) -> Result<(), String> {
        let binding = self
            .restored_source
            .as_ref()
            .ok_or("Read-only source re-audit requires an actual journal restore")?;
        self.require_reaudit_binding(binding)?;
        let raw = read_pin(
            &self.directory,
            &binding.receipt,
            "source-receipt.json",
            MAX_SPEC_BYTES,
        )?;
        self.parent
            .collection_session(binding.spec.clone(), self.prepared)?
            .for_restored_plan(&binding.plan)?
            .reaudit_restored_read_only(
                &binding.directory,
                &raw,
                &binding.receipt.sha256,
                &binding.inventory,
            )?;
        self.require_reaudit_binding(binding)
    }
    fn require_reaudit_binding(&self, binding: &RestoredSourceBinding) -> Result<(), String> {
        if binding.heads.len() as u64 != self.sequence
            || binding.heads.last().map(|p| p.sha256.as_str()) != Some(self.tip.as_str())
        {
            return Err("Restored source re-audit latest journal binding differs".into());
        }
        for head in &binding.heads {
            read_pin(&self.directory, head, &head.name, MAX_HEAD_BYTES)?;
        }
        for (p, cap) in &binding.pins {
            read_pin(&self.directory, p, &p.name, *cap)?;
        }
        self.require_current_inventory()?;
        self.usage()?;
        Ok(())
    }
    fn require_current_inventory(&self) -> Result<(), String> {
        let mut n = 0;
        for e in fs::read_dir(&self.directory).map_err(|e| e.to_string())? {
            n += 1;
            if n > 20 {
                return Err("Mixed owner inventory cap".into());
            }
            let e = e.map_err(|e| e.to_string())?;
            let name = e
                .file_name()
                .into_string()
                .map_err(|_| "Mixed owner non-UTF8 name")?;
            if let Some(sequence) = name
                .strip_prefix("head-")
                .and_then(|s| s.strip_suffix(".json"))
            {
                let sequence = sequence
                    .parse::<u64>()
                    .map_err(|_| "Mixed owner head leaf")?;
                if sequence == 0
                    || sequence > self.sequence
                    || name != format!("head-{sequence}.json")
                {
                    return Err("Mixed stale head or unreferenced journal entry".into());
                }
            } else {
                let allowed = match name.as_str() {
                    "training-spec.json" => true,
                    "collection-spec.json" | "source-receipt.json" => self.progress.plan.is_some(),
                    "audit" => !matches!(self.progress.phase, Phase::Initial | Phase::Reserved),
                    "report.json" => self.progress.report.is_some() || self.progress.unknown_work,
                    "artifact.json" => self.progress.owner.is_some() || self.progress.unknown_work,
                    _ => false,
                };
                if !allowed {
                    return Err("Mixed unreferenced output entry".into());
                }
            }
        }
        Ok(())
    }
}
fn audit_names() -> Vec<String> {
    let mut names = vec!["audit/cohort.json".to_owned(), "audit/plan.json".to_owned()];
    for member in 0..4 {
        names.push(format!("audit/audit-started-{member}.json"));
        names.push(format!("audit/audit-returned-{member}.json"));
    }
    names.sort();
    names
}
fn audit_inventory(directory: &Path) -> Result<Vec<Pin>, String> {
    let names = audit_names();
    let mut actual = Vec::new();
    for e in fs::read_dir(directory.join("audit")).map_err(|e| e.to_string())? {
        if actual.len() >= 10 {
            return Err("Mixed audit inventory count cap".into());
        }
        let e = e.map_err(|e| e.to_string())?;
        let m = fs::symlink_metadata(e.path()).map_err(|e| e.to_string())?;
        if !m.is_file() || m.is_symlink() {
            return Err("Mixed audit inventory requires regular files".into());
        }
        actual.push(format!(
            "audit/{}",
            e.file_name()
                .into_string()
                .map_err(|_| "Mixed audit leaf")?
        ));
    }
    actual.sort();
    if actual != names {
        return Err("Mixed audit inventory is incomplete/unexpected".into());
    }
    names
        .iter()
        .map(|name| {
            let raw = read_bounded(&directory.join(name), MAX_SPEC_BYTES)?;
            unique_json(&raw, MAX_SPEC_BYTES)?;
            Ok(pin(name, &raw))
        })
        .collect()
}
fn check_audit(v: &serde_json::Value, plan: &serde_json::Value) -> Result<(), String> {
    let counts: [usize; 3] =
        serde_json::from_value(v.get("learnerCounts").ok_or("Mixed audit counts")?.clone())
            .map_err(|e| e.to_string())?;
    let episodes: [String; 4] = serde_json::from_value(
        v.get("episodeChecksums")
            .ok_or("Mixed audit episodes")?
            .clone(),
    )
    .map_err(|e| e.to_string())?;
    let observed = v
        .get("observedCallbacks")
        .and_then(|v| v.as_u64())
        .ok_or("Mixed observed callbacks")? as usize;
    let plan_sha = plan
        .get("checksum")
        .and_then(|v| v.as_str())
        .ok_or("Mixed plan checksum")?;
    if v.get("schema").and_then(|v| v.as_str()) != Some("tzolkin-validated-mixed-rl-cohort-v1")
        || v.get("planChecksum") != plan.get("checksum")
        || counts[0] == 0
        || counts[1] < counts[0]
        || counts[2] > counts[0]
        || observed < counts[0]
        || observed > 4 * crate::public_stochastic::MAX_SAMPLES
        || counts[0] > 4 * crate::public_stochastic::MAX_SAMPLES
        || counts[1] > 4 * crate::public_stochastic::MAX_RESERVED_CANDIDATE_ROWS
        || episodes.iter().any(|s| !checkpoint_digest(s))
        || v.get("checksum").and_then(|v| v.as_str())
            != Some(
                hash(
                    b"tzolkin-completed-mixed-rl-cohort-v1\0",
                    &(plan_sha, &episodes, counts, observed),
                )?
                .as_str(),
            )
    {
        return Err("Mixed audit cohort checksum/count/source shape".into());
    }
    let seats = plan
        .get("headers")
        .and_then(|v| v.as_array())
        .ok_or("Mixed audit planned headers")?
        .iter()
        .map(|h| {
            h.get("basePolicy")
                .and_then(|p| p.get("learnerSeat"))
                .cloned()
                .ok_or("Mixed audit learner seat")
        })
        .collect::<Result<Vec<_>, _>>()?;
    if v != &serde_json::json!({
        "schema":"tzolkin-validated-mixed-rl-cohort-v1",
        "cohortContract":crate::public_rl_mixed_cohort::COHORT_CONTRACT,
        "planChecksum":plan_sha,"checksum":v["checksum"],
        "episodeChecksums":episodes,"learnerCounts":counts,
        "observedCallbacks":observed,"learnerSeats":seats,
        "actorNormalization":"learnerOwnSumTimes1/4",
        "criticTargets":"learnerCallbacksOnlyIncludingSingletons",
        "trainingDataAdmission":true,"selfPlayProducerAdmission":false,
        "mixedOptimizerAvailable":false
    }) {
        return Err("Mixed audit full receipt/absolute learner/contract differs".into());
    }
    Ok(())
}
fn check_report(
    r: &MixedJointReport,
    plan: &serde_json::Value,
    audit: &serde_json::Value,
) -> Result<(), String> {
    check_audit(audit, plan)?;
    let headers = plan
        .get("headers")
        .and_then(|v| v.as_array())
        .ok_or("Mixed report planned headers")?;
    if headers.len() != 4
        || plan.get("checksum").and_then(|v| v.as_str()) != Some(r.plan_checksum.as_str())
        || plan.get("families") != Some(&serde_json::json!(r.families))
        || audit.get("checksum").and_then(|v| v.as_str()) != Some(r.cohort_checksum.as_str())
        || audit.get("episodeChecksums")
            != Some(&serde_json::to_value(&r.episode_checksums).map_err(|e| e.to_string())?)
        || audit.get("learnerCounts") != Some(&serde_json::json!(r.counts))
        || audit.get("observedCallbacks") != Some(&serde_json::json!(r.observed_callbacks))
    {
        return Err("Mixed report differs from whole saved audit".into());
    }
    for (i, h) in headers.iter().enumerate() {
        if h.get("config")
            != Some(&serde_json::to_value(&r.configurations[i]).map_err(|e| e.to_string())?)
            || h.get("basePolicy")
                .and_then(|v| v.get("rosterChecksum"))
                .and_then(|v| v.as_str())
                != Some(r.roster_checksums[i].as_str())
            || h.get("basePolicy")
                .and_then(|v| v.get("learnerSeat"))
                .and_then(|v| v.as_u64())
                != Some(r.learner_seats[i] as u64)
        {
            return Err("Mixed report configuration/absolute learner differs".into());
        }
    }
    Ok(())
}
#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct MixedOneBatchStatus {
    schema: String,
    head_file: String,
    head_sha256: String,
    imported_update_count: u64,
    mixed_update_count: u64,
    next_episode_ordinal: u64,
    reserved_families: Vec<String>,
    unresolved_work: bool,
    outcome: Option<String>,
    error: Option<String>,
    unknown_work_possible: bool,
    continuation_available: bool,
}
impl MixedOneBatchStatus {
    pub fn head_file(&self) -> &str {
        &self.head_file
    }
    pub fn head_sha256(&self) -> &str {
        &self.head_sha256
    }
    pub fn mixed_update_count(&self) -> u64 {
        self.mixed_update_count
    }
    pub fn next_episode_ordinal(&self) -> u64 {
        self.next_episode_ordinal
    }
    pub fn reserved_families(&self) -> &[String] {
        &self.reserved_families
    }
    pub fn unknown_work_possible(&self) -> bool {
        self.unknown_work_possible
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    fn state() -> Progress {
        Progress {
            phase: Phase::Initial,
            next_ordinal: 8,
            consumed_families: vec![],
            plan: None,
            source_directory: None,
            source_spec: None,
            source_receipt: None,
            source_inventory: None,
            audit: None,
            audit_inventory: Vec::new(),
            outcome: None,
            report: None,
            owner: None,
            error: None,
            unknown_work: false,
        }
    }
    fn plan() -> serde_json::Value {
        let seeds = (0..100)
            .filter(|s| split_for_family(&seed_family_id(*s)) == Ok(DatasetSplit::Train))
            .take(4)
            .collect::<Vec<_>>();
        let headers=seeds.iter().enumerate().map(|(i,s)|serde_json::json!({"config":{
            "players":if i<2{3}else{4},"environmentSeed":s,"options":{"additionalBuildings":false,"tribes":false,"prophecies":false,"quickActions":false},
            "samplingSeed":"000000000000001d","replicateOrdinal":"0000000000000001","episodeOrdinal":format!("{:016x}",8+i),
            "maxCallbacks":8,"maxCandidateRows":1000,"maxSourceBytes":1048576,"maxTraceBytes":1048576}})).collect::<Vec<_>>();
        serde_json::json!({"schema":"tzolkin-planned-mixed-rl-cohort-v1","checksum":"a".repeat(64),"headers":headers})
    }
    fn file() -> Pin {
        pin("synthetic.json", b"{}")
    }
    fn adopt(plan: serde_json::Value) -> Event {
        Event::AdoptSaved {
            plan,
            source_directory: PathBuf::from("synthetic"),
            source_inventory: serde_json::json!({}),
            spec: file(),
            receipt: file(),
        }
    }
    #[test]
    fn saved_reservation_keeps_failed_unknown_families_and_refuses_rng_reset() {
        let prefix = [29, 1, 8];
        let expected = reserve_families(&plan(), prefix, 8, &[]).unwrap();
        let mut bad = plan();
        bad["headers"][1]["config"]["episodeOrdinal"] = serde_json::json!("0000000000000008");
        assert!(reserve_families(&bad, prefix, 8, &[]).is_err());
        bad = plan();
        bad["headers"][0]["config"]["samplingSeed"] = serde_json::json!("000000000000001e");
        assert!(reserve_families(&bad, prefix, 8, &[]).is_err());
        assert!(reserve_families(&plan(), prefix, 8, &expected[..1]).is_err());
        let mut p = state();
        apply(&mut p, &adopt(plan()), prefix).unwrap();
        apply(&mut p, &Event::AuditStart, prefix).unwrap();
        let mut restored: Progress =
            closed(&bytes(&p, MAX_HEAD_BYTES).unwrap(), MAX_HEAD_BYTES).unwrap();
        assert_eq!(restored.next_ordinal, 12);
        assert!(restored.unknown_work);
        assert_eq!(restored.consumed_families.len(), 4);
        apply(
            &mut restored,
            &Event::Resolve {
                outcome: "failed".into(),
                report: None,
                owner: None,
                error: Some("bounded failure".into()),
            },
            prefix,
        )
        .unwrap();
        assert_eq!(restored.consumed_families, p.consumed_families);
        assert!(restored.unknown_work);
        assert!(apply(&mut restored, &adopt(plan()), prefix).is_err());
    }
    #[test]
    fn gradient_unknown_rejects_retry_and_requires_joint_resolution() {
        let prefix = [29, 1, 8];
        let mut p = state();
        apply(&mut p, &adopt(plan()), prefix).unwrap();
        assert!(apply(&mut p, &Event::GradientStart, prefix).is_err());
        apply(&mut p, &Event::AuditStart, prefix).unwrap();
        let inventory = audit_names()
            .iter()
            .map(|n| pin(n, b"{}"))
            .collect::<Vec<_>>();
        let cohort = pin("audit/cohort.json", b"{}");
        assert!(
            apply(
                &mut p,
                &Event::AuditReturn {
                    cohort: cohort.clone(),
                    inventory: vec![cohort.clone(); 10]
                },
                prefix
            )
            .is_err()
        );
        apply(&mut p, &Event::AuditReturn { cohort, inventory }, prefix).unwrap();
        apply(&mut p, &Event::GradientStart, prefix).unwrap();
        assert!(apply(&mut p, &Event::GradientStart, prefix).is_err());
        assert!(
            apply(
                &mut p,
                &Event::Resolve {
                    outcome: "updated".into(),
                    report: None,
                    owner: Some(file()),
                    error: None
                },
                prefix
            )
            .is_err()
        );
        let mut nochange = p.clone();
        apply(
            &mut nochange,
            &Event::Resolve {
                outcome: "noChange".into(),
                report: Some(file()),
                owner: None,
                error: None,
            },
            prefix,
        )
        .unwrap();
        assert!(nochange.owner.is_none());
        assert!(!nochange.unknown_work);
        assert_eq!(nochange.next_ordinal, 12);
        assert_eq!(nochange.consumed_families, p.consumed_families);
        apply(
            &mut p,
            &Event::Resolve {
                outcome: "abandoned".into(),
                report: None,
                owner: None,
                error: Some("unknown update".into()),
            },
            prefix,
        )
        .unwrap();
        assert!(p.unknown_work);
        assert!(p.owner.is_none());
        assert_eq!(p.next_ordinal, 12);
        let mut expected_plan = plan();
        for (i, h) in expected_plan["headers"]
            .as_array_mut()
            .unwrap()
            .iter_mut()
            .enumerate()
        {
            h["basePolicy"] = serde_json::json!({"learnerSeat":i % 3});
        }
        let episodes = std::array::from_fn::<_, 4, _>(|i| format!("{i:064x}"));
        let counts = [4usize, 5, 3];
        let observed = 8usize;
        let plan_sha = expected_plan["checksum"].as_str().unwrap();
        let checksum = hash(
            b"tzolkin-completed-mixed-rl-cohort-v1\0",
            &(plan_sha, &episodes, counts, observed),
        )
        .unwrap();
        let mut audit = serde_json::json!({
            "schema":"tzolkin-validated-mixed-rl-cohort-v1", "cohortContract":crate::public_rl_mixed_cohort::COHORT_CONTRACT,
            "planChecksum":plan_sha,"checksum":checksum,"episodeChecksums":episodes,
            "learnerCounts":counts,"observedCallbacks":observed,"learnerSeats":[0,1,2,0],
            "actorNormalization":"learnerOwnSumTimes1/4", "criticTargets":"learnerCallbacksOnlyIncludingSingletons",
            "trainingDataAdmission":true,"selfPlayProducerAdmission":false,"mixedOptimizerAvailable":false
        });
        check_audit(&audit, &expected_plan).unwrap();
        audit["learnerSeats"][1] = serde_json::json!(0);
        assert!(check_audit(&audit, &expected_plan).is_err());
        assert!(
            closed::<MixedOneBatchSpec>(b"{\"schema\":\"a\",\"schema\":\"b\"}", MAX_SPEC_BYTES)
                .is_err()
        );
        assert!(
            closed::<MixedJointArtifact>(
                b"{\"schema\":\"tzolkin-public-rl-paired-long-v1\"}",
                ARTIFACT_BYTES
            )
            .is_err()
        );
    }
}
