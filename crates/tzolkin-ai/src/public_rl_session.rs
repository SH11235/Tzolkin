//! Batch-boundary checkpoint/resume for the existing count0..10 plain-ascent
//! contracts. A caller's external raw-content pin is a trust boundary, not
//! authentication of producers, chronology or optimization history. No file
//! I/O, collection, mid-game restart, new optimizer or count11 is provided.
use std::collections::BTreeSet;

use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};

use crate::dataset::{DatasetSplit, seed_family_id, split_for_family};
use crate::public_policy_cohort::{PlannedBcCohort, ValidatedBcCohort};
use crate::public_policy_repeat::{
    self, MAX_UPDATE_COUNT, RepeatedPublicRlPolicy, RepeatedStepOutcome, RlUpdateParent,
};
use crate::public_policy_update::{
    self, OneStepConfig, OneStepOutcome, UpdatedPublicRlPolicy, checkpoint_digest,
};
use crate::public_rl_artifact::InitializedPublicRlPolicy;
use crate::public_rl_native::{RepeatedPublicRlHandle, UpdatedPublicRlHandle};
use crate::public_rl_policy_cohort::{PlannedRlCohort, ValidatedRlCohort};
use crate::public_stochastic::{RNG_VERSION, SamplingSeed};
use crate::public_stochastic_native::{NativeStochasticConfig, numerical_target};
use crate::public_stochastic_record::{Config, unique_json};
use crate::replay::{RULES_BASELINE, RULES_VERSION, catalog_hash};

pub const CHECKPOINT_SCHEMA: &str = "tzolkin-public-rl-session-checkpoint-v1";
pub const MAX_CHECKPOINT_BYTES: usize = 16 * 1024 * 1024;
const MAX_RESERVATIONS: usize = 1024;

/// Fixed for the session, including resume. Actor streams start fresh from
/// each reserved episode identity; there is no carried shuffle/momentum state.
#[derive(Clone, Debug, Serialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct SessionConfig {
    step: OneStepConfig,
    sampling_seed: u64,
    replicate_ordinal: u64,
    first_episode_ordinal: u64,
}
impl SessionConfig {
    pub fn new(
        step: OneStepConfig,
        seed: SamplingSeed,
        replicate: u64,
        first_episode: u64,
    ) -> Result<Self, String> {
        step.validate()?;
        first_episode
            .checked_add(4)
            .ok_or("Session episode ordinal overflow")?;
        Ok(Self {
            step,
            sampling_seed: seed.value(),
            replicate_ordinal: replicate,
            first_episode_ordinal: first_episode,
        })
    }
    fn decode(value: Value) -> Result<Self, String> {
        #[derive(Deserialize)]
        #[serde(rename_all = "camelCase", deny_unknown_fields)]
        struct Fields {
            step: Value,
            sampling_seed: u64,
            replicate_ordinal: u64,
            first_episode_ordinal: u64,
        }
        let f: Fields = serde_json::from_value(value).map_err(|e| e.to_string())?;
        Self::new(
            OneStepConfig::from_checkpoint(f.step)?,
            SamplingSeed::new(f.sampling_seed),
            f.replicate_ordinal,
            f.first_episode_ordinal,
        )
    }
}
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct SessionProgress {
    reserved_batches: usize,
    completed_updates: u64,
    complete_cohort_counts: [u64; 3],
    next_episode_ordinal: u64,
    unresolved_batch: bool,
}
impl SessionProgress {
    /// Includes unresolved reservations; this is not a collection-start count.
    pub fn reserved_batches(&self) -> usize {
        self.reserved_batches
    }
    pub fn completed_updates(&self) -> u64 {
        self.completed_updates
    }
    /// Audited complete-cohort callbacks, candidate rows and singletons. Failed
    /// collection counts remain unknown, not zero or instrumented forward work.
    pub fn complete_cohort_counts(&self) -> [u64; 3] {
        self.complete_cohort_counts
    }
    pub fn next_episode_ordinal(&self) -> u64 {
        self.next_episode_ordinal
    }
    pub fn unresolved_batch(&self) -> bool {
        self.unresolved_batch
    }
}
#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
enum Status {
    Reserved,
    Updated,
    NoChange,
    Failed,
}
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct Reservation {
    parent_checksum: String,
    parent_count: u64,
    plan_checksum: String,
    configurations: [Config; 4],
    status: Status,
    counts: Option<[usize; 3]>,
    report_checksum: Option<String>,
    result_checksum: Option<String>,
    error: Option<String>,
}
enum Current {
    Count1(UpdatedPublicRlPolicy),
    Repeated(RepeatedPublicRlPolicy),
}
impl Current {
    fn parent(&self) -> RlUpdateParent<'_> {
        match self {
            Self::Count1(p) => RlUpdateParent::Count1(p),
            Self::Repeated(p) => RlUpdateParent::Repeated(p),
        }
    }
    fn value(&self) -> Result<Value, String> {
        match self {
            Self::Count1(p) => serde_json::to_value(p.artifact()),
            Self::Repeated(p) => serde_json::to_value(p.artifact()),
        }
        .map_err(|e| e.to_string())
    }
    fn report_binding(&self) -> (&str, &OneStepConfig, [usize; 3]) {
        match self {
            Self::Count1(p) => {
                let r = p.artifact().report();
                (r.checksum(), r.config(), r.counts())
            }
            Self::Repeated(p) => {
                let r = p.artifact().report();
                (r.checksum(), r.config(), r.counts())
            }
        }
    }
}
#[derive(Deserialize, Serialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct Checkpoint {
    schema: String,
    contract: Value,
    config: Value,
    root_init_checksum: String,
    bc_source: Value,
    owner: Option<Value>,
    reservations: Vec<Reservation>,
    consumed_families: Vec<String>,
    progress: SessionProgress,
}
fn contract() -> Value {
    serde_json::json!({
        "optimizer": "plainAscentNoOptimizerState", "maxUpdateCount": MAX_UPDATE_COUNT,
        "maxReservations": MAX_RESERVATIONS, "rngVersion": RNG_VERSION,
        "rulesVersion": RULES_VERSION, "rulesBaseline": RULES_BASELINE,
        "catalogHash": catalog_hash(), "numericalTarget": numerical_target(),
        "firstUpdateVersion": public_policy_update::UPDATE_VERSION,
        "repeatUpdateVersion": public_policy_repeat::UPDATE_VERSION,
        "modelVersion": crate::public_model::MODEL_VERSION,
        "inputContract": crate::public_model::INPUT_CONTRACT,
        "featureSchema": crate::features::PUBLIC_FEATURE_SCHEMA, "backend": "scalar"
    })
}

/// Owns the sealed initial/current policy, never an inference deployment or
/// caller's parameter array. Resume checks saved consistency against the same
/// prepared-BC initialization and external raw pin; it does not replay history.
pub struct RlTrainingSession {
    initial: InitializedPublicRlPolicy,
    config: SessionConfig,
    current: Option<Current>,
    reservations: Vec<Reservation>,
}
impl RlTrainingSession {
    pub fn start(
        initial: InitializedPublicRlPolicy,
        config: SessionConfig,
    ) -> Result<Self, String> {
        initial.artifact().validate()?;
        config.step.validate()?;
        Ok(Self {
            initial,
            config,
            current: None,
            reservations: Vec::new(),
        })
    }
    pub fn parent(&self) -> Option<RlUpdateParent<'_>> {
        self.current.as_ref().map(Current::parent)
    }
    pub fn config(&self) -> &SessionConfig {
        &self.config
    }
    pub fn progress(&self) -> Result<SessionProgress, String> {
        Ok(self.derive()?.1)
    }
    /// Recreate the same reserved plan after resume, without allocating new
    /// families or streams. Saved collection outcomes must still be audited.
    pub fn pending_first_plan(&self) -> Result<PlannedBcCohort, String> {
        if self.current.is_some() {
            return Err("Session pending batch is not its first update".into());
        }
        let configurations = self.pending_configurations()?;
        let plan = PlannedBcCohort::new(&self.initial, configurations.clone())?;
        self.check_pending(&configurations, plan.checksum())?;
        Ok(plan)
    }
    pub fn pending_next_plan(&self) -> Result<PlannedRlCohort, String> {
        let configurations = self.pending_configurations()?;
        let plan = self.rl_plan(configurations.clone())?;
        self.check_pending(&configurations, plan.checksum())?;
        Ok(plan)
    }
    fn pending_configurations(&self) -> Result<[NativeStochasticConfig; 4], String> {
        if !self.progress()?.unresolved_batch {
            return Err("Session has no unresolved batch".into());
        }
        let r = self.reservations.last().expect("checked unresolved batch");
        let [a, b, c, d] = r.configurations.each_ref().map(Config::checked_config);
        Ok([a?, b?, c?, d?])
    }
    pub fn plan_first(
        &mut self,
        configurations: [NativeStochasticConfig; 4],
    ) -> Result<PlannedBcCohort, String> {
        if self.current.is_some() {
            return Err("Session first update already completed".into());
        }
        self.check_next(&configurations)?;
        let plan = PlannedBcCohort::new(&self.initial, configurations.clone())?;
        self.reserve(configurations, plan.checksum());
        Ok(plan)
    }
    pub fn plan_next(
        &mut self,
        configurations: [NativeStochasticConfig; 4],
    ) -> Result<PlannedRlCohort, String> {
        self.check_next(&configurations)?;
        let plan = self.rl_plan(configurations.clone())?;
        self.reserve(configurations, plan.checksum());
        Ok(plan)
    }
    fn rl_plan(
        &self,
        configurations: [NativeStochasticConfig; 4],
    ) -> Result<PlannedRlCohort, String> {
        match self
            .current
            .as_ref()
            .ok_or("Session requires its first update")?
        {
            Current::Count1(p) => {
                PlannedRlCohort::new(&UpdatedPublicRlHandle::new(p)?, configurations)
            }
            Current::Repeated(p) => {
                PlannedRlCohort::new_repeated(&RepeatedPublicRlHandle::new(p)?, configurations)
            }
        }
    }
    fn check_next(&self, configurations: &[NativeStochasticConfig; 4]) -> Result<(), String> {
        let (families, progress) = self.derive()?;
        if progress.unresolved_batch
            || self.reservations.len() >= MAX_RESERVATIONS
            || progress.completed_updates >= MAX_UPDATE_COUNT
        {
            return Err(
                "Session has unresolved work, exhausted reservations or final count10".into(),
            );
        }
        check_configs(
            configurations,
            &self.config,
            progress.next_episode_ordinal,
            &families,
        )?;
        Ok(())
    }
    fn identity(&self) -> (&str, u64) {
        match &self.current {
            Some(Current::Count1(p)) => (p.artifact().checksum(), p.artifact().update_count()),
            Some(Current::Repeated(p)) => (p.artifact().checksum(), p.artifact().update_count()),
            None => (self.initial.artifact().checksum(), 0),
        }
    }
    fn reserve(&mut self, configurations: [NativeStochasticConfig; 4], checksum: &str) {
        let (parent, count) = self.identity();
        let entry = Reservation {
            parent_checksum: parent.into(),
            parent_count: count,
            plan_checksum: checksum.into(),
            configurations: configurations.each_ref().map(Config::from_config),
            status: Status::Reserved,
            counts: None,
            report_checksum: None,
            result_checksum: None,
            error: None,
        };
        self.reservations.push(entry);
    }
    fn check_pending(
        &self,
        configurations: &[NativeStochasticConfig; 4],
        checksum: &str,
    ) -> Result<(), String> {
        let r = self
            .reservations
            .last()
            .ok_or("Session has no reserved batch")?;
        let (parent, count) = self.identity();
        if r.status != Status::Reserved
            || r.parent_checksum != parent
            || r.parent_count != count
            || r.plan_checksum != checksum
            || r.configurations
                .iter()
                .zip(configurations)
                .any(|(a, b)| a.checked_config().as_ref() != Ok(b))
        {
            return Err("Session cohort differs from its pending reservation".into());
        }
        Ok(())
    }
    pub fn apply_first(&mut self, cohort: &ValidatedBcCohort) -> Result<bool, String> {
        if self.current.is_some() {
            return Err("Session first update already completed".into());
        }
        self.check_pending(cohort.plan().configurations(), cohort.plan().checksum())?;
        let counts = [
            cohort.total_callbacks(),
            cohort.total_candidate_rows(),
            cohort.total_singletons(),
        ];
        match public_policy_update::ascent_one_step(&self.initial, cohort, &self.config.step) {
            Ok(OneStepOutcome::Updated(p)) => {
                let report = p.artifact().report().checksum().to_owned();
                self.complete(Some(Current::Count1(p)), counts, report);
                Ok(true)
            }
            Ok(OneStepOutcome::NoChange(r)) => {
                self.complete(None, counts, r.checksum().into());
                Ok(false)
            }
            Err(error) => {
                self.fail(Some(counts), &error);
                Err(error)
            }
        }
    }
    pub fn apply_next(&mut self, cohort: &ValidatedRlCohort) -> Result<bool, String> {
        self.check_pending(cohort.plan().configurations(), cohort.plan().checksum())?;
        let counts = [
            cohort.total_callbacks(),
            cohort.total_candidate_rows(),
            cohort.total_singletons(),
        ];
        match public_policy_repeat::ascent_repeated(
            self.parent().ok_or("Session requires its first update")?,
            cohort,
            &self.config.step,
        ) {
            Ok(RepeatedStepOutcome::Updated(p)) => {
                let report = p.artifact().report().checksum().to_owned();
                self.complete(Some(Current::Repeated(p)), counts, report);
                Ok(true)
            }
            Ok(RepeatedStepOutcome::NoChange(r)) => {
                self.complete(None, counts, r.checksum().into());
                Ok(false)
            }
            Err(error) => {
                self.fail(Some(counts), &error);
                Err(error)
            }
        }
    }
    fn complete(&mut self, current: Option<Current>, counts: [usize; 3], report: String) {
        let changed = current.is_some();
        if let Some(current) = current {
            self.current = Some(current);
        }
        let result = self.identity().0.to_owned();
        let entry = self.reservations.last_mut().expect("checked reservation");
        entry.status = if changed {
            Status::Updated
        } else {
            Status::NoChange
        };
        entry.counts = Some(counts);
        entry.report_checksum = Some(report);
        entry.result_checksum = Some(result);
    }
    fn fail(&mut self, counts: Option<[usize; 3]>, error: &str) {
        let entry = self.reservations.last_mut().expect("checked reservation");
        entry.status = Status::Failed;
        entry.counts = counts;
        entry.result_checksum = Some(entry.parent_checksum.clone());
        entry.error = Some(error.chars().take(1024).collect());
    }
    /// Resolve a lost/failed collection explicitly. Its four reserved identities
    /// remain consumed. This never retries or claims an observed zero work count.
    pub fn abandon_pending(&mut self, reason: &str) -> Result<(), String> {
        if reason.is_empty()
            || self
                .reservations
                .last()
                .is_none_or(|r| r.status != Status::Reserved)
        {
            return Err("Session has no unresolved batch or empty failure reason".into());
        }
        self.fail(None, reason);
        Ok(())
    }
    pub fn checkpoint_bytes(&self) -> Result<Vec<u8>, String> {
        let (families, progress) = self.derive()?;
        self.check_owner(&progress)?;
        let checkpoint = Checkpoint {
            schema: CHECKPOINT_SCHEMA.into(),
            contract: contract(),
            config: serde_json::to_value(&self.config).map_err(|e| e.to_string())?,
            root_init_checksum: self.initial.artifact().checksum().into(),
            bc_source: serde_json::to_value(self.initial.artifact().bc_source())
                .map_err(|e| e.to_string())?,
            owner: self.current.as_ref().map(Current::value).transpose()?,
            reservations: self.reservations.clone(),
            consumed_families: families.into_iter().collect(),
            progress,
        };
        let bytes = serde_json::to_vec(&checkpoint).map_err(|e| e.to_string())?;
        if bytes.len() > MAX_CHECKPOINT_BYTES {
            return Err("Session checkpoint byte bound".into());
        }
        Ok(bytes)
    }
    pub fn restore(
        initial: InitializedPublicRlPolicy,
        bytes: &[u8],
        expected_raw_sha256: &str,
    ) -> Result<Self, String> {
        if !checkpoint_digest(expected_raw_sha256)
            || bytes.is_empty()
            || bytes.len() > MAX_CHECKPOINT_BYTES
            || format!("{:x}", Sha256::digest(bytes)) != expected_raw_sha256
        {
            return Err("Session checkpoint external raw pin/byte bound mismatch".into());
        }
        initial.artifact().validate()?;
        let raw = unique_json(bytes, MAX_CHECKPOINT_BYTES)?;
        let wire: Checkpoint = serde_json::from_value(raw.clone()).map_err(|e| e.to_string())?;
        if serde_json::to_value(&wire).map_err(|e| e.to_string())? != raw
            || wire.schema != CHECKPOINT_SCHEMA
            || wire.contract != contract()
            || wire.root_init_checksum != initial.artifact().checksum()
            || wire.bc_source
                != serde_json::to_value(initial.artifact().bc_source())
                    .map_err(|e| e.to_string())?
        {
            return Err("Session checkpoint closed contract/root mismatch".into());
        }
        let current = wire
            .owner
            .map(|v| match v.get("updateCount").and_then(Value::as_u64) {
                Some(1) => public_policy_update::restore_checkpoint_owner(v).map(Current::Count1),
                Some(2..=MAX_UPDATE_COUNT) => {
                    public_policy_repeat::restore_checkpoint_owner(v).map(Current::Repeated)
                }
                _ => Err("Invalid checkpoint owner count".into()),
            })
            .transpose()?;
        let session = Self {
            initial,
            config: SessionConfig::decode(wire.config)?,
            current,
            reservations: wire.reservations,
        };
        let (families, progress) = session.derive()?;
        if wire.progress != progress
            || wire.consumed_families != families.iter().cloned().collect::<Vec<_>>()
        {
            return Err("Checkpoint progress/consumed family mismatch".into());
        }
        session.check_owner(&progress)?;
        Ok(session)
    }
    fn derive(&self) -> Result<(BTreeSet<String>, SessionProgress), String> {
        if self.reservations.len() > MAX_RESERVATIONS {
            return Err("Session reservation bound".into());
        }
        let mut families = self
            .initial
            .artifact()
            .family_closure()
            .iter()
            .map(|f| f.family_id().to_owned())
            .collect::<BTreeSet<_>>();
        let mut parent = self.initial.artifact().checksum().to_owned();
        let mut count = 0;
        let mut total = [0u64; 3];
        let mut ordinal = self.config.first_episode_ordinal;
        for (i, r) in self.reservations.iter().enumerate() {
            if count >= MAX_UPDATE_COUNT
                || r.parent_count != count
                || r.parent_checksum != parent
                || !checkpoint_digest(&r.plan_checksum)
            {
                return Err("Session reservation parent/count/plan mismatch".into());
            }
            let [a, b, c, d] = r.configurations.each_ref().map(Config::checked_config);
            let configs = [a?, b?, c?, d?];
            check_configs(&configs, &self.config, ordinal, &families)?;
            for config in &configs {
                families.insert(seed_family_id(config.environment_seed()));
            }
            ordinal = ordinal
                .checked_add(4)
                .ok_or("Session episode ordinal overflow")?;
            if let Some(counts) = r.counts {
                if counts[0] == 0
                    || counts[1] < counts[0]
                    || counts[2] > counts[0]
                    || counts[0]
                        > configs
                            .iter()
                            .map(|c| c.limits().max_callbacks())
                            .sum::<usize>()
                    || counts[1]
                        > configs
                            .iter()
                            .map(|c| c.limits().max_candidate_rows())
                            .sum::<usize>()
                {
                    return Err("Session complete-cohort count bound".into());
                }
                for (sum, amount) in total.iter_mut().zip(counts) {
                    *sum = sum
                        .checked_add(amount as u64)
                        .ok_or("Session count overflow")?;
                }
            }
            match r.status {
                Status::Reserved
                    if i + 1 == self.reservations.len()
                        && r.counts.is_none()
                        && r.report_checksum.is_none()
                        && r.result_checksum.is_none()
                        && r.error.is_none() => {}
                Status::Updated | Status::NoChange
                    if r.counts.is_some()
                        && r.error.is_none()
                        && r.report_checksum.as_deref().is_some_and(checkpoint_digest)
                        && r.result_checksum.as_deref().is_some_and(checkpoint_digest) =>
                {
                    let result = r.result_checksum.as_ref().expect("checked result");
                    if r.status == Status::Updated {
                        if result == &parent {
                            return Err("Updated checkpoint retained old identity".into());
                        }
                        count += 1;
                        parent.clone_from(result);
                    } else if result != &parent {
                        return Err("NoChange checkpoint changed owner".into());
                    }
                }
                Status::Failed
                    if r.report_checksum.is_none()
                        && r.result_checksum.as_deref() == Some(parent.as_str())
                        && r.error
                            .as_ref()
                            .is_some_and(|s| !s.is_empty() && s.chars().count() <= 1024) => {}
                _ => return Err("Invalid session reservation outcome/unknown evidence".into()),
            }
        }
        let progress = SessionProgress {
            reserved_batches: self.reservations.len(),
            completed_updates: count,
            complete_cohort_counts: total,
            next_episode_ordinal: ordinal,
            unresolved_batch: self
                .reservations
                .last()
                .is_some_and(|r| r.status == Status::Reserved),
        };
        if self.identity() != (parent.as_str(), count) {
            return Err("Session current owner/progress mismatch".into());
        }
        Ok((families, progress))
    }
    fn check_owner(&self, progress: &SessionProgress) -> Result<(), String> {
        let Some(current) = &self.current else {
            return Ok(());
        };
        let parent = current.parent();
        let (report, config, counts) = current.report_binding();
        let last = self
            .reservations
            .iter()
            .rfind(|r| r.status == Status::Updated)
            .ok_or("Checkpoint owner lacks completed update")?;
        let report_parent_matches = match current {
            Current::Count1(p) => {
                p.validate_for_inference()?;
                let r = p.artifact().report();
                last.parent_count == 0
                    && r.parent_init_checksum() == last.parent_checksum
                    && r.checkpoint_plan_checksum() == last.plan_checksum
            }
            Current::Repeated(p) => {
                p.validate_for_handle()?;
                let r = p.artifact().report();
                r.parent_checksum() == last.parent_checksum
                    && r.parent_update_count() == last.parent_count
                    && r.checkpoint_plan_checksum() == last.plan_checksum
            }
        };
        let mut inherited = self
            .initial
            .artifact()
            .family_closure()
            .iter()
            .map(|f| f.family_id().to_owned())
            .collect::<BTreeSet<_>>();
        for r in self
            .reservations
            .iter()
            .filter(|r| r.status == Status::Updated)
        {
            for c in &r.configurations {
                inherited.insert(seed_family_id(c.environment_seed));
            }
        }
        if parent.root_init_checksum() != self.initial.artifact().checksum()
            || serde_json::to_value(parent.bc_source()).map_err(|e| e.to_string())?
                != serde_json::to_value(self.initial.artifact().bc_source())
                    .map_err(|e| e.to_string())?
            || parent.family_closure() != inherited.into_iter().collect::<Vec<_>>()
            || parent.update_count() != progress.completed_updates
            || config != &self.config.step
            || !report_parent_matches
            || last.report_checksum.as_deref() != Some(report)
            || last.counts != Some(counts)
        {
            return Err("Checkpoint owner root/family/config/report mismatch".into());
        }
        Ok(())
    }
}
fn check_configs(
    configs: &[NativeStochasticConfig; 4],
    config: &SessionConfig,
    ordinal: u64,
    inherited: &BTreeSet<String>,
) -> Result<(), String> {
    if configs.iter().filter(|c| c.players() == 3).count() != 2
        || configs.iter().filter(|c| c.players() == 4).count() != 2
    {
        return Err("Session requires two 3p and two 4p games".into());
    }
    let mut families = BTreeSet::new();
    for (i, c) in configs.iter().enumerate() {
        let identity = c.sampling_identity();
        let family = seed_family_id(c.environment_seed());
        if split_for_family(&family)? != DatasetSplit::Train
            || inherited.contains(&family)
            || !families.insert(family)
            || identity.sampling_seed().value() != config.sampling_seed
            || identity.replicate_ordinal() != config.replicate_ordinal
            || identity.episode_ordinal()
                != ordinal
                    .checked_add(i as u64)
                    .ok_or("Session ordinal overflow")?
        {
            return Err(
                "Session requires unused Train families and fixed consecutive sampling identities"
                    .into(),
            );
        }
    }
    ordinal.checked_add(4).ok_or("Session ordinal overflow")?;
    Ok(())
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::public_stochastic::SamplingStreamIdentity;
    use crate::public_stochastic_native::CollectionLimits;

    fn configs(session: &RlTrainingSession) -> [NativeStochasticConfig; 4] {
        let (excluded, progress) = session.derive().unwrap();
        let seeds = (0..1000)
            .filter(|s| {
                let f = seed_family_id(*s);
                split_for_family(&f).unwrap() == DatasetSplit::Train && !excluded.contains(&f)
            })
            .take(4)
            .collect::<Vec<_>>();
        std::array::from_fn(|i| {
            NativeStochasticConfig::new(
                if i < 2 { 3 } else { 4 },
                seeds[i],
                SamplingStreamIdentity::new(
                    SamplingSeed::new(session.config.sampling_seed),
                    progress.next_episode_ordinal + i as u64,
                    session.config.replicate_ordinal,
                    if i < 2 { 3 } else { 4 },
                )
                .unwrap(),
                CollectionLimits::default(),
            )
            .unwrap()
        })
    }
    fn updated_families(session: &RlTrainingSession) -> Vec<String> {
        let mut families = session
            .initial
            .artifact()
            .family_closure()
            .iter()
            .map(|f| f.family_id().to_owned())
            .collect::<BTreeSet<_>>();
        for r in &session.reservations {
            if r.status == Status::Updated || r.status == Status::Reserved {
                for c in &r.configurations {
                    families.insert(seed_family_id(c.environment_seed));
                }
            }
        }
        families.into_iter().collect()
    }
    fn snapshot(
        factory: &impl Fn() -> InitializedPublicRlPolicy,
        live: &RlTrainingSession,
    ) -> RlTrainingSession {
        let bytes = live.checkpoint_bytes().unwrap();
        let digest = format!("{:x}", Sha256::digest(&bytes));
        let resumed = RlTrainingSession::restore(factory(), &bytes, &digest).unwrap();
        assert_eq!(live.progress().unwrap(), resumed.progress().unwrap());
        assert_eq!(live.config(), resumed.config());
        assert_eq!(bytes, resumed.checkpoint_bytes().unwrap());
        assert!(RlTrainingSession::restore(factory(), &bytes, &"0".repeat(64)).is_err());
        for field in ["rootInitChecksum", "contract", "progress", "extra"] {
            let mut v: Value = serde_json::from_slice(&bytes).unwrap();
            v[field] = Value::Null;
            let bad = serde_json::to_vec(&v).unwrap();
            assert!(
                RlTrainingSession::restore(factory(), &bad, &format!("{:x}", Sha256::digest(&bad)))
                    .is_err()
            );
        }
        resumed
    }
    pub(crate) fn roundtrip(factory: impl Fn() -> InitializedPublicRlPolicy) {
        let config = SessionConfig::new(
            OneStepConfig::default(),
            SamplingSeed::new(u64::MAX),
            u64::MAX,
            7,
        )
        .unwrap();
        let mut live = RlTrainingSession::start(factory(), config).unwrap();
        let first = configs(&live);
        let first_plan = live.plan_first(first.clone()).unwrap();
        let mut unknown = snapshot(&factory, &live);
        assert!(unknown.progress().unwrap().unresolved_batch());
        assert_eq!(
            unknown.pending_first_plan().unwrap().checksum(),
            first_plan.checksum()
        );
        assert!(unknown.pending_next_plan().is_err());
        assert!(unknown.plan_first(first).is_err());
        unknown.abandon_pending("interrupted collection").unwrap();
        live = snapshot(&factory, &unknown);
        assert_eq!(live.progress().unwrap().complete_cohort_counts(), [0; 3]);
        assert_eq!(live.progress().unwrap().next_episode_ordinal(), 11);
        let nochange = configs(&live);
        live.plan_first(nochange).unwrap();
        live.complete(None, [8, 12, 2], "c".repeat(64));
        live = snapshot(&factory, &live);
        assert_eq!(live.progress().unwrap().completed_updates(), 0);
        let plan = live.plan_first(configs(&live)).unwrap();
        let owner = public_policy_update::checkpoint_test_owner(
            &live.initial,
            plan.checksum(),
            updated_families(&live),
            [8, 12, 2],
            &live.config.step,
        );
        let report = owner.artifact().report().checksum().to_owned();
        live.complete(Some(Current::Count1(owner)), [8, 12, 2], report);
        for expected_count in [1, 2] {
            let resumed = snapshot(&factory, &live);
            let parameters = |session: &RlTrainingSession| match session.current.as_ref().unwrap() {
                Current::Count1(p) => p.model().parameters().to_vec(),
                Current::Repeated(p) => p.model().parameters().to_vec(),
            };
            let old = parameters(&live);
            let loaded = parameters(&resumed);
            assert!(
                old.iter()
                    .zip(&loaded)
                    .all(|(a, b)| a.to_bits() == b.to_bits())
            );
            assert_eq!(old[0].to_bits(), (-0.0_f32).to_bits());
            let mut gradient = vec![0.0; old.len()];
            gradient[1] = 0.75;
            gradient[2] = -0.5;
            let a = public_policy_update::propose_parameters(&old, &gradient, &live.config.step)
                .unwrap();
            let b =
                public_policy_update::propose_parameters(&loaded, &gradient, &resumed.config.step)
                    .unwrap();
            assert!(
                a.parameters
                    .iter()
                    .zip(&b.parameters)
                    .all(|(a, b)| a.to_bits() == b.to_bits())
            );
            assert_eq!(live.progress().unwrap().completed_updates(), expected_count);
            if expected_count == 1 {
                let plan = live.plan_next(configs(&live)).unwrap();
                assert_eq!(
                    snapshot(&factory, &live)
                        .pending_next_plan()
                        .unwrap()
                        .checksum(),
                    plan.checksum()
                );
                let owner = public_policy_repeat::checkpoint_test_owner(
                    live.parent().unwrap(),
                    plan.checksum(),
                    updated_families(&live),
                    [8, 12, 2],
                    &live.config.step,
                );
                let report = owner.artifact().report().checksum().to_owned();
                live.complete(Some(Current::Repeated(owner)), [8, 12, 2], report);
            }
        }
        let original = live.checkpoint_bytes().unwrap();
        let mut v: Value = serde_json::from_slice(&original).unwrap();
        v["owner"]["model"]["parameters"][1] = serde_json::json!(0.25);
        let bad = serde_json::to_vec(&v).unwrap();
        assert!(
            RlTrainingSession::restore(factory(), &bad, &format!("{:x}", Sha256::digest(&bad)))
                .is_err()
        );
        let duplicate = b"{\"schema\":1,\"schema\":2}";
        assert!(
            RlTrainingSession::restore(
                factory(),
                duplicate,
                &format!("{:x}", Sha256::digest(duplicate))
            )
            .is_err()
        );
        let nonfinite = b"{\"model\":1e999}";
        assert!(
            RlTrainingSession::restore(
                factory(),
                nonfinite,
                &format!("{:x}", Sha256::digest(nonfinite))
            )
            .is_err()
        );
        let oversized = vec![0u8; MAX_CHECKPOINT_BYTES + 1];
        assert!(RlTrainingSession::restore(factory(), &oversized, &"0".repeat(64)).is_err());
        assert_eq!(live.progress().unwrap().reserved_batches(), 4);
        assert_eq!(live.progress().unwrap().next_episode_ordinal(), 23);
        assert_eq!(
            live.progress().unwrap().complete_cohort_counts(),
            [24, 36, 6]
        );
    }
}
