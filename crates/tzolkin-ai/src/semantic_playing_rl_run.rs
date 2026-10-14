//! A finite concrete loader/loop over the journal, not another learner or host.
//! Original inputs never change on resume. A newer declared global PRIOR checks
//! only prospective work; the separately pinned full allocation is host-consumed
//! even when its later batches never enter. Neither metadata authenticates an OS.
use super::*;
use crate::arena::teacher::PreparedSemanticPlayingTeacher;
use crate::conditional_human_rl_update::ConditionalNativeExclusions;
use crate::conditional_semantic_playing_rl::SemanticFreshOwner;
use crate::conditional_semantic_playing_rl::SemanticPlayingRlOwner;
use crate::dataset::{DatasetSplit, seed_family_id, split_for_family};
use crate::human_policy_training::PreparedConditionalHumanPolicy;
use crate::public_policy_baseline::ResidualFitConfig;
use crate::public_policy_update::{OneStepConfig, checkpoint_digest};
use crate::public_stochastic::{SamplingSeed, SamplingStreamIdentity};
use crate::public_stochastic_record::{Config, unique_json};
use crate::semantic_playing_rl_collection::{
    DeclaredSamplingHistory, DeclaredSamplingRange, SemanticCollectionComposition,
};
use crate::semantic_playing_rl_pool::SnapshotSlot;
use std::sync::Arc;

const RUN_SCHEMA: &str = "tzolkin-semantic-playing-journal-run-v1";
const AUDIT_SCHEMA: &str = "tzolkin-semantic-playing-journal-audit-v1";
const EVALUATION_SCHEMA: &str = "tzolkin-semantic-playing-journal-evaluation-v1";
const POLICY_EXPORT_SCHEMA: &str = "tzolkin-semantic-playing-checkpoint-policy-export-v1";
const POLICY_EVALUATION_SCHEMA: &str = "tzolkin-semantic-playing-policy-evaluation-v1";
const ALLOCATION_SCHEMA: &str = "tzolkin-semantic-playing-invocation-allocation-v1";
const MAX_CONTROL: usize = 1024 * 1024;
const MAX_HISTORY_PINS: usize = 128;
#[path = "semantic_playing_rl_checkpoint.rs"]
mod checkpoint;

#[derive(Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct PolicyExportWire {
    schema: String,
    checkpoint: checkpoint::Input,
}

/// A separate checked, closed-checkpoint export request, with no original
/// preparation, critic configuration, training schedule or raw actor input.
pub struct SemanticPolicyExportSpec {
    wire: PolicyExportWire,
    checksum: String,
}
impl SemanticPolicyExportSpec {
    pub fn from_json(raw: &[u8], expected_sha256: &str) -> Result<Self, String> {
        if !checkpoint_digest(expected_sha256) || crate::model::digest(raw) != expected_sha256 {
            return Err("Semantic policy export control pin differs".into());
        }
        let wire: PolicyExportWire =
            serde_json::from_value(unique_json(raw, MAX_CONTROL)?).map_err(|e| e.to_string())?;
        if wire.schema != POLICY_EXPORT_SCHEMA {
            return Err("Semantic policy export schema differs".into());
        }
        Ok(Self {
            wire,
            checksum: expected_sha256.into(),
        })
    }
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct SemanticPolicyExport {
    schema: String,
    control_sha256: String,
    artifact_file: String,
    artifact_bytes: usize,
    artifact_sha256: String,
    deployment_checksum: String,
    actor_checksum: String,
    current_identity_checksum: String,
    updates: u64,
    playing_credit: u64,
    source_authenticated: bool,
    source_qualified: bool,
    strength_measured: bool,
    adoption_granted: bool,
}

/// Exports only the actor from a completed locally checked checkpoint. No
/// Prepared load, archive replay, allocation, learning update or owner escapes.
pub fn export_semantic_playing_policy(
    spec: SemanticPolicyExportSpec,
    output: &Path,
) -> Result<SemanticPolicyExport, String> {
    checkpoint::export_policy(&spec.wire.checkpoint, &spec.checksum, output)
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct PolicyEvaluationWire {
    schema: String,
    candidate: ContentPin,
    reference: ContentPin,
    global_external: External,
    players: usize,
    seeds: Vec<u32>,
    bootstrap_seed: u64,
}

/// Policy-only evaluation accepts validated inference artifacts, never a
/// checkpoint/owner/journal under a different role. GLOBAL is caller declared.
pub struct SemanticPolicyEvaluationSpec {
    wire: PolicyEvaluationWire,
    checksum: String,
}
impl SemanticPolicyEvaluationSpec {
    pub fn from_json(raw: &[u8], expected_sha256: &str) -> Result<Self, String> {
        if !checkpoint_digest(expected_sha256) || crate::model::digest(raw) != expected_sha256 {
            return Err("Semantic policy evaluation control pin differs".into());
        }
        let wire: PolicyEvaluationWire =
            serde_json::from_value(unique_json(raw, MAX_CONTROL)?).map_err(|e| e.to_string())?;
        if wire.schema != POLICY_EVALUATION_SCHEMA {
            return Err("Semantic policy evaluation schema differs".into());
        }
        policy_evaluation_config(&wire, Vec::new()).validate()?;
        Ok(Self {
            wire,
            checksum: expected_sha256.into(),
        })
    }
}
fn policy_evaluation_config(
    w: &PolicyEvaluationWire,
    families: Vec<String>,
) -> crate::arena::numeric::NumericArenaConfig {
    crate::arena::numeric::NumericArenaConfig {
        schema: crate::arena::numeric::SCHEMA.into(),
        players: w.players,
        seeds: w.seeds.clone(),
        bootstrap_seed: w.bootstrap_seed,
        external_exclusions: crate::arena::numeric::ExternalExclusions {
            evidence_sha256: w.global_external.pin.sha256.clone(),
            excluded_family_ids: families,
        },
    }
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct SemanticPolicyEvaluation {
    schema: &'static str,
    control_sha256: String,
    candidate_artifact: ContentPin,
    reference_artifact: ContentPin,
    global_external: crate::conditional_human_rl_update::NativeExclusionDescriptor,
    arena: crate::arena::semantic_snapshot::SemanticPlayingCheckpointArenaReport,
    historical_core_replays: usize,
    source_authenticated: bool,
    source_qualified: bool,
    strength_measured: bool,
    adoption_granted: bool,
}
impl SemanticPolicyEvaluation {
    pub fn exit_code(&self) -> i32 {
        if self.arena.paired.statistics.failed_games == 0
            && self.arena.paired.blocks.iter().all(|b| b.complete)
        {
            0
        } else {
            1
        }
    }
}

/// Numeric paired evaluation with no Prepared load, archive restore or training
/// admission. The caller must inventory all excluded families independently.
pub fn evaluate_semantic_playing_policy(
    spec: SemanticPolicyEvaluationSpec,
    output: &Path,
) -> Result<SemanticPolicyEvaluation, String> {
    use crate::arena::{numeric, semantic_snapshot::SemanticPlayingCheckpointArenaPolicy};
    use tzolkin_inference::semantic_playing_deployment::{
        LoadedSemanticPlayingDeployment, MAX_DEPLOYMENT_BYTES,
    };
    let w = &spec.wire;
    let result = (|| {
        if !output.is_absolute() {
            return Err("Semantic policy evaluation needs an absolute output namespace".into());
        }
        let output = local(output, true)?;
        if output.to_string_lossy().len() > 1024 || fs::symlink_metadata(&output).is_ok() {
            return Err("Semantic policy evaluation needs a fresh bounded namespace".into());
        }
        reject_input_namespace(
            &output,
            &serde_json::to_value((&w.candidate, &w.reference, &w.global_external))
                .map_err(|e| e.to_string())?,
        )?;
        let candidate = SemanticPlayingCheckpointArenaPolicy::new(
            LoadedSemanticPlayingDeployment::load(&pinned(&w.candidate, MAX_DEPLOYMENT_BYTES)?)?,
        )?;
        let reference = SemanticPlayingCheckpointArenaPolicy::new(
            LoadedSemanticPlayingDeployment::load(&pinned(&w.reference, MAX_DEPLOYMENT_BYTES)?)?,
        )?;
        let raw = pinned(&w.global_external.pin, 32 * 1024 * 1024)?;
        let global = w.global_external.load()?;
        let mut external = unique_json(&raw, 32 * 1024 * 1024)?;
        let families =
            serde_json::from_value(external["families"].take()).map_err(|e| e.to_string())?;
        let config = policy_evaluation_config(w, families);
        config.validate()?;
        let arena =
            numeric::run_semantic_checkpoint_policy_validation(&config, candidate, reference)?;
        let report = SemanticPolicyEvaluation {
            schema: "tzolkin-semantic-playing-policy-evaluation-report-v1",
            control_sha256: spec.checksum,
            candidate_artifact: w.candidate.clone(),
            reference_artifact: w.reference.clone(),
            global_external: global.descriptor().clone(),
            arena,
            historical_core_replays: 0,
            source_authenticated: false,
            source_qualified: false,
            strength_measured: false,
            adoption_granted: false,
        };
        fs::create_dir(&output).map_err(|e| e.to_string())?;
        crate::model::write_new_json_bounded(
            &output.join("evaluation.json"),
            &report,
            numeric::MAX_REPORT_BYTES,
        )?;
        Ok(report)
    })();
    evaluation_post(
        result,
        [
            pinned(&w.candidate, MAX_DEPLOYMENT_BYTES).map(|_| ()),
            pinned(&w.reference, MAX_DEPLOYMENT_BYTES).map(|_| ()),
            pinned(&w.global_external.pin, 32 * 1024 * 1024)
                .and_then(|_| w.global_external.load().map(|_| ())),
        ],
    )
}

#[derive(Deserialize, Serialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct External {
    pin: ContentPin,
    count: usize,
    canonical_digest: String,
}
impl External {
    fn load(&self) -> Result<ConditionalNativeExclusions, String> {
        ConditionalNativeExclusions::load(
            Path::new(&self.pin.path),
            &self.pin,
            self.count,
            &self.canonical_digest,
        )
    }
}
#[derive(Deserialize, Serialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct History {
    // sampling seed, replicate, players, absolute actor, first, next; all exact.
    ranges: Vec<[u64; 6]>,
    pins: Vec<ContentPin>,
}
impl History {
    fn load(&self) -> Result<DeclaredSamplingHistory, String> {
        if self.pins.len() > MAX_HISTORY_PINS || self.ranges.len() > 8192 {
            return Err("Runner history cap".into());
        }
        let ranges = self
            .ranges
            .iter()
            .map(|r| {
                DeclaredSamplingRange::new(
                    SamplingSeed::new(r[0]),
                    r[1],
                    usize::try_from(r[2]).map_err(|_| "History players")?,
                    usize::try_from(r[3]).map_err(|_| "History actor")?,
                    r[4],
                    r[5],
                )
            })
            .collect::<Result<Vec<_>, String>>()?;
        DeclaredSamplingHistory::declare_complete(ranges, self.pins.clone())
    }
}
#[derive(Deserialize, Serialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
enum InitializationKind {
    NativeCornFirstV5SemanticPlayingTeacher,
}

enum LoadedInitialization {
    Human(PreparedConditionalHumanPolicy),
    Native(Arc<PreparedSemanticPlayingTeacher>),
}
#[derive(Clone, Copy)]
enum LoadedInitializationRef<'a> {
    Human(&'a PreparedConditionalHumanPolicy),
    Native(&'a Arc<PreparedSemanticPlayingTeacher>),
}
impl LoadedInitialization {
    fn borrowed(&self) -> LoadedInitializationRef<'_> {
        match self {
            Self::Human(prepared) => LoadedInitializationRef::Human(prepared),
            Self::Native(prepared) => LoadedInitializationRef::Native(prepared),
        }
    }
    fn recheck_sources(&self) -> Result<(), String> {
        match self {
            Self::Human(prepared) => prepared.recheck_sources(),
            Self::Native(prepared) => prepared.recheck_sources(),
        }
    }
    fn human(&self) -> Result<&PreparedConditionalHumanPolicy, String> {
        match self {
            Self::Human(prepared) => Ok(prepared),
            Self::Native(_) => Err("OriginalHuman4 requires a Human initialization".into()),
        }
    }
}
impl LoadedInitializationRef<'_> {
    fn fresh_owner(self, seed: u64) -> Result<SemanticFreshOwner, String> {
        match self {
            Self::Human(prepared) => Ok(SemanticFreshOwner::Human(Box::new(
                SemanticPlayingRlOwner::from_prepared(prepared, seed)?,
            ))),
            Self::Native(prepared) => SemanticFreshOwner::native(Arc::clone(prepared), seed),
        }
    }
}

#[derive(Deserialize, Serialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct Original {
    checkpoint: ContentPin,
    collection: ContentPin,
    external: External,
    history: History,
    critic_seed: u64,
    sampling_seed: u64,
    replicate: u64,
    first_ordinal: u64,
    actor_config: Value,
    critic_config: Value,
    limits: [usize; 4],
    // attempts, logical, stored, metadata, disk; exactly the original budget.
    budget: [u64; 5],
    #[serde(default, skip_serializing_if = "Option::is_none")]
    collection_composition: Option<SemanticCollectionComposition>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    initialization_kind: Option<InitializationKind>,
}
impl Original {
    fn composition(&self) -> Result<SemanticCollectionComposition, String> {
        if self.initialization_kind.is_some()
            && self.collection_composition != Some(SemanticCollectionComposition::GuardedProposalV2)
        {
            return Err("Native initialization requires guardedProposalV2".into());
        }
        match self.collection_composition {
            None => Ok(SemanticCollectionComposition::LegacyV1),
            Some(SemanticCollectionComposition::GuardedProposalV2) => {
                Ok(SemanticCollectionComposition::GuardedProposalV2)
            }
            _ => Err("Original collectionComposition must be absent or guardedProposalV2".into()),
        }
    }
    fn limits(&self) -> Result<CollectionLimits, String> {
        self.composition()?;
        CollectionLimits::new(
            self.limits[0],
            self.limits[1],
            self.limits[2],
            self.limits[3],
        )
    }
    fn budget(&self) -> Result<SemanticJournalBudget, String> {
        SemanticJournalBudget::new(
            usize::try_from(self.budget[0]).map_err(|_| "Attempt cap")?,
            self.budget[1],
            self.budget[2],
            self.budget[3],
            self.budget[4],
        )
    }
    fn load_prepared(&self) -> Result<LoadedInitialization, String> {
        self.composition()?;
        let dataset_directory = if self.initialization_kind.is_some() {
            let path = Path::new(&self.collection.path);
            if path.file_name().and_then(|name| name.to_str()) != Some("manifest.json") {
                return Err(
                    "Native initialization collection must pin dataset manifest.json".into(),
                );
            }
            Some(path.parent().ok_or("Native dataset manifest parent")?)
        } else {
            None
        };
        pinned(&self.checkpoint, 32 * 1024 * 1024)?;
        pinned(&self.collection, 16 * 1024 * 1024)?;
        let result = match dataset_directory {
            None => PreparedConditionalHumanPolicy::load_semantic_playing(
                Path::new(&self.checkpoint.path),
                Path::new(&self.collection.path),
            )
            .map(LoadedInitialization::Human),
            Some(directory) => {
                PreparedSemanticPlayingTeacher::load(Path::new(&self.checkpoint.path), directory)
                    .map(|prepared| LoadedInitialization::Native(Arc::new(prepared)))
            }
        };
        // Both raw pins remain checked even when the selected typed loader refuses.
        let post = pinned(&self.checkpoint, 32 * 1024 * 1024)
            .and_then(|_| pinned(&self.collection, 16 * 1024 * 1024));
        match (result, post) {
            (Ok(prepared), Ok(_)) => Ok(prepared),
            (Err(error), Ok(_)) => Err(error),
            (result, Err(post)) => Err(format!(
                "{}; initialization source POST: {post}",
                result.err().unwrap_or_default()
            )),
        }
    }
    fn fresh<'a>(
        &self,
        prepared: LoadedInitializationRef<'_>,
        external: &'a ConditionalNativeExclusions,
        history: &'a DeclaredSamplingHistory,
    ) -> Result<SemanticPlayingJointSession<'a>, String> {
        SemanticPlayingJointSession::from_fresh_owner(
            prepared.fresh_owner(self.critic_seed)?,
            external,
            history,
            SamplingSeed::new(self.sampling_seed),
            self.first_ordinal,
            self.replicate,
            OneStepConfig::from_checkpoint(self.actor_config.clone())?,
            ResidualFitConfig::from_checkpoint(self.critic_config.clone())?,
            self.composition()?,
        )
    }
}
#[derive(Deserialize, Serialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct Prior {
    external: External,
    history: History,
    // Unknown historical replicates are not filled with zero: forbid the seed.
    forbidden_sampling_seeds: Vec<u64>,
}
#[derive(Deserialize, Serialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct Restore {
    tip: ContentPin,
    close_declaration: ContentPin,
}
#[derive(Deserialize, Serialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct Chunk {
    parent: checkpoint::Input,
    sampling_seed: u64,
    replicate: u64,
    first_ordinal: u64,
    budget: [u64; 5],
    prior: Prior,
}
impl Chunk {
    fn budget(&self) -> Result<SemanticJournalBudget, String> {
        SemanticJournalBudget::new(
            usize::try_from(self.budget[0]).map_err(|_| "Chunk attempts")?,
            self.budget[1],
            self.budget[2],
            self.budget[3],
            self.budget[4],
        )
    }
}
fn source_prior<'a>(
    original: &'a Original,
    chunk: Option<&'a Chunk>,
) -> (&'a External, &'a History) {
    chunk.map_or((&original.external, &original.history), |c| {
        (&c.prior.external, &c.prior.history)
    })
}
fn source_budget(
    original: &Original,
    chunk: Option<&Chunk>,
) -> Result<SemanticJournalBudget, String> {
    chunk.map_or_else(|| original.budget(), Chunk::budget)
}
fn source_session<'a>(
    original: &Original,
    chunk: Option<&Chunk>,
    prepared: LoadedInitializationRef<'_>,
    external: &'a ConditionalNativeExclusions,
    history: &'a DeclaredSamplingHistory,
) -> Result<SemanticPlayingJointSession<'a>, String> {
    if let Some(chunk) = chunk {
        checkpoint::resume(original, chunk, prepared, external, history)
    } else {
        original.fresh(prepared, external, history)
    }
}

#[derive(Deserialize, Serialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct AuditWire {
    schema: String,
    journal_directory: String,
    original: Original,
    procedural_pool: ContentPin,
    restore: Restore,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    chunk: Option<Chunk>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    checkpoint_export: Option<checkpoint::Export>,
}

/// Separate restore-only input. It admits no schedule, allocation, new prior,
/// capture request or empty start/resume invocation. Original source pins and
/// the producing-close declaration still pass the existing cold restore gates.
pub struct SemanticJournalAuditSpec {
    wire: AuditWire,
    checksum: String,
}
impl SemanticJournalAuditSpec {
    pub fn from_json(raw: &[u8], expected_sha256: &str) -> Result<Self, String> {
        if !checkpoint_digest(expected_sha256) || crate::model::digest(raw) != expected_sha256 {
            return Err("Semantic audit control pin differs".into());
        }
        let wire: AuditWire =
            serde_json::from_value(unique_json(raw, MAX_CONTROL)?).map_err(|e| e.to_string())?;
        if wire.schema != AUDIT_SCHEMA || !Path::new(&wire.journal_directory).is_absolute() {
            return Err("Semantic audit schema/absolute journal differs".into());
        }
        wire.original.limits()?;
        wire.original.budget()?;
        if let Some(chunk) = &wire.chunk {
            chunk.budget()?;
        }
        OneStepConfig::from_checkpoint(wire.original.actor_config.clone())?;
        ResidualFitConfig::from_checkpoint(wire.original.critic_config.clone())?;
        Ok(Self {
            wire,
            checksum: expected_sha256.into(),
        })
    }
}

/// Descriptive replay result only: no owner, parameters, Ready token or new
/// learning admission escapes. Historical updates/draws are freshly recomputed,
/// not newly allocated. An outer observer measures CPU/RSS and actual OS close.
#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct SemanticJournalAudit {
    schema: &'static str,
    control_sha256: String,
    declared_producing_close: ContentPin,
    head: ContentPin,
    status: SemanticJournalStatus,
    complete_session_consumption: Value,
    archive_inventory: Vec<ContentPin>,
    capture_ids: Vec<String>,
    historical_archive_bytes_verified: bool,
    new_families: usize,
    new_actor_streams: usize,
    new_logical_updates: usize,
    actual_process_close_verified: bool,
    source_authenticated: bool,
    source_qualified: bool,
    verified_native_policy: bool,
    strength_claimed: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    checkpoint: Option<ContentPin>,
}

/// Replays an existing successful producing tip from its genuinely loaded
/// original checked initialization and fresh critic. Writes only a fresh, separate audit
/// report; the journal, its budget, burns and admitted lineage are not modified.
/// A saved report alone is insufficient: publication/POST errors return Err and
/// the host must independently require exit0/EOF/pins. No process is observed here.
pub fn audit_semantic_playing_journal(
    spec: SemanticJournalAuditSpec,
    output: &Path,
) -> Result<SemanticJournalAudit, String> {
    let w = &spec.wire;
    let output = read_only_output(w, output)?;
    let directory = local(Path::new(&w.journal_directory), false)?;
    let (source_external, source_history) = source_prior(&w.original, w.chunk.as_ref());
    let external = source_external.load()?;
    let history = source_history.load()?;
    let pool = ConditionalFrozenPool::from_json(&pinned(&w.procedural_pool, MAX_CONTROL)?)?;
    pool.procedural_binding()?;
    let prepared = w.original.load_prepared()?;
    let result = (|| {
        let restored = restore_observed_closed(
            &directory,
            &w.restore.tip,
            &w.restore.close_declaration,
            source_session(
                &w.original,
                w.chunk.as_ref(),
                prepared.borrowed(),
                &external,
                &history,
            )?,
            &[&pool],
        )?;
        let (journal, snapshots) = restored.into_journal_and_snapshots();
        if journal.limits != w.original.limits()?
            || bounded(&journal.budget)? != bounded(&source_budget(&w.original, w.chunk.as_ref())?)?
        {
            return Err("Semantic audit original limits/budget differ".into());
        }
        let mut report = audit_report(
            spec.checksum,
            &w.restore.close_declaration,
            &journal,
            &snapshots,
        )?;
        fs::create_dir(&output).map_err(|e| e.to_string())?;
        report.checkpoint = checkpoint::export(w, &journal, &output)?;
        let raw = bounded(&report)?;
        publish_mixed_metadata(&output, "audit.json", &raw)?;
        // The read-only report is not authority. Preserve late errors rather
        // than returning a success after changed archive/source evidence.
        journal.verify_archive_bytes()?;
        journal.session.journal_recheck()?;
        Ok(report)
    })();
    let post = prepared
        .recheck_sources()
        .and_then(|_| external.recheck())
        .and_then(|_| history.recheck())
        .and_then(|_| pinned(&w.procedural_pool, MAX_CONTROL).map(|_| ()))
        .and_then(|_| pinned(&w.restore.close_declaration, META_BYTES).map(|_| ()))
        .and_then(|_| pool.procedural_binding().map(|_| ()));
    let post = post.and_then(|_| checkpoint::recheck(w.chunk.as_ref()));
    let post = post.and_then(|_| checkpoint::recheck_export(w.checkpoint_export.as_ref()));
    match (result, post) {
        (Ok(report), Ok(())) => Ok(report),
        (Err(error), Ok(())) => Err(error),
        (result, Err(post)) => Err(format!(
            "{}; audit source POST: {post}",
            result.err().unwrap_or_default()
        )),
    }
}

fn read_only_output(w: &AuditWire, output: &Path) -> Result<PathBuf, String> {
    if !output.is_absolute() {
        return Err("Semantic audit output must be absolute".into());
    }
    let directory = local(Path::new(&w.journal_directory), false)?;
    let output = local(output, true)?;
    if !output.is_absolute()
        || output.to_string_lossy().len() > 1024
        || directory.to_string_lossy().len() > 1024
        || fs::symlink_metadata(&output).is_ok()
    {
        return Err("Semantic audit needs a fresh bounded output namespace".into());
    }
    // Compare existing canonical parents too: Windows path spelling/case must
    // not allow an output child to be created inside the archived journal.
    let canonical_directory = fs::canonicalize(&directory).map_err(|e| e.to_string())?;
    let canonical_output = fs::canonicalize(output.parent().ok_or("Audit output parent")?)
        .map_err(|e| e.to_string())?
        .join(output.file_name().ok_or("Audit output name")?);
    if canonical_output.starts_with(&canonical_directory)
        || canonical_directory.starts_with(&canonical_output)
    {
        return Err("Semantic audit output overlaps journal".into());
    }
    reject_input_namespace(
        &output,
        &serde_json::to_value(w).map_err(|e| e.to_string())?,
    )?;
    Ok(output)
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
enum EvaluationReference {
    OriginalHuman4,
    ResourceCycleV5,
}
#[derive(Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct EvaluationWire {
    schema: String,
    audit_control: ContentPin,
    global_external: External,
    players: usize,
    seeds: Vec<u32>,
    bootstrap_seed: u64,
    reference: EvaluationReference,
}
pub struct SemanticJournalEvaluationSpec {
    wire: EvaluationWire,
    checksum: String,
}
impl SemanticJournalEvaluationSpec {
    pub fn from_json(raw: &[u8], expected_sha256: &str) -> Result<Self, String> {
        if !checkpoint_digest(expected_sha256) || crate::model::digest(raw) != expected_sha256 {
            return Err("Semantic evaluation control pin differs".into());
        }
        let wire: EvaluationWire =
            serde_json::from_value(unique_json(raw, MAX_CONTROL)?).map_err(|e| e.to_string())?;
        if wire.schema != EVALUATION_SCHEMA {
            return Err("Semantic evaluation schema differs".into());
        }
        evaluation_config(&wire, Vec::new()).validate()?;
        Ok(Self {
            wire,
            checksum: expected_sha256.into(),
        })
    }
}
fn evaluation_config(
    w: &EvaluationWire,
    families: Vec<String>,
) -> crate::arena::numeric::NumericArenaConfig {
    crate::arena::numeric::NumericArenaConfig {
        schema: crate::arena::numeric::SCHEMA.into(),
        players: w.players,
        seeds: w.seeds.clone(),
        bootstrap_seed: w.bootstrap_seed,
        external_exclusions: crate::arena::numeric::ExternalExclusions {
            evidence_sha256: w.global_external.pin.sha256.clone(),
            excluded_family_ids: families,
        },
    }
}
fn load_evaluation_config(
    w: &EvaluationWire,
) -> Result<
    (
        ConditionalNativeExclusions,
        crate::arena::numeric::NumericArenaConfig,
    ),
    String,
> {
    let raw = pinned(&w.global_external.pin, 32 * 1024 * 1024)?;
    let global = w.global_external.load()?;
    let mut wire = unique_json(&raw, 32 * 1024 * 1024)?;
    let families = serde_json::from_value(wire["families"].take()).map_err(|e| e.to_string())?;
    let config = evaluation_config(w, families);
    config.validate()?;
    Ok((global, config))
}
#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct SemanticJournalEvaluation {
    schema: &'static str,
    control_sha256: String,
    restoration: SemanticJournalAudit,
    global_external: crate::conditional_human_rl_update::NativeExclusionDescriptor,
    arena: crate::arena::semantic_snapshot::SemanticPlayingSnapshotArenaReport,
}
impl SemanticJournalEvaluation {
    pub fn exit_code(&self) -> i32 {
        if self.arena.paired.statistics.failed_games == 0
            && self.arena.paired.blocks.iter().all(|b| b.complete)
        {
            0
        } else {
            1
        }
    }
}
fn evaluation_post<T>(
    result: Result<T, String>,
    checks: impl IntoIterator<Item = Result<(), String>>,
) -> Result<T, String> {
    let errors = checks
        .into_iter()
        .filter_map(Result::err)
        .collect::<Vec<_>>();
    if errors.is_empty() {
        result
    } else {
        Err(format!(
            "{}; evaluation POST: {}",
            result.err().unwrap_or_default(),
            errors.join("; ")
        ))
    }
}
/// Reconstruct the genuine current actor, evaluate fixed paired arms, then
/// drop it. No owner/capture/stream/update is published or allocated.
pub fn evaluate_semantic_playing_journal(
    spec: SemanticJournalEvaluationSpec,
    output: &Path,
) -> Result<SemanticJournalEvaluation, String> {
    use crate::arena::{numeric, resource_cycle::ResourceCycleConfig, semantic_snapshot};
    let e = &spec.wire;
    let raw = pinned(&e.audit_control, MAX_CONTROL)?;
    let audit = SemanticJournalAuditSpec::from_json(&raw, &e.audit_control.sha256)?;
    if audit.wire.checkpoint_export.is_some() {
        return Err("Evaluation does not admit a checkpoint export request".into());
    }
    let w = &audit.wire;
    if matches!(e.reference, EvaluationReference::OriginalHuman4)
        && w.original.initialization_kind.is_some()
    {
        return Err("OriginalHuman4 requires a Human initialization".into());
    }
    let output = read_only_output(w, output)?;
    reject_input_namespace(
        &output,
        &serde_json::to_value((&e.audit_control, &e.global_external)).map_err(|e| e.to_string())?,
    )?;
    let (global, config) = load_evaluation_config(e)?;
    let (source_external, source_history) = source_prior(&w.original, w.chunk.as_ref());
    let external = source_external.load()?;
    let history = source_history.load()?;
    let pool = ConditionalFrozenPool::from_json(&pinned(&w.procedural_pool, MAX_CONTROL)?)?;
    pool.procedural_binding()?;
    let prepared = w.original.load_prepared()?;
    let result = (|| {
        let original = unique_json(
            &pinned(&w.original.external.pin, 32 * 1024 * 1024)?,
            32 * 1024 * 1024,
        )?;
        if original["families"]
            .as_array()
            .ok_or("Original family closure")?
            .iter()
            .any(|f| f.as_str().is_none_or(|f| !global.contains_family(f)))
        {
            return Err("Evaluation GLOBAL omits original external families".into());
        }
        let restored = restore_observed_closed(
            Path::new(&w.journal_directory),
            &w.restore.tip,
            &w.restore.close_declaration,
            source_session(
                &w.original,
                w.chunk.as_ref(),
                prepared.borrowed(),
                &external,
                &history,
            )?,
            &[&pool],
        )?;
        let (journal, snapshots) = restored.into_journal_and_snapshots();
        let result = (|| {
            if journal.limits != w.original.limits()?
                || bounded(&journal.budget)?
                    != bounded(&source_budget(&w.original, w.chunk.as_ref())?)?
                || journal
                    .session
                    .consumed_families()
                    .iter()
                    .any(|f| !global.contains_family(f))
            {
                return Err("Evaluation original limits/budget or GLOBAL burns differ".into());
            }
            let restoration = audit_report(
                audit.checksum,
                &w.restore.close_declaration,
                &journal,
                &snapshots,
            )?;
            let arena =
                journal.with_evaluation_snapshot(|snapshot| {
                    let reference = match e.reference {
                    EvaluationReference::OriginalHuman4 => {
                        semantic_snapshot::SemanticPlayingSnapshotReference::SemanticHuman(Box::new(
                            crate::arena::conditional::SemanticPlayingConditionalPolicy::new(
                                prepared.human()?,
                                crate::arena::conditional::ConditionalSetup::CornFirst,
                                crate::public_trade_guard::TradeGuardConfig::default(),
                            )?,
                        ))
                    }
                    EvaluationReference::ResourceCycleV5 => {
                        semantic_snapshot::SemanticPlayingSnapshotReference::ResourceCycleV5
                    }
                };
                    numeric::run_semantic_playing_snapshot_validation(
                        &config,
                        semantic_snapshot::SemanticPlayingSnapshotArenaPolicy::new(snapshot)?,
                        reference,
                        (0..config.players)
                            .map(|_| numeric::Opponent::ResourceCycle(ResourceCycleConfig::v5()))
                            .collect(),
                    )
                })?;
            let report = SemanticJournalEvaluation {
                schema: "tzolkin-semantic-playing-journal-evaluation-report-v1",
                control_sha256: spec.checksum,
                restoration,
                global_external: global.descriptor().clone(),
                arena,
            };
            fs::create_dir(&output).map_err(|e| e.to_string())?;
            crate::model::write_new_json_bounded(
                &output.join("evaluation.json"),
                &report,
                numeric::MAX_REPORT_BYTES,
            )?;
            Ok(report)
        })();
        evaluation_post(result, [journal.evaluation_recheck()])
    })();
    evaluation_post(
        result,
        [
            prepared.recheck_sources(),
            external.recheck(),
            history.recheck(),
            global.recheck(),
            pinned(&w.original.checkpoint, 32 * 1024 * 1024).map(|_| ()),
            pinned(&w.original.collection, META_BYTES).map(|_| ()),
            pinned(&e.audit_control, MAX_CONTROL).map(|_| ()),
            pinned(&w.procedural_pool, MAX_CONTROL).map(|_| ()),
            pinned(&w.restore.close_declaration, META_BYTES).map(|_| ()),
            pool.procedural_binding().map(|_| ()),
            checkpoint::recheck(w.chunk.as_ref()),
        ],
    )
}

pub(super) fn audit_report(
    checksum: String,
    close: &ContentPin,
    journal: &SemanticPlayingJournal<'_>,
    snapshots: &[FrozenSemanticSnapshot<'_>],
) -> Result<SemanticJournalAudit, String> {
    journal.verify_archive_bytes()?;
    journal.session.journal_recheck()?;
    Ok(SemanticJournalAudit {
        schema: "tzolkin-semantic-playing-journal-audit-report-v1",
        control_sha256: checksum,
        declared_producing_close: close.clone(),
        head: journal.head(),
        status: journal.status(),
        complete_session_consumption: journal.session.capture_ledger(),
        archive_inventory: journal
            .members
            .values()
            .map(|p| p.absolute(&journal.directory))
            .collect(),
        capture_ids: snapshots.iter().map(|s| s.binding().to_string()).collect(),
        historical_archive_bytes_verified: true,
        new_families: 0,
        new_actor_streams: 0,
        new_logical_updates: 0,
        actual_process_close_verified: false,
        source_authenticated: false,
        source_qualified: false,
        verified_native_policy: false,
        strength_claimed: false,
        checkpoint: None,
    })
}
#[derive(Deserialize, Serialize)]
#[serde(tag = "kind", rename_all = "camelCase", deny_unknown_fields)]
enum SnapshotRef {
    Existing { checksum: String },
    Local { alias: String },
}
#[derive(Deserialize, Serialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct Slot {
    member: usize,
    actor: usize,
    snapshot: SnapshotRef,
}
#[derive(Deserialize, Serialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct Batch {
    seeds: [u32; 4],
    // Capture the genuine current actor at this closed boundary, once.
    capture_before: Option<String>,
    slots: Vec<Slot>,
}
#[derive(Deserialize, Serialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct Wire {
    schema: String,
    mode: String,
    journal_directory: String,
    original: Original,
    procedural_pool: ContentPin,
    restore: Option<Restore>,
    global_prior: Prior,
    permanent_allocation: ContentPin,
    batches: Vec<Batch>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    chunk: Option<Chunk>,
    #[serde(default, skip_serializing_if = "disabled")]
    checkpoint_on_completion: bool,
    /// Advisory invocation output only; excluded from the allocation/learning plan.
    #[serde(default, skip_serializing_if = "disabled")]
    signal_diagnostics: bool,
}
fn disabled(value: &bool) -> bool {
    !value
}

/// Closed pinned control; no learner, snapshot or raw parameter constructor.
pub struct SemanticJournalRunSpec {
    wire: Wire,
    checksum: String,
}
impl SemanticJournalRunSpec {
    pub fn from_json(raw: &[u8], expected_sha256: &str) -> Result<Self, String> {
        if !checkpoint_digest(expected_sha256) || crate::model::digest(raw) != expected_sha256 {
            return Err("Semantic run control pin differs".into());
        }
        let wire: Wire =
            serde_json::from_value(unique_json(raw, MAX_CONTROL)?).map_err(|e| e.to_string())?;
        if wire.schema != RUN_SCHEMA
            || !matches!(wire.mode.as_str(), "start" | "resume" | "continue")
            || (wire.mode == "resume") != wire.restore.is_some()
            || wire.mode == "continue" && wire.chunk.is_none()
            || wire.mode == "start" && wire.chunk.is_some()
            || wire.batches.is_empty()
            || wire.batches.len() > 64
            || wire.global_prior.forbidden_sampling_seeds.len() > 8192
            || wire
                .global_prior
                .forbidden_sampling_seeds
                .windows(2)
                .any(|p| p[0] >= p[1])
        {
            return Err("Semantic finite run mode/schedule/prior differs".into());
        }
        wire.original.limits()?;
        wire.original.budget()?;
        if let Some(chunk) = &wire.chunk {
            chunk.budget()?;
            if wire.mode == "continue" && bounded(&chunk.prior)? != bounded(&wire.global_prior)? {
                return Err("New chunk prior must equal the complete current GLOBAL PRIOR".into());
            }
        }
        if wire.chunk.is_some() && !wire.checkpoint_on_completion
            || wire.checkpoint_on_completion
                && (!wire.batches.len().is_multiple_of(5)
                    || wire.batches.iter().enumerate().any(|(i, b)| {
                        (i % 5 == 0) != b.capture_before.is_some()
                            || b.slots
                                .iter()
                                .any(|s| matches!(&s.snapshot, SnapshotRef::Existing { .. }))
                    }))
        {
            return Err(
                "Checkpoint chunks require complete five-batch local capture blocks".into(),
            );
        }
        OneStepConfig::from_checkpoint(wire.original.actor_config.clone())?;
        ResidualFitConfig::from_checkpoint(wire.original.critic_config.clone())?;
        let mut aliases = BTreeSet::new();
        for batch in &wire.batches {
            if batch.slots.len() > 10
                || batch
                    .slots
                    .windows(2)
                    .any(|p| (p[0].member, p[0].actor) >= (p[1].member, p[1].actor))
            {
                return Err("Semantic sorted unique roster slots differ".into());
            }
            if let Some(alias) = &batch.capture_before
                && (!valid_alias(alias) || !aliases.insert(alias.clone()))
            {
                return Err("Semantic capture alias differs/repeats".into());
            }
            for slot in &batch.slots {
                if slot.member >= 4 || slot.actor >= if slot.member < 2 { 3 } else { 4 } {
                    return Err("Semantic fixed roster member/seat differs".into());
                }
                match &slot.snapshot {
                    SnapshotRef::Existing { checksum } if checkpoint_digest(checksum) => {}
                    SnapshotRef::Local { alias } if aliases.contains(alias) => {}
                    _ => return Err("Semantic snapshot reference is unknown or forward".into()),
                }
            }
        }
        Ok(Self {
            wire,
            checksum: expected_sha256.into(),
        })
    }
    pub fn command(&self) -> &str {
        &self.wire.mode
    }
}

/// Complete descriptive return, including on errors after the journal was opened.
/// Serialized consumption and a successful library return do not observe an OS.
#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct SemanticJournalInvocation {
    schema: &'static str,
    control_sha256: String,
    disposition: String,
    error: Option<String>,
    head: ContentPin,
    status: SemanticJournalStatus,
    complete_session_consumption: Value,
    archive_inventory: Vec<ContentPin>,
    capture_aliases: BTreeMap<String, String>,
    capture_identities: Vec<Value>,
    returned_closes: Vec<ContentPin>,
    last_library_return: Option<SemanticJournalReturnedClose>,
    last_returned_close: Option<ContentPin>,
    declared_permanent_allocation: ContentPin,
    declared_families: usize,
    declared_actor_streams: usize,
    historical_archive_bytes_checked_at_return: bool,
    actual_process_close_verified: bool,
    source_authenticated: bool,
    source_qualified: bool,
    verified_native_policy: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    checkpoint: Option<ContentPin>,
    #[serde(skip_serializing_if = "Option::is_none")]
    signal_diagnostics: Option<ContentPin>,
}
impl SemanticJournalInvocation {
    pub fn exit_code(&self) -> i32 {
        match self.disposition.as_str() {
            "completed" => 0,
            "stoppedNoChange" => 2,
            _ => 1,
        }
    }
}

/// Loads only the declared original source through its genuine Human4 loader.
/// Resume replays before NEW work is admitted. A full fixed allocation is checked
/// first, but is not confused with the subset actually burned by this session.
pub fn run_semantic_playing_journal(
    spec: SemanticJournalRunSpec,
    output: &Path,
) -> Result<SemanticJournalInvocation, String> {
    let w = &spec.wire;
    let new_directory = w.mode != "resume";
    let directory = local(Path::new(&w.journal_directory), new_directory)?;
    let output = local(output, true)?;
    if output == directory
        || output.starts_with(&directory)
        || directory.starts_with(&output)
        || output.to_string_lossy().len() > 1024
        || directory.to_string_lossy().len() > 1024
        || fs::symlink_metadata(&output).is_ok()
    {
        return Err("Semantic invocation needs a separate fresh output namespace".into());
    }
    if new_directory && fs::symlink_metadata(&directory).is_ok()
        || w.mode == "resume"
            && !fs::metadata(&directory)
                .map_err(|e| e.to_string())?
                .is_dir()
        || !output.parent().is_some_and(Path::is_dir)
    {
        return Err("Semantic journal/invocation directory state differs".into());
    }
    let source = serde_json::to_value(w).map_err(|e| e.to_string())?;
    reject_input_namespace(&output, &source)?;
    reject_input_namespace(
        &directory,
        &json!({"original":w.original,
        "pool":w.procedural_pool,"allocation":w.permanent_allocation,"chunk":w.chunk,
        "closeDeclaration":w.restore.as_ref().map(|r|&r.close_declaration)}),
    )?;
    let (source_external, source_history) = source_prior(&w.original, w.chunk.as_ref());
    let original_external = source_external.load()?;
    let original_history = source_history.load()?;
    let prior = w.global_prior.external.load()?;
    let prior_history = w.global_prior.history.load()?;
    check_original_prior(w, &prior)?;
    if let Some(chunk) = &w.chunk {
        check_declared_prior(&chunk.prior, &w.global_prior, &prior)?;
    }
    let pool_raw = pinned(&w.procedural_pool, MAX_CONTROL)?;
    let pool = ConditionalFrozenPool::from_json(&pool_raw)?;
    pool.procedural_binding()?; // Explicitly refuse neural/search legacy slots.
    let allocation_raw = pinned(&w.permanent_allocation, MAX_CONTROL)?;
    // This is a plan checksum, not a control-pin cycle or an authentication claim.
    let plan_checksum = plan_checksum(w)?;
    let limits = w.original.limits()?;
    let expected_first = match &w.restore {
        None => w
            .chunk
            .as_ref()
            .map_or(w.original.first_ordinal, |c| c.first_ordinal),
        Some(_) => allocation_first(&allocation_raw)?,
    };
    let expected = allocation(w, &plan_checksum, expected_first, limits)?;
    if unique_json(&allocation_raw, MAX_CONTROL)? != expected {
        return Err("Semantic complete permanent-allocation declaration differs".into());
    }
    check_new(w, expected_first, &prior, &prior_history, limits)?;
    check_new(
        w,
        expected_first,
        &original_external,
        &original_history,
        limits,
    )?;
    // All control/allocation/prior checks above precede model loading or Game.
    let prepared = w.original.load_prepared()?;
    let fresh = source_session(
        &w.original,
        w.chunk.as_ref(),
        prepared.borrowed(),
        &original_external,
        &original_history,
    )?;
    let (mut journal, mut snapshots) = if let Some(restore) = &w.restore {
        restore_observed_closed(
            &directory,
            &restore.tip,
            &restore.close_declaration,
            fresh,
            &[&pool],
        )?
        .into_journal_and_snapshots()
    } else {
        (
            SemanticPlayingJournal::start_chunk(
                &directory,
                fresh,
                limits,
                source_budget(&w.original, w.chunk.as_ref())?,
            )?,
            Vec::new(),
        )
    };
    if journal.limits != limits
        || bounded(&journal.budget)? != bounded(&source_budget(&w.original, w.chunk.as_ref())?)?
        || journal.session.next_episode_ordinal() != expected_first
    {
        return Err("Semantic restored original limits/budget/next cursor differ".into());
    }
    check_prior(
        &journal,
        &prior,
        &prior_history,
        &w.global_prior.forbidden_sampling_seeds,
    )?;
    if journal.session.attempts() + w.batches.len() > journal.budget.max_attempts {
        return Err("Semantic complete invocation exceeds original attempt headroom".into());
    }
    // Missing restored capture IDs and learner-seat replacements refuse before a new Reserve.
    let rosters = pool.freeze_procedural()?;
    for slot in w.batches.iter().flat_map(|b| &b.slots) {
        if slot.actor == rosters[slot.member].learner_seat() {
            return Err("Snapshot cannot replace learner".into());
        }
        if let SnapshotRef::Existing { checksum } = &slot.snapshot
            && !snapshots.iter().any(|s| s.binding() == checksum)
        {
            return Err("Requested capture was not actually restored".into());
        }
    }
    fs::create_dir(&output).map_err(|e| e.to_string())?;
    let mut aliases = BTreeMap::new();
    let mut returns = Vec::new();
    let mut signal_rows = Vec::new();
    let mut last_library_return = None;
    let mut last_returned_pin = None;
    // Lookup metadata is cached once from genuine captures. Warm alias lookup
    // must not hash every old snapshot's complete origin closure per batch.
    let mut capture_identities = snapshots
        .iter()
        .map(|s| s.journal_identity())
        .collect::<Result<Vec<_>, _>>()?;
    let result = (|| {
        for batch in &w.batches {
            recheck_inputs(w, &prior, &prior_history)?;
            if let Some(alias) = &batch.capture_before {
                let current = journal.session.current_unchecked().checksum();
                let existing = capture_identities
                    .iter()
                    .find(|s| s["current"].as_str() == Some(current));
                let id = match existing {
                    Some(identity) => identity["checksum"]
                        .as_str()
                        .ok_or("Typed capture checksum")?
                        .to_string(),
                    None => {
                        let snapshot = journal.freeze_snapshot()?;
                        let identity = snapshot.journal_identity()?;
                        let id = snapshot.binding().to_string();
                        snapshots.push(snapshot);
                        capture_identities.push(identity);
                        id
                    }
                };
                aliases.insert(alias.clone(), id);
            }
            let mut close = if batch.slots.is_empty() {
                journal.update_procedural_whole_four(batch.seeds, &pool)?
            } else {
                let slots = batch
                    .slots
                    .iter()
                    .map(|slot| {
                        let id = match &slot.snapshot {
                            SnapshotRef::Existing { checksum } => checksum,
                            SnapshotRef::Local { alias } => {
                                aliases.get(alias).ok_or("Capture alias not produced")?
                            }
                        };
                        let snapshot = snapshots
                            .iter()
                            .find(|s| s.binding() == id)
                            .ok_or("Actual capture missing")?;
                        SnapshotSlot::new(slot.member, slot.actor, snapshot)
                    })
                    .collect::<Result<Vec<_>, String>>()?;
                let mixture = SemanticFrozenMixture::new(&pool, slots)?;
                journal.update_mixed_whole_four(batch.seeds, &mixture)?
            };
            let signals = close.signals.take();
            let signal_report = close.report.clone();
            let no_change = close.resolution == SemanticJournalResolution::NoChange;
            let name = format!("returned-{:04}.json", close.attempt);
            last_library_return = Some(close);
            last_returned_pin = None;
            let raw = bounded(&last_library_return)?;
            publish_mixed_metadata(&output, &name, &raw)?;
            let pin = Pin::new(&name, &raw).absolute(&output);
            last_returned_pin = Some(pin.clone());
            returns.push(pin.clone());
            if w.signal_diagnostics {
                let signals = signals.ok_or("Returned semantic close lacks same-pass signals")?;
                signal_rows.push(json!({"returnedClose":pin,
                    "report":signal_report,"signals":signals}));
            }
            // Source/global-prior/output failures stop before another attempt.
            recheck_inputs(w, &prior, &prior_history)?;
            if no_change {
                return Ok(false);
            }
        }
        Ok(true)
    })();
    let mut error = result.as_ref().err().cloned();
    let mut signal_diagnostics = None;
    if w.signal_diagnostics {
        match publish_signal_diagnostics(&output, &spec.checksum, w.batches.len(), &signal_rows) {
            Ok(pin) => signal_diagnostics = Some(pin),
            Err(post) => {
                error = Some(format!(
                    "{}; signal diagnostics: {post}",
                    error.unwrap_or_default()
                ))
            }
        }
    }
    // Sidecar publication errors still pass through all archive/source POST checks.
    let full_bytes = journal.verify_archive_bytes();
    if let Err(post) = &full_bytes {
        error = Some(format!(
            "{}; final archive: {post}",
            error.unwrap_or_default()
        ));
    }
    if let Err(post) = journal
        .session
        .journal_recheck()
        .and_then(|_| recheck_inputs(w, &prior, &prior_history))
    {
        error = Some(format!(
            "{}; final source/prior: {post}",
            error.unwrap_or_default()
        ));
    }
    let checkpoint = if error.is_none() && result == Ok(true) && w.checkpoint_on_completion {
        match checkpoint::produce(
            w,
            &journal,
            &output,
            last_returned_pin
                .as_ref()
                .ok_or("Checkpoint last close missing")?,
        ) {
            Ok(pin) => Some(pin),
            Err(post) => {
                error = Some(format!("Checkpoint publication: {post}"));
                None
            }
        }
    } else {
        None
    };
    let disposition = if error.is_some() {
        "refused"
    } else if result == Ok(true) {
        "completed"
    } else {
        "stoppedNoChange"
    };
    let report = SemanticJournalInvocation {
        schema: "tzolkin-semantic-playing-journal-invocation-v1",
        control_sha256: spec.checksum,
        disposition: disposition.into(),
        error,
        head: journal.head(),
        status: journal.status(),
        complete_session_consumption: journal.session.capture_ledger(),
        archive_inventory: journal
            .members
            .values()
            .map(|p| p.absolute(&directory))
            .collect(),
        capture_aliases: aliases,
        capture_identities,
        last_returned_close: last_returned_pin,
        last_library_return,
        returned_closes: returns,
        declared_permanent_allocation: w.permanent_allocation.clone(),
        declared_families: 4 * w.batches.len(),
        declared_actor_streams: 14 * w.batches.len(),
        historical_archive_bytes_checked_at_return: full_bytes.is_ok(),
        actual_process_close_verified: false,
        source_authenticated: false,
        source_qualified: false,
        verified_native_policy: false,
        checkpoint,
        signal_diagnostics,
    };
    let raw = bounded(&report)?;
    // On a late IO failure the CLI still receives exact accounting to print.
    if let Err(error) = publish_mixed_metadata(&output, "invocation.json", &raw) {
        return Ok(SemanticJournalInvocation {
            disposition: "refused".into(),
            error: Some(format!("Invocation publication: {error}")),
            ..report
        });
    }
    if let Err(error) = journal
        .session
        .journal_recheck()
        .and_then(|_| recheck_inputs(w, &prior, &prior_history))
    {
        return Ok(SemanticJournalInvocation {
            disposition: "refused".into(),
            error: Some(format!("Invocation postpublication: {error}")),
            ..report
        });
    }
    Ok(report)
}

fn publish_signal_diagnostics(
    output: &Path,
    control: &str,
    planned_batches: usize,
    rows: &[Value],
) -> Result<ContentPin, String> {
    if !(1..=64).contains(&planned_batches) || rows.len() > planned_batches {
        return Err("Semantic signal diagnostics row bound".into());
    }
    let mut value = json!({"schema":"tzolkin-semantic-playing-signal-diagnostics-v1",
        "scope":"samePassReturnedLibraryClosesOnly","controlSha256":control,
        "plannedBatches":planned_batches,"returnedRows":rows.len(),"rows":rows,
        "sourceAuthenticated":false,"strengthMeasured":false,"adoptionGranted":false,"checksum":""});
    value["checksum"] = hash(b"semantic-playing-signal-diagnostics-v1\0", &value)?.into();
    let raw = bounded(&value)?;
    if raw.len() > MAX_CONTROL {
        return Err("Semantic signal diagnostics byte bound".into());
    }
    publish_mixed_metadata(output, "diagnostics.json", &raw)?;
    Ok(Pin::new("diagnostics.json", &raw).absolute(output))
}

fn valid_alias(alias: &str) -> bool {
    !alias.is_empty()
        && alias.len() <= 32
        && alias
            .bytes()
            .all(|c| c.is_ascii_alphanumeric() || c == b'-')
}
fn pinned(pin: &ContentPin, cap: usize) -> Result<Vec<u8>, String> {
    if !Path::new(&pin.path).is_absolute()
        || pin.bytes == 0
        || pin.bytes > cap as u64
        || !checkpoint_digest(&pin.sha256)
    {
        return Err("Semantic runner bounded absolute pin differs".into());
    }
    let raw = read_bounded(&local(Path::new(&pin.path), false)?, pin.bytes as usize)?;
    if raw.len() as u64 != pin.bytes || crate::model::digest(&raw) != pin.sha256 {
        return Err("Semantic runner input changed".into());
    }
    Ok(raw)
}
fn plan_checksum(w: &Wire) -> Result<String, String> {
    let mut plan = json!({"mode":w.mode,"journalDirectory":w.journal_directory,
        "original":w.original,"proceduralPool":w.procedural_pool,"restore":w.restore,"globalPrior":w.global_prior,"batches":w.batches});
    if w.chunk.is_some() || w.checkpoint_on_completion {
        plan["chunk"] = serde_json::to_value(&w.chunk).map_err(|e| e.to_string())?;
        plan["checkpointOnCompletion"] = w.checkpoint_on_completion.into();
    }
    hash(b"semantic-playing-invocation-plan-v1\0", &plan)
}
fn allocation_first(raw: &[u8]) -> Result<u64, String> {
    unique_json(raw, MAX_CONTROL)?
        .get("firstOrdinal")
        .and_then(Value::as_u64)
        .ok_or("Semantic allocation first ordinal".into())
}
fn allocation(w: &Wire, plan: &str, first: u64, limits: CollectionLimits) -> Result<Value, String> {
    let (sampling_seed, replicate) = sampling(w);
    let next = first
        .checked_add(4 * w.batches.len() as u64)
        .ok_or("Semantic invocation ordinal overflow")?;
    let mut members = Vec::new();
    let mut families = BTreeSet::new();
    let mut streams = BTreeSet::new();
    for (batch, b) in w.batches.iter().enumerate() {
        for (member, &seed) in b.seeds.iter().enumerate() {
            let family = seed_family_id(seed);
            if split_for_family(&family)? != DatasetSplit::Train || !families.insert(family.clone())
            {
                return Err("Semantic allocation needs distinct Train families".into());
            }
            let players = if member < 2 { 3 } else { 4 };
            let ordinal = first + 4 * batch as u64 + member as u64;
            let identity = SamplingStreamIdentity::new(
                SamplingSeed::new(sampling_seed),
                ordinal,
                replicate,
                players,
            )?;
            let config = NativeStochasticConfig::new(players, seed, identity, limits)?;
            for actor in 0..players {
                streams.insert([
                    sampling_seed,
                    ordinal,
                    replicate,
                    players as u64,
                    actor as u64,
                ]);
            }
            members.push(json!({"batch":batch,"member":member,"family":family,"config":Config::from_config(&config)}));
        }
    }
    let mut allocation = json!({"schema":ALLOCATION_SCHEMA,"planChecksum":plan,"globalPrior":w.global_prior,
        "priorHead":w.restore.as_ref().map(|r|&r.tip),"permanent":true,"allUnenteredRemainSpent":true,
        "samplingSeed":sampling_seed,"replicate":replicate,"firstOrdinal":first,"nextOrdinal":next,
        "families":families,"actorStreams":streams,"members":members});
    if let Some(chunk) = &w.chunk {
        allocation["parentCheckpoint"] =
            serde_json::to_value(&chunk.parent).map_err(|e| e.to_string())?;
    }
    Ok(allocation)
}
fn sampling(w: &Wire) -> (u64, u64) {
    w.chunk
        .as_ref()
        .map_or((w.original.sampling_seed, w.original.replicate), |c| {
            (c.sampling_seed, c.replicate)
        })
}
fn check_new(
    w: &Wire,
    first: u64,
    prior: &ConditionalNativeExclusions,
    history: &DeclaredSamplingHistory,
    limits: CollectionLimits,
) -> Result<(), String> {
    let (sampling_seed, replicate) = sampling(w);
    if w.global_prior
        .forbidden_sampling_seeds
        .binary_search(&sampling_seed)
        .is_ok()
    {
        return Err("Prospective sampling seed is wholly forbidden".into());
    }
    for (batch, b) in w.batches.iter().enumerate() {
        for (member, &seed) in b.seeds.iter().enumerate() {
            if prior.contains_family(&seed_family_id(seed)) {
                return Err("Allocation intersects GLOBAL PRIOR (including a self-union)".into());
            }
            let players = if member < 2 { 3 } else { 4 };
            let identity = SamplingStreamIdentity::new(
                SamplingSeed::new(sampling_seed),
                first + 4 * batch as u64 + member as u64,
                replicate,
                players,
            )?;
            NativeStochasticConfig::new(players, seed, identity, limits)?;
            if (0..players).any(|actor| history.contains(identity, actor)) {
                return Err("New allocation intersects GLOBAL PRIOR streams".into());
            }
        }
    }
    Ok(())
}
fn check_prior(
    journal: &SemanticPlayingJournal<'_>,
    prior: &ConditionalNativeExclusions,
    history: &DeclaredSamplingHistory,
    forbidden: &[u64],
) -> Result<(), String> {
    if journal
        .session
        .consumed_families()
        .iter()
        .any(|f| !prior.contains_family(f))
    {
        return Err("GLOBAL PRIOR omits earlier journal family burns".into());
    }
    let ledger = journal.session.capture_ledger();
    for row in ledger["streams"]
        .as_array()
        .ok_or("Session stream ledger")?
    {
        let r: [u64; 5] = serde_json::from_value(row.clone()).map_err(|e| e.to_string())?;
        let identity = SamplingStreamIdentity::new(
            SamplingSeed::new(r[0]),
            r[1],
            r[2],
            usize::try_from(r[3]).map_err(|_| "Ledger players")?,
        )?;
        if forbidden.binary_search(&r[0]).is_err() && !history.contains(identity, r[4] as usize) {
            return Err("GLOBAL PRIOR omits earlier journal stream burns".into());
        }
    }
    Ok(())
}
fn check_original_prior(w: &Wire, prior: &ConditionalNativeExclusions) -> Result<(), String> {
    check_source_prior(&w.original, &w.global_prior, prior)
}
fn check_source_prior(
    original: &Original,
    declared: &Prior,
    prior: &ConditionalNativeExclusions,
) -> Result<(), String> {
    // The existing typed loader has already validated this exact pinned wire.
    let original_wire = unique_json(
        &pinned(&original.external.pin, 32 * 1024 * 1024)?,
        32 * 1024 * 1024,
    )?;
    if original_wire["families"]
        .as_array()
        .ok_or("Original family closure")?
        .iter()
        .any(|f| f.as_str().is_none_or(|f| !prior.contains_family(f)))
    {
        return Err("GLOBAL PRIOR omits original external families".into());
    }
    for r in &original.history.ranges {
        if declared
            .forbidden_sampling_seeds
            .binary_search(&r[0])
            .is_ok()
        {
            continue;
        }
        let mut cursor = r[4];
        for q in &declared.history.ranges {
            if q[..4] == r[..4] && q[4] <= cursor && cursor < q[5] {
                cursor = q[5];
            }
        }
        if cursor < r[5] {
            return Err("GLOBAL PRIOR omits original declared sampling history".into());
        }
    }
    Ok(())
}
fn check_declared_prior(
    old: &Prior,
    new: &Prior,
    external: &ConditionalNativeExclusions,
) -> Result<(), String> {
    for prior in [old, new] {
        if prior.forbidden_sampling_seeds.len() > 8192
            || prior
                .forbidden_sampling_seeds
                .windows(2)
                .any(|p| p[0] >= p[1])
        {
            return Err("Chunk forbidden sampling seeds must stay sorted and bounded".into());
        }
        prior.history.load()?;
    }
    let old_external = unique_json(
        &pinned(&old.external.pin, 32 * 1024 * 1024)?,
        32 * 1024 * 1024,
    )?;
    if old_external["families"]
        .as_array()
        .ok_or("Chunk prior families")?
        .iter()
        .any(|f| f.as_str().is_none_or(|f| !external.contains_family(f)))
        || old
            .forbidden_sampling_seeds
            .iter()
            .any(|seed| new.forbidden_sampling_seeds.binary_search(seed).is_err())
    {
        return Err("Chunk prior drops earlier families or unknown sampling seeds".into());
    }
    let history = new.history.load()?;
    for r in &old.history.ranges {
        if new.forbidden_sampling_seeds.binary_search(&r[0]).is_err()
            && !history.covers(
                SamplingSeed::new(r[0]),
                r[1],
                r[2] as usize,
                r[3] as usize,
                r[4],
                r[5],
            )
        {
            return Err("Chunk prior drops earlier declared sampling ranges".into());
        }
    }
    Ok(())
}
fn recheck_inputs(
    w: &Wire,
    prior: &ConditionalNativeExclusions,
    history: &DeclaredSamplingHistory,
) -> Result<(), String> {
    prior.recheck()?;
    history.recheck()?;
    pinned(&w.procedural_pool, MAX_CONTROL)?;
    pinned(&w.permanent_allocation, MAX_CONTROL)?;
    checkpoint::recheck(w.chunk.as_ref())?;
    Ok(())
}

// Called by the existing loaded-origin fixture. These are control/tuple
// assertions, not an execution authorization or an alternative game runner.
#[cfg(test)]
pub(super) fn control_fixture(
    root: &Path,
    limits: CollectionLimits,
    seeds: [u32; 4],
    prepared: &PreparedConditionalHumanPolicy,
    journal: &SemanticPlayingJournal<'_>,
    returned: &SemanticJournalReturnedClose,
) {
    let pin = |path: PathBuf| {
        let raw = fs::read(&path).unwrap();
        ContentPin {
            path: path.to_str().unwrap().into(),
            bytes: raw.len() as u64,
            sha256: crate::model::digest(&raw),
        }
    };
    let external_pin = pin(root.join("external.json"));
    let family = seed_family_id(42);
    let external = json!({"pin":external_pin,"count":1,
        "canonicalDigest":crate::model::digest(format!("{family}\n").as_bytes())});
    let history = json!({"ranges":[[17,0,3,0,90,94]],"pins":[external_pin]});
    // Continuation keeps earlier provenance instead of dropping old allocations.
    // Every newly admitted pin still passes the ordinary byte recheck.
    let provenance = (0..=MAX_HISTORY_PINS)
        .map(|i| {
            let path = root.join(format!("history-pin-{i:03}.json"));
            fs::write(&path, b"synthetic history provenance").unwrap();
            pin(path)
        })
        .collect::<Vec<_>>();
    for count in [65, MAX_HISTORY_PINS] {
        let declared = History {
            ranges: vec![[17, 0, 3, 0, 90, 94]],
            pins: provenance[..count].to_vec(),
        };
        assert!(declared.load().is_ok());
    }
    let admitted = History {
        ranges: vec![[17, 0, 3, 0, 90, 94]],
        pins: provenance[..65].to_vec(),
    };
    fs::write(&provenance[64].path, b"changed provenance").unwrap();
    assert!(admitted.load().is_err());
    fs::write(&provenance[64].path, b"synthetic history provenance").unwrap();
    assert!(matches!(
        History {
            ranges: vec![],
            pins: provenance,
        }
        .load(),
        Err(error) if error == "Runner history cap"
    ));
    let mut value = json!({"schema":RUN_SCHEMA,"mode":"start",
        "journalDirectory":root.join("new-cli-journal"),
        "original":{"checkpoint":pin(root.join("checkpoint/checkpoint.json")),
            "collection":pin(root.join("collection.json")),"external":external,"history":history,
            "criticSeed":11,"samplingSeed":17,"replicate":0,"firstOrdinal":200,
            "actorConfig":OneStepConfig::default(),"criticConfig":ResidualFitConfig::new(0.001,0.01,256).unwrap(),
            "limits":[limits.max_callbacks(),limits.max_candidate_rows(),limits.max_source_bytes(),limits.max_trace_bytes()],
            "budget":[3,1u64<<30,1u64<<30,512u64<<20,2u64<<30]},
        // These pins test closed control parsing only; run() would independently
        // require a genuine pool and exact permanent allocation before loading.
        "proceduralPool":external_pin,"permanentAllocation":external_pin,"restore":null,
        "globalPrior":{"external":external,"history":history,"forbiddenSamplingSeeds":[]},
        "batches":[{"seeds":seeds,"captureBefore":"current","slots":[
            {"member":3,"actor":3,"snapshot":{"kind":"local","alias":"current"}}]}]});
    let parse = |value: &Value| {
        let raw = serde_json::to_vec(value).unwrap();
        SemanticJournalRunSpec::from_json(&raw, &crate::model::digest(&raw))
    };
    let spec = parse(&value).unwrap();
    assert!(!spec.wire.signal_diagnostics);
    let old_wire = serde_json::to_vec(&spec.wire).unwrap();
    assert!(spec.wire.original.initialization_kind.is_none());
    let mut absent_initialization = value.clone();
    absent_initialization["original"]["initializationKind"] = Value::Null;
    assert_eq!(
        serde_json::to_vec(&parse(&absent_initialization).unwrap().wire).unwrap(),
        old_wire
    );
    let mut signal_control = value.clone();
    signal_control["signalDiagnostics"] = false.into();
    assert_eq!(
        serde_json::to_vec(&parse(&signal_control).unwrap().wire).unwrap(),
        old_wire
    );
    signal_control["signalDiagnostics"] = true.into();
    assert_eq!(
        plan_checksum(&parse(&signal_control).unwrap().wire).unwrap(),
        plan_checksum(&spec.wire).unwrap()
    );
    let signals_output = root.join("signal-diagnostics-fixture");
    fs::create_dir(&signals_output).unwrap();
    let rows = [json!({"report":returned.report,"signals":returned.signals.as_ref().unwrap()})];
    let signals_pin =
        publish_signal_diagnostics(&signals_output, &spec.checksum, 1, &rows).unwrap();
    let mut signals = unique_json(&fs::read(&signals_pin.path).unwrap(), MAX_CONTROL).unwrap();
    let checksum = signals["checksum"].take();
    signals["checksum"] = "".into();
    assert_eq!(
        checksum.as_str().unwrap(),
        hash(b"semantic-playing-signal-diagnostics-v1\0", &signals).unwrap()
    );
    assert_eq!(signals["rows"], json!(rows));
    assert_eq!(signals["controlSha256"], spec.checksum);
    let core = unique_json(&fs::read(&returned.report.path).unwrap(), META_BYTES).unwrap();
    assert_eq!(
        signals["rows"][0]["signals"]["reportChecksum"],
        core["checksum"]
    );
    assert_eq!(signals["rows"][0]["signals"]["next"], core["next"]);
    assert!(core.get("signals").is_none());
    assert!(
        serde_json::to_value(returned)
            .unwrap()
            .get("signals")
            .is_none()
    );
    assert!(publish_signal_diagnostics(&signals_output, &spec.checksum, 1, &rows).is_err());
    assert!(publish_signal_diagnostics(&signals_output, &spec.checksum, 65, &rows).is_err());
    assert!(
        serde_json::to_value(&spec.wire.original)
            .unwrap()
            .get("collectionComposition")
            .is_none()
    );
    let mut guarded_control = value.clone();
    guarded_control["original"]["collectionComposition"] = "guardedProposalV2".into();
    let guarded_spec = parse(&guarded_control).unwrap();
    assert_eq!(
        guarded_spec.wire.original.composition().unwrap(),
        SemanticCollectionComposition::GuardedProposalV2
    );
    assert_ne!(
        plan_checksum(&guarded_spec.wire).unwrap(),
        plan_checksum(&spec.wire).unwrap()
    );
    let mut native_control = guarded_control.clone();
    native_control["original"]["initializationKind"] =
        "nativeCornFirstV5SemanticPlayingTeacher".into();
    let native_spec = parse(&native_control).unwrap();
    // The discriminator is part of Original, hence the existing allocation plan hash.
    assert_ne!(
        plan_checksum(&native_spec.wire).unwrap(),
        plan_checksum(&guarded_spec.wire).unwrap()
    );
    assert!(
        native_spec
            .wire
            .original
            .load_prepared()
            .err()
            .unwrap()
            .contains("must pin dataset manifest.json")
    );
    let mut missing_manifest = native_control.clone();
    missing_manifest["original"]
        .as_object_mut()
        .unwrap()
        .remove("collection");
    assert!(parse(&missing_manifest).is_err());
    let mut wrong_initialization = native_control.clone();
    wrong_initialization["original"]["initializationKind"] = "human".into();
    assert!(parse(&wrong_initialization).is_err());
    let mut unguarded_native = native_control.clone();
    unguarded_native["original"]
        .as_object_mut()
        .unwrap()
        .remove("collectionComposition");
    assert!(parse(&unguarded_native).is_err());
    guarded_control["checkpointOnCompletion"] = true.into();
    assert!(parse(&guarded_control).is_err()); // V2 retains the complete five-block rule.
    guarded_control["checkpointOnCompletion"] = false.into();
    guarded_control["original"]["collectionComposition"] = "legacyV1".into();
    assert!(parse(&guarded_control).is_err()); // Legacy mode has one canonical absent spelling.
    let w = &spec.wire;
    let mut audit = json!({"schema":AUDIT_SCHEMA,"journalDirectory":root.join("journal"),
        "original":value["original"],"proceduralPool":external_pin,
        "restore":{"tip":external_pin,"closeDeclaration":external_pin}});
    let parse_audit = |value: &Value| {
        let raw = serde_json::to_vec(value).unwrap();
        SemanticJournalAuditSpec::from_json(&raw, &crate::model::digest(&raw))
    };
    assert!(parse_audit(&audit).is_ok());
    assert!(parse_audit(&value).is_err()); // No empty/current run-mode shortcut.
    let validation = (0..10_000)
        .find(|s| {
            *s != 42 && split_for_family(&seed_family_id(*s)).unwrap() == DatasetSplit::Validation
        })
        .unwrap();
    let test = (0..10_000)
        .find(|s| split_for_family(&seed_family_id(*s)).unwrap() == DatasetSplit::Test)
        .unwrap();
    let mut evaluation = json!({"schema":EVALUATION_SCHEMA,"auditControl":external_pin,
        "globalExternal":external,"players":3,"seeds":[validation],"bootstrapSeed":7,"reference":"originalHuman4"});
    let parse_evaluation = |v: &Value| {
        let raw = serde_json::to_vec(v).unwrap();
        SemanticJournalEvaluationSpec::from_json(&raw, &crate::model::digest(&raw))
    };
    let mut evaluation_spec = parse_evaluation(&evaluation).unwrap(); // No restore authority.
    let mut native_audit = audit.clone();
    native_audit["original"] = native_control["original"].clone();
    assert!(parse_audit(&native_audit).is_ok());
    let native_audit_path = root.join("native-human-reference-audit.json");
    fs::write(
        &native_audit_path,
        serde_json::to_vec(&native_audit).unwrap(),
    )
    .unwrap();
    let mut native_evaluation = evaluation.clone();
    native_evaluation["auditControl"] = serde_json::to_value(pin(native_audit_path)).unwrap();
    let refused_reference = evaluate_semantic_playing_journal(
        parse_evaluation(&native_evaluation).unwrap(),
        &root.join("native-human-reference-output"),
    )
    .err()
    .unwrap();
    assert!(refused_reference.contains("OriginalHuman4 requires a Human initialization"));
    let (_, config) = load_evaluation_config(&evaluation_spec.wire).unwrap();
    assert_eq!(config.external_exclusions.excluded_family_ids, vec![family]);
    evaluation_spec.wire.global_external.count = 2;
    assert!(load_evaluation_config(&evaluation_spec.wire).is_err());
    assert!(parse_evaluation(&audit).is_err());
    assert!(parse_audit(&evaluation).is_err());
    evaluation["reference"] = json!("resourceCycleV5");
    assert!(parse_evaluation(&evaluation).is_ok());
    evaluation["reference"] = json!("rawWeights");
    assert!(parse_evaluation(&evaluation).is_err());
    evaluation["reference"] = json!("originalHuman4");
    evaluation["seeds"] = json!([test]);
    assert!(parse_evaluation(&evaluation).is_err()); // No Test or Development shortcut.
    evaluation["seeds"] = json!([validation, validation]);
    assert!(parse_evaluation(&evaluation).is_err());
    evaluation["seeds"] = json!([validation]);
    evaluation["batches"] = json!([]);
    assert!(parse_evaluation(&evaluation).is_err()); // No allocation or capture fields.
    let error = evaluation_post::<()>(Err("callback".into()), [Err("archive".into())]).unwrap_err();
    assert!(error.contains("callback") && error.contains("archive"));
    audit["batches"] = json!([]);
    assert!(parse_audit(&audit).is_err()); // Even an empty schedule is refused.
    audit.as_object_mut().unwrap().remove("batches");
    audit["original"]["budget"][0] = 0.into();
    assert!(parse_audit(&audit).is_err()); // No budget/refund override.
    audit["original"]["budget"][0] = 3.into();
    audit["restore"] = Value::Null;
    assert!(parse_audit(&audit).is_err()); // A real producing-close input is mandatory.
    let declared = allocation(w, &plan_checksum(w).unwrap(), 200, limits).unwrap();
    let streams = declared["actorStreams"].as_array().unwrap();
    assert_eq!(streams.len(), 14);
    assert_eq!(streams[0], json!([17, 200, 0, 3, 0]));
    assert_eq!(streams[13], json!([17, 203, 0, 4, 3]));
    assert_eq!(declared["nextOrdinal"], 204);
    let prior = w.global_prior.external.load().unwrap();
    let history = w.global_prior.history.load().unwrap();
    check_original_prior(w, &prior).unwrap();
    check_new(w, 200, &prior, &history, limits).unwrap();
    value["batches"][0]["slots"][0]["snapshot"]["alias"] = "future".into();
    assert!(parse(&value).is_err());
    value["batches"][0]["slots"][0]["snapshot"]["alias"] = "current".into();
    value["batches"][0]["seeds"][0] = 42.into();
    let self_union = parse(&value).unwrap();
    assert!(check_new(&self_union.wire, 200, &prior, &history, limits).is_err());

    // Extend the same short loaded fixture: genuine owned numerics plus
    // synthetic local close declarations, never a claim of observing an OS.
    let save = |name: &str, value: &Value| {
        let raw = bounded(value).unwrap();
        fs::write(root.join(name), &raw).unwrap();
        Pin::new(name, &raw).absolute(root)
    };
    let mut producer = spec.wire;
    let ledger = journal.session.capture_ledger();
    producer.permanent_allocation = save(
        "checkpoint-fixture-allocation.json",
        &json!({
        "schema":ALLOCATION_SCHEMA,"permanent":true,"allUnenteredRemainSpent":true,
        "families":ledger["families"],"actorStreams":ledger["streams"]}),
    );
    let output = root.join("checkpoint-fixture-output");
    fs::create_dir(&output).unwrap();
    let returned_pin = save(
        "checkpoint-fixture-returned.json",
        &serde_json::to_value(returned).unwrap(),
    );
    let payload = checkpoint::produce(&producer, journal, &output, &returned_pin).unwrap();
    let process = save(
        "checkpoint-fixture-process.json",
        &json!({
        "schema":PROCESS_CLOSE_SCHEMA,"actualClose":true,"pid":1,"exitCode":0,"signal":null,
        "stdoutEof":true,"stderrEof":true,"allPostchecksPassed":true,
        "observedHead":journal.head(),"returnedClose":returned_pin}),
    );
    let declaration = save(
        "checkpoint-fixture-journal-close.json",
        &json!({
        "schema":CLOSE_DECLARATION_SCHEMA,"observedHead":journal.head(),
        "processClose":process,"returnedClose":returned_pin,"completeConsumptionDeclared":true,
        "consumption":ledger,"status":journal.status(),
        "inventory":journal.members.values().map(|p|p.absolute(&journal.directory)).collect::<Vec<_>>()}),
    );
    let invocation = save(
        "checkpoint-fixture-invocation.json",
        &json!({
        "disposition":"completed","error":null,"checkpoint":payload,"head":journal.head(),
        "status":journal.status(),"completeSessionConsumption":ledger,
        "historicalArchiveBytesCheckedAtReturn":true,"declaredPermanentAllocation":producer.permanent_allocation}),
    );
    let producer_close = |name: &str, payload: &ContentPin, invocation: &ContentPin| {
        save(
            name,
            &json!({
        "schema":"tzolkin-semantic-playing-checkpoint-observed-process-close-v1",
        "actualClose":true,"pid":2,"exitCode":0,"signal":null,"stdoutEof":true,
        "stderrEof":true,"allPostchecksPassed":true,"checkpoint":payload,"invocation":invocation}),
        )
    };
    let producing = producer_close("checkpoint-fixture-producer.json", &payload, &invocation);
    let close = save(
        "checkpoint-fixture-close.json",
        &json!({
        "schema":"tzolkin-semantic-playing-checkpoint-close-v1","checkpoint":payload,
        "journalCloseDeclaration":declaration,"invocation":invocation,"producerClose":producing}),
    );
    let policy_control = json!({"schema":POLICY_EXPORT_SCHEMA,
        "checkpoint":{"payload":payload,"closeDeclaration":close}});
    let parse_policy = |v: &Value| {
        let raw = serde_json::to_vec(v).unwrap();
        SemanticPolicyExportSpec::from_json(&raw, &crate::model::digest(&raw))
    };
    {
        use tzolkin_inference::{deployment, semantic_playing_deployment as policy};
        let output = root.join("checkpoint-fixture-policy");
        let report =
            export_semantic_playing_policy(parse_policy(&policy_control).unwrap(), &output)
                .unwrap();
        let raw = fs::read(output.join("policy.json")).unwrap();
        let encoded: Value = serde_json::from_slice(&raw).unwrap();
        let loaded = policy::LoadedSemanticPlayingDeployment::load(&raw).unwrap();
        let parameters = journal.session.actor_parameters().unwrap();
        assert_eq!(
            encoded["parameterBits"],
            json!(
                parameters
                    .iter()
                    .map(|v| format!("{:08x}", v.to_bits()))
                    .collect::<Vec<_>>()
            )
        );
        assert_eq!(report.artifact_sha256, crate::model::digest(&raw));
        assert_eq!(report.deployment_checksum, loaded.checksum());
        assert_eq!(
            loaded.identity().actor_checksum(),
            journal.session.identity().unwrap().actor_checksum()
        );
        assert!(!std::str::from_utf8(&raw).unwrap().contains("\"path\""));
        assert!(encoded.get("critic").is_none() && encoded.get("original").is_none());
        assert!(deployment::LoadedDeployment::load(&raw).is_err());
        assert!(
            tzolkin_inference::conditional_deployment::LoadedConditionalDeployment::load(&raw)
                .is_err()
        );
        // Only a few real Setup transitions, never a full game or campaign.
        let mut state = tzolkin_core::create_game_with_options(
            (0..3).map(|i| format!("P{i}")).collect(),
            17,
            tzolkin_core::GameOptions::default(),
        )
        .unwrap();
        let setup = tzolkin_core::observation::observe(&state, state.current_player).unwrap();
        assert!(loaded.distribution_playing(&setup).is_err());
        for _ in 0..3 {
            let o = tzolkin_core::observation::observe(&state, state.current_player).unwrap();
            state = tzolkin_core::apply_move(
                &state,
                crate::setup_policy::choose_move_with_weights(
                    &o,
                    &crate::policy::HeuristicWeights::default(),
                )
                .unwrap()
                .r#move,
            )
            .unwrap();
        }
        let o = tzolkin_core::observation::observe(&state, state.current_player).unwrap();
        let arena_policy =
            crate::arena::semantic_snapshot::SemanticPlayingCheckpointArenaPolicy::new(
                policy::LoadedSemanticPlayingDeployment::load(&raw).unwrap(),
            )
            .unwrap();
        arena_policy.fixture_checks(&setup, &o);
        assert_eq!(
            arena_policy.identity().actor_identity["actorChecksum"],
            loaded.identity().actor_checksum(),
        );
        assert!(!arena_policy.identity().source_authenticated);
        assert!(!arena_policy.identity().family_closure_authenticated);
        let artifact = Pin::new("policy.json", &raw).absolute(&output);
        let mut evaluation = json!({"schema":POLICY_EVALUATION_SCHEMA,
            "candidate":artifact,"reference":artifact,"globalExternal":external,
            "players":3,"seeds":[validation],"bootstrapSeed":7});
        let parse = |v: &Value| {
            let bytes = serde_json::to_vec(v).unwrap();
            SemanticPolicyEvaluationSpec::from_json(&bytes, &crate::model::digest(&bytes))
        };
        assert!(parse(&evaluation).is_ok());
        evaluation["seeds"] = json!([test]);
        assert!(parse(&evaluation).is_err());
        evaluation["seeds"] = json!([validation]);
        evaluation["journalDirectory"] = "not-an-owner".into();
        assert!(parse(&evaluation).is_err());
        evaluation
            .as_object_mut()
            .unwrap()
            .remove("journalDirectory");
        let spec = parse(&evaluation).unwrap();
        let mut changed = raw.clone();
        changed[0] ^= 1;
        fs::write(output.join("policy.json"), &changed).unwrap();
        let error = evaluate_semantic_playing_policy(spec, Path::new("relative-evaluation"))
            .err()
            .unwrap();
        assert!(
            error.contains("absolute output")
                && error.contains("evaluation POST")
                && error.contains("Semantic runner input changed")
        );
        fs::write(output.join("policy.json"), &raw).unwrap();
        let mut unsafe_global = evaluation.clone();
        unsafe_global["globalExternal"]["pin"]["path"] = root
            .join("unused-parent")
            .join("..")
            .join("global.json")
            .to_string_lossy()
            .into_owned()
            .into();
        let error = evaluate_semantic_playing_policy(
            parse(&unsafe_global).unwrap(),
            Path::new("relative-evaluation"),
        )
        .err()
        .unwrap();
        assert!(
            error.contains("absolute output")
                && error.contains("evaluation POST")
                && error.contains("local without parent traversal")
        );
        evaluation["globalExternal"]["count"] = 2.into();
        let refused = root.join("policy-evaluation-refused");
        assert!(evaluate_semantic_playing_policy(parse(&evaluation).unwrap(), &refused).is_err());
        assert!(!refused.exists());
        let encoder =
            tzolkin_inference::features::FeatureEncoder::new_semantic_playing(&o).unwrap();
        let rows = (0..o.legal_actions.len())
            .map(|i| encoder.encode_legal_tagged(i).unwrap())
            .collect::<Vec<_>>();
        let expected =
            tzolkin_inference::policy::BorrowedSemanticPlayingPolicy::from_validated_parameters(
                parameters,
            )
            .unwrap()
            .predict(crate::kernel::Kernel::Scalar.resolve().unwrap(), &rows)
            .unwrap();
        let actual = loaded.distribution_playing(&o).unwrap();
        assert_eq!(
            actual
                .logits
                .iter()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>(),
            expected
                .logits
                .iter()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>()
        );
        assert_eq!(
            actual
                .probabilities
                .iter()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>(),
            expected
                .probabilities
                .iter()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>()
        );
        for (key, value) in [
            ("featureSchema", json!(2)),
            ("catalogHash", json!("0000000000000000")),
        ] {
            let mut bad = encoded.clone();
            bad["contract"][key] = value;
            assert!(policy::numeric_checksum(&serde_json::to_vec(&bad).unwrap()).is_err());
        }
        for bits in ["7f800000", "7fc00000"] {
            let mut bad = encoded.clone();
            bad["parameterBits"][0] = bits.into();
            assert!(policy::numeric_checksum(&serde_json::to_vec(&bad).unwrap()).is_err());
        }
        let mut bad = encoded.clone();
        let first = u32::from_str_radix(bad["parameterBits"][0].as_str().unwrap(), 16).unwrap();
        bad["parameterBits"][0] = format!("{:08x}", first ^ 1).into();
        assert!(policy::numeric_checksum(&serde_json::to_vec(&bad).unwrap()).is_err());
        for key in ["updates", "playingCredit"] {
            let mut bad = encoded.clone();
            bad["identity"][key] = 0.into();
            assert!(policy::numeric_checksum(&serde_json::to_vec(&bad).unwrap()).is_err());
        }
        let mut bad = encoded.clone();
        bad["identity"]["path"] = "forbidden".into();
        assert!(
            policy::LoadedSemanticPlayingDeployment::load(&serde_json::to_vec(&bad).unwrap())
                .is_err()
        );
        let duplicate = format!(
            "{{\"schema\":{},{}",
            serde_json::to_string(&encoded["schema"]).unwrap(),
            &std::str::from_utf8(&raw).unwrap()[1..]
        );
        assert!(policy::LoadedSemanticPlayingDeployment::load(duplicate.as_bytes()).is_err());
        assert!(
            policy::LoadedSemanticPlayingDeployment::load(&vec![
                b' ';
                policy::MAX_DEPLOYMENT_BYTES + 1
            ])
            .is_err()
        );
        assert!(
            export_semantic_playing_policy(parse_policy(&policy_control).unwrap(), &output)
                .is_err()
        );
        assert!(
            export_semantic_playing_policy(
                parse_policy(&policy_control).unwrap(),
                &journal.directory.join("forbidden-policy")
            )
            .is_err()
        );
    }
    let mut wrong_policy = policy_control.clone();
    wrong_policy["actor"] = json!([]);
    assert!(parse_policy(&wrong_policy).is_err());
    assert!(parse_policy(&audit).is_err());
    assert!(parse_audit(&policy_control).is_err());
    let mut families = journal.session.consumed_families().clone();
    families.insert(seed_family_id(42));
    let union = save(
        "checkpoint-fixture-prior.json",
        &json!({
        "schema":"tzolkin-conditional-native-exclusions-v1","families":families}),
    );
    let external = json!({"pin":union,"count":families.len(),
        "canonicalDigest":crate::model::digest(families.iter().map(|f|format!("{f}\n")).collect::<String>().as_bytes())});
    let chunk_prior = json!({"external":external,"history":producer.global_prior.history,
        "forbiddenSamplingSeeds":[17]});
    let mut chunk_value = json!({"parent":{"payload":payload,"closeDeclaration":close},
        "samplingSeed":29,"replicate":1,"firstOrdinal":journal.session.next_episode_ordinal(),
        "budget":[5,1u64<<30,1u64<<30,512u64<<20,2u64<<30],"prior":chunk_prior});
    let chunk: Chunk = serde_json::from_value(chunk_value.clone()).unwrap();
    let external = chunk.prior.external.load().unwrap();
    let history = chunk.prior.history.load().unwrap();
    let resumed = checkpoint::resume(
        &producer.original,
        &chunk,
        LoadedInitializationRef::Human(prepared),
        &external,
        &history,
    )
    .unwrap();
    assert_eq!(
        resumed.identity().unwrap().checksum(),
        journal.session.identity().unwrap().checksum()
    );
    assert_eq!(resumed.attempts(), 0);
    assert_eq!(resumed.identity().unwrap().attempted_whole_four(), 3);
    assert_eq!(
        resumed.actor_parameters().unwrap(),
        journal.session.actor_parameters().unwrap()
    );
    assert_eq!(
        resumed.critic_parameters().unwrap(),
        journal.session.critic_parameters().unwrap()
    );
    assert!(checkpoint::recheck(Some(&chunk)).is_ok());
    let mut guarded_original: Original =
        serde_json::from_value(serde_json::to_value(&producer.original).unwrap()).unwrap();
    guarded_original.collection_composition =
        Some(SemanticCollectionComposition::GuardedProposalV2);
    let refused_mode = checkpoint::resume(
        &guarded_original,
        &chunk,
        LoadedInitializationRef::Human(prepared),
        &external,
        &history,
    )
    .err()
    .unwrap();
    assert!(refused_mode.contains("Original differs"));
    // The immutable Original's seed17 is wholly forbidden here. Only the
    // explicit new chunk seed29 can admit prospective work.
    let mut continuation = serde_json::to_value(&producer).unwrap();
    continuation["mode"] = "continue".into();
    continuation["chunk"] = serde_json::to_value(&chunk).unwrap();
    continuation["globalPrior"] = serde_json::to_value(&chunk.prior).unwrap();
    continuation["checkpointOnCompletion"] = true.into();
    let mut future = (10_000..11_000)
        .filter(|seed| split_for_family(&seed_family_id(*seed)).unwrap() == DatasetSplit::Train);
    continuation["batches"] = (0..5)
        .map(|index| {
            json!({
        "seeds":std::array::from_fn::<_,4,_>(|_|future.next().unwrap()),
        "captureBefore":if index==0 {Some("fresh")} else {None},"slots":[]})
        })
        .collect::<Vec<_>>()
        .into();
    let continued = parse(&continuation).unwrap();
    check_new(
        &continued.wire,
        chunk.first_ordinal,
        &external,
        &history,
        limits,
    )
    .unwrap();
    continuation["chunk"]["samplingSeed"] = 17.into();
    let forbidden = parse(&continuation).unwrap();
    assert!(
        check_new(
            &forbidden.wire,
            chunk.first_ordinal,
            &external,
            &history,
            limits
        )
        .is_err()
    );
    continuation["batches"][0]["captureBefore"] = Value::Null;
    assert!(parse(&continuation).is_err());
    chunk_value["samplingSeed"] = 17.into();
    let bad: Chunk = serde_json::from_value(chunk_value.clone()).unwrap();
    assert!(
        checkpoint::resume(
            &producer.original,
            &bad,
            LoadedInitializationRef::Human(prepared),
            &external,
            &history
        )
        .is_err()
    );
    chunk_value["samplingSeed"] = 29.into();
    chunk_value["firstOrdinal"] = (journal.session.next_episode_ordinal() + 4).into();
    let bad: Chunk = serde_json::from_value(chunk_value.clone()).unwrap();
    assert!(
        checkpoint::resume(
            &producer.original,
            &bad,
            LoadedInitializationRef::Human(prepared),
            &external,
            &history
        )
        .is_err()
    );
    let original_bytes = fs::read(&payload.path).unwrap();
    fs::write(&payload.path, b"{}").unwrap();
    assert!(checkpoint::recheck(Some(&chunk)).is_err());
    assert!(
        export_semantic_playing_policy(
            parse_policy(&policy_control).unwrap(),
            &root.join("checkpoint-fixture-refused-policy")
        )
        .is_err()
    );
    fs::write(&payload.path, original_bytes).unwrap();
    let mut bad_close = unique_json(&pinned(&close, META_BYTES).unwrap(), META_BYTES).unwrap();
    bad_close["schema"] = "invalid".into();
    let bad_close = save("checkpoint-fixture-bad-close.json", &bad_close);
    chunk_value["parent"]["closeDeclaration"] = serde_json::to_value(bad_close).unwrap();
    let bad: Chunk = serde_json::from_value(chunk_value).unwrap();
    assert!(checkpoint::recheck(Some(&bad)).is_err());
    let mut stopped = unique_json(&pinned(&invocation, META_BYTES).unwrap(), META_BYTES).unwrap();
    stopped["disposition"] = "stoppedNoChange".into();
    let stopped = save("checkpoint-fixture-stopped-invocation.json", &stopped);
    let stopped_producer = producer_close(
        "checkpoint-fixture-stopped-producer.json",
        &payload,
        &stopped,
    );
    let stopped_close = save(
        "checkpoint-fixture-stopped-close.json",
        &json!({
        "schema":"tzolkin-semantic-playing-checkpoint-close-v1","checkpoint":payload,
        "journalCloseDeclaration":declaration,"invocation":stopped,"producerClose":stopped_producer}),
    );
    let mut stopped_chunk = serde_json::to_value(&chunk).unwrap();
    stopped_chunk["parent"]["closeDeclaration"] = serde_json::to_value(stopped_close).unwrap();
    let stopped_chunk: Chunk = serde_json::from_value(stopped_chunk).unwrap();
    assert!(checkpoint::recheck(Some(&stopped_chunk)).is_err());
    assert!(checkpoint::recheck(Some(&chunk)).is_ok());

    // The exhausted source can emit its FIRST checkpoint through the existing
    // audit path without extra games, budget changes or journal writes.
    let export_wire = parse_audit(&json!({"schema":AUDIT_SCHEMA,"journalDirectory":journal.directory,
        "original":producer.original,"proceduralPool":producer.procedural_pool,
        "restore":{"tip":journal.head(),"closeDeclaration":declaration},
        "checkpointExport":{"prior":chunk.prior,"permanentAllocation":producer.permanent_allocation}})).unwrap();
    let export_output = root.join("checkpoint-fixture-audit-export");
    fs::create_dir(&export_output).unwrap();
    let before = (
        journal.head(),
        journal.session.capture_ledger(),
        journal.members.clone(),
    );
    let exported = checkpoint::export(&export_wire.wire, journal, &export_output)
        .unwrap()
        .unwrap();
    assert_eq!(
        serde_json::to_value(journal.head()).unwrap(),
        serde_json::to_value(&before.0).unwrap()
    );
    assert_eq!(journal.session.capture_ledger(), before.1);
    assert_eq!(journal.members, before.2);
    let export_return = pin(export_output.join("checkpoint-export.json"));
    let export_process = producer_close(
        "checkpoint-fixture-export-process.json",
        &exported,
        &export_return,
    );
    let export_close = save(
        "checkpoint-fixture-export-close.json",
        &json!({
        "schema":"tzolkin-semantic-playing-checkpoint-close-v1","checkpoint":exported,
        "journalCloseDeclaration":declaration,"invocation":export_return,"producerClose":export_process}),
    );
    let mut exported_chunk = serde_json::to_value(&chunk).unwrap();
    exported_chunk["parent"] = json!({"payload":exported,"closeDeclaration":export_close});
    let exported_chunk: Chunk = serde_json::from_value(exported_chunk.clone()).unwrap();
    let resumed = checkpoint::resume(
        &producer.original,
        &exported_chunk,
        LoadedInitializationRef::Human(prepared),
        &external,
        &history,
    )
    .unwrap();
    assert_eq!(
        resumed.identity().unwrap().checksum(),
        journal.session.identity().unwrap().checksum()
    );
    let mut failed =
        unique_json(&pinned(&export_process, META_BYTES).unwrap(), META_BYTES).unwrap();
    failed["exitCode"] = 1.into();
    let failed = save("checkpoint-fixture-export-failed-process.json", &failed);
    let failed_close = save(
        "checkpoint-fixture-export-failed-close.json",
        &json!({
        "schema":"tzolkin-semantic-playing-checkpoint-close-v1","checkpoint":exported,
        "journalCloseDeclaration":declaration,"invocation":export_return,"producerClose":failed}),
    );
    let mut failed_chunk = serde_json::to_value(&exported_chunk).unwrap();
    failed_chunk["parent"]["closeDeclaration"] = serde_json::to_value(failed_close).unwrap();
    let failed_chunk: Chunk = serde_json::from_value(failed_chunk).unwrap();
    assert!(checkpoint::recheck(Some(&failed_chunk)).is_err()); // Old successful journal close is insufficient.
}
