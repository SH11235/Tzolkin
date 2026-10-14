//! Public V2 policy datasets and a separate semantic Setup teacher route.
//! Complete source replays, not stored floats, are authoritative.
use crate::dataset::{DatasetSplit, seed_family_id, split_for_family};
use crate::features::{
    EncodedCandidate, FEATURE_COUNT, FeatureEncoder, PUBLIC_FEATURE_SCHEMA,
    SEMANTIC_SETUP_FEATURE_SCHEMA,
};
use crate::model::digest;
use crate::replay::{self, GameReplay, ReplayHeader, ReplaySource, ReplayStep};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::fs::{self, File};
use std::io::{BufRead, BufReader, BufWriter, Read, Write};
use std::path::{Component, Path, PathBuf};
use tzolkin_core::observation::{MOVE_SCHEMA, OBSERVATION_SCHEMA, observe};
use tzolkin_core::{
    FinalScore, GameOptions, GameState, Phase, apply_move, create_game_with_options,
};

pub const DATASET_SCHEMA: &str = "tzolkin-public-policy-dataset-v1";
pub const RESOURCE_CYCLE_V5_TEACHER_SOURCE_KIND: &str = "verifiedNativeResourceCycleV5Teacher";
pub const RESOURCE_CYCLE_V5_TEACHER_TASK: &str = "resourceCycleV5PolicyOnlyTeacher";
pub const RESOURCE_CYCLE_V5_SEMANTIC_SETUP_TEACHER_SOURCE_KIND: &str =
    "verifiedNativeResourceCycleV5SemanticSetupTeacher";
pub const RESOURCE_CYCLE_V5_SEMANTIC_SETUP_TEACHER_TASK: &str =
    "resourceCycleV5SemanticSetupPolicyOnlyTeacher";
pub const MAX_SOURCE_BYTES: u64 = 64 * 1024 * 1024;
pub const MAX_SAMPLE_BYTES: u64 = 16 * 1024 * 1024;
pub const MAX_SHARD_BYTES: u64 = 32 * 1024 * 1024;
pub const MAX_MANIFEST_BYTES: u64 = 16 * 1024 * 1024;
pub const MAX_FILES: usize = 100_000;
pub const MAX_PARTITION_SAMPLES: usize = 100_000;
pub const MAX_PARTITION_CANDIDATES: usize = 1_000_000;

#[derive(Clone, Copy)]
enum SourceRoute {
    Native,
    ResourceCycleV5Teacher,
    ResourceCycleV5SemanticSetupTeacher,
}
impl SourceRoute {
    fn feature_schema(self) -> u32 {
        match self {
            Self::Native | Self::ResourceCycleV5Teacher => PUBLIC_FEATURE_SCHEMA,
            Self::ResourceCycleV5SemanticSetupTeacher => SEMANTIC_SETUP_FEATURE_SCHEMA,
        }
    }
    fn labels(self) -> (&'static str, &'static str) {
        match self {
            Self::Native => ("verifiedNativePolicy", "policyOnlyBc"),
            Self::ResourceCycleV5Teacher => (
                RESOURCE_CYCLE_V5_TEACHER_SOURCE_KIND,
                RESOURCE_CYCLE_V5_TEACHER_TASK,
            ),
            Self::ResourceCycleV5SemanticSetupTeacher => (
                RESOURCE_CYCLE_V5_SEMANTIC_SETUP_TEACHER_SOURCE_KIND,
                RESOURCE_CYCLE_V5_SEMANTIC_SETUP_TEACHER_TASK,
            ),
        }
    }
    fn read(self, path: &Path) -> Result<(Vec<u8>, GameReplay), String> {
        match self {
            Self::Native => source(path),
            Self::ResourceCycleV5Teacher | Self::ResourceCycleV5SemanticSetupTeacher => {
                resource_cycle_v5_teacher_source(path)
            }
        }
    }
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct PolicyDatasetGame {
    pub game_id: String,
    pub family_id: String,
    pub split: DatasetSplit,
    pub source_file: String,
    pub source_sha256: String,
    pub source_bytes: u64,
    pub samples: usize,
    pub candidates: usize,
    pub players: usize,
    #[serde(deserialize_with = "strict_options")]
    pub options: GameOptions,
    /// Content-bound full generating source, including coefficients and per-seat model hashes.
    pub policy_id: String,
}
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct PolicyDatasetShard {
    pub file: String,
    pub sha256: String,
    pub bytes: u64,
    pub game_index: usize,
    pub first_source_index: usize,
    pub samples: usize,
}
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct PolicyDatasetStratum {
    pub players: usize,
    pub policy_id: String,
    pub games: usize,
    pub samples: usize,
    pub candidates: usize,
}
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct PolicyDatasetManifest {
    pub schema: String,
    pub feature_schema: u32,
    pub feature_count: usize,
    pub rules_version: u32,
    pub rules_baseline: String,
    pub catalog_hash: String,
    pub move_schema: u32,
    pub observation_schema: u32,
    pub source_kind: String,
    pub task: String,
    pub value_supervised: bool,
    /// Canonical typed manifest digest with this field empty, including every source/shard SHA.
    pub fingerprint: String,
    pub samples: usize,
    pub candidates: usize,
    pub games: Vec<PolicyDatasetGame>,
    pub shards: Vec<PolicyDatasetShard>,
    pub strata: Vec<PolicyDatasetStratum>,
}

pub(crate) fn strict_options<'de, D: serde::Deserializer<'de>>(
    d: D,
) -> Result<GameOptions, D::Error> {
    #[derive(Deserialize)]
    #[serde(rename_all = "camelCase", deny_unknown_fields)]
    struct Options {
        additional_buildings: bool,
        tribes: bool,
        prophecies: bool,
        quick_actions: bool,
    }
    let o = Options::deserialize(d)?;
    Ok(GameOptions {
        additional_buildings: o.additional_buildings,
        tribes: o.tribes,
        prophecies: o.prophecies,
        quick_actions: o.quick_actions,
    })
}
// Core GameOptions intentionally permits defaults. This source boundary does not.
#[derive(Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct NativeHeader {
    replay_schema: u32,
    rules_version: u32,
    rules_baseline: String,
    catalog_hash: String,
    move_schema: u32,
    observation_schema: u32,
    source: ReplaySource,
    names: Vec<String>,
    seed: u32,
    #[serde(deserialize_with = "strict_options")]
    options: GameOptions,
}
#[derive(Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct NativeReplay {
    header: NativeHeader,
    steps: Vec<ReplayStep>,
    final_scores: Vec<FinalScore>,
    final_state: String,
    verified_complete: bool,
}
impl NativeReplay {
    fn into_replay(self) -> GameReplay {
        let h = self.header;
        GameReplay {
            header: ReplayHeader {
                replay_schema: h.replay_schema,
                rules_version: h.rules_version,
                rules_baseline: h.rules_baseline,
                catalog_hash: h.catalog_hash,
                move_schema: h.move_schema,
                observation_schema: h.observation_schema,
                source: h.source,
                names: h.names,
                seed: h.seed,
                options: h.options,
            },
            steps: self.steps,
            final_scores: self.final_scores,
            final_state: self.final_state,
            verified_complete: self.verified_complete,
        }
    }
}

// Parsing only: the caller must use its own closed source and full replay verifier.
pub(crate) fn decode_strict_native_replay(bytes: &[u8]) -> Result<GameReplay, String> {
    let parsed: NativeReplay = serde_json::from_value(
        crate::public_stochastic_record::unique_json(bytes, MAX_SOURCE_BYTES as usize)?,
    )
    .map_err(|e| format!("Invalid strict native replay: {e}"))?;
    Ok(parsed.into_replay())
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct CandidateWire {
    feature_schema: u32,
    values: Vec<f32>,
}
impl PartialEq for CandidateWire {
    fn eq(&self, other: &Self) -> bool {
        self.feature_schema == other.feature_schema
            && self.values.len() == other.values.len()
            && self
                .values
                .iter()
                .zip(&other.values)
                .all(|(a, b)| a.to_bits() == b.to_bits())
    }
}
#[derive(Debug, Deserialize, Serialize, PartialEq)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct SampleWire {
    sample_id: String,
    game_id: String,
    family_id: String,
    source_index: usize,
    actor: usize,
    phase: Phase,
    candidates: Vec<CandidateWire>,
    chosen: usize,
    // Unit has no Option/default: both a missing target and a numeric target are rejected.
    value_target: (),
}
/// Constructed only by reconstructing a complete native source and matching its stored rows.
/// Source IDs and targets are metadata; only schema-tagged candidates enter inference.
///
/// ```compile_fail
/// use tzolkin_ai::policy_dataset::ValidatedPolicySample;
/// let _: ValidatedPolicySample = serde_json::from_str("{}").unwrap();
/// ```
#[derive(Clone, Debug)]
pub struct ValidatedPolicySample {
    sample_id: String,
    game_id: String,
    family_id: String,
    source_index: usize,
    actor: usize,
    phase: Phase,
    candidates: Vec<EncodedCandidate>,
    chosen: usize,
}
impl ValidatedPolicySample {
    pub(crate) fn owned_allocation_bytes(&self) -> Result<usize, String> {
        self.candidates
            .capacity()
            .checked_mul(std::mem::size_of::<EncodedCandidate>())
            .and_then(|n| n.checked_add(self.sample_id.capacity()))
            .and_then(|n| n.checked_add(self.game_id.capacity()))
            .and_then(|n| n.checked_add(self.family_id.capacity()))
            .ok_or_else(|| "Teacher encoded-cache allocation count overflow".into())
    }
    pub fn sample_id(&self) -> &str {
        &self.sample_id
    }
    pub fn game_id(&self) -> &str {
        &self.game_id
    }
    pub fn family_id(&self) -> &str {
        &self.family_id
    }
    pub fn source_index(&self) -> usize {
        self.source_index
    }
    pub fn actor(&self) -> usize {
        self.actor
    }
    pub fn phase(&self) -> &Phase {
        &self.phase
    }
    pub fn candidates(&self) -> &[EncodedCandidate] {
        &self.candidates
    }
    pub fn chosen(&self) -> usize {
        self.chosen
    }
    pub fn value_target(&self) -> Option<f32> {
        None
    }
    fn wire(&self, feature_schema: u32) -> Result<SampleWire, String> {
        Ok(SampleWire {
            sample_id: self.sample_id.clone(),
            game_id: self.game_id.clone(),
            family_id: self.family_id.clone(),
            source_index: self.source_index,
            actor: self.actor,
            phase: self.phase,
            chosen: self.chosen,
            value_target: (),
            candidates: self
                .candidates
                .iter()
                .map(|row| {
                    Ok(CandidateWire {
                        feature_schema: row.feature_schema(),
                        values: row.values_for_schema(feature_schema)?.to_vec(),
                    })
                })
                .collect::<Result<_, String>>()?,
        })
    }
}
fn hex(hash: impl AsRef<[u8]>) -> String {
    hash.as_ref().iter().map(|b| format!("{b:02x}")).collect()
}
pub(crate) fn digest_id(id: &str) -> bool {
    id.len() == 64
        && id
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}
pub(crate) fn content_id<T: Serialize>(domain: &[u8], value: &T) -> Result<String, String> {
    struct Sink(Sha256);
    impl Write for Sink {
        fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
            self.0.update(bytes);
            Ok(bytes.len())
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }
    let mut sink = Sink(Sha256::new());
    sink.0.update(domain);
    serde_json::to_writer(&mut sink, value).map_err(|e| e.to_string())?;
    Ok(hex(sink.0.finalize()))
}
pub(crate) fn resource_cycle_v5_teacher_policy_id(players: usize) -> Result<String, String> {
    if !(3..=4).contains(&players) {
        return Err("V5 teacher identity requires base 3/4p".into());
    }
    content_id(
        b"tzolkin-native-policy-source-v1\0",
        &ReplaySource::PolicySelfPlay {
            policies: vec![
                replay::SeatPolicy::ResourceCycle {
                    configuration: crate::arena::resource_cycle::ResourceCycleConfig::v5(),
                };
                players
            ],
        },
    )
}
fn fingerprint(m: &PolicyDatasetManifest) -> Result<String, String> {
    let mut canonical = m.clone();
    canonical.fingerprint.clear();
    Ok(digest(
        &serde_json::to_vec(&canonical).map_err(|e| e.to_string())?,
    ))
}
pub(crate) fn local_path(path: &Path) -> Result<(), String> {
    let text = path.to_string_lossy();
    if text.is_empty()
        || text.contains('\0')
        || text.contains("://")
        || text.starts_with("\\\\")
        || text.starts_with("//")
    {
        return Err("Expected an explicit local path".into());
    }
    // A drive-relative path has an implicit per-drive cwd; require an explicit
    // absolute drive path instead. Ordinary paths are anchored to our local cwd.
    if !path.is_absolute() && matches!(path.components().next(), Some(Component::Prefix(_))) {
        return Err("Drive-relative paths are not supported".into());
    }
    let absolute = if path.is_absolute() {
        path.to_path_buf()
    } else {
        std::env::current_dir()
            .map_err(|e| e.to_string())?
            .join(path)
    };
    let absolute_text = absolute.to_string_lossy();
    if absolute_text.starts_with("\\\\") || absolute_text.starts_with("//") {
        return Err("Network path hierarchy rejected".into());
    }
    // Inspect root first: inspecting a leaf beneath a junction could already
    // traverse its network target before the ancestor was rejected.
    let mut ancestor = PathBuf::new();
    for component in absolute.components() {
        ancestor.push(component.as_os_str());
        if matches!(component, Component::Prefix(_)) {
            continue;
        }
        match fs::symlink_metadata(&ancestor) {
            Ok(meta) if meta.file_type().is_symlink() => {
                return Err("Symlink source/output hierarchy rejected".into());
            }
            Ok(_) => {}
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
            Err(e) => return Err(e.to_string()),
        }
    }
    Ok(())
}
fn regular_file(path: &Path, max: u64) -> Result<File, String> {
    local_path(path)?;
    let meta = fs::symlink_metadata(path).map_err(|e| e.to_string())?;
    if !meta.is_file() || meta.len() > max {
        return Err("Source/dataset file type or byte bound rejected".into());
    }
    let file = File::open(path).map_err(|e| e.to_string())?;
    if file.metadata().map_err(|e| e.to_string())?.len() > max {
        return Err("File changed beyond byte bound".into());
    }
    Ok(file)
}
pub(crate) fn read_bytes(path: &Path, max: u64) -> Result<Vec<u8>, String> {
    let mut bytes = Vec::new();
    regular_file(path, max)?
        .take(max + 1)
        .read_to_end(&mut bytes)
        .map_err(|e| e.to_string())?;
    if bytes.len() as u64 > max {
        return Err("File exceeds bounded read".into());
    }
    Ok(bytes)
}
pub(crate) fn file_digest(path: &Path, max: u64) -> Result<(u64, String), String> {
    let mut reader = regular_file(path, max)?.take(max + 1);
    let mut hash = Sha256::new();
    let mut bytes = 0u64;
    // Source checks stream whole files repeatedly; amortize native read calls.
    let mut buffer = [0u8; 128 * 1024];
    loop {
        let read = reader.read(&mut buffer).map_err(|e| e.to_string())?;
        if read == 0 {
            break;
        }
        bytes += read as u64;
        if bytes > max {
            return Err("File changed beyond digest byte budget".into());
        }
        hash.update(&buffer[..read]);
    }
    Ok((bytes, hex(hash.finalize())))
}
fn source(path: &Path) -> Result<(Vec<u8>, GameReplay), String> {
    let bytes = read_bytes(path, MAX_SOURCE_BYTES)?;
    let parsed: NativeReplay =
        serde_json::from_slice(&bytes).map_err(|e| format!("Invalid strict native replay: {e}"))?;
    let record = parsed.into_replay();
    if !(3..=4).contains(&record.header.names.len())
        || record.header.options != GameOptions::default()
    {
        return Err("Public policy dataset requires base 3-4p with all four options false".into());
    }
    match &record.header.source {
        ReplaySource::SelfPlay {
            policy_version,
            weights,
        } => {
            // Recorded labels before the overdue-food reserve correction retain
            // their original identity. Replay verifies saved actions, without
            // executing either version of the heuristic.
            const HISTORICAL_SELF_PLAY_POLICY_VERSION: &str = "heuristic-v1";
            if policy_version != crate::POLICY_VERSION
                && policy_version != HISTORICAL_SELF_PLAY_POLICY_VERSION
            {
                return Err("Unsupported native heuristic source version".into());
            }
            weights.validate()?;
        }
        ReplaySource::PolicySelfPlay { policies } => {
            if policies.len() != record.header.names.len() {
                return Err("Native source policy/seat mismatch".into());
            }
            for policy in policies {
                if policy.is_rl_evaluation() {
                    return Err("RL evaluation source training admission is not implemented".into());
                }
                if matches!(policy, replay::SeatPolicy::PublicLearnedTradeGuard { .. }) {
                    return Err("Guarded public learned source admission is not implemented".into());
                }
                if matches!(policy, replay::SeatPolicy::ResourceCycle { .. }) {
                    return Err("ResourceCycle source training admission is not implemented".into());
                }
                policy.validate()?;
            }
        }
        ReplaySource::Human { .. } => {
            return Err("Human/public source admission is not implemented".into());
        }
        ReplaySource::CornFirstV5Teacher { .. } => {
            return Err(
                "Corn-first V5 Playing teacher has no legacy native dataset admission".into(),
            );
        }
        ReplaySource::CornFirstV7Teacher { .. } => {
            return Err(
                "Corn-first V7 Playing teacher requires its dedicated semantic4 dataset".into(),
            );
        }
    }
    replay::verify_replay(&record)?;
    Ok((bytes, record))
}

fn verify_resource_cycle_v5_teacher(record: &GameReplay) -> Result<(), String> {
    use crate::arena::resource_cycle::{self, ResourceCycleConfig};
    let configuration = ResourceCycleConfig::v5();
    configuration.validate()?;
    let ReplaySource::PolicySelfPlay { policies } = &record.header.source else {
        return Err("ResourceCycleV5 teacher requires explicit all-seat policy provenance".into());
    };
    if !(3..=4).contains(&record.header.names.len())
        || record.header.options != GameOptions::default()
        || policies.len() != record.header.names.len()
        || policies.iter().any(|policy| {
            !matches!(policy, replay::SeatPolicy::ResourceCycle { configuration: actual }
                if actual == &configuration)
        })
    {
        return Err("ResourceCycleV5 teacher requires fixed v5, all seats and base 3-4p".into());
    }
    // The complete independent replay verifies every observation, transition and
    // terminal score. Teacher quality is separate from this consistency check.
    replay::verify_replay(record)?;
    for step in &record.steps {
        let decision = resource_cycle::choose_move(&step.observation, &configuration)?;
        if decision.policy_version != resource_cycle::V5_POLICY_VERSION
            || !decision.score.is_finite()
            || decision.actor != step.actor
            || decision.r#move != step.chosen.r#move
        {
            return Err(format!(
                "ResourceCycleV5 regenerated teacher choice differs at step {}",
                step.index
            ));
        }
    }
    Ok(())
}
fn resource_cycle_v5_teacher_source(path: &Path) -> Result<(Vec<u8>, GameReplay), String> {
    let bytes = read_bytes(path, MAX_SOURCE_BYTES)?;
    let parsed: NativeReplay = serde_json::from_value(
        crate::public_stochastic_record::unique_json(&bytes, MAX_SOURCE_BYTES as usize)?,
    )
    .map_err(|e| format!("Invalid strict ResourceCycleV5 teacher replay: {e}"))?;
    let record = parsed.into_replay();
    verify_resource_cycle_v5_teacher(&record)?;
    Ok((bytes, record))
}
// State-MC consumes the same closed native source boundary without changing it.
pub(crate) fn read_state_native_source(
    path: &Path,
    index: usize,
) -> Result<(Vec<u8>, GameReplay, PolicyDatasetGame), String> {
    let (bytes, record) = source(path)?;
    let metadata = game_metadata(index, &bytes, &record)?;
    Ok((bytes, record, metadata))
}
fn expected_sample(
    state: &GameState,
    step: &ReplayStep,
    game: &PolicyDatasetGame,
    route: SourceRoute,
) -> Result<ValidatedPolicySample, String> {
    let observation = observe(state, state.current_player)?;
    if observation != step.observation
        || step.actor != observation.actor
        || !matches!(observation.phase, Phase::Setup | Phase::Playing)
    {
        return Err(format!(
            "Source observation mismatch at index {}",
            step.index
        ));
    }
    let matches: Vec<usize> = observation
        .legal_actions
        .iter()
        .enumerate()
        .filter_map(|(i, a)| (a == &step.chosen).then_some(i))
        .collect();
    if matches.len() != 1 {
        return Err("Verified chosen action is not unique in complete ordered legal set".into());
    }
    let encoder = match route {
        SourceRoute::Native | SourceRoute::ResourceCycleV5Teacher => {
            FeatureEncoder::new_public(&observation)?
        }
        SourceRoute::ResourceCycleV5SemanticSetupTeacher => {
            FeatureEncoder::new_semantic_setup(&observation)?
        }
    };
    let candidates: Vec<EncodedCandidate> = (0..observation.legal_actions.len())
        .map(|i| encoder.encode_legal_tagged(i))
        .collect::<Result<_, _>>()?;
    for row in &candidates {
        if row
            .values_for_schema(route.feature_schema())?
            .iter()
            .any(|v| !v.is_finite() || v.abs() > 1024.0)
        {
            return Err("Nonfinite/out-of-bound public source features".into());
        }
    }
    Ok(ValidatedPolicySample {
        sample_id: content_id(
            b"tzolkin-public-native-sample-v1\0",
            &(&game.game_id, step.index, observation.actor),
        )?,
        game_id: game.game_id.clone(),
        family_id: game.family_id.clone(),
        source_index: step.index,
        actor: observation.actor,
        phase: observation.phase,
        candidates,
        chosen: matches[0],
    })
}
fn initial(record: &GameReplay) -> Result<GameState, String> {
    create_game_with_options(
        record.header.names.clone(),
        record.header.seed,
        record.header.options.clone(),
    )
}
fn game_metadata(
    index: usize,
    bytes: &[u8],
    record: &GameReplay,
) -> Result<PolicyDatasetGame, String> {
    let family_id = seed_family_id(record.header.seed);
    Ok(PolicyDatasetGame {
        game_id: content_id(b"tzolkin-public-native-game-v1\0", record)?,
        split: split_for_family(&family_id)?,
        family_id,
        source_file: format!("sources/source-{index:06}.json"),
        source_sha256: digest(bytes),
        source_bytes: bytes.len() as u64,
        samples: record.steps.len(),
        candidates: record
            .steps
            .iter()
            .map(|s| s.observation.legal_actions.len())
            .sum(),
        players: record.header.names.len(),
        options: record.header.options.clone(),
        policy_id: content_id(b"tzolkin-native-policy-source-v1\0", &record.header.source)?,
    })
}
fn strata(games: &[PolicyDatasetGame]) -> Vec<PolicyDatasetStratum> {
    let mut groups: BTreeMap<(usize, String), (usize, usize, usize)> = BTreeMap::new();
    for g in games {
        let counts = groups.entry((g.players, g.policy_id.clone())).or_default();
        counts.0 += 1;
        counts.1 += g.samples;
        counts.2 += g.candidates;
    }
    groups
        .into_iter()
        .map(
            |((players, policy_id), (games, samples, candidates))| PolicyDatasetStratum {
                players,
                policy_id,
                games,
                samples,
                candidates,
            },
        )
        .collect()
}
/// Atomically create a new final file; staging cleanup cannot undo its commit.
/// The caller must bound bytes and validate the local directory and filename.
pub fn publish_bytes(directory: &Path, name: &str, bytes: &[u8]) -> Result<(), String> {
    let temp = directory.join(format!(".{name}.tmp"));
    let mut created = false;
    let result = (|| {
        let mut file = File::options()
            .write(true)
            .create_new(true)
            .open(&temp)
            .map_err(|e| e.to_string())?;
        created = true;
        file.write_all(bytes)
            .and_then(|_| file.sync_all())
            .map_err(|e| e.to_string())?;
        drop(file);
        fs::hard_link(&temp, directory.join(name)).map_err(|e| e.to_string())
    })();
    // Final publication is the commit point. A staging cleanup failure cannot
    // turn a successfully published completion marker into a reported failure.
    if created {
        let _ = fs::remove_file(&temp);
    }
    result
}
struct ShardWriter {
    writer: BufWriter<File>,
    hash: Sha256,
    meta: PolicyDatasetShard,
    temporary: PathBuf,
    destination: PathBuf,
}
impl ShardWriter {
    fn new(
        directory: &Path,
        index: usize,
        game_index: usize,
        first_source_index: usize,
    ) -> Result<Self, String> {
        let name = format!("shard-{index:06}.jsonl");
        let temporary = directory.join(format!(".{name}.tmp"));
        let file = File::options()
            .write(true)
            .create_new(true)
            .open(&temporary)
            .map_err(|e| e.to_string())?;
        Ok(Self {
            writer: BufWriter::new(file),
            hash: Sha256::new(),
            temporary,
            destination: directory.join(&name),
            meta: PolicyDatasetShard {
                file: name,
                sha256: String::new(),
                bytes: 0,
                game_index,
                first_source_index,
                samples: 0,
            },
        })
    }
    fn append(&mut self, bytes: &[u8]) -> Result<(), String> {
        self.writer
            .write_all(bytes)
            .and_then(|_| self.writer.write_all(b"\n"))
            .map_err(|e| e.to_string())?;
        self.hash.update(bytes);
        self.hash.update(b"\n");
        self.meta.bytes += bytes.len() as u64 + 1;
        self.meta.samples += 1;
        Ok(())
    }
    fn finish(mut self) -> Result<PolicyDatasetShard, String> {
        self.writer
            .flush()
            .and_then(|_| self.writer.get_ref().sync_all())
            .map_err(|e| e.to_string())?;
        drop(self.writer);
        fs::hard_link(&self.temporary, &self.destination).map_err(|e| e.to_string())?;
        fs::remove_file(&self.temporary).map_err(|e| e.to_string())?;
        self.meta.sha256 = hex(self.hash.finalize());
        Ok(self.meta)
    }
}
/// Enumerates only immediate regular `.json` native source files, in filename order.
/// A manifest/summary/public record masquerading as JSON fails strict replay parsing.
pub fn native_source_files(directory: &Path) -> Result<Vec<PathBuf>, String> {
    local_path(directory)?;
    if !fs::metadata(directory).map_err(|e| e.to_string())?.is_dir() {
        return Err("Native input must be a directory".into());
    }
    let mut files = Vec::new();
    for entry in fs::read_dir(directory).map_err(|e| e.to_string())? {
        let path = entry.map_err(|e| e.to_string())?.path();
        if path.extension().and_then(|s| s.to_str()) != Some("json") {
            return Err("Native input directory must contain only replay JSON files".into());
        }
        regular_file(&path, MAX_SOURCE_BYTES)?;
        files.push(path);
        if files.len() > MAX_FILES {
            return Err("Too many native source files".into());
        }
    }
    files.sort();
    if files.is_empty() {
        return Err("No native source replays".into());
    }
    Ok(files)
}
/// Two bounded passes. Every original source is fully verified before creating the new
/// directory, then exact bytes are rechecked/copied and all V2 rows reconstructed again.
/// Failed exports may leave a partial directory, but never publish a completion manifest.
pub fn export_native_files(
    paths: &[PathBuf],
    output: &Path,
) -> Result<PolicyDatasetManifest, String> {
    export_sources(paths.len(), |index| source(&paths[index]), output)
}
/// Export deterministic all-seat v5 demonstrations with separate teacher labels.
/// This grants no Human provenance, value supervision or policy qualification.
/// Original native, legacy and State-MC admission remains unchanged.
pub fn export_resource_cycle_v5_teacher_files(
    paths: &[PathBuf],
    output: &Path,
) -> Result<PolicyDatasetManifest, String> {
    export_sources_with_route(
        paths.len(),
        |index| resource_cycle_v5_teacher_source(&paths[index]),
        output,
        SourceRoute::ResourceCycleV5Teacher,
    )
}
/// Re-encode complete all-seat V5 sources into a new semantic Setup dataset.
/// Stored schema-2 shards and model artifacts are never an input to this builder.
/// This grants no Human provenance, value supervision or policy qualification.
pub fn export_resource_cycle_v5_semantic_setup_teacher_files(
    paths: &[PathBuf],
    output: &Path,
) -> Result<PolicyDatasetManifest, String> {
    let route = SourceRoute::ResourceCycleV5SemanticSetupTeacher;
    export_sources_with_route(
        paths.len(),
        |index| route.read(&paths[index]),
        output,
        route,
    )
}
fn export_sources(
    count: usize,
    read_source: impl FnMut(usize) -> Result<(Vec<u8>, GameReplay), String>,
    output: &Path,
) -> Result<PolicyDatasetManifest, String> {
    export_sources_with_route(count, read_source, output, SourceRoute::Native)
}
fn export_sources_with_route(
    count: usize,
    mut read_source: impl FnMut(usize) -> Result<(Vec<u8>, GameReplay), String>,
    output: &Path,
    route: SourceRoute,
) -> Result<PolicyDatasetManifest, String> {
    local_path(output)?;
    if fs::symlink_metadata(output).is_ok() {
        return Err("Dataset destination already exists".into());
    }
    if count == 0 || count > MAX_FILES {
        return Err("Invalid native source count".into());
    }
    let mut games = Vec::with_capacity(count);
    let mut seen = BTreeSet::new();
    let mut budgets: BTreeMap<DatasetSplit, (usize, usize)> = BTreeMap::new();
    for index in 0..count {
        let (bytes, record) = read_source(index)?;
        let game = game_metadata(index, &bytes, &record)?;
        if !seen.insert(game.game_id.clone()) {
            return Err("Duplicate native game content".into());
        }
        let mut state = initial(&record)?;
        for step in &record.steps {
            let sample = expected_sample(&state, step, &game, route)?;
            if serde_json::to_vec(&sample.wire(route.feature_schema())?)
                .map_err(|e| e.to_string())?
                .len() as u64
                + 1
                > MAX_SAMPLE_BYTES
            {
                return Err("Policy sample exceeds byte budget".into());
            }
            state = apply_move(&state, step.chosen.r#move.clone())?;
        }
        let budget = budgets.entry(game.split).or_default();
        budget.0 = budget
            .0
            .checked_add(game.samples)
            .ok_or("Sample count overflow")?;
        budget.1 = budget
            .1
            .checked_add(game.candidates)
            .ok_or("Candidate count overflow")?;
        if budget.0 > MAX_PARTITION_SAMPLES || budget.1 > MAX_PARTITION_CANDIDATES {
            return Err("Dataset partition exceeds policy training budget".into());
        }
        games.push(game);
    }
    fs::create_dir(output).map_err(|e| format!("Cannot create new dataset directory: {e}"))?;
    fs::create_dir(output.join("sources")).map_err(|e| e.to_string())?;
    let mut m = PolicyDatasetManifest {
        schema: DATASET_SCHEMA.into(),
        feature_schema: route.feature_schema(),
        feature_count: FEATURE_COUNT,
        rules_version: replay::RULES_VERSION,
        rules_baseline: replay::RULES_BASELINE.into(),
        catalog_hash: replay::catalog_hash(),
        move_schema: MOVE_SCHEMA,
        observation_schema: OBSERVATION_SCHEMA,
        source_kind: route.labels().0.into(),
        task: route.labels().1.into(),
        value_supervised: false,
        fingerprint: String::new(),
        samples: 0,
        candidates: 0,
        games,
        shards: Vec::new(),
        strata: Vec::new(),
    };
    for index in 0..count {
        let (bytes, record) = read_source(index)?;
        let game = &m.games[index];
        if game_metadata(index, &bytes, &record)? != *game {
            return Err("Native source changed between export passes".into());
        }
        publish_bytes(
            &output.join("sources"),
            &format!("source-{index:06}.json"),
            &bytes,
        )?;
        let mut state = initial(&record)?;
        let mut writer: Option<ShardWriter> = None;
        for step in &record.steps {
            let bytes = serde_json::to_vec(
                &expected_sample(&state, step, game, route)?.wire(route.feature_schema())?,
            )
            .map_err(|e| e.to_string())?;
            if writer
                .as_ref()
                .is_some_and(|s| s.meta.bytes + bytes.len() as u64 + 1 > MAX_SHARD_BYTES)
            {
                m.shards.push(writer.take().unwrap().finish()?);
            }
            if writer.is_none() {
                if m.shards.len() >= MAX_FILES {
                    return Err("Too many policy dataset shards".into());
                }
                writer = Some(ShardWriter::new(output, m.shards.len(), index, step.index)?);
            }
            writer.as_mut().unwrap().append(&bytes)?;
            state = apply_move(&state, step.chosen.r#move.clone())?;
        }
        if let Some(writer) = writer {
            m.shards.push(writer.finish()?);
        }
        m.samples += game.samples;
        m.candidates += game.candidates;
    }
    m.strata = strata(&m.games);
    m.fingerprint = fingerprint(&m)?;
    validate_manifest(&m, route)?;
    // Audit the published source/shards before the completion marker can exist.
    let dataset = ValidatedPolicyDataset {
        directory: output.to_path_buf(),
        manifest: m.clone(),
        route,
    };
    audit_all(&dataset)?;
    let bytes = serde_json::to_vec_pretty(&m).map_err(|e| e.to_string())?;
    if bytes.len() as u64 > MAX_MANIFEST_BYTES {
        return Err("Policy dataset manifest exceeds byte budget".into());
    }
    publish_bytes(output, "manifest.json", &bytes)?;
    Ok(m)
}
fn validate_manifest(m: &PolicyDatasetManifest, route: SourceRoute) -> Result<(), String> {
    if m.schema != DATASET_SCHEMA
        || m.feature_schema != route.feature_schema()
        || m.feature_count != FEATURE_COUNT
        || m.rules_version != replay::RULES_VERSION
        || m.rules_baseline != replay::RULES_BASELINE
        || m.catalog_hash != replay::catalog_hash()
        || m.move_schema != MOVE_SCHEMA
        || m.observation_schema != OBSERVATION_SCHEMA
        || m.source_kind != route.labels().0
        || m.task != route.labels().1
        || m.value_supervised
        || !digest_id(&m.fingerprint)
        || m.fingerprint != fingerprint(m)?
        || m.games.is_empty()
        || m.games.len() > MAX_FILES
        || m.shards.is_empty()
        || m.shards.len() > MAX_FILES
    {
        return Err("Unsupported/corrupt public policy dataset manifest".into());
    }
    let mut seen = BTreeSet::new();
    let mut budgets: BTreeMap<DatasetSplit, (usize, usize)> = BTreeMap::new();
    let mut total = (0usize, 0usize);
    for (index, game) in m.games.iter().enumerate() {
        if !digest_id(&game.game_id)
            || !seen.insert(&game.game_id)
            || !digest_id(&game.family_id)
            || !digest_id(&game.policy_id)
            || !digest_id(&game.source_sha256)
            || game.source_file != format!("sources/source-{index:06}.json")
            || game.source_bytes == 0
            || game.source_bytes > MAX_SOURCE_BYTES
            || !(3..=4).contains(&game.players)
            || game.options != GameOptions::default()
            || game.samples == 0
            || game.samples > replay::MAX_DECISIONS
            || game.candidates < game.samples
            || game.candidates > MAX_PARTITION_CANDIDATES
            || game.split != split_for_family(&game.family_id)?
        {
            return Err("Invalid/duplicate native source metadata".into());
        }
        let budget = budgets.entry(game.split).or_default();
        budget.0 = budget
            .0
            .checked_add(game.samples)
            .ok_or("Sample count overflow")?;
        budget.1 = budget
            .1
            .checked_add(game.candidates)
            .ok_or("Candidate count overflow")?;
        if budget.0 > MAX_PARTITION_SAMPLES || budget.1 > MAX_PARTITION_CANDIDATES {
            return Err("Policy partition count budget exceeded".into());
        }
        total.0 += game.samples;
        total.1 += game.candidates;
    }
    let mut game_index = 0;
    let mut source_index = 0;
    for (index, shard) in m.shards.iter().enumerate() {
        let game = m.games.get(game_index).ok_or("Too many source shards")?;
        if shard.file != format!("shard-{index:06}.jsonl")
            || !digest_id(&shard.sha256)
            || shard.bytes == 0
            || shard.bytes > MAX_SHARD_BYTES
            || shard.game_index != game_index
            || shard.first_source_index != source_index
            || shard.samples == 0
            || shard.samples > game.samples - source_index
        {
            return Err("Invalid shard ordering/count/filename".into());
        }
        source_index += shard.samples;
        if source_index == game.samples {
            game_index += 1;
            source_index = 0;
        }
    }
    if game_index != m.games.len()
        || source_index != 0
        || total != (m.samples, m.candidates)
        || m.strata != strata(&m.games)
    {
        return Err("Policy dataset counts/strata mismatch".into());
    }
    Ok(())
}

pub struct ValidatedPolicyDataset {
    directory: PathBuf,
    manifest: PolicyDatasetManifest,
    route: SourceRoute,
}
impl ValidatedPolicyDataset {
    pub fn manifest(&self) -> &PolicyDatasetManifest {
        &self.manifest
    }
    pub(crate) fn require_resource_cycle_v5_teacher_route(&self) -> Result<(), String> {
        if !matches!(self.route, SourceRoute::ResourceCycleV5Teacher) {
            return Err(
                "ResourceCycleV5 training requires its dedicated validated source route".into(),
            );
        }
        validate_manifest(&self.manifest, self.route)
    }
    pub(crate) fn recheck_resource_cycle_v5_teacher_manifest(
        &self,
    ) -> Result<(u64, String), String> {
        self.require_resource_cycle_v5_teacher_route()?;
        self.recheck_manifest_bytes()
    }
    fn recheck_manifest_bytes(&self) -> Result<(u64, String), String> {
        validate_manifest(&self.manifest, self.route)?;
        let bytes = read_bytes(&self.directory.join("manifest.json"), MAX_MANIFEST_BYTES)?;
        let manifest: PolicyDatasetManifest =
            serde_json::from_slice(&bytes).map_err(|e| e.to_string())?;
        if manifest != self.manifest {
            return Err("Teacher source manifest changed after load".into());
        }
        Ok((bytes.len() as u64, digest(&bytes)))
    }
    pub(crate) fn recheck_resource_cycle_v5_teacher_bytes(&self) -> Result<(u64, String), String> {
        self.require_resource_cycle_v5_teacher_route()?;
        self.recheck_all_bytes()
    }
    fn recheck_all_bytes(&self) -> Result<(u64, String), String> {
        let manifest = self.recheck_manifest_bytes()?;
        for game in &self.manifest.games {
            if file_digest(&self.directory.join(&game.source_file), MAX_SOURCE_BYTES)?
                != (game.source_bytes, game.source_sha256.clone())
            {
                return Err("Teacher source bytes changed after load".into());
            }
        }
        for shard in &self.manifest.shards {
            if file_digest(&self.directory.join(&shard.file), MAX_SHARD_BYTES)?
                != (shard.bytes, shard.sha256.clone())
            {
                return Err("Teacher shard bytes changed after load".into());
            }
        }
        Ok(manifest)
    }
    pub fn iter(&self) -> PolicyDatasetIter {
        self.iterator(None)
    }
    pub fn iter_split(&self, split: DatasetSplit) -> PolicyDatasetIter {
        self.iterator(Some(split))
    }
    fn iterator(&self, split: Option<DatasetSplit>) -> PolicyDatasetIter {
        PolicyDatasetIter {
            directory: self.directory.clone(),
            manifest: self.manifest.clone(),
            route: self.route,
            split,
            next_shard: 0,
            current: None,
            game: None,
            failed: false,
        }
    }
}
/// Complete fixed-V5 sources re-encoded with semantic Setup schema 3.
/// This separate handle cannot enter existing schema-2 trainers or native admission.
/// No mutable rows, deserialization or conversion into their handle is exposed.
///
/// ```compile_fail
/// use tzolkin_ai::policy_dataset::ValidatedSemanticSetupTeacherDataset;
/// use tzolkin_ai::policy_training::PolicyBcConfig;
/// fn old_teacher_trainer(dataset: &ValidatedSemanticSetupTeacherDataset) {
///     let _ = tzolkin_ai::resource_cycle_teacher_training::train_dataset(
///         dataset, &PolicyBcConfig::default(), None);
/// }
/// ```
pub struct ValidatedSemanticSetupTeacherDataset {
    dataset: ValidatedPolicyDataset,
    manifest_pin: (u64, String),
}
impl ValidatedSemanticSetupTeacherDataset {
    pub fn manifest(&self) -> &PolicyDatasetManifest {
        self.dataset.manifest()
    }
    /// Consume through EOF to complete replay, label regeneration and raw-byte checks.
    pub fn iter(&self) -> SemanticSetupTeacherDatasetIter<'_> {
        self.iterator(None)
    }
    pub fn iter_split(&self, split: DatasetSplit) -> SemanticSetupTeacherDatasetIter<'_> {
        self.iterator(Some(split))
    }
    // Used only by the distinct schema-3 teacher trainer's before/after cache
    // checks. The typed handle and its original load pin remain mandatory.
    pub(crate) fn recheck_teacher_bytes(&self) -> Result<(u64, String), String> {
        self.recheck_bytes()?;
        Ok(self.manifest_pin.clone())
    }
    fn recheck_bytes(&self) -> Result<(), String> {
        if self.dataset.recheck_all_bytes()? != self.manifest_pin {
            return Err("Semantic Setup teacher manifest bytes changed after load".into());
        }
        Ok(())
    }
    fn iterator(&self, split: Option<DatasetSplit>) -> SemanticSetupTeacherDatasetIter<'_> {
        SemanticSetupTeacherDatasetIter {
            dataset: self,
            rows: self.dataset.iterator(split),
            started: false,
            finished: false,
        }
    }
}
/// Only this iterator's schema-3 route receives additional manifest/source/shard pins.
pub struct SemanticSetupTeacherDatasetIter<'a> {
    dataset: &'a ValidatedSemanticSetupTeacherDataset,
    rows: PolicyDatasetIter,
    started: bool,
    finished: bool,
}
impl Iterator for SemanticSetupTeacherDatasetIter<'_> {
    type Item = Result<ValidatedPolicySample, String>;
    fn next(&mut self) -> Option<Self::Item> {
        if self.finished {
            return None;
        }
        if !self.started {
            self.started = true;
            if let Err(error) = self.dataset.recheck_bytes() {
                self.finished = true;
                return Some(Err(error));
            }
        }
        match self.rows.next() {
            Some(Ok(sample)) => Some(Ok(sample)),
            Some(Err(error)) => {
                self.finished = true;
                Some(Err(error))
            }
            None => {
                self.finished = true;
                self.dataset.recheck_bytes().err().map(Err)
            }
        }
    }
}
struct ReadingGame {
    index: usize,
    record: GameReplay,
    state: GameState,
    next_source_index: usize,
    candidates: usize,
}
struct ReadingShard {
    reader: BufReader<File>,
    hash: Sha256,
    meta: PolicyDatasetShard,
    bytes: u64,
    samples: usize,
}
pub struct PolicyDatasetIter {
    directory: PathBuf,
    manifest: PolicyDatasetManifest,
    route: SourceRoute,
    split: Option<DatasetSplit>,
    next_shard: usize,
    current: Option<ReadingShard>,
    game: Option<ReadingGame>,
    failed: bool,
}
impl PolicyDatasetIter {
    fn finish_shard(&mut self) -> Result<(), String> {
        let shard = self.current.take().unwrap();
        if shard.samples != shard.meta.samples
            || shard.bytes != shard.meta.bytes
            || hex(shard.hash.finalize()) != shard.meta.sha256
        {
            return Err("Policy shard checksum/count mismatch".into());
        }
        if file_digest(&self.directory.join(&shard.meta.file), MAX_SHARD_BYTES)?
            != (shard.meta.bytes, shard.meta.sha256)
        {
            return Err("Policy shard changed during iteration".into());
        }
        let game = self.game.as_ref().unwrap();
        let meta = &self.manifest.games[game.index];
        if game.next_source_index == meta.samples {
            if game.candidates != meta.candidates || game.state.phase != Phase::Finished {
                return Err("Policy source totals/terminal mismatch".into());
            }
            if file_digest(&self.directory.join(&meta.source_file), MAX_SOURCE_BYTES)?
                != (meta.source_bytes, meta.source_sha256.clone())
            {
                return Err("Copied native source changed during iteration".into());
            }
            self.game = None;
        }
        Ok(())
    }
    fn read_next(&mut self) -> Result<Option<ValidatedPolicySample>, String> {
        loop {
            if self.current.is_none() {
                let Some(meta) = self.manifest.shards.get(self.next_shard).cloned() else {
                    return Ok(None);
                };
                self.next_shard += 1;
                let game_meta = &self.manifest.games[meta.game_index];
                if self.split.is_some_and(|split| game_meta.split != split) {
                    continue;
                }
                if self.game.is_none() {
                    let (bytes, record) = self
                        .route
                        .read(&self.directory.join(&game_meta.source_file))?;
                    if game_metadata(meta.game_index, &bytes, &record)? != *game_meta {
                        return Err("Copied native source metadata/hash mismatch".into());
                    }
                    let state = initial(&record)?;
                    self.game = Some(ReadingGame {
                        index: meta.game_index,
                        record,
                        state,
                        next_source_index: 0,
                        candidates: 0,
                    });
                }
                let game = self.game.as_ref().unwrap();
                if game.index != meta.game_index
                    || game.next_source_index != meta.first_source_index
                {
                    return Err("Policy source/shard index mismatch".into());
                }
                let file = regular_file(&self.directory.join(&meta.file), MAX_SHARD_BYTES)?;
                if file.metadata().map_err(|e| e.to_string())?.len() != meta.bytes {
                    return Err("Policy shard byte count mismatch".into());
                }
                self.current = Some(ReadingShard {
                    reader: BufReader::new(file),
                    hash: Sha256::new(),
                    meta,
                    bytes: 0,
                    samples: 0,
                });
            }
            let shard = self.current.as_mut().unwrap();
            let mut line = Vec::new();
            let count = shard
                .reader
                .by_ref()
                .take(MAX_SAMPLE_BYTES + 1)
                .read_until(b'\n', &mut line)
                .map_err(|e| e.to_string())?;
            if count == 0 {
                self.finish_shard()?;
                continue;
            }
            if count as u64 > MAX_SAMPLE_BYTES
                || line.last() != Some(&b'\n')
                || shard.samples >= shard.meta.samples
            {
                return Err("Oversized/unterminated/excess policy sample".into());
            }
            shard.hash.update(&line);
            shard.bytes += count as u64;
            shard.samples += 1;
            let wire: SampleWire = serde_json::from_slice(&line)
                .map_err(|e| format!("Invalid public policy sample wire: {e}"))?;
            let game = self.game.as_mut().unwrap();
            let step = game
                .record
                .steps
                .get(game.next_source_index)
                .ok_or("Excess native sample index")?;
            let expected = expected_sample(
                &game.state,
                step,
                &self.manifest.games[game.index],
                self.route,
            )?;
            if wire != expected.wire(self.route.feature_schema())? {
                return Err(format!(
                    "Stored policy sample differs from full source re-encoding at index {}",
                    game.next_source_index
                ));
            }
            game.candidates += expected.candidates.len();
            game.next_source_index += 1;
            game.state = apply_move(&game.state, step.chosen.r#move.clone())?;
            return Ok(Some(expected));
        }
    }
}
impl Iterator for PolicyDatasetIter {
    type Item = Result<ValidatedPolicySample, String>;
    fn next(&mut self) -> Option<Self::Item> {
        if self.failed {
            return None;
        }
        match self.read_next() {
            Ok(Some(sample)) => Some(Ok(sample)),
            Ok(None) => None,
            Err(error) => {
                self.failed = true;
                Some(Err(error))
            }
        }
    }
}
fn audit_all(dataset: &ValidatedPolicyDataset) -> Result<(), String> {
    let mut count = (0usize, 0usize);
    for sample in dataset.iter() {
        let sample = sample?;
        count.0 += 1;
        count.1 += sample.candidates.len();
    }
    if count != (dataset.manifest.samples, dataset.manifest.candidates) {
        return Err("Public policy total count mismatch".into());
    }
    Ok(())
}
/// Full native verification and exact source re-encoding precede returning this handle.
/// Each subsequent iterator independently verifies sources and shard EOF digests again.
/// An iterator must be consumed through EOF to complete its integrity pass.
pub fn load_policy_dataset(directory: &Path) -> Result<ValidatedPolicyDataset, String> {
    load_dataset_with_route(directory, SourceRoute::Native)
}
/// The dedicated teacher handle rechecks fixed v5 choices on every iteration.
/// Its distinct source/task labels cannot enter original native admission.
pub fn load_resource_cycle_v5_teacher_dataset(
    directory: &Path,
) -> Result<ValidatedPolicyDataset, String> {
    load_dataset_with_route(directory, SourceRoute::ResourceCycleV5Teacher)
}
/// Admit only freshly reconstructed semantic Setup teacher rows from complete sources.
/// The loader and every consumed iterator pin raw manifest, source and shard bytes.
pub fn load_resource_cycle_v5_semantic_setup_teacher_dataset(
    directory: &Path,
) -> Result<ValidatedSemanticSetupTeacherDataset, String> {
    let bytes = read_bytes(&directory.join("manifest.json"), MAX_MANIFEST_BYTES)?;
    let manifest: PolicyDatasetManifest =
        serde_json::from_slice(&bytes).map_err(|e| e.to_string())?;
    let route = SourceRoute::ResourceCycleV5SemanticSetupTeacher;
    validate_manifest(&manifest, route)?;
    let dataset = ValidatedSemanticSetupTeacherDataset {
        dataset: ValidatedPolicyDataset {
            directory: directory.to_path_buf(),
            manifest,
            route,
        },
        manifest_pin: (bytes.len() as u64, digest(&bytes)),
    };
    dataset.recheck_bytes()?;
    audit_all(&dataset.dataset)?;
    dataset.recheck_bytes()?;
    Ok(dataset)
}
fn load_dataset_with_route(
    directory: &Path,
    route: SourceRoute,
) -> Result<ValidatedPolicyDataset, String> {
    let bytes = read_bytes(&directory.join("manifest.json"), MAX_MANIFEST_BYTES)?;
    let manifest: PolicyDatasetManifest =
        serde_json::from_slice(&bytes).map_err(|e| e.to_string())?;
    validate_manifest(&manifest, route)?;
    let dataset = ValidatedPolicyDataset {
        directory: directory.to_path_buf(),
        manifest,
        route,
    };
    audit_all(&dataset)?;
    Ok(dataset)
}

#[cfg(test)]
mod tests {
    use super::*;
    mod test_temp_root {
        include!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/support/temp_root.rs"
        ));
    }
    fn directory() -> PathBuf {
        test_temp_root::create("tzolkin-policy-publication").unwrap()
    }
    fn incomplete_source_error(provenance: ReplaySource) -> String {
        let parent = directory();
        let path = parent.join("source.json");
        let record = GameReplay {
            header: ReplayHeader {
                replay_schema: replay::REPLAY_SCHEMA,
                rules_version: replay::RULES_VERSION,
                rules_baseline: replay::RULES_BASELINE.into(),
                catalog_hash: replay::catalog_hash(),
                move_schema: MOVE_SCHEMA,
                observation_schema: OBSERVATION_SCHEMA,
                source: provenance,
                names: vec!["A".into(), "B".into(), "C".into()],
                seed: 42,
                options: GameOptions::default(),
            },
            steps: Vec::new(),
            final_scores: Vec::new(),
            final_state: String::new(),
            verified_complete: false,
        };
        let raw = serde_json::to_vec(&record).unwrap();
        fs::write(&path, &raw).unwrap();
        let error = source(&path).unwrap_err();
        if matches!(
            &record.header.source,
            ReplaySource::CornFirstV5Teacher { .. }
        ) {
            assert_eq!(
                replay::verify_corn_first_v5_teacher_replay(&record).unwrap_err(),
                "Partial logs cannot be verified as complete games",
            );
            let output = parent.join("scalar-dataset");
            assert_eq!(
                crate::dataset::export_dataset_files(std::slice::from_ref(&path), &output)
                    .unwrap_err(),
                "Corn-first V5 Playing teacher has no legacy native dataset admission",
            );
            assert!(!output.exists());
            for route in [
                SourceRoute::ResourceCycleV5Teacher,
                SourceRoute::ResourceCycleV5SemanticSetupTeacher,
            ] {
                assert_eq!(
                    route.read(&path).unwrap_err(),
                    "ResourceCycleV5 teacher requires explicit all-seat policy provenance",
                );
            }
        }
        if matches!(
            &record.header.source,
            ReplaySource::CornFirstV7Teacher { .. }
        ) {
            assert_eq!(
                replay::verify_corn_first_v7_teacher_replay(&record).unwrap_err(),
                "Partial logs cannot be verified as complete games"
            );
            let output = parent.join("v7-scalar-dataset");
            assert_eq!(
                crate::dataset::export_dataset_files(std::slice::from_ref(&path), &output)
                    .unwrap_err(),
                "Corn-first V7 Playing teacher has no legacy native dataset admission"
            );
            assert!(!output.exists());
            for route in [
                SourceRoute::ResourceCycleV5Teacher,
                SourceRoute::ResourceCycleV5SemanticSetupTeacher,
            ] {
                assert_eq!(
                    route.read(&path).unwrap_err(),
                    "ResourceCycleV5 teacher requires explicit all-seat policy provenance"
                );
            }
        }
        assert_eq!(fs::read(&path).unwrap(), raw);
        fs::remove_dir_all(parent).unwrap();
        error
    }
    #[test]
    fn historical_self_play_versions_still_require_complete_replay() {
        for policy_version in ["heuristic-v1", crate::POLICY_VERSION] {
            assert_eq!(
                incomplete_source_error(ReplaySource::SelfPlay {
                    policy_version: policy_version.into(),
                    weights: crate::policy::HeuristicWeights::default(),
                }),
                "Partial logs cannot be verified as complete games",
            );
        }
    }
    #[test]
    fn historical_self_play_versions_reject_unknown_and_spoofed_provenance() {
        for policy_version in ["heuristic-v3", "unknown", "heuristic-v1 "] {
            assert_eq!(
                incomplete_source_error(ReplaySource::SelfPlay {
                    policy_version: policy_version.into(),
                    weights: crate::policy::HeuristicWeights::default(),
                }),
                "Unsupported native heuristic source version",
            );
        }
    }
    #[test]
    fn historical_self_play_versions_still_validate_weights() {
        let weights = crate::policy::HeuristicWeights {
            corn_base: 1001.0,
            ..Default::default()
        };
        assert_eq!(
            incomplete_source_error(ReplaySource::SelfPlay {
                policy_version: "heuristic-v1".into(),
                weights,
            }),
            "Heuristic coefficients must be finite and within -1000..1000",
        );
    }
    #[test]
    fn historical_self_play_versions_do_not_extend_other_source_routes() {
        replay::corn_first_v5_teacher_fixture();
        let parent = directory();
        crate::semantic_playing_teacher_dataset::fixture(&parent);
        fs::remove_dir_all(parent).unwrap();
        assert_eq!(
            incomplete_source_error(replay::corn_first_v5_teacher_provenance()),
            "Corn-first V5 Playing teacher has no legacy native dataset admission",
        );
        assert_eq!(
            incomplete_source_error(replay::corn_first_v7_teacher_provenance()),
            "Corn-first V7 Playing teacher requires its dedicated semantic4 dataset"
        );
        let policy = replay::SeatPolicy::Heuristic {
            policy_version: "heuristic-v1".into(),
            weights: crate::policy::HeuristicWeights::default(),
        };
        assert!(policy.validate().is_err());
        assert_eq!(
            incomplete_source_error(ReplaySource::PolicySelfPlay {
                policies: vec![policy; 3],
            }),
            "Unsupported selfplay seat-policy provenance",
        );
        assert_eq!(
            incomplete_source_error(ReplaySource::Human {
                provider: "synthetic".into(),
                reference: "example".into(),
                skill_rating: None,
            }),
            "Human/public source admission is not implemented",
        );
    }
    #[test]
    fn source_changed_on_second_pass_leaves_no_completion_marker() {
        let parent = directory();
        let output = parent.join("dataset");
        let record = replay::play_game_fast(3, 42, GameOptions::default(), true)
            .unwrap()
            .2
            .unwrap();
        let bytes = serde_json::to_vec(&record).unwrap();
        let mut reads = 0;
        let error = export_sources(
            1,
            |_| {
                reads += 1;
                let mut bytes = bytes.clone();
                if reads == 2 {
                    bytes.push(b' ');
                }
                Ok((bytes, record.clone()))
            },
            &output,
        )
        .unwrap_err();
        assert!(error.contains("changed between export passes"));
        assert!(output.is_dir());
        assert!(!output.join("manifest.json").exists());
        assert!(load_policy_dataset(&output).is_err());
        fs::remove_dir_all(parent).unwrap();
    }
    #[test]
    fn publication_never_overwrites_and_null_is_required() {
        let dir = directory();
        fs::write(dir.join("manifest.json"), b"original").unwrap();
        assert!(publish_bytes(&dir, "manifest.json", b"replacement").is_err());
        assert_eq!(fs::read(dir.join("manifest.json")).unwrap(), b"original");
        assert!(serde_json::from_str::<SampleWire>(r#"{"sampleId":"x","gameId":"x","familyId":"x","sourceIndex":0,"actor":0,"phase":"playing","candidates":[],"chosen":0}"#).is_err());
        fs::remove_dir_all(dir).unwrap();
    }
}
