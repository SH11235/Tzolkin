//! Complete-native, actor-specific Monte Carlo targets with sealed public state inputs.
//! Stored targets and checksums never replace native source reconstruction.
use crate::dataset::{DatasetSplit, split_for_family};
use crate::features::PUBLIC_FEATURE_SCHEMA;
use crate::model::digest;
use crate::policy_dataset::{
    PolicyDatasetGame, PolicyDatasetShard, PolicyDatasetStratum, read_state_native_source,
};
use crate::public_state_critic::{
    CONTEXT_CONTRACT, CONTEXT_COUNT, CONTEXT_SCHEMA, PublicStateContext,
};
use crate::replay::{self, GameReplay, ReplayStep};
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

pub const DATASET_SCHEMA: &str = "tzolkin-public-state-mc-dataset-v1";
pub const TASK: &str = "actorWinnerShareStateMc";
pub const TARGET_CONTRACT: &str = "complete-native-own-decision-mc-gamma1-v1";
pub const MAX_SOURCE_BYTES: u64 = 64 * 1024 * 1024;
pub const MAX_SAMPLE_BYTES: u64 = 64 * 1024;
pub const MAX_SHARD_BYTES: u64 = 32 * 1024 * 1024;
pub const MAX_MANIFEST_BYTES: u64 = 16 * 1024 * 1024;
pub const MAX_TOTAL_BYTES: u64 = 2 * 1024 * 1024 * 1024;
pub const MAX_FILES: usize = 4096;
pub const MAX_PARTITION_SAMPLES: usize = 100_000;
pub const MAX_LEGAL_ACTIONS: usize = 4096;

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct StateMcManifest {
    pub schema: String,
    pub task: String,
    pub target_contract: String,
    pub context_contract: String,
    pub feature_schema: u32,
    pub context_schema: u32,
    pub context_count: usize,
    pub rules_version: u32,
    pub rules_baseline: String,
    pub catalog_hash: String,
    pub move_schema: u32,
    pub observation_schema: u32,
    pub source_kind: String,
    pub gamma: u32,
    pub lambda: u32,
    pub terminal_bootstrap: f32,
    pub fingerprint: String,
    pub samples: usize,
    pub games: Vec<PolicyDatasetGame>,
    pub shards: Vec<PolicyDatasetShard>,
    pub strata: Vec<PolicyDatasetStratum>,
}
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct SampleWire {
    sample_id: String,
    game_id: String,
    family_id: String,
    source_index: usize,
    actor: usize,
    phase: Phase,
    context_checksum: String,
    #[serde(deserialize_with = "explicit_option")]
    next_own_index: Option<usize>,
    #[serde(deserialize_with = "explicit_option")]
    next_own_sample_id: Option<String>,
    terminal_reward: f32,
    #[serde(deserialize_with = "explicit_option")]
    terminal_bootstrap: Option<f32>,
    return_target: f32,
    winner_count: usize,
    actor_terminal_rank: usize,
}
fn explicit_option<'de, T: Deserialize<'de>, D: serde::Deserializer<'de>>(
    d: D,
) -> Result<Option<T>, D::Error> {
    Option::<T>::deserialize(d)
}
impl PartialEq for SampleWire {
    fn eq(&self, rhs: &Self) -> bool {
        self.sample_id == rhs.sample_id
            && self.game_id == rhs.game_id
            && self.family_id == rhs.family_id
            && self.source_index == rhs.source_index
            && self.actor == rhs.actor
            && self.phase == rhs.phase
            && self.context_checksum == rhs.context_checksum
            && self.next_own_index == rhs.next_own_index
            && self.next_own_sample_id == rhs.next_own_sample_id
            && self.terminal_reward.to_bits() == rhs.terminal_reward.to_bits()
            && self.terminal_bootstrap.map(f32::to_bits) == rhs.terminal_bootstrap.map(f32::to_bits)
            && self.return_target.to_bits() == rhs.return_target.to_bits()
            && self.winner_count == rhs.winner_count
            && self.actor_terminal_rank == rhs.actor_terminal_rank
    }
}
/// Only the loader can construct a sample after source, actor, context, target and links match.
/// Metadata is kept separate from the sole model input, `context()`.
/// ```compile_fail
/// use tzolkin_ai::state_mc_dataset::ValidatedStateMcSample;
/// let _: ValidatedStateMcSample = serde_json::from_str("{}").unwrap();
/// ```
#[derive(Clone, Debug)]
pub struct ValidatedStateMcSample {
    context: PublicStateContext,
    wire: SampleWire,
}
impl ValidatedStateMcSample {
    pub fn context(&self) -> &PublicStateContext {
        &self.context
    }
    pub fn sample_id(&self) -> &str {
        &self.wire.sample_id
    }
    pub fn game_id(&self) -> &str {
        &self.wire.game_id
    }
    pub fn family_id(&self) -> &str {
        &self.wire.family_id
    }
    pub fn source_index(&self) -> usize {
        self.wire.source_index
    }
    pub fn actor(&self) -> usize {
        self.wire.actor
    }
    pub fn phase(&self) -> Phase {
        self.wire.phase
    }
    pub fn context_checksum(&self) -> &str {
        &self.wire.context_checksum
    }
    pub fn next_own_index(&self) -> Option<usize> {
        self.wire.next_own_index
    }
    pub fn next_own_sample_id(&self) -> Option<&str> {
        self.wire.next_own_sample_id.as_deref()
    }
    pub fn terminal_reward(&self) -> f32 {
        self.wire.terminal_reward
    }
    pub fn terminal_bootstrap(&self) -> Option<f32> {
        self.wire.terminal_bootstrap
    }
    pub fn return_target(&self) -> f32 {
        self.wire.return_target
    }
    pub fn winner_count(&self) -> usize {
        self.wire.winner_count
    }
    pub fn actor_terminal_rank(&self) -> usize {
        self.wire.actor_terminal_rank
    }
}
#[derive(Clone)]
struct Decision {
    actor: usize,
    phase: Phase,
    checksum: String,
    id: String,
    next: Option<usize>,
}
struct Plan {
    decisions: Vec<Decision>,
    shares: Vec<f32>,
    ranks: Vec<usize>,
    winners: usize,
}
fn initial(record: &GameReplay) -> Result<GameState, String> {
    create_game_with_options(
        record.header.names.clone(),
        record.header.seed,
        record.header.options.clone(),
    )
}
fn context(state: &GameState, step: &ReplayStep) -> Result<PublicStateContext, String> {
    let obs = observe(state, step.actor)?;
    if obs != step.observation
        || step.actor != obs.actor
        || !matches!(obs.phase, Phase::Setup | Phase::Playing)
        || !(1..=MAX_LEGAL_ACTIONS).contains(&obs.legal_actions.len())
        || obs
            .legal_actions
            .iter()
            .filter(|action| **action == step.chosen)
            .count()
            != 1
    {
        return Err(format!(
            "State-MC source observation/whole legal mask mismatch at {}",
            step.index
        ));
    }
    PublicStateContext::from_observation(&obs)
}
fn winner_shares(
    scores: &[FinalScore],
    players: usize,
) -> Result<(Vec<f32>, Vec<usize>, usize), String> {
    if scores.len() != players || !(3..=4).contains(&players) {
        return Err("Invalid native terminal players".into());
    }
    let mut ranks = vec![0; players];
    for score in scores {
        if score.player_id >= players
            || ranks[score.player_id] != 0
            || !(1..=players).contains(&score.rank)
            || [
                score.points_before_final,
                score.resource_points,
                score.skull_points,
                score.monument_points,
                score.total,
            ]
            .iter()
            .any(|value| !value.is_finite())
        {
            return Err("Invalid verified terminal scores/ranks".into());
        }
        ranks[score.player_id] = score.rank;
    }
    let winners = ranks.iter().filter(|&&rank| rank == 1).count();
    if winners == 0 {
        return Err("Native terminal has no rank-one player".into());
    }
    let share = 1.0_f32 / winners as f32;
    Ok((
        ranks
            .iter()
            .map(|&rank| if rank == 1 { share } else { 0.0_f32 })
            .collect(),
        ranks,
        winners,
    ))
}
fn sample_id(
    game: &PolicyDatasetGame,
    index: usize,
    actor: usize,
    checksum: &str,
) -> Result<String, String> {
    let bytes = serde_json::to_vec(&(
        TASK,
        TARGET_CONTRACT,
        &game.game_id,
        &game.source_sha256,
        index,
        actor,
        checksum,
    ))
    .map_err(|e| e.to_string())?;
    let mut hash = Sha256::new();
    hash.update(b"tzolkin-public-state-mc-sample-v1\0");
    hash.update(bytes);
    Ok(hex(hash.finalize()))
}
fn plan(record: &GameReplay, game: &PolicyDatasetGame) -> Result<Plan, String> {
    let terminal = replay::verify_replay(record)?;
    let (shares, ranks, winners) = winner_shares(&terminal.final_scores, game.players)?;
    let mut state = initial(record)?;
    let mut decisions = Vec::with_capacity(record.steps.len());
    let mut actor_counts = vec![0; game.players];
    for (index, step) in record.steps.iter().enumerate() {
        if step.index != index {
            return Err("Unordered native source steps".into());
        }
        let ctx = context(&state, step)?;
        let checksum = ctx.context_checksum();
        decisions.push(Decision {
            actor: ctx.actor(),
            phase: ctx.phase(),
            id: sample_id(game, index, ctx.actor(), &checksum)?,
            checksum,
            next: None,
        });
        actor_counts[ctx.actor()] += 1;
        state = apply_move(&state, step.chosen.r#move.clone())?;
    }
    if state != terminal || actor_counts.contains(&0) {
        return Err("State-MC terminal/all-actor mismatch".into());
    }
    let mut next = vec![None; game.players];
    for (index, decision) in decisions.iter_mut().enumerate().rev() {
        decision.next = next[decision.actor];
        next[decision.actor] = Some(index);
    }
    Ok(Plan {
        decisions,
        shares,
        ranks,
        winners,
    })
}
fn wire(game: &PolicyDatasetGame, plan: &Plan, index: usize) -> SampleWire {
    let decision = &plan.decisions[index];
    let target = plan.shares[decision.actor];
    SampleWire {
        sample_id: decision.id.clone(),
        game_id: game.game_id.clone(),
        family_id: game.family_id.clone(),
        source_index: index,
        actor: decision.actor,
        phase: decision.phase,
        context_checksum: decision.checksum.clone(),
        next_own_index: decision.next,
        next_own_sample_id: decision.next.map(|i| plan.decisions[i].id.clone()),
        terminal_reward: if decision.next.is_none() {
            target
        } else {
            0.0_f32
        },
        terminal_bootstrap: decision.next.is_none().then_some(0.0_f32),
        return_target: target,
        winner_count: plan.winners,
        actor_terminal_rank: plan.ranks[decision.actor],
    }
}
fn hex(hash: impl AsRef<[u8]>) -> String {
    hash.as_ref().iter().map(|b| format!("{b:02x}")).collect()
}
fn digest_id(id: &str) -> bool {
    id.len() == 64
        && id
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}
fn bounded_json(value: &impl Serialize, max: u64) -> Result<Vec<u8>, String> {
    struct Sink {
        bytes: Vec<u8>,
        max: u64,
    }
    impl Write for Sink {
        fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
            if self.bytes.len() as u64 + bytes.len() as u64 > self.max {
                return Err(std::io::Error::other(
                    "State-MC serialization byte budget exceeded",
                ));
            }
            self.bytes.extend_from_slice(bytes);
            Ok(bytes.len())
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }
    let mut sink = Sink {
        bytes: Vec::new(),
        max,
    };
    serde_json::to_writer(&mut sink, value).map_err(|e| e.to_string())?;
    Ok(sink.bytes)
}
fn fingerprint(manifest: &StateMcManifest) -> Result<String, String> {
    let mut canonical = manifest.clone();
    canonical.fingerprint.clear();
    Ok(digest(&bounded_json(&canonical, MAX_MANIFEST_BYTES)?))
}
fn strata(games: &[PolicyDatasetGame]) -> Vec<PolicyDatasetStratum> {
    let mut groups: BTreeMap<(usize, String), (usize, usize, usize)> = BTreeMap::new();
    for game in games {
        let counts = groups
            .entry((game.players, game.policy_id.clone()))
            .or_default();
        counts.0 += 1;
        counts.1 += game.samples;
        counts.2 += game.candidates;
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
fn local_path(path: &Path) -> Result<(), String> {
    let text = path.to_string_lossy();
    if text.is_empty()
        || text.contains('\0')
        || text.contains("://")
        || text.starts_with("\\\\")
        || text.starts_with("//")
        || (!path.is_absolute() && matches!(path.components().next(), Some(Component::Prefix(_))))
    {
        return Err("State-MC requires an explicit local path".into());
    }
    let absolute = if path.is_absolute() {
        path.to_path_buf()
    } else {
        std::env::current_dir()
            .map_err(|e| e.to_string())?
            .join(path)
    };
    if absolute.to_string_lossy().starts_with("\\\\")
        || absolute.to_string_lossy().starts_with("//")
    {
        return Err("Network hierarchy rejected".into());
    }
    let mut ancestor = PathBuf::new();
    for part in absolute.components() {
        ancestor.push(part.as_os_str());
        if matches!(part, Component::Prefix(_)) {
            continue;
        }
        match fs::symlink_metadata(&ancestor) {
            Ok(meta) => {
                #[cfg(windows)]
                let reparse = {
                    use std::os::windows::fs::MetadataExt;
                    meta.file_attributes() & 0x400 != 0
                };
                #[cfg(not(windows))]
                let reparse = false;
                if meta.file_type().is_symlink() || reparse {
                    return Err("Symlink/reparse hierarchy rejected".into());
                }
            }
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
        return Err("State-MC file type/byte budget rejected".into());
    }
    let file = File::open(path).map_err(|e| e.to_string())?;
    if file.metadata().map_err(|e| e.to_string())?.len() > max {
        return Err("File grew beyond budget".into());
    }
    Ok(file)
}
fn read_bytes(path: &Path, max: u64) -> Result<Vec<u8>, String> {
    let mut bytes = Vec::new();
    regular_file(path, max)?
        .take(max + 1)
        .read_to_end(&mut bytes)
        .map_err(|e| e.to_string())?;
    if bytes.len() as u64 > max {
        return Err("Bounded file read exceeded".into());
    }
    Ok(bytes)
}
fn file_digest(path: &Path, max: u64) -> Result<(u64, String), String> {
    let mut reader = regular_file(path, max)?.take(max + 1);
    let mut hash = Sha256::new();
    let mut bytes = 0;
    let mut buffer = [0u8; 16 * 1024];
    loop {
        let count = reader.read(&mut buffer).map_err(|e| e.to_string())?;
        if count == 0 {
            break;
        }
        bytes += count as u64;
        if bytes > max {
            return Err("Digest byte budget exceeded".into());
        }
        hash.update(&buffer[..count]);
    }
    Ok((bytes, hex(hash.finalize())))
}
fn read_source(
    path: &Path,
    index: usize,
) -> Result<(Vec<u8>, GameReplay, PolicyDatasetGame), String> {
    // Check every local ancestor before the existing A2 source decoder can open the file.
    regular_file(path, MAX_SOURCE_BYTES)?;
    read_state_native_source(path, index)
}
fn publish(directory: &Path, name: &str, bytes: &[u8]) -> Result<(), String> {
    let temporary = directory.join(format!(".{name}.tmp"));
    let mut created = false;
    let result = (|| {
        let mut file = File::options()
            .write(true)
            .create_new(true)
            .open(&temporary)
            .map_err(|e| e.to_string())?;
        created = true;
        file.write_all(bytes)
            .and_then(|_| file.sync_all())
            .map_err(|e| e.to_string())?;
        drop(file);
        fs::hard_link(&temporary, directory.join(name)).map_err(|e| e.to_string())
    })();
    if created {
        let _ = fs::remove_file(&temporary);
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
        if bytes.len() as u64 + 1 > MAX_SAMPLE_BYTES
            || self.meta.bytes + bytes.len() as u64 + 1 > MAX_SHARD_BYTES
        {
            return Err("State-MC sample/shard byte budget exceeded".into());
        }
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
        let _ = fs::remove_file(&self.temporary);
        self.meta.sha256 = hex(self.hash.finalize());
        Ok(self.meta)
    }
}
/// Only immediate regular replay JSON files; no recursive traversal or network paths.
pub fn native_source_files(directory: &Path) -> Result<Vec<PathBuf>, String> {
    local_path(directory)?;
    if !fs::symlink_metadata(directory)
        .map_err(|e| e.to_string())?
        .is_dir()
    {
        return Err("Expected native source directory".into());
    }
    let mut paths = Vec::new();
    for entry in fs::read_dir(directory).map_err(|e| e.to_string())? {
        let path = entry.map_err(|e| e.to_string())?.path();
        if path.extension().and_then(|s| s.to_str()) != Some("json") {
            return Err("Source directory must contain only replay JSON".into());
        }
        regular_file(&path, MAX_SOURCE_BYTES)?;
        paths.push(path);
        if paths.len() + 2 > MAX_FILES {
            return Err("Native source file count exceeded".into());
        }
    }
    paths.sort();
    if paths.is_empty() {
        return Err("No native source replays".into());
    }
    Ok(paths)
}
/// Two full bounded source passes. A failed export can leave an incomplete new directory,
/// but cannot publish manifest.json. Inputs are never rewritten.
pub fn export_native_files(paths: &[PathBuf], output: &Path) -> Result<StateMcManifest, String> {
    export_sources(
        paths.len(),
        |index| read_source(&paths[index], index),
        output,
    )
}
fn export_sources(
    count: usize,
    mut read: impl FnMut(usize) -> Result<(Vec<u8>, GameReplay, PolicyDatasetGame), String>,
    output: &Path,
) -> Result<StateMcManifest, String> {
    local_path(output)?;
    match fs::symlink_metadata(output) {
        Ok(_) => return Err("State-MC destination already exists".into()),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
        Err(e) => return Err(e.to_string()),
    }
    if count == 0 || count > (MAX_FILES - 1) / 2 {
        return Err("Invalid State-MC source count".into());
    }
    let mut games = Vec::with_capacity(count);
    let mut seen = BTreeSet::new();
    let mut budgets = BTreeMap::new();
    let mut total_bytes = 0u64;
    let mut planned_shards = 0usize;
    for index in 0..count {
        let (bytes, record, game) = read(index)?;
        if game.source_sha256 != digest(&bytes) || !seen.insert(game.game_id.clone()) {
            return Err("Changed/duplicate native source".into());
        }
        let plan = plan(&record, &game)?;
        let mut shard_bytes = 0u64;
        for i in 0..plan.decisions.len() {
            let size = bounded_json(&wire(&game, &plan, i), MAX_SAMPLE_BYTES - 1)?.len() as u64 + 1;
            if shard_bytes == 0 {
                planned_shards += 1;
            }
            if shard_bytes + size > MAX_SHARD_BYTES {
                planned_shards += 1;
                shard_bytes = 0;
            }
            shard_bytes += size;
            total_bytes = total_bytes
                .checked_add(size)
                .ok_or("Dataset byte overflow")?;
        }
        total_bytes = total_bytes
            .checked_add(bytes.len() as u64)
            .ok_or("Dataset byte overflow")?;
        if total_bytes + MAX_MANIFEST_BYTES > MAX_TOTAL_BYTES
            || count + planned_shards + 1 > MAX_FILES
        {
            return Err("State-MC aggregate byte/file budget exceeded".into());
        }
        let samples = budgets.entry(game.split).or_insert(0usize);
        *samples = samples.checked_add(game.samples).ok_or("Sample overflow")?;
        if *samples > MAX_PARTITION_SAMPLES {
            return Err("State-MC partition sample budget exceeded".into());
        }
        games.push(game);
    }
    fs::create_dir(output).map_err(|e| e.to_string())?;
    fs::create_dir(output.join("sources")).map_err(|e| e.to_string())?;
    let mut manifest = StateMcManifest {
        schema: DATASET_SCHEMA.into(),
        task: TASK.into(),
        target_contract: TARGET_CONTRACT.into(),
        context_contract: CONTEXT_CONTRACT.into(),
        feature_schema: PUBLIC_FEATURE_SCHEMA,
        context_schema: CONTEXT_SCHEMA,
        context_count: CONTEXT_COUNT,
        rules_version: replay::RULES_VERSION,
        rules_baseline: replay::RULES_BASELINE.into(),
        catalog_hash: replay::catalog_hash(),
        move_schema: MOVE_SCHEMA,
        observation_schema: OBSERVATION_SCHEMA,
        source_kind: "verifiedCompleteNativeStateMc".into(),
        gamma: 1,
        lambda: 1,
        terminal_bootstrap: 0.0_f32,
        fingerprint: String::new(),
        samples: 0,
        games,
        shards: Vec::new(),
        strata: Vec::new(),
    };
    for index in 0..count {
        let (bytes, record, game) = read(index)?;
        if game != manifest.games[index] {
            return Err("Native source changed between export passes".into());
        }
        let plan = plan(&record, &game)?;
        publish(
            &output.join("sources"),
            &format!("source-{index:06}.json"),
            &bytes,
        )?;
        let mut writer: Option<ShardWriter> = None;
        for i in 0..plan.decisions.len() {
            let bytes = bounded_json(&wire(&game, &plan, i), MAX_SAMPLE_BYTES - 1)?;
            if writer
                .as_ref()
                .is_some_and(|w| w.meta.bytes + bytes.len() as u64 + 1 > MAX_SHARD_BYTES)
            {
                manifest.shards.push(writer.take().unwrap().finish()?);
            }
            if writer.is_none() {
                writer = Some(ShardWriter::new(output, manifest.shards.len(), index, i)?);
            }
            writer.as_mut().unwrap().append(&bytes)?;
        }
        if let Some(writer) = writer {
            manifest.shards.push(writer.finish()?);
        }
        manifest.samples += game.samples;
    }
    manifest.strata = strata(&manifest.games);
    manifest.fingerprint = fingerprint(&manifest)?;
    validate_manifest(&manifest)?;
    let dataset = ValidatedStateMcDataset {
        directory: output.to_path_buf(),
        manifest: manifest.clone(),
        manifest_sha: None,
    };
    dataset.iter().finish_checked()?;
    let bytes = bounded_json(&manifest, MAX_MANIFEST_BYTES)?;
    publish(output, "manifest.json", &bytes)?;
    Ok(manifest)
}
fn validate_manifest(m: &StateMcManifest) -> Result<(), String> {
    if m.schema != DATASET_SCHEMA
        || m.task != TASK
        || m.target_contract != TARGET_CONTRACT
        || m.context_contract != CONTEXT_CONTRACT
        || m.feature_schema != PUBLIC_FEATURE_SCHEMA
        || m.context_schema != CONTEXT_SCHEMA
        || m.context_count != CONTEXT_COUNT
        || m.rules_version != replay::RULES_VERSION
        || m.rules_baseline != replay::RULES_BASELINE
        || m.catalog_hash != replay::catalog_hash()
        || m.move_schema != MOVE_SCHEMA
        || m.observation_schema != OBSERVATION_SCHEMA
        || m.source_kind != "verifiedCompleteNativeStateMc"
        || m.gamma != 1
        || m.lambda != 1
        || m.terminal_bootstrap.to_bits() != 0
        || !digest_id(&m.fingerprint)
        || m.fingerprint != fingerprint(m)?
        || m.games.is_empty()
        || m.shards.is_empty()
        || m.games.len() + m.shards.len() + 1 > MAX_FILES
    {
        return Err("Unsupported/corrupt State-MC manifest".into());
    }
    let mut seen = BTreeSet::new();
    let mut budgets = BTreeMap::new();
    let mut total = 0usize;
    let mut bytes = MAX_MANIFEST_BYTES;
    for (index, g) in m.games.iter().enumerate() {
        if !digest_id(&g.game_id)
            || !seen.insert(&g.game_id)
            || !digest_id(&g.family_id)
            || !digest_id(&g.policy_id)
            || !digest_id(&g.source_sha256)
            || g.source_file != format!("sources/source-{index:06}.json")
            || g.source_bytes == 0
            || g.source_bytes > MAX_SOURCE_BYTES
            || !(3..=4).contains(&g.players)
            || g.options != GameOptions::default()
            || g.samples == 0
            || g.samples > replay::MAX_DECISIONS
            || g.candidates < g.samples
            || g.candidates > g.samples * MAX_LEGAL_ACTIONS
            || g.split != split_for_family(&g.family_id)?
        {
            return Err("Invalid State-MC source metadata".into());
        }
        let count = budgets.entry(g.split).or_insert(0usize);
        *count = count.checked_add(g.samples).ok_or("Count overflow")?;
        if *count > MAX_PARTITION_SAMPLES {
            return Err("State-MC partition count exceeded".into());
        }
        total += g.samples;
        bytes = bytes.checked_add(g.source_bytes).ok_or("Byte overflow")?;
    }
    let mut game_index = 0;
    let mut source_index = 0;
    for (index, shard) in m.shards.iter().enumerate() {
        let g = m.games.get(game_index).ok_or("Excess State-MC shards")?;
        if shard.file != format!("shard-{index:06}.jsonl")
            || !digest_id(&shard.sha256)
            || shard.bytes == 0
            || shard.bytes > MAX_SHARD_BYTES
            || shard.game_index != game_index
            || shard.first_source_index != source_index
            || shard.samples == 0
            || shard.samples > g.samples - source_index
        {
            return Err("Invalid State-MC shard order/count".into());
        }
        source_index += shard.samples;
        bytes = bytes.checked_add(shard.bytes).ok_or("Byte overflow")?;
        if source_index == g.samples {
            game_index += 1;
            source_index = 0;
        }
    }
    if game_index != m.games.len()
        || source_index != 0
        || total != m.samples
        || bytes > MAX_TOTAL_BYTES
        || m.strata != strata(&m.games)
    {
        return Err("State-MC totals/strata/aggregate budget mismatch".into());
    }
    Ok(())
}
pub struct ValidatedStateMcDataset {
    directory: PathBuf,
    manifest: StateMcManifest,
    manifest_sha: Option<String>,
}
impl ValidatedStateMcDataset {
    pub fn manifest(&self) -> &StateMcManifest {
        &self.manifest
    }
    pub fn iter(&self) -> StateMcIter {
        self.iterator(None)
    }
    /// Fresh integrity covers this split's sources/shards and the shared manifest.
    /// Use iter() to freshly recheck every partition after loading.
    pub fn iter_split(&self, split: DatasetSplit) -> StateMcIter {
        self.iterator(Some(split))
    }
    fn iterator(&self, split: Option<DatasetSplit>) -> StateMcIter {
        StateMcIter {
            directory: self.directory.clone(),
            manifest: self.manifest.clone(),
            manifest_sha: self.manifest_sha.clone(),
            split,
            next_shard: 0,
            shard: None,
            game: None,
            failure: None,
            complete: false,
            samples: 0,
            games: 0,
        }
    }
}
/// An integrity receipt requires complete iterator EOF, including final named-file rechecks.
/// It attests content consistency, not producer authentication or training eligibility.
/// A split receipt covers only selected source/shard files plus the shared manifest.
/// ```compile_fail
/// use tzolkin_ai::state_mc_dataset::StateMcIntegrityReceipt;
/// let _: StateMcIntegrityReceipt = serde_json::from_str("{}").unwrap();
/// ```
#[derive(Debug)]
pub struct StateMcIntegrityReceipt {
    fingerprint: String,
    split: Option<DatasetSplit>,
    samples: usize,
    games: usize,
}
impl StateMcIntegrityReceipt {
    pub fn fingerprint(&self) -> &str {
        &self.fingerprint
    }
    pub fn split(&self) -> Option<DatasetSplit> {
        self.split
    }
    pub fn samples(&self) -> usize {
        self.samples
    }
    pub fn games(&self) -> usize {
        self.games
    }
}
struct ReadingGame {
    index: usize,
    record: GameReplay,
    state: GameState,
    plan: Plan,
    next: usize,
}
struct ReadingShard {
    reader: BufReader<File>,
    hash: Sha256,
    meta: PolicyDatasetShard,
    bytes: u64,
    samples: usize,
}
pub struct StateMcIter {
    directory: PathBuf,
    manifest: StateMcManifest,
    manifest_sha: Option<String>,
    split: Option<DatasetSplit>,
    next_shard: usize,
    shard: Option<ReadingShard>,
    game: Option<ReadingGame>,
    failure: Option<String>,
    complete: bool,
    samples: usize,
    games: usize,
}
impl StateMcIter {
    /// Consumes all remaining samples. A previous error cannot be erased by another next().
    pub fn finish_checked(mut self) -> Result<StateMcIntegrityReceipt, String> {
        for sample in self.by_ref() {
            sample?;
        }
        if let Some(error) = self.failure {
            return Err(error);
        }
        if !self.complete {
            return Err("State-MC iterator EOF not verified".into());
        }
        Ok(StateMcIntegrityReceipt {
            fingerprint: self.manifest.fingerprint,
            split: self.split,
            samples: self.samples,
            games: self.games,
        })
    }
    fn finish_shard(&mut self) -> Result<(), String> {
        let shard = self.shard.take().ok_or("Missing reading shard")?;
        if shard.samples != shard.meta.samples
            || shard.bytes != shard.meta.bytes
            || hex(shard.hash.finalize()) != shard.meta.sha256
            || file_digest(&self.directory.join(&shard.meta.file), MAX_SHARD_BYTES)?
                != (shard.meta.bytes, shard.meta.sha256)
        {
            return Err("State-MC shard EOF checksum/count mismatch".into());
        }
        let game = self.game.as_ref().ok_or("Missing reading game")?;
        let meta = &self.manifest.games[game.index];
        if game.next == meta.samples {
            if game.state.phase != Phase::Finished
                || game.state.final_scores != game.record.final_scores
                || replay::state_key(&game.state)? != game.record.final_state
                || file_digest(&self.directory.join(&meta.source_file), MAX_SOURCE_BYTES)?
                    != (meta.source_bytes, meta.source_sha256.clone())
            {
                return Err("State-MC source terminal/EOF checksum mismatch".into());
            }
            self.games += 1;
            self.game = None;
        }
        Ok(())
    }
    fn verify_eof(&self) -> Result<(), String> {
        let included = |g: &PolicyDatasetGame| self.split.is_none_or(|split| g.split == split);
        if self.game.is_some()
            || self.shard.is_some()
            || self.games != self.manifest.games.iter().filter(|g| included(g)).count()
            || self.samples
                != self
                    .manifest
                    .games
                    .iter()
                    .filter(|g| included(g))
                    .map(|g| g.samples)
                    .sum::<usize>()
        {
            return Err("State-MC iterator total mismatch".into());
        }
        // Catch changes to a source/shard which had already reached EOF earlier in this pass.
        for g in self.manifest.games.iter().filter(|g| included(g)) {
            if file_digest(&self.directory.join(&g.source_file), MAX_SOURCE_BYTES)?
                != (g.source_bytes, g.source_sha256.clone())
            {
                return Err("State-MC source changed before whole EOF".into());
            }
        }
        for s in self
            .manifest
            .shards
            .iter()
            .filter(|s| included(&self.manifest.games[s.game_index]))
        {
            if file_digest(&self.directory.join(&s.file), MAX_SHARD_BYTES)?
                != (s.bytes, s.sha256.clone())
            {
                return Err("State-MC shard changed before whole EOF".into());
            }
        }
        if let Some(sha) = &self.manifest_sha
            && digest(&read_bytes(
                &self.directory.join("manifest.json"),
                MAX_MANIFEST_BYTES,
            )?) != *sha
        {
            return Err("State-MC manifest changed during iteration".into());
        }
        Ok(())
    }
    fn read_next(&mut self) -> Result<Option<ValidatedStateMcSample>, String> {
        loop {
            if self.shard.is_none() {
                let Some(meta) = self.manifest.shards.get(self.next_shard).cloned() else {
                    self.verify_eof()?;
                    self.complete = true;
                    return Ok(None);
                };
                self.next_shard += 1;
                let game_meta = &self.manifest.games[meta.game_index];
                if self.split.is_some_and(|split| split != game_meta.split) {
                    continue;
                }
                if self.game.is_none() {
                    let (_, record, reconstructed) = read_source(
                        &self.directory.join(&game_meta.source_file),
                        meta.game_index,
                    )?;
                    if reconstructed != *game_meta {
                        return Err("State-MC copied source metadata mismatch".into());
                    }
                    let plan = plan(&record, game_meta)?;
                    let state = initial(&record)?;
                    self.game = Some(ReadingGame {
                        index: meta.game_index,
                        record,
                        state,
                        plan,
                        next: 0,
                    });
                }
                let game = self.game.as_ref().unwrap();
                if game.index != meta.game_index || game.next != meta.first_source_index {
                    return Err("State-MC game/shard boundary mismatch".into());
                }
                let file = regular_file(&self.directory.join(&meta.file), MAX_SHARD_BYTES)?;
                if file.metadata().map_err(|e| e.to_string())?.len() != meta.bytes {
                    return Err("State-MC shard byte count mismatch".into());
                }
                self.shard = Some(ReadingShard {
                    reader: BufReader::new(file),
                    hash: Sha256::new(),
                    meta,
                    bytes: 0,
                    samples: 0,
                });
            }
            let shard = self.shard.as_mut().unwrap();
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
                return Err("Oversized/unterminated/excess State-MC sample".into());
            }
            shard.hash.update(&line);
            shard.bytes += count as u64;
            shard.samples += 1;
            let stored: SampleWire =
                serde_json::from_slice(&line).map_err(|e| format!("Invalid State-MC wire: {e}"))?;
            let game = self.game.as_mut().unwrap();
            let step = game
                .record
                .steps
                .get(game.next)
                .ok_or("Excess State-MC source index")?;
            let ctx = context(&game.state, step)?;
            let expected = wire(&self.manifest.games[game.index], &game.plan, game.next);
            if stored != expected || ctx.context_checksum() != expected.context_checksum {
                return Err(format!(
                    "State-MC wire/context/target/link differs from native source at {}",
                    game.next
                ));
            }
            game.state = apply_move(&game.state, step.chosen.r#move.clone())?;
            game.next += 1;
            self.samples += 1;
            return Ok(Some(ValidatedStateMcSample {
                context: ctx,
                wire: expected,
            }));
        }
    }
}
impl Iterator for StateMcIter {
    type Item = Result<ValidatedStateMcSample, String>;
    fn next(&mut self) -> Option<Self::Item> {
        if self.failure.is_some() || self.complete {
            return None;
        }
        match self.read_next() {
            Ok(Some(sample)) => Some(Ok(sample)),
            Ok(None) => None,
            Err(error) => {
                self.failure = Some(error.clone());
                Some(Err(error))
            }
        }
    }
}
/// Replays every complete source and every ordered sample to EOF before returning.
/// Later iterators repeat this pass and issue an integrity receipt only after whole EOF.
pub fn load_state_mc_dataset(directory: &Path) -> Result<ValidatedStateMcDataset, String> {
    let bytes = read_bytes(&directory.join("manifest.json"), MAX_MANIFEST_BYTES)?;
    let manifest: StateMcManifest = serde_json::from_slice(&bytes).map_err(|e| e.to_string())?;
    validate_manifest(&manifest)?;
    let dataset = ValidatedStateMcDataset {
        directory: directory.to_path_buf(),
        manifest,
        manifest_sha: Some(digest(&bytes)),
    };
    dataset.iter().finish_checked()?;
    Ok(dataset)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::time::{SystemTime, UNIX_EPOCH};
    fn directory() -> PathBuf {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let path = std::env::temp_dir().join(format!(
            "tzolkin-state-mc-private-{}-{}-{}",
            std::process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir(&path).unwrap();
        path
    }
    fn score(player_id: usize, rank: usize) -> FinalScore {
        FinalScore {
            player_id,
            rank,
            points_before_final: 0.0,
            resource_points: 0.0,
            skull_points: 0.0,
            monument_points: 0.0,
            total: 0.0,
            workers_on_gears: 0,
        }
    }
    #[test]
    fn direct_f32_three_way_share_and_canonical_positive_zero() {
        let (shares, _, k) =
            winner_shares(&[score(2, 1), score(0, 1), score(1, 1), score(3, 4)], 4).unwrap();
        assert_eq!(k, 3);
        assert_eq!(shares[0].to_bits(), (1.0_f32 / 3.0_f32).to_bits());
        assert_eq!(shares[3].to_bits(), 0);
        assert!(winner_shares(&[score(0, 1), score(0, 1), score(2, 3)], 3).is_err());
        assert!(winner_shares(&[score(0, 2), score(1, 2), score(2, 3)], 3).is_err());
    }
    #[test]
    fn source_changed_second_pass_cannot_publish_manifest() {
        let parent = directory();
        let output = parent.join("dataset");
        let source_path = parent.join("native.json");
        let record = replay::play_game_fast(3, 17, GameOptions::default(), true)
            .unwrap()
            .2
            .unwrap();
        fs::write(&source_path, serde_json::to_vec(&record).unwrap()).unwrap();
        let (bytes, record, game) = read_source(&source_path, 0).unwrap();
        let mut calls = 0;
        let error = export_sources(
            1,
            |_| {
                calls += 1;
                let mut bytes = bytes.clone();
                let mut game = game.clone();
                if calls == 2 {
                    bytes.push(b' ');
                    game.source_sha256 = digest(&bytes);
                    game.source_bytes += 1;
                }
                Ok((bytes, record.clone(), game))
            },
            &output,
        )
        .unwrap_err();
        assert!(error.contains("changed between export passes"));
        assert!(output.exists());
        assert!(!output.join("manifest.json").exists());
        assert!(load_state_mc_dataset(&output).is_err());
        fs::remove_dir_all(parent).unwrap();
    }
    #[test]
    fn bounded_publication_is_new_only_and_all_null_link_fields_are_required() {
        let dir = directory();
        fs::write(dir.join("manifest.json"), b"original").unwrap();
        assert!(publish(&dir, "manifest.json", b"replacement").is_err());
        assert_eq!(fs::read(dir.join("manifest.json")).unwrap(), b"original");
        assert!(!dir.join(".manifest.json.tmp").exists());
        assert!(bounded_json(&vec!["long"; 100], 8).is_err());
        let valid = serde_json::json!({"sampleId":"x","gameId":"x","familyId":"x","sourceIndex":0,"actor":0,"phase":"playing",
            "contextChecksum":"x","nextOwnIndex":null,"nextOwnSampleId":null,"terminalReward":0,"terminalBootstrap":0,"returnTarget":0,
            "winnerCount":1,"actorTerminalRank":3});
        assert!(serde_json::from_value::<SampleWire>(valid.clone()).is_ok());
        for field in ["nextOwnIndex", "nextOwnSampleId", "terminalBootstrap"] {
            let mut missing = valid.clone();
            missing.as_object_mut().unwrap().remove(field);
            assert!(serde_json::from_value::<SampleWire>(missing).is_err());
        }
        fs::remove_dir_all(dir).unwrap();
    }
}
