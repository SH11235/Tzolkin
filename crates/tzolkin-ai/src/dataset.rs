//! Immutable, checksummed, bounded training shards from independently verified native selfplay.
use crate::features::{FEATURE_COUNT, FEATURE_SCHEMA, FeatureEncoder, MAX_LEGAL_ACTIONS};
use crate::replay::{self, GameReplay, ReplaySource};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::borrow::Cow;
use std::collections::{BTreeMap, BTreeSet};
use std::fs::{self, File};
use std::io::{BufRead, BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use tzolkin_core::observation::{MOVE_SCHEMA, OBSERVATION_SCHEMA, observe};
use tzolkin_core::{GameOptions, apply_move, create_game_with_options};

pub const DATASET_SCHEMA: u32 = 1;
pub const MAX_SHARD_BYTES: u64 = 32 * 1024 * 1024;
pub const MAX_SAMPLE_BYTES: u64 = 16 * 1024 * 1024;
pub const MAX_MANIFEST_BYTES: u64 = 16 * 1024 * 1024;
pub const MAX_REPLAY_BYTES: u64 = 64 * 1024 * 1024;
const MAX_FILES: usize = 100_000;

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct TrainingSample {
    pub features: Vec<Vec<f32>>,
    pub chosen: usize,
    pub utilities: [f32; 5],
    pub active: [bool; 5],
    pub actor: usize,
    pub game_id: String,
    pub family_id: String,
}
#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(rename_all = "camelCase")]
pub enum DatasetSplit {
    Train,
    Validation,
    Test,
}
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct DatasetGame {
    pub game_id: String,
    pub family_id: String,
    pub split: DatasetSplit,
    pub samples: usize,
    pub players: usize,
    #[serde(deserialize_with = "deserialize_options")]
    pub options: GameOptions,
    pub policy_version: String,
}
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct DatasetShard {
    pub file: String,
    pub sha256: String,
    pub bytes: u64,
    pub samples: usize,
    pub game_id: String,
    pub family_id: String,
    pub split: DatasetSplit,
}
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct DatasetStratum {
    pub players: usize,
    #[serde(deserialize_with = "deserialize_options")]
    pub options: GameOptions,
    pub policy_version: String,
    pub games: usize,
    pub samples: usize,
}
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct DatasetManifest {
    pub schema: u32,
    pub feature_schema: u32,
    pub feature_count: usize,
    pub rules_version: u32,
    pub rules_baseline: String,
    pub catalog_hash: String,
    pub move_schema: u32,
    pub observation_schema: u32,
    pub source_kind: String,
    /// Digest of canonical manifest JSON with this field empty, including shard digests.
    pub fingerprint: String,
    pub samples: usize,
    pub games: Vec<DatasetGame>,
    pub shards: Vec<DatasetShard>,
    pub strata: Vec<DatasetStratum>,
}

fn deserialize_options<'de, D: serde::Deserializer<'de>>(
    deserializer: D,
) -> Result<GameOptions, D::Error> {
    #[derive(Deserialize)]
    #[serde(rename_all = "camelCase", deny_unknown_fields)]
    struct Options {
        additional_buildings: bool,
        tribes: bool,
        prophecies: bool,
        quick_actions: bool,
    }
    let o = Options::deserialize(deserializer)?;
    Ok(GameOptions {
        additional_buildings: o.additional_buildings,
        tribes: o.tribes,
        prophecies: o.prophecies,
        quick_actions: o.quick_actions,
    })
}

fn hex(digest: impl AsRef<[u8]>) -> String {
    digest.as_ref().iter().map(|b| format!("{b:02x}")).collect()
}
fn digest_id(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}
/// Split assignment is fixed by seed family, across options, seats and generating policies.
pub fn split_for_family(family_id: &str) -> Result<DatasetSplit, String> {
    if !digest_id(family_id) {
        return Err("Invalid seed-family digest".into());
    }
    let first = u8::from_str_radix(&family_id[..2], 16).map_err(|e| e.to_string())?;
    Ok(match first % 10 {
        0 => DatasetSplit::Validation,
        1 => DatasetSplit::Test,
        _ => DatasetSplit::Train,
    })
}
pub fn validate_sample(sample: &TrainingSample) -> Result<(), String> {
    let n = sample.active.iter().take_while(|v| **v).count();
    if !(2..=5).contains(&n)
        || sample.active[n..].iter().any(|v| *v)
        || sample.actor >= n
        || !digest_id(&sample.game_id)
        || !digest_id(&sample.family_id)
        || sample.features.is_empty()
        || sample.features.len() > MAX_LEGAL_ACTIONS
        || sample.chosen >= sample.features.len()
        || sample.features.iter().any(|row| {
            row.len() != FEATURE_COUNT || row.iter().any(|v| !v.is_finite() || v.abs() > 1024.0)
        })
        || sample
            .utilities
            .iter()
            .enumerate()
            .any(|(i, v)| !v.is_finite() || !(0.0..=1.0).contains(v) || (i >= n && *v != 0.0))
        || (sample.utilities.iter().sum::<f32>() - 1.0).abs() > 1e-5
    {
        return Err("Invalid training sample shape, mask, identity or finite value".into());
    }
    if sample
        .features
        .iter()
        .any(|row| row[..384] != sample.features[0][..384])
    {
        return Err("Candidate rows do not share one observation context".into());
    }
    Ok(())
}
fn manifest_digest(manifest: &DatasetManifest) -> Result<String, String> {
    let mut canonical = manifest.clone();
    canonical.fingerprint.clear();
    Ok(hex(Sha256::digest(
        serde_json::to_vec(&canonical).map_err(|e| e.to_string())?,
    )))
}
struct DigestWriter(Sha256);
impl Write for DigestWriter {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        self.0.update(bytes);
        Ok(bytes.len())
    }
    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}
fn game_id(replay: &GameReplay) -> Result<String, String> {
    let mut sink = DigestWriter(Sha256::new());
    sink.0.update(b"tzolkin-native-game-v1\0");
    serde_json::to_writer(&mut sink, replay).map_err(|e| e.to_string())?;
    Ok(hex(sink.0.finalize()))
}
/// Stable seed-family identity shared by dataset export and strength evaluation.
/// Player counts, seats, options and policy coefficients do not change the family.
pub fn seed_family_id(seed: u32) -> String {
    let mut hash = Sha256::new();
    hash.update(b"tzolkin-seed-family-v1\0");
    hash.update(seed.to_le_bytes());
    hex(hash.finalize())
}
fn strata(games: &[DatasetGame]) -> Vec<DatasetStratum> {
    let mut groups: BTreeMap<(usize, u8, String), (usize, usize)> = BTreeMap::new();
    for game in games {
        let o = &game.options;
        let flags = u8::from(o.additional_buildings)
            | u8::from(o.tribes) << 1
            | u8::from(o.prophecies) << 2
            | u8::from(o.quick_actions) << 3;
        let counts = groups
            .entry((game.players, flags, game.policy_version.clone()))
            .or_default();
        counts.0 += 1;
        counts.1 += game.samples;
    }
    groups
        .into_iter()
        .map(
            |((players, mask, policy_version), (games, samples))| DatasetStratum {
                players,
                options: replay::options_from_mask(mask),
                policy_version,
                games,
                samples,
            },
        )
        .collect()
}
struct ShardWriter {
    writer: BufWriter<File>,
    hash: Sha256,
    meta: DatasetShard,
}
impl ShardWriter {
    fn new(directory: &Path, index: usize, game: &DatasetGame) -> Result<Self, String> {
        let name = format!("shard-{index:06}.jsonl");
        let file = File::options()
            .write(true)
            .create_new(true)
            .open(directory.join(&name))
            .map_err(|e| e.to_string())?;
        Ok(Self {
            writer: BufWriter::new(file),
            hash: Sha256::new(),
            meta: DatasetShard {
                file: name,
                sha256: String::new(),
                bytes: 0,
                samples: 0,
                game_id: game.game_id.clone(),
                family_id: game.family_id.clone(),
                split: game.split,
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
    fn finish(mut self) -> Result<DatasetShard, String> {
        self.writer.flush().map_err(|e| e.to_string())?;
        self.writer
            .get_ref()
            .sync_all()
            .map_err(|e| e.to_string())?;
        self.meta.sha256 = hex(self.hash.finalize());
        Ok(self.meta)
    }
}

/// Refuses an existing destination. The manifest is published last; failed exports remain
/// visibly incomplete and cannot be loaded. The caller supplies full native replays, never
/// public prefixes. Every replay is reconstructed and verified independently of its flags.
pub fn export_dataset(
    replays: &[GameReplay],
    output_dir: &Path,
) -> Result<DatasetManifest, String> {
    export_sources(
        replays.len(),
        |index| Ok(Cow::Borrowed(&replays[index])),
        output_dir,
    )
}
/// Two bounded sequential passes retain one decoded replay and small game metadata.
/// The second pass checks that each source is identical to the verified first pass.
pub fn export_dataset_files(
    paths: &[PathBuf],
    output_dir: &Path,
) -> Result<DatasetManifest, String> {
    export_sources(
        paths.len(),
        |index| {
            let file = regular_file(&paths[index], MAX_REPLAY_BYTES)?;
            let mut reader = BufReader::new(file.take(MAX_REPLAY_BYTES + 1));
            let record = serde_json::from_reader(&mut reader)
                .map_err(|e| format!("Invalid native replay: {e}"))?;
            if reader.get_ref().limit() == 0 {
                return Err("Native replay exceeds read byte limit".into());
            }
            Ok(Cow::Owned(record))
        },
        output_dir,
    )
}
fn export_sources<'a>(
    count: usize,
    mut source: impl FnMut(usize) -> Result<Cow<'a, GameReplay>, String>,
    output_dir: &Path,
) -> Result<DatasetManifest, String> {
    if count == 0 || count > MAX_FILES {
        return Err("Invalid dataset game count".into());
    }
    if output_dir.exists() {
        return Err("Dataset destination already exists".into());
    }
    let mut prepared = Vec::with_capacity(count);
    let mut seen = BTreeSet::new();
    for index in 0..count {
        let record = source(index)?;
        let policy_version = match &record.header.source {
            ReplaySource::SelfPlay {
                policy_version,
                weights,
            } => {
                if policy_version.is_empty()
                    || policy_version.len() > 128
                    || weights
                        .material_values
                        .iter()
                        .chain([
                            &weights.corn_base,
                            &weights.corn_when_short,
                            &weights.temple_step,
                            &weights.technology_step,
                            &weights.worker,
                        ])
                        .any(|v| !v.is_finite())
                {
                    return Err("Invalid selfplay policy metadata".into());
                }
                policy_version.clone()
            }
            ReplaySource::PolicySelfPlay { policies } => {
                if policies.len() != record.header.names.len() {
                    return Err("Selfplay policy/seat count mismatch".into());
                }
                for policy in policies {
                    policy.validate()?;
                }
                // The complete replay keeps explicit per-seat model SHA/heuristic metadata;
                // strata use a content-bound identifier for that exact composition.
                format!(
                    "policy-selfplay-v1:{}",
                    hex(Sha256::digest(
                        serde_json::to_vec(policies).map_err(|e| e.to_string())?
                    ))
                )
            }
            ReplaySource::Human { .. } => {
                return Err("Training export accepts native SelfPlay only".into());
            }
        };
        let terminal = replay::verify_replay(&record)?;
        let winners = terminal.final_scores.iter().filter(|s| s.rank == 1).count();
        if winners == 0 {
            return Err("Verified game has no winner".into());
        }
        let mut utilities = [0.0; 5];
        for score in &terminal.final_scores {
            if score.rank == 1 {
                utilities[score.player_id] = 1.0 / winners as f32;
            }
        }
        let id = game_id(&record)?;
        if !seen.insert(id.clone()) {
            return Err("Duplicate game in dataset".into());
        }
        let family = seed_family_id(record.header.seed);
        let game = DatasetGame {
            game_id: id,
            family_id: family.clone(),
            split: split_for_family(&family)?,
            samples: record.steps.len(),
            players: terminal.players.len(),
            options: record.header.options.clone(),
            policy_version,
        };
        prepared.push((game, utilities));
    }
    fs::create_dir(output_dir).map_err(|e| format!("Cannot create new dataset directory: {e}"))?;
    let mut manifest = DatasetManifest {
        schema: DATASET_SCHEMA,
        feature_schema: FEATURE_SCHEMA,
        feature_count: FEATURE_COUNT,
        rules_version: replay::RULES_VERSION,
        rules_baseline: replay::RULES_BASELINE.into(),
        catalog_hash: replay::catalog_hash(),
        move_schema: MOVE_SCHEMA,
        observation_schema: OBSERVATION_SCHEMA,
        source_kind: "selfPlay".into(),
        fingerprint: String::new(),
        samples: 0,
        games: Vec::new(),
        shards: Vec::new(),
        strata: Vec::new(),
    };
    for (index, (game, utilities)) in prepared.into_iter().enumerate() {
        let record = source(index)?;
        if game_id(&record)? != game.game_id {
            return Err("Replay source changed after independent verification".into());
        }
        let mut state = create_game_with_options(
            record.header.names.clone(),
            record.header.seed,
            record.header.options.clone(),
        )?;
        let mut current: Option<ShardWriter> = None;
        for step in &record.steps {
            let observation = observe(&state, state.current_player)?;
            let encoder = FeatureEncoder::new(&observation)?;
            let chosen = observation
                .legal_actions
                .iter()
                .position(|a| *a == step.chosen)
                .ok_or("Missing verified chosen action")?;
            let mut active = [false; 5];
            let mut rotated = [0.0; 5];
            for i in 0..game.players {
                active[i] = true;
                rotated[i] = utilities[(observation.actor + i) % game.players];
            }
            let sample = TrainingSample {
                features: (0..observation.legal_actions.len())
                    .map(|i| encoder.encode_legal(i).map(Vec::from))
                    .collect::<Result<_, _>>()?,
                chosen,
                utilities: rotated,
                active,
                actor: observation.actor,
                game_id: game.game_id.clone(),
                family_id: game.family_id.clone(),
            };
            validate_sample(&sample)?;
            let bytes = serde_json::to_vec(&sample).map_err(|e| e.to_string())?;
            if bytes.len() as u64 + 1 > MAX_SAMPLE_BYTES {
                return Err("Training sample exceeds byte limit".into());
            }
            if current
                .as_ref()
                .is_some_and(|s| s.meta.bytes + bytes.len() as u64 + 1 > MAX_SHARD_BYTES)
            {
                manifest.shards.push(current.take().unwrap().finish()?);
            }
            if current.is_none() {
                if manifest.shards.len() >= MAX_FILES {
                    return Err("Dataset exceeds shard limit".into());
                }
                current = Some(ShardWriter::new(output_dir, manifest.shards.len(), &game)?);
            }
            current.as_mut().unwrap().append(&bytes)?;
            state = apply_move(&state, step.chosen.r#move.clone())?;
        }
        if let Some(writer) = current {
            manifest.shards.push(writer.finish()?);
        }
        manifest.samples += game.samples;
        manifest.games.push(game);
    }
    manifest.strata = strata(&manifest.games);
    manifest.fingerprint = manifest_digest(&manifest)?;
    validate_manifest(&manifest)?;
    let bytes = serde_json::to_vec_pretty(&manifest).map_err(|e| e.to_string())?;
    if bytes.len() as u64 > MAX_MANIFEST_BYTES {
        return Err("Dataset manifest exceeds byte limit".into());
    }
    let temporary = output_dir.join(".manifest.tmp");
    let destination = output_dir.join("manifest.json");
    let mut file = File::options()
        .write(true)
        .create_new(true)
        .open(&temporary)
        .map_err(|e| e.to_string())?;
    file.write_all(&bytes)
        .and_then(|_| file.sync_all())
        .map_err(|e| e.to_string())?;
    drop(file);
    if destination.exists() {
        return Err("Dataset manifest already exists".into());
    }
    fs::rename(temporary, destination).map_err(|e| e.to_string())?;
    Ok(manifest)
}
fn validate_manifest(m: &DatasetManifest) -> Result<(), String> {
    if m.schema != DATASET_SCHEMA
        || m.feature_schema != FEATURE_SCHEMA
        || m.feature_count != FEATURE_COUNT
        || m.rules_version != replay::RULES_VERSION
        || m.rules_baseline != replay::RULES_BASELINE
        || m.catalog_hash != replay::catalog_hash()
        || m.move_schema != MOVE_SCHEMA
        || m.observation_schema != OBSERVATION_SCHEMA
        || m.source_kind != "selfPlay"
        || m.games.is_empty()
        || m.games.len() > MAX_FILES
        || m.shards.is_empty()
        || m.shards.len() > MAX_FILES
        || !digest_id(&m.fingerprint)
        || m.fingerprint != manifest_digest(m)?
    {
        return Err("Unsupported or corrupt dataset manifest".into());
    }
    let mut games = BTreeMap::new();
    let mut family_splits = BTreeMap::new();
    let mut total = 0usize;
    for game in &m.games {
        if !digest_id(&game.game_id)
            || !digest_id(&game.family_id)
            || !(2..=5).contains(&game.players)
            || game.samples == 0
            || game.samples > replay::MAX_DECISIONS
            || game.policy_version.is_empty()
            || game.policy_version.len() > 128
            || (game.players == 5 && !game.options.quick_actions)
            || game.split != split_for_family(&game.family_id)?
            || games.insert(game.game_id.clone(), game).is_some()
        {
            return Err("Invalid or duplicate dataset game".into());
        }
        if family_splits
            .insert(&game.family_id, game.split)
            .is_some_and(|s| s != game.split)
        {
            return Err("Seed family crosses partitions".into());
        }
        total = total
            .checked_add(game.samples)
            .ok_or("Dataset sample count overflow")?;
    }
    let mut counts: BTreeMap<&str, usize> = BTreeMap::new();
    for (index, shard) in m.shards.iter().enumerate() {
        let game = games.get(&shard.game_id).ok_or("Unknown shard game")?;
        if shard.file != format!("shard-{index:06}.jsonl")
            || !digest_id(&shard.sha256)
            || shard.bytes == 0
            || shard.bytes > MAX_SHARD_BYTES
            || shard.samples == 0
            || shard.samples > game.samples
            || shard.family_id != game.family_id
            || shard.split != game.split
        {
            return Err("Invalid shard metadata or unsafe file name".into());
        }
        let count = counts.entry(&shard.game_id).or_default();
        *count = count
            .checked_add(shard.samples)
            .ok_or("Shard count overflow")?;
    }
    if total != m.samples
        || m.games
            .iter()
            .any(|g| counts.get(g.game_id.as_str()).copied() != Some(g.samples))
        || m.strata != strata(&m.games)
    {
        return Err("Dataset counts or strata mismatch".into());
    }
    Ok(())
}

pub struct ValidatedDataset {
    directory: PathBuf,
    manifest: DatasetManifest,
}
impl ValidatedDataset {
    pub fn manifest(&self) -> &DatasetManifest {
        &self.manifest
    }
    pub fn iter(&self) -> DatasetIter {
        self.iterator(None)
    }
    pub fn iter_split(&self, split: DatasetSplit) -> DatasetIter {
        self.iterator(Some(split))
    }
    fn iterator(&self, split: Option<DatasetSplit>) -> DatasetIter {
        DatasetIter {
            directory: self.directory.clone(),
            shards: self
                .manifest
                .shards
                .iter()
                .filter(|s| split.is_none_or(|x| x == s.split))
                .cloned()
                .collect(),
            players: self
                .manifest
                .games
                .iter()
                .map(|g| (g.game_id.clone(), g.players))
                .collect(),
            next_shard: 0,
            current: None,
            failed: false,
        }
    }
}
struct ReadingShard {
    reader: BufReader<File>,
    hash: Sha256,
    meta: DatasetShard,
    bytes: u64,
    samples: usize,
}
pub struct DatasetIter {
    directory: PathBuf,
    shards: Vec<DatasetShard>,
    players: BTreeMap<String, usize>,
    next_shard: usize,
    current: Option<ReadingShard>,
    failed: bool,
}
fn regular_file(path: &Path, limit: u64) -> Result<File, String> {
    let meta = fs::symlink_metadata(path).map_err(|e| e.to_string())?;
    if !meta.is_file() || meta.file_type().is_symlink() || meta.len() > limit {
        return Err("Dataset file type or size rejected".into());
    }
    File::open(path).map_err(|e| e.to_string())
}
impl DatasetIter {
    fn read_next(&mut self) -> Result<Option<TrainingSample>, String> {
        loop {
            if self.current.is_none() {
                let Some(meta) = self.shards.get(self.next_shard).cloned() else {
                    return Ok(None);
                };
                let file = regular_file(&self.directory.join(&meta.file), MAX_SHARD_BYTES)?;
                if file.metadata().map_err(|e| e.to_string())?.len() != meta.bytes {
                    return Err("Shard byte count mismatch".into());
                }
                self.next_shard += 1;
                self.current = Some(ReadingShard {
                    reader: BufReader::new(file),
                    hash: Sha256::new(),
                    meta,
                    bytes: 0,
                    samples: 0,
                });
            }
            let current = self.current.as_mut().unwrap();
            let mut line = Vec::new();
            let read = current
                .reader
                .by_ref()
                .take(MAX_SAMPLE_BYTES + 1)
                .read_until(b'\n', &mut line)
                .map_err(|e| e.to_string())?;
            if read == 0 {
                let completed = self.current.take().unwrap();
                if completed.samples != completed.meta.samples
                    || completed.bytes != completed.meta.bytes
                    || hex(completed.hash.finalize()) != completed.meta.sha256
                {
                    return Err("Shard checksum or sample count mismatch".into());
                }
                continue;
            }
            if read as u64 > MAX_SAMPLE_BYTES || line.last() != Some(&b'\n') {
                return Err("Oversized or unterminated training sample".into());
            }
            current.hash.update(&line);
            current.bytes += read as u64;
            current.samples += 1;
            if current.samples > current.meta.samples {
                return Err("Shard has excess samples".into());
            }
            let sample: TrainingSample = serde_json::from_slice(&line)
                .map_err(|e| format!("Invalid training sample: {e}"))?;
            validate_sample(&sample)?;
            let n = self.players[&current.meta.game_id];
            if sample.game_id != current.meta.game_id
                || sample.family_id != current.meta.family_id
                || sample.actor >= n
                || sample.active.iter().filter(|v| **v).count() != n
            {
                return Err("Training sample game/family/seat mismatch".into());
            }
            return Ok(Some(sample));
        }
    }
}
impl Iterator for DatasetIter {
    type Item = Result<TrainingSample, String>;
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
/// Performs a bounded complete integrity/shape pass before returning an iterable handle.
/// Iterators recheck digests at each shard boundary so subsequent file changes fail too.
pub fn load_dataset(output_dir: &Path) -> Result<ValidatedDataset, String> {
    let file = regular_file(&output_dir.join("manifest.json"), MAX_MANIFEST_BYTES)?;
    let mut reader = BufReader::new(file.take(MAX_MANIFEST_BYTES + 1));
    let manifest: DatasetManifest =
        serde_json::from_reader(&mut reader).map_err(|e| e.to_string())?;
    if reader.get_ref().limit() == 0 {
        return Err("Manifest exceeds read byte limit".into());
    }
    validate_manifest(&manifest)?;
    let dataset = ValidatedDataset {
        directory: output_dir.to_path_buf(),
        manifest,
    };
    let mut count = 0usize;
    for sample in dataset.iter() {
        sample?;
        count += 1;
    }
    if count != dataset.manifest.samples {
        return Err("Dataset total sample count mismatch".into());
    }
    Ok(dataset)
}
