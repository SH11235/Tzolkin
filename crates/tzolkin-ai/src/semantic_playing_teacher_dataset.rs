//! A separate, policy-only native teacher dataset. Complete fixed-teacher
//! replays remain authoritative; Setup callbacks are verified but are not rows.
//! This grants no Human origin, new allocation, model admission, RL credit or
//! policy quality. Family roles do not establish globally unexposed Train data.
use crate::dataset::{DatasetSplit, seed_family_id, split_for_family};
use crate::features::{
    EncodedCandidate, FEATURE_COUNT, FeatureEncoder, SEMANTIC_PLAYING_FEATURE_SCHEMA,
};
use crate::model::digest;
use crate::policy_dataset::{self, content_id, digest_id};
use crate::replay::{self, GameReplay};
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::fs::{self, File};
use std::io::Write;
use std::path::{Component, Path, PathBuf};
use tzolkin_core::observation::{
    LegalAction, MOVE_SCHEMA, OBSERVATION_SCHEMA, Observation, observe,
};
use tzolkin_core::{GameOptions, Phase, apply_move, create_game_with_options};

pub const DATASET_SCHEMA: &str = "tzolkin-semantic-playing-native-teacher-dataset-v1";
pub const SOURCE_KIND: &str = "verifiedNativeCornFirstV5PlayingTeacher";
pub const TASK: &str = "cornFirstV5SemanticPlayingPolicyOnlyTeacher";
pub const INPUT_CONTRACT: &str = "native-corn-first-v5-playing-policy-only-semantic4-v1";
pub const MAX_STORED_BYTES: u64 = 256 * 1024 * 1024;
/// Retained encoded allocations, not a promise about total process RSS.
pub const MAX_CACHE_BYTES: usize = 256 * 1024 * 1024;
pub const MAX_GAMES: usize = 64;
pub const MAX_ROWS: usize = policy_dataset::MAX_PARTITION_SAMPLES;
pub const MAX_CANDIDATES: usize = policy_dataset::MAX_PARTITION_CANDIDATES;

#[derive(Clone, Debug, Default, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct TeacherCounts {
    pub setup_callbacks: usize,
    pub playing_callbacks: usize,
    pub rows: usize,
    pub candidates: usize,
}
impl TeacherCounts {
    fn add(&mut self, other: &Self) -> Result<(), String> {
        self.setup_callbacks = self
            .setup_callbacks
            .checked_add(other.setup_callbacks)
            .ok_or("Callback count overflow")?;
        self.playing_callbacks = self
            .playing_callbacks
            .checked_add(other.playing_callbacks)
            .ok_or("Callback count overflow")?;
        self.rows = self
            .rows
            .checked_add(other.rows)
            .ok_or("Row count overflow")?;
        self.candidates = self
            .candidates
            .checked_add(other.candidates)
            .ok_or("Candidate count overflow")?;
        if self.rows > MAX_ROWS
            || self.candidates > MAX_CANDIDATES
            || self.rows != self.playing_callbacks
        {
            return Err("Native semantic teacher aggregate row/candidate budget exceeded".into());
        }
        Ok(())
    }
}
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct TeacherGame {
    pub game_id: String,
    pub family_id: String,
    pub seed: u32,
    pub split: DatasetSplit,
    pub players: usize,
    pub source_file: String,
    pub source_bytes: u64,
    pub source_sha256: String,
    pub row_file: String,
    pub row_bytes: u64,
    pub row_sha256: String,
    pub counts: TeacherCounts,
}
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct TeacherFamily {
    pub family_id: String,
    pub split: DatasetSplit,
}
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct TeacherStratum {
    pub players: usize,
    pub games: usize,
    pub counts: TeacherCounts,
}
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct SemanticPlayingTeacherManifest {
    pub schema: String,
    pub source_kind: String,
    pub task: String,
    pub input_contract: String,
    pub feature_schema: u32,
    pub feature_count: usize,
    pub rules_version: u32,
    pub rules_baseline: String,
    pub catalog_hash: String,
    pub move_schema: u32,
    pub observation_schema: u32,
    #[serde(deserialize_with = "policy_dataset::strict_options")]
    pub options: GameOptions,
    pub teacher_policy_id: String,
    pub value_supervised: bool,
    pub fingerprint: String,
    /// Source plus row bytes; the manifest is also charged to the aggregate cap.
    pub stored_bytes: u64,
    pub counts: TeacherCounts,
    pub families: Vec<TeacherFamily>,
    pub games: Vec<TeacherGame>,
    pub strata: Vec<TeacherStratum>,
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
struct RowWire {
    sample_id: String,
    game_id: String,
    family_id: String,
    source_callback_index: usize,
    actor: usize,
    phase: Phase,
    ordered_legal_sha256: String,
    candidates: Vec<CandidateWire>,
    chosen: usize,
    value_target: (),
}
/// Created only from full teacher replay re-derivation, never from row floats.
#[derive(Debug)]
pub struct ValidatedSemanticPlayingTeacherSample {
    sample_id: String,
    game_id: String,
    family_id: String,
    source_callback_index: usize,
    actor: usize,
    ordered_legal_sha256: String,
    candidates: Vec<EncodedCandidate>,
    chosen: usize,
}
impl ValidatedSemanticPlayingTeacherSample {
    pub fn sample_id(&self) -> &str {
        &self.sample_id
    }
    pub fn game_id(&self) -> &str {
        &self.game_id
    }
    pub fn family_id(&self) -> &str {
        &self.family_id
    }
    pub fn source_callback_index(&self) -> usize {
        self.source_callback_index
    }
    pub fn actor(&self) -> usize {
        self.actor
    }
    /// Hash of the complete ordered legal DTOs retained in the original replay.
    pub fn ordered_legal_sha256(&self) -> &str {
        &self.ordered_legal_sha256
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
    fn allocation_bytes(&self) -> Result<usize, String> {
        self.candidates
            .capacity()
            .checked_mul(std::mem::size_of::<EncodedCandidate>())
            .and_then(|n| n.checked_add(std::mem::size_of::<Self>()))
            .and_then(|n| n.checked_add(self.sample_id.capacity()))
            .and_then(|n| n.checked_add(self.game_id.capacity()))
            .and_then(|n| n.checked_add(self.family_id.capacity()))
            .and_then(|n| n.checked_add(self.ordered_legal_sha256.capacity()))
            .ok_or_else(|| "Native semantic teacher cache allocation overflow".into())
    }
    fn wire(&self) -> Result<RowWire, String> {
        Ok(RowWire {
            sample_id: self.sample_id.clone(),
            game_id: self.game_id.clone(),
            family_id: self.family_id.clone(),
            source_callback_index: self.source_callback_index,
            actor: self.actor,
            phase: Phase::Playing,
            ordered_legal_sha256: self.ordered_legal_sha256.clone(),
            chosen: self.chosen,
            value_target: (),
            candidates: self
                .candidates
                .iter()
                .map(|c| {
                    Ok(CandidateWire {
                        feature_schema: c.feature_schema(),
                        values: c
                            .values_for_schema(SEMANTIC_PLAYING_FEATURE_SCHEMA)?
                            .to_vec(),
                    })
                })
                .collect::<Result<_, String>>()?,
        })
    }
}
fn json<T: DeserializeOwned>(bytes: &[u8], cap: u64) -> Result<T, String> {
    serde_json::from_value(crate::public_stochastic_record::unique_json(
        bytes,
        cap as usize,
    )?)
    .map_err(|e| e.to_string())
}
fn local(path: &Path) -> Result<(), String> {
    policy_dataset::local_path(path)?;
    let absolute = if path.is_absolute() {
        path.to_path_buf()
    } else {
        std::env::current_dir()
            .map_err(|e| e.to_string())?
            .join(path)
    };
    let mut p = PathBuf::new();
    for part in absolute.components() {
        p.push(part.as_os_str());
        if matches!(part, Component::Prefix(_)) {
            continue;
        }
        if let Ok(m) = fs::symlink_metadata(&p) {
            #[cfg(windows)]
            {
                use std::os::windows::fs::MetadataExt;
                if m.file_attributes() & 0x400 != 0 {
                    return Err("Reparse teacher input/output hierarchy rejected".into());
                }
            }
            if m.file_type().is_symlink() {
                return Err("Symlink teacher input/output hierarchy rejected".into());
            }
        }
    }
    Ok(())
}
fn read(path: &Path, cap: u64) -> Result<Vec<u8>, String> {
    local(path)?;
    policy_dataset::read_bytes(path, cap)
}
fn raw_pin(path: &Path, cap: u64) -> Result<(u64, String), String> {
    local(path)?;
    policy_dataset::file_digest(path, cap)
}
fn teacher_id() -> Result<String, String> {
    content_id(
        b"tzolkin-semantic-playing-native-teacher-v1\0",
        &replay::corn_first_v5_teacher_provenance(),
    )
}
fn fingerprint(m: &SemanticPlayingTeacherManifest) -> Result<String, String> {
    let mut m = m.clone();
    m.fingerprint.clear();
    Ok(digest(&serde_json::to_vec(&m).map_err(|e| e.to_string())?))
}
fn sample(
    observation: &Observation,
    chosen: &LegalAction,
    index: usize,
    game: &str,
    family: &str,
) -> Result<ValidatedSemanticPlayingTeacherSample, String> {
    if observation.phase != Phase::Playing {
        return Err("Native semantic teacher rows are Playing only".into());
    }
    let choices: Vec<_> = observation
        .legal_actions
        .iter()
        .enumerate()
        .filter_map(|(i, a)| (a == chosen).then_some(i))
        .collect();
    if choices.len() != 1 {
        return Err("Teacher chosen action is not unique in ordered legal set".into());
    }
    let encoder = FeatureEncoder::new_semantic_playing(observation)?;
    let candidates: Vec<_> = (0..observation.legal_actions.len())
        .map(|i| encoder.encode_legal_tagged(i))
        .collect::<Result<_, _>>()?;
    for row in &candidates {
        if row
            .values_for_schema(SEMANTIC_PLAYING_FEATURE_SCHEMA)?
            .iter()
            .any(|v| !v.is_finite() || v.abs() > 1024.0)
        {
            return Err("Nonfinite/out-of-bound semantic teacher features".into());
        }
    }
    Ok(ValidatedSemanticPlayingTeacherSample {
        sample_id: content_id(
            b"tzolkin-native-playing4-sample-v1\0",
            &(game, index, observation.actor),
        )?,
        game_id: game.into(),
        family_id: family.into(),
        source_callback_index: index,
        actor: observation.actor,
        ordered_legal_sha256: content_id(
            b"tzolkin-native-playing4-legal-order-v1\0",
            &observation.legal_actions,
        )?,
        candidates,
        chosen: choices[0],
    })
}
fn derive_game(
    index: usize,
    bytes: &[u8],
    record: &GameReplay,
    mut row: impl FnMut(ValidatedSemanticPlayingTeacherSample, &[u8]) -> Result<(), String>,
) -> Result<TeacherGame, String> {
    // Header/ordinary Core replay verification alone is intentionally insufficient.
    let terminal = replay::verify_corn_first_v5_teacher_replay(record)?;
    let game_id = content_id(b"tzolkin-native-playing4-game-v1\0", record)?;
    let family_id = seed_family_id(record.header.seed);
    let mut state = create_game_with_options(
        record.header.names.clone(),
        record.header.seed,
        record.header.options.clone(),
    )?;
    let mut counts = TeacherCounts::default();
    let mut row_bytes = 0u64;
    let mut hash = Sha256::new();
    for (callback, step) in record.steps.iter().enumerate() {
        let observation = observe(&state, state.current_player)?;
        if callback != step.index
            || observation != step.observation
            || observation.actor != step.actor
        {
            return Err("Teacher original callback/actor/state mismatch".into());
        }
        match observation.phase {
            Phase::Setup => counts.setup_callbacks += 1,
            Phase::Playing => {
                let s = sample(&observation, &step.chosen, callback, &game_id, &family_id)?;
                let mut line = serde_json::to_vec(&s.wire()?).map_err(|e| e.to_string())?;
                line.push(b'\n');
                if line.len() as u64 > policy_dataset::MAX_SAMPLE_BYTES {
                    return Err("Semantic teacher row byte budget exceeded".into());
                }
                row_bytes = row_bytes
                    .checked_add(line.len() as u64)
                    .ok_or("Row byte overflow")?;
                if row_bytes > policy_dataset::MAX_SHARD_BYTES {
                    return Err("Semantic teacher game row file exceeds byte budget".into());
                }
                counts.add(&TeacherCounts {
                    setup_callbacks: 0,
                    playing_callbacks: 1,
                    rows: 1,
                    candidates: s.candidates.len(),
                })?;
                hash.update(&line);
                row(s, &line)?;
            }
            _ => return Err("Unexpected native teacher callback phase".into()),
        }
        state = apply_move(&state, step.chosen.r#move.clone())?;
    }
    if state != terminal || counts.setup_callbacks == 0 || counts.playing_callbacks == 0 {
        return Err("Incomplete teacher callback phases/terminal".into());
    }
    Ok(TeacherGame {
        game_id,
        family_id: family_id.clone(),
        seed: record.header.seed,
        split: split_for_family(&family_id)?,
        players: record.header.names.len(),
        source_file: format!("sources/source-{index:06}.json"),
        source_bytes: bytes.len() as u64,
        source_sha256: digest(bytes),
        row_file: format!("rows-{index:06}.jsonl"),
        row_bytes,
        row_sha256: format!("{:x}", hash.finalize()),
        counts,
    })
}
fn source(path: &Path, cap: u64) -> Result<(Vec<u8>, GameReplay), String> {
    let bytes = read(path, cap.min(policy_dataset::MAX_SOURCE_BYTES))?;
    let replay = policy_dataset::decode_strict_native_replay(&bytes)?;
    Ok((bytes, replay))
}
fn manifest(games: Vec<TeacherGame>) -> Result<SemanticPlayingTeacherManifest, String> {
    let mut counts = TeacherCounts::default();
    let mut stored_bytes = 0u64;
    let mut families = BTreeMap::new();
    let mut strata: BTreeMap<usize, (usize, TeacherCounts)> = BTreeMap::new();
    for game in &games {
        counts.add(&game.counts)?;
        stored_bytes = stored_bytes
            .checked_add(game.source_bytes)
            .and_then(|n| n.checked_add(game.row_bytes))
            .ok_or("Stored byte overflow")?;
        if stored_bytes > MAX_STORED_BYTES {
            return Err("Teacher aggregate stored byte budget exceeded".into());
        }
        families.insert(game.family_id.clone(), game.split);
        let s = strata.entry(game.players).or_default();
        s.0 += 1;
        s.1.add(&game.counts)?;
    }
    let mut m = SemanticPlayingTeacherManifest {
        schema: DATASET_SCHEMA.into(),
        source_kind: SOURCE_KIND.into(),
        task: TASK.into(),
        input_contract: INPUT_CONTRACT.into(),
        feature_schema: SEMANTIC_PLAYING_FEATURE_SCHEMA,
        feature_count: FEATURE_COUNT,
        rules_version: replay::RULES_VERSION,
        rules_baseline: replay::RULES_BASELINE.into(),
        catalog_hash: replay::catalog_hash(),
        move_schema: MOVE_SCHEMA,
        observation_schema: OBSERVATION_SCHEMA,
        options: GameOptions::default(),
        teacher_policy_id: teacher_id()?,
        value_supervised: false,
        fingerprint: String::new(),
        stored_bytes,
        counts,
        families: families
            .into_iter()
            .map(|(family_id, split)| TeacherFamily { family_id, split })
            .collect(),
        games,
        strata: strata
            .into_iter()
            .map(|(players, (games, counts))| TeacherStratum {
                players,
                games,
                counts,
            })
            .collect(),
    };
    m.fingerprint = fingerprint(&m)?;
    Ok(m)
}
fn validate_manifest(m: &SemanticPlayingTeacherManifest) -> Result<(), String> {
    if m.games.is_empty() || m.games.len() > MAX_GAMES {
        return Err("Invalid native semantic teacher game count".into());
    }
    let mut game_ids = BTreeSet::new();
    let mut family_players = BTreeSet::new();
    for (i, g) in m.games.iter().enumerate() {
        if !digest_id(&g.game_id)
            || !game_ids.insert(&g.game_id)
            || g.family_id != seed_family_id(g.seed)
            || !family_players.insert((&g.family_id, g.players))
            || g.split != split_for_family(&g.family_id)?
            || !(3..=4).contains(&g.players)
            || g.source_file != format!("sources/source-{i:06}.json")
            || g.row_file != format!("rows-{i:06}.jsonl")
            || !digest_id(&g.source_sha256)
            || !digest_id(&g.row_sha256)
            || g.source_bytes == 0
            || g.source_bytes > policy_dataset::MAX_SOURCE_BYTES
            || g.row_bytes == 0
            || g.row_bytes > policy_dataset::MAX_SHARD_BYTES
            || g.counts.setup_callbacks == 0
            || g.counts.playing_callbacks == 0
            || g.counts
                .setup_callbacks
                .checked_add(g.counts.playing_callbacks)
                .is_none_or(|n| n > replay::MAX_DECISIONS)
            || g.counts.rows != g.counts.playing_callbacks
            || g.counts.candidates < g.counts.rows
        {
            return Err(
                "Invalid/duplicate teacher game, family, split or source/row counts".into(),
            );
        }
    }
    // Rebuild all fixed contracts, aggregate counts, per-player strata and family roles.
    if *m != manifest(m.games.clone())? {
        return Err("Unsupported/corrupt native semantic teacher manifest".into());
    }
    let bytes = serde_json::to_vec(m).map_err(|e| e.to_string())?.len() as u64;
    if bytes > policy_dataset::MAX_MANIFEST_BYTES
        || m.stored_bytes
            .checked_add(bytes)
            .is_none_or(|n| n > MAX_STORED_BYTES)
    {
        return Err("Teacher manifest/aggregate byte budget exceeded".into());
    }
    Ok(())
}
fn namespace(
    directory: &Path,
    m: &SemanticPlayingTeacherManifest,
    complete: bool,
) -> Result<(), String> {
    local(directory)?;
    let mut expected: BTreeSet<String> = m.games.iter().map(|g| g.row_file.clone()).collect();
    expected.insert("sources".into());
    if complete {
        expected.insert("manifest.json".into());
    }
    let entries = fs::read_dir(directory)
        .map_err(|e| e.to_string())?
        .map(|e| {
            e.map_err(|e| e.to_string()).and_then(|e| {
                e.file_name()
                    .into_string()
                    .map_err(|_| "Non-UTF8 teacher entry".to_string())
            })
        })
        .collect::<Result<BTreeSet<_>, _>>()?;
    if entries != expected {
        return Err("Teacher dataset namespace differs from manifest".into());
    }
    local(&directory.join("sources"))?;
    let expected: BTreeSet<_> = m
        .games
        .iter()
        .enumerate()
        .map(|(i, _)| format!("source-{i:06}.json"))
        .collect();
    let entries = fs::read_dir(directory.join("sources"))
        .map_err(|e| e.to_string())?
        .map(|e| {
            e.map_err(|e| e.to_string()).and_then(|e| {
                e.file_name()
                    .into_string()
                    .map_err(|_| "Non-UTF8 teacher source".to_string())
            })
        })
        .collect::<Result<BTreeSet<_>, _>>()?;
    if entries != expected {
        return Err("Teacher source namespace differs from manifest".into());
    }
    Ok(())
}
fn audit(
    directory: &Path,
    m: &SemanticPlayingTeacherManifest,
    cache: bool,
) -> Result<Vec<ValidatedSemanticPlayingTeacherSample>, String> {
    let mut samples = if cache {
        Vec::with_capacity(m.counts.rows)
    } else {
        Vec::new()
    };
    let mut allocation = 0usize;
    for (i, g) in m.games.iter().enumerate() {
        let (bytes, record) = source(&directory.join(&g.source_file), g.source_bytes)?;
        let rows = read(&directory.join(&g.row_file), g.row_bytes)?;
        if (bytes.len() as u64, digest(&bytes)) != (g.source_bytes, g.source_sha256.clone())
            || (rows.len() as u64, digest(&rows)) != (g.row_bytes, g.row_sha256.clone())
        {
            return Err("Teacher raw source/row pin mismatch".into());
        }
        let mut lines = rows.split_inclusive(|b| *b == b'\n');
        let derived = derive_game(i, &bytes, &record, |sample, expected| {
            let line = lines.next().ok_or("Missing Playing teacher row")?;
            match_row(line, &sample, expected)?;
            add_allocation(&mut allocation, &sample)?;
            if cache {
                samples.push(sample);
            }
            Ok(())
        })?;
        if derived != *g || lines.next().is_some() {
            return Err("Teacher source metadata/excess row mismatch".into());
        }
    }
    Ok(samples)
}
fn match_row(
    line: &[u8],
    sample: &ValidatedSemanticPlayingTeacherSample,
    expected: &[u8],
) -> Result<(), String> {
    if line.len() as u64 > policy_dataset::MAX_SAMPLE_BYTES || line.last() != Some(&b'\n') {
        return Err("Unterminated/oversized teacher row".into());
    }
    let wire: RowWire = json(line, policy_dataset::MAX_SAMPLE_BYTES)?;
    if wire != sample.wire()? || line != expected {
        return Err("Stored teacher row differs from original callback re-derivation".into());
    }
    Ok(())
}
fn add_allocation(
    allocation: &mut usize,
    sample: &ValidatedSemanticPlayingTeacherSample,
) -> Result<(), String> {
    *allocation = allocation
        .checked_add(sample.allocation_bytes()?)
        .ok_or("Teacher cache overflow")?;
    if *allocation > MAX_CACHE_BYTES {
        return Err("Native semantic teacher encoded-cache budget exceeded".into());
    }
    Ok(())
}
/// Validate every original before creating output. Sources and rows are immutable
/// create-new files; manifest.json is the last, audited completion marker.
pub fn export_semantic_playing_teacher_files(
    paths: &[PathBuf],
    output: &Path,
) -> Result<SemanticPlayingTeacherManifest, String> {
    local(output)?;
    if fs::symlink_metadata(output).is_ok() || paths.is_empty() || paths.len() > MAX_GAMES {
        return Err("Invalid teacher destination/source count".into());
    }
    let mut seen = BTreeSet::new();
    let mut games = Vec::new();
    let mut stored = 0u64;
    let mut allocation = 0usize;
    for (i, path) in paths.iter().enumerate() {
        local(path)?;
        if !seen.insert(fs::canonicalize(path).map_err(|e| e.to_string())?) {
            return Err("Duplicate teacher source path".into());
        }
        let (bytes, record) = source(path, MAX_STORED_BYTES - stored)?;
        let game = derive_game(i, &bytes, &record, |s, _| {
            add_allocation(&mut allocation, &s)
        })?;
        stored = stored
            .checked_add(game.source_bytes)
            .and_then(|n| n.checked_add(game.row_bytes))
            .filter(|n| *n <= MAX_STORED_BYTES)
            .ok_or("Teacher aggregate stored byte budget exceeded")?;
        games.push(game);
    }
    let m = manifest(games)?;
    validate_manifest(&m)?;
    fs::create_dir(output).map_err(|e| e.to_string())?;
    fs::create_dir(output.join("sources")).map_err(|e| e.to_string())?;
    for (i, path) in paths.iter().enumerate() {
        let g = &m.games[i];
        let (bytes, record) = source(path, g.source_bytes)?;
        if (bytes.len() as u64, digest(&bytes)) != (g.source_bytes, g.source_sha256.clone()) {
            return Err("Teacher original changed between export passes".into());
        }
        policy_dataset::publish_bytes(
            &output.join("sources"),
            &format!("source-{i:06}.json"),
            &bytes,
        )?;
        let mut file = File::options()
            .write(true)
            .create_new(true)
            .open(output.join(&g.row_file))
            .map_err(|e| e.to_string())?;
        let derived = derive_game(i, &bytes, &record, |_, line| {
            file.write_all(line).map_err(|e| e.to_string())
        })?;
        file.sync_all().map_err(|e| e.to_string())?;
        if derived != *g {
            return Err("Teacher source changed between derivation passes".into());
        }
    }
    namespace(output, &m, false)?;
    audit(output, &m, false)?;
    for (p, g) in paths.iter().zip(&m.games) {
        if raw_pin(p, g.source_bytes)? != (g.source_bytes, g.source_sha256.clone()) {
            return Err("Teacher original changed during export".into());
        }
    }
    let raw = serde_json::to_vec(&m).map_err(|e| e.to_string())?;
    policy_dataset::publish_bytes(output, "manifest.json", &raw)?;
    Ok(m)
}
/// A distinct schema-4 handle with no conversion to Human or legacy native data.
pub struct ValidatedSemanticPlayingTeacherDataset {
    directory: PathBuf,
    manifest: SemanticPlayingTeacherManifest,
    manifest_pin: (u64, String),
    samples: Vec<ValidatedSemanticPlayingTeacherSample>,
}
impl ValidatedSemanticPlayingTeacherDataset {
    pub fn manifest(&self) -> &SemanticPlayingTeacherManifest {
        &self.manifest
    }
    /// Raw custody recheck; it does not reclassify split, source kind or quality.
    pub fn recheck(&self) -> Result<(), String> {
        namespace(&self.directory, &self.manifest, true)?;
        if raw_pin(
            &self.directory.join("manifest.json"),
            policy_dataset::MAX_MANIFEST_BYTES,
        )? != self.manifest_pin
        {
            return Err("Teacher manifest changed after load".into());
        }
        for g in &self.manifest.games {
            if raw_pin(&self.directory.join(&g.source_file), g.source_bytes)?
                != (g.source_bytes, g.source_sha256.clone())
                || raw_pin(&self.directory.join(&g.row_file), g.row_bytes)?
                    != (g.row_bytes, g.row_sha256.clone())
            {
                return Err("Teacher source/row changed after load".into());
            }
        }
        Ok(())
    }
    /// Consume through EOF for both pre/post raw custody checks. Training callers
    /// must reject the whole operation on any error, including the final item.
    pub fn iter(&self) -> SemanticPlayingTeacherIter<'_> {
        self.iterator(None)
    }
    pub fn iter_split(&self, split: DatasetSplit) -> SemanticPlayingTeacherIter<'_> {
        self.iterator(Some(split))
    }
    fn iterator(&self, split: Option<DatasetSplit>) -> SemanticPlayingTeacherIter<'_> {
        SemanticPlayingTeacherIter {
            dataset: self,
            split,
            index: 0,
            started: false,
            finished: false,
        }
    }
}
pub struct SemanticPlayingTeacherIter<'a> {
    dataset: &'a ValidatedSemanticPlayingTeacherDataset,
    split: Option<DatasetSplit>,
    index: usize,
    started: bool,
    finished: bool,
}
impl<'a> Iterator for SemanticPlayingTeacherIter<'a> {
    type Item = Result<&'a ValidatedSemanticPlayingTeacherSample, String>;
    fn next(&mut self) -> Option<Self::Item> {
        if self.finished {
            return None;
        }
        if !self.started {
            self.started = true;
            if let Err(e) = self.dataset.recheck() {
                self.finished = true;
                return Some(Err(e));
            }
        }
        while let Some(sample) = self.dataset.samples.get(self.index) {
            self.index += 1;
            // Family roles are fixed across seats, player counts and every source.
            if self.split.is_none_or(|s| {
                self.dataset
                    .manifest
                    .families
                    .iter()
                    .any(|f| f.family_id == sample.family_id && f.split == s)
            }) {
                return Some(Ok(sample));
            }
        }
        self.finished = true;
        self.dataset.recheck().err().map(Err)
    }
}
pub fn load_semantic_playing_teacher_dataset(
    directory: &Path,
) -> Result<ValidatedSemanticPlayingTeacherDataset, String> {
    let bytes = read(
        &directory.join("manifest.json"),
        policy_dataset::MAX_MANIFEST_BYTES,
    )?;
    let m: SemanticPlayingTeacherManifest = json(&bytes, policy_dataset::MAX_MANIFEST_BYTES)?;
    validate_manifest(&m)?;
    namespace(directory, &m, true)?;
    let samples = audit(directory, &m, true)?;
    let dataset = ValidatedSemanticPlayingTeacherDataset {
        directory: directory.to_path_buf(),
        manifest: m,
        manifest_pin: (bytes.len() as u64, digest(&bytes)),
        samples,
    };
    dataset.recheck()?;
    Ok(dataset)
}

#[cfg(test)]
pub(crate) fn fixture(parent: &Path) {
    use crate::policy::HeuristicWeights;
    // Short real Core Setup prefixes only; no complete game generation in CI.
    for players in [3, 4] {
        let mut state = create_game_with_options(
            (0..players).map(|i| format!("P{i}")).collect(),
            42,
            GameOptions::default(),
        )
        .unwrap();
        let mut index = 0;
        while state.phase == Phase::Setup {
            let o = observe(&state, state.current_player).unwrap();
            let d = crate::setup_policy::choose_move_with_weights(&o, &HeuristicWeights::default())
                .unwrap();
            let chosen = o
                .legal_actions
                .iter()
                .find(|a| a.r#move == d.r#move)
                .unwrap();
            assert!(sample(&o, chosen, index, &"a".repeat(64), &seed_family_id(42)).is_err());
            state = apply_move(&state, d.r#move).unwrap();
            index += 1;
            assert!(index <= 64);
        }
        let o = observe(&state, state.current_player).unwrap();
        let d = crate::arena::resource_cycle::choose_move(
            &o,
            &crate::arena::resource_cycle::ResourceCycleConfig::v5(),
        )
        .unwrap();
        let chosen = o
            .legal_actions
            .iter()
            .find(|a| a.r#move == d.r#move)
            .unwrap();
        let row = sample(&o, chosen, index, &"a".repeat(64), &seed_family_id(42)).unwrap();
        assert_eq!(row.source_callback_index(), index);
        assert_eq!(row.actor(), o.actor);
        assert_eq!(&o.legal_actions[row.chosen()], chosen);
        let oracle = FeatureEncoder::new_semantic_playing(&o).unwrap();
        for (i, c) in row.candidates().iter().enumerate() {
            let expected = oracle.encode_legal_tagged(i).unwrap();
            assert!(
                c.values_for_schema(SEMANTIC_PLAYING_FEATURE_SCHEMA)
                    .unwrap()
                    .iter()
                    .zip(
                        expected
                            .values_for_schema(SEMANTIC_PLAYING_FEATURE_SCHEMA)
                            .unwrap()
                    )
                    .all(|(a, b)| a.to_bits() == b.to_bits())
            );
        }
        let mut raw = serde_json::to_vec(&row.wire().unwrap()).unwrap();
        raw.push(b'\n');
        match_row(&raw, &row, &raw).unwrap();
        assert_eq!(
            json::<RowWire>(&raw, policy_dataset::MAX_SAMPLE_BYTES).unwrap(),
            row.wire().unwrap()
        );
        let mut bad: serde_json::Value = serde_json::from_slice(&raw).unwrap();
        bad["sourceCallbackIndex"] = serde_json::json!(index + 1);
        let mut changed = serde_json::to_vec(&bad).unwrap();
        changed.push(b'\n');
        assert!(match_row(&changed, &row, &raw).is_err());
        bad["futureScore"] = serde_json::json!(80);
        changed = serde_json::to_vec(&bad).unwrap();
        changed.push(b'\n');
        assert!(match_row(&changed, &row, &raw).is_err());
        let duplicate = raw
            .strip_suffix(b"}\n")
            .unwrap()
            .iter()
            .copied()
            .chain(b",\"actor\":0}\n".iter().copied())
            .collect::<Vec<_>>();
        assert!(match_row(&duplicate, &row, &raw).is_err());
        let mut bad: serde_json::Value = serde_json::from_slice(&raw).unwrap();
        let value = bad["candidates"][0]["values"][0].as_f64().unwrap();
        bad["candidates"][0]["values"][0] = serde_json::json!(value + 0.25);
        changed = serde_json::to_vec(&bad).unwrap();
        changed.push(b'\n');
        assert!(match_row(&changed, &row, &raw).is_err());
        assert!(
            row.candidates()[0]
                .values_for_schema(crate::features::PUBLIC_FEATURE_SCHEMA)
                .is_err()
        );
        assert!(row.allocation_bytes().unwrap() < MAX_CACHE_BYTES);
        let mut allocation = MAX_CACHE_BYTES - row.allocation_bytes().unwrap() + 1;
        assert!(add_allocation(&mut allocation, &row).is_err());
    }
    let record = GameReplay {
        header: replay::ReplayHeader {
            replay_schema: replay::REPLAY_SCHEMA,
            rules_version: replay::RULES_VERSION,
            rules_baseline: replay::RULES_BASELINE.into(),
            catalog_hash: replay::catalog_hash(),
            move_schema: MOVE_SCHEMA,
            observation_schema: OBSERVATION_SCHEMA,
            source: replay::corn_first_v5_teacher_provenance(),
            names: vec!["A".into(), "B".into(), "C".into()],
            seed: 42,
            options: GameOptions::default(),
        },
        steps: Vec::new(),
        final_scores: Vec::new(),
        final_state: String::new(),
        verified_complete: false,
    };
    let source = parent.join("partial-native4.json");
    fs::write(&source, serde_json::to_vec(&record).unwrap()).unwrap();
    let output = parent.join("native4-dataset");
    assert_eq!(
        export_semantic_playing_teacher_files(&[source], &output).unwrap_err(),
        "Partial logs cannot be verified as complete games"
    );
    assert!(!output.exists());
    // These are untrusted declarations only; the typed loader still requires
    // full original replay verification. They exercise the production contract.
    let game = TeacherGame {
        game_id: "a".repeat(64),
        family_id: seed_family_id(42),
        seed: 42,
        split: split_for_family(&seed_family_id(42)).unwrap(),
        players: 3,
        source_file: "sources/source-000000.json".into(),
        source_bytes: 1,
        source_sha256: "b".repeat(64),
        row_file: "rows-000000.jsonl".into(),
        row_bytes: 1,
        row_sha256: "c".repeat(64),
        counts: TeacherCounts {
            setup_callbacks: 6,
            playing_callbacks: 1,
            rows: 1,
            candidates: 1,
        },
    };
    let m = manifest(vec![game.clone()]).unwrap();
    validate_manifest(&m).unwrap();
    for key in [
        "schema",
        "sourceKind",
        "task",
        "inputContract",
        "featureSchema",
    ] {
        let mut bad = serde_json::to_value(&m).unwrap();
        bad[key] = if key == "featureSchema" {
            serde_json::json!(2)
        } else {
            serde_json::json!("human")
        };
        let mut bad = json::<SemanticPlayingTeacherManifest>(
            &serde_json::to_vec(&bad).unwrap(),
            policy_dataset::MAX_MANIFEST_BYTES,
        )
        .unwrap();
        bad.fingerprint = fingerprint(&bad).unwrap();
        assert_eq!(
            validate_manifest(&bad).unwrap_err(),
            "Unsupported/corrupt native semantic teacher manifest"
        );
    }
    let mut bad = m.clone();
    bad.families[0].split = match bad.families[0].split {
        DatasetSplit::Train => DatasetSplit::Test,
        _ => DatasetSplit::Train,
    };
    bad.fingerprint = fingerprint(&bad).unwrap();
    assert!(validate_manifest(&bad).is_err());
    let mut big = game.clone();
    big.source_bytes = policy_dataset::MAX_SOURCE_BYTES;
    assert!(manifest(vec![big; 4]).is_err());
    let pretend = parent.join("declaration-only");
    fs::create_dir(&pretend).unwrap();
    fs::write(
        pretend.join("manifest.json"),
        serde_json::to_vec(&m).unwrap(),
    )
    .unwrap();
    assert!(load_semantic_playing_teacher_dataset(&pretend).is_err());
    assert!(policy_dataset::load_policy_dataset(&pretend).is_err());
    assert!(
        policy_dataset::load_resource_cycle_v5_semantic_setup_teacher_dataset(&pretend).is_err()
    );
    let mut counts = TeacherCounts {
        setup_callbacks: 0,
        playing_callbacks: MAX_ROWS,
        rows: MAX_ROWS,
        candidates: MAX_ROWS,
    };
    assert!(
        counts
            .add(&TeacherCounts {
                setup_callbacks: 0,
                playing_callbacks: 1,
                rows: 1,
                candidates: 1
            })
            .is_err()
    );
    assert!(
        json::<SemanticPlayingTeacherManifest>(
            b"{\"schema\":\"x\",\"schema\":\"y\"}",
            policy_dataset::MAX_MANIFEST_BYTES
        )
        .is_err()
    );
}
