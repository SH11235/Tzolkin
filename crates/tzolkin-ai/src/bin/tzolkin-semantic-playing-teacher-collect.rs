//! Finite all-seat CornFirst/V5 originals and verified semantic4 dataset export.
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeSet,
    fs::{self, File},
    io::{self, Write},
    path::{Component, Path, PathBuf},
};
use tzolkin_ai::{
    arena::resource_cycle::{self, ResourceCycleConfig},
    conditional_human_rl_update::{ConditionalNativeExclusions, MAX_EXTERNAL_BYTES},
    dataset::{DatasetSplit, MAX_REPLAY_BYTES, seed_family_id, split_for_family},
    human_initial_wealth::ContentPin,
    policy::HeuristicWeights,
    policy_training::validate_new_output_directory,
    public_rl_paired_session::read_bounded,
    replay::{self, ReplaySource},
    semantic_playing_teacher_dataset::{
        MAX_GAMES, MAX_STORED_BYTES, export_semantic_playing_teacher_files,
    },
    setup_policy,
};
use tzolkin_core::{GameOptions, Phase};

const SCHEMA: &str = "tzolkin-corn-first-v5-teacher-collect-v1";
const MAX_CONTROL: usize = 1024 * 1024;
const MAX_REPORT: u64 = 64 * 1024;
// Originals plus the exporter's <=256MiB dataset plus report fit <=512MiB.
const MAX_ORIGINALS: u64 = MAX_STORED_BYTES - MAX_REPORT;
const HELP: &str = "tzolkin-semantic-playing-teacher-collect CONTROL.json CONTROL_SHA256 NEW_OUTPUT_DIR\nAll paths must be absolute local paths. Closed control: schema=tzolkin-corn-first-v5-teacher-collect-v1, globalExclusions={pin:{path,bytes,sha256},count,canonicalDigest}, games=[{players,seed,role}]. Supply 1..64 distinct prospective base 3/4p families with their natural train/validation role; Test is refused. The pinned exclusions use tzolkin-conditional-native-exclusions-v1 and must declare the complete prior union. Permanent allocation and completeness are the caller's responsibility, not authenticated by this command.\nUses fixed all-seat CornFirst/default-weight Setup and ResourceCycle V5 Playing, then independent replay/teacher verification and semantic4 export. Control <=1MiB, exclusions <=32MiB, each original <=64MiB, originals+dataset+report <=512MiB; dataset/cache/row/candidate caps remain unchanged. Use an external wall/process watchdog. There is no retry or seed selection. On refusal retained originals and any incomplete export remain; only exit0 with report.json denotes whole-command completion. No Human origin, model, training, fresh Test, strength or adoption authority is granted.";

#[derive(Clone, Deserialize, Serialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct Game {
    players: usize,
    seed: u32,
    role: DatasetSplit,
}
#[derive(Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct External {
    pin: ContentPin,
    count: usize,
    canonical_digest: String,
}
#[derive(Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct Control {
    schema: String,
    global_exclusions: External,
    games: Vec<Game>,
}
impl Control {
    fn validate(&self, excluded: &ConditionalNativeExclusions) -> Result<(), String> {
        if self.schema != SCHEMA || !(1..=MAX_GAMES).contains(&self.games.len()) {
            return Err("Collector schema or game count refused".into());
        }
        let mut families = BTreeSet::new();
        for game in &self.games {
            let family = seed_family_id(game.seed);
            if !(3..=4).contains(&game.players)
                || game.role == DatasetSplit::Test
                || split_for_family(&family)? != game.role
                || excluded.contains_family(&family)
                || !families.insert(family)
            {
                return Err("Collector players/role, duplicate or prior family refused".into());
            }
        }
        Ok(())
    }
}
fn sha(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}
fn digest(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}
// Check the native Windows prefix before any filesystem access; lexical paths
// are retained for rereads, while canonical paths are only used for containment.
fn local(path: &Path) -> Result<(), String> {
    let text = path.to_str().ok_or("Collector path must be UTF-8")?;
    if !path.is_absolute()
        || text.starts_with('\\')
        || text.starts_with("//")
        || text.contains('\0')
        || text.contains("://")
        || path.components().any(|part| {
            matches!(part, Component::ParentDir)
                || matches!(part, Component::Prefix(p) if !matches!(p.kind(), std::path::Prefix::Disk(_)))
        })
    {
        return Err("Collector requires absolute local paths without parent traversal".into());
    }
    let mut ancestor = PathBuf::new();
    for part in path.components() {
        ancestor.push(part);
        if matches!(part, Component::Prefix(_)) {
            continue;
        }
        match fs::symlink_metadata(&ancestor) {
            Ok(m) => {
                if m.is_symlink() {
                    return Err("Collector link hierarchy refused".into());
                }
                #[cfg(windows)]
                {
                    use std::os::windows::fs::MetadataExt;
                    if m.file_attributes() & 0x400 != 0 {
                        return Err("Collector reparse hierarchy refused".into());
                    }
                }
            }
            Err(e) if e.kind() == io::ErrorKind::NotFound && ancestor == path => {}
            Err(e) => return Err(e.to_string()),
        }
    }
    Ok(())
}
fn pinned(pin: &ContentPin, cap: usize) -> Result<Vec<u8>, String> {
    if pin.bytes == 0 || pin.bytes > cap as u64 || !digest(&pin.sha256) {
        return Err("Collector input pin byte/hash bound".into());
    }
    let path = Path::new(&pin.path);
    local(path)?;
    let bytes = read_bounded(path, cap)?;
    if bytes.len() as u64 != pin.bytes || sha(&bytes) != pin.sha256 {
        return Err("Collector input bytes differ from pin".into());
    }
    Ok(bytes)
}
fn inputs_post(control: &ContentPin, external: &ContentPin) -> Result<(), String> {
    let checks = [
        pinned(control, MAX_CONTROL),
        pinned(external, MAX_EXTERNAL_BYTES),
    ];
    let errors: Vec<_> = checks.into_iter().filter_map(Result::err).collect();
    if errors.is_empty() {
        Ok(())
    } else {
        Err(format!("Collector input POST: {}", errors.join("; ")))
    }
}
fn with_post<T>(result: Result<T, String>, post: Result<(), String>) -> Result<T, String> {
    match (result, post) {
        (Ok(value), Ok(())) => Ok(value),
        (Err(error), Ok(())) | (Ok(_), Err(error)) => Err(error),
        (Err(error), Err(post)) => Err(format!("{error}; {post}")),
    }
}
fn destination(output: &Path, inputs: &[&Path]) -> Result<PathBuf, String> {
    local(output)?;
    validate_new_output_directory(output)?;
    let destination = output
        .parent()
        .ok_or("Collector output parent required")?
        .canonicalize()
        .map_err(|e| e.to_string())?
        .join(output.file_name().ok_or("Collector output name required")?);
    for input in inputs {
        let input = input.canonicalize().map_err(|e| e.to_string())?;
        if input.starts_with(&destination) || destination.starts_with(&input) {
            return Err("Collector output overlaps an immutable input".into());
        }
    }
    Ok(destination)
}
struct CountBytes {
    bytes: u64,
    cap: u64,
}
impl Write for CountBytes {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        self.bytes = self
            .bytes
            .checked_add(bytes.len() as u64)
            .filter(|n| *n <= self.cap)
            .ok_or_else(|| io::Error::other("Collector original byte budget exceeded"))?;
        Ok(bytes.len())
    }
    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}
#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct Original {
    game: Game,
    family_id: String,
    file: String,
    bytes: u64,
    sha256: String,
}
fn collect(control: &Control, output: &Path) -> Result<serde_json::Value, String> {
    let mut originals = Vec::new();
    let mut paths = Vec::new();
    let mut stored = 0u64;
    fs::create_dir(output).map_err(|e| e.to_string())?;
    fs::create_dir(output.join("originals")).map_err(|e| e.to_string())?;
    let weights = HeuristicWeights::default();
    let config = ResourceCycleConfig::v5();
    for (i, game) in control.games.iter().enumerate() {
        let source = ReplaySource::CornFirstV5Teacher {
            policy_version: replay::CORN_FIRST_V5_TEACHER_POLICY_VERSION.into(),
            setup_policy_version: setup_policy::POLICY_VERSION.into(),
            setup_weights: weights.clone(),
            playing_configuration: config.clone(),
        };
        let (_, _, record) = replay::play_game_using_fast(
            game.players,
            game.seed,
            GameOptions::default(),
            true,
            source,
            |o| match o.phase {
                Phase::Setup => setup_policy::choose_move_with_weights(o, &weights),
                Phase::Playing => resource_cycle::choose_move(o, &config),
                _ => Err("Teacher requires an actual Setup/Playing callback".into()),
            },
        )?;
        let record = record.ok_or("Collector did not produce a complete replay")?;
        let mut count = CountBytes {
            bytes: 0,
            cap: MAX_REPLAY_BYTES.min(MAX_ORIGINALS - stored),
        };
        serde_json::to_writer(&mut count, &record).map_err(|e| e.to_string())?;
        let file = format!("originals/replay-{i:02}.json");
        let path = output.join(&file);
        local(&path)?;
        replay::save_replay_new(&path, &record)?;
        let bytes = read_bounded(&path, MAX_REPLAY_BYTES as usize)?;
        if bytes.len() as u64 != count.bytes {
            return Err("Saved teacher original byte count differs".into());
        }
        stored += count.bytes;
        originals.push(Original {
            game: game.clone(),
            family_id: seed_family_id(game.seed),
            file,
            bytes: count.bytes,
            sha256: sha(&bytes),
        });
        paths.push(path);
    }
    // This performs full Core/phase-choice verification and measures all actual
    // source, row, candidate and cache allocations before creating the dataset.
    // A refusal deliberately leaves every completed original in place.
    let manifest = export_semantic_playing_teacher_files(&paths, &output.join("dataset"))?;
    let manifest_raw = read_bounded(&output.join("dataset/manifest.json"), MAX_CONTROL)?;
    Ok(serde_json::json!({
        "schema":"tzolkin-corn-first-v5-teacher-collection-report-v1",
        "complete":true, "originals":originals, "originalBytes":stored,
        "datasetFingerprint":manifest.fingerprint,
        "datasetManifest":{"bytes":manifest_raw.len(),"sha256":sha(&manifest_raw)},
        "counts":manifest.counts, "strata":manifest.strata,
        "sourceKind":manifest.source_kind,"task":manifest.task,
        "allocationAuthenticated":false,"sourceAuthenticated":false,
        "trainingPerformed":false,"strengthImprovementDeclared":false,
        "adoptionAuthorized":false
    }))
}
fn run(args: &[String]) -> Result<(), String> {
    if args == ["--help"] {
        println!("{HELP}");
        return Ok(());
    }
    if args.len() != 3 {
        return Err(HELP.into());
    }
    let control_path = Path::new(&args[0]);
    local(control_path)?;
    let bytes = read_bounded(control_path, MAX_CONTROL)?;
    let control_pin = ContentPin {
        path: args[0].clone(),
        bytes: bytes.len() as u64,
        sha256: args[1].clone(),
    };
    if !digest(&args[1]) || sha(&bytes) != args[1] {
        return Err("Collector control raw pin differs".into());
    }
    // Every field is a closed typed struct/enum, including duplicate-key refusal.
    let control: Control = serde_json::from_slice(&bytes).map_err(|e| e.to_string())?;
    let output = Path::new(&args[2]);
    let result = (|| {
        pinned(&control.global_exclusions.pin, MAX_EXTERNAL_BYTES)?;
        let external = &control.global_exclusions;
        let excluded = ConditionalNativeExclusions::load(
            Path::new(&external.pin.path),
            &external.pin,
            external.count,
            &external.canonical_digest,
        )?;
        control.validate(&excluded)?;
        destination(output, &[control_path, Path::new(&external.pin.path)])?;
        let mut report = collect(&control, output)?;
        inputs_post(&control_pin, &external.pin)?;
        report["controlSha256"] = serde_json::json!(control_pin.sha256);
        report["globalExclusions"] = serde_json::json!({
            "rawSha256":external.pin.sha256,"count":external.count,
            "canonicalDigest":external.canonical_digest,"completenessAuthenticated":false
        });
        let bytes = serde_json::to_vec(&report).map_err(|e| e.to_string())?;
        if bytes.len() as u64 > MAX_REPORT {
            return Err("Collector report exceeds byte cap".into());
        }
        local(output)?;
        let mut file = File::options()
            .write(true)
            .create_new(true)
            .open(output.join("report.json"))
            .map_err(|e| e.to_string())?;
        file.write_all(&bytes)
            .and_then(|_| file.sync_all())
            .map_err(|e| e.to_string())?;
        Ok(())
    })();
    with_post(
        result,
        inputs_post(&control_pin, &control.global_exclusions.pin),
    )?;
    println!(
        "Completed fixed CornFirst/V5 originals and verified semantic4 dataset; no training or strength qualification"
    );
    Ok(())
}
fn main() {
    if let Err(error) = run(&std::env::args().skip(1).collect::<Vec<_>>()) {
        eprintln!("{error}");
        std::process::exit(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{collections::hash_map::RandomState, hash::BuildHasher};

    #[test]
    fn closed_control_refusals_and_input_post() {
        struct Owned(PathBuf);
        impl Drop for Owned {
            fn drop(&mut self) {
                let _ = fs::remove_file(&self.0);
            }
        }
        let (owned, mut file) = (0..16)
            .find_map(|_| {
                let path = std::env::temp_dir().join(format!(
                    "tzolkin-teacher-collect-{:016x}.json",
                    RandomState::new().hash_one(())
                ));
                match File::options().write(true).create_new(true).open(&path) {
                    Ok(file) => Some((Owned(path), file)),
                    Err(e) if e.kind() == io::ErrorKind::AlreadyExists => None,
                    Err(e) => panic!("Owned test input: {e}"),
                }
            })
            .unwrap();
        let prior = (0..100)
            .find(|s| split_for_family(&seed_family_id(*s)).unwrap() == DatasetSplit::Train)
            .unwrap();
        let family = seed_family_id(prior);
        let raw = serde_json::to_vec(&serde_json::json!({
            "schema":"tzolkin-conditional-native-exclusions-v1","families":[family]
        }))
        .unwrap();
        file.write_all(&raw).unwrap();
        file.sync_all().unwrap();
        drop(file);
        let pin = ContentPin {
            path: owned.0.to_str().unwrap().into(),
            bytes: raw.len() as u64,
            sha256: sha(&raw),
        };
        let canonical = sha(format!("{family}\n").as_bytes());
        let excluded = ConditionalNativeExclusions::load(&owned.0, &pin, 1, &canonical).unwrap();
        let train = (prior + 1..100)
            .find(|s| split_for_family(&seed_family_id(*s)).unwrap() == DatasetSplit::Train)
            .unwrap();
        let value = serde_json::json!({"schema":SCHEMA,
            "globalExclusions":{"pin":pin,"count":1,"canonicalDigest":canonical},
            "games":[{"players":3,"seed":train,"role":"train"}]});
        let mut control: Control = serde_json::from_value(value.clone()).unwrap();
        control.validate(&excluded).unwrap();
        control.games.push(control.games[0].clone());
        control.games[1].players = 4; // Same family across player counts is reused.
        assert!(control.validate(&excluded).is_err());
        control.games.pop();
        for (players, role, seed) in [
            (2, DatasetSplit::Train, train),
            (3, DatasetSplit::Test, train),
            (3, DatasetSplit::Validation, train),
            (3, DatasetSplit::Train, prior),
        ] {
            control.games[0] = Game {
                players,
                seed,
                role,
            };
            assert!(control.validate(&excluded).is_err());
        }
        let validation = (0..100)
            .find(|s| split_for_family(&seed_family_id(*s)).unwrap() == DatasetSplit::Validation)
            .unwrap();
        control.games[0] = Game {
            players: 4,
            seed: validation,
            role: DatasetSplit::Validation,
        };
        control.validate(&excluded).unwrap();
        control.games.clear();
        assert!(control.validate(&excluded).is_err());
        control.games = vec![
            Game {
                players: 3,
                seed: train,
                role: DatasetSplit::Train
            };
            MAX_GAMES + 1
        ];
        assert!(control.validate(&excluded).is_err());
        let duplicate = serde_json::to_string(&value).unwrap().replacen(
            &format!("\"schema\":\"{SCHEMA}\""),
            &format!("\"schema\":\"{SCHEMA}\",\"schema\":\"{SCHEMA}\""),
            1,
        );
        assert!(serde_json::from_str::<Control>(&duplicate).is_err());
        let mut unknown = value;
        unknown["options"] = serde_json::json!({});
        assert!(serde_json::from_value::<Control>(unknown).is_err());
        assert!(inputs_post(&pin, &pin).is_ok());
        fs::write(&owned.0, b"changed").unwrap();
        assert!(
            with_post::<()>(Err("original failure".into()), inputs_post(&pin, &pin))
                .unwrap_err()
                .starts_with("original failure; Collector input POST:")
        );
        assert!(with_post(Ok(()), inputs_post(&pin, &pin)).is_err());
        let mut count = CountBytes { bytes: 0, cap: 3 };
        count.write_all(b"abc").unwrap();
        assert!(count.write_all(b"d").is_err());
        assert!(destination(&owned.0, &[&owned.0]).is_err());
        for path in ["relative", "//server/share", "\\\\?\\C:\\input"] {
            assert!(local(Path::new(path)).is_err());
        }
        #[cfg(windows)]
        for path in ["/\\server/share", "\\/server/share"] {
            assert!(local(Path::new(path)).is_err());
        }
    }
}
