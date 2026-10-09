use serde_json::Value;
use sha2::{Digest, Sha256};
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicU64, Ordering};
use tzolkin_ai::dataset::{self, DatasetSplit};
use tzolkin_ai::features::{FeatureEncoder, PUBLIC_FEATURE_SCHEMA};
use tzolkin_ai::policy::HeuristicWeights;
use tzolkin_ai::policy_dataset::{
    self, PolicyDatasetManifest, export_native_files, load_policy_dataset,
};
use tzolkin_ai::public_model::{LoadedPublicPolicy, PublicPolicyArtifact};
use tzolkin_ai::replay::{self, GameReplay, ReplaySource};
use tzolkin_core::{GameOptions, Phase};
type Mutation = Box<dyn Fn(&mut Value)>;

struct Temp(PathBuf);
impl Temp {
    fn new() -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let path = std::env::temp_dir().join(format!(
            "tzolkin-public-dataset-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir(&path).unwrap();
        Self(path)
    }
    fn source(&self, name: &str, record: &GameReplay) -> PathBuf {
        let path = self.0.join(name);
        fs::write(&path, serde_json::to_vec_pretty(record).unwrap()).unwrap();
        path
    }
    fn dataset(&self) -> PathBuf {
        self.0.join("dataset")
    }
}
impl Drop for Temp {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}
fn basic(players: usize) -> &'static GameReplay {
    static THREE: OnceLock<GameReplay> = OnceLock::new();
    static FOUR: OnceLock<GameReplay> = OnceLock::new();
    (if players == 3 { &THREE } else { &FOUR }).get_or_init(|| {
        replay::play_game_fast(players, 42, GameOptions::default(), true)
            .unwrap()
            .2
            .unwrap()
    })
}
fn hash(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}
fn resign(m: &mut PolicyDatasetManifest) {
    m.fingerprint.clear();
    m.fingerprint = hash(&serde_json::to_vec(m).unwrap());
}
fn write_manifest(dir: &Path, m: &PolicyDatasetManifest) {
    fs::write(dir.join("manifest.json"), serde_json::to_vec(m).unwrap()).unwrap();
}
fn first_lines(dir: &Path, m: &PolicyDatasetManifest) -> Vec<Value> {
    fs::read_to_string(dir.join(&m.shards[0].file))
        .unwrap()
        .lines()
        .map(|line| serde_json::from_str(line).unwrap())
        .collect()
}
fn rewrite_lines(dir: &Path, m: &mut PolicyDatasetManifest, lines: &[Value]) {
    let bytes: Vec<u8> = lines
        .iter()
        .flat_map(|line| {
            let mut bytes = serde_json::to_vec(line).unwrap();
            bytes.push(b'\n');
            bytes
        })
        .collect();
    fs::write(dir.join(&m.shards[0].file), &bytes).unwrap();
    m.shards[0].sha256 = hash(&bytes);
    m.shards[0].bytes = bytes.len() as u64;
    resign(m);
    write_manifest(dir, m);
}
fn exported(temp: &Temp) -> PolicyDatasetManifest {
    export_native_files(&[temp.source("game.json", basic(3))], &temp.dataset()).unwrap()
}

#[test]
fn complete_native_three_and_four_players_roundtrip_all_ordered_rows_and_null_targets() {
    let temp = Temp::new();
    let paths = [
        temp.source("three.json", basic(3)),
        temp.source("four.json", basic(4)),
    ];
    let manifest = export_native_files(&paths, &temp.dataset()).unwrap();
    assert_eq!(manifest.feature_schema, PUBLIC_FEATURE_SCHEMA);
    assert!(!manifest.value_supervised);
    assert_eq!(manifest.games[0].family_id, manifest.games[1].family_id);
    assert_eq!(
        manifest.games[0].split,
        dataset::split_for_family(&dataset::seed_family_id(42)).unwrap()
    );
    assert_eq!(manifest.strata.len(), 2);
    for (game, path) in manifest.games.iter().zip(&paths) {
        assert_eq!(
            fs::read(temp.dataset().join(&game.source_file)).unwrap(),
            fs::read(path).unwrap()
        );
    }
    let loaded = load_policy_dataset(&temp.dataset()).unwrap();
    assert_eq!(loaded.manifest(), &manifest);
    let model = PublicPolicyArtifact::new(7).unwrap();
    let policy = LoadedPublicPolicy::new(&model).unwrap();
    let mut count = 0;
    let mut setup = 0;
    let mut playing = 0;
    let mut ids = std::collections::BTreeSet::new();
    for (record, game) in [basic(3), basic(4)].into_iter().zip(&manifest.games) {
        let samples: Vec<_> = loaded
            .iter()
            .map(Result::unwrap)
            .filter(|s| s.game_id() == game.game_id)
            .collect();
        for (sample, step) in samples.iter().zip(&record.steps) {
            count += 1;
            assert!(ids.insert(sample.sample_id().to_owned()));
            assert_eq!(sample.source_index(), step.index);
            assert_eq!(sample.actor(), step.actor);
            assert_eq!(sample.family_id(), game.family_id);
            assert_eq!(sample.value_target(), None);
            assert_eq!(sample.phase(), &step.observation.phase);
            assert_eq!(
                sample.candidates().len(),
                step.observation.legal_actions.len()
            );
            assert_eq!(step.observation.legal_actions[sample.chosen()], step.chosen);
            let encoder = FeatureEncoder::new_public(&step.observation).unwrap();
            for (index, row) in sample.candidates().iter().enumerate() {
                assert_eq!(row, &encoder.encode_legal_tagged(index).unwrap());
            }
            match sample.phase() {
                Phase::Setup => setup += 1,
                Phase::Playing => playing += 1,
                _ => panic!(),
            }
        }
        let prediction = policy.predict(samples[0].candidates()).unwrap();
        assert_eq!(prediction.value, ());
    }
    assert_eq!(count, manifest.samples);
    assert!(setup > 0 && playing > 0);
    for split in [
        DatasetSplit::Train,
        DatasetSplit::Validation,
        DatasetSplit::Test,
    ] {
        assert_eq!(
            loaded.iter_split(split).map(Result::unwrap).count(),
            if split == manifest.games[0].split {
                count
            } else {
                0
            }
        );
    }
    let first = first_lines(&temp.dataset(), &manifest);
    assert!(
        first
            .iter()
            .all(|line| line.get("valueTarget") == Some(&Value::Null))
    );
    assert!(dataset::load_dataset(&temp.dataset()).is_err());
    let legacy_dir = temp.0.join("legacy");
    dataset::export_dataset(&[basic(3).clone()], &legacy_dir).unwrap();
    assert!(load_policy_dataset(&legacy_dir).is_err());
}

#[test]
fn hashes_do_not_allow_feature_mask_target_or_identity_forgery() {
    let temp = Temp::new();
    let original = exported(&temp);
    let original_lines = first_lines(&temp.dataset(), &original);
    let mutations: Vec<Mutation> = vec![
        Box::new(|line| {
            line.as_object_mut().unwrap().remove("valueTarget");
        }),
        Box::new(|line| line["valueTarget"] = Value::from(0)),
        Box::new(|line| line["candidates"][0]["featureSchema"] = Value::from(1)),
        Box::new(|line| line["candidates"][0]["values"][0] = Value::from(0.99)),
        Box::new(|line| {
            line["candidates"].as_array_mut().unwrap().pop();
        }),
        Box::new(|line| line["candidates"].as_array_mut().unwrap().reverse()),
        Box::new(|line| line["chosen"] = Value::from(999)),
        Box::new(|line| line["sourceIndex"] = Value::from(1)),
        Box::new(|line| line["actor"] = Value::from(999)),
        Box::new(|line| line["sampleId"] = Value::from("0".repeat(64))),
        Box::new(|line| line["extra"] = Value::from(true)),
        Box::new(|line| {
            line["candidates"][0]["values"]
                .as_array_mut()
                .unwrap()
                .pop()
                .map(|_| ())
                .unwrap()
        }),
    ];
    for mutation in mutations {
        let mut lines = original_lines.clone();
        let mut m = original.clone();
        mutation(&mut lines[0]);
        rewrite_lines(&temp.dataset(), &mut m, &lines);
        assert!(load_policy_dataset(&temp.dataset()).is_err());
    }
    // Negative zero is also a different stored feature bit pattern.
    let mut lines = original_lines.clone();
    let zero = lines[0]["candidates"][0]["values"]
        .as_array()
        .unwrap()
        .iter()
        .position(|v| v.as_f64() == Some(0.0))
        .unwrap();
    lines[0]["candidates"][0]["values"][zero] = serde_json::from_str("-0.0").unwrap();
    let mut m = original.clone();
    rewrite_lines(&temp.dataset(), &mut m, &lines);
    assert!(load_policy_dataset(&temp.dataset()).is_err());
    let mut lines = original_lines.clone();
    lines[1] = lines[0].clone();
    let mut m = original;
    rewrite_lines(&temp.dataset(), &mut m, &lines);
    assert!(load_policy_dataset(&temp.dataset()).is_err());
}

#[test]
fn copied_sources_and_shards_are_rechecked_after_load_and_at_eof() {
    let temp = Temp::new();
    let m = exported(&temp);
    let loaded = load_policy_dataset(&temp.dataset()).unwrap();
    let source_path = temp.dataset().join(&m.games[0].source_file);
    let original = fs::read(&source_path).unwrap();
    fs::write(&source_path, b"{}").unwrap();
    assert!(loaded.iter().next().unwrap().is_err());
    fs::write(&source_path, &original).unwrap();
    let mut iter = loaded.iter();
    iter.next().unwrap().unwrap();
    let mut changed = original.clone();
    changed.push(b' ');
    fs::write(&source_path, changed).unwrap();
    assert!(iter.any(|sample| sample.is_err()));
    fs::write(&source_path, &original).unwrap();
    let shard_path = temp.dataset().join(&m.shards[0].file);
    let bytes = fs::read(&shard_path).unwrap();
    let mut iter = loaded.iter();
    iter.next().unwrap().unwrap();
    let mut already_consumed = bytes.clone();
    already_consumed[0] = b' ';
    fs::write(&shard_path, already_consumed).unwrap();
    assert!(iter.any(|sample| sample.is_err()));
    fs::write(&shard_path, &bytes).unwrap();
    let mut corrupt = bytes.clone();
    let end = corrupt.len() - 1;
    corrupt[end] = b' ';
    fs::write(&shard_path, corrupt).unwrap();
    assert!(loaded.iter().any(|s| s.is_err()));
    fs::write(&shard_path, &bytes[..bytes.len() - 1]).unwrap();
    assert!(load_policy_dataset(&temp.dataset()).is_err());
    fs::write(&shard_path, &bytes).unwrap();
    let mut wrong = m.clone();
    wrong.games[0].split = if wrong.games[0].split == DatasetSplit::Train {
        DatasetSplit::Test
    } else {
        DatasetSplit::Train
    };
    resign(&mut wrong);
    write_manifest(&temp.dataset(), &wrong);
    assert!(load_policy_dataset(&temp.dataset()).is_err());
    let mut wrong = m.clone();
    wrong.games[0].source_file = "../game.json".into();
    resign(&mut wrong);
    write_manifest(&temp.dataset(), &wrong);
    assert!(load_policy_dataset(&temp.dataset()).is_err());
    let mut wrong = m;
    wrong.shards[0].first_source_index = 1;
    resign(&mut wrong);
    write_manifest(&temp.dataset(), &wrong);
    assert!(load_policy_dataset(&temp.dataset()).is_err());
}

#[test]
fn strict_required_options_native_source_and_terminal_are_not_claimed_flags() {
    let temp = Temp::new();
    let path = temp.source("game.json", basic(3));
    let original: Value = serde_json::from_slice(&fs::read(&path).unwrap()).unwrap();
    let mutations: Vec<Mutation> = vec![
        Box::new(|v| {
            v["header"]["options"]
                .as_object_mut()
                .unwrap()
                .remove("tribes");
        }),
        Box::new(|v| v["header"]["options"]["unexpected"] = Value::Bool(false)),
        Box::new(|v| v["header"]["options"]["additionalBuildings"] = Value::Bool(true)),
        Box::new(|v| v["header"]["options"]["tribes"] = Value::Bool(true)),
        Box::new(|v| v["header"]["options"]["prophecies"] = Value::Bool(true)),
        Box::new(|v| v["header"]["options"]["quickActions"] = Value::Bool(true)),
        Box::new(|v| v["verifiedComplete"] = Value::Bool(false)),
        Box::new(|v| {
            v["steps"].as_array_mut().unwrap().pop();
        }),
        Box::new(|v| v["steps"][0]["validated"] = Value::Bool(false)),
        Box::new(|v| v["steps"][0]["actor"] = Value::from(3)),
        Box::new(
            |v| v["header"]["source"] = serde_json::json!({"kind":"human","provider":"localFixture","reference":"local","skillRating":null}),
        ),
    ];
    for mutation in mutations {
        let mut v = original.clone();
        mutation(&mut v);
        fs::write(&path, serde_json::to_vec(&v).unwrap()).unwrap();
        assert!(export_native_files(std::slice::from_ref(&path), &temp.dataset()).is_err());
        assert!(!temp.dataset().exists());
    }
    for players in [2, 5] {
        let record = replay::play_game_fast(players, 42, GameOptions::default(), true)
            .unwrap()
            .2
            .unwrap();
        let path = temp.source("unsupported.json", &record);
        assert!(export_native_files(&[path], &temp.dataset()).is_err());
    }
    fs::write(&path, serde_json::to_vec(basic(3)).unwrap()).unwrap();
    assert!(export_native_files(&[path.clone(), path], &temp.dataset()).is_err());
    assert!(!temp.dataset().exists());
}

#[test]
fn weighted_policy_strata_are_content_bound_but_seed_family_is_shared() {
    let temp = Temp::new();
    let mut weights = HeuristicWeights::default();
    weights.worker *= 0.5;
    let source = ReplaySource::SelfPlay {
        policy_version: tzolkin_ai::POLICY_VERSION.into(),
        weights: weights.clone(),
    };
    let weighted = replay::play_game_using_fast(3, 42, GameOptions::default(), true, source, |o| {
        tzolkin_ai::choose_move_with_weights(o, &weights)
    })
    .unwrap()
    .2
    .unwrap();
    let m = export_native_files(
        &[
            temp.source("default.json", basic(3)),
            temp.source("weighted.json", &weighted),
        ],
        &temp.dataset(),
    )
    .unwrap();
    assert_eq!(m.strata.len(), 2);
    assert_ne!(m.games[0].policy_id, m.games[1].policy_id);
    assert_eq!(m.games[0].family_id, m.games[1].family_id);
    assert_eq!(m.games[0].split, m.games[1].split);
}

#[test]
fn new_directory_nooverwrite_bounds_symlinks_and_native_cli_are_enforced() {
    let temp = Temp::new();
    let input = temp.0.join("native");
    fs::create_dir(&input).unwrap();
    let path = input.join("one.json");
    fs::write(&path, serde_json::to_vec(basic(3)).unwrap()).unwrap();
    let binary = env!("CARGO_BIN_EXE_tzolkin-public-ml");
    let result = Command::new(binary)
        .args(["export-native", "--input"])
        .arg(&input)
        .arg("--output")
        .arg(temp.dataset())
        .output()
        .unwrap();
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    let m: PolicyDatasetManifest = serde_json::from_slice(&result.stdout).unwrap();
    assert_eq!(load_policy_dataset(&temp.dataset()).unwrap().manifest(), &m);
    let old = fs::read(temp.dataset().join("manifest.json")).unwrap();
    assert!(export_native_files(std::slice::from_ref(&path), &temp.dataset()).is_err());
    assert_eq!(fs::read(temp.dataset().join("manifest.json")).unwrap(), old);
    assert!(export_native_files(&[], &temp.0.join("empty")).is_err());
    assert!(
        export_native_files(
            &vec![path.clone(); policy_dataset::MAX_FILES + 1],
            &temp.0.join("many")
        )
        .is_err()
    );
    let huge = input.join("huge.json");
    let file = fs::File::create(&huge).unwrap();
    file.set_len(policy_dataset::MAX_SOURCE_BYTES + 1).unwrap();
    assert!(export_native_files(std::slice::from_ref(&huge), &temp.0.join("huge-output")).is_err());
    fs::remove_file(huge).unwrap();
    for flags in [
        vec!["export-native"],
        vec!["export-native", "--unknown", "x"],
        vec!["export-native", "--input", "x", "--input", "y"],
        vec!["export-native", "--kernel", "scalar"],
        vec!["train"],
    ] {
        let result = Command::new(binary).args(flags).output().unwrap();
        assert!(!result.status.success());
        assert!(result.stdout.is_empty());
    }
    fs::write(input.join("summary.json"), b"{}").unwrap();
    let result = Command::new(binary)
        .arg("export-native")
        .arg("--input")
        .arg(&input)
        .arg("--output")
        .arg(temp.0.join("summary-output"))
        .output()
        .unwrap();
    assert!(!result.status.success());
    assert!(!temp.0.join("summary-output").exists());
    fs::write(input.join("note.txt"), b"note").unwrap();
    assert!(policy_dataset::native_source_files(&input).is_err());
    #[cfg(unix)]
    {
        let link = temp.0.join("link.json");
        std::os::unix::fs::symlink(&path, &link).unwrap();
        assert!(export_native_files(&[link], &temp.0.join("link-output")).is_err());
    }
    assert!(
        export_native_files(
            &[PathBuf::from("//host/share/source.json")],
            &temp.0.join("network")
        )
        .is_err()
    );
}

#[test]
fn multiple_shards_preserve_source_continuation_and_serialized_bounds() {
    let temp = Temp::new();
    let mut m = exported(&temp);
    assert_eq!(m.shards.len(), 1);
    let lines = first_lines(&temp.dataset(), &m);
    let middle = lines.len() / 2;
    let encode = |rows: &[Value]| -> Vec<u8> {
        rows.iter()
            .flat_map(|line| {
                let mut bytes = serde_json::to_vec(line).unwrap();
                bytes.push(b'\n');
                bytes
            })
            .collect()
    };
    let a = encode(&lines[..middle]);
    let b = encode(&lines[middle..]);
    let mut second = m.shards[0].clone();
    second.file = "shard-000001.jsonl".into();
    second.first_source_index = middle;
    second.samples = lines.len() - middle;
    second.sha256 = hash(&b);
    second.bytes = b.len() as u64;
    m.shards[0].samples = middle;
    m.shards[0].sha256 = hash(&a);
    m.shards[0].bytes = a.len() as u64;
    fs::write(temp.dataset().join(&m.shards[0].file), a).unwrap();
    fs::write(temp.dataset().join(&second.file), b).unwrap();
    m.shards.push(second);
    resign(&mut m);
    write_manifest(&temp.dataset(), &m);
    let loaded = load_policy_dataset(&temp.dataset()).unwrap();
    assert_eq!(loaded.iter().map(Result::unwrap).count(), m.samples);
    // An over-limit JSONL line remains invalid even with fresh metadata hashes.
    let bytes = vec![b' '; policy_dataset::MAX_SAMPLE_BYTES as usize + 1];
    fs::write(temp.dataset().join(&m.shards[0].file), &bytes).unwrap();
    m.shards[0].bytes = bytes.len() as u64;
    m.shards[0].sha256 = hash(&bytes);
    resign(&mut m);
    write_manifest(&temp.dataset(), &m);
    assert!(load_policy_dataset(&temp.dataset()).is_err());
    // Manifest cap is checked before JSON parsing/allocation of game metadata.
    fs::File::create(temp.dataset().join("manifest.json"))
        .unwrap()
        .set_len(policy_dataset::MAX_MANIFEST_BYTES + 1)
        .unwrap();
    assert!(load_policy_dataset(&temp.dataset()).is_err());
}

#[test]
fn native_search_source_preserves_full_provenance_and_rejects_corrupt_traces() {
    use tzolkin_ai::search::{FallbackReason, PreparedSearch, SearchConfig, SearchStatus};
    let temp = Temp::new();
    let config = SearchConfig {
        worlds_per_action: 1,
        min_completed_worlds: 1,
        horizon_days: 1,
        max_rollout_steps: 64,
        max_total_steps: 1024,
        ..SearchConfig::default()
    };
    let policy = PreparedSearch::new(&config).unwrap();
    let mut records = Vec::new();
    let mut paths = Vec::new();
    for players in [3, 4] {
        let record = tzolkin_ai::search_native::play_game(
            players,
            11235,
            GameOptions::default(),
            &[0],
            true,
            true,
            &policy,
        )
        .unwrap()
        .game
        .unwrap()
        .2
        .unwrap();
        assert!(record.steps.iter().any(|step| {
            step.search
                .as_ref()
                .is_some_and(|trace| trace.status == SearchStatus::Searched)
        }));
        assert!(
            record
                .steps
                .iter()
                .any(|step| step.search.as_ref().is_some_and(|trace| trace.status
                    == SearchStatus::Fallback {
                        reason: FallbackReason::Setup
                    }))
        );
        assert!(
            record
                .steps
                .iter()
                .filter(|step| step.actor != 0)
                .all(|step| step.search.is_none())
        );
        paths.push(temp.source(&format!("search-{players}.json"), &record));
        records.push(record);
    }
    let reference = replay::play_game_fast(3, 11235, GameOptions::default(), true)
        .unwrap()
        .2
        .unwrap();
    paths.push(temp.source("reference.json", &reference));
    let manifest = export_native_files(&paths, &temp.dataset()).unwrap();
    assert_eq!(manifest.strata.len(), 3);
    assert_ne!(manifest.games[0].policy_id, manifest.games[2].policy_id);
    assert!(
        manifest
            .games
            .iter()
            .all(|game| game.family_id == dataset::seed_family_id(11235))
    );
    assert!(
        manifest
            .games
            .iter()
            .all(|game| game.split == manifest.games[0].split)
    );
    for (game, path) in manifest.games.iter().zip(&paths) {
        let copy = fs::read(temp.dataset().join(&game.source_file)).unwrap();
        assert_eq!(copy, fs::read(path).unwrap());
        assert_eq!(game.source_sha256, hash(&copy));
    }
    for (game, record) in manifest
        .games
        .iter()
        .zip(records.iter().chain([&reference]))
    {
        let mut bytes = b"tzolkin-native-policy-source-v1\0".to_vec();
        bytes.extend(serde_json::to_vec(&record.header.source).unwrap());
        assert_eq!(game.policy_id, hash(&bytes));
    }
    let loaded = load_policy_dataset(&temp.dataset()).unwrap();
    for sample in loaded.iter() {
        let sample = sample.unwrap();
        assert_eq!(sample.value_target(), None);
        if let Some(index) = manifest.games[..2]
            .iter()
            .position(|game| game.game_id == sample.game_id())
        {
            let step = &records[index].steps[sample.source_index()];
            let encoder = FeatureEncoder::new_public(&step.observation).unwrap();
            for (i, row) in sample.candidates().iter().enumerate() {
                assert_eq!(row, &encoder.encode_legal_tagged(i).unwrap());
            }
        }
    }
    let original = &records[0];
    let playing = original
        .steps
        .iter()
        .position(|step| {
            step.search
                .as_ref()
                .is_some_and(|trace| trace.status == SearchStatus::Searched)
        })
        .unwrap();
    let mut corrupt = original.clone();
    corrupt.steps[playing].search.as_mut().unwrap().status = SearchStatus::Fallback {
        reason: FallbackReason::Setup,
    };
    let bad = temp.source("bad-status.json", &corrupt);
    assert!(export_native_files(&[bad], &temp.0.join("bad-status-output")).is_err());
    assert!(!temp.0.join("bad-status-output").exists());
    let mut corrupt = original.clone();
    corrupt.steps[playing]
        .search
        .as_mut()
        .unwrap()
        .stats
        .atomic_steps = config.max_total_steps + 1;
    let bad = temp.source("bad-budget.json", &corrupt);
    assert!(export_native_files(&[bad], &temp.0.join("bad-budget-output")).is_err());
    assert!(!temp.0.join("bad-budget-output").exists());
    let mut corrupt = original.clone();
    corrupt
        .steps
        .iter_mut()
        .find(|step| step.actor == 0)
        .unwrap()
        .search = None;
    let bad = temp.source("bad-missing.json", &corrupt);
    assert!(export_native_files(&[bad], &temp.0.join("bad-missing-output")).is_err());
    assert!(!temp.0.join("bad-missing-output").exists());
    let mut corrupt = original.clone();
    if let ReplaySource::PolicySelfPlay { policies } = &mut corrupt.header.source
        && let replay::SeatPolicy::Search {
            configuration_key, ..
        } = &mut policies[0]
    {
        *configuration_key = "0".repeat(64);
    }
    let bad = temp.source("bad-config.json", &corrupt);
    assert!(export_native_files(&[bad], &temp.0.join("bad-config-output")).is_err());
    assert!(!temp.0.join("bad-config-output").exists());
}
