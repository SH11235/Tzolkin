use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};
use tzolkin_ai::dataset::{DatasetSplit, seed_family_id, split_for_family};
use tzolkin_ai::policy_dataset;
use tzolkin_ai::public_state_critic::{
    LoadedPublicStateCritic, PublicStateContext, PublicStateCriticArtifact,
};
use tzolkin_ai::replay::{self, GameReplay};
use tzolkin_ai::state_mc_dataset::{
    self, StateMcManifest, export_native_files, load_state_mc_dataset,
};
use tzolkin_core::{GameOptions, Phase};

struct Temp(PathBuf);
impl Temp {
    fn new() -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let path = std::env::temp_dir().join(format!(
            "tzolkin-state-mc-{}-{}-{}",
            std::process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos(),
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
        replay::play_game_fast(players, 17, GameOptions::default(), true)
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
fn resign(m: &mut StateMcManifest) {
    m.fingerprint.clear();
    m.fingerprint = hash(&serde_json::to_vec(m).unwrap());
}
fn save_manifest(dir: &Path, m: &StateMcManifest) {
    fs::write(dir.join("manifest.json"), serde_json::to_vec(m).unwrap()).unwrap();
}
fn lines(dir: &Path, m: &StateMcManifest) -> Vec<Value> {
    assert_eq!(m.shards.len(), 1);
    fs::read_to_string(dir.join(&m.shards[0].file))
        .unwrap()
        .lines()
        .map(|line| serde_json::from_str(line).unwrap())
        .collect()
}
fn rewrite(dir: &Path, m: &mut StateMcManifest, lines: &[Value]) {
    let bytes: Vec<u8> = lines
        .iter()
        .flat_map(|line| {
            let mut b = serde_json::to_vec(line).unwrap();
            b.push(b'\n');
            b
        })
        .collect();
    fs::write(dir.join(&m.shards[0].file), &bytes).unwrap();
    m.shards[0].bytes = bytes.len() as u64;
    m.shards[0].sha256 = hash(&bytes);
    resign(m);
    save_manifest(dir, m);
}
fn exported(temp: &Temp) -> StateMcManifest {
    export_native_files(&[temp.source("game.json", basic(3))], &temp.dataset()).unwrap()
}

#[test]
fn every_native_three_four_actor_context_and_own_decision_mc_target_roundtrips() {
    let temp = Temp::new();
    let paths = [
        temp.source("three.json", basic(3)),
        temp.source("four.json", basic(4)),
    ];
    let manifest = export_native_files(&paths, &temp.dataset()).unwrap();
    let loaded = load_state_mc_dataset(&temp.dataset()).unwrap();
    assert_eq!(loaded.manifest(), &manifest);
    assert_eq!(manifest.feature_schema, 2);
    assert_eq!(manifest.context_count, 384);
    assert_eq!(manifest.games[0].family_id, manifest.games[1].family_id);
    assert_eq!(
        manifest.games[0].split,
        split_for_family(&seed_family_id(17)).unwrap()
    );
    assert_eq!(manifest.strata.len(), 2);
    let samples: Vec<_> = loaded.iter().map(Result::unwrap).collect();
    assert_eq!(samples.len(), manifest.samples);
    let artifact = PublicStateCriticArtifact::new(17).unwrap();
    let critic = LoadedPublicStateCritic::new(&artifact).unwrap();
    let mut offset = 0;
    let mut off_turn_pending = 0;
    let mut consecutive_own = 0;
    for ((record, game), source) in [basic(3), basic(4)]
        .into_iter()
        .zip(&manifest.games)
        .zip(&paths)
    {
        assert_eq!(
            fs::read(temp.dataset().join(&game.source_file)).unwrap(),
            fs::read(source).unwrap()
        );
        let scores = replay::verify_replay(record).unwrap().final_scores;
        let winners = scores.iter().filter(|score| score.rank == 1).count();
        let mut seen_ids = std::collections::BTreeSet::new();
        let mut actors = std::collections::BTreeSet::new();
        for (i, step) in record.steps.iter().enumerate() {
            let sample = &samples[offset + i];
            let next = record
                .steps
                .iter()
                .enumerate()
                .skip(i + 1)
                .find(|(_, s)| s.actor == step.actor)
                .map(|(i, _)| i);
            let rank = scores
                .iter()
                .find(|score| score.player_id == step.actor)
                .unwrap()
                .rank;
            let target = if rank == 1 {
                1.0_f32 / winners as f32
            } else {
                0.0_f32
            };
            assert!(seen_ids.insert(sample.sample_id()));
            actors.insert(sample.actor());
            assert_eq!(sample.actor(), step.actor);
            assert_eq!(sample.source_index(), i);
            assert_eq!(sample.phase(), step.observation.phase);
            assert!(matches!(sample.phase(), Phase::Setup | Phase::Playing));
            assert_eq!(sample.game_id(), game.game_id);
            assert_eq!(sample.family_id(), game.family_id);
            assert_eq!(
                sample.context(),
                &PublicStateContext::from_observation(&step.observation).unwrap()
            );
            assert!(
                critic
                    .estimate(sample.context())
                    .unwrap()
                    .raw_return_estimate
                    .is_finite()
            );
            assert_eq!(sample.winner_count(), winners);
            assert_eq!(sample.actor_terminal_rank(), rank);
            assert_eq!(sample.return_target().to_bits(), target.to_bits());
            assert_eq!(sample.next_own_index(), next);
            assert_eq!(
                sample.next_own_sample_id(),
                next.map(|j| samples[offset + j].sample_id())
            );
            assert_eq!(
                sample.terminal_reward().to_bits(),
                if next.is_none() { target.to_bits() } else { 0 }
            );
            assert_eq!(
                sample.terminal_bootstrap().map(f32::to_bits),
                next.is_none().then_some(0)
            );
            off_turn_pending += usize::from(step.actor != step.turn_player);
            consecutive_own += usize::from(next == Some(i + 1));
        }
        assert_eq!(actors.len(), game.players);
        offset += record.steps.len();
    }
    assert!(
        off_turn_pending > 0,
        "fixture must exercise feeding/pending actor != turn player"
    );
    assert!(
        consecutive_own > 0,
        "fixture must exercise consecutive own decisions"
    );
    let receipt = loaded.iter().finish_checked().unwrap();
    assert_eq!(receipt.samples(), manifest.samples);
    assert_eq!(receipt.games(), 2);
    assert_eq!(receipt.fingerprint(), manifest.fingerprint);
    for split in [
        DatasetSplit::Train,
        DatasetSplit::Validation,
        DatasetSplit::Test,
    ] {
        let receipt = loaded.iter_split(split).finish_checked().unwrap();
        assert_eq!(
            receipt.samples(),
            if split == manifest.games[0].split {
                manifest.samples
            } else {
                0
            }
        );
        assert_eq!(receipt.split(), Some(split));
    }
}

#[test]
fn resigned_numeric_context_identity_actor_and_next_own_forgery_are_not_authority() {
    let temp = Temp::new();
    let manifest = exported(&temp);
    let original = lines(&temp.dataset(), &manifest);
    type Mutation = Box<dyn Fn(&mut Value)>;
    let mutations: Vec<Mutation> = vec![
        Box::new(|v| v["returnTarget"] = json!(0.123)),
        Box::new(|v| v["terminalReward"] = json!(1)),
        Box::new(|v| v["terminalBootstrap"] = json!(0)),
        Box::new(|v| v["winnerCount"] = json!(99)),
        Box::new(|v| v["actorTerminalRank"] = json!(99)),
        Box::new(|v| v["actor"] = json!(99)),
        Box::new(|v| v["sourceIndex"] = json!(99)),
        Box::new(|v| v["contextChecksum"] = json!("0".repeat(64))),
        Box::new(|v| v["sampleId"] = json!("0".repeat(64))),
        Box::new(|v| v["nextOwnIndex"] = json!(99999)),
        Box::new(|v| v["nextOwnSampleId"] = json!("0".repeat(64))),
        Box::new(|v| v["rawValues"] = json!([0, 1])),
        Box::new(|v| {
            v.as_object_mut().unwrap().remove("nextOwnIndex");
        }),
        Box::new(|v| {
            v.as_object_mut().unwrap().remove("terminalBootstrap");
        }),
        Box::new(|v| v["terminalReward"] = serde_json::from_str("-0.0").unwrap()),
    ];
    for mutate in mutations {
        let mut changed = original.clone();
        mutate(&mut changed[0]);
        let mut m = manifest.clone();
        rewrite(&temp.dataset(), &mut m, &changed);
        assert!(load_state_mc_dataset(&temp.dataset()).is_err());
    }
    let loser = original
        .iter()
        .position(|v| v["returnTarget"].as_f64() == Some(0.0))
        .unwrap();
    let mut changed = original.clone();
    changed[loser]["returnTarget"] = serde_json::from_str("-0.0").unwrap();
    let mut m = manifest.clone();
    rewrite(&temp.dataset(), &mut m, &changed);
    assert!(load_state_mc_dataset(&temp.dataset()).is_err());
    let mut changed = original;
    changed[1] = changed[0].clone();
    let mut m = manifest;
    rewrite(&temp.dataset(), &mut m, &changed);
    assert!(load_state_mc_dataset(&temp.dataset()).is_err());
}

#[test]
fn source_shard_manifest_changes_after_load_or_last_yield_prevent_eof_receipt() {
    let temp = Temp::new();
    let m = exported(&temp);
    let loaded = load_state_mc_dataset(&temp.dataset()).unwrap();
    let source = temp.dataset().join(&m.games[0].source_file);
    let source_bytes = fs::read(&source).unwrap();
    fs::write(&source, b"{}").unwrap();
    let mut broken = loaded.iter();
    assert!(broken.next().unwrap().is_err());
    assert!(broken.next().is_none());
    assert!(broken.finish_checked().is_err());
    fs::write(&source, &source_bytes).unwrap();
    for path in [
        source,
        temp.dataset().join(&m.shards[0].file),
        temp.dataset().join("manifest.json"),
    ] {
        let original = fs::read(&path).unwrap();
        let mut iter = loaded.iter();
        for _ in 0..m.samples {
            iter.next().unwrap().unwrap();
        }
        let mut corrupt = original.clone();
        corrupt[0] = b' ';
        fs::write(&path, corrupt).unwrap();
        assert!(iter.finish_checked().is_err());
        fs::write(&path, original).unwrap();
    }
    assert!(loaded.iter().finish_checked().is_ok());
}

#[test]
fn manual_ordered_multi_shard_boundary_and_earlier_shard_recheck() {
    let temp = Temp::new();
    let mut m = exported(&temp);
    let all = lines(&temp.dataset(), &m);
    let midpoint = all.len() / 2;
    let bytes = |rows: &[Value]| {
        rows.iter()
            .flat_map(|v| {
                let mut b = serde_json::to_vec(v).unwrap();
                b.push(b'\n');
                b
            })
            .collect::<Vec<_>>()
    };
    let first = bytes(&all[..midpoint]);
    let second = bytes(&all[midpoint..]);
    let mut next = m.shards[0].clone();
    m.shards[0].samples = midpoint;
    m.shards[0].bytes = first.len() as u64;
    m.shards[0].sha256 = hash(&first);
    next.file = "shard-000001.jsonl".into();
    next.first_source_index = midpoint;
    next.samples = all.len() - midpoint;
    next.bytes = second.len() as u64;
    next.sha256 = hash(&second);
    m.shards.push(next);
    fs::write(temp.dataset().join(&m.shards[0].file), &first).unwrap();
    fs::write(temp.dataset().join(&m.shards[1].file), second).unwrap();
    resign(&mut m);
    save_manifest(&temp.dataset(), &m);
    let loaded = load_state_mc_dataset(&temp.dataset()).unwrap();
    assert_eq!(loaded.iter().finish_checked().unwrap().samples(), m.samples);
    let mut iter = loaded.iter();
    for _ in 0..=midpoint {
        iter.next().unwrap().unwrap();
    }
    let mut changed = first;
    changed[0] = b' ';
    fs::write(temp.dataset().join(&m.shards[0].file), changed).unwrap();
    assert!(iter.finish_checked().is_err());
}

#[test]
fn incomplete_unknown_options_human_and_forged_legal_or_terminal_sources_leave_no_output() {
    let temp = Temp::new();
    let path = temp.source("native.json", basic(3));
    let original: Value = serde_json::from_slice(&fs::read(&path).unwrap()).unwrap();
    type Mutation = Box<dyn Fn(&mut Value)>;
    let mutations: Vec<Mutation> = vec![
        Box::new(|v| {
            v["header"]["options"]
                .as_object_mut()
                .unwrap()
                .remove("tribes");
        }),
        Box::new(|v| v["header"]["options"]["quickActions"] = json!(true)),
        Box::new(|v| v["verifiedComplete"] = json!(false)),
        Box::new(|v| {
            v["steps"].as_array_mut().unwrap().pop();
        }),
        Box::new(|v| {
            v["steps"][0]["observation"]["legalActions"]
                .as_array_mut()
                .unwrap()
                .reverse()
        }),
        Box::new(|v| v["finalScores"][0]["total"] = json!(99999)),
        Box::new(|v| {
            v["header"]["source"] = json!({"kind":"human","provider":"fixture","reference":"fixture","skillRating":null})
        }),
    ];
    for mutate in mutations {
        let mut changed = original.clone();
        mutate(&mut changed);
        fs::write(&path, serde_json::to_vec(&changed).unwrap()).unwrap();
        assert!(export_native_files(std::slice::from_ref(&path), &temp.dataset()).is_err());
        assert!(!temp.dataset().exists());
    }
    fs::write(&path, serde_json::to_vec(&original).unwrap()).unwrap();
    assert!(export_native_files(&[path.clone(), path], &temp.dataset()).is_err());
    assert!(!temp.dataset().exists());
}

#[test]
fn manifest_task_split_paths_targets_and_strata_cannot_be_resigned_into_validity() {
    let temp = Temp::new();
    let m = exported(&temp);
    let mut mutations = Vec::new();
    let mut bad = m.clone();
    bad.task = "policyOnlyBc".into();
    mutations.push(bad);
    let mut bad = m.clone();
    bad.context_count = 512;
    mutations.push(bad);
    let mut bad = m.clone();
    bad.gamma = 0;
    mutations.push(bad);
    let mut bad = m.clone();
    bad.terminal_bootstrap = -0.0;
    mutations.push(bad);
    let mut bad = m.clone();
    bad.games[0].source_file = "../game.json".into();
    mutations.push(bad);
    let mut bad = m.clone();
    bad.games[0].split = if bad.games[0].split == DatasetSplit::Train {
        DatasetSplit::Test
    } else {
        DatasetSplit::Train
    };
    mutations.push(bad);
    let mut bad = m.clone();
    bad.strata[0].samples -= 1;
    mutations.push(bad);
    let mut bad = m.clone();
    bad.shards[0].first_source_index = 1;
    mutations.push(bad);
    for mut bad in mutations {
        resign(&mut bad);
        save_manifest(&temp.dataset(), &bad);
        assert!(load_state_mc_dataset(&temp.dataset()).is_err());
    }
}

#[test]
fn local_new_only_and_cross_task_loads_do_not_modify_inputs_or_existing_outputs() {
    let temp = Temp::new();
    let path = temp.source("native.json", basic(3));
    let before = fs::read(&path).unwrap();
    let m = export_native_files(std::slice::from_ref(&path), &temp.dataset()).unwrap();
    let manifest_bytes = fs::read(temp.dataset().join("manifest.json")).unwrap();
    assert!(export_native_files(std::slice::from_ref(&path), &temp.dataset()).is_err());
    assert_eq!(fs::read(&path).unwrap(), before);
    assert_eq!(
        fs::read(temp.dataset().join("manifest.json")).unwrap(),
        manifest_bytes
    );
    assert!(policy_dataset::load_policy_dataset(&temp.dataset()).is_err());
    assert!(tzolkin_ai::dataset::load_dataset(&temp.dataset()).is_err());
    let policy_dir = temp.0.join("policy");
    policy_dataset::export_native_files(&[path], &policy_dir).unwrap();
    assert!(load_state_mc_dataset(&policy_dir).is_err());
    assert!(
        state_mc_dataset::native_source_files(Path::new("https://example.invalid/source")).is_err()
    );
    assert!(load_state_mc_dataset(Path::new("//server/share/dataset")).is_err());
    assert_eq!(m.samples, basic(3).steps.len());
}

#[test]
fn dedicated_cli_exports_complete_native_state_task_and_rejects_unknown_flags() {
    let temp = Temp::new();
    let inputs = temp.0.join("inputs");
    fs::create_dir(&inputs).unwrap();
    fs::write(
        inputs.join("native.json"),
        serde_json::to_vec(basic(3)).unwrap(),
    )
    .unwrap();
    let output = Command::new(env!("CARGO_BIN_EXE_tzolkin-public-ml"))
        .args(["export-state-native", "--input"])
        .arg(&inputs)
        .arg("--output")
        .arg(temp.dataset())
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let manifest: StateMcManifest = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(manifest.task, state_mc_dataset::TASK);
    assert_eq!(
        load_state_mc_dataset(&temp.dataset()).unwrap().manifest(),
        &manifest
    );
    let failed = Command::new(env!("CARGO_BIN_EXE_tzolkin-public-ml"))
        .args(["export-state-native", "--seed", "17"])
        .output()
        .unwrap();
    assert!(!failed.status.success());
    assert!(failed.stdout.is_empty());
}

#[test]
fn regular_source_and_resigned_jsonl_byte_bounds_fail_closed() {
    let temp = Temp::new();
    let mut m = exported(&temp);
    let shard_path = temp.dataset().join(&m.shards[0].file);
    let original = fs::read(&shard_path).unwrap();
    // JSON whitespace is otherwise valid; a claimed digest cannot enlarge a row budget.
    let mut oversized = vec![b' '; state_mc_dataset::MAX_SAMPLE_BYTES as usize];
    oversized.extend_from_slice(&original);
    fs::write(&shard_path, &oversized).unwrap();
    m.shards[0].bytes = oversized.len() as u64;
    m.shards[0].sha256 = hash(&oversized);
    resign(&mut m);
    save_manifest(&temp.dataset(), &m);
    assert!(load_state_mc_dataset(&temp.dataset()).is_err());
    let inputs = temp.0.join("oversized-input");
    fs::create_dir(&inputs).unwrap();
    let file = fs::File::create(inputs.join("native.json")).unwrap();
    file.set_len(state_mc_dataset::MAX_SOURCE_BYTES + 1)
        .unwrap();
    assert!(state_mc_dataset::native_source_files(&inputs).is_err());
}
