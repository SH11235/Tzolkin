use serde_json::json;
use sha2::{Digest, Sha256};
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};
use tzolkin_ai::dataset::{DatasetSplit, seed_family_id, split_for_family};
use tzolkin_ai::public_state_critic::{LoadedPublicStateCritic, PublicStateCriticArtifact};
use tzolkin_ai::state_mc_dataset::{export_native_files, load_state_mc_dataset};
use tzolkin_ai::state_mc_training::{
    self, StateMcConfig, StateMcTrainingCheckpoint, train_dataset,
};

struct Temp(PathBuf);
impl Temp {
    fn new() -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let p = std::env::temp_dir().join(format!(
            "tzolkin-state-mc-training-{}-{}-{}",
            std::process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir(&p).unwrap();
        Self(p)
    }
}
impl Drop for Temp {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}
struct Fixture {
    _temp: Temp,
    dir: PathBuf,
    pending_off_turn: usize,
}
fn fixture() -> &'static Fixture {
    static FIXTURE: OnceLock<Fixture> = OnceLock::new();
    FIXTURE.get_or_init(|| {
        // Five actual complete correctness-fixture generations per test process. These
        // consumed unit seeds are not fresh experimental Train/Test families.
        let temp = Temp::new();
        let mut files = Vec::new();
        let mut pending = 0;
        for (players, seed, split) in [
            (3, 0, DatasetSplit::Train),
            (4, 0, DatasetSplit::Train),
            (3, 17, DatasetSplit::Train),
            (3, 3, DatasetSplit::Validation),
            (4, 10, DatasetSplit::Test),
        ] {
            assert_eq!(split_for_family(&seed_family_id(seed)).unwrap(), split);
            let (_, _, record) =
                tzolkin_ai::replay::play_game_fast(players, seed, Default::default(), true)
                    .unwrap();
            let record = record.unwrap();
            pending += record
                .steps
                .iter()
                .filter(|s| s.actor != s.turn_player)
                .count();
            let path = temp.0.join(format!("source-{players}-{seed}.json"));
            fs::write(&path, serde_json::to_vec(&record).unwrap()).unwrap();
            files.push(path);
        }
        assert!(pending > 0);
        let dir = temp.0.join("dataset");
        export_native_files(&files, &dir).unwrap();
        Fixture {
            _temp: temp,
            dir,
            pending_off_turn: pending,
        }
    })
}
fn copy_tree(from: &Path, to: &Path) {
    fs::create_dir(to).unwrap();
    for e in fs::read_dir(from).unwrap() {
        let e = e.unwrap();
        let target = to.join(e.file_name());
        if e.file_type().unwrap().is_dir() {
            copy_tree(&e.path(), &target);
        } else {
            fs::copy(e.path(), target).unwrap();
        }
    }
}
fn one_config() -> StateMcConfig {
    StateMcConfig {
        epochs: 1,
        batch_size: 256,
        learning_rate: 0.001,
        seed: 7,
    }
}
fn trained() -> &'static StateMcTrainingCheckpoint {
    static C: OnceLock<StateMcTrainingCheckpoint> = OnceLock::new();
    C.get_or_init(|| {
        let dataset = load_state_mc_dataset(&fixture().dir).unwrap();
        train_dataset(&dataset, &one_config(), None)
            .unwrap()
            .checkpoint()
            .clone()
    })
}
fn cli(args: &[&str]) -> std::process::Output {
    Command::new(env!("CARGO_BIN_EXE_tzolkin-public-ml"))
        .args(args)
        .output()
        .unwrap()
}
fn bytes(out: &Path) -> [Vec<u8>; 3] {
    ["model.json", "checkpoint.json", "metrics.json"].map(|name| fs::read(out.join(name)).unwrap())
}

#[test]
#[ignore = "slow ML correctness integration; run npm run test:ml:slow"]
fn complete_native_scalar_continuous_resume_noop_and_actual_cli_are_exact() {
    let f = fixture();
    assert!(f.pending_off_turn > 0);
    let temp = Temp::new();
    let dataset = load_state_mc_dataset(&f.dir).unwrap();
    assert_eq!(dataset.manifest().games.len(), 5);
    let one = trained();
    let config = StateMcConfig {
        epochs: 2,
        ..one_config()
    };
    let continuous = train_dataset(&dataset, &config, None).unwrap();
    let resumed = train_dataset(&dataset, &config, Some(one)).unwrap();
    assert_eq!(continuous.checkpoint(), resumed.checkpoint());
    let continuous_dir = temp.0.join("continuous");
    continuous.save_new_directory(&continuous_dir).unwrap();
    let resume_dir = temp.0.join("resume");
    resumed.save_new_directory(&resume_dir).unwrap();
    assert_eq!(bytes(&continuous_dir), bytes(&resume_dir));
    let reloaded = StateMcTrainingCheckpoint::load(&resume_dir.join("checkpoint.json")).unwrap();
    let noop = train_dataset(&dataset, &config, Some(&reloaded)).unwrap();
    let noop_dir = temp.0.join("noop");
    noop.save_new_directory(&noop_dir).unwrap();
    assert_eq!(bytes(&continuous_dir), bytes(&noop_dir));
    assert!(!continuous.metrics().calibrated);
    assert!(!continuous.metrics().strength_measured);
    assert!(
        continuous.metrics().final_train.methods[0].mse.unwrap()
            < continuous.metrics().initial_train.methods[0].mse.unwrap()
    );
    let work = continuous.work_plan();
    assert_eq!(work.split_passes[2], 0);
    assert_eq!(work.source_read_bytes, 3 * work.declared_source_bytes);
    assert_eq!(work.native_replay_step_bound, 4 * work.reconstructed_rows);
    let cli_one = temp.0.join("cli-one");
    let result = cli(&[
        "train-state",
        "--input",
        f.dir.to_str().unwrap(),
        "--output",
        cli_one.to_str().unwrap(),
        "--epochs",
        "1",
        "--batch-size",
        "256",
    ]);
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    let cli_two = temp.0.join("cli-two");
    let result = cli(&[
        "resume-state",
        "--input",
        f.dir.to_str().unwrap(),
        "--checkpoint",
        cli_one.join("checkpoint.json").to_str().unwrap(),
        "--output",
        cli_two.to_str().unwrap(),
        "--epochs",
        "2",
    ]);
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    assert_eq!(bytes(&continuous_dir), bytes(&cli_two));
    for (name, split) in [
        ("validation", DatasetSplit::Validation),
        ("test", DatasetSplit::Test),
    ] {
        let report = state_mc_training::evaluate_dataset(&dataset, &reloaded, split).unwrap();
        let result = cli(&[
            "evaluate-state",
            "--input",
            f.dir.to_str().unwrap(),
            "--checkpoint",
            cli_two.join("checkpoint.json").to_str().unwrap(),
            "--split",
            name,
        ]);
        assert!(
            result.status.success(),
            "{}",
            String::from_utf8_lossy(&result.stderr)
        );
        assert_eq!(
            serde_json::from_slice::<state_mc_training::StateMcEvaluation>(&result.stdout).unwrap(),
            report
        );
    }
    let artifact = PublicStateCriticArtifact::load(&cli_two.join("model.json")).unwrap();
    let estimate = LoadedPublicStateCritic::new(&artifact)
        .unwrap()
        .estimate(dataset.iter().next().unwrap().unwrap().context())
        .unwrap();
    assert!(estimate.raw_return_estimate.is_finite());
}
#[test]
fn checkpoint_contract_rejects_resigned_progress_math_identity_and_task_changes() {
    let temp = Temp::new();
    let checkpoint = trained();
    let base = serde_json::to_value(checkpoint).unwrap();
    for (key, value) in [
        ("schema", json!("tzolkin-public-policy-checkpoint-v1")),
        ("task", json!("ppo")),
        ("mathContract", json!("normalize-last-batch")),
        ("randomState", json!(0)),
        ("completedEpochs", json!(2)),
        ("initialModelChecksum", json!("0".repeat(64))),
    ] {
        let mut altered = base.clone();
        altered[key] = value;
        altered["checksum"] = json!("");
        let typed: StateMcTrainingCheckpoint = serde_json::from_value(altered.clone()).unwrap();
        let hash = Sha256::digest(serde_json::to_vec(&typed).unwrap());
        altered["checksum"] = json!(hash.iter().map(|b| format!("{b:02x}")).collect::<String>());
        let path = temp.0.join(format!("{key}.json"));
        fs::write(&path, serde_json::to_vec(&altered).unwrap()).unwrap();
        assert!(
            StateMcTrainingCheckpoint::load(&path).is_err(),
            "accepted{key}"
        );
    }
    let dataset = load_state_mc_dataset(&fixture().dir).unwrap();
    let mut backwards = one_config();
    backwards.epochs = 0;
    assert!(train_dataset(&dataset, &backwards, Some(checkpoint)).is_err());
    let mut incompatible = one_config();
    incompatible.batch_size = 128;
    assert!(train_dataset(&dataset, &incompatible, Some(checkpoint)).is_err());
}
#[test]
fn whole_integrity_checks_include_test_and_save_refuses_changes_without_marker() {
    let temp = Temp::new();
    let dir = temp.0.join("dataset");
    copy_tree(&fixture().dir, &dir);
    let dataset = load_state_mc_dataset(&dir).unwrap();
    let output = train_dataset(&dataset, &one_config(), Some(trained())).unwrap();
    let test = dataset
        .manifest()
        .games
        .iter()
        .find(|g| g.split == DatasetSplit::Test)
        .unwrap();
    let path = dir.join(&test.source_file);
    let mut contents = fs::read(&path).unwrap();
    contents.push(b' ');
    fs::write(&path, contents).unwrap();
    let target = temp.0.join("refused");
    assert!(output.save_new_directory(&target).is_err());
    assert!(!target.exists());
    assert!(train_dataset(&dataset, &one_config(), Some(trained())).is_err());
}
#[test]
fn strict_cli_config_and_new_only_preflight_fail_before_loading_inputs() {
    let temp = Temp::new();
    let existing = temp.0.join("existing");
    fs::create_dir(&existing).unwrap();
    fs::write(existing.join("sentinel"), b"keep").unwrap();
    for args in [
        vec!["train-state", "--kernel", "auto"],
        vec!["train-state", "--epochs", "1", "--epochs", "2"],
        vec![
            "resume-state",
            "--input",
            "missing",
            "--output",
            temp.0.join("new").to_str().unwrap(),
        ],
        vec!["evaluate-state", "--split", "unknown"],
        vec![
            "train-state",
            "--input",
            "missing",
            "--output",
            existing.to_str().unwrap(),
        ],
    ] {
        let result = cli(&args);
        assert!(!result.status.success());
    }
    assert_eq!(fs::read(existing.join("sentinel")).unwrap(), b"keep");
    for c in [
        StateMcConfig {
            epochs: 0,
            ..one_config()
        },
        StateMcConfig {
            epochs: 1001,
            ..one_config()
        },
        StateMcConfig {
            batch_size: 0,
            ..one_config()
        },
        StateMcConfig {
            learning_rate: f32::NAN,
            ..one_config()
        },
        StateMcConfig {
            learning_rate: f32::INFINITY,
            ..one_config()
        },
    ] {
        assert!(c.validate().is_err());
    }
    let help = cli(&["--help"]);
    let help = String::from_utf8(help.stdout).unwrap();
    assert!(help.contains("train-state"));
    assert!(help.contains("evaluate-state"));
}
