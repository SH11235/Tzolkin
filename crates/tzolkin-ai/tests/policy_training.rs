use serde_json::Value;
use sha2::{Digest, Sha256};
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicU64, Ordering};
use tzolkin_ai::dataset::{self, DatasetSplit};
use tzolkin_ai::policy_dataset::{export_native_files, load_policy_dataset};
use tzolkin_ai::policy_training::{
    PolicyBcConfig, PolicyTrainingCheckpoint, evaluate_dataset, train_dataset,
};
use tzolkin_ai::public_model::{
    LoadedPublicPolicy, PARAMETER_COUNT, PublicPolicyArtifact, ValueValidity,
};
use tzolkin_ai::replay::{self, GameReplay};
use tzolkin_core::GameOptions;

struct Temp(PathBuf);
impl Temp {
    fn new() -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let dir = std::env::temp_dir().join(format!(
            "tzolkin-public-training-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir(&dir).unwrap();
        Self(dir)
    }
    fn export(&self, count: usize) -> PathBuf {
        let sources: Vec<_> = records()
            .iter()
            .take(count)
            .enumerate()
            .map(|(i, record)| {
                let path = self.0.join(format!("source-{i}.json"));
                fs::write(&path, serde_json::to_vec(record).unwrap()).unwrap();
                path
            })
            .collect();
        let path = self.0.join("dataset");
        export_native_files(&sources, &path).unwrap();
        path
    }
}
impl Drop for Temp {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}
fn records() -> &'static [GameReplay] {
    static GAMES: OnceLock<Vec<GameReplay>> = OnceLock::new();
    GAMES.get_or_init(|| {
        [
            DatasetSplit::Train,
            DatasetSplit::Validation,
            DatasetSplit::Test,
        ]
        .into_iter()
        .enumerate()
        .map(|(i, split)| {
            let seed = (0..1000)
                .find(|seed| {
                    dataset::split_for_family(&dataset::seed_family_id(*seed)).unwrap() == split
                })
                .unwrap();
            replay::play_game_fast(
                if i == 2 { 4 } else { 3 },
                seed,
                GameOptions::default(),
                true,
            )
            .unwrap()
            .2
            .unwrap()
        })
        .collect()
    })
}
fn hash(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}
fn resign(checkpoint: &mut PolicyTrainingCheckpoint) {
    checkpoint.checksum.clear();
    checkpoint.checksum = hash(&serde_json::to_vec(checkpoint).unwrap());
}
fn binary(args: &[&str]) -> std::process::Output {
    Command::new(env!("CARGO_BIN_EXE_tzolkin-public-ml"))
        .args(args)
        .output()
        .unwrap()
}
fn path(path: &Path) -> &str {
    path.to_str().unwrap()
}

#[test]
fn native_bc_epoch_resume_matches_continuous_bits_metrics_and_actual_train_resume_evaluate_cli() {
    let temp = Temp::new();
    let dataset_dir = temp.export(3);
    let dataset = load_policy_dataset(&dataset_dir).unwrap();
    let config = PolicyBcConfig {
        epochs: 2,
        batch_size: 128,
        learning_rate: 0.003,
        seed: 7,
    };
    let continuous = train_dataset(&dataset, &config, None).unwrap();
    let first = train_dataset(
        &dataset,
        &PolicyBcConfig {
            epochs: 1,
            ..config.clone()
        },
        None,
    )
    .unwrap();
    assert!(
        continuous.metrics.final_train.policy_loss < continuous.metrics.initial_train.policy_loss
    );
    assert_eq!(continuous.model.model.parameters().len(), 16449);
    assert!(!continuous.metrics.strength_measured);
    let first_dir = temp.0.join("first");
    first.save_new_directory(&first_dir).unwrap();
    let checkpoint_path = first_dir.join("checkpoint.json");
    let checkpoint = PolicyTrainingCheckpoint::load(&checkpoint_path).unwrap();
    assert_eq!(first.checkpoint, checkpoint);
    let resumed = train_dataset(&dataset, &config, Some(&checkpoint)).unwrap();
    // Includes all parameters, moments, step, RNG, metadata and f64 loss reports/checksum.
    assert_eq!(resumed, continuous);
    assert_eq!(
        serde_json::to_vec(&resumed).unwrap(),
        serde_json::to_vec(&continuous).unwrap()
    );
    let before = fs::read(&checkpoint_path).unwrap();
    assert!(continuous.save_new_directory(&first_dir).is_err());
    assert!(checkpoint.save_new(&checkpoint_path).is_err());
    assert_eq!(fs::read(&checkpoint_path).unwrap(), before);
    let inconsistent_dir = temp.0.join("inconsistent-output");
    let mut inconsistent = continuous.clone();
    inconsistent.metrics.final_train.correct_top1 = 0;
    inconsistent.metrics.final_train.policy_loss += 1.0;
    assert!(inconsistent.save_new_directory(&inconsistent_dir).is_err());
    assert!(!inconsistent_dir.exists());
    let no_op = train_dataset(&dataset, &config, Some(&continuous.checkpoint)).unwrap();
    assert_eq!(no_op, continuous);
    let inference = LoadedPublicPolicy::new(&continuous.model).unwrap();
    let sample = dataset
        .iter_split(DatasetSplit::Test)
        .next()
        .unwrap()
        .unwrap();
    let prediction = inference.predict(sample.candidates()).unwrap();
    assert_eq!(prediction.value, ());
    assert_eq!(
        prediction.value_validity,
        ValueValidity::UnavailablePolicyOnly
    );
    assert_eq!(sample.value_target(), None);
    let heldout = evaluate_dataset(&dataset, &continuous.model, DatasetSplit::Test).unwrap();
    assert_eq!(heldout.samples, records()[2].steps.len());
    assert_eq!(
        continuous.metrics.final_train.samples,
        records()[0].steps.len()
    );
    assert_eq!(
        continuous.metrics.final_validation.samples,
        records()[1].steps.len()
    );
    let json = serde_json::to_value(&continuous.metrics).unwrap();
    for label in [
        "initialTrain",
        "initialValidation",
        "finalTrain",
        "finalValidation",
    ] {
        assert_eq!(json[label]["valueLoss"], Value::Null);
        assert_eq!(json[label]["valueSamples"], 0);
        assert_eq!(json[label]["valueValidity"], "unavailablePolicyOnly");
    }

    let cli_first = temp.0.join("cli-first");
    let output = binary(&[
        "train",
        "--input",
        path(&dataset_dir),
        "--output",
        path(&cli_first),
        "--epochs",
        "1",
        "--batch-size",
        "128",
        "--learning-rate",
        "0.003",
        "--seed",
        "7",
    ]);
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert_eq!(
        serde_json::from_slice::<Value>(&output.stdout).unwrap(),
        serde_json::to_value(&first.metrics).unwrap()
    );
    assert_eq!(
        fs::read(cli_first.join("checkpoint.json")).unwrap(),
        serde_json::to_vec(&first.checkpoint).unwrap()
    );
    let cli_resume = temp.0.join("cli-resume");
    let output = binary(&[
        "resume",
        "--input",
        path(&dataset_dir),
        "--checkpoint",
        path(&cli_first.join("checkpoint.json")),
        "--epochs",
        "2",
        "--output",
        path(&cli_resume),
    ]);
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert_eq!(
        fs::read(cli_resume.join("checkpoint.json")).unwrap(),
        serde_json::to_vec(&continuous.checkpoint).unwrap()
    );
    let output = binary(&[
        "evaluate",
        "--input",
        path(&dataset_dir),
        "--model",
        path(&cli_resume.join("model.json")),
        "--split",
        "test",
    ]);
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert_eq!(
        serde_json::from_slice::<Value>(&output.stdout).unwrap(),
        serde_json::to_value(&heldout).unwrap()
    );

    let mut checkpoint_json = serde_json::to_value(&checkpoint).unwrap();
    checkpoint_json["metrics"]["finalTrain"]
        .as_object_mut()
        .unwrap()
        .remove("valueLoss");
    assert!(serde_json::from_value::<PolicyTrainingCheckpoint>(checkpoint_json).is_err());
    let mut checkpoint_json = serde_json::to_value(&checkpoint).unwrap();
    checkpoint_json["metrics"]["finalTrain"]["valueLoss"] = 0.into();
    assert!(serde_json::from_value::<PolicyTrainingCheckpoint>(checkpoint_json).is_err());
    let mut corruptions = vec![checkpoint.clone(); 14];
    corruptions[0].schema = "tzolkin-training-checkpoint-v1".into();
    corruptions[1].training_version = "other".into();
    corruptions[2].feature_schema = 1;
    corruptions[3].task = "policyValue".into();
    corruptions[4].backend = "avx2".into();
    corruptions[5].dataset.catalog_hash = "0".repeat(64);
    corruptions[6].dataset.input_contract = "all-rules".into();
    corruptions[7].optimizer.first_moment.pop();
    corruptions[8].optimizer.second_moment[0] = -1.0;
    corruptions[9].optimizer.step += 1;
    corruptions[10].completed_epochs += 1;
    corruptions[11].metrics.strength_measured = true;
    corruptions[12].metrics.final_train.value_samples = 1;
    corruptions[13].dataset.strata[0].players = 2;
    for mut corruption in corruptions {
        resign(&mut corruption);
        assert!(corruption.validate().is_err());
        assert!(train_dataset(&dataset, &config, Some(&corruption)).is_err());
    }
    let mut incompatible = checkpoint.clone();
    incompatible.dataset.fingerprint = "0".repeat(64);
    resign(&mut incompatible);
    incompatible.validate().unwrap();
    assert!(
        train_dataset(&dataset, &config, Some(&incompatible))
            .unwrap_err()
            .contains("mismatch")
    );
    let mut forged_metric = checkpoint.clone();
    forged_metric.metrics.final_train.policy_loss += 0.1;
    resign(&mut forged_metric);
    forged_metric.validate().unwrap();
    assert!(
        train_dataset(&dataset, &config, Some(&forged_metric))
            .unwrap_err()
            .contains("metric/model/source")
    );
    for incompatible in [
        PolicyBcConfig {
            batch_size: 64,
            ..config.clone()
        },
        PolicyBcConfig {
            seed: 8,
            ..config.clone()
        },
        PolicyBcConfig {
            learning_rate: 0.002,
            ..config.clone()
        },
        PolicyBcConfig {
            epochs: 1,
            ..config.clone()
        },
    ] {
        assert!(train_dataset(&dataset, &incompatible, Some(&continuous.checkpoint)).is_err());
    }
    let legacy = tzolkin_ai::training::TrainingCheckpoint::load(&checkpoint_path);
    assert!(legacy.is_err());
    assert!(PublicPolicyArtifact::load(&checkpoint_path).is_err());
    assert_eq!(checkpoint.optimizer.first_moment.len(), PARAMETER_COUNT);
}

#[test]
fn integrity_failure_missing_validation_and_after_load_source_or_shard_changes_are_not_skipped() {
    let temp = Temp::new();
    let dataset_dir = temp.export(1);
    let dataset = load_policy_dataset(&dataset_dir).unwrap();
    assert!(
        train_dataset(&dataset, &PolicyBcConfig::default(), None)
            .unwrap_err()
            .contains("Nonempty")
    );

    let temp = Temp::new();
    let dataset_dir = temp.export(3);
    let dataset = load_policy_dataset(&dataset_dir).unwrap();
    let source = dataset_dir.join(&dataset.manifest().games[2].source_file);
    let original = fs::read(&source).unwrap();
    fs::write(&source, b"{}").unwrap();
    let model = PublicPolicyArtifact::new(7).unwrap();
    // Even held-out source corruption aborts training and validation evaluation integrity.
    assert!(train_dataset(&dataset, &PolicyBcConfig::default(), None).is_err());
    assert!(evaluate_dataset(&dataset, &model, DatasetSplit::Validation).is_err());
    fs::write(&source, &original).unwrap();
    let shard = dataset_dir.join(&dataset.manifest().shards.last().unwrap().file);
    let original = fs::read(&shard).unwrap();
    fs::write(&shard, &original[..original.len() - 1]).unwrap();
    assert!(train_dataset(&dataset, &PolicyBcConfig::default(), None).is_err());
    assert!(evaluate_dataset(&dataset, &model, DatasetSplit::Train).is_err());
}

#[test]
fn cli_rejects_unknown_repeated_missing_nonfinite_or_value_and_simd_training_arguments() {
    let temp = Temp::new();
    let output = temp.0.join("output");
    for args in [
        vec!["train", "--kernel", "auto"],
        vec!["train", "--value-weight", "0"],
        vec!["train", "--input", "none", "--input", "again"],
        vec!["train", "--input", "none", "--output"],
        vec![
            "train",
            "--input",
            "none",
            "--output",
            path(&output),
            "--learning-rate",
            "NaN",
        ],
        vec![
            "train",
            "--input",
            "none",
            "--output",
            path(&output),
            "--epochs",
            "0",
        ],
        vec![
            "resume",
            "--input",
            "none",
            "--output",
            path(&output),
            "--checkpoint",
            "none",
        ],
        vec![
            "evaluate", "--input", "none", "--model", "none", "--split", "all",
        ],
        vec!["evaluate", "--kernel", "auto"],
        vec!["train", "--model", "legacy"],
        vec!["resume", "--resume", "legacy"],
    ] {
        let result = binary(&args);
        assert!(!result.status.success(), "{args:?}");
        assert!(!result.stderr.is_empty());
        assert!(result.stdout.is_empty());
        assert!(!output.exists());
    }
    let help = binary(&["--help"]);
    assert!(help.status.success());
    let help = String::from_utf8(help.stdout).unwrap();
    for command in [
        " train ",
        " resume ",
        " evaluate ",
        "value unavailable",
        "epochs=3",
    ] {
        assert!(help.contains(command));
    }
}

#[test]
fn config_rejects_nonfinite_out_of_range_and_required_wire_fields() {
    for config in [
        PolicyBcConfig {
            epochs: 0,
            ..Default::default()
        },
        PolicyBcConfig {
            epochs: 1001,
            ..Default::default()
        },
        PolicyBcConfig {
            batch_size: 0,
            ..Default::default()
        },
        PolicyBcConfig {
            batch_size: 257,
            ..Default::default()
        },
        PolicyBcConfig {
            learning_rate: 0.0,
            ..Default::default()
        },
        PolicyBcConfig {
            learning_rate: 1.01,
            ..Default::default()
        },
        PolicyBcConfig {
            learning_rate: f32::NAN,
            ..Default::default()
        },
        PolicyBcConfig {
            learning_rate: f32::INFINITY,
            ..Default::default()
        },
    ] {
        assert!(config.validate().is_err());
    }
    for wire in [
        r#"{"epochs":1,"batchSize":16,"learningRate":0.001}"#,
        r#"{"epochs":1,"batchSize":16,"learningRate":0.001,"seed":7,"valueWeight":0}"#,
    ] {
        assert!(serde_json::from_str::<PolicyBcConfig>(wire).is_err());
    }
}

#[test]
fn loaders_refuse_network_non_regular_oversized_and_cross_schema_inputs() {
    use tzolkin_ai::policy_training::load_evaluation_model;
    let temp = Temp::new();
    for rejected in [
        "https://example.invalid/model",
        "//server/share/checkpoint",
        r"\\server\share\checkpoint",
    ] {
        assert!(PolicyTrainingCheckpoint::load(Path::new(rejected)).is_err());
        assert!(load_evaluation_model(Path::new(rejected)).is_err());
    }
    assert!(PolicyTrainingCheckpoint::load(&temp.0).is_err());
    assert!(load_evaluation_model(&temp.0).is_err());
    let large = temp.0.join("large.json");
    fs::write(&large, vec![b' '; 8 * 1024 * 1024 + 1]).unwrap();
    assert!(
        PolicyTrainingCheckpoint::load(&large)
            .unwrap_err()
            .contains("bounded")
    );
    assert!(
        load_evaluation_model(&large)
            .unwrap_err()
            .contains("bounded")
    );
    let artifact_path = temp.0.join("model.json");
    let artifact = PublicPolicyArtifact::new(7).unwrap();
    artifact.save_new(&artifact_path).unwrap();
    assert_eq!(load_evaluation_model(&artifact_path).unwrap(), artifact);
    assert!(PolicyTrainingCheckpoint::load(&artifact_path).is_err());
    #[cfg(unix)]
    {
        let link = temp.0.join("link");
        std::os::unix::fs::symlink(&temp.0, &link).unwrap();
        assert!(load_evaluation_model(&link.join("model.json")).is_err());
    }
}

#[test]
fn output_preflight_rejects_existing_or_missing_parent_before_loading_inputs() {
    use tzolkin_ai::policy_training::validate_new_output_directory;
    let temp = Temp::new();
    let output = temp.0.join("new");
    validate_new_output_directory(&output).unwrap();
    assert!(!output.exists());
    assert!(validate_new_output_directory(&temp.0).is_err());
    assert!(validate_new_output_directory(&temp.0.join("missing-parent/output")).is_err());
    let output = temp.0.join("existing.json");
    fs::write(&output, b"original").unwrap();
    assert!(validate_new_output_directory(&output).is_err());
    let failed = binary(&[
        "train",
        "--input",
        "missing-input",
        "--output",
        path(&output),
    ]);
    assert!(!failed.status.success());
    assert!(String::from_utf8_lossy(&failed.stderr).contains("new directory"));
    assert!(failed.stdout.is_empty());
    assert_eq!(fs::read(&output).unwrap(), b"original");
}
