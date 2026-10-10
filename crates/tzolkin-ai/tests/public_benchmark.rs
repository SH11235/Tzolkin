use crate::test_temp_root;
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use tzolkin_ai::dataset::{self, DatasetSplit};
use tzolkin_ai::kernel::Kernel;
use tzolkin_ai::policy_dataset::{export_native_files, load_policy_dataset};
use tzolkin_ai::policy_training::{PolicyBcConfig, train_dataset};
use tzolkin_ai::public_model::PublicPolicyArtifact;
use tzolkin_ai::public_native::{PreparedPublicPolicy, play_game};
use tzolkin_ai::replay;
use tzolkin_core::GameOptions;

struct Temp(PathBuf);
impl Temp {
    fn new() -> Self {
        let path = test_temp_root::create("tzolkin-v2-benchmark").unwrap();
        Self(path)
    }
}
impl Drop for Temp {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}
fn text(path: &Path) -> &str {
    path.to_str().unwrap()
}
fn cli(args: &[&str]) -> std::process::Output {
    Command::new(env!("CARGO_BIN_EXE_tzolkin-public-ml-bench"))
        .args(args)
        .output()
        .unwrap()
}
fn json(output: &std::process::Output) -> Value {
    assert!(
        output.status.success(),
        "{}\n{}",
        String::from_utf8_lossy(&output.stderr),
        String::from_utf8_lossy(&output.stdout)
    );
    serde_json::from_slice(&output.stdout).unwrap()
}

#[test]
#[ignore = "slow ML correctness integration; run npm run test:ml:slow"]
fn genuine_trained_native_pipeline_verifies_corpus_and_complete_mixed_games_without_clocks() {
    let temp = Temp::new();
    let mut files = Vec::new();
    for split in [DatasetSplit::Train, DatasetSplit::Validation] {
        let seed = (0..1000)
            .find(|seed| {
                dataset::split_for_family(&dataset::seed_family_id(*seed)).unwrap() == split
            })
            .unwrap();
        let record = replay::play_game_fast(3, seed, GameOptions::default(), true)
            .unwrap()
            .2
            .unwrap();
        let path = temp.0.join(format!("teacher-{}.json", files.len()));
        fs::write(&path, serde_json::to_vec(&record).unwrap()).unwrap();
        files.push(path);
    }
    let data = temp.0.join("dataset");
    export_native_files(&files, &data).unwrap();
    let dataset = load_policy_dataset(&data).unwrap();
    let trained = train_dataset(
        &dataset,
        &PolicyBcConfig {
            epochs: 1,
            batch_size: 128,
            learning_rate: 0.003,
            seed: 7,
        },
        None,
    )
    .unwrap();
    let training = temp.0.join("trained");
    trained.save_new_directory(&training).unwrap();
    let model = training.join("model.json");
    let checkpoint = training.join("checkpoint.json");
    let four = temp.0.join("four.json");
    let record = replay::play_game_fast(4, 11235, GameOptions::default(), true)
        .unwrap()
        .2
        .unwrap();
    fs::write(&four, serde_json::to_vec(&record).unwrap()).unwrap();
    files.push(four);
    let source_data = temp.0.join("corpus");
    let source_manifest = export_native_files(&files, &source_data).unwrap();
    let common = [
        "--model",
        text(&model),
        "--checkpoint",
        text(&checkpoint),
        "--dataset",
        text(&data),
        "--verify-only",
    ];
    let mut infer = vec!["infer"];
    infer.extend(common);
    infer.extend([
        "--source-dataset",
        text(&source_data),
        "--players",
        "3",
        "--kernel",
        "auto",
    ]);
    let report = json(&cli(&infer));
    assert_eq!(report["schema"], "tzolkin-public-policy-performance-v1");
    assert_eq!(report["modelChecksum"], trained.model.checksum);
    assert_eq!(
        report["modelFileSha256"],
        format!("{:x}", Sha256::digest(fs::read(&model).unwrap()))
    );
    assert_eq!(
        report["checkpointFileSha256"],
        format!("{:x}", Sha256::digest(fs::read(&checkpoint).unwrap()))
    );
    assert_eq!(report["verifyOnly"], true);
    assert_eq!(report["strengthMeasured"], false);
    let measurements = &report["measurements"];
    let samples: usize = source_manifest
        .games
        .iter()
        .filter(|g| g.players == 3)
        .map(|g| g.samples)
        .sum();
    let rows: usize = source_manifest
        .games
        .iter()
        .filter(|g| g.players == 3)
        .map(|g| g.candidates)
        .sum();
    assert_eq!(measurements["observations"], samples);
    assert_eq!(measurements["candidateRows"], rows);
    assert_eq!(report["candidateEvaluationBudgetUsed"], rows * 10);
    assert_eq!(measurements["allWarmupTimedOutputsAudited"], true);
    assert_eq!(measurements["parity"]["numericalWithinTolerance"], true);
    for key in [
        "featureEncoding",
        "scalarPredict",
        "selectedPredict",
        "scalarChoose",
        "selectedChoose",
    ] {
        assert!(measurements[key].is_null());
    }
    let prepared =
        PreparedPublicPolicy::new(trained.checkpoint.clone(), &dataset, Kernel::Scalar).unwrap();
    let handle = prepared.handle().unwrap();
    for (players, seats) in [("3", "all"), ("4", "0,2")] {
        let mut args = vec!["selfplay"];
        args.extend(common);
        args.extend(["--players", players, "--seats", seats, "--kernel", "scalar"]);
        let output = temp.0.join(format!("verified-{players}.json"));
        args.extend(["--output", text(&output)]);
        let report = json(&cli(&args));
        assert_eq!(
            serde_json::from_slice::<Value>(&fs::read(&output).unwrap()).unwrap(),
            report
        );
        let result = &report["measurements"];
        assert_eq!(result["allComplete"], true);
        assert_eq!(result["comparable"], true);
        assert!(result["nativeSpeedup"].is_null());
        assert!(result["timings"]["scalar"].is_null());
        assert!(result["games"][0]["timed"].as_array().unwrap().is_empty());
        let warmups = result["games"][0]["warmups"].as_array().unwrap();
        assert_eq!(warmups.len(), 2);
        assert_eq!(warmups[0]["trajectoryHash"], warmups[1]["trajectoryHash"]);
        assert_eq!(warmups[0]["workloadHash"], warmups[1]["workloadHash"]);
        assert_eq!(warmups[0]["finalState"], warmups[1]["finalState"]);
        assert!(warmups.iter().all(|w| w["complete"] == true
            && w["decisions"].as_u64().unwrap() > 0
            && w["finalScores"].as_array().unwrap().len() == players.parse::<usize>().unwrap()));
        // The bench-local trace harness must match the independently used A4 API,
        // including all score bits reconstructed from each actual actor's Observation.
        let n = players.parse::<usize>().unwrap();
        let selected = if seats == "all" {
            (0..n).collect::<Vec<_>>()
        } else {
            vec![0, 2]
        };
        let (state, count, record) =
            play_game(&handle, n, 11235, GameOptions::default(), &selected, true).unwrap();
        let record = record.unwrap();
        let decisions = record
            .steps
            .iter()
            .map(|step| {
                if selected.contains(&step.actor) {
                    handle.choose_move(&step.observation).unwrap()
                } else {
                    tzolkin_ai::choose_move(&step.observation).unwrap()
                }
            })
            .collect::<Vec<_>>();
        let bits = decisions
            .iter()
            .map(|d| {
                (
                    d.actor,
                    &d.observation_key,
                    &d.policy_version,
                    &d.r#move,
                    d.score.to_bits(),
                )
            })
            .collect::<Vec<_>>();
        let mut hash = Sha256::new();
        hash.update(b"tzolkin-public-benchmark-decisions-v1\0");
        hash.update(serde_json::to_vec(&bits).unwrap());
        assert_eq!(
            warmups[0]["trajectoryHash"],
            format!("{:x}", hash.finalize())
        );
        assert_eq!(warmups[0]["decisions"], count);
        assert_eq!(warmups[0]["finalState"], replay::state_key(&state).unwrap());
        let failed = cli(&args);
        assert!(!failed.status.success());
        assert!(failed.stdout.is_empty());
    }
    let other_model = temp.0.join("other-model.json");
    PublicPolicyArtifact::new(42)
        .unwrap()
        .save_new(&other_model)
        .unwrap();
    infer[2] = text(&other_model);
    let failed = cli(&infer);
    assert!(!failed.status.success());
    assert!(failed.stdout.is_empty());
    assert!(String::from_utf8_lossy(&failed.stderr).contains("differs from checkpoint"));
    infer[2] = text(&model);
    // A changed non-selected (4p) source still fails full EOF validation for a 3p corpus.
    let source = source_manifest
        .games
        .iter()
        .find(|g| g.players == 4)
        .unwrap();
    fs::write(source_data.join(&source.source_file), b"{}").unwrap();
    let failed = cli(&infer);
    assert!(!failed.status.success());
    let report: Value = serde_json::from_slice(&failed.stdout).unwrap();
    assert_eq!(report["success"], false);
    assert!(report["measurements"].is_null());
    assert!(report["error"].as_str().unwrap().len() > 3);
}

#[test]
fn cli_rejects_raw_feature_and_unknown_input_without_running_a_measurement() {
    for args in [
        vec!["infer", "--raw-features", "x"],
        vec!["selfplay", "--model", "x"],
        vec![
            "infer",
            "--model",
            "x",
            "--checkpoint",
            "x",
            "--dataset",
            "x",
            "--source-dataset",
            "x",
            "--players",
            "2",
        ],
    ] {
        let output = cli(&args);
        assert!(!output.status.success());
        assert!(output.stdout.is_empty());
    }
    let help = cli(&["--help"]);
    assert!(help.status.success());
    assert!(String::from_utf8_lossy(&help.stdout).contains("scalar|auto|avx2|sse2|neon|simd128"));
}
