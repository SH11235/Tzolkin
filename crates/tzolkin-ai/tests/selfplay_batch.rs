use std::fs;
use std::path::PathBuf;
#[path = "support/temp_root.rs"]
mod test_temp_root;

use kernel::Kernel;
use sha2::{Digest, Sha256};
use tzolkin_ai::selfplay_batch::{BatchConfig, BatchManifest, generate_batch};
use tzolkin_ai::{kernel, model, replay};
use tzolkin_core::GameOptions;

struct Temporary(PathBuf);
impl Temporary {
    fn new() -> Self {
        let path = test_temp_root::create("tzolkin-batch").unwrap();
        Self(path)
    }
}
impl Drop for Temporary {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

fn config(players: usize, threads: usize) -> BatchConfig {
    BatchConfig {
        players,
        first_seed: 42,
        games: 2,
        threads,
        options: GameOptions::default(),
    }
}

#[test]
fn limits_seed_overflow_bad_model_and_existing_path_fail_before_source_creation() {
    let temporary = Temporary::new();
    let output = temporary.0.join("output");
    let valid = config(3, 1);
    let mut invalid = Vec::new();
    for players in [0, 1, 6] {
        invalid.push(BatchConfig {
            players,
            ..valid.clone()
        });
    }
    for games in [0, 10_001, usize::MAX] {
        invalid.push(BatchConfig {
            games,
            ..valid.clone()
        });
    }
    for threads in [0, 33, usize::MAX] {
        invalid.push(BatchConfig {
            threads,
            ..valid.clone()
        });
    }
    invalid.push(BatchConfig {
        first_seed: u32::MAX,
        ..valid.clone()
    });
    for invalid in invalid {
        assert!(generate_batch(&invalid, None, Kernel::Scalar, &output).is_err());
        assert!(!output.exists());
    }
    let mut invalid_model = model::ModelArtifact::new(replay::catalog_hash(), 7).unwrap();
    invalid_model.checksum = "0".repeat(64);
    assert!(generate_batch(&valid, Some(&invalid_model), Kernel::Scalar, &output).is_err());
    assert!(!output.exists());
    fs::create_dir(&output).unwrap();
    let marker = output.join("keep.txt");
    fs::write(&marker, b"existing output must remain unchanged").unwrap();
    assert!(generate_batch(&valid, None, Kernel::Scalar, &output).is_err());
    assert_eq!(
        fs::read(marker).unwrap(),
        b"existing output must remain unchanged"
    );
    assert!(!output.join("native").exists());
    let existing_file = temporary.0.join("existing-file");
    fs::write(&existing_file, b"existing file").unwrap();
    assert!(generate_batch(&valid, None, Kernel::Scalar, &existing_file).is_err());
    assert_eq!(fs::read(existing_file).unwrap(), b"existing file");
}

#[test]
fn native_json_and_official_results_are_identical_with_one_or_two_threads() {
    let temporary = Temporary::new();
    for players in [2, 3] {
        let serial_path = temporary.0.join(format!("serial-{players}"));
        let parallel_path = temporary.0.join(format!("parallel-{players}"));
        let serial =
            generate_batch(&config(players, 1), None, Kernel::Scalar, &serial_path).unwrap();
        let parallel =
            generate_batch(&config(players, 2), None, Kernel::Auto, &parallel_path).unwrap();
        for (manifest, path) in [(&serial, &serial_path), (&parallel, &parallel_path)] {
            assert!(manifest.complete);
            assert_eq!(manifest.completed, 2);
            assert_eq!(manifest.failed, 0);
            assert_eq!(manifest.backend, "heuristic");
            let persisted: BatchManifest =
                serde_json::from_slice(&fs::read(path.join("manifest.json")).unwrap()).unwrap();
            assert_eq!(
                serde_json::to_value(&persisted).unwrap(),
                serde_json::to_value(manifest).unwrap()
            );
            assert_eq!(fs::read_dir(path.join("native")).unwrap().count(), 2);
        }
        assert_eq!(serial.used_threads, 1);
        assert_eq!(parallel.used_threads, 2);
        for index in 0..2 {
            let a = &serial.games[index];
            let b = &parallel.games[index];
            assert_eq!(a.index, index);
            assert_eq!(a.seed, 42 + index as u32);
            assert_eq!(a.file, b.file);
            assert_eq!(a.sha256, b.sha256);
            assert_eq!(a.bytes, b.bytes);
            assert_eq!(a.decisions, b.decisions);
            assert_eq!(a.final_scores, b.final_scores);
            assert_eq!(a.final_state, b.final_state);
            assert!(a.error.is_none() && b.error.is_none());
            let bytes = fs::read(serial_path.join(a.file.as_ref().unwrap())).unwrap();
            assert_eq!(
                bytes,
                fs::read(parallel_path.join(b.file.as_ref().unwrap())).unwrap()
            );
            assert_eq!(Some(format!("{:x}", Sha256::digest(&bytes))), a.sha256);
            assert_eq!(Some(bytes.len() as u64), a.bytes);
            let source: replay::GameReplay = serde_json::from_slice(&bytes).unwrap();
            assert_eq!(Some(source.steps.len()), a.decisions);
            assert_eq!(source.final_scores, a.final_scores);
            assert_eq!(
                replay::verify_replay(&source).unwrap().final_scores,
                a.final_scores
            );
        }
    }
}

#[test]
fn learned_inference_failure_produces_partial_manifest_with_honest_null_counts() {
    let temporary = Temporary::new();
    // 有限・正しいshapeのartifactでもhead内積はoverflowし得る。最初の候補評価で
    // 必ず失敗させ、4000手を走らせずに本物のlearned runnerの失敗経路を確認する。
    let model = model::ModelArtifact::new(replay::catalog_hash(), 11235).unwrap();
    let mut value = serde_json::to_value(model).unwrap();
    let parameters = value["model"]["parameters"].as_array_mut().unwrap();
    parameters.fill(serde_json::json!(0.0));
    let hidden_bias = tzolkin_ai::features::FEATURE_COUNT * model::HIDDEN;
    let policy_head = hidden_bias + model::HIDDEN;
    parameters[hidden_bias..policy_head].fill(serde_json::json!(1.0));
    parameters[policy_head..policy_head + model::HIDDEN].fill(serde_json::json!(f32::MAX));
    value["checksum"] = serde_json::json!("");
    let mut artifact: model::ModelArtifact = serde_json::from_value(value).unwrap();
    artifact.checksum = format!(
        "{:x}",
        Sha256::digest(serde_json::to_vec(&artifact).unwrap())
    );
    artifact.validate().unwrap();
    let output = temporary.0.join("partial");
    let manifest = generate_batch(&config(3, 2), Some(&artifact), Kernel::Scalar, &output).unwrap();
    assert!(!manifest.complete);
    assert_eq!(manifest.completed, 0);
    assert_eq!(manifest.failed, 2);
    assert_eq!(fs::read_dir(output.join("native")).unwrap().count(), 0);
    let persisted: serde_json::Value =
        serde_json::from_slice(&fs::read(output.join("manifest.json")).unwrap()).unwrap();
    assert_eq!(persisted, serde_json::to_value(&manifest).unwrap());
    for (index, game) in manifest.games.iter().enumerate() {
        assert_eq!(game.seed, 42 + index as u32);
        assert!(!game.complete);
        assert!(
            game.error
                .as_ref()
                .is_some_and(|error| error.contains("Non-finite"))
        );
        assert!(game.decisions.is_none());
        assert!(persisted["games"][index]["decisions"].is_null());
        assert!(game.file.is_none() && game.sha256.is_none() && game.bytes.is_none());
        assert!(game.final_state.is_none() && game.final_scores.is_empty());
    }
}
