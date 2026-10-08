use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::{Command, Stdio};
use tzolkin_ai::{dataset, experiment, model::ModelArtifact, replay, training::TrainingCheckpoint};
use tzolkin_core::{GameOptions, create_game, observation::observe};

struct Temporary(PathBuf);
impl Drop for Temporary {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}
fn cli(args: &[&str]) -> std::process::Output {
    Command::new(env!("CARGO_BIN_EXE_tzolkin-ai"))
        .args(args)
        .output()
        .unwrap()
}
#[test]
fn native_cli_trains_resumes_selects_legal_moves_and_enforces_evaluation_provenance() {
    let root = std::env::temp_dir().join(format!("tzolkin-ml-pipeline-cli-{}", std::process::id()));
    fs::create_dir(&root).unwrap();
    let guard = Temporary(root);
    let records = [0, 3].map(|seed| {
        replay::play_game(2, seed, GameOptions::default(), true)
            .unwrap()
            .2
            .unwrap()
    });
    let dataset_path = guard.0.join("dataset");
    dataset::export_dataset(&records, &dataset_path).unwrap();
    let input = dataset_path.to_str().unwrap();
    let trained = guard.0.join("trained");
    let result = cli(&[
        "train",
        "--input",
        input,
        "--output",
        trained.to_str().unwrap(),
        "--epochs",
        "1",
        "--batch-size",
        "16",
        "--seed",
        "11235",
    ]);
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    let checkpoint_path = trained.join("checkpoint.json");
    let checkpoint = TrainingCheckpoint::load(&checkpoint_path).unwrap();
    let model_path = trained.join("model.json");
    let model = ModelArtifact::load(&model_path).unwrap();
    assert_eq!(checkpoint.model, model);
    let before = fs::read(&model_path).unwrap();
    assert!(
        !cli(&[
            "train",
            "--input",
            input,
            "--output",
            trained.to_str().unwrap()
        ])
        .status
        .success()
    );
    assert_eq!(fs::read(&model_path).unwrap(), before);
    let resumed = guard.0.join("resumed");
    assert!(
        cli(&[
            "train",
            "--input",
            input,
            "--output",
            resumed.to_str().unwrap(),
            "--resume",
            checkpoint_path.to_str().unwrap(),
            "--epochs",
            "2"
        ])
        .status
        .success()
    );
    assert_eq!(
        TrainingCheckpoint::load(&resumed.join("checkpoint.json"))
            .unwrap()
            .completed_epochs,
        2
    );

    let state = create_game(vec!["A".into(), "B".into(), "C".into()], 11235, false).unwrap();
    let observation = observe(&state, state.current_player).unwrap();
    let mut child = Command::new(env!("CARGO_BIN_EXE_tzolkin-ai"))
        .args(["choose", "--model", model_path.to_str().unwrap()])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
        .unwrap();
    child
        .stdin
        .take()
        .unwrap()
        .write_all(&serde_json::to_vec(&observation).unwrap())
        .unwrap();
    let choice = child.wait_with_output().unwrap();
    assert!(choice.status.success());
    let choice: tzolkin_ai::Decision = serde_json::from_slice(&choice.stdout).unwrap();
    assert!(
        observation
            .legal_actions
            .iter()
            .any(|action| action.r#move == choice.r#move)
    );
    assert_eq!(choice.observation_key, observation.observation_key);
    let loaded = dataset::load_dataset(&dataset_path).unwrap();
    assert!(
        experiment::evaluate_model(&checkpoint, &loaded, 3, &[0], GameOptions::default()).is_err()
    );
    let mut foreign = checkpoint.clone();
    foreign.dataset_fingerprint = "a".repeat(64);
    assert!(
        experiment::evaluate_model(&foreign, &loaded, 3, &[48], GameOptions::default()).is_err()
    );
    // The all-seat held-out evaluator records every failed game explicitly rather than
    // treating a bounded unfinished rollout as a completed loss or dropping its seat.
    let report =
        experiment::evaluate_model(&checkpoint, &loaded, 2, &[48], GameOptions::default()).unwrap();
    assert_eq!(report.games.len(), 2);
    assert_eq!(report.completed + report.failed, 2);
    assert!(
        report
            .games
            .iter()
            .enumerate()
            .all(|(seat, game)| game.learned_seat == seat)
    );
    assert!(
        report
            .games
            .iter()
            .all(|game| game.utility.is_some() != game.error.is_some())
    );
    assert!(
        report
            .games
            .iter()
            .all(|game| game.decisions.is_some() == game.error.is_none())
    );
    let unfinished = report
        .games
        .iter()
        .find(|game| {
            game.error
                .as_ref()
                .is_some_and(|error| error.contains("Game did not finish"))
        })
        .expect("Fixture must exercise the bounded unfinished-game path");
    assert_eq!(unfinished.decisions, None);
    let json = serde_json::to_value(unfinished).unwrap();
    assert!(json["decisions"].is_null());
}

#[test]
fn mixed_model_replay_records_each_seat_and_remains_independently_verifiable() {
    let (_, _, record) = replay::play_game(2, 0, GameOptions::default(), true).unwrap();
    let mut record = record.unwrap();
    let artifact = ModelArtifact::new(replay::catalog_hash(), 11235).unwrap();
    // Provenance does not assert the recorded decisions were optimal, or reproduce
    // the policy. Independent reconstruction still verifies every operation.
    record.header.source = replay::ReplaySource::PolicySelfPlay {
        policies: vec![
            experiment::learned_seat(&artifact),
            experiment::heuristic_seat(),
        ],
    };
    replay::verify_replay(&record).unwrap();
    if let replay::ReplaySource::PolicySelfPlay { policies } = &mut record.header.source {
        policies.pop();
    }
    assert!(replay::verify_replay(&record).is_err());
}
