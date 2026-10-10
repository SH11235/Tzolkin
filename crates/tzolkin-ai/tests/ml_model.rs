use crate::test_temp_root;
use std::fs;
use std::path::{Path, PathBuf};

use sha2::{Digest, Sha256};
use tzolkin_ai::dataset::{
    DatasetGame, DatasetManifest, DatasetShard, DatasetSplit, DatasetStratum, TrainingSample,
    load_dataset,
};
use tzolkin_ai::features::{FEATURE_COUNT, FEATURE_SCHEMA};
use tzolkin_ai::model::{LEARNED_POLICY_VERSION, ModelArtifact, PARAMETER_COUNT, scalar_dot};
use tzolkin_ai::replay;
use tzolkin_ai::training::{TrainingCheckpoint, TrainingConfig, train, train_dataset};
use tzolkin_core::observation::{MOVE_SCHEMA, OBSERVATION_SCHEMA, observe};
use tzolkin_core::{GameOptions, create_game};

struct Temporary(PathBuf);
impl Temporary {
    fn new() -> Self {
        let path = test_temp_root::create("tzolkin-ml").unwrap();
        Self(path)
    }
}
impl Drop for Temporary {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}
fn sample(game: &str, family: &str, players: usize, candidates: usize) -> TrainingSample {
    TrainingSample {
        features: (0..candidates)
            .map(|candidate| {
                let mut features = vec![0.0; FEATURE_COUNT];
                features[0] = 0.25;
                features[384] = if candidates == 1 {
                    1.0
                } else {
                    candidate as f32 * 2.0 / (candidates - 1) as f32 - 1.0
                };
                features
            })
            .collect(),
        chosen: candidates - 1,
        utilities: [1.0, 0.0, 0.0, 0.0, 0.0],
        active: std::array::from_fn(|side| side < players),
        actor: players - 1,
        game_id: game.repeat(32),
        family_id: family.repeat(32),
    }
}
fn examples() -> (Vec<TrainingSample>, Vec<TrainingSample>) {
    (
        vec![
            sample("aa", "22", 2, 2),
            sample("ab", "23", 3, 3),
            sample("ac", "24", 5, 2),
        ],
        vec![sample("bb", "00", 2, 3)],
    )
}
fn config(epochs: usize) -> TrainingConfig {
    TrainingConfig {
        epochs,
        batch_size: 2,
        learning_rate: 0.02,
        seed: 456,
        value_weight: 1.0,
    }
}
fn sha(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

#[test]
fn policy_training_reduces_joint_loss_with_ragged_actions_and_variable_players() {
    let (training, validation) = examples();
    let outcome = train(&training, &validation, &config(30), None).unwrap();
    assert!(
        outcome.metrics.final_train.policy_loss < outcome.metrics.initial_train.policy_loss * 0.3,
        "{:?}",
        outcome.metrics
    );
    assert!(
        outcome.metrics.final_train.value_loss < outcome.metrics.initial_train.value_loss * 0.3
    );
    assert!(
        outcome.metrics.final_validation.total_loss < outcome.metrics.initial_validation.total_loss
    );
    assert!(!outcome.metrics.strength_measured);
    for players in 2..=5 {
        let prediction = outcome
            .model
            .predict(
                &training[0].features[1],
                std::array::from_fn(|side| side < players),
            )
            .unwrap();
        assert!(
            prediction.utilities[..players]
                .iter()
                .all(|value| *value > 0.0)
        );
        assert!(
            prediction.utilities[players..]
                .iter()
                .all(|value| *value == 0.0)
        );
        assert!((prediction.utilities.iter().sum::<f32>() - 1.0).abs() < 1e-6);
    }
}

#[test]
fn tied_utilities_are_supported_and_inactive_targets_are_rejected() {
    let (mut training, mut validation) = examples();
    training.truncate(1);
    training[0].utilities = [0.5, 0.5, 0.0, 0.0, 0.0];
    validation[0].utilities = training[0].utilities;
    let outcome = train(&training, &validation, &config(20), None).unwrap();
    let prediction = outcome
        .model
        .predict(&training[0].features[1], training[0].active)
        .unwrap();
    assert!((prediction.utilities[0] - 0.5).abs() < 0.12);
    assert!((prediction.utilities[1] - 0.5).abs() < 0.12);
    training[0].utilities = [0.5, 0.0, 0.5, 0.0, 0.0];
    assert!(train(&training, &validation, &config(1), None).is_err());
}

#[test]
fn resumed_optimizer_and_shuffle_match_uninterrupted_training_exactly() {
    let (training, validation) = examples();
    let full = train(&training, &validation, &config(6), None).unwrap();
    let first = train(&training, &validation, &config(2), None).unwrap();
    let temporary = Temporary::new();
    let checkpoint_path = temporary.0.join("checkpoint.json");
    first.checkpoint.save_new(&checkpoint_path).unwrap();
    let restored = TrainingCheckpoint::load(&checkpoint_path).unwrap();
    assert_eq!(restored, first.checkpoint);
    let resumed = train(&training, &validation, &config(6), Some(&restored)).unwrap();
    assert_eq!(full.model, resumed.model);
    assert_eq!(full.checkpoint, resumed.checkpoint);
    assert_eq!(full.metrics, resumed.metrics);
    assert!(train(&training, &validation, &config(1), Some(&restored)).is_err());
    let mut changed_config = config(6);
    changed_config.seed += 1;
    assert!(train(&training, &validation, &changed_config, Some(&restored)).is_err());
    let mut changed_data = training.clone();
    changed_data[0].chosen = 0;
    assert!(train(&changed_data, &validation, &config(6), Some(&restored)).is_err());
}

#[test]
fn checkpoint_rejects_optimizer_or_metadata_corruption_and_existing_output() {
    let (training, validation) = examples();
    let outcome = train(&training, &validation, &config(1), None).unwrap();
    let temporary = Temporary::new();
    let path = temporary.0.join("checkpoint.json");
    outcome.checkpoint.save_new(&path).unwrap();
    let original = fs::read(&path).unwrap();
    assert!(outcome.checkpoint.save_new(&path).is_err());
    assert_eq!(fs::read(&path).unwrap(), original);
    for corruption in 0..5 {
        let mut changed = serde_json::to_value(&outcome.checkpoint).unwrap();
        match corruption {
            0 => changed["optimizer"]["step"] = serde_json::json!(999),
            1 => {
                changed["optimizer"]["firstMoment"]
                    .as_array_mut()
                    .unwrap()
                    .pop();
            }
            2 => changed["optimizer"]["secondMoment"][0] = serde_json::json!(-1.0),
            3 => changed["randomState"] = serde_json::json!(123),
            _ => changed["datasetFingerprint"] = serde_json::json!("0".repeat(64)),
        }
        let changed_path = temporary.0.join("bad.json");
        fs::write(&changed_path, serde_json::to_vec(&changed).unwrap()).unwrap();
        assert!(TrainingCheckpoint::load(&changed_path).is_err());
    }
}

#[test]
fn malformed_samples_or_validation_family_overlap_fail_before_training() {
    let (training, validation) = examples();
    let mut overlap = validation.clone();
    overlap[0].family_id = training[0].family_id.clone();
    assert!(train(&training, &overlap, &config(1), None).is_err());
    overlap[0].family_id = validation[0].family_id.clone();
    overlap[0].game_id = training[0].game_id.clone();
    assert!(train(&training, &overlap, &config(1), None).is_err());
    for mutate in [0, 1, 2, 3, 4] {
        let mut invalid = training.clone();
        match mutate {
            0 => invalid[0].chosen = 100,
            1 => {
                invalid[0].features[0].pop();
            }
            2 => invalid[0].features[0][384] = f32::NAN,
            3 => invalid[0].features[1][0] = 0.5,
            _ => invalid[0].active = [true, false, true, false, false],
        }
        assert!(train(&invalid, &validation, &config(1), None).is_err());
    }
    assert!(train(&training, &[], &config(1), None).is_err());
    let mut invalid_config = config(1);
    invalid_config.learning_rate = f32::INFINITY;
    assert!(train(&training, &validation, &invalid_config, None).is_err());
}

#[test]
fn artifacts_validate_shapes_versions_checksums_and_never_replace_existing_outputs() {
    let temporary = Temporary::new();
    let model = ModelArtifact::new(replay::catalog_hash(), 42).unwrap();
    assert_eq!(model.model.parameters().len(), PARAMETER_COUNT);
    let path = temporary.0.join("model.json");
    model.save_new(&path).unwrap();
    assert_eq!(ModelArtifact::load(&path).unwrap(), model);
    let original = fs::read(&path).unwrap();
    assert!(model.save_new(&path).is_err());
    assert_eq!(fs::read(&path).unwrap(), original);
    for field in [
        "schema",
        "featureSchema",
        "hiddenCount",
        "catalogHash",
        "policyVersion",
        "checksum",
    ] {
        let mut changed = serde_json::to_value(&model).unwrap();
        changed[field] = if matches!(field, "schema" | "featureSchema" | "hiddenCount") {
            serde_json::json!(999)
        } else {
            serde_json::json!("mismatch")
        };
        fs::write(
            temporary.0.join("bad.json"),
            serde_json::to_vec(&changed).unwrap(),
        )
        .unwrap();
        assert!(ModelArtifact::load(&temporary.0.join("bad.json")).is_err());
    }
    let mut shape = serde_json::to_value(&model).unwrap();
    shape["model"]["parameters"].as_array_mut().unwrap().pop();
    fs::write(
        temporary.0.join("bad.json"),
        serde_json::to_vec(&shape).unwrap(),
    )
    .unwrap();
    assert!(ModelArtifact::load(&temporary.0.join("bad.json")).is_err());
    let mut features = vec![0.0; FEATURE_COUNT];
    features[0] = f32::INFINITY;
    assert!(
        model
            .predict(&features, [true, true, false, false, false])
            .is_err()
    );
    assert!(
        model
            .predict(&vec![0.0; FEATURE_COUNT], [true, false, true, false, false])
            .is_err()
    );
}

#[test]
fn learned_choice_uses_only_redacted_observation_and_returns_a_legal_move() {
    let names = vec!["A".into(), "B".into(), "C".into()];
    let state = create_game(names, 51, false).unwrap();
    let observation = observe(&state, state.current_player).unwrap();
    let model = ModelArtifact::new(replay::catalog_hash(), 99).unwrap();
    let first = model.choose_move(&observation).unwrap();
    assert!(
        observation
            .legal_actions
            .iter()
            .any(|legal| legal.r#move == first.r#move)
    );
    assert_eq!(first.policy_version, LEARNED_POLICY_VERSION);
    let mut changed = state.clone();
    changed.seed += 1;
    changed.building_deck.reverse();
    changed.log.push("hidden".into());
    assert_eq!(
        model
            .choose_move(&observe(&changed, changed.current_player).unwrap())
            .unwrap(),
        first
    );
    let mut invalid = observation;
    invalid.observation_key = "changed".into();
    assert!(model.choose_move(&invalid).is_err());
    assert_eq!(
        scalar_dot(&[1.0, 2.0, -1.0], &[3.0, -2.0, 4.0]).unwrap(),
        -5.0
    );
    assert!(scalar_dot(&[1.0], &[]).is_err());
    assert!(scalar_dot(&[f32::MAX], &[f32::MAX]).is_err());
}

// An explicitly synthetic, checksummed shard fixture exercises the production
// streaming path. It is not a claim that these fabricated rows are real games.
fn streaming_fixture(directory: &Path) {
    let samples = [sample("aa", "22", 2, 2), sample("bb", "00", 2, 3)];
    let options = GameOptions::default();
    let games = samples
        .iter()
        .enumerate()
        .map(|(index, sample)| DatasetGame {
            game_id: sample.game_id.clone(),
            family_id: sample.family_id.clone(),
            split: if index == 0 {
                DatasetSplit::Train
            } else {
                DatasetSplit::Validation
            },
            samples: 1,
            players: 2,
            options: options.clone(),
            policy_version: "synthetic-ml-test-v1".into(),
        })
        .collect::<Vec<_>>();
    let shards = samples
        .iter()
        .enumerate()
        .map(|(index, sample)| {
            let mut bytes = serde_json::to_vec(sample).unwrap();
            bytes.push(b'\n');
            let file = format!("shard-{index:06}.jsonl");
            fs::write(directory.join(&file), &bytes).unwrap();
            DatasetShard {
                file,
                sha256: sha(&bytes),
                bytes: bytes.len() as u64,
                samples: 1,
                game_id: sample.game_id.clone(),
                family_id: sample.family_id.clone(),
                split: games[index].split,
            }
        })
        .collect();
    let mut manifest = DatasetManifest {
        schema: 1,
        feature_schema: FEATURE_SCHEMA,
        feature_count: FEATURE_COUNT,
        rules_version: replay::RULES_VERSION,
        rules_baseline: replay::RULES_BASELINE.into(),
        catalog_hash: replay::catalog_hash(),
        move_schema: MOVE_SCHEMA,
        observation_schema: OBSERVATION_SCHEMA,
        source_kind: "selfPlay".into(),
        fingerprint: String::new(),
        samples: 2,
        games,
        shards,
        strata: vec![DatasetStratum {
            players: 2,
            options,
            policy_version: "synthetic-ml-test-v1".into(),
            games: 2,
            samples: 2,
        }],
    };
    manifest.fingerprint = sha(&serde_json::to_vec(&manifest).unwrap());
    fs::write(
        directory.join("manifest.json"),
        serde_json::to_vec(&manifest).unwrap(),
    )
    .unwrap();
}

#[test]
fn production_trainer_streams_validated_partitioned_shards_and_checks_later_changes() {
    let temporary = Temporary::new();
    streaming_fixture(&temporary.0);
    let dataset = load_dataset(&temporary.0).unwrap();
    let outcome = train_dataset(&dataset, &config(3), None).unwrap();
    assert_eq!(outcome.metrics.train_samples, 1);
    assert_eq!(outcome.metrics.validation_samples, 1);
    assert_eq!(
        outcome.checkpoint.dataset_fingerprint,
        dataset.manifest().fingerprint
    );
    let mut bytes = fs::read(temporary.0.join("shard-000000.jsonl")).unwrap();
    let index = bytes.iter().position(|byte| *byte == b'2').unwrap();
    bytes[index] = b'3';
    fs::write(temporary.0.join("shard-000000.jsonl"), bytes).unwrap();
    assert!(train_dataset(&dataset, &config(4), Some(&outcome.checkpoint)).is_err());
}
