use serde_json::Value;
use sha2::{Digest, Sha256};
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
#[path = "support/temp_root.rs"]
mod test_temp_root;
use tzolkin_ai::arena::{ArenaConfig, Partition, PolicyConfig, partition_seeds, run_arena};
use tzolkin_ai::dataset::{self, DatasetSplit};
use tzolkin_ai::kernel::Kernel;
use tzolkin_ai::policy_dataset::{export_native_files, load_policy_dataset};
use tzolkin_ai::policy_training::{
    PolicyBcConfig, PolicyTrainingCheckpoint, train_dataset, validate_checkpoint_dataset,
};
use tzolkin_ai::public_model::{LoadedPublicPolicy, PublicPolicyArtifact};
use tzolkin_ai::public_native::{PreparedPublicPolicy, play_game};
use tzolkin_ai::replay::{self, ReplaySource, SeatPolicy};
use tzolkin_core::GameOptions;
use tzolkin_core::observation::observe;

struct Temp(PathBuf);
impl Temp {
    fn new() -> Self {
        let dir = test_temp_root::create("tzolkin-public-native").unwrap();
        Self(dir)
    }
}
impl Drop for Temp {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}
fn digest(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}
fn resign(checkpoint: &mut PolicyTrainingCheckpoint) {
    checkpoint.checksum.clear();
    checkpoint.checksum = digest(&serde_json::to_vec(checkpoint).unwrap());
}
fn native_cli(args: &[&str]) -> std::process::Output {
    Command::new(env!("CARGO_BIN_EXE_tzolkin-public-ml"))
        .args(args)
        .output()
        .unwrap()
}
fn path(path: &Path) -> &str {
    path.to_str().unwrap()
}

#[test]
#[ignore = "slow ML correctness integration; run npm run test:ml:slow"]
fn actual_trained_policy_completes_three_four_players_roundtrips_and_runs_qualified_arena() {
    let temp = Temp::new();
    let mut sources = Vec::new();
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
        let file = temp.0.join(format!("source-{}.json", sources.len()));
        fs::write(&file, serde_json::to_vec(&record).unwrap()).unwrap();
        sources.push(file);
    }
    let dataset_path = temp.0.join("dataset");
    export_native_files(&sources, &dataset_path).unwrap();
    let dataset = load_policy_dataset(&dataset_path).unwrap();
    let outcome = train_dataset(
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
    outcome.save_new_directory(&training).unwrap();
    let checkpoint_path = training.join("checkpoint.json");
    let checkpoint = PolicyTrainingCheckpoint::load(&checkpoint_path).unwrap();
    validate_checkpoint_dataset(&dataset, &checkpoint).unwrap();
    let prepared = PreparedPublicPolicy::new(checkpoint.clone(), &dataset, Kernel::Scalar).unwrap();
    let handle = prepared.handle().unwrap();
    let direct = LoadedPublicPolicy::new(prepared.model()).unwrap();
    let initialized =
        tzolkin_ai::public_rl_artifact::InitializedPublicRlPolicy::from_bc(&prepared).unwrap();
    initialized.artifact().validate().unwrap();
    assert!(
        initialized
            .model()
            .parameters()
            .iter()
            .zip(prepared.model().model.parameters())
            .all(|(a, b)| a.to_bits() == b.to_bits())
    );
    assert_eq!(initialized.artifact().bc_source(), handle.provenance());
    assert_eq!(initialized.artifact().family_closure().len(), 2);
    assert_eq!(prepared.backend(), "scalar");
    assert_eq!(handle.backend(), "scalar");
    let deployment_bytes = tzolkin_ai::public_deployment::export_prepared_bc(&prepared).unwrap();
    let deployed =
        tzolkin_inference::deployment::LoadedDeployment::load(&deployment_bytes).unwrap();
    assert_eq!(deployed.source_role(), "bcPrepared");
    assert_eq!(deployed.source_checksum(), prepared.model().checksum);
    assert_eq!(deployed.update_count(), None);
    let state =
        tzolkin_core::create_game(vec!["A".into(), "B".into(), "C".into()], 11235, false).unwrap();
    let observation = observe(&state, state.current_player).unwrap();
    assert_eq!(
        handle.choose_move(&observation).unwrap(),
        direct.choose_move(&observation).unwrap()
    );
    let deployed_setup = deployed.choose_move(&observation).unwrap();
    let original_setup = direct.choose_move(&observation).unwrap();
    assert_eq!(deployed_setup.actor, original_setup.actor);
    assert_eq!(
        deployed_setup.observation_key,
        original_setup.observation_key
    );
    assert_eq!(deployed_setup.r#move, original_setup.r#move);
    assert_eq!(
        deployed_setup.score.to_bits(),
        original_setup.score.to_bits()
    );
    let provenance = serde_json::to_value(handle.provenance()).unwrap();
    assert_eq!(provenance["kind"], "publicLearned");
    assert_eq!(provenance["featureSchema"], 2);
    assert_eq!(provenance["inferenceBackend"], "scalar");
    assert_eq!(
        provenance["trainingCheckpointChecksum"],
        checkpoint.checksum
    );
    assert_eq!(
        provenance["datasetFingerprint"],
        dataset.manifest().fingerprint
    );
    assert_eq!(provenance["valueValidity"], "unavailablePolicyOnly");
    let mut completed_records = Vec::new();
    for (players, seats) in [(3, vec![0, 1, 2]), (4, vec![0, 2])] {
        let (state, decisions, record) = play_game(
            &handle,
            players,
            11235,
            GameOptions::default(),
            &seats,
            true,
        )
        .unwrap();
        let record = record.expect("requested native record was missing");
        assert_eq!(record.steps.len(), decisions);
        assert_eq!(state, replay::verify_replay(&record).unwrap());
        let ReplaySource::PolicySelfPlay { policies } = &record.header.source else {
            panic!()
        };
        for (seat, policy) in policies.iter().enumerate() {
            assert_eq!(
                matches!(policy, SeatPolicy::PublicLearned { .. }),
                seats.contains(&seat)
            );
        }
        for step in record
            .steps
            .iter()
            .filter(|step| seats.contains(&step.actor))
            .take(8)
        {
            assert_eq!(
                handle.choose_move(&step.observation).unwrap().r#move,
                step.chosen.r#move
            );
            assert_eq!(
                handle.choose_move(&step.observation).unwrap(),
                direct.choose_move(&step.observation).unwrap()
            );
            let old = direct.distribution(&step.observation).unwrap();
            let new = deployed.distribution(&step.observation).unwrap();
            assert_eq!(
                new.logits
                    .iter()
                    .map(|value| value.to_bits())
                    .collect::<Vec<_>>(),
                old.logits
                    .iter()
                    .map(|value| value.to_bits())
                    .collect::<Vec<_>>()
            );
            assert_eq!(
                new.probabilities
                    .iter()
                    .map(|value| value.to_bits())
                    .collect::<Vec<_>>(),
                old.probabilities
                    .iter()
                    .map(|value| value.to_bits())
                    .collect::<Vec<_>>()
            );
            assert_eq!(
                deployed.choose_move(&step.observation).unwrap().r#move,
                step.chosen.r#move
            );
        }
        completed_records.push(record);
    }
    assert_eq!(completed_records.len(), 2);
    let paths = completed_records
        .iter()
        .enumerate()
        .map(|(i, record)| {
            let file = temp.0.join(format!("learned-{i}.json"));
            replay::save_replay_new(&file, record).unwrap();
            file
        })
        .collect::<Vec<_>>();
    let roundtrip_path = temp.0.join("roundtrip");
    let manifest = export_native_files(&paths, &roundtrip_path).unwrap();
    let roundtrip = load_policy_dataset(&roundtrip_path).unwrap();
    assert_eq!(
        roundtrip.iter().map(Result::unwrap).count(),
        manifest.samples
    );
    for sample in roundtrip.iter() {
        assert_eq!(sample.unwrap().value_target(), None);
    }
    assert_eq!(manifest.games.len(), 2);
    for record in &completed_records {
        let mut partial = record.clone();
        partial.verified_complete = false;
        assert!(
            replay::verify_replay(&partial)
                .unwrap_err()
                .contains("Partial")
        );
        for option in 0..4 {
            let mut unsupported = record.clone();
            match option {
                0 => unsupported.header.options.additional_buildings = true,
                1 => unsupported.header.options.tribes = true,
                2 => unsupported.header.options.prophecies = true,
                _ => unsupported.header.options.quick_actions = true,
            }
            assert!(
                replay::verify_replay(&unsupported)
                    .unwrap_err()
                    .contains("Public learned provenance")
            );
        }
    }

    let policy = PolicyConfig::PublicLearned {
        checkpoint: checkpoint_path.clone(),
        dataset: dataset_path.clone(),
        kernel: "scalar".into(),
    };
    let config = ArenaConfig {
        schema: 1,
        players: 3,
        partition: Partition::Test,
        seeds: partition_seeds(Partition::Test, 100, 1).unwrap(),
        candidate: policy.clone(),
        reference: policy.clone(),
        opponent_pool: vec![
            PolicyConfig::default(),
            policy.clone(),
            PolicyConfig::default(),
        ],
        bootstrap_seed: 7,
    };
    let report = run_arena(&config, &temp.0).unwrap();
    assert_eq!(report.statistics.planned_games, 6);
    assert_eq!(
        report.statistics.completed_games + report.statistics.failed_games,
        6
    );
    assert!(!report.statistics.strength_improvement_declared);
    assert_eq!(report.statistics.completed_games, 6);
    assert_eq!(report.statistics.failed_games, 0);
    assert!(report.blocks[0].complete);
    for pair in &report.blocks[0].pairs {
        assert_eq!(pair.candidate.final_scores, pair.reference.final_scores);
        assert_eq!(pair.candidate.error, pair.reference.error);
        if pair.candidate.error.is_some() {
            assert!(pair.candidate.decisions.is_none());
            assert!(pair.candidate.winner_utility.is_none());
            assert!(pair.candidate.score.is_none());
            assert!(pair.candidate.terminal_players.is_none());
        }
    }
    assert_eq!(report.statistics.mean_utility_delta, Some(0.0));
    for described in [
        &report.candidate,
        &report.reference,
        &report.opponent_pool[1],
    ] {
        assert!(matches!(
            described.provenance,
            SeatPolicy::PublicLearned { .. }
        ));
        assert_eq!(
            described.dataset_fingerprint.as_deref(),
            Some(dataset.manifest().fingerprint.as_str())
        );
    }
    for location in 0..3 {
        let mut rejected = config.clone();
        rejected.partition = Partition::Pilot;
        rejected.seeds = partition_seeds(Partition::Pilot, 100, 1).unwrap();
        rejected.candidate = PolicyConfig::default();
        rejected.reference = PolicyConfig::default();
        rejected.opponent_pool = vec![PolicyConfig::default(); 3];
        match location {
            0 => rejected.candidate = policy.clone(),
            1 => rejected.reference = policy.clone(),
            _ => rejected.opponent_pool[2] = policy.clone(),
        };
        assert!(
            run_arena(&rejected, &temp.0)
                .unwrap_err()
                .contains("held-out test")
        );
    }
    // Development deliberately permits a family already used for Validation metrics.
    let validation_seed = (0..1000)
        .find(|seed| {
            dataset::split_for_family(&dataset::seed_family_id(*seed)).unwrap()
                == DatasetSplit::Validation
        })
        .unwrap();
    assert!(dataset.manifest().games.iter().any(|game| {
        game.split == DatasetSplit::Validation
            && game.family_id == dataset::seed_family_id(validation_seed)
    }));
    for guard in [
        None,
        Some(tzolkin_ai::public_trade_guard::TradeGuardConfig::default()),
    ] {
        let development_policy = PolicyConfig::PublicLearnedDevelopment {
            checkpoint: checkpoint_path.clone(),
            dataset: dataset_path.clone(),
            kernel: "scalar".into(),
            guard: guard.clone(),
        };
        let mut development = config.clone();
        development.partition = Partition::Validation;
        development.seeds = vec![validation_seed];
        development.candidate = development_policy.clone();
        development.reference = development_policy.clone();
        development.opponent_pool[1] = development_policy.clone();
        let development_report = run_arena(&development, &temp.0).unwrap();
        assert_eq!(development_report.partition, Partition::Validation);
        assert_ne!(development_report.config_sha256, report.config_sha256);
        assert_eq!(development_report.statistics.planned_games, 6);
        assert_eq!(
            development_report.statistics.completed_games
                + development_report.statistics.failed_games,
            6
        );
        assert!(!development_report.statistics.strength_improvement_declared);
        for described in [
            &development_report.candidate,
            &development_report.reference,
            &development_report.opponent_pool[1],
        ] {
            assert_eq!(
                described.dataset_fingerprint.as_deref(),
                Some(dataset.manifest().fingerprint.as_str())
            );
            assert!(matches!(
                (&guard, &described.provenance),
                (None, SeatPolicy::PublicLearned { .. })
                    | (Some(_), SeatPolicy::PublicLearnedTradeGuard { .. })
            ));
        }
        // A policy loop remains a failed arm, with unavailable terminal results.
        for pair in &development_report.blocks[0].pairs {
            assert_eq!(pair.candidate.error, pair.reference.error);
            assert_eq!(pair.candidate.final_scores, pair.reference.final_scores);
            for arm in [&pair.candidate, &pair.reference] {
                if arm.error.is_some() {
                    assert!(arm.decisions.is_none());
                    assert!(arm.winner_utility.is_none());
                    assert!(arm.score.is_none());
                    assert!(arm.rank.is_none());
                    assert!(arm.final_scores.is_empty());
                    assert!(arm.terminal_players.is_none());
                }
            }
        }
        for partition in [Partition::Pilot, Partition::Test] {
            for location in 0..5 {
                let mut rejected = config.clone();
                rejected.partition = partition;
                rejected.seeds = partition_seeds(partition, 100, 1).unwrap();
                rejected.candidate = PolicyConfig::default();
                rejected.reference = PolicyConfig::default();
                rejected.opponent_pool = vec![PolicyConfig::default(); 3];
                match location {
                    0 => rejected.candidate = development_policy.clone(),
                    1 => rejected.reference = development_policy.clone(),
                    other => rejected.opponent_pool[other - 2] = development_policy.clone(),
                }
                assert!(
                    run_arena(&rejected, &temp.0)
                        .unwrap_err()
                        .contains("require the validation partition")
                );
            }
        }
    }
    let mut malformed = checkpoint.clone();
    malformed.metrics.final_train.policy_loss += 1.0;
    resign(&mut malformed);
    malformed.validate().unwrap();
    assert!(PreparedPublicPolicy::new(malformed, &dataset, Kernel::Scalar).is_err());
    let mut mismatch = checkpoint.clone();
    mismatch.dataset.fingerprint = "0".repeat(64);
    resign(&mut mismatch);
    assert!(PreparedPublicPolicy::new(mismatch, &dataset, Kernel::Scalar).is_err());
    assert!(
        PreparedPublicPolicy::load(&training.join("model.json"), &dataset_path, Kernel::Scalar)
            .is_err()
    );
    let source = dataset_path.join(&dataset.manifest().games[0].source_file);
    let bytes = fs::read(&source).unwrap();
    fs::write(&source, b"{}").unwrap();
    // An already-prepared immutable model remains usable; a fresh preparation re-audits disk.
    assert_eq!(
        handle.choose_move(&observation).unwrap(),
        direct.choose_move(&observation).unwrap()
    );
    assert!(PreparedPublicPolicy::new(checkpoint.clone(), &dataset, Kernel::Scalar).is_err());
    fs::write(&source, bytes).unwrap();
    let mut invalid_observation = observation.clone();
    invalid_observation.legal_actions.clear();
    assert!(handle.choose_move(&invalid_observation).is_err());
    let accelerated =
        PreparedPublicPolicy::new(checkpoint.clone(), &dataset, Kernel::Auto).unwrap();
    let accelerated_handle = accelerated.handle().unwrap();
    let accelerated_direct =
        LoadedPublicPolicy::with_kernel(accelerated.model(), Kernel::Auto).unwrap();
    assert_eq!(
        accelerated.backend(),
        Kernel::Auto.resolve().unwrap().backend()
    );
    assert_eq!(
        accelerated_handle.choose_move(&observation).unwrap(),
        accelerated_direct.choose_move(&observation).unwrap()
    );
    assert_eq!(
        serde_json::to_value(accelerated_handle.provenance()).unwrap()["inferenceBackend"],
        accelerated.backend()
    );
    for (players, options, seats) in [
        (2, GameOptions::default(), vec![0]),
        (5, GameOptions::default(), vec![0]),
        (
            3,
            GameOptions {
                tribes: true,
                ..Default::default()
            },
            vec![0],
        ),
        (3, GameOptions::default(), vec![3]),
        (3, GameOptions::default(), vec![0, 0]),
        (3, GameOptions::default(), vec![]),
    ] {
        assert!(play_game(&handle, players, 0, options, &seats, false).is_err());
    }

    let output_path = temp.0.join("cli-replay.json");
    let output = native_cli(&[
        "selfplay",
        "--checkpoint",
        path(&checkpoint_path),
        "--dataset",
        path(&dataset_path),
        "--seats",
        "0",
        "--players",
        "3",
        "--seed",
        "11235",
        "--output",
        path(&output_path),
    ]);
    assert!(
        !output.stdout.is_empty(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let summary: Value = serde_json::from_slice(&output.stdout).unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert_eq!(summary["complete"], true);
    assert_eq!(summary["success"], true);
    let record: replay::GameReplay =
        serde_json::from_slice(&fs::read(&output_path).unwrap()).unwrap();
    replay::verify_replay(&record).unwrap();
    let before = fs::read(&output_path).unwrap();
    let failure = native_cli(&[
        "selfplay",
        "--checkpoint",
        "missing",
        "--dataset",
        "missing",
        "--seats",
        "all",
        "--output",
        path(&output_path),
    ]);
    assert!(!failure.status.success());
    assert!(failure.stdout.is_empty());
    assert_eq!(fs::read(&output_path).unwrap(), before);
}

#[test]
fn public_provenance_is_closed_and_legacy_variants_keep_their_exact_fields() {
    let base = serde_json::json!({"kind":"publicLearned", "policyVersion":"learned-public-policy-v1", "modelVersion":"tiny-public-policy-mlp-v1",
        "trainingVersion":"public-policy-bc-scalar-adam-v1", "featureSchema":2, "inputContract":"base-3-4p-native-setup-public-playing-v2",
        "task":"policyOnlyBc", "valueValidity":"unavailablePolicyOnly", "modelChecksum":"a".repeat(64),
        "trainingCheckpointChecksum":"b".repeat(64), "datasetFingerprint":"c".repeat(64), "inferenceBackend":"scalar"});
    serde_json::from_value::<SeatPolicy>(base.clone())
        .unwrap()
        .validate()
        .unwrap();
    for (field, value) in [
        ("policyVersion", "learned-policy-v1".into()),
        ("featureSchema", 1.into()),
        ("task", "value".into()),
        ("modelChecksum", "A".repeat(64).into()),
        ("trainingCheckpointChecksum", "short".into()),
        ("datasetFingerprint", "0".repeat(63).into()),
        ("inferenceBackend", "auto".into()),
    ] {
        let mut invalid = base.clone();
        invalid[field] = value;
        assert!(
            serde_json::from_value::<SeatPolicy>(invalid)
                .unwrap()
                .validate()
                .is_err()
        );
    }
    for field in [
        "trainingCheckpointChecksum",
        "featureSchema",
        "inferenceBackend",
        "valueValidity",
    ] {
        let mut missing = base.clone();
        missing.as_object_mut().unwrap().remove(field);
        assert!(serde_json::from_value::<SeatPolicy>(missing).is_err());
    }
    let legacy = serde_json::json!({"kind":"learned","policyVersion":tzolkin_ai::model::LEARNED_POLICY_VERSION,"modelChecksum":"a".repeat(64)});
    let legacy_policy = serde_json::from_value::<SeatPolicy>(legacy.clone()).unwrap();
    legacy_policy.validate().unwrap();
    assert_eq!(serde_json::to_value(&legacy_policy).unwrap(), legacy);
    let mut invalid = base;
    invalid["qualified"] = true.into();
    assert!(serde_json::from_value::<SeatPolicy>(invalid).is_err());
}

#[test]
fn selfplay_cli_rejects_flags_limits_and_untrained_or_legacy_inputs_before_running() {
    for args in [
        vec![
            "selfplay",
            "--checkpoint",
            "x",
            "--dataset",
            "y",
            "--seats",
            "all",
            "--players",
            "2",
        ],
        vec![
            "selfplay",
            "--checkpoint",
            "x",
            "--dataset",
            "y",
            "--seats",
            "all",
            "--flags",
            "1",
        ],
        vec![
            "selfplay",
            "--checkpoint",
            "x",
            "--dataset",
            "y",
            "--seats",
            "0,0",
        ],
        vec![
            "selfplay",
            "--checkpoint",
            "x",
            "--dataset",
            "y",
            "--seats",
            "0",
            "--seed",
            "4294967296",
        ],
        vec!["selfplay", "--checkpoint", "x", "--dataset", "y"],
        vec!["selfplay", "--model", "x"],
        vec!["selfplay", "--checkpoint", "x", "--checkpoint", "z"],
        vec!["selfplay", "--fast"],
        vec![
            "selfplay",
            "--checkpoint",
            "x",
            "--dataset",
            "y",
            "--seats",
            "all",
            "--kernel",
            "unknown",
        ],
    ] {
        let result = native_cli(&args);
        assert!(!result.status.success());
        assert!(!result.stderr.is_empty());
        assert!(result.stdout.is_empty());
    }
    let temp = Temp::new();
    let artifact = temp.0.join("untrained.json");
    PublicPolicyArtifact::new(7)
        .unwrap()
        .save_new(&artifact)
        .unwrap();
    let result = native_cli(&[
        "selfplay",
        "--checkpoint",
        path(&artifact),
        "--dataset",
        "missing",
        "--seats",
        "all",
    ]);
    assert!(!result.status.success());
    assert!(result.stdout.is_empty());
    let help = String::from_utf8(native_cli(&["--help"]).stdout).unwrap();
    assert!(help.contains("selfplay --checkpoint"));
    assert!(help.contains("explicit --seats"));
}
