use std::path::{Path, PathBuf};
use std::process::Command;
use tzolkin_ai::arena::{
    ArenaConfig, Partition, PolicyConfig, family_id, partition_seeds, run_arena, seed_partition,
};
use tzolkin_ai::{
    choose_move, choose_move_with_weights, dataset, experiment, policy::HeuristicWeights, replay,
    training::TrainingConfig,
};
use tzolkin_core::observation::{observation_key, observe};
use tzolkin_core::{GameOptions, Phase, apply_move, create_game};

fn config(players: usize, count: usize) -> ArenaConfig {
    ArenaConfig {
        schema: 1,
        players,
        partition: Partition::Pilot,
        seeds: partition_seeds(Partition::Pilot, 4000, count).unwrap(),
        candidate: PolicyConfig::default(),
        reference: PolicyConfig::default(),
        opponent_pool: vec![PolicyConfig::default(); players],
        bootstrap_seed: 11235,
    }
}
struct Temporary(PathBuf);
impl Temporary {
    fn new(name: &str) -> Self {
        let path =
            std::env::temp_dir().join(format!("tzolkin-arena-{name}-{}", std::process::id()));
        std::fs::create_dir(&path).unwrap();
        Self(path)
    }
}
impl Drop for Temporary {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}
fn cli(args: &[&str]) -> std::process::Output {
    Command::new(env!("CARGO_BIN_EXE_tzolkin-ai"))
        .args(args)
        .output()
        .unwrap()
}

#[test]
fn explicit_default_keeps_all_decisions_scores_keys_and_hidden_information_contract() {
    for players in [3, 4] {
        let mut state =
            create_game((0..players).map(|i| format!("P{i}")).collect(), 17, false).unwrap();
        for _ in 0..replay::MAX_DECISIONS {
            if state.phase == Phase::Finished {
                break;
            }
            let observation = observe(&state, state.current_player).unwrap();
            let default = choose_move(&observation).unwrap();
            assert_eq!(
                default,
                choose_move_with_weights(&observation, &HeuristicWeights::default()).unwrap()
            );
            let mut hidden = state.clone();
            hidden.seed = 999999;
            hidden.building_deck.reverse();
            hidden.age2_deck.reverse();
            for player in &mut hidden.players {
                if player.id != observation.actor {
                    player.wealth_offer.reverse();
                }
            }
            assert_eq!(
                observation,
                observe(&hidden, hidden.current_player).unwrap()
            );
            state = apply_move(&state, default.r#move).unwrap();
        }
        assert_eq!(state.phase, Phase::Finished);
    }
    let state = create_game(vec!["A".into(), "B".into(), "C".into()], 0, false).unwrap();
    let mut observation = observe(&state, 0).unwrap();
    let mut invalid = HeuristicWeights {
        worker: f64::NAN,
        ..HeuristicWeights::default()
    };
    assert!(choose_move_with_weights(&observation, &invalid).is_err());
    invalid.worker = 1001.0;
    assert!(invalid.validate().is_err());
    invalid.worker = -1000.0;
    assert!(invalid.validate().is_ok());
    observation.round = 0;
    observation.observation_key = observation_key(&observation).unwrap();
    assert!(choose_move_with_weights(&observation, &HeuristicWeights::default()).is_err());
}

#[test]
fn identical_candidate_and_reference_have_exact_zero_deltas_for_every_seat_in_frozen_mixed_pool() {
    for players in [3, 4] {
        let mut config = config(players, 2);
        let varied = HeuristicWeights {
            worker: 3.0,
            technology_step: 8.0,
            ..HeuristicWeights::default()
        };
        config.opponent_pool[1] = PolicyConfig::Heuristic {
            weights: varied.clone(),
        };
        let report = run_arena(&config, Path::new(".")).unwrap();
        assert_eq!(report.statistics.mean_utility_delta, Some(0.0));
        assert_eq!(report.statistics.mean_score_delta, Some(0.0));
        assert_eq!(report.statistics.mean_rank_improvement, Some(0.0));
        assert_eq!(
            report.statistics.utility_delta_bootstrap95,
            Some([0.0, 0.0])
        );
        assert_eq!(report.statistics.completed_games, players * 2 * 2);
        assert_eq!(report.statistics.failed_games, 0);
        assert!(!report.feeding_events_collected);
        assert!(!report.statistics.strength_improvement_declared);
        for block in &report.blocks {
            assert_eq!(block.family_id, family_id(block.seed));
            for pair in &block.pairs {
                assert_eq!(pair.candidate.final_scores, pair.reference.final_scores);
                assert_eq!(pair.candidate.decisions, pair.reference.decisions);
                assert_eq!(pair.candidate.start_of_play, pair.reference.start_of_play);
                assert!(pair.candidate.start_of_play.is_some());
                assert!(pair.candidate.unfed_workers.is_none());
            }
        }
        match &report.opponent_pool[1].provenance {
            replay::SeatPolicy::Heuristic { weights, .. } => assert_eq!(*weights, varied),
            _ => panic!("pool provenance changed"),
        }
    }
}

#[test]
fn partitions_limits_and_unknown_configuration_are_checked_before_games() {
    for (partition, split) in [
        (Partition::Pilot, dataset::DatasetSplit::Train),
        (Partition::Validation, dataset::DatasetSplit::Validation),
        (Partition::Test, dataset::DatasetSplit::Test),
    ] {
        let seeds = partition_seeds(partition, 0, 12).unwrap();
        for seed in seeds {
            assert_eq!(seed_partition(seed).unwrap(), split);
        }
    }
    assert!(partition_seeds(Partition::Pilot, 0, 0).is_err());
    assert!(partition_seeds(Partition::Pilot, 0, 1025).is_err());
    assert!(partition_seeds(Partition::Pilot, u32::MAX, 2).is_err());
    let mut bad = config(3, 1);
    bad.players = 2;
    assert!(bad.validate().is_err());
    let mut bad = config(3, 1);
    bad.seeds.push(bad.seeds[0]);
    assert!(bad.validate().is_err());
    let mut bad = config(3, 1);
    bad.partition = Partition::Test;
    assert!(bad.validate().is_err());
    let mut bad = config(3, 1);
    bad.opponent_pool.pop();
    assert!(bad.validate().is_err());
    let mut bad = config(3, 1);
    bad.candidate = PolicyConfig::Heuristic {
        weights: HeuristicWeights {
            worker: 1001.0,
            ..HeuristicWeights::default()
        },
    };
    assert!(run_arena(&bad, Path::new(".")).is_err());
    let mut value = serde_json::to_value(config(3, 1)).unwrap();
    value["opponentPool"][0]["kernel"] = serde_json::json!("auto");
    assert!(serde_json::from_value::<ArenaConfig>(value).is_err());
}

#[test]
fn weighted_selfplay_is_verifiable_and_dataset_source_is_content_bound() {
    let guard = Temporary::new("weighted");
    let weights = HeuristicWeights {
        worker: 3.0,
        technology_step: 8.0,
        ..HeuristicWeights::default()
    };
    let policies = vec![
        replay::SeatPolicy::Heuristic {
            policy_version: tzolkin_ai::POLICY_VERSION.into(),
            weights: weights.clone()
        };
        3
    ];
    let (state, _, record) = replay::play_game_using_fast(
        3,
        0,
        GameOptions::default(),
        true,
        replay::ReplaySource::PolicySelfPlay {
            policies: policies.clone(),
        },
        |o| choose_move_with_weights(o, &weights),
    )
    .unwrap();
    let record = record.unwrap();
    assert_eq!(
        replay::verify_replay(&record).unwrap().final_scores,
        state.final_scores
    );
    let path = guard.0.join("dataset");
    dataset::export_dataset(std::slice::from_ref(&record), &path).unwrap();
    let loaded = dataset::load_dataset(&path).unwrap();
    assert!(
        loaded.manifest().games[0]
            .policy_version
            .starts_with("policy-selfplay-v1:")
    );
    let mut corrupted = record;
    if let replay::ReplaySource::PolicySelfPlay { policies } = &mut corrupted.header.source
        && let replay::SeatPolicy::Heuristic { weights, .. } = &mut policies[0]
    {
        weights.worker = f64::INFINITY;
    }
    assert!(replay::verify_replay(&corrupted).is_err());
}

#[test]
fn cli_preserves_existing_reports_and_learned_arena_binds_dataset_and_partition() {
    let guard = Temporary::new("cli");
    let config_path = guard.0.join("control.json");
    let report_path = guard.0.join("report.json");
    std::fs::write(&config_path, serde_json::to_vec(&config(3, 1)).unwrap()).unwrap();
    let result = cli(&[
        "arena",
        "--config",
        config_path.to_str().unwrap(),
        "--output",
        report_path.to_str().unwrap(),
    ]);
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    let report: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&report_path).unwrap()).unwrap();
    assert_eq!(report["statistics"]["failedGames"], 0);
    assert_eq!(report["statistics"]["meanUtilityDelta"], 0.0);
    let before = std::fs::read(&report_path).unwrap();
    assert!(
        !cli(&[
            "arena",
            "--config",
            "MISSING.json",
            "--output",
            report_path.to_str().unwrap()
        ])
        .status
        .success()
    );
    assert_eq!(before, std::fs::read(&report_path).unwrap());
    for args in [
        vec!["arena", "--config"],
        vec!["arena", "--config", "x", "--seed", "1"],
        vec!["arena-seeds", "--count", "1", "--count", "2"],
        vec!["arena-seeds", "--partition", "invalid"],
    ] {
        assert!(!cli(&args).status.success());
    }
    let result = cli(&["arena-seeds", "--partition", "test", "--count", "2"]);
    assert!(result.status.success());
    let selected: serde_json::Value = serde_json::from_slice(&result.stdout).unwrap();
    assert_eq!(selected["seeds"].as_array().unwrap().len(), 2);
    let dataset_path = guard.0.join("dataset");
    let records = [0, 3].map(|seed| {
        replay::play_game_fast(2, seed, GameOptions::default(), true)
            .unwrap()
            .2
            .unwrap()
    });
    dataset::export_dataset(&records, &dataset_path).unwrap();
    let loaded = dataset::load_dataset(&dataset_path).unwrap();
    let trained = guard.0.join("trained");
    experiment::train_to_directory(
        &loaded,
        &TrainingConfig {
            epochs: 1,
            ..TrainingConfig::default()
        },
        None,
        &trained,
    )
    .unwrap();
    let mut learned = config(3, 1);
    learned.candidate = PolicyConfig::Learned {
        checkpoint: PathBuf::from("trained/checkpoint.json"),
        dataset: PathBuf::from("dataset"),
        kernel: "scalar".into(),
    };
    assert!(
        run_arena(&learned, &guard.0)
            .unwrap_err()
            .contains("held-out test")
    );
    learned.partition = Partition::Test;
    learned.seeds = partition_seeds(Partition::Test, 5000, 1).unwrap();
    let report = run_arena(&learned, &guard.0).unwrap();
    assert_eq!(report.statistics.completed_games, 6);
    assert_eq!(
        report.candidate.dataset_fingerprint.as_deref(),
        Some(loaded.manifest().fingerprint.as_str())
    );
    let mut bad = learned.clone();
    if let PolicyConfig::Learned { kernel, .. } = &mut bad.candidate {
        *kernel = "unsupported".into();
    }
    assert!(
        run_arena(&bad, &guard.0)
            .unwrap_err()
            .contains("Unknown arena")
    );
    let other = guard.0.join("other");
    dataset::export_dataset(&records[..1], &other).unwrap();
    if let PolicyConfig::Learned { dataset, .. } = &mut learned.candidate {
        *dataset = PathBuf::from("other");
    }
    assert!(
        run_arena(&learned, &guard.0)
            .unwrap_err()
            .contains("fingerprint mismatch")
    );
}
