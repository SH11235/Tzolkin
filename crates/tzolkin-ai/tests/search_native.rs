use tzolkin_ai::{arena, dataset, replay, search, search_native};
use tzolkin_core::GameOptions;

fn config() -> search::SearchConfig {
    search::SearchConfig {
        worlds_per_action: 1,
        min_completed_worlds: 1,
        horizon_days: 1,
        ..search::SearchConfig::default()
    }
}
fn record(players: usize, config: &search::SearchConfig) -> replay::GameReplay {
    search_native::play_game(
        players,
        11235,
        GameOptions::default(),
        &(0..players).collect::<Vec<_>>(),
        true,
        true,
        &search::PreparedSearch::new(config).unwrap(),
    )
    .unwrap()
    .game
    .unwrap()
    .2
    .unwrap()
}
fn temporary(label: &str) -> std::path::PathBuf {
    std::env::temp_dir().join(format!(
        "tzolkin-search-{label}-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ))
}

#[test]
fn native_three_four_player_all_seats_match_direct_search_and_reference_replay_bytes() {
    let config = config();
    let policy = search::PreparedSearch::new(&config).unwrap();
    for players in [3, 4] {
        let seats = (0..players).collect::<Vec<_>>();
        let run = search_native::play_game(
            players,
            11235,
            GameOptions::default(),
            &seats,
            true,
            true,
            &policy,
        )
        .unwrap();
        let summaries = run.search;
        let (_, _, game) = run.game.unwrap();
        let game = game.unwrap();
        replay::verify_replay(&game).unwrap();
        let mut direct_counts = vec![0; players];
        for step in &game.steps {
            let direct = search::choose_move(&step.observation, &config).unwrap();
            assert_eq!(direct.decision.r#move, step.chosen.r#move);
            assert_eq!(
                step.search.as_ref().unwrap(),
                &search_native::SearchTrace::from_outcome(&direct)
            );
            direct_counts[step.actor] += 1;
        }
        for (seat, summary) in summaries.iter().enumerate() {
            let summary = summary.as_ref().unwrap();
            assert_eq!(summary.decisions, direct_counts[seat]);
            assert!(summary.searched > 10);
            assert!(
                summary
                    .fallback_reasons
                    .contains_key(&search::FallbackReason::Setup)
            );
            assert_eq!(summary.searched + summary.fallbacks, summary.decisions);
        }
        let reference = search_native::play_game(
            players,
            11235,
            GameOptions::default(),
            &seats,
            true,
            false,
            &policy,
        )
        .unwrap();
        assert_eq!(reference.search, summaries);
        assert_eq!(
            serde_json::to_vec(&reference.game.unwrap().2.unwrap()).unwrap(),
            serde_json::to_vec(&game).unwrap()
        );
        let trace = serde_json::to_value(game.steps[0].search.as_ref().unwrap()).unwrap();
        assert!(
            trace.get("candidates").is_none()
                && trace.get("worlds").is_none()
                && trace.get("deck").is_none()
        );
    }
}

#[test]
fn existing_replay_bytes_omit_search_and_diagnostic_adapter_preserves_native_moves() {
    let plain = replay::play_game_fast(3, 7, GameOptions::default(), true)
        .unwrap()
        .2
        .unwrap();
    let adapted = replay::play_game_using_diagnostics(
        3,
        7,
        GameOptions::default(),
        true,
        plain.header.source.clone(),
        true,
        |o| Ok((tzolkin_ai::choose_move(o)?, None)),
    )
    .unwrap()
    .2
    .unwrap();
    assert_eq!(
        serde_json::to_vec(&plain).unwrap(),
        serde_json::to_vec(&adapted).unwrap()
    );
    assert!(
        plain
            .steps
            .iter()
            .all(|step| serde_json::to_value(step).unwrap().get("search").is_none())
    );
    replay::verify_replay(&adapted).unwrap();
}

#[test]
fn search_provenance_trace_config_phase_and_bounds_are_verified_fail_closed() {
    let original = record(3, &config());
    let playing = original
        .steps
        .iter()
        .position(|step| step.search.as_ref().unwrap().status == search::SearchStatus::Searched)
        .unwrap();
    let mut invalid = Vec::new();
    let mut bad = original.clone();
    if let replay::ReplaySource::PolicySelfPlay { policies } = &mut bad.header.source
        && let replay::SeatPolicy::Search {
            configuration_key, ..
        } = &mut policies[0]
    {
        *configuration_key = "0".repeat(64);
    }
    invalid.push(bad);
    let mut bad = original.clone();
    if let replay::ReplaySource::PolicySelfPlay { policies } = &mut bad.header.source
        && let replay::SeatPolicy::Search { config, .. } = &mut policies[0]
    {
        config.sampling_salt += 1;
    }
    invalid.push(bad);
    let mut bad = original.clone();
    if let replay::ReplaySource::PolicySelfPlay { policies } = &mut bad.header.source
        && let replay::SeatPolicy::Search { policy_version, .. } = &mut policies[0]
    {
        *policy_version = tzolkin_ai::POLICY_VERSION.into();
    }
    invalid.push(bad);
    let mut bad = original.clone();
    bad.steps[0].search = None;
    invalid.push(bad);
    let mut bad = original.clone();
    bad.steps[0].search.as_mut().unwrap().status = search::SearchStatus::Fallback {
        reason: search::FallbackReason::InsufficientCompletedWorlds,
    };
    invalid.push(bad);
    let mut bad = original.clone();
    bad.steps[playing].search.as_mut().unwrap().status = search::SearchStatus::Fallback {
        reason: search::FallbackReason::Setup,
    };
    invalid.push(bad);
    let mut bad = original.clone();
    bad.steps[playing]
        .search
        .as_mut()
        .unwrap()
        .stats
        .atomic_steps = config().max_total_steps + 1;
    invalid.push(bad);
    let mut bad = original.clone();
    bad.steps[playing]
        .search
        .as_mut()
        .unwrap()
        .stats
        .completed_worlds += 1;
    invalid.push(bad);
    let mut bad = original.clone();
    bad.steps[playing].search.as_mut().unwrap().score_kind =
        search::DecisionScoreKind::HeuristicActionPriority;
    invalid.push(bad);
    let mut bad = original.clone();
    bad.steps[0].search.as_mut().unwrap().score = f64::NAN;
    invalid.push(bad);
    let mut bad = original.clone();
    if let replay::ReplaySource::PolicySelfPlay { policies } = &mut bad.header.source {
        policies[bad.steps[0].actor] = tzolkin_ai::experiment::heuristic_seat();
    }
    invalid.push(bad);
    let mut bad = original.clone();
    bad.header.options.additional_buildings = true;
    invalid.push(bad);
    let mut bad = original.clone();
    bad.verified_complete = false;
    invalid.push(bad);
    for (index, bad) in invalid.iter().enumerate() {
        assert!(replay::verify_replay(bad).is_err(), "mutation {index}");
    }
    let mut unknown = serde_json::to_value(&original).unwrap();
    unknown["steps"][0]["search"]["hiddenDeck"] = serde_json::json!([]);
    assert!(serde_json::from_value::<replay::GameReplay>(unknown).is_err());
}

#[test]
fn budget_fallback_keeps_search_source_and_distinct_dataset_strata_and_seed_family() {
    let capped = search::SearchConfig {
        worlds_per_action: 3,
        min_completed_worlds: 2,
        max_rollout_steps: 1,
        max_total_steps: 1,
        ..config()
    };
    let run = search_native::play_game(
        3,
        11235,
        GameOptions::default(),
        &[0, 1, 2],
        true,
        true,
        &search::PreparedSearch::new(&capped).unwrap(),
    )
    .unwrap();
    for summary in &run.search {
        let summary = summary.as_ref().unwrap();
        assert_eq!(summary.searched, 0);
        assert!(summary.atomic_steps > 0 && summary.discarded_worlds > 0);
        assert!(
            summary
                .fallback_reasons
                .contains_key(&search::FallbackReason::InsufficientCompletedWorlds)
        );
    }
    let capped_record = run.game.unwrap().2.unwrap();
    replay::verify_replay(&capped_record).unwrap();
    let normal = record(3, &config());
    let destination = temporary("dataset");
    let manifest = dataset::export_dataset(&[capped_record.clone(), normal], &destination).unwrap();
    assert_eq!(manifest.strata.len(), 2);
    assert_eq!(manifest.games[0].family_id, manifest.games[1].family_id);
    assert_eq!(manifest.games[0].split, manifest.games[1].split);
    assert_ne!(
        manifest.games[0].policy_version,
        manifest.games[1].policy_version
    );
    assert!(
        manifest
            .games
            .iter()
            .all(|game| game.policy_version.starts_with("policy-selfplay-v1:"))
    );
    let samples = dataset::load_dataset(&destination)
        .unwrap()
        .iter()
        .try_fold(0usize, |count, sample| sample.map(|_| count + 1))
        .unwrap();
    assert_eq!(samples, manifest.samples);
    let mut human = capped_record;
    human.header.source = replay::ReplaySource::Human {
        provider: "synthetic".into(),
        reference: "test".into(),
        skill_rating: None,
    };
    assert!(dataset::export_dataset(&[human], &temporary("human")).is_err());
    std::fs::remove_dir_all(destination).unwrap();
}

#[test]
fn attempted_first_apply_failure_can_have_no_reached_round_in_a_compact_trace() {
    // This is a trace-contract fixture, not a claim that the normal native
    // engine actually fails an authoritative first legal operation.
    let cfg = search::SearchConfig {
        worlds_per_action: 2,
        min_completed_worlds: 2,
        max_rollout_steps: 1,
        max_total_steps: 1,
        ..config()
    };
    let mut game = record(3, &cfg);
    let index = game
        .steps
        .iter()
        .position(|step| {
            step.observation.phase == tzolkin_core::Phase::Playing
                && step.observation.pending_task.is_none()
        })
        .unwrap();
    let trace = game.steps[index].search.as_mut().unwrap();
    trace.stats = search::SearchStats {
        attempted_worlds: 1,
        atomic_steps: 1,
        discarded_worlds: vec![search::DiscardedWorld {
            world_index: 0,
            failed_candidate: 0,
            reason: search::DiscardReason::SimulationError,
        }],
        ..search::SearchStats::default()
    };
    let bytes = serde_json::to_vec(&game).unwrap();
    let parsed: replay::GameReplay = serde_json::from_slice(&bytes).unwrap();
    assert!(
        parsed.steps[index]
            .search
            .as_ref()
            .unwrap()
            .stats
            .max_reached_round
            .is_none()
    );
    replay::verify_replay(&parsed).unwrap();
    game.steps[index]
        .search
        .as_mut()
        .unwrap()
        .stats
        .max_reached_round = Some(game.steps[index].observation.round);
    replay::verify_replay(&game).unwrap();
    // Successful scored rollouts necessarily reached a round.
    let mut successful = record(3, &config());
    let trace = successful
        .steps
        .iter_mut()
        .find(|step| step.search.as_ref().unwrap().status == search::SearchStatus::Searched)
        .unwrap()
        .search
        .as_mut()
        .unwrap();
    trace.stats.max_reached_round = None;
    assert!(replay::verify_replay(&successful).is_err());
}

#[test]
fn native_rejects_nonbase_rules_player_counts_and_invalid_seats() {
    let policy = search::PreparedSearch::new(&config()).unwrap();
    for players in [2, 5] {
        assert!(
            search_native::play_game(
                players,
                0,
                GameOptions::default(),
                &[0],
                false,
                true,
                &policy
            )
            .is_err()
        );
    }
    for seats in [vec![], vec![3], vec![0, 0]] {
        assert!(
            search_native::play_game(3, 0, GameOptions::default(), &seats, false, true, &policy)
                .is_err()
        );
    }
    assert!(
        search_native::play_game(
            3,
            0,
            replay::options_from_mask(8),
            &[0],
            false,
            true,
            &policy
        )
        .is_err()
    );
}

#[test]
fn arena_search_rotates_every_seat_counts_opponents_and_matches_native_policy_path() {
    for players in [3, 4] {
        let seed = arena::partition_seeds(arena::Partition::Pilot, 0, 1).unwrap()[0];
        let search = arena::PolicyConfig::Search { config: config() };
        let mut opponents = vec![arena::PolicyConfig::default(); players];
        opponents[players - 1] = search.clone();
        let cfg = arena::ArenaConfig {
            schema: arena::ARENA_SCHEMA,
            players,
            partition: arena::Partition::Pilot,
            seeds: vec![seed],
            candidate: search,
            reference: arena::PolicyConfig::default(),
            opponent_pool: opponents,
            bootstrap_seed: 1,
        };
        let report = arena::run_arena(&cfg, std::path::Path::new(".")).unwrap();
        assert_eq!(report.statistics.failed_games, 0);
        assert_eq!(report.statistics.completed_games, players * 2);
        assert!(!report.statistics.strength_improvement_declared);
        for pair in &report.blocks[0].pairs {
            let mut seats = vec![pair.seat];
            if pair.seat != players - 1 {
                seats.push(players - 1);
            }
            let native = search_native::play_game(
                players,
                seed,
                GameOptions::default(),
                &seats,
                false,
                true,
                &search::PreparedSearch::new(&config()).unwrap(),
            )
            .unwrap();
            assert_eq!(pair.candidate.search, native.search);
            assert_eq!(
                pair.candidate.final_scores,
                native.game.unwrap().0.final_scores
            );
            assert!(pair.candidate.search[pair.seat].as_ref().unwrap().searched > 0);
            if pair.seat != players - 1 {
                assert!(
                    pair.reference.search[players - 1]
                        .as_ref()
                        .unwrap()
                        .searched
                        > 0
                );
            }
        }
        for seat in 0..players {
            let expected: u64 = report.blocks[0]
                .pairs
                .iter()
                .flat_map(|pair| [&pair.candidate, &pair.reference])
                .filter_map(|arm| arm.search[seat].as_ref())
                .map(|s| s.atomic_steps)
                .sum();
            assert_eq!(report.search[seat].as_ref().unwrap().atomic_steps, expected);
        }
    }
}
