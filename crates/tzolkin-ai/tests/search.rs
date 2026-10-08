use tzolkin_ai::{
    choose_move, replay,
    search::{
        self, DiscardReason, FallbackReason, SEARCH_POLICY_VERSION, SearchConfig, SearchStatus,
    },
};
use tzolkin_core::observation::{Observation, TypedAction, observation_key, observe};
use tzolkin_core::{GameState, Phase, create_game};

fn playing(n: usize, seed: u32) -> GameState {
    let mut state = create_game(
        (0..n).map(|id| format!("Person {id}")).collect(),
        seed,
        false,
    )
    .unwrap();
    while state.phase == Phase::Setup {
        let observation = observe(&state, state.current_player).unwrap();
        state =
            tzolkin_core::apply_move(&state, choose_move(&observation).unwrap().r#move).unwrap();
    }
    state
}
fn small_config() -> SearchConfig {
    SearchConfig {
        worlds_per_action: 1,
        horizon_days: 1,
        min_completed_worlds: 1,
        ..SearchConfig::default()
    }
}
fn assert_legal(o: &Observation, outcome: &search::SearchOutcome) {
    assert_eq!(outcome.decision.actor, o.actor);
    assert_eq!(outcome.decision.observation_key, o.observation_key);
    assert_eq!(outcome.decision.policy_version, SEARCH_POLICY_VERSION);
    assert!(outcome.decision.score.is_finite());
    assert!(
        o.legal_actions
            .iter()
            .any(|a| a.r#move == outcome.decision.r#move)
    );
}

#[test]
fn default_six_day_worlds_are_balanced_deterministic_and_hidden_independent() {
    for n in [3, 4] {
        let original = playing(n, 11235);
        let observation = observe(&original, original.current_player).unwrap();
        let config = SearchConfig::default();
        let outcome = search::choose_move(&observation, &config).unwrap();
        assert_legal(&observation, &outcome);
        assert_eq!(
            outcome.status,
            SearchStatus::Searched,
            "{n}p: {:?}",
            outcome.stats
        );
        assert!(outcome.stats.completed_worlds >= config.min_completed_worlds);
        assert!(outcome.stats.atomic_steps > o_min_steps(&observation));
        assert!(outcome.stats.atomic_steps <= config.max_total_steps);
        assert!(
            outcome
                .candidates
                .iter()
                .all(|c| c.visits == outcome.stats.completed_worlds
                    && c.mean_score.is_some()
                    && c.cutoff_samples + c.terminal_samples == c.visits
                    && c.minimum_scored_round.unwrap() >= observation.round + config.horizon_days)
        );
        assert_eq!(outcome, search::choose_move(&observation, &config).unwrap());
        let mut changed = original;
        changed.seed = 987654;
        changed.building_deck.reverse();
        changed.age2_deck.reverse();
        changed.log.push("private engine diagnostic".into());
        for p in &mut changed.players {
            p.name = "private presentation".into();
            p.color = "#abcdef".into();
            if p.id != observation.actor {
                p.wealth_offer.reverse();
            }
        }
        let equivalent = observe(&changed, changed.current_player).unwrap();
        assert_eq!(equivalent, observation);
        assert_eq!(outcome, search::choose_move(&equivalent, &config).unwrap());
        let text = serde_json::to_string(&outcome).unwrap();
        for hidden in [
            "buildingDeck",
            "age2Deck",
            "wealthOffer",
            "private engine",
            "Person",
            "987654",
        ] {
            assert!(!text.contains(hidden));
        }
    }
}
fn o_min_steps(o: &Observation) -> usize {
    o.legal_actions.len() * 4
}

#[test]
fn configuration_is_bounded_unknown_fields_rejected_and_key_covers_behavior() {
    let config = SearchConfig::default();
    let key = config.configuration_key().unwrap();
    assert_eq!(key.len(), 64);
    assert_eq!(key, config.configuration_key().unwrap());
    let mut changed = config.clone();
    changed.sampling_salt += 1;
    assert_ne!(key, changed.configuration_key().unwrap());
    for field in [
        "worldsPerAction",
        "horizonDays",
        "maxRolloutSteps",
        "maxTotalSteps",
        "minCompletedWorlds",
    ] {
        let mut value = serde_json::to_value(&config).unwrap();
        value[field] = 0.into();
        assert!(
            serde_json::from_value::<SearchConfig>(value)
                .unwrap()
                .validate()
                .is_err()
        );
    }
    let mut value = serde_json::to_value(&config).unwrap();
    value["actualGameSeed"] = 7.into();
    assert!(serde_json::from_value::<SearchConfig>(value).is_err());
    let mut bad = config.clone();
    bad.worlds_per_action = 33;
    assert!(bad.validate().is_err());
    let mut bad = config.clone();
    bad.horizon_days = 28;
    assert!(bad.validate().is_err());
    let mut bad = config.clone();
    bad.max_rollout_steps = 2049;
    assert!(bad.validate().is_err());
    let mut bad = config.clone();
    bad.max_total_steps = 262145;
    assert!(bad.validate().is_err());
    let mut bad = config;
    bad.min_completed_worlds = 5;
    assert!(bad.validate().is_err());
}

#[test]
fn caps_discard_entire_blocks_and_fallback_keeps_real_policy_identity() {
    let state = playing(3, 11235);
    let o = observe(&state, state.current_player).unwrap();
    let mut config = small_config();
    config.max_rollout_steps = 1;
    let capped = search::choose_move(&o, &config).unwrap();
    assert_eq!(
        capped.status,
        SearchStatus::Fallback {
            reason: FallbackReason::InsufficientCompletedWorlds
        }
    );
    assert_eq!(capped.decision.r#move, choose_move(&o).unwrap().r#move);
    assert_legal(&o, &capped);
    assert_eq!(capped.stats.completed_worlds, 0);
    assert!(
        capped
            .candidates
            .iter()
            .all(|c| c.visits == 0 && c.mean_score.is_none())
    );
    assert_eq!(
        capped.stats.discarded_worlds[0].reason,
        DiscardReason::RolloutStepCap
    );
    let complete = search::choose_move(&o, &small_config()).unwrap();
    assert_eq!(complete.status, SearchStatus::Searched);
    assert!(complete.stats.atomic_steps > 1);
    let mut incomplete = small_config();
    incomplete.max_total_steps = complete.stats.atomic_steps - 1;
    let partial = search::choose_move(&o, &incomplete).unwrap();
    assert!(
        partial.stats.cutoff_rollouts > 0,
        "earlier candidates did complete"
    );
    assert_eq!(partial.stats.atomic_steps, incomplete.max_total_steps);
    assert_eq!(partial.stats.completed_worlds, 0);
    assert!(
        partial
            .candidates
            .iter()
            .all(|c| c.visits == 0 && c.mean_score.is_none())
    );
    assert_eq!(
        partial.stats.discarded_worlds[0].reason,
        DiscardReason::TotalStepCap
    );
}

#[test]
fn unsupported_roots_fallback_but_rekeyed_supported_forgeries_are_errors() {
    let setup = create_game(vec!["A".into(), "B".into(), "C".into()], 1, false).unwrap();
    let o = observe(&setup, setup.current_player).unwrap();
    let fallback = search::choose_move(&o, &SearchConfig::default()).unwrap();
    assert_eq!(
        fallback.status,
        SearchStatus::Fallback {
            reason: FallbackReason::Setup
        }
    );
    assert_legal(&o, &fallback);
    let mut expanded = create_game(vec!["A".into(), "B".into(), "C".into()], 7, true).unwrap();
    while expanded.phase == Phase::Setup {
        let observed = observe(&expanded, expanded.current_player).unwrap();
        expanded =
            tzolkin_core::apply_move(&expanded, choose_move(&observed).unwrap().r#move).unwrap();
    }
    let observed = observe(&expanded, expanded.current_player).unwrap();
    let fallback = search::choose_move(&observed, &SearchConfig::default()).unwrap();
    assert_eq!(
        fallback.status,
        SearchStatus::Fallback {
            reason: FallbackReason::UnsupportedRules
        }
    );
    assert_legal(&observed, &fallback);
    let two = playing(2, 1);
    let o2 = observe(&two, two.current_player).unwrap();
    assert_eq!(
        search::choose_move(&o2, &SearchConfig::default())
            .unwrap()
            .status,
        SearchStatus::Fallback {
            reason: FallbackReason::UnsupportedPlayerCount
        }
    );
    let mut state = playing(3, 11235);
    while state.pending.is_none() {
        let o = observe(&state, state.current_player).unwrap();
        state = tzolkin_core::apply_move(&state, choose_move(&o).unwrap().r#move).unwrap();
    }
    let pending = observe(&state, state.current_player).unwrap();
    assert_eq!(
        search::choose_move(&pending, &SearchConfig::default())
            .unwrap()
            .status,
        SearchStatus::Fallback {
            reason: FallbackReason::PendingContinuationUnavailable
        }
    );
    let state = playing(4, 11235);
    let mut forged = observe(&state, state.current_player).unwrap();
    if let Some(a) = forged
        .legal_actions
        .iter_mut()
        .find(|a| matches!(a.action, TypedAction::Place { .. }))
        && let TypedAction::Place { corn_cost, .. } = &mut a.action
    {
        *corn_cost += 99;
    }
    forged.observation_key = observation_key(&forged).unwrap();
    assert!(search::choose_move(&forged, &SearchConfig::default()).is_err());
}

#[test]
fn actual_search_decisions_finish_three_and_four_player_native_games_and_reverify() {
    for n in [3, 4] {
        let mut searched = 0;
        let mut fallback = 0;
        let mut pending_fallback = 0;
        // Explicit Search provenance and compact diagnostics. Human rejection
        // is covered separately by the dataset regression tests.
        let policy = search::PreparedSearch::new(&small_config()).unwrap();
        let source = replay::ReplaySource::PolicySelfPlay {
            policies: vec![tzolkin_ai::search_native::provenance(&policy); n],
        };
        let (final_state, decisions, record) = replay::play_game_using_diagnostics(
            n,
            11235,
            replay::options_from_mask(0),
            true,
            source,
            true,
            |o| {
                let outcome = search::choose_move(o, &small_config())?;
                assert_legal(o, &outcome);
                match outcome.status {
                    SearchStatus::Searched => searched += 1,
                    SearchStatus::Fallback { reason } => {
                        fallback += 1;
                        if reason == FallbackReason::PendingContinuationUnavailable {
                            pending_fallback += 1;
                        }
                    }
                }
                let trace = tzolkin_ai::search_native::SearchTrace::from_outcome(&outcome);
                Ok((outcome.decision, Some(trace)))
            },
        )
        .unwrap();
        assert!(
            searched > 10 && fallback > 0 && pending_fallback > 0,
            "{n}p searched={searched} fallback={fallback}"
        );
        assert!(decisions < replay::MAX_DECISIONS);
        assert_eq!(final_state.phase, Phase::Finished);
        assert_eq!(final_state.food_days, [8, 14, 21, 27]);
        let verified = replay::verify_replay(&record.unwrap()).unwrap();
        assert_eq!(verified.final_scores, final_state.final_scores);
    }
}

#[test]
fn multi_turn_search_can_wait_before_food_day_but_recalls_when_food_is_due_now() {
    use tzolkin_core::{GearId, GearWorker, Resource, Turn, TurnMode};
    let mut state = playing(4, 11235);
    let actor = state.current_player;
    state.round = 7;
    state.turn = Turn {
        mode: TurnMode::Remove,
        count: 1,
        ..Turn::default()
    };
    state.players[actor].resources.insert(Resource::Corn, 1);
    for player in &mut state.players {
        if player.id != actor {
            player.resources.insert(Resource::Corn, 20);
        }
    }
    state.gears.get_mut(&GearId::Palenque).unwrap()[4] = Some(GearWorker {
        player_id: actor as i64,
        dummy: false,
    });
    // Expose one corn tile at both candidate maturity positions. Otherwise a
    // burn changes future temple forecasts and this is a different tradeoff.
    for position in [4, 5] {
        state.jungle.get_mut(&position).unwrap().wood -= 1;
    }
    state.players[1].wood_tiles += 2;
    *state.players[1].resources.get_mut(&Resource::Wood).unwrap() += 7;
    assert!(tzolkin_core::validation::validate_game_state(
        &serde_json::to_value(&state).unwrap()
    ));
    for day in [7, 8] {
        state.round = day;
        let observation = observe(&state, actor).unwrap();
        let config = SearchConfig {
            worlds_per_action: 2,
            min_completed_worlds: 2,
            horizon_days: 9 - day,
            ..SearchConfig::default()
        };
        let outcome = search::choose_move(&observation, &config).unwrap();
        assert_eq!(outcome.status, SearchStatus::Searched);
        let recall = observation
            .legal_actions
            .iter()
            .position(|a| {
                matches!(
                    a.action,
                    TypedAction::Remove {
                        gear: GearId::Palenque,
                        position: 4
                    }
                )
            })
            .unwrap();
        let end = observation
            .legal_actions
            .iter()
            .position(|a| matches!(a.action, TypedAction::EndTurn { .. }))
            .unwrap();
        assert!(
            outcome
                .candidates
                .iter()
                .all(|c| c.minimum_scored_round.unwrap() >= 9)
        );
        // Under the frozen heuristic continuation, one day's maturity and the
        // next turn's placement/feeding costs change the preferred order. On
        // day8 the food loss is immediate, so the preference reverses.
        let expected = if day == 7 { end } else { recall };
        let other = if day == 7 { recall } else { end };
        assert!(
            outcome.candidates[expected].mean_score.unwrap()
                > outcome.candidates[other].mean_score.unwrap(),
            "day{day}: {:?}",
            outcome.candidates
        );
        assert_eq!(
            outcome.decision.r#move,
            observation.legal_actions[expected].r#move
        );
        assert_eq!(
            choose_move(&observation).unwrap().r#move,
            observation.legal_actions[recall].r#move
        );
        if day == 7 {
            assert_ne!(
                outcome.decision.r#move,
                choose_move(&observation).unwrap().r#move
            );
        }
    }
}
