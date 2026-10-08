use serde_json::{Value, json};
use tzolkin_core::observation::observe;
use tzolkin_core::public_replay::*;
use tzolkin_core::*;

fn source() -> Evidence {
    Evidence {
        reference: "synthetic native fixture; not a BGA observation".into(),
        action_ids: vec![],
    }
}

fn native_initial(count: usize, seed: u32) -> GameState {
    let mut state =
        create_game((0..count).map(|i| format!("P{i}")).collect(), seed, false).unwrap();
    while state.phase == Phase::Setup {
        let operation = get_available_moves(&state)
            .into_iter()
            .find(|choice| choice.disabled != Some(true))
            .unwrap()
            .r#move;
        state = apply_move(&state, operation).unwrap();
    }
    state
}
fn partial(count: usize) -> PublicReplayRecord {
    PublicReplayRecord {
        schema: PUBLIC_REPLAY_SCHEMA.into(),
        rules_version: PUBLIC_RULES_VERSION,
        catalog_hash: public_catalog_hash(),
        market: Market::Unlimited,
        source: source(),
        initial: PublicState::from_game_state(&native_initial(count, 42)).unwrap(),
        initial_checkpoint: None,
        steps: vec![],
        terminal_checkpoint: None,
    }
}
fn refills(state: &GameState, operation: &GameMove) -> Refills {
    if !matches!(operation, GameMove::EndTurn { .. }) {
        return Refills::default();
    }
    let count = (6 - state.buildings.len()).min(state.building_deck.len());
    let changes_age = state.age == 1
        && state.turn_index + 1 == state.players.len()
        && [8, 14, 21, 27]
            .into_iter()
            .find(|d| state.round >= *d && !state.food_days.contains(d))
            == Some(14);
    Refills {
        current_age: state.building_deck[..count].to_vec(),
        age2: if changes_age {
            state.age2_deck[..6].to_vec()
        } else {
            vec![]
        },
    }
}

#[test]
fn unknown_private_setup_and_decks_never_cross_public_boundary() {
    let record = partial(4);
    let value = serde_json::to_value(&record.initial).unwrap();
    assert_eq!(value["hidden"]["seed"], "unknown");
    for field in ["seed", "buildingDeck", "age2Deck", "expansion"] {
        assert!(value.get(field).is_none(), "{field}");
    }
    for player in value["players"].as_array().unwrap() {
        assert!(player.get("wealthOffer").is_none());
        assert!(player.get("tribeOffer").is_none());
    }
    assert!(!tzolkin_core::validation::validate_game_state(&value));
    assert!(
        tzolkin_core::api::dispatch_game(&json!({"operation":"inspect","state":value}).to_string())
            .is_err()
    );
    let report = verify_public_replay(&record).unwrap();
    assert_eq!(report.status, ReplayStatus::Partial);
    assert!(!report.verified_complete);
    let observation = report.frames[0].observation.as_ref().unwrap();
    assert!(observation.private.wealth_offer.is_empty());
    assert!(observation.private.tribe_offer.is_empty());
    assert_eq!(observation.building_deck_count, 8);
    assert_eq!(observation.age2_deck_count, 18);
    assert_eq!(record.initial.building_deck_count, 8);
}

#[test]
fn checked_api_rejects_hidden_fields_unsupported_rules_and_initial_corruption() {
    let record = partial(3);
    let state = serde_json::to_value(&record.initial).unwrap();
    let result: Value = serde_json::from_str(
        &tzolkin_core::api::dispatch_game(
            &json!({"operation":"publicReplay","replay":record}).to_string(),
        )
        .unwrap(),
    )
    .unwrap();
    assert_eq!(result["status"], "partial");
    for (path, value) in [
        ("/seed", json!(5)),
        (
            "/players/0/wealthOffer",
            json!(["w01", "w02", "w03", "w04"]),
        ),
        ("/buildingDeck", json!(["b01"])),
        ("/expansion", json!({})),
    ] {
        let mut invalid = state.clone();
        let (parent, key) = path.rsplit_once('/').unwrap();
        invalid
            .pointer_mut(parent)
            .unwrap()
            .as_object_mut()
            .unwrap()
            .insert(key.into(), value);
        assert!(
            tzolkin_core::api::dispatch_game(
                &json!({"operation":"publicInspect","state":invalid}).to_string()
            )
            .is_err(),
            "{path}"
        );
    }
    let mut invalid = partial(3);
    invalid.initial.additional_buildings = true;
    assert!(
        verify_public_replay(&invalid)
            .unwrap_err()
            .contains("initial")
    );
    let mut invalid = partial(4);
    *invalid.initial.players[0]
        .resources
        .get_mut(&Resource::Corn)
        .unwrap() += 1;
    assert!(
        verify_public_replay(&invalid)
            .unwrap_err()
            .contains("players[0]")
    );
    let mut invalid = partial(4);
    invalid.initial.building_deck_count += 1;
    assert!(
        verify_public_replay(&invalid)
            .unwrap_err()
            .contains("buildingDeckCount")
    );
    let mut invalid = partial(4);
    invalid.initial.round = 2;
    assert!(verify_public_replay(&invalid).is_err());
}

#[test]
fn source_checkpoints_have_field_errors_and_are_not_replaced_by_hashes() {
    let mut record = partial(4);
    record.initial_checkpoint = Some(Checkpoint {
        source: source(),
        expected: json!({"round":1,"players":record.initial.players.iter()
            .map(|p|json!({"id":p.id,"resources":p.resources,"workers":p.workers})).collect::<Vec<_>>()}),
    });
    assert_eq!(
        verify_public_replay(&record).unwrap().checkpoints_verified,
        1
    );
    record.initial_checkpoint.as_mut().unwrap().expected["players"][0]["resources"]["corn"] =
        json!(999);
    assert!(
        verify_public_replay(&record)
            .unwrap_err()
            .contains("initial checkpoint.expected.players[0].resources.corn")
    );
    record.initial_checkpoint.as_mut().unwrap().expected = json!({"seed":1});
    assert!(
        verify_public_replay(&record)
            .unwrap_err()
            .contains("field is not public")
    );
    record.initial_checkpoint.as_mut().unwrap().expected = json!({});
    assert!(verify_public_replay(&record).is_err());
}

#[test]
fn actor_illegal_move_refills_and_checkpoint_failure_leave_input_untouched() {
    let record = partial(4);
    let state = record.initial.clone();
    let operation = inspect_public(&state)
        .unwrap()
        .observation
        .unwrap()
        .legal_actions[0]
        .r#move
        .clone();
    assert!(
        apply_public(
            &state,
            (state.current_player + 1) % 4,
            operation.clone(),
            &Refills::default()
        )
        .unwrap_err()
        .contains("step 0 actor")
    );
    assert!(
        apply_public(
            &state,
            state.current_player,
            GameMove::EndTurn {
                double_advance: None
            },
            &Refills::default()
        )
        .unwrap_err()
        .contains("step 0 move")
    );
    let extra = Refills {
        current_age: vec!["b01".into()],
        age2: vec![],
    };
    assert!(
        apply_public(&state, state.current_player, operation.clone(), &extra)
            .unwrap_err()
            .contains("step 0 refills.currentAge")
    );
    assert_eq!(state, record.initial);
    let mut invalid = record.clone();
    invalid.steps.push(PublicReplayStep {
        actor: state.current_player,
        r#move: operation,
        refills: Refills::default(),
        source_action_ids: vec![6],
        checkpoint: Some(Checkpoint {
            source: source(),
            expected: json!({"round":99}),
        }),
    });
    assert!(
        verify_public_replay(&invalid)
            .unwrap_err()
            .contains("step 0 checkpoint.expected.round")
    );
    assert_eq!(state, record.initial);
}

#[test]
fn public_invariants_retain_skull_worker_owner_and_exclusive_temple_checks() {
    let record = partial(4);
    let mut invalid = record.initial.clone();
    invalid.skull_supply -= 1;
    assert!(inspect_public(&invalid).is_err());
    let mut invalid = record.initial.clone();
    invalid.gears.get_mut(&GearId::Palenque).unwrap()[0] = Some(GearWorker {
        player_id: 99,
        dummy: false,
    });
    assert!(inspect_public(&invalid).is_err());
    let mut invalid = record.initial.clone();
    for slot in 0..4 {
        invalid.gears.get_mut(&GearId::Palenque).unwrap()[slot] = Some(GearWorker {
            player_id: 0,
            dummy: false,
        });
    }
    invalid.players[0].workers = 3;
    assert!(inspect_public(&invalid).is_err());
    let mut invalid = record.initial.clone();
    for player in &mut invalid.players[..2] {
        *player.temples.get_mut(&TempleId::Chaac).unwrap() = 5;
    }
    assert!(inspect_public(&invalid).is_err());
    let mut invalid = record.initial.clone();
    invalid.players[1].wealth = invalid.players[0].wealth.clone();
    assert!(inspect_public(&invalid).is_err());
}

#[test]
fn age_change_can_omit_only_unobserved_immediately_retired_refill() {
    let mut native = native_initial(4, 42);
    native.round = 14;
    native.food_days = vec![8];
    native.turn_index = 3;
    native.current_player = 3;
    native.turn.mode = TurnMode::Place;
    native.turn.count = 1;
    native.gears.get_mut(&GearId::Palenque).unwrap()[0] = Some(GearWorker {
        player_id: 3,
        dummy: false,
    });
    let purchased = native.buildings.remove(0);
    native.players[0].buildings.push(purchased);
    let state = PublicState::from_game_state(&native).unwrap();
    let operation = GameMove::EndTurn {
        double_advance: None,
    };
    let draws = refills(&native, &operation);
    assert_eq!(draws.current_age.len(), 1);
    assert_eq!(draws.age2.len(), 6);
    let before = state.clone();
    let mut missing = draws.clone();
    missing.age2.clear();
    assert!(
        apply_public(&state, 3, operation.clone(), &missing)
            .unwrap_err()
            .contains("refills.age2")
    );
    let mut missing = draws.clone();
    missing.current_age.clear();
    let omitted = apply_public(&state, 3, operation.clone(), &missing).unwrap();
    assert_eq!(omitted.snapshot.state.retired_unknown_refill_count, 1);
    let mut wrong = draws.clone();
    wrong.age2[0] = draws.current_age[0].clone();
    assert!(
        apply_public(&state, 3, operation.clone(), &wrong)
            .unwrap_err()
            .contains("refills.age2[0]")
    );
    let result = apply_public(&state, 3, operation.clone(), &draws).unwrap();
    let expected = PublicState::from_game_state(&apply_move(&native, operation).unwrap()).unwrap();
    assert_eq!(result.snapshot.state, expected);
    assert_eq!(result.snapshot.state.age, 2);
    assert_eq!(result.snapshot.state.building_deck_count, 12);
    assert_eq!(result.snapshot.state.age2_deck_count, 0);
    let mut observed_equivalent = result.snapshot.state.clone();
    observed_equivalent.retired_unknown_refill_count = 1;
    assert_eq!(omitted.snapshot.state, observed_equivalent);
    assert_eq!(omitted.observation, result.observation);
    assert!(
        serde_json::to_value(&omitted.observation)
            .unwrap()
            .to_string()
            .find("retiredUnknownRefillCount")
            .is_none()
    );
    assert_eq!(state, before);
}

fn day_fourteen_board(count: usize, turn_index: usize) -> GameState {
    let mut state = native_initial(count, 42);
    state.round = 14;
    state.food_days = vec![8];
    state.turn_index = turn_index;
    state.current_player = state.turn_order[turn_index];
    state.turn.mode = TurnMode::Place;
    state.turn.count = 1;
    let free = state.gears[&GearId::Palenque]
        .iter()
        .take(8)
        .position(Option::is_none)
        .unwrap();
    state.gears.get_mut(&GearId::Palenque).unwrap()[free] = Some(GearWorker {
        player_id: state.current_player as i64,
        dummy: false,
    });
    let card = state.buildings.remove(0);
    state.players[0].buildings.push(card);
    for player in &mut state.players {
        *player.resources.get_mut(&Resource::Corn).unwrap() = 0;
    }
    state
}

#[test]
fn starvation_day_fourteen_in_three_and_four_players_is_atomic_without_temple_choice() {
    for count in [3, 4] {
        let native = day_fourteen_board(count, count - 1);
        let state = PublicState::from_game_state(&native).unwrap();
        let operation = GameMove::EndTurn {
            double_advance: None,
        };
        let mut draws = refills(&native, &operation);
        draws.current_age.clear();
        let result = apply_public(&state, state.current_player, operation.clone(), &draws).unwrap();
        let mut reference =
            PublicState::from_game_state(&apply_move(&native, operation).unwrap()).unwrap();
        reference.retired_unknown_refill_count = 1;
        assert_eq!(result.snapshot.state, reference);
        assert_eq!(result.snapshot.state.age, 2);
        assert!(result.snapshot.state.pending.is_none());
        assert_eq!(
            result
                .snapshot
                .state
                .log
                .iter()
                .filter(|text| text.contains("未給食"))
                .count(),
            count
        );
        // A later actor cannot recover the omitted retired card identities.
        let next_move = result.observation.as_ref().unwrap().legal_actions[0]
            .r#move
            .clone();
        let after = apply_public(
            &result.snapshot.state,
            result.snapshot.state.current_player,
            next_move,
            &Refills::default(),
        )
        .unwrap();
        assert_eq!(after.snapshot.state.retired_unknown_refill_count, 1);
        assert!(
            after
                .observation
                .as_ref()
                .unwrap()
                .private
                .wealth_offer
                .is_empty()
        );
    }
}

#[test]
fn ordinary_visible_refills_duplicate_future_draws_and_too_early_age_two_are_rejected() {
    for count in [3, 4] {
        let native = day_fourteen_board(count, count - 2);
        let state = PublicState::from_game_state(&native).unwrap();
        let operation = GameMove::EndTurn {
            double_advance: None,
        };
        let draws = refills(&native, &operation);
        assert_eq!(draws.current_age.len(), 1);
        assert!(draws.age2.is_empty());
        assert!(
            apply_public(
                &state,
                state.current_player,
                operation.clone(),
                &Refills::default()
            )
            .unwrap_err()
            .contains("refills.currentAge")
        );
        let mut wrong = draws.clone();
        wrong.current_age[0] = state.buildings[0].clone();
        assert!(
            apply_public(&state, state.current_player, operation.clone(), &wrong)
                .unwrap_err()
                .contains("refills.currentAge[0]")
        );
        let mut early = draws.clone();
        early.age2 = native.age2_deck[..6].to_vec();
        assert!(
            apply_public(&state, state.current_player, operation.clone(), &early)
                .unwrap_err()
                .contains("refills.age2")
        );
        let real = apply_public(&state, state.current_player, operation.clone(), &draws).unwrap();
        assert_eq!(real.snapshot.state.age, 1);
        assert_eq!(real.snapshot.state.retired_unknown_refill_count, 0);
        assert_eq!(real.snapshot.state.buildings.len(), 6);
    }
    let native = day_fourteen_board(4, 3);
    let state = PublicState::from_game_state(&native).unwrap();
    let operation = GameMove::EndTurn {
        double_advance: None,
    };
    let mut duplicate = refills(&native, &operation);
    duplicate.current_age.clear();
    duplicate.age2[1] = duplicate.age2[0].clone();
    assert!(
        apply_public(&state, 3, operation, &duplicate)
            .unwrap_err()
            .contains("refills.age2[1]")
    );
}

#[test]
fn three_and_four_player_public_games_match_every_native_transition_and_terminal_result() {
    for count in [3, 4] {
        let mut native = native_initial(count, 317);
        let mut record = partial(count);
        record.initial = PublicState::from_game_state(&native).unwrap();
        record.initial_checkpoint = Some(Checkpoint {
            source: source(),
            expected: serde_json::to_value(&record.initial).unwrap(),
        });
        let mut public = record.initial.clone();
        let mut rng = 7919_u32;
        for index in 0..1500 {
            if native.phase == Phase::Finished {
                break;
            }
            let native_observation = observe(&native, native.current_player).unwrap();
            let public_observation = inspect_public(&public).unwrap().observation.unwrap();
            assert_eq!(
                public_observation.legal_actions, native_observation.legal_actions,
                "count {count}, step {index}"
            );
            assert!(public_observation.private.wealth_offer.is_empty());
            rng ^= rng << 13;
            rng ^= rng >> 17;
            rng ^= rng << 5;
            let operation = native_observation.legal_actions
                [rng as usize % native_observation.legal_actions.len()]
            .r#move
            .clone();
            let draws = refills(&native, &operation);
            let actor = native.current_player;
            let previous_food_days = native.food_days.len();
            let resolved_calendar = native
                .pending
                .as_ref()
                .is_some_and(|pending| matches!(pending.task, Task::Rotation));
            native = apply_move(&native, operation.clone()).unwrap();
            public = apply_public(&public, actor, operation.clone(), &draws)
                .unwrap()
                .snapshot
                .state;
            assert_eq!(
                public,
                PublicState::from_game_state(&native).unwrap(),
                "count {count}, step {index}"
            );
            record.steps.push(PublicReplayStep {
                actor,
                r#move: operation,
                refills: draws,
                source_action_ids: vec![index as u64],
                checkpoint: if (native.food_days.len() > previous_food_days
                    && !native
                        .pending
                        .as_ref()
                        .is_some_and(|pending| matches!(pending.task, Task::Rotation)))
                    || resolved_calendar
                {
                    Some(Checkpoint {
                        source: source(),
                        expected: serde_json::to_value(&public).unwrap(),
                    })
                } else {
                    None
                },
            });
        }
        assert_eq!(native.phase, Phase::Finished, "count {count}");
        let incomplete = verify_public_replay(&record).unwrap();
        assert_eq!(incomplete.status, ReplayStatus::Partial);
        assert!(!incomplete.verified_complete);
        assert!(incomplete.frames.last().unwrap().observation.is_none());
        record.terminal_checkpoint = Some(TerminalCheckpoint {
            source: source(),
            scores: native
                .final_scores
                .iter()
                .map(|s| TerminalScore {
                    player_id: s.player_id,
                    total: s.total,
                    rank: s.rank,
                })
                .collect(),
        });
        let report = verify_public_replay(&record).unwrap();
        assert!(report.verified_complete);
        assert!(report.terminal_matched);
        assert!(report.source_coverage.initial);
        assert_eq!(report.source_coverage.food_days, [8, 14, 21, 27]);
        assert!(report.source_coverage.complete);
        assert!(!report.training_ready);
        assert_eq!(report.status, ReplayStatus::Complete);
        assert_eq!(report.verified_steps, record.steps.len());
        let mut markers_only = record.clone();
        markers_only.initial_checkpoint.as_mut().unwrap().expected = json!({"round":1});
        for step in &mut markers_only.steps {
            if let Some(point) = &mut step.checkpoint {
                point.expected = json!({"round":point.expected["round"]});
            }
        }
        let sparse = verify_public_replay(&markers_only).unwrap();
        assert!(sparse.verified_complete);
        assert!(!sparse.source_coverage.initial);
        assert!(sparse.source_coverage.food_days.is_empty());
        assert!(!sparse.source_coverage.complete);
        assert!(!sparse.training_ready);
        record.terminal_checkpoint.as_mut().unwrap().scores[0].total += 1.0;
        assert!(
            verify_public_replay(&record)
                .unwrap_err()
                .contains("terminal scores[0]")
        );
    }
}
