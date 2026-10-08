use tzolkin_ai::{
    choose_move, dispatch_cpu,
    policy::{HeuristicWeights, score_action},
    replay,
};
use tzolkin_core::observation::{TypedAction, observation_key, observe, resolve_action};
use tzolkin_core::prophecies::{FoodDayStage, ProphecyId};
use tzolkin_core::*;

fn names(n: usize) -> Vec<String> {
    (0..n).map(|i| format!("Player {i}")).collect()
}
#[test]
fn hidden_information_and_presentation_do_not_change_observation_or_decision() {
    let state = create_game_with_options(names(5), 42, replay::options_from_mask(15)).unwrap();
    let mut other = state.clone();
    other.seed = 876;
    other.building_deck.reverse();
    other.age2_deck.reverse();
    other.log.push("private diagnostic".into());
    for p in &mut other.players {
        p.name = "private name".into();
        p.color = "private color".into();
        if p.id != state.current_player {
            p.wealth_offer.reverse();
            p.tribe_offer.reverse();
        }
    }
    other
        .expansion
        .as_mut()
        .unwrap()
        .quick_actions
        .as_mut()
        .unwrap()
        .age2
        .reverse();
    let first = observe(&state, state.current_player).unwrap();
    let second = observe(&other, other.current_player).unwrap();
    assert_eq!(first, second);
    assert_eq!(choose_move(&first), choose_move(&second));
    assert!(first.expansion.as_ref().unwrap().quick_age2.is_none());
    let mut leak = serde_json::to_value(&first).unwrap();
    leak["seed"] = serde_json::json!(42);
    assert!(dispatch_cpu(&leak.to_string()).is_err());
    assert!(observe(&state, 1).is_err());
    let mut changed = first;
    changed.private.wealth_offer.reverse();
    changed.observation_key = observation_key(&changed).unwrap();
    assert_ne!(second, changed);
}
#[test]
fn public_score_and_achievement_counts_are_part_of_the_key() {
    let mut state = create_game(names(2), 5, false).unwrap();
    state.phase = Phase::Playing;
    let base = observe(&state, 0).unwrap();
    for field in 0..4 {
        let mut other = state.clone();
        match field {
            0 => other.players[1].score += 30.0,
            1 => other.players[1].corn_tiles += 1,
            2 => other.players[1].skulls_placed += 1,
            _ => other.players[1].monuments.push("public monument".into()),
        }
        assert_ne!(
            base.observation_key,
            observe(&other, 0).unwrap().observation_key
        );
    }
}
#[test]
fn opponents_setup_choices_and_shared_skull_effects_remain_hidden() {
    let initial = (0..100)
        .map(|seed| {
            create_game_with_options(names(3), seed, replay::options_from_mask(15)).unwrap()
        })
        .find(|s| s.players[0].wealth_offer.iter().any(|id| id == "w17"))
        .unwrap();
    let finish_first = |mut state: GameState, take_skull: bool| {
        for _ in 0..20 {
            if state.current_player != 0 {
                return state;
            }
            let choices = get_available_moves(&state)
                .into_iter()
                .filter(|c| c.disabled != Some(true))
                .collect::<Vec<_>>();
            let choice = choices
                .iter()
                .find(|c| match &c.r#move {
                    GameMove::Choose { choice_id } if choice_id.starts_with("wealth:") => {
                        choice_id.split(':').any(|p| p == "w17") == take_skull
                    }
                    _ => false,
                })
                .unwrap_or(&choices[0]);
            state = apply_move(&state, choice.r#move.clone()).unwrap();
        }
        panic!("Setup did not advance");
    };
    let with_skull = finish_first(initial.clone(), true);
    let without_skull = finish_first(initial, false);
    assert_ne!(
        with_skull.players[0].resources,
        without_skull.players[0].resources
    );
    assert_ne!(with_skull.skull_supply, without_skull.skull_supply);
    let a = observe(&with_skull, 1).unwrap();
    let b = observe(&without_skull, 1).unwrap();
    assert_eq!(a, b);
    assert_eq!(choose_move(&a), choose_move(&b));
    // Dummy placement is deferred until all setup choices have been revealed.
    assert_eq!(with_skull.gears, without_skull.gears);
}
#[test]
fn food_decision_actor_uses_their_own_private_view_and_resources() {
    let mut state = create_game_with_options(names(3), 0, replay::options_from_mask(14)).unwrap();
    state.phase = Phase::Playing;
    state.current_player = 1;
    state.turn_index = 0;
    state.round = 8;
    state.food_days = vec![8];
    state.expansion.as_mut().unwrap().prophecies = vec![ProphecyId::GoldShortage];
    state.expansion.as_mut().unwrap().active_prophecy = Some(0);
    state.players[1].resources.insert(Resource::Corn, 4);
    state.pending = Some(Pending {
        title: "Food income".into(),
        task: Task::ProphecyGain {
            player_id: 1,
            resources: [(Resource::Gold, 1)].into(),
        },
        after: vec![Task::FoodDay {
            day: 8,
            stage: FoodDayStage::Feeding,
            fed_workers: vec![],
        }],
    });
    let observation = observe(&state, 1).unwrap();
    assert_eq!(observation.actor, 1);
    assert_eq!(observation.turn_player, 0);
    assert_eq!(
        observation.private.wealth_offer,
        state.players[1].wealth_offer
    );
    assert!(observation.expansion.as_ref().unwrap().quick_age2.is_none());
    let result = choose_move(&observation).unwrap();
    assert_eq!(result.actor, 1);
    let action = observation
        .legal_actions
        .iter()
        .find(|a| a.r#move == result.r#move)
        .unwrap();
    assert!(matches!(
        action.action,
        TypedAction::ProphecyGain { recipient: 1, .. }
    ));
    state.pending.as_mut().unwrap().after = vec![Task::FoodDay {
        day: 8,
        stage: FoodDayStage::Scoring,
        fed_workers: vec![3; 3],
    }];
    assert!(
        observe(&state, 1)
            .unwrap()
            .expansion
            .as_ref()
            .unwrap()
            .quick_age2
            .is_some()
    );
}
#[test]
fn market_policy_has_a_monotone_exit_and_keeps_all_legal_candidates() {
    let mut state = create_game(names(2), 0, false).unwrap();
    state.phase = Phase::Playing;
    state.players[0].resources.insert(Resource::Corn, 0);
    state.players[0].resources.insert(Resource::Wood, 4);
    state.pending = Some(Pending {
        title: "Market".into(),
        task: Task::Trade,
        after: vec![],
    });
    for _ in 0..8 {
        if state.pending.is_none() {
            return;
        }
        let observation = observe(&state, 0).unwrap();
        assert!(
            observation
                .legal_actions
                .iter()
                .any(|c| matches!(c.action, TypedAction::Trade { buy: true, .. }))
                || state.players[0].resources[&Resource::Corn] < 2
        );
        let decision = choose_move(&observation).unwrap();
        state = apply_move(&state, decision.r#move).unwrap();
    }
    panic!("Market policy did not exit");
}
#[test]
fn reachable_semantic_operations_resolve_to_the_same_transition() {
    for (players, mask, seed) in [(2, 0, 0), (3, 7, 1), (4, 14, 9), (5, 15, 2)] {
        let states =
            replay::benchmark_states(players, seed, replay::options_from_mask(mask)).unwrap();
        for state in states.iter().step_by(3) {
            let observation = observe(state, state.current_player).unwrap();
            for candidate in &observation.legal_actions {
                let resolved = resolve_action(state, &candidate.action).unwrap();
                assert_eq!(
                    apply_move(state, candidate.r#move.clone()).unwrap(),
                    apply_move(state, resolved).unwrap()
                );
                assert!(
                    score_action(
                        &observation,
                        &candidate.action,
                        &HeuristicWeights::default()
                    )
                    .is_finite()
                );
            }
        }
    }
}
#[test]
fn complete_replay_checks_views_actors_transitions_and_results() {
    let (_, _, replay) = replay::play_game(5, 2, replay::options_from_mask(15), true).unwrap();
    let replay = replay.unwrap();
    replay::verify_replay(&replay).unwrap();
    let mut tampered = replay.clone();
    tampered.steps[3].actor = (tampered.steps[3].actor + 1) % 5;
    assert!(replay::verify_replay(&tampered).is_err());
    let mut tampered = replay.clone();
    tampered.steps[3].state_after = "incorrect".into();
    assert!(replay::verify_replay(&tampered).is_err());
    let mut tampered = replay.clone();
    tampered.header.replay_schema += 1;
    assert!(replay::verify_replay(&tampered).is_err());
    let mut tampered = replay.clone();
    tampered.final_scores[0].rank += 1;
    assert!(replay::verify_replay(&tampered).is_err());
    let mut human = replay;
    human.header.source = replay::ReplaySource::Human {
        provider: "import-fixture".into(),
        reference: "fixture".into(),
        skill_rating: None,
    };
    replay::verify_replay(&human).unwrap();
    human.verified_complete = false;
    assert!(replay::verify_replay(&human).is_err());
}
#[test]
fn all_supported_configurations_finish_with_legal_operations() {
    let summary = replay::verify_corpus(1).unwrap();
    assert_eq!(summary.games, 56);
    assert_eq!(summary.illegal_operations, 0);
    assert_eq!(summary.unfinished_games, 0);
}

fn playing_with_tribe(tribe: tzolkin_core::tribes::TribeId) -> GameState {
    let mut state = (0..1000)
        .map(|seed| {
            create_game_with_options(
                names(2),
                seed,
                GameOptions {
                    tribes: true,
                    ..GameOptions::default()
                },
            )
            .unwrap()
        })
        .find(|state| state.players[0].tribe_offer.contains(&tribe))
        .unwrap();
    while state.phase == Phase::Setup {
        let choices = get_available_moves(&state)
            .into_iter()
            .filter(|c| c.disabled != Some(true))
            .collect::<Vec<_>>();
        let chosen = if state.current_player == 0 && state.players[0].tribe.is_none() {
            choices.iter().find(|c| matches!(&c.r#move, GameMove::Choose { choice_id } if choice_id == &format!("tribe:{tribe}"))).unwrap()
        } else {
            &choices[0]
        };
        state = apply_move(&state, chosen.r#move.clone()).unwrap();
    }
    state
}
#[test]
fn balam_typed_cost_matches_corn_gain_for_normal_free_and_high_actions() {
    use tzolkin_core::observation::{ActionSource, action_for_move};
    use tzolkin_core::tribes::TribeId;
    let initial = playing_with_tribe(TribeId::Balam);
    for gear in [GearId::Palenque, GearId::Yaxchilan, GearId::ChichenItza] {
        for (position, free) in [
            (3, None),
            (3, Some(true)),
            (if gear == GearId::ChichenItza { 10 } else { 6 }, None),
        ] {
            let mut state = initial.clone();
            state.players[0].resources.insert(Resource::Corn, 6);
            let previous_skulls = state.players[0].resources[&Resource::Skull];
            state.players[0].resources.insert(Resource::Skull, 1);
            state.skull_supply += previous_skulls - 1;
            state.players[0]
                .technologies
                .insert(TechnologyId::Theology, 0);
            state.pending = Some(Pending {
                title: "Action cost fixture".into(),
                task: Task::Action {
                    gear,
                    position,
                    free,
                },
                after: vec![],
            });
            let operation = GameMove::Choose {
                choice_id: "action:1".into(),
            };
            let typed = action_for_move(&state, &operation).unwrap();
            let TypedAction::UseAction {
                corn_cost, source, ..
            } = typed
            else {
                panic!("Wrong semantic action")
            };
            assert_eq!(source, ActionSource::CurrentGear);
            assert_eq!(
                corn_cost, -1,
                "gear={gear:?},position={position},free={free:?}"
            );
            let after = apply_move(&state, operation).unwrap();
            let base_reward = if gear == GearId::Palenque { 3 } else { 0 };
            assert_eq!(
                after.players[0].resources[&Resource::Corn]
                    - state.players[0].resources[&Resource::Corn],
                base_reward - corn_cost
            );
        }
    }
    // Theology's forward operation does not receive Balam's backward reward.
    let mut state = initial;
    state.players[0].resources.insert(Resource::Corn, 6);
    let previous_skulls = state.players[0].resources[&Resource::Skull];
    state.players[0].resources.insert(Resource::Skull, 1);
    state.skull_supply += previous_skulls - 1;
    state.players[0]
        .technologies
        .insert(TechnologyId::Theology, 1);
    state.pending = Some(Pending {
        title: "Ahead cost fixture".into(),
        task: Task::Action {
            gear: GearId::ChichenItza,
            position: 3,
            free: None,
        },
        after: vec![],
    });
    let operation = GameMove::Choose {
        choice_id: "ahead:4".into(),
    };
    assert!(matches!(
        action_for_move(&state, &operation).unwrap(),
        TypedAction::UseAction {
            source: ActionSource::TheologyAhead,
            corn_cost: 0,
            ..
        }
    ));
    assert_eq!(
        apply_move(&state, operation).unwrap().players[0].resources[&Resource::Corn],
        6
    );
    let mut state = playing_with_tribe(TribeId::AhauChamahez);
    state.players[0].resources.insert(Resource::Corn, 6);
    state.pending = Some(Pending {
        title: "Tribe ahead fixture".into(),
        task: Task::Action {
            gear: GearId::Yaxchilan,
            position: 3,
            free: Some(true),
        },
        after: vec![],
    });
    let operation = GameMove::Choose {
        choice_id: "tribeAhead:4".into(),
    };
    assert!(matches!(
        action_for_move(&state, &operation).unwrap(),
        TypedAction::UseAction {
            source: ActionSource::TribeAhead,
            corn_cost: 1,
            ..
        }
    ));
    assert_eq!(
        apply_move(&state, operation).unwrap().players[0].resources[&Resource::Corn],
        5
    );
}
#[test]
fn mercy_and_normal_placement_typed_cost_matches_the_actual_payment() {
    use tzolkin_core::observation::action_for_move;
    use tzolkin_core::tribes::TribeId;
    let mut initial = playing_with_tribe(TribeId::CitBolonTum);
    initial.players[0].temples = TEMPLE_IDS.into_iter().map(|temple| (temple, -1)).collect();
    initial.first_player_claimed = Some(1);
    for gear in GEAR_IDS {
        initial.gears.get_mut(&gear).unwrap().fill(None);
        for pos in 0..2 {
            initial.gears.get_mut(&gear).unwrap()[pos] = Some(GearWorker {
                player_id: -1,
                dummy: true,
            });
        }
    }
    for pos in 3..5 {
        initial.gears.get_mut(&GearId::Palenque).unwrap()[pos] = Some(GearWorker {
            player_id: -1,
            dummy: true,
        });
    }
    assert_eq!(available_workers(&initial, 0), initial.players[0].workers);
    for corn in [0, 1, 2, 3] {
        let mut state = initial.clone();
        state.players[0].resources.insert(Resource::Corn, corn);
        let operation = GameMove::Place {
            gear: GearId::Yaxchilan,
        };
        let TypedAction::Place { corn_cost, .. } = action_for_move(&state, &operation).unwrap()
        else {
            panic!("Wrong semantic action")
        };
        assert_eq!(corn_cost, corn.min(2));
        let after = apply_move(&state, operation).unwrap();
        assert_eq!(
            state.players[0].resources[&Resource::Corn]
                - after.players[0].resources[&Resource::Corn],
            corn_cost
        );
        // The existing mercy predicate also controls discounted tribe placement.
        let operation = GameMove::TribeAbility {
            ability: "discount:yaxchilan".into(),
        };
        let TypedAction::Place {
            corn_cost,
            discount,
            ..
        } = action_for_move(&state, &operation).unwrap()
        else {
            panic!("Wrong semantic action")
        };
        assert!(discount);
        assert_eq!(corn_cost, if corn < 2 { corn } else { 0 });
        let after = apply_move(&state, operation).unwrap();
        assert_eq!(
            state.players[0].resources[&Resource::Corn]
                - after.players[0].resources[&Resource::Corn],
            corn_cost
        );
    }
}
