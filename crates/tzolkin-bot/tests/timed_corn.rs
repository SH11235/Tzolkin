use tzolkin_core::observation::{TypedAction, observe};
use tzolkin_core::*;

fn before_food_placement() -> GameState {
    let mut state = create_game(vec!["A".into(), "B".into(), "C".into()], 17, false).unwrap();
    state.phase = Phase::Playing;
    state.round = 7;
    state.pending = None;
    for player in &mut state.players {
        player.resources.insert(Resource::Corn, 100);
        player.feed_workers = 0;
        player.feed_discount = 0;
        player.feed_all = false;
    }
    state.players[0].resources.insert(Resource::Corn, 6);
    apply_move(
        &state,
        GameMove::Place {
            gear: GearId::Tikal,
        },
    )
    .unwrap()
}

fn finish_other_turns(mut state: GameState, actor: usize) -> GameState {
    for _ in 0..12 {
        if state.current_player == actor {
            return state;
        }
        let observation = observe(&state, state.current_player).unwrap();
        let placement = observation
            .legal_actions
            .iter()
            .find(|row| matches!(row.action, TypedAction::Place { .. }))
            .unwrap()
            .r#move
            .clone();
        state = apply_move(&state, placement).unwrap();
        state = apply_move(
            &state,
            GameMove::EndTurn {
                double_advance: None,
            },
        )
        .unwrap();
    }
    panic!("other turns did not finish");
}

#[test]
fn same_route_recovers_before_actual_core_feeding() {
    let state = before_food_placement();
    let observation = observe(&state, 0).unwrap();
    let baseline = tzolkin_bot::choose_move_resource_aligned_v2(&observation).unwrap();
    assert_eq!(
        baseline.r#move,
        GameMove::EndTurn {
            double_advance: None
        }
    );
    let candidate = tzolkin_bot::choose_move_timed_corn(&observation).unwrap();
    assert_eq!(
        candidate.r#move,
        GameMove::Place {
            gear: GearId::Palenque
        }
    );
    assert_eq!(
        candidate.policy_version,
        tzolkin_bot::TIMED_CORN_POLICY_VERSION
    );
    assert!((baseline.score - 13.3).abs() < 1e-12);
    assert!((candidate.score - 13.65).abs() < 1e-12);
    let placed = apply_move(
        &state,
        GameMove::Place {
            gear: GearId::Palenque,
        },
    )
    .unwrap();
    assert_eq!(placed.players[0].resources[&Resource::Corn], 5);
    let ended = apply_move(
        &placed,
        GameMove::EndTurn {
            double_advance: None,
        },
    )
    .unwrap();
    let mut next = finish_other_turns(ended, 0);
    assert_eq!(next.round, 8);
    assert!(!next.food_days.contains(&8));
    let recovery = tzolkin_bot::choose_move_timed_corn(&observe(&next, 0).unwrap()).unwrap();
    assert_eq!(recovery.r#move, GameMove::FirstPlayer);
    let claimed = apply_move(&next, recovery.r#move).unwrap();
    let end = tzolkin_bot::choose_move_timed_corn(&observe(&claimed, 0).unwrap()).unwrap();
    let actual = finish_other_turns(apply_move(&claimed, end.r#move).unwrap(), 0);
    assert!(actual.food_days.contains(&8));
    assert_eq!(actual.players[0].score, 0.0);
    next = apply_move(
        &next,
        GameMove::Remove {
            gear: GearId::Palenque,
            position: 1,
        },
    )
    .unwrap();
    let execution = tzolkin_bot::choose_move_timed_corn(&observe(&next, 0).unwrap()).unwrap();
    assert_eq!(
        execution.r#move,
        GameMove::Choose {
            choice_id: "action:1".into()
        }
    );
    next = apply_move(&next, execution.r#move).unwrap();
    assert_eq!(next.players[0].resources[&Resource::Corn], 8);
    next = apply_move(
        &next,
        GameMove::EndTurn {
            double_advance: None,
        },
    )
    .unwrap();
    next = finish_other_turns(next, 0);
    assert!(next.food_days.contains(&8));
    assert_eq!(next.players[0].resources[&Resource::Corn], 2);
    assert_eq!(next.players[0].score, 0.0);
}

#[test]
fn same_recovery_after_food_does_not_remove_the_core_loss() {
    let mut state = before_food_placement();
    state.round = 8;
    let observation = observe(&state, 0).unwrap();
    assert_eq!(
        tzolkin_bot::choose_move_timed_corn(&observation)
            .unwrap()
            .r#move,
        GameMove::EndTurn {
            double_advance: None
        }
    );
    let placed = apply_move(
        &state,
        GameMove::Place {
            gear: GearId::Palenque,
        },
    )
    .unwrap();
    let ended = apply_move(
        &placed,
        GameMove::EndTurn {
            double_advance: None,
        },
    )
    .unwrap();
    let mut next = finish_other_turns(ended, 0);
    assert_eq!(next.round, 9);
    assert!(next.food_days.contains(&8));
    assert_eq!(next.players[0].score, -3.0);
    next = apply_move(
        &next,
        GameMove::Remove {
            gear: GearId::Palenque,
            position: 1,
        },
    )
    .unwrap();
    next = apply_move(
        &next,
        GameMove::Choose {
            choice_id: "action:1".into(),
        },
    )
    .unwrap();
    assert_eq!(next.players[0].score, -3.0);
    assert_eq!(next.players[0].resources[&Resource::Corn], 4);
}

#[test]
fn worker_food_cost_changes_root_choice_and_pending_execution_coherently() {
    let mut state = before_food_placement();
    state.turn = Turn::default();
    state.gears.get_mut(&GearId::Tikal).unwrap()[0] = None;
    state.gears.get_mut(&GearId::Uxmal).unwrap()[3] = Some(GearWorker {
        player_id: 0,
        dummy: false,
    });
    let observation = observe(&state, 0).unwrap();
    let baseline = tzolkin_bot::choose_move_resource_aligned_v2(&observation).unwrap();
    let candidate = tzolkin_bot::choose_move_timed_corn(&observation).unwrap();
    assert_eq!(
        baseline.r#move,
        GameMove::Remove {
            gear: GearId::Uxmal,
            position: 3
        }
    );
    assert_eq!(
        candidate.r#move,
        GameMove::Place {
            gear: GearId::Uxmal
        }
    );
    let removed = apply_move(&state, baseline.r#move).unwrap();
    let pending = observe(&removed, 0).unwrap();
    let base_execution = tzolkin_bot::choose_move_resource_aligned_v2(&pending).unwrap();
    let execution = tzolkin_bot::choose_move_timed_corn(&pending).unwrap();
    assert_eq!(execution.r#move, base_execution.r#move);
    assert_eq!(
        execution.r#move,
        GameMove::Choose {
            choice_id: "action:3".into()
        }
    );
    assert!((base_execution.score - execution.score - 3.0).abs() < 1e-12);
    let grown = apply_move(&removed, execution.r#move).unwrap();
    assert_eq!(grown.players[0].workers, 4);
}

#[test]
fn paid_yaxchilan_recovery_is_followed_through_at_each_callback() {
    let mut state = before_food_placement();
    state.round = 8;
    state.turn = Turn::default();
    state.players[0].resources.insert(Resource::Corn, 5);
    state.gears.get_mut(&GearId::Tikal).unwrap()[0] = None;
    state.gears.get_mut(&GearId::Yaxchilan).unwrap()[4] = Some(GearWorker {
        player_id: 0,
        dummy: false,
    });
    state.skull_supply = 0;
    let decision = tzolkin_bot::choose_move_timed_corn(&observe(&state, 0).unwrap()).unwrap();
    assert_eq!(
        decision.r#move,
        GameMove::Remove {
            gear: GearId::Yaxchilan,
            position: 4
        }
    );
    state = apply_move(&state, decision.r#move).unwrap();
    let execution = tzolkin_bot::choose_move_timed_corn(&observe(&state, 0).unwrap()).unwrap();
    assert_eq!(
        execution.r#move,
        GameMove::Choose {
            choice_id: "action:3".into()
        }
    );
    state = apply_move(&state, execution.r#move).unwrap();
    assert_eq!(state.players[0].resources[&Resource::Corn], 6);
    state = apply_move(
        &state,
        GameMove::EndTurn {
            double_advance: None,
        },
    )
    .unwrap();
    state = finish_other_turns(state, 0);
    assert!(state.food_days.contains(&8));
    assert_eq!(state.players[0].score, 0.0);
}
