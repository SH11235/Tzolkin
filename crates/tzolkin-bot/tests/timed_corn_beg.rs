use tzolkin_core::observation::{TypedAction, observe};
use tzolkin_core::*;

fn position(round: i64, corn: i64) -> GameState {
    let mut state = create_game(vec!["A".into(), "B".into(), "C".into()], 17, false).unwrap();
    state.phase = Phase::Playing;
    state.round = round;
    state.food_days = [8, 14, 21, 27]
        .into_iter()
        .filter(|day| *day < round)
        .collect();
    state.pending = None;
    state.players[0].resources.insert(Resource::Corn, corn);
    state.players[0].feed_workers = 0;
    state.players[0].feed_discount = 0;
    state.players[0].feed_all = false;
    state
}

fn resolve_beg(state: &GameState) -> GameState {
    let pending = apply_move(state, GameMove::Beg).unwrap();
    assert_eq!(pending.players[0].resources, state.players[0].resources);
    let o = observe(&pending, 0).unwrap();
    let old = tzolkin_bot::choose_move_timed_corn(&o).unwrap();
    let new = tzolkin_bot::choose_move_timed_corn_beg(&o).unwrap();
    assert_eq!(new.r#move, old.r#move);
    assert_eq!(new.score.to_bits(), old.score.to_bits());
    apply_move(&pending, new.r#move).unwrap()
}

#[test]
fn beg_top_up_resolves_after_descent_and_unavailable_descent_stays_illegal() {
    for corn in [0, 1, 2] {
        let state = position(7, corn);
        let observation = observe(&state, 0).unwrap();
        let legal = observation.legal_actions.clone();
        let _ = tzolkin_bot::choose_move_timed_corn_beg(&observation).unwrap();
        assert_eq!(observation.legal_actions, legal);
        let after = resolve_beg(&state);
        assert_eq!(after.players[0].resources[&Resource::Corn], 3);
        assert!(after.turn.begged);
        assert!(
            !observe(&after, 0)
                .unwrap()
                .legal_actions
                .iter()
                .any(|row| matches!(row.action, TypedAction::Beg))
        );
    }
    let mut forbidden = position(7, 0);
    for level in forbidden.players[0].temples.values_mut() {
        *level = -1;
    }
    assert!(apply_move(&forbidden, GameMove::Beg).is_err());
    let o = observe(&forbidden, 0).unwrap();
    assert!(
        !o.legal_actions
            .iter()
            .any(|row| matches!(row.action, TypedAction::Beg))
    );
    let decision = tzolkin_bot::choose_move_timed_corn_beg(&o).unwrap();
    assert_ne!(decision.r#move, GameMove::Beg);
}

#[test]
fn aligned_currency_reaches_choice_order_without_banning_beg() {
    let mut state = position(6, 0);
    state.players[0].feed_workers = 1;
    state.players[0].buildings.push("b03".into());
    for (gear, slot) in [
        (GearId::Palenque, 2),
        (GearId::Yaxchilan, 1),
        (GearId::Tikal, 3),
    ] {
        state.gears.get_mut(&gear).unwrap()[slot] = Some(GearWorker {
            player_id: 0,
            dummy: false,
        });
    }
    let o = observe(&state, 0).unwrap();
    let old = tzolkin_bot::choose_move_timed_corn(&o).unwrap();
    let new = tzolkin_bot::choose_move_timed_corn_beg(&o).unwrap();
    assert_eq!(old.r#move, GameMove::Beg);
    assert_eq!(
        new.r#move,
        GameMove::Remove {
            gear: GearId::Palenque,
            position: 2
        }
    );
    assert!((old.score - 15.6).abs() < 1e-12);
    assert!((new.score - 15.5).abs() < 1e-12);
    assert_eq!(
        new.policy_version,
        tzolkin_bot::TIMED_CORN_BEG_POLICY_VERSION
    );
    let mut waiting = position(7, 0);
    for gear in [GearId::Palenque, GearId::Yaxchilan, GearId::Tikal] {
        waiting.gears.get_mut(&gear).unwrap()[0] = Some(GearWorker {
            player_id: 0,
            dummy: false,
        });
    }
    let decision = tzolkin_bot::choose_move_timed_corn_beg(&observe(&waiting, 0).unwrap()).unwrap();
    assert_eq!(decision.r#move, GameMove::Beg);
}

#[test]
fn beg_cash_can_enable_paid_placement_even_when_not_selected() {
    let mut state = position(24, 0);
    state.gears.get_mut(&GearId::Yaxchilan).unwrap()[0] = None;
    state.gears.get_mut(&GearId::Yaxchilan).unwrap()[1] = None;
    let first = apply_move(
        &state,
        GameMove::Place {
            gear: GearId::Yaxchilan,
        },
    )
    .unwrap();
    assert!(
        apply_move(
            &first,
            GameMove::Place {
                gear: GearId::Yaxchilan
            }
        )
        .is_err()
    );
    let funded = resolve_beg(&state);
    let first = apply_move(
        &funded,
        GameMove::Place {
            gear: GearId::Yaxchilan,
        },
    )
    .unwrap();
    let second = apply_move(
        &first,
        GameMove::Place {
            gear: GearId::Yaxchilan,
        },
    )
    .unwrap();
    assert_eq!(second.players[0].resources[&Resource::Corn], 1);
    let o = observe(&state, 0).unwrap();
    let score = tzolkin_bot::policy::score_action_timed_corn_beg(
        &o,
        &TypedAction::Beg,
        &tzolkin_bot::policy::HeuristicWeights::default(),
    );
    assert!(score.is_finite());
    assert!(
        o.legal_actions
            .iter()
            .any(|row| matches!(row.action, TypedAction::Beg))
    );
}
