use serde_json::{Value, json};
use tzolkin_core::api::dispatch_game;
use tzolkin_core::observation::{TypedAction, observe};
use tzolkin_core::public_replay::{PublicState, inspect_public};
use tzolkin_core::tribes::TribeId;
use tzolkin_core::{
    GEAR_IDS, GameMove, GameState, GearId, GearWorker, Phase, Resource, TEMPLE_IDS, apply_move,
    create_game, get_choices, get_placement_cost,
};

fn fixture(corn: i64) -> GameState {
    let mut state = create_game(
        vec!["A".into(), "B".into(), "C".into(), "D".into()],
        42,
        false,
    )
    .unwrap();
    while state.phase == Phase::Setup {
        let choice = get_choices(&state)
            .into_iter()
            .find(|choice| choice.disabled != Some(true))
            .unwrap();
        state = apply_move(&state, choice.r#move).unwrap();
    }
    state.round = 5;
    state.first_player = 1;
    state.turn_order = vec![1, 2, 3, 0];
    state.turn_index = 3;
    state.current_player = 0;
    state.first_player_claimed = Some(1);
    state.players[0].workers = 3;
    state.players[0].resources.insert(Resource::Corn, corn);
    for temple in TEMPLE_IDS {
        state.players[0].temples.insert(temple, -1);
    }
    for player in &mut state.players[1..] {
        player.workers = 6;
    }
    for (index, gear) in GEAR_IDS.into_iter().enumerate() {
        for position in 0..2 {
            state.gears.get_mut(&gear).unwrap()[position] = Some(GearWorker {
                player_id: (1 + (index * 2 + position) % 3) as i64,
                dummy: false,
            });
        }
    }
    state.gears.get_mut(&GearId::Palenque).unwrap()[2] = Some(GearWorker {
        player_id: 2,
        dummy: false,
    });
    state
}

fn inspect(state: &GameState) -> Value {
    serde_json::from_str(
        &dispatch_game(&json!({"operation":"inspect", "state":state}).to_string()).unwrap(),
    )
    .unwrap()
}

fn place<'a>(snapshot: &'a Value, gear: &str) -> &'a Value {
    snapshot["moves"]
        .as_array()
        .unwrap()
        .iter()
        .find(|choice| choice["id"] == format!("place:{gear}"))
        .unwrap()
}

#[test]
fn mercy_display_matches_typed_payment_and_actual_deduction() {
    for corn in [0, 1] {
        let state = fixture(corn);
        let snapshot = inspect(&state);
        assert_eq!(get_placement_cost(&state, "yaxchilan"), Some(2));
        assert_eq!(snapshot["placementCosts"]["yaxchilan"], corn);
        assert_eq!(place(&snapshot, "yaxchilan")["disabled"], false);
        assert_eq!(
            place(&snapshot, "yaxchilan")["description"],
            format!("コーン {corn} · 神の慈悲（所持コーンすべて）")
        );
        assert_eq!(snapshot["placementCosts"]["palenque"], 3);
        assert_eq!(place(&snapshot, "palenque")["disabled"], true);
        assert_eq!(place(&snapshot, "palenque")["description"], "コーン 3");
        let public = inspect_public(&PublicState::from_game_state(&state).unwrap()).unwrap();
        assert_eq!(
            public.snapshot.placement_costs[&GearId::Yaxchilan],
            Some(corn)
        );
        assert_eq!(
            public
                .snapshot
                .moves
                .iter()
                .find(|choice| choice.id == "place:yaxchilan")
                .unwrap()
                .description,
            Some(format!("コーン {corn} · 神の慈悲（所持コーンすべて）"))
        );
        let observation = observe(&state, 0).unwrap();
        assert!(observation.legal_actions.iter().any(|choice| matches!(
            choice.action,
            TypedAction::Place {
                gear: GearId::Yaxchilan,
                corn_cost,
                ..
            } if corn_cost == corn
        )));
        let next = apply_move(
            &state,
            GameMove::Place {
                gear: GearId::Yaxchilan,
            },
        )
        .unwrap();
        assert_eq!(next.players[0].resources[&Resource::Corn], 0);
        assert_eq!(
            next.gears[&GearId::Yaxchilan][2]
                .as_ref()
                .unwrap()
                .player_id,
            0
        );
    }
}

#[test]
fn a_worker_already_on_a_gear_does_not_receive_mercy_display() {
    let mut state = fixture(1);
    for gear in [GearId::Palenque, GearId::Uxmal] {
        state.gears.get_mut(&gear).unwrap()[0] = Some(GearWorker {
            player_id: 0,
            dummy: false,
        });
    }
    let snapshot = inspect(&state);
    assert_eq!(snapshot["availableWorkers"][0], 1);
    assert_eq!(snapshot["placementCosts"]["yaxchilan"], 2);
    assert_eq!(place(&snapshot, "yaxchilan")["description"], "コーン 2");
    assert_eq!(place(&snapshot, "yaxchilan")["disabled"], true);
    assert!(
        apply_move(
            &state,
            GameMove::Place {
                gear: GearId::Yaxchilan
            }
        )
        .is_err()
    );
}

#[test]
fn ordinary_and_full_gear_display_keep_their_existing_payment() {
    let mut state = fixture(5);
    let snapshot = inspect(&state);
    assert_eq!(snapshot["placementCosts"]["yaxchilan"], 2);
    assert_eq!(place(&snapshot, "yaxchilan")["description"], "コーン 2");
    assert_eq!(place(&snapshot, "yaxchilan")["disabled"], false);
    let next = apply_move(
        &state,
        GameMove::Place {
            gear: GearId::Yaxchilan,
        },
    )
    .unwrap();
    assert_eq!(next.players[0].resources[&Resource::Corn], 3);
    for gear in GEAR_IDS {
        state.gears.get_mut(&gear).unwrap().fill(None);
    }
    for position in 0..=10 {
        state.gears.get_mut(&GearId::ChichenItza).unwrap()[position] = Some(GearWorker {
            player_id: (1 + position % 3) as i64,
            dummy: false,
        });
    }
    let full = inspect(&state);
    assert!(full["placementCosts"]["chichenItza"].is_null());
    assert_eq!(
        place(&full, "chichenItza")["description"],
        "空きがありません"
    );
    assert_eq!(place(&full, "chichenItza")["disabled"], true);
}

#[test]
fn unaffordable_tribal_discount_keeps_nominal_description_without_mercy() {
    let mut state = fixture(0);
    state.version = 2;
    state.expansion = Some(Default::default());
    let offers = [
        [TribeId::CitBolonTum, TribeId::Huracan],
        [TribeId::Bacab, TribeId::Balam],
        [TribeId::Itzamna, TribeId::Ahmakiq],
        [TribeId::Yumkaax, TribeId::AhauChamahez],
    ];
    for (player, offer) in state.players.iter_mut().zip(offers) {
        player.tribe = Some(offer[0]);
        player.tribe_offer = offer.to_vec();
    }
    for (index, gear) in GEAR_IDS.into_iter().enumerate() {
        state.gears.get_mut(&gear).unwrap()[2] = Some(GearWorker {
            player_id: (1 + (index + 1) % 3) as i64,
            dummy: false,
        });
    }
    let snapshot = inspect(&state);
    assert_eq!(get_placement_cost(&state, "yaxchilan"), Some(3));
    assert_eq!(snapshot["placementCosts"]["yaxchilan"], 0);
    assert_eq!(place(&snapshot, "yaxchilan")["disabled"], false);
    let discount = snapshot["moves"]
        .as_array()
        .unwrap()
        .iter()
        .find(|choice| choice["id"] == "tribeAbility:discount:yaxchilan")
        .unwrap();
    assert_eq!(discount["disabled"], true);
    assert_eq!(
        discount["description"],
        "コーン 1 · この手番の配置割引を使用"
    );
    assert!(
        apply_move(
            &state,
            GameMove::TribeAbility {
                ability: "discount:yaxchilan".into()
            },
        )
        .is_err()
    );
    let placed = apply_move(
        &state,
        GameMove::Place {
            gear: GearId::Yaxchilan,
        },
    )
    .unwrap();
    assert_eq!(placed.players[0].resources[&Resource::Corn], 0);
    assert_eq!(
        placed.gears[&GearId::Yaxchilan][3]
            .as_ref()
            .unwrap()
            .player_id,
        0
    );
}
