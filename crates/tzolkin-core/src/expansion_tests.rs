use crate::quick_actions::{QuickActionId, QuickActionState};
use crate::tribes::{self, TribeId};
use crate::types::*;
use crate::{apply_move, create_game_with_options, get_available_moves, get_choices};

fn setup(mut state: GameState) -> GameState {
    for _ in 0..500 {
        if state.phase != Phase::Setup {
            return state;
        }
        let selected = get_choices(&state)
            .into_iter()
            .find(|c| c.disabled != Some(true))
            .expect("setup has a legal choice");
        state = apply_move(&state, selected.r#move).unwrap();
    }
    panic!("setup did not finish");
}
fn game(tribe: TribeId, quick: bool) -> GameState {
    let mut state = setup(
        create_game_with_options(
            vec!["A".into(), "B".into()],
            13,
            GameOptions {
                tribes: true,
                quick_actions: quick,
                ..GameOptions::default()
            },
        )
        .unwrap(),
    );
    state.players[0].tribe = Some(tribe);
    state.current_player = 0;
    state.turn = Turn::default();
    for r in RESOURCE_IDS {
        state.players[0]
            .resources
            .insert(r, if r == Resource::Skull { 1 } else { 30 });
    }
    for g in GEAR_IDS {
        state.gears.get_mut(&g).unwrap().fill(None);
    }
    state
}
fn choose(state: &GameState, id: &str) -> GameState {
    apply_move(
        state,
        GameMove::Choose {
            choice_id: id.into(),
        },
    )
    .unwrap()
}
fn worker(state: &mut GameState, gear: GearId, pos: usize) {
    state.gears.get_mut(&gear).unwrap()[pos] = Some(GearWorker {
        player_id: 0,
        dummy: false,
    });
}

#[test]
fn five_players_force_quick_actions_and_expand_setup() {
    let state = create_game_with_options(
        (0..5).map(|i| format!("P{i}")).collect(),
        12,
        GameOptions::default(),
    )
    .unwrap();
    assert_eq!(state.version, 2);
    assert_eq!(state.players[4].color, "#8b6095");
    assert!(state.jungle.values().all(|b| b.corn == 5));
    assert_eq!(state.jungle[&3].wood, 5);
    assert_eq!(state.monuments.len(), 7);
    assert_eq!(
        state
            .expansion
            .as_ref()
            .unwrap()
            .quick_actions
            .as_ref()
            .unwrap()
            .spaces,
        vec![None; 3]
    );
    assert_eq!(setup(state).phase, Phase::Playing);
}
#[test]
fn tribe_deal_uses_all_thirteen_unique_tiles() {
    assert_eq!(tribes::definitions().len(), 13);
    let state = create_game_with_options(
        (0..5).map(|i| format!("P{i}")).collect(),
        31,
        GameOptions {
            tribes: true,
            ..GameOptions::default()
        },
    )
    .unwrap();
    let mut offered: Vec<_> = state
        .players
        .iter()
        .flat_map(|p| p.tribe_offer.iter().copied())
        .collect();
    offered.sort();
    offered.dedup();
    assert_eq!(offered.len(), 10);
    assert!(
        get_choices(&state)
            .iter()
            .all(|c| c.id.starts_with("tribe:"))
    );
}
#[test]
fn quick_games_place_official_dummy_counts_even_after_discarded_wealth() {
    for count in 2..=4 {
        for seed in 0..30 {
            let state = setup(
                create_game_with_options(
                    (0..count).map(|i| format!("P{i}")).collect(),
                    seed,
                    GameOptions {
                        quick_actions: true,
                        ..GameOptions::default()
                    },
                )
                .unwrap(),
            );
            let expansion = state.expansion.as_ref().unwrap();
            let q = expansion.quick_actions.as_ref().unwrap();
            let quick_blockers = q.spaces.iter().filter(|id| **id == Some(-1)).count();
            let gear_blockers = state
                .gears
                .values()
                .flat_map(|slots| slots.iter().flatten())
                .filter(|w| w.dummy)
                .count();
            assert_eq!(quick_blockers, if count == 2 { 2 } else { 1 });
            assert_eq!(
                quick_blockers + gear_blockers,
                (5 - count) * 6,
                "seed={seed},players={count}"
            );
            assert_eq!(expansion.deferred_dummy_workers, 0);
        }
    }
}
#[test]
fn quick_decks_and_calendar_match_the_official_tiles() {
    assert_eq!(
        QuickActionId::AGE1
            .iter()
            .filter(|id| **id == QuickActionId::Technology)
            .count(),
        2
    );
    assert!(!QuickActionId::AGE1.contains(&QuickActionId::Gold));
    assert!(!QuickActionId::AGE2.contains(&QuickActionId::WoodCorn));
    let mut q = QuickActionState {
        age1: QuickActionId::AGE1.to_vec(),
        age2: QuickActionId::AGE2.to_vec(),
        current: QuickActionId::Corn,
        spaces: vec![Some(-1), Some(0), None],
        resolved: true,
    };
    q.update(14);
    assert_eq!(q.current, QuickActionId::Build);
    q.update(15);
    assert_eq!(q.current, QuickActionId::Corn);
    q.update(25);
    assert_eq!(q.current, QuickActionId::Build);
    q.update(27);
    assert_eq!(q.current, QuickActionId::Build);
    q.clear_workers();
    assert_eq!(q.spaces, vec![Some(-1), None, None]);
}
#[test]
fn quick_resource_action_is_delayed_until_placement_finishes_and_ignores_technology() {
    let mut state = game(TribeId::Ahmakiq, true);
    state.players[0]
        .technologies
        .insert(TechnologyId::Extraction, 3);
    state
        .expansion
        .as_mut()
        .unwrap()
        .quick_actions
        .as_mut()
        .unwrap()
        .current = QuickActionId::WoodCorn;
    let original_wood = state.players[0].resources[&Resource::Wood];
    let placed = apply_move(&state, GameMove::QuickAction).unwrap();
    assert_eq!(placed.players[0].resources[&Resource::Wood], original_wood);
    assert_eq!(crate::available_workers(&placed, 0), 2);
    assert!(apply_move(&placed, GameMove::QuickAction).is_err());
    let done = apply_move(
        &placed,
        GameMove::EndTurn {
            double_advance: None,
        },
    )
    .unwrap();
    assert_eq!(
        done.players[0].resources[&Resource::Wood],
        original_wood + 1
    );
    assert_eq!(
        done.expansion
            .as_ref()
            .unwrap()
            .quick_actions
            .as_ref()
            .unwrap()
            .spaces
            .iter()
            .filter(|id| **id == Some(0))
            .count(),
        1
    );
}
#[test]
fn quick_build_has_no_architecture_discount_and_cannot_be_declined() {
    let mut state = game(TribeId::Ahmakiq, true);
    state.players[0]
        .technologies
        .insert(TechnologyId::Architecture, 3);
    state
        .expansion
        .as_mut()
        .unwrap()
        .quick_actions
        .as_mut()
        .unwrap()
        .current = QuickActionId::Build;
    let state = apply_move(&state, GameMove::QuickAction).unwrap();
    let state = apply_move(
        &state,
        GameMove::EndTurn {
            double_advance: None,
        },
    )
    .unwrap();
    assert!(matches!(
        state.pending.as_ref().unwrap().task,
        Task::Build {
            architecture_available: Some(false),
            ..
        }
    ));
    let choices = get_choices(&state);
    assert!(!choices.iter().any(|c| c.id == "skip"));
    assert!(
        choices
            .iter()
            .all(|c| c.id.split(':').nth(2) == Some("none"))
    );
}
#[test]
fn ah_chuy_kak_picks_up_only_one_preexisting_worker_after_two_gear_placements() {
    let mut state = game(TribeId::AhChuyKak, false);
    state.players[0].workers = 5;
    worker(&mut state, GearId::Yaxchilan, 1);
    worker(&mut state, GearId::Tikal, 1);
    let state = apply_move(
        &state,
        GameMove::Place {
            gear: GearId::Palenque,
        },
    )
    .unwrap();
    assert!(
        apply_move(
            &state,
            GameMove::Remove {
                gear: GearId::Yaxchilan,
                position: 1
            }
        )
        .is_err()
    );
    let state = apply_move(
        &state,
        GameMove::Place {
            gear: GearId::Uxmal,
        },
    )
    .unwrap();
    assert!(
        apply_move(
            &state,
            GameMove::Remove {
                gear: GearId::Uxmal,
                position: 0
            }
        )
        .is_err()
    );
    let state = apply_move(
        &state,
        GameMove::Remove {
            gear: GearId::Yaxchilan,
            position: 1,
        },
    )
    .unwrap();
    let state = choose(&state, "action:1");
    assert_eq!(state.turn.mode, TurnMode::Place);
    assert!(
        apply_move(
            &state,
            GameMove::Remove {
                gear: GearId::Tikal,
                position: 1
            }
        )
        .is_err()
    );
}
#[test]
fn ahau_chamahez_combines_with_theology_and_reaches_any_action_spaces() {
    let mut state = game(TribeId::AhauChamahez, false);
    state.players[0]
        .technologies
        .insert(TechnologyId::Theology, 1);
    worker(&mut state, GearId::ChichenItza, 7);
    let state = apply_move(
        &state,
        GameMove::Remove {
            gear: GearId::ChichenItza,
            position: 7,
        },
    )
    .unwrap();
    assert!(
        get_choices(&state)
            .iter()
            .any(|c| c.id == "tribeAhead:9" && c.disabled != Some(true))
    );
    let mut state = game(TribeId::AhauChamahez, false);
    worker(&mut state, GearId::Tikal, 5);
    let state = apply_move(
        &state,
        GameMove::Remove {
            gear: GearId::Tikal,
            position: 5,
        },
    )
    .unwrap();
    let state = choose(&state, "tribeAhead:6");
    assert!(matches!(
        state.pending.unwrap().task,
        Task::Action {
            position: 6,
            free: Some(true),
            ..
        }
    ));
}
#[test]
fn balam_gains_one_corn_total_even_from_a_free_any_action_space() {
    let mut state = game(TribeId::Balam, false);
    worker(&mut state, GearId::Yaxchilan, 6);
    let corn = state.players[0].resources[&Resource::Corn];
    let state = apply_move(
        &state,
        GameMove::Remove {
            gear: GearId::Yaxchilan,
            position: 6,
        },
    )
    .unwrap();
    let state = choose(&state, "action:1");
    assert_eq!(state.players[0].resources[&Resource::Corn], corn + 1);
}
#[test]
fn huracan_uses_the_paired_city_including_lower_numbered_actions() {
    let mut state = game(TribeId::Huracan, false);
    worker(&mut state, GearId::Palenque, 4);
    let corn = state.players[0].resources[&Resource::Corn];
    let gold = state.players[0].resources[&Resource::Gold];
    let state = apply_move(
        &state,
        GameMove::Remove {
            gear: GearId::Palenque,
            position: 4,
        },
    )
    .unwrap();
    let state = choose(&state, "other:3");
    assert_eq!(state.players[0].resources[&Resource::Gold], gold + 1);
    assert_eq!(state.players[0].resources[&Resource::Corn], corn - 1 + 2);
}
#[test]
fn itzamna_gets_a_free_first_level_and_chooses_any_final_bonus() {
    let mut state = game(TribeId::Itzamna, false);
    for r in MATERIALS {
        state.players[0].resources.insert(r, 0);
    }
    worker(&mut state, GearId::Tikal, 1);
    let state = apply_move(
        &state,
        GameMove::Remove {
            gear: GearId::Tikal,
            position: 1,
        },
    )
    .unwrap();
    let state = choose(&state, "action:1");
    let state = choose(&state, "tech:agriculture");
    assert_eq!(state.players[0].technologies[&TechnologyId::Agriculture], 1);
    let mut state = game(TribeId::Itzamna, false);
    state.players[0]
        .technologies
        .insert(TechnologyId::Agriculture, 3);
    worker(&mut state, GearId::Tikal, 1);
    let state = apply_move(
        &state,
        GameMove::Remove {
            gear: GearId::Tikal,
            position: 1,
        },
    )
    .unwrap();
    let state = choose(&state, "action:1");
    let state = choose(&state, "tech:agriculture");
    let payment = get_choices(&state)
        .into_iter()
        .find(|c| c.disabled != Some(true))
        .unwrap();
    let state = apply_move(&state, payment.r#move).unwrap();
    assert!(matches!(
        state.pending.as_ref().unwrap().task,
        Task::TechnologyBonus
    ));
    let skulls = state.players[0].resources[&Resource::Skull];
    let state = choose(&state, "bonus:theology");
    assert_eq!(state.players[0].resources[&Resource::Skull], skulls + 1);
}
#[test]
fn vacub_caquiqx_skips_the_same_space_for_all_placements_in_that_turn() {
    let state = game(TribeId::VacubCaquix, false);
    let state = apply_move(
        &state,
        GameMove::TribeAbility {
            ability: "skipSpace".into(),
        },
    )
    .unwrap();
    let state = choose(&state, "space:palenque:0");
    let state = apply_move(
        &state,
        GameMove::Place {
            gear: GearId::Palenque,
        },
    )
    .unwrap();
    let state = apply_move(
        &state,
        GameMove::Place {
            gear: GearId::Palenque,
        },
    )
    .unwrap();
    assert!(state.gears[&GearId::Palenque][0].is_none());
    assert_eq!(
        state.gears[&GearId::Palenque][1]
            .as_ref()
            .unwrap()
            .player_id,
        0
    );
    assert_eq!(
        state.gears[&GearId::Palenque][2]
            .as_ref()
            .unwrap()
            .player_id,
        0
    );
}
#[test]
fn xaman_ek_can_trade_during_a_pending_action_once_per_turn() {
    let mut state = game(TribeId::XamanEk, false);
    worker(&mut state, GearId::Uxmal, 1);
    state.players[0].resources.insert(Resource::Corn, 0);
    let state = apply_move(
        &state,
        GameMove::Remove {
            gear: GearId::Uxmal,
            position: 1,
        },
    )
    .unwrap();
    assert!(
        get_available_moves(&state)
            .iter()
            .any(|m| matches!(m.r#move, GameMove::TribeAbility { .. }))
    );
    let pending = state.pending.clone();
    let state = apply_move(
        &state,
        GameMove::TribeAbility {
            ability: "sell:gold".into(),
        },
    )
    .unwrap();
    assert_eq!(state.pending, pending);
    assert_eq!(state.players[0].resources[&Resource::Corn], 4);
    assert!(
        apply_move(
            &state,
            GameMove::TribeAbility {
                ability: "sell:gold".into()
            }
        )
        .is_err()
    );
    let _ = choose(&state, "action:1");
}
#[test]
fn ahmakiq_can_pass_and_yumkaax_uses_the_printed_cost_table() {
    let state = game(TribeId::Ahmakiq, false);
    assert!(
        apply_move(
            &state,
            GameMove::EndTurn {
                double_advance: None
            }
        )
        .is_ok()
    );
    let state = game(TribeId::Yumkaax, false);
    let totals: Vec<i64> = (1..=6)
        .map(|n| {
            (0..n)
                .map(|placed| tribes::placement_surcharge(&state.players[0], placed))
                .sum()
        })
        .collect();
    assert_eq!(totals, [0, 0, 2, 5, 9, 13]);
}
#[test]
fn bacab_starts_with_two_corn_and_begging_reaches_four() {
    let mut state = game(TribeId::Bacab, false);
    state.players[0].resources.insert(Resource::Corn, 0);
    state.players[0].temples = TEMPLE_IDS.into_iter().map(|t| (t, 0)).collect();
    state.current_player = 1;
    state.turn_index = 1;
    state.turn.mode = TurnMode::Remove;
    state.turn.count = 1;
    let state = apply_move(
        &state,
        GameMove::EndTurn {
            double_advance: None,
        },
    )
    .unwrap();
    assert_eq!(state.current_player, 0);
    assert_eq!(state.players[0].resources[&Resource::Corn], 2);
    let state = apply_move(&state, GameMove::Beg).unwrap();
    let state = choose(&state, "temple:chaac");
    assert_eq!(state.players[0].resources[&Resource::Corn], 4);
    assert_eq!(state.players[0].temples[&TempleId::Chaac], -1);
}
#[test]
fn cit_bolon_tum_discount_is_once_per_turn_and_worker_survives_extra_space() {
    let mut state = game(TribeId::CitBolonTum, false);
    for gear in [GearId::Palenque, GearId::Uxmal] {
        for slot in state.gears.get_mut(&gear).unwrap().iter_mut().take(5) {
            *slot = Some(GearWorker {
                player_id: -1,
                dummy: true,
            });
        }
    }
    let state = apply_move(
        &state,
        GameMove::TribeAbility {
            ability: "discount:palenque".into(),
        },
    )
    .unwrap();
    assert_eq!(state.players[0].resources[&Resource::Corn], 27);
    assert!(
        apply_move(
            &state,
            GameMove::TribeAbility {
                ability: "discount:uxmal".into()
            }
        )
        .is_err()
    );
    let state = apply_move(
        &state,
        GameMove::Place {
            gear: GearId::Uxmal,
        },
    )
    .unwrap();
    assert_eq!(state.players[0].resources[&Resource::Corn], 21);
    let mut state = game(TribeId::CitBolonTum, false);
    worker(&mut state, GearId::Palenque, 6);
    state.current_player = 1;
    state.turn_index = 1;
    state.turn.mode = TurnMode::Place;
    state.turn.count = 1;
    state.first_player_claimed = Some(1);
    let state = apply_move(
        &state,
        GameMove::EndTurn {
            double_advance: Some(true),
        },
    )
    .unwrap();
    assert_eq!(
        state.gears[&GearId::Palenque][8]
            .as_ref()
            .unwrap()
            .player_id,
        0
    );
    let mut state = state;
    state.current_player = 0;
    state.turn_index = 1;
    let state = apply_move(
        &state,
        GameMove::Remove {
            gear: GearId::Palenque,
            position: 8,
        },
    )
    .unwrap();
    assert!(get_choices(&state).iter().any(|c| c.id == "action:5"));
}
#[test]
fn ixtab_chooses_three_wealth_then_loses_a_temple_step_and_yaluk_starts_with_five() {
    for tribe in [TribeId::Ixtab, TribeId::Yaluk] {
        let mut state = create_game_with_options(
            vec!["A".into(), "B".into()],
            99,
            GameOptions {
                tribes: true,
                ..GameOptions::default()
            },
        )
        .unwrap();
        state.players[0].tribe_offer = vec![tribe, TribeId::Balam];
        state.players[0].wealth_offer = ["w04", "w08", "w10", "w12"]
            .into_iter()
            .map(str::to_string)
            .collect();
        let state = choose(&state, &format!("tribe:{tribe}"));
        assert_eq!(
            state.players[0].workers,
            if tribe == TribeId::Yaluk { 5 } else { 3 }
        );
        let id = if tribe == TribeId::Ixtab {
            "wealth:w04:w08:w10"
        } else {
            "wealth:w04:w08"
        };
        let state = choose(&state, id);
        if tribe == TribeId::Ixtab {
            assert_eq!(state.players[0].wealth.len(), 3);
            assert!(matches!(
                state.pending.as_ref().unwrap().task,
                Task::Temple {
                    direction: Some(-1),
                    ..
                }
            ));
            let state = choose(&state, "temple:kukulkan");
            assert_eq!(state.players[0].temples[&TempleId::Kukulkan], -1);
        } else {
            assert_eq!(state.players[0].wealth.len(), 2);
        }
    }
}
#[test]
fn yaluk_requires_an_extra_corn_and_loses_five_per_unfed_worker() {
    let mut state = game(TribeId::Yaluk, false);
    state.players[0].workers = 5;
    state.players[0].feed_all = false;
    state.players[0].feed_discount = 0;
    state.players[0].feed_workers = 0;
    state.players[0].resources.insert(Resource::Corn, 14);
    state.players[0].temples = TEMPLE_IDS.into_iter().map(|t| (t, -1)).collect();
    state.players[0].score = 0.0;
    state.round = 8;
    state.current_player = 1;
    state.turn_index = 1;
    state.turn.mode = TurnMode::Remove;
    state.turn.count = 1;
    let state = apply_move(
        &state,
        GameMove::EndTurn {
            double_advance: None,
        },
    )
    .unwrap();
    assert_eq!(state.players[0].resources[&Resource::Corn], 2);
    assert_eq!(state.players[0].score, -5.0);
}

fn scarce_quick_game(
    tribe: TribeId,
    tile: QuickActionId,
    corn: i64,
    materials: &[(Resource, i64)],
) -> GameState {
    let mut state = game(tribe, true);
    for r in RESOURCE_IDS {
        state.players[0].resources.insert(r, 0);
    }
    for t in TECHNOLOGY_IDS {
        state.players[0].technologies.insert(t, 0);
    }
    state.players[0].resources.insert(Resource::Corn, corn);
    for &(r, n) in materials {
        state.players[0].resources.insert(r, n);
    }
    state
        .expansion
        .as_mut()
        .unwrap()
        .quick_actions
        .as_mut()
        .unwrap()
        .current = tile;
    state
}

#[test]
fn quick_build_optional_nested_build_and_taxed_technology_can_be_declined() {
    for (building, materials, teacher) in [
        ("b08", vec![(Resource::Wood, 1), (Resource::Gold, 1)], false),
        ("b23", vec![(Resource::Wood, 3)], true),
    ] {
        let mut state = scarce_quick_game(TribeId::Ahmakiq, QuickActionId::Build, 2, &materials);
        state.buildings = vec![building.into(), "b01".into()];
        if teacher {
            let expansion = state.expansion.as_mut().unwrap();
            expansion.prophecies = vec![crate::prophecies::ProphecyId::TeacherShortage];
            expansion.active_prophecy = Some(0);
        }
        let state = apply_move(&state, GameMove::QuickAction).unwrap();
        let state = apply_move(
            &state,
            GameMove::EndTurn {
                double_advance: None,
            },
        )
        .unwrap();
        assert!(!get_choices(&state).iter().any(|c| c.id == "skip"));
        let state = choose(&state, &format!("build:{building}:none"));
        let state = if building == "b08" {
            choose(&state, "temple:chaac")
        } else {
            state
        };
        assert!(
            get_choices(&state)
                .iter()
                .any(|c| c.id == "skip" && c.disabled != Some(true))
        );
        let state = choose(&state, "skip");
        assert_eq!(state.current_player, 1);
        assert!(state.pending.is_none());
        assert_eq!(state.players[0].buildings, vec![building]);
        if teacher {
            assert!(state.players[0].technologies.values().all(|n| *n == 0));
        }
    }
}

#[test]
fn xaman_cannot_sell_the_last_resource_required_by_an_existing_payment() {
    for position in [1, 5] {
        let mut state = game(TribeId::XamanEk, false);
        for r in RESOURCE_IDS {
            state.players[0].resources.insert(r, 0);
        }
        state.players[0].resources.insert(Resource::Wood, 1);
        worker(&mut state, GearId::Tikal, position);
        let state = apply_move(
            &state,
            GameMove::Remove {
                gear: GearId::Tikal,
                position: position as i64,
            },
        )
        .unwrap();
        let state = choose(&state, &format!("action:{position}"));
        let state = if position == 1 {
            choose(&state, "tech:agriculture")
        } else {
            state
        };
        assert!(matches!(
            state.pending.as_ref().unwrap().task,
            Task::PayTechnology { .. } | Task::PayResource { .. }
        ));
        let original = state.clone();
        assert!(
            apply_move(
                &state,
                GameMove::TribeAbility {
                    ability: "sell:wood".into()
                }
            )
            .is_err()
        );
        assert_eq!(state, original);
        let payment = get_choices(&state)
            .into_iter()
            .find(|c| c.disabled != Some(true))
            .unwrap();
        let state = apply_move(&state, payment.r#move).unwrap();
        assert_eq!(state.players[0].resources[&Resource::Wood], 0);
    }
}

#[test]
fn ah_chuy_cannot_start_a_payment_that_consumes_reserved_quick_technology_cost() {
    let mut state = scarce_quick_game(
        TribeId::AhChuyKak,
        QuickActionId::Technology,
        10,
        &[(Resource::Wood, 1)],
    );
    state.players[0].workers = 4;
    worker(&mut state, GearId::Tikal, 5);
    let state = apply_move(&state, GameMove::QuickAction).unwrap();
    let state = apply_move(
        &state,
        GameMove::Place {
            gear: GearId::Palenque,
        },
    )
    .unwrap();
    let state = apply_move(
        &state,
        GameMove::Place {
            gear: GearId::Yaxchilan,
        },
    )
    .unwrap();
    let state = apply_move(
        &state,
        GameMove::Remove {
            gear: GearId::Tikal,
            position: 5,
        },
    )
    .unwrap();
    assert!(
        get_choices(&state)
            .iter()
            .any(|c| c.id == "action:5" && c.disabled == Some(true))
    );
    assert!(
        apply_move(
            &state,
            GameMove::Choose {
                choice_id: "action:5".into()
            }
        )
        .is_err()
    );
    let state = choose(&state, "skip");
    let state = apply_move(
        &state,
        GameMove::EndTurn {
            double_advance: None,
        },
    )
    .unwrap();
    let state = choose(&state, "tech:extraction");
    let payment = get_choices(&state)
        .into_iter()
        .find(|c| c.disabled != Some(true))
        .unwrap();
    let state = apply_move(&state, payment.r#move).unwrap();
    assert_eq!(state.players[0].technologies[&TechnologyId::Extraction], 1);
    assert_eq!(state.current_player, 1);
}

#[test]
fn cit_reserved_quick_action_can_enumerate_and_apply_legal_discount_moves() {
    let state = scarce_quick_game(TribeId::CitBolonTum, QuickActionId::Corn, 10, &[]);
    let state = apply_move(&state, GameMove::QuickAction).unwrap();
    let moves = get_available_moves(&state);
    assert!(
        moves
            .iter()
            .any(|c| c.id == "tribeAbility:discount:palenque" && c.disabled != Some(true))
    );
    for mv in moves.into_iter().filter(|c| c.disabled != Some(true)) {
        assert!(
            apply_move(&state, mv.r#move).is_ok(),
            "enabled move {} must be applicable",
            mv.id
        );
    }
}

#[test]
fn a_temple_bonus_before_a_stone_reward_can_restore_reserved_quick_payment() {
    let mut state = scarce_quick_game(
        TribeId::AhChuyKak,
        QuickActionId::Technology,
        10,
        &[(Resource::Wood, 3)],
    );
    state.players[0].workers = 4;
    state.players[0]
        .technologies
        .insert(TechnologyId::Agriculture, 3);
    state.buildings = vec!["b10".into()];
    worker(&mut state, GearId::Tikal, 2);
    let state = apply_move(&state, GameMove::QuickAction).unwrap();
    let state = apply_move(
        &state,
        GameMove::Place {
            gear: GearId::Palenque,
        },
    )
    .unwrap();
    let state = apply_move(
        &state,
        GameMove::Place {
            gear: GearId::Yaxchilan,
        },
    )
    .unwrap();
    let state = apply_move(
        &state,
        GameMove::Remove {
            gear: GearId::Tikal,
            position: 2,
        },
    )
    .unwrap();
    let state = choose(&state, "action:2");
    assert!(
        get_choices(&state)
            .iter()
            .any(|c| c.id == "build:b10:none" && c.disabled != Some(true))
    );
    let state = choose(&state, "build:b10:none");
    assert!(matches!(
        state.pending.as_ref().unwrap().task,
        Task::Temple { .. }
    ));
    assert_eq!(state.players[0].resources[&Resource::Stone], 0);
    let state = choose(&state, "temple:chaac");
    assert_eq!(state.players[0].resources[&Resource::Stone], 1);
    let state = apply_move(
        &state,
        GameMove::EndTurn {
            double_advance: None,
        },
    )
    .unwrap();
    let state = choose(&state, "tech:extraction");
    let payment = get_choices(&state)
        .into_iter()
        .find(|c| c.disabled != Some(true))
        .unwrap();
    let state = apply_move(&state, payment.r#move).unwrap();
    assert_eq!(state.players[0].technologies[&TechnologyId::Extraction], 1);
    assert_eq!(state.players[0].resources[&Resource::Stone], 0);
    assert_eq!(state.current_player, 1);
}
