use serde::Deserialize;
use std::collections::BTreeSet;
use tzolkin_core::catalog::CATALOG;
use tzolkin_core::compact::catalog::{BUILDING_IDS, MONUMENT_IDS, WEALTH_IDS};
use tzolkin_core::compact::{
    BuildingId, CompactGame, CompactState, DUMMY_OWNER, EMPTY_OWNER, FixedList, GEAR_LENGTHS,
    GEAR_OFFSETS, MonumentId, ScoreQuarters, WealthId,
};
use tzolkin_core::prophecies::{FoodDayStage, ProphecyId};
use tzolkin_core::quick_actions::QuickActionId;
use tzolkin_core::tribes::TribeId;
use tzolkin_core::*;

fn enabled(state: &GameState) -> Vec<GameMove> {
    get_available_moves(state)
        .into_iter()
        .filter(|choice| choice.disabled != Some(true))
        .map(|choice| choice.r#move)
        .collect()
}

fn monument_payment_state(dedicated: bool, supply: i64) -> GameState {
    let mut state = create_game((0..4).map(|i| format!("P{i}")).collect(), 42, dedicated).unwrap();
    while state.phase == Phase::Setup {
        let operation = enabled(&state).remove(0);
        state = apply_move(&state, operation).unwrap();
    }
    if !state.monuments.iter().any(|id| id == "m13") {
        state.monuments[0] = "m13".into();
    }
    for player in &mut state.players {
        player.resources.insert(Resource::Skull, 0);
    }
    state.players[0].resources.insert(Resource::Skull, 1);
    state.players[1]
        .resources
        .insert(Resource::Skull, 12 - supply);
    for resource in [Resource::Corn, Resource::Wood, Resource::Stone] {
        state.players[0].resources.insert(resource, 10);
    }
    state.players[0].resources.insert(Resource::Gold, 4);
    state.skull_supply = supply;
    state.turn.mode = TurnMode::Remove;
    state.turn.count = 1;
    state.pending = Some(Pending {
        title: "Synthetic monument payment".into(),
        task: if dedicated {
            Task::BuildMonument
        } else {
            Task::Build {
                remaining: 2,
                allow_monument: true,
                corn_payment: false,
                architecture_available: None,
                mandatory: false,
            }
        },
        after: vec![],
    });
    assert_state(&state);
    state
}

#[test]
fn monument_skull_payments_stay_outside_the_bank_in_both_build_tasks() {
    for dedicated in [false, true] {
        for supply in [0, 4] {
            let before = monument_payment_state(dedicated, supply);
            let operation = GameMove::Choose {
                choice_id: "monument:m13".into(),
            };
            let compact = assert_state(&before);
            let after = assert_transition(&before, &compact, operation.clone());
            assert_eq!(after.skull_supply, supply);
            assert_eq!(after.players[0].resources[&Resource::Skull], 0);
            assert_eq!(after.players[0].monuments, ["m13"]);
            assert!(!after.monuments.iter().any(|id| id == "m13"));
            assert_eq!(after.players[0].building_skulls, 0);
            assert_eq!(after.players[0].skulls_placed, 0);
            assert_eq!(score_monument(&after, &after.players[0], "m13"), 0.0);
            assert_state(&after);
            if !dedicated {
                use tzolkin_core::public_replay::*;
                let initial = PublicState::from_game_state(&before).unwrap();
                let frame = apply_public(&initial, 0, operation, &Refills::default()).unwrap();
                assert_eq!(frame.snapshot.state.skull_supply, supply);
                validate_public_state(&frame.snapshot.state).unwrap();
            }
        }
    }
}

#[test]
fn paid_monument_skulls_cannot_fund_later_harvest_or_capped_theology() {
    let state = monument_payment_state(false, 0);
    let mut state = apply_move(
        &state,
        GameMove::Choose {
            choice_id: "monument:m13".into(),
        },
    )
    .unwrap();
    for task in [
        Task::Action {
            gear: GearId::Yaxchilan,
            position: 4,
            free: None,
        },
        Task::Technology {
            remaining: 1,
            free: true,
            mandatory: false,
        },
    ] {
        state.players[0]
            .technologies
            .insert(TechnologyId::Theology, 3);
        state.pending = Some(Pending {
            title: "Synthetic empty-bank reward".into(),
            task,
            after: vec![],
        });
        let choice = if matches!(state.pending.as_ref().unwrap().task, Task::Action { .. }) {
            "action:4"
        } else {
            "tech:theology"
        };
        let compact = assert_state(&state);
        state = assert_transition(
            &state,
            &compact,
            GameMove::Choose {
                choice_id: choice.into(),
            },
        );
        assert_eq!(state.skull_supply, 0);
        assert_eq!(state.players[0].resources[&Resource::Skull], 0);
    }
    // A real placement by a different player remains the source of allSkulls
    // points. The monument's paid skull must never count as a city placement.
    state.current_player = 1;
    state.turn_index = 1;
    state.players[1]
        .technologies
        .insert(TechnologyId::Theology, 0);
    state.pending = Some(Pending {
        title: "Synthetic city placement".into(),
        task: Task::Action {
            gear: GearId::ChichenItza,
            position: 1,
            free: None,
        },
        after: vec![],
    });
    let compact = assert_state(&state);
    state = assert_transition(
        &state,
        &compact,
        GameMove::Choose {
            choice_id: "action:1".into(),
        },
    );
    assert_eq!(state.skull_supply, 0);
    assert_eq!(state.players[1].skulls_placed, 1);
    assert_eq!(score_monument(&state, &state.players[0], "m13"), 3.0);
}

#[test]
fn monument_skull_costs_preserve_strict_conservation_and_reject_legacy_saves() {
    for dedicated in [false, true] {
        let before = monument_payment_state(dedicated, 0);
        let operation = GameMove::Choose {
            choice_id: "monument:m13".into(),
        };
        for missing in [Resource::Gold, Resource::Skull] {
            let mut insufficient = before.clone();
            if missing == Resource::Gold {
                insufficient.players[0].resources.insert(missing, 3);
            } else {
                insufficient.players[0].resources.insert(missing, 0);
                insufficient.players[1].resources.insert(missing, 13);
            }
            assert_state(&insufficient);
            let original = insufficient.clone();
            assert!(apply_move(&insufficient, operation.clone()).is_err());
            assert_eq!(insufficient, original);
        }
        let after = apply_move(&before, operation.clone()).unwrap();
        for restore_to_bank in [true, false] {
            let mut invalid = after.clone();
            if restore_to_bank {
                invalid.skull_supply += 1;
            } else {
                *invalid.players[0]
                    .resources
                    .get_mut(&Resource::Skull)
                    .unwrap() += 1;
            }
            assert!(!validation::validate_game_state(
                &serde_json::to_value(&invalid).unwrap()
            ));
        }
        assert!(apply_move(&after, operation).is_err());
        let mut ordinary = before;
        if !ordinary.monuments.iter().any(|id| id == "m01") {
            let index = ordinary
                .monuments
                .iter()
                .position(|id| id == "m13")
                .unwrap();
            ordinary.monuments[index] = "m01".into();
        }
        let compact = assert_state(&ordinary);
        let next = assert_transition(
            &ordinary,
            &compact,
            GameMove::Choose {
                choice_id: "monument:m01".into(),
            },
        );
        assert_eq!(next.skull_supply, 0);
        assert_eq!(next.players[0].resources[&Resource::Skull], 1);
    }
}

fn assert_masks(reference: &GameState, compact: &CompactState) {
    let mut occupied = [0u16; 5];
    let mut owners = [EMPTY_OWNER; 53];
    let mut player_masks = [0u64; 5];
    let mut dummy_mask = 0u64;
    for (gear_index, gear) in GEAR_IDS.into_iter().enumerate() {
        for (position, worker) in reference.gears[&gear].iter().enumerate() {
            if let Some(worker) = worker {
                let offset = GEAR_OFFSETS[gear_index] + position;
                occupied[gear_index] |= 1 << position;
                if worker.dummy {
                    owners[offset] = DUMMY_OWNER;
                    dummy_mask |= 1 << offset;
                } else {
                    owners[offset] = worker.player_id as u8;
                    player_masks[worker.player_id as usize] |= 1 << offset;
                }
            }
        }
        assert!(compact.gears().occupied()[gear_index] < (1 << GEAR_LENGTHS[gear_index]));
    }
    assert_eq!(*compact.gears().owners(), owners);
    assert_eq!(*compact.gears().occupied(), occupied);
    assert_eq!(*compact.gears().player_masks(), player_masks);
    assert_eq!(compact.gears().dummy_mask(), dummy_mask);
}

fn assert_state(reference: &GameState) -> CompactState {
    let converted = CompactGame::from_saved(reference).unwrap();
    assert_eq!(converted.to_saved(), *reference, "complete save roundtrip");
    assert_eq!(converted.state.actor(), reference.current_player);
    assert_eq!(
        converted.state.turn_owner(),
        reference.turn_order[reference.turn_index]
    );
    assert_eq!(
        converted.state.legal_moves(),
        enabled(reference),
        "legal semantic set/order"
    );
    for index in reference.players.len()..5 {
        assert!(
            converted.state.players()[index].is_none(),
            "unused seat normalization"
        );
    }
    assert_masks(reference, &converted.state);
    converted.state
}

fn assert_transition(
    reference: &GameState,
    compact: &CompactState,
    operation: GameMove,
) -> GameState {
    let expected = apply_move(reference, operation.clone()).unwrap();
    let actual = compact.apply_move(operation).unwrap();
    assert_eq!(
        actual,
        CompactState::from_saved(&expected).unwrap(),
        "rule transition parity"
    );
    assert_masks(&expected, &actual);
    expected
}

#[test]
fn exhausted_distinct_temple_keeps_building_attributes_in_compact_transitions() {
    let mut state = monument_payment_state(false, 4);
    state.players[0].temples.insert(TempleId::Chaac, 5);
    state.players[0].temples.insert(TempleId::Kukulkan, 6);
    state.players[0].temples.insert(TempleId::Quetzalcoatl, 2);
    let build = Task::Build {
        remaining: 1,
        allow_monument: false,
        corn_payment: true,
        architecture_available: Some(false),
        mandatory: false,
    };
    let tail = vec![build.clone(), Task::Trade];
    state.pending = Some(Pending {
        title: "Distinct temple advance".into(),
        task: Task::Temple {
            remaining: 2,
            distinct: Some(vec![]),
            direction: None,
            reason: None,
        },
        after: tail,
    });
    let before = state.clone();
    let compact = assert_state(&state);
    let after = assert_transition(
        &state,
        &compact,
        GameMove::Choose {
            choice_id: "temple:quetzalcoatl".into(),
        },
    );
    let pending = after.pending.as_ref().unwrap();
    assert_eq!(pending.task, build);
    assert_eq!(pending.after, [Task::Trade]);
    assert_eq!(after.players[0].temples[&TempleId::Quetzalcoatl], 3);
    assert_eq!(after.players[0].resources, before.players[0].resources);
    assert_eq!(after.players[0].score, before.players[0].score);
    assert_eq!(
        after.players[0].double_advance_available,
        before.players[0].double_advance_available
    );
    assert_state(&after);
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct ReferenceTraces {
    traces: Vec<ReferenceTrace>,
}
#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct ReferenceTrace {
    names: Vec<String>,
    seed: u32,
    additional_buildings: bool,
    steps: Vec<TraceStep>,
    final_state: GameState,
}
#[derive(Deserialize)]
struct TraceStep {
    r#move: GameMove,
}

#[test]
fn current_rule_reference_traces_preserve_saves_legal_sets_transitions_and_results() {
    let traces: ReferenceTraces = serde_json::from_str(include_str!(
        "../../../tests/fixtures/reference-traces.json"
    ))
    .unwrap();
    assert_eq!(traces.traces.len(), 6);
    for trace in traces.traces {
        let mut state = create_game(trace.names, trace.seed, trace.additional_buildings).unwrap();
        for step in trace.steps {
            let compact = assert_state(&state);
            assert!(compact.legal_moves().contains(&step.r#move));
            state = assert_transition(&state, &compact, step.r#move);
        }
        assert_state(&state);
        assert_eq!(state, trace.final_state);
        assert_eq!(state.phase, Phase::Finished);
    }
}

#[test]
fn reachable_states_for_every_player_count_and_independent_rule_flags_finish_with_parity() {
    // Additional buildings are an independent flag, as are the three expansion switches.
    for count in 2..=5 {
        for flags in 0..16 {
            let mut rng = 7919u32 + (count * 23 + flags) as u32;
            let mut state = create_game_with_options(
                (0..count).map(|seat| format!("P{seat}")).collect(),
                rng,
                GameOptions {
                    additional_buildings: flags & 1 != 0,
                    tribes: flags & 2 != 0,
                    prophecies: flags & 4 != 0,
                    quick_actions: flags & 8 != 0,
                },
            )
            .unwrap();
            for step in 0..1800 {
                let compact = assert_state(&state);
                assert_eq!(compact.flags().additional_buildings, flags & 1 != 0);
                assert_eq!(compact.flags().tribes, flags & 2 != 0);
                assert_eq!(compact.flags().prophecies, flags & 4 != 0);
                assert_eq!(compact.flags().quick_actions, flags & 8 != 0 || count == 5);
                if state.phase == Phase::Finished {
                    break;
                }
                let candidates = enabled(&state);
                assert!(
                    !candidates.is_empty(),
                    "count={count} flags={flags} step={step}"
                );
                // Examine all branches at regular checkpoints, including payments and continued tasks.
                if step % 31 == 0 {
                    for operation in &candidates {
                        assert_transition(&state, &compact, operation.clone());
                    }
                }
                rng ^= rng << 13;
                rng ^= rng >> 17;
                rng ^= rng << 5;
                let operation = candidates[(rng as usize) % candidates.len()].clone();
                state = assert_transition(&state, &compact, operation);
            }
            assert_eq!(state.phase, Phase::Finished, "count={count} flags={flags}");
            assert_state(&state);
            assert_eq!(state.final_scores.len(), count);
        }
    }
}

#[test]
fn catalog_ids_are_stable_bijective_and_independent_of_catalog_order() {
    let expected: BTreeSet<_> = CATALOG
        .all_buildings
        .iter()
        .map(|entry| entry.id.as_str())
        .collect();
    assert_eq!(expected, BUILDING_IDS.into_iter().collect());
    let expected: BTreeSet<_> = CATALOG
        .monuments
        .iter()
        .map(|entry| entry.id.as_str())
        .collect();
    assert_eq!(expected, MONUMENT_IDS.into_iter().collect());
    let expected: BTreeSet<_> = CATALOG
        .starting_wealth
        .iter()
        .map(|entry| entry.id.as_str())
        .collect();
    assert_eq!(expected, WEALTH_IDS.into_iter().collect());
    for (index, id) in BUILDING_IDS.into_iter().enumerate() {
        assert_eq!(id.parse::<BuildingId>().unwrap().index(), index);
        assert_eq!(BuildingId::from_index(index).unwrap().as_str(), id);
    }
    for (index, id) in MONUMENT_IDS.into_iter().enumerate() {
        assert_eq!(id.parse::<MonumentId>().unwrap().index(), index);
        assert_eq!(MonumentId::from_index(index).unwrap().as_str(), id);
    }
    for (index, id) in WEALTH_IDS.into_iter().enumerate() {
        assert_eq!(id.parse::<WealthId>().unwrap().index(), index);
        assert_eq!(WealthId::from_index(index).unwrap().as_str(), id);
    }
    assert!(BuildingId::from_index(40).is_err());
    assert!("b00".parse::<BuildingId>().is_err());
    assert!("m14".parse::<MonumentId>().is_err());
    assert!("w22".parse::<WealthId>().is_err());
    assert!(FixedList::<u8, 2>::try_from_iter([1, 2, 3]).is_err());
}

fn all_effects() -> Vec<Effect> {
    vec![
        Effect::Resources {
            resources: [(Resource::Corn, 7), (Resource::Skull, 1)].into(),
        },
        Effect::Feed {
            workers: FeedWorkers::Count(2),
        },
        Effect::Feed {
            workers: FeedWorkers::All(AllWorkers::All),
        },
        Effect::FeedDiscount { amount: 3 },
        Effect::Technology {
            technology: Target::Specific(TechnologyId::Architecture),
            steps: None,
        },
        Effect::Technology {
            technology: Target::Any(Any::Any),
            steps: Some(2),
        },
        Effect::Temple {
            temple: Target::Specific(TempleId::Chaac),
            steps: Some(2),
        },
        Effect::Temple {
            temple: Target::Any(Any::Any),
            steps: None,
        },
        Effect::Trade,
        Effect::Build,
        Effect::BuildMonument,
        Effect::Renovation,
        Effect::FoodReward {
            resources: [(Resource::Stone, 2)].into(),
        },
        Effect::FoodRewardSwitch,
        Effect::SkullBuilding {
            points: 7,
            temple: Target::Specific(TempleId::Kukulkan),
        },
        Effect::SkullBuilding {
            points: 3,
            temple: Target::Any(Any::Any),
        },
        Effect::TechnologyExchange,
        Effect::Points { amount: -5 },
        Effect::Worker,
        Effect::Action { anywhere: None },
        Effect::Action {
            anywhere: Some(false),
        },
        Effect::Action {
            anywhere: Some(true),
        },
    ]
}
fn task_fixtures() -> Vec<Task> {
    vec![
        Task::ChooseTribe,
        Task::TribeSkipSpace,
        Task::TechnologyBonus,
        Task::QuickAction {
            tile: QuickActionId::Technology,
        },
        Task::FinishTurn {
            double_advance: None,
        },
        Task::FinishTurn {
            double_advance: Some(false),
        },
        Task::FinishTurn {
            double_advance: Some(true),
        },
        Task::ProphecyGain {
            player_id: 1,
            resources: [(Resource::Gold, 3)].into(),
        },
        Task::ProphecyTemple {
            temple: TempleId::Chaac,
        },
        Task::FoodDay {
            day: 14,
            stage: FoodDayStage::Feeding,
            fed_workers: vec![3, 5],
        },
        Task::Effects {
            effects: all_effects(),
        },
        Task::Action {
            gear: GearId::Tikal,
            position: 4,
            free: None,
        },
        Task::Action {
            gear: GearId::Tikal,
            position: 4,
            free: Some(false),
        },
        Task::Action {
            gear: GearId::Tikal,
            position: 4,
            free: Some(true),
        },
        Task::Technology {
            remaining: 2,
            free: false,
            mandatory: true,
        },
        Task::PayTechnology {
            technology: TechnologyId::Theology,
            amount: 4,
            optional: true,
        },
        Task::PayResource { amount: 5 },
        Task::Temple {
            remaining: 2,
            distinct: Some(vec![TempleId::Chaac]),
            direction: Some(-1),
            reason: Some(TempleReason::Burn),
        },
        Task::Temple {
            remaining: 1,
            distinct: None,
            direction: None,
            reason: None,
        },
        Task::Resource { remaining: 3 },
        Task::Build {
            remaining: 2,
            allow_monument: true,
            corn_payment: true,
            architecture_available: None,
            mandatory: false,
        },
        Task::Build {
            remaining: 1,
            allow_monument: false,
            corn_payment: false,
            architecture_available: Some(false),
            mandatory: true,
        },
        Task::Build {
            remaining: 1,
            allow_monument: false,
            corn_payment: false,
            architecture_available: Some(true),
            mandatory: false,
        },
        Task::BuildMonument,
        Task::TechnologyExchange,
        Task::Trade,
        Task::AnyAction {
            exclude_skulls: None,
            cost: None,
        },
        Task::AnyAction {
            exclude_skulls: Some(false),
            cost: Some(1),
        },
        Task::AnyAction {
            exclude_skulls: Some(true),
            cost: Some(0),
        },
        Task::Palenque { position: 5 },
        Task::Theology { position: None },
        Task::Theology { position: Some(8) },
        Task::Rotation,
    ]
}
fn task_variant(task: &Task) -> u8 {
    match task {
        Task::ChooseTribe => 0,
        Task::TribeSkipSpace => 1,
        Task::TechnologyBonus => 2,
        Task::QuickAction { .. } => 3,
        Task::FinishTurn { .. } => 4,
        Task::ProphecyGain { .. } => 5,
        Task::ProphecyTemple { .. } => 6,
        Task::FoodDay { .. } => 7,
        Task::Effects { .. } => 8,
        Task::Action { .. } => 9,
        Task::Technology { .. } => 10,
        Task::PayTechnology { .. } => 11,
        Task::PayResource { .. } => 12,
        Task::Temple { .. } => 13,
        Task::Resource { .. } => 14,
        Task::Build { .. } => 15,
        Task::BuildMonument => 16,
        Task::TechnologyExchange => 17,
        Task::Trade => 18,
        Task::AnyAction { .. } => 19,
        Task::Palenque { .. } => 20,
        Task::Theology { .. } => 21,
        Task::Rotation => 22,
    }
}
fn effect_variant(effect: &Effect) -> u8 {
    match effect {
        Effect::Resources { .. } => 0,
        Effect::Feed { .. } => 1,
        Effect::FeedDiscount { .. } => 2,
        Effect::Technology { .. } => 3,
        Effect::Temple { .. } => 4,
        Effect::Trade => 5,
        Effect::Build => 6,
        Effect::BuildMonument => 7,
        Effect::Renovation => 8,
        Effect::FoodReward { .. } => 9,
        Effect::FoodRewardSwitch => 10,
        Effect::SkullBuilding { .. } => 11,
        Effect::TechnologyExchange => 12,
        Effect::Points { .. } => 13,
        Effect::Worker => 14,
        Effect::Action { .. } => 15,
    }
}

#[test]
fn every_typed_task_effect_and_optional_value_is_preserved_without_queue_truncation() {
    let mut reference = create_game_with_options(
        vec!["A".into(), "B".into()],
        42,
        GameOptions {
            additional_buildings: true,
            tribes: true,
            prophecies: true,
            quick_actions: true,
        },
    )
    .unwrap();
    let tasks = task_fixtures();
    assert_eq!(
        tasks.iter().map(task_variant).collect::<BTreeSet<_>>(),
        (0..23).collect()
    );
    assert_eq!(
        all_effects()
            .iter()
            .map(effect_variant)
            .collect::<BTreeSet<_>>(),
        (0..16).collect()
    );
    for task in &tasks {
        reference.pending = Some(Pending {
            title: "display-only task title".into(),
            task: task.clone(),
            after: tasks.clone(),
        });
        let converted = CompactGame::from_saved(&reference).unwrap();
        assert_eq!(converted.to_saved(), reference);
        assert_eq!(converted.state.pending().unwrap().after.len(), tasks.len());
        let without_presentation = converted.state.to_reference();
        assert!(without_presentation.log.is_empty());
        assert!(without_presentation.pending.unwrap().title.is_empty());
    }
    // Continuations can grow beyond imported-save bounds inside automatic rule execution.
    reference.pending.as_mut().unwrap().after = vec![Task::Rotation; 100];
    assert_eq!(
        CompactGame::from_saved(&reference).unwrap().to_saved(),
        reference
    );
}

#[test]
fn arrays_keep_large_resources_negative_temples_and_exact_quarter_results() {
    let mut reference = create_game(vec!["A".into(), "B".into()], 42, true).unwrap();
    reference.players[0]
        .resources
        .insert(Resource::Corn, 9_007_199_254_740_991);
    reference.players[0].temples.insert(TempleId::Chaac, -1);
    reference.players[0].score = -12.75;
    reference.final_scores = vec![FinalScore {
        player_id: 0,
        points_before_final: -12.75,
        resource_points: 0.25,
        skull_points: 3.0,
        monument_points: 6.5,
        total: -3.0,
        workers_on_gears: 2,
        rank: 2,
    }];
    assert_eq!(
        CompactGame::from_saved(&reference).unwrap().to_saved(),
        reference
    );
    assert_eq!(ScoreQuarters::from_points(-12.75).unwrap().quarters(), -51);
    assert!(ScoreQuarters::from_points(0.1).is_err());
    assert!(ScoreQuarters::from_points(f64::INFINITY).is_err());
    let max = ScoreQuarters::from_points(9_007_199_254_740_991.0 / 4.0).unwrap();
    assert!(
        max.checked_add(ScoreQuarters::from_points(0.25).unwrap())
            .is_err()
    );
}

#[test]
fn presentation_and_seed_are_outside_the_rule_copy() {
    let reference = create_game(vec!["A".into(), "B".into()], 42, false).unwrap();
    let compact = CompactState::from_saved(&reference).unwrap();
    let mut changed = reference.clone();
    changed.seed = 987654;
    changed.players[0].name = "different display name".into();
    changed.players[0].color = "#123456".into();
    changed.log = vec!["display history".into(); 500];
    assert_eq!(CompactState::from_saved(&changed).unwrap(), compact);
    assert_ne!(
        CompactGame::from_saved(&changed).unwrap().metadata,
        CompactGame::from_saved(&reference).unwrap().metadata
    );
}

#[test]
fn additional_jungle_save_keys_are_preserved_for_import_compatibility() {
    let mut reference = create_game(vec!["A".into(), "B".into()], 42, false).unwrap();
    reference.jungle.insert(99, JungleBox { corn: 7, wood: 3 });
    assert!(tzolkin_core::validation::validate_game_state(
        &serde_json::to_value(&reference).unwrap()
    ));
    assert_state(&reference);
}

#[test]
fn every_tribe_and_prophecy_variant_has_a_save_and_legal_transition_fixture() {
    for (index, tribe) in TribeId::ALL.into_iter().enumerate() {
        let mut state = create_game_with_options(
            vec!["A".into(), "B".into()],
            index as u32 + 42,
            GameOptions {
                tribes: true,
                ..Default::default()
            },
        )
        .unwrap();
        state.players[0].tribe_offer = vec![tribe, TribeId::ALL[(index + 1) % 13]];
        let compact = assert_state(&state);
        let operation = GameMove::Choose {
            choice_id: format!("tribe:{tribe}"),
        };
        assert!(compact.legal_moves().contains(&operation));
        state = assert_transition(&state, &compact, operation);
        assert_state(&state);
    }
    for (index, prophecy) in ProphecyId::ALL.into_iter().enumerate() {
        let mut state = create_game_with_options(
            vec!["A".into(), "B".into()],
            index as u32 + 42,
            GameOptions {
                prophecies: true,
                ..Default::default()
            },
        )
        .unwrap();
        state.expansion.as_mut().unwrap().prophecies = vec![
            prophecy,
            ProphecyId::ALL[(index + 1) % 13],
            ProphecyId::ALL[(index + 2) % 13],
        ];
        while state.phase == Phase::Setup {
            state = apply_move(&state, enabled(&state)[0].clone()).unwrap();
        }
        state.food_days = vec![8];
        state.round = 9;
        state.expansion.as_mut().unwrap().active_prophecy = Some(0);
        let compact = assert_state(&state);
        for operation in enabled(&state) {
            assert_transition(&state, &compact, operation);
        }
    }
}

#[test]
fn food_day_decision_actor_is_distinct_from_turn_owner_and_restores_on_rotation() {
    let mut state = create_game_with_options(
        vec!["A".into(), "B".into()],
        42,
        GameOptions {
            additional_buildings: true,
            prophecies: true,
            ..Default::default()
        },
    )
    .unwrap();
    while state.phase == Phase::Setup {
        state = apply_move(&state, enabled(&state)[0].clone()).unwrap();
    }
    state.round = 14;
    state.food_days = vec![8];
    state.current_player = 1;
    state.turn_index = 1;
    let expansion = state.expansion.as_mut().unwrap();
    expansion.prophecies = vec![
        ProphecyId::GoldShortage,
        ProphecyId::Drought,
        ProphecyId::Hunger,
    ];
    expansion.active_prophecy = Some(0);
    for player in &mut state.players {
        player.resources.insert(Resource::Corn, 30);
    }
    state.buildings.retain(|id| id != "b35");
    state.building_deck.retain(|id| id != "b35");
    state.players[0].buildings.push("b35".into());
    state = apply_move(
        &state,
        GameMove::Place {
            gear: GearId::Palenque,
        },
    )
    .unwrap();
    state = apply_move(
        &state,
        GameMove::EndTurn {
            double_advance: None,
        },
    )
    .unwrap();
    let compact = assert_state(&state);
    assert_eq!(compact.actor(), 0);
    assert_eq!(compact.turn_owner(), 1);
    assert!(matches!(
        compact.pending().unwrap().task,
        Task::ProphecyGain { player_id: 0, .. }
    ));
    let operation = enabled(&state).last().unwrap().clone();
    state = assert_transition(&state, &compact, operation);
    let compact = assert_state(&state);
    assert_eq!(state.round, 15);
    assert_eq!(compact.actor(), compact.turn_owner());
}
