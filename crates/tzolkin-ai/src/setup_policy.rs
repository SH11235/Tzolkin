//! Opt-in setup/opening hypotheses; the default heuristic stays unchanged.
use crate::Decision;
use crate::policy::HeuristicWeights;
use tzolkin_bot::policy::HeuristicRevision;
use tzolkin_core::observation::{LegalAction, Observation, TypedAction};
use tzolkin_core::{GameMove, GearId, Phase, TurnMode};

pub use tzolkin_bot::setup_policy::{
    LEGACY_POLICY_VERSION, POLICY_VERSION, choose_move_with_weights,
};
pub const UXMAL_OPENING_POLICY_VERSION: &str = "corn-first-uxmal-opening-v2";
pub const LEGACY_UXMAL_OPENING_POLICY_VERSION: &str = "corn-first-uxmal-opening-v1";

fn opening_policy_version(revision: HeuristicRevision) -> &'static str {
    match revision {
        HeuristicRevision::V1 => LEGACY_UXMAL_OPENING_POLICY_VERSION,
        HeuristicRevision::V2 => UXMAL_OPENING_POLICY_VERSION,
    }
}

pub(crate) fn choose_move_for_revision(
    o: &Observation,
    weights: &HeuristicWeights,
    revision: HeuristicRevision,
) -> Result<Decision, String> {
    tzolkin_bot::setup_policy::choose_move_for_revision(o, weights, revision)
}

/// Internal selector for callers that have validated the observation schema,
/// actor bounds and key. It grants no authority to an unchecked observation.
pub(crate) fn first_uxmal_opening(o: &Observation) -> Option<&LegalAction> {
    let p = &o.players[o.actor];
    let count = o.turn.count;
    if o.phase != Phase::Playing
        || !(3..=4).contains(&o.players.len())
        || o.additional_buildings
        || o.expansion.is_some()
        || o.round != 1
        || o.age != 1
        || o.actor != o.first_player
        || o.actor != o.turn_player
        || o.turn_index != 0
        || o.turn_order.first() != Some(&o.actor)
        || o.pending_task.is_some()
        || o.first_player_claimed.is_some()
        || ![3, 4].contains(&p.workers)
        || p.tribe.is_some()
        || !(0..=3).contains(&count)
        || o.turn.mode
            != if count == 0 {
                TurnMode::None
            } else {
                TurnMode::Place
            }
        || o.turn.begged
        || !o.turn.placed_workers.is_empty()
        || o.turn.tribe_ability_used
        || o.turn.placement_discount_used
        || o.turn.skipped_gear.is_some()
        || o.turn.skipped_position.is_some()
    {
        return None;
    }
    // Basic games do not record turn.placed_workers. On the first actor's
    // first turn, the public board itself identifies all preceding placements.
    let mut own_workers = 0;
    for (gear, slots) in &o.gears {
        for (position, worker) in slots.iter().enumerate() {
            if let Some(worker) = worker
                && !worker.dummy
            {
                if worker.player_id != o.actor as i64 || *gear != GearId::Uxmal || position > 7 {
                    return None;
                }
                own_workers += 1;
            }
        }
    }
    if own_workers != count || p.available_workers != p.workers - count {
        return None;
    }
    if count == 3 {
        return o.legal_actions.iter().find(|a| {
            matches!(
                a.r#move,
                GameMove::EndTurn {
                    double_advance: None
                }
            ) && matches!(
                a.action,
                TypedAction::EndTurn {
                    double_advance: None
                }
            )
        });
    }
    let remaining = (3 - count) as usize;
    let empty: Vec<_> = o
        .gears
        .get(&GearId::Uxmal)?
        .iter()
        .take(8)
        .enumerate()
        .filter_map(|(position, worker)| worker.is_none().then_some(position as i64))
        .take(remaining)
        .collect();
    // Reserve the whole remaining sequence, including each worker-count
    // surcharge. Dummy workers change slot prices; one affordable move alone
    // must never start a partially funded forced opening.
    let total: i64 = empty
        .iter()
        .enumerate()
        .map(|(index, position)| position + count + index as i64)
        .sum();
    if empty.len() != remaining || p.resources[0] < total {
        return None;
    }
    o.legal_actions.iter().find(|a| {
        matches!(
            a.r#move,
            GameMove::Place {
                gear: GearId::Uxmal
            }
        ) && matches!(a.action, TypedAction::Place {
                gear: GearId::Uxmal, corn_cost, discount: false
            } if corn_cost == empty[0] + count)
    })
}

/// Opt-in direct-corn setup plus a fully funded first-player Uxmal opening.
/// Other decisions keep the baseline move/score. Forced decisions still report
/// their baseline action score, not an opening bonus or expected utility.
pub fn choose_uxmal_opening_with_weights(
    o: &Observation,
    weights: &HeuristicWeights,
) -> Result<Decision, String> {
    choose_uxmal_opening_for_revision(o, weights, HeuristicRevision::V2)
}
pub(crate) fn choose_uxmal_opening_for_revision(
    o: &Observation,
    weights: &HeuristicWeights,
    revision: HeuristicRevision,
) -> Result<Decision, String> {
    let mut decision = choose_move_for_revision(o, weights, revision)?;
    decision.policy_version = opening_policy_version(revision).into();
    if let Some(action) = first_uxmal_opening(o) {
        decision.r#move = action.r#move.clone();
        decision.score =
            tzolkin_bot::policy::score_action_for_revision(o, &action.action, weights, revision);
    }
    Ok(decision)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::policy::score_action;
    use crate::replay::{self, ReplaySource, SeatPolicy};
    use tzolkin_core::catalog::wealth;
    use tzolkin_core::observation::{observation_key, observe};
    use tzolkin_core::{Effect, GameOptions, Resource, apply_move, create_game_with_options};

    fn setup(players: usize, seed: u32, options: GameOptions) -> tzolkin_core::GameState {
        create_game_with_options(
            (0..players).map(|i| format!("P{i}")).collect(),
            seed,
            options,
        )
        .unwrap()
    }
    fn corn(o: &Observation, mv: &tzolkin_core::GameMove) -> i64 {
        let TypedAction::ChooseWealth { ids } = &o
            .legal_actions
            .iter()
            .find(|a| &a.r#move == mv)
            .unwrap()
            .action
        else {
            panic!("wealth expected")
        };
        // The fixture independently reads the offered catalog effects; the
        // production selector has moved to Bot and remains a single body.
        ids.iter()
            .map(|id| {
                let tile = wealth(id).unwrap();
                tile.resources.get(&Resource::Corn).copied().unwrap_or(0)
                    + tile
                        .effects
                        .iter()
                        .map(|effect| match effect {
                            Effect::Resources { resources } => {
                                resources.get(&Resource::Corn).copied().unwrap_or(0)
                            }
                            _ => 0,
                        })
                        .sum::<i64>()
            })
            .sum()
    }

    fn playing_with_wealth(players: usize, seed: u32, choice_id: &str) -> tzolkin_core::GameState {
        let mut state = apply_move(
            &setup(players, seed, GameOptions::default()),
            GameMove::Choose {
                choice_id: choice_id.into(),
            },
        )
        .unwrap();
        while state.phase == Phase::Setup {
            let o = observe(&state, state.current_player).unwrap();
            state = apply_move(
                &state,
                choose_move_with_weights(&o, &HeuristicWeights::default())
                    .unwrap()
                    .r#move,
            )
            .unwrap();
        }
        state
    }

    fn assert_opening_fallback(o: &Observation) {
        let weights = HeuristicWeights::default();
        let mut expected = crate::choose_move_with_weights(o, &weights).unwrap();
        expected.policy_version = UXMAL_OPENING_POLICY_VERSION.into();
        assert_eq!(
            choose_uxmal_opening_with_weights(o, &weights).unwrap(),
            expected
        );
    }

    #[test]
    fn funded_first_turn_places_exactly_three_and_keeps_the_fourth_worker() {
        let weights = HeuristicWeights::default();
        for players in [3, 4] {
            for (seed, pair, workers) in [(3, "wealth:w10:w11", 3), (1, "wealth:w15:w21", 4)] {
                let mut state = playing_with_wealth(players, seed, pair);
                assert_eq!(state.players[0].workers, workers);
                for count in 1..=3 {
                    let o = observe(&state, state.current_player).unwrap();
                    assert!(o.turn.placed_workers.is_empty());
                    let chosen = choose_uxmal_opening_with_weights(&o, &weights).unwrap();
                    assert_eq!(chosen.policy_version, UXMAL_OPENING_POLICY_VERSION);
                    assert_eq!(
                        chosen.r#move,
                        GameMove::Place {
                            gear: GearId::Uxmal
                        }
                    );
                    state = apply_move(&state, chosen.r#move).unwrap();
                    assert_eq!(state.turn.count, count);
                }
                let o = observe(&state, state.current_player).unwrap();
                let chosen = choose_uxmal_opening_with_weights(&o, &weights).unwrap();
                assert_eq!(
                    chosen.r#move,
                    GameMove::EndTurn {
                        double_advance: None
                    }
                );
                state = apply_move(&state, chosen.r#move).unwrap();
                assert_eq!(tzolkin_core::available_workers(&state, 0), workers - 3);
                assert_eq!(
                    state.gears[&GearId::Uxmal]
                        .iter()
                        .flatten()
                        .filter(|w| !w.dummy && w.player_id == 0)
                        .count(),
                    3
                );
                assert_opening_fallback(&observe(&state, state.current_player).unwrap());
            }
        }
    }

    #[test]
    fn dummy_prices_reserve_the_whole_sequence_before_initial_and_mid_turn_forcing() {
        // This real initial pair grants seven corn, but dummy slot1 makes the
        // three Uxmal placements cost 0 + (2+1) + (3+2) = eight.
        let state = playing_with_wealth(3, 17, "wealth:w01:w19");
        let o = observe(&state, state.current_player).unwrap();
        assert_eq!(o.players[0].resources[0], 7);
        assert!(o.gears[&GearId::Uxmal][1].as_ref().unwrap().dummy);
        assert!(o.legal_actions.iter().any(|a| a.r#move
            == GameMove::Place {
                gear: GearId::Uxmal
            }));
        assert!(first_uxmal_opening(&o).is_none());
        assert_opening_fallback(&o);
        let after = apply_move(
            &state,
            GameMove::Place {
                gear: GearId::Uxmal,
            },
        )
        .unwrap();
        let o = observe(&after, after.current_player).unwrap();
        assert_eq!(o.turn.count, 1);
        assert!(first_uxmal_opening(&o).is_none());
        assert_opening_fallback(&o);
        // The same real board with exactly eight corn funds the entire opening.
        let mut funded = playing_with_wealth(3, 17, "wealth:w10:w06");
        assert_eq!(funded.players[0].resources[&Resource::Corn], 8);
        for _ in 0..3 {
            let o = observe(&funded, funded.current_player).unwrap();
            let chosen =
                choose_uxmal_opening_with_weights(&o, &HeuristicWeights::default()).unwrap();
            assert_eq!(
                chosen.r#move,
                GameMove::Place {
                    gear: GearId::Uxmal
                }
            );
            funded = apply_move(&funded, chosen.r#move).unwrap();
        }
        assert_eq!(funded.players[0].resources[&Resource::Corn], 0);
        assert_eq!(funded.turn.count, 3);
    }

    #[test]
    fn opening_delegates_setup_and_rejects_outside_scope_or_invalid_inputs() {
        let weights = HeuristicWeights::default();
        let o = observe(&setup(4, 3, GameOptions::default()), 0).unwrap();
        let mut expected = choose_move_with_weights(&o, &weights).unwrap();
        expected.policy_version = UXMAL_OPENING_POLICY_VERSION.into();
        assert_eq!(
            choose_uxmal_opening_with_weights(&o, &weights).unwrap(),
            expected
        );
        let mut states = vec![
            setup(2, 17, GameOptions::default()),
            setup(
                3,
                17,
                GameOptions {
                    additional_buildings: true,
                    ..GameOptions::default()
                },
            ),
        ];
        let state = playing_with_wealth(4, 3, "wealth:w10:w11");
        states.push(
            apply_move(
                &state,
                GameMove::Place {
                    gear: GearId::Yaxchilan,
                },
            )
            .unwrap(),
        );
        states.push(apply_move(&state, GameMove::FirstPlayer).unwrap());
        let state = playing_with_wealth(4, 3, "wealth:w15:w05");
        let pending = apply_move(&state, GameMove::Beg).unwrap();
        assert!(pending.pending.is_some());
        states.push(pending);
        for state in states {
            assert_opening_fallback(&observe(&state, state.current_player).unwrap());
        }
        let state = playing_with_wealth(4, 3, "wealth:w10:w11");
        let original = observe(&state, state.current_player).unwrap();
        for mutate in [0, 1, 2] {
            let mut o = original.clone();
            match mutate {
                0 => o.round = 2,
                1 => o.turn.begged = true,
                _ => o.turn.count = 1,
            }
            o.observation_key = observation_key(&o).unwrap();
            assert_opening_fallback(&o);
        }
        let mut stale = original.clone();
        stale.observation_key = "modified".into();
        assert!(
            choose_uxmal_opening_with_weights(&stale, &weights)
                .unwrap_err()
                .contains("key mismatch")
        );
        let mut invalid = weights;
        invalid.worker = f64::NAN;
        assert!(choose_uxmal_opening_with_weights(&original, &invalid).is_err());
    }

    #[test]
    fn real_setup_maximizes_direct_corn_and_actual_transition_without_material_conversion() {
        let weights = HeuristicWeights::default();
        for players in [3, 4] {
            let (state, o, chosen) = (0..128)
                .find_map(|seed| {
                    let s = setup(players, seed, GameOptions::default());
                    let o = observe(&s, s.current_player).unwrap();
                    let a = crate::choose_move_with_weights(&o, &weights).unwrap();
                    let b = choose_move_with_weights(&o, &weights).unwrap();
                    (corn(&o, &a.r#move) < corn(&o, &b.r#move)).then_some((s, o, b))
                })
                .expect("real offered pair with conflicting objectives");
            let maximum = o
                .legal_actions
                .iter()
                .map(|a| corn(&o, &a.r#move))
                .max()
                .unwrap();
            assert_eq!(corn(&o, &chosen.r#move), maximum);
            for revision in [HeuristicRevision::V1, HeuristicRevision::V2] {
                assert_eq!(
                    choose_move_for_revision(&o, &weights, revision).unwrap(),
                    tzolkin_bot::setup_policy::choose_move_for_revision(&o, &weights, revision)
                        .unwrap()
                );
            }
            let legal = o
                .legal_actions
                .iter()
                .find(|a| a.r#move == chosen.r#move)
                .unwrap();
            assert_eq!(
                chosen.score.to_bits(),
                tzolkin_bot::policy::score_action_for_revision(
                    &o,
                    &legal.action,
                    &weights,
                    HeuristicRevision::V2,
                )
                .to_bits()
            );
            let mut missing_offer = o.clone();
            missing_offer.private.wealth_offer.clear();
            missing_offer.observation_key = observation_key(&missing_offer).unwrap();
            assert!(choose_move_with_weights(&missing_offer, &weights).is_err());
            let after = apply_move(&state, chosen.r#move).unwrap();
            assert_eq!(
                after.players[o.actor].resources[&Resource::Corn],
                o.players[o.actor].resources[0] + maximum
            );
        }
    }

    #[test]
    fn ties_use_baseline_score_then_original_order_and_invalid_key_is_rejected() {
        let weights = HeuristicWeights {
            material_values: [0.0; 3],
            corn_base: 0.0,
            corn_when_short: 0.0,
            temple_step: 0.0,
            technology_step: 0.0,
            worker: 0.0,
        };
        let mut o = (0..1024)
            .find_map(|seed| {
                let s = setup(4, seed, GameOptions::default());
                let o = observe(&s, s.current_player).unwrap();
                let chosen = choose_move_with_weights(&o, &weights).unwrap();
                let count = o
                    .legal_actions
                    .iter()
                    .filter(|a| {
                        corn(&o, &a.r#move) == corn(&o, &chosen.r#move)
                            && score_action(&o, &a.action, &weights) == chosen.score
                    })
                    .count();
                (count >= 2).then_some(o)
            })
            .expect("real offer with both corn and baseline-score ties");
        for reverse in [false, true] {
            if reverse {
                o.legal_actions.reverse();
                o.observation_key = observation_key(&o).unwrap();
            }
            let chosen = choose_move_with_weights(&o, &weights).unwrap();
            let expected = o
                .legal_actions
                .iter()
                .find(|a| {
                    corn(&o, &a.r#move) == corn(&o, &chosen.r#move)
                        && score_action(&o, &a.action, &weights) == chosen.score
                })
                .unwrap();
            assert_eq!(chosen.r#move, expected.r#move);
            assert_eq!(chosen, choose_move_with_weights(&o, &weights).unwrap());
        }
        o.observation_key = "modified".into();
        assert!(
            choose_move_with_weights(&o, &weights)
                .unwrap_err()
                .contains("key mismatch")
        );
    }

    #[test]
    fn unsupported_setup_and_playing_keep_the_exact_baseline_move_and_score() {
        let weights = HeuristicWeights::default();
        let mut states = vec![
            setup(2, 17, GameOptions::default()),
            setup(
                3,
                17,
                GameOptions {
                    additional_buildings: true,
                    ..GameOptions::default()
                },
            ),
            setup(
                4,
                17,
                GameOptions {
                    tribes: true,
                    ..GameOptions::default()
                },
            ),
        ];
        let mut playing = setup(3, 17, GameOptions::default());
        while playing.phase == Phase::Setup {
            let o = observe(&playing, playing.current_player).unwrap();
            playing = apply_move(&playing, crate::choose_move(&o).unwrap().r#move).unwrap();
        }
        states.push(playing);
        for state in states {
            let o = observe(&state, state.current_player).unwrap();
            let mut expected = crate::choose_move_with_weights(&o, &weights).unwrap();
            expected.policy_version = POLICY_VERSION.into();
            assert_eq!(choose_move_with_weights(&o, &weights).unwrap(), expected);
            for revision in [HeuristicRevision::V1, HeuristicRevision::V2] {
                let baseline =
                    tzolkin_bot::choose_move_with_weights_for_revision(&o, &weights, revision)
                        .unwrap();
                let mut expected = baseline.clone();
                expected.policy_version = match revision {
                    HeuristicRevision::V1 => LEGACY_POLICY_VERSION,
                    HeuristicRevision::V2 => POLICY_VERSION,
                }
                .into();
                assert_eq!(
                    choose_move_for_revision(&o, &weights, revision).unwrap(),
                    expected
                );
                let mut expected = baseline;
                expected.policy_version = opening_policy_version(revision).into();
                assert_eq!(
                    choose_uxmal_opening_for_revision(&o, &weights, revision).unwrap(),
                    expected
                );
            }
        }
        // The old wrappers must execute V1 when the underlying fallback differs,
        // rather than merely replacing the V2 decision's version string.
        let mut overdue = setup(3, 17, GameOptions::default());
        overdue.phase = Phase::Playing;
        overdue.round = 9;
        overdue.food_days.clear();
        overdue.pending = Some(tzolkin_core::Pending {
            title: "Trade".into(),
            task: tzolkin_core::Task::Trade,
            after: vec![],
        });
        let player = &mut overdue.players[overdue.current_player];
        player.workers = 5;
        player.feed_workers = 0;
        player.feed_all = false;
        player.feed_discount = 0;
        player.resources = [
            (Resource::Corn, 9),
            (Resource::Wood, 1),
            (Resource::Stone, 0),
            (Resource::Gold, 0),
            (Resource::Skull, 0),
        ]
        .into_iter()
        .collect();
        let o = observe(&overdue, overdue.current_player).unwrap();
        for opening in [false, true] {
            let choose = if opening {
                choose_uxmal_opening_for_revision
            } else {
                choose_move_for_revision
            };
            let v1 = choose(&o, &weights, HeuristicRevision::V1).unwrap();
            let v2 = choose(&o, &weights, HeuristicRevision::V2).unwrap();
            assert_eq!(
                o.legal_actions
                    .iter()
                    .find(|row| row.r#move == v1.r#move)
                    .unwrap()
                    .action,
                TypedAction::Skip,
            );
            assert_ne!(v1.r#move, v2.r#move);
            assert_eq!(v1.score.to_bits(), 0.0_f64.to_bits());
            let policy = if opening {
                SeatPolicy::CornFirstUxmalOpening {
                    policy_version: v1.policy_version.clone(),
                    weights: weights.clone(),
                }
            } else {
                SeatPolicy::CornFirstSetup {
                    policy_version: v1.policy_version.clone(),
                    weights: weights.clone(),
                }
            };
            assert!(policy.validate().is_err());
            policy.validate_historical_restore().unwrap();
            let mut wire = serde_json::to_value(&policy).unwrap();
            wire["policyVersion"] = "unknown-historical-version".into();
            assert!(
                serde_json::from_value::<SeatPolicy>(wire)
                    .unwrap()
                    .validate_historical_restore()
                    .is_err()
            );
            let mut invalid = weights.clone();
            invalid.worker = f64::NAN;
            assert!(choose(&o, &invalid, HeuristicRevision::V1).is_err());
        }
        let v1 = SeatPolicy::Heuristic {
            policy_version: HeuristicRevision::V1.policy_version().into(),
            weights: weights.clone(),
        };
        assert!(v1.validate().is_err());
        v1.validate_historical_restore().unwrap();
        let mut wire = serde_json::to_value(&v1).unwrap();
        wire["policyVersion"] = "heuristic-v0".into();
        assert!(
            serde_json::from_value::<SeatPolicy>(wire)
                .unwrap()
                .validate_historical_restore()
                .is_err()
        );
        for mut policy in [
            v1,
            SeatPolicy::CornFirstSetup {
                policy_version: LEGACY_POLICY_VERSION.into(),
                weights: weights.clone(),
            },
            SeatPolicy::CornFirstUxmalOpening {
                policy_version: LEGACY_UXMAL_OPENING_POLICY_VERSION.into(),
                weights: weights.clone(),
            },
        ] {
            match &mut policy {
                SeatPolicy::Heuristic { weights, .. }
                | SeatPolicy::CornFirstSetup { weights, .. }
                | SeatPolicy::CornFirstUxmalOpening { weights, .. } => weights.worker = f64::NAN,
                _ => unreachable!(),
            }
            assert!(policy.validate_historical_restore().is_err());
        }
        let mut stale = o;
        stale.observation_key = "stale".into();
        assert!(choose_move_for_revision(&stale, &weights, HeuristicRevision::V1).is_err());
        assert!(
            choose_uxmal_opening_for_revision(&stale, &weights, HeuristicRevision::V1).is_err()
        );
    }

    #[test]
    fn complete_native_records_roundtrip_with_distinct_closed_provenance() {
        let weights = HeuristicWeights::default();
        for opening in [false, true] {
            let (policy, kind) = if opening {
                (
                    SeatPolicy::CornFirstUxmalOpening {
                        policy_version: UXMAL_OPENING_POLICY_VERSION.into(),
                        weights: weights.clone(),
                    },
                    "cornFirstUxmalOpening",
                )
            } else {
                (
                    SeatPolicy::CornFirstSetup {
                        policy_version: POLICY_VERSION.into(),
                        weights: weights.clone(),
                    },
                    "cornFirstSetup",
                )
            };
            policy.validate().unwrap();
            let wire = serde_json::to_value(&policy).unwrap();
            assert_eq!(wire["kind"], kind);
            let mut bad = wire.clone();
            bad["policyVersion"] = crate::POLICY_VERSION.into();
            assert!(
                serde_json::from_value::<SeatPolicy>(bad)
                    .unwrap()
                    .validate()
                    .is_err()
            );
            let mut unknown = wire;
            unknown["forceUxmal"] = true.into();
            assert!(serde_json::from_value::<SeatPolicy>(unknown).is_err());
            for players in [3, 4] {
                let (_, _, record) = replay::play_game_using_fast(
                    players,
                    17,
                    GameOptions::default(),
                    true,
                    ReplaySource::PolicySelfPlay {
                        policies: vec![policy.clone(); players],
                    },
                    |o| {
                        if opening {
                            choose_uxmal_opening_with_weights(o, &weights)
                        } else {
                            choose_move_with_weights(o, &weights)
                        }
                    },
                )
                .unwrap();
                let record = record.unwrap();
                let decoded =
                    serde_json::from_slice(&serde_json::to_vec(&record).unwrap()).unwrap();
                assert_eq!(record, decoded);
                assert_eq!(
                    replay::verify_replay(&decoded).unwrap().phase,
                    Phase::Finished
                );
            }
        }
    }
}
