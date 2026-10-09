//! Opt-in direct-corn setup hypothesis; the default heuristic stays unchanged.
use crate::Decision;
use crate::policy::{HeuristicWeights, score_action};
use tzolkin_core::catalog::wealth;
use tzolkin_core::observation::{Observation, TypedAction};
use tzolkin_core::{Effect, Phase, Resource, StartingWealth};

pub const POLICY_VERSION: &str = "corn-first-setup-v1";

fn direct_corn(tile: &StartingWealth) -> i64 {
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
}

fn selected_corn(o: &Observation, ids: &[String]) -> Result<i64, String> {
    if ids.len() != 2
        || ids[0] == ids[1]
        || ids.iter().any(|id| !o.private.wealth_offer.contains(id))
    {
        return Err("Corn-first setup requires two distinct own offered wealth tiles".into());
    }
    ids.iter()
        .map(|id| {
            wealth(id)
                .map(direct_corn)
                .ok_or_else(|| "Unknown starting wealth tile".into())
        })
        .sum()
}

/// Base 3/4-player setup orders wealth choices by direct corn, baseline score,
/// then original legal order. `score` is the baseline tie-break score, not the
/// primary corn objective or an estimate of expected utility. All other inputs
/// retain the baseline move and score, with this candidate's explicit version.
pub fn choose_move_with_weights(
    o: &Observation,
    weights: &HeuristicWeights,
) -> Result<Decision, String> {
    let mut decision = crate::choose_move_with_weights(o, weights)?;
    decision.policy_version = POLICY_VERSION.into();
    if o.phase != Phase::Setup
        || !(3..=4).contains(&o.players.len())
        || o.additional_buildings
        || o.expansion.is_some()
    {
        return Ok(decision);
    }
    let mut best = None;
    for legal in &o.legal_actions {
        if let TypedAction::ChooseWealth { ids } = &legal.action {
            let corn = selected_corn(o, ids)?;
            let score = score_action(o, &legal.action, weights);
            if best
                .as_ref()
                .is_none_or(|(_, c, s)| corn > *c || (corn == *c && score > *s))
            {
                best = Some((legal, corn, score));
            }
        }
    }
    if let Some((legal, _, score)) = best {
        decision.r#move = legal.r#move.clone();
        decision.score = score;
    }
    Ok(decision)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::replay::{self, ReplaySource, SeatPolicy};
    use tzolkin_core::observation::{observation_key, observe};
    use tzolkin_core::{GameOptions, apply_move, create_game_with_options};

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
        selected_corn(o, ids).unwrap()
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
            let after = apply_move(&state, chosen.r#move).unwrap();
            assert_eq!(
                after.players[o.actor].resources[&Resource::Corn],
                o.players[o.actor].resources[0] + maximum
            );
        }
        let mut tile = wealth("w15").unwrap().clone();
        assert_eq!(direct_corn(&tile), 0); // Worker+1 is not future corn.
        tile.effects.push(Effect::Resources {
            resources: [(Resource::Corn, 2), (Resource::Gold, 3)].into(),
        });
        assert_eq!(direct_corn(&tile), 2);
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
        }
    }

    #[test]
    fn complete_native_records_roundtrip_with_distinct_closed_provenance() {
        let weights = HeuristicWeights::default();
        let policy = SeatPolicy::CornFirstSetup {
            policy_version: POLICY_VERSION.into(),
            weights: weights.clone(),
        };
        let wire = serde_json::to_value(&policy).unwrap();
        assert_eq!(wire["kind"], "cornFirstSetup");
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
                |o| choose_move_with_weights(o, &weights),
            )
            .unwrap();
            let record = record.unwrap();
            let decoded = serde_json::from_slice(&serde_json::to_vec(&record).unwrap()).unwrap();
            assert_eq!(record, decoded);
            assert_eq!(
                replay::verify_replay(&decoded).unwrap().phase,
                Phase::Finished
            );
        }
    }
}
