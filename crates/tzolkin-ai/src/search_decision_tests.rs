//! Historical-loop and external native CLI byte oracles for decision caching.
use super::*;
use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::io::Write;
use tzolkin_core::{GameOptions, create_game};

// Frozen simulate body from commit 6fa4e7efa9eb127e8e27ff3412c243d86164fb67.
// Intentionally retains observation + the old independently checked World.apply.
fn legacy_simulate<F>(
    mut world: RolloutWorld,
    first: GameMove,
    actor: usize,
    target_round: i64,
    config: &SearchConfig,
    stats: &mut SearchStats,
    policy: &mut F,
) -> Result<Sample, DiscardReason>
where
    F: FnMut(&Observation) -> Result<Decision, String>,
{
    let mut local_steps = 0;
    let mut next = first;
    loop {
        if stats.atomic_steps >= config.max_total_steps {
            return Err(DiscardReason::TotalStepCap);
        }
        if local_steps >= config.max_rollout_steps {
            return Err(DiscardReason::RolloutStepCap);
        }
        stats.atomic_steps += 1;
        local_steps += 1;
        world
            .apply(next)
            .map_err(|_| DiscardReason::SimulationError)?;
        stats.max_reached_round = Some(
            stats
                .max_reached_round
                .map_or(world.round(), |v| v.max(world.round())),
        );
        if world.finished() {
            return Ok(Sample {
                score: margin(&world.score_projection(), actor)?,
                terminal: true,
                round: world.round(),
            });
        }
        let observation = world
            .observation()
            .map_err(|_| DiscardReason::SimulationError)?;
        // At a horizon reaching the final day, resolve the actual terminal
        // score rather than guessing which seats have a final action left.
        if target_round < 27 && world.settled_for(actor, target_round) {
            let scores = potential_scores(&observation, world.score_projection())?;
            return Ok(Sample {
                score: margin(&scores, actor)?,
                terminal: false,
                round: world.round(),
            });
        }
        let decision = policy(&observation).map_err(|_| DiscardReason::SimulationError)?;
        if decision.actor != observation.actor
            || decision.observation_key != observation.observation_key
            || !decision.score.is_finite()
            || !observation
                .legal_actions
                .iter()
                .any(|a| a.r#move == decision.r#move)
        {
            return Err(DiscardReason::SimulationError);
        }
        next = decision.r#move;
    }
}

struct HashSink(Sha256);
impl Write for HashSink {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        self.0.update(bytes);
        Ok(bytes.len())
    }
    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}
fn compare_run(
    o: &Observation,
    config: &SearchConfig,
    legacy: bool,
    mutation: u8,
) -> (Vec<u8>, usize, String) {
    let mut calls = 0;
    let mut trace = HashSink(Sha256::new());
    let policy = |observed: &Observation| {
        calls += 1;
        serde_json::to_writer(&mut trace, observed).map_err(|e| e.to_string())?;
        if mutation == 1 {
            return Err("test policy failure".into());
        }
        let mut decision = crate::choose_move(observed)?;
        match mutation {
            2 => decision.actor = (decision.actor + 1) % observed.players.len(),
            3 => decision.observation_key = "stale".into(),
            4 => decision.score = f64::NAN,
            5 => {
                decision.r#move = GameMove::Choose {
                    choice_id: "unknown-choice".into(),
                }
            }
            _ => {}
        }
        Ok(decision)
    };
    let key = config.configuration_key().unwrap();
    let outcome = if legacy {
        choose_prepared_with(o, config, &key, policy, legacy_simulate)
    } else {
        choose_prepared(o, config, &key, policy)
    }
    .unwrap();
    (
        serde_json::to_vec(&outcome).unwrap(),
        calls,
        format!("{:x}", trace.0.finalize()),
    )
}
fn playing(n: usize) -> Observation {
    let mut state = create_game(
        (0..n).map(|id| format!("Person {id}")).collect(),
        11235,
        false,
    )
    .unwrap();
    while state.phase == Phase::Setup {
        let observation = tzolkin_core::observation::observe(&state, state.current_player).unwrap();
        state = tzolkin_core::apply_move(&state, crate::choose_move(&observation).unwrap().r#move)
            .unwrap();
    }
    tzolkin_core::observation::observe(&state, state.current_player).unwrap()
}

#[test]
fn old_loop_outcome_bytes_and_complete_policy_trace_match_caps_and_error_precedence() {
    for n in [3, 4] {
        let observation = playing(n);
        let small = SearchConfig {
            worlds_per_action: 2,
            min_completed_worlds: 2,
            horizon_days: 1,
            ..SearchConfig::default()
        };
        let completed = choose_move(&observation, &small).unwrap();
        let mut cases = vec![small.clone(), SearchConfig::default()];
        let mut total = small.clone();
        total.max_total_steps = 1;
        cases.push(total);
        let mut local = small.clone();
        local.max_rollout_steps = 1;
        cases.push(local);
        let mut both = small.clone();
        both.max_total_steps = 1;
        both.max_rollout_steps = 1;
        cases.push(both);
        let mut partial = small.clone();
        partial.max_total_steps = completed.stats.atomic_steps - 1;
        cases.push(partial);
        for config in cases {
            for mutation in 0..=5 {
                let old = compare_run(&observation, &config, true, mutation);
                let new = compare_run(&observation, &config, false, mutation);
                assert_eq!(new, old, "{n}p config={config:?} mutation={mutation}");
                if config.max_total_steps == 1 || config.max_rollout_steps == 1 {
                    assert!(
                        new.1 > 0,
                        "policy still executes after reaching the first apply cap"
                    );
                    let outcome: SearchOutcome = serde_json::from_slice(&new.0).unwrap();
                    assert_eq!(
                        outcome.stats.discarded_worlds[0].reason,
                        if mutation != 0 {
                            DiscardReason::SimulationError
                        } else if config.max_total_steps == 1 {
                            DiscardReason::TotalStepCap
                        } else {
                            DiscardReason::RolloutStepCap
                        }
                    );
                }
            }
        }
    }
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct GoldenState {
    replay_index: usize,
    observation_key: String,
    direct_outcome_sha256: String,
}
#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct GoldenCase {
    name: String,
    players: usize,
    config: SearchConfig,
    configuration_key: String,
    source_replay_sha256: String,
    selected_corpus_sha256: String,
    states: Vec<GoldenState>,
}
#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct Goldens {
    source_commit: String,
    executable_sha256: String,
    seed: u32,
    cases: Vec<GoldenCase>,
}
fn digest<T: serde::Serialize>(value: &T) -> String {
    format!("{:x}", Sha256::digest(serde_json::to_vec(value).unwrap()))
}

#[test]
fn frozen_external_cli_full_outcome_bytes_match_default_caps_setup_pending_and_day27() {
    let goldens: Goldens = serde_json::from_str(include_str!(
        "../tests/fixtures/search-decision-baseline-6fa4e7e.json"
    ))
    .unwrap();
    assert_eq!(
        goldens.source_commit,
        "6fa4e7efa9eb127e8e27ff3412c243d86164fb67"
    );
    assert_eq!(
        goldens.executable_sha256,
        "301cb47ac3f2e0a11695e305cb8877711caa9f1b66cc9680149fe7b918a7c191"
    );
    assert_eq!(goldens.cases.len(), 6);
    for n in [3, 4] {
        let record = crate::replay::play_game_fast(n, goldens.seed, GameOptions::default(), true)
            .unwrap()
            .2
            .unwrap();
        crate::replay::verify_replay(&record).unwrap();
        for case in goldens.cases.iter().filter(|c| c.players == n) {
            assert_eq!(digest(&record), case.source_replay_sha256);
            assert_eq!(
                case.config.configuration_key().unwrap(),
                case.configuration_key
            );
            assert_eq!(case.states.len(), 12);
            let selected: Vec<_> = case
                .states
                .iter()
                .map(|s| &record.steps[s.replay_index].observation)
                .collect();
            assert_eq!(digest(&selected), case.selected_corpus_sha256);
            assert!(selected.iter().any(|o| o.phase == Phase::Setup));
            assert!(selected.iter().any(|o| o.pending_task.is_some()));
            assert!(
                selected
                    .iter()
                    .any(|o| o.phase == Phase::Playing && o.pending_task.is_none())
            );
            let mut reached_terminal = false;
            let mut reached_cutoff = false;
            let mut insufficient_worlds = false;
            for (at, expected) in case.states.iter().enumerate() {
                assert_eq!(selected[at].observation_key, expected.observation_key);
                let outcome = choose_move(selected[at], &case.config).unwrap();
                reached_terminal |= outcome.stats.terminal_rollouts > 0
                    && outcome.stats.max_reached_round == Some(27);
                reached_cutoff |= outcome.stats.cutoff_rollouts > 0;
                insufficient_worlds |= outcome.status
                    == SearchStatus::Fallback {
                        reason: FallbackReason::InsufficientCompletedWorlds,
                    };
                assert_eq!(
                    digest(&outcome),
                    expected.direct_outcome_sha256,
                    "{n}p case={} replayIndex={}",
                    case.name,
                    expected.replay_index
                );
            }
            if case.name == "default" {
                assert!(reached_terminal && reached_cutoff);
            } else {
                assert!(insufficient_worlds);
            }
        }
    }
}
