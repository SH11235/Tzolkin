//! Explicit learned-policy experiments. A training loss is never a strength claim.
use crate::dataset::{DatasetSplit, ValidatedDataset, split_for_family};
use crate::model::{LEARNED_POLICY_VERSION, LoadedPolicy, ModelArtifact};
use crate::policy::HeuristicWeights;
use crate::replay::{self, GameReplay, ReplaySource, SeatPolicy};
use crate::{POLICY_VERSION, choose_move};
use serde::Serialize;
use sha2::{Digest, Sha256};
use std::path::Path;
use std::time::Instant;
use tzolkin_core::{FinalScore, GameOptions, GameState};

pub fn heuristic_seat() -> SeatPolicy {
    SeatPolicy::Heuristic {
        policy_version: POLICY_VERSION.into(),
        weights: HeuristicWeights::default(),
    }
}
pub fn learned_seat(model: &ModelArtifact) -> SeatPolicy {
    SeatPolicy::Learned {
        policy_version: LEARNED_POLICY_VERSION.into(),
        model_checksum: model.checksum.clone(),
    }
}
pub fn learned_game(
    model: &ModelArtifact,
    players: usize,
    seed: u32,
    options: GameOptions,
    learned_seats: &[usize],
    record: bool,
) -> Result<(GameState, usize, Option<GameReplay>), String> {
    if !(2..=5).contains(&players)
        || learned_seats.is_empty()
        || learned_seats.iter().any(|seat| *seat >= players)
        || learned_seats
            .iter()
            .enumerate()
            .any(|(i, seat)| learned_seats[..i].contains(seat))
    {
        return Err("Invalid learned selfplay seats".into());
    }
    let policy = LoadedPolicy::new(model)?;
    let policies = (0..players)
        .map(|seat| {
            if learned_seats.contains(&seat) {
                learned_seat(model)
            } else {
                heuristic_seat()
            }
        })
        .collect();
    replay::play_game_using(
        players,
        seed,
        options,
        record,
        ReplaySource::PolicySelfPlay { policies },
        |observation| {
            if learned_seats.contains(&observation.actor) {
                policy.choose_move(observation)
            } else {
                choose_move(observation)
            }
        },
    )
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct EvaluationGame {
    pub seed: u32,
    pub learned_seat: usize,
    pub decisions: usize,
    pub elapsed_ms: f64,
    pub utility: Option<f64>,
    pub scores: Vec<FinalScore>,
    pub error: Option<String>,
}
#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct EvaluationReport {
    pub model_checksum: String,
    pub dataset_fingerprint: String,
    pub players: usize,
    pub options: GameOptions,
    pub games: Vec<EvaluationGame>,
    pub completed: usize,
    pub failed: usize,
    pub mean_winner_utility: Option<f64>,
    pub seat_rotation: bool,
    pub held_out_families: bool,
}

/// Evaluate an explicit fixed list of seed families at every absolute seat.
/// Reject overlap with train/validation and families not assigned to the held-out split.
pub fn evaluate_model(
    checkpoint: &crate::training::TrainingCheckpoint,
    dataset: &ValidatedDataset,
    players: usize,
    seeds: &[u32],
    mut options: GameOptions,
) -> Result<EvaluationReport, String> {
    checkpoint.validate()?;
    if checkpoint.dataset_fingerprint != dataset.manifest().fingerprint {
        return Err("Evaluation dataset/checkpoint fingerprint mismatch".into());
    }
    let model = &checkpoint.model;
    if !(2..=5).contains(&players)
        || seeds.is_empty()
        || seeds.len() > 256
        || seeds
            .iter()
            .enumerate()
            .any(|(i, seed)| seeds[..i].contains(seed))
    {
        return Err("Evaluation needs 1..256 distinct seeds and 2..5 players".into());
    }
    if players == 5 {
        options.quick_actions = true;
    }
    for seed in seeds {
        let mut hash = Sha256::new();
        hash.update(b"tzolkin-seed-family-v1\0");
        hash.update(seed.to_le_bytes());
        let family = format!("{:x}", hash.finalize());
        if split_for_family(&family)? != DatasetSplit::Test
            || dataset
                .manifest()
                .games
                .iter()
                .any(|game| game.family_id == family && game.split != DatasetSplit::Test)
        {
            return Err(format!("Seed {seed} is not in the held-out test partition"));
        }
    }
    let mut games = Vec::with_capacity(players * seeds.len());
    let mut completed = 0;
    let mut sum = 0.0;
    for seed in seeds {
        for learned_seat in 0..players {
            let started = Instant::now();
            match learned_game(
                model,
                players,
                *seed,
                options.clone(),
                &[learned_seat],
                false,
            ) {
                Ok((state, decisions, _)) => {
                    let winners = state
                        .final_scores
                        .iter()
                        .filter(|score| score.rank == 1)
                        .count();
                    let utility = if state
                        .final_scores
                        .iter()
                        .any(|score| score.player_id == learned_seat && score.rank == 1)
                    {
                        1.0 / winners as f64
                    } else {
                        0.0
                    };
                    completed += 1;
                    sum += utility;
                    games.push(EvaluationGame {
                        seed: *seed,
                        learned_seat,
                        decisions,
                        elapsed_ms: started.elapsed().as_secs_f64() * 1000.0,
                        utility: Some(utility),
                        scores: state.final_scores,
                        error: None,
                    });
                }
                Err(error) => games.push(EvaluationGame {
                    seed: *seed,
                    learned_seat,
                    decisions: 0,
                    elapsed_ms: started.elapsed().as_secs_f64() * 1000.0,
                    utility: None,
                    scores: vec![],
                    error: Some(error),
                }),
            }
        }
    }
    Ok(EvaluationReport {
        model_checksum: model.checksum.clone(),
        dataset_fingerprint: dataset.manifest().fingerprint.clone(),
        players,
        options,
        failed: games.len() - completed,
        games,
        completed,
        mean_winner_utility: if completed == 0 {
            None
        } else {
            Some(sum / completed as f64)
        },
        seat_rotation: true,
        held_out_families: true,
    })
}

/// Artifacts are published only into a new directory. metrics.json is the completion marker.
pub fn train_to_directory(
    dataset: &ValidatedDataset,
    config: &crate::training::TrainingConfig,
    resume: Option<&crate::training::TrainingCheckpoint>,
    output: &Path,
) -> Result<crate::training::TrainingMetrics, String> {
    if output.exists() {
        return Err("Training destination already exists".into());
    }
    let outcome = crate::training::train_dataset(dataset, config, resume)?;
    std::fs::create_dir(output).map_err(|error| error.to_string())?;
    outcome.model.save_new(&output.join("model.json"))?;
    outcome
        .checkpoint
        .save_new(&output.join("checkpoint.json"))?;
    crate::model::write_new_json(&output.join("metrics.json"), &outcome.metrics)?;
    Ok(outcome.metrics)
}
