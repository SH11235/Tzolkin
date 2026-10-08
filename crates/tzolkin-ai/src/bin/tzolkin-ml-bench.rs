//! Scalar と opt-in SIMD を同じ公開 Observation / 実モデルで測定する。
use std::hint::black_box;
use std::path::PathBuf;
use std::time::Instant;

use serde_json::{Value, json};
use tzolkin_ai::features::{FEATURE_COUNT, FeatureEncoder};
use tzolkin_ai::kernel::Kernel;
use tzolkin_ai::model::{LoadedPolicy, ModelArtifact, Prediction, VALUE_SIDES};
use tzolkin_ai::replay;
use tzolkin_core::GameOptions;
use tzolkin_core::observation::{Observation, fingerprint, observe};

const USAGE: &str = "tzolkin-ml-bench --model PATH [--kernel auto|scalar|avx2|sse2|neon|simd128] [--players 3|4] [--seed 11235] [--iterations 1..1000] [--rounds 3..15]\nDefaults: kernel=auto, players=4, seed=11235, iterations=1, rounds=5.\nRelease measurements compare actual inference, feature encoding and legal move selection. Playing strength is not measured.";
const MAX_FEATURE_VECTORS: usize = 65_536;
const MAX_TOTAL_PREDICTIONS: usize = 2_000_000;

struct Config {
    model: PathBuf,
    kernel: Kernel,
    players: usize,
    seed: u32,
    iterations: usize,
    rounds: usize,
}

fn config(args: &[String]) -> Result<Config, String> {
    let mut model = None;
    let mut kernel = Kernel::Auto;
    let mut players = 4;
    let mut seed = 11235;
    let mut iterations = 1;
    let mut rounds = 5;
    let mut seen = std::collections::HashSet::new();
    for pair in args.chunks(2) {
        if pair.len() != 2 || !seen.insert(pair[0].as_str()) {
            return Err("Expected unique flag/value pairs".into());
        }
        match pair[0].as_str() {
            "--model" => model = Some(PathBuf::from(&pair[1])),
            "--kernel" => {
                kernel = match pair[1].as_str() {
                    "scalar" => Kernel::Scalar,
                    "auto" => Kernel::Auto,
                    "avx2" => Kernel::Avx2,
                    "sse2" => Kernel::Sse2,
                    "neon" => Kernel::Neon,
                    "simd128" => Kernel::Simd128,
                    _ => return Err("Unknown kernel".into()),
                }
            }
            "--players" => players = pair[1].parse().map_err(|_| "Invalid players")?,
            "--seed" => seed = pair[1].parse().map_err(|_| "Invalid u32 seed")?,
            "--iterations" => iterations = pair[1].parse().map_err(|_| "Invalid iterations")?,
            "--rounds" => rounds = pair[1].parse().map_err(|_| "Invalid rounds")?,
            _ => return Err(format!("Unknown flag {}", pair[0])),
        }
    }
    if ![3, 4].contains(&players)
        || !(1..=1000).contains(&iterations)
        || !(3..=15).contains(&rounds)
    {
        return Err("Players/iterations/rounds outside bounded benchmark range".into());
    }
    Ok(Config {
        model: model.ok_or("--model is required")?,
        kernel,
        players,
        seed,
        iterations,
        rounds,
    })
}

struct Batch {
    features: Vec<[f32; FEATURE_COUNT]>,
    active: [bool; VALUE_SIDES],
}

fn predict_batch(policy: &LoadedPolicy<'_>, batches: &[Batch]) -> Result<(), String> {
    for batch in batches {
        for features in &batch.features {
            black_box(policy.predict(black_box(features), batch.active)?);
        }
    }
    Ok(())
}

fn choose_batch(policy: &LoadedPolicy<'_>, observations: &[Observation]) -> Result<(), String> {
    for observation in observations {
        black_box(policy.choose_move(black_box(observation))?);
    }
    Ok(())
}

fn encode_batch(observations: &[Observation]) -> Result<(), String> {
    for observation in observations {
        let encoder = FeatureEncoder::new(black_box(observation))?;
        for index in 0..observation.legal_actions.len() {
            black_box(encoder.encode_legal(index)?);
        }
    }
    Ok(())
}

fn time(
    iterations: usize,
    mut operation: impl FnMut() -> Result<(), String>,
) -> Result<f64, String> {
    let start = Instant::now();
    for _ in 0..iterations {
        operation()?;
    }
    Ok(start.elapsed().as_secs_f64())
}

fn timings(name: &str, samples: &[f64], operations: usize) -> Result<Value, String> {
    if samples.is_empty()
        || samples
            .iter()
            .any(|value| !value.is_finite() || *value <= 0.0)
        || operations == 0
    {
        return Err(format!(
            "{name}: benchmark requires positive finite round durations and nonzero operations; increase --iterations if the clock reports zero"
        ));
    }
    let mut ordered = samples.to_vec();
    ordered.sort_by(f64::total_cmp);
    let median = ordered[ordered.len() / 2];
    let operations_per_second = operations as f64 / median;
    let mean_ns_per_operation = median * 1_000_000_000.0 / operations as f64;
    if !operations_per_second.is_finite()
        || operations_per_second <= 0.0
        || !mean_ns_per_operation.is_finite()
        || mean_ns_per_operation <= 0.0
    {
        return Err(format!(
            "{name}: benchmark rate is outside the finite positive range"
        ));
    }
    Ok(json!({
        "roundSeconds": samples,
        "medianSeconds": median,
        "minSeconds": ordered[0],
        "maxSeconds": ordered[ordered.len() - 1],
        "operationsPerRound": operations,
        "operationsPerSecond": operations_per_second,
        "meanNsPerOperation": mean_ns_per_operation,
    }))
}

#[derive(Default)]
struct Parity {
    predictions: usize,
    decisions: usize,
    moves_matched: usize,
    max_policy_abs_error: f64,
    max_value_abs_error: f64,
    numerical_within_tolerance: bool,
    inactive_utilities_zero: bool,
}

impl Parity {
    fn prediction(
        &mut self,
        reference: &Prediction,
        actual: &Prediction,
        active: [bool; VALUE_SIDES],
    ) {
        self.predictions += 1;
        let policy_error =
            (f64::from(reference.policy_logit) - f64::from(actual.policy_logit)).abs();
        self.max_policy_abs_error = self.max_policy_abs_error.max(policy_error);
        self.numerical_within_tolerance &=
            policy_error <= 0.00005 * (1.0 + f64::from(reference.policy_logit).abs());
        for (side, is_active) in active.into_iter().enumerate() {
            let error =
                (f64::from(reference.utilities[side]) - f64::from(actual.utilities[side])).abs();
            self.max_value_abs_error = self.max_value_abs_error.max(error);
            self.numerical_within_tolerance &= error <= 0.00005;
            if !is_active {
                self.inactive_utilities_zero &=
                    reference.utilities[side] == 0.0 && actual.utilities[side] == 0.0;
            }
        }
    }
}

fn run(config: Config) -> Result<Value, String> {
    let artifact = ModelArtifact::load(&config.model)?;
    let scalar = LoadedPolicy::new(&artifact)?;
    let selected = LoadedPolicy::with_kernel(&artifact, config.kernel)?;
    let states = replay::benchmark_states(config.players, config.seed, GameOptions::default())?;
    let observations = states
        .iter()
        .map(|state| observe(state, state.current_player))
        .collect::<Result<Vec<_>, _>>()?;
    drop(states);
    if observations.is_empty() || observations.len() > 4096 {
        return Err("Empty/oversized observation corpus".into());
    }
    let feature_count = observations
        .iter()
        .try_fold(0_usize, |count, o| count.checked_add(o.legal_actions.len()))
        .ok_or("Feature count overflow")?;
    if feature_count > MAX_FEATURE_VECTORS
        || feature_count
            .checked_mul(config.iterations)
            .and_then(|n| n.checked_mul(config.rounds))
            .is_none_or(|n| n > MAX_TOTAL_PREDICTIONS)
    {
        return Err("Benchmark corpus/iteration workload exceeds limit".into());
    }
    let mut batches = Vec::with_capacity(observations.len());
    let mut corpus_bytes = Vec::new();
    for observation in &observations {
        corpus_bytes.extend_from_slice(observation.observation_key.as_bytes());
        let encoder = FeatureEncoder::new(observation)?;
        let features = (0..observation.legal_actions.len())
            .map(|index| encoder.encode_legal(index))
            .collect::<Result<Vec<_>, _>>()?;
        batches.push(Batch {
            features,
            active: std::array::from_fn(|side| side < observation.players.len()),
        });
    }
    let mut parity = Parity {
        numerical_within_tolerance: true,
        inactive_utilities_zero: true,
        ..Parity::default()
    };
    for (batch, observation) in batches.iter().zip(&observations) {
        for features in &batch.features {
            parity.prediction(
                &scalar.predict(features, batch.active)?,
                &selected.predict(features, batch.active)?,
                batch.active,
            );
        }
        let reference = scalar.choose_move(observation)?;
        let actual = selected.choose_move(observation)?;
        parity.decisions += 1;
        parity.moves_matched += usize::from(reference.r#move == actual.r#move);
        if actual.actor != observation.actor
            || actual.observation_key != observation.observation_key
            || !observation
                .legal_actions
                .iter()
                .any(|legal| legal.r#move == actual.r#move)
        {
            return Err("Optimized decision has wrong actor/key or non-legal move".into());
        }
    }
    // 準備・検証・warmupは計時外。全て同じ immutable artifact と公開 Observation。
    encode_batch(&observations)?;
    predict_batch(&scalar, &batches)?;
    predict_batch(&selected, &batches)?;
    choose_batch(&scalar, &observations)?;
    choose_batch(&selected, &observations)?;
    let mut encoding = Vec::with_capacity(config.rounds);
    let mut scalar_predict = Vec::with_capacity(config.rounds);
    let mut selected_predict = Vec::with_capacity(config.rounds);
    let mut scalar_choose = Vec::with_capacity(config.rounds);
    let mut selected_choose = Vec::with_capacity(config.rounds);
    for round in 0..config.rounds {
        encoding.push(time(config.iterations, || encode_batch(&observations))?);
        // 実行順を交互にし、同じ backend が常に最初になる偏りを避ける。
        if round % 2 == 0 {
            scalar_predict.push(time(config.iterations, || {
                predict_batch(&scalar, &batches)
            })?);
            selected_predict.push(time(config.iterations, || {
                predict_batch(&selected, &batches)
            })?);
            scalar_choose.push(time(config.iterations, || {
                choose_batch(&scalar, &observations)
            })?);
            selected_choose.push(time(config.iterations, || {
                choose_batch(&selected, &observations)
            })?);
        } else {
            selected_predict.push(time(config.iterations, || {
                predict_batch(&selected, &batches)
            })?);
            scalar_predict.push(time(config.iterations, || {
                predict_batch(&scalar, &batches)
            })?);
            selected_choose.push(time(config.iterations, || {
                choose_batch(&selected, &observations)
            })?);
            scalar_choose.push(time(config.iterations, || {
                choose_batch(&scalar, &observations)
            })?);
        }
    }
    Ok(json!({
        "schema": "tzolkin-ml-performance-v1",
        "buildProfile": if cfg!(debug_assertions) { "debug" } else { "optimized" },
        "targetArch": std::env::consts::ARCH,
        "targetOs": std::env::consts::OS,
        "requestedKernel": format!("{:?}", config.kernel),
        "backend": selected.backend(),
        "scalarBackend": scalar.backend(),
        "modelChecksum": artifact.checksum,
        "modelVersion": artifact.model_version,
        "featureSchema": artifact.feature_schema,
        "catalogHash": artifact.catalog_hash,
        "players": config.players, "seed": config.seed, "iterations": config.iterations,
        "rounds": config.rounds, "corpusHash": format!("{:016x}", fingerprint(&corpus_bytes)),
        "observations": observations.len(), "featureVectors": feature_count,
        "parity": { "predictions": parity.predictions, "decisions": parity.decisions,
            "movesMatched": parity.moves_matched, "allMovesMatched": parity.moves_matched == parity.decisions,
            "maxPolicyAbsError": parity.max_policy_abs_error, "maxValueAbsError": parity.max_value_abs_error,
            "numericalWithinTolerance": parity.numerical_within_tolerance,
            "inactiveUtilitiesZero": parity.inactive_utilities_zero,
            "policyTolerance": "5e-5 * (1 + abs(scalar logit))", "valueAbsTolerance": 0.00005 },
        "featureEncoding": timings("featureEncoding", &encoding, feature_count * config.iterations)?,
        "scalarPredict": timings("scalarPredict", &scalar_predict, feature_count * config.iterations)?,
        "selectedPredict": timings("selectedPredict", &selected_predict, feature_count * config.iterations)?,
        "scalarChoose": timings("scalarChoose", &scalar_choose, observations.len() * config.iterations)?,
        "selectedChoose": timings("selectedChoose", &selected_choose, observations.len() * config.iterations)?,
        "strengthMeasured": false,
        "humanReplayCorpus": false,
        "trainingKernel": "scalar",
    }))
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args == ["--help"] {
        println!("{USAGE}");
        return;
    }
    match config(&args).and_then(run) {
        Ok(report) => println!(
            "{}",
            serde_json::to_string_pretty(&report).expect("JSON report is serializable")
        ),
        Err(error) => {
            eprintln!("{error}\n{USAGE}");
            std::process::exit(1);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn timing_report_rejects_invalid_clock_samples_and_nonfinite_rates() {
        for invalid in [0.0, -0.0, -1.0, f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            // A positive median does not make an invalid round acceptable.
            let error = timings("scalarPredict", &[1.0, invalid, 2.0], 10).unwrap_err();
            assert!(error.starts_with("scalarPredict:"));
        }
        assert!(timings("empty", &[], 10).is_err());
        assert!(timings("zero work", &[1.0], 0).is_err());
        assert!(timings("rate overflow", &[f64::MIN_POSITIVE], usize::MAX).is_err());
        assert!(timings("ns overflow", &[f64::MAX], 1).is_err());
        let report = timings("valid", &[3.0, 1.0, 2.0], 4).unwrap();
        assert_eq!(report["medianSeconds"], 2.0);
        assert_eq!(report["minSeconds"], 1.0);
        assert_eq!(report["maxSeconds"], 3.0);
        assert_eq!(report["operationsPerSecond"], 2.0);
        assert_eq!(report["meanNsPerOperation"], 500_000_000.0);
        assert_eq!(report["roundSeconds"], json!([3.0, 1.0, 2.0]));
    }

    #[test]
    fn rejects_missing_unknown_duplicate_and_unbounded_arguments() {
        let parse =
            |items: &[&str]| config(&items.iter().map(|s| (*s).to_owned()).collect::<Vec<_>>());
        assert!(parse(&[]).is_err());
        assert!(parse(&["--model"]).is_err());
        assert!(parse(&["--model", "x", "--model", "y"]).is_err());
        assert!(parse(&["--model", "x", "--player", "3"]).is_err());
        for (flag, value) in [
            ("--players", "2"),
            ("--iterations", "0"),
            ("--iterations", "1001"),
            ("--rounds", "16"),
            ("--seed", "4294967296"),
        ] {
            assert!(parse(&["--model", "x", flag, value]).is_err());
        }
        assert!(parse(&["--model", "x", "--players", "3", "--kernel", "sse2"]).is_ok());
    }
}
