use std::io::{Read, Write};
use std::path::Path;
use std::time::Instant;

use tzolkin_ai::dataset::DatasetSplit;
use tzolkin_ai::kernel::Kernel;
use tzolkin_ai::policy_dataset::{export_native_files, load_policy_dataset, native_source_files};
use tzolkin_ai::policy_training::{
    PolicyBcConfig, PolicyTrainingCheckpoint, evaluate_dataset, load_evaluation_model,
    train_dataset, validate_new_output_directory,
};
use tzolkin_ai::public_model::{LoadedPublicPolicy, MAX_OBSERVATION_BYTES, PublicPolicyArtifact};
use tzolkin_ai::public_native::{self, PreparedPublicPolicy};
use tzolkin_ai::public_trade_guard;
use tzolkin_core::observation::Observation;

const HELP: &str = "tzolkin-public-ml choose --model PATH [--kernel scalar|auto|avx2|sse2|neon|simd128]\ntzolkin-public-ml export-native --input NATIVE_REPLAY_DIRECTORY --output NEW_DATASET_DIRECTORY\ntzolkin-public-ml train --input DATASET --output NEW_DIRECTORY [--epochs N --batch-size N --learning-rate F --seed N]\ntzolkin-public-ml resume --input DATASET --checkpoint PATH --epochs TOTAL --output NEW_DIRECTORY [--batch-size N --learning-rate F --seed N]\ntzolkin-public-ml evaluate --input DATASET --model PATH [--split train|validation|test]\ntzolkin-public-ml selfplay --checkpoint PATH --dataset DATASET --seats all|SEAT[,SEAT] [--players 3|4 --seed N --flags 0 --kernel scalar|auto|avx2|sse2|neon|simd128 --output NEW_REPLAY_FILE]\nchoose reads a bounded core Observation JSON from stdin and writes a Decision JSON.\nChoose defaults to scalar. BC training/resume/evaluation are Scalar only, base 3-4p, policy-only, value unavailable.\nTrain defaults: epochs=3, batch-size=16, learning-rate=0.001, seed=7. Resume inherits unchanged optimizer config; --epochs is required. Evaluate defaults to validation.\nSelfplay defaults: players=3, seed=0, flags=0, kernel=scalar; explicit --seats is required. Arena also accepts publicLearned configs. Human admission, PPO, calibrated estimates and CPU adoption are separate units.";

fn write_stdout(value: &impl serde::Serialize) -> Result<(), String> {
    let mut stdout = std::io::stdout().lock();
    serde_json::to_writer(&mut stdout, value).map_err(|error| error.to_string())?;
    stdout.write_all(b"\n").map_err(|error| error.to_string())
}

fn flags<'a>(
    args: &'a [String],
    allowed: &[&str],
) -> Result<std::collections::BTreeMap<&'a str, &'a str>, String> {
    let mut parsed = std::collections::BTreeMap::new();
    let mut index = 1;
    while index < args.len() {
        let name = args[index].as_str();
        if !allowed.contains(&name) {
            return Err(format!("Unknown {} flag {name}", args[0]));
        }
        let value = args
            .get(index + 1)
            .filter(|value| !value.starts_with("--"))
            .ok_or_else(|| format!("Missing value after {name}"))?;
        if parsed.insert(name, value.as_str()).is_some() {
            return Err(format!("Repeated flag {name}"));
        }
        index += 2;
    }
    Ok(parsed)
}

fn train(args: &[String]) -> Result<(), String> {
    let resume = args[0] == "resume";
    let allowed = if resume {
        &[
            "--input",
            "--output",
            "--checkpoint",
            "--epochs",
            "--batch-size",
            "--learning-rate",
            "--seed",
        ][..]
    } else {
        &[
            "--input",
            "--output",
            "--epochs",
            "--batch-size",
            "--learning-rate",
            "--seed",
        ][..]
    };
    let parsed = flags(args, allowed)?;
    let input = parsed.get("--input").ok_or("--input is required")?;
    let output = parsed.get("--output").ok_or("--output is required")?;
    validate_new_output_directory(Path::new(output))?;
    let checkpoint = if resume {
        if !parsed.contains_key("--epochs") {
            return Err("Resume requires --epochs TOTAL".into());
        }
        Some(PolicyTrainingCheckpoint::load(Path::new(
            parsed
                .get("--checkpoint")
                .ok_or("--checkpoint is required")?,
        ))?)
    } else {
        None
    };
    let mut config = checkpoint
        .as_ref()
        .map_or_else(PolicyBcConfig::default, |p| p.config.clone());
    if let Some(value) = parsed.get("--epochs") {
        config.epochs = value.parse().map_err(|_| "Invalid --epochs")?;
    }
    if let Some(value) = parsed.get("--batch-size") {
        config.batch_size = value.parse().map_err(|_| "Invalid --batch-size")?;
    }
    if let Some(value) = parsed.get("--learning-rate") {
        config.learning_rate = value.parse().map_err(|_| "Invalid --learning-rate")?;
    }
    if let Some(value) = parsed.get("--seed") {
        config.seed = value.parse().map_err(|_| "Invalid --seed")?;
    }
    config.validate()?;
    let dataset = load_policy_dataset(Path::new(input))?;
    let outcome = train_dataset(&dataset, &config, checkpoint.as_ref())?;
    outcome.save_new_directory(Path::new(output))?;
    write_stdout(&outcome.metrics)
}

fn evaluate(args: &[String]) -> Result<(), String> {
    let parsed = flags(args, &["--input", "--model", "--split"])?;
    let input = parsed.get("--input").ok_or("--input is required")?;
    let model = parsed.get("--model").ok_or("--model is required")?;
    let split = match parsed.get("--split").copied().unwrap_or("validation") {
        "train" => DatasetSplit::Train,
        "validation" => DatasetSplit::Validation,
        "test" => DatasetSplit::Test,
        _ => return Err("Unknown evaluation --split".into()),
    };
    let model = load_evaluation_model(Path::new(model))?;
    let dataset = load_policy_dataset(Path::new(input))?;
    write_stdout(&evaluate_dataset(&dataset, &model, split)?)
}

fn selfplay(args: &[String]) -> Result<(), String> {
    let parsed = flags(
        args,
        &[
            "--checkpoint",
            "--dataset",
            "--players",
            "--seed",
            "--flags",
            "--seats",
            "--kernel",
            "--output",
            "--trade-guard-config",
        ],
    )?;
    let checkpoint = parsed
        .get("--checkpoint")
        .ok_or("--checkpoint is required")?;
    let dataset = parsed.get("--dataset").ok_or("--dataset is required")?;
    let players: usize = parsed
        .get("--players")
        .copied()
        .unwrap_or("3")
        .parse()
        .map_err(|_| "Invalid --players")?;
    let seed: u32 = parsed
        .get("--seed")
        .copied()
        .unwrap_or("0")
        .parse()
        .map_err(|_| "Invalid --seed")?;
    let flag_mask: u8 = parsed
        .get("--flags")
        .copied()
        .unwrap_or("0")
        .parse()
        .map_err(|_| "Invalid --flags")?;
    if flag_mask != 0 {
        return Err("Public-policy selfplay supports --flags 0 only".into());
    }
    let seats = match *parsed
        .get("--seats")
        .ok_or("--seats all|SEAT[,SEAT] is required")?
    {
        "all" => (0..players.min(4)).collect::<Vec<_>>(),
        selected => {
            if selected.split(',').count() > 4 {
                return Err("Too many selected seats".into());
            }
            selected
                .split(',')
                .map(|value| value.parse::<usize>().map_err(|_| "Invalid --seats"))
                .collect::<Result<Vec<_>, _>>()?
        }
    };
    public_native::validate_seats(players, &tzolkin_core::GameOptions::default(), &seats)?;
    let output = parsed.get("--output").copied();
    if let Some(output) = output {
        validate_new_output_directory(Path::new(output))
            .map_err(|error| format!("Replay output must be a new local file: {error}"))?;
    }
    let kernel = public_native::parse_kernel(parsed.get("--kernel").copied().unwrap_or("scalar"))?;
    let guard = parsed
        .get("--trade-guard-config")
        .map(|path| public_trade_guard::load_config(Path::new(path)))
        .transpose()?;
    let prepared = PreparedPublicPolicy::load(Path::new(checkpoint), Path::new(dataset), kernel)?;
    let handle = prepared.handle()?;
    let guarded_provenance = guard
        .as_ref()
        .map(|guard| handle.guarded_provenance(guard))
        .transpose()?;
    // Dataset/model qualification and backend resolution are excluded from the game clock.
    let started = Instant::now();
    let (mut summary, failure) = if let Some(guard) = guard {
        let outcome = public_native::play_game_guarded_with_summary(
            &handle,
            players,
            seed,
            tzolkin_core::GameOptions::default(),
            &seats,
            output.is_some(),
            guard,
        )?;
        guarded_selfplay_report(
            players,
            seed,
            &seats,
            guarded_provenance
                .as_ref()
                .expect("prepared guarded provenance"),
            outcome.game,
            &outcome.trade_guard,
            output,
        )?
    } else {
        let played = public_native::play_game(
            &handle,
            players,
            seed,
            tzolkin_core::GameOptions::default(),
            &seats,
            output.is_some(),
        );
        selfplay_report(players, seed, &seats, handle.provenance(), played, output)?
    };
    summary["elapsedMs"] = (started.elapsed().as_secs_f64() * 1000.0).into();
    write_stdout(&summary)?;
    if let Some(error) = failure {
        return Err(error);
    }
    Ok(())
}

type PlayedGame = Result<
    (
        tzolkin_core::GameState,
        usize,
        Option<tzolkin_ai::replay::GameReplay>,
    ),
    String,
>;

fn guarded_selfplay_report(
    players: usize,
    seed: u32,
    seats: &[usize],
    policy: &tzolkin_ai::replay::SeatPolicy,
    played: PlayedGame,
    trade_guard: &impl serde::Serialize,
    output: Option<&str>,
) -> Result<(serde_json::Value, Option<String>), String> {
    let (mut summary, failure) = selfplay_report(players, seed, seats, policy, played, output)?;
    summary["schema"] = "tzolkin-public-policy-trade-guard-selfplay-v1".into();
    summary["tradeGuard"] = serde_json::to_value(trade_guard).map_err(|e| e.to_string())?;
    summary["tradeGuardCounts"] =
        "observed pre-apply choices; failed-game applied totals unavailable".into();
    Ok((summary, failure))
}

// Keep completion and publication separate: a late save failure must retain the completed result.
fn selfplay_report(
    players: usize,
    seed: u32,
    seats: &[usize],
    policy: &tzolkin_ai::replay::SeatPolicy,
    played: PlayedGame,
    output: Option<&str>,
) -> Result<(serde_json::Value, Option<String>), String> {
    let mut summary = serde_json::json!({
        "schema":"tzolkin-public-policy-selfplay-v1", "players":players, "seed":seed,
        "options":tzolkin_core::GameOptions::default(), "seats":seats, "policy":policy,
        "complete":false, "success":false, "decisions":null, "finalScores":[], "finalState":null,
        "error":null, "publication":null
    });
    let failure = match played {
        Ok((state, decisions, record)) => {
            summary["complete"] = true.into();
            summary["decisions"] = decisions.into();
            summary["finalScores"] =
                serde_json::to_value(&state.final_scores).map_err(|e| e.to_string())?;
            summary["finalState"] = tzolkin_ai::replay::state_key(&state)?.into();
            let publication = output.map(|path| {
                let result = record
                    .as_ref()
                    .ok_or("Missing public-policy replay".to_owned())
                    .and_then(|record| {
                        tzolkin_ai::replay::save_replay_new(Path::new(path), record)
                    });
                (path, result.err())
            });
            let failure = if let Some((path, error)) = publication {
                summary["publication"] =
                    serde_json::json!({"path":path,"success":error.is_none(),"error":error});
                error
            } else {
                None
            };
            summary["success"] = failure.is_none().into();
            failure
        }
        Err(error) => {
            summary["error"] = error.clone().into();
            Some(error)
        }
    };
    Ok((summary, failure))
}

fn export(args: &[String]) -> Result<(), String> {
    let mut input = None;
    let mut output = None;
    let mut index = 1;
    while index < args.len() {
        let name = &args[index];
        let slot = match name.as_str() {
            "--input" => &mut input,
            "--output" => &mut output,
            _ => return Err(format!("Unknown export-native flag {name}")),
        };
        let value = args
            .get(index + 1)
            .filter(|value| !value.starts_with("--"))
            .ok_or_else(|| format!("Missing value after {name}"))?;
        if slot.replace(value.as_str()).is_some() {
            return Err(format!("Repeated flag {name}"));
        }
        index += 2;
    }
    let paths = native_source_files(Path::new(input.ok_or("--input is required")?))?;
    let manifest = export_native_files(&paths, Path::new(output.ok_or("--output is required")?))?;
    serde_json::to_writer(std::io::stdout().lock(), &manifest).map_err(|e| e.to_string())?;
    println!();
    Ok(())
}

fn export_state_native(args: &[String]) -> Result<(), String> {
    let parsed = flags(args, &["--input", "--output"])?;
    let paths = tzolkin_ai::state_mc_dataset::native_source_files(Path::new(
        parsed.get("--input").ok_or("--input is required")?,
    ))?;
    let manifest = tzolkin_ai::state_mc_dataset::export_native_files(
        &paths,
        Path::new(parsed.get("--output").ok_or("--output is required")?),
    )?;
    write_stdout(&manifest)
}

fn train_state(args: &[String]) -> Result<(), String> {
    use tzolkin_ai::state_mc_training::{self, StateMcConfig, StateMcTrainingCheckpoint};
    let resume = args[0] == "resume-state";
    let allowed = if resume {
        &[
            "--input",
            "--output",
            "--checkpoint",
            "--epochs",
            "--batch-size",
            "--learning-rate",
            "--seed",
        ][..]
    } else {
        &[
            "--input",
            "--output",
            "--epochs",
            "--batch-size",
            "--learning-rate",
            "--seed",
        ][..]
    };
    let parsed = flags(args, allowed)?;
    let input = parsed.get("--input").ok_or("--input is required")?;
    let output = parsed.get("--output").ok_or("--output is required")?;
    state_mc_training::validate_new_output_directory(Path::new(output))?;
    let checkpoint = if resume {
        if !parsed.contains_key("--epochs") {
            return Err("Resume-state requires --epochs TOTAL".into());
        }
        Some(StateMcTrainingCheckpoint::load(Path::new(
            parsed
                .get("--checkpoint")
                .ok_or("--checkpoint is required")?,
        ))?)
    } else {
        None
    };
    let mut config = checkpoint
        .as_ref()
        .map_or_else(StateMcConfig::default, |p| p.config().clone());
    if let Some(value) = parsed.get("--epochs") {
        config.epochs = value.parse().map_err(|_| "Invalid --epochs")?;
    }
    if let Some(value) = parsed.get("--batch-size") {
        config.batch_size = value.parse().map_err(|_| "Invalid --batch-size")?;
    }
    if let Some(value) = parsed.get("--learning-rate") {
        config.learning_rate = value.parse().map_err(|_| "Invalid --learning-rate")?;
    }
    if let Some(value) = parsed.get("--seed") {
        config.seed = value.parse().map_err(|_| "Invalid --seed")?;
    }
    config.validate()?;
    let dataset = tzolkin_ai::state_mc_dataset::load_state_mc_dataset(Path::new(input))?;
    let outcome = state_mc_training::train_dataset(&dataset, &config, checkpoint.as_ref())?;
    outcome.save_new_directory(Path::new(output))?;
    write_stdout(&serde_json::json!({"metrics":outcome.metrics(),"workPlan":outcome.work_plan()}))
}

fn evaluate_state(args: &[String]) -> Result<(), String> {
    let parsed = flags(args, &["--input", "--checkpoint", "--split"])?;
    let input = parsed.get("--input").ok_or("--input is required")?;
    let checkpoint = parsed
        .get("--checkpoint")
        .ok_or("--checkpoint is required")?;
    let split = match parsed.get("--split").copied().unwrap_or("validation") {
        "train" => DatasetSplit::Train,
        "validation" => DatasetSplit::Validation,
        "test" => DatasetSplit::Test,
        _ => return Err("Unknown state evaluation --split".into()),
    };
    let checkpoint =
        tzolkin_ai::state_mc_training::StateMcTrainingCheckpoint::load(Path::new(checkpoint))?;
    let dataset = tzolkin_ai::state_mc_dataset::load_state_mc_dataset(Path::new(input))?;
    write_stdout(&tzolkin_ai::state_mc_training::evaluate_dataset(
        &dataset,
        &checkpoint,
        split,
    )?)
}

fn run() -> Result<(), String> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.is_empty() || args == ["help"] || args == ["--help"] {
        println!("{HELP}");
        println!(
            "tzolkin-public-ml export-state-native --input NATIVE_REPLAY_DIRECTORY --output NEW_STATE_MC_DIRECTORY\ntzolkin-public-ml train-state --input STATE_MC_DATASET --output NEW_DIRECTORY [--epochs N --batch-size N --learning-rate F --seed N]\ntzolkin-public-ml resume-state --input STATE_MC_DATASET --checkpoint PATH --epochs TOTAL --output NEW_DIRECTORY [--batch-size N --learning-rate F --seed N]\ntzolkin-public-ml evaluate-state --input STATE_MC_DATASET --checkpoint PATH [--split train|validation|test]\nState-MC is a separate Scalar complete-native gamma=1/lambda=1 raw state-MSE task. Defaults3/16/.001/7; resume inherits config except required total epochs; evaluation defaultsValidation. Estimates remain unqualified. PPO and CPU adoption are separate units."
        );
        println!(
            "Opt-in guarded selfplay adds --trade-guard-config LOCAL_FILE (closed config, at most 64 KiB); this flag is rejected by every other command. Arena also accepts the distinct publicLearnedTradeGuard kind. Guard diagnostics count observed pre-apply choices and retain failures; they do not qualify data for training or establish strength."
        );
        return Ok(());
    }
    if matches!(
        args.first().map(String::as_str),
        Some("train-state" | "resume-state")
    ) {
        return train_state(&args);
    }
    if args.first().map(String::as_str) == Some("evaluate-state") {
        return evaluate_state(&args);
    }
    if args.first().map(String::as_str) == Some("export-state-native") {
        return export_state_native(&args);
    }
    if args.first().map(String::as_str) == Some("export-native") {
        return export(&args);
    }
    if matches!(args.first().map(String::as_str), Some("train" | "resume")) {
        return train(&args);
    }
    if args.first().map(String::as_str) == Some("evaluate") {
        return evaluate(&args);
    }
    if args.first().map(String::as_str) == Some("selfplay") {
        return selfplay(&args);
    }
    if args.first().map(String::as_str) != Some("choose") {
        return Err("Unknown public ML command; use --help".into());
    }
    let mut model = None;
    let mut kernel = None;
    let mut index = 1;
    while index < args.len() {
        let name = &args[index];
        if !["--model", "--kernel"].contains(&name.as_str()) {
            return Err(format!("Unknown flag {name}"));
        }
        let value = args
            .get(index + 1)
            .filter(|value| !value.starts_with("--"))
            .ok_or_else(|| format!("Missing value after {name}"))?;
        let slot = if name == "--model" {
            &mut model
        } else {
            &mut kernel
        };
        if slot.replace(value.as_str()).is_some() {
            return Err(format!("Repeated flag {name}"));
        }
        index += 2;
    }
    let kernel = match kernel.unwrap_or("scalar") {
        "scalar" => Kernel::Scalar,
        "auto" => Kernel::Auto,
        "avx2" => Kernel::Avx2,
        "sse2" => Kernel::Sse2,
        "neon" => Kernel::Neon,
        "simd128" => Kernel::Simd128,
        _ => return Err("Unknown public policy inference kernel".into()),
    };
    let artifact = PublicPolicyArtifact::load(Path::new(model.ok_or("--model is required")?))?;
    let loaded = LoadedPublicPolicy::with_kernel(&artifact, kernel)?;
    let mut input = Vec::new();
    std::io::stdin()
        .lock()
        .take(MAX_OBSERVATION_BYTES as u64 + 1)
        .read_to_end(&mut input)
        .map_err(|error| error.to_string())?;
    if input.len() > MAX_OBSERVATION_BYTES {
        return Err("Observation stdin exceeds bounded 16 MiB input".into());
    }
    let observation: Observation =
        serde_json::from_slice(&input).map_err(|error| error.to_string())?;
    let decision = loaded.choose_move(&observation)?;
    let mut stdout = std::io::stdout().lock();
    serde_json::to_writer(&mut stdout, &decision).map_err(|error| error.to_string())?;
    stdout.write_all(b"\n").map_err(|error| error.to_string())
}
fn main() {
    if let Err(error) = run() {
        eprintln!("{error}");
        std::process::exit(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicU64, Ordering};

    struct Temp(PathBuf);
    impl Temp {
        fn new() -> Self {
            static NEXT: AtomicU64 = AtomicU64::new(0);
            let path = std::env::temp_dir().join(format!(
                "tzolkin-public-selfplay-report-{}-{}",
                std::process::id(),
                NEXT.fetch_add(1, Ordering::Relaxed)
            ));
            fs::create_dir(&path).unwrap();
            Self(path)
        }
    }
    impl Drop for Temp {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    #[test]
    fn g2_guarded_report_retains_observed_counts_on_game_and_late_publication_errors() {
        let temp = Temp::new();
        let destination = temp.0.join("guarded-report-only.json");
        let observed = serde_json::json!([{"choicesReturned":9,"failedChoices":1},null,null]);
        let provenance = tzolkin_ai::experiment::heuristic_seat();
        // Report serialization/publication only, not a qualified NN producer or
        // a real Guard trace. Library tests cover actual observed summaries.
        let (report, error) = guarded_selfplay_report(
            3,
            17,
            &[0],
            &provenance,
            Err("controlled NN error".into()),
            &observed,
            Some(destination.to_str().unwrap()),
        )
        .unwrap();
        assert_eq!(
            report["schema"],
            "tzolkin-public-policy-trade-guard-selfplay-v1"
        );
        assert_eq!(report["tradeGuard"], observed);
        assert_eq!(report["decisions"], serde_json::Value::Null);
        assert_eq!(report["finalState"], serde_json::Value::Null);
        assert_eq!(report["publication"], serde_json::Value::Null);
        assert!(!destination.exists());
        assert_eq!(error.as_deref(), Some("controlled NN error"));
        validate_new_output_directory(&destination).unwrap();
        fs::write(&destination, b"existing immutable file").unwrap();
        let game = tzolkin_ai::replay::play_game_fast(3, 17, Default::default(), true).unwrap();
        println!(
            "G2 report publication fixture seed17,3p callbacks{}",
            game.1
        );
        let decisions = game.1;
        let (report, error) = guarded_selfplay_report(
            3,
            17,
            &[0],
            &provenance,
            Ok(game),
            &observed,
            Some(destination.to_str().unwrap()),
        )
        .unwrap();
        assert!(error.is_some());
        assert_eq!(report["complete"], true);
        assert_eq!(report["success"], false);
        assert_eq!(report["decisions"], decisions);
        assert_eq!(report["tradeGuard"], observed);
        assert_eq!(report["publication"]["success"], false);
        assert_eq!(fs::read(&destination).unwrap(), b"existing immutable file");
    }

    #[test]
    fn late_publication_failure_keeps_the_complete_result_and_preserves_existing_bytes() {
        let temp = Temp::new();
        let destination = temp.0.join("replay.json");
        // The destination appears after a successful CLI-style preflight, without a timer/race.
        validate_new_output_directory(&destination).unwrap();
        fs::write(&destination, b"existing source").unwrap();
        let played = tzolkin_ai::replay::play_game_fast(3, 0, Default::default(), true).unwrap();
        let expected_scores = serde_json::to_value(&played.0.final_scores).unwrap();
        let expected_key = tzolkin_ai::replay::state_key(&played.0).unwrap();
        let expected_decisions = played.1;
        // This publication-only fixture uses its actual heuristic provenance; it claims no V2 producer.
        let provenance = tzolkin_ai::experiment::heuristic_seat();
        let (report, error) = selfplay_report(
            3,
            0,
            &[0],
            &provenance,
            Ok(played),
            Some(destination.to_str().unwrap()),
        )
        .unwrap();
        assert!(error.is_some());
        assert_eq!(report["complete"], true);
        assert_eq!(report["success"], false);
        assert_eq!(report["decisions"], expected_decisions);
        assert_eq!(report["finalScores"], expected_scores);
        assert_eq!(report["finalState"], expected_key);
        assert_eq!(report["error"], serde_json::Value::Null);
        assert_eq!(report["publication"]["success"], false);
        assert_eq!(report["publication"]["error"].as_str(), error.as_deref());
        assert_eq!(fs::read(&destination).unwrap(), b"existing source");
    }

    #[test]
    fn failed_game_has_null_results_and_does_not_publish_a_replay() {
        let temp = Temp::new();
        let destination = temp.0.join("not-published.json");
        let provenance = tzolkin_ai::experiment::heuristic_seat();
        let failure = "Game did not finish within the native decision bound".to_owned();
        let (report, error) = selfplay_report(
            3,
            0,
            &[0],
            &provenance,
            Err(failure.clone()),
            Some(destination.to_str().unwrap()),
        )
        .unwrap();
        assert_eq!(error, Some(failure.clone()));
        assert_eq!(report["error"], failure);
        assert_eq!(report["complete"], false);
        assert_eq!(report["success"], false);
        assert_eq!(report["decisions"], serde_json::Value::Null);
        assert_eq!(report["finalState"], serde_json::Value::Null);
        assert_eq!(report["finalScores"], serde_json::json!([]));
        assert_eq!(report["publication"], serde_json::Value::Null);
        assert!(!destination.exists());
    }
}
