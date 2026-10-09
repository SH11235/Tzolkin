use std::io::{Read, Write};
use std::path::Path;

use tzolkin_ai::dataset::DatasetSplit;
use tzolkin_ai::kernel::Kernel;
use tzolkin_ai::policy_dataset::{export_native_files, load_policy_dataset, native_source_files};
use tzolkin_ai::policy_training::{
    PolicyBcConfig, PolicyTrainingCheckpoint, evaluate_dataset, load_evaluation_model,
    train_dataset, validate_new_output_directory,
};
use tzolkin_ai::public_model::{LoadedPublicPolicy, MAX_OBSERVATION_BYTES, PublicPolicyArtifact};
use tzolkin_core::observation::Observation;

const HELP: &str = "tzolkin-public-ml choose --model PATH [--kernel scalar|auto|avx2|sse2|neon|simd128]\ntzolkin-public-ml export-native --input NATIVE_REPLAY_DIRECTORY --output NEW_DATASET_DIRECTORY\ntzolkin-public-ml train --input DATASET --output NEW_DIRECTORY [--epochs N --batch-size N --learning-rate F --seed N]\ntzolkin-public-ml resume --input DATASET --checkpoint PATH --epochs TOTAL --output NEW_DIRECTORY [--batch-size N --learning-rate F --seed N]\ntzolkin-public-ml evaluate --input DATASET --model PATH [--split train|validation|test]\nchoose reads a bounded core Observation JSON from stdin and writes a Decision JSON.\nChoose defaults to scalar. BC training/resume/evaluation are Scalar only, base 3-4p, policy-only, value unavailable.\nTrain defaults: epochs=3, batch-size=16, learning-rate=0.001, seed=7. Resume inherits unchanged optimizer config; --epochs is required. Evaluate defaults to validation.\nHuman admission, selfplay/Arena registration and PPO/critics are separate future units.";

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

fn run() -> Result<(), String> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.is_empty() || args == ["help"] || args == ["--help"] {
        println!("{HELP}");
        return Ok(());
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
