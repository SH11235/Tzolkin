//! Dedicated native schema-4 teacher BC; no Human, legacy model or RL admission.
use std::collections::BTreeMap;
use std::fs::File;
use std::io::Read;
use std::path::Path;
use tzolkin_ai::policy_training::{PolicyBcConfig, validate_new_output_directory};
use tzolkin_ai::semantic_playing_teacher_dataset::load_semantic_playing_teacher_dataset;
use tzolkin_ai::semantic_playing_teacher_training::{
    SemanticPlayingTeacherCheckpoint, train_dataset,
};

const USAGE: &str = "Usage: tzolkin-semantic-playing-teacher-train --dataset DATASET_DIR --config CONFIG.json --output NEW_OUTPUT_DIR [--resume CHECKPOINT.json]\nConfig requires epochs, batchSize, learningRate and seed. Epochs is the total target, including resumed epochs. This is fixed-teacher imitation, not strength evaluation.";
const MAX_CONFIG_BYTES: u64 = 64 * 1024;

fn config(path: &Path) -> Result<PolicyBcConfig, String> {
    let file = File::open(path).map_err(|e| e.to_string())?;
    let meta = file.metadata().map_err(|e| e.to_string())?;
    if !meta.is_file() || meta.len() > MAX_CONFIG_BYTES {
        return Err("BC config must be a bounded regular JSON file".into());
    }
    let mut bytes = Vec::new();
    file.take(MAX_CONFIG_BYTES + 1)
        .read_to_end(&mut bytes)
        .map_err(|e| e.to_string())?;
    if bytes.len() as u64 != meta.len() || bytes.len() as u64 > MAX_CONFIG_BYTES {
        return Err("BC config changed or exceeded its byte cap".into());
    }
    // The typed closed struct rejects unknown, duplicate and missing fields.
    let config: PolicyBcConfig = serde_json::from_slice(&bytes).map_err(|e| e.to_string())?;
    config.validate()?;
    Ok(config)
}

fn run() -> Result<(), String> {
    let mut args = std::env::args().skip(1).peekable();
    if args.peek().is_some_and(|arg| arg == "--help") {
        args.next();
        if args.next().is_some() {
            return Err(USAGE.into());
        }
        println!("{USAGE}");
        return Ok(());
    }
    let mut flags = BTreeMap::new();
    while let Some(flag) = args.next() {
        if !matches!(
            flag.as_str(),
            "--dataset" | "--config" | "--output" | "--resume"
        ) {
            return Err(format!("Unknown flag: {flag}\n{USAGE}"));
        }
        let value = args.next().ok_or_else(|| format!("Missing {flag} value"))?;
        if value.starts_with("--") {
            return Err(format!("Missing {flag} value"));
        }
        if flags.insert(flag, value).is_some() {
            return Err("Duplicate flag".into());
        }
    }
    let mut required = |name: &str| {
        flags
            .remove(name)
            .ok_or_else(|| format!("Missing {name}\n{USAGE}"))
    };
    let directory = required("--dataset")?;
    let config_path = required("--config")?;
    let output_path = required("--output")?;
    let resume_path = flags.remove("--resume");
    let directory = Path::new(&directory);
    let output = Path::new(&output_path);
    validate_new_output_directory(output)?;
    let parent = output
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    let destination = parent
        .canonicalize()
        .map_err(|e| e.to_string())?
        .join(output.file_name().ok_or("Missing output directory name")?);
    if destination.starts_with(directory.canonicalize().map_err(|e| e.to_string())?) {
        return Err("BC output must be outside the immutable teacher dataset".into());
    }
    let config = config(Path::new(&config_path))?;
    let resume = resume_path
        .map(|p| SemanticPlayingTeacherCheckpoint::load(Path::new(&p)))
        .transpose()?;
    let dataset = load_semantic_playing_teacher_dataset(directory)?;
    let outcome = train_dataset(&dataset, &config, resume.as_ref())?;
    outcome.save_new_directory(output)?;
    println!(
        "{}",
        serde_json::to_string(&outcome.metrics).map_err(|e| e.to_string())?
    );
    Ok(())
}

fn main() {
    if let Err(error) = run() {
        eprintln!("{error}");
        std::process::exit(1);
    }
}
