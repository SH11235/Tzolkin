//! Dedicated conditional Human policy-only BC; no native dataset or owner admission.
use std::path::Path;
use tzolkin_ai::human_policy_training::{
    ConditionalHumanPolicyCheckpoint, inspect_semantic_playing_collection_manifest,
    train_collection_manifest, train_semantic_playing_collection_manifest,
};
use tzolkin_ai::policy_training::validate_new_output_directory;

fn run() -> Result<(), String> {
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    if args
        .first()
        .is_some_and(|arg| arg == "--check-semantic-playing4")
    {
        if args.len() != 2 {
            return Err("Usage: tzolkin-conditional-human-policy-train --check-semantic-playing4 INPUT.json (read only; no model or optimizer)".into());
        }
        let inspection = inspect_semantic_playing_collection_manifest(Path::new(&args[1]))?;
        println!(
            "{}",
            serde_json::to_string(&inspection).map_err(|e| e.to_string())?
        );
        return Ok(());
    }
    if args.first().is_some_and(|arg| arg == "--semantic-playing4") {
        if args.len() != 3 {
            return Err("Usage: tzolkin-conditional-human-policy-train --semantic-playing4 INPUT.json NEW_OUTPUT_DIR (fresh only)".into());
        }
        let output = Path::new(&args[2]);
        validate_new_output_directory(output)?;
        let outcome = train_semantic_playing_collection_manifest(Path::new(&args[1]))?;
        outcome.save_new_semantic_playing_directory(output)?;
        println!(
            "{}",
            serde_json::to_string(&outcome.metrics).map_err(|e| e.to_string())?
        );
        return Ok(());
    }
    if !(2..=3).contains(&args.len()) {
        return Err("Usage: tzolkin-conditional-human-policy-train INPUT.json NEW_OUTPUT_DIR [RESUME_CHECKPOINT.json]".into());
    }
    let output = Path::new(&args[1]);
    validate_new_output_directory(output)?;
    let resume = args
        .get(2)
        .map(|path| ConditionalHumanPolicyCheckpoint::load(Path::new(path)))
        .transpose()?;
    let outcome = train_collection_manifest(Path::new(&args[0]), resume.as_ref())?;
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
