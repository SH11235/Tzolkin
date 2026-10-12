//! Dedicated conditional Human policy-only BC; no native dataset or owner admission.
use std::path::Path;
use tzolkin_ai::human_policy_training::{
    ConditionalHumanPolicyCheckpoint, train_collection_manifest,
};
use tzolkin_ai::policy_training::validate_new_output_directory;

fn run() -> Result<(), String> {
    let args = std::env::args().skip(1).collect::<Vec<_>>();
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
