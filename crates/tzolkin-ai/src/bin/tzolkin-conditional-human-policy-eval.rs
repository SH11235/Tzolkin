//! Read-only shared Validation for independently source-loaded conditional actors.
use std::path::Path;
use tzolkin_ai::human_policy_training::{
    PreparedConditionalHumanPolicy, compare_validation_collection,
};
use tzolkin_ai::policy_training::validate_new_output_directory;

fn run() -> Result<(), String> {
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    if args.len() != 6 {
        return Err("Usage: tzolkin-conditional-human-policy-eval LEFT_CHECKPOINT.json LEFT_ORIGIN.json RIGHT_CHECKPOINT.json RIGHT_ORIGIN.json VALIDATION_COLLECTION.json NEW_OUTPUT_DIR".into());
    }
    let output = Path::new(&args[5]);
    validate_new_output_directory(output)?;
    let left = PreparedConditionalHumanPolicy::load(Path::new(&args[0]), Path::new(&args[1]))?;
    let right = PreparedConditionalHumanPolicy::load(Path::new(&args[2]), Path::new(&args[3]))?;
    let report = compare_validation_collection([&left, &right], Path::new(&args[4]))?;
    report.save_new_directory(output)?;
    println!(
        "Shared Validation: {} samples, {} candidates; policy-only, no strength measurement",
        report.samples(),
        report.candidates()
    );
    Ok(())
}

fn main() {
    if let Err(error) = run() {
        eprintln!("{error}");
        std::process::exit(1);
    }
}
