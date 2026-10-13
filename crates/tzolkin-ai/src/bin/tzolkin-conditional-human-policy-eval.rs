//! Read-only shared Validation for independently source-loaded conditional actors.
use std::path::Path;
use tzolkin_ai::human_policy_training::{
    PreparedConditionalHumanPolicy, compare_semantic_playing_validation_collections,
    compare_validation_collection,
};
use tzolkin_ai::policy_training::validate_new_output_directory;

fn run() -> Result<(), String> {
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    if args.first().is_some_and(|arg| arg == "--semantic-playing4") {
        if args.len() != 8 {
            return Err("Usage: tzolkin-conditional-human-policy-eval --semantic-playing4 PUBLIC2_CHECKPOINT.json PUBLIC2_ORIGIN.json SEMANTIC4_CHECKPOINT.json SEMANTIC4_ORIGIN.json PUBLIC2_VALIDATION.json SEMANTIC4_VALIDATION.json NEW_OUTPUT_DIR".into());
        }
        let output = Path::new(&args[7]);
        validate_new_output_directory(output)?;
        let public =
            PreparedConditionalHumanPolicy::load(Path::new(&args[1]), Path::new(&args[2]))?;
        let semantic = PreparedConditionalHumanPolicy::load_semantic_playing(
            Path::new(&args[3]),
            Path::new(&args[4]),
        )?;
        let report = compare_semantic_playing_validation_collections(
            &public,
            &semantic,
            Path::new(&args[5]),
            Path::new(&args[6]),
        )?;
        report.save_new_directory(output)?;
        println!(
            "Shared source Validation: {} samples, {} candidates; distinct public2/semantic4 features, no strength measurement",
            report.samples(),
            report.candidates()
        );
        return Ok(());
    }
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
