//! Finite source-owned semantic Playing learning; an outer observer owns OS closure.
use std::path::Path;
use tzolkin_ai::public_rl_paired_session::read_bounded;
use tzolkin_ai::semantic_playing_rl_journal::{
    SemanticJournalRunSpec, run_semantic_playing_journal,
};

const HELP: &str = "tzolkin-semantic-playing-train start|resume --control FILE --expected-sha256 LOWERCASE64HEX --output NEW_INVOCATION_DIR\nOne finite predeclared whole-four schedule on one source-owned live session. Resume reconstructs from the exact original Human4/critic/external/history/config inputs and an externally pinned successful close declaration. New global PRIOR exclusions are checked separately, before this invocation's own permanently allocated schedule. Every entered/unentered allocation remains host-consumed; no retry or refund. Captured opponents are actual same-chain typed actors, with common Corn-first Setup. No checkpoint/parameter export, old owner conversion, Test/adoption/strength or process authentication. Each actual returned library close is saved outside the owned journal before the next batch. Exit0: all declared batches Updated; exit2: truthful NoChange stop, not Ready; exit1: refusal/failure/unknown. Use one writer and a separate bounded process/disk observer.\n";

fn run(args: &[String]) -> Result<i32, String> {
    if args.len() != 7 || !matches!(args[0].as_str(), "start" | "resume") {
        return Err("Expected start|resume and three flag/value pairs".into());
    }
    let mut flags = std::collections::BTreeMap::new();
    for pair in args[1..].as_chunks::<2>().0 {
        if !matches!(
            pair[0].as_str(),
            "--control" | "--expected-sha256" | "--output"
        ) || pair[1].is_empty()
            || flags.insert(pair[0].as_str(), pair[1].as_str()).is_some()
        {
            return Err("Unknown, empty or repeated flag".into());
        }
    }
    let raw = read_bounded(Path::new(flags["--control"]), 1024 * 1024)?;
    let spec = SemanticJournalRunSpec::from_json(&raw, flags["--expected-sha256"])?;
    if spec.command() != args[0] {
        return Err("Command/control mode differs".into());
    }
    let report = run_semantic_playing_journal(spec, Path::new(flags["--output"]))?;
    println!(
        "{}",
        serde_json::to_string(&report).map_err(|e| e.to_string())?
    );
    Ok(report.exit_code())
}
fn main() {
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    if args.as_slice() == ["--help"] {
        print!("{HELP}");
        return;
    }
    match run(&args) {
        Ok(code) => std::process::exit(code),
        Err(error) => {
            eprintln!(
                "{}",
                serde_json::json!({"schema":"tzolkin-semantic-playing-train-error-v1","error":error})
            );
            std::process::exit(1);
        }
    }
}
