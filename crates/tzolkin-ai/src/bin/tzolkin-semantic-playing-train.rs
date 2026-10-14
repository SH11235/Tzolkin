//! Finite source-owned semantic Playing learning; an outer observer owns OS closure.
use std::path::Path;
use tzolkin_ai::public_rl_paired_session::read_bounded;
use tzolkin_ai::semantic_playing_rl_journal::{
    SemanticJournalAuditSpec, SemanticJournalEvaluationSpec, SemanticJournalRunSpec,
    audit_semantic_playing_journal, evaluate_semantic_playing_journal,
    run_semantic_playing_journal,
};

const HELP: &str = "tzolkin-semantic-playing-train start|resume --control FILE --expected-sha256 LOWERCASE64HEX --output NEW_INVOCATION_DIR\nOne finite predeclared whole-four schedule on one source-owned live session. Resume reconstructs from the exact original Human4/critic/external/history/config inputs and an externally pinned successful close declaration. New global PRIOR exclusions are checked separately, before this invocation's own permanently allocated schedule. Every entered/unentered allocation remains host-consumed; no retry or refund. Captured opponents are actual same-chain typed actors, with common Corn-first Setup. No checkpoint/parameter export, old owner conversion, Test/adoption/strength or process authentication. Each actual returned library close is saved outside the owned journal before the next batch. Exit0: all declared batches Updated; exit2: truthful NoChange stop, not Ready; exit1: refusal/failure/unknown. Use one writer and a separate bounded process/disk observer.\n";
const AUDIT_HELP: &str = "tzolkin-semantic-playing-train audit --control FILE --expected-sha256 LOWERCASE64HEX --output NEW_AUDIT_DIR\nSeparate restore-only audit schema: exact original Human4/critic/external/history/config, procedural pool, journal tip and successful producing-close declaration. Freshly replays the saved procedural/mixed records, without allocating new families/streams/updates, changing the journal or returning an owner. Writes bounded audit.json; exit0 means audit returned successfully, not Ready/authentication/strength. Historical NoChange remains stopped. The outer host owns wall/CPU/RSS and actual exit/EOF/pin checks; saved JSON alone is insufficient.\n";
const EVALUATION_HELP: &str = "tzolkin-semantic-playing-train evaluate --control FILE --expected-sha256 LOWERCASE64HEX --output NEW_EVALUATION_DIR\nSeparate closed evaluation schema pins an unchanged audit control and current global family exclusions. Restores the genuine current actor, then evaluates ordinary disjoint Validation families with all-seat Corn-first Setup, Scalar Playing argmax/default TradeGuard, originalHuman4 or resourceCycleV5 reference and fixed V5 pool. originalHuman4 borrows the exact original preparation used to restore this journal. No new learning allocation, update, capture, weight conversion or journal write. Exit0 requires complete paired arms and all source/archive postchecks; no strength/adoption claim.\n";

fn run(args: &[String]) -> Result<i32, String> {
    if args.len() != 7 || !matches!(args[0].as_str(), "start" | "resume" | "audit" | "evaluate") {
        return Err("Expected start|resume|audit|evaluate and three flag/value pairs".into());
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
    if args[0] == "evaluate" {
        let spec = SemanticJournalEvaluationSpec::from_json(&raw, flags["--expected-sha256"])?;
        let result = evaluate_semantic_playing_journal(spec, Path::new(flags["--output"]));
        let post = read_bounded(Path::new(flags["--control"]), 1024 * 1024).and_then(|after| {
            if after == raw {
                Ok(())
            } else {
                Err("Evaluation control changed".into())
            }
        });
        let report = match (result, post) {
            (Ok(report), Ok(())) => report,
            (Err(error), Ok(())) => return Err(error),
            (result, Err(post)) => {
                return Err(format!(
                    "{}; evaluation control POST: {post}",
                    result.err().unwrap_or_default()
                ));
            }
        };
        let code = report.exit_code();
        println!(
            "{}",
            serde_json::json!({"schema":"tzolkin-semantic-playing-evaluation-summary-v1",
                "report":"evaluation.json","complete":code == 0})
        );
        return Ok(code);
    }
    if args[0] == "audit" {
        let spec = SemanticJournalAuditSpec::from_json(&raw, flags["--expected-sha256"])?;
        let report = audit_semantic_playing_journal(spec, Path::new(flags["--output"]))?;
        println!(
            "{}",
            serde_json::to_string(&report).map_err(|e| e.to_string())?
        );
        return Ok(0);
    }
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
        print!("{HELP}{AUDIT_HELP}{EVALUATION_HELP}");
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
