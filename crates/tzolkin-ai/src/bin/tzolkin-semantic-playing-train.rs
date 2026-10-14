//! Finite source-owned semantic Playing learning; an outer observer owns OS closure.
use std::path::Path;
use tzolkin_ai::public_rl_paired_session::read_bounded;
use tzolkin_ai::semantic_playing_rl_journal::{
    SemanticJournalAuditSpec, SemanticJournalEvaluationSpec, SemanticJournalRunSpec,
    audit_semantic_playing_journal, evaluate_semantic_playing_journal,
    run_semantic_playing_journal,
};

const HELP: &str = "tzolkin-semantic-playing-train start|resume|continue --control FILE --expected-sha256 LOWERCASE64HEX --output NEW_INVOCATION_DIR\nOne finite predeclared whole-four schedule. Resume optionally audits the current chunk from its checked parent checkpoint. Continue starts a fresh finite journal from a completed owned actor+critic checkpoint and observed producing close, retaining the exact Original and one Prepared load. Each chunk explicitly declares its sampling seed/replicate/ordinal, budget and cumulative prior; forbidden historical seeds remain forbidden. checkpointOnCompletion requires five-batch local capture blocks and emits training-checkpoint.json only on completed work. The separate checkpoint-close-v1 declaration pins that payload, the successful journal close declaration and invocation.json after actual process exit/EOF; it is local evidence, not authentication. Every entered/unentered allocation remains consumed, including failed chunks; no refund or strength/adoption claim. Captured opponents stay within their chunk, with Corn-first Setup. Exit0: completed; exit2: NoChange; exit1: refusal. Use one writer and a bounded process/disk observer.\n";
const AUDIT_HELP: &str = "tzolkin-semantic-playing-train audit --control FILE --expected-sha256 LOWERCASE64HEX --output NEW_AUDIT_DIR\nSeparate restore-only audit schema: exact original Human4/critic/external/history/config, procedural pool, journal tip and successful producing-close declaration. Freshly replays the saved procedural/mixed records, without allocating new families/streams/updates, changing the journal or returning an owner. Writes bounded audit.json; exit0 means audit returned successfully, not Ready/authentication/strength. Historical NoChange remains stopped. The outer host owns wall/CPU/RSS and actual exit/EOF/pin checks; saved JSON alone is insufficient.\n";
const EVALUATION_HELP: &str = "tzolkin-semantic-playing-train evaluate --control FILE --expected-sha256 LOWERCASE64HEX --output NEW_EVALUATION_DIR\nSeparate closed evaluation schema pins an unchanged audit control and current global family exclusions. Restores the genuine current actor, then evaluates ordinary disjoint Validation families with all-seat Corn-first Setup, Scalar Playing argmax/default TradeGuard, originalHuman4 or resourceCycleV5 reference and fixed V5 pool. originalHuman4 borrows the exact original preparation used to restore this journal. No new learning allocation, update, capture, weight conversion or journal write. Exit0 requires complete paired arms and all source/archive postchecks; no strength/adoption claim.\n";
const CHECKPOINT_HELP: &str = "To bootstrap an exhausted successfully updated journal, audit optionally accepts checkpointExport:{prior,permanentAllocation}. It reconstructs the genuine closed actor+critic and writes training-checkpoint.json plus checkpoint-export.json in the fresh audit output, without new games, allocations, updates or journal writes. Omit checkpointExport for unchanged audit behavior. A checkpoint-close-v1 input must separately pin producerClose (checkpoint-observed-process-close-v1, actual exit0/both EOF/checkpoint/invocation pins) and the old journalCloseDeclaration; the old journal close alone cannot attest the new exporter. Both are local evidence, not source authentication.\n";

fn run(args: &[String]) -> Result<i32, String> {
    if args.len() != 7
        || !matches!(
            args[0].as_str(),
            "start" | "resume" | "continue" | "audit" | "evaluate"
        )
    {
        return Err(
            "Expected start|resume|continue|audit|evaluate and three flag/value pairs".into(),
        );
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
        print!("{HELP}{AUDIT_HELP}{EVALUATION_HELP}{CHECKPOINT_HELP}");
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
