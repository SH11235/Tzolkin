//! Explicit local controller for the existing bounded RL session contract.
use std::collections::BTreeMap;
use std::path::Path;
use std::time::Instant;

use tzolkin_ai::kernel::Kernel;
use tzolkin_ai::public_native::PreparedPublicPolicy;
use tzolkin_ai::public_rl_controller::{self, MAX_SPEC_BYTES, MAX_STATE_BYTES, RunSpec};

const HELP: &str = "tzolkin-rl-train train --spec FILE --checkpoint FILE --dataset DIRECTORY --output NEW_DIRECTORY\n\
tzolkin-rl-train resume --spec FILE --checkpoint FILE --dataset DIRECTORY --output DIRECTORY --state FILE --expected-sha256 LOWERCASE64HEX [--abandon REASON]\n\
Run specification schema: tzolkin-public-rl-run-spec-v1. Required camelCase fields: schema, samplingSeed (u64), replicateOrdinal (u64), firstEpisodeOrdinal (u64), environmentSeedStart (u32), maxSeedCandidates, decisionBudget, evaluationInterval, maxBatches (1..10), learningRate, maxAbsActualDelta, maxGradientRows, maxCallbacks, maxCandidateRows, maxSourceBytes, maxTraceBytes, excludedFamilies (sorted distinct family digests).\n\
Saved BC checkpoint/dataset are qualified once per invocation, Scalar only. Training uses the existing four-game Train-only all-seat plain-ascent contracts. This invocation stops at decision budget, count10, batch limit, NoChange or failure; seedRangeExhausted and unknownPendingWork are also explicit error stops. Budget checks occur at whole-batch boundaries and count all known collected decisions, including failed and abandoned batches. Evaluation intervals record due notices only; no evaluation, PPO, mixed roster or count11 is provided.\n\
Resume requires the external raw SHA of a controller checkpoint and identical specification/BC inputs. Pending batches use only saved pinned records; missing/unknown work is never recollected automatically. Explicit abandon consumes reserved families and ordinals and stops this invocation. Resuming a resolved NoChange, failed or abandoned checkpoint starts a new whole batch unless a budget or limit already stops it. Synchronous calls require an external wall watchdog if desired. A killed process can leave Unknown work and retained temporary files.\n\
State/record files are bounded and newly published in a caller-owned directory. The directory must have a single writer. Content consistency is not producer or learning-history authentication. Exit0 means the requested logical decision budget was reached, including any known failed or abandoned work; it does not imply an applied update. Read sessionProgress.completedUpdates in the structured report. Exit2 is another explicit stop; exit1 is a command/storage/operation error, seed exhaustion or unknown pending work.\n";

fn arguments(args: &[String]) -> Result<(&str, BTreeMap<&str, &str>), String> {
    let command = args.first().ok_or("Missing command")?.as_str();
    if !matches!(command, "train" | "resume") {
        return Err("Unknown controller command".into());
    }
    let mut flags = BTreeMap::new();
    if !(args.len() - 1).is_multiple_of(2) {
        return Err("Flags require values".into());
    }
    for pair in args[1..].as_chunks::<2>().0 {
        let name = pair[0].as_str();
        if !matches!(
            name,
            "--spec"
                | "--checkpoint"
                | "--dataset"
                | "--output"
                | "--state"
                | "--expected-sha256"
                | "--abandon"
        ) || flags.insert(name, pair[1].as_str()).is_some()
            || pair[1].is_empty()
        {
            return Err("Unknown, repeated or empty controller flag".into());
        }
    }
    for name in ["--spec", "--checkpoint", "--dataset", "--output"] {
        if !flags.contains_key(name) {
            return Err(format!("Missing {name}"));
        }
    }
    let resume_fields = ["--state", "--expected-sha256", "--abandon"];
    if command == "train" && resume_fields.iter().any(|key| flags.contains_key(key)) {
        return Err("Resume flags are forbidden for train".into());
    }
    if command == "resume"
        && ["--state", "--expected-sha256"]
            .iter()
            .any(|key| !flags.contains_key(key))
    {
        return Err("Resume requires state and its external SHA".into());
    }
    Ok((command, flags))
}
fn run(args: &[String]) -> Result<i32, String> {
    let invocation_began = Instant::now();
    let (command, flags) = arguments(args)?;
    let spec = RunSpec::from_json(&public_rl_controller::read_bounded(
        Path::new(flags["--spec"]),
        MAX_SPEC_BYTES,
    )?)?;
    let state = if command == "resume" {
        Some(public_rl_controller::read_bounded(
            Path::new(flags["--state"]),
            MAX_STATE_BYTES,
        )?)
    } else {
        None
    };
    // Argument/spec/byte failures precede qualification. Model preparation has
    // its own time and can infer; it is not counted as native collection work.
    let began = Instant::now();
    let prepared = PreparedPublicPolicy::load(
        Path::new(flags["--checkpoint"]),
        Path::new(flags["--dataset"]),
        Kernel::Scalar,
    )?;
    let qualification_seconds = began.elapsed().as_secs_f64();
    let directory = Path::new(flags["--output"]);
    let report = if let Some(state) = state {
        public_rl_controller::resume(
            spec,
            &prepared,
            &state,
            flags["--expected-sha256"],
            directory,
            flags.get("--abandon").copied(),
        )?
    } else {
        public_rl_controller::train(spec, &prepared, directory)?
    };
    let status = if report.stopped_with_error() {
        1
    } else if report.decision_budget_reached() {
        0
    } else {
        2
    };
    println!("{}", serde_json::to_string(&serde_json::json!({
        "schema": "tzolkin-public-rl-controller-cli-report-v1", "command": command,
        "qualification": { "calls": 1, "seconds": qualification_seconds, "instrumentedForwardCalls": null },
        "invocationWallSeconds": invocation_began.elapsed().as_secs_f64(),
        "controller": report
    })).map_err(|e| e.to_string())?);
    Ok(status)
}
fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.as_slice() == ["--help"] {
        println!("{HELP}");
        return;
    }
    match run(&args) {
        Ok(code) => std::process::exit(code),
        Err(error) => {
            eprintln!(
                "{}",
                serde_json::json!({"schema": "tzolkin-public-rl-controller-cli-error-v1", "error": error})
            );
            std::process::exit(1);
        }
    }
}
