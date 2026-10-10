//! Explicit local Long-v2 training, bootstrapped from a resolved Count1 session.
use std::collections::BTreeMap;
use std::path::Path;
use std::time::Instant;

use sha2::{Digest, Sha256};
use tzolkin_ai::kernel::Kernel;
use tzolkin_ai::public_native::PreparedPublicPolicy;
use tzolkin_ai::public_rl_long_session::{
    LongRunSpec, LongTrainingSession, MAX_CHECKPOINT_BYTES, MAX_SPEC_BYTES, read_bounded,
};

const HELP: &str = "tzolkin-rl-long-train train --spec FILE --checkpoint BC_FILE --dataset DIRECTORY --output NEW_DIRECTORY --count1 SESSION_FILE --expected-sha256 LOWERCASE64HEX\n\
tzolkin-rl-long-train resume --spec FILE --checkpoint BC_FILE --dataset DIRECTORY --output DIRECTORY --state HEAD_FILE --expected-sha256 LOWERCASE64HEX [--abandon REASON]\n\
Specification schema: tzolkin-public-rl-long-run-spec-v2. Required camelCase fields: schema, samplingSeed, replicateOrdinal, environmentSeedStart, maxSeedCandidates, decisionBudget, evaluationInterval, maxUpdates, maxFamilies, maxBatches, maxJournalEntries, learningRate, maxAbsActualDelta, maxGradientRows, maxCallbacks, maxCandidateRows, maxSourceBytes, maxTraceBytes, maxMetadataBytes, maxRecordBytes, excludedFamilies. All capacities and sampling/step settings are fixed for the run.\n\
Train starts a new Long segment from an externally pinned resolved Count1-v1 RL session checkpoint, with the same qualified BC and sampling/step settings. A controller state wrapper, Count0, repeated-v1 checkpoint or inference model is not a bootstrap. Resume uses an externally pinned Long head and identical specification/BC inputs. Prepared BC is qualified once per invocation and borrowed throughout.\n\
The serial Train-only whole-four loop stops at the decision budget, explicit capacity, NoChange or a failed batch. Budget checks occur at batch boundaries. Exit0 means the requested decision budget was reached, not that parameters changed or playing strength improved. Exit2 is another explicit stop, including abandonment; exit1 is a command/storage error, failed batch or unknown gradient. Read the report's applied-update count and stop reason.\n\
Pending resume uses saved pinned records only; missing or unknown work is not recollected. Explicit abandon consumes its reservations and stops. Resuming an already resolved failed/NoChange head can begin a new whole batch within the fixed budget. Evaluation thresholds are notices only. This command provides no Arena evaluation, PPO, mixed roster or deployment model.\n\
Use a caller-owned single-writer directory and an external wall watchdog if required. Killing a synchronous call can leave unknown work and retained temporary files. Content consistency does not authenticate the producer or optimization history.\n";

fn arguments(args: &[String]) -> Result<(&str, BTreeMap<&str, &str>), String> {
    let command = args.first().ok_or("Missing command")?.as_str();
    if !matches!(command, "train" | "resume") || !(args.len() - 1).is_multiple_of(2) {
        return Err("Expected train/resume and flag-value pairs".into());
    }
    let mut flags = BTreeMap::new();
    for pair in args[1..].as_chunks::<2>().0 {
        let name = pair[0].as_str();
        if !matches!(
            name,
            "--spec"
                | "--checkpoint"
                | "--dataset"
                | "--output"
                | "--count1"
                | "--state"
                | "--expected-sha256"
                | "--abandon"
        ) || pair[1].is_empty()
            || flags.insert(name, pair[1].as_str()).is_some()
        {
            return Err("Unknown, repeated or empty flag".into());
        }
    }
    for name in [
        "--spec",
        "--checkpoint",
        "--dataset",
        "--output",
        "--expected-sha256",
    ] {
        if !flags.contains_key(name) {
            return Err(format!("Missing {name}"));
        }
    }
    if command == "train" {
        if !flags.contains_key("--count1")
            || flags.contains_key("--state")
            || flags.contains_key("--abandon")
        {
            return Err("Train requires count1 and forbids resume flags".into());
        }
    } else if !flags.contains_key("--state") || flags.contains_key("--count1") {
        return Err("Resume requires state and forbids count1".into());
    }
    let digest = flags["--expected-sha256"];
    if digest.len() != 64
        || !digest
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
    {
        return Err("Expected canonical lowercase raw SHA256".into());
    }
    Ok((command, flags))
}

fn run(args: &[String]) -> Result<i32, String> {
    let invocation = Instant::now();
    let (command, flags) = arguments(args)?;
    let spec = LongRunSpec::from_json(&read_bounded(Path::new(flags["--spec"]), MAX_SPEC_BYTES)?)?;
    let (source, maximum) = if command == "train" {
        (
            flags["--count1"],
            tzolkin_ai::public_rl_session::MAX_CHECKPOINT_BYTES,
        )
    } else {
        (flags["--state"], MAX_CHECKPOINT_BYTES)
    };
    let bytes = read_bounded(Path::new(source), maximum)?;
    if format!("{:x}", Sha256::digest(&bytes)) != flags["--expected-sha256"] {
        return Err("Pinned checkpoint raw SHA256 mismatch".into());
    }
    let began = Instant::now();
    let prepared = PreparedPublicPolicy::load(
        Path::new(flags["--checkpoint"]),
        Path::new(flags["--dataset"]),
        Kernel::Scalar,
    )?;
    let qualification_seconds = began.elapsed().as_secs_f64();
    let output = Path::new(flags["--output"]);
    let mut session = if command == "train" {
        LongTrainingSession::from_count1(
            spec,
            &prepared,
            &bytes,
            flags["--expected-sha256"],
            output,
        )?
    } else {
        LongTrainingSession::restore(spec, &prepared, &bytes, flags["--expected-sha256"], output)?
    };
    let report = session.run(flags.get("--abandon").copied())?;
    let status = if matches!(report.stop_reason(), "batchFailed" | "gradientUnknown") {
        1
    } else if report.decision_budget_reached() {
        0
    } else {
        2
    };
    println!("{}", serde_json::to_string(&serde_json::json!({
        "schema": "tzolkin-public-rl-long-cli-report-v2", "command": command,
        "qualification": { "calls": 1, "seconds": qualification_seconds, "instrumentedForwardCalls": null },
        "invocationWallSeconds": invocation.elapsed().as_secs_f64(), "controller": report,
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
                serde_json::json!({"schema": "tzolkin-public-rl-long-cli-error-v2", "error": error})
            );
            std::process::exit(1);
        }
    }
}
