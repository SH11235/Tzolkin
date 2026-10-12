//! Explicit portable Own Setup training; no actual data paths or model bootstrap.
use std::collections::BTreeMap;
use std::path::Path;

use tzolkin_ai::own_setup_run::{self, RunInputs};

const HELP: &str = "tzolkin-own-setup-train train --spec FILE --playing FILE --exclusions FILE --output NEW_DIRECTORY --artifact NEW_LEAF --report NEW_LEAF --tip NEW_LEAF [--deployment NEW_LEAF]\n\
tzolkin-own-setup-train resume --spec FILE --playing FILE --exclusions FILE --output DIRECTORY --state FILE --expected-sha256 LOWERCASE64HEX --artifact NEW_LEAF --report NEW_LEAF --tip NEW_LEAF [--deployment NEW_LEAF]\n\
tzolkin-own-setup-train inspect-tip --output DIRECTORY --state FILE --expected-sha256 LOWERCASE64HEX --tip NEW_LEAF\n\
tzolkin-own-setup-train digest-configs --spec FILE\n\
Specification schema: tzolkin-own-setup-run-spec-v1. Required camelCase fields: schema, initialSeedHex, playing {bytes, sha256}, exclusions {bytes, sha256}, exclusionCount, exclusionDigest, adam {learningRate, beta1, beta2, epsilon, maxGradientNorm, maxParameterDelta}, run {samplingSeedHex, replicateOrdinalHex, firstEpisodeOrdinalHex, maxAttemptedEpisodes}, archiveLimits {maxAttempts, maxDiskBytes}, collection {maxCallbacks, maxCandidateRows, maxSourceBytes, maxTraceBytes}, updateBudget, maxSeconds, batches [[{players, environmentSeed, learnerSeat}]], orderedConfigsDigest. Hex u64 words use exactly 16 lowercase digits. digest-configs computes the expanded ordered-config digest without model work, reservation or admission; train/resume validate the complete plan.\n\
Exclusions schema: tzolkin-own-setup-run-exclusions-v1, with count, digest and the strictly sorted complete historical families array. The plan pins the entire exclusion declaration and exact Playing bytes. The caller must reconcile all teacher, opponent, held-out and previously reserved families, including unentered suffixes. Pins check consistency, not completeness or producer authenticity.\n\
The complete immutable plan and reservation.json are published before any model load. All declared fresh Train families are spent at reservation publication. Each family occurs once in the original schedule. The training phase loads fixed currentMixed Playing once per invocation and borrows it throughout. A manifest-bound batch intent precedes each source call; only the actual validated Closed Updated ledger entry resolves it. An unfinished intent, NoChange, rejection or abandoned Pending stops this original run permanently, even if terminal-stop.json was never published. Source-owned Pending precedes collection; all callbacks are freshly audited before Closed and one actual Adam update.\n\
At most 64 batches and 256 original games are allowed; each batch contains 1..4 base 3/4p games. Existing archive limits remain in force. updateBudget and maxSeconds stop at batch boundaries. Exit0 means the update budget was reached, not improved strength or sufficient RL; exit2 means another finite stop, and exit1 a rejected batch or command/storage error. An archive holds the complete optimizer/checkpoint history; a saved trained artifact is an export and cannot bootstrap learning.\n\
Output names are fresh distinct lowercase leaf names in the caller-owned output directory. Optional --deployment publishes the genuine exporter's exact composite bytes while the trained owner lives, only when an actual trained artifact is available. A successful export makes three additional Playing validation loads at the end; reports separate trainingPlayingLoads and exportValidationLoads and pin the actual bytes. Omission keeps standalone artifact export. planned-run.json, reservation.json, initial-state.json, at most 64 intent leaves and later state/report/artifact/deployment files are outside the closed archive child. Retain an external SHA256 pin of the exact state receipt. Resume checks the original complete plan, Playing, full exclusions, actual source-root initialization/run/Adam/limits and all ordered source configs before model load, then uses the source's full cold audit/restore. inspect-tip loads no model and returns only a candidate latest pin after uncertain publication; source restore remains mandatory.\n\
Use one writer and an external wall watchdog. maxSeconds includes validation and cold restore but cannot interrupt a synchronous native call. Killing a call can leave spent reservations and unknown work; there is no automatic retry or recollection. This command supplies no opponent mixture, Arena evaluation or adoption claim.\n";

fn arguments(args: &[String]) -> Result<(&str, BTreeMap<&str, &str>), String> {
    let command = args.first().ok_or("Missing command")?.as_str();
    if !matches!(
        command,
        "train" | "resume" | "inspect-tip" | "digest-configs"
    ) || !(args.len() - 1).is_multiple_of(2)
    {
        return Err("Expected train/resume/inspect-tip/digest-configs and flag-value pairs".into());
    }
    let mut flags = BTreeMap::new();
    for pair in args[1..].as_chunks::<2>().0 {
        let name = pair[0].as_str();
        if !matches!(
            name,
            "--spec"
                | "--playing"
                | "--exclusions"
                | "--output"
                | "--artifact"
                | "--report"
                | "--tip"
                | "--deployment"
                | "--state"
                | "--expected-sha256"
        ) || pair[1].is_empty()
            || flags.insert(name, pair[1].as_str()).is_some()
        {
            return Err("Unknown, repeated or empty flag".into());
        }
    }
    let required: &[&str] = match command {
        "train" => &[
            "--spec",
            "--playing",
            "--exclusions",
            "--output",
            "--artifact",
            "--report",
            "--tip",
        ],
        "resume" => &[
            "--spec",
            "--playing",
            "--exclusions",
            "--output",
            "--artifact",
            "--report",
            "--tip",
            "--state",
            "--expected-sha256",
        ],
        "digest-configs" => &["--spec"],
        _ => &["--output", "--state", "--expected-sha256", "--tip"],
    };
    let optional =
        usize::from(matches!(command, "train" | "resume") && flags.contains_key("--deployment"));
    if flags.len() != required.len() + optional
        || required.iter().any(|name| !flags.contains_key(name))
    {
        return Err("Missing required or command-inappropriate flag".into());
    }
    Ok((command, flags))
}

fn run(args: &[String]) -> Result<i32, String> {
    let (command, flags) = arguments(args)?;
    if command == "digest-configs" {
        let bytes = tzolkin_ai::public_rl_paired_session::read_bounded(
            Path::new(flags["--spec"]),
            own_setup_run::MAX_SPEC_BYTES,
        )?;
        let spec = own_setup_run::OwnSetupRunSpec::from_json(&bytes)?;
        println!(
            "{}",
            serde_json::json!({"orderedConfigsDigest": spec.ordered_digest()?})
        );
        return Ok(0);
    }
    if command == "inspect-tip" {
        let pin = own_setup_run::inspect_tip(
            Path::new(flags["--output"]),
            Path::new(flags["--state"]),
            flags["--expected-sha256"],
            flags["--tip"],
        )?;
        println!(
            "{}",
            serde_json::to_string(&pin).map_err(|e| e.to_string())?
        );
        return Ok(0);
    }
    let inputs = RunInputs {
        spec: Path::new(flags["--spec"]),
        playing: Path::new(flags["--playing"]),
        exclusions: Path::new(flags["--exclusions"]),
        output: Path::new(flags["--output"]),
        artifact: flags["--artifact"],
        report: flags["--report"],
        tip: flags["--tip"],
        deployment: flags.get("--deployment").copied(),
    };
    let report = if command == "train" {
        own_setup_run::train(inputs)?
    } else {
        own_setup_run::resume(
            inputs,
            Path::new(flags["--state"]),
            flags["--expected-sha256"],
        )?
    };
    println!(
        "{}",
        serde_json::to_string(&report).map_err(|e| e.to_string())?
    );
    Ok(report.exit_code)
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
                serde_json::json!({"schema": "tzolkin-own-setup-run-error-v1", "error": error})
            );
            std::process::exit(1);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parser_rejects_duplicate_unknown_and_command_inappropriate_flags() {
        let args = |words: &[&str]| words.iter().map(|s| s.to_string()).collect::<Vec<_>>();
        assert!(arguments(&args(&["digest-configs", "--spec", "plan.json"])).is_ok());
        assert!(arguments(&args(&["digest-configs", "--spec", "a", "--spec", "b"])).is_err());
        assert!(arguments(&args(&["digest-configs", "--spec", "a", "--unknown", "b"])).is_err());
        assert!(arguments(&args(&["digest-configs", "--spec", "a", "--playing", "b"])).is_err());
        assert!(arguments(&args(&["train", "--spec", "a"])).is_err());
        assert!(
            arguments(&args(&[
                "digest-configs",
                "--spec",
                "a",
                "--deployment",
                "b"
            ]))
            .is_err()
        );
        assert!(
            arguments(&args(&[
                "train",
                "--spec",
                "a",
                "--playing",
                "b",
                "--exclusions",
                "c",
                "--output",
                "d",
                "--artifact",
                "artifact.json",
                "--report",
                "report.json",
                "--tip",
                "state.json",
                "--deployment",
                "deployment.json",
            ]))
            .is_ok()
        );
    }
}
