//! Explicit portable Own Setup training; no actual data paths or model bootstrap.
use std::collections::BTreeMap;
use std::path::Path;

use tzolkin_ai::own_setup_run::{self, OwnSetupRosterRunInputs, RunInputs};

const HELP: &str = "tzolkin-own-setup-train train --spec FILE --playing FILE --exclusions FILE --output NEW_DIRECTORY --artifact NEW_LEAF --report NEW_LEAF --tip NEW_LEAF [--deployment NEW_LEAF]\n\
tzolkin-own-setup-train resume --spec FILE --playing FILE --exclusions FILE --output DIRECTORY --state FILE --expected-sha256 LOWERCASE64HEX --artifact NEW_LEAF --report NEW_LEAF --tip NEW_LEAF [--deployment NEW_LEAF]\n\
tzolkin-own-setup-train inspect-tip --output DIRECTORY --state FILE --expected-sha256 LOWERCASE64HEX --tip NEW_LEAF\n\
tzolkin-own-setup-train digest-configs --spec FILE\n\
tzolkin-own-setup-train continue --spec FILE --lineage FILE --playing FILE --exclusions FILE --output NEW_DIRECTORY --artifact NEW_LEAF --report NEW_LEAF --tip NEW_LEAF [--deployment NEW_LEAF]\n\
tzolkin-own-setup-train resume-continuation --spec FILE --lineage FILE --playing FILE --exclusions FILE --output DIRECTORY --state FILE --expected-sha256 LOWERCASE64HEX --artifact NEW_LEAF --report NEW_LEAF --tip NEW_LEAF [--deployment NEW_LEAF]\n\
tzolkin-own-setup-train inspect-continuation-tip --output DIRECTORY --state FILE --expected-sha256 LOWERCASE64HEX --tip NEW_LEAF\n\
tzolkin-own-setup-train digest-continuation-configs --spec FILE\n\
tzolkin-own-setup-train continue-roster --spec FILE --lineage FILE --playing FILE --exclusions FILE --output NEW_DIRECTORY --artifact NEW_LEAF --report NEW_LEAF --tip NEW_LEAF [--max-new-batches POSITIVE] [--deployment NEW_LEAF]\n\
tzolkin-own-setup-train resume-roster --spec FILE --lineage FILE --playing FILE --exclusions FILE --output DIRECTORY --state FILE --expected-sha256 LOWERCASE64HEX --artifact NEW_LEAF --report NEW_LEAF --tip NEW_LEAF [--max-new-batches POSITIVE] [--deployment NEW_LEAF]\n\
tzolkin-own-setup-train inspect-roster-tip --output DIRECTORY --state FILE --expected-sha256 LOWERCASE64HEX --tip NEW_LEAF\n\
tzolkin-own-setup-train digest-roster-configs --spec FILE\n\
Specification schema: tzolkin-own-setup-run-spec-v1. Required camelCase fields: schema, initialSeedHex, playing {bytes, sha256}, exclusions {bytes, sha256}, exclusionCount, exclusionDigest, adam {learningRate, beta1, beta2, epsilon, maxGradientNorm, maxParameterDelta}, run {samplingSeedHex, replicateOrdinalHex, firstEpisodeOrdinalHex, maxAttemptedEpisodes}, archiveLimits {maxAttempts, maxDiskBytes}, collection {maxCallbacks, maxCandidateRows, maxSourceBytes, maxTraceBytes}, updateBudget, maxSeconds, batches [[{players, environmentSeed, learnerSeat}]], orderedConfigsDigest. Hex u64 words use exactly 16 lowercase digits. digest-configs computes the expanded ordered-config digest without model work, reservation or admission; train/resume validate the complete plan.\n\
Exclusions schema: tzolkin-own-setup-run-exclusions-v1, with count, digest and the strictly sorted complete historical families array. The plan pins the entire exclusion declaration and exact Playing bytes. The caller must reconcile all teacher, opponent, held-out and previously reserved families, including unentered suffixes. Pins check consistency, not completeness or producer authenticity.\n\
The complete immutable plan and reservation.json are published before any model load. All declared fresh Train families are spent at reservation publication. Each family occurs once in the original schedule. The training phase loads fixed currentMixed Playing once per invocation and borrows it throughout. A manifest-bound batch intent precedes each source call; only the actual validated Closed Updated ledger entry resolves it. An unfinished intent, NoChange, rejection or abandoned Pending stops this original run permanently, even if terminal-stop.json was never published. Source-owned Pending precedes collection; all callbacks are freshly audited before Closed and one actual Adam update.\n\
At most 64 batches and 256 original games are allowed; each batch contains 1..4 base 3/4p games. Existing archive limits remain in force. updateBudget and maxSeconds stop at batch boundaries. Exit0 means the update budget was reached, not improved strength or sufficient RL; exit2 means another finite stop, and exit1 a rejected batch or command/storage error. An archive holds the complete optimizer/checkpoint history; a saved trained artifact is an export and cannot bootstrap learning.\n\
Output names are fresh distinct lowercase leaf names in the caller-owned output directory. Optional --deployment publishes the genuine exporter's exact composite bytes while the trained owner lives, only when an actual trained artifact is available. A successful export makes three additional Playing validation loads at the end; reports separate trainingPlayingLoads and exportValidationLoads and pin the actual bytes. Omission keeps standalone artifact export. planned-run.json, reservation.json, initial-state.json, at most 64 intent leaves and later state/report/artifact/deployment files are outside the closed archive child. Retain an external SHA256 pin of the exact state receipt. Resume checks the original complete plan, Playing, full exclusions, actual source-root initialization/run/Adam/limits and all ordered source configs before model load, then uses the source's full cold audit/restore. inspect-tip loads no model and returns only a candidate latest pin after uncertain publication; source restore remains mandatory.\n\
Continuation uses schema tzolkin-own-setup-continue-spec-v1 with schema, lineage {bytes, sha256} and segment (the unchanged run-spec-v1 shape). Its segment budget and updateBudget are LOCAL new work. segment.run.firstEpisodeOrdinalHex must equal the original origin plus all previously attempted originals; sampling seed, replicate, initialization and all six Adam words remain unchanged. The new full union must include every original baseline and every permanently reserved ancestral family. Every named ancestor must be latest, complete and successful, including its original pre-call intents. At most15 ancestors plus one new segment,4096 cumulative originals and8GiB aggregate source archives are admitted.\n\
Lineage schema tzolkin-own-setup-run-lineage-v1 contains sources, in oldest-to-newest order. The first entry has kind train and output, spec, exclusions, state, expectedSha256; each later entry has kind continue and the same fields plus lineage (its original lineage declaration). Locations are caller-supplied local paths relative to the declaration directory or absolute paths. Every source's original spec/full exclusions/exact external state pin and every predecessor prefix remain immutable. New sources are reconstructed by the source API, carrying exact parameters, Adam, cumulative cursor/credit/ledger; numeric artifact imports never create owners. Outputs must be outside every ancestor archive. New segment reports distinguish cumulative attemptedEpisodes/updateCount from local closedAttempts/originalCollections/freshAuditSeals and include continuation offsets; coldRestore reports ancestral reaudits separately. Original train/resume and their v1 wire/digests retain their meanings.\n\
Roster continuation uses tzolkin-own-setup-roster-continue-spec-v1: schema, lineage, segment (unchanged run-spec-v1), roster {playing (the full FixedPlayingIdentity metadata), absoluteSeats (one ordered role vector per segment game), planChecksum}. Roles are learner, v5, fixedPlayingNn; exactly one learner must match the original learnerSeat. Rust derives the complete roster from the native schedule and batch grouping. digest-roster-configs computes orderedConfigsDigest and rosterPlanChecksum without loading models; runtime validates both expected fields. The complete roster and fixed identity bind collection, fresh audit, source root and exact-spec cold restore. A later lineage source uses kind rosterContinue with the same fields as continue. Each NN actor has an independent borrowed guard history over the one immutable loaded Playing; its ordinary Setup decisions are scored by that NN.\n\
Only roster execution accepts --max-new-batches in 1..=original batch count (at most64). It pauses after that many new Closed Updated batches, before the next intent, with nonterminal invocationBatchBudgetReached/exit2. The original whole reservation, updateBudget and plan remain immutable. Exact-spec resume may continue a successful prefix; unfinished intent, NoChange, rejection or source error remain terminal. Omission runs the original finite schedule.\n\
Use one writer and an external wall watchdog. maxSeconds includes validation and cold restore but cannot interrupt a synchronous native call. Killing a call can leave spent reservations and unknown work; there is no automatic retry or recollection. These commands supply no Arena evaluation or adoption claim.\n";

fn arguments(args: &[String]) -> Result<(&str, BTreeMap<&str, &str>), String> {
    let command = args.first().ok_or("Missing command")?.as_str();
    if !matches!(
        command,
        "train"
            | "resume"
            | "inspect-tip"
            | "digest-configs"
            | "continue"
            | "resume-continuation"
            | "inspect-continuation-tip"
            | "digest-continuation-configs"
            | "continue-roster"
            | "resume-roster"
            | "inspect-roster-tip"
            | "digest-roster-configs"
    ) || !(args.len() - 1).is_multiple_of(2)
    {
        return Err("Expected an explicit Own Setup command and flag-value pairs".into());
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
                | "--lineage"
                | "--max-new-batches"
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
        "continue" | "continue-roster" => &[
            "--spec",
            "--lineage",
            "--playing",
            "--exclusions",
            "--output",
            "--artifact",
            "--report",
            "--tip",
        ],
        "resume-continuation" | "resume-roster" => &[
            "--spec",
            "--lineage",
            "--playing",
            "--exclusions",
            "--output",
            "--artifact",
            "--report",
            "--tip",
            "--state",
            "--expected-sha256",
        ],
        "digest-configs" | "digest-continuation-configs" | "digest-roster-configs" => &["--spec"],
        _ => &["--output", "--state", "--expected-sha256", "--tip"],
    };
    let optional = usize::from(
        matches!(
            command,
            "train"
                | "resume"
                | "continue"
                | "resume-continuation"
                | "continue-roster"
                | "resume-roster"
        ) && flags.contains_key("--deployment"),
    ) + usize::from(
        matches!(command, "continue-roster" | "resume-roster")
            && flags.contains_key("--max-new-batches"),
    );
    if flags.len() != required.len() + optional
        || required.iter().any(|name| !flags.contains_key(name))
    {
        return Err("Missing required or command-inappropriate flag".into());
    }
    if let Some(value) = flags.get("--max-new-batches") {
        batch_allowance(value)?;
    }
    Ok((command, flags))
}

fn batch_allowance(value: &str) -> Result<usize, String> {
    let count: usize = value
        .parse()
        .map_err(|_| "Expected a positive batch allowance")?;
    if !(1..=64).contains(&count) || count.to_string() != value {
        return Err("Batch allowance must be canonical decimal 1..=64".into());
    }
    Ok(count)
}

fn run(args: &[String]) -> Result<i32, String> {
    let (command, flags) = arguments(args)?;
    if matches!(
        command,
        "digest-configs" | "digest-continuation-configs" | "digest-roster-configs"
    ) {
        let bytes = tzolkin_ai::public_rl_paired_session::read_bounded(
            Path::new(flags["--spec"]),
            own_setup_run::MAX_SPEC_BYTES,
        )?;
        if command == "digest-roster-configs" {
            let digests =
                own_setup_run::OwnSetupRosterContinueSpec::from_json(&bytes)?.ordered_digests()?;
            println!(
                "{}",
                serde_json::to_string(&digests).map_err(|e| e.to_string())?
            );
            return Ok(0);
        }
        let digest = if command == "digest-configs" {
            own_setup_run::OwnSetupRunSpec::from_json(&bytes)?.ordered_digest()?
        } else {
            own_setup_run::OwnSetupContinueSpec::from_json(&bytes)?
                .segment
                .ordered_digest()?
        };
        println!("{}", serde_json::json!({"orderedConfigsDigest": digest}));
        return Ok(0);
    }
    if matches!(
        command,
        "inspect-tip" | "inspect-continuation-tip" | "inspect-roster-tip"
    ) {
        let inspect = match command {
            "inspect-tip" => own_setup_run::inspect_tip,
            "inspect-roster-tip" => own_setup_run::inspect_rostered_continuation_tip,
            _ => own_setup_run::inspect_continuation_tip,
        };
        let pin = inspect(
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
    let report = match command {
        "train" => own_setup_run::train(inputs)?,
        "continue" => own_setup_run::continue_run(inputs, Path::new(flags["--lineage"]))?,
        "resume" => own_setup_run::resume(
            inputs,
            Path::new(flags["--state"]),
            flags["--expected-sha256"],
        )?,
        "resume-continuation" => own_setup_run::resume_continuation(
            inputs,
            Path::new(flags["--lineage"]),
            Path::new(flags["--state"]),
            flags["--expected-sha256"],
        )?,
        "continue-roster" => own_setup_run::continue_rostered_run(
            OwnSetupRosterRunInputs {
                run: inputs,
                max_new_batches: flags
                    .get("--max-new-batches")
                    .map(|v| batch_allowance(v))
                    .transpose()?,
            },
            Path::new(flags["--lineage"]),
        )?,
        "resume-roster" => own_setup_run::resume_rostered_continuation(
            OwnSetupRosterRunInputs {
                run: inputs,
                max_new_batches: flags
                    .get("--max-new-batches")
                    .map(|v| batch_allowance(v))
                    .transpose()?,
            },
            Path::new(flags["--lineage"]),
            Path::new(flags["--state"]),
            flags["--expected-sha256"],
        )?,
        _ => return Err("Invalid executable command".into()),
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
        assert!(arguments(&args(&["digest-continuation-configs", "--spec", "a"])).is_ok());
        assert!(arguments(&args(&["continue", "--spec", "a"])).is_err());
        let continued = args(&[
            "continue",
            "--spec",
            "a",
            "--lineage",
            "b",
            "--playing",
            "c",
            "--exclusions",
            "d",
            "--output",
            "e",
            "--artifact",
            "artifact.json",
            "--report",
            "report.json",
            "--tip",
            "state.json",
        ]);
        assert!(arguments(&continued).is_ok());
        let mut roster = continued.clone();
        roster[0] = "continue-roster".into();
        roster.extend(args(&[
            "--max-new-batches",
            "1",
            "--deployment",
            "deployment.json",
        ]));
        assert!(arguments(&roster).is_ok());
        roster[0] = "continue".into();
        assert!(arguments(&roster).is_err());
        roster[0] = "resume-roster".into();
        assert!(arguments(&roster).is_err());
        roster.extend(args(&["--state", "state.json", "--expected-sha256", "pin"]));
        assert!(arguments(&roster).is_ok());
        for bad in ["0", "01", "+1", "65", "-1", "1.0"] {
            assert!(batch_allowance(bad).is_err());
        }
        assert!(arguments(&args(&["digest-roster-configs", "--spec", "a"])).is_ok());
        assert!(
            arguments(&args(&[
                "digest-roster-configs",
                "--spec",
                "a",
                "--max-new-batches",
                "1"
            ]))
            .is_err()
        );
        let mut missing_lineage = continued;
        missing_lineage.remove(3);
        missing_lineage.remove(3);
        assert!(arguments(&missing_lineage).is_err());
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
