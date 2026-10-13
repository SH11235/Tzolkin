//! Portable fixed-pool campaign over an existing genuine current-mixed lineage.
use std::collections::BTreeMap;
use std::path::Path;
use tzolkin_ai::public_rl_current_mixed_durable::{
    CurrentMixedCampaignSpec, CurrentMixedLivePlanSpec, current_mixed_campaign,
    current_mixed_live_plan,
};
use tzolkin_ai::public_rl_paired_session::read_bounded;

const HELP: &str = "tzolkin-current-mixed-train live-continue|live-resume|live-inspect --spec FILE --expected-sha256 LOWERCASE64HEX --output DIRECTORY --report NEW_LEAF [--head FILE --head-sha256 LOWERCASE64HEX] [--max-new-updates 1..1000] [--deployment NEW_LEAF]\n\
Live spec: tzolkin-current-mixed-live-plan-spec-v1. Each separately pinned segment uses tzolkin-current-mixed-live-segment-plan-v1 with unchanged native local limits and the established global contract. The entire finite future Train/stream schedule is permanently reserved before one genuine qualification/cold restore. Live owner transfers only after Updated, with unchanged historical non-updated counters and full physical archive checks. Native update mathematics and old wire remain unchanged. Cooperative pauses occur after Updated and before next Start. Resume requires exact closed wrapper/native prefixes; an external host must independently observe actual process closure. No model checkpoints or caller receipts mint owner authority. Total live-plan bounds are 1000 updates/24 declared segments and the actual inherited contract can be stricter. Each plan declares an aggregate wrapper ceiling at most 2 GiB. Existing wrapper/log bytes, all remaining closed-return records, plan/reservation and any requested deployment must fit before model work; native archive and per-file limits remain separate. Report timings are separately bounded per call.\n\
tzolkin-current-mixed-train continue --spec FILE --expected-sha256 LOWERCASE64HEX --output NEW_DIRECTORY --report NEW_LEAF [--max-new-attempts 1..64] [--deployment NEW_LEAF]\n\
tzolkin-current-mixed-train resume|inspect-tip --spec FILE --expected-sha256 LOWERCASE64HEX --output DIRECTORY --head FILE --head-sha256 LOWERCASE64HEX --report NEW_LEAF [--max-new-attempts 1..64] [--deployment NEW_LEAF]\n\
Spec: tzolkin-current-mixed-campaign-spec-v1. It pins the original qualified BC checkpoint/dataset manifest, genuine Count1 or paired bootstrap, original first-mixed archive, complete durable ancestry, fixed pool, next segment contract, all 1..64 whole-four collection specs and the complete external exclusion declaration. Numerical inference JSON is not an owner.\n\
Optional seatCampaign uses tzolkin-current-mixed-ready-seat-campaign-v1, exact cumulative nonUpdatedBaseline rows [NoChange,Failed,Abandoned], each [cohorts,cohortsWithoutCounts,callbacks,candidateRows,singletons], and optional rotation {oldPool,newPool,shifts}. Shifts [3p,4p] cyclically move every complete role; latest Updated is required, without simultaneous budget amendment. Saved seatRotation ancestry pins both pools, shifts and the latest tip. Ordinary guarded chunks keep the same baseline too: inherited failures remain counted and any new non-updated work stops the campaign. Seat coverage and the entire multi-segment reservation require an explicit caller plan; this API makes no generic all-seat claim.\n\
Continue permanently publishes the entire declared chunk's Train families and sampling streams before loading any model. Resume validates the exact successful native prefix and prior invocation returns before qualification/restore. Failed, NoChange, unfinished, unknown or uncertain publication does not recollect or retry. All unentered declared families remain externally spent. A finite invocation retains one live owner across its calls; --max-new-attempts pauses only after successful calls.\n\
The existing actor/critic clipped SGD steps, source records, journal and digest domains are unchanged. Full historical exclusions are caller-declared and cannot prove external completeness. The genuine lineage still admits at most 4096 reserved families including inherited ones, 64 local attempts and 64 segments.\n\
For continue/resume, exit0 means the declared chunk completed with source-owned Updated resolutions; exit2 is a bounded successful pause. For inspect-tip, exit0 means metadata inspection succeeded; it does not evaluate successful wrapper-prefix closure or resume readiness and admits no owner. Exit1 is refusal/failure/unknown. Output leaves are 1..64 ASCII letters/digits, hyphen or period, without trailing periods, control aliases or Windows device stems; distinctness is ASCII case-insensitive. Reports contain existing learner callback/row/singleton counts and output-only call timings, without Setup/Playing timing partitions. Deployment export uses the live genuine handle.\n\
Use one writer and an external finite wall/disk watchdog. Published invocation close records describe synchronous source returns, not observed OS process closure. This command provides no fresh Test, Arena strength, provenance authentication or adoption approval.\n";

fn arguments(args: &[String]) -> Result<(&str, BTreeMap<&str, &str>), String> {
    let command = args.first().ok_or("Missing command")?.as_str();
    if !matches!(
        command,
        "continue" | "resume" | "inspect-tip" | "live-continue" | "live-resume" | "live-inspect"
    ) || !(args.len() - 1).is_multiple_of(2)
    {
        return Err("Expected continue/resume/inspect-tip and flag-value pairs".into());
    }
    let mut flags = BTreeMap::new();
    for pair in args[1..].as_chunks::<2>().0 {
        let name = pair[0].as_str();
        if !matches!(
            name,
            "--spec"
                | "--expected-sha256"
                | "--output"
                | "--head"
                | "--head-sha256"
                | "--report"
                | "--max-new-attempts"
                | "--max-new-updates"
                | "--deployment"
        ) || pair[1].is_empty()
            || flags.insert(name, pair[1].as_str()).is_some()
        {
            return Err("Unknown, repeated or empty flag".into());
        }
    }
    for name in ["--spec", "--expected-sha256", "--output", "--report"] {
        if !flags.contains_key(name) {
            return Err(format!("Missing {name}"));
        }
    }
    let live = command.starts_with("live-");
    if live && flags.contains_key("--max-new-attempts")
        || !live && flags.contains_key("--max-new-updates")
    {
        return Err("Invocation limit flag belongs to the other campaign mode".into());
    }
    if matches!(command, "continue" | "live-continue") {
        if flags.contains_key("--head") || flags.contains_key("--head-sha256") {
            return Err("Continue forbids child head flags".into());
        }
    } else if !flags.contains_key("--head") || !flags.contains_key("--head-sha256") {
        return Err("Resume/inspect-tip requires pinned latest child head".into());
    }
    if matches!(command, "inspect-tip" | "live-inspect")
        && (flags.contains_key("--deployment")
            || flags.contains_key("--max-new-attempts")
            || flags.contains_key("--max-new-updates"))
    {
        return Err("Inspect-tip forbids model/deployment/attempt flags".into());
    }
    Ok((command, flags))
}
fn run(args: &[String]) -> Result<i32, String> {
    let (command, flags) = arguments(args)?;
    let raw = read_bounded(Path::new(flags["--spec"]), 1024 * 1024)?;
    if command.starts_with("live-") {
        let spec = CurrentMixedLivePlanSpec::from_json(&raw, flags["--expected-sha256"])?;
        let updates = flags
            .get("--max-new-updates")
            .map(|s| s.parse::<usize>().map_err(|e| e.to_string()))
            .transpose()?
            .unwrap_or(spec.max_invocation_updates());
        return current_mixed_live_plan(
            spec,
            command,
            Path::new(flags["--output"]),
            flags
                .get("--head")
                .map(|path| (Path::new(path), flags["--head-sha256"])),
            updates,
            flags["--report"],
            flags.get("--deployment").copied(),
        );
    }
    let spec = CurrentMixedCampaignSpec::from_json(&raw, flags["--expected-sha256"])?;
    let attempts = flags
        .get("--max-new-attempts")
        .map(|value| value.parse::<usize>().map_err(|e| e.to_string()))
        .transpose()?
        .unwrap_or(64);
    current_mixed_campaign(
        spec,
        command,
        Path::new(flags["--output"]),
        flags
            .get("--head")
            .map(|path| (Path::new(path), flags["--head-sha256"])),
        attempts,
        flags["--report"],
        flags.get("--deployment").copied(),
    )
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
                serde_json::json!({"schema":"tzolkin-current-mixed-campaign-cli-error-v1","error":error})
            );
            std::process::exit(1);
        }
    }
}
