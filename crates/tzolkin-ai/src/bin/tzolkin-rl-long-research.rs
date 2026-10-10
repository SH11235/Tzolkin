//! Fixed development Validation between durable Long training updates.
use serde::{Deserialize, Serialize};
use serde_json::json;
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::Instant;
use tzolkin_ai::arena::{
    BORROWED_RL_ARENA_SCHEMA, BorrowedArenaConfig, BorrowedArenaPolicy,
    run_public_rl_validation_arena,
};
use tzolkin_ai::kernel::Kernel;
use tzolkin_ai::policy::HeuristicWeights;
use tzolkin_ai::public_native::PreparedPublicPolicy;
use tzolkin_ai::public_policy_long::LongRlParent;
use tzolkin_ai::public_rl_long_session::{
    LongRunReport, LongRunSpec, LongTrainingSession, MAX_SPEC_BYTES, read_bounded,
};
use tzolkin_ai::search::{PreparedSearch, SearchConfig};

const SMALL: usize = 64 * 1024;
const MODEL: usize = 16 * 1024 * 1024;
const ARENA: usize = 32 * 1024 * 1024;
const POINT_RESERVE: u64 = 2 * ARENA as u64 + 4 * SMALL as u64;

#[derive(Deserialize, Serialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct Input {
    path: PathBuf,
    sha256: String,
}
impl Input {
    fn read(&self, cap: usize) -> Result<Vec<u8>, String> {
        if self.sha256.len() != 64
            || !self
                .sha256
                .bytes()
                .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
        {
            return Err("Expected canonical lowercase raw SHA256".into());
        }
        let raw = read_bounded(&self.path, cap)?;
        if sha(&raw) != self.sha256 {
            return Err("Pinned input bytes differ".into());
        }
        Ok(raw)
    }
}
#[derive(Deserialize, Serialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct Inputs {
    schema: String,
    run_spec: Input,
    evaluation_spec: Input,
    bc_checkpoint: Input,
    dataset_manifest: Input,
    count1_checkpoint: Input,
}
#[derive(Deserialize, Serialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct EvaluationSpec {
    schema: String,
    catalog_hash: String,
    seeds: Vec<u32>,
    bootstrap_seed: u64,
    weights: HeuristicWeights,
    search: SearchConfig,
    max_points: usize,
    max_report_bytes: u64,
}
impl EvaluationSpec {
    fn validate(&self) -> Result<(), String> {
        self.weights.validate()?;
        self.search.validate()?;
        if self.schema != "tzolkin-rl-periodic-validation-spec-v1"
            || self.catalog_hash != tzolkin_ai::replay::catalog_hash()
            || !(1..=64).contains(&self.max_points)
            || !(POINT_RESERVE..=8 * 1024 * 1024 * 1024).contains(&self.max_report_bytes)
        {
            return Err("Invalid fixed evaluation specification".into());
        }
        for players in [3, 4] {
            self.arena(players).validate()?;
        }
        Ok(())
    }
    fn arena(&self, players: usize) -> BorrowedArenaConfig {
        BorrowedArenaConfig {
            schema: BORROWED_RL_ARENA_SCHEMA.into(),
            players,
            seeds: self.seeds.clone(),
            bootstrap_seed: self.bootstrap_seed,
        }
    }
}
fn sha(raw: &[u8]) -> String {
    format!("{:x}", Sha256::digest(raw))
}
fn decode<T: for<'de> Deserialize<'de>>(raw: &[u8]) -> Result<T, String> {
    serde_json::from_slice(raw).map_err(|e| e.to_string())
}
fn save_new(directory: &Path, name: &str, value: &impl Serialize) -> Result<usize, String> {
    let raw = serde_json::to_vec(value).map_err(|e| e.to_string())?;
    if raw.len() > SMALL {
        return Err("Small receipt byte cap exceeded".into());
    }
    let temporary = directory.join(format!("{name}.{}.tmp", std::process::id()));
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&temporary)
        .map_err(|e| e.to_string())?;
    file.write_all(&raw)
        .and_then(|()| file.sync_all())
        .map_err(|e| e.to_string())?;
    drop(file);
    fs::hard_link(&temporary, directory.join(name)).map_err(|e| e.to_string())?;
    fs::remove_file(temporary).map_err(|e| e.to_string())?;
    Ok(raw.len())
}

struct Research<'a> {
    session: LongTrainingSession,
    bc: &'a PreparedPublicPolicy,
    evaluation: EvaluationSpec,
    search: PreparedSearch,
    output: &'a Path,
    controls_sha256: String,
    recorded_due: u64,
    recorded_points: usize,
}
impl Research<'_> {
    fn point(&self, index: usize, previous_due: u64) -> Result<(u64, bool), String> {
        let snapshot = self.session.snapshot();
        let (candidate, artifact, count) = match self.session.current_parent() {
            LongRlParent::Count1(p) => (BorrowedArenaPolicy::Count1(p), p.artifact().checksum(), 1),
            LongRlParent::Long(p) => (
                BorrowedArenaPolicy::Long(p),
                p.artifact().checksum(),
                p.artifact().update_count(),
            ),
        };
        let started = json!({
            "schema":"tzolkin-rl-development-point-v1", "point":index,
            "kind":if index == 0 {"count1Baseline"} else {"updatedNotice"},
            "controlsSha256":self.controls_sha256,
            "checkpointFile":snapshot.checkpoint_file(), "checkpointSha256":snapshot.checkpoint_sha256(),
            "artifactChecksum":artifact, "updateCount":count,
            "knownAcceptedDecisions":snapshot.known_accepted_decisions(), "evaluationDue":snapshot.evaluation_due(),
            "crossedDue":if index == 0 {None} else {Some([previous_due + 1, snapshot.evaluation_due()])},
            "plannedGames":14 * self.evaluation.seeds.len(), "development":true, "freshTestClaimed":false,
        });
        let mut bytes = save_new(
            self.output,
            &format!("evaluation-{index:04}-started.json"),
            &started,
        )? as u64;
        let began = Instant::now();
        let mut reports = Vec::new();
        let mut failed = 0;
        for players in [3, 4] {
            // Absolute slots and frozen weights remain the same at every point.
            let pool = if players == 3 {
                vec![
                    BorrowedArenaPolicy::Heuristic(&self.evaluation.weights),
                    BorrowedArenaPolicy::CornFirstSetup(&self.evaluation.weights),
                    BorrowedArenaPolicy::PublicBc(self.bc),
                ]
            } else {
                vec![
                    BorrowedArenaPolicy::Heuristic(&self.evaluation.weights),
                    BorrowedArenaPolicy::CornFirstUxmalOpening(&self.evaluation.weights),
                    BorrowedArenaPolicy::PublicBc(self.bc),
                    BorrowedArenaPolicy::Search(&self.search),
                ]
            };
            let report = run_public_rl_validation_arena(
                &self.evaluation.arena(players),
                candidate,
                BorrowedArenaPolicy::PublicBc(self.bc),
                &pool,
            )?;
            failed += report.arena.statistics.failed_games;
            let name = format!("evaluation-{index:04}-{players}p.json");
            report.save_new(&self.output.join(&name))?;
            let raw = read_bounded(&self.output.join(&name), ARENA)?;
            bytes += raw.len() as u64;
            reports.push(json!({"name":name,"bytes":raw.len(),"sha256":sha(&raw)}));
        }
        bytes += save_new(
            self.output,
            &format!("evaluation-{index:04}-completed.json"),
            &json!({
                "started":started, "reports":reports, "failedGames":failed,
                "wallSeconds":began.elapsed().as_secs_f64(), "actualForwardCalls":null,
                "status":"completed", "strengthImprovementClaimed":false,
            }),
        )? as u64;
        Ok((bytes, failed != 0))
    }
    fn finish(
        &self,
        reason: &str,
        report: &LongRunReport,
        seconds: f64,
        status: i32,
    ) -> Result<i32, String> {
        let result = json!({
            "schema":"tzolkin-rl-research-stop-v1", "reason":reason, "controller":report,
            "recordedEvaluationPoints":self.recorded_points, "lastRecordedEvaluationDue":self.recorded_due,
            "deferredDueNotices":report.evaluation_due().saturating_sub(self.recorded_due),
            "invocationWallSeconds":seconds, "strengthImprovementClaimed":false,
        });
        save_new(self.output, "research-stop.json", &result)?;
        println!(
            "{}",
            serde_json::to_string(&result).map_err(|e| e.to_string())?
        );
        Ok(status)
    }
    fn drive(&mut self, invocation: Instant, mut stored: u64) -> Result<i32, String> {
        let mut last_due = 0;
        for index in 0..self.evaluation.max_points {
            if stored + POINT_RESERVE > self.evaluation.max_report_bytes {
                return self.finish(
                    "evaluationByteCapacity",
                    &self.session.snapshot(),
                    invocation.elapsed().as_secs_f64(),
                    2,
                );
            }
            if index != 0 {
                let report = match self.session.run_until_evaluation(last_due) {
                    Ok(report) => report,
                    Err(error) => {
                        save_new(
                            self.output,
                            "research-error.json",
                            &json!({"error":error,"controller":self.session.snapshot(),"actualWorkMayBeUnknown":true}),
                        )?;
                        return Err(error);
                    }
                };
                stored +=
                    save_new(self.output, &format!("training-{index:04}.json"), &report)? as u64;
                if report.stop_reason() != "evaluationDue" {
                    let status =
                        if matches!(report.stop_reason(), "batchFailed" | "gradientUnknown") {
                            1
                        } else if report.decision_budget_reached() {
                            0
                        } else {
                            2
                        };
                    return self.finish(
                        report.stop_reason(),
                        &report,
                        invocation.elapsed().as_secs_f64(),
                        status,
                    );
                }
            }
            let (bytes, failed) = self.point(index, last_due)?;
            stored += bytes;
            last_due = self.session.snapshot().evaluation_due();
            self.recorded_due = last_due;
            self.recorded_points += 1;
            if failed {
                return self.finish(
                    "evaluationFailedArms",
                    &self.session.snapshot(),
                    invocation.elapsed().as_secs_f64(),
                    1,
                );
            }
        }
        self.finish(
            "pointCapacity",
            &self.session.snapshot(),
            invocation.elapsed().as_secs_f64(),
            2,
        )
    }
}

fn arguments(args: &[String]) -> Result<BTreeMap<&str, &str>, String> {
    if args.first().map(String::as_str) != Some("train") || !(args.len() - 1).is_multiple_of(2) {
        return Err("Supports train only; research resume is not implemented".into());
    }
    let required = ["--inputs", "--expected-sha256", "--output"];
    let mut flags = BTreeMap::new();
    for pair in args[1..].as_chunks::<2>().0 {
        if !required.contains(&pair[0].as_str())
            || pair[1].is_empty()
            || flags.insert(pair[0].as_str(), pair[1].as_str()).is_some()
        {
            return Err("Unknown, repeated or empty flag".into());
        }
    }
    if required.iter().any(|f| !flags.contains_key(f)) {
        return Err("Missing required flag".into());
    }
    Ok(flags)
}
fn run(args: &[String]) -> Result<i32, String> {
    let invocation = Instant::now();
    let flags = arguments(args)?;
    let raw = Input {
        path: flags["--inputs"].into(),
        sha256: flags["--expected-sha256"].into(),
    }
    .read(SMALL)?;
    let inputs: Inputs = decode(&raw)?;
    if inputs.schema != "tzolkin-rl-research-input-pins-v1" {
        return Err("Unknown input pin schema".into());
    }
    let spec_raw = inputs.run_spec.read(MAX_SPEC_BYTES)?;
    let spec = LongRunSpec::from_json(&spec_raw)?;
    let evaluation: EvaluationSpec = decode(&inputs.evaluation_spec.read(SMALL)?)?;
    evaluation.validate()?;
    let search = PreparedSearch::new(&evaluation.search)?;
    let manifest = &inputs.dataset_manifest.path;
    if manifest.file_name() != Some(std::ffi::OsStr::new("manifest.json")) {
        return Err("Dataset input must pin manifest.json".into());
    }
    let dataset = manifest
        .parent()
        .ok_or("Dataset manifest has no directory")?;
    inputs.bc_checkpoint.read(MODEL)?;
    inputs
        .dataset_manifest
        .read(tzolkin_ai::policy_dataset::MAX_MANIFEST_BYTES as usize)?;
    let count1 = inputs
        .count1_checkpoint
        .read(tzolkin_ai::public_rl_session::MAX_CHECKPOINT_BYTES)?;
    let output = Path::new(flags["--output"]);
    // Keep the closed session journal directory free of research receipts, so
    // the existing Long restore can validate it independently of this CLI.
    let mut research_name = output
        .file_name()
        .ok_or("Output needs a directory name")?
        .to_os_string();
    research_name.push(".research");
    let research_output = output.with_file_name(research_name);
    match fs::symlink_metadata(&research_output) {
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
        Err(e) => return Err(e.to_string()),
        Ok(_) => return Err("Research output already exists; automatic retry is forbidden".into()),
    }
    match fs::symlink_metadata(output) {
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
        Err(e) => return Err(e.to_string()),
        Ok(_) => return Err("Output already exists; automatic resume/retry is forbidden".into()),
    }
    let began = Instant::now();
    let prepared = PreparedPublicPolicy::load(&inputs.bc_checkpoint.path, dataset, Kernel::Scalar)?;
    let qualification_seconds = began.elapsed().as_secs_f64();
    inputs.bc_checkpoint.read(MODEL)?;
    inputs
        .dataset_manifest
        .read(tzolkin_ai::policy_dataset::MAX_MANIFEST_BYTES as usize)?;
    // The library checks the complete destination path before creating/writing it.
    let session = LongTrainingSession::from_count1(
        spec,
        &prepared,
        &count1,
        &inputs.count1_checkpoint.sha256,
        output,
    )?;
    session.require_unused_validation_families(&evaluation.seeds)?;
    // from_count1 has checked the shared existing parent and created the
    // session with no links. create_dir refuses any concurrent collision.
    fs::create_dir(&research_output).map_err(|e| e.to_string())?;
    let bytes = save_new(
        &research_output,
        "research-controls.json",
        &json!({
            "inputs":inputs, "inputControlSha256":sha(&raw), "evaluationSpec":evaluation,
            "plannedMaxGames":14 * evaluation.seeds.len() * evaluation.max_points,
            "absolutePool3":["heuristic","cornFirstSetup","preparedBc"],
            "absolutePool4":["heuristic","cornFirstUxmalOpening","preparedBc","search"],
            "qualification":{"calls":1,"seconds":qualification_seconds,"instrumentedForwardCalls":null},
            "resumeImplemented":false, "externalWallWatchdogRequired":true,
            "sessionDirectory":output,
        }),
    )?;
    let controls = read_bounded(&research_output.join("research-controls.json"), SMALL)?;
    let mut research = Research {
        session,
        bc: &prepared,
        evaluation,
        search,
        output: &research_output,
        controls_sha256: sha(&controls),
        recorded_due: 0,
        recorded_points: 0,
    };
    research.drive(invocation, bytes as u64)
}
fn main() {
    match run(&std::env::args().skip(1).collect::<Vec<_>>()) {
        Ok(status) => std::process::exit(status),
        Err(error) => {
            eprintln!(
                "{}",
                json!({"schema":"tzolkin-rl-research-error-v1","error":error,"workUnknownUnlessReported":true})
            );
            std::process::exit(1);
        }
    }
}
