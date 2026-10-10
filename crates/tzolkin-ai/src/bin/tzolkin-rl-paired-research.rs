//! Fixed Validation research for durable public paired training sessions.
use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use tzolkin_ai::kernel::Kernel;
use tzolkin_ai::public_native::PreparedPublicPolicy;
use tzolkin_ai::public_rl_paired_research::{PairedEvaluationSpec, PairedPeriodicResearch};
use tzolkin_ai::public_rl_paired_session::{
    MAX_CHECKPOINT_BYTES, MAX_SPEC_BYTES, PairedRunSpec, PairedTrainingSession, read_bounded,
};

const INPUT_SCHEMA: &str = "tzolkin-rl-paired-research-input-pins-v1";
const CONTROL_BYTES: usize = 64 * 1024;
const MODEL_BYTES: usize = 16 * 1024 * 1024;
const MANIFEST_BYTES: usize = tzolkin_ai::policy_dataset::MAX_MANIFEST_BYTES as usize;
const HELP: &str = "tzolkin-rl-paired-research train --inputs FILE --expected-sha256 LOWERCASE64HEX --output NEW_SESSION_DIRECTORY\n\
tzolkin-rl-paired-research evaluate-current --inputs FILE --expected-sha256 LOWERCASE64HEX --output NEW_RESEARCH_DIRECTORY\n\
Inputs schema: tzolkin-rl-paired-research-input-pins-v1. Required fields: schema, runSpec, evaluationSpec, bcCheckpoint, datasetManifest. Each pinned input is {path, sha256}, with a lowercase raw SHA256. Train requires count1Checkpoint and forbids state. Evaluate-current requires state and forbids count1Checkpoint. DatasetManifest must name manifest.json.\n\
Train creates a new paired session plus a separate sibling directory ending in .research. It evaluates the actual Count1 baseline and updated paired owners using fixed Validation controls. Evaluate-current restores the pinned paired head from its existing session directory and writes a new research directory without collecting or updating. Specifications and capacities remain fixed. Prepared BC is qualified once with Scalar and borrowed throughout.\n\
Use a caller-owned single-writer directory and an external wall watchdog. Killing a synchronous call can leave unknown work and retained temporary files. Read the returned stop reason; exit status alone is not an evaluation result.\n";

#[derive(Deserialize, Serialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct Input {
    path: PathBuf,
    sha256: String,
}
impl Input {
    fn read(&self, maximum: usize) -> Result<Vec<u8>, String> {
        if !canonical_sha256(&self.sha256) {
            return Err("Expected canonical lowercase raw SHA256".into());
        }
        let raw = read_bounded(&self.path, maximum)?;
        if sha256(&raw) != self.sha256 {
            return Err("Pinned input raw SHA256 mismatch".into());
        }
        Ok(raw)
    }
}

#[derive(Deserialize, Serialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct InputsClosed {
    schema: String,
    run_spec: Input,
    evaluation_spec: Input,
    bc_checkpoint: Input,
    dataset_manifest: Input,
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        deserialize_with = "optional_pin"
    )]
    count1_checkpoint: Option<Input>,
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        deserialize_with = "optional_pin"
    )]
    state: Option<Input>,
}

fn optional_pin<'de, D: serde::Deserializer<'de>>(decoder: D) -> Result<Option<Input>, D::Error> {
    // An unused pin is omitted; an explicitly present pin must be an object.
    Input::deserialize(decoder).map(Some)
}

impl InputsClosed {
    fn from_json(raw: &[u8]) -> Result<Self, String> {
        // Decode the closed structs directly so repeated fields are rejected
        // before they could be collapsed into a generic JSON object.
        let mut decoder = serde_json::Deserializer::from_slice(raw);
        let inputs = Self::deserialize(&mut decoder).map_err(|e| e.to_string())?;
        decoder.end().map_err(|e| e.to_string())?;
        if inputs.schema != INPUT_SCHEMA {
            return Err("Unknown paired research input pin schema".into());
        }
        Ok(inputs)
    }

    fn source(&self, command: &str) -> Result<(&Input, usize), String> {
        if command == "train" {
            if self.state.is_some() {
                return Err("Train forbids state".into());
            }
            Ok((
                self.count1_checkpoint
                    .as_ref()
                    .ok_or("Train requires count1Checkpoint")?,
                tzolkin_ai::public_rl_session::MAX_CHECKPOINT_BYTES,
            ))
        } else {
            if self.count1_checkpoint.is_some() {
                return Err("Evaluate-current forbids count1Checkpoint".into());
            }
            Ok((
                self.state
                    .as_ref()
                    .ok_or("Evaluate-current requires state")?,
                MAX_CHECKPOINT_BYTES,
            ))
        }
    }
}

fn canonical_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}

fn sha256(raw: &[u8]) -> String {
    format!("{:x}", Sha256::digest(raw))
}

fn require_absent(path: &Path) -> Result<(), String> {
    match fs::symlink_metadata(path) {
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(error) => Err(error.to_string()),
        Ok(_) => Err("Output already exists; automatic resume/retry is forbidden".into()),
    }
}

fn arguments(args: &[String]) -> Result<(&str, BTreeMap<&str, &str>), String> {
    let command = args.first().ok_or("Missing command")?.as_str();
    if !matches!(command, "train" | "evaluate-current") || !(args.len() - 1).is_multiple_of(2) {
        return Err("Expected train/evaluate-current and flag-value pairs".into());
    }
    let required = ["--inputs", "--expected-sha256", "--output"];
    let mut flags = BTreeMap::new();
    for pair in args[1..].as_chunks::<2>().0 {
        let name = pair[0].as_str();
        if !required.contains(&name)
            || pair[1].is_empty()
            || flags.insert(name, pair[1].as_str()).is_some()
        {
            return Err("Unknown, repeated or empty flag".into());
        }
    }
    if required.iter().any(|name| !flags.contains_key(name)) {
        return Err("Missing required flag".into());
    }
    if !canonical_sha256(flags["--expected-sha256"]) {
        return Err("Expected canonical lowercase raw SHA256".into());
    }
    Ok((command, flags))
}

fn run(args: &[String]) -> Result<i32, String> {
    let (command, flags) = arguments(args)?;
    let control = Input {
        path: flags["--inputs"].into(),
        sha256: flags["--expected-sha256"].into(),
    };
    let raw = control.read(CONTROL_BYTES)?;
    let inputs = InputsClosed::from_json(&raw)?;
    let (source, source_maximum) = inputs.source(command)?;
    let spec = PairedRunSpec::from_json(&inputs.run_spec.read(MAX_SPEC_BYTES)?)?;
    let evaluation = PairedEvaluationSpec::from_json(&inputs.evaluation_spec.read(CONTROL_BYTES)?)?;
    if inputs.dataset_manifest.path.file_name() != Some(std::ffi::OsStr::new("manifest.json")) {
        return Err("Dataset input must pin manifest.json".into());
    }
    let dataset = inputs
        .dataset_manifest
        .path
        .parent()
        .ok_or("Dataset manifest has no directory")?;
    inputs.bc_checkpoint.read(MODEL_BYTES)?;
    inputs.dataset_manifest.read(MANIFEST_BYTES)?;
    let source_raw = source.read(source_maximum)?;
    let output = Path::new(flags["--output"]);
    let research_directory = if command == "train" {
        let mut name = output
            .file_name()
            .ok_or("Output needs a directory name")?
            .to_os_string();
        name.push(".research");
        let research = output.with_file_name(name);
        require_absent(output)?;
        research
    } else {
        output.to_path_buf()
    };
    require_absent(&research_directory)?;

    let prepared = PreparedPublicPolicy::load(&inputs.bc_checkpoint.path, dataset, Kernel::Scalar)?;
    // Recheck every raw pin after qualification, before any session or research
    // output is created. The retained source/spec bytes remain the pinned bytes.
    control.read(CONTROL_BYTES)?;
    inputs.run_spec.read(MAX_SPEC_BYTES)?;
    inputs.evaluation_spec.read(CONTROL_BYTES)?;
    inputs.bc_checkpoint.read(MODEL_BYTES)?;
    inputs.dataset_manifest.read(MANIFEST_BYTES)?;
    source.read(source_maximum)?;

    let mut research = if command == "train" {
        let session = PairedTrainingSession::from_count1(
            spec,
            &prepared,
            &source_raw,
            &source.sha256,
            output,
        )?;
        PairedPeriodicResearch::train(
            session,
            &prepared,
            evaluation,
            &research_directory,
            &control.sha256,
        )?
    } else {
        let session_directory = source
            .path
            .parent()
            .ok_or("Paired state has no directory")?;
        let session = PairedTrainingSession::restore(
            spec,
            &prepared,
            &source_raw,
            &source.sha256,
            session_directory,
        )?;
        PairedPeriodicResearch::evaluate_current(
            session,
            &prepared,
            evaluation,
            &research_directory,
            &control.sha256,
        )?
    };
    let stop = research.run()?;
    println!(
        "{}",
        serde_json::to_string(&stop).map_err(|e| e.to_string())?
    );
    Ok(stop.exit_status())
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.as_slice() == ["--help"] {
        println!("{HELP}");
        return;
    }
    match run(&args) {
        Ok(status) => std::process::exit(status),
        Err(error) => {
            eprintln!(
                "{}",
                serde_json::json!({"schema": "tzolkin-rl-paired-research-error-v1", "error": error})
            );
            std::process::exit(1);
        }
    }
}
