//! Opt-in bounded local stochastic collection/audit. No ML admission or producer authentication.
use std::collections::BTreeMap;
use std::fs::{self, File, OpenOptions};
use std::io::{Read, Write};
use std::path::{Component, Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

use serde::Serialize;
use sha2::{Digest, Sha256};
use tzolkin_ai::kernel::Kernel;
use tzolkin_ai::policy_dataset::MAX_FILES;
use tzolkin_ai::public_native::PreparedPublicPolicy;
use tzolkin_ai::public_stochastic::{SamplingSeed, SamplingStreamIdentity};
use tzolkin_ai::public_stochastic_native::{
    CollectionLimits, NativeStochasticConfig, collect_native,
};
use tzolkin_ai::public_stochastic_record::{
    AttemptStage, FailureReason, MAX_RECORD_BYTES, audit_record_bytes, encode_record,
};
use tzolkin_ai::replay::SeatPolicy;

const REPORT_SCHEMA: &str = "tzolkin-public-stochastic-cli-report-v1";
const REPORT_BYTES: usize = 64 * 1024;
const CHECKPOINT_BYTES: usize = 8 * 1024 * 1024;
const HELP: &str = "tzolkin-public-stochastic collect-native --checkpoint FILE --dataset DIRECTORY --players 3|4 --environment-seed U32 --sampling-seed U64 --episode-ordinal U64 --replicate-ordinal U64 --output NEW_FILE [--max-callbacks N --max-candidate-rows N --max-source-bytes N --max-trace-bytes N]\ntzolkin-public-stochastic audit-record --checkpoint FILE --dataset DIRECTORY --input FILE [--expect-sha256 LOWERCASE64HEX]\nAll seeds/ordinals are required decimal integers. Environment seed is u32; sampling seed and ordinals retain full u64. Base 3/4p, all options false, immutable qualified V2 policy, Scalar only.\nDefault limits: callbacks=4000, candidate-rows=1000000, source-bytes=67108864, trace-bytes=33554432. Record input/output cap=100663296 bytes; report cap=65536 bytes. These are serialized limits, not RAM or wall-clock limits.\nCollection exit: 0 complete/published, 2 valid failed prefix/published, 1 command/publication error. Audit exit0 means mechanical audit passed, including a valid failed prefix; see collectionComplete. Existing outputs are never overwritten. Parents must exist; regular inputs, no symlink/reparse, URL/UNC paths or Windows remote drives. Unix network mounts and hostile concurrent path changes are not a security sandbox guarantee.\nPreparation separately rechecks checkpoint/dataset/source EOF and final loss; it can perform model inference. No automatic audit after collection, fallback, retry, seed reservation, producer authentication, independent seed verification, training/PPO admission, or strength claim. Same-target likelihood bits are not guaranteed across compiler/OS targets.\n";

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct Counts {
    observed_callbacks: usize,
    accepted_samples: usize,
    sampler_attempts: Option<usize>,
    applied_choices: usize,
}
#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct Failure {
    reason: FailureReason,
    stage: Option<AttemptStage>,
    callback_index: Option<usize>,
    message: Option<String>,
}
#[derive(Serialize)]
struct Identity {
    bytes: usize,
    sha256: String,
}
#[derive(Default, Serialize)]
#[serde(rename_all = "camelCase")]
struct Publication {
    destination: String,
    published: bool,
    verified: bool,
    temp_path: Option<String>,
    temp_bytes: Option<usize>,
    temp_sha256: Option<String>,
    temp_removed: Option<bool>,
    cleanup_error: Option<String>,
    temp_inspection_error: Option<String>,
    error: Option<String>,
}
impl Publication {
    fn succeeded(&self) -> bool {
        self.published && self.verified && self.error.is_none() && self.cleanup_error.is_none()
    }
}
#[derive(Default, Serialize)]
#[serde(rename_all = "camelCase")]
struct Timings {
    prepare_seconds: Option<f64>,
    collection_seconds: Option<f64>,
    audit_seconds: Option<f64>,
}
#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct Configuration {
    players: usize,
    environment_seed: u32,
    // Decimal strings avoid loss in consumers whose JSON numbers are f64.
    sampling_seed: String,
    episode_ordinal: String,
    replicate_ordinal: String,
    max_callbacks: usize,
    max_candidate_rows: usize,
    max_source_bytes: usize,
    max_trace_bytes: usize,
}
#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct Report {
    schema: &'static str,
    command: Option<String>,
    phase: &'static str,
    operation_succeeded: bool,
    collection_started: bool,
    collection_complete: Option<bool>,
    audit_passed: Option<bool>,
    configuration: Option<Configuration>,
    counts: Option<Counts>,
    failure: Option<Failure>,
    terminal_state_key: Option<String>,
    provenance: Option<SeatPolicy>,
    raw_checkpoint: Option<Identity>,
    record: Option<Identity>,
    publication: Option<Publication>,
    timings: Timings,
    error: Option<String>,
    training_admission_available: bool,
    producer_authenticated: bool,
    independent_seed_origins_verified: bool,
}
impl Report {
    fn new() -> Self {
        Self {
            schema: REPORT_SCHEMA,
            command: None,
            phase: "arguments",
            operation_succeeded: false,
            collection_started: false,
            collection_complete: None,
            audit_passed: None,
            configuration: None,
            counts: None,
            failure: None,
            terminal_state_key: None,
            provenance: None,
            raw_checkpoint: None,
            record: None,
            publication: None,
            timings: Timings::default(),
            error: None,
            training_admission_available: false,
            producer_authenticated: false,
            independent_seed_origins_verified: false,
        }
    }
}

struct Common {
    checkpoint: PathBuf,
    dataset: PathBuf,
}
enum Command {
    Collect(Common, NativeStochasticConfig, PathBuf),
    Audit(Common, PathBuf, Option<String>),
}
fn flag<'a>(flags: &BTreeMap<&str, &'a str>, name: &str) -> Result<&'a str, String> {
    flags
        .get(name)
        .copied()
        .ok_or_else(|| format!("Required flag {name}"))
}
fn integer<T: std::str::FromStr>(value: &str, name: &str) -> Result<T, String> {
    if value.is_empty() || !value.bytes().all(|byte| byte.is_ascii_digit()) {
        return Err(format!("Expected decimal integer for {name}"));
    }
    value
        .parse()
        .map_err(|_| format!("Integer out of range for {name}"))
}
fn optional_integer(
    flags: &BTreeMap<&str, &str>,
    name: &str,
    default: usize,
) -> Result<usize, String> {
    flags
        .get(name)
        .map_or(Ok(default), |value| integer(value, name))
}
fn sha256_text(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}
fn parse(args: &[String], report: &mut Report) -> Result<Command, String> {
    if args.len() > 32 || args.iter().any(|arg| arg.len() > 4096) {
        return Err("CLI argument count/byte limit exceeded".into());
    }
    let command = args.first().ok_or("A command is required")?;
    if !["collect-native", "audit-record"].contains(&command.as_str()) {
        return Err("Unknown command; use --help".into());
    }
    report.command = Some(command.clone());
    let allowed: &[&str] = if command == "collect-native" {
        &[
            "--checkpoint",
            "--dataset",
            "--players",
            "--environment-seed",
            "--sampling-seed",
            "--episode-ordinal",
            "--replicate-ordinal",
            "--output",
            "--max-callbacks",
            "--max-candidate-rows",
            "--max-source-bytes",
            "--max-trace-bytes",
        ]
    } else {
        &["--checkpoint", "--dataset", "--input", "--expect-sha256"]
    };
    let mut flags = BTreeMap::new();
    for pair in args[1..].chunks(2) {
        let name = pair[0].as_str();
        if !allowed.contains(&name) {
            return Err("Unknown flag for this command".into());
        }
        let value = pair
            .get(1)
            .filter(|value| !value.starts_with("--"))
            .ok_or_else(|| format!("Missing value after {name}"))?;
        if flags.insert(name, value.as_str()).is_some() {
            return Err(format!("Repeated flag {name}"));
        }
    }
    let common = Common {
        checkpoint: flag(&flags, "--checkpoint")?.into(),
        dataset: flag(&flags, "--dataset")?.into(),
    };
    if command == "audit-record" {
        let expected = flags.get("--expect-sha256").map(|value| value.to_string());
        if expected.as_deref().is_some_and(|value| !sha256_text(value)) {
            return Err("--expect-sha256 requires lowercase 64 hex characters".into());
        }
        return Ok(Command::Audit(
            common,
            flag(&flags, "--input")?.into(),
            expected,
        ));
    }
    let players = integer(flag(&flags, "--players")?, "--players")?;
    let environment_seed = integer(flag(&flags, "--environment-seed")?, "--environment-seed")?;
    let sampling_seed = integer(flag(&flags, "--sampling-seed")?, "--sampling-seed")?;
    let episode = integer(flag(&flags, "--episode-ordinal")?, "--episode-ordinal")?;
    let replicate = integer(flag(&flags, "--replicate-ordinal")?, "--replicate-ordinal")?;
    let defaults = CollectionLimits::default();
    let limits = CollectionLimits::new(
        optional_integer(&flags, "--max-callbacks", defaults.max_callbacks())?,
        optional_integer(
            &flags,
            "--max-candidate-rows",
            defaults.max_candidate_rows(),
        )?,
        optional_integer(&flags, "--max-source-bytes", defaults.max_source_bytes())?,
        optional_integer(&flags, "--max-trace-bytes", defaults.max_trace_bytes())?,
    )?;
    let identity = SamplingStreamIdentity::new(
        SamplingSeed::new(sampling_seed),
        episode,
        replicate,
        players,
    )?;
    let config = NativeStochasticConfig::new(players, environment_seed, identity, limits)?;
    report.configuration = Some(Configuration {
        players,
        environment_seed,
        sampling_seed: sampling_seed.to_string(),
        episode_ordinal: episode.to_string(),
        replicate_ordinal: replicate.to_string(),
        max_callbacks: limits.max_callbacks(),
        max_candidate_rows: limits.max_candidate_rows(),
        max_source_bytes: limits.max_source_bytes(),
        max_trace_bytes: limits.max_trace_bytes(),
    });
    Ok(Command::Collect(
        common,
        config,
        flag(&flags, "--output")?.into(),
    ))
}

#[derive(Clone, Copy)]
enum PathKind {
    File,
    Directory,
    NewFile,
}
#[cfg(windows)]
fn reparse(metadata: &fs::Metadata) -> bool {
    use std::os::windows::fs::MetadataExt;
    metadata.file_attributes() & 0x400 != 0
}
#[cfg(not(windows))]
fn reparse(_: &fs::Metadata) -> bool {
    false
}
#[cfg(windows)]
fn local_drive(path: &Path) -> Result<(), String> {
    use std::os::windows::ffi::OsStrExt;
    #[link(name = "kernel32")]
    unsafe extern "system" {
        #[link_name = "GetDriveTypeW"]
        fn get_drive_type_w(root: *const u16) -> u32;
    }
    let root: PathBuf = path.components().take(2).collect();
    let root: Vec<u16> = root.as_os_str().encode_wide().chain(Some(0)).collect();
    // The null-terminated root lives through the call. Query before filesystem traversal.
    let kind = unsafe { get_drive_type_w(root.as_ptr()) };
    if ![2, 3, 5, 6].contains(&kind) {
        return Err("Network/unknown/nonexistent drive rejected".into());
    }
    Ok(())
}
#[cfg(not(windows))]
fn local_drive(_: &Path) -> Result<(), String> {
    Ok(())
}

#[cfg(any(windows, test))]
fn windows_component_rejected(name: &str) -> bool {
    if name.ends_with(' ')
        || name.ends_with('.')
        || name
            .chars()
            .any(|ch| matches!(ch, '<' | '>' | '"' | '|' | '?' | '*'))
    {
        return true;
    }
    // Win32 reserves device names even with an extension. Reject aliases before
    // any metadata/open call rather than relying on a device's file attributes.
    let stem = name.split('.').next().unwrap_or("").trim_end_matches(' ');
    let upper = stem.to_ascii_uppercase();
    if matches!(
        upper.as_str(),
        "CON" | "PRN" | "AUX" | "NUL" | "CONIN$" | "CONOUT$" | "CLOCK$"
    ) {
        return true;
    }
    ["COM", "LPT"].into_iter().any(|prefix| {
        upper.strip_prefix(prefix).is_some_and(|suffix| {
            matches!(
                suffix,
                "1" | "2" | "3" | "4" | "5" | "6" | "7" | "8" | "9" | "¹" | "²" | "³"
            )
        })
    })
}

fn local_path(path: &Path, kind: PathKind) -> Result<PathBuf, String> {
    let raw = path.to_str().ok_or("Non-UTF8 path rejected")?;
    if raw.is_empty()
        || raw.contains("://")
        || raw.starts_with('\\')
        || raw.starts_with("//")
        || raw.starts_with("/\\")
        || raw.chars().any(char::is_control)
        || path
            .components()
            .any(|part| matches!(part, Component::ParentDir))
    {
        return Err("Expected an explicit local path without parent traversal".into());
    }
    if matches!(path.components().next(), Some(Component::Prefix(_))) && !path.is_absolute() {
        return Err("Drive-relative path rejected".into());
    }
    if path.has_root() && !path.is_absolute() {
        return Err("Root-relative path rejected".into());
    }
    let absolute = if path.is_absolute() {
        path.to_path_buf()
    } else {
        std::env::current_dir()
            .map_err(|error| error.to_string())?
            .join(path)
    };
    if !absolute.is_absolute()
        || absolute.to_string_lossy().starts_with("\\\\")
        || absolute.to_string_lossy().starts_with("//")
    {
        return Err("Network/root-relative hierarchy rejected".into());
    }
    for part in absolute.components() {
        if let Component::Prefix(prefix) = part {
            if !matches!(prefix.kind(), std::path::Prefix::Disk(_)) {
                return Err("Device/UNC/verbatim path rejected".into());
            }
        } else if let Component::Normal(name) = part {
            if name.to_string_lossy().contains(':') {
                return Err("Alternate stream/device path rejected".into());
            }
            #[cfg(windows)]
            if windows_component_rejected(&name.to_string_lossy()) {
                return Err("Windows reserved device/ambiguous component rejected".into());
            }
        }
    }
    local_drive(&absolute)?;
    let mut current = PathBuf::new();
    for part in absolute.components() {
        if matches!(part, Component::CurDir) {
            continue;
        }
        current.push(part.as_os_str());
        if matches!(part, Component::Prefix(_)) {
            continue;
        }
        let leaf = current == absolute;
        match fs::symlink_metadata(&current) {
            Ok(metadata) => {
                if metadata.file_type().is_symlink() || reparse(&metadata) {
                    return Err("Symlink/reparse hierarchy rejected".into());
                }
                if leaf {
                    let valid = match kind {
                        PathKind::File => metadata.is_file(),
                        PathKind::Directory => metadata.is_dir(),
                        PathKind::NewFile => false,
                    };
                    if !valid {
                        return Err("Expected regular input/directory or an absent output".into());
                    }
                } else if !metadata.is_dir() {
                    return Err("Non-directory ancestor rejected".into());
                }
            }
            Err(error)
                if error.kind() == std::io::ErrorKind::NotFound
                    && leaf
                    && matches!(kind, PathKind::NewFile) => {}
            Err(error) => return Err(format!("Local hierarchy check: {error}")),
        }
    }
    Ok(absolute)
}
fn dataset_tree(directory: &Path) -> Result<(), String> {
    // Reject reparse entries before the existing loader can follow any referenced file.
    // Bounds cover the existing MAX_FILES source and shard collections, plus directories.
    let mut pending = vec![(local_path(directory, PathKind::Directory)?, 0)];
    let mut entries = 0usize;
    while let Some((directory, depth)) = pending.pop() {
        if depth > 8 {
            return Err("Dataset directory depth bound exceeded".into());
        }
        for entry in fs::read_dir(directory).map_err(|error| error.to_string())? {
            entries += 1;
            if entries > 2 * MAX_FILES + 16 {
                return Err("Dataset directory entry bound exceeded".into());
            }
            let path = entry.map_err(|error| error.to_string())?.path();
            let metadata = fs::symlink_metadata(&path).map_err(|error| error.to_string())?;
            if metadata.file_type().is_symlink() || reparse(&metadata) {
                return Err("Dataset symlink/reparse entry rejected".into());
            }
            if metadata.is_dir() {
                pending.push((path, depth + 1));
            } else if !metadata.is_file() {
                return Err("Dataset nonregular entry rejected".into());
            }
        }
    }
    Ok(())
}
fn bounded_read(path: &Path, maximum: usize) -> Result<Vec<u8>, String> {
    let path = local_path(path, PathKind::File)?;
    let metadata = fs::symlink_metadata(&path).map_err(|error| error.to_string())?;
    if metadata.len() > maximum as u64 {
        return Err(if maximum == MAX_RECORD_BYTES {
            "Record exceeds 96 MiB"
        } else {
            "File exceeds byte bound"
        }
        .into());
    }
    let file = File::open(path).map_err(|error| error.to_string())?;
    if file.metadata().map_err(|error| error.to_string())?.len() > maximum as u64 {
        return Err("Opened file exceeds byte bound".into());
    }
    let mut bytes = Vec::new();
    file.take(maximum as u64 + 1)
        .read_to_end(&mut bytes)
        .map_err(|error| error.to_string())?;
    if bytes.len() > maximum {
        return Err("File grew beyond byte bound".into());
    }
    Ok(bytes)
}
fn identity(bytes: &[u8]) -> Identity {
    Identity {
        bytes: bytes.len(),
        sha256: format!("{:x}", Sha256::digest(bytes)),
    }
}
fn unchanged(path: &Path, expected: &Identity, maximum: usize) -> Result<(), String> {
    let actual = identity(&bounded_read(path, maximum)?);
    if actual.bytes != expected.bytes || actual.sha256 != expected.sha256 {
        return Err("Input byte identity changed during command".into());
    }
    Ok(())
}
fn small_error(error: impl ToString) -> String {
    error.to_string().chars().take(1024).collect()
}
fn publish_bytes_with(
    output: &Path,
    temp: &Path,
    bytes: &[u8],
    before_link: impl FnOnce() -> std::io::Result<()>,
    cleanup: impl FnOnce(&Path) -> std::io::Result<()>,
) -> Publication {
    let mut result = Publication {
        destination: output.to_string_lossy().into_owned(),
        ..Publication::default()
    };
    let publish = (|| -> Result<(), String> {
        if bytes.len() > MAX_RECORD_BYTES {
            return Err("Output record exceeds 96 MiB".into());
        }
        local_path(output, PathKind::NewFile)?;
        local_path(temp, PathKind::NewFile)?;
        if output.parent() != temp.parent() || output == temp {
            return Err("Temporary must be a distinct sibling".into());
        }
        let mut file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(temp)
            .map_err(|error| error.to_string())?;
        result.temp_path = Some(temp.to_string_lossy().into_owned());
        file.write_all(bytes)
            .and_then(|()| file.sync_all())
            .map_err(|error| error.to_string())?;
        drop(file);
        before_link().map_err(|error| error.to_string())?;
        local_path(output, PathKind::NewFile)?;
        local_path(temp, PathKind::File)?;
        fs::hard_link(temp, output).map_err(|error| error.to_string())?;
        result.published = true;
        unchanged(output, &identity(bytes), MAX_RECORD_BYTES)?;
        result.verified = true;
        match cleanup(temp) {
            Ok(()) => result.temp_removed = Some(true),
            Err(error) => {
                result.temp_removed = Some(false);
                result.cleanup_error = Some(small_error(error));
            }
        }
        Ok(())
    })();
    if let Err(error) = publish {
        result.error = Some(small_error(error));
    }
    if result.temp_path.is_some() && result.temp_removed != Some(true) {
        result.temp_removed = Some(false);
        match bounded_read(temp, MAX_RECORD_BYTES) {
            Ok(bytes) => {
                let actual = identity(&bytes);
                result.temp_bytes = Some(actual.bytes);
                result.temp_sha256 = Some(actual.sha256);
            }
            Err(error) => result.temp_inspection_error = Some(small_error(error)),
        }
    }
    result
}
fn publish_bytes(output: &Path, bytes: &[u8]) -> Publication {
    static NEXT: AtomicU64 = AtomicU64::new(0);
    let temp = output.with_file_name(format!(
        ".tzolkin-stochastic-{}-{}.tmp",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    ));
    publish_bytes_with(
        output,
        &temp,
        bytes,
        || Ok(()),
        |path| fs::remove_file(path),
    )
}
fn run(args: &[String], report: &mut Report) -> Result<i32, String> {
    let command = parse(args, report)?;
    report.phase = "paths";
    let (common, input, output) = match &command {
        Command::Collect(common, _, output) => {
            (common, None, Some(local_path(output, PathKind::NewFile)?))
        }
        Command::Audit(common, input, _) => {
            (common, Some(local_path(input, PathKind::File)?), None)
        }
    };
    let input_bytes = input
        .as_ref()
        .map(|path| bounded_read(path, MAX_RECORD_BYTES))
        .transpose()?;
    if let Some(bytes) = &input_bytes {
        report.record = Some(identity(bytes));
        if let Command::Audit(_, _, Some(expected)) = &command
            && report
                .record
                .as_ref()
                .is_some_and(|actual| actual.sha256 != *expected)
        {
            return Err("Record SHA256 does not match --expect-sha256".into());
        }
    }
    let checkpoint = local_path(&common.checkpoint, PathKind::File)?;
    let dataset = local_path(&common.dataset, PathKind::Directory)?;
    dataset_tree(&dataset)?;
    report.raw_checkpoint = Some(identity(&bounded_read(&checkpoint, CHECKPOINT_BYTES)?));
    report.phase = "prepare";
    let started = Instant::now();
    let prepared = PreparedPublicPolicy::load(&checkpoint, &dataset, Kernel::Scalar);
    report.timings.prepare_seconds = Some(started.elapsed().as_secs_f64());
    let prepared = prepared?;
    unchanged(
        &checkpoint,
        report
            .raw_checkpoint
            .as_ref()
            .ok_or("Missing checkpoint identity")?,
        CHECKPOINT_BYTES,
    )?;
    let handle = prepared.handle()?;
    report.provenance = Some(handle.provenance().clone());
    match command {
        Command::Collect(_, config, _) => {
            report.phase = "collect";
            report.collection_started = true;
            let started = Instant::now();
            let collected = collect_native(&config, &handle);
            report.timings.collection_seconds = Some(started.elapsed().as_secs_f64());
            let collected = collected?;
            let complete = collected.complete();
            report.collection_complete = Some(complete);
            report.counts = Some(Counts {
                observed_callbacks: collected.observed_callbacks(),
                accepted_samples: collected.accepted_samples(),
                sampler_attempts: Some(collected.sampler_attempts()),
                applied_choices: collected.applied_choices(),
            });
            report.failure = collected.failure_reason().map(|reason| Failure {
                reason,
                stage: collected.failure_stage(),
                callback_index: collected.failure_callback_index(),
                message: collected.failure_message().map(small_error),
            });
            report.terminal_state_key = collected.terminal_state_key().map(str::to_owned);
            report.phase = "encode";
            let bytes = encode_record(&collected)?;
            report.record = Some(identity(&bytes));
            report.phase = "integrity";
            unchanged(
                &checkpoint,
                report
                    .raw_checkpoint
                    .as_ref()
                    .ok_or("Missing checkpoint identity")?,
                CHECKPOINT_BYTES,
            )?;
            report.phase = "publish";
            let publication = publish_bytes(output.as_ref().ok_or("Missing output")?, &bytes);
            let succeeded = publication.succeeded();
            report.publication = Some(publication);
            if !succeeded {
                return Err("Record publication failed; preserved publication details".into());
            }
            report.operation_succeeded = true;
            report.phase = "complete";
            Ok(if complete { 0 } else { 2 })
        }
        Command::Audit(_, _, _) => {
            report.phase = "audit";
            let started = Instant::now();
            let audited = audit_record_bytes(
                input_bytes.as_deref().ok_or("Missing record input")?,
                &handle,
            );
            report.timings.audit_seconds = Some(started.elapsed().as_secs_f64());
            report.audit_passed = Some(audited.is_ok());
            let audited = audited?;
            report.collection_complete = Some(audited.complete());
            report.counts = Some(Counts {
                observed_callbacks: audited.observed_callbacks(),
                accepted_samples: audited.accepted_samples(),
                sampler_attempts: None,
                applied_choices: audited.applied_choices(),
            });
            report.failure = audited.failure_reason().map(|reason| Failure {
                reason,
                stage: None,
                callback_index: None,
                message: None,
            });
            report.terminal_state_key = audited.terminal_state_key().map(str::to_owned);
            report.phase = "integrity";
            unchanged(
                input.as_ref().ok_or("Missing input path")?,
                report.record.as_ref().ok_or("Missing record identity")?,
                MAX_RECORD_BYTES,
            )?;
            unchanged(
                &checkpoint,
                report
                    .raw_checkpoint
                    .as_ref()
                    .ok_or("Missing checkpoint identity")?,
                CHECKPOINT_BYTES,
            )?;
            report.operation_succeeded = true;
            report.phase = "complete";
            Ok(0)
        }
    }
}
fn report_bytes(report: &Report) -> Result<Vec<u8>, String> {
    let mut bytes = serde_json::to_vec(report).map_err(|error| error.to_string())?;
    if bytes.len() >= REPORT_BYTES {
        return Err("Structured report exceeds 64 KiB".into());
    }
    bytes.push(b'\n');
    Ok(bytes)
}
fn main() {
    let args: Result<Vec<_>, _> = std::env::args_os()
        .skip(1)
        .map(|arg| arg.into_string())
        .collect();
    let mut report = Report::new();
    let args = match args {
        Ok(args) => args,
        Err(_) => {
            report.error = Some("Non-UTF8 argument rejected".into());
            if let Ok(bytes) = report_bytes(&report) {
                let _ = std::io::stdout().lock().write_all(&bytes);
            }
            std::process::exit(1);
        }
    };
    if args == ["--help"] || args == ["-h"] {
        let written = std::io::stdout().lock().write_all(HELP.as_bytes());
        std::process::exit(if written.is_ok() { 0 } else { 1 });
    }
    let mut exit = match run(&args, &mut report) {
        Ok(exit) => exit,
        Err(error) => {
            report.error = Some(small_error(error));
            1
        }
    };
    let written = report_bytes(&report).and_then(|bytes| {
        std::io::stdout()
            .lock()
            .write_all(&bytes)
            .map_err(|error| error.to_string())
    });
    if let Err(error) = written {
        report.phase = "report";
        report.operation_succeeded = false;
        report.error = Some(format!("stdout report failed: {}", small_error(error)));
        if let Ok(bytes) = report_bytes(&report) {
            let _ = std::io::stderr().lock().write_all(&bytes);
        }
        exit = 1;
    }
    std::process::exit(exit);
}

#[cfg(test)]
mod tests {
    use super::*;
    struct Temp(PathBuf);
    impl Temp {
        fn new() -> Self {
            static NEXT: AtomicU64 = AtomicU64::new(0);
            let path = std::env::temp_dir().join(format!(
                "tzolkin-stochastic-cli-unit-{}-{}",
                std::process::id(),
                NEXT.fetch_add(1, Ordering::Relaxed)
            ));
            fs::create_dir(&path).unwrap();
            Self(path)
        }
    }
    impl Drop for Temp {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }
    #[test]
    fn publication_exclusive_collision_and_cleanup_failure_preserve_facts() {
        let temp = Temp::new();
        let destination = temp.0.join("record.json");
        let scratch = temp.0.join("owned.tmp");
        let collision = publish_bytes_with(
            &destination,
            &scratch,
            b"new",
            || fs::write(&destination, b"foreign"),
            |path| fs::remove_file(path),
        );
        assert!(!collision.published && !collision.succeeded());
        assert_eq!(fs::read(&destination).unwrap(), b"foreign");
        assert_eq!(collision.temp_bytes, Some(3));
        assert_eq!(collision.temp_removed, Some(false));
        let destination2 = temp.0.join("second.json");
        let scratch2 = temp.0.join("owned2.tmp");
        let cleanup = publish_bytes_with(
            &destination2,
            &scratch2,
            b"valid",
            || Ok(()),
            |_| Err(std::io::Error::other("injected cleanup failure")),
        );
        assert!(cleanup.published && cleanup.verified && !cleanup.succeeded());
        assert!(cleanup.error.is_none() && cleanup.cleanup_error.is_some());
        assert_eq!(cleanup.temp_bytes, Some(5));
        assert_eq!(fs::read(&destination2).unwrap(), b"valid");
    }
    #[test]
    fn publisher_does_not_touch_foreign_temporary_and_verifies_success() {
        let temp = Temp::new();
        let destination = temp.0.join("record.json");
        let scratch = temp.0.join("foreign.tmp");
        fs::write(&scratch, b"foreign").unwrap();
        let failure = publish_bytes_with(
            &destination,
            &scratch,
            b"new",
            || Ok(()),
            |path| fs::remove_file(path),
        );
        assert!(failure.temp_path.is_none() && !failure.published);
        assert_eq!(fs::read(scratch).unwrap(), b"foreign");
        let success = publish_bytes(&destination, b"complete bytes");
        assert!(success.succeeded());
        assert_eq!(success.temp_removed, Some(true));
        assert_eq!(fs::read(&destination).unwrap(), b"complete bytes");
        let retry = publish_bytes(&destination, b"replacement");
        assert!(!retry.published && !retry.succeeded());
        assert_eq!(fs::read(destination).unwrap(), b"complete bytes");
    }
    #[test]
    fn local_hierarchy_and_report_bounds_are_enforced() {
        let temp = Temp::new();
        assert!(local_path(&temp.0.join("new.json"), PathKind::NewFile).is_ok());
        assert!(local_path(&temp.0.join("missing-parent/new.json"), PathKind::NewFile).is_err());
        for path in [
            "https://example.invalid/record",
            "//server/share/file",
            "\\\\server\\share\\file",
            "../file",
            "file:stream",
        ] {
            assert!(
                local_path(Path::new(path), PathKind::NewFile).is_err(),
                "{path}"
            );
        }
        let mut report = Report::new();
        report.error = Some("long".repeat(REPORT_BYTES));
        assert!(report_bytes(&report).is_err());
        report.error = Some(small_error("x".repeat(REPORT_BYTES)));
        assert!(report_bytes(&report).unwrap().len() < REPORT_BYTES);
    }
    #[test]
    fn windows_device_names_are_rejected_before_filesystem_access() {
        for name in [
            "NUL",
            "con.json",
            "aUx .json",
            "PRN",
            "COM1.log",
            "lpt9",
            "COM¹.txt",
            "LPT²",
            "CONIN$",
            "CONOUT$",
            "CLOCK$",
            "record.",
            "record ",
            "bad*name",
        ] {
            assert!(windows_component_rejected(name), "{name}");
            #[cfg(windows)]
            assert!(local_path(&std::env::temp_dir().join(name), PathKind::NewFile).is_err());
        }
        for name in [
            "record.json",
            "company",
            "COM0",
            "LPT10",
            "auxiliary",
            ".record",
        ] {
            assert!(!windows_component_rejected(name), "{name}");
        }
    }
}
