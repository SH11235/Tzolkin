use std::io::{Read, Write};
use std::path::Path;

use tzolkin_ai::kernel::Kernel;
use tzolkin_ai::policy_dataset::{export_native_files, native_source_files};
use tzolkin_ai::public_model::{LoadedPublicPolicy, MAX_OBSERVATION_BYTES, PublicPolicyArtifact};
use tzolkin_core::observation::Observation;

const HELP: &str = "tzolkin-public-ml choose --model PATH [--kernel scalar|auto|avx2|sse2|neon|simd128]\ntzolkin-public-ml export-native --input NATIVE_REPLAY_DIRECTORY --output NEW_DATASET_DIRECTORY\nchoose reads a bounded core Observation JSON from stdin and writes a Decision JSON.\nDefault kernel: scalar. Both commands require base 3-4p; export-native copies and independently verifies complete native replay JSON sources.\nTraining, human admission and selfplay/Arena registration are separate future units.";

fn export(args: &[String]) -> Result<(), String> {
    let mut input = None;
    let mut output = None;
    let mut index = 1;
    while index < args.len() {
        let name = &args[index];
        let slot = match name.as_str() {
            "--input" => &mut input,
            "--output" => &mut output,
            _ => return Err(format!("Unknown export-native flag {name}")),
        };
        let value = args
            .get(index + 1)
            .filter(|value| !value.starts_with("--"))
            .ok_or_else(|| format!("Missing value after {name}"))?;
        if slot.replace(value.as_str()).is_some() {
            return Err(format!("Repeated flag {name}"));
        }
        index += 2;
    }
    let paths = native_source_files(Path::new(input.ok_or("--input is required")?))?;
    let manifest = export_native_files(&paths, Path::new(output.ok_or("--output is required")?))?;
    serde_json::to_writer(std::io::stdout().lock(), &manifest).map_err(|e| e.to_string())?;
    println!();
    Ok(())
}

fn run() -> Result<(), String> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.is_empty() || args == ["help"] || args == ["--help"] {
        println!("{HELP}");
        return Ok(());
    }
    if args.first().map(String::as_str) == Some("export-native") {
        return export(&args);
    }
    if args.first().map(String::as_str) != Some("choose") {
        return Err("Only choose and export-native commands are implemented; use --help".into());
    }
    let mut model = None;
    let mut kernel = None;
    let mut index = 1;
    while index < args.len() {
        let name = &args[index];
        if !["--model", "--kernel"].contains(&name.as_str()) {
            return Err(format!("Unknown flag {name}"));
        }
        let value = args
            .get(index + 1)
            .filter(|value| !value.starts_with("--"))
            .ok_or_else(|| format!("Missing value after {name}"))?;
        let slot = if name == "--model" {
            &mut model
        } else {
            &mut kernel
        };
        if slot.replace(value.as_str()).is_some() {
            return Err(format!("Repeated flag {name}"));
        }
        index += 2;
    }
    let kernel = match kernel.unwrap_or("scalar") {
        "scalar" => Kernel::Scalar,
        "auto" => Kernel::Auto,
        "avx2" => Kernel::Avx2,
        "sse2" => Kernel::Sse2,
        "neon" => Kernel::Neon,
        "simd128" => Kernel::Simd128,
        _ => return Err("Unknown public policy inference kernel".into()),
    };
    let artifact = PublicPolicyArtifact::load(Path::new(model.ok_or("--model is required")?))?;
    let loaded = LoadedPublicPolicy::with_kernel(&artifact, kernel)?;
    let mut input = Vec::new();
    std::io::stdin()
        .lock()
        .take(MAX_OBSERVATION_BYTES as u64 + 1)
        .read_to_end(&mut input)
        .map_err(|error| error.to_string())?;
    if input.len() > MAX_OBSERVATION_BYTES {
        return Err("Observation stdin exceeds bounded 16 MiB input".into());
    }
    let observation: Observation =
        serde_json::from_slice(&input).map_err(|error| error.to_string())?;
    let decision = loaded.choose_move(&observation)?;
    let mut stdout = std::io::stdout().lock();
    serde_json::to_writer(&mut stdout, &decision).map_err(|error| error.to_string())?;
    stdout.write_all(b"\n").map_err(|error| error.to_string())
}
fn main() {
    if let Err(error) = run() {
        eprintln!("{error}");
        std::process::exit(1);
    }
}
