use crate::test_temp_root;
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::ffi::{OsStr, OsString};
use std::fs::{self, File};
use std::path::{Path, PathBuf};
use std::process::{Command, Output};
use tzolkin_ai::dataset::{self, DatasetSplit};
use tzolkin_ai::policy_dataset::{export_native_files, load_policy_dataset};
use tzolkin_ai::policy_training::{PolicyBcConfig, train_dataset};
use tzolkin_ai::public_stochastic_record::MAX_RECORD_BYTES;
use tzolkin_ai::replay;
use tzolkin_core::GameOptions;

struct Temp(PathBuf);
impl Temp {
    fn new() -> Self {
        let path = test_temp_root::create("tzolkin-stochastic-cli").unwrap();
        Self(path)
    }
}
impl Drop for Temp {
    fn drop(&mut self) {
        // Only this test's create-new temporary directory is removed.
        let _ = fs::remove_dir_all(&self.0);
    }
}
fn digest(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}
fn args(values: &[&str]) -> Vec<OsString> {
    values.iter().map(OsString::from).collect()
}
fn flag(values: &mut Vec<OsString>, name: &str, value: impl AsRef<OsStr>) {
    values.push(name.into());
    values.push(value.as_ref().into());
}
fn replace(values: &mut [OsString], name: &str, value: impl AsRef<OsStr>) {
    let index = values.iter().position(|v| v == name).unwrap();
    values[index + 1] = value.as_ref().into();
}
fn invoke(values: &[OsString]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_tzolkin-public-stochastic"))
        .args(values)
        .output()
        .unwrap()
}
fn report(output: &Output, exit: i32) -> Value {
    assert_eq!(
        output.status.code(),
        Some(exit),
        "stdout={}\nstderr={}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(output.stdout.len() <= 64 * 1024);
    let report: Value = serde_json::from_slice(&output.stdout).unwrap_or_else(|error| {
        panic!(
            "report parse: {error}; stdout={}; stderr={}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        )
    });
    assert_eq!(report["schema"], "tzolkin-public-stochastic-cli-report-v1");
    assert_eq!(report["trainingAdmissionAvailable"], false);
    assert_eq!(report["producerAuthenticated"], false);
    assert_eq!(report["independentSeedOriginsVerified"], false);
    report
}
fn early_failure(output: &Output) -> Value {
    let r = report(output, 1);
    assert_eq!(r["operationSucceeded"], false);
    assert_eq!(r["collectionStarted"], false);
    assert!(r["collectionComplete"].is_null());
    assert!(r["counts"].is_null());
    assert!(r["provenance"].is_null());
    assert!(r["error"].is_string());
    r
}
fn collect_args(temp: &Temp) -> Vec<OsString> {
    let mut values = args(&[
        "collect-native",
        "--players",
        "3",
        "--environment-seed",
        "17",
        "--sampling-seed",
        "7",
        "--episode-ordinal",
        "0",
        "--replicate-ordinal",
        "0",
        "--max-callbacks",
        "2",
    ]);
    flag(
        &mut values,
        "--checkpoint",
        temp.0.join("missing-checkpoint.json"),
    );
    flag(&mut values, "--dataset", temp.0.join("missing-dataset"));
    flag(&mut values, "--output", temp.0.join("record.json"));
    values
}
fn audit_args(temp: &Temp, input: &Path) -> Vec<OsString> {
    let mut values = args(&["audit-record"]);
    flag(
        &mut values,
        "--checkpoint",
        temp.0.join("missing-checkpoint.json"),
    );
    flag(&mut values, "--dataset", temp.0.join("missing-dataset"));
    flag(&mut values, "--input", input);
    values
}

#[test]
fn closed_flags_and_numeric_domains_refuse_before_collection() {
    let temp = Temp::new();
    let help = invoke(&args(&["--help"]));
    assert_eq!(help.status.code(), Some(0));
    assert!(help.stdout.len() < 64 * 1024);
    let help = String::from_utf8(help.stdout).unwrap();
    assert!(help.contains("--environment-seed U32"));
    assert!(help.contains("--sampling-seed U64"));
    assert!(help.contains("audit-record"));
    early_failure(&invoke(&[]));
    early_failure(&invoke(&args(&["unknown-command"])));
    let baseline = collect_args(&temp);
    let baseline_failure = early_failure(&invoke(&baseline));
    // Valid whole-u64 values reach the same missing-input preflight as small ones.
    let mut maximum = baseline.clone();
    for name in [
        "--sampling-seed",
        "--episode-ordinal",
        "--replicate-ordinal",
    ] {
        replace(&mut maximum, name, u64::MAX.to_string());
    }
    replace(&mut maximum, "--environment-seed", u32::MAX.to_string());
    assert_eq!(
        early_failure(&invoke(&maximum))["phase"],
        baseline_failure["phase"]
    );

    for (name, value) in [
        ("--players", "2"),
        ("--players", "5"),
        ("--environment-seed", "4294967296"),
        ("--sampling-seed", "18446744073709551616"),
        ("--episode-ordinal", "18446744073709551616"),
        ("--replicate-ordinal", "18446744073709551616"),
        ("--sampling-seed", "-1"),
        ("--sampling-seed", "+7"),
        ("--sampling-seed", "0x7"),
        ("--environment-seed", "1.0"),
        ("--max-callbacks", "0"),
        ("--max-callbacks", "4001"),
        ("--max-candidate-rows", "0"),
        ("--max-candidate-rows", "1000001"),
        ("--max-source-bytes", "65535"),
        ("--max-source-bytes", "67108865"),
        ("--max-trace-bytes", "4095"),
        ("--max-trace-bytes", "33554433"),
    ] {
        let mut invalid = baseline.clone();
        if invalid.iter().any(|v| v == name) {
            replace(&mut invalid, name, value);
        } else {
            flag(&mut invalid, name, value);
        }
        let r = early_failure(&invoke(&invalid));
        assert_ne!(r["phase"], baseline_failure["phase"], "{name} {value}: {r}");
    }
    for extra in [
        args(&["--kernel", "scalar"]),
        args(&["--tribes", "true"]),
        args(&["--players", "4"]),
        args(&["--output"]),
        args(&["unexpected-positional"]),
    ] {
        let mut invalid = baseline.clone();
        invalid.extend(extra);
        early_failure(&invoke(&invalid));
    }
    for removed in [
        "--checkpoint",
        "--dataset",
        "--players",
        "--environment-seed",
        "--sampling-seed",
        "--episode-ordinal",
        "--replicate-ordinal",
        "--output",
    ] {
        let mut invalid = baseline.clone();
        let index = invalid.iter().position(|v| v == removed).unwrap();
        invalid.drain(index..index + 2);
        early_failure(&invoke(&invalid));
    }
    assert!(!temp.0.join("record.json").exists());
}

#[test]
fn local_paths_new_only_and_record_bound_are_checked_before_qualification() {
    let temp = Temp::new();
    let baseline = collect_args(&temp);
    let existing = temp.0.join("existing.json");
    fs::write(&existing, b"preserved output bytes").unwrap();
    let original = fs::read(&existing).unwrap();
    let mut overwrite = baseline.clone();
    replace(&mut overwrite, "--output", &existing);
    early_failure(&invoke(&overwrite));
    assert_eq!(fs::read(&existing).unwrap(), original);

    for path in [
        OsString::from("https://example.invalid/output.json"),
        OsString::from("//server/share/output.json"),
        OsString::from("\\\\server\\share\\output.json"),
        OsString::from("/\\server/share/output.json"),
        OsString::from("\\/server/share/output.json"),
        temp.0.join("missing-parent/record.json").into_os_string(),
        temp.0.clone().into_os_string(),
    ] {
        let mut invalid = baseline.clone();
        replace(&mut invalid, "--output", path);
        early_failure(&invoke(&invalid));
    }
    assert!(!temp.0.join("missing-parent").exists());
    #[cfg(windows)]
    {
        let mut invalid = baseline.clone();
        replace(&mut invalid, "--output", "C:drive-relative-output.json");
        early_failure(&invoke(&invalid));
    }
    let checkpoint_directory = temp.0.join("checkpoint-directory");
    fs::create_dir(&checkpoint_directory).unwrap();
    let mut invalid = baseline.clone();
    replace(&mut invalid, "--checkpoint", &checkpoint_directory);
    early_failure(&invoke(&invalid));

    // Regular but deliberately qualification-invalid inputs let preflight prove
    // byte/hash refusals happen before any dataset replay or model inference.
    fs::write(temp.0.join("missing-checkpoint.json"), b"{}").unwrap();
    fs::create_dir(temp.0.join("missing-dataset")).unwrap();
    // Metadata rejects an oversized regular file, without a 96 MiB read or NN qualification.
    assert_eq!(MAX_RECORD_BYTES, 96 * 1024 * 1024);
    let oversized = temp.0.join("oversized.json");
    File::options()
        .create_new(true)
        .write(true)
        .open(&oversized)
        .unwrap()
        .set_len(MAX_RECORD_BYTES as u64 + 1)
        .unwrap();
    let r = early_failure(&invoke(&audit_args(&temp, &oversized)));
    assert_eq!(r["phase"], "paths");
    assert!(r["timings"]["prepareSeconds"].is_null());
    assert!(r["error"].as_str().unwrap().contains("96 MiB"), "{r}");
    early_failure(&invoke(&audit_args(&temp, &temp.0)));
    early_failure(&invoke(&audit_args(
        &temp,
        &temp.0.join("missing-input.json"),
    )));

    let input = temp.0.join("small.json");
    fs::write(&input, b"{}").unwrap();
    for hash in ["A".repeat(64), "0".repeat(63), "g".repeat(64)] {
        let mut invalid = audit_args(&temp, &input);
        flag(&mut invalid, "--expect-sha256", hash);
        early_failure(&invoke(&invalid));
    }
    for extra in [
        args(&["--output", "not-an-audit-option"]),
        args(&["--sampling-seed", "7"]),
    ] {
        let mut invalid = audit_args(&temp, &input);
        invalid.extend(extra);
        early_failure(&invoke(&invalid));
    }
    let mut wrong_hash = audit_args(&temp, &input);
    flag(&mut wrong_hash, "--expect-sha256", "0".repeat(64));
    let r = early_failure(&invoke(&wrong_hash));
    assert!(r["error"].as_str().unwrap().contains("SHA"), "{r}");
    assert_eq!(fs::read(&input).unwrap(), b"{}");
    assert_eq!(fs::read(&existing).unwrap(), original);
    assert!(!temp.0.join("record.json").exists());
}

#[cfg(unix)]
#[test]
fn symlink_input_and_output_hierarchies_refuse_before_qualification() {
    let temp = Temp::new();
    let target = temp.0.join("target");
    fs::create_dir(&target).unwrap();
    let link = temp.0.join("linked");
    std::os::unix::fs::symlink(&target, &link).unwrap();
    let mut invalid = collect_args(&temp);
    replace(&mut invalid, "--output", link.join("record.json"));
    early_failure(&invoke(&invalid));
    fs::write(target.join("input.json"), b"{}").unwrap();
    early_failure(&invoke(&audit_args(&temp, &link.join("input.json"))));
    assert!(!target.join("record.json").exists());
}

#[test]
fn legitimate_checkpoint_collects_three_four_player_prefixes_and_audits_separately() {
    let temp = Temp::new();
    let mut sources = Vec::new();
    let mut fixture_seeds = Vec::new();
    // Exactly the established public_native/policy_training fixture selection;
    // these are mechanical fixture games, never research/strength/Test families.
    for split in [DatasetSplit::Train, DatasetSplit::Validation] {
        let seed = (0..1000)
            .find(|seed| {
                dataset::split_for_family(&dataset::seed_family_id(*seed)).unwrap() == split
            })
            .unwrap();
        fixture_seeds.push(seed);
        let record = replay::play_game_fast(3, seed, GameOptions::default(), true)
            .unwrap()
            .2
            .unwrap();
        let source = temp.0.join(format!("source-{}.json", sources.len()));
        fs::write(&source, serde_json::to_vec(&record).unwrap()).unwrap();
        sources.push(source);
    }
    eprintln!(
        "stochastic CLI fixture: native teachers=2, seeds={fixture_seeds:?}; BC epochs=1; collection environments=[17,11235]; no strength experiment"
    );
    let dataset_path = temp.0.join("dataset");
    export_native_files(&sources, &dataset_path).unwrap();
    let dataset = load_policy_dataset(&dataset_path).unwrap();
    let outcome = train_dataset(
        &dataset,
        &PolicyBcConfig {
            epochs: 1,
            batch_size: 128,
            learning_rate: 0.003,
            seed: 7,
        },
        None,
    )
    .unwrap();
    let trained = temp.0.join("trained");
    outcome.save_new_directory(&trained).unwrap();
    let checkpoint = trained.join("checkpoint.json");
    let checkpoint_original = fs::read(&checkpoint).unwrap();
    let mut collected = Vec::new();
    for (players, environment) in [(3, 17), (4, 11235)] {
        let output_path = temp.0.join(format!("prefix-{players}.json"));
        let mut values = collect_args(&temp);
        replace(&mut values, "--checkpoint", &checkpoint);
        replace(&mut values, "--dataset", &dataset_path);
        replace(&mut values, "--output", &output_path);
        replace(&mut values, "--players", players.to_string());
        replace(&mut values, "--environment-seed", environment.to_string());
        for name in [
            "--sampling-seed",
            "--episode-ordinal",
            "--replicate-ordinal",
        ] {
            replace(&mut values, name, u64::MAX.to_string());
        }
        let r = report(&invoke(&values), 2);
        assert_eq!(r["operationSucceeded"], true);
        assert_eq!(r["collectionStarted"], true);
        assert_eq!(r["collectionComplete"], false);
        assert!(r["auditPassed"].is_null());
        assert_eq!(r["counts"]["observedCallbacks"], 2);
        assert_eq!(r["counts"]["acceptedSamples"], 2);
        assert_eq!(r["counts"]["appliedChoices"], 2);
        assert_eq!(r["failure"]["reason"], "callbackLimit");
        assert!(r["terminalStateKey"].is_null());
        assert_eq!(r["publication"]["published"], true);
        assert_eq!(r["publication"]["verified"], true);
        assert_eq!(r["provenance"]["kind"], "publicLearned");
        let raw = fs::read(&output_path).unwrap();
        assert!(raw.len() <= MAX_RECORD_BYTES);
        assert_eq!(r["record"]["bytes"], raw.len());
        assert_eq!(r["record"]["sha256"], digest(&raw));
        let record: Value = serde_json::from_slice(&raw).unwrap();
        assert_eq!(record["header"]["sourceKind"], "publicStochasticBc");
        assert_eq!(record["header"]["config"]["players"], players);
        assert_eq!(record["header"]["config"]["environmentSeed"], environment);
        for key in ["samplingSeed", "episodeOrdinal", "replicateOrdinal"] {
            assert_eq!(record["header"]["config"][key], "ffffffffffffffff");
        }
        assert!(record["terminal"].is_null());
        assert_eq!(record["callbacks"].as_array().unwrap().len(), 2);

        let mut audit = audit_args(&temp, &output_path);
        replace(&mut audit, "--checkpoint", &checkpoint);
        replace(&mut audit, "--dataset", &dataset_path);
        flag(&mut audit, "--expect-sha256", digest(&raw));
        let audited = report(&invoke(&audit), 0);
        assert_eq!(audited["operationSucceeded"], true);
        assert_eq!(audited["auditPassed"], true);
        assert_eq!(audited["collectionStarted"], false);
        assert_eq!(audited["collectionComplete"], false);
        assert_eq!(audited["counts"]["observedCallbacks"], 2);
        assert_eq!(audited["counts"]["appliedChoices"], 2);
        assert!(audited["counts"]["samplerAttempts"].is_null());
        assert_eq!(fs::read(&output_path).unwrap(), raw);
        collected.push((output_path, record));
    }

    // Syntactically valid source tampering must fail the independent native audit.
    let mut tampered = collected[0].1.clone();
    tampered["callbacks"][0]["stateBefore"] = Value::String("0000000000000000".into());
    let tampered_path = temp.0.join("tampered.json");
    fs::write(&tampered_path, serde_json::to_vec(&tampered).unwrap()).unwrap();
    let tampered_before = fs::read(&tampered_path).unwrap();
    let mut audit = audit_args(&temp, &tampered_path);
    replace(&mut audit, "--checkpoint", &checkpoint);
    replace(&mut audit, "--dataset", &dataset_path);
    let rejected = report(&invoke(&audit), 1);
    assert_eq!(rejected["auditPassed"], false);
    assert_eq!(rejected["operationSucceeded"], false);
    assert!(rejected["collectionComplete"].is_null());
    assert_eq!(fs::read(&tampered_path).unwrap(), tampered_before);

    // Dedicated raw records cannot be relabeled as any existing training dataset.
    let prefix = std::slice::from_ref(&collected[0].0);
    for (name, result) in [
        (
            "legacy-dataset",
            dataset::export_dataset_files(prefix, &temp.0.join("legacy-dataset")).map(|_| ()),
        ),
        (
            "policy-dataset",
            export_native_files(prefix, &temp.0.join("policy-dataset")).map(|_| ()),
        ),
        (
            "state-dataset",
            tzolkin_ai::state_mc_dataset::export_native_files(
                prefix,
                &temp.0.join("state-dataset"),
            )
            .map(|_| ()),
        ),
    ] {
        assert!(
            result.is_err(),
            "{name} unexpectedly admitted stochastic raw"
        );
        assert!(!temp.0.join(name).exists());
    }
    assert_eq!(fs::read(&checkpoint).unwrap(), checkpoint_original);
}
