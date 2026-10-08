use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::path::{Path, PathBuf};
use std::process::{Command, Output};
use std::sync::atomic::{AtomicU64, Ordering};
use tzolkin_ai::{
    replay,
    search::{PreparedSearch, SearchConfig},
};
use tzolkin_core::GameOptions;

static SERIAL: AtomicU64 = AtomicU64::new(0);
struct Fixture(PathBuf);
impl Fixture {
    fn new() -> Self {
        let dir = std::env::temp_dir().join(format!(
            "tzolkin-search-bench-{}-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos(),
            SERIAL.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir(&dir).unwrap();
        Self(dir)
    }
    fn write(&self, name: &str, bytes: impl AsRef<[u8]>) -> PathBuf {
        let path = self.0.join(name);
        std::fs::write(&path, bytes).unwrap();
        path
    }
    fn config(&self) -> (PathBuf, SearchConfig) {
        let config = SearchConfig {
            worlds_per_action: 1,
            horizon_days: 1,
            max_rollout_steps: 256,
            max_total_steps: 4096,
            min_completed_worlds: 1,
            sampling_salt: 17,
        };
        (
            self.write("search.json", serde_json::to_vec(&config).unwrap()),
            config,
        )
    }
}
impl Drop for Fixture {
    fn drop(&mut self) {
        std::fs::remove_dir_all(&self.0).unwrap();
    }
}
fn invoke(path: &Path, args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_tzolkin-search-bench"))
        .arg("--config")
        .arg(path)
        .args(args)
        .output()
        .unwrap()
}
fn report(output: Output) -> Value {
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(output.stderr.is_empty());
    serde_json::from_slice(&output.stdout).unwrap()
}
fn sha(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}
fn assert_digest(value: &Value) {
    let value = value.as_str().unwrap();
    assert_eq!(value.len(), 64);
    assert!(
        value
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
    );
}
fn expect_error(path: &Path, args: &[&str], message: &str) {
    let output = invoke(path, args);
    assert!(!output.status.success());
    assert!(output.stdout.is_empty());
    let error = String::from_utf8_lossy(&output.stderr);
    assert!(error.contains(message), "{error}");
}

#[test]
fn base_three_and_four_player_smoke_has_exact_direct_parity_and_finite_structured_rounds() {
    let fixture = Fixture::new();
    let (path, config) = fixture.config();
    for players in [3, 4] {
        let value = report(invoke(
            &path,
            &[
                "--players",
                &players.to_string(),
                "--states",
                "3",
                "--iterations",
                "1",
                "--rounds",
                "3",
            ],
        ));
        assert_eq!(value["schema"], "tzolkin-search-bench-v1");
        assert_eq!(value["players"], players);
        assert_eq!(value["seed"], 11235);
        assert_eq!(
            value["configurationKey"],
            config.configuration_key().unwrap()
        );
        assert_eq!(
            value["searchConfig"],
            serde_json::to_value(&config).unwrap()
        );
        assert_eq!(value["rulesVersion"], replay::RULES_VERSION);
        assert_eq!(value["catalogHash"], replay::catalog_hash());
        assert_eq!(value["selection"]["selectedByStratum"], json!([1, 1, 1]));
        assert_eq!(value["strengthMeasured"], false);
        assert_eq!(value["speedImprovementDeclared"], false);
        assert_eq!(value["quietMachineAttested"], false);
        assert_eq!(
            value["compiler"]["runtimeCompilerAttestsBuildCompiler"],
            false
        );
        assert_eq!(value["measurement"]["stageTimesAreNotAdditive"], true);
        assert_eq!(value["measurement"]["latencyQuantilesCollected"], false);
        assert_eq!(value["measurement"]["allocationMeasurement"], Value::Null);
        assert_eq!(
            value["parity"]["allWarmupAndTimedSearchOutcomesExact"],
            true
        );
        assert_eq!(value["parity"]["exactSerializedSearchDiagnostics"], true);
        for key in [
            "sourceReplaySha256",
            "selectedCorpusSha256",
            "executableSha256",
        ] {
            assert_digest(&value[key]);
        }
        let (_, _, record) =
            replay::play_game_fast(players, 11235, GameOptions::default(), true).unwrap();
        let record = record.unwrap();
        assert_eq!(
            value["sourceReplaySha256"],
            sha(&serde_json::to_vec(&record).unwrap())
        );
        let prepared = PreparedSearch::new(&config).unwrap();
        let states = value["selection"]["states"].as_array().unwrap();
        let mut observations = Vec::new();
        let mut prior = None;
        for state in states {
            let index = state["replayIndex"].as_u64().unwrap() as usize;
            assert!(prior.is_none_or(|p| p < index));
            prior = Some(index);
            let observation = &record.steps[index].observation;
            let outcome = prepared.choose(observation).unwrap();
            assert_eq!(state["observationKey"], observation.observation_key);
            assert_eq!(state["round"], observation.round);
            assert_eq!(
                state["directOutcomeSha256"],
                sha(&serde_json::to_vec(&outcome).unwrap())
            );
            assert_eq!(
                state["directMove"],
                serde_json::to_value(outcome.decision.r#move).unwrap()
            );
            assert_eq!(
                state["directStatus"],
                serde_json::to_value(outcome.status).unwrap()
            );
            assert_eq!(state["rootStagesEligible"], state["stratum"] == "normal");
            observations.push(observation);
        }
        assert_eq!(
            value["selectedCorpusSha256"],
            sha(&serde_json::to_vec(&observations).unwrap())
        );
        let planned = &value["plannedWorkload"];
        assert_eq!(
            planned["maximumPreparedSearchAtomicAttempts"],
            3 * 9 * config.max_total_steps
        );
        assert_eq!(planned["microApplyAndCheckCalls"], 3 * (2 + 4 * 4));
        assert_eq!(
            planned["maximumCorpusApplyCalls"],
            2 * replay::MAX_DECISIONS
        );
        assert!(planned["combinedMaximum"].as_u64().unwrap() <= planned["limit"].as_u64().unwrap());
        let stages = value["stages"].as_array().unwrap();
        assert_eq!(stages.len(), 12);
        for stage in stages {
            let name = stage["name"].as_str().unwrap();
            let inputs = stage["inputs"].as_u64().unwrap();
            let is_all = matches!(
                name,
                "preparedChooseAll" | "heuristicChoose" | "observationKey"
            );
            assert_eq!(inputs, if is_all { 3 } else { 1 });
            assert_eq!(stage["warmupOutputsChecked"], inputs);
            assert_eq!(stage["timedOutputsChecked"], inputs * 3);
            let timing = &stage["timing"];
            assert_eq!(timing["operationsPerRound"], inputs);
            assert_eq!(timing["roundSeconds"].as_array().unwrap().len(), 3);
            for seconds in timing["roundSeconds"].as_array().unwrap() {
                assert!(seconds.as_f64().unwrap().is_finite() && seconds.as_f64().unwrap() > 0.0);
            }
            for key in [
                "medianRoundSeconds",
                "operationsPerSecond",
                "meanNsPerOperation",
            ] {
                let number = timing[key].as_f64().unwrap();
                assert!(number.is_finite() && number > 0.0);
            }
            for (key, expected) in [
                ("warmupSearchWork", inputs),
                ("timedSearchWork", inputs * 3),
            ] {
                let work = &stage[key];
                if name.starts_with("preparedChoose") {
                    assert_eq!(work["decisions"], expected);
                    assert_eq!(
                        work["searched"].as_u64().unwrap() + work["fallbacks"].as_u64().unwrap(),
                        expected
                    );
                    assert_eq!(
                        work["completedWorlds"].as_u64().unwrap()
                            + work["discardedWorlds"].as_u64().unwrap(),
                        work["attemptedWorlds"]
                    );
                    assert!(
                        work["atomicAttempts"].as_u64().unwrap()
                            <= expected * config.max_total_steps as u64
                    );
                    if name == "preparedChooseSetup" {
                        assert_eq!(work["fallbackReasons"]["setup"], expected);
                    }
                    if name == "preparedChoosePending" {
                        assert_eq!(
                            work["fallbackReasons"]["pendingContinuationUnavailable"],
                            expected
                        );
                    }
                } else {
                    assert!(work.is_null());
                }
            }
        }
    }
}

#[test]
fn flags_unknown_duplicate_missing_and_numeric_bounds_fail_before_output() {
    let fixture = Fixture::new();
    let (path, _) = fixture.config();
    for args in [
        vec!["--wat", "1"],
        vec!["--model", "model.json"],
        vec!["--players", "3", "--players", "4"],
        vec!["--states"],
        vec!["--seed", "4294967296"],
        vec!["--players", "2"],
        vec!["--players", "5"],
        vec!["--states", "0"],
        vec!["--states", "65"],
        vec!["--iterations", "0"],
        vec!["--iterations", "33"],
        vec!["--rounds", "2"],
        vec!["--rounds", "16"],
        vec!["--rounds", "--seed"],
    ] {
        let output = invoke(&path, &args);
        assert!(!output.status.success(), "{args:?}");
        assert!(output.stdout.is_empty());
    }
    let missing = Command::new(env!("CARGO_BIN_EXE_tzolkin-search-bench"))
        .args(["--players", "3"])
        .output()
        .unwrap();
    assert!(!missing.status.success());
    assert!(missing.stdout.is_empty());
    let help = Command::new(env!("CARGO_BIN_EXE_tzolkin-search-bench"))
        .arg("--help")
        .output()
        .unwrap();
    assert!(help.status.success());
    assert!(
        String::from_utf8_lossy(&help.stdout)
            .contains("Models and playing strength are not measured")
    );
}

#[test]
fn configuration_unknown_duplicate_invalid_oversize_and_planned_budget_are_rejected() {
    let fixture = Fixture::new();
    let (_, config) = fixture.config();
    let mut value = serde_json::to_value(&config).unwrap();
    value["trusted"] = json!(true);
    let path = fixture.write("unknown.json", serde_json::to_vec(&value).unwrap());
    expect_error(&path, &[], "unknown field");
    let text = serde_json::to_string(&config)
        .unwrap()
        .replace("{", "{\"samplingSalt\":0,");
    let path = fixture.write("duplicate.json", text);
    expect_error(&path, &[], "duplicate field");
    let invalid = SearchConfig {
        worlds_per_action: 0,
        ..config.clone()
    };
    let path = fixture.write("invalid.json", serde_json::to_vec(&invalid).unwrap());
    expect_error(&path, &[], "Invalid bounded search configuration");
    let path = fixture.write("oversize.json", vec![b' '; 64 * 1024 + 1]);
    expect_error(&path, &[], "exceeds64 KiB");
    let large = SearchConfig {
        max_total_steps: 262_144,
        ..config
    };
    let path = fixture.write("budget.json", serde_json::to_vec(&large).unwrap());
    expect_error(
        &path,
        &["--states", "64", "--iterations", "32", "--rounds", "15"],
        "workload exceeds20 million",
    );
}
