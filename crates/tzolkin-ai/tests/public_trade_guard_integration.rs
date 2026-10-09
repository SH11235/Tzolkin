//! G2 correctness fixtures: untrained synthetic models, known seed17 only.
//! Mechanical provenance is format-valid; these are not qualified BC producers.
use sha2::{Digest, Sha256};
use std::fs;
use std::path::PathBuf;
use std::process::Command;
use std::sync::{
    OnceLock,
    atomic::{AtomicU64, Ordering},
};
use tzolkin_ai::arena::{ArenaConfig, PolicyConfig};
use tzolkin_ai::public_model::{LoadedPublicPolicy, PublicPolicyArtifact};
use tzolkin_ai::public_trade_guard::{
    self, PublicLearnedSource, TradeGuardConfig, TradeGuardSession,
};
use tzolkin_ai::replay::{self, GameReplay, ReplaySource, SeatPolicy};
use tzolkin_ai::search::{PreparedSearch, SearchConfig};
use tzolkin_core::observation::TypedAction;
use tzolkin_core::{GameOptions, Task};

struct Temp(PathBuf);
impl Temp {
    fn new() -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        static PROCESS_NONCE: OnceLock<u128> = OnceLock::new();
        let nonce = PROCESS_NONCE.get_or_init(|| {
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        });
        let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let root = manifest
            .parent()
            .unwrap()
            .parent()
            .unwrap()
            .join("implementation.local/g2/fixtures");
        fs::create_dir_all(&root).unwrap();
        let path = root.join(format!(
            "{}-{}-{}",
            std::process::id(),
            nonce,
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir(&path).unwrap();
        Self(path)
    }
}
fn model() -> PublicPolicyArtifact {
    let mut wire = serde_json::to_value(PublicPolicyArtifact::new(17).unwrap()).unwrap();
    wire["model"]["parameters"]
        .as_array_mut()
        .unwrap()
        .fill(0.0.into());
    let mut model: PublicPolicyArtifact = serde_json::from_value(wire).unwrap();
    model.checksum.clear();
    model.checksum = format!("{:x}", Sha256::digest(serde_json::to_vec(&model).unwrap()));
    model.validate().unwrap();
    model
}
fn provenance(model: &PublicPolicyArtifact, guard: &TradeGuardConfig) -> SeatPolicy {
    let pure = SeatPolicy::PublicLearned {
        policy_version: model.policy_version.clone(),
        model_version: model.model_version.clone(),
        training_version: tzolkin_ai::policy_training::TRAINING_VERSION.into(),
        feature_schema: 2,
        input_contract: model.input_contract.clone(),
        task: "policyOnlyBc".into(),
        value_validity: model.value_validity,
        model_checksum: model.checksum.clone(),
        training_checkpoint_checksum: "b".repeat(64),
        dataset_fingerprint: "c".repeat(64),
        inference_backend: "scalar".into(),
    };
    SeatPolicy::PublicLearnedTradeGuard {
        policy_version: public_trade_guard::POLICY_VERSION.into(),
        base: PublicLearnedSource::from_pure(&pure).unwrap(),
        guard: guard.clone(),
        configuration_key: guard.configuration_key().unwrap(),
    }
}
fn mixed_record() -> &'static GameReplay {
    static RECORD: OnceLock<GameReplay> = OnceLock::new();
    RECORD.get_or_init(|| {
        let model = model();
        let loaded = LoadedPublicPolicy::new(&model).unwrap();
        let guard = TradeGuardConfig {
            max_trades_per_episode: 1,
            ..Default::default()
        };
        let search = PreparedSearch::new(&SearchConfig {
            worlds_per_action: 1,
            horizon_days: 1,
            max_rollout_steps: 1,
            max_total_steps: 1,
            min_completed_worlds: 1,
            sampling_salt: 17,
        })
        .unwrap();
        let policies = vec![
            provenance(&model, &guard),
            tzolkin_ai::search_native::provenance(&search),
            provenance(&model, &guard),
        ];
        let mut sessions = [
            Some(TradeGuardSession::new(guard.clone()).unwrap()),
            None,
            Some(TradeGuardSession::new(guard).unwrap()),
        ];
        let mut seen_market = [false; 3];
        let mut indices = Vec::new();
        let (_, count, record) = replay::play_game_using_all_diagnostics(
            3,
            17,
            Default::default(),
            true,
            ReplaySource::PolicySelfPlay { policies },
            true,
            |index, o| {
                indices.push(index);
                for session in sessions.iter_mut().flatten() {
                    session.on_callback(index, o)?;
                }
                if o.actor == 1 {
                    let (d, t) = tzolkin_ai::search_native::decide(&search, o)?;
                    return Ok((d, t, None));
                }
                if o.pending_task == Some(Task::Trade) {
                    seen_market[o.actor] = true;
                    let (d, t) = sessions[o.actor]
                        .as_mut()
                        .unwrap()
                        .choose_loaded(index, o, &loaded)?;
                    return Ok((d, None, t));
                }
                let mut d = tzolkin_ai::choose_move(o)?;
                // Public-only, synthetic mechanical route; not an assertion of
                // NN generation outside Trade or checkpoint qualification.
                if !seen_market[o.actor] {
                    let preferred = o
                        .legal_actions
                        .iter()
                        .find(|a| {
                            matches!(
                                a.action,
                                TypedAction::UseAction {
                                    gear: tzolkin_core::GearId::Uxmal,
                                    position: 2,
                                    ..
                                }
                            )
                        })
                        .or_else(|| {
                            o.legal_actions.iter().find(|a| {
                                matches!(
                                    a.action,
                                    TypedAction::Remove {
                                        gear: tzolkin_core::GearId::Uxmal,
                                        position: 2
                                    }
                                )
                            })
                        })
                        .or_else(|| {
                            o.legal_actions.iter().find(|a| {
                                matches!(
                                    a.action,
                                    TypedAction::Place {
                                        gear: tzolkin_core::GearId::Uxmal,
                                        ..
                                    }
                                )
                            })
                        });
                    if let Some(a) = preferred {
                        d.r#move = a.r#move.clone();
                    }
                }
                d.policy_version = public_trade_guard::POLICY_VERSION.into();
                Ok((d, None, None))
            },
        )
        .unwrap();
        println!("G2 mixed indexed mechanical fixture seed17,3p callbacks{count}");
        assert_eq!(indices, (0..count).collect::<Vec<_>>());
        assert!(seen_market[0] && seen_market[2]);
        let record = record.unwrap();
        replay::verify_replay(&record).unwrap();
        record
    })
}

#[test]
fn mixed_global_runner_verifies_both_diagnostics_and_rejects_missing_spurious_or_changed_trace() {
    let record = mixed_record();
    let search = record
        .steps
        .iter()
        .position(|step| step.search.is_some())
        .unwrap();
    let guard = record
        .steps
        .iter()
        .position(|step| step.trade_guard.is_some())
        .unwrap();
    for mutation in 0..5 {
        let mut bad = record.clone();
        match mutation {
            0 => bad.steps[search].search = None,
            1 => bad.steps[guard].trade_guard = None,
            2 => {
                bad.steps[guard]
                    .trade_guard
                    .as_mut()
                    .unwrap()
                    .callback_index += 1
            }
            3 => bad.steps[search].trade_guard = bad.steps[guard].trade_guard.clone(),
            _ => {
                let ReplaySource::PolicySelfPlay { policies } = &mut bad.header.source else {
                    unreachable!()
                };
                let SeatPolicy::PublicLearnedTradeGuard {
                    configuration_key, ..
                } = &mut policies[0]
                else {
                    unreachable!()
                };
                *configuration_key = "a".repeat(64);
            }
        }
        assert!(replay::verify_replay(&bad).is_err());
    }
}

#[test]
fn new_indexed_adapter_preserves_old_pure_replay_bytes() {
    let (_, count, old) = replay::play_game_fast(3, 17, GameOptions::default(), true).unwrap();
    let mut indices = Vec::new();
    let (_, new_count, new) = replay::play_game_using_all_diagnostics(
        3,
        17,
        Default::default(),
        true,
        old.as_ref().unwrap().header.source.clone(),
        true,
        |index, o| {
            indices.push(index);
            Ok((tzolkin_ai::choose_move(o)?, None, None))
        },
    )
    .unwrap();
    println!("G2 pure byte-parity fixture seed17,3p two trajectories callbacks{count}");
    assert_eq!(count, new_count);
    assert_eq!(indices, (0..count).collect::<Vec<_>>());
    assert_eq!(
        serde_json::to_vec(&old).unwrap(),
        serde_json::to_vec(&new).unwrap()
    );
    assert!(!serde_json::to_string(&new).unwrap().contains("tradeGuard"));
}

#[test]
fn guarded_complete_source_remains_rejected_by_a2_and_a6_before_output_creation() {
    let temp = Temp::new();
    let source = temp.0.join("mixed.json");
    fs::write(&source, serde_json::to_vec(mixed_record()).unwrap()).unwrap();
    for state in [false, true] {
        let out = temp.0.join(if state {
            "state-dataset"
        } else {
            "policy-dataset"
        });
        let result = if state {
            tzolkin_ai::state_mc_dataset::export_native_files(std::slice::from_ref(&source), &out)
                .map(|_| ())
        } else {
            tzolkin_ai::policy_dataset::export_native_files(std::slice::from_ref(&source), &out)
                .map(|_| ())
        };
        assert!(result.is_err());
        assert!(!out.exists());
    }
}

#[test]
fn closed_arena_guard_kind_requires_explicit_valid_config_and_does_not_change_pure_kind() {
    let guard = serde_json::to_value(TradeGuardConfig::default()).unwrap();
    let wire = serde_json::json!({"kind":"publicLearnedTradeGuard","checkpoint":"x","dataset":"y","guard":guard});
    let config: PolicyConfig = serde_json::from_value(wire.clone()).unwrap();
    let PolicyConfig::PublicLearnedTradeGuard { guard, kernel, .. } = config else {
        unreachable!()
    };
    guard.validate().unwrap();
    assert_eq!(kernel, "scalar");
    for mutation in 0..4 {
        let mut bad = wire.clone();
        match mutation {
            0 => {
                bad.as_object_mut().unwrap().remove("guard");
            }
            1 => bad["guard"] = serde_json::Value::Null,
            2 => bad["guard"]["unknown"] = true.into(),
            _ => bad["unknown"] = true.into(),
        }
        assert!(serde_json::from_value::<PolicyConfig>(bad).is_err());
    }
    let pure: PolicyConfig = serde_json::from_value(
        serde_json::json!({"kind":"publicLearned","checkpoint":"x","dataset":"y"}),
    )
    .unwrap();
    assert!(serde_json::to_value(pure).unwrap().get("guard").is_none());
    let mut arena = serde_json::json!({"schema":1,"players":3,"partition":"pilot","seeds":[17],"candidate":wire,
        "reference":{"kind":"heuristic","weights":tzolkin_ai::policy::HeuristicWeights::default()},
        "opponentPool":vec![serde_json::json!({"kind":"heuristic","weights":tzolkin_ai::policy::HeuristicWeights::default()});3],"bootstrapSeed":17});
    let parsed: ArenaConfig = serde_json::from_value(arena.clone()).unwrap();
    assert!(
        tzolkin_ai::arena::run_arena(&parsed, &PathBuf::from("."))
            .unwrap_err()
            .contains("held-out test")
    );
    arena["candidate"]["guard"]["maxTradesPerEpisode"] = 0.into();
    let parsed: ArenaConfig = serde_json::from_value(arena).unwrap();
    assert!(
        tzolkin_ai::arena::run_arena(&parsed, &PathBuf::from("."))
            .unwrap_err()
            .contains("configuration")
    );
}

#[test]
fn local_config_loader_is_closed_bounded_and_rejects_network_directory_and_symlinks() {
    let temp = Temp::new();
    let file = temp.0.join("guard.json");
    fs::write(
        &file,
        serde_json::to_vec(&TradeGuardConfig::default()).unwrap(),
    )
    .unwrap();
    assert_eq!(
        public_trade_guard::load_config(&file).unwrap(),
        TradeGuardConfig::default()
    );
    let mut at_limit = serde_json::to_vec(&TradeGuardConfig::default()).unwrap();
    at_limit.resize(public_trade_guard::MAX_CONFIG_BYTES, b' ');
    fs::write(&file, at_limit).unwrap();
    assert_eq!(
        public_trade_guard::load_config(&file).unwrap(),
        TradeGuardConfig::default()
    );
    for wire in [
        b"{}".as_slice(),
        b"null",
        b"{\"schema\":1,\"schema\":1}",
        b"garbage",
    ] {
        fs::write(&file, wire).unwrap();
        assert!(public_trade_guard::load_config(&file).is_err());
    }
    for count in [0, 65] {
        let c = TradeGuardConfig {
            max_trades_per_episode: count,
            ..Default::default()
        };
        fs::write(&file, serde_json::to_vec(&c).unwrap()).unwrap();
        assert!(public_trade_guard::load_config(&file).is_err());
    }
    fs::write(&file, vec![b' '; public_trade_guard::MAX_CONFIG_BYTES + 1]).unwrap();
    assert!(
        public_trade_guard::load_config(&file)
            .unwrap_err()
            .contains("64 KiB")
    );
    assert!(public_trade_guard::load_config(&temp.0).is_err());
    for path in [
        "https://example.invalid/config",
        "//server/share/config",
        "\\\\server\\share\\config",
        "/\\server\\share\\config",
        "\\/server/share/config",
        "/\\.\\device",
        "\\/.\\device",
        "\\\\.\\device",
        "\\\\?\\C:\\guard.json",
        "../guard.json",
    ] {
        assert!(
            public_trade_guard::load_config(&PathBuf::from(path))
                .unwrap_err()
                .contains("local regular file"),
            "path must fail before hierarchy I/O: {path}"
        );
    }
    let link = temp.0.join("link.json");
    #[cfg(windows)]
    let linked = std::os::windows::fs::symlink_file(&file, &link).is_ok();
    #[cfg(unix)]
    let linked = std::os::unix::fs::symlink(&file, &link).is_ok();
    #[cfg(not(any(windows, unix)))]
    let linked = false;
    if linked {
        assert!(
            public_trade_guard::load_config(&link)
                .unwrap_err()
                .contains("symlink/junction")
        );
    } else {
        println!("G2 symlink runtime case unavailable on this host; static platform check remains");
    }
    #[cfg(windows)]
    {
        let target = temp.0.join("junction-target");
        fs::create_dir(&target).unwrap();
        fs::write(
            target.join("guard.json"),
            serde_json::to_vec(&TradeGuardConfig::default()).unwrap(),
        )
        .unwrap();
        let junction = temp.0.join("junction");
        // Native Windows creation only, no deletion or shell-built path list.
        let made = Command::new("cmd.exe")
            .args(["/C", "mklink", "/J"])
            // mklink treats slash-separated Cargo manifest paths as switches.
            .arg(junction.to_string_lossy().replace('/', "\\"))
            .arg(target.to_string_lossy().replace('/', "\\"))
            .output()
            .unwrap();
        fs::write(temp.0.join("junction-create.stdout"), &made.stdout).unwrap();
        fs::write(temp.0.join("junction-create.stderr"), &made.stderr).unwrap();
        fs::write(
            temp.0.join("junction-create-status.txt"),
            made.status.to_string(),
        )
        .unwrap();
        if made.status.success() {
            assert!(
                public_trade_guard::load_config(&junction.join("guard.json"))
                    .unwrap_err()
                    .contains("symlink/junction")
            );
            println!("G2 Windows junction hierarchy runtime rejected");
        } else {
            println!("G2 Windows junction creation unavailable; no runtime coverage claimed");
        }
    }
}

#[cfg(unix)]
#[test]
fn fifo_config_fails_before_open_with_bounded_child_deadline() {
    use std::process::Stdio;
    use std::time::{Duration, Instant};
    let temp = Temp::new();
    let fifo = temp.0.join("guard.fifo");
    let made = Command::new("mkfifo").arg(&fifo).output().unwrap();
    fs::write(temp.0.join("fifo-create.stdout"), &made.stdout).unwrap();
    fs::write(temp.0.join("fifo-create.stderr"), &made.stderr).unwrap();
    assert!(made.status.success(), "scoped FIFO fixture must be created");
    let mut child = Command::new(env!("CARGO_BIN_EXE_tzolkin-public-ml"))
        .args([
            "selfplay",
            "--checkpoint",
            "absent-checkpoint",
            "--dataset",
            "absent-dataset",
            "--seats",
            "all",
            "--trade-guard-config",
        ])
        .arg(&fifo)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    let start = Instant::now();
    let deadline = Duration::from_secs(5);
    let timed_out = loop {
        if child.try_wait().unwrap().is_some() {
            break false;
        }
        if start.elapsed() >= deadline {
            child.kill().unwrap();
            break true;
        }
        std::thread::sleep(Duration::from_millis(10));
    };
    let result = child.wait_with_output().unwrap();
    fs::write(temp.0.join("fifo-cli.stdout"), &result.stdout).unwrap();
    fs::write(temp.0.join("fifo-cli.stderr"), &result.stderr).unwrap();
    fs::write(
        temp.0.join("fifo-cli-status.txt"),
        format!("timedOut={timed_out}\nexit={}\n", result.status),
    )
    .unwrap();
    assert!(
        !timed_out,
        "FIFO config reader exceeded bounded child deadline"
    );
    assert!(!result.status.success());
    assert!(result.stdout.is_empty());
    assert!(String::from_utf8_lossy(&result.stderr).contains("regular file"));
}

#[test]
fn dedicated_cli_guard_flag_is_selfplay_only_closed_and_preflight_rejects_bad_inputs() {
    let temp = Temp::new();
    let invalid = temp.0.join("invalid.json");
    fs::write(&invalid, b"{}").unwrap();
    let cli = env!("CARGO_BIN_EXE_tzolkin-public-ml");
    for command in [
        "choose",
        "train",
        "resume",
        "evaluate",
        "export-native",
        "export-state-native",
        "train-state",
        "resume-state",
        "evaluate-state",
    ] {
        let out = Command::new(cli)
            .args([command, "--trade-guard-config", invalid.to_str().unwrap()])
            .output()
            .unwrap();
        assert!(!out.status.success());
        assert!(out.stdout.is_empty());
        assert!(String::from_utf8_lossy(&out.stderr).contains("Unknown"));
        assert!(String::from_utf8_lossy(&out.stderr).contains("trade-guard-config"));
    }
    let common = ["selfplay", "--checkpoint", "absent", "--dataset", "absent"];
    for (extra, expected) in [
        (
            vec![
                "--seats",
                "all",
                "--trade-guard-config",
                invalid.to_str().unwrap(),
            ],
            "missing field",
        ),
        (vec!["--seats", "all", "--players", "5"], "base 3/4p"),
        (vec!["--seats", "all", "--flags", "1"], "flags 0 only"),
        (vec!["--seats", "0,0"], "distinct nonempty"),
        (
            vec![
                "--seats",
                "all",
                "--trade-guard-config",
                "x",
                "--trade-guard-config",
                "y",
            ],
            "Repeated flag",
        ),
    ] {
        let out = Command::new(cli).args(common).args(extra).output().unwrap();
        assert!(!out.status.success());
        assert!(out.stdout.is_empty());
        assert!(
            String::from_utf8_lossy(&out.stderr).contains(expected),
            "{}",
            String::from_utf8_lossy(&out.stderr)
        );
    }
    let out = Command::new(env!("CARGO_BIN_EXE_tzolkin-ai"))
        .args([
            "selfplay",
            "--trade-guard-config",
            invalid.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert!(!out.status.success());
    assert!(out.stdout.is_empty());
    assert!(String::from_utf8_lossy(&out.stderr).contains("Unknown"));
}
