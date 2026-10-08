use std::io::Write;
use std::process::{Command, Output, Stdio};
use tzolkin_ai::{arena, replay, search, search_native};
use tzolkin_core::{GameOptions, Phase, observation::observe};

struct Temporary(std::path::PathBuf);
impl Temporary {
    fn new() -> Self {
        let path = std::env::temp_dir().join(format!(
            "tzolkin-search-cli-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir(&path).unwrap();
        Self(path)
    }
    fn config(&self, cfg: &search::SearchConfig) -> String {
        let path = self.0.join("search.json");
        std::fs::write(&path, serde_json::to_vec(cfg).unwrap()).unwrap();
        path.to_str().unwrap().into()
    }
}
impl Drop for Temporary {
    fn drop(&mut self) {
        std::fs::remove_dir_all(&self.0).unwrap();
    }
}
fn cli(args: &[&str], input: Option<&[u8]>) -> Output {
    let mut child = Command::new(env!("CARGO_BIN_EXE_tzolkin-ai"))
        .args(args)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    if let Some(input) = input {
        child.stdin.take().unwrap().write_all(input).unwrap();
    } else {
        drop(child.stdin.take());
    }
    child.wait_with_output().unwrap()
}
fn cheap() -> search::SearchConfig {
    search::SearchConfig {
        worlds_per_action: 1,
        min_completed_worlds: 1,
        horizon_days: 1,
        ..search::SearchConfig::default()
    }
}
fn capped() -> search::SearchConfig {
    search::SearchConfig {
        worlds_per_action: 2,
        min_completed_worlds: 2,
        max_rollout_steps: 1,
        max_total_steps: 1,
        ..cheap()
    }
}

#[test]
fn choose_search_returns_direct_outcome_and_plain_choose_retains_decision_shape() {
    let temporary = Temporary::new();
    let config = temporary.config(&cheap());
    for players in [3, 4] {
        let mut state = tzolkin_core::create_game(
            (0..players).map(|p| format!("CPU {p}")).collect(),
            11235,
            false,
        )
        .unwrap();
        while state.phase == Phase::Setup {
            let observation = observe(&state, state.current_player).unwrap();
            state = tzolkin_core::apply_move(
                &state,
                tzolkin_ai::choose_move(&observation).unwrap().r#move,
            )
            .unwrap();
        }
        let observation = observe(&state, state.current_player).unwrap();
        let input = serde_json::to_vec(&observation).unwrap();
        let result = cli(&["choose", "--search-config", &config], Some(&input));
        assert!(
            result.status.success(),
            "{}",
            String::from_utf8_lossy(&result.stderr)
        );
        let actual: search::SearchOutcome = serde_json::from_slice(&result.stdout).unwrap();
        assert_eq!(actual, search::choose_move(&observation, &cheap()).unwrap());
        let plain = cli(&["choose"], Some(&input));
        let actual: tzolkin_ai::Decision = serde_json::from_slice(&plain.stdout).unwrap();
        assert_eq!(actual, tzolkin_ai::choose_move(&observation).unwrap());
    }
}

#[test]
fn strict_search_cli_rejects_unknown_conflicting_oversized_and_nonbase_inputs() {
    let temporary = Temporary::new();
    let config = temporary.config(&cheap());
    let cases = [
        vec!["choose", "--search-config", &config, "--kernel", "auto"],
        vec![
            "choose",
            "--search-config",
            &config,
            "--model",
            "missing.json",
        ],
        vec![
            "choose",
            "--search-config",
            &config,
            "--search-config",
            &config,
        ],
        vec!["choose", "--search-config", &config, "--unknown", "1"],
        vec!["selfplay", "--search-seat", "0"],
        vec!["selfplay", "--search-config", &config, "--players", "2"],
        vec!["selfplay", "--search-config", &config, "--players", "5"],
        vec![
            "selfplay",
            "--search-config",
            &config,
            "--players",
            "3",
            "--flags",
            "1",
        ],
        vec![
            "selfplay",
            "--search-config",
            &config,
            "--players",
            "3",
            "--search-seat",
            "3",
        ],
        vec![
            "selfplay",
            "--search-config",
            &config,
            "--model-seat",
            "all",
        ],
        vec!["selfplay-batch", "--search-config", &config],
    ];
    for args in cases {
        assert!(!cli(&args, None).status.success(), "{args:?}");
    }
    let mut value = serde_json::to_value(cheap()).unwrap();
    value["hiddenSeed"] = serde_json::json!(1);
    std::fs::write(&config, serde_json::to_vec(&value).unwrap()).unwrap();
    assert!(
        !cli(&["choose", "--search-config", &config], None)
            .status
            .success()
    );
    let mut bad = cheap();
    bad.worlds_per_action = 33;
    temporary.config(&bad);
    assert!(
        !cli(&["choose", "--search-config", &config], None)
            .status
            .success()
    );
    std::fs::write(&config, vec![b' '; 64 * 1024 + 1]).unwrap();
    let result = cli(&["choose", "--search-config", &config], None);
    assert!(String::from_utf8_lossy(&result.stderr).contains("64 KiB"));
}

#[test]
fn selfplay_search_cli_completes_three_four_players_and_publishes_verified_source() {
    let temporary = Temporary::new();
    let config = temporary.config(&capped());
    for players in [3, 4] {
        let path = temporary.0.join(format!("game-{players}.json"));
        let args = [
            "selfplay",
            "--players",
            if players == 3 { "3" } else { "4" },
            "--seed",
            "11235",
            "--flags",
            "0",
            "--search-config",
            &config,
            "--search-seat",
            "all",
            "--fast",
            "--output",
            path.to_str().unwrap(),
        ];
        let result = cli(&args, None);
        assert!(
            result.status.success(),
            "{}",
            String::from_utf8_lossy(&result.stderr)
        );
        let report: serde_json::Value = serde_json::from_slice(&result.stdout).unwrap();
        assert_eq!(report["searchPolicy"]["kind"], "search");
        let actual: replay::GameReplay =
            serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
        replay::verify_replay(&actual).unwrap();
        let direct = search_native::play_game(
            players,
            11235,
            GameOptions::default(),
            &(0..players).collect::<Vec<_>>(),
            true,
            true,
            &search::PreparedSearch::new(&capped()).unwrap(),
        )
        .unwrap();
        assert_eq!(
            report["search"],
            serde_json::to_value(&direct.search).unwrap()
        );
        assert_eq!(
            serde_json::to_vec(&actual).unwrap(),
            serde_json::to_vec(&direct.game.unwrap().2.unwrap()).unwrap()
        );
        let old = std::fs::read(&path).unwrap();
        assert!(!cli(&args, None).status.success());
        assert_eq!(old, std::fs::read(&path).unwrap());
        assert!(
            actual
                .steps
                .iter()
                .all(|step| step.search.as_ref().unwrap().policy_version
                    == search::SEARCH_POLICY_VERSION)
        );
    }
}

#[test]
fn selfplay_publication_failure_keeps_completed_search_measurements_and_nonzero_exit() {
    let temporary = Temporary::new();
    let config = temporary.config(&capped());
    let blocked = temporary.0.join("blocked");
    std::fs::write(&blocked, b"existing file").unwrap();
    let destination = blocked.join("game.json");
    let result = cli(
        &[
            "selfplay",
            "--players",
            "3",
            "--seed",
            "11235",
            "--search-config",
            &config,
            "--fast",
            "--output",
            destination.to_str().unwrap(),
        ],
        None,
    );
    assert!(!result.status.success());
    let report: serde_json::Value = serde_json::from_slice(&result.stdout).unwrap();
    assert_eq!(report["publication"]["success"], false);
    assert!(report["decisions"].as_u64().unwrap() > 0);
    assert_eq!(report["finalScores"].as_array().unwrap().len(), 3);
    assert!(report["search"][0]["atomicSteps"].as_u64().unwrap() > 0);
    assert_eq!(std::fs::read(blocked).unwrap(), b"existing file");
    assert!(!destination.exists());
}

#[test]
fn arena_cli_accepts_search_for_both_player_counts_and_reports_all_absolute_seats() {
    let temporary = Temporary::new();
    for players in [3, 4] {
        let cfg = arena::ArenaConfig {
            schema: 1,
            players,
            partition: arena::Partition::Pilot,
            seeds: arena::partition_seeds(arena::Partition::Pilot, 0, 1).unwrap(),
            candidate: arena::PolicyConfig::Search { config: capped() },
            reference: arena::PolicyConfig::default(),
            opponent_pool: vec![arena::PolicyConfig::default(); players],
            bootstrap_seed: 1,
        };
        let path = temporary.0.join(format!("arena-{players}.json"));
        std::fs::write(&path, serde_json::to_vec(&cfg).unwrap()).unwrap();
        let result = cli(&["arena", "--config", path.to_str().unwrap()], None);
        assert!(
            result.status.success(),
            "{}",
            String::from_utf8_lossy(&result.stderr)
        );
        let report: serde_json::Value = serde_json::from_slice(&result.stdout).unwrap();
        assert_eq!(report["statistics"]["failedGames"], 0);
        assert_eq!(report["statistics"]["completedGames"], players * 2);
        assert_eq!(report["candidate"]["provenance"]["kind"], "search");
        for summary in report["search"].as_array().unwrap() {
            assert!(summary["atomicSteps"].as_u64().unwrap() > 0);
            assert_eq!(summary["searched"], 0);
            assert!(
                summary["fallbackReasons"]["insufficientCompletedWorlds"]
                    .as_u64()
                    .unwrap()
                    > 0
            );
        }
    }
}
