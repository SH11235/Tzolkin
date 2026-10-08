use std::fs;
use std::process::Command;
use tzolkin_ai::{dataset::load_dataset, replay};
use tzolkin_core::GameOptions;

#[test]
fn dataset_cli_exports_native_replays_and_refuses_partial_or_existing_outputs() {
    let root = std::env::temp_dir().join(format!("tzolkin-dataset-cli-{}", std::process::id()));
    fs::create_dir(&root).unwrap();
    let source = root.join("source");
    fs::create_dir(&source).unwrap();
    let (_, decisions, native) = replay::play_game(3, 11235, GameOptions::default(), true).unwrap();
    let mut native = native.unwrap();
    fs::write(
        source.join("game.json"),
        serde_json::to_vec(&native).unwrap(),
    )
    .unwrap();
    let destination = root.join("dataset");
    let invoke = |path: &std::path::Path| {
        Command::new(env!("CARGO_BIN_EXE_tzolkin-ai"))
            .args(["dataset", "--input"])
            .arg(&source)
            .arg("--output")
            .arg(path)
            .output()
            .unwrap()
    };
    let result = invoke(&destination);
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    let report: serde_json::Value = serde_json::from_slice(&result.stdout).unwrap();
    assert_eq!(report["samples"], decisions);
    let dataset = load_dataset(&destination).unwrap();
    assert_eq!(dataset.manifest().games.len(), 1);
    assert_eq!(
        dataset.iter().collect::<Result<Vec<_>, _>>().unwrap().len(),
        decisions
    );
    let before = fs::read(destination.join("manifest.json")).unwrap();
    assert!(!invoke(&destination).status.success());
    assert_eq!(fs::read(destination.join("manifest.json")).unwrap(), before);
    native.verified_complete = false;
    fs::write(
        source.join("game.json"),
        serde_json::to_vec(&native).unwrap(),
    )
    .unwrap();
    let rejected = root.join("rejected");
    assert!(!invoke(&rejected).status.success());
    assert!(!rejected.exists());
    fs::remove_dir_all(root).unwrap();
}
