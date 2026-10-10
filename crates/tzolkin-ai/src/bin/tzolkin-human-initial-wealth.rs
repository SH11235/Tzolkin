use std::collections::BTreeMap;
use std::path::Path;
use tzolkin_ai::human_initial_wealth::{
    InitialChoiceInput, InitialTrainConfig, InitialWealthArtifact, ValidatedInitialDataset,
    publish_new, train,
};

fn run() -> Result<(), String> {
    let mut args = std::env::args().skip(1);
    let command = args.next().ok_or("Expected qualify, train or predict")?;
    let mut flags = BTreeMap::new();
    while let Some(flag) = args.next() {
        if !flag.starts_with("--") {
            return Err("Expected a named flag".into());
        }
        let value = args.next().ok_or("Missing flag value")?;
        if flags.insert(flag, value).is_some() {
            return Err("Duplicate flag".into());
        }
    }
    let mut required = |name: &str| flags.remove(name).ok_or_else(|| format!("Missing {name}"));
    if command == "predict" {
        let model = required("--model")?;
        let sha = required("--model-sha256")?;
        let offers = required("--offers")?
            .split(',')
            .map(str::to_owned)
            .collect::<Vec<_>>();
        if !flags.is_empty() {
            return Err("Unexpected prediction flag".into());
        }
        let input = InitialChoiceInput::new(
            offers
                .try_into()
                .map_err(|_| "Expected four comma-separated catalog IDs")?,
        )?;
        let artifact = InitialWealthArtifact::load(Path::new(&model), &sha)?;
        println!(
            "{}",
            serde_json::to_string(&artifact.predict(&input)?).map_err(|e| e.to_string())?
        );
        return Ok(());
    }
    if command != "qualify" && command != "train" {
        return Err("Expected qualify, train or predict".into());
    }
    let manifest = required("--manifest")?;
    let manifest_sha = required("--manifest-sha256")?;
    let atlas_sha = required("--atlas-sha256")?;
    let nominations_sha = required("--nominations-sha256")?;
    let output = required("--output")?;
    let config = if command == "train" {
        Some(InitialTrainConfig {
            epochs: required("--epochs")?
                .parse()
                .map_err(|_| "Invalid --epochs")?,
            learning_rate: required("--learning-rate")?
                .parse()
                .map_err(|_| "Invalid --learning-rate")?,
            max_weight_delta: required("--max-weight-delta")?
                .parse()
                .map_err(|_| "Invalid --max-weight-delta")?,
        })
    } else {
        None
    };
    if !flags.is_empty() {
        return Err("Unexpected dataset/training flag".into());
    }
    let dataset = ValidatedInitialDataset::load(
        Path::new(&manifest),
        &manifest_sha,
        &atlas_sha,
        &nominations_sha,
    )?;
    let receipt = if let Some(config) = config {
        let trained = train(&dataset, &config)?;
        publish_new(Path::new(&output), &dataset, Some(&trained))?
    } else {
        publish_new(Path::new(&output), &dataset, None)?
    };
    println!(
        "{}",
        serde_json::to_string(&receipt).map_err(|e| e.to_string())?
    );
    Ok(())
}
fn main() {
    if let Err(error) = run() {
        eprintln!("{error}");
        std::process::exit(1);
    }
}
