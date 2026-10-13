//! Produce a conditional Human policy dataset; no model or training commands.
use std::collections::BTreeMap;
use std::path::Path;
use tzolkin_ai::human_policy_dataset::ConditionalHumanPolicyDataset;

fn run() -> Result<(), String> {
    let mut args = std::env::args().skip(1);
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
    let manifest = required("--manifest")?;
    let manifest_sha = required("--manifest-sha256")?;
    let nominations_sha = required("--nominations-sha256")?;
    let reservations_sha = required("--reservations-sha256")?;
    let output = required("--output")?;
    let semantic = match flags.remove("--feature-profile").as_deref() {
        None => false,
        Some("semantic-playing4") => true,
        Some(_) => return Err("Unknown conditional producer feature profile".into()),
    };
    if !flags.is_empty() {
        return Err("Unexpected producer flag".into());
    }
    let load = if semantic {
        ConditionalHumanPolicyDataset::load_semantic_playing
    } else {
        ConditionalHumanPolicyDataset::load
    };
    let dataset = load(
        Path::new(&manifest),
        &manifest_sha,
        &nominations_sha,
        &reservations_sha,
    )?;
    dataset.publish_new(Path::new(&output))?;
    println!(
        "{}",
        serde_json::to_string(dataset.counts()).map_err(|e| e.to_string())?
    );
    Ok(())
}
fn main() {
    if let Err(error) = run() {
        eprintln!("{error}");
        std::process::exit(1);
    }
}
