//! Numeric research bridge. Does not create a qualified BC/RL owner.
use std::fs::OpenOptions;
use std::io::Write;
use std::path::Path;

use serde::Serialize;
use tzolkin_ai::dataset::DatasetSplit;
use tzolkin_ai::features::{
    EncodedCandidate, FEATURE_COUNT, FeatureEncoder, PUBLIC_FEATURE_SCHEMA,
};
use tzolkin_ai::policy_dataset::load_policy_dataset;
use tzolkin_ai::public_model::{HIDDEN, PARAMETER_COUNT, PublicPolicyArtifact};
use tzolkin_ai::public_policy_pullback::ScalarPolicyPullback;
use tzolkin_core::observation::observe;

const MAX_ROWS: usize = 32_768;
const MAX_PADDED_ROWS: usize = 65_536;
const MAX_BATCH_BYTES: usize = 128 * 1024 * 1024;

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct Sample {
    id: String,
    split: &'static str,
    phase: String,
    chosen: usize,
    features: Vec<Vec<f32>>,
    logits: Vec<f32>,
    dlogits: Vec<f64>,
    gradient: Vec<f64>,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct Export {
    schema: &'static str,
    source_kind: &'static str,
    dataset_fingerprint: Option<String>,
    feature_schema: u32,
    feature_count: usize,
    hidden_count: usize,
    parameter_count: usize,
    parameters: Vec<f32>,
    samples: Vec<Sample>,
}

fn sample(
    model: &PublicPolicyArtifact,
    id: String,
    split: &'static str,
    phase: String,
    rows: &[EncodedCandidate],
    chosen: usize,
) -> Result<Sample, String> {
    let cache = ScalarPolicyPullback::new(&model.model, rows)?;
    let dlogits: Vec<_> = (0..rows.len())
        .map(|i| ((i % 7) as f64 - 3.0) / (7.0 * rows.len() as f64))
        .collect();
    let mut gradient = vec![0.0; PARAMETER_COUNT];
    cache.accumulate_parameter_gradient(&dlogits, &mut gradient)?;
    Ok(Sample {
        id,
        split,
        phase,
        chosen,
        features: rows
            .iter()
            .map(|row| Ok(row.values_for_schema(PUBLIC_FEATURE_SCHEMA)?.to_vec()))
            .collect::<Result<_, String>>()?,
        logits: cache.logits().to_vec(),
        dlogits,
        gradient,
    })
}

fn run() -> Result<(), String> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    if !matches!(args.len(), 2 | 3) {
        return Err(
            "Usage: jax-export fixture NEW_FILE | jax-export dataset DATASET_DIR NEW_FILE".into(),
        );
    }
    let artifact = PublicPolicyArtifact::new(7)?;
    let mut export = Export {
        schema: "tzolkin-jax-numeric-batch-v1",
        source_kind: "coreNumericFixture",
        dataset_fingerprint: None,
        feature_schema: PUBLIC_FEATURE_SCHEMA,
        feature_count: FEATURE_COUNT,
        hidden_count: HIDDEN,
        parameter_count: PARAMETER_COUNT,
        parameters: artifact.model.parameters().to_vec(),
        samples: Vec::new(),
    };
    let output = match args[0].as_str() {
        "fixture" if args.len() == 2 => {
            for players in [3, 4] {
                let mut state = tzolkin_core::create_game(
                    (0..players).map(|seat| format!("P{seat}")).collect(),
                    17,
                    false,
                )?;
                for step in 0..24 {
                    let observation = observe(&state, state.current_player)?;
                    let decision = tzolkin_ai::choose_move(&observation)?;
                    let chosen = observation
                        .legal_actions
                        .iter()
                        .position(|action| action.r#move == decision.r#move)
                        .ok_or("Heuristic returned non-legal move")?;
                    let encoder = FeatureEncoder::new_public(&observation)?;
                    let rows = (0..observation.legal_actions.len())
                        .map(|i| encoder.encode_legal_tagged(i))
                        .collect::<Result<Vec<_>, _>>()?;
                    export.samples.push(sample(
                        &artifact,
                        format!("fixture-{players}-{step}"),
                        if players == 3 { "train" } else { "validation" },
                        format!("{:?}", observation.phase),
                        &rows,
                        chosen,
                    )?);
                    state = tzolkin_core::apply_move(&state, decision.r#move)?;
                }
            }
            &args[1]
        }
        "dataset" if args.len() == 3 => {
            let dataset = load_policy_dataset(Path::new(&args[1]))?;
            export.source_kind = "verifiedNativePolicyExport";
            export.dataset_fingerprint = Some(dataset.manifest().fingerprint.clone());
            for (split, name) in [
                (DatasetSplit::Train, "train"),
                (DatasetSplit::Validation, "validation"),
            ] {
                // Consume through EOF, including unselected samples, to finish integrity checks.
                let mut selected = 0usize;
                let mut split_rows = 0usize;
                for item in dataset.iter_split(split) {
                    let item = item?;
                    let proposed_width = export
                        .samples
                        .iter()
                        .map(|s| s.features.len())
                        .chain(std::iter::once(item.candidates().len()))
                        .max()
                        .ok_or("No candidate rows")?
                        .next_power_of_two();
                    if selected < 64
                        && split_rows + item.candidates().len() <= MAX_ROWS / 2
                        && (export.samples.len() + 1) * proposed_width <= MAX_PADDED_ROWS
                    {
                        split_rows += item.candidates().len();
                        selected += 1;
                        export.samples.push(sample(
                            &artifact,
                            item.sample_id().into(),
                            name,
                            format!("{:?}", item.phase()),
                            item.candidates(),
                            item.chosen(),
                        )?);
                    }
                }
                if selected == 0 {
                    return Err(format!("No bounded {name} samples available"));
                }
            }
            &args[2]
        }
        _ => return Err("Invalid export arguments".into()),
    };
    let count: usize = export.samples.iter().map(|s| s.features.len()).sum();
    if count > MAX_ROWS {
        return Err("Fixture exceeds row bound".into());
    }
    let width = export
        .samples
        .iter()
        .map(|s| s.features.len())
        .max()
        .ok_or("No exported samples")?
        .next_power_of_two();
    if export.samples.len() * width > MAX_PADDED_ROWS {
        return Err("Export exceeds padded batch capacity".into());
    }
    let bytes = serde_json::to_vec(&export).map_err(|e| e.to_string())?;
    if bytes.len() > MAX_BATCH_BYTES {
        return Err("Export exceeds 128 MiB".into());
    }
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(output)
        .map_err(|e| e.to_string())?;
    file.write_all(&bytes).map_err(|e| e.to_string())?;
    file.sync_all().map_err(|e| e.to_string())?;
    println!(
        "Exported {} samples / {count} candidates to {output}",
        export.samples.len()
    );
    Ok(())
}

fn main() {
    if let Err(error) = run() {
        eprintln!("{error}");
        std::process::exit(1);
    }
}
