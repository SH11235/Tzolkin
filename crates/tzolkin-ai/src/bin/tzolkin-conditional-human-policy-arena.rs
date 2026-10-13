//! Development-only conditional Playing comparisons with shared Corn-first Setup.
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::fs::{self, File};
use std::io::{Read, Write};
use std::path::{Component, Path};
use tzolkin_ai::arena::conditional::{
    CHOICE_DIAGNOSTICS_SCHEMA, ConditionalPolicy, ConditionalSetup,
    SemanticPlayingConditionalPolicy,
};
use tzolkin_ai::arena::numeric::{
    ExternalExclusions, NumericArenaConfig, NumericPolicy, Opponent, SCHEMA,
    run_corn_first_conditional_validation, run_corn_first_semantic_playing_conditional_validation,
};
use tzolkin_ai::arena::resource_cycle::ResourceCycleConfig;
use tzolkin_ai::dataset::DatasetSplit;
use tzolkin_ai::human_initial_wealth::table_family_id;
use tzolkin_ai::human_policy_dataset::ConditionalFeatureProfile;
use tzolkin_ai::human_policy_training::PreparedConditionalHumanPolicy;
use tzolkin_ai::policy_training::validate_new_output_directory;
use tzolkin_inference::deployment::{LoadedDeployment, MAX_DEPLOYMENT_BYTES};
use tzolkin_inference::trade_guard::TradeGuardConfig;

const INPUT_SCHEMA: &str = "tzolkin-conditional-human-corn-first-arena-input-v1";
const PROFILED_INPUT_SCHEMA: &str =
    "tzolkin-conditional-human-semantic-playing-corn-first-arena-input-v1";
const MAX_INPUT: usize = 16 * 1024 * 1024;
const MAX_REPORT: usize = 32 * 1024 * 1024;
const MAX_TOTAL_OUTPUT: usize = 128 * 1024 * 1024;
const MAX_SEEDS: usize = 64;
const MAX_DIAGNOSTIC_SEEDS: usize = 4;

#[derive(Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct Input {
    schema: String,
    seeds: Vec<u32>,
    bootstrap_seed: u64,
    external_exclusions: ExternalExclusions,
    #[serde(default)]
    choice_diagnostics: bool,
}
#[derive(Deserialize)]
struct InputSchema {
    schema: String,
}
#[derive(Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct ProfiledInput {
    schema: String,
    seeds: Vec<u32>,
    bootstrap_seed: u64,
    external_exclusions: ExternalExclusions,
    actor_profiles: [ConditionalFeatureProfile; 2],
    #[serde(default)]
    choice_diagnostics: bool,
}
fn parse_input(bytes: &[u8]) -> Result<(Input, Option<[ConditionalFeatureProfile; 2]>), String> {
    let header: InputSchema = serde_json::from_slice(bytes).map_err(|e| e.to_string())?;
    match header.schema.as_str() {
        INPUT_SCHEMA => Ok((
            serde_json::from_slice(bytes).map_err(|e| e.to_string())?,
            None,
        )),
        PROFILED_INPUT_SCHEMA => {
            // Parse the original bytes again, retaining duplicate/unknown-field refusal.
            let input: ProfiledInput = serde_json::from_slice(bytes).map_err(|e| e.to_string())?;
            if !input
                .actor_profiles
                .contains(&ConditionalFeatureProfile::SemanticPlaying4)
            {
                return Err(
                    "Semantic Playing Arena input requires an explicit schema-4 actor".into(),
                );
            }
            Ok((
                Input {
                    schema: input.schema,
                    seeds: input.seeds,
                    bootstrap_seed: input.bootstrap_seed,
                    external_exclusions: input.external_exclusions,
                    choice_diagnostics: input.choice_diagnostics,
                },
                Some(input.actor_profiles),
            ))
        }
        _ => Err("Conditional Arena requires a closed input schema".into()),
    }
}
#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct InputPin {
    path: String,
    bytes: usize,
    sha256: String,
}
fn read_pin(path: &Path, limit: usize) -> Result<(Vec<u8>, InputPin), String> {
    let text = path.to_str().ok_or("Invalid local input path")?;
    if text.is_empty()
        || text.contains('\0')
        || text.contains("://")
        || text.starts_with("\\\\")
        || text.starts_with("//")
        || path.components().any(|c| c == Component::ParentDir)
    {
        return Err("Expected a local input path without traversal".into());
    }
    let absolute = if path.is_absolute() {
        path.to_path_buf()
    } else {
        std::env::current_dir()
            .map_err(|e| e.to_string())?
            .join(path)
    };
    for ancestor in absolute.ancestors() {
        let info = fs::symlink_metadata(ancestor).map_err(|e| e.to_string())?;
        if info.file_type().is_symlink() {
            return Err("Input symlink hierarchy rejected".into());
        }
        #[cfg(windows)]
        {
            use std::os::windows::fs::MetadataExt;
            if info.file_attributes() & 0x400 != 0 {
                return Err("Input reparse hierarchy rejected".into());
            }
        }
    }
    let info = fs::metadata(&absolute).map_err(|e| e.to_string())?;
    if !info.is_file() || info.len() == 0 || info.len() > limit as u64 {
        return Err("Input regular-file size outside bound".into());
    }
    let mut bytes = Vec::new();
    File::open(&absolute)
        .map_err(|e| e.to_string())?
        .take(limit as u64 + 1)
        .read_to_end(&mut bytes)
        .map_err(|e| e.to_string())?;
    if bytes.len() > limit || bytes.len() as u64 != info.len() {
        return Err("Input size changed".into());
    }
    let pin = InputPin {
        path: absolute.to_string_lossy().into_owned(),
        bytes: bytes.len(),
        sha256: format!("{:x}", Sha256::digest(&bytes)),
    };
    Ok((bytes, pin))
}
fn recheck(pins: &[InputPin], actors: &[PreparedConditionalHumanPolicy; 2]) -> Result<(), String> {
    for pin in pins {
        let (_, current) = read_pin(Path::new(&pin.path), MAX_INPUT)?;
        if current.bytes != pin.bytes || current.sha256 != pin.sha256 {
            return Err("Arena input changed".into());
        }
    }
    for actor in actors {
        actor.recheck_sources()?;
    }
    Ok(())
}
fn write_new(path: &Path, value: &impl Serialize, total: &mut usize) -> Result<(), String> {
    let bytes = serde_json::to_vec(value).map_err(|e| e.to_string())?;
    let next = total
        .checked_add(bytes.len())
        .ok_or("Output size overflow")?;
    if bytes.len() > MAX_REPORT || next > MAX_TOTAL_OUTPUT {
        return Err("Arena report output cap exceeded".into());
    }
    let mut file = File::options()
        .create_new(true)
        .write(true)
        .open(path)
        .map_err(|e| e.to_string())?;
    file.write_all(&bytes)
        .and_then(|_| file.sync_all())
        .map_err(|e| e.to_string())?;
    *total = next;
    Ok(())
}
fn run() -> Result<(), String> {
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    if args.len() != 7 {
        return Err("Usage: tzolkin-conditional-human-policy-arena LEFT_CHECKPOINT.json LEFT_ORIGIN.json RIGHT_CHECKPOINT.json RIGHT_ORIGIN.json NUMERIC_REFERENCE.json ARENA_INPUT.json NEW_OUTPUT_DIR".into());
    }
    let output = Path::new(&args[6]);
    validate_new_output_directory(output)?;
    let (input_bytes, input_pin) = read_pin(Path::new(&args[5]), MAX_INPUT)?;
    let (input, explicit_profiles) = parse_input(&input_bytes)?;
    let profiled = explicit_profiles.is_some();
    let profiles = explicit_profiles.unwrap_or([ConditionalFeatureProfile::Public2; 2]);
    if ![INPUT_SCHEMA, PROFILED_INPUT_SCHEMA].contains(&input.schema.as_str())
        || input.seeds.len() > MAX_SEEDS
    {
        return Err(
            "Conditional Arena requires its closed schema and at most 64 Validation seeds".into(),
        );
    }
    if input.choice_diagnostics && input.seeds.len() > MAX_DIAGNOSTIC_SEEDS {
        return Err("Conditional choice diagnostics require at most 4 Validation seeds".into());
    }
    let mut configs = [3, 4].map(|players| NumericArenaConfig {
        schema: SCHEMA.into(),
        players,
        seeds: input.seeds.clone(),
        bootstrap_seed: input.bootstrap_seed,
        external_exclusions: input.external_exclusions.clone(),
    });
    for config in &configs {
        config.validate()?;
    }
    let mut pins = vec![input_pin];
    for arg in &args[..4] {
        pins.push(read_pin(Path::new(arg), 8 * 1024 * 1024)?.1);
    }
    let (numeric_bytes, numeric_pin) = read_pin(Path::new(&args[4]), MAX_DEPLOYMENT_BYTES)?;
    pins.push(numeric_pin);
    let load_actor = |index: usize| match profiles[index] {
        ConditionalFeatureProfile::Public2 => PreparedConditionalHumanPolicy::load(
            Path::new(&args[index * 2]),
            Path::new(&args[index * 2 + 1]),
        ),
        ConditionalFeatureProfile::SemanticPlaying4 => {
            PreparedConditionalHumanPolicy::load_semantic_playing(
                Path::new(&args[index * 2]),
                Path::new(&args[index * 2 + 1]),
            )
        }
    };
    let actors = [load_actor(0)?, load_actor(1)?];
    let mut train_families = BTreeSet::new();
    let mut origin_families = BTreeSet::new();
    let mut reservations = BTreeMap::new();
    for actor in &actors {
        for (table, split) in actor.reservations() {
            if reservations
                .insert(table, *split)
                .is_some_and(|old| old != *split)
            {
                return Err("Conditional origins disagree on family split reservations".into());
            }
            if *split == DatasetSplit::Train {
                train_families.insert(table_family_id(table)?);
            }
            if profiled {
                // Include held-out and zero-row declarations from BOTH original collections.
                origin_families.insert(table_family_id(table)?);
            }
        }
    }
    let mut all_excluded = input
        .external_exclusions
        .excluded_family_ids
        .into_iter()
        .collect::<BTreeSet<_>>();
    let added_families = if profiled {
        &origin_families
    } else {
        &train_families
    };
    all_excluded.extend(added_families.iter().cloned());
    let evidence = if profiled {
        serde_json::to_vec(&(
            &pins,
            actors[0].checkpoint_checksum(),
            actors[1].checkpoint_checksum(),
            &origin_families,
            profiles,
        ))
    } else {
        serde_json::to_vec(&(
            &pins,
            actors[0].checkpoint_checksum(),
            actors[1].checkpoint_checksum(),
            &train_families,
        ))
    }
    .map_err(|e| e.to_string())?;
    for config in &mut configs {
        config.external_exclusions.excluded_family_ids = all_excluded.iter().cloned().collect();
        config.external_exclusions.evidence_sha256 = format!("{:x}", Sha256::digest(&evidence));
        config.validate()?;
    }
    recheck(&pins, &actors)?;
    fs::create_dir(output).map_err(|e| e.to_string())?;
    let guard = TradeGuardConfig::default();
    let mut total = 0usize;
    for (i, actor) in actors.iter().enumerate() {
        for config in &configs {
            recheck(&pins, &actors)?;
            let reference =
                NumericPolicy::new(LoadedDeployment::load(&numeric_bytes)?, guard.clone())?;
            let pool = (0..config.players)
                .map(|_| Opponent::ResourceCycle(ResourceCycleConfig::v5()))
                .collect();
            let report = match profiles[i] {
                ConditionalFeatureProfile::Public2 => {
                    let candidate =
                        ConditionalPolicy::new(actor, ConditionalSetup::CornFirst, guard.clone())?;
                    let candidate = if input.choice_diagnostics {
                        candidate.with_choice_diagnostics()
                    } else {
                        candidate
                    };
                    run_corn_first_conditional_validation(config, candidate, reference, pool)
                }
                ConditionalFeatureProfile::SemanticPlaying4 => {
                    let candidate = SemanticPlayingConditionalPolicy::new(
                        actor,
                        ConditionalSetup::CornFirst,
                        guard.clone(),
                    )?;
                    let candidate = if input.choice_diagnostics {
                        candidate.with_choice_diagnostics()
                    } else {
                        candidate
                    };
                    run_corn_first_semantic_playing_conditional_validation(
                        config, candidate, reference, pool,
                    )
                }
            };
            if profiled {
                recheck(&pins, &actors)?;
            }
            let report = report?;
            if !profiled {
                recheck(&pins, &actors)?;
            }
            write_new(
                &output.join(format!("actor{}-{}p.json", i + 1, config.players)),
                &report,
                &mut total,
            )?;
            if report.blocks.iter().any(|b| !b.complete) {
                return Err("Incomplete Arena block; no comparison completion claimed".into());
            }
        }
    }
    recheck(&pins, &actors)?;
    let mut summary = serde_json::json!({
        "schema": "tzolkin-conditional-human-paired-arena-cli-v1", "complete": true,
        "inputs": pins, "originIdentities": [actors[0].identity(), actors[1].identity()],
        "checkpointChecksums": [actors[0].checkpoint_checksum(), actors[1].checkpoint_checksum()],
        "trainFamiliesAddedToExclusions": train_families, "numericReferenceLoads": 4,
        "plannedGames": input.seeds.len() * 28, "guard": guard,
        "pool": ResourceCycleConfig::v5(), "development": true,
        "freshHoldoutClaimed": false, "sourceQualified": false,
        "verifiedNativePolicy": false, "trainingAdmissionGranted": false,
        "valueValidity": "unavailablePolicyOnly"
    });
    if input.choice_diagnostics {
        summary["choiceDiagnostics"] = serde_json::json!(CHOICE_DIAGNOSTICS_SCHEMA);
    }
    if profiled {
        summary["schema"] =
            serde_json::json!("tzolkin-conditional-human-semantic-playing-paired-arena-cli-v1");
        summary["actorFeatureProfiles"] = serde_json::json!(profiles);
        summary["originReservedFamiliesAddedToExclusions"] = serde_json::json!(origin_families);
    }
    write_new(&output.join("summary.json"), &summary, &mut total)?;
    println!(
        "Completed conditional Arena: 4 comparisons, {} games; Development Validation only",
        input.seeds.len() * 28
    );
    Ok(())
}
fn main() {
    if let Err(error) = run() {
        eprintln!("{error}");
        std::process::exit(1);
    }
}
