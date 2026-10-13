//! Explicit inference-only Validation; numeric exports never create owners.
use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::fs;
use std::path::{Component, Path, PathBuf};
use tzolkin_ai::arena::numeric::{
    self, AppDiagnosticConsent, AppOpponent, AppProfile, NumericArenaConfig, NumericPolicy,
    Opponent,
};
use tzolkin_ai::arena::resource_cycle::ResourceCycleConfig;
use tzolkin_ai::public_rl_paired_session::read_bounded;
use tzolkin_inference::deployment::{LoadedDeployment, MAX_DEPLOYMENT_BYTES};
use tzolkin_inference::trade_guard::TradeGuardConfig;

const COMMAND_SCHEMA: &str = "tzolkin-numeric-corn-first-validation-command-v1";
const APP_COMMAND_SCHEMA: &str = "tzolkin-app-profile-validation-command-v1";
const APP_DIAGNOSTIC_COMMAND_SCHEMA: &str = "tzolkin-app-profile-diagnostic-command-v1";
const APP_FEEDING_COMMAND_SCHEMA: &str = "tzolkin-app-profile-feeding-diagnostic-command-v1";
const MAX_CONFIG_BYTES: usize = 16 * 1024 * 1024;
const MAX_COMMAND_SEEDS: usize = 64;
const HELP: &str = "tzolkin-numeric-arena corn-first-validation CONFIG.json CONFIG_SHA256 NEW_REPORT.json\n\
Closed camelCase config: schema=tzolkin-numeric-corn-first-validation-command-v1, arena (NumericArenaConfig), candidate/reference {path, bytes, sha256}, opponentPool=resourceCycleV5. Deployment paths are relative to the config directory or absolute local paths. Both exact pinned exports use the regular numeric loader and default Trade guard; no owner or producer is imported.\n\
arena.schema=tzolkin-numeric-deployment-validation-arena-v1; players is 3 or 4, seeds has 1..64 unique base Validation families, bootstrapSeed is a Rust u64, externalExclusions has evidenceSha256 and distinct excludedFamilyIds. The caller must supply the complete Train/Validation/Test/teacher/opponent/spent exclusions and an external permanent reservation before execution; this command checks declared disjointness, not completeness or reservation authenticity.\n\
Every seat in both paired arms uses the same corn-first-setup-v2/default-weight Setup selector. Playing uses the candidate/reference numeric policy at each focal seat and the frozen V5 absolute-seat opponent pool. All numeric guards receive every actor callback and keep independent per-arm history. Original all-seat pairing, alternating arm order and whole-family bootstrap remain in force.\n\
The config is at most16MiB, each deployment at most1MiB, and the create-new report at most32MiB. Report names use 1..64 ASCII letters/digits/hyphen/period, without a leading/trailing period or Windows device stem. Use an external wall/process watchdog: the finite synchronous engine cannot be interrupted by this command. Exit0 means all declared games completed, exit2 means the saved report contains failed games, and exit1 means a command/load/storage error. No training, fresh Test, strength or adoption authority is granted.\n";
const APP_HELP: &str = "tzolkin-numeric-arena app-validation CONFIG.json CONFIG_SHA256 NEW_REPORT.json\n\
tzolkin-numeric-arena app-diagnostic CONFIG.json CONFIG_SHA256 NEW_REPORT.json\n\
tzolkin-numeric-arena app-feeding-diagnostic CONFIG.json CONFIG_SHA256 NEW_REPORT.json\n\
Closed camelCase App config: schema=tzolkin-app-profile-validation-command-v1 (or tzolkin-app-profile-diagnostic-command-v1), arena, candidate=foodAlignedV2|monumentAlignedV1|monumentAlignedV2|resourceAlignedV2, reference, opponentPool (one role per absolute seat), diagnosticConsent (absent for Validation, required for diagnostic). A role is {kind:appProfile,profile:...}, {kind:nativeV2}, {kind:cornFirstSetupV2}, {kind:resourceCycleV5}, or {kind:numeric,deployment:{path,bytes,sha256}}. App roles call the actual fixed-weight App chooser in both Setup and Playing; there is no implicit Setup override. Numeric roles use the regular loader/default guard, retain independent per-arm history and receive every actor callback.\n\
Validation preserves the full exclusions and unique Validation families. Diagnostic consent uses schema=tzolkin-app-development-reanalysis-consent-v1, evidenceSha256, exposedDevelopmentFamilyIds and protectedHoldoutFamilyIds. The exact selected allowlist must already be excluded, any declared additional protected list must be excluded and disjoint, and Test families are always refused. Caller digests bind declarations, not historical authenticity or completeness. Diagnostic is count-neutral exposed Development reanalysis, never fresh Test, training or adoption. Feeding events and actual unfed-worker counts are not collected and remain unknown.\n\
Both routes keep the finite 3/4p all-seat paired engine, 1..64 seeds, existing byte caps, create-new report rules and external watchdog/exit conventions above. The caller must bind complete exclusions and explicit prior exposure or future reservation before execution.\n";
const FEEDING_HELP: &str = "The explicit app-feeding-diagnostic sibling requires schema=tzolkin-app-profile-feeding-diagnostic-command-v1 and the same Development consent. It wraps the unchanged diagnostic report with actual Core Feeding receipts, indexed by seed/absolute focal seat/arm and zero-based applied move index. Every completed arm requires all four food days and every player; an arm or later paired-block failure discards the entire block's receipts as null, never zero. Scheduled food day and actual round are separate. Legacy unfedWorkers stays unknown. There are at most8192 events (64 seeds, 4 players, both arms/all focal seats), within the existing32MiB report cap. These output-only penalties do not guide policy selection, grant training/adoption authority or require zero early-food penalties.\n";

#[derive(Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct DeploymentInput {
    path: PathBuf,
    bytes: usize,
    sha256: String,
}
impl DeploymentInput {
    fn read(&self, directory: &Path) -> Result<Vec<u8>, String> {
        if self.bytes == 0 || self.bytes > MAX_DEPLOYMENT_BYTES || !digest(&self.sha256) {
            return Err("Invalid bounded deployment input pin".into());
        }
        let bytes = read_bounded(&directory.join(&self.path), MAX_DEPLOYMENT_BYTES)?;
        if bytes.len() != self.bytes || format!("{:x}", Sha256::digest(&bytes)) != self.sha256 {
            return Err("Numeric Arena deployment raw input pin differs".into());
        }
        Ok(bytes)
    }
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
enum OpponentPool {
    ResourceCycleV5,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct CommandConfig {
    schema: String,
    arena: NumericArenaConfig,
    candidate: DeploymentInput,
    reference: DeploymentInput,
    opponent_pool: OpponentPool,
}

#[derive(Deserialize)]
#[serde(tag = "kind", rename_all = "camelCase", deny_unknown_fields)]
enum AppRoleInput {
    AppProfile { profile: AppProfile },
    NativeV2 {},
    CornFirstSetupV2 {},
    ResourceCycleV5 {},
    Numeric { deployment: DeploymentInput },
}
impl AppRoleInput {
    fn into_opponent<'a>(
        self,
        directory: &Path,
        weights: &'a tzolkin_ai::policy::HeuristicWeights,
    ) -> Result<AppOpponent<'a>, String> {
        Ok(match self {
            Self::AppProfile { profile } => AppOpponent::App(profile),
            Self::NativeV2 {} => AppOpponent::Existing(Opponent::Heuristic(weights)),
            Self::CornFirstSetupV2 {} => AppOpponent::Existing(Opponent::CornFirstSetup(weights)),
            Self::ResourceCycleV5 {} => {
                AppOpponent::Existing(Opponent::ResourceCycle(ResourceCycleConfig::v5()))
            }
            Self::Numeric { deployment } => {
                AppOpponent::Existing(Opponent::GuardedNumeric(Box::new(NumericPolicy::new(
                    LoadedDeployment::load(&deployment.read(directory)?)?,
                    TradeGuardConfig::default(),
                )?)))
            }
        })
    }
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct AppCommandConfig {
    schema: String,
    arena: NumericArenaConfig,
    candidate: AppProfile,
    reference: AppRoleInput,
    opponent_pool: Vec<AppRoleInput>,
    diagnostic_consent: Option<AppDiagnosticConsent>,
}

fn digest(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}

fn new_report(path: &Path) -> Result<PathBuf, String> {
    let text = path.to_string_lossy();
    if text.starts_with('\\')
        || text.starts_with("//")
        || text.contains("://")
        || path.components().any(|c| matches!(c, Component::ParentDir))
    {
        return Err("Numeric Arena output must be a local path without parent traversal".into());
    }
    let name = path
        .file_name()
        .and_then(|s| s.to_str())
        .ok_or("Numeric Arena output requires a portable report filename")?;
    let stem = name.split('.').next().unwrap_or("").to_ascii_uppercase();
    let device = matches!(stem.as_str(), "CON" | "PRN" | "AUX" | "NUL")
        || (stem.len() == 4
            && (stem.starts_with("COM") || stem.starts_with("LPT"))
            && matches!(stem.as_bytes()[3], b'1'..=b'9'));
    if name.is_empty()
        || name.len() > 64
        || name.starts_with('.')
        || name.ends_with('.')
        || device
        || !name
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'-' | b'.'))
    {
        return Err("Numeric Arena report needs a fresh portable non-device leaf".into());
    }
    let parent = path
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    let parent = fs::canonicalize(parent).map_err(|e| e.to_string())?;
    if !parent.is_dir() {
        return Err("Numeric Arena report parent must be an existing directory".into());
    }
    let destination = parent.join(name);
    match fs::symlink_metadata(&destination) {
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(destination),
        _ => Err("Numeric Arena report destination must be absent".into()),
    }
}

fn run_corn_first(args: &[String]) -> Result<i32, String> {
    if args.len() != 4 || args[0] != "corn-first-validation" || !digest(&args[2]) {
        return Err("Expected corn-first-validation CONFIG CONFIG_SHA256 NEW_REPORT".into());
    }
    let config_path = Path::new(&args[1]);
    let bytes = read_bounded(config_path, MAX_CONFIG_BYTES)?;
    if format!("{:x}", Sha256::digest(&bytes)) != args[2] {
        return Err("Numeric Arena command raw configuration pin differs".into());
    }
    let config: CommandConfig = serde_json::from_slice(&bytes).map_err(|e| e.to_string())?;
    if config.schema != COMMAND_SCHEMA || config.arena.seeds.len() > MAX_COMMAND_SEEDS {
        return Err("Unsupported or oversized explicit Corn-first Validation command".into());
    }
    config.arena.validate()?;
    let output = new_report(Path::new(&args[3]))?;
    let directory = config_path
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    // Validate the entire finite plan and fresh output before either model load.
    let candidate = config.candidate.read(directory)?;
    let reference = config.reference.read(directory)?;
    let candidate = NumericPolicy::new(
        LoadedDeployment::load(&candidate)?,
        TradeGuardConfig::default(),
    )?;
    let reference = NumericPolicy::new(
        LoadedDeployment::load(&reference)?,
        TradeGuardConfig::default(),
    )?;
    let pool = match config.opponent_pool {
        OpponentPool::ResourceCycleV5 => (0..config.arena.players)
            .map(|_| Opponent::ResourceCycle(ResourceCycleConfig::v5()))
            .collect(),
    };
    let report = numeric::run_corn_first_validation(&config.arena, candidate, reference, pool)?;
    report.save_new(&output)?;
    println!(
        "{}",
        serde_json::json!({
            "schema": COMMAND_SCHEMA,
            "configSha256": args[2],
            "reportSchema": report.schema,
            "configurationSha256": report.configuration_sha256,
            "plannedGames": report.statistics.planned_games,
            "completedGames": report.statistics.completed_games,
            "failedGames": report.statistics.failed_games,
            "strengthImprovementDeclared": false
        })
    );
    Ok(if report.statistics.failed_games == 0 {
        0
    } else {
        2
    })
}

fn run_app(args: &[String]) -> Result<i32, String> {
    if args.len() != 4 || !digest(&args[2]) {
        return Err("Expected App route CONFIG CONFIG_SHA256 NEW_REPORT".into());
    }
    let config_path = Path::new(&args[1]);
    let bytes = read_bounded(config_path, MAX_CONFIG_BYTES)?;
    if format!("{:x}", Sha256::digest(&bytes)) != args[2] {
        return Err("App Arena command raw configuration pin differs".into());
    }
    let config: AppCommandConfig = serde_json::from_slice(&bytes).map_err(|e| e.to_string())?;
    if config.arena.seeds.len() > MAX_COMMAND_SEEDS
        || config.opponent_pool.len() != config.arena.players
    {
        return Err("App Arena requires bounded seeds and one role per absolute seat".into());
    }
    match (
        args[0].as_str(),
        config.schema.as_str(),
        &config.diagnostic_consent,
    ) {
        ("app-validation", APP_COMMAND_SCHEMA, None) => config.arena.validate()?,
        ("app-diagnostic", APP_DIAGNOSTIC_COMMAND_SCHEMA, Some(consent)) => {
            consent.validate_for(&config.arena)?
        }
        ("app-feeding-diagnostic", APP_FEEDING_COMMAND_SCHEMA, Some(consent)) => {
            consent.validate_for(&config.arena)?
        }
        _ => return Err("App Arena route/schema/diagnostic consent mismatch".into()),
    }
    let output = new_report(Path::new(&args[3]))?;
    let directory = config_path
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    let weights = tzolkin_ai::policy::HeuristicWeights::default();
    // Whole finite admission and fresh output precede all numeric loads.
    let reference = config.reference.into_opponent(directory, &weights)?;
    let pool = config
        .opponent_pool
        .into_iter()
        .map(|role| role.into_opponent(directory, &weights))
        .collect::<Result<Vec<_>, _>>()?;
    let (schema, checksum, planned, completed, failed) = match config.diagnostic_consent {
        Some(consent) if args[0] == "app-feeding-diagnostic" => {
            let report = numeric::run_app_profile_feeding_diagnostic(
                &config.arena,
                &consent,
                config.candidate,
                reference,
                pool,
            )?;
            report.save_new(&output)?;
            (
                report.schema,
                report.configuration_sha256,
                report.diagnostic.paired.statistics.planned_games,
                report.diagnostic.paired.statistics.completed_games,
                report.diagnostic.paired.statistics.failed_games,
            )
        }
        Some(consent) => {
            let report = numeric::run_app_profile_diagnostic(
                &config.arena,
                &consent,
                config.candidate,
                reference,
                pool,
            )?;
            report.save_new(&output)?;
            (
                report.schema,
                report.configuration_sha256,
                report.paired.statistics.planned_games,
                report.paired.statistics.completed_games,
                report.paired.statistics.failed_games,
            )
        }
        None => {
            let report = numeric::run_app_profile_validation(
                &config.arena,
                config.candidate,
                reference,
                pool,
            )?;
            report.save_new(&output)?;
            (
                report.schema,
                report.configuration_sha256,
                report.statistics.planned_games,
                report.statistics.completed_games,
                report.statistics.failed_games,
            )
        }
    };
    println!(
        "{}",
        serde_json::json!({
            "schema":schema, "configSha256":args[2], "configurationSha256":checksum,
            "plannedGames":planned, "completedGames":completed, "failedGames":failed,
            "countNeutral":args[0] != "app-validation", "strengthImprovementDeclared":false
        })
    );
    Ok(if failed == 0 { 0 } else { 2 })
}

fn run(args: &[String]) -> Result<i32, String> {
    match args.first().map(String::as_str) {
        Some("app-validation" | "app-diagnostic" | "app-feeding-diagnostic") => run_app(args),
        _ => run_corn_first(args),
    }
}

fn main() {
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    if args.as_slice() == ["--help"] {
        print!("{HELP}{APP_HELP}{FEEDING_HELP}");
        return;
    }
    match run(&args) {
        Ok(code) => std::process::exit(code),
        Err(error) => {
            eprintln!("{error}");
            std::process::exit(1);
        }
    }
}
