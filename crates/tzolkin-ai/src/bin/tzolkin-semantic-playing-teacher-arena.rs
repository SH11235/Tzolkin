//! Dedicated, source-loaded native4 teacher paired evaluation. No training route.
use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::{
    fs::{self, File},
    io::{Read, Write},
    path::{Path, PathBuf},
};
use tzolkin_ai::{
    arena::{
        numeric::{
            NativeTeacherDiagnosticConsent, NumericArenaConfig, NumericPolicy, Opponent,
            run_native_playing_teacher_diagnostic, run_native_playing_teacher_validation,
        },
        resource_cycle::ResourceCycleConfig,
        teacher::{PreparedSemanticPlayingTeacher, SemanticPlayingTeacherReference},
    },
    policy::HeuristicWeights,
    policy_training::validate_new_output_directory,
};
use tzolkin_inference::{
    deployment::{LoadedDeployment, MAX_DEPLOYMENT_BYTES},
    trade_guard::TradeGuardConfig,
};

#[derive(Deserialize)]
#[serde(tag = "kind", rename_all = "camelCase", deny_unknown_fields)]
enum Reference {
    ResourceCycleV5 {},
    Numeric { deployment: PathBuf },
}
#[derive(Deserialize)]
#[serde(tag = "kind", rename_all = "camelCase", deny_unknown_fields)]
enum Baseline {
    ResourceCycleV5 {},
    Numeric { deployment: PathBuf },
    Heuristic { weights: HeuristicWeights },
}
#[derive(Debug, Deserialize, PartialEq, Eq)]
#[serde(try_from = "String")]
enum InputSchema {
    V5,
    V7,
    V7Agreement,
    V7FundedUxmal,
}
impl TryFrom<String> for InputSchema {
    type Error = &'static str;
    fn try_from(schema: String) -> Result<Self, Self::Error> {
        match schema.as_str() {
            "tzolkin-native-playing4-teacher-arena-input-v1" => Ok(Self::V5),
            "tzolkin-native-playing4-v7-teacher-arena-input-v1" => Ok(Self::V7),
            "tzolkin-native-playing4-v7-agreement-diagnostic-input-v1" => Ok(Self::V7Agreement),
            "tzolkin-native-playing4-v7-funded-uxmal-arena-input-v1" => Ok(Self::V7FundedUxmal),
            _ => Err("Native teacher input schema/absolute-seat roster mismatch"),
        }
    }
}
impl InputSchema {
    fn prepare(
        &self,
        checkpoint: &Path,
        dataset: &Path,
    ) -> Result<PreparedSemanticPlayingTeacher, String> {
        match self {
            Self::V5 => PreparedSemanticPlayingTeacher::load(checkpoint, dataset),
            Self::V7 | Self::V7Agreement | Self::V7FundedUxmal => {
                PreparedSemanticPlayingTeacher::load_v7(checkpoint, dataset)
            }
        }
    }
}
#[derive(Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct Input {
    schema: InputSchema,
    config: NumericArenaConfig,
    consent: Option<NativeTeacherDiagnosticConsent>,
    reference: Reference,
    opponent_pool: Vec<Baseline>,
}
impl Input {
    fn agreement_mode(&self) -> Result<bool, String> {
        let enabled = self.schema == InputSchema::V7Agreement;
        if enabled && (self.consent.is_none() || self.config.seeds.len() > 4) {
            return Err(
                "V7 agreement requires exposed Development consent and at most four families"
                    .into(),
            );
        }
        Ok(enabled)
    }
}
struct Pin {
    path: PathBuf,
    bytes: usize,
    sha: Vec<u8>,
}
fn local_absolute(path: PathBuf) -> Result<PathBuf, String> {
    let text = path.to_str().ok_or("Arena input path must be UTF-8")?;
    if !path.is_absolute()
        || text.starts_with("\\\\")
        || text.starts_with("//")
        || path.components().any(|part| {
            matches!(part, std::path::Component::Prefix(prefix)
                if !matches!(prefix.kind(), std::path::Prefix::Disk(_)))
        })
    {
        return Err("Arena inputs must be local files".into());
    }
    Ok(path)
}
fn read(path: &Path, max: usize) -> Result<(Vec<u8>, Pin), String> {
    let text = path.to_str().ok_or("Arena input path must be UTF-8")?;
    if text.is_empty()
        || text.contains('\0')
        || text.contains("://")
        || text.starts_with("\\\\")
        || text.starts_with("//")
        || (!path.is_absolute()
            && matches!(
                path.components().next(),
                Some(std::path::Component::Prefix(_))
            ))
    {
        return Err("Arena inputs must be local files".into());
    }
    let path = if path.is_absolute() {
        path.to_path_buf()
    } else {
        std::env::current_dir()
            .map_err(|e| e.to_string())?
            .join(path)
    };
    let path = local_absolute(path)?;
    if path
        .components()
        .any(|p| matches!(p, std::path::Component::ParentDir))
    {
        return Err("Local input path must not traverse parents".into());
    }
    let mut ancestor = PathBuf::new();
    for component in path.components() {
        ancestor.push(component.as_os_str());
        if matches!(component, std::path::Component::Prefix(_)) {
            continue;
        }
        let meta = fs::symlink_metadata(&ancestor).map_err(|e| e.to_string())?;
        if meta.file_type().is_symlink() {
            return Err("Input symlink hierarchy refused".into());
        }
        #[cfg(windows)]
        {
            use std::os::windows::fs::MetadataExt;
            if meta.file_attributes() & 0x400 != 0 {
                return Err("Input reparse hierarchy refused".into());
            }
        }
    }
    let info = fs::metadata(&path).map_err(|e| e.to_string())?;
    if !info.is_file() || info.len() > max as u64 {
        return Err("Arena input byte/type cap".into());
    }
    let mut bytes = Vec::new();
    File::open(&path)
        .map_err(|e| e.to_string())?
        .take(max as u64 + 1)
        .read_to_end(&mut bytes)
        .map_err(|e| e.to_string())?;
    if bytes.len() as u64 != info.len() || bytes.len() > max {
        return Err("Arena input changed/too large".into());
    }
    let pin = Pin {
        path,
        bytes: bytes.len(),
        sha: Sha256::digest(&bytes).to_vec(),
    };
    Ok((bytes, pin))
}
fn numeric(path: &Path, pins: &mut Vec<Pin>) -> Result<NumericPolicy, String> {
    let (bytes, pin) = read(path, MAX_DEPLOYMENT_BYTES)?;
    pins.push(pin);
    NumericPolicy::new(LoadedDeployment::load(&bytes)?, TradeGuardConfig::default())
}
fn baseline<'a>(b: &'a Baseline, pins: &mut Vec<Pin>) -> Result<Opponent<'a>, String> {
    Ok(match b {
        Baseline::ResourceCycleV5 {} => Opponent::ResourceCycle(ResourceCycleConfig::v5()),
        Baseline::Numeric { deployment } => {
            Opponent::GuardedNumeric(Box::new(numeric(deployment, pins)?))
        }
        Baseline::Heuristic { weights } => Opponent::Heuristic(weights),
    })
}
fn run() -> Result<(), String> {
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    if args.len() != 4 {
        return Err("Usage: tzolkin-semantic-playing-teacher-arena CHECKPOINT.json DATASET_DIR INPUT.json NEW_OUTPUT_DIR".into());
    }
    let output = Path::new(&args[3]);
    validate_new_output_directory(output)?;
    let parent = output
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    let destination = parent
        .canonicalize()
        .map_err(|e| e.to_string())?
        .join(output.file_name().ok_or("Output directory name required")?);
    if destination.starts_with(
        Path::new(&args[1])
            .canonicalize()
            .map_err(|e| e.to_string())?,
    ) {
        return Err("Arena output must be outside its immutable dataset".into());
    }
    let (bytes, pin) = read(Path::new(&args[2]), 16 * 1024 * 1024)?;
    let input: Input = serde_json::from_slice(&bytes).map_err(|e| e.to_string())?;
    if input.opponent_pool.len() != input.config.players || input.config.seeds.len() > 64 {
        return Err("Native teacher input schema/absolute-seat roster mismatch".into());
    }
    match &input.consent {
        Some(c) => c.validate_for(&input.config)?,
        None => input.config.validate()?,
    }
    let agreement = input.agreement_mode()?;
    let mut pins = vec![pin];
    let prepared = input
        .schema
        .prepare(Path::new(&args[0]), Path::new(&args[1]))?;
    let result = (|| {
        let reference = match &input.reference {
            Reference::ResourceCycleV5 {} => SemanticPlayingTeacherReference::ResourceCycleV5,
            Reference::Numeric { deployment } => SemanticPlayingTeacherReference::GuardedNumeric(
                Box::new(numeric(deployment, &mut pins)?),
            ),
        };
        let pool = input
            .opponent_pool
            .iter()
            .map(|b| baseline(b, &mut pins))
            .collect::<Result<Vec<_>, _>>()?;
        let candidate = prepared.arena_handle()?;
        let candidate = if agreement {
            candidate.with_v7_agreement_diagnostics()?
        } else {
            candidate
        };
        let candidate = if input.schema == InputSchema::V7FundedUxmal {
            candidate.with_v7_funded_uxmal_opening()?
        } else {
            candidate
        };
        match &input.consent {
            Some(c) => {
                run_native_playing_teacher_diagnostic(&input.config, c, candidate, reference, pool)
            }
            None => {
                run_native_playing_teacher_validation(&input.config, candidate, reference, pool)
            }
        }
    })();
    let mut post = Ok(());
    for p in &pins {
        let checked = read(&p.path, 16 * 1024 * 1024).and_then(|(_, now)| {
            if now.bytes == p.bytes && now.sha == p.sha {
                Ok(())
            } else {
                Err("Arena config/reference changed".into())
            }
        });
        if let Err(e) = checked {
            post = Err(e);
        }
    }
    let native_post = prepared.recheck_sources();
    post?;
    native_post?;
    let report = result?;
    let bytes = serde_json::to_vec(&report).map_err(|e| e.to_string())?;
    if bytes.len() > 32 * 1024 * 1024 {
        return Err("Arena report exceeds byte cap".into());
    }
    fs::create_dir(output).map_err(|e| e.to_string())?;
    let mut file = File::options()
        .write(true)
        .create_new(true)
        .open(output.join("report.json"))
        .map_err(|e| e.to_string())?;
    file.write_all(&bytes)
        .and_then(|_| file.sync_all())
        .map_err(|e| e.to_string())?;
    println!(
        "Completed paired native4 teacher evaluation; source-loaded Development report, no training or strength qualification"
    );
    Ok(())
}
fn main() {
    if let Err(e) = run() {
        eprintln!("{e}");
        std::process::exit(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{collections::hash_map::RandomState, hash::BuildHasher, time::SystemTime};

    #[test]
    fn local_input_pin_can_be_read_again() {
        let mut control = serde_json::json!({
            "schema":"tzolkin-native-playing4-teacher-arena-input-v1",
            "config":{"schema":tzolkin_ai::arena::numeric::SCHEMA,"players":3,
                "seeds":[17],"bootstrapSeed":7,
                "externalExclusions":{"evidenceSha256":"0".repeat(64),
                    "excludedFamilyIds":[]}},
            "reference":{"kind":"resourceCycleV5"},
            "opponentPool":[{"kind":"resourceCycleV5"},
                {"kind":"resourceCycleV5"},{"kind":"resourceCycleV5"}]
        });
        for (schema, expected) in [
            (
                "tzolkin-native-playing4-teacher-arena-input-v1",
                InputSchema::V5,
            ),
            (
                "tzolkin-native-playing4-v7-teacher-arena-input-v1",
                InputSchema::V7,
            ),
            (
                "tzolkin-native-playing4-v7-funded-uxmal-arena-input-v1",
                InputSchema::V7FundedUxmal,
            ),
        ] {
            control["schema"] = schema.into();
            let parsed: Input = serde_json::from_value(control.clone()).unwrap();
            assert_eq!(parsed.schema, expected);
            assert!(matches!(parsed.reference, Reference::ResourceCycleV5 {}));
            assert_eq!(parsed.opponent_pool.len(), 3);
            assert!(parsed.consent.is_none());
        }
        control["schema"] = "tzolkin-native-playing4-v7-agreement-diagnostic-input-v1".into();
        let diagnostic: Input = serde_json::from_value(control.clone()).unwrap();
        assert_eq!(diagnostic.schema, InputSchema::V7Agreement);
        assert!(diagnostic.agreement_mode().is_err()); // no consent, no loading
        control["schema"] = "tzolkin-native-playing4-v8-teacher-arena-input-v1".into();
        assert!(serde_json::from_value::<Input>(control.clone()).is_err());
        control["schema"] =
            serde_json::json!({"tzolkin-native-playing4-v7-teacher-arena-input-v1":null});
        assert!(serde_json::from_value::<Input>(control.clone()).is_err());
        control.as_object_mut().unwrap().remove("schema");
        assert!(serde_json::from_value::<Input>(control).is_err());
        struct OwnedTemp {
            path: PathBuf,
            file: Option<File>,
        }
        impl Drop for OwnedTemp {
            fn drop(&mut self) {
                drop(self.file.take());
                let _ = fs::remove_file(&self.path);
            }
        }
        let mut owned = (0..16)
            .find_map(|_| {
                let path = std::env::temp_dir().join(format!(
                    "tzolkin-native4-arena-read-{:016x}-{:032x}.json",
                    RandomState::new().hash_one(()),
                    SystemTime::now()
                        .duration_since(SystemTime::UNIX_EPOCH)
                        .unwrap()
                        .as_nanos(),
                ));
                match File::options().write(true).create_new(true).open(&path) {
                    Ok(file) => Some(OwnedTemp {
                        path,
                        file: Some(file),
                    }),
                    Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => None,
                    Err(e) => panic!("Cannot create owned temporary input: {e}"),
                }
            })
            .expect("Unique create-new temporary input");
        let file = owned.file.as_mut().unwrap();
        file.write_all(b"{\"closed\":true}").unwrap();
        file.sync_all().unwrap();
        drop(owned.file.take());
        let (bytes, first) = read(&owned.path, 1024).unwrap();
        assert_eq!(first.path, owned.path);
        assert!(first.path.is_absolute());
        let (reread, second) = read(&first.path, 1024).unwrap();
        assert_eq!(reread, bytes);
        assert_eq!((second.bytes, second.sha), (first.bytes, first.sha));
        for refused in [
            "\\\\?\\C:\\untrusted",
            "\\\\server\\share\\input",
            "https://invalid/input",
        ] {
            assert_eq!(
                read(Path::new(refused), 1024).err().as_deref(),
                Some("Arena inputs must be local files")
            );
        }
        #[cfg(windows)]
        for refused in ["\\/server/share/input", "/\\server/share/input"] {
            assert_eq!(
                read(Path::new(refused), 1024).err().as_deref(),
                Some("Arena inputs must be local files")
            );
        }
        for cwd in [
            "\\\\server\\share\\cwd",
            "//server/share/cwd",
            "\\\\?\\C:\\cwd",
        ] {
            let joined = Path::new(cwd).join("relative-input.json");
            assert_eq!(
                local_absolute(joined).err().as_deref(),
                Some("Arena inputs must be local files")
            );
        }
        #[cfg(windows)]
        for cwd in ["\\/server/share/cwd", "/\\server/share/cwd"] {
            let joined = Path::new(cwd).join("relative-input.json");
            assert_eq!(
                local_absolute(joined).err().as_deref(),
                Some("Arena inputs must be local files")
            );
        }
        assert_eq!(
            local_absolute(PathBuf::from("relative-input.json"))
                .err()
                .as_deref(),
            Some("Arena inputs must be local files")
        );
        let local = std::env::temp_dir().join("relative-input.json");
        assert_eq!(local_absolute(local.clone()).unwrap(), local);
    }
}
