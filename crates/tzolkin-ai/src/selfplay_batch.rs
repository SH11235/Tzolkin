//! CPU並列のnative自己対局。sourceは保存前に全手を独立再検証する。
use std::fs::{self, OpenOptions};
use std::io::{self, BufWriter, Write};
use std::path::Path;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::time::Instant;

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use tzolkin_core::{FinalScore, GameOptions};

use crate::experiment::{InferenceOptions, learned_game_with_options};
use crate::kernel::Kernel;
use crate::model::ModelArtifact;
use crate::replay;

pub const BATCH_SCHEMA: &str = "tzolkin-selfplay-batch-v1";
pub const MAX_BATCH_GAMES: usize = 10_000;
pub const MAX_BATCH_THREADS: usize = 32;
pub const MAX_BATCH_REPLAY_BYTES: u64 = 64 * 1024 * 1024;
pub const MAX_BATCH_MANIFEST_BYTES: u64 = 32 * 1024 * 1024;

/// seedはfirst_seedから連続。games/threadsとseed範囲を出力作成前に検証する。
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct BatchConfig {
    pub players: usize,
    pub first_seed: u32,
    pub games: usize,
    pub threads: usize,
    pub options: GameOptions,
}

/// 完了した対局だけfile/SHA/公式結果を持つ。未取得のdecision数はNone。
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct BatchGame {
    pub index: usize,
    pub seed: u32,
    pub complete: bool,
    pub file: Option<String>,
    pub sha256: Option<String>,
    pub bytes: Option<u64>,
    pub decisions: Option<usize>,
    pub final_scores: Vec<FinalScore>,
    pub final_state: Option<String>,
    pub elapsed_ms: f64,
    pub error: Option<String>,
    pub warnings: Vec<String>,
}

/// manifestは全worker終了後に最後に公開する。complete=falseも処理結果として返す。
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct BatchManifest {
    pub schema: String,
    pub config: BatchConfig,
    pub requested_kernel: String,
    pub backend: String,
    pub model_checksum: Option<String>,
    pub used_threads: usize,
    pub completed: usize,
    pub failed: usize,
    pub complete: bool,
    pub elapsed_ms: f64,
    pub games: Vec<BatchGame>,
    pub warnings: Vec<String>,
}

fn bounded_message(message: impl AsRef<str>) -> String {
    message.as_ref().chars().take(2048).collect()
}

impl BatchGame {
    fn failed(index: usize, seed: u32, error: impl AsRef<str>) -> Self {
        Self {
            index,
            seed,
            complete: false,
            file: None,
            sha256: None,
            bytes: None,
            decisions: None,
            final_scores: Vec::new(),
            final_state: None,
            elapsed_ms: 0.0,
            error: Some(bounded_message(error)),
            warnings: Vec::new(),
        }
    }
}

fn validated_config(config: &BatchConfig) -> Result<BatchConfig, String> {
    if !(2..=5).contains(&config.players)
        || !(1..=MAX_BATCH_GAMES).contains(&config.games)
        || !(1..=MAX_BATCH_THREADS).contains(&config.threads)
    {
        return Err("Batch requires players 2..5, games 1..10000 and threads 1..32".into());
    }
    config
        .first_seed
        .checked_add((config.games - 1) as u32)
        .ok_or("Batch seed range exceeds u32")?;
    let mut effective = config.clone();
    if effective.players == 5 {
        effective.options.quick_actions = true;
    }
    Ok(effective)
}

struct HashWriter<W> {
    inner: W,
    hash: Sha256,
    bytes: u64,
    maximum: u64,
}

impl<W: Write> Write for HashWriter<W> {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        if self
            .bytes
            .checked_add(bytes.len() as u64)
            .is_none_or(|total| total > self.maximum)
        {
            return Err(io::Error::other("Batch JSON exceeds bounded output size"));
        }
        let written = self.inner.write(bytes)?;
        self.hash.update(&bytes[..written]);
        self.bytes += written as u64;
        Ok(written)
    }
    fn flush(&mut self) -> io::Result<()> {
        self.inner.flush()
    }
}

struct Published {
    sha256: String,
    bytes: u64,
    warning: Option<String>,
}

static TEMPORARY_SERIAL: AtomicU64 = AtomicU64::new(0);

/// serialization→flush→sync→create-new hardlink。宛先に既存ファイルがあれば拒否する。
fn publish_json<T: Serialize>(
    value: &T,
    destination: &Path,
    staging: &Path,
    maximum: u64,
) -> Result<Published, String> {
    let mut temporary = None;
    for _ in 0..100 {
        let serial = TEMPORARY_SERIAL.fetch_add(1, Ordering::Relaxed);
        let path = staging.join(format!("{}.{}.tmp", std::process::id(), serial));
        match OpenOptions::new().write(true).create_new(true).open(&path) {
            Ok(file) => {
                temporary = Some((path, file));
                break;
            }
            Err(error) if error.kind() == io::ErrorKind::AlreadyExists => continue,
            Err(error) => return Err(format!("Reserve batch staging file: {error}")),
        }
    }
    let (temporary_path, file) = temporary.ok_or("Could not reserve batch staging file")?;
    let result = (|| {
        let mut writer = HashWriter {
            inner: BufWriter::new(file),
            hash: Sha256::new(),
            bytes: 0,
            maximum,
        };
        serde_json::to_writer(&mut writer, value).map_err(|error| error.to_string())?;
        writer.flush().map_err(|error| error.to_string())?;
        writer
            .inner
            .get_ref()
            .sync_all()
            .map_err(|error| error.to_string())?;
        let HashWriter {
            inner, hash, bytes, ..
        } = writer;
        drop(inner);
        fs::hard_link(&temporary_path, destination)
            .map_err(|error| format!("Publish exclusive batch file: {error}"))?;
        Ok(Published {
            sha256: format!("{:x}", hash.finalize()),
            bytes,
            warning: None,
        })
    })();
    let cleanup = fs::remove_file(&temporary_path);
    match (result, cleanup) {
        (Ok(mut published), Err(error)) => {
            // sourceはすでに検証済み・sync済みで公開された。cleanupだけを対局失敗にしない。
            published.warning = Some(format!("Published source; staging cleanup failed: {error}"));
            Ok(published)
        }
        (result, _) => result,
    }
}

fn process_game(
    config: &BatchConfig,
    index: usize,
    model: Option<&ModelArtifact>,
    kernel: Kernel,
    native: &Path,
    staging: &Path,
) -> BatchGame {
    let seed = config.first_seed + index as u32; // validated_configで範囲確認済み。
    let started = Instant::now();
    let mut game = BatchGame::failed(index, seed, "Game generation did not return a result");
    let result: Result<(), String> = (|| {
        let (state, decisions, replay) = match model {
            Some(model) => learned_game_with_options(
                model,
                config.players,
                seed,
                config.options.clone(),
                &(0..config.players).collect::<Vec<_>>(),
                true,
                InferenceOptions { kernel, fast: true },
            )?,
            None => replay::play_game_fast(config.players, seed, config.options.clone(), true)?,
        };
        game.decisions = Some(decisions);
        let replay = replay.ok_or("Native runner returned no replay")?;
        let expected_source = match (model, &replay.header.source) {
            (None, replay::ReplaySource::SelfPlay { .. }) => true,
            (Some(model), replay::ReplaySource::PolicySelfPlay { policies }) => {
                policies.len() == config.players
                    && policies.iter().all(|policy| {
                        matches!(policy, replay::SeatPolicy::Learned { model_checksum, .. } if model_checksum == &model.checksum)
                    })
            }
            _ => false,
        };
        if replay.header.seed != seed
            || replay.header.options != config.options
            || replay.steps.len() != decisions
            || !expected_source
        {
            return Err("Native replay seed/options/decision count mismatch".into());
        }
        let verified = replay::verify_replay(&replay)?;
        if verified != state {
            return Err("Independently verified replay differs from native final state".into());
        }
        let file = format!("game-{index:05}-seed-{seed}.json");
        let published = publish_json(
            &replay,
            &native.join(&file),
            staging,
            MAX_BATCH_REPLAY_BYTES,
        )?;
        game.file = Some(format!("native/{file}"));
        game.sha256 = Some(published.sha256);
        game.bytes = Some(published.bytes);
        game.final_scores = verified.final_scores;
        game.final_state = Some(replay.final_state);
        game.complete = true;
        game.error = None;
        if let Some(warning) = published.warning {
            game.warnings.push(bounded_message(warning));
        }
        Ok(())
    })();
    if let Err(error) = result {
        game.error = Some(bounded_message(error));
    }
    game.elapsed_ms = started.elapsed().as_secs_f64() * 1000.0;
    game
}

/// 新規directoryへ対局を並列生成する。既存directory/fileは空でも拒否する。
///
/// 個別seedの失敗はpartial manifestで返す。completeは全対局の独立再検証と公開が
/// 成功した場合のみtrue。config/model/出力/manifest公開の失敗はErrで、完了markerを作らない。
pub fn generate_batch(
    config: &BatchConfig,
    model: Option<&ModelArtifact>,
    kernel: Kernel,
    output: &Path,
) -> Result<BatchManifest, String> {
    let effective = validated_config(config)?;
    let resolved = kernel.resolve()?;
    if let Some(model) = model {
        model.validate()?;
    }
    if output.file_name().is_none() {
        return Err("Batch output must name a new directory".into());
    }
    if let Some(parent) = output.parent().filter(|path| !path.as_os_str().is_empty()) {
        fs::create_dir_all(parent).map_err(|error| format!("Create batch parent: {error}"))?;
    }
    fs::create_dir(output).map_err(|error| format!("Create new batch directory: {error}"))?;
    let native = output.join("native");
    let staging = output.join(".staging");
    fs::create_dir(&native).map_err(|error| format!("Create native directory: {error}"))?;
    fs::create_dir(&staging).map_err(|error| format!("Create staging directory: {error}"))?;
    let started = Instant::now();
    let next = AtomicUsize::new(0);
    let results: Vec<Mutex<Option<BatchGame>>> =
        (0..effective.games).map(|_| Mutex::new(None)).collect();
    let mut warnings = Vec::new();
    let mut used_threads = 0;
    std::thread::scope(|scope| {
        let mut handles = Vec::new();
        for worker in 0..effective.threads.min(effective.games) {
            let config = &effective;
            let next = &next;
            let results = &results;
            let native = &native;
            let staging = &staging;
            match std::thread::Builder::new()
                .name(format!("tzolkin-selfplay-{worker}"))
                .spawn_scoped(scope, move || {
                    loop {
                        let index = next.fetch_add(1, Ordering::Relaxed);
                        if index >= config.games {
                            break;
                        }
                        let game_started = Instant::now();
                        let game = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                            process_game(config, index, model, kernel, native, staging)
                        }));
                        let game = match game {
                            Ok(game) => game,
                            Err(payload) => {
                                let message = payload
                                    .downcast_ref::<String>()
                                    .map(String::as_str)
                                    .or_else(|| payload.downcast_ref::<&str>().copied())
                                    .unwrap_or("Unknown worker panic");
                                let mut game = BatchGame::failed(
                                    index,
                                    config.first_seed + index as u32,
                                    format!("Native game panicked: {message}"),
                                );
                                game.elapsed_ms = game_started.elapsed().as_secs_f64() * 1000.0;
                                game
                            }
                        };
                        *results[index]
                            .lock()
                            .unwrap_or_else(std::sync::PoisonError::into_inner) = Some(game);
                    }
                }) {
                Ok(handle) => {
                    used_threads += 1;
                    handles.push(handle);
                }
                Err(error) => warnings.push(format!("Could not start worker {worker}: {error}")),
            }
        }
        for handle in handles {
            if handle.join().is_err() {
                warnings
                    .push("Worker stopped outside a game; unreported seeds are failures".into());
            }
        }
    });
    let games: Vec<_> = results
        .into_iter()
        .enumerate()
        .map(|(index, slot)| {
            slot.into_inner()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .unwrap_or_else(|| {
                    BatchGame::failed(
                        index,
                        effective.first_seed + index as u32,
                        "No worker reported this seed",
                    )
                })
        })
        .collect();
    let completed = games.iter().filter(|game| game.complete).count();
    let manifest = BatchManifest {
        schema: BATCH_SCHEMA.into(),
        config: effective,
        requested_kernel: format!("{kernel:?}"),
        backend: if model.is_some() {
            resolved.backend().into()
        } else {
            "heuristic".into()
        },
        model_checksum: model.map(|model| model.checksum.clone()),
        used_threads,
        completed,
        failed: config.games - completed,
        complete: completed == config.games,
        elapsed_ms: started.elapsed().as_secs_f64() * 1000.0,
        games,
        warnings,
    };
    // stagingはnativeと別に置くので、失敗・中断時にもsource JSONへ混ざらない。
    // manifest公開後は返却値を変更せず、保存済みmanifestと同じ内容を返す。
    publish_json(
        &manifest,
        &output.join("manifest.json"),
        &staging,
        MAX_BATCH_MANIFEST_BYTES,
    )?;
    // 後処理だけで成功source/manifestを削除しない。残った.tmpはnativeには置かない。
    let _ = fs::remove_dir(&staging);
    Ok(manifest)
}

#[cfg(test)]
mod publication_tests {
    use super::*;

    #[test]
    fn publication_is_bounded_exclusive_and_never_exposes_partial_json() {
        let directory = std::env::temp_dir().join(format!(
            "tzolkin-batch-publication-{}-{}",
            std::process::id(),
            TEMPORARY_SERIAL.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir(&directory).unwrap();
        let result = (|| {
            let staging = directory.join(".staging");
            fs::create_dir(&staging)?;
            let oversized = directory.join("oversized.json");
            assert!(publish_json(&"0123456789abcdef", &oversized, &staging, 8).is_err());
            assert!(!oversized.exists());
            assert_eq!(fs::read_dir(&staging)?.count(), 0);
            let source = directory.join("source.json");
            let published = publish_json(&vec![1, 2, 3], &source, &staging, 1024).unwrap();
            let original = fs::read(&source)?;
            assert_eq!(published.bytes, original.len() as u64);
            assert_eq!(published.sha256, format!("{:x}", Sha256::digest(&original)));
            assert!(publish_json(&vec![4, 5, 6], &source, &staging, 1024).is_err());
            assert_eq!(fs::read(&source)?, original);
            assert_eq!(fs::read_dir(&staging)?.count(), 0);
            Ok::<_, io::Error>(())
        })();
        fs::remove_dir_all(&directory).unwrap();
        result.unwrap();
    }
}
