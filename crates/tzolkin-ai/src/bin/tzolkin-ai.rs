use std::alloc::{GlobalAlloc, Layout, System};
use std::hint::black_box;
use std::io::{Read, Write};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;
use tzolkin_ai::{choose_move, dispatch_cpu, replay};
use tzolkin_core::compact::CompactState;
use tzolkin_core::observation::observe;

struct CountingAllocator;
static ALLOCATIONS: AtomicUsize = AtomicUsize::new(0);
unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        unsafe { System.alloc(layout) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, size: usize) -> *mut u8 {
        ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        unsafe { System.realloc(ptr, layout, size) }
    }
}
#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;
fn value(args: &[String], name: &str, default: &str) -> Result<String, String> {
    let Some(index) = args.iter().position(|s| s == name) else {
        return Ok(default.into());
    };
    args.get(index + 1)
        .cloned()
        .ok_or_else(|| format!("Missing value after {name}"))
}
fn number<T: std::str::FromStr>(args: &[String], name: &str, default: &str) -> Result<T, String> {
    value(args, name, default)?
        .parse()
        .map_err(|_| format!("Invalid {name}"))
}
fn output(value: &impl serde::Serialize) -> Result<(), String> {
    println!(
        "{}",
        serde_json::to_string(value).map_err(|e| e.to_string())?
    );
    Ok(())
}
fn run() -> Result<(), String> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let command = args.first().map(String::as_str).unwrap_or("help");
    if command == "choose" {
        let mut request = String::new();
        std::io::stdin()
            .read_to_string(&mut request)
            .map_err(|e| e.to_string())?;
        println!("{}", dispatch_cpu(&request)?);
        return Ok(());
    }
    if command == "dispatch" {
        // Use the same validated boundary as Wasm/Tauri for offline public replay tooling.
        let mut request = String::new();
        std::io::stdin()
            .take(16 * 1024 * 1024 + 1)
            .read_to_string(&mut request)
            .map_err(|e| e.to_string())?;
        if request.len() > 16 * 1024 * 1024 {
            return Err("Dispatch input exceeds 16 MiB".into());
        }
        println!("{}", tzolkin_core::api::dispatch_game(&request)?);
        return Ok(());
    }
    if command == "replay" {
        let path = args.get(1).ok_or("Usage: tzolkin-ai replay PATH")?;
        let replay: replay::GameReplay =
            serde_json::from_slice(&std::fs::read(path).map_err(|e| e.to_string())?)
                .map_err(|e| e.to_string())?;
        let state = replay::verify_replay(&replay)?;
        return output(
            &serde_json::json!({"verified":true,"decisions":replay.steps.len(),"finalScores":state.final_scores}),
        );
    }
    if command == "corpus" {
        let seeds = number(&args, "--seeds", "32")?;
        return output(&replay::verify_corpus(seeds)?);
    }
    if command == "dataset" {
        let input = value(&args, "--input", "")?;
        let destination = value(&args, "--output", "")?;
        if input.is_empty() || destination.is_empty() {
            return Err(
                "Usage: tzolkin-ai dataset --input REPLAY_DIRECTORY --output NEW_DIRECTORY".into(),
            );
        }
        let mut paths = Vec::new();
        for entry in std::fs::read_dir(&input).map_err(|error| error.to_string())? {
            let path = entry.map_err(|error| error.to_string())?.path();
            if path
                .extension()
                .is_some_and(|extension| extension == "json")
            {
                paths.push(path);
                if paths.len() > 100_000 {
                    return Err("Too many dataset source files".into());
                }
            }
        }
        paths.sort();
        let manifest =
            tzolkin_ai::dataset::export_dataset_files(&paths, std::path::Path::new(&destination))?;
        return output(
            &serde_json::json!({"fingerprint":manifest.fingerprint,"games":manifest.games.len(),"samples":manifest.samples,"shards":manifest.shards.len(),"strata":manifest.strata,"sourceKind":manifest.source_kind}),
        );
    }
    let players: usize = number(&args, "--players", "2")?;
    let seed: u32 = number(&args, "--seed", "0")?;
    let mut flags: u8 = number(&args, "--flags", "0")?;
    if flags > 15 {
        return Err("Flags must be 0..15 (additional=1, tribes=2, prophecies=4, quick=8)".into());
    }
    if players == 5 {
        flags |= 8;
    }
    let options = replay::options_from_mask(flags);
    if command == "selfplay" {
        let path = args
            .iter()
            .position(|s| s == "--output")
            .map(|i| args.get(i + 1).ok_or("Missing output path"))
            .transpose()?;
        let started = Instant::now();
        let (state, decisions, record) = replay::play_game(players, seed, options, path.is_some())?;
        if let Some(path) = path {
            if std::path::Path::new(path).exists() {
                return Err("Replay output already exists".into());
            }
            let replay = record.ok_or("Missing replay")?;
            replay::verify_replay(&replay)?;
            let temporary = format!("{path}.tmp-{}", std::process::id());
            let mut file = std::fs::OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&temporary)
                .map_err(|e| e.to_string())?;
            file.write_all(&serde_json::to_vec(&replay).map_err(|e| e.to_string())?)
                .map_err(|e| e.to_string())?;
            file.sync_all().map_err(|e| e.to_string())?;
            std::fs::rename(&temporary, path).map_err(|e| e.to_string())?;
        }
        return output(
            &serde_json::json!({"players":players,"seed":seed,"flags":flags,"decisions":decisions,"elapsedMs":started.elapsed().as_secs_f64()*1000.0,"finalScores":state.final_scores}),
        );
    }
    if command == "bench" {
        let iterations: usize = number(&args, "--iterations", "10")?;
        if iterations == 0 {
            return Err("Iterations must be positive".into());
        }
        let states = replay::benchmark_states(players, seed, options)?;
        let observations = states
            .iter()
            .map(|s| observe(s, s.current_player))
            .collect::<Result<Vec<_>, _>>()?;
        let decisions = observations
            .iter()
            .map(choose_move)
            .collect::<Result<Vec<_>, _>>()?;
        let compact = states
            .iter()
            .map(CompactState::from_saved)
            .collect::<Result<Vec<_>, _>>()?;
        let state_keys = states
            .iter()
            .map(replay::state_key)
            .collect::<Result<Vec<_>, _>>()?;
        let corpus_hash = format!(
            "{:016x}",
            tzolkin_core::observation::fingerprint(
                &serde_json::to_vec(&state_keys).map_err(|e| e.to_string())?
            )
        );
        let mut coverage = std::collections::BTreeMap::<String, usize>::new();
        for observation in &observations {
            let kind = observation
                .pending_task
                .as_ref()
                .map(|task| serde_json::to_value(task).map_err(|e| e.to_string()))
                .transpose()?
                .and_then(|value| {
                    value
                        .get("type")
                        .and_then(serde_json::Value::as_str)
                        .map(str::to_owned)
                })
                .unwrap_or_else(|| observation.phase.to_string());
            *coverage.entry(kind).or_default() += 1;
            if observation.actor != observation.turn_player {
                *coverage.entry("actorSwitch".into()).or_default() += 1;
            }
            if observation.legal_actions.iter().any(|a| {
                matches!(
                    a.action,
                    tzolkin_core::observation::TypedAction::Build {
                        renovation: Some(_),
                        ..
                    }
                )
            }) {
                *coverage.entry("renovation".into()).or_default() += 1;
            }
        }
        let count = iterations * states.len();
        let mut reports = serde_json::Map::new();
        for phase in [
            "referenceClone",
            "legal",
            "transition",
            "compactConversion",
            "compactClone",
            "compactLegal",
            "compactTransition",
            "observation",
            "policy",
        ] {
            let mut times = Vec::with_capacity(count);
            let allocation_start = ALLOCATIONS.load(Ordering::Relaxed);
            let start = Instant::now();
            for _ in 0..iterations {
                for (index, state) in states.iter().enumerate() {
                    let decision_start = Instant::now();
                    match phase {
                        "referenceClone" => {
                            black_box(state.clone());
                        }
                        "legal" => {
                            black_box(tzolkin_core::get_available_moves(black_box(state)));
                        }
                        "transition" => {
                            black_box(tzolkin_core::apply_move(
                                black_box(state),
                                decisions[index].r#move.clone(),
                            )?);
                        }
                        "observation" => {
                            black_box(observe(black_box(state), state.current_player)?);
                        }
                        "compactConversion" => {
                            black_box(CompactState::from_saved(black_box(state))?);
                        }
                        "compactClone" => {
                            black_box(compact[index].clone());
                        }
                        "compactLegal" => {
                            black_box(compact[index].legal_moves());
                        }
                        "compactTransition" => {
                            black_box(compact[index].apply_move(decisions[index].r#move.clone())?);
                        }
                        _ => {
                            black_box(choose_move(black_box(&observations[index]))?);
                        }
                    }
                    times.push(decision_start.elapsed().as_nanos());
                }
            }
            let seconds = start.elapsed().as_secs_f64();
            let allocations = ALLOCATIONS.load(Ordering::Relaxed) - allocation_start;
            times.sort_unstable();
            reports.insert(phase.into(),serde_json::json!({"operations":count,"seconds":seconds,"operationsPerSecond":count as f64/seconds,"p50Ns":times[count/2],"p95Ns":times[(count*95/100).min(count-1)],"allocations":allocations}));
        }
        let rss = std::fs::read_to_string("/proc/self/status")
            .ok()
            .and_then(|s| {
                s.lines()
                    .find(|l| l.starts_with("VmRSS:"))
                    .map(str::to_owned)
            });
        return output(
            &serde_json::json!({"schema":1,"policyVersion":tzolkin_ai::POLICY_VERSION,"weights":tzolkin_ai::policy::HeuristicWeights::default(),"rulesBaseline":replay::RULES_BASELINE,"catalogHash":replay::catalog_hash(),"players":players,"seed":seed,"flags":flags,"iterations":iterations,"buildProfile":if cfg!(debug_assertions) { "debug" } else { "optimized" },"targetArch":std::env::consts::ARCH,"corpusHash":corpus_hash,"corpusStates":states.len(),"coverage":coverage,"stateStackBytes":std::mem::size_of::<tzolkin_core::GameState>(),"compactStateStackBytes":std::mem::size_of::<tzolkin_core::compact::CompactState>(),"stackSizesExcludeHeap":true,"rss":rss,"phases":reports,"strengthMeasured":false,"simdOrUndoMeasured":false}),
        );
    }
    println!(
        "tzolkin-ai choose | dispatch (validated JSON on stdin) | selfplay --players 2..5 --seed N --flags 0..15 [--output PATH] | replay PATH | dataset --input REPLAY_DIRECTORY --output NEW_DIRECTORY | corpus [--seeds 32] | bench --players 2..5 --flags 0..15 [--iterations 10]\nFlags: additional=1 tribes=2 prophecies=4 quick=8; five players force quick.\nBench measures correctness-neutral baseline operations; it does not measure playing strength."
    );
    Ok(())
}
fn main() {
    if let Err(error) = run() {
        eprintln!("{error}");
        std::process::exit(1);
    }
}
