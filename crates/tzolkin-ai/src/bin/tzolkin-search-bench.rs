//! In-process Search measurements. Overlapping stages are not a decomposition.
use serde::Serialize;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, HashSet};
use std::hint::black_box;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;
use tzolkin_ai::replay;
use tzolkin_ai::search::{
    DiscardReason, FallbackReason, LEAF_VERSION, PreparedSearch, SEARCH_POLICY_VERSION,
    SearchConfig, SearchOutcome, SearchStatus,
};
use tzolkin_core::observation::{Observation, observation_key};
use tzolkin_core::rollout::{RolloutRoot, RolloutWorld, SAMPLING_VERSION};
use tzolkin_core::{GameMove, GameOptions, Phase};

const MAX_SEARCH_OPERATIONS: usize = 20_000_000;
const USAGE: &str = "tzolkin-search-bench --config SEARCH.json [--players 3|4 --seed 11235 --states 1..64 --iterations 1..32 --rounds 3..15]\nDefaults: players=4, seed=11235, states=12, iterations=1, rounds=5. Config is bounded to64 KiB; combined planned Search/micro-apply attempts <=20 million. Models and playing strength are not measured. Stop competing work before interpreting release timings.";

struct Config {
    path: PathBuf,
    players: usize,
    seed: u32,
    states: usize,
    iterations: usize,
    rounds: usize,
}
fn config(args: &[String]) -> Result<Config, String> {
    let mut result = Config {
        path: PathBuf::new(),
        players: 4,
        seed: 11235,
        states: 12,
        iterations: 1,
        rounds: 5,
    };
    let mut seen = HashSet::new();
    for pair in args.chunks(2) {
        if pair.len() != 2 || pair[1].starts_with("--") || !seen.insert(pair[0].as_str()) {
            return Err("Expected unique flag/value pairs".into());
        }
        match pair[0].as_str() {
            "--config" => result.path = PathBuf::from(&pair[1]),
            "--players" => result.players = pair[1].parse().map_err(|_| "Invalid players")?,
            "--seed" => result.seed = pair[1].parse().map_err(|_| "Invalid u32 seed")?,
            "--states" => result.states = pair[1].parse().map_err(|_| "Invalid states")?,
            "--iterations" => {
                result.iterations = pair[1].parse().map_err(|_| "Invalid iterations")?
            }
            "--rounds" => result.rounds = pair[1].parse().map_err(|_| "Invalid rounds")?,
            _ => return Err(format!("Unknown flag {}", pair[0])),
        }
    }
    if result.path.as_os_str().is_empty()
        || ![3, 4].contains(&result.players)
        || !(1..=64).contains(&result.states)
        || !(1..=32).contains(&result.iterations)
        || !(3..=15).contains(&result.rounds)
    {
        return Err(
            "Requires --config, base3|4 players, states1..64, iterations1..32, rounds3..15".into(),
        );
    }
    Ok(result)
}
fn load_search(path: &Path) -> Result<SearchConfig, String> {
    let mut bytes = Vec::new();
    std::fs::File::open(path)
        .map_err(|error| error.to_string())?
        .take(64 * 1024 + 1)
        .read_to_end(&mut bytes)
        .map_err(|error| error.to_string())?;
    if bytes.len() > 64 * 1024 {
        return Err("Search configuration exceeds64 KiB".into());
    }
    let config: SearchConfig = serde_json::from_slice(&bytes).map_err(|error| error.to_string())?;
    config.validate()?;
    Ok(config)
}
#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct Budget {
    maximum_prepared_search_atomic_attempts: usize,
    micro_apply_and_check_calls: usize,
    maximum_corpus_apply_calls: usize,
    combined_maximum: usize,
    limit: usize,
    uses_requested_states_conservatively: bool,
}
fn budget(config: &Config, search: &SearchConfig) -> Result<Budget, String> {
    let timed = config
        .rounds
        .checked_mul(config.iterations)
        .ok_or("Workload overflow")?;
    // One direct baseline, warmup of whole+stratum and both timed series.
    let repetitions = timed
        .checked_mul(2)
        .and_then(|n| n.checked_add(3))
        .ok_or("Workload overflow")?;
    let atomic = config
        .states
        .checked_mul(repetitions)
        .and_then(|n| n.checked_mul(search.max_total_steps))
        .ok_or("Workload overflow")?;
    // Two prepared after-operation references; root/sample/clone output checks
    // each apply once outside clock, and clone+apply+observe applies inside clock.
    let micro = timed
        .checked_add(1)
        .and_then(|n| n.checked_mul(4))
        .and_then(|n| n.checked_add(2))
        .and_then(|n| n.checked_mul(config.states))
        .ok_or("Workload overflow")?;
    let corpus = replay::MAX_DECISIONS
        .checked_mul(2)
        .ok_or("Workload overflow")?;
    let combined = atomic
        .checked_add(micro)
        .and_then(|n| n.checked_add(corpus))
        .ok_or("Workload overflow")?;
    if combined > MAX_SEARCH_OPERATIONS {
        return Err(
            "Planned worst-case Search/micro-apply workload exceeds20 million attempts".into(),
        );
    }
    Ok(Budget {
        maximum_prepared_search_atomic_attempts: atomic,
        micro_apply_and_check_calls: micro,
        maximum_corpus_apply_calls: corpus,
        combined_maximum: combined,
        limit: MAX_SEARCH_OPERATIONS,
        uses_requested_states_conservatively: true,
    })
}

#[derive(Clone, Copy, Serialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
enum Stratum {
    Normal,
    Setup,
    Pending,
}
impl Stratum {
    fn index(self) -> usize {
        match self {
            Self::Normal => 0,
            Self::Setup => 1,
            Self::Pending => 2,
        }
    }
    fn classify(o: &Observation) -> Self {
        if o.phase == Phase::Setup {
            Self::Setup
        } else if o.pending_task.is_some() {
            Self::Pending
        } else {
            Self::Normal
        }
    }
}
fn select(record: &replay::GameReplay, count: usize) -> Result<(Vec<usize>, [usize; 3]), String> {
    let mut groups = [Vec::new(), Vec::new(), Vec::new()];
    for (index, step) in record.steps.iter().enumerate() {
        groups[Stratum::classify(&step.observation).index()].push(index);
    }
    if record.steps.len() < count || groups[0].is_empty() {
        return Err("Corpus has insufficient states or no normal public roots".into());
    }
    let mut quotas = [0; 3];
    for _ in 0..count {
        // Round robin normal/setup/pending, skipping exhausted strata.
        let available = (0..3).filter(|&i| quotas[i] < groups[i].len());
        let next = available
            .min_by_key(|&i| (quotas[i], i))
            .ok_or("Empty stratum selection")?;
        quotas[next] += 1;
    }
    let mut selected = Vec::with_capacity(count);
    for (group, &quota) in groups.iter().zip(&quotas) {
        for i in 0..quota {
            selected.push(group[(2 * i + 1) * group.len() / (2 * quota)]);
        }
    }
    selected.sort_unstable();
    Ok((selected, groups.map(|group| group.len())))
}

#[derive(Default, Serialize)]
#[serde(rename_all = "camelCase")]
struct Work {
    decisions: u64,
    searched: u64,
    fallbacks: u64,
    attempted_worlds: u64,
    completed_worlds: u64,
    discarded_worlds: u64,
    atomic_attempts: u64,
    terminal_rollouts: u64,
    cutoff_rollouts: u64,
    max_reached_round: Option<i64>,
    minimum_scored_round: Option<i64>,
    maximum_scored_round: Option<i64>,
    fallback_reasons: BTreeMap<FallbackReason, u64>,
    discard_reasons: BTreeMap<DiscardReason, u64>,
}
impl Work {
    fn add(&mut self, o: &SearchOutcome) {
        self.decisions += 1;
        match o.status {
            SearchStatus::Searched => self.searched += 1,
            SearchStatus::Fallback { reason } => {
                self.fallbacks += 1;
                *self.fallback_reasons.entry(reason).or_default() += 1;
            }
        }
        self.attempted_worlds += u64::from(o.stats.attempted_worlds);
        self.completed_worlds += u64::from(o.stats.completed_worlds);
        self.discarded_worlds += o.stats.discarded_worlds.len() as u64;
        self.atomic_attempts += o.stats.atomic_steps as u64;
        self.terminal_rollouts += o.stats.terminal_rollouts as u64;
        self.cutoff_rollouts += o.stats.cutoff_rollouts as u64;
        if let Some(round) = o.stats.max_reached_round {
            self.max_reached_round = Some(self.max_reached_round.map_or(round, |v| v.max(round)));
        }
        for candidate in &o.candidates {
            if let Some(round) = candidate.minimum_scored_round {
                self.minimum_scored_round =
                    Some(self.minimum_scored_round.map_or(round, |v| v.min(round)));
            }
            if let Some(round) = candidate.maximum_scored_round {
                self.maximum_scored_round =
                    Some(self.maximum_scored_round.map_or(round, |v| v.max(round)));
            }
        }
        for d in &o.stats.discarded_worlds {
            *self.discard_reasons.entry(d.reason).or_default() += 1;
        }
    }
}
#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct Timing {
    round_seconds: Vec<f64>,
    median_round_seconds: f64,
    operations_per_round: usize,
    operations_per_second: f64,
    mean_ns_per_operation: f64,
}
fn timings(samples: Vec<f64>, operations: usize) -> Result<Timing, String> {
    if samples.is_empty() || operations == 0 || samples.iter().any(|v| !v.is_finite() || *v <= 0.0)
    {
        return Err(
            "Requires positive finite round durations; increase --iterations if clock reports zero"
                .into(),
        );
    }
    let mut ordered = samples.clone();
    ordered.sort_by(f64::total_cmp);
    let mid = ordered.len() / 2;
    let median = if ordered.len().is_multiple_of(2) {
        ordered[mid - 1] / 2.0 + ordered[mid] / 2.0
    } else {
        ordered[mid]
    };
    let rate = operations as f64 / median;
    let ns = median / operations as f64 * 1e9;
    if !rate.is_finite() || rate <= 0.0 || !ns.is_finite() || ns <= 0.0 {
        return Err("Timing rates outside finite positive range".into());
    }
    Ok(Timing {
        round_seconds: samples,
        median_round_seconds: median,
        operations_per_round: operations,
        operations_per_second: rate,
        mean_ns_per_operation: ns,
    })
}
#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct Stage {
    name: &'static str,
    inputs: usize,
    timing: Timing,
    warmup_outputs_checked: usize,
    timed_outputs_checked: usize,
    warmup_search_work: Option<Work>,
    timed_search_work: Option<Work>,
}
fn measure<T>(
    name: &'static str,
    indices: &[usize],
    config: &Config,
    mut operation: impl FnMut(usize) -> Result<T, String>,
    mut validate: impl FnMut(usize, &T, &mut Work) -> Result<(), String>,
    search: bool,
) -> Result<Stage, String> {
    if indices.is_empty() {
        return Err("Cannot measure empty stage".into());
    }
    let mut warm = Work::default();
    for &index in indices {
        let output = black_box(operation(black_box(index))?);
        validate(index, &output, &mut warm)?;
    }
    let mut timed_work = Work::default();
    let operations = indices
        .len()
        .checked_mul(config.iterations)
        .ok_or("Operations overflow")?;
    let mut samples = Vec::with_capacity(config.rounds);
    for _ in 0..config.rounds {
        // Preallocation is outside the clock. Push/loop/black_box overhead is included.
        let mut outputs = Vec::with_capacity(operations);
        let start = Instant::now();
        for _ in 0..config.iterations {
            for &index in indices {
                outputs.push(black_box(operation(black_box(index))?));
            }
        }
        let elapsed = start.elapsed().as_secs_f64();
        for (at, output) in outputs.iter().enumerate() {
            validate(indices[at % indices.len()], output, &mut timed_work)?;
        }
        // Disposal of returned results is excluded; API-internal drops remain measured.
        samples.push(elapsed);
    }
    Ok(Stage {
        name,
        inputs: indices.len(),
        timing: timings(samples, operations)?,
        warmup_outputs_checked: indices.len(),
        timed_outputs_checked: operations * config.rounds,
        warmup_search_work: search.then_some(warm),
        timed_search_work: search.then_some(timed_work),
    })
}
fn bytes(value: &impl Serialize) -> Result<Vec<u8>, String> {
    serde_json::to_vec(value).map_err(|error| error.to_string())
}
fn equal(expected: &[u8], value: &impl Serialize) -> Result<(), String> {
    if bytes(value)? == expected {
        Ok(())
    } else {
        Err("Exact public output/diagnostic byte parity failed".into())
    }
}
fn sha(value: &impl Serialize) -> Result<String, String> {
    Ok(format!("{:x}", Sha256::digest(bytes(value)?)))
}
fn world_signature(world: &RolloutWorld, operation: &GameMove) -> Result<Vec<u8>, String> {
    let before = world.observation()?;
    let scores = world.score_projection();
    let mut after = world.clone();
    after.apply(operation.clone())?;
    bytes(&(
        before,
        scores,
        after.observation()?,
        after.score_projection(),
        after.finished(),
    ))
}
fn basename(value: Option<&str>) -> Option<String> {
    value
        .and_then(|s| Path::new(s).file_name())
        .map(|s| s.to_string_lossy().into_owned())
}
fn compiler_metadata() -> Value {
    let version =
        std::process::Command::new(std::env::var_os("RUSTC").unwrap_or_else(|| "rustc".into()))
            .arg("-Vv")
            .output();
    let version = version.ok().filter(|o| o.status.success()).map(|o| {
        String::from_utf8_lossy(&o.stdout)
            .chars()
            .take(4096)
            .collect::<String>()
    });
    json!({"runtimeRustcVerboseVersion":version,"runtimeCompilerAttestsBuildCompiler":false,
        "compileEnvironmentRustcName":basename(option_env!("RUSTC")),
        "compileEnvironmentWrapperName":basename(option_env!("RUSTC_WRAPPER")),
        "compileEnvironmentWorkspaceWrapperName":basename(option_env!("RUSTC_WORKSPACE_WRAPPER")),
        "runtimeWrapperName":basename(std::env::var("RUSTC_WRAPPER").ok().as_deref())})
}
fn executable_sha() -> Result<String, String> {
    let mut file = std::fs::File::open(std::env::current_exe().map_err(|e| e.to_string())?)
        .map_err(|e| e.to_string())?;
    if file.metadata().map_err(|e| e.to_string())?.len() > 128 * 1024 * 1024 {
        return Err("Executable metadata exceeds128 MiB".into());
    }
    let mut hash = Sha256::new();
    let mut buffer = [0; 65_536];
    loop {
        let n = file.read(&mut buffer).map_err(|e| e.to_string())?;
        if n == 0 {
            break;
        }
        hash.update(&buffer[..n]);
    }
    Ok(format!("{:x}", hash.finalize()))
}

fn run(config: Config) -> Result<Value, String> {
    let search = load_search(&config.path)?;
    let planned = budget(&config, &search)?; // Fail before corpus/game creation.
    let prepared = PreparedSearch::new(&search)?;
    let (_, _, record) =
        replay::play_game_fast(config.players, config.seed, GameOptions::default(), true)?;
    let record = record.ok_or("Missing native corpus replay")?;
    replay::verify_replay(&record)?;
    let replay_sha = sha(&record)?;
    let (selected, available_counts) = select(&record, config.states)?;
    let observations = selected
        .iter()
        .map(|&i| record.steps[i].observation.clone())
        .collect::<Vec<_>>();
    let selected_sha = sha(&observations)?;
    let heuristic = observations
        .iter()
        .map(tzolkin_ai::choose_move)
        .collect::<Result<Vec<_>, _>>()?;
    for (at, &index) in selected.iter().enumerate() {
        if heuristic[at].r#move != record.steps[index].chosen.r#move {
            return Err("Native corpus/default heuristic mismatch".into());
        }
    }
    let heuristic_bytes = heuristic.iter().map(bytes).collect::<Result<Vec<_>, _>>()?;
    let reference = observations
        .iter()
        .map(|o| prepared.choose(o))
        .collect::<Result<Vec<_>, _>>()?;
    let reference_bytes = reference.iter().map(bytes).collect::<Result<Vec<_>, _>>()?;
    let mut reference_work = Work::default();
    for output in &reference {
        reference_work.add(output);
    }
    let mut groups = [Vec::new(), Vec::new(), Vec::new()];
    let mut roots = vec![None; observations.len()];
    let mut worlds = vec![None; observations.len()];
    let mut signatures = vec![Vec::new(); observations.len()];
    let mut after_bytes = vec![Vec::new(); observations.len()];
    let mut score_bytes = vec![Vec::new(); observations.len()];
    let mut metadata = Vec::new();
    for (at, o) in observations.iter().enumerate() {
        let stratum = Stratum::classify(o);
        groups[stratum.index()].push(at);
        if stratum == Stratum::Normal {
            let root = RolloutRoot::from_observation(o)?;
            let world = root.sample(0, search.sampling_salt);
            if world.observation()? != *o {
                return Err("Public root roundtrip mismatch".into());
            }
            let scores = world.score_projection();
            if scores.iter().any(|v| !v.is_finite()) {
                return Err("Non-finite public projection".into());
            }
            signatures[at] = world_signature(&world, &heuristic[at].r#move)?;
            let mut after = world.clone();
            after.apply(heuristic[at].r#move.clone())?;
            after_bytes[at] = bytes(&after.observation()?)?;
            score_bytes[at] = bytes(&scores)?;
            roots[at] = Some(root);
            worlds[at] = Some(world);
        }
        metadata.push(json!({"replayIndex":selected[at],"stratum":stratum,"phase":o.phase,
            "round":o.round,"actor":o.actor,"turnPlayer":o.turn_player,"legalActions":o.legal_actions.len(),
            "observationKey":o.observation_key,"rootStagesEligible":stratum == Stratum::Normal,
            "directOutcomeSha256":sha(&reference[at])?,"directStatus":reference[at].status,
            "directScoreKind":reference[at].score_kind,"directMove":reference[at].decision.r#move}));
    }
    let all = (0..observations.len()).collect::<Vec<_>>();
    let mut stages = Vec::new();
    for (name, indices) in [
        ("preparedChooseAll", &all),
        ("preparedChooseNormal", &groups[0]),
        ("preparedChooseSetup", &groups[1]),
        ("preparedChoosePending", &groups[2]),
    ] {
        if !indices.is_empty() {
            stages.push(measure(
                name,
                indices,
                &config,
                |i| prepared.choose(black_box(&observations[i])),
                |i, output, work| {
                    if output != &reference[i] {
                        return Err("Exact Search outcome parity failed".into());
                    }
                    equal(&reference_bytes[i], output)?;
                    work.add(output);
                    Ok(())
                },
                true,
            )?);
        }
    }
    let normal = &groups[0];
    stages.push(measure(
        "rootConstruction",
        normal,
        &config,
        |i| RolloutRoot::from_observation(black_box(&observations[i])),
        |i, root, _| {
            if world_signature(&root.sample(0, search.sampling_salt), &heuristic[i].r#move)?
                == signatures[i]
            {
                Ok(())
            } else {
                Err("Root public signature mismatch".into())
            }
        },
        false,
    )?);
    stages.push(measure(
        "worldSample",
        normal,
        &config,
        |i| Ok(roots[i].as_ref().unwrap().sample(0, search.sampling_salt)),
        |i, world, _| {
            if world_signature(world, &heuristic[i].r#move)? == signatures[i] {
                Ok(())
            } else {
                Err("Sample public signature mismatch".into())
            }
        },
        false,
    )?);
    stages.push(measure(
        "worldClone",
        normal,
        &config,
        |i| Ok(worlds[i].as_ref().unwrap().clone()),
        |i, world, _| {
            if world_signature(world, &heuristic[i].r#move)? == signatures[i] {
                Ok(())
            } else {
                Err("Clone public signature mismatch".into())
            }
        },
        false,
    )?);
    stages.push(measure(
        "cloneApplyObserve",
        normal,
        &config,
        |i| {
            let mut world = worlds[i].as_ref().unwrap().clone();
            world.apply(heuristic[i].r#move.clone())?;
            world.observation()
        },
        |i, o, _| equal(&after_bytes[i], o),
        false,
    )?);
    stages.push(measure(
        "worldObservation",
        normal,
        &config,
        |i| worlds[i].as_ref().unwrap().observation(),
        |i, o, _| {
            if *o == observations[i] {
                Ok(())
            } else {
                Err("Observation parity failed".into())
            }
        },
        false,
    )?);
    stages.push(measure(
        "publicScoreProjection",
        normal,
        &config,
        |i| Ok(worlds[i].as_ref().unwrap().score_projection()),
        |i, scores, _| equal(&score_bytes[i], scores),
        false,
    )?);
    stages.push(measure(
        "heuristicChoose",
        &all,
        &config,
        |i| tzolkin_ai::choose_move(black_box(&observations[i])),
        |i, decision, _| equal(&heuristic_bytes[i], decision),
        false,
    )?);
    stages.push(measure(
        "observationKey",
        &all,
        &config,
        |i| observation_key(black_box(&observations[i])),
        |i, key, _| {
            if *key == observations[i].observation_key {
                Ok(())
            } else {
                Err("Key parity failed".into())
            }
        },
        false,
    )?);
    Ok(
        json!({"schema":"tzolkin-search-bench-v1","players":config.players,"seed":config.seed,
        "options":GameOptions::default(),"source":record.header.source,"sourceReplaySha256":replay_sha,
        "selectedCorpusSha256":selected_sha,"sourceReplayDecisions":record.steps.len(),
        "selection":{"algorithm":"balanced-normal-setup-pending-centered-quantiles-v1","requested":config.states,
            "availableByStratum":available_counts,"selectedByStratum":groups.map(|v|v.len()),"states":metadata},
        "iterations":config.iterations,"rounds":config.rounds,"searchConfig":search,
        "configurationKey":prepared.configuration_key(),"searchPolicyVersion":SEARCH_POLICY_VERSION,
        "leafVersion":LEAF_VERSION,"samplingVersion":SAMPLING_VERSION,"rulesVersion":replay::RULES_VERSION,
        "objective":tzolkin_ai::search::Objective::ScoreMargin,"heuristicPolicyVersion":tzolkin_ai::POLICY_VERSION,
        "rolloutWeights":tzolkin_ai::policy::HeuristicWeights::default(),
        "rulesBaseline":replay::RULES_BASELINE,"catalogHash":replay::catalog_hash(),
        "plannedWorkload":planned,"directReferenceSearchWork":reference_work,"stages":stages,
        "parity":{"allWarmupAndTimedSearchOutcomesExact":true,"exactSerializedSearchDiagnostics":true,
            "opaqueStageCheck":"public observation/projection and one public continuation; hidden state not serialized"},
        "measurement":{"clock":"Instant monotonic","latencyQuantilesCollected":false,"stagesOverlap":true,
            "stageTimesAreNotAdditive":true,"libraryInternalValidationAndSerializationIncluded":true,
            "collectorPushAndLoopIncluded":true,"resultDisposalOutsideClock":true,"allocationMeasurement":null,
            "preparationParityReportAndFileIoOutsideClock":true,"wholeAndStratumSearchSeriesOverlap":true},
        "policyInput":"current actor Observation only; runner seed/fullstate used only for corpus preparation",
        "buildProfile":if cfg!(debug_assertions) {"debug"} else {"optimized"},
        "targetArch":std::env::consts::ARCH,"targetOs":std::env::consts::OS,
        "compiler":compiler_metadata(),"executableSha256":executable_sha()?,
        "strengthMeasured":false,"speedImprovementDeclared":false,"quietMachineAttested":false}),
    )
}
fn main() {
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    if args.is_empty() || args == ["--help"] {
        println!("{USAGE}");
        return;
    }
    let result = config(&args).and_then(run).and_then(|value| {
        let bytes = bytes(&value)?;
        if bytes.len() > 2 * 1024 * 1024 {
            return Err("Benchmark report exceeds2 MiB".into());
        }
        let mut out = std::io::stdout().lock();
        out.write_all(&bytes)
            .and_then(|_| out.write_all(b"\n"))
            .map_err(|e| e.to_string())
    });
    if let Err(error) = result {
        eprintln!("{error}");
        std::process::exit(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn timing_rejects_zero_nonfinite_empty_and_invalid_rates_without_thresholds() {
        for values in [
            vec![],
            vec![0.0],
            vec![-1.0],
            vec![f64::NAN],
            vec![f64::INFINITY],
        ] {
            assert!(timings(values, 1).is_err());
        }
        assert!(timings(vec![1.0], 0).is_err());
        assert!(timings(vec![f64::MIN_POSITIVE], usize::MAX).is_err());
        let result = timings(vec![4.0, 1.0, 2.0, 3.0], 10).unwrap();
        assert_eq!(result.median_round_seconds, 2.5);
        assert_eq!(result.operations_per_second, 4.0);
    }
}
