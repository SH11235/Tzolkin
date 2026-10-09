//! Separate V2 baseline measurements. Model/source preparation and audits are outside clocks.
use serde::Serialize;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::fs::{self, File};
use std::io::{Read, Write};
use std::path::{Component, Path, PathBuf};
use std::time::Instant;
use tzolkin_ai::Decision;
use tzolkin_ai::features::{EncodedCandidate, FeatureEncoder, PUBLIC_FEATURE_SCHEMA};
use tzolkin_ai::kernel::Kernel;
use tzolkin_ai::model::MAX_ARTIFACT_BYTES;
use tzolkin_ai::policy_dataset::{MAX_SOURCE_BYTES, load_policy_dataset};
use tzolkin_ai::policy_training::{
    PolicyTrainingCheckpoint, load_evaluation_model, validate_new_output_directory,
};
use tzolkin_ai::public_model::{
    LoadedPublicPolicy, PublicPolicyArtifact, PublicPolicyDistribution,
};
use tzolkin_ai::public_native::{
    PreparedPublicPolicy, PublicPolicyHandle, parse_kernel, validate_seats,
};
use tzolkin_ai::replay::{self, GameReplay, ReplaySource};
use tzolkin_core::observation::{Observation, observe};
use tzolkin_core::{GameOptions, GameState, apply_move, create_game_with_options};

const SCHEMA: &str = "tzolkin-public-policy-performance-v1";
const MAX_OBSERVATIONS: usize = 4096;
const MAX_ROWS: usize = 65_536;
const MAX_EVALUATIONS: usize = 2_000_000;
const MAX_REPORT_BYTES: usize = 32 * 1024 * 1024;
const HELP: &str = "tzolkin-public-ml-bench infer --model PATH --checkpoint PATH --dataset DIR --source-dataset DIR --players 3|4 [--kernel scalar|auto|avx2|sse2|neon|simd128 --iterations 1..1000 --rounds 3..15 --verify-only --output NEW_JSON]\ntzolkin-public-ml-bench selfplay --model PATH --checkpoint PATH --dataset DIR --players 3|4 --seats all|SEAT[,SEAT] [--kernel scalar|auto|avx2|sse2|neon|simd128 --seed N --games 1..16 --rounds 3..15 --verify-only --output NEW_JSON]\nDefaults: kernel=scalar, iterations=1, rounds=5, seed=11235, games=1. Base options only. Verify-only runs correctness warmups without clocks or speed claims. Preparation/audits are excluded from timing; V1/default CPU unchanged. Performance is not strength.";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Mode {
    Infer,
    Selfplay,
}
struct Config {
    mode: Mode,
    model: PathBuf,
    checkpoint: PathBuf,
    dataset: PathBuf,
    source_dataset: Option<PathBuf>,
    players: usize,
    kernel: Kernel,
    iterations: usize,
    rounds: usize,
    seed: u32,
    games: usize,
    seats: Vec<usize>,
    verify_only: bool,
    output: Option<PathBuf>,
}
fn config(args: &[String]) -> Result<Config, String> {
    let mode = match args.first().map(String::as_str) {
        Some("infer") => Mode::Infer,
        Some("selfplay") => Mode::Selfplay,
        _ => return Err("Expected infer or selfplay".into()),
    };
    let mut flags = BTreeMap::new();
    let mut verify_only = false;
    let mut index = 1;
    while index < args.len() {
        let flag = args[index].as_str();
        if flag == "--verify-only" {
            if verify_only {
                return Err("Repeated --verify-only".into());
            }
            verify_only = true;
            index += 1;
            continue;
        }
        let common = [
            "--model",
            "--checkpoint",
            "--dataset",
            "--players",
            "--kernel",
            "--rounds",
            "--output",
        ];
        let extra = if mode == Mode::Infer {
            &["--source-dataset", "--iterations"][..]
        } else {
            &["--seed", "--games", "--seats"][..]
        };
        if !common.contains(&flag) && !extra.contains(&flag) {
            return Err(format!("Unknown mode flag {flag}"));
        }
        let value = args
            .get(index + 1)
            .filter(|v| !v.starts_with("--"))
            .ok_or_else(|| format!("Missing value for {flag}"))?;
        if flags.insert(flag, value.as_str()).is_some() {
            return Err(format!("Repeated flag {flag}"));
        }
        index += 2;
    }
    let required = |name| {
        flags
            .get(name)
            .copied()
            .ok_or_else(|| format!("{name} is required"))
    };
    let number = |name, default| {
        flags
            .get(name)
            .copied()
            .unwrap_or(default)
            .parse::<usize>()
            .map_err(|_| format!("Invalid {name}"))
    };
    let players = required("--players")?
        .parse::<usize>()
        .map_err(|_| "Invalid --players")?;
    let iterations = number("--iterations", "1")?;
    let rounds = number("--rounds", "5")?;
    let games = number("--games", "1")?;
    let seed = flags
        .get("--seed")
        .copied()
        .unwrap_or("11235")
        .parse::<u32>()
        .map_err(|_| "Invalid u32 --seed")?;
    if !(3..=4).contains(&players)
        || !(1..=1000).contains(&iterations)
        || !(3..=15).contains(&rounds)
        || !(1..=16).contains(&games)
    {
        return Err("Benchmark bounds:3/4p,iterations1..1000,rounds3..15,games1..16".into());
    }
    seed.checked_add((games - 1) as u32)
        .ok_or("Selfplay seed range overflow")?;
    let seats = if mode == Mode::Selfplay {
        match required("--seats")? {
            "all" => (0..players).collect(),
            list => {
                if list.split(',').count() > players {
                    return Err("Too many seats".into());
                }
                list.split(',')
                    .map(|v| v.parse::<usize>().map_err(|_| "Invalid --seats"))
                    .collect::<Result<Vec<_>, _>>()?
            }
        }
    } else {
        Vec::new()
    };
    if mode == Mode::Selfplay {
        validate_seats(players, &GameOptions::default(), &seats)?;
    }
    let source_dataset = if mode == Mode::Infer {
        Some(PathBuf::from(required("--source-dataset")?))
    } else {
        None
    };
    Ok(Config {
        mode,
        model: required("--model")?.into(),
        checkpoint: required("--checkpoint")?.into(),
        dataset: required("--dataset")?.into(),
        source_dataset,
        players,
        kernel: parse_kernel(flags.get("--kernel").copied().unwrap_or("scalar"))?,
        iterations,
        rounds,
        seed,
        games,
        seats,
        verify_only,
        output: flags.get("--output").map(PathBuf::from),
    })
}
#[derive(Default)]
struct Budget {
    used: usize,
}
impl Budget {
    fn claim(&mut self, count: usize) -> Result<(), String> {
        let next = self
            .used
            .checked_add(count)
            .ok_or("Candidate budget overflow")?;
        if next > MAX_EVALUATIONS {
            return Err("Total benchmark candidate evaluation budget exceeded".into());
        }
        self.used = next;
        Ok(())
    }
}
fn hex(bytes: impl AsRef<[u8]>) -> String {
    bytes.as_ref().iter().map(|b| format!("{b:02x}")).collect()
}
fn hash_json(domain: &[u8], value: &impl Serialize) -> Result<String, String> {
    let mut hash = Sha256::new();
    hash.update(domain);
    hash.update(serde_json::to_vec(value).map_err(|e| e.to_string())?);
    Ok(hex(hash.finalize()))
}
// Root-first checks avoid following a junction/UNC hierarchy before rejecting it.
fn local_path(path: &Path) -> Result<(), String> {
    let text = path.to_string_lossy();
    if text.is_empty()
        || text.contains('\0')
        || text.contains("://")
        || text.starts_with("\\\\")
        || text.starts_with("//")
        || (!path.is_absolute() && matches!(path.components().next(), Some(Component::Prefix(_))))
    {
        return Err("Expected local benchmark path".into());
    }
    let absolute = if path.is_absolute() {
        path.to_path_buf()
    } else {
        std::env::current_dir()
            .map_err(|e| e.to_string())?
            .join(path)
    };
    if absolute.to_string_lossy().starts_with("\\\\")
        || absolute.to_string_lossy().starts_with("//")
    {
        return Err("Network hierarchy rejected".into());
    }
    let mut current = PathBuf::new();
    for component in absolute.components() {
        current.push(component.as_os_str());
        if matches!(component, Component::Prefix(_)) {
            continue;
        }
        match fs::symlink_metadata(&current) {
            Ok(m) if m.file_type().is_symlink() => {
                return Err("Symlink/junction hierarchy rejected".into());
            }
            Ok(_) => {}
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
            Err(e) => return Err(e.to_string()),
        }
    }
    Ok(())
}
fn source_bytes(path: &Path) -> Result<Vec<u8>, String> {
    local_bytes(path, MAX_SOURCE_BYTES)
}
fn local_bytes(path: &Path, maximum: u64) -> Result<Vec<u8>, String> {
    local_path(path)?;
    if !fs::symlink_metadata(path)
        .map_err(|e| e.to_string())?
        .is_file()
    {
        return Err("Source must be regular file".into());
    }
    let mut bytes = Vec::new();
    File::open(path)
        .map_err(|e| e.to_string())?
        .take(maximum + 1)
        .read_to_end(&mut bytes)
        .map_err(|e| e.to_string())?;
    if bytes.len() as u64 > maximum {
        return Err(format!("Local input exceeds{maximum}bytes"));
    }
    Ok(bytes)
}
struct Entry {
    id: String,
    observation: Observation,
    rows: Vec<EncodedCandidate>,
}
struct Corpus {
    entries: Vec<Entry>,
    rows: usize,
    hash: String,
    metadata: Value,
}
fn corpus(path: &Path, players: usize) -> Result<Corpus, String> {
    let dataset = load_policy_dataset(path)?;
    for sample in dataset.iter() {
        sample?;
    } // Do not truncate before EOF integrity checks.
    let mut entries = Vec::new();
    let mut rows_count = 0usize;
    let mut hash = Sha256::new();
    hash.update(b"tzolkin-public-benchmark-corpus-v1\0");
    let mut sources = Vec::new();
    let mut phases = BTreeMap::<String, usize>::new();
    let mut bins = BTreeMap::<String, usize>::new();
    for game in &dataset.manifest().games {
        if game.players != players {
            continue;
        }
        let bytes = source_bytes(&path.join(&game.source_file))?;
        if bytes.len() as u64 != game.source_bytes
            || hex(Sha256::digest(&bytes)) != game.source_sha256
        {
            return Err("Corpus source changed after A2 validation".into());
        }
        let replay: GameReplay = serde_json::from_slice(&bytes).map_err(|e| e.to_string())?;
        let terminal = replay::verify_replay(&replay)?;
        let mut state = create_game_with_options(
            replay.header.names.clone(),
            replay.header.seed,
            replay.header.options.clone(),
        )?;
        for step in &replay.steps {
            let observation = observe(&state, state.current_player)?;
            if observation != step.observation {
                return Err(format!(
                    "Source Observation mismatch {}:{}",
                    game.game_id, step.index
                ));
            }
            rows_count = rows_count
                .checked_add(observation.legal_actions.len())
                .ok_or("Corpus rows overflow")?;
            if entries.len() >= MAX_OBSERVATIONS || rows_count > MAX_ROWS {
                return Err(
                    "Corpus exceeds4096observations/65536rows; no automatic truncation".into(),
                );
            }
            let encoder = FeatureEncoder::new_public(&observation)?;
            let rows = (0..observation.legal_actions.len())
                .map(|i| encoder.encode_legal_tagged(i))
                .collect::<Result<Vec<_>, _>>()?;
            let id = format!("{}:{}:{}", game.game_id, step.index, observation.actor);
            hash.update(
                serde_json::to_vec(&(&id, &observation.legal_actions))
                    .map_err(|e| e.to_string())?,
            );
            for row in &rows {
                hash.update(row.feature_schema().to_le_bytes());
                for value in row.values_for_schema(PUBLIC_FEATURE_SCHEMA)? {
                    hash.update(value.to_bits().to_le_bytes());
                }
            }
            *phases
                .entry(format!("{:?}", observation.phase))
                .or_default() += 1;
            let bin = match rows.len() {
                1 => "1",
                2..=7 => "2..7",
                8..=31 => "8..31",
                32..=127 => "32..127",
                _ => "128+",
            };
            *bins.entry(bin.into()).or_default() += 1;
            entries.push(Entry {
                id,
                observation,
                rows,
            });
            state = apply_move(&state, step.chosen.r#move.clone())?;
        }
        if state != terminal {
            return Err("Corpus terminal reconstruction mismatch".into());
        }
        sources.push(json!({"gameId":game.game_id,"sourceSha256":game.source_sha256,"sourceBytes":game.source_bytes,"samples":game.samples,"familyId":game.family_id}));
    }
    if entries.is_empty() || rows_count == 0 {
        return Err("Empty requested player-count corpus".into());
    }
    Ok(Corpus {
        entries,
        rows: rows_count,
        hash: hex(hash.finalize()),
        metadata: json!({"datasetFingerprint":dataset.manifest().fingerprint,"sources":sources,"phaseCounts":phases,"candidateCountBins":bins,"sampling":"allmatchingsteps-manifest-source-step-order"}),
    })
}
fn distribution_equal(a: &PublicPolicyDistribution, b: &PublicPolicyDistribution) -> bool {
    a.policy_version == b.policy_version
        && a.feature_schema == b.feature_schema
        && a.value_validity == b.value_validity
        && a.logits.len() == b.logits.len()
        && a.probabilities.len() == b.probabilities.len()
        && a.logits
            .iter()
            .zip(&b.logits)
            .all(|(x, y)| x.to_bits() == y.to_bits())
        && a.probabilities
            .iter()
            .zip(&b.probabilities)
            .all(|(x, y)| x.to_bits() == y.to_bits())
}
fn decision_equal(a: &Decision, b: &Decision) -> bool {
    a.actor == b.actor
        && a.observation_key == b.observation_key
        && a.policy_version == b.policy_version
        && a.r#move == b.r#move
        && a.score.to_bits() == b.score.to_bits()
}
fn check_decision(o: &Observation, d: &Decision) -> Result<(), String> {
    if d.actor != o.actor
        || d.observation_key != o.observation_key
        || d.policy_version != tzolkin_ai::public_model::POLICY_VERSION
        || !d.score.is_finite()
        || !o.legal_actions.iter().any(|a| a.r#move == d.r#move)
    {
        return Err("Wrong actor/key/version/score/legal Decision".into());
    }
    Ok(())
}
fn timings(samples: &[f64], operations: usize) -> Result<Value, String> {
    if samples.is_empty() || operations == 0 || samples.iter().any(|n| !n.is_finite() || *n <= 0.0)
    {
        return Err("Positive finite clock samples and nonzero work required".into());
    }
    let mut ordered = samples.to_vec();
    ordered.sort_by(f64::total_cmp);
    let middle = ordered.len() / 2;
    let median = if ordered.len().is_multiple_of(2) {
        let lower = ordered[middle - 1];
        lower + (ordered[middle] - lower) / 2.0
    } else {
        ordered[middle]
    };
    let ns = median * 1e9 / operations as f64;
    let rate = operations as f64 / median;
    if !ns.is_finite() || ns <= 0.0 || !rate.is_finite() || rate <= 0.0 {
        return Err("Nonfinite timing rate".into());
    }
    Ok(
        json!({"roundSeconds":samples,"medianSeconds":median,"minSeconds":ordered[0],"maxSeconds":ordered[ordered.len()-1],"operationsPerRound":operations,"meanNsPerOperation":ns,"operationsPerSecond":rate}),
    )
}
fn timed<T>(
    enabled: bool,
    operation: impl FnOnce() -> Result<T, String>,
) -> Result<(T, Option<f64>), String> {
    let start = enabled.then(Instant::now);
    let output = operation()?;
    Ok((output, start.map(|t| t.elapsed().as_secs_f64())))
}
fn distribution_hash(values: &[PublicPolicyDistribution]) -> Result<String, String> {
    hash_json(
        b"tzolkin-public-benchmark-predictions-v1\0",
        &values
            .iter()
            .map(|p| {
                (
                    &p.policy_version,
                    p.feature_schema,
                    p.logits.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                    p.probabilities
                        .iter()
                        .map(|v| v.to_bits())
                        .collect::<Vec<_>>(),
                    p.value_validity,
                )
            })
            .collect::<Vec<_>>(),
    )
}
fn decision_hash(values: &[Decision]) -> Result<String, String> {
    hash_json(
        b"tzolkin-public-benchmark-decisions-v1\0",
        &values
            .iter()
            .map(|d| {
                (
                    d.actor,
                    &d.observation_key,
                    &d.policy_version,
                    &d.r#move,
                    d.score.to_bits(),
                )
            })
            .collect::<Vec<_>>(),
    )
}
struct Expected {
    predictions: Vec<PublicPolicyDistribution>,
    decisions: Vec<Decision>,
}
fn expected(
    policy: &LoadedPublicPolicy<'_>,
    corpus: &Corpus,
    budget: &mut Budget,
) -> Result<Expected, String> {
    let mut predictions = Vec::with_capacity(corpus.entries.len());
    let mut decisions = Vec::with_capacity(corpus.entries.len());
    for entry in &corpus.entries {
        budget.claim(entry.rows.len())?;
        let oracle = entry
            .rows
            .iter()
            .map(|row| policy.policy_logit(row))
            .collect::<Result<Vec<_>, _>>()?;
        budget.claim(entry.rows.len())?;
        let prediction = policy.predict(&entry.rows)?;
        if !prediction
            .logits
            .iter()
            .zip(&oracle)
            .all(|(a, b)| a.to_bits() == b.to_bits())
            || prediction.logits.len() != oracle.len()
        {
            return Err(format!("Batch/single-row oracle mismatch {}", entry.id));
        }
        budget.claim(entry.rows.len())?;
        let decision = policy.choose_move(&entry.observation)?;
        check_decision(&entry.observation, &decision)?;
        let best = first_best(&prediction.logits)?;
        if decision.r#move != entry.observation.legal_actions[best].r#move
            || decision.score.to_bits() != f64::from(prediction.logits[best]).to_bits()
        {
            return Err(format!(
                "Choose/batch first argmax/score mismatch {}",
                entry.id
            ));
        }
        predictions.push(prediction);
        decisions.push(decision);
    }
    Ok(Expected {
        predictions,
        decisions,
    })
}
fn first_best(logits: &[f32]) -> Result<usize, String> {
    if logits.is_empty() || logits.iter().any(|n| !n.is_finite()) {
        return Err("Expected nonempty finite policy logits".into());
    }
    Ok((1..logits.len()).fold(0, |best, index| {
        if logits[index] > logits[best] {
            index
        } else {
            best
        }
    }))
}
fn within_margin(logits: &[f32], candidate: usize) -> Result<bool, String> {
    let best = first_best(logits)?;
    let candidate = f64::from(*logits.get(candidate).ok_or("Wrong candidate index")?);
    let maximum = f64::from(logits[best]);
    Ok(maximum - candidate <= 5e-5 * (2.0 + maximum.abs() + candidate.abs()))
}
fn infer_pass(
    policy: &LoadedPublicPolicy<'_>,
    corpus: &Corpus,
    expected: &Expected,
    budget: &mut Budget,
    clock: bool,
) -> Result<(Option<f64>, Option<f64>), String> {
    if expected.predictions.len() != corpus.entries.len()
        || expected.decisions.len() != corpus.entries.len()
    {
        return Err("Expected prediction/Decision work count mismatch".into());
    }
    budget.claim(corpus.rows)?;
    let mut predictions = Vec::with_capacity(corpus.entries.len());
    let (_, predict_time) = timed(clock, || {
        for entry in &corpus.entries {
            predictions.push(policy.predict(std::hint::black_box(&entry.rows))?);
        }
        Ok(())
    })?;
    for (index, (actual, reference)) in predictions.iter().zip(&expected.predictions).enumerate() {
        if !distribution_equal(actual, reference) {
            return Err(format!(
                "Timed/warmup prediction drift {}",
                corpus.entries[index].id
            ));
        }
    }
    if predictions.len() != expected.predictions.len() {
        return Err("Prediction work count drift".into());
    }
    drop(predictions);
    budget.claim(corpus.rows)?;
    let mut decisions = Vec::with_capacity(corpus.entries.len());
    let (_, choose_time) = timed(clock, || {
        for entry in &corpus.entries {
            decisions.push(policy.choose_move(std::hint::black_box(&entry.observation))?);
        }
        Ok(())
    })?;
    for (index, (actual, reference)) in decisions.iter().zip(&expected.decisions).enumerate() {
        check_decision(&corpus.entries[index].observation, actual)?;
        if !decision_equal(actual, reference) {
            return Err(format!(
                "Timed/warmup Decision drift {}",
                corpus.entries[index].id
            ));
        }
    }
    if decisions.len() != expected.decisions.len() {
        return Err("Decision work count drift".into());
    }
    Ok((predict_time, choose_time))
}
fn encode_pass(corpus: &Corpus, clock: bool) -> Result<Option<f64>, String> {
    let mut rows = Vec::with_capacity(corpus.entries.len());
    let (_, seconds) = timed(clock, || {
        for entry in &corpus.entries {
            let encoder = FeatureEncoder::new_public(&entry.observation)?;
            rows.push(
                (0..entry.rows.len())
                    .map(|i| encoder.encode_legal_tagged(i))
                    .collect::<Result<Vec<_>, _>>()?,
            );
        }
        Ok(())
    })?;
    if rows.len() != corpus.entries.len() {
        return Err("Encoding work count drift".into());
    }
    for (entry, actual) in corpus.entries.iter().zip(&rows) {
        if actual.len() != entry.rows.len() {
            return Err(format!("Encoding count drift {}", entry.id));
        }
        for (a, b) in actual.iter().zip(&entry.rows) {
            if a.feature_schema() != b.feature_schema()
                || !a
                    .values_for_schema(PUBLIC_FEATURE_SCHEMA)?
                    .iter()
                    .zip(b.values_for_schema(PUBLIC_FEATURE_SCHEMA)?)
                    .all(|(x, y)| x.to_bits() == y.to_bits())
            {
                return Err(format!("Encoding bit drift {}", entry.id));
            }
        }
    }
    Ok(seconds)
}
fn infer(
    config: &Config,
    scalar: &LoadedPublicPolicy<'_>,
    selected: &LoadedPublicPolicy<'_>,
    budget: &mut Budget,
) -> Result<Value, String> {
    let corpus = corpus(
        config.source_dataset.as_ref().ok_or("Missing corpus")?,
        config.players,
    )?;
    let passes = 6usize
        + 4
        + if config.verify_only {
            0
        } else {
            config
                .rounds
                .checked_mul(config.iterations)
                .and_then(|n| n.checked_mul(4))
                .ok_or("Work budget overflow")?
        };
    if corpus
        .rows
        .checked_mul(passes)
        .is_none_or(|n| n > MAX_EVALUATIONS)
    {
        return Err("Planned inference candidate workload exceeds total budget".into());
    }
    let reference = expected(scalar, &corpus, budget)?;
    let actual = expected(selected, &corpus, budget)?;
    let mut max_error = 0f64;
    let mut max_probability_error = 0f64;
    let mut numerical = true;
    let mut moves_matched = 0usize;
    let mut near_ties = 0usize;
    for (index, (a, b)) in reference
        .predictions
        .iter()
        .zip(&actual.predictions)
        .enumerate()
    {
        for (x, y) in a.logits.iter().zip(&b.logits) {
            let error = (f64::from(*x) - f64::from(*y)).abs();
            max_error = max_error.max(error);
            numerical &= error <= 5e-5 * (1.0 + f64::from(*x).abs());
        }
        for (x, y) in a.probabilities.iter().zip(&b.probabilities) {
            let error = (f64::from(*x) - f64::from(*y)).abs();
            max_probability_error = max_probability_error.max(error);
            numerical &= error <= 5e-5;
        }
        let mut sorted = a.logits.clone();
        sorted.sort_by(|a, b| b.total_cmp(a));
        if sorted.len() > 1 && within_margin(&sorted, 1)? {
            near_ties += 1;
        }
        let matched = reference.decisions[index].r#move == actual.decisions[index].r#move;
        moves_matched += usize::from(matched);
        if !matched {
            let chosen = corpus.entries[index]
                .observation
                .legal_actions
                .iter()
                .position(|legal| legal.r#move == actual.decisions[index].r#move)
                .ok_or("Selected backend chose outside ordered legal batch")?;
            if !within_margin(&a.logits, chosen)? {
                return Err(format!(
                    "Non-near-tie backend move mismatch {}",
                    corpus.entries[index].id
                ));
            }
        }
    }
    if !numerical {
        return Err("Backend policy logits/probabilities outside numerical tolerance".into());
    }
    encode_pass(&corpus, false)?;
    infer_pass(scalar, &corpus, &reference, budget, false)?;
    infer_pass(selected, &corpus, &actual, budget, false)?;
    let mut encoding = Vec::new();
    let mut sp = Vec::new();
    let mut sc = Vec::new();
    let mut ap = Vec::new();
    let mut ac = Vec::new();
    if !config.verify_only {
        for round in 0..config.rounds {
            let mut e = 0.;
            let mut p = [0.; 2];
            let mut c = [0.; 2];
            for _ in 0..config.iterations {
                e += encode_pass(&corpus, true)?.ok_or("Missing encode duration")?;
                for arm in if round % 2 == 0 { [0, 1] } else { [1, 0] } {
                    let (policy, expected) = if arm == 0 {
                        (scalar, &reference)
                    } else {
                        (selected, &actual)
                    };
                    let (pt, ct) = infer_pass(policy, &corpus, expected, budget, true)?;
                    p[arm] += pt.ok_or("Missing predict duration")?;
                    c[arm] += ct.ok_or("Missing choose duration")?;
                }
            }
            encoding.push(e);
            sp.push(p[0]);
            ap.push(p[1]);
            sc.push(c[0]);
            ac.push(c[1]);
        }
    }
    let summary = |samples: &[f64], ops| {
        if config.verify_only {
            Ok(Value::Null)
        } else {
            timings(samples, ops)
        }
    };
    Ok(
        json!({"corpusHash":corpus.hash,"corpus":corpus.metadata,"observations":corpus.entries.len(),"candidateRows":corpus.rows,
        "outputHashes":{"scalarPredict":distribution_hash(&reference.predictions)?,"selectedPredict":distribution_hash(&actual.predictions)?,"scalarChoose":decision_hash(&reference.decisions)?,"selectedChoose":decision_hash(&actual.decisions)?},
        "parity":{"maxPolicyAbsError":max_error,"maxProbabilityAbsError":max_probability_error,"numericalWithinTolerance":numerical,"policyTolerance":"5e-5*(1+abs(scalar logit))","probabilityTolerance":5e-5,"movesMatched":moves_matched,"decisions":corpus.entries.len(),"nearTieBatches":near_ties},
        "featureEncoding":summary(&encoding,corpus.rows*config.iterations)?,"scalarPredict":summary(&sp,corpus.entries.len()*config.iterations)?,"selectedPredict":summary(&ap,corpus.entries.len()*config.iterations)?,"scalarChoose":summary(&sc,corpus.entries.len()*config.iterations)?,"selectedChoose":summary(&ac,corpus.entries.len()*config.iterations)?,
        "predictUnit":"one complete legal batch","chooseUnit":"one Observation","meanCandidateNs":{"scalar":if config.verify_only{Value::Null}else{timings(&sp,corpus.rows*config.iterations)?["meanNsPerOperation"].clone()},"selected":if config.verify_only{Value::Null}else{timings(&ap,corpus.rows*config.iterations)?["meanNsPerOperation"].clone()}},"allWarmupTimedOutputsAudited":true}),
    )
}

#[derive(Default)]
struct Trace {
    attempts: usize,
    candidate_rows: usize,
    charged_rows: usize,
    decisions: Vec<Decision>,
    candidates: Vec<usize>,
    phases: Vec<String>,
    learned: Vec<bool>,
}
impl Trace {
    fn same(&self, other: &Self, scores: bool) -> bool {
        self.attempts == other.attempts
            && self.candidate_rows == other.candidate_rows
            && self.charged_rows == other.charged_rows
            && self.candidates == other.candidates
            && self.phases == other.phases
            && self.learned == other.learned
            && self.decisions.len() == other.decisions.len()
            && self.decisions.iter().zip(&other.decisions).all(|(a, b)| {
                if scores {
                    decision_equal(a, b)
                } else {
                    a.actor == b.actor
                        && a.observation_key == b.observation_key
                        && a.policy_version == b.policy_version
                        && a.r#move == b.r#move
                }
            })
    }
    fn output_hash(&self) -> Result<String, String> {
        decision_hash(&self.decisions)
    }
    fn workload_hash(&self) -> Result<String, String> {
        hash_json(
            b"tzolkin-public-benchmark-native-work-v1\0",
            &(
                self.attempts,
                self.candidate_rows,
                self.charged_rows,
                &self.candidates,
                &self.phases,
                &self.learned,
            ),
        )
    }
    // Scores are logits for learned seats. Arithmetic may differ across SIMD backends,
    // while each backend's repetitions must still match all score bits exactly.
    fn cross_backend_scores(&self, other: &Self) -> Option<f64> {
        if !self.same(other, false) {
            return None;
        }
        let mut maximum = 0f64;
        for (a, b) in self.decisions.iter().zip(&other.decisions) {
            let error = (a.score - b.score).abs();
            if !error.is_finite() || error > 5e-5 * (1.0 + a.score.abs()) {
                return None;
            }
            maximum = maximum.max(error);
        }
        Some(maximum)
    }
}
fn native_game(
    policy: &PublicPolicyHandle<'_>,
    config: &Config,
    seed: u32,
    record: bool,
    budget: &mut Budget,
    trace: &mut Trace,
) -> Result<(GameState, usize, Option<GameReplay>), String> {
    validate_seats(config.players, &GameOptions::default(), &config.seats)?;
    let policies = (0..config.players)
        .map(|seat| {
            if config.seats.contains(&seat) {
                policy.provenance().clone()
            } else {
                tzolkin_ai::experiment::heuristic_seat()
            }
        })
        .collect();
    replay::play_game_using_fast(
        config.players,
        seed,
        GameOptions::default(),
        record,
        ReplaySource::PolicySelfPlay { policies },
        |o| {
            trace.attempts = trace
                .attempts
                .checked_add(1)
                .ok_or("Native attempt count overflow")?;
            trace.candidate_rows = trace
                .candidate_rows
                .checked_add(o.legal_actions.len())
                .ok_or("Native candidate count overflow")?;
            let learned = config.seats.contains(&o.actor);
            if learned {
                budget.claim(o.legal_actions.len())?;
                trace.charged_rows = trace
                    .charged_rows
                    .checked_add(o.legal_actions.len())
                    .ok_or("Native charged count overflow")?;
            }
            let decision = if learned {
                policy.choose_move(o)?
            } else {
                tzolkin_ai::choose_move(o)?
            };
            if decision.actor != o.actor
                || decision.observation_key != o.observation_key
                || decision.policy_version
                    != if learned {
                        tzolkin_ai::public_model::POLICY_VERSION
                    } else {
                        tzolkin_ai::POLICY_VERSION
                    }
                || !decision.score.is_finite()
                || !o.legal_actions.iter().any(|a| a.r#move == decision.r#move)
            {
                return Err("Native Decision actor/key/version/legal/finite mismatch".into());
            }
            trace.decisions.push(decision.clone());
            trace.candidates.push(o.legal_actions.len());
            trace.phases.push(format!("{:?}", o.phase));
            trace.learned.push(learned);
            Ok(decision)
        },
    )
}
fn game_metadata(
    seed: u32,
    backend: &str,
    played: &Result<(GameState, usize, Option<GameReplay>), String>,
    trace: &Trace,
) -> Result<Value, String> {
    let terminal = played.as_ref().ok();
    Ok(
        json!({"seed":seed,"backend":backend,"complete":terminal.is_some(),"decisions":terminal.map(|v|v.1),"attemptedDecisions":trace.attempts,"attemptedCandidateRows":trace.candidate_rows,"capturedDecisions":trace.decisions.len(),"learnedCandidateBudgetUsed":trace.charged_rows,"error":played.as_ref().err(),
        "trajectoryHash":trace.output_hash()?,"workloadHash":trace.workload_hash()?,"finalState":terminal.map(|v|replay::state_key(&v.0)).transpose()?,"finalScores":terminal.map(|v|&v.0.final_scores),
        "terminalPublicPlayers":terminal.map(|v|observe(&v.0,v.0.current_player).map(|o|o.players)).transpose()?}),
    )
}
fn selfplay(
    config: &Config,
    scalar: &PublicPolicyHandle<'_>,
    selected: &PublicPolicyHandle<'_>,
    budget: &mut Budget,
) -> Result<(Value, bool), String> {
    let mut games = Vec::new();
    let mut all_complete = true;
    let mut comparable = true;
    let mut round_times = [Vec::new(), Vec::new()];
    for game in 0..config.games {
        let seed = config.seed + game as u32;
        let mut traces = [Trace::default(), Trace::default()];
        let policies = [scalar, selected];
        let mut reference_keys = [String::new(), String::new()];
        let mut failures = false;
        let mut warmups = Vec::new();
        for arm in 0..2 {
            let played = native_game(policies[arm], config, seed, true, budget, &mut traces[arm]);
            if let Ok((state, count, Some(record))) = &played {
                if *count != traces[arm].decisions.len()
                    || replay::verify_replay(record).map_err(|e| {
                        format!(
                            "Seed {seed} backend {} warmup verify: {e}",
                            policies[arm].backend()
                        )
                    })? != *state
                {
                    return Err(format!(
                        "Seed {seed} backend {} warmup verification/count mismatch",
                        policies[arm].backend()
                    ));
                }
                reference_keys[arm] = replay::state_key(state)?;
            } else {
                failures = true;
            }
            warmups.push(game_metadata(
                seed,
                policies[arm].backend(),
                &played,
                &traces[arm],
            )?);
        }
        if failures {
            all_complete = false;
            comparable = false;
            games.push(json!({"seed":seed,"warmups":warmups,"timed":[],"comparable":false,"error":"Warmup incomplete; no terminal zero/timing extrapolation"}));
            continue;
        }
        let score_error = traces[0].cross_backend_scores(&traces[1]);
        let same = traces[0].same(&traces[1], false)
            && reference_keys[0] == reference_keys[1]
            && score_error.is_some();
        comparable &= same;
        let mut timed_results = Vec::new();
        let mut local_times = [Vec::new(), Vec::new()];
        if !config.verify_only {
            for round in 0..config.rounds {
                for arm in if round % 2 == 0 { [0, 1] } else { [1, 0] } {
                    let mut actual = Trace::default();
                    let start = Instant::now();
                    let played =
                        native_game(policies[arm], config, seed, false, budget, &mut actual);
                    let elapsed = start.elapsed().as_secs_f64();
                    let matches = played.as_ref().is_ok_and(|(state, count, _)| {
                        *count == actual.decisions.len()
                            && replay::state_key(state).is_ok_and(|k| k == reference_keys[arm])
                    }) && actual.same(&traces[arm], true);
                    let mut metadata =
                        game_metadata(seed, policies[arm].backend(), &played, &actual)?;
                    metadata["round"] = round.into();
                    metadata["outputsMatchWarmup"] = matches.into();
                    metadata["elapsedSeconds"] = elapsed.into();
                    if !matches && played.is_ok() {
                        metadata["error"] = "Completed game drifted from its backend warmup".into();
                    }
                    timed_results.push(metadata);
                    if !matches {
                        all_complete = false;
                        comparable = false;
                    } else {
                        local_times[arm].push(elapsed);
                    }
                }
            }
        }
        if !config.verify_only && local_times.iter().all(|times| times.len() == config.rounds) {
            for arm in 0..2 {
                if round_times[arm].is_empty() {
                    round_times[arm] = vec![0.; config.rounds];
                }
                for (round, elapsed) in local_times[arm].iter().enumerate() {
                    round_times[arm][round] += elapsed;
                }
            }
        }
        games.push(json!({"seed":seed,"warmups":warmups,"timed":timed_results,"comparable":same,"crossBackendScoresWithinTolerance":score_error.is_some(),"maxDecisionScoreAbsError":score_error}));
    }
    let summaries = if config.verify_only || !all_complete {
        json!({"scalar":null,"selected":null})
    } else {
        json!({"scalar":timings(&round_times[0],config.games)?,"selected":timings(&round_times[1],config.games)?})
    };
    let ratio = if config.verify_only || !all_complete || !comparable {
        Value::Null
    } else {
        let a = summaries["scalar"]["medianSeconds"]
            .as_f64()
            .ok_or("Missing scalar timing")?;
        let b = summaries["selected"]["medianSeconds"]
            .as_f64()
            .ok_or("Missing selected timing")?;
        let ratio = a / b;
        if !ratio.is_finite() || ratio <= 0. {
            return Err("Invalid native ratio".into());
        }
        ratio.into()
    };
    Ok((
        json!({"games":games,"allComplete":all_complete,"comparable":comparable,"timings":summaries,"nativeSpeedup":ratio,"unit":"one native game;trace capture included;verification/hash excluded"}),
        all_complete,
    ))
}
fn publish(path: &Path, report: &Value) -> Result<(), String> {
    validate_new_output_directory(path)?;
    local_path(path)?;
    let bytes = serde_json::to_vec(report).map_err(|e| e.to_string())?;
    if bytes.len() > MAX_REPORT_BYTES {
        return Err("Benchmark report exceeds32MiB".into());
    }
    let parent = path
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    let name = path
        .file_name()
        .ok_or("Output filename required")?
        .to_string_lossy();
    let temp = parent.join(format!(".{name}.{}.tmp", std::process::id()));
    let mut created = false;
    let result = (|| {
        let mut file = File::options()
            .write(true)
            .create_new(true)
            .open(&temp)
            .map_err(|e| e.to_string())?;
        created = true;
        file.write_all(&bytes)
            .and_then(|_| file.sync_all())
            .map_err(|e| e.to_string())?;
        drop(file);
        fs::hard_link(&temp, path).map_err(|e| e.to_string())
    })();
    if created {
        let _ = fs::remove_file(&temp);
    }
    result
}
fn run(config: &Config) -> Result<(Value, bool), String> {
    if let Some(path) = &config.output {
        validate_new_output_directory(path)?;
    }
    config.kernel.resolve()?;
    let model = load_evaluation_model(&config.model)?;
    let checkpoint = PolicyTrainingCheckpoint::load(&config.checkpoint)?;
    if model != checkpoint.model {
        return Err("Explicit V2 artifact differs from checkpoint model".into());
    }
    let model_bytes = local_bytes(&config.model, MAX_ARTIFACT_BYTES as u64)?;
    let observed_model: PublicPolicyArtifact =
        serde_json::from_slice(&model_bytes).map_err(|e| e.to_string())?;
    observed_model.validate()?;
    let checkpoint_bytes = local_bytes(&config.checkpoint, MAX_ARTIFACT_BYTES as u64)?;
    let observed_checkpoint: PolicyTrainingCheckpoint =
        serde_json::from_slice(&checkpoint_bytes).map_err(|e| e.to_string())?;
    observed_checkpoint.validate()?;
    if observed_model != model || observed_checkpoint != checkpoint {
        return Err("Model/checkpoint file changed during preparation".into());
    }
    let dataset = load_policy_dataset(&config.dataset)?;
    let scalar = PreparedPublicPolicy::new(checkpoint.clone(), &dataset, Kernel::Scalar)?;
    let selected = PreparedPublicPolicy::new(checkpoint.clone(), &dataset, config.kernel)?;
    let scalar_handle = scalar.handle()?;
    let selected_handle = selected.handle()?;
    let scalar_loaded = LoadedPublicPolicy::new(scalar.model())?;
    let selected_loaded = LoadedPublicPolicy::with_kernel(selected.model(), config.kernel)?;
    if scalar_loaded.backend() != scalar_handle.backend()
        || selected_loaded.backend() != selected_handle.backend()
    {
        return Err("Resolved backend mismatch".into());
    }
    let mut budget = Budget::default();
    let mut report = json!({"schema":SCHEMA,"mode":format!("{:?}",config.mode).to_lowercase(),"verifyOnly":config.verify_only,"buildProfile":if cfg!(debug_assertions){"debug"}else{"optimized"},"targetArch":std::env::consts::ARCH,"targetOs":std::env::consts::OS,
        "requestedKernel":format!("{:?}",config.kernel),"scalarBackend":scalar.backend(),"backend":selected.backend(),"modelChecksum":model.checksum,"checkpointChecksum":checkpoint.checksum,"modelFileSha256":hex(Sha256::digest(&model_bytes)),"checkpointFileSha256":hex(Sha256::digest(&checkpoint_bytes)),"datasetFingerprint":dataset.manifest().fingerprint,"modelVersion":model.model_version,"policyVersion":model.policy_version,"featureSchema":model.feature_schema,"catalogHash":model.catalog_hash,"players":config.players,"seats":config.seats,"seed":config.seed,"plannedGames":config.games,"iterations":config.iterations,"rounds":config.rounds,
        "success":false,"error":null,"measurements":null,"strengthMeasured":false,"humanReplayCorpus":false,"automaticAdoption":false,"valueValidity":"unavailablePolicyOnly","timedOutputStorage":"onepass;captureallocation/pushincluded;audit/hash/destructionexcluded"});
    let result = match config.mode {
        Mode::Infer => {
            infer(config, &scalar_loaded, &selected_loaded, &mut budget).map(|v| (v, true))
        }
        Mode::Selfplay => selfplay(config, &scalar_handle, &selected_handle, &mut budget),
    };
    let success = match result {
        Ok((measurements, success)) => {
            report["measurements"] = measurements;
            if !success {
                report["error"] =
                    "Native benchmark incomplete or outputs drifted; inspect per-seed results"
                        .into();
            }
            success
        }
        Err(error) => {
            report["error"] = error.into();
            false
        }
    };
    report["success"] = success.into();
    report["candidateEvaluationBudgetUsed"] = budget.used.into();
    report["candidateEvaluationAccounting"] = "Benchmark oracle/expected/warmup/timed batch budget only; qualification/final-loss recomputation excluded. Full legal batch charged before forward; may exceed completed forwards on error".into();
    report["maxCandidateEvaluations"] = MAX_EVALUATIONS.into();
    if success
        && let Some(path) = &config.output
        && let Err(error) = publish(path, &report)
    {
        report["success"] = false.into();
        report["publication"] = json!({"success":false,"error":error,"path":path});
        return Ok((report, false));
    }
    Ok((report, success))
}
fn main() {
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    if args == ["--help"] {
        println!("{HELP}");
        return;
    }
    match config(&args).and_then(|c| run(&c)) {
        Ok((report, success)) => {
            match serde_json::to_vec(&report) {
                Ok(bytes) if bytes.len() <= MAX_REPORT_BYTES => {
                    let mut stdout = std::io::stdout().lock();
                    if stdout
                        .write_all(&bytes)
                        .and_then(|_| stdout.write_all(b"\n"))
                        .is_err()
                    {
                        std::process::exit(1);
                    }
                }
                _ => {
                    eprintln!("Invalid/oversized benchmark report");
                    std::process::exit(1);
                }
            }
            if !success {
                std::process::exit(1);
            }
        }
        Err(error) => {
            eprintln!("{error}\n{HELP}");
            std::process::exit(1);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};
    use tzolkin_ai::public_model::PublicPolicyArtifact;

    fn args(mode: &str, extra: &[&str]) -> Vec<String> {
        [
            mode,
            "--model",
            "model",
            "--checkpoint",
            "checkpoint",
            "--dataset",
            "dataset",
            "--players",
            "3",
        ]
        .into_iter()
        .chain(extra.iter().copied())
        .map(String::from)
        .collect()
    }
    fn single_corpus() -> Corpus {
        let state =
            tzolkin_core::create_game(vec!["A".into(), "B".into(), "C".into()], 11235, false)
                .unwrap();
        let observation = observe(&state, state.current_player).unwrap();
        let encoder = FeatureEncoder::new_public(&observation).unwrap();
        let rows = (0..observation.legal_actions.len())
            .map(|i| encoder.encode_legal_tagged(i).unwrap())
            .collect::<Vec<_>>();
        Corpus {
            rows: rows.len(),
            entries: vec![Entry {
                id: "test:0:0".into(),
                observation,
                rows,
            }],
            hash: String::new(),
            metadata: Value::Null,
        }
    }
    #[test]
    fn parser_is_closed_bounded_and_kernel_defaults_scalar() {
        let c = config(&args(
            "infer",
            &["--source-dataset", "sources", "--verify-only"],
        ))
        .unwrap();
        assert!(c.verify_only);
        assert_eq!(c.kernel, Kernel::Scalar);
        assert_eq!((c.iterations, c.rounds), (1, 5));
        for extra in [
            vec![],
            vec!["--source-dataset", "x", "--iterations", "0"],
            vec!["--source-dataset", "x", "--rounds", "16"],
            vec!["--source-dataset", "x", "--raw-features", "x"],
            vec!["--source-dataset", "x", "--seats", "all"],
            vec!["--source-dataset", "x", "--verify-only", "--verify-only"],
            vec![
                "--source-dataset",
                "x",
                "--kernel",
                "auto",
                "--kernel",
                "scalar",
            ],
        ] {
            assert!(config(&args("infer", &extra)).is_err());
        }
        for extra in [
            vec![],
            vec!["--seats", "0,0"],
            vec!["--seats", "3"],
            vec!["--seats", "all", "--games", "17"],
            vec!["--seats", "all", "--seed", "4294967295", "--games", "2"],
        ] {
            assert!(config(&args("selfplay", &extra)).is_err());
        }
        let c = config(&args("selfplay", &["--seats", "0,2", "--verify-only"])).unwrap();
        assert_eq!(c.seats, [0, 2]);
    }
    #[test]
    fn total_budget_and_nonfinite_zero_clock_samples_fail_closed() {
        let mut budget = Budget::default();
        budget.claim(MAX_EVALUATIONS).unwrap();
        assert!(budget.claim(1).is_err());
        assert_eq!(budget.used, MAX_EVALUATIONS);
        assert!(budget.claim(usize::MAX).is_err());
        for sample in [
            0.,
            -0.,
            -1.,
            f64::NAN,
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::MAX,
            f64::from_bits(1),
        ] {
            assert!(timings(&[sample], 1).is_err());
        }
        assert!(timings(&[], 1).is_err());
        assert!(timings(&[1.], 0).is_err());
        let valid = timings(&[3., 1., 2.], 10).unwrap();
        assert_eq!(valid["medianSeconds"], 2.);
        let even = timings(&[100., 2., 1., 9.], 10).unwrap();
        assert_eq!(even["medianSeconds"], 5.5);
        let (result, elapsed) = timed(false, || Ok(42)).unwrap();
        assert_eq!(result, 42);
        assert!(elapsed.is_none());
    }
    #[test]
    fn all_prediction_probability_and_decision_bits_are_audited_without_clocks() {
        let artifact = PublicPolicyArtifact::new(7).unwrap();
        let policy = LoadedPublicPolicy::new(&artifact).unwrap();
        let corpus = single_corpus();
        let mut budget = Budget::default();
        let mut oracle = expected(&policy, &corpus, &mut budget).unwrap();
        assert_eq!(
            infer_pass(&policy, &corpus, &oracle, &mut budget, false).unwrap(),
            (None, None)
        );
        assert!(encode_pass(&corpus, false).unwrap().is_none());
        let old = oracle.predictions[0].probabilities[0];
        oracle.predictions[0].probabilities[0] = f32::from_bits(old.to_bits() ^ 1);
        assert!(
            infer_pass(&policy, &corpus, &oracle, &mut budget, false)
                .unwrap_err()
                .contains("prediction drift")
        );
        oracle.predictions[0].probabilities[0] = old;
        let old = oracle.decisions[0].score;
        oracle.decisions[0].score = f64::from_bits(old.to_bits() ^ 1);
        assert!(
            infer_pass(&policy, &corpus, &oracle, &mut budget, false)
                .unwrap_err()
                .contains("Decision drift")
        );
        oracle.decisions.clear();
        assert!(
            infer_pass(&policy, &corpus, &oracle, &mut budget, false)
                .unwrap_err()
                .contains("work count")
        );
    }
    #[test]
    fn actor_key_policy_legality_and_cross_backend_trace_are_separate_checks() {
        let corpus = single_corpus();
        let artifact = PublicPolicyArtifact::new(7).unwrap();
        let decision = LoadedPublicPolicy::new(&artifact)
            .unwrap()
            .choose_move(&corpus.entries[0].observation)
            .unwrap();
        let o = &corpus.entries[0].observation;
        for field in 0..4 {
            let mut wrong = decision.clone();
            match field {
                0 => wrong.actor += 1,
                1 => wrong.observation_key.push('x'),
                2 => wrong.policy_version.push('x'),
                _ => wrong.score = f64::NAN,
            }
            assert!(check_decision(o, &wrong).is_err());
        }
        let make = || Trace {
            decisions: vec![decision.clone()],
            candidates: vec![o.legal_actions.len()],
            phases: vec![format!("{:?}", o.phase)],
            learned: vec![true],
            ..Trace::default()
        };
        let a = make();
        let mut b = make();
        b.decisions[0].score += 1e-7;
        assert!(a.same(&b, false));
        assert!(!a.same(&b, true));
        assert!(a.cross_backend_scores(&b).is_some());
        b.decisions[0].score += 1.;
        assert!(a.cross_backend_scores(&b).is_none());
        b = make();
        b.candidates[0] += 1;
        assert!(!a.same(&b, false));
        b = make();
        b.learned[0] = false;
        assert!(!a.same(&b, false));
        b = make();
        b.decisions[0].score = -0.;
        let mut z = make();
        z.decisions[0].score = 0.;
        assert!(!z.same(&b, true));
        assert_ne!(z.output_hash().unwrap(), b.output_hash().unwrap());
    }
    #[test]
    fn failures_keep_attempted_work_but_terminal_fields_null() {
        let corpus = single_corpus();
        let decision = tzolkin_ai::choose_move(&corpus.entries[0].observation).unwrap();
        let trace = Trace {
            attempts: 1,
            candidate_rows: 6,
            decisions: vec![decision],
            candidates: vec![6],
            phases: vec!["Setup".into()],
            learned: vec![false],
            ..Trace::default()
        };
        let report =
            game_metadata(11235, "scalar", &Err("4000-decision cap".into()), &trace).unwrap();
        assert_eq!(report["complete"], false);
        assert_eq!(report["attemptedDecisions"], 1);
        assert_eq!(report["attemptedCandidateRows"], 6);
        assert_eq!(report["error"], "4000-decision cap");
        for key in [
            "decisions",
            "finalState",
            "finalScores",
            "terminalPublicPlayers",
        ] {
            assert!(report[key].is_null());
        }
    }
    #[test]
    fn near_tie_cannot_admit_a_far_third_candidate_and_first_tie_is_stable() {
        assert_eq!(first_best(&[1., 1., 0.]).unwrap(), 0);
        assert!(within_margin(&[1., 1. - 1e-7, -2.], 1).unwrap());
        assert!(!within_margin(&[1., 1. - 1e-7, -2.], 2).unwrap());
        assert!(within_margin(&[1.], 1).is_err());
        assert!(first_best(&[]).is_err());
        assert!(first_best(&[f32::NAN]).is_err());
    }
    #[test]
    fn reports_publish_exclusively_and_reject_network_paths() {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let dir = std::env::temp_dir().join(format!(
            "tzolkin-v2-bench-publish-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir(&dir).unwrap();
        let file = dir.join("result.json");
        publish(&file, &json!({"test":1})).unwrap();
        let old = fs::read(&file).unwrap();
        assert!(publish(&file, &json!({"test":2})).is_err());
        assert_eq!(old, fs::read(&file).unwrap());
        for path in [
            "https://example.invalid/input",
            "//server/share",
            "\\\\server\\share",
        ] {
            assert!(local_path(Path::new(path)).is_err());
        }
        fs::remove_dir_all(&dir).unwrap();
    }
}
