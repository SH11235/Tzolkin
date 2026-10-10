//! Bounded, opt-in native stochastic collection and mechanical likelihood audit.
//!
//! The environment seed belongs to trusted replay metadata. Only actual actor
//! Observations enter the fresh Scalar sampler. This format is intentionally
//! separate from GameReplay, SeatPolicy and all existing training admission.
//! Audit reproduces transitions and sampling; it authenticates neither the
//! producer nor independent sampling/environment seed origins.
use tzolkin_core::observation::{MOVE_SCHEMA, OBSERVATION_SCHEMA, Observation, observe};
use tzolkin_core::{GameMove, GameOptions, GameState, Phase, apply_move, create_game_with_options};

use serde::Serialize;

use crate::dataset::seed_family_id;
use crate::public_native::PublicPolicyHandle;
use crate::public_policy_repeat::REPEATED_SAMPLING_VERSION;
use crate::public_rl_native::{RepeatedPublicRlHandle, UpdatedPublicRlHandle};
use crate::public_stochastic::{
    MAX_RESERVED_CANDIDATE_ROWS, MAX_SAMPLES, RL_SAMPLING_VERSION, RNG_VERSION,
    RepeatedStochasticSession, RlStochasticSession, SAMPLING_VERSION, SampledDecision,
    SamplingCounters, SamplingStreamIdentity, StochasticSession, UNIFORM_MIXTURE,
};
use crate::public_stochastic_record::{
    AttemptStage, CALLBACK_RESERVE, Callback, Config, Counts, FINAL_RESERVE, Failure,
    FailureReason, FinalScoreWire, Header, MAX_RECORD_BYTES, MAX_SOURCE_BYTES, MAX_TRACE_BYTES,
    OpponentChoice, RECORD_SCHEMA, Record, SOURCE_KIND, Sample, TRACE_RESERVE, Terminal, hex64,
    serialized_size,
};
use crate::replay::{RULES_BASELINE, RULES_VERSION, catalog_hash, state_key};

/// Serialized payload bounds, not a fixed resident-memory limit. Each game and
/// its record are retained in memory. There is no batch/IO publisher in this unit.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CollectionLimits {
    callbacks: usize,
    candidate_rows: usize,
    source_bytes: usize,
    trace_bytes: usize,
}
impl CollectionLimits {
    pub fn new(
        callbacks: usize,
        candidate_rows: usize,
        source_bytes: usize,
        trace_bytes: usize,
    ) -> Result<Self, String> {
        if !(1..=MAX_SAMPLES).contains(&callbacks)
            || !(1..=MAX_RESERVED_CANDIDATE_ROWS).contains(&candidate_rows)
            || !(64 * 1024..=MAX_SOURCE_BYTES).contains(&source_bytes)
            || !(TRACE_RESERVE..=MAX_TRACE_BYTES).contains(&trace_bytes)
        {
            return Err("Invalid stochastic collection limits".into());
        }
        Ok(Self {
            callbacks,
            candidate_rows,
            source_bytes,
            trace_bytes,
        })
    }
    pub fn max_callbacks(&self) -> usize {
        self.callbacks
    }
    pub fn max_candidate_rows(&self) -> usize {
        self.candidate_rows
    }
    pub fn max_source_bytes(&self) -> usize {
        self.source_bytes
    }
    pub fn max_trace_bytes(&self) -> usize {
        self.trace_bytes
    }
}
impl Default for CollectionLimits {
    fn default() -> Self {
        Self {
            callbacks: MAX_SAMPLES,
            candidate_rows: MAX_RESERVED_CANDIDATE_ROWS,
            source_bytes: MAX_SOURCE_BYTES,
            trace_bytes: MAX_TRACE_BYTES,
        }
    }
}

/// No raw options/observation/model constructor. All four base options are false.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NativeStochasticConfig {
    players: usize,
    environment_seed: u32,
    sampling_identity: SamplingStreamIdentity,
    limits: CollectionLimits,
}
impl NativeStochasticConfig {
    pub fn new(
        players: usize,
        environment_seed: u32,
        sampling_identity: SamplingStreamIdentity,
        limits: CollectionLimits,
    ) -> Result<Self, String> {
        if !(3..=4).contains(&players) || sampling_identity.players() != players {
            return Err("Stochastic native collection requires matching base 3/4p identity".into());
        }
        Ok(Self {
            players,
            environment_seed,
            sampling_identity,
            limits,
        })
    }
    pub fn players(&self) -> usize {
        self.players
    }
    pub fn environment_seed(&self) -> u32 {
        self.environment_seed
    }
    pub fn sampling_identity(&self) -> &SamplingStreamIdentity {
        &self.sampling_identity
    }
    pub fn limits(&self) -> &CollectionLimits {
        &self.limits
    }
}

/// Sealed collection result, including failed prefixes. No caller can set a
/// completion bit, inject a raw trace, or deserialize this type.
///
/// ```compile_fail
/// use tzolkin_ai::public_stochastic_native::CollectedStochasticGame;
/// let _: CollectedStochasticGame = serde_json::from_str("{}").unwrap();
/// ```
/// ```compile_fail
/// use tzolkin_ai::{public_stochastic_native::CollectedStochasticGame, replay::GameReplay};
/// fn legacy_source(game: CollectedStochasticGame) -> GameReplay { game.into() }
/// ```
pub struct CollectedStochasticGame {
    record: Record,
}
impl CollectedStochasticGame {
    pub fn complete(&self) -> bool {
        self.record.terminal.is_some() && self.record.failure.is_none()
    }
    pub fn observed_callbacks(&self) -> usize {
        self.record.counts.observed_callbacks
    }
    pub fn accepted_samples(&self) -> usize {
        self.record.counts.accepted_samples
    }
    pub fn sampler_attempts(&self) -> usize {
        self.record.counts.sampler_attempts
    }
    pub fn applied_choices(&self) -> usize {
        self.record.counts.apply_successes
    }
    pub fn terminal_state_key(&self) -> Option<&str> {
        self.record
            .terminal
            .as_ref()
            .map(|t| t.final_state.as_str())
    }
    pub fn failure_reason(&self) -> Option<FailureReason> {
        self.record.failure.as_ref().map(|f| f.reason)
    }
    pub fn failure_stage(&self) -> Option<AttemptStage> {
        self.record.failure.as_ref().map(|f| f.stage)
    }
    pub fn failure_callback_index(&self) -> Option<usize> {
        self.record
            .failure
            .as_ref()
            .and_then(|f| f.global_callback_index)
    }
    pub fn failure_message(&self) -> Option<&str> {
        self.record.failure.as_ref().map(|f| f.message.as_str())
    }
    pub(crate) fn record(&self) -> &Record {
        &self.record
    }
}

/// A completed audit may describe a complete game or a reproducible failed
/// prefix. Neither status is ML admission, producer authentication or strength.
///
/// ```compile_fail
/// use tzolkin_ai::public_stochastic_native::AuditedStochasticGame;
/// let _: AuditedStochasticGame = serde_json::from_str("{}").unwrap();
/// ```
pub struct AuditedStochasticGame {
    record: Record,
}
impl AuditedStochasticGame {
    pub(crate) fn record(&self) -> &Record {
        &self.record
    }
    pub fn complete(&self) -> bool {
        self.record.terminal.is_some() && self.record.failure.is_none()
    }
    pub fn observed_callbacks(&self) -> usize {
        self.record.counts.observed_callbacks
    }
    pub fn accepted_samples(&self) -> usize {
        self.record.counts.accepted_samples
    }
    pub fn applied_choices(&self) -> usize {
        self.record.counts.apply_successes
    }
    pub fn failure_reason(&self) -> Option<FailureReason> {
        self.record.failure.as_ref().map(|f| f.reason)
    }
    pub fn terminal_state_key(&self) -> Option<&str> {
        self.record
            .terminal
            .as_ref()
            .map(|t| t.final_state.as_str())
    }
    pub fn training_admission_available(&self) -> bool {
        false
    }
    pub fn producer_authenticated(&self) -> bool {
        false
    }
    pub fn independent_seed_origins_verified(&self) -> bool {
        false
    }
}

pub(crate) fn numerical_target() -> String {
    // exp/ln is a same-target contract. No unobserved cross-platform bit claim.
    format!(
        "scalar-f32-f64-tick53-v1:{}:{}:{}",
        std::env::consts::ARCH,
        std::env::consts::OS,
        usize::BITS
    )
}
fn header(
    config: &NativeStochasticConfig,
    policy: &PublicPolicyHandle<'_>,
) -> Result<Header, String> {
    policy.provenance().validate()?;
    // This checks Scalar and the distinct immutable V2 owner boundary without
    // sampling/forwarding. Each actual run creates its own session below.
    StochasticSession::new(policy, config.sampling_identity)?;
    Ok(Header {
        source_kind: SOURCE_KIND.into(),
        rules_version: RULES_VERSION,
        rules_baseline: RULES_BASELINE.into(),
        catalog_hash: catalog_hash(),
        observation_schema: OBSERVATION_SCHEMA,
        move_schema: MOVE_SCHEMA,
        config: Config::from_config(config),
        base_policy: policy.provenance().clone(),
        sampling_version: SAMPLING_VERSION.into(),
        rng_version: RNG_VERSION.into(),
        denominator_bits: 53,
        uniform_mixture_bits: hex64(UNIFORM_MIXTURE.to_bits()),
        backend: policy.backend().into(),
        numerical_target: numerical_target(),
        native_family_id: seed_family_id(config.environment_seed),
        raw_artifact_identity_available: false,
        producer_authenticated: false,
        independent_seed_origins_verified: false,
    })
}

/// One game, all actual decision actors, using one fresh session. Runtime
/// failure keeps every observation/accepted sample up to the failed attempt;
/// terminal fields remain null. Configuration/policy rejection occurs before
/// game creation. There is no silent heuristic fallback or retry.
pub fn collect_native(
    config: &NativeStochasticConfig,
    policy: &PublicPolicyHandle<'_>,
) -> Result<CollectedStochasticGame, String> {
    run(config, policy, None).map(|record| CollectedStochasticGame { record })
}

/// Closed internal dispatch: both variants borrow an already sealed handle.
/// Caller-supplied prediction providers or raw models are never accepted.
#[derive(Clone, Copy)]
pub(crate) enum NativePolicy<'handle, 'model> {
    Bc(&'handle PublicPolicyHandle<'model>),
    Rl(&'handle UpdatedPublicRlHandle<'model>),
    Repeated(&'handle RepeatedPublicRlHandle<'model>),
    Long(&'handle crate::public_rl_native::LongRlHandle<'model>),
}
impl<'handle, 'model> NativePolicy<'handle, 'model> {
    fn session(
        self,
        identity: SamplingStreamIdentity,
    ) -> Result<NativeSession<'handle, 'model>, String> {
        match self {
            Self::Bc(policy) => StochasticSession::new(policy, identity).map(NativeSession::Bc),
            Self::Rl(policy) => RlStochasticSession::new(policy, identity).map(NativeSession::Rl),
            Self::Repeated(policy) => {
                RepeatedStochasticSession::new(policy, identity).map(NativeSession::Repeated)
            }
            Self::Long(policy) => {
                crate::public_stochastic::LongStochasticSession::new(policy, identity)
                    .map(NativeSession::Long)
            }
        }
    }
    fn sampling_version(self) -> &'static str {
        match self {
            Self::Bc(_) => SAMPLING_VERSION,
            Self::Rl(_) => RL_SAMPLING_VERSION,
            Self::Repeated(_) => REPEATED_SAMPLING_VERSION,
            Self::Long(_) => crate::public_policy_long::SAMPLING_VERSION,
        }
    }
}
enum NativeSession<'handle, 'model> {
    Bc(StochasticSession<'handle, 'model>),
    Rl(RlStochasticSession<'handle, 'model>),
    Repeated(RepeatedStochasticSession<'handle, 'model>),
    Long(crate::public_stochastic::LongStochasticSession<'handle, 'model>),
}
impl NativeSession<'_, '_> {
    fn sample(&mut self, observation: &Observation) -> Result<SampledDecision, String> {
        match self {
            Self::Bc(session) => session.sample(observation),
            Self::Rl(session) => session.sample(observation),
            Self::Repeated(session) => session.sample(observation),
            Self::Long(session) => session.sample(observation),
        }
    }
    fn counters(&self) -> SamplingCounters {
        match self {
            Self::Bc(session) => session.counters(),
            Self::Rl(session) => session.counters(),
            Self::Repeated(session) => session.counters(),
            Self::Long(session) => session.counters(),
        }
    }
    fn stop(&mut self) {
        match self {
            Self::Bc(session) => session.stop(),
            Self::Rl(session) => session.stop(),
            Self::Repeated(session) => session.stop(),
            Self::Long(session) => session.stop(),
        }
    }
}

fn blank_counts() -> Counts {
    Counts {
        observed_callbacks: 0,
        sampler_attempts: 0,
        accepted_samples: 0,
        candidate_rows_reserved: 0,
        actor_draws: vec![hex64(0); 4],
        apply_attempts: 0,
        apply_successes: 0,
        validated_applied_steps: 0,
        session_stopped: false,
    }
}
fn fail<P>(
    record: &mut Record<P>,
    reason: FailureReason,
    stage: AttemptStage,
    index: Option<usize>,
    message: &str,
) {
    let failure = Failure::new(reason, stage, index, message);
    if let Some(index) = index
        && let Some(callback) = record.callbacks.get_mut(index)
    {
        callback.failure = Some(failure.clone());
    }
    record.failure = Some(failure);
    record.terminal = None;
}
fn precheck<P>(
    expected: Option<&Record<P>>,
    index: usize,
    observation: &Observation,
    state_before: &str,
    omit_observation: bool,
) -> Result<(), String> {
    let Some(expected) = expected else {
        return Ok(());
    };
    let claimed = expected
        .callbacks
        .get(index)
        .ok_or_else(|| format!("Callback {index}: missing actual observed callback"))?;
    if claimed.global_callback_index != index
        || claimed.actor != observation.actor
        || claimed.turn_player != observation.turn_player
        || claimed.observation_key != observation.observation_key
        || claimed.state_before != state_before
    {
        return Err(format!(
            "Callback {index}: actor/key/index/state mismatch before sampling"
        ));
    }
    match (&claimed.observation, omit_observation) {
        (None, true) => (),
        (Some(o), false) => {
            // Preserve signed zero and native JSON float roundtrip, rather than
            // relying on PartialEq for f64 nested fields.
            if serde_json::to_vec(o).map_err(|e| e.to_string())?
                != serde_json::to_vec(observation).map_err(|e| e.to_string())?
            {
                return Err(format!(
                    "Callback {index}: native Observation mismatch before sampling"
                ));
            }
        }
        _ => {
            return Err(format!(
                "Callback {index}: admitted observation/null mismatch"
            ));
        }
    }
    Ok(())
}
fn check_callback<P>(
    expected: Option<&Record<P>>,
    record: &Record<P>,
    index: usize,
) -> Result<(), String> {
    if let Some(expected) = expected
        && expected.callbacks.get(index) != record.callbacks.get(index)
    {
        return Err(format!(
            "Callback {index}: sampling bits, apply, budget or failure mismatch"
        ));
    }
    Ok(())
}
fn checked_add_with_cap(current: usize, amount: usize, limit: usize) -> Option<usize> {
    current.checked_add(amount).filter(|n| *n <= limit)
}
fn validate_state(state: &GameState) -> Result<(), String> {
    let value = serde_json::to_value(state).map_err(|e| e.to_string())?;
    if !tzolkin_core::validation::validate_game_state(&value) {
        return Err("Native game state invariant failed".into());
    }
    Ok(())
}

/// The same checked transition machinery serves collection and audit, but
/// audit compares each actual Observation before NN use and each derived sample
/// and transition. Source Observations are never fed to either the model or core.
fn run(
    config: &NativeStochasticConfig,
    policy: &PublicPolicyHandle<'_>,
    expected: Option<&Record>,
) -> Result<Record, String> {
    run_with_apply(config, policy, expected, apply_move)
}
// Private injection point permits a test of a sampled-but-unapplied choice.
// Collection and audit always pass the same authoritative core apply_move.
fn run_with_apply(
    config: &NativeStochasticConfig,
    policy: &PublicPolicyHandle<'_>,
    expected: Option<&Record>,
    apply: impl FnMut(&GameState, GameMove) -> Result<GameState, String>,
) -> Result<Record, String> {
    run_shared_with_apply(
        config,
        NativePolicy::Bc(policy),
        RECORD_SCHEMA,
        header(config, policy)?,
        expected,
        apply,
    )
}

/// Generic only over a private wire identity. Sampling dispatch stays closed.
pub(crate) fn run_shared<P: Serialize + PartialEq>(
    config: &NativeStochasticConfig,
    policy: NativePolicy<'_, '_>,
    schema: &str,
    header: Header<P>,
    expected: Option<&Record<P>>,
) -> Result<Record<P>, String> {
    run_shared_with_apply(config, policy, schema, header, expected, apply_move)
}

pub(crate) fn run_mixed_shared<P: Serialize + PartialEq>(
    config: &NativeStochasticConfig,
    policy: NativePolicy<'_, '_>,
    learner_seat: usize,
    header: Header<P>,
    expected: Option<&Record<P>>,
) -> Result<Record<P>, String> {
    run_loop(
        config,
        policy,
        SamplingActors::One(learner_seat),
        crate::public_mixed_native::RECORD_SCHEMA,
        header,
        expected,
        apply_move,
    )
}

#[derive(Clone, Copy)]
enum SamplingActors {
    All,
    One(usize),
}
impl SamplingActors {
    fn samples(self, actor: usize) -> bool {
        match self {
            Self::All => true,
            Self::One(learner) => actor == learner,
        }
    }
}
fn run_shared_with_apply<P: Serialize + PartialEq>(
    config: &NativeStochasticConfig,
    policy: NativePolicy<'_, '_>,
    schema: &str,
    header: Header<P>,
    expected: Option<&Record<P>>,
    apply: impl FnMut(&GameState, GameMove) -> Result<GameState, String>,
) -> Result<Record<P>, String> {
    run_loop(
        config,
        policy,
        SamplingActors::All,
        schema,
        header,
        expected,
        apply,
    )
}
fn run_loop<P: Serialize + PartialEq>(
    config: &NativeStochasticConfig,
    policy: NativePolicy<'_, '_>,
    actors: SamplingActors,
    schema: &str,
    header: Header<P>,
    expected: Option<&Record<P>>,
    mut apply: impl FnMut(&GameState, GameMove) -> Result<GameState, String>,
) -> Result<Record<P>, String> {
    if let SamplingActors::One(learner) = actors
        && (learner >= config.players || schema != crate::public_mixed_native::RECORD_SCHEMA)
    {
        return Err("Invalid mixed learner seat or schema".into());
    }
    if matches!(actors, SamplingActors::All)
        && expected.is_some_and(|record| {
            record
                .callbacks
                .iter()
                .any(|callback| callback.opponent_choice.is_some() || callback.opponent_attempted)
        })
    {
        return Err("Opponent fields are forbidden in an all-seat stochastic schema".into());
    }
    if let Some(expected) = expected
        && expected.header != header
    {
        return Err("Stochastic record header/content binding mismatch".into());
    }
    let mut record = Record {
        schema: schema.into(),
        header,
        callbacks: vec![],
        terminal: None,
        failure: None,
        counts: blank_counts(),
        source_payload_reserved_bytes: 0,
        trace_payload_reserved_bytes: 0,
        training_admission: "unavailable-distinct-codec-required".into(),
    };
    record.source_payload_reserved_bytes =
        serialized_size(&record.header, config.limits.source_bytes)?
            .checked_add(FINAL_RESERVE)
            .ok_or("Header payload accounting overflow")?;
    if record.source_payload_reserved_bytes > config.limits.source_bytes {
        return Err("Header and final diagnostic reservation exceed source cap".into());
    }
    let mut session = policy.session(config.sampling_identity)?;
    // This fixed opponent snapshot is used only by the closed mixed route.
    // Its action-row work is bounded by callbacks * the full-legal cap.
    config
        .limits
        .callbacks
        .checked_mul(4096)
        .ok_or("Opponent action-row work bound overflow")?;
    let opponent_weights = crate::policy::HeuristicWeights::default();
    let start = create_game_with_options(
        (0..config.players)
            .map(|p| format!("Stochastic P{p}"))
            .collect(),
        config.environment_seed,
        GameOptions::default(),
    );
    let mut state = match start {
        Ok(state) => state,
        Err(error) => {
            fail(
                &mut record,
                FailureReason::CreationError,
                AttemptStage::Startup,
                None,
                &error,
            );
            session.stop();
            finish_counts(&mut record, &session);
            return finish(record, config, expected);
        }
    };
    if let Err(error) = validate_state(&state) {
        fail(
            &mut record,
            FailureReason::CoreValidationError,
            AttemptStage::Startup,
            None,
            &error,
        );
    }
    while record.failure.is_none() && state.phase != Phase::Finished {
        let index = record.callbacks.len();
        if index >= config.limits.callbacks {
            fail(
                &mut record,
                FailureReason::CallbackLimit,
                AttemptStage::BeforeObservation,
                None,
                "Callback limit reached before the next observation",
            );
            break;
        }
        let observation = match observe(&state, state.current_player) {
            Ok(o) => o,
            Err(error) => {
                fail(
                    &mut record,
                    FailureReason::ObservationError,
                    AttemptStage::BeforeObservation,
                    None,
                    &error,
                );
                break;
            }
        };
        let before = match state_key(&state) {
            Ok(key) => key,
            Err(error) => {
                fail(
                    &mut record,
                    FailureReason::ObservationError,
                    AttemptStage::BeforeObservation,
                    None,
                    &error,
                );
                break;
            }
        };
        // Reserve the current observation, two legal-action copies and bounded
        // callback/failure metadata before NN/draw/apply. Final fields were
        // reserved in the header. The trace has a separate conservative quota.
        let observation_size = serialized_size(&observation, config.limits.source_bytes).ok();
        let action_size = observation
            .legal_actions
            .iter()
            .map(|a| serialized_size(a, config.limits.source_bytes))
            .collect::<Result<Vec<_>, _>>()
            .ok()
            .and_then(|sizes| sizes.into_iter().max());
        let reservation = observation_size
            .zip(action_size)
            .and_then(|(o, a)| a.checked_mul(2).and_then(|a| o.checked_add(a)))
            .and_then(|n| n.checked_add(CALLBACK_RESERVE));
        let admitted = reservation.and_then(|n| {
            checked_add_with_cap(
                record.source_payload_reserved_bytes,
                n,
                config.limits.source_bytes,
            )
        });
        precheck(expected, index, &observation, &before, admitted.is_none())?;
        record.callbacks.push(Callback {
            global_callback_index: index,
            actor: observation.actor,
            turn_player: observation.turn_player,
            observation_key: observation.observation_key.clone(),
            observation: admitted.map(|_| observation.clone()),
            state_before: before,
            sample: None,
            opponent_choice: None,
            opponent_attempted: false,
            sampler_attempted: false,
            apply_attempted: false,
            apply_succeeded: false,
            state_after: None,
            failure: None,
        });
        record.counts.observed_callbacks += 1;
        let Some(admitted) = admitted else {
            // Stub/failure metadata uses the initial final reserve, without
            // storing the oversized observation. The key/actual actor survives.
            fail(
                &mut record,
                FailureReason::SourcePayloadLimit,
                AttemptStage::BeforeSample,
                Some(index),
                "Source payload refused before sampling",
            );
            check_callback(expected, &record, index)?;
            break;
        };
        record.source_payload_reserved_bytes = admitted;
        let candidates = observation.legal_actions.len();
        let sampling = actors.samples(observation.actor);
        let rows = session.counters().reserved_candidate_rows();
        if !(1..=4096).contains(&candidates)
            || (sampling
                && checked_add_with_cap(rows, candidates, config.limits.candidate_rows).is_none())
        {
            fail(
                &mut record,
                FailureReason::CandidateLimit,
                AttemptStage::BeforeSample,
                Some(index),
                "Candidate row limit refused before sampling",
            );
            check_callback(expected, &record, index)?;
            break;
        }
        let decision = if sampling {
            let Some(trace_reservation) = checked_add_with_cap(
                record.trace_payload_reserved_bytes,
                TRACE_RESERVE,
                config.limits.trace_bytes,
            ) else {
                fail(
                    &mut record,
                    FailureReason::TracePayloadLimit,
                    AttemptStage::BeforeSample,
                    Some(index),
                    "Trace payload refused before sampling",
                );
                check_callback(expected, &record, index)?;
                break;
            };
            record.trace_payload_reserved_bytes = trace_reservation;
            record.callbacks[index].sampler_attempted = true;
            record.counts.sampler_attempts += 1;
            let sample = match session.sample(&observation) {
                Ok(sample) => sample,
                Err(error) => {
                    fail(
                        &mut record,
                        FailureReason::ModelOrSamplingError,
                        AttemptStage::BeforeSample,
                        Some(index),
                        &error,
                    );
                    check_callback(expected, &record, index)?;
                    break;
                }
            };
            let wire = Sample::from_sample(&sample);
            // Every successful draw is retained even if the following operation
            // cannot be applied. No poisoned session can return another decision.
            record.callbacks[index].sample = Some(wire);
            let d = sample.decision();
            if d.actor != observation.actor
                || d.observation_key != observation.observation_key
                || d.policy_version != policy.sampling_version()
                || !d.score.is_finite()
                || observation
                    .legal_actions
                    .iter()
                    .filter(|a| **a == *sample.legal_action())
                    .count()
                    != 1
                || sample.legal_action().r#move != d.r#move
                || serialized_size(
                    &record.callbacks[index].sample.as_ref().unwrap().trace,
                    TRACE_RESERVE,
                )
                .is_err()
            {
                fail(
                    &mut record,
                    FailureReason::DecisionMismatch,
                    AttemptStage::BeforeApply,
                    Some(index),
                    "Returned sample/decision/trace binding mismatch",
                );
                check_callback(expected, &record, index)?;
                break;
            }
            d.clone()
        } else {
            record.callbacks[index].opponent_attempted = true;
            let decision = match crate::choose_move_with_weights(&observation, &opponent_weights) {
                Ok(decision) => decision,
                Err(error) => {
                    fail(
                        &mut record,
                        FailureReason::ModelOrSamplingError,
                        AttemptStage::BeforeSample,
                        Some(index),
                        &error,
                    );
                    check_callback(expected, &record, index)?;
                    break;
                }
            };
            let chosen = observation
                .legal_actions
                .iter()
                .filter(|action| action.r#move == decision.r#move)
                .collect::<Vec<_>>();
            if decision.actor != observation.actor
                || decision.observation_key != observation.observation_key
                || decision.policy_version != crate::POLICY_VERSION
                || !decision.score.is_finite()
                || chosen.len() != 1
            {
                fail(
                    &mut record,
                    FailureReason::DecisionMismatch,
                    AttemptStage::BeforeApply,
                    Some(index),
                    "Opponent decision/full legal binding mismatch",
                );
                check_callback(expected, &record, index)?;
                break;
            }
            record.callbacks[index].opponent_choice = Some(OpponentChoice::from_decision(
                &decision,
                (*chosen[0]).clone(),
            ));
            decision
        };
        record.callbacks[index].apply_attempted = true;
        record.counts.apply_attempts += 1;
        match apply(&state, decision.r#move) {
            Err(error) => {
                fail(
                    &mut record,
                    FailureReason::CoreApplyError,
                    AttemptStage::BeforeApply,
                    Some(index),
                    &error,
                );
                check_callback(expected, &record, index)?;
                break;
            }
            Ok(next) => state = next,
        }
        record.callbacks[index].apply_succeeded = true;
        record.counts.apply_successes += 1;
        match state_key(&state) {
            Ok(key) => record.callbacks[index].state_after = Some(key),
            Err(error) => {
                fail(
                    &mut record,
                    FailureReason::CoreValidationError,
                    AttemptStage::AfterApply,
                    Some(index),
                    &error,
                );
                check_callback(expected, &record, index)?;
                break;
            }
        }
        if let Err(error) = validate_state(&state) {
            fail(
                &mut record,
                FailureReason::CoreValidationError,
                AttemptStage::AfterApply,
                Some(index),
                &error,
            );
            check_callback(expected, &record, index)?;
            break;
        }
        record.counts.validated_applied_steps += 1;
        check_callback(expected, &record, index)?;
    }
    if record.failure.is_none() && state.phase == Phase::Finished {
        if let Err(error) = validate_state(&state) {
            fail(
                &mut record,
                FailureReason::CoreValidationError,
                AttemptStage::TerminalValidation,
                None,
                &error,
            );
        } else if state.final_scores.len() != config.players {
            fail(
                &mut record,
                FailureReason::CoreValidationError,
                AttemptStage::TerminalValidation,
                None,
                "Terminal score coverage mismatch",
            );
        } else {
            match state_key(&state) {
                Ok(final_state) => {
                    record.terminal = Some(Terminal {
                        final_state,
                        final_scores: state
                            .final_scores
                            .iter()
                            .map(FinalScoreWire::from_score)
                            .collect(),
                    })
                }
                Err(error) => fail(
                    &mut record,
                    FailureReason::CoreValidationError,
                    AttemptStage::TerminalValidation,
                    None,
                    &error,
                ),
            }
        }
    }
    session.stop();
    finish_counts(&mut record, &session);
    finish(record, config, expected)
}
fn finish_counts<P>(record: &mut Record<P>, session: &NativeSession<'_, '_>) {
    let counters = session.counters();
    record.counts.accepted_samples = counters.accepted_samples();
    record.counts.candidate_rows_reserved = counters.reserved_candidate_rows();
    record.counts.actor_draws = counters.actor_draws().iter().copied().map(hex64).collect();
    record.counts.session_stopped = counters.stopped();
}
fn finish<P: Serialize + PartialEq>(
    record: Record<P>,
    config: &NativeStochasticConfig,
    expected: Option<&Record<P>>,
) -> Result<Record<P>, String> {
    // Verify actual serialization stays within the conservative reservations.
    // This is a late invariant, not a license to perform work before admission.
    let traces = record
        .callbacks
        .iter()
        .filter_map(|c| c.sample.as_ref())
        .map(|s| serialized_size(&s.trace, TRACE_RESERVE))
        .collect::<Result<Vec<_>, _>>()?;
    let trace_bytes = traces.into_iter().try_fold(0usize, |sum, size| {
        sum.checked_add(size).ok_or("Trace bytes overflow")
    })?;
    let total = serialized_size(&record, MAX_RECORD_BYTES)?;
    let source_bytes = total
        .checked_sub(trace_bytes)
        .ok_or("Source bytes underflow")?;
    if source_bytes > config.limits.source_bytes
        || trace_bytes > config.limits.trace_bytes
        || source_bytes > record.source_payload_reserved_bytes
        || trace_bytes > record.trace_payload_reserved_bytes
    {
        return Err("Internal payload reservation invariant failed".into());
    }
    if let Some(expected) = expected
        && *expected != record
    {
        return Err("Stochastic record final counts/terminal/failure/budget mismatch".into());
    }
    Ok(record)
}

pub(crate) fn audit_record(
    record: &Record,
    policy: &PublicPolicyHandle<'_>,
) -> Result<AuditedStochasticGame, String> {
    if record.schema != RECORD_SCHEMA
        || record.header.source_kind != SOURCE_KIND
        || record.callbacks.len() > MAX_SAMPLES
        || record.training_admission != "unavailable-distinct-codec-required"
    {
        return Err("Unsupported stochastic record schema/kind/admission/count".into());
    }
    let config = record.header.config.checked_config()?;
    if record.callbacks.len() > config.limits.callbacks {
        return Err("Callback count exceeds declared cap".into());
    }
    let checked = run(&config, policy, Some(record))?;
    Ok(AuditedStochasticGame { record: checked })
}

#[cfg(test)]
mod tests {
    use super::*;
    mod test_temp_root {
        include!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/support/temp_root.rs"
        ));
    }
    use crate::public_native::integration_fixture;
    use crate::public_policy_episode::ValidatedPolicyEpisode;
    use crate::public_stochastic::SamplingSeed;
    use crate::public_stochastic_record::{audit_record_bytes, encode_record};

    pub(crate) fn config(players: usize, limits: CollectionLimits) -> NativeStochasticConfig {
        NativeStochasticConfig::new(
            players,
            17,
            SamplingStreamIdentity::new(SamplingSeed::new(11235), 4, 2, players).unwrap(),
            limits,
        )
        .unwrap()
    }
    #[test]
    fn complete_three_four_player_games_roundtrip_and_sessions_restart() {
        let model = integration_fixture::model(false);
        let original = serde_json::to_vec(&model).unwrap();
        let policy = integration_fixture::handle(&model);
        for players in [3, 4] {
            let config = config(players, CollectionLimits::default());
            let game = collect_native(&config, &policy).unwrap();
            println!(
                "A8b unqualified synthetic fixture17,{players}p callbacks{} samples{} complete{} failure{:?}",
                game.observed_callbacks(),
                game.accepted_samples(),
                game.complete(),
                game.failure_reason()
            );
            assert!(game.complete(), "{:?}", game.failure_message());
            assert_eq!(game.observed_callbacks(), game.accepted_samples());
            assert_eq!(game.applied_choices(), game.accepted_samples());
            assert!(game.record.counts.session_stopped);
            assert_eq!(
                game.record.terminal.as_ref().unwrap().final_scores.len(),
                players
            );
            assert!(game.record.callbacks.iter().any(|c| {
                c.observation
                    .as_ref()
                    .is_some_and(|o| o.phase == Phase::Playing && o.pending_task.is_some())
            }));
            assert!(game.record.callbacks.iter().any(|c| {
                c.actor != c.turn_player
                    && c.observation
                        .as_ref()
                        .is_some_and(|o| o.phase == Phase::Playing)
            }));
            for (index, callback) in game.record.callbacks.iter().enumerate() {
                let sample = callback.sample.as_ref().unwrap();
                assert_eq!(callback.global_callback_index, index);
                assert_eq!(sample.trace.sample_index, index);
                assert_eq!(sample.trace.actor, callback.actor);
                assert!(
                    callback.sampler_attempted
                        && callback.apply_attempted
                        && callback.apply_succeeded
                );
                assert_eq!(sample.decision.observation_key, callback.observation_key);
                assert_eq!(sample.chosen.r#move, sample.decision.r#move);
                let trace = serde_json::to_value(&sample.trace).unwrap();
                for key in [
                    "seed",
                    "environmentSeed",
                    "samplingSeed",
                    "private",
                    "wealthOffer",
                    "buildingDeck",
                    "features",
                ] {
                    assert!(trace.get(key).is_none());
                }
            }
            let bytes = encode_record(&game).unwrap();
            if players == 3 {
                reject_legacy_exports(&bytes);
            }
            let receipt = audit_record_bytes(&bytes, &policy).unwrap();
            assert!(receipt.complete());
            assert_eq!(receipt.observed_callbacks(), game.observed_callbacks());
            assert_eq!(receipt.terminal_state_key(), game.terminal_state_key());
            assert!(!receipt.training_admission_available());
            assert!(!receipt.producer_authenticated());
            assert!(!receipt.independent_seed_origins_verified());
            let mut unapplied = receipt.record.clone();
            if players == 3 {
                unapplied.callbacks[0].apply_succeeded = false;
            } else {
                unapplied.callbacks[0].state_after = None;
            }
            assert!(
                ValidatedPolicyEpisode::from_audited(AuditedStochasticGame { record: unapplied })
                    .is_err()
            );
            let episode = ValidatedPolicyEpisode::from_audited(receipt).unwrap();
            assert_eq!(episode.source_config(), &config);
            assert_eq!(episode.len(), game.applied_choices());
            assert_eq!(
                episode.terminal_state_key(),
                game.terminal_state_key().unwrap()
            );
            assert_eq!(episode.canonical_record_checksum().len(), 64);
            assert!(!episode.training_admission_available());
            let scores = &game.record.terminal.as_ref().unwrap().final_scores;
            let winners = scores.iter().filter(|score| score.rank == 1).count();
            let mut own_indices = [0; 4];
            for step in episode.steps() {
                let actor = step.actor();
                assert_eq!(step.actor_step_index(), own_indices[actor]);
                assert_eq!(step.actor_draw_ordinal(), own_indices[actor]);
                own_indices[actor] += 1;
                assert_eq!(step.observation().actor, actor);
                assert_eq!(
                    step.chosen(),
                    &step.observation().legal_actions[step.chosen_index()]
                );
                assert_eq!(step.session_sample_index(), step.global_callback_index());
                let rank = scores
                    .iter()
                    .find(|score| score.player_id == actor)
                    .unwrap()
                    .rank;
                let target = if rank == 1 { 1.0 / winners as f64 } else { 0.0 };
                assert_eq!(step.actor_terminal_rank(), rank);
                assert_eq!(step.return_target(), target);
                if let Some(next) = step.next_own_global_index() {
                    assert_eq!(episode.step(next).unwrap().actor(), actor);
                    assert_eq!(
                        step.intervening_callbacks(),
                        next - step.global_callback_index() - 1
                    );
                    assert_eq!(step.reward(), 0.0);
                    assert_eq!(step.terminal_bootstrap(), None);
                } else {
                    assert_eq!(
                        step.intervening_callbacks(),
                        episode.len() - step.global_callback_index() - 1
                    );
                    assert_eq!(step.reward(), target);
                    assert_eq!(step.terminal_bootstrap(), Some(0.0));
                }
            }
            assert!(
                episode
                    .steps()
                    .any(|step| step.actor() != step.observation().turn_player)
            );
            // Both runs construct fresh worlds/sessions; no counters/RNG leak.
            assert_eq!(
                bytes,
                encode_record(&collect_native(&config, &policy).unwrap()).unwrap()
            );
        }
        assert_eq!(serde_json::to_vec(&model).unwrap(), original);
    }

    #[test]
    fn failed_model_retains_actual_observation_zero_draws_and_null_terminal() {
        let model = integration_fixture::model(true);
        let policy = integration_fixture::handle(&model);
        let game = collect_native(&config(3, CollectionLimits::default()), &policy).unwrap();
        assert!(!game.complete());
        assert_eq!(
            game.failure_reason(),
            Some(FailureReason::ModelOrSamplingError)
        );
        assert_eq!(game.failure_stage(), Some(AttemptStage::BeforeSample));
        assert_eq!(game.failure_callback_index(), Some(0));
        assert!(
            game.failure_message()
                .unwrap()
                .contains("hidden activation")
        );
        assert_eq!(game.observed_callbacks(), 1);
        assert_eq!(game.sampler_attempts(), 1);
        assert_eq!(game.accepted_samples(), 0);
        assert_eq!(game.applied_choices(), 0);
        assert!(game.record.callbacks[0].observation.is_some());
        assert!(game.record.callbacks[0].sample.is_none());
        assert!(game.record.terminal.is_none());
        assert!(
            game.record
                .counts
                .actor_draws
                .iter()
                .all(|draw| draw == &hex64(0))
        );
        assert!(game.record.counts.candidate_rows_reserved > 0);
        let receipt = audit_record_bytes(&encode_record(&game).unwrap(), &policy).unwrap();
        assert!(ValidatedPolicyEpisode::from_audited(receipt).is_err());
    }

    #[test]
    fn callback_candidate_trace_and_source_caps_refuse_before_sampling() {
        let model = integration_fixture::model(false);
        let policy = integration_fixture::handle(&model);
        let cases = [
            (
                CollectionLimits::new(1, 1_000_000, MAX_SOURCE_BYTES, MAX_TRACE_BYTES).unwrap(),
                FailureReason::CallbackLimit,
            ),
            (
                CollectionLimits::new(4000, 1, MAX_SOURCE_BYTES, MAX_TRACE_BYTES).unwrap(),
                FailureReason::CandidateLimit,
            ),
            (
                CollectionLimits::new(4000, 1_000_000, MAX_SOURCE_BYTES, TRACE_RESERVE).unwrap(),
                FailureReason::TracePayloadLimit,
            ),
            (
                CollectionLimits::new(4000, 1_000_000, 64 * 1024, MAX_TRACE_BYTES).unwrap(),
                FailureReason::SourcePayloadLimit,
            ),
        ];
        for (limits, reason) in cases {
            let game = collect_native(&config(3, limits), &policy).unwrap();
            assert!(!game.complete());
            assert_eq!(game.failure_reason(), Some(reason));
            assert!(game.record.terminal.is_none());
            assert!(game.record.counts.session_stopped);
            assert!(
                game.accepted_samples() < game.observed_callbacks()
                    || reason == FailureReason::CallbackLimit
            );
            let receipt = audit_record_bytes(&encode_record(&game).unwrap(), &policy).unwrap();
            assert!(ValidatedPolicyEpisode::from_audited(receipt).is_err());
            if reason == FailureReason::CandidateLimit {
                assert_eq!(game.sampler_attempts(), 0);
                assert_eq!(game.accepted_samples(), 0);
                assert_eq!(game.record.counts.candidate_rows_reserved, 0);
            }
            if reason == FailureReason::SourcePayloadLimit {
                let last = game.record.callbacks.last().unwrap();
                assert!(last.observation.is_none() && last.sample.is_none());
                assert!(!last.sampler_attempted && !last.apply_attempted);
            }
        }
    }

    #[test]
    fn sampled_but_unapplied_choice_is_kept_and_false_core_failure_is_not_authenticated() {
        let model = integration_fixture::model(false);
        let policy = integration_fixture::handle(&model);
        let config = config(3, CollectionLimits::default());
        // Deliberate test-only core failure, not an observed production failure.
        let record = run_with_apply(&config, &policy, None, |_, _| {
            Err("injected apply refusal".into())
        })
        .unwrap();
        assert_eq!(record.counts.accepted_samples, 1);
        assert_eq!(record.counts.apply_attempts, 1);
        assert_eq!(record.counts.apply_successes, 0);
        assert_eq!(record.counts.validated_applied_steps, 0);
        let callback = &record.callbacks[0];
        assert!(
            callback.sample.is_some() && callback.sampler_attempted && callback.apply_attempted
        );
        assert!(!callback.apply_succeeded);
        assert!(callback.state_after.is_none() && record.terminal.is_none());
        assert_eq!(
            record.failure.as_ref().unwrap().reason,
            FailureReason::CoreApplyError
        );
        assert_eq!(
            record
                .counts
                .actor_draws
                .iter()
                .filter(|d| *d != &hex64(0))
                .count(),
            1
        );
        // The authoritative auditor rejects the invented failure while keeping
        // its mechanics separate from this test's failure-retention assertion.
        assert!(audit_record(&record, &policy).is_err());
    }

    #[test]
    fn audit_rejects_actor_global_index_rng_mass_apply_and_content_corruption() {
        let model = integration_fixture::model(false);
        let policy = integration_fixture::handle(&model);
        let limits = CollectionLimits::new(2, 1000000, MAX_SOURCE_BYTES, MAX_TRACE_BYTES).unwrap();
        let game = collect_native(&config(3, limits), &policy).unwrap();
        assert_eq!(game.failure_reason(), Some(FailureReason::CallbackLimit));
        let mutations: [fn(&mut Record); 12] = [
            |r| r.callbacks[0].actor = (r.callbacks[0].actor + 1) % 3,
            |r| r.callbacks[0].global_callback_index = 1,
            |r| {
                r.callbacks[0]
                    .observation
                    .as_mut()
                    .unwrap()
                    .legal_actions
                    .reverse()
            },
            |r| r.callbacks[0].sample.as_mut().unwrap().trace.sample_index = 2,
            |r| r.callbacks[0].sample.as_mut().unwrap().trace.raw_u64 = hex64(0),
            |r| r.callbacks[0].sample.as_mut().unwrap().trace.mass_ticks = "0".into(),
            |r| {
                r.callbacks[0]
                    .sample
                    .as_mut()
                    .unwrap()
                    .trace
                    .behavior_logp_bits = hex64(0)
            },
            |r| r.callbacks[0].apply_succeeded = false,
            |r| r.callbacks[0].state_after = None,
            |r| r.counts.accepted_samples = 0,
            |r| r.header.producer_authenticated = true,
            |r| r.header.config.options.tribes = true,
        ];
        for change in mutations {
            let mut record = game.record.clone();
            change(&mut record);
            assert!(audit_record(&record, &policy).is_err());
        }
        let mut record = game.record.clone();
        record.callbacks.pop();
        assert!(audit_record(&record, &policy).is_err());
        let other_model = crate::public_model::PublicPolicyArtifact::new(17).unwrap();
        let other_policy = integration_fixture::handle(&other_model);
        assert!(audit_record(&game.record, &other_policy).is_err());
    }

    #[test]
    fn distinct_collector_wire_rejected_by_both_legacy_exporters_before_output() {
        let model = integration_fixture::model(false);
        let policy = integration_fixture::handle(&model);
        let game = collect_native(
            &config(
                3,
                CollectionLimits::new(1, 1000000, MAX_SOURCE_BYTES, MAX_TRACE_BYTES).unwrap(),
            ),
            &policy,
        )
        .unwrap();
        reject_legacy_exports(&encode_record(&game).unwrap());
    }
    fn reject_legacy_exports(bytes: &[u8]) {
        let directory = test_temp_root::create("tzolkin-a8b-export-reject").unwrap();
        let source = directory.join("stochastic.json");
        std::fs::write(&source, bytes).unwrap();
        let policy_output = directory.join("policy");
        let critic_output = directory.join("critic");
        assert!(
            crate::policy_dataset::export_native_files(
                std::slice::from_ref(&source),
                &policy_output
            )
            .is_err()
        );
        assert!(
            crate::state_mc_dataset::export_native_files(
                std::slice::from_ref(&source),
                &critic_output
            )
            .is_err()
        );
        assert!(!policy_output.exists());
        assert!(!critic_output.exists());
        std::fs::remove_file(source).unwrap();
        std::fs::remove_dir(directory).unwrap();
    }
}
