//! Actual native Setup sampling with one immutable guarded Playing continuation.
//! A fresh audit reproduces observations, decisions and every core transition.
//! Mechanical episode seals grant no Train reservation, learning owner or adoption.
use serde::Serialize;
use sha2::{Digest, Sha256};
use tzolkin_core::observation::{LegalAction, Observation, observe};
use tzolkin_core::{GameOptions, GameState, Phase, apply_move, create_game_with_options};
use tzolkin_inference::deployment::{
    GUARDED_DEPLOYMENT_VERSION, GuardedDeployment, LoadedDeployment,
};
use tzolkin_inference::trade_guard::{GUARD_VERSION, PROJECTION_VERSION, TradeGuardConfig};

use crate::arena::resource_cycle::{self, ResourceCycleConfig};
use crate::dataset::seed_family_id;
use crate::own_setup_model::{
    BorrowedInitializedOwnSetup, InitializedOwnSetupArtifact, SetupCandidates,
};
use crate::own_setup_record::{
    self, Callback, CallbackRole, Choice, FixedPlayingIdentity, GuardedWire, Header,
    OwnSetupCounts, RECORD_SCHEMA, Record, SAMPLING_VERSION, SOURCE_KIND, SeatRole, SetupSample,
};
use crate::public_stochastic::{
    DENOMINATOR_BITS, RNG_VERSION, TICKS, UNIFORM_MIXTURE, behavior_distribution,
    full_logits_digest, legal_digest, next_splitmix,
};
use crate::public_stochastic_native::{NativeStochasticConfig, numerical_target};
use crate::public_stochastic_record::{
    AttemptStage, CALLBACK_RESERVE, Config, DecisionWire, FINAL_RESERVE, Failure, FailureReason,
    FinalScoreWire, MAX_RECORD_BYTES, TRACE_RESERVE, Terminal, hex64, parse_hex64, serialized_size,
};
use crate::replay::state_key;

pub const MAX_ROWS_PER_CALLBACK: usize = 1024;

/// Closed base generator/stream/cap configuration plus one absolute learner seat.
pub struct OwnSetupNativeConfig {
    native: NativeStochasticConfig,
    learner_seat: usize,
}
impl OwnSetupNativeConfig {
    pub fn new(native: NativeStochasticConfig, learner_seat: usize) -> Result<Self, String> {
        if learner_seat >= native.players() {
            return Err("Own Setup learner seat outside base game".into());
        }
        Ok(Self {
            native,
            learner_seat,
        })
    }
    pub fn native(&self) -> &NativeStochasticConfig {
        &self.native
    }
    pub fn learner_seat(&self) -> usize {
        self.learner_seat
    }
}

/// Retains the same loaded Playing parameters across games/audits without copying.
/// Every run owns fresh guard history; all exits reclaim only the immutable policy.
/// Payload limits bound serialization; each game and its record remain in memory.
pub struct OwnSetupCollector<'a> {
    setup: BorrowedInitializedOwnSetup<'a>,
    playing: Option<LoadedDeployment>,
}
impl<'a> OwnSetupCollector<'a> {
    pub fn new(
        setup: &'a InitializedOwnSetupArtifact,
        playing: LoadedDeployment,
    ) -> Result<Self, String> {
        fixed_playing_identity(&playing)?;
        Ok(Self {
            setup: BorrowedInitializedOwnSetup::new(setup)?,
            playing: Some(playing),
        })
    }
    pub fn playing(&self) -> Result<&LoadedDeployment, String> {
        self.playing
            .as_ref()
            .ok_or_else(|| "Own Setup immutable Playing storage unavailable".into())
    }
    pub fn collect(
        &mut self,
        config: &OwnSetupNativeConfig,
    ) -> Result<CollectedOwnSetupGame, String> {
        self.run(config, None)
            .map(|record| CollectedOwnSetupGame { record })
    }
    pub fn audit(&mut self, bytes: &[u8]) -> Result<AuditedOwnSetupGame, String> {
        let record = own_setup_record::decode(bytes)?;
        let native = record.header.config.checked_config()?;
        let config = OwnSetupNativeConfig::new(native, record.header.learner_seat)?;
        if record.callbacks.len() > config.native.limits().max_callbacks() {
            return Err("Own Setup saved callbacks exceed declared cap".into());
        }
        // All identities and bounds are checked before any fresh model work.
        let actual = header(&config, &self.setup, self.playing()?)?;
        same_bytes(
            &actual,
            &record.header,
            "Own Setup header/immutable identities mismatch",
        )?;
        self.run(&config, Some(&record))
            .map(|record| AuditedOwnSetupGame { record })
    }
    fn run(
        &mut self,
        config: &OwnSetupNativeConfig,
        expected: Option<&Record>,
    ) -> Result<Record, String> {
        // The fixed default satisfies both constructor checks before storage
        // transfers: configuration_key validates the stricter history bounds.
        let guard_config = TradeGuardConfig::default();
        guard_config.configuration_key()?;
        let playing = self
            .playing
            .take()
            .ok_or("Own Setup Playing storage unavailable")?;
        let mut guarded = GuardedDeployment::new(playing, guard_config)?;
        let result = run(config, &self.setup, &mut guarded, expected);
        self.playing = Some(guarded.into_policy());
        result
    }
}

pub struct CollectedOwnSetupGame {
    record: Record,
}
impl CollectedOwnSetupGame {
    pub fn complete(&self) -> bool {
        self.record.terminal.is_some() && self.record.failure.is_none()
    }
    pub fn counts(&self) -> &OwnSetupCounts {
        &self.record.counts
    }
    pub fn failure_reason(&self) -> Option<FailureReason> {
        self.record.failure.as_ref().map(|f| f.reason)
    }
    pub fn checksum(&self) -> &str {
        &self.record.checksum
    }
    pub(crate) fn record(&self) -> &Record {
        &self.record
    }
}
pub struct AuditedOwnSetupGame {
    record: Record,
}
impl AuditedOwnSetupGame {
    pub fn complete(&self) -> bool {
        self.record.terminal.is_some() && self.record.failure.is_none()
    }
    pub fn counts(&self) -> &OwnSetupCounts {
        &self.record.counts
    }
    /// Only a complete fresh audited game containing one applied Setup sample seals.
    pub fn into_episode(self) -> Result<ValidatedOwnSetupEpisode, String> {
        let r = &self.record;
        let c = &r.counts;
        if !self.complete()
            || c.setup_accepted != 1
            || c.setup_applied != 1
            || c.setup_attempts != 1
            || c.setup_reserved_rows != 6
            || c.apply_successes != c.observed_callbacks
            || c.validated_transitions != c.observed_callbacks
            || c.guard_notifications != c.observed_callbacks
            || c.setup_accepted + c.playing_choices + c.opponent_choices != c.observed_callbacks
        {
            return Err("Own Setup episode requires a complete fresh audited single sample".into());
        }
        let setup_index = r
            .callbacks
            .iter()
            .position(|cb| matches!(cb.choice, Some(Choice::SetupSample { .. })))
            .ok_or("Missing applied Own Setup sample")?;
        let scores = &r.terminal.as_ref().ok_or("Missing terminal")?.final_scores;
        let winner_share = winner_share(scores, r.header.learner_seat, r.header.config.players)?;
        let mut hash = Sha256::new();
        hash.update(b"tzolkin-own-setup-audited-episode-v1\0");
        hash.update(r.checksum.as_bytes());
        Ok(ValidatedOwnSetupEpisode {
            record: self.record,
            setup_index,
            winner_share,
            episode_id: format!("{:x}", hash.finalize()),
        })
    }
}

/// Non-Deserialize successful mechanical seal, separate from any learning owner.
pub struct ValidatedOwnSetupEpisode {
    record: Record,
    setup_index: usize,
    winner_share: f64,
    episode_id: String,
}
impl ValidatedOwnSetupEpisode {
    pub fn episode_id(&self) -> &str {
        &self.episode_id
    }
    pub fn record_checksum(&self) -> &str {
        &self.record.checksum
    }
    pub fn family_id(&self) -> &str {
        &self.record.header.native_family_id
    }
    pub fn players(&self) -> usize {
        self.record.header.config.players
    }
    pub fn learner_seat(&self) -> usize {
        self.record.header.learner_seat
    }
    pub fn native_config(&self) -> Result<NativeStochasticConfig, String> {
        self.record.header.config.checked_config()
    }
    pub fn setup_identity(&self) -> &crate::own_setup_model::InitializedOwnSetupIdentity {
        &self.record.header.setup
    }
    pub fn counts(&self) -> &OwnSetupCounts {
        &self.record.counts
    }
    pub fn setup_observation(&self) -> &Observation {
        self.record.callbacks[self.setup_index]
            .observation
            .as_ref()
            .expect("audited admitted Setup")
    }
    pub fn winner_share(&self) -> f64 {
        self.winner_share
    }
    pub fn chosen_index(&self) -> usize {
        self.sample().chosen_index
    }
    pub fn chosen_nominal_probability(&self) -> f64 {
        f64::from_bits(
            parse_hex64(&self.sample().chosen_nominal_probability_bits)
                .expect("audited probability bits"),
        )
    }
    pub fn chosen_behavior_probability(&self) -> f64 {
        f64::from_bits(
            parse_hex64(&self.sample().chosen_behavior_probability_bits)
                .expect("audited behavior bits"),
        )
    }
    pub fn logits_bits(&self) -> &[String] {
        &self.sample().logits_bits
    }
    pub fn mass_ticks(&self) -> &[u64] {
        &self.sample().mass_ticks
    }
    fn sample(&self) -> &SetupSample {
        match self.record.callbacks[self.setup_index]
            .choice
            .as_ref()
            .expect("audited choice")
        {
            Choice::SetupSample { sample, .. } => sample,
            _ => unreachable!("sealed Setup sample"),
        }
    }
}

fn fixed_playing_identity(loaded: &LoadedDeployment) -> Result<FixedPlayingIdentity, String> {
    let identity = loaded
        .current_mixed_identity()
        .ok_or("Own Setup continuation requires actual current-mixed deployment")?;
    if loaded.backend() != "scalar" || loaded.source_role() != "rlCurrentMixed" {
        return Err("Own Setup continuation requires current-mixed Scalar".into());
    }
    Ok(FixedPlayingIdentity {
        deployment_checksum: loaded.checksum().into(),
        source_role: loaded.source_role().into(),
        source_checksum: loaded.source_checksum().into(),
        update_count: loaded.update_count(),
        current_mixed_identity: serde_json::from_slice(
            &serde_json::to_vec(identity).map_err(|e| e.to_string())?,
        )
        .map_err(|e| e.to_string())?,
        source_numerical_target: loaded.source_numerical_target().into(),
        raw_policy_version: loaded.policy_version().into(),
        backend: loaded.backend().into(),
        guard: TradeGuardConfig::default(),
        configuration_key: TradeGuardConfig::default().configuration_key()?,
        inference_version: GUARDED_DEPLOYMENT_VERSION.into(),
        guard_version: GUARD_VERSION.into(),
        projection_version: PROJECTION_VERSION.into(),
    })
}
fn header(
    config: &OwnSetupNativeConfig,
    setup: &BorrowedInitializedOwnSetup<'_>,
    playing: &LoadedDeployment,
) -> Result<Header, String> {
    Ok(Header {
        source_kind: SOURCE_KIND.into(),
        config: Config::from_config(&config.native),
        learner_seat: config.learner_seat,
        native_family_id: seed_family_id(config.native.environment_seed()),
        setup: setup.identity().clone(),
        playing: fixed_playing_identity(playing)?,
        roles: (0..config.native.players())
            .map(|seat| {
                if seat == config.learner_seat {
                    SeatRole::LearnerSetupFixedPlaying
                } else {
                    SeatRole::V5Opponent
                }
            })
            .collect(),
        opponent: ResourceCycleConfig::v5(),
        sampling_version: SAMPLING_VERSION.into(),
        rng_version: RNG_VERSION.into(),
        numerical_target: numerical_target(),
        denominator_bits: DENOMINATOR_BITS,
        uniform_mixture_bits: hex64(UNIFORM_MIXTURE.to_bits()),
        producer_authenticated: false,
        independent_seed_origins_verified: false,
        training_admission: "unavailable-own-setup-owner-required".into(),
    })
}
fn same_bytes<A: Serialize, B: Serialize>(a: &A, b: &B, message: &str) -> Result<(), String> {
    if serde_json::to_vec(a).map_err(|e| e.to_string())?
        != serde_json::to_vec(b).map_err(|e| e.to_string())?
    {
        return Err(message.into());
    }
    Ok(())
}
fn state_valid(state: &GameState) -> Result<(), String> {
    if !tzolkin_core::validation::validate_game_state(
        &serde_json::to_value(state).map_err(|e| e.to_string())?,
    ) {
        return Err("Own Setup core state invariant failed".into());
    }
    Ok(())
}
fn failed(
    record: &mut Record,
    reason: FailureReason,
    stage: AttemptStage,
    index: Option<usize>,
    message: &str,
) {
    let failure = Failure::new(reason, stage, index, message);
    if let Some(index) = index
        && let Some(cb) = record.callbacks.get_mut(index)
    {
        cb.failure = Some(failure.clone());
    }
    record.failure = Some(failure);
    record.terminal = None;
}
fn reserve(used: &mut usize, amount: usize, limit: usize) -> Result<(), String> {
    *used = used
        .checked_add(amount)
        .filter(|n| *n <= limit)
        .ok_or("Own Setup reservation limit")?;
    Ok(())
}
fn sample(
    setup: &BorrowedInitializedOwnSetup<'_>,
    o: &Observation,
    rng: u64,
) -> Result<SetupSample, String> {
    let candidates = SetupCandidates::from_observation(o)?;
    let logits = setup.predict(&candidates)?.logits;
    let (nominal, masses, distribution_digest) = behavior_distribution(&logits)?;
    if logits.len() != 6 || masses.iter().sum::<u64>() != TICKS {
        return Err("Own Setup requires six full positive masses".into());
    }
    let ordered_legal_digest = legal_digest(&o.legal_actions)?;
    let logits_digest = full_logits_digest(&logits);
    let (after, raw) = next_splitmix(rng);
    let u53 = raw >> 11;
    let mut cumulative = 0;
    let chosen = masses
        .iter()
        .position(|mass| {
            cumulative += mass;
            u53 < cumulative
        })
        .ok_or("Own Setup draw outside mass")?;
    let behavior = masses[chosen] as f64 / TICKS as f64;
    Ok(SetupSample {
        chosen_index: chosen,
        logits_bits: logits
            .iter()
            .map(|z| format!("{:08x}", z.to_bits()))
            .collect(),
        logits_digest,
        ordered_legal_digest,
        nominal_probability_bits: nominal.iter().map(|p| hex64(p.to_bits())).collect(),
        mass_ticks: masses,
        distribution_digest,
        chosen_nominal_probability_bits: hex64(nominal[chosen].to_bits()),
        chosen_behavior_probability_bits: hex64(behavior.to_bits()),
        chosen_behavior_logp_bits: hex64(behavior.ln().to_bits()),
        rng_state_before: hex64(rng),
        rng_state_after: hex64(after),
        raw_u64: hex64(raw),
        u53,
        draws_before: hex64(0),
        draws_after: hex64(1),
    })
}
fn unique_action(
    o: &Observation,
    actor: usize,
    key: &str,
    score: f64,
    mv: &tzolkin_core::GameMove,
) -> Result<LegalAction, String> {
    let actions: Vec<_> = o.legal_actions.iter().filter(|a| &a.r#move == mv).collect();
    if actor != o.actor || key != o.observation_key || !score.is_finite() || actions.len() != 1 {
        return Err("Own Setup decision/current legal binding mismatch".into());
    }
    Ok(actions[0].clone())
}

fn run(
    config: &OwnSetupNativeConfig,
    setup: &BorrowedInitializedOwnSetup<'_>,
    guard: &mut GuardedDeployment,
    expected: Option<&Record>,
) -> Result<Record, String> {
    let native = &config.native;
    let limits = native.limits();
    let header = header(config, setup, guard.policy())?;
    let initial_reserved = serialized_size(&header, limits.max_source_bytes())?
        .checked_add(FINAL_RESERVE)
        .ok_or("Own Setup header reservation overflow")?;
    if initial_reserved > limits.max_source_bytes() {
        return Err("Own Setup header exceeds source cap".into());
    }
    let mut record = Record {
        schema: RECORD_SCHEMA.into(),
        header,
        callbacks: Vec::new(),
        terminal: None,
        failure: None,
        counts: OwnSetupCounts {
            actor_draws: vec![hex64(0); native.players()],
            source_reserved_bytes: initial_reserved,
            ..OwnSetupCounts::default()
        },
        checksum: "0".repeat(64),
    };
    let start = create_game_with_options(
        (0..native.players())
            .map(|seat| format!("P{seat}"))
            .collect(),
        native.environment_seed(),
        GameOptions::default(),
    );
    let mut state = match start {
        Ok(state) => state,
        Err(error) => {
            failed(
                &mut record,
                FailureReason::CreationError,
                AttemptStage::Startup,
                None,
                &error,
            );
            return finish(record, config, expected);
        }
    };
    if let Err(error) = state_valid(&state) {
        failed(
            &mut record,
            FailureReason::CoreValidationError,
            AttemptStage::Startup,
            None,
            &error,
        );
    }
    while record.failure.is_none() && state.phase != Phase::Finished {
        let index = record.callbacks.len();
        if index >= limits.max_callbacks() {
            failed(
                &mut record,
                FailureReason::CallbackLimit,
                AttemptStage::BeforeObservation,
                None,
                "Own Setup callback cap reached",
            );
            break;
        }
        let observation = match observe(&state, state.current_player) {
            Ok(o) => o,
            Err(error) => {
                failed(
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
                failed(
                    &mut record,
                    FailureReason::ObservationError,
                    AttemptStage::BeforeObservation,
                    None,
                    &error,
                );
                break;
            }
        };
        let reservation = serialized_size(&observation, limits.max_source_bytes())
            .ok()
            .and_then(|n| {
                observation
                    .legal_actions
                    .iter()
                    .map(|a| serialized_size(a, limits.max_source_bytes()))
                    .collect::<Result<Vec<_>, _>>()
                    .ok()
                    .and_then(|sizes| sizes.into_iter().max())
                    .and_then(|a| a.checked_mul(2))
                    .and_then(|a| {
                        Some((
                            n.checked_add(a)?.checked_add(CALLBACK_RESERVE)?,
                            a.checked_add(TRACE_RESERVE)?,
                        ))
                    })
            });
        let admitted = reservation.is_some_and(|(n, _)| {
            record
                .counts
                .source_reserved_bytes
                .checked_add(n)
                .is_some_and(|n| n <= limits.max_source_bytes())
        });
        if let Some(expected) = expected {
            let claimed = expected
                .callbacks
                .get(index)
                .ok_or("Own Setup audit missing current callback")?;
            if claimed.state_before != before
                || claimed.actor != observation.actor
                || claimed.observation_key != observation.observation_key
            {
                return Err("Own Setup audit state/actor/key mismatch before choice".into());
            }
            match (&claimed.observation, admitted) {
                (Some(saved), true) => same_bytes(
                    saved,
                    &observation,
                    "Own Setup fresh observation mismatch before choice",
                )?,
                (None, false) => (),
                _ => return Err("Own Setup audit observation admission mismatch".into()),
            }
        }
        let role = if observation.actor != config.learner_seat {
            CallbackRole::V5Opponent
        } else if observation.phase == Phase::Setup {
            CallbackRole::SetupSample
        } else {
            CallbackRole::FixedPlaying
        };
        record.callbacks.push(Callback {
            index,
            actor: observation.actor,
            turn_player: observation.turn_player,
            observation_key: observation.observation_key.clone(),
            observation: admitted.then(|| observation.clone()),
            state_before: before,
            role,
            guard_notification_attempted: false,
            guard_notified: false,
            choice_attempted: false,
            choice: None,
            apply_attempted: false,
            apply_succeeded: false,
            state_after: None,
            state_validated: false,
            failure: None,
        });
        record.counts.observed_callbacks += 1;
        let result = step(
            &mut record,
            config,
            setup,
            guard,
            &mut state,
            &observation,
            reservation.filter(|_| admitted),
        );
        if let Err((reason, stage, message)) = result {
            failed(&mut record, reason, stage, Some(index), &message);
        }
        if let Some(expected) = expected {
            same_bytes(
                &record.callbacks[index],
                &expected.callbacks[index],
                "Own Setup fresh choice/apply/failure mismatch",
            )?;
        }
    }
    if record.failure.is_none() && state.phase == Phase::Finished {
        if let Err(error) = state_valid(&state).and_then(|()| {
            winner_share(
                &state
                    .final_scores
                    .iter()
                    .map(FinalScoreWire::from_score)
                    .collect::<Vec<_>>(),
                config.learner_seat,
                native.players(),
            )
            .map(|_| ())
        }) {
            failed(
                &mut record,
                FailureReason::CoreValidationError,
                AttemptStage::TerminalValidation,
                None,
                &error,
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
                Err(error) => failed(
                    &mut record,
                    FailureReason::CoreValidationError,
                    AttemptStage::TerminalValidation,
                    None,
                    &error,
                ),
            }
        }
    }
    finish(record, config, expected)
}

type StepFailure = (FailureReason, AttemptStage, String);
fn step(
    record: &mut Record,
    config: &OwnSetupNativeConfig,
    setup: &BorrowedInitializedOwnSetup<'_>,
    guard: &mut GuardedDeployment,
    state: &mut GameState,
    observation: &Observation,
    reservation: Option<(usize, usize)>,
) -> Result<(), StepFailure> {
    let index = record.callbacks.len() - 1;
    let limits = config.native.limits();
    let before_choice = |reason, message: String| (reason, AttemptStage::BeforeSample, message);
    let (source_size, trace_size) = reservation.ok_or_else(|| {
        before_choice(
            FailureReason::SourcePayloadLimit,
            "Own Setup source reservation refused".into(),
        )
    })?;
    reserve(
        &mut record.counts.source_reserved_bytes,
        source_size,
        limits.max_source_bytes(),
    )
    .map_err(|e| before_choice(FailureReason::SourcePayloadLimit, e))?;
    let rows = observation.legal_actions.len();
    if rows == 0 || rows > MAX_ROWS_PER_CALLBACK {
        return Err(before_choice(
            FailureReason::CandidateLimit,
            "Own Setup full-mask row cap".into(),
        ));
    }
    // Choice envelopes have fixed fields; reserve both copies of the largest
    // actual legal action before any expert prediction or stochastic draw.
    reserve(
        &mut record.counts.trace_reserved_bytes,
        trace_size,
        limits.max_trace_bytes(),
    )
    .map_err(|e| before_choice(FailureReason::TracePayloadLimit, e))?;
    reserve(
        &mut record.counts.all_role_reserved_rows,
        rows,
        limits.max_candidate_rows(),
    )
    .map_err(|e| before_choice(FailureReason::CandidateLimit, e))?;
    if record.callbacks[index].role == CallbackRole::SetupSample {
        record.counts.setup_reserved_rows += rows;
    }
    record.callbacks[index].guard_notification_attempted = true;
    record.counts.guard_notification_attempts += 1;
    guard
        .on_callback(index, observation)
        .map_err(|e| before_choice(FailureReason::ModelOrSamplingError, e))?;
    record.callbacks[index].guard_notified = true;
    record.counts.guard_notifications += 1;
    record.callbacks[index].choice_attempted = true;
    let role = record.callbacks[index].role;
    let choice = match role {
        CallbackRole::SetupSample => {
            record.counts.setup_attempts += 1;
            if record.counts.setup_attempts != 1 || rows != 6 {
                return Err(before_choice(
                    FailureReason::ModelOrSamplingError,
                    "Own Setup requires its single six-pair sample".into(),
                ));
            }
            let rng = config
                .native
                .sampling_identity()
                .initial_actor_state(config.learner_seat)
                .map_err(|e| before_choice(FailureReason::ModelOrSamplingError, e))?;
            let sample = sample(setup, observation, rng)
                .map_err(|e| before_choice(FailureReason::ModelOrSamplingError, e))?;
            let chosen = observation.legal_actions[sample.chosen_index].clone();
            record.counts.actor_draws[config.learner_seat] = hex64(1);
            record.counts.setup_accepted += 1;
            Choice::SetupSample { sample, chosen }
        }
        CallbackRole::FixedPlaying => {
            record.counts.playing_attempts += 1;
            if observation.phase != Phase::Playing {
                return Err(before_choice(
                    FailureReason::ModelOrSamplingError,
                    "Own Setup continuation requires Playing".into(),
                ));
            }
            let decision = guard
                .choose_move(index, observation)
                .map_err(|e| before_choice(FailureReason::ModelOrSamplingError, e))?;
            let chosen = unique_action(
                observation,
                decision.decision.actor,
                &decision.decision.observation_key,
                decision.decision.score,
                &decision.decision.r#move,
            )
            .map_err(|e| before_choice(FailureReason::DecisionMismatch, e))?;
            record.counts.playing_choices += 1;
            Choice::FixedPlaying {
                guarded: GuardedWire::from_decision(decision),
                chosen,
            }
        }
        CallbackRole::V5Opponent => {
            record.counts.opponent_attempts += 1;
            let decision = resource_cycle::choose_move(observation, &record.header.opponent)
                .map_err(|e| before_choice(FailureReason::ModelOrSamplingError, e))?;
            if decision.policy_version != resource_cycle::V5_POLICY_VERSION {
                return Err(before_choice(
                    FailureReason::DecisionMismatch,
                    "Own Setup V5 version mismatch".into(),
                ));
            }
            let chosen = unique_action(
                observation,
                decision.actor,
                &decision.observation_key,
                decision.score,
                &decision.r#move,
            )
            .map_err(|e| before_choice(FailureReason::DecisionMismatch, e))?;
            record.counts.opponent_choices += 1;
            Choice::V5Opponent {
                decision: DecisionWire {
                    actor: decision.actor,
                    observation_key: decision.observation_key,
                    policy_version: decision.policy_version,
                    r#move: decision.r#move,
                    score_bits: hex64(decision.score.to_bits()),
                },
                chosen,
            }
        }
    };
    if choice.role() != role {
        return Err(before_choice(
            FailureReason::DecisionMismatch,
            "Own Setup exclusive role mismatch".into(),
        ));
    }
    record.callbacks[index].choice = Some(choice);
    let choice = record.callbacks[index].choice.as_ref().unwrap();
    serialized_size(choice, trace_size).map_err(|e| {
        (
            FailureReason::TracePayloadLimit,
            AttemptStage::BeforeApply,
            e,
        )
    })?;
    record.callbacks[index].apply_attempted = true;
    record.counts.apply_attempts += 1;
    *state = apply_move(
        state,
        record.callbacks[index]
            .choice
            .as_ref()
            .unwrap()
            .chosen()
            .r#move
            .clone(),
    )
    .map_err(|e| (FailureReason::CoreApplyError, AttemptStage::BeforeApply, e))?;
    record.callbacks[index].apply_succeeded = true;
    record.counts.apply_successes += 1;
    if role == CallbackRole::SetupSample {
        record.counts.setup_applied += 1;
    }
    record.callbacks[index].state_after = Some(state_key(state).map_err(|e| {
        (
            FailureReason::CoreValidationError,
            AttemptStage::AfterApply,
            e,
        )
    })?);
    state_valid(state).map_err(|e| {
        (
            FailureReason::CoreValidationError,
            AttemptStage::AfterApply,
            e,
        )
    })?;
    record.callbacks[index].state_validated = true;
    record.counts.validated_transitions += 1;
    Ok(())
}

fn winner_share(scores: &[FinalScoreWire], learner: usize, players: usize) -> Result<f64, String> {
    if scores.len() != players
        || (0..players).any(|seat| scores.iter().filter(|s| s.player_id == seat).count() != 1)
        || scores.iter().any(|s| {
            s.rank == 0
                || s.rank > players
                || !f64::from_bits(parse_hex64(&s.total_bits).unwrap_or(u64::MAX)).is_finite()
        })
    {
        return Err("Own Setup terminal coverage/rank/finite score mismatch".into());
    }
    let winners = scores.iter().filter(|s| s.rank == 1).count();
    if winners == 0 {
        return Err("Own Setup terminal missing first rank".into());
    }
    Ok(
        if scores.iter().any(|s| s.player_id == learner && s.rank == 1) {
            1.0 / winners as f64
        } else {
            0.0
        },
    )
}
fn source_size(record: &Record) -> Result<usize, String> {
    // Exact sizes of source fragments excluding the separately charged choices.
    let mut bytes = serialized_size(
        &(
            &record.header,
            &record.terminal,
            &record.failure,
            &record.counts,
        ),
        MAX_RECORD_BYTES,
    )?;
    for cb in &record.callbacks {
        let source = (
            &cb.index,
            &cb.actor,
            &cb.turn_player,
            &cb.observation_key,
            &cb.observation,
            &cb.state_before,
            &cb.role,
            &cb.guard_notification_attempted,
            &cb.guard_notified,
            &cb.choice_attempted,
            cb.choice.as_ref().map(Choice::chosen),
            &cb.apply_attempted,
            &cb.apply_succeeded,
            &cb.state_after,
            &cb.state_validated,
            &cb.failure,
        );
        bytes = bytes
            .checked_add(serialized_size(&source, MAX_RECORD_BYTES)?)
            .ok_or("Own Setup source size overflow")?;
    }
    Ok(bytes)
}
fn finish(
    mut record: Record,
    config: &OwnSetupNativeConfig,
    expected: Option<&Record>,
) -> Result<Record, String> {
    record.counts.stopped = true;
    record.counts.actual_trace_bytes = record
        .callbacks
        .iter()
        .filter_map(|cb| cb.choice.as_ref())
        .try_fold(0usize, |n, choice| {
            n.checked_add(serialized_size(choice, MAX_RECORD_BYTES)?)
                .ok_or_else(|| "Own Setup trace size overflow".to_string())
        })?;
    // Self-count fields change only decimal widths; bounded fixed-point closure.
    let mut closed = false;
    for _ in 0..8 {
        let source = source_size(&record)?;
        let wire = serialized_size(&record, MAX_RECORD_BYTES)?;
        if record.counts.actual_source_bytes == source && record.counts.actual_record_bytes == wire
        {
            closed = true;
            break;
        }
        record.counts.actual_source_bytes = source;
        record.counts.actual_record_bytes = wire;
    }
    let limits = config.native.limits();
    if !closed
        || record.counts.actual_source_bytes > record.counts.source_reserved_bytes
        || record.counts.actual_trace_bytes > record.counts.trace_reserved_bytes
        || record.counts.actual_record_bytes
            > limits
                .max_source_bytes()
                .checked_add(limits.max_trace_bytes())
                .ok_or("Own Setup byte cap overflow")?
    {
        return Err("Own Setup actual payload exceeds closed reservation".into());
    }
    record.checksum = own_setup_record::checksum(&record)?;
    if let Some(expected) = expected {
        same_bytes(
            &record,
            expected,
            "Own Setup complete record/failure/counter mismatch",
        )?;
    }
    Ok(record)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::public_stochastic::{SamplingSeed, SamplingStreamIdentity};
    use crate::public_stochastic_native::CollectionLimits;
    use serde_json::json;
    use tzolkin_inference::{deployment as d, features, policy};

    // Synthetic numeric-only identity with actual zero-word actor fingerprint;
    // this fixture supplies no native learner or trained owner.
    fn playing() -> LoadedDeployment {
        let target = numerical_target();
        let mut hash = Sha256::new();
        hash.update(b"tzolkin-mixed-inference-actor-v1\0");
        hash.update(serde_json::to_vec(&vec![0.0_f32; policy::PARAMETER_COUNT]).unwrap());
        let mut wire = json!({
            "schema": d::DEPLOYMENT_SCHEMA,
            "contract": {"modelVersion":d::MODEL_VERSION,"inputContract":d::INPUT_CONTRACT,
                "featureSchema":features::PUBLIC_FEATURE_SCHEMA,"featureCount":features::FEATURE_COUNT,
                "hiddenCount":policy::HIDDEN,"parameterCount":policy::PARAMETER_COUNT,
                "rulesVersion":d::RULES_VERSION,"rulesBaseline":d::RULES_BASELINE,
                "observationSchema":tzolkin_core::observation::OBSERVATION_SCHEMA,
                "moveSchema":tzolkin_core::observation::MOVE_SCHEMA,"catalogHash":d::catalog_hash(),
                "backend":"scalar","numericContract":d::NUMERIC_CONTRACT,"scope":d::SCOPE,"valueValidity":d::VALUE_VALIDITY},
            "source":{"kind":"rlCurrentMixed","identity":{"kind":"initialMixedV1","policy":{
                "schema":"tzolkin-public-rl-mixed-joint-inference-identity-v1",
                "policyVersion":"learned-public-rl-mixed-joint-argmax-v1",
                "samplingVersion":"public-stochastic-rl-mixed-joint-uniform-tick53-v1",
                "task":"mixedLearnerOwnSumQuarterJointOneBatchV1","artifactChecksum":"a".repeat(64),
                "importedUpdateCount":1,"mixedUpdateCount":1,"actorModelVersion":d::MODEL_VERSION,
                "criticModelVersion":"tiny-public-state-critic-mlp-v1","actorParametersChecksum":format!("{:x}",hash.finalize()),
                "residualParametersChecksum":"b".repeat(64),"featureSchema":2,"inputContract":d::INPUT_CONTRACT,
                "contextSchema":1,"contextContract":"public-v2-state-prefix384-v1","backend":"scalar","numericalTarget":target}}},
            "sourceNumericalTarget":target,"parameterBits":vec!["00000000";policy::PARAMETER_COUNT],"checksum":""
        });
        wire["checksum"] = d::numeric_checksum(&serde_json::to_vec(&wire).unwrap())
            .unwrap()
            .into();
        LoadedDeployment::load(&serde_json::to_vec(&wire).unwrap()).unwrap()
    }
    fn config(
        players: usize,
        learner: usize,
        callbacks: usize,
        rows: usize,
    ) -> OwnSetupNativeConfig {
        OwnSetupNativeConfig::new(
            NativeStochasticConfig::new(
                players,
                17,
                SamplingStreamIdentity::new(SamplingSeed::new(u64::MAX), u64::MAX - 1, 7, players)
                    .unwrap(),
                CollectionLimits::new(callbacks, rows, 1024 * 1024, 1024 * 1024).unwrap(),
            )
            .unwrap(),
            learner,
        )
        .unwrap()
    }
    fn resealed(mut record: Record) -> Vec<u8> {
        record.checksum = own_setup_record::checksum(&record).unwrap();
        serde_json::to_vec(&record).unwrap()
    }

    #[test]
    fn real_setup_prefix_audit_and_guarded_playing_parity_recover_same_storage() {
        let artifact = InitializedOwnSetupArtifact::new(7).unwrap();
        let mut collector = OwnSetupCollector::new(&artifact, playing()).unwrap();
        let checksum = collector.playing().unwrap().checksum().to_string();
        for (players, learner, callbacks) in [(3, 0, 4), (4, 3, 4)] {
            let config = config(players, learner, callbacks, 1024);
            let collected = collector.collect(&config).unwrap();
            assert!(!collected.complete());
            assert_eq!(
                collected.failure_reason(),
                Some(FailureReason::CallbackLimit)
            );
            let counts = collected.counts();
            assert_eq!(
                (
                    counts.setup_attempts,
                    counts.setup_accepted,
                    counts.setup_applied,
                    counts.setup_reserved_rows
                ),
                (1, 1, 1, 6)
            );
            assert_eq!(
                (
                    counts.observed_callbacks,
                    counts.guard_notifications,
                    counts.apply_successes,
                    counts.validated_transitions
                ),
                (callbacks, callbacks, callbacks, callbacks)
            );
            assert_eq!(
                counts.setup_accepted + counts.playing_choices + counts.opponent_choices,
                callbacks
            );
            assert_eq!(
                counts.actor_draws,
                (0..players)
                    .map(|seat| hex64(u64::from(seat == learner)))
                    .collect::<Vec<_>>()
            );
            let bytes = own_setup_record::encode_record(&collected).unwrap();
            assert!(collector.audit(&bytes).unwrap().into_episode().is_err());
            assert_eq!(collector.playing().unwrap().checksum(), checksum);
            assert_eq!(
                bytes,
                own_setup_record::encode_record(&collector.collect(&config).unwrap()).unwrap()
            );

            let sample = collected
                .record
                .callbacks
                .iter()
                .find_map(|cb| match cb.choice.as_ref() {
                    Some(Choice::SetupSample { sample, .. }) => Some(sample),
                    _ => None,
                })
                .unwrap();
            assert_eq!(sample.logits_bits.len(), 6);
            assert_eq!(sample.mass_ticks.iter().sum::<u64>(), TICKS);
            let lower = sample.mass_ticks[..sample.chosen_index].iter().sum::<u64>();
            assert!(
                lower <= sample.u53 && sample.u53 < lower + sample.mass_ticks[sample.chosen_index]
            );
            assert_eq!(sample.u53, parse_hex64(&sample.raw_u64).unwrap() >> 11);
            assert_eq!(sample.draws_before, hex64(0));
            assert_eq!(sample.draws_after, hex64(1));
            assert_eq!(
                f64::from_bits(parse_hex64(&sample.chosen_behavior_probability_bits).unwrap())
                    .to_bits(),
                (sample.mass_ticks[sample.chosen_index] as f64 / TICKS as f64).to_bits()
            );

            // Independent reference guard sees fresh native observations for
            // all actors; saved observations never drive this reference engine.
            let mut reference =
                GuardedDeployment::new(playing(), TradeGuardConfig::default()).unwrap();
            let mut state = create_game_with_options(
                (0..players).map(|seat| format!("P{seat}")).collect(),
                17,
                GameOptions::default(),
            )
            .unwrap();
            for cb in &collected.record.callbacks {
                let o = observe(&state, state.current_player).unwrap();
                reference.on_callback(cb.index, &o).unwrap();
                if let Some(Choice::FixedPlaying { guarded, .. }) = &cb.choice {
                    let expected =
                        GuardedWire::from_decision(reference.choose_move(cb.index, &o).unwrap());
                    same_bytes(guarded, &expected, "guarded reference mismatch").unwrap();
                }
                state = apply_move(&state, cb.choice.as_ref().unwrap().chosen().r#move.clone())
                    .unwrap();
            }
            if learner == 0 {
                assert_eq!(counts.playing_choices, 1);
            }

            let mut tampered = own_setup_record::decode(&bytes).unwrap();
            tampered.callbacks[0].observation.as_mut().unwrap().players[0].resources[0] += 1;
            assert!(collector.audit(&resealed(tampered)).is_err());
            let mut tampered = own_setup_record::decode(&bytes).unwrap();
            if let Some(Choice::SetupSample { sample, .. }) =
                tampered.callbacks.iter_mut().find_map(|cb| {
                    cb.choice
                        .as_mut()
                        .filter(|choice| matches!(choice, Choice::SetupSample { .. }))
                })
            {
                sample.mass_ticks[0] += 1;
            }
            assert!(collector.audit(&resealed(tampered)).is_err());
            assert!(collector.audit(&bytes).is_ok());
            assert_eq!(collector.playing().unwrap().checksum(), checksum);
        }
    }

    #[test]
    fn row_cap_prevents_draw_and_closed_codec_rejects_unknown_missing_duplicate_and_old_records() {
        let artifact = InitializedOwnSetupArtifact::new(7).unwrap();
        let mut collector = OwnSetupCollector::new(&artifact, playing()).unwrap();
        let game = collector.collect(&config(3, 0, 1, 5)).unwrap();
        assert_eq!(game.failure_reason(), Some(FailureReason::CandidateLimit));
        assert_eq!(
            (
                game.counts().observed_callbacks,
                game.counts().setup_attempts,
                game.counts().setup_accepted,
                game.counts().apply_attempts
            ),
            (1, 0, 0, 0)
        );
        assert_eq!(game.counts().actor_draws, vec![hex64(0); 3]);
        let bytes = own_setup_record::encode_record(&game).unwrap();
        assert!(collector.audit(&bytes).is_ok());
        let trace_cap = OwnSetupNativeConfig::new(
            NativeStochasticConfig::new(
                3,
                17,
                *config(3, 0, 1, 6).native.sampling_identity(),
                CollectionLimits::new(1, 6, 1024 * 1024, TRACE_RESERVE).unwrap(),
            )
            .unwrap(),
            0,
        )
        .unwrap();
        let refused = collector.collect(&trace_cap).unwrap();
        assert_eq!(
            refused.failure_reason(),
            Some(FailureReason::TracePayloadLimit)
        );
        assert_eq!(
            (
                refused.counts().guard_notifications,
                refused.counts().setup_attempts,
                refused.counts().setup_accepted,
                refused.counts().apply_attempts
            ),
            (0, 0, 0, 0)
        );
        assert_eq!(refused.counts().actor_draws, vec![hex64(0); 3]);
        assert!(
            collector
                .audit(&own_setup_record::encode_record(&refused).unwrap())
                .is_ok()
        );
        assert!(
            crate::public_stochastic_record::decode_typed_record::<crate::replay::SeatPolicy>(
                &bytes
            )
            .is_err()
        );
        let text = String::from_utf8(bytes.clone()).unwrap();
        let duplicate = text.replacen("\"schema\":", "\"schema\":\"duplicate\",\"schema\":", 1);
        assert!(own_setup_record::decode(duplicate.as_bytes()).is_err());
        let mut missing = serde_json::from_slice::<serde_json::Value>(&bytes).unwrap();
        missing.as_object_mut().unwrap().remove("terminal");
        assert!(own_setup_record::decode(&serde_json::to_vec(&missing).unwrap()).is_err());
        let mut unknown = serde_json::from_slice::<serde_json::Value>(&bytes).unwrap();
        unknown["callbacks"][0]["observation"]["turn"]["unknown"] = true.into();
        assert!(own_setup_record::decode(&serde_json::to_vec(&unknown).unwrap()).is_err());
        let mut old = serde_json::from_slice::<serde_json::Value>(&bytes).unwrap();
        old["schema"] = crate::public_stochastic_record::RECORD_SCHEMA.into();
        assert!(own_setup_record::decode(&serde_json::to_vec(&old).unwrap()).is_err());
        let mut identity = own_setup_record::decode(&bytes).unwrap();
        identity.header.playing.source_numerical_target = "scalar-f32-f64-tick53-v1:a:b:64".into();
        assert!(collector.audit(&resealed(identity)).is_err());
        assert!(collector.audit(&bytes).is_ok());
    }

    #[test]
    fn terminal_winner_share_uses_tied_first_ranks_and_refuses_missing_coverage() {
        let score = |player_id, rank| FinalScoreWire {
            player_id,
            rank,
            points_before_final_bits: hex64(0.0f64.to_bits()),
            resource_points_bits: hex64(0.0f64.to_bits()),
            skull_points_bits: hex64(0.0f64.to_bits()),
            monument_points_bits: hex64(0.0f64.to_bits()),
            total_bits: hex64(if rank == 1 {
                100.0f64.to_bits()
            } else {
                80.0f64.to_bits()
            }),
            workers_on_gears: 0,
        };
        let scores = vec![score(0, 1), score(1, 1), score(2, 3)];
        assert_eq!(winner_share(&scores, 0, 3).unwrap(), 0.5);
        assert_eq!(winner_share(&scores, 2, 3).unwrap(), 0.0);
        assert!(winner_share(&scores[..2], 0, 3).is_err());
        assert!(winner_share(&[score(0, 1), score(0, 1), score(2, 3)], 0, 3).is_err());
    }
}
