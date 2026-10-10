//! Library-only, opt-in categorical sampling from an immutable native V2 policy.
//!
//! Scalar f32 logits are widened into a new f64 softmax with a fixed `2^-20`
//! uniform mixture. The actual behavior mass is the positive integer CDF
//! interval divided by `2^53`, not the nominal softmax probability. Its logp
//! includes the mixture and tick rounding. A future continuous-policy gradient
//! would require a separate approximation contract; this module implements no PPO.
//!
//! The caller supplies the complete ordered core Observation. This boundary
//! does not independently reconstruct omitted legal actions or authenticate a
//! sampling seed's independence from an engine seed. Legitimate native Setup
//! retains the actor's private offer; Playing uses the existing public V2 encoder.
//! Same-build/platform f64 math is fixed here; cross-platform exp/ln bits are not
//! guaranteed. Admission row counts are distinct from measured forward counts.
//! A trace attests sampling consistency, not core application, producer identity,
//! complete-game collection, or training eligibility. Source wire/admission,
//! collector, CLI, policy updates and optimizer are separate future units.
use std::io::{self, Write};

use sha2::{Digest, Sha256};
use tzolkin_core::observation::{LegalAction, Observation};

use crate::Decision;
use crate::features::PUBLIC_FEATURE_SCHEMA;
use crate::model::{MAX_CANDIDATES, digest};
use crate::public_model::{
    MAX_OBSERVATION_BYTES, POLICY_VERSION, PublicPolicyDistribution, ValueValidity,
};
use crate::public_native::PublicPolicyHandle;
use crate::public_rl_native::{PublicRlPolicyDistribution, UpdatedPublicRlHandle};
use crate::replay::SeatPolicy;

pub const SAMPLING_VERSION: &str = "public-stochastic-bc-uniform-tick53-v1";
pub const RL_SAMPLING_VERSION: &str = "public-stochastic-rl-count1-uniform-tick53-v1";
pub const RNG_VERSION: &str = "splitmix64-actor-domain-sha256-v1";
pub const DENOMINATOR_BITS: u32 = 53;
pub const TICKS: u64 = 1_u64 << DENOMINATOR_BITS;
pub const UNIFORM_MIXTURE: f64 = 1.0 / (1_u64 << 20) as f64;
pub const MAX_SAMPLES: usize = 4000;
pub const MAX_RESERVED_CANDIDATE_ROWS: usize = 1_000_000;
const RNG_DOMAIN: &[u8] = b"tzolkin-public-stochastic-actor-stream-v1\0";

/// An independent sampling seed. No engine seed or world is accepted by this module.
/// The caller must establish independence; this type does not authenticate its origin.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SamplingSeed(u64);
impl SamplingSeed {
    pub const fn new(value: u64) -> Self {
        Self(value)
    }
    pub const fn value(self) -> u64 {
        self.0
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SamplingStreamIdentity {
    seed: SamplingSeed,
    episode_ordinal: u64,
    replicate_ordinal: u64,
    players: usize,
}
impl SamplingStreamIdentity {
    pub fn new(
        seed: SamplingSeed,
        episode_ordinal: u64,
        replicate_ordinal: u64,
        players: usize,
    ) -> Result<Self, String> {
        if !(3..=4).contains(&players) {
            return Err("Stochastic sampling requires base 3/4p".into());
        }
        Ok(Self {
            seed,
            episode_ordinal,
            replicate_ordinal,
            players,
        })
    }
    pub fn sampling_seed(&self) -> SamplingSeed {
        self.seed
    }
    pub fn players(&self) -> usize {
        self.players
    }
    pub fn episode_ordinal(&self) -> u64 {
        self.episode_ordinal
    }
    pub fn replicate_ordinal(&self) -> u64 {
        self.replicate_ordinal
    }
    pub fn initial_actor_state(&self, actor: usize) -> Result<u64, String> {
        if actor >= self.players {
            return Err("Invalid absolute sampling actor".into());
        }
        let mut hash = Sha256::new();
        hash.update(RNG_DOMAIN);
        for word in [
            self.seed.0,
            self.episode_ordinal,
            self.replicate_ordinal,
            self.players as u64,
            actor as u64,
        ] {
            hash.update(word.to_le_bytes());
        }
        let bytes = hash.finalize();
        Ok(u64::from_le_bytes(
            bytes[..8].try_into().map_err(|_| "RNG digest shape")?,
        ))
    }
}

/// Private fields and no Deserialize/raw constructor keep the decision and likelihood together.
/// ```compile_fail
/// use tzolkin_ai::public_stochastic::SamplingTrace;
/// let _: SamplingTrace = serde_json::from_str("{}").unwrap();
/// ```
pub struct SamplingTrace {
    sampling_version: &'static str,
    identity: SamplingStreamIdentity,
    policy_binding_key: String,
    sample_index: usize,
    actor: usize,
    legal_count: usize,
    chosen_index: usize,
    state_before: u64,
    state_after: u64,
    draws_before: u64,
    draws_after: u64,
    raw_u64: u64,
    u53: u64,
    mass_ticks: u64,
    nominal_probability: f64,
    behavior_logp: f64,
    chosen_logit: f32,
    ordered_legal_digest: String,
    logits_digest: String,
    distribution_digest: String,
    reserved_rows_before: usize,
    reserved_rows_after: usize,
}
impl SamplingTrace {
    pub fn identity(&self) -> &SamplingStreamIdentity {
        &self.identity
    }
    pub fn policy_binding_key(&self) -> &str {
        &self.policy_binding_key
    }
    /// Successful samples in this session, not the native global callback index.
    pub fn sample_index(&self) -> usize {
        self.sample_index
    }
    pub fn actor(&self) -> usize {
        self.actor
    }
    pub fn legal_count(&self) -> usize {
        self.legal_count
    }
    pub fn chosen_index(&self) -> usize {
        self.chosen_index
    }
    pub fn state_before(&self) -> u64 {
        self.state_before
    }
    pub fn state_after(&self) -> u64 {
        self.state_after
    }
    pub fn draws_before(&self) -> u64 {
        self.draws_before
    }
    pub fn draws_after(&self) -> u64 {
        self.draws_after
    }
    pub fn raw_u64(&self) -> u64 {
        self.raw_u64
    }
    pub fn u53(&self) -> u64 {
        self.u53
    }
    pub fn mass_ticks(&self) -> u64 {
        self.mass_ticks
    }
    pub fn denominator_bits(&self) -> u32 {
        DENOMINATOR_BITS
    }
    pub fn nominal_probability(&self) -> f64 {
        self.nominal_probability
    }
    pub fn behavior_probability(&self) -> f64 {
        self.mass_ticks as f64 / TICKS as f64
    }
    pub fn behavior_logp(&self) -> f64 {
        self.behavior_logp
    }
    pub fn chosen_logit(&self) -> f32 {
        self.chosen_logit
    }
    pub fn ordered_legal_digest(&self) -> &str {
        &self.ordered_legal_digest
    }
    pub fn logits_digest(&self) -> &str {
        &self.logits_digest
    }
    pub fn distribution_digest(&self) -> &str {
        &self.distribution_digest
    }
    pub fn reserved_rows_before(&self) -> usize {
        self.reserved_rows_before
    }
    pub fn reserved_rows_after(&self) -> usize {
        self.reserved_rows_after
    }
    pub fn backend(&self) -> &'static str {
        "scalar"
    }
    pub fn sampling_version(&self) -> &'static str {
        self.sampling_version
    }
    pub fn rng_version(&self) -> &'static str {
        RNG_VERSION
    }
}

/// This is a sampled attempt, not proof that a core action was applied.
/// ```compile_fail
/// use tzolkin_ai::public_stochastic::SampledDecision;
/// let _: SampledDecision = serde_json::from_str("{}").unwrap();
/// ```
pub struct SampledDecision {
    decision: Decision,
    legal_action: LegalAction,
    trace: SamplingTrace,
}
impl SampledDecision {
    pub fn decision(&self) -> &Decision {
        &self.decision
    }
    pub fn legal_action(&self) -> &LegalAction {
        &self.legal_action
    }
    pub fn trace(&self) -> &SamplingTrace {
        &self.trace
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SamplingCounters {
    accepted_samples: usize,
    reserved_candidate_rows: usize,
    actor_draws: [u64; 4],
    stopped: bool,
}
impl SamplingCounters {
    pub fn accepted_samples(&self) -> usize {
        self.accepted_samples
    }
    /// NN admission rows, not a runtime forward counter. Owner qualification is separate.
    pub fn reserved_candidate_rows(&self) -> usize {
        self.reserved_candidate_rows
    }
    pub fn actor_draws(&self) -> &[u64; 4] {
        &self.actor_draws
    }
    pub fn stopped(&self) -> bool {
        self.stopped
    }
}

#[derive(Clone, Copy)]
struct ActorStream {
    state: u64,
    draws: u64,
}
fn next_splitmix(state: u64) -> (u64, u64) {
    let after = state.wrapping_add(0x9e3779b97f4a7c15);
    let mut z = after;
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
    (after, z ^ (z >> 31))
}
#[derive(Clone, Copy)]
enum SamplingPolicyKind {
    Bc,
    Rl,
}
impl SamplingPolicyKind {
    fn sampling_version(self) -> &'static str {
        match self {
            Self::Bc => SAMPLING_VERSION,
            Self::Rl => RL_SAMPLING_VERSION,
        }
    }
}
// Only sealed BC/RL wrappers can produce these inputs outside private tests.
// No arbitrary-logit prediction provider is exposed.
enum SamplingPrediction {
    Bc(PublicPolicyDistribution),
    Rl(PublicRlPolicyDistribution),
}
impl SamplingPrediction {
    fn compatible(&self, kind: SamplingPolicyKind) -> bool {
        match (self, kind) {
            (Self::Bc(p), SamplingPolicyKind::Bc) => {
                p.policy_version == POLICY_VERSION
                    && p.feature_schema == PUBLIC_FEATURE_SCHEMA
                    && p.value_validity == ValueValidity::UnavailablePolicyOnly
            }
            (Self::Rl(p), SamplingPolicyKind::Rl) => {
                p.policy_version() == crate::public_rl_native::POLICY_VERSION
                    && p.feature_schema() == PUBLIC_FEATURE_SCHEMA
                    && p.value_validity() == ValueValidity::UnavailablePolicyOnly
                    && p.update_count() == 1
                    && p.backend() == "scalar"
            }
            _ => false,
        }
    }
    fn logits(&self) -> &[f32] {
        match self {
            Self::Bc(p) => &p.logits,
            Self::Rl(p) => p.logits(),
        }
    }
    fn probability_count(&self) -> usize {
        match self {
            Self::Bc(p) => p.probabilities.len(),
            Self::Rl(p) => p.probabilities().len(),
        }
    }
}

struct SessionState {
    identity: SamplingStreamIdentity,
    streams: [ActorStream; 4],
    accepted: usize,
    reserved: usize,
    stopped: bool,
}
impl SessionState {
    fn new(identity: SamplingStreamIdentity) -> Result<Self, String> {
        let mut streams = [ActorStream { state: 0, draws: 0 }; 4];
        for (actor, stream) in streams.iter_mut().enumerate().take(identity.players) {
            stream.state = identity.initial_actor_state(actor)?;
        }
        Ok(Self {
            identity,
            streams,
            accepted: 0,
            reserved: 0,
            stopped: false,
        })
    }
    fn counters(&self) -> SamplingCounters {
        SamplingCounters {
            accepted_samples: self.accepted,
            reserved_candidate_rows: self.reserved,
            actor_draws: self.streams.map(|s| s.draws),
            stopped: self.stopped,
        }
    }
    fn reserve(&mut self, actor: usize, players: usize, count: usize) -> Result<usize, String> {
        if self.stopped {
            return Err("Sampling session is stopped; no retry or reset".into());
        }
        let next = self.reserved.checked_add(count);
        if players != self.identity.players
            || actor >= players
            || count == 0
            || count > MAX_CANDIDATES
            || self.accepted >= MAX_SAMPLES
            || next.is_none_or(|n| n > MAX_RESERVED_CANDIDATE_ROWS)
        {
            self.stopped = true;
            return Err("Invalid sampling actor/mask or episode work limit".into());
        }
        let before = self.reserved;
        self.reserved = next.ok_or("Candidate admission overflow")?;
        Ok(before)
    }
    fn choose(&mut self, actor: usize, distribution: &TickDistribution) -> Result<Draw, String> {
        if self.stopped || actor >= self.identity.players || self.accepted >= MAX_SAMPLES {
            return Err("Invalid or stopped sampling session".into());
        }
        let stream = self.streams[actor];
        let (after, raw) = next_splitmix(stream.state);
        let u53 = raw >> 11;
        let index = distribution.chosen(u53)?;
        let mass = distribution.mass(index)?;
        let probability = mass as f64 / TICKS as f64;
        let logp = if mass == TICKS { 0.0 } else { probability.ln() };
        if !logp.is_finite() || logp > 0.0 {
            return Err("Non-finite sampling logp".into());
        }
        let draws_after = stream.draws.checked_add(1).ok_or("Actor draw overflow")?;
        // No operation which can fail follows these state mutations.
        self.streams[actor] = ActorStream {
            state: after,
            draws: draws_after,
        };
        self.accepted += 1;
        Ok(Draw {
            index,
            raw,
            u53,
            mass,
            logp,
            state_before: stream.state,
            state_after: after,
            draws_before: stream.draws,
            draws_after,
        })
    }
    fn sample_with(
        &mut self,
        policy_binding_key: &str,
        observation: &Observation,
        predict: impl FnOnce(&Observation) -> Result<PublicPolicyDistribution, String>,
    ) -> Result<SampledDecision, String> {
        self.sample_with_kind(
            policy_binding_key,
            SamplingPolicyKind::Bc,
            observation,
            |o| predict(o).map(SamplingPrediction::Bc),
        )
    }
    fn sample_with_kind(
        &mut self,
        policy_binding_key: &str,
        kind: SamplingPolicyKind,
        observation: &Observation,
        predict: impl FnOnce(&Observation) -> Result<SamplingPrediction, String>,
    ) -> Result<SampledDecision, String> {
        let result = self.sample_inner(policy_binding_key, kind, observation, predict);
        if result.is_err() {
            self.stopped = true;
        }
        result
    }
    fn sample_inner(
        &mut self,
        policy_binding_key: &str,
        kind: SamplingPolicyKind,
        observation: &Observation,
        predict: impl FnOnce(&Observation) -> Result<SamplingPrediction, String>,
    ) -> Result<SampledDecision, String> {
        let before_rows = self.reserve(
            observation.actor,
            observation.players.len(),
            observation.legal_actions.len(),
        )?;
        let prediction = predict(observation)?;
        if !prediction.compatible(kind)
            || prediction.logits().len() != observation.legal_actions.len()
            || prediction.probability_count() != prediction.logits().len()
        {
            return Err("Unsupported stochastic policy distribution contract".into());
        }
        let ticks = TickDistribution::new(prediction.logits())?;
        let ordered_legal_digest = legal_digest(&observation.legal_actions)?;
        let logits_digest = full_logits_digest(prediction.logits());
        let distribution_digest = ticks.digest();
        let sample_index = self.accepted;
        let draw = self.choose(observation.actor, &ticks)?;
        let legal = observation.legal_actions[draw.index].clone();
        let chosen_logit = prediction.logits()[draw.index];
        let decision = Decision {
            actor: observation.actor,
            observation_key: observation.observation_key.clone(),
            policy_version: kind.sampling_version().into(),
            r#move: legal.r#move.clone(),
            score: f64::from(chosen_logit),
        };
        let trace = SamplingTrace {
            sampling_version: kind.sampling_version(),
            identity: self.identity,
            policy_binding_key: policy_binding_key.into(),
            sample_index,
            actor: observation.actor,
            legal_count: prediction.logits().len(),
            chosen_index: draw.index,
            state_before: draw.state_before,
            state_after: draw.state_after,
            draws_before: draw.draws_before,
            draws_after: draw.draws_after,
            raw_u64: draw.raw,
            u53: draw.u53,
            mass_ticks: draw.mass,
            nominal_probability: ticks.nominal[draw.index],
            behavior_logp: draw.logp,
            chosen_logit,
            ordered_legal_digest,
            logits_digest,
            distribution_digest,
            reserved_rows_before: before_rows,
            reserved_rows_after: self.reserved,
        };
        Ok(SampledDecision {
            decision,
            legal_action: legal,
            trace,
        })
    }
}
struct Draw {
    index: usize,
    raw: u64,
    u53: u64,
    mass: u64,
    logp: f64,
    state_before: u64,
    state_after: u64,
    draws_before: u64,
    draws_after: u64,
}
pub(crate) fn full_logits_digest(logits: &[f32]) -> String {
    let mut hash = Sha256::new();
    hash.update(b"tzolkin-stochastic-full-f32-logits-v1\0");
    for z in logits {
        hash.update(z.to_bits().to_le_bytes());
    }
    format!("{:x}", hash.finalize())
}

// The original max/exp/ordered-sum body is shared without changing arithmetic,
// finite checks or error strings. exp/ln bits remain platform-bound.
pub(crate) fn nominal_weights(logits: &[f32]) -> Result<(Vec<f64>, f64), String> {
    if logits.is_empty() || logits.len() > MAX_CANDIDATES || logits.iter().any(|z| !z.is_finite()) {
        return Err("Invalid finite full-mask sampling logits".into());
    }
    let maximum = logits
        .iter()
        .map(|z| f64::from(*z))
        .fold(f64::NEG_INFINITY, f64::max);
    let weights: Vec<f64> = logits
        .iter()
        .map(|z| (f64::from(*z) - maximum).exp())
        .collect();
    let sum = weights.iter().fold(0.0_f64, |a, b| a + b);
    if !sum.is_finite() || sum <= 0.0 {
        return Err("Invalid f64 softmax denominator".into());
    }
    Ok((weights, sum))
}

struct TickDistribution {
    nominal: Vec<f64>,
    boundaries: Vec<u64>,
}
impl TickDistribution {
    fn new(logits: &[f32]) -> Result<Self, String> {
        let (weights, sum) = nominal_weights(logits)?;
        let floor = UNIFORM_MIXTURE / logits.len() as f64;
        let mut nominal = Vec::with_capacity(logits.len());
        let mut boundaries = Vec::with_capacity(logits.len());
        let mut cumulative = 0.0_f64;
        let mut previous = 0_u64;
        for (i, weight) in weights.iter().enumerate() {
            let p = (1.0 - UNIFORM_MIXTURE) * (weight / sum) + floor;
            cumulative += p;
            if !p.is_finite() || p <= 0.0 || !cumulative.is_finite() {
                return Err("Invalid nominal sampling probability/CDF".into());
            }
            let boundary = if i + 1 == logits.len() {
                TICKS
            } else {
                let scaled = cumulative * TICKS as f64;
                if !scaled.is_finite() || scaled <= 0.0 || scaled >= TICKS as f64 {
                    return Err("Sampling CDF outside the tick grid".into());
                }
                let ceil = scaled.ceil();
                if ceil >= TICKS as f64 {
                    return Err("Premature final CDF boundary".into());
                }
                ceil as u64
            };
            if boundary <= previous || boundary > TICKS {
                return Err("Nonpositive sampling interval".into());
            }
            nominal.push(p);
            boundaries.push(boundary);
            previous = boundary;
        }
        if previous != TICKS {
            return Err("Sampling mass does not sum to 2^53".into());
        }
        Ok(Self {
            nominal,
            boundaries,
        })
    }
    fn chosen(&self, u53: u64) -> Result<usize, String> {
        if u53 >= TICKS {
            return Err("Sampling draw outside the 53-bit grid".into());
        }
        self.boundaries
            .iter()
            .position(|b| u53 < *b)
            .ok_or_else(|| "Missing final sampling boundary".into())
    }
    fn mass(&self, index: usize) -> Result<u64, String> {
        let bound = *self.boundaries.get(index).ok_or("Invalid sampled index")?;
        let previous = if index == 0 {
            0
        } else {
            self.boundaries[index - 1]
        };
        bound
            .checked_sub(previous)
            .filter(|m| *m > 0)
            .ok_or_else(|| "Nonpositive sampling mass".into())
    }
    fn digest(&self) -> String {
        let mut hash = Sha256::new();
        hash.update(b"tzolkin-stochastic-tick-distribution-v1\0");
        hash.update((self.boundaries.len() as u64).to_le_bytes());
        for (p, b) in self.nominal.iter().zip(&self.boundaries) {
            hash.update(p.to_bits().to_le_bytes());
            hash.update(b.to_le_bytes());
        }
        format!("{:x}", hash.finalize())
    }
}
pub(crate) fn behavior_distribution(
    logits: &[f32],
) -> Result<(Vec<f64>, Vec<u64>, String), String> {
    let distribution = TickDistribution::new(logits)?;
    let masses = (0..logits.len())
        .map(|index| distribution.mass(index))
        .collect::<Result<_, _>>()?;
    let digest = distribution.digest();
    Ok((distribution.nominal, masses, digest))
}
struct BoundedHash {
    hash: Sha256,
    bytes: usize,
}
impl Write for BoundedHash {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        self.bytes = self
            .bytes
            .checked_add(bytes.len())
            .filter(|n| *n <= MAX_OBSERVATION_BYTES)
            .ok_or_else(|| io::Error::other("Sampling legal mask exceeds 16 MiB"))?;
        self.hash.update(bytes);
        Ok(bytes.len())
    }
    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}
fn legal_digest(legal: &[LegalAction]) -> Result<String, String> {
    let mut out = BoundedHash {
        hash: Sha256::new(),
        bytes: 0,
    };
    serde_json::to_writer(&mut out, legal).map_err(|e| e.to_string())?;
    Ok(format!("{:x}", out.hash.finalize()))
}

/// One fresh episode, all absolute actor streams, no reset or hidden world parameter.
/// ```compile_fail
/// use tzolkin_ai::public_stochastic::StochasticSession;
/// fn cannot_pass_hidden_world(session: &mut StochasticSession<'_, '_>, game: &tzolkin_core::GameState) {
///     session.sample(game).unwrap();
/// }
/// ```
pub struct StochasticSession<'handle, 'model> {
    policy: &'handle PublicPolicyHandle<'model>,
    policy_binding_key: String,
    state: SessionState,
}
impl<'handle, 'model> StochasticSession<'handle, 'model> {
    pub fn new(
        policy: &'handle PublicPolicyHandle<'model>,
        identity: SamplingStreamIdentity,
    ) -> Result<Self, String> {
        if policy.backend() != "scalar"
            || !matches!(
                policy.provenance(),
                SeatPolicy::PublicLearned {
                    feature_schema: PUBLIC_FEATURE_SCHEMA,
                    ..
                }
            )
        {
            return Err(
                "Stochastic sampling requires an immutable Scalar native V2 policy handle".into(),
            );
        }
        let binding = serde_json::to_vec(policy.provenance()).map_err(|e| e.to_string())?;
        Ok(Self {
            policy,
            policy_binding_key: digest(&binding),
            state: SessionState::new(identity)?,
        })
    }
    pub fn counters(&self) -> SamplingCounters {
        self.state.counters()
    }
    /// Collector must stop after apply/IO failure. It must preserve the already sampled attempt.
    pub fn stop(&mut self) {
        self.state.stopped = true;
    }
    pub fn sample(&mut self, observation: &Observation) -> Result<SampledDecision, String> {
        self.state
            .sample_with(&self.policy_binding_key, observation, |o| {
                self.policy.distribution(o)
            })
    }
}

/// Logical identity only: no path, BC resealing or raw artifact ownership.
/// Strict record decoding uses this DTO internally; it cannot construct a handle.
#[derive(Clone, Debug, serde::Deserialize, serde::Serialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub(crate) struct RlSamplingPolicy {
    policy_version: String,
    artifact_checksum: String,
    update_count: u64,
    task: String,
    model_version: String,
    input_contract: String,
    feature_schema: u32,
    backend: String,
    numerical_target: String,
}
impl RlSamplingPolicy {
    pub(crate) fn from_handle(policy: &UpdatedPublicRlHandle<'_>) -> Result<Self, String> {
        if policy.backend() != "scalar" || policy.update_count() != 1 {
            return Err("RL sampling requires a sealed Scalar count1 handle".into());
        }
        Ok(Self {
            policy_version: crate::public_rl_native::POLICY_VERSION.into(),
            artifact_checksum: policy.artifact_checksum().into(),
            update_count: policy.update_count(),
            task: "policyOnlyRlOneStep".into(),
            model_version: crate::public_model::MODEL_VERSION.into(),
            input_contract: crate::public_model::INPUT_CONTRACT.into(),
            feature_schema: PUBLIC_FEATURE_SCHEMA,
            backend: policy.backend().into(),
            numerical_target: crate::public_stochastic_native::numerical_target(),
        })
    }
    fn binding_key(&self) -> Result<String, String> {
        let mut hash = Sha256::new();
        hash.update(b"tzolkin-public-stochastic-rl-count1-policy-v1\0");
        hash.update(serde_json::to_vec(self).map_err(|e| e.to_string())?);
        Ok(format!("{:x}", hash.finalize()))
    }
}

/// Fresh actor-local streams borrowing only an immutable, sealed RL handle.
/// CDF/RNG/caps match BC; identity and Decision version remain distinctly RL.
pub struct RlStochasticSession<'handle, 'model> {
    policy: &'handle UpdatedPublicRlHandle<'model>,
    policy_binding_key: String,
    state: SessionState,
}
impl<'handle, 'model> RlStochasticSession<'handle, 'model> {
    pub fn new(
        policy: &'handle UpdatedPublicRlHandle<'model>,
        identity: SamplingStreamIdentity,
    ) -> Result<Self, String> {
        Ok(Self {
            policy,
            policy_binding_key: RlSamplingPolicy::from_handle(policy)?.binding_key()?,
            state: SessionState::new(identity)?,
        })
    }
    pub fn counters(&self) -> SamplingCounters {
        self.state.counters()
    }
    pub fn stop(&mut self) {
        self.state.stopped = true;
    }
    pub fn sample(&mut self, observation: &Observation) -> Result<SampledDecision, String> {
        self.state.sample_with_kind(
            &self.policy_binding_key,
            SamplingPolicyKind::Rl,
            observation,
            |o| self.policy.distribution(o).map(SamplingPrediction::Rl),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::public_model::{LoadedPublicPolicy, PublicPolicyArtifact};
    use crate::public_native::integration_fixture;
    use crate::public_policy_likelihood::BehaviorLikelihood;
    use tzolkin_core::compact::catalog::WEALTH_IDS;
    use tzolkin_core::observation::{observation_key, observe};
    use tzolkin_core::tribes::TribeId;
    use tzolkin_core::{
        GameState, GearId, GearWorker, Phase, TurnMode, apply_move, create_game, get_choices,
    };
    fn identity(players: usize) -> SamplingStreamIdentity {
        SamplingStreamIdentity::new(SamplingSeed::new(17), 0, 0, players).unwrap()
    }
    fn setup(players: usize) -> GameState {
        create_game((0..players).map(|p| format!("P{p}")).collect(), 17, false).unwrap()
    }
    fn started(players: usize) -> GameState {
        let mut state = setup(players);
        for _ in 0..64 {
            if state.phase != Phase::Setup {
                return state;
            }
            let choice = get_choices(&state)
                .into_iter()
                .find(|c| c.disabled != Some(true))
                .unwrap();
            state = apply_move(&state, choice.r#move).unwrap();
        }
        panic!("Setup fixture must finish within 64 choices")
    }
    fn rekey(o: &mut Observation) {
        o.observation_key = observation_key(o).unwrap();
    }
    #[test]
    fn tick_boundaries_equal_the_strict_float_grid_selection() {
        let d = TickDistribution::new(&[0.0, 0.0]).unwrap();
        assert_eq!(d.boundaries, [TICKS / 2, TICKS]);
        assert_eq!(d.chosen(0).unwrap(), 0);
        assert_eq!(d.chosen(TICKS / 2 - 1).unwrap(), 0);
        assert_eq!(d.chosen(TICKS / 2).unwrap(), 1);
        assert_eq!(d.chosen(TICKS - 1).unwrap(), 1);
        assert!(d.chosen(TICKS).is_err());
        assert_eq!(d.mass(0).unwrap(), TICKS / 2);
        assert_eq!(d.mass(1).unwrap(), TICKS / 2);
    }
    #[test]
    fn singleton_uses_one_draw_with_positive_zero_logp() {
        let mut state = SessionState::new(identity(3)).unwrap();
        let d = TickDistribution::new(&[-0.0]).unwrap();
        state.reserve(2, 3, 1).unwrap();
        let draw = state.choose(2, &d).unwrap();
        assert_eq!(draw.index, 0);
        assert_eq!(draw.mass, TICKS);
        assert_eq!(draw.logp.to_bits(), 0.0_f64.to_bits());
        assert_eq!(state.counters().actor_draws(), &[0, 0, 1, 0]);
        assert_eq!(state.counters().accepted_samples(), 1);
    }
    #[test]
    fn full_support_survives_extreme_finite_logits_and_4096_rows() {
        let mut logits = vec![f32::MIN; MAX_CANDIDATES];
        logits[2048] = f32::MAX;
        let d = TickDistribution::new(&logits).unwrap();
        let mut total = 0_u64;
        for i in 0..MAX_CANDIDATES {
            let mass = d.mass(i).unwrap();
            total += mass;
            assert!(mass > 0);
            assert!(d.nominal[i] >= 1.0 / (1_u64 << 32) as f64);
            let before = if i == 0 { 0 } else { d.boundaries[i - 1] };
            assert_eq!(d.chosen(before).unwrap(), i);
            assert_eq!(d.chosen(d.boundaries[i] - 1).unwrap(), i);
        }
        assert_eq!(total, TICKS);
    }
    #[test]
    fn malformed_logits_are_errors_before_any_rng_draw() {
        for logits in [
            vec![],
            vec![f32::NAN],
            vec![f32::INFINITY],
            vec![f32::NEG_INFINITY],
            vec![0.0; MAX_CANDIDATES + 1],
        ] {
            let state = SessionState::new(identity(4)).unwrap();
            assert!(TickDistribution::new(&logits).is_err());
            assert_eq!(state.counters().actor_draws(), &[0; 4]);
        }
    }
    #[test]
    fn signed_zero_logit_bits_remain_distinct_but_probabilities_equal() {
        // Independent SHA-256 vector of the domain and ordered little-endian f32 bits.
        assert_eq!(
            full_logits_digest(&[0.0, -0.0, 1.25, -2.5]),
            "93860294c2dbf980bb96284298ec0f875ba695e528182fcecf9c926aa5e3488a"
        );
        let plus = TickDistribution::new(&[0.0, 0.0]).unwrap();
        let minus = TickDistribution::new(&[-0.0, 0.0]).unwrap();
        assert_eq!(plus.boundaries, minus.boundaries);
        assert_eq!(plus.nominal, minus.nominal);
        assert_ne!(0.0_f32.to_bits(), (-0.0_f32).to_bits());
        let s = setup(3);
        let o = observe(&s, s.current_player).unwrap();
        let model = PublicPolicyArtifact::new(11235).unwrap();
        let mut p = LoadedPublicPolicy::new(&model)
            .unwrap()
            .distribution(&o)
            .unwrap();
        p.logits.fill(0.0);
        let mut q = p.clone();
        q.logits[0] = -0.0;
        let mut a = SessionState::new(identity(3)).unwrap();
        let mut b = SessionState::new(identity(3)).unwrap();
        let x = a.sample_with("synthetic", &o, |_| Ok(p)).unwrap();
        let y = b.sample_with("synthetic", &o, |_| Ok(q)).unwrap();
        assert_ne!(x.trace().logits_digest(), y.trace().logits_digest());
        assert_eq!(
            x.trace().distribution_digest(),
            y.trace().distribution_digest()
        );
        assert_eq!(x.trace().raw_u64(), y.trace().raw_u64());
        assert_eq!(x.trace().mass_ticks(), y.trace().mass_ticks());
    }
    #[test]
    fn per_actor_streams_are_isolated_and_fresh_episodes_reproduce() {
        let mut a = SessionState::new(identity(4)).unwrap();
        let mut b = SessionState::new(identity(4)).unwrap();
        let d = TickDistribution::new(&[0.0, 1.0, -1.0]).unwrap();
        for actor in [3, 1, 3] {
            a.reserve(actor, 4, 3).unwrap();
            a.choose(actor, &d).unwrap();
        }
        a.reserve(0, 4, 3).unwrap();
        b.reserve(0, 4, 3).unwrap();
        let x = a.choose(0, &d).unwrap();
        let y = b.choose(0, &d).unwrap();
        assert_eq!(x.raw, y.raw);
        assert_eq!(x.index, y.index);
        assert_eq!(x.state_before, y.state_before);
        assert_eq!(a.counters().actor_draws(), &[1, 1, 0, 2]);
    }
    #[test]
    fn domain_identity_separates_actor_players_episode_replicate() {
        let original = identity(3).initial_actor_state(0).unwrap();
        for other in [
            identity(3).initial_actor_state(1).unwrap(),
            identity(4).initial_actor_state(0).unwrap(),
            SamplingStreamIdentity::new(SamplingSeed::new(17), 1, 0, 3)
                .unwrap()
                .initial_actor_state(0)
                .unwrap(),
            SamplingStreamIdentity::new(SamplingSeed::new(17), 0, 1, 3)
                .unwrap()
                .initial_actor_state(0)
                .unwrap(),
        ] {
            assert_ne!(original, other);
        }
        assert!(identity(3).initial_actor_state(3).is_err());
    }
    #[test]
    fn splitmix_vector_and_wrapping_state_are_fixed() {
        let (next, raw) = next_splitmix(0);
        assert_eq!(next, 0x9e3779b97f4a7c15);
        assert_eq!(raw, 0xe220a8397b1dcdaf);
        let (after, _) = next_splitmix(u64::MAX);
        assert_eq!(after, 0x9e3779b97f4a7c14);
    }
    #[test]
    fn work_limit_failure_consumes_no_draw_and_poisoned_session_cannot_retry() {
        let mut s = SessionState::new(identity(3)).unwrap();
        s.reserved = MAX_RESERVED_CANDIDATE_ROWS;
        assert!(s.reserve(0, 3, 1).is_err());
        assert_eq!(s.counters().actor_draws(), &[0; 4]);
        assert!(s.stopped);
        assert!(s.reserve(0, 3, 1).is_err());
        let mut s = SessionState::new(identity(3)).unwrap();
        s.accepted = MAX_SAMPLES;
        assert!(s.reserve(0, 3, 1).is_err());
        assert_eq!(s.counters().actor_draws(), &[0; 4]);
        let mut s = SessionState::new(identity(3)).unwrap();
        assert!(s.reserve(3, 3, 1).is_err());
        assert!(s.stopped);
    }
    #[test]
    fn immutable_handle_samples_setup_playing_and_keeps_trace_consistent() {
        // G2's private fixture is synthetic, not a qualified training artifact.
        let model = PublicPolicyArtifact::new(11235).unwrap();
        let handle = integration_fixture::handle(&model);
        for players in [3, 4] {
            for state in [setup(players), started(players)] {
                let o = observe(&state, state.current_player).unwrap();
                let logits = handle.distribution(&o).unwrap().logits;
                let d = TickDistribution::new(&logits).unwrap();
                let mut a = StochasticSession::new(&handle, identity(players)).unwrap();
                let mut b = StochasticSession::new(&handle, identity(players)).unwrap();
                let result = a.sample(&o).unwrap();
                let repeated = b.sample(&o).unwrap();
                let t = result.trace();
                let likelihood =
                    BehaviorLikelihood::from_logits(&logits, t.chosen_index()).unwrap();
                assert_eq!(likelihood.mass_ticks()[t.chosen_index()], t.mass_ticks());
                assert_eq!(likelihood.distribution_digest(), t.distribution_digest());
                assert_eq!(
                    likelihood.smooth().probabilities()[t.chosen_index()].to_bits(),
                    t.nominal_probability().to_bits()
                );
                assert_eq!(
                    likelihood.behavior_log_probability().to_bits(),
                    t.behavior_logp().to_bits()
                );
                assert_eq!(result.decision(), repeated.decision());
                assert_eq!(t.raw_u64(), repeated.trace().raw_u64());
                assert_eq!(result.legal_action(), &o.legal_actions[t.chosen_index()]);
                assert_eq!(t.mass_ticks(), d.mass(t.chosen_index()).unwrap());
                assert_eq!(
                    t.behavior_logp().to_bits(),
                    t.behavior_probability().ln().to_bits()
                );
                assert_eq!(result.decision().score, f64::from(logits[t.chosen_index()]));
                assert_eq!(result.decision().policy_version, SAMPLING_VERSION);
                assert_eq!(result.decision().observation_key, o.observation_key);
                assert_eq!(
                    t.policy_binding_key(),
                    digest(&serde_json::to_vec(handle.provenance()).unwrap())
                );
                assert_eq!(t.draws_before(), 0);
                assert_eq!(t.draws_after(), 1);
                assert_eq!(t.sample_index(), 0);
                assert_eq!(t.reserved_rows_before(), 0);
                assert_eq!(t.reserved_rows_after(), o.legal_actions.len());
                assert_eq!(a.counters().accepted_samples(), 1);
                a.stop();
                assert!(a.sample(&o).is_err());
                assert_eq!(a.counters().accepted_samples(), 1);
            }
        }
    }
    #[test]
    fn playing_private_changes_preserve_distribution_draw_and_selection() {
        let model = PublicPolicyArtifact::new(11235).unwrap();
        let handle = integration_fixture::handle(&model);
        for players in [3, 4] {
            let state = started(players);
            let o = observe(&state, state.current_player).unwrap();
            let mut a = StochasticSession::new(&handle, identity(players)).unwrap();
            let original = a.sample(&o).unwrap();
            let mut variants = vec![o.clone(); 4];
            variants[0].private.wealth_offer = vec!["w01".into()];
            variants[1].private.selected_wealth = vec!["w02".into()];
            variants[2].private.tribe_offer = vec![TribeId::Bacab];
            variants[3].private.selected_tribe = Some(TribeId::CitBolonTum);
            for mut changed in variants {
                rekey(&mut changed);
                let mut b = StochasticSession::new(&handle, identity(players)).unwrap();
                let result = b.sample(&changed).unwrap();
                assert_ne!(
                    result.decision().observation_key,
                    original.decision().observation_key
                );
                assert_eq!(result.decision().r#move, original.decision().r#move);
                assert_eq!(
                    result.trace().logits_digest(),
                    original.trace().logits_digest()
                );
                assert_eq!(
                    result.trace().distribution_digest(),
                    original.trace().distribution_digest()
                );
                assert_eq!(result.trace().raw_u64(), original.trace().raw_u64());
                assert_eq!(result.trace().mass_ticks(), original.trace().mass_ticks());
                assert_eq!(
                    result.trace().behavior_logp().to_bits(),
                    original.trace().behavior_logp().to_bits()
                );
            }
        }
    }
    #[test]
    fn legitimate_setup_offer_changes_reach_logits_and_mask_unknown_offers_fail() {
        let model = PublicPolicyArtifact::new(11235).unwrap();
        let handle = integration_fixture::handle(&model);
        for players in [3, 4] {
            let before = setup(players);
            let mut after = before.clone();
            let unused = WEALTH_IDS
                .iter()
                .find(|id| {
                    !before
                        .players
                        .iter()
                        .any(|p| p.wealth_offer.iter().any(|offer| offer == *id))
                })
                .unwrap();
            after.players[after.current_player].wealth_offer[3] = (*unused).into();
            assert!(tzolkin_core::validation::validate_game_state(
                &serde_json::to_value(&after).unwrap()
            ));
            let o = observe(&before, before.current_player).unwrap();
            let changed = observe(&after, after.current_player).unwrap();
            let mut a = StochasticSession::new(&handle, identity(players)).unwrap();
            let mut b = StochasticSession::new(&handle, identity(players)).unwrap();
            let x = a.sample(&o).unwrap();
            let y = b.sample(&changed).unwrap();
            assert_ne!(x.trace().logits_digest(), y.trace().logits_digest());
            assert_ne!(
                x.trace().ordered_legal_digest(),
                y.trace().ordered_legal_digest()
            );
            let mut missing = o;
            missing.private.wealth_offer.clear();
            rekey(&mut missing);
            let mut c = StochasticSession::new(&handle, identity(players)).unwrap();
            assert!(c.sample(&missing).is_err());
            assert_eq!(c.counters().actor_draws(), &[0; 4]);
            assert!(c.counters().stopped());
        }
    }
    #[test]
    fn scope_key_and_affine_overflow_errors_stop_without_draw_or_fallback() {
        let good_model = PublicPolicyArtifact::new(11235).unwrap();
        let bad_model = integration_fixture::model(true);
        let good = integration_fixture::handle(&good_model);
        let bad = integration_fixture::handle(&bad_model);
        let s = setup(3);
        let o = observe(&s, s.current_player).unwrap();
        let mut variants = vec![o.clone(); 4];
        variants[0].observation_key = "0".repeat(64);
        variants[1].additional_buildings = true;
        rekey(&mut variants[1]);
        variants[2].phase = Phase::Finished;
        rekey(&mut variants[2]);
        variants[3].legal_actions.clear();
        rekey(&mut variants[3]);
        for changed in variants {
            let mut session = StochasticSession::new(&good, identity(3)).unwrap();
            assert!(session.sample(&changed).is_err());
            assert_eq!(session.counters().actor_draws(), &[0; 4]);
            assert_eq!(session.counters().accepted_samples(), 0);
            assert!(session.sample(&o).is_err());
        }
        let mut session = StochasticSession::new(&bad, identity(3)).unwrap();
        assert!(
            session
                .sample(&o)
                .err()
                .unwrap()
                .contains("hidden activation")
        );
        assert_eq!(session.counters().actor_draws(), &[0; 4]);
        assert!(session.counters().stopped());
    }
    #[test]
    fn rotation_pending_uses_absolute_actor_and_only_that_stream() {
        let model = PublicPolicyArtifact::new(11235).unwrap();
        let handle = integration_fixture::handle(&model);
        for players in [3, 4] {
            let mut s = started(players);
            s.turn_index = players - 1;
            s.current_player = s.turn_order[s.turn_index];
            s.first_player_claimed = Some(s.turn_order[0]);
            s.turn.mode = TurnMode::Place;
            s.turn.count = 1;
            let free = s.gears[&GearId::Palenque]
                .iter()
                .position(Option::is_none)
                .unwrap();
            s.gears.get_mut(&GearId::Palenque).unwrap()[free] = Some(GearWorker {
                player_id: s.current_player as i64,
                dummy: false,
            });
            let pending = apply_move(
                &s,
                tzolkin_core::GameMove::EndTurn {
                    double_advance: None,
                },
            )
            .unwrap();
            let o = observe(&pending, pending.current_player).unwrap();
            assert_eq!(o.pending_task, Some(tzolkin_core::Task::Rotation));
            assert_ne!(o.actor, o.turn_player);
            let mut session = StochasticSession::new(&handle, identity(players)).unwrap();
            let x = session.sample(&o).unwrap();
            let y = session.sample(&o).unwrap();
            assert_eq!(x.trace().actor(), o.actor);
            assert_eq!(x.trace().sample_index(), 0);
            assert_eq!(y.trace().sample_index(), 1);
            assert_eq!(y.trace().draws_before(), 1);
            assert_eq!(session.counters().actor_draws()[o.actor], 2);
            assert_eq!(session.counters().actor_draws()[o.turn_player], 0);
        }
    }
    #[test]
    fn admission_limits_and_bad_distribution_fail_before_draw() {
        let s = setup(3);
        let o = observe(&s, s.current_player).unwrap();
        let mut state = SessionState::new(identity(3)).unwrap();
        state.reserved = MAX_RESERVED_CANDIDATE_ROWS;
        assert!(
            state
                .sample_with("synthetic", &o, |_| panic!(
                    "Prediction must not run beyond admission cap"
                ))
                .is_err()
        );
        assert_eq!(state.counters().actor_draws(), &[0; 4]);
        let model = PublicPolicyArtifact::new(11235).unwrap();
        let loaded = LoadedPublicPolicy::new(&model).unwrap();
        let prediction = loaded.distribution(&o).unwrap();
        let mut variants = vec![prediction; 4];
        variants[0].policy_version = "unknown".into();
        variants[1].feature_schema = 1;
        variants[2].logits.pop();
        variants[3].logits[0] = f32::NAN;
        for p in variants {
            let mut state = SessionState::new(identity(3)).unwrap();
            assert!(state.sample_with("synthetic", &o, |_| Ok(p)).is_err());
            assert!(state.stopped);
            assert_eq!(state.counters().actor_draws(), &[0; 4]);
        }
    }
}
