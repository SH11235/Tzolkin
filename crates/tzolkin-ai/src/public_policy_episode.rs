//! Completed, applied actor trajectories with gamma-one terminal winner shares.
//! Native/sampling audit remains authoritative. This target view does not admit
//! a training partition or authenticate producers, seed origins, or episode selection.
use std::io::{self, Write};

use sha2::{Digest, Sha256};
use tzolkin_core::observation::{LegalAction, MOVE_SCHEMA, OBSERVATION_SCHEMA, Observation};
use tzolkin_core::{FinalScore, Phase};

use crate::dataset::seed_family_id;
use crate::public_stochastic::{
    DENOMINATOR_BITS, MAX_SAMPLES, RNG_VERSION, SAMPLING_VERSION, TICKS, UNIFORM_MIXTURE,
};
use crate::public_stochastic_native::{AuditedStochasticGame, NativeStochasticConfig};
use crate::public_stochastic_record::{
    Callback, Counts, FinalScoreWire, MAX_RECORD_BYTES, RECORD_SCHEMA, Record, SOURCE_KIND, hex64,
    parse_hex64,
};
use crate::replay::{RULES_BASELINE, RULES_VERSION, SeatPolicy, catalog_hash};

pub const TARGET_CONTRACT: &str = "applied-actor-gamma1-terminal-winner-share-v1";
pub const GAMMA: u32 = 1;

#[derive(Clone, Copy)]
struct ActorLink {
    actor_index: usize,
    next: Option<usize>,
    gap: usize,
    mass: u64,
    nominal: f64,
    logp: f64,
}

/// Owns one audited record; step views borrow its observations and full legal order.
/// No raw record constructor, deserialization, or model selection is available.
///
/// ```compile_fail
/// use tzolkin_ai::public_policy_episode::ValidatedPolicyEpisode;
/// let _: ValidatedPolicyEpisode = serde_json::from_str("{}").unwrap();
/// ```
pub struct ValidatedPolicyEpisode {
    source: AuditedStochasticGame,
    config: NativeStochasticConfig,
    links: Vec<ActorLink>,
    ranks: Vec<usize>,
    winners: usize,
    checksum: String,
}
impl ValidatedPolicyEpisode {
    pub fn from_audited(source: AuditedStochasticGame) -> Result<Self, String> {
        let record = source.record();
        let config = record.header.config.checked_config()?;
        let (links, ranks, winners) = plan(record, &config)?;
        let checksum = canonical_checksum(record)?;
        Ok(Self {
            source,
            config,
            links,
            ranks,
            winners,
            checksum,
        })
    }
    /// This is a checksum of the audited canonical content, not original file bytes.
    pub fn canonical_record_checksum(&self) -> &str {
        &self.checksum
    }
    pub fn target_contract(&self) -> &'static str {
        TARGET_CONTRACT
    }
    pub fn source_config(&self) -> &NativeStochasticConfig {
        &self.config
    }
    pub fn family_id(&self) -> &str {
        &self.source.record().header.native_family_id
    }
    pub fn policy_provenance(&self) -> &SeatPolicy {
        &self.source.record().header.base_policy
    }
    pub fn catalog_hash(&self) -> &str {
        &self.source.record().header.catalog_hash
    }
    pub fn numerical_target(&self) -> &str {
        &self.source.record().header.numerical_target
    }
    pub fn terminal_state_key(&self) -> &str {
        &self
            .source
            .record()
            .terminal
            .as_ref()
            .expect("validated terminal")
            .final_state
    }
    pub fn len(&self) -> usize {
        self.links.len()
    }
    pub fn is_empty(&self) -> bool {
        self.links.is_empty()
    }
    /// Family membership is metadata; update split admission belongs to a future trainer.
    pub fn training_admission_available(&self) -> bool {
        false
    }
    pub fn producer_authenticated(&self) -> bool {
        false
    }
    pub fn independent_seed_origins_verified(&self) -> bool {
        false
    }
    pub fn step(&self, index: usize) -> Option<AppliedActorStep<'_>> {
        let link = self.links.get(index)?;
        let callback = &self.source.record().callbacks[index];
        let rank = self.ranks[callback.actor];
        Some(AppliedActorStep {
            callback,
            link,
            rank,
            share: utility(rank, self.winners),
        })
    }
    pub fn steps(&self) -> impl ExactSizeIterator<Item = AppliedActorStep<'_>> {
        (0..self.len()).map(|index| self.step(index).expect("validated step"))
    }
}
fn utility(rank: usize, winners: usize) -> f64 {
    if rank == 1 { 1.0 / winners as f64 } else { 0.0 }
}

/// Applied decision input and targets, borrowing the immutable audited source.
pub struct AppliedActorStep<'a> {
    callback: &'a Callback,
    link: &'a ActorLink,
    rank: usize,
    share: f64,
}
impl AppliedActorStep<'_> {
    pub fn observation(&self) -> &Observation {
        self.callback
            .observation
            .as_ref()
            .expect("validated observation")
    }
    pub fn chosen(&self) -> &LegalAction {
        &self
            .callback
            .sample
            .as_ref()
            .expect("validated sample")
            .chosen
    }
    pub fn chosen_index(&self) -> usize {
        self.callback
            .sample
            .as_ref()
            .expect("validated sample")
            .trace
            .chosen_index
    }
    pub fn global_callback_index(&self) -> usize {
        self.callback.global_callback_index
    }
    pub fn actor(&self) -> usize {
        self.callback.actor
    }
    pub fn actor_step_index(&self) -> usize {
        self.link.actor_index
    }
    pub fn session_sample_index(&self) -> usize {
        self.callback
            .sample
            .as_ref()
            .expect("validated sample")
            .trace
            .sample_index
    }
    /// Zero-based ordinal in this actor's stream of successful sampling draws.
    pub fn actor_draw_ordinal(&self) -> usize {
        self.link.actor_index
    }
    pub fn state_before(&self) -> &str {
        &self.callback.state_before
    }
    pub fn state_after(&self) -> &str {
        self.callback
            .state_after
            .as_deref()
            .expect("validated post-state")
    }
    pub fn next_own_global_index(&self) -> Option<usize> {
        self.link.next
    }
    /// Callbacks strictly between this action and the next own action, or terminal.
    pub fn intervening_callbacks(&self) -> usize {
        self.link.gap
    }
    pub fn behavior_mass_ticks(&self) -> u64 {
        self.link.mass
    }
    pub fn behavior_probability(&self) -> f64 {
        self.link.mass as f64 / TICKS as f64
    }
    pub fn behavior_logp(&self) -> f64 {
        self.link.logp
    }
    pub fn logits_digest(&self) -> &str {
        &self
            .callback
            .sample
            .as_ref()
            .expect("validated sample")
            .trace
            .logits_digest
    }
    pub fn distribution_digest(&self) -> &str {
        &self
            .callback
            .sample
            .as_ref()
            .expect("validated sample")
            .trace
            .distribution_digest
    }
    pub fn nominal_probability(&self) -> f64 {
        self.link.nominal
    }
    pub fn actor_terminal_rank(&self) -> usize {
        self.rank
    }
    pub fn return_target(&self) -> f64 {
        self.share
    }
    pub fn reward(&self) -> f64 {
        if self.link.next.is_none() {
            self.share
        } else {
            0.0
        }
    }
    pub fn terminal_bootstrap(&self) -> Option<f64> {
        self.link.next.is_none().then_some(0.0)
    }
}

fn complete_counts(counts: &Counts, length: usize) -> Result<(), String> {
    if length == 0
        || length > MAX_SAMPLES
        || !counts.session_stopped
        || [
            counts.observed_callbacks,
            counts.sampler_attempts,
            counts.accepted_samples,
            counts.apply_attempts,
            counts.apply_successes,
            counts.validated_applied_steps,
        ]
        .iter()
        .any(|&count| count != length)
    {
        return Err("Actor episode requires complete applied callback counts".into());
    }
    Ok(())
}
fn state_link(previous: Option<&str>, before: &str, after: Option<&str>) -> Result<(), String> {
    parse_hex64(before)?;
    parse_hex64(after.ok_or("Actor episode is missing an applied post-state")?)?;
    if previous.is_some_and(|key| key != before) {
        return Err("Actor episode state chain mismatch".into());
    }
    Ok(())
}
fn actor_links(
    actors: &[usize],
    players: usize,
) -> Result<Vec<(usize, Option<usize>, usize)>, String> {
    if !(3..=4).contains(&players) || actors.is_empty() || actors.len() > MAX_SAMPLES {
        return Err("Invalid actor episode coverage".into());
    }
    let mut counts = [0; 4];
    let mut links = Vec::with_capacity(actors.len());
    for &actor in actors {
        if actor >= players {
            return Err("Actor episode absolute seat is out of range".into());
        }
        links.push((counts[actor], None, 0));
        counts[actor] += 1;
    }
    if counts[..players].contains(&0) {
        return Err("Actor episode is missing a decision actor".into());
    }
    let mut next = [None; 4];
    for (index, &actor) in actors.iter().enumerate().rev() {
        links[index].1 = next[actor];
        links[index].2 = next[actor].unwrap_or(actors.len()) - index - 1;
        next[actor] = Some(index);
    }
    Ok(links)
}
fn final_score(wire: &FinalScoreWire) -> Result<FinalScore, String> {
    let value = |bits: &str| -> Result<f64, String> { Ok(f64::from_bits(parse_hex64(bits)?)) };
    if wire.workers_on_gears < 0 {
        return Err("Invalid terminal worker count".into());
    }
    Ok(FinalScore {
        player_id: wire.player_id,
        points_before_final: value(&wire.points_before_final_bits)?,
        resource_points: value(&wire.resource_points_bits)?,
        skull_points: value(&wire.skull_points_bits)?,
        monument_points: value(&wire.monument_points_bits)?,
        total: value(&wire.total_bits)?,
        workers_on_gears: wire.workers_on_gears,
        rank: wire.rank,
    })
}
fn plan(
    record: &Record,
    config: &NativeStochasticConfig,
) -> Result<(Vec<ActorLink>, Vec<usize>, usize), String> {
    let header = &record.header;
    header.base_policy.validate()?;
    if record.schema != RECORD_SCHEMA
        || header.source_kind != SOURCE_KIND
        || record.training_admission != "unavailable-distinct-codec-required"
        || record.failure.is_some()
        || header.rules_version != RULES_VERSION
        || header.rules_baseline != RULES_BASELINE
        || header.catalog_hash != catalog_hash()
        || header.observation_schema != OBSERVATION_SCHEMA
        || header.move_schema != MOVE_SCHEMA
        || header.backend != "scalar"
        || header.sampling_version != SAMPLING_VERSION
        || header.rng_version != RNG_VERSION
        || header.denominator_bits != DENOMINATOR_BITS
        || header.uniform_mixture_bits != hex64(UNIFORM_MIXTURE.to_bits())
        || header.native_family_id != seed_family_id(config.environment_seed())
        || record.callbacks.len() > config.limits().max_callbacks()
        || !matches!(&header.base_policy, SeatPolicy::PublicLearned { feature_schema: 2, inference_backend, .. } if inference_backend == "scalar")
    {
        return Err("Incompatible completed actor episode source".into());
    }
    complete_counts(&record.counts, record.callbacks.len())?;
    let terminal = record
        .terminal
        .as_ref()
        .ok_or("Actor episode has no verified terminal")?;
    let scores = terminal
        .final_scores
        .iter()
        .map(final_score)
        .collect::<Result<Vec<_>, _>>()?;
    // Share the existing absolute-seat/finite/rank checks; MC's f32 result is unchanged.
    let (_, ranks, winners) = crate::state_mc_dataset::winner_shares(&scores, config.players())?;
    let positions = actor_links(
        &record.callbacks.iter().map(|c| c.actor).collect::<Vec<_>>(),
        config.players(),
    )?;
    let binding =
        crate::model::digest(&serde_json::to_vec(&header.base_policy).map_err(|e| e.to_string())?);
    let mut previous = None;
    let mut rows = 0;
    let mut rng_states = [0; 4];
    let mut draws = [0usize; 4];
    for (actor, slot) in rng_states.iter_mut().enumerate().take(config.players()) {
        *slot = config.sampling_identity().initial_actor_state(actor)?;
    }
    let mut links = Vec::with_capacity(record.callbacks.len());
    for (index, callback) in record.callbacks.iter().enumerate() {
        state_link(
            previous,
            &callback.state_before,
            callback.state_after.as_deref(),
        )?;
        let observation = callback
            .observation
            .as_ref()
            .ok_or("Actor episode is missing an observation")?;
        let sample = callback
            .sample
            .as_ref()
            .ok_or("Actor episode is missing an accepted sample")?;
        let trace = &sample.trace;
        let actor = callback.actor;
        if callback.global_callback_index != index
            || callback.failure.is_some()
            || !callback.sampler_attempted
            || !callback.apply_attempted
            || !callback.apply_succeeded
            || observation.actor != actor
            || observation.turn_player != callback.turn_player
            || observation.observation_key != callback.observation_key
            || observation.players.len() != config.players()
            || !matches!(observation.phase, Phase::Setup | Phase::Playing)
            || sample.decision.actor != actor
            || sample.decision.observation_key != callback.observation_key
            || sample.decision.policy_version != SAMPLING_VERSION
            || sample.decision.r#move != sample.chosen.r#move
            || trace.sample_index != index
            || trace.actor != actor
            || trace.policy_binding_key != binding
            || trace.legal_count != observation.legal_actions.len()
            || !(1..=crate::features::MAX_LEGAL_ACTIONS).contains(&trace.legal_count)
            || observation.legal_actions.get(trace.chosen_index) != Some(&sample.chosen)
            || trace.denominator_bits != DENOMINATOR_BITS
            || trace.reserved_rows_before != rows
            || parse_hex64(&trace.draws_before)? != draws[actor] as u64
            || parse_hex64(&trace.draws_after)? != draws[actor] as u64 + 1
            || parse_hex64(&trace.state_before)? != rng_states[actor]
        {
            return Err("Actor episode applied decision/trace link mismatch".into());
        }
        rows += trace.legal_count;
        if trace.reserved_rows_after != rows || rows > config.limits().max_candidate_rows() {
            return Err("Actor episode candidate accounting mismatch".into());
        }
        draws[actor] += 1;
        rng_states[actor] = parse_hex64(&trace.state_after)?;
        let mass = trace.mass_ticks.parse::<u64>().map_err(|e| e.to_string())?;
        let nominal = f64::from_bits(parse_hex64(&trace.nominal_probability_bits)?);
        let logp = f64::from_bits(parse_hex64(&trace.behavior_logp_bits)?);
        let probability = mass as f64 / TICKS as f64;
        if mass == 0
            || mass > TICKS
            || trace.mass_ticks != mass.to_string()
            || !nominal.is_finite()
            || nominal <= 0.0
            || nominal > 1.0
            || !logp.is_finite()
            || trace.behavior_probability_bits != hex64(probability.to_bits())
            || logp.to_bits() != (if mass == TICKS { 0.0 } else { probability.ln() }).to_bits()
        {
            return Err("Actor episode behavior likelihood mismatch".into());
        }
        let (actor_index, next, gap) = positions[index];
        links.push(ActorLink {
            actor_index,
            next,
            gap,
            mass,
            nominal,
            logp,
        });
        previous = callback.state_after.as_deref();
    }
    if previous != Some(terminal.final_state.as_str())
        || record.counts.candidate_rows_reserved != rows
        || record.counts.actor_draws.len() != 4
        || record
            .counts
            .actor_draws
            .iter()
            .zip(draws)
            .any(|(wire, count)| wire != &hex64(count as u64))
    {
        return Err("Actor episode terminal/count coverage mismatch".into());
    }
    Ok((links, ranks, winners))
}
struct CanonicalHash {
    hash: Sha256,
    bytes: usize,
}
impl Write for CanonicalHash {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        self.bytes = self
            .bytes
            .checked_add(bytes.len())
            .filter(|n| *n <= MAX_RECORD_BYTES)
            .ok_or_else(|| io::Error::other("Actor episode canonical record exceeds byte bound"))?;
        self.hash.update(bytes);
        Ok(bytes.len())
    }
    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}
fn canonical_checksum(record: &Record) -> Result<String, String> {
    let mut writer = CanonicalHash {
        hash: Sha256::new(),
        bytes: 0,
    };
    writer.hash.update(b"tzolkin-applied-actor-episode-v1\0");
    writer.hash.update(TARGET_CONTRACT.as_bytes());
    writer.hash.update(b"\0");
    serde_json::to_writer(&mut writer, record).map_err(|e| e.to_string())?;
    Ok(format!("{:x}", writer.hash.finalize()))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn actor_links_ties_and_incomplete_boundaries_are_explicit() {
        assert_eq!(
            actor_links(&[0, 1, 0, 2, 1], 3).unwrap(),
            vec![
                (0, Some(2), 1),
                (0, Some(4), 2),
                (1, None, 2),
                (0, None, 1),
                (1, None, 0),
            ]
        );
        assert!(actor_links(&[0, 1], 3).is_err());
        assert!(actor_links(&[0, 1, 3], 3).is_err());
        let score = |player_id, rank| FinalScore {
            player_id,
            rank,
            points_before_final: 0.0,
            resource_points: 0.0,
            skull_points: 0.0,
            monument_points: 0.0,
            total: 0.0,
            workers_on_gears: 0,
        };
        let (_, ranks, winners) =
            crate::state_mc_dataset::winner_shares(&[score(2, 3), score(0, 1), score(1, 1)], 3)
                .unwrap();
        assert_eq!((ranks, winners), (vec![1, 1, 3], 2));
        assert_eq!(utility(1, winners), 0.5);
        assert_eq!(utility(3, winners), 0.0);
        let (mc_shares, _, tied) =
            crate::state_mc_dataset::winner_shares(&[score(0, 1), score(1, 1), score(2, 1)], 3)
                .unwrap();
        assert_eq!(utility(1, tied), 1.0_f64 / 3.0);
        assert_ne!(utility(1, tied), f64::from(mc_shares[0]));
        assert!(
            crate::state_mc_dataset::winner_shares(&[score(0, 1), score(0, 1), score(2, 3)], 3)
                .is_err()
        );
        let mut invalid = score(1, 1);
        invalid.total = f64::NAN;
        assert!(
            crate::state_mc_dataset::winner_shares(&[score(0, 1), invalid, score(2, 3)], 3)
                .is_err()
        );
        let mut counts = Counts {
            observed_callbacks: 5,
            sampler_attempts: 5,
            accepted_samples: 5,
            candidate_rows_reserved: 5,
            actor_draws: vec![],
            apply_attempts: 5,
            apply_successes: 5,
            validated_applied_steps: 5,
            session_stopped: true,
        };
        complete_counts(&counts, 5).unwrap();
        counts.validated_applied_steps = 4;
        assert!(complete_counts(&counts, 5).is_err());
        assert!(complete_counts(&counts, 0).is_err());
        state_link(
            Some("0000000000000001"),
            "0000000000000001",
            Some("0000000000000002"),
        )
        .unwrap();
        assert!(
            state_link(
                Some("0000000000000001"),
                "0000000000000002",
                Some("0000000000000003")
            )
            .is_err()
        );
        assert!(state_link(None, "0000000000000001", None).is_err());
    }
}
