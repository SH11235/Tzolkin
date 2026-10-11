//! One sealed Count1/Paired learner against an immutable reactive roster.
//! Opponent transitions remain in the full fresh audit but never carry learner
//! likelihoods. This source has its own codec and grants no self-play admission.
use serde::{Deserialize, Serialize};
use tzolkin_core::observation::{MOVE_SCHEMA, OBSERVATION_SCHEMA};

use crate::arena::{BorrowedArenaPolicy, FrozenReactiveOpponent};
use crate::dataset::{DatasetSplit, seed_family_id, split_for_family};
use crate::public_policy_paired_long::PairedRlParent;
use crate::public_policy_update::hash;
use crate::public_rl_native::{PairedRlHandle, UpdatedPublicRlHandle};
use crate::public_stochastic::{
    PairedSamplingPolicy, RL_SAMPLING_VERSION, RNG_VERSION, RlSamplingPolicy, UNIFORM_MIXTURE,
};
use crate::public_stochastic_native::{
    NativePolicy, NativeStochasticConfig, numerical_target, run_frozen_mixed_shared,
};
use crate::public_stochastic_record::{
    Config, Header, Record, decode_typed_record, encode_typed_record, hex64,
};
use crate::replay::{RULES_BASELINE, RULES_VERSION, catalog_hash};

pub const RECORD_SCHEMA: &str = "tzolkin-public-stochastic-rl-mixed-native-record-v1";
pub const SOURCE_KIND: &str = "publicStochasticRlMixedV1";
pub const ROSTER_CONTRACT: &str = "frozen-absolute-reactive-opponents-one-learner-v1";

/// Absolute seats are frozen. Exactly one `None` denotes the learner; every
/// other seat borrows a validated policy. No raw policy DTO is executable.
pub struct FrozenMixedRoster<'a> {
    learner_seat: usize,
    opponents: Vec<Option<FrozenReactiveOpponent<'a>>>,
    descriptions: Vec<Option<FrozenOpponentIdentity>>,
    checksum: String,
}
impl<'a> FrozenMixedRoster<'a> {
    #[cfg(test)]
    pub(crate) fn synthetic_bc(
        model: &'a crate::public_model::PublicPolicyArtifact,
        guard: Option<&'a crate::public_trade_guard::TradeGuardConfig>,
        weights: &'a crate::policy::HeuristicWeights,
    ) -> Self {
        let opponents = vec![
            Some(FrozenReactiveOpponent::synthetic_bc(model, guard)),
            None,
            Some(FrozenReactiveOpponent::new(BorrowedArenaPolicy::Heuristic(weights)).unwrap()),
        ];
        let descriptions = opponents
            .iter()
            .map(|p| {
                p.as_ref().map(|p| FrozenOpponentIdentity {
                    provenance: p.description().provenance.clone(),
                    dataset_fingerprint: p.description().dataset_fingerprint.clone(),
                    model_version: p.description().model_version.map(str::to_owned),
                })
            })
            .collect::<Vec<_>>();
        let checksum = hash(
            b"tzolkin-frozen-mixed-roster-v1\0",
            &(ROSTER_CONTRACT, 1_usize, &descriptions),
        )
        .unwrap();
        Self {
            learner_seat: 1,
            opponents,
            descriptions,
            checksum,
        }
    }
    pub fn new(seats: Vec<Option<BorrowedArenaPolicy<'a>>>) -> Result<Self, String> {
        learner_slot(&seats)?;
        if seats.iter().flatten().any(|p| {
            matches!(
                p,
                BorrowedArenaPolicy::Mixed(_)
                    | BorrowedArenaPolicy::ContinuedMixed(_)
                    | BorrowedArenaPolicy::CurrentMixedTradeGuard(..)
            )
        }) {
            return Err(
                "Mixed owner requires a distinct native source codec; v1 roster refused".into(),
            );
        }
        Self::freeze(seats, b"tzolkin-frozen-mixed-roster-v1\0", ROSTER_CONTRACT)
    }
    pub(crate) fn for_current(
        seats: Vec<Option<BorrowedArenaPolicy<'a>>>,
        parent: &crate::public_rl_current_mixed_source::CurrentMixedParentIdentity,
    ) -> Result<Self, String> {
        parent.validate()?;
        for p in seats.iter().flatten() {
            match p {
                BorrowedArenaPolicy::Mixed(owner)
                    if crate::public_policy_mixed_continued::CurrentMixedPolicyHandle::from_initial(owner)?.identity()!=parent =>
                {
                    return Err("Current mixed opponent is not the actual learner parent".into());
                }
                BorrowedArenaPolicy::ContinuedMixed(owner)
                    if crate::public_policy_mixed_continued::CurrentMixedPolicyHandle::from_continued(owner)?.identity()!=parent =>
                {
                    return Err("Continued mixed opponent is not the actual learner parent".into());
                }
                BorrowedArenaPolicy::CurrentMixedTradeGuard(..) => {
                    return Err("Guarded current mixed dispatch has no native source codec".into());
                }
                BorrowedArenaPolicy::Count1(_) | BorrowedArenaPolicy::Paired(_) => {
                    return Err("Current mixed roster cannot substitute an imported parent".into());
                }
                _ => (),
            }
        }
        Self::freeze(
            seats,
            b"tzolkin-frozen-current-mixed-roster-v2\0",
            crate::public_rl_current_mixed_source::ROSTER_CONTRACT,
        )
    }
    fn freeze(
        seats: Vec<Option<BorrowedArenaPolicy<'a>>>,
        domain: &[u8],
        contract: &str,
    ) -> Result<Self, String> {
        let learner_seat = learner_slot(&seats)?;
        let opponents = seats
            .into_iter()
            .map(|p| p.map(FrozenReactiveOpponent::new).transpose())
            .collect::<Result<Vec<_>, _>>()?;
        let descriptions = opponents
            .iter()
            .map(|p| {
                p.as_ref().map(|p| {
                    let d = p.description();
                    FrozenOpponentIdentity {
                        provenance: d.provenance.clone(),
                        dataset_fingerprint: d.dataset_fingerprint.clone(),
                        model_version: d.model_version.map(str::to_owned),
                    }
                })
            })
            .collect::<Vec<_>>();
        let checksum = hash(domain, &(contract, learner_seat, &descriptions))?;
        Ok(Self {
            learner_seat,
            opponents,
            descriptions,
            checksum,
        })
    }
    pub fn learner_seat(&self) -> usize {
        self.learner_seat
    }
    pub fn players(&self) -> usize {
        self.opponents.len()
    }
    pub fn checksum(&self) -> &str {
        &self.checksum
    }
    pub(crate) fn opponent(&self, actor: usize) -> Result<&FrozenReactiveOpponent<'a>, String> {
        self.opponents
            .get(actor)
            .and_then(Option::as_ref)
            .ok_or_else(|| "Missing frozen absolute opponent seat".into())
    }
    pub(crate) fn contains_family(&self, family: &str) -> bool {
        self.opponents
            .iter()
            .flatten()
            .any(|p| p.contains_family(family))
    }
    pub(crate) fn descriptions(&self) -> &[Option<FrozenOpponentIdentity>] {
        &self.descriptions
    }
    pub fn max_opponent_search_steps(&self, callbacks: usize) -> Result<usize, String> {
        callbacks
            .checked_mul(
                self.opponents
                    .iter()
                    .flatten()
                    .map(|p| p.max_search_steps())
                    .max()
                    .unwrap_or(0),
            )
            .ok_or_else(|| "Mixed Search work bound overflow".into())
    }
}
pub(crate) fn legacy_opponent_identity_allowed(policy: &crate::replay::SeatPolicy) -> bool {
    !matches!(
        policy,
        crate::replay::SeatPolicy::PublicRlMixed { .. }
            | crate::replay::SeatPolicy::PublicRlContinuedMixed { .. }
            | crate::replay::SeatPolicy::PublicRlCurrentMixedTradeGuard { .. }
    )
}
fn learner_slot<T>(seats: &[Option<T>]) -> Result<usize, String> {
    if !(3..=4).contains(&seats.len()) || seats.iter().filter(|s| s.is_none()).count() != 1 {
        return Err(
            "Frozen mixed roster requires 3/4 absolute seats and exactly one learner".into(),
        );
    }
    Ok(seats
        .iter()
        .position(Option::is_none)
        .expect("checked learner slot"))
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(tag = "kind", rename_all = "camelCase", deny_unknown_fields)]
pub(crate) enum MixedLearnerIdentity {
    Count1 { policy: RlSamplingPolicy },
    Paired { policy: PairedSamplingPolicy },
}
impl MixedLearnerIdentity {
    pub(crate) fn checksum(&self) -> &str {
        match self {
            Self::Count1 { policy } => policy.artifact_checksum(),
            Self::Paired { policy } => policy.artifact_checksum(),
        }
    }
    pub(crate) fn count(&self) -> u64 {
        match self {
            Self::Count1 { policy } => policy.update_count(),
            Self::Paired { policy } => policy.update_count(),
        }
    }
    pub(crate) fn sampling_version(&self) -> &'static str {
        match self {
            Self::Count1 { .. } => RL_SAMPLING_VERSION,
            Self::Paired { .. } => crate::public_policy_paired_long::SAMPLING_VERSION,
        }
    }
    pub(crate) fn binding_key(&self) -> Result<String, String> {
        match self {
            Self::Count1 { policy } => policy.binding_key(),
            Self::Paired { policy } => policy.binding_key(),
        }
    }
}
enum LearnerHandle<'a> {
    Count1(UpdatedPublicRlHandle<'a>),
    Paired(PairedRlHandle<'a>),
}
/// Borrows one immutable genuine owner. Its existing sampler/RNG/numeric target
/// stays unchanged; only actual learner-seat observations reach that sampler.
pub struct MixedRlLearner<'a> {
    handle: LearnerHandle<'a>,
    identity: MixedLearnerIdentity,
    families: Vec<String>,
}
impl<'a> MixedRlLearner<'a> {
    pub(crate) fn new(parent: PairedRlParent<'a>) -> Result<Self, String> {
        match parent {
            PairedRlParent::Count1(p) => {
                let h = UpdatedPublicRlHandle::new(p)?;
                let identity = MixedLearnerIdentity::Count1 {
                    policy: RlSamplingPolicy::from_handle(&h)?,
                };
                let families = h.family_closure().to_vec();
                Ok(Self {
                    handle: LearnerHandle::Count1(h),
                    identity,
                    families,
                })
            }
            PairedRlParent::Paired(p) => {
                let h = PairedRlHandle::new(p)?;
                let identity = MixedLearnerIdentity::Paired {
                    policy: PairedSamplingPolicy::from_handle(&h)?,
                };
                let families = h.family_set().iter().cloned().collect();
                Ok(Self {
                    handle: LearnerHandle::Paired(h),
                    identity,
                    families,
                })
            }
        }
    }
    pub fn parent_artifact_checksum(&self) -> &str {
        self.identity.checksum()
    }
    pub fn parent_update_count(&self) -> u64 {
        self.identity.count()
    }
    fn policy(&self) -> NativePolicy<'_, 'a> {
        match &self.handle {
            LearnerHandle::Count1(h) => NativePolicy::Rl(h),
            LearnerHandle::Paired(h) => NativePolicy::Paired(h),
        }
    }
    pub(crate) fn header(
        &self,
        native: &NativeStochasticConfig,
        roster: &FrozenMixedRoster<'_>,
    ) -> Result<Header<MixedRlSource>, String> {
        let family = seed_family_id(native.environment_seed());
        if native.players() != roster.players()
            || split_for_family(&family)? != DatasetSplit::Train
            || self.families.contains(&family)
            || roster.contains_family(&family)
        {
            return Err(
                "Mixed Train family overlaps learner/opponent closure or roster size".into(),
            );
        }
        let max_opponent_action_rows = native
            .limits()
            .max_callbacks()
            .checked_mul(crate::features::MAX_LEGAL_ACTIONS)
            .ok_or("Mixed opponent action-row bound overflow")?;
        let base_policy = MixedRlSource {
            contract: ROSTER_CONTRACT.into(),
            learner_seat: roster.learner_seat,
            learner: self.identity.clone(),
            opponents: roster.descriptions.clone(),
            roster_checksum: roster.checksum.clone(),
            max_opponent_action_rows,
            max_opponent_search_steps: roster
                .max_opponent_search_steps(native.limits().max_callbacks())?,
        };
        Ok(Header {
            source_kind: SOURCE_KIND.into(),
            rules_version: RULES_VERSION,
            rules_baseline: RULES_BASELINE.into(),
            catalog_hash: catalog_hash(),
            observation_schema: OBSERVATION_SCHEMA,
            move_schema: MOVE_SCHEMA,
            config: Config::from_config(native),
            base_policy,
            sampling_version: self.identity.sampling_version().into(),
            rng_version: RNG_VERSION.into(),
            denominator_bits: 53,
            uniform_mixture_bits: hex64(UNIFORM_MIXTURE.to_bits()),
            backend: "scalar".into(),
            numerical_target: numerical_target(),
            native_family_id: family,
            raw_artifact_identity_available: false,
            producer_authenticated: false,
            independent_seed_origins_verified: false,
        })
    }
}
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub(crate) struct MixedRlSource {
    pub contract: String,
    pub learner_seat: usize,
    pub learner: MixedLearnerIdentity,
    pub opponents: Vec<Option<FrozenOpponentIdentity>>,
    pub roster_checksum: String,
    pub max_opponent_action_rows: usize,
    pub max_opponent_search_steps: usize,
}
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub(crate) struct FrozenOpponentIdentity {
    provenance: crate::replay::SeatPolicy,
    dataset_fingerprint: Option<String>,
    model_version: Option<String>,
}
#[cfg(test)]
impl FrozenOpponentIdentity {
    pub(crate) fn test_heuristic() -> Self {
        Self {
            provenance: crate::replay::SeatPolicy::Heuristic {
                policy_version: crate::POLICY_VERSION.into(),
                weights: crate::policy::HeuristicWeights::default(),
            },
            dataset_fingerprint: None,
            model_version: None,
        }
    }
}

pub struct CollectedMixedRlGame {
    record: Record<MixedRlSource>,
}
pub struct AuditedMixedRlGame {
    record: Record<MixedRlSource>,
}
macro_rules! counters {
    ($ty:ty) => {
        impl $ty {
            pub fn complete(&self) -> bool {
                self.record.terminal.is_some() && self.record.failure.is_none()
            }
            pub fn observed_callbacks(&self) -> usize {
                self.record.counts.observed_callbacks
            }
            pub fn learner_callbacks(&self) -> usize {
                self.record.counts.accepted_samples
            }
            pub fn learner_candidate_rows(&self) -> usize {
                self.record.counts.candidate_rows_reserved
            }
            pub fn opponent_attempts(&self) -> usize {
                self.record
                    .callbacks
                    .iter()
                    .filter(|c| c.opponent_attempted)
                    .count()
            }
            pub fn applied_choices(&self) -> usize {
                self.record.counts.apply_successes
            }
        }
    };
}
counters!(CollectedMixedRlGame);
counters!(AuditedMixedRlGame);
impl AuditedMixedRlGame {
    pub(crate) fn record(&self) -> &Record<MixedRlSource> {
        &self.record
    }
}
pub(crate) fn collect_mixed_rl_native(
    native: &NativeStochasticConfig,
    learner: &MixedRlLearner<'_>,
    roster: &FrozenMixedRoster<'_>,
) -> Result<CollectedMixedRlGame, String> {
    let record = run_frozen_mixed_shared(
        native,
        learner.policy(),
        roster,
        learner.header(native, roster)?,
        None,
    )?;
    Ok(CollectedMixedRlGame { record })
}
pub(crate) fn encode_mixed_rl_record(game: &CollectedMixedRlGame) -> Result<Vec<u8>, String> {
    encode_typed_record(&game.record)
}
pub(crate) fn audit_mixed_rl_record_bytes(
    bytes: &[u8],
    learner: &MixedRlLearner<'_>,
    roster: &FrozenMixedRoster<'_>,
) -> Result<AuditedMixedRlGame, String> {
    let record: Record<MixedRlSource> = decode_typed_record(bytes)?;
    if record.schema != RECORD_SCHEMA
        || record.header.source_kind != SOURCE_KIND
        || record.training_admission != "unavailable-distinct-codec-required"
        || record
            .header
            .base_policy
            .opponents
            .iter()
            .flatten()
            .any(|p| !legacy_opponent_identity_allowed(&p.provenance))
    {
        return Err("Wrong mixed RL record schema/source/admission".into());
    }
    let native = record.header.config.checked_config()?;
    if record.callbacks.len() > native.limits().max_callbacks()
        || record.header != learner.header(&native, roster)?
    {
        return Err("Mixed RL full native/learner/frozen-roster header differs before work".into());
    }
    for c in &record.callbacks {
        let own = c.actor == roster.learner_seat();
        if c.actor >= native.players()
            || (own && (c.opponent_attempted || c.opponent_choice.is_some()))
            || (!own && (c.sampler_attempted || c.sample.is_some()))
        {
            return Err("Mixed RL sample/opponent role differs before work".into());
        }
    }
    let record = run_frozen_mixed_shared(
        &native,
        learner.policy(),
        roster,
        learner.header(&native, roster)?,
        Some(&record),
    )?;
    Ok(AuditedMixedRlGame { record })
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn frozen_roster_requires_one_absolute_learner_and_binds_policy_order() {
        assert!(learner_slot::<u8>(&[Some(1), Some(2), Some(3)]).is_err());
        assert!(learner_slot::<u8>(&[None, None, Some(3)]).is_err());
        assert_eq!(learner_slot::<u8>(&[Some(1), None, Some(2)]).unwrap(), 1);
        let w = crate::policy::HeuristicWeights::default();
        let a = FrozenMixedRoster::new(vec![
            None,
            Some(BorrowedArenaPolicy::Heuristic(&w)),
            Some(BorrowedArenaPolicy::CornFirstSetup(&w)),
        ])
        .unwrap();
        let b = FrozenMixedRoster::new(vec![
            None,
            Some(BorrowedArenaPolicy::CornFirstSetup(&w)),
            Some(BorrowedArenaPolicy::Heuristic(&w)),
        ])
        .unwrap();
        assert_ne!(a.checksum(), b.checksum());
        assert!(a.opponent(0).is_err());
        assert_eq!(a.max_opponent_search_steps(4000).unwrap(), 0);
    }
}
