//! Owned actor snapshots and a fixed, explicit semantic/procedural mixture.
//! Captured source custody and allocation closure are not source authentication.
use std::collections::{BTreeMap, BTreeSet};

use tzolkin_core::observation::Observation;
use tzolkin_inference::policy::NumericPolicyDistribution;

use crate::conditional_human_rl_update::{
    ConditionalFrozenMixRoster, ConditionalFrozenPool, ConditionalNativeExclusions,
};
use crate::conditional_semantic_playing_rl::{
    SemanticOriginCustody, SemanticOwnerView, parameter_checksum,
};
use crate::dataset::DatasetSplit;
use crate::features::FeatureEncoder;
use crate::kernel::ResolvedKernel;
use crate::public_model::{PublicPolicyModel, predict_semantic_playing_numeric};
use crate::public_policy_update::hash;
use crate::public_stochastic::SamplingStreamIdentity;
use crate::semantic_playing_rl_collection::DeclaredSamplingHistory;
use crate::semantic_playing_rl_update::SemanticPlayingCurrentIdentity;

pub const MIXTURE_VERSION: &str = "semantic-playing-fixed-snapshot-corn-first-pool-v1";
pub const OPPONENT_SAMPLING_VERSION: &str = "semantic-playing-frozen-opponent-tick53-v1";

fn source_binding(
    origin: &SemanticOriginCustody,
    external: &ConditionalNativeExclusions,
    history: &DeclaredSamplingHistory,
) -> Result<String, String> {
    hash(
        b"semantic-frozen-source-closure-v1\0",
        &(origin.binding()?, external.descriptor(), history.checksum()),
    )
}

/// The weight identity and allocation capture identity are deliberately separate.
#[derive(Debug)]
pub struct FrozenSemanticSnapshotIdentity {
    current: String,
    root: String,
    actor: String,
    capture: String,
    sources: String,
    checksum: String,
}
impl FrozenSemanticSnapshotIdentity {
    pub fn current_identity_checksum(&self) -> &str {
        &self.current
    }
    pub fn root_checksum(&self) -> &str {
        &self.root
    }
    pub fn actor_checksum(&self) -> &str {
        &self.actor
    }
    pub fn capture_ledger_checksum(&self) -> &str {
        &self.capture
    }
    pub fn checksum(&self) -> &str {
        &self.checksum
    }
    pub fn source_authenticated(&self) -> bool {
        false
    }
}

/// Owned immutable actor-only copy; no Deserialize, raw constructor or conversion
/// back to a training owner. The current learner can change after this capture.
pub struct FrozenSemanticSnapshot<'a> {
    actor: PublicPolicyModel,
    origin: SemanticOriginCustody,
    external: &'a ConditionalNativeExclusions,
    history: &'a DeclaredSamplingHistory,
    families: BTreeSet<String>,
    streams: BTreeSet<[u64; 5]>,
    human_families: BTreeSet<String>,
    identity: FrozenSemanticSnapshotIdentity,
}
pub(crate) struct SnapshotCaptureLedger<'a> {
    pub(crate) external: &'a ConditionalNativeExclusions,
    pub(crate) history: &'a DeclaredSamplingHistory,
    pub(crate) families: BTreeSet<String>,
    pub(crate) streams: BTreeSet<[u64; 5]>,
    pub(crate) sampling_seed: u64,
    pub(crate) replicate: u64,
    pub(crate) next: u64,
    pub(crate) attempts: usize,
    pub(crate) stopped: bool,
}
impl<'a> FrozenSemanticSnapshot<'a> {
    pub(crate) fn capture(
        owner: &SemanticOwnerView<'_>,
        current: &SemanticPlayingCurrentIdentity,
        ledger: SnapshotCaptureLedger<'a>,
    ) -> Result<Self, String> {
        let SnapshotCaptureLedger {
            external,
            history,
            families,
            streams,
            sampling_seed,
            replicate,
            next,
            attempts,
            stopped,
        } = ledger;
        history.recheck()?;
        owner.with_checked(external, history.pins(), |scope| {
            let actor_checksum = parameter_checksum(
                b"semantic-current-actor-bits-v1\0",
                scope.actor().parameters(),
            );
            if scope.parent_checksum() != current.checksum()
                || actor_checksum != current.actor_checksum()
            {
                return Err("Snapshot differs from last admitted semantic actor".into());
            }
            let origin = owner.origin_custody();
            let human_families = origin
                .reservations()
                .keys()
                .map(|table| crate::human_initial_wealth::table_family_id(table))
                .collect::<Result<BTreeSet<_>, _>>()?;
            let sources = source_binding(&origin, external, history)?;
            let capture = hash(
                b"semantic-frozen-allocation-capture-v1\0",
                &(
                    (&families, &streams),
                    (sampling_seed, replicate, next, attempts, stopped),
                ),
            )?;
            let checksum = hash(
                b"semantic-frozen-actor-snapshot-v1\0",
                &(
                    (
                        current.checksum(),
                        current.root_checksum(),
                        &actor_checksum,
                        &capture,
                        &sources,
                    ),
                    (
                        crate::features::SEMANTIC_PLAYING_FEATURE_SCHEMA,
                        "scalar",
                        false,
                    ),
                ),
            )?;
            // One bounded copy outside scoring. No critic values or ownership
            // authority are copied from the parameter slices.
            let actor = scope.actor().clone();
            actor.validate()?;
            Ok(Self {
                actor,
                origin,
                external,
                history,
                families,
                streams,
                human_families,
                identity: FrozenSemanticSnapshotIdentity {
                    current: current.checksum().into(),
                    root: current.root_checksum().into(),
                    actor: actor_checksum,
                    capture,
                    sources,
                    checksum,
                },
            })
        })
    }
    pub fn identity(&self) -> Result<&FrozenSemanticSnapshotIdentity, String> {
        self.recheck()?;
        Ok(&self.identity)
    }
    pub fn human_table_reservations(&self) -> Result<&BTreeMap<String, DatasetSplit>, String> {
        self.recheck()?;
        Ok(self.origin.reservations())
    }
    pub fn consumed_families(&self) -> &BTreeSet<String> {
        &self.families
    }
    pub(crate) fn recheck(&self) -> Result<(), String> {
        self.check_actor()?;
        self.recheck_sources()
    }
    fn check_actor(&self) -> Result<(), String> {
        self.actor.validate()?;
        if parameter_checksum(b"semantic-current-actor-bits-v1\0", self.actor.parameters())
            != self.identity.actor
        {
            return Err("Frozen semantic actor bits changed".into());
        }
        Ok(())
    }
    fn recheck_sources(&self) -> Result<(), String> {
        self.origin.recheck()?;
        self.external.recheck()?;
        self.history.recheck()
    }
    pub(crate) fn binding(&self) -> &str {
        &self.identity.checksum
    }
    pub(crate) fn contains_family(&self, family: &str) -> bool {
        self.families.contains(family)
            || self.human_families.contains(family)
            || self.external.contains_family(family)
    }
    pub(crate) fn contains_stream(&self, identity: SamplingStreamIdentity, actor: usize) -> bool {
        self.streams.contains(&[
            identity.sampling_seed().value(),
            identity.episode_ordinal(),
            identity.replicate_ordinal(),
            identity.players() as u64,
            actor as u64,
        ]) || self.history.contains(identity, actor)
    }
    /// Called only inside the closed source-checked batch/episode borrow.
    pub(crate) fn distribution(
        &self,
        o: &Observation,
        kernel: ResolvedKernel,
    ) -> Result<NumericPolicyDistribution, String> {
        let encoder = FeatureEncoder::new_semantic_playing(o)?;
        let rows = (0..o.legal_actions.len())
            .map(|i| encoder.encode_legal_tagged(i))
            .collect::<Result<Vec<_>, _>>()?;
        predict_semantic_playing_numeric(&self.actor, kernel, &rows)
    }
}

/// One preselected member/absolute-seat assignment, never a model constructor.
pub struct SnapshotSlot<'a> {
    member: usize,
    actor: usize,
    snapshot: &'a FrozenSemanticSnapshot<'a>,
}
impl<'a> SnapshotSlot<'a> {
    pub fn new(
        member: usize,
        actor: usize,
        snapshot: &'a FrozenSemanticSnapshot<'a>,
    ) -> Result<Self, String> {
        if member >= 4 || actor >= if member < 2 { 3 } else { 4 } {
            return Err("Snapshot slot is outside the fixed whole-four roster".into());
        }
        Ok(Self {
            member,
            actor,
            snapshot,
        })
    }
}

/// This opt-in route uses common CornFirst Setup and procedural-only legacy
/// Playing slots. Collection admits only past snapshots of the same sealed
/// learning chain, with identical source declarations and already retained burns.
/// The legacy pool's wire and existing methods are unchanged.
pub struct SemanticFrozenMixture<'a> {
    legacy: &'a ConditionalFrozenPool,
    slots: Vec<SnapshotSlot<'a>>,
    binding: String,
}
impl<'a> SemanticFrozenMixture<'a> {
    pub fn new(
        legacy: &'a ConditionalFrozenPool,
        slots: Vec<SnapshotSlot<'a>>,
    ) -> Result<Self, String> {
        let rosters = legacy.freeze_procedural()?;
        if slots.is_empty()
            || slots.len() > 10
            || slots
                .windows(2)
                .any(|p| (p[0].member, p[0].actor) >= (p[1].member, p[1].actor))
            || slots
                .iter()
                .any(|s| s.actor == rosters[s.member].learner_seat())
        {
            return Err("Mixture needs sorted unique preselected opponent slots".into());
        }
        let identities = slots
            .iter()
            .map(|s| (s.member, s.actor, s.snapshot.binding()))
            .collect::<Vec<_>>();
        let binding = hash(
            b"semantic-frozen-fixed-mixture-v1\0",
            &(
                MIXTURE_VERSION,
                legacy.procedural_binding()?,
                identities,
                crate::setup_policy::POLICY_VERSION,
                crate::policy::HeuristicWeights::default(),
            ),
        )?;
        let mixture = Self {
            legacy,
            slots,
            binding,
        };
        mixture.recheck()?;
        Ok(mixture)
    }
    pub fn checksum(&self) -> &str {
        &self.binding
    }
    pub(crate) fn require_receiver(
        &self,
        owner: &SemanticOwnerView<'_>,
        external: &ConditionalNativeExclusions,
        history: &DeclaredSamplingHistory,
        burns: (&BTreeSet<String>, &BTreeSet<[u64; 5]>),
        admitted: &BTreeSet<String>,
    ) -> Result<(), String> {
        let sources = source_binding(&owner.origin_custody(), external, history)?;
        for slot in &self.slots {
            let snapshot = slot.snapshot;
            if snapshot.identity.root != owner.root_checksum() {
                return Err("Semantic snapshot has a foreign fresh root".into());
            }
            if snapshot.identity.sources != sources {
                return Err("Semantic snapshot source declaration differs".into());
            }
            if !snapshot.families.is_subset(burns.0) || !snapshot.streams.is_subset(burns.1) {
                return Err("Semantic snapshot prior burns are not retained".into());
            }
            if !admitted.contains(&snapshot.identity.current) {
                return Err("Semantic snapshot is not an admitted ancestor".into());
            }
        }
        Ok(())
    }
    pub(crate) fn recheck(&self) -> Result<(), String> {
        // Different weight snapshots often retain identical source files.
        // Recheck each exact source closure once at this boundary, never per move.
        let mut sources = BTreeSet::new();
        let mut actors = BTreeSet::new();
        for slot in &self.slots {
            if actors.insert(slot.snapshot.binding()) {
                slot.snapshot.check_actor()?;
            }
            if sources.insert(&slot.snapshot.identity.sources) {
                slot.snapshot.recheck_sources()?;
            }
        }
        Ok(())
    }
    pub(crate) fn freeze(&self) -> Result<[SemanticMixtureRoster<'_>; 4], String> {
        self.legacy
            .freeze_procedural()?
            .into_iter()
            .enumerate()
            .map(|(member, legacy)| {
                let mut snapshots = (0..if member < 2 { 3 } else { 4 })
                    .map(|_| None)
                    .collect::<Vec<_>>();
                for slot in self.slots.iter().filter(|s| s.member == member) {
                    snapshots[slot.actor] = Some(slot.snapshot);
                }
                let binding = hash(
                    b"semantic-frozen-mixture-absolute-roster-v1\0",
                    &(
                        &self.binding,
                        member,
                        legacy.learner_seat(),
                        legacy.checksum(),
                    ),
                )?;
                Ok(SemanticMixtureRoster {
                    legacy,
                    snapshots,
                    binding,
                })
            })
            .collect::<Result<Vec<_>, String>>()?
            .try_into()
            .map_err(|_| "Semantic mixture roster shape".into())
    }
}
pub(crate) struct SemanticMixtureRoster<'a> {
    pub(crate) legacy: ConditionalFrozenMixRoster<'a>,
    snapshots: Vec<Option<&'a FrozenSemanticSnapshot<'a>>>,
    binding: String,
}
impl<'a> SemanticMixtureRoster<'a> {
    pub(crate) fn snapshot(&self, actor: usize) -> Option<&'a FrozenSemanticSnapshot<'a>> {
        self.snapshots.get(actor).copied().flatten()
    }
    pub(crate) fn contains_family(&self, family: &str) -> bool {
        self.legacy.contains_family(family)
            || self
                .snapshots
                .iter()
                .flatten()
                .any(|s| s.contains_family(family))
    }
    pub(crate) fn contains_stream(&self, identity: SamplingStreamIdentity, actor: usize) -> bool {
        self.snapshots
            .iter()
            .flatten()
            .any(|s| s.contains_stream(identity, actor))
    }
    pub(crate) fn checksum(&self) -> &str {
        &self.binding
    }
}
