//! Inference-only paired evaluation, also reused by explicit conditional origins.
//! Numeric identities and external exclusions do not authenticate a producer,
//! recreate an owner, or grant training, fresh Test, or strength authority.
use super::conditional::{
    ConditionalCountZeroArenaIdentity, ConditionalCountZeroArenaPolicy, ConditionalDiagnostics,
    ConditionalIdentity, ConditionalMixedArenaIdentity, ConditionalMixedArenaPolicy,
    ConditionalPolicy, GameSession,
};
use super::phase_routed::{
    PhaseRouteDiagnostics, PhaseRoutedDevelopmentIdentity, PhaseRoutedDevelopmentPolicy,
};
use super::teacher::{
    ResourceCycleTeacherArenaIdentity, ResourceCycleTeacherArenaPolicy,
    SemanticSetupResourceCycleTeacherArenaPolicy, SemanticSetupTeacherArenaIdentity,
    SemanticSetupTeacherReference,
};
use super::{
    ArmResult, BorrowedArenaPolicy, BorrowedArenaWorkBounds, InitialConditions, PairedStatistics,
    PolicyChooser, PolicyHandle, SeatPair, SeedBlock, block, family_id, seed_partition, summarize,
};
use crate::Decision;
use crate::dataset::DatasetSplit;
use crate::policy::HeuristicWeights;
use crate::replay::{self, SeatPolicy};
use crate::search::PreparedSearch;
use crate::search_native::SearchSummary;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;
use std::time::Instant;
use tzolkin_core::observation::{Observation, observe};
use tzolkin_core::{GameOptions, Phase};
use tzolkin_inference::deployment::{GuardedDeployment, LoadedDeployment, MAX_GAME_CALLBACKS};
use tzolkin_inference::trade_guard::TradeGuardConfig;

pub const SCHEMA: &str = "tzolkin-numeric-deployment-validation-arena-v1";
pub const RESOURCE_CYCLE_ARENA_SCHEMA: &str =
    "tzolkin-resource-cycle-v5-development-validation-arena-v1";
pub const MAX_ROWS_PER_CALLBACK: usize = 1024;
const MAX_EXCLUDED_FAMILIES: usize = 65_536;
const _: () = assert!(MAX_GAME_CALLBACKS == replay::MAX_DECISIONS);

fn digest(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}

/// The caller supplies this evidence for the fixed candidate/reference/pool.
/// This module checks disjointness, but cannot attest completeness or lineage.
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct ExternalExclusions {
    pub evidence_sha256: String,
    pub excluded_family_ids: Vec<String>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct NumericArenaConfig {
    pub schema: String,
    pub players: usize,
    pub seeds: Vec<u32>,
    pub bootstrap_seed: u64,
    pub external_exclusions: ExternalExclusions,
}
impl NumericArenaConfig {
    pub fn validate(&self) -> Result<(), String> {
        let exclusions = &self.external_exclusions;
        if self.schema != SCHEMA
            || !(3..=4).contains(&self.players)
            || self.seeds.is_empty()
            || self.seeds.len() > super::MAX_SEEDS
            || !digest(&exclusions.evidence_sha256)
            || exclusions.excluded_family_ids.len() > MAX_EXCLUDED_FAMILIES
        {
            return Err(
                "Numeric Arena requires base 3/4p, bounded Validation and external evidence".into(),
            );
        }
        let excluded = exclusions
            .excluded_family_ids
            .iter()
            .collect::<BTreeSet<_>>();
        if excluded.len() != exclusions.excluded_family_ids.len()
            || excluded.iter().any(|id| !digest(id))
        {
            return Err("Invalid or repeated externally excluded family".into());
        }
        let mut selected = BTreeSet::new();
        for seed in &self.seeds {
            let id = family_id(*seed);
            if !selected.insert(id.clone())
                || seed_partition(*seed)? != DatasetSplit::Validation
                || excluded.contains(&id)
            {
                return Err(
                    "Numeric Arena rejects repeated, excluded or non-Validation families".into(),
                );
            }
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct NumericIdentity {
    pub deployment_checksum: String,
    pub claimed_source_role: String,
    pub claimed_source_checksum: String,
    pub claimed_update_count: Option<u64>,
    pub claimed_current_mixed_identity: Option<serde_json::Value>,
    pub source_numerical_target: String,
    pub raw_policy_version: String,
    pub backend: String,
    pub guard: TradeGuardConfig,
    pub configuration_key: String,
    pub inference_version: String,
}

/// No raw owner constructor or training roster conversion is provided.
pub struct NumericPolicy {
    loaded: Option<LoadedDeployment>,
    identity: NumericIdentity,
}
impl NumericPolicy {
    pub fn new(loaded: LoadedDeployment, guard: TradeGuardConfig) -> Result<Self, String> {
        let configuration_key = guard.configuration_key()?;
        let identity = NumericIdentity {
            deployment_checksum: loaded.checksum().into(),
            claimed_source_role: loaded.source_role().into(),
            claimed_source_checksum: loaded.source_checksum().into(),
            claimed_update_count: loaded.update_count(),
            claimed_current_mixed_identity: loaded
                .current_mixed_identity()
                .map(serde_json::to_value)
                .transpose()
                .map_err(|e| e.to_string())?,
            source_numerical_target: loaded.source_numerical_target().into(),
            raw_policy_version: loaded.policy_version().into(),
            backend: loaded.backend().into(),
            guard,
            configuration_key,
            inference_version: tzolkin_inference::deployment::GUARDED_DEPLOYMENT_VERSION.into(),
        };
        Ok(Self {
            loaded: Some(loaded),
            identity,
        })
    }
    pub fn identity(&self) -> &NumericIdentity {
        &self.identity
    }
    pub(super) fn borrowed_deployment(&self) -> Result<&LoadedDeployment, String> {
        self.loaded
            .as_ref()
            .ok_or_else(|| "Numeric Arena immutable storage is unavailable".into())
    }
    pub(super) fn distribution(
        &self,
        observation: &Observation,
    ) -> Result<tzolkin_inference::policy::NumericPolicyDistribution, String> {
        if observation.legal_actions.len() > MAX_ROWS_PER_CALLBACK {
            return Err("Numeric Arena candidate-row cap exceeded before prediction".into());
        }
        self.borrowed_deployment()?.distribution(observation)
    }
}

pub enum Opponent<'a> {
    GuardedNumeric(Box<NumericPolicy>),
    Heuristic(&'a HeuristicWeights),
    CornFirstSetup(&'a HeuristicWeights),
    CornFirstUxmalOpening(&'a HeuristicWeights),
    Search(&'a PreparedSearch),
    ResourceCycle(super::resource_cycle::ResourceCycleConfig),
}

#[derive(Clone, Debug, Serialize)]
#[serde(
    tag = "kind",
    rename_all = "camelCase",
    rename_all_fields = "camelCase"
)]
pub enum PolicyIdentity {
    GuardedNumeric {
        identity: NumericIdentity,
    },
    Reactive {
        provenance: SeatPolicy,
    },
    ResourceCycle {
        configuration: super::resource_cycle::ResourceCycleConfig,
    },
    ConditionalHumanHybrid {
        identity: Box<ConditionalIdentity>,
    },
    ConditionalHumanMixedRl {
        identity: Box<ConditionalMixedArenaIdentity>,
    },
    ConditionalHumanCountZeroAllPhase {
        identity: Box<ConditionalCountZeroArenaIdentity>,
    },
    ResourceCycleV5LearnedTeacher {
        identity: Box<ResourceCycleTeacherArenaIdentity>,
    },
    ResourceCycleV5SemanticSetupLearnedTeacher {
        identity: Box<SemanticSetupTeacherArenaIdentity>,
    },
    SemanticSetupTeacherNumericPlayingDevelopment {
        identity: Box<PhaseRoutedDevelopmentIdentity>,
    },
}
enum Role<'a> {
    Numeric(NumericPolicy),
    Conditional(Box<ConditionalPolicy<'a>>),
    ConditionalMixed(Box<ConditionalMixedArenaPolicy<'a>>),
    ConditionalCountZero(Box<ConditionalCountZeroArenaPolicy<'a>>),
    Teacher(Box<ResourceCycleTeacherArenaPolicy<'a>>),
    SemanticSetupTeacher(Box<SemanticSetupResourceCycleTeacherArenaPolicy<'a>>),
    PhaseRouted(Box<PhaseRoutedDevelopmentPolicy<'a>>),
    Reactive(PolicyHandle<'a>),
    ResourceCycle(super::resource_cycle::ResourceCycleConfig),
}
impl<'a> Role<'a> {
    fn from_semantic_setup_reference(reference: SemanticSetupTeacherReference<'a>) -> Self {
        match reference {
            SemanticSetupTeacherReference::GuardedNumeric(policy) => Self::Numeric(*policy),
            SemanticSetupTeacherReference::ConditionalHumanCountZeroAllPhase(policy) => {
                Self::ConditionalCountZero(policy)
            }
        }
    }
    fn from_opponent(opponent: Opponent<'a>) -> Result<Self, String> {
        let reactive = match opponent {
            Opponent::GuardedNumeric(policy) => return Ok(Self::Numeric(*policy)),
            Opponent::ResourceCycle(configuration) => {
                configuration.validate()?;
                return Ok(Self::ResourceCycle(configuration));
            }
            Opponent::Heuristic(weights) => BorrowedArenaPolicy::Heuristic(weights),
            Opponent::CornFirstSetup(weights) => BorrowedArenaPolicy::CornFirstSetup(weights),
            Opponent::CornFirstUxmalOpening(weights) => {
                BorrowedArenaPolicy::CornFirstUxmalOpening(weights)
            }
            Opponent::Search(policy) => BorrowedArenaPolicy::Search(policy),
        };
        Ok(Self::Reactive(PolicyHandle::borrowed(reactive, &[])?.0))
    }
    fn identity(&self) -> PolicyIdentity {
        match self {
            Self::Numeric(policy) => PolicyIdentity::GuardedNumeric {
                identity: policy.identity.clone(),
            },
            Self::Conditional(policy) => PolicyIdentity::ConditionalHumanHybrid {
                identity: Box::new(policy.identity().clone()),
            },
            Self::ConditionalMixed(policy) => PolicyIdentity::ConditionalHumanMixedRl {
                identity: Box::new(policy.identity().clone()),
            },
            Self::ConditionalCountZero(policy) => {
                PolicyIdentity::ConditionalHumanCountZeroAllPhase {
                    identity: Box::new(policy.identity().clone()),
                }
            }
            Self::Teacher(policy) => PolicyIdentity::ResourceCycleV5LearnedTeacher {
                identity: Box::new(policy.identity().clone()),
            },
            Self::SemanticSetupTeacher(policy) => {
                PolicyIdentity::ResourceCycleV5SemanticSetupLearnedTeacher {
                    identity: Box::new(policy.identity().clone()),
                }
            }
            Self::PhaseRouted(policy) => {
                PolicyIdentity::SemanticSetupTeacherNumericPlayingDevelopment {
                    identity: Box::new(policy.identity().clone()),
                }
            }
            Self::Reactive(policy) => PolicyIdentity::Reactive {
                provenance: policy.description.provenance.clone(),
            },
            Self::ResourceCycle(configuration) => PolicyIdentity::ResourceCycle {
                configuration: configuration.clone(),
            },
        }
    }
    fn guard_key(&self) -> Option<&str> {
        match self {
            Self::Numeric(policy) => Some(&policy.identity.configuration_key),
            Self::Conditional(policy) => Some(&policy.identity().configuration_key),
            Self::ConditionalMixed(policy) => Some(&policy.identity().configuration_key),
            Self::ConditionalCountZero(policy) => Some(&policy.identity().configuration_key),
            Self::Teacher(policy) => Some(&policy.identity().configuration_key),
            Self::SemanticSetupTeacher(policy) => {
                Some(&policy.identity().teacher.configuration_key)
            }
            Self::PhaseRouted(policy) => Some(&policy.identity().configuration_key),
            _ => None,
        }
    }
    fn search(&self) -> Option<&PreparedSearch> {
        match self {
            Self::Reactive(PolicyHandle {
                chooser: PolicyChooser::Search(policy),
                ..
            }) => Some(policy),
            _ => None,
        }
    }
}

/// Choices are observed before core apply. No applied-choice count is inferred
/// on a failed game; callbacks/rows include attempted work before that failure.
#[derive(Clone, Debug, Default, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct GuardDiagnostics {
    pub notifications_attempted: usize,
    pub notifications_returned: usize,
    pub choices_attempted: usize,
    pub choices_returned: usize,
    pub candidate_rows_attempted: usize,
    pub candidate_rows_returned: usize,
    pub forced_exits: usize,
    pub repeat_exits: usize,
    pub budget_exits: usize,
    pub applied_choices: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub conditional: Option<ConditionalDiagnostics>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub phase_routes: Option<PhaseRouteDiagnostics>,
}
#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct GuardPair {
    pub seat: usize,
    pub candidate: Vec<Option<GuardDiagnostics>>,
    pub reference: Vec<Option<GuardDiagnostics>>,
}
#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct GuardBlock {
    pub seed: u32,
    pub pairs: Vec<GuardPair>,
}
#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct NumericArenaReport {
    pub schema: String,
    pub config: NumericArenaConfig,
    pub options: GameOptions,
    pub configuration_sha256: String,
    pub candidate: PolicyIdentity,
    pub reference: PolicyIdentity,
    pub opponent_pool: Vec<PolicyIdentity>,
    pub work_bounds: BorrowedArenaWorkBounds,
    pub blocks: Vec<SeedBlock>,
    pub guards: Vec<GuardBlock>,
    pub statistics: PairedStatistics,
    pub development: bool,
    pub fresh_holdout_claimed: bool,
    pub producer_authenticated: bool,
    pub training_admission_granted: bool,
    pub rules_baseline: String,
    pub catalog_hash: String,
    pub target_arch: String,
    pub target_os: String,
}

fn work_bounds(
    config: &NumericArenaConfig,
    pool: &[Role<'_>],
) -> Result<BorrowedArenaWorkBounds, String> {
    let planned_games = config.seeds.len() as u64 * config.players as u64 * 2;
    let max_callbacks = planned_games * MAX_GAME_CALLBACKS as u64;
    let mut max_search_atomic_steps = 0u64;
    for seat in 0..config.players {
        let cap = pool
            .iter()
            .enumerate()
            .filter(|(i, _)| *i != seat)
            .filter_map(|(_, role)| role.search())
            .map(|p| p.config().max_total_steps as u64)
            .max()
            .unwrap_or(0);
        max_search_atomic_steps = max_search_atomic_steps
            .checked_add(
                cap.checked_mul(MAX_GAME_CALLBACKS as u64 * config.seeds.len() as u64 * 2)
                    .ok_or("Numeric Arena Search bound overflow")?,
            )
            .ok_or("Numeric Arena Search bound overflow")?;
    }
    Ok(BorrowedArenaWorkBounds {
        planned_games,
        max_callbacks,
        max_public_policy_candidate_rows: max_callbacks * MAX_ROWS_PER_CALLBACK as u64,
        max_search_atomic_steps,
    })
}

fn choose(
    index: usize,
    observation: &Observation,
    roster: &[&mut Role<'_>],
    guards: &mut [Option<GuardedDeployment>],
    conditional_sessions: &mut [Option<GameSession>],
    diagnostics: &mut [Option<GuardDiagnostics>],
    search: &mut [Option<SearchSummary>],
) -> Result<Decision, String> {
    for (i, guard) in guards.iter_mut().enumerate() {
        if let Some(guard) = guard {
            let d = diagnostics[i].as_mut().expect("numeric diagnostic");
            d.notifications_attempted += 1;
            guard.on_callback(index, observation)?;
            d.notifications_returned += 1;
        }
    }
    for (i, session) in conditional_sessions.iter_mut().enumerate() {
        if let Some(session) = session {
            let d = diagnostics[i].as_mut().expect("conditional diagnostic");
            d.notifications_attempted += 1;
            session.on_callback(index, observation)?;
            d.notifications_returned += 1;
        }
    }
    let actor = observation.actor;
    let decision = if let Some(guard) = guards.get_mut(actor).and_then(Option::as_mut) {
        if observation.legal_actions.len() > MAX_ROWS_PER_CALLBACK {
            return Err("Numeric Arena candidate-row cap exceeded before prediction".into());
        }
        let d = diagnostics[actor].as_mut().expect("numeric diagnostic");
        d.choices_attempted += 1;
        d.candidate_rows_attempted += observation.legal_actions.len();
        let result = guard.choose_move(index, observation)?;
        d.choices_returned += 1;
        d.candidate_rows_returned += observation.legal_actions.len();
        if let Some(trade) = result.trade {
            d.forced_exits += usize::from(trade.overridden);
            d.repeat_exits += usize::from(trade.overridden && trade.repeat_seen);
            d.budget_exits += usize::from(trade.overridden && trade.budget_reached);
        }
        let decision = result.decision;
        Decision {
            actor: decision.actor,
            observation_key: decision.observation_key,
            r#move: decision.r#move,
            policy_version: decision.policy_version,
            score: decision.score,
        }
    } else if let Some(session) = conditional_sessions.get_mut(actor).and_then(Option::as_mut) {
        let role = &**roster
            .get(actor)
            .ok_or("Conditional Arena actor outside roster")?;
        let d = diagnostics[actor].as_mut().expect("conditional diagnostic");
        d.choices_attempted += 1;
        d.candidate_rows_attempted += observation.legal_actions.len();
        if let Some(routes) = d.phase_routes.as_mut() {
            match observation.phase {
                Phase::Setup => routes.setup_choices_attempted += 1,
                Phase::Playing => routes.playing_choices_attempted += 1,
                _ => {}
            }
        }
        let result = match role {
            Role::Conditional(policy) => session.choose_move(index, observation, policy)?,
            Role::ConditionalMixed(policy) => session.choose_mixed(index, observation, policy)?,
            Role::ConditionalCountZero(policy) => {
                session.choose_count_zero(index, observation, policy)?
            }
            Role::Teacher(policy) => session.choose_teacher(index, observation, policy)?,
            Role::SemanticSetupTeacher(policy) => {
                session.choose_semantic_setup_teacher(index, observation, policy)?
            }
            Role::PhaseRouted(policy) => session.choose_phase_routed(index, observation, policy)?,
            _ => return Err("Conditional Arena game session/role mismatch".into()),
        };
        d.choices_returned += 1;
        d.candidate_rows_returned += observation.legal_actions.len();
        if let Some(routes) = d.phase_routes.as_mut() {
            match observation.phase {
                Phase::Setup => routes.setup_choices_returned += 1,
                Phase::Playing => routes.playing_choices_returned += 1,
                _ => unreachable!("phase-routed session returned an unsupported phase"),
            }
        }
        if let Some(trade) = result.trade {
            d.forced_exits += usize::from(trade.overridden);
            d.repeat_exits += usize::from(trade.overridden && trade.repeat_seen);
            d.budget_exits += usize::from(trade.overridden && trade.budget_reached);
        }
        if let Some(conditional) = d.conditional.as_mut() {
            if observation.phase == Phase::Setup {
                conditional.setup_choices += 1;
            } else {
                conditional.playing_choices += 1;
                let action = observation
                    .legal_actions
                    .iter()
                    .find(|a| a.r#move == result.decision.r#move)
                    .expect("conditional session returned a legal row");
                let kind = serde_json::to_value(&action.action).map_err(|e| e.to_string())?;
                let kind = kind["type"]
                    .as_str()
                    .ok_or("Conditional action kind missing")?;
                *conditional
                    .selected_action_kinds
                    .entry(kind.into())
                    .or_default() += 1;
                let in_origin_grammar = match role {
                    Role::Conditional(policy) => policy.teacher_grammar_contains(&action.action),
                    Role::ConditionalMixed(policy) => {
                        policy.teacher_grammar_contains(&action.action)
                    }
                    Role::ConditionalCountZero(policy) => {
                        policy.teacher_grammar_contains(&action.action)
                    }
                    _ => unreachable!("conditional session requires a conditional role"),
                };
                conditional.outside_teacher_grammar_choices += usize::from(!in_origin_grammar);
            }
        }
        result.decision
    } else {
        let role = &**roster
            .get(actor)
            .ok_or("Numeric Arena actor outside roster")?;
        let (decision, trace) = match role {
            Role::Reactive(policy) => policy.choose_with_diagnostics(observation)?,
            Role::ResourceCycle(configuration) => (
                super::resource_cycle::choose_move(observation, configuration)?,
                None,
            ),
            Role::Numeric(_) => return Err("Numeric Arena missing game guard".into()),
            Role::Conditional(_) => return Err("Conditional Arena missing game session".into()),
            Role::ConditionalMixed(_) => {
                return Err("Conditional Arena missing game session".into());
            }
            Role::ConditionalCountZero(_) => {
                return Err("Conditional Arena missing game session".into());
            }
            Role::Teacher(_) => return Err("Teacher Arena missing game session".into()),
            Role::SemanticSetupTeacher(_) => {
                return Err("Semantic Setup teacher Arena missing game session".into());
            }
            Role::PhaseRouted(_) => {
                return Err("Phase-routed Arena missing game session".into());
            }
        };
        match (roster[actor].search(), trace) {
            (Some(prepared), Some(trace)) => {
                search[actor]
                    .as_mut()
                    .expect("Search diagnostic")
                    .add(&trace);
                trace.validate(observation, prepared.config(), prepared.configuration_key())?;
                if decision.policy_version != trace.policy_version || decision.score != trace.score
                {
                    return Err("Numeric Arena Search decision/trace mismatch".into());
                }
            }
            (None, None) => {}
            _ => return Err("Numeric Arena Search diagnostic/role mismatch".into()),
        }
        decision
    };
    if !decision.score.is_finite()
        || observation
            .legal_actions
            .iter()
            .filter(|a| a.r#move == decision.r#move)
            .count()
            != 1
    {
        return Err("Numeric Arena requires one legal finite decision".into());
    }
    Ok(decision)
}

fn play_arm<'a>(
    seed: u32,
    seat: usize,
    focal: &mut Role<'a>,
    pool: &mut [Role<'a>],
) -> (ArmResult, Vec<Option<GuardDiagnostics>>) {
    play_arm_using(seed, seat, focal, pool, |players, seed, decide| {
        replay::play_unrecorded_inference(players, seed, decide)
    })
}

type ArmDecide<'a> = dyn FnMut(usize, &Observation) -> Result<Decision, String> + 'a;
fn play_arm_using<'a>(
    seed: u32,
    seat: usize,
    focal: &mut Role<'a>,
    pool: &mut [Role<'a>],
    play: impl FnOnce(
        usize,
        u32,
        &mut ArmDecide<'_>,
    ) -> Result<(tzolkin_core::GameState, usize), String>,
) -> (ArmResult, Vec<Option<GuardDiagnostics>>) {
    let started = Instant::now();
    let players = pool.len();
    let mut focal = Some(focal);
    let mut roster = pool
        .iter_mut()
        .enumerate()
        .map(|(i, role)| {
            if i == seat {
                focal.take().expect("one focal seat")
            } else {
                role
            }
        })
        .collect::<Vec<_>>();
    let mut guards = roster
        .iter_mut()
        .map(|role| match role {
            Role::Numeric(policy) => Some(
                GuardedDeployment::new(
                    policy
                        .loaded
                        .take()
                        .expect("immutable model returned after each arm"),
                    policy.identity.guard.clone(),
                )
                .expect("constructor already validated the immutable guard configuration"),
            ),
            _ => None,
        })
        .collect::<Vec<_>>();
    let mut conditional_sessions = roster
        .iter()
        .map(|role| match &**role {
            Role::Conditional(policy) => Some(GameSession::new(policy)),
            Role::ConditionalMixed(policy) => Some(GameSession::new_mixed(policy)),
            Role::ConditionalCountZero(policy) => Some(GameSession::new_count_zero(policy)),
            Role::Teacher(policy) => Some(GameSession::new_teacher(policy)),
            Role::SemanticSetupTeacher(policy) => {
                Some(GameSession::new_semantic_setup_teacher(policy))
            }
            Role::PhaseRouted(policy) => Some(GameSession::new_phase_routed(policy)),
            _ => None,
        })
        .collect::<Vec<_>>();
    let mut diagnostics = roster
        .iter()
        .map(|role| match &**role {
            Role::Numeric(_) | Role::Teacher(_) | Role::SemanticSetupTeacher(_) => {
                Some(GuardDiagnostics::default())
            }
            Role::Conditional(_) | Role::ConditionalMixed(_) | Role::ConditionalCountZero(_) => {
                Some(GuardDiagnostics {
                    conditional: Some(ConditionalDiagnostics::default()),
                    ..GuardDiagnostics::default()
                })
            }
            Role::PhaseRouted(_) => Some(GuardDiagnostics {
                phase_routes: Some(PhaseRouteDiagnostics::default()),
                ..GuardDiagnostics::default()
            }),
            _ => None,
        })
        .collect::<Vec<_>>();
    let mut search = roster
        .iter()
        .map(|r| r.search().map(|_| SearchSummary::default()))
        .collect::<Vec<_>>();
    let mut start_of_play = None;
    let played = play(players, seed, &mut |index, observation| {
        if start_of_play.is_none() && observation.phase == Phase::Playing {
            start_of_play = Some(InitialConditions::from_observation(observation));
        }
        choose(
            index,
            observation,
            &roster,
            &mut guards,
            &mut conditional_sessions,
            &mut diagnostics,
            &mut search,
        )
    })
    .and_then(|(state, decisions)| {
        let public = observe(&state, state.current_player)?.players;
        Ok((state, decisions, public))
    });
    // Always discard the game history, also after a policy or core error. Only
    // move the same immutable weights back; never reload or copy parameters.
    for (role, guard) in roster.iter_mut().zip(guards) {
        if let (Role::Numeric(policy), Some(guard)) = (role, guard) {
            policy.loaded = Some(guard.into_policy());
        }
    }
    // Conditional policies borrow immutable parameters. Their sessions have no
    // weight ownership to return, and every arm drops all callback/Trade state.
    drop(conditional_sessions);
    let mut result = ArmResult {
        search,
        trade_guard: None,
        decisions: None,
        elapsed_ms: started.elapsed().as_secs_f64() * 1000.0,
        winner_utility: None,
        score: None,
        rank: None,
        unfed_workers: None,
        final_scores: Vec::new(),
        terminal_players: None,
        start_of_play,
        error: None,
    };
    match played {
        Ok((state, decisions, terminal_players)) => {
            let winners = state.final_scores.iter().filter(|s| s.rank == 1).count();
            let score = state
                .final_scores
                .iter()
                .find(|s| s.player_id == seat)
                .expect("validated terminal seat");
            result.winner_utility = Some(if score.rank == 1 {
                1.0 / winners as f64
            } else {
                0.0
            });
            result.score = Some(score.total);
            result.rank = Some(score.rank);
            result.decisions = Some(decisions);
            result.final_scores = state.final_scores;
            result.terminal_players = Some(terminal_players);
            for d in diagnostics.iter_mut().flatten() {
                d.applied_choices = Some(d.choices_returned);
            }
        }
        Err(error) => result.error = Some(error),
    }
    (result, diagnostics)
}

/// Development Validation only. All seats share each seed block, the absolute
/// opponent assignments stay fixed, and any failed arm invalidates its block.
pub fn run_validation(
    config: &NumericArenaConfig,
    candidate: NumericPolicy,
    reference: NumericPolicy,
    opponent_pool: Vec<Opponent<'_>>,
) -> Result<NumericArenaReport, String> {
    run_roles(
        config,
        Role::Numeric(candidate),
        Role::Numeric(reference),
        opponent_pool,
        SCHEMA,
    )
}

/// Frozen ResourceCycle V5 candidate paired against a preloaded numeric
/// reference through the same all-seat Development Validation simulation.
/// The candidate has no Trade guard; numeric roles keep their shared guard.
/// This grants no training, source/native qualification or strength authority.
pub fn run_resource_cycle_validation<'a>(
    config: &NumericArenaConfig,
    candidate: super::resource_cycle::ResourceCycleConfig,
    reference: NumericPolicy,
    opponent_pool: Vec<Opponent<'a>>,
) -> Result<NumericArenaReport, String> {
    if candidate != super::resource_cycle::ResourceCycleConfig::v5() {
        return Err("Resource-cycle candidate requires the exact frozen V5 configuration".into());
    }
    run_roles(
        config,
        Role::ResourceCycle(candidate),
        Role::Numeric(reference),
        opponent_pool,
        RESOURCE_CYCLE_ARENA_SCHEMA,
    )
}

/// Audited learned V5 teacher, using the same paired all-seat callback/Trade
/// engine. Source families from every partition are excluded locally. This is
/// Development Validation, without native, training or fresh Test authority.
pub fn run_resource_cycle_teacher_validation<'a>(
    config: &NumericArenaConfig,
    candidate: ResourceCycleTeacherArenaPolicy<'a>,
    reference: NumericPolicy,
    opponent_pool: Vec<Opponent<'a>>,
) -> Result<NumericArenaReport, String> {
    candidate.validate_arena_config(config)?;
    let prepared = candidate.prepared;
    let result = run_roles(
        config,
        Role::Teacher(Box::new(candidate)),
        Role::Numeric(reference),
        opponent_pool,
        super::teacher::ARENA_SCHEMA,
    );
    // Also audit after an attempted run returning an error. No report may be
    // published with a changed retained manifest, copied replay or shard.
    prepared.recheck_sources()?;
    result
}

/// Schema-3 teacher candidate through the existing paired all-seat engine.
/// The reference retains its own validated profile and guard. This is known
/// Development Validation, without training, native or fresh Test authority.
pub fn run_semantic_setup_resource_cycle_teacher_validation<'a>(
    config: &NumericArenaConfig,
    candidate: SemanticSetupResourceCycleTeacherArenaPolicy<'a>,
    reference: SemanticSetupTeacherReference<'a>,
    opponent_pool: Vec<Opponent<'a>>,
) -> Result<NumericArenaReport, String> {
    candidate.validate_arena_config(config)?;
    let prepared = candidate.prepared;
    let reference = Role::from_semantic_setup_reference(reference);
    let result = run_roles(
        config,
        Role::SemanticSetupTeacher(Box::new(candidate)),
        reference,
        opponent_pool,
        super::teacher::SEMANTIC_SETUP_ARENA_SCHEMA,
    );
    prepared.recheck_sources()?;
    result
}

/// Setup-only semantic teacher followed by the unchanged numeric reference's
/// Playing predictor. The outer identity is a distinct Development experiment;
/// it cannot enter either all-phase teacher or numeric deployment entry point.
pub fn run_phase_routed_development_validation<'a>(
    config: &NumericArenaConfig,
    candidate: PhaseRoutedDevelopmentPolicy<'a>,
    reference: NumericPolicy,
    opponent_pool: Vec<Opponent<'a>>,
) -> Result<NumericArenaReport, String> {
    candidate.validate_arena_config(config)?;
    candidate.validate_reference(&reference)?;
    let prepared = candidate.setup.prepared;
    let result = run_roles(
        config,
        Role::PhaseRouted(Box::new(candidate)),
        Role::Numeric(reference),
        opponent_pool,
        super::phase_routed::ARENA_SCHEMA,
    );
    prepared.recheck_sources()?;
    result
}

/// Conditional Human Playing with explicit Corn-first Setup, paired against a
/// preloaded numeric reference. This grants no native or training qualification.
pub fn run_conditional_validation<'a>(
    config: &NumericArenaConfig,
    candidate: ConditionalPolicy<'a>,
    reference: NumericPolicy,
    opponent_pool: Vec<Opponent<'a>>,
) -> Result<NumericArenaReport, String> {
    run_roles(
        config,
        Role::Conditional(Box::new(candidate)),
        Role::Numeric(reference),
        opponent_pool,
        super::conditional::ARENA_SCHEMA,
    )
}

/// Genuine updated conditional Human actor, predicting both native Setup and
/// Playing, paired against an immutable numeric reference with the same guard.
/// This grants no training/source qualification, Ready proof or fresh Test.
pub fn run_conditional_mixed_validation<'a>(
    config: &NumericArenaConfig,
    candidate: ConditionalMixedArenaPolicy<'a>,
    reference: NumericPolicy,
    opponent_pool: Vec<Opponent<'a>>,
) -> Result<NumericArenaReport, String> {
    run_roles(
        config,
        Role::ConditionalMixed(Box::new(candidate)),
        Role::Numeric(reference),
        opponent_pool,
        super::conditional::MIXED_ARENA_SCHEMA,
    )
}

/// Genuine fresh BC count0 control with the same Setup/Playing prediction,
/// callback ownership and Trade guard as the updated conditional actor.
/// Development only; this does not relax the positive-current adapter or grant
/// learning, native/source qualification, value supervision or fresh Test.
pub fn run_conditional_count_zero_validation<'a>(
    config: &NumericArenaConfig,
    candidate: ConditionalCountZeroArenaPolicy<'a>,
    reference: NumericPolicy,
    opponent_pool: Vec<Opponent<'a>>,
) -> Result<NumericArenaReport, String> {
    run_roles(
        config,
        Role::ConditionalCountZero(Box::new(candidate)),
        Role::Numeric(reference),
        opponent_pool,
        super::conditional::COUNT_ZERO_ARENA_SCHEMA,
    )
}

fn run_roles<'a>(
    config: &NumericArenaConfig,
    mut candidate: Role<'a>,
    mut reference: Role<'a>,
    opponent_pool: Vec<Opponent<'a>>,
    schema: &str,
) -> Result<NumericArenaReport, String> {
    config.validate()?;
    if opponent_pool.len() != config.players {
        return Err("Numeric Arena absolute-seat pool count mismatch".into());
    }
    let mut pool = opponent_pool
        .into_iter()
        .map(Role::from_opponent)
        .collect::<Result<Vec<_>, _>>()?;
    shared_guard(&candidate, &reference, &pool)?;
    let identities = (
        candidate.identity(),
        reference.identity(),
        pool.iter().map(Role::identity).collect::<Vec<_>>(),
    );
    let configuration_sha256 = format!(
        "{:x}",
        Sha256::digest(serde_json::to_vec(&(config, &identities)).map_err(|e| e.to_string())?)
    );
    let work_bounds = work_bounds(config, &pool)?;
    let mut blocks = Vec::with_capacity(config.seeds.len());
    let mut guards = Vec::with_capacity(config.seeds.len());
    for (index, seed) in config.seeds.iter().enumerate() {
        let mut pairs = Vec::with_capacity(config.players);
        let mut guard_pairs = Vec::with_capacity(config.players);
        for seat in 0..config.players {
            let (c, r) = if (index + seat) % 2 == 0 {
                (
                    play_arm(*seed, seat, &mut candidate, &mut pool),
                    play_arm(*seed, seat, &mut reference, &mut pool),
                )
            } else {
                let reference_result = play_arm(*seed, seat, &mut reference, &mut pool);
                (
                    play_arm(*seed, seat, &mut candidate, &mut pool),
                    reference_result,
                )
            };
            pairs.push(SeatPair {
                seat,
                candidate: c.0,
                reference: r.0,
            });
            guard_pairs.push(GuardPair {
                seat,
                candidate: c.1,
                reference: r.1,
            });
        }
        blocks.push(block(*seed, pairs));
        guards.push(GuardBlock {
            seed: *seed,
            pairs: guard_pairs,
        });
    }
    Ok(NumericArenaReport {
        schema: schema.into(),
        config: config.clone(),
        options: GameOptions::default(),
        configuration_sha256,
        candidate: identities.0,
        reference: identities.1,
        opponent_pool: identities.2,
        work_bounds,
        statistics: summarize(&blocks, config.bootstrap_seed),
        blocks,
        guards,
        development: true,
        fresh_holdout_claimed: false,
        producer_authenticated: false,
        training_admission_granted: false,
        rules_baseline: replay::RULES_BASELINE.into(),
        catalog_hash: crate::replay::catalog_hash(),
        target_arch: std::env::consts::ARCH.into(),
        target_os: std::env::consts::OS.into(),
    })
}

fn shared_guard(
    candidate: &Role<'_>,
    reference: &Role<'_>,
    pool: &[Role<'_>],
) -> Result<(), String> {
    let guard_key = if matches!(candidate, Role::ResourceCycle(_)) {
        reference.guard_key().expect("guarded numeric reference")
    } else {
        candidate.guard_key().expect("guarded focal role")
    };
    if Some(guard_key) != reference.guard_key()
        || pool
            .iter()
            .any(|role| role.guard_key().is_some_and(|key| key != guard_key))
    {
        return Err("Numeric Arena requires one fixed shared guard configuration".into());
    }
    Ok(())
}

#[cfg(test)]
pub(super) mod tests {
    use super::*;
    use serde_json::json;
    use tzolkin_inference::{deployment as d, features, policy};

    pub(crate) fn synthetic_policy() -> NumericPolicy {
        let mut wire = json!({
            "schema": d::DEPLOYMENT_SCHEMA,
            "contract": {
                "modelVersion": d::MODEL_VERSION, "inputContract": d::INPUT_CONTRACT,
                "featureSchema": features::PUBLIC_FEATURE_SCHEMA,
                "featureCount": features::FEATURE_COUNT, "hiddenCount": policy::HIDDEN,
                "parameterCount": policy::PARAMETER_COUNT, "rulesVersion": d::RULES_VERSION,
                "rulesBaseline": d::RULES_BASELINE,
                "observationSchema": tzolkin_core::observation::OBSERVATION_SCHEMA,
                "moveSchema": tzolkin_core::observation::MOVE_SCHEMA,
                "catalogHash": d::catalog_hash(), "backend": "scalar",
                "numericContract": d::NUMERIC_CONTRACT, "scope": d::SCOPE,
                "valueValidity": d::VALUE_VALIDITY
            },
            "source": {"kind": "bcPrepared", "policyVersion": "learned-public-policy-v1",
                "task": "policyOnlyBc", "modelChecksum": "a".repeat(64),
                "trainingCheckpointChecksum": "b".repeat(64)},
            "sourceNumericalTarget": "scalar-f32-f64-tick53-v1:x86_64:windows:64",
            "parameterBits": vec!["00000000"; policy::PARAMETER_COUNT], "checksum": ""
        });
        wire["checksum"] = d::numeric_checksum(&serde_json::to_vec(&wire).unwrap())
            .unwrap()
            .into();
        NumericPolicy::new(
            LoadedDeployment::load(&serde_json::to_vec(&wire).unwrap()).unwrap(),
            TradeGuardConfig::default(),
        )
        .unwrap()
    }

    pub(crate) fn synthetic_guard() -> GuardedDeployment {
        let mut policy = synthetic_policy();
        GuardedDeployment::new(policy.loaded.take().unwrap(), TradeGuardConfig::default()).unwrap()
    }

    #[test]
    fn phase_routed_numeric_pair_and_borrow_reject_mismatches() {
        use super::super::phase_routed::validate_numeric_pair;
        let playing = synthetic_policy();
        let mut reference = synthetic_policy();
        validate_numeric_pair(&playing, &reference).unwrap();
        for field in ["checksum", "backend", "update", "guard"] {
            let mut wrong = synthetic_policy();
            match field {
                "checksum" => wrong.identity.deployment_checksum = "c".repeat(64),
                "backend" => wrong.identity.backend = "avx2".into(),
                "update" => wrong.identity.claimed_update_count = Some(1),
                "guard" => wrong.identity.guard.max_trades_per_episode = 1,
                _ => unreachable!(),
            }
            assert!(validate_numeric_pair(&playing, &wrong).is_err());
        }
        reference.loaded = None;
        assert!(validate_numeric_pair(&playing, &reference).is_err());
        let state =
            tzolkin_core::create_game(vec!["A".into(), "B".into(), "C".into()], 17, false).unwrap();
        let mut o = observe(&state, state.current_player).unwrap();
        o.legal_actions = vec![o.legal_actions[0].clone(); MAX_ROWS_PER_CALLBACK + 1];
        assert_eq!(
            playing.distribution(&o).err().unwrap(),
            "Numeric Arena candidate-row cap exceeded before prediction"
        );
        let legacy = serde_json::to_value(GuardDiagnostics::default()).unwrap();
        assert!(legacy.get("phaseRoutes").is_none());
    }

    #[test]
    fn numeric_validation_preflight_and_failed_arm_discard_only_game_history() {
        let numeric_reference = Role::from_semantic_setup_reference(
            SemanticSetupTeacherReference::GuardedNumeric(Box::new(synthetic_policy())),
        );
        assert_eq!(
            serde_json::to_value(numeric_reference.identity()).unwrap(),
            serde_json::to_value(Role::Numeric(synthetic_policy()).identity()).unwrap()
        );
        assert!(matches!(numeric_reference, Role::Numeric(_)));
        let config = NumericArenaConfig {
            schema: SCHEMA.into(),
            players: 3,
            seeds: vec![3],
            bootstrap_seed: 7,
            external_exclusions: ExternalExclusions {
                evidence_sha256: "e".repeat(64),
                excluded_family_ids: vec![],
            },
        };
        config.validate().unwrap();
        for seeds in [vec![0], vec![10], vec![3, 3]] {
            let mut invalid = config.clone();
            invalid.seeds = seeds;
            assert!(invalid.validate().is_err());
        }
        let mut excluded = config.clone();
        excluded
            .external_exclusions
            .excluded_family_ids
            .push(family_id(3));
        assert!(excluded.validate().is_err());
        let mut unknown = serde_json::to_value(&config).unwrap();
        unknown["allowTest"] = true.into();
        assert!(serde_json::from_value::<NumericArenaConfig>(unknown).is_err());
        assert!(
            serde_json::to_value(GuardDiagnostics::default())
                .unwrap()
                .get("conditional")
                .is_none()
        );
        let mut other_guard = synthetic_policy();
        other_guard.identity.guard.max_trades_per_episode = 1;
        other_guard.identity.configuration_key =
            other_guard.identity.guard.configuration_key().unwrap();
        let refusal = run_validation(
            &config,
            synthetic_policy(),
            other_guard,
            (0..3)
                .map(|_| Opponent::GuardedNumeric(Box::new(synthetic_policy())))
                .collect(),
        )
        .err()
        .unwrap();
        assert_eq!(
            refusal,
            "Numeric Arena requires one fixed shared guard configuration"
        );

        let mut focal = Role::Numeric(synthetic_policy());
        let mut pool = (0..3)
            .map(|_| Role::Numeric(synthetic_policy()))
            .collect::<Vec<_>>();
        let identities = (
            focal.identity(),
            pool.iter().map(Role::identity).collect::<Vec<_>>(),
        );
        let bounds = work_bounds(&config, &pool).unwrap();
        assert_eq!(
            (
                bounds.planned_games,
                bounds.max_callbacks,
                bounds.max_public_policy_candidate_rows
            ),
            (6, 24_000, 24_576_000)
        );
        assert_eq!(bounds.max_search_atomic_steps, 0);
        let mut results = Vec::new();
        // A partial arm and a wrong-actor refusal in the real simulator. The
        // second arm starts at zero: carrying the old guard would fail first.
        for (wrong_actor, callbacks, expected_error) in [
            (false, 2, "Injected downstream core failure"),
            (true, 1, "Policy returned stale or wrong-actor decision"),
        ] {
            let (result, diagnostics) =
                play_arm_using(3, 1, &mut focal, &mut pool, |players, seed, decide| {
                    if wrong_actor {
                        return replay::play_unrecorded_inference(players, seed, |index, o| {
                            let mut decision = decide(index, o)?;
                            decision.actor = (decision.actor + 1) % players;
                            Ok(decision)
                        });
                    }
                    let mut state = tzolkin_core::create_game(
                        (0..players).map(|i| format!("Seat {i}")).collect(),
                        seed,
                        false,
                    )?;
                    for index in 0..2 {
                        let o = observe(&state, state.current_player)?;
                        let decision = decide(index, &o)?;
                        state = tzolkin_core::apply_move(&state, decision.r#move)?;
                    }
                    Err("Injected downstream core failure".into())
                });
            assert_eq!(result.error.as_deref(), Some(expected_error));
            assert!(result.decisions.is_none() && result.score.is_none() && result.rank.is_none());
            assert!(
                diagnostics
                    .iter()
                    .flatten()
                    .all(|d| d.notifications_attempted == callbacks
                        && d.notifications_returned == callbacks
                        && d.applied_choices.is_none())
            );
            assert_eq!(
                diagnostics
                    .iter()
                    .flatten()
                    .map(|d| d.choices_returned)
                    .sum::<usize>(),
                callbacks
            );
            assert_eq!(
                serde_json::to_value((
                    focal.identity(),
                    pool.iter().map(Role::identity).collect::<Vec<_>>()
                ))
                .unwrap(),
                serde_json::to_value(&identities).unwrap()
            );
            assert!(matches!(&focal, Role::Numeric(p) if p.loaded.is_some()));
            assert!(
                pool.iter()
                    .all(|r| matches!(r, Role::Numeric(p) if p.loaded.is_some()))
            );
            results.push(result);
        }
        let mut results = results.into_iter();
        let failed = block(
            3,
            vec![SeatPair {
                seat: 1,
                candidate: results.next().unwrap(),
                reference: results.next().unwrap(),
            }],
        );
        assert!(!failed.complete && failed.mean_utility_delta.is_none());
        let stats = summarize(&[failed], 7);
        assert_eq!((stats.complete_blocks, stats.failed_games), (0, 2));
        assert!(stats.failures_preclude_adoption && !stats.strength_improvement_declared);

        let frozen = super::super::resource_cycle::ResourceCycleConfig::v5();
        for candidate in [
            super::super::resource_cycle::ResourceCycleConfig::default(),
            super::super::resource_cycle::ResourceCycleConfig {
                harvest_before_day: 26,
                ..frozen.clone()
            },
        ] {
            assert_eq!(
                run_resource_cycle_validation(&config, candidate, synthetic_policy(), vec![])
                    .err()
                    .unwrap(),
                "Resource-cycle candidate requires the exact frozen V5 configuration"
            );
        }
        assert!(
            run_resource_cycle_validation(&excluded, frozen.clone(), synthetic_policy(), vec![])
                .is_err()
        );
        assert_eq!(
            run_resource_cycle_validation(&config, frozen.clone(), synthetic_policy(), vec![])
                .err()
                .unwrap(),
            "Numeric Arena absolute-seat pool count mismatch"
        );
        let mut teacher = Role::ResourceCycle(frozen.clone());
        let reference = Role::Numeric(synthetic_policy());
        let mut teacher_pool = (0..3)
            .map(|_| Role::ResourceCycle(frozen.clone()))
            .collect::<Vec<_>>();
        assert!(teacher.guard_key().is_none());
        shared_guard(&teacher, &reference, &teacher_pool).unwrap();
        let mut wrong_pool_guard = synthetic_policy();
        wrong_pool_guard.identity.configuration_key = "different guard".into();
        assert!(shared_guard(&teacher, &reference, &[Role::Numeric(wrong_pool_guard)]).is_err());
        let teacher_identity = serde_json::to_value(teacher.identity()).unwrap();
        assert_eq!(teacher_identity["kind"], "resourceCycle");
        assert_eq!(
            teacher_identity["configuration"],
            serde_json::to_value(&frozen).unwrap()
        );
        assert_ne!(RESOURCE_CYCLE_ARENA_SCHEMA, SCHEMA);
        let bounds = work_bounds(&config, &teacher_pool).unwrap();
        assert_eq!((bounds.planned_games, bounds.max_callbacks), (6, 24_000));
        assert_eq!(bounds.max_search_atomic_steps, 0);
        let (arm, diagnostics) = play_arm_using(
            3,
            1,
            &mut teacher,
            &mut teacher_pool,
            |players, seed, decide| {
                let mut state = tzolkin_core::create_game(
                    (0..players).map(|i| format!("Seat {i}")).collect(),
                    seed,
                    false,
                )?;
                for index in 0..2 {
                    let o = observe(&state, state.current_player)?;
                    let decision = decide(index, &o)?;
                    state = tzolkin_core::apply_move(&state, decision.r#move)?;
                }
                Err("Injected teacher downstream failure".into())
            },
        );
        assert_eq!(
            arm.error.as_deref(),
            Some("Injected teacher downstream failure")
        );
        assert!(arm.decisions.is_none() && arm.score.is_none() && arm.rank.is_none());
        assert!(diagnostics.iter().all(Option::is_none));
        assert_eq!(
            serde_json::to_value(teacher.identity()).unwrap(),
            teacher_identity
        );

        let zero = crate::conditional_human_rl_update::tests::owner();
        assert!(ConditionalMixedArenaPolicy::new(&zero, TradeGuardConfig::default()).is_err());
        assert_eq!(
            ConditionalCountZeroArenaPolicy::new_bc_control(&zero, TradeGuardConfig::default())
                .err()
                .unwrap(),
            "Count0 BC control requires original loaded source custody"
        );
        let zero_before = serde_json::to_value(zero.identity()).unwrap();
        let zero_actor = zero.actor_parameters().to_vec();
        let zero_critic = zero.critic_parameters().to_vec();
        let mut zero_focal = Role::ConditionalCountZero(Box::new(
            ConditionalCountZeroArenaPolicy::fixture(&zero, TradeGuardConfig::default()).unwrap(),
        ));
        let zero_identity = serde_json::to_value(zero_focal.identity()).unwrap();
        let zero_reference = Role::from_semantic_setup_reference(
            SemanticSetupTeacherReference::ConditionalHumanCountZeroAllPhase(Box::new(
                ConditionalCountZeroArenaPolicy::fixture(&zero, TradeGuardConfig::default())
                    .unwrap(),
            )),
        );
        assert!(matches!(zero_reference, Role::ConditionalCountZero(_)));
        assert_eq!(
            serde_json::to_value(zero_reference.identity()).unwrap(),
            zero_identity
        );
        assert_eq!(zero_reference.guard_key(), zero_focal.guard_key());
        assert_eq!(zero_identity["kind"], "conditionalHumanCountZeroAllPhase");
        assert_eq!(zero_identity["identity"]["countZero"], zero_before);
        assert_eq!(
            zero_identity["identity"]["selection"],
            "orderedFirstMaxSetupPlaying"
        );
        assert_eq!(zero_identity["identity"]["sourceQualified"], false);
        assert_eq!(zero_identity["identity"]["verifiedNativePolicy"], false);
        assert_eq!(
            zero_identity["identity"]["valueValidity"],
            "unavailablePolicyOnly"
        );
        let mut wrong_guard = synthetic_policy();
        wrong_guard.identity.configuration_key = "wrong".into();
        assert_eq!(
            run_conditional_count_zero_validation(
                &config,
                ConditionalCountZeroArenaPolicy::fixture(&zero, TradeGuardConfig::default())
                    .unwrap(),
                wrong_guard,
                (0..3)
                    .map(|_| Opponent::ResourceCycle(
                        super::super::resource_cycle::ResourceCycleConfig::v5()
                    ))
                    .collect(),
            )
            .err()
            .unwrap(),
            "Numeric Arena requires one fixed shared guard configuration"
        );
        for phase in [Phase::Setup, Phase::Playing, Phase::Playing] {
            let (arm, diagnostics) =
                play_arm_using(3, 0, &mut zero_focal, &mut pool, |players, seed, decide| {
                    let state = if phase == Phase::Setup {
                        tzolkin_core::create_game(
                            (0..players).map(|i| format!("Seat {i}")).collect(),
                            seed,
                            false,
                        )?
                    } else {
                        crate::conditional_human_rl_update::tests::terminal_start(players, seed)
                    };
                    let o = observe(&state, state.current_player)?;
                    let prediction = zero.handle()?.distribution(&o)?;
                    let expected = (1..prediction.logits().len()).fold(0, |best, i| {
                        if prediction.logits()[i] > prediction.logits()[best] {
                            i
                        } else {
                            best
                        }
                    });
                    let decision = decide(0, &o)?;
                    assert_eq!(decision.r#move, o.legal_actions[expected].r#move);
                    assert_eq!(decision.score, f64::from(prediction.logits()[expected]));
                    assert_eq!(
                        decision.policy_version,
                        super::super::conditional::COUNT_ZERO_POLICY_VERSION
                    );
                    Err("Injected count0 downstream failure".into())
                });
            assert_eq!(
                arm.error.as_deref(),
                Some("Injected count0 downstream failure")
            );
            assert!(arm.decisions.is_none() && arm.score.is_none() && arm.rank.is_none());
            assert!(
                diagnostics
                    .iter()
                    .flatten()
                    .all(|d| d.notifications_attempted == 1
                        && d.notifications_returned == 1
                        && d.applied_choices.is_none())
            );
            let d = diagnostics[0]
                .as_ref()
                .unwrap()
                .conditional
                .as_ref()
                .unwrap();
            assert_eq!(d.setup_choices, usize::from(phase == Phase::Setup));
            assert_eq!(d.playing_choices, usize::from(phase == Phase::Playing));
            assert_eq!(
                serde_json::to_value(zero_focal.identity()).unwrap(),
                zero_identity
            );
        }
        assert_eq!(serde_json::to_value(zero.identity()).unwrap(), zero_before);
        assert_eq!(zero.actor_parameters(), zero_actor);
        assert_eq!(zero.critic_parameters(), zero_critic);
        let updated = crate::conditional_human_rl_update::tests::updated_owner();
        assert!(
            ConditionalCountZeroArenaPolicy::fixture(&updated, TradeGuardConfig::default())
                .is_err()
        );
        let actor_bits = updated
            .actor_parameters()
            .iter()
            .map(|x| x.to_bits())
            .collect::<Vec<_>>();
        let critic_bits = updated
            .critic_parameters()
            .iter()
            .map(|x| x.to_bits())
            .collect::<Vec<_>>();
        let policy =
            ConditionalMixedArenaPolicy::new(&updated, TradeGuardConfig::default()).unwrap();
        assert_eq!(policy.identity().current, *updated.identity());
        assert_eq!(policy.identity().dataset, *updated.origin().dataset());
        assert_eq!(
            policy.identity().reservations,
            *updated.origin().table_reservations()
        );
        assert!(!policy.identity().source_qualified && !policy.identity().verified_native_policy);
        let mut mismatched_reference = synthetic_policy();
        mismatched_reference.identity.guard.max_trades_per_episode = 1;
        mismatched_reference.identity.configuration_key = mismatched_reference
            .identity
            .guard
            .configuration_key()
            .unwrap();
        assert_eq!(
            run_conditional_mixed_validation(
                &config,
                policy,
                mismatched_reference,
                (0..3)
                    .map(|_| Opponent::ResourceCycle(
                        super::super::resource_cycle::ResourceCycleConfig::v5()
                    ))
                    .collect(),
            )
            .err()
            .unwrap(),
            "Numeric Arena requires one fixed shared guard configuration"
        );
        let mut focal = Role::ConditionalMixed(Box::new(
            ConditionalMixedArenaPolicy::new(&updated, TradeGuardConfig::default()).unwrap(),
        ));
        let identity = serde_json::to_value(focal.identity()).unwrap();
        assert_eq!(identity["kind"], "conditionalHumanMixedRl");
        assert_eq!(identity["identity"]["current"]["mixedUpdateCount"], 1);
        for phase in [Phase::Setup, Phase::Playing, Phase::Playing] {
            let (arm, diagnostics) =
                play_arm_using(3, 0, &mut focal, &mut pool, |players, seed, decide| {
                    let state = if phase == Phase::Setup {
                        tzolkin_core::create_game(
                            (0..players).map(|i| format!("Seat {i}")).collect(),
                            seed,
                            false,
                        )?
                    } else {
                        crate::conditional_human_rl_update::tests::terminal_start(players, seed)
                    };
                    let o = observe(&state, state.current_player)?;
                    assert_eq!(o.phase, phase);
                    let prediction = updated.handle()?.distribution(&o)?;
                    let expected = (1..prediction.logits().len()).fold(0, |best, i| {
                        if prediction.logits()[i] > prediction.logits()[best] {
                            i
                        } else {
                            best
                        }
                    });
                    let decision = decide(0, &o)?;
                    assert_eq!(decision.r#move, o.legal_actions[expected].r#move);
                    assert_eq!(
                        decision.policy_version,
                        super::super::conditional::MIXED_POLICY_VERSION
                    );
                    Err("Injected conditional downstream failure".into())
                });
            assert_eq!(
                arm.error.as_deref(),
                Some("Injected conditional downstream failure")
            );
            assert!(arm.decisions.is_none() && arm.score.is_none() && arm.rank.is_none());
            assert!(
                diagnostics
                    .iter()
                    .flatten()
                    .all(|d| d.notifications_attempted == 1
                        && d.notifications_returned == 1
                        && d.applied_choices.is_none())
            );
            let conditional = diagnostics[0]
                .as_ref()
                .unwrap()
                .conditional
                .as_ref()
                .unwrap();
            assert_eq!(
                conditional.setup_choices,
                usize::from(phase == Phase::Setup)
            );
            assert_eq!(
                conditional.playing_choices,
                usize::from(phase == Phase::Playing)
            );
            assert_eq!(serde_json::to_value(focal.identity()).unwrap(), identity);
        }
        assert_eq!(
            updated
                .actor_parameters()
                .iter()
                .map(|x| x.to_bits())
                .collect::<Vec<_>>(),
            actor_bits
        );
        assert_eq!(
            updated
                .critic_parameters()
                .iter()
                .map(|x| x.to_bits())
                .collect::<Vec<_>>(),
            critic_bits
        );
        assert_eq!(updated.identity().mixed_update_count(), 1);
    }
}
