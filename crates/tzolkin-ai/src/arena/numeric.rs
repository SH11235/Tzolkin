//! Inference-only paired evaluation, also reused by explicit conditional origins.
//! Numeric identities and external exclusions do not authenticate a producer,
//! recreate an owner, or grant training, fresh Test, or strength authority.
use super::conditional::{
    ConditionalDiagnostics, ConditionalIdentity, ConditionalPolicy, GameSession,
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
}
enum Role<'a> {
    Numeric(NumericPolicy),
    Conditional(Box<ConditionalPolicy<'a>>),
    Reactive(PolicyHandle<'a>),
    ResourceCycle(super::resource_cycle::ResourceCycleConfig),
}
impl<'a> Role<'a> {
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
        let Role::Conditional(policy) = &**roster
            .get(actor)
            .ok_or("Conditional Arena actor outside roster")?
        else {
            return Err("Conditional Arena game session/role mismatch".into());
        };
        let d = diagnostics[actor].as_mut().expect("conditional diagnostic");
        d.choices_attempted += 1;
        d.candidate_rows_attempted += observation.legal_actions.len();
        let result = session.choose_move(index, observation, policy)?;
        d.choices_returned += 1;
        d.candidate_rows_returned += observation.legal_actions.len();
        if let Some(trade) = result.trade {
            d.forced_exits += usize::from(trade.overridden);
            d.repeat_exits += usize::from(trade.overridden && trade.repeat_seen);
            d.budget_exits += usize::from(trade.overridden && trade.budget_reached);
        }
        let conditional = d
            .conditional
            .as_mut()
            .expect("conditional choice diagnostic");
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
            conditional.outside_teacher_grammar_choices +=
                usize::from(!matches!(kind, "place" | "remove" | "firstPlayer"));
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
            _ => None,
        })
        .collect::<Vec<_>>();
    let mut diagnostics = roster
        .iter()
        .map(|role| match &**role {
            Role::Numeric(_) => Some(GuardDiagnostics::default()),
            Role::Conditional(_) => Some(GuardDiagnostics {
                conditional: Some(ConditionalDiagnostics::default()),
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
    let guard_key = candidate.guard_key().expect("guarded focal role");
    if Some(guard_key) != reference.guard_key()
        || pool
            .iter()
            .any(|role| role.guard_key().is_some_and(|key| key != guard_key))
    {
        return Err("Numeric Arena requires one fixed shared guard configuration".into());
    }
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

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use tzolkin_inference::{deployment as d, features, policy};

    fn synthetic_policy() -> NumericPolicy {
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

    #[test]
    fn numeric_validation_preflight_and_failed_arm_discard_only_game_history() {
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
    }
}
