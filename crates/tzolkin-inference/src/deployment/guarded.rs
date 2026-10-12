//! Explicit deterministic deployment composition using the shared Trade guard.
use serde::Serialize;
use sha2::{Digest, Sha256};
use tzolkin_core::observation::{Observation, observation_key};

use super::{DeploymentDecision, LoadedDeployment};
use crate::policy;
use crate::trade_guard::{
    GUARD_VERSION, PROJECTION_VERSION, TradeGuardConfig, TradeHistory, is_trade, public_bytes,
    skip_index,
};

pub const GUARDED_DEPLOYMENT_VERSION: &str = "numeric-public-trade-guard-v1";
pub const MAX_GAME_CALLBACKS: usize = 4000;

/// One immutable model and a fresh, bounded history for one game. Notify every
/// seat's decision callback, including human and heuristic turns, in order.
/// Choose at most once for the notified callback. A failed callback/choice or a
/// failed downstream core application requires discarding this game session.
/// Loading this numeric model establishes no training or playing-strength claim.
pub struct GuardedDeployment {
    policy: LoadedDeployment,
    game: GuardedGame,
}

/// One game-local guard borrowing an immutable loaded deployment. Multiple
/// views may share its model storage; each view has its own callback and Trade
/// history. Notify every seat's callback to each view, in order, and discard a
/// view after a callback, choice or downstream core application fails.
pub struct BorrowedGuardedDeployment<'a> {
    policy: &'a LoadedDeployment,
    game: GuardedGame,
}

/// Shared callback/Trade lifecycle; immutable model storage stays with each
/// closed deployment type. No serialized state can restore this game history.
pub(crate) struct GuardedGame {
    configuration_key: String,
    history: TradeHistory,
    next_callback: usize,
    notified: Option<(usize, usize, bool, String)>,
    chosen: bool,
    failed: bool,
    actor: Option<usize>,
}

impl GuardedDeployment {
    pub fn new(policy: LoadedDeployment, config: TradeGuardConfig) -> Result<Self, String> {
        Ok(Self {
            policy,
            game: GuardedGame::new(config)?,
        })
    }

    pub fn policy(&self) -> &LoadedDeployment {
        &self.policy
    }

    pub fn into_policy(self) -> LoadedDeployment {
        self.policy
    }

    pub fn configuration_key(&self) -> &str {
        self.game.configuration_key()
    }

    pub fn on_callback(&mut self, index: usize, o: &Observation) -> Result<(), String> {
        self.game.on_callback(index, o)
    }

    pub fn choose_move(
        &mut self,
        index: usize,
        o: &Observation,
    ) -> Result<GuardedDeploymentDecision, String> {
        self.game.choose_move(
            index,
            o,
            || self.policy.distribution(o),
            self.policy.policy_version(),
            self.policy.checksum(),
            GUARDED_DEPLOYMENT_VERSION,
        )
    }
}

impl<'a> BorrowedGuardedDeployment<'a> {
    pub fn new(policy: &'a LoadedDeployment, config: TradeGuardConfig) -> Result<Self, String> {
        Ok(Self {
            policy,
            game: GuardedGame::new(config)?,
        })
    }

    pub fn policy(&self) -> &LoadedDeployment {
        self.policy
    }

    pub fn configuration_key(&self) -> &str {
        self.game.configuration_key()
    }

    pub fn on_callback(&mut self, index: usize, o: &Observation) -> Result<(), String> {
        self.game.on_callback(index, o)
    }

    pub fn choose_move(
        &mut self,
        index: usize,
        o: &Observation,
    ) -> Result<GuardedDeploymentDecision, String> {
        self.game.choose_move(
            index,
            o,
            || self.policy.distribution(o),
            self.policy.policy_version(),
            self.policy.checksum(),
            GUARDED_DEPLOYMENT_VERSION,
        )
    }
}

impl GuardedGame {
    pub(crate) fn new(config: TradeGuardConfig) -> Result<Self, String> {
        let configuration_key = config.configuration_key()?;
        Ok(Self {
            configuration_key,
            history: TradeHistory::new(
                config.max_trades_per_episode,
                config.max_public_state_bytes,
            )?,
            next_callback: 0,
            notified: None,
            chosen: false,
            failed: false,
            actor: None,
        })
    }

    pub(crate) fn configuration_key(&self) -> &str {
        &self.configuration_key
    }

    fn reset_episode(&mut self) {
        self.actor = None;
        self.history.reset();
    }

    fn invalidate(&mut self) {
        self.reset_episode();
        self.failed = true;
    }

    pub(crate) fn on_callback(&mut self, index: usize, o: &Observation) -> Result<(), String> {
        let result = (|| {
            if self.failed || index != self.next_callback || index >= MAX_GAME_CALLBACKS {
                return Err("Stale/out-of-order guarded deployment callback".into());
            }
            policy::validate_contract(o)?;
            if o.actor >= o.players.len() || o.observation_key != observation_key(o)? {
                return Err("Guarded deployment callback observation mismatch".into());
            }
            let trade = is_trade(o);
            if !trade || self.actor.is_some_and(|actor| actor != o.actor) {
                self.reset_episode();
            }
            self.next_callback += 1;
            self.notified = Some((index, o.actor, trade, o.observation_key.clone()));
            self.chosen = false;
            Ok(())
        })();
        if result.is_err() {
            self.invalidate();
        }
        result
    }

    pub(crate) fn choose_move(
        &mut self,
        index: usize,
        o: &Observation,
        predict: impl FnOnce() -> Result<crate::policy::NumericPolicyDistribution, String>,
        raw_policy_version: &str,
        deployment_checksum: &str,
        guarded_policy_version: &str,
    ) -> Result<GuardedDeploymentDecision, String> {
        let result = (|| {
            if self.failed
                || self.chosen
                || !self
                    .notified
                    .as_ref()
                    .is_some_and(|(notified, actor, trade, key)| {
                        *notified == index
                            && *actor == o.actor
                            && *trade == is_trade(o)
                            && key == &o.observation_key
                    })
            {
                return Err(
                    "Guarded deployment choice requires its single current callback".into(),
                );
            }
            // Distribution validates the contract and recomputes the public
            // key before prediction. One Scalar actor; no critic or weight copy.
            let prediction = predict()?;
            let raw = (1..prediction.logits.len()).fold(0, |best, i| {
                if prediction.logits[i] > prediction.logits[best] {
                    i
                } else {
                    best
                }
            });
            let mut effective = raw;
            let trade = if is_trade(o) {
                let skip = skip_index(o)?;
                let bytes = public_bytes(o)?;
                let public_state_sha256 = format!("{:x}", Sha256::digest(&bytes));
                let transition = self.history.transition(index, bytes, raw == skip)?;
                if transition.force_exit {
                    effective = skip;
                }
                self.actor = Some(o.actor);
                if effective == skip {
                    self.reset_episode();
                }
                Some(GuardedTradeTrace {
                    public_state_sha256,
                    first_seen_callback_index: transition.first_seen_index,
                    repeat_seen: transition.repeat_seen,
                    budget_reached: transition.budget_reached,
                    overridden: transition.force_exit,
                    trade_count_before: transition.count_before,
                    trade_count_after: transition.count_after,
                    episode_closed: effective == skip,
                })
            } else {
                None
            };
            self.chosen = true;
            Ok(GuardedDeploymentDecision {
                decision: DeploymentDecision {
                    actor: o.actor,
                    observation_key: o.observation_key.clone(),
                    r#move: o.legal_actions[effective].r#move.clone(),
                    policy_version: guarded_policy_version.into(),
                    score: f64::from(prediction.logits[effective]),
                },
                raw_policy_version: raw_policy_version.into(),
                deployment_checksum: deployment_checksum.into(),
                guard_version: GUARD_VERSION.into(),
                projection_version: PROJECTION_VERSION.into(),
                configuration_key: self.configuration_key.clone(),
                callback_index: index,
                raw_legal_index: raw,
                effective_legal_index: effective,
                trade,
            })
        })();
        if result.is_err() {
            self.invalidate();
        }
        result
    }
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct GuardedDeploymentDecision {
    #[serde(flatten)]
    pub decision: DeploymentDecision,
    pub raw_policy_version: String,
    pub deployment_checksum: String,
    pub guard_version: String,
    pub projection_version: String,
    pub configuration_key: String,
    pub callback_index: usize,
    pub raw_legal_index: usize,
    pub effective_legal_index: usize,
    pub trade: Option<GuardedTradeTrace>,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct GuardedTradeTrace {
    pub public_state_sha256: String,
    pub first_seen_callback_index: Option<usize>,
    pub repeat_seen: bool,
    pub budget_reached: bool,
    pub overridden: bool,
    pub trade_count_before: usize,
    pub trade_count_after: usize,
    pub episode_closed: bool,
}
