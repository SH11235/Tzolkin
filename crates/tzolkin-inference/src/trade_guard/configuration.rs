//! Shared closed configuration for the public Trade exit guard.
use super::{GUARD_VERSION, MAX_STATE_BYTES, MAX_TRADES, PROJECTION_VERSION};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct TradeGuardConfig {
    pub schema: u32,
    pub algorithm_version: String,
    pub state_projection_version: String,
    pub repeat_state_policy: String,
    pub max_trades_per_episode: usize,
    pub max_public_state_bytes: usize,
}
impl Default for TradeGuardConfig {
    fn default() -> Self {
        Self {
            schema: 1,
            algorithm_version: GUARD_VERSION.into(),
            state_projection_version: PROJECTION_VERSION.into(),
            repeat_state_policy: "secondVisitExit".into(),
            max_trades_per_episode: MAX_TRADES,
            max_public_state_bytes: MAX_STATE_BYTES,
        }
    }
}
impl TradeGuardConfig {
    pub fn validate(&self) -> Result<(), String> {
        if self.schema != 1
            || self.algorithm_version != GUARD_VERSION
            || self.state_projection_version != PROJECTION_VERSION
            || self.repeat_state_policy != "secondVisitExit"
            || !(1..=MAX_TRADES).contains(&self.max_trades_per_episode)
            || self.max_public_state_bytes != MAX_STATE_BYTES
        {
            return Err("Unsupported public Trade guard configuration".into());
        }
        Ok(())
    }
    pub fn configuration_key(&self) -> Result<String, String> {
        self.validate()?;
        let mut hash = Sha256::new();
        hash.update(b"tzolkin-public-trade-guard-config-v1\0");
        hash.update(serde_json::to_vec(self).map_err(|e| e.to_string())?);
        Ok(format!("{:x}", hash.finalize()))
    }
}
