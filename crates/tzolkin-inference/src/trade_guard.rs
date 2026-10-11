//! Bounded byte-history reducer for an explicit Trade exit composition.
//! Callers own observation projection, legal exits, policy identity and callbacks.
//! This state grants no model, game, replay or learning authority.

pub const MAX_TRADES: usize = 64;
pub const MAX_STATE_BYTES: usize = 131_072;

struct SeenState {
    bytes: Vec<u8>,
    index: usize,
}

/// Fresh history for one uninterrupted actor's Trade episode. No wire state.
pub struct TradeHistory {
    max_trades: usize,
    max_state_bytes: usize,
    trades: usize,
    seen: Vec<SeenState>,
}

#[derive(Debug, PartialEq, Eq)]
pub struct TradeTransition {
    pub first_seen_index: Option<usize>,
    pub repeat_seen: bool,
    pub budget_reached: bool,
    pub force_exit: bool,
    pub count_before: usize,
    pub count_after: usize,
}

impl TradeHistory {
    pub fn new(max_trades: usize, max_state_bytes: usize) -> Result<Self, String> {
        if !(1..=MAX_TRADES).contains(&max_trades)
            || !(1..=MAX_STATE_BYTES).contains(&max_state_bytes)
        {
            return Err("Unsupported bounded Trade history".into());
        }
        Ok(Self {
            max_trades,
            max_state_bytes,
            trades: 0,
            seen: Vec::new(),
        })
    }

    pub fn reset(&mut self) {
        self.trades = 0;
        self.seen.clear();
    }

    /// Projection/legal/logit checks precede this single atomic reducer step.
    /// An exit closes the episode; otherwise retain the exact bytes, not a hash.
    pub fn transition(
        &mut self,
        index: usize,
        bytes: Vec<u8>,
        raw_is_exit: bool,
    ) -> Result<TradeTransition, String> {
        if bytes.len() > self.max_state_bytes {
            return Err("Trade history state byte budget exceeded".into());
        }
        let first_seen_index = self.seen.iter().find(|s| s.bytes == bytes).map(|s| s.index);
        let repeat_seen = first_seen_index.is_some();
        let budget_reached = self.trades >= self.max_trades;
        let force_exit = !raw_is_exit && (repeat_seen || budget_reached);
        let exits = raw_is_exit || force_exit;
        let result = TradeTransition {
            first_seen_index,
            repeat_seen,
            budget_reached,
            force_exit,
            count_before: self.trades,
            count_after: self.trades + usize::from(!exits),
        };
        if !exits && self.seen.len() >= MAX_TRADES {
            return Err("Public Trade cache budget exceeded".into());
        }
        if exits {
            self.reset();
        } else {
            self.seen.push(SeenState { bytes, index });
            self.trades = result.count_after;
        }
        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bounded_exact_history_repeat_cap_exit_and_reset() {
        let mut history = TradeHistory::new(2, 3).unwrap();
        let first = history.transition(4, vec![1], false).unwrap();
        assert_eq!(
            (first.count_before, first.count_after, first.force_exit),
            (0, 1, false)
        );
        // Oversized input fails before changing the previously accepted history.
        assert!(history.transition(5, vec![0; 4], false).is_err());
        let repeat = history.transition(5, vec![1], false).unwrap();
        assert_eq!(repeat.first_seen_index, Some(4));
        assert!(repeat.force_exit && repeat.repeat_seen && !repeat.budget_reached);
        assert_eq!((repeat.count_before, repeat.count_after), (1, 1));
        history.transition(6, vec![1], false).unwrap();
        history.transition(7, vec![2], false).unwrap();
        let cap = history.transition(8, vec![3], false).unwrap();
        assert!(cap.force_exit && cap.budget_reached && !cap.repeat_seen);
        let voluntary = history.transition(9, vec![1], true).unwrap();
        assert_eq!((voluntary.count_before, voluntary.count_after), (0, 0));
        history.transition(10, vec![1], false).unwrap();
        history.reset();
        assert!(!history.transition(11, vec![1], false).unwrap().repeat_seen);
        assert!(TradeHistory::new(0, 3).is_err());
        assert!(TradeHistory::new(MAX_TRADES + 1, 3).is_err());
    }
}
