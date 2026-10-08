//! Compact, lossless rule state with an adapter to the reference engine.
//!
//! Array/bitset storage and presentation ownership are established here. Rules,
//! choice generation and automatic continuations still run in the reference
//! engine: this adapter is not an independent optimized rules implementation.
pub mod catalog;
mod state;

pub use catalog::{BuildingId, MonumentId, WealthId};
pub use state::*;
