use crate::engine::{
    apply_move, available_workers, create_game_with_options, get_available_moves, get_choices,
    get_placement_cost, score_monument,
};
use crate::types::{Choice, GEAR_IDS, GameMove, GameOptions, GameState, GearId, Player};
use crate::validation::{normalize_json_integers, validate_game_state};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::BTreeMap;

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct GameSnapshot {
    pub state: GameState,
    pub choices: Vec<Choice>,
    pub moves: Vec<Choice>,
    pub placement_costs: BTreeMap<GearId, Option<i64>>,
    pub available_workers: Vec<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub expansion_catalog: Option<Value>,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "operation", rename_all = "camelCase")]
enum Request {
    Create {
        names: Vec<String>,
        seed: u32,
        #[serde(flatten)]
        options: GameOptions,
    },
    Apply {
        state: Value,
        r#move: GameMove,
    },
    Inspect {
        state: Value,
    },
    Observe {
        state: Value,
        actor: usize,
    },
    Validate {
        value: Value,
    },
    Catalog,
    Score {
        state: Value,
        player: Box<Player>,
        id: String,
    },
}

fn snapshot(state: GameState) -> GameSnapshot {
    GameSnapshot {
        expansion_catalog: state.expansion.as_ref().map(|_| {
            serde_json::json!({
                "tribes": crate::tribes::definitions(),
                "prophecies": crate::prophecies::definitions(),
                "quickActions": crate::quick_actions::definitions(),
            })
        }),
        choices: get_choices(&state),
        moves: get_available_moves(&state),
        placement_costs: GEAR_IDS
            .into_iter()
            .map(|gear| (gear, get_placement_cost(&state, &gear.to_string())))
            .collect(),
        available_workers: (0..state.players.len())
            .map(|id| available_workers(&state, id))
            .collect(),
        state,
    }
}

fn state_from_json(state: Value, checked: bool) -> Result<GameState, String> {
    if checked && !validate_game_state(&state) {
        return Err("対局データが不正です。".into());
    }
    serde_json::from_value(state).map_err(|error| format!("対局データが不正です：{error}"))
}

fn result_snapshot(state: GameState, checked: bool) -> Result<String, String> {
    if checked {
        let value = serde_json::to_value(&state).map_err(|error| error.to_string())?;
        if !validate_game_state(&value) {
            return Err("ゲーム処理が不正な対局データを返しました。".into());
        }
    }
    serde_json::to_string(&snapshot(state)).map_err(|error| error.to_string())
}

fn dispatch(request: &str, checked: bool) -> Result<String, String> {
    let mut value: Value =
        serde_json::from_str(request).map_err(|error| format!("リクエストが不正です：{error}"))?;
    normalize_json_integers(&mut value);
    let request: Request =
        serde_json::from_value(value).map_err(|error| format!("リクエストが不正です：{error}"))?;
    match request {
        Request::Create {
            names,
            seed,
            options,
        } => result_snapshot(create_game_with_options(names, seed, options)?, checked),
        Request::Apply { state, r#move } => {
            let state = state_from_json(state, checked)?;
            result_snapshot(apply_move(&state, r#move)?, checked)
        }
        Request::Inspect { state } => result_snapshot(state_from_json(state, checked)?, checked),
        Request::Observe { state, actor } => {
            let state = state_from_json(state, true)?;
            serde_json::to_string(&crate::observation::observe(&state, actor)?)
                .map_err(|error| error.to_string())
        }
        Request::Validate { value } => Ok(validate_game_state(&value).to_string()),
        Request::Catalog => Ok(include_str!("../data/catalog.json").into()),
        Request::Score { state, player, id } => {
            if checked {
                return Err("未対応のリクエストです。".into());
            }
            let state = state_from_json(state, false)?;
            serde_json::to_string(&score_monument(&state, &player, &id))
                .map_err(|error| error.to_string())
        }
    }
}

/// The browser and native shells share this validated JSON boundary.
pub fn dispatch_game(request: &str) -> Result<String, String> {
    dispatch(request, true)
}

/// Only the separate, feature-gated Node fixture build exposes this entry point.
pub fn dispatch_for_test(request: &str) -> Result<String, String> {
    dispatch(request, false)
}
