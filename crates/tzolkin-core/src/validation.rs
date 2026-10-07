use crate::types::GameState;
use serde_json::{Number, Value};
use std::collections::HashSet;
use std::sync::OnceLock;

const MAX_SAFE_INTEGER: f64 = 9_007_199_254_740_991.0;
const GEARS: [&str; 5] = ["palenque", "yaxchilan", "tikal", "uxmal", "chichenItza"];
const TEMPLES: [&str; 3] = ["chaac", "quetzalcoatl", "kukulkan"];
const TECHNOLOGIES: [&str; 4] = ["agriculture", "extraction", "architecture", "theology"];
const RESOURCES: [&str; 5] = ["corn", "wood", "stone", "gold", "skull"];

fn catalog() -> &'static Value {
    static CATALOG: OnceLock<Value> = OnceLock::new();
    CATALOG.get_or_init(|| {
        serde_json::from_str(include_str!("../data/catalog.json"))
            .expect("the bundled component catalog is valid JSON")
    })
}

fn integer(value: &Value, minimum: i64, maximum: i64) -> bool {
    value.as_f64().is_some_and(|number| {
        number.is_finite()
            && number.fract() == 0.0
            && number.abs() <= MAX_SAFE_INTEGER
            && number >= minimum as f64
            && number <= maximum as f64
    })
}

fn nonnegative_integer(value: &Value) -> bool {
    integer(value, 0, MAX_SAFE_INTEGER as i64)
}

fn number(value: &Value) -> f64 {
    value.as_f64().unwrap_or(f64::NAN)
}

fn quarter_points(value: &Value) -> bool {
    value.as_f64().is_some_and(|number| {
        let quarters = number * 4.0;
        number.is_finite()
            && quarters.is_finite()
            && quarters.fract() == 0.0
            && quarters.abs() <= MAX_SAFE_INTEGER
    })
}

fn has(values: &[&str], value: &Value) -> bool {
    value.as_str().is_some_and(|text| values.contains(&text))
}

fn optional(value: &Value, key: &str, predicate: impl FnOnce(&Value) -> bool) -> bool {
    value.get(key).is_none_or(predicate)
}

fn maximum_position(gear: &str) -> i64 {
    if gear == "chichenItza" { 10 } else { 7 }
}

fn temple_maximum(temple: &str) -> i64 {
    catalog()["TEMPLE_TRACKS"][temple]["points"]
        .as_array()
        .expect("the component catalog contains every temple track")
        .len() as i64
        - 2
}

fn string_length_at_most(value: &Value, maximum: usize) -> bool {
    value
        .as_str()
        .is_some_and(|text| text.encode_utf16().count() <= maximum)
}

fn js_whitespace(character: char) -> bool {
    matches!(
        character,
        '\u{0009}'..='\u{000d}'
            | '\u{0020}'
            | '\u{00a0}'
            | '\u{1680}'
            | '\u{2000}'..='\u{200a}'
            | '\u{2028}'
            | '\u{2029}'
            | '\u{202f}'
            | '\u{205f}'
            | '\u{3000}'
            | '\u{feff}'
    )
}

/// Keep player name normalization consistent with the v1 browser implementation.
pub fn trim_player_name(name: &str) -> &str {
    name.trim_matches(js_whitespace)
}

fn resource_record(value: &Value, partial: bool) -> bool {
    value.as_object().is_some_and(|resources| {
        resources
            .keys()
            .all(|key| RESOURCES.contains(&key.as_str()))
            && RESOURCES.iter().all(|resource| {
                resources
                    .get(*resource)
                    .map_or(partial, nonnegative_integer)
            })
    })
}

fn valid_effect(value: &Value) -> bool {
    if !value.is_object() {
        return false;
    }
    match value["type"].as_str() {
        Some("resources" | "foodReward") => resource_record(&value["resources"], true),
        Some("feed") => value["workers"] == "all" || integer(&value["workers"], 0, 6),
        Some("feedDiscount") => integer(&value["amount"], 0, 2),
        Some("technology") => {
            (value["technology"] == "any" || has(&TECHNOLOGIES, &value["technology"]))
                && optional(value, "steps", |steps| integer(steps, 1, 2))
        }
        Some("temple") => {
            (value["temple"] == "any" || has(&TEMPLES, &value["temple"]))
                && optional(value, "steps", |steps| integer(steps, 1, 2))
        }
        Some("points") => integer(&value["amount"], -1000, 1000),
        Some("skullBuilding") => {
            integer(&value["points"], 0, 100)
                && (value["temple"] == "any" || has(&TEMPLES, &value["temple"]))
        }
        Some("action") => optional(value, "anywhere", Value::is_boolean),
        Some(
            "worker" | "build" | "trade" | "renovation" | "foodRewardSwitch" | "buildMonument"
            | "technologyExchange",
        ) => true,
        _ => false,
    }
}

fn valid_task(value: &Value) -> bool {
    if !value.is_object() {
        return false;
    }
    match value["type"].as_str() {
        Some("effects") => value["effects"]
            .as_array()
            .is_some_and(|effects| effects.len() <= 12 && effects.iter().all(valid_effect)),
        Some("action") => {
            has(&GEARS, &value["gear"])
                && integer(
                    &value["position"],
                    0,
                    maximum_position(value["gear"].as_str().unwrap_or("")),
                )
                && optional(value, "free", Value::is_boolean)
        }
        Some("technology") => integer(&value["remaining"], 0, 2) && value["free"].is_boolean(),
        Some("payTechnology") => {
            has(&TECHNOLOGIES, &value["technology"]) && integer(&value["amount"], 1, 3)
        }
        Some("payResource") => integer(&value["amount"], 1, 3),
        Some("temple") => {
            integer(&value["remaining"], 0, 2)
                && optional(value, "distinct", |distinct| {
                    distinct.as_array().is_some_and(|temples| {
                        temples.len() <= 2
                            && temples.iter().all(|temple| has(&TEMPLES, temple))
                            && unique_values(temples)
                    })
                })
                && optional(value, "direction", |direction| {
                    number(direction) == -1.0 || number(direction) == 1.0
                })
                && optional(value, "reason", |reason| has(&["beg", "burn"], reason))
        }
        Some("resource") => integer(&value["remaining"], 0, 2),
        Some("build") => {
            integer(&value["remaining"], 0, 2)
                && value["allowMonument"].is_boolean()
                && value["cornPayment"].is_boolean()
                && optional(value, "architectureAvailable", Value::is_boolean)
        }
        Some("trade" | "rotation" | "buildMonument" | "technologyExchange") => true,
        Some("anyAction") => {
            optional(value, "excludeSkulls", Value::is_boolean)
                && optional(value, "cost", |cost| integer(cost, 0, 1))
        }
        Some("palenque") => integer(&value["position"], 2, 5),
        Some("theology") => optional(value, "position", |position| integer(position, 1, 9)),
        _ => false,
    }
}

fn component(catalog_key: &str, id: &str) -> Option<&'static Value> {
    catalog()[catalog_key]
        .as_array()?
        .iter()
        .find(|component| component["id"].as_str() == Some(id))
}

fn unique_values(values: &[Value]) -> bool {
    values.iter().enumerate().all(|(index, value)| {
        !values[..index].iter().any(|previous| {
            if previous.is_number() && value.is_number() {
                number(previous) == number(value)
            } else {
                previous == value
            }
        })
    })
}

fn ids(value: &Value, catalog_key: &str, maximum: usize) -> bool {
    value.as_array().is_some_and(|values| {
        values.len() <= maximum
            && unique_values(values)
            && values.iter().all(|id| {
                id.as_str()
                    .is_some_and(|id| component(catalog_key, id).is_some())
            })
    })
}

/// JSON numbers such as `1.0` have the same integer meaning as `1` in v1 browser saves.
/// Normalize those spellings before deserializing the typed Rust state.
pub fn normalize_json_integers(value: &mut Value) {
    match value {
        Value::Number(number) => {
            if let Some(integer) = number.as_f64().filter(|number| {
                number.is_finite() && number.fract() == 0.0 && number.abs() <= MAX_SAFE_INTEGER
            }) {
                *number = Number::from(integer as i64);
            }
        }
        Value::Array(values) => values.iter_mut().for_each(normalize_json_integers),
        Value::Object(values) => values.values_mut().for_each(normalize_json_integers),
        _ => {}
    }
}

/// Validate untrusted v1 save data before restoring it, including component conservation
/// and at least one enabled choice for every active pending task.
pub fn validate_game_state(value: &Value) -> bool {
    if !valid_shape(value) || !valid_components(value) {
        return false;
    }
    let mut normalized = value.clone();
    normalize_json_integers(&mut normalized);
    let Ok(state) = serde_json::from_value::<GameState>(normalized) else {
        return false;
    };
    if state.pending.is_some() {
        crate::engine::get_choices(&state)
            .iter()
            .any(|choice| !choice.disabled.unwrap_or(false))
    } else {
        true
    }
}

fn valid_shape(value: &Value) -> bool {
    if !value.is_object()
        || number(&value["version"]) != 1.0
        || !value["additionalBuildings"].is_boolean()
        || !integer(&value["seed"], 0, u32::MAX.into())
        || !has(&["setup", "playing", "finished"], &value["phase"])
        || !integer(&value["round"], 1, 27)
        || ![1.0, 2.0].contains(&number(&value["age"]))
    {
        return false;
    }
    let Some(players) = value["players"]
        .as_array()
        .filter(|players| (2..=4).contains(&players.len()))
    else {
        return false;
    };
    let player_count = players.len();
    for (id, player) in players.iter().enumerate() {
        if !player.is_object()
            || number(&player["id"]) != id as f64
            || !player["name"].as_str().is_some_and(|name| {
                !trim_player_name(name).is_empty() && name.encode_utf16().count() <= 100
            })
            || !player["color"].as_str().is_some_and(|color| {
                color.len() == 7
                    && color.starts_with('#')
                    && color.as_bytes()[1..].iter().all(u8::is_ascii_hexdigit)
            })
            || !resource_record(&player["resources"], false)
            || !quarter_points(&player["score"])
            || !integer(&player["workers"], 3, 6)
        {
            return false;
        }
        if !player["temples"].is_object()
            || !TEMPLES
                .iter()
                .all(|temple| integer(&player["temples"][temple], -1, temple_maximum(temple)))
            || !player["technologies"].is_object()
            || !TECHNOLOGIES
                .iter()
                .all(|technology| integer(&player["technologies"][technology], 0, 3))
        {
            return false;
        }
        if !ids(&player["buildings"], "ALL_BUILDINGS", 40)
            || !ids(&player["monuments"], "MONUMENTS", 6)
            || !ids(&player["wealthOffer"], "STARTING_WEALTH", 4)
            || player["wealthOffer"].as_array().unwrap().len() != 4
            || !ids(&player["wealth"], "STARTING_WEALTH", 2)
            || !player["wealth"]
                .as_array()
                .unwrap()
                .iter()
                .all(|id| player["wealthOffer"].as_array().unwrap().contains(id))
        {
            return false;
        }
        if !integer(&player["feedWorkers"], 0, 32)
            || !player["feedAll"].is_boolean()
            || !integer(&player["feedDiscount"], 0, 32)
            || !integer(&player["cornTiles"], 0, 16)
            || !integer(&player["woodTiles"], 0, 12)
            || !integer(&player["skullsPlaced"], 0, 10)
            || !integer(&player["buildingSkulls"], 0, 1)
            || !player["doubleAdvanceAvailable"].is_boolean()
            || !integer(&player["templePoints"], -12, 100)
        {
            return false;
        }
    }
    if !["currentPlayer", "firstPlayer", "turnIndex"]
        .iter()
        .all(|key| integer(&value[key], 0, player_count as i64 - 1))
        || !value["turnOrder"].as_array().is_some_and(|order| {
            order.len() == player_count
                && unique_values(order)
                && order
                    .iter()
                    .all(|id| integer(id, 0, player_count as i64 - 1))
        })
        || !value["turn"].is_object()
        || !has(&["none", "place", "remove"], &value["turn"]["mode"])
        || !integer(&value["turn"]["count"], 0, 6)
        || !value["turn"]["begged"].is_boolean()
        || (value["turn"]["mode"] == "none") != (number(&value["turn"]["count"]) == 0.0)
    {
        return false;
    }
    if !value["gears"].is_object()
        || !value["jungle"].is_object()
        || !integer(&value["skullSupply"], 0, 13)
        || !integer(&value["accumulatedCorn"], 0, 27)
        || !value.get("firstPlayerClaimed").is_some_and(|claimed| {
            claimed.is_null() || integer(claimed, 0, player_count as i64 - 1)
        })
    {
        return false;
    }
    for gear in GEARS {
        let length = if gear == "chichenItza" { 13 } else { 10 };
        let Some(workers) = value["gears"][gear]
            .as_array()
            .filter(|workers| workers.len() == length)
        else {
            return false;
        };
        for (position, worker) in workers.iter().enumerate() {
            if worker.is_null() {
                continue;
            }
            if !worker.is_object() || !worker["dummy"].is_boolean() {
                return false;
            }
            let dummy = worker["dummy"].as_bool().unwrap();
            if !integer(
                &worker["playerId"],
                if dummy { -1 } else { 0 },
                if dummy { -1 } else { player_count as i64 - 1 },
            ) || (!dummy && position as i64 > maximum_position(gear))
            {
                return false;
            }
        }
    }
    for position in 2..=5 {
        let jungle = &value["jungle"][position.to_string()];
        if !jungle.is_object()
            || !integer(&jungle["corn"], 0, player_count as i64)
            || !integer(
                &jungle["wood"],
                0,
                if position == 2 {
                    0
                } else {
                    player_count as i64
                },
            )
            || number(&jungle["wood"]) > number(&jungle["corn"])
        {
            return false;
        }
    }
    if !value["skullSpaces"].as_array().is_some_and(|spaces| {
        spaces.len() == 10
            && spaces[0].is_null()
            && spaces
                .iter()
                .all(|id| id.is_null() || integer(id, 0, player_count as i64 - 1))
    }) || !ids(&value["buildings"], "ALL_BUILDINGS", 6)
        || !ids(&value["buildingDeck"], "ALL_BUILDINGS", 22)
        || !ids(&value["age2Deck"], "ALL_BUILDINGS", 22)
        || !ids(&value["monuments"], "MONUMENTS", player_count + 2)
    {
        return false;
    }
    let Some(pending) = value.get("pending") else {
        return false;
    };
    if !pending.is_null()
        && (!pending.is_object()
            || !string_length_at_most(&pending["title"], 200)
            || !valid_task(&pending["task"])
            || !pending["after"]
                .as_array()
                .is_some_and(|tasks| tasks.len() <= 32 && tasks.iter().all(valid_task)))
    {
        return false;
    }
    if !pending.is_null()
        && (pending["task"]["type"] == "effects"
            || pending["task"]
                .get("remaining")
                .is_some_and(|remaining| number(remaining) == 0.0))
    {
        return false;
    }
    if !value["log"].as_array().is_some_and(|log| {
        log.len() <= 500 && log.iter().all(|entry| string_length_at_most(entry, 1000))
    }) || !value["foodDays"].as_array().is_some_and(|days| {
        let food_days = [8, 14, 21, 27];
        days.len() <= 4
            && days
                .iter()
                .zip(food_days)
                .all(|(day, expected)| number(day) == expected as f64)
    }) || !value["finalScores"].as_array().is_some_and(|scores| {
        scores.len()
            == if value["phase"] == "finished" {
                player_count
            } else {
                0
            }
    }) {
        return false;
    }
    if value["phase"] == "finished" {
        let mut seen = HashSet::new();
        for score in value["finalScores"].as_array().unwrap() {
            if !score.is_object()
                || !integer(&score["playerId"], 0, player_count as i64 - 1)
                || !seen.insert(number(&score["playerId"]) as usize)
                || !integer(&score["rank"], 1, player_count as i64)
                || !integer(&score["workersOnGears"], 0, 6)
                || ![
                    "pointsBeforeFinal",
                    "resourcePoints",
                    "skullPoints",
                    "monumentPoints",
                    "total",
                ]
                .iter()
                .all(|key| quarter_points(&score[key]))
            {
                return false;
            }
            let id = number(&score["playerId"]) as usize;
            let total = number(&score["total"]);
            if total != number(&players[id]["score"])
                || total
                    != number(&score["pointsBeforeFinal"])
                        + number(&score["resourcePoints"])
                        + number(&score["skullPoints"])
                        + number(&score["monumentPoints"])
            {
                return false;
            }
        }
    }
    true
}

fn valid_components(value: &Value) -> bool {
    let players = value["players"].as_array().unwrap();
    let player_count = players.len();
    let collect_ids = |key: &str| -> Vec<&str> {
        players
            .iter()
            .flat_map(|player| {
                player[key]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|id| id.as_str().unwrap())
            })
            .collect()
    };
    let unique_ids = |ids: &[&str]| ids.iter().copied().collect::<HashSet<_>>().len() == ids.len();
    if !unique_ids(&collect_ids("wealthOffer")) {
        return false;
    }
    let mut all_buildings = collect_ids("buildings");
    for key in ["buildings", "buildingDeck", "age2Deck"] {
        all_buildings.extend(
            value[key]
                .as_array()
                .unwrap()
                .iter()
                .map(|id| id.as_str().unwrap()),
        );
    }
    let mut all_monuments = collect_ids("monuments");
    all_monuments.extend(
        value["monuments"]
            .as_array()
            .unwrap()
            .iter()
            .map(|id| id.as_str().unwrap()),
    );
    if !unique_ids(&all_buildings)
        || !unique_ids(&all_monuments)
        || all_monuments.len() != player_count + 2
        || (value["additionalBuildings"] == false
            && all_buildings
                .iter()
                .any(|id| component("EXPANSION_BUILDINGS", id).is_some()))
    {
        return false;
    }
    for key in ["buildings", "buildingDeck", "age2Deck"] {
        let age = if key == "age2Deck" {
            2.0
        } else {
            number(&value["age"])
        };
        if value[key].as_array().unwrap().iter().any(|id| {
            number(&component("ALL_BUILDINGS", id.as_str().unwrap()).unwrap()["age"]) != age
        }) {
            return false;
        }
    }
    if value["phase"] != "setup"
        && players
            .iter()
            .any(|player| player["wealth"].as_array().unwrap().len() != 2)
    {
        return false;
    }
    let gear_workers: Vec<&Value> = GEARS
        .iter()
        .flat_map(|gear| value["gears"][gear].as_array().unwrap())
        .collect();
    if gear_workers
        .iter()
        .filter(|worker| worker["dummy"] == true)
        .count()
        != (4 - player_count) * 6
    {
        return false;
    }
    let skull_spaces = value["skullSpaces"].as_array().unwrap();
    let skull_total = number(&value["skullSupply"])
        + players
            .iter()
            .map(|player| number(&player["resources"]["skull"]) + number(&player["buildingSkulls"]))
            .sum::<f64>()
        + skull_spaces.iter().filter(|id| !id.is_null()).count() as f64;
    if skull_total != 13.0 {
        return false;
    }
    for player in players {
        let id = number(&player["id"]);
        let on_gears = gear_workers
            .iter()
            .filter(|worker| worker["dummy"] == false && number(&worker["playerId"]) == id)
            .count();
        let first_player_worker = usize::from(
            !value["firstPlayerClaimed"].is_null() && number(&value["firstPlayerClaimed"]) == id,
        );
        if number(&player["workers"]) - ((on_gears + first_player_worker) as f64) < 0.0
            || number(&player["skullsPlaced"])
                != skull_spaces
                    .iter()
                    .filter(|owner| !owner.is_null() && number(owner) == id)
                    .count() as f64
                    + number(&player["buildingSkulls"])
        {
            return false;
        }
        let building_skulls: f64 = player["buildings"]
            .as_array()
            .unwrap()
            .iter()
            .map(|id| {
                let building = component("ALL_BUILDINGS", id.as_str().unwrap()).unwrap();
                if building["effects"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .any(|effect| effect["type"] == "skullBuilding")
                {
                    building["cost"]["skull"].as_f64().unwrap_or(0.0)
                } else {
                    0.0
                }
            })
            .sum();
        if number(&player["buildingSkulls"]) != building_skulls {
            return false;
        }
    }
    if TEMPLES.iter().any(|temple| {
        players
            .iter()
            .filter(|player| number(&player["temples"][temple]) == temple_maximum(temple) as f64)
            .count()
            > 1
    }) {
        return false;
    }
    let rotation = value["pending"]["task"]["type"] == "rotation";
    if rotation
        && (value["firstPlayerClaimed"].is_null()
            || number(&value["currentPlayer"]) != number(&value["firstPlayerClaimed"]))
    {
        return false;
    }
    if value["phase"] == "playing" && !rotation {
        let turn_index = number(&value["turnIndex"]) as usize;
        if number(&value["currentPlayer"]) != number(&value["turnOrder"][turn_index]) {
            return false;
        }
    }
    let food_days = value["foodDays"].as_array().unwrap();
    let passed_age_one = food_days.iter().any(|day| number(day) == 14.0);
    if (number(&value["age"]) == 2.0) != passed_age_one {
        return false;
    }
    if value["phase"] == "finished"
        && (number(&value["round"]) != 27.0
            || !food_days.iter().any(|day| number(day) == 27.0)
            || !value["pending"].is_null()
            || !value["firstPlayerClaimed"].is_null())
    {
        return false;
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn setup_fixture() -> Value {
        let wealth = catalog()["STARTING_WEALTH"].as_array().unwrap();
        let players: Vec<Value> = (0..4)
            .map(|id| {
                let offer: Vec<Value> = wealth[id * 4..id * 4 + 4]
                    .iter()
                    .map(|tile| tile["id"].clone())
                    .collect();
                json!({
                    "id":id, "name":format!("player {id}"), "color":"#378575",
                    "resources":{"corn":0, "wood":0, "stone":0, "gold":0, "skull":0},
                    "score":0, "workers":3,
                    "temples":{"chaac":0, "quetzalcoatl":0, "kukulkan":0},
                    "technologies":{"agriculture":0, "extraction":0, "architecture":0, "theology":0},
                    "buildings":[], "monuments":[], "wealth":[], "wealthOffer":offer,
                    "feedWorkers":0, "feedAll":false, "feedDiscount":0,
                    "cornTiles":0, "woodTiles":0, "skullsPlaced":0, "buildingSkulls":0,
                    "doubleAdvanceAvailable":true, "templePoints":0
                })
            })
            .collect();
        let buildings = catalog()["BUILDINGS"].as_array().unwrap();
        let age_one: Vec<Value> = buildings
            .iter()
            .filter(|building| number(&building["age"]) == 1.0)
            .map(|building| building["id"].clone())
            .collect();
        let age_two: Vec<Value> = buildings
            .iter()
            .filter(|building| number(&building["age"]) == 2.0)
            .map(|building| building["id"].clone())
            .collect();
        let monuments: Vec<Value> = catalog()["MONUMENTS"].as_array().unwrap()[..6]
            .iter()
            .map(|monument| monument["id"].clone())
            .collect();
        json!({
            "version":1, "seed":4, "additionalBuildings":false, "phase":"setup", "round":1, "age":1,
            "players":players, "currentPlayer":0, "firstPlayer":0, "turnOrder":[0,1,2,3], "turnIndex":0,
            "turn":{"mode":"none", "count":0, "begged":false},
            "gears":{"palenque":vec![Value::Null;10], "yaxchilan":vec![Value::Null;10], "tikal":vec![Value::Null;10], "uxmal":vec![Value::Null;10], "chichenItza":vec![Value::Null;13]},
            "jungle":{"2":{"corn":4, "wood":0}, "3":{"corn":4, "wood":4}, "4":{"corn":4, "wood":4}, "5":{"corn":4, "wood":4}},
            "skullSupply":13, "skullSpaces":vec![Value::Null;10], "firstPlayerClaimed":null, "accumulatedCorn":0,
            "buildings":age_one[..6], "buildingDeck":age_one[6..], "age2Deck":age_two, "monuments":monuments,
            "pending":null, "log":[], "foodDays":[], "finalScores":[]
        })
    }

    fn playing_fixture() -> Value {
        let mut state = setup_fixture();
        state["phase"] = json!("playing");
        for player in state["players"].as_array_mut().unwrap() {
            player["wealth"] = json!(player["wealthOffer"].as_array().unwrap()[..2]);
        }
        state
    }

    #[test]
    fn numeric_validation_retains_js_safe_integer_and_quarter_point_limits() {
        assert!(integer(&json!(1.0), 0, 6));
        assert!(nonnegative_integer(&json!(9_007_199_254_740_991_i64)));
        assert!(!nonnegative_integer(&json!(9_007_199_254_740_992_i64)));
        assert!(!nonnegative_integer(&json!(-1)));
        assert!(!nonnegative_integer(&json!(1.25)));
        assert!(!unique_values(&[json!(0), json!(0.0)]));
        assert!(quarter_points(&json!(-12.25)));
        assert!(!quarter_points(&json!(1.1)));
        let mut value = json!({"count": 1.0, "score": 0.25, "array": [2.0, null]});
        normalize_json_integers(&mut value);
        assert_eq!(value["count"].as_i64(), Some(1));
        assert_eq!(value["array"][0].as_i64(), Some(2));
        assert_eq!(value["score"].as_f64(), Some(0.25));
    }

    #[test]
    fn optional_fields_reject_explicit_null_and_internal_tasks_are_bounded() {
        assert!(valid_task(
            &json!({"type":"action", "gear":"tikal", "position":1})
        ));
        assert!(!valid_task(
            &json!({"type":"action", "gear":"tikal", "position":1, "free":null})
        ));
        assert!(!valid_task(
            &json!({"type":"effects", "effects":[{"type":"resources", "resources":{"invented":1}}]})
        ));
        assert!(!valid_task(
            &json!({"type":"temple", "remaining":1, "distinct":["chaac", "chaac"]})
        ));
        assert!(!valid_task(&json!({"type":"invented"})));
        assert!(!validate_game_state(&json!(null)));
        assert!(!validate_game_state(&json!({"version":1})));
    }

    #[test]
    fn limits_count_utf16_units_and_ecmascript_whitespace() {
        assert!(string_length_at_most(&json!("😀".repeat(50)), 100));
        assert!(!string_length_at_most(&json!("😀".repeat(51)), 100));
        assert!("\u{feff}\u{3000}".trim_matches(js_whitespace).is_empty());
        assert!(!"\u{0085}".trim_matches(js_whitespace).is_empty());
    }

    #[test]
    fn accepts_v1_setup_playing_and_alternative_integer_spellings() {
        assert!(validate_game_state(&setup_fixture()));
        let mut state = playing_fixture();
        assert!(validate_game_state(&state));
        state["version"] = json!(1.0);
        state["players"][0]["id"] = json!(0.0);
        state["round"] = json!(1.0);
        assert!(validate_game_state(&state));
    }

    #[test]
    fn accepts_generated_two_three_and_four_player_saves_with_exact_dummy_counts() {
        for count in 2..=4 {
            for seed in [0, 4, u32::MAX] {
                for additional_buildings in [false, true] {
                    let names = (0..count).map(|id| format!("player {id}")).collect();
                    let state =
                        crate::engine::create_game(names, seed, additional_buildings).unwrap();
                    let mut value = serde_json::to_value(state).unwrap();
                    assert!(
                        validate_game_state(&value),
                        "count={count}, seed={seed}, additional={additional_buildings}"
                    );
                    let mut changed = false;
                    for gear in GEARS {
                        for worker in value["gears"][gear].as_array_mut().unwrap() {
                            if worker["dummy"] == true {
                                *worker = Value::Null;
                                changed = true;
                                break;
                            }
                        }
                        if changed {
                            break;
                        }
                    }
                    if !changed {
                        value["gears"]["palenque"][0] = json!({"playerId":-1, "dummy":true});
                    }
                    assert!(!validate_game_state(&value));
                }
            }
        }
    }

    #[test]
    fn expansion_skull_buildings_retain_exactly_one_skull_outside_the_bank() {
        let mut state = playing_fixture();
        state["additionalBuildings"] = json!(true);
        state["players"][0]["buildings"] = json!(["b40"]);
        state["players"][0]["buildingSkulls"] = json!(1);
        state["players"][0]["skullsPlaced"] = json!(1);
        state["skullSupply"] = json!(12);
        assert!(validate_game_state(&state));
        state["additionalBuildings"] = json!(false);
        assert!(!validate_game_state(&state));
        state["additionalBuildings"] = json!(true);
        state["players"][0]["buildingSkulls"] = json!(0);
        state["players"][0]["skullsPlaced"] = json!(0);
        state["skullSupply"] = json!(13);
        assert!(!validate_game_state(&state));
    }

    #[test]
    fn rejects_catalog_prototype_names_in_all_id_collections() {
        for id in ["toString", "constructor", "__proto__"] {
            for path in [
                "/players/0/wealthOffer/0",
                "/players/0/wealth/0",
                "/players/0/buildings",
                "/players/0/monuments",
                "/buildings/0",
                "/buildingDeck/0",
                "/age2Deck/0",
                "/monuments/0",
            ] {
                let mut state = playing_fixture();
                *state.pointer_mut(path).unwrap() =
                    if path.ends_with("buildings") || path.ends_with("monuments") {
                        json!([id])
                    } else {
                        json!(id)
                    };
                assert!(!validate_game_state(&state), "{id} at {path}");
            }
        }
    }

    #[test]
    fn rejects_automatic_zero_remaining_and_stranded_pending_tasks() {
        for task in [
            json!({"type":"effects", "effects":[]}),
            json!({"type":"effects", "effects":[{"type":"points", "amount":2}]}),
            json!({"type":"resource", "remaining":0}),
            json!({"type":"technology", "remaining":0, "free":true}),
            json!({"type":"temple", "remaining":0}),
            json!({"type":"build", "remaining":0, "allowMonument":false, "cornPayment":false}),
            json!({"type":"payResource", "amount":1}),
        ] {
            let mut state = playing_fixture();
            state["pending"] = json!({"title":"pending", "task":task, "after":[]});
            assert!(!validate_game_state(&state), "{task}");
        }
        let mut state = playing_fixture();
        state["players"][0]["temples"] = json!({"chaac":-1, "quetzalcoatl":-1, "kukulkan":-1});
        state["pending"] = json!({"title":"descend", "task":{"type":"temple", "remaining":1, "direction":-1}, "after":[]});
        assert!(!validate_game_state(&state));
        state["pending"] = json!({"title":"choose", "task":{"type":"resource", "remaining":1}, "after":[{"type":"effects", "effects":[]}, {"type":"effects", "effects":[{"type":"points", "amount":2}]}]});
        assert!(validate_game_state(&state));
    }

    #[test]
    fn rejects_shape_numeric_owner_conservation_and_turn_corruption() {
        let changes = [
            ("/version", json!(2)),
            ("/players/0/resources/corn", json!(-1)),
            ("/players/0/resources/skull", json!(1)),
            (
                "/players/0/resources/wood",
                json!(9_007_199_254_740_992_i64),
            ),
            ("/players/0/score", json!(0.1)),
            ("/players/0/id", json!(1)),
            ("/turnOrder", json!([0, 0.0, 2, 3])),
            ("/gears/palenque", json!([])),
            ("/gears/palenque/0", json!({"playerId":20, "dummy":false})),
            ("/gears/palenque/8", json!({"playerId":0, "dummy":false})),
            ("/gears/palenque/0", json!({"playerId":0, "dummy":true})),
            ("/skullSpaces/0", json!(0)),
            ("/jungle/2/wood", json!(1)),
            ("/jungle/3/corn", json!(3)),
            ("/foodDays", json!([14])),
            ("/age", json!(2)),
            ("/currentPlayer", json!(1)),
            ("/turn/count", json!(1)),
            (
                "/pending",
                json!({"title":"fake", "task":{"type":"invented"}, "after":[]}),
            ),
        ];
        for (path, replacement) in changes {
            let mut state = playing_fixture();
            *state.pointer_mut(path).unwrap() = replacement;
            assert!(!validate_game_state(&state), "{path}");
        }
        for field in ["firstPlayerClaimed", "pending"] {
            let mut state = setup_fixture();
            state.as_object_mut().unwrap().remove(field);
            assert!(!validate_game_state(&state), "missing {field}");
        }
        let mut state = playing_fixture();
        state["buildings"][1] = state["buildings"][0].clone();
        assert!(!validate_game_state(&state));
        let mut state = playing_fixture();
        state["players"][0]["wealthOffer"] = state["players"][1]["wealthOffer"].clone();
        state["players"][0]["wealth"] = state["players"][1]["wealth"].clone();
        assert!(!validate_game_state(&state));
    }

    #[test]
    fn rejects_utf16_overlong_names_and_optional_null_tasks_in_full_states() {
        let mut state = setup_fixture();
        state["players"][0]["name"] = json!("😀".repeat(50));
        assert!(validate_game_state(&state));
        state["players"][0]["name"] = json!("😀".repeat(51));
        assert!(!validate_game_state(&state));
        state["players"][0]["name"] = json!("\u{feff}\u{3000}");
        assert!(!validate_game_state(&state));
        let mut state = playing_fixture();
        state["pending"] = json!({"title":"action", "task":{"type":"action", "gear":"tikal", "position":1, "free":null}, "after":[]});
        assert!(!validate_game_state(&state));
    }

    #[test]
    fn enforces_exclusive_temple_tops_worker_counts_and_rotation_owner() {
        let mut state = playing_fixture();
        state["players"][0]["temples"]["chaac"] = json!(5);
        state["players"][1]["temples"]["chaac"] = json!(5);
        assert!(!validate_game_state(&state));
        let mut state = playing_fixture();
        for position in 0..4 {
            state["gears"]["palenque"][position] = json!({"playerId":0, "dummy":false});
        }
        assert!(!validate_game_state(&state));
        state["gears"]["palenque"][3] = Value::Null;
        assert!(validate_game_state(&state));
        state["firstPlayerClaimed"] = json!(0);
        assert!(!validate_game_state(&state));
        let mut state = playing_fixture();
        state["pending"] = json!({"title":"rotate", "task":{"type":"rotation"}, "after":[]});
        assert!(!validate_game_state(&state));
        state["firstPlayerClaimed"] = json!(1);
        assert!(!validate_game_state(&state));
        state["currentPlayer"] = json!(1);
        assert!(validate_game_state(&state));
    }

    #[test]
    fn finished_scores_require_consistent_quarter_points_and_final_day() {
        let mut state = playing_fixture();
        state["phase"] = json!("finished");
        state["round"] = json!(27);
        state["age"] = json!(2);
        state["foodDays"] = json!([8, 14, 21, 27]);
        let age_two = state["age2Deck"].as_array().unwrap().clone();
        state["buildings"] = json!(age_two[..6]);
        state["buildingDeck"] = json!(age_two[6..]);
        state["age2Deck"] = json!([]);
        state["players"][0]["score"] = json!(0.25);
        state["finalScores"] = json!((0..4).map(|id| json!({
            "playerId":id, "pointsBeforeFinal":0, "resourcePoints":if id == 0 {0.25} else {0.0},
            "skullPoints":0, "monumentPoints":0, "total":if id == 0 {0.25} else {0.0},
            "workersOnGears":0, "rank":if id == 0 {1} else {2}
        })).collect::<Vec<_>>());
        assert!(validate_game_state(&state));
        state["finalScores"][0]["total"] = json!(0.5);
        assert!(!validate_game_state(&state));
        state["finalScores"][0]["total"] = json!(0.25);
        state["finalScores"][0]["skullPoints"] = json!(0.25);
        state["finalScores"][0]["resourcePoints"] = json!(0);
        assert!(validate_game_state(&state));
        state["round"] = json!(26);
        assert!(!validate_game_state(&state));
    }
}
