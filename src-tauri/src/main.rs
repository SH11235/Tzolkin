use serde::{Deserialize, Serialize};
use std::sync::Mutex;
use tauri::State;
use utils::constants::{CHAAC, KUKULKAN, QUETZALCOATL};

mod game_object;
mod utils;

use game_object::{
    first_resource_tiles::Tile,
    food_day::FoodDayStatus,
    game::Game,
    player::{
        technology::{TechnologyProgressReward, TechnologyType},
        Player, PlayerColor,
    },
    resources::{FieldSkulls, Resource},
    temple::Temple,
};

use crate::game_object::first_resource_tiles::shuffle_tile_list;

// Prevents additional console window on Windows in release, DO NOT REMOVE!!
#[cfg_attr(not(debug_assertions), windows_subsystem = "windows")]
#[derive(Serialize)]
struct AppState {
    game_players: Mutex<Vec<Player>>,
    field_skulls: Mutex<FieldSkulls>,
}

#[tauri::command]
fn get_players(app_state: State<AppState>) -> Vec<Player> {
    let players = app_state.game_players.lock().unwrap();
    players.clone()
}

#[tauri::command]
fn set_players(number: u32, app_state: State<AppState>) -> Result<Vec<Player>, String> {
    if (number > 0) && (number < 5) {
        println!("number of players: {}", number);
        let players: Vec<Player> = (1..=number)
            .map(|i| Player::new(i, format!("Player {}", i), PlayerColor::from(i), i))
            .collect();
        let mut game_players = app_state.game_players.lock().unwrap();
        *game_players = players.clone();
        Ok(players)
    } else {
        Err("number of players must be between 1 and 4".to_string())
    }
}

#[tauri::command]
fn add_worker(player_id: u32, app_state: State<AppState>) -> Result<Player, String> {
    let mut player = get_player(player_id, app_state.clone())?;
    player.add_worker();
    Ok(player.clone())
}

#[tauri::command]
fn get_first_resource_tiles(app_state: State<AppState>) -> Vec<Vec<&Tile>> {
    let players = app_state.game_players.lock().unwrap();
    let tile_list = shuffle_tile_list();
    // TODO ダミーworker init処理

    // プレイヤー数の長さのvecでtile_listから4つずつ取り出す
    let mut tile_list_vec: Vec<Vec<&Tile>> = Vec::new();
    for i in 0..players.len() {
        let mut tile_list_slice: Vec<&Tile> = Vec::new();
        for j in 0..4 {
            tile_list_slice.push(&tile_list[i * 4 + j]);
        }
        tile_list_vec.push(tile_list_slice);
    }
    tile_list_vec
}

#[tauri::command]
fn add_resource(
    player_id: u32,
    resource_type: String,
    amount: u32,
    app_state: State<AppState>,
) -> Result<Player, String> {
    let mut player = get_player(player_id, app_state.clone())?;
    match resource_type.as_str() {
        "corn" => {
            player.corns += amount;
        }
        "wood" => {
            player.resource.woods.add(amount);
        }
        "stone" => {
            player.resource.stones.add(amount);
        }
        "gold" => {
            player.resource.golds.add(amount);
        }
        "skull" => {
            let mut field_skulls = app_state.field_skulls.lock().unwrap();
            field_skulls.decrease_skulls(amount);
            player.resource.skulls.add(amount);
        }
        _ => return Err("resource type not found".to_string()),
    }
    Ok(player.clone())
}

#[tauri::command]
fn get_field_skulls(app_state: State<AppState>) -> u32 {
    app_state
        .field_skulls
        .lock()
        .unwrap()
        .get_remaining_skulls()
}

#[derive(Deserialize)]
struct RewardOption {
    point: Option<u32>,
    faith: Option<String>,
    resource: Option<Vec<String>>,
    skull: Option<u32>,
}

#[tauri::command]
fn raise_technology_level(
    player_id: u32,
    technology_type: String,
    reward_option: Option<RewardOption>,
    app_state: State<AppState>,
) -> Result<Player, String> {
    let mut player = get_player(player_id, app_state.clone())?;
    let reward = match technology_type.as_str() {
        "agriculture" => player.technology.progress(TechnologyType::Agriculture),
        "resource" => player.technology.progress(TechnologyType::Resource),
        "construction" => player.technology.progress(TechnologyType::Construction),
        "temple" => player.technology.progress(TechnologyType::Temple),
        _ => return Err("technology type not found".to_string()),
    };
    if let Some(reward) = reward {
        match reward_option {
            Some(reward_option) => match reward {
                TechnologyProgressReward::Point(point) => {
                    if let Some(_reward_point) = reward_option.point {
                        player.add_points(3.0);
                    } else {
                        return Err("reward point is not correct".to_string());
                    }
                    player.add_points(point);
                }
                TechnologyProgressReward::Faith => {
                    if let Some(reward_faith) = reward_option.faith {
                        match reward_faith.as_str() {
                            CHAAC => {
                                player.temple_faith.chaac.raise_faith();
                            }
                            QUETZALCOATL => {
                                player.temple_faith.quetzalcoatl.raise_faith();
                            }
                            KUKULKAN => {
                                player.temple_faith.kukulkan.raise_faith();
                            }
                            _ => return Err("reward faith is not correct".to_string()),
                        }
                    } else {
                        return Err("reward faith is not selected".to_string());
                    }
                }
                TechnologyProgressReward::Resource => {
                    if let Some(reward_resource) = reward_option.resource {
                        if reward_resource.len() != 2 {
                            return Err("reward resource is not correct".to_string());
                        }
                        for resource in reward_resource {
                            match resource.as_str() {
                                "wood" => {
                                    player.resource.woods.add(1);
                                }
                                "stone" => {
                                    player.resource.stones.add(1);
                                }
                                "gold" => {
                                    player.resource.golds.add(1);
                                }
                                _ => return Err("reward resource is not correct".to_string()),
                            }
                        }
                    } else {
                        return Err("reward resources are not selected".to_string());
                    }
                }
                TechnologyProgressReward::Skull => {
                    if let Some(_reward_skull) = reward_option.skull {
                        player.resource.skulls.add(1);
                    } else {
                        return Err("reward skull is not correct".to_string());
                    }
                }
            },
            None => return Err("reward option is not selected".to_string()),
        }
    }
    Ok(player.clone())
}

#[tauri::command]
fn raise_temple_faith(
    player_id: u32,
    temple_type: String,
    amount: u32,
    app_state: State<AppState>,
) -> Result<Player, String> {
    let mut player = get_player(player_id, app_state.clone())?;
    match temple_type.as_str() {
        CHAAC => {
            for _ in 0..amount {
                player.raise_chaac_faith();
            }
        }
        QUETZALCOATL => {
            for _ in 0..amount {
                player.raise_quetzalcoatl_faith();
            }
        }
        KUKULKAN => {
            for _ in 0..amount {
                player.raise_kukulkan_faith();
            }
        }
        _ => return Err("temple type not found".to_string()),
    }
    println!("player: {:?}", player);
    Ok(player.clone())
}

fn get_player(player_id: u32, app_state: State<AppState>) -> Result<Player, String> {
    let mut players = app_state.game_players.lock().unwrap();
    let player = players.iter_mut().find(|p| p.get_id() == player_id);
    if let Some(player) = player {
        Ok(player.clone())
    } else {
        Err(format!("player {} not found", player_id))
    }
}

fn main() {
    let players: Vec<Player> = Vec::new();
    let app_state = AppState {
        game_players: Mutex::new(players),
        field_skulls: Mutex::new(FieldSkulls::new()),
    };

    tauri::Builder::default()
        .manage(app_state)
        .invoke_handler(tauri::generate_handler![
            get_players,
            set_players,
            add_worker,
            get_first_resource_tiles,
            add_resource,
            get_field_skulls,
            raise_technology_level,
            raise_temple_faith
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}

// fn main() -> Result<()> {
// let mut game = Game::new(number_of_players).unwrap();
// let mut field_skull = FieldSkulls::new();
// let mut food_day_status = FoodDayStatus::new();
// let mut players: Vec<Player> = (1..=number_of_players)
//     .map(|i| Player::new(format!("Player {}", i), get_color(i).unwrap(), i.into()))
//     .collect();
//     // start
//     print!("game start\n");
//     while game.get_round() < FOURTH_FOOD_DAY + 1 {
//         // サンプルコード
//         if game.get_round() == 1 {
//             players[0].workers[0].set_position(WorkerPosition::Palenque(PalenqueSpace(0)));
//         }
//         // players action
//         players.iter().for_each(|player| {
//             // TODO playerの行動を実装
//         });
//         game.end_round(&mut food_day_status, &mut players, &mut field_skull);
//         players.iter().for_each(|player| {
//             execute!(
//                 std::io::stderr(),
//                 Print(format!(
//                     "Name: {}, Workers: {:?}, Corns: {}, Points: {}\n",
//                     player.get_name(),
//                     player.workers,
//                     player.get_corns(),
//                     player.get_points(),
//                 ))
//             ).unwrap();
//         });
//     }
//     print!("game end\n");
//     players.iter().for_each(|player| {
//         // TODO playerの資源、髑髏、モニュメントを得点に変換する
//         println!(
//             "Name: {}, Acrive Workers: {}, Corns: {}, Points: {}\n",
//             player.get_name(),
//             player.get_active_workers(),
//             player.get_corns(),
//             player.get_points(),
//         );
//     });
//     Ok(())
// }
