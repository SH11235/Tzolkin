use serde::Serialize;
use std::sync::Mutex;
use tauri::State;

mod game_object;
mod utils;

use game_object::{
    food_day::FoodDayStatus,
    game::Game,
    player::{Player, PlayerColor},
    resources::FieldSkulls,
};

// Prevents additional console window on Windows in release, DO NOT REMOVE!!
#[cfg_attr(not(debug_assertions), windows_subsystem = "windows")]
#[derive(Serialize)]
struct AppState {
    game_players: Mutex<Vec<Player>>,
}

#[tauri::command]
fn set_players(number: u8, app_state: State<AppState>) -> Result<Vec<Player>, String> {
    if (number > 0) && (number < 5) {
        println!("number of players: {}", number);
        let players: Vec<Player> = (1..=number)
            .map(|i| Player::new(format!("Player {}", i), PlayerColor::from(i), i.into()))
            .collect();
        let mut game_players = app_state.game_players.lock().unwrap();
        *game_players = players.clone();
        Ok(players)
    } else {
        Err("number of players must be between 1 and 4".to_string())
    }
}

fn main() {
    let players: Vec<Player> = Vec::new();
    let app_state = AppState {
        game_players: Mutex::new(players),
    };

    tauri::Builder::default()
        .manage(app_state)
        .invoke_handler(tauri::generate_handler![set_players])
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
