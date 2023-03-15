mod game_object;
mod utils;

use game_object::{food_day::FoodDayStatus, game::Game, player::Player, resources::FieldSkulls};
use utils::constants::FOURTH_FOOD_DAY;

use crate::game_object::{
    action_space::{palenque::PalenqueSpace, WorkerPosition},
    player::get_color,
};

// Prevents additional console window on Windows in release, DO NOT REMOVE!!
#[cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

static mut NUMBER_OF_PLAYERS: u8 = 0;

#[tauri::command]
fn set_number_of_players(number: u8) -> Result<u8, String> {
    if (number > 0) && (number < 5) {
        unsafe {
            NUMBER_OF_PLAYERS = number;
        }
        println!("number of players: {}", number);
        Ok(number)
    } else {
        Err("number of players must be between 1 and 4".to_string())
    }
}

fn main() {
    tauri::Builder::default()
        .invoke_handler(tauri::generate_handler![set_number_of_players])
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
