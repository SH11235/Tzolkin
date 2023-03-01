mod game_object;
mod utils;

use game_object::game::Game;

use crate::game_object::{food_day::FoodDayStatus, player::Player, resources::FieldSkulls};

fn main() {
    let number_of_players = 4; // TODO inputで受け取る
    let mut game = Game::new(number_of_players).unwrap();
    let mut field_skull = FieldSkulls::new();
    let mut food_day_status = FoodDayStatus::new();
    let mut players: Vec<Player> = (1..=number_of_players)
        .map(|i| Player::new(format!("Player {}", i), i.into()))
        .collect();
    // start
    print!("game start");
    loop {
        println!(
            "Round: {}, Generation: {}, Corns: {}",
            game.get_round(),
            game.get_generation(),
            game.get_corns()
        );
        // players action
        players.iter().for_each(|player| {
            println!(
                "Name: {}, Acrive Workers: {}, Corns: {}, Points: {}",
                player.get_name(),
                player.get_active_workers(),
                player.get_corns(),
                player.get_points(),
            );
            // TODO playerの行動を実装
        });
        let is_end = game.end_round(&mut food_day_status, &mut players, &mut field_skull);
        if is_end {
            break;
        }
    }
}
