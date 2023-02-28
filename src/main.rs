mod game_object;
mod utils;

use game_object::game::Game;

fn main() {
    let number_of_players = 4; // TODO inputで受け取る
    let mut game = Game::new(number_of_players).unwrap();
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
        game.players.iter().for_each(|player| {
            println!(
                "Name: {}, Acrive Workers: {}, Corns: {}, Points: {}",
                player.get_name(),
                player.get_active_workers(),
                player.get_corns(),
                player.get_points(),
            );
            // TODO playerの行動を実装
        });
        let is_end = game.end_round();
        if is_end {
            break;
        }
    }
}
