mod game_object;
mod utils;

use game_object::game::{Game, Generation, Round};

// struct Jungle {
//     second_corn_tiles: u32,
//     third_wood_tiles: u32,
//     third_corn_tiles: u32,
//     fourth_wood_tiles: u32,
//     fourth_corn_tiles: u32,
//     fifth_wood_tiles: u32,
//     fifth_corn_tiles: u32,
// }

fn main() {
    let number_of_players = 4; // TODO inputで受け取る
    let mut game = Game::new(Round(1), Generation(1), number_of_players).unwrap();
    // start
    print!("game start");
    loop {
        println!(
            "Round: {}, Generation: {}, Corns: {}",
            game.get_round().0,
            game.get_generation().0,
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
