mod game_object;
mod utils;

use game_object::game::{Game, Generation, Round};
use game_object::player::Player;

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
    let mut game = Game::new(Round(1), Generation(1)).unwrap();
    // start
    print!("game start");
    let player1 = Player::new("Player 1".to_string(), 1);
    let player2 = Player::new("Player 2".to_string(), 2);
    let mut players = vec![player1, player2];

    loop {
        println!(
            "Round: {}, Generation: {}, Corns: {}",
            game.get_round().0,
            game.get_generation().0,
            game.get_corns()
        );
        // print players
        players.iter().for_each(|player| {
            println!(
                "Name: {}, Acrive Workers: {}, Corns: {}",
                player.get_name(),
                player.get_active_workers(),
                player.get_corns(),
            );
        });
        let is_end = game.end_round(&mut players);
        if is_end {
            break;
        }
    }
}
