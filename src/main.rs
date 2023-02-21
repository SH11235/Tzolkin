mod game_object;

use game_object::game::{Game, Generation, Increment, Round};

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
    let mut game = Game::new(Round(1), "Alice".to_string(), Generation(1)).unwrap();
    game.round.increment();
}
