mod game_object;

use game_object::game::{FoodDay, Game, Generation, Increment, Round};

// enum Position {
//     Hand,
//     Palenque(u32),
//     Yaxchilan(u32),
//     Tikal(u32),
//     Uxmal(u32),
//     ChichenItza(u32),
//     StartPlayer,
//     // QuickActions,
// }

// struct Jungle {
//     second_corn_tiles: u32,
//     third_wood_tiles: u32,
//     third_corn_tiles: u32,
//     fourth_wood_tiles: u32,
//     fourth_corn_tiles: u32,
//     fifth_wood_tiles: u32,
//     fifth_corn_tiles: u32,
// }

// struct Worker {
//     position: Position,
// }

// struct Technology {
//     agriculture: u32,
//     resource: u32,
//     construction: u32,
//     temple: u32,
// }

// struct Resource {
//     woods: u32,
//     stones: u32,
//     golds: u32,
// }

// struct Player {
//     name: String,
//     order: u32,
//     workers: Vec<Worker>,
//     technology: Technology,
//     corns: u32,
//     resource: Resource,
//     skulls: u32,
//     corn_tiles: u32,
//     wood_tiles: u32,
//     points: u32,
// }

fn main() {
    let mut game = Game::new(
        Round(1),
        "Alice".to_string(),
        vec![FoodDay(8), FoodDay(14), FoodDay(21)],
        Generation(1),
        100,
    )
    .unwrap();
    game.round.increment();
}
