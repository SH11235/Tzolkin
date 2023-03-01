mod game_object;
mod utils;

use crossterm::{
    cursor::{Hide, MoveTo, Show},
    event::{read, Event, KeyCode, KeyEvent},
    execute,
    style::{Color, Print, SetBackgroundColor},
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
    Result,
};
use game_object::{food_day::FoodDayStatus, game::Game, player::Player, resources::FieldSkulls};
use utils::constants::FOURTH_FOOD_DAY;

fn view<T: std::io::Write>(
    output: &mut T,
    game: &Game,
) -> Result<()> {
    execute!(
        output,
        SetBackgroundColor(Color::Black),
        Print(format!(
            "Round: {}, Generation: {}, Corns: {}",
            game.get_round(),
            game.get_generation(),
            game.get_corns()
        ))
    )?;
    Ok(())
}


fn main() -> Result<()> {
    let number_of_players = 1; // TODO inputで受け取る
    let mut game = Game::new(number_of_players).unwrap();
    let mut field_skull = FieldSkulls::new();
    let mut food_day_status = FoodDayStatus::new();
    let mut players: Vec<Player> = (1..=number_of_players)
        .map(|i| Player::new(format!("Player {}", i), i.into()))
        .collect();

    enable_raw_mode()?;
    execute!(std::io::stderr(), Hide, EnterAlternateScreen)?;
    // start
    print!("game start\n");
    while game.get_round() < FOURTH_FOOD_DAY + 1 {
        execute!(
            std::io::stderr(),
            Print(format!("Round {} start\n", game.get_round()))
        )?;
        execute!(
            std::io::stderr(),
            Print(format!(
                "Round: {}, Generation: {}, Corns: {}",
                game.get_round(),
                game.get_generation(),
                game.get_corns()
            ))
        )?;
        // players action
        players.iter().for_each(|player| {
            // TODO playerの行動を実装
        });
        game.end_round(&mut food_day_status, &mut players, &mut field_skull);
        players.iter().for_each(|player| {
            execute!(
                std::io::stderr(),
                Print(format!(
                    "Name: {}, Acrive Workers: {}, Corns: {}, Points: {}\n",
                    player.get_name(),
                    player.get_active_workers(),
                    player.get_corns(),
                    player.get_points(),
                ))
            ).unwrap();
        });
        execute!(
            std::io::stderr(),
            Print(format!("Round {} end\n", game.get_round() - 1))
        )?;
    }
    print!("game end\n");
    players.iter().for_each(|player| {
        // TODO playerの資源、髑髏、モニュメントを得点に変換する
        println!(
            "Name: {}, Acrive Workers: {}, Corns: {}, Points: {}\n",
            player.get_name(),
            player.get_active_workers(),
            player.get_corns(),
            player.get_points(),
        );
    });
    Ok(())
}
