use crate::utils::constants::{FIRST_FOOD_DAY, FOURTH_FOOD_DAY, SECOND_FOOD_DAY, THIRD_FOOD_DAY};
use crate::utils::increment::Increment;

use super::chichen_itza_skull::ChichenItzaSkull;
use super::food_day::{FoodDay, FoodDayStatus};
use super::jungle::Jungle;
use super::player::Player;
use super::resources::FieldSkulls;

#[derive(Debug, Default, PartialEq, PartialOrd)]
pub struct Round(u32);
impl Round {
    pub fn new(num: u32) -> Self {
        Self(num)
    }
}
impl Increment for Round {
    fn increment(&mut self) {
        if self.0 < FOURTH_FOOD_DAY {
            self.0.increment();
        }
    }
}

#[derive(Debug, Default, PartialEq, PartialOrd)]
pub struct Generation(pub u32);
impl Generation {
    pub fn new(num: u32) -> Self {
        Self(num)
    }
}
impl Increment for Generation {
    fn increment(&mut self) {
        if self.0 == 1 {
            self.0 = 2;
        }
    }
}

#[derive(Debug, Default)]
pub struct Game {
    round: Round,
    generation: Generation,
    corns: u32,
    pub jungle: Jungle,
    chichen_itza_skull: ChichenItzaSkull,
}

impl Game {
    pub fn new(number_of_players: u32) -> Result<Self, &'static str> {
        let corns = 0;
        Ok(Self {
            round: Round::new(1),
            generation: Generation::new(1),
            corns,
            jungle: Jungle::new(number_of_players)?,
            chichen_itza_skull: ChichenItzaSkull::new(),
        })
    }

    pub fn get_round(&self) -> u32 {
        self.round.0
    }

    pub fn next_round(&mut self) {
        self.round.increment();
    }

    pub fn get_generation(&self) -> u32 {
        self.generation.0
    }

    fn next_generation(&mut self) {
        self.generation.increment();
    }

    fn add_corns(&mut self) {
        self.corns += 1;
    }

    pub fn get_corns(&self) -> u32 {
        self.corns
    }

    // round終了時の処理をまとめた関数
    pub fn end_round(
        &mut self,
        food_day_status: &mut FoodDayStatus,
        players: &mut Vec<Player>,
        field_skull: &mut FieldSkulls,
    ) {
        if !food_day_status.first_food_day_done && self.get_round() >= FIRST_FOOD_DAY {
            food_day_status.food_day(FoodDay::First, players, field_skull);
        }
        if !food_day_status.second_food_day_done && self.get_round() >= SECOND_FOOD_DAY {
            food_day_status.food_day(FoodDay::Second, players, field_skull);
        }
        if !food_day_status.third_food_day_done && self.get_round() >= THIRD_FOOD_DAY {
            food_day_status.food_day(FoodDay::Third, players, field_skull);
        }
        if !food_day_status.fourth_food_day_done && self.get_round() >= FOURTH_FOOD_DAY {
            food_day_status.food_day(FoodDay::Fourth, players, field_skull);
        }
        self.next_round();
        if (self.generation.0 == 1) && (self.round.0 >= SECOND_FOOD_DAY + 1) {
            self.next_generation();
        }
        self.add_corns();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_increase_round() {
        let mut round = Round::new(1);
        round.increment();
        assert_eq!(round.0, 2);

        let mut round = Round::new(27);
        round.increment();
        assert_eq!(round.0, 27);
    }

    #[test]
    fn test_food_day_next_generation() {
        let number_of_players = 3;
        let mut game = Game::new(number_of_players).unwrap();
        let mut field_skull = FieldSkulls::new();
        let mut food_day_status = FoodDayStatus::new();
        let mut players: Vec<Player> = (1..=number_of_players)
            .map(|i| Player::new(format!("Player {}", i), i.into()))
            .collect();

        // FoodDay::Firstの境界テスト
        for _ in 0..7 {
            game.end_round(&mut food_day_status, &mut players, &mut field_skull);
        }
        // 7th roundの終了時は1回目のfood_dayではない
        assert_eq!(food_day_status.get_food_day_status(FoodDay::First), false);
        // 8th roundの終了時に1回目のfood_dayになる
        game.end_round(&mut food_day_status, &mut players, &mut field_skull);
        assert_eq!(food_day_status.get_food_day_status(FoodDay::First), true);

        // FoodDay::SecondとGeneration切り替わりの境界テスト
        // game.round.0 == 9
        for _ in 0..5 {
            game.end_round(&mut food_day_status, &mut players, &mut field_skull);
        }
        // game.round.0 == 14
        // 13th roundの終了時点（14th 開始時点）は2回目のfood_dayを終えていなく、第一世代である
        assert_eq!(food_day_status.get_food_day_status(FoodDay::Second), false);
        assert_eq!(game.get_generation(), 1);
        // 14th roundの終了時は2回目のfood_dayで、第二世代になる
        game.end_round(&mut food_day_status, &mut players, &mut field_skull);
        assert_eq!(food_day_status.get_food_day_status(FoodDay::Second), true);
        assert_eq!(game.get_generation(), 2);

        // FoodDay::Thirdの境界テスト
        for _ in 0..6 {
            game.end_round(&mut food_day_status, &mut players, &mut field_skull);
        }
        // game.round.0 == 21
        // 21th 開始時点は3回目のfood_dayを終えていない
        assert_eq!(food_day_status.get_food_day_status(FoodDay::Third), false);
        game.end_round(&mut food_day_status, &mut players, &mut field_skull);
        // 21th 終了後は3回目のfood_dayを終えている
        assert_eq!(food_day_status.get_food_day_status(FoodDay::Third), true);

        // FoodDay::Fourthの境界テスト
        // game.round.0 == 22
        for _ in 0..5 {
            game.end_round(&mut food_day_status, &mut players, &mut field_skull);
        }
        // game.round.0 == 27
        // 27th 開始時点は4回目のfood_dayを終えていない
        assert_eq!(food_day_status.get_food_day_status(FoodDay::Fourth), false);
        game.end_round(&mut food_day_status, &mut players, &mut field_skull);
        // 27th 終了後は4回目のfood_dayを終えている
        assert_eq!(food_day_status.get_food_day_status(FoodDay::Fourth), true);
    }
}
