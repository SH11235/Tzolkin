use crate::utils::constants::{FIRST_FOOD_DAY, FOURTH_FOOD_DAY, SECOND_FOOD_DAY, THIRD_FOOD_DAY};
use crate::utils::increment::Increment;

use super::player;

#[derive(Debug, PartialEq, PartialOrd)]
pub struct Round(pub u32);
pub trait ValidRound {
    fn is_valid(&self) -> bool;
}
impl ValidRound for Round {
    fn is_valid(&self) -> bool {
        self.0 >= 1 && self.0 <= FOURTH_FOOD_DAY
    }
}
impl Increment for Round {
    fn increment(&mut self) {
        if self.0 < FOURTH_FOOD_DAY {
            self.0.increment();
        }
    }
}
impl std::cmp::PartialEq<u32> for Round {
    fn eq(&self, other: &u32) -> bool {
        self.0 == *other
    }
}
impl std::cmp::PartialOrd<u32> for Round {
    fn partial_cmp(&self, other: &u32) -> Option<std::cmp::Ordering> {
        self.0.partial_cmp(other)
    }
}

#[derive(Debug, PartialEq, PartialOrd)]
pub struct Generation(pub u32);
pub trait ValidGeneration {
    fn is_valid(&self) -> bool;
}
impl ValidGeneration for Generation {
    fn is_valid(&self) -> bool {
        self.0 == 1 || self.0 == 2
    }
}
impl Increment for Generation {
    fn increment(&mut self) {
        if self.0 == 1 {
            self.0 = 2;
        }
    }
}
impl std::cmp::PartialEq<i32> for Generation {
    fn eq(&self, other: &i32) -> bool {
        self.0 == *other as u32
    }
}

#[derive(Debug)]
pub(crate) struct Game<
    T: ValidRound + Increment + std::cmp::PartialEq<u32> + std::cmp::PartialOrd<u32>,
    G: ValidGeneration + Increment + std::cmp::PartialEq<i32>,
> {
    round: T,
    generation: G,
    corns: u32,
}

impl<
        T: ValidRound + Increment + std::cmp::PartialEq<u32> + std::cmp::PartialOrd<u32>,
        G: ValidGeneration + Increment + std::cmp::PartialEq<i32>,
    > Game<T, G>
{
    pub fn new(round: T, generation: G) -> Result<Self, &'static str> {
        if !round.is_valid() {
            return Err("Invalid round value");
        }
        if !generation.is_valid() {
            return Err("Invalid generation value");
        }
        let corns = 0;
        Ok(Self {
            round,
            generation,
            corns,
        })
    }

    pub fn get_round(&self) -> &T {
        &self.round
    }

    fn next_round(&mut self) {
        self.round.increment();
    }

    pub fn get_generation(&self) -> &G {
        &self.generation
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

    fn is_food_day(&self) -> bool {
        self.round == FIRST_FOOD_DAY
            || self.round == SECOND_FOOD_DAY
            || self.round == THIRD_FOOD_DAY
            || self.round == FOURTH_FOOD_DAY
    }

    fn is_first_food_day(&self) -> bool {
        self.round == FIRST_FOOD_DAY
    }

    fn is_second_food_day(&self) -> bool {
        self.round == SECOND_FOOD_DAY
    }

    fn is_third_food_day(&self) -> bool {
        self.round == THIRD_FOOD_DAY
    }

    fn is_fourth_food_day(&self) -> bool {
        self.round == FOURTH_FOOD_DAY
    }

    // round終了時の処理をまとめた関数
    pub fn end_round(&mut self, players: &mut Vec<player::Player>) -> bool {
        // food_day処理
        if self.is_food_day() {
            players.iter_mut().for_each(|player| player.feed());
        }
        // 神殿判定: 資源
        if self.is_first_food_day() || self.is_third_food_day() {
            // players.iter_mut().for_each(|player| player.check_temple());
        }
        // 神殿判定: 得点
        if self.is_second_food_day() || self.is_fourth_food_day() {
            // players.iter_mut().for_each(|player| player.check_temple());
        }
        if self.round >= FOURTH_FOOD_DAY {
            true
        } else {
            self.next_round();
            if (self.generation == 1) && (self.round >= SECOND_FOOD_DAY) {
                self.next_generation();
            }
            self.add_corns();
            false
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_increase_round() {
        let mut round = Round(1);
        round.increment();
        assert_eq!(round.0, 2);

        let mut round = Round(27);
        round.increment();
        assert_eq!(round.0, 27);
    }
}
