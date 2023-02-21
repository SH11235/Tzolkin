use crate::utils::constants::{
    FIRST_FOOD_DAY, FOURTH_FOOD_DAY, SECOND_FOOD_DAY, THIRD_FOOD_DAY,
};
use crate::utils::increment::Increment;

#[derive(Debug)]
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
impl Round {
    pub fn as_u32(&self) -> u32 {
        self.0
    }
}

#[derive(Debug)]
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

#[derive(Debug)]
pub(crate) struct Game<
    T: ValidRound + Increment + std::cmp::PartialEq<u32>,
    G: ValidGeneration + Increment,
> {
    round: T,
    generation: G,
}

impl<T: ValidRound + Increment + std::cmp::PartialEq<u32>, G: ValidGeneration + Increment>
    Game<T, G>
{
    pub fn new(round: T, generation: G) -> Result<Self, &'static str> {
        if !round.is_valid() {
            return Err("Invalid round value");
        }
        if !generation.is_valid() {
            return Err("Invalid generation value");
        }
        Ok(Self { round, generation })
    }

    pub fn get_round(&self) -> &T {
        &self.round
    }

    pub fn next_round(&mut self) {
        self.round.increment();
    }

    pub fn get_generation(&self) -> &G {
        &self.generation
    }

    pub fn next_generation(&mut self) {
        self.generation.increment();
    }

    pub fn is_food_day(&self) -> bool {
        self.round == FIRST_FOOD_DAY
            || self.round == SECOND_FOOD_DAY
            || self.round == THIRD_FOOD_DAY
            || self.round == FOURTH_FOOD_DAY
    }

    pub fn is_first_food_day(&self) -> bool {
        self.round == FIRST_FOOD_DAY
    }

    pub fn is_second_food_day(&self) -> bool {
        self.round == SECOND_FOOD_DAY
    }

    pub fn is_third_food_day(&self) -> bool {
        self.round == THIRD_FOOD_DAY
    }

    pub fn is_fourth_food_day(&self) -> bool {
        self.round == FOURTH_FOOD_DAY
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
