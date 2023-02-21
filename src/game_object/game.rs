pub(crate) const FIRST_FOOD_DAY: u32 = 8;
pub(crate) const SECOND_FOOD_DAY: u32 = 14;
pub(crate) const THIRD_FOOD_DAY: u32 = 21;
pub(crate) const FOURTH_FOOD_DAY: u32 = 27;

#[derive(Debug)]
pub struct Round(pub u32);
impl ValidRound for Round {
    fn is_valid(&self) -> bool {
        self.0 >= 1 && self.0 <= FOURTH_FOOD_DAY
    }
}
pub trait Increment {
    fn increment(&mut self);
}
impl Increment for u32 {
    fn increment(&mut self) {
        *self += 1;
    }
}
impl Increment for Round {
    fn increment(&mut self) {
        if self.0 < FOURTH_FOOD_DAY {
            self.0.increment();
        }
    }
}

#[derive(Debug)]
pub struct Generation(pub u32);
pub trait ValidRound {
    fn is_valid(&self) -> bool;
}
pub trait ValidGeneration {
    fn is_valid(&self) -> bool;
}
impl ValidGeneration for Generation {
    fn is_valid(&self) -> bool {
        self.0 == 1 || self.0 == 2
    }
}

#[derive(Debug)]
pub(crate) struct Game<T: ValidRound, G: ValidGeneration> {
    pub round: T,
    pub turn_player: String,
    pub food_days: [u32; 4],
    pub generation: G,
}

impl<T: ValidRound, G: ValidGeneration> Game<T, G> {
    const FOOD_DAYS: [u32; 4] = [
        FIRST_FOOD_DAY,
        SECOND_FOOD_DAY,
        THIRD_FOOD_DAY,
        FOURTH_FOOD_DAY,
    ];
    pub fn new(round: T, turn_player: String, generation: G) -> Result<Self, &'static str> {
        if !round.is_valid() {
            return Err("Invalid round value");
        }
        if !generation.is_valid() {
            return Err("Invalid generation value");
        }
        Ok(Self {
            round,
            turn_player,
            food_days: Self::FOOD_DAYS,
            generation,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_increase_round() {
        let mut game = Game::new(Round(1), "Alice".to_string(), Generation(1)).unwrap();
        assert_eq!(game.round.0, 1);
        game.round.increment();
        assert_eq!(game.round.0, 2);
    }
}
