#[derive(Debug)]
pub struct Round(pub u32);
#[derive(Debug)]
pub struct FoodDay(pub u32);
#[derive(Debug)]
pub struct Generation(pub u32);
pub trait ValidRound {
    fn is_valid(&self) -> bool;
}

pub trait ValidFoodDay {
    fn is_valid(&self) -> bool;
}

pub trait ValidGeneration {
    fn is_valid(&self) -> bool;
}

impl ValidRound for Round {
    fn is_valid(&self) -> bool {
        self.0 >= 1 && self.0 <= 27
    }
}

impl ValidFoodDay for FoodDay {
    fn is_valid(&self) -> bool {
        self.0 == 8 || self.0 == 14 || self.0 == 21 || self.0 == 27
    }
}

impl ValidGeneration for Generation {
    fn is_valid(&self) -> bool {
        self.0 == 1 || self.0 == 2
    }
}

#[derive(Debug)]
pub(crate) struct Game<T: ValidRound, F: ValidFoodDay, G: ValidGeneration> {
    pub round: T,
    pub turn_player: String,
    pub food_days: Vec<F>,
    pub generation: G,
    pub corn_stock: u32,
}

impl<T: ValidRound, F: ValidFoodDay, G: ValidGeneration> Game<T, F, G> {
    pub fn new(
        round: T,
        turn_player: String,
        food_days: Vec<F>,
        generation: G,
        corn_stock: u32,
    ) -> Result<Self, &'static str> {
        if !round.is_valid() {
            return Err("Invalid round value");
        }
        if !generation.is_valid() {
            return Err("Invalid generation value");
        }
        for food_day in &food_days {
            if !food_day.is_valid() {
                return Err("Invalid food day value");
            }
        }
        Ok(Self {
            round,
            turn_player,
            food_days,
            generation,
            corn_stock,
        })
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
        self.0.increment();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_increase_round() {
        let mut game = Game::new(
            Round(1),
            "Alice".to_string(),
            vec![FoodDay(8), FoodDay(14), FoodDay(21)],
            Generation(1),
            100,
        )
        .unwrap();
        assert_eq!(game.round.0, 1);
        game.round.increment();
        assert_eq!(game.round.0, 2);
    }
}
