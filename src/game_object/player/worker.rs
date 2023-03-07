use crate::game_object::action_space::WorkerPosition;

#[derive(Debug)]
pub struct Worker {
    position: WorkerPosition,
}
impl Worker {
    pub fn new() -> Self {
        Self {
            position: WorkerPosition::Hand,
        }
    }
    pub fn locked_worker() -> Self {
        Self {
            position: WorkerPosition::Locked,
        }
    }

    pub fn next_space(&mut self) {
        if !self.position.next_space() {
            self.back_to_hand();
        }
    }

    pub fn can_pick_up_worker(&self) -> Result<(), String>{
        match self.position {
            WorkerPosition::Hand => {
                Err("You can't pick up a worker from your hand".to_string())
            }
            WorkerPosition::StartPlayer => {
                Err("You can't pick up a worker from your start player".to_string())
            }
            WorkerPosition::Locked => {
                Err("You can't pick up a worker from your locked worker".to_string())
            }
            _ => Ok(()),
        }
    }

    pub fn back_to_hand(&mut self) {
        self.position = WorkerPosition::Hand;
    }

    pub fn get_position(&self) -> &WorkerPosition {
        &self.position
    }

    pub fn set_position(&mut self, position: WorkerPosition) {
        self.position = position;
    }
}
