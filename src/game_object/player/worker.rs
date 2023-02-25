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
