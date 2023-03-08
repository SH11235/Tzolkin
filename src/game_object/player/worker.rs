use crate::game_object::action_space::WorkerPosition;

pub const worker_set_cost: [u32; 6] = [0, 1, 2, 3, 4, 5];

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

    pub fn calculate_set_cost(number: u32, space_number: u32) -> u32 {
        worker_set_cost[number as usize] + space_number
    }

    pub fn can_pick_up_worker(&self) -> bool {
        match self.position {
            WorkerPosition::Hand => false,
            WorkerPosition::StartPlayer => false,
            WorkerPosition::Locked => false,
            _ => true,
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
