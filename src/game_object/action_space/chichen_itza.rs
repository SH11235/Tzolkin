// use crate::game_object::{
//     chichen_itza_skull::ChichenItzaSkull, player::Player, temple::TempleName,
// };

use super::ActionSpace;

#[derive(Debug)]
pub struct ChichenItzaSpace(u32);
impl ActionSpace for ChichenItzaSpace {
    fn get_space(&self) -> u32 {
        self.0
    }
    fn next_space(&mut self) {
        self.0 += 1;
    }
}
impl ChichenItzaSpace {
    fn action(
        &self,
        // player: &mut Player,
        // target_temple: Option<TempleName>,
        // chichen_itza_skull: Option<&mut ChichenItzaSkull>,
    ) -> Result<(), String> {
        Ok(())
    }
}
