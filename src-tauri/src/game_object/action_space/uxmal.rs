use crate::game_object::{
    chichen_itza_skull::ChichenItzaSkull, jungle::Jungle, player::Player, temple::TempleName,
};

use super::ActionSpace;

#[derive(Debug)]
pub struct UxmalSpace(u32);
impl ActionSpace for UxmalSpace {
    fn get_space(&self) -> u32 {
        self.0
    }
    fn next_space(&mut self) {
        self.0 += 1;
    }
}
impl UxmalSpace {
    fn action(
        &self,
        player: &mut Player,
        // worker: &mut Worker,
        // jungle: Option<&mut Jungle>,
        // technology: Option<TechnologyType>,
        // _target_temple: Option<TempleName>,
        // chichen_itza_skull: Option<&mut ChichenItzaSkull>,
    ) -> Result<(), String> {
        Ok(())
    }
}
