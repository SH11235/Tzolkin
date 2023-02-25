use crate::game_object::{
    chichen_itza_skull::ChichenItzaSkull, player::Player, temple::TempleName,
};

pub struct ChichenItza(u32);
impl ChichenItza {
    fn action(
        &self,
        player: &mut Player,
        target_temple: Option<TempleName>,
        chichen_itza_skull: Option<&mut ChichenItzaSkull>,
    ) -> Result<(), String> {
        Ok(())
    }
}
