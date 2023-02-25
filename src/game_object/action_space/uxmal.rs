use crate::game_object::{
    chichen_itza_skull::ChichenItzaSkull, jungle::Jungle, player::Player, temple::TempleName,
};

pub struct Uxmal(u32);
impl Uxmal {
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
