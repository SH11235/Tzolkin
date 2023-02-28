use crate::{game_object::player::Player, utils::constants::MAX_TECHNOLOGY_LEVEL};

pub struct Yaxchilan(u32);
impl Yaxchilan {
    fn action(&self, player: &mut Player) -> Result<(), String> {
        let action_space = self.0;
        match action_space {
            1 => {
                player.resource.woods.0 += 1;
                if player.technology.get_resource_level() >= 1 {
                    player.resource.woods.0 += 1;
                }
            }
            2 => {
                player.resource.stones.0 += 1;
                player.corns += 1;
                if player.technology.get_resource_level() >= 2 {
                    player.resource.stones.0 += 1;
                }
            }
            3 => {
                player.resource.golds.0 += 1;
                player.corns += 2;
                if player.technology.get_resource_level() == MAX_TECHNOLOGY_LEVEL {
                    player.resource.golds.0 += 1;
                }
            }
            4 => {
                player.resource.skulls.0 += 1;
                if player.technology.get_temple_level() == MAX_TECHNOLOGY_LEVEL {
                    player.resource.skulls.0 += 1;
                }
            }
            5 => {
                player.resource.stones.0 += 1;
                player.resource.golds.0 += 1;
                player.corns += 2;
                if player.technology.get_resource_level() >= 2 {
                    player.resource.stones.0 += 1;
                }
                if player.technology.get_resource_level() == MAX_TECHNOLOGY_LEVEL {
                    player.resource.golds.0 += 1;
                }
            }
            _ => {
                return Err("ヤシュチランのアクションスペース番号が不正です".to_string());
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::game_object::action_space::WorkerPosition;

    #[test]
    fn test_yaxchilan_action_space_1() {
        let mut player = Player::new("test".to_string(), 1);
        let action_space_num = 1;
        player.workers[0].set_position(WorkerPosition::Yaxchilan(action_space_num));
        Yaxchilan(action_space_num).action(&mut player).unwrap();
        assert_eq!(player.resource.woods.0, 1);
        assert_eq!(player.resource.stones.0, 0);
        assert_eq!(player.resource.golds.0, 0);
        assert_eq!(player.resource.skulls.0, 0);
        assert_eq!(player.corns, 0);
    }

    // #[test]
    // fn test_yaxchilan_action_space_2() {
    //     let mut player = Player::new();
    //     let mut yaxchilan = Yaxchilan(2);
    //     yaxchilan.action(&mut player).unwrap();
    //     assert_eq!(player.resource.woods.0, 0);
    //     assert_eq!(player.resource.stones.0, 1);
    //     assert_eq!(player.resource.golds.0, 0);
    //     assert_eq!(player.resource.skulls.0, 0);
    //     assert_eq!(player.corns, 1);
    // }

    // #[test]
    // fn test_yaxchilan_action_space_3() {
    //     let mut player = Player::new();
    //     let mut yaxchilan = Yaxchilan(3);
    //     yaxchilan.action(&mut player).unwrap();
    //     assert_eq!(player.resource.woods.0, 0);
    //     assert_eq!(player.resource.stones.0, 0);
    //     assert_eq!(player.resource.golds.0, 1);
    //     assert_eq!(player.resource.skulls.0, 0);
    //     assert_eq!(player.corns, 2);
    // }

    // #[test]
    // fn test_yaxchilan_action_space_4() {
    //     let mut player = Player::new();
    //     let mut yaxchilan = Yaxchilan(4);
    //     yaxchilan.action(&mut player).unwrap();
    //     assert_eq!(player.resource.woods.0, 0);
    //     assert_eq!(player.resource.stones.0, 0);
    //     assert_eq!(player.resource.golds.0, 0);
    //     assert_eq!(player.resource.skulls.0, 1);
    //     assert_eq!(player.corns, 0);
    // }
}