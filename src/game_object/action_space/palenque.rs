use crate::{
    game_object::{
        jungle::Jungle,
        player::{Player},
    },
    utils::constants::MAX_TECHNOLOGY_LEVEL,
};

pub enum CornOrWood {
    Corn,
    Wood,
}

pub struct Palenque(u32);
impl Palenque {
    fn action(
        &self,
        player: &mut Player,
        corn_or_wood: Option<CornOrWood>,
        jungle: Option<&mut Jungle>,
    ) -> Result<(), String> {
        let action_space = self.0;
        let corn_or_wood = match corn_or_wood {
            Some(corn_or_wood) => corn_or_wood,
            None => return Err("コーンか木を選択してください".to_string()),
        };
        let jungle = match jungle {
            Some(jungle) => jungle,
            None => return Err("ジャングルの状態が不明です".to_string()),
        };
        match action_space {
            1 => {
                player.corns += 3;
                if player.technology.get_agriculture_level() >= 2 {
                    player.corns += 1;
                }
                Ok(())
            }
            2 => {
                if jungle.corn_tiles(2) == 0 && player.technology.get_agriculture_level() <= 1 {
                    return Err("コーンタイルがありません".to_string());
                }
                if jungle.corn_tiles(2) > 0 {
                    jungle.decrease_corn_tiles(2)?;
                    player.corn_tiles += 1;
                }
                player.corns += 4;
                if player.technology.get_agriculture_level() >= 1 {
                    player.corns += 1;
                }
                if player.technology.get_agriculture_level() == MAX_TECHNOLOGY_LEVEL {
                    player.corns += 2;
                }
                Ok(())
            }
            3 | 4 | 5 => match corn_or_wood {
                CornOrWood::Corn => {
                    if jungle.corn_tiles(self.0) == 0 && player.technology.get_agriculture_level() <= 2 {
                        return Err("コーンタイルがありません".to_string());
                    }
                    if jungle.corn_tiles(action_space) > 0 {
                        jungle.decrease_corn_tiles(action_space)?;
                        player.corn_tiles += 1;
                    }
                    let get_corns = match action_space {
                        3 => 5,
                        4 => 7,
                        5 => 9,
                        _ => 0,
                    };
                    player.corns += get_corns;
                    if player.technology.get_agriculture_level() >= 2 {
                        player.corns += 1;
                    }
                    if player.technology.get_agriculture_level() == MAX_TECHNOLOGY_LEVEL {
                        player.corns += 2;
                    }
                    Ok(())
                }
                CornOrWood::Wood => {
                    if jungle.wood_tiles(self.0) == 0 {
                        return Err("木材タイルがありません".to_string());
                    }
                    if jungle.wood_tiles(action_space) > 0 {
                        jungle.decrease_wood_tiles(action_space)?;
                        player.wood_tiles += 1;
                    }
                    let get_woods = match action_space {
                        3 => 2,
                        4 => 3,
                        5 => 4,
                        _ => 0,
                    };
                    player.resource.woods.0 += get_woods;
                    if player.technology.get_resource_level() >= 1 {
                        player.resource.woods.0 += 1;
                    }
                    Ok(())
                }
            },
            _ => {
                return Err("パレンケのアクションスペース番号が不正です".to_string());
            }
        }
    }
}
