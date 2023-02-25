use crate::{
    game_object::{
        player::{
            technology::{TechnologyProgressReward, TechnologyType},
            Player,
        },
        resources::FieldSkulls,
        temple::TempleName,
    },
    utils::constants::{
        MAX_CHAAC_RANK, MAX_KUKULKAN_RANK, MAX_QUETZALCOATL_RANK, MAX_TECHNOLOGY_LEVEL,
    },
};

pub struct Tikal(u32);
pub enum ResourceOption {
    Wood,
    Stone,
    Gold,
}
impl Tikal {
    fn technology_level_action(
        &self,
        player: &mut Player,
        field_skells: &mut FieldSkulls,
        technology_type: Option<TechnologyType>,
        target_temple: Option<TempleName>,
        resources: Option<[ResourceOption; 2]>,
    ) -> Result<(), String> {
        let technology_type = match technology_type {
            Some(technology) => technology,
            None => {
                return Err("ティカルのアクションスペース1は技術を選択してください".to_string());
            }
        };
        let reward = player.technology.progress(technology_type);

        match reward {
            Some(reward) => match reward {
                TechnologyProgressReward::Faith => {
                    let target_temple = match target_temple {
                        Some(target_temple) => target_temple,
                        None => {
                            return Err("対象の神殿を選択してください".to_string());
                        }
                    };
                    match target_temple {
                        TempleName::Chaac => {
                            if player.temple_faith.chaac.0 == MAX_CHAAC_RANK {
                                return Err("チャクの信仰は既に上限です".to_string());
                            } else {
                                player.temple_faith.chaac.0 += 1;
                            }
                        }
                        TempleName::Quetzalcoatl => {
                            if player.temple_faith.quetzalcoatl.0 == MAX_QUETZALCOATL_RANK {
                                return Err("ケツァルコアトルの信仰は既に上限です".to_string());
                            } else {
                                player.temple_faith.quetzalcoatl.0 += 1;
                            }
                        }
                        TempleName::Kukulkan => {
                            if player.temple_faith.kukulkan.0 == MAX_KUKULKAN_RANK {
                                return Err("ククルカンの信仰は既に上限です".to_string());
                            } else {
                                player.temple_faith.kukulkan.0 += 1;
                            }
                        }
                    }
                }
                TechnologyProgressReward::Point(point) => {
                    player.add_points(point);
                }
                TechnologyProgressReward::Resource => {
                    let resources = match resources {
                        Some(resources) => resources,
                        None => {
                            return Err("欲しい資源を選択してください".to_string());
                        }
                    };
                    resources.iter().for_each(|resource| match resource {
                        ResourceOption::Wood => {
                            player.resource.woods.0 += 1;
                        }
                        ResourceOption::Stone => {
                            player.resource.stones.0 += 1;
                        }
                        ResourceOption::Gold => {
                            player.resource.golds.0 += 1;
                        }
                    })
                }
                TechnologyProgressReward::Skull => {
                    if field_skells.get_remaining_skulls() > 0 {
                        field_skells.decrease_skulls(1);
                        player.resource.skulls.0 += 1;
                    }
                }
            },
            None => {}
        }
        Ok(())
    }
    // 2 => {
    //     // TODO 建築アクションの実装
    //     Ok(())
    // }
    // 3 => {
    //     // TODO 2回技術上げ
    //     Ok(())
    // }
    // 4 => {
    //     // TODO 建築アクションの実装
    //     Ok(())
    // }
    // _ => {
    //     return Err("ティカルのアクションスペース番号が不正です".to_string());
    // }
}
