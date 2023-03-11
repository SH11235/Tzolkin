use crate::utils::constants::{
    TOP_BONUS_FOURTH_FOOD_DAY_CHAAC, TOP_BONUS_FOURTH_FOOD_DAY_KUKULKAN,
    TOP_BONUS_FOURTH_FOOD_DAY_QUETZALCOATL, TOP_BONUS_SECOND_FOOD_DAY_CHAAC,
    TOP_BONUS_SECOND_FOOD_DAY_KUKULKAN, TOP_BONUS_SECOND_FOOD_DAY_QUETZALCOATL,
};

use super::{player::Player, resources::FieldSkulls};

pub struct FoodDayStatus {
    pub first_food_day_done: bool,
    pub second_food_day_done: bool,
    pub third_food_day_done: bool,
    pub fourth_food_day_done: bool,
}

pub enum FoodDay {
    First,
    Second,
    Third,
    Fourth,
}

impl FoodDayStatus {
    pub fn new() -> Self {
        FoodDayStatus {
            first_food_day_done: false,
            second_food_day_done: false,
            third_food_day_done: false,
            fourth_food_day_done: false,
        }
    }

    pub fn food_day(
        &mut self,
        food_day: FoodDay,
        players: &mut Vec<Player>,
        field_skull: &mut FieldSkulls,
    ) {
        match food_day {
            FoodDay::First => {
                self.first_food_day_done = true;
                let mut required_skulls = 0;
                players.iter().for_each(|player| {
                    if player.get_kukulkan() >= 4 {
                        required_skulls += 1;
                    }
                });
                players.iter_mut().for_each(|player| {
                    player.feed();
                    player.get_resource_reward_from_temple();
                    if field_skull.get_remaining_skulls() >= required_skulls {
                        field_skull.decrease_skulls(required_skulls);
                        player.get_skull_reward_from_kukulkan();
                    }
                });
            }
            FoodDay::Second => {
                self.second_food_day_done = true;
                players.iter_mut().for_each(|player| {
                    player.feed();
                    player.get_point_reward_from_temple();
                });
                get_food_day_point_reward(
                    players,
                    TOP_BONUS_SECOND_FOOD_DAY_CHAAC as f32,
                    TOP_BONUS_SECOND_FOOD_DAY_QUETZALCOATL as f32,
                    TOP_BONUS_SECOND_FOOD_DAY_KUKULKAN as f32,
                );
            }
            FoodDay::Third => {
                self.third_food_day_done = true;
                players.iter_mut().for_each(|player| {
                    player.feed();
                    player.get_resource_reward_from_temple();
                });
            }
            FoodDay::Fourth => {
                self.fourth_food_day_done = true;
                players.iter_mut().for_each(|player| {
                    player.feed();
                    player.get_point_reward_from_temple();
                });
                get_food_day_point_reward(
                    players,
                    TOP_BONUS_FOURTH_FOOD_DAY_CHAAC as f32,
                    TOP_BONUS_FOURTH_FOOD_DAY_QUETZALCOATL as f32,
                    TOP_BONUS_FOURTH_FOOD_DAY_KUKULKAN as f32,
                );
            }
        }
    }

    #[cfg(test)]
    pub fn get_food_day_status(&self, food_day: FoodDay) -> bool {
        match food_day {
            FoodDay::First => self.first_food_day_done,
            FoodDay::Second => self.second_food_day_done,
            FoodDay::Third => self.third_food_day_done,
            FoodDay::Fourth => self.fourth_food_day_done,
        }
    }
}

pub fn get_temple_top_player_index(
    players: &mut Vec<Player>,
) -> (Vec<usize>, Vec<usize>, Vec<usize>) {
    // playerの中で信仰が最も大きいplayerを取得する
    let mut max_chaac = 0;
    let mut max_quetzalcoatl = 0;
    let mut max_kukulkan = 0;
    players.iter().for_each(|player| {
        if player.get_chaac() > max_chaac {
            max_chaac = player.get_chaac();
        }
        if player.get_quetzalcoatl() > max_quetzalcoatl {
            max_quetzalcoatl = player.get_quetzalcoatl();
        }
        if player.get_kukulkan() > max_kukulkan {
            max_kukulkan = player.get_kukulkan();
        }
    });
    // 各神殿で信仰がトップのplayerを取得する
    let mut max_chaac_player = vec![];
    let mut max_quetzalcoatl_player = vec![];
    let mut max_kukulkan_player = vec![];
    players.iter().enumerate().for_each(|(i, player)| {
        if player.get_chaac() == max_chaac {
            max_chaac_player.push(i);
        }
        if player.get_quetzalcoatl() == max_quetzalcoatl {
            max_quetzalcoatl_player.push(i);
        }
        if player.get_kukulkan() == max_kukulkan {
            max_kukulkan_player.push(i);
        }
    });
    (
        max_chaac_player,
        max_quetzalcoatl_player,
        max_kukulkan_player,
    )
}

pub fn get_food_day_point_reward(
    players: &mut Vec<Player>,
    chaac_point: f32,
    quetzalcoatl_point: f32,
    kukulkan_point: f32,
) {
    let (max_chaac_player, max_quetzalcoatl_player, max_kukulkan_player) =
        get_temple_top_player_index(players);
    // トップが1人ならそのplayerにトップボーナスの得点を与える
    // トップが複数ならそのplayerにトップボーナスの半分の得点を与える
    if max_chaac_player.len() > 1 {
        max_chaac_player.into_iter().for_each(|index| {
            players[index].add_points(chaac_point / 2.0);
        });
    } else {
        players[max_chaac_player[0]].add_points(chaac_point);
    }
    if max_quetzalcoatl_player.len() > 1 {
        max_quetzalcoatl_player.into_iter().for_each(|index| {
            players[index].add_points(quetzalcoatl_point / 2.0);
        });
    } else {
        players[max_quetzalcoatl_player[0]].add_points(quetzalcoatl_point);
    }
    if max_kukulkan_player.len() > 1 {
        max_kukulkan_player.into_iter().for_each(|index| {
            players[index].add_points(kukulkan_point / 2.0);
        });
    } else {
        players[max_kukulkan_player[0]].add_points(kukulkan_point);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::game_object::player::temple_faith::TempleFaith;
    use crate::game_object::player::Player;
    use crate::game_object::resources::FieldSkulls;
    use crate::game_object::temple::{Chaac, Temple, Quetzalcoatl, Kukulkan};

    #[test]
    fn test_get_temple_top_player_index_points() {
        let mut food_day_status = FoodDayStatus::new();
        let mut field_skull = FieldSkulls::new();
        let mut players = vec![
            Player {
                name: String::from("player1"),
                temple_faith: TempleFaith {
                    chaac: Chaac::new(-1),
                    quetzalcoatl: Quetzalcoatl::new(-1),
                    kukulkan: Kukulkan::new(-1),
                },
                ..Default::default()
            },
            Player {
                name: String::from("player2"),
                temple_faith: TempleFaith {
                    chaac: Chaac::new(5),
                    quetzalcoatl: Quetzalcoatl::new(7),
                    kukulkan: Kukulkan::new(6),
                },
                ..Default::default()
            },
            Player {
                name: String::from("player3"),
                temple_faith: TempleFaith {
                    chaac: Chaac::new(0),
                    quetzalcoatl: Quetzalcoatl::new(0),
                    kukulkan: Kukulkan::new(0),
                },
                ..Default::default()
            },
            Player {
                name: String::from("player4"),
                temple_faith: TempleFaith {
                    chaac: Chaac::new(-1),
                    quetzalcoatl: Quetzalcoatl::new(0),
                    kukulkan: Kukulkan::new(2),
                },
                ..Default::default()
            },
        ];
        let (max_chaac_player, max_quetzalcoatl_player, max_kukulkan_player) =
            get_temple_top_player_index(&mut players);
        assert_eq!(max_chaac_player, vec![1]);
        assert_eq!(max_quetzalcoatl_player, vec![1]);
        assert_eq!(max_kukulkan_player, vec![1]);
        food_day_status.food_day(FoodDay::Second, &mut players, &mut field_skull);
        assert_eq!(players[0].points, -6.0);
        assert_eq!(players[1].points, 43.0);
        assert_eq!(players[2].points, 0.0);
        assert_eq!(players[3].points, 2.0);

        let mut players = vec![
            Player {
                name: String::from("player1"),
                temple_faith: TempleFaith {
                    chaac: Chaac::new(-1),
                    quetzalcoatl: Quetzalcoatl::new(-1),
                    kukulkan: Kukulkan::new(-1),
                },
                ..Default::default()
            },
            Player {
                name: String::from("player2"),
                temple_faith: TempleFaith {
                    chaac: Chaac::new(5),
                    quetzalcoatl: Quetzalcoatl::new(7),
                    kukulkan: Kukulkan::new(6),
                },
                ..Default::default()
            },
            Player {
                name: String::from("player3"),
                temple_faith: TempleFaith {
                    chaac: Chaac::new(0),
                    quetzalcoatl: Quetzalcoatl::new(0),
                    kukulkan: Kukulkan::new(0),
                },
                ..Default::default()
            },
            Player {
                name: String::from("player4"),
                temple_faith: TempleFaith {
                    chaac: Chaac::new(-1),
                    quetzalcoatl: Quetzalcoatl::new(0),
                    kukulkan: Kukulkan::new(2),
                },
                ..Default::default()
            },
        ];

        food_day_status.food_day(FoodDay::Fourth, &mut players, &mut field_skull);
        assert_eq!(players[0].points, -6.0);
        assert_eq!(players[1].points, 43.0);
        assert_eq!(players[2].points, 0.0);
        assert_eq!(players[3].points, 2.0);

        let mut players = vec![
            Player {
                name: String::from("player1"),
                temple_faith: TempleFaith {
                    chaac: Chaac::new(4),
                    quetzalcoatl: Quetzalcoatl::new(-1),
                    kukulkan: Kukulkan::new(-1),
                },
                ..Default::default()
            },
            Player {
                name: String::from("player2"),
                temple_faith: TempleFaith {
                    chaac: Chaac::new(4),
                    quetzalcoatl: Quetzalcoatl::new(6),
                    kukulkan: Kukulkan::new(5),
                },
                ..Default::default()
            },
            Player {
                name: String::from("player3"),
                temple_faith: TempleFaith {
                    chaac: Chaac::new(0),
                    quetzalcoatl: Quetzalcoatl::new(6),
                    kukulkan: Kukulkan::new(0),
                },
                ..Default::default()
            },
            Player {
                name: String::from("player4"),
                temple_faith: TempleFaith {
                    chaac: Chaac::new(4),
                    quetzalcoatl: Quetzalcoatl::new(6),
                    kukulkan: Kukulkan::new(5),
                },
                ..Default::default()
            },
        ];
        let (max_chaac_player, max_quetzalcoatl_player, max_kukulkan_player) =
            get_temple_top_player_index(&mut players);
        assert_eq!(max_chaac_player, vec![0, 1, 3]);
        assert_eq!(max_quetzalcoatl_player, vec![1, 2, 3]);
        assert_eq!(max_kukulkan_player, vec![1, 3]);
        food_day_status.food_day(FoodDay::Second, &mut players, &mut field_skull);
        assert_eq!(players[0].points, 5.0);
        assert_eq!(players[1].points, 34.0);
        assert_eq!(players[2].points, 13.0);
        assert_eq!(players[3].points, 34.0);
    }
}
