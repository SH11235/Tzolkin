use crate::utils::constants::{
    FIRST_FOOD_DAY, FOURTH_FOOD_DAY, SECOND_FOOD_DAY, THIRD_FOOD_DAY,
    TOP_BONUS_FOURTH_FOOD_DAY_CHAAC, TOP_BONUS_FOURTH_FOOD_DAY_KUKULKAN,
    TOP_BONUS_FOURTH_FOOD_DAY_QUETZALCOATL, TOP_BONUS_SECOND_FOOD_DAY_CHAAC,
    TOP_BONUS_SECOND_FOOD_DAY_KUKULKAN, TOP_BONUS_SECOND_FOOD_DAY_QUETZALCOATL,
};
use crate::utils::increment::Increment;

use super::chichen_itza_skull::ChichenItzaSkull;
use super::jungle::Jungle;
use super::player;
use super::resources::FieldSkulls;

pub enum FoodDay {
    First,
    Second,
    Third,
    Fourth,
}

#[derive(Debug, Default, PartialEq, PartialOrd)]
pub struct Round(u32);
impl Round {
    pub fn new(num: u32) -> Self {
        Self(num)
    }
}
impl Increment for Round {
    fn increment(&mut self) {
        if self.0 < FOURTH_FOOD_DAY {
            self.0.increment();
        }
    }
}

#[derive(Debug, Default, PartialEq, PartialOrd)]
pub struct Generation(pub u32);
impl Generation {
    pub fn new(num: u32) -> Self {
        Self(num)
    }
}
impl Increment for Generation {
    fn increment(&mut self) {
        if self.0 == 1 {
            self.0 = 2;
        }
    }
}

#[derive(Debug, Default)]
pub(crate) struct Game {
    round: Round,
    generation: Generation,
    pub players: Vec<player::Player>,
    corns: u32,
    jungle: Jungle,
    chichen_itza_skull: ChichenItzaSkull,
    field_skull: FieldSkulls,
    first_food_day_done: bool,
    second_food_day_done: bool,
    third_food_day_done: bool,
    fourth_food_day_done: bool,
}

impl Game {
    pub fn new(number_of_players: u32) -> Result<Self, &'static str> {
        let corns = 0;
        Ok(Self {
            round: Round::new(1),
            generation: Generation::new(1),
            // number_of_playersの数だけplayerを作成する
            players: (1..=number_of_players)
                .map(|i| player::Player::new(format!("Player {}", i), i.into()))
                .collect(),
            corns,
            jungle: Jungle::new(&number_of_players)?,
            chichen_itza_skull: ChichenItzaSkull::new(),
            field_skull: FieldSkulls::new(),
            first_food_day_done: false,
            second_food_day_done: false,
            third_food_day_done: false,
            fourth_food_day_done: false,
        })
    }

    pub fn get_round(&self) -> u32 {
        self.round.0
    }

    pub fn next_round(&mut self) {
        self.round.increment();
    }

    pub fn get_generation(&self) -> u32 {
        self.generation.0
    }

    # [cfg (test)]
    pub fn get_food_day_status(&self, food_day: FoodDay) -> bool {
        match food_day {
            FoodDay::First => self.first_food_day_done,
            FoodDay::Second => self.second_food_day_done,
            FoodDay::Third => self.third_food_day_done,
            FoodDay::Fourth => self.fourth_food_day_done,
        }
    }

    fn next_generation(&mut self) {
        self.generation.increment();
    }

    fn add_corns(&mut self) {
        self.corns += 1;
    }

    pub fn get_corns(&self) -> u32 {
        self.corns
    }

    fn get_temple_top_player_index(&self) -> (Vec<usize>, Vec<usize>, Vec<usize>) {
        // playerの中で信仰が最も大きいplayerを取得する
        let mut max_chaac = 0;
        let mut max_quetzalcoatl = 0;
        let mut max_kukulkan = 0;
        self.players.iter().for_each(|player| {
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
        self.players.iter().enumerate().for_each(|(i, player)| {
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

    fn get_food_day_point_reward(
        &mut self,
        chaac_point: f32,
        quetzalcoatl_point: f32,
        kukulkan_point: f32,
    ) {
        let (max_chaac_player, max_quetzalcoatl_player, max_kukulkan_player) =
            self.get_temple_top_player_index();
        // トップが1人ならそのplayerにトップボーナスの得点を与える
        // トップが複数ならそのplayerにトップボーナスの半分の得点を与える
        if max_chaac_player.len() > 1 {
            max_chaac_player.into_iter().for_each(|index| {
                self.players[index].add_points(chaac_point / 2.0);
            });
        } else {
            self.players[max_chaac_player[0]].add_points(chaac_point);
        }
        if max_quetzalcoatl_player.len() > 1 {
            max_quetzalcoatl_player.into_iter().for_each(|index| {
                self.players[index].add_points(quetzalcoatl_point / 2.0);
            });
        } else {
            self.players[max_quetzalcoatl_player[0]].add_points(quetzalcoatl_point);
        }
        if max_kukulkan_player.len() > 1 {
            max_kukulkan_player.into_iter().for_each(|index| {
                self.players[index].add_points(kukulkan_point / 2.0);
            });
        } else {
            self.players[max_kukulkan_player[0]].add_points(kukulkan_point);
        }
    }

    fn food_day(&mut self, food_day: FoodDay) {
        match food_day {
            FoodDay::First => {
                self.first_food_day_done = true;
                let mut required_skulls = 0;
                self.players.iter().for_each(|player| {
                    if player.get_kukulkan() >= 4 {
                        required_skulls += 1;
                    }
                });
                self.players.iter_mut().for_each(|player| {
                    player.feed();
                    player.get_resource_reward_from_temple();
                    if self.field_skull.get_remaining_skulls() >= required_skulls {
                        self.field_skull.decrease_skulls(required_skulls);
                        player.get_skull_reward_from_kukulkan();
                    }
                });
            }
            FoodDay::Second => {
                self.second_food_day_done = true;
                self.players.iter_mut().for_each(|player| {
                    player.feed();
                    player.get_point_reward_from_temple();
                });
                self.get_food_day_point_reward(
                    TOP_BONUS_SECOND_FOOD_DAY_CHAAC as f32,
                    TOP_BONUS_SECOND_FOOD_DAY_QUETZALCOATL as f32,
                    TOP_BONUS_SECOND_FOOD_DAY_KUKULKAN as f32,
                );
            }
            FoodDay::Third => {
                self.third_food_day_done = true;
                self.players.iter_mut().for_each(|player| {
                    player.feed();
                    player.get_resource_reward_from_temple();
                });
            }
            FoodDay::Fourth => {
                self.fourth_food_day_done = true;
                self.players.iter_mut().for_each(|player| {
                    player.feed();
                    player.get_point_reward_from_temple();
                });
                self.get_food_day_point_reward(
                    TOP_BONUS_FOURTH_FOOD_DAY_CHAAC as f32,
                    TOP_BONUS_FOURTH_FOOD_DAY_QUETZALCOATL as f32,
                    TOP_BONUS_FOURTH_FOOD_DAY_KUKULKAN as f32,
                );
            }
        }
    }
    // round終了時の処理をまとめた関数
    pub fn end_round(&mut self) -> bool {
        if !self.first_food_day_done && self.round.0 >= FIRST_FOOD_DAY {
            self.food_day(FoodDay::First);
        }
        if !self.second_food_day_done && self.round.0 >= SECOND_FOOD_DAY {
            self.food_day(FoodDay::Second);
        }
        if !self.third_food_day_done && self.round.0 >= THIRD_FOOD_DAY {
            self.food_day(FoodDay::Third);
        }
        if !self.fourth_food_day_done && self.round.0 >= FOURTH_FOOD_DAY {
            self.food_day(FoodDay::Fourth);
        }

        if self.round.0 >= FOURTH_FOOD_DAY {
            true
        } else {
            self.next_round();
            if (self.generation.0 == 1) && (self.round.0 >= SECOND_FOOD_DAY + 1) {
                self.next_generation();
            }
            self.add_corns();
            false
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::game_object::{
        player::temple_faith::TempleFaith,
        temple::{Chaac, Kukulkan, Quetzalcoatl, Temple},
    };

    use super::*;

    #[test]
    fn test_increase_round() {
        let mut round = Round::new(1);
        round.increment();
        assert_eq!(round.0, 2);

        let mut round = Round::new(27);
        round.increment();
        assert_eq!(round.0, 27);
    }

    #[test]
    fn test_get_temple_top_player_index_points() {
        let mut game = Game {
            round: Round::new(1),
            generation: Generation(1),
            players: vec![
                player::Player {
                    name: String::from("player1"),
                    temple_faith: TempleFaith {
                        chaac: Chaac::new(-1),
                        quetzalcoatl: Quetzalcoatl::new(-1),
                        kukulkan: Kukulkan::new(-1),
                    },
                    ..Default::default()
                },
                player::Player {
                    name: String::from("player2"),
                    temple_faith: TempleFaith {
                        chaac: Chaac::new(5),
                        quetzalcoatl: Quetzalcoatl::new(7),
                        kukulkan: Kukulkan::new(6),
                    },
                    ..Default::default()
                },
                player::Player {
                    name: String::from("player3"),
                    temple_faith: TempleFaith {
                        chaac: Chaac::new(0),
                        quetzalcoatl: Quetzalcoatl::new(0),
                        kukulkan: Kukulkan::new(0),
                    },
                    ..Default::default()
                },
                player::Player {
                    name: String::from("player4"),
                    temple_faith: TempleFaith {
                        chaac: Chaac::new(-1),
                        quetzalcoatl: Quetzalcoatl::new(0),
                        kukulkan: Kukulkan::new(2),
                    },
                    ..Default::default()
                },
            ],
            ..Default::default()
        };
        let (max_chaac_player, max_quetzalcoatl_player, max_kukulkan_player) =
            game.get_temple_top_player_index();
        assert_eq!(max_chaac_player, vec![1]);
        assert_eq!(max_quetzalcoatl_player, vec![1]);
        assert_eq!(max_kukulkan_player, vec![1]);
        game.food_day(FoodDay::Second);
        assert_eq!(game.players[0].points, -6.0);
        assert_eq!(game.players[1].points, 43.0);
        assert_eq!(game.players[2].points, 0.0);
        assert_eq!(game.players[3].points, 2.0);

        let mut game = Game {
            round: Round::new(1),
            generation: Generation(1),
            players: vec![
                player::Player {
                    name: String::from("player1"),
                    temple_faith: TempleFaith {
                        chaac: Chaac::new(-1),
                        quetzalcoatl: Quetzalcoatl::new(-1),
                        kukulkan: Kukulkan::new(-1),
                    },
                    ..Default::default()
                },
                player::Player {
                    name: String::from("player2"),
                    temple_faith: TempleFaith {
                        chaac: Chaac::new(5),
                        quetzalcoatl: Quetzalcoatl::new(7),
                        kukulkan: Kukulkan::new(6),
                    },
                    ..Default::default()
                },
                player::Player {
                    name: String::from("player3"),
                    temple_faith: TempleFaith {
                        chaac: Chaac::new(0),
                        quetzalcoatl: Quetzalcoatl::new(0),
                        kukulkan: Kukulkan::new(0),
                    },
                    ..Default::default()
                },
                player::Player {
                    name: String::from("player4"),
                    temple_faith: TempleFaith {
                        chaac: Chaac::new(-1),
                        quetzalcoatl: Quetzalcoatl::new(0),
                        kukulkan: Kukulkan::new(2),
                    },
                    ..Default::default()
                },
            ],
            ..Default::default()
        };
        game.food_day(FoodDay::Fourth);
        assert_eq!(game.players[0].points, -6.0);
        assert_eq!(game.players[1].points, 43.0);
        assert_eq!(game.players[2].points, 0.0);
        assert_eq!(game.players[3].points, 2.0);

        let mut game = Game {
            round: Round::new(1),
            generation: Generation(1),
            players: vec![
                player::Player {
                    name: String::from("player1"),
                    temple_faith: TempleFaith {
                        chaac: Chaac::new(4),
                        quetzalcoatl: Quetzalcoatl::new(-1),
                        kukulkan: Kukulkan::new(-1),
                    },
                    ..Default::default()
                },
                player::Player {
                    name: String::from("player2"),
                    temple_faith: TempleFaith {
                        chaac: Chaac::new(4),
                        quetzalcoatl: Quetzalcoatl::new(6),
                        kukulkan: Kukulkan::new(5),
                    },
                    ..Default::default()
                },
                player::Player {
                    name: String::from("player3"),
                    temple_faith: TempleFaith {
                        chaac: Chaac::new(0),
                        quetzalcoatl: Quetzalcoatl::new(6),
                        kukulkan: Kukulkan::new(0),
                    },
                    ..Default::default()
                },
                player::Player {
                    name: String::from("player4"),
                    temple_faith: TempleFaith {
                        chaac: Chaac::new(4),
                        quetzalcoatl: Quetzalcoatl::new(6),
                        kukulkan: Kukulkan::new(5),
                    },
                    ..Default::default()
                },
            ],
            ..Default::default()
        };
        let (max_chaac_player, max_quetzalcoatl_player, max_kukulkan_player) =
            game.get_temple_top_player_index();
        assert_eq!(max_chaac_player, vec![0, 1, 3]);
        assert_eq!(max_quetzalcoatl_player, vec![1, 2, 3]);
        assert_eq!(max_kukulkan_player, vec![1, 3]);
        game.food_day(FoodDay::Second);
        assert_eq!(game.players[0].points, 5.0);
        assert_eq!(game.players[1].points, 34.0);
        assert_eq!(game.players[2].points, 13.0);
        assert_eq!(game.players[3].points, 34.0);
    }

    #[test]
    fn test_food_day_next_generation() {
        let mut game = Game::new(3).unwrap();
        // FoodDay::Firstの境界テスト
        for _ in 0..7 {
            game.end_round();
        }
        // 7th roundの終了時は1回目のfood_dayではない
        assert_eq!(game.get_food_day_status(FoodDay::First), false);
        // 8th roundの終了時に1回目のfood_dayになる
        game.end_round();
        assert_eq!(game.get_food_day_status(FoodDay::First), true);

        // FoodDay::SecondとGeneration切り替わりの境界テスト
        // game.round.0 == 9
        for _ in 0..5 {
            game.end_round();
        }
        // game.round.0 == 14
        // 13th roundの終了時点（14th 開始時点）は2回目のfood_dayを終えていなく、第一世代である
        assert_eq!(game.get_food_day_status(FoodDay::Second), false);
        assert_eq!(game.generation, Generation(1));
        // 14th roundの終了時は2回目のfood_dayで、第二世代になる
        game.end_round();
        assert_eq!(game.get_food_day_status(FoodDay::Second), true);
        assert_eq!(game.generation.0, 2);

        // FoodDay::Thirdの境界テスト
        for _ in 0..6 {
            game.end_round();
        }
        // game.round.0 == 21
        // 21th 開始時点は3回目のfood_dayを終えていない
        assert_eq!(game.get_food_day_status(FoodDay::Third), false);
        game.end_round();
        // 21th 終了後は3回目のfood_dayを終えている
        assert_eq!(game.get_food_day_status(FoodDay::Third), true);

        // FoodDay::Fourthの境界テスト
        // game.round.0 == 22
        for _ in 0..5 {
            game.end_round();
        }
        // game.round.0 == 27
        // 27th 開始時点は4回目のfood_dayを終えていない
        assert_eq!(game.get_food_day_status(FoodDay::Fourth), false);
        game.end_round();
        // 27th 終了後は4回目のfood_dayを終えている
        assert_eq!(game.get_food_day_status(FoodDay::Fourth), true);
    }
}
