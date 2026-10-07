//! Quick action tiles and their two-day calendar schedule.
use serde::{Deserialize, Serialize};
#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub enum QuickActionId {
    Corn,
    WoodCorn,
    Stone,
    Gold,
    Technology,
    Trade,
    Build,
}
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct QuickActionState {
    pub age1: Vec<QuickActionId>,
    pub age2: Vec<QuickActionId>,
    pub current: QuickActionId,
    /// -1 is a permanent dummy blocker; other values are player IDs.
    pub spaces: Vec<Option<i64>>,
    pub resolved: bool,
}
impl QuickActionId {
    pub const AGE1: [Self; 7] = [
        Self::Corn,
        Self::WoodCorn,
        Self::Stone,
        Self::Technology,
        Self::Technology,
        Self::Trade,
        Self::Build,
    ];
    pub const AGE2: [Self; 6] = [
        Self::Corn,
        Self::Stone,
        Self::Gold,
        Self::Technology,
        Self::Trade,
        Self::Build,
    ];
}
impl QuickActionState {
    pub fn update(&mut self, round: i64) {
        self.current = if round >= 15 {
            self.age2[((round - 15) / 2).min(5) as usize]
        } else {
            self.age1[((round - 1) / 2).min(6) as usize]
        };
    }
    pub fn clear_workers(&mut self) {
        for space in &mut self.spaces {
            if space.is_some_and(|id| id >= 0) {
                *space = None;
            }
        }
        self.resolved = false;
    }
}
#[derive(Clone, Debug, Serialize)]
pub struct QuickActionDefinition {
    pub id: QuickActionId,
    pub name: &'static str,
    pub description: &'static str,
}
pub fn definitions() -> Vec<QuickActionDefinition> {
    use QuickActionId::*;
    [
        (Corn, "コーン", "コーン 3 を得ます。"),
        (WoodCorn, "木材とコーン", "木材 1 とコーン 1 を得ます。"),
        (Stone, "石材", "石材 1 を得ます。"),
        (Gold, "金", "金 1 を得ます。"),
        (Technology, "技術", "通常の費用で技術を 1 レベル進めます。"),
        (Trade, "市場", "市場で何度でも資源とコーンを交換します。"),
        (
            Build,
            "建設",
            "コーン 1 を払い建物を 1 枚建設します。建築技術は使えません。",
        ),
    ]
    .into_iter()
    .map(|(id, name, description)| QuickActionDefinition {
        id,
        name,
        description,
    })
    .collect()
}
