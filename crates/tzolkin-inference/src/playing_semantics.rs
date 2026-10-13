//! Schema-4 operation descriptors, not a simulator, payoff forecast or source audit.
//! Costs/grants concern only the named operation. The public Observation omits the
//! pending continuation and hidden replenishment; neither is predicted here.
use super::{resource_index, technology_index, temple_index};
use tzolkin_core::catalog::{self, CATALOG};
use tzolkin_core::observation::{ActionSource, HarvestKind, Observation, TypedAction};
use tzolkin_core::*;

pub const PLAYING_SEMANTIC_FEATURE_START: usize = 417;
pub const PLAYING_SEMANTIC_FEATURE_COUNT: usize = 95;

/// Raw integer quantities, with RESOURCE/TECHNOLOGY/TEMPLE_IDS ordering in each
/// contiguous group. Zero means no quantity of that described kind, never an
/// unknown outcome. Catalog atoms describe instructions, not their settled result.
/// In particular, zero choice atoms do not deny conditional tasks emitted while
/// resolving specific catalog effects (such as a Theology offering), or tasks in
/// the omitted continuation. No aggregate describes complete available choices.
/// Existing action tags/gear/position and schema-2 state columns remain available.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(usize)]
pub enum PlayingSemanticFeature {
    /// 0..5: resources paid by this operation, excluding later choices.
    CostCorn = 0,
    /// 5..10: resources received by this operation; Skull is supply-capped.
    GainCorn = 5,
    ImmediatePoints = 10,
    WorkerReleased = 11,
    WorkerAdded = 12,
    /// Stock queued for the claiming actor's later FinishTurn, never spendable now.
    PendingFirstPlayerCorn = 13,
    TaskRemaining = 14,
    TechnologyCurrent = 15,
    /// Target if the selected advancement is paid/applied; not a paid-choice receipt.
    TechnologyTarget = 16,
    MaterialPaymentRequired = 17,
    TechnologyFree = 18,
    TempleCurrent = 19,
    TempleTarget = 20,
    /// 21..26: bundled building Resource effect atoms, not received resources.
    CatalogResourceCorn = 21,
    /// 26..30: catalog specific-track instruction atoms, never applied deltas.
    CatalogTechnologyAgriculture = 26,
    /// 30..33: catalog specific-temple instruction atoms, never applied deltas.
    CatalogTempleChaac = 30,
    CatalogWorker = 33,
    CatalogFedWorkers = 34,
    CatalogFeedAll = 35,
    CatalogFeedDiscount = 36,
    CatalogPoints = 37,
    /// 38..43: per-Food-Day building reward atoms, excluding future recipient/switch choices.
    FoodRewardCorn = 38,
    FoodRewardSwitch = 43,
    BuildingArchitecture = 44,
    SkullBuilding = 45,
    /// Directly scheduled tasks or catalog choice instructions, not completed
    /// choices/grants or a complete count of conditional/continuation tasks.
    ActionChoice = 46,
    TechnologyChoice = 47,
    TempleChoice = 48,
    ResourceChoice = 49,
    BuildingChoice = 50,
    MonumentChoice = 51,
    TradeChoice = 52,
    HarvestChoice = 53,
    OfferingChoice = 54,
    AnyActionChoice = 55,
    TempleDownChoice = 56,
    BuildingCornPayment = 57,
    TempleDistinct = 58,
    /// 1=current gear, 2=Theology ahead, 3=anywhere; expansion sources are refused.
    ActionSource = 59,
    /// 1=corn, 2=wood, 3=burn, 4=empty corn.
    HarvestKind = 60,
    CornTilesConsumed = 61,
    WoodTilesConsumed = 62,
    /// CornTilesConsumed is also the number of earned corn tiles; burned wood
    /// is consumed without earning a wood tile.
    WoodTilesAdded = 63,
    TechnologyExchangeChoice = 64,
    CappedTechnologyBonus = 65,
    /// Beg/down-temple sets Corn to this quantity, rather than granting this much.
    BegCornTarget = 66,
    SkullPlaced = 67,
    /// Unnormalized 1-based position in bundled base BUILDINGS, not a rank/value.
    /// This replaces the old candidate hash as the explicit building identity carrier.
    BuildingIdentity = 68,
    /// 69..82: MONUMENT_IDS one-hot identifying the score rule, not awarded VP.
    MonumentIdentity = 69,
    /// Closed 1-based TECHNOLOGY_IDS / TEMPLE_IDS ordinals, not effect quantities.
    SelectedTechnology = 82,
    SelectedTemple = 83,
    /// 84..88 and 88..91: actual level deltas of this operation only. Catalog
    /// instructions and paid-but-not-yet-advanced Technology selections stay zero.
    TechnologyDeltaAgriculture = 84,
    TempleDeltaChaac = 88,
    /// Corn fee for an offered AnyAction, not paid until that choice is made.
    AnyActionCornRequired = 91,
    AnyActionAllowsSkulls = 92,
    TaskMandatory = 93,
    BuildingAllowsMonument = 94,
}
impl PlayingSemanticFeature {
    pub const fn index(self) -> usize {
        PLAYING_SEMANTIC_FEATURE_START + self as usize
    }
}
const _: () = assert!(
    PlayingSemanticFeature::BuildingAllowsMonument as usize + 1 == PLAYING_SEMANTIC_FEATURE_COUNT
);

struct Descriptor([i64; PLAYING_SEMANTIC_FEATURE_COUNT]);
impl Descriptor {
    fn add(&mut self, field: PlayingSemanticFeature, amount: i64) -> Result<(), String> {
        self.at(field as usize, amount)
    }
    fn at(&mut self, index: usize, amount: i64) -> Result<(), String> {
        self.0[index] = self.0[index]
            .checked_add(amount)
            .filter(|value| (-1024..=1024).contains(value))
            .ok_or("Playing semantic quantity exceeds exact integer bound")?;
        Ok(())
    }
    fn resources(
        &mut self,
        field: PlayingSemanticFeature,
        values: &[i64; 5],
    ) -> Result<(), String> {
        if values.iter().any(|value| *value < 0) {
            return Err("Negative semantic resource quantity".into());
        }
        for (i, value) in values.iter().enumerate() {
            self.at(field as usize + i, *value)?;
        }
        Ok(())
    }
    fn cost(&mut self, o: &Observation, values: [i64; 5]) -> Result<(), String> {
        if values
            .iter()
            .zip(o.players[o.actor].resources)
            .any(|(need, have)| *need > have)
        {
            return Err("Semantic operation cannot pay its stated cost".into());
        }
        self.resources(PlayingSemanticFeature::CostCorn, &values)
    }
    fn gain(&mut self, o: &Observation, mut values: [i64; 5]) -> Result<(), String> {
        values[4] = values[4].min(o.skull_supply);
        self.resources(PlayingSemanticFeature::GainCorn, &values)
    }
}

pub(super) fn validate_values(values: &[f32; super::FEATURE_COUNT]) -> Result<(), String> {
    use PlayingSemanticFeature as F;
    if values[231] != 0.0
        || values[232] != 1.0
        || values[233] != 0.0
        || values[PLAYING_SEMANTIC_FEATURE_START..]
            .iter()
            .any(|n| !n.is_finite() || n.fract() != 0.0 || n.abs() > 1024.0)
    {
        return Err("Invalid semantic Playing phase/integer descriptor".into());
    }
    let signed = [F::TempleCurrent.index(), F::TempleTarget.index()];
    for (index, value) in values
        .iter()
        .enumerate()
        .skip(PLAYING_SEMANTIC_FEATURE_START)
    {
        let level_delta = (F::TechnologyDeltaAgriculture.index()..F::TempleDeltaChaac.index() + 3)
            .contains(&index);
        if *value
            < if signed.contains(&index) || level_delta {
                -1.0
            } else {
                0.0
            }
        {
            return Err("Negative semantic quantity outside a level/delta field".into());
        }
    }
    for field in [
        F::TechnologyFree,
        F::BuildingArchitecture,
        F::BuildingCornPayment,
        F::TempleDistinct,
        F::AnyActionAllowsSkulls,
        F::TaskMandatory,
        F::BuildingAllowsMonument,
    ] {
        if ![0.0, 1.0].contains(&values[field.index()]) {
            return Err("Invalid semantic boolean descriptor".into());
        }
    }
    let monuments = &values[F::MonumentIdentity.index()..F::MonumentIdentity.index() + 13];
    if monuments.iter().any(|n| ![0.0, 1.0].contains(n))
        || monuments.iter().sum::<f32>() != f32::from(values[384 + 19] == 1.0)
        || !(0.0..=CATALOG.buildings.len() as f32).contains(&values[F::BuildingIdentity.index()])
        || (values[F::BuildingIdentity.index()] > 0.0) != (values[384 + 18] == 1.0)
        || !(0.0..=4.0).contains(&values[F::SelectedTechnology.index()])
        || !(0.0..=3.0).contains(&values[F::SelectedTemple.index()])
        || !(0.0..=3.0).contains(&values[F::ActionSource.index()])
        || !(0.0..=4.0).contains(&values[F::HarvestKind.index()])
        || !(0.0..=3.0).contains(&values[F::TechnologyCurrent.index()])
        || !(0.0..=3.0).contains(&values[F::TechnologyTarget.index()])
    {
        return Err("Invalid semantic catalog/track identity".into());
    }
    Ok(())
}

pub(super) fn validate_observation(o: &Observation) -> Result<(), String> {
    if o.phase != Phase::Playing
        || o.skull_supply < 0
        || o.accumulated_corn < 0
        || o.players.iter().any(|p| {
            p.technologies.iter().any(|level| !(0..=3).contains(level))
                || TEMPLE_IDS
                    .iter()
                    .enumerate()
                    .any(|(i, temple)| !(-1..=temple_max(*temple)).contains(&p.temples[i]))
        })
        || o.jungle
            .values()
            .any(|box_| box_.corn < 0 || box_.wood < 0 || box_.wood > box_.corn)
        || matches!(
            o.pending_task,
            Some(
                Task::ChooseTribe
                    | Task::TribeSkipSpace
                    | Task::TechnologyBonus
                    | Task::QuickAction { .. }
                    | Task::FinishTurn { .. }
                    | Task::ProphecyGain { .. }
                    | Task::ProphecyTemple { .. }
                    | Task::FoodDay { .. }
                    | Task::Effects { .. }
            )
        )
    {
        return Err("Semantic Playing requires a supported base 3-4p decision".into());
    }
    Ok(())
}

pub(super) fn encode(
    o: &Observation,
    action: &TypedAction,
) -> Result<[f32; PLAYING_SEMANTIC_FEATURE_COUNT], String> {
    use PlayingSemanticFeature as F;
    let mut d = Descriptor([0; PLAYING_SEMANTIC_FEATURE_COUNT]);
    match action {
        TypedAction::Place {
            corn_cost,
            discount: false,
            ..
        }
        | TypedAction::FirstPlayer { corn_cost }
            if o.pending_task.is_none() =>
        {
            d.cost(o, [*corn_cost, 0, 0, 0, 0])?;
            if matches!(action, TypedAction::FirstPlayer { .. }) {
                d.add(F::PendingFirstPlayerCorn, o.accumulated_corn)?;
            }
        }
        TypedAction::Remove { gear, position } if o.pending_task.is_none() => {
            usize::try_from(*position)
                .ok()
                .and_then(|i| o.gears.get(gear)?.get(i)?.as_ref())
                .filter(|w| !w.dummy && w.player_id == o.actor as i64)
                .ok_or("Semantic Remove requires the actor's worker")?;
            d.add(F::WorkerReleased, 1)?;
            d.add(F::ActionChoice, 1)?;
        }
        TypedAction::Beg if o.pending_task.is_none() => {
            d.add(F::TempleDownChoice, 1)?;
            d.add(F::BegCornTarget, 3)?;
        }
        TypedAction::EndTurn { .. } if o.pending_task.is_none() => {
            if o.first_player_claimed == Some(o.actor) {
                d.gain(o, [o.accumulated_corn, 0, 0, 0, 0])?;
            }
        }
        TypedAction::Skip if o.pending_task.is_some() => {}
        TypedAction::UseAction {
            gear,
            position,
            source,
            corn_cost,
        } => {
            use_action(o, &mut d, *gear, *position, *source, *corn_cost)?;
        }
        TypedAction::Technology { technology } => {
            let Some(Task::Technology {
                remaining,
                free,
                mandatory,
            }) = o.pending_task
            else {
                return Err("Semantic Technology requires its pending task".into());
            };
            remaining_positive(remaining)?;
            let level = o.players[o.actor].technologies[technology_index(*technology)];
            d.add(F::TaskRemaining, remaining)?;
            d.add(F::TechnologyCurrent, level)?;
            d.add(F::TechnologyTarget, (level + 1).min(3))?;
            d.add(F::TechnologyFree, i64::from(free))?;
            d.add(F::TaskMandatory, i64::from(mandatory))?;
            d.add(
                F::SelectedTechnology,
                technology_index(*technology) as i64 + 1,
            )?;
            if free {
                advance(o, &mut d, *technology)?;
            } else {
                d.add(
                    F::MaterialPaymentRequired,
                    tzolkin_core::tribes::technology_step_cost(level, false),
                )?;
            }
        }
        TypedAction::Payment { resources } => {
            let amount = match o.pending_task {
                Some(Task::PayTechnology { amount, .. } | Task::PayResource { amount }) => amount,
                _ => return Err("Semantic Payment requires its pending amount".into()),
            };
            if resources[0] != 0
                || resources[4] != 0
                || resources[1..4]
                    .iter()
                    .try_fold(0i64, |sum, n| sum.checked_add(*n))
                    != Some(amount)
            {
                return Err("Semantic Payment differs from required materials".into());
            }
            d.cost(o, *resources)?;
            d.add(F::MaterialPaymentRequired, amount)?;
            if let Some(Task::PayTechnology { technology, .. }) = o.pending_task {
                let level = o.players[o.actor].technologies[technology_index(technology)];
                d.add(F::TechnologyCurrent, level)?;
                d.add(F::TechnologyTarget, (level + 1).min(3))?;
                d.add(
                    F::SelectedTechnology,
                    technology_index(technology) as i64 + 1,
                )?;
                advance(o, &mut d, technology)?;
            }
        }
        TypedAction::Temple { temple, direction } => {
            let Some(Task::Temple {
                remaining,
                direction: expected,
                reason,
                ref distinct,
            }) = o.pending_task
            else {
                return Err("Semantic Temple requires its pending task".into());
            };
            remaining_positive(remaining)?;
            if *direction != expected.unwrap_or(1) || ![-1, 1].contains(direction) {
                return Err("Semantic Temple direction differs from its task".into());
            }
            d.add(F::TaskRemaining, remaining)?;
            d.add(F::TempleDistinct, i64::from(distinct.is_some()))?;
            temple_step(o, &mut d, *temple, *direction)?;
            if reason == Some(TempleReason::Beg) {
                d.add(F::BegCornTarget, 3)?;
            }
        }
        TypedAction::Resource { resource } => {
            let Some(Task::Resource { remaining }) = o.pending_task else {
                return Err("Semantic Resource requires its pending task".into());
            };
            remaining_positive(remaining)?;
            d.add(F::TaskRemaining, remaining)?;
            let mut gain = [0; 5];
            gain[resource_index(*resource)] = 1;
            d.gain(o, gain)?;
        }
        TypedAction::Build {
            id,
            cost,
            architecture,
            renovation: None,
            ..
        } => {
            let Some(Task::Build {
                remaining,
                corn_payment,
                allow_monument,
                architecture_available,
                mandatory,
            }) = o.pending_task
            else {
                return Err("Semantic Build requires its pending task".into());
            };
            remaining_positive(remaining)?;
            if !o.buildings.contains(id) {
                return Err("Semantic building is outside the visible market".into());
            }
            let building = catalog::building(id).ok_or("Unknown semantic building")?;
            // The typed legal plan quotes the final cost/discount. This encoder
            // describes it; source consumers must still regenerate the legal set.
            if *architecture && architecture_available == Some(false) {
                return Err("Semantic Architecture is unavailable in this task".into());
            }
            d.cost(o, *cost)?;
            d.add(F::TaskRemaining, remaining)?;
            d.add(F::TaskMandatory, i64::from(mandatory))?;
            d.add(F::BuildingCornPayment, i64::from(corn_payment))?;
            d.add(F::BuildingAllowsMonument, i64::from(allow_monument))?;
            d.add(F::BuildingArchitecture, i64::from(*architecture))?;
            d.add(
                F::BuildingIdentity,
                CATALOG
                    .buildings
                    .iter()
                    .position(|b| &b.id == id)
                    .ok_or("Semantic building must be in the base catalog")? as i64
                    + 1,
            )?;
            for effect in &building.effects {
                effect_atom(&mut d, effect)?;
            }
            if building
                .effects
                .iter()
                .any(|e| matches!(e, Effect::SkullBuilding { .. }))
            {
                d.add(F::SkullPlaced, cost[4])?;
            }
            if *architecture {
                let level =
                    o.players[o.actor].technologies[technology_index(TechnologyId::Architecture)];
                if level < 1 {
                    return Err("Semantic Architecture bonus requires its technology".into());
                }
                d.gain(o, [1, 0, 0, 0, 0])?;
                d.add(F::ImmediatePoints, 2 * i64::from(level >= 2))?;
            }
        }
        TypedAction::Monument {
            id,
            cost,
            renovation: None,
        } => {
            if !matches!(
                o.pending_task,
                Some(
                    Task::Build {
                        allow_monument: true,
                        ..
                    } | Task::BuildMonument
                )
            ) || !o.monuments.contains(id)
            {
                return Err("Semantic Monument requires its visible pending choice".into());
            }
            d.cost(o, *cost)?;
            let monument = catalog::monument(id).ok_or("Unknown semantic monument")?;
            if *cost != RESOURCE_IDS.map(|r| monument.cost.get(&r).copied().unwrap_or(0)) {
                return Err("Semantic monument cost differs from its catalog".into());
            }
            d.at(
                F::MonumentIdentity as usize
                    + tzolkin_core::compact::catalog::MONUMENT_IDS
                        .iter()
                        .position(|known| *known == id)
                        .ok_or("Unknown semantic monument formula")?,
                1,
            )?;
        }
        TypedAction::TechnologyExchange { technology }
            if matches!(o.pending_task, Some(Task::TechnologyExchange)) =>
        {
            if o.players[o.actor].technologies[technology_index(*technology)] < 1 {
                return Err("Semantic exchange cannot lower this technology".into());
            }
            d.at(
                F::TechnologyDeltaAgriculture as usize + technology_index(*technology),
                -1,
            )?;
            d.add(
                F::SelectedTechnology,
                technology_index(*technology) as i64 + 1,
            )?;
            for t in TECHNOLOGY_IDS {
                if t != *technology {
                    advance(o, &mut d, t)?;
                }
            }
        }
        TypedAction::Trade { resource, buy } if matches!(o.pending_task, Some(Task::Trade)) => {
            let rate = match resource {
                Resource::Wood => 2,
                Resource::Stone => 3,
                Resource::Gold => 4,
                _ => return Err("Semantic market requires an ordinary material".into()),
            };
            let mut cost = [0; 5];
            let mut gain = [0; 5];
            if *buy {
                cost[0] = rate;
                gain[resource_index(*resource)] = 1;
            } else {
                cost[resource_index(*resource)] = 1;
                gain[0] = rate;
            }
            d.cost(o, cost)?;
            d.gain(o, gain)?;
            d.add(F::TradeChoice, 1)?;
        }
        TypedAction::Harvest { kind, position } => harvest(o, &mut d, *kind, *position)?,
        TypedAction::Offering { resource }
            if matches!(o.pending_task, Some(Task::Theology { .. })) =>
        {
            if !MATERIALS.contains(resource) {
                return Err("Semantic Offering requires a material".into());
            }
            let mut cost = [0; 5];
            cost[resource_index(*resource)] = 1;
            d.cost(o, cost)?;
            d.add(F::TempleChoice, 1)?;
        }
        TypedAction::Rotate { days }
            if matches!(o.pending_task, Some(Task::Rotation)) && [1, 2].contains(days) =>
        {
            // Rotation distance retains its canonical position column (416),
            // rather than overloading the pending-choice remaining count.
        }
        _ => return Err("Unsupported or mismatched semantic Playing operation".into()),
    }
    Ok(d.0.map(|n| n as f32))
}

fn remaining_positive(remaining: i64) -> Result<(), String> {
    if remaining <= 0 {
        Err("Semantic pending count must be positive".into())
    } else {
        Ok(())
    }
}
fn temple_max(t: TempleId) -> i64 {
    CATALOG.temple_tracks[&t].points.len() as i64 - 2
}
fn temple_step(
    o: &Observation,
    d: &mut Descriptor,
    t: TempleId,
    direction: i64,
) -> Result<(), String> {
    use PlayingSemanticFeature as F;
    let i = temple_index(t);
    let current = o.players[o.actor].temples[i];
    let target = if direction == -1 {
        if current <= -1 {
            return Err("Semantic Temple cannot fall below its track".into());
        }
        current - 1
    } else if current < temple_max(t)
        && (current + 1 != temple_max(t)
            || o.players
                .iter()
                .all(|p| p.id == o.actor || p.temples[i] != temple_max(t)))
    {
        current + 1
    } else {
        current
    };
    d.add(F::TempleCurrent, current)?;
    d.add(F::TempleTarget, target)?;
    d.add(F::SelectedTemple, i as i64 + 1)?;
    d.at(F::TempleDeltaChaac as usize + i, target - current)
}
fn advance(o: &Observation, d: &mut Descriptor, t: TechnologyId) -> Result<(), String> {
    use PlayingSemanticFeature as F;
    let level = o.players[o.actor].technologies[technology_index(t)];
    d.at(
        F::TechnologyDeltaAgriculture as usize + technology_index(t),
        i64::from(level < 3),
    )?;
    if level == 3 {
        d.add(F::CappedTechnologyBonus, 1)?;
        match t {
            TechnologyId::Agriculture => d.add(F::TempleChoice, 1)?,
            TechnologyId::Extraction => d.add(F::ResourceChoice, 2)?,
            TechnologyId::Architecture => d.add(F::ImmediatePoints, 3)?,
            TechnologyId::Theology => d.gain(o, [0, 0, 0, 0, 1])?,
        }
    }
    Ok(())
}

fn use_action(
    o: &Observation,
    d: &mut Descriptor,
    gear: GearId,
    pos: i64,
    source: ActionSource,
    cost: i64,
) -> Result<(), String> {
    use PlayingSemanticFeature as F;
    let maximum = if gear == GearId::ChichenItza { 9 } else { 5 };
    let expected = match (&o.pending_task, source) {
        (
            Some(Task::Action {
                gear: g,
                position,
                free,
            }),
            ActionSource::CurrentGear,
        ) if *g == gear
            && (1..=if *position > maximum {
                maximum
            } else {
                *position
            })
                .contains(&pos) =>
        {
            d.add(F::ActionSource, 1)?;
            if *position > maximum || *free == Some(true) {
                0
            } else {
                *position - pos
            }
        }
        (
            Some(Task::Action {
                gear: GearId::ChichenItza,
                position,
                ..
            }),
            ActionSource::TheologyAhead,
        ) if gear == GearId::ChichenItza
            && *position < 10
            && pos == *position + 1
            && o.players[o.actor].technologies[technology_index(TechnologyId::Theology)] >= 1 =>
        {
            d.add(F::ActionSource, 2)?;
            0
        }
        (
            Some(Task::AnyAction {
                cost,
                exclude_skulls,
            }),
            ActionSource::Anywhere,
        ) if (1..=maximum).contains(&pos)
            && !(gear == GearId::ChichenItza && *exclude_skulls == Some(true)) =>
        {
            d.add(F::ActionSource, 3)?;
            cost.unwrap_or(0)
        }
        _ => return Err("Semantic action source/position differs from its pending task".into()),
    };
    if cost != expected {
        return Err("Semantic action cost differs from its pending task".into());
    }
    let extra = if gear == GearId::Uxmal && pos == 1 {
        3
    } else {
        0
    };
    d.cost(
        o,
        [
            cost.checked_add(extra)
                .ok_or("Semantic action cost overflow")?,
            0,
            0,
            0,
            0,
        ],
    )?;
    match (gear, pos) {
        (GearId::Palenque, 1) => d.gain(
            o,
            [
                tzolkin_core::engine::palenque_fishing_quantity(
                    o.players[o.actor].technologies[0] >= 2,
                ),
                0,
                0,
                0,
                0,
            ],
        )?,
        (GearId::Palenque, 2..=5) => {
            d.add(F::HarvestChoice, 1)?;
        }
        (GearId::Yaxchilan, 1..=5) => {
            let ex = o.players[o.actor].technologies[1];
            let gain = tzolkin_core::engine::yaxchilan_resource_quantities(
                pos,
                [
                    ex >= 1,
                    ex >= 2,
                    ex >= 3,
                    o.players[o.actor].technologies[3] >= 3,
                ],
            )
            .ok_or("Unknown semantic Yaxchilan endpoint")?;
            d.gain(o, gain)?;
        }
        (GearId::Tikal, 1 | 3) => d.add(F::TechnologyChoice, if pos == 3 { 2 } else { 1 })?,
        (GearId::Tikal, 2 | 4) => {
            d.add(F::BuildingChoice, if pos == 4 { 2 } else { 1 })?;
            d.add(F::MonumentChoice, i64::from(pos == 4))?;
        }
        (GearId::Tikal, 5) => {
            d.add(F::MaterialPaymentRequired, 1)?;
            d.add(F::TempleChoice, 2)?;
        }
        (GearId::Uxmal, 1) => d.add(F::TempleChoice, 1)?,
        (GearId::Uxmal, 2) => d.add(F::TradeChoice, 1)?,
        (GearId::Uxmal, 3) => d.add(F::WorkerAdded, i64::from(o.players[o.actor].workers < 6))?,
        (GearId::Uxmal, 4) => d.add(F::BuildingChoice, 1)?,
        (GearId::Uxmal, 5) => {
            d.add(F::AnyActionChoice, 1)?;
            d.add(F::AnyActionCornRequired, 1)?;
        }
        (GearId::ChichenItza, 10) if source == ActionSource::TheologyAhead => {
            d.add(F::ActionChoice, 1)?
        }
        (GearId::ChichenItza, 1..=9) => {
            if o.skull_spaces[pos as usize].is_some() || o.players[o.actor].resources[4] < 1 {
                return Err("Semantic skull action requires an empty space and Skull".into());
            }
            d.cost(o, [0, 0, 0, 0, 1])?;
            let reward = CATALOG
                .skull_rewards
                .get(&pos)
                .ok_or("Unknown semantic skull reward")?;
            d.add(F::SkullPlaced, 1)?;
            d.add(F::ImmediatePoints, reward.points)?;
            temple_step(o, d, reward.temple, 1)?;
            d.add(F::ResourceChoice, i64::from(reward.resource))?;
            d.add(
                F::OfferingChoice,
                i64::from(o.players[o.actor].technologies[3] >= 2),
            )?;
        }
        _ => return Err("Unsupported semantic gear endpoint".into()),
    }
    Ok(())
}

fn harvest(o: &Observation, d: &mut Descriptor, kind: HarvestKind, pos: i64) -> Result<(), String> {
    use PlayingSemanticFeature as F;
    if o.pending_task != Some(Task::Palenque { position: pos }) || !(2..=5).contains(&pos) {
        return Err("Semantic Harvest requires its pending Palenque task".into());
    }
    let jungle = o.jungle.get(&pos).ok_or("Missing semantic jungle")?;
    let p = &o.players[o.actor];
    let can_corn = jungle.corn > jungle.wood;
    let ordinal = match kind {
        HarvestKind::Corn if can_corn => 1,
        HarvestKind::Wood if pos >= 3 && jungle.wood > 0 => 2,
        HarvestKind::Burn
            if pos >= 3
                && jungle.wood > 0
                && jungle.corn > 0
                && p.temples.iter().any(|level| *level > -1) =>
        {
            3
        }
        HarvestKind::EmptyCorn if !can_corn && p.technologies[0] >= 2 => 4,
        _ => return Err("Semantic Harvest contradicts visible tiles/technology".into()),
    };
    d.add(F::HarvestKind, ordinal)?;
    if kind == HarvestKind::Wood {
        d.gain(
            o,
            [
                0,
                tzolkin_core::engine::palenque_wood_quantity(pos, p.technologies[1] >= 1),
                0,
                0,
                0,
            ],
        )?;
        d.add(F::WoodTilesConsumed, 1)?;
        d.add(F::WoodTilesAdded, 1)?;
    } else {
        d.gain(
            o,
            [
                tzolkin_core::engine::palenque_corn_quantity(pos, p.technologies[0])
                    .ok_or("Unknown semantic Palenque endpoint")?,
                0,
                0,
                0,
                0,
            ],
        )?;
        if kind != HarvestKind::EmptyCorn {
            d.add(F::CornTilesConsumed, 1)?;
        }
        if kind == HarvestKind::Burn {
            d.add(F::WoodTilesConsumed, 1)?;
            d.add(F::TempleDownChoice, 1)?;
        }
    }
    Ok(())
}

fn effect_atom(d: &mut Descriptor, effect: &Effect) -> Result<(), String> {
    use PlayingSemanticFeature as F;
    let resource_atoms = |r: &Resources| RESOURCE_IDS.map(|id| r.get(&id).copied().unwrap_or(0));
    match effect {
        Effect::Resources { resources } => {
            d.resources(F::CatalogResourceCorn, &resource_atoms(resources))
        }
        Effect::Feed {
            workers: FeedWorkers::Count(n),
        } => d.add(F::CatalogFedWorkers, *n),
        Effect::Feed {
            workers: FeedWorkers::All(_),
        } => d.add(F::CatalogFeedAll, 1),
        Effect::FeedDiscount { amount } => d.add(F::CatalogFeedDiscount, *amount),
        Effect::Technology { technology, steps } => match technology {
            Target::Specific(t) => d.at(
                F::CatalogTechnologyAgriculture as usize + technology_index(*t),
                steps.unwrap_or(1),
            ),
            Target::Any(_) => d.add(F::TechnologyChoice, steps.unwrap_or(1)),
        },
        Effect::Temple { temple, steps } => match temple {
            Target::Specific(t) => d.at(
                F::CatalogTempleChaac as usize + temple_index(*t),
                steps.unwrap_or(1),
            ),
            Target::Any(_) => d.add(F::TempleChoice, steps.unwrap_or(1)),
        },
        Effect::Trade => d.add(F::TradeChoice, 1),
        Effect::Build => d.add(F::BuildingChoice, 1),
        Effect::BuildMonument => d.add(F::MonumentChoice, 1),
        Effect::Renovation => Err("Renovation is outside base semantic Playing".into()),
        Effect::FoodReward { resources } => {
            d.resources(F::FoodRewardCorn, &resource_atoms(resources))
        }
        Effect::FoodRewardSwitch => d.add(F::FoodRewardSwitch, 1),
        Effect::SkullBuilding { points, temple } => {
            d.add(F::SkullBuilding, 1)?;
            d.add(F::CatalogPoints, *points)?;
            match temple {
                Target::Specific(t) => d.at(F::CatalogTempleChaac as usize + temple_index(*t), 1),
                Target::Any(_) => d.add(F::TempleChoice, 1),
            }
        }
        Effect::TechnologyExchange => {
            // This instruction has no chosen lowered track yet.
            d.add(F::TechnologyExchangeChoice, 1)
        }
        Effect::Points { amount } => d.add(F::CatalogPoints, *amount),
        Effect::Worker => d.add(F::CatalogWorker, 1),
        Effect::Action { anywhere } => {
            d.add(F::AnyActionChoice, 1)?;
            d.add(F::AnyActionCornRequired, 1)?;
            d.add(
                F::AnyActionAllowsSkulls,
                i64::from(anywhere.unwrap_or(false)),
            )
        }
    }
}
