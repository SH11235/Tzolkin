pub mod palenque;
pub mod yaxchilan;
pub mod tikal;
pub mod uxmal;
pub mod chichen_itza;

#[derive(Debug)]
pub enum WorkerPosition {
    Hand,
    Palenque(u32),
    Yaxchilan(u32),
    Tikal(u32),
    Uxmal(u32),
    ChichenItza(u32),
    StartPlayer,
    Locked,
    // QuickActions,
}
