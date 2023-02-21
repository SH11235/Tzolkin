pub trait Increment {
    fn increment(&mut self);
}
impl Increment for u32 {
    fn increment(&mut self) {
        *self += 1;
    }
}
