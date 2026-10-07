use std::io::{self, BufRead, Write};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let stdin = io::stdin();
    let mut stdout = io::BufWriter::new(io::stdout().lock());
    for line in stdin.lock().lines() {
        let result = match tzolkin_core::api::dispatch_game(&line?) {
            Ok(response) => {
                serde_json::json!({"result": serde_json::from_str::<serde_json::Value>(&response)?})
            }
            Err(error) => serde_json::json!({"error": error}),
        };
        writeln!(stdout, "{result}")?;
        stdout.flush()?;
    }
    Ok(())
}
