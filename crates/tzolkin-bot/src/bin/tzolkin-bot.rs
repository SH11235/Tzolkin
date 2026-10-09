//! Native entry point for the shipped policy: an observation on stdin, the decision on stdout.
use std::io::Read;

const MAX_INPUT: u64 = 16 * 1024 * 1024;

fn run() -> Result<(), String> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args != ["choose"] {
        return Err("usage: tzolkin-bot choose (observation JSON on stdin)".into());
    }
    let mut request = String::new();
    std::io::stdin()
        .take(MAX_INPUT + 1)
        .read_to_string(&mut request)
        .map_err(|e| e.to_string())?;
    if request.len() as u64 > MAX_INPUT {
        return Err("CPU input exceeds 16 MiB".into());
    }
    println!("{}", tzolkin_bot::dispatch_cpu(&request)?);
    Ok(())
}

fn main() {
    if let Err(error) = run() {
        eprintln!("{error}");
        std::process::exit(1);
    }
}
