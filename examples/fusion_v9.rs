//! Read-only causal replay of the v9 cross-model successful-small-pool consensus.
//! ```sh
//! cargo run --release --locked --example fusion_v9 -- data/ssq.csv > fusion-v9-replay.json
//! ```
fn main() -> Result<(), String> {
    let args: Vec<_> = std::env::args().collect();
    let draws = ssq::draw::load_csv(std::path::Path::new(args.get(1).ok_or("需要CSV路径")?))?;
    let report = ssq::fusion_v9_study::replay(&draws)?;
    println!(
        "{}",
        serde_json::to_string_pretty(&report).map_err(|e| e.to_string())?
    );
    Ok(())
}
