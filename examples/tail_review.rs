fn main() -> Result<(), String> {
    let path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "data/ssq.csv".into());
    let draws = ssq::draw::load_csv(std::path::Path::new(&path))?;
    let (mut report, next) = ssq::tail::replay(&draws)?;
    report["next_decision"] = serde_json::json!({"accepted":next.accepted,"rank_positions":next.order.map(|n|n+1).to_vec(),"evidence":next.evidence});
    println!(
        "{}",
        serde_json::to_string_pretty(&report).map_err(|e| e.to_string())?
    );
    Ok(())
}
