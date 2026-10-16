fn main() -> Result<(), String> {
    let path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "data/ssq.csv".into());
    let draws = ssq::draw::load_csv(std::path::Path::new(&path))?;
    println!(
        "{}",
        serde_json::to_string_pretty(&ssq::conditional::analyze(&draws)?)
            .map_err(|e| e.to_string())?
    );
    Ok(())
}
