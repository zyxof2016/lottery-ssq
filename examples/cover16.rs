fn main() -> Result<(), String> {
    let a: Vec<_> = std::env::args().collect();
    let d = ssq::draw::load_csv(std::path::Path::new(a.get(1).ok_or("需要CSV")?))?;
    println!(
        "{}",
        serde_json::to_string_pretty(&ssq::cover16::review(&d)?).map_err(|e| e.to_string())?
    );
    Ok(())
}
