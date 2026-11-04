fn main() -> Result<(), String> {
    let a: Vec<_> = std::env::args().collect();
    let d = ssq::draw::load_csv(std::path::Path::new(
        a.get(1).map(String::as_str).unwrap_or("data/ssq.csv"),
    ))?;
    let root = std::path::Path::new(a.get(2).map(String::as_str).unwrap_or("recommendations"));
    println!(
        "{}",
        serde_json::to_string_pretty(&ssq::blue_deep::report(root, &d)?)
            .map_err(|e| e.to_string())?
    );
    Ok(())
}
