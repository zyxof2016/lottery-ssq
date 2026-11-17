fn main() -> Result<(), String> {
    let args: Vec<_> = std::env::args().collect();
    let draws = ssq::draw::load_csv(std::path::Path::new(args.get(1).ok_or("需要CSV路径")?))?;
    let root = std::path::Path::new(args.get(2).ok_or("需要recommendations路径")?);
    let report = ssq::fusion_v3::report(root, &draws, args.get(3).map(String::as_str))?;
    println!(
        "{}",
        serde_json::to_string_pretty(&report).map_err(|e| e.to_string())?
    );
    Ok(())
}
