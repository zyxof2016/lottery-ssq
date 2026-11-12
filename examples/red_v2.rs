fn main() -> Result<(), String> {
    let args: Vec<_> = std::env::args().collect();
    let draws = ssq::draw::load_csv(std::path::Path::new(args.get(1).ok_or("需要CSV路径")?))?;
    let root = std::path::Path::new(args.get(2).map(String::as_str).unwrap_or("/nonexistent-ssq-red-v2"));
    let v = ssq::red_score_v2::report(root, &draws)?;
    println!("{}", serde_json::to_string_pretty(&v).map_err(|e| e.to_string())?);
    Ok(())
}
