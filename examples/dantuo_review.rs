fn main() -> Result<(), String> {
    let args: Vec<_> = std::env::args().collect();
    let file = args.get(1).map(String::as_str).unwrap_or("data/ssq.csv");
    let d = ssq::draw::load_csv(std::path::Path::new(file))?;
    let report = if args.iter().any(|s| s == "--replay") {
        ssq::dantuo_audit::replay(&d)?
    } else {
        ssq::dantuo_audit::report(
            std::path::Path::new(
                args.get(2)
                    .map(String::as_str)
                    .unwrap_or("/nonexistent-ssq-runtime"),
            ),
            &d,
            args.iter().find_map(|a| a.strip_prefix("--issue=")),
        )?
    };
    println!(
        "{}",
        serde_json::to_string_pretty(&report).map_err(|e| e.to_string())?
    );
    Ok(())
}
