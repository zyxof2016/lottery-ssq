fn main() -> Result<(), String> {
    let args: Vec<_> = std::env::args().collect();
    let draws = ssq::draw::load_csv(std::path::Path::new(
        args.get(1).map(String::as_str).unwrap_or("data/ssq.csv"),
    ))?;
    let value = if args.iter().any(|s| s == "--replay") {
        ssq::fusion_review::replay(&draws)?
    } else {
        ssq::fusion_review::report(
            std::path::Path::new(
                args.get(2)
                    .map(String::as_str)
                    .unwrap_or("/nonexistent-ssq-runtime"),
            ),
            &draws,
        )?
    };
    println!(
        "{}",
        serde_json::to_string_pretty(&value).map_err(|e| e.to_string())?
    );
    Ok(())
}
