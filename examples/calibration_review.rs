fn main() -> Result<(), String> {
    let args: Vec<_> = std::env::args().collect();
    let file = args.get(1).map(String::as_str).unwrap_or("data/ssq.csv");
    let draws = ssq::draw::load_csv(std::path::Path::new(file))?;
    let root = args
        .get(2)
        .map(String::as_str)
        .unwrap_or("/nonexistent-ssq-runtime");
    let mut report = ssq::calibration::report(std::path::Path::new(root), &draws)?;
    if args.iter().any(|s| s == "--features") {
        let study = ssq::calibration_model_v1::study(&draws)?;
        report["training_features"] = serde_json::json!(study
            .samples
            .iter()
            .filter(|s| s.row.issue <= ssq::calibration_model_v1::TRAIN_END)
            .map(|s| serde_json::json!({"issue":s.row.issue,"red":s.red,"blue":s.blue}))
            .collect::<Vec<_>>());
    }
    println!(
        "{}",
        serde_json::to_string_pretty(&report).map_err(|e| e.to_string())?
    );
    Ok(())
}
