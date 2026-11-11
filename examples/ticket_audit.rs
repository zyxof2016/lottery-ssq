fn main() -> Result<(), String> {
    let args: Vec<_> = std::env::args().collect();
    let draws = ssq::draw::load_csv(std::path::Path::new(args.get(1).ok_or("需要CSV路径")?))?;
    let value = if args.iter().any(|s| s == "--replay") {
        ssq::ticket_audit::replay(&draws)?
    } else {
        ssq::ticket_audit::report(
            std::path::Path::new(args.get(2).ok_or("需要原档目录")?),
            &draws,
        )?
    };
    println!(
        "{}",
        serde_json::to_string_pretty(&value).map_err(|e| e.to_string())?
    );
    Ok(())
}
