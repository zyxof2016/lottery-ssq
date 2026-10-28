fn main() -> Result<(), String> {
    let args: Vec<_> = std::env::args().collect();
    let file = args.get(1).map(String::as_str).unwrap_or("data/ssq.csv");
    let d = ssq::draw::load_csv(std::path::Path::new(file))?;
    let mut report = ssq::blue::analyze(&d)?;
    report["input_file_sha256"] = serde_json::json!(ssq::hash::sha256_hex(
        &std::fs::read(file).map_err(|e| e.to_string())?
    ));
    if args.iter().any(|a| a == "--features") {
        let snap = ssq::model::roll_all(&d)?;
        let (mut sums, mut n, mut warm_hits) = ([0.; 6], 0usize, [0usize; 6]);
        let rows:Vec<_>=snap.rows.iter().filter_map(|r|{
            let out=(n>=30).then(||serde_json::json!({"issue":r.issue,"winner":r.blue,"p":r.blue_p,"baseline":ssq::select::fuse6(&r.blue_p,&ssq::select::softmax_shrink(&sums,n as f64))}));
            if n<30{for e in 0..6{warm_hits[e]+=usize::from(ssq::select::rank_desc(&r.blue_p[e])[..2].contains(&(r.blue as usize)));}}
            if r.issue<ssq::protocol::Protocol::CONFIRM_START_ISSUE{for e in 0..6{sums[e]+=r.blue_p[e][r.blue as usize].clamp(1e-12,1.).ln();}n+=1;}out
        }).collect();
        report["features"] = serde_json::json!({"rows":rows,"warmup_hits":warm_hits,"warmup_periods":30,"draws":d.iter().map(|d|serde_json::json!({"issue":d.issue,"blue":d.blue-1})).collect::<Vec<_>>()});
    }
    println!(
        "{}",
        serde_json::to_string_pretty(&report).map_err(|e| e.to_string())?
    );
    Ok(())
}
