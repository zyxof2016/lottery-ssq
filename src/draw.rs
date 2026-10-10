use std::fs;
use std::path::Path;
use std::process::Command;

use crate::hash::sha256_hex;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Draw {
    pub issue: u32,
    pub date: i32,
    pub red: [u8; 6],
    pub blue: u8,
    pub red_mask: u64,
}

impl Draw {
    pub fn from_parts(issue: u32, date: i32, mut red: [u8; 6], blue: u8) -> Result<Self, String> {
        red.sort_unstable();
        validate_balls(&red, blue)?;
        let mut red_mask = 0u64;
        for r in red {
            red_mask |= 1u64 << (r - 1);
        }
        if red_mask.count_ones() != 6 {
            return Err(format!("期号 {issue} 红球有重复"));
        }
        Ok(Self {
            issue,
            date,
            red,
            blue,
            red_mask,
        })
    }
}

fn validate_balls(red: &[u8; 6], blue: u8) -> Result<(), String> {
    for (i, &r) in red.iter().enumerate() {
        if !(1..=33).contains(&r) {
            return Err(format!("红球 {r} 超出 1-33"));
        }
        if i > 0 && red[i] == red[i - 1] {
            return Err(format!("红球 {r} 重复"));
        }
    }
    if !(1..=16).contains(&blue) {
        return Err(format!("蓝球 {blue} 超出 1-16"));
    }
    Ok(())
}

pub fn fmt_issue(issue: u32) -> String {
    format!("{issue:05}")
}

pub fn fmt_date(yyyymmdd: i32) -> String {
    format!(
        "{:04}-{:02}-{:02}",
        yyyymmdd / 10000,
        (yyyymmdd / 100) % 100,
        yyyymmdd % 100
    )
}

pub fn canonical_csv(draws: &[Draw]) -> String {
    let mut s = String::from("issue,date,r1,r2,r3,r4,r5,r6,blue\n");
    for d in draws {
        s.push_str(&format!(
            "{},{},{:02},{:02},{:02},{:02},{:02},{:02},{:02}\n",
            fmt_issue(d.issue),
            fmt_date(d.date),
            d.red[0],
            d.red[1],
            d.red[2],
            d.red[3],
            d.red[4],
            d.red[5],
            d.blue
        ));
    }
    s
}

pub fn data_sha256(draws: &[Draw]) -> String {
    sha256_hex(canonical_csv(draws).as_bytes())
}

pub fn parse_csv(text: &str) -> Result<Vec<Draw>, String> {
    let mut draws = Vec::new();
    for (lineno, line) in text.lines().enumerate() {
        let line = line.trim().trim_start_matches('\u{feff}');
        if line.is_empty() {
            continue;
        }
        let cells: Vec<&str> = line.split(',').map(|c| c.trim()).collect();
        if lineno == 0 && cells.first().is_some_and(|c| !c.chars().all(|ch| ch.is_ascii_digit()))
        {
            continue;
        }
        if cells.len() != 9 {
            return Err(format!("第 {} 行应有 9 列，实际 {}", lineno + 1, cells.len()));
        }
        let issue: u32 = cells[0]
            .parse()
            .map_err(|_| format!("第 {} 行期号无法解析", lineno + 1))?;
        let date = parse_date(cells[1]).map_err(|e| format!("第 {} 行{e}", lineno + 1))?;
        let mut red = [0u8; 6];
        for i in 0..6 {
            red[i] = cells[i + 2]
                .parse()
                .map_err(|_| format!("第 {} 行红球无法解析", lineno + 1))?;
        }
        let blue: u8 = cells[8]
            .parse()
            .map_err(|_| format!("第 {} 行蓝球无法解析", lineno + 1))?;
        let draw = Draw::from_parts(issue, date, red, blue)
            .map_err(|e| format!("第 {} 行{e}", lineno + 1))?;
        draws.push(draw);
    }
    if draws.is_empty() {
        return Err("开奖数据为空".into());
    }
    draws.sort_by_key(|d| d.issue);
    for w in draws.windows(2) {
        if w[1].issue <= w[0].issue {
            return Err(format!("期号重复：{}", fmt_issue(w[1].issue)));
        }
    }
    Ok(draws)
}

pub fn load_csv(path: &Path) -> Result<Vec<Draw>, String> {
    let text = fs::read_to_string(path)
        .map_err(|e| format!("无法读取 {}: {e}", path.display()))?;
    parse_csv(&text)
}

pub fn write_csv(path: &Path, draws: &[Draw]) -> Result<(), String> {
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent).map_err(|e| format!("无法创建目录 {}: {e}", parent.display()))?;
        }
    }
    fs::write(path, canonical_csv(draws))
        .map_err(|e| format!("无法写入 {}: {e}", path.display()))
}

fn parse_date(s: &str) -> Result<i32, String> {
    let p: Vec<&str> = s.split('-').collect();
    if p.len() != 3 {
        return Err(format!("日期 {s} 不是 YYYY-MM-DD"));
    }
    let y: i32 = p[0].parse().map_err(|_| format!("日期 {s} 无法解析"))?;
    let m: i32 = p[1].parse().map_err(|_| format!("日期 {s} 无法解析"))?;
    let d: i32 = p[2].parse().map_err(|_| format!("日期 {s} 无法解析"))?;
    if !(1..=12).contains(&m) || !(1..=31).contains(&d) {
        return Err(format!("日期 {s} 越界"));
    }
    Ok(y * 10000 + m * 100 + d)
}

pub fn fetch_500(out: &Path) -> Result<usize, String> {
    let tmp = std::env::temp_dir().join(format!("ssq-500-{}.html", std::process::id()));
    let url = "https://datachart.500.com/ssq/history/newinc/history.php?start=03001&end=99999";
    let status = Command::new("curl")
        .args([
            "-fsSL",
            "--max-time",
            "90",
            "-A",
            "Mozilla/5.0",
            "-o",
        ])
        .arg(&tmp)
        .arg(url)
        .status()
        .map_err(|e| format!("无法启动 curl：{e}"))?;
    if !status.success() {
        return Err(format!("下载双色球历史失败，curl 退出码 {status}"));
    }
    let html = fs::read_to_string(&tmp).map_err(|e| format!("无法读取下载结果：{e}"))?;
    let _ = fs::remove_file(&tmp);
    let draws = parse_500_html(&html)?;
    let n = draws.len();
    write_csv(out, &draws)?;
    Ok(n)
}

pub fn parse_500_html(html: &str) -> Result<Vec<Draw>, String> {
    let Some(start) = html.find("id=\"tdata\"") else {
        return Err("页面里没有 tdata 表体".into());
    };
    let after = &html[start..];
    let Some(gt) = after.find('>') else {
        return Err("tdata 标签不完整".into());
    };
    let body = &after[gt + 1..];
    let end = body.find("</tbody>").unwrap_or(body.len());
    let body = &body[..end];
    let mut draws = Vec::new();
    for chunk in body.split("<tr class=\"t_tr1\">").skip(1) {
        let cells = extract_tds(chunk);
        if cells.len() < 9 {
            continue;
        }
        let issue: u32 = match cells[0].parse() {
            Ok(v) => v,
            Err(_) => continue,
        };
        let mut red = [0u8; 6];
        let mut ok = true;
        for i in 0..6 {
            match cells[i + 1].parse::<u8>() {
                Ok(v) => red[i] = v,
                Err(_) => ok = false,
            }
        }
        let blue = match cells[7].parse::<u8>() {
            Ok(v) => v,
            Err(_) => continue,
        };
        let date = match parse_date(cells.last().map(|s| s.as_str()).unwrap_or("")) {
            Ok(v) => v,
            Err(_) => continue,
        };
        if !ok {
            continue;
        }
        draws.push(Draw::from_parts(issue, date, red, blue)?);
    }
    if draws.is_empty() {
        return Err("页面中没有解析到开奖记录".into());
    }
    draws.sort_by_key(|d| d.issue);
    for w in draws.windows(2) {
        if w[1].issue == w[0].issue {
            return Err(format!("来源数据期号重复：{}", fmt_issue(w[1].issue)));
        }
    }
    Ok(draws)
}

fn extract_tds(row: &str) -> Vec<String> {
    let row = strip_comments(row);
    let mut out = Vec::new();
    let mut rest = row.as_str();
    while let Some(start) = rest.find("<td") {
        let after = &rest[start..];
        let Some(gt) = after.find('>') else { break };
        let content = &after[gt + 1..];
        let Some(end) = content.find("</td>") else { break };
        out.push(clean_cell(&content[..end]));
        rest = &content[end + 5..];
    }
    out
}

fn strip_comments(input: &str) -> String {
    let mut out = String::with_capacity(input.len());
    let mut rest = input;
    while let Some(start) = rest.find("<!--") {
        out.push_str(&rest[..start]);
        if let Some(end) = rest[start + 4..].find("-->") {
            rest = &rest[start + 4 + end + 3..];
        } else {
            rest = "";
            break;
        }
    }
    out.push_str(rest);
    out
}

fn clean_cell(raw: &str) -> String {
    let mut s = String::new();
    let mut in_tag = false;
    for ch in raw.chars() {
        if ch == '<' {
            in_tag = true;
            continue;
        }
        if ch == '>' {
            in_tag = false;
            continue;
        }
        if !in_tag {
            s.push(ch);
        }
    }
    s.replace("&nbsp;", " ")
        .replace('\u{00a0}', " ")
        .trim()
        .to_string()
}

pub fn synth_draws(n: usize, seed: u64) -> Vec<Draw> {
    let mut rng = crate::stats::Rng::new(seed);
    let mut draws = Vec::with_capacity(n);
    for i in 0..n {
        let mask = rng.sample_red_mask();
        let mut red = [0u8; 6];
        let mut k = 0;
        for b in 0..33 {
            if (mask >> b) & 1 == 1 {
                red[k] = b as u8 + 1;
                k += 1;
            }
        }
        let blue = rng.gen_range(16) as u8 + 1;
        let day = (i % 28) as i32 + 1;
        let month = ((i / 28) % 12) as i32 + 1;
        let year = 2003 + ((i / 28) / 12) as i32;
        let date = year * 10000 + month * 100 + day;
        draws.push(
            Draw::from_parts(3001 + i as u32, date, red, blue).expect("合成数据合法"),
        );
    }
    draws
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn csv_roundtrip_and_bundle() {
        let text = "\
issue,date,r1,r2,r3,r4,r5,r6,blue
03001,2003-02-23,10,11,12,13,26,28,11
03002,2003-02-27,04,09,19,20,21,26,12
";
        let draws = parse_csv(text).unwrap();
        assert_eq!(draws.len(), 2);
        assert_eq!(canonical_csv(&draws), text.replace("\r\n", "\n"));
        let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("data/ssq.csv");
        let all = load_csv(&path).unwrap();
        assert!(all.len() > 3000);
        assert_eq!(all[0].issue, 3001);
        assert_eq!(all[0].red, [10, 11, 12, 13, 26, 28]);
        assert_eq!(all[0].blue, 11);
        let file = fs::read_to_string(&path).unwrap().replace("\r\n", "\n");
        assert_eq!(file, canonical_csv(&all));
    }

    #[test]
    fn html_snippet() {
        let html = r#"
<tbody id="tdata">
<tr class="t_tr1"><!--<td>2</td>--><td>03001</td><td>10</td><td>11</td><td>12</td><td>13</td><td>26</td><td>28</td><td>11</td><td>&nbsp;</td><td>1</td><td>2003-02-23</td></tr>
</tbody>
"#;
        let draws = parse_500_html(html).unwrap();
        assert_eq!(draws.len(), 1);
        assert_eq!(draws[0].issue, 3001);
        assert_eq!(draws[0].blue, 11);
    }
}
