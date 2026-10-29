//! Compare an imported reference trace with causal local rankings; not prospective evidence.
use serde_json::{json,Value};
use ssq::{draw,model,protocol::Protocol,select};
use std::collections::BTreeMap;
fn main()->Result<(),String>{
 let args:Vec<_>=std::env::args().collect();let draws=draw::load_csv(std::path::Path::new(args.get(1).ok_or("CSV required")?))?;
 let imported:Value=serde_json::from_slice(&std::fs::read(args.get(2).ok_or("reference trace required")?).map_err(|e|e.to_string())?).map_err(|e|e.to_string())?;
 let records:BTreeMap<u32,&Value>=imported["records"].as_array().ok_or("records")?.iter().map(|r|(r["issue"].as_str().unwrap().parse().unwrap(),r)).collect();
 let roll=model::roll_all(&draws)?;let(mut sums,mut n)=([0.;7],0usize);let mut rows=Vec::new();let mut excluded=Vec::new();
 for r in &roll.rows {
  if let Some(reference)=records.get(&r.issue){
   let tickets=reference["tickets"].as_array().ok_or("tickets")?;
   if tickets.len()!=5{excluded.push(r.issue);}else{
    let original=select::fuse7(&r.expert_p,&select::softmax_shrink(&sums,n as f64));
    let small=ssq::small_red::scores(&original,&r.expert_p);let mut exposure=[0.;33];let mut union=0u64;
    for t in tickets {for x in t["red"].as_array().ok_or("red")?{let k=x.as_u64().ok_or("number")? as usize-1;exposure[k]+=1./30.;union|=1<<k;}}
    let fused=std::array::from_fn(|i|small[i]*0.5+exposure[i]*0.5);
    let orders=[select::rank_desc(&original),select::rank_desc(&small),select::rank_desc(&fused)];
    let hit=|order:&[usize;33],k:usize|order[..k].iter().filter(|&&x|r.red_mask&(1<<x)!=0).count();
    let size=union.count_ones() as usize;
    rows.push(json!({"issue":r.issue,"reference_size":size,"reference_hits":(union&r.red_mask).count_ones(),"same_size_original_hits":hit(&orders[0],size),"hits":orders.map(|o|[12,14,15,16].map(|k|hit(&o,k)))}));
   }
  }
  if r.issue<Protocol::CONFIRM_START_ISSUE{for e in 0..7{sums[e]+=select::rank_log_score(&r.expert_p[e],r.red_mask);}n+=1;}
 }
 let mut windows=serde_json::Map::new();
 for (name,start) in [("all",0),("first200",0),("last100",rows.len().saturating_sub(100)),("recent60",rows.len().saturating_sub(60))]{
  let end=if name=="first200"{200.min(rows.len())}else{rows.len()};let span=&rows[start..end];let mut cells=Vec::new();
  for (m,label) in ["weighted","small_recent","reference_fused"].iter().enumerate(){for (p,k) in [12,14,15,16].iter().enumerate(){let full=span.iter().filter(|r|r["hits"][m][p]==6).count();let gain=span.iter().filter(|r|r["hits"][m][p]==6&&r["hits"][1][p]!=6).count();let loss=span.iter().filter(|r|r["hits"][m][p]!=6&&r["hits"][1][p]==6).count();cells.push(json!({"model":label,"red_count":k,"full_red_periods":full,"paired_gains_vs_small":gain,"paired_losses_vs_small":loss}));}}
  windows.insert(name.into(),json!({"n":span.len(),"reference_full_red":span.iter().filter(|r|r["reference_hits"]==6).count(),"same_size_original_full_red":span.iter().filter(|r|r["same_size_original_hits"]==6).count(),"cells":cells}));
 }
 println!("{}",serde_json::to_string_pretty(&json!({"kind":"imported_reference_exploratory_replay","actual_recommendation":false,"reference_runtime_independently_reproduced":false,"input_data_sha256":draw::data_sha256(&draws),"reference_trace_sha256":imported["source_sha256"],"fusion":"half_small_recent_plus_half_original_five_ticket_red_exposure","parameters_selected_on_this_replay":false,"excluded_not_five_tickets":excluded,"windows":windows,"rows":rows})).unwrap());Ok(())
}
