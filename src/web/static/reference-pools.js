(function (global) {
 'use strict';
 const pad = n => String(n).padStart(2, '0');
 const sorted = ns => (ns || []).slice().sort((a, b) => a - b);
 const percent = n => n == null ? '—' : (Number(n) * 100).toFixed(2) + '%';
 const nullable = n => n == null ? '—' : n;
 const flag = n => n == null ? '—' : n ? '是' : '否';
 const time = n => n ? new Date(n * 1000).toLocaleString('zh-CN', {timeZone:'Asia/Shanghai', hour12:false}) + '（北京时间）' : '未留存';
 const fileUrl = path => SSQApp.url('api/recommendations/files', {path});
 // v2 was first published for 26128. Earlier archives retain their original
 // version; an explicit URL choice always wins and is carried into reviews.
 const preferredFusionVersion=(search,issue)=>{const v=new URLSearchParams(search||'').get('fusion');return v==='v1'||v==='v2'?v:issue&&String(issue)<'26128'?'v1':'v2';};
 async function copyText(text) {
  if (navigator.clipboard) return navigator.clipboard.writeText(text);
  const input = document.createElement('textarea');
  input.value = text; document.body.appendChild(input); input.select();
  const ok = document.execCommand('copy'); input.remove();
  if (!ok) throw Error('复制未成功，请手动选择号码。');
 }
 const shortIssue=value=>{const s=String(value??'');if(/^20\d{5}$/.test(s))return s.slice(2);if(/^\d{5}$/.test(s))return s;throw Error('报告期号不是有效的5位或7位期号');};
 function sourceCandidates(file){if(file?.forecast_archive)return[file.forecast_archive];if(Array.isArray(file?.forecasts)&&file.forecasts.length)return file.forecasts;if(file?.tickets&&file?.source_period)return[file];throw Error('请选择源码的forecast_audit.json、较小的experiment_report.json或单条预测记录JSON');}
 function ticketKeys(rows){if(!Array.isArray(rows)||rows.length!==5)throw Error('所选原报告必须有5注');const keys=rows.map(t=>{const r=t.red;if(!Array.isArray(r)||r.length!==6||new Set(r).size!==6||r.some(n=>!Number.isInteger(n)||n<1||n>33)||!Number.isInteger(t.blue)||t.blue<1||t.blue>16)throw Error('报告票面红蓝球无效');return r.slice().sort((a,b)=>a-b).join(',')+'+'+t.blue;});if(new Set(keys).size!==5)throw Error('报告的5注有重复整注');return keys;}
 function planKey(p){if(!p)return null;const a=p.red_dan,b=p.red_tuo,c=p.blue??p.blues;if(![a,b,c].every(Array.isArray))throw Error('报告胆拖缺少胆、拖或蓝球');for(const[ns,high]of[[a,33],[b,33],[c,16]])if(new Set(ns).size!==ns.length||ns.some(n=>!Number.isInteger(n)||n<1||n>high))throw Error('报告胆拖号码重复或越界');if(a.length>5||b.length<6-a.length||!c.length||a.some(n=>b.includes(n)))throw Error('报告胆拖结构无效');let bets=1;for(let i=0;i<6-a.length;i++)bets=bets*(b.length-i)/(i+1);bets=Math.round(bets)*c.length;const unit=p.unit_price??2;if(unit!==2||(p.bet_count!=null&&p.bet_count!==bets)||((p.amount??p.cost)!=null&&(p.amount??p.cost)!==bets*2))throw Error('报告胆拖注数或金额与展开不一致');return JSON.stringify([a,b,c].map(ns=>ns.slice().sort((a,b)=>a-b)));}
 function compareSource(site,file,index=0){const choices=sourceCandidates(file),f=choices[index];if(!f)throw Error('请选择一条原预测记录');const target=shortIssue(f.target_period),cutoff=shortIssue(f.source_period),own=ticketKeys(site.tickets),other=ticketKeys(f.tickets),plan=f.dantuo_plan??f.metadata?.dantuo_plan??null;
  const differences=[],unknown=[];const add=(label,a,b)=>{if(a==null||b==null){unknown.push(label);return;}if(String(a)!==String(b))differences.push({label,site:String(a),source:String(b)});};
  add('目标期',site.issue,target);add('输入截止',site.data_cutoff,cutoff);add('历史期数',site.history_count,file.data?.periods??f.data_version?.periods);
  add('回测期数',site.profile?.backtest_periods,f.metadata?.forecast_publish_gate?.effective_backtest_periods??file.config_snapshot?.backtest?.periods);
  add('运行档位',site.profile?.runtime_profile,file.config_snapshot?.runtime?.active_profile);
  add('蒙特卡洛次数',site.profile?.monte_carlo_samples,file.config_snapshot?.generation?.monte_carlo_samples);
  add('优化方法',site.profile?.optimizer_methods?.join('、'),file.config_snapshot?.optimizer?.methods?.join('、'));
  add('随机种子',site.profile?.random_seed,file.run_manifest?.seed??file.config_snapshot?.project?.random_seed??file.run_metadata?.random_seed);
  add('清洗后数据摘要',site.clean_frame_sha256,f.data_version?.clean_frame_sha256??file.data?.cleaning?.data_manifest?.clean_frame_sha256);
  const packages=file.run_metadata?.packages??file.run_manifest?.package_versions;if(packages)for(const[name,version]of Object.entries(site.profile?.packages||{}))if(packages[name]!=null)add(name,version,packages[name]);
  const ticketsEqual=JSON.stringify(own.slice().sort())===JSON.stringify(other.slice().sort()),orderEqual=JSON.stringify(own)===JSON.stringify(other),dantuoEqual=site.dantuo&&plan?planKey(site.dantuo)===planKey(plan):null;
  return{status:target!==site.issue?'different_issue':ticketsEqual&&dantuoEqual!==false?'same_numbers':'different_numbers',source_issue:target,source_cutoff:cutoff,forecast_id:f.forecast_id??'',tickets_equal:ticketsEqual,ticket_order_equal:orderEqual,dantuo_equal:dantuoEqual,differences,unknown,record_count:choices.length};
 }
 const referenceProvenance={props:{issue:{type:String,required:true},snapshot:{type:Object,required:true}},setup(props){
  const report=Vue.ref(null),error=Vue.ref(''),comparison=Vue.ref(null),fileError=Vue.ref(''),fileName=Vue.ref(''),selected=Vue.ref(0),choices=Vue.ref([]);let source=null,request=0;
  const sourceUrl=path=>path?new URL(path,SSQApp.base).href:'';
  async function load(){const id=++request;error.value='';try{const r=await fetch(SSQApp.url('api/reference-pools/provenance',{issue:props.issue})),v=await r.json();if(id!==request)return;if(!r.ok)throw Error(v.error||'来源核验失败');report.value=v;recheck();}catch(e){if(id===request){error.value=e.message;report.value=null;}}}
  function recheck(){comparison.value=null;fileError.value='';if(!source||!report.value?.tickets)return;try{comparison.value=compareSource(report.value,source,Number(selected.value));}catch(e){fileError.value=e.message;}}
  async function chooseFile(event){source=null;choices.value=[];comparison.value=null;fileError.value='';const f=event.target.files?.[0];fileName.value=f?.name||'';if(!f)return;if(f.size>16*1024*1024){fileError.value='文件超过16MB，请选择源码logs目录下较小的forecast_audit.json，包含同样的5注和原始胆拖。';return;}try{const parsed=JSON.parse(await f.text());choices.value=sourceCandidates(parsed).map((r,index)=>({index,label:String(r.target_period)+' · '+(r.forecast_scope||'原预测')+' · '+(r.created_at||r.forecast_id||index+1)}));source=parsed;const matches=choices.value.filter((_,i)=>shortIssue(sourceCandidates(parsed)[i].target_period)===props.issue);selected.value=matches.length?matches[matches.length-1].index:choices.value.length-1;recheck();}catch(e){source=null;fileError.value=e.message||'无法读取原报告';}}
  Vue.onMounted(load);Vue.watch(()=>[props.issue,props.snapshot],()=>{report.value=null;comparison.value=null;load();});Vue.watch(selected,recheck);return{report,error,comparison,fileError,fileName,selected,choices,sourceUrl,chooseFile};
 },template:`<section class="reference-provenance" data-reference-provenance aria-label="参考源码运行来源核验">
  <p v-if="error" class="notice error" role="alert">{{error}}</p>
  <template v-if="report?.profile"><p class="muted" data-source-profile>本站运行：目标{{report.issue}}期 · 档位{{report.profile.runtime_profile??'未记录'}} · 回测{{report.profile.backtest_periods??'未知'}}期 · 历史{{report.history_count}}期，截至{{report.data_cutoff}} · 种子{{report.profile.random_seed??'未知'}}。</p>
   <details class="reference-union" data-source-details><summary>核对同次原报告／与本地源码结果比对</summary>
    <p class="muted" data-source-verification>5注与同次原报告{{report.verification.five_matches_same_run_report===true?'一致':'尚不能核验'}}；原始胆拖{{report.verification.dantuo_matches_same_run_report===true?'一致':'尚不能核验'}}。此核验只对应参考引擎原始输出，融合方案按本站规则另行生成。</p>
    <p v-if="report.profile.runtime_profile==='supplied_default'" class="notice" data-source-legacy>本档是旧入口直接运行默认配置的首次结果，未应用源码网页档位。后续新数据计算已改走源码网页默认standard档；本档保留原值。</p>
    <p class="muted">Python {{report.profile.python??'未记录'}} · XGBoost {{report.profile.packages?.xgboost??'未记录'}} · scikit-learn {{report.profile.packages?.['scikit-learn']??'未记录'}}。</p>
    <p v-if="report.date_policy" class="muted">本站保留输入的{{report.date_policy.source_exceptions?.length??0}}条早期例外日期，关闭源码重复的星期预检。直接运行源码时须对齐历史内容、日期策略、回测期数和依赖版本。</p>
    <div class="row reference-actions"><a v-if="report.source_forecast_url" :href="sourceUrl(report.source_forecast_url)" :download="'reference-source-'+issue+'.json'">下载同次原预测JSON</a><a :href="sourceUrl(report.source_input_url)" :download="'Tsys-input-'+issue+'.csv'">下载本次源码格式输入CSV</a></div>
    <label class="field">选择源码原报告在浏览器本地比对<input type="file" accept=".json,application/json" data-source-file @change="chooseFile"></label><p class="muted">文件不会上传。较大的experiment_report.json可改选logs/forecast_audit.json；多条记录可选择目标期与版本。</p>
    <label v-if="choices.length>1" class="field">原报告记录<select v-model="selected" data-source-record><option v-for="c in choices" :key="c.index" :value="c.index">{{c.label}}</option></select></label>
    <p v-if="fileError" class="notice error" role="alert">{{fileError}}</p>
    <div v-if="comparison" data-source-comparison><p class="notice">{{comparison.status==='different_issue'?'目标期不同：所选报告是'+comparison.source_issue+'期，页面是'+issue+'期。':comparison.status==='same_numbers'?'所选报告与页面号码一致。':'同目标期出现号码差异，请结合下列运行条件核对。'}}</p><p class="muted">5注集合{{comparison.tickets_equal?'一致':'不同'}}<template v-if="comparison.tickets_equal&&!comparison.ticket_order_equal">（排列顺序不同）</template>；原始胆拖{{comparison.dantuo_equal==null?'原报告信息不足':comparison.dantuo_equal?'一致':'不同'}}。</p>
     <div v-if="comparison.differences.length" class="scroll"><table><thead><tr><th>不同条件</th><th>本站</th><th>所选源码报告</th></tr></thead><tbody><tr v-for="d in comparison.differences" :key="d.label" data-source-difference><td>{{d.label}}</td><td>{{d.site}}</td><td>{{d.source}}</td></tr></tbody></table></div><p v-else class="muted">可从文件中核验的运行条件未发现差异。</p><p v-if="comparison.unknown.length" class="muted">所选记录未提供完整对照信息：{{comparison.unknown.join('、')}}。</p>
    </div>
   </details>
  </template><p v-else-if="report" class="muted">{{report.reason}}</p>
 </section>`};
 const dantuo = {
  props:{bundle:{type:Object,default:null},issue:{type:String,default:''},archived:{type:Boolean,default:false}},
  setup(props){
   const message=Vue.ref(''),plan=Vue.computed(()=>props.bundle?.plan);
   async function copy(expanded){const p=plan.value;if(!p)return;const label=p.red_dan.length?'参考原始胆拖':'参考原始预算复式';
    const body=expanded?p.tickets.map((t,i)=>(i+1)+'. '+t.red.map(pad).join(' ')+' + '+pad(t.blue)).join('\n'):
     (p.red_dan.length?'红胆：'+p.red_dan.map(pad).join(' ')+'\n红拖：':'复式红球：')+p.red_tuo.map(pad).join(' ')+'\n蓝球：'+p.blues.map(pad).join(' ');
    try{await copyText('双色球 '+props.issue+' 期 · '+label+'\n'+body+'\n'+p.bet_count+'注，共'+p.cost+'元（每注2元）\n'+(props.bundle.status==='frozen'?'来自参考同次原始报告的独立封存档。':'原始报告预览，尚未封存。'));message.value=expanded?'完整展开票面已复制。':'预算方案已复制。';}catch(e){message.value=e.message||'复制未成功。';}
   }
   return{plan,message,copy,pad,nullable,flag,time,fileUrl};
  },
  template:`<section class="reference-dantuo" data-reference-dantuo aria-label="参考原始预算方案">
   <h4>参考同次原始方案 · 胆拖／复式</h4>
   <template v-if="plan"><p class="muted">参考程序同次生成的原始方案，原样保留胆码、拖码和蓝球。原始5注与本方案可选择一种。</p>
    <p class="reference-buyable-budget"><strong>{{plan.bet_count}}注 · {{plan.cost}}元</strong><span>{{plan.red_dan.length?plan.red_dan.length+'胆＋'+plan.red_tuo.length+'拖':plan.red_tuo.length+'红复式'}}＋{{plan.blues.length}}蓝；每注2元。</span></p>
    <div v-if="plan.red_dan.length" class="reference-plan-row"><b>红胆</b><div class="balls"><span v-for="n in plan.red_dan" :key="n" class="ball selected red">{{pad(n)}}</span></div></div>
    <div class="reference-plan-row"><b>{{plan.red_dan.length?'红拖':'复式红球'}}</b><div class="balls"><span v-for="n in plan.red_tuo" :key="n" class="ball selected red">{{pad(n)}}</span></div></div>
    <div class="reference-plan-row"><b>蓝球</b><div class="balls"><span v-for="n in plan.blues" :key="n" class="ball selected blue">{{pad(n)}}</span></div></div>
    <p class="muted">展开 C({{plan.red_tuo.length}},{{6-plan.red_dan.length}}) × {{plan.blues.length}} ＝ {{plan.bet_count}}注。</p>
    <p v-if="bundle.status==='frozen'" class="muted">预算方案独立保存于{{time(bundle.published_at_unix)}}。</p><p v-else class="muted">原始报告预览，预算方案尚未独立封存。</p>
    <p v-if="plan.summary?.max_ticket_red_hits!=null" class="muted">红胆命中{{plan.dan_hits}}/{{plan.red_dan.length}} · 红球并集中{{plan.red_union_hits}}/6 · 最佳单注中{{plan.summary.max_ticket_red_hits}}红 · 常规中奖{{plan.summary.prize_tickets}}注 · 已知固定奖金{{plan.summary.fixed_prize}}元。浮动奖金与特别奖另核。</p>
    <div class="row reference-actions"><button type="button" @click="copy(false)">复制胆拖／复式方案</button><button type="button" @click="copy(true)">复制展开{{plan.bet_count}}注</button></div>
    <details class="reference-union"><summary>查看展开{{plan.bet_count}}注完整票面与逐注对奖</summary><ol class="reference-tickets" aria-label="参考预算方案展开票面"><li v-for="(ticket,index) in plan.tickets" :key="index" data-dantuo-ticket><span class="reference-ticket-label">第{{index+1}}注</span><div class="balls"><span v-for="n in ticket.red" :key="n" class="ball selected red">{{pad(n)}}</span><span class="reference-plus">+</span><span class="ball selected blue">{{pad(ticket.blue)}}</span></div><small class="muted">中红{{nullable(plan.ticket_results?.[index]?.red_hits)}}/6 · 中蓝{{flag(plan.ticket_results?.[index]?.blue_hit)}}<template v-if="plan.ticket_results?.[index]?.red_hits!=null"> · {{plan.ticket_results[index].prize_level?plan.ticket_results[index].prize_level+'等奖':plan.ticket_results[index].special_prize_candidate?'福运奖待核实':'未中常规奖'}}</template></small></li></ol></details>
    <div class="row reference-actions"><a v-if="bundle.raw_file" :href="fileUrl(bundle.raw_file)">参考胆拖原档JSON</a><a v-if="bundle.review_file" :href="fileUrl(bundle.review_file)">参考胆拖逐注复盘JSON</a></div>
   </template><p v-else class="muted">{{archived?'旧档未保存参考胆拖，不补造历史号码。':bundle?.reason||'等待参考程序导出原始胆拖方案。'}}</p>
   <p v-if="message" class="muted" role="status">{{message}}</p>
  </section>`
 };
 const dantuoAudit={props:{issue:{type:String,required:true}},setup(props){
  const report=Vue.ref(null),error=Vue.ref(''),loading=Vue.ref(false),opened=Vue.ref(false);let request=0;
  const score=n=>n==null?'—':Number(n).toFixed(4),gain=n=>n==null?'—':(Number(n)*100).toFixed(2)+'%';
  async function load(){if(!opened.value||report.value||loading.value)return;const id=++request;loading.value=true;error.value='';try{const r=await fetch(SSQApp.url('api/reference-pools/dantuo-audit',{issue:props.issue})),v=await r.json();if(id!==request)return;if(!r.ok)throw Error(v.error||'胆拖优化核验失败');report.value=v;}catch(e){if(id===request)error.value=e.message;}finally{if(id===request)loading.value=false;}}
  function toggle(e){opened.value=e.target.open;load();}Vue.watch(()=>props.issue,()=>{request++;report.value=null;error.value='';loading.value=false;load();});return{report,error,loading,toggle,load,score,gain};
 },template:`<details class="reference-evidence dantuo-audit" data-dantuo-audit @toggle="toggle"><summary>胆拖深度核验 · 选胆与完整票面优化</summary>
  <p v-if="loading" class="muted" role="status">正在比较同一16红内的全部合法胆拖结构…</p><p v-if="error" class="notice error" role="alert">{{error}} <button type="button" @click="load">重试</button></p>
  <template v-if="report?.rows"><p class="muted">第{{report.issue}}期 · 同一融合16红、相同蓝球选择、每种10注20元。按完整展开票面的模型概率之和搜索，不以并集命中或单号曝光代替完整单注。</p><p class="muted" data-dantuo-audit-origin>{{report.prefix_plan_sealed?'前缀方案已有本期首次原档；这里仅作只读模型评分。':'本期缺少已封存的胆拖原档，前缀规则仅为历史算法回放，不算实际推荐。'}}</p>
   <p class="notice" data-dantuo-audit-adoption>{{report.evidence.reason}}</p>
   <div class="scroll"><table><thead><tr><th>规格</th><th>前缀规则评分</th><th>同池最优评分</th><th>模型分数变化</th><th>合法红球结构</th></tr></thead><tbody><tr v-for="r in report.rows" :key="r.id" :data-dantuo-audit-spec="r.id"><td>{{r.id}}</td><td>{{score(r.original_prefix.ratio_to_uniform_ten_tickets)}}</td><td>{{score(r.exact_same_pool.ratio_to_uniform_ten_tickets)}}</td><td>{{gain(r.model_mass_relative_gain)}}</td><td>{{r.certificate.all_legal_red_plans.toLocaleString('en-US')}}</td></tr></tbody></table></div>
   <p class="muted">评分使用已冻结的完整组合校准模型。表中以均匀10注为1；分数上涨不代表实际中奖率上涨。模型在{{report.evidence.confirmation_exploratory.n}}期确认窗的6＋1平均对数评分差为{{report.evidence.confirmation_exploratory.joint_log_excess_vs_uniform==null?'—':Number(report.evidence.confirmation_exploratory.joint_log_excess_vs_uniform).toFixed(6)}}，0为均匀基准。</p>
   <p class="muted">2+5+2：7红并集有7组六红，全部胆码入票后保留5组，再配2蓝。3+5+1：8红并集有28组六红，保留10组，再配1蓝。胆码错一个，两种均无法形成完整6红；两种各10个不同6＋1结果，公平头奖基准相同。</p>
   <p class="muted">搜索比较所有合法候选结构，证明只限给定模型、同16红与固定蓝球。16红已漏开奖号时，组票优化无法补回；历史探索和当前模型评分均不计作首次留档成绩。</p>
  </template><p v-else-if="report" class="muted">{{report.reason}}</p>
 </details>`};
 const fusionDantuo={components:{DantuoAudit:dantuoAudit},props:{issue:{type:String,required:true},snapshot:{type:Object,required:true}},setup(props){
  const report=Vue.ref(null),error=Vue.ref(''),message=Vue.ref(''),busy=Vue.ref(false),selected=Vue.ref('2+5+2');let request=0;
  const current=Vue.computed(()=>report.value?.current),plan=Vue.computed(()=>current.value?.plans?.find(p=>p.id===selected.value));
  const api='api/reference-pools/fusion-dantuo';
  async function load(){const id=++request;error.value='';try{const r=await fetch(SSQApp.url(api,{issue:props.issue})),v=await r.json();if(id!==request)return;if(!r.ok)throw Error(v.error||'融合胆拖读取失败');report.value=v;}catch(e){if(id===request){error.value=e.message;report.value=null;}}}
  async function freeze(){if(busy.value||!report.value)return;busy.value=true;message.value='';try{const r=await fetch(SSQApp.url(api+'/freeze'),{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({target:report.value.target,data_sha256:report.value.data_sha256})}),v=await r.json();if(!r.ok)throw Error(v.error||'融合胆拖封存失败');message.value=v.status==='waiting'?v.reason:v.status==='already_frozen'?'已保留两种融合胆拖的首次原档。':'两种融合胆拖已封存，开奖后按展开票面逐注复盘。';await load();}catch(e){message.value=e.message;}finally{busy.value=false;}}
  async function copy(expanded){const p=plan.value;if(!p)return;const body=expanded?p.tickets.map((t,i)=>(i+1)+'. '+t.red.map(pad).join(' ')+' + '+pad(t.blue)).join('\n'):'红胆：'+p.red_dan.map(pad).join(' ')+'\n红拖：'+p.red_tuo.map(pad).join(' ')+'\n蓝球：'+p.blues.map(pad).join(' ');try{await copyText('双色球 '+props.issue+'期 · '+p.title+'\n'+body+'\n10注，共20元（每注2元）\n'+(current.value.status==='frozen'?'来自首次封存的融合胆拖原档。':'当前预览，尚未封存。'));message.value=expanded?'展开10注打单文本已复制。':'融合胆拖方案已复制。';}catch(e){message.value=e.message||'复制失败';}}
  const rawUrl=Vue.computed(()=>current.value?.raw_url?new URL(current.value.raw_url,SSQApp.base).href:'');
  Vue.onMounted(load);Vue.watch(()=>[props.issue,props.snapshot],()=>{message.value='';report.value=null;load();});return{report,error,message,busy,selected,current,plan,load,freeze,copy,rawUrl,pad,nullable,flag,time};
 },template:`<section class="reference-dantuo fusion-dantuo" data-fusion-dantuo aria-label="融合胆拖推荐">
  <div class="row between"><h4>融合胆拖 · 每种10注20元</h4><button type="button" @click="load">刷新胆拖</button></div>
  <p v-if="error" class="notice error" role="alert">{{error}}</p><p v-if="!report&&!error" class="muted" role="status">正在读取融合胆拖…</p>
  <template v-if="report"><p v-if="!plan" class="muted" data-fusion-dantuo-unavailable>{{current?.reason||'本期暂无融合胆拖。'}}</p>
   <template v-if="plan"><p class="muted">按融合原档排序选胆、再取5个拖码，蓝球取原档前2或1个。两种独立方案，可选择其中一种。</p>
    <div class="small-red-sizes" aria-label="融合胆拖规格"><button v-for="p in current.plans" :key="p.id" type="button" :data-fusion-dantuo-spec="p.id" :aria-pressed="selected===p.id" :class="{selected:selected===p.id}" @click="selected=p.id">{{p.id}}</button></div>
    <p class="reference-buyable-budget" data-fusion-dantuo-budget><strong>{{plan.red_dan.length}}胆＋5拖＋{{plan.blues.length}}蓝 · {{plan.bet_count}}注 · {{plan.cost}}元</strong></p>
    <div class="reference-plan-row" data-fusion-dan><b>红胆</b><div class="balls"><span v-for="n in plan.red_dan" :key="n" class="ball selected red">{{pad(n)}}</span></div></div>
    <div class="reference-plan-row" data-fusion-tuo><b>红拖</b><div class="balls"><span v-for="n in plan.red_tuo" :key="n" class="ball selected red">{{pad(n)}}</span></div></div>
    <div class="reference-plan-row" data-fusion-blue><b>蓝球</b><div class="balls"><span v-for="n in plan.blues" :key="n" class="ball selected blue">{{pad(n)}}</span></div></div>
    <p class="muted">C(5,{{6-plan.red_dan.length}}) × {{plan.blues.length}} ＝ 10注，每注2元。完整六红要求胆码全部命中。</p>
    <p v-if="current.status==='frozen'" class="muted" data-fusion-dantuo-sealed>首次原档已封存：{{time(current.published_at_unix)}}。<a :href="rawUrl">下载两种方案原档</a></p><p v-else class="muted">当前为预览，未封存；仅截止前首次封存的完整票面计入实际复盘。</p>
    <p v-if="plan.summary?.max_ticket_red_hits!=null" class="muted" data-fusion-dantuo-result>红胆中{{plan.dan_hits}}/{{plan.red_dan.length}} · 并集中{{plan.red_union_hits}}/6 · 最佳单注中{{plan.summary.max_ticket_red_hits}}红 · 常规中奖{{plan.summary.prize_tickets}}注 · 已知固定奖金{{plan.summary.fixed_prize}}元。</p>
    <div class="row reference-actions"><button type="button" @click="copy(false)">复制融合胆拖方案</button><button type="button" @click="copy(true)">复制展开10注</button><button v-if="current.status!=='frozen'" type="button" :disabled="busy||!report.publication.can_freeze" @click="freeze">{{busy?'封存中…':'封存两种融合胆拖'}}</button></div>
    <details class="reference-union" data-fusion-dantuo-expanded><summary>查看{{selected}}展开10注与逐注对奖</summary><ol class="reference-tickets"><li v-for="(ticket,index) in plan.tickets" :key="index" data-fusion-dantuo-ticket><span class="reference-ticket-label">第{{index+1}}注</span><div class="balls"><span v-for="n in ticket.red" :key="n" class="ball selected red">{{pad(n)}}</span><span class="reference-plus">+</span><span class="ball selected blue">{{pad(ticket.blue)}}</span></div><small class="muted">中红{{nullable(plan.ticket_results?.[index]?.red_hits)}}/6 · 中蓝{{flag(plan.ticket_results?.[index]?.blue_hit)}}<template v-if="plan.ticket_results?.[index]?.red_hits!=null"> · {{plan.ticket_results[index].prize_level?plan.ticket_results[index].prize_level+'等奖':plan.ticket_results[index].special_prize_candidate?'福运奖待核实':'未中常规奖'}}</template></small></li></ol></details>
   </template><details v-if="report.actual" class="reference-evidence"><summary>融合胆拖 · 实际封存逐注统计</summary><p class="muted">{{report.actual.note}}</p><p v-if="report.actual.errors.length" class="notice error" role="alert">{{report.actual.errors.join('；')}}</p><div class="scroll"><table><thead><tr><th>规格</th><th>已开奖 / 待开奖</th><th>至少一注常规奖</th><th>平均最佳中红</th></tr></thead><tbody><tr v-for="r in report.actual.cells" :key="r.id"><td>{{r.id}} · 10注20元</td><td>{{r.n}} / {{r.pending}}</td><td>{{r.n?r.any_prize_periods+'/'+r.n:'—'}}</td><td>{{r.mean_best_red==null?'—':Number(r.mean_best_red).toFixed(2)}}</td></tr></tbody></table></div></details>
   <dantuo-audit :issue="issue"></dantuo-audit>
  </template><p v-if="message" class="muted" role="status">{{message}}</p>
 </section>`};
 const fusionReview={props:{issue:{type:String,required:true}},setup(props){
  const report=Vue.ref(null),error=Vue.ref(''),loading=Vue.ref(false),opened=Vue.ref(false);let request=0;
  const ticketReport=Vue.ref(null),ticketError=Vue.ref(''),ticketLoading=Vue.ref(false),ticketOpened=Vue.ref(false);let ticketRequest=0;
  const redReport=Vue.ref(null),redError=Vue.ref(''),redLoading=Vue.ref(false),redOpened=Vue.ref(false);let redRequest=0;
  const ticketLabels={sealed_original:'首次封存8注',red_grouping_same_blue_counts:'仅优化分组，保留每蓝注数',blue_only_same_red_sets:'仅换蓝，保留8组六红',joint_top_eight:'完整6＋1联合最优'};
  const expertLabels={frequency:'历史频率',ewma:'近期加权',overdue:'遗漏',hot:'近期热度',anti_repeat:'蓝球反重号',markov:'蓝球转移'};
  const redLabels={frequency:'频率',ewma:'近期加权',log_gap:'间隔方向',cooccur:'上期共现',zone:'区间',repeat:'上期重号'};
  const latestBlue=Vue.computed(()=>ticketReport.value?.cases.filter(c=>!c.pending&&c.blue_attribution).at(-1)?.blue_attribution);
  const url=path=>path?new URL(path,SSQApp.base).href:'',pct=n=>n==null?'—':(n*100).toFixed(4)+'%',gap=n=>n==null?'—':Number(n).toExponential(2);
  async function load(){if(!opened.value||loading.value||report.value)return;const id=++request;loading.value=true;error.value='';try{const r=await fetch(SSQApp.url('api/reference-pools/fusion-review')),v=await r.json();if(id!==request)return;if(!r.ok)throw Error(v.error||'融合深度复盘读取失败');if(v.target_issue!==props.issue)throw Error('开奖数据已更新，请刷新当前融合推荐');report.value=v;}catch(e){if(id===request)error.value=e.message;}finally{if(id===request)loading.value=false;}}
  async function loadTickets(){if(!ticketOpened.value||ticketReport.value||ticketLoading.value)return;const id=++ticketRequest;ticketLoading.value=true;ticketError.value='';try{const r=await fetch(SSQApp.url('api/reference-pools/ticket-audit')),v=await r.json();if(id!==ticketRequest)return;if(!r.ok)throw Error(v.error||'完整票面核验失败');if(v.target_issue!==props.issue)throw Error('开奖数据已更新，请刷新当前融合推荐');ticketReport.value=v;}catch(e){if(id===ticketRequest)ticketError.value=e.message;}finally{if(id===ticketRequest)ticketLoading.value=false;}}
  function toggleTickets(e){ticketOpened.value=e.target.open;loadTickets();}
  async function loadRed(){if(!redOpened.value||redReport.value||redLoading.value)return;const id=++redRequest;redLoading.value=true;redError.value='';try{const r=await fetch(SSQApp.url('api/reference-pools/red-score-v2')),v=await r.json();if(id!==redRequest)return;if(!r.ok)throw Error(v.error||'红球去重评分读取失败');if(v.target_issue!==props.issue)throw Error('开奖数据已更新，请刷新当前融合推荐');redReport.value=v;}catch(e){if(id===redRequest)redError.value=e.message;}finally{if(id===redRequest)redLoading.value=false;}}
  function toggleRed(e){redOpened.value=e.target.open;loadRed();}
  const redWindow=Vue.computed(()=>redReport.value?.windows?.confirmation_exploratory_25111_26115);
  function toggle(e){opened.value=e.target.open;load();}Vue.watch(()=>props.issue,()=>{request++;report.value=null;error.value='';loading.value=false;ticketRequest++;ticketReport.value=null;ticketError.value='';ticketLoading.value=false;redRequest++;redReport.value=null;redError.value='';redLoading.value=false;load();loadTickets();loadRed();});return{report,error,loading,toggle,load,url,pct,gap,pad,ticketReport,ticketError,ticketLoading,toggleTickets,loadTickets,ticketLabels,expertLabels,latestBlue,redReport,redError,redLoading,toggleRed,redLabels,redWindow};
 },template:`<details class="reference-evidence fusion-review" data-fusion-review @toggle="toggle"><summary>最新v2复盘 · 完整六红入池优化</summary>
  <p v-if="loading" class="muted" role="status">正在核对首次票面并计算完整六红入池目标…</p><p v-if="error" class="notice error" role="alert">{{error}} <button type="button" @click="load">重试</button></p>
  <template v-if="report"><template v-if="report.latest"><p class="muted" data-fusion-review-draw>第{{report.latest.issue}}期：{{report.latest.draw.red.map(pad).join(' ')}} ＋ {{pad(report.latest.draw.blue)}}。16红候选漏{{report.latest.global_sixteen_missed.map(pad).join('、')||'无'}}；蓝球原榜第{{report.latest.winning_blue_rank??'—'}}位。</p>
   <p class="notice" data-fusion-review-loss>红池缺号时，重新组票无法补齐完整六红。错胆与蓝球截选另行核对；小奖与头奖分开看。</p>
   <div class="row reference-actions"><a :href="url(report.latest.raw_url)">本期v2首次原档</a><a v-if="report.latest.review_url" :href="url(report.latest.review_url)">本期v2逐注复盘</a></div>
   <details data-fusion-review-plans><summary>四档单式与两种胆拖 · 分层损失</summary><p class="muted">表格可横向滚动；全部使用首次保存的号码。</p><div class="scroll"><table><thead><tr><th>方案</th><th>候选中红</th><th>最佳单注红</th><th>错胆</th><th>蓝球入票</th><th>头奖</th></tr></thead><tbody><tr v-for="p in report.latest.plans" :key="p.id" :data-fusion-review-plan="p.id"><td>{{p.id}}</td><td>{{p.red_pool_hits}}/6</td><td>{{p.summary.max_ticket_red_hits}}/6</td><td>{{p.wrong_bankers.map(pad).join('、')||'无'}}</td><td>{{p.blue_selected?'是':'否'}}</td><td>{{p.jackpot?'中':'未中'}}</td></tr></tbody></table></div></details>
   <details data-fusion-review-numbers><summary>6个开奖号 · 评分和参考支持的影响</summary><p class="muted">基础名次由首次分数按原公式分解，不是补造另一套历史推荐。</p><div class="scroll"><table><thead><tr><th>开奖号</th><th>无参考支持名次</th><th>v2最终名次</th><th>参考并集支持</th><th>入16红</th></tr></thead><tbody><tr v-for="r in report.latest.winning_numbers" :key="r.number" :data-fusion-review-number="r.number"><td>{{pad(r.number)}}</td><td>{{r.without_reference_rank}}</td><td>{{r.v2_rank}}</td><td>{{r.reference_supported?'有':'无'}}</td><td>{{r.in_sixteen?'是':'否'}}</td></tr></tbody></table></div></details>
  </template><p v-else class="muted">最新开奖缺少v2首次原档，不补造历史推荐。</p>
  <details data-fusion-review-search><summary>16红筛选优化 · 完整六红覆盖目标</summary><p class="muted">研究目标是全部六红同时入池，使用已冻结的完整分布模型；单号分数之和侧重平均中红，两者不等价。下表都在同一校准模型下评分。</p>
   <p class="notice" data-fusion-review-adoption>{{report.adoption.reason}}v2已开奖真实样本{{report.adoption.v2_actual_drawn_periods}}期。</p>
   <p class="muted">第{{report.current_search.issue}}期模型内全6入池质量：已封存v2池{{pct(report.current_search.sealed_v2_pool?.model_full_six_pool_mass)}}；校准单号边际池{{pct(report.current_search.marginal_pool.model_full_six_pool_mass)}}；完整入池搜索{{pct(report.current_search.model_full_six_pool_mass)}}。这些数值尚未验证为真实预测概率。</p>
   <p class="muted" data-fusion-review-certificate>{{report.current_search.certificate.numerically_optimal?'已达到数值容差内最优。':'搜索尚未证明最优。'}}节点{{report.current_search.certificate.nodes}}/{{report.current_search.certificate.node_budget}}；剩余相对差距上界{{gap(report.current_search.certificate.relative_gap)}}。证明限于此模型的16红候选目标。</p>
   <details data-fusion-review-research-pool><summary>查看研究候选及同分限制</summary><p class="reference-number-text">{{report.current_search.reds.map(pad).join(' ')}}</p><p class="muted">没有生成新购票票面或替换已封存v2。相同专家分布组内的成员选择不代表对某个号码更有把握；代表数字顺序不计作命中提升。</p></details>
  </details>
  <details data-ticket-audit @toggle="toggleTickets"><summary>同预算8注 · 完整6＋1组票核验</summary>
   <p v-if="ticketLoading" class="muted" role="status">正在枚举同池全部六红组合及16蓝…</p><p v-if="ticketError" class="notice error" role="alert">{{ticketError}} <button type="button" @click="loadTickets">重试</button></p>
   <template v-if="ticketReport"><p class="notice" data-ticket-adoption>{{ticketReport.reason}}</p><p class="muted">每行独立比较8注16元，使用相同原16红候选。优化后实际用红、用蓝数量见表；不要求候选16红全部入票。评分以均匀8个不同6＋1结果为1。全部研究对照都未发布新票面。</p>
    <section v-for="c in ticketReport.cases" :key="c.issue" :data-ticket-case="c.issue"><h4>第{{c.issue}}期 · {{c.pending?'待开奖':'首次票面与研究对照'}}</h4>
     <p v-if="!c.pending" class="muted">原16红漏{{c.red_pool_missed.map(pad).join('、')||'无'}}；蓝{{pad(c.draw.blue)}}原榜第{{c.blue_attribution.original_rank}}位，{{c.blue_attribution.selected?'已入票':'未入票'}}。同池重组仍受原候选限制。</p>
     <div class="scroll"><table><thead><tr><th>同预算方案</th><th>模型评分</th><th>相对原票分数</th><th>用红／用蓝</th><th>完整六红组数</th><th>单注6＋1</th></tr></thead><tbody><tr v-for="r in c.comparison.rows" :key="r.id" :data-ticket-method="r.id"><td>{{ticketLabels[r.id]}}</td><td>{{r.ratio_to_uniform_eight.toFixed(6)}}</td><td>{{pct(r.model_gain_vs_original)}}</td><td>{{r.red_union_count}}／{{r.blue_count}}</td><td>{{r.distinct_red_sets}}</td><td>{{c.pending?'待开奖':r.complete_6_plus_1_hit?(r.actual_recommendation?'已中':'研究覆盖'):'未覆盖'}}</td></tr></tbody></table></div>
     <p class="muted" data-ticket-compression>原16红×原蓝球共有{{c.comparison.original_blue_cartesian_outcomes.toLocaleString('en-US')}}个不同6＋1结果，原票仅覆盖8个；在这些结果内保留的模型质量为{{pct(c.comparison.original_ticket_retention_given_pool_and_original_blues)}}。入池和单注覆盖须分别核对。</p>
     <p class="muted" data-ticket-certificate>精确枚举{{c.comparison.certificate.complete_outcomes_evaluated.toLocaleString('en-US')}}个完整结果；联合最优搜索差距为0。该结论限于给定模型和原候选池。</p><a :href="url(c.raw_url)">核对本期首次原档</a>
    </section>
    <details v-if="latestBlue" data-ticket-blue-attribution><summary>最新开奖蓝球 · 六个评分分量</summary><p class="muted">蓝{{pad(latestBlue.winning_blue)}}原榜第{{latestBlue.original_rank}}位；按首次输入前缀核对原公式。去掉单个分量的名次仅作归因，不是另一份历史推荐。</p><div class="scroll"><table><thead><tr><th>分量</th><th>原权重</th><th>该专家内名次</th><th>去掉该分量后名次</th></tr></thead><tbody><tr v-for="e in latestBlue.experts" :key="e.expert" :data-ticket-blue-expert="e.expert"><td>{{expertLabels[e.expert]}}</td><td>{{pct(e.weight)}}</td><td>{{e.winning_blue_expert_rank}}</td><td>{{e.winning_blue_rank_without_this_expert}}</td></tr></tbody></table></div></details>
    <p class="muted">完整概率模型6＋1确认窗相对均匀的平均对数评分差为{{ticketReport.confirmation_joint_log_excess==null?'—':ticketReport.confirmation_joint_log_excess.toFixed(6)}}；模型内分数提高尚未证明真实头奖优势。当前已封存推荐保留。</p><a :href="url('api/reference-pools/ticket-audit')">核对完整票面搜索与蓝球归因JSON</a>
   </template>
  </details>
  <details data-red-score-v2 @toggle="toggleRed"><summary>红球去重评分 · 完整六红似然与蓝球分配</summary>
   <p v-if="redLoading" class="muted" role="status">正在拟合去重后的完整六红似然…</p><p v-if="redError" class="notice error" role="alert">{{redError}} <button type="button" @click="loadRed">重试</button></p>
   <template v-if="redReport"><p class="notice" data-red-score-adoption>{{redReport.adoption.reason}}</p>
    <p class="muted">遗漏和热度在对数上方向相反，只保留一个间隔系数；近期和重号各计一次。评分对象是完整六红，不是六个单号分数相加。下表前16按单号入选概率排列，只对照16红截线，不发布新票。</p>
    <p v-if="redReport.latest" class="muted" data-red-score-latest>第{{redReport.latest.issue}}期模型前16覆盖{{redReport.latest.reds_inside_model_top16}}/6个开奖红，完整六红{{redReport.latest.full_six_inside_model_top16?'在前16内':'仍不在前16'}}。相对均匀的完整六红对数差{{Number(redReport.latest.log_excess_vs_uniform).toFixed(6)}}。</p>
    <div v-if="redReport.latest" class="scroll"><table><thead><tr><th>开奖红</th><th>去重模型名次</th><th>入模型16</th><th>相对第16名入选概率</th><th>融合v2名次</th><th>入融合16</th><th>上期红</th></tr></thead><tbody><tr v-for="r in redReport.latest.numbers" :key="r.number" :data-red-score-number="r.number"><td>{{pad(r.number)}}</td><td>{{r.model_rank}}</td><td>{{r.in_model_top16?'是':'否'}}</td><td>{{r.model_inclusion_ratio_to_rank16==null?'—':Number(r.model_inclusion_ratio_to_rank16).toFixed(3)}}</td><td>{{r.fusion_v2_rank??'—'}}</td><td>{{r.in_fusion_v2_top16==null?'—':r.in_fusion_v2_top16?'是':'否'}}</td><td>{{r.previous_red?'是':'否'}}</td></tr></tbody></table></div>
    <p v-if="redWindow" class="muted" data-red-score-window>确认窗探索{{redWindow.n}}期：平均对数差{{redWindow.mean_log_excess_vs_uniform==null?'—':Number(redWindow.mean_log_excess_vs_uniform).toFixed(6)}}，前16平均中红{{Number(redWindow.mean_reds_inside_top16).toFixed(3)}}，均匀期望{{redWindow.uniform_expected_reds_inside_top16.toFixed(3)}}，前16包住全部六红{{redWindow.full_six_inside_top16}}期。历史窗口已被查看。</p>
    <div class="scroll"><table><thead><tr><th>方向</th><th>拟合系数</th></tr></thead><tbody><tr v-for="(a,i) in redReport.policy.axes" :key="a" :data-red-score-axis="a"><td>{{redLabels[a]}}</td><td>{{Number(redReport.training.coefficients[i]).toFixed(6)}}</td></tr></tbody></table></div>
    <p class="muted">系数可以为负。训练只到23100，共{{redReport.training.n}}条；求解剩余目标差距上界{{Number(redReport.training.objective_gap_upper_bound).toExponential(2)}}。第{{redReport.next.issue}}期研究前16不进入购票、手动交集或已封存v2。</p>
    <template v-if="redReport.blue_decision"><p class="notice" data-blue-flatness>已封存蓝球v2把本期蓝{{pad(redReport.blue_decision.winning_blue)}}排在第{{redReport.blue_decision.v2_rank}}，概率{{pct(redReport.blue_decision.v2_probability)}}，均匀为{{pct(redReport.blue_decision.uniform_probability)}}。前2个蓝的概率和{{pct(redReport.blue_decision.top2_probability_mass)}}，均匀前2为{{pct(redReport.blue_decision.uniform_top2_probability_mass)}}。</p>
     <p class="muted">{{redReport.blue_decision.objective_note}}</p>
     <div class="scroll"><table><thead><tr><th>8注蓝球分配</th><th>单注概率之和</th><th>开奖蓝被用到的概率</th><th>本期开奖蓝</th></tr></thead><tbody><tr v-for="a in [redReport.blue_decision.assignments.sealed_v2_16_red_8_tickets,redReport.blue_decision.assignments.all_eight_tickets_on_highest_probability,redReport.blue_decision.assignments.eight_distinct_highest].filter(a=>a&&a.id)" :key="a.id" :data-blue-assignment="a.id"><td>{{a.id==='sealed_v2_two_blues_four_each'?'已封存：两个蓝各4注':a.id==='all_eight_on_highest'?'8注都用概率最高蓝':'概率最高的8个不同蓝'}}</td><td>{{Number(a.sum_of_ticket_probabilities).toFixed(6)}}</td><td>{{pct(a.probability_winning_blue_is_used)}}</td><td>{{a.includes_winning_blue?'在所用蓝球中':'不在'}}</td></tr></tbody></table></div>
    </template><p v-else class="muted">本期没有已核验的蓝球v2首次分布，不补造蓝球分配对照。</p>
    <a :href="url('api/reference-pools/red-score-v2')">核对红球去重评分与蓝球分配JSON</a>
   </template>
  </details><a :href="url('api/reference-pools/fusion-review')">核对完整复盘与搜索证据JSON</a>
 </template>
 </details>`};
 const fusionV2={components:{FusionReview:fusionReview},props:{issue:{type:String,required:true},snapshot:{type:Object,required:true}},setup(props){
  const report=Vue.ref(null),error=Vue.ref(''),message=Vue.ref(''),loading=Vue.ref(false),busy=Vue.ref(false),mode=Vue.ref('single'),size=Vue.ref(16),shape=Vue.ref('2+5+2');let request=0;
  const current=Vue.computed(()=>report.value?.current),bundle=Vue.computed(()=>current.value?.bundle),plan=Vue.computed(()=>mode.value==='single'?bundle.value?.singles?.find(p=>p.red_count===size.value):bundle.value?.dantuo?.find(p=>p.id===shape.value));
  const overlap=Vue.computed(()=>{const previous=new Set(bundle.value?.previous_reds||[]),p=plan.value;if(!p||previous.size!==6)return null;const counts=p.tickets.map(t=>t.red.filter(n=>previous.has(n)).length);return{minimum:Math.min(...counts),maximum:Math.max(...counts),entire:counts.filter(n=>n===6).length,previousBankers:p.red_dan.filter(n=>previous.has(n))};});
  const rawUrl=Vue.computed(()=>current.value?.raw_url?new URL(current.value.raw_url,SSQApp.base).href:'');const reviewUrl=Vue.computed(()=>current.value?.review_url?new URL(current.value.review_url,SSQApp.base).href:'');
  async function load(){const id=++request;loading.value=true;error.value='';try{const r=await fetch(SSQApp.url('api/reference-pools/fusion-v2',{issue:props.issue})),v=await r.json();if(id!==request)return;if(!r.ok)throw Error(v.error||'融合修正版读取失败');if(v.issue!==props.issue)throw Error('融合期号与当前页面不一致，请刷新原档');report.value=v;}catch(e){if(id===request){error.value=e.message;report.value=null;}}finally{if(id===request)loading.value=false;}}
  async function freeze(){if(busy.value||!report.value)return;busy.value=true;message.value='';try{const r=await fetch(SSQApp.url('api/reference-pools/fusion-v2/freeze'),{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({target:report.value.target,data_sha256:report.value.data_sha256})}),v=await r.json();if(!r.ok)throw Error(v.error||'融合修正版封存失败');message.value=v.status==='waiting'?v.reason:v.status==='already_frozen'?'已保留v2首次票面。':'v2单式与两种胆拖已独立封存。';await load();}catch(e){message.value=e.message;}finally{busy.value=false;}}
  async function copy(compact){const p=plan.value;if(!p)return;let body=p.tickets.map((t,i)=>(i+1)+'. '+t.red.map(pad).join(' ')+' + '+pad(t.blue)).join('\n');if(compact&&p.mode==='dantuo')body='红胆：'+p.red_dan.map(pad).join(' ')+'\n红拖：'+p.red_tuo.map(pad).join(' ')+'\n蓝球：'+p.blues.map(pad).join(' ');try{await copyText('双色球 '+props.issue+'期 · '+p.title+'\n'+body+'\n'+p.bet_count+'注，共'+p.cost+'元（每注2元）\n'+(current.value.status==='frozen'?'来自修正版v2首次封存原档。':'修正版v2预览，尚未封存。'));message.value='修正版v2号码已复制。';}catch(e){message.value=e.message||'复制失败';}}
  Vue.onMounted(load);Vue.watch(()=>[props.issue,props.snapshot],()=>{message.value='';report.value=null;load();});return{report,error,message,loading,busy,mode,size,shape,current,bundle,plan,overlap,rawUrl,reviewUrl,load,freeze,copy,pad,time,nullable,flag};
 },template:`<section class="fusion-v2" data-fusion-v2 aria-label="融合修正版v2">
  <p class="notice" data-fusion-v2-repair>修正版v2：已去除两层显式重号加分。保留参考支持、正常重号候选和原蓝球排序；旧版v1首次档可切换查看。</p>
  <p v-if="loading&&!report" class="muted" role="status">正在读取融合修正版…</p><p v-if="error" class="notice error" role="alert">{{error}}</p>
  <template v-if="report"><p v-if="!bundle" class="muted" data-fusion-v2-missing>{{current.reason}}</p>
   <template v-if="bundle"><p class="muted">第{{issue}}期 · 单式四档与两种胆拖各为独立选项，按所选方案显示费用。</p>
    <div class="small-red-sizes" aria-label="修正版买法"><button type="button" data-fusion-v2-mode="single" :aria-pressed="mode==='single'" :class="{selected:mode==='single'}" @click="mode='single'">8注单式 · 16元</button><button type="button" data-fusion-v2-mode="dantuo" :aria-pressed="mode==='dantuo'" :class="{selected:mode==='dantuo'}" @click="mode='dantuo'">胆拖 · 20元</button></div>
    <div v-if="mode==='single'" class="small-red-sizes" aria-label="修正版候选红球数量"><button v-for="p in bundle.singles" :key="p.red_count" type="button" :data-fusion-v2-size="p.red_count" :aria-pressed="size===p.red_count" :class="{selected:size===p.red_count}" @click="size=p.red_count">{{p.red_count}}红</button></div>
    <div v-else class="small-red-sizes" aria-label="修正版胆拖规格"><button v-for="p in bundle.dantuo" :key="p.id" type="button" :data-fusion-v2-shape="p.id" :aria-pressed="shape===p.id" :class="{selected:shape===p.id}" @click="shape=p.id">{{p.id}}</button></div>
    <template v-if="plan"><p class="reference-buyable-budget" data-fusion-v2-budget><strong>{{plan.bet_count}}注 · {{plan.cost}}元</strong><span>{{plan.title}}</span></p>
     <p v-if="overlap" class="muted" data-fusion-v2-overlap>依据上期{{current.data_cutoff||snapshot.data_cutoff}}的六红{{bundle.previous_reds.map(pad).join(' ')}}：本方案每注重合{{overlap.minimum}}～{{overlap.maximum}}红，整组六红重复{{overlap.entire}}注。<template v-if="mode==='dantuo'">胆码中有{{overlap.previousBankers.length}}个上期红球{{overlap.previousBankers.length?'：'+overlap.previousBankers.map(pad).join('、'):''}}；胆码仍须全部命中。</template></p>
     <template v-if="mode==='dantuo'"><div class="reference-plan-row" data-v2-dan><b>红胆</b><div class="balls"><span v-for="n in plan.red_dan" :key="n" class="ball selected red">{{pad(n)}}</span></div></div><div class="reference-plan-row" data-v2-tuo><b>红拖</b><div class="balls"><span v-for="n in plan.red_tuo" :key="n" class="ball selected red">{{pad(n)}}</span></div></div><div class="reference-plan-row" data-v2-blue><b>蓝球</b><div class="balls"><span v-for="n in plan.blues" :key="n" class="ball selected blue">{{pad(n)}}</span></div></div><p class="muted">C(5,{{6-plan.red_dan.length}}) × {{plan.blues.length}} ＝ 10注；完整六红仍要求全部胆码命中。</p></template>
     <p v-if="current.status==='frozen'" class="muted" data-fusion-v2-sealed>v2独立保存于{{time(current.published_at_unix)}}。<a :href="rawUrl">修正版v2原档JSON</a> <a v-if="reviewUrl" :href="reviewUrl">v2逐注复盘JSON</a></p><p v-else class="muted">当前为v2预览，尚未独立封存。</p>
     <ol class="reference-tickets" aria-label="融合修正版完整票面"><li v-for="(t,i) in plan.tickets" :key="i" data-fusion-v2-ticket><span class="reference-ticket-label">第{{i+1}}注</span><div class="balls"><span v-for="n in t.red" :key="n" class="ball selected red">{{pad(n)}}</span><span class="reference-plus">+</span><span class="ball selected blue">{{pad(t.blue)}}</span></div><small class="muted">含上期{{plan.previous_overlap[i]}}红<template v-if="!current.pending&&current.status==='frozen'"> · 中红{{nullable(plan.ticket_results?.[i]?.red_hits)}}/6 · 中蓝{{flag(plan.ticket_results?.[i]?.blue_hit)}} · {{plan.ticket_results[i].prize_level?plan.ticket_results[i].prize_level+'等奖':plan.ticket_results[i].special_prize_candidate?'特别奖待核实':'未中常规奖'}}</template></small></li></ol>
     <p v-if="plan.summary?.max_ticket_red_hits!=null" class="muted">最佳单注中{{plan.summary.max_ticket_red_hits}}红 · 常规中奖{{plan.summary.prize_tickets}}注 · 已知固定奖金{{plan.summary.fixed_prize}}元；浮动奖金及特别奖另核。</p>
     <div class="row reference-actions"><button v-if="mode==='dantuo'" type="button" @click="copy(true)">复制v2胆拖方案</button><button type="button" @click="copy(false)">复制v2完整{{plan.bet_count}}注</button><button v-if="current.status!=='frozen'" type="button" :disabled="busy||!report.publication.can_freeze" @click="freeze">{{busy?'封存中…':'封存修正版v2'}}</button></div>
     <details class="reference-evidence"><summary>查看v2评分修复与候选池</summary><p class="muted">原模型中显式重号专家的叠加权重约{{(bundle.evidence.legacy_explicit_repeat_weight*100).toFixed(2)}}%，v2为0；其余专家保留原相对权重，近期与参考支持继续参与。修复评分偏置尚未证明中奖率提升。</p><p class="reference-number-text">候选红球：{{plan.reds.slice().sort((a,b)=>a-b).map(pad).join(' ')}}</p><p class="muted">蓝球：{{plan.blues.map(pad).join(' ')}}。</p></details>
    </template>
   </template><p class="muted" data-fusion-v2-actual>v2独立留档：已开奖{{report.actual.n}}期 · 待开奖{{report.actual.pending}}期。{{report.actual.note}}</p><p v-if="report.actual.errors.length" class="notice error">{{report.actual.errors.join('；')}}</p>
   <fusion-review v-if="report.issue===report.target" :issue="issue"></fusion-review>
  </template><button type="button" @click="load" :disabled="loading">刷新修正版v2</button><p v-if="message" class="muted" role="status">{{message}}</p>
 </section>`};
 const numbers = {
  components:{ReferenceDantuo:dantuo,FusionDantuo:fusionDantuo,ReferenceProvenance:referenceProvenance,FusionV2:fusionV2},
  props:{snapshot:{type:Object,required:true},ticketBundle:{type:Object,default:null},dantuoBundle:{type:Object,default:null},issue:{type:String,default:''},archived:{type:Boolean,default:false}},
  emits:['fusion-version-change'],
  setup(props,{emit}) {
   const count = Vue.ref(16), message = Vue.ref(''),fusionVersion=Vue.ref(preferredFusionVersion(global.location?.search,props.issue));
   function chooseFusionVersion(version){if(version!=='v1'&&version!=='v2')return;fusionVersion.value=version;message.value='';try{const url=new URL(global.location.href);url.searchParams.set('fusion',version);global.history.replaceState(global.history.state,'',url.href);}catch(_){}}
   Vue.watch(fusionVersion,version=>emit('fusion-version-change',version),{immediate:true});
   Vue.watch(()=>props.issue,()=>{message.value='';fusionVersion.value=preferredFusionVersion(global.location?.search,props.issue);});
   const original = Vue.computed(() => props.snapshot.reference);
   const pool = Vue.computed(() => props.snapshot.fusion.pools.find(p => p.red_count === count.value));
   const ticketPlan=Vue.computed(()=>props.ticketBundle?.plans?.find(p=>p.red_count===count.value));
   const scored = Vue.computed(() => Object.prototype.hasOwnProperty.call(original.value, 'red_hits'));
   async function copy(which) {
    const title=which==='reference'?'参考代码原始5注':'旧版v1融合'+count.value+'红 · 单式8注16元';
    if(which==='fusion'&&!ticketPlan.value){message.value='旧档未保存完整融合票组，请查看原始红球池。';return;}
    const tickets=which==='reference'?original.value.tickets:ticketPlan.value.tickets;
    const body=tickets.map((t,i)=>(i+1)+'. '+t.red.map(pad).join(' ')+' + '+pad(t.blue)).join('\n');
    const note=which==='reference'?'5注红球并集'+original.value.red_count+'个；并集覆盖与单注命中分别核对。':'每注6红1蓝，共8注16元；本次只选择'+count.value+'红这一档。';
    const sealed=which==='reference'?props.archived:props.ticketBundle?.status==='frozen';
    try{await copyText('双色球 '+props.issue+' 期 · '+title+'\n'+body+'\n'+note+'\n'+(sealed?'完整号码来自首次封存档。':'当前完整票面预览，尚未封存。'));message.value=title+'已复制。';}
    catch (e) { message.value = e.message || '复制未成功，请手动选择号码。'; }
   }
   return {count, message, original, pool, ticketPlan, scored, copy, pad, sorted, nullable, flag,fusionVersion,chooseFusionVersion};
  },
  template:`<div class="reference-numbers">
   <div class="reference-grid">
    <article class="reference-box" data-reference-model="reference_original5">
     <h3>参考引擎 · 本站运行5注</h3><h4>单式5注 · 10元</h4><reference-provenance :issue="issue" :snapshot="snapshot"></reference-provenance>
     <p class="muted">5注合计覆盖 <b>{{ original.red_count }} 个不同红球</b>、{{ original.blue_count }} 个蓝球。</p>
     <ol class="reference-tickets" aria-label="参考代码原始5注票面">
      <li v-for="(ticket,index) in original.tickets" :key="index"><span class="reference-ticket-label">第{{ index+1 }}注</span><div class="balls"><span v-for="n in ticket.red" :key="'r'+n" class="ball selected red">{{ pad(n) }}</span><span class="reference-plus" aria-hidden="true">+</span><span class="ball selected blue">{{ pad(ticket.blue) }}</span></div><small v-if="scored" class="muted">中红 {{ nullable(original.ticket_results?.[index]?.red_hits) }}/6 · 中蓝 {{ flag(original.ticket_results?.[index]?.blue_hit) }}</small></li>
     </ol>
     <p v-if="scored" class="muted">并集中红 {{ nullable(original.red_hits) }}/6 · 全6红 {{ flag(original.full_red) }}<br>最佳单注中红 {{ nullable(original.best_ticket_red_hits) }}/6 · 蓝球入池 {{ flag(original.blue_hit) }}</p>
     <details class="reference-union"><summary>查看{{ original.red_count }}红并集及同数量对照</summary><p class="reference-number-text">{{ sorted(original.reds).map(pad).join(' ') }}</p><p class="muted">原加权融合榜同取{{ original.red_count }}红：{{ sorted(original.baseline.reds).map(pad).join(' ') }}</p><p v-if="scored" class="muted">同数量原榜中红 {{ nullable(original.baseline.red_hits) }}/6 · 全6红 {{ flag(original.baseline.full_red) }}</p></details>
     <button type="button" @click="copy('reference')">复制原始5注</button><reference-dantuo :bundle="dantuoBundle" :issue="issue" :archived="archived"></reference-dantuo>
    </article>
    <article class="reference-box" data-reference-model="reference_fused16">
     <h3>融合推荐 · 单式与胆拖</h3><div class="small-red-sizes" aria-label="融合推荐版本"><button type="button" data-fusion-version="v2" :aria-pressed="fusionVersion==='v2'" :class="{selected:fusionVersion==='v2'}" @click="chooseFusionVersion('v2')">修正版v2</button><button type="button" data-fusion-version="v1" :aria-pressed="fusionVersion==='v1'" :class="{selected:fusionVersion==='v1'}" @click="chooseFusionVersion('v1')">旧版v1原档</button></div>
     <fusion-v2 v-if="fusionVersion==='v2'" :issue="issue" :snapshot="snapshot"></fusion-v2><template v-else><p class="notice">这是旧版v1首次原档，旧算法可能整组复用上期六红；当前默认推荐为v2。查看、复制本区内容均为旧v1，不会替换已保存号码。</p>
     <p class="muted">现有小池评分90%＋参考5注并集支持10%；并集内均匀分配，重复出现不加权，最多16红。</p>
     <p class="muted">历史对照尚未优于现有小池，独立观察中。</p>
     <div class="small-red-sizes" aria-label="参考融合红球数量"><button v-for="p in snapshot.fusion.pools" :key="p.red_count" type="button" :data-reference-size="p.red_count" :aria-pressed="count===p.red_count" :class="{selected:count===p.red_count}" @click="count=p.red_count">{{ p.red_count }}红{{ p.red_count===16?' · 上限':'' }}</button></div>
     <template v-if="pool">
      <template v-if="ticketPlan"><p class="reference-buyable-budget"><strong>{{ticketPlan.bet_count}}注单式 · {{ticketPlan.cost}}元</strong><span>每注6红＋1蓝，2个蓝球交替；四档中选择一档。</span></p>
       <ol class="reference-tickets fusion-ticket-list" aria-label="融合8注可投注票面"><li v-for="(ticket,index) in ticketPlan.tickets" :key="index" data-fusion-ticket><span class="reference-ticket-label">第{{index+1}}注</span><div class="balls"><span v-for="n in ticket.red" :key="n" class="ball selected red">{{pad(n)}}</span><span class="reference-plus">+</span><span class="ball selected blue">{{pad(ticket.blue)}}</span></div><small v-if="ticketPlan.ticket_results?.[index]?.red_hits!=null" class="muted">中红{{ticketPlan.ticket_results[index].red_hits}}/6 · 蓝球{{flag(ticketPlan.ticket_results[index].blue_hit)}} · {{ticketPlan.ticket_results[index].prize_level?ticketPlan.ticket_results[index].prize_level+'等奖':ticketPlan.ticket_results[index].special_prize_candidate?'福运奖待核实':'未中常规奖'}}</small></li></ol>
       <p v-if="ticketPlan.summary?.max_ticket_red_hits!=null" class="muted">单注最多中{{ticketPlan.summary.max_ticket_red_hits}}红 · 常规中奖{{ticketPlan.summary.prize_tickets}}注 · 已知固定奖金{{ticketPlan.summary.fixed_prize}}元。浮动奖金与特别奖另核。</p>
       <p v-if="ticketBundle.status!=='frozen'" class="muted">当前完整票面预览，尚未封存；封存后按这8注逐注复盘。</p>
       <button type="button" @click="copy('fusion')">复制旧v1完整8注</button>
      </template>
      <p v-else class="notice">旧档只保存了红球候选池，没有当时的完整融合票组；原始号码在下方保留。</p>
      <details class="reference-union"><summary>查看{{count}}红候选池、2蓝与原榜对照</summary><div class="balls reference-fused-balls" aria-label="参考融合红球号码"><span v-for="n in sorted(pool.reds)" :key="n" class="ball selected red">{{pad(n)}}</span></div><p class="muted">蓝球：{{snapshot.fusion.blues.map(pad).join(' ')}}</p><p v-if="scored" class="muted">池中红{{nullable(pool.red_hits)}}/6 · 全6红{{flag(pool.full_red)}}；漏红{{pool.missed_reds==null?'—':pool.missed_reds.map(pad).join(' ')||'无'}}。</p><p class="reference-number-text">同{{count}}红的现有小池：{{sorted(pool.baseline.reds).map(pad).join(' ')}}</p><p v-if="scored" class="muted">现有小池中红{{nullable(pool.baseline.red_hits)}}/6 · 全6红{{flag(pool.baseline.full_red)}}</p></details>
     </template>
     <fusion-dantuo :issue="issue" :snapshot="snapshot"></fusion-dantuo></template>
    </article>
   </div><p v-if="message" class="muted reference-message" role="status">{{ message }}</p>
  </div>`
 };
 const archive = {
  components:{ReferenceNumbers:numbers}, props:{record:{type:Object,required:true}},
  setup(){return {time,fileUrl};},
  template:`<section class="card reference-observation" id="independent-reference" aria-label="参考与融合独立观察复盘"><h2>参考5注、胆拖与融合8注 · 独立原档</h2><p class="muted">第 {{ record.issue }} 期 · 参考原档保存时间 {{ time(record.published_at_unix) }} · 依据截止 {{ record.data_cutoff }}</p><p class="muted">逐注命中与红球并集覆盖分别核对；待开奖显示“—”。这里只使用首次保存的号码。</p><p v-if="record.fusion_tickets" class="muted">旧v1融合8注保存于{{time(record.fusion_tickets.published_at_unix)}}；这是旧v1的保存时间，v2本期若有原档会在对应版本中单独显示。</p><reference-numbers :dantuo-bundle="record.dantuo" :ticket-bundle="record.fusion_tickets" :snapshot="record.reference_pools" :issue="record.issue" archived></reference-numbers><div class="row reference-actions"><a :href="fileUrl(record.raw_file)">下载参考观察原档JSON</a><a v-if="record.fusion_tickets?.raw_file" :href="fileUrl(record.fusion_tickets.raw_file)">旧v1融合8注原档JSON</a><a v-if="record.fusion_tickets?.review_file" :href="fileUrl(record.fusion_tickets.review_file)">旧v1融合逐注复盘JSON</a><a v-if="record.review_file" :href="fileUrl(record.review_file)">下载首次复盘JSON</a></div><details class="reference-evidence"><summary>查看封存依据</summary><p class="muted">原档摘要 <code class="hash">{{ record.archive_sha256 }}</code></p><p class="muted">{{ record.review_persisted?'复盘已保存。':'开奖后核对并保存复盘。' }}本地留档无独立可信时间戳证明。</p></details></section>`
 };
 const component = {
  components:{ReferenceNumbers:numbers}, props:{busy:{type:Boolean,default:false}},
  setup() {
   const report=Vue.ref(null), error=Vue.ref(''), message=Vue.ref(''), loading=Vue.ref(false), freezing=Vue.ref(false),fusionVersion=Vue.ref(preferredFusionVersion(global.location?.search));
   async function load() {
    if(loading.value) return; loading.value=true; error.value='';
    try { const response=await fetch(SSQApp.url('api/reference-pools')), data=await response.json(); if(!response.ok) throw Error(data.error || '参考观察读取失败'); report.value=data; }
    catch(e) {error.value=e.message;} finally {loading.value=false;}
   }
   async function freeze() {
    if(freezing.value || !report.value?.next) return; freezing.value=true; message.value='';
    try {const response=await fetch(SSQApp.url('api/reference-pools/freeze'),{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({data_sha256:report.value.data_sha256,target:report.value.target})}), data=await response.json();if(!response.ok)throw Error(data.error||'参考观察封存失败');await load();message.value=data.dantuo_archive?.status==='frozen'?'参考胆拖／复式方案已独立封存。':data.dantuo_archive?.status==='waiting'?data.dantuo_archive.reason:data.fusion_tickets_archive?.status==='frozen'?'融合8注票组已独立封存。':'已保留本期首次参考、胆拖与融合原档。';}
    catch(e){message.value=e.message;}finally{freezing.value=false;}
   }
   function download() {
    if(!report.value?.next)return;
    const value={kind:'reference_observation_preview',actual_recommendation:false,target_issue:report.value.target,data_cutoff:report.value.data_cutoff,input_history_sha256:report.value.data_sha256,reference_pools:report.value.next,fusion_tickets:report.value.fusion_tickets,dantuo:report.value.dantuo};
    const url=URL.createObjectURL(new Blob([JSON.stringify(value,null,2)],{type:'application/json'})),link=document.createElement('a');link.href=url;link.download='ssq-reference-preview-'+report.value.target+'.json';link.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
   }
   const reviewUrl=Vue.computed(()=>SSQApp.url('reviews',{issue:report.value?.target,archive:'daily',fusion:fusionVersion.value})+'#independent-reference');
   const cells=Vue.computed(()=>report.value?.actual?.cells || []);
   const archive=Vue.computed(()=>report.value?.next_archive);
   Vue.onMounted(load);
   return {report,error,message,loading,freezing,load,freeze,download,archive,cells,reviewUrl,fileUrl,percent,time,fusionVersion};
  },
  template:`<section class="card reference-observation" id="reference-pools" aria-label="参考代码与融合观察池">
   <div class="row between"><h2>参考5注、预算方案与融合推荐</h2><button type="button" :disabled="loading||freezing" @click="load">{{ loading?'读取中…':'刷新观察状态' }}</button></div>
   <p class="muted">参考原始5注、同次原始胆拖／复式和融合8注分别展示，可复制完整票面。预算方案按原报告注数计价；融合单式8注16元，两种融合胆拖各10注20元，可独立选择。</p>
   <p v-if="error" class="notice error" role="alert">{{ error }}</p><p v-if="!report&&!error" class="muted" role="status">正在读取参考运行结果与首次观察原档…</p>
   <template v-if="report"><p class="muted">第 {{ report.target }} 期 · 依据截止 {{ report.data_cutoff }}</p>
    <p v-if="report.status==='error'" class="notice error" role="alert">{{ report.reason }}</p><p v-else-if="!report.next" class="notice" role="status">{{ report.reason || '参考代码正在准备本期结果，完成后显示原始5注和融合小池。' }}</p><p v-if="!report.next&&report.job?.started_at_unix" class="muted">本期运算开始于 {{ time(report.job.started_at_unix) }}。{{ report.job.status==='running'?'完成后刷新即可查看两个观察池。':report.job.error||'' }}</p>
    <template v-if="report.next"><p v-if="archive" class="notice success reference-archive-status">参考原档已封存，保存于 {{ time(archive.published_at_unix) }}。<a :href="reviewUrl">查看原始号码与复盘</a></p><p v-if="report.fusion_tickets?.status==='frozen'" class="notice success">旧版v1融合8注已封存，保存于{{time(report.fusion_tickets.published_at_unix)}}。</p><p v-else-if="!archive" class="notice reference-archive-status">当前为本期预览，尚未封存。完成截止前留档后，才计入后续实际观察。</p>
     <reference-numbers :dantuo-bundle="report.dantuo" :ticket-bundle="report.fusion_tickets" :snapshot="report.next" :issue="report.target" :archived="!!archive" @fusion-version-change="fusionVersion=$event"></reference-numbers>
     <div class="row reference-actions"><template v-if="archive"><a :href="fileUrl(archive.raw_file)">参考原档JSON</a><a :href="reviewUrl">查看独立复盘</a></template><template v-if="report.fusion_tickets?.status==='frozen'"><a :href="fileUrl(report.fusion_tickets.raw_file)">旧v1融合8注原档JSON</a></template><template v-if="report.fusion_tickets?.status!=='frozen'"><button type="button" :disabled="busy||freezing||!report.publication?.can_freeze" @click="freeze">{{freezing?'正在封存…':archive?'封存融合8注票组':'封存原始5注与融合票组'}}</button><button type="button" @click="download">导出参考及旧v1预览JSON</button></template><button v-if="report.dantuo?.status==='preview'" type="button" :disabled="busy||freezing||!report.publication?.can_freeze" @click="freeze">{{freezing?'正在封存…':'封存参考胆拖／复式方案'}}</button></div>
     <p v-if="!archive&&report.publication" class="muted">{{ report.publication.reason }}</p>
    </template><p v-if="message" class="reference-message muted" role="status">{{ message }}</p>
    <details v-if="report.dantuo_actual" class="reference-evidence"><summary>参考胆拖／复式 · 实际逐注对奖统计</summary><p class="muted">{{report.dantuo_actual.note}}</p><p class="muted">已开奖{{report.dantuo_actual.n}}期 · 待开奖{{report.dantuo_actual.pending}}期 · 缺档{{report.dantuo_actual.excluded_missing}}期；至少一注常规中奖{{report.dantuo_actual.n?report.dantuo_actual.any_prize_periods+'/'+report.dantuo_actual.n:'—'}}；平均最佳单注中红{{report.dantuo_actual.mean_best_red==null?'—':Number(report.dantuo_actual.mean_best_red).toFixed(2)}}。</p><p v-if="report.dantuo_actual.errors?.length" class="notice error">{{report.dantuo_actual.errors.join('；')}}</p></details>
    <details v-if="report.ticket_actual" class="reference-evidence"><summary>旧版v1融合8注 · 实际逐注对奖统计</summary><p class="muted">{{report.ticket_actual.note}}</p><div class="scroll"><table><thead><tr><th>规格</th><th>已开奖 / 待开奖</th><th>至少一注常规中奖</th><th>平均单注最好中红</th><th>已知固定奖金累计</th></tr></thead><tbody><tr v-for="r in report.ticket_actual.cells" :key="r.red_count"><td>{{r.red_count}}红候选 · 8注16元</td><td>{{r.n}} / {{r.pending}}</td><td>{{r.n?r.any_prize_periods+'/'+r.n:'—'}}</td><td>{{r.mean_best_red==null?'—':Number(r.mean_best_red).toFixed(2)}}</td><td>{{r.fixed_prize_total==null?'—':r.fixed_prize_total+'元'}}</td></tr></tbody></table></div></details>
    <details class="reference-evidence"><summary>查看实际覆盖与同数量对照</summary><p class="muted">仅统计本系统截止前首次封存且已开奖的期数。待开奖和缺档不计入全包率；参考目录里的历史回放不混入实际成绩。</p>
     <p v-if="report.actual?.errors?.length" class="notice error">部分原档未计入：{{ report.actual.errors.join('；') }}</p>
     <div class="scroll"><table class="reference-stat-table"><thead><tr><th>观察池</th><th>红球数量</th><th>全6红次数 / 比例</th><th>比同数量对照<br>补中 / 丢失</th><th>待开奖 / 缺档</th></tr></thead><tbody><tr v-for="r in cells" :key="r.model+'-'+r.red_count" :data-reference-actual="r.model+'-'+r.red_count"><td>{{r.model==='reference_fused16'?'旧版v1 · '+r.label:r.label}}</td><td>{{ r.red_count==null?(r.mean_red_count==null?'随原5注并集变化':'平均'+Number(r.mean_red_count).toFixed(1)+'红'):r.red_count+'红' }}</td><td>{{ r.n?r.full_red_periods+'/'+r.n+' · '+percent(r.full_red_rate):'暂无已开奖样本' }}</td><td>{{ r.n?r.paired_gains+' / '+r.paired_losses+'期':'—' }}</td><td>{{ r.pending }} / {{ r.excluded_missing_pair }}</td></tr></tbody></table></div>
     <p class="muted">原始5注对照同数量原加权榜；融合池对照同数量现有小池。旧版v1采用90%现有评分＋10%参考并集支持，修正版v2另行留档统计。留档无独立可信时间戳证明。</p>
    </details>
   </template>
  </section>`
 };
 global.SSQReferencePools={component,archive,numbers,parity:{shortIssue,sourceCandidates,compareSource,preferredFusionVersion}};
 if(typeof module!=='undefined'&&module.exports)module.exports=global.SSQReferencePools.parity;
})(typeof window==='undefined'?globalThis:window);
