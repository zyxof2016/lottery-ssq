(function (global) {
 'use strict';
 const pad = n => String(n).padStart(2, '0');
 const sorted = ns => (ns || []).slice().sort((a, b) => a - b);
 const percent = n => n == null ? '—' : (Number(n) * 100).toFixed(2) + '%';
 const nullable = n => n == null ? '—' : n;
 const flag = n => n == null ? '—' : n ? '是' : '否';
 const time = n => n ? new Date(n * 1000).toLocaleString('zh-CN', {timeZone:'Asia/Shanghai', hour12:false}) + '（北京时间）' : '未留存';
 const fileUrl = path => SSQApp.url('api/recommendations/files', {path});
 async function copyText(text) {
  if (navigator.clipboard) return navigator.clipboard.writeText(text);
  const input = document.createElement('textarea');
  input.value = text; document.body.appendChild(input); input.select();
  const ok = document.execCommand('copy'); input.remove();
  if (!ok) throw Error('复制未成功，请手动选择号码。');
 }
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
   <h4>参考预算方案 · 胆拖／复式</h4>
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
 const fusionDantuo={props:{issue:{type:String,required:true},snapshot:{type:Object,required:true}},setup(props){
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
  </template><p v-if="message" class="muted" role="status">{{message}}</p>
 </section>`};
 const numbers = {
  components:{ReferenceDantuo:dantuo,FusionDantuo:fusionDantuo},
  props:{snapshot:{type:Object,required:true},ticketBundle:{type:Object,default:null},dantuoBundle:{type:Object,default:null},issue:{type:String,default:''},archived:{type:Boolean,default:false}},
  setup(props) {
   const count = Vue.ref(16), message = Vue.ref('');
   const original = Vue.computed(() => props.snapshot.reference);
   const pool = Vue.computed(() => props.snapshot.fusion.pools.find(p => p.red_count === count.value));
   const ticketPlan=Vue.computed(()=>props.ticketBundle?.plans?.find(p=>p.red_count===count.value));
   const scored = Vue.computed(() => Object.prototype.hasOwnProperty.call(original.value, 'red_hits'));
   async function copy(which) {
    const title=which==='reference'?'参考代码原始5注':'参考融合'+count.value+'红 · 单式8注16元';
    if(which==='fusion'&&!ticketPlan.value){message.value='旧档未保存完整融合票组，请查看原始红球池。';return;}
    const tickets=which==='reference'?original.value.tickets:ticketPlan.value.tickets;
    const body=tickets.map((t,i)=>(i+1)+'. '+t.red.map(pad).join(' ')+' + '+pad(t.blue)).join('\n');
    const note=which==='reference'?'5注红球并集'+original.value.red_count+'个；并集覆盖与单注命中分别核对。':'每注6红1蓝，共8注16元；本次只选择'+count.value+'红这一档。';
    const sealed=which==='reference'?props.archived:props.ticketBundle?.status==='frozen';
    try{await copyText('双色球 '+props.issue+' 期 · '+title+'\n'+body+'\n'+note+'\n'+(sealed?'完整号码来自首次封存档。':'当前完整票面预览，尚未封存。'));message.value=title+'已复制。';}
    catch (e) { message.value = e.message || '复制未成功，请手动选择号码。'; }
   }
   return {count, message, original, pool, ticketPlan, scored, copy, pad, sorted, nullable, flag};
  },
  template:`<div class="reference-numbers">
   <div class="reference-grid">
    <article class="reference-box" data-reference-model="reference_original5">
     <h3>参考代码 · 原始推荐</h3><h4>单式5注 · 10元</h4>
     <p class="muted">5注合计覆盖 <b>{{ original.red_count }} 个不同红球</b>、{{ original.blue_count }} 个蓝球。</p>
     <ol class="reference-tickets" aria-label="参考代码原始5注票面">
      <li v-for="(ticket,index) in original.tickets" :key="index"><span class="reference-ticket-label">第{{ index+1 }}注</span><div class="balls"><span v-for="n in ticket.red" :key="'r'+n" class="ball selected red">{{ pad(n) }}</span><span class="reference-plus" aria-hidden="true">+</span><span class="ball selected blue">{{ pad(ticket.blue) }}</span></div><small v-if="scored" class="muted">中红 {{ nullable(original.ticket_results?.[index]?.red_hits) }}/6 · 中蓝 {{ flag(original.ticket_results?.[index]?.blue_hit) }}</small></li>
     </ol>
     <p v-if="scored" class="muted">并集中红 {{ nullable(original.red_hits) }}/6 · 全6红 {{ flag(original.full_red) }}<br>最佳单注中红 {{ nullable(original.best_ticket_red_hits) }}/6 · 蓝球入池 {{ flag(original.blue_hit) }}</p>
     <details class="reference-union"><summary>查看{{ original.red_count }}红并集及同数量对照</summary><p class="reference-number-text">{{ sorted(original.reds).map(pad).join(' ') }}</p><p class="muted">原加权融合榜同取{{ original.red_count }}红：{{ sorted(original.baseline.reds).map(pad).join(' ') }}</p><p v-if="scored" class="muted">同数量原榜中红 {{ nullable(original.baseline.red_hits) }}/6 · 全6红 {{ flag(original.baseline.full_red) }}</p></details>
     <button type="button" @click="copy('reference')">复制原始5注</button><reference-dantuo :bundle="dantuoBundle" :issue="issue" :archived="archived"></reference-dantuo>
    </article>
    <article class="reference-box" data-reference-model="reference_fused16">
     <h3>融合推荐 · 单式与胆拖</h3>
     <p class="muted">现有小池评分90%＋参考5注并集支持10%；并集内均匀分配，重复出现不加权，最多16红。</p>
     <p class="muted">历史对照尚未优于现有小池，独立观察中。</p>
     <div class="small-red-sizes" aria-label="参考融合红球数量"><button v-for="p in snapshot.fusion.pools" :key="p.red_count" type="button" :data-reference-size="p.red_count" :aria-pressed="count===p.red_count" :class="{selected:count===p.red_count}" @click="count=p.red_count">{{ p.red_count }}红{{ p.red_count===16?' · 上限':'' }}</button></div>
     <template v-if="pool">
      <template v-if="ticketPlan"><p class="reference-buyable-budget"><strong>{{ticketPlan.bet_count}}注单式 · {{ticketPlan.cost}}元</strong><span>每注6红＋1蓝，2个蓝球交替；四档中选择一档。</span></p>
       <ol class="reference-tickets fusion-ticket-list" aria-label="融合8注可投注票面"><li v-for="(ticket,index) in ticketPlan.tickets" :key="index" data-fusion-ticket><span class="reference-ticket-label">第{{index+1}}注</span><div class="balls"><span v-for="n in ticket.red" :key="n" class="ball selected red">{{pad(n)}}</span><span class="reference-plus">+</span><span class="ball selected blue">{{pad(ticket.blue)}}</span></div><small v-if="ticketPlan.ticket_results?.[index]?.red_hits!=null" class="muted">中红{{ticketPlan.ticket_results[index].red_hits}}/6 · 蓝球{{flag(ticketPlan.ticket_results[index].blue_hit)}} · {{ticketPlan.ticket_results[index].prize_level?ticketPlan.ticket_results[index].prize_level+'等奖':ticketPlan.ticket_results[index].special_prize_candidate?'福运奖待核实':'未中常规奖'}}</small></li></ol>
       <p v-if="ticketPlan.summary?.max_ticket_red_hits!=null" class="muted">单注最多中{{ticketPlan.summary.max_ticket_red_hits}}红 · 常规中奖{{ticketPlan.summary.prize_tickets}}注 · 已知固定奖金{{ticketPlan.summary.fixed_prize}}元。浮动奖金与特别奖另核。</p>
       <p v-if="ticketBundle.status!=='frozen'" class="muted">当前完整票面预览，尚未封存；封存后按这8注逐注复盘。</p>
       <button type="button" @click="copy('fusion')">复制8注打单文本</button>
      </template>
      <p v-else class="notice">旧档只保存了红球候选池，没有当时的完整融合票组；原始号码在下方保留。</p>
      <details class="reference-union"><summary>查看{{count}}红候选池、2蓝与原榜对照</summary><div class="balls reference-fused-balls" aria-label="参考融合红球号码"><span v-for="n in sorted(pool.reds)" :key="n" class="ball selected red">{{pad(n)}}</span></div><p class="muted">蓝球：{{snapshot.fusion.blues.map(pad).join(' ')}}</p><p v-if="scored" class="muted">池中红{{nullable(pool.red_hits)}}/6 · 全6红{{flag(pool.full_red)}}；漏红{{pool.missed_reds==null?'—':pool.missed_reds.map(pad).join(' ')||'无'}}。</p><p class="reference-number-text">同{{count}}红的现有小池：{{sorted(pool.baseline.reds).map(pad).join(' ')}}</p><p v-if="scored" class="muted">现有小池中红{{nullable(pool.baseline.red_hits)}}/6 · 全6红{{flag(pool.baseline.full_red)}}</p></details>
     </template>
     <fusion-dantuo :issue="issue" :snapshot="snapshot"></fusion-dantuo>
    </article>
   </div><p v-if="message" class="muted reference-message" role="status">{{ message }}</p>
  </div>`
 };
 const archive = {
  components:{ReferenceNumbers:numbers}, props:{record:{type:Object,required:true}},
  setup(){return {time,fileUrl};},
  template:`<section class="card reference-observation" id="independent-reference" aria-label="参考与融合独立观察复盘"><h2>参考5注、胆拖与融合8注 · 独立原档</h2><p class="muted">第 {{ record.issue }} 期 · 参考原档保存时间 {{ time(record.published_at_unix) }} · 依据截止 {{ record.data_cutoff }}</p><p class="muted">逐注命中与红球并集覆盖分别核对；待开奖显示“—”。这里只使用首次保存的号码。</p><p v-if="record.fusion_tickets" class="muted">融合8注单独保存于{{time(record.fusion_tickets.published_at_unix)}}，按这份票组原档逐注核对。</p><reference-numbers :dantuo-bundle="record.dantuo" :ticket-bundle="record.fusion_tickets" :snapshot="record.reference_pools" :issue="record.issue" archived></reference-numbers><div class="row reference-actions"><a :href="fileUrl(record.raw_file)">下载参考观察原档JSON</a><a v-if="record.fusion_tickets?.raw_file" :href="fileUrl(record.fusion_tickets.raw_file)">融合8注票组原档JSON</a><a v-if="record.fusion_tickets?.review_file" :href="fileUrl(record.fusion_tickets.review_file)">融合逐注复盘JSON</a><a v-if="record.review_file" :href="fileUrl(record.review_file)">下载首次复盘JSON</a></div><details class="reference-evidence"><summary>查看封存依据</summary><p class="muted">原档摘要 <code class="hash">{{ record.archive_sha256 }}</code></p><p class="muted">{{ record.review_persisted?'复盘已保存。':'开奖后核对并保存复盘。' }}本地留档无独立可信时间戳证明。</p></details></section>`
 };
 const component = {
  components:{ReferenceNumbers:numbers}, props:{busy:{type:Boolean,default:false}},
  setup() {
   const report=Vue.ref(null), error=Vue.ref(''), message=Vue.ref(''), loading=Vue.ref(false), freezing=Vue.ref(false);
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
   const reviewUrl=Vue.computed(()=>SSQApp.url('reviews',{issue:report.value?.target,archive:'daily'})+'#independent-reference');
   const cells=Vue.computed(()=>report.value?.actual?.cells || []);
   const archive=Vue.computed(()=>report.value?.next_archive);
   Vue.onMounted(load);
   return {report,error,message,loading,freezing,load,freeze,download,archive,cells,reviewUrl,fileUrl,percent,time};
  },
  template:`<section class="card reference-observation" id="reference-pools" aria-label="参考代码与融合观察池">
   <div class="row between"><h2>参考5注、预算方案与融合推荐</h2><button type="button" :disabled="loading||freezing" @click="load">{{ loading?'读取中…':'刷新观察状态' }}</button></div>
   <p class="muted">参考原始5注、同次原始胆拖／复式和融合8注分别展示，可复制完整票面。预算方案按原报告注数计价；融合单式8注16元，两种融合胆拖各10注20元，可独立选择。</p>
   <p v-if="error" class="notice error" role="alert">{{ error }}</p><p v-if="!report&&!error" class="muted" role="status">正在读取参考运行结果与首次观察原档…</p>
   <template v-if="report"><p class="muted">第 {{ report.target }} 期 · 依据截止 {{ report.data_cutoff }}</p>
    <p v-if="report.status==='error'" class="notice error" role="alert">{{ report.reason }}</p><p v-else-if="!report.next" class="notice" role="status">{{ report.reason || '参考代码正在准备本期结果，完成后显示原始5注和融合小池。' }}</p><p v-if="!report.next&&report.job?.started_at_unix" class="muted">本期运算开始于 {{ time(report.job.started_at_unix) }}。{{ report.job.status==='running'?'完成后刷新即可查看两个观察池。':report.job.error||'' }}</p>
    <template v-if="report.next"><p v-if="archive" class="notice success reference-archive-status">参考原档已封存，保存于 {{ time(archive.published_at_unix) }}。<a :href="reviewUrl">查看原始号码与复盘</a></p><p v-if="report.fusion_tickets?.status==='frozen'" class="notice success">融合8注票组已独立封存，保存于{{time(report.fusion_tickets.published_at_unix)}}。</p><p v-else-if="!archive" class="notice reference-archive-status">当前为本期预览，尚未封存。完成截止前留档后，才计入后续实际观察。</p>
     <reference-numbers :dantuo-bundle="report.dantuo" :ticket-bundle="report.fusion_tickets" :snapshot="report.next" :issue="report.target" :archived="!!archive"></reference-numbers>
     <div class="row reference-actions"><template v-if="archive"><a :href="fileUrl(archive.raw_file)">参考原档JSON</a><a :href="reviewUrl">查看独立复盘</a></template><template v-if="report.fusion_tickets?.status==='frozen'"><a :href="fileUrl(report.fusion_tickets.raw_file)">融合8注票组原档JSON</a></template><template v-if="report.fusion_tickets?.status!=='frozen'"><button type="button" :disabled="busy||freezing||!report.publication?.can_freeze" @click="freeze">{{freezing?'正在封存…':archive?'封存融合8注票组':'封存原始5注与融合票组'}}</button><button type="button" @click="download">导出完整预览JSON</button></template><button v-if="report.dantuo?.status==='preview'" type="button" :disabled="busy||freezing||!report.publication?.can_freeze" @click="freeze">{{freezing?'正在封存…':'封存参考胆拖／复式方案'}}</button></div>
     <p v-if="!archive&&report.publication" class="muted">{{ report.publication.reason }}</p>
    </template><p v-if="message" class="reference-message muted" role="status">{{ message }}</p>
    <details v-if="report.dantuo_actual" class="reference-evidence"><summary>参考胆拖／复式 · 实际逐注对奖统计</summary><p class="muted">{{report.dantuo_actual.note}}</p><p class="muted">已开奖{{report.dantuo_actual.n}}期 · 待开奖{{report.dantuo_actual.pending}}期 · 缺档{{report.dantuo_actual.excluded_missing}}期；至少一注常规中奖{{report.dantuo_actual.n?report.dantuo_actual.any_prize_periods+'/'+report.dantuo_actual.n:'—'}}；平均最佳单注中红{{report.dantuo_actual.mean_best_red==null?'—':Number(report.dantuo_actual.mean_best_red).toFixed(2)}}。</p><p v-if="report.dantuo_actual.errors?.length" class="notice error">{{report.dantuo_actual.errors.join('；')}}</p></details>
    <details v-if="report.ticket_actual" class="reference-evidence"><summary>融合8注 · 实际逐注对奖统计</summary><p class="muted">{{report.ticket_actual.note}}</p><div class="scroll"><table><thead><tr><th>规格</th><th>已开奖 / 待开奖</th><th>至少一注常规中奖</th><th>平均单注最好中红</th><th>已知固定奖金累计</th></tr></thead><tbody><tr v-for="r in report.ticket_actual.cells" :key="r.red_count"><td>{{r.red_count}}红候选 · 8注16元</td><td>{{r.n}} / {{r.pending}}</td><td>{{r.n?r.any_prize_periods+'/'+r.n:'—'}}</td><td>{{r.mean_best_red==null?'—':Number(r.mean_best_red).toFixed(2)}}</td><td>{{r.fixed_prize_total==null?'—':r.fixed_prize_total+'元'}}</td></tr></tbody></table></div></details>
    <details class="reference-evidence"><summary>查看实际覆盖与同数量对照</summary><p class="muted">仅统计本系统截止前首次封存且已开奖的期数。待开奖和缺档不计入全包率；参考目录里的历史回放不混入实际成绩。</p>
     <p v-if="report.actual?.errors?.length" class="notice error">部分原档未计入：{{ report.actual.errors.join('；') }}</p>
     <div class="scroll"><table class="reference-stat-table"><thead><tr><th>观察池</th><th>红球数量</th><th>全6红次数 / 比例</th><th>比同数量对照<br>补中 / 丢失</th><th>待开奖 / 缺档</th></tr></thead><tbody><tr v-for="r in cells" :key="r.model+'-'+r.red_count" :data-reference-actual="r.model+'-'+r.red_count"><td>{{ r.label }}</td><td>{{ r.red_count==null?(r.mean_red_count==null?'随原5注并集变化':'平均'+Number(r.mean_red_count).toFixed(1)+'红'):r.red_count+'红' }}</td><td>{{ r.n?r.full_red_periods+'/'+r.n+' · '+percent(r.full_red_rate):'暂无已开奖样本' }}</td><td>{{ r.n?r.paired_gains+' / '+r.paired_losses+'期':'—' }}</td><td>{{ r.pending }} / {{ r.excluded_missing_pair }}</td></tr></tbody></table></div>
     <p class="muted">原始5注对照同数量原加权榜；融合池对照同数量现有小池。90%现有评分＋10%参考并集支持为固定试验，尚待实际样本验证。留档无独立可信时间戳证明。</p>
    </details>
   </template>
  </section>`
 };
 global.SSQReferencePools={component,archive,numbers};
})(typeof window==='undefined'?globalThis:window);
