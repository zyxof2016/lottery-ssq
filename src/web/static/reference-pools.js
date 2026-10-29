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
 const numbers = {
  props:{snapshot:{type:Object,required:true},issue:{type:String,default:''},archived:{type:Boolean,default:false}},
  setup(props) {
   const count = Vue.ref(16), message = Vue.ref('');
   const original = Vue.computed(() => props.snapshot.reference);
   const pool = Vue.computed(() => props.snapshot.fusion.pools.find(p => p.red_count === count.value));
   const scored = Vue.computed(() => Object.prototype.hasOwnProperty.call(original.value, 'red_hits'));
   async function copy(which) {
    const title = which === 'reference' ? '参考代码原始5注' : '参考融合 · ' + count.value + '红观察池';
    const body = which === 'reference' ? original.value.tickets.map((t, i) => (i + 1) + '. ' + t.red.map(pad).join(' ') + ' + ' + pad(t.blue)).join('\n') : sorted(pool.value.reds).map(pad).join(' ');
    const note = which === 'reference' ? '\n5注红球并集 ' + original.value.red_count + ' 个；并集覆盖与单注命中分别核对。' : '\n此为红球候选池，不是完整投注票组。';
    try { await copyText('双色球 ' + props.issue + ' 期 · ' + title + '\n' + body + note + '\n' + (props.archived ? '号码来自首次独立封存档。' : '当前预览，尚未封存。')); message.value = title + '已复制。'; }
    catch (e) { message.value = e.message || '复制未成功，请手动选择号码。'; }
   }
   return {count, message, original, pool, scored, copy, pad, sorted, nullable, flag};
  },
  template:`<div class="reference-numbers">
   <div class="reference-grid">
    <article class="reference-box" data-reference-model="reference_original5">
     <h3>参考代码 · 原始5注</h3>
     <p class="muted">5注合计覆盖 <b>{{ original.red_count }} 个不同红球</b>、{{ original.blue_count }} 个蓝球。</p>
     <ol class="reference-tickets" aria-label="参考代码原始5注票面">
      <li v-for="(ticket,index) in original.tickets" :key="index"><span class="reference-ticket-label">第{{ index+1 }}注</span><div class="balls"><span v-for="n in ticket.red" :key="'r'+n" class="ball selected red">{{ pad(n) }}</span><span class="reference-plus" aria-hidden="true">+</span><span class="ball selected blue">{{ pad(ticket.blue) }}</span></div><small v-if="scored" class="muted">中红 {{ nullable(original.ticket_results?.[index]?.red_hits) }}/6 · 中蓝 {{ flag(original.ticket_results?.[index]?.blue_hit) }}</small></li>
     </ol>
     <p v-if="scored" class="muted">并集中红 {{ nullable(original.red_hits) }}/6 · 全6红 {{ flag(original.full_red) }}<br>最佳单注中红 {{ nullable(original.best_ticket_red_hits) }}/6 · 蓝球入池 {{ flag(original.blue_hit) }}</p>
     <details class="reference-union"><summary>查看{{ original.red_count }}红并集及同数量对照</summary><p class="reference-number-text">{{ sorted(original.reds).map(pad).join(' ') }}</p><p class="muted">原加权融合榜同取{{ original.red_count }}红：{{ sorted(original.baseline.reds).map(pad).join(' ') }}</p><p v-if="scored" class="muted">同数量原榜中红 {{ nullable(original.baseline.red_hits) }}/6 · 全6红 {{ flag(original.baseline.full_red) }}</p></details>
     <button type="button" @click="copy('reference')">复制原始5注</button>
    </article>
    <article class="reference-box" data-reference-model="reference_fused16">
     <h3>现有模型＋参考 · 融合小池</h3>
     <p class="muted">现有小池评分90%＋参考5注并集支持10%；并集内均匀分配，重复出现不加权，最多16红。</p>
     <p class="muted">历史对照尚未优于现有小池，独立观察中。</p>
     <div class="small-red-sizes" aria-label="参考融合红球数量"><button v-for="p in snapshot.fusion.pools" :key="p.red_count" type="button" :data-reference-size="p.red_count" :aria-pressed="count===p.red_count" :class="{selected:count===p.red_count}" @click="count=p.red_count">{{ p.red_count }}红{{ p.red_count===16?' · 上限':'' }}</button></div>
     <template v-if="pool"><div class="balls reference-fused-balls" aria-label="参考融合红球号码"><span v-for="n in sorted(pool.reds)" :key="n" class="ball selected red">{{ pad(n) }}</span></div>
      <p class="muted">{{ count }}个红球候选，同一排名下逐层包含；全6红入池需等待开奖核对。</p>
      <p v-if="scored" class="muted">中红 {{ nullable(pool.red_hits) }}/6 · 全6红 {{ flag(pool.full_red) }}<br>漏红：{{ pool.missed_reds==null?'—':pool.missed_reds.map(pad).join(' ')||'无' }}</p>
      <details class="reference-union"><summary>查看同{{ count }}红的现有小池对照</summary><p class="reference-number-text">{{ sorted(pool.baseline.reds).map(pad).join(' ') }}</p><p v-if="scored" class="muted">现有小池中红 {{ nullable(pool.baseline.red_hits) }}/6 · 全6红 {{ flag(pool.baseline.full_red) }}</p><p class="muted">蓝球沿用现有原榜前2个：{{ snapshot.fusion.blues.map(pad).join(' ') }}。{{ scored?'蓝球入池 '+flag(pool.blue_hit):'蓝球覆盖单独观察。' }}</p></details>
      <button type="button" @click="copy('fusion')">复制{{ count }}红融合池</button>
     </template>
    </article>
   </div><p v-if="message" class="muted reference-message" role="status">{{ message }}</p>
  </div>`
 };
 const archive = {
  components:{ReferenceNumbers:numbers}, props:{record:{type:Object,required:true}},
  setup(){return {time,fileUrl};},
  template:`<section class="card reference-observation" id="independent-reference" aria-label="参考与融合独立观察复盘"><h2>参考5注与融合小池 · 独立观察原档</h2><p class="muted">第 {{ record.issue }} 期 · 保存时间 {{ time(record.published_at_unix) }} · 依据截止 {{ record.data_cutoff }}</p><p class="muted">逐注命中与红球并集覆盖分别核对；待开奖显示“—”。这里只使用首次保存的号码。</p><reference-numbers :snapshot="record.reference_pools" :issue="record.issue" archived></reference-numbers><div class="row reference-actions"><a :href="fileUrl(record.raw_file)">下载参考观察原档JSON</a><a v-if="record.review_file" :href="fileUrl(record.review_file)">下载首次复盘JSON</a></div><details class="reference-evidence"><summary>查看封存依据</summary><p class="muted">原档摘要 <code class="hash">{{ record.archive_sha256 }}</code></p><p class="muted">{{ record.review_persisted?'复盘已保存。':'开奖后核对并保存复盘。' }}本地留档无独立可信时间戳证明。</p></details></section>`
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
    try {const response=await fetch(SSQApp.url('api/reference-pools/freeze'),{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({data_sha256:report.value.data_sha256,target:report.value.target})}), data=await response.json();if(!response.ok)throw Error(data.error||'参考观察封存失败');await load();message.value=data.status==='frozen'?'两个参考观察池已独立封存。':'已保留本期首次参考观察原档。';}
    catch(e){message.value=e.message;}finally{freezing.value=false;}
   }
   function download() {
    if(!report.value?.next)return;
    const value={kind:'reference_observation_preview',actual_recommendation:false,target_issue:report.value.target,data_cutoff:report.value.data_cutoff,input_history_sha256:report.value.data_sha256,reference_pools:report.value.next};
    const url=URL.createObjectURL(new Blob([JSON.stringify(value,null,2)],{type:'application/json'})),link=document.createElement('a');link.href=url;link.download='ssq-reference-preview-'+report.value.target+'.json';link.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
   }
   const reviewUrl=Vue.computed(()=>SSQApp.url('reviews',{issue:report.value?.target,archive:'daily'})+'#independent-reference');
   const cells=Vue.computed(()=>report.value?.actual?.cells || []);
   const archive=Vue.computed(()=>report.value?.next_archive);
   Vue.onMounted(load);
   return {report,error,message,loading,freezing,load,freeze,download,archive,cells,reviewUrl,fileUrl,percent,time};
  },
  template:`<section class="card reference-observation" id="reference-pools" aria-label="参考代码与融合观察池">
   <div class="row between"><h2>参考代码＋现有模型 · 两个观察池</h2><button type="button" :disabled="loading||freezing" @click="load">{{ loading?'读取中…':'刷新观察状态' }}</button></div>
   <p class="muted">保留参考代码原始5注，同时观察16红以内的融合结果。原始5注按实际并集数量统计，两组各自与同数量原榜比较。</p>
   <p v-if="error" class="notice error" role="alert">{{ error }}</p><p v-if="!report&&!error" class="muted" role="status">正在读取参考运行结果与首次观察原档…</p>
   <template v-if="report"><p class="muted">第 {{ report.target }} 期 · 依据截止 {{ report.data_cutoff }}</p>
    <p v-if="report.status==='error'" class="notice error" role="alert">{{ report.reason }}</p><p v-else-if="!report.next" class="notice" role="status">{{ report.reason || '参考代码正在准备本期结果，完成后显示原始5注和融合小池。' }}</p><p v-if="!report.next&&report.job?.started_at_unix" class="muted">本期运算开始于 {{ time(report.job.started_at_unix) }}。{{ report.job.status==='running'?'完成后刷新即可查看两个观察池。':report.job.error||'' }}</p>
    <template v-if="report.next"><p v-if="archive" class="notice success reference-archive-status">本期两个观察池已独立封存，保存于 {{ time(archive.published_at_unix) }}。<a :href="reviewUrl">查看原始号码与复盘</a></p><p v-else class="notice reference-archive-status">当前为本期预览，尚未封存。完成截止前留档后，才计入后续实际观察。</p>
     <reference-numbers :snapshot="report.next" :issue="report.target" :archived="!!archive"></reference-numbers>
     <div class="row reference-actions"><template v-if="archive"><a :href="fileUrl(archive.raw_file)">下载两个观察池原档JSON</a><a :href="reviewUrl">查看独立复盘</a></template><template v-else><button type="button" :disabled="busy||freezing||!report.publication?.can_freeze" @click="freeze">{{ freezing?'正在封存…':'独立封存两个观察池' }}</button><button type="button" @click="download">导出当前预览JSON</button></template></div>
     <p v-if="!archive&&report.publication" class="muted">{{ report.publication.reason }}</p>
    </template><p v-if="message" class="reference-message muted" role="status">{{ message }}</p>
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
