(function(global){
 'use strict';
 const component={props:{apiUrl:{type:String,required:true},busy:{type:Boolean,default:false}},emits:['generate'],setup(){
  const report=Vue.ref(null),error=Vue.ref(''),count=Vue.ref(16),model=Vue.ref('small_recent'),window=Vue.ref('recent60'),message=Vue.ref('');
  const selected=Vue.computed(()=>report.value?.next.models.find(m=>m.model===model.value));
  const pool=Vue.computed(()=>selected.value?.pools.find(p=>p.red_count===count.value));
  const rows=Vue.computed(()=>report.value?.windows[window.value]?.cells.filter(c=>c.red_count===count.value)||[]);
  const actual=Vue.computed(()=>report.value?.actual.cells.filter(c=>c.red_count===count.value)||[]);
  const pad=n=>String(n).padStart(2,'0'),percent=n=>n==null?'—':(n*100).toFixed(2)+'%',decimal=n=>n==null?'—':n.toFixed(2);
  function download(){const value={kind:'small_red_pool_preview',actual_recommendation:false,issue:report.value.target_issue,cutoff_issue:report.value.cutoff_issue,input_history_sha256:report.value.data_sha256,model:model.value,red_count:count.value,reds:pool.value.reds,policy:model.value==='weighted'?{model:'weighted',method:'original_weighted_fusion'}:report.value.next.policy,source_sha256:report.value.source_sha256};const u=URL.createObjectURL(new Blob([JSON.stringify(value,null,2)],{type:'application/json'})),a=document.createElement('a');a.href=u;a.download='ssq-small-red-'+value.issue+'-'+count.value+'.json';a.click();setTimeout(()=>URL.revokeObjectURL(u),1000);}
  async function copy(){const text=`双色球 ${report.value.target_issue} 期 · ${selected.value.label}\n${count.value}红：${pool.value.reds.slice().sort((a,b)=>a-b).map(pad).join(' ')}\n依据截止 ${report.value.cutoff_issue}；研究预览，未封存，不是单注票面。`;try{if(navigator.clipboard)await navigator.clipboard.writeText(text);else{const a=document.createElement('textarea');a.value=text;document.body.appendChild(a);a.select();const ok=document.execCommand('copy');a.remove();if(!ok)throw Error();}message.value='小池号码已复制。';}catch{message.value='复制未成功，请手动选择号码。';}}
  return{report,error,count,model,window,message,selected,pool,rows,actual,pad,percent,decimal,download,copy};
 },async mounted(){try{const r=await fetch(this.apiUrl),d=await r.json();if(!r.ok)throw Error(d.error||'小红球池研究读取失败');this.report=d;}catch(e){this.error=e.message;}},template:`<section class="card small-red-focus" aria-label="16红以内全6红覆盖">
  <h2>16红以内 · 全6红覆盖</h2><p class="muted">目标是让全部6个开奖红球进入同一号码池；不以平均中红替代全包。提供12／14／15／16红，最多16个，不通过合并多池扩大数量。</p>
  <p v-if="error" class="notice error" role="alert">{{ error }}</p><p v-if="!report&&!error" class="muted" role="status">正在核对同数量小池的全6红表现…</p>
  <template v-if="report"><p class="muted">第 {{ report.target_issue }} 期研究预览 · 依据截止 {{ report.cutoff_issue }}</p>
   <label class="field" style="max-width:430px">小池排序<select data-small-model v-model="model"><option v-for="m in report.next.models" :key="m.model" :value="m.model">{{ m.label }}</option></select></label>
   <div class="small-red-sizes"><button v-for="n in report.red_sizes" :key="n" type="button" :data-small-size="n" :aria-pressed="count===n" :class="{selected:count===n}" @click="count=n">{{ n }}红{{ n===16?' · 上限':'' }}</button></div>
   <div class="balls" aria-label="小池红球号码"><span v-for="n in pool.reds.slice().sort((a,b)=>a-b)" :key="n" class="ball red">{{ pad(n) }}</span></div>
   <p v-if="report.actual.next_archive.daily_exists&&!report.actual.next_archive.has_small_red_pools" class="notice">本期每日原档已存在，未保存这个小池扩展；以上仍为预览，不覆盖原档。可选择生成完整8注，作为新的个人方案独立留档。</p>
   <p v-else-if="report.actual.next_archive.has_small_red_pools" class="notice">本期已保存小池原档，号码以首次原档为准；下方实际统计只读取原始号码。</p>
   <p v-else class="muted">当前为研究预览；后续新期首次每日留档会同时保存原榜与试验的四个小池。</p>
   <div class="row"><button type="button" @click="copy">复制{{ count }}红小池</button><button type="button" @click="download">导出小池JSON</button><button type="button" :disabled="busy" @click="$emit('generate',{model,red_count:count,issue:report.target_issue,data_sha256:report.data_sha256})">用这组红球生成8注16元</button></div><p v-if="message" class="muted" role="status">{{ message }}</p>
   <details class="small-red-evidence"><summary>查看同数量全包次数与实际原档</summary><p class="muted">{{ report.next.policy.note }}</p>
   <label class="field" style="max-width:300px">比较窗口<select data-small-window v-model="window"><option value="recent60">近60期</option><option value="recent30">近30期</option><option value="recent156">近156期</option><option value="confirmation_exploratory">原确认窗 · 探索性</option><option value="tracking">确认窗后跟踪</option><option value="development">开发期</option><option value="all_oos">全部因果回放</option></select></label>
   <p class="muted">同为{{ count }}红的均匀全包基准 {{ percent(rows[0]?.uniform_full_red_rate) }}。全包率是回放结果，不是下一期覆盖概率。</p>
   <div class="scroll"><table class="small-red-table"><thead><tr><th>相同{{ count }}红</th><th>全6红次数 / 比例</th><th>补中 / 丢失</th><th>平均漏红</th></tr></thead><tbody><tr v-for="r in rows" :key="r.model" :data-small-row="r.model"><td>{{ report.next.models.find(m=>m.model===r.model)?.label }}</td><td>{{ r.full_red_periods }}/{{ r.n }} · {{ percent(r.full_red_rate) }}</td><td>{{ r.paired_gains }} / {{ r.paired_losses }}期</td><td>{{ decimal(r.mean_missed_reds) }}</td></tr></tbody></table></div>
   <h3>同批小池原档 · 实际覆盖</h3><p class="muted">{{ report.actual.note }}</p><p v-if="report.actual.errors.length" class="notice error">部分原档未计入：{{ report.actual.errors.join('；') }}</p><p v-for="r in actual" :key="r.model" :data-small-actual="r.model" class="muted">{{ report.next.models.find(m=>m.model===r.model)?.label }}：{{ r.n?r.full_red_periods+'/'+r.n+'期全包':'暂无已开奖样本' }}；待开奖{{ r.pending }}期，缺少成对原池{{ r.excluded_missing_pair }}期。</p><p class="muted">{{ report.note }}</p>
   </details>
  </template></section>`};global.SSQSmallRed={component};
})(typeof window==='undefined'?globalThis:window);
