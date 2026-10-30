(function(global){
  'use strict';
  const component={props:{allowGenerate:{type:Boolean,default:true},apiUrl:{type:String,required:true},busy:{type:Boolean,default:false}},emits:['generate'],setup(props){
    const report=Vue.ref(null),error=Vue.ref(''),window=Vue.ref('recent60'),count=Vue.ref(2);
    const choices=[{preset:'bluefocus2',model:'weighted',count:2,label:'原榜2蓝 · 同数量对照'},{preset:'bluehit2',model:'blue_hit_frozen',count:2,label:'命中目标2蓝 · 试验'},{preset:'bluefocus4',model:'weighted',count:4,label:'原榜4蓝 · 分散覆盖'},{preset:'bluehit4',model:'blue_hit_frozen',count:4,label:'命中目标4蓝 · 试验'}];
    const selected=Vue.computed(()=>report.value?.windows[window.value]);
    const cells=Vue.computed(()=>(selected.value?.cells||[]).filter(c=>c.blue_count===count.value));
    const percent=n=>n==null?'—':(n*100).toFixed(1)+'%',pad=n=>String(n).padStart(2,'0');
    function numbers(c){return report.value?.next.find(m=>m.model===c.model)?.ranking.slice(0,c.count)||[];}
    function label(model){return report.value?.next.find(m=>m.model===model)?.label||model;}
    function download(){const url=URL.createObjectURL(new Blob([JSON.stringify(report.value,null,2)],{type:'application/json'})),a=document.createElement('a');a.href=url;a.download='ssq-blue-review-'+report.value.cutoff_issue+'.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);}
    Vue.onMounted(async()=>{try{const response=await fetch(props.apiUrl),data=await response.json();if(!response.ok)throw Error(data.error||'蓝球研究读取失败');report.value=data;}catch(e){error.value=e.message;}});
    return {report,error,window,count,choices,selected,cells,percent,pad,numbers,label,download};
  },template:`<section class="card blue-focus" aria-label="蓝球命中优化">
    <h2>蓝球优先 · 相同8注16元</h2>
    <p class="muted">原加权16红排相同8组红球，每注配1个蓝球。2蓝每蓝4注；4蓝每蓝2注，总注数和预算相同。增加不同蓝球提高覆盖范围，同时减少每个蓝球命中时的注数。</p>
    <p v-if="error" class="notice error" role="alert">{{ error }}</p><p v-if="!report&&!error" role="status" class="muted">正在比较同数量的蓝球命中与同预算分散覆盖…</p>
    <template v-if="report">
      <p class="muted">第 {{ report.target_issue }} 期预览，依据截止 {{ report.cutoff_issue }}；点击生成后按完整票面独立留档。已有每日原档不追加新方案。</p>
      <div class="blue-choice-grid"><article v-for="c in choices" :key="c.preset" class="blue-choice" :data-blue-preset="c.preset"><h3>{{ c.label }}</h3><div class="balls"><span v-for="n in numbers(c)" :key="n" class="ball blue">{{ pad(n) }}</span></div><p class="muted">{{ c.count }}个不同蓝球 · 8注16元</p><button v-if="allowGenerate" type="button" :disabled="busy" @click="$emit('generate',{preset:c.preset,issue:report.target_issue,data_sha256:report.data_sha256})">生成{{ c.count }}蓝{{ c.model==='weighted'?'原榜':'试验' }}8注</button></article></div>
      <details class="blue-evidence"><summary>查看蓝球同数量比较与实际留档</summary>
        <p class="muted">命中目标试验直接按此前专家前2蓝命中拟合，25111起权重冻结。改善尚不稳定，原协议保持原蓝榜；新增比较属于探索性回放。</p>
        <div class="row"><label class="field">回放窗口<select data-blue-window v-model="window"><option value="recent30">近30期</option><option value="recent60">近60期</option><option value="recent156">近156期</option><option value="confirmation_exploratory">原确认窗 · 探索性</option><option value="tracking">确认窗后跟踪</option><option value="development">开发期</option><option value="all_oos">全部回放</option></select></label><label class="field">相同蓝球数量<select data-blue-count v-model.number="count"><option v-for="n in [1,2,3,4,5,6,7,8]" :key="n" :value="n">{{ n }}蓝</option></select></label><button type="button" @click="download">导出蓝球研究与实际统计</button></div>
        <p v-if="selected" class="muted">{{ selected.from }}—{{ selected.to }}，{{ selected.n }}期；同数量均匀覆盖基准 {{ percent(count/16) }}。下表为蓝球至少命中一次的期数，不是中奖注数或收益率。</p>
        <div class="scroll"><table class="blue-table"><thead><tr><th>蓝球排序</th><th>命中期数 / 比例</th><th>相对原榜补中 / 丢失</th></tr></thead><tbody><tr v-for="c in cells" :key="c.model" :data-blue-model="c.model"><td>{{ label(c.model) }}</td><td>{{ c.hit_periods }}/{{ c.n }} · {{ percent(c.hit_rate) }}</td><td>补中{{ c.paired_gains }}期 / 丢失{{ c.paired_losses }}期</td></tr></tbody></table></div>
        <h3>每日前瞻留档 · 四方案共同期次</h3><p class="muted">{{ report.actual.note }}</p><p v-if="report.actual.errors.length" class="notice error">部分原档未计入：{{ report.actual.errors.join('；') }}</p>
        <div class="blue-actual"><p v-for="c in report.actual.cells" :key="c.preset" :data-blue-actual="c.preset">{{ choices.find(v=>v.preset===c.preset)?.label }}：{{ c.n?c.hit_periods+'/'+c.n+'期命中 · '+percent(c.hit_rate):'暂无已开奖样本' }}；待开奖{{ c.pending }}期，缺少共同方案{{ c.excluded_missing_pair }}期。</p></div>
        <p class="muted">{{ report.note }}</p>
      </details>
    </template>
  </section>`};
  global.SSQBlueFocus={component};
})(typeof window==='undefined'?globalThis:window);
