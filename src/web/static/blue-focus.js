(function(global){
  'use strict';
  const blueScoreV2={props:{apiUrl:{type:String,required:true}},setup(props){
    const report=Vue.ref(null),error=Vue.ref(''),busy=Vue.ref(false),message=Vue.ref(''),window=Vue.ref('all_held_out_exploratory');let request=0;
    const labels={frequency:'频率',ewma:'近期加权',log_gap:'间隔方向',previous_blue_correction:'上期蓝球修正',markov:'转移方向'};
    const current=Vue.computed(()=>report.value?.current_observation),prediction=Vue.computed(()=>current.value?.prediction||report.value?.prediction),selected=Vue.computed(()=>report.value?.windows[window.value]),cell=Vue.computed(()=>selected.value?.cells.find(c=>c.blue_count===2));
    const pad=n=>String(n).padStart(2,'0'),pct=n=>n==null?'—':(n*100).toFixed(2)+'%',num=n=>n==null?'—':Number(n).toFixed(6),link=p=>new URL(p,SSQApp.base).href;
    async function load(){const id=++request;error.value='';try{const r=await fetch(props.apiUrl),v=await r.json();if(id!==request)return;if(!r.ok)throw Error(v.error||'蓝球v2读取失败');report.value=v;}catch(e){if(id===request)error.value=e.message;}}
    async function freeze(){if(busy.value||!report.value)return;busy.value=true;message.value='';try{const r=await fetch(props.apiUrl+'/freeze',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({target:report.value.target_issue,data_sha256:report.value.data_sha256})}),v=await r.json();if(!r.ok)throw Error(v.error||'蓝球v2封存失败');message.value=v.status==='already_frozen'?'已保留本期首次蓝球分布。':'已首次封存蓝球分布，等待开奖。';await load();}catch(e){message.value=e.message;}finally{busy.value=false;}}
    Vue.onMounted(load);Vue.watch(()=>props.apiUrl,()=>{report.value=null;load();});return{report,error,busy,message,window,current,prediction,selected,cell,labels,pad,pct,num,link,freeze};
  },template:`<section class="blue-evidence" data-blue-v2 aria-label="蓝球评分v2">
   <h3>蓝球评分v2 · 去重复联合拟合</h3><p v-if="error" class="notice error" role="alert">{{error}}</p><p v-if="!report&&!error" class="muted" role="status">正在核对蓝球v2评分…</p>
   <template v-if="report&&prediction"><p class="muted">第{{report.target_issue}}期 · {{current?'来自首次封存分布':'尚未封存的研究预览'}}。与原榜相同，下面比较固定2蓝。</p>
    <div class="blue-choice-grid"><article class="blue-choice" data-blue-v2-top2><h4>v2前2蓝 · 研究前瞻</h4><div class="balls"><span v-for="n in prediction.ranking.slice(0,2)" :key="n" class="ball selected blue">{{pad(n)}}</span></div></article><article class="blue-choice" data-blue-v2-original><h4>同输入原榜2蓝</h4><div class="balls"><span v-for="n in prediction.original_ranking.slice(0,2)" :key="n" class="ball selected blue">{{pad(n)}}</span></div></article></div>
    <p class="notice" data-blue-v2-adoption>{{report.reason}}</p><p class="muted">将遗漏／热度合并为一个方向，反重号只保留相对频率的修正；联合拟合5个方向，允许系数为0或反向，避免重复加分和弱专家最低权重。</p>
    <p class="muted" data-blue-v2-prospective>真实前瞻已开奖{{report.prospective.n}}期，待开奖{{report.prospective.pending}}期；v2同2蓝命中{{report.prospective.n?report.prospective.v2_top2_hits+'/'+report.prospective.n:'—'}}，原榜{{report.prospective.n?report.prospective.original_top2_hits+'/'+report.prospective.n:'—'}}。只观察蓝球分布，不生成新票面或增加投注预算。</p>
    <p v-if="current" class="muted" data-blue-v2-sealed><a :href="link(current.raw_url)">下载本期首次蓝球分布</a> · 输入截止{{current.data_cutoff}}，保存时间{{new Date(current.published_at_unix*1000).toLocaleString('zh-CN',{timeZone:'Asia/Shanghai',hour12:false})}}（北京时间）。</p><button v-else type="button" :disabled="busy||!report.publication.can_observe" @click="freeze">{{busy?'封存中…':'封存本期蓝球分布前瞻'}}</button><p v-if="message" class="muted" role="status">{{message}}</p><p v-if="report.prospective.errors.length" class="notice error">{{report.prospective.errors.join('；')}}</p>
    <details data-blue-v2-evidence><summary>同2蓝验证、完整排名与系数</summary><label class="field">验证窗口<select v-model="window" data-blue-v2-window><option value="all_held_out_exploratory">全部边界后探索（{{report.windows.all_held_out_exploratory.n}}期）</option><option value="validation">预留312期</option><option value="confirmation_exploratory">确认156期 · 探索</option><option value="tracking">确认后跟踪</option><option value="recent60">近60期 · 描述</option></select></label>
     <p v-if="cell" class="muted" data-blue-v2-hit-comparison>相同2蓝、{{cell.n}}期：原榜命中{{cell.original_hits}}期，v2命中{{cell.v2_hits}}期；相对原榜补中{{cell.paired_gains}}、丢失{{cell.paired_losses}}。均匀覆盖期望{{cell.uniform_expected_hits.toFixed(2)}}期。</p><p v-if="selected" class="muted">v2相对均匀的蓝球平均对数评分差{{num(selected.v2_log_excess_vs_uniform?.mean)}}；原榜{{num(selected.original_log_excess_vs_uniform?.mean)}}。历史已被查看，不作为新的独立前瞻，也不等于头奖命中率。</p>
     <p v-if="report.latest_exploratory" class="muted" data-blue-v2-latest>第{{report.latest_exploratory.issue}}期蓝{{pad(report.latest_exploratory.winning_blue)}}：原榜第{{report.latest_exploratory.original_rank}}，v2研究第{{report.latest_exploratory.v2_rank}}。这是同输入的事后研究，未补记实际成绩。</p>
     <div class="scroll"><table class="blue-table"><thead><tr><th>v2名次</th><th>蓝球</th><th>模型分布</th><th>原榜名次</th></tr></thead><tbody><tr v-for="(n,i) in prediction.ranking" :key="n" data-blue-v2-ranking><td>{{i+1}}</td><td>{{pad(n)}}</td><td>{{pct(prediction.probabilities[n-1])}}</td><td>{{prediction.original_ranking.indexOf(n)+1}}</td></tr></tbody></table></div><p class="muted">分布已归一化，尚未验证为真实中奖概率；数字顺序用于复现同分，不代表预测证据。</p>
     <div class="scroll"><table class="blue-table"><thead><tr><th>独立方向</th><th>拟合系数</th></tr></thead><tbody><tr v-for="(a,i) in prediction.policy.axes" :key="a" data-blue-v2-axis><td>{{labels[a]}}</td><td>{{num(prediction.coefficients[i])}}</td></tr></tbody></table></div><p class="muted">系数允许反向，全部为0时回到均匀。只用23100及以前{{prediction.training_n}}条样本拟合，正则强度固定40；此后冻结系数，仅更新开奖前特征。求解剩余目标差距上界{{prediction.solver.objective_gap_upper_bound.toExponential(2)}}。</p><a :href="apiUrl">核对蓝球v2完整研究JSON</a>
    </details>
   </template>
  </section>`};
  const component={components:{BlueDeep:global.SSQBlueDeep.component,BlueScoreV2:blueScoreV2},props:{allowGenerate:{type:Boolean,default:true},apiUrl:{type:String,required:true},busy:{type:Boolean,default:false}},emits:['generate'],setup(props){
    const coverage=Vue.ref(null),coverageError=Vue.ref(''),report=Vue.ref(null),error=Vue.ref(''),window=Vue.ref('recent60'),count=Vue.ref(2);
    const choices=[{preset:'bluefocus8',model:'weighted',count:8,label:'小奖分散覆盖 · 8蓝8注'},{preset:'bluefocus2',model:'weighted',count:2,label:'原榜2蓝 · 同数量对照'},{preset:'bluehit2',model:'blue_hit_frozen',count:2,label:'命中目标2蓝 · 试验'},{preset:'bluefocus4',model:'weighted',count:4,label:'原榜4蓝 · 分散覆盖'},{preset:'bluehit4',model:'blue_hit_frozen',count:4,label:'命中目标4蓝 · 试验'}];
    const selected=Vue.computed(()=>report.value?.windows[window.value]);
    const cells=Vue.computed(()=>(selected.value?.cells||[]).filter(c=>c.blue_count===count.value));
    const percent=n=>n==null?'—':(n*100).toFixed(1)+'%',pad=n=>String(n).padStart(2,'0');
    function numbers(c){return report.value?.next.find(m=>m.model===c.model)?.ranking.slice(0,c.count)||[];}
    function label(model){return report.value?.next.find(m=>m.model===model)?.label||model;}
    function download(){const url=URL.createObjectURL(new Blob([JSON.stringify(report.value,null,2)],{type:'application/json'})),a=document.createElement('a');a.href=url;a.download='ssq-blue-review-'+report.value.cutoff_issue+'.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);}
    const budgetCells=Vue.computed(()=>coverage.value?.windows?.[window.value]?.cells||[]);
    const reviewHref=Vue.computed(()=>coverage.value?.current?.url?new URL(coverage.value.current.url,SSQApp.base).href:'');
    Vue.onMounted(async()=>{try{const response=await fetch(props.apiUrl.replace(/api\/blue(?=$|\?)/,'api/prize-coverage'));const data=await response.json();if(!response.ok)throw Error(data.error||'小奖覆盖读取失败');coverage.value=data;}catch(e){coverageError.value=e.message;}});
    Vue.onMounted(async()=>{try{const response=await fetch(props.apiUrl),data=await response.json();if(!response.ok)throw Error(data.error||'蓝球研究读取失败');report.value=data;}catch(e){error.value=e.message;}});
    return {coverage,coverageError,budgetCells,reviewHref,report,error,window,count,choices,selected,cells,percent,pad,numbers,label,download};
  },template:`<section class="card blue-focus" aria-label="蓝球命中优化">
    <h2>蓝球优先 · 相同8注16元</h2>
    <p class="muted">原加权16红排相同8组红球，每注配1个蓝球。2蓝每蓝4注；4蓝每蓝2注；8蓝每蓝1注，总注数和预算相同。增加不同蓝球提高覆盖范围，同时减少每个蓝球命中时的注数。</p>
    <p v-if="error" class="notice error" role="alert">{{ error }}</p><p v-if="!report&&!error" role="status" class="muted">正在比较同数量的蓝球命中与同预算分散覆盖…</p>
    <template v-if="report">
      <p class="muted">第 {{ report.target_issue }} 期预览，依据截止 {{ report.cutoff_issue }}；点击生成后按完整票面独立留档。已有每日原档不追加新方案。</p>
      <blue-score-v2 :api-url="apiUrl+'/score-v2'" />
      <blue-deep :api-url="apiUrl+'/deep'" />
      <div class="blue-choice-grid"><article v-for="c in choices" :key="c.preset" class="blue-choice" :data-blue-preset="c.preset"><h3>{{ c.label }}</h3><div class="balls"><span v-for="n in numbers(c)" :key="n" class="ball selected blue">{{ pad(n) }}</span></div><p class="muted">{{ c.count }}个不同蓝球 · 8注16元</p><button v-if="allowGenerate" type="button" :disabled="busy" @click="$emit('generate',{preset:c.preset,issue:report.target_issue,data_sha256:report.data_sha256})">生成{{ c.count }}蓝{{ c.model==='weighted'?'原榜':'试验' }}8注</button></article></div>
      <p class="notice">小奖分散覆盖优先减少蓝球重复。8蓝在均匀开奖假设下覆盖8/16个蓝球结果；不是模型预测概率。只中普通六等奖为5元，低于16元成本，不保证中奖或回本。</p>
      <p v-if="coverageError" class="notice error" role="alert">{{coverageError}}</p>
      <details v-if="coverage" class="blue-evidence" data-prize-coverage open><summary>小奖覆盖复盘 · 同样8注16元</summary>
       <p class="muted">{{coverage.current.status==='frozen'?'本期8蓝票组已首次留档':'本期8蓝票组尚未留档'}}；已开奖{{coverage.actual.n}}期，待开奖{{coverage.actual.pending}}期，旧期缺档{{coverage.actual.excluded_missing}}期。<a v-if="reviewHref" :href="reviewHref">查看本期8蓝完整原档与复盘</a></p>
       <label class="field">相同预算回放窗口<select data-prize-window v-model="window"><option value="recent30">近30期</option><option value="recent60">近60期</option><option value="recent156">近156期</option><option value="all_oos">全部回放</option><option value="tracking">确认窗后跟踪</option></select></label>
       <p class="muted">每期只用此前数据，三种方案使用完全相同的8组红球。历史回放单列；固定奖金不含浮动奖和待核特别奖。</p>
       <div class="scroll"><table class="blue-table"><thead><tr><th>蓝球分配</th><th>至少一注常规中奖</th><th>中奖注数</th><th>固定奖金达16元期数</th><th>已知固定奖金累计</th></tr></thead><tbody><tr v-for="r in budgetCells" :key="r.blue_count" :data-prize-blue-count="r.blue_count"><td>{{r.blue_count}}蓝 · 8注16元</td><td>{{r.prize_periods}}/{{r.n}} · {{percent(r.prize_rate)}}</td><td>{{r.prize_bets}}</td><td>{{r.fixed_at_least_cost_periods}}/{{r.n}}</td><td>{{r.fixed_prize_total}}元</td></tr></tbody></table></div>
       <h3>最近真实留档为什么没中奖</h3><p class="muted">{{coverage.audit.note}}</p><div class="scroll"><table class="blue-table"><thead><tr><th>期号</th><th>常规中奖组数</th><th>蓝球是否入票</th><th>最佳单注中红</th></tr></thead><tbody><tr v-for="r in coverage.audit.periods.slice(0,3)" :key="r.issue" :data-prize-audit-issue="r.issue"><td>{{r.issue}}</td><td>{{r.winning_groups}}/{{r.group_count}}</td><td>{{r.any_blue_covered?'已入至少一组':'所列票组全部漏蓝'}}</td><td>{{r.max_single_red}}/6</td></tr></tbody></table></div><p v-if="coverage.audit.errors.length||coverage.actual.errors.length" class="notice error">{{[...coverage.audit.errors,...coverage.actual.errors].join('；')}}</p>
      </details>
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
