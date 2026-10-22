(function(global){
  'use strict';
  const labels={weighted:'协议加权融合',equal:'等权融合',uniform:'均匀红球对照',repeat_calibrated:'重号偏置校准 · 试验',conditional_gap:'条件频率 · 试验',rank_consensus:'名次共识 · 试验',scenario_cover:'全6情景覆盖 · 试验',robust_cover:'三窗稳健覆盖 · 试验'};
  function modelsFrom(snapshot){
    const root=snapshot?.next?.weighted_pools;
    if(!root)return [];
    return [{...root,model:'weighted'},...(root.comparison_models||[]),...(root.coverage_model?[root.coverage_model]:[]),...(root.algorithm_models||[])];
  }
  function name(model){return model.policy?.label||labels[model.model]||model.model;}
  function fromArchive(view){
    const p=view?.provenance,root=view?.number_pools;
    if(!root?.observation_tiers?.length||!p?.input_history_sha256||!p.sha256)return null;
    return {suggested_target:view.issue,cutoff_issue:p.data_cutoff,cutoff_date:p.data_cutoff_date,data_sha256:p.input_history_sha256,source_sha256:p.source_sha256,archive_sha256:p.sha256,archived:true,observation_tiers:root.observation_tiers,next:{weighted_pools:root}};
  }
  function numbers(values,count,maximum){
    if(!Array.isArray(values)||values.length!==count||values.some(n=>!Number.isInteger(n)||n<1||n>maximum)||new Set(values).size!==values.length)throw Error('原号码池的号码或数量无效，请刷新后重试。');
    return values.slice().sort((a,b)=>a-b);
  }
  function intersect(snapshot,selections){
    if(!snapshot||!/^\d{5}$/.test(snapshot.suggested_target)||!/^\d{5}$/.test(snapshot.cutoff_issue)||!/^[a-f0-9]{64}$/.test(snapshot.data_sha256))throw Error('号码池缺少期号或输入历史摘要，请刷新后重试。');
    if(!Array.isArray(selections)||selections.length<2||selections.length>8)throw Error('请选择2至8个号码池。');
    const models=modelsFrom(snapshot),seen=new Set();
    const sources=selections.map(selection=>{
      const model=models.find(m=>m.model===selection.model),redCount=Number(selection.red_count),blueCount=Number(selection.blue_count);
      if(!model||(model.target_issue&&model.target_issue!==snapshot.suggested_target)||(model.input_history_sha256&&model.input_history_sha256!==snapshot.data_sha256))throw Error('所选号码池的期号或输入历史不一致，请刷新后重试。');
      const pool=model.pools?.find(p=>p.group==='observation'&&p.red_count===redCount&&p.blue_count===blueCount);
      if(!pool||pool.model!==model.model)throw Error('所选号码池不存在，请重新选择模型和数量。');
      const key=`${model.model}:${redCount}:${blueCount}`;
      if(seen.has(key))throw Error('重复选择了同一个号码池，请更换模型或数量。');
      seen.add(key);
      return {model:model.model,label:name(model),red_count:redCount,blue_count:blueCount,reds:numbers(pool.reds,redCount,33),blues:numbers(pool.blues,blueCount,16)};
    });
    const common=color=>sources[0][color].filter(n=>sources.every(source=>source[color].includes(n)));
    const reds=common('reds'),blues=common('blues');
    return {kind:snapshot.archived?'archived_pool_intersection_diagnostic':'pool_intersection_preview',version:1,actual_recommendation:false,archive_sha256:snapshot.archive_sha256,issue:snapshot.suggested_target,cutoff_issue:snapshot.cutoff_issue,cutoff_date:snapshot.cutoff_date,input_history_sha256:snapshot.data_sha256,source_sha256:snapshot.source_sha256,sources,reds,blues,red_count:reds.length,blue_count:blues.length,can_form_single:reds.length>=6&&blues.length>=1,note:'只保留所有所选原池共同的号码；不自动补号。交集是原池的子集，不代表全覆盖能力提高。'};
  }
  function reviewIntersection(result,draw){
    if(!draw)return null;
    if(draw.issue!==result.issue)throw Error('开奖期号与原档池不一致，禁止混期复盘。');
    const wins=numbers(draw.red,6,33);numbers([draw.blue],1,16);
    const union=new Set(result.sources.flatMap(s=>s.reds));
    return {kind:'post_draw_derived_intersection_diagnostic',is_original_recommendation:false,
      hit_reds:wins.filter(n=>result.reds.includes(n)),missed_reds:wins.filter(n=>!result.reds.includes(n)),
      red_hits:wins.filter(n=>result.reds.includes(n)).length,full_red:wins.every(n=>result.reds.includes(n)),blue_hit:result.blues.includes(draw.blue),
      full_draw:wins.every(n=>result.reds.includes(n))&&result.blues.includes(draw.blue),
      removed_winning_reds:wins.filter(n=>union.has(n)&&!result.reds.includes(n)),missing_in_all_sources:wins.filter(n=>!union.has(n)),
      sources:result.sources.map(s=>({model:s.model,label:s.label,red_count:s.red_count,blue_count:s.blue_count,
        red_hits:wins.filter(n=>s.reds.includes(n)).length,blue_hit:s.blues.includes(draw.blue),
        removed_winning_reds:wins.filter(n=>s.reds.includes(n)&&!result.reds.includes(n))}))};
  }
  const pad=n=>String(n).padStart(2,'0');
  function asText(result){
    return [`【双色球 ${result.issue} 期 · ${result.archive_sha256?'原档池交集诊断':'号码池交集'}】`,`依据截止 ${result.cutoff_issue}（${result.cutoff_date||'日期未留存'}）`,...result.sources.map((s,i)=>`来源${i+1}：${s.label} · ${s.red_count}红＋${s.blue_count}蓝`),`${result.red_count}红交集：${result.reds.map(pad).join(' ')||'无'}`,`${result.blue_count}蓝交集：${result.blues.map(pad).join(' ')||'无'}`,...(result.review?[`开奖后诊断：中${result.review.red_hits}/6红，蓝球${result.review.blue_hit?'入池':'未入池'}`,`被交集剔除的开奖红球：${result.review.removed_winning_reds.map(pad).join(' ')||'无'}`]:[]),result.note,result.archive_sha256?'根据原档池推导，不是当时已冻结的交集推荐。':'交集观察结果，非单注投注票面。'].join('\n');
  }
  let instance=0;
  const component={
    props:{apiUrl:{type:String,required:true},snapshot:{type:Object,default:null},archived:{type:Boolean,default:false},draw:{type:Object,default:null}},
    setup(props){
      const {ref,computed,watch}=Vue,prefix='pool-intersection-'+(++instance),data=ref(null),rows=ref([]),busy=ref(false),loadError=ref(''),message=ref('');let nextRow=0;
      const models=computed(()=>modelsFrom(data.value)),sizes=computed(()=>(data.value?.observation_tiers||[]).map(t=>t.red_count));
      const calculation=computed(()=>{if(!data.value)return {result:null,error:''};try{const r=intersect(data.value,rows.value);return {result:props.archived?{...r,review:reviewIntersection(r,props.draw)}:r,error:''};}catch(e){return {result:null,error:e.message};}});
      const result=computed(()=>calculation.value.result),analysis=computed(()=>result.value?.review),selectionError=computed(()=>calculation.value.error);
      function showResult(){const el=document.getElementById(prefix+'-result');el?.focus({preventScroll:true});el?.scrollIntoView({block:'start',behavior:'instant'});}
      function reset(){
        if(!models.value.length)return;
        const first=models.value.find(m=>m.model==='weighted')||models.value[0],second=models.value.find(m=>m.model==='rank_consensus')||models.value.find(m=>m.model!==first.model)||first;
        rows.value=[{id:++nextRow,model:first.model,red_count:16,blue_count:2},{id:++nextRow,model:second.model,red_count:second===first?18:16,blue_count:2}];message.value='';
      }
      function accept(snapshot){if(!modelsFrom(snapshot).length)throw Error('没有可供取交集的号码池。');data.value=snapshot;message.value='';loadError.value='';if(!rows.value.length)reset();}
      async function load(){
        if(busy.value)return;busy.value=true;loadError.value='';message.value='';
        try{if(props.archived){if(!props.snapshot)throw Error('本档案缺少可核对的原号码池。');accept(props.snapshot);}else{const response=await fetch(props.apiUrl),snapshot=await response.json();if(!response.ok)throw Error(snapshot.error||'读取号码池失败');accept(snapshot);}}catch(e){loadError.value=e.message;}finally{busy.value=false;}
      }
      function opened(event){if(event.target.open&&!data.value&&!busy.value){if(props.snapshot){try{accept(props.snapshot);}catch(e){loadError.value=e.message;}}else load();}}
      watch(()=>props.snapshot,snapshot=>{if(snapshot&&data.value){try{accept(snapshot);}catch(e){loadError.value=e.message;}}});
      function add(){
        if(rows.value.length>=8)return;
        const unused=models.value.find(m=>!rows.value.some(r=>r.model===m.model&&r.red_count===16&&r.blue_count===2));
        const choices=models.value.flatMap(m=>sizes.value.map(size=>({model:m.model,red_count:size,blue_count:2})));
        const next=unused?{model:unused.model,red_count:16,blue_count:2}:choices.find(c=>!rows.value.some(r=>r.model===c.model&&r.red_count===c.red_count&&r.blue_count===c.blue_count));
        if(next)rows.value.push({id:++nextRow,...next});message.value='';
      }
      function remove(id){if(rows.value.length>2)rows.value=rows.value.filter(row=>row.id!==id);message.value='';}
      async function copy(){
        const selected=result.value;if(!selected||busy.value||loadError.value)return;
        const text=asText(selected);
        try{
          let copied=false;
          if(navigator.clipboard?.writeText){try{await navigator.clipboard.writeText(text);copied=true;}catch{}}
          if(!copied){const input=document.createElement('textarea');input.value=text;document.body.appendChild(input);try{input.select();if(!document.execCommand('copy'))throw Error('浏览器未允许复制，请手动选择交集号码。');}finally{input.remove();}}
          message.value=`已复制${selected.red_count}红＋${selected.blue_count}蓝交集及来源。`;
        }catch(e){message.value=e.message||'复制失败，请手动选择交集号码。';}
      }
      function exportJson(){
        if(!result.value||busy.value||loadError.value)return;
        const selected=result.value,url=URL.createObjectURL(new Blob([JSON.stringify(selected,null,2)],{type:'application/json'})),link=document.createElement('a');link.href=url;link.download=`ssq-pool-intersection-${selected.issue}.json`;link.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
      }
      return {prefix,data,rows,busy,loadError,message,models,sizes,result,analysis,selectionError,name,pad,opened,load,reset,add,remove,copy,exportJson,showResult};
    },
    template:`<details class="card pool-intersection" data-pool-intersection @toggle="opened">
      <summary><strong>{{ archived?'原档池取交集 · 覆盖损失诊断':'号码池取交集' }}</strong><span class="muted"> · 选择多套号码池，只保留共同号码</span></summary>
      <div class="pool-intersection-body">
        <div class="row between"><p class="muted" v-if="data">第 {{ data.suggested_target }} 期 · 依据截止 {{ data.cutoff_issue }}（{{ data.cutoff_date }}）的同一份数据</p><button type="button" @click="load" :disabled="busy">{{ busy?'读取中…':archived?'重新核对原池':'刷新号码池' }}</button></div>
        <p v-if="archived" class="notice">只使用本期原档实际保存的号码池；交集是当前推导的诊断，不补记为当时推荐。{{ draw?'命中与剔除损失按本期开奖计算。':'待开奖，命中与剔除损失暂不计算。' }}</p>
        <p v-if="busy" class="muted" role="status">正在读取原号码池…</p><p v-if="loadError" class="notice error" role="alert">{{ loadError }}</p>
        <template v-if="data">
          <p class="muted">可组合2至8个不同的号码池，分别计算红球与蓝球的交集。</p>
          <div v-if="result" class="intersection-overview row between" data-intersection-overview><span role="status">{{ rows.length }}个来源 · 交集 <b>{{ result.red_count }}红＋{{ result.blue_count }}蓝</b> · 比最小来源池少{{ Math.min(...result.sources.map(s=>s.red_count))-result.red_count }}红</span><button type="button" @click="showResult" :aria-controls="prefix+'-result'">查看交集号码</button></div>
          <div v-for="(row,i) in rows" :key="row.id" class="pool-intersection-row" data-intersection-row @change="message=''">
            <label class="field" :for="prefix+'-model-'+row.id">号码池 {{ i+1 }} · 模型<select :id="prefix+'-model-'+row.id" v-model="row.model" data-intersection-model :disabled="busy"><option v-for="m in models" :key="m.model" :value="m.model">{{ name(m) }}</option></select></label>
            <label class="field" :for="prefix+'-red-'+row.id">红球数量<select :id="prefix+'-red-'+row.id" v-model.number="row.red_count" data-intersection-red :disabled="busy"><option v-for="n in sizes" :key="n" :value="n">{{ n }} 红</option></select></label>
            <label class="field" :for="prefix+'-blue-'+row.id">蓝球数量<select :id="prefix+'-blue-'+row.id" v-model.number="row.blue_count" data-intersection-blue :disabled="busy"><option v-for="n in [1,2,3]" :key="n" :value="n">{{ n }} 蓝</option></select></label>
            <button type="button" @click="remove(row.id)" :disabled="busy||rows.length<=2" :aria-label="'移除号码池'+(i+1)">移除</button>
          </div>
          <div class="row"><button type="button" @click="add" :disabled="busy||rows.length>=8">添加号码池</button><button type="button" @click="reset" :disabled="busy">重置选择</button></div>
          <p v-if="selectionError" class="notice error" role="alert">{{ selectionError }}</p>
          <section v-if="result" class="pool-intersection-result" :id="prefix+'-result'" tabindex="-1" data-intersection-result :aria-labelledby="prefix+'-result-title'">
            <h3 :id="prefix+'-result-title'">交集结果 · {{ result.red_count }}红＋{{ result.blue_count }}蓝</h3>
            <p class="muted">红球交集：{{ result.red_count }}个</p><div class="balls" data-intersection-reds><span v-for="n in result.reds" :key="n" class="ball red">{{ pad(n) }}</span><span v-if="!result.reds.length" class="muted">红球无交集</span></div>
            <p class="muted">蓝球交集：{{ result.blue_count }}个</p><div class="balls" data-intersection-blues><span v-for="n in result.blues" :key="n" class="ball blue">{{ pad(n) }}</span><span v-if="!result.blues.length" class="muted">蓝球无交集</span></div>
            <p v-if="!result.can_form_single" class="notice" data-intersection-insufficient>{{ result.red_count<6?'红球交集少于6个。':'' }}{{ !result.blue_count?'蓝球无交集。':'' }}当前结果不能组成完整的6红1蓝单注，未自动补号。</p>
            <p class="muted">{{ result.note }} {{ archived?'原档池交集仅用于诊断。':'交集为当前预览，不写入原冻结档案。' }}</p>
            <section v-if="analysis" data-intersection-review>
              <h3>开奖后诊断 · 中{{ analysis.red_hits }}/6红 · 蓝球{{ analysis.blue_hit?'入池':'未入池' }}</h3>
              <p>原池至少出现过、被交集剔除的开奖红球：<b>{{ analysis.removed_winning_reds.map(pad).join(' ')||'无' }}</b></p>
              <p class="muted">所有所选原池都缺少的开奖红球：{{ analysis.missing_in_all_sources.map(pad).join(' ')||'无' }}。</p>
              <div v-for="(s,i) in analysis.sources" :key="i" class="muted">来源{{ i+1 }} {{ s.label }}（{{ s.red_count }}红＋{{ s.blue_count }}蓝）：原池中{{ s.red_hits }}/6红，蓝球{{ s.blue_hit?'入池':'未入池' }}；其开奖红球被交集剔除 {{ s.removed_winning_reds.map(pad).join(' ')||'无' }}。</div>
            </section>
            <div class="row"><button type="button" @click="copy" :disabled="busy||!!loadError">复制交集</button><button type="button" @click="exportJson" :disabled="busy||!!loadError">导出交集 JSON</button></div>
            <details class="pool-intersection-sources"><summary>查看参与取交集的原号码</summary><div v-for="(source,i) in result.sources" :key="i"><p><b>{{ i+1 }}. {{ source.label }} · {{ source.red_count }}红＋{{ source.blue_count }}蓝</b></p><p class="hash">红 {{ source.reds.map(pad).join(' ') }}<br>蓝 {{ source.blues.map(pad).join(' ') }}</p></div></details>
          </section>
          <p v-if="message" class="notice" role="status">{{ message }}</p>
        </template>
      </div>
    </details>`
  };
  global.SSQPoolIntersection={modelsFrom,fromArchive,intersect,reviewIntersection,asText,component};
})(window);
