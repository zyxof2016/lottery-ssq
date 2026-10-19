const {createApp,ref,computed,onMounted,onUnmounted}=Vue;
createApp({setup(){
  const {url}=SSQApp,initial=new URLSearchParams(location.search);
  const tabs=[{id:'generator',name:'方案生成',icon:'🎯'},{id:'records',name:'我的方案与对奖',icon:'📊'},{id:'reviews',name:'每期推荐与复盘',icon:'📋'},{id:'draws',name:'开奖历史',icon:'🎱'},{id:'backtest',name:'科学回测与优化',icon:'🧪'},{id:'research',name:'研究评估',icon:'🔬'},{id:'settings',name:'系统设置',icon:'⚙️'}];
  const currentTab=ref(tabs.some(t=>t.id===initial.get('tab'))?initial.get('tab'):'generator');
  const overview=ref({latest_draw:null,total_draws:0,total_plans:0,pending:0,scored:0}),cycle=ref(null),toast=ref({show:false}),target=ref(''),deadline=ref('');
  const form=ref({ranking_model:'weighted',mode:'pool_observation',observation_size:16,tickets:8,pool:16,overlap:4,red_count:7,blue_count:2,bankers:2,drags:6});
  const compactChoices=[{size:14,name:'14红 · 更精简',desc:'比24红少10个；号码更少，也更容易漏号。'},{size:16,name:'16红 · 默认',desc:'比24红少8个；先集中看这一组。'},{size:18,name:'18红 · 扩展',desc:'在16红上增加2个号码，保留更多候选。'}];
  const presets=[{id:'pool_observation',name:'分档号码池 · 尽量少红覆盖全部开奖号',desc:'14／16／18／20／22／24红六档，蓝球1–3个；保存原池观察覆盖。'},{id:'coverage24',name:'24红全6覆盖观察 · 均衡8注',desc:'按历史条件频率排序，24红全部入票、每红2次；试验方案。'},{id:'equal24',name:'等权24红覆盖 · 对照8注',desc:'原等权排名24红全部入票、每红2次；同数量持续对照。'},{id:'calibrated14',name:'校准排名14红 · 均衡8注',desc:'校正上期重号集中偏置，按新排名选择14红；试验方案单独留档。'},{id:'balanced14',name:'14红均衡覆盖 · 8注16元',desc:'原融合前14红全部入票，每红出现3–4次；按原号码留档。'},{id:'frozen',name:'协议冻结 · 8注票组',desc:'温和救援，按冻结规则组票；保存时封存开奖前哈希。'},{id:'disjoint5',name:'5注红球不重叠',desc:'前30红球中分散组票，2蓝轮换，共10元。'},{id:'single8',name:'8注低重合',desc:'前15红球，注间重合不超过4个，2蓝轮换。'},{id:'compound7',name:'7红复式',desc:'7个红球配1蓝，完整展开7注。'},{id:'compound8',name:'8红复式',desc:'8个红球配1蓝，完整展开28注。'},{id:'blue2',name:'蓝球复式 · 2蓝',desc:'固定6红配2蓝，共2注。'},{id:'dantuo',name:'2胆6拖',desc:'前2个红球做胆，其后6个做拖。'}];
  const activePreset=ref(''),plan=ref(null),poolPreview=ref(null),lastRequest=ref(null),planSaved=ref(false),generating=ref(false),saving=ref(false),syncing=ref(false);
  const records=ref([]),recordFilter=ref('all'),recordsLoading=ref(false),settling=ref(false),draws=ref([]),drawBefore=ref(''),moreDraws=ref(true),drawsLoading=ref(false);
  const tests=ref([]),reportCards=ref([]),reportLoading=ref(false),reportLoaded=ref(false),verdict=ref('选择评估后显示原冻结确认窗结果。');
  const reviewsSrc=ref(url('reviews',{embed:1,issue:initial.get('issue'),archive:initial.get('archive'),strategy:initial.get('strategy')}));
  const researchSrc=ref(url('research',{embed:1,view:initial.get('view')||'pools'})),openedFrames=ref({reviews:false,research:false});
  const currentResult=computed(()=>poolPreview.value||plan.value),generatedIssue=computed(()=>currentResult.value?.issue||''),resultArchive=computed(()=>currentResult.value?.recommendation_archive);
  const poolCanFreeze=computed(()=>!!cycle.value?.publish&&poolPreview.value?.issue===cycle.value?.target);
  const stalePreview=computed(()=>!!currentResult.value&&!!overview.value.data_sha256&&currentResult.value.input_history_sha256!==overview.value.data_sha256);
  const planRedUnion=computed(()=>[...new Set((plan.value?.tickets||[]).flatMap(t=>t.red))]),planBlueUnion=computed(()=>[...new Set((plan.value?.tickets||[]).map(t=>t.blue))]);
  const shownRecords=computed(()=>records.value.filter(r=>recordFilter.value==='all'||r.status===recordFilter.value));
  const scoredCover=computed(()=>{const rs=records.value.filter(r=>r.status==='scored'&&r.coverage?.red_hit_n!=null);return rs.length?(rs.reduce((s,r)=>s+r.coverage.red_hit_n,0)/rs.length).toFixed(2)+'/6':'—';});
  const reviewsStandalone=computed(()=>{const u=new URL(reviewsSrc.value);u.searchParams.delete('embed');return u.href;}),researchStandalone=computed(()=>{const u=new URL(researchSrc.value);u.searchParams.delete('embed');return u.href;}),researchUrl=url('research',{view:'observations'});
  const pad=n=>String(n).padStart(2,'0'),f4=n=>n==null?'—':Number(n).toFixed(4),percent=n=>n==null?'—':(n*100).toFixed(2)+'%',freezeTime=n=>n?new Date(n*1000).toLocaleString('zh-CN',{timeZone:'Asia/Shanghai',hour12:false}):'时间未留存';
  let toastTimer;
  function showToast(message,type='success'){clearTimeout(toastTimer);toast.value={show:true,message,type};toastTimer=setTimeout(()=>toast.value.show=false,6000);}
  async function read(path,options){const response=await fetch(url(path),options);const data=await response.json();if(!response.ok)throw Error(data.error||'请求失败');return data;}
  function post(path,data){return read(path,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(data||{})});}
  async function loadOverview(){const previous=overview.value.next_issue;overview.value=await read('api/overview');if(!target.value||target.value===previous)target.value=overview.value.next_issue;}
  async function loadCycle(){cycle.value=await read('api/cycle');if(!deadline.value)deadline.value=cycle.value.deadline?.replace(' ','T')||'';}
  async function loadRecords(){recordsLoading.value=true;try{records.value=(await read('api/records')).records;}catch(e){showToast(e.message,'error');}finally{recordsLoading.value=false;}}
  async function loadDraws(before=''){if(drawsLoading.value)return;drawsLoading.value=true;try{const q=new URLSearchParams({limit:30});if(before)q.set('before',before);const d=await read('api/draws?'+q);draws.value=before?draws.value.concat(d.rows):d.rows;drawBefore.value=d.rows.at(-1)?.issue||'';moreDraws.value=d.rows.length===30;}catch(e){showToast(e.message,'error');}finally{drawsLoading.value=false;}}
  function setRoute(tab,params={}){const u=new URL(location.href);u.search='';if(tab!=='generator')u.searchParams.set('tab',tab);for(const [key,value]of Object.entries(params))if(['issue','archive','strategy','view'].includes(key)&&value)u.searchParams.set(key,value);history.replaceState(null,'',u);}
  async function openTab(id,params={}){
    if(!tabs.some(t=>t.id===id))return;
    currentTab.value=id;
    if(id==='reviews'||id==='research'){
      if(Object.keys(params).length){const old=new URL(id==='reviews'?reviewsSrc.value:researchSrc.value);if(id==='reviews'){for(const key of ['issue','archive','strategy'])old.searchParams.delete(key);}for(const [k,v]of Object.entries(params))if(['issue','archive','strategy','view'].includes(k)&&v)old.searchParams.set(k,v);if(id==='reviews')reviewsSrc.value=old.href;else researchSrc.value=old.href;}
      openedFrames.value[id]=true;
      const src=new URL(id==='reviews'?reviewsSrc.value:researchSrc.value);setRoute(id,Object.fromEntries(src.searchParams));
    }else setRoute(id);
    if(id==='records')await loadRecords();
    if(id==='backtest'&&!reportLoaded.value)await loadReport(false);
    if(id==='settings'){try{await loadCycle();}catch(e){showToast(e.message,'error');}}
  }
  function openReviewUrl(href){const u=new URL(href,SSQApp.base);return openTab('reviews',Object.fromEntries(u.searchParams));}
  function deadlineUnix(){const value=Date.parse(deadline.value+'+08:00');if(!Number.isFinite(value))throw Error('请填写有效的北京时间封存截止。');return Math.floor(value/1000);}
  function validateTarget(){const issue=target.value.trim();if(!/^\d{5}$/.test(issue))throw Error('目标期号需要5位数字。');return issue;}
  async function generate(body){
    if(generating.value||saving.value||syncing.value)return;
    generating.value=true;
    try{
      const issue=validateTarget(),request={...body,issue,ranking_model:body.preset==='coverage24'?'conditional_gap':body.preset==='calibrated14'?'repeat_calibrated':body.preset==='equal24'?'equal':form.value.mode==='frozen'?'weighted':form.value.ranking_model},config={...form.value};
      if(config.mode==='pool_observation'){
        const data=await read('api/pools');
        if(issue!==data.suggested_target)throw Error('观察池仅生成当前数据紧邻下一期；历史覆盖请到研究评估查看。');
        const redCount=config.observation_size,blueCount=config.blue_count,saved=config.ranking_model==='weighted'?data.next.weighted_pools:config.ranking_model==='conditional_gap'?data.next.weighted_pools.coverage_model:data.next.weighted_pools.comparison_models.find(m=>m.model===config.ranking_model),pool=saved?.pools.find(p=>p.red_count===redCount&&p.blue_count===blueCount);
        if(!pool||pool.group!=='observation')throw Error('请选择六档红球与1–3蓝的有效规格。');
        const baseline=data.windows.all_oos.red_curves.find(c=>c.model==='weighted'&&c.red_count===redCount).uniform_full_red_rate;
        const recent=data.windows.recent60,curve=recent?.red_curves.find(c=>c.model===config.ranking_model&&c.red_count===redCount),reference=recent?.red_curves.find(c=>c.model===config.ranking_model&&c.red_count===24);
        const compressionEvidence=curve&&reference?{from:recent.from,to:recent.to,n:recent.n,full_red_periods:curve.full_red_periods,reference_full_red_periods:reference.full_red_periods}:null;
        poolPreview.value={issue,model:config.ranking_model,model_label:config.ranking_model==='weighted'?'协议加权融合':config.ranking_model==='conditional_gap'?'条件频率 · 24红试验':config.ranking_model==='equal'?'等权融合 · 对照':'重号偏置校准 · 试验',cutoff_issue:data.cutoff_issue,cutoff_date:data.cutoff_date,input_history_sha256:data.data_sha256,red_count:redCount,blue_count:blueCount,reds:[...pool.reds].sort((a,b)=>a-b),blues:[...pool.blues].sort((a,b)=>a-b),uniform_full_red_rate:baseline,compression_evidence:compressionEvidence};plan.value=null;lastRequest.value=null;
      }else{
        if(config.mode==='frozen')request.not_after=deadlineUnix();
        const result=await post('api/generate',request);
        plan.value=result;poolPreview.value=null;lastRequest.value=request;
      }
      planSaved.value=false;showToast(poolPreview.value?'观察号码池已生成。':`${plan.value.title}：${plan.value.bet_count}注，${plan.value.cost}元。`);
    }catch(e){showToast(e.message,'error');}finally{generating.value=false;}
  }
  function generateCompact(size){
    if(generating.value||saving.value||syncing.value||!overview.value.next_issue||![14,16,18].includes(size))return;
    target.value=overview.value.next_issue;activePreset.value='compact'+size;
    Object.assign(form.value,{mode:'pool_observation',ranking_model:'weighted',observation_size:size,blue_count:2});
    return generate({});
  }
  function applyPreset(id){
    if(generating.value||saving.value||syncing.value)return;
    activePreset.value=id;
    const configs={pool_observation:{mode:'pool_observation',observation_size:24,blue_count:2},coverage24:{mode:'balanced',ranking_model:'conditional_gap',tickets:8,pool:24,blue_count:2},equal24:{mode:'balanced',ranking_model:'equal',tickets:8,pool:24,blue_count:2},calibrated14:{mode:'balanced',ranking_model:'repeat_calibrated',tickets:8,pool:14,blue_count:2},balanced14:{mode:'balanced',tickets:8,pool:14,blue_count:2},frozen:{mode:'frozen'},disjoint5:{mode:'single',tickets:5,pool:30,overlap:0,blue_count:2},single8:{mode:'single',tickets:8,pool:15,overlap:4,blue_count:2},compound7:{mode:'compound_red',red_count:7,blue_count:1},compound8:{mode:'compound_red',red_count:8,blue_count:1},blue2:{mode:'compound_blue',blue_count:2},dantuo:{mode:'dantuo',bankers:2,drags:6,blue_count:1}};
    Object.assign(form.value,configs[id]);if(!['pool_observation','calibrated14','coverage24','equal24'].includes(id))form.value.ranking_model='weighted';return generate(id==='pool_observation'?{}:{preset:id});
  }
  function generateCustom(){activePreset.value='';const f=form.value;return generate({mode:f.mode,tickets:f.tickets,pool:f.pool,overlap:f.overlap,red_count:f.red_count,blue_count:f.blue_count,bankers:f.bankers,drags:f.drags});}
  async function savePlan(){
    if(saving.value||generating.value||planSaved.value||!currentResult.value)return;
    if(stalePreview.value){showToast('数据已经变化，请重新生成后保存。','error');return;}
    saving.value=true;
    try{
      if(poolPreview.value){
        const result=await post('api/recommendations/freeze',{data_sha256:poolPreview.value.input_history_sha256,target:poolPreview.value.issue});
        if(result.observation_tiers!==6)throw Error('本期首次档案未保存六档，已保留旧档案；新期再开始观察。');
        poolPreview.value.recommendation_archive={url:url('reviews',{issue:result.issue,archive:'daily'}),sha256:result.sha256};
        if(poolPreview.value.model==='conditional_gap'&&result.comparison_models<5){poolPreview.value.recommendation_archive=null;showToast('本期首次档案未保存条件频率排名，当前预览未封存；从后续新期开始自动冻结。');refreshFrame('reviews');refreshFrame('research');return;}
        if(poolPreview.value.model==='repeat_calibrated'&&result.comparison_models<4){poolPreview.value.recommendation_archive=null;showToast('本期首次档案未保存校准排名，当前预览未封存；校准排名从后续新期开始自动冻结。');refreshFrame('reviews');refreshFrame('research');return;}
        if(poolPreview.value.model==='equal'&&result.comparison_models<3){poolPreview.value.recommendation_archive=null;showToast('本期首次档案未保存等权排名，当前预览未封存；从后续新期开始自动冻结。');refreshFrame('reviews');refreshFrame('research');return;}
        planSaved.value=true;showToast(result.comparison_models===5?'五模型共90个六档观察池和9个原小池已封存。':result.comparison_models===4?'四模型共72个六档观察池和9个原小池已封存。':result.comparison_models===3?'三模型共54个六档观察池和9个原小池已封存。':'已保留本期原加权模型六档；三模型对照从后续新期开始冻结。');
      }else{
        const body={...lastRequest.value,expected_archive_sha256:plan.value.recommendation_archive.sha256};
        const result=await post('api/plans',body);plan.value=result.plan;planSaved.value=true;
        await Promise.all([loadOverview(),loadRecords()]);showToast(result.already?'这组号码已经保存过。':'已保存到我的方案库。');
      }
      refreshFrame('reviews');refreshFrame('research');
    }catch(e){showToast(e.message,'error');}finally{saving.value=false;}
  }
  async function copyPlan(){
    if(!currentResult.value)return;
    const lines=[`【双色球 ${generatedIssue.value} 期 · ${poolPreview.value?'模型观察池':plan.value.title}】`];
    if(poolPreview.value){lines.push(poolPreview.value.model_label,`${poolPreview.value.red_count}红：${poolPreview.value.reds.map(pad).join(' ')}`,`${poolPreview.value.blue_count}蓝：${poolPreview.value.blues.map(pad).join(' ')}`,'号码池观察，不是单注投注票面。');}
    else{lines.push(`${plan.value.bet_count}注，${plan.value.cost}元`);plan.value.tickets.forEach((t,i)=>lines.push(`第${pad(i+1)}注：${t.red.map(pad).join(' ')} + ${pad(t.blue)}`));}
    const text=lines.join('\n');
    try{if(navigator.clipboard)await navigator.clipboard.writeText(text);else throw Error();}
    catch{const el=document.createElement('textarea');el.value=text;document.body.appendChild(el);el.select();const ok=document.execCommand('copy');el.remove();if(!ok){showToast('浏览器未允许复制，请手动选择号码。','error');return;}}
    showToast('号码文本已复制。');
  }
  async function settle(){if(settling.value)return;settling.value=true;try{const data=await post('api/settle');records.value=data.records;await loadOverview();refreshFrame('reviews');showToast(data.errors?.length?data.errors.join('；'):`新增${data.settled}条对奖记录。`,data.errors?.length?'error':'success');}catch(e){showToast(e.message,'error');}finally{settling.value=false;}}
  function refreshFrame(tab){if(!openedFrames.value[tab])return;const u=new URL(tab==='reviews'?reviewsSrc.value:researchSrc.value);u.searchParams.set('refresh',Date.now());if(tab==='reviews')reviewsSrc.value=u.href;else researchSrc.value=u.href;}
  async function syncDraws(){if(syncing.value||generating.value||saving.value)return;syncing.value=true;try{const data=await post('api/sync');await Promise.all([loadOverview(),loadRecords(),loadDraws(),loadCycle()]);reportLoaded.value=false;refreshFrame('reviews');refreshFrame('research');showToast(`已同步到${data.latest_draw.issue}期，共${data.n}期。`);}catch(e){showToast(e.message,'error');}finally{syncing.value=false;}}
  async function loadReport(force=false){if(reportLoading.value)return;reportLoading.value=true;try{const d=force?await post('api/backtest'):await read('api/report'),s=d.summary,c=s.confirmation;tests.value=s.tests||[];const good=tests.value.filter(t=>t.p_holm<.05&&(t.better_if_positive?t.estimate>0:t.estimate<0));verdict.value=good.length?`确认窗有${good.length}项经Holm校正后指向有利方向。`:'确认窗的预注册检验经Holm校正后，没有一项指向稳定优势。';reportCards.value=[{k:'确认窗',v:`${c.start}–${c.end}`,s:`${c.n}期`},{k:'有效独立源',v:f4(s.n_eff),s:'参与比'},{k:'随机单注中红期望',v:'1.09',s:'6×6/33'},{k:'可检出超额',v:f4(s.mde.red_hits),s:'80%功效'}];reportLoaded.value=true;}catch(e){showToast(e.message,'error');}finally{reportLoading.value=false;}}
  const hitRed=(n,r)=>r.draw?.red.includes(n)===true,hitBlue=(n,r)=>r.draw?.blue===n;
  function frameLoaded(event){try{event.target.style.height=Math.max(700,event.target.contentDocument.body.scrollHeight+24)+'px';}catch{}}
  function frameMessage(event){
    if(event.origin!==location.origin)return;
    const frame=[...document.querySelectorAll('iframe.shell-frame')].find(f=>f.contentWindow===event.source);if(!frame)return;
    const data=event.data;if(!data||typeof data!=='object')return;
    if(data.type==='ssq:resize'&&Number.isFinite(data.height))frame.style.height=Math.max(500,Math.min(30000,data.height))+'px';
    if(data.type==='ssq:navigate'&&tabs.some(t=>t.id===data.tab))openTab(data.tab,data.params||{});
  }
  onMounted(async()=>{
    window.addEventListener('message',frameMessage);
    const results=await Promise.allSettled([loadOverview(),loadRecords(),loadDraws(),loadCycle()]);
    for(const result of results)if(result.status==='rejected')showToast(result.reason.message,'error');
    await openTab(currentTab.value);
  });
  onUnmounted(()=>{window.removeEventListener('message',frameMessage);clearTimeout(toastTimer);});
  return {tabs,currentTab,overview,cycle,toast,target,deadline,form,compactChoices,presets,activePreset,plan,poolPreview,planSaved,generating,saving,syncing,currentResult,generatedIssue,resultArchive,stalePreview,poolCanFreeze,planRedUnion,planBlueUnion,records,recordFilter,recordsLoading,shownRecords,scoredCover,settling,draws,drawBefore,drawsLoading,moreDraws,tests,reportCards,reportLoading,verdict,reviewsSrc,researchSrc,reviewsStandalone,researchStandalone,openedFrames,researchUrl,pad,f4,percent,freezeTime,openTab,openReviewUrl,generateCompact,applyPreset,generateCustom,savePlan,copyPlan,loadRecords,loadDraws,settle,syncDraws,loadReport,hitRed,hitBlue,frameLoaded};
}}).mount('#app');
