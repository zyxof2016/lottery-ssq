const {createApp,ref,computed,onMounted,onUnmounted,nextTick}=Vue;
createApp({components:{PoolIntersection:SSQPoolIntersection.component,PoolCompression:SSQPoolCompression.component,BlueFocus:SSQBlueFocus.component,SmallRed:SSQSmallRed.component,ReferencePools:SSQReferencePools.component},setup(){
  const {url}=SSQApp,initial=new URLSearchParams(location.search);
  const tabs=[{id:'generator',name:'方案生成',icon:'🎯'},{id:'pools',name:'观察池',icon:''},{id:'records',name:'我的方案',icon:'📊'},{id:'reviews',name:'推荐与复盘',icon:'📋'},{id:'draws',name:'开奖历史',icon:'🎱'},{id:'backtest',name:'科学回测',icon:'🧪'},{id:'research',name:'研究评估',icon:'🔬'},{id:'settings',name:'系统设置',icon:'⚙️'}];
  const intersectionApiUrl=url('api/pools'),blueApiUrl=url('api/blue'),smallRedApiUrl=url('api/small-red');
  const currentTab=ref(tabs.some(t=>t.id===initial.get('tab'))?initial.get('tab'):'generator');
  const overview=ref({latest_draw:null,total_draws:0,total_plans:0,pending:0,scored:0}),cycle=ref(null),toast=ref({show:false}),target=ref(''),deadline=ref('');
  const form=ref({blue_model:'weighted',ranking_model:'weighted',mode:'pool_observation',observation_size:16,tickets:8,pool:16,overlap:4,red_count:7,blue_count:2,bankers:2,drags:6});
  const compactModel=ref('weighted'),compactAlgorithms=[{model:'weighted',label:'原加权融合',desc:'沿用原协议排名作为对照。'},{model:'repeat_calibrated',label:'重号偏置校准 · 试验',desc:'平衡上期红球与其他红球的评分，保持组内原顺序。'},{model:'rank_consensus',label:'名次共识 · 试验',desc:'汇总七个专家的名次，减少评分尺度差异。'},{model:'scenario_cover',label:'全6情景覆盖 · 试验',desc:'优先保留此前六红情景的完整覆盖。'},{model:'robust_cover',label:'三窗稳健覆盖 · 试验',desc:'同时照顾60／156／312期的完整六红情景。'},{model:'dual_support',label:'双专家支持 · 24红试验',desc:'仅观察24红；16红及长期2蓝联合全包回放退步，不替换默认。'}];
  const compactChoices=[{size:14,name:'14红 · 更精简',desc:'比24红少10个；号码更少，也更容易漏号。'},{size:16,name:'16红 · 默认',desc:'比24红少8个；先集中看这一组。'},{size:18,name:'18红 · 扩展',desc:'在16红上增加2个号码，保留更多候选。'}];
  const presets=[{id:'pool_observation',name:'分档号码池 · 尽量少红覆盖全部开奖号',desc:'14／16／18／20／22／24红六档，蓝球1–3个；保存原池观察覆盖。'},{id:'weighted24',name:'原排名24红覆盖 · 均衡8注',desc:'协议原加权前24红全部入票、每红2次；与试验排名分开观察。'},{id:'coverage24',name:'条件频率24红 · 试验8注',desc:'按历史条件频率排序，24红全部入票、每红2次；试验方案。'},{id:'equal24',name:'等权24红覆盖 · 对照8注',desc:'原等权排名24红全部入票、每红2次；同数量持续对照。'},{id:'calibrated14',name:'校准排名14红 · 均衡8注',desc:'校正上期重号集中偏置，按新排名选择14红；试验方案单独留档。'},{id:'balanced14',name:'14红均衡覆盖 · 8注16元',desc:'原融合前14红全部入票，每红出现3–4次；按原号码留档。'},{id:'frozen',name:'协议冻结 · 8注票组',desc:'温和救援，按冻结规则组票；保存时封存开奖前哈希。'},{id:'disjoint5',name:'5注红球不重叠',desc:'前30红球中分散组票，2蓝轮换，共10元。'},{id:'single8',name:'8注低重合',desc:'前15红球，注间重合不超过4个，2蓝轮换。'},{id:'compound7',name:'7红复式',desc:'7个红球配1蓝，完整展开7注。'},{id:'compound8',name:'8红复式',desc:'8个红球配1蓝，完整展开28注。'},{id:'blue2',name:'蓝球复式 · 2蓝',desc:'固定6红配2蓝，共2注。'},{id:'dantuo',name:'2胆6拖',desc:'前2个红球做胆，其后6个做拖。'}];
  const activePreset=ref(''),plan=ref(null),poolPreview=ref(null),lastRequest=ref(null),planSaved=ref(false),generating=ref(false),saving=ref(false),syncing=ref(false);
  const records=ref([]),recordFilter=ref('all'),recordsLoading=ref(false),settling=ref(false),draws=ref([]),drawBefore=ref(''),moreDraws=ref(true),drawsLoading=ref(false);
  const tests=ref([]),reportCards=ref([]),reportLoading=ref(false),reportLoaded=ref(false),verdict=ref('选择评估后显示原冻结确认窗结果。');
  const poolPane=ref(initial.get('pane')||'small');
  const poolsSrc=ref(url('pools',{embed:1,pane:poolPane.value}));
  const poolsStandalone=computed(()=>{const u=new URL(poolsSrc.value);u.searchParams.delete('embed');u.searchParams.set('pane',poolPane.value);return u.href;});
  const reviewsSrc=ref(url('reviews',{embed:1,issue:initial.get('issue'),archive:initial.get('archive'),strategy:initial.get('strategy')}));
  const researchSrc=ref(url('research',{embed:1,view:initial.get('view')||'pools'})),openedFrames=ref({pools:false,reviews:false,research:false});
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
  function setRoute(tab,params={}){const u=new URL(location.href);u.search='';if(tab!=='generator')u.searchParams.set('tab',tab);for(const [key,value]of Object.entries(params))if(['issue','archive','strategy','view','pane'].includes(key)&&value)u.searchParams.set(key,value);history.replaceState(null,'',u);}
  async function openTab(id,params={}){
    if(!tabs.some(t=>t.id===id))return;
    const changed=currentTab.value!==id;currentTab.value=id;nextTick(()=>document.getElementById('tab-'+id)?.scrollIntoView({block:'nearest',inline:'nearest'}));if(changed)window.scrollTo({top:0,behavior:'instant'});
    if(id==='pools'){const u=new URL(poolsSrc.value);if(params.pane){u.searchParams.set('pane',params.pane);poolPane.value=params.pane;}poolsSrc.value=u.href;openedFrames.value.pools=true;setRoute(id,{pane:poolPane.value});}
    else if(id==='reviews'||id==='research'){
      if(Object.keys(params).length){const old=new URL(id==='reviews'?reviewsSrc.value:researchSrc.value);if(id==='reviews'){for(const key of ['issue','archive','strategy'])old.searchParams.delete(key);}for(const [k,v]of Object.entries(params))if(['issue','archive','strategy','view'].includes(k)&&v)old.searchParams.set(k,v);if(id==='reviews')reviewsSrc.value=old.href;else researchSrc.value=old.href;}
      openedFrames.value[id]=true;
      const src=new URL(id==='reviews'?reviewsSrc.value:researchSrc.value);setRoute(id,Object.fromEntries(src.searchParams));
    }else setRoute(id);
    if(id==='records')await loadRecords();
    if(id==='backtest'&&!reportLoaded.value)await loadReport(false);
    if(id==='settings'){try{await loadCycle();}catch(e){showToast(e.message,'error');}}
  }
  async function navigateTabs(event){
    if(!['ArrowLeft','ArrowRight','Home','End'].includes(event.key)||event.target.getAttribute('role')!=='tab')return;
    event.preventDefault();const i=tabs.findIndex(t=>t.id===currentTab.value),n=event.key==='Home'?0:event.key==='End'?tabs.length-1:(i+(event.key==='ArrowRight'?1:-1)+tabs.length)%tabs.length;
    openTab(tabs[n].id);await nextTick();document.getElementById('tab-'+tabs[n].id)?.focus();
  }
  function openReviewUrl(href){const u=new URL(href,SSQApp.base);return openTab('reviews',Object.fromEntries(u.searchParams));}
  function deadlineUnix(){const value=Date.parse(deadline.value+'+08:00');if(!Number.isFinite(value))throw Error('请填写有效的北京时间封存截止。');return Math.floor(value/1000);}
  function validateTarget(){const issue=target.value.trim();if(!/^\d{5}$/.test(issue))throw Error('目标期号需要5位数字。');return issue;}
  async function generate(body){
    if(generating.value||saving.value||syncing.value)return;
    generating.value=true;let completed=false;
    try{
      const issue=validateTarget(),request={...body,issue,blue_model:['bluehit2','bluehit4'].includes(body.preset)?'blue_hit_frozen':body.preset?'weighted':form.value.blue_model,ranking_model:body.preset==='weighted24'?'weighted':body.preset==='coverage24'?'conditional_gap':body.preset==='calibrated14'?'repeat_calibrated':body.preset==='equal24'?'equal':form.value.mode==='frozen'?'weighted':form.value.ranking_model},config={...form.value};
      if(config.mode==='pool_observation'){
        const [data,archive,actual]=await Promise.all([read('api/pools'),read('api/recommendations/issues/'+issue+'?archive=daily').catch(()=>null),read('api/pools/observations').then(data=>({data})).catch(error=>({error:error.message}))]);
        if(actual.data&&actual.data.data_sha256!==data.data_sha256)throw Error('开奖数据已变化，请重新生成。');
        if(issue!==data.suggested_target)throw Error('观察池仅生成当前数据紧邻下一期；历史覆盖请到研究评估查看。');
        const root=data.next.weighted_pools,models=[{model:'weighted',...root},...root.comparison_models,...(root.coverage_model?[root.coverage_model]:[]),...(root.algorithm_models||[])];
        const redCount=config.observation_size,blueCount=config.blue_count,saved=models.find(m=>m.model===config.ranking_model),pool=saved?.pools.find(p=>p.red_count===redCount&&p.blue_count===blueCount);
        if(!pool||pool.group!=='observation')throw Error('请选择六档红球与1–3蓝的有效规格。');
        const baseline=data.windows.all_oos.red_curves.find(c=>c.model==='weighted'&&c.red_count===redCount).uniform_full_red_rate;
        let recent=data.windows.recent60;
        if((root.algorithm_models||[]).some(m=>m.model===config.ranking_model)){const study=await read('api/pool-algorithms');if(study.data_sha256!==data.data_sha256)throw Error('开奖数据已变化，请重新生成。');recent={...study.windows.recent60,red_curves:study.windows.recent60?.cells||[]};}
        if(['repeat_calibrated','conditional_gap'].includes(config.ranking_model)){const study=await read(config.ranking_model==='repeat_calibrated'?'api/coverage':'api/full6');if(study.data_sha256!==data.data_sha256)throw Error('开奖数据已变化，请重新生成。');recent={...study.windows.recent60,red_curves:study.windows.recent60?.cells||[]};}
        const curve=recent?.red_curves.find(c=>c.model===config.ranking_model&&c.red_count===redCount),reference=recent?.red_curves.find(c=>c.model===config.ranking_model&&c.red_count===24);
        const compressionEvidence=curve&&reference?{from:recent.from,to:recent.to,n:recent.n,full_red_periods:curve.full_red_periods,reference_full_red_periods:reference.full_red_periods,curves:recent.red_curves,window:{n:recent.n,from:recent.from,to:recent.to},actual_cells:actual.data?.compression_cells||[],actual_error:actual.error||(actual.data?.errors?.length?actual.data.errors.join('；'):'')}:null;
        const archiveState=poolArchiveState(archive,issue,config.ranking_model,pool,data.data_sha256);
        poolPreview.value={archive_state:archiveState,issue,model:config.ranking_model,policy:saved.policy,model_label:saved.policy?.label||(config.ranking_model==='weighted'?'协议加权融合':config.ranking_model==='conditional_gap'?'条件频率 · 24红试验':config.ranking_model==='equal'?'等权融合 · 对照':'重号偏置校准 · 试验'),cutoff_issue:data.cutoff_issue,cutoff_date:data.cutoff_date,input_history_sha256:data.data_sha256,red_count:redCount,blue_count:blueCount,reds:[...pool.reds].sort((a,b)=>a-b),blues:[...pool.blues].sort((a,b)=>a-b),uniform_full_red_rate:baseline,compression_evidence:compressionEvidence};plan.value=null;lastRequest.value=null;
      }else{
        if(config.mode==='frozen')request.not_after=deadlineUnix();
        const result=await post('api/generate',request);
        plan.value=result;poolPreview.value=null;lastRequest.value=request;
      }
      planSaved.value=false;completed=true;showToast(poolPreview.value?'观察号码池已生成。':`${plan.value.title}：${plan.value.bet_count}注，${plan.value.cost}元。`);
    }catch(e){showToast(e.message,'error');}finally{generating.value=false;if(completed){await nextTick();const result=document.getElementById('generated-plan');result?.focus({preventScroll:true});result?.scrollIntoView({block:'start',behavior:'instant'});}}
  }
  function poolArchiveState(archive,issue,model,pool,inputHash){
    const href=url('reviews',{issue,archive:'daily'});if(!archive)return {unavailable:true,url:href};
    if(archive.missing)return {exists:false,url:href};
    if(archive.issue!==issue||archive.archive_id!=='daily'||!archive.provenance?.sha256)return {unavailable:true,url:href};
    const root=archive.number_pools,models=root?[{...root,model:'weighted'},...(root.comparison_models||[]),...(root.coverage_model?[root.coverage_model]:[]),...(root.algorithm_models||[])]:[];
    const original=models.find(m=>m.model===model)?.pools?.find(p=>p.red_count===pool.red_count&&p.blue_count===pool.blue_count&&p.group==='observation');
    const same=(a,b)=>{if(!Array.isArray(a)||!Array.isArray(b)||a.length!==b.length)return false;const expected=[...b].sort((x,y)=>x-y);return [...a].sort((x,y)=>x-y).every((n,i)=>n===expected[i]);};
    return {exists:true,model_saved:!!original,url:href,matches:archive.provenance.input_history_sha256===inputHash&&!!original&&same(original.reds,pool.reds)&&same(original.blues,pool.blues)};
  }
  function generateCoverage(model,size){
    if(generating.value||saving.value||syncing.value||!['repeat_calibrated','rank_consensus','weighted','dual_support'].includes(model)||![22,24].includes(size))return;
    compactModel.value=model;return generateCompact(size);
  }
  function generateCompact(size){
    if(generating.value||saving.value||syncing.value||!overview.value.next_issue||![14,16,18,22,24].includes(size))return;
    target.value=overview.value.next_issue;activePreset.value='compact'+size;
    Object.assign(form.value,{mode:'pool_observation',ranking_model:compactModel.value,observation_size:size,blue_count:2});
    return generate({});
  }
  function generatePoolTier(size){
    if(generating.value||saving.value||syncing.value||!poolPreview.value||![14,16,18,20,22,24].includes(size))return;
    const preview=poolPreview.value;target.value=preview.issue;
    Object.assign(form.value,{mode:'pool_observation',ranking_model:preview.model,observation_size:size,blue_count:preview.blue_count});
    activePreset.value='';return generate({});
  }
  function generateSmall(choice){
    if(generating.value||saving.value||syncing.value)return;
    if(choice.data_sha256!==overview.value.data_sha256||choice.issue!==overview.value.next_issue){showToast('小池预览与当前开奖不一致，请同步后重新生成。','error');return;}
    if(!['weighted','small_recent'].includes(choice.model)||![12,14,15,16].includes(choice.red_count))return;
    openTab('generator');target.value=choice.issue;activePreset.value='';Object.assign(form.value,{mode:'balanced',ranking_model:choice.model,blue_model:'weighted',pool:choice.red_count,tickets:8,blue_count:2});
    return generate({mode:'balanced',pool:choice.red_count,tickets:8,blue_count:2,expected_data_sha256:choice.data_sha256});
  }
  function generateBlue(choice){
    if(generating.value||saving.value||syncing.value)return;
    if(choice.data_sha256!==overview.value.data_sha256||choice.issue!==overview.value.next_issue){showToast('蓝球预览与当前开奖不一致，请同步后重新生成。','error');return;}
    openTab('generator');target.value=choice.issue;return applyPreset(choice.preset,{expected_data_sha256:choice.data_sha256});
  }
  function applyPreset(id,expected={}){
    if(generating.value||saving.value||syncing.value)return;
    activePreset.value=id;
    const configs={balanced16:{mode:'balanced',ranking_model:'weighted',tickets:8,pool:16,blue_count:2},bluehit4:{mode:'balanced',ranking_model:'weighted',blue_model:'blue_hit_frozen',tickets:8,pool:16,blue_count:4},bluefocus2:{mode:'balanced',ranking_model:'weighted',tickets:8,pool:16,blue_count:2},bluehit2:{mode:'balanced',ranking_model:'weighted',blue_model:'blue_hit_frozen',tickets:8,pool:16,blue_count:2},bluefocus4:{mode:'balanced',ranking_model:'weighted',tickets:8,pool:16,blue_count:4},pool_observation:{mode:'pool_observation',observation_size:24,blue_count:2},weighted24:{mode:'balanced',ranking_model:'weighted',tickets:8,pool:24,blue_count:2},coverage24:{mode:'balanced',ranking_model:'conditional_gap',tickets:8,pool:24,blue_count:2},equal24:{mode:'balanced',ranking_model:'equal',tickets:8,pool:24,blue_count:2},calibrated14:{mode:'balanced',ranking_model:'repeat_calibrated',tickets:8,pool:14,blue_count:2},balanced14:{mode:'balanced',tickets:8,pool:14,blue_count:2},frozen:{mode:'frozen'},disjoint5:{mode:'single',tickets:5,pool:30,overlap:0,blue_count:2},single8:{mode:'single',tickets:8,pool:15,overlap:4,blue_count:2},compound7:{mode:'compound_red',red_count:7,blue_count:1},compound8:{mode:'compound_red',red_count:8,blue_count:1},blue2:{mode:'compound_blue',blue_count:2},dantuo:{mode:'dantuo',bankers:2,drags:6,blue_count:1}};
    Object.assign(form.value,{blue_model:'weighted'},configs[id]);if(!['pool_observation','calibrated14','coverage24','equal24'].includes(id))form.value.ranking_model='weighted';return generate(id==='pool_observation'?{}:id==='balanced16'?{mode:'balanced',tickets:8,pool:16,blue_count:2,...expected}:{preset:id,...expected});
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
        if(!result.saved_models?.includes(poolPreview.value.model)){poolPreview.value.recommendation_archive=null;showToast('本期首次档案未保存所选模型，当前预览未封存；从后续新期首次冻结开始保存。');refreshFrame('reviews');refreshFrame('research');return;}
        planSaved.value=true;showToast(`已保留${result.comparison_models}个模型、${result.number_pools+result.comparison_number_pools}个号码池的首次冻结档案。`);
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
  function refreshFrame(tab){if(!openedFrames.value[tab])return;const source={reviews:reviewsSrc,research:researchSrc,pools:poolsSrc}[tab];const u=new URL(source.value);if(tab==='pools')u.searchParams.set('pane',poolPane.value);u.searchParams.set('refresh',Date.now());source.value=u.href;}
  async function syncDraws(){if(syncing.value||generating.value||saving.value)return;syncing.value=true;try{const data=await post('api/sync');await Promise.all([loadOverview(),loadRecords(),loadDraws(),loadCycle()]);reportLoaded.value=false;refreshFrame('reviews');refreshFrame('research');refreshFrame('pools');showToast(`已同步到${data.latest_draw.issue}期，共${data.n}期。`);}catch(e){showToast(e.message,'error');}finally{syncing.value=false;}}
  async function loadReport(force=false){if(reportLoading.value)return;reportLoading.value=true;try{const d=force?await post('api/backtest'):await read('api/report'),s=d.summary,c=s.confirmation;tests.value=s.tests||[];const good=tests.value.filter(t=>t.p_holm<.05&&(t.better_if_positive?t.estimate>0:t.estimate<0));verdict.value=good.length?`确认窗有${good.length}项经Holm校正后指向有利方向。`:'确认窗的预注册检验经Holm校正后，没有一项指向稳定优势。';reportCards.value=[{k:'确认窗',v:`${c.start}–${c.end}`,s:`${c.n}期`},{k:'有效独立源',v:f4(s.n_eff),s:'参与比'},{k:'随机单注中红期望',v:'1.09',s:'6×6/33'},{k:'可检出超额',v:f4(s.mde.red_hits),s:'80%功效'}];reportLoaded.value=true;}catch(e){showToast(e.message,'error');}finally{reportLoading.value=false;}}
  const hitRed=(n,r)=>r.draw?.red.includes(n)===true,hitBlue=(n,r)=>r.draw?.blue===n;
  function frameLoaded(event){try{event.target.style.height=Math.max(700,event.target.contentDocument.body.scrollHeight+24)+'px';}catch{}}
  function frameMessage(event){
    if(event.origin!==location.origin)return;
    const frame=[...document.querySelectorAll('iframe.shell-frame')].find(f=>f.contentWindow===event.source);if(!frame)return;
    const data=event.data;if(!data||typeof data!=='object')return;
    if(data.type==='ssq:generate-small')generateSmall(data.choice||{});
    if(data.type==='ssq:generate-blue')generateBlue(data.choice||{});
    if(data.type==='ssq:pool-pane'&&data.pane&&currentTab.value==='pools'){poolPane.value=data.pane;setRoute('pools',{pane:data.pane});}
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
  return {poolsSrc,poolsStandalone,smallRedApiUrl,blueApiUrl,intersectionApiUrl,tabs,currentTab,overview,cycle,toast,target,deadline,form,compactModel,compactAlgorithms,compactChoices,presets,activePreset,plan,poolPreview,planSaved,generating,saving,syncing,currentResult,generatedIssue,resultArchive,stalePreview,poolCanFreeze,planRedUnion,planBlueUnion,records,recordFilter,recordsLoading,shownRecords,scoredCover,settling,draws,drawBefore,drawsLoading,moreDraws,tests,reportCards,reportLoading,verdict,reviewsSrc,researchSrc,reviewsStandalone,researchStandalone,openedFrames,researchUrl,pad,f4,percent,freezeTime,openTab,navigateTabs,openReviewUrl,generateSmall,generateBlue,generateCompact,generatePoolTier,generateCoverage,applyPreset,generateCustom,savePlan,copyPlan,loadRecords,loadDraws,settle,syncDraws,loadReport,hitRed,hitBlue,frameLoaded};
}}).mount('#app');
