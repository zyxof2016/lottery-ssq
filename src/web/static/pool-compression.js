(function(global){
  'use strict';
  const sizes=[14,16,18,20,22,24];
  function compare(curves,model,n){
    if(!Number.isInteger(n)||n<=0)return [];
    const source=curves.filter(c=>c.model===model),reference=source.find(c=>c.red_count===24);
    if(!reference||(reference.n!=null&&reference.n!==n)||!Number.isInteger(reference.full_red_periods)||reference.full_red_periods<0||reference.full_red_periods>n)return [];
    return sizes.flatMap(red=>{
      const cell=source.find(c=>c.red_count===red),full=cell?.full_red_periods;
      if(!Number.isInteger(full)||full<0||full>reference.full_red_periods|| (cell.n!=null&&cell.n!==n))return [];
      return [{red_count:red,full_red_periods:full,reference_full_red_periods:reference.full_red_periods,
        removed_red_count:24-red,compression_loss_periods:reference.full_red_periods-full,
        retained_full_red_fraction:reference.full_red_periods?full/reference.full_red_periods:null,
        joint_counts:cell.full_draw_periods_by_blue||[],uniform_full_red_rate:cell.uniform_full_red_rate}];
    });
  }
  const outcomes=[['full_draw','红蓝全入池'],['red_miss_only','只漏红球'],['blue_miss_only','只漏蓝球'],['both_miss','红蓝都漏']];
  function jointBreakdown(cell){
    if(!Number.isInteger(cell?.n)||cell.n<=0||!cell.joint_coverage)return [];
    const rows=outcomes.map(([status,label])=>({status,label,...cell.joint_coverage[status]}));
    if(rows.some(r=>!Number.isInteger(r.periods)||r.periods<0||!Array.isArray(r.issues)||r.issues.length!==r.periods)
      ||rows.reduce((n,r)=>n+r.periods,0)!==cell.n)return [];
    const issues=rows.flatMap(r=>r.issues);
    if(new Set(issues).size!==cell.n||!Array.isArray(cell.scored_issues)||cell.scored_issues.length!==cell.n
      ||!cell.scored_issues.every(i=>issues.includes(i)))return [];
    return rows;
  }
  const component={
    props:{curves:{type:Array,default:()=>[]},window:{type:Object,default:null},actual:{type:Array,default:()=>[]},model:{type:String,required:true},redCount:{type:Number,default:16},blueCount:{type:Number,default:2},busy:{type:Boolean,default:false},switchable:{type:Boolean,default:false}},
    emits:['select'],
    setup(props){
      const rows=Vue.computed(()=>compare(props.curves,props.model,props.window?.n).map(r=>({...r,actual:props.actual.find(c=>c.model===props.model&&c.red_count===r.red_count&&c.blue_count===props.blueCount)})));
      const selectedActual=Vue.computed(()=>props.actual.find(c=>c.model===props.model&&c.red_count===props.redCount&&c.blue_count===props.blueCount));
      const jointRows=Vue.computed(()=>jointBreakdown(selectedActual.value));
      const percent=n=>n==null?'—':(n*100).toFixed(1)+'%';
      return {rows,percent,selectedActual,jointRows};
    },
    template:`<div class="pool-compression" aria-label="号码池压缩对照">
      <p class="muted">同一模型、同一期、同一蓝球数量，以24红为参照。压缩损失只计24红原本全包、缩小后漏号的期数；24红也未全包的期次属于排名漏号。</p>
      <p v-if="!rows.length" class="notice" role="status">当前没有可比较的同模型回放样本，压缩保留率暂为空。</p>
      <template v-else>
      <p class="muted">近 {{ window.n }} 期因果回放（{{ window.from }}—{{ window.to }}）。保留率的分母是24红全包次数，并非全部期数或下一期概率。</p>
      <p class="mobile-scroll-hint muted">左右滑动表格，查看历史全包次数与实际留档。</p>
      <div class="scroll"><table class="compression-table"><thead><tr><th>红球数量</th><th>压缩损失 / 保留率</th><th>历史全6红 / {{ window.n }}期</th><th>实际共同留档</th></tr></thead>
        <tbody><tr v-for="r in rows" :key="r.red_count" :data-compression-red="r.red_count" :class="{active:r.red_count===redCount}">
        <td><button v-if="switchable" type="button" @click="$emit('select',r.red_count)" :disabled="busy" :aria-label="'查看'+r.red_count+'红号码池'" :aria-pressed="r.red_count===redCount">{{ r.red_count }}红</button><b v-else>{{ r.red_count }}红</b><small>少选 {{ r.removed_red_count }} 个</small></td>
        <td>损失 {{ r.compression_loss_periods }} 期<small>保留 {{ r.full_red_periods }}/{{ r.reference_full_red_periods }} · {{ percent(r.retained_full_red_fraction) }}</small><small v-if="!r.reference_full_red_periods">24红全包为0，无法评估保留率</small></td>
        <td>{{ r.full_red_periods }} / {{ window.n }}<small>{{ blueCount }}蓝同时全包 {{ r.joint_counts[blueCount-1]??'—' }} / {{ window.n }}</small></td>
        <td v-if="r.actual?.n">{{ r.actual.full_red_periods }}/{{ r.actual.n }}期全包<small>同批24红 {{ r.actual.reference_full_red_periods }}/{{ r.actual.n }}，压缩损失 {{ r.actual.compression_loss_periods }} 期</small><small>排名漏号 {{ r.actual.ranking_miss_periods }} 期 · 待开奖 {{ r.actual.pending }} 期</small><details v-if="r.actual.compression_loss_issues.length||r.actual.ranking_miss_issues.length"><summary>查看归因期号</summary><small>压缩损失：{{ r.actual.compression_loss_issues.join('、')||'无' }}</small><small>排名漏号：{{ r.actual.ranking_miss_issues.join('、')||'无' }}</small></details></td>
        <td v-else>暂无已开奖样本<small v-if="r.actual">待开奖 {{ r.actual.pending }} 期；缺少成对原池 {{ r.actual.excluded_missing_pair }} 期</small></td>
        </tr></tbody></table></div>
      <p class="muted">历史回放与实际留档分别统计；新模型缺档及待开奖不算失败。全包次数相同也不能证明压缩不影响未来覆盖。</p>
      </template>
      <section class="joint-loss" aria-label="实际留档红蓝覆盖归因">
        <h3>实际留档 · {{ redCount }}红 + {{ blueCount }}蓝的覆盖归因</h3>
        <template v-if="jointRows.length">
          <p class="muted">同模型、同规格并已保存24红参照的 {{ selectedActual.n }} 期原档，每期只计入一类；待开奖 {{ selectedActual.pending }} 期、缺少成对原池 {{ selectedActual.excluded_missing_pair }} 期均未计入。</p>
          <div class="joint-loss-grid"><div v-for="r in jointRows" :key="r.status" class="joint-loss-item" :data-joint-status="r.status"><span>{{ r.label }}</span><strong>{{ r.periods }} / {{ selectedActual.n }} 期</strong><details v-if="r.periods"><summary>查看期号</summary><span class="muted">{{ r.issues.join('、') }}</span></details></div></div>
          <p class="muted">只漏蓝球：红球已全入池，增加红球或换红球模型不能补回该期蓝球。只漏红球：蓝球已入池；红蓝都漏需要分别检查两端。号码池全入池不等于单注中奖。</p>
        </template>
        <p v-else-if="selectedActual && !selectedActual.n" class="muted">暂无已开奖样本；待开奖 {{ selectedActual.pending }} 期，缺少成对原池 {{ selectedActual.excluded_missing_pair }} 期。覆盖归因暂为空。</p>
        <p v-else class="muted">暂未取得完整的实际原档覆盖归因。</p>
      </section>
    </div>`
  };
  global.SSQPoolCompression={compare,jointBreakdown,component};
})(typeof window==='undefined'?globalThis:window);
