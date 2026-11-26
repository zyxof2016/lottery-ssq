(function (global) {
  'use strict';
  const component = {
    setup() {
      const overview = Vue.ref(null);
      const detail = Vue.ref(null);
      const error = Vue.ref('');
      const loading = Vue.ref(false);
      const issue = Vue.ref('');
      const panel = Vue.ref('rank');
      const windowName = Vue.ref('confirmation_25111_26115');
      const sort = Vue.ref('group');
      const rankSpan = Vue.ref(33);
      const poolSort = Vue.ref('size');
      const poolKind = Vue.ref('all');
      const poolSearch = Vue.ref('');
      const numberAlgo = Vue.ref('fusion_v4');
      const blueAlgo = Vue.ref('blue_weighted');
      const blueSort = Vue.ref('group');
      const openedKey = Vue.ref('');
      const cover16 = Vue.ref(null), cover16Error = Vue.ref(''), cover16Loading = Vue.ref(false);
      let issueRequest = 0;
      const lossNames = {red_pool_miss:'红并集缺号',blue_selection_miss:'蓝球未入票',complete_red_set_miss:'六红未在同一注',blue_pairing_miss:'正确六红未配中奖蓝球'};
      const yes = value => value == null ? '待开奖' : value ? '是' : '否';
      const cover16Url = SSQApp.url('api/rank-board/coverage16');
      const sourceUrl = Vue.computed(() => detail.value?.pools?.source?.raw_file ? SSQApp.url('api/recommendations/files',{path:detail.value.pools.source.raw_file}) : '');
      const pad = (n) => String(n).padStart(2, '0');
      const decimal = (n) => (n == null || Number.isNaN(Number(n)) ? '—' : Number(n).toFixed(2));
      const roleName = (role) => ({main: '主排序', expert: '单项专家', trial: '试验', baseline: '均匀对照'}[role] || role);
      const kindName = (kind) => ({tier: '分档池', small: '小池', strategy: '方案并集'}[kind] || kind);
      const windows = [
        ['confirmation_25111_26115', '确认窗 25111–26115'],
        ['tracking_after_26115', '跟踪窗 26116 之后'],
        ['all_from_25111', '25111 之后全部']
      ];
      const selectedWindow = Vue.computed(() => overview.value?.windows[windowName.value]);
      const fiveWindow = Vue.computed(() => overview.value?.pool_five?.[windowName.value]);
      const fiveGenerateUrl = Vue.computed(() => SSQApp.url('', {
        mode: 'pool_five', issue: detail.value?.issue, expected_data_sha256: detail.value?.data_sha256,
      }));
      const fiveArchiveUrl = Vue.computed(() => detail.value?.pools?.items.some(row => row.model === 'pool_five')
        ? SSQApp.url('reviews', {issue: detail.value.issue, archive: 'daily', strategy: 'pool_five'}) : '');
      const rate = n => (n * 100).toFixed(2) + '%';
      function historyMean(name) {
        const row = (overview.value?.windows?.all_from_25111?.algorithms || []).find((item) => item.name === name);
        return row ? row.mean_cover_rank : Number.POSITIVE_INFINITY;
      }
      const algorithms = Vue.computed(() => {
        const rows = (detail.value?.algorithms || []).slice();
        const key = sort.value;
        if (key === 'cover' && detail.value?.settled) rows.sort((a, b) => (a.cover_all_six_rank ?? 99) - (b.cover_all_six_rank ?? 99));
        else if (key === 'top16' && detail.value?.settled) rows.sort((a, b) => (b.top16_hits ?? -1) - (a.top16_hits ?? -1) || (a.cover_all_six_rank ?? 99) - (b.cover_all_six_rank ?? 99));
        else if (key === 'history') rows.sort((a, b) => historyMean(a.name) - historyMean(b.name));
        return rows;
      });
      const pools = Vue.computed(() => {
        const query = poolSearch.value.trim().toLowerCase();
        const rows = (detail.value?.pools?.items || []).filter((row) => (poolKind.value === 'all' || row.kind === poolKind.value) && (!query || (kindName(row.kind) + row.label + row.model).toLowerCase().includes(query)));
        const key = poolSort.value;
        rows.sort((a, b) => {
          if (key === 'covered') return (b.red_hits ?? -1) - (a.red_hits ?? -1) || a.red_count - b.red_count;
          if (key === 'name') return a.label.localeCompare(b.label, 'zh');
          return a.red_count - b.red_count || a.label.localeCompare(b.label, 'zh');
        });
        return rows;
      });
      function poolKey(pool) {
        return [pool.kind, pool.model, pool.label, pool.reds.join(','), pool.blues.join(',')].join('|');
      }
      const selectedAlgo = Vue.computed(() => algorithms.value.find((row) => row.name === numberAlgo.value) || algorithms.value[0] || null);
      const openedPool = Vue.computed(() => pools.value.find((pool) => poolKey(pool) === openedKey.value) || pools.value[0] || null);
      const redBalls = Vue.computed(() => {
        if (!selectedAlgo.value) return [];
        const drawn = new Set(detail.value?.draw?.red || []);
        return selectedAlgo.value.red_ranking.map((number, index) => ({number, rank: index + 1, hit: drawn.has(number)}));
      });
      const blueBalls = Vue.computed(() => {
        if (!selectedAlgo.value) return [];
        const blue = detail.value?.draw?.blue;
        return selectedAlgo.value.blue_ranking.map((number, index) => ({number, rank: index + 1, hit: detail.value?.settled && number === blue}));
      });
      function redRanks(row) {
        if (!detail.value?.draw) return [];
        return detail.value.draw.red.map((number) => row.red_ranking.indexOf(number) + 1).filter((rank) => rank > 0);
      }
      function hitStyle(rank) {
        return {left: ((rank - 0.5) / 33 * 100) + '%'};
      }
      function head(ranking) {
        return (ranking || []).slice(0, Number(rankSpan.value) || 33);
      }
      function scaleLabel(index) {
        const shown = Number(rankSpan.value) || 33;
        return index === 1 || index % 10 === 0 || index === shown ? String(index) : '';
      }
      function rankClass(index, number, color) {
        const hit = color === 'blue' ? detail.value?.settled && number === detail.value.draw.blue : (detail.value?.draw?.red || []).includes(number);
        return ['rank-num', hit ? 'hit' : '', color === 'blue' ? 'blue' : '', index > 0 && index % 10 === 0 ? 'decade' : ''].filter(Boolean).join(' ');
      }
      function lineClass(row) {
        return [row.name === numberAlgo.value ? 'selected' : '', row.role === 'trial' ? 'trial' : ''].filter(Boolean).join(' ');
      }
      function blueLineClass(row) {
        return [row.name === blueAlgo.value ? 'selected' : '', row.role === 'trial' ? 'trial' : ''].filter(Boolean).join(' ');
      }
      function historyBlue(name) {
        const row = (overview.value?.blue_study?.confirmation?.algorithms || []).find((item) => item.name === name);
        return row ? row.mean_rank : Number.POSITIVE_INFINITY;
      }
      function blueHistory(name, field) {
        const row = (overview.value?.blue_study?.confirmation?.algorithms || []).find((item) => item.name === name);
        return row ? row[field] + '/' + row.n : '—';
      }
      const blueRows = Vue.computed(() => {
        const rows = (detail.value?.blue_rankings || []).slice();
        if (blueSort.value === 'rank' && detail.value?.settled) rows.sort((a, b) => (a.blue_rank ?? 99) - (b.blue_rank ?? 99));
        else if (blueSort.value === 'history') rows.sort((a, b) => historyBlue(a.name) - historyBlue(b.name));
        return rows;
      });
      function issueAt(offset) {
        const items = overview.value?.issues || [];
        const index = items.findIndex((row) => row.issue === issue.value);
        return index >= 0 ? items[index + offset] : null;
      }
      function rememberIssue() {
        const url = new URL(location.href);
        url.searchParams.set('pane', 'rankboard');
        if (issue.value) url.searchParams.set('issue', issue.value);
        history.replaceState(null, '', url);
      }
      async function loadIssue() {
        if (!issue.value) return;
        const id = ++issueRequest, wanted = issue.value;
        loading.value = true; detail.value = null;
        error.value = '';
        try {
          const response = await fetch(SSQApp.url('api/rank-board/' + wanted));
          const data = await response.json();
          if (id !== issueRequest || wanted !== issue.value) return;
          if (!response.ok) throw Error(data.error || '本期排序读取失败');
          if (data.issue !== wanted) throw Error('返回排序期号与所选期号不一致，请刷新');
          detail.value = data;
          openedKey.value = '';
          if (!data.settled && ['cover', 'top16'].includes(sort.value)) sort.value = 'group';
          if (!(data.algorithms || []).some((row) => row.name === numberAlgo.value)) numberAlgo.value = 'weighted';
          rememberIssue();
        } catch (e) {
          if (id === issueRequest) { detail.value = null; error.value = e.message; }
        } finally {
          if (id === issueRequest) loading.value = false;
        }
      }
      async function loadCover16(event) {
        if (!event.target.open || cover16.value || cover16Loading.value) return;
        cover16Loading.value = true; cover16Error.value = '';
        try { const r = await fetch(cover16Url), v = await r.json();
          if (!r.ok) throw Error(v.error || '16红目标核验失败');
          if (v.data_sha256 !== overview.value.data_sha256) throw Error('开奖数据已更新，请刷新号码排序页');
          cover16.value = v;
        } catch(e) {cover16Error.value = e.message;} finally {cover16Loading.value = false;}
      }
      function step(offset) {
        const next = issueAt(offset);
        if (!next || loading.value) return;
        issue.value = next.issue;
        loadIssue();
      }
      return {
        overview, detail, error, loading, issue, panel, windowName, sort, rankSpan, poolSort, poolKind, poolSearch,
        cover16, cover16Error, cover16Loading, loadCover16, cover16Url, lossNames, yes, sourceUrl,
        windows, selectedWindow, fiveWindow, fiveGenerateUrl, fiveArchiveUrl, rate, algorithms, pools, pad, decimal, roleName, kindName, loadIssue, step, issueAt,
        numberAlgo, blueAlgo, blueSort, blueRows, openedKey, selectedAlgo, redBalls, blueBalls, openedPool, poolKey, redRanks, hitStyle, head,
        scaleLabel, rankClass, lineClass, blueLineClass, historyBlue, blueHistory, historyMean
      };
    },
    async mounted() {
      try {
        const response = await fetch(SSQApp.url('api/rank-board'));
        const data = await response.json();
        if (!response.ok) throw Error(data.error || '排序汇总读取失败');
        this.overview = data;
        const wanted = new URLSearchParams(location.search).get('issue');
        const known = data.issues.some((row) => row.issue === wanted);
        this.issue = known ? wanted : (data.issues[0]?.issue || '');
        if (location.hash === 'frozen-pools') this.panel = 'pools';
        else if (location.hash === 'history-summary') this.panel = 'study';
        await this.loadIssue();
      } catch (e) {
        this.error = e.message;
      }
    },
    template: `<section class="ssq-rank" aria-label="号码池与排序">
      <h2>号码池与排序</h2>
      <p class="muted">先看本期排序，再按需打开冻结号码池或历史验证。一行从左到右是第1名起的具体号码。盖住全部6个红球的名次写在分布图右侧，均匀随机排序的期望是前29.14名。这些对照不是中奖概率，也不替换已封存方案。</p>
      <p v-if="error" class="notice error" role="alert">{{ error }}</p>
      <p v-if="!overview&&!error" class="muted" role="status">正在按冻结权重回放确认窗之后的排序…</p>
      <template v-if="overview">
        <p v-if="overview.cover_rank" class="notice" id="cover-rank-note">{{ overview.cover_rank.note }}</p>
        <div class="toolbar" id="issue-picker">
          <label for="rank-issue">期号<select id="rank-issue" v-model="issue" @change="loadIssue" :disabled="loading"><option v-for="row in overview.issues" :key="row.issue" :value="row.issue">{{ row.issue }}{{ row.settled?'':' · 待开奖' }}</option></select></label>
          <button type="button" @click="step(1)" :disabled="loading||!issueAt(1)">上一期</button>
          <button type="button" @click="step(-1)" :disabled="loading||!issueAt(-1)">下一期</button>
          <span v-if="loading" class="muted" role="status">正在读取第 {{ issue }} 期…</span>
        </div>
        <p v-if="detail" class="muted">第 {{ detail.issue }} 期 · {{ detail.settled ? '已开奖' : '待开奖，开奖号位置在开奖后显示' }}<template v-if="detail.date"> · {{ detail.date }}</template></p>
        <div v-if="detail?.draw" class="draw" id="draw-numbers" aria-label="开奖号码"><span v-for="n in detail.draw.red" :key="n">{{ pad(n) }}</span><span class="blue">{{ pad(detail.draw.blue) }}</span></div>
        <div class="k8-tabs" role="tablist" aria-label="页面分区">
          <button type="button" role="tab" id="tab-rank" :aria-selected="panel==='rank'" @click="panel='rank'">本期排序</button>
          <button type="button" role="tab" id="tab-pools" :aria-selected="panel==='pools'" @click="panel='pools'">冻结号码池<template v-if="detail?.pools?.items?.length"> · {{ detail.pools.items.length }}</template></button>
          <button type="button" role="tab" id="tab-study" :aria-selected="panel==='study'" @click="panel='study'">历史与验证</button>
        </div>

        <template v-if="panel==='rank' && detail">
          <section v-if="detail.pool_five" class="inner-card" id="pool-five-ranking">
            <div class="toolbar"><h3>号码池分散推荐 · 5注10元</h3><a v-if="fiveArchiveUrl" :href="fiveArchiveUrl">查看首次封存</a><a v-if="!detail.settled" class="primary" :href="fiveGenerateUrl">生成并复制票面</a></div>
            <p class="muted">原加权榜圈定30红，历史拟合重排池内红球并选前5蓝。每个红球只用一次，每注配一个不同蓝球。均匀开奖下，30红并集覆盖全部6红的概率为 {{ rate(detail.pool_five.uniform.full_red_union) }}。</p>
            <h4>入池红球 · 按名次排列</h4>
            <div class="rank-strip" aria-label="前30名入池红球"><span v-for="(n,i) in detail.pool_five.red_ranking.slice(0,30)" :key="n" :title="'第'+(i+1)+'名'" :class="{hit:detail.draw?.red.includes(n)}">{{ pad(n) }}</span></div>
            <p class="muted">未入池红球 {{ detail.pool_five.excluded_reds.map(pad).join('、') }} · 前5蓝 {{ detail.pool_five.blue_ranking.slice(0,5).map(pad).join('、') }}</p>
            <p v-if="detail.settled && detail.pool_five.outcome" class="muted">回放命中 {{ detail.pool_five.outcome.red_hits }}/6 红 · 蓝球{{ detail.pool_five.outcome.blue_covered?'入池':'未入池' }} · 最佳单注 {{ detail.pool_five.outcome.max_ticket_red_hits }} 红 · 常规奖{{ detail.pool_five.outcome.any_regular_prize?'有':'无' }}</p>
            <details><summary>{{ detail.settled?'查看回放票面':'查看本期5注' }}</summary>
              <ol class="reference-tickets"><li v-for="ticket in detail.pool_five.tickets" :key="ticket.blue">{{ ticket.red.map(pad).join(' ') }} + {{ pad(ticket.blue) }}</li></ol>
            </details>
          </section>
          <p v-if="overview.equal_rank" class="notice" id="equal-rank-note">{{ overview.equal_rank.note }}</p>
          <p v-if="overview.historical_fit" class="notice" id="historical-fit-note">{{ overview.historical_fit.note }}</p>
          <div class="toolbar">
            <label for="rank-sort">排序方式<select id="rank-sort" v-model="sort"><option value="group">按算法分组</option><template v-if="detail.settled"><option value="cover">本期盖住6红从少到多</option><option value="top16">本期前16名命中从多到少</option></template><option value="history">历史平均盖住名次从少到多</option></select></label>
            <label for="rank-span">红球显示<select id="rank-span" v-model.number="rankSpan"><option :value="16">前16名</option><option :value="22">前22名</option><option :value="33">全部33名</option></select></label>
          </div>
          <section v-if="detail.settled" class="inner-card" id="draw-distribution">
            <h3>开奖红球在各排序中的位置</h3>
            <p class="muted">每行从左到右是第1名到第33名，红点是6个开奖红球，灰线是第11、16、22名，右侧是盖住全部红球的名次。</p>
            <div class="dist-head" aria-hidden="true"><span></span><span class="dist-ticks"><span>1</span><span>16</span><span>33</span></span><span>全覆盖</span></div>
            <button type="button" v-for="row in algorithms" :key="'dist-'+row.name" class="dist-row" :class="lineClass(row)" @click="numberAlgo=row.name" :title="row.label">
              <span class="dist-name">{{ row.label }}</span>
              <span class="dist-track"><i class="dist-guide" style="left:33.333%"></i><i class="dist-guide" style="left:48.485%"></i><i class="dist-guide" style="left:66.667%"></i><i v-for="rank in redRanks(row)" :key="rank" class="dist-hit" :style="hitStyle(rank)"></i></span>
              <span class="dist-need">{{ row.cover_all_six_rank }}</span>
            </button>
          </section>
          <section class="inner-card" id="ranked-numbers">
            <h3>各排序后的红球</h3>
            <p class="muted">从左到右是第1名起的具体红球。已开奖时红色为开奖号，每10名有一条分隔。左右滑动可看后面的名次，排序名保持在左侧。</p>
            <div class="rank-board">
              <div class="rank-line rank-scale" aria-hidden="true"><span class="rank-name"></span><span class="rank-nums"><span v-for="i in rankSpan" :key="'scale-'+i" class="rank-num">{{ scaleLabel(i) }}</span></span></div>
              <button type="button" v-for="row in algorithms" :key="'red-'+row.name" class="rank-line" :class="lineClass(row)" @click="numberAlgo=row.name" :title="row.label">
                <span class="rank-name">{{ row.label }}</span>
                <span class="rank-nums"><span v-for="(n,i) in head(row.red_ranking)" :key="row.name+'-'+n" :class="rankClass(i,n,'red')">{{ pad(n) }}</span></span>
              </button>
            </div>
            <h3>各算法的蓝球</h3>
            <p v-if="overview.blue_study" class="notice" id="blue-rank-note">{{ overview.blue_study.note }}</p>
            <p class="muted">每条是一套独立的16个蓝球名次，从左到右是第1名起。已开奖时蓝色为开奖蓝球。前2、前4表示开奖蓝球是否落在这个范围内。</p>
            <div class="toolbar">
              <label for="blue-sort">蓝球排序<select id="blue-sort" v-model="blueSort"><option value="group">按算法分组</option><option v-if="detail.settled" value="rank">本期名次从少到多</option><option value="history">确认窗平均名次从少到多</option></select></label>
            </div>
            <div class="rank-board" id="blue-rankings">
              <div class="rank-line rank-scale" aria-hidden="true"><span class="rank-name"></span><span class="rank-nums"><span v-for="i in 16" :key="'bscale-'+i" class="rank-num">{{ i===1||i===8||i===16?i:'' }}</span></span></div>
              <button type="button" v-for="row in blueRows" :key="'blue-'+row.name" class="rank-line" :class="blueLineClass(row)" @click="blueAlgo=row.name" :title="row.label">
                <span class="rank-name">{{ row.label }}</span>
                <span class="rank-nums"><span v-for="(n,i) in row.blue_ranking" :key="row.name+'-b'+n" :class="rankClass(i,n,'blue')">{{ pad(n) }}</span></span>
              </button>
            </div>
            <div class="scroll"><table id="blue-coverage">
              <thead><tr><th>蓝球排序</th><th>角色</th><th v-if="detail.settled">本期名次</th><th v-if="detail.settled">前2</th><th v-if="detail.settled">前4</th><th>确认窗平均名次</th><th>确认窗前2</th><th>确认窗前4</th></tr></thead>
              <tbody>
                <tr v-for="row in blueRows" :key="'bcov-'+row.name" :class="{selected:row.name===blueAlgo}" @click="blueAlgo=row.name"><td>{{ row.label }}</td><td>{{ roleName(row.role) }}</td><td v-if="detail.settled" class="num">第{{ row.blue_rank }}名</td><td v-if="detail.settled">{{ row.top2_hit?'是':'否' }}</td><td v-if="detail.settled">{{ row.top4_hit?'是':'否' }}</td><td class="num">{{ decimal(historyBlue(row.name)) }}</td><td class="num">{{ blueHistory(row.name,'top2') }}</td><td class="num">{{ blueHistory(row.name,'top4') }}</td></tr>
                <tr class="muted"><td>均匀理论期望</td><td></td><td v-if="detail.settled" class="num">8.50</td><td v-if="detail.settled">12.5%</td><td v-if="detail.settled">25%</td><td class="num">8.50</td><td class="num">{{ decimal((overview.blue_study?.confirmation?.n||0)*2/16) }}</td><td class="num">{{ decimal((overview.blue_study?.confirmation?.n||0)*4/16) }}</td></tr>
              </tbody>
            </table></div>
          </section>
          <details v-if="selectedAlgo" class="inner-card" id="ranking-detail">
            <summary>查看{{ selectedAlgo.label }}的33个红球和16个蓝球</summary>
            <p class="muted">方块上方是号码，下方是名次，顺序就是该排序。红底或蓝底是开奖号码。</p>
            <div class="number-grid" id="red-number-grid"><span v-for="ball in redBalls" :key="'tile-r'+ball.number" class="number-tile" :class="{hit:ball.hit}">{{ pad(ball.number) }}<small>第{{ ball.rank }}</small></span></div>
            <div class="number-grid blue" id="blue-number-grid"><span v-for="ball in blueBalls" :key="'tile-b'+ball.number" class="number-tile" :class="{hit:ball.hit}">{{ pad(ball.number) }}<small>第{{ ball.rank }}</small></span></div>
          </details>
          <details v-if="detail.settled" class="inner-card" id="algorithm-coverage">
            <summary>覆盖数字对照</summary>
            <div class="scroll"><table>
              <thead><tr><th>排序</th><th>角色</th><th>盖住6红</th><th>前16名命中</th><th>蓝球名次</th><th>历史平均盖住</th></tr></thead>
              <tbody>
                <tr v-for="row in algorithms" :key="'cov-'+row.name" :class="{selected:row.name===numberAlgo}" @click="numberAlgo=row.name"><td>{{ row.label }}</td><td>{{ roleName(row.role) }}</td><td class="num">第{{ row.cover_all_six_rank }}名</td><td class="num">{{ row.top16_hits }} / 6</td><td class="num">第{{ row.blue_rank }}名</td><td class="num">{{ decimal(historyMean(row.name)) }}</td></tr>
                <tr class="muted"><td>均匀理论期望</td><td></td><td class="num">29.14</td><td class="num">2.91 / 6</td><td class="num">8.50</td><td class="num">29.14</td></tr>
              </tbody>
            </table></div>
          </details>
        </template>

        <template v-if="panel==='pools'">
          <section class="inner-card" id="frozen-pools">
            <h3>本期实际冻结号码池</h3>
            <p v-if="!detail" class="muted">正在读取本期档案…</p>
            <p v-else-if="!detail.pools?.available" class="notice">{{ detail.pools?.note }}</p>
            <template v-else>
              <p v-if="detail.pools.source" class="muted" id="pool-source">首次每日原档：{{detail.pools.source.issue}}期，输入截至{{detail.pools.source.data_cutoff}}。<a :href="sourceUrl">核对首次原档JSON</a></p>
              <p v-if="detail.pools.strategy_funnel" class="muted" id="pool-ticket-funnel">本期{{detail.pools.strategy_funnel.strategy_count}}个方案选项<template v-if="detail.settled">：红并集盖住六红{{detail.pools.strategy_funnel.red_union_full_six_strategies}}种，真实单注六红{{detail.pools.strategy_funnel.single_six_red_strategies}}种，真实单注6＋1头奖{{detail.pools.strategy_funnel.single_jackpot_strategies}}种</template><template v-else>，待开奖</template>。同一期方案不合并为多期成绩。</p>
              <div class="toolbar">
                <label for="pool-kind">类型<select id="pool-kind" v-model="poolKind"><option value="all">全部</option><option value="tier">分档池</option><option value="small">小池</option><option value="strategy">方案并集</option></select></label>
                <label for="pool-search">筛选<input id="pool-search" v-model="poolSearch" type="search" placeholder="方案名"></label>
                <label for="pool-sort">排序<select id="pool-sort" v-model="poolSort"><option value="size">红球数从少到多</option><option value="covered" v-if="detail.settled">覆盖开奖红球从多到少</option><option value="name">方案名</option></select></label>
              </div>
              <p class="muted">共 {{ pools.length }} 个号码池。点选一行查看号码，红球按从小到大排列。</p>
              <div class="scroll"><table id="pool-list">
                <thead><tr><th>类型</th><th>方案</th><th>红球</th><th>蓝球</th><th v-if="detail.settled">红并集中红</th><th v-if="detail.settled">蓝并集命中</th><th v-if="detail.settled">最佳单注红</th><th v-if="detail.settled">单注6＋1</th></tr></thead>
                <tbody><tr v-for="pool in pools" :data-pool-model="pool.model" :key="poolKey(pool)" :class="{selected:openedPool && poolKey(openedPool)===poolKey(pool)}" @click="openedKey=poolKey(pool)"><td>{{ kindName(pool.kind) }}</td><td>{{ pool.label }}</td><td class="num">{{ pool.red_count }}</td><td class="num">{{ pool.blue_count||'—' }}</td><td v-if="detail.settled" class="num">{{ pool.red_hits }} / 6</td><td v-if="detail.settled">{{ pool.blue_hit==null?'—':pool.blue_hit?'是':'否' }}</td><td v-if="detail.settled">{{pool.ticket_audit?pool.ticket_audit.max_ticket_red_hits+' / 6':'仅候选池'}}</td><td v-if="detail.settled">{{pool.ticket_audit?yes(pool.ticket_audit.single_jackpot):'—'}}</td></tr></tbody>
              </table></div>
              <article v-if="openedPool" class="opened-pool" id="opened-pool">
                <h3>{{ openedPool.label }}</h3>
                <p class="muted">{{ kindName(openedPool.kind) }} · {{ openedPool.red_count }} 个红球<template v-if="openedPool.blue_count"> · {{ openedPool.blue_count }} 个蓝球</template><template v-if="detail.settled"> · 覆盖开奖红球 <strong>{{ openedPool.red_hits }}</strong> / 6</template></p>
                <div class="pool-balls" aria-label="号码池红球"><span v-for="n in openedPool.reds" :key="'pool-red-'+n" class="pool-ball" :class="{hit:detail.draw && detail.draw.red.includes(n)}">{{ pad(n) }}</span></div>
                <div v-if="openedPool.blues.length" class="pool-balls" aria-label="号码池蓝球"><span v-for="n in openedPool.blues" :key="'pool-blue-'+n" class="pool-ball blue" :class="{hit:detail.draw && n===detail.draw.blue}">{{ pad(n) }}</span></div>
                <div v-if="openedPool.ticket_audit" id="opened-ticket-audit"><p>首次保存{{openedPool.ticket_audit.bets}}注，共{{openedPool.ticket_audit.cost}}元。单注最多中红{{openedPool.ticket_audit.max_ticket_red_hits==null?'待开奖':openedPool.ticket_audit.max_ticket_red_hits+' / 6'}}；单注完整六红：{{yes(openedPool.ticket_audit.single_six_red)}}；同一注6＋1头奖：{{yes(openedPool.ticket_audit.single_jackpot)}}。</p><p v-if="detail.settled&&openedPool.ticket_audit.losses.length" class="muted">漏因：{{openedPool.ticket_audit.losses.map(k=>lossNames[k]||k).join('；')}}。</p></div>
                <p v-else class="muted" id="candidate-only">本行只提供候选号码池，没有已保存的单注票面；并集覆盖不记作单注命中。</p>
              </article>
            </template>
          </section>
        </template>

        <template v-if="panel==='study'">
          <section v-if="fiveWindow" class="inner-card" id="pool-five-history">
            <h3>同预算5注 · 历史回放</h3>
            <p class="muted">按下方验证窗口统计。新旧方案均覆盖30红。全6红入池容易受池大小影响，逐注命中及常规奖按实际票面计算。</p>
            <div class="scroll"><table><thead><tr><th>方案</th><th>全6红入池</th><th>蓝球入池</th><th>6红＋蓝均入池</th><th>有单注4红以上</th><th>至少一注常规奖</th><th>单注6＋1</th></tr></thead><tbody>
              <tr v-for="[label,row] in [['原加权30红 · 拟合5蓝',fiveWindow.ranked],['原加权30红 · 2蓝',fiveWindow.legacy]]" :key="label"><th>{{ label }}</th><td>{{ row.full_red_union }}/{{ fiveWindow.n }}</td><td>{{ row.blue_union }}</td><td>{{ row.full_draw_union }}</td><td>{{ row.any_ticket_four_reds }}</td><td>{{ row.any_regular_prize }}</td><td>{{ row.jackpot }}</td></tr>
            </tbody></table></div>
            <p v-if="fiveWindow.n" class="muted">若直接用历史拟合榜取前30红，全6红入池 {{ fiveWindow.fit_only_red_union.full_red_union }}/{{ fiveWindow.n }}，相对原加权补中 {{ fiveWindow.fit_only_red_union.paired_gains }} 期、丢失 {{ fiveWindow.fit_only_red_union.paired_losses }} 期。均匀开奖下，30红并集全中6红的概率为 {{ rate(fiveWindow.ranked.uniform.full_red_union) }}。</p>
            <p v-else class="muted">所选窗口暂无已开奖样本。</p>
          </section>
          <section class="inner-card" id="history-summary">
            <h3>历史汇总</h3>
            <details id="direct-cover16" @toggle="loadCover16"><summary>16红全六覆盖 · 直接目标核验</summary>
             <p v-if="cover16Loading" class="muted" role="status">正在核对完整六红分布和16红覆盖目标…</p><p v-if="cover16Error" class="notice error" role="alert">{{cover16Error}}</p>
             <template v-if="cover16"><p class="notice">{{cover16.reason}}</p><p class="muted">验证截至{{cover16.cutoff_issue}}，只用23100及以前拟合。先在全部33红上建立16红潜在池，再在池内选六红，完整结果精确归一化；筛选直接计算全部六红同时入16红的模型质量。均匀分量{{(cover16.weights[7]*100).toFixed(4)}}%。</p>
              <div class="scroll"><table id="cover16-validation"><thead><tr><th>探索窗口</th><th>期数</th><th>完整六红对数评分差</th><th>边际16红全六次数</th><th>直接目标全六次数</th><th>组内同分期望</th></tr></thead><tbody><tr v-for="(r,k) in cover16.windows" :key="k"><td>{{({validation:'预留',confirmation_exploratory:'确认探索',tracking:'跟踪'})[k]}}</td><td>{{r.n}}</td><td>{{r.mean_log_excess==null?'—':r.mean_log_excess.toFixed(6)}}</td><td>{{r.marginal_full_six}}</td><td>{{r.search_full_six}}</td><td>{{r.search_tie_expected.toFixed(3)}}</td></tr></tbody></table></div>
              <p class="muted">同分期望是等价组内均匀选择的覆盖分数，不是中奖次数；这些回放不是历史融合ML首次票面。没有发布新号码池、票面或新前瞻。</p><p class="muted" id="cover16-certificate">当前模型搜索{{cover16.current.evaluations}}次／{{cover16.policy.candidate_budget}}次预算；{{cover16.current.globally_optimal_within_tolerance?'模型内上界差距已闭合':'只报告单交换搜索及剩余上界，不宣称全局最优'}}。训练权重对偶差距{{cover16.fit_gap.toExponential(2)}}。求解正确不代表真实预测有效。</p><a :href="cover16Url">核对完整目标与搜索证书JSON</a>
             </template>
            </details>
            <div class="toolbar">
              <label for="rank-window">历史窗口<select id="rank-window" v-model="windowName"><option v-for="item in windows" :key="item[0]" :value="item[0]">{{ item[1] }}</option></select></label>
              <span class="muted">{{ selectedWindow?.n || 0 }} 期</span>
            </div>
            <p class="muted">平均需要前多少名才盖住6个红球。对照期望一列是和29.14的差，负数表示平均更靠前。这是已查看历史的回放，不是未来中奖概率。</p>
            <div class="scroll"><table>
              <thead><tr><th>排序</th><th>角色</th><th>平均盖住6红</th><th>对照期望</th><th>前16平均命中</th><th>前16盖住6红</th><th>平均蓝球名次</th></tr></thead>
              <tbody><tr v-for="row in selectedWindow?.algorithms||[]" :key="row.name" :class="{selected:row.name===numberAlgo}" @click="numberAlgo=row.name;panel='rank'"><td>{{ row.label }}</td><td>{{ roleName(row.role) }}</td><td class="num">{{ decimal(row.mean_cover_rank) }}</td><td class="num">{{ decimal(row.mean_cover_rank-overview.uniform_expected_cover_rank) }}</td><td class="num">{{ decimal(row.mean_top16_hits) }}</td><td class="num">{{ row.top16_covers_all_six }}/{{ row.n }}</td><td class="num">{{ decimal(row.mean_blue_rank) }}</td></tr></tbody>
            </table></div>
          </section>
        </template>
      </template>
    </section>`
  };
  global.SSQRankBoard = {component};
})(window);
