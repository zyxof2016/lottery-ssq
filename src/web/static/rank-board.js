(function (global) {
  'use strict';
  const component = {
    setup() {
      const overview = Vue.ref(null);
      const detail = Vue.ref(null);
      const error = Vue.ref('');
      const loading = Vue.ref(false);
      const issue = Vue.ref('');
      const windowName = Vue.ref('confirmation_25111_26115');
      const prefix = Vue.ref(16);
      const sort = Vue.ref('cover');
      const poolSort = Vue.ref('hits');
      const poolKind = Vue.ref('all');
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
      const algorithms = Vue.computed(() => {
        const rows = (detail.value?.algorithms || []).slice();
        const key = sort.value;
        rows.sort((a, b) => {
          if (key === 'top16') return (b.top16_hits ?? -1) - (a.top16_hits ?? -1);
          if (key === 'name') return a.label.localeCompare(b.label, 'zh');
          return (a.cover_all_six_rank ?? 99) - (b.cover_all_six_rank ?? 99);
        });
        return rows;
      });
      const pools = Vue.computed(() => {
        const rows = (detail.value?.pools?.items || []).filter((row) => poolKind.value === 'all' || row.kind === poolKind.value);
        const key = poolSort.value;
        rows.sort((a, b) => {
          if (key === 'size') return a.red_count - b.red_count || (b.red_hits ?? -1) - (a.red_hits ?? -1);
          if (key === 'name') return a.label.localeCompare(b.label, 'zh');
          return (b.red_hits ?? -1) - (a.red_hits ?? -1) || a.red_count - b.red_count;
        });
        return rows;
      });
      function cellClass(ranking, number, index) {
        const drawn = detail.value?.draw?.red || [];
        const winner = drawn.includes(number);
        if (winner && index < prefix.value) return 'hit';
        if (winner) return 'miss';
        if (index < prefix.value) return 'inside';
        return '';
      }
      function blueClass(ranking, number, index) {
        if (!detail.value?.settled) return index < 2 ? 'inside' : '';
        if (number === detail.value.draw.blue && index < 2) return 'hit';
        if (number === detail.value.draw.blue) return 'miss';
        return index < 2 ? 'inside' : '';
      }
      async function loadIssue() {
        if (!issue.value) return;
        loading.value = true;
        error.value = '';
        try {
          const response = await fetch(SSQApp.url('api/rank-board/' + issue.value));
          const data = await response.json();
          if (!response.ok) throw Error(data.error || '本期排序读取失败');
          detail.value = data;
        } catch (e) {
          detail.value = null;
          error.value = e.message;
        } finally {
          loading.value = false;
        }
      }
      return {
        overview, detail, error, loading, issue, windowName, prefix, sort, poolSort, poolKind,
        windows, selectedWindow, algorithms, pools, pad, decimal, roleName, kindName, cellClass, blueClass, loadIssue
      };
    },
    async mounted() {
      try {
        const response = await fetch(SSQApp.url('api/rank-board'));
        const data = await response.json();
        if (!response.ok) throw Error(data.error || '排序汇总读取失败');
        this.overview = data;
        const settled = data.issues.find((row) => row.settled);
        this.issue = settled?.issue || data.next_issue;
        await this.loadIssue();
      } catch (e) {
        this.error = e.message;
      }
    },
    template: `<section class="card rank-board" aria-label="号码池与排序">
      <h2>号码池与排序</h2>
      <p class="muted">每一行是开奖前就确定的 33 个红球完整名次。盖住全部 6 个红球的名次，等于其中最靠后的那个红球。均匀随机顺序的期望名次是 29.14，前 16 名期望命中 2.91 个，蓝球期望名次是 8.5。这些数字只用来对照，不是中奖概率，也不替换已封存方案。</p>
      <p v-if="error" class="notice error" role="alert">{{ error }}</p>
      <p v-if="!overview&&!error" class="muted" role="status">正在按冻结权重回放确认窗之后的排序…</p>
      <template v-if="overview">
        <div class="toolbar">
          <label for="rank-window">历史窗口</label>
          <select id="rank-window" v-model="windowName"><option v-for="w in windows" :key="w[0]" :value="w[0]">{{ w[1] }}</option></select>
          <span class="muted">{{ selectedWindow?.n || 0 }} 期</span>
        </div>
        <div class="scroll"><table class="rank-summary">
          <thead><tr><th>排序</th><th>角色</th><th>平均盖住6红</th><th>对照期望</th><th>前16平均命中</th><th>前16盖住6红</th><th>平均蓝球名次</th></tr></thead>
          <tbody><tr v-for="row in selectedWindow?.algorithms||[]" :key="row.name"><td>{{ row.label }}</td><td>{{ roleName(row.role) }}</td><td>{{ decimal(row.mean_cover_rank) }}</td><td>{{ decimal(row.mean_cover_rank-overview.uniform_expected_cover_rank) }}</td><td>{{ decimal(row.mean_top16_hits) }}</td><td>{{ row.top16_covers_all_six }}/{{ row.n }}</td><td>{{ decimal(row.mean_blue_rank) }}</td></tr></tbody>
        </table></div>
        <div class="toolbar">
          <label for="rank-issue">期号</label>
          <select id="rank-issue" v-model="issue" @change="loadIssue"><option v-for="row in overview.issues" :key="row.issue" :value="row.issue">{{ row.issue }}{{ row.settled?'':' · 待开奖' }}</option></select>
          <label for="rank-prefix">前缀</label>
          <input id="rank-prefix" v-model.number="prefix" type="range" min="6" max="33">
          <span>{{ prefix }} 名</span>
          <label for="rank-sort">算法排序</label>
          <select id="rank-sort" v-model="sort"><option value="cover">按盖住6红的名次</option><option value="top16">按前16命中</option><option value="name">按名称</option></select>
        </div>
        <p v-if="loading" class="muted" role="status">正在读取第 {{ issue }} 期…</p>
        <p v-if="detail&&!detail.settled" class="notice">第 {{ detail.issue }} 期尚未开奖。下面是开奖前排序，命中要等开奖后才计算。</p>
        <p v-else-if="detail?.draw" class="balls" aria-label="当期开奖"><span v-for="n in detail.draw.red" :key="n" class="ball">{{ pad(n) }}</span><span class="ball blue">{{ pad(detail.draw.blue) }}</span></p>
        <article v-for="row in algorithms" :key="row.name" class="rank-algorithm">
          <div class="toolbar"><h3>{{ row.label }}</h3><span class="muted">{{ roleName(row.role) }} · 蓝球用{{ row.blue_source }}</span><strong v-if="detail?.settled">盖住6红：第 {{ row.cover_all_six_rank }} 名 · 前16命中 {{ row.top16_hits }} · 蓝球第 {{ row.blue_rank }} 名</strong></div>
          <div class="rank-strip" :aria-label="row.label+'红球名次'">
            <span v-for="(n,i) in row.red_ranking" :key="row.name+'-'+n" :class="cellClass(row.red_ranking,n,i)" :title="'第'+(i+1)+'名'">{{ pad(n) }}</span>
          </div>
          <div class="rank-strip blue" :aria-label="row.label+'蓝球名次'">
            <span v-for="(n,i) in row.blue_ranking" :key="row.name+'-b'+n" :class="blueClass(row.blue_ranking,n,i)" :title="'蓝球第'+(i+1)+'名'">{{ pad(n) }}</span>
          </div>
        </article>
        <h3>已冻结号码池</h3>
        <p class="muted">{{ detail?.pools?.note }}</p>
        <div v-if="detail?.pools?.available" class="toolbar">
          <label for="pool-kind">池类型</label>
          <select id="pool-kind" v-model="poolKind"><option value="all">全部</option><option value="tier">分档池</option><option value="small">小池</option><option value="strategy">方案并集</option></select>
          <label for="pool-sort">号码池排序</label>
          <select id="pool-sort" v-model="poolSort"><option value="hits">按红球命中</option><option value="size">按红球数量</option><option value="name">按名称</option></select>
        </div>
        <div v-if="detail?.pools?.available" class="scroll"><table>
          <thead><tr><th>类型</th><th>号码池</th><th>红球</th><th>蓝球</th><th>红球命中</th><th>蓝球命中</th><th>号码</th></tr></thead>
          <tbody><tr v-for="(pool,i) in pools" :key="pool.kind+pool.model+pool.label+i"><td>{{ kindName(pool.kind) }}</td><td>{{ pool.label }}</td><td>{{ pool.red_count }}</td><td>{{ pool.blue_count||'—' }}</td><td>{{ pool.red_hits==null?'—':pool.red_hits }}</td><td>{{ pool.blue_hit==null?'—':pool.blue_hit?'是':'否' }}</td><td>{{ pool.reds.map(pad).join(' ') }}<template v-if="pool.blues.length"> · 蓝 {{ pool.blues.map(pad).join(' ') }}</template></td></tr></tbody>
        </table></div>
      </template>
    </section>`
  };
  global.SSQRankBoard = {component};
})(window);
