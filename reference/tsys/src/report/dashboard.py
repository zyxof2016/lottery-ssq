from __future__ import annotations

import errno
import json
import logging
import socket
import sys
import threading
import webbrowser
from datetime import datetime, timezone
from email import policy
from email.parser import BytesParser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlparse

from src.data_handler.versioning import json_default


WEB_PAGE = r"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>概率实验系统</title>
  <style>
    :root{--bg:#f4f6f8;--surface:#fff;--line:#d9dee5;--text:#17212b;--muted:#66717e;--red:#c62d43;--red-dark:#9f2034;--green:#16845b;--amber:#a66a00;--nav:#202a34}
    *{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font:14px/1.55 "Microsoft YaHei",Arial,sans-serif;letter-spacing:0}
    header{background:var(--nav);color:#fff;border-top:4px solid var(--red)}
    .header-inner,.nav-inner,main{width:min(1180px,calc(100% - 32px));margin:auto}.header-inner{height:70px;display:flex;align-items:center;justify-content:space-between;gap:20px}
    h1{font-size:20px;margin:0;font-weight:700}.subtitle{color:#c8d0d8;font-size:13px;margin-top:3px}.header-stat{text-align:right}.header-stat strong{font-size:22px}.header-stat span{display:block;color:#c8d0d8;font-size:12px}
    nav{background:var(--surface);border-bottom:1px solid var(--line)}.nav-inner{display:flex;gap:2px;overflow-x:auto}.tab{border:0;background:transparent;padding:15px 20px;color:#46515d;cursor:pointer;border-bottom:3px solid transparent;white-space:nowrap}.tab.active{color:var(--red);border-color:var(--red);font-weight:700}
    main{padding:20px 0 36px}.notice{border-left:3px solid var(--amber);background:#fff8e7;padding:11px 14px;color:#714b00;margin-bottom:20px}.view{display:none}.view.active{display:block}
    .section-head{display:flex;justify-content:space-between;align-items:flex-end;gap:16px;margin:0 0 14px}.section-head h2{font-size:21px;margin:0}.section-head p{margin:2px 0 0;color:var(--muted)}
    .band{background:var(--surface);border:1px solid var(--line);padding:18px;margin-bottom:16px}.band h3{font-size:15px;margin:0 0 14px}.grid{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px}.metric{border:1px solid var(--line);border-radius:6px;padding:13px;background:#fff}.metric span{display:block;color:var(--muted);font-size:12px}.metric strong{display:block;font-size:20px;margin-top:3px;overflow-wrap:anywhere}
    .form-row{display:flex;align-items:end;gap:12px;flex-wrap:wrap}.field{display:flex;flex-direction:column;gap:5px;min-width:145px}.field.grow{flex:1;min-width:260px}label{font-size:12px;color:var(--muted)}input,select{height:39px;border:1px solid #b9c1ca;background:#fff;padding:0 10px;font:inherit;border-radius:4px;min-width:0}input:focus,select:focus{outline:2px solid #efb8c0;border-color:var(--red)}
    button.action{height:39px;border:1px solid var(--red);background:var(--red);color:#fff;padding:0 17px;font:inherit;font-weight:700;border-radius:4px;cursor:pointer}button.action:hover{background:var(--red-dark)}button.action:disabled{background:#aeb5bd;border-color:#aeb5bd;cursor:not-allowed}.secondary{background:#fff!important;color:var(--text)!important;border-color:#aeb5bd!important}
    .status-line{display:flex;align-items:center;gap:9px;margin-top:14px;color:var(--muted)}.dot{width:9px;height:9px;border-radius:50%;background:#8b949e}.dot.running{background:var(--amber)}.dot.success{background:var(--green)}.dot.error{background:var(--red)}
    .table-wrap{overflow-x:auto;border:1px solid var(--line)}table{width:100%;border-collapse:collapse;background:#fff;min-width:720px}th,td{text-align:left;padding:10px 12px;border-bottom:1px solid var(--line);white-space:nowrap}th{font-size:12px;color:#596572;background:#f7f8fa}tbody tr:last-child td{border-bottom:0}.positive{color:var(--green);font-weight:700}.negative{color:var(--red);font-weight:700}
    .research-hidden{display:none!important}
    .chart-grid{display:grid;grid-template-columns:1fr 1fr;gap:16px}.chart-grid img{display:block;width:100%;border:1px solid var(--line);background:#fff;min-height:240px;object-fit:contain}
    .empty{padding:28px;text-align:center;color:var(--muted)}.explain{border-left:3px solid var(--green);padding:10px 12px;background:#eef9f4;margin-top:14px}.error-box{border-left-color:var(--red);background:#fff0f2;color:#8d1e2f}.number-set{font:700 14px Consolas,monospace}.red-number{color:var(--red)}.blue-number{color:#1468a8}.matched-red-number{color:#1468a8;font-weight:900;text-decoration:underline;text-underline-offset:3px}.matched-blue-number{display:inline-block;color:#fff;background:#1468a8;border-radius:50%;min-width:22px;text-align:center}
    .result-option{display:flex;align-items:center;gap:8px;color:var(--text);font-size:13px;cursor:pointer}.result-option input{width:18px;height:18px;min-width:18px;margin:0;accent-color:var(--red)}.logic-grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:18px}.logic-grid section{border-left:3px solid var(--line);padding-left:13px}.logic-grid h4{margin:0 0 7px;font-size:14px}.logic-grid p{margin:0;color:var(--muted)}
    .trend-toolbar{justify-content:space-between;margin:0 0 10px}.trend-toolbar .status-line{margin:0}.trend-table{min-width:1280px;font-size:12px;table-layout:fixed}.trend-table th,.trend-table td{text-align:left;padding:7px 8px;vertical-align:top;white-space:normal;overflow-wrap:anywhere;word-break:break-word}.trend-table th{position:sticky;top:0;z-index:2;background:#f7f8fa}.trend-table .trend-fixed{min-width:70px}.trend-table .trend-zone-a{background:#fff3f5}.trend-table .trend-zone-b{background:#fff8e7}.trend-table .trend-zone-c{background:#f2f8ff}.trend-table .trend-blue-zone{background:#edf6ff}.trend-miss{color:#87919d}.trend-hit{display:inline-flex;width:22px;height:22px;align-items:center;justify-content:center;border-radius:50%;font-weight:700;color:#fff}.trend-hit.red{background:var(--red)}.trend-hit.blue{background:#1468a8}.trace-scroll-wrap{max-height:640px;overflow:auto;position:relative;scrollbar-gutter:stable both-edges}.trace-compact{font-size:12px;line-height:1.5}.trace-muted{color:var(--muted)}.trace-chip{display:inline-block;border:1px solid var(--line);border-radius:999px;padding:1px 7px;margin:1px 3px 1px 0;background:#fff}.trace-section{margin:0 0 8px}.trace-section-title{font-weight:700;color:#46515d;margin:0 0 4px}.trace-section+.trace-section{border-top:1px solid #e5e9ef;padding-top:7px}.trace-strategy{font-size:11px;line-height:1.35;color:#46515d}.trend-table th .trace-hit-filter{height:30px!important;min-width:0;width:100%;margin-top:6px;padding:0 6px;font-size:12px}
    @media(max-width:820px){.grid{grid-template-columns:repeat(2,minmax(0,1fr))}.chart-grid,.logic-grid{grid-template-columns:1fr}.header-inner{height:auto;padding:14px 0}.header-stat{display:none}.form-row{align-items:stretch}.field,.field.grow{min-width:100%}button.action{width:100%}}
    @media(max-width:460px){.header-inner,.nav-inner,main{width:min(100% - 20px,1180px)}.grid{grid-template-columns:1fr}.tab{padding:13px 12px}.section-head{align-items:flex-start;flex-direction:column}}
  </style>
</head>
<body>
<header><div class="header-inner"><div><h1>概率实验系统</h1><div class="subtitle">数据清洗、概率融合、覆盖优化与严格回测</div></div><div class="header-stat"><strong id="headerPeriods">-</strong><span>期有效历史数据</span></div></div></header>
<nav><div class="nav-inner"><button class="tab active" data-view="dataView">数据管理</button><button class="tab" data-view="runView">运行实验</button><button class="tab" data-view="resultView">结果报告</button></div></nav>
<main>
  <div class="notice">所有打分和组合仅为历史数据模拟，不代表下期开奖趋势，不存在提升中奖概率的效果。</div>
  <section id="dataView" class="view active">
    <div class="section-head"><div><h2>历史数据</h2><p>上传 CSV 或 XLSX，系统自动识别字段、校验并重建处理数据</p></div></div>
    <div class="grid" id="dataMetrics"></div>
    <div class="band"><h3>上传历史数据</h3><form id="uploadForm" class="form-row"><div class="field grow"><label for="historyFile">CSV / XLSX 文件</label><input id="historyFile" name="file" type="file" accept=".csv,.xlsx,text/csv,application/vnd.openxmlformats-officedocument.spreadsheetml.sheet" required></div><input type="hidden" id="uploadMode" name="mode" value="replace"><button class="action" id="uploadButton" type="submit">上传并校验</button></form><div class="status-line"><span class="dot" id="uploadDot"></span><span id="uploadMessage">等待上传；上传后将覆盖原始历史数据</span></div></div>
    <div class="band"><h3>标准字段</h3><code>period,date,r1,r2,r3,r4,r5,r6,blue</code><div class="explain">也识别中文表头：期号、开奖日期、红球1-6、蓝球；或合并列「开奖号码」（6个红球+蓝球）。Excel 含多个工作表时自动选用双色球，忽略福彩3D。红球范围 1-33 且同一期不得重复，蓝球范围 1-16。日期缺失时只在处理数据中按配置日历补全，原始文件不会被伪造成官方日期。</div></div>
  </section>
  <section id="runView" class="view">
    <div class="section-head"><div><h2>运行实验</h2><p>完整执行模型训练、融合、候选生成、覆盖优化和随机对照回测</p></div></div>
    <div class="band"><div class="form-row"><div class="field"><label for="runtimeProfile">运行档位</label><select id="runtimeProfile"><option value="fast">快速检查（约10-20分钟）</option><option value="standard" selected>标准验证（约25-45分钟）</option><option value="research">完整研究（数小时）</option></select></div><div class="field"><label for="periods">保留验证期数</label><input id="periods" type="number" min="1" step="1" value="100"></div><div class="field"><label for="samples">蒙特卡洛次数</label><input id="samples" type="number" min="100" max="100000" step="100" value="5000"></div><div class="field"><label for="outputCount">每个方案每期输出注数</label><input id="outputCount" type="number" min="2" max="50" value="5"></div><button class="action" id="runButton" type="button">开始标准验证</button></div><div class="explain" id="runtimeProfileExplain">标准验证：默认推荐；保留100期验证，关闭逐期重型整数规划，适合日常优化。</div><div class="explain">验证期数没有预设上限；可执行期数只受已上传历史数据和最小训练窗口约束。正式发布至少需要 200 期独立样本外验证；日常快速/标准档主要用于发现问题和生成可参考实验结果。输出注数只控制覆盖广度，不代表命中概率。</div><div class="status-line"><span class="dot" id="runDot"></span><span id="runMessage">尚未运行</span></div><div id="runError" class="explain error-box" hidden></div></div>
    <div class="grid" id="runMetrics"></div>
  </section>
  <section id="resultView" class="view">
    <div class="section-head"><div><h2>实验结果</h2><p>同时比较同流程无信号基线和同等注数纯随机组合</p></div><button class="action secondary" id="refreshResult" type="button">刷新结果</button></div>
    <div id="resultEmpty" class="band empty">尚无实验报告，请先运行完整实验。</div>
    <div id="resultContent" hidden>
       <div id="resultStale" class="explain error-box" hidden>历史数据已更新，当前报告基于旧数据。请重新运行完整实验后再解读结果。</div>
       <div id="reportProvenance" class="explain" hidden></div>
       <div class="grid" id="resultMetrics"></div>
       <div class="band"><h3>V7证据优先总览</h3><div class="explain" id="v7SummaryExplanation"></div><div class="grid" id="v7SummaryMetrics"></div><div class="table-wrap" style="margin-top:14px"><table><thead><tr><th>检查项</th><th>状态</th><th>说明</th></tr></thead><tbody id="v7ReleaseRows"></tbody></table></div></div>
       <div class="band" id="finalOutputBand"><h3 id="finalOutputHeading">本期最终实验输出</h3><div class="explain" id="finalOutputExplanation"></div><div class="table-wrap"><table><thead><tr><th>训练截止期</th><th>目标期</th><th>最终组合序号</th><th>红球</th><th>蓝球</th><th>固化状态</th></tr></thead><tbody id="finalOutputRows"></tbody></table></div></div>
       <div class="band"><h3>预算内胆拖压缩方案</h3><div class="explain" id="dantuoExplanation"></div><div class="table-wrap"><table><thead><tr><th>训练截止期</th><th>目标期</th><th>红胆</th><th>红拖</th><th>蓝球</th><th>展开注数</th><th>金额</th><th>规则说明</th></tr></thead><tbody id="dantuoRows"></tbody></table></div></div>
       <div class="band"><h3>核心号码提前告知</h3><div class="explain" id="coreNumbersExplanation"></div><div class="grid" id="coreNumberMetrics"></div><div class="table-wrap" style="margin-top:14px"><table><thead><tr><th>类型</th><th>号码</th><th>综合评分</th><th>模型共识</th><th>5注出现次数</th><th>全局覆盖支持</th><th>评级</th></tr></thead><tbody id="coreNumberRows"></tbody></table></div></div>
       <div class="band"><h3>开奖前组合留痕与开奖后独立验证</h3><div class="explain">这里首先展示运行时已固化的实验组合；目标期开奖尚未导入时，实际号码和命中字段会明确显示“待开奖或待导入”。它用于事后独立核验，不等同于最终预测或推荐。</div><div class="table-wrap"><table style="min-width:1120px"><thead><tr><th>开奖前训练截止</th><th>验证期</th><th>生成模式</th><th>开奖前组合及命中</th><th>实际号码</th><th>平均红球命中</th><th>随机理论期望</th><th>蓝球</th><th>结论</th></tr></thead><tbody id="forecastAuditRows"></tbody></table></div></div>
        <div class="band"><h3>完整历史号码走势图</h3><div class="explain">采用“期号 + 星期 + 三区红球 01–33 + 蓝球 01–16 + 和值、跨度、区间比、奇偶比”的逐期遗漏矩阵。彩色圆点为当期出现的号码，灰色数字为截至该期的遗漏值；可翻页查看所有已上传的历史期次。</div><div class="form-row trend-toolbar"><div class="field"><label for="trendPageSize">每页显示期数</label><select id="trendPageSize"><option value="50">50 期</option><option value="100">100 期</option><option value="200">200 期</option><option value="500">500 期</option></select></div><div class="status-line" id="trendPageInfo"></div><div><button class="action secondary" id="trendPrev" type="button">上一页</button><button class="action secondary" id="trendNext" type="button">下一页</button></div></div><div class="table-wrap"><table class="trend-table"><thead><tr><th class="trend-fixed" rowspan="2">期号</th><th rowspan="2">星期</th><th class="trend-zone-a" colspan="11">一区</th><th class="trend-zone-b" colspan="11">二区</th><th class="trend-zone-c" colspan="11">三区</th><th class="trend-blue-zone" colspan="16">蓝球</th><th rowspan="2">和值</th><th rowspan="2">跨度</th><th rowspan="2">区间比</th><th rowspan="2">奇偶比</th></tr><tr><th class="trend-zone-a">01</th><th class="trend-zone-a">02</th><th class="trend-zone-a">03</th><th class="trend-zone-a">04</th><th class="trend-zone-a">05</th><th class="trend-zone-a">06</th><th class="trend-zone-a">07</th><th class="trend-zone-a">08</th><th class="trend-zone-a">09</th><th class="trend-zone-a">10</th><th class="trend-zone-a">11</th><th class="trend-zone-b">12</th><th class="trend-zone-b">13</th><th class="trend-zone-b">14</th><th class="trend-zone-b">15</th><th class="trend-zone-b">16</th><th class="trend-zone-b">17</th><th class="trend-zone-b">18</th><th class="trend-zone-b">19</th><th class="trend-zone-b">20</th><th class="trend-zone-b">21</th><th class="trend-zone-b">22</th><th class="trend-zone-c">23</th><th class="trend-zone-c">24</th><th class="trend-zone-c">25</th><th class="trend-zone-c">26</th><th class="trend-zone-c">27</th><th class="trend-zone-c">28</th><th class="trend-zone-c">29</th><th class="trend-zone-c">30</th><th class="trend-zone-c">31</th><th class="trend-zone-c">32</th><th class="trend-zone-c">33</th><th class="trend-blue-zone">01</th><th class="trend-blue-zone">02</th><th class="trend-blue-zone">03</th><th class="trend-blue-zone">04</th><th class="trend-blue-zone">05</th><th class="trend-blue-zone">06</th><th class="trend-blue-zone">07</th><th class="trend-blue-zone">08</th><th class="trend-blue-zone">09</th><th class="trend-blue-zone">10</th><th class="trend-blue-zone">11</th><th class="trend-blue-zone">12</th><th class="trend-blue-zone">13</th><th class="trend-blue-zone">14</th><th class="trend-blue-zone">15</th><th class="trend-blue-zone">16</th></tr></thead><tbody id="trendMatrixRows"></tbody></table></div><h3 style="margin-top:20px">历史统计分析</h3><div class="explain" id="trendExplanation"></div><div class="grid" id="trendMetrics"></div><div class="table-wrap" style="margin-top:14px"><table style="min-width:1080px"><thead><tr><th>颜色</th><th>号码</th><th>分析窗口出现次数</th><th>近期出现次数</th><th>二项分布期望</th><th>标准化偏离 z</th><th>当前遗漏</th><th>线性斜率/期</th></tr></thead><tbody id="trendRows"></tbody></table></div></div>
      <div class="band"><div class="section-head"><div><h3 id="consensusHeading">全流程综合筛选</h3><p id="consensusSubtitle">融合多模型、相对联合分数及两类覆盖算法的共同结果</p></div><label class="result-option"><input id="showConsensus" type="checkbox" checked>显示综合组合</label></div><div id="consensusContent"><div class="explain">筛选分不是中奖概率。先按相对均匀基线的联合分数排序，再加入贪心/遗传入选支持，最后扣除号码重叠并优先使用不同蓝球；结构典型度仅展示，不参与默认排序。</div><div class="table-wrap"><table><thead><tr><th>序号</th><th>红球</th><th>蓝球</th><th>筛选分（非概率）</th><th>候选池排名</th><th>算法支持</th><th>相对联合分</th><th>结构典型度（仅说明）</th></tr></thead><tbody id="consensusRows"></tbody></table></div></div></div>
      <div class="band"><h3 id="predictionHeading">多模型融合后的当前实验组合</h3><div class="explain" id="predictionExplanation">多个概率模型先根据上传的历史数据训练和评分，再自适应融合；贪心覆盖与遗传算法只负责从融合候选池中筛选差异化组合。</div><div class="table-wrap"><table><thead><tr><th>完整方法链路</th><th>序号</th><th>红球</th><th>蓝球</th><th>筛选质量（非概率）</th><th>结构典型度</th></tr></thead><tbody id="predictionRows"></tbody></table></div></div>
      <div class="band"><h3>多模型训练、评分与融合</h3><div class="explain" id="pipelineExplanation"></div><div class="table-wrap"><table><thead><tr><th>概率模型</th><th>红球权重</th><th>蓝球权重</th><th>当前红球评分前 6</th><th>当前蓝球评分前 3</th><th>滚动 Brier</th><th>训练/测试差距</th><th>状态</th></tr></thead><tbody id="weightRows"></tbody></table></div></div>
      <div class="band research-hidden"><h3>多模型独立样本外诊断</h3><div class="explain">Brier技能分大于0表示概率校准优于均匀基线；是否可用还必须通过多模型 Holm 多重校正。蓝球首选命中另以精确二项检验及 Holm 校正检验是否高于 6.25%。</div><div class="table-wrap"><table><thead><tr><th>模型</th><th>红球Brier技能</th><th>红球校正p值</th><th>红球前6平均命中</th><th>蓝球Brier技能</th><th>蓝球Brier校正p值</th><th>蓝球首选命中</th><th>蓝球首选校正p值</th><th>证据结论</th></tr></thead><tbody id="modelDiagnosticRows"></tbody></table></div></div>
      <div class="band"><h3>蓝球专项证据、分布与跨度</h3><div class="explain" id="blueEvidenceConclusion"></div><div class="grid" id="blueEvidenceMetrics"></div><div class="table-wrap" style="margin-top:14px"><table><thead><tr><th>模型</th><th>样本外 Brier 技能</th><th>首选命中率</th><th>Brier Holm p 值</th><th>首选 Holm p 值</th><th>结论</th></tr></thead><tbody id="blueEvidenceRows"></tbody></table></div><h3 style="margin-top:20px">蓝球逐期验证与原因审计</h3><div class="explain">这里按保留期逐期展示蓝球验证结果：既看融合首选是否命中，也看前 3 覆盖、综合 5 注是否覆盖，并把每期未中的直接原因单独记录，便于后续按失败模式调参。</div><div class="grid" id="blueAuditMetrics"></div><div class="table-wrap" style="margin-top:14px"><table style="min-width:1320px"><thead><tr><th>验证期</th><th>训练截止期</th><th>实际蓝球</th><th>融合首选</th><th>融合前 3</th><th>综合 5 注蓝球</th><th>首选</th><th>前 3</th><th>综合 5 注</th><th>原因归类</th><th>直接原因</th></tr></thead><tbody id="blueAuditRows"></tbody></table></div><h3 style="margin-top:20px">蓝球历史分布与相邻期跨度</h3><div class="explain" id="bluePatternConclusion"></div><div class="table-wrap" style="margin-top:14px"><table><thead><tr><th>分析项</th><th>近期窗口</th><th>分析窗口</th><th>独立均匀参考</th><th>说明</th></tr></thead><tbody id="bluePatternRows"></tbody></table></div><h3 style="margin-top:20px">由最新蓝球出发的历史转移</h3><div class="table-wrap"><table><thead><tr><th>当前蓝球</th><th>历史后继蓝球</th><th>出现次数</th><th>拉普拉斯平滑占比</th><th>均匀参考</th></tr></thead><tbody id="blueTransitionRows"></tbody></table></div><h3 style="margin-top:20px">蓝球遗漏周期</h3><div class="table-wrap"><table><thead><tr><th>蓝球</th><th>窗口内出现次数</th><th>平均间隔</th><th>最大间隔</th><th>当前遗漏</th></tr></thead><tbody id="blueCycleRows"></tbody></table></div></div>
      <div class="band research-hidden"><h3>随机性与数据质量审计</h3><div class="explain">卡方检查边际均匀性，游程检查连续聚集，Ljung–Box 检查短期自相关，置换频谱检查异常周期峰值。所有结论均经过 Holm 多重校正，只用于数据质量诊断，不参与号码评分。</div><div class="grid" id="randomnessAuditMetrics"></div><div class="table-wrap" style="margin-top:14px"><table><thead><tr><th>审计方法</th><th>有效检验数</th><th>最小原始p值</th><th>最小校正p值</th><th>校正后异常数</th><th>最强信号</th><th>结论</th></tr></thead><tbody id="randomnessAuditRows"></tbody></table></div><div class="explain" id="randomnessAuditConclusion"></div></div>
      <div class="band research-hidden"><h3>覆盖优化对比</h3><div class="explain">蓝球单注理论基准始终是 1/16（6.25%）；多注实验优先使用不同蓝球，以提高整组“至少一注命中”的覆盖范围。</div><div class="table-wrap"><table><thead><tr><th>完整方法链路</th><th>红球覆盖</th><th>蓝球覆盖</th><th>蓝球组覆盖率</th><th>平均红球重叠</th><th>2 元子集去重率</th></tr></thead><tbody id="coverageRows"></tbody></table></div></div>
      <div class="band research-hidden"><h3>红球组合层消融实验（同一 Candidate Pool）</h3><div class="explain" id="redMethodAblationExplanation">只替换最终红球 Portfolio optimizer，保持每期训练窗口、模型输出、Candidate Pool、随机种子和注数一致，用于判断 red_ge3_objective 是否真正贡献。</div><div class="table-wrap"><table style="min-width:1320px"><thead><tr><th>实验</th><th>方法</th><th>期数</th><th>Candidate Recall</th><th>Final Recall</th><th>Candidate→Final Retention</th><th>Compression Regret</th><th>Best Red Hit</th><th>Any 3+</th><th>Any 4+</th><th>Portfolio Union</th><th>Jaccard</th></tr></thead><tbody id="redMethodAblationRows"></tbody></table></div></div>
      <div class="band research-hidden"><h3>严格回测与双基线</h3><div class="explain">主检验基线使用均匀号码分布，但经过完全相同的蒙特卡洛、联合分数排序和覆盖优化；它用于隔离多模型是否真正提供了额外信号。纯随机票仅作为现实参照。</div><div class="table-wrap"><table style="min-width:1540px"><thead><tr><th>完整方法链路</th><th>每期组合数</th><th>模型平均红球</th><th>同流程无信号</th><th>纯随机票</th><th>模型-无信号</th><th>差值95%区间</th><th>平均最佳红球</th><th>整组至少一注3+</th><th>模型蓝球组命中</th><th>无信号蓝球组命中</th><th>红球 3+</th><th>红球 4+</th><th>置换检验校正后p值</th><th>结论</th></tr></thead><tbody id="backtestRows"></tbody></table></div><div class="explain" id="resultExplanation"></div></div>
      <div class="chart-grid"><img id="performanceChart" alt="模型与随机基线指标对比图"><img id="distributionChart" alt="逐期命中波动图"></div>
    </div>
  </section>
</main>
<script>
const API_BASE='';
const state={status:null,trendMatrix:[],trendPage:0,traceVisibleCount:10,traceScrollBound:false,traceResetPending:false,traceHitFilter:''};const names={frequency:'频率模型',time_decay:'时间衰减',bayesian:'贝叶斯模型',logistic_regression:'逻辑回归',xgboost:'XGBoost',lightgbm_ranker:'LightGBM排序',mlp:'浅层MLP',catboost:'CatBoost',pattern_memory:'形态记忆',adaptive_fusion:'分色自适应融合',greedy:'贪心覆盖',genetic:'遗传算法',pareto:'Pareto多目标',historical_hit:'历史回放命中优化',red_ge3_objective:'红球3+目标优化',global_coverage:'整数规划全局覆盖',rotation_fallback:'轮转覆盖兜底',consensus:'综合筛选'};
const fmt=(v,d=3)=>v===null||v===undefined||Number.isNaN(Number(v))?'-':Number(v).toFixed(d);const pct=v=>v===null||v===undefined?'-':`${(Number(v)*100).toFixed(1)}%`;
const numberSet=(values,color)=>`<span class="number-set ${color}-number">${(values||[]).map(number=>String(number).padStart(2,'0')).join(' ')||'-'}</span>`;
const markedRedSet=(values,actualRed=[])=>{const hits=new Set((actualRed||[]).map(Number));return `<span class="number-set">${(values||[]).map(number=>`<span class="${hits.has(Number(number))?'matched-red-number':'red-number'}">${String(number).padStart(2,'0')}</span>`).join(' ')||'-'}</span>`};
const markedBlue=(number,hit)=>`<span class="number-set ${hit?'matched-blue-number':'blue-number'}">${number===null||number===undefined?'-':String(number).padStart(2,'0')}</span>`;
const matchedRed=(ticket,actualRed=[])=>Array.isArray(ticket?.matched_red)?ticket.matched_red:(ticket?.red||[]).filter(number=>new Set((actualRed||[]).map(Number)).has(Number(number)));
const rankedSet=(detail,color,limit)=>detail?.[`${color}_is_uniform`]?'<span>均匀退化，无有效排名</span>':numberSet((detail?.[`top_${color}`]||[]).slice(0,limit),color);
const fullMethod=method=>`多模型融合 + ${names[method]||method}`;
const currentMethod=(method,uniform)=>`${uniform?'均匀无信号':'多模型融合'} + ${names[method]||method}`;
const modeLabel=mode=>mode==='uniform_no_signal_baseline'?'无模型信号，基线实验留痕（非预测）':mode==='shrunk_weak_model_signal'?'弱信号收缩实验留痕（待验证）':'模型加权实验留痕';
function methodValidation(record,method){const entry=record.methods?.[method]||{},metrics=entry.model||{},tickets=entry.tickets||[];const blueHits=tickets.filter(ticket=>ticket.blue_hit).length;const anyBlue=metrics.any_blue_hit?'本期至少一注命中':'本期未中';return `平均红球 ${fmt(metrics.average_red_hits)} / 最佳 ${fmt(metrics.best_red_hits,0)} / 蓝球 ${blueHits}/${tickets.length}（${anyBlue}）`}
function experimentalTickets(record){const actualRed=record.actual?.red||[];return ['greedy','genetic','pareto','red_ge3_objective','historical_hit','global_coverage','consensus'].map(method=>{const tickets=record.methods?.[method]?.tickets||[];if(!tickets.length)return '';const rows=tickets.map((ticket,index)=>{const hits=matchedRed(ticket,actualRed);return `${index+1}. ${markedRedSet(ticket.red,actualRed)} + ${markedBlue(ticket.blue,ticket.blue_hit)}（红球命中 ${hits.length} 个：${hits.length?numberSet(hits,'blue'):'无'}；蓝球${ticket.blue_hit?'命中':'未命中'}）`}).join('<br>');return `<strong>${names[method]||method}（${tickets.length}注）</strong><br>${rows}`}).filter(Boolean).join('<hr style="border:0;border-top:1px solid #d9dee5;margin:7px 0">')||'-'}
document.querySelectorAll('.tab').forEach(btn=>btn.addEventListener('click',()=>{document.querySelectorAll('.tab').forEach(x=>x.classList.toggle('active',x===btn));document.querySelectorAll('.view').forEach(x=>x.classList.toggle('active',x.id===btn.dataset.view));}));
function metric(label,value,detail=''){return `<div class="metric"><span>${label}</span><strong>${value}</strong><span>${detail}</span></div>`}
function renderV7Summary(report){const v7=report.v7_diagnostics||report.backtest?.v7_diagnostics||{},gate=report.v7_release_gate||{},evidence=v7.evidence_report||{},best=evidence.best_score||{},prediction=v7.blue_prediction_set||{},audit=v7.compression_audit||{},auditRed=audit.red||{},auditBlue=audit.blue||{},regret=v7.regret_report||{},survival=v7.number_survival?.red||{},portfolio=report.candidate_experiment?.v7_portfolio||{},budget=portfolio.budget||{},retention=portfolio.information_retention||{};document.getElementById('v7SummaryExplanation').textContent=v7.principle||'V7强调先看严格样本外证据：没有稳定Edge时，系统应收缩概率、扩大候选集合并覆盖优先。';document.getElementById('v7SummaryMetrics').innerHTML=metric('V7模式',v7.enabled===false?'关闭':(v7.mode||'evidence_first'),v7.schema||'-')+metric('最佳证据方法',evidence.best_method||'-',`score ${fmt(best.score,4)} / confidence ${fmt(best.confidence,4)} / ${best.decision||'NO_EDGE'}`)+metric('蓝球候选集合',prediction.edge_state||'-',`宽度 ${prediction.selected_width??'-'}；模式 ${prediction.blue_mode||'-'}`)+metric('当前IRR',pct(retention.information_retention_rate??portfolio.information_retention_rate),`当前CR ${pct(retention.compression_regret??portfolio.compression_regret)}`)+metric('回测压缩审计IRR',pct(auditRed.information_retention_rate),`候选→最终CR ${pct(auditRed.compression_regret)}；蓝CE ${pct(auditBlue.coverage_efficiency)}`)+metric('动态预算',budget.edge_state||'-',budget.slots?`概率 ${budget.slots.probability??'-'} / 覆盖 ${budget.slots.coverage??'-'} / 探索 ${budget.slots.exploration??'-'}`:'等待当前预测生成')+metric('组合层Regret',regret.stage_counts?.final_portfolio_loss??'-',`模型/特征损失 ${regret.stage_counts?.model_or_feature_loss??'-'}；方法层损失 ${regret.stage_counts?.method_optimizer_loss??'-'}`)+metric('号码生存追踪',survival.periods??'-',`Top Lost ${(survival.top_lost_numbers||[]).slice(0,3).map(x=>String(x.number).padStart(2,'0')).join('、')||'-'}`)+metric('Falsification',v7.falsification_report?.checks?.strict_left_truncation?'通过':'待验证','反事实/置换/无信号检查优先')+metric('V7门禁',gate.decision||'shadow_or_reference_only',gate.approved?'允许生产':'仅影子/参考');document.getElementById('v7ReleaseRows').innerHTML=(gate.checks||[]).map(item=>`<tr><td>${item.item}</td><td class="${item.passed?'positive':'negative'}">${item.passed?'通过':'未通过'}</td><td>${item.detail||'-'}</td></tr>`).join('')||'<tr><td colspan="3">当前报告尚未生成 V7 Release Gate；请重新运行实验。</td></tr>'}
function renderRedMethodAblation(report){const ablation=report.backtest?.red_method_ablation||{},methods=ablation.methods||{},ranking=ablation.ranking||Object.keys(methods),leader=ablation.leader||ranking[0]||'-',redRank=ablation.red_ge3_rank??'-';const rows=ranking.map(method=>methods[method]).filter(Boolean);const target=document.getElementById('redMethodAblationRows'),explain=document.getElementById('redMethodAblationExplanation');if(explain)explain.textContent=rows.length?`同一 Candidate Pool 消融已生成：当前保留率第一为 ${names[leader]||leader}；red_ge3_objective 排名 ${redRank}/${rows.length}。本表只改变 Portfolio optimizer，不改变生产默认链路。`:'当前报告尚未生成红球组合层消融实验；重新跑一次严格 Walk-forward 后会自动生成。';if(!target)return;target.innerHTML=rows.map(item=>{const isLeader=item.method===leader,isRedGe3=item.method==='red_ge3_objective';return `<tr><td>${item.code||'-'}${isLeader?' 🥇':''}</td><td>${names[item.method]||item.label||item.method}${isRedGe3?'（待验证对象）':''}</td><td>${item.periods??'-'}</td><td>${pct(item.candidate_recall)}<br><span class="muted">${fmt(item.average_candidate_red_hits,2)}/6</span></td><td>${pct(item.final_recall)}<br><span class="muted">${fmt(item.average_final_red_union_hits,2)}/6</span></td><td class="${isLeader?'positive':''}">${pct(item.candidate_to_final_retention)}<br><span class="muted">P10/P50/P90 ${pct(item.retention_p10)} / ${pct(item.retention_p50)} / ${pct(item.retention_p90)}</span></td><td class="${Number(item.compression_regret)>0?'negative':'positive'}">${fmt(item.compression_regret,3)}</td><td>${fmt(item.average_best_red_hit,3)}</td><td>${pct(item.any_3_plus)}</td><td>${pct(item.any_4_plus)}</td><td>${fmt(item.portfolio_union,2)}</td><td>${fmt(item.jaccard,3)}</td></tr>`}).join('')||'<tr><td colspan="12">暂无消融数据。请重新运行一次严格 Walk-forward。</td></tr>'}
function buildTrendMatrix(trace){return trace?.rows||[]}
function traceTickets(tickets){return (tickets||[]).map(ticket=>`${ticket.index}. ${numberSet(ticket.red||[],'red')} + ${numberSet([ticket.blue],'blue')}（红中${(ticket.matched_red||[]).length}：${(ticket.matched_red||[]).length?numberSet(ticket.matched_red,'blue'):'无'}；蓝球${ticket.blue_hit?'命中':'未中'}）`).join('<br>')||'-'}
function traceModelText(models){return (models||[]).map(model=>`${names[model.model]||model.model}#${model.rank}`).join('、')||'无模型排名'}
function traceDropStages(row){const missedRed=new Set((row.missed_red||[]).map(Number));const red=(row.red_drop_stages||[]).filter(item=>missedRed.has(Number(item.number))).map(item=>`<div class="trace-compact"><span class="trace-chip">${String(item.number).padStart(2,'0')}</span>${item.eliminated_by||item.stage||'-'}<br><span class="trace-muted">候选${item.entered_candidate_pool?'已进入':'未进入'}，出现${item.candidate_pool_support_count??0}次${item.best_candidate_rank?`，最好第${item.best_candidate_rank}`:''}；Top模型：${traceModelText(item.top_models)}</span></div>`).join('')||'无';const blue=row.missed_blue?`<div class="trace-compact"><span class="trace-chip">蓝${String(row.missed_blue).padStart(2,'0')}</span>${row.blue_eliminated_by||row.blue_drop_stage||'-'}<br><span class="trace-muted">候选${row.blue_entered_candidate_pool?'已进入':'未进入'}，出现${row.blue_candidate_pool_support_count??0}次${row.blue_best_candidate_rank?`，最好第${row.blue_best_candidate_rank}`:''}；Top模型：${traceModelText(row.blue_top_models)}</span></div>`:'无';return `${red}<hr style="border:0;border-top:1px solid #d9dee5;margin:7px 0">${blue}`}
function traceReasonSection(title,body){return `<div class="trace-section"><div class="trace-section-title">${title}</div>${body||'<span class="trace-muted">无</span>'}</div>`}
function traceBlueHit(row){if(row?.covered_blue!==undefined&&row?.covered_blue!==null)return 1;return row?.blue_hit?1:0}function traceHitCount(row){const red=Number(row?.red_hit_count);const redN=Number.isFinite(red)?red:(row?.covered_red||[]).length;return redN+traceBlueHit(row)}
function tracePageSize(){const n=Number(document.getElementById('trendPageSize')?.value||50);return Number.isFinite(n)&&n>0?n:50}
function pagedTraceRows(){const all=state.trendMatrix||[],size=tracePageSize(),newestFirst=all.slice().reverse(),pages=Math.max(1,Math.ceil((newestFirst.length||0)/size)||1);if(state.trendPage>pages-1)state.trendPage=pages-1;if(state.trendPage<0)state.trendPage=0;return newestFirst.slice(state.trendPage*size,state.trendPage*size+size)}
function filteredTraceRows(){const page=pagedTraceRows(),filter=String(state.traceHitFilter||'');if(!filter)return page;if(filter==='blue')return page.filter(row=>traceBlueHit(row)===1);const m=filter.match(/^red(\d)$/);if(m){const n=Number(m[1]);return page.filter(row=>Number(row?.red_hit_count??(row?.covered_red||[]).length)===n)}return page}
function traceHitFilterOptions(){const opts=['<option value="">全部</option>'];for(let k=1;k<=6;k++)opts.push(`<option value="red${k}">红${k}</option>`);opts.push('<option value="blue">篮球</option>');return opts.join('')}
function setupTraceHitFilter(select){if(!select||select.dataset.bound)return;select.addEventListener('change',()=>{state.traceHitFilter=select.value;state.traceVisibleCount=10;state.traceResetPending=true;renderTrendMatrix()});select.dataset.bound='1'}
function setupTraceScroll(container){if(!container||state.traceScrollBound)return;container.addEventListener('scroll',()=>{if(state.traceVisibleCount>=filteredTraceRows().length)return;if(container.scrollTop+container.clientHeight>=container.scrollHeight-80){state.traceVisibleCount+=10;renderTrendMatrix()}});state.traceScrollBound=true}
function traceDantuoPlan(row){const plan=row.dantuo_plan||null;if(!plan)return '<span class="trace-muted">本期未生成预算胆拖，暂无压缩展示。</span>';const redDan=plan.red_dan||[],redTuo=plan.red_tuo||[],blue=plan.blue||[],layout=plan.requested_layout||'2胆 + 5拖 + 2蓝';const redText=redDan.length?`红胆：${numberSet(redDan,'red')}<br>红拖：${numberSet(redTuo,'red')}`:`红球：${numberSet(redTuo,'red')}`;return `<strong>${layout}</strong><br>${redText}<br>蓝球：${numberSet(blue,'blue')}<br><span class="trace-muted">${plan.bet_count??'-'}注 / ${plan.amount??'-'}元；${plan.rule||''}</span>`}
function tracePlanRole(row,number){const plan=row.dantuo_plan||{};if((plan.red_dan||[]).map(Number).includes(Number(number)))return '红胆';if((plan.red_tuo||[]).map(Number).includes(Number(number)))return plan.red_dan?.length?'红拖':'红球复式';return '未进入预算胆拖'}
function traceHitReasons(row){const redItems=row.red_drop_stages||[],coveredRed=(row.covered_red||[]).map(Number);const red=coveredRed.map(number=>{const item=redItems.find(x=>Number(x.number)===number)||{};return `<div class="trace-compact"><span class="trace-chip">${String(number).padStart(2,'0')}</span><strong>结果：已覆盖</strong>，位置：预算胆拖-${tracePlanRole(row,number)}<br><span class="trace-muted">关口：进入候选池 → 进入最终筛选 → 预算胆拖保留；证据：候选出现${item.candidate_pool_support_count??0}次${item.best_candidate_rank?`，最佳第${item.best_candidate_rank}`:''}；模型：${traceModelText(item.top_models)}</span></div>`}).join('');const blue=row.covered_blue!==undefined&&row.covered_blue!==null?`<div class="trace-compact"><span class="trace-chip">蓝${String(row.covered_blue).padStart(2,'0')}</span><strong>结果：已覆盖</strong>，位置：预算胆拖蓝球<br><span class="trace-muted">关口：进入候选池 → 进入最终筛选 → 预算胆拖保留；证据：候选出现${row.blue_candidate_pool_support_count??0}次${row.blue_best_candidate_rank?`，最佳第${row.blue_best_candidate_rank}`:''}；模型：${traceModelText(row.blue_top_models)}</span></div>`:'';return traceReasonSection('红球（已覆盖）',red)+traceReasonSection('蓝球（已覆盖）',blue)}
function traceMissReasons(row){const missedRed=new Set((row.missed_red||[]).map(Number));const red=(row.red_drop_stages||[]).filter(item=>missedRed.has(Number(item.number))).map(item=>{const reason=item.budget_eliminated_by||item.eliminated_by||item.stage||'-';return `<div class="trace-compact"><span class="trace-chip">${String(item.number).padStart(2,'0')}</span><strong>结果：未覆盖</strong><br><span class="trace-muted">关口：${reason}<br>证据：候选${item.entered_candidate_pool?'已进入':'未进入'}，出现${item.candidate_pool_support_count??0}次${item.best_candidate_rank?`，最佳第${item.best_candidate_rank}`:''}；方法：${(item.method_support||[]).join('、')||'无'}；模型：${traceModelText(item.top_models)}</span></div>`}).join('');const blue=row.missed_blue!==undefined&&row.missed_blue!==null?`<div class="trace-compact"><span class="trace-chip">蓝${String(row.missed_blue).padStart(2,'0')}</span><strong>结果：未覆盖</strong><br><span class="trace-muted">关口：${row.blue_budget_eliminated_by||row.blue_eliminated_by||row.blue_drop_stage||'-'}<br>证据：候选${row.blue_entered_candidate_pool?'已进入':'未进入'}，出现${row.blue_candidate_pool_support_count??0}次${row.blue_best_candidate_rank?`，最佳第${row.blue_best_candidate_rank}`:''}；方法：${(row.blue_method_support||[]).join('、')||'无'}；模型：${traceModelText(row.blue_top_models)}</span></div>`:'';return traceReasonSection('红球（未覆盖）',red)+traceReasonSection('蓝球（未覆盖）',blue)}
function renderTrendMatrix(){const source=state.trendMatrix||[],pageRows=pagedTraceRows(),matrix=filteredTraceRows(),size=tracePageSize(),total=source.length,pages=Math.max(1,Math.ceil(total/size)||1);const band=document.getElementById('trendMatrixRows').closest('.band');if(band){const title=band.querySelector('h3');if(title)title.textContent='逐期验证归因追踪';const explain=band.querySelector('.explain');if(explain)explain.textContent='数据来源：data/validation_trace.jsonl。请先用「每页显示期数」选定要看的验证期，再在「已覆盖号码原因」里按红球命中数（红1~红6）或蓝球命中（篮球）筛选当前页。口径为预算胆拖。';const toolbar=band.querySelector('.trend-toolbar'),field=toolbar?.querySelector('.field'),buttons=toolbar?.querySelector('div:last-child');if(field)field.hidden=false;if(buttons)buttons.hidden=false;const wrap=band.querySelector('.table-wrap');if(wrap){wrap.classList.add('trace-scroll-wrap');setupTraceScroll(wrap);if(state.traceResetPending){wrap.scrollTop=0;wrap.scrollLeft=0;state.traceResetPending=false}}const table=band.querySelector('table');if(table){table.classList.add('trend-table');table.style.minWidth='1280px';table.style.tableLayout='fixed';if(!table.querySelector('colgroup'))table.insertAdjacentHTML('afterbegin','<colgroup><col style="width:68px"><col style="width:78px"><col style="width:120px"><col style="width:180px"><col style="width:260px"><col style="width:375px"><col style="width:115px"><col style="width:70px"></colgroup>');if(table.tHead){let select=document.getElementById('traceHitFilter');if(!select){table.tHead.innerHTML='<tr><th>验证期</th><th>训练截止</th><th>实际开奖</th><th>预算胆拖</th><th>已覆盖号码原因<select id="traceHitFilter" class="trace-hit-filter"></select></th><th>未覆盖号码归因：步骤 / 模型 / 算法</th><th>最终策略</th><th>防泄漏</th></tr>';select=document.getElementById('traceHitFilter')}if(select){select.innerHTML=traceHitFilterOptions();select.value=state.traceHitFilter||'';setupTraceHitFilter(select)}}}}const visible=matrix;const emptyText=state.traceHitFilter?`当前 ${size} 期内没有符合「${state.traceHitFilter==='blue'?'篮球':state.traceHitFilter}」的验证期。`:'当前还没有可展示的逐期验证归因数据。请重新运行一次实验。';document.getElementById('trendMatrixRows').innerHTML=visible.map(row=>`<tr><td>${row.validation_period||'-'}</td><td>${row.training_through_period||'-'}</td><td>${numberSet(row.actual_red||[],'red')} + ${row.actual_blue?numberSet([row.actual_blue],'blue'):'-'}</td><td>${traceDantuoPlan(row)}</td><td><div class="trace-section-title">命中 ${traceHitCount(row)} 个</div>${traceHitReasons(row)}</td><td>${traceMissReasons(row)}</td><td><div class="trace-strategy">${row.consensus_strategy||'-'}</div><span class="trace-muted">候选池 ${row.candidate_pool_size??'-'}</span></td><td class="${row.strict_left_truncation_passed?'positive':'negative'}">${row.strict_left_truncation_passed?'通过':'失败'}</td></tr>`).join('')||`<tr><td colspan="8">${emptyText}</td></tr>`;const filterLabel=state.traceHitFilter?`其中符合「${state.traceHitFilter==='blue'?'篮球':state.traceHitFilter}」${matrix.length} 期`:`共 ${pageRows.length} 期`;document.getElementById('trendPageInfo').textContent=total?`先按每页 ${size} 期取第 ${state.trendPage+1}/${pages} 页（共 ${total} 期），${filterLabel}`:'等待生成追踪数据';document.getElementById('trendPrev').disabled=state.trendPage<=0;document.getElementById('trendNext').disabled=state.trendPage>=pages-1}
async function request(url,options){const response=await fetch(API_BASE+url,options);const data=await response.json();if(!response.ok)throw new Error(data.error||`请求失败 ${response.status}`);return data}
function setDot(id,status){const dot=document.getElementById(id);dot.className=`dot ${status||''}`}
async function refreshStatus(){try{const data=await request(`/api/status?_=${Date.now()}`);state.status=data;renderStatus(data);return true}catch(error){document.getElementById('runMessage').textContent=error.message;setDot('runDot','error');return false}}
function syncRunButton(){const periods=Number(document.getElementById('periods').value)||100,profile=document.getElementById('runtimeProfile').value,profiles=state.status?.config?.runtime_profiles||{},label=profiles[profile]?.label||document.getElementById('runtimeProfile').selectedOptions[0]?.textContent||'实验';document.getElementById('runButton').textContent=`开始${periods}期${label}`}
function applyRuntimeProfileDefaults(){const profile=document.getElementById('runtimeProfile').value,profiles=state.status?.config?.runtime_profiles||{},item=profiles[profile]||{};if(item.periods)document.getElementById('periods').value=item.periods;if(item.monte_carlo_samples)document.getElementById('samples').value=item.monte_carlo_samples;const methods=(item.methods||[]).map(method=>names[method]||method).join('、')||'-';document.getElementById('runtimeProfileExplain').textContent=`${item.label||profile}：预计 ${item.estimated_minutes||'-'} 分钟；${item.description||''} 验证期 ${item.periods||'-'}，蒙特卡洛 ${item.monte_carlo_samples||'-'}，随机基线使用精确期望，优化方法：${methods}。`;syncRunButton()}
function setupRuntimeProfiles(data){const select=document.getElementById('runtimeProfile'),profiles=data.config?.runtime_profiles||{};if(!state.profileInitialized&&Object.keys(profiles).length){const defaultProfile=data.config?.default_runtime_profile||'standard';select.innerHTML=Object.keys(profiles).map(key=>{const item=profiles[key]||{};return `<option value="${key}">${item.label||key}（约${item.estimated_minutes||'-'}分钟）</option>`}).join('');select.value=profiles[defaultProfile]?defaultProfile:Object.keys(profiles)[0];state.profileInitialized=true;applyRuntimeProfileDefaults()}}
function renderStatus(data){setupRuntimeProfiles(data);const d=data.data||{};document.getElementById('headerPeriods').textContent=d.clean_rows??'-';document.getElementById('dataMetrics').innerHTML=metric('原始记录',d.raw_rows??'-','history.csv')+metric('有效期数',d.clean_rows??'-','clean_data.pkl')+metric('最新期号',d.latest_period??'-','按期号升序')+metric('补全日期',d.dates_filled??'-','仅处理数据');const job=data.job||{};const running=!!job.running;document.getElementById('runButton').disabled=running;document.getElementById('runMessage').textContent=running?`实验运行中，开始于 ${job.started_at||'-'}`:job.last_error?`运行失败：${job.last_error}`:job.finished_at?`最近一次完成：${job.finished_at}`:'尚未运行';setDot('runDot',running?'running':job.last_error?'error':job.finished_at?'success':'');document.getElementById('runError').hidden=!job.last_error;document.getElementById('runError').textContent=job.last_error||'';document.getElementById('runMetrics').innerHTML=metric('运行状态',running?'执行中':'空闲',running?'详细步骤正在启动命令行输出':'详细日志仅在命令行和 pipeline.log')+metric('默认运行档位',data.config?.runtime_profiles?.[data.config?.default_runtime_profile]?.label||data.config?.default_runtime_profile||'-',data.config?.runtime_profiles?.[data.config?.default_runtime_profile]?.description||'')+metric('默认验证期数',data.config?.holdout_periods??30,'随机基线使用精确期望')+metric('最小训练窗口',data.config?.min_train_size??'-','严格左截断')+metric('批量导出',data.config?.enable_export_bet?'允许':'禁止','合规开关');renderReport(data.report,!!d.report_stale,data.report_meta||{});if(running)setTimeout(refreshStatus,1800)}
function renderReport(report,reportStale=false,reportMeta={}){
  const empty=document.getElementById('resultEmpty'),content=document.getElementById('resultContent');
  if(!report){empty.hidden=false;content.hidden=true;document.getElementById('resultStale').hidden=true;document.getElementById('reportProvenance').hidden=true;return}
  empty.hidden=true;content.hidden=false;
  document.getElementById('resultStale').hidden=!reportStale;
  const provenance=document.getElementById('reportProvenance'),generatedAt=reportMeta.generated_at_utc||report.run_metadata?.generated_at_utc||'-';
  provenance.hidden=false;provenance.textContent=reportMeta.source==='current_session'?`本会话刚完成的实验报告（生成于 ${generatedAt}）。`:`已载入历史保存报告（生成于 ${generatedAt}）；当前会话尚未启动新的实验。`;
  const back=report.backtest||{},summary=back.summary||{},candidate=report.candidate_experiment||{},release=report.model_release||candidate.release||{};
  const fusionSummary=back.fusion_validation_summary||{},candidateRecall=back.candidate_recall_funnel||{},candidateRecallRed=candidateRecall.red||{},candidateRecallBlue=candidateRecall.blue||{},releaseChecklist=report.release_checklist||{},firstMethodSummary=Object.values(summary)[0]||{};
  const calibration=back.calibration_summary||{};
  const releaseEvidence=back.release_evidence||{},releaseWindows=releaseEvidence.windows||{},developmentWindow=releaseWindows.development||{},confirmationWindow=releaseWindows.confirmation||{};
  const runCalibration=report.config_snapshot?.calibration||{};
  const validationPeriods=calibration.completed_holdout_periods??firstMethodSummary.periods??0;
  const validationHeading=document.getElementById('validationHeading');
  if(validationHeading) validationHeading.textContent=`保留${validationPeriods||'-'}期逐期严格验证`;
  const tuningEnabled=Boolean(runCalibration.enable_hyperparameter_adjustment);
  const tuningRule=tuningEnabled?`自动调参需累计满足 ${runCalibration.minimum_adjustment_observations??'-'} 期，并连续 ${runCalibration.minimum_failure_streak??'-'} 期未达标；仅依据 Brier 目标触发，冷却期 ${runCalibration.adjustment_cooldown_periods??'-'} 期，防止对随机波动过度拟合。`:'当前未启用自动调参。';
  const confirmationDetail=releaseEvidence.available?`开发窗 ${developmentWindow.periods??0} 期，确认窗 ${confirmationWindow.periods??0} 期，超参数${confirmationWindow.hyperparameters_frozen?'已冻结':'未冻结'}`:'暂无双窗口确认数据';
  const validationExplanation=document.getElementById('validationExplanation');
  if(validationExplanation) validationExplanation.textContent=`系统先用历史窗口完成融合预热，再按时间顺序逐期验证；每期只允许使用当期之前的数据生成预测，并在开奖后回填结果更新下一期权重。${tuningRule} 结果页展示的是严格留后验证留痕，不是事后回看拼接结果。`;
  document.getElementById('resultMetrics').innerHTML=
    metric('历史数据期数',report.data?.periods??'-','当前用于本次实验的有效历史期数')
    +metric('严格验证期数',validationPeriods||'-',`含 ${calibration.fusion_warmup_completed_periods??0} 期融合预热，训练截止 ${calibration.initial_training_through_period||'-'}`)
    +metric('双窗/确认',releaseEvidence.available?'已生成':'未生成',confirmationDetail)
    +metric('正式发布状态',release.formal_status_label||(release.approved?'已满足正式发布':'未满足正式发布'),release.approved?'当前结果已满足正式发布条件':release.reason||`至少需要 ${release.minimum_out_of_sample_periods??200} 期严格留后验证`)
    +metric('实验展示状态',release.experimental_status_label||(release.provisional_approved?'允许展示':'暂不展示'),release.provisional_approved?'当前允许展示实验结果':'当前实验结果仍受展示门槛限制')
    +metric('蓝球信号覆盖',pct(release.blue_signal_coverage),`门槛 ${pct(release.minimum_blue_signal_coverage)}；仅在有信号期检验首选命中`)
    +metric('开奖前验证样本',`${release.completed_prospective_validations??0}/${release.minimum_prospective_validations??'-'}`,`计入口径 ${release.prospective_code_scope==='any'?'所有版本':'当前版本'}；当前 ${release.completed_prospective_validations_current_code??release.completed_prospective_validations??0}，所有 ${release.completed_prospective_validations_any_code??release.completed_prospective_validations??0}`)
    +metric('超参数调整',tuningEnabled?calibration.parameter_adjustment_rounds??0:'关闭',tuningEnabled?`滚动目标 ${runCalibration.rolling_objective_periods??'-'} 期；权重重刷新 ${calibration.weight_refresh_rounds??'-'} 次`:`当前仅刷新权重 ${calibration.weight_refresh_rounds??'-'} 次`)
    +metric('防泄漏审计',back.strict_left_truncation&&!calibration.future_validation_rows_used_for_adjustment?'通过':'失败','预热与保留期均严格左截断')
    +metric('V7放行清单',releaseChecklist.passed?'通过':'未通过',`${(releaseChecklist.items||[]).filter(item=>item.passed).length}/${(releaseChecklist.items||[]).length||'-'} 项通过`)
    +metric('红球召回漏斗',fmt(candidateRecallRed.avg_final_layer_hits,2),`Top12 ${fmt(candidateRecallRed.avg_model_top12_hits,2)} / Top18 ${fmt(candidateRecallRed.avg_model_top18_hits,2)} / Candidate Set ${fmt(candidateRecallRed.avg_candidate_set_hits??candidateRecallRed.avg_candidate_pool_hits,2)} / Candidate→Method ${pct(candidateRecallRed.avg_candidate_to_method_true_retention)} / Candidate→Final ${pct(candidateRecallRed.avg_candidate_to_final_true_retention)} / Final丢失 ${fmt(candidateRecallRed.avg_candidate_to_final_true_loss,2)}`)
    +metric('蓝球召回漏斗',pct(candidateRecallBlue.final_layer_hit),`Top5 ${pct(candidateRecallBlue.hit_at_5)} / 候选池 ${pct(candidateRecallBlue.candidate_pool_hit)} / 方法层 ${pct(candidateRecallBlue.method_layer_hit)}`);
  renderV7Summary(report);
  renderRedMethodAblation(report);
  const finalArchive=report.forecast_archive||{},finalTickets=finalArchive.tickets||[],finalMode=finalArchive.prediction_mode||candidate.prediction_mode,finalIsBaseline=finalMode==='uniform_no_signal_baseline',finalIsWeak=finalMode==='shrunk_weak_model_signal';
  const mathematicalBaseline=report.mathematical_baseline||{};
  const baselineText=`单注红球理论期望 ${fmt(mathematicalBaseline.single_ticket_expected_red_hits,2)} 个；红球命中 3 个及以上的理论概率 ${pct(mathematicalBaseline.single_ticket_red_ge_3_probability)}；${mathematicalBaseline.diversified_group_ticket_count??'-'} 注不同蓝球组合时，至少一注蓝球命中的理论覆盖 ${pct(mathematicalBaseline.diversified_group_any_blue_hit_probability)}。`;
  const canPublish=Boolean(release.approved)&&!finalIsBaseline;
  const canShowFinalExperiment=Boolean(finalTickets.length);
  const showObservedExperiment=canShowFinalExperiment&&!finalIsBaseline;
  const weakStructureFinal=showObservedExperiment&&!canPublish&&Boolean(release.weak_structure_reference_ready);
  const finalStatusText=
    finalArchive.status==='validated'?'已核验':
    finalArchive.display_status==='archived_experimental_baseline'?'基线留痕（非预测）':
    canPublish?'正式发布':
    weakStructureFinal?'可参考弱结构':
    showObservedExperiment?'实验观察（待独立确认）':'暂无输出';
  document.getElementById('finalOutputHeading').textContent=
    !canShowFinalExperiment?'本期暂无实验输出':
    finalIsBaseline?'本期基线实验留痕（非预测）':
    canPublish?'本期正式发布结果':
    weakStructureFinal?'本期可参考弱结构结果':'本期实验观察结果（待独立确认）';
  document.getElementById('finalOutputExplanation').textContent=
    !canShowFinalExperiment
      ?`当前还没有可展示的开奖前留痕。${release.reason||'请先运行一轮完整实验。'} ${baselineText}`
      :finalIsBaseline
        ?`已按训练截止期 ${finalArchive.source_period||'-'} 固化 ${finalTickets.length} 注基线实验组合，目标期为 ${finalArchive.target_period||'-'}，审计编号 ${finalArchive.forecast_id||'-'}。这部分只用于开奖后独立核验，不解释为预测结果。${baselineText}`
        :`已按训练截止期 ${finalArchive.source_period||'-'} 固化 ${finalTickets.length} 注实验组合，目标期为 ${finalArchive.target_period||'-'}，审计编号 ${finalArchive.forecast_id||'-'}。${canPublish?'该组合已通过当前正式发布门槛，可作为本期发布结果展示。':weakStructureFinal?'该组合尚未通过正式发布门槛，但已存在非基线弱结构，因此按“可参考弱结构结果”单独展示，便于后续独立核验。':'该组合仍保留为开奖前实验观察结果；即使正式发布门槛尚未通过，页面也会继续展示这份留痕，便于后续独立核验。'} ${baselineText}`;
  document.getElementById('finalOutputRows').innerHTML=canShowFinalExperiment?finalTickets.map((ticket,index)=>`<tr><td>${finalArchive.source_period||'-'}</td><td>${finalArchive.target_period||'-'}</td><td>${index+1}</td><td>${numberSet(ticket.red,'red')}</td><td>${numberSet([ticket.blue],'blue')}</td><td>${finalStatusText}</td></tr>`).join(''):'<tr><td colspan="6">当前还没有可展示的开奖前留痕。</td></tr>';
  const dantuo=finalArchive.dantuo_plan||candidate.dantuo_plan||candidate.consensus?.dantuo_plan||{};
  const dantuoAvailable=Array.isArray(dantuo.red_dan)&&Array.isArray(dantuo.red_tuo)&&Array.isArray(dantuo.blue);
  const dantuoAudit=dantuo.final_to_dantuo_audit||{},dantuoDropped=(dantuoAudit.dropped||[]).map(n=>String(n).padStart(2,'0')).join(' ');
  const blueDantuoAudit=dantuo.blue_final_to_dantuo_audit||{},blueDantuoDropped=(blueDantuoAudit.dropped||[]).map(n=>String(n).padStart(2,'0')).join(' ');
  const dantuoAuditText=dantuoAudit.enabled?` 红球Final5→胆拖保留率 ${pct(dantuoAudit.retention)}，高价值保留率 ${pct(dantuoAudit.high_value_retention)}；丢弃红球：${dantuoDropped||'无'}。`:'';
  const blueDantuoAuditText=blueDantuoAudit.enabled?` 蓝球Final5→胆拖保留率 ${pct(blueDantuoAudit.retention)}；丢弃蓝球：${blueDantuoDropped||'无'}。`:'';
  document.getElementById('dantuoExplanation').textContent=dantuoAvailable?`胆拖现在只作为最终组合的预算压缩器：Final5先生成，再压缩为 ${dantuo.requested_layout||'2胆 + 5拖 + 2蓝'}；候选池和方法票只作为证据，不再重新决定另一套预测号码，蓝球同样只压缩Final5已产生的蓝球证据。金额按 ${dantuo.unit_price??2} 元/注计算，不超过 ${dantuo.max_amount??20} 元。${dantuoAuditText}${blueDantuoAuditText}该方案仍是历史实验输出，不代表下期开奖趋势。`:'当前报告还没有生成胆拖压缩方案；请重新运行一次实验。';
  const dantuoHit=dantuo.hit_summary||{};
  const hitText=dantuoHit.red_union_hits?.length?`；已导入实际号时，红胆/红拖覆盖 ${numberSet(dantuoHit.red_union_hits,'blue')}，蓝球${dantuoHit.blue_hit?'命中':'未中'}`:'';
  const auditText=dantuoAudit.enabled?`；红球Final5保留 ${pct(dantuoAudit.retention)}，丢弃 ${dantuoDropped||'无'}`:'';
  const blueAuditText=blueDantuoAudit.enabled?`；蓝球Final5保留 ${pct(blueDantuoAudit.retention)}，丢弃 ${blueDantuoDropped||'无'}`:'';
  document.getElementById('dantuoRows').innerHTML=dantuoAvailable?`<tr><td>${finalArchive.source_period||'-'}</td><td>${finalArchive.target_period||'-'}</td><td>${numberSet(dantuo.red_dan,'red')}</td><td>${numberSet(dantuo.red_tuo,'red')}</td><td>${numberSet(dantuo.blue,'blue')}</td><td>${dantuo.bet_count??'-'} 注</td><td>${dantuo.amount??'-'} 元</td><td>${dantuo.rule||'-'}${auditText}${blueAuditText}${hitText}</td></tr>`:'<tr><td colspan="8">当前没有可展示的胆拖压缩方案。</td></tr>';
  const core=finalArchive.core_numbers||candidate.core_numbers||{};
  const coreRed=core.core_red||[],coreBlue=core.core_blue||[],coreTicket=core.concentrated_ticket||{};
  document.getElementById('coreNumbersExplanation').textContent=core.enabled?`核心号码不是新增“保证命中”的模型，而是把融合概率、多模型共识、最终筛选和全局覆盖支持合成一张核心清单，用来解释哪些号码被系统反复看好，避免只看单一输出时误以为命中号码毫无线索。${core.risk_note||''}`:'当前报告没有核心号码摘要，请重新运行一次实验。';
  document.getElementById('coreNumberMetrics').innerHTML=metric('核心红球Top6',numberSet(coreRed,'red'),'用于观察高集中候选')+metric('核心蓝球Top3',numberSet(coreBlue,'blue'),'蓝球仍以覆盖验证为准')+metric('集中型参考单注',`${numberSet(coreTicket.red,'red')} + ${numberSet([coreTicket.blue],'blue')}`,'仅为实验观察，不改变风险提示')+metric('评分方法',core.method?'已生成':'无','融合概率/模型共识/5注高频/全局覆盖支持');
  const coreRows=[...(core.red||[]).map(item=>({type:'红球',color:'red',...item})),...(core.blue||[]).map(item=>({type:'蓝球',color:'blue',...item}))];
  document.getElementById('coreNumberRows').innerHTML=coreRows.map(item=>`<tr><td>${item.type}</td><td>${numberSet([item.number],item.color)}</td><td>${fmt(item.score,4)}</td><td>${item.model_consensus_count??0}</td><td>${item.ticket_frequency??0}</td><td>${item.global_coverage_frequency??0}</td><td>${item.grade==='high'?'高':item.grade==='medium'?'中':'参考'}</td></tr>`).join('')||'<tr><td colspan="7">当前没有可展示的核心号码。</td></tr>';
  const audit=report.out_of_sample_forecast_audit||{},forecastRecords=audit.records||[];
  const latestForecastRecords=Object.values(forecastRecords.reduce((acc,record)=>{const key=`${record.source_period||''}_${record.target_period||''}`;if(!acc[key]||Number(record.forecast_revision||0)>=Number(acc[key].forecast_revision||0))acc[key]=record;return acc;},{})).sort((left,right)=>String(right.target_period||'').localeCompare(String(left.target_period||''))||Number(right.forecast_revision||0)-Number(left.forecast_revision||0));
  document.getElementById('forecastAuditRows').innerHTML=latestForecastRecords.map(record=>{const validation=record.validation||null,actual=validation?.actual||{},metrics=validation?.metrics||{},baseline=validation?.theoretical_random_baseline||{},tickets=validation?.tickets||record.tickets||[],ticketText=tickets.map((ticket,index)=>{if(!validation)return `${index+1}. ${numberSet(ticket.red,'red')} + ${numberSet([ticket.blue],'blue')}（待核验）`;const hits=matchedRed(ticket,actual.red);return `${index+1}. ${markedRedSet(ticket.red,actual.red)} + ${markedBlue(ticket.blue,ticket.blue_hit)}（红球命中 ${hits.length} 个：${hits.length?numberSet(hits,'blue'):'无'}；蓝球${ticket.blue_hit?'命中':'未命中'}）`}).join('<br>');const actualText=validation?`${numberSet(actual.red,'red')} + ${numberSet([actual.blue],'blue')}`:'待开奖或待导入';const revisionText=record.forecast_revision&&Number(record.forecast_revision)>1?`修订版 r${record.forecast_revision}`:'首版留痕';return `<tr><td>${record.source_period}</td><td>${record.target_period}</td><td>${modeLabel(record.prediction_mode)}</td><td>${ticketText||'-'}</td><td>${actualText}</td><td>${validation?fmt(metrics.average_red_hits):'-'}</td><td>${validation?fmt(baseline.average_red_hits_per_ticket):'-'}</td><td>${validation?(metrics.any_blue_hit?'至少一注命中':'未命中'):'待核验'}</td><td>${validation?`仅展示当前目标期最新留痕（${revisionText}），用于独立复核`:`已固化当前目标期最新留痕（${revisionText}），等待目标期数据后独立核验`}</td></tr>`}).join('')||'<tr><td colspan="9">尚无组合留痕。请运行一次实验；运行完成后，这里会先显示待核验组合。</td></tr>';
  const trace=report.validation_trace||{},traceRows=trace.rows||[];
  const traceBand=document.getElementById('trendRows').closest('.band'),traceHeads=traceBand?traceBand.querySelectorAll('h3'):[];
  if(traceHeads[1])traceHeads[1].textContent='丢失阶段汇总';
  const traceComplete=trace.run_complete||{},missedBlueCount=traceRows.filter(row=>row.missed_blue!==null&&row.missed_blue!==undefined).length;
  const redMissCount=traceRows.reduce((total,row)=>total+(row.missed_red||[]).length,0),avgRedHits=traceRows.length?traceRows.reduce((total,row)=>total+Number(row.red_hit_count||0),0)/traceRows.length:null;
  const regretRows=traceRows.filter(row=>row.final5_drop_regret&&row.final5_drop_regret.drop_regret!==undefined);
  const avgRegret=regretRows.length?regretRows.reduce((total,row)=>total+Number(row.final5_drop_regret.drop_regret||0),0)/regretRows.length:null;
  document.getElementById('trendExplanation').textContent=trace.exists?`逐期归因来自 ${trace.path||'data/validation_trace.jsonl'}；运行完成记录显示 ${traceComplete.completed_holdout_periods??trace.periods??'-'} 个验证期。${avgRegret===null?'':` Final5 Drop Regret 平均 ${avgRegret.toFixed(2)}，这是候选已找到、最终5注丢掉的号码，归因是组合优化失败。`}`:'当前没有逐期验证归因文件，请重新运行一次实验。';
  document.getElementById('trendMetrics').innerHTML=metric('追踪验证期数',trace.periods??0,traceRows.length?`${traceRows[0].validation_period||'-'} 至 ${traceRows[traceRows.length-1].validation_period||'-'}`:'暂无')+metric('平均红球命中',fmt(avgRedHits,3),`未命中红球累计 ${redMissCount} 个`)+metric('蓝球命中期数',traceRows.filter(row=>row.blue_hit).length,`未中 ${missedBlueCount} 期`)+metric('防泄漏',traceRows.every(row=>row.strict_left_truncation_passed)?'通过':'存在异常','训练截止期必须严格早于验证期');
  const stageCounter={};
  function addStageReason(stage,color,number,example){
    const key=stage||'未知阶段';
    if(!stageCounter[key])stageCounter[key]={count:0,red:new Set(),blue:new Set(),examples:[]};
    stageCounter[key].count+=1;
    if(color==='blue')stageCounter[key].blue.add(Number(number));else stageCounter[key].red.add(Number(number));
    if(example&&stageCounter[key].examples.length<3)stageCounter[key].examples.push(example);
  }
  function classifyRedMiss(item){
    const methods=item.method_support||[],text=`${item.budget_eliminated_by||item.eliminated_by||item.stage||''}`;
    if(text.includes('Final5组合优化失败'))return '红球：Candidate已找到，Final5组合优化失败';
    if(!item.entered_candidate_pool)return '红球：未进入候选池（模型概率 / 采样 / 候选质量未达保留线）';
    if(!methods.length)return '红球：已进候选池，但未被 greedy / pareto / historical_hit 等优化方法选中';
    if(item.budget_eliminated_by)return '红球：已进最终组合，但在预算胆拖压缩层未保留';
    if(text.includes('重叠'))return '红球：已被方法选中，但被最终重叠约束 / 覆盖约束压缩';
    if(text.includes('压缩')||text.includes('最终')||text.includes('Portfolio')||text.includes('portfolio'))return '红球：已被方法选中，但最终Portfolio未保留';
    return '红球：已进候选池，但综合分 / 结构约束 / 覆盖目标不占优';
  }
  function classifyBlueMiss(row){
    const methods=row.blue_method_support||[],text=`${row.blue_budget_eliminated_by||row.blue_eliminated_by||row.blue_drop_stage||''}`;
    if(!row.blue_entered_candidate_pool)return '蓝球：未进入候选池 / Prediction Set（模型排序或采样未覆盖）';
    if(!methods.length)return '蓝球：已进候选池，但未被优化方法选中';
    if(row.blue_budget_eliminated_by)return '蓝球：已通过最终蓝球集合，但预算胆拖蓝球层未保留';
    if(text.includes('压缩')||text.includes('final')||text.includes('最终'))return '蓝球：已进候选池/方法层，但最终单蓝选择未保留';
    return '蓝球：候选池已覆盖，但单号排序无稳定优势导致未保留';
  }
  traceRows.forEach(row=>{
    const missedRed=new Set((row.missed_red||[]).map(Number));
    (row.red_drop_stages||[]).filter(item=>missedRed.has(Number(item.number))).forEach(item=>{
      const methods=(item.method_support||[]).slice(0,5).join('、')||'无';
      const models=traceModelText((item.top_models||[]).slice(0,3));
      const detail=item.budget_eliminated_by||item.eliminated_by||item.stage||'暂无细节';
      addStageReason(classifyRedMiss(item),'red',item.number,`红${String(item.number).padStart(2,'0')}：候选${item.entered_candidate_pool?'已进入':'未进入'}，最佳第${item.best_candidate_rank??'-'}，方法 ${methods}，模型 ${models}；${detail}`);
    });
    if(row.missed_blue!==null&&row.missed_blue!==undefined){
      const methods=(row.blue_method_support||[]).slice(0,5).join('、')||'无';
      const models=traceModelText((row.blue_top_models||[]).slice(0,3));
      const detail=row.blue_budget_eliminated_by||row.blue_eliminated_by||row.blue_drop_stage||'暂无细节';
      addStageReason(classifyBlueMiss(row),'blue',row.missed_blue,`蓝${String(row.missed_blue).padStart(2,'0')}：候选${row.blue_entered_candidate_pool?'已进入':'未进入'}，最佳第${row.blue_best_candidate_rank??'-'}，方法 ${methods}，模型 ${models}；${detail}`);
    }
  });
  const traceStageRows=Object.entries(stageCounter).sort((a,b)=>b[1].count-a[1].count).map(([stage,item])=>({stage,...item,red:[...item.red].sort((a,b)=>a-b),blue:[...item.blue].sort((a,b)=>a-b)}));
  document.getElementById('trendRows').closest('table').querySelector('thead').innerHTML='<tr><th>归因类型</th><th>出现次数</th><th>具体号码</th><th>样例证据</th></tr>';
  document.getElementById('trendRows').innerHTML=traceStageRows.map(item=>{const nums=[item.red.length?`红：${numberSet(item.red,'red')}`:'',item.blue.length?`蓝：${numberSet(item.blue,'blue')}`:''].filter(Boolean).join('<br>')||'-';return `<tr><td>${item.stage}</td><td>${item.count}</td><td>${nums}</td><td>${item.examples.join('<br>')||'仅统计未命中的实际号码；归因包含候选池、优化方法、最终综合层和蓝球专家等步骤。'}</td></tr>`}).join('')||'<tr><td colspan="4">暂无丢失阶段统计。请重新运行一次严格实验。</td></tr>';
  state.trendPage=0;state.traceVisibleCount=10;state.traceResetPending=true;state.trendMatrix=buildTrendMatrix(trace);renderTrendMatrix();
  const consensus=candidate.consensus||{};
  const uniformMode=candidate.prediction_mode==='uniform_no_signal_baseline'||candidate.model_pipeline?.fusion?.is_uniform;
  const confirmedCombination=Boolean(release.approved)&&!uniformMode;
  const displayConsensusSelected=consensus.display_selected||consensus.experimental_selected||[];
  const displaySelectedByMethod=candidate.display_selected||candidate.experimental_selected||{};
  const experimentalCombination=!uniformMode&&canShowFinalExperiment&&Boolean(displayConsensusSelected.length||Object.keys(displaySelectedByMethod).length||candidate.has_displayable_candidates);
  const weakStructureReference=experimentalCombination&&!confirmedCombination&&Boolean(release.weak_structure_reference_ready||release.experimental_status==='weak_structure_reference');
  const combinationDisplayAllowed=confirmedCombination||experimentalCombination;
  const consensusSelected=confirmedCombination?(consensus.selected||[]):experimentalCombination?displayConsensusSelected:[];
  const finalOutputBand=document.getElementById('finalOutputBand');
  if(finalOutputBand)finalOutputBand.hidden=weakStructureReference;
  document.getElementById('consensusHeading').textContent=confirmedCombination?`本期发布综合组合（${consensusSelected.length} 注）`:weakStructureReference?`本期可参考弱结构综合组合（${consensusSelected.length} 注）`:experimentalCombination?`本期最终实验综合输出（${consensusSelected.length} 注）`:'本期暂无综合组合';
  document.getElementById('consensusSubtitle').textContent=confirmedCombination?'当前综合组合已通过正式发布门槛。':weakStructureReference?`当前已存在非基线弱结构，先按“可参考弱结构结果”分层展示，便于后续独立核验，但不作为正式发布结果。${(release.display_reason||release.reason)?` 正式门槛说明：${release.display_reason||release.reason}`:''}`:experimentalCombination?`当前综合组合仍属于待确认实验输出，先展示留痕，后续以开奖后独立核验为准。${(release.display_reason||release.reason)?` 正式门槛说明：${release.display_reason||release.reason}`:''}`:(release.display_reason||release.reason)||'当前没有可展示的综合组合。';
  document.getElementById('consensusRows').innerHTML=consensusSelected.map((ticket,index)=>`<tr><td>${index+1}</td><td>${numberSet(ticket.red,'red')}</td><td>${numberSet([ticket.blue],'blue')}</td><td>${fmt(ticket.consensus_score,4)}</td><td>${ticket.candidate_pool_rank??'-'}</td><td>${(ticket.method_support||[]).map(method=>names[method]||method).join('、')||'无方法支持'}</td><td>${fmt(ticket.quality_score,4)}</td><td>${fmt(ticket.structure_score,4)}</td></tr>`).join('')||'<tr><td colspan="8">当前没有可展示的综合组合。</td></tr>';
  const selected=confirmedCombination?(candidate.selected||{}):experimentalCombination?displaySelectedByMethod:{};
  document.getElementById('predictionHeading').textContent=confirmedCombination?'本期发布模型组合':weakStructureReference?'本期可参考弱结构模型组合':experimentalCombination?'本期最终实验模型输出':'本期暂无模型组合';
  document.getElementById('predictionExplanation').textContent=combinationDisplayAllowed?`这里展示的是当前数据跑出的最终模型实验输出。${confirmedCombination?'由于已通过正式门槛，所以作为发布结果展示。':weakStructureReference?'虽然正式门槛和实验门槛尚未完全通过，但只要已经存在非基线结构，我们就将其作为“可参考弱结构结果”分层展示，不再让页面看起来像“没结果”。':'虽然正式门槛尚未通过，但只要存在非基线实验输出，我们仍保留展示，避免页面看起来像“没结果”。'}`:(release.display_reason||release.reason)||'当前没有可展示的模型组合。';
  const predictionRows=Object.keys(selected).flatMap(method=>selected[method].map((ticket,index)=>`<tr><td>${fullMethod(method)}</td><td>${index+1}</td><td>${numberSet(ticket.red,'red')}</td><td>${numberSet([ticket.blue],'blue')}</td><td>${fmt(ticket.quality_score)}</td><td>${fmt(ticket.structure_score)}</td></tr>`));
  document.getElementById('predictionRows').innerHTML=predictionRows.length?predictionRows.join(''):'<tr><td colspan="6">当前没有可展示的模型组合。</td></tr>';
  const weights=back.final_weights||{},status=back.final_model_status||{},pipeline=candidate.model_pipeline||{},modelDetails=pipeline.models||{};
  const fusion=pipeline.fusion||{},strength=fusion.signal_strength_by_color||{},uniformNotice=fusion.is_uniform?' 当前全部模型未证明优于同流程无信号基线，已退化为均匀采样，不生成虚假号码排名。':fusion.signal_tier==='weak_shrunk_signal'?` 当前为弱信号收缩：红球模型占比 ${pct(strength.red)}，蓝球模型占比 ${pct(strength.blue)}，其余权重由均匀基线承担。`:'';
  document.getElementById('pipelineExplanation').textContent=pipeline.training_through_period?`当前组合使用截至 ${pipeline.training_through_period} 的 ${pipeline.training_periods} 期数据训练；红球使用固定6个号码的排序目标，蓝球使用每期16选1互斥目标；融合同时检查30/100/200期证据并按不确定性收缩。融合归一化熵：红球 ${fmt(fusion.red_normalized_entropy,4)}，蓝球 ${fmt(fusion.blue_normalized_entropy,4)}，越接近1表示越接近均匀、可用信息越少。${uniformNotice}`:'请重新运行实验以生成多模型明细。';
  const modelRows=Object.keys(weights).map(key=>{const s=status[key]||{},detail=modelDetails[key]||{},badRed=!!s.by_color?.red?.invalid_for_overfit,badBlue=!!s.by_color?.blue?.invalid_for_overfit,bad=badRed||badBlue,redWeight=detail.red_weight??candidate.weights_by_color?.red?.[key]??weights[key],blueWeight=detail.blue_weight??candidate.weights_by_color?.blue?.[key]??weights[key],active=Number(redWeight)>0||Number(blueWeight)>0,badText=[badRed?'红球过拟合':'',badBlue?'蓝球过拟合':''].filter(Boolean).join('、');return `<tr><td>${names[key]||key}</td><td>${pct(redWeight)}</td><td>${pct(blueWeight)}</td><td>${rankedSet(detail,'red',6)}</td><td>${rankedSet(detail,'blue',3)}</td><td>${fmt(s.walk_forward_brier,5)}</td><td>${fmt(s.generalization_gap,5)}</td><td class="${bad||!active?'negative':'positive'}">${bad?`${badText}失效`:active?'按红/蓝球分别参与融合':'历史验证未优于基线，权重归零'}</td></tr>`});
  if(fusion.top_red){const redFusionWeight=pct(strength.red??(fusion.red_is_uniform?0:1)),blueFusionWeight=pct(strength.blue??(fusion.blue_is_uniform?0:1));modelRows.push(`<tr><td><strong>分色自适应融合结果</strong></td><td>${redFusionWeight}</td><td>${blueFusionWeight}</td><td>${rankedSet(fusion,'red',6)}</td><td>${rankedSet(fusion,'blue',3)}</td><td>-</td><td>-</td><td class="${fusion.is_uniform?'negative':'positive'}">${fusion.is_uniform?'无有效权重，均匀随机采样':fusion.signal_tier==='weak_shrunk_signal'?'弱信号已向均匀基线收缩':'供蒙特卡洛候选池采样'}</td></tr>`)}
  document.getElementById('weightRows').innerHTML=modelRows.join('')||'<tr><td colspan="8">请重新运行实验以生成多模型训练明细。</td></tr>';
  const diagnostics=back.model_diagnostics?.by_color||{},redDiagnostics=diagnostics.red||{},blueDiagnostics=diagnostics.blue||{};
  document.getElementById('modelDiagnosticRows').innerHTML=Object.keys(redDiagnostics).map(key=>{const red=redDiagnostics[key]||{},blue=blueDiagnostics[key]||{},redSig=red.significance||{},blueSig=blue.significance||{},blueTop1Sig=blue.top1_significance||{},proven=red.proven_better_than_uniform&&blue.proven_better_than_uniform;return `<tr><td>${names[key]||key}</td><td>${fmt(red.brier_skill_score,4)}</td><td>${fmt(redSig.holm_adjusted_p_value??redSig.p_value,4)}</td><td>${fmt(red.average_top6_hits,3)}</td><td>${fmt(blue.brier_skill_score,4)}</td><td>${fmt(blueSig.holm_adjusted_p_value??blueSig.p_value,4)}</td><td>${pct(blue.top1_hit_ratio)}</td><td>${fmt(blueTop1Sig.holm_adjusted_p_value??blueTop1Sig.p_value,4)}</td><td class="${proven?'positive':'negative'}">${proven?'通过概率证据；Top1仅诊断':'未证明概率优于均匀'}</td></tr>`}).join('')||'<tr><td colspan="9">旧报告没有独立模型诊断，请重新运行一次严格实验。</td></tr>';
  const blueEvidence=report.blue_evidence||{},blueTop1=blueEvidence.top1||{},blueTop3=blueEvidence.top3||{},bluePredictionSet=blueEvidence.prediction_set||{},blueRanking=blueEvidence.ranking||{},blueAblation=blueEvidence.ablation||{},blueBrier=blueEvidence.brier||{},blueCoverage=blueEvidence.coverage||{},blueRandomness=blueEvidence.randomness||{},bluePatterns=blueEvidence.patterns||{},blueAudit=blueEvidence.audit||{},blueTopK=blueEvidence.topk_ladder||[];
  const blueTopKText=blueTopK.length?blueTopK.map(row=>`T${row.k}:${pct(row.hit_ratio)}`).join(' / '):'-';
  document.getElementById('blueEvidenceConclusion').textContent=blueEvidence.conclusion||'旧报告没有蓝球专项证据，请重新运行一次实验。';
  document.getElementById('blueEvidenceMetrics').innerHTML=metric('Prediction Set Hit@5',pct(bluePredictionSet.hit_at_5??blueRanking.hit_at_5),`主目标；均匀 ${pct(bluePredictionSet.uniform_probability??(5/16))} / Edge ${pct(bluePredictionSet.edge_vs_uniform)}`)+metric('Top1仅诊断',blueTop1.hits===undefined?'-':`${blueTop1.hits}/${blueEvidence.periods??'-'}`,`实际 ${pct(blueTop1.hit_ratio)} / 均匀 ${pct(blueTop1.uniform_probability)} / p=${fmt(blueTop1.exact_one_sided_p_value,4)}`)+metric('前3覆盖命中',blueTop3.hits===undefined?'-':`${blueTop3.hits}/${blueEvidence.periods??'-'}`,`实际 ${pct(blueTop3.hit_ratio)} / 均匀 ${pct(blueTop3.uniform_probability)} / p=${fmt(blueTop3.exact_one_sided_p_value,4)}`)+metric('蓝球Top-K梯子',blueTopKText,'与随机理论 K/16 逐档比较')+metric('融合Top5排序',pct(blueRanking.hit_at_5),`MRR ${fmt(blueRanking.mrr,4)} / NDCG@5 ${fmt(blueRanking.ndcg_at_5,4)} / 平均排名 ${fmt(blueRanking.mean_rank,2)}`)+metric('最终覆盖层',pct(blueAblation.stages?.[2]?.coverage_hit_ratio),`覆盖命中 ${blueAblation.stages?.[2]?.coverage_hits??'-'}/${blueEvidence.periods??'-'}；与融合Top5分开统计`)+metric('蓝球 Brier 技能',fmt(blueBrier.skill_score,4),`均匀基线 ${fmt(blueBrier.uniform_baseline,5)}；p=${fmt(blueBrier.significance?.p_value,4)}`)+metric('不同蓝球覆盖',`${blueCoverage.ticket_count??'-'} 注`, `整组理论至少一注 ${pct(blueCoverage.any_blue_hit_probability_with_distinct_blues)}`)+metric('历史规则审计',blueRandomness.global_holm_significant_count??'-',`已检验 ${blueRandomness.tested_series??'-'} 条序列`);
  document.getElementById('blueEvidenceRows').innerHTML=(blueEvidence.model_rows||[]).map(item=>{const proven=item.proven_better_than_uniform;return `<tr><td>${names[item.model]||item.model}</td><td>${fmt(item.brier_skill_score,4)}</td><td>${pct(item.top1_hit_ratio)}</td><td>${fmt(item.holm_adjusted_p_value,4)}</td><td>${fmt(item.top1_holm_adjusted_p_value,4)}</td><td class="${proven?'positive':'negative'}">${proven?'概率证据通过；Top1仅诊断':'未证明概率优于均匀'}</td></tr>`}).join('')||'<tr><td colspan="6">旧报告没有蓝球专项证据，请重新运行一次实验。</td></tr>';
  const blueAuditReasons=blueAudit.reason_rows||[],primaryBlueReason=blueAuditReasons[0]||{};
  document.getElementById('blueAuditMetrics').innerHTML=metric('逐期样本',blueAudit.periods??'-',`本次逐期核验的蓝球期数`)+metric('Hit@5候选覆盖',blueAudit.top5_hits===undefined?'-':`${blueAudit.top5_hits}/${blueAudit.periods??'-'}`,`覆盖率 ${pct(blueAudit.top5_hit_ratio)}`)+metric('Top1仅诊断',blueAudit.top1_hits===undefined?'-':`${blueAudit.top1_hits}/${blueAudit.periods??'-'}`,`命中率 ${pct(blueAudit.top1_hit_ratio)}`)+metric('综合5注覆盖',blueAudit.consensus_hits===undefined?'-':`${blueAudit.consensus_hits}/${blueAudit.periods??'-'}`,`覆盖率 ${pct(blueAudit.consensus_hit_ratio)}`)+metric('主因',primaryBlueReason.reason||'-',primaryBlueReason.ratio===undefined?'':`占比 ${pct(primaryBlueReason.ratio)}`);
  document.getElementById('blueAuditRows').innerHTML=(blueAudit.rows||[]).map(item=>{const direct=[item.blue_reason,item.gate_reason].filter(Boolean).join('；');return `<tr><td>${item.validation_period||'-'}</td><td>${item.training_through_period||'-'}</td><td>${item.actual_blue===undefined||item.actual_blue===null?'-':numberSet([item.actual_blue],'blue')}</td><td>${item.fused_top1===undefined||item.fused_top1===null?'-':numberSet([item.fused_top1],'blue')}</td><td>${(item.fused_top3||[]).length?numberSet(item.fused_top3,'blue'):'-'}</td><td>${(item.consensus_blues||[]).length?numberSet(item.consensus_blues,'blue'):'-'}</td><td class="${item.top1_hit?'positive':'negative'}">${item.top1_hit?'命中':'未中'}</td><td class="${item.top3_hit?'positive':'negative'}">${item.top3_hit?'覆盖':'未覆盖'}</td><td class="${item.consensus_any_hit?'positive':'negative'}">${item.consensus_any_hit?'至少一注命中':'未覆盖'}</td><td>${item.reason||'-'}</td><td>${direct||'-'}</td></tr>`}).join('')||'<tr><td colspan="11">暂无蓝球逐期原因审计。请重新运行一次严格实验。</td></tr>';
  document.getElementById('bluePatternConclusion').textContent=bluePatterns.conclusion||bluePatterns.reason||'旧报告没有蓝球分布与跨度分析，请重新运行一次实验。';
  document.getElementById('bluePatternRows').innerHTML=(bluePatterns.metric_rows||[]).map(item=>`<tr><td>${item.metric}</td><td>${item.metric.includes('跨度')&&!item.metric.includes('占比')?fmt(item.recent,3):pct(item.recent)}</td><td>${item.metric.includes('跨度')&&!item.metric.includes('占比')?fmt(item.analysis,3):pct(item.analysis)}</td><td>${item.metric.includes('跨度')&&!item.metric.includes('占比')?fmt(item.reference,3):pct(item.reference)}</td><td>${item.interpretation}</td></tr>`).join('')||'<tr><td colspan="5">暂无蓝球历史分布分析。</td></tr>';
  document.getElementById('blueTransitionRows').innerHTML=(bluePatterns.transition_rows||[]).map(item=>`<tr><td>${numberSet([item.previous],'blue')}</td><td>${numberSet([item.next],'blue')}</td><td>${item.observed_count}</td><td>${pct(item.laplace_probability)}</td><td>${pct(item.uniform_probability)}</td></tr>`).join('')||'<tr><td colspan="5">暂无蓝球转移统计。</td></tr>';
  document.getElementById('blueCycleRows').innerHTML=(bluePatterns.cycle_rows||[]).map(item=>`<tr><td>${numberSet([item.number],'blue')}</td><td>${item.occurrences}</td><td>${fmt(item.mean_interval,2)}</td><td>${item.maximum_interval??'-'}</td><td>${item.current_omission}</td></tr>`).join('')||'<tr><td colspan="5">暂无蓝球遗漏周期统计。</td></tr>';
  const randomness=report.randomness_audit||{},randomFamilies=randomness.families||{},auditNames={chi_square:'卡方均匀性',runs:'游程检验',autocorrelation:'Ljung–Box自相关',spectral:'置换频谱峰值'};
  const auditTotal=randomness.significant_after_global_holm_total??randomness.significant_after_holm_total;
  document.getElementById('randomnessAuditMetrics').innerHTML=metric('审计历史期数',randomness.periods??'-',randomness.latest_period||'')+metric('审计方法',Object.keys(randomFamilies).length||'-','卡方、游程、自相关、频谱')+metric('校正后异常',auditTotal??'-','Holm族内多重校正')+metric('审计状态',randomness.enabled===false?'已关闭':randomness.requires_manual_review?'需要复核':'未见显著异常','不参与选号和模型权重');
  document.getElementById('randomnessAuditRows').innerHTML=Object.keys(randomFamilies).map(key=>{const family=randomFamilies[key]||{},signals=(family.strongest_signals||[]).slice(0,3).map(item=>`${item.label} (${fmt(item.holm_adjusted_p_value,4)})`).join('、')||'-',abnormal=Number(family.significant_after_holm_count||0);return `<tr><td>${auditNames[key]||key}</td><td>${family.valid_test_count??'-'}</td><td>${fmt(family.minimum_raw_p_value,4)}</td><td>${fmt(family.minimum_holm_adjusted_p_value,4)}</td><td class="${abnormal?'negative':'positive'}">${abnormal}</td><td>${signals}</td><td>${abnormal?'发现需人工复核的历史异常':'未发现校正后显著异常'}</td></tr>`}).join('')||'<tr><td colspan="7">旧报告没有随机性审计，请重新运行一次完整实验。</td></tr>';
  document.getElementById('randomnessAuditConclusion').textContent=randomness.conclusion||'尚无随机性审计结论。重新运行完整实验后生成；审计结果不会参与候选号码评分。';
  const validationRows=document.getElementById('validationRows');
  if(validationRows){const validationRecords=(back.period_records||[]).filter(record=>record.actual&&record.fused_prediction);
  validationRows.innerHTML=validationRecords.map(record=>{const fused=record.fused_prediction,actual=record.actual,hit=record.fused_validation||{},cal=record.calibration_adjustment||{},assessment=record.validation_assessment||{},audit=Number(record.history_max_idx)+1===Number(record.target_idx)&&cal.uses_only_completed_validation!==false,redHit=hit.red_hits===null||hit.red_hits===undefined?'不统计':hit.red_hits,blueHit=hit.blue_hit===null||hit.blue_hit===undefined?'不统计':hit.blue_hit?'命中':'未中',changeCount=(cal.changes||[]).length,gate=String(cal.parameter_gate_reason||''),calText=assessment.status==='abstained_no_signal'?'无模型信号，本期展示基线实验组合':assessment.passed?'达标，保留参数':changeCount?`未达标，受限调整 ${changeCount} 项`:gate.includes('均匀分布')?'均匀退化，禁止拟合随机波动':'未达标，未满足调参闸门';const actualDraw=`${numberSet(actual.red,'red')} + ${numberSet([actual.blue],'blue')}`;return `<tr><td>${record.validation_period||record.period}</td><td>${record.training_through_period}</td><td>${rankedSet(fused,'red',6)}</td><td>${rankedSet(fused,'blue',1)}</td><td>${experimentalTickets(record)}</td><td>${actualDraw}</td><td>红球 ${redHit} / 蓝球 ${blueHit}</td><td>${methodValidation(record,'greedy')}</td><td>${methodValidation(record,'genetic')}</td><td>${methodValidation(record,'consensus')}</td><td class="${assessment.passed?'positive':'negative'}">${calText}</td><td class="${audit?'positive':'negative'}">${audit?'通过':'失败'}</td></tr>`}).join('')||'<tr><td colspan="12">旧报告没有逐期校准明细，请重新运行实验。</td></tr>';}
  const coverage=candidate.coverage||{};
  document.getElementById('coverageRows').innerHTML=Object.keys(coverage).map(key=>{const c=coverage[key],s=c.subsets?.['2']||{};return `<tr><td>${currentMethod(key,uniformMode)}</td><td>${c.red_coverage}/33</td><td>${c.blue_coverage}/16</td><td>${pct(c.blue_group_coverage_ratio)}</td><td>${fmt(c.average_red_overlap,2)}</td><td>${pct(s.uniqueness_ratio)}</td></tr>`}).join('');
  let anySignificant=false;
  document.getElementById('backtestRows').innerHTML=Object.keys(summary).map(key=>{const x=summary[key],sig=x.significance||{},good=(sig.significant_after_holm_0_05??sig.significant_at_0_05)&&sig.mean_difference>0,ci=sig.confidence_interval||sig.confidence_interval_95||[];if(good)anySignificant=true;return `<tr><td>${fullMethod(key)}</td><td>${x.ticket_count_per_period??'-'}</td><td>${fmt(x.model_average_red_hits)}</td><td>${fmt(x.matched_no_signal_average_red_hits)}</td><td>${fmt(x.random_average_red_hits)}</td><td class="${sig.mean_difference>0?'positive':'negative'}">${fmt(sig.mean_difference)}</td><td>[${fmt(ci[0])}, ${fmt(ci[1])}]</td><td>${fmt(x.average_best_red_hits)}</td><td>${pct(x.period_any_red_ge3_ratio)}</td><td>${pct(x.period_any_blue_hit_ratio)}</td><td>${pct(x.matched_no_signal_period_any_blue_hit_ratio)}</td><td>${pct(x.red_ge3_ratio)}</td><td>${pct(x.red_ge4_ratio)}</td><td>${fmt(sig.holm_adjusted_p_value??sig.p_value,4)}</td><td>${good?'相对无信号基线仍显著':'未证明模型提供额外信号'}</td></tr>`}).join('');
  document.getElementById('resultExplanation').textContent=anySignificant?'当前窗口相对同流程无信号基线出现统计显著差异，仍需在冻结参数后的新开奖样本中复验。':'当前回测没有证明多模型相对同流程无信号基线提供额外信息；覆盖分和结构典型度不能解释为中奖概率提升。';
  const stamp=Date.now();document.getElementById('performanceChart').src=`${API_BASE}/logs/performance_comparison.png?t=${stamp}`;document.getElementById('distributionChart').src=`${API_BASE}/logs/hit_distribution.png?t=${stamp}`;
}
document.getElementById('uploadForm').addEventListener('submit',async event=>{event.preventDefault();const button=document.getElementById('uploadButton'),file=document.getElementById('historyFile').files[0];if(!file)return;button.disabled=true;setDot('uploadDot','running');document.getElementById('uploadMessage').textContent='正在上传、校验和清洗...';try{const form=new FormData();form.append('file',file);form.append('mode',document.getElementById('uploadMode').value);const result=await request('/api/upload',{method:'POST',body:form});setDot('uploadDot','success');document.getElementById('uploadMessage').textContent=`导入完成：有效 ${result.clean_rows} 期，新增/更新原始记录 ${result.upload_rows} 行`;await refreshStatus()}catch(error){setDot('uploadDot','error');document.getElementById('uploadMessage').textContent=error.message}finally{button.disabled=false}});
document.getElementById('runtimeProfile').addEventListener('change',applyRuntimeProfileDefaults);
document.getElementById('periods').addEventListener('input',syncRunButton);syncRunButton();
document.getElementById('runButton').addEventListener('click',async()=>{try{await request('/api/run',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({runtime_profile:document.getElementById('runtimeProfile').value,periods:Number(document.getElementById('periods').value),monte_carlo_samples:Number(document.getElementById('samples').value),output_count:Number(document.getElementById('outputCount').value)})});setDot('runDot','running');document.getElementById('runMessage').textContent='任务已启动，正在准备数据...';refreshStatus()}catch(error){setDot('runDot','error');document.getElementById('runMessage').textContent=error.message}});
async function refreshResults(){const button=document.getElementById('refreshResult');button.disabled=true;button.textContent='刷新中...';const ok=await refreshStatus();button.textContent=ok?'已刷新':'刷新失败';setTimeout(()=>{button.textContent='刷新结果';button.disabled=false},900)}
function syncConsensusVisibility(){document.getElementById('consensusContent').hidden=!document.getElementById('showConsensus').checked}
document.getElementById('showConsensus').addEventListener('change',syncConsensusVisibility);syncConsensusVisibility();
document.getElementById('trendPageSize').addEventListener('change',()=>{state.trendPage=0;state.traceVisibleCount=10;state.traceResetPending=true;renderTrendMatrix()});
document.getElementById('trendPrev').addEventListener('click',()=>{state.trendPage=Math.max(0,(state.trendPage||0)-1);state.traceResetPending=true;renderTrendMatrix()});
document.getElementById('trendNext').addEventListener('click',()=>{state.trendPage=(state.trendPage||0)+1;state.traceResetPending=true;renderTrendMatrix()});
document.getElementById('refreshResult').addEventListener('click',refreshResults);refreshStatus();
</script>
</body></html>"""


class DashboardState:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.running = False
        self.data_mutating = False
        self.started_at: str | None = None
        self.finished_at: str | None = None
        self.last_error: str | None = None
        self.last_result: dict[str, Any] | None = None

    def start(self) -> bool:
        with self._lock:
            if self.running or self.data_mutating:
                return False
            self.running = True
            self.started_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
            self.finished_at = None
            self.last_error = None
            self.last_result = None
            return True

    def start_upload(self) -> bool:
        with self._lock:
            if self.running or self.data_mutating:
                return False
            self.data_mutating = True
            self.last_error = None
            return True

    def finish_upload(self, error: Exception | None = None) -> None:
        with self._lock:
            self.data_mutating = False
            if error is not None:
                self.last_error = str(error)

    def finish(self, error: Exception | None = None, result: dict[str, Any] | None = None) -> None:
        with self._lock:
            self.running = False
            self.finished_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
            self.last_error = str(error) if error else None
            if result is not None:
                self.last_result = result

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "running": self.running,
                "data_mutating": self.data_mutating,
                "started_at": self.started_at,
                "finished_at": self.finished_at,
                "last_error": self.last_error,
                "last_result": self.last_result,
            }


class ExclusiveThreadingHTTPServer(ThreadingHTTPServer):
    allow_reuse_address = False

    def server_bind(self) -> None:
        if hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        super().server_bind()

    def handle_error(self, request, client_address) -> None:
        error = sys.exc_info()[1]
        if _is_client_disconnect(error):
            logging.debug(
                "网页客户端已断开 | 事件=WEB_CLIENT_DISCONNECT | 客户端=%s | 原因=%s",
                client_address,
                error,
            )
            return
        super().handle_error(request, client_address)


_CLIENT_DISCONNECT_WINERRORS = {10053, 10054, 10058}
_CLIENT_DISCONNECT_ERRNOS = {
    errno.EPIPE,
    errno.ECONNRESET,
    errno.ECONNABORTED,
    10053,
    10054,
}


def _is_client_disconnect(error: BaseException | None) -> bool:
    seen: set[int] = set()
    current: BaseException | None = error
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, (BrokenPipeError, ConnectionAbortedError, ConnectionResetError, TimeoutError)):
            return True
        if isinstance(current, OSError):
            if getattr(current, "winerror", None) in _CLIENT_DISCONNECT_WINERRORS:
                return True
            if current.errno in _CLIENT_DISCONNECT_ERRNOS:
                return True
        current = current.__cause__ or current.__context__
    return False


def _address_in_use(error: OSError) -> bool:
    return getattr(error, "winerror", None) == 10048 or getattr(error, "errno", None) in {
        errno.EADDRINUSE,
        10048,
    }


def _bind_dashboard_server(
    host: str,
    port: int,
    handler: type[BaseHTTPRequestHandler],
    *,
    attempts: int = 20,
) -> tuple[ExclusiveThreadingHTTPServer, int]:
    requested = int(port)
    last_error: OSError | None = None
    for candidate in range(requested, requested + max(1, int(attempts))):
        try:
            return ExclusiveThreadingHTTPServer((host, candidate), handler), candidate
        except OSError as error:
            last_error = error
            if _address_in_use(error):
                logging.warning(
                    "监听端口已被占用，改试下一端口 | 事件=DASHBOARD_PORT_BUSY | 地址=http://%s:%s",
                    host,
                    candidate,
                )
                print(f"端口 {candidate} 已被占用，正在尝试 {candidate + 1} ...", flush=True)
                continue
            raise
    raise SystemExit(
        f"启动失败：http://{host}:{requested} 起连续 {max(1, int(attempts))} 个端口均被占用，请先关闭旧窗口后再试。"
    ) from last_error


def dashboard_project_slug(project_root: Path) -> str:
    root = Path(project_root).resolve()
    if root.parent.name.lower() == "tsystem":
        return "Tsystem"
    return "Tsys"


def dashboard_url_prefix(project_root: Path) -> str:
    return f"/{dashboard_project_slug(project_root)}"


def _strip_dashboard_prefix(path: str, prefix: str) -> str:
    if path == prefix:
        return "/"
    if path.startswith(prefix + "/"):
        stripped = path[len(prefix) :]
        return stripped or "/"
    return path


def _dashboard_page(slug: str, url_prefix: str) -> str:
    page = WEB_PAGE.replace("const API_BASE='';", f"const API_BASE='{url_prefix}';", 1)
    page = page.replace("<title>概率实验系统</title>", f"<title>概率实验系统 · {slug}</title>", 1)
    page = page.replace("<h1>概率实验系统</h1>", f"<h1>概率实验系统 · {slug}</h1>", 1)
    return page


def _multipart_fields(content_type: str, body: bytes) -> tuple[dict[str, str], dict[str, tuple[str, bytes]]]:
    message = BytesParser(policy=policy.default).parsebytes(
        b"Content-Type: " + content_type.encode("ascii") + b"\r\nMIME-Version: 1.0\r\n\r\n" + body
    )
    fields: dict[str, str] = {}
    files: dict[str, tuple[str, bytes]] = {}
    for part in message.iter_parts():
        name = part.get_param("name", header="content-disposition")
        if not name:
            continue
        filename = part.get_filename()
        payload = part.get_payload(decode=True) or b""
        if filename:
            files[name] = (Path(filename).name, payload)
        else:
            fields[name] = payload.decode(part.get_content_charset() or "utf-8").strip()
    return fields, files


def serve_dashboard(
    *,
    project_root: Path,
    host: str,
    port: int,
    status_callback: Callable[[dict[str, Any]], dict[str, Any]],
    run_callback: Callable[[dict[str, Any]], Any],
    upload_callback: Callable[[str, bytes, str], dict[str, Any]],
    notification_config: dict[str, Any] | None = None,
    open_browser: bool = True,
) -> None:
    if host not in {"127.0.0.1", "::1", "localhost"}:
        raise ValueError("仪表盘仅允许绑定本机地址；如需共享请通过受认证的反向代理发布")
    listen_port = int(port)
    slug = dashboard_project_slug(project_root)
    url_prefix = dashboard_url_prefix(project_root)
    state = DashboardState()
    allowed_assets = {"performance_comparison.png", "hit_distribution.png", "number_trend.png"}

    class Handler(BaseHTTPRequestHandler):
        server_version = "TsysDashboard/1.0"

        def log_message(self, message: str, *args: Any) -> None:
            logging.debug("网页请求 | 事件=WEB_REQUEST | 客户端=%s | 信息=%s", self.client_address[0], message % args)

        def _client_disconnected(self, error: BaseException) -> bool:
            if not _is_client_disconnect(error):
                return False
            logging.debug(
                "网页客户端已断开 | 事件=WEB_CLIENT_DISCONNECT | 路径=%s | 原因=%s",
                getattr(self, "path", "-"),
                error,
            )
            self.close_connection = True
            return True

        def _json(self, payload: Any, status: int = 200) -> None:
            try:
                body = json.dumps(payload, ensure_ascii=False, default=json_default).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Cache-Control", "no-store, max-age=0")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            except Exception as error:
                if self._client_disconnected(error):
                    return
                raise

        def _error(self, error: Exception, status: int = 400) -> None:
            if self._client_disconnected(error):
                return
            logging.exception("网页请求失败 | 事件=WEB_ERROR | 路径=%s", self.path)
            try:
                self._json({"error": str(error)}, status)
            except Exception as write_error:
                if not self._client_disconnected(write_error):
                    raise

        def _redirect(self, location: str) -> None:
            try:
                self.send_response(302)
                self.send_header("Location", location)
                self.send_header("Cache-Control", "no-store, max-age=0")
                self.send_header("Content-Length", "0")
                self.end_headers()
            except Exception as error:
                if self._client_disconnected(error):
                    return
                raise

        def finish(self) -> None:
            try:
                super().finish()
            except Exception as error:
                if not self._client_disconnected(error):
                    raise

        def _route_path(self) -> str:
            path = urlparse(self.path).path
            return _strip_dashboard_prefix(path, url_prefix)

        def do_GET(self) -> None:
            try:
                raw_path = urlparse(self.path).path
                if raw_path in {"/", url_prefix}:
                    self._redirect(url_prefix + "/")
                    return
                path = self._route_path()
                if path == "/favicon.ico":
                    self.send_response(204)
                    self.end_headers()
                    return
                if path == "/":
                    body = _dashboard_page(slug, url_prefix).encode("utf-8")
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.send_header("Cache-Control", "no-store, max-age=0, must-revalidate")
                    self.send_header("Pragma", "no-cache")
                    self.send_header("Expires", "0")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                    return
                if path == "/api/status":
                    try:
                        self._json(status_callback(state.snapshot()))
                    except Exception as error:
                        if not self._client_disconnected(error):
                            self._error(error, 500)
                    return
                if path.startswith("/logs/"):
                    filename = Path(path).name
                    if filename not in allowed_assets:
                        self.send_error(404)
                        return
                    asset = project_root / "logs" / filename
                    if not asset.exists():
                        self.send_error(404)
                        return
                    body = asset.read_bytes()
                    self.send_response(200)
                    self.send_header("Content-Type", "image/png")
                    self.send_header("Cache-Control", "no-store, max-age=0, must-revalidate")
                    self.send_header("Pragma", "no-cache")
                    self.send_header("Expires", "0")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                    return
                self.send_error(404)
            except Exception as error:
                if not self._client_disconnected(error):
                    self._error(error, 500)

        def do_POST(self) -> None:
            path = self._route_path()
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if length > 25 * 1024 * 1024:
                    raise ValueError("上传内容超过 25MB 限制")
                if path == "/api/run":
                    payload = json.loads(self.rfile.read(length) or b"{}")
                    if not state.start():
                        self._json({"error": "已有实验正在运行"}, 409)
                        return
                    logging.info("网页实验已接收 | 事件=DASHBOARD_RUN_ACCEPTED | 参数=%s", payload)

                    def job() -> None:
                        try:
                            result = run_callback(payload)
                        except Exception as error:
                            logging.exception("网页实验运行失败 | 事件=DASHBOARD_RUN_FAILED")
                            state.finish(error)
                        else:
                            state.finish(result=result)
                            logging.info(
                                "网页实验运行完成 | 事件=DASHBOARD_RUN_FINISHED | 有效期数=%s | 最新期=%s",
                                result.get("data", {}).get("periods"),
                                result.get("data", {}).get("latest_period"),
                            )
                            try:
                                from src.report.notification import send_result_email

                                send_result_email(
                                    result,
                                    notification_config or result.get("config_snapshot", {}),
                                    project_root=project_root,
                                    dashboard_url=f"http://{host}:{listen_port}{url_prefix}/",
                                    delay_seconds=1.0,
                                )
                            except Exception:
                                logging.exception("网页实验邮件通知失败 | 事件=DASHBOARD_EMAIL_NOTIFICATION_FAILED")

                    threading.Thread(target=job, name="tsys-experiment", daemon=True).start()
                    self._json({"started": True}, 202)
                    return
                if path == "/api/upload":
                    if not state.start_upload():
                        self._json({"error": "实验运行或数据导入进行中，暂不允许修改历史数据"}, 409)
                        return
                    try:
                        fields, files = _multipart_fields(self.headers.get("Content-Type", ""), self.rfile.read(length))
                        if "file" not in files:
                            raise ValueError("请选择 CSV 或 XLSX 文件")
                        filename, content = files["file"]
                        result = upload_callback(filename, content, fields.get("mode", "replace"))
                    except Exception as error:
                        state.finish_upload(error)
                        raise
                    else:
                        state.finish_upload()
                        self._json(result)
                        return
                self.send_error(404)
            except ValueError as error:
                if self._client_disconnected(error):
                    return
                self._json({"error": str(error)}, 400)
            except Exception as error:
                if not self._client_disconnected(error):
                    self._error(error, 500)

    server, listen_port = _bind_dashboard_server(host, listen_port, Handler)
    url = f"http://{host}:{listen_port}{url_prefix}/"
    if listen_port != int(port):
        print(f"端口 {int(port)} 已被占用，已改用 {listen_port}。", flush=True)
    logging.info("本地网页服务启动 | 事件=DASHBOARD_STARTED | 地址=%s", url)
    logging.info(
        "网页流程说明 | 事件=DASHBOARD_PIPELINE | 模型=频率、时间衰减、贝叶斯、逻辑回归、XGBoost、形态记忆 | "
        "融合=红蓝球分色自适应 | 优化=贪心覆盖、遗传算法、综合3注 | 验证=N到N+1严格左截断"
    )
    print(f"本地操作界面已启动：{url}", flush=True)
    print("详细运行日志将在此窗口实时输出，关闭本窗口或按 Ctrl+C 停止服务。", flush=True)
    if open_browser:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever(poll_interval=0.4)
    except KeyboardInterrupt:
        print("\n服务已停止。")
    finally:
        server.server_close()
        logging.info("本地网页服务停止 | 事件=DASHBOARD_STOPPED")

