# 双色球滚动验证

这个程序度量一组冻结规则相对均匀开奖的离差。开奖按规则是均匀抽取，报告里的数字不是中奖承诺，也不是扣完奖池结构之后的期望收益。确认窗里没有通过 Holm 校正的优势，是符合这个设计的读法。

确认窗钉在期号 25111，长度 156 期，在当前数据上止于 26115。以后追加的开奖进入跟踪窗，不把确认窗往后滑，也不进入预注册的 25 项检验。融合权重只用 25111 之前的样本外结果。专家状态可以用到发布截止期。权衡网格只描述证据保留和覆盖，不回写发布规则。

## 命令

```bash
cargo run --release -- protocol
cargo run --release -- backtest --data data/ssq.csv --out report.txt --json report.json
cargo run --release -- publish --data data/ssq.csv --audit audit/log.jsonl --out publish --target 下一期期号 --not-after 开奖时刻的Unix秒
cargo run --release -- verify --data data/ssq.csv --manifest publish/对应清单.json --audit audit/log.jsonl
cargo run --release -- fetch --out data/ssq.csv
cargo run --release -- serve --data data/ssq.csv --audit audit/log.jsonl --out publish
cargo run --release -- cycle --data data/ssq.csv --audit audit/log.jsonl --out publish --review-dir reviews
```

浏览器打开 http://1.14.208.189/ssq/ 可以查看下一期票面、确认窗报告，并做开奖前留痕和开奖后复算。

首页按 lottery-k8 的七个页签组织：方案生成、我的方案与对奖、每期推荐与复盘、开奖历史、科学回测与优化、研究评估、系统设置。预设或参数需点击生成，打开页面和同步开奖只刷新数据；保存、复制绑定生成时的期号与参数。复盘直接展示各策略原始红蓝号码，再查看完整票组与逐注对奖。静态资源随程序内置，支持 `/ssq/` 代理路径。

本地网页默认监听 http://127.0.0.1:6791/ 。`serve` 只提供网页和接口；自动同步、封存和复盘需要外部定时任务周期性调用 `cycle`，建议每 30 分钟一次。

每期推荐与复盘入口是 `/reviews`（反向代理部署时为 `/ssq/reviews`），首页导航也可进入。新建的每期留档包含原 7 个预设和少注覆盖 v1 的 9 个方案，共 16 个完整单注票组，并保留所用的评分、名次、专家权重和筛选轨迹；均匀分散对照不填写模型评分。页面支持同预算对照、同覆盖减注结构试算、漏斗、号码筛选、逐注对奖及 JSON/CSV 导出，已有档案保持原方案数量。

`cycle` 在常规开奖截止前自动留档下一期，开奖数据同步后按原票组保存复盘；网页也提供手动留档、同步和核对入口。生成或保存个人方案时会留下实际推荐档案，重复操作保留首次版本。旧版保存记录可按原票面复盘，但不补造当时的模型证据。历史没有实际推荐的期号明确标为缺档，待开奖的命中和奖金显示为空。档案放在 `--out` 发布目录同级的 `recommendations/`，详情见 [每期推荐与复盘说明](docs/recommendation-reviews.md)。

`backtest` 不访问网络。`fetch` 从第三方数据源 500.com 下载，在解析失败、或期数少于 3000 时不会替换已有 CSV。

发布必须提供 `--not-after`（留痕截止时间的 Unix 秒），到达该时刻即拒绝写入。显式目标期号必须是晚于截止期的 5 位数字；复算时必须匹配截止期后的第一期开奖，否则拒绝计分。未指定目标时仍使用 `after:截止期`，旧版这类清单继续支持复算。发布同一目标时，内容哈希不同会拒绝覆盖。网页保存冻结方案时先封存，封存失败不会追加方案记录。

融合权重始终只使用期号小于 25111 的样本外记录，即使数据缺少 25111 期也不会回退使用确认窗；边界前不足 30 个样本外记录时拒绝拟合。哈希用于核对内容和复算一致性；要证明开奖前已经发布，仍需把哈希保存到独立、可信的时间记录中。

排序前分数收到 1e-9，权重收到 1e-12，使回测、发布和复算走同一条名次。

## 小号码池筛选

入口 `/pools`（代理部署时 `/ssq/pools`）。提供 14／16／18／20／22／24 红六档模型观察池，每个模型逐层包含，每档可比较 1–3 蓝。新版首次留档同时冻结加权融合、等权融合、固定随机红球对照各 18 个观察池，三者共用加权蓝球排名；另保留原有 9 个小池，共 63 个池。页面以相同规格、三模型都已保存的共同期次比较全6红率和漏红数，待开奖不计入比例，旧档不追加对照。开奖后按原排名显示六个红球的位置、精确最小全包前缀及最小全包观察档；最小前缀属于开奖后诊断，不冒充原推荐。历史因果回放与实际留档统计分开展示，模型和冻结权重不按近期结果自动更换。

第一份六档观察记录为 [26118 期号码池](docs/model-pool-observation-26118.txt)，[JSON](docs/model-pool-observation-26118.json) 绑定原始每日档案摘要，当前待开奖。

`cargo run --release -- pool-review --data data/ssq.csv --out small-pool.txt --json small-pool.json` 可复现因果回放。[红球数量比较](docs/red-pool-size-review-2026-10-14.txt) 与 [完整曲线数据](docs/red-pool-size-review-2026-10-14.json) 为探索性回放，不作为预测优势证明；[原小池复盘](docs/small-pool-review-2026-10-14.txt) 保留原快照。

## 选号依据复盘

`cargo run --release --example selection_review -- data/ssq.csv > selection-review.json` 提供最近 30/60 期、固定确认窗和后续跟踪窗的同预算回放，比较原冻结票组、同池均衡组票、15 码候选池和四蓝。每组均为 8 注、16 元，同时报告单注命中、并集覆盖及其均匀基准、单号曝光和蓝球命中注数。研究程序只读数据，实验票面用于前瞻观察。

本次结论和复现说明见 [2026-10-13 复盘](docs/selection-review-2026-10-13.txt)，精确结果见 [复盘数据](docs/selection-review-2026-10-13.json)。
