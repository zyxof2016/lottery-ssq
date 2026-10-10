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
```

`backtest` 不访问网络。`fetch` 在解析失败、或期数少于 3000 时不会替换已有 CSV。发布同一目标时，内容哈希不同会拒绝覆盖。`--not-after` 过后拒绝再发布。

排序前分数收到 1e-9，权重收到 1e-12，使回测、发布和复算走同一条名次。
