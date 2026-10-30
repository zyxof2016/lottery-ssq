# 现有生产部署

入口为 <http://1.14.208.189/ssq/>，HTTPS 同路径也可访问。通过 `ssh ubuntu@1.14.208.189` 使用现有 SSH 密钥连接，管理服务可用 `sudo -n systemctl`。不要在仓库中保存密钥或其他凭据。

应用根目录为 `/data/apps/lottery-ssq`，`current` 是指向 `releases/` 内某个版本目录的符号链接。版本目录包含 `ssq` 二进制和 `data/ssq.csv`。二进制内置网页与静态资源，无需单独上传前端。

`ssq-web.service` 以 `ubuntu` 运行，监听 `127.0.0.1:6791`；Nginx 将 `/ssq/` 代理到该端口。`ssq-cycle.timer` 每 30 分钟调用 `ssq-cycle.service`，继续使用原有同步、封存和复盘规则。两个服务都从 `current/ssq` 启动，开奖文件为 `current/data/ssq.csv`，审计和发布参数分别为根目录下的 `audit/log.jsonl` 与 `publish`。

以下目录属于持续保存的运行数据，位于版本目录外：`audit`、`publish`、`history`、`plans`、`reviews`、`recommendations`。不存在的目录由相应功能按需创建。部署不得清空或用本地目录整体覆盖这些记录。旧推荐不补造，已存在的每日档案不替换。

## 发布步骤

1. 核对实际分支和远端，完成本次改动所需验证，提交并推送。使用 `cargo build --release --locked` 构建；生产环境为 Linux x86_64、glibc 2.39，上传前确认二进制兼容。
2. 新建带提交号与时间的版本目录，上传二进制和版本清单，校验 SHA256。保存切换前的 `current` 目标和运行数据备份。
3. 暂停 `ssq-cycle.timer`，等当前 `ssq-cycle.service` 结束。短暂停止网页服务后，将生产正在使用的开奖 CSV 复制到新版本；保留生产数据，不用仓库内的快照替换它。
4. 通过同目录临时符号链接与 `mv -T` 原子切换 `current`，启动 `ssq-web.service`。运行 `systemctl is-active ssq-web.service`、访问 `/ssq/health` 和 `/ssq/api/overview`，并核对网页、静态资源及本次新增功能。
5. 失败时将 `current` 原子恢复到原目标并重启网页服务。成功或回滚后均恢复原本启用的 `ssq-cycle.timer`，保留旧版本和备份。

本地存在而生产缺少的真实推荐档案，只能在核对原始摘要、输入历史及目标期号后单独导入；用原子创建保证不覆盖同名档案。导入不改原文件内容、推荐时间或独立时间证明字段。临时验证目录及模拟开奖不得上传到生产。

## 发布后核验

核对提交号、二进制摘要、服务状态和定时任务状态；浏览器检查八个页签、复盘原始号码、六档观察池、代理前缀和移动端。线上检查优先使用只读接口，避免为验证而生成或保存新方案。存档、审计和已有个人方案必须保留。

## 算法集合v2兼容性（2026-10-26）

旧的三算法集合没有 `algorithm_models_version` 字段；新版保存字段值 `2` 和四个算法（新增 `dual_support`）。两种原档均按各自首次保存的内容核验，不补造新算法。26123原档继续保留8模型153池，后续新期首次冻结为9模型171池，票组策略仍为21个。

回滚前检查是否已有算法集合v2原档：`2d007ea` 及更早的服务版本无法读取四算法集合。首次发布的验证阶段暂停自动任务，因此失败时仍可按上述步骤回滚；一旦生成了v2原档，后续应选择支持v2的版本恢复或向前修复，不删除原档或改写为旧集合。

## 蓝球试验兼容性（2026-10-28）

新增四个蓝球同预算策略及可选 `blue_ranking` 训练证据，未来首次每日留档为25策略，原号码池结构不变，26124已存档保持21策略。发布后核对 `/api/blue`、四个8注16元入口及旧期原始蓝球；线上只读验收，不为测试生成或保存新方案。生产运行目录继续按原规则保护。

## 小红球池扩展兼容性（2026-10-28）

新期首次每日留档可带可选 `small_red_pools` v1，原榜与试验各4个12–16红的池，共8个红球独立池；原25策略、171个常规池不变。26124旧档不追加。发布后核对 `/api/small-red`、小池复制／导出及旧期不显示新增原档，生产只读验收；生成与未来首次留档仅由实际用户或既有自动任务触发。

2026-10-29 起，既有每日档案未包含小池时，周期可在开奖截止前另外创建 `recommendations/small-red/{issue}.json`，复盘保存到 `recommendations/reviews/{issue}/small-red.json`。备份继续覆盖整个 recommendations 目录。新接口 `POST /api/small-red/freeze` 绑定目标期号和输入摘要；生产功能验证使用只读接口，正式启用观察由实际封存或原定时任务执行。发布前暂停定时任务，核对原文件字节未变，再恢复定时任务。

## 参考5注观察运行环境（2026-10-29）

参考观察版本的release还包含 `reference/`，原有Rust服务入口、端口和CSV目录继续沿用。Python3.10独立环境为根目录下 `reference-venv`，解释器/安装文件及工作数据为 `reference-runtime`。这些路径不随release覆盖或删除；工作目录中的本地审计密钥只允许服务用户访问。安装版本与CPU wheel摘要见reference目录的requirements及constraints，禁止复制refs中的通知配置或用户模型pickle。

新增 `ssq-reference.service` 和 `ssq-reference.timer`，每30分钟检查新数据，同一输入只计算一次；CPU及内存有限额。重计算在独立进程进行，不占用网页请求或原180秒cycle。生成缓存后调用当前Rust的 `reference-freeze`，仍检查截止。两个新unit安装完成后daemon-reload、enable --now timer，并核对实际执行状态。部署后续版本前须暂停该timer并等待运行结束或安排安全切换，不能把半成品当成已发布缓存。

`recommendations/reference-cache`、`recommendations/reference`及`recommendations/reviews/{issue}/reference.json`随原recommendations一起备份。首次发布的原档核验只读；正式运算、封存另外记录输入摘要、源码/配置摘要及保存时间。回滚保留所有新观察数据，并使用能够读取参考观察的版本。`reference-runtime/work`保留运行诊断，部署前备份状态与必要诊断，不复制到公共下载路径。
