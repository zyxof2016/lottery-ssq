# 现有生产部署

2026-11-07融合重号修复v2：默认融合推荐移除两层显式重号加分，旧版v1原档及原参考5注保持。新增 `recommendations/fusion-v2/{issue}.json` 与 `recommendations/reviews/{issue}/fusion-v2.json`，随recommendations完整备份；cycle和参考冻结回调沿用现有timer。部署复制原reference目录、不改Python环境，先只读验证23个旧接口和运行原档，再正式首次封存当前v2并核验线上默认票面及手动交集。说明见 [融合重号修复](fusion-repeat-fix-2026-11-07.md)。

2026-11-06胆拖完整票面搜索优化：新增只读GET `/api/reference-pools/dantuo-audit?issue=26128`，在原融合胆拖折叠区展示同池精确搜索对照；不新增观察池、运行目录或自动封存，不改原概率模型、参考环境和已存票面。保留当前reference目录及两个timer，发布只读核对22个旧接口、全部原运行文件与新搜索结果。复盘及复现见 [胆拖深度优化](dantuo-deep-review-2026-11-06.md)。

2026-11-06参考运行档位修复：新计算从源码dashboard_run入口应用完整档位，旧首次缓存保持真实supplied_default标记。部署时复制旧release的reference目录，再覆盖本次worker.py、test_worker.py、README.txt；tsys与manifest不变，保留原环境、工作报告和私有审计密钥。新增只读来源核验及单条预测/输入CSV下载，线上验收不触发模型重算。具体差异见 [参考一致性核对](reference-parity-2026-11-06.md)。

2026-11-06融合推荐新增2胆5拖2蓝、3胆5拖1蓝，各10注20元。独立记录在 `recommendations/fusion-dantuo/{issue}.json`，复盘在 `recommendations/reviews/{issue}/fusion-dantuo.json`，随recommendations全目录备份。新增GET/POST封存/原档下载接口 `/api/reference-pools/fusion-dantuo`，沿用原cycle；不改原参考、融合8注或参考预算档。发布先只读核对20个旧接口及全部运行文件，再正式封存当前下一期两种方案。前端同一组件用于当前推荐及历史独立复盘，缺档不补造。规格和入口见 [融合胆拖说明](fusion-dantuo.md)。

2026-11-06完整概率校准增加 `recommendations/calibration/{issue}.json` 及 `recommendations/reviews/{issue}/calibration.json`，继续随recommendations备份。使用原cycle和timer，仅新增概率评分，不增加购票池；旧joint-red若全33红在原精度下同分，跳过新自动研究票面，已有原档继续按旧规则读取结算。新增GET `/api/jackpot/calibration`、POST同路径`/freeze`、GET同路径`/archives/{issue}`。发布先只读核对19个旧接口、全部原运行文件和参考环境，再正式启动当前下一期概率前瞻；不上传本地测试档案。模型v1单独冻结在calibration_model_v1.rs，后续模型改动须另建版本。详见 [26127深度复盘](deep-review-2026-11-06.md)。

2026-11-05新增只读 `GET /api/jackpot/score-v2` 和头奖页完整组合评分v2对照。新评分代码独立，不修改旧策略、Protocol、封存或cycle，不新增运行目录。当前候选未过验证门槛，不输出v2推荐或开启新观察。发布核对18个原只读接口不变、全部运行原档字节保留、旧参考环境及两个timer正常；只读访问新接口不会创建推荐。数学、时间切分与限制见 [本次深度复盘](deep-score-review-2026-11-05.md)。

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

融合完整票组自2026-10-30起另存于 `recommendations/reference-tickets`，对应复盘为 `recommendations/reviews/{issue}/reference-tickets.json`，继续随recommendations目录完整备份。旧reference文件不补写；发布后先只读核验原档，再由实际封存或原自动任务创建当前下一期的完整票组。

2026-11-02起参考同次报告的原始胆拖／复式单独导出为 `recommendations/reference-dantuo-cache/{输入摘要}.json`，截止前首次封存到 `recommendations/reference-dantuo/{期号}.json`，逐注复盘为 `recommendations/reviews/{期号}/reference-dantuo.json`，继续随整个recommendations目录备份。更新release中的reference/worker.py，保留reference-runtime原始工作报告与私有审计密钥；旧5注缓存及reference原档不改写，未封存的历史胆拖不补造。后台可从已有同次工作报告导出当前方案，无需重跑模型。

2026-11-02小奖覆盖扩展新增bluefocus8，新期首次daily为26策略；已封存旧daily不追加。旧日档缺该策略时，cycle通过既有interactive目录封存当前下一期完整票组（prize_coverage v1），继续随recommendations完整备份和原有reviews流程对奖。部署只读核验后再正式生成当前方案，不能把本地浏览器测试产生的interactive档案上传生产。新增GET /api/prize-coverage不写盘，核对实际样本与历史回放分列、16红/8个不同蓝球/8注16元、原档保留及截止边界。

2026-11-02头奖目标纠偏：后续首次daily仍26策略，jackpot8替代自动小奖bluefocus8；已保存的小奖、每日及参考档案不改写。cycle通过既有interactive通道为当前下一期单独封存jackpot_goal v1。新增GET /api/jackpot只读，核对4个完整六红组合×2蓝=8注16元、首次留档和逐注6红1蓝指标；头奖模式禁用仅按并集减注。新旧目标档案使用同一recommendations备份及原复盘通道，部署不得上传本地模拟头奖或测试档案。

2026-11-03新增只读GET /api/jackpot/model-review及头奖页评分复盘。研究模型不接入生成或cycle，不新增运行目录或改写原档；候选尚未优于均匀且无独立前瞻样本，不能自动替换头奖v1。发布核对原15个只读接口结果、当前头奖与小奖原档摘要、待开奖空指标，以及研究面板的集中度、评分对照和禁用发布状态。两个timer和全部运行目录继续保护，部署验收仅GET，不为测试新增推荐。

2026-11-04固定联合评分研究启用独立前瞻观察：recommendations/joint-red/{issue}.json及recommendations/reviews/{issue}/joint-red.json随现有recommendations全目录备份。沿用cycle，不新增timer；它只追加研究记录，不调整每日策略或现有头奖原档。GET /api/jackpot/model-review增加最新原档归因、prospective及current_observation；POST /api/jackpot/model-review/freeze绑定当前target/data_sha256，GET /api/jackpot/model-review/archives/{issue}核验并下载原文件。部署前暂停两个timer并等运行结束；先只读核验16个旧接口和运行文件，再正式启动当前下一期研究观察，单独记录新增文件、封存时间和输入摘要。历史预览及隔离测试档案不导入。

2026-11-04红蓝联合选票观察新增recommendations/blue-joint/{issue}.json和reviews/{issue}/blue-joint.json，继续随recommendations完整备份。沿用原cycle封存及复盘，不新增timer、不改变旧每日策略。GET /api/blue/deep为只读，POST /api/blue/deep/freeze绑定target/data_sha256，GET /api/blue/deep/archives/{issue}下载核验后的首次原档。新静态资源blue-deep.js同时在主界面和pools加载，核对代理前缀、手机显示及号码文字对比度。发布前保护两个timer和全部运行文件；先只读核验17个旧接口、原头奖与红球研究摘要，再正式启动当前下一期联合选票观察。不得上传隔离环境生成的blue-joint文件。
