参考引擎接入

tsys/ 来自用户提供的 refs/Tsys；保留原 Python 代码和算法配置。config/config.yaml 的 notification 块被整体替换为 email.enabled=false，所有邮件地址和凭据均未打包。manifest.json 记录原文件摘要和交付文件摘要。没有复制用户历史日志、模型 pickle 或生成报告。

worker.py 使用独立 Python 3.10 环境，通过原 main.run_pipeline(config) 运行默认100期滚动验证和当前原始5注。随机种子2026，模型和生成算法配置保持原值。项目CSV有4条早期非标准星期日期，适配器保持日期原值，按年份、唯一、递增、常规星期及明确白名单验证；仅关闭参考cleaner重复的固定星期预检，并在结果metadata列出例外和实际有效配置摘要。白名单为04113/2004-12-10、04114/2004-12-13、05017/2005-02-16、06084/2006-07-21；新的例外会拒绝。日历特征和source日期来源标记保持启用。输入是本项目 CSV 的规范化快照，7位参考期号由20加5位项目期号形成。每次运行使用新的私有工作目录，既不读取上次模型缓存，也不改动 refs 原件。通知入口不会调用。

依赖使用官方同发行包xgboost==3.0.2的manylinux2014 CPU wheel（固定官方URL和SHA），无需CUDA/NCCL，CPU模型配置保持原值。

安装：uv venv --python 3.10 /data/apps/lottery-ssq/reference-venv
uv pip install --python /data/apps/lottery-ssq/reference-venv/bin/python -r current/reference/requirements.txt

手动计算：reference-venv/bin/python current/reference/worker.py --data current/data/ssq.csv --cache-dir recommendations/reference-cache --work-dir reference-runtime/work --ssq current/ssq --publish-dir publish

参考计算较慢，使用独立 ssq-reference.service 和 timer，不在网页请求或原180秒周期服务内运行。完成后调用 ssq reference-freeze，仍由 Rust 检查开奖截止、数据摘要和目标期。已有结果保持首次文件字节；输入改变时重新运行。status.json 显示 running/ready/cached/error。reference-runtime/work 保存计算日志及报告，属于独立运行数据，部署不覆盖；recommendations/reference-cache 随原 recommendations 备份。

生产应复制两个 unit 至 /etc/systemd/system，daemon-reload 后 enable --now ssq-reference.timer。发布切换前暂停新timer并等待计算结束；回滚保留参考数据。工作目录中含本地审计密钥，应只由服务用户访问，禁止公开下载整个工作目录。

worker cache 不是实际开奖前观察档；只有 Rust 成功截止前封存的档案计入实际观察。已开奖历史回放、参考旧日志和新期缓存均不补造旧期实际推荐。
