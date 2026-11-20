# Phase 0/1 构建日志

## 2026-11-15

- 初始化 Windows 工作区的 Git 仓库，分支名为 main。系统安装有常规和自由线程 Python 3.13。使用常规 Python 3.13 创建 .venv。系统列出 NVIDIA GeForce GTX 1080 Ti，本轮执行未用 GPU。
- 执行 `git init -b main`、`py -3.13 -m venv .venv` 和 `.venv/Scripts/python.exe -m pip install -e ".[test]"`。锁定项目直接依赖。PyAV 使用随包提供的 FFmpeg 库。
- 执行 `.venv/Scripts/python.exe tools/inspect_video.py VIDEO --at 0 --range 21 22 0.5 --output outputs/video-inspection`，输入素材见清单中的 prior-desktop-reference。完整解码 4,724 帧，720×330，约 157.47 秒，未报损坏帧或缺失呈现时间戳。输出 4 张 PNG 和 [检查记录](../outputs/video-inspection/reference-inspection.json)。
- 执行 `.venv/Scripts/python.exe -m pytest -q` 与 `.venv/Scripts/python.exe tools/verify.py --profile phase-01`。最终 26 项测试通过，pip check 通过，两个已知文件 SHA-256 一致，三个预期素材条目 SKIPPED。参阅 [机器可读报告](reports/phase-01.json)。
- 独立 Reviewer 复跑同一 profile 并生成 [复验报告](reports/phase-01-independent.json)。审查发现和修正结果见 [审查记录](reviews/phase-01.md)。

## 2026-11-15 · Phase 2

- 登记两份微信回放及十个截图来源，并验证复制品的哈希。两份回放都为 2800×1280，分别解码 9,417 帧与 8,216 帧。九种阵法配置图中，地载阵的两份附件哈希相同。
- 建立本场计数区域、回合区域和 7 帧人工对照。OpenCV 安装请求的网络代理发生连接重置，本轮从同一台机器的常规 Python 3.13 拷贝了已安装的固定版本 OpenCV 及 NumPy 到项目环境，pip check 通过。
- 执行 `.venv/Scripts/python.exe tools/verify.py --profile phase-02`。29 项测试通过，所有素材哈希通过，详细回放提取检查通过。报告记录完整读数覆盖率和遮挡误读。

## Phase 3 · 双录像对齐和复核片段

- 使用两份已登记高分辨率回放的出场、技能、结算数字与换回合节点建立分段时间映射。检查了 12 个拟合锚点和 11 个检查点，误差中位数 40 ms，最大 281 ms。
- 运行 `tools/align_videos.py` 与 `tools/build_dataset.py --export-clips`。生成 16 条映射计数变化、3 条超出范围变更记录及 5 条困难负样本。人工复核完成 7 条行动者标注。其余 9 条映射记录保留候选。21 段普通录像的复核副本均可解码。
- 执行 `tools/verify.py --profile phase-03`。结果见 [统一验证报告](reports/phase-03.json)。报告覆盖回归测试、输入哈希、Phase 2 重跑、配对对齐、片段生成及比赛分组校验。

## Phase 0–3 审查修复 · 2026-11-16

- 修复倒地宠通过固定槽位误计药次，约束官方来源为配对中的详细录像并校验事件日志哈希。补充 3 项回归用例。
- 在原校准录像中裁出双方阵法图标，固化 7 帧参考读数及各栏稳定读数覆盖下限。以 Windows 系统 OCR 验证原始与颜色分割后的计数区域，默认提取继续使用模板法。
- 实际运行 `python tools/compare_counter_ocr.py` 和 `python tools/verify.py --profile phase-03`。证据见 [审查记录](reviews/phase-00-03-audit.md)。

## Phase 4 · 2026-11-17

- 以普通回放的源视频帧配置人物和宠位，运行 `python tools/predict_clean.py` 和 `python tools/evaluate.py`。修整误判较多的候选阈值，合并同动作多帧技能飘字。
- 复核报告中的归属、误报和未标注片段。使用合成测试检查连续帧去重、事件一对一、来源哈希和未知标注隔离。
- 运行 `python tools/verify.py --profile phase-04`。36 项测试通过。单场结果及接续点见 [Phase 4 记录](plans/phase-04.md)。

## Phase 5 · 2026-11-17

新增片段训练入口、可重复参数和分组检查。实际执行 `tools/train_model.py --help`、无 `--smoke` 的独立验证拒绝检查及 `tools/verify.py --profile phase-05`。38 项测试通过，开发试跑及 ONNX 序列化回读通过。Runtime 推理、跨场模型比较未验证。

## Phase 5 验收及 Phase 1–5 复核 · 2026-11-19

- 登记独立的龙飞阵对虎翼阵录像及复核标签。冻结跨场分组和时间容差，执行三个模型的训练、导出、CPU 推理一致性和整段候选对照。结果见 [Phase 5 计划与证据](plans/phase-05.md)。
- 修复重放中的已知空宠位用药误计。评估改用事件类型和时间做一对一匹配，允许空预测，未知行动者从对应归属分母排除。
- 复跑 Phase 1–5 验证。独立检查见 [阶段审查](reviews/phase1-5-audit-20261118.md)。

## Phase 6 人工反馈准备

从现有训练比赛选出两个已核实用药额度、行动者仍未知的片段。两个本地短片完整解码分别为 56 和 78 帧。已保存帧图和来源索引，等待反馈后核对归属。见 [人工复核清单](../data/annotations/human-review-01.json)。

收到反馈后重新抽帧核对。A 指认的快乐加倍对应相邻的 L5 药次，R4 原候选仍保留未知。B 确认 L2 宠物施药。使用 v2 标签重建训练片段索引，9 条已复核资源事件、5 条困难负样本、7 条待核实候选，比赛分组检查通过。见 [v2 标注](../data/annotations/thunder-tiger-review-v2.json)和[构建结果](../outputs/phase-06/review-01/dataset/dataset_report.json)。

## Phase 6 鸟翔阵重叠区间

- 独立复核 5 个事件和第 1–13 回合末计数。冻结基线复跑后，与详细版计数逐项比较。导出漏报帧和开场误报证据。见 [重叠评估](../outputs/phase-06/bird-evaluation.json)。
- 独立审查发现，时间容差可能使已核实无资源事件区间的候选抢占后续匹配，计数检查点也可能漏掉中间读数倒退。两处均已修复并复测。完整区间验证入口通过，44 项测试通过。见 [复跑报告](reports/phase-06-overlap.json)。
- 对详细录像的数字区域单独校准，20 个检查点可复核的药次和宠次读数分别达到 152/188 和 180/191。十个阵位的药次和宠次增量与人工起止计数一致。L1 第 8、9 回合的两次用药被自动读数合并，逐事件仍需结合画面。见 [详细版提取报告](../outputs/phase-06/bird-counter-numeric/counter_extraction_report.json)。
- 根据详细版数字变化定位事件后逐帧检查两份录像，参考标注扩为 10 次召唤及 5 次用药。复跑评估，10 次召唤均无匹配，5 次用药均有匹配，其中 1 次记错主人。44 项测试复跑通过。见 [独立审查](reviews/phase-06-overlap.md)及[人工事件标注](../data/annotations/bird-mirror-reviewed-events.json)。

## Phase 6 风扬阵对云垂阵整场对照 · 2026-11-20

- 登记两份新录像为同场独立测试资产。校准风扬、云垂人宠站位，完成 18 个回合末检查点及终局核对。复核了 55 次药次和 22 次宠次增加。见 [素材清单](../data/manifests/assets.json)及[终局与逐事件标注](../data/annotations/wind-cloud-reviewed-events.json)。
- 使用旧训练场开发金环入场和用药施法时序规则。从普通画面输出候选，用同场详细版评分。旧版与新版回合末计数一致由 94/360 升至 190/360，终局由 0/20 升至 4/20。见 [阶段对照](plans/phase-06.md)。
- 独立审查修复一对一匹配重复计分，补齐整场参考和回合末计数校验、同帧归属歧义处理、模板分组约束和实际候选可用时点约束。见 [独立审查](reviews/phase-06-final.md)和[终版复跑](reports/phase-06-final.json)。
- 终版 25 项流程检查通过，加入匹配回归后独立运行 56 项测试通过。校验两份报告输入和检测代码哈希，共导出 64 张新版误报或漏报证据。阵位人工校准，设备实时性能未测。见 [新版报告](../outputs/phase-06/wind-cloud/temporal-metrics.json)。

## Phase 6 第二批人工复核 · 2026-11-20

- 保存 M1 到 M3 的用户答复，追加 v5 训练标注。21 段复核片段可解码，13 条已复核资源事件、5 条困难负样本及 3 条待核事件，分组检查通过。见 [复核记录](../data/annotations/human-review-02.json)和[数据集报告](../outputs/phase-06/review-02/dataset/dataset_report.json)。
- 当前时序检测识别到 9 条已复核用药中的 8 条，M1 的受药特效偏弱。用 v4、v5 分别重训线性分类器，在同一独立验证集分类正确数由 2/6 降为 1/6。见 [阶段对照](plans/phase-06.md#人工反馈)。
- 在普通版两组旧回放逐帧复核六次确认出宠，保留金环、新宠入场及「死亡召唤」的对照。六个窗口均缺可确认的独立「召唤」文字正样本。见 [训练录像](../outputs/phase-06/summon-text/train/training-text-audit.json)和[验证录像](../outputs/phase-06/summon-text/validation/validation-review.json)。
- 实现可选的弱受药特效补判规则。独立对照验证回放候选相同，原始训练录像经普通预测入口复跑后补回 M1，已复核用药命中 9/9。63 项测试通过。见 [原始录像复跑](../outputs/phase-06/review-02/medicine-opt-in-integrated-comparison.json)和[验证对照](../outputs/phase-06/review-02/medicine-weak-amber-trial/production-dragon-tiger.json)。

## Phase 6 参战可见性接续点 · 2026-11-20

- 按用户确认的 G17 约束完成离线模拟。左队状态条在特征提取前遮蔽，敌方用药和召唤读取可见动作与特效。我方状态条仍可参与检测。完整训练片段和针对性测试已复核。
- 用户请求暂停时，`python tools/verify.py --profile phase-06-participant` 已生成预测、整场评估和 PASS 报告。下次从现有报告继续，按左右阵营核对用药与召唤的命中、归属和累计误差，再汇总实战视角的证据边界。实际参战录像验证待开展。见 [验证报告](reports/phase-06-participant.json)、[整场模拟结果](../outputs/phase-06/wind-cloud/participant-simulation/metrics.json)及[独立审查](reviews/phase-06-participant.md)。

## Phase 6 参战可见性整场结果 · 2026-11-20

- 并行复核用药、出宠和证据。用保存的同场输出按队重做一对一匹配，补记了两版各一次跨队用药错归。回合末和终局逐人计数均已核对。见 [分队对照](../outputs/phase-06/wind-cloud/participant-simulation/comparison.json)和[阶段结果](plans/phase-06.md#参战可见性模拟)。
- 保存的整场 profile 共 23 项流程检查通过，哈希核对后沿用该次录像输出。最新代码独立复跑 76 项测试通过。离线参战模拟识别仍低于目标，重点跟进左方宠物施药漏检和出宠事件时点。见 [独立审查](reviews/phase-06-participant.md)。

## Phase 6 参战模拟动作时序复验 · 2026-11-20

- 并行验证敌方出宠记时和宠物施药的可见线索。出宠用首个获新宠模型变化佐证的金环帧记时，保留实际候选可用时点。分队事件评估纳入统一报告。用药金色脉冲规则保留为同场实验。见 [训练与测试结果](plans/phase-06.md#参战模拟动作时序复验)。
- 运行 `python tools/verify.py --profile phase-06-participant --output outputs/phase-06/participant-action-cues/wind-cloud-after --report harness/reports/phase-06-participant-action-cues.json`。79 项测试和 23 项流程检查通过。敌方 8 条出宠候选时点前移，3 条落入原评分容差，18 个回合末及终局计数逐项一致于改动前。见 [验证报告](reports/phase-06-participant-action-cues.json)和[改动前后对照](../outputs/phase-06/participant-action-cues/wind-cloud-comparison.json)。

## Phase 6 固定宠位留空复核 · 2026-11-20

- 并行复核雷绝、龙飞和鸟翔的 16 次已确认出宠。遮蔽左方状态条后，每例都采到至少一帧固定宠位留空，随后该位可见新宠。短暂位移、复生与外观相似的换宠均有对照。试验阶段记录可见状态和对照证据，自动识别待进一步验证。见 [留空复核](plans/phase-06.md#固定宠位留空复核)。

## Phase 6 出宠多线索对照 · 2026-11-20

- 现有检测用 `tools/predict_clean.py --layout configs/vision/bird-mirror.json --detector temporal --view participant` 处理整段普通录像 4,114 个采样帧，通过 `tools/evaluate_phase6.py` 对齐、评分。独立宠位实验固定由雷绝场拟合的参数后复查鸟翔十例和开场无出宠区间。现有检测命中 2 例，宠位实验命中 6 例，重合 1 例。开场无出宠时，宠位实验误发 7 条候选。见 [逐例对照](../outputs/phase-06/pet-vacancy-multicue/bird-cue-comparison.json)及[执行证据](plans/phase-06.md#出宠多线索对照)。
- 并行检索公开手游录像动作素材，保存四个近期入口、可复核范围及游戏内录像计次说明。见 [公开录像线索](../docs/RESEARCH_NOTES.md#公开录像线索)。
- 复跑 79 项测试通过。当前出宠计数仍按原检测执行，多线索结果为离线验证记录。

## Phase 6 计次反例与跨场训练 · 2026-11-20

- 从雷绝对虎翼普通和详细成对回放确认原宠返位、死亡召唤两段无资源计次负例。复跑了全宠位出宠融合，以及遮住左方状态条的线性片段训练。鸟翔出宠六条命中中有四条是新增，但回合末宠次误差总和由 49 增至 65。遮罩片段模型在龙飞六个窗口无准确率增益。见 [跨场训练记录](plans/phase-06.md#计次反例与跨场训练)。
- 冻结用药补判规则后复跑风扬、鸟翔两场，分别处理 6,004、4,114 个采样帧。风扬整场指标与基线一致，鸟翔已核实用药由 2/5 升至 3/5。79 项测试通过。参战实录和未知施药者身份仍待核实。

## Phase 6 第三批人工反馈 · 2026-11-21

- 登记三张孩子造型图，记录「小星星」出战及风扬 2 号宠物待机。新增 `SUMMON_CHILD` 账本及其快照、候选确认、重复、重放和撤回测试。用户确认的孩子事件从宠次修订为单独孩子计次，云垂阵 4 号重放为孩子 `1/1`、宠物 `1/6`。见 [人工反馈](../data/annotations/human-review-03.json)及[账本复跑](../outputs/phase-06/child-idle-feedback/manual-child-replay/ledger-check.json)。
- 运行 `python tools/verify.py --profile phase-06-participant --output outputs/phase-06/child-idle-feedback/default-replay --report harness/reports/phase-06-child-idle-feedback.json`。83 项测试和统一验证通过。独立审查另复跑 28 项账本测试。视觉侧的孩子外观和用药起手对照保留为离线试验。见 [验证报告](reports/phase-06-child-idle-feedback.json)及[阶段结果](plans/phase-06.md#第三批人工反馈)。

## Phase 6 第四批人工反馈 · 2026-11-21

- 接收并核对「大圣佑」原宠出击与复活后群体治疗两段反馈。第一段作为训练中的出宠反例，第二段追加两行动标注并保留计次疑点。两场对局的数据分组已核对。见 [复核标注](../data/annotations/human-review-04.json)和[阶段对照](plans/phase-06.md#第四批人工反馈)。
- 新增多受药候选的逐目标时间证据。执行 `python tools/verify.py --profile phase-06-participant --output outputs/phase-06/feedback-shengyou-group/default-replay --report harness/reports/phase-06-feedback-shengyou-group.json`，83 项测试及统一验证通过。见 [验证报告](reports/phase-06-feedback-shengyou-group.json)。
