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

## Phase 6 第五批人工反馈 · 2026-11-22

- 保存鸟翔阵左方 1 号宠物「超级神马」的用药动作复核。按新标签复评先前冻结的参战模拟结果。默认输出仍为未知，补判实验正确匹配施药者、药次主人和受药者。见 [复评](../outputs/phase-06/user-review-followup/bird-medicine-review-comparison.json)。
- 执行 `python tools/verify.py --profile phase-06-overlap --report harness/reports/phase-06-feedback-super-horse.json`，28 项流程检查通过，包含 83 项测试。独立复跑 6 项评估测试，核对素材哈希、归属、时间边界和测试分组。见 [验证报告](reports/phase-06-feedback-super-horse.json)和[独立审查](reviews/phase-06-overlap.md#第五批人工反馈独立复核)。

## Phase 6 第六批人工反馈

- 更新两场测试回放参考。补齐 A 和 D 施药归属，把 B 扬手后新宠入位与 C 原鬼魂宠持续占位纳入正反例。原帧复跑重现 C 误报，确认 B 缺默认检测所需的金环候选。见 [动作诊断](plans/phase-06.md#第六批人工反馈)。
- 执行 `python tools/verify.py --profile phase-06-overlap --report harness/reports/phase-06-feedback-ghost-pet.json`，28 项检查与 83 项测试通过。风扬和鸟翔参战模拟以原预测分别重算，独立核对四段短片、源录像哈希和分组。见 [验证报告](reports/phase-06-feedback-ghost-pet.json)和[独立审查](reviews/phase-06-participant.md#第六批人工反馈复核)。

## Phase 6 默认识别复跑 · 2026-11-23

- 运行 `python tools/verify.py --profile phase-06-participant`，指定单独的风扬输出目录。26 项检查通过，含 83 项测试。鸟翔以相同参战模拟重新预测和评分，龙飞按原六个标注片段重放。三组的用药、出宠命中和错误逐项核对一致。见 [验证报告](../outputs/phase-06/accuracy-recheck-20261123/wind-validation.json)及[错例全表](../outputs/phase-06/accuracy-recheck-20261123/识别错例全表.md)。

## Phase 6 第七批人工反馈

- 登记 E 和 F 的可见动作。G 在初版评估被误配为用药时间命中，复核后保留计数变化，攻击负例从匹配候选中剔除并计入误报。现场复核使用同一对普通和详细录像的原帧。见 [第七批复核](plans/phase-06.md#第七批人工反馈)。
- 执行 `python tools/verify.py --profile phase-06-overlap --report harness/reports/phase-06-feedback-g-attack.json`，28 项检查与 84 项测试通过。另对冻结的风扬和鸟翔参战模拟重新评分，独立复跑 7 项专项测试。见 [验证报告](reports/phase-06-feedback-g-attack.json)和[独立复核](reviews/phase-06-overlap.md#第七批人工反馈独立复核)。

## Phase 6 动作起手与宠位连续性试验

- 实现两项可选离线分支，分别从施放者连续动作反查同次结算、从固定宠位验证稳定模型与名称轮廓。独立审查后复跑冻结修订版。在风扬和鸟翔比较命中、误报及累计误差。见 [比较结果](../outputs/phase-06/action-identity-20261123/识别试验结果.md)。
- 运行 `python tools/verify.py --profile phase-06-participant`，指定分开的默认输出和报告。26 项检查通过，包含 94 项测试。独立复跑用药 19 项和召唤 18 项。见 [统一验证报告](reports/phase-06-action-identity.json)及[独立审查](reviews/phase-06-participant.md#动作起手与宠位连续性独立审查)。

## Phase 6 用药动作训练与新宠入位复评 · 2026-11-24

- 用 19 段雷绝对虎翼录像拟合两个等长窗模型，冻结权重后计算龙飞、鸟翔和风扬的片段分类。另跑两场整段已检用药候选过滤，以及延长新宠模型稳定观察的同帧对照。分别保留模型、逐候选输出和对照报告。见 [训练复评](../outputs/phase-06/action-training-20261124/训练复评.md)。
- 执行 `python tools/verify.py --profile phase-06-participant`，26 项检查与 97 项测试通过。风扬默认预测与存档逐字节一致。独立复跑 5 项出宠延时相关测试。实战采集、设备识别延迟仍待真实参战录像验证。见 [统一验证](reports/phase-06-training-20261124.json)和[独立审查](reviews/phase-06-participant.md#用药训练与延时出宠独立审查)。

## Phase 6 窗口偏移训练与回合末治疗反馈 · 2026-11-24

- 保存 95 个偏移训练窗的权重、ONNX 对照和逐候选评分。两场整段对照采用相同的 49 条原用药候选，复评后保留默认识别器。鸟翔 `BM-N05` 按用户反馈补标，并分开保存原参考评分和补标后的评分。见 [多窗复评](../outputs/phase-06/window-robustness-20261124/多窗复评.md)和[独立审查](reviews/phase-06-participant.md#窗口偏移与第八批反馈独立审查)。
- 以 `python tools/verify.py --profile phase-06-participant` 重跑阶段验证，26 项检查和 98 项测试通过。回合末治疗专项只验证已复核样例和评估范围，实战采集待验证。见[验证报告](reports/phase-06-window-robustness-20261124.json)。

## Phase 6 回合末治疗线索复评 · 2026-11-24

- 对四个普通回放组执行状态和回合线索扫描，复用鸟翔和风扬整段默认候选独立评分。原候选输出保留，几何试验和筛选只存于本地实验目录。记录了龙飞的先回血、后药次变化片段。见 [复评](../outputs/phase-06/periodic-heal-20261124/回合末治疗复评.md)和[独立审查](reviews/phase-06-participant.md#回合末治疗证据独立审查)。
- 执行 `python tools/verify.py --profile phase-06-participant`，26 项检查和 98 项测试通过，默认预测与归档逐字节一致。见 [验证报告](reports/phase-06-periodic-heal-20261124.json)。

## Phase 6 龙飞回合末回血反馈 · 2026-11-24

- 确认短片内右方治疗为普陀山技能，保存为右侧用药反例并保留独立读数观察。用详细版原生帧纠正旧拼图的一帧偏移，再核对该片段冻结预测。见 [第九批反馈](../data/annotations/human-review-09.json)、[专项复评](../outputs/phase-06/dragon-heal-feedback-20261124/focused-regression.json)和[独立复核](reviews/phase-06-participant.md#龙飞普陀回血反馈独立复核)。
- `phase-06-participant` 的 26 项检查和 98 项测试通过。见 [验证报告](reports/phase-06-dragon-heal-feedback-20261124.json)。

## Phase 6 分批回血复评 · 2026-11-24

- 在四场普通回放留存的画面观察量中试验延迟绿字关联，保留原识别和账本。根据两条用户回复补标龙飞与风扬的自动治疗，重算风扬冻结预测。检查点校验同时使用动作参考和未关联动作的计数变化，后者不参与动作匹配。见 [复评](../outputs/phase-06/staggered-heal-20261124/分批回血复评.md)和[独立审查](reviews/phase-06-participant.md#分批回血证据独立审查)。
- `phase-06-participant` 的 26 项检查和 98 项测试通过。补标后风扬默认预测与归档逐字节一致。见 [验证报告](reports/phase-06-staggered-heal-20261124.json)。

## Phase 6 绿字分离复评 · 2026-11-24

- 保存背景扣除和亮色笔画两套试验，使用遮蔽原帧复跑并逐条复核增补轨迹。录入鸟翔第十一批人工反馈，用同一份原预测复评。见 [复评](../outputs/phase-06/green-separation-20261124/绿字分离复评.md)和[独立审查](reviews/phase-06-participant.md#绿字分离独立审查)。
- `phase-06-participant` 的 26 项检查和 98 项测试通过，默认风扬预测与归档逐字节一致。见 [验证报告](reports/phase-06-green-separation-20261124.json)。

## Phase 6 回合末施药对照 · 2026-11-24

- 固定普通回放的视觉候选后选取 A、B 两段，记录用户确认的药物复活、莲花和回生。先用原预测复评补标，再用同一预测入口运行整段笔画补检。见 [对照报告](../outputs/phase-06/medicine-active-review-20261124/施药与莲花对照.md)和[独立审查](reviews/phase-06-participant.md#回合末施药对照独立审查)。
- `phase-06-participant` 的 26 项检查和 98 项测试通过。同版本默认复跑与对照归档匹配。见 [验证报告](reports/phase-06-medicine-active-review-20261124.json)。

## Phase 6 用药误报与归属复评 · 2026-11-25

- 记录第十三批人工反馈，在旧评估上增加主人加时间的独立匹配。离线复跑青底数字筛查和起手前低纹理宠位重排，两场重新评分。见 [复评](../outputs/phase-06/medicine-error-20261124/用药误报与归属复评.md)和[独立审查](reviews/phase-06-participant.md#用药误报与归属独立审查)。
- `phase-06-participant` 的 26 项检查和 99 项测试通过。见 [验证报告](reports/phase-06-medicine-error.json)。

## Phase 6 用药筛选组合复评 · 2026-11-25

- 固定上轮整场预测，在两场普通回放的已存视觉线索上复评回合末治疗筛选。核对两条鸟翔误报、两场既有命中与召唤，以及清除误报后的最后可核点偏差。见 [组合复评](../outputs/phase-06/medicine-review-20261125/用药筛选组合复评.md)和[独立审查](reviews/phase-06-participant.md#用药筛选组合独立审查)。
- 执行 `python tools/verify.py --profile phase-06-participant`，26 项检查和 99 项测试通过。见 [验证报告](reports/phase-06-medicine-review-20261125.json)。

## Phase 6 双侧复活复评 · 2026-11-25

- 补记两条风扬人工动作证据，以现有分组复评冻结预测。针对青光重叠截断飘字，用我方可见气血加橙色复活效果补检目标。新增预测均为未知。验证了瞬时回升、两帧飘字、相邻目标去重与两场评分一致。见 [复评](../outputs/phase-06/medicine-feedback-20261125/双侧复活复评.md)及[独立审查](reviews/phase-06-participant.md#双侧复活独立审查)。
- `phase-06-participant` 的 26 项检查和 99 项测试通过。见 [验证报告](reports/phase-06-medicine-feedback-20261125.json)。
- 用含 `BM-M06` 的新参考独立复跑鸟翔冻结版与补检版。完整计数一致与双方匹配结果在两次复评间一致。见 [专项断言结果](../outputs/phase-06/medicine-feedback-20261125/hp-return/verification.json)及[补标](../data/annotations/human-review-15.json)。

## Phase 6 飘字续接与分轨复评 · 2026-11-25

- 第十六批用户复核补记两次群药，保留详细版同帧计数更新。用冻结观察复跑原轨与分轨，独立评分使用普通版动作证据。检查青光遮挡下的两帧续接、目标位置、轨迹歧义和既有事件一致性。见 [复评](../outputs/phase-06/medicine-clues-20261125/飘字续接与分轨复评.md)及[独立审查](reviews/phase-06-participant.md#飘字续接与分轨独立审查)。
- `phase-06-participant` 的 26 项检查和 99 项测试通过，默认风扬预测哈希与上轮一致。见 [验证报告](reports/phase-06-medicine-clues-20261125.json)。

## Phase 6 群药归因复评 · 2026-11-25

- 固定上一轮整场候选、宠位及状态线索，复跑共同施药者关联。补记两条人工施药归属，分别按普通动作时间和详细版检查点评分。受益宠物作为同一候选的观察列表，保留原 ID。见 [复评](../outputs/phase-06/group-medicine-20261125/群药归因复评.md)和[独立审查](reviews/phase-06-participant.md#群药归因独立审查)。
- `phase-06-participant` 的 26 项检查和 99 项测试通过，默认风扬输出哈希保持一致。见 [阶段报告](reports/phase-06-group-medicine-20261125.json)。

## Phase 6 群药后续复评 · 2026-11-25

- 用冻结普通遮蔽观察量复算首次橙光配对和受遮挡宠位数字片段。两个阶段的预测先落盘，再用第十八批普通动作标签和已记录官方快照评分。核验原候选 ID、施药归属和单次药量。见 [后续复评](../outputs/phase-06/group-followup-20261125/群药后续复评.md)。
- `phase-06-participant` 的 26 项检查和 99 项测试通过。见 [阶段验证](reports/phase-06-group-followup-20261125.json)。

## Phase 6 群药跨画面对照 · 2026-11-25

- 固定普通录像窗口和动作时间参考，分别重放保留右方状态条、遮住双方状态条及动作时间余量试验。对龙飞验证源复跑完整普通画面，另行核对双版证据。见 [跨画面对照](../outputs/phase-06/group-transfer-20261125/群药跨画面对照.md)。
- `phase-06-participant` 的 26 项检查和 99 项测试通过。见 [阶段验证](reports/phase-06-group-transfer-20261125.json)。
- 第十九批人工反馈落盘。风扬通过补接同次动作尾段找回群药，技能归属仅作为人工记录。龙飞分别映射旧人物和宠位的编号并复跑整场，左 3 宠位归属得到确认，目标归并仍待改进。见 [标注](../data/annotations/human-review-19.json)、[评分核验](../outputs/phase-06/group-transfer-20261125/feedback-19-verification.json)和 [独立审查](reviews/phase-06-participant.md#跨画面群药独立审查)。

## Phase 6 人物与宠物群药归并 · 2026-11-25

- 复核龙飞两条独立受益轨，使用原生帧重算提前动作并复验四种采样起点。四场普通录像共 16,802 个采样帧，离线候选只变化一条。独立对照另核 19 个已复核动作窗口。见 [归并复评](../outputs/phase-06/group-receivers-20261125/人物与宠物群药归并复评.md)。
- `phase-06-participant` 的 26 项检查和 99 项测试通过。见 [阶段验证](reports/phase-06-group-receivers-20261125.json)。

## Phase 6 异宠受益与单人物复评 · 2026-11-25

- 在普通录像原帧上核对人物和另一只受药宠物，加入同槽两轨首帧一致检查。将第 20 批用户确认的龙飞左 2 水灵兽作用和站位验证落盘，整场复跑找回一条离线候选。见 [异宠受益复评](../outputs/phase-06/group-nonself-20261125/非自身宠物受益与群药复评.md)和[复核记录](../data/annotations/human-review-20.json)。
- `phase-06-participant` 的 26 项检查和 99 项测试通过。见 [阶段验证](reports/phase-06-group-nonself-20261125.json)。

## Phase 6 施药动作与免扣复评

- 第 21 批反馈确认白兔「寸草心」可免扣本次施药。详细版原帧证实本场暂记药次在切回合时恢复。视觉候选、确认账本及计次评分区分提示、确认免扣和扣次未知。见 [本轮复评](../outputs/phase-06/medicine-skill-separation-20261125/施药动作与免扣药次复评.md)和[阶段验证](reports/phase-06-medicine-waiver-verified-20261125.json)。
- `phase-06-participant` 的 26 项检查和 109 项测试通过。风扬复跑 6004 帧，预测校验值与上一轮一致。

## Phase 6 详细版暂记药次复评 · 2026-11-26

- 新增已复核免扣动作的离线结算。用龙飞详细版原帧读数核对暂增和回退，保留两段瓶数缺读。普通扣药及合成边界复跑通过。见 [复评](../outputs/phase-06/medicine-counter-settlement-20261126/详细版暂记药次复评.md)和[阶段报告](reports/phase-06-medicine-counter-settlement-final-20261126.json)。
- `phase-06-participant` 的 26 项检查和 118 项测试通过。雷绝详细版复跑 630 帧，官方事件文件与原基准逐字节一致。

## Phase 6 稀疏读数与原帧补读 · 2026-11-26

- 龙飞已复核的免扣在 250 毫秒间隔的全部 250 个起点均能补回证据。原视频实跑执行 143 帧 OCR，逐帧基线为 869 帧。普通扣药跨回合保持 +1。见 [复评与对照](../outputs/phase-06/medicine-adaptive-sampling-20261126/稀疏读数与原帧补读复评.md)。
- 阶段验证 26 项通过，风扬遮敌方状态条的 6004 帧复跑预测与上一轮逐字节一致。阶段运行所用环境跳过的模型测试已用项目环境补跑，完整测试 123 项通过。见 [阶段验证](reports/phase-06-medicine-adaptive-sampling-20261126.json)和[完整测试](../outputs/phase-06/medicine-adaptive-sampling-20261126/workspace-tests.txt)。

## Phase 6 普通画面免扣提示复评 · 2026-11-26

- 从普通原帧读取「寸草心」，在双方状态条均遮蔽的四个起点复现一条已知施药关联。改用字幕起点配对以避免串到随后回血。第 22 批反馈确认白兔有概率触发。见 [阶段复评](../outputs/phase-06/medicine-visible-waiver-20261126/普通画面免扣提示复评.md)。
- 阶段验证 26 项和完整测试 126 项通过。风扬 6004 帧默认参战模拟预测文件与上一轮逐字节一致。见 [阶段报告](reports/phase-06-medicine-visible-waiver-20261126.json)。

## Phase 6 龙飞第 8 回合复活复评 · 2026-11-26

- 普通版左 1 人物的 825 与详细版同一效果已用原帧对齐。核到左 1 即时药次增加及左 3 切回合药次增加两项独立观测。遮表复跑定位受益者，施放者排名对取帧起点敏感，保留待复核。相关药次、识别和对齐测试 49 项通过。见 [复评与证据](../outputs/phase-06/medicine-revival-20261126/龙飞第8回合复活复评.md)。

## Phase 6 复活飘字与本体动作复评 · 2026-11-26

- 检查青光遮挡下已上升的绿字续接。十个窗口的八十组对照只增加四个风扬 590 受益候选。独立光流试验暂未改善施放者总体排序。见 [复评](../outputs/phase-06/revival-body-20261126/复活飘字与施药本体复评.md)。
- 第 23 批反馈补齐已有参考 WC-M46。普通版右 5 短促起手可核，原检测与续接试验均漏掉 801 的受益轨迹。阶段检查 26 项、项目测试 129 项通过，补标后使用冻结预测重新评分。见 [阶段报告](reports/phase-06-revival-body-final-20261126.json)和[补标复评](../outputs/phase-06/revival-body-20261126/final-check/metrics-with-r15-review.json)。

## Phase 6 自动定位绿字补读 · 2026-11-26

- 试验自动选区，无需指定右 2。801 在八次遮蔽复跑匹配六次，590 保留八次受益候选。弱起点无法跨空帧连接到下一波。见 [自动补读复评](../outputs/phase-06/medicine-auto-glyph-20261126/自动定位绿字补读复评.md)。
- 风扬整场找回 801，并发现一条已核 207 的施药归属错误。维持可选试验。阶段检查 26 项、项目测试 135 项通过。见 [整场差异](../outputs/phase-06/medicine-auto-glyph-20261126/wind-full-overlap-comparison.json)和[阶段验证](reports/phase-06-medicine-auto-glyph-verified-20261126.json)。
- 207 的独立局部对照用此前宠位纹理观察复核动作，两个错误均改归左 4。第 24 批反馈标注雷绝 659 为右 4 宠物攻击后的吸血，遮蔽状态条的八次复跑都未判为用药。见 [用户标注](../data/annotations/human-review-24.json)和[反例复跑](../outputs/phase-06/medicine-auto-glyph-20261126/controls/thunder-659-lifesteal-regression.json)。

## Phase 6 宠位本体与施药归属 · 2026-11-26

- 增加可选的起手前宠位纹理复核。技能和已知入场保护了原有证据，组内推迟到所需的最晚判定帧。全场配对仅纠正风扬 207 的归属。四场 88 次局部复跑包含鬼魂宠、孩子和遮挡对照。阶段 26 项检查、项目 137 项测试通过。见 [复评](../outputs/phase-06/medicine-pet-presence-20261126/宠位本体与施药归属复评.md)和[阶段报告](reports/phase-06-medicine-pet-presence-20261126.json)。

## Phase 6 左方阵法站位与施药归属 · 2026-11-26

- 四场左方人物和宠位的真实号位从原帧校准。主预测的输出保留画面键并附阵法号。风扬整场默认和冻结的补读预测分别用新标注评分，阶段检查 26 项通过。独立复查覆盖施药、召唤、标注导入及未校准宠位。见 [复评](../outputs/phase-06/formation-identity-left-20261126/左方站位与归属复评.md)和[阶段报告](reports/phase-06-formation-identity-left-20261126.json)。

## Phase 6 龙飞剩余宠位及复活补取 · 2026-11-26

- 确认龙飞剩余三个宠物主人。配对原帧覆盖全部五位人物的宠次，三笔取样的前后只有对应主人变化。第 26 批反馈补录两项用药，单独复核风扬迟至切回合的药瓶刷新。局部诊断采用普通录像的不同取帧起点，找回两笔已知复活的六次漏归。代码测试 142 项、阶段检查 26 项通过。见 [复评](../outputs/phase-06/dragon-revival-followup-20261126/施药动作和药次刷新复评.md)和[阶段报告](reports/phase-06-dragon-revival-followup-20261126.json)。

## Phase 6 弱金环出宠 · 2026-11-27

- 增加左侧光柱和原位模型稳定入场的可选路径。三场完整模拟参战回放新增五条已复核出宠，52 组局部前后对照仅补回龙飞左 5 漏检。修正长模板与新候选重叠时的重复输出。阶段验证 26 项，最后全套测试 144 项通过。见 [复评](../outputs/phase-06/summon-weak-ring-20261126/弱金环出宠复评.md)及[阶段报告](reports/phase-06-vacant-beam-20261126.json)。

## Phase 6 入场后回位 · 2026-11-27

- 增加连续空位前后按名字字形关联召唤模型的可选路径，避免第二轮光柱触发重复关联。64 次短窗对照只补回风扬 `WC-S01` 的四次漏检。阶段检查 26 项、最终项目测试 146 项通过。见 [回位复评](../outputs/phase-06/summon-return-20261127/入场后回位复评.md)及[阶段报告](reports/phase-06-beam-return-20261127.json)。

## Phase 6 孩子入场 · 2026-11-27

- 保存双区域外观和名字字形的实验结果，使用四场已核普通召唤作为反例。通过来源、人物、宠位与入场时间应用人工反馈并独立计次。加入风扬孩子和鸟翔同一旧宠的已核误报。阶段检查 26 项、项目测试 147 项通过。见 [复评](../outputs/phase-06/child-entry-20261127/孩子入场与宠次复评.md)及[阶段报告](reports/phase-06-child-entry-20261127.json)。

## Phase 6 原宠留场 · 2026-11-27

- 增加默认关闭的 `summon_guard_retained`。同一已核实宠位的连续名字和本体外观用于复核弱出宠候选。保留已观察的原候选信号，并记录判定所用原帧。技能冲突和独立入场证据沿用原路径。
- 六段原宠的四起点对照少报 11 次，三段真实出宠的 12 次已检出候选均保留。鸟翔、风扬整段继续命中各 4、11 笔真实出宠。阶段检查 26 项、最终项目测试 153 项通过。见 [复评](../outputs/phase-06/retained-pet-20261127/原宠留场复评.md)及[阶段报告](reports/phase-06-retained-pet-20261127.json)。
