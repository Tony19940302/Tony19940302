# Phase 6 · 整场评估和回归

## 目标

在独立比赛测量准确率、累计偏差和识别延迟。

## 范围

按动作、阵法、视角和遮挡等维度评估，保存典型回归样本。

## 非目标

以单事件平均指标替代整场资源一致性。

## 依赖

传统或 AI 识别基线，锁定的独立比赛，复核参考数据。

## 任务

一对一匹配；输出人物用药、宠物用药、召唤分类指标；统计归属、未知率、回合和全场资源偏差；保存误报、漏报截图。

## 验收标准

独立测试完整可重做，记录目标达成和未验证维度。

## 验证命令

整场对照可运行 `python tools/verify.py --profile phase-06-final`。该入口执行测试、核验素材、复跑对齐，从普通录像一次提取旧版和新版候选，再用已复核的详细版参考评分并导出错误画面。

重叠区间可运行 `python tools/verify.py --profile phase-06-overlap`。该入口复跑对齐、详细计数提取、普通画面基线、人工参考评估及回归截图导出。

参战可见性的离线模拟可运行 `python tools/verify.py --profile phase-06-participant`。该入口先屏蔽普通回放左方人物及宠位的状态条区域，再基于相同参考验证，报告标明模拟输入来源。

## 已知风险

精度均值掩盖累计误差；测试污染；样本稀有动作计量不足。

## 完成证据

锁定数据版本、正式评估和回归报告。 当前状态见 [阶段进度](../../docs/PROJECT_PROGRESS.md)。

## 人工反馈

先对照详细录像的计数变化和普通录像中的动作，准备短片和来源时间。画面能确认的事件直接记录，施药者、受药者或技能存在歧义时请项目负责人按片段编号回答。首批选用已有训练比赛中的两个用药窗口，见 [复核清单](../../data/annotations/human-review-01.json)。

收到反馈后在复核清单保留答复、来源及时间，将能确认的行动者映射到已有的人工标注，升级标注版本并用 Phase 3 验证重建片段索引。看不清的字段保留未知，仍可记录已确认的资源归属。后续调整或撤回保留原答复并追加修订。用户主动指出漏报时记录录像、回合、时间、动作和可见证据，先与已有候选或官方计数变化核对，归并同一事件。待核实的动作暂以候选保存，确认后才纳入训练数据。

Phase 5 验证组继续用于比较。Phase 6 完整比赛评估需要另有一场冻结的独立比赛和覆盖全场的参考记录。训练反馈使用训练比赛。

第二批复核选取雷绝阵对虎翼阵训练录像中的三次已核实药次变化。用户确认 M1 为雷绝阵 4 号宠物施药，M2、M3 均为雷绝阵 5 号宠物施药，受药者继续待核。v5 训练集已有 9 条用药和 4 条召唤的已复核标注。现行时序检测命中其中 8 条用药，包含 M2、M3。M1 动作评分首位为 4 号宠物，受药特效强度未达到门槛。见 [人工反馈](../../data/annotations/human-review-02.json)、[v5 标注](../../data/annotations/thunder-tiger-review-v5.json)及[检测对照](../../outputs/phase-06/review-02/v5-medicine-comparison.json)。

相同配置的线性片段分类器分别用 v4、v5 训练，独立龙飞阵对虎翼阵 6 个片段的分类正确数从 2 降到 1，两版的 ONNX Runtime 一致性均通过。该实验继续保留为离线对照。见 [v4 实验](../../outputs/phase-06/review-02/linear-v4/experiment.json)及[v5 实验](../../outputs/phase-06/review-02/linear-v5/experiment.json)。

受药特效偏弱时，可选规则要求宠物施法动作明确、受药方在多个连续帧从空血条恢复。用普通训练录像完整复跑，命中 8/9 升至 9/9，复核范围内误报均为 0。唯一变化是补回 M1 的雷绝阵 4 号宠物施药。龙飞对虎翼验证录像的既有候选结果相同，仍匹配唯一已复核的用药。该规则以 `medicine_allow_weak_amber` 单独开启，等待更多独立录像验证。见 [原始录像复跑](../../outputs/phase-06/review-02/medicine-opt-in-integrated-comparison.json)和[验证对照](../../outputs/phase-06/review-02/medicine-weak-amber-trial/production-dragon-tiger.json)。

## 鸟翔阵独立测试的重叠区间

保留旧训练场的视觉模板和阈值，在新场仅校准阵法位置。用详细版计数和普通版动作做独立复核。8 个对齐锚点覆盖普通版约 2.414–314.284 秒，7 个检查点最大残差为 188 毫秒。详版约 315.4 秒的第 13 回合末计数可用，之后录像进入第 14 回合。两份录像都未录到比赛结算。

- 15 条独立事件参考中，5 次用药均有时间匹配，4 次记对主人。2 次受药者可以复核且均匹配，5 次施药者均缺乏足够证据。首发后、截至第 13 回合末的 10 次召唤全部漏报。普通版共输出 64 条用药候选、33 条未知动作，召唤光圈模板在这场没有发出候选。
- 首批评分窗为普通版 3.5–19.5 秒，详细版两端十人计数一致，未发现误报。补充诊断根据后来核实的计数边界，扩大了开场无计次事件的区间至 65.7 秒。59 条未匹配用药候选中，4 条由此确认为误报，余下 55 条待核。整体精确率与人物、宠物用药各自的召回率尚未验证。见 [开场误报证据](../../outputs/phase-06/bird-opening-diagnostic.json)。
- 从详细版 10 秒的人工快照起算，13 个回合末共有 253 项可比计数，其中 121 项一致，占 47.8%。第 13 回合末 20 项可比计数中 4 项一致，累计绝对误差 50 次，单项最大偏差 14 次。遮挡字段与终局计数保持未知。
- 详细版另对数字区域重新裁切。在本场 20 张人工对照帧上，正确读取的可辨药次由 61/188 升至 152/188，宠次由 29/191 升至 180/191。按次数增量汇总，十个阵位的药次和宠次均与人工起止计数相符。L1 在第 8、9 回合各用药一次，自动提取合并成一次跨回合增加两次的候选。该结果用于辅助复核这场录像的计数。见 [数字区域配置](../../configs/counters/bird-mirror-numeric.json)和[详细版提取报告](../../outputs/phase-06/bird-counter-numeric/counter_extraction_report.json)。

见 [人工事件标注](../../data/annotations/bird-mirror-reviewed-events.json)、[人工计数与回合边界](../../data/annotations/bird-mirror-counters.json)和[重叠评估](../../outputs/phase-06/bird-evaluation.json)。首版标注已完成 23 项端到端验证，扩充标注后重跑评估与测试。见 [复跑报告](../reports/phase-06-overlap.json)及[独立审查](../reviews/phase-06-overlap.md)。

## 风扬阵对云垂阵的整场独立对照

两份录像完整解码，并按整场配对冻结为 `test`。普通版人工校准十人站位和十个固定宠位，模板来自雷绝阵对虎翼阵训练场。18 个锚点覆盖开场到战斗退出前，24 个检查点的最大残差为 102 毫秒。详细版提供 18 个回合末及终局账本，共确认 55 次药次增加、22 次首发后的召唤。第 18 回合最后一击后左方五名人物血条清空，双方回放均离开战斗。见 [时间对齐](../../data/annotations/wind-cloud-alignment.json)、[回合计数](../../data/annotations/wind-cloud-counters.json)和[逐事件参考](../../data/annotations/wind-cloud-reviewed-events.json)。

用药检测结合上浮回复数字、受药者血条或特效及先行施法，宠物动作按固定宠位查主人。召唤检测增加金环、光柱、人物起手及固定宠位新血条的时序佐证。旧版仍可按 `--detector baseline` 复跑。参考事件按类型和时间做一对一匹配，同步或重叠计数的施药归属保留歧义。

用户确认，普通出宠的提示全文为「召唤」，「死亡召唤」归入宠物技能。后续文字识别按完整提示验收，文字不清的帧保留已有入场候选及画面证据。见 [G06](../../docs/GAME_RULES.md)。

逐帧复核雷绝对虎翼四次、龙飞对虎翼两次已确认召唤。录像里能定位金环、新宠和入场后的效果文字，这六个窗口均缺可确认的独立「召唤」字样。肉眼可读的「死亡召唤」作为对照，Windows OCR 未完整转录该词。文字识别需要可辨的出宠正样本。见 [训练录像核验](../../outputs/phase-06/summon-text/train/training-text-audit.json)及[验证录像核验](../../outputs/phase-06/summon-text/validation/validation-review.json)。

| 项目 | 旧版 | 新版 |
|---|---:|---:|
| 已复核用药命中 / 总数 | 39/55 | 28/55 |
| 用药误报 | 162 | 6 |
| 已复核召唤命中 / 总数 | 1/22 | 3/22 |
| 召唤误报 | 12 | 12 |
| 归属可逐笔核对的命中中，主人正确 | 25/35 | 30/30 |
| 回合末已用次数一致 | 94/360 | 190/360 |
| 终局已用次数一致 | 0/20 | 4/20 |
| 终局累计绝对误差 | 175 | 48 |

新版用药精确率 82.4%、召回率 50.9%，召唤精确率 20.0%、召回率 13.6%，回合末已用次数一致率 52.8%。均低于原计划的初始目标。29 条可比较的候选确认时点相对官方计数变化，中位为录像内 693 毫秒，区间上界最大 1,166 毫秒。设备真实采集延迟、实战视角和人物、宠物用药分类的整场表现仍待验证。见 [旧版结果](../../outputs/phase-06/wind-cloud/baseline-metrics.json)、[新版结果](../../outputs/phase-06/wind-cloud/temporal-metrics.json)和[复跑报告](../reports/phase-06-final.json)。

## 参战可见性模拟

在同一场风扬阵对云垂阵录像上，特征提取前遮住左方人物和宠物的状态条区域。敌方用药依靠可见的施药动作、提示和特效，出宠依靠人物起手、金环和固定宠位的新模型。我方可见状态条仍可用于识别。事后使用同一份详细回放评分。分队事件在队内按类型和时间一对一匹配，沿用 500 毫秒容差。整场原报告各有一次跨队的用药时间匹配，分队核对将其记为左方误报和右方漏报。

| 项目 | 普通回放 | 参战模拟 |
|---|---:|---:|
| 左方用药命中 / 已核实次数 | 20/30 | 19/30 |
| 左方用药误报 | 2 | 1 |
| 左方出宠命中 / 已核实次数 | 1/15 | 3/15 |
| 左方出宠误报 | 11 | 9 |
| 左方回合末计数一致 | 82/180 | 81/180 |
| 左方终局计数一致 | 2/10 | 2/10 |
| 左方终局累计绝对误差 | 29 | 21 |
| 右方用药命中 / 已核实次数 | 7/25 | 7/25 |
| 右方出宠命中 / 已核实次数 | 2/7 | 2/7 |
| 右方回合末计数一致 | 108/180 | 108/180 |
| 右方终局计数一致 | 2/10 | 2/10 |

模拟中，左方命中的 19 次用药和 3 次出宠均记到正确主人。左方唯一确认由宠物施药的事件漏检，现有结果不足以量化敌方宠物施药的识别准确率。整场回合末计数一致为 189/360，终局为 4/20，总绝对误差为 40 次。见 [分队与逐人结果](../../outputs/phase-06/wind-cloud/participant-simulation/comparison.json)、[整场原报告](../../outputs/phase-06/wind-cloud/participant-simulation/metrics.json)及[独立审查](../reviews/phase-06-participant.md)。

上述版本有三个同主人出宠候选的金环末帧超过评分容差。真实参战录像仍需用于验证可见区域、动作检出和端到端延迟。

## 参战模拟动作时序复验

先在雷绝阵对虎翼阵训练录像验证记时。雷绝 2 号出宠的人眼首见金环为 90.043 秒。采用首个获后续模型变化佐证的金环采样帧后，动作记时由 90.410 秒提前到 90.177 秒，候选仍于 91.210 秒形成。召唤去重和用药冲突继续以金环末帧关联同一段动画。见 [训练对照](../../outputs/phase-06/participant-summon/training-timing-comparison.json)。

用同一遮罩、时间对齐和 500 毫秒容差复跑风扬阵对云垂阵。181 条候选中有 8 条敌方出宠的事件时点前移，候选数量、归属和可用时点逐条一致。敌方出宠时间命中由 3/15 升至 6/15，误报由 9 降到 6。新增匹配三次的主人和宠位均正确。全场出宠时间命中由 5/22 升至 8/22。回合末计数一致仍为 189/360，终局仍为 4/20，终局累计绝对误差仍为 40 次。敌方用药命中 19/30，我方用药 7/25、出宠 2/7。见 [改动前后对照](../../outputs/phase-06/participant-action-cues/wind-cloud-comparison.json)、[复跑评估](../../outputs/phase-06/participant-action-cues/wind-cloud-after/metrics.json)及[验证报告](../reports/phase-06-participant-action-cues.json)。

遮蔽训练场的用药识别命中 7/9。M1 的受药方金色特效偏弱，另一例的受药方局部动作偏弱。直接放宽两处门槛会在 111.601 秒错记一个敌方宠物用药。详细回放中该主人计次一直为 0/10。加入施药宠物附近的金色脉冲及先后关系后，同场实验命中 9/9，未新增这条错归。门槛取自同一训练场，五个已有干扰片段均缺相似的弱特效恢复轨迹。试验规则未进入默认检测。见 [用药实验](../../outputs/phase-06/participant-medicine-dev/training-visible-medicine-experiment.json)和[错归计次复核](../../outputs/phase-06/participant-medicine-dev/training-111601-reference-review.json)。

龙飞阵对虎翼阵另复跑六个已标注验证片段。我方用药命中 1/1，出宠命中 1/2，三个干扰片段无资源误报。该场的标注仅能验证我方正例。见 [跨场回归](../../outputs/phase-06/participant-action-cues/dragon-tiger-window-regression.json)。

## 固定宠位留空复核

用户确认已讨论的回放出宠画面缺独立「召唤」文字，提出跟踪旧宠离开后的宠位。使用参战模拟遮住左方状态条，沿固定宠位抽帧。雷绝对虎翼四次、龙飞对虎翼两次及鸟翔对鸟翔十次已确认出宠，均看到至少一帧宠位留空后出现新宠。该结果来自按已知事件抽样的人工复核。见 [雷绝和龙飞抽帧审查](../../outputs/phase-06/pet-vacancy/visual-review.json)及[鸟翔阵连帧](../../outputs/phase-06/pet-vacancy-bird/positive-2.jpg)。

雷绝阵 2 号的旧宠在 79.204 秒受击后淡出，79.404 秒起采到多帧留空，90.043 秒出现金环，90.760 秒及其后可见新宠。龙飞对虎翼 `DT-S01` 在 88.800 秒仍有旧宠，89.000 秒和 89.183 秒位置留空，89.350 秒出现金环。该例旧宠离场原因仍待核。见 [旧宠受击离场](../../outputs/phase-06/pet-vacancy/frames/T4_L2_departure_detail.jpg)、[同一位置入场](../../outputs/phase-06/pet-slot-vacancy/enemy-arrival.jpg)和[短暂留空](../../outputs/phase-06/pet-vacancy/frames/DT-S01.jpg)。

同场「高级神佑复生」画面中原宠恢复后曾暂时离开裁图区，约 100.714 秒可见原外观回到宠位。鸟翔阵 `BM-S03` 的两只宠物外观相似，详细版宠次仍由 2/6 变为 3/6。留空时长和外观差异应作为辅助证据。将宠位观测记录为有宠、留空或遮挡未知，结合人物起手、入场特效和宠物持续在位确认同一动作。空位可触发候选，确认为成功入场后才按主人加一次。离场和复活按各自事件处理。自动判别和整场误报待验证。见 [复生对照](../../outputs/phase-06/pet-vacancy/frames/N04_revive.jpg)及[鸟翔正式参考](../../data/annotations/bird-mirror-reviewed-events.json)。

## 出宠多线索对照

鸟翔阵普通录像遮蔽敌方状态条后，现有检测从开场跑到录像结束。宠位实验用雷绝对虎翼四个已标注片段拟合局部亮度、彩度和轮廓评分，固定参数后检查鸟翔阵十个已标注片段的对应宠位。该实验遮蔽双方状态条，单独根据宠体区域由空到有生成候选。两种方法均按同一份官方计次参考和 500 毫秒容差核对主人与时间。 [整段现有检测](../../outputs/phase-06/pet-vacancy-multicue/bird-current/metrics.json) · [宠位实验](../../outputs/phase-06/pet-vacancy-multicue/model/occupancy-probe.json)

| 方法 | 十次已复核出宠的匹配 | 开场确认无出宠区间的错报候选 |
|---|---:|---:|
| 现有检测 | 2/10 | 0 |
| 宠位留空后重新出现 | 6/10 | 7 |
| 两边都匹配到的参考事件 | 1/10 | — |
| 任一边匹配到的参考事件 | 7/10 | — |

两边覆盖的是不同事件。开场 3.5–19.5 秒两者均核对全体宠位，宠位实验的正例覆盖仅在指定片段和号位上验证。交集和并集表示已核实事件的覆盖，实际候选去重、融合后的误报和整场精度仍需单独评估。原宠复生返位也触发了宠位候选。可以分别收集候选，从动作、入场光效、持续占位及复活信息核实，再将同主人同一次入场计一次。见 [逐例对照](../../outputs/phase-06/pet-vacancy-multicue/bird-cue-comparison.json)。

风扬对云垂另抽查现有检测的误报。右方 R4 在 434.4 秒附近出现可见入场金环和名为「小星星」的模型，详细版在入场前、中、后对应人物的宠次均为 1/6。局部颜色实验将这段始终判成在位，漏过了人眼可见的空档。画面中的入场与计次差异原因仍待核。 [普通回放](../../outputs/phase-06/pet-vacancy-multicue/report/wind-cloud-fp-170-context.jpg) · [详细回放](../../outputs/phase-06/pet-vacancy-multicue/report/wind-cloud-fp-170-detailed.jpg) · [局部颜色实验](../../outputs/phase-06/pet-vacancy-multicue/model/wind-cloud-r4-170.json)

异步检索找到 2026 年下半年四条《梦幻西游手游》公开录像入口，以及游戏内赛事录像详情的公告转载。网页核实了录像标题日期，当前缺可复核动作帧。已记录拟筛选动作和同场计次标注方法。见 [素材线索](../../docs/RESEARCH_NOTES.md#公开录像线索)。

## 计次反例与跨场训练

- 雷绝对虎翼详细回放支持两条全场计次不变的训练反例。普通版 99.800–101.300 秒可见「高级神佑复生」后的原宠返位和绿色恢复数字，R1 宠次维持 2/6。普通版 32.550–34.050 秒「死亡召唤」动作经过另一处宠位，双方药次与宠次均未变化。训练开场约 10.866–37.203 秒可核实十人的已用宠次均为 1，包含一次上限增加。见 [计次核验](../../outputs/phase-06/action-training/labels/training-negative-verification.json)。
- 出宠融合保留现有候选，另用宠位由空到有、人物活动和位置相符的弱金环补充候选。同一主人近邻重复候选合并。先在雷绝训练片段评分，36 组门槛均命中 4/4，均通过当时已覆盖的负例。在同场扩至全宠位后，融合规则另误报一次 R1 出宠，对应官方计次未变。仅用该训练场重评 36 组门槛，均保留四次真出宠，也均放过这次误报。见 [门槛与跨场片段](../../outputs/phase-06/action-training/summon/fusion-review.json)、[训练场整段](../../outputs/phase-06/action-training/summon/thunder-full-mine.json)及[参数重评](../../outputs/phase-06/action-training/summon/thunder-full-grid.json)。
- 冻结的融合规则复跑鸟翔整段 4,114 个采样帧，十个宠位全部参与。确认出宠命中由 2/10 升至 6/10。按各主人检查点的已用次数不变区间核对，可确认的错加由 1 次升至 11 次。回合末宠次一致由 79/127 升至 99/127，累计绝对误差由 49 升至 65。第 13 回合最后可核的宠次绝对误差由 7 升至 11，其中 L3 多记 6 次。连帧可见同一模型反复从固定位置消失又重现，另有相邻特效干扰。融合规则用于候选研究。见 [整段对比](../../outputs/phase-06/action-training/summon/bird-fusion/comparison.json)和[L3 连帧](../../outputs/phase-06/action-training/summon/bird-fusion/L3-checkpoint-overcount.jpg)。
- 用药补判门槛从原训练试验冻结，独立重提两场模拟参战画面。风扬整场仍为 27/55、误报 5 次，唯一有明确敌方宠物施药者标签的 `WC-M02` 仍然漏检，两个候选施药宠物动作分数差 0.044。鸟翔命中由 2/5 升至 3/5，回合末药次绝对误差由 98 降至 89。新增 `BM-M02` 的主人和受药者可核实，实际施药者仍未知。新增事件来自弱受药动作分支。施药金色闪光分支在两场均未补回事件。见 [风扬评估](../../outputs/phase-06/action-training/medicine/wind-cloud-frozen-crosscheck.json)和[鸟翔评估](../../outputs/phase-06/action-training/medicine/bird-mirror-frozen-crosscheck.json)。
- 比较固定宠位名称区域的颜色文字轮廓。雷绝新宠名称变化清楚，返位前后名称相近，但鸟翔 `BM-S03` 确有宠次增加，名称区域仍高度相似且有叠字。名称与外观只作辅助。另把原宠返位纳入遮挡训练，按同一配置重训线性片段模型。独立龙飞场六个片段中最高分分类由 2/6 降至 1/6，置信阈值以上正确数均为 1/6。见 [名称对照](../../outputs/phase-06/action-training/nameplate/nameplate-comparison.json)及[遮罩训练](../../outputs/phase-06/action-training/clip-model/masked-clip-comparison.json)。

## 第三批人工反馈

- 用户确认风扬对云垂案例 A 的「小星星」是孩子，每名人物可单独出战一次孩子。登记三张正面参考图，保留对局的测试分组。加入 `SUMMON_CHILD` 后，把已复核事件重放到云垂阵 4 号账本，孩子已用 `1/1`，宠物已用 `1/6`。该场手工修订这一条后，已核实宠次误报由 7 降到 6，终局一致由 4/20 到 5/20。自动检测基线单独复跑。见 [用户反馈](../../data/annotations/human-review-03.json)、[账本重放](../../outputs/phase-06/child-idle-feedback/manual-child-replay/ledger-check.json)及[人工修订评分](../../outputs/phase-06/child-idle-feedback/manual-child-replay/metrics.json)。
- 孩子背面截取头发、发髻和衣服可检索同次入场的稳定帧。孩子与短小人形宠物的可见高度相近，遮挡时已知孩子的模板分也可能低于外观相近的宠物。三张用户截图用作正面造型参考，独立的孩子识别尚待验证。见 [画面对照](../../outputs/phase-06/child-idle-feedback/child/child-visual-probe.json)。
- 用户确认案例 B 的风扬阵 2 号宠位变化是待机摆动。加入相对自身待机且持续先起手的补判后，能在反馈场补回 WC-M02。实验测得 4 号宠位持续动作比 2 号宠位变化早 234 毫秒。风扬全场用药时间匹配由 27/55 到 28/55，确认误报仍是 5 次。用规则跨场复核时，鸟翔另报出 L3 宠施药，该人物详细版药次在事件前后均为 `0/10`。试验门槛保留在离线对照。见 [用药实验](../../outputs/phase-06/child-idle-feedback/medicine/lead-assist-probe.json)及[跨场反例](../../outputs/phase-06/child-idle-feedback/medicine/bird-98723-detail-probe.json)。

## 第四批人工反馈

- 雷绝对虎翼案例 A 确认为右侧虎翼阵 4 号的原宠「超级神虎」受击触发「大圣佑」，随后出击打左队 2 号宠。详细回放宠次维持 `2/6`。同宠位连续两帧字形对照在原融合候选可用前读到「大圣佑」，离线排除该条误报，训练场四次真出宠均保留。鸟翔的旧候选未因此被排除，该场缺已确认的「大圣佑」样本，字形规则留在试验。见 [用户反馈](../../data/annotations/human-review-04.json)、[受击与原宠连帧](../../outputs/phase-06/feedback-shengyou-group/entry/pet-cue-preview.jpg)及[字形对照](../../outputs/phase-06/feedback-shengyou-group/entry/shengyou-glyph-probe.json)。
- 鸟翔案例 B 按用户反馈记录两次动作。3 号人物用「杨柳甘露」复活 4 号，出现 `2320`。约 0.8 秒后，3 号宠物使用群体加血药品，4 号出现 `314`。另有绿色数字靠近宠位，受药者尚未核定。3 号的详细版药次在这段及第 4 回合结束均是 `0/10`，下一次读到 `1/10` 时画面已出现另一组回复。动作标注和计次参考分开保存，该次计次保留为待核。见 [同场两版连帧](../../outputs/phase-06/feedback-shengyou-group/counter/B-revive-multiheal-and-later.jpg)及[计次连帧](../../outputs/phase-06/feedback-shengyou-group/counter/L1-L3-native-counter-strips.jpg)。
- 对相邻多受药候选保留所有目标和原轨迹时点。风扬整场参考数据中，五组被程序归并的多受药候选附近都缺药次变化，继续保留未知并查施放者证据。完整风扬参战模拟复跑确认默认用药与出宠评分及账本一致。见 [轨迹对照](../../outputs/phase-06/feedback-shengyou-group/medicine/group-heal-track-probe.json)及[验证报告](../reports/phase-06-feedback-shengyou-group.json)。
