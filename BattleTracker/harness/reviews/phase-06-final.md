# Phase 6 整场评估独立审查

风扬阵对云垂阵录像在第 18 回合结束，普通版和详细版归入同一独立测试场。以固定阵位布局和训练场模板复验，两份录像的对齐最大检查点误差为 102 毫秒。测试场阵位和宠物归属由人工标定，归属指标以已知阵位为前提。见 [阵位配置](../../configs/vision/wind-cloud.json)和[录像对齐](../../data/annotations/wind-cloud-alignment.json)。

## 问题与复验

1. **P1 · 重叠参考区间允许同一真值匹配多条预测。** 原匹配器递归改变分配后返回空值，调用者继续尝试并留下重复分配。实际旧版结果把同场的 `WC-M14` 与 `WC-M17` 各重复计为两次命中。修复后，两个同时间真值和三个候选只产生两条命中及一条误报。见 [一对一匹配](../../tools/evaluate.py)和[回归测试](../../tests/test_overlap_evaluation.py)。
2. **P2 · 完整覆盖判定原先未强制事件与各回合末计数相符。** 一条第一回合前已计数的参考事件放到第二回合，原实现仍接受全场覆盖。现按照每个回合末的详细版时间和计数核对参考区间，并要求回合末字段齐全。复验覆盖了事件错回合和遮挡字段两种情况。见 [整场评估](../../tools/evaluate_phase6.py)和[覆盖回归](../../tests/test_overlap_evaluation.py)。
3. **P2 · 同帧计次可能产生任意归属判定。** 详细版存在两人的药次在同一帧变动的情况，普通版多个回复相邻，配对仍用于动作类型统计。评估将标明同一计数时点组的配对从行动者、主人、目标和决策偏移的分母中排除。见 [事件标注](../../data/annotations/wind-cloud-reviewed-events.json)。
4. **P2 · 测试录像模板来源原先只校验角色。** 测试配置可以指向其他测试场模板。现要求测试场引用训练场模板，并阻止直接从测试源提取模板。旧召唤候选缺少可用时刻时也不再充当用药时序排除证据。见 [普通版预测](../../tools/predict_clean.py)和[用药检测](../../src/battletracker/medicine_vision.py)。

第 10 回合 L4 的宠物计数在详细版 249194 毫秒仍为 1/6，249211 毫秒变为 2/6。普通版在 249280 毫秒尚无该宠血条，249296 毫秒出现血条。逐帧证据补足了原先受飘字遮挡的召唤时点。见 [详细版逐帧图](../../outputs/phase-06/final-intake/independent-L4-r10-detail.png)和[普通版逐帧图](../../outputs/phase-06/final-intake/independent-L4-r10-clean.png)。

## 结果与边界

独立复跑 56 项测试通过。用两份冻结预测重算，终局计数和 77 条资源参考与 18 个回合末全部对上。动作检测和累计计数尚未达到工程目标。归属准确率仅使用普通版命中且具备可辨归属的配对。人物和宠物用药的完整分项指标仍缺行动者参考。决策偏移来自录像时间戳，实时采集与流式处理延迟待验证。见 [旧版独立复算](../../outputs/phase-06/wind-cloud/independent-baseline-metrics.json)、[新版独立复算](../../outputs/phase-06/wind-cloud/independent-temporal-metrics.json)和[整场复跑报告](../reports/phase-06-final.json)。
