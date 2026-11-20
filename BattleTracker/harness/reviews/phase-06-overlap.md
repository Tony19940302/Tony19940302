# Phase 6 鸟翔阵重叠区间审查

独立复核了两份录像的来源、回合边界、首批人工事件和评估器。两份录像按同一 `battle_id` 归入独立测试组。普通版仅校准站位，沿用训练场的视觉模板和阈值。见 [独立评估结果](../../outputs/phase-06/bird-evaluation-audit.json)。

## 问题与修复

1. **P2 · 已核实空窗内的候选会占用相邻事件的匹配。** 在空窗内外各放一条同时落入容差的预测。先匹配后判空窗会让更近的空窗预测占用唯一的正例。修改后先排除空窗候选，再做一对一匹配。见 [事件评估](../../tools/evaluate_phase6.py)和[回归测试](../../tests/test_overlap_evaluation.py)。
2. **P2 · 手工计数回退可能漏检。** 检查点读数出现 `0 → 2 → 1` 时，末值比起点高，累计检查仍可通过。修改后将每个已知计数与前一次可读计数比较，即使中间有遮挡也能发现回退。见 [检查点评估](../../tools/evaluate_phase6.py)和[回归测试](../../tests/test_overlap_evaluation.py)。
3. **P2 · 计数参考文件缺少单独来源校验和字段顺序。** 其他对局的计数或顺序不同的十列可以产生无意义的归属误差。现从标注读取明确的槽位顺序，并验证详细录像资产、源文件哈希和槽位集合。事件时间窗也按对齐映射复算。见 [加载与校验](../../tools/evaluate_phase6.py)。

## 复验结果

- 独立复验两项回归测试通过。首版标签的端到端验证 23 项均通过，包含 44 项单元和集成测试。见 [验证报告](../reports/phase-06-overlap.json)。
- 扩充标签后，重新执行 `python tools/evaluate_phase6.py --export-evidence` 和 `python -m pytest -q`。44 项测试通过。普通版匹配 5 次已标注用药，记对 4 次主人。第 13 回合末以前的 10 次召唤全部漏报。回合末 121/253 项计数一致。见 [最新标注](../../data/annotations/bird-mirror-reviewed-events.json)和[复验结果](../../outputs/phase-06/bird-evaluation.json)。
- 补充核对确认开场有 4 条用药误报。剩余 55 条候选仍缺少事件证据。两段录像均未覆盖终局，完整比赛精确率、最终计数一致率和实时识别延迟待验证。见 [开场误报证据](../../outputs/phase-06/bird-opening-diagnostic.json)。
