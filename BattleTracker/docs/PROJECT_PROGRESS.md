# 项目进度

| 阶段 | 状态 | 验收依据 |
|---|---|---|
| Phase 0 | 已验收，素材已补齐 | Git 和 Python 3.13 环境可用，详细与普通回放已完整解码，九类阵法截图已归档 |
| Phase 1 | 已验收 | 22 项核心测试和 4 项视频集成测试通过，完成独立审查及复验 |
| Phase 2 | 单场解析已验收 | 十人计数区域均有稳定输出，7 帧人工对照，29 项测试通过。见 [验证报告](../harness/reports/phase-02.json) |
| Phase 3 | 本场成对回放与样本流程已验收 | 12 个拟合锚点，11 个检查点，16 条计数变化映射，5 个困难负样本。见 [验证报告](../harness/reports/phase-03.json) |
| Phase 4 | 单场离线基线已验收 | 1,714 帧离线预测，7 条人工参考一对一评估，36 项测试通过。见 [统一验证](../harness/reports/phase-04.json) |
| Phase 5 | 跨场实验验收通过 | 两场隔离，12 个训练片段和 6 个验证片段。三种模型完成整段录像对照及 ONNX Runtime 一致性验证。未观察到识别收益，保留传统基线。见 [报告](../harness/reports/phase-05.json) |
| Phase 6 | 已完成整场独立评估和参战可见性模拟，识别未达标 | 风扬参战模拟基线敌方用药命中 19/30，出宠时间匹配 6/15，终局一致 4/20。新增孩子账本、多受药证据及人工复核，真实参战录像和采集延迟待验证。见 [阶段记录](../harness/plans/phase-06.md#第四批人工反馈) |
| Phase 7–8 | 已拟定执行计划，待实施 | [计划目录](../harness/plans) |

见 [统一验证报告](../harness/reports/phase-01.json)、[独立复验](../harness/reports/phase-01-independent.json)、[独立审查](../harness/reviews/phase-01.md)和[构建日志](../harness/build-log.md)。



Phase 0–3 审查发现的固定宠位误计、官方计数来源校验及计数回归缺口已修复。见 [审查与复跑证据](../harness/reviews/phase-00-03-audit.md)。




Phase 4 接续点见 [阶段记录](../harness/plans/phase-04.md)。本场传统检测和评估可复跑。当前主要问题为连续动作中的用药误报、施药归属和相邻受药槽位混淆。详细记录见 [评估](../outputs/phase-04/evaluation.json)和[失败片段索引](../outputs/phase-04/review_cases.jsonl)。Phase 5 已获确认并开始。



2026-11-19 按原始方案第 11–15 节复核 Phase 1–5。修复了空宠位施药误计、评估匹配使用真实主人信息和空预测无法评估的问题。Phase 2–4 结论保持原计划的单场证据范围。Phase 5 的跨场检测跑完约 398 秒录像，复核范围含 3 条资源事件，三个增强方案均漏掉这 3 条。见 [实验记录](../harness/plans/phase-05.md)和[独立审查](../harness/reviews/phase1-5-audit-20261118.md)。
