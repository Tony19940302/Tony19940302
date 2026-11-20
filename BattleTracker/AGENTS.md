# BattleTracker 项目导航

BattleTracker 从《梦幻西游手游》战斗录像提取事件并重放资源账本。已验收范围及证据见 [PROJECT_PROGRESS](docs/PROJECT_PROGRESS.md)。继续推进时以用户确认的阶段为准。

- 开发前先读 [原始方案](PROJECT_SPEC.md)、[游戏规则](docs/GAME_RULES.md)、[架构](docs/ARCHITECTURE.md)及对应[阶段计划](harness/plans)。
- 人物身份、固定宠位和实际宠物实例分别建模。未知计数和上限保留原状态。用真实战斗画面校准坐标。
- 事件修订保留原记录，按有效事件重放。视觉候选经确认才计入账本。同场录像须在同一数据分组。
- 视频和原始用户数据在本地处理。当前工作限定于离线工具及业务核心。
- 按需使用项目级 [阶段开发](.agents/skills/phase-development/SKILL.md)和[代码审查](.agents/skills/code-review/SKILL.md)指南。
- 修改后运行对应阶段的验证命令，例如 `python tools/verify.py --profile phase-03`。以真实报告、独立审查和更新后的进度记录作为完成证据。SKIPPED 表示尚未验证。

