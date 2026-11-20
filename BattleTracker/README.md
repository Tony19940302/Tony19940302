# BattleTracker

用于《梦幻西游手游》战斗录像复盘。Phase 0/1 提供阵法及宠位配置、可重放的战斗资源账本、视频检查和验证入口。工作进展见 [阶段进度](docs/PROJECT_PROGRESS.md)。

## 本地运行

在项目根目录使用 Python 3.11 或更新版本。项目通过 PyAV 读取视频及源时间戳。

```powershell
python -m pip install -e ".[test]"
python tools/verify.py --profile phase-01
python tools/inspect_video.py --help
```

验证生成 `harness/reports/phase-01.json`。定位输入录像及其他素材时查阅 [资料清单](data/manifests)。业务规则、测试策略和后续阶段见 [游戏规则](docs/GAME_RULES.md)、[测试计划](docs/TEST_PLAN.md)和 [阶段计划](harness/plans)。

原始录像、截图和 `outputs/` 中的生成结果保存在本地。从 Git 克隆后，按资料清单放置对应素材，再运行依赖录像的阶段验证。历史报告记录生成结果在原工作区的位置。

## 检查录像

用实际的录像路径替换下方示例路径。该命令截取约 21.3 秒处的画面，并从 31 到 32 秒每隔 0.25 秒取样。标准输出给出视频元数据及取样结果。

```powershell
python tools/inspect_video.py "C:/path/to/battle.mp4" --at 21.3 --range 31 32 0.25 --output outputs/video-inspection
```

采样时刻以第一个有呈现时间戳的解码帧为起点。结果保留实际对应帧的源时间信息，可用于区分请求的采样时刻和画面实际时刻。

## 运行一个模拟事件

在项目根目录运行 Python，以下示例先加载可信开局，再录入 A2 的宠物为 A1 用药。输出为 `9`。

```python
from pathlib import Path

from battletracker.battle import BattleEvent, BattleStateEngine, EventType
from battletracker.formation import BattleLayout, load_formations

formations = load_formations(Path("configs/formations/formations.json"))
layout = BattleLayout.build(formations["hu_yi"], formations["long_fei"])
engine = BattleStateEngine("demo", layout)
engine.record_many([
    BattleEvent(
        "start", "demo", 1, 0, EventType.BATTLE_START,
        payload={
            "players": {"A2": {"medicine_used": 0, "medicine_limit": 10,
                               "pet_used": 1, "pet_limit": 6}},
            "initial_pets": {"A2": {"instance_id": "demo-pet"}},
        },
    ),
    BattleEvent(
        "dose", "demo", 1, 1200, EventType.USE_MEDICINE,
        actor_id="A_PET_2", target_id="A1",
    ),
])
path = Path("outputs/demo.events.jsonl")
engine.save(path)
replayed = BattleStateEngine.load(path, "demo", layout)
print(replayed.state.players["A2"].medicine_remaining)
```

事件日志保存 JSONL。重载时传入相同的比赛标识、布局和模式。初始用药或上限来自可靠快照时，填入 `players`。缺失字段会保留未知。撤销通过追加 `EVENT_RETRACTED` 指向原事件 ID。

## 站位校准格式

`BattleLayout.build` 的第三个参数接受校准字典，可以从外部 JSON 读取。根对象包括 `coordinate_system`（`normalized_screen`）、`calibration_version` 和 `teams`。为 `teams.A` 或 `teams.B` 填写 `formation_id` 及 `players`、`pets` 两个映射。映射的键为字符串 `"1"` 至 `"5"`，宠位使用主人号位作键，值为实际画面测得的归一化 `[x, y]`。缺省坐标及 `null` 表示未知。两侧分别填写。


## Phase 2 详细观战计数解析

```powershell
python -m pip install -e ".[test,vision]"
python tools/extract_counters.py
python tools/verify.py --profile phase-02
```

解析器按资产清单读取并校验已标定的详细回放。默认每隔 0.25 秒取样，输出原始读数、稳定快照、观察到的变化区间和可疑读数截图到 `outputs/phase-02`。校准范围与已知漏读见 [Phase 2](harness/plans/phase-02.md)。

## Phase 3 双录像对齐

```powershell
python tools/verify.py --profile phase-03
```

在 Phase 2 计数结果已生成时，可单独运行 `python tools/align_videos.py` 和 `python tools/build_dataset.py --export-clips`。默认结果在 `outputs/phase-03`。时间锚点和复核备注位于 `data/annotations`。片段索引保存普通录像原始时间区间，复核副本仅供快速回看。详见 [Phase 3](harness/plans/phase-03.md)。

Windows 下可用系统已安装的 OCR 语言对同一批人工参考 ROI 运行对照。先运行 Phase 2 或 Phase 3 验证，再运行 `python tools/compare_counter_ocr.py`。对照结果保存在 `outputs/phase-02/ocr-comparison/comparison.json`。

## Phase 4 普通回放视觉基线

```powershell
python tools/verify.py --profile phase-04
```

验证读取已生成的 Phase 3 片段索引。单独运行 `python tools/predict_clean.py` 只读取普通录像并输出视觉候选，`python tools/evaluate.py` 可将候选与人工复核标注匹配。输出位于 `outputs/phase-04`。
