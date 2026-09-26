from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_todolist_matches_recruiting_showcase_scope() -> None:
    text = (ROOT / "todolist.md").read_text(encoding="utf-8")
    required = (
        "当前已完成招聘展示闭环 A6",
        "舱内最多 50 条",
        "舱外最多 50 条",
        "40 条 AI 自审内部评测场景与 80 条开发回归变体",
        "不将 80 条变体计为独立金标",
        "未经确认的敏感动作执行次数为 0",
        "A6 任务状态已更新",
        "[x] 报告 Context Grounding Correctness",
        "[x] 报告 stale/UNKNOWN Handling",
        "[x] 同时报告失败次数、重复 trial 波动、token、延迟和成本",
    )
    forbidden = (
        "实现 GRU 时序分类模型",
        "实现 TCN 时序分类模型",
        "完成去除 PERCLOS 的消融实验",
        "完成去除头姿的消融实验",
        "将 CARLA 真值作为道路感知与跟踪模块的评测参考",
        "Agent 自动化评测任务不少于 120 条",
    )
    assert all(item in text for item in required)
    assert all(item not in text for item in forbidden)
