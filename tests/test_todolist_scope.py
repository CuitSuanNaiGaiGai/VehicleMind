from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_todolist_matches_demo_first_scope() -> None:
    text = (ROOT / "todolist.md").read_text(encoding="utf-8")
    required = (
        "先完成可离线回放的端到端 Demo",
        "舱内最多 50 条",
        "舱外最多 50 条",
        "Agent 自动化评测任务不少于 120 条",
        "未经确认的敏感动作执行次数为 0",
    )
    forbidden = (
        "实现 GRU 时序分类模型",
        "实现 TCN 时序分类模型",
        "完成去除 PERCLOS 的消融实验",
        "完成去除头姿的消融实验",
        "将 CARLA 真值作为道路感知与跟踪模块的评测参考",
    )
    assert all(item in text for item in required)
    assert all(item not in text for item in forbidden)
