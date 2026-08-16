from pathlib import Path


def test_product_http_does_not_import_langgraph_live_path():
    root = Path(__file__).resolve().parents[2]
    forbidden = (
        "get_orchestrator_graph",
        "create_deep_agent",
        "agents.orchestrator",
        "VISIBLE_TASK_NODES",
    )
    paths = [
        root / "app" / "backend" / "app" / "api" / "agent.py",
        root / "app" / "backend" / "app" / "main.py",
        root / "harness" / "agent" / "loop.py",
        root / "harness" / "runtime.py",
        root / "harness" / "runner.py",
    ]
    for path in paths:
        text = path.read_text(encoding="utf-8")
        for token in forbidden:
            assert token not in text, f"{path} still references {token}"
