"""工具场景测试的终端摘要和机器可读报告。"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPORT_PATH = Path(".pytest_cache/tool_scenarios/latest.json")


def pytest_addoption(parser: Any) -> None:
    group = parser.getgroup("tool-model-behavior")
    group.addoption(
        "--run-model-behavior",
        action="store_true",
        default=False,
        help="运行会产生真实 DeepSeek 调用的模型行为测试",
    )
    group.addoption(
        "--model-repetitions",
        type=int,
        default=1,
        help="每条模型行为用例重复次数，默认1",
    )
    group.addoption(
        "--model-all",
        action="store_true",
        default=False,
        help="运行全部70条agent用例；默认运行20条校准集",
    )
    group.addoption(
        "--model-case-id",
        action="append",
        default=[],
        help="只运行指定case id，可重复传入",
    )


def _property_row(properties: dict[str, Any], *, status: str, duration: float, nodeid: str) -> dict[str, Any]:
    assertions = properties.get("tool_assertions", {})
    if status == "failed" or "failed" in assertions.values():
        evaluation_status = "failed"
    elif "not_evaluated" in assertions.values():
        evaluation_status = "not_evaluated"
    else:
        evaluation_status = "passed"
    return {
        "case_id": properties.get("tool_case_id", ""),
        "title": properties.get("tool_title", ""),
        "bucket": properties.get("tool_bucket", ""),
        "execution_mode": properties.get("tool_execution_mode", ""),
        "profile": properties.get("tool_profile", "structural"),
        "repetition": properties.get("tool_repetition"),
        "model": properties.get("tool_model", ""),
        "status": status,
        "evaluation_status": evaluation_status,
        "tools": properties.get("tool_names", []),
        "tool_call_count": properties.get("tool_call_count", 0),
        "answer": properties.get("tool_answer", ""),
        "finish_reason": properties.get("tool_finish_reason", ""),
        "assertions": assertions,
        "metrics": properties.get("tool_metrics", {}),
        "duration_ms": round(float(duration) * 1000, 3),
        "nodeid": nodeid,
    }


def pytest_configure(config: Any) -> None:
    config._tool_scenario_results = []


def pytest_runtest_makereport(item: Any, call: Any) -> None:
    if call.when != "call":
        return
    properties = dict(item.user_properties)
    case_id = properties.get("tool_case_id")
    if not case_id:
        return
    outcome = "failed" if call.excinfo is not None else "passed"
    item.config._tool_scenario_results.append(
        _property_row(properties, status=outcome, duration=call.duration, nodeid=item.nodeid)
    )


def pytest_terminal_summary(terminalreporter: Any, exitstatus: int, config: Any) -> None:
    rows = list(getattr(config, "_tool_scenario_results", []))
    if not rows:
        return
    structural_passed = sum(row["status"] == "passed" for row in rows)
    evaluated_rows = [row for row in rows if row["evaluation_status"] != "not_evaluated"]
    evaluated_passed = sum(row["evaluation_status"] == "passed" for row in evaluated_rows)
    failed = sum(row["evaluation_status"] == "failed" for row in rows)
    not_evaluated = sum(row["evaluation_status"] == "not_evaluated" for row in rows)
    metrics = [row["metrics"] for row in rows]
    logical_calls = sum(int(item.get("logical_calls", 0)) for item in metrics)
    actual_requests = sum(int(item.get("actual_requests", 0)) for item in metrics)
    first_attempt_successes = sum(int(item.get("first_attempt_successes", 0)) for item in metrics)
    recovered_calls = sum(int(item.get("bounded_recovery_successes", 0)) for item in metrics)
    retry_eligible_calls = sum(int(item.get("retry_eligible_calls", 0)) for item in metrics)
    duplicate_requests = sum(int(item.get("duplicate_requests", 0)) for item in metrics)
    unbudgeted_subcalls = sum(int(item.get("unbudgeted_subcalls", 0)) for item in metrics)
    profiles = {str(row.get("profile") or "structural") for row in rows}
    report_path = (
        Path(".pytest_cache/tool_scenarios/latest-model-behavior.json")
        if profiles == {"model_behavior"}
        else REPORT_PATH
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "summary": {
            "total": len(rows),
            "structural_passed": structural_passed,
            "structural_failed": len(rows) - structural_passed,
            "structural_pass_rate": structural_passed / len(rows),
            "fully_evaluated": len(evaluated_rows),
            "fully_passed": evaluated_passed,
            "failed": failed,
            "not_evaluated": not_evaluated,
            "evaluation_coverage": len(evaluated_rows) / len(rows),
            "scenario_pass_rate": (
                evaluated_passed / len(evaluated_rows) if evaluated_rows else None
            ),
            "tool_calls": sum(int(row["tool_call_count"]) for row in rows),
            "logical_calls": logical_calls,
            "actual_requests": actual_requests,
            "first_attempt_success_rate": (
                first_attempt_successes / logical_calls if logical_calls else None
            ),
            "bounded_recovery_rate": (
                recovered_calls / retry_eligible_calls if retry_eligible_calls else None
            ),
            "duplicate_request_rate": (
                duplicate_requests / actual_requests if actual_requests else None
            ),
            "unbudgeted_subcalls": unbudgeted_subcalls,
            "task_completion_rate": None,
        },
        "cases": rows,
    }
    payload["profiles"] = sorted(profiles)
    report_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    terminalreporter.section("tool scenario results")
    visible_rows = rows if len(rows) <= 20 else [
        row for row in rows if row["evaluation_status"] == "failed"
    ]
    for row in visible_rows:
        tools = " -> ".join(row["tools"]) or "none"
        terminalreporter.write_line(
            f"{row['case_id']} {row['evaluation_status'].upper()} | {row['title']} | "
            f"calls={row['tool_call_count']} | {tools}"
        )
        if row["answer"]:
            terminalreporter.write_line(f"  answer: {row['answer']}")
        observed = row["metrics"]
        terminalreporter.write_line(
            "  metrics: "
            f"logical={observed.get('logical_calls', 0)}, "
            f"requests={observed.get('actual_requests', 0)}, "
            f"retries={observed.get('retries', 0)}, "
            f"duplicates={observed.get('duplicate_requests', 0)}, "
            f"unbudgeted={observed.get('unbudgeted_subcalls', 0)}, "
            f"finish={row['finish_reason'] or 'unknown'}"
        )
        checks = ", ".join(
            f"{key}={value}" for key, value in row["assertions"].items()
        )
        if checks:
            terminalreporter.write_line(f"  assertions: {checks}")
    if len(visible_rows) < len(rows):
        terminalreporter.write_line(
            f"per-case detail: showing {len(visible_rows)} failed cases; "
            f"all {len(rows)} cases are in the JSON report"
        )
    terminalreporter.write_line(
        f"structural pass rate: {structural_passed}/{len(rows)} = "
        f"{structural_passed / len(rows):.2%}"
    )
    terminalreporter.write_line(
        f"evaluation coverage: {len(evaluated_rows)}/{len(rows)} = "
        f"{len(evaluated_rows) / len(rows):.2%}; not_evaluated={not_evaluated}"
    )
    terminalreporter.write_line(
        "scenario pass rate: "
        + (
            f"{evaluated_passed}/{len(evaluated_rows)} = {evaluated_passed / len(evaluated_rows):.2%}"
            if evaluated_rows
            else "N/A (no fully evaluated cases)"
        )
    )
    if logical_calls:
        terminalreporter.write_line(
            "first-attempt success rate: "
            f"{first_attempt_successes}/{logical_calls} = "
            f"{first_attempt_successes / logical_calls:.2%}"
        )
    terminalreporter.write_line(
        "bounded recovery rate: "
        + (
            f"{recovered_calls}/{retry_eligible_calls} = {recovered_calls / retry_eligible_calls:.2%}"
            if retry_eligible_calls
            else "N/A (no retry-eligible calls)"
        )
    )
    if actual_requests:
        terminalreporter.write_line(
            "duplicate request rate: "
            f"{duplicate_requests}/{actual_requests} = {duplicate_requests / actual_requests:.2%}"
        )
    terminalreporter.write_line(f"unbudgeted subcalls: {unbudgeted_subcalls}")
    terminalreporter.write_line(f"JSON report: {report_path.resolve()}")
