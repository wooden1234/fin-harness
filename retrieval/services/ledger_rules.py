"""账房规则：从 Markdown 文首元数据筛选，不走向量检索。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_RULES_DIR = _PROJECT_ROOT / "knowledge" / "账房"
_REQUIRED = (
    "rule_id",
    "name",
    "version",
    "enterprises",
    "effective_from",
    "status",
    "topic",
)
_ENTERPRISE_ALIASES = {
    "C001": "C001",
    "甲企业": "C001",
    "甲企业（比赛模拟）": "C001",
    "C002": "C002",
    "乙企业": "C002",
    "乙企业（比赛模拟）": "C002",
}


@dataclass(frozen=True, slots=True)
class LedgerRule:
    rule_id: str
    name: str
    version: str
    enterprises: tuple[str, ...]
    effective_from: str
    status: str
    topic: str
    body: str
    path: Path
    purpose_required: str | None = None
    purpose_threshold: str | None = None

    def public_dict(self) -> dict[str, object]:
        return {
            "rule_id": self.rule_id,
            "name": self.name,
            "version": self.version,
            "effective_from": self.effective_from,
            "topic": self.topic,
            "body": self.body,
        }


def rules_dir() -> Path:
    return _RULES_DIR


def normalize_enterprise(value: str | None) -> str | None:
    text = str(value or "").strip()
    if not text:
        return None
    upper = text.upper()
    if upper in {"C001", "C002"}:
        return upper
    return _ENTERPRISE_ALIASES.get(text)


def normalize_month(value: str | None) -> str | None:
    text = str(value or "").strip().replace("/", "-")
    parts = text.split("-")
    if len(parts) != 2 or not (len(parts[0]) == 4 and parts[0].isdigit()):
        return None
    if not parts[1].isdigit():
        return None
    month = int(parts[1])
    if month < 1 or month > 12:
        return None
    return f"{parts[0]}-{month:02d}"


def load_rules(root: Path | None = None) -> list[LedgerRule]:
    directory = Path(root) if root is not None else _RULES_DIR
    if not directory.is_dir():
        return []
    rules: list[LedgerRule] = []
    for path in sorted(directory.glob("*.md")):
        rule = _rule_from_file(path)
        if rule is not None:
            rules.append(rule)
    return rules


def search_rules(
    *,
    enterprise: str,
    month: str,
    topic: str | None = None,
    include_disabled: bool = False,
    root: Path | None = None,
) -> list[LedgerRule]:
    """按企业、业务月份和启用状态选规则。同一编号只保留生效日最新的启用版。"""
    code = normalize_enterprise(enterprise)
    month_key = normalize_month(month)
    if code is None or month_key is None:
        return []
    topic_key = str(topic or "").strip()
    matched: list[LedgerRule] = []
    for rule in load_rules(root):
        if code not in rule.enterprises:
            continue
        if topic_key and rule.topic != topic_key:
            continue
        if rule.effective_from[:7] > month_key:
            continue
        if rule.status != "active" and not include_disabled:
            continue
        matched.append(rule)
    if include_disabled:
        return sorted(matched, key=lambda item: (item.rule_id, item.effective_from, item.version))
    chosen: dict[str, LedgerRule] = {}
    for rule in matched:
        current = chosen.get(rule.rule_id)
        if current is None or rule.effective_from > current.effective_from:
            chosen[rule.rule_id] = rule
    return [chosen[key] for key in sorted(chosen)]


def set_rule_status(
    rule_id: str,
    version: str,
    status: str,
    *,
    root: Path | None = None,
) -> Path:
    if status not in {"active", "disabled"}:
        raise ValueError("invalid_status")
    match = _find_rule(rule_id, version, root)
    text = match.path.read_text(encoding="utf-8")
    end = text.find("\n---", 4)
    if not text.startswith("---\n") or end < 0:
        raise ValueError("front_matter_missing")
    lines: list[str] = []
    found = False
    for line in text[:end].splitlines():
        if line.startswith("status:"):
            lines.append(f"status: {status}")
            found = True
        else:
            lines.append(line)
    if not found:
        raise ValueError("status_field_missing")
    updated = "\n".join(lines) + text[end:]
    if not updated.endswith("\n"):
        updated += "\n"
    match.path.write_text(updated, encoding="utf-8")
    return match.path


def import_rule_file(source: Path, *, root: Path | None = None) -> Path:
    src = Path(source)
    text = src.read_text(encoding="utf-8")
    parsed = _rule_from_text(text, path=src)
    if parsed is None:
        raise ValueError("rule_front_matter_invalid")
    directory = Path(root) if root is not None else _RULES_DIR
    directory.mkdir(parents=True, exist_ok=True)
    dest = directory / src.name
    if dest.exists():
        raise FileExistsError(dest)
    dest.write_text(text if text.endswith("\n") else text + "\n", encoding="utf-8")
    return dest


def _find_rule(rule_id: str, version: str, root: Path | None) -> LedgerRule:
    for rule in load_rules(root):
        if rule.rule_id == rule_id and rule.version == version:
            return rule
    raise FileNotFoundError(f"{rule_id} {version}")


def _rule_from_file(path: Path) -> LedgerRule | None:
    return _rule_from_text(path.read_text(encoding="utf-8"), path=path)


def _rule_from_text(text: str, *, path: Path) -> LedgerRule | None:
    if not text.startswith("---\n"):
        return None
    end = text.find("\n---", 4)
    if end < 0:
        return None
    meta: dict[str, str] = {}
    for line in text[4:end].splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        key, separator, value = line.partition(":")
        if not separator:
            continue
        meta[key.strip()] = value.strip().strip("\"'")
    if any(not meta.get(key) for key in _REQUIRED):
        return None
    enterprises = _parse_enterprises(meta["enterprises"])
    if not enterprises:
        return None
    return LedgerRule(
        rule_id=meta["rule_id"],
        name=meta["name"],
        version=meta["version"],
        enterprises=enterprises,
        effective_from=meta["effective_from"],
        status=meta["status"],
        topic=meta["topic"],
        body=text[end + 4 :].strip(),
        path=path,
        purpose_required=meta.get("purpose_required") or None,
        purpose_threshold=meta.get("purpose_threshold") or None,
    )


def _parse_enterprises(value: str) -> tuple[str, ...]:
    text = value.strip()
    if text.startswith("[") and text.endswith("]"):
        text = text[1:-1]
    parts = [normalize_enterprise(item) for item in text.split(",")]
    return tuple(item for item in parts if item)
