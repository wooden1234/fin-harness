"""领域工具薄 provider。"""

from __future__ import annotations

from typing import Any, Mapping

from harness.tools.definition import ToolDefinition, function_schema

_EXCLUDED_PRODUCT_TOOLS = frozenset({"finance.query_advanced"})


def _schema_from_langchain(name: str, description: str, langchain_tool: Any) -> dict[str, Any]:
    try:
        from langchain_core.utils.function_calling import convert_to_openai_tool

        schema = convert_to_openai_tool(langchain_tool)
        if isinstance(schema, dict) and schema.get("type") == "function":
            return schema
    except Exception:  # noqa: BLE001
        pass
    return function_schema(name, description)


async def _handle_weather(arguments: dict[str, Any]) -> Mapping[str, Any]:
    from capabilities.weather import get_current_weather

    return await get_current_weather(str(arguments.get("city") or ""))


async def _handle_faq(arguments: dict[str, Any]) -> Mapping[str, Any]:
    from capabilities.knowledge import search_faq

    return await search_faq(
        str(arguments.get("query") or ""),
        domain=str(arguments.get("domain") or "capital_market"),
        top_k=int(arguments.get("top_k") or 3),
        research_question_id=str(arguments.get("research_question_id") or "ad-hoc"),
    )


async def _handle_pdf_catalog(arguments: dict[str, Any]) -> Mapping[str, Any]:
    from capabilities.knowledge import catalog_pdf

    return await catalog_pdf(
        query=str(arguments.get("query") or ""),
        entity=str(arguments.get("entity") or ""),
        year=arguments.get("year"),
        categories=arguments.get("categories") or (),
        doc_id=str(arguments.get("doc_id") or ""),
    )


async def _handle_pdf_search(arguments: dict[str, Any]) -> Mapping[str, Any]:
    from capabilities.knowledge import search_pdf

    return await search_pdf(
        query=str(arguments.get("query") or ""),
        research_question_id=str(arguments.get("research_question_id") or "ad-hoc"),
        categories=arguments.get("categories") or (),
        doc_ids=arguments.get("doc_ids") or (),
        top_k=int(arguments.get("top_k") or 5),
        source_locked=bool(arguments.get("source_locked")),
    )


async def _handle_web(arguments: dict[str, Any]) -> Mapping[str, Any]:
    from capabilities.knowledge import search_web

    return await search_web(str(arguments.get("query") or ""))


async def _handle_fact(arguments: dict[str, Any]) -> Mapping[str, Any]:
    from capabilities.finance import lookup_fact

    return await lookup_fact(arguments)


async def _handle_calculation(arguments: dict[str, Any]) -> Mapping[str, Any]:
    from capabilities.calculation import run_calculation

    return await run_calculation(arguments)


async def _handle_analysis(arguments: dict[str, Any]) -> Mapping[str, Any]:
    from capabilities.analysis import synthesize_answer

    return await synthesize_answer(
        question=str(arguments.get("question") or ""),
        materials=arguments.get("materials") or [],
    )


def _iwencai_handler(tool_id: str):
    async def _handle(arguments: dict[str, Any]) -> Mapping[str, Any]:
        from capabilities.market import run_iwencai

        return await run_iwencai(tool_id, arguments)

    return _handle


_CAPABILITY_HANDLERS = {
    "weather.get": _handle_weather,
    "knowledge.faq.search": _handle_faq,
    "knowledge.pdf.catalog": _handle_pdf_catalog,
    "knowledge.pdf.search": _handle_pdf_search,
    "web.search": _handle_web,
    "knowledge.fact.lookup": _handle_fact,
    "finance.fact.lookup": _handle_fact,
    "calculation.run": _handle_calculation,
    "finalign.analyze": _handle_analysis,
}


def definition_from_registered(entry: Any) -> ToolDefinition:
    spec = entry.spec
    handler = _CAPABILITY_HANDLERS.get(spec.tool_id)
    if handler is None and str(spec.tool_id).startswith("iwencai."):
        handler = _iwencai_handler(spec.tool_id)
    if handler is None:

        async def _fallback(arguments: dict[str, Any], *, _entry=entry) -> Mapping[str, Any]:
            result = await _entry.handler(arguments)
            payload = dict(result) if isinstance(result, dict) else {"ok": True, "data": result}
            payload.setdefault("ok", True)
            return payload

        handler = _fallback

    langchain_tool = entry.langchain_tool
    schema = (
        _schema_from_langchain(spec.name, spec.description, langchain_tool)
        if langchain_tool is not None
        else function_schema(spec.name, spec.description)
    )
    return ToolDefinition(
        tool_id=spec.tool_id,
        name=spec.name,
        description=spec.description,
        handler=handler,
        openai_schema=schema,
        is_concurrency_safe=bool(spec.read_only) and spec.tool_id != "finalign.analyze",
        read_only=bool(spec.read_only),
        requires_human_approval=bool(spec.requires_human_approval),
        timeout_seconds=float(spec.timeout_seconds),
        max_retries=int(spec.max_retries),
    )


def product_definitions() -> tuple[ToolDefinition, ...]:
    from tools.core.catalog import load_all_tools
    from tools.core.registry import list_registered_tools

    load_all_tools()
    definitions: list[ToolDefinition] = []
    for entry in list_registered_tools():
        if entry.spec.tool_id in _EXCLUDED_PRODUCT_TOOLS:
            continue
        definitions.append(definition_from_registered(entry))
    return tuple(definitions)
