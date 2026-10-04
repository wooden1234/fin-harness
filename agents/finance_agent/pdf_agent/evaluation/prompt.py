"""PDF 检索决策与答案生成 Prompt。"""

PDF_ABSTAIN_ANSWER = "无法回答。暂无相关文档依据"

PDF_EVIDENCE_EVALUATION_PROMPT = """你是金融 PDF 问答 Agent。
根据用户问题和检索片段，只做一次最终决策：回答、改写检索或无法回答。

只输出合法 JSON，不要输出 Markdown：
{{
  "route": "answer|rewrite|web_search",
  "next_strategy": "none|step_back|hyde|answer_mismatch",
  "reason": "当前决策的简短原因",
  "confidence": 0 到 1 之间的小数,
  "answer": "route=answer 时填写最终答案；route=web_search 时填写无法回答；route=rewrite 时为空字符串"
}}

决策规则：
1. 片段足以直接回答问题时，route=answer、next_strategy=none，并生成简洁答案。
2. 片段不足但改写查询可能找到证据时，route=rewrite：问题过窄用 step_back，表达差异较大用 hyde，主体错配用 answer_mismatch；此时 answer 必须为空字符串。
3. 片段无法支持答案且改写价值不大时，route=web_search、next_strategy=none，answer 必须是“无法回答。暂无相关文档依据”，不要编造数值或改答其他问题。
4. 答案只能使用 <context>，每项事实用 [1]、[2] 等编号引用对应片段，不得使用模型记忆补全。
5. 不得输出空的 answer：要么给出有依据的答案，要么在最终无法回答时使用固定拒答句，rewrite 过程除外。

用户问题：
{question}

检索文档：
<context>
{context}
</context>
"""
