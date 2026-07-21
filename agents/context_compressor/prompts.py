"""上下文压缩器 Prompt。"""

SUMMARY_PROMPT = """请更新金融对话摘要（控制在约 {summary_limit} tokens 以内，简洁中文）。

已有摘要：
{existing_summary}

新增对话：
{conversation}

请保留：
1. 用户正在讨论的公司、证券或金融产品
2. 时间范围、财务指标、币种和单位
3. 用户已经确认的口径与限制条件
4. 尚未解决的问题
5. 已经给出的重要结论

不要记录：
1. 寒暄和重复内容
2. 工具执行细节
3. 无法确认的用户偏好

更新后的摘要：
"""

SUMMARY_SHRINK_PROMPT = """请将以下金融对话摘要压缩到更短，保留关键实体、指标、时间与未决问题，控制在约 {summary_limit} tokens 以内。

原摘要：
{summary}

压缩后的摘要：
"""
