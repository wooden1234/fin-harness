"""生成赛题二完成说明 PPT，版式对齐深蓝标题 + 蓝色说明条 + 黄字重点。"""

from __future__ import annotations

from pathlib import Path

from lxml import etree
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.oxml.ns import qn
from pptx.util import Emu, Inches, Pt

NAVY = RGBColor(0x00, 0x20, 0x58)
HEX = RGBColor(0x24, 0x3B, 0x6D)
BLUE = RGBColor(0x5A, 0x9B, 0xD5)
YELLOW = RGBColor(0xF0, 0xF8, 0x28)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
GRAY = RGBColor(0x42, 0x42, 0x42)
BODY = RGBColor(0x33, 0x33, 0x33)
CARD = RGBColor(0xF7, 0xF9, 0xFC)
LINE = RGBColor(0xD5, 0xDE, 0xEA)
FONT = "微软雅黑"

SLIDE_W = Inches(13.333)
SLIDE_H = Inches(7.5)


def _font(run, size: float, color: RGBColor, bold: bool = False) -> None:
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.color.rgb = color
    run.font.name = FONT
    r_pr = run._r.get_or_add_rPr()
    r_fonts = r_pr.find(qn("a:rFonts"))
    if r_fonts is None:
        r_fonts = etree.SubElement(r_pr, qn("a:rFonts"))
    for attr in ("ascii", "hAnsi", "eastAsia", "cs"):
        r_fonts.set(qn(f"a:{attr}"), FONT)


def _box(slide, left, top, width, height, fill: RGBColor, rounded: bool = True, line: RGBColor | None = None):
    kind = MSO_SHAPE.ROUNDED_RECTANGLE if rounded else MSO_SHAPE.RECTANGLE
    shape = slide.shapes.add_shape(kind, left, top, width, height)
    shape.fill.solid()
    shape.fill.fore_color.rgb = fill
    if line is None:
        shape.line.fill.background()
    else:
        shape.line.color.rgb = line
        shape.line.width = Pt(1)
    if rounded:
        shape.adjustments[0] = 0.08
    return shape


def _text(slide, left, top, width, height, text, size, color, bold=False, align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP):
    box = slide.shapes.add_textbox(left, top, width, height)
    frame = box.text_frame
    frame.word_wrap = True
    frame.auto_size = None
    frame.margin_left = Emu(0)
    frame.margin_right = Emu(0)
    frame.margin_top = Emu(0)
    frame.margin_bottom = Emu(0)
    try:
        frame._txBody.bodyPr.set("anchor", {MSO_ANCHOR.TOP: "t", MSO_ANCHOR.MIDDLE: "ctr", MSO_ANCHOR.BOTTOM: "b"}[anchor])
    except Exception:
        pass
    paragraph = frame.paragraphs[0]
    paragraph.alignment = align
    paragraph.space_before = Pt(0)
    paragraph.space_after = Pt(0)
    run = paragraph.add_run()
    run.text = text
    _font(run, size, color, bold)
    return box


def _header(slide, title: str) -> None:
    mark = slide.shapes.add_shape(MSO_SHAPE.HEXAGON, Inches(0.38), Inches(0.20), Inches(0.70), Inches(0.62))
    mark.fill.solid()
    mark.fill.fore_color.rgb = HEX
    mark.line.fill.background()
    frame = mark.text_frame
    frame.word_wrap = False
    frame.margin_left = Emu(0)
    frame.margin_right = Emu(0)
    paragraph = frame.paragraphs[0]
    paragraph.alignment = PP_ALIGN.CENTER
    run = paragraph.add_run()
    run.text = "账"
    _font(run, 18, WHITE, True)
    frame._txBody.bodyPr.set("anchor", "ctr")
    _text(slide, Inches(1.22), Inches(0.24), Inches(11.6), Inches(0.58), title, 30, NAVY, True, anchor=MSO_ANCHOR.MIDDLE)


def _section(slide, text: str) -> None:
    _text(slide, Inches(0.42), Inches(0.98), Inches(12.4), Inches(0.38), text, 18, GRAY, True)


def _narrative(slide, parts: list[tuple[str, bool]], top=Inches(1.46), height=Inches(1.92)) -> None:
    shape = _box(slide, Inches(0.38), top, Inches(12.56), height, BLUE)
    frame = shape.text_frame
    frame.word_wrap = True
    frame.margin_left = Inches(0.24)
    frame.margin_right = Inches(0.24)
    frame.margin_top = Inches(0.16)
    frame.margin_bottom = Inches(0.12)
    frame._txBody.bodyPr.set("anchor", "ctr")
    paragraph = frame.paragraphs[0]
    paragraph.alignment = PP_ALIGN.LEFT
    paragraph.line_spacing = 1.15
    for text, emphasis in parts:
        pieces = text.split("\n")
        for index, piece in enumerate(pieces):
            if index:
                paragraph = frame.add_paragraph()
                paragraph.alignment = PP_ALIGN.LEFT
                paragraph.line_spacing = 1.15
            if not piece:
                continue
            run = paragraph.add_run()
            run.text = piece
            _font(run, 16, YELLOW if emphasis else WHITE, bold=emphasis)


def _caption(slide, left, top, width, text: str) -> None:
    _text(slide, left, top, width, Inches(0.32), text, 12, GRAY, False, PP_ALIGN.CENTER)


def _card(slide, left, top, width, height, title: str, lines: list[str]) -> None:
    _box(slide, left, top, width, height, CARD, line=LINE)
    bar = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, left, top, Inches(0.08), height)
    bar.fill.solid()
    bar.fill.fore_color.rgb = BLUE
    bar.line.fill.background()
    _text(slide, left + Inches(0.22), top + Inches(0.12), width - Inches(0.34), Inches(0.34), title, 14, NAVY, True)
    body = slide.shapes.add_textbox(left + Inches(0.22), top + Inches(0.48), width - Inches(0.36), height - Inches(0.58))
    frame = body.text_frame
    frame.word_wrap = True
    frame.margin_left = Emu(0)
    frame.margin_right = Emu(0)
    frame.margin_top = Emu(0)
    for index, line in enumerate(lines):
        paragraph = frame.paragraphs[0] if index == 0 else frame.add_paragraph()
        paragraph.alignment = PP_ALIGN.LEFT
        paragraph.space_after = Pt(4)
        run = paragraph.add_run()
        run.text = line
        _font(run, 13, BODY, False)


def _pair(slide, left_title, left_lines, right_title, right_lines, left_caption, right_caption) -> None:
    top = Inches(3.56)
    height = Inches(2.92)
    gap = Inches(0.28)
    width = Inches(6.14)
    left = Inches(0.38)
    right = left + width + gap
    _card(slide, left, top, width, height, left_title, left_lines)
    _card(slide, right, top, width, height, right_title, right_lines)
    _caption(slide, left, Inches(6.58), width, left_caption)
    _caption(slide, right, Inches(6.58), width, right_caption)


def build(path: Path) -> None:
    presentation = Presentation()
    presentation.slide_width = SLIDE_W
    presentation.slide_height = SLIDE_H
    blank = presentation.slide_layouts[6]

    slide = presentation.slides.add_slide(blank)
    _header(slide, "财税 Agent Harness（初赛）")
    _section(slide, "赛题二 · 完成说明")
    _narrative(
        slide,
        [
            ("比赛要做的是账房对话，不是公开行情。", False),
            ("甲企业、乙企业", True),
            ("的发票金额、银行收支和用途规则，都从账本与规则库读取。模型只提出查询和补录，", False),
            ("身份、金额和能否写入由程序决定", True),
            ("，不能由对话改口。", False),
        ],
    )
    _pair(
        slide,
        "三类用户任务",
        ["查金额：发票总额、银行收入、银行支出分开汇总。", "对用途：按当时生效的规则找出仍缺用途的发票。", "补用途：只保存用户明确说出的用途。"],
        "两家企业",
        ["甲企业 C001：2026-09 起每张发票都要用途。", "乙企业 C002：单张不低于 1000.00 元才必须填写。", "银行流水两家都只读，不能在对话里改。"],
        "图1  比赛要完成的任务",
        "图2  企业与规则差异",
    )

    slide = presentation.slides.add_slide(blank)
    _header(slide, "财税 Agent Harness（初赛）")
    _section(slide, "1. 赛题要解决的三类任务")
    _narrative(
        slide,
        [
            ("用户问的是某企业某月的发票和银行。发票按", False),
            ("价税合计", True),
            ("相加；银行", False),
            ("收入和支出分开相加", True),
            ("，不用差额代替支出。用途按该月生效规则判断，缺了再由用户口述补上，备注不能顶替用途。", False),
        ],
    )
    _pair(
        slide,
        "要回答的问题",
        ["这个月发票一共多少钱，银行进了多少、出了多少。", "按哪条规则，哪些发票还没写用途。", "把某一张的用途写成用户刚说的那一句。"],
        "不能答错的边界",
        ["月份没说清就先问，不换成当前月。", "查询失败或无权限，不能说成 0.00。", "发票和银行对不上，不能推断没付款。"],
        "图3  用户会提出的问题",
        "图4  回答时必须守住的口径",
    )

    slide = presentation.slides.add_slide(blank)
    _header(slide, "财税 Agent Harness（初赛）")
    _section(slide, "2. 数据与规则放在哪里")
    _narrative(
        slide,
        [
            ("业务数据在 PostgreSQL 的 ", False),
            ("ledger", True),
            (" 模式，含企业、权限、发票和银行流水。\n规则按编号、版本、企业和生效月筛选，", False),
            ("不使用向量", True),
            ("。甲企业 2026 年 9 月用用途规则 2.0，停用的 1.0 只留作版本管理。", False),
        ],
    )
    _pair(
        slide,
        "账本里有什么",
        ["企业：甲企业 C001、乙企业 C002。", "发票金额以分为单位保存，展示时再还原成两位小数。", "银行流水按收入、支出分别保存，不能改写。"],
        "四条规则",
        ["R001  金额统计，两家都用，2026-01-01 起。", "R002  甲企业用途。2.0 要求每张都填。", "R003  乙企业用途。单张 ≥ 1000.00 元必填。", "R004  银行只读，两家都适用。"],
        "图5  业务数据",
        "图6  规则知识库",
    )

    slide = presentation.slides.add_slide(blank)
    _header(slide, "财税 Agent Harness（初赛）")
    _section(slide, "3. 一轮对话怎么走完")
    _narrative(
        slide,
        [
            ("用户的一句话对应一轮。模型不调用工具时，", False),
            ("直接发布回答并结束", True),
            ("。\n查账时规则和账本分成两个工具。发票、银行和用途检查按动作分别计数，", False),
            ("同一组参数最多再试一次", True),
            ("。", False),
        ],
    )
    _pair(
        slide,
        "规则工具",
        ["ledger_search_rules 只读。", "按企业、月份、主题取出当前生效版本。", "回答时写明规则名称和版本，不凭记忆转述。"],
        "账本工具",
        ["ledger_records 负责查询和补录。", "query_invoices：发票总额。", "query_bank：银行收入和支出。", "check_purpose：仍缺用途的发票。", "update_invoice：只写用户给出的用途或备注。"],
        "图7  规则从哪里来",
        "图8  金额和用途从哪里来",
    )

    slide = presentation.slides.add_slide(blank)
    _header(slide, "财税 Agent Harness（初赛）")
    _section(slide, "4. 身份按登录账户绑定")
    _narrative(
        slide,
        [
            ("app.users 的 ", False),
            ("ledger_code", True),
            (" 选择 U001 或 U002，聊天内容不能改身份。U001 可查看并修改两家。U002 ", False),
            ("只能查看甲企业", True),
            ("，不能看乙企业，也不能写入。拒绝和无权限都不回答成 0.00。", False),
        ],
    )
    _pair(
        slide,
        "U001",
        ["可查看甲企业、乙企业。", "可以补录用户明确给出的用途。", "不能修改金额、销售方、项目和开票日期。"],
        "U002",
        ["只能查看甲企业的发票和银行。", "查询乙企业会被拒绝，不返回 0，也不返回真实支出。", "修改甲企业发票同样被拒绝，库中的用途不变。"],
        "图9  可看可改",
        "图10  只读且限一家",
    )

    slide = presentation.slides.add_slide(blank)
    _header(slide, "财税 Agent Harness（初赛）")
    _section(slide, "5. 月份、金额和用途口径")
    _narrative(
        slide,
        [
            ("这句话和前文都没有写明哪一月时，", False),
            ("先追问", True),
            ("。「这个月」不能换成当前月，也不能拿未完成任务里的月份去查。只有查询成功且没有记录才回答 ", False),
            ("0.00", True),
            ("。发票与银行支出不相等，不能推断未付款。", False),
        ],
    )
    _pair(
        slide,
        "金额",
        ["发票总额是所选月份全部发票的价税合计。", "银行收入、支出各自相加，不互相抵销。", "2026-07 没有记录时，成功查询应回答 0.00。"],
        "用途",
        ["甲企业 9 月：每张发票都要填写。", "乙企业：等于 1000.00 元也要填写。", "备注、销售方、项目名称都不能当作用途。"],
        "图11  金额怎么加",
        "图12  用途怎么判",
    )

    slide = presentation.slides.add_slide(blank)
    _header(slide, "财税 Agent Harness（初赛）")
    _section(slide, "6. 未完成的补录会留下来")
    _narrative(
        slide,
        [
            ("每个登录用户、每家企业只保留", False),
            ("一条", True),
            ("未完成的用途补录。本轮工具成功返回后写入，", False),
            ("关闭页面也不会丢", True),
            ("。新对话里只有用户要接着补时才用它，并重新检查当前还缺哪些发票。", False),
        ],
    )
    _pair(
        slide,
        "什么时候写入",
        ["用途检查成功：按结果替换或删除这条记录。", "补录成功：从清单里去掉这一张。", "清单空了：删掉该企业的未完成记录。"],
        "什么时候不用",
        ["用户新问「这个月发票多少钱」，不能拿旧月份去查。", "不能只复述上次留下的发票号。", "列出仍缺用途之前，要重新做一次检查。"],
        "图13  进度怎么留下",
        "图14  新对话怎么接上",
    )

    slide = presentation.slides.add_slide(blank)
    _header(slide, "财税 Agent Harness（初赛）")
    _section(slide, "7. 示例数据上已经对过的结果")
    _narrative(
        slide,
        [
            ("甲企业 2026-09：发票 ", False),
            ("10 张、9622.06 元", True),
            ("，银行收入 13000.50 元、支出 7180.74 元。乙企业同月：发票 ", False),
            ("10 张、12667.24 元", True),
            ("，银行收入 13700.75 元、支出 8087.24 元。金额先按分累加，再还原成两位小数。", False),
        ],
    )
    _pair(
        slide,
        "甲企业 2026-09",
        ["发票合计 9622.06 元，共 10 张。", "银行收入 13000.50 元，支出 7180.74 元。", "用途规则：《甲企业发票用途规则》2.0。", "两边差额不是未付款。"],
        "乙企业 2026-09",
        ["发票合计 12667.24 元，共 10 张。", "银行收入 13700.75 元，支出 8087.24 元。", "用途规则：《乙企业发票用途规则》1.0。", "1000.00 元达到门槛，必须填写。"],
        "图15  甲企业核对结果",
        "图16  乙企业核对结果",
    )

    slide = presentation.slides.add_slide(blank)
    _header(slide, "财税 Agent Harness（初赛）")
    _section(slide, "8. 已经补上的运行问题")
    _narrative(
        slide,
        [
            ("模型若连续 ", False),
            ("30 秒", True),
            ("没有新内容，本轮结束并提示超时，不再停在「正在理解问题」。退出或切换账号会清掉页面上一位用户的对话。服务端会话列表", False),
            ("本来就按登录用户隔离", True),
            ("。", False),
        ],
    )
    _pair(
        slide,
        "请求会结束",
        ["DeepSeek 调用带有连接和读取超时。", "流式内容静默 30 秒即视为本轮失败。", "页面收到错误后停止计时，不再空转。"],
        "换账号不串对话",
        ["会话列表按当前登录用户查询。", "退出登录时清掉页面里的消息和当前会话。", "新账号登录后中间区域回到空白会话。"],
        "图17  长时间无响应",
        "图18  账号之间的页面隔离",
    )

    slide = presentation.slides.add_slide(blank)
    _header(slide, "财税 Agent Harness（初赛）")
    _section(slide, "9. 完成范围")
    _narrative(
        slide,
        [
            ("查询、规则选择、用途补录、权限拒绝、月份追问和未完成任务，已经接进现有对话。", False),
            ("设计说明、运行记录和不超过 5 分钟的演示", True),
            ("，以及启动说明里的导入步骤，还要单独成稿。", False),
        ],
    )
    _pair(
        slide,
        "已经接上",
        ["两个账房工具和 PostgreSQL 账本。", "登录身份、只读限制和写入拒绝。", "空结果、失败结果和未完成补录。", "简单问候会带上目前的比赛功能。"],
        "提交前还要整理",
        ["两三页设计说明。", "导出的运行记录。", "不超过 5 分钟的演示。", "启动文档写明导入和两个账户的权限。"],
        "图19  项目里已经完成的部分",
        "图20  比赛材料还缺的部分",
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    presentation.save(path)


if __name__ == "__main__":
    target = Path(__file__).resolve().parents[1] / "docs" / "赛题二-财税Agent-Harness完成说明.pptx"
    build(target)
    print(target)
