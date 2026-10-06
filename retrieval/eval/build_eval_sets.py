#!/usr/bin/env python3
"""Generate recall / parent / HyDE-Step-back / PDF answer eval JSONL."""

from __future__ import annotations

import json
from pathlib import Path

OUT = Path(__file__).resolve().parent


def node(doc: str, idx: int) -> dict:
    return {"doc_id": doc, "chunk_id": str(idx), "node_id": f"{doc}:L3:{idx:06d}"}


def dump(name: str, rows: list[dict]) -> None:
    path = OUT / name
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    buckets: dict[str, int] = {}
    for row in rows:
        buckets[row["bucket"]] = buckets.get(row["bucket"], 0) + 1
    print(f"{name}: {len(rows)}  {buckets}")


def recall_cases() -> list[dict]:
    rows: list[dict] = []

    def add(i: int, bucket: str, query: str, *, cats: list[str], gold: list[tuple[str, int]],
            hint: str, entities: list[str], expect_hit: bool = True,
            must_not: list[str] | None = None, hard: list[dict] | None = None, notes: str = "") -> None:
        rows.append({
            "id": f"rec-{i:03d}",
            "set": "recall",
            "bucket": bucket,
            "query": query,
            "categories": cats,
            "relevant_chunks": [node(d, c) for d, c in gold],
            "hard_negatives": hard or [],
            "must_not_docs": must_not or [],
            "expect_hit": expect_hit,
            "gold_answer_hint": hint,
            "must_keep_entities": entities,
            "eval_notes": notes,
        })

    # exact_hit 10
    add(1, "exact_hit", "寒武纪2024年营业收入是多少",
        cats=["annual_reports"], gold=[("PDF-AR-688256-2024", 6)],
        hint="1,174,464,377.35元，同比+65.56%", entities=["寒武纪", "2024", "营业收入"])
    add(2, "exact_hit", "龙芯中科2024年营业收入是多少",
        cats=["annual_reports"], gold=[("PDF-AR-688047-2024", 5)],
        hint="50,425.72万元，同比-0.28%", entities=["龙芯", "2024", "营业收入"])
    add(3, "exact_hit", "宁德时代2024年营业收入是多少",
        cats=["annual_reports"], gold=[("PDF-AR-CATL-2024", 5)],
        hint="362,012,554千元，同比-9.70%", entities=["宁德时代", "2024", "营业收入"])
    add(4, "exact_hit", "腾讯2025年收入是多少",
        cats=["annual_reports"], gold=[("PDF-AR-TCEHY-2025", 5)],
        hint="人民币751,766百万元", entities=["腾讯", "2025", "收入"])
    add(5, "exact_hit", "2026年一季度中国GDP同比增长多少",
        cats=["macro_research"], gold=[("PDF-MACRO-03", 1)],
        hint="同比增长5%", entities=["2026", "GDP", "一季度"])
    add(6, "exact_hit", "2026年3月末M2同比增长多少",
        cats=["macro_research"], gold=[("PDF-MACRO-03", 2)],
        hint="M2同比8.5%，社融存量同比7.9%", entities=["M2", "2026"])
    add(7, "exact_hit", "腾讯2024年末微信及WeChat合并月活跃账户数是多少",
        cats=["annual_reports"], gold=[("PDF-AR-TCEHY-2024", 4)],
        hint="1,385百万", entities=["腾讯", "微信", "2024"])
    add(8, "exact_hit", "时尚春熙平台累计为多少银发用户提供服务",
        cats=["industry_whitepapers"], gold=[("PDF-WP-07", 3)],
        hint="7.8万特定客群，数字化工具使用率79.99%", entities=["春熙", "银发"])
    add(9, "exact_hit", "光模块在光通信系统设备中的成本占比大约是多少",
        cats=["research_reports"], gold=[("PDF-RR-20260420", 1)],
        hint="成本占比超过50%", entities=["光模块", "成本"])
    add(10, "exact_hit", "2026年3月1年期LPR是多少",
        cats=["macro_research"], gold=[("PDF-MACRO-03", 17)],
        hint="1年期3.0%、5年期以上3.5%，均同比下降0.1个百分点", entities=["LPR", "2026"])

    # table_numeric 8
    add(11, "table_numeric", "寒武纪2024年其他业务销售毛利率是多少",
        cats=["annual_reports"], gold=[("PDF-AR-688256-2024", 31)],
        hint="其他业务毛利率95.78%", entities=["寒武纪", "2024", "其他业务", "毛利率"])
    add(12, "table_numeric", "寒武纪2024年境内业务毛利率是多少",
        cats=["annual_reports"], gold=[("PDF-AR-688256-2024", 31)],
        hint="境内毛利率56.73%", entities=["寒武纪", "2024", "境内", "毛利率"])
    add(13, "table_numeric", "龙芯中科2024年第四季度营业收入是多少",
        cats=["annual_reports"], gold=[("PDF-AR-688047-2024", 7)],
        hint="Q4营业收入19,647.66万元", entities=["龙芯", "2024", "第四季度"])
    add(14, "table_numeric", "宁德时代2024年第一季度营业收入是多少",
        cats=["annual_reports"], gold=[("PDF-AR-CATL-2024", 7)],
        hint="Q1营业收入79,770,779（与年报单位一致，千元）", entities=["宁德时代", "2024", "第一季度"])
    add(15, "table_numeric", "腾讯2024年增值服务收入是多少",
        cats=["annual_reports"], gold=[("PDF-AR-TCEHY-2024", 7)],
        hint="增值服务319,168百万，占收入49%", entities=["腾讯", "2024", "增值服务"])
    add(16, "table_numeric", "2023年东部地区商品房销售面积增长率是多少",
        cats=["macro_research"], gold=[("PDF-MACRO-01", 28)],
        hint="东部销售面积-6.7%，销售额-5.8%", entities=["东部", "商品房", "2023"])
    add(17, "table_numeric", "中国数字经济指数里基础设施一级指标权重是多少",
        cats=["industry_whitepapers"], gold=[("PDF-WP-02", 2)],
        hint="基础设施权重0.36，产业发展0.33，数字治理0.20", entities=["数字经济", "基础设施", "权重"])
    add(18, "table_numeric", "CoreWeave在2026年一季度经调整EBITDA大约是多少",
        cats=["research_reports"], gold=[("PDF-RR-20260531", 97)],
        hint="adj-EBITDA约12亿美元，Margin 56%，净亏损7.40亿美元", entities=["CoreWeave", "EBITDA", "2026"])

    # year_swap 8
    add(19, "year_swap", "寒武纪2025年营业收入是多少",
        cats=["annual_reports"], gold=[("PDF-AR-688256-2025", 6)],
        hint="6,497,196,198.68元，同比+453.21%", entities=["寒武纪", "2025", "营业收入"],
        must_not=["PDF-AR-688256-2024"],
        hard=[{"doc_id": "PDF-AR-688256-2024", "chunk_id": "6", "reason": "同公司2024营收表"}])
    add(20, "year_swap", "寒武纪2024年营业收入同比增速是多少",
        cats=["annual_reports"], gold=[("PDF-AR-688256-2024", 6)],
        hint="同比+65.56%，不要答成2025年的453.21%", entities=["寒武纪", "2024"],
        must_not=["PDF-AR-688256-2025"])
    add(21, "year_swap", "龙芯中科2025年营业收入是多少",
        cats=["annual_reports"], gold=[("PDF-AR-688047-2025", 6)],
        hint="63,532.06万元，同比+25.99%", entities=["龙芯", "2025"],
        must_not=["PDF-AR-688047-2024"])
    add(22, "year_swap", "龙芯中科2024年归属于上市公司股东的净利润是多少",
        cats=["annual_reports"], gold=[("PDF-AR-688047-2024", 5)],
        hint="-62,534.71万元", entities=["龙芯", "2024", "净利润"],
        must_not=["PDF-AR-688047-2025"])
    add(23, "year_swap", "宁德时代2025年营业收入是多少",
        cats=["annual_reports"], gold=[("PDF-AR-CATL-2025", 19)],
        hint="423,701,834，同比+17.04%（港股年报口径）", entities=["宁德时代", "2025"],
        must_not=["PDF-AR-CATL-2024"])
    add(24, "year_swap", "腾讯2024年全年收入是多少",
        cats=["annual_reports"], gold=[("PDF-AR-TCEHY-2024", 5)],
        hint="660,257百万，不要答成2025年的751,766", entities=["腾讯", "2024"],
        must_not=["PDF-AR-TCEHY-2025"])
    add(25, "year_swap", "腾讯2025年末微信及WeChat合并月活跃账户数是多少",
        cats=["annual_reports"], gold=[("PDF-AR-TCEHY-2025", 4)],
        hint="1,418百万，同比+2%", entities=["腾讯", "微信", "2025"],
        must_not=["PDF-AR-TCEHY-2024"])
    add(26, "year_swap", "宁德时代2024年营业收入同比变动是多少",
        cats=["annual_reports"], gold=[("PDF-AR-CATL-2024", 5)],
        hint="同比-9.70%，不要答成2025年的+17.04%", entities=["宁德时代", "2024"],
        must_not=["PDF-AR-CATL-2025"])

    # entity_swap 6
    add(27, "entity_swap", "寒武纪2024年归母净利润是多少",
        cats=["annual_reports"], gold=[("PDF-AR-688256-2024", 6)],
        hint="-452,338,791.01元，不要串成龙芯-62,534.71万元", entities=["寒武纪", "2024", "净利润"],
        must_not=["PDF-AR-688047-2024", "PDF-AR-688047-2025"])
    add(28, "entity_swap", "龙芯中科的法定代表人是谁",
        cats=["annual_reports"], gold=[("PDF-AR-688047-2024", 0)],
        hint="胡伟武", entities=["龙芯", "法定代表人"],
        must_not=["PDF-AR-688256-2024"])
    add(29, "entity_swap", "宁德时代2025年全球有多少家电池工厂",
        cats=["annual_reports"], gold=[("PDF-AR-CATL-2025", 50)],
        hint="六大研发中心、24家电池工厂", entities=["宁德时代", "工厂"],
        must_not=["PDF-AR-TCEHY-2025"])
    add(30, "entity_swap", "腾讯2024年营销服务收入是多少",
        cats=["annual_reports"], gold=[("PDF-AR-TCEHY-2024", 7)],
        hint="营销服务121,374百万，不要答成增值服务319,168", entities=["腾讯", "营销服务", "2024"],
        must_not=["PDF-AR-CATL-2024"])
    add(31, "entity_swap", "东方证券国产算力研报建议关注哪些国产GPU标的",
        cats=["research_reports"], gold=[("PDF-RR-20260525", 2)],
        hint="寒武纪、海光信息、沐曦、摩尔线程等", entities=["东方证券", "GPU"],
        must_not=["PDF-AR-688256-2024"],
        notes="研报点名寒武纪，gold应在RR而非年报")
    add(32, "entity_swap", "中际旭创1.6T光模块何时开始向重点客户出货",
        cats=["research_reports"], gold=[("PDF-RR-20260420", 39)],
        hint="2025年第三季度开始正式向重点客户出货", entities=["中际旭创", "1.6T"],
        must_not=["PDF-RR-20260607"])

    # cross_category 5
    add(33, "cross_category", "东方证券如何评价寒武纪思元590相对H200的算力水平",
        cats=["research_reports"], gold=[("PDF-RR-20260525", 11)],
        hint="以H200为基准，思元590约49%", entities=["寒武纪", "思元590"],
        must_not=["PDF-AR-688256-2024", "PDF-AR-688256-2025"])
    add(34, "cross_category", "寒武纪2024年年报中云端产品线营业收入是多少",
        cats=["annual_reports"], gold=[("PDF-AR-688256-2024", 288)],
        hint="云端产品线营业收入1,166,278,485.36元", entities=["寒武纪", "云端产品线", "2024"],
        must_not=["PDF-RR-20260525"])
    add(35, "cross_category", "DeepSeek算力网白皮书称1000GB自动驾驶数据确定性网络传输要多久",
        cats=["industry_whitepapers"], gold=[("PDF-WP-05", 4)],
        hint="不到5分钟，传统网络约10天", entities=["DeepSeek", "1000GB"],
        must_not=["PDF-RR-20260525"])
    add(36, "cross_category", "东吴证券给出的CoreWeave 2026年资本开支指引是多少",
        cats=["research_reports"], gold=[("PDF-RR-20260531", 100)],
        hint="全年指引310-350亿美元，1Q26 capex约68亿美元", entities=["CoreWeave", "资本开支"],
        must_not=["PDF-WP-04"])
    add(37, "cross_category", "微软出海白皮书中的Pegasus计划是做什么的",
        cats=["industry_whitepapers"], gold=[("PDF-WP-03", 120)],
        hint="Microsoft for Startups进阶GTM与联合销售计划", entities=["微软", "Pegasus"],
        must_not=["PDF-WP-01"])

    # paraphrase 5
    add(38, "paraphrase", "寒武纪2024年卖了多少货，营收大概什么量级",
        cats=["annual_reports"], gold=[("PDF-AR-688256-2024", 6)],
        hint="约11.74亿元", entities=["寒武纪", "2024"])
    add(39, "paraphrase", "龙芯中科2024年亏了多少",
        cats=["annual_reports"], gold=[("PDF-AR-688047-2024", 5)],
        hint="归母净利润-62,534.71万元", entities=["龙芯", "2024"])
    add(40, "paraphrase", "宁德时代2024年卖电池收入是增了还是掉了",
        cats=["annual_reports"], gold=[("PDF-AR-CATL-2024", 5)],
        hint="营业收入同比-9.70%", entities=["宁德时代", "2024"])
    add(41, "paraphrase", "央行说2026年开年经济增了几个点",
        cats=["macro_research"], gold=[("PDF-MACRO-03", 1)],
        hint="一季度GDP同比增长5%", entities=["2026", "GDP"])
    add(42, "paraphrase", "春熙路银发商圈这两年大概做了多少交易额",
        cats=["industry_whitepapers"], gold=[("PDF-WP-07", 3)],
        hint="促成交易超184万笔，消费金额超1.6亿元", entities=["春熙", "交易"])

    # hard_negative 5
    add(43, "hard_negative", "寒武纪2024年其他业务的毛利率是多少，不要用2025年报",
        cats=["annual_reports"], gold=[("PDF-AR-688256-2024", 31)],
        hint="2024其他业务95.78%；2025“其他”毛利率22.11%是干扰", entities=["寒武纪", "2024", "其他业务"],
        must_not=["PDF-AR-688256-2025"],
        hard=[{"doc_id": "PDF-AR-688256-2025", "chunk_id": "29", "reason": "2025“其他”毛利率22.11%，数值完全不同"}])
    add(44, "hard_negative", "龙芯中科2024年合并利润表里的资产减值损失是多少",
        cats=["annual_reports"], gold=[("PDF-AR-688047-2024", 155)],
        hint="资产减值损失-145,857,438.68元", entities=["龙芯", "2024", "资产减值"],
        must_not=["PDF-AR-688256-2024"],
        hard=[{"doc_id": "PDF-AR-688047-2024", "chunk_id": "8", "reason": "非经常性损益里的减值冲销，不是利润表损失科目"}])
    add(45, "hard_negative", "哪篇研报写了光模块占系统设备成本超过50%",
        cats=["research_reports"], gold=[("PDF-RR-20260420", 1)],
        hint="慧博光模块深度第2页，不要召回CPO或白皮书", entities=["光模块", "成本占比"],
        must_not=["PDF-RR-20260607", "PDF-WP-05"])
    add(46, "hard_negative", "2021年人身险公司保费收入是多少",
        cats=["macro_research"], gold=[("PDF-MACRO-02", 94)],
        hint="31,224亿元，同比+5.01%", entities=["人身险", "保费", "2021"],
        must_not=["PDF-MACRO-03", "PDF-MACRO-01"])
    add(47, "hard_negative", "LightCounting对CPO市场规模2027年的预测是多少",
        cats=["research_reports"], gold=[("PDF-RR-20260607", 3)],
        hint="2027年有望突破50亿美元", entities=["CPO", "2027"],
        must_not=["PDF-RR-20260420"])

    # unanswerable 3
    add(48, "unanswerable", "华为2024年净利润是多少",
        cats=["annual_reports"], gold=[], hint="语料无华为年报", entities=["华为", "2024"],
        expect_hit=False, notes="Top-K 命中寒武纪/龙芯视为误召回")
    add(49, "unanswerable", "比亚迪2025年动力电池装机量是多少",
        cats=["annual_reports"], gold=[], hint="语料无比亚迪年报", entities=["比亚迪", "2025"],
        expect_hit=False, notes="宁德时代工厂/动力电池段落是硬负例")
    add(50, "unanswerable", "工信部第十一次中小企业圆桌会议是谁主持的",
        cats=["policy"], gold=[], hint="policy 未清洗入库", entities=["工信部", "圆桌会议"],
        expect_hit=False, notes="旧索引里的 PDF-POL 不在本次 cleaned 集合")

    assert len(rows) == 50
    return rows


def parent_cases() -> list[dict]:
    rows: list[dict] = []

    def add(i: int, bucket: str, query: str, *, cats: list[str], leaves: list[tuple[str, int]],
            facts: list[str], reason: str, entities: list[str]) -> None:
        rows.append({
            "id": f"par-{i:03d}",
            "set": "parent",
            "bucket": bucket,
            "query": query,
            "categories": cats,
            "relevant_chunks": [node(d, c) for d, c in leaves],
            "min_leaf_count": 2,
            "gold_facts": facts,
            "leaf_insufficient_reason": reason,
            "must_keep_entities": entities,
            "success_if": "任意单叶子缺至少一条 gold_facts；同 section 连续叶子或 L2 父块覆盖全部 gold_facts",
        })

    # table_split 12
    add(1, "table_split", "寒武纪2024年分行业和其他业务毛利率分别是多少",
        cats=["annual_reports"], leaves=[("PDF-AR-688256-2024", 30), ("PDF-AR-688256-2024", 31)],
        facts=["集成电路行业分产品表在30", "其他业务毛利率95.78%在31", "境内毛利率56.73%在31"],
        reason="同一张分行业分地区表被切成连续叶子，单块看不到完整行列",
        entities=["寒武纪", "2024", "毛利率"])
    add(2, "table_split", "寒武纪2024年主营业务分地区和分销售模式的毛利率对比",
        cats=["annual_reports"], leaves=[("PDF-AR-688256-2024", 31), ("PDF-AR-688256-2024", 32)],
        facts=["境内/境外毛利率在31", "销售模式表接在后续叶子"],
        reason="地区表与销售模式表跨叶子",
        entities=["寒武纪", "分地区"])
    add(3, "table_split", "寒武纪2024年成本分析里云端产品线直接材料发生了什么",
        cats=["annual_reports"], leaves=[("PDF-AR-688256-2024", 34), ("PDF-AR-688256-2024", 35)],
        facts=["成本分析表头在34/35", "云端产品线直接材料261,563,612.41，同比大增因收入增长"],
        reason="成本分析表跨两片，材料金额与变动说明不在同一叶子",
        entities=["寒武纪", "云端产品线", "直接材料"])
    add(4, "table_split", "龙芯中科2024年主要控股子公司里广东龙芯和北京龙芯净利润分别是多少",
        cats=["annual_reports"], leaves=[("PDF-AR-688047-2024", 41), ("PDF-AR-688047-2024", 42)],
        facts=["广东龙芯净利润-706.33", "北京龙芯净利润-2,356量级，在下一张表"],
        reason="控股参股公司表按行切开，两家公司不在同一叶子",
        entities=["龙芯", "广东龙芯", "北京龙芯"])
    add(5, "table_split", "龙芯中科2025年营业收入扣除项目和扣除后收入分别是什么",
        cats=["annual_reports"], leaves=[("PDF-AR-688047-2025", 12), ("PDF-AR-688047-2025", 13)],
        facts=["营业收入63,532.06、扣除项目合计0", "与主营无关收入小计等明细在下一片"],
        reason="扣除情况表被切成表头块和明细块",
        entities=["龙芯", "2025", "营业收入扣除"])
    add(6, "table_split", "宁德时代2024年分业务毛利率和分行业收入结构分别是什么",
        cats=["annual_reports"], leaves=[("PDF-AR-CATL-2024", 14), ("PDF-AR-CATL-2024", 16)],
        facts=["营业收入合计362,012,554，电气机械占98.48%", "电气机械毛利率24.69%"],
        reason="收入结构表与毛利率表是连续但分开的叶子",
        entities=["宁德时代", "2024", "毛利率"])
    add(7, "table_split", "宁德时代2024年引用了哪些锂电和新能源相关政策文件",
        cats=["annual_reports"], leaves=[("PDF-AR-CATL-2024", 10), ("PDF-AR-CATL-2024", 11), ("PDF-AR-CATL-2024", 13)],
        facts=["2024年3月国务院以旧换新方案", "2024年6月工信部锂电规范条件", "2024年12月电力系统调节能力方案"],
        reason="政策表按时间行切开，单行无法回答“哪些政策”",
        entities=["宁德时代", "政策"])
    add(8, "table_split", "数字经济白皮书各省数字治理指数表里北京和湖北2024年分别多少",
        cats=["industry_whitepapers"], leaves=[("PDF-WP-02", 38), ("PDF-WP-02", 39)],
        facts=["北京2024年12.05（排名1）", "湖北2024年8.56（排名约11）"],
        reason="同一张省际表按排名切片，北京与湖北分属不同叶子",
        entities=["数字治理指数", "北京", "湖北"])
    add(9, "table_split", "数字经济白皮书数字治理指数表陕西和西藏2024年分别多少",
        cats=["industry_whitepapers"], leaves=[("PDF-WP-02", 40), ("PDF-WP-02", 41)],
        facts=["陕西2024年6.11", "西藏2024年2.97"],
        reason="排名21以后的省份在后续切片",
        entities=["陕西", "西藏", "数字治理"])
    add(10, "table_split", "腾讯2024年增值服务的收入和毛利率分别是多少",
        cats=["annual_reports"], leaves=[("PDF-AR-TCEHY-2024", 7), ("PDF-AR-TCEHY-2024", 8)],
        facts=["增值服务收入319,168百万", "增值服务毛利181,657百万、毛利率57%"],
        reason="收入结构表与毛利率表被拆开",
        entities=["腾讯", "增值服务"])
    add(11, "table_split", "货币政策报告2026年3月LPR水平以及贷款利率相对LPR的区间分布",
        cats=["macro_research"], leaves=[("PDF-MACRO-03", 17), ("PDF-MACRO-03", 19)],
        facts=["1年期LPR 3.0%、5年期以上3.5%", "表4给出减点/加点区间占比"],
        reason="LPR文字与区间占比表分属相邻叶子",
        entities=["LPR", "贷款利率"])
    add(12, "table_split", "宁德时代港股年报释义里CTP和DPPB分别指什么",
        cats=["annual_reports"], leaves=[("PDF-AR-CATL-2025", 3), ("PDF-AR-CATL-2025", 4)],
        facts=["本公司=宁德时代新能源科技股份有限公司", "CTP=电芯直接集成到电池包", "DPPB=十亿分之一失效率"],
        reason="释义表跨多片，公司主体与技术缩写不在同一叶子",
        entities=["宁德时代", "CTP"])

    # header_body 8
    add(13, "header_body", "寒武纪2024年主要会计数据表中营收、归母净利润、经营现金流分别是多少",
        cats=["annual_reports"], leaves=[("PDF-AR-688256-2024", 6), ("PDF-AR-688256-2024", 8)],
        facts=["营收1,174,464,377.35", "归母净利润-452,338,791.01", "基本每股收益-1.09在财务指标表"],
        reason="会计数据表与财务指标表分开，完整画像需要两块",
        entities=["寒武纪", "2024"])
    add(14, "header_body", "龙芯2024年营收几乎持平的同时净利润为什么更差，需要看主表还是季度表",
        cats=["annual_reports"], leaves=[("PDF-AR-688047-2024", 5), ("PDF-AR-688047-2024", 7)],
        facts=["全年营收50,425.72、同比-0.28%", "归母净利润-62,534.71", "Q4净利润-28,258.75是全年最差季度"],
        reason="全年主表看不到季度亏损集中，季度表看不到全年同比",
        entities=["龙芯", "2024", "净利润"])
    add(15, "header_body", "宁德时代2025年全年营收和分季度营收是否对得上",
        cats=["annual_reports"], leaves=[("PDF-AR-CATL-2025", 19), ("PDF-AR-CATL-2025", 22)],
        facts=["全年营业收入423,701,834", "四季营收84,704,589+94,181,664+104,185,734+140,629,847"],
        reason="全年摘要与分季度表拆开，单块无法核对加总",
        entities=["宁德时代", "2025"])
    add(16, "header_body", "腾讯2025年收入、毛利和经营盈利分别是多少",
        cats=["annual_reports"], leaves=[("PDF-AR-TCEHY-2025", 5), ("PDF-AR-TCEHY-2025", 7)],
        facts=["收入751,766百万、毛利422,593百万", "分业务收入在下一张表"],
        reason="利润表总览与分业务收入表分离",
        entities=["腾讯", "2025"])
    add(17, "header_body", "春熙银发白皮书7.8万用户覆盖是怎么从2023年涨上来的",
        cats=["industry_whitepapers"], leaves=[("PDF-WP-07", 3), ("PDF-WP-07", 4)],
        facts=["累计服务7.8万、交易超184万笔", "表2-1：2023年8月1.45万人到2025年11月7.8万"],
        reason="成效综述在3、时间序列表在4，单块无法回答“怎么涨”",
        entities=["春熙", "7.8万"])
    add(18, "header_body", "慧博光模块报告里低速/中高速/超高速模块怎么划分",
        cats=["research_reports"], leaves=[("PDF-RR-20260420", 3), ("PDF-RR-20260420", 4)],
        facts=["文字：低速1/2.5/10G、中高速25/40/100G", "表格补充400/800G等超高速档"],
        reason="分类叙述与分类表被切开",
        entities=["光模块", "传输速率"])
    add(19, "header_body", "东兴证券CPO报告投资摘要里方案定义和市场规模预测分别在哪",
        cats=["research_reports"], leaves=[("PDF-RR-20260607", 2), ("PDF-RR-20260607", 3)],
        facts=["CPO=光引擎与交换芯片共基板封装", "2027年市场有望突破50亿美元"],
        reason="定义与规模预测分属摘要连续段",
        entities=["CPO", "50亿"])
    add(20, "header_body", "OpenClaw白皮书典型工作流有哪几步、能省多少时间",
        cats=["industry_whitepapers"], leaves=[("PDF-WP-01", 5), ("PDF-WP-01", 6)],
        facts=["早晨总结、会前15分钟提醒、会后纪要、下班汇报", "每天节省2-3小时", "邮件自动分类在后续叶子"],
        reason="工作流步骤与邮件场景被拆开，省时结论只在工作流块",
        entities=["OpenClaw", "工作流"])

    # section_span 10
    add(21, "section_span", "央行2026Q1货币政策报告内容摘要如何同时描述总量和结构信贷",
        cats=["macro_research"], leaves=[("PDF-MACRO-03", 1), ("PDF-MACRO-03", 2)],
        facts=["一季度GDP同比5%、实施适度宽松", "社融7.9%、M2 8.5%，科技/绿色/普惠贷款两位数增长"],
        reason="摘要被切成连续段，总量与结构不在同一叶子",
        entities=["货币政策", "社融", "M2"])
    add(22, "section_span", "2026Q1货币政策报告对外部风险和下阶段利率汇率政策怎么写",
        cats=["macro_research"], leaves=[("PDF-MACRO-03", 3), ("PDF-MACRO-03", 4)],
        facts=["地缘风险上升、主要经济体分化", "完善利率传导、结构性工具、保持汇率弹性"],
        reason="风险判断与政策应对分属摘要后两段",
        entities=["货币政策", "汇率"])
    add(23, "section_span", "2024区域金融运行报告内容摘要如何概括产业和信贷结构",
        cats=["macro_research"], leaves=[("PDF-MACRO-01", 5), ("PDF-MACRO-01", 6), ("PDF-MACRO-01", 7)],
        facts=["区域增速差距收窄", "装备制造贡献接近五成", "绿色贷款同比36.5%、普惠小微23.5%"],
        reason="摘要连续三段分别讲协调、产业、信贷",
        entities=["区域金融", "绿色贷款"])
    add(24, "section_span", "2022金融稳定报告综述如何同时写资本市场改革和宏观审慎框架",
        cats=["macro_research"], leaves=[("PDF-MACRO-02", 6), ("PDF-MACRO-02", 7)],
        facts=["设立北交所、注册制条件逐步具备", "发布宏观审慎政策指引、D-SIB名单"],
        reason="综述条目跨叶子",
        entities=["金融稳定", "宏观审慎"])
    add(25, "section_span", "微软出海白皮书序言认为企业出海内涵发生了什么变化",
        cats=["industry_whitepapers"], leaves=[("PDF-WP-03", 1), ("PDF-WP-03", 2), ("PDF-WP-03", 3)],
        facts=["从产品渠道成本转向品牌运营合规智能", "AI成为能力底座", "谁更早建立全球能力谁更能长期增长"],
        reason="序言论点跨三段，单段只覆盖部分命题",
        entities=["微软", "出海"])
    add(26, "section_span", "微软白皮书认为当前出海趋势有哪些方向",
        cats=["industry_whitepapers"], leaves=[("PDF-WP-03", 4), ("PDF-WP-03", 5), ("PDF-WP-03", 6)],
        facts=["出海从阶段性选择变长期战略", "市场布局更多元", "数智技术重塑出海方式"],
        reason="趋势章标题段与分论点段被切开",
        entities=["出海", "AI"])
    add(27, "section_span", "DeepSeek算力网白皮书前言如何定义通用大模型局限和确定性网络价值",
        cats=["industry_whitepapers"], leaves=[("PDF-WP-05", 3), ("PDF-WP-05", 4)],
        facts=["通用大模型不知工厂/医院行业数据", "1000GB自动驾驶数据不到5min传完"],
        reason="问题定义与方案收益被拆开",
        entities=["DeepSeek", "行业大模型"])
    add(28, "section_span", "东吴证券如何定义NeoCloud以及投资建议关注谁",
        cats=["research_reports"], leaves=[("PDF-RR-20260531", 1), ("PDF-RR-20260531", 2), ("PDF-RR-20260531", 3)],
        facts=["AI算力供需错配催生NeoCloud", "盈利拆成单位算力收入-TCO-资本成本", "海外关注CoreWeave等"],
        reason="投资要点连续三段，定义/框架/标的分离",
        entities=["NeoCloud", "CoreWeave"])
    add(29, "section_span", "慧博算电协同报告如何从定义讲到国家战略",
        cats=["research_reports"], leaves=[("PDF-RR-20260601", 0), ("PDF-RR-20260601", 1), ("PDF-RR-20260601", 2)],
        facts=["2023年12月提出、2026年政府工作报告升为国家战略", "电支撑算+算优化电", "枢纽节点绿电占比目标"],
        reason="封面定义、内涵、政策升级跨三叶子",
        entities=["算电协同"])
    add(30, "section_span", "宁德时代2025年动力电池产品覆盖哪些化学体系和哪些应用",
        cats=["annual_reports"], leaves=[("PDF-AR-CATL-2025", 53), ("PDF-AR-CATL-2025", 54)],
        facts=["磷酸铁锂/三元/钠离子/凝聚态等", "乘用车BEV/PHEV及商用车船舶航空"],
        reason="产品化学体系与应用场景分属连续叶子",
        entities=["宁德时代", "动力电池"])

    # multi_fact 10
    add(31, "multi_fact", "CoreWeave 1Q26为什么EBITDA很强但净利润和自由现金流很差",
        cats=["research_reports"], leaves=[("PDF-RR-20260531", 97), ("PDF-RR-20260531", 100)],
        facts=["adj-EBITDA约12亿美元、Margin 56%", "净亏损7.40亿美元，利息费用升至5.36亿", "1Q26 capex约68亿，全年指引310-350亿"],
        reason="利润表特征与现金流/capex在同节不同叶子",
        entities=["CoreWeave", "EBITDA", "资本开支"])
    add(32, "multi_fact", "中际旭创1.6T进度和800G份额分别怎么说",
        cats=["research_reports"], leaves=[("PDF-RR-20260420", 39), ("PDF-RR-20260420", 55)],
        facts=["1.6T于2025Q3向重点客户出货", "2025Q3中际旭创与新易盛合计占全球800G及以上大部分份额"],
        reason="出货进度与竞争格局不在同一叶子",
        entities=["中际旭创", "1.6T"])
    add(33, "multi_fact", "寒武纪2025年营收暴增的同时研发投入和扣股份支付净利润怎么变",
        cats=["annual_reports"], leaves=[("PDF-AR-688256-2025", 6), ("PDF-AR-688256-2025", 12), ("PDF-AR-688256-2025", 22)],
        facts=["营收6,497,196,198.68，同比+453.21%", "扣股份支付后净利润2,297,212,898.47", "费用化研发投入1,169,100,962.15，+9.03%"],
        reason="主表、扣股份支付表、研发表分属不同叶子",
        entities=["寒武纪", "2025"])
    add(34, "multi_fact", "龙芯2025年营收增长后是否已经盈利，扣除股份支付后呢",
        cats=["annual_reports"], leaves=[("PDF-AR-688047-2025", 6), ("PDF-AR-688047-2025", 14)],
        facts=["营收63,532.06，同比+25.99%", "扣股份支付后净利润-42,608.11，仍亏损"],
        reason="主表与扣股份支付表分开",
        entities=["龙芯", "2025", "净利润"])
    add(35, "multi_fact", "2026Q1美国经济韧性和后续值得关注的外部冲击分别是什么",
        cats=["macro_research"],         leaves=[("PDF-MACRO-03", 79), ("PDF-MACRO-03", 86)],
        facts=["美国GDP环比折年率2.0%，AI投资和政府消费支持", "中东冲突、霍尔木兹、15%进口关税"],
        reason="经济概况与“值得关注的问题”分节，叶子检索常只中一块",
        entities=["美国", "韧性", "关税"])
    add(36, "multi_fact", "人身险2021年保费还在增长，利润为什么大幅下降",
        cats=["macro_research"], leaves=[("PDF-MACRO-02", 94)],
        facts=["保费31,224亿元同比+5.01%", "税前利润同比-47.36%，27家公司亏损"],
        reason="同段内两事实；若再切更碎则必须父块。评测时对照相邻保险节叶子是否被漏召",
        entities=["人身险", "2021"])
    add(37, "multi_fact", "施耐德AI就绪数据中心如何同时讲液冷、供应链脱碳和DCIM",
        cats=["industry_whitepapers"], leaves=[("PDF-WP-04", 4), ("PDF-WP-04", 13), ("PDF-WP-04", 21)],
        facts=["Uniflair CDU液冷", "改造电气组件最多可节省90%资源", "EcoStruxure IT不依赖特定厂商"],
        reason="卖点分布在不同小节，需要父级或多叶子合并才像完整方案",
        entities=["施耐德", "液冷", "脱碳"])
    add(38, "multi_fact", "国海证券如何同时定义超节点并给出2026年国内CSP开支判断",
        cats=["research_reports"], leaves=[("PDF-RR-20260330", 6), ("PDF-RR-20260330", 7)],
        facts=["超节点是高速互联整合数十至数百GPU的架构", "2026年国内CSP资本开支展望乐观"],
        reason="定义与需求判断是相邻投资要点",
        entities=["超节点", "CSP"])
    add(39, "multi_fact", "东方证券国产算力报告核心观点如何同时覆盖模型和光模块测试设备",
        cats=["research_reports"], leaves=[("PDF-RR-20260525", 0), ("PDF-RR-20260525", 1)],
        facts=["2026年万亿参数MoE与国产芯片全栈适配", "光模块测试设备国产替代进入关键窗口"],
        reason="核心观点被切成模型段与运力段",
        entities=["国产算力", "光模块"])
    add(40, "multi_fact", "腾讯2024年微信月活和全年收入、增值服务占比能否放在一起看",
        cats=["annual_reports"], leaves=[("PDF-AR-TCEHY-2024", 4), ("PDF-AR-TCEHY-2024", 5), ("PDF-AR-TCEHY-2024", 7)],
        facts=["微信月活1,385百万", "收入660,257百万", "增值服务占收入49%"],
        reason="KPI、利润表、分业务表分属经营综述不同叶子",
        entities=["腾讯", "2024", "微信"])

    # comparison 6
    add(41, "comparison", "寒武纪2024与2025年营业收入和同比增速对比",
        cats=["annual_reports"],
        leaves=[("PDF-AR-688256-2024", 6), ("PDF-AR-688256-2025", 6)],
        facts=["2024营收11.74亿同比+65.56%", "2025营收64.97亿同比+453.21%"],
        reason="跨文档对比；父块不能跨年报，但检索侧应同时召回两年主表而不是混成一张表",
        entities=["寒武纪", "2024", "2025"])
    add(42, "comparison", "龙芯2024与2025年营业收入和是否扭亏对比",
        cats=["annual_reports"],
        leaves=[("PDF-AR-688047-2024", 5), ("PDF-AR-688047-2025", 6)],
        facts=["2024营收50,425.72仍大额亏损", "2025营收63,532.06，扣股份支付后仍亏"],
        reason="跨年对比需要两份主表，单叶子会串年",
        entities=["龙芯", "2024", "2025"])
    add(43, "comparison", "宁德时代2024年A股年报与2025年港股年报营业收入口径对比时要注意什么",
        cats=["annual_reports"],
        leaves=[("PDF-AR-CATL-2024", 5), ("PDF-AR-CATL-2025", 19)],
        facts=["2024年报362,012,554千元同比-9.70%", "2025港股年报423,701,834同比+17.04%"],
        reason="两套报表语言和上市地不同，父块无效跨文档，但应用来测“不要用错年报当父上下文”",
        entities=["宁德时代", "2024", "2025"])
    add(44, "comparison", "腾讯2024与2025年收入和微信月活怎么变",
        cats=["annual_reports"],
        leaves=[("PDF-AR-TCEHY-2024", 4), ("PDF-AR-TCEHY-2024", 5), ("PDF-AR-TCEHY-2025", 4), ("PDF-AR-TCEHY-2025", 5)],
        facts=["2024收入660,257、微信1,385百万", "2025收入751,766、微信1,418百万"],
        reason="KPI与利润表、两年四块，单叶子无法完成对比",
        entities=["腾讯", "微信"])
    add(45, "comparison", "东部与东北2023年商品房销售面积降幅谁更大",
        cats=["macro_research"], leaves=[("PDF-MACRO-01", 28)],
        facts=["东部-6.7%", "东北-3.0%", "中部-13.2%最大"],
        reason="表在一块但表头与分区行若再切会丢；作为对照题：父块应保住表头单位=%",
        entities=["商品房", "东部", "东北"])
    add(46, "comparison", "CPO投资摘要里技术路线（COUPE）和出货预测是否在同一证据里",
        cats=["research_reports"], leaves=[("PDF-RR-20260607", 3), ("PDF-RR-20260607", 4)],
        facts=["2027年市场50亿美元、2029年1.6T CPO出货上调至约900万个", "台积电COUPE把EIC堆叠在PIC上"],
        reason="规模预测与COUPE技术路径是连续摘要段",
        entities=["CPO", "COUPE"])

    # narrative_split 4
    add(47, "narrative_split", "OpenClaw如何从“数字分身”讲到邮件清理效果",
        cats=["industry_whitepapers"], leaves=[("PDF-WP-01", 0), ("PDF-WP-01", 6), ("PDF-WP-01", 7)],
        facts=["OpenClaw=数字分身+私人秘书+自动化管家", "自动分类归档回复", "从每天200+封邮件变为只关注重要事项"],
        reason="定位、能力、效果分属封面/场景叶子",
        entities=["OpenClaw", "邮件"])
    add(48, "narrative_split", "银发白皮书用王阿姨的故事要说明什么结构性问题",
        cats=["industry_whitepapers"], leaves=[("PDF-WP-07", 1), ("PDF-WP-07", 13)],
        facts=["72岁王阿姨现金支付、不会扫码", "中国银发数字化仍处基础信息化向智慧融合早期"],
        reason="故事在开篇，阶段判断在后文，叶子检索常只中故事",
        entities=["银发", "数字化"])
    add(49, "narrative_split", "区域金融报告开头如何同时给出报告定位和主要内容框架",
        cats=["macro_research"], leaves=[("PDF-MACRO-01", 0), ("PDF-MACRO-01", 3), ("PDF-MACRO-01", 5)],
        facts=["人行货币政策分析小组2024年7月26日", "2023年是二十大开局之年", "区域发展协调性增强"],
        reason="目录/摘要定位跨多段",
        entities=["区域金融", "2023"])
    add(50, "narrative_split", "施耐德如何把可持续咨询和AI数据中心交付收束成合作呼吁",
        cats=["industry_whitepapers"], leaves=[("PDF-WP-04", 7), ("PDF-WP-04", 22), ("PDF-WP-04", 23)],
        facts=["可持续战略须与AI增长相适应", "规划设计建造运营一站式伙伴", "给出WP110/WP106等参考设计"],
        reason="咨询卖点、收束段、资源列表被拆开",
        entities=["施耐德", "AI就绪"])

    assert len(rows) == 50
    return rows


def rewrite_cases() -> list[dict]:
    rows: list[dict] = []

    def add(i: str, bucket: str, query: str, strategy: str, *, cats: list[str],
            gold: list[tuple[str, int]], entities: list[str], notes: str = "",
            terms: list[str] | None = None, anchors: list[str] | None = None) -> None:
        row = {
            "id": i,
            "set": "hyde_stepback",
            "bucket": bucket,
            "query": query,
            "expect_strategy": strategy,
            "must_keep_entities": entities,
            "categories": cats,
            "relevant_chunks": [node(d, c) for d, c in gold],
            "eval_notes": notes,
        }
        if terms:
            row["must_keep_terms"] = terms
        if anchors:
            row["expected_anchors"] = anchors
        rows.append(row)

    # keep original ids
    add("sb-001", "too_narrow", "寒武纪2024年其他业务销售毛利率是多少", "step_back",
        cats=["annual_reports"], gold=[("PDF-AR-688256-2024", 31)],
        entities=["寒武纪", "2024", "毛利率"])
    add("hyde-001", "vocab_gap", "龙芯2024年利润为什么亏这么多", "hyde",
        cats=["annual_reports"], gold=[("PDF-AR-688047-2024", 5), ("PDF-AR-688047-2024", 155)],
        entities=["龙芯", "2024"])
    add("hyde-002", "vocab_gap", "一季度美国经济为什么还有韧性，机制是什么", "hyde",
        cats=["macro_research"], gold=[("PDF-MACRO-03", 79)],
        entities=["美国", "经济"])
    add("ctrl-001", "control", "寒武纪2024年营业收入是多少", "none",
        cats=["annual_reports"], gold=[("PDF-AR-688256-2024", 6)],
        entities=["寒武纪", "2024", "营业收入"])
    add("ctrl-002", "control", "2026年3月1年期LPR是多少", "none",
        cats=["macro_research"], gold=[("PDF-MACRO-03", 17)],
        entities=["LPR", "2026"])

    # too_narrow / step_back 再 15 条，合计 16
    add("sb-002", "too_narrow", "龙芯中科2024年第四季度归母净利润是多少", "step_back",
        cats=["annual_reports"], gold=[("PDF-AR-688047-2024", 7)],
        entities=["龙芯", "2024", "净利润"])
    add("sb-003", "too_narrow", "2023年东部地区商品房销售面积增长率是多少个百分点", "step_back",
        cats=["macro_research"], gold=[("PDF-MACRO-01", 28)],
        entities=["东部", "商品房", "2023"])
    add("sb-004", "too_narrow", "CoreWeave 1Q26利息费用从多少升到多少", "step_back",
        cats=["research_reports"], gold=[("PDF-RR-20260531", 97)],
        entities=["CoreWeave", "利息费用"])
    add("sb-005", "too_narrow", "中国数字经济指数基础设施一级指标权重精确值", "step_back",
        cats=["industry_whitepapers"], gold=[("PDF-WP-02", 2)],
        entities=["数字经济", "基础设施"])
    add("sb-006", "too_narrow", "腾讯2024年增值服务毛利率百分比", "step_back",
        cats=["annual_reports"], gold=[("PDF-AR-TCEHY-2024", 8)],
        entities=["腾讯", "增值服务", "毛利率"])
    add("sb-007", "too_narrow", "寒武纪2024年云端产品线直接材料金额", "step_back",
        cats=["annual_reports"], gold=[("PDF-AR-688256-2024", 35)],
        entities=["寒武纪", "云端产品线"])
    add("sb-008", "too_narrow", "广东龙芯2024年净利润是多少", "step_back",
        cats=["annual_reports"], gold=[("PDF-AR-688047-2024", 41)],
        entities=["广东龙芯", "净利润"])
    add("sb-009", "too_narrow", "数字治理指数表中北京2024年数值", "step_back",
        cats=["industry_whitepapers"], gold=[("PDF-WP-02", 38)],
        entities=["北京", "数字治理"])
    add("sb-010", "too_narrow", "中际旭创1.6T是在2025年第几季度向重点客户出货", "step_back",
        cats=["research_reports"], gold=[("PDF-RR-20260420", 39)],
        entities=["中际旭创", "1.6T"])
    add("sb-011", "too_narrow", "OpenClaw典型工作流声称每天节省几小时", "step_back",
        cats=["industry_whitepapers"], gold=[("PDF-WP-01", 5)],
        entities=["OpenClaw"])
    add("sb-012", "too_narrow", "2021年人身险公司退保率是多少", "step_back",
        cats=["macro_research"], gold=[("PDF-MACRO-02", 94)],
        entities=["人身险", "退保率"])
    add("sb-013", "too_narrow", "宁德时代2024年第一季度营业收入精确值", "step_back",
        cats=["annual_reports"], gold=[("PDF-AR-CATL-2024", 7)],
        entities=["宁德时代", "2024", "第一季度"])
    add("sb-014", "too_narrow", "2026年3月末科技贷款同比增速", "step_back",
        cats=["macro_research"], gold=[("PDF-MACRO-03", 2)],
        entities=["科技贷款", "2026"])
    add("sb-015", "too_narrow", "时尚春熙数字化工具使用率精确到小数点", "step_back",
        cats=["industry_whitepapers"], gold=[("PDF-WP-07", 3)],
        entities=["春熙", "数字化工具"])
    add("sb-016", "too_narrow", "寒武纪2024年境外业务毛利率是多少", "step_back",
        cats=["annual_reports"], gold=[("PDF-AR-688256-2024", 31)],
        entities=["寒武纪", "境外", "毛利率"])

    # vocab_gap / hyde 再 16 条，合计 18（含 hyde-001/002）
    add("hyde-003", "vocab_gap", "寒武纪2025年怎么突然赚这么多", "hyde",
        cats=["annual_reports"], gold=[("PDF-AR-688256-2025", 6), ("PDF-AR-688256-2025", 27)],
        entities=["寒武纪", "2025"])
    add("hyde-004", "vocab_gap", "宁德时代2024年卖电池怎么还少了", "hyde",
        cats=["annual_reports"], gold=[("PDF-AR-CATL-2024", 5), ("PDF-AR-CATL-2024", 14)],
        entities=["宁德时代", "2024"])
    add("hyde-005", "vocab_gap", "腾讯游戏和广告到底哪块更赚钱", "hyde",
        cats=["annual_reports"], gold=[("PDF-AR-TCEHY-2024", 7), ("PDF-AR-TCEHY-2024", 8)],
        entities=["腾讯", "增值服务", "营销服务"])
    add("hyde-006", "vocab_gap", "AI数据中心为啥要上液冷", "hyde",
        cats=["industry_whitepapers"], gold=[("PDF-WP-04", 4)],
        entities=["液冷", "数据中心"])
    add("hyde-007", "vocab_gap", "中国企业出海现在最大的坑是什么", "hyde",
        cats=["industry_whitepapers"], gold=[("PDF-WP-03", 33)],
        entities=["出海", "合规"])
    add("hyde-008", "vocab_gap", "算力租赁公司为啥利润难看但EBITDA看起来很好", "hyde",
        cats=["research_reports"], gold=[("PDF-RR-20260531", 97)],
        entities=["CoreWeave", "EBITDA"])
    add("hyde-009", "vocab_gap", "光模块为啥要从800G往1.6T切", "hyde",
        cats=["research_reports"], gold=[("PDF-RR-20260420", 14), ("PDF-RR-20260420", 39)],
        entities=["光模块", "1.6T"])
    add("hyde-010", "vocab_gap", "老人不会扫码、商圈该怎么做数字化", "hyde",
        cats=["industry_whitepapers"], gold=[("PDF-WP-07", 1), ("PDF-WP-07", 3)],
        entities=["银发", "数字化"])
    add("hyde-011", "vocab_gap", "通用大模型为啥干不了工厂产线的活", "hyde",
        cats=["industry_whitepapers"], gold=[("PDF-WP-05", 3)],
        entities=["行业大模型", "通用大模型"])
    add("hyde-012", "vocab_gap", "超节点到底是个啥，和普通GPU服务器有啥差别", "hyde",
        cats=["research_reports"], gold=[("PDF-RR-20260330", 6)],
        entities=["超节点"])
    add("hyde-013", "vocab_gap", "算电协同是不是就是数据中心改用绿电", "hyde",
        cats=["research_reports"], gold=[("PDF-RR-20260601", 1), ("PDF-RR-20260601", 3)],
        entities=["算电协同"])
    add("hyde-014", "vocab_gap", "数字经济评分里基建这项为啥那么重要", "hyde",
        cats=["industry_whitepapers"], gold=[("PDF-WP-02", 2)],
        entities=["数字经济", "基础设施"])
    add("hyde-015", "vocab_gap", "寒武纪2024年还没盈利吗", "hyde",
        cats=["annual_reports"], gold=[("PDF-AR-688256-2024", 6)],
        entities=["寒武纪", "2024"])
    add("hyde-016", "vocab_gap", "央行这阵子货币政策算松还是紧", "hyde",
        cats=["macro_research"], gold=[("PDF-MACRO-03", 1), ("PDF-MACRO-03", 2)],
        entities=["货币政策", "适度宽松"])
    add("hyde-017", "vocab_gap", "CPO是不是把光模块焊到交换机芯片旁边", "hyde",
        cats=["research_reports"], gold=[("PDF-RR-20260607", 2)],
        entities=["CPO"])
    add("hyde-018", "vocab_gap", "微软给初创企业出海到底能提供啥额度", "hyde",
        cats=["industry_whitepapers"], gold=[("PDF-WP-03", 120)],
        entities=["微软", "Startups"])

    # control 再 8 条，合计 10
    add("ctrl-003", "control", "宁德时代2024年营业收入是多少", "none",
        cats=["annual_reports"], gold=[("PDF-AR-CATL-2024", 5)],
        entities=["宁德时代", "2024", "营业收入"])
    add("ctrl-004", "control", "腾讯2025年末微信及WeChat合并月活跃账户数", "none",
        cats=["annual_reports"], gold=[("PDF-AR-TCEHY-2025", 4)],
        entities=["腾讯", "微信", "2025"])
    add("ctrl-005", "control", "CoreWeave 2026年资本开支指引是多少", "none",
        cats=["research_reports"], gold=[("PDF-RR-20260531", 100)],
        entities=["CoreWeave", "资本开支", "2026"])
    add("ctrl-006", "control", "中际旭创是否已向重点客户出货1.6T光模块", "none",
        cats=["research_reports"], gold=[("PDF-RR-20260420", 39)],
        entities=["中际旭创", "1.6T"])
    add("ctrl-007", "control", "DeepSeek算力网白皮书的主要编写单位有哪些", "none",
        cats=["industry_whitepapers"], gold=[("PDF-WP-05", 1)],
        entities=["紫金山实验室", "未来网络"])
    add("ctrl-008", "control", "OpenClaw白皮书里的典型工作流包含哪些步骤", "none",
        cats=["industry_whitepapers"], gold=[("PDF-WP-01", 5)],
        entities=["OpenClaw", "工作流"])
    add("ctrl-009", "control", "2026年一季度国内生产总值同比增长5%是哪份报告写的", "none",
        cats=["macro_research"], gold=[("PDF-MACRO-03", 1)],
        entities=["GDP", "2026"])
    add("ctrl-010", "control", "龙芯中科2024年营业收入是多少", "none",
        cats=["annual_reports"], gold=[("PDF-AR-688047-2024", 5)],
        entities=["龙芯", "2024", "营业收入"])

    # answer_mismatch 6
    add("mm-001", "answer_mismatch", "寒武纪2024年营业收入是多少，不要答成龙芯", "answer_mismatch",
        cats=["annual_reports"], gold=[("PDF-AR-688256-2024", 6)],
        entities=["寒武纪", "2024", "营业收入"],
        notes="两家芯片年报主表结构几乎一样",
        terms=["寒武纪", "2024", "营业收入"], anchors=["年度报告", "主要财务指标"])
    add("mm-002", "answer_mismatch", "龙芯中科2025年营业收入，不要用2024年报", "answer_mismatch",
        cats=["annual_reports"], gold=[("PDF-AR-688047-2025", 6)],
        entities=["龙芯", "2025"],
        notes="同年公司相邻财年",
        terms=["龙芯中科", "2025", "营业收入"], anchors=["年度报告", "主要财务指标"])
    add("mm-003", "answer_mismatch", "宁德时代2025年营业收入，不要用腾讯收入", "answer_mismatch",
        cats=["annual_reports"], gold=[("PDF-AR-CATL-2025", 19)],
        entities=["宁德时代", "2025", "营业收入"],
        terms=["宁德时代", "2025", "营业收入"], anchors=["年度报告", "主要财务指标"])
    add("mm-004", "answer_mismatch", "光模块占光通信系统设备成本超过50%，不是电池成本", "answer_mismatch",
        cats=["research_reports"], gold=[("PDF-RR-20260420", 1)],
        entities=["光模块", "成本"],
        terms=["光模块", "光通信系统设备", "成本", "50%"], anchors=["成本构成"])
    add("mm-005", "answer_mismatch", "美国一季度经济韧性，不是中国GDP 5%", "answer_mismatch",
        cats=["macro_research"], gold=[("PDF-MACRO-03", 79)],
        entities=["美国", "韧性"],
        terms=["美国", "一季度", "经济韧性"], anchors=["主要经济指标"])
    add("mm-006", "answer_mismatch", "数字治理指数北京2024年数值，不是数字经济总指数", "answer_mismatch",
        cats=["industry_whitepapers"], gold=[("PDF-WP-02", 38)],
        entities=["数字治理", "北京", "2024"],
        terms=["数字治理指数", "北京", "2024", "数值"], anchors=["分指数表"])

    assert len(rows) == 50, len(rows)
    return rows


def answer_cases() -> list[dict]:
    """PDF Agent 最终答案的准确性与忠实度。

    准确性：required_facts 的 aliases 必须都能在答案里找到。
    忠实度：作答必须带 [n] 引用；forbidden_values 不得出现。
    expect_abstain 时必须包含“暂无相关文档依据”，且不得写出具体数值。
    gold_answer 是给人看的一句金标，便于对照模型输出；拒答题写明无法回答。
    """
    rows: list[dict] = []

    def gold_answer(*, abstain: bool, facts: list[dict], notes: str = "") -> str:
        if abstain:
            reason = str(notes or "").strip()
            if reason:
                return f"无法回答。暂无相关文档依据（{reason}）"
            return "无法回答。暂无相关文档依据"
        parts: list[str] = []
        for item in facts:
            subject = str(item.get("subject") or "").strip()
            period = str(item.get("period") or "").strip()
            metric = str(item.get("metric") or "").strip()
            unit = str(item.get("unit") or "").strip()
            aliases = [str(alias).strip() for alias in (item.get("aliases") or []) if str(alias).strip()]
            display = str(item.get("value") or "").strip()
            first_alias = ""
            formatted_alias = ""
            for alias in aliases:
                if alias == unit:
                    continue
                if not first_alias:
                    first_alias = alias
                if "," in alias or "万" in alias:
                    formatted_alias = alias
                    break
            display = formatted_alias or first_alias or display
            if unit and unit not in display and not display.endswith(("元", "万", "亿", "%", "百万")):
                display = f"{display}{unit}"
            label = " ".join(part for part in (subject, period, metric) if part)
            parts.append(f"{label}：{display}" if label else display)
        return "；".join(parts)

    def add(
        i: str,
        bucket: str,
        query: str,
        *,
        cats: list[str],
        gold: list[tuple[str, int]],
        facts: list[dict],
        forbidden: list[str],
        abstain: bool = False,
        notes: str = "",
    ) -> None:
        rows.append({
            "id": i,
            "set": "pdf_answer",
            "bucket": bucket,
            "query": query,
            "categories": cats,
            "relevant_chunks": [node(d, c) for d, c in gold],
            "gold_answer": gold_answer(abstain=abstain, facts=facts, notes=notes),
            "expect_abstain": abstain,
            "require_citation": not abstain,
            "abstain_markers": ["暂无相关文档依据"],
            "required_facts": facts,
            "forbidden_values": forbidden,
            "eval_notes": notes,
        })

    def fact(subject: str, period: str, metric: str, value: str, unit: str, aliases: list[str]) -> dict:
        return {
            "subject": subject,
            "period": period,
            "metric": metric,
            "value": value,
            "unit": unit,
            "aliases": aliases,
        }

    # exact_number：单点数值必须答对，并引用上下文
    add("ans-001", "exact_number", "寒武纪2024年营业收入是多少，同比增速是多少",
        cats=["annual_reports"], gold=[("PDF-AR-688256-2024", 6)],
        facts=[
            fact("寒武纪", "2024", "营业收入", "1174464377.35", "元", ["1174464377.35", "1,174,464,377.35"]),
            fact("寒武纪", "2024", "营业收入同比", "65.56", "%", ["65.56"]),
        ],
        forbidden=["6497196198.68", "453.21", "龙芯"])
    add("ans-002", "exact_number", "龙芯中科2024年营业收入和归母净利润分别是多少",
        cats=["annual_reports"], gold=[("PDF-AR-688047-2024", 5)],
        facts=[
            fact("龙芯中科", "2024", "营业收入", "50425.72", "万元", ["50425.72", "50,425.72"]),
            fact("龙芯中科", "2024", "归母净利润", "-62534.71", "万元", ["-62534.71", "-62,534.71"]),
        ],
        forbidden=["寒武纪", "63532.06"])
    add("ans-003", "exact_number", "宁德时代2024年营业收入是多少，同比变动是多少",
        cats=["annual_reports"], gold=[("PDF-AR-CATL-2024", 5)],
        facts=[
            fact("宁德时代", "2024", "营业收入", "362012554", "千元", ["362012554", "362,012,554"]),
            fact("宁德时代", "2024", "营业收入同比", "-9.70", "%", ["-9.70", "-9.7"]),
        ],
        forbidden=["423701834", "17.04"])
    add("ans-004", "exact_number", "腾讯2024年全年收入是多少",
        cats=["annual_reports"], gold=[("PDF-AR-TCEHY-2024", 5)],
        facts=[fact("腾讯", "2024", "收入", "660257", "百万元", ["660257", "660,257"])],
        forbidden=["751766", "751,766"])
    add("ans-005", "exact_number", "2026年3月1年期和5年期以上LPR分别是多少",
        cats=["macro_research"], gold=[("PDF-MACRO-03", 17)],
        facts=[
            fact("LPR", "2026-03", "1年期", "3.0", "%", ["3.0", "3.0%"]),
            fact("LPR", "2026-03", "5年期以上", "3.5", "%", ["3.5", "3.5%"]),
        ],
        forbidden=[])
    add("ans-006", "exact_number", "寒武纪2024年其他业务毛利率和境内业务毛利率分别是多少",
        cats=["annual_reports"], gold=[("PDF-AR-688256-2024", 31)],
        facts=[
            fact("寒武纪", "2024", "其他业务毛利率", "95.78", "%", ["95.78"]),
            fact("寒武纪", "2024", "境内毛利率", "56.73", "%", ["56.73"]),
        ],
        forbidden=["22.11"])
    add("ans-007", "exact_number", "腾讯2024年末微信及WeChat合并月活跃账户数是多少",
        cats=["annual_reports"], gold=[("PDF-AR-TCEHY-2024", 4)],
        facts=[fact("腾讯", "2024", "微信及WeChat合并月活", "1385", "百万", ["1385", "1,385"])],
        forbidden=["1418", "1,418"])
    add("ans-008", "exact_number", "2026年一季度中国GDP同比增长多少",
        cats=["macro_research"], gold=[("PDF-MACRO-03", 1)],
        facts=[fact("中国", "2026Q1", "GDP同比", "5", "%", ["5%", "同比增长5"])],
        forbidden=["美国"])

    # entity_year：主体或年份错了就是不准确，也是把别的文档事实写进答案
    add("ans-009", "entity_year", "寒武纪2025年营业收入和同比增速是多少",
        cats=["annual_reports"], gold=[("PDF-AR-688256-2025", 6)],
        facts=[
            fact("寒武纪", "2025", "营业收入", "6497196198.68", "元", ["6497196198.68", "6,497,196,198.68"]),
            fact("寒武纪", "2025", "营业收入同比", "453.21", "%", ["453.21"]),
        ],
        forbidden=["1174464377.35", "1,174,464,377.35", "65.56"])
    add("ans-010", "entity_year", "龙芯中科2025年营业收入是多少",
        cats=["annual_reports"], gold=[("PDF-AR-688047-2025", 6)],
        facts=[fact("龙芯中科", "2025", "营业收入", "63532.06", "万元", ["63532.06", "63,532.06"])],
        forbidden=["50425.72", "50,425.72"])
    add("ans-011", "entity_year", "腾讯2025年收入是多少",
        cats=["annual_reports"], gold=[("PDF-AR-TCEHY-2025", 5)],
        facts=[fact("腾讯", "2025", "收入", "751766", "百万元", ["751766", "751,766"])],
        forbidden=["660257", "660,257"])
    add("ans-012", "entity_year", "寒武纪2024年归母净利润是多少",
        cats=["annual_reports"], gold=[("PDF-AR-688256-2024", 6)],
        facts=[fact("寒武纪", "2024", "归母净利润", "-452338791.01", "元", ["-452338791.01", "-452,338,791.01"])],
        forbidden=["-62534.71", "-62,534.71", "龙芯"])
    add("ans-013", "entity_year", "腾讯2024年营销服务收入是多少",
        cats=["annual_reports"], gold=[("PDF-AR-TCEHY-2024", 7)],
        facts=[fact("腾讯", "2024", "营销服务收入", "121374", "百万元", ["121374", "121,374"])],
        forbidden=["319168", "319,168"])
    add("ans-014", "entity_year", "腾讯2025年末微信及WeChat合并月活跃账户数是多少",
        cats=["annual_reports"], gold=[("PDF-AR-TCEHY-2025", 4)],
        facts=[fact("腾讯", "2025", "微信及WeChat合并月活", "1418", "百万", ["1418", "1,418"])],
        forbidden=["1385", "1,385"])

    # unit_caliber：数字可以换写，但不得改掉原文单位
    add("ans-015", "unit_caliber", "宁德时代2024年营业收入是多少，请保留年报原文单位",
        cats=["annual_reports"], gold=[("PDF-AR-CATL-2024", 5)],
        facts=[fact("宁德时代", "2024", "营业收入", "362012554", "千元", ["362012554", "362,012,554", "千元"])],
        forbidden=["362012554亿元", "362,012,554亿元", "362012554万元"])
    add("ans-016", "unit_caliber", "龙芯中科2024年营业收入是多少，请保留年报原文单位",
        cats=["annual_reports"], gold=[("PDF-AR-688047-2024", 5)],
        facts=[fact("龙芯中科", "2024", "营业收入", "50425.72", "万元", ["50425.72", "50,425.72", "万元"])],
        forbidden=["50425.72元", "50,425.72元", "50425.72亿元"])
    add("ans-017", "unit_caliber", "腾讯2024年收入是多少，请保留年报原文单位",
        cats=["annual_reports"], gold=[("PDF-AR-TCEHY-2024", 5)],
        facts=[fact("腾讯", "2024", "收入", "660257", "百万元", ["660257", "660,257", "百万"])],
        forbidden=["660257亿元", "660,257亿元"])
    add("ans-018", "unit_caliber", "寒武纪2024年营业收入是多少，请保留年报原文单位",
        cats=["annual_reports"], gold=[("PDF-AR-688256-2024", 6)],
        facts=[fact("寒武纪", "2024", "营业收入", "1174464377.35", "元", ["1174464377.35", "1,174,464,377.35"]),
               ],
        forbidden=["1174464377.35万元", "1,174,464,377.35万元"])

    # multi_fact：多个事实都要出现，不能用其中一个替换另一个
    add("ans-019", "multi_fact", "腾讯2024年增值服务的收入和毛利率分别是多少",
        cats=["annual_reports"], gold=[("PDF-AR-TCEHY-2024", 7), ("PDF-AR-TCEHY-2024", 8)],
        facts=[
            fact("腾讯", "2024", "增值服务收入", "319168", "百万元", ["319168", "319,168"]),
            fact("腾讯", "2024", "增值服务毛利率", "57", "%", ["57%"]),
        ],
        forbidden=["121374", "121,374"])
    add("ans-020", "multi_fact", "CoreWeave 2026年一季度经调整EBITDA、利润率和净亏损分别是多少",
        cats=["research_reports"], gold=[("PDF-RR-20260531", 97)],
        facts=[
            fact("CoreWeave", "2026Q1", "经调整EBITDA", "12", "亿美元", ["12亿"]),
            fact("CoreWeave", "2026Q1", "经调整EBITDA利润率", "56", "%", ["56%"]),
            fact("CoreWeave", "2026Q1", "净亏损", "7.40", "亿美元", ["7.40", "7.4"]),
        ],
        forbidden=["净利润12亿", "盈利12亿"])
    add("ans-021", "multi_fact", "对比寒武纪2024年和2025年的营业收入及同比增速",
        cats=["annual_reports"], gold=[("PDF-AR-688256-2024", 6), ("PDF-AR-688256-2025", 6)],
        facts=[
            fact("寒武纪", "2024", "营业收入", "1174464377.35", "元", ["1174464377.35", "1,174,464,377.35"]),
            fact("寒武纪", "2024", "营业收入同比", "65.56", "%", ["65.56"]),
            fact("寒武纪", "2025", "营业收入", "6497196198.68", "元", ["6497196198.68", "6,497,196,198.68"]),
            fact("寒武纪", "2025", "营业收入同比", "453.21", "%", ["453.21"]),
        ],
        forbidden=[])
    add("ans-022", "multi_fact", "龙芯中科2025年营业收入增长后，扣股份支付后的净利润是多少，是否已经盈利",
        cats=["annual_reports"], gold=[("PDF-AR-688047-2025", 6), ("PDF-AR-688047-2025", 14)],
        facts=[
            fact("龙芯中科", "2025", "营业收入", "63532.06", "万元", ["63532.06", "63,532.06"]),
            fact("龙芯中科", "2025", "扣股份支付后净利润", "-42608.11", "万元", ["-42608.11", "-42,608.11"]),
        ],
        forbidden=["实现盈利", "已经扭亏"])
    add("ans-023", "multi_fact", "寒武纪2025年营业收入、费用化研发投入和扣股份支付后净利润分别是多少",
        cats=["annual_reports"], gold=[("PDF-AR-688256-2025", 6), ("PDF-AR-688256-2025", 12), ("PDF-AR-688256-2025", 22)],
        facts=[
            fact("寒武纪", "2025", "营业收入", "6497196198.68", "元", ["6497196198.68", "6,497,196,198.68"]),
            fact("寒武纪", "2025", "费用化研发投入", "1169100962.15", "元", ["1169100962.15", "1,169,100,962.15"]),
            fact("寒武纪", "2025", "扣股份支付后净利润", "2297212898.47", "元", ["2297212898.47", "2,297,212,898.47"]),
        ],
        forbidden=[])
    add("ans-024", "multi_fact", "中际旭创1.6T光模块于何时向重点客户出货",
        cats=["research_reports"], gold=[("PDF-RR-20260420", 39)],
        facts=[fact("中际旭创", "2025Q3", "1.6T出货", "2025年第三季度", "", ["2025年第三季度", "2025Q3"])],
        forbidden=["尚未出货", "没有出货"])

    # abstain：语料没有依据时必须拒答，不能补一个看起来像真的数
    add("ans-025", "abstain", "华为2024年净利润是多少",
        cats=["annual_reports"], gold=[],
        facts=[], forbidden=["亿元", "万元", "%"],
        abstain=True, notes="语料无华为年报")
    add("ans-026", "abstain", "比亚迪2025年动力电池装机量是多少",
        cats=["annual_reports"], gold=[],
        facts=[], forbidden=["GWh", "吉瓦时", "%"],
        abstain=True, notes="语料无比亚迪年报")
    add("ans-027", "abstain", "工信部第十一次中小企业圆桌会议是谁主持的",
        cats=["policy"], gold=[],
        facts=[], forbidden=[],
        abstain=True, notes="policy 未入库")
    add("ans-028", "abstain", "寒武纪2026年营业收入是多少",
        cats=["annual_reports"], gold=[],
        facts=[], forbidden=["1174464377.35", "6497196198.68", "65.56", "453.21"],
        abstain=True, notes="语料无2026年报，不得用2024或2025年营收充数")

    add("ans-029", "exact_number", "龙芯中科2024年第四季度营业收入是多少",
        cats=["annual_reports"], gold=[("PDF-AR-688047-2024", 7)],
        facts=[fact("龙芯中科", "2024Q4", "营业收入", "19647.66", "万元", ["19647.66", "19,647.66"])],
        forbidden=["第四季度营业收入50,425.72", "第四季度营业收入50425.72"])
    add("ans-030", "exact_number", "宁德时代2024年第一季度营业收入是多少",
        cats=["annual_reports"], gold=[("PDF-AR-CATL-2024", 7)],
        facts=[fact("宁德时代", "2024Q1", "营业收入", "79770779", "千元", ["79770779", "79,770,779"])],
        forbidden=["第一季度营业收入362,012,554", "第一季度营业收入362012554"])
    add("ans-031", "exact_number", "2026年3月末M2和社会融资规模存量同比分别是多少",
        cats=["macro_research"], gold=[("PDF-MACRO-03", 2)],
        facts=[
            fact("中国", "2026-03", "M2同比", "8.5", "%", ["8.5"]),
            fact("中国", "2026-03", "社融存量同比", "7.9", "%", ["7.9"]),
        ],
        forbidden=[])
    add("ans-032", "exact_number", "光模块在光通信系统设备中的成本占比大约是多少",
        cats=["research_reports"], gold=[("PDF-RR-20260420", 1)],
        facts=[fact("光模块", "", "成本占比", "50", "%", ["50%", "超过50"])],
        forbidden=["电池"])
    add("ans-033", "exact_number", "时尚春熙平台累计为多少银发用户提供服务",
        cats=["industry_whitepapers"], gold=[("PDF-WP-07", 3)],
        facts=[fact("春熙", "", "银发用户", "7.8万", "", ["7.8万"])],
        forbidden=[])
    add("ans-034", "exact_number", "数字治理指数表中北京2024年的数值是多少",
        cats=["industry_whitepapers"], gold=[("PDF-WP-02", 38)],
        facts=[fact("北京", "2024", "数字治理指数", "12.05", "", ["12.05"])],
        forbidden=[])

    add("ans-035", "entity_year", "龙芯中科2024年归母净利润是多少",
        cats=["annual_reports"], gold=[("PDF-AR-688047-2024", 5)],
        facts=[fact("龙芯中科", "2024", "归母净利润", "-62534.71", "万元", ["-62534.71", "-62,534.71"])],
        forbidden=["-452338791.01", "-452,338,791.01", "-42608.11", "-42,608.11"])
    add("ans-036", "entity_year", "宁德时代2025年营业收入和同比增速是多少",
        cats=["annual_reports"], gold=[("PDF-AR-CATL-2025", 19)],
        facts=[
            fact("宁德时代", "2025", "营业收入", "423701834", "", ["423701834", "423,701,834"]),
            fact("宁德时代", "2025", "营业收入同比", "17.04", "%", ["17.04"]),
        ],
        forbidden=["362012554", "362,012,554", "-9.70", "-9.7"])
    add("ans-037", "entity_year", "龙芯中科的法定代表人是谁",
        cats=["annual_reports"], gold=[("PDF-AR-688047-2024", 0)],
        facts=[fact("龙芯中科", "", "法定代表人", "胡伟武", "", ["胡伟武"])],
        forbidden=["寒武纪"])
    add("ans-038", "entity_year", "寒武纪2024年年报中云端产品线营业收入是多少",
        cats=["annual_reports"], gold=[("PDF-AR-688256-2024", 288)],
        facts=[fact("寒武纪", "2024", "云端产品线营业收入", "1166278485.36", "元", ["1166278485.36", "1,166,278,485.36"])],
        forbidden=["49%", "思元590"])

    add("ans-039", "unit_caliber", "2021年人身险公司保费收入是多少，请保留原文单位",
        cats=["macro_research"], gold=[("PDF-MACRO-02", 94)],
        facts=[fact("人身险", "2021", "保费收入", "31224", "亿元", ["31224", "31,224", "亿元"])],
        forbidden=["31224万元", "31,224万元", "31224元"])
    add("ans-040", "unit_caliber", "CoreWeave 2026年资本开支指引是多少，请保留原文单位",
        cats=["research_reports"], gold=[("PDF-RR-20260531", 100)],
        facts=[
            fact("CoreWeave", "2026", "资本开支指引下限", "310", "亿美元", ["310"]),
            fact("CoreWeave", "2026", "资本开支指引上限", "350", "亿美元", ["350", "亿美元"]),
        ],
        forbidden=["310亿元", "350亿元"])
    add("ans-041", "unit_caliber", "寒武纪2024年云端产品线直接材料金额是多少，请保留原文单位",
        cats=["annual_reports"], gold=[("PDF-AR-688256-2024", 35)],
        facts=[fact("寒武纪", "2024", "云端产品线直接材料", "261563612.41", "元", ["261563612.41", "261,563,612.41"])],
        forbidden=["261563612.41万元", "261,563,612.41万元"])
    add("ans-042", "unit_caliber", "寒武纪2024年云端产品线营业收入是多少，请保留原文单位",
        cats=["annual_reports"], gold=[("PDF-AR-688256-2024", 288)],
        facts=[fact("寒武纪", "2024", "云端产品线营业收入", "1166278485.36", "元", ["1166278485.36", "1,166,278,485.36"])],
        forbidden=["1166278485.36万元", "1,166,278,485.36万元"])

    add("ans-043", "multi_fact", "龙芯中科2024年全年营收同比和第四季度净利润分别是多少",
        cats=["annual_reports"], gold=[("PDF-AR-688047-2024", 5), ("PDF-AR-688047-2024", 7)],
        facts=[
            fact("龙芯中科", "2024", "营业收入同比", "-0.28", "%", ["-0.28"]),
            fact("龙芯中科", "2024Q4", "净利润", "-28258.75", "万元", ["-28258.75", "-28,258.75"]),
        ],
        forbidden=[])
    add("ans-044", "multi_fact", "寒武纪2024年营业收入、归母净利润和基本每股收益分别是多少",
        cats=["annual_reports"], gold=[("PDF-AR-688256-2024", 6), ("PDF-AR-688256-2024", 8)],
        facts=[
            fact("寒武纪", "2024", "营业收入", "1174464377.35", "元", ["1174464377.35", "1,174,464,377.35"]),
            fact("寒武纪", "2024", "归母净利润", "-452338791.01", "元", ["-452338791.01", "-452,338,791.01"]),
            fact("寒武纪", "2024", "基本每股收益", "-1.09", "元", ["-1.09"]),
        ],
        forbidden=[])
    add("ans-045", "multi_fact", "CoreWeave 2026年一季度利息费用、一季度资本开支和全年资本开支指引分别是多少",
        cats=["research_reports"], gold=[("PDF-RR-20260531", 97), ("PDF-RR-20260531", 100)],
        facts=[
            fact("CoreWeave", "2026Q1", "利息费用", "5.36", "亿美元", ["5.36"]),
            fact("CoreWeave", "2026Q1", "资本开支", "68", "亿美元", ["68亿"]),
            fact("CoreWeave", "2026", "资本开支指引", "310-350", "亿美元", ["310", "350"]),
        ],
        forbidden=[])
    add("ans-046", "multi_fact", "宁德时代2024年电气机械及器材制造业占营业收入的比重和毛利率分别是多少",
        cats=["annual_reports"], gold=[("PDF-AR-CATL-2024", 14), ("PDF-AR-CATL-2024", 16)],
        facts=[
            fact("宁德时代", "2024", "电气机械收入占比", "98.48", "%", ["98.48"]),
            fact("宁德时代", "2024", "电气机械毛利率", "24.69", "%", ["24.69"]),
        ],
        forbidden=[])

    add("ans-047", "abstain", "腾讯2026年收入是多少",
        cats=["annual_reports"], gold=[],
        facts=[], forbidden=["660257", "660,257", "751766", "751,766"],
        abstain=True, notes="语料无2026年报")
    add("ans-048", "abstain", "龙芯中科2026年归母净利润是多少",
        cats=["annual_reports"], gold=[],
        facts=[], forbidden=["-62534.71", "-62,534.71", "-42608.11", "-42,608.11"],
        abstain=True, notes="语料无2026年报")
    add("ans-049", "abstain", "苹果公司2024年营业收入是多少",
        cats=["annual_reports"], gold=[],
        facts=[], forbidden=[],
        abstain=True, notes="语料无苹果年报")
    add("ans-050", "abstain", "宁德时代2026年动力电池装机量是多少",
        cats=["annual_reports"], gold=[],
        facts=[], forbidden=["GWh", "吉瓦时"],
        abstain=True, notes="语料无2026年装机量")

    assert len(rows) == 50, len(rows)
    return rows


def main() -> None:
    dump("recall.jsonl", recall_cases())
    dump("parent.jsonl", parent_cases())
    dump("hyde_stepback.jsonl", rewrite_cases())
    dump("answer_faithfulness.jsonl", answer_cases())


if __name__ == "__main__":
    main()
