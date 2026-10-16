# 知识库治理报告 CLI（dry-run）
#
# 作用：按时间窗聚合各主题的检索质量指标，输出"诊断结论 + 治理建议"到 Markdown 报告。
# 安全：本工具为只读 + 只写报告文件，绝不修改 Milvus / Mongo 的生产数据（无 --apply）。
#
# 用法：
#   python -m app.tool.kb_governance_report --days 14
#   python -m app.tool.kb_governance_report --days 7 --out docs/治理报告.md
import argparse
import os
import time
from datetime import datetime

from app.clients.mongo_monitor_utils import aggregate_topic_metrics
from app.utils.quality_metrics_utils import classify_diagnosis, build_recommendations
from app.conf.monitor_config import monitor_config
from app.utils.path_util import PROJECT_ROOT
from app.core.logger import logger


def build_report(days: int, metrics: list) -> str:
    """由主题指标生成 Markdown 报告文本（纯函数，便于测试）"""
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = [
        "# 知识库治理诊断报告（dry-run）",
        "",
        f"- 生成时间：{now_str}",
        f"- 统计窗口：近 {days} 天",
        f"- 主题数量：{len(metrics)}",
        f"- 数据来源：MongoDB `{monitor_config.retrieval_logs_collection}` + `{monitor_config.retrieval_feedback_collection}`",
        "",
        "> 说明：本报告仅输出诊断结论与治理建议，**不会自动修改生产数据**。",
        "> 执行任何删除/合并/下线操作前，需人工确认。",
        "",
        "## 一、主题汇总",
        "",
        "| 主题 | 检索次数 | 反馈样本 | 准确率 | 趋势 | 诊断 |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    rows = []
    topics_out = []
    for m in metrics:
        ratios = {
            "avg_topk": m.get("avg_topk", 0.0),
            "dup_ratio": m.get("dup_ratio", 0.0),
            "stale_ratio": m.get("stale_ratio", 0.0),
            "web_ratio": m.get("web_ratio", 0.0),
        }
        diagnosis = classify_diagnosis(m.get("sample_count", 0), m.get("accuracy", 0.0),
                                       m.get("trend", "insufficient"), ratios)
        recos = build_recommendations(diagnosis, m)
        rows.append(f"| {m.get('topic')} | {m.get('query_count', 0)} | {m.get('sample_count', 0)} "
                    f"| {round(m.get('accuracy', 0.0) * 100)}% | {m.get('trend')} | {diagnosis} |")
        topics_out.append((m, diagnosis, recos))

    lines.extend(rows or ["| （窗口内无数据） | - | - | - | - | - |"])
    lines += ["", "## 二、主题明细与治理建议", ""]

    for m, diagnosis, recos in topics_out:
        lines.append(f"### {m.get('topic')}（诊断：{diagnosis}）")
        lines.append("")
        lines.append(f"- 检索次数：{m.get('query_count', 0)}")
        lines.append(f"- 反馈样本：{m.get('sample_count', 0)}（正确 {m.get('correct', 0)} / 错误 {m.get('incorrect', 0)}），准确率 {round(m.get('accuracy', 0.0) * 100)}%")
        lines.append(f"- 趋势：{m.get('trend')}")
        lines.append(f"- 平均 TopK：{m.get('avg_topk')}；重复切片占比：{round(m.get('dup_ratio', 0.0) * 100)}%；"
                     f"陈旧切片占比：{round(m.get('stale_ratio', 0.0) * 100)}%；平均文档年龄：{m.get('avg_age_days')} 天")
        lines.append(f"- 来源部门：{'、'.join(m.get('source_depts') or []) or '-'}")
        lines.append("- 治理建议：")
        for r in recos:
            lines.append(f"  - {r}")
        lines.append("")

    lines += [
        "## 三、遗留与说明",
        "",
        "- 本报告为 **dry-run**，仅用于人工决策，不执行任何自动治理动作。",
        "- 反馈样本不足的主题会被标注为「数据不足」，需持续采集人工反馈后再评估。",
        "",
    ]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="知识库治理诊断报告（dry-run，只读+写报告）")
    parser.add_argument("--days", type=int, default=monitor_config.window_days, help="统计窗口（天）")
    parser.add_argument("--out", type=str, default="", help="报告输出路径（默认 docs/治理报告-<时间戳>.md）")
    args = parser.parse_args()

    since_ts = time.time() - max(1, args.days) * 86400
    metrics = aggregate_topic_metrics(since_ts=since_ts)

    report_text = build_report(args.days, metrics)

    if args.out:
        out_path = args.out if os.path.isabs(args.out) else os.path.join(PROJECT_ROOT, args.out)
    else:
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        out_path = os.path.join(PROJECT_ROOT, "docs", f"治理报告-{ts}.md")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(report_text)

    logger.info(f"治理报告已生成：{out_path}")
    print(f"治理报告已生成：{out_path}")
    print(f"统计主题数：{len(metrics)}（近 {args.days} 天）")


if __name__ == "__main__":
    main()
