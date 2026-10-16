"""
检索质量度量工具（纯函数，便于单元测试与复用）。

职责：
  - 正确率统计：把人工/自动反馈的 True/False 汇总为准确率指标
  - 趋势检测：判断某主题的检索准确率是"下降/稳定/上升"
  - TopK 结构分析：重复切片占比、陈旧切片占比、平均数量、来源分布
  - 诊断分类：把上述指标映射为"文档过时/重复过多/覆盖不足/准确率下降/正常/数据不足"
  - 建议生成：给出可执行的治理建议（仅建议，不自动改生产数据）
本模块不访问数据库、不调用 LLM，全部为确定性纯函数。
"""
import time
from typing import List, Tuple, Dict, Any, Optional

from app.conf.monitor_config import monitor_config


def compute_accuracy_stats(correct_flags: List[Any], min_samples: Optional[int] = None) -> Dict[str, Any]:
    """
    统计准确率。
    :param correct_flags: 反馈正确性列表，元素为 True/False/None（None 视为未反馈，不计入）
    :param min_samples: 判定所需最小样本数，默认取配置
    :return: {total, correct, incorrect, accuracy, has_enough_samples}
    """
    if min_samples is None:
        min_samples = monitor_config.min_samples
    labeled = [bool(x) for x in (correct_flags or []) if x is not None]
    total = len(labeled)
    correct = sum(1 for x in labeled if x)
    incorrect = total - correct
    accuracy = round(correct / total, 4) if total else 0.0
    return {
        "total": total,
        "correct": correct,
        "incorrect": incorrect,
        "accuracy": accuracy,
        "has_enough_samples": total >= min_samples,
    }


def detect_accuracy_trend(points: List[Tuple[float, Any]],
                          recent_n: Optional[int] = None,
                          decline_threshold: Optional[float] = None) -> str:
    """
    检测准确率趋势。
    :param points: [(ts, correct_bool_or_None), ...]（顺序不限，内部会按 ts 升序）
    :param recent_n: "近期"窗口样本数，默认取配置
    :param decline_threshold: 下滑判定阈值，默认取配置
    :return: "declining"（下降） | "improving"（上升） | "stable"（稳定） | "insufficient"（样本不足）
    """
    if recent_n is None:
        recent_n = monitor_config.trend_recent_n
    if decline_threshold is None:
        decline_threshold = monitor_config.trend_decline_threshold

    labeled = [(float(t), bool(c)) for t, c in (points or []) if c is not None]
    labeled.sort(key=lambda x: x[0])
    if len(labeled) < 2:
        return "insufficient"

    n = max(1, int(recent_n))
    recent = labeled[-n:]
    # 历史窗口：优先取"近期窗口之前"的样本；不足时回退到前半段
    history = labeled[:-len(recent)] if len(labeled) > len(recent) else labeled[: len(labeled) // 2]
    if not history:
        return "insufficient"

    recent_acc = sum(1 for _, c in recent if c) / len(recent)
    history_acc = sum(1 for _, c in history if c) / len(history)
    if history_acc - recent_acc >= decline_threshold:
        return "declining"
    if recent_acc - history_acc >= decline_threshold:
        return "improving"
    return "stable"


def summarize_topk(topk: List[Dict[str, Any]], stale_warn_days: Optional[int] = None,
                   now_ts: Optional[int] = None) -> Dict[str, Any]:
    """
    分析单次检索 TopK 的结构特征。
    :param topk: TopK 文档列表，元素含 doc_id/title/chunk_id/source/publish_date/expire_date/
                 timeliness_status/source_dept
    :param stale_warn_days: 超过该天数视为陈旧（默认取时效配置）
    :param now_ts: 当前时间戳，默认取当前时间
    :return: 结构指标字典
    """
    if now_ts is None:
        now_ts = int(time.time())
    if stale_warn_days is None:
        from app.conf.timeliness_config import timeliness_config
        stale_warn_days = timeliness_config.timeliness_stale_warn_days

    topk = topk or []
    total = len(topk)
    unique_keys = set()
    local_count = web_count = stale_count = 0
    ages = []
    source_depts = set()
    for d in topk:
        key = d.get("doc_id") or d.get("title") or d.get("chunk_id")
        if key:
            unique_keys.add(key)
        if (d.get("source") or "local") == "web":
            web_count += 1
        else:
            local_count += 1
        publish_date = int(d.get("publish_date", 0) or 0)
        expire_date = int(d.get("expire_date", 0) or 0)
        status = str(d.get("timeliness_status", "") or "").lower()
        is_expired = bool(expire_date and 0 < expire_date <= now_ts)
        is_stale = status in ("stale", "expired") or (
            bool(publish_date) and (now_ts - publish_date) / 86400.0 >= stale_warn_days)
        if is_expired or is_stale:
            stale_count += 1
        if publish_date:
            ages.append((now_ts - publish_date) / 86400.0)
        dept = (d.get("source_dept") or "").strip()
        if dept:
            source_depts.add(dept)

    dup_count = total - len(unique_keys) if total else 0
    return {
        "count": total,
        "unique_docs": len(unique_keys),
        "dup_count": dup_count,
        "dup_ratio": round(dup_count / total, 4) if total else 0.0,
        "local_count": local_count,
        "web_count": web_count,
        "web_ratio": round(web_count / total, 4) if total else 0.0,
        "stale_count": stale_count,
        "stale_ratio": round(stale_count / total, 4) if total else 0.0,
        "avg_age_days": round(sum(ages) / len(ages), 2) if ages else 0.0,
        "source_depts": sorted(source_depts),
    }


def classify_diagnosis(sample_count: int, accuracy: float, trend: str,
                       ratios: Dict[str, Any], cfg=None) -> str:
    """
    根据指标诊断某主题的质量问题类型。
    :return: "数据不足" | "文档过时" | "重复过多" | "覆盖不足" | "准确率下降" | "质量待排查" | "正常"
    """
    cfg = cfg or monitor_config
    if sample_count < cfg.min_samples:
        return "数据不足"
    if accuracy >= cfg.low_accuracy_threshold and trend != "declining":
        return "正常"
    # 质量出问题：定位主因（按影响面从大到小）
    if ratios.get("stale_ratio", 0.0) >= cfg.stale_ratio_threshold:
        return "文档过时"
    if ratios.get("dup_ratio", 0.0) >= cfg.dup_ratio_threshold:
        return "重复过多"
    if ratios.get("avg_topk", 0.0) < cfg.min_avg_topk or ratios.get("web_ratio", 0.0) >= 0.5:
        return "覆盖不足"
    if trend == "declining":
        return "准确率下降"
    return "质量待排查"


def build_recommendations(diagnosis: str, metrics: Dict[str, Any]) -> List[str]:
    """根据诊断结论生成治理建议（dry-run：仅建议，不自动修改生产数据）"""
    tips = {
        "文档过时": [
            "核查该主题文档时效性，补充/替换最新版本并重设失效日期",
            "对已过期切片执行下线（需人工确认后执行）",
        ],
        "重复过多": [
            "合并重复切片，按 doc_id 归一历史版本，仅保留最新版",
            "检查是否存在同一内容多次入库 / 同主题多文档高度重叠",
        ],
        "覆盖不足": [
            "补充该主题的新文档，扩大知识覆盖",
            "检查 item_name 归一是否导致召回受限（可放宽 item_name 过滤）",
        ],
        "准确率下降": [
            "抽查近期错误答案对应的 TopK 文档，确认内容是否过时或答非所问",
            "回溯该主题近期新入库文档，排查质量回退",
        ],
        "质量待排查": ["人工复核该主题 TopK 文档与错误反馈明细"],
        "数据不足": ["继续采集人工反馈，样本充足后再评估"],
        "正常": ["暂无需处理"],
    }
    recos = list(tips.get(diagnosis, ["人工复核"]))
    # 追加指标快照，便于人工判断
    if metrics.get("stale_count"):
        recos.append(f"当前陈旧切片数：{metrics.get('stale_count')}，平均文档年龄：{metrics.get('avg_age_days')} 天")
    if metrics.get("dup_count"):
        recos.append(f"当前重复切片数：{metrics.get('dup_count')}")
    return recos
