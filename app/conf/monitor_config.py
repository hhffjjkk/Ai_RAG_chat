# 检索质量监控 + 反馈闭环 配置
# 说明：所有开关默认开启/合理默认，可通过 .env 覆盖；关闭 MONITOR_ENABLED 即退回改造前行为（便于回滚）
from dataclasses import dataclass
import os
from dotenv import load_dotenv

# 提前加载.env，保证 os.getenv 能读取到配置
load_dotenv()


def _to_bool(value, default=False) -> bool:
    """将 .env 中的字符串/数字转为布尔值，兼容 1/0、True/False、true/false"""
    if value is None:
        return default
    return str(value).strip().lower() in ("1", "true", "yes", "on")


def _to_int(value, default: int) -> int:
    """安全转换为整数，失败则返回默认值"""
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _to_float(value, default: float) -> float:
    """安全转换为浮点数，失败则返回默认值"""
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


@dataclass
class MonitorConfig:
    # ---- 总开关与存储 ----
    monitor_enabled: bool                 # 是否启用检索质量监控
    retrieval_logs_collection: str        # 检索日志集合名
    retrieval_feedback_collection: str    # 检索反馈集合名
    topk_limit: int                       # 单条日志记录的最大 TopK 数

    # ---- 正确性判定 ----
    llm_judge_enabled: bool               # 是否启用 LLM 自动判定答案正确性（默认关闭）

    # ---- 趋势与诊断阈值 ----
    window_days: int                      # 主题指标统计窗口（天）
    trend_recent_n: int                   # 趋势分析"近期"样本数
    trend_decline_threshold: float        # 近期准确率较历史下滑超过该值即视为下降
    low_accuracy_threshold: float         # 判定主题质量出问题的准确率阈值
    min_samples: int                      # 判定所需的最小反馈样本数
    dup_ratio_threshold: float            # 判定"重复过多"的重复切片占比阈值
    stale_ratio_threshold: float          # 判定"文档过时"的陈旧切片占比阈值
    min_avg_topk: float                   # 判定"覆盖不足"的平均 TopK 数量下限


monitor_config = MonitorConfig(
    monitor_enabled=_to_bool(os.getenv("MONITOR_ENABLED"), True),
    retrieval_logs_collection=(os.getenv("MONITOR_LOGS_COLLECTION") or "retrieval_logs").strip(),
    retrieval_feedback_collection=(os.getenv("MONITOR_FEEDBACK_COLLECTION") or "retrieval_feedback").strip(),
    topk_limit=_to_int(os.getenv("MONITOR_TOPK_LIMIT"), 10),

    llm_judge_enabled=_to_bool(os.getenv("MONITOR_LLM_JUDGE_ENABLED"), False),

    window_days=_to_int(os.getenv("MONITOR_WINDOW_DAYS"), 14),
    trend_recent_n=_to_int(os.getenv("MONITOR_TREND_RECENT_N"), 20),
    trend_decline_threshold=_to_float(os.getenv("MONITOR_TREND_DECLINE_THRESHOLD"), 0.15),
    low_accuracy_threshold=_to_float(os.getenv("MONITOR_LOW_ACCURACY_THRESHOLD"), 0.6),
    min_samples=_to_int(os.getenv("MONITOR_MIN_SAMPLES"), 3),
    dup_ratio_threshold=_to_float(os.getenv("MONITOR_DUP_RATIO_THRESHOLD"), 0.4),
    stale_ratio_threshold=_to_float(os.getenv("MONITOR_STALE_RATIO_THRESHOLD"), 0.5),
    min_avg_topk=_to_float(os.getenv("MONITOR_MIN_AVG_TOPK"), 2.0),
)
