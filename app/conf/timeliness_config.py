# 文档版本控制 + 文档时效性标记 配置
# 说明：所有开关默认开启，可通过 .env 覆盖；关闭开关即退回改造前行为（便于回滚）
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
class TimelinessConfig:
    # ---- 版本控制 ----
    version_enabled: bool            # 是否启用版本控制
    doc_identity_strategy: str       # doc_id 身份策略 auto | form | file_title | item_name
    duplicate_policy: str            # 内容未变时的处理策略 skip(幂等跳过) | refresh(仅刷新时间)
    keep_all_versions_in_milvus: bool  # false=仅保留最新版(方案A) true=全版本(方案B)
    archive_history_versions: bool   # 是否归档历史版本切片到本地 chunks 目录

    # ---- 时效性 ----
    timeliness_enabled: bool         # 是否启用时效性标记
    timeliness_half_life_days: float  # 时效衰减半衰期（天）
    timeliness_min_factor: float     # 时效因子下限（防止过度降权）
    timeliness_expired_policy: str   # exclude(硬过滤) | soft(软降权+提示)
    timeliness_effective_filter_enabled: bool  # 是否启用"生效下界"过滤（publish_date<=now）
    timeliness_stale_warn_days: int  # 超过该天数视为陈旧，答案中提示
    timeliness_warn_in_answer: bool  # 是否在答案中追加时效提示
    timeliness_extract_llm_enabled: bool  # 是否启用 LLM 辅助提取文档日期


timeliness_config = TimelinessConfig(
    version_enabled=_to_bool(os.getenv("VERSION_ENABLED"), True),
    doc_identity_strategy=(os.getenv("DOC_IDENTITY_STRATEGY") or "auto").strip().lower(),
    duplicate_policy=(os.getenv("DUPLICATE_POLICY") or "skip").strip().lower(),
    keep_all_versions_in_milvus=_to_bool(os.getenv("KEEP_ALL_VERSIONS_IN_MILVUS"), False),
    archive_history_versions=_to_bool(os.getenv("ARCHIVE_HISTORY_VERSIONS"), True),

    timeliness_enabled=_to_bool(os.getenv("TIMELINESS_ENABLED"), True),
    timeliness_half_life_days=_to_float(os.getenv("TIMELINESS_HALF_LIFE_DAYS"), 365.0),
    timeliness_min_factor=_to_float(os.getenv("TIMELINESS_MIN_FACTOR"), 0.6),
    timeliness_expired_policy=(os.getenv("TIMELINESS_EXPIRED_POLICY") or "exclude").strip().lower(),
    timeliness_effective_filter_enabled=_to_bool(os.getenv("EFFECTIVE_DATE_FILTER_ENABLED"), True),
    timeliness_stale_warn_days=_to_int(os.getenv("TIMELINESS_STALE_WARN_DAYS"), 730),
    timeliness_warn_in_answer=_to_bool(os.getenv("TIMELINESS_WARN_IN_ANSWER"), True),
    timeliness_extract_llm_enabled=_to_bool(os.getenv("TIMELINESS_EXTRACT_LLM_ENABLED"), False),
)
