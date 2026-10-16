"""
文档元信息工具：doc_id 归一、内容指纹、日期解析、时效状态与衰减因子计算。
导入侧（node_doc_version / node_timeliness）与查询侧（node_search_embedding / node_rerank）
共用本模块，保证口径一致。
"""
import hashlib
import re
import time
from datetime import datetime, date
from typing import Optional

# 常见日期格式（正则 -> 解析）
# 支持：2024-05-01、2024/05/01、2024.05.01、2024年5月1日、20240501
_DATE_PATTERNS = [
    re.compile(r"(?P<y>20\d{2})[-/.年](?P<m>\d{1,2})[-/.月](?P<d>\d{1,2})日?"),
    re.compile(r"(?P<y>20\d{2})(?P<m>\d{2})(?P<d>\d{2})"),
]


def now_epoch() -> int:
    """当前时间的 epoch 秒（服务端统一口径，便于过期判断）"""
    return int(time.time())


def normalize_doc_id(file_title: str, form_doc_id: Optional[str] = None,
                    strategy: str = "auto") -> str:
    """
    归一化文档逻辑标识 doc_id（跨版本不变）。
    优先级：上传表单指定 > 规范化 file_title > 兜底 unknown。
    :param file_title: 文件名去后缀
    :param form_doc_id: 上传表单显式指定的 doc_id（可为空）
    :param strategy: auto|form|file_title|item_name
    """
    strategy = (strategy or "auto").strip().lower()
    if strategy == "form":
        return _clean(form_doc_id) or _clean(file_title) or "unknown"
    # auto / file_title / item_name 均优先使用表单值（若提供），否则用 file_title
    if form_doc_id and str(form_doc_id).strip():
        return _clean(form_doc_id)
    return _clean(file_title) or "unknown"


def _clean(value: Optional[str]) -> str:
    """去掉首尾空白，压缩内部空白，作为稳定的逻辑标识"""
    if value is None:
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def compute_content_hash(text: str) -> str:
    """计算文档正文的 MD5 指纹，用于识别"内容未变"的重复上传"""
    if text is None:
        text = ""
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def parse_date_to_epoch(value) -> Optional[int]:
    """
    将多种日期表示解析为 epoch 秒。
    支持：int/float(epoch 秒)、'2024-05-01'、'2024/05/01'、'2024.05.01'、'2024年5月1日'。
    解析失败返回 None。
    """
    if value is None:
        return None
    # 数字：视为 epoch 秒（>0 才有意义）
    if isinstance(value, (int, float)):
        v = int(value)
        return v if v > 0 else None
    text = str(value).strip()
    if not text:
        return None
    # 纯数字字符串：10 位 epoch 秒 或 8 位日期
    if text.isdigit():
        if len(text) >= 10:
            return int(text)
        if len(text) == 8:
            try:
                return int(datetime.strptime(text, "%Y%m%d").timestamp())
            except ValueError:
                return None
    # 按常见格式解析
    for pattern in _DATE_PATTERNS:
        m = pattern.search(text)
        if not m:
            continue
        try:
            dt = date(int(m.group("y")), int(m.group("m")), int(m.group("d")))
            return int(datetime(dt.year, dt.month, dt.day).timestamp())
        except ValueError:
            continue
    return None


def epoch_to_date_str(epoch, default: str = "未知日期") -> str:
    """epoch 秒 -> 'YYYY-MM-DD'（用于答案上下文展示）；0/空 返回默认"""
    if not epoch:
        return default
    try:
        return datetime.fromtimestamp(int(epoch)).strftime("%Y-%m-%d")
    except (ValueError, OSError, OverflowError):
        return default


def extract_date_from_text(md_content: str) -> Optional[int]:
    """
    从正文正则提取日期（辅助手段）：取文本中第一个匹配的日期。
    仅作为表单未填写时的辅助来源。
    """
    if not md_content:
        return None
    for pattern in _DATE_PATTERNS:
        m = pattern.search(md_content)
        if m:
            try:
                dt = date(int(m.group("y")), int(m.group("m")), int(m.group("d")))
                return int(datetime(dt.year, dt.month, dt.day).timestamp())
            except ValueError:
                continue
    return None


def compute_timeliness_status(publish_date: int, expire_date: int, now_ts: int,
                              stale_warn_days: int = 730) -> str:
    """
    计算时效状态：
      expired：已过失效日期（expire_date>0 且 <= now）
      stale  ：未过期但发布日期距今超过 stale_warn_days
      aging  ：未过期，且发布距今超过半衰期警告（> stale_warn_days/2）
      fresh  ：较新内容
    """
    if expire_date and expire_date > 0 and expire_date <= now_ts:
        return "expired"
    if not publish_date:
        return "fresh"
    age_days = (now_ts - publish_date) / 86400.0
    if age_days >= stale_warn_days:
        return "stale"
    if age_days >= stale_warn_days / 2.0:
        return "aging"
    return "fresh"


def compute_timeliness_factor(publish_date: int, now_ts: int,
                              half_life_days: float = 365.0,
                              min_factor: float = 0.6) -> float:
    """
    时效衰减因子：factor = clamp(0.5 ** (age_days / half_life_days), min_factor, 1.0)
    用于在重排阶段对陈旧（但未过期）内容做软降权。
    """
    if not publish_date:
        return 1.0
    age_days = max(0.0, (now_ts - publish_date) / 86400.0)
    if half_life_days <= 0:
        return 1.0
    factor = 0.5 ** (age_days / half_life_days)
    if factor < min_factor:
        factor = min_factor
    if factor > 1.0:
        factor = 1.0
    return factor
