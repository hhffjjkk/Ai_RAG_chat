"""
检索质量监控 MongoDB 工具类。

职责：
  - 存储每次检索的关键事实（TopK、分数、来源、时效/版本元信息）到 retrieval_logs
  - 存储人工/自动正确性反馈到 retrieval_feedback（一条日志一条反馈，可更新）
  - 提供按时间窗聚合的"主题级"质量指标，供诊断与治理报告使用
设计原则：
  - 监控写入失败绝不影响主查询链路（上层用 try/except 兜底）
  - 只读/只写监控集合，不触碰 chat_message / doc_registry 等业务集合
"""
import os
import time
import uuid
from typing import List, Dict, Any, Optional

from pymongo import MongoClient, ASCENDING, DESCENDING
from dotenv import load_dotenv

from app.conf.monitor_config import monitor_config
from app.utils.quality_metrics_utils import compute_accuracy_stats, detect_accuracy_trend
from app.core.logger import logger

load_dotenv()


class MonitorMongoTool:
    """retrieval_logs / retrieval_feedback 集合的连接与索引初始化"""

    def __init__(self):
        try:
            self.mongo_url = os.getenv("MONGO_URL")
            self.db_name = os.getenv("MONGO_DB_NAME")
            self.client = MongoClient(self.mongo_url)
            self.db = self.client[self.db_name]

            self.retrieval_logs = self.db[monitor_config.retrieval_logs_collection]
            self.retrieval_feedback = self.db[monitor_config.retrieval_feedback_collection]

            # 索引：create_index 自带幂等性
            self.retrieval_logs.create_index([("log_id", ASCENDING)], unique=True)
            self.retrieval_logs.create_index([("created_at", DESCENDING)])
            self.retrieval_logs.create_index([("session_id", ASCENDING), ("created_at", DESCENDING)])
            self.retrieval_logs.create_index([("topic", ASCENDING), ("created_at", DESCENDING)])
            self.retrieval_feedback.create_index([("log_id", ASCENDING)], unique=True)
            self.retrieval_feedback.create_index([("created_at", DESCENDING)])

            logger.info(f"MonitorMongoTool 已连接 MongoDB：{self.db_name}")
        except Exception as e:
            logger.error(f"MonitorMongoTool 初始化失败：{e}")
            raise


_monitor_tool: Optional[MonitorMongoTool] = None

try:
    _monitor_tool = MonitorMongoTool()
except Exception as e:
    # 模块加载阶段失败只告警，保留懒加载兜底
    logger.warning(f"MonitorMongoTool 模块加载初始化失败：{e}")


def get_monitor_tool() -> MonitorMongoTool:
    """获取单例（懒加载）"""
    global _monitor_tool
    if _monitor_tool is None:
        _monitor_tool = MonitorMongoTool()
    return _monitor_tool


def save_retrieval_log(log: Dict[str, Any]) -> Optional[str]:
    """
    写入一条检索日志。
    :param log: 日志字典（由 quality_hook.record_query_quality 组装）
    :return: log_id；失败返回 None
    """
    try:
        tool = get_monitor_tool()
        log_id = log.get("log_id") or str(uuid.uuid4())
        doc = dict(log)
        doc["log_id"] = log_id
        doc.setdefault("created_at", time.time())
        tool.retrieval_logs.update_one({"log_id": log_id}, {"$set": doc}, upsert=True)
        return log_id
    except Exception as e:
        logger.error(f"写入 retrieval_logs 失败：{e}")
        return None


def save_feedback(log_id: str, correct: bool, comment: str = "",
                  session_id: str = "",
                  source: str = "manual") -> bool:
    """
    写入/更新一条检索反馈（log_id 唯一）。
    :param correct: 答案是否正确（人工反馈为主）
    :param source: 反馈来源 manual（人工） | llm（自动判定）
    :return: 是否写入成功
    """
    try:
        tool = get_monitor_tool()
        tool.retrieval_feedback.update_one(
            {"log_id": log_id},
            {"$set": {
                "log_id": log_id,
                "session_id": session_id,
                "correct": bool(correct),
                "comment": comment or "",
                "source": source or "manual",
                "created_at": time.time(),
            }},
            upsert=True,
        )
        return True
    except Exception as e:
        logger.error(f"写入 retrieval_feedback 失败：log_id={log_id}，{e}")
        return False


def get_recent_logs(limit: int = 200, since_ts: float = 0) -> List[Dict[str, Any]]:
    """按时间倒序获取最近的检索日志"""
    try:
        tool = get_monitor_tool()
        query: Dict[str, Any] = {}
        if since_ts:
            query["created_at"] = {"$gte": since_ts}
        cursor = tool.retrieval_logs.find(query).sort("created_at", DESCENDING).limit(int(limit))
        return list(cursor)
    except Exception as e:
        logger.error(f"查询 retrieval_logs 失败：{e}")
        return []


def get_log(log_id: str) -> Optional[Dict[str, Any]]:
    """按 log_id 查询单条检索日志"""
    try:
        tool = get_monitor_tool()
        return tool.retrieval_logs.find_one({"log_id": log_id})
    except Exception as e:
        logger.error(f"查询单条 retrieval_logs 失败：log_id={log_id}，{e}")
        return None


def get_logs_by_session(session_id: str, limit: int = 50) -> List[Dict[str, Any]]:
    """按 session_id 查询检索日志"""
    try:
        tool = get_monitor_tool()
        cursor = tool.retrieval_logs.find({"session_id": session_id}).sort("created_at", DESCENDING).limit(int(limit))
        return list(cursor)
    except Exception as e:
        logger.error(f"按会话查询 retrieval_logs 失败：session_id={session_id}，{e}")
        return []


def get_feedback_by_log(log_id: str) -> Optional[Dict[str, Any]]:
    """查询某条日志的反馈"""
    try:
        tool = get_monitor_tool()
        return tool.retrieval_feedback.find_one({"log_id": log_id})
    except Exception as e:
        logger.error(f"查询 retrieval_feedback 失败：log_id={log_id}，{e}")
        return None


def get_recent_feedbacks(since_ts: float = 0) -> List[Dict[str, Any]]:
    """获取时间窗内的全部反馈"""
    try:
        tool = get_monitor_tool()
        query: Dict[str, Any] = {}
        if since_ts:
            query["created_at"] = {"$gte": since_ts}
        return list(tool.retrieval_feedback.find(query))
    except Exception as e:
        logger.error(f"查询 retrieval_feedback 失败：{e}")
        return []


def aggregate_topic_metrics(since_ts: float = 0) -> List[Dict[str, Any]]:
    """
    按主题（topic = 主 item_name）聚合质量指标。
    数据来源：窗口内的 retrieval_logs（含 ratios 快照） + retrieval_feedback。
    :param since_ts: 统计窗口起点（epoch 秒），0 表示不限制
    :return: 主题指标列表（按样本数倒序）
    """
    logs = get_recent_logs(limit=10000, since_ts=since_ts)
    feedbacks = get_recent_feedbacks(since_ts=since_ts)
    feedback_map: Dict[str, Dict[str, Any]] = {f.get("log_id"): f for f in feedbacks if f.get("log_id")}

    topics: Dict[str, Dict[str, Any]] = {}
    for log in logs:
        topic = log.get("topic") or (log.get("item_names") or ["未知主题"])[0] or "未知主题"
        bucket = topics.setdefault(topic, {
            "topic": topic,
            "query_count": 0,
            "dup_ratio_sum": 0.0, "stale_ratio_sum": 0.0, "web_ratio_sum": 0.0,
            "topk_sum": 0, "age_sum": 0.0,
            "feedback_points": [],   # [(ts, correct)]
            "source_depts": set(),
            "log_ids": [],
        })
        bucket["query_count"] += 1
        ratios = log.get("ratios") or {}
        bucket["dup_ratio_sum"] += float(ratios.get("dup_ratio", 0.0) or 0.0)
        bucket["stale_ratio_sum"] += float(ratios.get("stale_ratio", 0.0) or 0.0)
        bucket["web_ratio_sum"] += float(ratios.get("web_ratio", 0.0) or 0.0)
        bucket["topk_sum"] += int(log.get("topk_count", 0) or 0)
        bucket["age_sum"] += float(ratios.get("avg_age_days", 0.0) or 0.0)
        for dept in (ratios.get("source_depts") or []):
            bucket["source_depts"].add(dept)
        bucket["log_ids"].append(log.get("log_id"))

        fb = feedback_map.get(log.get("log_id"))
        if fb is not None:
            bucket["feedback_points"].append((fb.get("created_at", 0), fb.get("correct")))

    # 组装为可诊断的指标结构
    results: List[Dict[str, Any]] = []
    for topic, bucket in topics.items():
        n = max(1, bucket["query_count"])
        correct_flags = [c for _, c in bucket["feedback_points"]]
        acc_stats = compute_accuracy_stats(correct_flags)
        trend = detect_accuracy_trend(bucket["feedback_points"])
        results.append({
            "topic": topic,
            "query_count": bucket["query_count"],
            "avg_topk": round(bucket["topk_sum"] / n, 3),
            "dup_ratio": round(bucket["dup_ratio_sum"] / n, 4),
            "stale_ratio": round(bucket["stale_ratio_sum"] / n, 4),
            "web_ratio": round(bucket["web_ratio_sum"] / n, 4),
            "avg_age_days": round(bucket["age_sum"] / n, 2),
            "source_depts": sorted(bucket["source_depts"]),
            "sample_count": acc_stats["total"],
            "correct": acc_stats["correct"],
            "incorrect": acc_stats["incorrect"],
            "accuracy": acc_stats["accuracy"],
            "has_enough_samples": acc_stats["has_enough_samples"],
            "trend": trend,
            "sample_log_ids": bucket["log_ids"][:20],
        })
    results.sort(key=lambda x: (x["sample_count"], x["query_count"]), reverse=True)
    return results
