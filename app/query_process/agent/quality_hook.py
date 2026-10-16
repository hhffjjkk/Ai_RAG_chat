"""
检索质量监控 Hook。

职责：在答案生成节点（node_answer_output）末尾，把本次检索的关键事实
（question / 主题 / TopK / 分数 / 时效与来源元信息 / 最终答案）落库到 retrieval_logs，
为"检索质量监控 + 反馈闭环"提供数据基础。

设计原则（对齐用户要求：不改变图拓扑、不影响主链路）：
  - 以"内联函数调用"方式接入，不改动 LangGraph 拓扑
  - MONITOR_ENABLED=false 直接跳过（便于回滚）
  - 任何异常都被吞掉，绝不阻断查询主流程
"""
from typing import Dict, Any, List
import time

from app.conf.monitor_config import monitor_config
from app.utils.quality_metrics_utils import summarize_topk
from app.core.logger import logger


def _collect_topk_facts(reranked_docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """把 reranked_docs 裁剪为监控所需的 TopK 事实（受 MONITOR_TOPK_LIMIT 限制）"""
    facts = []
    for doc in (reranked_docs or [])[:monitor_config.topk_limit]:
        facts.append({
            "chunk_id": str(doc.get("chunk_id", "") or ""),
            "doc_id": str(doc.get("doc_id", "") or ""),
            "title": doc.get("title", "") or "",
            "source": doc.get("source", "local") or "local",
            "score": float(doc.get("score", 0.0) or 0.0),
            "version": doc.get("version", 0) or 0,
            "publish_date": int(doc.get("publish_date", 0) or 0),
            "expire_date": int(doc.get("expire_date", 0) or 0),
            "timeliness_status": doc.get("timeliness_status", "") or "",
            "source_dept": doc.get("source_dept", "") or "",
        })
    return facts


def build_retrieval_log(state: Dict[str, Any]) -> Dict[str, Any]:
    """由查询 state 组装检索日志字典（纯组装，无副作用，便于测试）"""
    reranked_docs = state.get("reranked_docs", []) or []
    topk = _collect_topk_facts(reranked_docs)
    item_names = state.get("item_names", []) or []
    topic = item_names[0] if item_names else "未知主题"
    answer = state.get("answer", "") or ""
    start_ts = float(state.get("query_start_ts", 0) or 0)
    latency_ms = int((time.time() - start_ts) * 1000) if start_ts else int(state.get("latency_ms", 0) or 0)
    return {
        "session_id": state.get("session_id", "") or "",
        "query": state.get("original_query", "") or "",
        "rewritten_query": state.get("rewritten_query", "") or "",
        "item_names": item_names,
        "topic": topic,
        "answer": answer,
        "answer_len": len(answer),
        "topk": topk,
        "topk_count": len(topk),
        "sources": sorted({f["source"] for f in topk}),
        "ratios": summarize_topk(topk),
        "latency_ms": max(0, latency_ms),
        "llm_judge": None,
    }


def _llm_judge_answer(query: str, answer: str, topk: List[Dict[str, Any]]):
    """（可选）使用 LLM 自动判定答案正确性；默认关闭，失败返回 None"""
    if not monitor_config.llm_judge_enabled:
        return None
    try:
        from langchain_core.messages import HumanMessage
        from app.lm.lm_utils import get_llm_client

        context = "\n".join(f"- {d.get('title')}: {d.get('score')}" for d in topk[:5])
        prompt = (
            "你是检索质量评审员。请判断下面的回答是否准确回答了用户问题，"
            "只输出 JSON：{\"correct\": true/false}。\n"
            f"用户问题：{query}\n回答：{answer}\n参考资料：\n{context}"
        )
        response = get_llm_client(json_mode=True).invoke([HumanMessage(content=prompt)])
        import json
        text = (response.content or "").strip().replace("```json", "").replace("```", "")
        data = json.loads(text) if text else {}
        correct = data.get("correct")
        return bool(correct) if correct is not None else None
    except Exception as e:
        logger.warning(f"LLM 正确性判定失败，跳过自动反馈：{e}")
        return None


def record_query_quality(state: Dict[str, Any]) -> None:
    """
    记录一次检索的质量事实（内联调用，异常不抛出）。
    :param state: 查询图 state（含 session_id/item_names/reranked_docs/answer 等）
    """
    if not monitor_config.monitor_enabled:
        return
    try:
        log = build_retrieval_log(state)
        # 无有效检索内容（例如仅返回澄清提示）也记录，便于统计"覆盖不足"场景
        from app.clients.mongo_monitor_utils import save_retrieval_log, save_feedback

        log_id = save_retrieval_log(log)
        if not log_id:
            return
        # 可选：LLM 自动判定
        judged = _llm_judge_answer(log["query"], log["answer"], log["topk"])
        if judged is not None:
            save_feedback(log_id=log_id, correct=judged, comment="auto:llm_judge",
                          session_id=log["session_id"], source="llm")
            logger.info(f"已记录检索日志与 LLM 自动反馈：log_id={log_id}, correct={judged}")
        else:
            logger.info(f"已记录检索日志：log_id={log_id}, topic={log['topic']}, topk={log['topk_count']}")
    except Exception as e:
        # 监控失败绝不影响查询主流程
        logger.error(f"record_query_quality 记录失败（已忽略）：{e}")
