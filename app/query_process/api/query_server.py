# 6个接口  健康状态 返回页面  发起提问   sse长连接  查看历史对话  清空历史对话
from pathlib import Path
import time
import uuid
import uvicorn
from fastapi import FastAPI, BackgroundTasks, HTTPException, Request
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field
from starlette.middleware.cors import CORSMiddleware
from app.core.logger import logger
from app.query_process.agent.state import create_query_default_state
from app.utils.path_util import PROJECT_ROOT

from app.utils.task_utils import *
from app.utils.sse_utils import create_sse_queue, SSEEvent, sse_generator
from app.clients.mongo_history_utils import *
from app.clients.mongo_monitor_utils import (
    get_recent_logs,
    get_logs_by_session,
    get_feedback_by_log,
    get_recent_feedbacks,
    save_feedback,
    aggregate_topic_metrics,
)
from app.utils.quality_metrics_utils import classify_diagnosis, build_recommendations
from app.conf.monitor_config import monitor_config
from app.query_process.agent.main_graph import query_app


# 定义fastapi对象
app = FastAPI(title="query service", description="智汇智库查询服务！")

# 跨域配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 健康状态
@app.get("/health")
async def health():
    logger.info(f"触发后台检测检查接口，数据一切正常！！")
    return {"status": "ok"}

# 返回chat.html
@app.get("/chat.html")
async def chat_html():
    # 查找chat.html页面的地址
    chat_html_path  = PROJECT_ROOT / 'app' / 'query_process' / 'page' / 'chat.html'
    if not chat_html_path.exists():
        raise HTTPException(status_code=404, detail="chat.html文件不存在")
    return FileResponse(chat_html_path)

# 返回查询监控页面（检索质量监控 + 反馈闭环）
@app.get("/monitor.html")
async def monitor_html():
    monitor_html_path = PROJECT_ROOT / 'app' / 'query_process' / 'page' / 'query_monitor.html'
    if not monitor_html_path.exists():
        raise HTTPException(status_code=404, detail="query_monitor.html文件不存在")
    return FileResponse(monitor_html_path)

# 发起提问接口
# 接收参数的类型
class QueryRequest(BaseModel):
    query: str = Field(..., title="查询内容,必须传递")
    session_id:str = Field(None, title="会话id，可以不传递，后台uuid生成第一个！")
    is_stream:bool = Field(False, title="是否流式返回结果")

def run_query_graph(query: str, session_id: str, is_stream: bool):
    # 一会回调用 main_graph执行
    # 本次任务开启了！ is_stream = True 把结果加入到队列，sse可以取到
    update_task_status(session_id, "processing", is_stream)
    state = create_query_default_state(
        session_id=session_id,
        original_query=query,
        is_stream=is_stream
    )
    # 记录查询开始时间，供质量监控统计端到端延迟
    state["query_start_ts"] = time.time()
    try:
        query_app.invoke(state)
        update_task_status(session_id, "completed", is_stream)
    except Exception as e:
        update_task_status(session_id, "failed", is_stream)
        logger.exception(f"---session_id = {session_id},查询流程出现异常！！{str(e)}")
        push_to_session(session_id,SSEEvent.ERROR,data={"error":str(e)})


@app.post("/query")  # 客户端 -》 问题 -》 graph开启了 -》 查到rag的结果 -》 返回即可！！
async def query(request: QueryRequest,background_tasks: BackgroundTasks):
    """
    :param request: 请求参数
    :param background_tasks: 异步执行函数  is_stream = True
    :return:
    """
    query = request.query
    session_id = request.session_id or str(uuid.uuid4())
    is_stream = request.is_stream
    # 判断是不是流式处理 （异步 -》 先返回一个结果 开始处理 | 后台运行图，结果向前端推送）
    if is_stream:
        # 只要开启流式处理，我们业务中就是将数据，插入到队列中！ {session_id , queue [update_task_state , add_running_task,add_done_list]}
        # 创建当前session_id对应的队列 =》 _session_stream
        create_sse_queue(session_id)
        update_task_status(session_id,"processing",is_stream)
        #异步执行
        background_tasks.add_task(run_query_graph, query, session_id, is_stream)
        logger.info(f"query:{query}已经开启了异步和流式处理！！")
        return {
            "session_id": session_id,
            "message": "本次查询处理中，请稍候..."
        }
    else:
        run_query_graph(query, session_id, is_stream)
        answer = get_task_result(session_id,"answer")
        logger.info(f"query:{query}开启同步处理！处理结果为：{answer}!")
        return {
            "answer": answer,
            "session_id": session_id,
            "message": "本次查询处理完毕！",
            "done_list": []
        }

@app.get("/stream/{session_id}")
async def stream(session_id: str,request:Request):
    """
    :param session_id:
    :param request: 前端的原生请求对象，可以判断是否断开连接！！
    :return:
    """
    logger.info(f"session_id = {session_id}客户端，已经和后台建立了长连接！")
    return StreamingResponse(
        sse_generator(session_id,request),
        media_type="text/event-stream"
    )

@app.get("/history/{session_id}")
async def history(session_id: str,limit: int = 10):
    chats = get_recent_messages(session_id, limit)
    logger.info(f"session_id = {session_id}查询历史记录！")
    return {
        "session_id": session_id,
        "items": chats
    }

@app.delete("/history/{session_id}")
async def delete_history(session_id: str):
    delete_count = clear_history(session_id)
    logger.info(f"session_id = {session_id}清空历史记录！删除{delete_count}条记录")
    return {
        "deleted_count": delete_count,
        "message": f"{session_id}历史记录已清空！"
    }


# ===================== 检索质量监控 + 反馈闭环 接口 =====================

class FeedbackRequest(BaseModel):
    log_id: str = Field(..., title="检索日志ID（来自监控列表）")
    correct: bool = Field(..., title="该次检索的答案是否正确")
    comment: str = Field("", title="反馈备注（可选）")


def _feedback_map(logs):
    """按 log_id 建立反馈映射"""
    ids = [x.get("log_id") for x in logs if x.get("log_id")]
    mapping = {}
    for log_id in ids:
        fb = get_feedback_by_log(log_id)
        if fb:
            mapping[log_id] = fb
    return mapping


# 注意：必须声明在 /query/monitor/{session_id} 之前，否则 "recent" 会被当作 session_id
@app.get("/query/monitor/recent")
async def monitor_recent(limit: int = 200):
    """
    最近检索日志列表 + 汇总指标（供 query_monitor.html 使用）。
    """
    logs = get_recent_logs(limit=limit)
    fb_map = _feedback_map(logs)
    items = []
    correct_cnt = 0
    labeled_cnt = 0
    today_start = time.time() - (time.time() % 86400)
    today_cnt = 0
    topics = set()
    topk_sum = 0
    for log in logs:
        fb = fb_map.get(log.get("log_id"))
        correct = fb.get("correct") if fb else None
        if correct is not None:
            labeled_cnt += 1
            correct_cnt += 1 if correct else 0
        if (log.get("created_at") or 0) >= today_start:
            today_cnt += 1
        topics.add(log.get("topic") or "未知主题")
        topk_sum += int(log.get("topk_count", 0) or 0)
        items.append({
            "log_id": log.get("log_id"),
            "session_id": log.get("session_id", ""),
            "query": log.get("query", ""),
            "topic": log.get("topic", ""),
            "item_names": log.get("item_names", []),
            "topk_count": log.get("topk_count", 0),
            "sources": log.get("sources", []),
            "latency_ms": log.get("latency_ms", 0),
            "correct": correct,
            "comment": (fb or {}).get("comment", ""),
            "created_at": log.get("created_at", 0),
        })
    total = len(items)
    summary = {
        "total": total,
        "feedback_count": labeled_cnt,
        "accuracy": round(correct_cnt / labeled_cnt, 4) if labeled_cnt else 0.0,
        "topics": len(topics),
        "avg_topk": round(topk_sum / total, 2) if total else 0,
        "today": today_cnt,
    }
    return {"summary": summary, "items": items}


@app.get("/query/monitor/{session_id}")
async def monitor_detail(session_id: str):
    """某会话的检索日志详情（含最新一条的 TopK 与反馈，供反馈操作）"""
    logs = get_logs_by_session(session_id, limit=50)
    if not logs:
        raise HTTPException(status_code=404, detail="未找到该会话的检索日志")
    latest = logs[0]
    fb = get_feedback_by_log(latest.get("log_id"))
    return {
        "session_id": session_id,
        "count": len(logs),
        "latest": {
            "log_id": latest.get("log_id"),
            "query": latest.get("query", ""),
            "topic": latest.get("topic", ""),
            "item_names": latest.get("item_names", []),
            "answer_len": latest.get("answer_len", 0),
            "latency_ms": latest.get("latency_ms", 0),
            "topk": latest.get("topk", []),
            "ratios": latest.get("ratios", {}),
            "created_at": latest.get("created_at", 0),
            "feedback": {"correct": fb.get("correct"), "comment": fb.get("comment", "")} if fb else None,
        },
        "logs": [
            {"log_id": x.get("log_id"), "query": x.get("query", ""),
             "topic": x.get("topic", ""), "created_at": x.get("created_at", 0)}
            for x in logs
        ],
    }


@app.post("/query/feedback")
async def submit_feedback(req: FeedbackRequest):
    """提交人工反馈（答案是否正确），用于质量监控闭环"""
    ok = save_feedback(log_id=req.log_id, correct=req.correct,
                       comment=req.comment or "", source="manual")
    if not ok:
        raise HTTPException(status_code=500, detail="反馈写入失败，请稍后重试")
    logger.info(f"已记录人工反馈：log_id={req.log_id}, correct={req.correct}")
    return {"ok": True, "log_id": req.log_id, "correct": req.correct}


@app.get("/monitor/diagnose")
async def monitor_diagnose(days: int = 14):
    """
    按时间窗聚合各主题质量指标并给出诊断与治理建议（dry-run：仅建议，不修改生产数据）。
    """
    since_ts = time.time() - max(1, days) * 86400
    metrics = aggregate_topic_metrics(since_ts=since_ts)
    topics = []
    for m in metrics:
        ratios = {
            "avg_topk": m.get("avg_topk", 0.0),
            "dup_ratio": m.get("dup_ratio", 0.0),
            "stale_ratio": m.get("stale_ratio", 0.0),
            "web_ratio": m.get("web_ratio", 0.0),
        }
        diagnosis = classify_diagnosis(m.get("sample_count", 0), m.get("accuracy", 0.0),
                                       m.get("trend", "insufficient"), ratios)
        # 供建议引用的计数快照
        snap = dict(m)
        snap["stale_count"] = round(m.get("stale_ratio", 0.0) * m.get("avg_topk", 0.0) * m.get("query_count", 0))
        snap["dup_count"] = round(m.get("dup_ratio", 0.0) * m.get("avg_topk", 0.0) * m.get("query_count", 0))
        topics.append({**m, "diagnosis": diagnosis,
                       "recommendations": build_recommendations(diagnosis, snap)})
    return {
        "window_days": days,
        "generated_at": time.time(),
        "topic_count": len(topics),
        "topics": topics,
    }



if __name__ == '__main__':
    uvicorn.run(app, host="127.0.0.1", port=8001)