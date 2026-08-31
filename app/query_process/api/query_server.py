# 6个接口  健康状态 返回页面  发起提问   sse长连接  查看历史对话  清空历史对话
from pathlib import Path
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
from app.query_process.agent.main_graph import query_app


# 定义fastapi对象
app = FastAPI(title="query service", description="掌柜智库查询服务！")

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



if __name__ == '__main__':
    uvicorn.run(app, host="127.0.0.1", port=8001)