import os
import sys
from pathlib import Path

from app.core.logger import logger
from app.import_process.agent.state import ImportGraphState
from app.utils.task_utils import add_running_task, add_done_task


def node_entry(state: ImportGraphState) -> ImportGraphState:
    """
    节点: 入口节点 (node_entry)
    为什么叫这个名字: 作为图的 Entry Point，负责接收外部输入并决定流程走向。
    未来要实现:
        1.日志进入节点-》前端
        2.参数校验(local_file_path -> 没有传入文件 -》 end | local_dir -> 没有则默认创建一个临时文件夹)
        3.解析文件类型, 修改state local_file_path -》 pdf | md
            -> is_pdf_read_enabled | is_md_read_enabled
            -> md_path/pdf_path = local_file_path
            -> file_title = 读取文件名
        4.日志节点结束-》前端
    """
    #1.日志进入节点-》前端
    function_name = sys._getframe().f_code.co_name
    logger.info(f">>> [{function_name}]开始执行节点: {state}")
    add_running_task(state['task_id'],function_name)

    #2.非空校验
    local_file_path = state.get("local_file_path")
    if not local_file_path:
        logger.error(f"[{function_name}]参数校验失败: local_file_path 为空")
        raise state

    #3.解析文件类型, 修改state local_file_path -》 pdf | md
    if local_file_path.endswith(".pdf"):
        state["is_pdf_read_enabled"] = True
        state["pdf_path"] = local_file_path
    elif local_file_path.endswith(".md"):
        state["is_md_read_enabled"] = True
        state["md_path"] = local_file_path
    else:
        logger.error(f"[{function_name}]参数校验失败: local_file_path 不是 pdf | md 文件")

    file_title_os = os.path.basename(local_file_path).split(".")[0]
    file_title = Path(local_file_path).stem
    state["file_title"] = file_title


    #4.日志节点结束 -》前端
    logger.info(f">>> [{function_name}]结束执行节点: {state}")
    add_done_task(state['task_id'], function_name)
    return state