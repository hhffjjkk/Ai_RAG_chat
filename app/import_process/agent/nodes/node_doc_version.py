import sys

from app.conf.timeliness_config import timeliness_config
from app.import_process.agent.state import ImportGraphState
from app.utils.task_utils import add_running_task, add_done_task
from app.utils.doc_meta_utils import normalize_doc_id, compute_content_hash
from app.clients.mongo_doc_registry_utils import (
    get_latest_registry,
    upsert_registry,
    add_version_record,
    mark_old_versions_not_latest,
)
from app.core.logger import logger

"""
node_doc_version —— 文档版本控制节点

职责（插在 node_md_img 之后、node_document_split 之前）：
  1. 归一化逻辑文档标识 doc_id（表单指定 > file_title 归一）
  2. 计算正文内容指纹 content_hash（MD5）
  3. 查版本注册表，决定本次上传的 version / 是否重复 / 是否最新
  4. 写入 doc_registry + doc_versions 版本链
短流程原则：
  - VERSION_ENABLED=false 时退化为"始终 version=1 / is_latest=True"，不查注册表
  - 命中重复且 DUPLICATE_POLICY=skip 时置 is_duplicate=True，由图中条件边短路结束
"""


def step_1_resolve_doc_id(state):
    """归一 doc_id：表单显式指定优先，否则用 file_title"""
    file_title = state.get("file_title", "")
    form_doc_id = state.get("doc_id", "")
    doc_id = normalize_doc_id(file_title, form_doc_id, timeliness_config.doc_identity_strategy)
    logger.info(f"doc_id 归一结果：{doc_id}（file_title={file_title}, 表单={form_doc_id}）")
    return doc_id


def step_2_decide_version(doc_id, content_hash):
    """
    版本决策：
      无历史记录        -> version=1, is_duplicate=False
      有记录且指纹相同  -> is_duplicate=True, version=历史最新版本
      有记录且指纹不同  -> version=历史最新版本+1, is_duplicate=False
    """
    latest_registry = get_latest_registry(doc_id)
    if not latest_registry:
        return 1, False
    latest_version = int(latest_registry.get("latest_version", 1) or 1)
    if latest_registry.get("content_hash") == content_hash:
        return latest_version, True
    return latest_version + 1, False


def step_3_write_registry(state, doc_id, version, is_duplicate):
    """写入版本链与注册表；重复上传按策略 refresh/skip 处理"""
    file_title = state.get("file_title", "")
    version_note = state.get("version_note", "") or ""
    content_hash = state.get("content_hash", "")
    publish_date = state.get("publish_date", 0) or 0
    expire_date = state.get("expire_date", 0) or 0
    date_source = state.get("date_source", "fallback")
    source_dept = state.get("source_dept", "") or ""
    chunks_path = state.get("archived_chunks_path", "") or ""

    if is_duplicate:
        if timeliness_config.duplicate_policy == "refresh":
            # 仅刷新注册表 updated_at，不新增版本记录
            upsert_registry(
                doc_id=doc_id, file_title=file_title, item_name=state.get("item_name", ""),
                latest_version=version, content_hash=content_hash, version_note=version_note,
                publish_date=publish_date, expire_date=expire_date, date_source=date_source,
                source_dept=source_dept,
            )
            logger.info(f"重复上传（内容未变），按 DUPLICATE_POLICY=refresh 刷新注册表：doc_id={doc_id}")
        else:
            logger.info(f"重复上传（内容未变），按 DUPLICATE_POLICY=skip 跳过入库：doc_id={doc_id}")
        return

    # 新版本：写版本记录 + 翻转旧版本 is_latest + 更新注册表
    add_version_record(
        doc_id=doc_id, version=version, file_title=file_title, item_name=state.get("item_name", ""),
        content_hash=content_hash, version_note=version_note,
        publish_date=publish_date, expire_date=expire_date, date_source=date_source,
        chunks_path=chunks_path, is_latest=True, source_dept=source_dept,
    )
    changed = mark_old_versions_not_latest(doc_id, version)
    upsert_registry(
        doc_id=doc_id, file_title=file_title, item_name=state.get("item_name", ""),
        latest_version=version, content_hash=content_hash, version_note=version_note,
        publish_date=publish_date, expire_date=expire_date, date_source=date_source,
        source_dept=source_dept,
    )
    logger.info(f"新版本写入完成：doc_id={doc_id}, version={version}, 历史版本标记非最新数={changed}")


def node_doc_version(state: ImportGraphState) -> ImportGraphState:
    """文档版本控制节点：归一 doc_id、计算指纹、决定版本、写入版本链"""
    function_name = sys._getframe().f_code.co_name
    logger.info(f">>> [{function_name}]开始执行了！现在的状态为：{state}")
    add_running_task(state["task_id"], function_name)
    try:
        # 1. 归一 doc_id
        doc_id = step_1_resolve_doc_id(state)
        # 2. 计算内容指纹
        content_hash = compute_content_hash(state.get("md_content", ""))
        # 3. 版本决策（开关关闭时退化为单版本）
        if not timeliness_config.version_enabled:
            version, is_duplicate = 1, False
            logger.info("VERSION_ENABLED=false，跳过版本链判定，统一使用 version=1")
        else:
            version, is_duplicate = step_2_decide_version(doc_id, content_hash)
        # 4. 回写 state
        state["doc_id"] = doc_id
        state["content_hash"] = content_hash
        state["version"] = version
        state["is_duplicate"] = is_duplicate
        state["is_latest"] = (not is_duplicate)
        # 5. 写入版本链（重复时按策略处理）
        if timeliness_config.version_enabled:
            step_3_write_registry(state, doc_id, version, is_duplicate)
    except Exception as e:
        logger.error(f">>> [{function_name}]发生异常，异常信息：{e}")
        raise
    finally:
        logger.info(f">>> [{function_name}]结束执行了！现在的状态为：{state}")
        add_done_task(state["task_id"], function_name)
    return state
