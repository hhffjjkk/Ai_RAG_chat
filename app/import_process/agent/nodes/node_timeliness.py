import sys

from app.conf.timeliness_config import timeliness_config
from app.import_process.agent.state import ImportGraphState
from app.utils.task_utils import add_running_task, add_done_task
from app.utils.doc_meta_utils import (
    parse_date_to_epoch,
    extract_date_from_text,
    compute_timeliness_status,
    now_epoch,
)
from app.clients.mongo_doc_registry_utils import upsert_registry
from app.core.logger import logger

"""
node_timeliness —— 文档时效性标记节点

职责（插在 node_doc_version 之后、node_document_split 之前）：
  1. 解析生效日期 publish_date / 失效日期 expire_date
     优先级：上传表单 > 正文正则提取 > （可选）LLM 提取 > 兜底（上传时间 / 长期有效）
  2. 计算时效状态 timeliness_status
  3. 回写 state，并刷新版本注册表中的日期字段
"""


def step_1_from_form(state):
    """表单主路径：解析上传时填写的生效/失效日期"""
    publish_date = parse_date_to_epoch(state.get("publish_date"))
    expire_date = parse_date_to_epoch(state.get("expire_date"))
    source = None
    if publish_date or expire_date:
        source = "form"
    return publish_date or 0, expire_date or 0, source


def step_2_from_regex(state):
    """辅助路径：从正文正则提取第一个日期作为发布日期"""
    return extract_date_from_text(state.get("md_content", ""))


def step_3_from_llm(state):
    """辅助路径（可选）：使用 LLM 从正文首部提取日期；默认关闭"""
    if not timeliness_config.timeliness_extract_llm_enabled:
        return None
    try:
        from app.core.load_prompt import load_prompt
        from app.lm.lm_utils import get_llm_client
        from langchain_core.messages import HumanMessage

        content = (state.get("md_content", "") or "")[:2000]
        prompt = load_prompt("timeliness_extract", content=content)
        llm = get_llm_client(json_mode=True)
        response = llm.invoke([HumanMessage(content=prompt)])
        import json
        text = (response.content or "").strip()
        if text.startswith("```json"):
            text = text.replace("```json", "").replace("```", "")
        data = json.loads(text) if text else {}
        return parse_date_to_epoch(data.get("publish_date"))
    except Exception as e:
        logger.warning(f"LLM 日期提取失败，回退到其他来源：{e}")
        return None


def node_timeliness(state: ImportGraphState) -> ImportGraphState:
    """时效性标记节点：解析日期、计算时效状态并回写"""
    function_name = sys._getframe().f_code.co_name
    logger.info(f">>> [{function_name}]开始执行了！现在的状态为：{state}")
    add_running_task(state["task_id"], function_name)
    try:
        # 关闭开关：直接使用兜底值（上传时间 / 长期有效）
        if not timeliness_config.timeliness_enabled:
            state["publish_date"] = now_epoch()
            state["expire_date"] = 0
            state["date_source"] = "fallback"
            state["timeliness_status"] = "fresh"
            logger.info("TIMELINESS_ENABLED=false，使用兜底时效信息")
            return state

        # 1. 表单主路径
        publish_date, expire_date, source = step_1_from_form(state)
        # 2. 正则辅助
        if not publish_date and source is None:
            regex_date = step_2_from_regex(state)
            if regex_date:
                publish_date, source = regex_date, "regex"
        # 3. LLM 辅助
        if not publish_date and source is None:
            llm_date = step_3_from_llm(state)
            if llm_date:
                publish_date, source = llm_date, "llm"
        # 4. 兜底
        if not publish_date:
            publish_date = now_epoch()
            if source is None:
                source = "fallback"

        now_ts = now_epoch()
        status = compute_timeliness_status(
            publish_date=publish_date,
            expire_date=expire_date,
            now_ts=now_ts,
            stale_warn_days=timeliness_config.timeliness_stale_warn_days,
        )

        # 5. 回写 state
        state["publish_date"] = publish_date
        state["expire_date"] = expire_date
        state["date_source"] = source
        state["timeliness_status"] = status

        # 6. 刷新注册表中的日期字段（重复上传时仅 refresh 场景需要；这里幂等调用无害）
        if timeliness_config.version_enabled and state.get("doc_id"):
            upsert_registry(
                doc_id=state["doc_id"],
                file_title=state.get("file_title", ""),
                item_name=state.get("item_name", ""),
                latest_version=int(state.get("version", 1) or 1),
                content_hash=state.get("content_hash", ""),
                version_note=state.get("version_note", "") or "",
                publish_date=publish_date,
                expire_date=expire_date,
                date_source=source,
                source_dept=state.get("source_dept", "") or "",
            )
        logger.info(
            f"时效解析完成：publish_date={publish_date}, expire_date={expire_date}, "
            f"source={source}, status={status}"
        )
    except Exception as e:
        logger.error(f">>> [{function_name}]发生异常，异常信息：{e}")
        raise
    finally:
        logger.info(f">>> [{function_name}]结束执行了！现在的状态为：{state}")
        add_done_task(state["task_id"], function_name)
    return state
