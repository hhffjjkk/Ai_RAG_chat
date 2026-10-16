"""
知识库文档管理工具：已导入文档列表 + 按文档删除。

文档识别口径（两类，兼容历史数据）：
  1. 新形态文档（有 doc_id）：以 doc_id 作为逻辑文档标识，权威目录为 MongoDB 的 doc_registry，
     切片存于 kb_chunks，doc_id / version / is_latest 等为 Milvus 动态字段。
  2. 历史形态文档（无 doc_id）：仅以 item_name(+file_title) 标识，目录来源 kb_item_names 集合。

删除动作（整篇逻辑文档，全量不可逆）：
  - kb_chunks      ：按 doc_id 或 item_name 精确删除切片（含全部历史版本）
  - kb_item_names  ：删除对应主体目录条目（用于查询侧 item_name 匹配）
  - doc_registry   ：删除逻辑文档注册表记录
  - doc_versions   ：删除该文档的全部版本历史记录
"""
from typing import Any, Dict, List

from app.clients.milvus_utils import (
    get_milvus_client,
    count_by_filter,
    delete_by_filter,
    list_item_name_docs,
    list_chunk_docs,
)
from app.clients.mongo_doc_registry_utils import (
    list_registry,
    get_latest_registry,
    delete_registry,
    delete_versions,
)
from app.conf.milvus_config import milvus_config
from app.utils.escape_milvus_string_utils import escape_milvus_string
from app.core.logger import logger

# 支持的删除口径类型
KIND_DOC_ID = "doc_id"
KIND_ITEM_NAME = "item_name"
_VALID_KINDS = (KIND_DOC_ID, KIND_ITEM_NAME)


def build_doc_chunk_filter(kind: str, key: str) -> str:
    """构建 kb_chunks 的切片过滤表达式：doc_id 或 item_name 精确匹配"""
    safe = escape_milvus_string(key)
    if kind == KIND_DOC_ID:
        return f'doc_id == "{safe}"'
    return f'item_name == "{safe}"'


def list_kb_documents() -> List[Dict[str, Any]]:
    """
    列出知识库中已导入的文档（合并 doc_registry 与 kb_item_names 两个来源）。
    注册表文档以 doc_id 为删除键；历史文档以 item_name 为删除键。
    每条记录附带该文档当前切片数量（kb_chunks 统计）。
    :return: 文档列表（按更新时间倒序）
    """
    milvus = get_milvus_client()
    chunks_coll = milvus_config.chunks_collection
    items: List[Dict[str, Any]] = []
    registry_titles = set()

    # 1. 新形态文档：版本注册表（doc_id 权威目录）
    for rec in list_registry():
        doc_id = rec.get("doc_id") or ""
        file_title = rec.get("file_title") or ""
        registry_titles.add(file_title)
        chunk_count = 0
        if doc_id:
            chunk_count = count_by_filter(
                milvus, chunks_coll, build_doc_chunk_filter(KIND_DOC_ID, doc_id))
        items.append({
            "kind": KIND_DOC_ID,
            "key": doc_id,
            "doc_id": doc_id,
            "file_title": file_title,
            "item_name": rec.get("item_name") or "",
            "latest_version": int(rec.get("latest_version", 1) or 1),
            "publish_date": int(rec.get("publish_date", 0) or 0),
            "expire_date": int(rec.get("expire_date", 0) or 0),
            "source_dept": rec.get("source_dept") or "",
            "updated_at": float(rec.get("updated_at", 0) or 0),
            "chunk_count": chunk_count,
            "source": "registry",
        })

    # 2. 历史形态文档：item_name 目录 + 切片兜底扫描（合并去重），排除已由注册表覆盖的同名文档
    seen_legacy = set()
    legacy_docs = (list_item_name_docs(milvus, milvus_config.item_name_collection)
                   + list_chunk_docs(milvus, chunks_coll))
    for doc in legacy_docs:
        file_title = doc.get("file_title") or ""
        item_name = doc.get("item_name") or ""
        dedup_key = (file_title, item_name)
        if dedup_key in seen_legacy:
            continue
        seen_legacy.add(dedup_key)
        if file_title in registry_titles:
            continue
        items.append({
            "kind": KIND_ITEM_NAME,
            "key": item_name,
            "doc_id": "",
            "file_title": file_title,
            "item_name": item_name,
            "latest_version": 0,
            "publish_date": 0,
            "expire_date": 0,
            "source_dept": "",
            "updated_at": 0.0,
            "chunk_count": count_by_filter(
                milvus, chunks_coll, build_doc_chunk_filter(KIND_ITEM_NAME, item_name)),
            "source": "item_name",
        })

    # 3. 排序：注册表文档（有更新时间）优先，其次按标题
    items.sort(key=lambda x: (-x["updated_at"], x["file_title"]))
    logger.info(f"已导入文档列表读取完成，共 {len(items)} 篇")
    return items


def delete_kb_document(kind: str, key: str) -> Dict[str, Any]:
    """
    删除指定文档：清理 Milvus 切片、item_name 目录与 MongoDB 版本注册表。
    :param kind: 删除口径，"doc_id"（新形态）或 "item_name"（历史形态）
    :param key: 对应的 doc_id 或 item_name
    :return: 删除统计 {ok, deleted_chunks, deleted_item_names, deleted_registry, deleted_versions}
    :raises ValueError: 参数非法
    """
    kind = (kind or "").strip()
    key = (key or "").strip()
    if kind not in _VALID_KINDS:
        raise ValueError(f"不支持的删除类型：{kind}")
    if not key:
        raise ValueError("删除标识不能为空")

    milvus = get_milvus_client()
    chunks_coll = milvus_config.chunks_collection
    item_coll = milvus_config.item_name_collection
    result: Dict[str, Any] = {
        "ok": True,
        "kind": kind,
        "key": key,
        "deleted_chunks": 0,
        "deleted_item_names": 0,
        "deleted_registry": 0,
        "deleted_versions": 0,
    }

    if kind == KIND_DOC_ID:
        # 1. 删除 kb_chunks 中该 doc_id 的全部切片（含历史版本）
        result["deleted_chunks"] = delete_by_filter(
            milvus, chunks_coll, build_doc_chunk_filter(KIND_DOC_ID, key))
        # 2. 删除 kb_item_names 中同名文档的主体条目（doc_id 与 item_name 目录无直接键关联，按 file_title 匹配）
        rec = get_latest_registry(key)
        file_title = (rec or {}).get("file_title") or ""
        if file_title:
            result["deleted_item_names"] = delete_by_filter(
                milvus, item_coll, f'file_title == "{escape_milvus_string(file_title)}"')
        # 3. 删除 MongoDB 版本链
        result["deleted_versions"] = delete_versions(key)
        result["deleted_registry"] = delete_registry(doc_id=key)
    else:
        # 1. 历史形态：按 item_name 删除切片
        result["deleted_chunks"] = delete_by_filter(
            milvus, chunks_coll, build_doc_chunk_filter(KIND_ITEM_NAME, key))
        # 2. 删除主体目录条目
        result["deleted_item_names"] = delete_by_filter(
            milvus, item_coll, build_doc_chunk_filter(KIND_ITEM_NAME, key))
        # 3. 兼容清理注册表（历史文档通常无记录，幂等无害）
        result["deleted_registry"] = delete_registry(item_name=key)

    logger.info(f"文档删除完成：{result}")
    return result
