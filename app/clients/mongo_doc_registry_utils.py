"""
文档版本注册表 MongoDB 工具类。
承载"逻辑文档 -> 版本链"的结构化元信息，供导入侧版本判定与（可选）查询侧溯源使用。

集合同步说明：
- doc_registry ：每个逻辑文档一条记录（doc_id 唯一），记录最新版本号与最新元信息。
- doc_versions ：每个版本一条记录（doc_id + version 复合唯一），记录版本历史与归档路径。
"""
import os
import time
from typing import List, Dict, Any, Optional

from pymongo import MongoClient, ASCENDING
from dotenv import load_dotenv

from app.core.logger import logger

load_dotenv()


class DocRegistryMongoTool:
    """doc_registry / doc_versions 集合的连接与索引初始化"""

    def __init__(self):
        try:
            self.mongo_url = os.getenv("MONGO_URL")
            self.db_name = os.getenv("MONGO_DB_NAME")
            self.client = MongoClient(self.mongo_url)
            self.db = self.client[self.db_name]

            # 逻辑文档注册表
            self.doc_registry = self.db["doc_registry"]
            # 版本历史表
            self.doc_versions = self.db["doc_versions"]

            # 索引：create_index 自带幂等性
            self.doc_registry.create_index([("doc_id", ASCENDING)], unique=True)
            self.doc_versions.create_index([("doc_id", ASCENDING), ("version", ASCENDING)], unique=True)

            logger.info(f"DocRegistryMongoTool 已连接 MongoDB：{self.db_name}")
        except Exception as e:
            logger.error(f"DocRegistryMongoTool 初始化失败：{e}")
            raise


_doc_registry_tool: Optional[DocRegistryMongoTool] = None

try:
    _doc_registry_tool = DocRegistryMongoTool()
except Exception as e:
    # 模块加载阶段失败只告警，保留懒加载兜底
    logger.warning(f"DocRegistryMongoTool 模块加载初始化失败：{e}")


def get_doc_registry_tool() -> DocRegistryMongoTool:
    """获取单例（懒加载）"""
    global _doc_registry_tool
    if _doc_registry_tool is None:
        _doc_registry_tool = DocRegistryMongoTool()
    return _doc_registry_tool


def get_latest_registry(doc_id: str) -> Optional[Dict[str, Any]]:
    """查询某逻辑文档的注册表记录，不存在返回 None"""
    try:
        tool = get_doc_registry_tool()
        return tool.doc_registry.find_one({"doc_id": doc_id})
    except Exception as e:
        logger.error(f"查询 doc_registry 失败：doc_id={doc_id}，{e}")
        return None


def upsert_registry(doc_id: str, file_title: str, item_name: str, latest_version: int,
                    content_hash: str, version_note: str = "",
                    publish_date: int = 0, expire_date: int = 0,
                    date_source: str = "fallback", source_dept: str = "") -> None:
    """
    写入/更新逻辑文档注册表（doc_id 唯一）。
    记录最新版本号与最新元信息；updated_at 每次刷新。
    """
    try:
        tool = get_doc_registry_tool()
        tool.doc_registry.update_one(
            {"doc_id": doc_id},
            {"$set": {
                "doc_id": doc_id,
                "file_title": file_title,
                "item_name": item_name,
                "latest_version": latest_version,
                "content_hash": content_hash,
                "version_note": version_note,
                "publish_date": publish_date,
                "expire_date": expire_date,
                "date_source": date_source,
                "source_dept": source_dept or "",
                "updated_at": time.time(),
            }},
            upsert=True,
        )
    except Exception as e:
        logger.error(f"更新 doc_registry 失败：doc_id={doc_id}，{e}")


def add_version_record(doc_id: str, version: int, file_title: str, item_name: str,
                       content_hash: str, version_note: str = "",
                       publish_date: int = 0, expire_date: int = 0,
                       date_source: str = "fallback", chunks_path: str = "",
                       is_latest: bool = True, source_dept: str = "") -> None:
    """新增一条版本历史记录（doc_id + version 唯一，重复写入幂等）"""
    try:
        tool = get_doc_registry_tool()
        tool.doc_versions.update_one(
            {"doc_id": doc_id, "version": version},
            {"$set": {
                "doc_id": doc_id,
                "version": version,
                "file_title": file_title,
                "item_name": item_name,
                "content_hash": content_hash,
                "version_note": version_note,
                "publish_date": publish_date,
                "expire_date": expire_date,
                "date_source": date_source,
                "source_dept": source_dept or "",
                "chunks_path": chunks_path,
                "is_latest": is_latest,
                "created_at": time.time(),
            }},
            upsert=True,
        )
    except Exception as e:
        logger.error(f"写入 doc_versions 失败：doc_id={doc_id}, version={version}，{e}")


def mark_old_versions_not_latest(doc_id: str, keep_version: int) -> int:
    """将某文档除 keep_version 外的历史版本标记为非最新，返回修改条数"""
    try:
        tool = get_doc_registry_tool()
        result = tool.doc_versions.update_many(
            {"doc_id": doc_id, "version": {"$ne": keep_version}},
            {"$set": {"is_latest": False}},
        )
        return result.modified_count
    except Exception as e:
        logger.error(f"更新历史版本 is_latest 失败：doc_id={doc_id}，{e}")
        return 0


def list_versions(doc_id: str) -> List[Dict[str, Any]]:
    """列出某逻辑文档的全部版本（版本号升序）"""
    try:
        tool = get_doc_registry_tool()
        cursor = tool.doc_versions.find({"doc_id": doc_id}).sort("version", ASCENDING)
        return list(cursor)
    except Exception as e:
        logger.error(f"查询 doc_versions 失败：doc_id={doc_id}，{e}")
        return []


def find_doc_ids_by_item_names(item_names: List[str]) -> List[str]:
    """由 item_name 反查逻辑文档 doc_id 列表（用于查询侧可选溯源）"""
    if not item_names:
        return []
    try:
        tool = get_doc_registry_tool()
        cursor = tool.doc_registry.find({"item_name": {"$in": list(item_names)}})
        return list({doc.get("doc_id") for doc in cursor if doc.get("doc_id")})
    except Exception as e:
        logger.error(f"由 item_name 反查 doc_id 失败：{e}")
        return []


def list_registry() -> List[Dict[str, Any]]:
    """列出全部逻辑文档注册表记录（已剔除 MongoDB ObjectId，可直接 JSON 序列化）"""
    try:
        tool = get_doc_registry_tool()
        docs = []
        for doc in tool.doc_registry.find({}):
            doc.pop("_id", None)
            docs.append(doc)
        return docs
    except Exception as e:
        logger.error(f"列出 doc_registry 失败：{e}")
        return []


def delete_registry(doc_id: str = None, item_name: str = None) -> int:
    """
    删除注册表记录（按 doc_id 或 item_name 精确匹配，二者至少给一个）。
    :return: 删除条数
    """
    cond = {}
    if doc_id:
        cond["doc_id"] = doc_id
    if item_name:
        cond["item_name"] = item_name
    if not cond:
        return 0
    try:
        tool = get_doc_registry_tool()
        return int(tool.doc_registry.delete_many(cond).deleted_count)
    except Exception as e:
        logger.error(f"删除 doc_registry 失败：cond={cond}，{e}")
        return 0


def delete_versions(doc_id: str) -> int:
    """删除某逻辑文档的全部版本历史记录，返回删除条数"""
    if not doc_id:
        return 0
    try:
        tool = get_doc_registry_tool()
        return int(tool.doc_versions.delete_many({"doc_id": doc_id}).deleted_count)
    except Exception as e:
        logger.error(f"删除 doc_versions 失败：doc_id={doc_id}，{e}")
        return 0
