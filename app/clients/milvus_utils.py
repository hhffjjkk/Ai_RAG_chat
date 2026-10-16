import os
from pymilvus import MilvusClient, AnnSearchRequest, WeightedRanker
from app.conf.milvus_config import milvus_config
from app.core.logger import logger

# 全局Milvus客户端实例，实现单例复用
_milvus_client = None


def get_milvus_client():
    """
    Milvus客户端单例获取方法
    实现客户端连接复用，避免重复创建连接消耗资源
    :return: MilvusClient实例，连接失败返回None
    """
    try:
        global _milvus_client
        # 单例判断：未初始化则创建新连接
        if _milvus_client is None:
            milvus_uri = milvus_config.milvus_url
            # 校验Milvus连接地址配置
            if not milvus_uri:
                logger.error("Milvus客户端连接失败：缺少MILVUS_URL环境变量配置")
                return None
            # 初始化Milvus客户端
            _milvus_client = MilvusClient(uri=milvus_uri)
            logger.info("Milvus客户端连接成功")
        return _milvus_client
    except Exception as e:
        logger.error(f"Milvus客户端连接异常：{str(e)}", exc_info=True)
        return None


def _coerce_int64_ids(ids):
    """
    转换chunk_id为Milvus要求的INT64类型（主键字段schema为INT64）
    过滤无效ID，分离可转换/不可转换的ID
    :param ids: 待转换的chunk_id列表
    :return: 元组(ok_ids, bad_ids)，ok_ids为可转换的int64类型ID列表，bad_ids为无效ID列表
    """
    ok, bad = [], []
    for x in (ids or []):
        if x is None:
            continue
        try:
            ok.append(int(x))
        except Exception:
            bad.append(x)
    return ok, bad


def fetch_chunks_by_chunk_ids(
        client,
        collection_name: str,
        chunk_ids,
        *,
        output_fields=None,
        batch_size: int = 100,
):
    """
    通过chunk_id主键批量查询Milvus中的切片数据
    用于补全「仅拥有chunk_id无文本内容」场景的切片信息
    优先使用get方法（主键直查，性能最优），失败则回退query过滤查询
    :param client: MilvusClient实例
    :param collection_name: 集合名称
    :param chunk_ids: 待查询的chunk_id列表
    :param output_fields: 需要返回的字段列表，默认返回核心切片字段
    :param batch_size: 分批查询大小，避免单次查询数据量过大，默认100
    :return: List[dict]，Milvus实体字典列表，查询失败返回空列表
    """
    # 前置校验：客户端/集合名无效直接返回空
    if client is None:
        return []
    if not collection_name:
        return []
    # 默认返回字段：核心切片标识与内容字段（含版本/时效/来源部门，供监控与溯源）
    if output_fields is None:
        output_fields = ["chunk_id", "content", "title", "parent_title", "item_name",
                         "doc_id", "version", "is_latest", "publish_date", "expire_date",
                         "timeliness_status", "source_dept"]

    # 转换ID为INT64类型，分离有效/无效ID
    ok_ids, bad_ids = _coerce_int64_ids(chunk_ids)
    if bad_ids:
        # 记录无效ID，跳过查询
        logger.warning(f"存在无法转换为INT64的chunk_id，将跳过查询：{bad_ids}")

    # 无有效ID直接返回空
    if not ok_ids:
        return []

    results = []
    # 分批查询：按batch_size切分有效ID，循环查询
    for i in range(0, len(ok_ids), batch_size):
        batch = ok_ids[i: i + batch_size]

        # 方式1：优先使用主键get方法查询（性能最优）
        if hasattr(client, "get"):
            try:
                got = client.get(collection_name=collection_name, ids=batch, output_fields=output_fields)
                if got:
                    results.extend(got)
                continue
            except Exception as e:
                logger.warning(f"Milvus get方法查询失败，将回退至query方法：{str(e)}")

        # 方式2：get方法失败，回退使用filter过滤查询
        try:
            expr = f"chunk_id in [{', '.join(str(x) for x in batch)}]"
            q = client.query(collection_name=collection_name, filter=expr, output_fields=output_fields)
            if q:
                results.extend(q)
        except Exception as e:
            logger.error(f"Milvus query方法批量查询chunk_id失败：{str(e)}", exc_info=True)

    return results


def build_chunk_filter_expr(item_names, now_ts=None):
    """
    构建 kb_chunks 检索过滤表达式：主体过滤 + 版本过滤 + 时效过滤。
    :param item_names: 已确认的主体名称列表
    :param now_ts: 当前时间戳（epoch 秒），默认取服务端当前时间
    :return: Milvus filter 表达式字符串
    说明：过滤项均受开关控制（VERSION_ENABLED / TIMELINESS_ENABLED / GENERAL_RETRIEVAL_ENABLED），
          关闭开关即退回"仅按 item_name 过滤"的改造前行为（便于回滚）。
          方案A：当未确认到任何商品名且 GENERAL_RETRIEVAL_ENABLED=true 时，
          退化为通用检索——不再追加 item_name 过滤，仅保留版本/时效约束。
    """
    from app.utils.escape_milvus_string_utils import escape_milvus_string
    from app.utils.doc_meta_utils import now_epoch
    from app.conf.timeliness_config import timeliness_config

    # 过滤掉空主体，避免生成 item_name in [] 这类非法/恒空表达式
    names = [x for x in (item_names or []) if x]
    parts = []
    if names:
        item_name_str = ', '.join(f'"{escape_milvus_string(item)}"' for item in names)
        parts.append(f"item_name in [{item_name_str}]")
    elif not timeliness_config.general_retrieval_enabled:
        # 未开启通用检索兜底：维持"无主体即无召回"的旧语义（恒假条件）
        parts.append("chunk_id < 0")
    else:
        logger.info("未确认商品名，按通用检索构建过滤表达式（不加 item_name 过滤）")
    # 版本过滤：默认只命中最新版本（方案A下为防御性条件）
    if timeliness_config.version_enabled:
        parts.append("is_latest == true")
    # 时效过滤：默认硬过滤排除已过期内容
    if (timeliness_config.timeliness_enabled
            and timeliness_config.timeliness_expired_policy == "exclude"):
        ts = now_ts or now_epoch()
        parts.append(f"(expire_date == 0 or expire_date >= {ts})")
    # 生效下界：默认排除"尚未生效"（publish_date > now）的内容，保证只返回当前有效文档
    if (timeliness_config.timeliness_enabled
            and timeliness_config.timeliness_effective_filter_enabled):
        ts = now_ts or now_epoch()
        parts.append(f"(publish_date == 0 or publish_date <= {ts})")
    return " and ".join(parts)


def create_hybrid_search_requests(dense_vector, sparse_vector, dense_params=None, sparse_params=None, expr=None,
                                  limit=5):
    """
    构建Milvus混合搜索请求对象
    分别创建稠密/稀疏向量的搜索请求，用于后续混合搜索融合
    :param dense_vector: 文本生成的稠密向量
    :param sparse_vector: 文本生成的稀疏向量
    :param dense_params: 稠密向量搜索参数，默认使用余弦相似度
    :param sparse_params: 稀疏向量搜索参数，默认使用内积相似度
    :param expr: 搜索过滤表达式，用于精准筛选数据
    :param limit: 单向量搜索返回结果数量，默认5
    :return: 搜索请求列表，包含[dense_req, sparse_req]
    """
    # 稠密向量默认搜索参数：余弦相似度（COSINE），适配BGE-M3稠密向量并与建库参数保持一致
    if dense_params is None:
        dense_params = {"metric_type": "COSINE"}
    # 稀疏向量默认搜索参数：内积（IP），适配BGE-M3稀疏向量
    if sparse_params is None:
        sparse_params = {"metric_type": "IP"}

    # 构建稠密向量搜索请求，关联Milvus的dense_vector字段 近似最近邻（ANN）检索请求的核心类
    dense_req = AnnSearchRequest(
        data=[dense_vector],
        anns_field="dense_vector",
        param=dense_params,
        expr=expr,
        limit=limit
    )

    # 构建稀疏向量搜索请求，关联Milvus的sparse_vector字段
    sparse_req = AnnSearchRequest(
        data=[sparse_vector],
        anns_field="sparse_vector",
        param=sparse_params,
        expr=expr,
        limit=limit
    )

    return [dense_req, sparse_req]


def hybrid_search(client, collection_name, reqs, ranker_weights=(0.5, 0.5), norm_score=False, limit=5,
                  output_fields=None, search_params=None):
    """
    执行Milvus稠密+稀疏向量混合搜索
    基于WeightedRanker实现双向量搜索结果加权融合，提升检索准确性
    :param client: MilvusClient实例
    :param collection_name: 集合名称
    :param reqs: 搜索请求列表，固定为[dense_req, sparse_req]
    :param ranker_weights: 加权融合权重，默认(0.5,0.5)，依次对应稠密/稀疏向量
    :param norm_score: 是否归一化评分后再融合，避免评分量级差异导致权重失效
    :param limit: 混合搜索最终返回结果数量，默认5
    :param output_fields: 需要返回的字段列表，默认返回item_name
    :param search_params: 搜索参数，如ef/topk等，默认None
    :return: 混合搜索结果列表，搜索失败返回None
    """
    try:
        # 初始化加权排名器：按权重融合稠密/稀疏向量的搜索结果
        # norm_score=True：先将两个向量评分归一化到0~1区间，再加权计算
        rerank = WeightedRanker(ranker_weights[0], ranker_weights[1], norm_score=norm_score)

        # 默认返回字段：文档标识字段
        if output_fields is None:
            output_fields = ["item_name"]

        # 执行混合搜索：融合稠密+稀疏向量结果，按权重重新排序
        res = client.hybrid_search(
            collection_name=collection_name,
            reqs=reqs,
            ranker=rerank,
            limit=limit,
            output_fields=output_fields,
            search_params=search_params
        )

        logger.info(f"Milvus混合搜索完成，集合[{collection_name}]共检索到{len(res[0])}条结果")
        return res
    except Exception as e:
        logger.error(f"Milvus混合搜索执行失败，集合[{collection_name}]：{str(e)}", exc_info=True)
        return None


def count_by_filter(client, collection_name: str, filter_expr: str = "") -> int:
    """
    统计集合中满足过滤条件的实体数量（优先 count(*)，失败回退逐行统计）。
    :param client: MilvusClient实例
    :param collection_name: 集合名称
    :param filter_expr: 过滤表达式，空串表示全量统计
    :return: 命中条数
    """
    if client is None or not collection_name:
        return 0
    expr = (filter_expr or "").strip() or "chunk_id > 0"
    try:
        res = client.query(collection_name=collection_name, filter=expr, output_fields=["count(*)"])
        if res:
            return int(res[0].get("count(*)", 0) or 0)
    except Exception as e:
        logger.warning(f"count(*) 统计失败，回退逐行统计：{str(e)}")
    try:
        rows = client.query(collection_name=collection_name, filter=expr,
                            output_fields=["chunk_id"], limit=16384)
        return len(rows or [])
    except Exception as e:
        logger.error(f"统计集合条数失败，集合[{collection_name}]：{str(e)}")
        return 0


def delete_by_filter(client, collection_name: str, filter_expr: str) -> int:
    """
    按过滤表达式删除集合实体，返回删除条数。
    注意：不依据删除前统计决定是否删除——未 flush 的增量数据统计可能为 0，
    直接执行 delete 并以 Milvus 返回的 delete_count 为准，避免漏删。
    :param client: MilvusClient实例
    :param collection_name: 集合名称
    :param filter_expr: 过滤表达式（不能为空，避免误删全库）
    :return: 删除条数
    """
    if client is None or not collection_name or not (filter_expr or "").strip():
        logger.warning("删除条件为空，已跳过删除操作以避免误删全库")
        return 0
    matched_ref = count_by_filter(client, collection_name, filter_expr)  # 仅供 delete_count 缺失时兜底
    try:
        res = client.delete(collection_name=collection_name, filter=filter_expr)
        # 删除后必须 flush，否则软删除不会立即对 query/search 生效（Milvus 增量段语义）
        if hasattr(client, "flush"):
            client.flush(collection_name=collection_name)
        client.load_collection(collection_name=collection_name)
        deleted = None
        if isinstance(res, dict):
            deleted = res.get("delete_count")
        deleted = int(deleted) if deleted is not None else matched_ref
        logger.info(f"已按条件删除集合[{collection_name}]实体：条件={filter_expr}，删除条数={deleted}")
        return deleted
    except Exception as e:
        logger.error(f"按条件删除集合[{collection_name}]失败：条件={filter_expr}，{str(e)}", exc_info=True)
        return 0


def list_item_name_docs(client, collection_name: str) -> list:
    """
    列出 item_name 目录集合中的（file_title, item_name）文档条目（已按二者去重，跳过空 item_name）。
    :return: [{"file_title": str, "item_name": str}, ...]
    """
    if client is None or not collection_name:
        return []
    try:
        rows = client.query(collection_name=collection_name,
                            filter="pk > 0",
                            output_fields=["file_title", "item_name"],
                            limit=16384)
    except Exception as e:
        logger.error(f"列出 item_name 目录失败，集合[{collection_name}]：{str(e)}")
        return []
    return _dedup_docs(rows)


def list_chunk_docs(client, collection_name: str, limit: int = 16384) -> list:
    """
    从切片集合中列出（file_title, item_name）文档条目（跳过空 item_name，按二者去重）。
    用于兜底发现"item_name 目录缺失但在库中确有切片"的历史文档，保证列表与库内实际内容一致。
    注意：为控制开销采用 limit 上限扫描，单个知识库文档规模在此量级内可完整覆盖。
    :return: [{"file_title": str, "item_name": str}, ...]
    """
    if client is None or not collection_name:
        return []
    try:
        rows = client.query(collection_name=collection_name,
                            filter="chunk_id > 0",
                            output_fields=["file_title", "item_name"],
                            limit=limit)
    except Exception as e:
        logger.error(f"从切片集合列出文档失败，集合[{collection_name}]：{str(e)}")
        return []
    return _dedup_docs(rows)


def _dedup_docs(rows) -> list:
    """按 (file_title, item_name) 去重，跳过空 item_name"""
    seen, docs = set(), []
    for row in rows or []:
        item_name = row.get("item_name") or ""
        file_title = row.get("file_title") or ""
        if not item_name:
            continue
        key = (file_title, item_name)
        if key in seen:
            continue
        seen.add(key)
        docs.append({"file_title": file_title, "item_name": item_name})
    return docs