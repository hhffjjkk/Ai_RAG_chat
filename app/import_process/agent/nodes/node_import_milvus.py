import os
import sys
from typing import List, Dict, Any
# 导入Milvus相关依赖
from pymilvus import DataType
# 导入自定义模块
from app.import_process.agent.state import ImportGraphState
from app.clients.milvus_utils import get_milvus_client
from app.utils.task_utils import add_running_task, add_done_task
from app.core.logger import logger
from app.conf.milvus_config import milvus_config
from app.conf.timeliness_config import timeliness_config
from app.utils.escape_milvus_string_utils import escape_milvus_string

# 从配置文件读取切片集合名称，与配置解耦，便于环境切换
CHUNKS_COLLECTION_NAME = milvus_config.chunks_collection


def step_2_prepare_collections(state):
    """
        创建chunks对应的集合！！！
        :param state:
        :return:
        """
    # 1. 获取milvus的客户端
    milvus_client = get_milvus_client()
    # 2. 判断是否存在集合（表），存在创建集合（表）
    if not milvus_client.has_collection(collection_name=milvus_config.chunks_collection):
        # 创建集合
        # 3.1. 创建集合对应的列的信息
        schema = milvus_client.create_schema(
            auto_id=True,  # 主键自增长
            enable_dynamic_field=True,  # 动态字段
        )

        # 3.2. Add fields to schema
        # pk file_title item_name dense_vector sparse_vector
        schema.add_field(field_name="chunk_id", datatype=DataType.INT64, is_primary=True, auto_id=True)
        schema.add_field(field_name="file_title", datatype=DataType.VARCHAR, max_length=65535)
        schema.add_field(field_name="item_name", datatype=DataType.VARCHAR, max_length=65535)
        schema.add_field(field_name="content", datatype=DataType.VARCHAR, max_length=65535)
        schema.add_field(field_name="title", datatype=DataType.VARCHAR, max_length=65535)
        schema.add_field(field_name="parent_title", datatype=DataType.VARCHAR, max_length=65535)
        schema.add_field(field_name="part", datatype=DataType.INT8)
        # --- 版本控制 + 时效性 新增标量字段（用于检索期过滤） ---
        schema.add_field(field_name="doc_id", datatype=DataType.VARCHAR, max_length=65535)
        schema.add_field(field_name="version", datatype=DataType.INT64)
        schema.add_field(field_name="is_latest", datatype=DataType.BOOL)
        schema.add_field(field_name="publish_date", datatype=DataType.INT64)
        schema.add_field(field_name="expire_date", datatype=DataType.INT64)
        schema.add_field(field_name="timeliness_status", datatype=DataType.VARCHAR, max_length=64)
        schema.add_field(field_name="source_dept", datatype=DataType.VARCHAR, max_length=512)
        schema.add_field(field_name="dense_vector", datatype=DataType.FLOAT_VECTOR, dim=1024)
        schema.add_field(field_name="sparse_vector", datatype=DataType.SPARSE_FLOAT_VECTOR)
        # 3.3 查询快，配置索引
        index_params = milvus_client.prepare_index_params()

        # 标量倒排索引：加速 item_name / 版本 / 时效 / 来源部门 的过滤
        for scalar_field in ("item_name", "doc_id", "is_latest",
                             "publish_date", "expire_date", "timeliness_status", "source_dept"):
            index_params.add_index(field_name=scalar_field, index_type="INVERTED")

        index_params.add_index(
            field_name="dense_vector",  # 给哪个列创建索引 稠密
            index_name="dense_vector_index",  # 索引的名字
            index_type="HNSW",  # 配置查找所用的算法
            metric_type="COSINE",  # 配置向量匹配和对比的 IP COSINE
            params={"M": 32,  # Maximum number of neighbors each node can connect to in the graph
                    "efConstruction": 300},  # or "DAAT_WAND" or "TAAT_NAIVE"
        )
        """
           10000  M = 16  efConstruction = 200
           50000  M = 32  efConstruction = 300
           100000  M = 64  efConstruction = 400
           M:图中每个节点在层次结构的每个层级所能拥有的最大边数或连接数。M 越高，图的密度就越大，搜索结果的召回率和准确率也就越高，因为有更多的路径可以探索，但同时也会消耗更多内存，并由于连接数的增加而减慢插入时间。如上图所示，M = 5表示 HNSW 图中的每个节点最多与 5 个其他节点直接相连。这就形成了一个中等密度的图结构，节点有多条路径到达其他节点。
           efConstruction:索引构建过程中考虑的候选节点数量。efConstruction 越高，图的质量越好，但需要更多时间来构建。
        """
        index_params.add_index(
            field_name="sparse_vector",  # Name of the vector field to be indexed
            index_type="SPARSE_INVERTED_INDEX",  # Type of the index to create
            index_name="sparse_vector_index",  # Name of the index to create
            metric_type="IP",  # Metric type used to measure similarity
            # 只计算可能得高分的向量，跳过大量的 0
            params={"inverted_index_algo": "DAAT_MAXSCORE"},  # Algorithm used for building and querying the index
        )

        milvus_client.create_collection(
            collection_name=milvus_config.chunks_collection,
            schema=schema,  # 字段
            index_params=index_params  # 索引
        )
    return milvus_client


def step_3_delete_old_data(milvus_client, doc_id):
    """
        删除旧数据：按 doc_id 精确删除（替换原来的按 item_name 删除）。
        方案A：Milvus 只保留每个逻辑文档的最新版本切片，导入新版本前先精确清理旧切片。
        方案B：保留全部版本，不做删除（由 is_latest 标记区分）。
        :param milvus_client:
        :param doc_id: 逻辑文档标识
        :return:
        """
    if not doc_id:
        logger.warning("doc_id 为空，跳过旧切片清理（避免误删全库数据）")
        return
    if timeliness_config.keep_all_versions_in_milvus:
        logger.info("KEEP_ALL_VERSIONS_IN_MILVUS=true（方案B），保留历史版本切片，跳过删除")
        return
    safe_doc_id = escape_milvus_string(doc_id)
    milvus_client.delete(collection_name=CHUNKS_COLLECTION_NAME,
                         filter=f'doc_id=="{safe_doc_id}"')

    milvus_client.load_collection(collection_name=CHUNKS_COLLECTION_NAME)


def step_4_insert_collections(milvus_client, chunks):
    """
        插入集合的数据！
        :param chunks:
        :return:  chunks -> 主键回显
        """
    insert_result = milvus_client.insert(collection_name=CHUNKS_COLLECTION_NAME, data=chunks)
    # 成功插入了几条
    insert_count = insert_result.get("insert_count", 0)
    logger.info(f"完成了数据插入，成功插入了 {insert_count} 条数据")
    # 获取回显的ids
    ids = insert_result.get("ids", [])

    if ids and len(ids) == len(chunks):
        for index, chunk in enumerate(chunks):
            chunk['chunk_id'] = ids[index]

    return chunks


def node_import_milvus(state: ImportGraphState) -> ImportGraphState:
    """
    节点: 导入向量库 (node_import_milvus)
    为什么叫这个名字: 将处理好的向量数据写入 Milvus 数据库。
    未来要实现:
    1. 连接 Milvus。
    2. 根据 item_name 删除旧数据 (幂等性)。
    3. 批量插入新的向量数据。
    """
    function_name = sys._getframe().f_code.co_name
    logger.info(f">>> [{function_name}]开始执行了！现在的状态为：{state}")
    add_running_task(state['task_id'], function_name)
    try:
    # 0. 幂等短路：内容未变且策略为 skip 时，跳过入库
        if state.get("is_duplicate") and timeliness_config.duplicate_policy == "skip":
            logger.info(f">>> [{function_name}]检测到重复上传（内容未变），按 DUPLICATE_POLICY=skip 跳过入库")
            return state
    # 1. 获取输入的数据 （校验）
        chunks = state.get('chunks')
        if not chunks:
            logger.error(f">>> [{function_name}]chunks没有数据")
            raise ValueError("chunks没有数据")
    # 2. 没有集合，要创建集合 collection (filed,index,collection)
        milvus_client = step_2_prepare_collections(state)
    # 3. 删除旧数据（按 doc_id 精确清理，方案A）
        doc_id = state.get("doc_id") or chunks[0].get("doc_id")
        step_3_delete_old_data(milvus_client, doc_id)
    # 4. 插入chunks的数据即可
        with_id_chunks = step_4_insert_collections(milvus_client, chunks)

        state['chunks'] = with_id_chunks

    except Exception as e:
        # 处理异常
        logger.error(f">>> [{function_name}]使用Milvus发生了异常，异常信息：{e}")
        raise  # 终止工作流
    finally:
        # 6. 结束的日志和任务状态的配置
        logger.info(f">>> [{function_name}]开始结束了！现在的状态为：{state}")
        add_done_task(state['task_id'], function_name)
    return state


if __name__ == '__main__':
    # --- 单元测试 ---
    # 目的：验证 Milvus 导入节点的完整流程，包括连接、创建集合、清理旧数据和插入新数据。
    import sys
    import os
    from dotenv import load_dotenv

    # 加载环境变量 (自动寻找项目根目录的 .env)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(current_dir))
    load_dotenv(os.path.join(project_root, ".env"))

    # 构造测试数据
    dim = 1024
    test_state = {
        "task_id": "test_milvus_task",
        "chunks": [
            {
                "content": "Milvus 测试文本 1",
                "title": "测试标题",
                "item_name": "测试项目_Milvus",  # 必须有 item_name，用于幂等清理
                "parent_title":"test.pdf",
                "part":1,
                "file_title": "test.pdf",
                "dense_vector": [0.1] * dim,  # 模拟 Dense Vector
                "sparse_vector": {1: 0.5, 10: 0.8}  # 模拟 Sparse Vector
            },{
                "content": "Milvus 测试文本 2",
                "title": "测试标题2",
                "item_name": "测试项目_Milvus",  # 必须有 item_name，用于幂等清理
                "parent_title":"test.pdf",
                "part":1,
                "file_title": "test.pdf",
                "dense_vector": [0.1] * dim,  # 模拟 Dense Vector
                "sparse_vector": {1: 0.5, 10: 0.8}  # 模拟 Sparse Vector
            },{
                "content": "Milvus 测试文本 3",
                "title": "测试标题3",
                "item_name": "测试项目_Milvus",  # 必须有 item_name，用于幂等清理
                "parent_title":"test.pdf",
                "part":1,
                "file_title": "test.pdf",
                "dense_vector": [0.1] * dim,  # 模拟 Dense Vector
                "sparse_vector": {1: 0.5, 10: 0.8}  # 模拟 Sparse Vector
            }
        ]
    }

    print("正在执行 Milvus 导入节点测试...")
    try:
        # 检查必要的环境变量
        if not os.getenv("MILVUS_URL"):
            print("❌ 未设置 MILVUS_URL，无法连接 Milvus")
        elif not os.getenv("CHUNKS_COLLECTION"):
            print("❌ 未设置 CHUNKS_COLLECTION")
        else:
            # 执行节点函数
            result_state = node_import_milvus(test_state)

            # 验证结果
            chunks = result_state.get("chunks", [])
            if chunks and chunks[0].get("chunk_id"):
                print(f"✅ Milvus 导入测试通过，生成 ID: {chunks[0]['chunk_id']}")
            else:
                print("❌ 测试失败：未能获取 chunk_id")

    except Exception as e:
        print(f"❌ 测试失败: {e}")