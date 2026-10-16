import sys
import os

from app.conf.milvus_config import milvus_config
from app.utils.task_utils import add_running_task,add_done_task
from app.lm.embedding_utils import generate_embeddings
from app.clients.milvus_utils import create_hybrid_search_requests,hybrid_search,get_milvus_client,build_chunk_filter_expr
from app.core.logger import logger
from dotenv import load_dotenv,find_dotenv
load_dotenv(find_dotenv())


def node_search_embedding(state):
    """
    节点功能：进行向量内容检索
    """
    print("---量内容检索 开始处理---")
    add_running_task(state["session_id"], sys._getframe().f_code.co_name, state.get("is_stream"))

    # 搜索向量内容
    rewritten_query = state.get("rewritten_query")
    item_names = state.get("item_names")
    embeddings = generate_embeddings([rewritten_query])
    # 构建过滤表达式：item_name + 版本(仅最新) + 时效(排除过期)，开关受配置控制
    expr = build_chunk_filter_expr(item_names, state.get("now_ts"))
    hybrid_search_requests = create_hybrid_search_requests(
        dense_vector=embeddings['dense'][0],
        sparse_vector=embeddings['sparse'][0],
        expr=expr
    )
    # 3.2 进行混合查询触发
    milvus_client = get_milvus_client()
    resp = hybrid_search(
        client=milvus_client,
        collection_name=milvus_config.chunks_collection,
        reqs=hybrid_search_requests,
        ranker_weights=(0.9, 0.1),
        norm_score=True,
        limit=5,
        output_fields=["chunk_id", "content", "file_title", "title", "parent_title", "item_name",
                       "doc_id", "version", "is_latest", "publish_date", "expire_date", "timeliness_status",
                       "source_dept"]
    )
    """
          [

            [
               {id ,
               distance，
               entity:
                  {
                     "chunk_id", "content","file_title", "title", "parent_title", "item_name"
                  }
            ]
          ]

       """
    # 4. 处理查询结果赋值 embedding_chunks 属性即可
    embedding_chunks = resp[0] if resp else []


    # ...
    add_done_task(state["session_id"], sys._getframe().f_code.co_name, state.get("is_stream"))

    print("---量内容检索 处理结束---")
    return {"embedding_chunks": embedding_chunks}


if __name__ == "__main__":
    # 模拟测试数据
    test_state = {
        "session_id": "test_search_embedding_001",
        "rewritten_query": "RS-12万用表使用说明",  # 模拟改写后的查询
        "item_names": ["RS-12万用表"],  # 模拟已确认的商品名
        "is_stream": False
    }

    print("\n>>> 开始测试 node_search_embedding 节点...")
    try:
        # 执行节点函数
        result = node_search_embedding(test_state)
        logger.info(f"检索结果汇总：{result}")
        # 验证结果
        chunks = result.get("embedding_chunks", [])
        print(f"\n>>> 测试完成！检索到 {len(chunks)} 条结果,结果为：{chunks}")

    except Exception as e:
        logger.error(f"测试运行失败: {e}", exc_info=True)