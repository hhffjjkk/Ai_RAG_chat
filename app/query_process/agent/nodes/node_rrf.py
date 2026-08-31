import sys
from typing import List, Dict, Any
from app.utils.task_utils import add_running_task, add_done_task
from app.core.logger import logger


def step_3_reciprocal_rank_fusion(source_with_weight,top_k:int =5):
    """
        进行同源数据排名+权重处理
        :param source_with_weight:  [(集合,权重),(集合,权重)]
        :return: [{},{},{}]
        """
    # 1. 准备两个容器 记录历史得分
    score_dict = {}  # id | chunk_id -> key  ||  计算得分 -》 value
    # 2. 记录我chunk片段
    chunk_dict = {}  # 记录 chunk_id -> key || chunk {}  不重复
    # 3. 循环处理每个集合中的数据，进行积分计算
    for source, weight in source_with_weight:
        for rank ,chunk in enumerate(source,start=1):
            chunk_id = chunk.get("id") or chunk.get("entity").get("chunk_id")
            score_dict[chunk_id] = score_dict.get(chunk_id,0.0) + (1.0/(60 + rank)) * weight
            # chunk_dict[chunk_id] = chunk  # 1 -  覆盖 《- 1  保留一份
            chunk_dict.setdefault(chunk_id, chunk)  # 没有的时候才会添加 1  - 失败 《 - 1
    #4.排序
    merged = []
    for chunk_id, score in score_dict.items():
        chunk = chunk_dict.get(chunk_id)
        merged.append((chunk,score))
    merged.sort(key=lambda x:x[1],reverse=True)
    merged = merged[:top_k]
    rank_chunks = [chunk for chunk, score in merged]
    logger.info(f"完成了rrf排序处理完毕，结果为：{rank_chunks}")
    return rank_chunks


def node_rrf(state):
    """
    节点功能：Reciprocal Rank Fusion
    将多路召回的结果（向量、HyDE、Web、KG）进行加权融合排序。
    """
    print("---RRF---")
    add_running_task(state["session_id"], sys._getframe().f_code.co_name, state.get("is_stream"))

    # 1. 获取同源路的数据
    # 长啥样？？？？
    # milvus => [[],[]]  1. 单列查询  data = [向量1，向量2]   -> [[向量1],[向量2]]
    #                    2. 混合查询 reps = [anns....]      -> [[]]
    #                     [向量1]  =》 【{id: 实体的主键,distance:得分 0.8,entity:{chunk_id,content,title..}} ,{} ,{}】
    embedding_chunks = state.get("embedding_chunks")
    hyde_embedding_chunks = state.get("hyde_embedding_chunks")
    # 2. 数据进行整合 （捏到一起）
    # 权重后续方便动态调整！ 相同 1.0 1.0
    source_with_weight = [
        (embedding_chunks, 1.0),
        (hyde_embedding_chunks, 1.0)
    ]
    # 3. rrf算法进行数据排序处理
    rrf_response = step_3_reciprocal_rank_fusion(source_with_weight)
    state["rrf_chunks"] = rrf_response


    # ...
    add_done_task(state['session_id'], sys._getframe().f_code.co_name, state.get("is_stream"))
    return state


# ================================
# 本地测试入口
# ================================
if __name__ == "__main__":
    print("\n" + "=" * 50)
    print(">>> 启动 node_rrf 本地测试")
    print("=" * 50)

    mock_state = {
        "session_id": "test_rrf_session",
        "is_stream": False,
        "original_query": "RS-12万用表怎么操作？",
        "rewritten_query": "RS-12万用表的具体操作步骤是什么？",
        "item_names": ["RS-12万用表"]
    }

    try:
        from app.query_process.agent.nodes.node_search_embedding import node_search_embedding
        from app.query_process.agent.nodes.node_search_embedding_hyde import node_search_embedding_hyde

        emb_res = node_search_embedding(mock_state)
        hyde_res = node_search_embedding_hyde(mock_state)
        mock_state['embedding_chunks'] = emb_res.get("embedding_chunks") or []
        mock_state['hyde_embedding_chunks'] = hyde_res.get("hyde_embedding_chunks") or []

        result = node_rrf(mock_state)
        rrf_chunks = result.get("rrf_chunks", [])

        emb_cnt = len(mock_state.get("embedding_chunks") or [])
        hyde_cnt = len(mock_state.get("hyde_embedding_chunks") or [])

        print("\n" + "=" * 50)
        print(">>> 测试结果摘要:")
        print(f"输入数量: Embedding={emb_cnt}, HyDE={hyde_cnt}")
        print(f"输出数量: {len(rrf_chunks)}")
        print("-" * 30)

        print("最终排名:")
        for i, doc in enumerate(rrf_chunks, 1):
            doc_id = doc.get("chunk_id") or doc.get("id")
            content = (doc.get("content") or "")[:20]
            print(f"Rank {i}: ID={doc_id}, Content={content}...")

        print("=" * 50)

    except Exception as e:
        logger.exception(f"测试运行期间发生未捕获异常: {e}")