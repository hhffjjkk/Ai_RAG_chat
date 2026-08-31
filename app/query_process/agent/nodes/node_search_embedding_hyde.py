import sys

from langchain_core.messages import HumanMessage
from pymupdf import message

from app.utils.task_utils import add_running_task, add_done_task
from app.lm.lm_utils import *
from app.lm.embedding_utils import *
from app.clients.milvus_utils import *
from app.core.logger import logger
from app.core.load_prompt import load_prompt
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())


def step_1_create_hyde_doc(rewritten_query):
    llm = get_llm_client()
    hyde_prompt = load_prompt("hyde_prompt",rewritten_query = rewritten_query)
    messages = [
        HumanMessage(content=hyde_prompt)
    ]
    response = llm.invoke(messages)
    hyde_doc = response.content
    logger.info(f"使用模型生成假设性答案，问题：{rewritten_query},答案：{hyde_doc}")
    return hyde_doc


def step_2_search_embedding_hyde(rewritten_query, hyde_doc, item_names):
    """
        根据问题+假设性答案查询向量数据库，进行混合查询
        :param rewritten_query:
        :param hyde_doc:
        :param item_names:
        :return: [[] -> 结果  id 分数 实体列信息 ]
        """
    # 1.拼接重写问题 + lm生成的假设性答案
    query_str = rewritten_query + hyde_doc
    # 2.拼接查询字符串生成对应向量
    embeddings = generate_embeddings([query_str])
    # 3.生成查询AnnSearchRequest列表
    item_name_str = ', '.join(f'"{item}"' for item in item_names)
    reqs = create_hybrid_search_requests(
        dense_vector=embeddings['dense'][0],
        sparse_vector=embeddings['sparse'][0],
        expr=f"item_name in [{item_name_str}]"
    )
    # 4.进行混合查询处理
    milvus_client = get_milvus_client()
    resp = hybrid_search(
        client=milvus_client,
        collection_name=milvus_config.chunks_collection,
        reqs=reqs,
        ranker_weights=(0.9, 0.1),
        output_fields=["item_name", "content", "title", "parent_title", "chunk_id"]
    )
    # 5.处理返回结果
    #rrf出bug
    result = resp[0] if resp else []

    logger.info(f"假设性问题检索结果：{result}")
    return result


def node_search_embedding_hyde(state):
    """
    节点功能：HyDE (Hypothetical Document Embedding)
    先让 LLM 生成假设性答案，再对答案进行向量检索，提高召回率。
    """
    print("---HyDE 开始处理---")
    add_running_task(state["session_id"], sys._getframe().f_code.co_name, state.get("is_stream"))
    # 1. 提取参数 （item_names || rewritten_query）
    rewritten_query = state.get("rewritten_query")
    item_names = state.get("item_names")
    # 2. 调用 LLM 生成假设性答案 rewritten_query
    hyde_doc = step_1_create_hyde_doc(rewritten_query)
    # 3. 问题+答案，进行向量检索（混合检索）
    resp = step_2_search_embedding_hyde(rewritten_query, hyde_doc, item_names)
    # 4. 赋值和返回结果  hyde_embedding_chunks
    # ...
    add_done_task(state["session_id"], sys._getframe().f_code.co_name, state.get("is_stream"))

    print("---HyDE 处理结束---")
    return {"hyde_embedding_chunks":resp}

if __name__ == "__main__":
    # 本地测试代码
    print("\n" + "=" * 50)
    print(">>> 启动 node_search_embedding_hyde 本地测试")
    print("=" * 50)

    # 模拟输入状态
    mock_state = {
        "session_id": "test_hyde_session_001",
        "original_query": "RS-12万用表怎么操作？",
        "rewritten_query": "RS-12万用表的具体操作步骤是什么？",
        "item_names": ["RS-12万用表"],
        "is_stream": False
    }

    try:
        # 运行节点
        result = node_search_embedding_hyde(mock_state)

        print(result)

    except Exception as e:
        logger.exception(f"测试运行期间发生未捕获异常: {e}")
