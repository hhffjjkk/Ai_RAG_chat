# 导入基础库：系统、路径、类型注解（类型注解提升代码可读性和可维护性）
import os
import sys
from typing import List, Dict, Any, Tuple

# 导入Milvus客户端（向量数据库核心操作）、数据类型枚举（定义集合Schema）
from pymilvus import MilvusClient, DataType
# 导入LangChain消息类（标准化大模型对话消息格式）
from langchain_core.messages import SystemMessage, HumanMessage

from app.conf.milvus_config import milvus_config
# 导入自定义模块：
# 1. 流程状态载体：ImportGraphState为LangGraph流程的统一状态管理对象
from app.import_process.agent.state import ImportGraphState
# 2. Milvus工具：获取单例Milvus客户端，实现连接复用
from app.clients.milvus_utils import get_milvus_client
# 3. 大模型工具：获取大模型客户端，统一模型调用入口
from app.lm.lm_utils import get_llm_client
# 4. 向量工具：BGE-M3模型实例、向量生成方法（稠密+稀疏向量）
from app.lm.embedding_utils import get_bge_m3_ef, generate_embeddings
# 5. 稀疏向量工具：归一化处理，保证向量长度为1，提升检索准确性
from app.utils.normalize_sparse_vector import normalize_sparse_vector
# 6. 任务工具：更新任务运行状态，用于任务监控和管理
from app.utils.task_utils import add_running_task, add_done_task
# 7. 日志工具：项目统一日志入口，分级输出（info/warning/error）
from app.core.logger import logger
# 8. 提示词工具：加载本地prompt模板，实现提示词与代码解耦
from app.core.load_prompt import load_prompt

from app.utils.escape_milvus_string_utils import escape_milvus_string

# --- 配置参数 (Configuration) ---
# 大模型识别商品名称的上下文切片数：取前5个切片，避免上下文过长导致大模型输入超限
DEFAULT_ITEM_NAME_CHUNK_K = 5
# 单个切片内容截断长度：防止单切片内容过长，占满大模型上下文
SINGLE_CHUNK_CONTENT_MAX_LEN = 800
# 大模型上下文总字符数上限：适配主流大模型输入限制，默认2500
CONTEXT_TOTAL_MAX_CHARS = 2500

"""
  主要目标：
     1. 录用文本大模型识别当前chunks对应的item_name！用于区分不同的文档
     2. 使用嵌入式模型，将item_name生成向量存储到向量数据库 
     3. 修改state[chunks] -> chunk {title parent_title part file_title content item_name => 每个赋值 }
  实现步骤：
     1. 校验和取值 （file_title,chunks）
     2. 构建上下文环境  chunks -> top 5 -> 拼接成context文本 
     3. 调用模型，拼接提示词，识别chunks对应item_name
     4. 修改state chunks -》 item_name 
     5. item_name生成向量（稠密/稀疏）
     6. 存储向量到向量数据库 kb_item_name (id / file_title / item_name / 稠密 和 稀疏)
 """


def step_1_get_chunks(state):
    chunks = state.get('chunks')
    file_title = state.get('file_title')
    if not chunks:
        raise ValueError("Chunks没有在state中")
    if not file_title:
        #获取md文件名
        file_title = os.path.basename(state.get('md_path'))
        logger.info(f"file_title缺少,{file_title}")
        state['file_title'] = file_title
    return chunks,file_title


def step_2_build_context(chunks):
    """
            根据chunks切面的content内容进行分拼接！ （2000）
            截取内容限制： 1. 最多截取前top个 （5） 2. 最多字符不能超过 CONTEXT_TOTAL_MAX_CHARS
            截取内容处理：
                  切片：{1}，标题:{title},内容：{content} \n\n
                  切片：{2}，标题:{title},内容：{content} \n\n
                  切片：{3}，标题:{title},内容：{content} \n\n
                  切片：{4}，标题:{title},内容：{content} \n\n
                  切片：{5}，标题:{title},内容：{content} \n\n
            :param chunks:
            :return:
            """
    parts = []
    total_chars = 0
    for index, chunk in enumerate(chunks[:DEFAULT_ITEM_NAME_CHUNK_K],start=1):
        chunk_title = chunk['title']
        chunk_content = chunk['content']
        data = f"切片：{index}，标题:{chunk_title},内容：{chunk_content}"
        parts.append(data)
        total_chars += len(data)
        if total_chars >= CONTEXT_TOTAL_MAX_CHARS:
            logger.info(f"已达到最大字符数限制，截取内容长度为：{total_chars}")
            break

    context = "\n\n".join(parts)
    final_context = context[:SINGLE_CHUNK_CONTENT_MAX_LEN]

    return final_context


def step_3_call_llm(context, file_title):
    """
            想模型调用! 获取item_name!
            使用file_tile进行兜底！！
            :param context:
            :param file_title:
            :return:
            """
    # 1. 构建提示词
    human_prompt = load_prompt("item_name_recognition",file_title=file_title,context=context)
    system_prompt = load_prompt("product_recognition_system")
    # 2.获取模型对象
    llm = get_llm_client(json_mode=False)
    # 3. 调用模型
    message=[SystemMessage(content=system_prompt), HumanMessage(content=human_prompt)]
    response = llm.invoke(message)
    # 4. 阶段判断和兜底
    item_name = response.content
    if not item_name:
        item_name = file_title
    return item_name


def step_4_update_chunks_and_state(state, chunks, item_name):
    """
            state[item_name] = item_name
            chunks -> {item_name:item_name}
            :param state:
            :param item_name:
            :param chunks:
            :return:
        """
    state['item_name'] = item_name
    for chunk in chunks:
        chunk['item_name'] = item_name
    state['chunks'] = chunks
    logger.info(f"更新state和chunks成功！")



def step_5_generate_embeddings(item_name):
    """
            根据item_name生成向量 -》 稠密 + 稀疏
            :param item_name:
            :return: dense_vector [稠密] ,  sparse_vector [稀疏]
            """
    """
    generate_embeddings 自己封装的嵌入式模式生成向量的函数！！ 
          embeddings list对应的向量 = model.encode_documents(texts) 传入的字符串list 
          参数：生成向量的字符串 ["1","2","3"] 
          返回结果： 
             result = {
                        "dense": [1的稠密,2的稠密,3的稠密],  #稠密向量
                        "sparse": [1的稀疏,2的稀疏,3的稀疏], #稀疏向量
                      }
    """
    result = generate_embeddings([item_name])
    dense_vector, sparse_vector = result['dense'][0],result['sparse'][0]
    return dense_vector, sparse_vector


def step_6_save_to_vector_db(file_title, item_name, dense_vector, sparse_vector):
    """
            将向量和对应的字段保存到向量数据库中！！
            :param file_title:
            :param item_name:
            :param dense_vector:
            :param sparse_vector:
            :return:
            """
    # 1.获取milvus向量库
    milvus_client = get_milvus_client()
    # 2.判断是否存在集合
    if not milvus_client.has_collection(collection_name=milvus_config.item_name_collection):
        # 3.1. Create schema
        schema = MilvusClient.create_schema(
            auto_id=False,
            enable_dynamic_field=True,
        )

        # 3.2. Add fields to schema
        schema.add_field(field_name="pk", datatype=DataType.INT64, is_primary=True, auto_id=True)
        schema.add_field(field_name="file_title", datatype=DataType.VARCHAR, max_length=65535)
        schema.add_field(field_name="item_name", datatype=DataType.VARCHAR, max_length=65535)
        schema.add_field(field_name="dense_vector", datatype=DataType.FLOAT_VECTOR, dim=1024)
        schema.add_field(field_name="sparse_vector", datatype=DataType.SPARSE_FLOAT_VECTOR)
        # 3.3 查询快，配置索引
        index_params = milvus_client.prepare_index_params()
        index_params.add_index(
            field_name="dense_vector",  # 给哪个列创建索引 稠密
            index_name="dense_vector_index",  # 索引的名字
            index_type="HNSW",  # 配置查找所用的算法
            metric_type="COSINE",  # 配置向量匹配和对比的 IP COSINE
            params={"M": 16,  # Maximum number of neighbors each node can connect to in the graph
                    "efConstruction": 200},  # or "DAAT_WAND" or "TAAT_NAIVE"
        )
        index_params.add_index(
            field_name="sparse_vector",  # Name of the sparse vector field to index
            index_type="SPARSE_INVERTED_INDEX",  # Type of the index to create
            index_name="sparse_vector_index",  # Name of the index to create
            metric_type="IP",  # Metric type used for full text search
            params={"inverted_index_algo": "DAAT_MAXSCORE"},
        )
        milvus_client.create_collection(
            collection_name=milvus_config.item_name_collection,
            schema=schema,  # 字段
            index_params=index_params  # 索引
        )
    # 3. 先删除之前存在的item_name
    milvus_client.load_collection(collection_name=milvus_config.item_name_collection)
    # milvus_client.delete(collection_name=milvus_config.item_name_collection,
    #                      filter=f"item_name='{item_name}'")
    # 4. 向集合插入最新的item_name数据和对应的向量即可
    item = {
        "file_title": file_title,
        "item_name": item_name,
        "dense_vector": dense_vector,
        "sparse_vector": sparse_vector
    }
    milvus_client.insert(collection_name=milvus_config.item_name_collection, data=[item])
    milvus_client.load_collection(collection_name=milvus_config.item_name_collection)
    logger.info(f"保存了item_name:{item_name}的数据到向量数据库中！！")


def node_item_name_recognition(state: ImportGraphState) -> ImportGraphState:
    """
    节点: 主体识别 (node_item_name_recognition)
    为什么叫这个名字: 识别文档核心描述的物品/商品名称 (Item Name)。
    未来要实现:
    1. 取文档前几段内容。
    2. 调用 LLM 识别这篇文档讲的是什么东西 (如: "Fluke 17B+ 万用表")。
    3. 存入 state["item_name"] 用于后续数据幂等性清理。
    """
    # 1.日志进入节点-》前端
    function_name = sys._getframe().f_code.co_name
    logger.info(f">>> [{function_name}]开始执行了！现在的状态为：{state}")
    add_running_task(state['task_id'], function_name)

    try:
        #1. 校验和取值 （file_title,chunks）
        chunks,file_title = step_1_get_chunks(state)
        #2. 构建上下文环境  chunks -> top 5 -> 拼接成context文本
        context = step_2_build_context(chunks)
        #3. 调用模型，拼接提示词，识别chunks对应item_name
        item_name = step_3_call_llm(context,file_title)
        #4. 修改state chunks -》 item_name
        step_4_update_chunks_and_state(state, chunks, item_name)
        #5. item_name生成向量（稠密/稀疏）
        dense_vector, sparse_vector = step_5_generate_embeddings(item_name)
        #6. 存储向量到向量数据库 kb_item_name (id / file_title / item_name / 稠密 和 稀疏)
        step_6_save_to_vector_db(file_title,item_name, dense_vector, sparse_vector)

    except Exception as e:
        # 处理异常
        logger.error(f">>> [{function_name}] 发生异常，异常信息：{e}")
        raise  # 终止工作流

    finally:
        # 6. 结束的日志和任务状态的配置
        logger.info(f">>> [{function_name}]开始结束了！现在的状态为：{state}")
        add_done_task(state['task_id'], function_name)
    return state

# ===================== 本地测试方法（直接运行调试，无需启动LangGraph） =====================
def test_node_item_name_recognition():
    """
    商品名称识别节点本地测试方法
    功能：模拟LangGraph流程输入，独立测试node_item_name_recognition节点全链路逻辑
    适用场景：本地开发、调试、单节点功能验证，无需启动整个LangGraph流程
    测试前准备：
        1. 确保项目环境变量配置完成（MILVUS_URL/ITEM_NAME_COLLECTION等）
        2. 确保大模型、Milvus、BGE-M3服务均可正常访问
        3. 确保prompt模板（item_name_recognition/product_recognition_system）已存在
    使用方法：
        直接运行该函数：if __name__ == "__main__": test_node_item_name_recognition()
    """
    logger.info("=== 开始执行商品名称识别节点本地测试 ===")
    try:
        # 1. 构造模拟的ImportGraphState状态（模拟上游节点产出数据）
        mock_state = ImportGraphState({
            "task_id": "test_task_123456",  # 测试任务ID
            "file_title": "华为Mate60 Pro手机使用说明书",  # 模拟文件标题
            "file_name": "华为Mate60Pro说明书.pdf",  # 模拟原始文件名（兜底用）
            # 模拟文本切片列表（上游切片节点产出，含title/content字段）
            "chunks": [
                {
                    "title": "产品简介",
                    "content": "华为Mate60 Pro是华为公司2023年发布的旗舰智能手机，搭载麒麟9000S芯片，支持卫星通话功能，屏幕尺寸6.82英寸，分辨率2700×1224。"
                },
                {
                    "title": "拍照功能",
                    "content": "华为Mate60 Pro后置5000万像素超光变摄像头+1200万像素超广角摄像头+4800万像素长焦摄像头，支持5倍光学变焦，100倍数字变焦。"
                },
                {
                    "title": "电池参数",
                    "content": "电池容量5000mAh，支持88W有线超级快充，50W无线超级快充，反向无线充电功能。"
                }
            ]
        })

        # 2. 调用商品名称识别核心节点
        result_state = node_item_name_recognition(mock_state)

        # 3. 打印测试结果（调试用）
        logger.info("=== 商品名称识别节点本地测试完成 ===")
        logger.info(f"测试任务ID：{result_state.get('task_id')}")
        logger.info(f"最终识别商品名称：{result_state.get('item_name')}")
        logger.info(f"切片数量：{len(result_state.get('chunks', []))}")
        logger.info(f"第一个切片商品名称：{result_state.get('chunks', [{}])[0].get('item_name')}")

        # 4. 验证Milvus存储（可选）
        milvus_client = get_milvus_client()
        collection_name = os.environ.get("ITEM_NAME_COLLECTION")
        if milvus_client and collection_name:
            milvus_client.load_collection(collection_name)
            # 检索测试结果
            item_name = result_state.get('item_name')
            safe_name = escape_milvus_string(item_name)
            res = milvus_client.query(
                collection_name=collection_name,
                filter=f'item_name=="{safe_name}"',
                output_fields=["file_title", "item_name"]
            )
            logger.info(f"Milvus中检索到的数据：{res}")

    except Exception as e:
        logger.error(f"商品名称识别节点本地测试失败，原因：{str(e)}", exc_info=True)


# 测试方法运行入口：直接执行该文件即可触发测试
if __name__ == "__main__":
    # 执行本地测试
    test_node_item_name_recognition()