import sys
import os
import json
import logging
from typing import List, Dict, Any, Optional
from langchain_core.messages import SystemMessage, HumanMessage
from mpmath import limit

from app.conf.milvus_config import milvus_config
from app.core.load_prompt import load_prompt
from app.query_process.agent.state import QueryGraphState
from app.utils.task_utils import add_running_task, add_done_task
from app.clients.mongo_history_utils import get_recent_messages, save_chat_message, update_message_item_names
from app.lm.lm_utils import get_llm_client
from app.lm.embedding_utils import generate_embeddings
from app.clients.milvus_utils import get_milvus_client, create_hybrid_search_requests, hybrid_search
from dotenv import load_dotenv,find_dotenv
from app.core.logger import logger

load_dotenv(find_dotenv())


def step_3_llm_item_name_and_rewrite_query(original_query, history_chats):
    """
    根据历史记录 -》 识别item_names 和 重写问题
    :param original_query: 用户原有的提问
    :param history_chats:  聊天记录
    :return:  {  item_name = [] , rewritten_query:问题
              }
    """
    # 1. 准备提示词
    history_text = ""
    for chat in history_chats:
        history_text += f"聊天角色：{chat['role']}，回答内容： {chat['text']}，重写问题： {chat['rewritten_query']}，关联主体： {','.join(chat.get('item_names',[]))},时间： {chat['ts']}\n"

    prompt = load_prompt("rewritten_query_and_itemnames",history_text=history_text,query= original_query)
    # 2. 模型调用
    lm_client = get_llm_client(json_mode=True)
    # system -> 模型的角色边界！ -> 应该是不变！  【角色，规则，格式】
    # user  ->  每次任务提示 -》 多条动态调整！  【提问/聊天】
    # 事实上，你嫌麻烦，你可以把模型的角色和边界写到user 功能也是完全一样！！
    messages = [
        HumanMessage(content=prompt)
    ]
    response = lm_client.invoke(messages)
    # 怎么确保，模型一定能返回格式化数据？ json!  1.设置json格式化！   2. 提示词中明确   3. 一定要给模型参考示例  4. 做好返回格式的校验
    # 3. 结果解析
    content = response.content
    # json -> ```json   json  ```
    if content.startswith("```json"):
        content = content.replace("```json","").replace("```","")
    dict_content = json.loads(content)

    if "item_names" not in dict_content:
        dict_content["item_names"] = []
    if "rewritten_query" not in dict_content:
        dict_content["rewritten_query"] = original_query # 原提问
    # 4. 封装返回
    logger.info(f"已经完成问题的重写和item_name的提取！ 结果为：{dict_content}")
    return  dict_content


def step_4_query_milvus_item_names(item_names):
    """
     查询向量数据库！ 进行item_name的确定
    :param item_names: 模型提取的item_name可能不准！
    :return:
           [{extracted:模型item_name,matches:[{item_name:xx,score:0.9...}]}]
    """
    # 明确 我们一定做的混合查询 （稠密向量 + 稀疏向量）
    final_result = []
    # 1. 获取milvus的客户端
    milvus_client = get_milvus_client()
    # 2. 将item_name转成向量（稠密和稀疏） 【循环】
    embeddings = generate_embeddings(item_names)
    # 3. 混合查询 （创建稠密和稀疏的AnnSearchRequest  ||  设置权重重排  ||  进行混合查询 ）
    for index,item_name in enumerate(item_names):
        # 1. 获取当前item_name对应的向量
        dense_vector = embeddings["dense"][index]
        sparse_vector = embeddings["sparse"][index]
        # 2. 拼对应的AnnSearchRequest
        reqs = create_hybrid_search_requests(
            dense_vector = dense_vector,
            sparse_vector = sparse_vector
        )
        # 3. 定义权重重排
        # 4. 进行混合检索
        response = hybrid_search(
            client=milvus_client,
            collection_name=milvus_config.item_name_collection,
            reqs=reqs,
            ranker_weights=(0.8,0.2),
            norm_score=True   # 0 - 1
        )

        """
          [
            [
              {id:xx , distance: 0.x,entity:{item_name:xxx} } ,
              {id:xx , distance: 0.x,entity:{item_name:xxx} } 
             ]
          ]
        """
        # 5. 结果解析
        matches = []  #当前item对应的匹配结果
        if response and len(response) > 0:
            for hit in response[0]:
                entity = hit.get("entity",{})
                hit_name = entity.get("item_name")
                score = hit.get("distance",0)
                if hit_name:
                    matches.append({
                        "item_name":hit_name,
                        "score":score
                    })
    # 4. 提取查询结果封装返回的数据格式
        final_result.append({
            "extracted":item_name, #模型给的！
            "matches":matches  # 查询到的
        })
    # 5. 封装返回数据
    logger.info(f"查询向量数据库结果为：{final_result}")
    return final_result


def step_5_confirmed_and_optional_item_name(query_milvus_results):
    """
    通过向量数据库查询的item_name,根据分数归纳出确定和可选的item_name列表
    :param query_milvus_results: 元数据 [{extracted:item_name,matches:[{item_name: , score:},{}]}
                                       ,{extracted:item_name,matches:[{item_name: , score:},{}]}]
    :return:
          {
             confirmed_item_names:[确定item_name], 分高
             options_item_names:[可选item_name]  分低
          }
    评分规则：
          确定item_name
          0.85 【根据权重和数据分析进行调整】
          可选item_name
          0.60
          忽略
    思路： 1. 循环处理每个item_name列表和分  2. 高分 只要1个  3. 可选 可以要2   4. 不区分extracted:item_name 装到对应的 确认或者可选集合中
    """
    #1. 准备两个列表 确认 可选的
    confirmed_item_names = [] #确定
    options_item_names = [] #可选
    #2. 循环处理元数据 query_milvus_results
    for item_name_meta in query_milvus_results:
        extracted_name = item_name_meta.get("extracted")
        matches = item_name_meta.get("matches",[])
        #3. 进行分数排序（倒序） || 列表推导式 提取0.85 || 0.6
        # matches [{score: xx , item_name="" }]
        matches.sort(key=lambda x:x.get("score",0),reverse=True)
        # >= 0.8 [{item_name: , score:},{}]
        # 处理 -》 先处理高分 -》 有 -》 正常执行 || 如果没有 -》 才处理低分
        high_score_matches = [ x for x in matches if x.get("score",0) >= 0.85 ]
        middle_score_matches = [ x for x in matches if x.get("score",0) >= 0.6]
        #4. 处理高分的列表 只有一个1  获取一个1  ||  多个【item_name = extracted 】 or 获取最高分的1个
        # 4.1 只有一个，获取一个
        if len(high_score_matches) ==1:
            confirmed_item_names.append(high_score_matches[0].get("item_name"))
            continue
        # 4.2 有多个高分
        if len(high_score_matches) >1:
            # 同一个名 = 分不是1 也可能不是最高！！
            # 优先考虑名字相同
            same_name_item = None
            for item in high_score_matches:
                if item.get("item_name") == extracted_name:
                    same_name_item = item
                    break
            if not same_name_item:
               same_name_item = high_score_matches[0] #获取分数最高的
            confirmed_item_names.append(same_name_item.get("item_name"))
            continue
            # 没有相同获取分数最高的
        #5. 处理可选分数列表 给用户返回提示，可以多带几个，截取2个！
        if len(middle_score_matches) > 0:
            for item in middle_score_matches[:2]:
                options_item_names.append(item.get("item_name"))
            continue
        logger.info(f"没有匹配的item_name，忽略：{extracted_name}")
    #6. 处理返回结果即可(去重复)
    result = {
        "confirmed_item_names":list(set(confirmed_item_names)),
        "options_item_names":list(set(options_item_names))
    }
    logger.info(f"处理结果为：{result}")
    return result


def step_6_deal_list(state,item_results, history_chats,rewritten_query):
    """
    根据集合类型中数据，判定是否要赋值answer内容
    :param item_results:   # result = {
        #         "confirmed_item_names":list(set(confirmed_item_names)),
        #         "options_item_names":list(set(options_item_names))
        #     }
    :param history_chats:
           [
           ]
    :return:
    """
    # 1. 先获取两个集合 （确认 | 可选的）
    confirmed_item_names = item_results.get("confirmed_item_names",[])
    options_item_names = item_results.get("options_item_names",[])
    # 2. 确认集合有数据 （处理）
    if len(confirmed_item_names) > 0:
        # 2.1 更新下聊天记录 -》 item_names - > confirmed_item_names (空着)
        # 2.2 修改和存储state状态
        state['item_names'] = confirmed_item_names
        state['rewritten_query'] =rewritten_query
        state['history'] = history_chats
        if "answer" in state:
            del state['answer']
        logger.info(f"有确定的item_name:{confirmed_item_names}")
        return state
    # 3. 确认集合没数据，处理可选集合
    if len(options_item_names) > 0:
        option_names = '、'.join(options_item_names)
        answer = f"您是想咨询以下哪个商品：{option_names}?请下次提问明确商品名称！！"
        state['answer'] = answer
        logger.info(f"有可选的item_name:{options_item_names}")
        return state
    # 4. 确认和可选集合都没数据 （处理）
    answer = "没有匹配的商品名，请重新提问！！"
    state['answer'] = answer
    logger.info(f"没有匹配的的item_name")
    return state

def node_item_name_confirm(state):
    """
    节点功能：确认用户问题中的核心商品名称。
    # 核心目标： 1. 提取【 item_name 】 （大模型从历史对话 + 本次提问 提取  -》 item_name -> 向量库搜索 ->  打分 -》 ABC）
               2. 利用模型重写用户的问题，确保后续查询召回率更高！！！
    # 核心参数： state['original_query' -> 用户的原问题 ]  ||  session_id
    # 响应数据： item_names: List[str]  # 提取出的商品名称
    #          rewritten_query: str  # 改写后的问题
    #          history: list  # 历史对话记录
    #          answer : 可选的答案
        1. 获取历史条件记录（作为依据）
        2. 保存当前次的聊天记录
        3. 利用模型lm -> 1. 提取item_names  2.重写提问内容
        4. 进行item_name的向量数据库查询
        5. 对item_name结果进行打分分类处理 A 【确认集合】  B【可选集合】
        6. 处理确认和可选集合！ 有确认 =》 继续下个节点执行  || 有可选 or 没有item_names -> answer赋值结果
        7. 补充state状态 item_names rewritten_query  history
    """
    print(f"---node_item_name_confirm---开始处理")
    # 记录任务开始
    add_running_task(state["session_id"], sys._getframe().f_code.co_name,state["is_stream"])

    #  1. 获取历史条件记录（作为依据）
    history_chats = get_recent_messages(session_id=state["session_id"],limit=10)
    #  3. 利用模型lm -> 1. 提取item_names  2.重写提问内容
    #  参数： state["original_query"] || history_chats
    #  响应： { item_names : [华为 p60]  默认根据历史聊天记录给我们的！！ , rewritten_query : str }
    #  1. 为啥问题要重写？
    """
       1. 消除指代歧义    他 Ta 它 不明确！  明确查询主体 item_name
       2. 补全上下文     他的问题需要有历史记录支持！ 
       3. 去掉口语和冗余  同学们 同志们  为啥 咋弄 完犊子了 
       4. 润色问题增加召回率  模型查询的时候也会更精准
    """
    item_names_and_rewritten_query = step_3_llm_item_name_and_rewrite_query(state["original_query"],history_chats)
    item_names = item_names_and_rewritten_query.get("item_names",[])
    rewritten_query = item_names_and_rewritten_query.get("rewritten_query","")
    item_results = {}
    if len(item_names) > 0 :
        # 向量数据库查询 item_name
        # 4. milvus向量查询 item_names -》 模型提取 不一定跟我们向量数据库的完全相同(华为手机 P60)
        # 参数： item_names = [1,2,3,4]
        # 返回： 1 -> 向量数据库中item_names (向量查询) 2 -> 向量数据库中item_names (向量查询)
        #      [ { extracted:（模型提取的item_name）, matches:[{item_name:名字,score:0.8},{item_name:名字,score:0.8}]  }，
    #            { extracted:（模型提取的item_name）, matches:[{item_name:名字,score:0.8},{item_name:名字,score:0.8}]  }，
    #          ]
        query_milvus_results = step_4_query_milvus_item_names(item_names)
        # 5. 查询结果进行处理 区分 确定的item_name 以及可选的item_name  -》  没有对应的item_name
        # 参数： query_milvus_results
        # 返回： {确定item_name:[x,x,x,x,x] ,可选的item_name:[x,x,x,x,x]}
        # result = {
        #         "confirmed_item_names":list(set(confirmed_item_names)),
        #         "options_item_names":list(set(options_item_names))
        #     }
        item_results = step_5_confirmed_and_optional_item_name(query_milvus_results)

    #6. 根据item_name确定的集合进行用户反馈结果的处理 -》 answer赋值结果
    # 参数： item_results （两个集合） || 修改历史聊天记录对应item_names history_chats
    state = step_6_deal_list(state,item_results,history_chats,rewritten_query)
    #7. 记录本次的聊天对话 （answer回答）
    save_chat_message(
        session_id=state["session_id"],
        role="user",
        text=state["original_query"],
        rewritten_query=state.get("rewritten_query", ""),
        item_names=state.get("item_names", []),
        image_urls=[]
    )
    # 记录任务结束
    add_done_task(state["session_id"], sys._getframe().f_code.co_name,state["is_stream"])
    print(f"---node_item_name_confirm---处理结束")

    return state



if __name__ == "__main__":
    # 模拟输入状态
    mock_state = {
        "session_id": "test_session_001",
        "original_query": "华为B3-211H显示器好用么？？",
        "is_stream": False
    }

    print(">>> 开始测试 node_item_name_confirm...")
    try:
        # 运行节点
        result_state = node_item_name_confirm(mock_state)

        print("\n>>> 测试完成！最终状态:")
        print(json.dumps(result_state, indent=2, ensure_ascii=False,default=str))

        # 简单验证
        if result_state.get("item_names"):
            print(f"\n[PASS] 成功提取并确认商品名: {result_state['item_names']}")
        else:
            print(f"\n[WARN] 未确认到商品名 (可能是向量库无匹配或LLM未提取)")

    except Exception as e:
        logger.exception("==========")