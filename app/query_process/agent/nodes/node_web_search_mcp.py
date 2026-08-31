import asyncio
import os
import json
import sys
from itertools import count

from agents.mcp import MCPServerSse # pip install openai-agents
from agents.mcp import MCPServerStreamableHttp # pip install openai-agents
from app.core.logger import  logger

from app.conf.bailian_mcp_config import mcp_config
from app.utils.task_utils import add_running_task,add_done_task

DASHSCOPE_BASE_URL_STREAMABLE = mcp_config.mcp_base_url
DASHSCOPE_API_KEY = mcp_config.api_key

async def mcp_call_streamable(query):
    """
    调用百炼的网络搜索工具
    :param query:
    :return:
    """
    # 1. 创建MCPServerStreamableHttp对象
    search_mcp = MCPServerStreamableHttp(
        name="search_mcp",
        params={
            # 核心参数
            "url": DASHSCOPE_BASE_URL_STREAMABLE,
            "headers": {"Authorization": f"Bearer {DASHSCOPE_API_KEY}"},
            "timeout": 10,  # 连接超时时间
        },
        max_retry_attempts=3
    )
    # 2. 连接 - 调用 - 关闭
    try:
        await search_mcp.connect()
        # 获取工具
        tools = await search_mcp.list_tools()
        print(f"工具列表：{tools}")

        result = await search_mcp.call_tool(
            tool_name="bailian_web_search",
            arguments={
                "query": query,
                "count": 5,
            }
        )
        return result

    finally:
        await search_mcp.cleanup()






def node_web_search_mcp(state):
    """
    节点功能，调用外部搜索引擎补充信息
    :param state:
    :return:
    """
    add_running_task(state["session_id"], sys._getframe().f_code.co_name,state["is_stream"])
    print("---node-web-search-mcp处理---")

    query = state.get("rewritten_query")

    result = asyncio.run(mcp_call_streamable(query))

    # 3. 结果处理即可
    # {
    #   "isError": false,
    #   "content": [
    #     {
    #       "text": "{\"pages\":[{\"snippet\":\"和讯首页|手机和讯 登录注册 股票客户端 Android 股票客户端 iPhone\",\"hostname\":\"和讯网\",\"hostlogo\":\"https://img.alicdn.com/imgextra/i3/O1CN01VcUfI91cc0kCH3Gt2_!!6000000003620-73-tps-32-32.ico\",
    #                               \"title\":\"行情中心-和讯网 国内全面的即时行情数据服务中心\",
    #                               \"url\":\"https://quote.hexun.com/\"},
    #                            {\"snippet\":\"数据中心\",\"hostname\":\"东方财富网\",\"hostlogo\":\"https://img.alicdn.com/imgextra/i1/O1CN01iL4mYC1cF6vgiem0A_!!6000000003570-55-tps-32-32.svg\",\"title\":\"股票\",\"url\":\"https://stock.eastmoney.com/\"},{\"snippet\":\"意见反馈\",\"hostname\":\"东方财富网\",\"hostlogo\":\"https://quote.eastmoney.com/favicon.ico\",\"title\":\"行情中心:国内快捷全面的股票、基金、期货、美股、港股、外汇、黄金、债券行情系统_东方财富网\",\"url\":\"https://quote.eastmoney.com/center/qqzs.html#!/stealingyourhistory\"}],\"request_id\":\"faa40120-ee17-4401-a6c5-9970da077c05\",\"tools\":[],\"status\":0}",
    #       "type": "text"
    #     }
    #   ]
    # }
    # web_documents = []

    web_documents = json.loads(result.content[0].text).get("pages", [])

    logger.info(f"mcp搜索的结果为:{web_documents}")
    print("---node-web-search-mcp处理结束---")
    add_done_task(state["session_id"], sys._getframe().f_code.co_name, state["is_stream"])
    return {"web_search_docs": [web_documents]}

from dotenv import load_dotenv

if __name__ == '__main__':
    load_dotenv()
    test_state = {
        "session_id":"mcp_01",
        "rewritten_query": "现在的5060桌面端显卡多少钱？",
        "is_stream":True
    }

    # 调用 websearch_node 函数
    result_state = node_web_search_mcp(test_state)

    # 验证结果
    print("测试结果:")
    print(f"查询内容: {test_state.get('rewritten_query')}")

    # 输出搜索结果
    search_results = result_state.get('web_search_docs', [])
    print(f"搜索结果数量: {len(search_results)}")
    print("search_results", search_results)