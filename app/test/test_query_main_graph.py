import json

from app.query_process.agent.main_graph import query_app
from app.query_process.agent.state import create_query_default_state
from app.core.logger import logger

logger.info("===== 开始测试 =====")

initial_state = create_query_default_state(session_id="test_001",
        original_query="华为P60怎么样?")
final_state = None

# 只输出更最终的状态值（字典形式），不包含节点名称、执行日志、元数据等额外信息
for event in query_app.stream(initial_state):
    for key, value in event.items():
        logger.info(f"节点: {key} , 输出结果：{value}")
        final_state = value

# 格式化输出最终状态
logger.info(f"最终状态: {json.dumps(final_state, indent=4, ensure_ascii=False)}")

logger.info("图结构:")
# uv add grandalf
query_app.get_graph().print_ascii()

logger.info("===== 测试结束 =====")