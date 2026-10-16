from typing import TypedDict
import copy
from app.core.logger import logger

class ImportGraphState(TypedDict):
    """
    图的状态定义，包含所有节点产生和消费的数据字段。
    TypedDict 让我们在代码中能有自动补全和类型检查。
    使用字典式访问（如state["session_id"]、state.get("embedding_chunks")）
    """
    task_id: str          # 任务唯一ID，用于追踪日志

    # --- 流程控制标记 ---
    is_md_read_enabled: bool   # 是否启用 Markdown 读取路径
    is_pdf_read_enabled: bool  # 是否启用 PDF 读取路径


    # --- 切块相关 ---
    is_normal_split_enabled: bool
    is_silicon_flow_api_enabled: bool
    is_advanced_split_enabled: bool
    is_vllm_enabled: bool

    # --- 路径相关 ---
    local_dir: str        # 当前工作目录或输出目录
    local_file_path: str  # 原始输入文件路径
    file_title: str       # 文件标题（文件名去后缀）
    pdf_path: str         # PDF 文件路径 (如果输入是PDF)
    md_path: str          # Markdown 文件路径 (转换后或直接输入的)
    split_path: str       # 分块后的文件路径
    embeddings_path: str  # 向量数据库文件路径

    # --- 内容数据 ---
    md_content: str       # Markdown 的全文内容
    chunks: list          # 切片后的文本列表，包含 metadata
    item_name: str        # 识别出的主体名称 (如: "万用表")，用于增强检索

    # --- 版本控制 ---
    doc_id: str           # 逻辑文档稳定标识（跨版本不变）
    version: int          # 版本序号（从 1 递增）
    content_hash: str     # md_content 的 MD5，用于识别重复上传
    version_note: str     # 版本说明（上传表单填写）
    is_duplicate: bool    # 是否为"内容未变"的重复上传
    is_latest: bool       # 是否为文档最新版本
    archived_chunks_path: str  # 历史版本切片归档路径（可选）

    # --- 时效性 ---
    publish_date: int     # 生效/发布日期（epoch 秒）
    expire_date: int      # 失效日期（epoch 秒，0 表示长期有效）
    date_source: str      # 日期来源 form | regex | llm | fallback
    timeliness_status: str  # fresh/aging/stale/expired
    source_dept: str      # 文档来源部门（上传表单原样填写，留空为 ""）

    # --- 数据库相关 ---
    embeddings_content: list # 包含向量数据的列表，准备写入 Milvus


# 建议定一个初始化对象，方便后续使用
# 定义图状态的默认初始值
graph_default_state: ImportGraphState = {
    "task_id":"",
    "is_pdf_read_enabled": False,
    "is_md_read_enabled": False,
    "is_normal_split_enabled": True,
    "is_silicon_flow_api_enabled": True,
    "is_advanced_split_enabled": False,
    "is_vllm_enabled": False,
    "local_dir": "",
    "local_file_path": "",
    "pdf_path": "",
    "md_path": "",
    "file_title": "",
    "split_path": "",
    "embeddings_path": "",
    "md_content": "",
    "chunks": [],
    "item_name": "",
    # 版本控制
    "doc_id": "",
    "version": 0,
    "content_hash": "",
    "version_note": "",
    "is_duplicate": False,
    "is_latest": True,
    "archived_chunks_path": "",
    # 时效性
    "publish_date": 0,
    "expire_date": 0,
    "date_source": "fallback",
    "timeliness_status": "fresh",
    "source_dept": "",
    "embeddings_content": []
}

def create_default_state(**overrides) -> ImportGraphState:
    """
    创建默认状态，支持覆盖

    Args:
        **overrides: 要覆盖的字段（关键字参数解包）

    Returns:
        新的状态实例

    Examples:
        state = create_default_state(task_id="task_001", local_file_path="doc.pdf")
    """

    # 默认状态
    state = copy.deepcopy(graph_default_state)
    # 用 overrides 覆盖默认值
    state.update(overrides)
    # 返回创建好的状态字典实例
    return state

def get_default_state() -> ImportGraphState:
    """
    返回一个新的状态实例，避免全局变量污染
    """
    return copy.deepcopy(graph_default_state)


if __name__ == "__main__":
    """
    测试
    """
    # 创建默认状态
    state = create_default_state(local_file_path="万用表RS-12的使用.pdf",pdf_path="倪毅涛洗澡.pdf")
    logger.info(state)